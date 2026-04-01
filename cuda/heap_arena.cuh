// heap_arena.cuh — Large-object heap arena allocator.
//
// DeviceHeap: 64 independent arenas, each with a first-fit free list.
// Allocations are 4K-aligned. New memory comes from a backing DevicePool.
//
// Block layout: [HeapBlockHdr (16 B)][data (data_size B)]
//   Free blocks: data[0..7] = next free header addr as unsigned long long, 0=end.
//
// Thread rules (same convention as pool_alloc):
//   heap_alloc   — thread 0 only; broadcast via __shared__ + __syncthreads__
//   heap_free    — thread 0 only
//   heap_compact — entire block; NO concurrent heap ops on other blocks allowed
//
// Host init: allocate a DevicePool and a separate compact_buf of
// HEAP_COMPACT_BUF_BYTES bytes.  Zero-init DeviceHeap (zeroed arenas =
// empty free lists and unlocked spin-locks).
#pragma once
#include "warp_sort.cuh"   // warp_sort_t; transitively includes common.cuh

// ============================================================================
// Configuration
// ============================================================================

#define HEAP_NUM_ARENAS        64
#define HEAP_HDR_SIZE          16           // header bytes (16-aligned)
#define HEAP_ALIGN             4096         // minimum allocation alignment (4 KB)
#define HEAP_MIN_POOL_ALLOC    131072       // minimum new slab from pool (128 KB)
#define HEAP_COMPACT_BUF_BYTES (16 << 20)  // compact buffer total size (16 MB)

// Max free-block addresses storable in compact_buf, leaving room for the
// warp_sort scratch (tmp = n*8 + WS_MAX_STACK*2*4 B; data+tmp <= BUF_BYTES).
#define HEAP_COMPACT_CAP \
    ((HEAP_COMPACT_BUF_BYTES - WS_MAX_STACK * 2 * (int)sizeof(int)) \
     / (2 * (int)sizeof(unsigned long long)))

// ============================================================================
// Error codes
// ============================================================================

#define HEAP_OK          0
#define HEAP_ERR_OOM     1   // pool exhausted
#define HEAP_ERR_COMPACT 2   // compact buffer overflow or warp_sort stack overflow

// ============================================================================
// Data structures
// ============================================================================

// Block header — 16 bytes, immediately before the data region.
// When the block is on a free list, data[0..7] stores the next header address.
struct HeapBlockHdr {
    unsigned int data_size;  // size of the data region (4K-aligned, header excluded)
    unsigned int _pad[3];
};

// Per-arena free list with a spin-lock.
struct HeapArena {
    unsigned long long head;  // header addr of first free block (0 = empty)
    int                lock;  // spin-lock: 0 = unlocked
    int                _pad;
};

// Heap allocator handle — lives in device-accessible memory.
struct DeviceHeap {
    DevicePool*         pool;                    // backing bump allocator
    HeapArena           arenas[HEAP_NUM_ARENAS]; // per-arena free lists
    unsigned long long* compact_buf;             // HEAP_COMPACT_BUF_BYTES of device memory
};

// ============================================================================
// Internal helpers
// ============================================================================

__device__ inline void arena_lock(HeapArena* a) {
    while (atomicCAS(&a->lock, 0, 1)) { /* spin */ }
    __threadfence();
}
__device__ inline void arena_unlock(HeapArena* a) {
    __threadfence();
    atomicExch(&a->lock, 0);
}

__device__ inline unsigned long long heap_get_next(unsigned long long hdr) {
    return *(unsigned long long*)(hdr + HEAP_HDR_SIZE);
}
__device__ inline void heap_set_next(unsigned long long hdr, unsigned long long next) {
    *(unsigned long long*)(hdr + HEAP_HDR_SIZE) = next;
}

__device__ inline unsigned int heap_next_pow2(unsigned int v) {
    v--; v |= v>>1; v |= v>>2; v |= v>>4; v |= v>>8; v |= v>>16; return v + 1;
}

// MurmurHash3 finalizer — maps a block address to an arena index.
// Provides uniform distribution even when blocks are all large (sparse page indices).
__device__ inline int heap_arena_for_addr(unsigned long long addr) {
    unsigned long long h = addr;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)(h % (unsigned long long)HEAP_NUM_ARENAS);
}

// ============================================================================
// heap_alloc — thread 0 only
// ============================================================================

// Allocate req_size bytes. Arena chosen by blockIdx.x % HEAP_NUM_ARENAS.
// On success: sets *out and returns HEAP_OK.
// On failure: returns HEAP_ERR_OOM (*out unchanged).
__device__ int heap_alloc(DeviceHeap* heap, unsigned int req_size, void** out) {
    unsigned int aligned = (req_size + HEAP_ALIGN - 1) & ~(unsigned int)(HEAP_ALIGN - 1);
    if (!aligned) aligned = HEAP_ALIGN;

    int        aidx  = (int)(blockIdx.x % (unsigned int)HEAP_NUM_ARENAS);
    HeapArena* arena = &heap->arenas[aidx];

    // --- First-fit search in free list ---
    arena_lock(arena);
    {
        unsigned long long* prev = &arena->head;
        unsigned long long  cur  = arena->head;
        while (cur) {
            HeapBlockHdr*      hdr = (HeapBlockHdr*)cur;
            unsigned long long nxt = heap_get_next(cur);
            if (hdr->data_size >= aligned) {
                *prev = nxt;  // unlink from free list
                // Split remainder if it has room for a new header + >= 4K data
                unsigned int rem = hdr->data_size - aligned;
                if (rem >= (unsigned int)(HEAP_HDR_SIZE + HEAP_ALIGN)) {
                    unsigned long long ra = cur + HEAP_HDR_SIZE + aligned;
                    ((HeapBlockHdr*)ra)->data_size = rem - HEAP_HDR_SIZE;
                    heap_set_next(ra, arena->head);
                    arena->head = ra;
                }
                hdr->data_size = aligned;
                arena_unlock(arena);
                *out = (void*)(cur + HEAP_HDR_SIZE);
                return HEAP_OK;
            }
            prev = (unsigned long long*)(cur + HEAP_HDR_SIZE);
            cur  = nxt;
        }
    }
    arena_unlock(arena);

    // --- Allocate a new slab from the backing pool ---
    unsigned int slab  = (aligned > HEAP_MIN_POOL_ALLOC)
                        ? heap_next_pow2(aligned) : HEAP_MIN_POOL_ALLOC;
    unsigned long long total = (unsigned long long)HEAP_HDR_SIZE + slab;
    unsigned long long off   = atomicAdd(heap->pool->offset, total);
    if (off + total > heap->pool->capacity) return HEAP_ERR_OOM;

    unsigned long long block = (unsigned long long)(heap->pool->base + off);
    ((HeapBlockHdr*)block)->data_size = slab;

    // Split remainder into the same arena's free list
    unsigned int rem = slab - aligned;
    if (rem >= (unsigned int)(HEAP_HDR_SIZE + HEAP_ALIGN)) {
        unsigned long long ra = block + HEAP_HDR_SIZE + aligned;
        ((HeapBlockHdr*)ra)->data_size = rem - HEAP_HDR_SIZE;
        arena_lock(arena);
        heap_set_next(ra, arena->head);
        arena->head = ra;
        arena_unlock(arena);
    }

    ((HeapBlockHdr*)block)->data_size = aligned;
    *out = (void*)(block + HEAP_HDR_SIZE);
    return HEAP_OK;
}

// ============================================================================
// heap_free — thread 0 only
// ============================================================================

// Return ptr to the free list of arena blockIdx.x % HEAP_NUM_ARENAS.
__device__ int heap_free(DeviceHeap* heap, void* ptr) {
    if (!ptr) return HEAP_OK;
    unsigned long long hdr   = (unsigned long long)ptr - HEAP_HDR_SIZE;
    int                aidx  = (int)(blockIdx.x % (unsigned int)HEAP_NUM_ARENAS);
    HeapArena*         arena = &heap->arenas[aidx];
    arena_lock(arena);
    heap_set_next(hdr, arena->head);
    arena->head = hdr;
    arena_unlock(arena);
    return HEAP_OK;
}

// ============================================================================
// Comparator for address-sorted warp_sort_t
// ============================================================================

struct HeapAddrCmp {
    static __device__ inline int cmp(unsigned long long a, unsigned long long b) {
        return (a < b) ? -1 : (a > b) ? 1 : 0;
    }
    static __device__ inline unsigned long long sentinel() { return ~0ULL; }
};

// ============================================================================
// heap_compact — one block; caller must ensure no concurrent heap operations
// ============================================================================

// Collect all free blocks from all arenas, sort by address, coalesce
// physically adjacent blocks, and re-insert into arenas.
// compact_buf (HEAP_COMPACT_BUF_BYTES) provides scratch for collection + sort.
// Returns HEAP_OK or HEAP_ERR_COMPACT (buffer overflow / sort stack overflow).
__device__ int heap_compact(DeviceHeap* heap) {
    int tid  = threadIdx.x;
    int ntid = blockDim.x;
    int lane = tid & 31;

    __shared__ unsigned int s_n;
    __shared__ int          s_err;
    if (tid == 0) { s_n = 0; s_err = HEAP_OK; }
    __syncthreads();

    // Phase 1: drain all arenas into compact_buf (parallel across threads).
    // No locks needed: caller guarantees no concurrent heap ops.
    for (int ai = tid; ai < HEAP_NUM_ARENAS; ai += ntid) {
        HeapArena*         arena = &heap->arenas[ai];
        unsigned long long cur   = arena->head;
        arena->head = 0;
        while (cur) {
            unsigned long long nxt = heap_get_next(cur);
            unsigned int idx = atomicAdd(&s_n, 1u);
            if (idx < (unsigned int)HEAP_COMPACT_CAP) {
                heap->compact_buf[idx] = cur;
            } else {
                // Buffer overflow: put block back uncompacted into same arena.
                heap_set_next(cur, arena->head);
                arena->head = cur;
                s_err = HEAP_ERR_COMPACT;  // benign race: same value
            }
            cur = nxt;
        }
    }
    __syncthreads();

    unsigned int n = (s_n < (unsigned int)HEAP_COMPACT_CAP)
                   ? s_n : (unsigned int)HEAP_COMPACT_CAP;
    if (n == 0) return s_err;

    // Phase 2: warp 0 sorts compact_buf[0..n) by address ascending.
    // Scratch occupies compact_buf[HEAP_COMPACT_CAP..] — see size derivation above.
    if (tid < 32) {
        char* scratch = (char*)(heap->compact_buf + HEAP_COMPACT_CAP);
        int rc = warp_sort_t<unsigned long long, HeapAddrCmp>(
            heap->compact_buf, scratch, (int)n, lane);
        if (rc && lane == 0) s_err = HEAP_ERR_COMPACT;
    }
    __syncthreads();

    // Phase 3+4 (tid 0): coalesce adjacent blocks and re-insert into arenas.
    // Sequential to avoid intra-block lock contention.
    if (tid == 0) {
        unsigned int i = 0;
        while (i < n) {
            unsigned long long base = heap->compact_buf[i];
            unsigned int       ds   = ((HeapBlockHdr*)base)->data_size;
            unsigned int       j    = i + 1;
            // Extend: merge while the next block is physically adjacent.
            while (j < n) {
                unsigned long long nxt = heap->compact_buf[j];
                if (base + HEAP_HDR_SIZE + ds == nxt) {
                    ds += HEAP_HDR_SIZE + ((HeapBlockHdr*)nxt)->data_size;
                    j++;
                } else break;
            }
            ((HeapBlockHdr*)base)->data_size = ds;
            int aidx = heap_arena_for_addr(base);
            heap_set_next(base, heap->arenas[aidx].head);
            heap->arenas[aidx].head = base;
            i = j;
        }
    }
    __syncthreads();

    return s_err;
}
