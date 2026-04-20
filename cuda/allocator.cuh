// allocator.cuh — DevicePool bump allocator + binned heap arena allocator.
//
// DevicePool: bump allocator (base/offset/capacity) with two embedded DeviceHeap
//   allocators (heap = output, scratch = scratch).
//   DevicePool→DeviceHeap is a direct field (not a pointer).
//   DeviceHeap→DevicePool is a back-pointer (set by heap_init_kernel).
//
// DeviceHeap: 64 independent arenas, each with:
//   - 64 sub-bin doubly-linked free lists (32 pow-of-2 bins × 2 linear halves)
//   - 64-bit occupancy bitmap for O(1) bin selection
//   - spin-lock for concurrent access
//
// Sub-bin layout (64 sub-bins indexed 0..63):
//   Bin b covers [4K·2^b, 8K·2^b). Two linear halves:
//     sub-bin 2b   = [4K·2^b, 6K·2^b)  lower half
//     sub-bin 2b+1 = [6K·2^b, 8K·2^b)  upper half
//
// Block layout: [HeapBlockHdr (16B)] [user data (data_size B)] [HeapBlockFtr (16B)]
//   Free block data: data[0..7]=prev ptr, data[8..15]=next ptr (HeapBlockHdr addrs, 0=end)
//
// Slab layout (each new pool slab of S bytes):
//   [leading sentinel 32B] [free block (S-96 B data)] [trailing sentinel 32B]
//   Sentinels: data_size=0, is_free=0 — prevents coalescing across slab boundaries.
//
// Thread rules:
//   pool_alloc        — thread 0 only; broadcast via __shared__ + __syncthreads__
//   global_alloc_warp — lane 0 only; broadcast via __shfl_sync
//   heap_alloc        — thread 0 only
//   heap_free         — thread 0 only
//
// Host init: cuMemAlloc sizeof(DevicePool) bytes, set base/offset/capacity, launch
//   heap_init_kernel<<<128,32>>> (in mm.cu) which sets the pool back-pointers and
//   zeroes all arena state in both embedded heaps.
//
// Must stay in sync with csrc/structs.h.
#pragma once
#include "common.cuh"
#include "error_codes.cuh"

// ============================================================================
// Configuration
// ============================================================================

#ifndef HEAP_NUM_ARENAS
#define HEAP_NUM_ARENAS     64
#endif
#define HEAP_NUM_SUBBINS    64
#define HEAP_HDR_SIZE       16      // HeapBlockHdr bytes
#define HEAP_FTR_SIZE       16      // HeapBlockFtr bytes (same layout)
#define HEAP_ALIGN          512     // 512 B minimum user-data alignment
#define HEAP_MIN_POOL_ALLOC 131072  // 128 KB minimum new slab from pool
// Slab overhead: leading sentinel (32B) + block hdr+ftr (32B) + trailing sentinel (32B)
#define HEAP_SLAB_OVERHEAD  96

// Success code (error codes in error_codes.cuh: KERR_HEAP_*)
#define HEAP_OK      0

// ============================================================================
// Structures (must match csrc/structs.h)
// Forward declaration of DevicePool required because DeviceHeap references it.
// ============================================================================

struct DevicePool;  // forward declaration

// 16-byte block header — HeapBlockFtr has identical layout.
struct HeapBlockHdr {
    unsigned int   data_size;  // user data bytes; 0 for sentinel
    unsigned short arena_idx;  // owning arena 0..63
    unsigned char  is_free;    // 1=free, 0=allocated or sentinel
    unsigned char  _reserved;
    unsigned int   _pad;
};
typedef HeapBlockHdr HeapBlockFtr;

// Per-arena state: bitmap + spin-lock + 64 sub-bin doubly-linked free lists.
// sizeof(HeapArena) == 8 + 8 + 512 + 512 == 1040 bytes.
struct HeapArena {
    unsigned long long bitmap;                   // bit i set <=> heads[i] != 0
    int                lock;                     // spin-lock: 0=unlocked
    int                _pad;
    unsigned long long heads[HEAP_NUM_SUBBINS];  // free-list head addresses
    unsigned long long tails[HEAP_NUM_SUBBINS];  // free-list tail addresses
};

// Heap allocator: 64 arenas + back-pointer to parent DevicePool.
struct DeviceHeap {
    DevicePool* pool;                        // back-pointer set by heap_init_kernel
    unsigned long long outstanding_bytes;    // live (alloc'd not freed) block user-bytes
    unsigned long long alloc_count;          // total heap_alloc calls (cumulative)
    unsigned long long free_count;           // total heap_free calls (cumulative)
    HeapArena   arenas[HEAP_NUM_ARENAS];     // 64 * 1040 = 66560 bytes
};

// DevicePool: bump allocator + two embedded heap allocators.
// sizeof(DevicePool) == 24 + 2*sizeof(DeviceHeap).
struct DevicePool {
    char*               base;      // bump alloc base (device ptr)
    unsigned long long* offset;    // bump alloc offset counter (device ptr to ull)
    unsigned long long  capacity;  // pool capacity in bytes
    DeviceHeap          heap;      // output heap  (heap.pool = this)
    DeviceHeap          scratch;   // scratch heap (scratch.pool = this)
};

// ============================================================================
// Bump allocators
// ============================================================================

// Thread-0 / block-scope allocation.
// CRITICAL: call from thread 0 only; broadcast result via __shared__ + __syncthreads__.
__device__ inline void* pool_alloc(DevicePool* pool, unsigned int size) {
    unsigned long long aligned = (unsigned long long)((size + 15) & ~15);
    unsigned long long old = atomicAdd(pool->offset, aligned);
    if (old + aligned > pool->capacity) return NULL;
    return pool->base + old;
}

// ============================================================================
// Heap internal helpers
// ============================================================================

__device__ inline void arena_lock(HeapArena* a) {
    while (atomicCAS(&a->lock, 0, 1)) { /* spin */ }
    __threadfence();
}
__device__ inline void arena_unlock(HeapArena* a) {
    __threadfence();
    atomicExch(&a->lock, 0);
}

// Doubly-linked list prev/next pointers live in the first 16 bytes of block data.
// blk = address of HeapBlockHdr as an unsigned long long (for pointer arithmetic).
#define HEAP_BLK_PREV(blk) (*(unsigned long long*)((blk) + HEAP_HDR_SIZE))
#define HEAP_BLK_NEXT(blk) (*(unsigned long long*)((blk) + HEAP_HDR_SIZE + 8))

// Sub-bin index for a block of given data_size.
// Uses 512-byte units: bin b covers [512·2^b, 1024·2^b).
// Sub-bin 2b = lower half [512·2^b, 768·2^b), sub-bin 2b+1 = upper half.
__device__ inline int heap_subbin_for_size(unsigned int sz) {
    unsigned int units = sz >> 9;   // 512-byte units
    if (units == 0) return 0;
    int b = 31 - __clz(units);
    unsigned int lo  = 1u << b;
    unsigned int mid = lo + (lo >> 1);  // 1.5 * lo in 512-byte units
    int sub = 2 * b + (units >= mid ? 1 : 0);
    return (sub < HEAP_NUM_SUBBINS - 1) ? sub : HEAP_NUM_SUBBINS - 1;
}

// Minimum sub-bin whose lower_bound >= sz (any block in it satisfies the request).
__device__ inline int heap_min_subbin_for_alloc(unsigned int sz) {
    if (sz <= (unsigned int)HEAP_ALIGN) return 0;
    unsigned int units = sz >> 9;   // 512-byte units
    if (units == 0) return 0;
    int b = 31 - __clz(units);
    unsigned long long lb_2b  = (unsigned long long)512 << b;  // lower_bound(sub-bin 2b)
    unsigned long long lb_2b1 = (unsigned long long)768 << b;  // lower_bound(sub-bin 2b+1)
    int s;
    if      (lb_2b  >= sz) s = 2 * b;
    else if (lb_2b1 >= sz) s = 2 * b + 1;
    else                   s = 2 * (b + 1);
    return (s < HEAP_NUM_SUBBINS) ? s : HEAP_NUM_SUBBINS - 1;
}

__device__ inline unsigned int heap_next_pow2_u32(unsigned int v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; return v + 1;
}

// Write header and matching footer for a block at hdr_addr.
// Footer placed at hdr_addr + HEAP_HDR_SIZE + data_size.
__device__ inline void heap_write_block(unsigned long long hdr_addr,
                                        unsigned int       data_size,
                                        unsigned short     arena_idx,
                                        unsigned char      is_free) {
    HeapBlockHdr* h = (HeapBlockHdr*)hdr_addr;
    h->data_size = data_size; h->arena_idx = arena_idx;
    h->is_free   = is_free;   h->_reserved = 0; h->_pad = 0;
    HeapBlockFtr* f = (HeapBlockFtr*)(hdr_addr + HEAP_HDR_SIZE + data_size);
    f->data_size = data_size; f->arena_idx = arena_idx;
    f->is_free   = is_free;   f->_reserved = 0; f->_pad = 0;
}

// Remove blk from sub-bin s of arena (O(1) doubly-linked).
__device__ inline void subbin_remove(HeapArena* arena, int s,
                                     unsigned long long blk) {
    unsigned long long prev = HEAP_BLK_PREV(blk);
    unsigned long long next = HEAP_BLK_NEXT(blk);
    if (prev) HEAP_BLK_NEXT(prev) = next; else arena->heads[s] = next;
    if (next) HEAP_BLK_PREV(next) = prev; else arena->tails[s] = prev;
    if (arena->heads[s] == 0) arena->bitmap &= ~(1ULL << s);
}

// Push blk to head of sub-bin s of arena.
__device__ inline void subbin_push_head(HeapArena* arena, int s,
                                        unsigned long long blk) {
    unsigned long long old = arena->heads[s];
    HEAP_BLK_PREV(blk) = 0;
    HEAP_BLK_NEXT(blk) = old;
    if (old) HEAP_BLK_PREV(old) = blk; else arena->tails[s] = blk;
    arena->heads[s] = blk;
    arena->bitmap  |= (1ULL << s);
}

// ============================================================================
// heap_alloc — thread 0 only
// ============================================================================
//
// Allocate req_size bytes from heap h. Uses h->pool for new slabs.
// Arena = blockIdx.x % HEAP_NUM_ARENAS.
// On success: sets *out and returns HEAP_OK.  On failure: KERR_HEAP_OOM.

__device__ inline int heap_alloc(DeviceHeap* h, unsigned int req_size, void** out) {
    unsigned int aligned = (req_size + HEAP_ALIGN - 1) & ~(unsigned int)(HEAP_ALIGN - 1);
    if (!aligned) aligned = HEAP_ALIGN;

    int        aidx  = (int)(blockIdx.x % (unsigned int)HEAP_NUM_ARENAS);
    HeapArena* arena = &h->arenas[aidx];

    arena_lock(arena);

    // --- Search free lists via bitmap ---
    int s_start = heap_min_subbin_for_alloc(aligned);
    unsigned long long avail = arena->bitmap & (~0ULL << s_start);
    unsigned long long blk   = 0;
    if (avail) {
        int s = __ffsll((long long)avail) - 1;
        blk = arena->heads[s];
        subbin_remove(arena, s, blk);
    }

    // --- Allocate new slab from pool if no free block found ---
    if (!blk) {
        DevicePool* pool = h->pool;
        // Allocate 2x the needed data so that after the user frees the allocated
        // portion and it coalesces with the remainder, the merged free block is
        // always in a strictly higher sub-bin than heap_min_subbin_for_alloc(aligned)
        // requires — guaranteeing it will be found on the next same-size request.
        unsigned int need_total = aligned * 2 + HEAP_SLAB_OVERHEAD;
        unsigned int slab_total = (need_total > HEAP_MIN_POOL_ALLOC)
                                ? heap_next_pow2_u32(need_total)
                                : (unsigned int)HEAP_MIN_POOL_ALLOC;
        unsigned int slab_data = slab_total - HEAP_SLAB_OVERHEAD;

        unsigned long long off = atomicAdd(pool->offset, (unsigned long long)slab_total);
        if (off + slab_total > pool->capacity) {
            arena_unlock(arena);
            return KERR_HEAP_OOM;
        }

        unsigned long long base = (unsigned long long)pool->base + off;

        // Leading sentinel (hdr+ftr, data_size=0, is_free=0).
        heap_write_block(base, 0, (unsigned short)aidx, 0);
        // One large free block covering the interior.
        blk = base + HEAP_HDR_SIZE + HEAP_FTR_SIZE;  // base + 32
        heap_write_block(blk, slab_data, (unsigned short)aidx, 1);
        // Trailing sentinel.
        unsigned long long trail = base + slab_total - HEAP_HDR_SIZE - HEAP_FTR_SIZE;
        heap_write_block(trail, 0, (unsigned short)aidx, 0);
        // blk is our candidate (slab_data >= aligned, split handles it below).
    }

    // --- Optionally split remainder ---
    unsigned int blk_sz = ((HeapBlockHdr*)blk)->data_size;
    unsigned int rem    = blk_sz - aligned;
    // Need rem to form a full block: Hdr + data(>=HEAP_ALIGN) + Ftr.
    if (rem >= (unsigned int)(HEAP_HDR_SIZE + HEAP_ALIGN + HEAP_FTR_SIZE)) {
        // Write footer for allocated portion.
        unsigned long long alloc_ftr_addr = blk + HEAP_HDR_SIZE + aligned;
        HeapBlockFtr* af = (HeapBlockFtr*)alloc_ftr_addr;
        af->data_size = aligned; af->arena_idx = (unsigned short)aidx;
        af->is_free = 0; af->_reserved = 0; af->_pad = 0;

        // Create remainder free block right after allocated footer.
        unsigned long long rem_hdr  = alloc_ftr_addr + HEAP_FTR_SIZE;
        unsigned int       rem_data = rem - HEAP_HDR_SIZE - HEAP_FTR_SIZE;
        heap_write_block(rem_hdr, rem_data, (unsigned short)aidx, 1);

        // Update allocated block header.
        HeapBlockHdr* bh = (HeapBlockHdr*)blk;
        bh->data_size = aligned; bh->is_free = 0;

        // Push remainder into its sub-bin.
        subbin_push_head(arena, heap_subbin_for_size(rem_data), rem_hdr);
    } else {
        // No split: mark full block allocated (keeps blk_sz as data_size).
        HeapBlockHdr* bh = (HeapBlockHdr*)blk;
        bh->is_free = 0;
        HeapBlockFtr* bf = (HeapBlockFtr*)(blk + HEAP_HDR_SIZE + blk_sz);
        bf->is_free = 0;
    }

    // Track live user-bytes using the final data_size of the block (matches what heap_free reads).
    unsigned int alloc_ds = ((HeapBlockHdr*)blk)->data_size;
    arena_unlock(arena);
    atomicAdd(&h->outstanding_bytes, (unsigned long long)alloc_ds);
    atomicAdd(&h->alloc_count, 1ULL);
    *out = (void*)(blk + HEAP_HDR_SIZE);
    return HEAP_OK;
}

// ============================================================================
// heap_free — thread 0 only
// ============================================================================
//
// Free ptr back to heap h. Coalesces with immediately adjacent free blocks.
// Uses the block's stored arena_idx as its home arena. All blocks within a
// pool slab share the same arena_idx, so coalescing is always intra-arena.

__device__ inline int heap_free(DeviceHeap* h, void* ptr) {
    if (!ptr) return HEAP_OK;
    unsigned long long blk = (unsigned long long)ptr - HEAP_HDR_SIZE;
    unsigned int ds = ((HeapBlockHdr*)blk)->data_size;
    int        aidx  = (int)((HeapBlockHdr*)blk)->arena_idx;
    atomicAdd(&h->outstanding_bytes, (unsigned long long)(0ULL - (unsigned long long)ds));
    atomicAdd(&h->free_count, 1ULL);

    if (aidx < 0 || aidx >= HEAP_NUM_ARENAS) {
        return KERR_HEAP_CORRUPT;
    }

    HeapArena* arena = &h->arenas[aidx];

    arena_lock(arena);

    // --- Coalesce with previous block (inspect its footer) ---
    {
        HeapBlockFtr* prev_ftr = (HeapBlockFtr*)(blk - HEAP_FTR_SIZE);
        if (prev_ftr->is_free) {
            unsigned int       prev_ds  = prev_ftr->data_size;
            unsigned long long prev_blk = blk - HEAP_FTR_SIZE - prev_ds - HEAP_HDR_SIZE;
            int prev_aidx = (int)((HeapBlockHdr*)prev_blk)->arena_idx;
            if (prev_aidx < 0 || prev_aidx >= HEAP_NUM_ARENAS) {
                arena_unlock(arena);
                return KERR_HEAP_CORRUPT_PREV;
            }
            subbin_remove(arena, heap_subbin_for_size(prev_ds), prev_blk);
            ds  = prev_ds + HEAP_FTR_SIZE + HEAP_HDR_SIZE + ds;
            blk = prev_blk;
        }
    }

    // --- Coalesce with next block (inspect its header) ---
    {
        unsigned long long next_blk = blk + HEAP_HDR_SIZE + ds + HEAP_FTR_SIZE;
        HeapBlockHdr* next_hdr = (HeapBlockHdr*)next_blk;
        if (next_hdr->is_free) {
            int next_aidx = (int)next_hdr->arena_idx;
            unsigned int next_ds = next_hdr->data_size;
            if (next_aidx < 0 || next_aidx >= HEAP_NUM_ARENAS) {
                // skip next coalesce — block is corrupt
            } else {
                subbin_remove(arena, heap_subbin_for_size(next_ds), next_blk);
                ds = ds + HEAP_FTR_SIZE + HEAP_HDR_SIZE + next_ds;
            }
        }
    }

    // --- Write merged block and insert into free list ---
    heap_write_block(blk, ds, (unsigned short)aidx, 1);
    subbin_push_head(arena, heap_subbin_for_size(ds), blk);
    arena_unlock(arena);
    return HEAP_OK;
}
