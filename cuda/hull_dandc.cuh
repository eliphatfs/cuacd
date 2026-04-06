// hull_dandc.cuh — Warp-based D&C convex hull volume and mesh extraction.
//
// Faithful port of btConvexHullComputer by Ole Kniemeyer (MAXON, zlib license).
// Parallel tree merge: points split into 16 groups (BT_HULL_GROUPS), each
// primary lane builds a small hull independently, then 4 rounds of pairwise
// merging (8×2 → 4×2 → 2×2 → 1×2). Within each merge, paired lanes run
// bt_findMaxAngle in parallel: lane 2g handles c0/h0, lane 2g+1 handles c1/h1.
// Uses int32 coordinates with exact int64/int128 predicates.
//
// All 32 threads must call hull_dandc_warp_mesh with identical arguments.
// Returns hull volume (>= 0) or -1.0f on error.
//
// Requires: warp_common.cuh, allocator.cuh, structs.cuh
#pragma once
#include "warp_common.cuh"
#include "warp_sort.cuh"
#include "allocator.cuh"
#include "structs.cuh"

// ============================================================================
// Exact arithmetic types (ported from Bullet)
// ============================================================================

struct BtInt128 {
    unsigned long long low;
    unsigned long long high;
};

__device__ inline BtInt128 bt128_make(unsigned long long lo, unsigned long long hi) {
    BtInt128 r; r.low = lo; r.high = hi; return r;
}
__device__ inline BtInt128 bt128_from_i64(long long v) {
    BtInt128 r; r.low = (unsigned long long)v; r.high = (v >= 0) ? 0ULL : ~0ULL; return r;
}
__device__ inline BtInt128 bt128_from_u64(unsigned long long v) {
    BtInt128 r; r.low = v; r.high = 0; return r;
}
__device__ inline BtInt128 bt128_neg(BtInt128 a) {
    BtInt128 r;
    r.low = (unsigned long long)(-(long long)a.low);
    r.high = ~a.high + (a.low == 0);
    return r;
}
__device__ inline BtInt128 bt128_add(BtInt128 a, BtInt128 b) {
    unsigned long long lo = a.low + b.low;
    BtInt128 r; r.low = lo; r.high = a.high + b.high + (lo < a.low);
    return r;
}
__device__ inline BtInt128 bt128_sub(BtInt128 a, BtInt128 b) {
    return bt128_add(a, bt128_neg(b));
}
__device__ inline int bt128_sign(BtInt128 a) {
    return ((long long)a.high < 0) ? -1 : (a.high || a.low) ? 1 : 0;
}
__device__ inline int bt128_ucmp(BtInt128 a, BtInt128 b) {
    if (a.high < b.high) return -1;
    if (a.high > b.high) return 1;
    if (a.low < b.low) return -1;
    if (a.low > b.low) return 1;
    return 0;
}
__device__ inline bool bt128_lt(BtInt128 a, BtInt128 b) {
    return (a.high < b.high) || ((a.high == b.high) && (a.low < b.low));
}

// Unsigned 64x64 -> 128 multiply
__device__ inline BtInt128 bt128_umul(unsigned long long a, unsigned long long b) {
    unsigned long long a_lo = a & 0xffffffffULL, a_hi = a >> 32;
    unsigned long long b_lo = b & 0xffffffffULL, b_hi = b >> 32;
    unsigned long long p00 = a_lo * b_lo;
    unsigned long long p01 = a_lo * b_hi;
    unsigned long long p10 = a_hi * b_lo;
    unsigned long long p11 = a_hi * b_hi;
    unsigned long long mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    BtInt128 r;
    r.low = (p00 & 0xffffffffULL) | ((mid & 0xffffffffULL) << 32);
    r.high = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

// Signed 64x64 -> 128 multiply
__device__ inline BtInt128 bt128_smul(long long a, long long b) {
    bool neg = (a < 0);
    if (neg) a = -a;
    if (b < 0) { neg = !neg; b = -b; }
    BtInt128 r = bt128_umul((unsigned long long)a, (unsigned long long)b);
    return neg ? bt128_neg(r) : r;
}

// 128 * 64 -> 128
__device__ inline BtInt128 bt128_mul_i64(BtInt128 a, long long b) {
    bool neg = ((long long)a.high < 0);
    if (neg) a = bt128_neg(a);
    if (b < 0) { neg = !neg; b = -b; }
    BtInt128 r = bt128_umul(a.low, (unsigned long long)b);
    r.high += a.high * (unsigned long long)b;
    return neg ? bt128_neg(r) : r;
}

__device__ inline float bt128_to_float(BtInt128 a) {
    return ((long long)a.high >= 0)
        ? (float)((double)a.high * 18446744073709551616.0 + (double)a.low)
        : -(float)((double)(bt128_neg(a)).high * 18446744073709551616.0 + (double)(bt128_neg(a)).low);
}

// ============================================================================
// Point types
// ============================================================================

#ifndef BT_POINT32_DEFINED
#define BT_POINT32_DEFINED
struct BtPoint32 {
    int x, y, z, index;
};
#endif
__device__ inline BtPoint32 bp32(int x, int y, int z) {
    BtPoint32 p; p.x = x; p.y = y; p.z = z; p.index = -1; return p;
}
__device__ inline bool bp32_eq(BtPoint32 a, BtPoint32 b) { return a.x==b.x && a.y==b.y && a.z==b.z; }
__device__ inline bool bp32_ne(BtPoint32 a, BtPoint32 b) { return !bp32_eq(a,b); }
__device__ inline BtPoint32 bp32_add(BtPoint32 a, BtPoint32 b) { return bp32(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline BtPoint32 bp32_sub(BtPoint32 a, BtPoint32 b) { return bp32(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline bool bp32_isZero(BtPoint32 a) { return a.x==0 && a.y==0 && a.z==0; }

struct BtPoint64 {
    long long x, y, z;
};
__device__ inline BtPoint64 bp64(long long x, long long y, long long z) {
    BtPoint64 p; p.x = x; p.y = y; p.z = z; return p;
}
__device__ inline bool bp64_isZero(BtPoint64 a) { return a.x==0 && a.y==0 && a.z==0; }
__device__ inline long long bp64_dot64(BtPoint64 a, BtPoint64 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// Point32 cross -> Point64
__device__ inline BtPoint64 bp32_cross(BtPoint32 a, BtPoint32 b) {
    return bp64((long long)a.y*b.z - (long long)a.z*b.y,
                (long long)a.z*b.x - (long long)a.x*b.z,
                (long long)a.x*b.y - (long long)a.y*b.x);
}
// Point32 cross Point64
__device__ inline BtPoint64 bp32_cross64(BtPoint32 a, BtPoint64 b) {
    return bp64((long long)a.y*b.z - (long long)a.z*b.y,
                (long long)a.z*b.x - (long long)a.x*b.z,
                (long long)a.x*b.y - (long long)a.y*b.x);
}
// Point32 dot Point32 -> int64
__device__ inline long long bp32_dot(BtPoint32 a, BtPoint32 b) {
    return (long long)a.x*b.x + (long long)a.y*b.y + (long long)a.z*b.z;
}
// Point32 dot Point64 -> int64
__device__ inline long long bp32_dot64(BtPoint32 a, BtPoint64 b) {
    return (long long)a.x*b.x + (long long)a.y*b.y + (long long)a.z*b.z;
}
// Point32 dot Point32 as int64 (shorthand matching bt original's dot calls)
__device__ inline long long bp32_dot64_32(BtPoint32 a, BtPoint32 b) {
    return (long long)a.x*b.x + (long long)a.y*b.y + (long long)a.z*b.z;
}

// ============================================================================
// Rational types for exact comparisons
// ============================================================================

struct BtRational64 {
    unsigned long long num;
    unsigned long long den;
    int sign;
};

__device__ inline BtRational64 br64_make(long long numerator, long long denominator) {
    BtRational64 r;
    if (numerator > 0) { r.sign = 1; r.num = (unsigned long long)numerator; }
    else if (numerator < 0) { r.sign = -1; r.num = (unsigned long long)(-numerator); }
    else { r.sign = 0; r.num = 0; }
    if (denominator > 0) { r.den = (unsigned long long)denominator; }
    else if (denominator < 0) { r.sign = -r.sign; r.den = (unsigned long long)(-denominator); }
    else { r.den = 0; }
    return r;
}
__device__ inline bool br64_isNegInf(BtRational64 r) { return (r.sign < 0) && (r.den == 0); }
__device__ inline bool br64_isNaN(BtRational64 r) { return (r.sign == 0) && (r.den == 0); }

__device__ inline int br64_cmp(BtRational64 a, BtRational64 b) {
    if (a.sign != b.sign) return a.sign - b.sign;
    if (a.sign == 0) return 0;
    return a.sign * bt128_ucmp(bt128_umul(a.num, b.den), bt128_umul(a.den, b.num));
}

// ============================================================================
// PointR128 for intersection vertices
// ============================================================================

// ============================================================================
// Rational128 for exact vertex dot products
// ============================================================================

struct BtRational128 {
    BtInt128 num;
    BtInt128 den;
    int sign;
    bool isInt64;
};

__device__ inline BtRational128 br128_from_i64(long long val) {
    BtRational128 r;
    if (val > 0) { r.sign = 1; r.num = bt128_from_i64(val); }
    else if (val < 0) { r.sign = -1; r.num = bt128_from_i64(-val); }
    else { r.sign = 0; r.num = bt128_from_u64(0); }
    r.den = bt128_from_u64(1);
    r.isInt64 = true;
    return r;
}

// DMul<Int128, uint64_t>::mul — 128x128 -> (256 bits as) low128, high128
__device__ inline void bt_dmul_128(BtInt128 a, BtInt128 b, BtInt128* lo, BtInt128* hi) {
    BtInt128 p00 = bt128_umul(a.low, b.low);
    BtInt128 p01 = bt128_umul(a.low, b.high);
    BtInt128 p10 = bt128_umul(a.high, b.low);
    BtInt128 p11 = bt128_umul(a.high, b.high);
    BtInt128 p0110 = bt128_add(bt128_from_u64(p01.low), bt128_from_u64(p10.low));
    p11 = bt128_add(p11, bt128_from_u64(p01.high));
    p11 = bt128_add(p11, bt128_from_u64(p10.high));
    p11 = bt128_add(p11, bt128_from_u64(p0110.high));
    // shlHalf(p0110): p0110.high = p0110.low; p0110.low = 0;
    BtInt128 p0110s = bt128_make(0, p0110.low);
    BtInt128 sum = bt128_add(p00, p0110s);
    // carry: if sum < p00
    if (bt128_lt(sum, p00)) p11 = bt128_add(p11, bt128_from_u64(1));
    *lo = sum;
    *hi = p11;
}

__device__ inline int br128_cmp_i64(BtRational128 a, long long b);

__device__ inline int br128_cmp(BtRational128 a, BtRational128 b) {
    if (a.sign != b.sign) return a.sign - b.sign;
    if (a.sign == 0) return 0;
    if (a.isInt64) return -br128_cmp_i64(b, a.sign * (long long)a.num.low);
    BtInt128 nbdLo, nbdHi, dbnLo, dbnHi;
    bt_dmul_128(a.num, b.den, &nbdLo, &nbdHi);
    bt_dmul_128(a.den, b.num, &dbnLo, &dbnHi);
    int c = bt128_ucmp(nbdHi, dbnHi);
    if (c) return c * a.sign;
    return bt128_ucmp(nbdLo, dbnLo) * a.sign;
}

__device__ inline int br128_cmp_i64(BtRational128 a, long long b) {
    if (a.isInt64) {
        long long av = a.sign * (long long)a.num.low;
        return (av > b) ? 1 : (av < b) ? -1 : 0;
    }
    if (b > 0) { if (a.sign <= 0) return -1; }
    else if (b < 0) { if (a.sign >= 0) return 1; b = -b; }
    else return a.sign;
    return bt128_ucmp(a.num, bt128_mul_i64(a.den, b)) * a.sign;
}

// ============================================================================
// Graph types
// ============================================================================

struct BtVertex;
struct BtEdge;
typedef int BtVIndex;
#define BT_VI_NULL (-1)
struct BtEdge {
    BtEdge* next;
    BtEdge* prev;
    BtEdge* reverse;
    BtVIndex target;
    int copy;
};

struct BtVertex {
    BtVIndex next;
    BtVIndex prev;
    BtEdge* edges;
    BtPoint32 point;
    int copy;
};

struct BtIntermediateHull {
    BtVIndex minXy;
    BtVIndex maxXy;
    BtVIndex minYx;
    BtVIndex maxYx;
};

// D&C stack item for iterative computeInternal
struct BtDCStackItem {
    int start;
    int end;
    int stage;                   // 0 = descend, 1 = merge
    BtIntermediateHull right_hull;
    BtIntermediateHull* result;  // where to write the merged hull (left child writes here directly)
};

__device__ inline void bt_edge_link(BtEdge* a, BtEdge* n) {
    a->next = n;
    n->prev = a;
}

// ============================================================================
// Number of hull groups for parallel tree merge (16 groups × 2 threads each)
// ============================================================================
#define BT_HULL_GROUPS (WARP_SIZE / 2)

// Warp-shuffle helpers for pointer and BtRational64 transfer between lane pairs.
// shfl_ll: shuffle a long long (2 int32 shuffles).
__device__ inline long long shfl_ll(long long v, int src, unsigned mask) {
    unsigned int lo = __shfl_sync(mask, (unsigned int)(unsigned long long)v,         src);
    unsigned int hi = __shfl_sync(mask, (unsigned int)((unsigned long long)v >> 32), src);
    return (long long)((unsigned long long)lo | ((unsigned long long)hi << 32));
}
// shfl_edge_ptr: shuffle a BtEdge* (pointer) from src lane.
__device__ inline BtEdge* shfl_edge_ptr(BtEdge* p, int src, unsigned mask) {
    return (BtEdge*)(unsigned long long)shfl_ll((long long)(unsigned long long)p, src, mask);
}
// shfl_br64: shuffle a BtRational64 (5 int32 shuffles) from src lane.
__device__ inline BtRational64 shfl_br64(BtRational64 r, int src, unsigned mask) {
    BtRational64 out;
    out.num  = (unsigned long long)shfl_ll((long long)r.num, src, mask);
    out.den  = (unsigned long long)shfl_ll((long long)r.den, src, mask);
    out.sign = __shfl_sync(mask, r.sign, src);
    return out;
}

// ============================================================================
// Pool allocator backed by DeviceHeap (lane 0 only)
// ============================================================================

#define BT_ERR_POOL_EXHAUST 8
#define BTPOOL_BLOCK_SIZE   1024   // edges per slab
#define BTPOOL_MAX_BLOCKS   64     // max heap slabs tracked for cleanup

// BtPool: free list of fixed-size objects, backed by heap slabs.
// All operations (init, new, free) are lane-0-only.
struct BtPool {
    void*       freeList;
    DeviceHeap* scratch_heap;
    int         objSize;    // padded to multiple of 4
    int         error;
    int         nblocks;
    CheckedBuf<void*> blocks;  // allocated from WarpPool, capacity BTPOOL_MAX_BLOCKS
    int         growSlabSize;   // edges per growth slab
};

// Allocate one slab from scratch_heap and prepend to the free list.
// Uses growSlabSize if set, else BTPOOL_BLOCK_SIZE. Heap-backed only.
__device__ inline int btpool_add_block(BtPool* p) {
    if (p->nblocks >= BTPOOL_MAX_BLOCKS || !p->blocks.raw()) { p->error = BT_ERR_POOL_EXHAUST; return -1; }
    int slab_n = (p->growSlabSize > 0) ? p->growSlabSize : BTPOOL_BLOCK_SIZE;
    void* block = NULL;
    if (heap_alloc(p->scratch_heap, (unsigned int)(slab_n * p->objSize), &block) != HEAP_OK) {
        p->error = BT_ERR_POOL_EXHAUST; return -1;
    }
    p->blocks[p->nblocks++] = block;
    // slot[0] -> existing freeList; slot[i] -> slot[i-1] for i > 0
    char* b = (char*)block;
    int   padded = p->objSize;
    *(void**)b = p->freeList;
    for (int i = 1; i < slab_n; i++)
        *(void**)(b + i * padded) = b + (i - 1) * padded;
    p->freeList = b + (slab_n - 1) * padded;  // head = last slot
    return 0;
}

// Initialise per-lane pool with one heap-allocated slab of slab_edges objects.
// Safe to call from multiple lanes concurrently (heap_alloc uses per-arena locks).
__device__ inline int btpool_init_sized(BtPool* p, DeviceHeap* scratch_heap, int objSize, int slab_edges,
                                        CheckedBuf<void*> blocks_buf) {
    p->scratch_heap = scratch_heap;
    p->objSize      = (objSize + 3) & ~3;
    p->freeList     = NULL;
    p->error        = 0;
    p->nblocks      = 0;
    p->blocks       = blocks_buf;
    p->growSlabSize = slab_edges;
    if (btpool_add_block(p) < 0) return -1;
    return 0;
}

// Initialise pool with 2 pre-allocated slabs. Lane 0 only.
__device__ inline int btpool_init(BtPool* p, DeviceHeap* scratch_heap, int objSize,
                                  CheckedBuf<void*> blocks_buf) {
    p->scratch_heap = scratch_heap;
    p->objSize      = (objSize + 3) & ~3;
    p->freeList     = NULL;
    p->error        = 0;
    p->nblocks      = 0;
    p->blocks       = blocks_buf;
    p->growSlabSize = 0;
    if (btpool_add_block(p) < 0) return -1;
    if (btpool_add_block(p) < 0) return -1;
    return 0;
}

__device__ inline void* btpool_new(BtPool* p) {
    if (!p->freeList) {
        if (btpool_add_block(p) < 0) return NULL;
    }
    void* obj = p->freeList;
    p->freeList = *(void**)obj;
    int* c = (int*)obj;
    for (int i = 0; i < p->objSize / 4; i++) c[i] = 0;
    return obj;
}

__device__ inline void btpool_free(BtPool* p, void* obj) {
    *(void**)obj = p->freeList;
    p->freeList = obj;
}

// Typed wrapper
__device__ inline BtEdge* btpool_new_edge(BtPool* p) { return (BtEdge*)btpool_new(p); }

// Simple alloc from WarpPool (lane 0 only)
__device__ inline void* bt_alloc(WarpPool* wp, int bytes) {
    int aligned = (bytes + 15) & ~15;
    if (wp->offset + aligned > wp->capacity) { wp->error = 1; return NULL; }
    void* ptr = wp->base + wp->offset;
    wp->offset += aligned;
    return ptr;
}

// Rewind WarpPool to a saved offset (lane 0 only).
// Use to free the last allocation(s) back to the pool.
__device__ inline void bt_rewind(WarpPool* wp, int saved_offset) {
    wp->offset = saved_offset;
}

// ============================================================================
// Constants
// ============================================================================

#define BT_DC_MAX_STACK  4096
#define BT_DC_MAX_STACK_LOCAL 64

// Error codes (binary flags, combined via OR)
#define BT_ERR_WARP_POOL_OOM  1   // bt_alloc failed (WarpPool capacity exceeded)
#define BT_ERR_SORT_STACK     2
#define BT_ERR_HEAP_TO_WARP  16   // heap_alloc for WarpPool backing failed
#define BT_ERR_HEAP_OUTPUT   32   // heap_alloc for output mesh chunk failed

// ============================================================================
// Scratch size calculation
// ============================================================================

// Aligned size matching bt_alloc's 16-byte alignment.
#define BT_ALIGN16(x) (((x) + 15) & ~15)

// Compute total WarpPool scratch bytes needed for D&C hull with n points.
// Sort scratch is rewound after sort. D&C stacks are rewound after computeInternal.
// BFS queues are allocated after D&C, sharing that space.
// Edge pool is backed by DeviceHeap (scratch_heap), not included here.
// Per-lane D&C stacks and BtPool blocks arrays are allocated from WarpPool.
// Peak = presort + max(sort_scratch, persistent + dc_stacks + pool_blocks + cleanup_blocks + bfs_queues).
__host__ __device__ inline int dandc_scratch_bytes(int n) {
    int total = 0;
    // presort: BtPoint32 array (persists through sort, NOT rewound)
    total += BT_ALIGN16(n * (int)sizeof(BtPoint32));

    int sort_scratch = BT_ALIGN16(n * (int)sizeof(BtPoint32) + WS_MAX_STACK * 2 * (int)sizeof(int));

    int postsort_persistent = 0;
    // postsort: vertex block (persists)
    postsort_persistent += BT_ALIGN16(n * (int)sizeof(BtVertex));
    // per-lane BtPool blocks arrays: 32 lanes × BTPOOL_MAX_BLOCKS pointers (persists until cleanup)
    postsort_persistent += BT_ALIGN16(WARP_SIZE * BTPOOL_MAX_BLOCKS * (int)sizeof(void*));

    // Per-lane D&C stacks: 32 lanes × BT_DC_MAX_STACK_LOCAL items (rewound after D&C)
    int dc_stacks = BT_ALIGN16(WARP_SIZE * BT_DC_MAX_STACK_LOCAL * (int)sizeof(BtDCStackItem));

    // BFS queues for mesh extraction + volume (each n * sizeof(BtVIndex))
    int phase_bfs = BT_ALIGN16(n * (int)sizeof(BtVIndex));

    // dc_stacks and bfs_queues don't overlap (dc rewound before bfs)
    int postsort = postsort_persistent + (dc_stacks > phase_bfs ? dc_stacks : phase_bfs);

    total += (sort_scratch > postsort) ? sort_scratch : postsort;
    // alignment padding headroom
    total += 32 * 1024;
    return total;
}

// ============================================================================
// State struct (replaces btConvexHullInternal class members)
// ============================================================================

struct BtHullState {
    float scaling[3];
    float center[3];
    int mergeStamp;
    int* mergeStampPtr;    // if non-null, use atomicAdd on shared stamp
    int minAxis, medAxis, maxAxis;
    BtVIndex vertexList;
    BtVertex* __restrict__ vblock;
    WarpPool*   wp;
    DeviceHeap* scratch_heap;  // backing for edgePool slabs
    int         npoints;       // original point count (queue size bound)
#ifdef COACD_BEAM_DEBUG
    int fma_total_edges;       // instrumentation: total edges chased in bt_findMaxAngle
    int fma_min_edges;         // instrumentation: min edges per call
    int fma_max_edges;         // instrumentation: max edges per call
    int fma_calls;             // instrumentation: number of bt_findMaxAngle calls
#endif
};

// ============================================================================
// Per-lane D&C state (lightweight: only mutable per-lane fields)
// ============================================================================

struct BtDCState {
    BtPool edgePool;
    int mergeStamp;
    int* mergeStampPtr;    // if non-null, use atomicAdd on shared stamp
    BtVertex* __restrict__ vblock;
#ifdef TRACK_MAX_EDGE_PAIRS
    int usedEdgePairs;
    int maxEdgePairs;
#endif
#ifdef COACD_BEAM_DEBUG
    int fma_total_edges;
    int fma_min_edges;
    int fma_max_edges;
    int fma_calls;
#endif
};

// ============================================================================
// Core algorithm functions
// ============================================================================

__device__ inline BtEdge* bt_newEdgePair(BtDCState* __restrict__ dc, BtVIndex from, BtVIndex to) {
    BtEdge* e = btpool_new_edge(&dc->edgePool);
    BtEdge* r = btpool_new_edge(&dc->edgePool);
    if (!e || !r) return NULL;
    e->reverse = r;
    r->reverse = e;
    e->copy = dc->mergeStamp;
    r->copy = dc->mergeStamp;
    e->target = to;
    r->target = from;
    e->next = NULL; e->prev = NULL;
    r->next = NULL; r->prev = NULL;
#ifdef TRACK_MAX_EDGE_PAIRS
    dc->usedEdgePairs++;
    if (dc->usedEdgePairs > dc->maxEdgePairs) {
        dc->maxEdgePairs = dc->usedEdgePairs;
    }
#endif
    return e;
}

__device__ inline void bt_removeEdgePair(BtDCState* dc, BtEdge* edge) {
    BtEdge* n = edge->next;
    BtEdge* r = edge->reverse;
    if (n != edge) {
        n->prev = edge->prev;
        edge->prev->next = n;
        dc->vblock[r->target].edges = n;
    } else {
        dc->vblock[r->target].edges = NULL;
    }
    n = r->next;
    if (n != r) {
        n->prev = r->prev;
        r->prev->next = n;
        dc->vblock[edge->target].edges = n;
    } else {
        dc->vblock[edge->target].edges = NULL;
    }
    btpool_free(&dc->edgePool, edge);
    btpool_free(&dc->edgePool, r);
#ifdef TRACK_MAX_EDGE_PAIRS
    dc->usedEdgePairs--;
#endif
}

enum BtOrientation { BT_NONE, BT_CLOCKWISE, BT_COUNTER_CLOCKWISE };

__device__ inline BtOrientation bt_getOrientation(BtEdge* prev_e, BtEdge* next_e, BtPoint32 s_dir, BtPoint32 t_dir, BtVertex* __restrict__ vblock) {
    if (prev_e->next == next_e) {
        if (prev_e->prev == next_e) {
            BtPoint64 n = bp32_cross(t_dir, s_dir);
            BtPoint64 m = bp32_cross(
                bp32_sub(vblock[prev_e->target].point, vblock[next_e->reverse->target].point),
                bp32_sub(vblock[next_e->target].point, vblock[next_e->reverse->target].point));
            long long dot = bp64_dot64(n, m);
            return (dot > 0) ? BT_COUNTER_CLOCKWISE : BT_CLOCKWISE;
        }
        return BT_COUNTER_CLOCKWISE;
    }
    else if (prev_e->prev == next_e) {
        return BT_CLOCKWISE;
    }
    return BT_NONE;
}

__device__ inline BtEdge* bt_findMaxAngle(int mergeStamp, bool ccw, BtVIndex start,
    BtPoint32 s_dir, BtPoint64 rxs, BtPoint64 sxrxs, BtRational64* minCot,
    int* edge_count, BtVertex* __restrict__ vblock)
{
    BtEdge* minEdge = NULL;
    BtEdge* e = vblock[start].edges;
    if (!e) { if (edge_count) *edge_count = 0; return NULL; }
    int count = 0;
    do {
        count++;
        if (e->copy > mergeStamp) {
            BtPoint32 t = bp32_sub(vblock[e->target].point, vblock[start].point);
            BtRational64 cot = br64_make(bp32_dot64(t, sxrxs), bp32_dot64(t, rxs));
            if (!br64_isNaN(cot)) {
                if (minEdge == NULL) {
                    *minCot = cot;
                    minEdge = e;
                } else {
                    int c = br64_cmp(cot, *minCot);
                    if (c < 0) {
                        *minCot = cot;
                        minEdge = e;
                    } else if (c == 0 && (ccw == (bt_getOrientation(minEdge, e, s_dir, t, vblock) == BT_COUNTER_CLOCKWISE))) {
                        minEdge = e;
                    }
                }
            }
        }
        e = e->next;
        if (!e) {
            DPRINTF("[BUG] bt_findMaxAngle: NULL edge->next after %d edges, "
                    "blk=%d lane=%d start=%d mergeStamp=%d\n",
                    count, blockIdx.x, threadIdx.x, start, mergeStamp);
            if (edge_count) *edge_count = count;
            return minEdge;  // bail out instead of crashing
        }
    } while (e != vblock[start].edges);
    if (edge_count) *edge_count = count;
    return minEdge;
}

__device__ inline void bt_findEdgeForCoplanarFaces(int mergeStamp, BtVIndex c0, BtVIndex c1,
    BtEdge** e0, BtEdge** e1, BtVIndex stop0, BtVIndex stop1, BtVertex* __restrict__ vblock)
{
    BtEdge* start0 = *e0;
    BtEdge* start1 = *e1;
    BtPoint32 et0 = start0 ? vblock[start0->target].point : vblock[c0].point;
    BtPoint32 et1 = start1 ? vblock[start1->target].point : vblock[c1].point;
    BtPoint32 s_dir = bp32_sub(vblock[c1].point, vblock[c0].point);
    BtPoint64 normal = bp32_cross(bp32(0,0,-1), s_dir);
    // Use whichever start edge exists to define the coplanar normal
    if (start0 || start1) {
        BtVIndex ref = (start0 ? start0 : start1)->target;
        normal = bp32_cross(bp32_sub(vblock[ref].point, vblock[c0].point), s_dir);
    }
    long long dist = bp32_dot64(vblock[c0].point, normal);
    BtPoint64 perp = bp32_cross64(s_dir, normal);

    long long maxDot0 = bp32_dot64(et0, perp);
    if (*e0) {
        while ((*e0)->target != stop0) {
            BtEdge* e = (*e0)->reverse->prev;
            if (bp32_dot64(vblock[e->target].point, normal) < dist) break;
            if (e->copy == mergeStamp) break;
            long long dot = bp32_dot64(vblock[e->target].point, perp);
            if (dot <= maxDot0) break;
            maxDot0 = dot;
            *e0 = e;
            et0 = vblock[e->target].point;
        }
    }

    long long maxDot1 = bp32_dot64(et1, perp);
    if (*e1) {
        while ((*e1)->target != stop1) {
            BtEdge* e = (*e1)->reverse->next;
            if (bp32_dot64(vblock[e->target].point, normal) < dist) break;
            if (e->copy == mergeStamp) break;
            long long dot = bp32_dot64(vblock[e->target].point, perp);
            if (dot <= maxDot1) break;
            maxDot1 = dot;
            *e1 = e;
            et1 = vblock[e->target].point;
        }
    }

    long long dx = maxDot1 - maxDot0;
    if (dx > 0) {
        while (true) {
            long long dy = bp32_dot64_32(bp32_sub(et1, et0), s_dir);
            if (*e0 && ((*e0)->target != stop0)) {
                BtEdge* f0 = (*e0)->next->reverse;
                if (f0->copy > mergeStamp) {
                    long long dx0 = bp32_dot64(bp32_sub(vblock[f0->target].point, et0), perp);
                    long long dy0 = bp32_dot64_32(bp32_sub(vblock[f0->target].point, et0), s_dir);
                    if ((dx0 == 0) ? (dy0 < 0) : ((dx0 < 0) && (br64_cmp(br64_make(dy0, dx0), br64_make(dy, dx)) >= 0))) {
                        et0 = vblock[f0->target].point;
                        dx = bp32_dot64(bp32_sub(et1, et0), perp);
                        *e0 = (*e0 == start0) ? NULL : f0;
                        continue;
                    }
                }
            }
            if (*e1 && ((*e1)->target != stop1)) {
                BtEdge* f1 = (*e1)->reverse->next;
                if (f1->copy > mergeStamp) {
                    BtPoint32 d1 = bp32_sub(vblock[f1->target].point, et1);
                    if (bp32_dot64(d1, normal) == 0) {
                        long long dx1 = bp32_dot64(d1, perp);
                        long long dy1 = bp32_dot64_32(d1, s_dir);
                        long long dxn = bp32_dot64(bp32_sub(vblock[f1->target].point, et0), perp);
                        if ((dxn > 0) && ((dx1 == 0) ? (dy1 < 0) : ((dx1 < 0) && (br64_cmp(br64_make(dy1, dx1), br64_make(dy, dx)) > 0)))) {
                            *e1 = f1;
                            et1 = vblock[(*e1)->target].point;
                            dx = dxn;
                            continue;
                        }
                    }
                }
            }
            break;
        }
    } else if (dx < 0) {
        while (true) {
            long long dy = bp32_dot64_32(bp32_sub(et1, et0), s_dir);
            if (*e1 && ((*e1)->target != stop1)) {
                BtEdge* f1 = (*e1)->prev->reverse;
                if (f1->copy > mergeStamp) {
                    long long dx1 = bp32_dot64(bp32_sub(vblock[f1->target].point, et1), perp);
                    long long dy1 = bp32_dot64_32(bp32_sub(vblock[f1->target].point, et1), s_dir);
                    if ((dx1 == 0) ? (dy1 > 0) : ((dx1 < 0) && (br64_cmp(br64_make(dy1, dx1), br64_make(dy, dx)) <= 0))) {
                        et1 = vblock[f1->target].point;
                        dx = bp32_dot64(bp32_sub(et1, et0), perp);
                        *e1 = (*e1 == start1) ? NULL : f1;
                        continue;
                    }
                }
            }
            if (*e0 && ((*e0)->target != stop0)) {
                BtEdge* f0 = (*e0)->reverse->prev;
                if (f0->copy > mergeStamp) {
                    BtPoint32 d0 = bp32_sub(vblock[f0->target].point, et0);
                    if (bp32_dot64(d0, normal) == 0) {
                        long long dx0 = bp32_dot64(d0, perp);
                        long long dy0 = bp32_dot64_32(d0, s_dir);
                        long long dxn = bp32_dot64(bp32_sub(et1, vblock[f0->target].point), perp);
                        if ((dxn < 0) && ((dx0 == 0) ? (dy0 > 0) : ((dx0 < 0) && (br64_cmp(br64_make(dy0, dx0), br64_make(dy, dx)) < 0)))) {
                            *e0 = f0;
                            et0 = vblock[(*e0)->target].point;
                            dx = dxn;
                            continue;
                        }
                    }
                }
            }
            break;
        }
    }
}

__device__ inline bool bt_mergeProjection(BtIntermediateHull* h0, BtIntermediateHull* h1, BtVIndex* c0, BtVIndex* c1, BtVertex* __restrict__ vblock) {
    BtVIndex v0 = h0->maxYx;
    BtVIndex v1 = h1->minYx;
    if ((vblock[v0].point.x == vblock[v1].point.x) && (vblock[v0].point.y == vblock[v1].point.y)) {
        BtVIndex v1p = vblock[v1].prev;
        if (v1p == v1) {
            *c0 = v0;
            if (vblock[v1].edges) {
                v1 = vblock[v1].edges->target;
            }
            *c1 = v1;
            return false;
        }
        BtVIndex v1n = vblock[v1].next;
        vblock[v1p].next = v1n;
        vblock[v1n].prev = v1p;
        if (v1 == h1->minXy) {
            h1->minXy = ((vblock[v1n].point.x < vblock[v1p].point.x) || ((vblock[v1n].point.x == vblock[v1p].point.x) && (vblock[v1n].point.y < vblock[v1p].point.y))) ? v1n : v1p;
        }
        if (v1 == h1->maxXy) {
            h1->maxXy = ((vblock[v1n].point.x > vblock[v1p].point.x) || ((vblock[v1n].point.x == vblock[v1p].point.x) && (vblock[v1n].point.y > vblock[v1p].point.y))) ? v1n : v1p;
        }
    }

    v0 = h0->maxXy;
    v1 = h1->maxXy;
    BtVIndex v00 = BT_VI_NULL;
    BtVIndex v10 = BT_VI_NULL;
    int sign = 1;

    for (int side = 0; side <= 1; side++) {
        int dx = (vblock[v1].point.x - vblock[v0].point.x) * sign;
        if (dx > 0) {
            while (true) {
                int dy = vblock[v1].point.y - vblock[v0].point.y;
                BtVIndex w0 = side ? vblock[v0].next : vblock[v0].prev;
                if (w0 != v0) {
                    int dx0 = (vblock[w0].point.x - vblock[v0].point.x) * sign;
                    int dy0 = vblock[w0].point.y - vblock[v0].point.y;
                    if ((dy0 <= 0) && ((dx0 == 0) || ((dx0 < 0) && ((long long)dy0 * dx <= (long long)dy * dx0)))) {
                        v0 = w0; dx = (vblock[v1].point.x - vblock[v0].point.x) * sign; continue;
                    }
                }
                BtVIndex w1 = side ? vblock[v1].next : vblock[v1].prev;
                if (w1 != v1) {
                    int dx1 = (vblock[w1].point.x - vblock[v1].point.x) * sign;
                    int dy1 = vblock[w1].point.y - vblock[v1].point.y;
                    int dxn = (vblock[w1].point.x - vblock[v0].point.x) * sign;
                    if ((dxn > 0) && (dy1 < 0) && ((dx1 == 0) || ((dx1 < 0) && ((long long)dy1 * dx < (long long)dy * dx1)))) {
                        v1 = w1; dx = dxn; continue;
                    }
                }
                break;
            }
        } else if (dx < 0) {
            while (true) {
                int dy = vblock[v1].point.y - vblock[v0].point.y;
                BtVIndex w1 = side ? vblock[v1].prev : vblock[v1].next;
                if (w1 != v1) {
                    int dx1 = (vblock[w1].point.x - vblock[v1].point.x) * sign;
                    int dy1 = vblock[w1].point.y - vblock[v1].point.y;
                    if ((dy1 >= 0) && ((dx1 == 0) || ((dx1 < 0) && ((long long)dy1 * dx <= (long long)dy * dx1)))) {
                        v1 = w1; dx = (vblock[v1].point.x - vblock[v0].point.x) * sign; continue;
                    }
                }
                BtVIndex w0 = side ? vblock[v0].prev : vblock[v0].next;
                if (w0 != v0) {
                    int dx0 = (vblock[w0].point.x - vblock[v0].point.x) * sign;
                    int dy0 = vblock[w0].point.y - vblock[v0].point.y;
                    int dxn = (vblock[v1].point.x - vblock[w0].point.x) * sign;
                    if ((dxn < 0) && (dy0 > 0) && ((dx0 == 0) || ((dx0 < 0) && ((long long)dy0 * dx < (long long)dy * dx0)))) {
                        v0 = w0; dx = dxn; continue;
                    }
                }
                break;
            }
        } else {
            int x = vblock[v0].point.x;
            int y0 = vblock[v0].point.y;
            BtVIndex w0 = v0;
            BtVIndex t;
            while (((t = side ? vblock[w0].next : vblock[w0].prev) != v0) && (vblock[t].point.x == x) && (vblock[t].point.y <= y0)) {
                w0 = t; y0 = vblock[t].point.y;
            }
            v0 = w0;
            int y1 = vblock[v1].point.y;
            BtVIndex w1 = v1;
            while (((t = side ? vblock[w1].prev : vblock[w1].next) != v1) && (vblock[t].point.x == x) && (vblock[t].point.y >= y1)) {
                w1 = t; y1 = vblock[t].point.y;
            }
            v1 = w1;
        }
        if (side == 0) {
            v00 = v0; v10 = v1;
            v0 = h0->minXy; v1 = h1->minXy; sign = -1;
        }
    }

    vblock[v0].prev = v1;
    vblock[v1].next = v0;
    vblock[v00].next = v10;
    vblock[v10].prev = v00;

    if (vblock[h1->minXy].point.x < vblock[h0->minXy].point.x) h0->minXy = h1->minXy;
    if (vblock[h1->maxXy].point.x >= vblock[h0->maxXy].point.x) h0->maxXy = h1->maxXy;
    h0->maxYx = h1->maxYx;

    *c0 = v00;
    *c1 = v10;
    return true;
}

// Check circularity of a vertex's edge ring. Returns 0 if OK, -1 if broken.
__device__ inline int bt_checkEdgeRing(BtVIndex v, const char* /*label*/, BtVertex* __restrict__ vblock) {
    BtEdge* e = vblock[v].edges;
    if (!e) return 0;
    BtEdge* start = e;
    for (int i = 0; i < 10000; i++) {
        e = e->next;
        if (!e) return -1;
        if (e == start) return 0;
    }
    return -1;
}

__device__ inline void bt_merge(BtDCState* dc, BtIntermediateHull* h0, BtIntermediateHull* h1) {
    // Single-thread merge (caller is the one active thread).
    if (h1->maxXy == BT_VI_NULL) return;
    if (h0->maxXy == BT_VI_NULL) { *h0 = *h1; return; }

    BtVIndex c0 = BT_VI_NULL;
    BtEdge* toPrev0 = NULL;
    BtEdge* firstNew0 = NULL;
    BtEdge* pendingHead0 = NULL;
    BtEdge* pendingTail0 = NULL;
    BtVIndex c1 = BT_VI_NULL;
    BtEdge* toPrev1 = NULL;
    BtEdge* firstNew1 = NULL;
    BtEdge* pendingHead1 = NULL;
    BtEdge* pendingTail1 = NULL;
    BtPoint32 prevPoint;

    if (dc->mergeStampPtr) {
        dc->mergeStamp = atomicAdd(dc->mergeStampPtr, -1) - 1;
    } else {
        dc->mergeStamp--;
    }
    int mergeStamp = dc->mergeStamp;

    if (bt_mergeProjection(h0, h1, &c0, &c1, dc->vblock)) {
        BtPoint32 sd = bp32_sub(dc->vblock[c1].point, dc->vblock[c0].point);
        BtPoint64 normal = bp32_cross(bp32(0,0,-1), sd);
        BtPoint64 t = bp32_cross64(sd, normal);

        BtEdge* e = dc->vblock[c0].edges;
        BtEdge* start0 = NULL;
        if (e) {
            do {
                long long dot = bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c0].point), normal);
                if ((dot == 0) && (bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c0].point), t) > 0)) {
                    if (!start0 || (bt_getOrientation(start0, e, sd, bp32(0,0,-1), dc->vblock) == BT_CLOCKWISE))
                        start0 = e;
                }
                e = e->next;
            } while (e != dc->vblock[c0].edges);
        }

        e = dc->vblock[c1].edges;
        BtEdge* start1 = NULL;
        if (e) {
            do {
                long long dot = bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c1].point), normal);
                if ((dot == 0) && (bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c1].point), t) > 0)) {
                    if (!start1 || (bt_getOrientation(start1, e, sd, bp32(0,0,-1), dc->vblock) == BT_COUNTER_CLOCKWISE))
                        start1 = e;
                }
                e = e->next;
            } while (e != dc->vblock[c1].edges);
        }

        if (start0 || start1) {
            bt_findEdgeForCoplanarFaces(mergeStamp, c0, c1, &start0, &start1, BT_VI_NULL, BT_VI_NULL, dc->vblock);
            if (start0) c0 = start0->target;
            if (start1) c1 = start1->target;
        }
        prevPoint = dc->vblock[c1].point;
        prevPoint.z++;
    } else {
        prevPoint = dc->vblock[c1].point;
        prevPoint.x++;
    }

    BtVIndex first0 = c0;
    BtVIndex first1 = c1;
    bool firstRun = true;

    while (true) {
        BtPoint32 sd = bp32_sub(dc->vblock[c1].point, dc->vblock[c0].point);
        BtPoint32 r = bp32_sub(prevPoint, dc->vblock[c0].point);
        BtPoint64 rxs = bp32_cross(r, sd);
        BtPoint64 sxrxs = bp32_cross64(sd, rxs);

        BtRational64 minCot0, minCot1;
        int ec0 = 0, ec1 = 0;
        BtEdge* min0 = bt_findMaxAngle(mergeStamp, false, c0, sd, rxs, sxrxs, &minCot0, &ec0, dc->vblock);
        BtEdge* min1 = bt_findMaxAngle(mergeStamp, true,  c1, sd, rxs, sxrxs, &minCot1, &ec1, dc->vblock);

#ifdef COACD_BEAM_DEBUG
        dc->fma_total_edges += ec0 + ec1;
        if (ec0 < dc->fma_min_edges) dc->fma_min_edges = ec0;
        if (ec1 < dc->fma_min_edges) dc->fma_min_edges = ec1;
        if (ec0 > dc->fma_max_edges) dc->fma_max_edges = ec0;
        if (ec1 > dc->fma_max_edges) dc->fma_max_edges = ec1;
        dc->fma_calls += 2;
#endif

        if (!min0 && !min1) {
            BtEdge* e = bt_newEdgePair(dc, c0, c1);
            if (!e) { DPRINTF("[BUG] bt_merge OOM at final edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); return; }
            bt_edge_link(e, e);
            dc->vblock[c0].edges = e;
            e = e->reverse;
            bt_edge_link(e, e);
            dc->vblock[c1].edges = e;
            return;
        }

        int cmp = !min0 ? 1 : !min1 ? -1 : br64_cmp(minCot0, minCot1);

        if (firstRun || ((cmp >= 0) ? !br64_isNegInf(minCot1) : !br64_isNegInf(minCot0))) {
            BtEdge* e = bt_newEdgePair(dc, c0, c1);
            if (!e) { DPRINTF("[BUG] bt_merge OOM at bridge edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); return; }
            if (pendingTail0) pendingTail0->prev = e;
            else pendingHead0 = e;
            e->next = pendingTail0;
            pendingTail0 = e;

            e = e->reverse;
            if (pendingTail1) pendingTail1->next = e;
            else pendingHead1 = e;
            e->prev = pendingTail1;
            pendingTail1 = e;
        }

        BtEdge* e0 = min0;
        BtEdge* e1 = min1;

        if (cmp == 0) {
            bt_findEdgeForCoplanarFaces(mergeStamp, c0, c1, &e0, &e1, BT_VI_NULL, BT_VI_NULL, dc->vblock);
        }

        if ((cmp >= 0) && e1) {
            if (toPrev1) {
                for (BtEdge *e = toPrev1->next, *n = NULL; e != min1; e = n) {
                    n = e->next;
                    bt_removeEdgePair(dc, e);
                }
            }
            if (pendingTail1) {
                if (toPrev1) bt_edge_link(toPrev1, pendingHead1);
                else { bt_edge_link(min1->prev, pendingHead1); firstNew1 = pendingHead1; }
                bt_edge_link(pendingTail1, min1);
                pendingHead1 = NULL; pendingTail1 = NULL;
            } else if (!toPrev1) {
                firstNew1 = min1;
            }
            prevPoint = dc->vblock[c1].point;
            c1 = e1->target;
            toPrev1 = e1->reverse;
        }

        if ((cmp <= 0) && e0) {
            if (toPrev0) {
                for (BtEdge *e = toPrev0->prev, *n = NULL; e != min0; e = n) {
                    n = e->prev;
                    bt_removeEdgePair(dc, e);
                }
            }
            if (pendingTail0) {
                if (toPrev0) bt_edge_link(pendingHead0, toPrev0);
                else { bt_edge_link(pendingHead0, min0->next); firstNew0 = pendingHead0; }
                bt_edge_link(min0, pendingTail0);
                pendingHead0 = NULL; pendingTail0 = NULL;
            } else if (!toPrev0) {
                firstNew0 = min0;
            }
            prevPoint = dc->vblock[c0].point;
            c0 = e0->target;
            toPrev0 = e0->reverse;
        }

        if ((c0 == first0) && (c1 == first1)) {
            if (toPrev0 == NULL) {
                bt_edge_link(pendingHead0, pendingTail0);
                dc->vblock[c0].edges = pendingTail0;
            } else {
                for (BtEdge *e = toPrev0->prev, *n = NULL; e != firstNew0; e = n) {
                    n = e->prev;
                    bt_removeEdgePair(dc, e);
                }
                if (pendingTail0) {
                    bt_edge_link(pendingHead0, toPrev0);
                    bt_edge_link(firstNew0, pendingTail0);
                }
            }
            if (toPrev1 == NULL) {
                bt_edge_link(pendingTail1, pendingHead1);
                dc->vblock[c1].edges = pendingTail1;
            } else {
                for (BtEdge *e = toPrev1->next, *n = NULL; e != firstNew1; e = n) {
                    n = e->next;
                    bt_removeEdgePair(dc, e);
                }
                if (pendingTail1) {
                    bt_edge_link(toPrev1, pendingHead1);
                    bt_edge_link(pendingTail1, firstNew1);
                }
            }
            return;
        }
        firstRun = false;
    }
}

// bt_merge_pair — two-thread cooperative bt_merge.
//
// Both threads of a group (primary: lane&1==0, secondary: lane&1==1) enter.
// Primary owns all state (c0, c1, prevPoint, edge lists). At each loop
// iteration the primary broadcasts the bt_findMaxAngle arguments via warp
// shuffle; both threads then call bt_findMaxAngle convergedly (primary for
// the c0/h0 side, secondary for the c1/h1 side). Primary reads the secondary's
// result back with another shuffle, then does all bookkeeping.
//
// partner_lane = lane ^ 1 (the other thread in this group).
// pair_mask    = 3u << (lane & ~1u)  — only the two threads participate in shuffles.
__device__ inline void bt_merge_pair(
    BtDCState* dc, BtIntermediateHull* h0, BtIntermediateHull* h1,
    bool is_primary, int partner_lane)
{
    int my_lane     = (int)(threadIdx.x % WARP_SIZE);
    int primary_lane = is_primary ? my_lane : partner_lane;
    unsigned pair_mask = 3u << (my_lane & ~1u);

    // --- Early exit (primary decides, broadcasts to secondary) ---
    int skip = 0;
    if (is_primary) {
        if (h1->maxXy == BT_VI_NULL) skip = 1;
        else if (h0->maxXy == BT_VI_NULL) { *h0 = *h1; skip = 1; }
    }
    skip = __shfl_sync(pair_mask, skip, primary_lane);
    if (skip) return;

    // --- Primary initialises merge state ---
    BtVIndex c0 = BT_VI_NULL, c1 = BT_VI_NULL;
    BtEdge* toPrev0    = NULL; BtEdge* firstNew0   = NULL;
    BtEdge* pendingHead0 = NULL; BtEdge* pendingTail0 = NULL;
    BtEdge* toPrev1    = NULL; BtEdge* firstNew1   = NULL;
    BtEdge* pendingHead1 = NULL; BtEdge* pendingTail1 = NULL;
    BtPoint32 prevPoint;
    BtVIndex first0 = BT_VI_NULL, first1 = BT_VI_NULL;
    bool firstRun = true;
    int mergeStamp = 0;

    if (is_primary) {
        if (dc->mergeStampPtr) {
            dc->mergeStamp = atomicAdd(dc->mergeStampPtr, -1) - 1;
        } else {
            dc->mergeStamp--;
        }
        mergeStamp = dc->mergeStamp;

        if (bt_mergeProjection(h0, h1, &c0, &c1, dc->vblock)) {
            BtPoint32 sd = bp32_sub(dc->vblock[c1].point, dc->vblock[c0].point);
            BtPoint64 normal = bp32_cross(bp32(0,0,-1), sd);
            BtPoint64 t = bp32_cross64(sd, normal);
            BtEdge* e = dc->vblock[c0].edges;
            BtEdge* start0 = NULL;
            if (e) {
                do {
                    long long dot = bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c0].point), normal);
                    if ((dot == 0) && (bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c0].point), t) > 0)) {
                        if (!start0 || (bt_getOrientation(start0, e, sd, bp32(0,0,-1), dc->vblock) == BT_CLOCKWISE))
                            start0 = e;
                    }
                    e = e->next;
                } while (e != dc->vblock[c0].edges);
            }
            e = dc->vblock[c1].edges;
            BtEdge* start1 = NULL;
            if (e) {
                do {
                    long long dot = bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c1].point), normal);
                    if ((dot == 0) && (bp32_dot64(bp32_sub(dc->vblock[e->target].point, dc->vblock[c1].point), t) > 0)) {
                        if (!start1 || (bt_getOrientation(start1, e, sd, bp32(0,0,-1), dc->vblock) == BT_COUNTER_CLOCKWISE))
                            start1 = e;
                    }
                    e = e->next;
                } while (e != dc->vblock[c1].edges);
            }
            if (start0 || start1) {
                bt_findEdgeForCoplanarFaces(mergeStamp, c0, c1, &start0, &start1, BT_VI_NULL, BT_VI_NULL, dc->vblock);
                if (start0) c0 = start0->target;
                if (start1) c1 = start1->target;
            }
            prevPoint = dc->vblock[c1].point;
            prevPoint.z++;
        } else {
            prevPoint = dc->vblock[c1].point;
            prevPoint.x++;
        }
        first0 = c0; first1 = c1;
    }

    // Broadcast mergeStamp to secondary (needed by bt_findMaxAngle).
    mergeStamp = __shfl_sync(pair_mask, mergeStamp, primary_lane);

    // --- Main merge loop ---
    // Primary sets done=0 (keep going) or done!=0 (stop).
    // Secondary reads the broadcast at the top of each iteration.
    int done = 0;
    while (true) {
        // Broadcast continuation to secondary.
        int cont = __shfl_sync(pair_mask, done == 0 ? 1 : 0, primary_lane);
        if (!cont) break;

        // Primary computes per-iteration geometry; secondary will read via shuffle.
        BtPoint32 s_dir = {0,0,0,0};
        BtPoint64 rxs   = {0,0,0};
        BtPoint64 sxrxs = {0,0,0};
        BtVIndex  c1_bcast = 0;
        if (is_primary) {
            BtPoint32 sd  = bp32_sub(dc->vblock[c1].point, dc->vblock[c0].point);
            BtPoint32 r   = bp32_sub(prevPoint, dc->vblock[c0].point);
            rxs   = bp32_cross(r, sd);
            sxrxs = bp32_cross64(sd, rxs);
            s_dir = sd;
            c1_bcast = c1;
        }

        // Broadcast geometry to secondary (both threads execute every __shfl_sync).
        s_dir.x  = __shfl_sync(pair_mask, s_dir.x,  primary_lane);
        s_dir.y  = __shfl_sync(pair_mask, s_dir.y,  primary_lane);
        s_dir.z  = __shfl_sync(pair_mask, s_dir.z,  primary_lane);
        rxs.x    = shfl_ll(rxs.x,    primary_lane, pair_mask);
        rxs.y    = shfl_ll(rxs.y,    primary_lane, pair_mask);
        rxs.z    = shfl_ll(rxs.z,    primary_lane, pair_mask);
        sxrxs.x  = shfl_ll(sxrxs.x,  primary_lane, pair_mask);
        sxrxs.y  = shfl_ll(sxrxs.y,  primary_lane, pair_mask);
        sxrxs.z  = shfl_ll(sxrxs.z,  primary_lane, pair_mask);
        c1_bcast = __shfl_sync(pair_mask, c1_bcast, primary_lane);

        // Both threads call bt_findMaxAngle convergedly.
        BtVIndex my_start = is_primary ? c0 : c1_bcast;
        bool     my_ccw   = !is_primary;
        BtRational64 my_minCot;
        BtEdge* my_min = bt_findMaxAngle(mergeStamp, my_ccw, my_start,
                                          s_dir, rxs, sxrxs, &my_minCot, NULL, dc->vblock);

        // Both execute shuffles; primary reads secondary's results.
        BtEdge*      min1_shfl    = shfl_edge_ptr(my_min,    partner_lane, pair_mask);
        BtRational64 minCot1_shfl = shfl_br64(my_minCot, partner_lane, pair_mask);

        // Primary does all bookkeeping.
        if (is_primary) {
            BtEdge*      min0    = my_min;
            BtRational64 minCot0 = my_minCot;
            BtEdge*      min1    = min1_shfl;
            BtRational64 minCot1 = minCot1_shfl;

#ifdef COACD_BEAM_DEBUG
            dc->fma_calls += 2;
#endif

            if (!min0 && !min1) {
                BtEdge* e = bt_newEdgePair(dc, c0, c1);
                if (!e) { DPRINTF("[BUG] bt_merge_pair OOM at final edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); done = -1; }
                else {
                    bt_edge_link(e, e); dc->vblock[c0].edges = e;
                    e = e->reverse; bt_edge_link(e, e); dc->vblock[c1].edges = e;
                    done = 1;
                }
            } else {
                int cmp = !min0 ? 1 : !min1 ? -1 : br64_cmp(minCot0, minCot1);

                if (firstRun || ((cmp >= 0) ? !br64_isNegInf(minCot1) : !br64_isNegInf(minCot0))) {
                    BtEdge* e = bt_newEdgePair(dc, c0, c1);
                    if (!e) { DPRINTF("[BUG] bt_merge_pair OOM at bridge edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); done = -1; }
                    else {
                        if (pendingTail0) pendingTail0->prev = e;
                        else pendingHead0 = e;
                        e->next = pendingTail0; pendingTail0 = e;
                        e = e->reverse;
                        if (pendingTail1) pendingTail1->next = e;
                        else pendingHead1 = e;
                        e->prev = pendingTail1; pendingTail1 = e;
                    }
                }

                if (done == 0) {
                    BtEdge* e0 = min0;
                    BtEdge* e1 = min1;
                    if (cmp == 0) {
                        bt_findEdgeForCoplanarFaces(mergeStamp, c0, c1, &e0, &e1, BT_VI_NULL, BT_VI_NULL, dc->vblock);
                    }

                    if ((cmp >= 0) && e1) {
                        if (toPrev1) {
                            for (BtEdge *e = toPrev1->next, *n = NULL; e != min1; e = n) {
                                n = e->next; bt_removeEdgePair(dc, e);
                            }
                        }
                        if (pendingTail1) {
                            if (toPrev1) bt_edge_link(toPrev1, pendingHead1);
                            else { bt_edge_link(min1->prev, pendingHead1); firstNew1 = pendingHead1; }
                            bt_edge_link(pendingTail1, min1);
                            pendingHead1 = NULL; pendingTail1 = NULL;
                        } else if (!toPrev1) {
                            firstNew1 = min1;
                        }
                        prevPoint = dc->vblock[c1].point;
                        c1 = e1->target;
                        toPrev1 = e1->reverse;
                    }

                    if ((cmp <= 0) && e0) {
                        if (toPrev0) {
                            for (BtEdge *e = toPrev0->prev, *n = NULL; e != min0; e = n) {
                                n = e->prev; bt_removeEdgePair(dc, e);
                            }
                        }
                        if (pendingTail0) {
                            if (toPrev0) bt_edge_link(pendingHead0, toPrev0);
                            else { bt_edge_link(pendingHead0, min0->next); firstNew0 = pendingHead0; }
                            bt_edge_link(min0, pendingTail0);
                            pendingHead0 = NULL; pendingTail0 = NULL;
                        } else if (!toPrev0) {
                            firstNew0 = min0;
                        }
                        prevPoint = dc->vblock[c0].point;
                        c0 = e0->target;
                        toPrev0 = e0->reverse;
                    }

                    if ((c0 == first0) && (c1 == first1)) {
                        if (toPrev0 == NULL) {
                            bt_edge_link(pendingHead0, pendingTail0);
                            dc->vblock[c0].edges = pendingTail0;
                        } else {
                            for (BtEdge *e = toPrev0->prev, *n = NULL; e != firstNew0; e = n) {
                                n = e->prev; bt_removeEdgePair(dc, e);
                            }
                            if (pendingTail0) {
                                bt_edge_link(pendingHead0, toPrev0);
                                bt_edge_link(firstNew0, pendingTail0);
                            }
                        }
                        if (toPrev1 == NULL) {
                            bt_edge_link(pendingTail1, pendingHead1);
                            dc->vblock[c1].edges = pendingTail1;
                        } else {
                            for (BtEdge *e = toPrev1->next, *n = NULL; e != firstNew1; e = n) {
                                n = e->next; bt_removeEdgePair(dc, e);
                            }
                            if (pendingTail1) {
                                bt_edge_link(toPrev1, pendingHead1);
                                bt_edge_link(pendingTail1, firstNew1);
                            }
                        }
                        done = 1;
                    }
                    firstRun = false;
                }
            }
        }
    }
}

// ============================================================================
// computeInternal base case — handles n <= 2
// ============================================================================

__device__ inline void bt_computeBase(BtDCState* __restrict__ dc, int start, int end, BtIntermediateHull* result) {
    int n = end - start;
    switch (n) {
    case 0:
        result->minXy = BT_VI_NULL; result->maxXy = BT_VI_NULL;
        result->minYx = BT_VI_NULL; result->maxYx = BT_VI_NULL;
        return;
    case 2: {
        BtVIndex v = start;
        BtVIndex w = start + 1;
        if (bp32_ne(dc->vblock[v].point, dc->vblock[w].point)) {
            int dx = dc->vblock[v].point.x - dc->vblock[w].point.x;
            int dy = dc->vblock[v].point.y - dc->vblock[w].point.y;
            if ((dx == 0) && (dy == 0)) {
                if (dc->vblock[v].point.z > dc->vblock[w].point.z) { BtVIndex t = w; w = v; v = t; }
                dc->vblock[v].next = v; dc->vblock[v].prev = v;
                result->minXy = v; result->maxXy = v;
                result->minYx = v; result->maxYx = v;
            } else {
                dc->vblock[v].next = w; dc->vblock[v].prev = w; dc->vblock[w].next = v; dc->vblock[w].prev = v;
                if ((dx < 0) || ((dx == 0) && (dy < 0))) {
                    result->minXy = v; result->maxXy = w;
                } else {
                    result->minXy = w; result->maxXy = v;
                }
                if ((dy < 0) || ((dy == 0) && (dx < 0))) {
                    result->minYx = v; result->maxYx = w;
                } else {
                    result->minYx = w; result->maxYx = v;
                }
            }
            BtEdge* e = bt_newEdgePair(dc, v, w);
            if (!e) { DPRINTF("[BUG] bt_merge OOM at coplanar edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); return; }
            bt_edge_link(e, e); dc->vblock[v].edges = e;
            e = e->reverse;
            bt_edge_link(e, e); dc->vblock[w].edges = e;
            return;
        }
    }
    // fallthrough
    case 1: {
        BtVIndex v = start;
        dc->vblock[v].edges = NULL;
        dc->vblock[v].next = v; dc->vblock[v].prev = v;
        result->minXy = v; result->maxXy = v;
        result->minYx = v; result->maxYx = v;
        return;
    }
    }
}

// ============================================================================
// computeInternal — iterative D&C with explicit local stack (single-thread)
// ============================================================================

#define BT_ERR_DC_STACK     4

__device__ inline void bt_computeInternal(BtDCState* __restrict__ dc, int start, int end, BtIntermediateHull* result,
                                          CheckedBuf<BtDCStackItem> stack) {
    int sp = 0;

    // Push root
    stack[sp].start = start;
    stack[sp].end = end;
    stack[sp].stage = 0;
    stack[sp].right_hull.minXy = BT_VI_NULL; stack[sp].right_hull.maxXy = BT_VI_NULL;
    stack[sp].right_hull.minYx = BT_VI_NULL; stack[sp].right_hull.maxYx = BT_VI_NULL;
    stack[sp].result = result;
    sp++;

    while (sp > 0) {
        sp--;
        BtDCStackItem* item = &stack[sp];

        if (item->stage == 0) {
            int n = item->end - item->start;
            if (n <= 2) {
                bt_computeBase(dc, item->start, item->end, item->result);
            } else {
                int split0 = item->start + n / 2;
                BtPoint32 p = dc->vblock[split0 - 1].point;
                int split1 = split0;
                while ((split1 < item->end) && bp32_eq(dc->vblock[split1].point, p)) split1++;

                BtIntermediateHull* res = item->result;

                // Convert current item to merge (stage 1)
                item->stage = 1;
                item->right_hull.minXy = BT_VI_NULL; item->right_hull.maxXy = BT_VI_NULL;
                item->right_hull.minYx = BT_VI_NULL; item->right_hull.maxYx = BT_VI_NULL;
                item->result = res;
                sp++;  // re-push (it's already in place)
                BtDCStackItem* merge_ptr = &stack[sp - 1];

                // Push right child (result → merge parent's right_hull)
                if (sp >= BT_DC_MAX_STACK_LOCAL) { dc->edgePool.error = BT_ERR_DC_STACK; return; }
                stack[sp].start = split1;
                stack[sp].end = item->end;
                stack[sp].stage = 0;
                stack[sp].right_hull.minXy = BT_VI_NULL; stack[sp].right_hull.maxXy = BT_VI_NULL;
                stack[sp].right_hull.minYx = BT_VI_NULL; stack[sp].right_hull.maxYx = BT_VI_NULL;
                stack[sp].result = &merge_ptr->right_hull;
                sp++;

                // Push left child (result → merge parent's result directly)
                if (sp >= BT_DC_MAX_STACK_LOCAL) { dc->edgePool.error = BT_ERR_DC_STACK; return; }
                stack[sp].start = item->start;
                stack[sp].end = split0;
                stack[sp].stage = 0;
                stack[sp].right_hull.minXy = BT_VI_NULL; stack[sp].right_hull.maxXy = BT_VI_NULL;
                stack[sp].right_hull.minYx = BT_VI_NULL; stack[sp].right_hull.maxYx = BT_VI_NULL;
                stack[sp].result = merge_ptr->result;
                sp++;
            }
        } else {
            bt_merge(dc, item->result, &item->right_hull);
        }
    }
}

// ============================================================================
// Point sorting
// ============================================================================

struct BtPointCmp {
    __device__ inline bool operator()(const BtPoint32& p, const BtPoint32& q) const {
        return (p.y < q.y) || ((p.y == q.y) && ((p.x < q.x) || ((p.x == q.x) && (p.z < q.z))));
    }
};

// Sort is handled by warp_sort.cuh (warp_sort_bp32) — no CUB dependency.

// ============================================================================
// Mesh extraction from half-edge hull (lane 0 only)
// ============================================================================

// bt_extractMesh: BFS over the half-edge graph, assigns sequential indices to
// vertices, converts BtPoint32 back to float world space, and fan-triangulates
// each face. Lane 0 only.
//
// When out_verts == NULL: count-only mode — counts nv and nt without writing.
// When out_verts != NULL: extract mode — writes vertices and triangles.
// Callers rewind the WarpPool offset between count and extract passes.
//
// Returns 0 on success, -1 on pool OOM.
__device__ inline int bt_extractMesh(BtHullState* s,
    float* out_verts, int* out_tris,
    int* n_verts_out, int* n_tris_out)
{
    *n_verts_out = 0;
    *n_tris_out  = 0;

    if (s->vertexList == BT_VI_NULL) return 0;

    BtVIndex* queue = (BtVIndex*)bt_alloc(s->wp, s->npoints * (int)sizeof(BtVIndex));
    if (!queue) return -1;
    int n_verts = 0;
    int n_tris  = 0;

    int vstamp = --s->mergeStamp;
    s->vblock[s->vertexList].copy = vstamp;
    queue[n_verts++] = s->vertexList;

    // BFS: discover all hull vertices
    int qhead = 0;
    while (qhead < n_verts) {
        BtVIndex v = queue[qhead++];
        BtEdge* e = s->vblock[v].edges;
        if (!e) continue;
        do {
            if (s->vblock[e->target].copy != vstamp) {
                s->vblock[e->target].copy = vstamp;
                queue[n_verts++] = e->target;
            }
            e = e->next;
        } while (e != s->vblock[v].edges);
    }

    if (out_verts) {
        // Extract mode: encode vertex indices, write verts, fan-triangulate faces.
        int idx_base = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++)
            s->vblock[queue[i]].copy = idx_base - i;

        int medAx = s->medAxis, maxAx = s->maxAxis, minAx = s->minAxis;
        float sc_med = s->scaling[medAx], sc_max = s->scaling[maxAx], sc_min = s->scaling[minAx];
        float cn_med = s->center[medAx],  cn_max = s->center[maxAx],  cn_min = s->center[minAx];
        for (int i = 0; i < n_verts; i++) {
            BtVIndex v = queue[i];
            out_verts[i * 3 + medAx] = (float)s->vblock[v].point.x * sc_med + cn_med;
            out_verts[i * 3 + maxAx] = (float)s->vblock[v].point.y * sc_max + cn_max;
            out_verts[i * 3 + minAx] = (float)s->vblock[v].point.z * sc_min + cn_min;
        }

        int fstamp = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++) {
            BtVIndex v = queue[i];
            BtEdge* e = s->vblock[v].edges;
            if (!e) continue;
            do {
                if (e->copy != fstamp) {
                    BtVIndex a = BT_VI_NULL, b = BT_VI_NULL;
                    BtEdge* f = e;
                    do {
                        if (a != BT_VI_NULL && b != BT_VI_NULL) {
                            out_tris[n_tris * 3 + 0] = idx_base - s->vblock[v].copy;
                            out_tris[n_tris * 3 + 1] = idx_base - s->vblock[a].copy;
                            out_tris[n_tris * 3 + 2] = idx_base - s->vblock[b].copy;
                            n_tris++;
                        }
                        f->copy = fstamp;
                        a = b;
                        b = f->target;
                        f = f->reverse->prev;
                    } while (f != e);
                }
                e = e->next;
            } while (e != s->vblock[v].edges);
        }
    } else {
        // Count-only mode: count triangles via fan formula (k edges → k-2 tris).
        int fstamp = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++) {
            BtVIndex v = queue[i];
            BtEdge* e = s->vblock[v].edges;
            if (!e) continue;
            do {
                if (e->copy != fstamp) {
                    int k = 0;
                    BtEdge* f = e;
                    do { f->copy = fstamp; k++; f = f->reverse->prev; } while (f != e);
                    if (k >= 3) n_tris += k - 2;
                }
                e = e->next;
            } while (e != s->vblock[v].edges);
        }
    }

    *n_verts_out = n_verts;
    *n_tris_out  = n_tris;
    return 0;
}

// ============================================================================
// Main entry: compute hull and return volume
// ============================================================================

// Pre-sort phase (lane 0): AABB, scaling, fill Point32 array, allocate pool memory.
// Returns pointer to the unsorted BtPoint32 array (in pool), or NULL on error.
__device__ inline BtPoint32* bt_compute_presort(BtHullState* s, const float* pts, int count, int lane) {
    // --- AABB: warp-parallel strided reduction ---
    float mn0 = 1e30f, mn1 = 1e30f, mn2 = 1e30f;
    float mx0 = -1e30f, mx1 = -1e30f, mx2 = -1e30f;
    for (int i = lane; i < count; i += WARP_SIZE) {
        float v0 = pts[i * 3 + 0];
        float v1 = pts[i * 3 + 1];
        float v2 = pts[i * 3 + 2];
        mn0 = fminf(mn0, v0); mx0 = fmaxf(mx0, v0);
        mn1 = fminf(mn1, v1); mx1 = fmaxf(mx1, v1);
        mn2 = fminf(mn2, v2); mx2 = fmaxf(mx2, v2);
    }
    mn0 = warp_min_f(mn0); mn1 = warp_min_f(mn1); mn2 = warp_min_f(mn2);
    mx0 = warp_max_f(mx0); mx1 = warp_max_f(mx1); mx2 = warp_max_f(mx2);

    // --- Axis/scaling: scalar logic, all lanes compute identically ---
    float span[3] = {mx0-mn0, mx1-mn1, mx2-mn2};
    int maxAx = (span[0] >= span[1]) ? ((span[0] >= span[2]) ? 0 : 2) : ((span[1] >= span[2]) ? 1 : 2);
    int minAx = (span[0] <= span[1]) ? ((span[0] <= span[2]) ? 0 : 2) : ((span[1] <= span[2]) ? 1 : 2);
    if (minAx == maxAx) minAx = (maxAx + 1) % 3;
    int medAx = 3 - maxAx - minAx;

    float sc[3];
    sc[0] = span[0] / 10216.0f;
    sc[1] = span[1] / 10216.0f;
    sc[2] = span[2] / 10216.0f;
    if (((medAx + 1) % 3) != maxAx) {
        sc[0] = -sc[0]; sc[1] = -sc[1]; sc[2] = -sc[2];
    }

    // --- Lane 0: write state and allocate ---
    BtPoint32* points = NULL;
    if (lane == 0) {
        s->maxAxis = maxAx;
        s->minAxis = minAx;
        s->medAxis = medAx;
        s->scaling[0] = sc[0]; s->scaling[1] = sc[1]; s->scaling[2] = sc[2];
        s->center[0] = (mn0+mx0)*0.5f; s->center[1] = (mn1+mx1)*0.5f; s->center[2] = (mn2+mx2)*0.5f;
        points = (BtPoint32*)bt_alloc(s->wp, count * (int)sizeof(BtPoint32));
    }
    // Broadcast pointer from lane 0; syncwarp ensures s->center/scaling visible to all lanes
    long long pp = __shfl_sync(WARP_MASK, (long long)points, 0);
    points = (BtPoint32*)pp;
    if (!points) return NULL;
    __syncwarp();

    float cen_med = s->center[medAx], cen_max = s->center[maxAx], cen_min = s->center[minAx];
    float inv_med = (s->scaling[medAx] != 0.0f) ? 1.0f / s->scaling[medAx] : 0.0f;
    float inv_max = (s->scaling[maxAx] != 0.0f) ? 1.0f / s->scaling[maxAx] : 0.0f;
    float inv_min = (s->scaling[minAx] != 0.0f) ? 1.0f / s->scaling[minAx] : 0.0f;

    // --- Point conversion: warp-parallel strided ---
    for (int i = lane; i < count; i += WARP_SIZE) {
        points[i].x = (int)((pts[i*3+medAx] - cen_med) * inv_med);
        points[i].y = (int)((pts[i*3+maxAx] - cen_max) * inv_max);
        points[i].z = (int)((pts[i*3+minAx] - cen_min) * inv_min);
        points[i].index = i;
    }
    __syncwarp();
    return points;
}

// Post-sort: init vertices (all lanes), parallel D&C + tree merge.
// 1. Split sorted points into 32 groups (one per warp lane).
// 2. Each lane independently builds a small hull of its group.
// Per-lane edge pool cleanup info saved for deferred freeing after extractMesh.
struct BtLanePoolCleanup {
    int    nblocks;
    void** blocks;  // points into WarpPool-allocated array, capacity BTPOOL_MAX_BLOCKS
};

// 3. Tree merge: 5 rounds (16×2 → 8×4 → … → 1×32), each round's
//    independent merges run in parallel across warp lanes.
// out_cleanup: __shared__ BtLanePoolCleanup[WARP_SIZE] — each lane saves its pool blocks here.
__device__ inline void bt_compute_postsort(BtHullState* s, BtPoint32* points, int count, int lane,
                                           BtLanePoolCleanup* out_cleanup) {
    // --- Broadcast shared state from lane 0 (only lane 0's s is valid) ---
    float sc0 = 0, sc1 = 0, sc2 = 0, cn0 = 0, cn1 = 0, cn2 = 0;
    int minAx = 0, medAx = 0, maxAx = 0;
    long long wp_ll = 0, sh_ll = 0;
    if (lane == 0) {
        sc0 = s->scaling[0]; sc1 = s->scaling[1]; sc2 = s->scaling[2];
        cn0 = s->center[0];  cn1 = s->center[1];  cn2 = s->center[2];
        minAx = s->minAxis; medAx = s->medAxis; maxAx = s->maxAxis;
        wp_ll = (long long)s->wp;
        sh_ll = (long long)s->scratch_heap;
    }
    sc0 = __shfl_sync(WARP_MASK, sc0, 0); sc1 = __shfl_sync(WARP_MASK, sc1, 0); sc2 = __shfl_sync(WARP_MASK, sc2, 0);
    cn0 = __shfl_sync(WARP_MASK, cn0, 0); cn1 = __shfl_sync(WARP_MASK, cn1, 0); cn2 = __shfl_sync(WARP_MASK, cn2, 0);
    minAx = __shfl_sync(WARP_MASK, minAx, 0); medAx = __shfl_sync(WARP_MASK, medAx, 0); maxAx = __shfl_sync(WARP_MASK, maxAx, 0);
    wp_ll = __shfl_sync(WARP_MASK, wp_ll, 0); sh_ll = __shfl_sync(WARP_MASK, sh_ll, 0);
    WarpPool* shared_wp = (WarpPool*)wp_ll;
    DeviceHeap* shared_sh = (DeviceHeap*)sh_ll;

    // Lane 0: allocate vblock + per-lane pool blocks
    BtVertex* vblock = NULL;
    void** all_pool_blocks = NULL;
    if (lane == 0) {
        vblock = (BtVertex*)bt_alloc(shared_wp, count * (int)sizeof(BtVertex));
        all_pool_blocks = (void**)bt_alloc(shared_wp, WARP_SIZE * BTPOOL_MAX_BLOCKS * (int)sizeof(void*));
    }
    long long vb = __shfl_sync(WARP_MASK, (long long)vblock, 0);
    vblock = (BtVertex*)vb;
    if (!vblock) return;
    { long long pb = __shfl_sync(WARP_MASK, (long long)all_pool_blocks, 0);
      all_pool_blocks = (void**)pb; }
    if (!all_pool_blocks) return;

    // All lanes: init vertices in parallel
    for (int i = lane; i < count; i += WARP_SIZE) {
        vblock[i].edges = NULL;
        vblock[i].next = BT_VI_NULL; vblock[i].prev = BT_VI_NULL;
        vblock[i].point = points[i];
        vblock[i].copy = -1;
    }
    __syncwarp();

    if (lane == 0) s->npoints = count;

    __shared__ BtIntermediateHull s_hulls[BT_HULL_GROUPS];
    __shared__ int   s_mergeStamp;

    if (lane == 0) s_mergeStamp = -3;

    // --- Per-group boundaries (BT_HULL_GROUPS groups; 2 lanes share each group) ---
    // Like serial D&C's split logic, advance each split past runs of
    // equal points so that identical vertices are never split across
    // two groups. Without this, degenerate sub-hulls at group boundaries
    // trigger edge cases in bt_merge (pending chain use-after-free).
    __shared__ int s_splits[BT_HULL_GROUPS + 1];
    if (lane == 0) {
        s_splits[0] = 0;
        for (int g = 1; g < BT_HULL_GROUPS; g++) {
            int split = g * count / BT_HULL_GROUPS;
            // Advance past equal points (same logic as bt_computeInternal)
            BtPoint32 p = vblock[split - 1].point;
            while (split < count && bp32_eq(vblock[split].point, p)) split++;
            s_splits[g] = split;
        }
        s_splits[BT_HULL_GROUPS] = count;
    }
    __syncwarp();
    // group = lane / 2; is_primary = (lane & 1) == 0
    int group      = lane / 2;
    bool is_primary = (lane & 1) == 0;
    int partner_lane = lane ^ 1;
    int my_start = s_splits[group];
    int my_end   = s_splits[group + 1];

    // --- Per-lane D&C state (lightweight: only edgePool + mergeStamp) ---
    // Common fields (scaling, center, axes, wp, etc.) stay in shared BtHullState *s.
    BtDCState my_dc;
    my_dc.mergeStamp = -3;
    my_dc.mergeStampPtr = &s_mergeStamp;
#ifdef TRACK_MAX_EDGE_PAIRS
    my_dc.usedEdgePairs = 0;
    my_dc.maxEdgePairs = 0;
#endif
#ifdef COACD_BEAM_DEBUG
    my_dc.fma_total_edges = 0;
    my_dc.fma_min_edges   = 0x7fffffff;
    my_dc.fma_max_edges   = 0;
    my_dc.fma_calls       = 0;
#endif

    // Lane 0 allocates 2 slabs per lane (64 total) from scratch_heap.
    // Each lane then builds its own BtPool from the two pre-allocated slabs.
    // Growth slabs are allocated dynamically per-lane via heap_alloc during D&C.
    __shared__ void* s_edge_slabs[WARP_SIZE * 2];
    {
        int slab_edges = 3 * count;
        if (slab_edges > BTPOOL_BLOCK_SIZE) slab_edges = BTPOOL_BLOCK_SIZE;
        int padded = ((int)sizeof(BtEdge) + 3) & ~3;
        int slab_bytes = slab_edges * padded;
        if (lane == 0) {
            for (int g = 0; g < WARP_SIZE * 2; g++) {
                void* blk = NULL;
                if (heap_alloc(shared_sh, (unsigned int)slab_bytes, &blk) != HEAP_OK) {
                    shared_wp->error = BT_ERR_POOL_EXHAUST;
                    break;
                }
                s_edge_slabs[g] = blk;
            }
        }
        __syncwarp();
        if (shared_wp->error) return;

        // Each lane initialises its BtPool from two pre-allocated slabs.
        // Per-lane blocks array from the WarpPool-allocated flat buffer.
        my_dc.edgePool.blocks = CheckedBuf<void*>(
            all_pool_blocks + lane * BTPOOL_MAX_BLOCKS, BTPOOL_MAX_BLOCKS, "edgePool.blocks");
        my_dc.edgePool.scratch_heap = shared_sh;
        my_dc.edgePool.objSize      = padded;
        my_dc.edgePool.freeList     = NULL;
        my_dc.edgePool.error        = 0;
        my_dc.edgePool.nblocks      = 0;
        my_dc.edgePool.growSlabSize = slab_edges;
        // Build free list from both slabs (slab1 -> slab0 -> NULL)
        for (int si = 0; si < 2; si++) {
            void* blk = s_edge_slabs[lane * 2 + si];
            my_dc.edgePool.blocks[my_dc.edgePool.nblocks++] = blk;
            char* b = (char*)blk;
            *(void**)b = my_dc.edgePool.freeList;
            for (int i = 1; i < slab_edges; i++)
                *(void**)(b + i * padded) = b + (i - 1) * padded;
            my_dc.edgePool.freeList = b + (slab_edges - 1) * padded;
        }
    }
    __syncwarp();

    //if (lane == 0) DPRINTF("[postsort] blk=%d n=%d arena=%d\n", blockIdx.x, count, arena_bytes);

    // --- Allocate per-lane D&C stacks from WarpPool (lane 0), rewind after D&C ---
    BtDCStackItem* all_dc_stacks = NULL;
    int pre_dc_offset = 0;
    if (lane == 0) {
        pre_dc_offset = shared_wp->offset;
        all_dc_stacks = (BtDCStackItem*)bt_alloc(shared_wp,
            WARP_SIZE * BT_DC_MAX_STACK_LOCAL * (int)sizeof(BtDCStackItem));
    }
    { long long ds = __shfl_sync(WARP_MASK, (long long)all_dc_stacks, 0);
      all_dc_stacks = (BtDCStackItem*)ds; }
    pre_dc_offset = __shfl_sync(WARP_MASK, pre_dc_offset, 0);
    if (!all_dc_stacks) return;
    CheckedBuf<BtDCStackItem> my_dc_stack(
        all_dc_stacks + lane * BT_DC_MAX_STACK_LOCAL, BT_DC_MAX_STACK_LOCAL, "dc_stack");

#ifdef BT_SERIAL_MERGE
    // --- Serial D&C: only lane 0 builds full hull (for debugging) ---
    s_hulls[0].minXy = BT_VI_NULL; s_hulls[0].maxXy = BT_VI_NULL;
    s_hulls[0].minYx = BT_VI_NULL; s_hulls[0].maxYx = BT_VI_NULL;
    if (lane == 0 && count > 0) {
        my_dc.vblock = vblock;
        bt_computeInternal(&my_dc, 0, count, &s_hulls[0], my_dc_stack);
    }
    __syncwarp();
    {
        int my_err = my_dc.edgePool.error;
        for (int off = 16; off > 0; off >>= 1)
            my_err |= __shfl_xor_sync(WARP_MASK, my_err, off);
        if (my_err) { shared_wp->error = my_err; return; }
    }
#else
    // --- Phase 1: Group primaries build BT_HULL_GROUPS sub-hulls in parallel ---
    // Secondary lanes (lane&1==1) are idle here; they will participate in Phase 2.
    my_dc.vblock = vblock;  // All lanes need vblock for Phase 2 bt_findMaxAngle calls.
    if (is_primary) {
        s_hulls[group].minXy = BT_VI_NULL; s_hulls[group].maxXy = BT_VI_NULL;
        s_hulls[group].minYx = BT_VI_NULL; s_hulls[group].maxYx = BT_VI_NULL;
        if (my_end - my_start > 0) {
            bt_computeInternal(&my_dc, my_start, my_end, &s_hulls[group], my_dc_stack);
        }
    }
    __syncwarp();

    // Check for errors from any lane — propagate actual error bits
    {
        int my_err = my_dc.edgePool.error;
        // OR all lanes' errors together via warp reduction
        for (int off = 16; off > 0; off >>= 1)
            my_err |= __shfl_xor_sync(WARP_MASK, my_err, off);
        if (my_err) { shared_wp->error = my_err; return; }
    }

    // --- Phase 2: Tree merge (4 rounds, 2-thread cooperative) ---
    // Round r: BT_HULL_GROUPS>>>(r+1) independent merges, each handled by
    // a pair (2g, 2g+1). Group g merges s_hulls[g*stride] + s_hulls[g*stride+stride/2].
    // Both threads in the pair call bt_findMaxAngle convergedly each iteration.
    for (int round = 0; round < 4; round++) {
        int stride      = 1 << (round + 1);               // 2, 4, 8, 16
        int merge_count = BT_HULL_GROUPS >> (round + 1);  // 8, 4, 2, 1
        if (group < merge_count) {
            int left  = group * stride;
            int right = left + (stride >> 1);
            bt_merge_pair(&my_dc, &s_hulls[left], &s_hulls[right],
                          is_primary, partner_lane);
        }
        __syncwarp();

        // Check for errors (pool exhaustion during merge)
        {
            int my_err = my_dc.edgePool.error;
            for (int off = 16; off > 0; off >>= 1)
                my_err |= __shfl_xor_sync(WARP_MASK, my_err, off);
            if (my_err) { shared_wp->error = my_err; return; }
        }
    }
#endif

    // Rewind D&C stacks (no longer needed after tree merge)
    if (lane == 0) bt_rewind(shared_wp, pre_dc_offset);
    __syncwarp();

    // --- Result is in s_hulls[0] ---
    if (lane == 0) {
        s->vertexList = s_hulls[0].minXy;
        s->vblock = vblock;
        // Set mergeStamp from shared counter for use by extractMesh
        s->mergeStamp = s_mergeStamp;
        s->mergeStampPtr = NULL;  // extractMesh uses local stamp, not shared
#ifdef COACD_BEAM_DEBUG
        // Aggregate instrumentation from lane 0 (other lanes' stats are lost)
        s->fma_total_edges = my_dc.fma_total_edges;
        s->fma_min_edges   = my_dc.fma_min_edges;
        s->fma_max_edges   = my_dc.fma_max_edges;
        s->fma_calls       = my_dc.fma_calls;
#endif
    }

    // Each lane's edgePool.blocks already points into all_pool_blocks; just record
    // the pointer and count for deferred cleanup in hull_dandc_warp_mesh.
    out_cleanup[lane].blocks = all_pool_blocks + lane * BTPOOL_MAX_BLOCKS;
    out_cleanup[lane].nblocks = my_dc.edgePool.nblocks;
    __syncwarp();
}

// ============================================================================
// Warp entry point: extract hull mesh into heap-allocated Mesh
// ============================================================================

// hull_dandc_warp_mesh: D&C convex hull mesh extraction.
// All 32 lanes must call with identical arguments.
//
// heap         — output heap: one chunk allocated for [verts | tris].
// scratch_heap — scratch heap: WarpPool backing + edge pool slabs, all freed on return.
//
// Returns a Mesh with verts/tris in heap. Returns {NULL,NULL,0,0} on error or n<4.
// *err is set to a nonzero error code on failure (all lanes see the same value).
__device__ inline Mesh hull_dandc_warp_mesh(
    const float* pts, int n, int lane,
    DeviceHeap* heap, DeviceHeap* scratch_heap,
    int* err)
{
    __shared__ WarpPool           s_pool;
    __shared__ void*              s_pool_backing;
    __shared__ Mesh               s_result;
    __shared__ BtLanePoolCleanup  s_lane_cleanup[WARP_SIZE];
    __shared__ BtHullState        s_state;

    // Use a local error variable to avoid racing on *err with other blocks.
    // Only lane 0 writes; atomicOr to *err at the end.
    int local_err = 0;
    s_lane_cleanup[lane].nblocks = 0;
    if (lane == 0) { s_result.verts = NULL; s_result.tris = NULL;
                     s_result.nv = 0; s_result.nt = 0; s_result.refcount = NULL; }

    if (n < 4) { __syncwarp(); return s_result; }

    // --- Allocate WarpPool backing from scratch_heap (lane 0) ---
    if (lane == 0) {
        int sz = dandc_scratch_bytes(n);
        void* bk = NULL;
        if (heap_alloc(scratch_heap, (unsigned int)sz, &bk) == HEAP_OK) {
            s_pool_backing  = bk;
            s_pool.base     = (char*)bk;
            s_pool.offset   = 0;
            s_pool.capacity = sz;
            s_pool.error    = 0;
        } else {
            s_pool_backing = NULL;
            local_err = BT_ERR_HEAP_TO_WARP;
        }
    }
    __syncwarp();
    if (!s_pool_backing) {
        local_err = __shfl_sync(WARP_MASK, local_err, 0);
        if (lane == 0 && local_err) atomicOr(err, local_err);
        return s_result;
    }

    // --- Phase 1: pre-sort (all lanes) ---
    // s_state is __shared__ to avoid 32 copies in local memory (only lane 0 uses it).
    BtHullState* state = &s_state;
    if (lane == 0) {
        state->wp           = &s_pool;
        state->scratch_heap = scratch_heap;
        state->vertexList   = BT_VI_NULL;
        state->mergeStampPtr = NULL;
#ifdef COACD_BEAM_DEBUG
        state->fma_total_edges = 0;
        state->fma_min_edges   = 0x7fffffff;
        state->fma_max_edges   = 0;
        state->fma_calls       = 0;
#endif
    }
    __syncwarp();

    BtPoint32* points = bt_compute_presort(state, pts, n, lane);
    if (!points) { local_err = 1; goto done; }

    {
        // --- Phase 2: sort (all lanes) ---
        int pre_sort_offset = s_pool.offset;
        char* sort_scratch = NULL;
        if (lane == 0) {
            int sb = n * (int)sizeof(BtPoint32) + WS_MAX_STACK * 2 * (int)sizeof(int);
            sort_scratch = (char*)bt_alloc(&s_pool, sb);
        }
        { long long sp = __shfl_sync(WARP_MASK, (long long)sort_scratch, 0);
          sort_scratch = (char*)sp; }
        if (!sort_scratch) { local_err = 1; goto done; }

        int sort_err = warp_sort_bp32(points, sort_scratch, n, lane);
        __syncwarp();
        if (sort_err) { local_err = BT_ERR_SORT_STACK; goto done; }

        if (lane == 0) bt_rewind(&s_pool, pre_sort_offset);
        __syncwarp();

        // --- Phase 3: post-sort D&C (vertex init: all lanes, D&C + edgePool init: lane 0) ---
        bt_compute_postsort(state, points, n, lane, s_lane_cleanup);

        if (lane == 0) {
            if (s_pool.error) {
                local_err = s_pool.error;
                goto done;
            }

            DPRINTF("[hull] n=%d fma_calls=%d edges: total=%d avg=%.1f min=%d max=%d\n",
                n, state->fma_calls, state->fma_total_edges,
                state->fma_calls > 0 ? (float)state->fma_total_edges / state->fma_calls : 0.f,
                state->fma_min_edges == 0x7fffffff ? 0 : state->fma_min_edges,
                state->fma_max_edges);

            // Count pass: exact nv and nt without writing output
            int pre_count = s_pool.offset;
            int nv = 0, nt = 0;
            if (bt_extractMesh(state, NULL, NULL, &nv, &nt) < 0)
                { local_err = 6; goto done; }
            bt_rewind(&s_pool, pre_count);

            // Allocate output Mesh chunk from heap: [verts (16-byte aligned) | tris (16-byte aligned) | refcount]
            if (nv > 0) {
                size_t vb = (size_t)nv * 3 * sizeof(float);
                size_t va = (vb + 15) & ~(size_t)15;
                size_t tb = (size_t)nt * 3 * sizeof(int);
                size_t ta = (tb + 15) & ~(size_t)15;
                size_t rb = 16;
                void* chunk = NULL;
                if (heap_alloc(heap, (unsigned int)(va + ta + rb), &chunk) != HEAP_OK)
                    { local_err = BT_ERR_HEAP_OUTPUT; goto done; }
                float* ov = (float*)chunk;
                int*   ot = (int*)((char*)chunk + va);
                int*   rc = (int*)((char*)chunk + va + ta);
                *rc = 1;

                // Extract pass
                int pre_ext = s_pool.offset;
                int nv2 = 0, nt2 = 0;
                if (bt_extractMesh(state, ov, ot, &nv2, &nt2) < 0)
                    { local_err = 6; goto done; }
                bt_rewind(&s_pool, pre_ext);

                s_result.verts = ov; s_result.tris = ot;
                s_result.nv    = nv; s_result.nt   = nt;
                s_result.refcount = rc;
            }
        }
    }

done:
    // Cleanup scratch: free all lane edge pool blocks first (their pointers are stored
    // in WarpPool-backed arrays), then free WarpPool backing itself.
    if (lane == 0) {
        for (int g = 0; g < WARP_SIZE; g++) {
            BtLanePoolCleanup* c = &s_lane_cleanup[g];
            for (int i = 0; i < c->nblocks; i++)
                heap_free(scratch_heap, c->blocks[i]);
        }
        heap_free(scratch_heap, s_pool_backing);
    }
    __syncwarp();

    // Publish local error to the global error word (visible to host / other blocks).
    local_err = __shfl_sync(WARP_MASK, local_err, 0);
    if (lane == 0 && local_err) atomicOr(err, local_err);
    return s_result;
}
