// hull_dandc.cuh — Warp-based D&C convex hull volume and mesh extraction.
//
// Faithful port of btConvexHullComputer by Ole Kniemeyer (MAXON, zlib license).
// Parallel tree merge: points split into 32 groups, each lane builds a small
// hull independently, then 5 rounds of pairwise merging (16×2 → 8 → 4 → 2 → 1).
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

struct BtPointR128 {
    BtInt128 x, y, z, den;
};

__device__ inline float bpr128_xval(BtPointR128 p) { return bt128_to_float(p.x) / bt128_to_float(p.den); }
__device__ inline float bpr128_yval(BtPointR128 p) { return bt128_to_float(p.y) / bt128_to_float(p.den); }
__device__ inline float bpr128_zval(BtPointR128 p) { return bt128_to_float(p.z) / bt128_to_float(p.den); }

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

__device__ inline BtRational128 br128_from_128(BtInt128 num, BtInt128 den) {
    BtRational128 r;
    r.sign = bt128_sign(num);
    r.num = (r.sign >= 0) ? num : bt128_neg(num);
    int dsign = bt128_sign(den);
    if (dsign >= 0) { r.den = den; }
    else { r.sign = -r.sign; r.den = bt128_neg(den); }
    r.isInt64 = false;
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
struct BtEdge {
    BtEdge* next;
    BtEdge* prev;
    BtEdge* reverse;
    BtVertex* target;
    int copy;
};

struct BtVertex {
    BtVertex* next;
    BtVertex* prev;
    BtEdge* edges;
    BtPointR128 point128;
    BtPoint32 point;
    int copy;
};

struct BtIntermediateHull {
    BtVertex* minXy;
    BtVertex* maxXy;
    BtVertex* minYx;
    BtVertex* maxYx;
};

// D&C stack item for iterative computeInternal
struct BtDCStackItem {
    int start;
    int end;
    int stage;                   // 0 = descend, 1 = merge
    BtIntermediateHull left_hull;
    BtIntermediateHull right_hull;
    BtIntermediateHull* result;  // where to write the merged hull
};

// Vertex operations
__device__ inline BtPoint32 bv_sub(BtVertex* a, BtVertex* b) { return bp32_sub(a->point, b->point); }

// Vertex dot with Point64 -> Rational128
__device__ inline BtRational128 bv_dot(BtVertex* v, BtPoint64 b) {
    if (v->point.index >= 0) {
        return br128_from_i64(bp32_dot64(v->point, b));
    }
    BtInt128 sum = bt128_add(
        bt128_add(bt128_mul_i64(v->point128.x, b.x), bt128_mul_i64(v->point128.y, b.y)),
        bt128_mul_i64(v->point128.z, b.z));
    return br128_from_128(sum, v->point128.den);
}

__device__ inline float bv_xval(BtVertex* v) {
    return (v->point.index >= 0) ? (float)v->point.x : bpr128_xval(v->point128);
}
__device__ inline float bv_yval(BtVertex* v) {
    return (v->point.index >= 0) ? (float)v->point.y : bpr128_yval(v->point128);
}
__device__ inline float bv_zval(BtVertex* v) {
    return (v->point.index >= 0) ? (float)v->point.z : bpr128_zval(v->point128);
}

__device__ inline void bt_edge_link(BtEdge* a, BtEdge* n) {
    a->next = n;
    n->prev = a;
}

// ============================================================================
// Pool allocator backed by DeviceHeap (lane 0 only)
// ============================================================================

#define BT_ERR_POOL_EXHAUST 5
#define BTPOOL_BLOCK_SIZE   8192   // edges per slab
#define BTPOOL_MAX_BLOCKS   32     // max heap slabs tracked for cleanup

// BtPool: free list of fixed-size objects, backed by heap slabs.
// All operations (init, new, free) are lane-0-only.
struct BtPool {
    void*       freeList;
    DeviceHeap* scratch_heap;
    int         objSize;    // padded to multiple of 4
    int         error;
    int         nblocks;
    void*       blocks[BTPOOL_MAX_BLOCKS];
    // Arena mode: when arena_base != NULL, growth uses atomic bump alloc
    // from a shared pre-allocated arena (warp-safe, no heap lock needed).
    char*       arena_base;
    int*        arena_offset;   // pointer to shared-memory atomic counter
    int         arena_cap;
    int         growSlabSize;   // edges per growth slab (arena mode only)
};

// Allocate one slab of BTPOOL_BLOCK_SIZE objects from scratch_heap and prepend
// to the free list. Serial (no warp parallelism). Lane 0 only.
__device__ inline int btpool_add_block(BtPool* p) {
    if (p->nblocks >= BTPOOL_MAX_BLOCKS) { p->error = BT_ERR_POOL_EXHAUST; return -1; }
    int slab_n = (p->arena_base && p->growSlabSize > 0) ? p->growSlabSize : BTPOOL_BLOCK_SIZE;
    void* block = NULL;
    if (p->arena_base) {
        // Arena-backed: atomic bump alloc (warp-safe, no heap lock)
        int slab_bytes = slab_n * p->objSize;
        int offset = atomicAdd(p->arena_offset, slab_bytes);
        if (offset + slab_bytes > p->arena_cap) { p->error = BT_ERR_POOL_EXHAUST; return -1; }
        block = p->arena_base + offset;
    } else {
        // Heap-backed (original path, lane 0 only)
        if (heap_alloc(p->scratch_heap, (unsigned int)(BTPOOL_BLOCK_SIZE * p->objSize), &block) != HEAP_OK) {
            p->error = BT_ERR_POOL_EXHAUST; return -1;
        }
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

// Initialise pool with 2 pre-allocated slabs. Lane 0 only.
__device__ inline int btpool_init(BtPool* p, DeviceHeap* scratch_heap, int objSize) {
    p->scratch_heap = scratch_heap;
    p->objSize      = (objSize + 3) & ~3;
    p->freeList     = NULL;
    p->error        = 0;
    p->nblocks      = 0;
    p->arena_base   = NULL;
    p->arena_offset = NULL;
    p->arena_cap    = 0;
    p->growSlabSize = 0;
    if (btpool_add_block(p) < 0) return -1;
    if (btpool_add_block(p) < 0) return -1;
    return 0;
}

// Init an arena-backed pool with one initial slab of custom size.
// arena_base/arena_offset/arena_cap describe a shared pre-allocated arena.
// initial_slab_edges = number of edges in the first slab.
__device__ inline int btpool_init_arena(BtPool* p, int objSize,
    char* arena_base, int* arena_offset, int arena_cap, int initial_slab_edges) {
    p->scratch_heap = NULL;
    p->objSize      = (objSize + 3) & ~3;
    p->freeList     = NULL;
    p->error        = 0;
    p->nblocks      = 0;
    p->arena_base   = arena_base;
    p->arena_offset = arena_offset;
    p->arena_cap    = arena_cap;
    // Growth slabs use same size as initial slab (not BTPOOL_BLOCK_SIZE)
    p->growSlabSize = initial_slab_edges;
    // Allocate initial slab from arena via atomic bump
    int slab_bytes = initial_slab_edges * p->objSize;
    int offset = atomicAdd(arena_offset, slab_bytes);
    if (offset + slab_bytes > arena_cap) { p->error = BT_ERR_POOL_EXHAUST; return -1; }
    void* block = arena_base + offset;
    p->blocks[p->nblocks++] = block;
    char* b = (char*)block;
    int padded = p->objSize;
    *(void**)b = NULL;
    for (int i = 1; i < initial_slab_edges; i++)
        *(void**)(b + i * padded) = b + (i - 1) * padded;
    p->freeList = b + (initial_slab_edges - 1) * padded;
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

// Error codes (distinct from pool OOM = 1)
#define BT_ERR_SORT_STACK  2

// ============================================================================
// Scratch size calculation
// ============================================================================

// Aligned size matching bt_alloc's 16-byte alignment.
#define BT_ALIGN16(x) (((x) + 15) & ~15)

// Compute total WarpPool scratch bytes needed for D&C hull with n points.
// Sort scratch is rewound after sort. D&C stack is rewound after computeInternal.
// BFS queues are allocated after D&C, sharing that space.
// Edge pool is backed by DeviceHeap (scratch_heap) via arena, not included here.
// D&C stack is per-lane local memory (BT_DC_MAX_STACK_LOCAL), not in WarpPool.
// Peak = presort + max(sort_scratch, persistent + bfs_queues).
__host__ __device__ inline int dandc_scratch_bytes(int n) {
    int total = 0;
    // presort: BtPoint32 array (persists through sort, NOT rewound)
    total += BT_ALIGN16(n * (int)sizeof(BtPoint32));

    int sort_scratch = BT_ALIGN16(n * (int)sizeof(BtPoint32) + WS_MAX_STACK * 2 * (int)sizeof(int));

    int postsort_persistent = 0;
    // postsort: vertex block (persists)
    postsort_persistent += BT_ALIGN16(n * (int)sizeof(BtVertex));

    // BFS queues for mesh extraction + volume (each n * sizeof(BtVertex*))
    int phase_bfs = BT_ALIGN16(n * (int)sizeof(BtVertex*));

    int postsort = postsort_persistent + phase_bfs;

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
    BtPool edgePool;
    BtVertex* vertexBase;
    int mergeStamp;
    int* mergeStampPtr;    // if non-null, use atomicAdd on shared stamp
    int minAxis, medAxis, maxAxis;
    int usedEdgePairs;
#ifdef TRACK_MAX_EDGE_PAIRS
    int maxEdgePairs;
#endif
    BtVertex* vertexList;
    WarpPool*   wp;
    DeviceHeap* scratch_heap;  // backing for edgePool slabs
    int         npoints;       // original point count (queue size bound)
    int fma_total_edges;       // instrumentation: total edges chased in bt_findMaxAngle
    int fma_min_edges;         // instrumentation: min edges per call
    int fma_max_edges;         // instrumentation: max edges per call
    int fma_calls;             // instrumentation: number of bt_findMaxAngle calls
};

// ============================================================================
// Core algorithm functions
// ============================================================================

__device__ inline BtEdge* bt_newEdgePair(BtHullState* s, BtVertex* from, BtVertex* to) {
    BtEdge* e = btpool_new_edge(&s->edgePool);
    BtEdge* r = btpool_new_edge(&s->edgePool);
    if (!e || !r) return NULL;
    e->reverse = r;
    r->reverse = e;
    e->copy = s->mergeStamp;
    r->copy = s->mergeStamp;
    e->target = to;
    r->target = from;
    e->next = NULL; e->prev = NULL;
    r->next = NULL; r->prev = NULL;
    s->usedEdgePairs++;
#ifdef TRACK_MAX_EDGE_PAIRS
    if (s->usedEdgePairs > s->maxEdgePairs) {
        s->maxEdgePairs = s->usedEdgePairs;
    }
#endif
    return e;
}

__device__ inline void bt_removeEdgePair(BtHullState* s, BtEdge* edge) {
    BtEdge* n = edge->next;
    BtEdge* r = edge->reverse;
    if (n != edge) {
        n->prev = edge->prev;
        edge->prev->next = n;
        r->target->edges = n;
    } else {
        r->target->edges = NULL;
    }
    n = r->next;
    if (n != r) {
        n->prev = r->prev;
        r->prev->next = n;
        edge->target->edges = n;
    } else {
        edge->target->edges = NULL;
    }
    btpool_free(&s->edgePool, edge);
    btpool_free(&s->edgePool, r);
    s->usedEdgePairs--;
}

enum BtOrientation { BT_NONE, BT_CLOCKWISE, BT_COUNTER_CLOCKWISE };

__device__ inline BtOrientation bt_getOrientation(BtEdge* prev_e, BtEdge* next_e, BtPoint32 s_dir, BtPoint32 t_dir) {
    if (prev_e->next == next_e) {
        if (prev_e->prev == next_e) {
            BtPoint64 n = bp32_cross(t_dir, s_dir);
            BtPoint64 m = bp32_cross(
                bp32_sub(prev_e->target->point, next_e->reverse->target->point),
                bp32_sub(next_e->target->point, next_e->reverse->target->point));
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

__device__ inline BtEdge* bt_findMaxAngle(int mergeStamp, bool ccw, BtVertex* start,
    BtPoint32 s_dir, BtPoint64 rxs, BtPoint64 sxrxs, BtRational64* minCot,
    int* edge_count)
{
    BtEdge* minEdge = NULL;
    BtEdge* e = start->edges;
    if (!e) { if (edge_count) *edge_count = 0; return NULL; }
    int count = 0;
    do {
        count++;
        if (e->copy > mergeStamp) {
            BtPoint32 t = bp32_sub(e->target->point, start->point);
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
                    } else if (c == 0 && (ccw == (bt_getOrientation(minEdge, e, s_dir, t) == BT_COUNTER_CLOCKWISE))) {
                        minEdge = e;
                    }
                }
            }
        }
        e = e->next;
        if (!e) {
            DPRINTF("[BUG] bt_findMaxAngle: NULL edge->next after %d edges, "
                    "blk=%d lane=%d start=%p mergeStamp=%d\n",
                    count, blockIdx.x, threadIdx.x, start, mergeStamp);
            if (edge_count) *edge_count = count;
            return minEdge;  // bail out instead of crashing
        }
    } while (e != start->edges);
    if (edge_count) *edge_count = count;
    return minEdge;
}

__device__ inline void bt_findEdgeForCoplanarFaces(BtHullState* s, BtVertex* c0, BtVertex* c1,
    BtEdge** e0, BtEdge** e1, BtVertex* stop0, BtVertex* stop1)
{
    BtEdge* start0 = *e0;
    BtEdge* start1 = *e1;
    BtPoint32 et0 = start0 ? start0->target->point : c0->point;
    BtPoint32 et1 = start1 ? start1->target->point : c1->point;
    BtPoint32 s_dir = bp32_sub(c1->point, c0->point);
    BtPoint64 normal = bp32_cross(bp32(0,0,-1), s_dir);
    // Use whichever start edge exists to define the coplanar normal
    if (start0 || start1) {
        BtVertex* ref = (start0 ? start0 : start1)->target;
        normal = bp32_cross(bp32_sub(ref->point, c0->point), s_dir);
    }
    long long dist = bp32_dot64(c0->point, normal);
    BtPoint64 perp = bp32_cross64(s_dir, normal);

    long long maxDot0 = bp32_dot64(et0, perp);
    if (*e0) {
        while ((*e0)->target != stop0) {
            BtEdge* e = (*e0)->reverse->prev;
            if (bp32_dot64(e->target->point, normal) < dist) break;
            if (e->copy == s->mergeStamp) break;
            long long dot = bp32_dot64(e->target->point, perp);
            if (dot <= maxDot0) break;
            maxDot0 = dot;
            *e0 = e;
            et0 = e->target->point;
        }
    }

    long long maxDot1 = bp32_dot64(et1, perp);
    if (*e1) {
        while ((*e1)->target != stop1) {
            BtEdge* e = (*e1)->reverse->next;
            if (bp32_dot64(e->target->point, normal) < dist) break;
            if (e->copy == s->mergeStamp) break;
            long long dot = bp32_dot64(e->target->point, perp);
            if (dot <= maxDot1) break;
            maxDot1 = dot;
            *e1 = e;
            et1 = e->target->point;
        }
    }

    long long dx = maxDot1 - maxDot0;
    if (dx > 0) {
        while (true) {
            long long dy = bp32_dot64_32(bp32_sub(et1, et0), s_dir);
            if (*e0 && ((*e0)->target != stop0)) {
                BtEdge* f0 = (*e0)->next->reverse;
                if (f0->copy > s->mergeStamp) {
                    long long dx0 = bp32_dot64(bp32_sub(f0->target->point, et0), perp);
                    long long dy0 = bp32_dot64_32(bp32_sub(f0->target->point, et0), s_dir);
                    if ((dx0 == 0) ? (dy0 < 0) : ((dx0 < 0) && (br64_cmp(br64_make(dy0, dx0), br64_make(dy, dx)) >= 0))) {
                        et0 = f0->target->point;
                        dx = bp32_dot64(bp32_sub(et1, et0), perp);
                        *e0 = (*e0 == start0) ? NULL : f0;
                        continue;
                    }
                }
            }
            if (*e1 && ((*e1)->target != stop1)) {
                BtEdge* f1 = (*e1)->reverse->next;
                if (f1->copy > s->mergeStamp) {
                    BtPoint32 d1 = bp32_sub(f1->target->point, et1);
                    if (bp32_dot64(d1, normal) == 0) {
                        long long dx1 = bp32_dot64(d1, perp);
                        long long dy1 = bp32_dot64_32(d1, s_dir);
                        long long dxn = bp32_dot64(bp32_sub(f1->target->point, et0), perp);
                        if ((dxn > 0) && ((dx1 == 0) ? (dy1 < 0) : ((dx1 < 0) && (br64_cmp(br64_make(dy1, dx1), br64_make(dy, dx)) > 0)))) {
                            *e1 = f1;
                            et1 = (*e1)->target->point;
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
                if (f1->copy > s->mergeStamp) {
                    long long dx1 = bp32_dot64(bp32_sub(f1->target->point, et1), perp);
                    long long dy1 = bp32_dot64_32(bp32_sub(f1->target->point, et1), s_dir);
                    if ((dx1 == 0) ? (dy1 > 0) : ((dx1 < 0) && (br64_cmp(br64_make(dy1, dx1), br64_make(dy, dx)) <= 0))) {
                        et1 = f1->target->point;
                        dx = bp32_dot64(bp32_sub(et1, et0), perp);
                        *e1 = (*e1 == start1) ? NULL : f1;
                        continue;
                    }
                }
            }
            if (*e0 && ((*e0)->target != stop0)) {
                BtEdge* f0 = (*e0)->reverse->prev;
                if (f0->copy > s->mergeStamp) {
                    BtPoint32 d0 = bp32_sub(f0->target->point, et0);
                    if (bp32_dot64(d0, normal) == 0) {
                        long long dx0 = bp32_dot64(d0, perp);
                        long long dy0 = bp32_dot64_32(d0, s_dir);
                        long long dxn = bp32_dot64(bp32_sub(et1, f0->target->point), perp);
                        if ((dxn < 0) && ((dx0 == 0) ? (dy0 > 0) : ((dx0 < 0) && (br64_cmp(br64_make(dy0, dx0), br64_make(dy, dx)) < 0)))) {
                            *e0 = f0;
                            et0 = (*e0)->target->point;
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

__device__ inline bool bt_mergeProjection(BtHullState* s, BtIntermediateHull* h0, BtIntermediateHull* h1, BtVertex** c0, BtVertex** c1) {
    BtVertex* v0 = h0->maxYx;
    BtVertex* v1 = h1->minYx;
    if ((v0->point.x == v1->point.x) && (v0->point.y == v1->point.y)) {
        BtVertex* v1p = v1->prev;
        if (v1p == v1) {
            *c0 = v0;
            if (v1->edges) {
                v1 = v1->edges->target;
            }
            *c1 = v1;
            return false;
        }
        BtVertex* v1n = v1->next;
        v1p->next = v1n;
        v1n->prev = v1p;
        if (v1 == h1->minXy) {
            h1->minXy = ((v1n->point.x < v1p->point.x) || ((v1n->point.x == v1p->point.x) && (v1n->point.y < v1p->point.y))) ? v1n : v1p;
        }
        if (v1 == h1->maxXy) {
            h1->maxXy = ((v1n->point.x > v1p->point.x) || ((v1n->point.x == v1p->point.x) && (v1n->point.y > v1p->point.y))) ? v1n : v1p;
        }
    }

    v0 = h0->maxXy;
    v1 = h1->maxXy;
    BtVertex* v00 = NULL;
    BtVertex* v10 = NULL;
    int sign = 1;

    for (int side = 0; side <= 1; side++) {
        int dx = (v1->point.x - v0->point.x) * sign;
        if (dx > 0) {
            while (true) {
                int dy = v1->point.y - v0->point.y;
                BtVertex* w0 = side ? v0->next : v0->prev;
                if (w0 != v0) {
                    int dx0 = (w0->point.x - v0->point.x) * sign;
                    int dy0 = w0->point.y - v0->point.y;
                    if ((dy0 <= 0) && ((dx0 == 0) || ((dx0 < 0) && ((long long)dy0 * dx <= (long long)dy * dx0)))) {
                        v0 = w0; dx = (v1->point.x - v0->point.x) * sign; continue;
                    }
                }
                BtVertex* w1 = side ? v1->next : v1->prev;
                if (w1 != v1) {
                    int dx1 = (w1->point.x - v1->point.x) * sign;
                    int dy1 = w1->point.y - v1->point.y;
                    int dxn = (w1->point.x - v0->point.x) * sign;
                    if ((dxn > 0) && (dy1 < 0) && ((dx1 == 0) || ((dx1 < 0) && ((long long)dy1 * dx < (long long)dy * dx1)))) {
                        v1 = w1; dx = dxn; continue;
                    }
                }
                break;
            }
        } else if (dx < 0) {
            while (true) {
                int dy = v1->point.y - v0->point.y;
                BtVertex* w1 = side ? v1->prev : v1->next;
                if (w1 != v1) {
                    int dx1 = (w1->point.x - v1->point.x) * sign;
                    int dy1 = w1->point.y - v1->point.y;
                    if ((dy1 >= 0) && ((dx1 == 0) || ((dx1 < 0) && ((long long)dy1 * dx <= (long long)dy * dx1)))) {
                        v1 = w1; dx = (v1->point.x - v0->point.x) * sign; continue;
                    }
                }
                BtVertex* w0 = side ? v0->prev : v0->next;
                if (w0 != v0) {
                    int dx0 = (w0->point.x - v0->point.x) * sign;
                    int dy0 = w0->point.y - v0->point.y;
                    int dxn = (v1->point.x - w0->point.x) * sign;
                    if ((dxn < 0) && (dy0 > 0) && ((dx0 == 0) || ((dx0 < 0) && ((long long)dy0 * dx < (long long)dy * dx0)))) {
                        v0 = w0; dx = dxn; continue;
                    }
                }
                break;
            }
        } else {
            int x = v0->point.x;
            int y0 = v0->point.y;
            BtVertex* w0 = v0;
            BtVertex* t;
            while (((t = side ? w0->next : w0->prev) != v0) && (t->point.x == x) && (t->point.y <= y0)) {
                w0 = t; y0 = t->point.y;
            }
            v0 = w0;
            int y1 = v1->point.y;
            BtVertex* w1 = v1;
            while (((t = side ? w1->prev : w1->next) != v1) && (t->point.x == x) && (t->point.y >= y1)) {
                w1 = t; y1 = t->point.y;
            }
            v1 = w1;
        }
        if (side == 0) {
            v00 = v0; v10 = v1;
            v0 = h0->minXy; v1 = h1->minXy; sign = -1;
        }
    }

    v0->prev = v1;
    v1->next = v0;
    v00->next = v10;
    v10->prev = v00;

    if (h1->minXy->point.x < h0->minXy->point.x) h0->minXy = h1->minXy;
    if (h1->maxXy->point.x >= h0->maxXy->point.x) h0->maxXy = h1->maxXy;
    h0->maxYx = h1->maxYx;

    *c0 = v00;
    *c1 = v10;
    return true;
}

// Check circularity of a vertex's edge ring. Returns 0 if OK, -1 if broken.
__device__ inline int bt_checkEdgeRing(BtVertex* v, const char* /*label*/) {
    BtEdge* e = v->edges;
    if (!e) return 0;
    BtEdge* start = e;
    for (int i = 0; i < 10000; i++) {
        e = e->next;
        if (!e) return -1;
        if (e == start) return 0;
    }
    return -1;
}

__device__ inline void bt_merge(BtHullState* s, BtIntermediateHull* h0, BtIntermediateHull* h1) {
    // Single-thread merge (caller is the one active thread).
    if (!h1->maxXy) return;
    if (!h0->maxXy) { *h0 = *h1; return; }

    BtVertex* c0 = NULL;
    BtEdge* toPrev0 = NULL;
    BtEdge* firstNew0 = NULL;
    BtEdge* pendingHead0 = NULL;
    BtEdge* pendingTail0 = NULL;
    BtVertex* c1 = NULL;
    BtEdge* toPrev1 = NULL;
    BtEdge* firstNew1 = NULL;
    BtEdge* pendingHead1 = NULL;
    BtEdge* pendingTail1 = NULL;
    BtPoint32 prevPoint;

    if (s->mergeStampPtr) {
        s->mergeStamp = atomicAdd(s->mergeStampPtr, -1) - 1;
    } else {
        s->mergeStamp--;
    }
    int mergeStamp = s->mergeStamp;

    if (bt_mergeProjection(s, h0, h1, &c0, &c1)) {
        BtPoint32 sd = bp32_sub(c1->point, c0->point);
        BtPoint64 normal = bp32_cross(bp32(0,0,-1), sd);
        BtPoint64 t = bp32_cross64(sd, normal);

        BtEdge* e = c0->edges;
        BtEdge* start0 = NULL;
        if (e) {
            do {
                long long dot = bp32_dot64(bp32_sub(e->target->point, c0->point), normal);
                if ((dot == 0) && (bp32_dot64(bp32_sub(e->target->point, c0->point), t) > 0)) {
                    if (!start0 || (bt_getOrientation(start0, e, sd, bp32(0,0,-1)) == BT_CLOCKWISE))
                        start0 = e;
                }
                e = e->next;
            } while (e != c0->edges);
        }

        e = c1->edges;
        BtEdge* start1 = NULL;
        if (e) {
            do {
                long long dot = bp32_dot64(bp32_sub(e->target->point, c1->point), normal);
                if ((dot == 0) && (bp32_dot64(bp32_sub(e->target->point, c1->point), t) > 0)) {
                    if (!start1 || (bt_getOrientation(start1, e, sd, bp32(0,0,-1)) == BT_COUNTER_CLOCKWISE))
                        start1 = e;
                }
                e = e->next;
            } while (e != c1->edges);
        }

        if (start0 || start1) {
            bt_findEdgeForCoplanarFaces(s, c0, c1, &start0, &start1, NULL, NULL);
            if (start0) c0 = start0->target;
            if (start1) c1 = start1->target;
        }
        prevPoint = c1->point;
        prevPoint.z++;
    } else {
        prevPoint = c1->point;
        prevPoint.x++;
    }

    BtVertex* first0 = c0;
    BtVertex* first1 = c1;
    bool firstRun = true;

    while (true) {
        BtPoint32 sd = bp32_sub(c1->point, c0->point);
        BtPoint32 r = bp32_sub(prevPoint, c0->point);
        BtPoint64 rxs = bp32_cross(r, sd);
        BtPoint64 sxrxs = bp32_cross64(sd, rxs);

        BtRational64 minCot0, minCot1;
        int ec0 = 0, ec1 = 0;
        BtEdge* min0 = bt_findMaxAngle(mergeStamp, false, c0, sd, rxs, sxrxs, &minCot0, &ec0);
        BtEdge* min1 = bt_findMaxAngle(mergeStamp, true,  c1, sd, rxs, sxrxs, &minCot1, &ec1);

        s->fma_total_edges += ec0 + ec1;
        if (ec0 < s->fma_min_edges) s->fma_min_edges = ec0;
        if (ec1 < s->fma_min_edges) s->fma_min_edges = ec1;
        if (ec0 > s->fma_max_edges) s->fma_max_edges = ec0;
        if (ec1 > s->fma_max_edges) s->fma_max_edges = ec1;
        s->fma_calls += 2;

        if (!min0 && !min1) {
            BtEdge* e = bt_newEdgePair(s, c0, c1);
            if (!e) { DPRINTF("[BUG] bt_merge OOM at final edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); return; }
            bt_edge_link(e, e);
            c0->edges = e;
            e = e->reverse;
            bt_edge_link(e, e);
            c1->edges = e;
            return;
        }

        int cmp = !min0 ? 1 : !min1 ? -1 : br64_cmp(minCot0, minCot1);

        if (firstRun || ((cmp >= 0) ? !br64_isNegInf(minCot1) : !br64_isNegInf(minCot0))) {
            BtEdge* e = bt_newEdgePair(s, c0, c1);
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
            bt_findEdgeForCoplanarFaces(s, c0, c1, &e0, &e1, NULL, NULL);
        }

        if ((cmp >= 0) && e1) {
            if (toPrev1) {
                for (BtEdge *e = toPrev1->next, *n = NULL; e != min1; e = n) {
                    n = e->next;
                    bt_removeEdgePair(s, e);
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
            prevPoint = c1->point;
            c1 = e1->target;
            toPrev1 = e1->reverse;
        }

        if ((cmp <= 0) && e0) {
            if (toPrev0) {
                for (BtEdge *e = toPrev0->prev, *n = NULL; e != min0; e = n) {
                    n = e->prev;
                    bt_removeEdgePair(s, e);
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
            prevPoint = c0->point;
            c0 = e0->target;
            toPrev0 = e0->reverse;
        }

        if ((c0 == first0) && (c1 == first1)) {
            if (toPrev0 == NULL) {
                bt_edge_link(pendingHead0, pendingTail0);
                c0->edges = pendingTail0;
            } else {
                for (BtEdge *e = toPrev0->prev, *n = NULL; e != firstNew0; e = n) {
                    n = e->prev;
                    bt_removeEdgePair(s, e);
                }
                if (pendingTail0) {
                    bt_edge_link(pendingHead0, toPrev0);
                    bt_edge_link(firstNew0, pendingTail0);
                }
            }
            if (toPrev1 == NULL) {
                bt_edge_link(pendingTail1, pendingHead1);
                c1->edges = pendingTail1;
            } else {
                for (BtEdge *e = toPrev1->next, *n = NULL; e != firstNew1; e = n) {
                    n = e->next;
                    bt_removeEdgePair(s, e);
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

// ============================================================================
// computeInternal base case — handles n <= 2
// ============================================================================

__device__ inline void bt_computeBase(BtHullState* s, int start, int end, BtIntermediateHull* result) {
    int n = end - start;
    switch (n) {
    case 0:
        result->minXy = NULL; result->maxXy = NULL;
        result->minYx = NULL; result->maxYx = NULL;
        return;
    case 2: {
        BtVertex* v = &s->vertexBase[start];
        BtVertex* w = v + 1;
        if (bp32_ne(v->point, w->point)) {
            int dx = v->point.x - w->point.x;
            int dy = v->point.y - w->point.y;
            if ((dx == 0) && (dy == 0)) {
                if (v->point.z > w->point.z) { BtVertex* t = w; w = v; v = t; }
                v->next = v; v->prev = v;
                result->minXy = v; result->maxXy = v;
                result->minYx = v; result->maxYx = v;
            } else {
                v->next = w; v->prev = w; w->next = v; w->prev = v;
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
            BtEdge* e = bt_newEdgePair(s, v, w);
            if (!e) { DPRINTF("[BUG] bt_merge OOM at coplanar edge, blk=%d lane=%d\n", blockIdx.x, threadIdx.x); return; }
            bt_edge_link(e, e); v->edges = e;
            e = e->reverse;
            bt_edge_link(e, e); w->edges = e;
            return;
        }
    }
    // fallthrough
    case 1: {
        BtVertex* v = &s->vertexBase[start];
        v->edges = NULL;
        v->next = v; v->prev = v;
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
#define BT_DC_MAX_STACK_LOCAL 64

__device__ inline void bt_computeInternal(BtHullState* s, int start, int end, BtIntermediateHull* result) {
    BtDCStackItem stack[BT_DC_MAX_STACK_LOCAL];
    int sp = 0;

    // Push root
    stack[sp].start = start;
    stack[sp].end = end;
    stack[sp].stage = 0;
    stack[sp].left_hull.minXy = NULL; stack[sp].left_hull.maxXy = NULL;
    stack[sp].left_hull.minYx = NULL; stack[sp].left_hull.maxYx = NULL;
    stack[sp].right_hull.minXy = NULL; stack[sp].right_hull.maxXy = NULL;
    stack[sp].right_hull.minYx = NULL; stack[sp].right_hull.maxYx = NULL;
    stack[sp].result = result;
    sp++;

    while (sp > 0) {
        sp--;
        BtDCStackItem* item = &stack[sp];

        if (item->stage == 0) {
            int n = item->end - item->start;
            if (n <= 2) {
                bt_computeBase(s, item->start, item->end, item->result);
            } else {
                int split0 = item->start + n / 2;
                BtPoint32 p = s->vertexBase[split0 - 1].point;
                int split1 = split0;
                while ((split1 < item->end) && bp32_eq(s->vertexBase[split1].point, p)) split1++;

                BtIntermediateHull* res = item->result;

                // Convert current item to merge (stage 1)
                item->stage = 1;
                item->left_hull.minXy = NULL; item->left_hull.maxXy = NULL;
                item->left_hull.minYx = NULL; item->left_hull.maxYx = NULL;
                item->right_hull.minXy = NULL; item->right_hull.maxXy = NULL;
                item->right_hull.minYx = NULL; item->right_hull.maxYx = NULL;
                item->result = res;
                sp++;  // re-push (it's already in place)
                BtDCStackItem* merge_ptr = &stack[sp - 1];

                // Push right child
                if (sp >= BT_DC_MAX_STACK_LOCAL) { s->edgePool.error = BT_ERR_DC_STACK; return; }
                stack[sp].start = split1;
                stack[sp].end = item->end;
                stack[sp].stage = 0;
                stack[sp].left_hull.minXy = NULL; stack[sp].left_hull.maxXy = NULL;
                stack[sp].left_hull.minYx = NULL; stack[sp].left_hull.maxYx = NULL;
                stack[sp].right_hull.minXy = NULL; stack[sp].right_hull.maxXy = NULL;
                stack[sp].right_hull.minYx = NULL; stack[sp].right_hull.maxYx = NULL;
                stack[sp].result = &merge_ptr->right_hull;
                sp++;

                // Push left child
                if (sp >= BT_DC_MAX_STACK_LOCAL) { s->edgePool.error = BT_ERR_DC_STACK; return; }
                stack[sp].start = item->start;
                stack[sp].end = split0;
                stack[sp].stage = 0;
                stack[sp].left_hull.minXy = NULL; stack[sp].left_hull.maxXy = NULL;
                stack[sp].left_hull.minYx = NULL; stack[sp].left_hull.maxYx = NULL;
                stack[sp].right_hull.minXy = NULL; stack[sp].right_hull.maxXy = NULL;
                stack[sp].right_hull.minYx = NULL; stack[sp].right_hull.maxYx = NULL;
                stack[sp].result = &merge_ptr->left_hull;
                sp++;
            }
        } else {
            bt_merge(s, &item->left_hull, &item->right_hull);
            *item->result = item->left_hull;
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
// Volume extraction from half-edge hull
// ============================================================================

__device__ inline float bt_computeVolume(BtHullState* s) {
    if (!s->vertexList) return 0.0f;

    // BFS over vertices (Bullet getVertexCopy pattern).
    // Queue size bounded by number of hull vertices (≤ n), no stack overflow possible.
    int vstamp = --s->mergeStamp;  // vertex visited stamp
    int fstamp = --s->mergeStamp;  // face/edge visited stamp

    // Allocate BFS queue from pool (n vertex pointers)
    BtVertex** queue = (BtVertex**)bt_alloc(s->wp, s->npoints * (int)sizeof(BtVertex*));
    if (!queue) return -1.0f;
    int qhead = 0, qtail = 0;

    s->vertexList->copy = vstamp;
    queue[qtail++] = s->vertexList;

    BtPoint32 ref = s->vertexList->point;
    BtInt128 volume = bt128_from_u64(0);

    while (qhead < qtail) {
        BtVertex* v = queue[qhead++];
        BtEdge* e = v->edges;
        if (!e) continue;
        do {
            if (e->target->copy != vstamp) {
                e->target->copy = vstamp;
                queue[qtail++] = e->target;
            }
            if (e->copy != fstamp) {
                // Walk face: fan-triangulate from first vertex
                BtVertex* a = NULL;
                BtVertex* b = NULL;
                BtEdge* f = e;
                do {
                    if (a && b) {
                        BtPoint32 va = bp32_sub(v->point, ref);
                        BtPoint32 pa = bp32_sub(a->point, ref);
                        BtPoint32 pb = bp32_sub(b->point, ref);
                        long long vol = bp32_dot64(va, bp32_cross(pa, pb));
                        volume = bt128_add(volume, bt128_from_i64(vol));
                    }
                    f->copy = fstamp;
                    a = b;
                    b = f->target;
                    f = f->reverse->prev;
                } while (f != e);
            }
            e = e->next;
        } while (e != v->edges);
    }

    float sv = bt128_to_float(volume);
    float scale = s->scaling[0] * s->scaling[1] * s->scaling[2];
    float vol = fabsf(sv * scale) / 6.0f;
    return vol;
}

// ============================================================================
// Mesh extraction from half-edge hull (lane 0 only, after bt_computeVolume)
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

    if (!s->vertexList) return 0;

    BtVertex** queue = (BtVertex**)bt_alloc(s->wp, s->npoints * (int)sizeof(BtVertex*));
    if (!queue) return -1;
    int n_verts = 0;
    int n_tris  = 0;

    int vstamp = --s->mergeStamp;
    s->vertexList->copy = vstamp;
    queue[n_verts++] = s->vertexList;

    // BFS: discover all hull vertices
    int qhead = 0;
    while (qhead < n_verts) {
        BtVertex* v = queue[qhead++];
        BtEdge* e = v->edges;
        if (!e) continue;
        do {
            if (e->target->copy != vstamp) {
                e->target->copy = vstamp;
                queue[n_verts++] = e->target;
            }
            e = e->next;
        } while (e != v->edges);
    }

    if (out_verts) {
        // Extract mode: encode vertex indices, write verts, fan-triangulate faces.
        int idx_base = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++)
            queue[i]->copy = idx_base - i;

        float sc[3]  = { s->scaling[0], s->scaling[1], s->scaling[2] };
        float cen[3] = { s->center[0],  s->center[1],  s->center[2]  };
        int medAx = s->medAxis, maxAx = s->maxAxis, minAx = s->minAxis;
        for (int i = 0; i < n_verts; i++) {
            BtVertex* v = queue[i];
            float xyz[3];
            xyz[medAx] = bv_xval(v) * sc[medAx] + cen[medAx];
            xyz[maxAx] = bv_yval(v) * sc[maxAx] + cen[maxAx];
            xyz[minAx] = bv_zval(v) * sc[minAx] + cen[minAx];
            out_verts[i * 3 + 0] = xyz[0];
            out_verts[i * 3 + 1] = xyz[1];
            out_verts[i * 3 + 2] = xyz[2];
        }

        int fstamp = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++) {
            BtVertex* v = queue[i];
            BtEdge* e = v->edges;
            if (!e) continue;
            do {
                if (e->copy != fstamp) {
                    BtVertex* a = NULL, *b = NULL;
                    BtEdge* f = e;
                    do {
                        if (a && b) {
                            out_tris[n_tris * 3 + 0] = idx_base - v->copy;
                            out_tris[n_tris * 3 + 1] = idx_base - a->copy;
                            out_tris[n_tris * 3 + 2] = idx_base - b->copy;
                            n_tris++;
                        }
                        f->copy = fstamp;
                        a = b;
                        b = f->target;
                        f = f->reverse->prev;
                    } while (f != e);
                }
                e = e->next;
            } while (e != v->edges);
        }
    } else {
        // Count-only mode: count triangles via fan formula (k edges → k-2 tris).
        int fstamp = --s->mergeStamp;
        for (int i = 0; i < n_verts; i++) {
            BtVertex* v = queue[i];
            BtEdge* e = v->edges;
            if (!e) continue;
            do {
                if (e->copy != fstamp) {
                    int k = 0;
                    BtEdge* f = e;
                    do { f->copy = fstamp; k++; f = f->reverse->prev; } while (f != e);
                    if (k >= 3) n_tris += k - 2;
                }
                e = e->next;
            } while (e != v->edges);
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

    float inv[3];
    inv[0] = (sc[0] != 0.0f) ? 1.0f / sc[0] : 0.0f;
    inv[1] = (sc[1] != 0.0f) ? 1.0f / sc[1] : 0.0f;
    inv[2] = (sc[2] != 0.0f) ? 1.0f / sc[2] : 0.0f;

    float cen[3] = {(mn0+mx0)*0.5f, (mn1+mx1)*0.5f, (mn2+mx2)*0.5f};

    // --- Lane 0: write state and allocate ---
    BtPoint32* points = NULL;
    if (lane == 0) {
        s->maxAxis = maxAx;
        s->minAxis = minAx;
        s->medAxis = medAx;
        s->scaling[0] = sc[0]; s->scaling[1] = sc[1]; s->scaling[2] = sc[2];
        s->center[0] = cen[0]; s->center[1] = cen[1]; s->center[2] = cen[2];
        points = (BtPoint32*)bt_alloc(s->wp, count * (int)sizeof(BtPoint32));
    }
    // Broadcast pointer from lane 0
    long long pp = __shfl_sync(WARP_MASK, (long long)points, 0);
    points = (BtPoint32*)pp;
    if (!points) return NULL;

    // --- Point conversion: warp-parallel strided ---
    for (int i = lane; i < count; i += WARP_SIZE) {
        float p[3];
        for (int k = 0; k < 3; k++)
            p[k] = (pts[i*3+k] - cen[k]) * inv[k];
        points[i].x = (int)p[medAx];
        points[i].y = (int)p[maxAx];
        points[i].z = (int)p[minAx];
        points[i].index = i;
    }
    __syncwarp();
    return points;
}

// Post-sort: init vertices (all lanes), parallel D&C + tree merge.
// 1. Split sorted points into 32 groups (one per warp lane).
// 2. Each lane independently builds a small hull of its group.
// 3. Tree merge: 5 rounds (16×2 → 8×4 → … → 1×32), each round's
//    independent merges run in parallel across warp lanes.
__device__ inline void bt_compute_postsort(BtHullState* s, BtPoint32* points, int count, int lane) {
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

    // Lane 0: allocate vblock
    BtVertex* vblock = NULL;
    if (lane == 0) {
        vblock = (BtVertex*)bt_alloc(shared_wp, count * (int)sizeof(BtVertex));
        s->vertexBase = vblock;
    }
    long long vb = __shfl_sync(WARP_MASK, (long long)vblock, 0);
    vblock = (BtVertex*)vb;
    if (!vblock) return;

    // All lanes: init vertices in parallel
    BtInt128 zero128 = bt128_from_u64(0);
    BtInt128 one128 = bt128_from_u64(1);
    for (int i = lane; i < count; i += WARP_SIZE) {
        BtVertex* v = &vblock[i];
        v->edges = NULL;
        v->next = NULL; v->prev = NULL;
        v->point = points[i];
        v->copy = -1;
        v->point128.x = zero128;
        v->point128.y = zero128;
        v->point128.z = zero128;
        v->point128.den = one128;
    }
    __syncwarp();

    if (lane == 0) s->npoints = count;

    // --- Allocate shared edge arena from scratch_heap (lane 0 only) ---
    __shared__ char* s_arena_base;
    __shared__ int   s_arena_offset;
    __shared__ int   s_arena_cap;
    __shared__ BtIntermediateHull s_hulls[WARP_SIZE];
    __shared__ int   s_postsort_error;
    __shared__ int   s_mergeStamp;

    int padded_edge = ((int)sizeof(BtEdge) + 3) & ~3;
    // Arena budget: initial slabs (32 lanes × initial_slab_edges) + growth headroom (16 × count edges).
    // Growth slabs use BTPOOL_BLOCK_SIZE but that's bounded by the 16×count term.
    int initial_slab_edges = 8 * count / WARP_SIZE;
    if (initial_slab_edges < 32) initial_slab_edges = 32;
    int arena_bytes = (WARP_SIZE * initial_slab_edges + 16 * count) * padded_edge;
    if (arena_bytes < 64 * 1024) arena_bytes = 64 * 1024;  // 64 KB minimum

    if (lane == 0) {
        s_postsort_error = 0;
        s_mergeStamp = -3;
        void* arena_ptr = NULL;
        if (heap_alloc(shared_sh, (unsigned int)arena_bytes, &arena_ptr) != HEAP_OK)
            s_postsort_error = BT_ERR_POOL_EXHAUST;
        s_arena_base = (char*)arena_ptr;
        s_arena_offset = 0;
        s_arena_cap = arena_bytes;
    }
    __syncwarp();
    if (s_postsort_error) { if (lane == 0) shared_wp->error = s_postsort_error; return; }

    // --- Per-lane group boundaries (with dedup at split points) ---
    // Like serial D&C's split logic, advance each split past runs of
    // equal points so that identical vertices are never split across
    // two lanes. Without this, degenerate sub-hulls at lane boundaries
    // trigger edge cases in bt_merge (pending chain use-after-free).
    __shared__ int s_splits[WARP_SIZE + 1];
    if (lane == 0) {
        s_splits[0] = 0;
        for (int g = 1; g < WARP_SIZE; g++) {
            int split = g * count / WARP_SIZE;
            // Advance past equal points (same logic as bt_computeInternal)
            BtPoint32 p = vblock[split - 1].point;
            while (split < count && bp32_eq(vblock[split].point, p)) split++;
            s_splits[g] = split;
        }
        s_splits[WARP_SIZE] = count;
    }
    __syncwarp();
    int my_start = s_splits[lane];
    int my_end   = s_splits[lane + 1];

    // --- Per-lane BtHullState with arena-backed edge pool ---
    BtHullState my_state;
    my_state.scaling[0] = sc0; my_state.scaling[1] = sc1; my_state.scaling[2] = sc2;
    my_state.center[0]  = cn0; my_state.center[1]  = cn1; my_state.center[2]  = cn2;
    my_state.vertexBase  = vblock;
    my_state.minAxis = minAx; my_state.medAxis = medAx; my_state.maxAxis = maxAx;
    my_state.wp           = shared_wp;
    my_state.scratch_heap = shared_sh;
    my_state.npoints      = count;
    my_state.usedEdgePairs = 0;
#ifdef TRACK_MAX_EDGE_PAIRS
    my_state.maxEdgePairs = 0;
#endif
    my_state.vertexList = NULL;
    my_state.fma_total_edges = 0;
    my_state.fma_min_edges   = 0x7fffffff;
    my_state.fma_max_edges   = 0;
    my_state.fma_calls       = 0;
    // All lanes share a single atomic mergeStamp counter so that edge stamps
    // are on a common timeline (required for cross-lane merges in tree merge).
    my_state.mergeStamp = -3;
    my_state.mergeStampPtr = &s_mergeStamp;

    // Init arena-backed edge pool (each lane gets an initial slab via atomicAdd)
    // initial_slab_edges must match the value used in arena_bytes calculation above.
    if (btpool_init_arena(&my_state.edgePool, (int)sizeof(BtEdge),
                          s_arena_base, &s_arena_offset, s_arena_cap,
                          initial_slab_edges) < 0) {
        // Arena exhausted during init — signal error
        s->wp->error = BT_ERR_POOL_EXHAUST;
    }
    __syncwarp();
    if (s->wp->error) return;

    //if (lane == 0) DPRINTF("[postsort] blk=%d n=%d arena=%d\n", blockIdx.x, count, arena_bytes);

#ifdef BT_SERIAL_MERGE
    // --- Serial D&C: only lane 0 builds full hull (for debugging) ---
    BtIntermediateHull my_hull;
    my_hull.minXy = NULL; my_hull.maxXy = NULL;
    my_hull.minYx = NULL; my_hull.maxYx = NULL;
    if (lane == 0 && count > 0) {
        bt_computeInternal(&my_state, 0, count, &my_hull);
    }
    s_hulls[0] = my_hull;
    __syncwarp();
    {
        int any_err = (lane == 0 && my_state.edgePool.error != 0) ? 1 : 0;
        any_err = __any_sync(WARP_MASK, any_err);
        if (any_err) { s->wp->error = BT_ERR_POOL_EXHAUST; return; }
    }
#else
    // --- Phase 1: Each lane builds hull of its group (parallel D&C) ---
    BtIntermediateHull my_hull;
    my_hull.minXy = NULL; my_hull.maxXy = NULL;
    my_hull.minYx = NULL; my_hull.maxYx = NULL;

    if (my_end - my_start > 0) {
        bt_computeInternal(&my_state, my_start, my_end, &my_hull);
    }
    s_hulls[lane] = my_hull;
    __syncwarp();

    // Check for errors from any lane
    {
        int any_err = (my_state.edgePool.error != 0) ? 1 : 0;
        any_err = __any_sync(WARP_MASK, any_err);
        if (any_err) { s->wp->error = BT_ERR_POOL_EXHAUST; return; }
    }

    // --- Phase 2: Tree merge (5 rounds) ---
    // Round r: 32>>>(r+1) independent merges, each handled by one lane.
    // Lane i merges s_hulls[i*stride] with s_hulls[i*stride + stride/2].
    for (int round = 0; round < 5; round++) {
        int stride = 1 << (round + 1);
        int merge_count = WARP_SIZE >> (round + 1);
        if (lane < merge_count) {
            int left  = lane * stride;
            int right = left + (stride >> 1);
            //DPRINTF("[tree] blk=%d lane=%d round=%d merge hulls[%d]+hulls[%d]\n",
            //        blockIdx.x, lane, round, left, right);
            bt_merge(&my_state, &s_hulls[left], &s_hulls[right]);
            // bt_merge writes result into s_hulls[left] via bt_mergeProjection
        }
        __syncwarp();

        // Check for errors (pool exhaustion during merge)
        {
            int any_err = (my_state.edgePool.error != 0) ? 1 : 0;
            any_err = __any_sync(WARP_MASK, any_err);
            if (any_err) { s->wp->error = BT_ERR_POOL_EXHAUST; return; }
        }
    }
#endif

    // --- Result is in s_hulls[0] ---
    if (lane == 0) {
        s->vertexList = s_hulls[0].minXy;
        // Copy edge pool info for cleanup (arena slabs freed by hull_dandc_warp_mesh)
        s->edgePool = my_state.edgePool;
        // Set mergeStamp from shared counter for use by extractMesh/computeVolume
        s->mergeStamp = s_mergeStamp;
        s->mergeStampPtr = NULL;  // extractMesh uses local stamp, not shared
        // Aggregate instrumentation from lane 0 (other lanes' stats are lost)
        s->fma_total_edges = my_state.fma_total_edges;
        s->fma_min_edges   = my_state.fma_min_edges;
        s->fma_max_edges   = my_state.fma_max_edges;
        s->fma_calls       = my_state.fma_calls;
    }
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
    __shared__ WarpPool s_pool;
    __shared__ void*    s_pool_backing;
    __shared__ Mesh     s_result;

    // Use a local error variable to avoid racing on *err with other blocks.
    // Only lane 0 writes; atomicOr to *err at the end.
    int local_err = 0;
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
            local_err = 1;
        }
    }
    __syncwarp();
    if (!s_pool_backing) {
        local_err = __shfl_sync(WARP_MASK, local_err, 0);
        if (lane == 0 && local_err) atomicOr(err, local_err);
        return s_result;
    }

    // --- Phase 1: pre-sort (all lanes) ---
    BtHullState state;
    if (lane == 0) {
        state.wp           = &s_pool;
        state.scratch_heap = scratch_heap;
        state.vertexList   = NULL;
        state.mergeStampPtr = NULL;
        state.edgePool.arena_base = NULL;
        state.edgePool.nblocks = 0;
        state.fma_total_edges = 0;
        state.fma_min_edges   = 0x7fffffff;
        state.fma_max_edges   = 0;
        state.fma_calls       = 0;
    }
    __syncwarp();

    BtPoint32* points = bt_compute_presort(&state, pts, n, lane);
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
        bt_compute_postsort(&state, points, n, lane);

        if (lane == 0) {
            if (s_pool.error || state.edgePool.error) {
                local_err = s_pool.error ? s_pool.error : state.edgePool.error;
                goto done;
            }

            DPRINTF("[hull] n=%d fma_calls=%d edges: total=%d avg=%.1f min=%d max=%d\n",
                n, state.fma_calls, state.fma_total_edges,
                state.fma_calls > 0 ? (float)state.fma_total_edges / state.fma_calls : 0.f,
                state.fma_min_edges == 0x7fffffff ? 0 : state.fma_min_edges,
                state.fma_max_edges);

            // Count pass: exact nv and nt without writing output
            int pre_count = s_pool.offset;
            int nv = 0, nt = 0;
            if (bt_extractMesh(&state, NULL, NULL, &nv, &nt) < 0)
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
                    { local_err = 1; goto done; }
                float* ov = (float*)chunk;
                int*   ot = (int*)((char*)chunk + va);
                int*   rc = (int*)((char*)chunk + va + ta);
                *rc = 1;

                // Extract pass
                int pre_ext = s_pool.offset;
                int nv2 = 0, nt2 = 0;
                if (bt_extractMesh(&state, ov, ot, &nv2, &nt2) < 0)
                    { local_err = 6; goto done; }
                bt_rewind(&s_pool, pre_ext);

                s_result.verts = ov; s_result.tris = ot;
                s_result.nv    = nv; s_result.nt   = nt;
                s_result.refcount = rc;
            }
        }
    }

done:
    // Cleanup scratch: free WarpPool backing and edge arena (lane 0).
    // Edge pool slabs are sub-ranges of the arena — free the arena as one chunk.
    if (lane == 0) {
        heap_free(scratch_heap, s_pool_backing);
        if (state.edgePool.arena_base)
            heap_free(scratch_heap, state.edgePool.arena_base);
    }
    __syncwarp();

    // Publish local error to the global error word (visible to host / other blocks).
    local_err = __shfl_sync(WARP_MASK, local_err, 0);
    if (lane == 0 && local_err) atomicOr(err, local_err);
    return s_result;
}
