// hull_warp_common.cuh — Shared utilities for warp-based convex hull algorithms.
//
// WarpPool bump allocator and warp-level reductions used by both
// QuickHull (hull_quickhull.cuh) and D&C (hull_dandc.cuh).
//
// Requires: common.cuh (EPS, signed_tet_volume from geometry.cuh)

#ifndef HULL_WARP_COMMON_CUH
#define HULL_WARP_COMMON_CUH

#define WARP_SIZE 32
#define WARP_MASK 0xffffffffu

// ============================================================================
// WarpPool — bump allocator for warp-scope scratch
// ============================================================================

struct WarpPool {
    char*         base;
    int           offset;
    int           capacity;
    int           error;      // 1 = OOM set by warp_pool_alloc
};

// Allocate `bytes` (16-byte aligned) from pool.
// Thread 0 bumps the pointer; result is broadcast to all lanes via __shfl_sync.
// Returns NULL on OOM (pool->error is set).
__device__ inline void* warp_pool_alloc(WarpPool* pool, int bytes, int lane) {
    long long ptr_ll = 0LL;
    if (lane == 0) {
        int aligned = (bytes + 15) & ~15;
        if (pool->offset + aligned > pool->capacity) {
            pool->error = 1;
            ptr_ll = 0LL;
        } else {
            ptr_ll = (long long)(pool->base + pool->offset);
            pool->offset += aligned;
        }
    }
    ptr_ll = __shfl_sync(WARP_MASK, ptr_ll, 0);
    return (void*)ptr_ll;
}

// ============================================================================
// Warp reductions
// ============================================================================

// Reduce maximum float value across all 32 lanes.
__device__ inline float warp_max_f(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, off));
    return val;
}

// Reduce minimum float value across all 32 lanes.
__device__ inline float warp_min_f(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fminf(val, __shfl_xor_sync(WARP_MASK, val, off));
    return val;
}

// OR reduction: returns nonzero iff any lane passes nonzero flag.
__device__ inline int warp_any_i(int flag) {
    return (int)__any_sync(WARP_MASK, flag);
}

// (value, index) argmax across all 32 lanes.
// Each lane passes its local (val, idx); after the call every lane sees the
// global winner.
__device__ inline void warp_argmax_f(float* val, int* idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float v2 = __shfl_xor_sync(WARP_MASK, *val, off);
        int   i2 = __shfl_xor_sync(WARP_MASK, *idx, off);
        if (v2 > *val) { *val = v2; *idx = i2; }
    }
}

// Broadcast a float from lane 0 to all lanes.
__device__ inline float warp_bcast_f(float val) {
    return __shfl_sync(WARP_MASK, val, 0);
}

// Broadcast an int from lane 0 to all lanes.
__device__ inline int warp_bcast_i(int val) {
    return __shfl_sync(WARP_MASK, val, 0);
}

#endif // HULL_WARP_COMMON_CUH
