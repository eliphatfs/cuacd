// allocator.cuh — DevicePool bump allocator.
//
// DevicePool is a global bump allocator backed by a pre-allocated device buffer.
// All alloc functions are 16-byte aligned.
//
// Usage rules:
//   pool_alloc        — thread 0 only (block scope); broadcast via __shared__ + __syncthreads__
//   global_alloc_warp — lane 0 only; result broadcast to warp via __shfl_sync
//
// See CLAUDE.md "Pool Allocator Pattern" for details.
#pragma once

// ============================================================================
// DevicePool (must match csrc/structs.h)
// ============================================================================
struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
};

// ============================================================================
// Allocators
// ============================================================================

// Thread-0 / block-scope allocation.
// CRITICAL: call from thread 0 only; store result in __shared__, __syncthreads__ before use.
__device__ inline void* pool_alloc(DevicePool* pool, unsigned int size) {
    unsigned long long aligned = (unsigned long long)((size + 15) & ~15);
    unsigned long long old = atomicAdd(pool->offset, aligned);
    if (old + aligned > pool->capacity) return NULL;
    return pool->base + old;
}

// Lane-0 / warp-scope allocation. Result is broadcast to all lanes via __shfl_sync.
__device__ inline void* global_alloc_warp(DevicePool* gpool, int bytes, int lane) {
    long long ptr_ll = 0LL;
    if (lane == 0) {
        unsigned long long aligned = (unsigned long long)((bytes + 15) & ~15);
        unsigned long long old = atomicAdd(gpool->offset, aligned);
        if (old + aligned <= gpool->capacity)
            ptr_ll = (long long)(gpool->base + old);
    }
    ptr_ll = __shfl_sync(0xffffffffu, ptr_ll, 0);
    return (void*)ptr_ll;
}
