// Shared constants, data structures, and device utilities.
// Included by all kernel modules.
#pragma once
// ============================================================================
// Constants (must match beam.c)
// ============================================================================

#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

// ============================================================================
// Data structures (must match csrc/structs.h)
// ============================================================================

#include "structs.cuh"

// ============================================================================
// Pool allocator
// ============================================================================
// CRITICAL: only call from thread 0, broadcast pointer via __shared__ memory.
// See CLAUDE.md "Pool Allocator Pattern" for why.

__device__ inline void* pool_alloc(DevicePool* pool, unsigned int size) {
    unsigned long long aligned = (unsigned long long)((size + 15) & ~15);
    unsigned long long old = atomicAdd(pool->offset, aligned);
    if (old + aligned > pool->capacity) return NULL;
    return pool->base + old;
}

// Allocate from a DevicePool — thread 0 only, block-level use.
__device__ inline void* global_alloc_t0(DevicePool* gpool, int bytes) {
    unsigned long long aligned = (unsigned long long)((bytes + 15) & ~15);
    unsigned long long old = atomicAdd(gpool->offset, aligned);
    if (old + aligned > gpool->capacity) return NULL;
    return gpool->base + old;
}

// Allocate from a DevicePool — lane 0 only, result broadcast to warp.
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

// ============================================================================
// Atomic float min/max via CAS
// ============================================================================

__device__ inline float atomicMinF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fminf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ inline float atomicMaxF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fmaxf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}
