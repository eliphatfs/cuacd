// Shared constants and device utilities.
// Included by all kernel modules.
#pragma once
#include "allocator.cuh"

// ============================================================================
// Constants (must match beam.c)
// ============================================================================

#define BLOCK_SIZE 256
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

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
