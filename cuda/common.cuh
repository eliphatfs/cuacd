// Shared constants, data structures, and device utilities.
// Included by all kernel modules.

#ifndef COMMON_CUH
#define COMMON_CUH

// ============================================================================
// Constants (must match beam.c)
// ============================================================================

#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

// ============================================================================
// Data structures (must match beam.c)
// ============================================================================

struct PartInfo {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    float bbox[6];    // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;
};

struct BeamItem {
    int num_parts;
    int worst_part_idx;
    float worst_cost;
    int cut_count;
};

struct DevicePool {
    char*         base;
    unsigned int* offset;
    unsigned int  capacity;
};

// ============================================================================
// Pool allocator
// ============================================================================
// CRITICAL: only call from thread 0, broadcast pointer via __shared__ memory.
// See CLAUDE.md "Pool Allocator Pattern" for why.

__device__ inline void* pool_alloc(DevicePool* pool, unsigned int size) {
    size = (size + 15) & ~15;
    unsigned int old = atomicAdd(pool->offset, size);
    if (old + size > pool->capacity) return NULL;
    return pool->base + old;
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

#endif // COMMON_CUH
