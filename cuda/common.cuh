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

// ============================================================================
// New beam search data structures (V2)
// ============================================================================

#define EXPANSION_BLOCK_SIZE 64
#define HAUSDORFF_BLOCK_SIZE 256
#define MAX_BEAM_V2 64

struct PartInfoV2 {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    int hull_vert_offset, hull_vert_count;
    int hull_tri_offset, hull_tri_count;
    float bbox[6];       // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;
    float hausdorff;     // -1 = not yet computed
    float mesh_volume;
    float hull_volume;
};

struct WorkItem {
    int part_indices[MAX_PARTS_PER_BEAM];  // indirect into PartInfoV2 array
    int num_parts;
    int worst_part_idx;    // index into part_indices
    float worst_metric;    // max(rv, hausdorff) of worst part
};

struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
};

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
