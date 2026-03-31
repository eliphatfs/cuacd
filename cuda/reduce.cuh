// Block-level parallel reductions.
// Requires common.cuh for BLOCK_SIZE.

#ifndef REDUCE_CUH
#define REDUCE_CUH

// Shared-memory parallel reduction (sum).
__device__ inline float block_reduce_sum(float val, float* smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

// Shared-memory parallel reduction (max).
__device__ inline float block_reduce_max(float val, float* smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

// Shared-memory parallel reduction (count of nonzero flags).
// Each thread passes its flag (0 or 1). Returns total count.
__device__ inline int block_reduce_count(int flag, int* smem_i, int tid) {
    smem_i[tid] = flag;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem_i[tid] += smem_i[tid + s];
        __syncthreads();
    }
    return smem_i[0];
}

// Shared-memory parallel reduction for 3D bounding box.
// Computes per-axis min/max across all threads in the block.
__device__ inline void block_reduce_bbox(
    const float* verts, int n_verts, int vert_offset,
    int tid, float* smem,
    float* out_min, float* out_max)
{
    for (int axis = 0; axis < 3; axis++) {
        float lo = 1e30f, hi = -1e30f;
        for (int i = tid; i < n_verts; i += BLOCK_SIZE) {
            float v = verts[(vert_offset + i) * 3 + axis];
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
        // Reduce min
        smem[tid] = lo;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fminf(smem[tid], smem[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out_min[axis] = smem[0];
        // Reduce max
        smem[tid] = hi;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out_max[axis] = smem[0];
        __syncthreads();
    }
}

// ============================================================================
// Two-warp reductions for EXPANSION_BLOCK_SIZE (64 threads)
// ============================================================================

__device__ inline float twowarp_reduce_sum(float val, float* smem, int tid) {
    // Intra-warp reduction via shuffle
    for (int off = 16; off > 0; off >>= 1)
        val = val + __shfl_xor_sync(0xffffffffu, val, off);
    // Write warp result to smem
    if ((tid & 31) == 0) smem[tid >> 5] = val;
    __syncthreads();
    // Combine two warps
    float result = smem[0] + smem[1];
    __syncthreads();
    return result;
}

__device__ inline float twowarp_reduce_max(float val, float* smem, int tid) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));
    if ((tid & 31) == 0) smem[tid >> 5] = val;
    __syncthreads();
    float result = fmaxf(smem[0], smem[1]);
    __syncthreads();
    return result;
}

__device__ inline int twowarp_reduce_count(int flag, int* smem, int tid) {
    int val = flag;
    for (int off = 16; off > 0; off >>= 1)
        val = val + __shfl_xor_sync(0xffffffffu, val, off);
    if ((tid & 31) == 0) smem[tid >> 5] = val;
    __syncthreads();
    int result = smem[0] + smem[1];
    __syncthreads();
    return result;
}

#endif // REDUCE_CUH
