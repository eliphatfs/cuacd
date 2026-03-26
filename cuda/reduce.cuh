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

#endif // REDUCE_CUH
