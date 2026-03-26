// Mesh coordinate transformation kernels: normalize to [-1,1]³ and recover.
// Requires: common.cuh for BLOCK_SIZE

// ============================================================================
// normalize_mesh — normalize vertices to [-1,1]³, store transform info
// ============================================================================

__global__ void normalize_mesh(
    float* __restrict__ vertices, int n_verts,
    float* __restrict__ norm_info)   // [4]: cx,cy,cz,scale
{
    int tid = threadIdx.x;
    extern __shared__ float smem_norm[];
    float* s_min = smem_norm;
    float* s_max = smem_norm + BLOCK_SIZE * 3;

    float lmin[3] = {1e30f, 1e30f, 1e30f};
    float lmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = tid; i < n_verts; i += BLOCK_SIZE)
        for (int k = 0; k < 3; k++) {
            float v = vertices[i*3+k];
            lmin[k] = fminf(lmin[k], v);
            lmax[k] = fmaxf(lmax[k], v);
        }
    for (int k = 0; k < 3; k++) {
        s_min[tid*3+k] = lmin[k];
        s_max[tid*3+k] = lmax[k];
    }
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) for (int k = 0; k < 3; k++) {
            s_min[tid*3+k] = fminf(s_min[tid*3+k], s_min[(tid+s)*3+k]);
            s_max[tid*3+k] = fmaxf(s_max[tid*3+k], s_max[(tid+s)*3+k]);
        }
        __syncthreads();
    }

    __shared__ float center[3], scale;
    if (tid == 0) {
        float range = 0.0f;
        for (int k = 0; k < 3; k++) {
            center[k] = (s_min[k] + s_max[k]) * 0.5f;
            float r = s_max[k] - s_min[k];
            if (r > range) range = r;
        }
        scale = (range > 1e-10f) ? (2.0f / range) : 1.0f;
        norm_info[0]=center[0]; norm_info[1]=center[1]; norm_info[2]=center[2];
        norm_info[3]=scale;
    }
    __syncthreads();
    for (int i = tid; i < n_verts; i += BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            vertices[i*3+k] = (vertices[i*3+k] - center[k]) * scale;
}

// ============================================================================
// recover_coordinates — inverse of normalize_mesh
// ============================================================================

__global__ void recover_coordinates(
    float* __restrict__ vertices, int n_verts,
    const float* __restrict__ norm_info)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_verts) return;
    float inv = 1.0f / norm_info[3];
    vertices[idx*3+0] = vertices[idx*3+0] * inv + norm_info[0];
    vertices[idx*3+1] = vertices[idx*3+1] * inv + norm_info[1];
    vertices[idx*3+2] = vertices[idx*3+2] * inv + norm_info[2];
}
