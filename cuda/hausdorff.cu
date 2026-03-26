// Hausdorff distance kernels: surface sampling, point-mesh distance,
// max reduction, pairwise merge cost.
// Requires: common.cuh, geometry.cuh

// ============================================================================
// sample_surface — area-weighted surface sampling via xorshift RNG
// ============================================================================

__global__ void sample_surface(
    const float* __restrict__ vertices, const int* __restrict__ triangles,
    int n_tris, int vert_offset,
    float* __restrict__ samples, int n_samples, unsigned int seed)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_samples) return;

    unsigned int st = seed ^ (idx * 2654435761u);
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    int tri = st % n_tris;
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    int i0 = triangles[tri*3] + vert_offset;
    int i1 = triangles[tri*3+1] + vert_offset;
    int i2 = triangles[tri*3+2] + vert_offset;
    float u = (float)(st & 0xFFFF) / 65535.0f;
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    float v = (float)(st & 0xFFFF) / 65535.0f;
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    float w = 1.0f - u - v;

    samples[idx*3+0] = w*vertices[i0*3+0] + u*vertices[i1*3+0] + v*vertices[i2*3+0];
    samples[idx*3+1] = w*vertices[i0*3+1] + u*vertices[i1*3+1] + v*vertices[i2*3+1];
    samples[idx*3+2] = w*vertices[i0*3+2] + u*vertices[i1*3+2] + v*vertices[i2*3+2];
}

// ============================================================================
// point_mesh_distance — brute-force min distance, one thread per point
// ============================================================================

__global__ void point_mesh_distance(
    const float* __restrict__ points,
    const float* __restrict__ vertices,
    const int*   __restrict__ triangles,
    float*       __restrict__ distances,
    int N, int T, int vert_offset)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N) return;

    float px = points[idx*3], py = points[idx*3+1], pz = points[idx*3+2];
    float md = 1e30f;
    for (int t = 0; t < T; t++) {
        int i0 = triangles[t*3]   + vert_offset;
        int i1 = triangles[t*3+1] + vert_offset;
        int i2 = triangles[t*3+2] + vert_offset;
        md = fminf(md, point_triangle_dist(px, py, pz,
            vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2],
            vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2],
            vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]));
    }
    distances[idx] = md;
}

// ============================================================================
// reduce_max — shared-memory parallel max reduction (two-stage)
// ============================================================================

__global__ void reduce_max(
    const float* __restrict__ data,
    float*       __restrict__ output,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = -1e30f;
    if (idx < N) val = data[idx];
    if (idx + blockDim.x < N) val = fmaxf(val, data[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// pairwise_hausdorff — merge cost matrix
// ============================================================================
// Grid: (n_pairs, sample_blocks). Each block handles one sample slice for one
// (i,j) pair, computes min dist to the other part's mesh, atomically updates
// the max into cost_matrix[i*P+j].

__global__ void pairwise_hausdorff(
    const float* __restrict__ all_samples,
    const int*   __restrict__ sample_offsets, // [P+1]
    const float* __restrict__ all_vertices,
    const int*   __restrict__ all_triangles,
    const int*   __restrict__ tri_offsets,    // [P+1]
    const int*   __restrict__ vert_offsets,   // [P+1]
    float*       __restrict__ cost_matrix,    // [P*P]
    int P)
{
    int pair_idx = blockIdx.x;
    int sample_local = blockIdx.y * blockDim.x + threadIdx.x;

    // Map flat pair_idx to (i,j) with i > j
    int i = (int)(0.5f + sqrtf(0.25f + 2.0f * (float)pair_idx));
    while (i * (i - 1) / 2 > pair_idx) i--;
    while ((i + 1) * i / 2 <= pair_idx) i++;
    int j = pair_idx - i * (i - 1) / 2;
    if (i >= P || j >= P || i <= j) return;

    int ns_i = sample_offsets[i + 1] - sample_offsets[i];
    int ns_j = sample_offsets[j + 1] - sample_offsets[j];

    // Samples from part i → mesh of part j
    if (sample_local < ns_i) {
        int si = sample_offsets[i] + sample_local;
        float px = all_samples[si*3], py = all_samples[si*3+1], pz = all_samples[si*3+2];
        int ts = tri_offsets[j], te = tri_offsets[j+1], vo = vert_offsets[j];
        float md = 1e30f;
        for (int t = ts; t < te; t++) {
            int i0 = all_triangles[t*3]+vo, i1 = all_triangles[t*3+1]+vo, i2 = all_triangles[t*3+2]+vo;
            float d = point_triangle_dist(px, py, pz,
                all_vertices[i0*3], all_vertices[i0*3+1], all_vertices[i0*3+2],
                all_vertices[i1*3], all_vertices[i1*3+1], all_vertices[i1*3+2],
                all_vertices[i2*3], all_vertices[i2*3+1], all_vertices[i2*3+2]);
            md = fminf(md, d);
        }
        atomicMaxF(&cost_matrix[i*P+j], md);
    }

    // Samples from part j → mesh of part i
    if (sample_local < ns_j) {
        int si = sample_offsets[j] + sample_local;
        float px = all_samples[si*3], py = all_samples[si*3+1], pz = all_samples[si*3+2];
        int ts = tri_offsets[i], te = tri_offsets[i+1], vo = vert_offsets[i];
        float md = 1e30f;
        for (int t = ts; t < te; t++) {
            int i0 = all_triangles[t*3]+vo, i1 = all_triangles[t*3+1]+vo, i2 = all_triangles[t*3+2]+vo;
            float d = point_triangle_dist(px, py, pz,
                all_vertices[i0*3], all_vertices[i0*3+1], all_vertices[i0*3+2],
                all_vertices[i1*3], all_vertices[i1*3+1], all_vertices[i1*3+2],
                all_vertices[i2*3], all_vertices[i2*3+1], all_vertices[i2*3+2]);
            md = fminf(md, d);
        }
        atomicMaxF(&cost_matrix[i*P+j], md);
    }
}
