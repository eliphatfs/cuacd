// hull_batch.cu — Batch hull volume and mesh volume kernels.
//
// Kernels (one block or one warp per item):
//   batch_hull_dandc        — warp D&C (Preparata-Hong), 1 warp per hull
//   batch_mesh_volume       — divergence theorem on watertight mesh, 1 block per mesh

#include "hull_warp_common.cuh"
#include "hull_dandc.cuh"

// ---------------------------------------------------------------------------
// batch_hull_dandc
// Uses 1 warp per block (32 threads) — register pressure is the occupancy bottleneck.
// ---------------------------------------------------------------------------
#define DANDC_BLOCK_SIZE 32

extern "C" __global__ void batch_hull_dandc(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    float*       __restrict__ volumes,
    int*         __restrict__ errors,
    char*        __restrict__ scratch,
    int                       scratch_per_hull,
    int                       n_hulls)
{
    int warps_per_block = DANDC_BLOCK_SIZE / WARP_SIZE;
    int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_hulls) return;

    int pt_start = offsets[warp_id];
    int pt_count = offsets[warp_id + 1] - pt_start;

    long long base_ll = 0;
    if (lane == 0)
        base_ll = (long long)(scratch + (long long)warp_id * scratch_per_hull);
    base_ll = __shfl_sync(WARP_MASK, base_ll, 0);

    WarpPool pool;
    pool.base     = (char*)base_ll;
    pool.offset   = 0;
    pool.capacity = scratch_per_hull;
    pool.error    = 0;

    int err = 0;
    float vol = hull_dandc_warp(
        pts + (long long)pt_start * 3, pt_count, lane, &pool, &err);

    if (lane == 0) {
        volumes[warp_id] = vol;
        errors[warp_id]  = pool.error ? pool.error : err;
    }
}

// ---------------------------------------------------------------------------
// Query kernel: write dandc_scratch_bytes(n) to output.
// ---------------------------------------------------------------------------
extern "C" __global__ void query_dandc_scratch(int n, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *out = dandc_scratch_bytes(n);
}

// ---------------------------------------------------------------------------
// batch_hull_dandc_mesh
// Same as batch_hull_dandc but also extracts the convex hull mesh (vertices + triangles).
// Each hull writes its output into a pre-allocated slice of out_verts and out_tris.
//   out_verts[hull_id * max_hull_verts * 3 ... (hull_id+1)*max_hull_verts*3 - 1]
//   out_tris [hull_id * max_hull_tris  * 3 ... (hull_id+1)*max_hull_tris *3 - 1]
// out_vert_counts[hull_id] and out_tri_counts[hull_id] hold the actual counts.
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_hull_dandc_mesh(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    float*       __restrict__ volumes,
    int*         __restrict__ errors,
    float*       __restrict__ out_verts,       // [n_hulls * max_hull_verts * 3]
    int*         __restrict__ out_tris,        // [n_hulls * max_hull_tris * 3]
    int*         __restrict__ out_vert_counts, // [n_hulls]
    int*         __restrict__ out_tri_counts,  // [n_hulls]
    char*        __restrict__ scratch,
    int                       scratch_per_hull,
    int                       n_hulls,
    int                       max_hull_verts,
    int                       max_hull_tris)
{
    int warps_per_block = DANDC_BLOCK_SIZE / WARP_SIZE;
    int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_hulls) return;

    int pt_start = offsets[warp_id];
    int pt_count = offsets[warp_id + 1] - pt_start;

    long long base_ll = 0;
    if (lane == 0)
        base_ll = (long long)(scratch + (long long)warp_id * scratch_per_hull);
    base_ll = __shfl_sync(WARP_MASK, base_ll, 0);

    WarpPool pool;
    pool.base     = (char*)base_ll;
    pool.offset   = 0;
    pool.capacity = scratch_per_hull;
    pool.error    = 0;

    float* hull_verts = out_verts + (long long)warp_id * max_hull_verts * 3;
    int*   hull_tris  = out_tris  + (long long)warp_id * max_hull_tris  * 3;
    int n_hv = 0, n_ht = 0;

    int err = 0;
    float vol = hull_dandc_warp_mesh(
        pts + (long long)pt_start * 3, pt_count, lane, &pool, &err,
        hull_verts, hull_tris,
        max_hull_verts, max_hull_tris,
        &n_hv, &n_ht);

    if (lane == 0) {
        volumes[warp_id]          = vol;
        errors[warp_id]           = pool.error ? pool.error : err;
        out_vert_counts[warp_id]  = n_hv;
        out_tri_counts[warp_id]   = n_ht;
    }
}

// ---------------------------------------------------------------------------
// batch_mesh_volume
// One BLOCK_SIZE-thread block per mesh.
// Computes V = (1/6) |sum_t p0.(p1 x p2)| via parallel reduction.
// tris uses RELATIVE vertex indices within each mesh (0-based per mesh).
// Absolute vertex = vert_offsets[mesh] + relative_idx.
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_mesh_volume(
    const float* __restrict__ verts,        // packed [total_V * 3]
    const int*   __restrict__ tris,         // packed [total_T * 3] relative indices
    const int*   __restrict__ tri_offsets,  // [n_meshes + 1]
    const int*   __restrict__ vert_offsets, // [n_meshes + 1] (or NULL for no rebasing)
    float*       __restrict__ volumes,      // [n_meshes]
    int                       n_meshes)
{
    int mid = blockIdx.x;
    if (mid >= n_meshes) return;
    int tid = threadIdx.x;

    int t_start = tri_offsets[mid];
    int t_count = tri_offsets[mid + 1] - t_start;
    int v_base  = vert_offsets ? vert_offsets[mid] : 0;

    __shared__ float smem[BLOCK_SIZE];

    float local_sum = 0.0f;
    for (int t = tid; t < t_count; t += BLOCK_SIZE) {
        int ai = v_base + tris[(t_start + t) * 3 + 0];
        int bi = v_base + tris[(t_start + t) * 3 + 1];
        int ci = v_base + tris[(t_start + t) * 3 + 2];
        local_sum += signed_tet_volume(
            verts[ai*3], verts[ai*3+1], verts[ai*3+2],
            verts[bi*3], verts[bi*3+1], verts[bi*3+2],
            verts[ci*3], verts[ci*3+1], verts[ci*3+2]);
    }

    // Block reduce sum
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) volumes[mid] = fabsf(smem[0]);
}
