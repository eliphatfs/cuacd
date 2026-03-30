// hull_batch.cu — Batch hull volume and mesh volume kernels.
//
// Kernels (one block or one warp per item):
//   batch_hull_incremental  — existing incremental hull, 1 block per hull (max 256 pts)
//   batch_hull_quickhull    — warp QuickHull, 1 warp per hull, scratch in global memory
//   batch_hull_dandc        — warp D&C (Preparata-Hong), 1 warp per hull
//   batch_mesh_volume       — divergence theorem on watertight mesh, 1 block per mesh
//
// Requires: common.cuh, geometry.cuh, hull.cuh, hull_warp.cuh

#include "hull_warp.cuh"

// ---- Scratch formula ----
// QuickHull: max_faces = 2*N+8. ~19 int/float arrays of max_faces, plus N-sized arrays.
// D&C: ~218*N bytes.  Use 256*N + 4096 bytes as a safe upper bound for both.
#define HULL_SCRATCH_PER_PT 256
#define HULL_SCRATCH_FIXED  4096

// ---------------------------------------------------------------------------
// batch_hull_incremental
// One BLOCK_SIZE-thread block per hull; shared-memory HullWorkspace.
// Caps at MAX_HULL_VERTS (256); sets error=3 if hull exceeds cap.
// ---------------------------------------------------------------------------
__global__ void batch_hull_incremental(
    const float* __restrict__ pts,       // packed [total_pts * 3]
    const int*   __restrict__ offsets,   // [n_hulls + 1] point indices
    float*       __restrict__ volumes,   // [n_hulls]
    int*         __restrict__ errors,    // [n_hulls]  0=ok 3=too_large
    int                       n_hulls)
{
    int hull_idx = blockIdx.x;
    if (hull_idx >= n_hulls) return;

    __shared__ HullWorkspace ws;
    int tid = threadIdx.x;

    int pt_start = offsets[hull_idx];
    int pt_count = offsets[hull_idx + 1] - pt_start;

    if (pt_count > MAX_HULL_VERTS) {
        if (tid == 0) { volumes[hull_idx] = -1.0f; errors[hull_idx] = 3; }
        __syncthreads();
        return;
    }

    float vol = compute_hull_volume(
        pts + (long long)pt_start * 3, pt_count, tid, &ws);

    if (tid == 0) {
        volumes[hull_idx] = vol;
        errors[hull_idx]  = 0;
    }
}

// ---------------------------------------------------------------------------
// batch_hull_quickhull
// Launch with blockDim.x = BLOCK_SIZE (256). Each warp handles one hull.
// 8 hulls per block.  scratch must be [n_hulls * scratch_per_hull] bytes.
// ---------------------------------------------------------------------------
__global__ void batch_hull_quickhull(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    float*       __restrict__ volumes,
    int*         __restrict__ errors,
    char*        __restrict__ scratch,
    int                       scratch_per_hull,
    int                       n_hulls)
{
    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_hulls) return;

    int pt_start = offsets[warp_id];
    int pt_count = offsets[warp_id + 1] - pt_start;

    // Broadcast scratch base address to all lanes
    long long base_ll = 0;
    if (lane == 0)
        base_ll = (long long)(scratch + (long long)warp_id * scratch_per_hull);
    base_ll = __shfl_sync(WARP_MASK, base_ll, 0);

    // Each lane holds its own WarpPool; only lane 0 ever modifies pool.offset
    WarpPool pool;
    pool.base     = (char*)base_ll;
    pool.offset   = 0;
    pool.capacity = scratch_per_hull;
    pool.error    = 0;

    int err = 0;
    float vol = hull_quickhull_warp(
        pts + (long long)pt_start * 3, pt_count, lane, &pool, &err);

    if (lane == 0) {
        volumes[warp_id] = vol;
        errors[warp_id]  = pool.error ? 1 : err;
    }
}

// ---------------------------------------------------------------------------
// batch_hull_dandc
// Uses block size 64 (2 warps/block) for better occupancy with deep stacks.
// ---------------------------------------------------------------------------
#define DANDC_BLOCK_SIZE 64

__global__ void batch_hull_dandc(
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
__global__ void query_dandc_scratch(int n, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *out = dandc_scratch_bytes(n);
}

// ---------------------------------------------------------------------------
// batch_mesh_volume
// One BLOCK_SIZE-thread block per mesh.
// Computes V = (1/6) |sum_t p0.(p1 x p2)| via parallel reduction.
// tris uses RELATIVE vertex indices within each mesh (0-based per mesh).
// Absolute vertex = vert_offsets[mesh] + relative_idx.
// ---------------------------------------------------------------------------
__global__ void batch_mesh_volume(
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
