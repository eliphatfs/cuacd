// test_hull_dandc.cu — Test/diagnostic kernels for D&C hull.
//
// Kernels:
//   batch_hull_dandc_mesh — D&C hull volume + mesh extraction (used by tests)

#include "hull_warp_common.cuh"
#include "hull_dandc.cuh"

// ---------------------------------------------------------------------------
// batch_hull_dandc_mesh
// Same as batch_hull_dandc but also extracts the convex hull mesh (vertices + triangles).
// Each hull writes its output into a pre-allocated slice of out_verts and out_tris.
//   out_verts[hull_id * max_hull_verts * 3 ... (hull_id+1)*max_hull_verts*3 - 1]
//   out_tris [hull_id * max_hull_tris  * 3 ... (hull_id+1)*max_hull_tris *3 - 1]
// out_vert_counts[hull_id] and out_tri_counts[hull_id] hold the actual counts.
// ---------------------------------------------------------------------------
#define DANDC_BLOCK_SIZE 32

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
