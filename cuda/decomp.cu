// decomp.cu — Batch plane cut wrapper and mesh compaction kernels.
//
// Kernels:
//   batch_plane_cut      — 1 block per cut (PC_BLOCK=64 threads), calls plane_cut_block
//   batch_compact_mesh   — 1 block per mesh (BLOCK_SIZE threads), vertex dedup + bbox
//   batch_bbox           — 1 block per mesh (BLOCK_SIZE threads), bbox only
//
// plane_cut_block is defined in plane_cut.cu, which is #include'd before this in kernels.cu.

#include "common.cuh"
#include "reduce.cuh"
#include "plane_cut.cuh"

// ============================================================================
// compact_mesh_block — device function: vertex compaction for one mesh
// 1 block, BLOCK_SIZE threads.
// Removes unused vertices from a shared-vertex mesh (plane-cut output).
// Produces: compact_verts, remapped_tris, actual_n_verts, bbox[6].
// ============================================================================

__device__ void compact_mesh_block(
    const float* in_verts,   // [n_all_verts * 3] — original + intersection verts
    int           n_all_verts,
    const int*   in_tris,    // [n_tris * 3]
    int           n_tris,
    float*       out_verts,  // [n_all_verts * 3] pre-allocated (natural upper bound)
    int*         out_tris,   // [n_tris * 3]
    int*         out_n_verts,
    float*       out_bbox,   // [6]: xmin,xmax,ymin,ymax,zmin,zmax
    DevicePool   scratch,
    int*         kerr)
{
    int tid = threadIdx.x;

    __shared__ float s_smem[BLOCK_SIZE];
    __shared__ int* s_used;
    __shared__ int* s_remap;
    __shared__ int  s_n_verts;

    // Thread 0 allocates scratch: used[] and remap[]
    if (tid == 0) {
        s_used  = (int*)global_alloc_t0(&scratch, n_all_verts * (int)sizeof(int));
        s_remap = (int*)global_alloc_t0(&scratch, n_all_verts * (int)sizeof(int));
        if (!s_used || !s_remap) atomicOr(kerr, 1);
    }
    __syncthreads();
    if (*kerr) return;

    int* used  = s_used;
    int* remap = s_remap;

    // Zero used[]
    for (int i = tid; i < n_all_verts; i += blockDim.x)
        used[i] = 0;
    __syncthreads();

    // Mark used vertices from triangles (atomicOr is idempotent)
    for (int t = tid; t < n_tris; t += blockDim.x) {
        int a = in_tris[t*3+0], b = in_tris[t*3+1], c = in_tris[t*3+2];
        if (a >= 0 && a < n_all_verts) atomicOr(&used[a], 1);
        if (b >= 0 && b < n_all_verts) atomicOr(&used[b], 1);
        if (c >= 0 && c < n_all_verts) atomicOr(&used[c], 1);
    }
    __syncthreads();

    // Thread 0: sequential prefix sum → remap[]
    if (tid == 0) {
        int cnt = 0;
        for (int i = 0; i < n_all_verts; i++)
            remap[i] = used[i] ? cnt++ : -1;
        s_n_verts = cnt;
        *out_n_verts = cnt;
    }
    __syncthreads();

    int n_out = s_n_verts;

    // Compact vertices (parallel)
    for (int i = tid; i < n_all_verts; i += blockDim.x) {
        int ni = remap[i];
        if (ni >= 0) {
            out_verts[ni*3+0] = in_verts[i*3+0];
            out_verts[ni*3+1] = in_verts[i*3+1];
            out_verts[ni*3+2] = in_verts[i*3+2];
        }
    }
    __syncthreads();

    // Remap triangles (parallel)
    for (int t = tid; t < n_tris; t += blockDim.x) {
        out_tris[t*3+0] = remap[in_tris[t*3+0]];
        out_tris[t*3+1] = remap[in_tris[t*3+1]];
        out_tris[t*3+2] = remap[in_tris[t*3+2]];
    }
    __syncthreads();

    // Bbox
    if (n_out == 0) {
        if (tid == 0)
            for (int k = 0; k < 6; k++) out_bbox[k] = 0.f;
        return;
    }
    float lo[3], hi[3];
    block_reduce_bbox(out_verts, n_out, 0, tid, s_smem, lo, hi);
    if (tid == 0) {
        out_bbox[0] = lo[0]; out_bbox[1] = hi[0];
        out_bbox[2] = lo[1]; out_bbox[3] = hi[1];
        out_bbox[4] = lo[2]; out_bbox[5] = hi[2];
    }
}

// ============================================================================
// batch_compact_mesh — global kernel: 1 block per mesh (BLOCK_SIZE threads)
// ============================================================================

extern "C" __global__ void batch_compact_mesh(
    const float* in_verts_pool,
    const int*   in_tris_pool,
    const int*   in_vert_offsets,          // [n_meshes]
    const int*   in_vert_counts,           // [n_meshes]
    const int*   in_tri_offsets,           // [n_meshes]
    const int*   in_tri_counts,            // [n_meshes]
    float*       out_verts_pool,
    int*         out_tris_pool,
    const int*   out_vert_offsets,         // [n_meshes] — packed by natural bounds
    const int*   out_tri_offsets,          // [n_meshes]
    float*       out_bboxes,               // [n_meshes * 6]
    int*         out_n_verts,              // [n_meshes]
    int*         out_n_tris,              // [n_meshes]
    int*         kerrs,                    // [n_meshes]
    char*        scratch_pool,
    const unsigned int* scratch_byte_offsets, // [n_meshes]
    const unsigned int* scratch_caps,
    unsigned long long* scratch_ull_counters, // [n_meshes] — zeroed before launch
    int n_meshes)
{
    int mid = blockIdx.x;
    if (mid >= n_meshes) return;

    DevicePool dp;
    dp.base     = scratch_pool + (size_t)scratch_byte_offsets[mid];
    dp.offset   = &scratch_ull_counters[mid];
    dp.capacity = (unsigned long long)scratch_caps[mid];

    compact_mesh_block(
        in_verts_pool  + (long long)in_vert_offsets[mid] * 3,
        in_vert_counts[mid],
        in_tris_pool   + (long long)in_tri_offsets[mid]  * 3,
        in_tri_counts[mid],
        out_verts_pool + (long long)out_vert_offsets[mid] * 3,
        out_tris_pool  + (long long)out_tri_offsets[mid]  * 3,
        &out_n_verts[mid],
        out_bboxes + mid * 6,
        dp,
        &kerrs[mid]);

    if (threadIdx.x == 0)
        out_n_tris[mid] = in_tri_counts[mid];
}

// ============================================================================
// batch_bbox — 1 block per mesh (BLOCK_SIZE threads), compute bbox only
// ============================================================================

extern "C" __global__ void batch_bbox(
    const float* verts_pool,
    const int*   vert_offsets,   // [n_meshes]
    const int*   vert_counts,    // [n_meshes]
    float*       out_bboxes,     // [n_meshes * 6]: xmin,xmax,ymin,ymax,zmin,zmax
    int n_meshes)
{
    int mid = blockIdx.x;
    if (mid >= n_meshes) return;
    int tid = threadIdx.x;

    if (vert_counts[mid] == 0) {
        if (tid == 0)
            for (int k = 0; k < 6; k++) out_bboxes[mid*6+k] = 0.f;
        return;
    }

    __shared__ float smem[BLOCK_SIZE];
    float lo[3], hi[3];
    block_reduce_bbox(verts_pool, vert_counts[mid], vert_offsets[mid], tid, smem, lo, hi);
    if (tid == 0) {
        out_bboxes[mid*6+0] = lo[0]; out_bboxes[mid*6+1] = hi[0];
        out_bboxes[mid*6+2] = lo[1]; out_bboxes[mid*6+3] = hi[1];
        out_bboxes[mid*6+4] = lo[2]; out_bboxes[mid*6+5] = hi[2];
    }
}

// ============================================================================
// batch_plane_cut — 1 block per cut (PC_BLOCK=64 threads), calls plane_cut_block
// plane_cut_block is defined in plane_cut.cu (included before this file in kernels.cu)
// ============================================================================

extern "C" __global__ void batch_plane_cut(
    const float* verts_pool,
    const int*   tris_pool,
    const int*   cut_vert_offsets,   // [n_cuts]
    const int*   cut_vert_counts,
    const int*   cut_tri_offsets,
    const int*   cut_tri_counts,
    const float* plane_params,       // [n_cuts * 4]
    float*       out_verts_pool,
    int*         out_pos_pool,
    int*         out_neg_pool,
    const int*   out_vert_offsets,   // [n_cuts] in vertex units
    const int*   out_pos_offsets,    // [n_cuts] in triangle units
    const int*   out_neg_offsets,
    const int*   out_vert_caps,
    const int*   out_pos_caps,
    const int*   out_neg_caps,
    int*         out_n_verts,
    int*         out_n_pos,
    int*         out_n_neg,
    int*         kerrs,
    char*        scratch_pool,
    const unsigned int* scratch_byte_offsets, // [n_cuts]
    const unsigned int* scratch_caps,
    unsigned long long* scratch_ull_counters, // [n_cuts] — zeroed before launch
    int n_cuts)
{
    int cid = blockIdx.x;
    if (cid >= n_cuts) return;

    DevicePool dp;
    dp.base     = scratch_pool + (size_t)scratch_byte_offsets[cid];
    dp.offset   = &scratch_ull_counters[cid];
    dp.capacity = (unsigned long long)scratch_caps[cid];

    const float* verts = verts_pool + (long long)cut_vert_offsets[cid] * 3;
    const int*   tris  = tris_pool  + (long long)cut_tri_offsets[cid]  * 3;
    float pa = plane_params[cid*4+0], pb = plane_params[cid*4+1];
    float pc_n = plane_params[cid*4+2], pd = plane_params[cid*4+3];

    float* ov = out_verts_pool + (long long)out_vert_offsets[cid] * 3;
    int*   op = out_pos_pool   + (long long)out_pos_offsets[cid]  * 3;
    int*   on = out_neg_pool   + (long long)out_neg_offsets[cid]  * 3;

    plane_cut_block(verts, tris,
                    cut_vert_counts[cid], cut_tri_counts[cid],
                    pa, pb, pc_n, pd,
                    ov, op, on,
                    out_vert_caps[cid], out_pos_caps[cid], out_neg_caps[cid],
                    &out_n_verts[cid], &out_n_pos[cid], &out_n_neg[cid],
                    dp, &kerrs[cid]);
}
