// C API for GPU kernels: hull volume, mesh volume, plane cut, warp sort.
// Uses CUDA driver API internally — no CUDA runtime dependency.
// Linked into a CPython extension module; not a standalone shared library.

#ifndef BEAM_H
#define BEAM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to GPU context
typedef struct beam_ctx* beam_ctx_t;

int  beam_init(beam_ctx_t* ctx, int device_ordinal);
void beam_destroy(beam_ctx_t ctx);

const char* beam_last_error(beam_ctx_t ctx);

// Test: warp_sort — sort BtPoint32 sub-arrays in-place
// points: packed int[total_pts * 4] (x,y,z,index), modified in-place
// offsets: [n_arrays + 1]
int beam_test_warp_sort(beam_ctx_t ctx,
    int* points, int total_pts, const int* offsets, int n_arrays);

// ---------------------------------------------------------------------------
// Batch hull volume (three algorithms)
// ---------------------------------------------------------------------------
// algo: 0=incremental (max 256 pts), 1=quickhull(warp), 2=dandc(warp)
// pts:     [total_pts * 3] float32  — all point clouds packed
// offsets: [n_hulls + 1]  int32    — point index boundaries
// max_pts_per_hull: used to compute per-warp scratch (algo 1/2 only)
// out_volumes: [n_hulls] float32 output
// out_errors:  [n_hulls] int32   0=ok 1=OOM 2=degenerate 3=too_large
int beam_batch_hull_volume(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          algo,
    int          max_pts_per_hull,
    float*       out_volumes,
    int*         out_errors);

// ---------------------------------------------------------------------------
// Batch mesh volume (divergence theorem on watertight meshes)
// ---------------------------------------------------------------------------
// verts:        [total_V * 3] float32 — packed vertex positions
// tris:         [total_T * 3] int32   — packed triangles, RELATIVE per-mesh indices
// tri_offsets:  [n_meshes + 1] int32  — triangle index boundaries
// vert_offsets: [n_meshes + 1] int32  — vertex index boundaries (for rebasing)
// out_volumes:  [n_meshes] float32 output
int beam_batch_mesh_volume(
    beam_ctx_t   ctx,
    const float* verts,
    int          total_verts,
    const int*   tris,
    int          total_tris,
    const int*   tri_offsets,
    const int*   vert_offsets,
    int          n_meshes,
    float*       out_volumes);

// ---------------------------------------------------------------------------
// batch_hull_dandc_mesh — D&C hull volume + mesh extraction
// ---------------------------------------------------------------------------
// out_verts: [n_hulls * max_hull_verts * 3] float32
// out_tris:  [n_hulls * max_hull_tris * 3]  int32
// out_vert_counts, out_tri_counts: [n_hulls] int32
int beam_batch_hull_dandc_mesh(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_pts_per_hull,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_volumes,
    int*         out_errors,
    float*       out_verts,
    int*         out_tris,
    int*         out_vert_counts,
    int*         out_tri_counts);

void beam_set_plane_cut_ctx(beam_ctx_t ctx);

int beam_test_plane_cut(
    const float* vertices, int n_verts,
    const int* triangles, int n_tris,
    float pa, float pb, float pc, float pd,
    float* out_all_verts, int out_verts_cap,
    int* out_pos_tris, int out_pos_cap,
    int* out_neg_tris, int out_neg_cap,
    int* out_n_verts,
    int* out_n_pos_tris,
    int* out_n_neg_tris);

#ifdef __cplusplus
}
#endif

#endif // BEAM_H
