// C API for GPU kernels: hull mesh, mesh volume, plane cut, warp sort.
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

// ---------------------------------------------------------------------------
// Test: warp_sort — sort BtPoint32 sub-arrays in-place
// points: packed int[total_pts * 4] (x,y,z,index), modified in-place
// offsets: [n_arrays + 1]
// ---------------------------------------------------------------------------
int beam_test_warp_sort(beam_ctx_t ctx,
    int* points, int total_pts, const int* offsets, int n_arrays);

// ---------------------------------------------------------------------------
// hull_dandc — extract convex hull mesh for a batch of point clouds
// ---------------------------------------------------------------------------
// pts:            float[total_pts * 3]
// offsets:        int[n_hulls + 1]
// max_pts_per_hull: used to compute per-hull scratch size
// out_verts:      float[n_hulls * max_hull_verts * 3]
// out_tris:       int[n_hulls * max_hull_tris * 3]  (per-hull relative indices)
// out_nv, out_nt, out_errors: int[n_hulls]
int beam_hull_dandc(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_pts_per_hull,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_verts,
    int*         out_tris,
    int*         out_nv,
    int*         out_nt,
    int*         out_errors);

// ---------------------------------------------------------------------------
// test_mesh_volume — compute volume of a single mesh
// ---------------------------------------------------------------------------
int beam_test_mesh_volume(
    beam_ctx_t   ctx,
    const float* verts,
    int          n_verts,
    const int*   tris,
    int          n_tris,
    float*       out_volume);

// ---------------------------------------------------------------------------
// batch_mesh_volume — compute volume for a batch of watertight meshes
// ---------------------------------------------------------------------------
// verts:        float[total_verts * 3]
// tris:         int[total_tris * 3]  (per-mesh relative indices)
// tri_offsets:  int[n_meshes + 1]
// vert_offsets: int[n_meshes + 1] or NULL (treated as all-zero)
// out_volumes:  float[n_meshes]
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
// test_plane_cut — GPU plane cut with cap triangulation
// ---------------------------------------------------------------------------
// Returns separate pos and neg vertex + triangle arrays.
// Vertex indices in pos_tris are relative to pos_verts (same for neg).
//
// pa, pb, pc_n, pd: plane equation  pa*x + pb*y + pc_n*z + pd = 0
//   positive side: pa*x + pb*y + pc_n*z + pd > 0
int beam_test_plane_cut(
    beam_ctx_t   ctx,
    const float* vertices, int n_verts,
    const int*   triangles, int n_tris,
    float pa, float pb, float pc_n, float pd,
    float* out_pos_verts, int out_pos_verts_cap,
    int*   out_pos_tris,  int out_pos_tris_cap,
    float* out_neg_verts, int out_neg_verts_cap,
    int*   out_neg_tris,  int out_neg_tris_cap,
    int* out_n_pv, int* out_n_pt,
    int* out_n_nv, int* out_n_nt);

#ifdef __cplusplus
}
#endif

#endif // BEAM_H
