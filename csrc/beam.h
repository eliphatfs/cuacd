// C API for GPU-accelerated convex decomposition.
// Provides beam search decomposition + Hausdorff distance computation.
// Uses CUDA driver API internally — no CUDA runtime dependency.
// Linked into a CPython extension module; not a standalone shared library.

#ifndef BEAM_H
#define BEAM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to beam search context
typedef struct beam_ctx* beam_ctx_t;

// Parameters for beam search decomposition
typedef struct beam_params {
    int   beam_width;       // Number of beam items to keep (default 8)
    int   cuts_per_axis;    // Planes per axis direction, 3*N total (default 10)
    float threshold;        // Concavity threshold for termination (default 0.05)
    float rv_k;             // Scaling factor in Rv formula (default 0.3)
    int   max_parts;        // Max parts per beam item (default 64)
    int   max_iterations;   // Max decomposition iterations (default 64)
    int   hausdorff_samples;// Surface samples for Hausdorff check (default 1000)
} beam_params_t;

void beam_params_default(beam_params_t* params);
int  beam_init(beam_ctx_t* ctx, int device_ordinal);
void beam_destroy(beam_ctx_t ctx);

int  beam_get_num_parts(beam_ctx_t ctx);

int beam_get_part(
    beam_ctx_t ctx,
    int        part_idx,
    float*     out_vertices,
    int*       out_n_verts,
    int*       out_triangles,
    int*       out_n_tris);

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

// ---------------------------------------------------------------------------
// CPU plane cut with cap triangulation
// ---------------------------------------------------------------------------
// Cuts mesh by plane pa*x + pb*y + pc*z + pd = 0.
// Produces two closed meshes (positive and negative halves) with properly
// triangulated caps. Handles simple loop, ring, and multi-hole topologies.
// Pure CPU implementation — no GPU required.
//
// out_all_verts: [out_verts_cap * 3] float — original + intersection vertices
// out_pos_tris/out_neg_tris: [cap * 3] int — triangle indices into out_all_verts
// Returns 0 on success, -1 on error (buffer overflow).
// ---------------------------------------------------------------------------
// Batch GPU plane cut (decomp kernels)
// ---------------------------------------------------------------------------
// Cuts n_cuts meshes with their respective planes.
// Input meshes are packed in verts/tris with per-cut offsets.
// Output (all pre-allocated by caller, packed with per-cut offsets):
//   out_verts:    shared vert pool [total_out_verts * 3] float32
//   out_pos_tris: pos-half tris   [total_out_pos * 3]   int32
//   out_neg_tris: neg-half tris   [total_out_neg * 3]   int32
//   out_n_verts/pos/neg: actual counts per cut [n_cuts] int32
int beam_batch_plane_cut(
    beam_ctx_t   ctx,
    const float* verts,  int total_verts,
    const int*   tris,   int total_tris,
    const int*   cut_vert_offsets, const int* cut_vert_counts,
    const int*   cut_tri_offsets,  const int* cut_tri_counts,
    const float* plane_params,     // [n_cuts * 4]
    int          n_cuts,
    const int*   out_vert_offsets, const int* out_pos_offsets, const int* out_neg_offsets,
    const int*   out_vert_caps,    const int* out_pos_caps,    const int* out_neg_caps,
    int total_out_verts, int total_out_pos, int total_out_neg,
    float* out_verts, int* out_pos_tris, int* out_neg_tris,
    int* out_n_verts, int* out_n_pos, int* out_n_neg);

// ---------------------------------------------------------------------------
// Batch compact mesh (vertex deduplication after plane cut)
// ---------------------------------------------------------------------------
// For n_meshes half-meshes: compact shared-vert format to used-verts only.
// in_vert_counts[i]: n_all_verts (shared pool size) for mesh i.
// in_tri_counts[i]:  actual triangle count for mesh i.
// out_vert_offsets: pre-computed prefix sums (natural bound = in_vert_counts).
// out_tri_offsets:  pre-computed prefix sums (= in_tri_counts).
// out_n_verts: actual compact vert count per mesh.
// out_n_tris:  = in_tri_counts (tris count unchanged).
// out_bboxes: [n_meshes * 6] xmin,xmax,ymin,ymax,zmin,zmax.
int beam_batch_compact_mesh(
    beam_ctx_t   ctx,
    const float* in_verts, int total_in_verts,
    const int*   in_tris,  int total_in_tris,
    const int*   in_vert_offsets, const int* in_vert_counts,
    const int*   in_tri_offsets,  const int* in_tri_counts,
    int          n_meshes,
    const int*   out_vert_offsets, const int* out_tri_offsets,
    int total_out_verts, int total_out_tris,
    float* out_verts, int* out_tris,
    float* out_bboxes, int* out_n_verts, int* out_n_tris);

// ---------------------------------------------------------------------------
// Batch bbox
// ---------------------------------------------------------------------------
int beam_batch_bbox(
    beam_ctx_t   ctx,
    const float* verts, int total_verts,
    const int*   vert_offsets, const int* vert_counts,
    int n_meshes,
    float* out_bboxes);  // [n_meshes * 6]

// Set the context for plane_cut (avoids API change to beam_test_plane_cut)
// ---------------------------------------------------------------------------
// beam_decompose — GPU-resident beam-search convex decomposition
// ---------------------------------------------------------------------------
// Decomposes input mesh into approximately convex parts using beam search.
// Results stored in ctx and retrieved via beam_get_num_parts / beam_get_part.
// Returns 0 on success, non-zero on error (check beam_last_error).
int beam_decompose(
    beam_ctx_t          ctx,
    const float*        verts,   // [n_verts * 3] float32
    int                 n_verts,
    const int*          tris,    // [n_tris * 3] int32, 0-based
    int                 n_tris,
    const beam_params_t* params);

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
