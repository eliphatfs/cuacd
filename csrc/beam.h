// C API for GPU-accelerated convex decomposition.
// Provides beam search decomposition + Hausdorff distance computation.
// Uses CUDA driver API internally — no CUDA runtime dependency.
// Linked into a CPython extension module; not a standalone shared library.

#ifndef BEAM_H
#define BEAM_H

#include <stdint.h>

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

int beam_run(
    beam_ctx_t           ctx,
    const float*         vertices,
    int                  n_verts,
    const int*           triangles,
    int                  n_tris,
    const beam_params_t* params);

int  beam_get_num_parts(beam_ctx_t ctx);

int beam_get_part(
    beam_ctx_t ctx,
    int        part_idx,
    float*     out_vertices,
    int*       out_n_verts,
    int*       out_triangles,
    int*       out_n_tris);

const char* beam_last_error(beam_ctx_t ctx);

// -----------------------------------------------------------------------
// Hausdorff distance computation
// -----------------------------------------------------------------------

// Compute per-point minimum distance from points to a triangle mesh.
// All pointers are HOST memory. The library handles upload/download.
int beam_point_mesh_distances(
    beam_ctx_t      ctx,
    const float*    points,     // [n_points, 3]
    int             n_points,
    const float*    vertices,   // [n_verts, 3]
    int             n_verts,
    const int*      triangles,  // [n_tris, 3]
    int             n_tris,
    float*          distances); // [n_points] output

// Compute Hausdorff distance between two sampled meshes.
int beam_hausdorff(
    beam_ctx_t      ctx,
    const float*    samples_a,    int n_samples_a,
    const float*    vertices_a,   int n_verts_a,
    const int*      triangles_a,  int n_tris_a,
    const float*    samples_b,    int n_samples_b,
    const float*    vertices_b,   int n_verts_b,
    const int*      triangles_b,  int n_tris_b,
    float*          result);

// Compute pairwise Hausdorff cost matrix for merge phase.
// Parts' data is packed contiguously with prefix-sum offset arrays.
int beam_pairwise_hausdorff(
    beam_ctx_t      ctx,
    const float*    all_samples,
    const int*      sample_offsets,  // [n_parts + 1]
    const float*    all_vertices,
    const int*      all_triangles,
    const int*      tri_offsets,     // [n_parts + 1]
    const int*      vert_offsets,    // [n_parts + 1]
    int             n_parts,
    float*          cost_matrix);    // [n_parts * n_parts] output

// Diagnostic: test hull volume computation directly
int beam_test_hull_volume(beam_ctx_t ctx,
                          const float* points, int n_points,
                          float* out_volume, int* out_n_faces);

// ---------------------------------------------------------------------------
// Batch test functions for scalar device functions
// ---------------------------------------------------------------------------

int beam_batch_signed_tet_volume(beam_ctx_t ctx,
    const float* tets, int n, float* out_volumes);

int beam_batch_point_triangle_dist(beam_ctx_t ctx,
    const float* points, const float* triangles, int n, float* out_dists);

int beam_batch_intersect_edge(beam_ctx_t ctx,
    const float* segments, const float* planes, int n, float* out_results);

int beam_batch_rv_from_volumes(beam_ctx_t ctx,
    const float* mesh_vols, const float* hull_vols, int n,
    float rv_k, float* out_rvs);

#ifdef __cplusplus
}
#endif

#endif // BEAM_H
