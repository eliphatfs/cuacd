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

// Get per-part diagnostic info (rv_cost, hausdorff, mesh_volume, hull_volume).
// Returns 0 on success, -1 if part_idx is out of range or no info available.
int beam_get_part_info(
    beam_ctx_t ctx,
    int        part_idx,
    float*     rv_cost,
    float*     hausdorff,
    float*     mesh_volume,
    float*     hull_volume);

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
// New V2 constants and structs (mirror of common.cuh V2 additions)
// ---------------------------------------------------------------------------

#define EXPANSION_BLOCK_SIZE 64
#define HAUSDORFF_BLOCK_SIZE 256
#define MAX_BEAM_V2 64
#ifndef MAX_PARTS_PER_BEAM
#define MAX_PARTS_PER_BEAM 64
#endif

typedef struct PartInfoV2 {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    int hull_vert_offset, hull_vert_count;
    int hull_tri_offset, hull_tri_count;
    float bbox[6];       // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;
    float hausdorff;     // -1 = not yet computed
    float mesh_volume;
    float hull_volume;
} PartInfoV2;

typedef struct WorkItem {
    int part_indices[MAX_PARTS_PER_BEAM];  // indirect into PartInfoV2 array
    int num_parts;
    int worst_part_idx;    // index into part_indices
    float worst_metric;    // max(rv, hausdorff) of worst part
} WorkItem;

// ---------------------------------------------------------------------------
// V2 beam search — 3-kernel architecture
// ---------------------------------------------------------------------------
// hull_verts/hull_tris: initial convex hull from scipy (CPU-side)
// hull_volume: volume of initial hull from scipy
// scratch_size: GPU scratch pool size (0 = auto)
int beam_run_v2(
    beam_ctx_t           ctx,
    const float*         vertices,
    int                  n_verts,
    const int*           triangles,
    int                  n_tris,
    const float*         hull_verts,
    int                  n_hull_verts,
    const int*           hull_tris,
    int                  n_hull_tris,
    float                hull_volume,
    size_t               scratch_size,
    const beam_params_t* params);

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
// Per-kernel test functions for V2 beam search
// ---------------------------------------------------------------------------

// Test beam_termination kernel in isolation.
// parts: [n_parts] PartInfoV2 array (host)
// work_items: [n_items] WorkItem array (host, modified in-place with worst_part_idx/worst_metric)
// Returns: index of terminated work item (or -1 if none).
int beam_test_termination(
    beam_ctx_t       ctx,
    PartInfoV2*      parts,
    int              n_parts,
    WorkItem*        work_items,
    int              n_items,
    float            threshold);

// Test beam_hausdorff_parts kernel in isolation.
// vertex_pool, triangle_pool: packed pools (host)
// parts: [n_parts] PartInfoV2 array (host, hausdorff field updated in-place)
// part_indices: [n_indices] indices into parts to process
int beam_test_hausdorff_parts(
    beam_ctx_t       ctx,
    const float*     vertex_pool,
    int              n_pool_verts,
    const int*       triangle_pool,
    int              n_pool_tris,
    PartInfoV2*      parts,
    int              n_parts,
    const int*       part_indices,
    int              n_indices,
    float            threshold);

// Test beam_expansion kernel in isolation (single iteration).
// Input: mesh (vertex_pool, triangle_pool) + parts + work_items (1 item).
// Output: out_work_items (up to beam_width), out_parts (newly created).
// Returns number of output work items, or -1 on error.
// out_parts_count: number of new PartInfoV2 entries created.
// kernel_error: bit flags from KERR_* constants.
int beam_test_expansion(
    beam_ctx_t       ctx,
    const float*     vertex_pool,
    int              n_pool_verts,
    const int*       triangle_pool,
    int              n_pool_tris,
    const PartInfoV2* parts,
    int              n_parts,
    const WorkItem*  work_items,
    int              n_items,
    int              cuts_per_axis,
    float            rv_k,
    int              beam_width,
    size_t           scratch_size,
    // Outputs (host buffers, caller-allocated)
    WorkItem*        out_work_items,   // [beam_width]
    PartInfoV2*      out_parts,        // [out_parts_capacity]
    int              out_parts_capacity,
    int*             out_parts_count,
    float*           out_vertex_pool,  // [out_vert_capacity * 3]
    int              out_vert_capacity,
    int*             out_triangle_pool,// [out_tri_capacity * 3]
    int              out_tri_capacity,
    int*             out_vert_count,
    int*             out_tri_count,
    int*             kernel_error);

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
// Set the context for plane_cut (avoids API change to beam_test_plane_cut)
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
