// C API for GPU kernels: hull mesh, mesh volume, plane cut, warp sort.
// Uses CUDA driver API internally — no CUDA runtime dependency.
// Linked into a CPython extension module; not a standalone shared library.

#ifndef BEAM_H
#define BEAM_H

#include <stdint.h>
#include <stddef.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to GPU context
typedef struct beam_ctx* beam_ctx_t;

// pool_bytes: backing memory for both heaps combined.
//   0 = auto: use 70% of free device memory at init time.
int  beam_init(beam_ctx_t* ctx, int device_ordinal, size_t pool_bytes);
void beam_destroy(beam_ctx_t ctx);

const char* beam_last_error(beam_ctx_t ctx);

// beam_heap_compact — coalesce free blocks in both persistent heaps.
// Call periodically to recover fragmented memory between kernel launches.
// Blocks until compaction is complete.
int  beam_heap_compact(beam_ctx_t ctx);

// beam_pool_usage — read back how many bytes have been bump-allocated from
// the shared pool (= peak live device memory used by both heaps combined).
// The offset never decreases; freed blocks go to heap free-lists, not back
// to the pool. Returns 0 on error.
size_t beam_pool_usage(beam_ctx_t ctx);

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
// beam_decompose — full beam-search convex decomposition
// ---------------------------------------------------------------------------
// Inputs: mesh (verts/tris) and its precomputed convex hull (hull_verts/hull_tris).
// Hyperparams:
//   max_iters:      outer iteration cap
//   cuts_per_axis:  number of candidate planes per axis (3 axes)
//   threshold:      stop when best-part cost falls below this value
//   max_keep:       beam width (max WorkItems retained per round)
//
// Output: beam_result filled with the parts of the best WorkItem.
//   All verts/tris in beam_part_result are malloc'd; call beam_result_free to release.
//   Returns 0 on success, non-zero on error (see beam_last_error).

// Host-side mirrors of CUDA device structs.
// Pointer fields use CUdeviceptr (uint64) to match 64-bit device pointers.
struct Mesh_h {
    CUdeviceptr verts;     // float* on device
    CUdeviceptr tris;      // int*   on device
    int         nv;
    int         nt;
    CUdeviceptr refcount;  // int*   on device
};

struct Part_h {
    struct Mesh_h mesh;
    struct Mesh_h hull;
    float mesh_vol;
    float hull_vol;
    float hausdorff;
};

struct beam_part_result {
    float* verts;     // malloc'd, nv*3 floats
    int*   tris;      // malloc'd, nt*3 ints
    int    nv;
    int    nt;
    float  mesh_vol;
    float  hull_vol;
    float  hausdorff;
};

struct beam_result {
    struct beam_part_result* parts;  // malloc'd array of nparts entries
    int nparts;
};

// verbose: if non-zero, print per-iteration sync wait time and final readback time to stderr.
int beam_decompose(
    beam_ctx_t   ctx,
    const float* verts,      int nv,
    const int*   tris,       int nt,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    int max_iters, int cuts_per_axis, float threshold, int max_keep,
    int verbose, int debug,
    struct beam_result* out);

void beam_result_free(struct beam_result* result);

// ---------------------------------------------------------------------------
// kdop_hull — approximate convex hull via k-DOP for a batch of point clouds
// ---------------------------------------------------------------------------
int beam_kdop_hull(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_verts,
    int*         out_tris,
    int*         out_nv,
    int*         out_nt,
    float*       out_volumes,
    int*         out_errors);

// ---------------------------------------------------------------------------
// test_hausdorff — bidirectional Hausdorff distance between two meshes
// ---------------------------------------------------------------------------
int beam_test_hausdorff(
    beam_ctx_t   ctx,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    const float* mesh_verts, int mesh_nv,
    const int*   mesh_tris,  int mesh_nt,
    float*       out_hausdorff);

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

// ---------------------------------------------------------------------------
// lookahead_decompose — lookahead tree search convex decomposition
// ---------------------------------------------------------------------------
// Inputs: mesh (verts/tris) and its precomputed convex hull (hull_verts/hull_tris).
// Hyperparams:
//   max_iters:      outer iteration cap
//   width:          number of candidate cuts per expansion level (per axis: width/3)
//   threshold:      stop when all part costs fall below this value
//   depth:          number of full expansion levels
//   quick_depth:    number of quick expansion levels (1 child per item, best-axis midpoint)
//   max_n_cutting:  max parts processed in parallel per iteration
//
// Output: beam_result (reused) filled with the parts of the final decomposition.
//   Returns 0 on success, non-zero on error (see beam_last_error).

int lookahead_decompose(
    beam_ctx_t   ctx,
    const float* verts,      int nv,
    const int*   tris,       int nt,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    int max_iters, int width, float threshold,
    int depth, int quick_depth, int max_n_cutting,
    int verbose, int debug,
    struct beam_result* out);

#ifdef __cplusplus
}
#endif

#endif // BEAM_H
