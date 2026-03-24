// Public C API for coacd_gpu. Loaded via ctypes from Python.
// Uses CUDA driver API internally — no CUDA runtime dependency.

#ifndef COACD_GPU_H
#define COACD_GPU_H

#include <stdint.h>

#ifdef _WIN32
  #define COACD_GPU_API __declspec(dllexport)
#else
  #define COACD_GPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the GPU context (holds CUmodule, CUfunction handles).
typedef struct coacd_gpu_ctx* coacd_gpu_ctx_t;

// Initialize: loads the fatbin, resolves kernel functions.
// Returns 0 on success, non-zero CUDA driver error code on failure.
// device_ordinal: which GPU to use (-1 = current context's device).
COACD_GPU_API int coacd_gpu_init(coacd_gpu_ctx_t* ctx, int device_ordinal);

// Destroy context and free resources.
COACD_GPU_API void coacd_gpu_destroy(coacd_gpu_ctx_t ctx);

// Compute per-point minimum distance from sample points to a triangle mesh.
//
// All pointers are HOST memory. The library handles upload/download.
//   points:     [n_points, 3] float32 — query points
//   vertices:   [n_verts, 3]  float32 — mesh vertices
//   triangles:  [n_tris, 3]   int32   — mesh face indices
//   distances:  [n_points]    float32 — output, caller-allocated
//
// stream: optional CUDA stream (CUstream). Pass NULL / 0 for default stream.
//
// Returns 0 on success.
COACD_GPU_API int coacd_gpu_point_mesh_distances(
    coacd_gpu_ctx_t ctx,
    const float*    points,
    int             n_points,
    const float*    vertices,
    int             n_verts,
    const int*      triangles,
    int             n_tris,
    float*          distances,
    void*           stream);

// Compute Hausdorff distance between two sampled meshes.
// Returns the scalar result in *result. Returns 0 on success.
COACD_GPU_API int coacd_gpu_hausdorff(
    coacd_gpu_ctx_t ctx,
    const float*    samples_a,    int n_samples_a,
    const float*    vertices_a,   int n_verts_a,
    const int*      triangles_a,  int n_tris_a,
    const float*    samples_b,    int n_samples_b,
    const float*    vertices_b,   int n_verts_b,
    const int*      triangles_b,  int n_tris_b,
    float*          result,
    void*           stream);

// Compute pairwise Hausdorff cost matrix for the merge phase.
//
// Parts' data is packed contiguously:
//   all_samples:    [total_samples, 3]   float32
//   sample_offsets: [n_parts + 1]        int32 (prefix sums)
//   all_vertices:   [total_verts, 3]     float32
//   all_triangles:  [total_tris, 3]      int32 (local indices per part)
//   tri_offsets:    [n_parts + 1]        int32
//   vert_offsets:   [n_parts + 1]        int32
//   cost_matrix:    [n_parts * n_parts]  float32 — output, caller-allocated
//                   Lower triangle filled; upper = 0.
//
// Returns 0 on success.
COACD_GPU_API int coacd_gpu_pairwise_hausdorff(
    coacd_gpu_ctx_t ctx,
    const float*    all_samples,
    const int*      sample_offsets,
    const float*    all_vertices,
    const int*      all_triangles,
    const int*      tri_offsets,
    const int*      vert_offsets,
    int             n_parts,
    float*          cost_matrix,
    void*           stream);

// Returns a human-readable error string for the last error, or NULL.
COACD_GPU_API const char* coacd_gpu_last_error(coacd_gpu_ctx_t ctx);

#ifdef __cplusplus
}
#endif

#endif // COACD_GPU_H
