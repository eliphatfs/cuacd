// C API for GPU beam search convex decomposition.
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

#ifdef __cplusplus
}
#endif

#endif // BEAM_H
