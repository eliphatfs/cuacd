// Host-side API for lookahead tree search decomposition.
// Implemented in lookahead.c.

#ifndef LOOKAHEAD_H
#define LOOKAHEAD_H

#include "heap.h"

#ifdef __cplusplus
extern "C" {
#endif

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
// Output: gpu_result filled with the parts of the final decomposition.
//   Returns 0 on success, non-zero on error (see gpu_last_error).

int lookahead_decompose(
    gpu_ctx_t    ctx,
    const float* verts,      int nv,
    const int*   tris,       int nt,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    int max_iters, int width, int width2, float threshold,
    int depth, int quick_depth, int max_n_cutting,
    int verbose, int debug, int decompose_components,
    struct gpu_result* out);

#ifdef __cplusplus
}
#endif

#endif // LOOKAHEAD_H
