// Host launchers for post-processing GPU kernels.
// Implemented in postprocess.c.

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "heap.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// test_postprocess_dc — split mesh into connected components
// ---------------------------------------------------------------------------
// Returns separate per-component vertex + triangle arrays in flat buffers.
// out_verts: float[max_components * max_verts_per * 3]
// out_tris:  int[max_components * max_tris_per * 3]
// out_nv, out_nt: int[max_components]
int gpu_test_postprocess_dc(
    gpu_ctx_t    ctx,
    const float* verts,  int n_verts,
    const int*   tris,   int n_tris,
    int max_components, int max_verts_per, int max_tris_per,
    float* out_verts, int* out_tris,
    int* out_nv, int* out_nt,
    int* out_n_components);

#ifdef __cplusplus
}
#endif

#endif // POSTPROCESS_H
