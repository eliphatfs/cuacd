// preprocess.h — GPU mesh preprocess (PaMO stage-1 port: cumesh2sdf UDF/SDF
// + PDMC Dual Marching Cubes), orchestrated on the host through the CUDA
// driver API only.
#ifndef CUACD_PREPROCESS_H
#define CUACD_PREPROCESS_H

#include "heap.h"

#ifdef __cplusplus
extern "C" {
#endif
// Remesh (verts, tris) into a watertight manifold mesh via
// normalize → UDF/SDF voxel grid → Dual Marching Cubes.
//
//   resolution : power-of-two voxel grid resolution (32/64/128/256)
//   out_verts  : *out float[nv*3]  (malloc'd; caller frees)
//   out_tris   : *out int[nt*3]    (malloc'd; caller frees)
int gpu_preprocess(
    gpu_ctx_t ctx,
    const float* verts, int nv,
    const int*   tris,  int nt,
    int resolution,
    float** out_verts, int* out_nv,
    int**   out_tris,  int* out_nt);

#ifdef __cplusplus
}
#endif

#endif
