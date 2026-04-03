// mesh_volume.cuh — Per-warp mesh volume via divergence theorem.
#pragma once
#include "warp_common.cuh"
#include "geometry.cuh"
#include "structs.cuh"

// Compute |volume| of a watertight Mesh using the signed tetrahedron sum.
// All 32 lanes must call with identical arguments.
// Returns the absolute volume on every lane.
__device__ inline float mesh_volume_warp(const Mesh* mesh, int lane)
{
    float local = 0.0f;
    for (int t = lane; t < mesh->nt; t += WARP_SIZE) {
        int ai = mesh->tris[t * 3 + 0];
        int bi = mesh->tris[t * 3 + 1];
        int ci = mesh->tris[t * 3 + 2];
        local += signed_tet_volume(
            mesh->verts[ai*3], mesh->verts[ai*3+1], mesh->verts[ai*3+2],
            mesh->verts[bi*3], mesh->verts[bi*3+1], mesh->verts[bi*3+2],
            mesh->verts[ci*3], mesh->verts[ci*3+1], mesh->verts[ci*3+2]);
    }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(WARP_MASK, local, off);
    return fabsf(local);
}
