// mesh_volume.cuh — Per-warp mesh volume via divergence theorem.
#pragma once
#include "warp_common.cuh"
#include "geometry.cuh"
#include "structs.cuh"

// Compute signed volume of a mesh using the signed tetrahedron sum.
// All 32 lanes must call with identical arguments.
// Returns the signed volume on every lane (positive = outward winding).
//
// Numerical note: each tet contribution is `(p0-O)·((p1-O)×(p2-O))/6`, where
// O is the divergence-theorem origin. Mathematically O is arbitrary for a
// closed mesh (signed volume is translation-invariant), but the per-tet
// magnitudes are O(|p-O|^3). With O at the world origin and verts of order
// unity, individual tets are O(1) but they cancel down to ~|svol|, so f32
// noise scales as nt·|p|^3·ε and easily flips signs of CCs whose true svol
// is small. We anchor O at vertex 0 of the mesh: per-tet magnitudes drop to
// O(local_extent^3), eliminating the catastrophic cancellation. Measured on
// the bistro mesh: 923 sign flips → 3 (median rel-err 10× → 3e-5).
__device__ inline float mesh_signed_volume_warp(const Mesh* mesh, int lane)
{
    int nt = mesh->nt;
    if (nt == 0) return 0.0f;  // uniform across warp — safe to early-exit

    // Read anchor vertex (mesh->verts[0..2]) on lane 0; broadcast to all lanes.
    float v0x = 0.f, v0y = 0.f, v0z = 0.f;
    if (lane == 0) {
        v0x = mesh->verts[0];
        v0y = mesh->verts[1];
        v0z = mesh->verts[2];
    }
    v0x = __shfl_sync(WARP_MASK, v0x, 0);
    v0y = __shfl_sync(WARP_MASK, v0y, 0);
    v0z = __shfl_sync(WARP_MASK, v0z, 0);

    float local = 0.0f;
    for (int t = lane; t < nt; t += WARP_SIZE) {
        int ai = mesh->tris[t * 3 + 0];
        int bi = mesh->tris[t * 3 + 1];
        int ci = mesh->tris[t * 3 + 2];
        local += signed_tet_volume(
            mesh->verts[ai*3+0] - v0x, mesh->verts[ai*3+1] - v0y, mesh->verts[ai*3+2] - v0z,
            mesh->verts[bi*3+0] - v0x, mesh->verts[bi*3+1] - v0y, mesh->verts[bi*3+2] - v0z,
            mesh->verts[ci*3+0] - v0x, mesh->verts[ci*3+1] - v0y, mesh->verts[ci*3+2] - v0z);
    }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(WARP_MASK, local, off);
    return local;
}

// Compute |volume| of a watertight Mesh using the signed tetrahedron sum.
// All 32 lanes must call with identical arguments.
// Returns the absolute volume on every lane.
__device__ inline float mesh_volume_warp(const Mesh* mesh, int lane)
{
    return fabsf(mesh_signed_volume_warp(mesh, lane));
}
