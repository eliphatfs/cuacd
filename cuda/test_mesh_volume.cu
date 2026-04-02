// test_mesh_volume.cu — Test kernel wrapping mesh_volume_warp.
//
// Kernels:
//   mesh_volume_kernel      — single mesh volume (one warp, one block)
//   batch_mesh_volume_kernel — batch mesh volume (one warp per mesh)

#include "mesh_volume.cuh"

// ---------------------------------------------------------------------------
// mesh_volume_kernel — single mesh, 1 block × 32 threads
// ---------------------------------------------------------------------------

extern "C" __global__ void mesh_volume_kernel(
    const float* __restrict__ verts,
    const int*   __restrict__ tris,
    int          n_tris,
    float*       __restrict__ out_volume)
{
    int lane = threadIdx.x & (WARP_SIZE - 1);
    Mesh m;
    m.verts = (float*)verts;
    m.tris  = (int*)tris;
    m.nv    = 0;
    m.nt    = n_tris;
    float vol = mesh_volume_warp(&m, lane);
    if (lane == 0) *out_volume = vol;
}

// ---------------------------------------------------------------------------
// batch_mesh_volume_kernel — n_meshes blocks × 32 threads
//
// verts:        packed float[total_verts * 3]  (relative per-mesh coords)
// tris:         packed int[total_tris * 3]     (per-mesh relative indices)
// vert_offsets: int[n_meshes + 1]  or NULL (all meshes start at 0)
// tri_offsets:  int[n_meshes + 1]
// out_volumes:  float[n_meshes]
// ---------------------------------------------------------------------------

extern "C" __global__ void batch_mesh_volume_kernel(
    const float* __restrict__ verts,
    const int*   __restrict__ tris,
    const int*   __restrict__ vert_offsets,
    const int*   __restrict__ tri_offsets,
    int          n_meshes,
    float*       __restrict__ out_volumes)
{
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_meshes) return;

    int ti = tri_offsets[warp_id];
    int nt = tri_offsets[warp_id + 1] - ti;
    int vi = vert_offsets ? vert_offsets[warp_id] : 0;

    Mesh m;
    m.verts = (float*)verts + (long long)vi * 3;
    m.tris  = (int*)tris   + (long long)ti * 3;
    m.nv    = 0;
    m.nt    = nt;

    float vol = mesh_volume_warp(&m, lane);
    if (lane == 0) out_volumes[warp_id] = vol;
}
