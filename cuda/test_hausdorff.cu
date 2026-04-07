// test_hausdorff.cu — Test kernel wrapping hausdorff_block.
//
// Kernels:
//   hausdorff_kernel — single pair of meshes, 1 block x 256 threads

#include "hausdorff.cuh"

extern "C" __global__ void hausdorff_kernel(
    const float* __restrict__ hull_verts,
    const int*   __restrict__ hull_tris,
    int         hull_nv,
    int         hull_nt,
    const float* __restrict__ mesh_verts,
    const int*   __restrict__ mesh_tris,
    int         mesh_nv,
    int         mesh_nt,
    float*      __restrict__ out_hausdorff,
    DeviceHeap* __restrict__ scratch_heap,
    int*        __restrict__ kernel_error)
{
    __shared__ Mesh s_hull, s_mesh;
    if (threadIdx.x == 0) {
        s_hull.verts    = (float*)hull_verts;
        s_hull.tris     = (int*)hull_tris;
        s_hull.nv       = hull_nv;
        s_hull.nt       = hull_nt;
        s_hull.refcount = NULL;
        s_mesh.verts    = (float*)mesh_verts;
        s_mesh.tris     = (int*)mesh_tris;
        s_mesh.nv       = mesh_nv;
        s_mesh.nt       = mesh_nt;
        s_mesh.refcount = NULL;
    }
    __syncthreads();

    float h = hausdorff_block(&s_hull, &s_mesh, scratch_heap, kernel_error);

    if (threadIdx.x == 0) *out_hausdorff = h;
}
