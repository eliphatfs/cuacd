// test_hull_dandc.cu — Test kernels for D&C convex hull.
//
// Kernels:
//   hull_dandc_kernel       — extract hull mesh (verts + tris)
//   query_dandc_scratch     — query scratch bytes per hull

#include "hull_dandc.cuh"
#include "mesh_volume.cuh"

// ---------------------------------------------------------------------------
// query_dandc_scratch — return scratch bytes needed per hull (1 thread)
// ---------------------------------------------------------------------------

extern "C" __global__ void query_dandc_scratch(int n, int* out)
{
    if (threadIdx.x == 0)
        *out = dandc_scratch_bytes(n);
}

// ---------------------------------------------------------------------------
// hull_dandc_kernel
//
// One warp per hull.  Extracts the convex hull mesh into pre-allocated flat
// output buffers.  Does NOT compute volume.
//
// pts:            float[total_pts * 3]
// offsets:        int[n_hulls + 1]
// out_verts:      float[n_hulls * max_hull_verts * 3]
// out_tris:       int[n_hulls * max_hull_tris * 3]
// out_nv, out_nt, out_errors: int[n_hulls]
// heap:           DeviceHeap* (for output mesh data, freed after copy)
// scratch_heap:   DeviceHeap* (for warp scratch + edge pool; clean after call)
// ---------------------------------------------------------------------------

extern "C" __global__ void hull_dandc_kernel(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    int          n_hulls,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       __restrict__ out_verts,
    int*         __restrict__ out_tris,
    int*         __restrict__ out_nv,
    int*         __restrict__ out_nt,
    int*         __restrict__ out_errors,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap)
{
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= n_hulls) return;

    int start = offsets[warp_id];
    int count = offsets[warp_id + 1] - start;

    int err = 0;
    Mesh mesh = hull_dandc_warp_mesh(
        pts + (long long)start * 3, count, lane,
        heap, scratch_heap, &err);

    // Write counts (lane 0)
    if (lane == 0) {
        out_errors[warp_id] = err;
        out_nv[warp_id]     = mesh.nv;
        out_nt[warp_id]     = mesh.nt;
    }

    // Parallel warp copy of verts
    if (mesh.verts && mesh.nv <= max_hull_verts) {
        float* dv = out_verts + (long long)warp_id * max_hull_verts * 3;
        for (int i = lane; i < mesh.nv * 3; i += WARP_SIZE)
            dv[i] = mesh.verts[i];
    }

    // Parallel warp copy of tris
    if (mesh.tris && mesh.nt <= max_hull_tris) {
        int* dt = out_tris + (long long)warp_id * max_hull_tris * 3;
        for (int i = lane; i < mesh.nt * 3; i += WARP_SIZE)
            dt[i] = mesh.tris[i];
    }

    __syncwarp();

    // Free output allocation from heap (lane 0 only)
    if (lane == 0 && mesh.verts)
        heap_free(heap, mesh.verts);
}
