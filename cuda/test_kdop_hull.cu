// test_kdop_hull.cu — Test kernel for k-DOP approximate convex hull.

#include "kdop_hull.cuh"

// One block (KDOP_BLOCK threads) per point cloud.
// Extracts k-DOP hull mesh into pre-allocated flat output buffers.
extern "C" __global__ void kdop_hull_kernel(
    const float* __restrict__ pts,
    const int*   __restrict__ offsets,
    int          n_hulls,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       __restrict__ out_verts,
    int*         __restrict__ out_tris,
    int*         __restrict__ out_nv,
    int*         __restrict__ out_nt,
    float*       __restrict__ out_volumes,
    int*         __restrict__ out_errors,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= n_hulls) return;

    int start = offsets[bid];
    int count = offsets[bid + 1] - start;

    float volume = 0.0f;
    int err = 0;
    Mesh mesh = kdop_hull_block(
        pts + (long long)start * 3, count,
        heap, scratch_heap,
        &volume, &err);

    // Write counts (thread 0)
    if (tid == 0) {
        out_errors[bid]  = err;
        out_nv[bid]      = mesh.nv;
        out_nt[bid]      = mesh.nt;
        out_volumes[bid] = volume;
    }
    __syncwarp();

    // Parallel block copy of verts
    if (mesh.verts && mesh.nv <= max_hull_verts) {
        float* dv = out_verts + (long long)bid * max_hull_verts * 3;
        for (int i = tid; i < mesh.nv * 3; i += KDOP_BLOCK)
            dv[i] = mesh.verts[i];
        __syncwarp();
    }

    // Parallel block copy of tris
    if (mesh.tris && mesh.nt <= max_hull_tris) {
        int* dt = out_tris + (long long)bid * max_hull_tris * 3;
        for (int i = tid; i < mesh.nt * 3; i += KDOP_BLOCK)
            dt[i] = mesh.tris[i];
    }

    // Free the mesh from heap (it was allocated for output copy)
    __syncthreads();
    if (tid == 0 && mesh.verts) {
        heap_free(heap, mesh.verts);
    }
}
