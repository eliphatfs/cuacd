// test_plane_cut.cu — Test kernel wrapping plane_cut_block.
//
// Kernels:
//   plane_cut_kernel — one block (64 threads) per cut
//
// Output: separate pos and neg vertex + triangle arrays.
// Vertex indices in pos_tris are relative to pos_verts (same for neg).

#include "plane_cut.cuh"

extern "C" __global__ void plane_cut_kernel(
    const float* __restrict__ in_verts,
    const int*   __restrict__ in_tris,
    int          n_verts,
    int          n_tris,
    float        pa, float pb, float pc_n, float pd,
    float*       __restrict__ out_pos_verts,
    int*         __restrict__ out_pos_tris,
    float*       __restrict__ out_neg_verts,
    int*         __restrict__ out_neg_tris,
    int          out_pos_verts_cap,
    int          out_pos_tris_cap,
    int          out_neg_verts_cap,
    int          out_neg_tris_cap,
    int*         __restrict__ out_n_pv,
    int*         __restrict__ out_n_pt,
    int*         __restrict__ out_n_nv,
    int*         __restrict__ out_n_nt,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    int*         __restrict__ kernel_error)
{
    __shared__ Mesh s_input;
    if (threadIdx.x == 0) {
        s_input.verts = (float*)in_verts;
        s_input.tris  = (int*)in_tris;
        s_input.nv    = n_verts;
        s_input.nt    = n_tris;
    }
    __syncthreads();

    PartPair result = plane_cut_block(
        &s_input, pa, pb, pc_n, pd,
        heap, scratch_heap, kernel_error);

    __syncthreads();

    int tid = threadIdx.x;

    // Write counts
    if (tid == 0) {
        *out_n_pv = result.pos.mesh.nv;
        *out_n_pt = result.pos.mesh.nt;
        *out_n_nv = result.neg.mesh.nv;
        *out_n_nt = result.neg.mesh.nt;
    }
    __syncthreads();

    // Parallel copy of pos verts (64 threads)
    if (result.pos.mesh.verts && result.pos.mesh.nv <= out_pos_verts_cap)
        for (int i = tid; i < result.pos.mesh.nv * 3; i += blockDim.x)
            out_pos_verts[i] = result.pos.mesh.verts[i];

    // Parallel copy of pos tris
    if (result.pos.mesh.tris && result.pos.mesh.nt <= out_pos_tris_cap)
        for (int i = tid; i < result.pos.mesh.nt * 3; i += blockDim.x)
            out_pos_tris[i] = result.pos.mesh.tris[i];

    // Parallel copy of neg verts
    if (result.neg.mesh.verts && result.neg.mesh.nv <= out_neg_verts_cap)
        for (int i = tid; i < result.neg.mesh.nv * 3; i += blockDim.x)
            out_neg_verts[i] = result.neg.mesh.verts[i];

    // Parallel copy of neg tris
    if (result.neg.mesh.tris && result.neg.mesh.nt <= out_neg_tris_cap)
        for (int i = tid; i < result.neg.mesh.nt * 3; i += blockDim.x)
            out_neg_tris[i] = result.neg.mesh.tris[i];

    __syncthreads();

    // Free heap allocation (thread 0 only)
    // pos.mesh.verts points to the start of the combined heap chunk
    // (or is NULL if empty; heap_free handles NULL)
    if (tid == 0) {
        void* chunk = result.pos.mesh.verts
                      ? (void*)result.pos.mesh.verts
                      : (void*)result.neg.mesh.verts;
        heap_free(heap, chunk);
    }
}
