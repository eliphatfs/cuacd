#include "plane_cut.cuh"



extern "C" __global__ void plane_cut_kernel(
    const float* __restrict__ vertices,
    const int*   __restrict__ triangles,
    int n_verts, int n_tris,
    float pa, float pb, float pc_n, float pd,
    float* __restrict__ out_verts,
    int*   __restrict__ out_pos_tris,
    int*   __restrict__ out_neg_tris,
    int out_verts_cap, int out_pos_cap, int out_neg_cap,
    int* __restrict__ out_n_verts,
    int* __restrict__ out_n_pos_tris,
    int* __restrict__ out_n_neg_tris,
    DevicePool scratch,
    int* __restrict__ kernel_error)
{
    plane_cut_block(vertices, triangles, n_verts, n_tris,
                    pa, pb, pc_n, pd,
                    out_verts, out_pos_tris, out_neg_tris,
                    out_verts_cap, out_pos_cap, out_neg_cap,
                    out_n_verts, out_n_pos_tris, out_n_neg_tris,
                    scratch, kernel_error);
}
