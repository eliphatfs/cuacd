// test_beam.c — Host launchers for test/diagnostic GPU kernels.
// Compiled into the same extension as beam.c; separated to keep beam.c focused
// on production functionality.

#include "beam.h"
#include "structs.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at test_beam.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        return (int)_r; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// beam_test_warp_sort
// ---------------------------------------------------------------------------
// points: packed int[total_pts * 4] (x,y,z,index)
// offsets: int[n_arrays + 1]
// Sorts each sub-array in-place.

int beam_test_warp_sort(beam_ctx_t ctx,
    int* points, int total_pts, const int* offsets, int n_arrays)
{
    if (!ctx || !ctx->fn_test_warp_sort) return -1;
    CUstream s = NULL;

    // Compute per-array scratch sizes and offsets
    // Each array needs: n * sizeof(BtPoint32) + WS_MAX_STACK * 2 * sizeof(int)
    const int stack_bytes = 2048 * 2 * (int)sizeof(int);  // WS_MAX_STACK = 2048
    int* scratch_offsets = (int*)malloc((size_t)(n_arrays + 1) * sizeof(int));
    if (!scratch_offsets) return -1;
    scratch_offsets[0] = 0;
    for (int i = 0; i < n_arrays; i++) {
        int n_pts = offsets[i + 1] - offsets[i];
        int sz = n_pts * 4 * (int)sizeof(int) + stack_bytes;
        sz = (sz + 15) & ~15;  // align to 16
        scratch_offsets[i + 1] = scratch_offsets[i] + sz;
    }
    size_t total_scratch = (size_t)scratch_offsets[n_arrays];

    CUdeviceptr d_pts, d_off, d_scratch, d_soff;
    size_t pts_bytes = (size_t)total_pts * 4 * sizeof(int);
    CHECK_CU(cuMemAlloc(&d_pts,     pts_bytes));
    CHECK_CU(cuMemAlloc(&d_off,     (size_t)(n_arrays + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_scratch, total_scratch));
    CHECK_CU(cuMemAlloc(&d_soff,    (size_t)(n_arrays + 1) * sizeof(int)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, points, pts_bytes, s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_arrays + 1) * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_soff, scratch_offsets, (size_t)(n_arrays + 1) * sizeof(int), s));
    free(scratch_offsets);

    int block_size = 64;
    int warps_per_block = block_size / 32;
    int n_blocks = (n_arrays + warps_per_block - 1) / warps_per_block;

    void* args[] = { &d_pts, &d_off, &d_scratch, &d_soff, &n_arrays };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_warp_sort,
        n_blocks, 1, 1, block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(points, d_pts, pts_bytes, s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts);
    cuMemFree(d_off);
    cuMemFree(d_scratch);
    cuMemFree(d_soff);
    return 0;
}

// ---------------------------------------------------------------------------
// beam_batch_hull_dandc_mesh
// ---------------------------------------------------------------------------

int beam_batch_hull_dandc_mesh(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_pts_per_hull,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_volumes,
    int*         out_errors,
    float*       out_verts,
    int*         out_tris,
    int*         out_vert_counts,
    int*         out_tri_counts)
{
    if (!ctx || !ctx->fn_batch_hull_dandc_mesh) return -1;
    CUstream s = NULL;

    CUdeviceptr d_pts, d_off, d_vols, d_errs, d_scratch = 0;
    CUdeviceptr d_overts, d_otris, d_ovc, d_otc;
    CHECK_CU(cuMemAlloc(&d_pts,  (size_t)total_pts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_off,  (size_t)(n_hulls + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_vols, (size_t)n_hulls * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_errs, (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_otris,  (size_t)n_hulls * max_hull_tris * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_ovc, (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_otc, (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, pts, (size_t)total_pts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls + 1) * sizeof(int), s));

    // Query scratch
    CUdeviceptr d_qout;
    CHECK_CU(cuMemAlloc(&d_qout, sizeof(int)));
    int max_pts_i = max_pts_per_hull;
    void* qargs[] = { &max_pts_i, &d_qout };
    CHECK_CU(cuLaunchKernel(ctx->fn_query_dandc_scratch, 1, 1, 1,
                            1, 1, 1, 0, s, qargs, NULL));
    int scratch_result = 0;
    CHECK_CU(cuMemcpyDtoHAsync(&scratch_result, d_qout, sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));
    cuMemFree(d_qout);
    size_t scratch_per = (size_t)scratch_result;

    size_t total_scratch = (size_t)n_hulls * scratch_per;
    CHECK_CU(cuMemAlloc(&d_scratch, total_scratch));
    CHECK_CU(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8 * 1024));

    int block_size = 32;
    int n_blocks = n_hulls;
    int scratch_per_i = (int)scratch_per;

    void* args[] = { &d_pts, &d_off, &d_vols, &d_errs,
                     &d_overts, &d_otris, &d_ovc, &d_otc,
                     &d_scratch, &scratch_per_i, &n_hulls,
                     &max_hull_verts, &max_hull_tris };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_hull_dandc_mesh, n_blocks, 1, 1,
                            block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_vols, (size_t)n_hulls * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_errors,  d_errs, (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_verts, d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_tris, d_otris, (size_t)n_hulls * max_hull_tris * 3 * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_vert_counts, d_ovc, (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_tri_counts, d_otc, (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts); cuMemFree(d_off); cuMemFree(d_vols); cuMemFree(d_errs);
    cuMemFree(d_overts); cuMemFree(d_otris); cuMemFree(d_ovc); cuMemFree(d_otc);
    if (d_scratch) cuMemFree(d_scratch);
    return 0;
}

// ---------------------------------------------------------------------------
// beam_test_plane_cut
// ---------------------------------------------------------------------------

// GPU kernel error bit flags (must match plane_cut.cu)
#define PC_KERR_SCRATCH_OOM 1
#define PC_KERR_POOL_OOM    2
#define PC_KERR_SORT_ERR    4

// Global ctx pointer set by beam_set_plane_cut_ctx (avoids API change to beam_test_plane_cut)
static beam_ctx_t g_plane_cut_ctx = NULL;

void beam_set_plane_cut_ctx(beam_ctx_t ctx) { g_plane_cut_ctx = ctx; }

int beam_test_plane_cut(
    const float* vertices, int n_verts,
    const int* triangles, int n_tris,
    float pa, float pb, float pc_n, float pd,
    float* out_all_verts, int out_verts_cap,
    int* out_pos_tris, int out_pos_cap,
    int* out_neg_tris, int out_neg_cap,
    int* out_n_verts,
    int* out_n_pos_tris,
    int* out_n_neg_tris)
{
    beam_ctx_t ctx = g_plane_cut_ctx;
    if (!ctx) return -1;

    *out_n_verts = 0; *out_n_pos_tris = 0; *out_n_neg_tris = 0;

    CUstream s = NULL;
    CUdeviceptr d_verts = 0, d_tris = 0;
    CUdeviceptr d_out_verts = 0, d_out_pos = 0, d_out_neg = 0;
    CUdeviceptr d_counts = 0, d_kerr = 0;
    CUdeviceptr d_scratch_base = 0, d_scratch_off = 0;

    CHECK_CU(cuMemAlloc(&d_verts, (size_t)n_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris, (size_t)n_tris * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out_verts, (size_t)out_verts_cap * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_out_pos, (size_t)out_pos_cap * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out_neg, (size_t)out_neg_cap * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_counts, 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_kerr, sizeof(int)));

    // Scratch pool
    size_t free_mem = 0, total_mem = 0;
    cuMemGetInfo(&free_mem, &total_mem);
    size_t scratch_sz = (size_t)(free_mem * 0.5);
    if (scratch_sz < 64 * 1024 * 1024) scratch_sz = 64 * 1024 * 1024;
    CHECK_CU(cuMemAlloc(&d_scratch_base, scratch_sz));
    CHECK_CU(cuMemAlloc(&d_scratch_off, sizeof(unsigned long long)));

    struct DevicePool sp;
    sp.base = (char*)(uintptr_t)d_scratch_base;
    sp.offset = (unsigned long long*)(uintptr_t)d_scratch_off;
    sp.capacity = (unsigned long long)scratch_sz;

    // Upload input
    CHECK_CU(cuMemcpyHtoDAsync(d_verts, vertices, (size_t)n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris, triangles, (size_t)n_tris * 3 * sizeof(int), s));

    // Zero scratch offset and kernel error
    unsigned long long zero_ull = 0;
    CHECK_CU(cuMemcpyHtoDAsync(d_scratch_off, &zero_ull, sizeof(unsigned long long), s));
    int zero_i = 0;
    CHECK_CU(cuMemcpyHtoDAsync(d_kerr, &zero_i, sizeof(int), s));

    // Count pointers
    CUdeviceptr d_nv = d_counts;
    CUdeviceptr d_np = d_counts + sizeof(int);
    CUdeviceptr d_nn = d_counts + 2 * sizeof(int);

    void* args[] = {
        &d_verts, &d_tris,
        &n_verts, &n_tris,
        &pa, &pb, &pc_n, &pd,
        &d_out_verts, &d_out_pos, &d_out_neg,
        &out_verts_cap, &out_pos_cap, &out_neg_cap,
        &d_nv, &d_np, &d_nn,
        &sp, &d_kerr
    };

    CHECK_CU(cuLaunchKernel(ctx->fn_plane_cut,
        1, 1, 1, 64, 1, 1, 0, s, args, NULL));

    int kerr_h = 0;
    int counts_h[3] = {0, 0, 0};
    CHECK_CU(cuMemcpyDtoHAsync(&kerr_h, d_kerr, sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(counts_h, d_counts, 3 * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    if (kerr_h) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "plane_cut kernel error: 0x%x", kerr_h);
        goto gpu_cleanup_err;
    }

    *out_n_verts = counts_h[0];
    *out_n_pos_tris = counts_h[1];
    *out_n_neg_tris = counts_h[2];

    if (*out_n_verts > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_all_verts, d_out_verts,
            (size_t)(*out_n_verts) * 3 * sizeof(float), s));
    if (*out_n_pos_tris > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_pos_tris, d_out_pos,
            (size_t)(*out_n_pos_tris) * 3 * sizeof(int), s));
    if (*out_n_neg_tris > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_neg_tris, d_out_neg,
            (size_t)(*out_n_neg_tris) * 3 * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_out_verts); cuMemFree(d_out_pos); cuMemFree(d_out_neg);
    cuMemFree(d_counts); cuMemFree(d_kerr);
    cuMemFree(d_scratch_base); cuMemFree(d_scratch_off);
    return 0;

gpu_cleanup_err:
    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_out_verts); cuMemFree(d_out_pos); cuMemFree(d_out_neg);
    cuMemFree(d_counts); cuMemFree(d_kerr);
    cuMemFree(d_scratch_base); cuMemFree(d_scratch_off);
    return -1;
}
