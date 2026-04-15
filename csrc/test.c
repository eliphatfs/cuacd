// test.c — Host launchers for test/diagnostic GPU kernels.
// Compiled into the same extension as heap.c; separated to keep heap.c focused
// on init/destroy.

#include "heap.h"
#include "structs.h"
#include "error_codes.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at test.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        return (int)_r; \
    } \
} while(0)

// Device addresses of the two embedded DeviceHeap instances inside d_pool_struct.
// These are passed as DeviceHeap* to kernel functions.
#define D_HEAP(ctx)    ((ctx)->d_pool_struct + offsetof(struct DevicePool, heap))
#define D_SCRATCH(ctx) ((ctx)->d_pool_struct + offsetof(struct DevicePool, scratch))

// ---------------------------------------------------------------------------
// gpu_test_warp_sort
// ---------------------------------------------------------------------------
// points: packed int[total_pts * 4] (x,y,z,index)
// offsets: int[n_arrays + 1]
// Sorts each sub-array in-place.

int gpu_test_warp_sort(gpu_ctx_t ctx,
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
// gpu_hull_dandc
// ---------------------------------------------------------------------------
// Extract convex hull mesh for each point cloud.
// Uses persistent pool->heap (output) and pool->scratch (scratch).
// Both heaps share the same pool; all allocations are freed by the kernel.

int gpu_hull_dandc(
    gpu_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_pts_per_hull,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_verts,
    int*         out_tris,
    int*         out_nv,
    int*         out_nt,
    int*         out_errors)
{
    if (!ctx || !ctx->fn_hull_dandc) return -1;
    if (!ctx->d_pool_struct) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "persistent heaps not initialized; call gpu_init first");
        return -1;
    }
    CUstream s = NULL;

    // Device addresses of embedded heaps within d_pool_struct
    CUdeviceptr d_heap    = D_HEAP(ctx);
    CUdeviceptr d_scratch = D_SCRATCH(ctx);

    // Device buffers for input and output
    CUdeviceptr d_pts, d_off, d_overts, d_otris, d_onv, d_ont, d_oerr;
    CHECK_CU(cuMemAlloc(&d_pts,    (size_t)total_pts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_off,    (size_t)(n_hulls + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_otris,  (size_t)n_hulls * max_hull_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_onv,    (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_ont,    (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_oerr,   (size_t)n_hulls * sizeof(int)));

    CHECK_CU(cuMemcpyHtoDAsync(d_pts, pts,     (size_t)total_pts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls + 1) * sizeof(int),   s));

    CHECK_CU(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8 * 1024));

    int block_size = 32;   // 1 warp per block
    int n_blocks   = n_hulls;

    void* args[] = {
        &d_pts, &d_off, &n_hulls,
        &max_hull_verts, &max_hull_tris,
        &d_overts, &d_otris, &d_onv, &d_ont, &d_oerr,
        &d_heap, &d_scratch
    };
    CHECK_CU(cuLaunchKernel(ctx->fn_hull_dandc, n_blocks, 1, 1,
                            block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_verts,  d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_tris,   d_otris,  (size_t)n_hulls * max_hull_tris  * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyDtoHAsync(out_nv,     d_onv,    (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_nt,     d_ont,    (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_errors, d_oerr,   (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts); cuMemFree(d_off);
    cuMemFree(d_overts); cuMemFree(d_otris);
    cuMemFree(d_onv); cuMemFree(d_ont); cuMemFree(d_oerr);
    return 0;
}

// ---------------------------------------------------------------------------
// gpu_test_mesh_volume
// ---------------------------------------------------------------------------
// Compute volume of a single mesh via divergence theorem (GPU warp).

int gpu_test_mesh_volume(
    gpu_ctx_t   ctx,
    const float* verts,
    int          n_verts,
    const int*   tris,
    int          n_tris,
    float*       out_volume)
{
    if (!ctx || !ctx->fn_mesh_volume) return -1;
    CUstream s = NULL;

    CUdeviceptr d_verts, d_tris, d_vol;
    CHECK_CU(cuMemAlloc(&d_verts, (size_t)n_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,  (size_t)n_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_vol,   sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts, verts, (size_t)n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,  tris,  (size_t)n_tris  * 3 * sizeof(int),   s));

    void* args[] = { &d_verts, &d_tris, &n_tris, &d_vol };
    CHECK_CU(cuLaunchKernel(ctx->fn_mesh_volume, 1, 1, 1, 32, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volume, d_vol, sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts); cuMemFree(d_tris); cuMemFree(d_vol);
    return 0;
}

// ---------------------------------------------------------------------------
// gpu_batch_mesh_volume
// ---------------------------------------------------------------------------

int gpu_batch_mesh_volume(
    gpu_ctx_t   ctx,
    const float* verts,
    int          total_verts,
    const int*   tris,
    int          total_tris,
    const int*   tri_offsets,
    const int*   vert_offsets,
    int          n_meshes,
    float*       out_volumes)
{
    if (!ctx || !ctx->fn_batch_mesh_volume) return -1;
    CUstream s = NULL;

    CUdeviceptr d_verts, d_tris, d_toff, d_voff = 0, d_vols;
    CHECK_CU(cuMemAlloc(&d_verts, (size_t)total_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,  (size_t)total_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_toff,  (size_t)(n_meshes + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_vols,  (size_t)n_meshes * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts, verts, (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,  tris,  (size_t)total_tris  * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyHtoDAsync(d_toff,  tri_offsets, (size_t)(n_meshes + 1) * sizeof(int), s));

    if (vert_offsets) {
        CHECK_CU(cuMemAlloc(&d_voff, (size_t)(n_meshes + 1) * sizeof(int)));
        CHECK_CU(cuMemcpyHtoDAsync(d_voff, vert_offsets, (size_t)(n_meshes + 1) * sizeof(int), s));
    }

    // 1 warp per mesh
    int block_size = 32;
    int n_blocks = n_meshes;

    void* args[] = { &d_verts, &d_tris, &d_voff, &d_toff, &n_meshes, &d_vols };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_mesh_volume,
        n_blocks, 1, 1, block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_vols, (size_t)n_meshes * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts); cuMemFree(d_tris); cuMemFree(d_toff); cuMemFree(d_vols);
    if (d_voff) cuMemFree(d_voff);
    return 0;
}

// ---------------------------------------------------------------------------
// gpu_test_plane_cut
// ---------------------------------------------------------------------------
// Uses persistent pool->heap (output) and pool->scratch (scratch).
// Both heaps share the same pool; all allocations are freed by the kernel.

int gpu_test_plane_cut(
    gpu_ctx_t   ctx,
    const float* vertices, int n_verts,
    const int*   triangles, int n_tris,
    float pa, float pb, float pc_n, float pd,
    float* out_pos_verts, int out_pos_verts_cap,
    int*   out_pos_tris,  int out_pos_tris_cap,
    float* out_neg_verts, int out_neg_verts_cap,
    int*   out_neg_tris,  int out_neg_tris_cap,
    int* out_n_pv, int* out_n_pt,
    int* out_n_nv, int* out_n_nt)
{
    if (!ctx || !ctx->fn_plane_cut) return -1;
    if (!ctx->d_pool_struct) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "persistent heaps not initialized; call gpu_init first");
        return -1;
    }

    *out_n_pv = 0; *out_n_pt = 0;
    *out_n_nv = 0; *out_n_nt = 0;

    CUstream s = NULL;
    CUdeviceptr d_verts = 0, d_tris = 0;
    CUdeviceptr d_opv = 0, d_opt = 0, d_onv = 0, d_ont = 0;
    CUdeviceptr d_counts = 0, d_kerr = 0;

    CHECK_CU(cuMemAlloc(&d_verts, (size_t)n_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,  (size_t)n_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_opv,   (size_t)out_pos_verts_cap * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_opt,   (size_t)out_pos_tris_cap  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_onv,   (size_t)out_neg_verts_cap * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_ont,   (size_t)out_neg_tris_cap  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_counts, 4 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_kerr,   sizeof(int)));

    CHECK_CU(cuMemcpyHtoDAsync(d_verts, vertices,  (size_t)n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,  triangles, (size_t)n_tris  * 3 * sizeof(int),   s));

    int zero_i = 0;
    CHECK_CU(cuMemcpyHtoDAsync(d_kerr, &zero_i, sizeof(int), s));

    CUdeviceptr d_npv = d_counts;
    CUdeviceptr d_npt = d_counts +     sizeof(int);
    CUdeviceptr d_nnv = d_counts + 2 * sizeof(int);
    CUdeviceptr d_nnt = d_counts + 3 * sizeof(int);

    // Device addresses of embedded heaps within d_pool_struct
    CUdeviceptr d_heap    = D_HEAP(ctx);
    CUdeviceptr d_scratch = D_SCRATCH(ctx);

    void* args[] = {
        &d_verts, &d_tris, &n_verts, &n_tris,
        &pa, &pb, &pc_n, &pd,
        &d_opv, &d_opt, &d_onv, &d_ont,
        &out_pos_verts_cap, &out_pos_tris_cap,
        &out_neg_verts_cap, &out_neg_tris_cap,
        &d_npv, &d_npt, &d_nnv, &d_nnt,
        &d_heap, &d_scratch,
        &d_kerr
    };
    CHECK_CU(cuLaunchKernel(ctx->fn_plane_cut,
        1, 1, 1, 64, 1, 1, 0, s, args, NULL));

    int kerr_h = 0;
    int counts_h[4] = {0, 0, 0, 0};
    CHECK_CU(cuMemcpyDtoHAsync(&kerr_h,   d_kerr,   sizeof(int),       s));
    CHECK_CU(cuMemcpyDtoHAsync(counts_h,  d_counts, 4 * sizeof(int),   s));
    CHECK_CU(cuStreamSynchronize(s));

    if (kerr_h) {
        char errbuf[512];
        kerr_decode(kerr_h, errbuf, sizeof(errbuf));
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "plane_cut %s", errbuf);
        goto cleanup_err;
    }

    *out_n_pv = counts_h[0]; *out_n_pt = counts_h[1];
    *out_n_nv = counts_h[2]; *out_n_nt = counts_h[3];

    if (*out_n_pv > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_pos_verts, d_opv,
            (size_t)(*out_n_pv) * 3 * sizeof(float), s));
    if (*out_n_pt > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_pos_tris, d_opt,
            (size_t)(*out_n_pt) * 3 * sizeof(int), s));
    if (*out_n_nv > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_neg_verts, d_onv,
            (size_t)(*out_n_nv) * 3 * sizeof(float), s));
    if (*out_n_nt > 0)
        CHECK_CU(cuMemcpyDtoHAsync(out_neg_tris, d_ont,
            (size_t)(*out_n_nt) * 3 * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_opv); cuMemFree(d_opt);
    cuMemFree(d_onv); cuMemFree(d_ont);
    cuMemFree(d_counts); cuMemFree(d_kerr);
    return 0;

cleanup_err:
    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_opv); cuMemFree(d_opt);
    cuMemFree(d_onv); cuMemFree(d_ont);
    cuMemFree(d_counts); cuMemFree(d_kerr);
    return -1;
}

// ---------------------------------------------------------------------------
// gpu_kdop_hull
// ---------------------------------------------------------------------------
// Approximate convex hull via k-DOP for a batch of point clouds.
// Uses persistent pool->heap (output) and pool->scratch (scratch).

int gpu_kdop_hull(
    gpu_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          max_hull_verts,
    int          max_hull_tris,
    float*       out_verts,
    int*         out_tris,
    int*         out_nv,
    int*         out_nt,
    float*       out_volumes,
    int*         out_errors)
{
    if (!ctx || !ctx->fn_kdop_hull) return -1;
    if (!ctx->d_pool_struct) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "persistent heaps not initialized; call gpu_init first");
        return -1;
    }
    CUstream s = NULL;

    CUdeviceptr d_heap    = D_HEAP(ctx);
    CUdeviceptr d_scratch = D_SCRATCH(ctx);

    CUdeviceptr d_pts, d_off, d_overts, d_otris, d_onv, d_ont, d_ovols, d_oerr;
    CHECK_CU(cuMemAlloc(&d_pts,    (size_t)total_pts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_off,    (size_t)(n_hulls + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_otris,  (size_t)n_hulls * max_hull_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_onv,    (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_ont,    (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_ovols,  (size_t)n_hulls * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_oerr,   (size_t)n_hulls * sizeof(int)));

    CHECK_CU(cuMemcpyHtoDAsync(d_pts, pts,     (size_t)total_pts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls + 1) * sizeof(int),   s));

    int block_size = 32;  // KDOP_BLOCK
    int n_blocks   = n_hulls;

    void* args[] = {
        &d_pts, &d_off, &n_hulls,
        &max_hull_verts, &max_hull_tris,
        &d_overts, &d_otris, &d_onv, &d_ont, &d_ovols, &d_oerr,
        &d_heap, &d_scratch
    };
    CHECK_CU(cuLaunchKernel(ctx->fn_kdop_hull, n_blocks, 1, 1,
                            block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_verts,   d_overts, (size_t)n_hulls * max_hull_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_tris,    d_otris,  (size_t)n_hulls * max_hull_tris  * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyDtoHAsync(out_nv,      d_onv,    (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_nt,      d_ont,    (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_ovols,  (size_t)n_hulls * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_errors,  d_oerr,   (size_t)n_hulls * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts); cuMemFree(d_off);
    cuMemFree(d_overts); cuMemFree(d_otris);
    cuMemFree(d_onv); cuMemFree(d_ont);
    cuMemFree(d_ovols); cuMemFree(d_oerr);
    return 0;
}

// ---------------------------------------------------------------------------
// gpu_test_hausdorff
// ---------------------------------------------------------------------------

int gpu_test_hausdorff(
    gpu_ctx_t   ctx,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    const float* mesh_verts, int mesh_nv,
    const int*   mesh_tris,  int mesh_nt,
    float*       out_hausdorff)
{
    if (!ctx || !ctx->fn_hausdorff) return -1;
    CUstream s = NULL;

    CUdeviceptr d_hverts, d_htris, d_mverts, d_mtris, d_out, d_err;
    CUdeviceptr d_scratch = ctx->d_pool_struct + offsetof(struct DevicePool, scratch);

    CHECK_CU(cuMemAlloc(&d_hverts, (size_t)hull_nv * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_htris,  (size_t)hull_nt * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_mverts, (size_t)mesh_nv * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_mtris,  (size_t)mesh_nt * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out,    sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_err,    sizeof(int)));

    CHECK_CU(cuMemcpyHtoDAsync(d_hverts, hull_verts, (size_t)hull_nv * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_htris,  hull_tris,  (size_t)hull_nt * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyHtoDAsync(d_mverts, mesh_verts, (size_t)mesh_nv * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_mtris,  mesh_tris,  (size_t)mesh_nt * 3 * sizeof(int),   s));
    CHECK_CU(cuMemsetD32Async(d_err, 0, 1, s));

    void* args[] = {
        &d_hverts, &d_htris, &hull_nv, &hull_nt,
        &d_mverts, &d_mtris, &mesh_nv, &mesh_nt,
        &d_out, &d_scratch, &d_err
    };
    CHECK_CU(cuLaunchKernel(ctx->fn_hausdorff, 1, 1, 1, 256, 1, 1,
                            0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_hausdorff, d_out, sizeof(float), s));
    int h_err = 0;
    CHECK_CU(cuMemcpyDtoHAsync(&h_err, d_err, sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_hverts); cuMemFree(d_htris);
    cuMemFree(d_mverts); cuMemFree(d_mtris);
    cuMemFree(d_out); cuMemFree(d_err);

    if (h_err) {
        char errbuf[512];
        kerr_decode(h_err, errbuf, sizeof(errbuf));
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "hausdorff %s", errbuf);
        return h_err;
    }
    return 0;
}
