// Host-side implementation for GPU beam search convex decomposition.
// Uses CUDA driver API exclusively — links only against libcuda.so.

#include "beam.h"
#include "structs.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Embedded fatbin — generated at build time by setup.py
#include "kernels_fatbin.h"

// Compile with -DCOACD_DEBUG=1 (or COACD_DEBUG=1 pip install -e .) for verbose host-side output.
#ifndef COACD_DEBUG
#define COACD_DEBUG 0
#endif
#if COACD_DEBUG
#define DBG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

// Must match cuda/common.cuh
#define BLOCK_SIZE 256

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at beam.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        return (int)_r; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Init / Destroy
// ---------------------------------------------------------------------------

int beam_init(beam_ctx_t* out, int device_ordinal) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return (int)r;

    beam_ctx_t ctx = (beam_ctx_t)calloc(1, sizeof(struct beam_ctx));
    if (!ctx) return -1;
    *out = ctx;

    CUcontext existing = NULL;
    cuCtxGetCurrent(&existing);

    if (existing && device_ordinal < 0) {
        ctx->cuda_ctx = existing;
        ctx->owns_context = 0;
        cuCtxGetDevice(&ctx->device);
    } else {
        int dev = (device_ordinal >= 0) ? device_ordinal : 0;
        CHECK_CU(cuDeviceGet(&ctx->device, dev));
        CHECK_CU(cuCtxCreate(&ctx->cuda_ctx, 0, ctx->device));
        ctx->owns_context = 1;
    }

    CHECK_CU(cuModuleLoadFatBinary(&ctx->module, kernels_fatbin));

    // Resolve kernel functions
    cuModuleGetFunction(&ctx->fn_test_warp_sort,        ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc,       ctx->module, "batch_hull_dandc");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc_mesh, ctx->module, "batch_hull_dandc_mesh");
    cuModuleGetFunction(&ctx->fn_query_dandc_scratch,   ctx->module, "query_dandc_scratch");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,      ctx->module, "batch_mesh_volume");
    cuModuleGetFunction(&ctx->fn_plane_cut,              ctx->module, "plane_cut_kernel");

    return 0;
}

void beam_destroy(beam_ctx_t ctx) {
    if (!ctx) return;
    if (ctx->scratch.base)   cuMemFree((CUdeviceptr)(uintptr_t)ctx->scratch.base);
    if (ctx->module) cuModuleUnload(ctx->module);
    if (ctx->owns_context && ctx->cuda_ctx) cuCtxDestroy(ctx->cuda_ctx);
    free(ctx);
}

const char* beam_last_error(beam_ctx_t ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return NULL;
    return ctx->last_error;
}

// ---------------------------------------------------------------------------
// beam_batch_hull_volume
// ---------------------------------------------------------------------------
// algo: 2=dandc (warp D&C Preparata-Hong)

int beam_batch_hull_volume(
    beam_ctx_t   ctx,
    const float* pts,
    int          total_pts,
    const int*   offsets,
    int          n_hulls,
    int          algo,
    int          max_pts_per_hull,
    float*       out_volumes,
    int*         out_errors)
{
    if (!ctx) return -1;
    if (algo != 2) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "batch_hull_volume: only algo 2 (D&C) is supported");
        return -1;
    }
    CUfunction fn = ctx->fn_batch_hull_dandc;

    CUstream s = NULL;
    CUdeviceptr d_pts, d_off, d_vols, d_errs, d_scratch = 0;
    CHECK_CU(cuMemAlloc(&d_pts,  (size_t)total_pts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_off,  (size_t)(n_hulls + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_vols, (size_t)n_hulls * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_errs, (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, pts, (size_t)total_pts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls + 1) * sizeof(int), s));

    // Query exact scratch from device via dandc_scratch_bytes()
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

    // D&C uses iterative stack — moderate thread stack for bt_merge call depth
    CHECK_CU(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8 * 1024));

    int block_size = 32;  // DANDC_BLOCK_SIZE — 1 warp per block
    int warps_per_block = block_size / 32;
    int n_blocks = (n_hulls + warps_per_block - 1) / warps_per_block;
    int scratch_per_i = (int)scratch_per;

    void* args[] = { &d_pts, &d_off, &d_vols, &d_errs,
                     &d_scratch, &scratch_per_i, &n_hulls };
    CHECK_CU(cuLaunchKernel(fn, n_blocks, 1, 1,
                            block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_vols, (size_t)n_hulls * sizeof(float), s));
    CHECK_CU(cuMemcpyDtoHAsync(out_errors,  d_errs, (size_t)n_hulls * sizeof(int),   s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts); cuMemFree(d_off); cuMemFree(d_vols); cuMemFree(d_errs);
    if (d_scratch) cuMemFree(d_scratch);
    return 0;
}

// ---------------------------------------------------------------------------
// beam_batch_mesh_volume
// ---------------------------------------------------------------------------

int beam_batch_mesh_volume(
    beam_ctx_t   ctx,
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

    void* args[] = { &d_verts, &d_tris, &d_toff, &d_voff, &d_vols, &n_meshes };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_mesh_volume,
        n_meshes, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_vols, (size_t)n_meshes * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts); cuMemFree(d_tris); cuMemFree(d_toff); cuMemFree(d_vols);
    if (d_voff) cuMemFree(d_voff);
    return 0;
}

