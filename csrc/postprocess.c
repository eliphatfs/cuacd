// Host-side launchers for post-processing GPU kernels.
// Uses CUDA driver API exclusively.

#include "postprocess.h"
#include "structs.h"
#include <cuda.h>
#include <stdio.h>

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed: %s", #call, _msg ? _msg : "unknown"); \
        goto cleanup_err; \
    } \
} while(0)

// Device heap address helpers (must match heap.c / test.c).
#define D_HEAP(ctx)    ((ctx)->d_pool_struct + offsetof(struct DevicePool, heap))
#define D_SCRATCH(ctx) ((ctx)->d_pool_struct + offsetof(struct DevicePool, scratch))

// ---------------------------------------------------------------------------
// gpu_test_postprocess_dc
// ---------------------------------------------------------------------------

int gpu_test_postprocess_dc(
    gpu_ctx_t   ctx,
    const float* verts,  int n_verts,
    const int*   tris,   int n_tris,
    int max_components, int max_verts_per, int max_tris_per,
    float* out_verts, int* out_tris,
    int* out_nv, int* out_nt,
    int* out_n_components)
{
    if (!ctx || !ctx->fn_test_postprocess_dc) return -1;
    if (!ctx->d_pool_struct) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "persistent heaps not initialized; call gpu_init first");
        return -1;
    }

    *out_n_components = 0;

    CUstream s = NULL;
    CUdeviceptr d_verts = 0, d_tris = 0;
    CUdeviceptr d_out_verts = 0, d_out_tris = 0;
    CUdeviceptr d_out_nv = 0, d_out_nt = 0;
    CUdeviceptr d_out_nc = 0, d_kerr = 0;

    size_t verts_bytes = (size_t)max_components * max_verts_per * 3 * sizeof(float);
    size_t tris_bytes  = (size_t)max_components * max_tris_per  * 3 * sizeof(int);
    size_t nv_bytes    = (size_t)max_components * sizeof(int);

    CHECK_CU(cuMemAlloc(&d_verts,     (size_t)n_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,      (size_t)n_tris  * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out_verts, verts_bytes));
    CHECK_CU(cuMemAlloc(&d_out_tris,  tris_bytes));
    CHECK_CU(cuMemAlloc(&d_out_nv,    nv_bytes));
    CHECK_CU(cuMemAlloc(&d_out_nt,    nv_bytes));
    CHECK_CU(cuMemAlloc(&d_out_nc,    sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_kerr,      sizeof(int)));

    CHECK_CU(cuMemcpyHtoDAsync(d_verts, verts, (size_t)n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,  tris,  (size_t)n_tris  * 3 * sizeof(int),   s));

    int zero_i = 0;
    CHECK_CU(cuMemcpyHtoDAsync(d_kerr,   &zero_i, sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_out_nc, &zero_i, sizeof(int), s));

    CUdeviceptr d_heap    = D_HEAP(ctx);
    CUdeviceptr d_scratch = D_SCRATCH(ctx);

    void* args[] = {
        &d_verts, &d_tris, &n_verts, &n_tris,
        &d_out_verts, &d_out_tris,
        &d_out_nv, &d_out_nt, &d_out_nc,
        &max_components, &max_verts_per, &max_tris_per,
        &d_heap, &d_scratch,
        &d_kerr
    };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_postprocess_dc,
        1, 1, 1, 128, 1, 1, 0, s, args, NULL));

    int kerr_h = 0;
    int nc_h = 0;
    CHECK_CU(cuMemcpyDtoHAsync(&kerr_h, d_kerr,   sizeof(int), s));
    CHECK_CU(cuMemcpyDtoHAsync(&nc_h,   d_out_nc, sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    if (kerr_h) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "postprocess_dc kernel error: 0x%x", kerr_h);
        goto cleanup_err;
    }

    *out_n_components = nc_h;

    if (nc_h > 0) {
        CHECK_CU(cuMemcpyDtoHAsync(out_nv, d_out_nv, (size_t)nc_h * sizeof(int), s));
        CHECK_CU(cuMemcpyDtoHAsync(out_nt, d_out_nt, (size_t)nc_h * sizeof(int), s));
        CHECK_CU(cuMemcpyDtoHAsync(out_verts, d_out_verts, verts_bytes, s));
        CHECK_CU(cuMemcpyDtoHAsync(out_tris,  d_out_tris,  tris_bytes,  s));
        CHECK_CU(cuStreamSynchronize(s));
    }

    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_out_verts); cuMemFree(d_out_tris);
    cuMemFree(d_out_nv); cuMemFree(d_out_nt);
    cuMemFree(d_out_nc); cuMemFree(d_kerr);
    return 0;

cleanup_err:
    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_out_verts); cuMemFree(d_out_tris);
    cuMemFree(d_out_nv); cuMemFree(d_out_nt);
    cuMemFree(d_out_nc); cuMemFree(d_kerr);
    return -1;
}
