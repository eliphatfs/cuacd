// Host-side implementation using CUDA driver API exclusively.
// Links only against libcuda.so (the driver), NOT libcudart.

#include "coacd_gpu.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Embedded fatbin — generated at build time by nvcc, converted to a C array
// by bin2c or xxd and linked in.
extern const unsigned char coacd_kernels_fatbin[];
extern const unsigned int  coacd_kernels_fatbin_len;

struct coacd_gpu_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context; // 1 if we created the context, 0 if reusing

    // Kernel handles
    CUfunction fn_point_mesh_distance;
    CUfunction fn_reduce_max;
    CUfunction fn_pairwise_hausdorff;

    char last_error[256];
};

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed: %s", #call, _msg ? _msg : "unknown"); \
        return (int)_r; \
    } \
} while(0)

#define CHECK_CU_NOCTX(call, err_ret) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) return (int)_r; \
} while(0)

// ---------------------------------------------------------------------------
// Init / Destroy
// ---------------------------------------------------------------------------

COACD_GPU_API int coacd_gpu_init(coacd_gpu_ctx_t* out, int device_ordinal) {
    // Ensure driver is initialized (safe to call multiple times)
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return (int)r;

    coacd_gpu_ctx_t ctx = (coacd_gpu_ctx_t)calloc(1, sizeof(struct coacd_gpu_ctx));
    if (!ctx) return -1;
    *out = ctx;

    // Check if there's already a CUDA context on this thread
    CUcontext existing = NULL;
    cuCtxGetCurrent(&existing);

    if (existing && device_ordinal < 0) {
        // Reuse existing context (e.g. from PyTorch)
        ctx->cuda_ctx = existing;
        ctx->owns_context = 0;
        cuCtxGetDevice(&ctx->device);
    } else {
        int dev = (device_ordinal >= 0) ? device_ordinal : 0;
        CHECK_CU(cuDeviceGet(&ctx->device, dev));
        CHECK_CU(cuCtxCreate(&ctx->cuda_ctx, 0, ctx->device));
        ctx->owns_context = 1;
    }

    // Load fatbin
    CHECK_CU(cuModuleLoadFatBinary(&ctx->module, coacd_kernels_fatbin));

    // Resolve kernels
    CHECK_CU(cuModuleGetFunction(&ctx->fn_point_mesh_distance, ctx->module, "point_mesh_distance"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_reduce_max,          ctx->module, "reduce_max"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_pairwise_hausdorff,  ctx->module, "pairwise_hausdorff"));

    return 0;
}

COACD_GPU_API void coacd_gpu_destroy(coacd_gpu_ctx_t ctx) {
    if (!ctx) return;
    if (ctx->module) cuModuleUnload(ctx->module);
    if (ctx->owns_context && ctx->cuda_ctx) cuCtxDestroy(ctx->cuda_ctx);
    free(ctx);
}

COACD_GPU_API const char* coacd_gpu_last_error(coacd_gpu_ctx_t ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return NULL;
    return ctx->last_error;
}

// ---------------------------------------------------------------------------
// Helper: multi-pass max reduction on GPU
// ---------------------------------------------------------------------------

static int reduce_max_gpu(coacd_gpu_ctx_t ctx, CUdeviceptr d_data, int N,
                          float* result, CUstream stream) {
    const int BLOCK = 256;

    // Allocate temp buffer for block results
    int n_blocks = (N + BLOCK * 2 - 1) / (BLOCK * 2);
    if (n_blocks < 1) n_blocks = 1;

    CUdeviceptr d_tmp;
    CHECK_CU(cuMemAlloc(&d_tmp, n_blocks * sizeof(float)));

    // First pass
    void* args1[] = { &d_data, &d_tmp, &N };
    unsigned int smem = BLOCK * sizeof(float);
    CHECK_CU(cuLaunchKernel(ctx->fn_reduce_max,
        n_blocks, 1, 1, BLOCK, 1, 1, smem, stream, args1, NULL));

    // Iterative passes until 1 element remains
    int remaining = n_blocks;
    CUdeviceptr d_in = d_tmp;
    CUdeviceptr d_out = 0;

    while (remaining > 1) {
        int nb = (remaining + BLOCK * 2 - 1) / (BLOCK * 2);
        if (nb < 1) nb = 1;

        if (!d_out) CHECK_CU(cuMemAlloc(&d_out, nb * sizeof(float)));

        void* args[] = { &d_in, &d_out, &remaining };
        CHECK_CU(cuLaunchKernel(ctx->fn_reduce_max,
            nb, 1, 1, BLOCK, 1, 1, BLOCK * sizeof(float), stream, args, NULL));

        remaining = nb;
        // Swap
        CUdeviceptr t = d_in; d_in = d_out; d_out = t;
    }

    CHECK_CU(cuMemcpyDtoHAsync(result, d_in, sizeof(float), stream));
    CHECK_CU(cuStreamSynchronize(stream));

    cuMemFree(d_tmp);
    if (d_out) cuMemFree(d_out);
    return 0;
}

// ---------------------------------------------------------------------------
// point_mesh_distances
// ---------------------------------------------------------------------------

COACD_GPU_API int coacd_gpu_point_mesh_distances(
    coacd_gpu_ctx_t ctx,
    const float* points, int n_points,
    const float* vertices, int n_verts,
    const int* triangles, int n_tris,
    float* distances, void* stream) {

    CUstream s = (CUstream)stream;

    // Allocate device memory
    CUdeviceptr d_points, d_verts, d_tris, d_dist;
    CHECK_CU(cuMemAlloc(&d_points, n_points * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_verts,  n_verts  * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,   n_tris   * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_dist,   n_points * sizeof(float)));

    // Upload
    CHECK_CU(cuMemcpyHtoDAsync(d_points, points,    n_points * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts,  vertices,  n_verts  * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,   triangles, n_tris   * 3 * sizeof(int),   s));

    // Launch
    const int BLOCK = 256;
    int grid = (n_points + BLOCK - 1) / BLOCK;
    void* args[] = { &d_points, &d_verts, &d_tris, &d_dist, &n_points, &n_tris };
    CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
        grid, 1, 1, BLOCK, 1, 1, 0, s, args, NULL));

    // Download
    CHECK_CU(cuMemcpyDtoHAsync(distances, d_dist, n_points * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_points);
    cuMemFree(d_verts);
    cuMemFree(d_tris);
    cuMemFree(d_dist);
    return 0;
}

// ---------------------------------------------------------------------------
// hausdorff
// ---------------------------------------------------------------------------

COACD_GPU_API int coacd_gpu_hausdorff(
    coacd_gpu_ctx_t ctx,
    const float* samples_a, int n_sa,
    const float* vertices_a, int n_va,
    const int* triangles_a, int n_ta,
    const float* samples_b, int n_sb,
    const float* vertices_b, int n_vb,
    const int* triangles_b, int n_tb,
    float* result, void* stream) {

    CUstream s = (CUstream)stream;

    // Alloc device
    CUdeviceptr d_sa, d_va, d_ta, d_sb, d_vb, d_tb, d_dist_a, d_dist_b;
    CHECK_CU(cuMemAlloc(&d_sa, n_sa * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_va, n_va * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_ta, n_ta * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_sb, n_sb * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_vb, n_vb * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tb, n_tb * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_dist_a, n_sa * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_dist_b, n_sb * sizeof(float)));

    // Upload all
    CHECK_CU(cuMemcpyHtoDAsync(d_sa, samples_a,   n_sa * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_va, vertices_a,  n_va * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_ta, triangles_a, n_ta * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyHtoDAsync(d_sb, samples_b,   n_sb * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_vb, vertices_b,  n_vb * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tb, triangles_b, n_tb * 3 * sizeof(int),   s));

    const int BLOCK = 256;

    // B samples → mesh A
    {
        int grid = (n_sb + BLOCK - 1) / BLOCK;
        void* args[] = { &d_sb, &d_va, &d_ta, &d_dist_b, &n_sb, &n_ta };
        CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
            grid, 1, 1, BLOCK, 1, 1, 0, s, args, NULL));
    }
    // A samples → mesh B
    {
        int grid = (n_sa + BLOCK - 1) / BLOCK;
        void* args[] = { &d_sa, &d_vb, &d_tb, &d_dist_a, &n_sa, &n_tb };
        CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
            grid, 1, 1, BLOCK, 1, 1, 0, s, args, NULL));
    }

    // Reduce each direction
    float max_a, max_b;
    int r1 = reduce_max_gpu(ctx, d_dist_b, n_sb, &max_b, s);
    int r2 = reduce_max_gpu(ctx, d_dist_a, n_sa, &max_a, s);

    *result = (max_a > max_b) ? max_a : max_b;

    cuMemFree(d_sa);  cuMemFree(d_va);  cuMemFree(d_ta);
    cuMemFree(d_sb);  cuMemFree(d_vb);  cuMemFree(d_tb);
    cuMemFree(d_dist_a); cuMemFree(d_dist_b);

    return r1 ? r1 : r2;
}

// ---------------------------------------------------------------------------
// pairwise_hausdorff
// ---------------------------------------------------------------------------

COACD_GPU_API int coacd_gpu_pairwise_hausdorff(
    coacd_gpu_ctx_t ctx,
    const float* all_samples,
    const int*   sample_offsets,
    const float* all_vertices,
    const int*   all_triangles,
    const int*   tri_offsets,
    const int*   vert_offsets,
    int          n_parts,
    float*       cost_matrix,
    void*        stream) {

    CUstream s = (CUstream)stream;

    // Compute sizes from offsets
    int total_samples  = sample_offsets[n_parts];
    int total_verts    = vert_offsets[n_parts];
    int total_tris     = tri_offsets[n_parts];
    int n_pairs        = n_parts * (n_parts - 1) / 2;

    if (n_pairs == 0) {
        memset(cost_matrix, 0, n_parts * n_parts * sizeof(float));
        return 0;
    }

    // Find max samples per part for grid sizing
    int max_samples = 0;
    for (int i = 0; i < n_parts; i++) {
        int ns = sample_offsets[i + 1] - sample_offsets[i];
        if (ns > max_samples) max_samples = ns;
    }

    // Alloc device
    CUdeviceptr d_samples, d_soff, d_verts, d_tris, d_toff, d_voff, d_cost;
    CHECK_CU(cuMemAlloc(&d_samples, total_samples * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_soff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_verts,   total_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,    total_tris * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_toff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_voff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_cost,    n_parts * n_parts * sizeof(float)));

    // Upload
    CHECK_CU(cuMemcpyHtoDAsync(d_samples, all_samples,    total_samples * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_soff,    sample_offsets, (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts,   all_vertices,   total_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,    all_triangles,  total_tris * 3 * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_toff,    tri_offsets,     (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_voff,    vert_offsets,    (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemsetD8Async(d_cost, 0, n_parts * n_parts * sizeof(float), s));

    // Launch
    const int BLOCK = 256;
    int sample_blocks = (max_samples + BLOCK - 1) / BLOCK;
    void* args[] = { &d_samples, &d_soff, &d_verts, &d_tris, &d_toff, &d_voff, &d_cost, &n_parts };
    CHECK_CU(cuLaunchKernel(ctx->fn_pairwise_hausdorff,
        n_pairs, sample_blocks, 1, BLOCK, 1, 1, 0, s, args, NULL));

    // Download
    CHECK_CU(cuMemcpyDtoHAsync(cost_matrix, d_cost, n_parts * n_parts * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_samples); cuMemFree(d_soff);
    cuMemFree(d_verts);   cuMemFree(d_tris);
    cuMemFree(d_toff);    cuMemFree(d_voff);
    cuMemFree(d_cost);
    return 0;
}
