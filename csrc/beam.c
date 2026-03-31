// Host-side implementation for GPU beam search convex decomposition.
// Uses CUDA driver API exclusively — links only against libcuda.so.

#include "beam.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Embedded fatbin — generated at build time by setup.py
#include "kernels_fatbin.h"

// Must match beam_kernels.cu
#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64

struct DevicePool {
    char*         base;
    unsigned int* offset;
    unsigned int  capacity;
};

struct OutputPart {
    float* vertices;
    int*   triangles;
    int    n_verts;
    int    n_tris;
};

struct beam_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context;

    CUfunction fn_normalize_mesh;
    CUfunction fn_recover_coordinates;
    CUfunction fn_sample_surface;
    CUfunction fn_point_mesh_distance;
    CUfunction fn_reduce_max;
    CUfunction fn_pairwise_hausdorff;
    CUfunction fn_test_warp_sort;

    CUfunction fn_batch_hull_dandc;
    CUfunction fn_batch_hull_dandc_mesh;
    CUfunction fn_query_dandc_scratch;
    CUfunction fn_batch_mesh_volume;

    // V2 kernels
    CUfunction fn_beam_expansion;
    CUfunction fn_beam_hausdorff_parts;
    CUfunction fn_beam_termination;

    struct DevicePool scratch;

    struct OutputPart* output_parts;
    int num_output_parts;

    char last_error[256];
};

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
// Default parameters
// ---------------------------------------------------------------------------

void beam_params_default(beam_params_t* params) {
    params->beam_width = 16;
    params->cuts_per_axis = 15;
    params->threshold = 0.05f;
    params->rv_k = 0.3f;
    params->max_parts = 64;
    params->max_iterations = 64;
    params->hausdorff_samples = 1000;
}

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
    CHECK_CU(cuModuleGetFunction(&ctx->fn_normalize_mesh,        ctx->module, "normalize_mesh"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_recover_coordinates,   ctx->module, "recover_coordinates"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_sample_surface,        ctx->module, "sample_surface"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_point_mesh_distance,  ctx->module, "point_mesh_distance"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_reduce_max,           ctx->module, "reduce_max"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_pairwise_hausdorff,   ctx->module, "pairwise_hausdorff"));
    cuModuleGetFunction(&ctx->fn_test_warp_sort,        ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc,       ctx->module, "batch_hull_dandc");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc_mesh, ctx->module, "batch_hull_dandc_mesh");
    cuModuleGetFunction(&ctx->fn_query_dandc_scratch,   ctx->module, "query_dandc_scratch");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,      ctx->module, "batch_mesh_volume");

    // V2 kernels
    cuModuleGetFunction(&ctx->fn_beam_expansion,       ctx->module, "beam_expansion");
    cuModuleGetFunction(&ctx->fn_beam_hausdorff_parts, ctx->module, "beam_hausdorff_parts");
    cuModuleGetFunction(&ctx->fn_beam_termination,     ctx->module, "beam_termination");

    return 0;
}

static void free_output_parts(struct beam_ctx* ctx) {
    if (ctx->output_parts) {
        for (int i = 0; i < ctx->num_output_parts; i++) {
            free(ctx->output_parts[i].vertices);
            free(ctx->output_parts[i].triangles);
        }
        free(ctx->output_parts);
        ctx->output_parts = NULL;
        ctx->num_output_parts = 0;
    }
}

void beam_destroy(beam_ctx_t ctx) {
    if (!ctx) return;
    free_output_parts(ctx);
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
// Get results
// ---------------------------------------------------------------------------

int beam_get_num_parts(beam_ctx_t ctx) {
    if (!ctx) return 0;
    return ctx->num_output_parts;
}

int beam_get_part(
    beam_ctx_t ctx,
    int part_idx,
    float* out_vertices,
    int* out_n_verts,
    int* out_triangles,
    int* out_n_tris)
{
    if (!ctx || part_idx < 0 || part_idx >= ctx->num_output_parts) return -1;

    struct OutputPart* op = &ctx->output_parts[part_idx];

    if (!out_vertices || !out_triangles) {
        // Query sizes only
        if (out_n_verts) *out_n_verts = op->n_verts;
        if (out_n_tris)  *out_n_tris = op->n_tris;
        return 0;
    }

    if (*out_n_verts < op->n_verts || *out_n_tris < op->n_tris) {
        *out_n_verts = op->n_verts;
        *out_n_tris = op->n_tris;
        return -1;  // buffer too small
    }

    memcpy(out_vertices, op->vertices, op->n_verts * 3 * sizeof(float));
    memcpy(out_triangles, op->triangles, op->n_tris * 3 * sizeof(int));
    *out_n_verts = op->n_verts;
    *out_n_tris = op->n_tris;
    return 0;
}

// ---------------------------------------------------------------------------
// Hausdorff distance computation
// ---------------------------------------------------------------------------

static int reduce_max_host(beam_ctx_t ctx, CUdeviceptr d_data, int N,
                           float* result, CUstream s) {
    int n_blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    if (n_blocks < 1) n_blocks = 1;

    CUdeviceptr d_tmp;
    CHECK_CU(cuMemAlloc(&d_tmp, n_blocks * sizeof(float)));

    void* args1[] = { &d_data, &d_tmp, &N };
    unsigned int smem = BLOCK_SIZE * sizeof(float);
    CHECK_CU(cuLaunchKernel(ctx->fn_reduce_max,
        n_blocks, 1, 1, BLOCK_SIZE, 1, 1, smem, s, args1, NULL));

    int remaining = n_blocks;
    CUdeviceptr d_in = d_tmp, d_out = 0;

    while (remaining > 1) {
        int nb = (remaining + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        if (nb < 1) nb = 1;
        if (!d_out) CHECK_CU(cuMemAlloc(&d_out, nb * sizeof(float)));
        void* args[] = { &d_in, &d_out, &remaining };
        CHECK_CU(cuLaunchKernel(ctx->fn_reduce_max,
            nb, 1, 1, BLOCK_SIZE, 1, 1, BLOCK_SIZE * sizeof(float), s, args, NULL));
        remaining = nb;
        CUdeviceptr t = d_in; d_in = d_out; d_out = t;
    }

    CHECK_CU(cuMemcpyDtoHAsync(result, d_in, sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));
    cuMemFree(d_tmp);
    if (d_out) cuMemFree(d_out);
    return 0;
}

int beam_point_mesh_distances(
    beam_ctx_t ctx,
    const float* points, int n_points,
    const float* vertices, int n_verts,
    const int* triangles, int n_tris,
    float* distances)
{
    CUstream s = NULL;
    int vert_offset = 0;

    CUdeviceptr d_points, d_verts, d_tris, d_dist;
    CHECK_CU(cuMemAlloc(&d_points, n_points * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_verts,  n_verts  * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,   n_tris   * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_dist,   n_points * sizeof(float)));

    CHECK_CU(cuMemcpyHtoDAsync(d_points, points,    n_points * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts,  vertices,  n_verts  * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,   triangles, n_tris   * 3 * sizeof(int),   s));

    int grid = (n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_points, &d_verts, &d_tris, &d_dist, &n_points, &n_tris, &vert_offset };
    CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
        grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(distances, d_dist, n_points * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_points);
    cuMemFree(d_verts);
    cuMemFree(d_tris);
    cuMemFree(d_dist);
    return 0;
}

int beam_hausdorff(
    beam_ctx_t ctx,
    const float* samples_a, int n_sa,
    const float* vertices_a, int n_va,
    const int* triangles_a, int n_ta,
    const float* samples_b, int n_sb,
    const float* vertices_b, int n_vb,
    const int* triangles_b, int n_tb,
    float* result)
{
    CUstream s = NULL;
    int vert_offset = 0;

    CUdeviceptr d_sa, d_va, d_ta, d_sb, d_vb, d_tb, d_dist_a, d_dist_b;
    CHECK_CU(cuMemAlloc(&d_sa, n_sa * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_va, n_va * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_ta, n_ta * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_sb, n_sb * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_vb, n_vb * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tb, n_tb * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_dist_a, n_sa * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_dist_b, n_sb * sizeof(float)));

    CHECK_CU(cuMemcpyHtoDAsync(d_sa, samples_a,   n_sa * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_va, vertices_a,  n_va * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_ta, triangles_a, n_ta * 3 * sizeof(int),   s));
    CHECK_CU(cuMemcpyHtoDAsync(d_sb, samples_b,   n_sb * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_vb, vertices_b,  n_vb * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tb, triangles_b, n_tb * 3 * sizeof(int),   s));

    // B samples → mesh A
    {
        int grid = (n_sb + BLOCK_SIZE - 1) / BLOCK_SIZE;
        void* args[] = { &d_sb, &d_va, &d_ta, &d_dist_b, &n_sb, &n_ta, &vert_offset };
        CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
            grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }
    // A samples → mesh B
    {
        int grid = (n_sa + BLOCK_SIZE - 1) / BLOCK_SIZE;
        void* args[] = { &d_sa, &d_vb, &d_tb, &d_dist_a, &n_sa, &n_tb, &vert_offset };
        CHECK_CU(cuLaunchKernel(ctx->fn_point_mesh_distance,
            grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }

    float max_a, max_b;
    int r1 = reduce_max_host(ctx, d_dist_b, n_sb, &max_b, s);
    int r2 = reduce_max_host(ctx, d_dist_a, n_sa, &max_a, s);

    *result = (max_a > max_b) ? max_a : max_b;

    cuMemFree(d_sa);  cuMemFree(d_va);  cuMemFree(d_ta);
    cuMemFree(d_sb);  cuMemFree(d_vb);  cuMemFree(d_tb);
    cuMemFree(d_dist_a); cuMemFree(d_dist_b);

    return r1 ? r1 : r2;
}

int beam_pairwise_hausdorff(
    beam_ctx_t ctx,
    const float* all_samples,
    const int*   sample_offsets,
    const float* all_vertices,
    const int*   all_triangles,
    const int*   tri_offsets,
    const int*   vert_offsets,
    int          n_parts,
    float*       cost_matrix)
{
    CUstream s = NULL;

    int total_samples  = sample_offsets[n_parts];
    int total_verts    = vert_offsets[n_parts];
    int total_tris     = tri_offsets[n_parts];
    int n_pairs        = n_parts * (n_parts - 1) / 2;

    if (n_pairs == 0) {
        memset(cost_matrix, 0, n_parts * n_parts * sizeof(float));
        return 0;
    }

    int max_samples = 0;
    for (int i = 0; i < n_parts; i++) {
        int ns = sample_offsets[i + 1] - sample_offsets[i];
        if (ns > max_samples) max_samples = ns;
    }

    CUdeviceptr d_samples, d_soff, d_verts, d_tris, d_toff, d_voff, d_cost;
    CHECK_CU(cuMemAlloc(&d_samples, total_samples * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_soff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_verts,   total_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris,    total_tris * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_toff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_voff,    (n_parts + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_cost,    n_parts * n_parts * sizeof(float)));

    CHECK_CU(cuMemcpyHtoDAsync(d_samples, all_samples,    total_samples * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_soff,    sample_offsets, (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts,   all_vertices,   total_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris,    all_triangles,  total_tris * 3 * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_toff,    tri_offsets,     (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_voff,    vert_offsets,    (n_parts + 1) * sizeof(int), s));
    CHECK_CU(cuMemsetD8Async(d_cost, 0, n_parts * n_parts * sizeof(float), s));

    int sample_blocks = (max_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_samples, &d_soff, &d_verts, &d_tris, &d_toff, &d_voff, &d_cost, &n_parts };
    CHECK_CU(cuLaunchKernel(ctx->fn_pairwise_hausdorff,
        n_pairs, sample_blocks, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(cost_matrix, d_cost, n_parts * n_parts * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_samples); cuMemFree(d_soff);
    cuMemFree(d_verts);   cuMemFree(d_tris);
    cuMemFree(d_toff);    cuMemFree(d_voff);
    cuMemFree(d_cost);
    return 0;
}

// ---------------------------------------------------------------------------
// Test: warp_sort
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
// beam_run_v2 — 3-kernel architecture beam search
// ---------------------------------------------------------------------------

// PartInfoV2 and WorkItem must match common.cuh definitions
#define MAX_PARTS_PER_BEAM_V2 64

struct PartInfoV2_host {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    int hull_vert_offset, hull_vert_count;
    int hull_tri_offset, hull_tri_count;
    float bbox[6];
    float rv_cost;
    float hausdorff;
    float mesh_volume;
    float hull_volume;
};

struct WorkItem_host {
    int part_indices[MAX_PARTS_PER_BEAM_V2];
    int num_parts;
    int worst_part_idx;
    float worst_metric;
};

int beam_run_v2(
    beam_ctx_t ctx,
    const float* vertices, int n_verts,
    const int* triangles, int n_tris,
    const float* hull_verts, int n_hull_verts,
    const int* hull_tris, int n_hull_tris,
    float hull_volume,
    size_t scratch_size,
    const beam_params_t* params)
{
    CUstream s = NULL;

    beam_params_t p;
    if (params) p = *params;
    else beam_params_default(&p);

    int beam_width = p.beam_width;
    if (beam_width > MAX_BEAM) beam_width = MAX_BEAM;
    int cpa = p.cuts_per_axis;
    int num_planes = 3 * cpa;

    free_output_parts(ctx);

    // --- Memory layout: single large allocation ---
    // Vertex pool, triangle pool, PartInfoV2 array, WorkItem arrays (double-buffered),
    // cost buffer, work scratch, D&C scratch, atomic counters

    int grid_max        = beam_width * num_planes;
    // Hull mesh output limits (max vertices/triangles per extracted hull mesh).
    // These must be small enough that pool allocations don't overflow.
    int max_hull_verts_per_part = 64;
    int max_hull_tris_per_part  = 128;

    // Pool capacity: must hold all parts across all iterations.
    // Total expansion blocks = max_iterations * beam_width * num_planes + num_planes (first iter).
    // Each block creates 2 parts: 2 hull meshes + 2 mesh copies.
    int vert_capacity, tri_capacity, part_capacity;
    {
        long long max_blocks = (long long)p.max_iterations * beam_width * num_planes + num_planes;
        long long vc64 = (long long)(n_verts + n_hull_verts)
            + max_blocks * (max_hull_verts_per_part * 2 + (n_verts + 32) * 2);
        long long tc64 = (long long)(n_tris + n_hull_tris)
            + max_blocks * (max_hull_tris_per_part * 2 + (n_tris + 16) * 3);
        long long pc64 = 1 + max_blocks * 2;
        if (vc64 > 4*1024*1024) vc64 = 4*1024*1024;
        if (tc64 > 4*1024*1024) tc64 = 4*1024*1024;
        if (pc64 > 128*1024)    pc64 = 128*1024;
        // Minimum floor: original V1-style formula
        long long vc_min = (long long)(n_verts + n_hull_verts + 4096) * beam_width * 8;
        long long tc_min = (long long)(n_tris + n_hull_tris + 4096) * beam_width * 8;
        long long pc_min = (long long)beam_width * MAX_PARTS_PER_BEAM * 4;
        vert_capacity  = (int)(vc64 > vc_min ? vc64 : vc_min);
        tri_capacity   = (int)(tc64 > tc_min ? tc64 : tc_min);
        part_capacity  = (int)(pc64 > pc_min ? pc64 : pc_min);
    }

    // Compute total allocation
    size_t vp_bytes   = (size_t)vert_capacity * 3 * sizeof(float);
    size_t tp_bytes   = (size_t)tri_capacity * 3 * sizeof(int);
    size_t pp_bytes   = (size_t)part_capacity * sizeof(struct PartInfoV2_host);
    size_t wi_bytes   = (size_t)beam_width * sizeof(struct WorkItem_host) * 2; // double buffer
    size_t cost_bytes = (size_t)grid_max * sizeof(float);
    size_t ws_bytes   = (size_t)grid_max * sizeof(struct WorkItem_host);
    size_t counter_bytes = 4 * sizeof(unsigned int); // vert, tri, part, done counters
    size_t result_bytes  = sizeof(int);
    size_t norm_bytes    = 7 * sizeof(float);
    // Part indices for hausdorff
    size_t pidx_bytes = (size_t)part_capacity * sizeof(int);

    // Allocate everything separately for clarity
    CUdeviceptr d_vp, d_tp, d_pp, d_wi_a, d_wi_b, d_cost, d_ws;
    CUdeviceptr d_counters, d_result, d_norm, d_pidx, d_kerr;
    CUdeviceptr d_scratch_base, d_scratch_off;

    CHECK_CU(cuMemAlloc(&d_vp, vp_bytes));
    CHECK_CU(cuMemAlloc(&d_tp, tp_bytes));
    CHECK_CU(cuMemAlloc(&d_pp, pp_bytes));
    CHECK_CU(cuMemAlloc(&d_wi_a, (size_t)beam_width * sizeof(struct WorkItem_host)));
    CHECK_CU(cuMemAlloc(&d_wi_b, (size_t)beam_width * sizeof(struct WorkItem_host)));
    CHECK_CU(cuMemAlloc(&d_cost, cost_bytes));
    CHECK_CU(cuMemAlloc(&d_ws, ws_bytes));
    CHECK_CU(cuMemAlloc(&d_counters, counter_bytes));
    CHECK_CU(cuMemAlloc(&d_result, result_bytes));
    CHECK_CU(cuMemAlloc(&d_norm, norm_bytes));
    CHECK_CU(cuMemAlloc(&d_pidx, pidx_bytes));
    CHECK_CU(cuMemAlloc(&d_kerr, sizeof(int)));
    // Global scratch pool for split + D&C (2GB)
    {
        size_t free_mem = 0, total_mem = 0;
        cuMemGetInfo(&free_mem, &total_mem);
        size_t scratch_sz = (size_t)(free_mem * 0.7);
        if (scratch_sz < 256 * 1024 * 1024) scratch_sz = 256 * 1024 * 1024;
        if (scratch_sz > 4000000000ULL) scratch_sz = 4000000000ULL;
        if (scratch_size > 0) scratch_sz = scratch_size;
        CHECK_CU(cuMemAlloc(&d_scratch_base, scratch_sz));
        CHECK_CU(cuMemAlloc(&d_scratch_off, sizeof(unsigned int)));
        // Build host-side DevicePool struct (passed by value to kernels)
        ctx->scratch.base = (char*)(uintptr_t)d_scratch_base;
        ctx->scratch.offset = (unsigned int*)(uintptr_t)d_scratch_off;
        ctx->scratch.capacity = (unsigned int)scratch_sz;
    }

    // Set stack size for D&C
    CHECK_CU(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8 * 1024));

    // Upload mesh to vertex/triangle pool
    CHECK_CU(cuMemcpyHtoDAsync(d_vp, vertices, (size_t)n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tp, triangles, (size_t)n_tris * 3 * sizeof(int), s));

    // Upload hull mesh right after the original mesh
    CHECK_CU(cuMemcpyHtoDAsync(
        d_vp + (CUdeviceptr)((size_t)n_verts * 3 * sizeof(float)),
        hull_verts, (size_t)n_hull_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(
        d_tp + (CUdeviceptr)((size_t)n_tris * 3 * sizeof(int)),
        hull_tris, (size_t)n_hull_tris * 3 * sizeof(int), s));

    // Initialize counters: vert_counter = n_verts + n_hull_verts,
    //                       tri_counter = n_tris + n_hull_tris,
    //                       part_counter = 1, done_counter = 0
    {
        unsigned int counters[4];
        counters[0] = (unsigned int)(n_verts + n_hull_verts);
        counters[1] = (unsigned int)(n_tris + n_hull_tris);
        counters[2] = 1;  // 1 part already exists
        counters[3] = 0;  // done counter
        CHECK_CU(cuMemcpyHtoDAsync(d_counters, counters, sizeof(counters), s));
    }

    // Normalize mesh
    {
        int total_nv = n_verts + n_hull_verts;
        unsigned int smem = BLOCK_SIZE * 3 * sizeof(float) * 2;
        void* args[] = { &d_vp, &total_nv, &d_norm };
        CHECK_CU(cuLaunchKernel(ctx->fn_normalize_mesh,
            1, 1, 1, BLOCK_SIZE, 1, 1, smem, s, args, NULL));
    }

    // Compute initial Rv from hull_volume and mesh volume
    // mesh_volume: compute from signed tet volumes of the original mesh
    // For now, use hull_volume directly (the caller computes both via scipy)
    // The initial Rv is computed from the mesh volume and hull volume passed from Python

    // Initialize first PartInfoV2
    struct PartInfoV2_host h_part;
    memset(&h_part, 0, sizeof(h_part));
    h_part.vert_offset = 0;
    h_part.vert_count = n_verts;
    h_part.tri_offset = 0;
    h_part.tri_count = n_tris;
    h_part.hull_vert_offset = n_verts;
    h_part.hull_vert_count = n_hull_verts;
    h_part.hull_tri_offset = n_tris;
    h_part.hull_tri_count = n_hull_tris;
    for (int k = 0; k < 6; k++) h_part.bbox[k] = 0.0f;
    // Compute Rv from volumes
    {
        // We need mesh volume — compute it from the normalized mesh on GPU
        // Actually, hull_volume was computed on the original coordinates.
        // After normalization the volumes change by scale^3.
        // Simpler: just set rv_cost from the pre-computed hull_volume
        // and let the first Hausdorff kernel refine it.
        // For a first approximation: if hull_volume ~= mesh_volume, Rv ~= 0
        float diff = fabsf(hull_volume - hull_volume);  // This is 0, wrong
        // Actually we need the mesh volume too. The caller should pass it.
        // For now, compute a rough Rv: since hull_volume >= mesh_volume for convex,
        // and the caller passes hull_volume from scipy, approximate:
        h_part.rv_cost = 0.0f;  // Will be computed by compute_part_costs
        h_part.hausdorff = -1.0f;
        h_part.mesh_volume = 0.0f;  // unknown at this point
        h_part.hull_volume = hull_volume;
    }
    CHECK_CU(cuMemcpyHtoDAsync(d_pp, &h_part, sizeof(struct PartInfoV2_host), s));

    // Initialize first WorkItem
    struct WorkItem_host h_wi;
    memset(&h_wi, 0, sizeof(h_wi));
    h_wi.part_indices[0] = 0;
    h_wi.num_parts = 1;
    h_wi.worst_part_idx = 0;
    h_wi.worst_metric = 1e30f;
    CHECK_CU(cuMemcpyHtoDAsync(d_wi_a, &h_wi, sizeof(struct WorkItem_host), s));

    // Compute initial Rv via the existing compute_part_costs kernel
    // (Uses old structs — but we need to compute the initial Rv somehow)
    // Alternative: use the V1 compute_part_costs for the initial part,
    // or compute mesh volume on CPU. Let's use the GPU mesh volume kernel.
    {
        // Compute mesh volume using batch_mesh_volume kernel
        CUdeviceptr d_mv_tri_off, d_mv_vert_off, d_mv_vol;
        int tri_offsets[2] = {0, n_tris};
        int vert_offsets[2] = {0, 0};  // no rebasing, tris already have absolute indices
        CHECK_CU(cuMemAlloc(&d_mv_tri_off, 2 * sizeof(int)));
        CHECK_CU(cuMemAlloc(&d_mv_vert_off, 2 * sizeof(int)));
        CHECK_CU(cuMemAlloc(&d_mv_vol, sizeof(float)));
        CHECK_CU(cuMemcpyHtoDAsync(d_mv_tri_off, tri_offsets, 2 * sizeof(int), s));
        CHECK_CU(cuMemcpyHtoDAsync(d_mv_vert_off, vert_offsets, 2 * sizeof(int), s));

        int n_meshes = 1;
        void* mv_args[] = { &d_vp, &d_tp, &d_mv_tri_off, &d_mv_vert_off, &d_mv_vol, &n_meshes };
        CHECK_CU(cuLaunchKernel(ctx->fn_batch_mesh_volume,
            1, 1, 1, BLOCK_SIZE, 1, 1, 0, s, mv_args, NULL));

        float mesh_vol = 0.0f;
        CHECK_CU(cuMemcpyDtoHAsync(&mesh_vol, d_mv_vol, sizeof(float), s));
        CHECK_CU(cuStreamSynchronize(s));
        cuMemFree(d_mv_tri_off); cuMemFree(d_mv_vert_off); cuMemFree(d_mv_vol);

        // Now we have mesh_vol (in normalized coordinates)
        // hull_volume was passed in original coordinates; after normalization
        // volumes scale by (2/range)^3. We can't easily match.
        // Better approach: also compute hull volume on the normalized hull mesh.
        int hull_tri_offsets[2] = {n_tris, n_tris + n_hull_tris};
        int hull_vert_offsets[2] = {n_verts, n_verts};
        CHECK_CU(cuMemAlloc(&d_mv_tri_off, 2 * sizeof(int)));
        CHECK_CU(cuMemAlloc(&d_mv_vert_off, 2 * sizeof(int)));
        CHECK_CU(cuMemAlloc(&d_mv_vol, sizeof(float)));
        CHECK_CU(cuMemcpyHtoDAsync(d_mv_tri_off, hull_tri_offsets, 2 * sizeof(int), s));
        CHECK_CU(cuMemcpyHtoDAsync(d_mv_vert_off, hull_vert_offsets, 2 * sizeof(int), s));
        void* hv_args[] = { &d_vp, &d_tp, &d_mv_tri_off, &d_mv_vert_off, &d_mv_vol, &n_meshes };
        CHECK_CU(cuLaunchKernel(ctx->fn_batch_mesh_volume,
            1, 1, 1, BLOCK_SIZE, 1, 1, 0, s, hv_args, NULL));
        float hull_vol_norm = 0.0f;
        CHECK_CU(cuMemcpyDtoHAsync(&hull_vol_norm, d_mv_vol, sizeof(float), s));
        CHECK_CU(cuStreamSynchronize(s));
        cuMemFree(d_mv_tri_off); cuMemFree(d_mv_vert_off); cuMemFree(d_mv_vol);

        // Compute Rv
        float diff = fabsf(mesh_vol - hull_vol_norm);
        float rv = cbrtf(3.0f * diff / (4.0f * 3.14159265f)) * p.rv_k;

        h_part.rv_cost = rv;
        h_part.mesh_volume = mesh_vol;
        h_part.hull_volume = hull_vol_norm;
        h_part.hausdorff = -1.0f;
        CHECK_CU(cuMemcpyHtoDAsync(d_pp, &h_part, sizeof(struct PartInfoV2_host), s));

        h_wi.worst_metric = rv;
        CHECK_CU(cuMemcpyHtoDAsync(d_wi_a, &h_wi, sizeof(struct WorkItem_host), s));

        // Check if already convex
        if (rv <= p.threshold) {
            // Download and return single part (original mesh)
            ctx->num_output_parts = 1;
            ctx->output_parts = (struct OutputPart*)calloc(1, sizeof(struct OutputPart));
            ctx->output_parts[0].n_verts = n_verts;
            ctx->output_parts[0].n_tris = n_tris;
            ctx->output_parts[0].vertices = (float*)malloc(n_verts * 3 * sizeof(float));
            ctx->output_parts[0].triangles = (int*)malloc(n_tris * 3 * sizeof(int));
            memcpy(ctx->output_parts[0].vertices, vertices, n_verts * 3 * sizeof(float));
            memcpy(ctx->output_parts[0].triangles, triangles, n_tris * 3 * sizeof(int));
            goto cleanup;
        }
    }

    // --- Main beam search loop ---
    {
        CUdeviceptr d_wi_cur = d_wi_a;
        CUdeviceptr d_wi_nxt = d_wi_b;
        int num_items = 1;

        // Counters layout: [vert_counter, tri_counter, part_counter, done_counter]
        CUdeviceptr d_vert_ctr  = d_counters;
        CUdeviceptr d_tri_ctr   = d_counters + sizeof(unsigned int);
        CUdeviceptr d_part_ctr  = d_counters + 2 * sizeof(unsigned int);
        CUdeviceptr d_done_ctr  = d_counters + 3 * sizeof(unsigned int);

        // Set result to -1
        int neg1 = -1;
        CHECK_CU(cuMemcpyHtoDAsync(d_result, &neg1, sizeof(int), s));

        for (int iter = 0; iter < p.max_iterations; iter++) {
            // Kernel 2: Hausdorff for parts that need it
            // Build part index list on host
            {
                struct WorkItem_host h_items[MAX_BEAM];
                int n_to_read = (num_items < MAX_BEAM) ? num_items : MAX_BEAM;
                CHECK_CU(cuMemcpyDtoHAsync(h_items, d_wi_cur,
                    n_to_read * sizeof(struct WorkItem_host), s));
                CHECK_CU(cuStreamSynchronize(s));

                // Collect unique part indices
                int pidx[MAX_BEAM * MAX_PARTS_PER_BEAM_V2];
                int n_pidx = 0;
                for (int i = 0; i < n_to_read; i++) {
                    for (int j = 0; j < h_items[i].num_parts; j++) {
                        int gi = h_items[i].part_indices[j];
                        // Check uniqueness
                        int found = 0;
                        for (int k = 0; k < n_pidx; k++) {
                            if (pidx[k] == gi) { found = 1; break; }
                        }
                        if (!found && n_pidx < MAX_BEAM * MAX_PARTS_PER_BEAM_V2)
                            pidx[n_pidx++] = gi;
                    }
                }
                if (n_pidx > 0) {
                    CHECK_CU(cuMemcpyHtoDAsync(d_pidx, pidx, n_pidx * sizeof(int), s));
                    unsigned int kerr_zero = 0;
                    CHECK_CU(cuMemcpyHtoDAsync(d_kerr, &kerr_zero, sizeof(int), s));
                    void* h_args[] = { &d_vp, &d_tp, &d_pp, &d_pidx, &n_pidx,
                                       &p.threshold, &d_kerr };
                    CHECK_CU(cuLaunchKernel(ctx->fn_beam_hausdorff_parts,
                        n_pidx, 1, 1, HAUSDORFF_BLOCK_SIZE, 1, 1, 0, s, h_args, NULL));
                    // Async download kernel error
                    int kerr_h = 0;
                    CHECK_CU(cuMemcpyDtoHAsync(&kerr_h, d_kerr, sizeof(int), s));
                    CHECK_CU(cuStreamSynchronize(s));
                    if (kerr_h) {
                        snprintf(ctx->last_error, sizeof(ctx->last_error),
                            "beam_hausdorff_parts kernel error 0x%x at iter %d", kerr_h, iter);
                        goto cleanup;
                    }
                }
            }

            // Kernel 3: Termination check
            {
                int neg1_v = -1;
                unsigned int kerr_zero = 0;
                CHECK_CU(cuMemcpyHtoDAsync(d_result, &neg1_v, sizeof(int), s));
                CHECK_CU(cuMemcpyHtoDAsync(d_kerr, &kerr_zero, sizeof(int), s));
                void* t_args[] = { &d_pp, &d_wi_cur, &num_items, &p.threshold,
                                   &d_result, &d_kerr };
                CHECK_CU(cuLaunchKernel(ctx->fn_beam_termination,
                    num_items, 1, 1, 32, 1, 1, 0, s, t_args, NULL));
            }
            // Async download result + kernel error simultaneously
            int result_val = -1;
            int kerr_term = 0;
            CHECK_CU(cuMemcpyDtoHAsync(&result_val, d_result, sizeof(int), s));
            CHECK_CU(cuMemcpyDtoHAsync(&kerr_term, d_kerr, sizeof(int), s));
            CHECK_CU(cuStreamSynchronize(s));

            if (kerr_term) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                    "beam_termination kernel error 0x%x at iter %d", kerr_term, iter);
                goto cleanup;
            }

            if (result_val >= 0) {
                // Terminated — download the winning work item's parts
                break;
            }

            // Kernel 1: Expansion
            int grid_size = num_items * num_planes;
            {
                // Reset done counter, scratch pool offset, and kernel error
                unsigned int zero = 0;
                CHECK_CU(cuMemcpyHtoDAsync(d_done_ctr, &zero, sizeof(unsigned int), s));
                CHECK_CU(cuMemsetD32Async(d_scratch_off, 0, 1, s));
                CHECK_CU(cuMemcpyHtoDAsync(d_kerr, &zero, sizeof(int), s));

                struct DevicePool sp = ctx->scratch;
                unsigned int uvc = (unsigned int)vert_capacity;
                unsigned int utc = (unsigned int)tri_capacity;
                unsigned int upc = (unsigned int)part_capacity;
                void* e_args[] = {
                    &d_vp, &d_tp, &d_pp, &d_wi_cur,
                    &num_items, &cpa, &p.rv_k,
                    &d_vp, &d_tp, &d_pp,
                    &d_vert_ctr, &d_tri_ctr, &d_part_ctr,
                    &uvc, &utc, &upc,
                    &sp,
                    &d_ws, &d_cost,
                    &d_wi_nxt, &beam_width,
                    &d_done_ctr,
                    &max_hull_verts_per_part, &max_hull_tris_per_part,
                    &d_kerr
                };
                CHECK_CU(cuLaunchKernel(ctx->fn_beam_expansion,
                    grid_size, 1, 1, EXPANSION_BLOCK_SIZE, 1, 1, 0, s, e_args, NULL));
            }
            // Async download kernel error while GPU runs
            int kerr_exp = 0;
            CHECK_CU(cuMemcpyDtoHAsync(&kerr_exp, d_kerr, sizeof(int), s));
            CHECK_CU(cuStreamSynchronize(s));
            if (kerr_exp) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                    "beam_expansion kernel error 0x%x at iter %d (SCRATCH_OOM=1,SPLIT_OOM=2,HULL_PTS_OOM=4,POOL_OOM=8,HULL_ERR=16)",
                    kerr_exp, iter);
                goto cleanup;
            }

            // Swap work item buffers
            CUdeviceptr tmp = d_wi_cur;
            d_wi_cur = d_wi_nxt;
            d_wi_nxt = tmp;
            num_items = beam_width;
        }

        // Download results
        struct WorkItem_host h_final;
        CHECK_CU(cuMemcpyDtoHAsync(&h_final, d_wi_cur, sizeof(struct WorkItem_host), s));
        CHECK_CU(cuStreamSynchronize(s));

        int np = h_final.num_parts;
        if (np <= 0) np = 1;

        // Download all parts referenced by the winning work item
        struct PartInfoV2_host* h_parts = (struct PartInfoV2_host*)malloc(
            np * sizeof(struct PartInfoV2_host));
        for (int i = 0; i < np; i++) {
            CHECK_CU(cuMemcpyDtoHAsync(&h_parts[i],
                d_pp + (CUdeviceptr)(h_final.part_indices[i] * sizeof(struct PartInfoV2_host)),
                sizeof(struct PartInfoV2_host), s));
        }
        CHECK_CU(cuStreamSynchronize(s));

        // Recover coordinates
        {
            unsigned int h_vert_ctr;
            CHECK_CU(cuMemcpyDtoHAsync(&h_vert_ctr, d_vert_ctr, sizeof(unsigned int), s));
            CHECK_CU(cuStreamSynchronize(s));
            int total_verts = (int)h_vert_ctr;
            if (total_verts > 0) {
                int grid = (total_verts + BLOCK_SIZE - 1) / BLOCK_SIZE;
                void* rc_args[] = { &d_vp, &total_verts, &d_norm };
                CHECK_CU(cuLaunchKernel(ctx->fn_recover_coordinates,
                    grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, rc_args, NULL));
            }
        }

        // Download part meshes
        ctx->num_output_parts = np;
        ctx->output_parts = (struct OutputPart*)calloc(np, sizeof(struct OutputPart));

        for (int i = 0; i < np; i++) {
            struct PartInfoV2_host* pi = &h_parts[i];
            int nv = pi->vert_count;
            int nt = pi->tri_count;
            if (nv <= 0 || nt <= 0) {
                ctx->output_parts[i].n_verts = 0;
                ctx->output_parts[i].n_tris = 0;
                continue;
            }
            ctx->output_parts[i].n_verts = nv;
            ctx->output_parts[i].n_tris = nt;
            ctx->output_parts[i].vertices = (float*)malloc(nv * 3 * sizeof(float));
            ctx->output_parts[i].triangles = (int*)malloc(nt * 3 * sizeof(int));

            CUdeviceptr v_src = d_vp + (CUdeviceptr)((size_t)pi->vert_offset * 3 * sizeof(float));
            CUdeviceptr t_src = d_tp + (CUdeviceptr)((size_t)pi->tri_offset * 3 * sizeof(int));
            CHECK_CU(cuMemcpyDtoHAsync(ctx->output_parts[i].vertices, v_src,
                nv * 3 * sizeof(float), s));
            CHECK_CU(cuMemcpyDtoHAsync(ctx->output_parts[i].triangles, t_src,
                nt * 3 * sizeof(int), s));
        }
        CHECK_CU(cuStreamSynchronize(s));

        // Adjust triangle indices to be local (0-based) per part
        for (int i = 0; i < np; i++) {
            int vo_adj = h_parts[i].vert_offset;
            int nt = ctx->output_parts[i].n_tris;
            int* tris = ctx->output_parts[i].triangles;
            if (!tris) continue;
            for (int t = 0; t < nt * 3; t++)
                tris[t] -= vo_adj;
        }
        free(h_parts);
    }

cleanup:
    cuMemFree(d_vp); cuMemFree(d_tp); cuMemFree(d_pp);
    cuMemFree(d_wi_a); cuMemFree(d_wi_b);
    cuMemFree(d_cost); cuMemFree(d_ws);
    cuMemFree(d_counters); cuMemFree(d_result);
    cuMemFree(d_norm); cuMemFree(d_pidx); cuMemFree(d_kerr);
    cuMemFree(d_scratch_base); cuMemFree(d_scratch_off);
    return 0;
}
