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

// Compile with -DCOACD_DEBUG=1 (or COACD_DEBUG=1 pip install -e .) for verbose host-side output.
#ifndef COACD_DEBUG
#define COACD_DEBUG 0
#endif
#if COACD_DEBUG
#define DBG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

// Must match beam_kernels.cu
#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64

struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
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
    CUfunction fn_test_warp_sort;

    CUfunction fn_batch_hull_dandc;
    CUfunction fn_batch_hull_dandc_mesh;
    CUfunction fn_query_dandc_scratch;
    CUfunction fn_batch_mesh_volume;
    CUfunction fn_plane_cut;
    CUfunction fn_batch_plane_cut;
    CUfunction fn_batch_compact_mesh;
    CUfunction fn_batch_bbox;

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
    cuModuleGetFunction(&ctx->fn_test_warp_sort,        ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc,       ctx->module, "batch_hull_dandc");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc_mesh, ctx->module, "batch_hull_dandc_mesh");
    cuModuleGetFunction(&ctx->fn_query_dandc_scratch,   ctx->module, "query_dandc_scratch");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,      ctx->module, "batch_mesh_volume");
    cuModuleGetFunction(&ctx->fn_plane_cut,              ctx->module, "plane_cut_kernel");
    cuModuleGetFunction(&ctx->fn_batch_plane_cut,        ctx->module, "batch_plane_cut");
    cuModuleGetFunction(&ctx->fn_batch_compact_mesh,     ctx->module, "batch_compact_mesh");
    cuModuleGetFunction(&ctx->fn_batch_bbox,             ctx->module, "batch_bbox");

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

// ============================================================================
// Scratch size helpers (must match decomp.cu / plane_cut.cu formulas)
// ============================================================================

#define DC_WS_MAX_STACK 2048
#define DC_ALIGN16(x) (((int)(x) + 15) & ~15)

static size_t dc_pc_scratch_bytes(int nv, int nt) {
    int max_cross = nt * 2;
    int max_out   = nt * 3;
    int max_dir   = max_out * 3;
    int sort_scratch = max_cross * 8 + DC_WS_MAX_STACK * 2 * 4;
    int dir_sort     = max_dir   * 8 + DC_WS_MAX_STACK * 2 * 4;
    size_t s = 0;
    s += DC_ALIGN16(nv * 4);
    s += DC_ALIGN16((nv + nt * 2) * 12);
    s += DC_ALIGN16(max_cross * 8);
    s += DC_ALIGN16(sort_scratch);
    s += DC_ALIGN16(max_cross * 4);
    s += DC_ALIGN16(max_out * 3 * 4);
    s += DC_ALIGN16(max_out * 3 * 4);
    s += DC_ALIGN16(max_dir * 8);
    s += DC_ALIGN16(dir_sort);
    s += DC_ALIGN16(max_dir * 4);
    s += DC_ALIGN16(max_dir * 4);
    s += DC_ALIGN16(max_dir * 4);
    s += DC_ALIGN16(max_dir * 4);
    s += DC_ALIGN16((max_dir + 64) * 4);
    s += DC_ALIGN16(max_dir * 3 * 4);
    s += DC_ALIGN16(max_dir * 2 * 4);
    s += 64;
    return s;
}

static size_t dc_compact_scratch_bytes(int n_all_verts) {
    return (size_t)n_all_verts * 2 * sizeof(int) + 32;
}

#undef DC_ALIGN16

// ============================================================================
// Helper macros for functions with cleanup
// ============================================================================

#define CHECK_CU2(call) do { \
    CUresult _r2 = (call); \
    if (_r2 != CUDA_SUCCESS) { \
        const char* _m = NULL; cuGetErrorString(_r2, &_m); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at beam.c:%d: %s", #call, __LINE__, _m ? _m : "unknown"); \
        rc = (int)_r2; goto cleanup; \
    } \
} while(0)

// ============================================================================
// beam_batch_plane_cut
// ============================================================================

int beam_batch_plane_cut(
    beam_ctx_t   ctx,
    const float* verts,  int total_verts,
    const int*   tris,   int total_tris,
    const int*   cut_vert_offsets, const int* cut_vert_counts,
    const int*   cut_tri_offsets,  const int* cut_tri_counts,
    const float* plane_params,
    int          n_cuts,
    const int*   out_vert_offsets, const int* out_pos_offsets, const int* out_neg_offsets,
    const int*   out_vert_caps,    const int* out_pos_caps,    const int* out_neg_caps,
    int total_out_verts, int total_out_pos, int total_out_neg,
    float* out_verts, int* out_pos_tris, int* out_neg_tris,
    int* out_n_verts, int* out_n_pos, int* out_n_neg)
{
    if (!ctx || !ctx->fn_batch_plane_cut) return -1;
    int rc = 0;
    CUstream s = NULL;

    // Compute per-cut scratch sizes and offsets (host side)
    unsigned int* scratch_byte_offsets = (unsigned int*)malloc((size_t)n_cuts * sizeof(unsigned int));
    unsigned int* scratch_caps         = (unsigned int*)malloc((size_t)n_cuts * sizeof(unsigned int));
    if (!scratch_byte_offsets || !scratch_caps) {
        free(scratch_byte_offsets); free(scratch_caps); return -1;
    }
    size_t total_scratch = 0;
    for (int i = 0; i < n_cuts; i++) {
        size_t sz = dc_pc_scratch_bytes(cut_vert_counts[i], cut_tri_counts[i]);
        scratch_byte_offsets[i] = (unsigned int)total_scratch;
        scratch_caps[i]         = (unsigned int)sz;
        total_scratch += sz;
    }

    CUdeviceptr d_verts=0, d_tris=0, d_cvo=0, d_cvc=0, d_cto=0, d_ctc=0, d_pp=0;
    CUdeviceptr d_ovo=0, d_opo=0, d_ono=0, d_ovc=0, d_opc=0, d_onc=0;
    CUdeviceptr d_out_verts=0, d_out_pos=0, d_out_neg=0;
    CUdeviceptr d_nv=0, d_np=0, d_nn=0, d_kerrs=0;
    CUdeviceptr d_scratch=0, d_sbo=0, d_scaps=0, d_sull=0;

    CHECK_CU2(cuMemAllocAsync(&d_verts,     (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_tris,      (size_t)total_tris  * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemAllocAsync(&d_cvo,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_cvc,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_cto,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ctc,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_pp,        (size_t)n_cuts * 4 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_ovo,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_opo,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ono,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ovc,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_opc,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_onc,       (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_out_verts, (size_t)total_out_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_out_pos,   (size_t)total_out_pos   * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemAllocAsync(&d_out_neg,   (size_t)total_out_neg   * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemAllocAsync(&d_nv,        (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_np,        (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_nn,        (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_kerrs,     (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_scratch,   total_scratch, s));
    CHECK_CU2(cuMemAllocAsync(&d_sbo,       (size_t)n_cuts * sizeof(unsigned int), s));
    CHECK_CU2(cuMemAllocAsync(&d_scaps,     (size_t)n_cuts * sizeof(unsigned int), s));
    CHECK_CU2(cuMemAllocAsync(&d_sull,      (size_t)n_cuts * sizeof(unsigned long long), s));

    // Upload inputs
    CHECK_CU2(cuMemcpyHtoDAsync(d_verts, verts, (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_tris,  tris,  (size_t)total_tris  * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_cvo,   cut_vert_offsets, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_cvc,   cut_vert_counts,  (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_cto,   cut_tri_offsets,  (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ctc,   cut_tri_counts,   (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_pp,    plane_params,     (size_t)n_cuts * 4 * sizeof(float), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ovo,   out_vert_offsets, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_opo,   out_pos_offsets,  (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ono,   out_neg_offsets,  (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ovc,   out_vert_caps,    (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_opc,   out_pos_caps,     (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_onc,   out_neg_caps,     (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_sbo,   scratch_byte_offsets, (size_t)n_cuts * sizeof(unsigned int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_scaps, scratch_caps,         (size_t)n_cuts * sizeof(unsigned int), s));
    // Zero kernel errors and scratch ULL counters
    CHECK_CU2(cuMemsetD8Async(d_kerrs, 0, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemsetD8Async(d_sull,  0, (size_t)n_cuts * sizeof(unsigned long long), s));

    {
        void* args[] = {
            &d_verts, &d_tris,
            &d_cvo, &d_cvc, &d_cto, &d_ctc, &d_pp,
            &d_out_verts, &d_out_pos, &d_out_neg,
            &d_ovo, &d_opo, &d_ono, &d_ovc, &d_opc, &d_onc,
            &d_nv, &d_np, &d_nn, &d_kerrs,
            &d_scratch, &d_sbo, &d_scaps, &d_sull,
            &n_cuts
        };
        CHECK_CU2(cuLaunchKernel(ctx->fn_batch_plane_cut,
            n_cuts, 1, 1, 64, 1, 1, 0, s, args, NULL));
    }

    // Download counts first (small)
    CHECK_CU2(cuMemcpyDtoHAsync(out_n_verts, d_nv, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyDtoHAsync(out_n_pos,   d_np, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuMemcpyDtoHAsync(out_n_neg,   d_nn, (size_t)n_cuts * sizeof(int), s));
    CHECK_CU2(cuStreamSynchronize(s));

    // Download bulk output (only write what was actually produced)
    if (total_out_verts > 0)
        CHECK_CU2(cuMemcpyDtoH(out_verts,    d_out_verts, (size_t)total_out_verts * 3 * sizeof(float)));
    if (total_out_pos > 0)
        CHECK_CU2(cuMemcpyDtoH(out_pos_tris, d_out_pos,   (size_t)total_out_pos   * 3 * sizeof(int)));
    if (total_out_neg > 0)
        CHECK_CU2(cuMemcpyDtoH(out_neg_tris, d_out_neg,   (size_t)total_out_neg   * 3 * sizeof(int)));

cleanup:
    free(scratch_byte_offsets);
    free(scratch_caps);
    if (d_verts)     cuMemFreeAsync(d_verts,     s);
    if (d_tris)      cuMemFreeAsync(d_tris,      s);
    if (d_cvo)       cuMemFreeAsync(d_cvo,       s);
    if (d_cvc)       cuMemFreeAsync(d_cvc,       s);
    if (d_cto)       cuMemFreeAsync(d_cto,       s);
    if (d_ctc)       cuMemFreeAsync(d_ctc,       s);
    if (d_pp)        cuMemFreeAsync(d_pp,        s);
    if (d_ovo)       cuMemFreeAsync(d_ovo,       s);
    if (d_opo)       cuMemFreeAsync(d_opo,       s);
    if (d_ono)       cuMemFreeAsync(d_ono,       s);
    if (d_ovc)       cuMemFreeAsync(d_ovc,       s);
    if (d_opc)       cuMemFreeAsync(d_opc,       s);
    if (d_onc)       cuMemFreeAsync(d_onc,       s);
    if (d_out_verts) cuMemFreeAsync(d_out_verts, s);
    if (d_out_pos)   cuMemFreeAsync(d_out_pos,   s);
    if (d_out_neg)   cuMemFreeAsync(d_out_neg,   s);
    if (d_nv)        cuMemFreeAsync(d_nv,        s);
    if (d_np)        cuMemFreeAsync(d_np,        s);
    if (d_nn)        cuMemFreeAsync(d_nn,        s);
    if (d_kerrs)     cuMemFreeAsync(d_kerrs,     s);
    if (d_scratch)   cuMemFreeAsync(d_scratch,   s);
    if (d_sbo)       cuMemFreeAsync(d_sbo,       s);
    if (d_scaps)     cuMemFreeAsync(d_scaps,     s);
    if (d_sull)      cuMemFreeAsync(d_sull,      s);
    cuStreamSynchronize(s);
    return rc;
}

// ============================================================================
// beam_batch_compact_mesh
// ============================================================================

int beam_batch_compact_mesh(
    beam_ctx_t   ctx,
    const float* in_verts, int total_in_verts,
    const int*   in_tris,  int total_in_tris,
    const int*   in_vert_offsets, const int* in_vert_counts,
    const int*   in_tri_offsets,  const int* in_tri_counts,
    int          n_meshes,
    const int*   out_vert_offsets, const int* out_tri_offsets,
    int total_out_verts, int total_out_tris,
    float* out_verts, int* out_tris,
    float* out_bboxes, int* out_n_verts, int* out_n_tris)
{
    if (!ctx || !ctx->fn_batch_compact_mesh) return -1;
    int rc = 0;
    CUstream s = NULL;

    // Compute per-mesh scratch sizes
    unsigned int* scratch_byte_offsets = (unsigned int*)malloc((size_t)n_meshes * sizeof(unsigned int));
    unsigned int* scratch_caps         = (unsigned int*)malloc((size_t)n_meshes * sizeof(unsigned int));
    if (!scratch_byte_offsets || !scratch_caps) {
        free(scratch_byte_offsets); free(scratch_caps); return -1;
    }
    size_t total_scratch = 0;
    for (int i = 0; i < n_meshes; i++) {
        size_t sz = dc_compact_scratch_bytes(in_vert_counts[i]);
        scratch_byte_offsets[i] = (unsigned int)total_scratch;
        scratch_caps[i]         = (unsigned int)sz;
        total_scratch += sz;
    }

    CUdeviceptr d_iv=0, d_it=0, d_ivo=0, d_ivc=0, d_ito=0, d_itc=0;
    CUdeviceptr d_ov=0, d_ot=0, d_ovo=0, d_oto=0;
    CUdeviceptr d_bbox=0, d_nv=0, d_nt=0, d_kerrs=0;
    CUdeviceptr d_scratch=0, d_sbo=0, d_scaps=0, d_sull=0;

    CHECK_CU2(cuMemAllocAsync(&d_iv,     (size_t)total_in_verts  * 3 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_it,     (size_t)total_in_tris   * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemAllocAsync(&d_ivo,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ivc,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ito,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_itc,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_ov,     (size_t)total_out_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_ot,     (size_t)total_out_tris  * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemAllocAsync(&d_ovo,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_oto,    (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_bbox,   (size_t)n_meshes * 6 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_nv,     (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_nt,     (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_kerrs,  (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_scratch,total_scratch, s));
    CHECK_CU2(cuMemAllocAsync(&d_sbo,    (size_t)n_meshes * sizeof(unsigned int), s));
    CHECK_CU2(cuMemAllocAsync(&d_scaps,  (size_t)n_meshes * sizeof(unsigned int), s));
    CHECK_CU2(cuMemAllocAsync(&d_sull,   (size_t)n_meshes * sizeof(unsigned long long), s));

    CHECK_CU2(cuMemcpyHtoDAsync(d_iv,   in_verts,        (size_t)total_in_verts  * 3 * sizeof(float), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_it,   in_tris,         (size_t)total_in_tris   * 3 * sizeof(int),   s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ivo,  in_vert_offsets, (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ivc,  in_vert_counts,  (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ito,  in_tri_offsets,  (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_itc,  in_tri_counts,   (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_ovo,  out_vert_offsets,(size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_oto,  out_tri_offsets, (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_sbo,  scratch_byte_offsets,(size_t)n_meshes * sizeof(unsigned int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_scaps,scratch_caps,       (size_t)n_meshes * sizeof(unsigned int), s));
    CHECK_CU2(cuMemsetD8Async(d_kerrs, 0, (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemsetD8Async(d_sull,  0, (size_t)n_meshes * sizeof(unsigned long long), s));

    {
        void* args[] = {
            &d_iv, &d_it,
            &d_ivo, &d_ivc, &d_ito, &d_itc,
            &d_ov, &d_ot,
            &d_ovo, &d_oto,
            &d_bbox, &d_nv, &d_nt, &d_kerrs,
            &d_scratch, &d_sbo, &d_scaps, &d_sull,
            &n_meshes
        };
        CHECK_CU2(cuLaunchKernel(ctx->fn_batch_compact_mesh,
            n_meshes, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }

    CHECK_CU2(cuMemcpyDtoHAsync(out_bboxes,  d_bbox, (size_t)n_meshes * 6 * sizeof(float), s));
    CHECK_CU2(cuMemcpyDtoHAsync(out_n_verts, d_nv,   (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyDtoHAsync(out_n_tris,  d_nt,   (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuStreamSynchronize(s));
    if (total_out_verts > 0)
        CHECK_CU2(cuMemcpyDtoH(out_verts, d_ov, (size_t)total_out_verts * 3 * sizeof(float)));
    if (total_out_tris > 0)
        CHECK_CU2(cuMemcpyDtoH(out_tris,  d_ot, (size_t)total_out_tris  * 3 * sizeof(int)));

cleanup:
    free(scratch_byte_offsets);
    free(scratch_caps);
    if (d_iv)      cuMemFreeAsync(d_iv,      s);
    if (d_it)      cuMemFreeAsync(d_it,      s);
    if (d_ivo)     cuMemFreeAsync(d_ivo,     s);
    if (d_ivc)     cuMemFreeAsync(d_ivc,     s);
    if (d_ito)     cuMemFreeAsync(d_ito,     s);
    if (d_itc)     cuMemFreeAsync(d_itc,     s);
    if (d_ov)      cuMemFreeAsync(d_ov,      s);
    if (d_ot)      cuMemFreeAsync(d_ot,      s);
    if (d_ovo)     cuMemFreeAsync(d_ovo,     s);
    if (d_oto)     cuMemFreeAsync(d_oto,     s);
    if (d_bbox)    cuMemFreeAsync(d_bbox,    s);
    if (d_nv)      cuMemFreeAsync(d_nv,      s);
    if (d_nt)      cuMemFreeAsync(d_nt,      s);
    if (d_kerrs)   cuMemFreeAsync(d_kerrs,   s);
    if (d_scratch) cuMemFreeAsync(d_scratch, s);
    if (d_sbo)     cuMemFreeAsync(d_sbo,     s);
    if (d_scaps)   cuMemFreeAsync(d_scaps,   s);
    if (d_sull)    cuMemFreeAsync(d_sull,    s);
    cuStreamSynchronize(s);
    return rc;
}

// ============================================================================
// beam_batch_bbox
// ============================================================================

int beam_batch_bbox(
    beam_ctx_t   ctx,
    const float* verts, int total_verts,
    const int*   vert_offsets, const int* vert_counts,
    int n_meshes,
    float* out_bboxes)
{
    if (!ctx || !ctx->fn_batch_bbox) return -1;
    int rc = 0;
    CUstream s = NULL;

    CUdeviceptr d_v=0, d_vo=0, d_vc=0, d_bbox=0;

    CHECK_CU2(cuMemAllocAsync(&d_v,    (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemAllocAsync(&d_vo,   (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_vc,   (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemAllocAsync(&d_bbox, (size_t)n_meshes * 6 * sizeof(float), s));

    CHECK_CU2(cuMemcpyHtoDAsync(d_v,  verts,        (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_vo, vert_offsets, (size_t)n_meshes * sizeof(int), s));
    CHECK_CU2(cuMemcpyHtoDAsync(d_vc, vert_counts,  (size_t)n_meshes * sizeof(int), s));

    {
        void* args[] = { &d_v, &d_vo, &d_vc, &d_bbox, &n_meshes };
        CHECK_CU2(cuLaunchKernel(ctx->fn_batch_bbox,
            n_meshes, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }

    CHECK_CU2(cuMemcpyDtoHAsync(out_bboxes, d_bbox, (size_t)n_meshes * 6 * sizeof(float), s));
    CHECK_CU2(cuStreamSynchronize(s));

cleanup:
    if (d_v)    cuMemFreeAsync(d_v,    s);
    if (d_vo)   cuMemFreeAsync(d_vo,   s);
    if (d_vc)   cuMemFreeAsync(d_vc,   s);
    if (d_bbox) cuMemFreeAsync(d_bbox, s);
    cuStreamSynchronize(s);
    return rc;
}

// ============================================================================
// beam_decompose — GPU-resident beam-search convex decomposition
// All mesh data stays on GPU throughout iterations.
// Only metadata (hull_vol, mesh_vol, bbox, counts) is downloaded per iteration.
// Final hull meshes are extracted at the end only.
// ============================================================================

#define DC_MAX_PARTS     16384
#define DC_MAX_ITEMS      8192
#define DC_MAX_ITEM_PARTS  256
#define DC_MAX_EPOCHS      256

typedef struct {
    CUdeviceptr d_verts;    // absolute GPU pointer (into epoch buf or own alloc)
    CUdeviceptr d_tris;
    int n_verts, n_tris;
    float hull_vol, mesh_vol, rv;
    float bbox[6];          // xmin,xmax,ymin,ymax,zmin,zmax
    int epoch_idx;          // -1 = own GPU alloc; >= 0 = points into epoch buffer
    int ref_count;          // # of items referencing this part
    int alive;
} DCPart;

typedef struct {
    CUdeviceptr d_pos_verts, d_pos_tris;  // compact pos halves for all cuts
    CUdeviceptr d_neg_verts, d_neg_tris;  // compact neg halves for all cuts
    int ref_count;  // # of DCParts pointing into this epoch
    int alive;
} DCEpoch;

typedef struct {
    int part_ids[DC_MAX_ITEM_PARTS];
    int n_parts;
    float worst_rv;
    int worst_local;
} DCItem;

static int dc_alloc_part(DCPart* pool) {
    for (int i = 0; i < DC_MAX_PARTS; i++) {
        if (!pool[i].alive) {
            memset(&pool[i], 0, sizeof(DCPart));
            pool[i].alive = 1;
            pool[i].epoch_idx = -1;
            return i;
        }
    }
    return -1;
}

static void dc_release_part(DCPart* parts, DCEpoch* epochs, int pid) {
    if (pid < 0 || !parts[pid].alive) return;
    if (--parts[pid].ref_count > 0) return;
    if (parts[pid].epoch_idx < 0) {
        if (parts[pid].d_verts) cuMemFree(parts[pid].d_verts);
        if (parts[pid].d_tris)  cuMemFree(parts[pid].d_tris);
    } else {
        DCEpoch* e = &epochs[parts[pid].epoch_idx];
        if (e->alive && --e->ref_count <= 0) {
            if (e->d_pos_verts) cuMemFree(e->d_pos_verts);
            if (e->d_pos_tris)  cuMemFree(e->d_pos_tris);
            if (e->d_neg_verts) cuMemFree(e->d_neg_verts);
            if (e->d_neg_tris)  cuMemFree(e->d_neg_tris);
            e->alive = 0;
        }
    }
    parts[pid].alive = 0;
}

// ---------------------------------------------------------------------------
// dc_run_compact: batch compact mesh on GPU-resident plane-cut output.
// Input verts/tris in shared-vert pool format; output to epoch GPU buffers.
// Downloads actual_nv and bboxes to CPU; epoch GPU buffers stay alive.
// ---------------------------------------------------------------------------
static int dc_run_compact(
    beam_ctx_t ctx, CUstream s, int n_cuts,
    CUdeviceptr d_iv, int total_iv,   // shared vert pool (d_cut_v)
    CUdeviceptr d_it, int total_it,   // tri pool (d_cut_pos or d_cut_neg)
    const int* in_vo, const int* in_vc,   // in vert offsets/counts [n_cuts]
    const int* in_to, const int* in_tc,   // in tri  offsets/counts [n_cuts]
    CUdeviceptr d_epoch_v, CUdeviceptr d_epoch_t,
    const int* out_vo, const int* out_to, // output offsets [n_cuts]
    float* out_bboxes,     // [n_cuts * 6] CPU output
    int*   out_actual_nv,  // [n_cuts]
    int*   out_actual_nt)  // [n_cuts]
{
    if (n_cuts == 0) return 0;
    int rc = 0;

    unsigned int* h_sbo   = (unsigned int*)malloc(n_cuts * sizeof(unsigned int));
    unsigned int* h_scaps = (unsigned int*)malloc(n_cuts * sizeof(unsigned int));
    if (!h_sbo || !h_scaps) { free(h_sbo); free(h_scaps); return -1; }
    size_t total_scratch = 0;
    for (int i = 0; i < n_cuts; i++) {
        size_t sz = dc_compact_scratch_bytes(in_vc[i]);
        h_sbo[i]   = (unsigned int)total_scratch;
        h_scaps[i] = (unsigned int)sz;
        total_scratch += sz;
    }

    CUdeviceptr d_ivo=0,d_ivc=0,d_ito=0,d_itc=0;
    CUdeviceptr d_ovo=0,d_oto=0;
    CUdeviceptr d_bbox=0,d_nv=0,d_nt=0,d_kerrs=0;
    CUdeviceptr d_sc=0,d_sbo=0,d_scaps=0,d_sull=0;

#define DCC(c) do { CUresult _r=(c); if(_r!=CUDA_SUCCESS){ \
    const char* _m=NULL; cuGetErrorString(_r,&_m); \
    snprintf(ctx->last_error,sizeof(ctx->last_error),"%s:%d: %s",#c,__LINE__,_m?_m:"?"); \
    rc=(int)_r; goto dc_compact_done; } } while(0)

    DCC(cuMemAllocAsync(&d_ivo,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_ivc,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_ito,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_itc,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_ovo,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_oto,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_bbox, (size_t)n_cuts*6*sizeof(float), s));
    DCC(cuMemAllocAsync(&d_nv,   (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_nt,   (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemAllocAsync(&d_kerrs,(size_t)n_cuts*sizeof(int), s));
    if (total_scratch > 0) DCC(cuMemAllocAsync(&d_sc, total_scratch, s));
    DCC(cuMemAllocAsync(&d_sbo,  (size_t)n_cuts*sizeof(unsigned int), s));
    DCC(cuMemAllocAsync(&d_scaps,(size_t)n_cuts*sizeof(unsigned int), s));
    DCC(cuMemAllocAsync(&d_sull, (size_t)n_cuts*sizeof(unsigned long long), s));

    DCC(cuMemcpyHtoDAsync(d_ivo,  in_vo,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_ivc,  in_vc,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_ito,  in_to,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_itc,  in_tc,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_ovo,  out_vo, (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_oto,  out_to, (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyHtoDAsync(d_sbo,  h_sbo,  (size_t)n_cuts*sizeof(unsigned int), s));
    DCC(cuMemcpyHtoDAsync(d_scaps,h_scaps,(size_t)n_cuts*sizeof(unsigned int), s));
    DCC(cuMemsetD8Async(d_kerrs, 0, (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemsetD8Async(d_sull,  0, (size_t)n_cuts*sizeof(unsigned long long), s));

    {
        void* args[] = {
            &d_iv, &d_it,
            &d_ivo, &d_ivc, &d_ito, &d_itc,
            &d_epoch_v, &d_epoch_t,
            &d_ovo, &d_oto,
            &d_bbox, &d_nv, &d_nt, &d_kerrs,
            &d_sc, &d_sbo, &d_scaps, &d_sull,
            &n_cuts
        };
        DCC(cuLaunchKernel(ctx->fn_batch_compact_mesh,
            n_cuts, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }
    DCC(cuMemcpyDtoHAsync(out_bboxes,   d_bbox,(size_t)n_cuts*6*sizeof(float), s));
    DCC(cuMemcpyDtoHAsync(out_actual_nv,d_nv,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuMemcpyDtoHAsync(out_actual_nt,d_nt,  (size_t)n_cuts*sizeof(int), s));
    DCC(cuStreamSynchronize(s));

dc_compact_done:
    free(h_sbo); free(h_scaps);
    if(d_ivo)   cuMemFreeAsync(d_ivo,   s);
    if(d_ivc)   cuMemFreeAsync(d_ivc,   s);
    if(d_ito)   cuMemFreeAsync(d_ito,   s);
    if(d_itc)   cuMemFreeAsync(d_itc,   s);
    if(d_ovo)   cuMemFreeAsync(d_ovo,   s);
    if(d_oto)   cuMemFreeAsync(d_oto,   s);
    if(d_bbox)  cuMemFreeAsync(d_bbox,  s);
    if(d_nv)    cuMemFreeAsync(d_nv,    s);
    if(d_nt)    cuMemFreeAsync(d_nt,    s);
    if(d_kerrs) cuMemFreeAsync(d_kerrs, s);
    if(d_sc)    cuMemFreeAsync(d_sc,    s);
    if(d_sbo)   cuMemFreeAsync(d_sbo,   s);
    if(d_scaps) cuMemFreeAsync(d_scaps, s);
    if(d_sull)  cuMemFreeAsync(d_sull,  s);
    cuStreamSynchronize(s);
#undef DCC
    return rc;
}

// ---------------------------------------------------------------------------
// dc_run_hull_vol: batch D&C hull volume on GPU-resident dense pts.
// offsets[n_hulls+1]: host-side prefix sums (uploaded internally).
// ---------------------------------------------------------------------------
static int dc_run_hull_vol(
    beam_ctx_t ctx, CUstream s,
    CUdeviceptr d_pts, int total_pts,
    const int* offsets, int n_hulls, int max_pts,
    float* out_vols)  // [n_hulls] CPU output
{
    if (n_hulls == 0) return 0;
    (void)total_pts;
    int rc = 0;
    CUdeviceptr d_off=0,d_vols=0,d_errs=0,d_qout=0,d_scratch=0;

#define DCV(c) do { CUresult _r=(c); if(_r!=CUDA_SUCCESS){ \
    const char* _m=NULL; cuGetErrorString(_r,&_m); \
    snprintf(ctx->last_error,sizeof(ctx->last_error),"%s:%d: %s",#c,__LINE__,_m?_m:"?"); \
    rc=(int)_r; goto dc_hv_done; } } while(0)

    DCV(cuMemAllocAsync(&d_off,  (size_t)(n_hulls+1)*sizeof(int), s));
    DCV(cuMemAllocAsync(&d_vols, (size_t)n_hulls*sizeof(float), s));
    DCV(cuMemAllocAsync(&d_errs, (size_t)n_hulls*sizeof(int), s));
    DCV(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls+1)*sizeof(int), s));
    DCV(cuMemAllocAsync(&d_qout, sizeof(int), s));
    {
        int mpi = max_pts;
        void* qa[] = {&mpi, &d_qout};
        DCV(cuLaunchKernel(ctx->fn_query_dandc_scratch,1,1,1,1,1,1,0,s,qa,NULL));
    }
    {
        int sq = 0;
        DCV(cuMemcpyDtoHAsync(&sq, d_qout, sizeof(int), s));
        DCV(cuStreamSynchronize(s));
        cuMemFreeAsync(d_qout, s); d_qout = 0;
        DCV(cuMemAllocAsync(&d_scratch, (size_t)n_hulls*(size_t)sq, s));
        DCV(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8*1024));
        int spi = sq;
        void* ha[] = {&d_pts,&d_off,&d_vols,&d_errs,&d_scratch,&spi,&n_hulls};
        DCV(cuLaunchKernel(ctx->fn_batch_hull_dandc,
            n_hulls,1,1, 32,1,1, 0,s, ha, NULL));
    }
    DCV(cuMemcpyDtoHAsync(out_vols, d_vols, (size_t)n_hulls*sizeof(float), s));
    DCV(cuStreamSynchronize(s));

dc_hv_done:
    if(d_off)     cuMemFreeAsync(d_off,     s);
    if(d_vols)    cuMemFreeAsync(d_vols,    s);
    if(d_errs)    cuMemFreeAsync(d_errs,    s);
    if(d_qout)    cuMemFreeAsync(d_qout,    s);
    if(d_scratch) cuMemFreeAsync(d_scratch, s);
    cuStreamSynchronize(s);
#undef DCV
    return rc;
}

// ---------------------------------------------------------------------------
// dc_run_mesh_vol: batch mesh volume on GPU-resident data.
// d_tris uses relative indices; vert_offsets[i] = base vertex for mesh i.
// ---------------------------------------------------------------------------
static int dc_run_mesh_vol(
    beam_ctx_t ctx, CUstream s,
    CUdeviceptr d_verts, int total_verts,
    CUdeviceptr d_tris,  int total_tris,
    const int* tri_offsets, const int* vert_offsets,
    int n_meshes,
    float* out_vols)  // [n_meshes] CPU output
{
    if (n_meshes == 0) return 0;
    (void)total_verts; (void)total_tris;
    int rc = 0;
    CUdeviceptr d_toff=0,d_voff=0,d_vols=0;

#define DCM(c) do { CUresult _r=(c); if(_r!=CUDA_SUCCESS){ \
    const char* _m=NULL; cuGetErrorString(_r,&_m); \
    snprintf(ctx->last_error,sizeof(ctx->last_error),"%s:%d: %s",#c,__LINE__,_m?_m:"?"); \
    rc=(int)_r; goto dc_mv_done; } } while(0)

    DCM(cuMemAllocAsync(&d_toff,(size_t)(n_meshes+1)*sizeof(int), s));
    DCM(cuMemAllocAsync(&d_voff,(size_t)(n_meshes+1)*sizeof(int), s));
    DCM(cuMemAllocAsync(&d_vols,(size_t)n_meshes*sizeof(float), s));
    DCM(cuMemcpyHtoDAsync(d_toff,tri_offsets, (size_t)(n_meshes+1)*sizeof(int), s));
    DCM(cuMemcpyHtoDAsync(d_voff,vert_offsets,(size_t)(n_meshes+1)*sizeof(int), s));
    {
        void* ma[] = {&d_verts,&d_tris,&d_toff,&d_voff,&d_vols,&n_meshes};
        DCM(cuLaunchKernel(ctx->fn_batch_mesh_volume,
            n_meshes,1,1, BLOCK_SIZE,1,1, 0,s, ma, NULL));
    }
    DCM(cuMemcpyDtoHAsync(out_vols, d_vols, (size_t)n_meshes*sizeof(float), s));
    DCM(cuStreamSynchronize(s));

dc_mv_done:
    if(d_toff) cuMemFreeAsync(d_toff, s);
    if(d_voff) cuMemFreeAsync(d_voff, s);
    if(d_vols) cuMemFreeAsync(d_vols, s);
    cuStreamSynchronize(s);
#undef DCM
    return rc;
}

// ---------------------------------------------------------------------------
// beam_decompose: main entry point
// ---------------------------------------------------------------------------
int beam_decompose(
    beam_ctx_t ctx,
    const float* verts, int n_verts,
    const int*   tris,  int n_tris,
    const beam_params_t* params)
{
    if (!ctx) return -1;
    if (n_verts < 4 || n_tris < 4) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "beam_decompose: degenerate input (nv=%d nt=%d)", n_verts, n_tris);
        return -1;
    }
    free_output_parts(ctx);

    int beam_width    = params->beam_width    > 0 ? params->beam_width    : 8;
    int cuts_per_axis = params->cuts_per_axis > 0 ? params->cuts_per_axis : 10;
    int n_planes      = 3 * cuts_per_axis;
    float threshold   = params->threshold;
    int max_iters     = params->max_iterations > 0 ? params->max_iterations : 64;

    int rc = 0;
    CUstream s = NULL;

    DCPart*  parts  = (DCPart*) calloc(DC_MAX_PARTS,  sizeof(DCPart));
    DCItem*  items  = (DCItem*) calloc(DC_MAX_ITEMS,   sizeof(DCItem));
    DCEpoch* epochs = (DCEpoch*)calloc(DC_MAX_EPOCHS,  sizeof(DCEpoch));
    int n_items = 0, n_epochs = 0;

    if (!parts || !items || !epochs) {
        rc = -1;
        snprintf(ctx->last_error, sizeof(ctx->last_error), "beam_decompose: OOM");
        goto cleanup;
    }

    // -----------------------------------------------------------------------
    // Phase 0: Upload initial mesh, compute hull + mesh vol + bbox
    // -----------------------------------------------------------------------
    {
        CUdeviceptr d_v = 0, d_t = 0;
        CHECK_CU2(cuMemAllocAsync(&d_v,(size_t)n_verts*3*sizeof(float), s));
        CHECK_CU2(cuMemAllocAsync(&d_t,(size_t)n_tris *3*sizeof(int),   s));
        CHECK_CU2(cuMemcpyHtoDAsync(d_v, verts,(size_t)n_verts*3*sizeof(float), s));
        CHECK_CU2(cuMemcpyHtoDAsync(d_t, tris, (size_t)n_tris *3*sizeof(int),   s));
        CHECK_CU2(cuStreamSynchronize(s));

        float hull_vol0 = 0, mesh_vol0 = 0;
        int off2[2] = {0, n_verts};
        rc = dc_run_hull_vol(ctx, s, d_v, n_verts, off2, 1, n_verts, &hull_vol0);
        if (rc) goto cleanup;
        int toff2[2] = {0, n_tris}, voff2[2] = {0, n_verts};
        rc = dc_run_mesh_vol(ctx, s, d_v, n_verts, d_t, n_tris, toff2, voff2, 1, &mesh_vol0);
        if (rc) goto cleanup;

        // Bbox from CPU verts (available on host)
        float lo[3]={1e30f,1e30f,1e30f}, hi[3]={-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < n_verts; i++)
            for (int k = 0; k < 3; k++) {
                if (verts[i*3+k] < lo[k]) lo[k] = verts[i*3+k];
                if (verts[i*3+k] > hi[k]) hi[k] = verts[i*3+k];
            }

        if (hull_vol0 < 1e-12f) hull_vol0 = 1e-12f;

        int pid0 = dc_alloc_part(parts);
        if (pid0 < 0) { rc=-1; goto cleanup; }
        DCPart* p0 = &parts[pid0];
        p0->d_verts   = d_v;
        p0->d_tris    = d_t;
        p0->n_verts   = n_verts;
        p0->n_tris    = n_tris;
        p0->hull_vol  = hull_vol0;
        p0->mesh_vol  = mesh_vol0;
        p0->rv        = (hull_vol0 - mesh_vol0) / hull_vol0;
        p0->epoch_idx = -1;
        p0->ref_count = 1;
        p0->bbox[0]=lo[0]; p0->bbox[1]=hi[0];
        p0->bbox[2]=lo[1]; p0->bbox[3]=hi[1];
        p0->bbox[4]=lo[2]; p0->bbox[5]=hi[2];

        items[0].n_parts = 1;
        items[0].part_ids[0] = pid0;
        items[0].worst_rv = p0->rv;
        items[0].worst_local = 0;
        n_items = 1;
    }

    // -----------------------------------------------------------------------
    // Main beam-search loop
    // -----------------------------------------------------------------------
    for (int iter = 0; iter < max_iters && rc == 0; iter++) {

        // Step 1: Find worst part per item, check termination
        float best_worst = 1e30f;
        for (int ii = 0; ii < n_items; ii++) {
            float worst = -1e30f; int wl = 0;
            for (int k = 0; k < items[ii].n_parts; k++) {
                float rv = parts[items[ii].part_ids[k]].rv;
                if (rv > worst) { worst = rv; wl = k; }
            }
            items[ii].worst_rv    = worst;
            items[ii].worst_local = wl;
            if (worst < best_worst) best_worst = worst;
        }
        if (best_worst < threshold) break;

        // Prune: keep beam_width best (smallest worst_rv) via partial selection sort
        if (n_items > beam_width) {
            for (int i = 0; i < beam_width; i++) {
                int best = i;
                for (int j = i+1; j < n_items; j++)
                    if (items[j].worst_rv < items[best].worst_rv) best = j;
                if (best != i) { DCItem tmp=items[i]; items[i]=items[best]; items[best]=tmp; }
            }
            for (int i = beam_width; i < n_items; i++)
                for (int k = 0; k < items[i].n_parts; k++)
                    dc_release_part(parts, epochs, items[i].part_ids[k]);
            n_items = beam_width;
        }

        // Step 2: Compute pack offsets + build cut plans (CPU only)
        int n_cuts = n_items * n_planes;
        int* iv_off = (int*)malloc(n_items * sizeof(int));
        int* it_off = (int*)malloc(n_items * sizeof(int));
        int* cut_vo = (int*)malloc(n_cuts * sizeof(int));
        int* cut_vc = (int*)malloc(n_cuts * sizeof(int));
        int* cut_to = (int*)malloc(n_cuts * sizeof(int));
        int* cut_tc = (int*)malloc(n_cuts * sizeof(int));
        float* plane_params = (float*)malloc(n_cuts * 4 * sizeof(float));
        int* out_vc_arr = (int*)malloc(n_cuts * sizeof(int));
        int* out_pc_arr = (int*)malloc(n_cuts * sizeof(int));
        int* out_nc_arr = (int*)malloc(n_cuts * sizeof(int));
        int* out_vo_arr = (int*)malloc(n_cuts * sizeof(int));
        int* out_po_arr = (int*)malloc(n_cuts * sizeof(int));
        int* out_no_arr = (int*)malloc(n_cuts * sizeof(int));
        unsigned int* h_pc_sbo   = (unsigned int*)malloc(n_cuts * sizeof(unsigned int));
        unsigned int* h_pc_scaps = (unsigned int*)malloc(n_cuts * sizeof(unsigned int));

        if (!iv_off||!it_off||!cut_vo||!cut_vc||!cut_to||!cut_tc||!plane_params||
            !out_vc_arr||!out_pc_arr||!out_nc_arr||!out_vo_arr||!out_po_arr||!out_no_arr||
            !h_pc_sbo||!h_pc_scaps) {
            free(iv_off); free(it_off); free(cut_vo); free(cut_vc); free(cut_to); free(cut_tc);
            free(plane_params); free(out_vc_arr); free(out_pc_arr); free(out_nc_arr);
            free(out_vo_arr); free(out_po_arr); free(out_no_arr); free(h_pc_sbo); free(h_pc_scaps);
            rc = -1; break;
        }

        int total_pack_v = 0, total_pack_t = 0;
        for (int ii = 0; ii < n_items; ii++) {
            int pid = items[ii].part_ids[items[ii].worst_local];
            iv_off[ii] = total_pack_v;
            it_off[ii] = total_pack_t;
            total_pack_v += parts[pid].n_verts;
            total_pack_t += parts[pid].n_tris;
        }

        int total_ov = 0, total_op = 0, total_on = 0;
        size_t pc_scratch_total = 0;
        for (int ii = 0; ii < n_items; ii++) {
            int pid = items[ii].part_ids[items[ii].worst_local];
            float* bb = parts[pid].bbox;
            int nv = parts[pid].n_verts, nt = parts[pid].n_tris;
            for (int local = 0; local < n_planes; local++) {
                int cid = ii * n_planes + local;
                cut_vo[cid] = iv_off[ii]; cut_vc[cid] = nv;
                cut_to[cid] = it_off[ii]; cut_tc[cid] = nt;
                out_vo_arr[cid] = total_ov; out_vc_arr[cid] = nv + nt*2;
                out_po_arr[cid] = total_op; out_pc_arr[cid] = nt*3;
                out_no_arr[cid] = total_on; out_nc_arr[cid] = nt*3;
                total_ov += nv + nt*2;
                total_op += nt*3;
                total_on += nt*3;
                int axis = local / cuts_per_axis;
                int lc   = local % cuts_per_axis;
                float mn = bb[axis*2], mx = bb[axis*2+1];
                float tpos = (mx-mn > 1e-8f) ?
                    mn + (lc+1.0f)/(cuts_per_axis+1.0f)*(mx-mn) : (mn+mx)*0.5f;
                plane_params[cid*4+0] = 0; plane_params[cid*4+1] = 0;
                plane_params[cid*4+2] = 0; plane_params[cid*4+3] = 0;
                plane_params[cid*4+axis] = 1.0f;
                plane_params[cid*4+3] = -tpos;
                size_t sz = dc_pc_scratch_bytes(nv, nt);
                h_pc_sbo[cid]   = (unsigned int)pc_scratch_total;
                h_pc_scaps[cid] = (unsigned int)sz;
                pc_scratch_total += sz;
            }
        }

        // Step 3: DtoD pack worst-part meshes into temp GPU buffer
        CUdeviceptr d_pack_v=0, d_pack_t=0;
        if (total_pack_v > 0) rc = (int)cuMemAllocAsync(&d_pack_v,(size_t)total_pack_v*3*sizeof(float),s);
        if (!rc && total_pack_t > 0) rc = (int)cuMemAllocAsync(&d_pack_t,(size_t)total_pack_t*3*sizeof(int),s);
        if (!rc) {
            for (int ii = 0; ii < n_items; ii++) {
                int pid = items[ii].part_ids[items[ii].worst_local];
                DCPart* wp = &parts[pid];
                if (wp->n_verts > 0)
                    cuMemcpyDtoDAsync(d_pack_v + (size_t)iv_off[ii]*3*sizeof(float),
                                      wp->d_verts, (size_t)wp->n_verts*3*sizeof(float), s);
                if (wp->n_tris > 0)
                    cuMemcpyDtoDAsync(d_pack_t + (size_t)it_off[ii]*3*sizeof(int),
                                      wp->d_tris,  (size_t)wp->n_tris *3*sizeof(int),   s);
            }
        }
        free(iv_off); free(it_off);

        // Step 4: batch_plane_cut — upload cut metadata + launch
        CUdeviceptr d_cut_v=0, d_cut_pos=0, d_cut_neg=0;
        CUdeviceptr d_nv_g=0, d_np_g=0, d_nn_g=0;
        CUdeviceptr d_cvo=0,d_cvc=0,d_cto=0,d_ctc=0,d_pp=0;
        CUdeviceptr d_ovo=0,d_opo=0,d_ono=0,d_ovc=0,d_opc=0,d_onc=0,d_kerrs=0;
        CUdeviceptr d_pc_sc=0,d_pc_sbo=0,d_pc_scaps=0,d_pc_sull=0;

        if (!rc && total_ov>0) rc=(int)cuMemAllocAsync(&d_cut_v,  (size_t)total_ov*3*sizeof(float),s);
        if (!rc && total_op>0) rc=(int)cuMemAllocAsync(&d_cut_pos,(size_t)total_op*3*sizeof(int),  s);
        if (!rc && total_on>0) rc=(int)cuMemAllocAsync(&d_cut_neg,(size_t)total_on*3*sizeof(int),  s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_nv_g, (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_np_g, (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_nn_g, (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_cvo,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_cvc,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_cto,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_ctc,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_pp,   (size_t)n_cuts*4*sizeof(float), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_ovo,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_opo,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_ono,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_ovc,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_opc,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_onc,  (size_t)n_cuts*sizeof(int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_kerrs,(size_t)n_cuts*sizeof(int), s);
        if (!rc && pc_scratch_total>0) rc=(int)cuMemAllocAsync(&d_pc_sc,   pc_scratch_total, s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_pc_sbo,  (size_t)n_cuts*sizeof(unsigned int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_pc_scaps,(size_t)n_cuts*sizeof(unsigned int), s);
        if (!rc) rc=(int)cuMemAllocAsync(&d_pc_sull, (size_t)n_cuts*sizeof(unsigned long long), s);

        if (!rc) {
            cuMemcpyHtoDAsync(d_cvo, cut_vo,      (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_cvc, cut_vc,      (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_cto, cut_to,      (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_ctc, cut_tc,      (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_pp,  plane_params,(size_t)n_cuts*4*sizeof(float), s);
            cuMemcpyHtoDAsync(d_ovo, out_vo_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_opo, out_po_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_ono, out_no_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_ovc, out_vc_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_opc, out_pc_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_onc, out_nc_arr,  (size_t)n_cuts*sizeof(int), s);
            cuMemcpyHtoDAsync(d_pc_sbo,  h_pc_sbo,  (size_t)n_cuts*sizeof(unsigned int), s);
            cuMemcpyHtoDAsync(d_pc_scaps,h_pc_scaps,(size_t)n_cuts*sizeof(unsigned int), s);
            cuMemsetD8Async(d_kerrs, 0, (size_t)n_cuts*sizeof(int), s);
            cuMemsetD8Async(d_pc_sull,  0, (size_t)n_cuts*sizeof(unsigned long long), s);

            void* pc_args[] = {
                &d_pack_v, &d_pack_t,
                &d_cvo, &d_cvc, &d_cto, &d_ctc, &d_pp,
                &d_cut_v, &d_cut_pos, &d_cut_neg,
                &d_ovo, &d_opo, &d_ono, &d_ovc, &d_opc, &d_onc,
                &d_nv_g, &d_np_g, &d_nn_g, &d_kerrs,
                &d_pc_sc, &d_pc_sbo, &d_pc_scaps, &d_pc_sull,
                &n_cuts
            };
            rc = (int)cuLaunchKernel(ctx->fn_batch_plane_cut,
                n_cuts,1,1, 64,1,1, 0,s, pc_args, NULL);
        }
        free(cut_vo); free(cut_vc); free(cut_to); free(cut_tc); free(plane_params);
        free(out_vc_arr); free(out_pc_arr); free(out_nc_arr);
        free(h_pc_sbo); free(h_pc_scaps);

        // Step 5: Download counts, free per-cut metadata + scratch
        int* pc_nv = (int*)calloc(n_cuts, sizeof(int));
        int* pc_np = (int*)calloc(n_cuts, sizeof(int));
        int* pc_nn = (int*)calloc(n_cuts, sizeof(int));
        if (!rc && pc_nv && pc_np && pc_nn) {
            rc=(int)cuMemcpyDtoHAsync(pc_nv, d_nv_g,(size_t)n_cuts*sizeof(int),s);
            if(!rc) rc=(int)cuMemcpyDtoHAsync(pc_np, d_np_g,(size_t)n_cuts*sizeof(int),s);
            if(!rc) rc=(int)cuMemcpyDtoHAsync(pc_nn, d_nn_g,(size_t)n_cuts*sizeof(int),s);
            if(!rc) rc=(int)cuStreamSynchronize(s);
        }
        if(d_pack_v)   cuMemFreeAsync(d_pack_v,  s); d_pack_v=0;
        if(d_pack_t)   cuMemFreeAsync(d_pack_t,  s); d_pack_t=0;
        if(d_cvo)      cuMemFreeAsync(d_cvo,     s);
        if(d_cvc)      cuMemFreeAsync(d_cvc,     s);
        if(d_cto)      cuMemFreeAsync(d_cto,     s);
        if(d_ctc)      cuMemFreeAsync(d_ctc,     s);
        if(d_pp)       cuMemFreeAsync(d_pp,      s);
        if(d_ovo)      cuMemFreeAsync(d_ovo,     s);
        if(d_opo)      cuMemFreeAsync(d_opo,     s);
        if(d_ono)      cuMemFreeAsync(d_ono,     s);
        if(d_ovc)      cuMemFreeAsync(d_ovc,     s);
        if(d_opc)      cuMemFreeAsync(d_opc,     s);
        if(d_onc)      cuMemFreeAsync(d_onc,     s);
        if(d_kerrs)    cuMemFreeAsync(d_kerrs,   s);
        if(d_pc_sc)    cuMemFreeAsync(d_pc_sc,   s);
        if(d_pc_sbo)   cuMemFreeAsync(d_pc_sbo,  s);
        if(d_pc_scaps) cuMemFreeAsync(d_pc_scaps,s);
        if(d_pc_sull)  cuMemFreeAsync(d_pc_sull, s);
        if(d_nv_g)     cuMemFreeAsync(d_nv_g,    s);
        if(d_np_g)     cuMemFreeAsync(d_np_g,    s);
        if(d_nn_g)     cuMemFreeAsync(d_nn_g,    s);

        if (!pc_nv || !pc_np || !pc_nn) rc = -1;
        if (rc) { free(pc_nv); free(pc_np); free(pc_nn); break; }

        // Step 6: Compute epoch output offsets (caps = pc_nv/pc_np/pc_nn)
        int* ep_pos_voff = (int*)malloc((n_cuts+1)*sizeof(int));
        int* ep_pos_toff = (int*)malloc((n_cuts+1)*sizeof(int));
        int* ep_neg_voff = (int*)malloc((n_cuts+1)*sizeof(int));
        int* ep_neg_toff = (int*)malloc((n_cuts+1)*sizeof(int));
        if (!ep_pos_voff||!ep_pos_toff||!ep_neg_voff||!ep_neg_toff) {
            free(ep_pos_voff); free(ep_pos_toff); free(ep_neg_voff); free(ep_neg_toff);
            free(pc_nv); free(pc_np); free(pc_nn); rc = -1; break;
        }
        ep_pos_voff[0]=ep_pos_toff[0]=ep_neg_voff[0]=ep_neg_toff[0]=0;
        for (int c = 0; c < n_cuts; c++) {
            ep_pos_voff[c+1] = ep_pos_voff[c] + pc_nv[c]; // vert cap = pc_nv
            ep_pos_toff[c+1] = ep_pos_toff[c] + pc_np[c];
            ep_neg_voff[c+1] = ep_neg_voff[c] + pc_nv[c]; // same vert cap
            ep_neg_toff[c+1] = ep_neg_toff[c] + pc_nn[c];
        }
        int total_ep_pv = ep_pos_voff[n_cuts], total_ep_pt = ep_pos_toff[n_cuts];
        int total_ep_nv = ep_neg_voff[n_cuts], total_ep_nt = ep_neg_toff[n_cuts];

        // Allocate new epoch
        int eidx = n_epochs;
        if (eidx >= DC_MAX_EPOCHS) {
            free(ep_pos_voff); free(ep_pos_toff); free(ep_neg_voff); free(ep_neg_toff);
            free(pc_nv); free(pc_np); free(pc_nn); rc = -1; break;
        }
        n_epochs++;
        DCEpoch* ep = &epochs[eidx];
        memset(ep, 0, sizeof(DCEpoch));
        ep->alive = 1; ep->ref_count = 0;

        if (total_ep_pv>0) rc=(int)cuMemAllocAsync(&ep->d_pos_verts,(size_t)total_ep_pv*3*sizeof(float),s);
        if (!rc&&total_ep_pt>0) rc=(int)cuMemAllocAsync(&ep->d_pos_tris, (size_t)total_ep_pt*3*sizeof(int),  s);
        if (!rc&&total_ep_nv>0) rc=(int)cuMemAllocAsync(&ep->d_neg_verts,(size_t)total_ep_nv*3*sizeof(float),s);
        if (!rc&&total_ep_nt>0) rc=(int)cuMemAllocAsync(&ep->d_neg_tris, (size_t)total_ep_nt*3*sizeof(int),  s);

        // Step 7: batch_compact_mesh for pos + neg (uses d_cut_v shared pool)
        float* pos_bboxes    = (float*)calloc(n_cuts*6, sizeof(float));
        float* neg_bboxes    = (float*)calloc(n_cuts*6, sizeof(float));
        int*   pos_actual_nv = (int*)  calloc(n_cuts,   sizeof(int));
        int*   pos_actual_nt = (int*)  calloc(n_cuts,   sizeof(int));
        int*   neg_actual_nv = (int*)  calloc(n_cuts,   sizeof(int));
        int*   neg_actual_nt = (int*)  calloc(n_cuts,   sizeof(int));

        if (!pos_bboxes||!neg_bboxes||!pos_actual_nv||!pos_actual_nt||
            !neg_actual_nv||!neg_actual_nt) { rc = -1; }

        if (!rc && total_ep_pv > 0 && total_ep_pt > 0) {
            rc = dc_run_compact(ctx, s, n_cuts,
                d_cut_v, total_ov, d_cut_pos, total_op,
                out_vo_arr, pc_nv, out_po_arr, pc_np,
                ep->d_pos_verts, ep->d_pos_tris,
                ep_pos_voff, ep_pos_toff,
                pos_bboxes, pos_actual_nv, pos_actual_nt);
        }
        if (!rc && total_ep_nv > 0 && total_ep_nt > 0) {
            rc = dc_run_compact(ctx, s, n_cuts,
                d_cut_v, total_ov, d_cut_neg, total_on,
                out_vo_arr, pc_nv, out_no_arr, pc_nn,
                ep->d_neg_verts, ep->d_neg_tris,
                ep_neg_voff, ep_neg_toff,
                neg_bboxes, neg_actual_nv, neg_actual_nt);
        }

        // Free plane-cut output buffers (no longer needed)
        if(d_cut_v)   cuMemFreeAsync(d_cut_v,   s); d_cut_v=0;
        if(d_cut_pos) cuMemFreeAsync(d_cut_pos, s); d_cut_pos=0;
        if(d_cut_neg) cuMemFreeAsync(d_cut_neg, s); d_cut_neg=0;
        cuStreamSynchronize(s);

        // Step 8: Dense-pack compact verts for hull/mesh volume
        // pos side: DtoD copy actual compact verts into dense buffer
        int* dense_pos_off = (int*)malloc((n_cuts+1)*sizeof(int));
        int* dense_neg_off = (int*)malloc((n_cuts+1)*sizeof(int));
        if (!dense_pos_off||!dense_neg_off) rc=-1;

        if (!rc) {
            dense_pos_off[0] = dense_neg_off[0] = 0;
            int max_pnv = 0, max_nnv = 0;
            for (int c = 0; c < n_cuts; c++) {
                dense_pos_off[c+1] = dense_pos_off[c] + pos_actual_nv[c];
                dense_neg_off[c+1] = dense_neg_off[c] + neg_actual_nv[c];
                if (pos_actual_nv[c] > max_pnv) max_pnv = pos_actual_nv[c];
                if (neg_actual_nv[c] > max_nnv) max_nnv = neg_actual_nv[c];
            }
            int total_dense_pv = dense_pos_off[n_cuts];
            int total_dense_nv = dense_neg_off[n_cuts];

            CUdeviceptr d_dense_pos=0, d_dense_neg=0;
            if (total_dense_pv>0) rc=(int)cuMemAllocAsync(&d_dense_pos,(size_t)total_dense_pv*3*sizeof(float),s);
            if (!rc&&total_dense_nv>0) rc=(int)cuMemAllocAsync(&d_dense_neg,(size_t)total_dense_nv*3*sizeof(float),s);

            if (!rc) {
                for (int c = 0; c < n_cuts; c++) {
                    if (pos_actual_nv[c] > 0)
                        cuMemcpyDtoDAsync(d_dense_pos+(size_t)dense_pos_off[c]*3*sizeof(float),
                            ep->d_pos_verts+(size_t)ep_pos_voff[c]*3*sizeof(float),
                            (size_t)pos_actual_nv[c]*3*sizeof(float), s);
                    if (neg_actual_nv[c] > 0)
                        cuMemcpyDtoDAsync(d_dense_neg+(size_t)dense_neg_off[c]*3*sizeof(float),
                            ep->d_neg_verts+(size_t)ep_neg_voff[c]*3*sizeof(float),
                            (size_t)neg_actual_nv[c]*3*sizeof(float), s);
                }
            }

            // Step 9: Hull volume + mesh volume for all compact halves
            float* pos_hvols = (float*)calloc(n_cuts, sizeof(float));
            float* neg_hvols = (float*)calloc(n_cuts, sizeof(float));
            float* pos_mvols = (float*)calloc(n_cuts, sizeof(float));
            float* neg_mvols = (float*)calloc(n_cuts, sizeof(float));

            if (!pos_hvols||!neg_hvols||!pos_mvols||!neg_mvols) rc=-1;

            if (!rc && total_dense_pv>0 && max_pnv>=4) {
                rc = dc_run_hull_vol(ctx, s, d_dense_pos, total_dense_pv,
                    dense_pos_off, n_cuts, max_pnv, pos_hvols);
            }
            if (!rc && total_dense_nv>0 && max_nnv>=4) {
                rc = dc_run_hull_vol(ctx, s, d_dense_neg, total_dense_nv,
                    dense_neg_off, n_cuts, max_nnv, neg_hvols);
            }
            // Mesh volume: uses epoch tris (relative), dense verts (base at dense_off)
            if (!rc && total_dense_pv>0 && ep_pos_toff[n_cuts]>0) {
                rc = dc_run_mesh_vol(ctx, s,
                    d_dense_pos, total_dense_pv,
                    ep->d_pos_tris, ep_pos_toff[n_cuts],
                    ep_pos_toff, dense_pos_off, n_cuts, pos_mvols);
            }
            if (!rc && total_dense_nv>0 && ep_neg_toff[n_cuts]>0) {
                rc = dc_run_mesh_vol(ctx, s,
                    d_dense_neg, total_dense_nv,
                    ep->d_neg_tris, ep_neg_toff[n_cuts],
                    ep_neg_toff, dense_neg_off, n_cuts, neg_mvols);
            }

            if(d_dense_pos) cuMemFreeAsync(d_dense_pos, s);
            if(d_dense_neg) cuMemFreeAsync(d_dense_neg, s);
            cuStreamSynchronize(s);

            // Step 10: Create new DCParts + work items
            if (!rc) {
                // Allocate new DCItems array
                int max_new = n_items * n_planes + 4;
                if (max_new > DC_MAX_ITEMS) max_new = DC_MAX_ITEMS;
                DCItem* new_items = (DCItem*)calloc(max_new, sizeof(DCItem));
                int n_new = 0;

                for (int ii = 0; ii < n_items && n_new < max_new; ii++) {
                    int wl  = items[ii].worst_local;
                    int wpid = items[ii].part_ids[wl];
                    // keep_parts = all parts except worst
                    int keep_ids[DC_MAX_ITEM_PARTS];
                    int n_keep = 0;
                    for (int k = 0; k < items[ii].n_parts && n_keep < DC_MAX_ITEM_PARTS; k++) {
                        if (k != wl) keep_ids[n_keep++] = items[ii].part_ids[k];
                    }

                    for (int local = 0; local < n_planes && n_new < max_new; local++) {
                        int cid = ii * n_planes + local;
                        int pnv = pos_actual_nv[cid], pnt = pos_actual_nt[cid];
                        int nnv = neg_actual_nv[cid], nnt = neg_actual_nt[cid];
                        float phv = pos_hvols[cid], pmv = pos_mvols[cid];
                        float nhv = neg_hvols[cid], nmv = neg_mvols[cid];

                        // Alloc pos part if valid
                        int pos_pid = -1, neg_pid = -1;
                        if (pnv >= 4 && pnt >= 4) {
                            if (phv < 1e-12f) phv = 1e-12f;
                            pos_pid = dc_alloc_part(parts);
                            if (pos_pid >= 0) {
                                DCPart* pp = &parts[pos_pid];
                                pp->d_verts  = ep->d_pos_verts + (size_t)ep_pos_voff[cid]*3*sizeof(float);
                                pp->d_tris   = ep->d_pos_tris  + (size_t)ep_pos_toff[cid]*3*sizeof(int);
                                pp->n_verts  = pnv;
                                pp->n_tris   = pnt;
                                pp->hull_vol = phv;
                                pp->mesh_vol = pmv;
                                pp->rv       = (phv - pmv) / phv;
                                pp->epoch_idx= eidx;
                                pp->ref_count= 0; // incremented when added to item
                                memcpy(pp->bbox, &pos_bboxes[cid*6], 6*sizeof(float));
                                ep->ref_count++;
                            }
                        }
                        if (nnv >= 4 && nnt >= 4) {
                            if (nhv < 1e-12f) nhv = 1e-12f;
                            neg_pid = dc_alloc_part(parts);
                            if (neg_pid >= 0) {
                                DCPart* np2 = &parts[neg_pid];
                                np2->d_verts  = ep->d_neg_verts + (size_t)ep_neg_voff[cid]*3*sizeof(float);
                                np2->d_tris   = ep->d_neg_tris  + (size_t)ep_neg_toff[cid]*3*sizeof(int);
                                np2->n_verts  = nnv;
                                np2->n_tris   = nnt;
                                np2->hull_vol = nhv;
                                np2->mesh_vol = nmv;
                                np2->rv       = (nhv - nmv) / nhv;
                                np2->epoch_idx= eidx;
                                np2->ref_count= 0;
                                memcpy(np2->bbox, &neg_bboxes[cid*6], 6*sizeof(float));
                                ep->ref_count++;
                            }
                        }

                        if (pos_pid < 0 && neg_pid < 0 && n_keep == 0) continue; // skip empty

                        DCItem* ni = &new_items[n_new++];
                        ni->n_parts = 0;
                        // Add kept parts (increment ref_count)
                        for (int k = 0; k < n_keep && ni->n_parts < DC_MAX_ITEM_PARTS; k++) {
                            int kpid = keep_ids[k];
                            parts[kpid].ref_count++;
                            ni->part_ids[ni->n_parts++] = kpid;
                        }
                        // Add new pos part
                        if (pos_pid >= 0 && ni->n_parts < DC_MAX_ITEM_PARTS) {
                            parts[pos_pid].ref_count++;
                            ni->part_ids[ni->n_parts++] = pos_pid;
                        }
                        // Add new neg part
                        if (neg_pid >= 0 && ni->n_parts < DC_MAX_ITEM_PARTS) {
                            parts[neg_pid].ref_count++;
                            ni->part_ids[ni->n_parts++] = neg_pid;
                        }
                    }

                    // Release old item's parts (worst part gets released; kept parts got new refs above)
                    // worst part: decrement (if ref_count reaches 0, freed)
                    dc_release_part(parts, epochs, wpid);
                    // kept parts: were incremented n_planes times above, now decrement old reference
                    for (int k = 0; k < n_keep; k++)
                        dc_release_part(parts, epochs, keep_ids[k]);
                }

                // Also release items that didn't expand (shouldn't happen, but safety)
                // (The old items array is replaced by new_items)

                // Prune new_items to beam_width
                // (Compute worst_rv for each new item)
                for (int i = 0; i < n_new; i++) {
                    float w = -1e30f; int wl2 = 0;
                    for (int k = 0; k < new_items[i].n_parts; k++) {
                        float rv = parts[new_items[i].part_ids[k]].rv;
                        if (rv > w) { w = rv; wl2 = k; }
                    }
                    new_items[i].worst_rv    = w;
                    new_items[i].worst_local = wl2;
                }

                // Replace items with new_items
                if (n_new <= beam_width) {
                    memcpy(items, new_items, (size_t)n_new * sizeof(DCItem));
                    n_items = n_new;
                } else {
                    // Partial sort: keep beam_width best
                    for (int i = 0; i < beam_width; i++) {
                        int best = i;
                        for (int j = i+1; j < n_new; j++)
                            if (new_items[j].worst_rv < new_items[best].worst_rv) best = j;
                        if (best != i) {
                            DCItem tmp = new_items[i]; new_items[i] = new_items[best]; new_items[best] = tmp;
                        }
                    }
                    for (int i = beam_width; i < n_new; i++)
                        for (int k = 0; k < new_items[i].n_parts; k++)
                            dc_release_part(parts, epochs, new_items[i].part_ids[k]);
                    memcpy(items, new_items, (size_t)beam_width * sizeof(DCItem));
                    n_items = beam_width;
                }
                free(new_items);
            }

            free(pos_hvols); free(neg_hvols); free(pos_mvols); free(neg_mvols);
        } // dense pack block

        free(dense_pos_off); free(dense_neg_off);
        free(pos_bboxes); free(neg_bboxes);
        free(pos_actual_nv); free(pos_actual_nt);
        free(neg_actual_nv); free(neg_actual_nt);
        free(ep_pos_voff); free(ep_pos_toff);
        free(ep_neg_voff); free(ep_neg_toff);
        free(pc_nv); free(pc_np); free(pc_nn);
        free(out_vo_arr); free(out_po_arr); free(out_no_arr);

        // If epoch has no references (e.g., all cuts degenerate), free it now
        if (ep->alive && ep->ref_count <= 0) {
            if(ep->d_pos_verts) cuMemFree(ep->d_pos_verts);
            if(ep->d_pos_tris)  cuMemFree(ep->d_pos_tris);
            if(ep->d_neg_verts) cuMemFree(ep->d_neg_verts);
            if(ep->d_neg_tris)  cuMemFree(ep->d_neg_tris);
            ep->alive = 0;
        }

        DBG("[iter %d] n_items=%d rc=%d\n", iter, n_items, rc);
    } // main loop

    // -----------------------------------------------------------------------
    // Final: Extract hull meshes for best item
    // -----------------------------------------------------------------------
    if (rc == 0 && n_items > 0) {
        // Find best item (smallest worst_rv)
        int best_ii = 0;
        for (int ii = 1; ii < n_items; ii++)
            if (items[ii].worst_rv < items[best_ii].worst_rv) best_ii = ii;

        DCItem* best = &items[best_ii];
        int nfp = best->n_parts;
        if (nfp > 0) {
            // Download all final part verts to CPU, then call batch_hull_dandc_mesh
            // Build packed CPU arrays
            int* ns = (int*)malloc(nfp * sizeof(int));
            int total_fv = 0;
            if (!ns) { rc = -1; goto cleanup; }
            for (int k = 0; k < nfp; k++) {
                ns[k] = parts[best->part_ids[k]].n_verts;
                total_fv += ns[k];
            }
            float* h_fverts = (float*)malloc((size_t)total_fv * 3 * sizeof(float));
            int*   h_foff   = (int*)  malloc((size_t)(nfp+1)  * sizeof(int));
            if (!h_fverts || !h_foff) { free(ns); free(h_fverts); free(h_foff); rc=-1; goto cleanup; }

            h_foff[0] = 0;
            for (int k = 0; k < nfp; k++) {
                DCPart* fp = &parts[best->part_ids[k]];
                cuMemcpyDtoH(h_fverts + (size_t)h_foff[k]*3,
                             fp->d_verts, (size_t)fp->n_verts*3*sizeof(float));
                h_foff[k+1] = h_foff[k] + fp->n_verts;
            }

            // Compute natural hull bounds
            int max_fv = 0;
            for (int k = 0; k < nfp; k++) if (ns[k] > max_fv) max_fv = ns[k];
            int max_ht = (max_fv > 2) ? (2*max_fv - 4) : 4;
            if (max_ht < 4) max_ht = 4;

            float* h_hvols   = (float*)calloc(nfp, sizeof(float));
            int*   h_herrs   = (int*)  calloc(nfp, sizeof(int));
            float* h_overts  = (float*)malloc((size_t)nfp * max_fv * 3 * sizeof(float));
            int*   h_otris   = (int*)  malloc((size_t)nfp * max_ht * 3 * sizeof(int));
            int*   h_ovc     = (int*)  calloc(nfp, sizeof(int));
            int*   h_otc     = (int*)  calloc(nfp, sizeof(int));

            if (h_hvols&&h_herrs&&h_overts&&h_otris&&h_ovc&&h_otc) {
                rc = beam_batch_hull_dandc_mesh(ctx,
                    h_fverts, total_fv, h_foff, nfp,
                    max_fv, max_fv, max_ht,
                    h_hvols, h_herrs,
                    h_overts, h_otris, h_ovc, h_otc);
            }

            if (!rc) {
                ctx->output_parts = (struct OutputPart*)calloc(nfp, sizeof(struct OutputPart));
                ctx->num_output_parts = 0;
                if (ctx->output_parts) {
                    for (int k = 0; k < nfp; k++) {
                        int nv = h_ovc[k], nt = h_otc[k];
                        if (nv < 4 || nt < 4) continue;
                        struct OutputPart* op = &ctx->output_parts[ctx->num_output_parts++];
                        op->n_verts = nv; op->n_tris = nt;
                        op->vertices  = (float*)malloc((size_t)nv * 3 * sizeof(float));
                        op->triangles = (int*)  malloc((size_t)nt * 3 * sizeof(int));
                        if (op->vertices && op->triangles) {
                            memcpy(op->vertices,  h_overts + (size_t)k*max_fv*3, (size_t)nv*3*sizeof(float));
                            memcpy(op->triangles, h_otris  + (size_t)k*max_ht*3, (size_t)nt*3*sizeof(int));
                        }
                    }
                }
            }

            free(ns); free(h_fverts); free(h_foff);
            free(h_hvols); free(h_herrs); free(h_overts); free(h_otris);
            free(h_ovc); free(h_otc);
        }
    }

cleanup:
    // Release all alive parts (frees GPU memory via dc_release_part)
    if (parts && epochs) {
        for (int i = 0; i < DC_MAX_PARTS; i++) {
            if (parts[i].alive) {
                parts[i].ref_count = 1; // force release
                dc_release_part(parts, epochs, i);
            }
        }
        // Free any remaining alive epochs directly
        for (int i = 0; i < DC_MAX_EPOCHS; i++) {
            if (epochs[i].alive) {
                if(epochs[i].d_pos_verts) cuMemFree(epochs[i].d_pos_verts);
                if(epochs[i].d_pos_tris)  cuMemFree(epochs[i].d_pos_tris);
                if(epochs[i].d_neg_verts) cuMemFree(epochs[i].d_neg_verts);
                if(epochs[i].d_neg_tris)  cuMemFree(epochs[i].d_neg_tris);
                epochs[i].alive = 0;
            }
        }
    }
    free(parts);
    free(items);
    free(epochs);
    return rc;
}

// ============================================================================
// GPU Plane Cut with Cap Triangulation
// ============================================================================
// Launches plane_cut_kernel (cuda/plane_cut.cu) — one block of 64 threads.
// Handles simple loop, ring (annular), and multi-hole boundary topologies.


// GPU kernel error bit flags (must match plane_cut.cu)
#define PC_KERR_SCRATCH_OOM 1
#define PC_KERR_POOL_OOM    2
#define PC_KERR_SORT_ERR    4

// Global ctx pointer set by py_test_plane_cut binding (avoids API change)
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
