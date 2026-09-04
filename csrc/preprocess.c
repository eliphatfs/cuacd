// preprocess.c — host orchestration of the GPU mesh preprocess
// (PaMO stage-1 port). Driver API only — no cuda_runtime, no torch.
//
// Pipeline (mirrors pamo/simp_cuda/pamo/__init__.py stage 1):
//   1. build normalized triangle soup on device
//        p_norm = ((p - vmin) / extent + band) / margin
//        band = 3/R, margin = 1 + 2*band   (scalar extent keeps aspect ratio)
//   2. cumesh2sdf rasterize: (R,R,R) UDF grid + axis collide flags
//   3. cumesh2sdf fill_signs: union-find flood fill → signed SDF
//   4. d -= 0.9/R  (pamo SDF shift: shrink the reconstructed surface slightly
//      inwards so thin walls stay separated)
//   5. PDMC forward: Dual Marching Cubes at iso = 0
//   6. download, invert normalization:  p = ((g+0.5)/R * margin - band) * extent + vmin
//
// The PDMC host flow mirrors pdmc src/cudualmc.cu forward(): count →
// exclusive-sum → index … with the cub::DeviceScan::ExclusiveSum passes
// replaced by host-side prefix sums (download / scan / upload), which keeps the
// build free of cub/thrust and preserves the driver-API-only constraint.

#include "preprocess.h"
#include "heap.h"
#include "structs.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stddef.h>

#define C2S_NTHREAD 512
#define PDMC_BLOCK  512

// Mirror of the device-side C2S_RasterizeResult (cuda/c2s_kernels.cu), used
// only to marshal the two pointers by value into the sign kernels.
typedef struct {
    float* gridDist;            // [R,R,R] distance (in normalized units), 1e9 far
    unsigned char* gridCollide;  // [R,R,R,3] axis-ray hit flags
} C2S_RasterizeResult;

// ---------------------------------------------------------------------------
// Driver-API helpers
// ---------------------------------------------------------------------------

#define PCHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at preprocess.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        goto fail; \
    } \
} while (0)

static CUresult p_launch1d(CUfunction fn, size_t total, unsigned int block, void** args,
                           CUstream s) {
    size_t blocks = (total + block - 1) / block;
    if (blocks == 0) blocks = 1;
    if (blocks > 2147483647u) return CUDA_ERROR_INVALID_VALUE;
    return cuLaunchKernel(fn, (unsigned int)blocks, 1, 1, block, 1, 1, 0, s, args, NULL);
}

// Host exclusive prefix sum: in[n] → out[n+1], out[n] = total.
static int p_exclusive_sum(const int* in, int n, long long* out) {
    long long acc = 0;
    for (int i = 0; i < n; i++) {
        out[i] = acc;
        acc += in[i];
    }
    out[n] = acc;
    return 0;
}

// ---------------------------------------------------------------------------
// cumesh2sdf rasterize + sign (driver-API port of rasterize_tris / fill_signs)
// ---------------------------------------------------------------------------

typedef struct {
    CUdeviceptr gridDist;    // R^3 floats
    CUdeviceptr gridCollide;  // R^3*3 bytes
    CUdeviceptr totalSize;   // 1 uint (grid tail)
    int R;
} p_rast_t;

// One rasterize_layer pass pair: probe (counts only) then fill with exact-size
// buffers. Mirrors rasterize_layer_internal (upstream) with the allocator
// replaced by plain cuMemAlloc/cuMemFree.
static int p_rasterize_layer(gpu_ctx_t ctx, CUfunction probe_fn, CUfunction fill_fn,
                             CUdeviceptr soup, int S, CUdeviceptr* io_idx, CUdeviceptr* io_grid,
                             int* io_M, int N, float band, CUdeviceptr totalSize, CUstream s) {
    int M = *io_M;
    CUdeviceptr idx = *io_idx, grid = *io_grid;
    CUdeviceptr tbofs = 0, outIdx = 0, outGrid = 0;
    int rc = 0;

    long long total_threads = (long long)S * S * S * M;
    unsigned int blocks = (unsigned int)((total_threads + C2S_NTHREAD - 1) / C2S_NTHREAD);
    if (blocks == 0) blocks = 1;
    if (blocks > 2147483647u) { rc = CUDA_ERROR_INVALID_VALUE; goto fail; }

    PCHECK_CU(cuMemAlloc(&tbofs, (size_t)blocks * sizeof(unsigned int)));
    PCHECK_CU(cuMemsetD32Async(totalSize, 0, 1, s));

    // Probe: counts only (outIdx/outGrid null, preloc 0 ⇒ no writes).
    {
        CUdeviceptr nullp = 0;
        unsigned int preloc0 = 0;
        void* args[] = { &soup, &idx, &grid, &M, &N, &band, &tbofs, &totalSize, &nullp, &nullp,
                        &preloc0 };
        PCHECK_CU(cuLaunchKernel(probe_fn, blocks, 1, 1, C2S_NTHREAD, 1, 1, 0, s, args, NULL));
    }

    unsigned int las = 0;
    PCHECK_CU(cuMemcpyDtoH(&las, totalSize, sizeof(unsigned int)));

    if (las > 0) {
        PCHECK_CU(cuMemAlloc(&outIdx, (size_t)las * sizeof(unsigned int)));
        PCHECK_CU(cuMemAlloc(&outGrid, (size_t)las * sizeof(unsigned int)));
        // Fill: preloc = las ⇒ every hit is written.
        void* args[] = { &soup, &idx, &grid, &M, &N, &band, &tbofs, &totalSize, &outIdx, &outGrid,
                         &las };
        PCHECK_CU(cuLaunchKernel(fill_fn, blocks, 1, 1, C2S_NTHREAD, 1, 1, 0, s, args, NULL));
    }

    PCHECK_CU(cuMemFree(idx));
    PCHECK_CU(cuMemFree(grid));
    idx = outIdx; outIdx = 0;
    grid = outGrid; outGrid = 0;
    *io_idx = idx; *io_grid = grid;
    *io_M = (int)las;
    PCHECK_CU(cuMemFree(tbofs));
    tbofs = 0;
    return 0;

fail:
    if (tbofs) cuMemFree(tbofs);
    if (outIdx) cuMemFree(outIdx);
    if (outGrid) cuMemFree(outGrid);
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}

static int p_rasterize_kernel(gpu_ctx_t ctx, CUdeviceptr d_soup, int F, int R, float band,
                              p_rast_t* out, CUstream s) {
    CUdeviceptr idx = 0, grid = 0;
    int rc = 0;
    out->gridDist = 0; out->gridCollide = 0; out->totalSize = 0;

    // Allocation sizes grow with the grid, so allocate the outputs first.
    PCHECK_CU(cuMemAlloc(&out->gridDist, (size_t)R * R * R * sizeof(float) + sizeof(unsigned int)));
    PCHECK_CU(cuMemAlloc(&out->gridCollide, (size_t)R * R * R * 3));
    out->totalSize = out->gridDist + (CUdeviceptr)((size_t)R * R * R * sizeof(float));
    out->R = R;

    // Subdivision schedule (upstream rasterize_tris).
    int SS[16];
    int nSS = 0;
    int N = R;
    if (N > 8) { SS[nSS++] = 8; N /= 8; }
    while (N > 4) { SS[nSS++] = 4; N /= 4; }
    SS[nSS++] = N;

    const int B = 65536;  // upstream driver.cu batch

    for (int j = 0; j < F; j += B) {
        int M = (F - j) < B ? (F - j) : B;

        PCHECK_CU(cuMemAlloc(&idx, (size_t)M * sizeof(unsigned int)));
        PCHECK_CU(cuMemAlloc(&grid, (size_t)M * sizeof(unsigned int)));
        {
            unsigned int ofs = (unsigned int)j;
            unsigned int Mu = (unsigned int)M;
            unsigned int startIdu = 0;
            void* a1[] = { &idx, &Mu, &ofs };
            PCHECK_CU(p_launch1d(ctx->fn_c2s_arange, (size_t)M, C2S_NTHREAD, a1, s));
            void* a2[] = { &startIdu, &Mu, &grid };
            PCHECK_CU(p_launch1d(ctx->fn_c2s_fill_u32, (size_t)M, C2S_NTHREAD, a2, s));
        }

        int Ncur = 1;
        for (int i = 0; i < nSS; i++) {
            Ncur *= SS[i];
            int S = SS[i];
            CUfunction probe_fn, fill_fn;
            switch (S) {
                case 1: probe_fn = ctx->fn_c2s_rlayer_p1; fill_fn = ctx->fn_c2s_rlayer_f1; break;
                case 2: probe_fn = ctx->fn_c2s_rlayer_p2; fill_fn = ctx->fn_c2s_rlayer_f2; break;
                case 3: probe_fn = ctx->fn_c2s_rlayer_p3; fill_fn = ctx->fn_c2s_rlayer_f3; break;
                case 4: probe_fn = ctx->fn_c2s_rlayer_p4; fill_fn = ctx->fn_c2s_rlayer_f4; break;
                case 5: probe_fn = ctx->fn_c2s_rlayer_p5; fill_fn = ctx->fn_c2s_rlayer_f5; break;
                case 6: probe_fn = ctx->fn_c2s_rlayer_p6; fill_fn = ctx->fn_c2s_rlayer_f6; break;
                case 7: probe_fn = ctx->fn_c2s_rlayer_p7; fill_fn = ctx->fn_c2s_rlayer_f7; break;
                case 8: probe_fn = ctx->fn_c2s_rlayer_p8; fill_fn = ctx->fn_c2s_rlayer_f8; break;
                default: rc = CUDA_ERROR_INVALID_VALUE; goto fail;
            }
            int rc2 = p_rasterize_layer(ctx, probe_fn, fill_fn, d_soup, S, &idx, &grid, &M, Ncur,
                                        band, out->totalSize, s);
            if (rc2) { rc = rc2; goto fail; }
        }

        if (j == 0) {
            float big = 1e9f;
            unsigned int total = (unsigned int)((long long)R * R * R);
            void* a3[] = { &big, &total, &out->gridDist };
            PCHECK_CU(p_launch1d(ctx->fn_c2s_fill_f32, (size_t)total, C2S_NTHREAD, a3, s));
            unsigned int collide_total = total * 3;
            void* a4[] = { &collide_total, &out->gridCollide };
            PCHECK_CU(p_launch1d(ctx->fn_c2s_fill_collide, (size_t)collide_total, C2S_NTHREAD,
                                 a4, s));
        }
        {
            int Ru = R;
            void* a5[] = { &d_soup, &idx, &grid, &M, &Ru, &out->gridDist, &out->gridCollide };
            PCHECK_CU(p_launch1d(ctx->fn_c2s_rasterize_reduce, (size_t)M, C2S_NTHREAD, a5, s));
        }

        PCHECK_CU(cuMemFree(idx)); idx = 0;
        PCHECK_CU(cuMemFree(grid)); grid = 0;
    }
    PCHECK_CU(cuStreamSynchronize(s));
    return 0;

fail:
    if (idx) cuMemFree(idx);
    if (grid) cuMemFree(grid);
    if (out->gridDist) cuMemFree(out->gridDist);
    if (out->gridCollide) cuMemFree(out->gridCollide);
    out->gridDist = 0; out->gridCollide = 0;
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}

static int p_fill_signs(gpu_ctx_t ctx, p_rast_t* rast, CUstream s) {
    int N = rast->R;
    int rc = 0;
    CUdeviceptr d_parents = 0;

    // nodeCount = N^3 + 1 (+ 1024 slack, upstream)
    long long nodeCount = (long long)N * N * N + 1;
    PCHECK_CU(cuMemAlloc(&d_parents, (size_t)(nodeCount + 1024) * sizeof(unsigned int)));
    CUdeviceptr parents = d_parents + 1023 * (CUdeviceptr)sizeof(unsigned int);

    // shfBitmask is unused by the identity shuffler but kept for parity.
    int shfBitmask = 0;

    // prescan: grid (cdiv(N,32), cdiv(N,8), 1), block (32, 8, 1)
    {
        unsigned int gx = (unsigned int)((N + 31) / 32);
        unsigned int gy = (unsigned int)((N + 7) / 8);
        unsigned int Nu = (unsigned int)N;
        C2S_RasterizeResult st_rast;
        st_rast.gridDist = (float*)rast->gridDist;
        st_rast.gridCollide = (unsigned char*)rast->gridCollide;
        void* args[] = { &st_rast, &parents, &Nu, &shfBitmask };
        PCHECK_CU(cuLaunchKernel(ctx->fn_c2s_volume_sign_prescan, gx, gy, 1, 32, 8, 1, 0, s,
                                 args, NULL));
    }
    // cts: grid (cdiv(N,32), cdiv(N,16), N), block (32, 16, 1)
    {
        unsigned int gx = (unsigned int)((N + 31) / 32);
        unsigned int gy = (unsigned int)((N + 15) / 16);
        unsigned int gz = (unsigned int)N;
        unsigned int Nu = (unsigned int)N;
        C2S_RasterizeResult st_rast;
        st_rast.gridDist = (float*)rast->gridDist;
        st_rast.gridCollide = (unsigned char*)rast->gridCollide;
        void* args[] = { &st_rast, &parents, &Nu, &shfBitmask };
        PCHECK_CU(cuLaunchKernel(ctx->fn_c2s_volume_cts, gx, gy, gz, 32, 16, 1, 0, s, args,
                                 NULL));
    }
    // apply: grid (cdiv(N,32), cdiv(N,16), N), block (32, 16, 1)
    {
        unsigned int gx = (unsigned int)((N + 31) / 32);
        unsigned int gy = (unsigned int)((N + 15) / 16);
        unsigned int gz = (unsigned int)N;
        C2S_RasterizeResult st_rast;
        st_rast.gridDist = (float*)rast->gridDist;
        st_rast.gridCollide = (unsigned char*)rast->gridCollide;
        void* args[] = { &st_rast, &parents, &N, &shfBitmask };
        PCHECK_CU(cuLaunchKernel(ctx->fn_c2s_volume_apply_sign, gx, gy, gz, 32, 16, 1, 0, s,
                                 args, NULL));
    }

    PCHECK_CU(cuStreamSynchronize(s));
    cuMemFree(d_parents);
    return 0;

fail:
    if (d_parents) cuMemFree(d_parents);
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// PDMC forward (driver-API port of CUDualMC::forward, cub scans → host scans)
// ---------------------------------------------------------------------------

typedef struct {
    int i, j, k, l;
} p_quad_t;
typedef struct {
    int i, j, k;
} p_tri_t;
typedef struct {
    float x, y, z;
} p_vert_t;
typedef struct {
    p_vert_t i, j;
} p_edge_t;

// Must mirror the device-side CUDualMC<float,int> layout (cuda/pdmc_kernels.cu)
// field-for-field so it can be passed by value to the kernels.
typedef struct {
    int dims[3];
    int n_cells;
    int n_used_cells;
    int n_verts;
    int n_quads;
    int n_tris;
    size_t allocated_temp_storage_size;
    void* temp_storage;
    size_t allocated_cell_count;
    int* first_cell_used;
    size_t allocated_used_cell_count;
    int* used_cell_index;
    int* used_to_first_mc_vert;
    unsigned char* used_cell_code;
    int* used_to_first_mc_patch;
    size_t allocated_quad_count;
    size_t allocated_tris_count;
    int* mc_vert_to_cell;
    p_edge_t* mc_vert_to_edge;
    unsigned char* mc_vert_type;
    p_quad_t* quads;
    p_tri_t* tris;
    int* first_tris_create;
    int* first_verts_create;
    size_t allocated_vert_count;
    p_vert_t* verts;
} p_pdmc_state_t;

static int pdmc_launch(gpu_ctx_t ctx, CUfunction fn, long long total, p_pdmc_state_t* st,
                       CUdeviceptr data, float iso, CUstream s) {
    if (total <= 0) return 0;
    unsigned int blocks = (unsigned int)((total + PDMC_BLOCK - 1) / PDMC_BLOCK);
    if (blocks > 2147483647u) return (int)CUDA_ERROR_INVALID_VALUE;
    void* args_data[] = { st, &data, &iso };
    void* args_plain[] = { st };
    void** args = data ? args_data : args_plain;
    CUresult r = cuLaunchKernel(fn, blocks, 1, 1, PDMC_BLOCK, 1, 1, 0, s, args, NULL);
    return (int)r;
}

// Device-side exclusive prefix sum into out[n+1]; returns host copy of the
// total (out[n]) or negative CUresult on failure. Uses host staging to keep
// cub/thrust out of the build (see file header).
static int pdmc_scan(gpu_ctx_t ctx, CUdeviceptr d_in, int n, long long total_hint,
                     CUdeviceptr* d_out, long long* out_total, CUstream s) {
    int rc = 0;
    CUdeviceptr dev = 0;
    int* host = NULL;
    long long* prefix = NULL;

    if (n <= 0) { *out_total = 0; if (d_out) *d_out = 0; return 0; }

    host = (int*)malloc((size_t)n * sizeof(int));
    prefix = (long long*)malloc((size_t)(n + 1) * sizeof(long long));
    if (!host || !prefix) { rc = (int)CUDA_ERROR_OUT_OF_MEMORY; goto fail; }
    PCHECK_CU(cuMemcpyDtoH(host, d_in, (size_t)n * sizeof(int)));
    p_exclusive_sum(host, n, prefix);
    // counts fit int for all sane grids; keep int32 device side
    for (int i = 0; i <= n; i++) {
        if (prefix[i] > 2147483647ll) { rc = (int)CUDA_ERROR_OUT_OF_MEMORY; goto fail; }
    }
    PCHECK_CU(cuMemAlloc(&dev, (size_t)(n + 1) * sizeof(int)));
    {
        int* narrow = (int*)malloc((size_t)(n + 1) * sizeof(int));
        if (!narrow) { rc = (int)CUDA_ERROR_OUT_OF_MEMORY; goto fail; }
        for (int i = 0; i <= n; i++) narrow[i] = (int)prefix[i];
        PCHECK_CU(cuMemcpyHtoD(dev, narrow, (size_t)(n + 1) * sizeof(int)));
        free(narrow);
    }
    (void)total_hint;
    *out_total = prefix[n];
    if (d_out) *d_out = dev; else { rc = (int)CUDA_ERROR_INVALID_VALUE; goto fail; }
    free(host); free(prefix);
    return 0;

fail:
    if (dev) cuMemFree(dev);
    free(host); free(prefix);
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}

typedef struct {
    CUdeviceptr first_cell_used;
    CUdeviceptr used_cell_index;
    CUdeviceptr used_to_first_mc_vert;
    CUdeviceptr used_cell_code;
    CUdeviceptr used_to_first_mc_patch;
    CUdeviceptr mc_vert_to_cell;
    CUdeviceptr mc_vert_to_edge;
    CUdeviceptr mc_vert_type;
    CUdeviceptr quads;
    CUdeviceptr tris;
    CUdeviceptr first_tris_create;
    CUdeviceptr first_verts_create;
    CUdeviceptr verts;
} pdmc_bufs_t;

static void pdmc_free_bufs(pdmc_bufs_t* b) {
    if (b->first_cell_used) cuMemFree(b->first_cell_used);
    if (b->used_cell_index) cuMemFree(b->used_cell_index);
    if (b->used_to_first_mc_vert) cuMemFree(b->used_to_first_mc_vert);
    if (b->used_cell_code) cuMemFree(b->used_cell_code);
    if (b->used_to_first_mc_patch) cuMemFree(b->used_to_first_mc_patch);
    if (b->mc_vert_to_cell) cuMemFree(b->mc_vert_to_cell);
    if (b->mc_vert_to_edge) cuMemFree(b->mc_vert_to_edge);
    if (b->mc_vert_type) cuMemFree(b->mc_vert_type);
    if (b->quads) cuMemFree(b->quads);
    if (b->tris) cuMemFree(b->tris);
    if (b->first_tris_create) cuMemFree(b->first_tris_create);
    if (b->first_verts_create) cuMemFree(b->first_verts_create);
    if (b->verts) cuMemFree(b->verts);
    memset(b, 0, sizeof(*b));
}

static int p_pdmc_forward(gpu_ctx_t ctx, CUdeviceptr d_data, int dimX, int dimY, int dimZ,
                          float iso, pdmc_bufs_t* out_bufs, int* out_nv, int* out_nt,
                          CUstream s) {
    int rc = 0;
    pdmc_bufs_t bufs;
    memset(&bufs, 0, sizeof(bufs));
    p_pdmc_state_t st;
    memset(&st, 0, sizeof(st));
    long long n_cells = (long long)dimX * dimY * dimZ;
    long long scan_total = 0;

    st.dims[0] = dimX; st.dims[1] = dimY; st.dims[2] = dimZ;
    st.n_cells = (int)n_cells;

    // --- find and index used cells -------------------------------------
    PCHECK_CU(cuMemAlloc(&bufs.first_cell_used, (size_t)(n_cells + 1) * sizeof(int)));
    PCHECK_CU(cuMemsetD32Async(bufs.first_cell_used + (CUdeviceptr)(n_cells * sizeof(int)), 0, 1, s));
    st.first_cell_used = (int*)bufs.first_cell_used;
    {
        p_pdmc_state_t sth = st;
        void* args[] = { &sth, &d_data, &iso };
        unsigned int blocks = (unsigned int)((n_cells + PDMC_BLOCK - 1) / PDMC_BLOCK);
        PCHECK_CU(cuLaunchKernel(ctx->fn_pdmc_count_used_cells, blocks, 1, 1, PDMC_BLOCK, 1, 1,
                                 0, s, args, NULL));
    }
    {
        CUdeviceptr scanned = 0;
        int r2 = pdmc_scan(ctx, bufs.first_cell_used, (int)n_cells, n_cells, &scanned, &scan_total, s);
        if (r2) { rc = r2; goto fail; }
        cuMemFree(bufs.first_cell_used);
        bufs.first_cell_used = scanned;
        st.first_cell_used = (int*)scanned;
    }
    st.n_used_cells = (int)scan_total;
    long long n_used = scan_total;

    if (n_used > 0) {
        PCHECK_CU(cuMemAlloc(&bufs.used_to_first_mc_vert, (size_t)(n_used + 1) * sizeof(int)));
        PCHECK_CU(cuMemAlloc(&bufs.used_to_first_mc_patch, (size_t)(n_used + 1) * sizeof(int)));
        PCHECK_CU(cuMemAlloc(&bufs.used_cell_code, (size_t)n_used));
        PCHECK_CU(cuMemAlloc(&bufs.used_cell_index, (size_t)n_used * sizeof(int)));
        PCHECK_CU(cuMemsetD32Async(bufs.used_to_first_mc_vert + (CUdeviceptr)(n_used * sizeof(int)), 0, 1, s));
        PCHECK_CU(cuMemsetD32Async(bufs.used_to_first_mc_patch + (CUdeviceptr)(n_used * sizeof(int)), 0, 1, s));
    }
    st.used_to_first_mc_vert = (int*)bufs.used_to_first_mc_vert;
    st.used_to_first_mc_patch = (int*)bufs.used_to_first_mc_patch;
    st.used_cell_code = (unsigned char*)bufs.used_cell_code;
    st.used_cell_index = (int*)bufs.used_cell_index;
    {
        p_pdmc_state_t sth = st;
        void* args[] = { &sth };
        unsigned int blocks = (unsigned int)((n_cells + PDMC_BLOCK - 1) / PDMC_BLOCK);
        PCHECK_CU(cuLaunchKernel(ctx->fn_pdmc_index_used_cells, blocks, 1, 1, PDMC_BLOCK, 1, 1, 0,
                                 s, args, NULL));
    }

    // --- count mc verts → quads ----------------------------------------
    if (n_used > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_count_cell_mc_verts, n_used, &st, d_data, iso, s);
        if (r2) { rc = r2; goto fail; }
        CUdeviceptr scanned = 0;
        int r3 = pdmc_scan(ctx, bufs.used_to_first_mc_vert, (int)n_used, n_used, &scanned,
                           &scan_total, s);
        if (r3) { rc = r3; goto fail; }
        cuMemFree(bufs.used_to_first_mc_vert);
        bufs.used_to_first_mc_vert = scanned;
        st.used_to_first_mc_vert = (int*)scanned;
    }
    st.n_quads = (int)scan_total;
    long long n_quads = scan_total;

    if (n_quads > 0) {
        PCHECK_CU(cuMemAlloc(&bufs.mc_vert_to_cell, (size_t)n_quads * sizeof(int)));
        PCHECK_CU(cuMemAlloc(&bufs.mc_vert_type, (size_t)n_quads));
        PCHECK_CU(cuMemAlloc(&bufs.quads, (size_t)n_quads * sizeof(p_quad_t)));
        PCHECK_CU(cuMemAlloc(&bufs.mc_vert_to_edge, (size_t)n_quads * sizeof(p_edge_t)));
        PCHECK_CU(cuMemAlloc(&bufs.first_tris_create, (size_t)(n_quads + 1) * sizeof(int)));
        PCHECK_CU(cuMemAlloc(&bufs.first_verts_create, (size_t)(n_quads + 1) * sizeof(int)));
    }
    st.mc_vert_to_cell = (int*)bufs.mc_vert_to_cell;
    st.mc_vert_type = (unsigned char*)bufs.mc_vert_type;
    st.quads = (p_quad_t*)bufs.quads;
    st.mc_vert_to_edge = (p_edge_t*)bufs.mc_vert_to_edge;
    st.first_tris_create = (int*)bufs.first_tris_create;
    st.first_verts_create = (int*)bufs.first_verts_create;

    if (n_used > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_index_cell_mc_verts, n_used, &st, d_data, iso, s);
        if (r2) { rc = r2; goto fail; }
    }

    // --- count patches → dmc verts -------------------------------------
    if (n_used > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_count_cell_patches, n_used, &st, d_data, iso, s);
        if (r2) { rc = r2; goto fail; }
        CUdeviceptr scanned = 0;
        int r3 = pdmc_scan(ctx, bufs.used_to_first_mc_patch, (int)n_used, n_used, &scanned,
                           &scan_total, s);
        if (r3) { rc = r3; goto fail; }
        cuMemFree(bufs.used_to_first_mc_patch);
        bufs.used_to_first_mc_patch = scanned;
        st.used_to_first_mc_patch = (int*)scanned;
    }
    st.n_verts = (int)scan_total;
    long long n_verts = scan_total;
    long long n_verts_alloc = n_verts + n_quads;

    if (n_verts_alloc > 0) {
        PCHECK_CU(cuMemAlloc(&bufs.verts, (size_t)n_verts_alloc * sizeof(p_vert_t)));
    }
    st.verts = (p_vert_t*)bufs.verts;

    if (n_used > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_create_dmc_verts, n_used, &st, d_data, iso, s);
        if (r2) { rc = r2; goto fail; }
    }
    if (n_quads > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_create_quads, n_quads, &st, 0, 0, s);
        if (r2) { rc = r2; goto fail; }
        int r3 = pdmc_launch(ctx, ctx->fn_pdmc_count_div_quads, n_quads, &st, 0, 0, s);
        if (r3) { rc = r3; goto fail; }
    }

    // --- quad division ---------------------------------------------------
    if (n_quads > 0) {
        PCHECK_CU(cuMemsetD32Async(bufs.first_tris_create + (CUdeviceptr)(n_quads * sizeof(int)), 0, 1, s));
        CUdeviceptr scanned = 0;
        int r3 = pdmc_scan(ctx, bufs.first_tris_create, (int)n_quads, n_quads, &scanned,
                           &scan_total, s);
        if (r3) { rc = r3; goto fail; }
        cuMemFree(bufs.first_tris_create);
        bufs.first_tris_create = scanned;
        st.first_tris_create = (int*)scanned;
    }
    st.n_tris = (int)scan_total;
    long long n_tris = scan_total;
    if (n_tris > 0) {
        PCHECK_CU(cuMemAlloc(&bufs.tris, (size_t)n_tris * sizeof(p_tri_t)));
    }
    st.tris = (p_tri_t*)bufs.tris;

    if (n_quads > 0) {
        PCHECK_CU(cuMemsetD32Async(bufs.first_verts_create + (CUdeviceptr)(n_quads * sizeof(int)), 0, 1, s));
        CUdeviceptr scanned = 0;
        int r3 = pdmc_scan(ctx, bufs.first_verts_create, (int)n_quads, n_quads, &scanned,
                           &scan_total, s);
        if (r3) { rc = r3; goto fail; }
        cuMemFree(bufs.first_verts_create);
        bufs.first_verts_create = scanned;
        st.first_verts_create = (int*)scanned;
    }
    long long n_verts_added = scan_total;

    if (n_quads > 0) {
        int r2 = pdmc_launch(ctx, ctx->fn_pdmc_divide_quads, n_quads, &st, d_data, iso, s);
        if (r2) { rc = r2; goto fail; }
    }
    st.n_verts = (int)(n_verts + n_verts_added);

    PCHECK_CU(cuStreamSynchronize(s));

    *out_bufs = bufs;
    *out_nv = st.n_verts;
    *out_nt = st.n_tris;
    memset(&bufs, 0, sizeof(bufs));  // ownership transferred
    return 0;

fail:
    pdmc_free_bufs(&bufs);
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// gpu_preprocess
// ---------------------------------------------------------------------------

static int p_is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

int gpu_preprocess(
    gpu_ctx_t ctx,
    const float* verts, int nv,
    const int*   tris,  int nt,
    int resolution,
    float** out_verts, int* out_nv,
    int**   out_tris,  int* out_nt)
{
    if (!ctx || !verts || !tris || nv <= 0 || nt <= 0) return (int)CUDA_ERROR_INVALID_VALUE;
    if (!p_is_pow2(resolution) || resolution < 8 || resolution > 1024)
        return (int)CUDA_ERROR_INVALID_VALUE;

    int rc = 0;
    CUstream s = NULL;
    CUdeviceptr d_verts = 0, d_tris = 0, d_soup = 0;
    float* h_verts = NULL;
    int* h_tris = NULL;
    p_rast_t rast;
    memset(&rast, 0, sizeof(rast));
    pdmc_bufs_t pb;
    memset(&pb, 0, sizeof(pb));
    int nv_out = 0, nt_out = 0;
    float* verts_out = NULL;
    int* tris_out = NULL;

    const int R = resolution;
    const float band = 3.0f / R;
    const float margin = 1.0f + 2.0f * band;

    // bbox (scalar extent, aspect-preserving — pamo convention)
    float vmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, vmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (long long i = 0; i < (long long)nv; i++) {
        for (int d = 0; d < 3; d++) {
            float p = verts[i * 3 + d];
            if (p < vmin[d]) vmin[d] = p;
            if (p > vmax[d]) vmax[d] = p;
        }
    }
    float extent = -1.0f;
    for (int d = 0; d < 3; d++)
        if (vmax[d] - vmin[d] > extent) extent = vmax[d] - vmin[d];
    if (!(extent > 0.0f)) return (int)CUDA_ERROR_INVALID_VALUE;

    // upload + build soup
    PCHECK_CU(cuMemAlloc(&d_verts, (size_t)nv * 3 * sizeof(float)));
    PCHECK_CU(cuMemAlloc(&d_tris, (size_t)nt * 3 * sizeof(int)));
    PCHECK_CU(cuMemcpyHtoDAsync(d_verts, verts, (size_t)nv * 3 * sizeof(float), s));
    PCHECK_CU(cuMemcpyHtoDAsync(d_tris, tris, (size_t)nt * 3 * sizeof(int), s));
    PCHECK_CU(cuMemAlloc(&d_soup, (size_t)nt * 9 * sizeof(float)));
    {
        int ntu = nt;
        float vminx = vmin[0], vminy = vmin[1], vminz = vmin[2];
        void* args[] = { &d_verts, &d_tris, &ntu, &vminx, &vminy, &vminz,
                         &extent, &band, &margin, &d_soup };
        PCHECK_CU(p_launch1d(ctx->fn_c2s_build_trisoup, (size_t)nt, C2S_NTHREAD, args, s));
    }

    // rasterize + sign → signed SDF grid
    int r2 = p_rasterize_kernel(ctx, d_soup, nt, R, band, &rast, s);
    if (r2) { rc = r2; goto fail; }
    r2 = p_fill_signs(ctx, &rast, s);
    if (r2) { rc = r2; goto fail; }

    // pamo SDF shift
    {
        unsigned int total = (unsigned int)((long long)R * R * R);
        float shift = 0.9f / R;
        void* args[] = { &total, &rast.gridDist, &shift };
        PCHECK_CU(p_launch1d(ctx->fn_c2s_sdf_shift, (size_t)total, C2S_NTHREAD, args, s));
    }

    // DualMC
    r2 = p_pdmc_forward(ctx, rast.gridDist, R, R, R, 0.0f, &pb, &nv_out, &nt_out, s);
    if (r2) { rc = r2; goto fail; }

    // download + invert normalization:
    //   p_norm = (g + 0.5) / R      (voxel g is a sample at (g+0.5)/R)
    //   p3 = (p_norm * margin - band) * extent + vmin
    verts_out = (float*)malloc((size_t)(nv_out > 0 ? nv_out : 1) * 3 * sizeof(float));
    tris_out = (int*)malloc((size_t)(nt_out > 0 ? nt_out : 1) * 3 * sizeof(int));
    if (!verts_out || !tris_out) { rc = (int)CUDA_ERROR_OUT_OF_MEMORY; goto fail; }

    if (nv_out > 0) {
        h_verts = (float*)malloc((size_t)nv_out * 3 * sizeof(float));
        if (!h_verts) { rc = (int)CUDA_ERROR_OUT_OF_MEMORY; goto fail; }
        PCHECK_CU(cuMemcpyDtoH(h_verts, pb.verts, (size_t)nv_out * 3 * sizeof(float)));
        const float kn = margin * extent / (float)R;
        for (long long i = 0; i < (long long)nv_out; i++) {
            for (int d = 0; d < 3; d++) {
                verts_out[i * 3 + d] =
                    (h_verts[i * 3 + d] + 0.5f) * kn - band * extent + vmin[d];
            }
        }
    }
    if (nt_out > 0) {
        PCHECK_CU(cuMemcpyDtoH(tris_out, pb.tris, (size_t)nt_out * 3 * sizeof(int)));
    }

    // cleanup
    cuMemFree(d_verts); cuMemFree(d_tris); cuMemFree(d_soup);
    cuMemFree(rast.gridDist); cuMemFree(rast.gridCollide);
    pdmc_free_bufs(&pb);
    free(h_verts);

    *out_verts = verts_out;
    *out_tris = tris_out;
    *out_nv = nv_out;
    *out_nt = nt_out;
    return 0;

fail:
    if (d_verts) cuMemFree(d_verts);
    if (d_tris) cuMemFree(d_tris);
    if (d_soup) cuMemFree(d_soup);
    if (rast.gridDist) cuMemFree(rast.gridDist);
    if (rast.gridCollide) cuMemFree(rast.gridCollide);
    pdmc_free_bufs(&pb);
    free(h_verts);
    free(verts_out);
    free(tris_out);
    return rc ? rc : (int)CUDA_ERROR_UNKNOWN;
}
