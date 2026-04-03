// Host-side implementation for GPU beam search convex decomposition.
// Uses CUDA driver API exclusively — links only against libcuda.so.

#include "beam.h"
#include "structs.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
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

int beam_init(beam_ctx_t* out, int device_ordinal, size_t pool_bytes) {
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
    cuModuleGetFunction(&ctx->fn_test_warp_sort,     ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_hull_dandc,          ctx->module, "hull_dandc_kernel");
    cuModuleGetFunction(&ctx->fn_mesh_volume,         ctx->module, "mesh_volume_kernel");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,   ctx->module, "batch_mesh_volume_kernel");
    cuModuleGetFunction(&ctx->fn_plane_cut,           ctx->module, "plane_cut_kernel");
    cuModuleGetFunction(&ctx->fn_heap_init,           ctx->module, "heap_init_kernel");

    // Determine pool size: default to 70% of free device memory
    if (pool_bytes == 0) {
        size_t free_bytes = 0, total_bytes = 0;
        cuMemGetInfo(&free_bytes, &total_bytes);
        pool_bytes = (size_t)(free_bytes * 0.70);
        if (pool_bytes < 64 * 1024 * 1024)
            pool_bytes = 64 * 1024 * 1024;  // minimum 64 MB
    }
    DBG("[beam] pool_bytes = %zu MB\n", pool_bytes >> 20);

    CUstream s = NULL;

    // Allocate pool backing memory and offset counter
    CHECK_CU(cuMemAlloc(&ctx->d_pool_mem, pool_bytes));
    CHECK_CU(cuMemAlloc(&ctx->d_pool_off, sizeof(unsigned long long)));

    // Allocate and zero-fill DevicePool struct (contains embedded heaps, ~133 KB)
    CHECK_CU(cuMemAlloc(&ctx->d_pool_struct, sizeof(struct DevicePool)));
    CHECK_CU(cuMemsetD8Async(ctx->d_pool_struct, 0, sizeof(struct DevicePool), s));

    // Build host-side DevicePool (only scalar fields; heaps zeroed by memset above)
    struct DevicePool dp;
    memset(&dp, 0, sizeof(dp));
    dp.base     = (char*)(uintptr_t)ctx->d_pool_mem;
    dp.offset   = (unsigned long long*)(uintptr_t)ctx->d_pool_off;
    dp.capacity = (unsigned long long)pool_bytes;

    // Upload base/offset/capacity (first 3 fields, before the embedded heaps)
    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_pool_struct, &dp,
                               offsetof(struct DevicePool, heap), s));

    // Zero the pool offset counter
    unsigned long long zero = 0;
    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_pool_off, &zero, sizeof(unsigned long long), s));

    // Launch heap_init_kernel: 128 blocks × 32 threads (1 warp per arena per heap).
    // Initialises both pool->heap and pool->scratch: sets pool back-pointers and
    // zeroes all arena state (bitmap, lock, heads, tails).
    {
        CUdeviceptr d_pool = ctx->d_pool_struct;
        void* init_args[] = { &d_pool };
        CHECK_CU(cuLaunchKernel(ctx->fn_heap_init,
            HEAP_NUM_ARENAS * 2, 1, 1,   // 2 × HEAP_NUM_ARENAS blocks
             32, 1, 1,                   // 32 threads per block (1 warp)
            0, s, init_args, NULL));
    }

    CHECK_CU(cuStreamSynchronize(s));

    return 0;
}

void beam_destroy(beam_ctx_t ctx) {
    if (!ctx) return;

    if (ctx->d_pool_struct) cuMemFree(ctx->d_pool_struct);
    if (ctx->d_pool_off)    cuMemFree(ctx->d_pool_off);
    if (ctx->d_pool_mem)    cuMemFree(ctx->d_pool_mem);

    if (ctx->module)                cuModuleUnload(ctx->module);
    if (ctx->owns_context && ctx->cuda_ctx) cuCtxDestroy(ctx->cuda_ctx);
    free(ctx);
}

const char* beam_last_error(beam_ctx_t ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return NULL;
    return ctx->last_error;
}

// ---------------------------------------------------------------------------
// beam_pool_usage
// ---------------------------------------------------------------------------

size_t beam_pool_usage(beam_ctx_t ctx) {
    if (!ctx || !ctx->d_pool_off) return 0;
    unsigned long long off = 0;
    if (cuMemcpyDtoH(&off, ctx->d_pool_off, sizeof(unsigned long long)) != CUDA_SUCCESS)
        return 0;
    return (size_t)off;
}

// ---------------------------------------------------------------------------
// beam_heap_compact — no-op (coalescing now happens in-place on free)
// ---------------------------------------------------------------------------
// The binned heap with boundary sentinels performs O(1) coalescing in heap_free,
// so no offline compaction pass is required. This function is retained for API
// compatibility; callers that previously relied on compact to recover fragmented
// memory will benefit automatically from the new allocator.

int beam_heap_compact(beam_ctx_t ctx) {
    (void)ctx;
    return 0;
}

// ---------------------------------------------------------------------------
// beam_decompose — full beam-search convex decomposition
// ---------------------------------------------------------------------------

// Host-side mirrors of CUDA device structs from cuda/structs.cuh.
// Pointer fields use CUdeviceptr (uint64) to match 64-bit device pointers.
// Padding matches the device struct layout automatically via C alignment rules.
#define WORK_ITEM_MAX_PARTS 512

struct Mesh_h {
    CUdeviceptr verts;     // float* on device
    CUdeviceptr tris;      // int*   on device
    int         nv;
    int         nt;
    CUdeviceptr refcount;  // int*   on device
};

struct Part_h {
    struct Mesh_h mesh;
    struct Mesh_h hull;
    float mesh_vol;
    float hull_vol;
    float hausdorff;
    // 4 bytes trailing padding added by compiler to reach alignment of 8
};

struct WorkItem_h {
    struct Part_h parts[WORK_ITEM_MAX_PARTS];
    int nparts;
    // 4 bytes trailing padding added by compiler
};

struct AlgoState_h {
    CUdeviceptr items;   // WorkItem* on device
    int         nitems;
    // 4 bytes trailing padding added by compiler
};

// Read back the best WorkItem from d_current into out.
// Caller must have synced the stream before calling.
static int decomp_read_result(
    beam_ctx_t ctx, CUdeviceptr d_current, struct beam_result* out, CUstream s)
{
    struct AlgoState_h h_state;
    CUresult r = cuMemcpyDtoH(&h_state, d_current, sizeof(h_state));
    if (r != CUDA_SUCCESS) {
        const char* msg = NULL;
        cuGetErrorString(r, &msg);
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "read AlgoState failed: %s", msg ? msg : "unknown");
        return (int)r;
    }

    if (h_state.nitems == 0) {
        out->nparts = 0;
        out->parts  = NULL;
        return 0;
    }

    struct WorkItem_h wi;
    r = cuMemcpyDtoH(&wi, h_state.items, sizeof(wi));
    if (r != CUDA_SUCCESS) {
        const char* msg = NULL;
        cuGetErrorString(r, &msg);
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "read WorkItem failed: %s", msg ? msg : "unknown");
        return (int)r;
    }

    int np = wi.nparts;
    out->nparts = np;
    if (np == 0) { out->parts = NULL; return 0; }

    out->parts = (struct beam_part_result*)calloc(np, sizeof(struct beam_part_result));
    if (!out->parts) return -1;

    // Allocate all host buffers and issue async D2H copies for every part.
    for (int i = 0; i < np; i++) {
        struct Part_h*           p  = &wi.parts[i];
        struct beam_part_result* pr = &out->parts[i];

        pr->nv        = p->mesh.nv;
        pr->nt        = p->mesh.nt;
        pr->mesh_vol  = p->mesh_vol;
        pr->hull_vol  = p->hull_vol;
        pr->hausdorff = p->hausdorff;

        if (pr->nv > 0 && p->mesh.verts) {
            pr->verts = (float*)malloc((size_t)pr->nv * 3 * sizeof(float));
            if (!pr->verts) return -1;
            r = cuMemcpyDtoHAsync(pr->verts, p->mesh.verts,
                                  (size_t)pr->nv * 3 * sizeof(float), s);
            if (r != CUDA_SUCCESS) return (int)r;
        }
        if (pr->nt > 0 && p->mesh.tris) {
            pr->tris = (int*)malloc((size_t)pr->nt * 3 * sizeof(int));
            if (!pr->tris) return -1;
            r = cuMemcpyDtoHAsync(pr->tris, p->mesh.tris,
                                  (size_t)pr->nt * 3 * sizeof(int), s);
            if (r != CUDA_SUCCESS) return (int)r;
        }
    }

    // Single sync after all async copies have been enqueued.
    r = cuStreamSynchronize(s);
    if (r != CUDA_SUCCESS) return (int)r;
    return 0;
}

int beam_decompose(
    beam_ctx_t   ctx,
    const float* verts,      int nv,
    const int*   tris,       int nt,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    int max_iters, int cuts_per_axis, float threshold, int max_keep,
    int verbose,
    struct beam_result* out)
{
    // Local CHECK that sets result_code and jumps to cleanup.
#define LCHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at beam.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        result_code = (int)_r; \
        goto cleanup; \
    } \
} while(0)

    CUstream s = NULL;
    int result_code = 0;
    LCHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT));

    // Timing helpers (active only when verbose != 0).
#if defined(_POSIX_C_SOURCE) || defined(__linux__)
#include <time.h>
    struct timespec _t0, _t1;
#define TSTAMP(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define TELAPSED_MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)*1e-6)
#else
#define TSTAMP(t)        ((void)0)
#define TELAPSED_MS(a,b) 0.0
#endif

    CUdeviceptr d_verts      = 0, d_tris       = 0;
    CUdeviceptr d_hverts     = 0, d_htris       = 0;
    CUdeviceptr d_wi_a       = 0, d_wi_b        = 0;
    CUdeviceptr d_state_a    = 0, d_state_b     = 0;
    CUdeviceptr d_err        = 0, d_finish      = 0;

    // Resolve beam kernels
    CUfunction fn_init = NULL, fn_expand = NULL, fn_hull = NULL,
               fn_sort = NULL, fn_finalize = NULL;
    LCHECK(cuModuleGetFunction(&fn_init,     ctx->module, "beam_initialize"));
    LCHECK(cuModuleGetFunction(&fn_expand,   ctx->module, "beam_expansion"));
    LCHECK(cuModuleGetFunction(&fn_hull,     ctx->module, "beam_hull"));
    LCHECK(cuModuleGetFunction(&fn_sort,     ctx->module, "beam_sort"));
    LCHECK(cuModuleGetFunction(&fn_finalize, ctx->module, "beam_finalize"));

    // ------------------------------------------------------------------
    // Allocate and upload input mesh + hull buffers
    // ------------------------------------------------------------------
    LCHECK(cuMemAllocAsync(&d_verts,  (size_t)nv     * 3 * sizeof(float), s));
    LCHECK(cuMemAllocAsync(&d_tris,   (size_t)nt     * 3 * sizeof(int),   s));
    LCHECK(cuMemAllocAsync(&d_hverts, (size_t)hull_nv * 3 * sizeof(float), s));
    LCHECK(cuMemAllocAsync(&d_htris,  (size_t)hull_nt * 3 * sizeof(int),   s));

    LCHECK(cuMemcpyHtoDAsync(d_verts,  verts,      (size_t)nv     * 3 * sizeof(float), s));
    LCHECK(cuMemcpyHtoDAsync(d_tris,   tris,       (size_t)nt     * 3 * sizeof(int),   s));
    LCHECK(cuMemcpyHtoDAsync(d_hverts, hull_verts, (size_t)hull_nv * 3 * sizeof(float), s));
    LCHECK(cuMemcpyHtoDAsync(d_htris,  hull_tris,  (size_t)hull_nt * 3 * sizeof(int),   s));

    // ------------------------------------------------------------------
    // Allocate double-buffered WorkItem arrays and AlgoState structs
    // max_expand: maximum WorkItems that can exist after one expansion round
    // ------------------------------------------------------------------
    int max_expand = 3 * cuts_per_axis * max_keep;
    size_t wi_pool_bytes = (size_t)max_expand * sizeof(struct WorkItem_h);

    LCHECK(cuMemAllocAsync(&d_wi_a, wi_pool_bytes, s));
    LCHECK(cuMemAllocAsync(&d_wi_b, wi_pool_bytes, s));
    LCHECK(cuMemsetD8Async(d_wi_a, 0, wi_pool_bytes, s));
    LCHECK(cuMemsetD8Async(d_wi_b, 0, wi_pool_bytes, s));

    LCHECK(cuMemAllocAsync(&d_state_a, sizeof(struct AlgoState_h), s));
    LCHECK(cuMemAllocAsync(&d_state_b, sizeof(struct AlgoState_h), s));

    struct AlgoState_h ha, hb;
    memset(&ha, 0, sizeof(ha)); ha.items = d_wi_a; ha.nitems = 0;
    memset(&hb, 0, sizeof(hb)); hb.items = d_wi_b; hb.nitems = 0;
    LCHECK(cuMemcpyHtoDAsync(d_state_a, &ha, sizeof(ha), s));
    LCHECK(cuMemcpyHtoDAsync(d_state_b, &hb, sizeof(hb), s));

    // ------------------------------------------------------------------
    // Allocate error / finish scalars
    // ------------------------------------------------------------------
    LCHECK(cuMemAllocAsync(&d_err,    sizeof(int), s));
    LCHECK(cuMemAllocAsync(&d_finish, sizeof(int), s));
    LCHECK(cuMemsetD32Async(d_err,    0, 1, s));
    LCHECK(cuMemsetD32Async(d_finish, 0, 1, s));

    // ------------------------------------------------------------------
    // Seed: beam_initialize <<<3, 32>>>
    // ------------------------------------------------------------------
    {
        void* args[] = { &d_verts, &d_tris, &nv, &nt,
                         &d_hverts, &d_htris, &hull_nv, &hull_nt,
                         &d_state_a };
        LCHECK(cuLaunchKernel(fn_init, 3, 1, 1, 32, 1, 1, 0, s, args, NULL));
    }

    // Double-buffer pointers: current holds the live beam, prev is empty scratch.
    CUdeviceptr d_current = d_state_a;
    CUdeviceptr d_prev    = d_state_b;
    int cur_nitems        = 1;  // beam_initialize always produces exactly 1 item

    // ------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------
    for (int iter = 0; iter < max_iters; iter++) {

        // ---- beam_finalize: clear prev, compact current to max_keep ----
        {
            void* args[] = { &d_prev, &d_current,
                             &ctx->d_pool_struct, &d_finish, &d_err,
                             &max_keep, &threshold };
            LCHECK(cuLaunchKernel(fn_finalize,
                                  max_keep, 1, 1, 1024, 1, 1,
                                  0, s, args, NULL));
        }
        // Batch three D2H transfers async, then sync once before reading any.
        int h_finish = 0, h_err = 0;
        struct AlgoState_h h_st;
        memset(&h_st, 0, sizeof(h_st));
        LCHECK(cuMemcpyDtoHAsync(&h_finish, d_finish,  sizeof(int),                s));
        LCHECK(cuMemcpyDtoHAsync(&h_err,    d_err,     sizeof(int),                s));
        LCHECK(cuMemcpyDtoHAsync(&h_st,     d_current, sizeof(struct AlgoState_h), s));
        TSTAMP(_t0);
        LCHECK(cuStreamSynchronize(s));
        TSTAMP(_t1);
        if (verbose)
            fprintf(stderr, "[beam] iter %d: sync wait %.3f ms  nitems=%d\n",
                    iter, TELAPSED_MS(_t0, _t1), h_st.nitems);

        if (h_err) {
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                     "beam_decompose: GPU error 0x%x at iter %d", h_err, iter);
            result_code = h_err;
            goto cleanup;
        }

        if (h_finish) {
            TSTAMP(_t0);
            result_code = decomp_read_result(ctx, d_current, out, s);
            TSTAMP(_t1);
            if (verbose)
                fprintf(stderr, "[beam] readback: %.3f ms\n", TELAPSED_MS(_t0, _t1));
            goto cleanup;
        }

        cur_nitems = h_st.nitems;

        // ---- beam_expansion: current → prev (now repurposed as next) ----
        // d_prev was cleared by beam_finalize; repurpose as "next".
        {
            void* args[] = { &d_current, &d_prev,
                             &ctx->d_pool_struct, &cuts_per_axis, &d_err };
            int nblocks = 3 * cuts_per_axis * cur_nitems;
            if (nblocks < 1) nblocks = 1;
            LCHECK(cuLaunchKernel(fn_expand, nblocks, 1, 1, 64, 1, 1,
                                  0, s, args, NULL));
        }

        // DEBUG: sync + check after expansion
        {
            CUresult _sr = cuStreamSynchronize(s);
            if (_sr != CUDA_SUCCESS) {
                const char* _msg = NULL; cuGetErrorString(_sr, &_msg);
                fprintf(stderr, "[beam] iter %d EXPAND CRASH: %s\n", iter, _msg ? _msg : "?");
                result_code = (int)_sr; goto cleanup;
            }
            int _he = 0;
            cuMemcpyDtoH(&_he, d_err, sizeof(int));
            if (_he) {
                fprintf(stderr, "[beam] iter %d EXPAND err=0x%x\n", iter, _he);
                result_code = _he; goto cleanup;
            }
            fprintf(stderr, "[beam] iter %d: expand OK\n", iter);
        }

        // Swap: next (d_prev) becomes current; old current becomes prev.
        CUdeviceptr tmp = d_current;
        d_current = d_prev;
        d_prev    = tmp;

        // ---- beam_hull: over-provisioned grid, self-checks nitems ----
        // Max possible new items: 3*cuts_per_axis*cur_nitems; 2 blocks per item.
        {
            void* args[] = { &d_current, &ctx->d_pool_struct, &d_err };
            int nblocks = 2 * 3 * cuts_per_axis * cur_nitems;
            if (nblocks < 1) nblocks = 1;
            LCHECK(cuLaunchKernel(fn_hull, nblocks, 1, 1, 32, 1, 1,
                                  0, s, args, NULL));
        }

        // DEBUG: sync + check after hull
        {
            CUresult _sr = cuStreamSynchronize(s);
            if (_sr != CUDA_SUCCESS) {
                const char* _msg = NULL; cuGetErrorString(_sr, &_msg);
                fprintf(stderr, "[beam] iter %d HULL CRASH: %s\n", iter, _msg ? _msg : "?");
                result_code = (int)_sr; goto cleanup;
            }
            int _he = 0;
            cuMemcpyDtoH(&_he, d_err, sizeof(int));
            if (_he) {
                fprintf(stderr, "[beam] iter %d HULL err=0x%x\n", iter, _he);
                result_code = _he; goto cleanup;
            }
            fprintf(stderr, "[beam] iter %d: hull OK\n", iter);
        }

        // ---- beam_sort: over-provisioned grid, self-checks nitems ----
        {
            void* args[] = { &d_current, &ctx->d_pool_struct, &d_err };
            int nblocks = 3 * cuts_per_axis * cur_nitems;
            if (nblocks < 1) nblocks = 1;
            LCHECK(cuLaunchKernel(fn_sort, nblocks, 1, 1, 32, 1, 1,
                                  0, s, args, NULL));
        }

        // DEBUG: sync + check after sort
        {
            CUresult _sr = cuStreamSynchronize(s);
            if (_sr != CUDA_SUCCESS) {
                const char* _msg = NULL; cuGetErrorString(_sr, &_msg);
                fprintf(stderr, "[beam] iter %d SORT CRASH: %s\n", iter, _msg ? _msg : "?");
                result_code = (int)_sr; goto cleanup;
            }
            int _he = 0;
            cuMemcpyDtoH(&_he, d_err, sizeof(int));
            if (_he) {
                fprintf(stderr, "[beam] iter %d SORT err=0x%x\n", iter, _he);
                result_code = _he; goto cleanup;
            }
            fprintf(stderr, "[beam] iter %d: sort OK\n", iter);
        }
    }

    // Iterations exhausted — read back current state.
    TSTAMP(_t0);
    LCHECK(cuStreamSynchronize(s));
    TSTAMP(_t1);
    if (verbose)
        fprintf(stderr, "[beam] exhausted: sync wait %.3f ms\n", TELAPSED_MS(_t0, _t1));
    TSTAMP(_t0);
    result_code = decomp_read_result(ctx, d_current, out, s);
    TSTAMP(_t1);
    if (verbose)
        fprintf(stderr, "[beam] readback: %.3f ms\n", TELAPSED_MS(_t0, _t1));

cleanup:
    // Ensure stream is idle before releasing buffers.
    cuStreamSynchronize(s);
    if (d_verts)   cuMemFreeAsync(d_verts,   s);
    if (d_tris)    cuMemFreeAsync(d_tris,    s);
    if (d_hverts)  cuMemFreeAsync(d_hverts,  s);
    if (d_htris)   cuMemFreeAsync(d_htris,   s);
    if (d_wi_a)    cuMemFreeAsync(d_wi_a,    s);
    if (d_wi_b)    cuMemFreeAsync(d_wi_b,    s);
    if (d_state_a) cuMemFreeAsync(d_state_a, s);
    if (d_state_b) cuMemFreeAsync(d_state_b, s);
    if (d_err)     cuMemFreeAsync(d_err,     s);
    if (d_finish)  cuMemFreeAsync(d_finish,  s);
    cuStreamSynchronize(s);
    cuStreamDestroy(s);

#undef LCHECK
    return result_code;
}

void beam_result_free(struct beam_result* result) {
    if (!result) return;
    for (int i = 0; i < result->nparts; i++) {
        free(result->parts[i].verts);
        free(result->parts[i].tris);
    }
    free(result->parts);
    result->parts  = NULL;
    result->nparts = 0;
}
