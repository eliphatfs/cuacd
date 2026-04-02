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
// Internal: allocate and init one DeviceHeap on device, sharing d_pool_struct.
// d_heap_out and d_compact_out are outputs (device ptrs).
// ---------------------------------------------------------------------------

static int _init_heap(beam_ctx_t ctx, CUstream s,
                      CUdeviceptr d_pool_struct,
                      CUdeviceptr* d_heap_out,
                      CUdeviceptr* d_compact_out)
{
    CUresult r;

    if ((r = cuMemAlloc(d_heap_out,    sizeof(struct DeviceHeap)))    != CUDA_SUCCESS) goto fail;
    if ((r = cuMemAlloc(d_compact_out, HEAP_COMPACT_BUF_BYTES))       != CUDA_SUCCESS) goto fail;

    // Zero-fill DeviceHeap (clears all arenas: empty free lists + unlocked)
    if ((r = cuMemsetD8(*d_heap_out, 0, sizeof(struct DeviceHeap)))   != CUDA_SUCCESS) goto fail;

    // Build DeviceHeap on host and patch in pool + compact_buf pointers
    struct DeviceHeap dh;
    memset(&dh, 0, sizeof(dh));
    dh.pool        = (struct DevicePool*)(uintptr_t)d_pool_struct;
    dh.compact_buf = (unsigned long long*)(uintptr_t)*d_compact_out;
    if ((r = cuMemcpyHtoDAsync(*d_heap_out, &dh,
                               sizeof(struct DeviceHeap), s))         != CUDA_SUCCESS) goto fail;

    return 0;

fail: {
    const char* _msg = NULL;
    cuGetErrorString(r, &_msg);
    snprintf(ctx->last_error, sizeof(ctx->last_error),
             "_init_heap failed at beam.c: %s", _msg ? _msg : "unknown");
    if (*d_heap_out)    { cuMemFree(*d_heap_out);    *d_heap_out    = 0; }
    if (*d_compact_out) { cuMemFree(*d_compact_out); *d_compact_out = 0; }
    return (int)r;
    }
}

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
    cuModuleGetFunction(&ctx->fn_test_warp_sort,      ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_hull_dandc,           ctx->module, "hull_dandc_kernel");
    cuModuleGetFunction(&ctx->fn_query_dandc_scratch,  ctx->module, "query_dandc_scratch");
    cuModuleGetFunction(&ctx->fn_mesh_volume,          ctx->module, "mesh_volume_kernel");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,    ctx->module, "batch_mesh_volume_kernel");
    cuModuleGetFunction(&ctx->fn_plane_cut,            ctx->module, "plane_cut_kernel");
    cuModuleGetFunction(&ctx->fn_heap_compact,         ctx->module, "heap_compact_kernel");

    // Determine pool size: default to 70% of free device memory
    if (pool_bytes == 0) {
        size_t free_bytes = 0, total_bytes = 0;
        cuMemGetInfo(&free_bytes, &total_bytes);
        pool_bytes = (size_t)(free_bytes * 0.70);
        if (pool_bytes < 64 * 1024 * 1024)
            pool_bytes = 64 * 1024 * 1024;  // minimum 64 MB
    }
    DBG("[beam] pool_bytes = %zu MB\n", pool_bytes >> 20);

    // Allocate shared pool backing: pool_mem + offset counter + DevicePool struct
    CUstream s = NULL;
    CHECK_CU(cuMemAlloc(&ctx->d_pool_mem,    pool_bytes));
    CHECK_CU(cuMemAlloc(&ctx->d_pool_off,    sizeof(unsigned long long)));
    CHECK_CU(cuMemAlloc(&ctx->d_pool_struct, sizeof(struct DevicePool)));

    // Zero pool offset
    unsigned long long zero = 0;
    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_pool_off, &zero, sizeof(unsigned long long), s));

    // Build and upload shared DevicePool
    struct DevicePool dp;
    dp.base     = (char*)(uintptr_t)ctx->d_pool_mem;
    dp.offset   = (unsigned long long*)(uintptr_t)ctx->d_pool_off;
    dp.capacity = (unsigned long long)pool_bytes;
    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_pool_struct, &dp, sizeof(struct DevicePool), s));

    // Init output heap
    int rc = _init_heap(ctx, s, ctx->d_pool_struct,
                        &ctx->d_heap, &ctx->d_heap_compact_buf);
    if (rc != 0) return rc;

    // Init scratch heap (shares same pool)
    rc = _init_heap(ctx, s, ctx->d_pool_struct,
                    &ctx->d_scratch, &ctx->d_scratch_compact_buf);
    if (rc != 0) return rc;

    CHECK_CU(cuStreamSynchronize(s));

    return 0;
}

void beam_destroy(beam_ctx_t ctx) {
    if (!ctx) return;

    // Free persistent heap resources
    if (ctx->d_heap)                cuMemFree(ctx->d_heap);
    if (ctx->d_heap_compact_buf)    cuMemFree(ctx->d_heap_compact_buf);
    if (ctx->d_scratch)             cuMemFree(ctx->d_scratch);
    if (ctx->d_scratch_compact_buf) cuMemFree(ctx->d_scratch_compact_buf);
    if (ctx->d_pool_struct)         cuMemFree(ctx->d_pool_struct);
    if (ctx->d_pool_off)            cuMemFree(ctx->d_pool_off);
    if (ctx->d_pool_mem)            cuMemFree(ctx->d_pool_mem);

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
// beam_heap_compact
// ---------------------------------------------------------------------------
// Compact both heaps (output and scratch) sequentially.
// Both share the same pool, so coalescing adjacent freed blocks is
// the primary way to recover fragmented memory for future allocations.

int beam_heap_compact(beam_ctx_t ctx) {
    if (!ctx || !ctx->fn_heap_compact) return -1;
    if (!ctx->d_heap || !ctx->d_scratch) return -1;

    CUstream s = NULL;

    void* args_out[]     = { &ctx->d_heap };
    void* args_scratch[] = { &ctx->d_scratch };

    CHECK_CU(cuLaunchKernel(ctx->fn_heap_compact,
        1, 1, 1, 64, 1, 1, 0, s, args_out, NULL));
    CHECK_CU(cuLaunchKernel(ctx->fn_heap_compact,
        1, 1, 1, 64, 1, 1, 0, s, args_scratch, NULL));
    CHECK_CU(cuStreamSynchronize(s));
    return 0;
}
