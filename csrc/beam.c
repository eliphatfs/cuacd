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
    cuModuleGetFunction(&ctx->fn_query_dandc_scratch, ctx->module, "query_dandc_scratch");
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
