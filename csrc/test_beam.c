// test_beam.c — Host launchers for test/diagnostic GPU kernels.
// Compiled into the same extension as beam.c; separated to keep beam.c focused
// on init/destroy.

#include "beam.h"
#include "structs.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at test_beam.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        return (int)_r; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// BeamHeapD — device-side DeviceHeap allocation helper
// ---------------------------------------------------------------------------

typedef struct {
    CUdeviceptr d_heap;       // struct DeviceHeap on device
    CUdeviceptr d_pool;       // struct DevicePool on device
    CUdeviceptr d_pool_mem;   // pool backing memory
    CUdeviceptr d_pool_off;   // unsigned long long offset counter
    CUdeviceptr d_compact;    // compact buffer (HEAP_COMPACT_BUF_BYTES)
} BeamHeapD;

// Allocate and zero-initialize a DeviceHeap with the given pool_size.
// Returns CUDA_SUCCESS (0) on success.
static int heap_alloc_d(beam_ctx_t ctx, BeamHeapD* bh, size_t pool_size, CUstream s)
{
    memset(bh, 0, sizeof(*bh));

    CUresult r;
    if ((r = cuMemAlloc(&bh->d_heap,     sizeof(struct DeviceHeap)))    != CUDA_SUCCESS) goto fail;
    if ((r = cuMemAlloc(&bh->d_pool,     sizeof(struct DevicePool)))    != CUDA_SUCCESS) goto fail;
    if ((r = cuMemAlloc(&bh->d_pool_mem, pool_size))                    != CUDA_SUCCESS) goto fail;
    if ((r = cuMemAlloc(&bh->d_pool_off, sizeof(unsigned long long)))   != CUDA_SUCCESS) goto fail;
    if ((r = cuMemAlloc(&bh->d_compact,  HEAP_COMPACT_BUF_BYTES))       != CUDA_SUCCESS) goto fail;

    // Zero-fill DeviceHeap (clears all arenas: empty free lists + unlocked)
    if ((r = cuMemsetD8(bh->d_heap, 0, sizeof(struct DeviceHeap)))      != CUDA_SUCCESS) goto fail;

    // Init pool offset to 0
    unsigned long long zero = 0;
    if ((r = cuMemcpyHtoDAsync(bh->d_pool_off, &zero,
                               sizeof(unsigned long long), s))           != CUDA_SUCCESS) goto fail;

    // Build DevicePool on host and copy to device
    struct DevicePool dp;
    dp.base     = (char*)(uintptr_t)bh->d_pool_mem;
    dp.offset   = (unsigned long long*)(uintptr_t)bh->d_pool_off;
    dp.capacity = (unsigned long long)pool_size;
    if ((r = cuMemcpyHtoDAsync(bh->d_pool, &dp,
                               sizeof(struct DevicePool), s))            != CUDA_SUCCESS) goto fail;

    // Build DeviceHeap on host (zero arenas already on device; patch pointers)
    struct DeviceHeap dh;
    memset(&dh, 0, sizeof(dh));
    dh.pool        = (struct DevicePool*)(uintptr_t)bh->d_pool;
    dh.compact_buf = (unsigned long long*)(uintptr_t)bh->d_compact;
    if ((r = cuMemcpyHtoDAsync(bh->d_heap, &dh,
                               sizeof(struct DeviceHeap), s))            != CUDA_SUCCESS) goto fail;

    return 0;

fail:
    if (bh->d_heap)     cuMemFree(bh->d_heap);
    if (bh->d_pool)     cuMemFree(bh->d_pool);
    if (bh->d_pool_mem) cuMemFree(bh->d_pool_mem);
    if (bh->d_pool_off) cuMemFree(bh->d_pool_off);
    if (bh->d_compact)  cuMemFree(bh->d_compact);
    memset(bh, 0, sizeof(*bh));
    const char* _msg = NULL;
    cuGetErrorString(r, &_msg);
    snprintf(ctx->last_error, sizeof(ctx->last_error),
             "heap_alloc_d failed at test_beam.c: %s", _msg ? _msg : "unknown");
    return (int)r;
}

static void heap_free_d(BeamHeapD* bh) {
    if (bh->d_heap)     cuMemFree(bh->d_heap);
    if (bh->d_pool)     cuMemFree(bh->d_pool);
    if (bh->d_pool_mem) cuMemFree(bh->d_pool_mem);
    if (bh->d_pool_off) cuMemFree(bh->d_pool_off);
    if (bh->d_compact)  cuMemFree(bh->d_compact);
    memset(bh, 0, sizeof(*bh));
}

// ---------------------------------------------------------------------------
// beam_test_warp_sort
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
// beam_hull_dandc
// ---------------------------------------------------------------------------
// Extract convex hull mesh for each point cloud.
// out_verts:  float[n_hulls * max_hull_verts * 3]
// out_tris:   int[n_hulls * max_hull_tris * 3]
// out_nv, out_nt, out_errors: int[n_hulls]

int beam_hull_dandc(
    beam_ctx_t   ctx,
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
    CUstream s = NULL;

    // Query scratch bytes per hull (WarpPool portion)
    int warp_scratch = 0;
    {
        CUdeviceptr d_q;
        CHECK_CU(cuMemAlloc(&d_q, sizeof(int)));
        int mpi = max_pts_per_hull;
        void* qargs[] = { &mpi, &d_q };
        CHECK_CU(cuLaunchKernel(ctx->fn_query_dandc_scratch, 1, 1, 1,
                                1, 1, 1, 0, s, qargs, NULL));
        CHECK_CU(cuMemcpyDtoHAsync(&warp_scratch, d_q, sizeof(int), s));
        CHECK_CU(cuStreamSynchronize(s));
        cuMemFree(d_q);
    }

    // Heap sizes: generous to handle concurrent warps
    // Output heap: per hull ≈ (max_hull_verts + max_hull_tris) * 12 B + 8 KB overhead, 4 KB aligned
    size_t per_hull_out = (size_t)(max_hull_verts * 3 * (int)sizeof(float)
                                 + max_hull_tris  * 3 * (int)sizeof(int)) + 8192;
    size_t heap_pool_size = (size_t)n_hulls * per_hull_out;
    if (heap_pool_size < 16 * 1024 * 1024) heap_pool_size = 16 * 1024 * 1024;

    // Scratch heap: warp scratch + generous extra for edge pool slabs (≈512KB per hull)
    size_t per_hull_scratch = (size_t)warp_scratch + 512 * 1024;
    size_t scratch_pool_size = (size_t)n_hulls * per_hull_scratch;
    if (scratch_pool_size < 64 * 1024 * 1024) scratch_pool_size = 64 * 1024 * 1024;

    BeamHeapD heap, scratch;
    int rc;
    if ((rc = heap_alloc_d(ctx, &heap,    heap_pool_size,    s)) != 0) return rc;
    if ((rc = heap_alloc_d(ctx, &scratch, scratch_pool_size, s)) != 0) {
        heap_free_d(&heap); return rc;
    }
    CHECK_CU(cuStreamSynchronize(s));

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
        &heap.d_heap, &scratch.d_heap
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
    heap_free_d(&heap);
    heap_free_d(&scratch);
    return 0;
}

// ---------------------------------------------------------------------------
// beam_test_mesh_volume
// ---------------------------------------------------------------------------
// Compute volume of a single mesh via divergence theorem (GPU warp).

int beam_test_mesh_volume(
    beam_ctx_t   ctx,
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
// beam_test_plane_cut
// ---------------------------------------------------------------------------
// New API: separate pos and neg vertex arrays.

int beam_test_plane_cut(
    beam_ctx_t   ctx,
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

    *out_n_pv = 0; *out_n_pt = 0;
    *out_n_nv = 0; *out_n_nt = 0;

    CUstream s = NULL;
    CUdeviceptr d_verts = 0, d_tris = 0;
    CUdeviceptr d_opv = 0, d_opt = 0, d_onv = 0, d_ont = 0;
    CUdeviceptr d_counts = 0, d_kerr = 0;

    // Heap sizes: generous
    // Output: n_verts * 2 * 12 bytes + overhead; n_tris * 3 * 12 * 2 + overhead
    size_t out_heap_size = (size_t)(n_verts * 2 + n_tris * 4) * 16 + 2 * 1024 * 1024;
    size_t scratch_size  = (size_t)(n_tris * 8) * 16 + 32 * 1024 * 1024;
    if (out_heap_size < 4  * 1024 * 1024) out_heap_size = 4  * 1024 * 1024;
    if (scratch_size  < 32 * 1024 * 1024) scratch_size  = 32 * 1024 * 1024;

    BeamHeapD heap, scratch;
    int rc;
    if ((rc = heap_alloc_d(ctx, &heap,    out_heap_size, s)) != 0) return rc;
    if ((rc = heap_alloc_d(ctx, &scratch, scratch_size,  s)) != 0) {
        heap_free_d(&heap); return rc;
    }
    CHECK_CU(cuStreamSynchronize(s));

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

    void* args[] = {
        &d_verts, &d_tris, &n_verts, &n_tris,
        &pa, &pb, &pc_n, &pd,
        &d_opv, &d_opt, &d_onv, &d_ont,
        &out_pos_verts_cap, &out_pos_tris_cap,
        &out_neg_verts_cap, &out_neg_tris_cap,
        &d_npv, &d_npt, &d_nnv, &d_nnt,
        &heap.d_heap, &scratch.d_heap,
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
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "plane_cut kernel error: 0x%x", kerr_h);
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
    heap_free_d(&heap); heap_free_d(&scratch);
    return 0;

cleanup_err:
    cuMemFree(d_verts); cuMemFree(d_tris);
    cuMemFree(d_opv); cuMemFree(d_opt);
    cuMemFree(d_onv); cuMemFree(d_ont);
    cuMemFree(d_counts); cuMemFree(d_kerr);
    heap_free_d(&heap); heap_free_d(&scratch);
    return -1;
}
