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
#define MAX_HULL_FACES 512
#define MAX_HULL_VERTS 256

struct PartInfo {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    float bbox[6];
    float rv_cost;
};

struct BeamItem {
    int num_parts;
    int worst_part_idx;
    float worst_cost;
    int cut_count;
};

struct DevicePool {
    char*         base;
    unsigned int* offset;
    unsigned int  capacity;
};

struct MeshPool {
    CUdeviceptr vertices;
    CUdeviceptr triangles;
    int         vert_capacity;
    int         tri_capacity;
    CUdeviceptr vert_offset;
    CUdeviceptr tri_offset;
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

    CUfunction fn_evaluate_candidates;
    CUfunction fn_select_top_k;
    CUfunction fn_apply_cuts;
    CUfunction fn_compute_part_costs;
    CUfunction fn_normalize_mesh;
    CUfunction fn_recover_coordinates;
    CUfunction fn_sample_surface;
    CUfunction fn_point_mesh_distance;
    CUfunction fn_reduce_max;
    CUfunction fn_pairwise_hausdorff;
    CUfunction fn_test_hull_volume;
    CUfunction fn_batch_signed_tet_volume;
    CUfunction fn_batch_point_triangle_dist;
    CUfunction fn_batch_intersect_edge;
    CUfunction fn_batch_rv_from_volumes;
    CUfunction fn_test_block_reduce_max;
    CUfunction fn_test_block_reduce_count;
    CUfunction fn_test_block_reduce_sum;
    CUfunction fn_test_block_reduce_bbox;
    CUfunction fn_test_warp_sort;

    CUfunction fn_batch_hull_incremental;
    CUfunction fn_batch_hull_quickhull;
    CUfunction fn_batch_hull_dandc;
    CUfunction fn_batch_mesh_volume;

    struct MeshPool pool_a, pool_b;
    CUdeviceptr d_parts;
    CUdeviceptr d_beam;
    CUdeviceptr d_planes;
    CUdeviceptr d_cost_buffer;
    CUdeviceptr d_comp_info;
    CUdeviceptr d_winner_beam;
    CUdeviceptr d_winner_plane;
    CUdeviceptr d_winner_costs;
    CUdeviceptr d_norm_info;

    struct DevicePool scratch;
    CUdeviceptr d_scratch_offset;

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
                 "%s failed: %s", #call, _msg ? _msg : "unknown"); \
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
    CHECK_CU(cuModuleGetFunction(&ctx->fn_evaluate_candidates,   ctx->module, "evaluate_candidates"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_select_top_k,          ctx->module, "select_top_k"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_apply_cuts,            ctx->module, "apply_cuts"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_compute_part_costs,    ctx->module, "compute_part_costs"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_normalize_mesh,        ctx->module, "normalize_mesh"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_recover_coordinates,   ctx->module, "recover_coordinates"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_sample_surface,        ctx->module, "sample_surface"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_point_mesh_distance,  ctx->module, "point_mesh_distance"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_reduce_max,           ctx->module, "reduce_max"));
    CHECK_CU(cuModuleGetFunction(&ctx->fn_pairwise_hausdorff,   ctx->module, "pairwise_hausdorff"));
    cuModuleGetFunction(&ctx->fn_test_hull_volume, ctx->module, "test_hull_volume");
    cuModuleGetFunction(&ctx->fn_batch_signed_tet_volume, ctx->module, "batch_signed_tet_volume");
    cuModuleGetFunction(&ctx->fn_batch_point_triangle_dist, ctx->module, "batch_point_triangle_dist");
    cuModuleGetFunction(&ctx->fn_batch_intersect_edge, ctx->module, "batch_intersect_edge");
    cuModuleGetFunction(&ctx->fn_batch_rv_from_volumes, ctx->module, "batch_rv_from_volumes");
    cuModuleGetFunction(&ctx->fn_test_block_reduce_max, ctx->module, "test_block_reduce_max");
    cuModuleGetFunction(&ctx->fn_test_block_reduce_count, ctx->module, "test_block_reduce_count");
    cuModuleGetFunction(&ctx->fn_test_block_reduce_sum, ctx->module, "test_block_reduce_sum");
    cuModuleGetFunction(&ctx->fn_test_block_reduce_bbox, ctx->module, "test_block_reduce_bbox");
    cuModuleGetFunction(&ctx->fn_test_warp_sort,        ctx->module, "test_warp_sort_kernel");
    cuModuleGetFunction(&ctx->fn_batch_hull_incremental, ctx->module, "batch_hull_incremental");
    cuModuleGetFunction(&ctx->fn_batch_hull_quickhull,   ctx->module, "batch_hull_quickhull");
    cuModuleGetFunction(&ctx->fn_batch_hull_dandc,       ctx->module, "batch_hull_dandc");
    cuModuleGetFunction(&ctx->fn_batch_mesh_volume,      ctx->module, "batch_mesh_volume");

    return 0;
}

static void free_mesh_pool(struct MeshPool* pool) {
    if (pool->vertices)    cuMemFree(pool->vertices);
    if (pool->triangles)   cuMemFree(pool->triangles);
    if (pool->vert_offset) cuMemFree(pool->vert_offset);
    if (pool->tri_offset)  cuMemFree(pool->tri_offset);
    memset(pool, 0, sizeof(struct MeshPool));
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
    free_mesh_pool(&ctx->pool_a);
    free_mesh_pool(&ctx->pool_b);
    if (ctx->d_parts)        cuMemFree(ctx->d_parts);
    if (ctx->d_beam)         cuMemFree(ctx->d_beam);
    if (ctx->d_planes)       cuMemFree(ctx->d_planes);
    if (ctx->d_cost_buffer)  cuMemFree(ctx->d_cost_buffer);
    if (ctx->d_comp_info)    cuMemFree(ctx->d_comp_info);
    if (ctx->d_winner_beam)  cuMemFree(ctx->d_winner_beam);
    if (ctx->d_winner_plane) cuMemFree(ctx->d_winner_plane);
    if (ctx->d_winner_costs) cuMemFree(ctx->d_winner_costs);
    if (ctx->d_norm_info)    cuMemFree(ctx->d_norm_info);
    if (ctx->scratch.base)   cuMemFree((CUdeviceptr)(uintptr_t)ctx->scratch.base);
    if (ctx->d_scratch_offset) cuMemFree(ctx->d_scratch_offset);
    if (ctx->module) cuModuleUnload(ctx->module);
    if (ctx->owns_context && ctx->cuda_ctx) cuCtxDestroy(ctx->cuda_ctx);
    free(ctx);
}

const char* beam_last_error(beam_ctx_t ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return NULL;
    return ctx->last_error;
}

// ---------------------------------------------------------------------------
// Helper: allocate mesh pool
// ---------------------------------------------------------------------------

static int alloc_mesh_pool(beam_ctx_t ctx, struct MeshPool* pool, int vert_cap, int tri_cap) {
    pool->vert_capacity = vert_cap;
    pool->tri_capacity = tri_cap;
    CHECK_CU(cuMemAlloc(&pool->vertices,    vert_cap * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&pool->triangles,   tri_cap * 3 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&pool->vert_offset, sizeof(unsigned int)));
    CHECK_CU(cuMemAlloc(&pool->tri_offset,  sizeof(unsigned int)));
    return 0;
}

static int reset_pool_offsets(beam_ctx_t ctx, struct MeshPool* pool, CUstream s) {
    CHECK_CU(cuMemsetD32Async(pool->vert_offset, 0, 1, s));
    CHECK_CU(cuMemsetD32Async(pool->tri_offset, 0, 1, s));
    return 0;
}

// ---------------------------------------------------------------------------
// Helper: generate cutting planes
// ---------------------------------------------------------------------------

static int generate_planes(beam_ctx_t ctx, int cuts_per_axis, CUstream s) {
    int num_planes = 3 * cuts_per_axis;
    float* h_planes = (float*)malloc(num_planes * 4 * sizeof(float));
    if (!h_planes) return -1;

    int idx = 0;
    for (int axis = 0; axis < 3; axis++) {
        for (int i = 0; i < cuts_per_axis; i++) {
            // Plane position: uniformly spaced in [-0.9, 0.9] (normalized coords)
            float pos = -0.9f + 1.8f * (float)(i + 1) / (float)(cuts_per_axis + 1);
            h_planes[idx * 4 + 0] = (axis == 0) ? 1.0f : 0.0f;  // a
            h_planes[idx * 4 + 1] = (axis == 1) ? 1.0f : 0.0f;  // b
            h_planes[idx * 4 + 2] = (axis == 2) ? 1.0f : 0.0f;  // c
            h_planes[idx * 4 + 3] = -pos;                         // d
            idx++;
        }
    }

    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_planes, h_planes, num_planes * 4 * sizeof(float), s));
    free(h_planes);
    return 0;
}



// ---------------------------------------------------------------------------
// Run beam search
// ---------------------------------------------------------------------------

int beam_run(
    beam_ctx_t ctx,
    const float* vertices,
    int n_verts,
    const int* triangles,
    int n_tris,
    const beam_params_t* params)
{
    CUstream s = NULL;  // default stream

    beam_params_t p;
    if (params) {
        p = *params;
    } else {
        beam_params_default(&p);
    }

    int beam_width = p.beam_width;
    if (beam_width > MAX_BEAM) beam_width = MAX_BEAM;
    int num_planes = 3 * p.cuts_per_axis;
    if (num_planes > 64) num_planes = 64;

    // Free previous output
    free_output_parts(ctx);

    // Allocate mesh pools — generous capacity for splitting.
    // With beam_width items, each having up to max_parts parts, and cap
    // triangles added at each iteration, capacity must cover all beam items.
    int vert_cap = (n_verts + 4096) * beam_width * 8;
    int tri_cap  = (n_tris  + 4096) * beam_width * 8;
    if (vert_cap < n_verts * 64) vert_cap = n_verts * 64;
    if (tri_cap  < n_tris  * 64) tri_cap  = n_tris  * 64;

    int rc;
    rc = alloc_mesh_pool(ctx, &ctx->pool_a, vert_cap, tri_cap);
    if (rc) return rc;
    rc = alloc_mesh_pool(ctx, &ctx->pool_b, vert_cap, tri_cap);
    if (rc) return rc;

    // Allocate other device buffers
    CHECK_CU(cuMemAlloc(&ctx->d_parts,
        MAX_BEAM * MAX_PARTS_PER_BEAM * sizeof(struct PartInfo)));
    CHECK_CU(cuMemAlloc(&ctx->d_beam,
        MAX_BEAM * sizeof(struct BeamItem)));
    CHECK_CU(cuMemAlloc(&ctx->d_planes,
        num_planes * 4 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&ctx->d_cost_buffer,
        MAX_BEAM * num_planes * sizeof(float)));
    CHECK_CU(cuMemAlloc(&ctx->d_comp_info,
        MAX_BEAM * num_planes * 2 * sizeof(int)));
    CHECK_CU(cuMemAlloc(&ctx->d_winner_beam,
        beam_width * sizeof(int)));
    CHECK_CU(cuMemAlloc(&ctx->d_winner_plane,
        beam_width * sizeof(int)));
    CHECK_CU(cuMemAlloc(&ctx->d_winner_costs,
        beam_width * sizeof(float)));
    CHECK_CU(cuMemAlloc(&ctx->d_norm_info,
        7 * sizeof(float)));

    // Allocate scratch pool (use available GPU memory)
    size_t free_mem = 0, total_mem = 0;
    cuMemGetInfo(&free_mem, &total_mem);
    size_t scratch_size = (size_t)(free_mem * 0.7);
    if (scratch_size < 64 * 1024 * 1024) scratch_size = 64 * 1024 * 1024;
    // Cap at 4GB — DevicePool.capacity is unsigned int (32-bit)
    if (scratch_size > 4000000000ULL) scratch_size = 4000000000ULL;

    CUdeviceptr d_scratch_base;
    CHECK_CU(cuMemAlloc(&d_scratch_base, scratch_size));
    CHECK_CU(cuMemAlloc(&ctx->d_scratch_offset, sizeof(unsigned int)));

    ctx->scratch.base = (char*)(uintptr_t)d_scratch_base;
    ctx->scratch.offset = (unsigned int*)(uintptr_t)ctx->d_scratch_offset;
    ctx->scratch.capacity = (unsigned int)scratch_size;

    // Upload input mesh to pool_a
    CHECK_CU(cuMemcpyHtoDAsync(ctx->pool_a.vertices, vertices,
        n_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(ctx->pool_a.triangles, triangles,
        n_tris * 3 * sizeof(int), s));

    // Set pool_a offsets
    unsigned int uv = (unsigned int)n_verts;
    unsigned int ut = (unsigned int)n_tris;
    CHECK_CU(cuMemcpyHtoDAsync(ctx->pool_a.vert_offset, &uv, sizeof(unsigned int), s));
    CHECK_CU(cuMemcpyHtoDAsync(ctx->pool_a.tri_offset, &ut, sizeof(unsigned int), s));

    // Normalize mesh to [-1,1]^3
    {
        unsigned int smem = BLOCK_SIZE * 3 * sizeof(float) * 2;
        void* args[] = { &ctx->pool_a.vertices, &n_verts, &ctx->d_norm_info };
        CHECK_CU(cuLaunchKernel(ctx->fn_normalize_mesh,
            1, 1, 1, BLOCK_SIZE, 1, 1, smem, s, args, NULL));
    }

    // Generate cutting planes
    rc = generate_planes(ctx, p.cuts_per_axis, s);
    if (rc) return rc;

    // Initialize beam: 1 item with 1 part (the whole mesh)
    struct PartInfo h_part;
    memset(&h_part, 0, sizeof(h_part));
    h_part.vert_offset = 0;
    h_part.vert_count = n_verts;
    h_part.tri_offset = 0;
    h_part.tri_count = n_tris;
    h_part.rv_cost = 0.0f;  // will be computed

    struct BeamItem h_beam;
    memset(&h_beam, 0, sizeof(h_beam));
    h_beam.num_parts = 1;
    h_beam.worst_part_idx = 0;
    h_beam.worst_cost = 1e30f;
    h_beam.cut_count = 0;

    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_parts, &h_part, sizeof(struct PartInfo), s));
    CHECK_CU(cuMemcpyHtoDAsync(ctx->d_beam, &h_beam, sizeof(struct BeamItem), s));

    // Compute initial Rv
    CHECK_CU(cuMemsetD32Async(ctx->d_scratch_offset, 0, 1, s));
    {
        int one = 1;
        struct DevicePool sp = ctx->scratch;
        void* args[] = {
            &ctx->pool_a.vertices, &ctx->pool_a.triangles,
            &ctx->d_parts, &ctx->d_beam, &one, &p.rv_k, &sp
        };
        CHECK_CU(cuLaunchKernel(ctx->fn_compute_part_costs,
            1, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    }
    CHECK_CU(cuStreamSynchronize(s));

    // Read back worst cost to check if already convex
    CHECK_CU(cuMemcpyDtoHAsync(&h_beam, ctx->d_beam, sizeof(struct BeamItem), s));
    CHECK_CU(cuStreamSynchronize(s));

    if (h_beam.worst_cost <= p.threshold) {
        // Already convex — return single part
        ctx->num_output_parts = 1;
        ctx->output_parts = (struct OutputPart*)calloc(1, sizeof(struct OutputPart));
        ctx->output_parts[0].n_verts = n_verts;
        ctx->output_parts[0].n_tris = n_tris;
        ctx->output_parts[0].vertices = (float*)malloc(n_verts * 3 * sizeof(float));
        ctx->output_parts[0].triangles = (int*)malloc(n_tris * 3 * sizeof(int));
        memcpy(ctx->output_parts[0].vertices, vertices, n_verts * 3 * sizeof(float));
        memcpy(ctx->output_parts[0].triangles, triangles, n_tris * 3 * sizeof(int));
        return 0;
    }

    // Main beam search loop
    struct MeshPool* pool_cur = &ctx->pool_a;
    struct MeshPool* pool_nxt = &ctx->pool_b;
    int num_beam_items = 1;

    for (int iter = 0; iter < p.max_iterations; iter++) {
        // Reset scratch pool
        CHECK_CU(cuMemsetD32Async(ctx->d_scratch_offset, 0, 1, s));

        // Step 1: Evaluate all candidates
        int num_candidates = num_beam_items * num_planes;
        {
            struct DevicePool sp = ctx->scratch;
            void* args[] = {
                &pool_cur->vertices, &pool_cur->triangles,
                &ctx->d_parts, &ctx->d_beam, &ctx->d_planes,
                &num_planes, &num_beam_items, &p.rv_k, &p.threshold,
                &sp, &ctx->d_cost_buffer, &ctx->d_comp_info
            };
            CHECK_CU(cuLaunchKernel(ctx->fn_evaluate_candidates,
                num_candidates, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
        }
        CHECK_CU(cuStreamSynchronize(s));

        // Step 2: Select top-K candidates
        {
            void* args[] = {
                &ctx->d_cost_buffer, &num_candidates,
                &beam_width, &num_planes,
                &ctx->d_winner_beam, &ctx->d_winner_plane, &ctx->d_winner_costs
            };
            CHECK_CU(cuLaunchKernel(ctx->fn_select_top_k,
                1, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
        }

        // Read back winners to check for valid candidates
        float h_winner_costs[MAX_BEAM];
        int h_winner_beam[MAX_BEAM];
        CHECK_CU(cuMemcpyDtoHAsync(h_winner_costs, ctx->d_winner_costs,
            beam_width * sizeof(float), s));
        CHECK_CU(cuMemcpyDtoHAsync(h_winner_beam, ctx->d_winner_beam,
            beam_width * sizeof(int), s));
        CHECK_CU(cuStreamSynchronize(s));

        // Check if best candidate is valid
        if (h_winner_beam[0] < 0 || h_winner_costs[0] >= 1e29f) {
            break;
        }

        // Step 3: Apply cuts — write to pool_nxt
        CHECK_CU(reset_pool_offsets(ctx, pool_nxt, s));
        CHECK_CU(cuMemsetD32Async(ctx->d_scratch_offset, 0, 1, s));

        // Allocate new parts/beam arrays (use same device buffers, offset by MAX_BEAM)
        // We use a separate region for dst parts/beam
        CUdeviceptr d_parts_dst;
        CUdeviceptr d_beam_dst;
        CHECK_CU(cuMemAlloc(&d_parts_dst,
            MAX_BEAM * MAX_PARTS_PER_BEAM * sizeof(struct PartInfo)));
        CHECK_CU(cuMemAlloc(&d_beam_dst,
            MAX_BEAM * sizeof(struct BeamItem)));
        CHECK_CU(cuMemsetD8Async(d_parts_dst, 0,
            MAX_BEAM * MAX_PARTS_PER_BEAM * sizeof(struct PartInfo), s));
        CHECK_CU(cuMemsetD8Async(d_beam_dst, 0,
            MAX_BEAM * sizeof(struct BeamItem), s));

        int actual_winners = 0;
        for (int k = 0; k < beam_width; k++) {
            if (h_winner_beam[k] >= 0 && h_winner_costs[k] < 1e29f)
                actual_winners++;
        }
        if (actual_winners == 0) {
            cuMemFree(d_parts_dst);
            cuMemFree(d_beam_dst);
            break;
        }

        {
            struct DevicePool sp = ctx->scratch;
            void* args[] = {
                &pool_cur->vertices, &pool_cur->triangles,
                &ctx->d_parts, &ctx->d_beam,
                &pool_nxt->vertices, &pool_nxt->triangles,
                &d_parts_dst, &d_beam_dst,
                &ctx->d_planes,
                &ctx->d_winner_beam, &ctx->d_winner_plane,
                &num_planes, &p.rv_k, &sp,
                &pool_nxt->vert_offset, &pool_nxt->tri_offset
            };
            CHECK_CU(cuLaunchKernel(ctx->fn_apply_cuts,
                actual_winners, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
        }

        CHECK_CU(cuStreamSynchronize(s));

        // Swap parts/beam buffers
        cuMemFree(ctx->d_parts);
        cuMemFree(ctx->d_beam);
        ctx->d_parts = d_parts_dst;
        ctx->d_beam = d_beam_dst;

        // Swap pools
        struct MeshPool* tmp = pool_cur;
        pool_cur = pool_nxt;
        pool_nxt = tmp;

        num_beam_items = actual_winners;

        // Recompute Rv for all parts
        CHECK_CU(cuMemsetD32Async(ctx->d_scratch_offset, 0, 1, s));
        {
            struct DevicePool sp = ctx->scratch;
            void* args[] = {
                &pool_cur->vertices, &pool_cur->triangles,
                &ctx->d_parts, &ctx->d_beam, &num_beam_items, &p.rv_k, &sp
            };
            CHECK_CU(cuLaunchKernel(ctx->fn_compute_part_costs,
                num_beam_items, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
        }
        CHECK_CU(cuStreamSynchronize(s));

        // Read back beam items to check termination
        struct BeamItem h_beams[MAX_BEAM];
        CHECK_CU(cuMemcpyDtoHAsync(h_beams, ctx->d_beam,
            num_beam_items * sizeof(struct BeamItem), s));
        CHECK_CU(cuStreamSynchronize(s));

        // Check if best beam item is done (all parts below threshold)
        int best_item = 0;
        float best_worst = h_beams[0].worst_cost;
        for (int k = 1; k < num_beam_items; k++) {
            if (h_beams[k].worst_cost < best_worst) {
                best_worst = h_beams[k].worst_cost;
                best_item = k;
            }
        }

        if (best_worst <= p.threshold) {
            // Done! Select the best beam item (fewest parts among those that satisfy)
            int fewest = p.max_parts + 1;
            for (int k = 0; k < num_beam_items; k++) {
                if (h_beams[k].worst_cost <= p.threshold &&
                    h_beams[k].num_parts < fewest) {
                    fewest = h_beams[k].num_parts;
                    best_item = k;
                }
            }
            num_beam_items = 1;

            // Copy best item to position 0 if not already there
            if (best_item != 0) {
                struct PartInfo h_parts[MAX_PARTS_PER_BEAM];
                CHECK_CU(cuMemcpyDtoHAsync(h_parts,
                    ctx->d_parts + best_item * MAX_PARTS_PER_BEAM * sizeof(struct PartInfo),
                    h_beams[best_item].num_parts * sizeof(struct PartInfo), s));
                CHECK_CU(cuStreamSynchronize(s));
                CHECK_CU(cuMemcpyHtoDAsync(ctx->d_parts, h_parts,
                    h_beams[best_item].num_parts * sizeof(struct PartInfo), s));
                CHECK_CU(cuMemcpyHtoDAsync(ctx->d_beam, &h_beams[best_item],
                    sizeof(struct BeamItem), s));
            }
            break;
        }

        // Continue with only the top beam_width items
        // (already ensured by select_top_k)
    }

    // Recover coordinates on the current pool
    {
        unsigned int h_vert_off;
        CHECK_CU(cuMemcpyDtoHAsync(&h_vert_off, pool_cur->vert_offset,
            sizeof(unsigned int), s));
        CHECK_CU(cuStreamSynchronize(s));

        int total_verts = (int)h_vert_off;
        if (total_verts > 0) {
            int grid = (total_verts + BLOCK_SIZE - 1) / BLOCK_SIZE;
            void* args[] = { &pool_cur->vertices, &total_verts, &ctx->d_norm_info };
            CHECK_CU(cuLaunchKernel(ctx->fn_recover_coordinates,
                grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
        }
    }

    // Download results
    struct BeamItem h_final_beam;
    CHECK_CU(cuMemcpyDtoHAsync(&h_final_beam, ctx->d_beam,
        sizeof(struct BeamItem), s));
    CHECK_CU(cuStreamSynchronize(s));

    int np = h_final_beam.num_parts;
    if (np <= 0) np = 1;

    struct PartInfo* h_parts = (struct PartInfo*)malloc(np * sizeof(struct PartInfo));
    CHECK_CU(cuMemcpyDtoHAsync(h_parts, ctx->d_parts,
        np * sizeof(struct PartInfo), s));
    CHECK_CU(cuStreamSynchronize(s));

    ctx->num_output_parts = np;
    ctx->output_parts = (struct OutputPart*)calloc(np, sizeof(struct OutputPart));

    for (int i = 0; i < np; i++) {
        struct PartInfo* pi = &h_parts[i];
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

        CUdeviceptr v_src = pool_cur->vertices + (CUdeviceptr)(pi->vert_offset * 3 * sizeof(float));
        CUdeviceptr t_src = pool_cur->triangles + (CUdeviceptr)(pi->tri_offset * 3 * sizeof(int));

        CHECK_CU(cuMemcpyDtoHAsync(ctx->output_parts[i].vertices,
            v_src, nv * 3 * sizeof(float), s));
        CHECK_CU(cuMemcpyDtoHAsync(ctx->output_parts[i].triangles,
            t_src, nt * 3 * sizeof(int), s));
    }
    CHECK_CU(cuStreamSynchronize(s));

    // Adjust triangle indices to be local (0-based) per part
    for (int i = 0; i < np; i++) {
        int vo = h_parts[i].vert_offset;
        int nt = ctx->output_parts[i].n_tris;
        int* tris = ctx->output_parts[i].triangles;
        if (!tris) continue;
        for (int t = 0; t < nt * 3; t++) {
            tris[t] -= vo;
        }
    }

    free(h_parts);
    return 0;
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

int beam_test_hull_volume(beam_ctx_t ctx,
                          const float* points, int n_points,
                          float* out_volume, int* out_n_faces) {
    if (!ctx || !ctx->fn_test_hull_volume) return -1;
    CUstream s = NULL;

    CUdeviceptr d_pts, d_res;
    CHECK_CU(cuMemAlloc(&d_pts, n_points * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_res, 3 * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, points, n_points * 3 * sizeof(float), s));

    void* args[] = { &d_pts, &n_points, &d_res };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_hull_volume,
        1, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    float h_res[3];
    CHECK_CU(cuMemcpyDtoHAsync(h_res, d_res, 3 * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    *out_volume = h_res[0];
    if (out_n_faces) *out_n_faces = (int)h_res[1];

    cuMemFree(d_pts);
    cuMemFree(d_res);
    return 0;
}

// ---------------------------------------------------------------------------
// Batch test: signed_tet_volume
// ---------------------------------------------------------------------------

int beam_batch_signed_tet_volume(beam_ctx_t ctx,
    const float* tets, int n, float* out_volumes) {
    if (!ctx || !ctx->fn_batch_signed_tet_volume) return -1;
    CUstream s = NULL;

    CUdeviceptr d_tets, d_vols;
    CHECK_CU(cuMemAlloc(&d_tets, (size_t)n * 9 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_vols, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_tets, tets, (size_t)n * 9 * sizeof(float), s));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_tets, &d_vols, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_signed_tet_volume,
        grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_volumes, d_vols, (size_t)n * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_tets);
    cuMemFree(d_vols);
    return 0;
}

// ---------------------------------------------------------------------------
// Batch test: point_triangle_dist
// ---------------------------------------------------------------------------

int beam_batch_point_triangle_dist(beam_ctx_t ctx,
    const float* points, const float* triangles, int n, float* out_dists) {
    if (!ctx || !ctx->fn_batch_point_triangle_dist) return -1;
    CUstream s = NULL;

    CUdeviceptr d_pts, d_tris, d_dists;
    CHECK_CU(cuMemAlloc(&d_pts, (size_t)n * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_tris, (size_t)n * 9 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_dists, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, points, (size_t)n * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_tris, triangles, (size_t)n * 9 * sizeof(float), s));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_pts, &d_tris, &d_dists, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_point_triangle_dist,
        grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_dists, d_dists, (size_t)n * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts);
    cuMemFree(d_tris);
    cuMemFree(d_dists);
    return 0;
}

// ---------------------------------------------------------------------------
// Batch test: intersect_edge
// ---------------------------------------------------------------------------

int beam_batch_intersect_edge(beam_ctx_t ctx,
    const float* segments, const float* planes, int n, float* out_results) {
    if (!ctx || !ctx->fn_batch_intersect_edge) return -1;
    CUstream s = NULL;

    CUdeviceptr d_segs, d_planes, d_res;
    CHECK_CU(cuMemAlloc(&d_segs, (size_t)n * 6 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_planes, (size_t)n * 4 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_res, (size_t)n * 3 * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_segs, segments, (size_t)n * 6 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_planes, planes, (size_t)n * 4 * sizeof(float), s));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_segs, &d_planes, &d_res, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_intersect_edge,
        grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_results, d_res, (size_t)n * 3 * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_segs);
    cuMemFree(d_planes);
    cuMemFree(d_res);
    return 0;
}

// ---------------------------------------------------------------------------
// Batch test: rv_from_volumes
// ---------------------------------------------------------------------------

int beam_batch_rv_from_volumes(beam_ctx_t ctx,
    const float* mesh_vols, const float* hull_vols, int n,
    float rv_k, float* out_rvs) {
    if (!ctx || !ctx->fn_batch_rv_from_volumes) return -1;
    CUstream s = NULL;

    CUdeviceptr d_mv, d_hv, d_rvs;
    CHECK_CU(cuMemAlloc(&d_mv, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_hv, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_rvs, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_mv, mesh_vols, (size_t)n * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_hv, hull_vols, (size_t)n * sizeof(float), s));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &d_mv, &d_hv, &d_rvs, &rv_k, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_batch_rv_from_volumes,
        grid, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_rvs, d_rvs, (size_t)n * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_mv);
    cuMemFree(d_hv);
    cuMemFree(d_rvs);
    return 0;
}

// ---------------------------------------------------------------------------
// Test: block_reduce_max
// ---------------------------------------------------------------------------

int beam_test_block_reduce_max(beam_ctx_t ctx,
    const float* data, int n, float* out) {
    if (!ctx || !ctx->fn_test_block_reduce_max) return -1;
    CUstream s = NULL;

    int n_blocks = n / BLOCK_SIZE;
    if (n_blocks <= 0) return -1;

    CUdeviceptr d_data, d_out;
    CHECK_CU(cuMemAlloc(&d_data, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_out, (size_t)n_blocks * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_data, data, (size_t)n * sizeof(float), s));

    void* args[] = { &d_data, &d_out, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_block_reduce_max,
        n_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out, d_out, (size_t)n_blocks * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_data);
    cuMemFree(d_out);
    return 0;
}

// ---------------------------------------------------------------------------
// Test: block_reduce_count
// ---------------------------------------------------------------------------

int beam_test_block_reduce_count(beam_ctx_t ctx,
    const int* flags, int n, int* out) {
    if (!ctx || !ctx->fn_test_block_reduce_count) return -1;
    CUstream s = NULL;

    int n_blocks = n / BLOCK_SIZE;
    if (n_blocks <= 0) return -1;

    CUdeviceptr d_flags, d_out;
    CHECK_CU(cuMemAlloc(&d_flags, (size_t)n * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out, (size_t)n_blocks * sizeof(int)));
    CHECK_CU(cuMemcpyHtoDAsync(d_flags, flags, (size_t)n * sizeof(int), s));

    void* args[] = { &d_flags, &d_out, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_block_reduce_count,
        n_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out, d_out, (size_t)n_blocks * sizeof(int), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_flags);
    cuMemFree(d_out);
    return 0;
}

// ---------------------------------------------------------------------------
// Test: block_reduce_sum
// ---------------------------------------------------------------------------

int beam_test_block_reduce_sum(beam_ctx_t ctx,
    const float* data, int n, float* out) {
    if (!ctx || !ctx->fn_test_block_reduce_sum) return -1;
    CUstream s = NULL;

    int n_blocks = n / BLOCK_SIZE;
    if (n_blocks <= 0) return -1;

    CUdeviceptr d_data, d_out;
    CHECK_CU(cuMemAlloc(&d_data, (size_t)n * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_out, (size_t)n_blocks * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_data, data, (size_t)n * sizeof(float), s));

    void* args[] = { &d_data, &d_out, &n };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_block_reduce_sum,
        n_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out, d_out, (size_t)n_blocks * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_data);
    cuMemFree(d_out);
    return 0;
}

// ---------------------------------------------------------------------------
// Test: block_reduce_bbox
// ---------------------------------------------------------------------------

int beam_test_block_reduce_bbox(beam_ctx_t ctx,
    const float* verts, int total_verts,
    const int* offsets, const int* counts, int n_groups,
    float* out_bbox) {
    if (!ctx || !ctx->fn_test_block_reduce_bbox) return -1;
    CUstream s = NULL;

    CUdeviceptr d_verts, d_offsets, d_counts, d_out;
    CHECK_CU(cuMemAlloc(&d_verts, (size_t)total_verts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_offsets, (size_t)n_groups * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_counts, (size_t)n_groups * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_out, (size_t)n_groups * 6 * sizeof(float)));
    CHECK_CU(cuMemcpyHtoDAsync(d_verts, verts, (size_t)total_verts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_offsets, offsets, (size_t)n_groups * sizeof(int), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_counts, counts, (size_t)n_groups * sizeof(int), s));

    void* args[] = { &d_verts, &d_offsets, &d_counts, &d_out, &n_groups };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_block_reduce_bbox,
        n_groups, 1, 1, BLOCK_SIZE, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(out_bbox, d_out, (size_t)n_groups * 6 * sizeof(float), s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_verts);
    cuMemFree(d_offsets);
    cuMemFree(d_counts);
    cuMemFree(d_out);
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

    CUdeviceptr d_pts, d_off, d_scratch;
    size_t pts_bytes = (size_t)total_pts * 4 * sizeof(int);
    CHECK_CU(cuMemAlloc(&d_pts,     pts_bytes));
    CHECK_CU(cuMemAlloc(&d_off,     (size_t)(n_arrays + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_scratch, pts_bytes));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, points, pts_bytes, s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_arrays + 1) * sizeof(int), s));

    int block_size = 64;
    int warps_per_block = block_size / 32;
    int n_blocks = (n_arrays + warps_per_block - 1) / warps_per_block;

    void* args[] = { &d_pts, &d_off, &d_scratch, &n_arrays };
    CHECK_CU(cuLaunchKernel(ctx->fn_test_warp_sort,
        n_blocks, 1, 1, block_size, 1, 1, 0, s, args, NULL));

    CHECK_CU(cuMemcpyDtoHAsync(points, d_pts, pts_bytes, s));
    CHECK_CU(cuStreamSynchronize(s));

    cuMemFree(d_pts);
    cuMemFree(d_off);
    cuMemFree(d_scratch);
    return 0;
}

// ---------------------------------------------------------------------------
// beam_batch_hull_volume
// ---------------------------------------------------------------------------
// algo: 0=incremental (block, max 256 pts), 1=quickhull (warp), 2=dandc (warp)

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
    CUfunction fn = NULL;
    if      (algo == 0) fn = ctx->fn_batch_hull_incremental;
    else if (algo == 1) fn = ctx->fn_batch_hull_quickhull;
    else if (algo == 2) fn = ctx->fn_batch_hull_dandc;
    if (!fn) { snprintf(ctx->last_error, sizeof(ctx->last_error),
                        "batch_hull_volume: algo %d not loaded", algo); return -1; }

    CUstream s = NULL;
    CUdeviceptr d_pts, d_off, d_vols, d_errs, d_scratch = 0;
    CHECK_CU(cuMemAlloc(&d_pts,  (size_t)total_pts * 3 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_off,  (size_t)(n_hulls + 1) * sizeof(int)));
    CHECK_CU(cuMemAlloc(&d_vols, (size_t)n_hulls * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_errs, (size_t)n_hulls * sizeof(int)));
    CHECK_CU(cuMemcpyHtoDAsync(d_pts, pts, (size_t)total_pts * 3 * sizeof(float), s));
    CHECK_CU(cuMemcpyHtoDAsync(d_off, offsets, (size_t)(n_hulls + 1) * sizeof(int), s));

    if (algo == 0) {
        // Incremental: 1 block per hull, shared-memory workspace
        void* args[] = { &d_pts, &d_off, &d_vols, &d_errs, &n_hulls };
        CHECK_CU(cuLaunchKernel(fn, n_hulls, 1, 1,
                                BLOCK_SIZE, 1, 1, 0, s, args, NULL));
    } else {
        // Warp kernels: 8 warps per block (BLOCK_SIZE=256), need scratch
        // D&C hull (Bullet port) needs large structs (~1KB/vertex + edge/face pools).
        // QuickHull needs ~256 bytes/point.
        size_t scratch_per;
        if (algo == 2) {
            // D&C: generous 2MB per hull
            scratch_per = 2 * 1024 * 1024;
        } else {
            scratch_per = (size_t)max_pts_per_hull * 512 + 8192;
        }
        size_t total_scratch = (size_t)n_hulls * scratch_per;
        CHECK_CU(cuMemAlloc(&d_scratch, total_scratch));
        CHECK_CU(cuMemsetD8Async(d_scratch, 0, total_scratch, s));

        int block_size = (algo == 2) ? 64 : BLOCK_SIZE;
        int warps_per_block = block_size / 32;
        int n_blocks = (n_hulls + warps_per_block - 1) / warps_per_block;
        int scratch_per_i = (int)scratch_per;

        // D&C uses recursion — increase thread stack size
        if (algo == 2) {
            CHECK_CU(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 32 * 1024));
        }

        void* args[] = { &d_pts, &d_off, &d_vols, &d_errs,
                         &d_scratch, &scratch_per_i, &n_hulls };
        CHECK_CU(cuLaunchKernel(fn, n_blocks, 1, 1,
                                block_size, 1, 1, 0, s, args, NULL));
    }

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
