// lookahead.c — Host-side implementation for lookahead tree search decomposition.
// Uses CUDA driver API exclusively — links only against libcuda.so.

#include "beam.h"
#include "structs.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>

// Timing helpers
#include <time.h>
#if defined(_POSIX_C_SOURCE) || defined(__linux__) || defined(__APPLE__)
#define TSTAMP(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define TELAPSED_MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)*1e-6)
#else
#define TSTAMP(t)        ((void)0)
#define TELAPSED_MS(a,b) 0.0
#endif

#define LA_MAX_PARTS_H     16
#define LA_MAX_CUTTING_H   16
#define LA_MAX_DECOMP_H    1024
#define LA_MAX_LEVELS_H    4

struct LaWorkItem_h {
    struct Part_h parts[LA_MAX_PARTS_H];
    int nparts;
    int src_part_idx;
    int initial_cut_idx;
    float level_costs[LA_MAX_LEVELS_H];
    int n_levels;
    int _pad[2];
};

struct LaDecompState_h {
    struct Part_h parts[LA_MAX_DECOMP_H];
    int nparts;
    int _pad;
};

struct LaEvalResult_h {
    float best_cost;
    int best_cut_idx;
    int _pad[2];
};

// Read back the decomposition from device into host result.
static int la_read_result(
    beam_ctx_t ctx, CUdeviceptr d_decomp, struct beam_result* out, CUstream s)
{
    struct LaDecompState_h h_decomp;
    CUresult r = cuMemcpyDtoH(&h_decomp, d_decomp, sizeof(h_decomp));
    if (r != CUDA_SUCCESS) {
        const char* msg = NULL;
        cuGetErrorString(r, &msg);
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "read LaDecompState failed: %s", msg ? msg : "unknown");
        return (int)r;
    }

    int np = h_decomp.nparts;
    out->nparts = np;
    if (np == 0) { out->parts = NULL; return 0; }

    out->parts = (struct beam_part_result*)calloc(np, sizeof(struct beam_part_result));
    if (!out->parts) return -1;

    for (int i = 0; i < np; i++) {
        struct Part_h*           p  = &h_decomp.parts[i];
        struct beam_part_result* pr = &out->parts[i];

        pr->nv        = p->mesh.nv;
        pr->nt        = p->mesh.nt;
        pr->mesh_vol  = p->mesh_vol;
        pr->hull_vol  = p->hull_vol;
        pr->hausdorff = p->hausdorff;

        if (p->mesh.nv > 0 && p->mesh.verts) {
            pr->verts = (float*)malloc((size_t)p->mesh.nv * 3 * sizeof(float));
            cuMemcpyDtoHAsync(pr->verts, (CUdeviceptr)p->mesh.verts,
                              (size_t)p->mesh.nv * 3 * sizeof(float), s);
        }
        if (p->mesh.nt > 0 && p->mesh.tris) {
            pr->tris = (int*)malloc((size_t)p->mesh.nt * 3 * sizeof(int));
            cuMemcpyDtoHAsync(pr->tris, (CUdeviceptr)p->mesh.tris,
                              (size_t)p->mesh.nt * 3 * sizeof(int), s);
        }
    }

    cuStreamSynchronize(s);
    return 0;
}

int lookahead_decompose(
    beam_ctx_t   ctx,
    const float* verts,      int nv,
    const int*   tris,       int nt,
    const float* hull_verts, int hull_nv,
    const int*   hull_tris,  int hull_nt,
    int max_iters, int width, int width2, float threshold,
    int depth, int quick_depth, int max_n_cutting,
    int verbose, int debug,
    struct beam_result* out)
{
#define LCHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _msg = NULL; \
        cuGetErrorString(_r, &_msg); \
        snprintf(ctx->last_error, sizeof(ctx->last_error), \
                 "%s failed at lookahead.c:%d: %s", #call, __LINE__, _msg ? _msg : "unknown"); \
        result_code = (int)_r; \
        goto cleanup; \
    } \
} while(0)

    CUstream s = NULL;
    int result_code = 0;
    LCHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT));

    struct timespec _t0, _t1;

    CUdeviceptr d_verts      = 0, d_tris       = 0;
    CUdeviceptr d_hverts     = 0, d_htris       = 0;
    CUdeviceptr d_decomp     = 0;
    CUdeviceptr d_items_a    = 0, d_items_b     = 0;
    CUdeviceptr d_level0     = 0;
    CUdeviceptr d_err        = 0, d_n_cutting   = 0;
    CUdeviceptr d_cutting_idx = 0, d_results    = 0;
    CUdeviceptr d_nitems     = 0;
    CUdeviceptr d_extra_leaves = 0, d_n_extra   = 0;

    // Pinned host memory for async D2H
    void* h_pinned = NULL;
    LCHECK(cuMemAllocHost(&h_pinned, sizeof(int) * 2));  // err + n_cutting
    int* h_err_p       = (int*)h_pinned;
    int* h_ncutting_p  = (int*)((char*)h_pinned + sizeof(int));

    // Allocate and upload input mesh + hull buffers
    LCHECK(cuMemAllocAsync(&d_verts,  (size_t)nv     * 3 * sizeof(float), s));
    LCHECK(cuMemAllocAsync(&d_tris,   (size_t)nt     * 3 * sizeof(int),   s));
    LCHECK(cuMemAllocAsync(&d_hverts, (size_t)hull_nv * 3 * sizeof(float), s));
    LCHECK(cuMemAllocAsync(&d_htris,  (size_t)hull_nt * 3 * sizeof(int),   s));

    LCHECK(cuMemcpyHtoDAsync(d_verts,  verts,      (size_t)nv     * 3 * sizeof(float), s));
    LCHECK(cuMemcpyHtoDAsync(d_tris,   tris,       (size_t)nt     * 3 * sizeof(int),   s));
    LCHECK(cuMemcpyHtoDAsync(d_hverts, hull_verts, (size_t)hull_nv * 3 * sizeof(float), s));
    LCHECK(cuMemcpyHtoDAsync(d_htris,  hull_tris,  (size_t)hull_nt * 3 * sizeof(int),   s));

    // Persistent decomposition
    LCHECK(cuMemAllocAsync(&d_decomp, sizeof(struct LaDecompState_h), s));
    LCHECK(cuMemsetD8Async(d_decomp, 0, sizeof(struct LaDecompState_h), s));

    // Tree item buffers (double-buffered for expansion levels)
    int max_leaf_items = max_n_cutting;
    for (int d = 0; d < depth; d++)
        max_leaf_items *= (d == 0) ? width : width2;
    // quick_depth doesn't increase item count (1 child per item)

    size_t items_bytes = (size_t)max_leaf_items * sizeof(struct LaWorkItem_h);
    LCHECK(cuMemAllocAsync(&d_items_a, items_bytes, s));
    LCHECK(cuMemAllocAsync(&d_items_b, items_bytes, s));
    LCHECK(cuMemsetD8Async(d_items_a, 0, items_bytes, s));
    LCHECK(cuMemsetD8Async(d_items_b, 0, items_bytes, s));

    // Extra leaves buffer (for parts too small to cut further)
    int max_extra = max_leaf_items;  // conservative upper bound
    size_t extra_bytes = (size_t)max_extra * sizeof(struct LaWorkItem_h);
    LCHECK(cuMemAllocAsync(&d_extra_leaves, extra_bytes, s));
    LCHECK(cuMemsetD8Async(d_extra_leaves, 0, extra_bytes, s));
    LCHECK(cuMemAllocAsync(&d_n_extra, sizeof(int), s));

    // Level-0 items buffer (preserved for la_apply_cuts)
    size_t level0_bytes = (size_t)max_n_cutting * width * sizeof(struct LaWorkItem_h);
    LCHECK(cuMemAllocAsync(&d_level0, level0_bytes, s));
    LCHECK(cuMemsetD8Async(d_level0, 0, level0_bytes, s));

    // Error and status scalars
    LCHECK(cuMemAllocAsync(&d_err,          sizeof(int), s));
    LCHECK(cuMemAllocAsync(&d_n_cutting,    sizeof(int), s));
    LCHECK(cuMemAllocAsync(&d_cutting_idx,  (size_t)max_n_cutting * sizeof(int), s));
    LCHECK(cuMemAllocAsync(&d_results,      (size_t)max_n_cutting * sizeof(struct LaEvalResult_h), s));
    LCHECK(cuMemAllocAsync(&d_nitems,       sizeof(int), s));
    LCHECK(cuMemsetD32Async(d_err,       0, 1, s));
    LCHECK(cuMemsetD32Async(d_n_cutting, 0, 1, s));

    // Seed: la_initialize <<<3, 32>>>
    {
        void* args[] = { &d_verts, &d_tris, &nv, &nt,
                         &d_hverts, &d_htris, &hull_nv, &hull_nt,
                         &d_decomp };
        LCHECK(cuLaunchKernel(ctx->fn_la_init, 1, 1, 1, 64, 1, 1, 0, s, args, NULL));
    }
    LCHECK(cuStreamSynchronize(s));
    if (verbose) fprintf(stderr, "[la] init OK\n");

    #define LA_SYNC_CHECK(label) do { if (debug) { \
        CUresult _sr = cuStreamSynchronize(s); \
        if (_sr != CUDA_SUCCESS) { \
            const char* _m = NULL; cuGetErrorString(_sr, &_m); \
            fprintf(stderr, "[la] CRASH after %s: %s\n", label, _m ? _m : "?"); \
            result_code = (int)_sr; goto cleanup; \
        } \
        int _e = 0; cuMemcpyDtoH(&_e, d_err, sizeof(int)); \
        if (_e) { fprintf(stderr, "[la] ERR 0x%x after %s\n", _e, label); result_code = _e; goto cleanup; } \
        if (verbose) fprintf(stderr, "[la] %s OK\n", label); \
    } } while(0)

    // Main loop
    for (int iter = 0; iter < max_iters; iter++) {
        TSTAMP(_t0);

        // Sort parts by rv-only cost first
        {
            void* args[] = { &d_decomp, &ctx->d_pool_struct, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_sort_parts, 1, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("sort_parts1");

        // Compute Hausdorff for parts with rv-cost below threshold
        {
            void* args[] = { &d_decomp, &ctx->d_pool_struct, &threshold, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_hausdorff_parts,
                                   LA_MAX_DECOMP_H, 1, 1, 256, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("hausdorff_parts");

        // Sort again by full cost (rv + hausdorff)
        {
            void* args[] = { &d_decomp, &ctx->d_pool_struct, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_sort_parts, 1, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("sort_parts2");

        // Diagnostic: print per-part cost breakdown
        if (verbose) {
            struct LaDecompState_h h_diag;
            CUresult _dr = cuMemcpyDtoH(&h_diag, d_decomp, sizeof(h_diag));
            if (_dr == CUDA_SUCCESS) {
                const float pi = 3.14159265358979f;
                fprintf(stderr, "[la]   part   nv   nt  mesh_vol  hull_vol    rv_cost  hausdorff  full_cost\n");
                for (int _i = 0; _i < h_diag.nparts; _i++) {
                    struct Part_h* _p = &h_diag.parts[_i];
                    float rv = cbrtf((3.0f/(4.0f*pi)) * fmaxf(_p->hull_vol - _p->mesh_vol, 0.0f));
                    float rv_cost = 0.3f * rv;
                    float full_cost = fmaxf(rv_cost, _p->hausdorff);
                    fprintf(stderr, "[la]   %3d  %4d %4d  %9.6f %9.6f  %9.6f %10.6f %10.6f%s\n",
                            _i, _p->mesh.nv, _p->mesh.nt,
                            _p->mesh_vol, _p->hull_vol,
                            rv_cost, _p->hausdorff, full_cost,
                            full_cost >= threshold ? " *" : "");
                }
            }
        }

        // Count cutting parts
        LCHECK(cuMemsetD32Async(d_n_cutting, 0, 1, s));
        {
            void* args[] = { &d_decomp, &threshold, &d_n_cutting,
                             &d_cutting_idx, &max_n_cutting, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_count_cutting, 1, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }

        // Read back n_cutting
        LCHECK(cuMemcpyDtoHAsync(h_ncutting_p, d_n_cutting, sizeof(int), s));
        LCHECK(cuStreamSynchronize(s));

        int n_cutting = *h_ncutting_p;
        if (n_cutting > max_n_cutting) n_cutting = max_n_cutting;

        if (verbose) {
            fprintf(stderr, "[la] iter %d: n_cutting=%d", iter, n_cutting);
            if (verbose >= 2) {
                // Print cutting_indices
                int* h_cidx = (int*)malloc((size_t)n_cutting * sizeof(int));
                if (h_cidx) {
                    CUresult _rc = cuMemcpyDtoH(h_cidx, d_cutting_idx,
                                                (size_t)n_cutting * sizeof(int));
                    if (_rc == CUDA_SUCCESS) {
                        fprintf(stderr, " cutting_indices=[");
                        for (int _ci = 0; _ci < n_cutting; _ci++) {
                            if (_ci > 0) fprintf(stderr, " ");
                            fprintf(stderr, "%d", h_cidx[_ci]);
                        }
                        fprintf(stderr, "]");
                    }
                    free(h_cidx);
                }
            }
            fprintf(stderr, "\n");
        }

        if (n_cutting == 0) {
            // All parts converged
            result_code = la_read_result(ctx, d_decomp, out, s);
            goto cleanup;
        }

        // Check error
        *h_err_p = 0;
        LCHECK(cuMemcpyDtoHAsync(h_err_p, d_err, sizeof(int), s));
        LCHECK(cuStreamSynchronize(s));
        if (*h_err_p) {
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                     "lookahead_decompose: GPU error 0x%x at iter %d (sort/hausdorff/count)", *h_err_p, iter);
            result_code = *h_err_p;
            goto cleanup;
        }

        // Seed level-0 items into d_items_a (first expansion's cur buffer)
        {
            void* args[] = { &d_decomp, &d_cutting_idx, &n_cutting, &d_items_a };
            LCHECK(cuLaunchKernel(ctx->fn_la_seed_tree,
                                   n_cutting, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("seed_tree");

        // Double-buffer expansion
        CUdeviceptr d_cur  = d_items_a;  // seeds already written here by la_seed_tree
        CUdeviceptr d_next = d_items_b;
        int cur_n = n_cutting;

        // Minimum distance from bbox edge for cuts (prevents degenerate slivers)
        float min_edge_dist = threshold * 0.25f;

        // Reset n_extra for this iteration
        LCHECK(cuMemsetD32Async(d_n_extra, 0, 1, s));

        // Pre-compute total_levels for this iteration
        int total_levels = depth + quick_depth;

        // Full expansion levels
        for (int d = 0; d < depth; d++) {
            LCHECK(cuMemsetD32Async(d_nitems, 0, 1, s));

            int d_width = (d == 0) ? width : width2;
            int nblocks = d_width * cur_n;
            {
                // First expansion level writes to d_level0; deeper levels pass NULL
                CUdeviceptr l0_ptr = (d == 0) ? d_level0 : (CUdeviceptr)0;
                void* args[] = { &d_cur, &cur_n, &d_next, &d_nitems,
                                 &ctx->d_pool_struct, &d_width, &l0_ptr,
                                 &min_edge_dist, &d_err,
                                 &d_extra_leaves, &d_n_extra, &total_levels };
                LCHECK(cuLaunchKernel(ctx->fn_la_expand,
                                       nblocks, 1, 1, 64, 1, 1, 0, s, args, NULL));
            }
            LA_SYNC_CHECK("expand");

            // Read back next_nitems
            int next_n = 0;
            LCHECK(cuMemcpyDtoHAsync(&next_n, d_nitems, sizeof(int), s));
            LCHECK(cuStreamSynchronize(s));

            if (next_n == 0) break;  // all cuts produced empty halves

            // Hull for 2 new parts per item
            {
                int two = 2;
                void* args[] = { &d_next, &next_n, &two,
                                 &ctx->d_pool_struct, &d_err };
                LCHECK(cuLaunchKernel(ctx->fn_la_hull_la,
                                       2 * next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }
            LA_SYNC_CHECK("hull");

            // Sort parts within items
            {
                void* args[] = { &d_next, &next_n, &ctx->d_pool_struct, &d_err };
                LCHECK(cuLaunchKernel(ctx->fn_la_sort_items,
                                       next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }
            LA_SYNC_CHECK("sort_items");

            // Record level cost
            {
                void* args[] = { &d_next, &next_n };
                LCHECK(cuLaunchKernel(ctx->fn_la_record_level_cost,
                                       next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }

            if (debug) {
                LCHECK(cuStreamSynchronize(s));
                int _he = 0;
                cuMemcpyDtoH(&_he, d_err, sizeof(int));
                if (_he) {
                    fprintf(stderr, "[la] iter %d expand depth=%d err=0x%x\n", iter, d, _he);
                    result_code = _he; goto cleanup;
                }
                unsigned long long _pu = 0;
                cuMemcpyDtoH(&_pu, ctx->d_pool_off, sizeof(unsigned long long));
                fprintf(stderr, "[la] iter %d: expand d=%d n=%d pool=%.1f MB\n",
                        iter, d, next_n, (double)_pu / (1024*1024));
            }

            // Swap buffers
            CUdeviceptr tmp = d_cur;
            d_cur  = d_next;
            d_next = tmp;
            cur_n  = next_n;
        }

        // Quick expansion levels (1 child per item)
        for (int q = 0; q < quick_depth; q++) {
            LCHECK(cuMemsetD32Async(d_nitems, 0, 1, s));

            {
                void* args[] = { &d_cur, &cur_n, &d_next, &d_nitems,
                                 &ctx->d_pool_struct, &min_edge_dist, &d_err,
                                 &d_extra_leaves, &d_n_extra, &total_levels };
                LCHECK(cuLaunchKernel(ctx->fn_la_expand_quick,
                                       cur_n, 1, 1, 64, 1, 1, 0, s, args, NULL));
            }
            {
                LCHECK(cuStreamSynchronize(s));
                int _he = 0;
                cuMemcpyDtoH(&_he, d_err, sizeof(int));
                if (_he) {
                    unsigned long long _pu = 0;
                    cuMemcpyDtoH(&_pu, ctx->d_pool_off, sizeof(unsigned long long));
                    fprintf(stderr, "[la] ERR 0x%x after expand_quick (iter %d, q=%d, cur_n=%d, pool=%.1f MB)\n",
                            _he, iter, q, cur_n, (double)_pu / (1024*1024));
                    result_code = _he; goto cleanup;
                }
                if (verbose) fprintf(stderr, "[la] expand_quick OK\n");
            }

            int next_n = 0;
            LCHECK(cuMemcpyDtoHAsync(&next_n, d_nitems, sizeof(int), s));
            LCHECK(cuStreamSynchronize(s));

            if (next_n == 0) break;

            {
                int two = 2;
                void* args[] = { &d_next, &next_n, &two,
                                 &ctx->d_pool_struct, &d_err };
                LCHECK(cuLaunchKernel(ctx->fn_la_hull_la,
                                       2 * next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }
            LA_SYNC_CHECK("hull_quick");

            {
                void* args[] = { &d_next, &next_n, &ctx->d_pool_struct, &d_err };
                LCHECK(cuLaunchKernel(ctx->fn_la_sort_items,
                                       next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }
            LA_SYNC_CHECK("sort_items_quick");

            {
                void* args[] = { &d_next, &next_n };
                LCHECK(cuLaunchKernel(ctx->fn_la_record_level_cost,
                                       next_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
            }

            CUdeviceptr tmp = d_cur;
            d_cur  = d_next;
            d_next = tmp;
            cur_n  = next_n;
        }

        // Evaluate: find best initial cut per input part
        {
            int h_n_extra = 0;
            LCHECK(cuMemcpyDtoH(&h_n_extra, d_n_extra, sizeof(int)));
            void* args[] = { &d_cur, &cur_n, &n_cutting, &width,
                             &total_levels, &d_results,
                             &ctx->d_pool_struct, &d_err,
                             &d_level0, &min_edge_dist,
                             &d_extra_leaves, &h_n_extra };
            LCHECK(cuLaunchKernel(ctx->fn_la_evaluate,
                                   n_cutting, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("evaluate");

        // Diagnostic: dump tree exploration info when verbose >= 2
        if (verbose >= 2) {
            LCHECK(cuStreamSynchronize(s));

            // Read back eval results
            struct LaEvalResult_h* h_results =
                (struct LaEvalResult_h*)malloc((size_t)n_cutting * sizeof(struct LaEvalResult_h));
            if (h_results) {
                CUresult _rr = cuMemcpyDtoH(h_results, d_results,
                                            (size_t)n_cutting * sizeof(struct LaEvalResult_h));
                if (_rr == CUDA_SUCCESS) {
                    for (int _ci = 0; _ci < n_cutting; _ci++) {
                        fprintf(stderr, "[la]   eval: src_part=%d  best_cut=%d  best_cost=%.6f\n",
                                _ci, h_results[_ci].best_cut_idx, h_results[_ci].best_cost);
                    }
                }
                free(h_results);
            }

            // Read back leaf items (kept alive for per-cut summary below)
            struct LaWorkItem_h* h_leaves =
                (struct LaWorkItem_h*)malloc((size_t)cur_n * sizeof(struct LaWorkItem_h));
            int h_leaves_valid = 0;
            if (h_leaves) {
                CUresult _rl = cuMemcpyDtoH(h_leaves, d_cur,
                                            (size_t)cur_n * sizeof(struct LaWorkItem_h));
                h_leaves_valid = (_rl == CUDA_SUCCESS);
            }
            if (h_leaves_valid) {
                const float pi = 3.14159265358979f;
                fprintf(stderr, "[la]   leaf_items: %d items\n", cur_n);
                for (int _li = 0; _li < cur_n; _li++) {
                    struct LaWorkItem_h* _wi = &h_leaves[_li];
                    if (_wi->nparts <= 0 || _wi->n_levels == 0) continue;
                    // Compute worst rv cost in this item
                    float worst_rv = 0.0f;
                    for (int _p = 0; _p < _wi->nparts; _p++) {
                        struct Part_h* _pp = &_wi->parts[_p];
                        float rv = cbrtf((3.0f/(4.0f*pi)) *
                                         fmaxf(_pp->hull_vol - _pp->mesh_vol, 0.0f));
                        float rv_cost = 0.3f * rv;
                        if (rv_cost > worst_rv) worst_rv = rv_cost;
                    }
                    // Compute path cost = average of level_costs
                    float path_cost = 0.0f;
                    for (int _l = 0; _l < _wi->n_levels; _l++)
                        path_cost += _wi->level_costs[_l];
                    if (_wi->n_levels > 0) path_cost /= (float)_wi->n_levels;

                    fprintf(stderr, "[la]     leaf[%d]: src=%d cut=%d nparts=%d "
                            "n_levels=%d worst_rv=%.6f path_cost=%.6f levels=[",
                            _li, _wi->src_part_idx, _wi->initial_cut_idx,
                            _wi->nparts, _wi->n_levels, worst_rv, path_cost);
                    for (int _l = 0; _l < _wi->n_levels; _l++) {
                        if (_l > 0) fprintf(stderr, " ");
                        fprintf(stderr, "%.6f", _wi->level_costs[_l]);
                    }
                    fprintf(stderr, "]\n");
                }
            }

            // Read back level-0 items for per-cut detail
            {
                int n_l0 = n_cutting * width;
                struct LaWorkItem_h* h_l0 =
                    (struct LaWorkItem_h*)malloc((size_t)n_l0 * sizeof(struct LaWorkItem_h));
                if (h_l0) {
                    CUresult _rl0 = cuMemcpyDtoH(h_l0, d_level0,
                                                  (size_t)n_l0 * sizeof(struct LaWorkItem_h));
                    if (_rl0 == CUDA_SUCCESS) {
                        const float pi = 3.14159265358979f;
                        fprintf(stderr, "[la]   level0_items: %d items\n", n_l0);
                        for (int _src = 0; _src < n_cutting; _src++) {
                            fprintf(stderr, "[la]     src_part=%d:\n", _src);
                            for (int _c = 0; _c < width; _c++) {
                                struct LaWorkItem_h* _wi = &h_l0[_src * width + _c];
                                if (_wi->nparts <= 0) {
                                    fprintf(stderr, "[la]       cut[%d]: EMPTY (no valid split)\n", _c);
                                    continue;
                                }
                                // Per-part rv cost
                                fprintf(stderr, "[la]       cut[%d]: nparts=%d n_levels=%d",
                                        _c, _wi->nparts, _wi->n_levels);
                                // Print level costs
                                fprintf(stderr, " levels=[");
                                for (int _l = 0; _l < _wi->n_levels; _l++) {
                                    if (_l > 0) fprintf(stderr, " ");
                                    fprintf(stderr, "%.6f", _wi->level_costs[_l]);
                                }
                                fprintf(stderr, "]");
                                // Print per-part costs
                                fprintf(stderr, " parts=[");
                                for (int _p = 0; _p < _wi->nparts && _p < 4; _p++) {
                                    struct Part_h* _pp = &_wi->parts[_p];
                                    float rv = cbrtf((3.0f/(4.0f*pi)) *
                                                     fmaxf(_pp->hull_vol - _pp->mesh_vol, 0.0f));
                                    float rv_cost = 0.3f * rv;
                                    if (_p > 0) fprintf(stderr, " ");
                                    fprintf(stderr, "{rv=%.6f hv=%.6f mv=%.6f nv=%d}",
                                            rv_cost, _pp->hull_vol, _pp->mesh_vol, _pp->mesh.nv);
                                }
                                if (_wi->nparts > 4) fprintf(stderr, " ...(%d more)", _wi->nparts - 4);
                                fprintf(stderr, "]\n");
                            }
                        }
                    }
                    free(h_l0);
                }
            }

            // Per-cut summary from leaf items: best path cost per initial_cut_idx
            // (This mirrors what la_evaluate computes on the GPU, but visible on host.)
            if (h_leaves_valid && n_cutting > 0) {
                float* cut_best = (float*)malloc((size_t)n_cutting * (size_t)width * sizeof(float));
                if (cut_best) {
                    for (int _i = 0; _i < n_cutting * width; _i++)
                        cut_best[_i] = 1e30f;

                    for (int _li = 0; _li < cur_n; _li++) {
                        struct LaWorkItem_h* _wi = &h_leaves[_li];
                        if (_wi->nparts <= 0 || _wi->n_levels == 0) continue;
                        if (_wi->src_part_idx < 0 || _wi->src_part_idx >= n_cutting) continue;
                        if (_wi->initial_cut_idx < 0 || _wi->initial_cut_idx >= width) continue;

                        float path_cost = 0.0f;
                        for (int _l = 0; _l < _wi->n_levels; _l++)
                            path_cost += _wi->level_costs[_l];
                        if (_wi->n_levels > 0) path_cost /= (float)_wi->n_levels;

                        int idx = _wi->src_part_idx * width + _wi->initial_cut_idx;
                        if (path_cost < cut_best[idx])
                            cut_best[idx] = path_cost;
                    }

                    for (int _src = 0; _src < n_cutting; _src++) {
                        fprintf(stderr, "[la]   per_cut_best: src_part=%d:", _src);
                        for (int _c = 0; _c < width; _c++) {
                            float v = cut_best[_src * width + _c];
                            if (v < 1e29f)
                                fprintf(stderr, " %d:%.6f", _c, v);
                        }
                        fprintf(stderr, "\n");
                    }
                    free(cut_best);
                }
            }

            free(h_leaves);
        }

        // Apply best cuts to the persistent decomposition
        {
            void* args[] = { &d_decomp, &d_cutting_idx, &n_cutting,
                             &d_level0, &d_results, &width,
                             &ctx->d_pool_struct, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_apply_cuts,
                                   n_cutting, 1, 1, 64, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("apply_cuts");

        // Compute hulls for new parts (la_apply_cuts sets hull_vol=0)
        {
            void* args[] = { &d_decomp, &ctx->d_pool_struct, &d_err };
            LCHECK(cuLaunchKernel(ctx->fn_la_hull_decomp,
                                   LA_MAX_DECOMP_H, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("hull_decomp");

        // Cleanup tree: free meshes from level-0 items
        {
            int n_l0 = n_cutting * width;
            void* args[] = { &d_level0, &n_l0, &ctx->d_pool_struct };
            LCHECK(cuLaunchKernel(ctx->fn_la_cleanup_tree,
                                   n_l0, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("cleanup_level0");

        // Cleanup tree: free meshes from leaf items (d_cur)
        {
            void* args[] = { &d_cur, &cur_n, &ctx->d_pool_struct };
            LCHECK(cuLaunchKernel(ctx->fn_la_cleanup_tree,
                                   cur_n, 1, 1, 32, 1, 1, 0, s, args, NULL));
        }
        LA_SYNC_CHECK("cleanup_leaves");

        // Also cleanup the other buffer (may have intermediate level items)
        {
            int other_n = max_leaf_items;  // over-provisioned; nparts=0 → no-op
            void* args[] = { &d_next, &other_n, &ctx->d_pool_struct };
            LCHECK(cuLaunchKernel(ctx->fn_la_cleanup_tree,
                                   other_n > 1024 ? 1024 : other_n, 1, 1, 32, 1, 1,
                                   0, s, args, NULL));
        }
        LA_SYNC_CHECK("cleanup_other");

        *h_err_p = 0;
        LCHECK(cuMemcpyDtoHAsync(h_err_p, d_err, sizeof(int), s));
        LCHECK(cuStreamSynchronize(s));
        if (*h_err_p) {
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                     "lookahead_decompose: GPU error 0x%x at iter %d", *h_err_p, iter);
            result_code = *h_err_p;
            goto cleanup;
        }

        // Reset error for next iteration
        LCHECK(cuMemsetD32Async(d_err, 0, 1, s));

        TSTAMP(_t1);
        if (verbose)
            fprintf(stderr, "[la] iter %d: %.1f ms\n", iter, TELAPSED_MS(_t0, _t1));
    }

    // Iterations exhausted — read back decomposition
    result_code = la_read_result(ctx, d_decomp, out, s);

cleanup:
    cuStreamSynchronize(s);
    if (d_verts)        cuMemFreeAsync(d_verts,        s);
    if (d_tris)         cuMemFreeAsync(d_tris,         s);
    if (d_hverts)       cuMemFreeAsync(d_hverts,       s);
    if (d_htris)        cuMemFreeAsync(d_htris,        s);
    if (d_decomp)       cuMemFreeAsync(d_decomp,       s);
    if (d_items_a)      cuMemFreeAsync(d_items_a,      s);
    if (d_items_b)      cuMemFreeAsync(d_items_b,      s);
    if (d_level0)       cuMemFreeAsync(d_level0,       s);
    if (d_err)          cuMemFreeAsync(d_err,          s);
    if (d_n_cutting)    cuMemFreeAsync(d_n_cutting,    s);
    if (d_cutting_idx)  cuMemFreeAsync(d_cutting_idx,  s);
    if (d_results)      cuMemFreeAsync(d_results,      s);
    if (d_nitems)       cuMemFreeAsync(d_nitems,       s);
    if (d_extra_leaves) cuMemFreeAsync(d_extra_leaves, s);
    if (d_n_extra)      cuMemFreeAsync(d_n_extra,      s);
    cuStreamSynchronize(s);
    if (h_pinned) cuMemFreeHost(h_pinned);
    cuStreamDestroy(s);

#undef LCHECK
    return result_code;
}
