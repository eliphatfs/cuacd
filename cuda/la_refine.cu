// la_refine.cu — refinement and quality kernels: la_expand_quick,
// la_hausdorff_parts, la_evaluate, la_sort_items, la_record_level_cost.
//
// Heavy includes: plane_cut.cuh, hausdorff.cuh, mesh_volume.cuh

#include <cstdio>
#include "la_common.cuh"
#include "plane_cut.cuh"
#include "hausdorff.cuh"
#include "mesh_volume.cuh"
#include "warp_common.cuh"

// ============================================================================
// la_hausdorff_parts: <<<LA_MAX_DECOMP, HD_BLOCK>>>
// Compute Hausdorff for parts whose rv-cost is below threshold (lazy eval).
// Parts above threshold already have cost >= threshold, so Hausdorff won't
// change the decision.
// ============================================================================
extern "C" __global__ void la_hausdorff_parts(
    LaDecompState* decomp,
    DevicePool*    pool,
    float          threshold,
    int*           err)
{
    if (*err) return;

    int i = blockIdx.x;
    if (i >= decomp->nparts) return;

    Part* p = &decomp->parts[i];

    // Skip if hull not computed yet, hausdorff already computed,
    // or rv-cost already >= threshold.
    // hausdorff == 0 is the "not computed" sentinel — set by part-creation sites
    // (la_initialize, la_expand_quick, la_apply_cuts, la_decompose_components).
    if (p->hull.verts == NULL) return;
    if (p->hausdorff > 0.0f) return;
    float rv_cost = la_part_cost_rv(*p);
    if (rv_cost >= threshold) return;

    float h = hausdorff_block(&p->hull, &p->mesh, &pool->scratch, err);

    if (threadIdx.x == 0)
        p->hausdorff = h;
}

// ============================================================================
// la_expand_quick: <<<cur_nitems, PC_BLOCK=64>>>
// Quick expansion: each block tries all 3 axes at midpoint and picks the
// best single cut. Produces exactly 1 child per item.
// ============================================================================
extern "C" __global__ void la_expand_quick(
    LaWorkItem* cur_items,
    int         cur_nitems,
    LaWorkItem* next_items,
    int*        next_nitems,
    DevicePool* pool,
    float       min_edge_dist,
    int*        err,
    LaWorkItem* extra_leaves,
    int*        n_extra,
    int         total_levels)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int item_idx = blockIdx.x;
    if (item_idx >= cur_nitems) return;

    LaWorkItem* wi = &cur_items[item_idx];
    int np = wi->nparts;
    if (np <= 0) return;

    Mesh* mesh = &wi->parts[np - 1].mesh;

    // Validate mesh — all threads must see the result
    __shared__ int s_mesh_bad;
    if (tid == 0) {
        s_mesh_bad = (mesh->nv <= 0 || mesh->nt <= 0 ||
                      mesh->nv > 1000000 || mesh->nt > 1000000 ||
                      mesh->verts == NULL || mesh->tris == NULL) ? 1 : 0;
        if (s_mesh_bad) atomicOr(err, KERR_MESH_INVALID);
    }
    __syncthreads();
    if (s_mesh_bad) return;

    // Compute bounding box
    __shared__ float s_lo[3], s_hi[3];
    if (tid < 3) { s_lo[tid] = 1e30f; s_hi[tid] = -1e30f; }
    __syncthreads();

    int  lane    = tid & 31;
    float tlo[3] = { 1e30f,  1e30f,  1e30f};
    float thi[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = tid; i < mesh->nv; i += blockDim.x) {
        float x = mesh->verts[3*i + 0];
        float y = mesh->verts[3*i + 1];
        float z = mesh->verts[3*i + 2];
        tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
        tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
        tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
    }
    for (int a = 0; a < 3; a++) {
        tlo[a] = warp_min_f(tlo[a]);
        thi[a] = warp_max_f(thi[a]);
    }
    if (lane == 0) {
        atomicMinF(&s_lo[0], tlo[0]); atomicMaxF(&s_hi[0], thi[0]);
        atomicMinF(&s_lo[1], tlo[1]); atomicMaxF(&s_hi[1], thi[1]);
        atomicMinF(&s_lo[2], tlo[2]); atomicMaxF(&s_hi[2], thi[2]);
    }
    __syncthreads();

    // Check if worst part is too small to cut in ALL axes
    {
        float eps = 1e-6f;
        bool all_small = (s_hi[0]-s_lo[0] <= 2.f*min_edge_dist + eps) &&
                         (s_hi[1]-s_lo[1] <= 2.f*min_edge_dist + eps) &&
                         (s_hi[2]-s_lo[2] <= 2.f*min_edge_dist + eps);

        if (all_small) {
            if (tid == 0 && extra_leaves != NULL) {
                int ei = atomicAdd(n_extra, 1);
                LaWorkItem* xl = &extra_leaves[ei];
                *xl = *wi;
                for (int l = xl->n_levels; l < total_levels && l < LA_MAX_LEVELS; l++)
                    xl->level_costs[l] = 0.0f;
                xl->n_levels = (total_levels < LA_MAX_LEVELS) ? total_levels : LA_MAX_LEVELS;
            }
            return;
        }
    }

    // Try all 3 axes at midpoint, keep the best cut
    // NOTE: We cannot store the best PartPair in __shared__ memory because
    // plane_cut_block's __shared__ PartPair s_result may be aliased to the same
    // shared memory by the compiler (they are never simultaneously live).
    // Instead, we claim output slots immediately and free non-best ones after.
    __shared__ float s_costs[3];        // cost per axis (1e30 = invalid/empty)
    __shared__ float s_vols[3][2];      // [axis][pos/neg] mesh volumes
    __shared__ int   s_nvs[3][2];       // [axis][pos/neg] nv
    __shared__ int   s_nts[3][2];       // [axis][pos/neg] nt
    __shared__ void* s_vptrs[3][2];     // [axis][pos/neg] verts heap ptr
    __shared__ void* s_tptrs[3][2];     // [axis][pos/neg] tris heap ptr
    __shared__ int*  s_rcptrs[3][2];    // [axis][pos/neg] refcount ptr
    __shared__ int   s_valid[3];        // 1 if cut produced two non-empty halves

    if (tid < 3) { s_costs[tid] = 1e30f; s_valid[tid] = 0; }
    __syncthreads();

    for (int axis = 0; axis < 3; axis++) {
        // Skip axes where the extent is too small for a valid midpoint cut
        if (s_hi[axis] - s_lo[axis] <= 2.0f * min_edge_dist + 1e-6f) continue;

        float pa = (axis == 0) ? 1.f : 0.f;
        float pb = (axis == 1) ? 1.f : 0.f;
        float pc = (axis == 2) ? 1.f : 0.f;
        float mid = (s_lo[axis] + s_hi[axis]) * 0.5f;
        float pd  = -mid;

        PartPair pp = plane_cut_block(mesh, pa, pb, pc, pd,
                                      &pool->heap, &pool->scratch, err);
        __syncthreads();

        // If either side empty, free and skip
        if (pp.pos.mesh.nv == 0 || pp.neg.mesh.nv == 0) {
            if (tid == 0) {
                if (pp.pos.mesh.verts) {
                    int r = heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
                    if (r) { DPRINTF("expand_quick[%d]: empty-free pos err=%d axis=%d ptr=%p\n",
                                    item_idx, r, axis, pp.pos.mesh.verts); atomicOr(err, r); }
                }
                if (pp.neg.mesh.verts) {
                    int r = heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
                    if (r) { DPRINTF("expand_quick[%d]: empty-free neg err=%d axis=%d ptr=%p\n",
                                    item_idx, r, axis, pp.neg.mesh.verts); atomicOr(err, r); }
                }
                s_costs[axis] = 1e30f;
                s_valid[axis] = 0;
            }
            __syncthreads();
            continue;
        }

        // Compute mesh volumes for both halves
        {
            __shared__ Mesh s_pos_mesh, s_neg_mesh;
            __shared__ float s_neg_vol;
            if (tid == 0) {
                s_pos_mesh = pp.pos.mesh;
                s_neg_mesh = pp.neg.mesh;
            }
            __syncthreads();

            float pos_vol = 0.0f, neg_vol = 0.0f;
            int warp_id = tid / WARP_SIZE;
            int wlane   = tid & (WARP_SIZE - 1);
            if (warp_id == 0)
                pos_vol = mesh_volume_warp(&s_pos_mesh, wlane);
            else if (warp_id == 1)
                neg_vol = mesh_volume_warp(&s_neg_mesh, wlane);

            // Broadcast neg_vol from warp 1 lane 0 to shared memory
            if (warp_id == 1 && wlane == 0)
                s_neg_vol = neg_vol;
            __syncthreads();

            // Compute cost using mesh volume as proxy
            if (tid == 0) neg_vol = s_neg_vol;
            float max_vol = fmaxf(pos_vol, neg_vol);
            float total_vol = pos_vol + neg_vol + 1e-10f;
            float cost = max_vol / total_vol;  // range (0.5, 1.0]; lower is more balanced

            if (tid == 0) {
                s_costs[axis] = cost;
                s_vols[axis][0] = pos_vol;
                s_vols[axis][1] = neg_vol;
                s_nvs[axis][0] = pp.pos.mesh.nv;
                s_nvs[axis][1] = pp.neg.mesh.nv;
                s_nts[axis][0] = pp.pos.mesh.nt;
                s_nts[axis][1] = pp.neg.mesh.nt;
                s_vptrs[axis][0] = (void*)pp.pos.mesh.verts;
                s_vptrs[axis][1] = (void*)pp.neg.mesh.verts;
                s_tptrs[axis][0] = (void*)pp.pos.mesh.tris;
                s_tptrs[axis][1] = (void*)pp.neg.mesh.tris;
                s_rcptrs[axis][0] = pp.pos.mesh.refcount;
                s_rcptrs[axis][1] = pp.neg.mesh.refcount;
                s_valid[axis] = 1;
            }
        }
        __syncthreads();
    }

    // Find the best axis
    __shared__ int s_best_axis;
    __shared__ float s_best_cost;
    if (tid == 0) {
        s_best_cost = 1e30f;
        s_best_axis = 0;
        for (int a = 0; a < 3; a++) {
            if (s_valid[a] && s_costs[a] < s_best_cost) {
                s_best_cost = s_costs[a];
                s_best_axis = a;
            }
        }
    }
    __syncthreads();

    // Free non-best cuts
    if (tid == 0) {
        for (int a = 0; a < 3; a++) {
            if (s_valid[a] && a != s_best_axis) {
                int r0 = heap_free(&pool->heap, s_vptrs[a][0]);
                int r1 = heap_free(&pool->heap, s_vptrs[a][1]);
                if (r0 || r1) {
                    DPRINTF("expand_quick[%d]: free axis=%d pos_err=%d neg_err=%d ptrs=%p %p\n",
                           item_idx, a, r0, r1, s_vptrs[a][0], s_vptrs[a][1]);
                    atomicOr(err, r0 | r1);
                }
            }
        }
    }
    __syncthreads();

    if (s_best_cost >= 1e30f) return;  // no valid cut found

    // Guard against exceeding part capacity
    if (np + 1 > LA_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, KERR_LA_OVERFLOW);
            heap_free(&pool->heap, s_vptrs[s_best_axis][0]);
            heap_free(&pool->heap, s_vptrs[s_best_axis][1]);
        }
        return;
    }

    // Claim a slot in next
    __shared__ int s_idx;
    if (tid == 0) s_idx = atomicAdd(next_nitems, 1);
    __syncthreads();
    LaWorkItem* wo = &next_items[s_idx];

    // Copy parts[0..np-2] in parallel
    int copy_ints = (np - 1) * (int)(sizeof(Part) / sizeof(int));
    int* src_i = (int*)wi->parts;
    int* dst_i = (int*)wo->parts;
    for (int i = tid; i < copy_ints; i += blockDim.x)
        dst_i[i] = src_i[i];
    __syncthreads();

    // Increment refcounts for copied parts; null out hull mesh pointers
    // (avoid double-free from shared refcount=NULL hulls), but keep hull_vol
    // so rv cost is accurate for inherited parts.
    for (int i = tid; i < np - 1; i += blockDim.x) {
        if (wo->parts[i].mesh.refcount) atomicAdd(wo->parts[i].mesh.refcount, 1);
        wo->parts[i].hull.verts    = NULL;
        wo->parts[i].hull.tris     = NULL;
        wo->parts[i].hull.nv       = 0;
        wo->parts[i].hull.nt       = 0;
        wo->parts[i].hull.refcount = NULL;
        // hull_vol preserved — inherited from parent item
    }

    // Reconstruct best PartPair from saved per-axis data
    int ba = s_best_axis;  // shorthand
    if (tid == 0) {
        // Pos part
        wo->parts[np - 1].mesh.verts    = (float*)s_vptrs[ba][0];
        wo->parts[np - 1].mesh.tris     = (int*)s_tptrs[ba][0];
        wo->parts[np - 1].mesh.nv       = s_nvs[ba][0];
        wo->parts[np - 1].mesh.nt       = s_nts[ba][0];
        wo->parts[np - 1].mesh.refcount = s_rcptrs[ba][0];
        wo->parts[np - 1].hull.verts    = NULL;
        wo->parts[np - 1].hull.tris     = NULL;
        wo->parts[np - 1].hull.nv       = 0;
        wo->parts[np - 1].hull.nt       = 0;
        wo->parts[np - 1].hull.refcount = NULL;
        wo->parts[np - 1].mesh_vol      = s_vols[ba][0];
        wo->parts[np - 1].hull_vol      = 0.0f;
        wo->parts[np - 1].hausdorff     = 0.0f;
        // Neg part
        wo->parts[np].mesh.verts    = (float*)s_vptrs[ba][1];
        wo->parts[np].mesh.tris     = (int*)s_tptrs[ba][1];
        wo->parts[np].mesh.nv       = s_nvs[ba][1];
        wo->parts[np].mesh.nt       = s_nts[ba][1];
        wo->parts[np].mesh.refcount = s_rcptrs[ba][1];
        wo->parts[np].hull.verts    = NULL;
        wo->parts[np].hull.tris     = NULL;
        wo->parts[np].hull.nv       = 0;
        wo->parts[np].hull.nt       = 0;
        wo->parts[np].hull.refcount = NULL;
        wo->parts[np].mesh_vol      = s_vols[ba][1];
        wo->parts[np].hull_vol      = 0.0f;
        wo->parts[np].hausdorff     = 0.0f;

        wo->nparts = np + 1;

        // Inherit tracking from parent
        wo->src_part_idx    = wi->src_part_idx;
        wo->initial_cut_idx = wi->initial_cut_idx;

        // Copy level costs from parent
        wo->n_levels = wi->n_levels;
        for (int l = 0; l < wi->n_levels; l++)
            wo->level_costs[l] = wi->level_costs[l];
    }
}

// ============================================================================
// la_sort_items: <<<nitems, 32>>>
// Sort parts within each LaWorkItem by part_cost ascending.
// ============================================================================
extern "C" __global__ void la_sort_items(
    LaWorkItem* items,
    int         nitems,
    DevicePool* pool,
    int*        err)
{
    if (*err) return;

    int lane     = threadIdx.x;
    int item_idx = blockIdx.x;
    if (item_idx >= nitems) return;

    LaWorkItem* wi = &items[item_idx];
    int np = wi->nparts;
    if (np <= 1) return;

    // Simple insertion sort by rv cost ascending.
    if (lane == 0) {
        for (int i = 1; i < np; i++) {
            Part key = wi->parts[i];
            float kcost = la_part_cost_rv(key);
            int j = i - 1;
            while (j >= 0 && la_part_cost_rv(wi->parts[j]) > kcost) {
                wi->parts[j + 1] = wi->parts[j];
                j--;
            }
            wi->parts[j + 1] = key;
        }
    }
}

// ============================================================================
// la_record_level_cost: <<<nitems, 32>>>
// Record the worst-part cost at the current expansion level.
// Called after la_sort_items so parts are sorted by cost ascending.
// ============================================================================
extern "C" __global__ void la_record_level_cost(
    LaWorkItem* items,
    int         nitems)
{
    int i = blockIdx.x;
    if (i >= nitems) return;

    LaWorkItem* wi = &items[i];
    if (threadIdx.x == 0 && wi->nparts > 0 && wi->n_levels < LA_MAX_LEVELS) {
        float worst_cost = la_part_cost_rv(wi->parts[wi->nparts - 1]);
        wi->level_costs[wi->n_levels] = worst_cost;
        wi->n_levels++;
    }
}

// ============================================================================
// la_evaluate: <<<n_cutting, 32>>>
// For each input part, scan all leaf items to find the best initial cut.
// Path cost = average of level_costs[0..n_levels-1].
// Per initial cut: take minimum path cost among descendant leaves.
// Select the initial cut with the overall minimum cost.
// ============================================================================
extern "C" __global__ void la_evaluate(
    LaWorkItem*    leaf_items,
    int            n_leaf_items,
    int            n_cutting,
    int            total_width,
    int            total_levels,
    LaEvalResult*  results,
    DevicePool*    pool,
    int*           err,
    LaWorkItem*    level0_items,
    float          min_edge_dist,
    LaWorkItem*    extra_leaves,
    int            n_extra_leaves)
{
    if (*err) return;

    int my_idx = blockIdx.x;  // which input part (0..n_cutting-1)
    if (my_idx >= n_cutting) return;

    int lane = threadIdx.x;

    // Per-cut accumulators: track the minimum path cost per initial_cut_idx.
    __shared__ float s_cut_best[512];
    if (lane == 0) {
        for (int c = 0; c < total_width; c++)
            s_cut_best[c] = 1e30f;
    }
    __syncwarp();

    // Scan all leaf items belonging to this input part
    // Each lane processes a strided subset
    for (int i = lane; i < n_leaf_items; i += WARP_SIZE) {
        LaWorkItem* wi = &leaf_items[i];
        if (wi->src_part_idx != my_idx || wi->n_levels == 0) continue;

        // Compute path cost = average of level_costs
        float path_cost = 0.0f;
        for (int l = 0; l < wi->n_levels; l++)
            path_cost += wi->level_costs[l];
        path_cost /= (float)wi->n_levels;

        int cut = wi->initial_cut_idx;
        if (cut >= 0 && cut < total_width) {
            float old = s_cut_best[cut];
            while (path_cost < old) {
                old = atomicMinF(&s_cut_best[cut], path_cost);
                if (old <= path_cost) break;
            }
        }
    }
    __syncwarp();

    // Scan extra leaves (parts too small to cut — padded with cost=0)
    for (int i = lane; i < n_extra_leaves; i += WARP_SIZE) {
        LaWorkItem* wi = &extra_leaves[i];
        if (wi->src_part_idx != my_idx || wi->n_levels == 0) continue;

        float path_cost = 0.0f;
        for (int l = 0; l < wi->n_levels; l++)
            path_cost += wi->level_costs[l];
        path_cost /= (float)wi->n_levels;

        int cut = wi->initial_cut_idx;
        if (cut >= 0 && cut < total_width) {
            float old = s_cut_best[cut];
            while (path_cost < old) {
                old = atomicMinF(&s_cut_best[cut], path_cost);
                if (old <= path_cost) break;
            }
        }
    }
    __syncwarp();

    // Lane 0 finds the best cut
    if (lane == 0) {
        float best = 1e30f;
        int best_idx = 0;
        for (int c = 0; c < total_width; c++) {
            if (s_cut_best[c] < best) {
                best = s_cut_best[c];
                best_idx = c;
            }
        }
        results[my_idx].best_cost     = best;
        results[my_idx].best_cut_idx  = best_idx;
    }
}
