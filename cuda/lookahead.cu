// lookahead.cu — lookahead tree search decomposition kernels.
//
// Unlike beam search (which maintains a beam of WorkItems across iterations),
// lookahead search maintains a flat decomposition (LaDecompState) and for each
// part needing a cut, explores a shallow tree of candidate cuts to find the
// best single cut to apply.
//
// No Hausdorff is computed during tree exploration — only rv-based cost.
// Hausdorff is computed lazily on the persistent decomposition for the
// stopping criterion.

#include "plane_cut.cuh"
#include "warp_common.cuh"
#include "hull_dandc.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"
#include "warp_sort.cuh"
#include "hausdorff.cuh"

// Error codes (distinct from beam search and plane_cut codes).
#define LA_ERR_OVERFLOW     0x100000  // LaWorkItem exceeded LA_MAX_PARTS
#define LA_ERR_SORT_OOM     0x200000  // scratch heap allocation failed in sort
#define LA_ERR_SORT_STACK   0x400000  // warp_sort_t stack overflow
#define LA_ERR_EVAL_OOM     0x800000  // scratch heap allocation failed in evaluate

// Lookahead uses full cost including Hausdorff — the lookahead tree structure
// handles non-monotonicity well (unlike beam search which fails to converge).
// Cost = max(k_rv * rv, hausdorff) where rv = cbrt(3/(4pi)*(hull_vol - mesh_vol)).
#define LA_PART_COST_K_RV 0.3f

static __device__ inline float la_part_cost(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return fmaxf(LA_PART_COST_K_RV * rv, p.hausdorff);
}

// rv-only cost (no Hausdorff) — used during tree exploration where Hausdorff
// is too expensive and not needed for cut selection.
static __device__ inline float la_part_cost_rv(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return LA_PART_COST_K_RV * rv;
}

// Part comparator for sorting — full cost (rv + hausdorff).
struct LAPartKeyCmp {
    static __device__ inline float key(const Part& p) { return la_part_cost(p); }
    static __device__ inline int cmp(Part a, Part b) {
        float ka = key(a), kb = key(b);
        if (ka < kb) return -1;
        if (ka > kb) return  1;
        unsigned long long pa = (unsigned long long)a.mesh.verts;
        unsigned long long pb = (unsigned long long)b.mesh.verts;
        if (pa < pb) return -1;
        if (pa > pb) return  1;
        return 0;
    }
    static __device__ inline Part sentinel() {
        Part s = {};
        s.hausdorff = 1e30f;
        s.mesh_vol  = 1e30f;
        s.hull_vol  = 0.0f;
        return s;
    }
};

// ============================================================================
// la_initialize: <<<3, 32>>>
// Seed the persistent decomposition with a single part (original mesh + hull).
// ============================================================================
extern "C" __global__ void la_initialize(
    float*         verts,
    int*           tris,
    int            nv,
    int            nt,
    float*         hull_verts,
    int*           hull_tris,
    int            hull_nv,
    int            hull_nt,
    LaDecompState* decomp)
{
    int lane = threadIdx.x;

    if (blockIdx.x == 0) {
        // Compute mesh volume
        Mesh m;
        m.verts    = verts;
        m.tris     = tris;
        m.nv       = nv;
        m.nt       = nt;
        m.refcount = NULL;
        float vol = mesh_volume_warp(&m, lane);
        if (lane == 0)
            decomp->parts[0].mesh_vol = vol;
    } else if (blockIdx.x == 1) {
        // Compute hull volume
        Mesh h;
        h.verts    = hull_verts;
        h.tris     = hull_tris;
        h.nv       = hull_nv;
        h.nt       = hull_nt;
        h.refcount = NULL;
        float vol = mesh_volume_warp(&h, lane);
        if (lane == 0)
            decomp->parts[0].hull_vol = vol;
    } else {
        // Set up part 0 metadata
        if (lane == 0) {
            Part* p          = &decomp->parts[0];
            p->mesh.verts    = verts;
            p->mesh.tris     = tris;
            p->mesh.nv       = nv;
            p->mesh.nt       = nt;
            p->mesh.refcount = NULL;
            p->hull.verts    = hull_verts;
            p->hull.tris     = hull_tris;
            p->hull.nv       = hull_nv;
            p->hull.nt       = hull_nt;
            p->hull.refcount = NULL;
            p->hausdorff     = 0.0f;
            decomp->nparts   = 1;
        }
    }
}

// ============================================================================
// la_sort_parts: <<<1, 32>>>
// Sort the persistent decomposition's parts array by part_cost ascending.
// Single warp handles up to LA_MAX_DECOMP=256 parts.
// ============================================================================
extern "C" __global__ void la_sort_parts(
    LaDecompState* decomp,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int lane = threadIdx.x;
    int np   = decomp->nparts;
    if (np <= 1) return;

    // Allocate sort scratch
    __shared__ char* s_scratch;
    if (lane == 0) {
        int nb = np * (int)sizeof(Part) + WS_MAX_STACK * 2 * (int)sizeof(int);
        if (heap_alloc(&pool->scratch, nb, (void**)&s_scratch) != 0) {
            s_scratch = NULL;
            atomicOr(err, LA_ERR_SORT_OOM);
        }
    }
    __syncwarp();
    if (s_scratch == NULL) return;

    int rc = warp_sort_t<Part, LAPartKeyCmp>(decomp->parts, s_scratch, np, lane);
    if (rc != 0 && lane == 0)
        atomicOr(err, LA_ERR_SORT_STACK);

    if (lane == 0)
        heap_free(&pool->scratch, s_scratch);
}

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

    // Skip if hull not computed yet or rv-cost already >= threshold
    if (p->hull.verts == NULL) return;
    float rv_cost = la_part_cost_rv(*p);
    if (rv_cost >= threshold) return;

    float h = hausdorff_block(&p->hull, &p->mesh, &pool->scratch, err);

    if (threadIdx.x == 0)
        p->hausdorff = h;
}

// ============================================================================
// la_count_cutting: <<<1, 32>>>
// Count parts with cost >= threshold and record their indices.
// ============================================================================
extern "C" __global__ void la_count_cutting(
    LaDecompState* decomp,
    float          threshold,
    int*           n_cutting,
    int*           cutting_indices,
    int            max_cutting,
    int*           err)
{
    int lane = threadIdx.x;
    int np   = decomp->nparts;

    for (int i = lane; i < np; i += WARP_SIZE) {
        float cost = la_part_cost(decomp->parts[i]);
        if (cost >= threshold) {
            int idx = atomicAdd(n_cutting, 1);
            if (idx < max_cutting)
                cutting_indices[idx] = i;
        }
    }
}

// ============================================================================
// la_seed_tree: <<<n_cutting, 32>>>
// Create level-0 LaWorkItems from the cutting parts.
// Each block creates one LaWorkItem with nparts=1, copying the part from
// the persistent decomposition and incrementing its refcounts.
// ============================================================================
extern "C" __global__ void la_seed_tree(
    LaDecompState* decomp,
    int*           cutting_indices,
    int            n_cutting,
    LaWorkItem*    level0_items)
{
    int i = blockIdx.x;
    if (i >= n_cutting) return;

    int part_idx = cutting_indices[i];
    Part* src    = &decomp->parts[part_idx];
    LaWorkItem* wi = &level0_items[i];

    // Copy the part
    wi->parts[0] = *src;

    // Increment refcounts (the source part is still in the decomposition)
    if (src->mesh.refcount) atomicAdd(src->mesh.refcount, 1);
    if (src->hull.refcount) atomicAdd(src->hull.refcount, 1);

    if (threadIdx.x == 0) {
        wi->nparts         = 1;
        wi->src_part_idx   = i;  // index into cutting list
        wi->initial_cut_idx = 0; // will be overwritten by la_expand
        wi->n_levels       = 0;
        for (int l = 0; l < LA_MAX_LEVELS; l++)
            wi->level_costs[l] = 0.0f;
    }
}

// ============================================================================
// la_expand: <<<width * cur_nitems, PC_BLOCK=64>>>
// Full expansion: each block produces one new LaWorkItem by cutting the
// worst part of one input item along one axis-aligned plane.
// Width cuts per item (cpa = width/3 cuts per axis).
// ============================================================================
extern "C" __global__ void la_expand(
    LaWorkItem* cur_items,
    int         cur_nitems,
    LaWorkItem* next_items,
    int*        next_nitems,
    DevicePool* pool,
    int         width,
    LaWorkItem* level0_out,  // if non-NULL, also write output here at [item_idx*width + cut_idx]
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int cpa      = width / 3;
    int item_idx = blockIdx.x / width;
    int cut_idx  = blockIdx.x % width;
    int axis     = cut_idx / cpa;
    int slice    = cut_idx % cpa;

    if (item_idx >= cur_nitems) return;

    LaWorkItem* wi = &cur_items[item_idx];
    int np = wi->nparts;

    if (np <= 0) return;

    Mesh* mesh = &wi->parts[np - 1].mesh;

    // Validate mesh
    if (tid == 0 && (mesh->nv <= 0 || mesh->nt <= 0 ||
                     mesh->nv > 1000000 || mesh->nt > 1000000 ||
                     mesh->verts == NULL || mesh->tris == NULL)) {
        atomicOr(err, 0x100000);
        return;
    }

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

    // Axis-aligned plane at (slice+1)/(cpa+1) fraction of bbox extent
    float pa = (axis == 0) ? 1.f : 0.f;
    float pb = (axis == 1) ? 1.f : 0.f;
    float pc = (axis == 2) ? 1.f : 0.f;
    float pd = -(s_lo[axis] + (slice + 1.f) / (cpa + 1.f) * (s_hi[axis] - s_lo[axis]));

    PartPair pp = plane_cut_block(mesh, pa, pb, pc, pd,
                                  &pool->heap, &pool->scratch, err);
    __syncthreads();

    // If either side is empty, discard and exit
    if (pp.pos.mesh.nv == 0 || pp.neg.mesh.nv == 0) {
        if (tid == 0) {
            if (pp.pos.mesh.verts) heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
            if (pp.neg.mesh.verts) heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
        }
        return;
    }

    // Guard against exceeding part capacity
    if (np + 1 > LA_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, LA_ERR_OVERFLOW);
            heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
            heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
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

    // Increment refcounts for copied parts
    for (int i = tid; i < np - 1; i += blockDim.x) {
        if (wo->parts[i].mesh.refcount) atomicAdd(wo->parts[i].mesh.refcount, 1);
        if (wo->parts[i].hull.refcount) atomicAdd(wo->parts[i].hull.refcount, 1);
    }

    // Write the two new parts and metadata
    if (tid == 0) {
        wo->parts[np - 1] = pp.pos;
        wo->parts[np]     = pp.neg;
        wo->nparts        = np + 1;

        // Track which input part and which initial cut produced this item
        if (wi->src_part_idx >= 0) {
            // Level 0: this is the first expansion
            wo->src_part_idx   = wi->src_part_idx;
            wo->initial_cut_idx = cut_idx;
        } else {
            // Deeper level: inherit from parent
            wo->src_part_idx   = wi->src_part_idx;
            wo->initial_cut_idx = wi->initial_cut_idx;
        }

        // Copy level costs from parent
        wo->n_levels = wi->n_levels;
        for (int l = 0; l < wi->n_levels; l++)
            wo->level_costs[l] = wi->level_costs[l];
    }
    __syncthreads();

    // Warp 0 computes pos mesh volume, warp 1 computes neg mesh volume
    int warp_id = tid / WARP_SIZE;
    int wlane   = tid & (WARP_SIZE - 1);
    if (warp_id == 0) {
        float vol = mesh_volume_warp(&wo->parts[np - 1].mesh, wlane);
        if (wlane == 0) wo->parts[np - 1].mesh_vol = vol;
    } else if (warp_id == 1) {
        float vol = mesh_volume_warp(&wo->parts[np].mesh, wlane);
        if (wlane == 0) wo->parts[np].mesh_vol = vol;
    }

    // Also write to level0_out if provided (first expansion level only)
    if (level0_out != NULL && wi->src_part_idx >= 0) {
        LaWorkItem* l0 = &level0_out[item_idx * width + cut_idx];
        // Copy the output item to level0
        int out_ints = (np + 1) * (int)(sizeof(Part) / sizeof(int));
        int* src_i2 = (int*)wo->parts;
        int* dst_i2 = (int*)l0->parts;
        for (int i = tid; i < out_ints; i += blockDim.x)
            dst_i2[i] = src_i2[i];
        // Copy metadata
        if (tid == 0) {
            l0->nparts         = wo->nparts;
            l0->src_part_idx   = wo->src_part_idx;
            l0->initial_cut_idx = wo->initial_cut_idx;
            l0->n_levels       = wo->n_levels;
            for (int l = 0; l < wo->n_levels; l++)
                l0->level_costs[l] = wo->level_costs[l];
        }
    }
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
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int item_idx = blockIdx.x;
    if (item_idx >= cur_nitems) return;

    LaWorkItem* wi = &cur_items[item_idx];
    int np = wi->nparts;
    if (np <= 0) return;

    Mesh* mesh = &wi->parts[np - 1].mesh;

    // Validate mesh
    if (tid == 0 && (mesh->nv <= 0 || mesh->nt <= 0 ||
                     mesh->nv > 1000000 || mesh->nt > 1000000 ||
                     mesh->verts == NULL || mesh->tris == NULL)) {
        atomicOr(err, 0x100000);
        return;
    }

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

    // Try all 3 axes at midpoint, keep the best cut
    __shared__ PartPair s_best_pp;
    __shared__ float    s_best_cost;
    __shared__ int      s_best_axis;
    __shared__ int      s_valid_cut;

    if (tid == 0) { s_best_cost = 1e30f; s_valid_cut = 0; }
    __syncthreads();

    for (int axis = 0; axis < 3; axis++) {
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
                if (pp.pos.mesh.verts) heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
                if (pp.neg.mesh.verts) heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
            }
            __syncthreads();
            continue;
        }

        // Compute mesh volumes for both halves
        float pos_vol = 0.0f, neg_vol = 0.0f;
        int warp_id = tid / WARP_SIZE;
        int wlane   = tid & (WARP_SIZE - 1);
        if (warp_id == 0)
            pos_vol = mesh_volume_warp(&pp.pos.mesh, wlane);
        else if (warp_id == 1)
            neg_vol = mesh_volume_warp(&pp.neg.mesh, wlane);

        // Compute rv-only cost of this cut = max(rv_cost(pos), rv_cost(neg))
        // rv = k_rv * cbrt(3/(4pi) * max(hull_vol - mesh_vol, 0))
        // Since hull not computed yet, approximate hull_vol ≈ mesh_vol (rv ≈ 0).
        // This means rv-only cost is 0 for both halves. We need hull for a
        // meaningful comparison. Use volume ratio as a simple proxy instead:
        //   cut_cost = max(pos_vol, neg_vol) / (pos_vol + neg_vol + eps)
        // Better: just use the volume of the larger half as cost proxy.
        // For now: store the pair and let hull computation fill in the real cost.

        // Check if this is the best cut so far (using mesh volume as proxy)
        float max_vol = fmaxf(pos_vol, neg_vol);
        float total_vol = pos_vol + neg_vol + 1e-10f;
        float cost = max_vol / total_vol;  // range (0.5, 1.0]; lower is more balanced

        if (tid == 0) {
            if (cost < s_best_cost) {
                // Free previous best
                if (s_valid_cut) {
                    heap_free(&pool->heap, (void*)s_best_pp.pos.mesh.verts);
                    heap_free(&pool->heap, (void*)s_best_pp.neg.mesh.verts);
                }
                s_best_pp   = pp;
                s_best_cost = cost;
                s_best_axis = axis;
                s_valid_cut = 1;
                // Store volumes in the PartPair's mesh_vol fields
                s_best_pp.pos.mesh_vol = pos_vol;
                s_best_pp.neg.mesh_vol = neg_vol;
            } else {
                // Free this cut's meshes
                heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
                heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
            }
        }
        __syncthreads();
    }

    if (!s_valid_cut) return;  // no valid cut found for this item

    // Guard against exceeding part capacity
    if (np + 1 > LA_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, LA_ERR_OVERFLOW);
            heap_free(&pool->heap, (void*)s_best_pp.pos.mesh.verts);
            heap_free(&pool->heap, (void*)s_best_pp.neg.mesh.verts);
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

    // Increment refcounts for copied parts
    for (int i = tid; i < np - 1; i += blockDim.x) {
        if (wo->parts[i].mesh.refcount) atomicAdd(wo->parts[i].mesh.refcount, 1);
        if (wo->parts[i].hull.refcount) atomicAdd(wo->parts[i].hull.refcount, 1);
    }

    // Write the two new parts and metadata
    if (tid == 0) {
        wo->parts[np - 1] = s_best_pp.pos;
        wo->parts[np]     = s_best_pp.neg;
        wo->nparts        = np + 1;

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
// la_hull: <<<n_new_parts * nitems, KDOP_BLOCK=32>>>
// Compute k-DOP hull for the newest parts of each LaWorkItem.
// n_new_parts = 2 for full/quick expansion (the two halves).
// ============================================================================
extern "C" __global__ void la_hull(
    LaWorkItem* items,
    int         nitems,
    int         n_new_parts,
    DevicePool* pool,
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int item_idx = blockIdx.x / n_new_parts;
    int part_off = blockIdx.x % n_new_parts;

    if (item_idx >= nitems) return;

    LaWorkItem* wi = &items[item_idx];
    int np = wi->nparts;
    int part_idx = np - n_new_parts + part_off;
    if (part_idx < 0 || part_idx >= np) return;

    Part* p = &wi->parts[part_idx];
    if (p->hull.verts != NULL) return;

    float hvol = 0.0f;
    Mesh hull = kdop_hull_block(
        p->mesh.verts, p->mesh.nv,
        &pool->heap, &pool->scratch,
        &hvol, err);

    if (tid == 0) {
        p->hull     = hull;
        p->hull_vol = hvol;
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

    __shared__ char* s_scratch;
    if (lane == 0) {
        int nb = np * (int)sizeof(Part) + WS_MAX_STACK * 2 * (int)sizeof(int);
        if (heap_alloc(&pool->scratch, nb, (void**)&s_scratch) != 0) {
            s_scratch = NULL;
            atomicOr(err, LA_ERR_SORT_OOM);
        }
    }
    __syncwarp();
    if (s_scratch == NULL) return;

    int rc = warp_sort_t<Part, LAPartKeyCmp>(wi->parts, s_scratch, np, lane);
    if (rc != 0 && lane == 0)
        atomicOr(err, LA_ERR_SORT_STACK);

    if (lane == 0)
        heap_free(&pool->scratch, s_scratch);
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
        float worst_cost = la_part_cost(wi->parts[wi->nparts - 1]);
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
    int            width,
    int            total_levels,
    LaEvalResult*  results,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int my_idx = blockIdx.x;  // which input part (0..n_cutting-1)
    if (my_idx >= n_cutting) return;

    int lane = threadIdx.x;

    // Per-cut accumulators: track the minimum path cost per initial_cut_idx.
    __shared__ float s_cut_best[32];  // indexed by initial_cut_idx (max width=30)
    if (lane == 0) {
        for (int c = 0; c < width; c++)
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
        if (cut >= 0 && cut < width) {
            // Atomic min into shared memory (only lane 0 needs to update,
            // but since different lanes might update different cuts, we use
            // a simple comparison loop)
            // Since each lane has its own cut index, we need a per-cut lock-free min.
            // Use atomicMin on a float via atomicMinF or int CAS.
            // Simplest: lane 0 does all updates via sequential scan.
            // But we're parallel across lanes. Use atomicCAS float min.
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
        for (int c = 0; c < width; c++) {
            if (s_cut_best[c] < best) {
                best = s_cut_best[c];
                best_idx = c;
            }
        }
        results[my_idx].best_cost     = best;
        results[my_idx].best_cut_idx  = best_idx;
    }
}

// ============================================================================
// la_apply_cuts: <<<n_cutting, 64>>>
// Apply the best cuts to the persistent decomposition.
// Each block replaces one part in the decomposition with its two halves.
// ============================================================================
extern "C" __global__ void la_apply_cuts(
    LaDecompState* decomp,
    int*           cutting_indices,
    int            n_cutting,
    LaWorkItem*    level0_items,
    LaEvalResult*  results,
    int            width,
    int*           err)
{
    if (*err) return;

    int tid = threadIdx.x;
    int i   = blockIdx.x;
    if (i >= n_cutting) return;

    int part_idx = cutting_indices[i];
    int best_cut = results[i].best_cut_idx;

    // The level-0 item produced by the best cut
    LaWorkItem* best_wi = &level0_items[i * width + best_cut];

    if (best_wi->nparts < 2) return;  // shouldn't happen if cut was valid

    // Claim a new slot in decomp for the second half
    __shared__ int s_new_idx;
    if (tid == 0) s_new_idx = atomicAdd(&decomp->nparts, 1);
    __syncthreads();

    if (s_new_idx >= LA_MAX_DECOMP) {
        if (tid == 0) atomicOr(err, LA_ERR_OVERFLOW);
        return;
    }

    // Replace the cutting part with the first half
    // Add the second half at the new slot
    if (tid == 0) {
        // Increment refcounts for the two adopted halves
        if (best_wi->parts[0].mesh.refcount)
            atomicAdd(best_wi->parts[0].mesh.refcount, 1);
        if (best_wi->parts[0].hull.refcount)
            atomicAdd(best_wi->parts[0].hull.refcount, 1);
        if (best_wi->parts[1].mesh.refcount)
            atomicAdd(best_wi->parts[1].mesh.refcount, 1);
        if (best_wi->parts[1].hull.refcount)
            atomicAdd(best_wi->parts[1].hull.refcount, 1);

        // Write first half into the original slot
        decomp->parts[part_idx] = best_wi->parts[0];

        // Write second half into the new slot
        decomp->parts[s_new_idx] = best_wi->parts[1];
    }
}

// ============================================================================
// la_cleanup_tree: <<<nitems, 32>>>
// Decrement refcounts and free meshes for all parts in each LaWorkItem.
// Items with nparts=0 are no-ops (already cleaned up or never used).
// ============================================================================
extern "C" __global__ void la_cleanup_tree(
    LaWorkItem* items,
    int         nitems,
    DevicePool* pool)
{
    int tid = threadIdx.x;
    int i   = blockIdx.x;
    if (i >= nitems) return;

    LaWorkItem* wi = &items[i];
    int np = wi->nparts;

    for (int p = tid; p < np; p += WARP_SIZE) {
        Part* pp = &wi->parts[p];
        if (pp->mesh.refcount) {
            int old = atomicAdd(pp->mesh.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)pp->mesh.verts);
        }
        if (pp->hull.refcount) {
            int old = atomicAdd(pp->hull.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)pp->hull.verts);
        }
    }
    if (tid == 0) wi->nparts = 0;
}
