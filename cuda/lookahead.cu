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

#include <cstdio>
#include "plane_cut.cuh"
#include "warp_common.cuh"
#include "hull_dandc.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"
#include "warp_sort.cuh"
#include "hausdorff.cuh"

// Sentinel refcount value: marks hull as heap-allocated by kdop_hull_block
// (should be freed with heap_free, not refcount-decremented).
// Valid refcounts are always aligned pointers; (int*)1 is never a valid address.
#define LA_REFCOUNT_HEAP ((int*)1)

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

// rv-only cost (no Hausdorff) — used during tree exploration.
// rv = cbrt(3/(4pi)*(hull_vol - mesh_vol)) — equivalent sphere radius of
// the volume difference. Monotonically decreases with each good cut.
static __device__ inline float la_part_cost_rv(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return LA_PART_COST_K_RV * rv;
}

// Part comparator for sorting decomposition — full cost (rv + hausdorff).
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

// Part comparator for sorting tree items — rv-only cost.
// Tree search should use rv-only for cut selection, not Hausdorff.
struct LAPartKeyCmpRV {
    static __device__ inline float key(const Part& p) { return la_part_cost_rv(p); }
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
        s.hull_vol  = 1e30f;
        s.mesh_vol  = 0.0f;
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

    // Warp 0: mesh volume, warp 1: hull volume, lane 0: metadata
    float mesh_vol = 0.0f;
    float hull_vol = 0.0f;

    int warp_id = lane / WARP_SIZE;
    int wlane   = lane & (WARP_SIZE - 1);

    if (warp_id == 0) {
        Mesh m;
        m.verts    = verts;
        m.tris     = tris;
        m.nv       = nv;
        m.nt       = nt;
        m.refcount = NULL;
        mesh_vol = mesh_volume_warp(&m, wlane);
    } else if (warp_id == 1) {
        Mesh h;
        h.verts    = hull_verts;
        h.tris     = hull_tris;
        h.nv       = hull_nv;
        h.nt       = hull_nt;
        h.refcount = NULL;
        hull_vol = mesh_volume_warp(&h, wlane);
    }

    // Broadcast hull_vol from warp 1 to warp 0 so lane 0 can write both
    __shared__ float s_hull_vol;
    if (warp_id == 1 && wlane == 0)
        s_hull_vol = hull_vol;
    __syncthreads();

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
        p->mesh_vol      = mesh_vol;
        p->hull_vol      = s_hull_vol;
        decomp->nparts   = 1;
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

    // Insertion sort by full cost (rv + hausdorff) ascending.
    // Single-threaded (lane 0) — fine for LA_MAX_DECOMP <= 1024.
    if (lane == 0) {
        for (int i = 1; i < np; i++) {
            Part key = decomp->parts[i];
            float kcost = la_part_cost(key);
            int j = i - 1;
            while (j >= 0 && la_part_cost(decomp->parts[j]) > kcost) {
                decomp->parts[j + 1] = decomp->parts[j];
                j--;
            }
            decomp->parts[j + 1] = key;
        }
    }
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
    // Parts are sorted by cost ascending (la_sort_parts runs before this).
    // Above-threshold parts form a contiguous suffix.  We want the LAST
    // max_cutting of them (highest cost = most in need of cutting).
    //
    // Lane 0 does a sequential scan to find the first above-threshold index,
    // then fills cutting_indices deterministically.
    if (threadIdx.x != 0) return;

    int np = decomp->nparts;

    // Find first part above threshold (binary-search-like, but np ≤ 1024)
    int first_above = np;  // default: none above
    for (int i = 0; i < np; i++) {
        if (la_part_cost(decomp->parts[i]) >= threshold) {
            first_above = i;
            break;
        }
    }

    int n_above = np - first_above;
    // Skip the lowest-cost above-threshold parts if there are too many
    int start = first_above;
    if (n_above > max_cutting)
        start = np - max_cutting;

    int count = 0;
    for (int i = start; i < np && count < max_cutting; i++) {
        if (la_part_cost(decomp->parts[i]) >= threshold) {
            cutting_indices[count] = i;
            count++;
        }
    }
    *n_cutting = count;
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

    // Increment mesh refcount (the source part is still in the decomposition)
    if (src->mesh.refcount) atomicAdd(src->mesh.refcount, 1);

    // Null out hull mesh pointers in the copy — the decomp part still owns
    // its hull, and sharing hull.verts with refcount=NULL → double-free.
    // But preserve hull_vol so the seed item's rv cost is accurate.
    wi->parts[0].hull.verts    = NULL;
    wi->parts[0].hull.tris     = NULL;
    wi->parts[0].hull.nv       = 0;
    wi->parts[0].hull.nt       = 0;
    wi->parts[0].hull.refcount = NULL;
    // hull_vol preserved — inherited from decomp part

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
    float       min_edge_dist,
    int*        err,
    LaWorkItem* extra_leaves,
    int*        n_extra,
    int         total_levels)
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

    // Validate mesh — all threads must see the result
    __shared__ int s_mesh_bad;
    if (tid == 0) {
        s_mesh_bad = (mesh->nv <= 0 || mesh->nt <= 0 ||
                      mesh->nv > 1000000 || mesh->nt > 1000000 ||
                      mesh->verts == NULL || mesh->tris == NULL) ? 1 : 0;
        if (s_mesh_bad) atomicOr(err, 0x100000);
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
            if (cut_idx == 0 && tid == 0 && extra_leaves != NULL) {
                int ei = atomicAdd(n_extra, 1);
                LaWorkItem* xl = &extra_leaves[ei];
                *xl = *wi;  // copy parent state
                // Pad level_costs with 0 from current n_levels to total_levels
                for (int l = xl->n_levels; l < total_levels && l < LA_MAX_LEVELS; l++)
                    xl->level_costs[l] = 0.0f;
                xl->n_levels = (total_levels < LA_MAX_LEVELS) ? total_levels : LA_MAX_LEVELS;
            }
            return;
        }
    }

    // Place cuts evenly in the valid range [lo+min_edge_dist, hi-min_edge_dist]
    float lo_valid = s_lo[axis] + min_edge_dist;
    float hi_valid = s_hi[axis] - min_edge_dist;
    if (lo_valid >= hi_valid) return;  // this axis too narrow (but not all_small — other axes may work)

    float frac = (slice + 1.f) / (cpa + 1.f);
    float cut_pos = lo_valid + frac * (hi_valid - lo_valid);

    float pa = (axis == 0) ? 1.f : 0.f;
    float pb = (axis == 1) ? 1.f : 0.f;
    float pc = (axis == 2) ? 1.f : 0.f;
    float pd = -cut_pos;

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

    // Increment refcounts for copied parts; null out hull mesh pointers
    // (they have refcount=NULL → sharing = double-free), but keep hull_vol
    // so the rv cost is accurate for inherited parts during tree search.
    for (int i = tid; i < np - 1; i += blockDim.x) {
        if (wo->parts[i].mesh.refcount) atomicAdd(wo->parts[i].mesh.refcount, 1);
        wo->parts[i].hull.verts    = NULL;
        wo->parts[i].hull.tris     = NULL;
        wo->parts[i].hull.nv       = 0;
        wo->parts[i].hull.nt       = 0;
        wo->parts[i].hull.refcount = NULL;
        // hull_vol preserved — inherited from parent item
    }

    // Write the two new parts and metadata
    if (tid == 0) {
        wo->parts[np - 1] = pp.pos;
        wo->parts[np]     = pp.neg;
        wo->nparts        = np + 1;

        // Track which input part and which initial cut produced this item.
        // n_levels == 0 means this is the first expansion (seeds from
        // la_seed_tree have n_levels=0; after la_record_level_cost at
        // depth 0, items have n_levels>=1).
        if (wi->n_levels == 0) {
            // Level 0: this is the first expansion
            wo->src_part_idx    = wi->src_part_idx;
            wo->initial_cut_idx = cut_idx;
        } else {
            // Deeper level: inherit from parent
            wo->src_part_idx    = wi->src_part_idx;
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
    if (level0_out != NULL && wi->n_levels == 0) {
        LaWorkItem* l0 = &level0_out[item_idx * width + cut_idx];
        // Copy the output item to level0
        int out_ints = (np + 1) * (int)(sizeof(Part) / sizeof(int));
        int* src_i2 = (int*)wo->parts;
        int* dst_i2 = (int*)l0->parts;
        for (int i = tid; i < out_ints; i += blockDim.x)
            dst_i2[i] = src_i2[i];
        __syncthreads();

        // Null out inherited parts' mesh/hull pointers in the level-0 copy.
        // These parts are copies of the leaf items' inherited parts — their
        // refcounts were already incremented for the leaf copy (line 415).
        // The level-0 copy does NOT get its own refcount increment for inherited
        // parts, so cleanup must not decrement them.  Only the two new parts
        // (at positions np-1 and np) need refcount management here.
        for (int i = tid; i < np - 1; i += blockDim.x) {
            l0->parts[i].mesh.refcount = NULL;
            l0->parts[i].mesh.verts    = NULL;
            l0->parts[i].mesh.nv       = 0;
            l0->parts[i].hull.refcount = NULL;
            l0->parts[i].hull.verts    = NULL;
            l0->parts[i].hull.nv       = 0;
        }

        // Copy metadata
        if (tid == 0) {
            l0->nparts         = wo->nparts;
            l0->src_part_idx   = wo->src_part_idx;
            l0->initial_cut_idx = wo->initial_cut_idx;
            l0->n_levels       = wo->n_levels;
            for (int l = 0; l < wo->n_levels; l++)
                l0->level_costs[l] = wo->level_costs[l];

            // Increment refcounts for the two new parts (pos/neg from plane_cut).
            // The level-0 items share mesh memory with the leaf items (d_cur).
            // Without this increment, cleanup would double-decrement and free
            // memory that the decomp (via la_apply_cuts) still references.
            if (l0->parts[np - 1].mesh.refcount)
                atomicAdd(l0->parts[np - 1].mesh.refcount, 1);
            if (l0->parts[np].mesh.refcount)
                atomicAdd(l0->parts[np].mesh.refcount, 1);
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
        if (s_mesh_bad) atomicOr(err, 0x100000);
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
                    if (r) { printf("expand_quick[%d]: empty-free pos err=%d axis=%d ptr=%p\n",
                                    item_idx, r, axis, pp.pos.mesh.verts); atomicOr(err, r); }
                }
                if (pp.neg.mesh.verts) {
                    int r = heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
                    if (r) { printf("expand_quick[%d]: empty-free neg err=%d axis=%d ptr=%p\n",
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
                    printf("expand_quick[%d]: free axis=%d pos_err=%d neg_err=%d ptrs=%p %p\n",
                           item_idx, a, r0, r1, s_vptrs[a][0], s_vptrs[a][1]);
                    atomicOr(err, r0 ? r0 : r1);
                }
            }
        }
    }
    __syncthreads();

    if (s_best_cost >= 1e30f) return;  // no valid cut found

    // Guard against exceeding part capacity
    if (np + 1 > LA_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, LA_ERR_OVERFLOW);
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
        p->hull.refcount = LA_REFCOUNT_HEAP;
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
    int            width,
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
    __shared__ float s_cut_best[64];
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
        if (cut >= 0 && cut < width) {
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
    DevicePool*    pool,
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
        // Free the old part's mesh and hull (being replaced)
        Part* old = &decomp->parts[part_idx];
        int free_err = 0;
        if (old->mesh.refcount) {
            int om = atomicAdd(old->mesh.refcount, -1);
            if (om == 1) {
                free_err = heap_free(&pool->heap, (void*)old->mesh.verts);
                if (free_err) printf("apply_cuts[%d]: mesh free err=%d ptr=%p rc_was=%d\n",
                                     i, free_err, old->mesh.verts, om);
            }
        }
        // Mesh with refcount=NULL is owned externally (original input) — do not free.
        if (old->hull.refcount == LA_REFCOUNT_HEAP) {
            // Heap-allocated hull from la_hull_decomp — free directly
            free_err = heap_free(&pool->heap, (void*)old->hull.verts);
            if (free_err) printf("apply_cuts[%d]: hull(HEAP) free err=%d ptr=%p\n",
                                 i, free_err, old->hull.verts);
        } else if (old->hull.refcount) {
            int oh = atomicAdd(old->hull.refcount, -1);
            if (oh == 1) {
                free_err = heap_free(&pool->heap, (void*)old->hull.verts);
                if (free_err) printf("apply_cuts[%d]: hull(rc) free err=%d ptr=%p rc_was=%d\n",
                                     i, free_err, old->hull.verts, oh);
            }
        }
        // Hull with refcount=NULL is owned externally (initial input) — do not free.

        // Increment refcounts for the two adopted halves.
        // Skip LA_REFCOUNT_HEAP sentinel — those hulls are owned directly
        // (not refcounted) and will be freed via the sentinel check when
        // this decomp part is eventually replaced.
        if (best_wi->parts[0].mesh.refcount)
            atomicAdd(best_wi->parts[0].mesh.refcount, 1);
        if (best_wi->parts[0].hull.refcount &&
            best_wi->parts[0].hull.refcount != LA_REFCOUNT_HEAP)
            atomicAdd(best_wi->parts[0].hull.refcount, 1);
        if (best_wi->parts[1].mesh.refcount)
            atomicAdd(best_wi->parts[1].mesh.refcount, 1);
        if (best_wi->parts[1].hull.refcount &&
            best_wi->parts[1].hull.refcount != LA_REFCOUNT_HEAP)
            atomicAdd(best_wi->parts[1].hull.refcount, 1);

        // Write first half into the original slot
        decomp->parts[part_idx] = best_wi->parts[0];

        // Write second half into the new slot
        decomp->parts[s_new_idx] = best_wi->parts[1];

        // Null out hulls and meshes in level-0 item so la_cleanup_tree won't
        // double-decrement refcounts or free memory the decomp now owns.
        // The level-0 items and the leaf items (d_cur) share mesh memory from
        // plane_cut_block. The refcount was incremented in la_expand for the
        // level-0 copy and again here for the decomp copy. Nulling out the
        // level-0 item prevents cleanup from decrementing those refcounts.
        best_wi->parts[0].hull.verts    = NULL;
        best_wi->parts[0].hull.nv       = 0;
        best_wi->parts[0].hull.refcount = NULL;
        best_wi->parts[1].hull.verts    = NULL;
        best_wi->parts[1].hull.nv       = 0;
        best_wi->parts[1].hull.refcount = NULL;
        best_wi->parts[0].mesh.verts    = NULL;
        best_wi->parts[0].mesh.nv       = 0;
        best_wi->parts[0].mesh.refcount = NULL;
        best_wi->parts[1].mesh.verts    = NULL;
        best_wi->parts[1].mesh.nv       = 0;
        best_wi->parts[1].mesh.refcount = NULL;
        // Clear nparts so la_cleanup_tree skips this item entirely
        best_wi->nparts = 0;
    }
}

// ============================================================================
// la_hull_decomp: <<<LA_MAX_DECOMP, KDOP_BLOCK=32>>>
// Compute k-DOP hull for decomposition parts that lack hull_vol.
// Called after la_apply_cuts to ensure all parts have valid hull_vol.
// ============================================================================
extern "C" __global__ void la_hull_decomp(
    LaDecompState* decomp,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int i = blockIdx.x;
    if (i >= decomp->nparts) return;

    Part* p = &decomp->parts[i];
    if (p->hull_vol > 0.0f) return;  // already computed
    if (p->mesh.nv == 0) return;     // empty part

    float hvol = 0.0f;
    Mesh hull = kdop_hull_block(
        p->mesh.verts, p->mesh.nv,
        &pool->heap, &pool->scratch,
        &hvol, err);

    if (threadIdx.x == 0) {
        p->hull     = hull;
        p->hull_vol = hvol;
        // Mark hull as heap-allocated (vs initial input hull from la_initialize)
        p->hull.refcount = LA_REFCOUNT_HEAP;

        // Clamp mesh_vol to at most hull_vol — if mesh_volume_warp overestimated
        // due to non-watertight mesh, the part is essentially convex and
        // hull_vol - mesh_vol should be 0.
        if (p->mesh_vol > p->hull_vol)
            p->mesh_vol = p->hull_vol;
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
        // Mesh with refcount=NULL is owned externally (original input) — do not free.
        if (pp->hull.refcount == LA_REFCOUNT_HEAP) {
            // Heap-allocated hull from la_hull — free directly
            heap_free(&pool->heap, (void*)pp->hull.verts);
        } else if (pp->hull.refcount) {
            int old = atomicAdd(pp->hull.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)pp->hull.verts);
        }
    }
    if (tid == 0) wi->nparts = 0;
}
