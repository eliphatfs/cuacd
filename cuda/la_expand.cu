// la_expand.cu — tree expansion kernels: la_expand, la_hull,
// la_seed_tree, la_cleanup_tree.
//
// Heavy includes: plane_cut.cuh, kdop_hull.cuh, mesh_volume.cuh

#include <cstdio>
#include "la_common.cuh"
#include "plane_cut.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"
#include "warp_common.cuh"

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
    LaWorkItem* level0_out,  // if non-NULL, also write output here at [item_idx*total_width + cut_idx]
    float       min_edge_dist,
    int*        err,
    LaWorkItem* extra_leaves,
    int*        n_extra,
    int         total_levels,
    int         n_edge_cuts,           // number of edge-based cut planes
    int         n_concave_edges_max,   // stride per part in edge_planes
    ConcaveEdgePlane* edge_planes)     // edge plane data [n_cutting * 4 * n_concave_edges_max]
{
    if (*err) return;

    int tid      = threadIdx.x;
    int total_width = width + n_edge_cuts;
    int item_idx = blockIdx.x / total_width;
    int cut_idx  = blockIdx.x % total_width;

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

    // For edge-based cuts, read plane coefficients directly and skip bbox computation
    __shared__ float s_pa, s_pb, s_pc, s_pd;
    __shared__ int   s_edge_cut;  // 1 if this is an edge-based cut

    if (cut_idx >= width) {
        // Edge-based cut
        if (tid == 0) {
            s_edge_cut = 1;
            int edge_cut_idx = cut_idx - width;
            ConcaveEdgePlane ep = edge_planes[item_idx * 4 * n_concave_edges_max + edge_cut_idx];
            s_pa = ep.pa; s_pb = ep.pb; s_pc = ep.pc; s_pd = ep.pd;
        }
        __syncthreads();
        // Skip degenerate edge planes (all zeros)
        if (s_pa == 0.0f && s_pb == 0.0f && s_pc == 0.0f) return;
    } else {
        s_edge_cut = 0;
    }

    // Compute bounding box (needed for axis-aligned cuts and all_small check)
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

    // Compute plane coefficients
    float pa, pb, pc, pd;
    if (s_edge_cut) {
        // Use edge-based plane coefficients (already in shared memory)
        pa = s_pa; pb = s_pb; pc = s_pc; pd = s_pd;
    } else {
        // Axis-aligned cut
        int cpa   = width / 3;
        int axis  = cut_idx / cpa;
        int slice = cut_idx % cpa;

        float lo_valid = s_lo[axis] + min_edge_dist;
        float hi_valid = s_hi[axis] - min_edge_dist;
        if (lo_valid >= hi_valid) return;  // this axis too narrow

        float frac = (slice + 1.f) / (cpa + 1.f);
        float cut_pos = lo_valid + frac * (hi_valid - lo_valid);

        pa = (axis == 0) ? 1.f : 0.f;
        pb = (axis == 1) ? 1.f : 0.f;
        pc = (axis == 2) ? 1.f : 0.f;
        pd = -cut_pos;
    }

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
            atomicOr(err, KERR_LA_OVERFLOW);
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
        LaWorkItem* l0 = &level0_out[item_idx * total_width + cut_idx];
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
// la_cleanup_tree_item: shared per-block body for cleanup kernels.
// ============================================================================
static __device__ __forceinline__ void la_cleanup_tree_item(
    LaWorkItem* wi, DevicePool* pool, int tid)
{
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
    int i = blockIdx.x;
    if (i >= nitems) return;
    la_cleanup_tree_item(&items[i], pool, threadIdx.x);
}

// ============================================================================
// la_cleanup_tree3: <<<na + nb + nc, 32>>>
// Fused cleanup across three buffers. Dispatches each block to the appropriate
// buffer based on blockIdx.x. Pass nb=nc=0 (and NULL) to skip a buffer.
// ============================================================================
extern "C" __global__ void la_cleanup_tree3(
    LaWorkItem* items_a, int na,
    LaWorkItem* items_b, int nb,
    LaWorkItem* items_c, int nc,
    DevicePool* pool)
{
    int bi = blockIdx.x;
    LaWorkItem* items;
    int i;
    if (bi < na)                { items = items_a; i = bi; }
    else if (bi < na + nb)      { items = items_b; i = bi - na; }
    else if (bi < na + nb + nc) { items = items_c; i = bi - na - nb; }
    else return;

    la_cleanup_tree_item(&items[i], pool, threadIdx.x);
}
