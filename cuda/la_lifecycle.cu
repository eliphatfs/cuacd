// la_lifecycle.cu — init, apply, and finalize kernels: la_initialize,
// la_sort_and_count_cutting, la_apply_cuts, la_hull_decomp,
// la_decompose_components, la_free_decomp.
//
// Heavy includes: kdop_hull.cuh, postprocess.cuh, mesh_volume.cuh

#include <cstdio>
#include "la_common.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"
#include "postprocess.cuh"
#include "warp_common.cuh"

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
// la_sort_and_count_cutting: <<<1, 32>>>
// Phase 1 (lane 0): insertion sort the decomposition's parts array by full
// part cost ascending.  Phase 2 (lane 0, after __syncthreads): count parts
// with cost >= threshold and record their indices (highest-cost first).
// Single warp; both phases are sequential on lane 0 — fine for LA_MAX_DECOMP.
// ============================================================================
extern "C" __global__ void la_sort_and_count_cutting(
    LaDecompState* decomp,
    float          threshold,
    int*           n_cutting,
    int*           cutting_indices,
    int            max_cutting,
    int*           err)
{
    if (*err) return;

    int lane = threadIdx.x;
    int np   = decomp->nparts;

    // Phase 1: insertion sort by full cost (rv + hausdorff) ascending.
    if (lane == 0 && np > 1) {
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

    __syncthreads();

    // Phase 2: count cutting parts.  Above-threshold parts form a contiguous
    // suffix; take the LAST max_cutting (highest cost).
    if (lane != 0) return;

    int first_above = np;
    for (int i = 0; i < np; i++) {
        if (la_part_cost(decomp->parts[i]) >= threshold) {
            first_above = i;
            break;
        }
    }

    int n_above = np - first_above;
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
    int            total_width,
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
    LaWorkItem* best_wi = &level0_items[i * total_width + best_cut];

    if (best_wi->nparts < 2) return;  // shouldn't happen if cut was valid

    // Claim a new slot in decomp for the second half
    __shared__ int s_new_idx;
    if (tid == 0) s_new_idx = atomicAdd(&decomp->nparts, 1);
    __syncthreads();

    if (s_new_idx >= LA_MAX_DECOMP) {
        if (tid == 0) atomicOr(err, KERR_LA_OVERFLOW);
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
                if (free_err) DPRINTF("apply_cuts[%d]: mesh free err=%d ptr=%p rc_was=%d\n",
                                     i, free_err, old->mesh.verts, om);
            }
        }
        // Mesh with refcount=NULL is owned externally (original input) — do not free.
        if (old->hull.refcount == LA_REFCOUNT_HEAP) {
            // Heap-allocated hull from la_hull_decomp — free directly
            free_err = heap_free(&pool->heap, (void*)old->hull.verts);
            if (free_err) DPRINTF("apply_cuts[%d]: hull(HEAP) free err=%d ptr=%p\n",
                                 i, free_err, old->hull.verts);
        } else if (old->hull.refcount) {
            int oh = atomicAdd(old->hull.refcount, -1);
            if (oh == 1) {
                free_err = heap_free(&pool->heap, (void*)old->hull.verts);
                if (free_err) DPRINTF("apply_cuts[%d]: hull(rc) free err=%d ptr=%p rc_was=%d\n",
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
// la_decompose_components: <<<original_nparts, DC_BLOCK=128>>>
// Post-processing: split each part in the decomposition into its connected
// mesh components.  New component parts are appended via atomicAdd on
// decomp->nparts.
// ============================================================================
#define DC_MAX_COMP_LA DC_MAX_OUT   // max components per part in the lookahead kernel

extern "C" __global__ void la_decompose_components(
    LaDecompState* decomp,
    int            original_nparts,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int i = blockIdx.x;
    if (i >= original_nparts) return;

    Part* p = &decomp->parts[i];
    if (p->mesh.nv == 0 || p->mesh.nt == 0) return;

    __shared__ Part s_parts[DC_MAX_OUT];

    int n_comp = decompose_components_block(
        p, s_parts,
        &pool->heap, &pool->scratch, err);

    __syncthreads();

    if (n_comp <= 0) {
        // Error — nothing to do.
        return;
    }

    // [BUG] Fix: decompose_components_block always frees the input mesh/hull
    // (Phase 12), even when inner-shell compaction reduces n_comp back to 1.
    // The surviving component is in s_parts[0] with its own heap-allocated
    // mesh. We must always write s_parts[0] back to decomp->parts[i].
    if (threadIdx.x == 0)
        decomp->parts[i] = s_parts[0];

    if (n_comp == 1) {
        return;
    }

    // Claim new slots for remaining components.
    __shared__ int s_base_idx;
    if (threadIdx.x == 0)
        s_base_idx = atomicAdd(&decomp->nparts, n_comp - 1);
    __syncthreads();

    if (s_base_idx + n_comp - 1 >= LA_MAX_DECOMP) {
        if (threadIdx.x == 0)
            atomicOr(err, KERR_LA_OVERFLOW);
        return;
    }

    if (threadIdx.x == 0) {
        for (int c = 1; c < n_comp; c++)
            decomp->parts[s_base_idx + c - 1] = s_parts[c];
    }
}

// ============================================================================
// la_free_decomp: <<<LA_MAX_DECOMP, 32>>>
// Free heap-allocated mesh/hull data for all parts in a LaDecompState.
// Call after reading results back to host, before freeing d_decomp itself.
// ============================================================================
extern "C" __global__ void la_free_decomp(
    LaDecompState* decomp,
    DevicePool*    pool)
{
    int i = blockIdx.x;
    if (i >= decomp->nparts) return;

    Part* pp = &decomp->parts[i];
    if (threadIdx.x == 0) {
        if (pp->mesh.refcount) {
            int old = atomicAdd(pp->mesh.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)pp->mesh.verts);
        }
        if (pp->hull.refcount == LA_REFCOUNT_HEAP) {
            heap_free(&pool->heap, (void*)pp->hull.verts);
        } else if (pp->hull.refcount) {
            int old = atomicAdd(pp->hull.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)pp->hull.verts);
        }
    }
}
