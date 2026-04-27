// postprocess_merge.cu — CoACD merge-hulls post-processing pass (GPU port).
//
// Pipeline (host orchestrates launches only):
//   1. la_merge_cost_matrix  <<<P*(P-1)/2, 32>>>
//        Per pair (p1>p2): rv quick-reject, bbox reject, concat verts, kdop_hull,
//        compute merged-rv cost. Writes cost[idx] and, if cost < threshold,
//        stores merged-hull + volume in pair_state[idx].
//   2. la_merge_hausdorff    <<<P*(P-1)/2, 256>>>
//        For pairs where pair_state[idx].merged_hull.verts != NULL:
//        compute the shared plane from the two parts' hulls, build a filtered
//        concat mesh (skipping coplanar triangles), call hausdorff_block vs
//        the merged hull, and update cost to max(rv, hausdorff). Frees the
//        merged hull afterwards unless the pair still beats threshold (we
//        keep it for apply; otherwise free here).
//   3. la_merge_match        <<<1, 256>>>
//        Greedy min-cost maximum matching on cost[] under the threshold.
//        Writes (p1,p2) pairs into matches[] and n_matches.
//   4. la_merge_apply        <<<n_matches, 32>>>
//        For each match: concat meshes into a new mesh (on heap), refcount=1.
//        If pair_state has a cached merged hull, adopt it; otherwise recompute
//        via kdop_hull_block. Mark the two source slots with mesh.verts=NULL
//        (tombstone) and free their existing mesh+hull.
//   5. la_merge_compact      <<<1, 128>>>
//        Compact decomp->parts[] in place (drop tombstones).
//
// All merged-hull allocations are refcount = LA_REFCOUNT_HEAP (freed directly).

#include <cstdio>
#include "la_common.cuh"
#include "merge.cuh"
#include "kdop_hull.cuh"
#include "hausdorff.cuh"
#include "warp_common.cuh"

#define LA_MERGE_MATCH_BLOCK 256
#define LA_MERGE_COMPACT_BLOCK 128

// ============================================================================
// la_merge_cost_matrix: <<<P*(P-1)/2, 32>>>
// ============================================================================
extern "C" __global__ void la_merge_cost_matrix(
    LaDecompState* decomp,
    int            P,                 // decomp->nparts at the time of launch
    float          threshold,
    float*         cost_matrix,       // [P*(P-1)/2] out
    LaMergePair*   pair_state,        // [P*(P-1)/2] out
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int idx = blockIdx.x;
    int pair_count = P * (P - 1) / 2;
    if (idx >= pair_count) return;
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // Decode (p1, p2) with p1 > p2.
    int p1, p2;
    la_merge_pair_from_index(idx, P, p1, p2);

    Part* A = &decomp->parts[p1];
    Part* B = &decomp->parts[p2];

    // Shared state
    __shared__ float s_amin[3], s_amax[3], s_bmin[3], s_bmax[3];
    __shared__ float* s_concat;
    __shared__ int   s_bbox_reject;
    __shared__ int   s_local_err;

    if (lane == 0) {
        s_concat = NULL;
        s_bbox_reject = 0;
        s_local_err = 0;
        pair_state[idx].merged_hull.verts = NULL;
        pair_state[idx].merged_hull.tris  = NULL;
        pair_state[idx].merged_hull.nv    = 0;
        pair_state[idx].merged_hull.nt    = 0;
        pair_state[idx].merged_hull.refcount = NULL;
        pair_state[idx].hull_vol     = 0.0f;
        pair_state[idx].mesh_vol_sum = 0.0f;
    }
    __syncwarp();

    // Tombstoned parts (from a previous merge round within same pass) — skip.
    if (A->mesh.nv == 0 || B->mesh.nv == 0 ||
        A->mesh.verts == NULL || B->mesh.verts == NULL) {
        if (lane == 0) cost_matrix[idx] = LA_MERGE_COST_INVALID;
        return;
    }

    const float pi_f = 3.14159265358979f;
    float mesh_vol_sum = A->mesh_vol + B->mesh_vol;

    // ---- Step 1: rv quick reject ----
    // Lower bound on merged-rv:   sum(hull) - sum(mesh) <= merged_hull - merged_mesh
    // is NOT a lower bound (merged_hull can be smaller). However, an upper bound
    // on the rv gap is the one CoACD uses: if hull_A + hull_B - mesh_A - mesh_B
    // already exceeds the threshold in rv terms, the merge is almost certainly
    // bad. This matches the user's request.
    float gap0 = fmaxf(A->hull_vol + B->hull_vol - mesh_vol_sum, 0.0f);
    float rv_upper = LA_PART_COST_K_RV * cbrtf((3.0f / (4.0f * pi_f)) * gap0);
    if (rv_upper >= threshold) {
        if (lane == 0) { cost_matrix[idx] = LA_MERGE_COST_RV_REJECT; }
        return;
    }

    // ---- Step 2: bbox reject ----
    la_merge_bbox_warp(A->mesh.verts, A->mesh.nv, lane, s_amin, s_amax);
    la_merge_bbox_warp(B->mesh.verts, B->mesh.nv, lane, s_bmin, s_bmax);
    __syncwarp();
    if (lane == 0) {
        float gap2 = la_merge_bbox_gap2(s_amin, s_amax, s_bmin, s_bmax);
        float half = threshold * 0.5f;
        if (gap2 >= half * half) {
            cost_matrix[idx] = LA_MERGE_COST_BBOX_REJECT;
            s_bbox_reject = 1;
        }
    }
    __syncwarp();
    if (s_bbox_reject) return;

    // ---- Step 3: allocate concat verts on scratch, copy ----
    int nva = A->hull.nv, nvb = B->hull.nv;
    int nvab = nva + nvb;
    if (lane == 0) {
        void* ptr = NULL;
        if (heap_alloc(&pool->scratch, (unsigned int)(nvab * 3 * sizeof(float)), &ptr) != HEAP_OK
            || !ptr) {
            s_local_err = KERR_HEAP_OOM;
        } else {
            s_concat = (float*)ptr;
        }
    }
    __syncwarp();
    if (s_local_err) {
        if (lane == 0) {
            atomicOr(err, s_local_err);
            cost_matrix[idx] = LA_MERGE_COST_INVALID;
        }
        return;
    }
    la_merge_concat_verts_warp(A->hull.verts, nva, B->hull.verts, nvb, s_concat, lane);
    __syncwarp();

    // ---- Step 4: kdop_hull on concat verts ----
    float hvol = 0.0f;
    Mesh merged_hull = kdop_hull_block(
        s_concat, nvab,
        &pool->heap, &pool->scratch,
        &hvol, err,
        /*prune_vol_gap_threshold=*/1e30f,
        /*mesh_vol=*/mesh_vol_sum);

    // Free scratch concat verts (kdop is done reading them).
    if (lane == 0) heap_free(&pool->scratch, (void*)s_concat);
    __syncwarp();

    if (!merged_hull.verts) {
        // kdop failed (e.g. OOM) — cost invalid.
        if (lane == 0) cost_matrix[idx] = LA_MERGE_COST_INVALID;
        return;
    }

    // ---- Step 5: compute rv cost ----
    float gap = fmaxf(hvol - mesh_vol_sum, 0.0f);
    float rv_cost = LA_PART_COST_K_RV * cbrtf((3.0f / (4.0f * pi_f)) * gap);

    if (rv_cost >= threshold) {
        // Over threshold from rv alone — free hull, write cost, done.
        if (lane == 0) {
            heap_free(&pool->heap, (void*)merged_hull.verts);
            cost_matrix[idx] = rv_cost;
        }
        return;
    }

    // ---- Step 6: pass through to hausdorff kernel ----
    // Stash merged hull + vols in pair_state. Refcount = LA_REFCOUNT_HEAP so
    // the hausdorff/apply kernels can free it directly later.
    if (lane == 0) {
        merged_hull.refcount = LA_REFCOUNT_HEAP;
        pair_state[idx].merged_hull  = merged_hull;
        pair_state[idx].hull_vol     = hvol;
        pair_state[idx].mesh_vol_sum = mesh_vol_sum;
        cost_matrix[idx] = rv_cost;   // provisional; la_merge_hausdorff may raise
    }
}

// ============================================================================
// la_merge_hausdorff: <<<P*(P-1)/2, 256>>>
// For each pair where pair_state[idx].merged_hull.verts != NULL, compute the
// shared-plane-filtered Hausdorff distance and update cost to max(rv, hd).
// ============================================================================
extern "C" __global__ void la_merge_hausdorff(
    LaDecompState* decomp,
    int            P,
    float          threshold,
    float*         cost_matrix,
    LaMergePair*   pair_state,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int idx = blockIdx.x;
    int pair_count = P * (P - 1) / 2;
    if (idx >= pair_count) return;

    // Quick reject: pair didn't reach hausdorff stage.
    if (pair_state[idx].merged_hull.verts == NULL) return;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);

    int p1, p2;
    la_merge_pair_from_index(idx, P, p1, p2);
    Part* A = &decomp->parts[p1];
    Part* B = &decomp->parts[p2];

    __shared__ float4 s_plane;
    __shared__ int    s_has_plane;
    __shared__ float* s_concat_v;
    __shared__ int*   s_concat_t;
    __shared__ int    s_nvab;
    __shared__ int    s_ntab;
    __shared__ int    s_local_err;

    if (tid == 0) {
        s_has_plane = 0;
        s_concat_v = NULL;
        s_concat_t = NULL;
        s_nvab = 0; s_ntab = 0;
        s_local_err = 0;
    }
    __syncthreads();

    // ---- Step A: find shared plane using warp 0 over A's hull faces ----
    if (tid < 32) {
        int found = la_merge_find_shared_plane_warp(
            A->hull, B->hull, 1e-3f, lane, &s_plane);
        if (lane == 0) s_has_plane = found;
    }
    __syncthreads();

    // ---- Step B: build filtered concat mesh on scratch ----
    // Output: [v_a|v_b] and [t_a (optionally filtered) | t_b (optionally filtered, offset+=nva)].
    int nva = A->mesh.nv,  nvb = B->mesh.nv;
    int nta = A->mesh.nt,  ntb = B->mesh.nt;
    int nvab = nva + nvb;
    int ntab_cap = nta + ntb;

    if (tid == 0) {
        unsigned int vb = (unsigned int)(nvab * 3 * sizeof(float));
        unsigned int tb = (unsigned int)(ntab_cap * 3 * sizeof(int));
        vb = (vb + 15u) & ~15u;
        tb = (tb + 15u) & ~15u;
        void* ptr = NULL;
        if (heap_alloc(&pool->scratch, (unsigned int)(vb + tb), &ptr) != HEAP_OK || !ptr) {
            s_local_err = KERR_HEAP_OOM;
        } else {
            s_concat_v = (float*)ptr;
            s_concat_t = (int*)((char*)ptr + vb);
        }
    }
    __syncthreads();
    if (s_local_err) {
        if (tid == 0) {
            atomicOr(err, s_local_err);
            // Free merged hull (will not be used downstream).
            heap_free(&pool->heap, (void*)pair_state[idx].merged_hull.verts);
            pair_state[idx].merged_hull.verts = NULL;
            cost_matrix[idx] = LA_MERGE_COST_INVALID;
        }
        return;
    }

    // Copy vertices.
    for (int i = tid; i < nva * 3; i += HD_BLOCK)
        s_concat_v[i] = A->mesh.verts[i];
    for (int i = tid; i < nvb * 3; i += HD_BLOCK)
        s_concat_v[nva * 3 + i] = B->mesh.verts[i];

    __syncthreads();

    // Filter+copy tris, optionally skipping ones with all 3 verts on shared plane.
    __shared__ int s_nt_written;
    if (tid == 0) s_nt_written = 0;
    __syncthreads();

    float pnx = 0.f, pny = 0.f, pnz = 0.f, pd = 0.f;
    if (s_has_plane) {
        pnx = s_plane.x; pny = s_plane.y; pnz = s_plane.z; pd = s_plane.w;
    }
    const float plane_tol = 1e-3f;

    // Triangles from A.
    for (int t = tid; t < nta; t += HD_BLOCK) {
        int i0 = A->mesh.tris[t*3+0];
        int i1 = A->mesh.tris[t*3+1];
        int i2 = A->mesh.tris[t*3+2];
        int keep = 1;
        if (s_has_plane) {
            float x0 = A->mesh.verts[i0*3+0], y0 = A->mesh.verts[i0*3+1], z0 = A->mesh.verts[i0*3+2];
            float x1 = A->mesh.verts[i1*3+0], y1 = A->mesh.verts[i1*3+1], z1 = A->mesh.verts[i1*3+2];
            float x2 = A->mesh.verts[i2*3+0], y2 = A->mesh.verts[i2*3+1], z2 = A->mesh.verts[i2*3+2];
            float s0 = pnx*x0 + pny*y0 + pnz*z0 - pd;
            float s1 = pnx*x1 + pny*y1 + pnz*z1 - pd;
            float s2 = pnx*x2 + pny*y2 + pnz*z2 - pd;
            if (fabsf(s0) <= plane_tol && fabsf(s1) <= plane_tol && fabsf(s2) <= plane_tol)
                keep = 0;
        }
        if (keep) {
            int w = atomicAdd(&s_nt_written, 1);
            s_concat_t[w*3+0] = i0;
            s_concat_t[w*3+1] = i1;
            s_concat_t[w*3+2] = i2;
        }
    }
    // Triangles from B (offset indices by nva).
    for (int t = tid; t < ntb; t += HD_BLOCK) {
        int i0 = B->mesh.tris[t*3+0];
        int i1 = B->mesh.tris[t*3+1];
        int i2 = B->mesh.tris[t*3+2];
        int keep = 1;
        if (s_has_plane) {
            float x0 = B->mesh.verts[i0*3+0], y0 = B->mesh.verts[i0*3+1], z0 = B->mesh.verts[i0*3+2];
            float x1 = B->mesh.verts[i1*3+0], y1 = B->mesh.verts[i1*3+1], z1 = B->mesh.verts[i1*3+2];
            float x2 = B->mesh.verts[i2*3+0], y2 = B->mesh.verts[i2*3+1], z2 = B->mesh.verts[i2*3+2];
            float s0 = pnx*x0 + pny*y0 + pnz*z0 - pd;
            float s1 = pnx*x1 + pny*y1 + pnz*z1 - pd;
            float s2 = pnx*x2 + pny*y2 + pnz*z2 - pd;
            if (fabsf(s0) <= plane_tol && fabsf(s1) <= plane_tol && fabsf(s2) <= plane_tol)
                keep = 0;
        }
        if (keep) {
            int w = atomicAdd(&s_nt_written, 1);
            s_concat_t[w*3+0] = nva + i0;
            s_concat_t[w*3+1] = nva + i1;
            s_concat_t[w*3+2] = nva + i2;
        }
    }
    __syncthreads();

    if (tid == 0) { s_nvab = nvab; s_ntab = s_nt_written; }
    __syncthreads();

    // ---- Step C: build a local Mesh view and run hausdorff_block ----
    __shared__ Mesh s_concat_mesh;
    __shared__ Mesh s_merged_hull_copy;
    if (tid == 0) {
        s_concat_mesh.verts    = s_concat_v;
        s_concat_mesh.tris     = s_concat_t;
        s_concat_mesh.nv       = s_nvab;
        s_concat_mesh.nt       = s_ntab;
        s_concat_mesh.refcount = NULL;
        s_merged_hull_copy     = pair_state[idx].merged_hull;
    }
    __syncthreads();

    float hd = 0.0f;
    if (s_ntab > 0) {
        hd = hausdorff_block(&s_concat_mesh, &s_merged_hull_copy, &pool->scratch, err);
    }

    // ---- Step D: update cost, free scratch ----
    if (tid == 0) {
        heap_free(&pool->scratch, (void*)s_concat_v);
        float rv = cost_matrix[idx];
        float cost = fmaxf(rv, hd);
        cost_matrix[idx] = cost;
        // If the post-hausdorff cost is over threshold, we can free the merged
        // hull now (match kernel will skip, apply kernel won't see it). If it
        // passes threshold, retain hull for la_merge_apply to adopt.
        if (cost >= threshold) {
            heap_free(&pool->heap, (void*)pair_state[idx].merged_hull.verts);
            pair_state[idx].merged_hull.verts    = NULL;
            pair_state[idx].merged_hull.tris     = NULL;
            pair_state[idx].merged_hull.nv       = 0;
            pair_state[idx].merged_hull.nt       = 0;
            pair_state[idx].merged_hull.refcount = NULL;
        }
    }
}

// ============================================================================
// la_merge_match: <<<1, 256>>> — greedy min-cost maximum matching.
// Iteratively:
//   1. Each thread scans cost_matrix for its local min below threshold.
//   2. Block reduction picks global min (tie-break: lower total-degree sum).
//   3. Thread 0 records the matched pair, then all threads atomically set all
//      cost entries involving either endpoint to LA_MERGE_COST_INVALID.
//   4. Repeat until no more edges below threshold.
// Output: matches[k] = packed p1<<16 | p2 for k in [0, n_matches).
// ============================================================================
extern "C" __global__ void la_merge_match(
    int            P,
    float          threshold,
    float*         cost_matrix,     // in/out (invalidated as we go)
    int*           matches,         // out: max P/2 entries
    int*           n_matches,       // out
    int*           err)
{
    (void)err;
    int tid = threadIdx.x;
    int pair_count = P * (P - 1) / 2;

    // Per-iteration block min-reduction buffers.
    __shared__ float s_vals[LA_MERGE_MATCH_BLOCK];
    __shared__ int   s_idxs[LA_MERGE_MATCH_BLOCK];
    __shared__ int   s_n;

    if (tid == 0) s_n = 0;
    __syncthreads();

    while (true) {
        // Find thread-local min over assigned stride of cost matrix.
        float lmin = threshold;
        int   lidx = -1;
        for (int i = tid; i < pair_count; i += LA_MERGE_MATCH_BLOCK) {
            float v = cost_matrix[i];
            if (v < lmin) { lmin = v; lidx = i; }
        }
        s_vals[tid] = lmin;
        s_idxs[tid] = lidx;
        __syncthreads();

        // Tree reduction.
        for (int off = LA_MERGE_MATCH_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) {
                float vo = s_vals[tid + off];
                int   io = s_idxs[tid + off];
                if (vo < s_vals[tid]) {
                    s_vals[tid] = vo;
                    s_idxs[tid] = io;
                }
            }
            __syncthreads();
        }

        int best_idx = s_idxs[0];
        float best_val = s_vals[0];
        if (best_idx < 0 || best_val >= threshold) break;

        // Record match.
        int p1, p2;
        la_merge_pair_from_index(best_idx, P, p1, p2);
        if (tid == 0) {
            matches[s_n] = (p1 << 16) | (p2 & 0xffff);
            s_n++;
            // Invalidate the matched pair immediately (covered below too, but
            // ensures the entry is out for stalled threads reading early).
            cost_matrix[best_idx] = LA_MERGE_COST_INVALID;
        }
        __syncthreads();

        // Invalidate all entries involving p1 or p2 (stride across all threads).
        // Pair (a,b) with a>b is index a*(a-1)/2+b.
        //   - Rows: a in {p1,p2}, b < a  → indices (a*(a-1)/2 + b).
        //   - Cols: a > r, b = r, r in {p1,p2}  → (a*(a-1)/2 + r).
        for (int k = tid; k < P; k += LA_MERGE_MATCH_BLOCK) {
            // entries ending at p1 or p2 as lower index
            if (k < p1) cost_matrix[p1*(p1-1)/2 + k] = LA_MERGE_COST_INVALID;
            if (k < p2) cost_matrix[p2*(p2-1)/2 + k] = LA_MERGE_COST_INVALID;
            if (k > p1) cost_matrix[k*(k-1)/2 + p1]  = LA_MERGE_COST_INVALID;
            if (k > p2) cost_matrix[k*(k-1)/2 + p2]  = LA_MERGE_COST_INVALID;
        }
        __syncthreads();
    }

    if (tid == 0) *n_matches = s_n;
}

// ============================================================================
// la_merge_apply: <<<n_matches, 32>>>
// For each matched pair:
//   - Allocate a merged mesh on heap (concat verts + retargeted tris),
//     refcount = 1.
//   - Adopt the cached merged hull from pair_state (refcount = LA_REFCOUNT_HEAP).
//   - Free old A mesh (refcount--), old A hull (LA_REFCOUNT_HEAP or refcount--).
//   - Tombstone part slot p2 (zero-everything); write merged into p1.
// ============================================================================
extern "C" __global__ void la_merge_apply(
    LaDecompState* decomp,
    int            P,
    int*           matches,
    int            n_matches,
    LaMergePair*   pair_state,
    DevicePool*    pool,
    int*           err)
{
    if (*err) return;

    int m = blockIdx.x;
    if (m >= n_matches) return;
    int tid = threadIdx.x;
    if (tid >= 32) return;
    int lane = tid;

    int packed = matches[m];
    int p1 = (packed >> 16) & 0xffff;
    int p2 = packed & 0xffff;

    // Recompute flat index to read pair_state.
    int idx = la_merge_pair_index(p1, p2);

    Part* A = &decomp->parts[p1];
    Part* B = &decomp->parts[p2];

    int nva = A->mesh.nv, nvb = B->mesh.nv;
    int nta = A->mesh.nt, ntb = B->mesh.nt;
    int nv_new = nva + nvb;
    int nt_new = nta + ntb;

    __shared__ char* s_chunk;
    __shared__ float* s_verts;
    __shared__ int*   s_tris;
    __shared__ int*   s_refcnt;
    __shared__ int    s_local_err;

    if (lane == 0) {
        s_chunk = NULL;
        s_verts = NULL;
        s_tris  = NULL;
        s_refcnt = NULL;
        s_local_err = 0;

        unsigned int vb = (unsigned int)((nv_new * 3u * sizeof(float) + 15u) & ~15u);
        unsigned int tb = (unsigned int)((nt_new * 3u * sizeof(int)   + 15u) & ~15u);
        unsigned int rb = (unsigned int)((sizeof(int) + 15u) & ~15u);
        void* ptr = NULL;
        if (heap_alloc(&pool->heap, vb + tb + rb, &ptr) != HEAP_OK || !ptr) {
            s_local_err = KERR_HEAP_OOM;
        } else {
            s_chunk = (char*)ptr;
            s_verts = (float*)s_chunk;
            s_tris  = (int*)(s_chunk + vb);
            s_refcnt = (int*)(s_chunk + vb + tb);
            *s_refcnt = 1;
        }
    }
    __syncwarp();
    if (s_local_err) {
        if (lane == 0) atomicOr(err, s_local_err);
        return;
    }

    // Copy verts from A then B.
    for (int i = lane; i < nva * 3; i += WARP_SIZE)
        s_verts[i] = A->mesh.verts[i];
    for (int i = lane; i < nvb * 3; i += WARP_SIZE)
        s_verts[nva * 3 + i] = B->mesh.verts[i];
    // Tris from A unchanged; tris from B offset by nva.
    for (int i = lane; i < nta * 3; i += WARP_SIZE)
        s_tris[i] = A->mesh.tris[i];
    for (int i = lane; i < ntb * 3; i += WARP_SIZE)
        s_tris[nta * 3 + i] = B->mesh.tris[i] + nva;
    __syncwarp();

    if (lane == 0) {
        // Adopt cached merged hull.
        Mesh merged_hull = pair_state[idx].merged_hull;
        float new_hull_vol = pair_state[idx].hull_vol;
        float new_mesh_vol = A->mesh_vol + B->mesh_vol;

        // Free old A mesh/hull.
        if (A->mesh.refcount) {
            int old = atomicAdd(A->mesh.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)A->mesh.verts);
        }
        if (A->hull.refcount == LA_REFCOUNT_HEAP) {
            heap_free(&pool->heap, (void*)A->hull.verts);
        } else if (A->hull.refcount) {
            int old = atomicAdd(A->hull.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)A->hull.verts);
        }

        // Free old B mesh/hull.
        if (B->mesh.refcount) {
            int old = atomicAdd(B->mesh.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)B->mesh.verts);
        }
        if (B->hull.refcount == LA_REFCOUNT_HEAP) {
            heap_free(&pool->heap, (void*)B->hull.verts);
        } else if (B->hull.refcount) {
            int old = atomicAdd(B->hull.refcount, -1);
            if (old == 1) heap_free(&pool->heap, (void*)B->hull.verts);
        }

        // Write merged into slot p1.
        A->mesh.verts    = s_verts;
        A->mesh.tris     = s_tris;
        A->mesh.nv       = nv_new;
        A->mesh.nt       = nt_new;
        A->mesh.refcount = s_refcnt;
        A->hull          = merged_hull;   // refcount = LA_REFCOUNT_HEAP already
        A->mesh_vol      = new_mesh_vol;
        A->hull_vol      = new_hull_vol;
        A->hausdorff     = 0.0f;

        // Tombstone slot p2.
        B->mesh.verts = NULL; B->mesh.tris = NULL;
        B->mesh.nv = 0; B->mesh.nt = 0; B->mesh.refcount = NULL;
        B->hull.verts = NULL; B->hull.tris = NULL;
        B->hull.nv = 0; B->hull.nt = 0; B->hull.refcount = NULL;
        B->mesh_vol = 0.0f; B->hull_vol = 0.0f; B->hausdorff = 0.0f;

        // Clear pair_state so subsequent sweeps don't double-free.
        pair_state[idx].merged_hull.verts    = NULL;
        pair_state[idx].merged_hull.tris     = NULL;
        pair_state[idx].merged_hull.nv       = 0;
        pair_state[idx].merged_hull.nt       = 0;
        pair_state[idx].merged_hull.refcount = NULL;
    }
}

// ============================================================================
// la_merge_free_unused: <<<P*(P-1)/2, 32>>>
// After la_merge_match, the cached merged hulls for pairs that were NOT
// chosen must be freed (they're still allocated from la_merge_cost_matrix).
// ============================================================================
extern "C" __global__ void la_merge_free_unused(
    int          P,
    LaMergePair* pair_state,
    DevicePool*  pool)
{
    int idx = blockIdx.x;
    int pair_count = P * (P - 1) / 2;
    if (idx >= pair_count) return;
    if (threadIdx.x != 0) return;

    Mesh h = pair_state[idx].merged_hull;
    if (h.verts != NULL && h.refcount == LA_REFCOUNT_HEAP) {
        heap_free(&pool->heap, (void*)h.verts);
        pair_state[idx].merged_hull.verts = NULL;
        pair_state[idx].merged_hull.refcount = NULL;
    }
}

// ============================================================================
// la_merge_compact: <<<1, 128>>>
// Compact decomp->parts[] in place to remove tombstones (mesh.verts==NULL).
// Single block, serial on thread 0 — nparts is small.
// ============================================================================
extern "C" __global__ void la_merge_compact(LaDecompState* decomp)
{
    if (threadIdx.x != 0) return;
    int np = decomp->nparts;
    int w = 0;
    for (int r = 0; r < np; r++) {
        Part* p = &decomp->parts[r];
        if (p->mesh.verts == NULL) continue;
        if (r != w) decomp->parts[w] = *p;
        w++;
    }
    // Zero trailing tombstones.
    for (int r = w; r < np; r++) {
        Part* p = &decomp->parts[r];
        p->mesh.verts = NULL; p->mesh.tris = NULL;
        p->mesh.nv = 0; p->mesh.nt = 0; p->mesh.refcount = NULL;
        p->hull.verts = NULL; p->hull.tris = NULL;
        p->hull.nv = 0; p->hull.nt = 0; p->hull.refcount = NULL;
        p->mesh_vol = 0.0f; p->hull_vol = 0.0f; p->hausdorff = 0.0f;
    }
    decomp->nparts = w;
}
