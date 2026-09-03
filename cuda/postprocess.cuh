// postprocess.cuh — GPU post-processing kernels (connected-components decomposition, etc.).
//
// One block (128 threads) splits a Part into its connected mesh components.
// Uses lock-free rank-based union-find over vertex adjacency.
//
// Algorithm:
//   1  Read input dims (thread 0), early-return if nv==0 or nt==0.
//   2  Scratch alloc (thread 0): parents[nv] + vert_comp[nv] + vert_local_idx[nv]
//                              + comp_nv[DC_MAX_OUT] + comp_nt[DC_MAX_OUT]
//                              + is_inner[DC_MAX_OUT].
//      (DC_MAX_OUT=4096 makes shared-memory arrays infeasible — counters live on
//       the scratch heap alongside the per-vertex tables.)
//   3  Init parents[v] = pack(0,v)   (all threads).
//   4  Union edges from triangles     (all threads).
//   5  Assign sequential component IDs: parallel root walk + clear (all threads),
//      then sequential ID assignment (thread 0).
//   6  Zero comp_nv / comp_nt         (all threads).
//   7  Count verts/tris per component (all threads, atomicAdd).
//   8  Alloc output meshes + reset comp_nv/comp_nt to zero for scatter (thread 0).
//   9  Scatter vertices with atomicAdd index (all threads).
//  10  Re-zero comp_nt                (all threads).
//  11  Scatter triangles with remapped indices (all threads).
// 11b  Compute mesh_vol = |signed_vol| per output component; flag negative-svol
//      components as unreachable interiors (s_is_inner).
// 11c  Compact: discard inner-shell components (only when mixed with outers,
//      keeps at least one component).
//  12  Free input mesh + scratch (thread 0).
//
#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "mesh_volume.cuh"

// Block size for decompose_components_block.
#define DC_BLOCK 128

// Maximum output components (capacity of output_parts[]). Caller must
// supply an output_parts buffer sized >= DC_MAX_OUT (heap-allocated, since
// 4096 * sizeof(Part) far exceeds the 48 KB static shared-memory budget).
#define DC_MAX_OUT 4096

// Pack rank+parent into one unsigned int.
//   Bits [31..DC_RANK_SHIFT] = rank,  bits [DC_RANK_SHIFT-1..0] = parent id.
#define DC_RANK_SHIFT 24
#define DC_ID_MASK    ((1u << DC_RANK_SHIFT) - 1)

// Alignment helper (matches PC_ALIGN16 convention).
#define DC_ALIGN16(x) (((x) + 15) & ~15)

// Error codes: see error_codes.cuh (KERR_DC_*)

// LA_REFCOUNT_HEAP sentinel — hull allocated by kdop_hull_block, freed directly.
#ifndef LA_REFCOUNT_HEAP
#define LA_REFCOUNT_HEAP ((int*)1)
#endif

// Zero all fields of a Part (local version; equivalent to dc_zero_part in plane_cut.cuh).
__device__ inline void dc_zero_part(Part* __restrict__ p) {
    p->mesh.verts = NULL; p->mesh.tris = NULL; p->mesh.nv = 0; p->mesh.nt = 0; p->mesh.refcount = NULL;
    p->hull.verts = NULL; p->hull.tris = NULL; p->hull.nv = 0; p->hull.nt = 0; p->hull.refcount = NULL;
    p->mesh_vol = 0.0f; p->hull_vol = 0.0f; p->hausdorff = 0.0f;
    p->cc_id = 0;
}

// ============================================================================
// Pack / unpack helpers
// ============================================================================

__device__ inline unsigned int dc_pack(unsigned int rank, unsigned int id) {
    return (rank << DC_RANK_SHIFT) | (id & DC_ID_MASK);
}

__device__ inline unsigned int dc_getId(unsigned int packed) {
    return packed & DC_ID_MASK;
}

__device__ inline unsigned int dc_getRank(unsigned int packed) {
    return packed >> DC_RANK_SHIFT;
}

// ============================================================================
// dc_find — read-only root walk, NO path compression
// ============================================================================

__device__ inline unsigned int dc_find(unsigned int* parents, unsigned int x) {
    while (true) {
        unsigned int px = dc_getId(parents[x]);
        if (px == x) return x;
        x = px;
    }
}

// ============================================================================
// dc_union — lock-free rank-based union
// ============================================================================

__device__ inline void dc_union(unsigned int* parents, unsigned int x, unsigned int y) {
    if (x == y) return;
    while (true) {
        unsigned int vx = parents[x];
        unsigned int vy = parents[y];
        unsigned int px = dc_getId(vx), py = dc_getId(vy);
        unsigned int rx = dc_getRank(vx), ry = dc_getRank(vy);
        if (px == py) return;  // same root already
        // Only act when both x and y are currently roots.
        if (px == x && py == y) {
            // Rank decides winner; lower index breaks ties.
            unsigned int winner, loser, vloser;
            if (rx > ry || (rx == ry && x < y)) {
                winner = x; loser = y; vloser = vy;
            } else {
                winner = y; loser = x; vloser = vx;
            }
            unsigned int new_val = dc_pack(dc_getRank(vloser), winner);
            if (atomicCAS(&parents[loser], vloser, new_val) == vloser) {
                if (rx == ry) {
                    // Only bump rank if winner is still a root (self-pointing).
                    // Without the id check, a concurrent merge of winner into
                    // another node Z could leave parents[winner] = pack(r, Z);
                    // our CAS would then write pack(r+1, winner), re-rooting
                    // winner and silently undoing the other merge.
                    unsigned int vw = parents[winner];
                    if (dc_getId(vw) == winner)
                        atomicCAS(&parents[winner], vw, dc_pack(dc_getRank(vw) + 1, winner));
                }
                return;
            }
            // CAS failed — someone else modified parents[loser]; retry from roots.
        }
        x = px; y = py;
    }
}

// ============================================================================
// decompose_components_block — main block function
// ============================================================================

__device__ inline int decompose_components_block(
    Part*       input_part,
    Part*       output_parts,   // caller-provided array [DC_MAX_OUT]
    DeviceHeap* heap,           // main heap for output mesh allocations
    DeviceHeap* scratch,        // scratch heap for temporaries
    int*        err)
{
    int tid = (int)threadIdx.x;

    // Shared pointers broadcast from thread 0 to the block.
    // comp_nv/comp_nt/is_inner live on the scratch heap (DC_MAX_OUT=4096 →
    // ~36 KB per kernel call; static shared-memory was overflowing).
    __shared__ int    s_nv, s_nt;
    __shared__ float* s_in_verts;
    __shared__ int*   s_in_tris;
    __shared__ unsigned int* s_scratch_base;
    __shared__ unsigned int* s_parents;
    __shared__ unsigned int* s_vert_comp;
    __shared__ unsigned int* s_vert_local_idx;
    __shared__ unsigned int* s_comp_nv;
    __shared__ unsigned int* s_comp_nt;
    __shared__ char*         s_is_inner;
    __shared__ int    s_n_components;

    // -------------------------------------------------------------------------
    // Phase 1: Read input dims (thread 0).
    // -------------------------------------------------------------------------
    if (tid == 0) {
        s_nv       = input_part->mesh.nv;
        s_nt       = input_part->mesh.nt;
        s_in_verts = input_part->mesh.verts;
        s_in_tris  = input_part->mesh.tris;
    }
    __syncthreads();

    int nv = s_nv, nt = s_nt;

    // Early return: trivially 1 component.
    if (nv == 0 || nt == 0) {
        if (tid == 0) {
            output_parts[0] = *input_part;
        }
        __syncthreads();
        return 1;
    }

    // -------------------------------------------------------------------------
    // Phase 2: Single scratch alloc (thread 0).
    // Layout (16-byte aligned segments):
    //   [parents(nv) | vert_comp(nv) | vert_local_idx(nv)
    //    | comp_nv(DC_MAX_OUT) | comp_nt(DC_MAX_OUT) | is_inner(DC_MAX_OUT)]
    // -------------------------------------------------------------------------
    if (tid == 0) {
        unsigned int verts_bytes  = DC_ALIGN16((unsigned int)nv * 3u * sizeof(unsigned int));
        unsigned int counts_bytes = DC_ALIGN16(2u * (unsigned int)DC_MAX_OUT * sizeof(unsigned int));
        unsigned int inner_bytes  = DC_ALIGN16((unsigned int)DC_MAX_OUT * sizeof(char));
        unsigned int total = verts_bytes + counts_bytes + inner_bytes;
        void* raw = NULL;
        int rc = heap_alloc(scratch, total, &raw);
        if (rc != HEAP_OK || !raw) {
            atomicOr(err, KERR_DC_OOM);
            s_scratch_base = NULL;
        } else {
            unsigned char* base_b = (unsigned char*)raw;
            unsigned int*  base_u = (unsigned int*)raw;
            s_scratch_base    = base_u;
            s_parents         = base_u;
            s_vert_comp       = base_u + nv;
            s_vert_local_idx  = base_u + 2 * nv;
            s_comp_nv         = (unsigned int*)(base_b + verts_bytes);
            s_comp_nt         = s_comp_nv + DC_MAX_OUT;
            s_is_inner        = (char*)(base_b + verts_bytes + counts_bytes);
        }
    }
    __syncthreads();

    if (!s_scratch_base) return 0;  // OOM

    // -------------------------------------------------------------------------
    // Phase 3: Init parents[v] = pack(0, v).
    // -------------------------------------------------------------------------
    for (int v = tid; v < nv; v += DC_BLOCK)
        s_parents[v] = dc_pack(0u, (unsigned int)v);
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 4: Union over triangles.
    // -------------------------------------------------------------------------
    for (int t = tid; t < nt; t += DC_BLOCK) {
        int i0 = s_in_tris[t * 3 + 0];
        int i1 = s_in_tris[t * 3 + 1];
        int i2 = s_in_tris[t * 3 + 2];
        // [BUG] Debug: catch negative or OOB triangle vertex indices
        if (i0 < 0 || i0 >= nv || i1 < 0 || i1 >= nv || i2 < 0 || i2 >= nv)
            DPRINTF("[BUG] phase4 OOB tri idx: t=%d i0=%d i1=%d i2=%d nv=%d blk=%d tid=%d\n",
                    t, i0, i1, i2, nv, blockIdx.x, tid);
        dc_union(s_parents, (unsigned int)i0, (unsigned int)i1);
        dc_union(s_parents, (unsigned int)i0, (unsigned int)i2);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: Assign sequential component IDs.
    //   A) All threads: vert_comp[v] = dc_find(parents, v)
    //   B) All threads: parents[v] = ~0u  (must complete after A — dc_find
    //      reads parents[], so clearing it concurrently would corrupt the walk)
    //   C) Thread 0: assign sequential IDs using vert_comp[] + parents[] map
    // -------------------------------------------------------------------------
    {
        // Sub-pass A: root walk (reads parents[]).
        for (int v = tid; v < nv; v += DC_BLOCK)
            s_vert_comp[v] = dc_find(s_parents, (unsigned int)v);
        __syncthreads();

        // Sub-pass B: clear parents[] for root→id map (writes parents[]).
        for (int v = tid; v < nv; v += DC_BLOCK)
            s_parents[v] = ~0u;
        __syncthreads();

        // Sub-pass C: sequential ID assignment (thread 0).
        if (tid == 0) {
            int n_comp = 0;
            for (int v = 0; v < nv; v++) {
                unsigned int root = s_vert_comp[v];
                if (s_parents[root] == ~0u) {
                    if (n_comp < DC_MAX_OUT)
                        s_parents[root] = (unsigned int)(n_comp++);
                    else
                        s_parents[root] = (unsigned int)(DC_MAX_OUT - 1);  // clamp: merge into last slot
                }
                s_vert_comp[v] = s_parents[root];
            }

            if (n_comp > DC_MAX_OUT)
                n_comp = DC_MAX_OUT;  // actual distinct output count

            s_n_components = n_comp;

            // Early-exit if already 1 component.
            if (n_comp <= 1) {
                output_parts[0] = *input_part;
            }
        }
    }
    __syncthreads();

    int n_comp = s_n_components;

    if (n_comp <= 1) {
        // Free scratch on the trivial path (thread 0).
        if (tid == 0)
            heap_free(scratch, (void*)s_scratch_base);
        __syncthreads();
        return (n_comp == 1) ? 1 : 0;
    }

    // -------------------------------------------------------------------------
    // Phase 6: Zero comp_nv / comp_nt counters (shared memory).
    // -------------------------------------------------------------------------
    for (int c = tid; c < DC_MAX_OUT; c += DC_BLOCK) {
        s_comp_nv[c] = 0u;
        s_comp_nt[c] = 0u;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 7: Count verts and tris per component.
    // -------------------------------------------------------------------------
    for (int v = tid; v < nv; v += DC_BLOCK) {
        unsigned int cv = s_vert_comp[v];
        atomicAdd(&s_comp_nv[cv], 1u);
    }

    for (int t = tid; t < nt; t += DC_BLOCK) {
        int i0 = s_in_tris[t * 3];
        unsigned int cv = s_vert_comp[(unsigned int)i0];
        // [BUG] Debug: catch negative tri idx
        if (i0 < 0 || i0 >= nv)
            DPRINTF("[BUG] phase7 tri idx OOB: t=%d i0=%d nv=%d blk=%d tid=%d\n",
                    t, i0, nv, blockIdx.x, tid);
        atomicAdd(&s_comp_nt[cv], 1u);
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 8: Alloc output meshes (thread 0); reset comp_nv/comp_nt to 0 for
    //          scatter counters.
    // -------------------------------------------------------------------------
    if (tid == 0) {
        for (int c = 0; c < n_comp; c++) {
            unsigned int cnv = s_comp_nv[c];
            unsigned int cnt = s_comp_nt[c];

            unsigned int vb = DC_ALIGN16(cnv * 3u * sizeof(float));
            unsigned int tb = DC_ALIGN16(cnt * 3u * sizeof(int));
            unsigned int rb = DC_ALIGN16(sizeof(int));
            unsigned int sz = vb + tb + rb;
            if (!sz) sz = 1u;

            void* chunk = NULL;
            int rc = heap_alloc(heap, sz, &chunk);
            if (rc != HEAP_OK || !chunk) {
                atomicOr(err, KERR_DC_OOM);
                // Leave remaining parts zeroed; caller checks err.
                for (int cc = c; cc < n_comp; cc++)
                    dc_zero_part(&output_parts[cc]);
                n_comp = c;  // tell scatter loops how many are valid
                s_n_components = c;
                break;
            }

            char* base_ptr = (char*)chunk;
            float* verts_ptr   = (float*)(base_ptr);
            int*   tris_ptr    = (int*)(base_ptr + vb);
            int*   refcnt_ptr  = (int*)(base_ptr + vb + tb);

            *refcnt_ptr = 1;

            dc_zero_part(&output_parts[c]);
            output_parts[c].mesh.verts    = verts_ptr;
            output_parts[c].mesh.tris     = tris_ptr;
            output_parts[c].mesh.refcount = refcnt_ptr;
            output_parts[c].mesh.nv       = (int)cnv;
            output_parts[c].mesh.nt       = (int)cnt;
            // Reset hausdorff to 0 (sentinel for "not computed") — components
            // have different meshes than the input so the cached value is stale.
            output_parts[c].hausdorff     = 0.0f;

            // Reset counters to zero — will be reused as scatter cursors.
            s_comp_nv[c] = 0u;
            s_comp_nt[c] = 0u;
        }
    }
    __syncthreads();

    n_comp = s_n_components;  // may have been truncated on OOM

    if (n_comp == 0) {
        if (tid == 0)
            heap_free(scratch, (void*)s_scratch_base);
        __syncthreads();
        return 0;
    }

    // -------------------------------------------------------------------------
    // Phase 9: Build vertex remap and scatter verts.
    // -------------------------------------------------------------------------
    for (int v = tid; v < nv; v += DC_BLOCK) {
        int c = (int)s_vert_comp[v];
        if (c >= n_comp) continue;  // guard against truncated alloc
        int local_idx = (int)atomicAdd(&s_comp_nv[c], 1u);
        s_vert_local_idx[v] = (unsigned int)local_idx;
        // [BUG] Debug: catch OOB vertex scatter
        if (local_idx >= output_parts[c].mesh.nv)
            DPRINTF("[BUG] phase9 vert scatter OOB: v=%d c=%d local_idx=%d mesh_nv=%d blk=%d tid=%d\n",
                    v, c, local_idx, output_parts[c].mesh.nv, blockIdx.x, tid);
        float* dst = output_parts[c].mesh.verts;
        dst[local_idx * 3 + 0] = s_in_verts[v * 3 + 0];
        dst[local_idx * 3 + 1] = s_in_verts[v * 3 + 1];
        dst[local_idx * 3 + 2] = s_in_verts[v * 3 + 2];
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 10: Re-zero comp_nt for triangle scatter cursor.
    // -------------------------------------------------------------------------
    for (int c = tid; c < DC_MAX_OUT; c += DC_BLOCK)
        s_comp_nt[c] = 0u;
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 11: Scatter triangles with remapped vertex indices.
    // -------------------------------------------------------------------------
    for (int t = tid; t < nt; t += DC_BLOCK) {
        int i0 = s_in_tris[t * 3 + 0];
        int i1 = s_in_tris[t * 3 + 1];
        int i2 = s_in_tris[t * 3 + 2];
        // [BUG] Debug: catch negative or OOB tri indices before vert_comp lookup
        if (i0 < 0 || i0 >= nv || i1 < 0 || i1 >= nv || i2 < 0 || i2 >= nv)
            DPRINTF("[BUG] phase11 OOB tri idx: t=%d i0=%d i1=%d i2=%d nv=%d blk=%d tid=%d\n",
                    t, i0, i1, i2, nv, blockIdx.x, tid);
        int c  = (int)s_vert_comp[i0];
        if (c >= n_comp) continue;
        // [BUG] Debug: catch cross-component triangle (verts in different components)
        int c1 = (int)s_vert_comp[i1];
        int c2 = (int)s_vert_comp[i2];
        if (c1 != c || c2 != c)
            DPRINTF("[BUG] phase11 cross-comp tri: t=%d i0=%d(i0_c=%d) i1=%d(i1_c=%d) i2=%d(i2_c=%d) blk=%d tid=%d\n",
                    t, i0, c, i1, c1, i2, c2, blockIdx.x, tid);
        int local_t = (int)atomicAdd(&s_comp_nt[c], 1u);
        int* dst = output_parts[c].mesh.tris;
        // [BUG] Debug: catch OOB triangle scatter index
        if (local_t >= output_parts[c].mesh.nt)
            DPRINTF("[BUG] phase11 tri scatter OOB: t=%d c=%d local_t=%d mesh_nt=%d blk=%d tid=%d\n",
                    t, c, local_t, output_parts[c].mesh.nt, blockIdx.x, tid);
        dst[local_t * 3 + 0] = (int)s_vert_local_idx[i0];
        dst[local_t * 3 + 1] = (int)s_vert_local_idx[i1];
        dst[local_t * 3 + 2] = (int)s_vert_local_idx[i2];
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 11b: Compute mesh_vol per output component and flag negative-svol
    // (unreachable interiors). 1 warp each; DC_BLOCK=128 → 4 warps stride.
    // mesh_signed_volume_warp now anchors at v[0] so the f32 sign is reliable
    // (without that, ~30% of CCs in scenes like bistro had random signs from
    // catastrophic cancellation and we'd discard valid pieces).
    // -------------------------------------------------------------------------
    {
        for (int c = tid; c < DC_MAX_OUT; c += DC_BLOCK)
            s_is_inner[c] = 0;
        __syncthreads();

        int warp_id = tid / WARP_SIZE;
        int lane    = tid & (WARP_SIZE - 1);
        for (int c = warp_id; c < n_comp; c += (DC_BLOCK / WARP_SIZE)) {
            float svol = mesh_signed_volume_warp(&output_parts[c].mesh, lane);
            if (lane == 0) {
                output_parts[c].mesh_vol = fabsf(svol);
                if (svol < 0.0f)
                    s_is_inner[c] = 1;
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------------
        // Phase 11c: Compact, removing unreachable-interior components. Only
        // when n_comp > 1 and the inner/outer split is mixed (if all are
        // inner — unusual — keep them all so the caller still gets data).
        // -------------------------------------------------------------------------
        if (n_comp > 1) {
            int n_inner = 0;
            for (int c = 0; c < n_comp; c++)
                n_inner += s_is_inner[c];

            if (n_inner > 0 && n_inner < n_comp) {
                if (tid == 0) {
                    int keep = 0;
                    for (int c = 0; c < n_comp; c++) {
                        if (s_is_inner[c]) {
                            if (output_parts[c].mesh.refcount) {
                                int old = atomicAdd(output_parts[c].mesh.refcount, -1);
                                if (old == 1)
                                    heap_free(heap, (void*)output_parts[c].mesh.verts);
                            }
                        } else {
                            if (keep != c)
                                output_parts[keep] = output_parts[c];
                            keep++;
                        }
                    }
                    for (int c = keep; c < n_comp; c++)
                        dc_zero_part(&output_parts[c]);
                    s_n_components = keep;
                }
                __syncthreads();
                n_comp = s_n_components;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 12: Free input mesh + scratch (thread 0).
    // -------------------------------------------------------------------------
    if (tid == 0) {
        // Free input mesh.
        if (input_part->mesh.refcount) {
            int old = atomicAdd(input_part->mesh.refcount, -1);
            if (old == 1)
                heap_free(heap, (void*)input_part->mesh.verts);
        }

        // Free input hull.
        if (input_part->hull.refcount == LA_REFCOUNT_HEAP) {
            heap_free(heap, (void*)input_part->hull.verts);
        } else if (input_part->hull.refcount) {
            int old = atomicAdd(input_part->hull.refcount, -1);
            if (old == 1)
                heap_free(heap, (void*)input_part->hull.verts);
        }

        // Free scratch block.
        heap_free(scratch, (void*)s_scratch_base);
    }
    __syncthreads();

    return n_comp;
}
