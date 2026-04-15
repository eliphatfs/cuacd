// postprocess.cuh — GPU post-processing kernels (connected-components decomposition, etc.).
//
// One block (128 threads) splits a Part into its connected mesh components.
// Uses lock-free rank-based union-find over vertex adjacency.
//
// Algorithm:
//   1  Read input dims (thread 0), early-return if nv==0 or nt==0.
//   2  Scratch alloc: parents[nv] + vert_comp[nv] + vert_local_idx[nv]  (thread 0).
//                     comp_nv/comp_nt are shared memory (DC_MAX_OUT each).
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
// 11b  Compute mesh_vol per output component (1 warp each, 4 warps stride).
//  12  Free input mesh + scratch (thread 0).
//
#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "mesh_volume.cuh"

// Block size for decompose_components_block.
#define DC_BLOCK 128

// Maximum output components (capacity of output_parts[]).
#define DC_MAX_OUT 32

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
    __shared__ int    s_nv, s_nt;
    __shared__ float* s_in_verts;
    __shared__ int*   s_in_tris;
    __shared__ unsigned int* s_scratch_base;
    __shared__ unsigned int* s_parents;
    __shared__ unsigned int* s_vert_comp;
    __shared__ unsigned int* s_vert_local_idx;
    __shared__ unsigned int s_comp_nv[DC_MAX_OUT];
    __shared__ unsigned int s_comp_nt[DC_MAX_OUT];
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
    // Layout: [parents(nv) | vert_comp(nv) | vert_local_idx(nv)]
    // comp_nv/comp_nt are in shared memory (DC_MAX_OUT entries each).
    // -------------------------------------------------------------------------
    if (tid == 0) {
        unsigned int total = (unsigned int)(3 * nv) * sizeof(unsigned int);
        void* raw = NULL;
        int rc = heap_alloc(scratch, total, &raw);
        if (rc != HEAP_OK || !raw) {
            atomicOr(err, KERR_DC_OOM);
            s_scratch_base = NULL;
        } else {
            unsigned int* base = (unsigned int*)raw;
            s_scratch_base    = base;
            s_parents         = base;
            s_vert_comp       = base + nv;
            s_vert_local_idx  = base + 2 * nv;
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
        dc_union(s_parents, (unsigned int)i0, (unsigned int)i1);
        dc_union(s_parents, (unsigned int)i0, (unsigned int)i2);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: Assign sequential component IDs.
    // First two sub-passes run in parallel across all threads:
    //   A) vert_comp[v] = dc_find(parents, v)  — root walk per vertex
    //   B) parents[v] = ~0u                     — clear for root→id map
    // Then thread 0 assigns sequential IDs (depends on both A and B).
    // -------------------------------------------------------------------------
    {
        // Sub-pass A + B in parallel (one pass over vertices).
        for (int v = tid; v < nv; v += DC_BLOCK) {
            s_vert_comp[v] = dc_find(s_parents, (unsigned int)v);
            s_parents[v] = ~0u;
        }
        __syncthreads();

        // Sub-pass C: sequential ID assignment (thread 0).
        if (tid == 0) {
            int n_comp = 0;
            for (int v = 0; v < nv; v++) {
                unsigned int root = s_vert_comp[v];
                if (s_parents[root] == ~0u)
                    s_parents[root] = (unsigned int)(n_comp++);
                s_vert_comp[v] = s_parents[root];
            }

            s_n_components = n_comp;

            // Early-exit if already 1 component or too many components.
            if (n_comp <= 1) {
                output_parts[0] = *input_part;
            } else if (n_comp > DC_MAX_OUT) {
                atomicOr(err, KERR_DC_OOM);
                s_n_components = -1;  // signal error
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
        return (n_comp == 1) ? 1 : 0;  // 0 if error was set above (-1)
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
    for (int v = tid; v < nv; v += DC_BLOCK)
        atomicAdd(&s_comp_nv[s_vert_comp[v]], 1u);

    for (int t = tid; t < nt; t += DC_BLOCK)
        atomicAdd(&s_comp_nt[s_vert_comp[(unsigned int)s_in_tris[t * 3]]], 1u);

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
            output_parts[c].hausdorff     = input_part->hausdorff;

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
        int c  = (int)s_vert_comp[i0];
        if (c >= n_comp) continue;
        int local_t = (int)atomicAdd(&s_comp_nt[c], 1u);
        int* dst = output_parts[c].mesh.tris;
        dst[local_t * 3 + 0] = (int)s_vert_local_idx[i0];
        dst[local_t * 3 + 1] = (int)s_vert_local_idx[i1];
        dst[local_t * 3 + 2] = (int)s_vert_local_idx[i2];
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 11b: Compute signed mesh_vol for each output component (1 warp each).
    // DC_BLOCK=128 → 4 warps stride over n_comp components.
    // Components with negative signed volume are inner shells (cavity surfaces)
    // and are discarded.  At least one component is always kept.
    // -------------------------------------------------------------------------
    {
        // Use a shared flag array to mark inner shells without mutating Part.
        __shared__ char s_is_inner[DC_MAX_OUT];
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

        // Compact: remove inner-shell components (only if n_comp > 1 and
        // at least one is an outer shell).
        if (n_comp > 1) {
            int n_inner = 0;
            for (int c = 0; c < n_comp; c++)
                n_inner += s_is_inner[c];

            if (n_inner > 0 && n_inner < n_comp) {
                // Some inner shells, some outer — compact
                if (tid == 0) {
                    int keep = 0;
                    for (int c = 0; c < n_comp; c++) {
                        if (s_is_inner[c]) {
                            // Free discarded component's mesh allocation
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
                    // Zero out stale slots
                    for (int c = keep; c < n_comp; c++)
                        dc_zero_part(&output_parts[c]);
                    n_comp = keep;
                    s_n_components = keep;
                }
                __syncthreads();
                n_comp = s_n_components;
            }
            // If all components are inner shells (n_inner == n_comp), keep all.
            // This is unusual but preserves the original data intact.
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
