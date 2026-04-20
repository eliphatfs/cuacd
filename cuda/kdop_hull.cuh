// kdop_hull.cuh — Convex hull via extreme-point prefilter + exact D&C hull.
//
// Single-warp (KDOP_BLOCK = 32) algorithm, 1 block per mesh.
//
// Fast path (nv <= 1024): run hull_dandc_warp_mesh directly.
//
// Main path (nv > 1024):
//   1. Find 40 max + 40 min extreme vertices using KDOP_AXES (warp argmax/argmin).
//   2. Collect up to 80 extreme vertices (hull_dandc handles deduplication).
//   3. Run hull_dandc_warp_mesh on extreme vertices → rough inner hull.
//   4. Filter original vertices: keep any point strictly outside at least one face
//      of the rough hull (ballot/popcount collect).
//   5. Append extreme-hull vertices to the filtered set (ensures boundary coverage).
//   6. Run hull_dandc_warp_mesh on filtered set → final hull.
//   7. Compute volume.

#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "hull_dandc.cuh"
#include "mesh_volume.cuh"

#define KDOP_BLOCK   32
#define KDOP_N_AXES  40
#define KDOP_MAX_EXTREMES (KDOP_N_AXES * 2)   // 80

// 40 unique face normals from level-1 icosphere (80 faces, antipodal pairs merged).
// Defined in kdop_const.cu; __constant__ gives broadcast-cached reads for uniform warp access.
extern __constant__ float KDOP_AXES[KDOP_N_AXES][3];

// kdop_hull_block: convex hull of a point cloud.
//
// All KDOP_BLOCK (= 32) threads in the block must call with identical arguments.
// Returns Mesh (heap-allocated) and volume via out_volume.
// On error, returns {NULL,NULL,0,0} and sets *kernel_error.
//
// Optional rough-hull LB pruning (slow path only): pass mesh_vol and
// prune_vol_gap_threshold (an upper bound on the allowable
// max(hull_vol - mesh_vol, 0)). After the rough inner hull is built
// (step 3 on extreme points), its volume V_rough satisfies V_rough ≤
// V_exact (rough hull ⊂ exact hull), so max(V_rough - mesh_vol, 0) is a
// lower bound on the exact volume gap. If that LB exceeds the threshold,
// the exact D&C hull is skipped; *out_volume receives V_rough and the
// returned Mesh has verts=NULL. Caller writing hull_vol=V_rough yields
// la_part_cost_rv = 0.3 * cbrt(3/(4π) * (V_rough - mesh_vol)). Pass
// prune_vol_gap_threshold = +inf (1e30f) to disable; mesh_vol is unused
// when disabled.
__device__ __forceinline__ Mesh kdop_hull_block(
    const float* verts, int nv,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    float*       out_volume,
    int*         kernel_error,
    float        prune_vol_gap_threshold = 1e30f,
    float        mesh_vol = 0.0f)
{
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Shared state
    __shared__ float* s_extreme_pts;  // extreme pts (scratch)
    __shared__ int    s_n_extreme;
    __shared__ Mesh   s_ext_hull;     // rough hull of extreme pts (scratch)
    __shared__ float* s_planes;        // precomputed oriented planes (scratch): nt*4 floats (nx,ny,nz,d)
    __shared__ float* s_filtered;     // filtered verts (scratch)
    __shared__ int    s_n_filtered;
    __shared__ Mesh   s_result;
    __shared__ float  s_volume;
    __shared__ int    s_local_err;

    if (lane == 0) {
        s_result.verts = NULL; s_result.tris = NULL;
        s_result.nv = 0; s_result.nt = 0; s_result.refcount = NULL;
        s_ext_hull.verts = NULL; s_ext_hull.tris = NULL;
        s_ext_hull.nv = 0; s_ext_hull.nt = 0; s_ext_hull.refcount = NULL;
        s_extreme_pts = NULL;
        s_planes = NULL;
        s_filtered = NULL;
        s_n_extreme = 0; s_n_filtered = 0;
        s_volume = 0.0f;
        s_local_err = 0;
    }
    __syncwarp();

#ifdef COACD_BEAM_DEBUG
    long long t_start = clock64();
    long long t_ep = t_start, t_ephull = t_start, t_filtered = t_start, t_final = t_start;
#endif

    if (nv < 4) { *out_volume = 0.0f; return s_result; }

    // ========================================================================
    // Fast path: small mesh → exact D&C hull directly
    // ========================================================================
    if (nv <= 1024) {
        int err = 0;
        hull_dandc_warp_mesh(verts, nv, lane, heap, scratch_heap, &err, &s_result);
        if (lane == 0 && err) s_local_err = err;
        __syncwarp();
        if (!s_local_err && s_result.verts) {
            float vol = mesh_volume_warp(&s_result, lane);
            if (lane == 0) s_volume = vol;
            __syncwarp();
        } else if (s_local_err) {
            if (lane == 0) atomicOr(kernel_error, s_local_err);
        }
        if (lane == 0) DPRINTF("[kdop] block=%d FAST nv=%d result_nv=%d result_nt=%d dt=%lld\n",
            blockIdx.x, nv, s_result.nv, s_result.nt, clock64() - t_start);
        *out_volume = s_volume;
        return s_result;
    }

    // ========================================================================
    // Step 1+2: Find extreme vertices and fill extreme point array directly
    // ========================================================================
    if (lane == 0) {
        void* ptr = NULL;
        if (heap_alloc(scratch_heap, KDOP_MAX_EXTREMES * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = KERR_KDOP_SCRATCH_OOM;
            DPRINTF("[kdop blk=%d] step1: extreme_pts alloc failed\n", blockIdx.x);
        } else {
            s_extreme_pts = (float*)ptr;
            s_n_extreme = KDOP_MAX_EXTREMES;
        }
    }
    __syncwarp();
    if (s_local_err) {
        if (lane == 0) atomicOr(kernel_error, s_local_err);
        *out_volume = 0.0f;
        return s_result;
    }

    for (int a = 0; a < KDOP_N_AXES; a++) {
        float dx = KDOP_AXES[a][0], dy = KDOP_AXES[a][1], dz = KDOP_AXES[a][2];
        float lmax = -1e30f; int lmax_i = 0;
        float lmin =  1e30f; int lmin_i = 0;
        for (int i = lane; i < nv; i += WARP_SIZE) {
            float d = dx * verts[i*3+0]
                    + dy * verts[i*3+1]
                    + dz * verts[i*3+2];
            if (d > lmax) { lmax = d; lmax_i = i; }
            if (d < lmin) { lmin = d; lmin_i = i; }
        }
        // Warp argmax
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float om = __shfl_xor_sync(WARP_MASK, lmax, off);
            int   oi = __shfl_xor_sync(WARP_MASK, lmax_i, off);
            if (om > lmax) { lmax = om; lmax_i = oi; }
        }
        // Warp argmin
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float om = __shfl_xor_sync(WARP_MASK, lmin, off);
            int   oi = __shfl_xor_sync(WARP_MASK, lmin_i, off);
            if (om < lmin) { lmin = om; lmin_i = oi; }
        }
        if (lane == 0) {
            s_extreme_pts[(a * 2) * 3 + 0] = verts[lmax_i * 3 + 0];
            s_extreme_pts[(a * 2) * 3 + 1] = verts[lmax_i * 3 + 1];
            s_extreme_pts[(a * 2) * 3 + 2] = verts[lmax_i * 3 + 2];
            s_extreme_pts[(a * 2 + 1) * 3 + 0] = verts[lmin_i * 3 + 0];
            s_extreme_pts[(a * 2 + 1) * 3 + 1] = verts[lmin_i * 3 + 1];
            s_extreme_pts[(a * 2 + 1) * 3 + 2] = verts[lmin_i * 3 + 2];
        }
    }
    __syncwarp();

#ifdef COACD_BEAM_DEBUG
    t_ep = clock64();
#endif
    // ========================================================================
    // Step 3: D&C hull on extreme points → rough inner hull (on scratch_heap)
    // ========================================================================
    {
        int err = 0;
        hull_dandc_warp_mesh(
            s_extreme_pts, s_n_extreme, lane,
            scratch_heap, scratch_heap, &err, &s_ext_hull);
        if (lane == 0 && err) s_local_err = err;
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 3b (optional): rough-hull LB pruning against caller's volume-gap
    // threshold. rough hull ⊂ exact hull ⇒ rough_vol ≤ exact_vol, so
    // max(rough_vol - mesh_vol, 0) is a lower bound on the exact volume gap.
    // If that LB exceeds the caller-supplied threshold, skip steps 4-8 and
    // return {verts=NULL} with hull_vol=rough_vol.
    // ========================================================================
    if (prune_vol_gap_threshold < 1e29f && s_ext_hull.verts) {
        float rough_vol = mesh_volume_warp(&s_ext_hull, lane);
        __shared__ int s_pruned;
        if (lane == 0) {
            s_pruned = 0;
            float vol_gap = fmaxf(rough_vol - mesh_vol, 0.0f);
            if (vol_gap > prune_vol_gap_threshold) {
                s_volume = rough_vol;
                s_pruned = 1;
            }
        }
        __syncwarp();
        if (s_pruned) {
            if (lane == 0) {
                if (s_ext_hull.verts) heap_free(scratch_heap, s_ext_hull.verts);
                s_ext_hull.verts = NULL;
                if (s_extreme_pts) heap_free(scratch_heap, s_extreme_pts);
                s_extreme_pts = NULL;
            }
            __syncwarp();
            *out_volume = s_volume;
            return s_result;
        }
    }

    // ========================================================================
    // Step 4: Allocate filtered vertex buffer (worst case: nv + ext_hull.nv)
    // ========================================================================
    if (lane == 0) {
        void* ptr = NULL;
        int max_pts = nv + s_ext_hull.nv;
        if (heap_alloc(scratch_heap, max_pts * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = KERR_KDOP_SCRATCH_OOM;
        } else {
            s_filtered = (float*)ptr;
            s_n_filtered = 0;
        }
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

#ifdef COACD_BEAM_DEBUG
    t_ephull = clock64();
#endif
    // ========================================================================
    // Step 5: Filter original vertices (keep if outside any rough-hull face)
    //         Collect survivors via ballot/popcount.
    //         Uses first rough-hull vertex as interior reference for face orientation.
    // ========================================================================
    {
        int nt_ext = s_ext_hull.nt;
        int nv_ext = s_ext_hull.nv;
        int max_filtered_floats = (nv + nv_ext) * 3;
        PC_BUF(float, ev, s_ext_hull.verts, nv_ext * 3);
        PC_BUF(int,   et, s_ext_hull.tris,  nt_ext * 3);
        PC_BUF(float, fv, s_filtered, max_filtered_floats);
        PC_BUF(float, vb, (float*)verts, nv * 3);
        // Interior reference: centroid of rough hull vertices (thread 0 computes, shared via broadcast)
        __shared__ float s_ref[3];
        if (lane == 0) {
            float rx = 0.f, ry = 0.f, rz = 0.f;
            for (int j = 0; j < nv_ext; j++) {
                rx += ev[j*3+0]; ry += ev[j*3+1]; rz += ev[j*3+2];
            }
            float inv = 1.f / (float)nv_ext;
            s_ref[0] = rx * inv; s_ref[1] = ry * inv; s_ref[2] = rz * inv;
        }
        __syncwarp();
        float rx = s_ref[0], ry = s_ref[1], rz = s_ref[2];

        // Precompute oriented planes (nx,ny,nz,d) into scratch — one float4 per triangle.
        if (lane == 0) {
            void* ptr = NULL;
            if (heap_alloc(scratch_heap, nt_ext * 4 * sizeof(float), &ptr) != HEAP_OK) {
                s_local_err = KERR_KDOP_SCRATCH_OOM;
            } else {
                s_planes = (float*)ptr;
            }
        }
        __syncwarp();
        if (s_local_err) goto cleanup;

        // Planes stored as float4 (nx,ny,nz,d) — 16B aligned, one LD/ST.E.128
        // each. heap_alloc returns 16B-aligned pointers and t*4 floats == t*16B.
        float4* pl4 = (float4*)s_planes;
        {
            for (int t = lane; t < nt_ext; t += WARP_SIZE) {
                int ia = et[t*3+0], ib = et[t*3+1], ic = et[t*3+2];
                float ax = ev[ia*3+0], ay = ev[ia*3+1], az = ev[ia*3+2];
                float e1x = ev[ib*3+0] - ax, e1y = ev[ib*3+1] - ay, e1z = ev[ib*3+2] - az;
                float e2x = ev[ic*3+0] - ax, e2y = ev[ic*3+1] - ay, e2z = ev[ic*3+2] - az;
                float pnx = e1y*e2z - e1z*e2y;
                float pny = e1z*e2x - e1x*e2z;
                float pnz = e1x*e2y - e1y*e2x;
                float pd  = pnx*ax + pny*ay + pnz*az;
                if (pnx*rx + pny*ry + pnz*rz > pd) { pnx=-pnx; pny=-pny; pnz=-pnz; pd=-pd; }
                pl4[t] = make_float4(pnx, pny, pnz, pd);
            }
        }
        __syncwarp();

        for (int i0 = 0; i0 < nv; i0 += WARP_SIZE) {
            int i = i0 + lane;
            bool keep = false;
            if (i < nv) {
                float px = vb[i*3+0];
                float py = vb[i*3+1];
                float pz = vb[i*3+2];
                // Degenerate triangles get pnx=pny=pnz=0 and pd=0 from the
                // cross product, so 0 > 0 is false and they auto-fail the
                // outward test below — no explicit skip needed.
                for (int t = 0; t < nt_ext; t++) {
                    float4 pl = pl4[t];
                    if (pl.x*px + pl.y*py + pl.z*pz > pl.w) { keep = true; break; }
                }
            }
            unsigned ballot = __ballot_sync(WARP_MASK, keep);
            int n_keep  = __popc(ballot);
            int prefix  = __popc(ballot & ((1u << lane) - 1));
            int base;
            if (lane == 0) { base = s_n_filtered; s_n_filtered += n_keep; }
            base = __shfl_sync(WARP_MASK, base, 0);
            if (keep) {
                int dst = base + prefix;
                fv[dst*3+0] = vb[i*3+0];
                fv[dst*3+1] = vb[i*3+1];
                fv[dst*3+2] = vb[i*3+2];
            }
        }
    }
    __syncwarp();

    // ========================================================================
    // Step 6: Append extreme-hull vertices to filtered set (boundary coverage)
    // ========================================================================
    {
        int ext_base;
        int nv_ext = s_ext_hull.nv;
        int max_filtered_floats = (nv + nv_ext) * 3;
        if (lane == 0) { ext_base = s_n_filtered; s_n_filtered += nv_ext; }
        ext_base = __shfl_sync(WARP_MASK, ext_base, 0);
        nv_ext   = __shfl_sync(WARP_MASK, nv_ext, 0);
        PC_BUF(float, fv, s_filtered, max_filtered_floats);
        PC_BUF(float, ev, s_ext_hull.verts, nv_ext * 3);
        for (int i = lane; i < nv_ext * 3; i += WARP_SIZE)
            fv[ext_base * 3 + i] = ev[i];
    }
    __syncwarp();

    // Free rough hull (no longer needed)
    if (lane == 0 && s_ext_hull.verts) {
        heap_free(scratch_heap, s_ext_hull.verts);
        s_ext_hull.verts = NULL;
    }
    __syncwarp();

#ifdef COACD_BEAM_DEBUG
    t_filtered = clock64();
#endif
    // ========================================================================
    // Step 7: D&C hull on filtered set → final hull (on heap)
    // ========================================================================
    {
        int n_filt = s_n_filtered;
        if (n_filt >= 4) {
            int err = 0;
            hull_dandc_warp_mesh(
                s_filtered, n_filt, lane,
                heap, scratch_heap, &err, &s_result);
            if (lane == 0 && err) s_local_err = err;
        }
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 8: Volume
    // ========================================================================
    if (s_result.verts) {
        float vol = mesh_volume_warp(&s_result, lane);
        if (lane == 0) s_volume = vol;
        __syncwarp();
    }

cleanup:
    if (lane == 0) {
        if (s_planes)      heap_free(scratch_heap, s_planes);
        if (s_filtered)    heap_free(scratch_heap, s_filtered);
        if (s_ext_hull.verts) heap_free(scratch_heap, s_ext_hull.verts);
        if (s_extreme_pts) heap_free(scratch_heap, s_extreme_pts);
        if (s_local_err) {
            atomicOr(kernel_error, s_local_err);
            // Free result mesh on error (it was allocated on heap)
            if (s_result.verts) heap_free(heap, s_result.verts);
            s_result.verts = NULL; s_result.tris = NULL;
            s_result.nv = 0; s_result.nt = 0;
        }
    }
    __syncwarp();

#ifdef COACD_BEAM_DEBUG
    t_final = clock64();
#endif

    if (lane == 0 && (s_n_filtered == 5549 || s_n_filtered == 4423)) DPRINTF("[kdop] block=%d SLOW nv=%d n_filtered=%d result_nv=%d result_nt=%d dt=%lld dt_ep=%lld dt_ephull=%lld dt_filtered=%lld dt_final=%lld \n",
        blockIdx.x, nv, s_n_filtered, s_result.nv, s_result.nt, t_final - t_start, t_ep - t_start, t_ephull - t_ep, t_filtered - t_ephull, t_final - t_filtered);
    *out_volume = s_volume;
    return s_result;
}
