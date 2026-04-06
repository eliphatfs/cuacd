// kdop_hull.cuh — Convex hull via extreme-point prefilter + exact D&C hull.
//
// Single-warp (KDOP_BLOCK = 32) algorithm, 1 block per mesh.
//
// Fast path (nv <= 1024): run hull_dandc_warp_mesh directly.
//
// Main path (nv > 1024):
//   1. Compute centroid (warp reduce).
//   2. Find 40 max + 40 min extreme vertices using KDOP_AXES (warp argmax/argmin).
//   3. Deduplicate and collect up to 80 unique extreme vertices (centroid-subtracted).
//   4. Run hull_dandc_warp_mesh on extreme vertices → rough inner hull.
//   5. Filter original vertices: keep any point strictly outside at least one face
//      of the rough hull (ballot/popcount collect).
//   6. Append extreme-hull vertices to the filtered set (ensures boundary coverage).
//   7. Run hull_dandc_warp_mesh on filtered set → final hull.
//   8. Translate back by centroid, compute volume.

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
__device__ static const float KDOP_AXES[KDOP_N_AXES][3] = {
    { -0.5831287832f,  0.7487834504f,  0.3150939013f },
    { -0.7487834504f,  0.3150939013f,  0.5831287832f },
    { -0.3150939013f,  0.5831287832f,  0.7487834504f },
    { -0.5773502692f,  0.5773502692f,  0.5773502692f },
    { -0.2680348819f,  0.9435221910f,  0.1947387407f },
    {  0.0000000000f,  0.7778675239f,  0.6284282897f },
    {  0.2680348819f,  0.9435221910f,  0.1947387407f },
    {  0.0000000000f,  0.9341723590f,  0.3568220898f },
    {  0.2680348819f, -0.9435221910f,  0.1947387407f },
    { -0.2680348819f, -0.9435221910f,  0.1947387407f },
    {  0.0000000000f, -0.7778675239f,  0.6284282897f },
    {  0.0000000000f, -0.9341723590f,  0.3568220898f },
    {  0.5831287832f, -0.7487834504f,  0.3150939013f },
    {  0.3150939013f, -0.5831287832f,  0.7487834504f },
    {  0.7487834504f, -0.3150939013f,  0.5831287832f },
    {  0.5773502692f, -0.5773502692f,  0.5773502692f },
    { -0.7778675239f,  0.6284282897f,  0.0000000000f },
    {  0.9435221910f, -0.1947387407f,  0.2680348819f },
    { -0.9435221910f,  0.1947387407f,  0.2680348819f },
    { -0.9341723590f,  0.3568220898f,  0.0000000000f },
    {  0.5831287832f,  0.7487834504f,  0.3150939013f },
    {  0.3150939013f,  0.5831287832f,  0.7487834504f },
    {  0.7487834504f,  0.3150939013f,  0.5831287832f },
    {  0.5773502692f,  0.5773502692f,  0.5773502692f },
    { -0.1947387407f,  0.2680348819f,  0.9435221910f },
    { -0.6284282897f,  0.0000000000f,  0.7778675239f },
    { -0.1947387407f, -0.2680348819f,  0.9435221910f },
    { -0.3568220898f,  0.0000000000f,  0.9341723590f },
    { -0.9435221910f, -0.1947387407f,  0.2680348819f },
    {  0.9435221910f,  0.1947387407f,  0.2680348819f },
    {  0.7778675239f,  0.6284282897f,  0.0000000000f },
    {  0.9341723590f,  0.3568220898f,  0.0000000000f },
    {  0.6284282897f,  0.0000000000f,  0.7778675239f },
    {  0.1947387407f, -0.2680348819f,  0.9435221910f },
    {  0.1947387407f,  0.2680348819f,  0.9435221910f },
    {  0.3568220898f,  0.0000000000f,  0.9341723590f },
    { -0.3150939013f, -0.5831287832f,  0.7487834504f },
    { -0.5831287832f, -0.7487834504f,  0.3150939013f },
    { -0.7487834504f, -0.3150939013f,  0.5831287832f },
    { -0.5773502692f, -0.5773502692f,  0.5773502692f },
};

// kdop_hull_block: convex hull of a point cloud.
//
// All KDOP_BLOCK (= 32) threads in the block must call with identical arguments.
// tris/nt are accepted for API compatibility but not used.
// Returns Mesh (heap-allocated) and volume via out_volume.
// On error, returns {NULL,NULL,0,0} and sets *kernel_error.
__device__ inline Mesh kdop_hull_block(
    const float* verts, int nv,
    const int*   tris,  int nt,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    float*       out_volume,
    int*         kernel_error)
{
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Shared state
    __shared__ float  s_centroid[3];
    __shared__ float  s_max[KDOP_N_AXES];
    __shared__ float  s_min[KDOP_N_AXES];
    __shared__ int    s_extreme_idx[KDOP_MAX_EXTREMES]; // max/min index per axis
    __shared__ float* s_extreme_pts;  // centroid-subtracted extreme pts (scratch)
    __shared__ int    s_n_extreme;
    __shared__ Mesh   s_ext_hull;     // rough hull of extreme pts (scratch)
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
        s_filtered = NULL;
        s_n_extreme = 0; s_n_filtered = 0;
        s_volume = 0.0f;
        s_local_err = 0;
    }
    __syncwarp();

    if (nv < 4) { *out_volume = 0.0f; return s_result; }

    // ========================================================================
    // Fast path: small mesh → exact D&C hull directly
    // ========================================================================
    if (nv <= 1024) {
        DPRINTF("[kdop blk=%d lane=%d] fast path nv=%d\n", blockIdx.x, lane, nv);
        int err = 0;
        Mesh m = hull_dandc_warp_mesh(verts, nv, lane, heap, scratch_heap, &err);
        if (lane == 0) {
            s_result = m;
            if (err) s_local_err = err;
            DPRINTF("[kdop blk=%d] fast path done: nv=%d nt=%d err=%d\n",
                    blockIdx.x, m.nv, m.nt, err);
        }
        __syncwarp();
        if (!s_local_err && s_result.verts) {
            float vol = mesh_volume_warp(&s_result, lane);
            if (lane == 0) s_volume = vol;
            __syncwarp();
        } else if (s_local_err) {
            if (lane == 0) atomicOr(kernel_error, 0x100000);
        }
        *out_volume = s_volume;
        return s_result;
    }

    // ========================================================================
    // Step 1: Centroid (warp reduce)
    // ========================================================================
    for (int axis = 0; axis < 3; axis++) {
        float sum = 0.0f;
        for (int i = lane; i < nv; i += WARP_SIZE)
            sum += verts[i * 3 + axis];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(WARP_MASK, sum, off);
        if (lane == 0) s_centroid[axis] = sum / (float)nv;
    }
    __syncwarp();
    float cx = s_centroid[0], cy = s_centroid[1], cz = s_centroid[2];

    // ========================================================================
    // Step 2: Find extreme vertices (warp argmax/argmin, with index)
    // ========================================================================
    for (int a = 0; a < KDOP_N_AXES; a++) {
        float dx = KDOP_AXES[a][0], dy = KDOP_AXES[a][1], dz = KDOP_AXES[a][2];
        float lmax = -1e30f; int lmax_i = 0;
        float lmin =  1e30f; int lmin_i = 0;
        for (int i = lane; i < nv; i += WARP_SIZE) {
            float d = dx * (verts[i*3+0] - cx)
                    + dy * (verts[i*3+1] - cy)
                    + dz * (verts[i*3+2] - cz);
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
            s_max[a] = lmax;  s_extreme_idx[a * 2    ] = lmax_i;
            s_min[a] = lmin;  s_extreme_idx[a * 2 + 1] = lmin_i;
        }
    }
    __syncwarp();

    // ========================================================================
    // Step 3: Build deduplicated extreme point array (thread 0, centroid-sub)
    // ========================================================================
    if (lane == 0) {
        void* ptr = NULL;
        if (heap_alloc(scratch_heap, KDOP_MAX_EXTREMES * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = 1;
            DPRINTF("[kdop blk=%d] step3: extreme_pts alloc failed\n", blockIdx.x);
        } else {
            s_extreme_pts = (float*)ptr;
            PC_BUF(float, ep, s_extreme_pts, KDOP_MAX_EXTREMES * 3);
            PC_BUF(float, vb, (float*)verts, nv * 3);
            int n_ext = 0;
            int seen[KDOP_MAX_EXTREMES];
            for (int i = 0; i < KDOP_MAX_EXTREMES; i++) {
                int idx = s_extreme_idx[i];
                bool dup = false;
                for (int j = 0; j < n_ext; j++) {
                    if (seen[j] == idx) { dup = true; break; }
                }
                if (!dup) {
                    seen[n_ext] = idx;
                    ep[n_ext * 3 + 0] = vb[idx * 3 + 0] - cx;
                    ep[n_ext * 3 + 1] = vb[idx * 3 + 1] - cy;
                    ep[n_ext * 3 + 2] = vb[idx * 3 + 2] - cz;
                    n_ext++;
                }
            }
            s_n_extreme = n_ext;
            DPRINTF("[kdop blk=%d] step3: %d unique extremes (nv=%d)\n", blockIdx.x, n_ext, nv);
        }
    }
    __syncwarp();
    if (s_local_err) {
        if (lane == 0) atomicOr(kernel_error, 0x100000);
        *out_volume = 0.0f;
        return s_result;
    }

    // ========================================================================
    // Step 4: D&C hull on extreme points → rough inner hull (on scratch_heap)
    // ========================================================================
    {
        int err = 0;
        Mesh em = hull_dandc_warp_mesh(
            s_extreme_pts, s_n_extreme, lane,
            scratch_heap, scratch_heap, &err);
        if (lane == 0) {
            s_ext_hull = em;
            if (err) s_local_err = 2;
            DPRINTF("[kdop blk=%d] step4: rough hull nv=%d nt=%d err=%d\n",
                    blockIdx.x, em.nv, em.nt, err);
        }
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 5: Allocate filtered vertex buffer (worst case: nv + ext_hull.nv)
    // ========================================================================
    if (lane == 0) {
        void* ptr = NULL;
        int max_pts = nv + s_ext_hull.nv;
        if (heap_alloc(scratch_heap, max_pts * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = 3;
            DPRINTF("[kdop blk=%d] step5: filtered alloc failed (max_pts=%d)\n",
                    blockIdx.x, max_pts);
        } else {
            s_filtered = (float*)ptr;
            s_n_filtered = 0;
        }
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 6: Filter original vertices (keep if outside any rough-hull face)
    //         Collect survivors via ballot/popcount.
    // ========================================================================
    {
        int nt_ext = s_ext_hull.nt;
        int nv_ext = s_ext_hull.nv;
        int max_filtered_floats = (nv + nv_ext) * 3;
        PC_BUF(float, ev, s_ext_hull.verts, nv_ext * 3);
        PC_BUF(int,   et, s_ext_hull.tris,  nt_ext * 3);
        PC_BUF(float, fv, s_filtered, max_filtered_floats);
        PC_BUF(float, vb, (float*)verts, nv * 3);
        for (int i0 = 0; i0 < nv; i0 += WARP_SIZE) {
            int i = i0 + lane;
            bool keep = false;
            if (i < nv) {
                float px = vb[i*3+0] - cx;
                float py = vb[i*3+1] - cy;
                float pz = vb[i*3+2] - cz;
                for (int t = 0; t < nt_ext && !keep; t++) {
                    int ia = et[t*3+0];
                    int ib = et[t*3+1];
                    int ic = et[t*3+2];
                    float ax = ev[ia*3+0], ay = ev[ia*3+1], az = ev[ia*3+2];
                    float e1x = ev[ib*3+0] - ax;
                    float e1y = ev[ib*3+1] - ay;
                    float e1z = ev[ib*3+2] - az;
                    float e2x = ev[ic*3+0] - ax;
                    float e2y = ev[ic*3+1] - ay;
                    float e2z = ev[ic*3+2] - az;
                    // Outward normal (ensure centroid=origin is inside: dot(n,0) < d → d > 0)
                    float nx = e1y*e2z - e1z*e2y;
                    float ny = e1z*e2x - e1x*e2z;
                    float nz = e1x*e2y - e1y*e2x;
                    float d  = nx*ax + ny*ay + nz*az;
                    if (d < 0.0f) { nx = -nx; ny = -ny; nz = -nz; d = -d; }
                    if (d == 0.0f) continue; // degenerate face, skip
                    if (nx*px + ny*py + nz*pz > d) keep = true;
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
                fv[dst*3+0] = vb[i*3+0] - cx;
                fv[dst*3+1] = vb[i*3+1] - cy;
                fv[dst*3+2] = vb[i*3+2] - cz;
            }
        }
    }
    __syncwarp();
    DPRINTF("[kdop blk=%d lane=%d] step6: filtered %d / %d verts\n",
            blockIdx.x, lane, s_n_filtered, nv);

    // ========================================================================
    // Step 7: Append extreme-hull vertices to filtered set (boundary coverage)
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

    // ========================================================================
    // Step 8: D&C hull on filtered set → final hull (on heap)
    // ========================================================================
    {
        int n_filt = s_n_filtered;
        DPRINTF("[kdop blk=%d lane=%d] step8: final dandc on %d pts\n",
                blockIdx.x, lane, n_filt);
        if (n_filt >= 4) {
            int err = 0;
            Mesh fm = hull_dandc_warp_mesh(
                s_filtered, n_filt, lane,
                heap, scratch_heap, &err);
            if (lane == 0) {
                s_result = fm;
                if (err) s_local_err = 4;
                DPRINTF("[kdop blk=%d] step8: final hull nv=%d nt=%d err=%d\n",
                        blockIdx.x, fm.nv, fm.nt, err);
            }
        }
    }
    __syncwarp();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 8b: Translate result vertices back by centroid (warp-parallel)
    // ========================================================================
    {
        int final_nv = s_result.nv;
        for (int i = lane; i < final_nv; i += WARP_SIZE) {
            s_result.verts[i*3+0] += cx;
            s_result.verts[i*3+1] += cy;
            s_result.verts[i*3+2] += cz;
        }
    }
    __syncwarp();

    // ========================================================================
    // Step 9: Volume
    // ========================================================================
    if (s_result.verts) {
        float vol = mesh_volume_warp(&s_result, lane);
        if (lane == 0) s_volume = vol;
        __syncwarp();
    }

cleanup:
    if (lane == 0) {
        if (s_filtered)    heap_free(scratch_heap, s_filtered);
        if (s_ext_hull.verts) heap_free(scratch_heap, s_ext_hull.verts);
        if (s_extreme_pts) heap_free(scratch_heap, s_extreme_pts);
        if (s_local_err) {
            atomicOr(kernel_error, 0x100000 + s_local_err);
            // Free result mesh on error (it was allocated on heap)
            if (s_result.verts) heap_free(heap, s_result.verts);
            s_result.verts = NULL; s_result.tris = NULL;
            s_result.nv = 0; s_result.nt = 0;
        }
    }
    __syncwarp();

    *out_volume = s_volume;
    return s_result;
}
