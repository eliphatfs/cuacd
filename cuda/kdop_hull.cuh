// kdop_hull.cuh — Fast approximate convex hull via k-DOP (80 half-spaces).
//
// Block-level algorithm (KDOP_BLOCK threads per block, 1 block per mesh).
// Uses 40 icosphere-L1 face normals as k-DOP axes.
//
// Algorithm:
//   1. Compute centroid (block-parallel reduce).
//   2. For each of 40 axes, find extreme projections (block-parallel reduce).
//   3. Build 80 half-spaces, compute polar dual vertices (n_i / h_i).
//   4. Convex hull of dual points via hull_dandc_warp_mesh (warp 0).
//   5. Solve 3-plane intersections per dual triangle → primal vertices.
//   6. Convex hull of primal vertices (warp 0) → final k-DOP mesh.
//   7. Translate mesh back by centroid, compute volume (warp 0).

#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "hull_dandc.cuh"
#include "mesh_volume.cuh"

#define KDOP_BLOCK 64
#define KDOP_N_AXES 40
#define KDOP_MAX_DUAL_PTS (KDOP_N_AXES * 2)
#define KDOP_EPS 1e-7f

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

// Solve 3x3 system Ax = b using Cramer's rule.
// Rows of A are the three plane normals; b = (1, 1, 1) for polar dual.
// Returns false if singular (|det| < eps).
__device__ inline bool kdop_solve3x3(
    float a0x, float a0y, float a0z,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float b0, float b1, float b2,
    float* ox, float* oy, float* oz)
{
    float det = a0x * (a1y * a2z - a1z * a2y)
              - a0y * (a1x * a2z - a1z * a2x)
              + a0z * (a1x * a2y - a1y * a2x);
    if (fabsf(det) < 1e-12f) return false;
    float inv = 1.0f / det;
    *ox = (b0 * (a1y * a2z - a1z * a2y)
         - a0y * (b1 * a2z - b2 * a1z)
         + a0z * (b1 * a2y - b2 * a1y)) * inv;
    *oy = (a0x * (b1 * a2z - b2 * a1z)
         - b0 * (a1x * a2z - a1z * a2x)
         + a0z * (a1x * b2 - b1 * a2x)) * inv;
    *oz = (a0x * (a1y * b2 - b1 * a2y)
         - a0y * (a1x * b2 - b1 * a2x)
         + b0 * (a1x * a2y - a1y * a2x)) * inv;
    return true;
}

// kdop_hull_block: approximate convex hull of a point cloud via k-DOP.
//
// All KDOP_BLOCK threads in the block must call with identical arguments.
// Returns Mesh (heap-allocated) and volume via out_volume.
// On error, returns {NULL,NULL,0,0} and sets *kernel_error.
__device__ inline Mesh kdop_hull_block(
    const float* verts, int nv,
    DeviceHeap* heap,
    DeviceHeap* scratch_heap,
    float* out_volume,
    int* kernel_error)
{
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);

    // Shared memory for reductions and intermediate results
    __shared__ float s_reduce[KDOP_BLOCK];
    __shared__ float s_centroid[3];
    __shared__ float s_max[KDOP_N_AXES];
    __shared__ float s_min[KDOP_N_AXES];
    __shared__ float* s_dual_pts;
    __shared__ float* s_primal_pts;
    __shared__ Mesh s_dual_mesh;
    __shared__ Mesh s_result;
    __shared__ int s_n_dual;
    __shared__ int s_n_primal;
    __shared__ float s_volume;
    __shared__ int s_local_err;

    if (tid == 0) {
        s_result.verts = NULL; s_result.tris = NULL;
        s_result.nv = 0; s_result.nt = 0; s_result.refcount = NULL;
        s_dual_pts = NULL; s_primal_pts = NULL;
        s_dual_mesh.verts = NULL; s_dual_mesh.tris = NULL;
        s_dual_mesh.nv = 0; s_dual_mesh.nt = 0; s_dual_mesh.refcount = NULL;
        s_volume = 0.0f;
        s_local_err = 0;
    }
    __syncthreads();

    if (nv < 4) { *out_volume = 0.0f; return s_result; }

    // ========================================================================
    // Step 1: Compute centroid (block-parallel)
    // ========================================================================
    for (int axis = 0; axis < 3; axis++) {
        float sum = 0.0f;
        for (int i = tid; i < nv; i += KDOP_BLOCK)
            sum += verts[i * 3 + axis];
        // Warp-level reduction
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(WARP_MASK, sum, off);
        // Inter-warp reduction via shared memory
        if (lane == 0) s_reduce[tid / WARP_SIZE] = sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < KDOP_BLOCK / WARP_SIZE; w++)
                total += s_reduce[w];
            s_centroid[axis] = total / (float)nv;
        }
        __syncthreads();
    }

    float cx = s_centroid[0], cy = s_centroid[1], cz = s_centroid[2];

    // ========================================================================
    // Step 2: Find k-DOP extremes (block-parallel)
    // ========================================================================
    for (int a = 0; a < KDOP_N_AXES; a++) {
        float dx = KDOP_AXES[a][0], dy = KDOP_AXES[a][1], dz = KDOP_AXES[a][2];
        float local_max = -1e30f;
        float local_min =  1e30f;
        for (int i = tid; i < nv; i += KDOP_BLOCK) {
            float px = verts[i * 3 + 0] - cx;
            float py = verts[i * 3 + 1] - cy;
            float pz = verts[i * 3 + 2] - cz;
            float d = dx * px + dy * py + dz * pz;
            local_max = fmaxf(local_max, d);
            local_min = fminf(local_min, d);
        }
        // Warp reduce max
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(WARP_MASK, local_max, off));
            local_min = fminf(local_min, __shfl_down_sync(WARP_MASK, local_min, off));
        }
        if (lane == 0) {
            s_reduce[tid / WARP_SIZE] = local_max;
            // Reuse a second slot for min (offset by number of warps)
        }
        __syncthreads();
        if (tid == 0) {
            float mx = s_reduce[0];
            for (int w = 1; w < KDOP_BLOCK / WARP_SIZE; w++)
                mx = fmaxf(mx, s_reduce[w]);
            s_max[a] = mx;
        }
        // Now reduce min
        if (lane == 0) s_reduce[tid / WARP_SIZE] = local_min;
        __syncthreads();
        if (tid == 0) {
            float mn = s_reduce[0];
            for (int w = 1; w < KDOP_BLOCK / WARP_SIZE; w++)
                mn = fminf(mn, s_reduce[w]);
            s_min[a] = mn;
        }
        __syncthreads();
    }

    // ========================================================================
    // Step 3: Build polar dual vertices (thread 0)
    // ========================================================================
    // Allocate scratch for dual points: up to 80 points × 3 floats
    if (tid == 0) {
        void* ptr = NULL;
        if (heap_alloc(scratch_heap, KDOP_MAX_DUAL_PTS * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = 1;
        } else {
            s_dual_pts = (float*)ptr;
        }
    }
    __syncthreads();
    if (s_local_err) {
        if (tid == 0) atomicOr(kernel_error, 0x100000);
        *out_volume = 0.0f;
        return s_result;
    }

    // Build dual points: for each half-space n_i · x <= h_i, dual point = n_i / h_i
    if (tid == 0) {
        int nd = 0;
        for (int a = 0; a < KDOP_N_AXES; a++) {
            float h_pos = s_max[a];  // axis_a · x <= h_pos
            float h_neg = -s_min[a]; // (-axis_a) · x <= h_neg = -min(axis_a · x)
            if (h_pos > KDOP_EPS) {
                float inv_h = 1.0f / h_pos;
                s_dual_pts[nd * 3 + 0] = KDOP_AXES[a][0] * inv_h;
                s_dual_pts[nd * 3 + 1] = KDOP_AXES[a][1] * inv_h;
                s_dual_pts[nd * 3 + 2] = KDOP_AXES[a][2] * inv_h;
                nd++;
            }
            if (h_neg > KDOP_EPS) {
                float inv_h = 1.0f / h_neg;
                s_dual_pts[nd * 3 + 0] = -KDOP_AXES[a][0] * inv_h;
                s_dual_pts[nd * 3 + 1] = -KDOP_AXES[a][1] * inv_h;
                s_dual_pts[nd * 3 + 2] = -KDOP_AXES[a][2] * inv_h;
                nd++;
            }
        }
        s_n_dual = nd;
    }
    __syncthreads();

    int n_dual = s_n_dual;

    // ========================================================================
    // Step 4: Convex hull of dual points (warp 0)
    // ========================================================================
    if (tid < WARP_SIZE) {
        int err = 0;
        Mesh dm = hull_dandc_warp_mesh(
            s_dual_pts, n_dual, lane,
            scratch_heap, scratch_heap, &err);
        if (lane == 0) {
            s_dual_mesh = dm;
            if (err) s_local_err = 2;
        }
    }
    __syncthreads();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 5: Compute primal vertices from dual triangles (thread 0)
    // ========================================================================
    // Each dual triangle (d_a, d_b, d_c) maps to a primal vertex at the
    // intersection of planes d_a·x=1, d_b·x=1, d_c·x=1.
    if (tid == 0) {
        int nt_dual = s_dual_mesh.nt;
        void* ptr = NULL;
        if (heap_alloc(scratch_heap, nt_dual * 3 * sizeof(float), &ptr) != HEAP_OK) {
            s_local_err = 3;
        } else {
            s_primal_pts = (float*)ptr;
            int np = 0;
            for (int t = 0; t < nt_dual; t++) {
                int ia = s_dual_mesh.tris[t * 3 + 0];
                int ib = s_dual_mesh.tris[t * 3 + 1];
                int ic = s_dual_mesh.tris[t * 3 + 2];
                float ax = s_dual_mesh.verts[ia * 3 + 0];
                float ay = s_dual_mesh.verts[ia * 3 + 1];
                float az = s_dual_mesh.verts[ia * 3 + 2];
                float bx = s_dual_mesh.verts[ib * 3 + 0];
                float by = s_dual_mesh.verts[ib * 3 + 1];
                float bz = s_dual_mesh.verts[ib * 3 + 2];
                float ccx = s_dual_mesh.verts[ic * 3 + 0];
                float ccy = s_dual_mesh.verts[ic * 3 + 1];
                float ccz = s_dual_mesh.verts[ic * 3 + 2];
                float ox, oy, oz;
                if (kdop_solve3x3(ax, ay, az, bx, by, bz, ccx, ccy, ccz,
                                  1.0f, 1.0f, 1.0f, &ox, &oy, &oz)) {
                    s_primal_pts[np * 3 + 0] = ox;
                    s_primal_pts[np * 3 + 1] = oy;
                    s_primal_pts[np * 3 + 2] = oz;
                    np++;
                }
            }
            s_n_primal = np;
        }
    }
    __syncthreads();
    if (s_local_err) goto cleanup;

    // Free dual mesh now (it was allocated on scratch_heap)
    if (tid == 0 && s_dual_mesh.refcount) {
        // The dual hull chunk is on scratch_heap; find the allocation start.
        // hull_dandc_warp_mesh allocates [verts | tris | refcount] as one chunk.
        // verts pointer is the start of the chunk.
        heap_free(scratch_heap, s_dual_mesh.verts);
        s_dual_mesh.verts = NULL;
    }
    __syncthreads();

    // ========================================================================
    // Step 6: Convex hull of primal vertices → final k-DOP mesh (warp 0)
    // ========================================================================
    {
        int n_primal = s_n_primal;
        if (tid < WARP_SIZE && n_primal >= 4) {
            int err = 0;
            Mesh pm = hull_dandc_warp_mesh(
                s_primal_pts, n_primal, lane,
                heap, scratch_heap, &err);
            if (lane == 0) {
                s_result = pm;
                if (err) s_local_err = 4;
            }
        }
    }
    __syncthreads();
    if (s_local_err) goto cleanup;

    // ========================================================================
    // Step 6b: Translate vertices back by centroid (block-parallel)
    // ========================================================================
    {
        int final_nv = s_result.nv;
        for (int i = tid; i < final_nv; i += KDOP_BLOCK) {
            s_result.verts[i * 3 + 0] += cx;
            s_result.verts[i * 3 + 1] += cy;
            s_result.verts[i * 3 + 2] += cz;
        }
    }
    __syncthreads();

    // ========================================================================
    // Step 7: Compute volume (warp 0)
    // ========================================================================
    if (tid < WARP_SIZE) {
        float vol = mesh_volume_warp(&s_result, lane);
        if (lane == 0) s_volume = vol;
    }
    __syncthreads();

cleanup:
    // Free scratch allocations
    if (tid == 0) {
        if (s_primal_pts) heap_free(scratch_heap, s_primal_pts);
        if (s_dual_mesh.verts) heap_free(scratch_heap, s_dual_mesh.verts);
        if (s_dual_pts) heap_free(scratch_heap, s_dual_pts);
        if (s_local_err && s_local_err != 1)
            atomicOr(kernel_error, 0x100000 + s_local_err);
    }
    __syncthreads();

    *out_volume = s_volume;
    return s_result;
}
