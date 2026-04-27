// merge.cuh — helpers for the CoACD merge-hulls post-processing pass.
//
// Shared inline device helpers used by postprocess_merge.cu:
//   - la_merge_pair_index  : flat upper-triangle index <-> (p1,p2)
//   - la_merge_bbox_dist2  : squared distance between two AABBs (0 if overlap)
//   - la_merge_find_shared_plane_warp : port of CoACD ComputeOverlapFace.
//       Finds a separating hyperplane between two convex hulls; returns the
//       face plane (pa,pb,pc,pd) from hullA whose opposite half-space contains
//       all of hullB's vertices (and hullA's verts are either on the plane or
//       on the correct side). Equivalent plane is returned.
//   - la_merge_concat_verts_warp : concat two vertex buffers into scratch.

#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "warp_common.cuh"

// Flat upper-triangle indexing used across merge kernels.
//   p1 > p2,  idx = p1*(p1-1)/2 + p2,   range = P*(P-1)/2 entries.
__device__ __forceinline__ int la_merge_pair_index(int p1, int p2) {
    return p1 * (p1 - 1) / 2 + p2;
}

// Inverse: given a flat index and P, return (p1,p2) with p1 > p2.
__device__ __forceinline__ void la_merge_pair_from_index(int idx, int P, int& p1, int& p2) {
    // Quadratic-formula inversion of k = p1*(p1-1)/2 + p2.
    // p1 = floor((1 + sqrt(1 + 8k)) / 2)
    float k = (float)idx;
    int guess = (int)floorf((1.0f + sqrtf(1.0f + 8.0f * k)) * 0.5f);
    // Clamp — float rounding can put us one off either way.
    while (guess > 1 && guess * (guess - 1) / 2 > idx) guess--;
    while (guess * (guess + 1) / 2 <= idx) guess++;
    p1 = guess;
    p2 = idx - guess * (guess - 1) / 2;
    (void)P;
}

// Squared AABB gap (0 if overlap). Both bboxes are 3-vecs min/max.
__device__ __forceinline__ float la_merge_bbox_gap2(
    const float amn[3], const float amx[3],
    const float bmn[3], const float bmx[3])
{
    float gx = fmaxf(0.0f, fmaxf(amn[0] - bmx[0], bmn[0] - amx[0]));
    float gy = fmaxf(0.0f, fmaxf(amn[1] - bmx[1], bmn[1] - amx[1]));
    float gz = fmaxf(0.0f, fmaxf(amn[2] - bmx[2], bmn[2] - amx[2]));
    return gx * gx + gy * gy + gz * gz;
}

// Per-warp AABB reduction of a vertex buffer. Writes min/max into out[3][2]
// via shared-memory atomics-free reduction (warp-exclusive).
// (min, max) layout: out_min[3], out_max[3]. Caller writes results to smem.
__device__ __forceinline__ void la_merge_bbox_warp(
    const float* verts, int nv, int lane,
    float out_min[3], float out_max[3])
{
    float mx = -1e30f, my = -1e30f, mz = -1e30f;
    float nx =  1e30f, ny =  1e30f, nz =  1e30f;
    for (int i = lane; i < nv; i += WARP_SIZE) {
        float x = verts[i*3+0], y = verts[i*3+1], z = verts[i*3+2];
        mx = fmaxf(mx, x); my = fmaxf(my, y); mz = fmaxf(mz, z);
        nx = fminf(nx, x); ny = fminf(ny, y); nz = fminf(nz, z);
    }
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(WARP_MASK, mx, off));
        my = fmaxf(my, __shfl_xor_sync(WARP_MASK, my, off));
        mz = fmaxf(mz, __shfl_xor_sync(WARP_MASK, mz, off));
        nx = fminf(nx, __shfl_xor_sync(WARP_MASK, nx, off));
        ny = fminf(ny, __shfl_xor_sync(WARP_MASK, ny, off));
        nz = fminf(nz, __shfl_xor_sync(WARP_MASK, nz, off));
    }
    if (lane == 0) {
        out_min[0] = nx; out_min[1] = ny; out_min[2] = nz;
        out_max[0] = mx; out_max[1] = my; out_max[2] = mz;
    }
}

// Concatenate verts_a[0..nva) and verts_b[0..nvb) into dst[0..nva+nvb).
// Warp-cooperative strided copy. Caller has allocated dst on scratch.
__device__ __forceinline__ void la_merge_concat_verts_warp(
    const float* va, int nva,
    const float* vb, int nvb,
    float* dst, int lane)
{
    for (int i = lane; i < nva * 3; i += WARP_SIZE)
        dst[i] = va[i];
    __syncwarp();
    for (int i = lane; i < nvb * 3; i += WARP_SIZE)
        dst[nva * 3 + i] = vb[i];
}

// ---------------------------------------------------------------------------
// Shared-plane detection (port of CoACD ComputeOverlapFace).
//
// Input: two convex hulls A and B. For each triangle of A, compute its plane
// (oriented pointing outward). If ALL verts of A lie on the inward side
// (n·v <= d + tol) AND ALL verts of B lie on the plane itself (|n·v - d| <= tol),
// then this face is a shared separating plane — record it and stop.
//
// Return: 1 if a plane was found (written to *out_plane as float4{nx,ny,nz,d});
//         0 otherwise.
// Uses 32 threads (warp-cooperative scan over A's faces).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int la_merge_find_shared_plane_warp(
    const Mesh& hullA, const Mesh& hullB,
    float tol, int lane,
    float4* out_plane)
{
    __shared__ int   s_found;
    __shared__ float4 s_plane;
    if (lane == 0) { s_found = 0; }
    __syncwarp();

    const float* av = hullA.verts;
    const int*   at = hullA.tris;
    const float* bv = hullB.verts;
    int nva = hullA.nv, nta = hullA.nt;
    int nvb = hullB.nv;

    // Each lane processes strided faces of A.
    for (int t = lane; t < nta && !s_found; t += WARP_SIZE) {
        int ia = at[t*3+0], ib = at[t*3+1], ic = at[t*3+2];
        float ax = av[ia*3+0], ay = av[ia*3+1], az = av[ia*3+2];
        float bx = av[ib*3+0], by = av[ib*3+1], bz = av[ib*3+2];
        float cx = av[ic*3+0], cy = av[ic*3+1], cz = av[ic*3+2];

        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y*e2z - e1z*e2y;
        float ny = e1z*e2x - e1x*e2z;
        float nz = e1x*e2y - e1y*e2x;
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len < 1e-20f) continue;
        float inv = 1.0f / len;
        nx *= inv; ny *= inv; nz *= inv;
        float d  = nx*ax + ny*ay + nz*az;

        // Check hullA verts: each must satisfy n·v <= d + tol.
        bool ok = true;
        for (int j = 0; j < nva && ok; j++) {
            float s = nx*av[j*3+0] + ny*av[j*3+1] + nz*av[j*3+2] - d;
            if (s > tol) ok = false;
        }
        if (!ok) continue;

        // Check hullB verts: each must satisfy |n·v - d| <= tol (on the plane).
        for (int j = 0; j < nvb && ok; j++) {
            float s = nx*bv[j*3+0] + ny*bv[j*3+1] + nz*bv[j*3+2] - d;
            if (fabsf(s) > tol) ok = false;
        }
        if (!ok) continue;

        // First finder wins.
        if (atomicCAS(&s_found, 0, 1) == 0) {
            s_plane = make_float4(nx, ny, nz, d);
        }
    }
    __syncwarp();

    if (s_found) {
        if (lane == 0) *out_plane = s_plane;
        __syncwarp();
        return 1;
    }
    return 0;
}
