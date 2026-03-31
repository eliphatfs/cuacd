// Geometry device functions: edge intersection, point-triangle distance,
// bbox concavity metric.

#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "common.cuh"

// Signed volume of tetrahedron formed by triangle and origin.
// V = p0 . (p1 x p2) / 6
__device__ inline float signed_tet_volume(
    float p0x, float p0y, float p0z,
    float p1x, float p1y, float p1z,
    float p2x, float p2y, float p2z)
{
    float cx = p1y * p2z - p1z * p2y;
    float cy = p1z * p2x - p1x * p2z;
    float cz = p1x * p2y - p1y * p2x;
    return (p0x * cx + p0y * cy + p0z * cz) / 6.0f;
}

// Compute intersection point of edge (v0→v1) with plane (a,b,c,d).
__device__ inline void intersect_edge(
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float a, float b, float c, float d,
    float* ix, float* iy, float* iz)
{
    float d0 = a * v0x + b * v0y + c * v0z + d;
    float d1 = a * v1x + b * v1y + c * v1z + d;
    float t = d0 / (d0 - d1);
    *ix = v0x + t * (v1x - v0x);
    *iy = v0y + t * (v1y - v0y);
    *iz = v0z + t * (v1z - v0z);
}

// Minimum distance from point to triangle (Eberly's region-based method).
__device__ inline float point_triangle_dist(
    float px, float py, float pz,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z)
{
    float e0x = v1x - v0x, e0y = v1y - v0y, e0z = v1z - v0z;
    float e1x = v2x - v0x, e1y = v2y - v0y, e1z = v2z - v0z;
    float dx = v0x - px, dy = v0y - py, dz = v0z - pz;

    float a = e0x * e0x + e0y * e0y + e0z * e0z;
    float b = e0x * e1x + e0y * e1y + e0z * e1z;
    float c = e1x * e1x + e1y * e1y + e1z * e1z;
    float d = e0x * dx + e0y * dy + e0z * dz;
    float e = e1x * dx + e1y * dy + e1z * dz;

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if (s + t <= det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                if (d < 0.0f) { t = 0.0f; s = (-d >= a) ? 1.0f : -d / a; }
                else { s = 0.0f; t = (e >= 0.0f) ? 0.0f : ((-e >= c) ? 1.0f : -e / c); }
            } else {
                s = 0.0f;
                t = (e >= 0.0f) ? 0.0f : ((-e >= c) ? 1.0f : -e / c);
            }
        } else if (t < 0.0f) {
            t = 0.0f;
            s = (d >= 0.0f) ? 0.0f : ((-d >= a) ? 1.0f : -d / a);
        } else {
            float inv_det = 1.0f / det;
            s *= inv_det;
            t *= inv_det;
        }
    } else {
        if (s < 0.0f) {
            float tmp0 = b + d, tmp1 = c + e;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0, denom = a - 2.0f * b + c;
                s = (numer >= denom) ? 1.0f : numer / denom;
                t = 1.0f - s;
            } else {
                s = 0.0f;
                t = (tmp1 <= 0.0f) ? 1.0f : ((e >= 0.0f) ? 0.0f : -e / c);
            }
        } else if (t < 0.0f) {
            float tmp0 = b + e, tmp1 = a + d;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0, denom = a - 2.0f * b + c;
                t = (numer >= denom) ? 1.0f : numer / denom;
                s = 1.0f - t;
            } else {
                t = 0.0f;
                s = (tmp1 <= 0.0f) ? 1.0f : ((d >= 0.0f) ? 0.0f : -d / a);
            }
        } else {
            float numer = (c + e) - (b + d);
            if (numer <= 0.0f) { s = 0.0f; t = 1.0f; }
            else {
                float denom = a - 2.0f * b + c;
                s = (numer >= denom) ? 1.0f : numer / denom;
                t = 1.0f - s;
            }
        }
    }

    float cx = v0x + s * e0x + t * e1x - px;
    float cy = v0y + s * e0y + t * e1y - py;
    float cz = v0z + s * e0z + t * e1z - pz;
    return sqrtf(cx * cx + cy * cy + cz * cz);
}

// Rv (volume-ratio concavity) from mesh and hull volumes.
// Rv = (3 * |V_mesh - V_hull| / (4*pi))^(1/3) * k
__device__ inline float rv_from_volumes(float mesh_vol, float hull_vol, float rv_k) {
    float diff = fabsf(mesh_vol - hull_vol);
    return cbrtf(3.0f * diff / (4.0f * PI_F)) * rv_k;
}

// Bbox concavity metric: cbrt(bbox_volume) of triangle vertices.
// Uses block-level reduction. All threads must call this.
__device__ inline float compute_concavity_tris(
    const float* verts,
    const int*   tris,
    int          n_tris,
    int          tid,
    float*       smem)       // [BLOCK_SIZE]
{
    float lo[3] = {1e30f, 1e30f, 1e30f};
    float hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int t = tid; t < n_tris; t += BLOCK_SIZE) {
        for (int e = 0; e < 3; e++) {
            int vi = tris[t*3+e];
            for (int k = 0; k < 3; k++) {
                float v = verts[vi*3+k];
                lo[k] = fminf(lo[k], v);
                hi[k] = fmaxf(hi[k], v);
            }
        }
    }

    __shared__ float s_lo[3], s_hi[3];
    for (int k = 0; k < 3; k++) {
        smem[tid] = lo[k];
        __syncthreads();
        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fminf(smem[tid], smem[tid+s]);
            __syncthreads();
        }
        if (tid == 0) s_lo[k] = smem[0];

        smem[tid] = hi[k];
        __syncthreads();
        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
            __syncthreads();
        }
        if (tid == 0) s_hi[k] = smem[0];
        __syncthreads();
    }

    __shared__ float result;
    if (tid == 0) {
        float dims[3];
        for (int k = 0; k < 3; k++)
            dims[k] = fmaxf(s_hi[k] - s_lo[k], 0.0f);
        float vol = dims[0] * dims[1] * dims[2];
        result = cbrtf(fmaxf(vol, 0.0f));
    }
    __syncthreads();
    return result;
}

#endif // GEOMETRY_CUH
