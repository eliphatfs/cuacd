// c2s_kernels.cu — device-only port of cumesh2sdf (mesh → signed/unsigned
// distance field on a power-of-two voxel grid) for cuacd.
//
// Ported from github.com/eliphatfs/cumesh2sdf (rasterize.cuh, sign.cuh,
// geometry.cuh, grid.cuh, commons.cuh, structs.cuh, Apache-2.0).
// Included with authorization by the author (eliphatfs) for the cuacd
// project.
//
// Host orchestration lives in csrc/preprocess.c and launches these kernels
// through the CUDA driver API. The upstream torch/cuda-runtime glue has been
// stripped; all device math is preserved verbatim. The `probe` two-pass
// rasterize flow is preserved but the second (fill) pass always allocates the
// exact count probed, so `preloc` in the fill pass equals the total size.
//
// Coordinate convention (same as upstream): mesh normalized into [0,1]^3;
// voxel (i,j,k) samples the point ((i+0.5)/R, (j+0.5)/R, (k+0.5)/R).

#include <cuda.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Minimal vector math (subset of upstream helper_math.h used by these kernels)
// ---------------------------------------------------------------------------

struct c2s_float3 { float x, y, z; };
struct c2s_uint3  { unsigned int x, y, z; };

static __forceinline__ __host__ __device__ c2s_float3 c2s_f3(float x, float y, float z) {
  return c2s_float3{x, y, z};
}
static __forceinline__ __host__ __device__ c2s_float3 operator+(const c2s_float3& a, const c2s_float3& b) {
  return c2s_f3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __host__ __device__ c2s_float3 operator-(const c2s_float3& a, const c2s_float3& b) {
  return c2s_f3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __host__ __device__ c2s_float3 operator*(const c2s_float3& a, float s) {
  return c2s_f3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __host__ __device__ float c2s_dot(const c2s_float3& a, const c2s_float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __host__ __device__ c2s_float3 c2s_cross(const c2s_float3& a, const c2s_float3& b) {
  return c2s_f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static __forceinline__ __host__ __device__ float c2s_len(const c2s_float3& a) {
  return sqrtf(c2s_dot(a, a));
}
static __forceinline__ __host__ __device__ c2s_float3 c2s_normalize(const c2s_float3& a) {
  float l = c2s_len(a);
  return (l > 0.0f) ? a * (1.0f / l) : a;
}
static __forceinline__ __host__ __device__ float c2s_clampf(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}
static __forceinline__ __host__ __device__ float c2s_lerpf(float a, float b, float t) {
  return a + t * (b - a);
}
static __forceinline__ __host__ __device__ c2s_uint3 c2s_u3(unsigned int x, unsigned int y, unsigned int z) {
  return c2s_uint3{x, y, z};
}
static __forceinline__ __host__ __device__ c2s_uint3 c2s_uadd(const c2s_uint3& a, const c2s_uint3& b) {
  return c2s_u3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __host__ __device__ c2s_uint3 c2s_umul(const c2s_uint3& a, unsigned int s) {
  return c2s_u3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __host__ __device__ c2s_uint3 c2s_ucmp(const c2s_uint3& v, unsigned int lo, unsigned int hi) {
  return c2s_u3(
      v.x < lo ? lo : (v.x > hi ? hi : v.x),
      v.y < lo ? lo : (v.y > hi ? hi : v.y),
      v.z < lo ? lo : (v.z > hi ? hi : v.z));
}
static __forceinline__ __host__ __device__ c2s_uint3 c2s_f2i(const c2s_float3& a) {
  return c2s_u3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}
static __forceinline__ __host__ __device__ c2s_float3 c2s_i2f(const c2s_uint3& a) {
  return c2s_f3((float)a.x, (float)a.y, (float)a.z);
}

// grid.cuh — packed 30-bit grid ids (10 bits per axis, supports R <= 1024)
#define C2S_GRID_BITS 10
#define C2S_GRID_MASK ((1u << C2S_GRID_BITS) - 1u)

static __forceinline__ __host__ __device__ c2s_uint3 c2s_unpack_id(const unsigned int gridId) {
  return c2s_u3(gridId >> (C2S_GRID_BITS << 1), (gridId >> C2S_GRID_BITS) & C2S_GRID_MASK,
               gridId & C2S_GRID_MASK);
}
static __forceinline__ __host__ __device__ unsigned int c2s_pack_id(const c2s_uint3& g) {
  return g.z | (g.y << C2S_GRID_BITS) | (g.x << (C2S_GRID_BITS << 1));
}
static __forceinline__ __host__ __device__ unsigned int c2s_to_gidx(const c2s_uint3& ijk, const int N) {
  return ijk.x * (unsigned int)(N * N) + ijk.y * (unsigned int)N + ijk.z;
}

static __forceinline__ __host__ __device__ int c2s_ceil_div(const long long a, const int b) {
  return (int)((a + b - 1) / b);
}

static constexpr const float C2S_EPS = 1e-12f;

// ---------------------------------------------------------------------------
// geometry.cuh (verbatim math, adapted to c2s_float3)
// ---------------------------------------------------------------------------

static __forceinline__ __device__ float c2s_lensqr(const c2s_float3& v) { return c2s_dot(v, v); }

static __forceinline__ __device__ float c2s_point_to_segment_dist_sqr(c2s_float3 v, c2s_float3 w, c2s_float3 p) {
  w = w - v;
  p = p - v;
  const float l2 = c2s_lensqr(w);
  if (l2 < C2S_EPS) return c2s_lensqr(p);
  const float t = c2s_clampf(c2s_dot(p, w) / l2, 0.0f, 1.0f);
  const c2s_float3 projection = w * t;
  return c2s_lensqr(p - projection);
}

static __forceinline__ __device__ float c2s_origin_to_segment_dist_sqr(const c2s_float3& v, const c2s_float3& w) {
  const float l2 = c2s_lensqr(v - w);
  if (l2 < C2S_EPS) return c2s_lensqr(v);
  const float t = c2s_clampf(c2s_dot(v, v - w) / l2, 0.0f, 1.0f);
  const c2s_float3 projection = c2s_f3(c2s_lerpf(v.x, w.x, t), c2s_lerpf(v.y, w.y, t),
                                       c2s_lerpf(v.z, w.z, t));
  return c2s_lensqr(projection);
}

static __forceinline__ __device__ float c2s_point_to_tri_dist_sqr(c2s_float3 v1, c2s_float3 v2,
                                                                  c2s_float3 v3, c2s_float3 p) {
  v1 = v1 - p;
  v2 = v2 - p;
  v3 = v3 - p;
  const float d1 = c2s_origin_to_segment_dist_sqr(v1, v2);
  const float d2 = c2s_origin_to_segment_dist_sqr(v2, v3);
  const float d3 = c2s_origin_to_segment_dist_sqr(v3, v1);
  const float min_edge = fminf(fminf(d1, d2), d3);

  const c2s_float3 e0 = v2 - v1;
  const c2s_float3 e1 = v3 - v1;
  const c2s_float3 noru = c2s_cross(e0, e1);
  const float scl = c2s_lensqr(noru);
  if (scl < C2S_EPS) return min_edge;  // 0-area tri

  const c2s_float3 proj = noru * (c2s_dot(v1, noru) / scl);
  const c2s_float3 e2 = proj - v1;

  const float dot00 = c2s_dot(e0, e0);
  const float dot01 = c2s_dot(e0, e1);
  const float dot11 = c2s_dot(e1, e1);
  const float dot02 = c2s_dot(e0, e2);
  const float dot12 = c2s_dot(e1, e2);

  const float denom = dot00 * dot11 - dot01 * dot01;
  if (denom < C2S_EPS) return min_edge;
  const float invDenom = 1.0f / denom;
  const float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  const float uc = c2s_clampf(u, 0.0f, 1.0f);
  const float vc = c2s_clampf(v, 0.0f, 1.0f - uc);

  const c2s_float3 prc = v1 + (e0 * uc) + (e1 * vc);
  return fminf(c2s_lensqr(prc), min_edge);
}

#define C2S_EPS_PLANE_CLOSE 1e-5f

static __forceinline__ __device__ float c2s_ray_triangle_hit_dist(c2s_float3 v1, c2s_float3 v2,
                                                                 c2s_float3 v3, c2s_float3 ro,
                                                                 c2s_float3 rd, float pdist) {
  v2 = v2 - v1;
  v3 = v3 - v1;
  ro = ro - v1;
  const c2s_float3 cr = c2s_cross(rd, v3);
  const float det = c2s_dot(cr, v2);

  const float u = c2s_dot(ro, cr) / det;
  const c2s_float3 scr = c2s_cross(ro, v2);
  const float v = c2s_dot(rd, scr) / det;
  const float t = c2s_dot(v3, scr) / det;
  const bool di = fabsf(det) > C2S_EPS;
  const bool hit = di && t >= 0 && u >= -FLT_EPSILON && v >= -FLT_EPSILON &&
                   u + v <= 1.0f + FLT_EPSILON;
  const bool planeclose = !di && fabsf(c2s_dot(ro, c2s_normalize(c2s_cross(v2, v3)))) < C2S_EPS_PLANE_CLOSE;
  return hit ? t : (planeclose ? pdist : FLT_MAX);
}

// ---------------------------------------------------------------------------
// commons.cuh device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ static float c2s_atomicMin(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

// ---------------------------------------------------------------------------
// structs.cuh — RasterizeResult by value
// ---------------------------------------------------------------------------

struct C2S_RasterizeResult {
  float* gridDist;    // [R,R,R] distance (in normalized units), 1e9 far
  unsigned char* gridCollide;  // [R,R,R,3] axis-ray hit flags
};

// ---------------------------------------------------------------------------
// kernels: fill / arange / trisoup build (upstream commons.cuh + host glue)
// ---------------------------------------------------------------------------

extern "C" __global__ void c2s_fill_f32(const float val, const unsigned int L, float* outGrid) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= L) return;
  outGrid[g] = val;
}

// upstream common_fill_kernel<uint>: fills a uint buffer with a constant
extern "C" __global__ void c2s_fill_u32(const unsigned int val, const unsigned int L,
                                        unsigned int* outGrid) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= L) return;
  outGrid[g] = val;
}

// Fills 3*R^3 bytes of collide flags (upstream fills (R^3/4)*3 ints as int).
extern "C" __global__ void c2s_fill_collide(const unsigned int L, unsigned char* outCollide) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= L) return;
  outCollide[g] = 0;
}

extern "C" __global__ void c2s_arange_u32(unsigned int* t, const unsigned int limit,
                                          const unsigned int ofs) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= limit) return;
  t[g] = g + ofs;
}

// Build normalized triangle soup: soup[f*9 ..] = normalize(verts[tris[f*3+{0,1,2}]])
// p_norm = ((p - vmin) / extent + band) / margin  (pamo stage-1 convention:
// scalar extent over all axes keeps aspect ratio; band/margin pad the grid so
// the surface never touches the boundary)
extern "C" __global__ void c2s_build_trisoup(const float* __restrict__ verts,
                                             const int* __restrict__ tris, const int nF,
                                             const float vminx, const float vminy,
                                             const float vminz, const float extent,
                                             const float band, const float margin,
                                             float* __restrict__ soup) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (unsigned int)nF) return;
  const float vmin[3] = {vminx, vminy, vminz};
  const int f = (int)g;
  for (int k = 0; k < 3; k++) {
    const int vi = tris[f * 3 + k];
    for (int d = 0; d < 3; d++) {
      const float p = verts[vi * 3 + d];
      const float s = (p - vmin[d]) / extent + band;
      soup[(f * 3 + k) * 3 + d] = s / margin;
    }
  }
}

// ---------------------------------------------------------------------------
// rasterize.cuh — progressive grid refinement (verbatim, S templated)
// ---------------------------------------------------------------------------

template <bool probe, unsigned int S>
__device__ __forceinline__ void c2s_rasterize_layer_impl(
    const c2s_float3* __restrict__ tris, const unsigned int* __restrict__ idx,
    const unsigned int* __restrict__ grid, const int M, const int N, const float band,
    unsigned int* __restrict__ tempBlockOffset, unsigned int* __restrict__ totalSize,
    unsigned int* __restrict__ outIdx, unsigned int* __restrict__ outGrid,
    const unsigned int preloc) {
  const unsigned int b = blockIdx.x;
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= S * S * S * (unsigned int)M) return;

  __shared__ unsigned int blockSize;

  if (threadIdx.x == 0) blockSize = 0;
  __syncthreads();

  const int t = g / (S * S * S);
  const int mo = g - t * (S * S * S);
  const int k = mo / (S * S);
  const int yo = mo - k * (S * S);
  const int j = yo / S;
  const int i = yo - j * S;

  const unsigned int tofs = idx[t];
  const c2s_float3 v1 = tris[tofs * 3];
  const c2s_float3 v2 = tris[tofs * 3 + 1];
  const c2s_float3 v3 = tris[tofs * 3 + 2];

  const unsigned int gid = grid[t];
  const c2s_uint3 nxyz = c2s_uadd(c2s_umul(c2s_unpack_id(gid), S), c2s_u3(i, j, k));
  const c2s_float3 fxyz = (c2s_i2f(nxyz) + c2s_f3(0.5f, 0.5f, 0.5f)) * (1.0f / (float)N);

  const float thresh = 0.87f / N + band;
  const bool intersect = c2s_point_to_tri_dist_sqr(v1, v2, v3, fxyz) < thresh * thresh;

  unsigned int inblock = 0;
  if (intersect) inblock = atomicAdd(&blockSize, 1);

  __syncthreads();
  __shared__ unsigned int bofs;
  if (threadIdx.x == 0) {
    if (probe) bofs = tempBlockOffset[b] = atomicAdd(totalSize, blockSize);
    else bofs = tempBlockOffset[b];
  }
  __syncthreads();

  if (intersect && bofs + inblock < preloc) {
    outIdx[bofs + inblock] = tofs;
    outGrid[bofs + inblock] = c2s_pack_id(nxyz);
  }
}

extern "C" __global__ void c2s_rasterize_layer_p1(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 1>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f1(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 1>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p2(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 2>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f2(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 2>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p3(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 3>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f3(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 3>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p4(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 4>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f4(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 4>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p5(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 5>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f5(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 5>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p6(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 6>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f6(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 6>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p7(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 7>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f7(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 7>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_p8(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<true, 8>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}
extern "C" __global__ void c2s_rasterize_layer_f8(
    const c2s_float3* tris, const unsigned int* idx, const unsigned int* grid, const int M,
    const int N, const float band, unsigned int* tempBlockOffset, unsigned int* totalSize,
    unsigned int* outIdx, unsigned int* outGrid, const unsigned int preloc) {
  c2s_rasterize_layer_impl<false, 8>(tris, idx, grid, M, N, band, tempBlockOffset,
                                                     totalSize, outIdx, outGrid, preloc);
}


extern "C" __global__ void c2s_rasterize_reduce(
    const c2s_float3* __restrict__ tris, const unsigned int* __restrict__ idx,
    const unsigned int* __restrict__ grid, const int M, const int N,
    float* __restrict__ outGridDist, unsigned char* __restrict__ outGridCollide) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (unsigned int)M) return;
  const c2s_uint3 nxyz = c2s_unpack_id(grid[g]);
  const c2s_float3 fxyz = (c2s_i2f(nxyz) + c2s_f3(0.5f, 0.5f, 0.5f)) * (1.0f / (float)N);
  const unsigned int access = c2s_to_gidx(nxyz, N);

  const unsigned int tofs = idx[g];
  const c2s_float3 v1 = tris[tofs * 3];
  const c2s_float3 v2 = tris[tofs * 3 + 1];
  const c2s_float3 v3 = tris[tofs * 3 + 2];

  const float finalDist = sqrtf(c2s_point_to_tri_dist_sqr(v1, v2, v3, fxyz));
  c2s_atomicMin(outGridDist + access, finalDist);

  const float rayth = 1.0f / N + FLT_EPSILON;
  if (finalDist > rayth) return;
  if (c2s_ray_triangle_hit_dist(v1, v2, v3, fxyz, c2s_f3(1, 0, 0), finalDist) <= rayth)
    outGridCollide[access * 3] = 1;
  if (c2s_ray_triangle_hit_dist(v1, v2, v3, fxyz, c2s_f3(0, 1, 0), finalDist) <= rayth)
    outGridCollide[access * 3 + 1] = 1;
  if (c2s_ray_triangle_hit_dist(v1, v2, v3, fxyz, c2s_f3(0, 0, 1), finalDist) <= rayth)
    outGridCollide[access * 3 + 2] = 1;
}

// ---------------------------------------------------------------------------
// sign.cuh — flood-fill sign via lock-free union-find (verbatim)
// ---------------------------------------------------------------------------

static __forceinline__ __device__ unsigned int c2s_shuffler(unsigned int v, unsigned int bmask) {
  (void)bmask;
  return v;  // trade off algorithmic complexity for memory coalesce (upstream)
}

static __forceinline__ __device__ unsigned int c2s_cts_find(const unsigned int* parents,
                                                            const unsigned int i) {
  unsigned int c = i;
  while (parents[c] != c) c = parents[c];
  return c;
}

static __forceinline__ __device__ void c2s_cts_atomic_union(unsigned int* __restrict__ parents,
                                                            unsigned int x, unsigned int y) {
  if (x == y) return;
  while (true) {
    unsigned int px = parents[x];
    unsigned int py = parents[y];
    if (px == y || x == py || px == py) return;
    if (px == x && py == y) atomicCAS(&parents[(x > y) ? x : y], (x > y) ? x : y,
                                      (x < y) ? x : y);
    x = px;
    y = py;
  }
}

extern "C" __global__ void c2s_volume_sign_prescan(
    const C2S_RasterizeResult rast, unsigned int* __restrict__ parents, const unsigned int N,
    const int shfBitmask) {
  const unsigned int xyx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int xyy = blockIdx.y * blockDim.y + threadIdx.y;
  if (xyx >= N || xyy >= N) return;
  bool skip = false;
  bool flag = false;
  unsigned int shfcn = c2s_shuffler(0, shfBitmask);
  if (xyx == 0 && xyy == 0) parents[shfcn] = shfcn;
  for (unsigned int i = 0; i < N; i++) {
    const c2s_uint3 xyz = c2s_u3(i, xyy, xyx);
    const unsigned int access = c2s_to_gidx(xyz, N);
    const unsigned int shfm = c2s_shuffler(access + 1, shfBitmask);
    if (skip) {
      skip = false;
      if (flag) {
        flag = false;
        shfcn = shfm;
      }
    } else {
      const float dist = rast.gridDist[access] * N;
      if (dist < 0.87f) {
        flag = true;
        shfcn = shfm;
      } else if (flag) {
        flag = false;
        shfcn = shfm;
      }
      skip = dist > 2.0f;
    }
    parents[shfm] = shfcn;
  }
}

extern "C" __global__ void c2s_volume_cts(const C2S_RasterizeResult rast,
                                          unsigned int* __restrict__ parents,
                                          const unsigned int N, const int shfBitmask) {
  const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int tidz = blockIdx.z * blockDim.z + threadIdx.z;
  const c2s_uint3 xyz = c2s_u3(tidy, tidz, tidx);
  if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
  const unsigned int access = c2s_to_gidx(xyz, N);
  const unsigned int shfm = c2s_shuffler(access + 1, shfBitmask);

  if (xyz.x == 0 || xyz.y == 0 || xyz.z == 0) {
    // NB: upstream reads gridDist into an `int` (truncation) before comparing
    // against 0.86603f/N; this effectively unions a boundary cell with the
    // exterior only when its raw distance >= 1.0, which keeps in-band
    // boundary cells out of the exterior component. Preserved verbatim.
    const int dist = (int)rast.gridDist[access];
    if (dist >= 0.86603f / N) {
      const unsigned int shfex = c2s_shuffler(0, shfBitmask);
      c2s_cts_atomic_union(parents, shfm, shfex);
    }
  }

  const unsigned int a3 = access * 3;
  if (!rast.gridCollide[a3]) {
    const c2s_uint3 nxyz = c2s_ucmp(c2s_uadd(xyz, c2s_u3(1, 0, 0)), 0, N - 1);
    const unsigned int shfn = c2s_shuffler(c2s_to_gidx(nxyz, N) + 1, shfBitmask);
    c2s_cts_atomic_union(parents, shfm, shfn);
  }
  if (!rast.gridCollide[a3 + 1]) {
    const c2s_uint3 nxyz = c2s_ucmp(c2s_uadd(xyz, c2s_u3(0, 1, 0)), 0, N - 1);
    const unsigned int shfn = c2s_shuffler(c2s_to_gidx(nxyz, N) + 1, shfBitmask);
    c2s_cts_atomic_union(parents, shfm, shfn);
  }
  if (!rast.gridCollide[a3 + 2]) {
    const c2s_uint3 nxyz = c2s_ucmp(c2s_uadd(xyz, c2s_u3(0, 0, 1)), 0, N - 1);
    const unsigned int shfn = c2s_shuffler(c2s_to_gidx(nxyz, N) + 1, shfBitmask);
    c2s_cts_atomic_union(parents, shfm, shfn);
  }
}

extern "C" __global__ void c2s_volume_apply_sign(const C2S_RasterizeResult rast,
                                                 const unsigned int* __restrict__ parents,
                                                 const int N, const int shfBitmask) {
  const unsigned int root = c2s_cts_find(parents, c2s_shuffler(0, shfBitmask));

  const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int tidz = blockIdx.z * blockDim.z + threadIdx.z;
  const c2s_uint3 xyz = c2s_u3(tidz, tidy, tidx);
  if (xyz.x >= (unsigned int)N || xyz.y >= (unsigned int)N || xyz.z >= (unsigned int)N) return;
  const unsigned int access = c2s_to_gidx(xyz, N);
  const unsigned int shfm = c2s_shuffler(access + 1, shfBitmask);

  if (c2s_cts_find(parents, shfm) != root) rast.gridDist[access] *= -1;
}

// Apply the pamo stage-1 SDF shift in-place: d -= 0.9/R. pamo does this on the
// torch tensor between get_sdf and DualMC.
extern "C" __global__ void c2s_sdf_shift(const unsigned int L, float* __restrict__ grid,
                                         const float shift) {
  const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= L) return;
  grid[g] -= shift;
}
