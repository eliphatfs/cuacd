// Pure device code — compiled to fatbin, loaded via CUDA driver API.
// No host-side includes, no runtime API calls.

extern "C" {

__device__ float point_triangle_dist(
    float px, float py, float pz,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z) {
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

  // Eberly's region-based closest point on triangle
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

// For each query point, find the minimum distance to any triangle in the mesh.
// One thread per query point, brute-force scan over all triangles.
__global__ void point_mesh_distance(
    const float* __restrict__ points,    // [N, 3]
    const float* __restrict__ vertices,  // [V, 3]
    const int*   __restrict__ triangles, // [T, 3]
    float*       __restrict__ distances, // [N]
    int N, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float px = points[idx * 3 + 0];
  float py = points[idx * 3 + 1];
  float pz = points[idx * 3 + 2];

  float min_dist = 1e30f;
  for (int t = 0; t < T; t++) {
    int i0 = triangles[t * 3 + 0];
    int i1 = triangles[t * 3 + 1];
    int i2 = triangles[t * 3 + 2];
    float dist = point_triangle_dist(
        px, py, pz,
        vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2],
        vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2],
        vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
    min_dist = fminf(min_dist, dist);
  }
  distances[idx] = min_dist;
}

// Shared-memory max reduction. Two-stage: first pass reduces blocks, second
// pass reduces the block results. Launched with dynamic shared memory.
__global__ void reduce_max(
    const float* __restrict__ data,
    float*       __restrict__ output,
    int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float val = -1e30f;
  if (idx < N) val = data[idx];
  if (idx + blockDim.x < N) val = fmaxf(val, data[idx + blockDim.x]);
  sdata[tid] = val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Atomic max for floats via CAS on int reinterpretation.
__device__ float atomicMaxF(float* addr, float value) {
  int* addr_i = (int*)addr;
  int old = *addr_i, expected;
  do {
    expected = old;
    old = atomicCAS(addr_i, expected,
                    __float_as_int(fmaxf(value, __int_as_float(expected))));
  } while (old != expected);
  return __int_as_float(old);
}

// Pairwise Hausdorff for merge phase.
// Grid: (pair_idx, sample_block). Each thread handles one sample point for one
// (i,j) pair, computes min dist to the other part's mesh, atomically updates
// the max into cost_matrix[i*P+j].
__global__ void pairwise_hausdorff(
    const float* __restrict__ all_samples,
    const int*   __restrict__ sample_offsets, // [P+1]
    const float* __restrict__ all_vertices,
    const int*   __restrict__ all_triangles,
    const int*   __restrict__ tri_offsets,    // [P+1]
    const int*   __restrict__ vert_offsets,   // [P+1]
    float*       __restrict__ cost_matrix,    // [P*P]
    int P) {
  int pair_idx = blockIdx.x;
  int sample_local = blockIdx.y * blockDim.x + threadIdx.x;

  // Map flat pair_idx to (i,j) with i > j
  int i = (int)(0.5f + sqrtf(0.25f + 2.0f * (float)pair_idx));
  while (i * (i - 1) / 2 > pair_idx) i--;
  while ((i + 1) * i / 2 <= pair_idx) i++;
  int j = pair_idx - i * (i - 1) / 2;
  if (i >= P || j >= P || i <= j) return;

  int ns_i = sample_offsets[i + 1] - sample_offsets[i];
  int ns_j = sample_offsets[j + 1] - sample_offsets[j];

  // Helper: compute min dist from a sample point to a part's mesh
  #define SCAN_TRIS(s_off, s_local, t_start, t_end, v_off, cell) \
    if (s_local < (s_off)) { \
      int si = sample_offsets[cell##_src] + s_local; \
      float px = all_samples[si*3], py = all_samples[si*3+1], pz = all_samples[si*3+2]; \
      float md = 1e30f; \
      for (int _t = (t_start); _t < (t_end); _t++) { \
        int _i0 = all_triangles[_t*3] + (v_off); \
        int _i1 = all_triangles[_t*3+1] + (v_off); \
        int _i2 = all_triangles[_t*3+2] + (v_off); \
        float _d = point_triangle_dist(px, py, pz, \
            all_vertices[_i0*3], all_vertices[_i0*3+1], all_vertices[_i0*3+2], \
            all_vertices[_i1*3], all_vertices[_i1*3+1], all_vertices[_i1*3+2], \
            all_vertices[_i2*3], all_vertices[_i2*3+1], all_vertices[_i2*3+2]); \
        md = fminf(md, _d); \
      } \
      atomicMaxF(&cost_matrix[i * P + j], md); \
    }

  // Samples from part i → mesh of part j
  if (sample_local < ns_i) {
    int si = sample_offsets[i] + sample_local;
    float px = all_samples[si*3], py = all_samples[si*3+1], pz = all_samples[si*3+2];
    int ts = tri_offsets[j], te = tri_offsets[j+1], vo = vert_offsets[j];
    float md = 1e30f;
    for (int t = ts; t < te; t++) {
      int i0 = all_triangles[t*3]+vo, i1 = all_triangles[t*3+1]+vo, i2 = all_triangles[t*3+2]+vo;
      float d = point_triangle_dist(px,py,pz,
          all_vertices[i0*3],all_vertices[i0*3+1],all_vertices[i0*3+2],
          all_vertices[i1*3],all_vertices[i1*3+1],all_vertices[i1*3+2],
          all_vertices[i2*3],all_vertices[i2*3+1],all_vertices[i2*3+2]);
      md = fminf(md, d);
    }
    atomicMaxF(&cost_matrix[i*P+j], md);
  }

  // Samples from part j → mesh of part i
  if (sample_local < ns_j) {
    int si = sample_offsets[j] + sample_local;
    float px = all_samples[si*3], py = all_samples[si*3+1], pz = all_samples[si*3+2];
    int ts = tri_offsets[i], te = tri_offsets[i+1], vo = vert_offsets[i];
    float md = 1e30f;
    for (int t = ts; t < te; t++) {
      int i0 = all_triangles[t*3]+vo, i1 = all_triangles[t*3+1]+vo, i2 = all_triangles[t*3+2]+vo;
      float d = point_triangle_dist(px,py,pz,
          all_vertices[i0*3],all_vertices[i0*3+1],all_vertices[i0*3+2],
          all_vertices[i1*3],all_vertices[i1*3+1],all_vertices[i1*3+2],
          all_vertices[i2*3],all_vertices[i2*3+1],all_vertices[i2*3+2]);
      md = fminf(md, d);
    }
    atomicMaxF(&cost_matrix[i*P+j], md);
  }
}

} // extern "C"
