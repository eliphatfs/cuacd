// hausdorff.cuh — GPU bidirectional Hausdorff distance (sampling + linear BVH).
//
// One block (256 threads = 8 warps) computes the bidirectional Hausdorff
// distance between two meshes (typically a convex hull and its source part).
//
// Algorithm:
//   1  Compute per-triangle areas, prefix-sum sample counts   (256 threads)
//   2  Generate sample points via barycentric sampling         (256 threads)
//   3  Brute-force path if target mesh has <= 64 triangles     (256 threads)
//   4  Morton code computation for BVH path                    (256 threads)
//   5  Cooperative 4-warp sort of Morton codes (x2 sets)       (8 warps)
//   6  Linear BVH construction (Karras 2012)                   (256 threads)
//   7  BVH traversal — nearest-neighbor queries                (256 threads)
//
// Scratch memory is allocated from scratch_heap and freed before return.

#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "reduce.cuh"
#include "warp_common.cuh"
#include "warp_sort.cuh"

#define HD_BLOCK       256
#define HD_BRUTE_THRESH 128
#define HD_RESOLUTION  2000
#define HD_MIN_SAMPLES 1000
#define HD_BVH_STACK    32

// Error codes (bits in kernel_error).
#define HD_KERR_SCRATCH_OOM  0x200000
#define HD_KERR_SORT_ERR     0x400000

// ============================================================================
// Helper: Wang hash for deterministic pseudo-random sampling
// ============================================================================

__device__ inline unsigned int hd_wang_hash(unsigned int seed) {
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}

// ============================================================================
// Helper: point-to-segment squared distance
// ============================================================================

__device__ inline float hd_dist_pt_seg_sq(
    float px, float py, float pz,
    float ax, float ay, float az,
    float bx, float by, float bz)
{
    float abx = bx - ax, aby = by - ay, abz = bz - az;
    float apx = px - ax, apy = py - ay, apz = pz - az;
    float ab2 = abx * abx + aby * aby + abz * abz;
    float t = (ab2 > 1e-20f) ? (apx * abx + apy * aby + apz * abz) / ab2 : 0.0f;
    t = fmaxf(0.0f, fminf(1.0f, t));
    float dx = apx - t * abx, dy = apy - t * aby, dz = apz - t * abz;
    return dx * dx + dy * dy + dz * dz;
}

// ============================================================================
// Helper: point-to-triangle squared distance
// ============================================================================

__device__ inline float hd_dist_pt_tri_sq(
    float px, float py, float pz,
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz)
{
    float abx = bx - ax, aby = by - ay, abz = bz - az;
    float acx = cx - ax, acy = cy - ay, acz = cz - az;
    float apx = px - ax, apy = py - ay, apz = pz - az;

    // Triangle normal (unnormalized).
    float nx = aby * acz - abz * acy;
    float ny = abz * acx - abx * acz;
    float nz = abx * acy - aby * acx;
    float n2 = nx * nx + ny * ny + nz * nz;

    if (n2 < 1e-20f) {
        // Degenerate triangle — return min vertex distance.
        float da = apx * apx + apy * apy + apz * apz;
        float bpx = px - bx, bpy = py - by, bpz = pz - bz;
        float db = bpx * bpx + bpy * bpy + bpz * bpz;
        float cpx = px - cx, cpy = py - cy, cpz = pz - cz;
        float dc = cpx * cpx + cpy * cpy + cpz * cpz;
        return fminf(da, fminf(db, dc));
    }

    // Barycentric test for projection inside triangle.
    float d00 = abx * abx + aby * aby + abz * abz;
    float d01 = abx * acx + aby * acy + abz * acz;
    float d02 = abx * apx + aby * apy + abz * apz;
    float d11 = acx * acx + acy * acy + acz * acz;
    float d12 = acx * apx + acy * apy + acz * apz;

    float inv = 1.0f / (d00 * d11 - d01 * d01);
    float u = (d11 * d02 - d01 * d12) * inv;
    float v = (d00 * d12 - d01 * d02) * inv;

    if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
        float dist = nx * apx + ny * apy + nz * apz;
        return (dist * dist) / n2;
    }

    // Outside triangle: min distance to 3 edges.
    float e0 = hd_dist_pt_seg_sq(px, py, pz, ax, ay, az, bx, by, bz);
    float e1 = hd_dist_pt_seg_sq(px, py, pz, bx, by, bz, cx, cy, cz);
    float e2 = hd_dist_pt_seg_sq(px, py, pz, cx, cy, cz, ax, ay, az);
    return fminf(e0, fminf(e1, e2));
}

// ============================================================================
// Helper: point-to-AABB squared distance
// ============================================================================

__device__ inline float hd_pt_aabb_dist_sq(
    float px, float py, float pz,
    const float* bmin, const float* bmax)
{
    float dx = fmaxf(fmaxf(bmin[0] - px, 0.0f), px - bmax[0]);
    float dy = fmaxf(fmaxf(bmin[1] - py, 0.0f), py - bmax[1]);
    float dz = fmaxf(fmaxf(bmin[2] - pz, 0.0f), pz - bmax[2]);
    return dx * dx + dy * dy + dz * dz;
}

// ============================================================================
// Helper: triangle area (float)
// ============================================================================

__device__ inline float hd_tri_area(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz)
{
    float abx = bx - ax, aby = by - ay, abz = bz - az;
    float acx = cx - ax, acy = cy - ay, acz = cz - az;
    float nx = aby * acz - abz * acy;
    float ny = abz * acx - abx * acz;
    float nz = abx * acy - aby * acx;
    return 0.5f * sqrtf(nx * nx + ny * ny + nz * nz);
}

// ============================================================================
// Helper: point-to-triangle distance using mesh data
// ============================================================================

__device__ inline float hd_pt_mesh_tri_dist_sq(
    float px, float py, float pz,
    int tri_idx, const Mesh* mesh)
{
    int ia = mesh->tris[tri_idx * 3 + 0];
    int ib = mesh->tris[tri_idx * 3 + 1];
    int ic = mesh->tris[tri_idx * 3 + 2];
    return hd_dist_pt_tri_sq(
        px, py, pz,
        mesh->verts[ia * 3], mesh->verts[ia * 3 + 1], mesh->verts[ia * 3 + 2],
        mesh->verts[ib * 3], mesh->verts[ib * 3 + 1], mesh->verts[ib * 3 + 2],
        mesh->verts[ic * 3], mesh->verts[ic * 3 + 1], mesh->verts[ic * 3 + 2]);
}

// ============================================================================
// Morton code helpers
// ============================================================================

__device__ inline unsigned int hd_expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline unsigned int hd_morton3D(unsigned int x, unsigned int y, unsigned int z) {
    return (hd_expand_bits(x) << 2) | (hd_expand_bits(y) << 1) | hd_expand_bits(z);
}

// ============================================================================
// MortonPoint — sortable element for BVH construction
// ============================================================================

struct MortonPoint {
    unsigned int code;
    int index;
};

struct MortonPointCmp {
    static __device__ inline int cmp(MortonPoint a, MortonPoint b) {
        if (a.code != b.code) return (a.code < b.code) ? -1 : 1;
        if (a.index != b.index) return (a.index < b.index) ? -1 : 1;
        return 0;
    }
    static __device__ inline MortonPoint sentinel() {
        MortonPoint s; s.code = 0xFFFFFFFFu; s.index = 0x7fffffff; return s;
    }
};

// ============================================================================
// BVH node
// ============================================================================

struct BVHNode {
    float bmin[3];
    float bmax[3];
    int left;       // child index; -1 = leaf marker
    int right;
    int parent;
    int tri_idx;    // source triangle index (leaf only)
};

// ============================================================================
// Karras delta: longest common prefix between keys[i] and keys[j].
// Returns -1 if j out of range.
// ============================================================================

__device__ inline int hd_delta(const MortonPoint* keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    if (keys[i].code == keys[j].code)
        return 32 + __clz(i ^ j);
    return __clz(keys[i].code ^ keys[j].code);
}

// ============================================================================
// Block-level exclusive prefix sum on int array in global memory.
// counts[] is in global (scratch heap); smem_i is shared [HD_BLOCK].
// After call, counts[i] = sum of counts[0..i-1] (sequential exclusive
// prefix sum), and *total is the sum of all elements.
// ============================================================================

__device__ inline void hd_block_prefix_sum(
    int* counts, int n, int* smem_i, int tid, int* total)
{
    // Sequential exclusive prefix sum by thread 0.
    // n is the triangle count (at most a few thousand), so this is fast.
    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < n; i++) {
            int c = counts[i];
            counts[i] = running;
            running += c;
        }
        *total = running;
    }
    __syncthreads();
}

// ============================================================================
// Free-all macro for scratch cleanup on error.
// ============================================================================

#define HD_FREE_ALL_SCRATCH() do { \
    heap_free(scratch_heap, (void*)s_counts_a); \
    heap_free(scratch_heap, (void*)s_counts_b); \
    heap_free(scratch_heap, (void*)s_samples_a); /* single block for samples+tri_ids */ \
    heap_free(scratch_heap, (void*)s_morton_a); \
    heap_free(scratch_heap, (void*)s_morton_b); \
    heap_free(scratch_heap, (void*)s_sort_scratch); \
    heap_free(scratch_heap, (void*)s_bvh_a); \
    heap_free(scratch_heap, (void*)s_bvh_b); \
    heap_free(scratch_heap, (void*)s_bvh_counters); \
} while(0)

// ============================================================================
// hausdorff_block — main entry point (256 threads per block)
// ============================================================================

__device__ __forceinline__ float hausdorff_block(
    const Mesh* __restrict__ a,
    const Mesh* __restrict__ b,
    DeviceHeap* __restrict__ scratch_heap,
    int* __restrict__ kernel_error)
{
    int tid = threadIdx.x;

    // Shared memory.
    __shared__ float  s_reduce[HD_BLOCK];
    __shared__ int    s_reduce_i[HD_BLOCK];
    __shared__ int    s_alloc_ok;
    __shared__ float  s_result;

    // Scratch pointers (all init to NULL for safe cleanup).
    __shared__ int*          s_counts_a;
    __shared__ int*          s_counts_b;
    __shared__ float*        s_samples_a;
    __shared__ float*        s_samples_b;
    __shared__ int*          s_tri_ids_a;
    __shared__ int*          s_tri_ids_b;
    __shared__ MortonPoint*  s_morton_a;
    __shared__ MortonPoint*  s_morton_b;
    __shared__ char*         s_sort_scratch;
    __shared__ BVHNode*      s_bvh_a;
    __shared__ BVHNode*      s_bvh_b;
    __shared__ int*          s_bvh_counters;

    // Scalars.
    __shared__ int   s_n_samples_a;
    __shared__ int   s_n_samples_b;
    __shared__ int   s_resolution_a;
    __shared__ int   s_resolution_b;
    __shared__ float s_total_area_a;
    __shared__ float s_total_area_b;
    __shared__ float s_dir_ab;
    __shared__ float s_dir_ba;
    __shared__ float s_bbox_min[3];
    __shared__ float s_bbox_max[3];

    // Sort coordination.
    __shared__ int s_seg_a[6];  // segment boundaries for 4-warp sort
    __shared__ int s_seg_b[6];

    if (tid == 0) {
        s_counts_a = NULL; s_counts_b = NULL;
        s_samples_a = NULL; s_samples_b = NULL;
        s_tri_ids_a = NULL; s_tri_ids_b = NULL;
        s_morton_a = NULL; s_morton_b = NULL;
        s_sort_scratch = NULL;
        s_bvh_a = NULL; s_bvh_b = NULL;
        s_bvh_counters = NULL;
        s_result = 0.0f;
        s_dir_ab = 0.0f;
        s_dir_ba = 0.0f;
    }
    __syncthreads();

    // Early exit: degenerate meshes.
    if (a->nt <= 0 || a->nv <= 0 || b->nt <= 0 || b->nv <= 0)
        return 0.0f;

    // Checked wrappers for input mesh data.
    PC_BUF(float, av, a->verts, a->nv * 3);
    PC_BUF(int,   at, a->tris,  a->nt * 3);
    PC_BUF(float, bv, b->verts, b->nv * 3);
    PC_BUF(int,   bt, b->tris,  b->nt * 3);

    // =================================================================
    // Phase 1: Compute triangle areas and sample counts.
    // =================================================================

    // --- Mesh A ---
    {
        float local_area = 0.0f;
        for (int t = tid; t < a->nt; t += HD_BLOCK) {
            int ia = at[t * 3 + 0], ib = at[t * 3 + 1], ic = at[t * 3 + 2];
            local_area += hd_tri_area(
                av[ia*3], av[ia*3+1], av[ia*3+2],
                av[ib*3], av[ib*3+1], av[ib*3+2],
                av[ic*3], av[ic*3+1], av[ic*3+2]);
        }
        float ta = block_reduce_sum(local_area, s_reduce, tid);
        if (tid == 0) {
            s_total_area_a = ta;
            int res = (int)(HD_RESOLUTION * ta);
            if (res < HD_MIN_SAMPLES) res = HD_MIN_SAMPLES;
            s_resolution_a = res;
        }
    }
    __syncthreads();

    // --- Mesh B ---
    {
        float local_area = 0.0f;
        for (int t = tid; t < b->nt; t += HD_BLOCK) {
            int ia = bt[t * 3 + 0], ib = bt[t * 3 + 1], ic = bt[t * 3 + 2];
            local_area += hd_tri_area(
                bv[ia*3], bv[ia*3+1], bv[ia*3+2],
                bv[ib*3], bv[ib*3+1], bv[ib*3+2],
                bv[ic*3], bv[ic*3+1], bv[ic*3+2]);
        }
        float tb = block_reduce_sum(local_area, s_reduce, tid);
        if (tid == 0) {
            s_total_area_b = tb;
            int res = (int)(HD_RESOLUTION * tb);
            if (res < HD_MIN_SAMPLES) res = HD_MIN_SAMPLES;
            s_resolution_b = res;
        }
    }
    __syncthreads();

    float area_a = s_total_area_a;
    float area_b = s_total_area_b;
    int   res_a  = s_resolution_a;
    int   res_b  = s_resolution_b;

    // Alloc per-triangle count arrays.
    if (tid == 0) {
        void *p1 = NULL, *p2 = NULL;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(a->nt * (int)sizeof(int)), &p1) != HEAP_OK)
            { s_alloc_ok = 0; p1 = NULL; }
        if (heap_alloc(scratch_heap, (unsigned int)(b->nt * (int)sizeof(int)), &p2) != HEAP_OK)
            { s_alloc_ok = 0; p2 = NULL; }
        s_counts_a = (int*)p1;
        s_counts_b = (int*)p2;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
        return 0.0f;
    }

    PC_BUF(int, counts_a, s_counts_a, a->nt);
    PC_BUF(int, counts_b, s_counts_b, b->nt);

    // Compute per-triangle sample counts for mesh A.
    for (int t = tid; t < a->nt; t += HD_BLOCK) {
        int ia = at[t * 3 + 0], ib = at[t * 3 + 1], ic = at[t * 3 + 2];
        float ta = hd_tri_area(
            av[ia*3], av[ia*3+1], av[ia*3+2],
            av[ib*3], av[ib*3+1], av[ib*3+2],
            av[ic*3], av[ic*3+1], av[ic*3+2]);
        int N;
        int area_count = (area_a > 1e-20f) ? (int)((float)res_a / area_a * ta) : 0;
        if (a->nt > res_a) {
            int step = a->nt / res_a;
            N = (area_count > 0) ? area_count : ((t % step == 0) ? 1 : 0);
        } else {
            N = (area_count > 0) ? area_count : ((t % 2 == 0) ? 1 : 0);
        }
        counts_a[t] = N;
    }

    // Compute per-triangle sample counts for mesh B.
    for (int t = tid; t < b->nt; t += HD_BLOCK) {
        int ia = bt[t * 3 + 0], ib = bt[t * 3 + 1], ic = bt[t * 3 + 2];
        float tb = hd_tri_area(
            bv[ia*3], bv[ia*3+1], bv[ia*3+2],
            bv[ib*3], bv[ib*3+1], bv[ib*3+2],
            bv[ic*3], bv[ic*3+1], bv[ic*3+2]);
        int N;
        int area_count = (area_b > 1e-20f) ? (int)((float)res_b / area_b * tb) : 0;
        if (b->nt > res_b) {
            int step = b->nt / res_b;
            N = (area_count > 0) ? area_count : ((t % step == 0) ? 1 : 0);
        } else {
            N = (area_count > 0) ? area_count : ((t % 2 == 0) ? 1 : 0);
        }
        counts_b[t] = N;
    }
    __syncthreads();

    // Prefix sum to get offsets and totals.
    int total_a = 0, total_b = 0;
    hd_block_prefix_sum(s_counts_a, a->nt, s_reduce_i, tid, &total_a);
    hd_block_prefix_sum(s_counts_b, b->nt, s_reduce_i, tid, &total_b);

    if (tid == 0) {
        s_n_samples_a = total_a;
        s_n_samples_b = total_b;
    }
    __syncthreads();

    int n_sa = s_n_samples_a;
    int n_sb = s_n_samples_b;

    // Edge case: no samples.
    if (n_sa <= 0 || n_sb <= 0) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); }
        return 0.0f;
    }

    // =================================================================
    // Phase 2: Allocate sample buffers and generate sample points.
    // =================================================================

    // Allocate all sample buffers as a single heap block to avoid
    // overlap from fragmented free-list allocations.
    if (tid == 0) {
        unsigned int sa_bytes = (unsigned int)(n_sa * 3) * (unsigned int)sizeof(float);
        unsigned int ta_bytes = (unsigned int)n_sa * (unsigned int)sizeof(int);
        unsigned int sb_bytes = (unsigned int)(n_sb * 3) * (unsigned int)sizeof(float);
        unsigned int tb_bytes = (unsigned int)n_sb * (unsigned int)sizeof(int);
        unsigned int total = sa_bytes + ta_bytes + sb_bytes + tb_bytes;
        void* buf = NULL;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, total, &buf) != HEAP_OK)
            { s_alloc_ok = 0; buf = NULL; }
        char* p = (char*)buf;
        s_samples_a = (float*)p;               p += sa_bytes;
        s_tri_ids_a = (int*)p;                 p += ta_bytes;
        s_samples_b = (float*)p;               p += sb_bytes;
        s_tri_ids_b = (int*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
        return 0.0f;
    }

    PC_BUF(float, samples_a, s_samples_a, n_sa * 3);
    PC_BUF(int,   tri_ids_a, s_tri_ids_a, n_sa);
    PC_BUF(float, samples_b, s_samples_b, n_sb * 3);
    PC_BUF(int,   tri_ids_b, s_tri_ids_b, n_sb);

    // Generate samples for mesh A.
    for (int t = tid; t < a->nt; t += HD_BLOCK) {
        int off = counts_a[t];
        int next_off = (t + 1 < a->nt) ? counts_a[t + 1] : n_sa;
        int N = next_off - off;
        int ia = at[t * 3], ib = at[t * 3 + 1], ic = at[t * 3 + 2];
        float p0x = av[ia*3], p0y = av[ia*3+1], p0z = av[ia*3+2];
        float p1x = av[ib*3], p1y = av[ib*3+1], p1z = av[ib*3+2];
        float p2x = av[ic*3], p2y = av[ic*3+1], p2z = av[ic*3+2];
        for (int k = 0; k < N; k++) {
            unsigned int ha = hd_wang_hash((unsigned int)t * 65537u + (unsigned int)k);
            unsigned int hb = hd_wang_hash(ha + 0x9e3779b9u);
            float ra = (float)(ha & 0xFFFFFFu) / (float)0xFFFFFFu;
            float rb = (float)(hb & 0xFFFFFFu) / (float)0xFFFFFFu;
            float sqa = sqrtf(ra);
            int idx = off + k;
            samples_a[idx * 3 + 0] = (1.0f - sqa) * p0x + sqa * (1.0f - rb) * p1x + sqa * rb * p2x;
            samples_a[idx * 3 + 1] = (1.0f - sqa) * p0y + sqa * (1.0f - rb) * p1y + sqa * rb * p2y;
            samples_a[idx * 3 + 2] = (1.0f - sqa) * p0z + sqa * (1.0f - rb) * p1z + sqa * rb * p2z;
            tri_ids_a[idx] = t;
        }
    }

    // Generate samples for mesh B.
    for (int t = tid; t < b->nt; t += HD_BLOCK) {
        int off = counts_b[t];
        int next_off = (t + 1 < b->nt) ? counts_b[t + 1] : n_sb;
        int N = next_off - off;
        int ia = bt[t * 3], ib = bt[t * 3 + 1], ic = bt[t * 3 + 2];
        float p0x = bv[ia*3], p0y = bv[ia*3+1], p0z = bv[ia*3+2];
        float p1x = bv[ib*3], p1y = bv[ib*3+1], p1z = bv[ib*3+2];
        float p2x = bv[ic*3], p2y = bv[ic*3+1], p2z = bv[ic*3+2];
        for (int k = 0; k < N; k++) {
            unsigned int ha = hd_wang_hash((unsigned int)t * 65537u + (unsigned int)k);
            unsigned int hb = hd_wang_hash(ha + 0x9e3779b9u);
            float ra = (float)(ha & 0xFFFFFFu) / (float)0xFFFFFFu;
            float rb = (float)(hb & 0xFFFFFFu) / (float)0xFFFFFFu;
            float sqa = sqrtf(ra);
            int idx = off + k;
            samples_b[idx * 3 + 0] = (1.0f - sqa) * p0x + sqa * (1.0f - rb) * p1x + sqa * rb * p2x;
            samples_b[idx * 3 + 1] = (1.0f - sqa) * p0y + sqa * (1.0f - rb) * p1y + sqa * rb * p2y;
            samples_b[idx * 3 + 2] = (1.0f - sqa) * p0z + sqa * (1.0f - rb) * p1z + sqa * rb * p2z;
            tri_ids_b[idx] = t;
        }
    }
    __syncthreads();

    // Free count arrays — no longer needed.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_counts_a); s_counts_a = NULL;
        heap_free(scratch_heap, (void*)s_counts_b); s_counts_b = NULL;
    }
    __syncthreads();

    // =================================================================
    // Phase 3: Dispatch brute force or BVH per direction.
    // =================================================================

    // Direction A->B: query samples_a against mesh b's triangles.
    if (b->nt <= HD_BRUTE_THRESH) {
        // Brute force.
        float local_max = 0.0f;
        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float qx = samples_a[i * 3], qy = samples_a[i * 3 + 1], qz = samples_a[i * 3 + 2];
            float best = 1e30f;
            for (int t = 0; t < b->nt; t++) {
                int i0 = bt[t*3+0], i1 = bt[t*3+1], i2 = bt[t*3+2];
                float d = hd_dist_pt_tri_sq(qx,qy,qz,
                    bv[i0*3],bv[i0*3+1],bv[i0*3+2],
                    bv[i1*3],bv[i1*3+1],bv[i1*3+2],
                    bv[i2*3],bv[i2*3+1],bv[i2*3+2]);
                best = fminf(best, d);
            }
            local_max = fmaxf(local_max, best);
        }
        float dir_ab = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ab = dir_ab;
        __syncthreads();
    }

    // Direction B->A: query samples_b against mesh a's triangles.
    if (a->nt <= HD_BRUTE_THRESH) {
        float local_max = 0.0f;
        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float qx = samples_b[i * 3], qy = samples_b[i * 3 + 1], qz = samples_b[i * 3 + 2];
            float best = 1e30f;
            for (int t = 0; t < a->nt; t++) {
                int i0 = at[t*3+0], i1 = at[t*3+1], i2 = at[t*3+2];
                float d = hd_dist_pt_tri_sq(qx,qy,qz,
                    av[i0*3],av[i0*3+1],av[i0*3+2],
                    av[i1*3],av[i1*3+1],av[i1*3+2],
                    av[i2*3],av[i2*3+1],av[i2*3+2]);
                best = fminf(best, d);
            }
            local_max = fmaxf(local_max, best);
        }
        float dir_ba = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ba = dir_ba;
        __syncthreads();
    }

    // If both directions used brute force, we're done.
    if (a->nt <= HD_BRUTE_THRESH && b->nt <= HD_BRUTE_THRESH) {
        float result = sqrtf(fmaxf(s_dir_ab, s_dir_ba));
        if (tid == 0) {
            HD_FREE_ALL_SCRATCH();
        }
        return result;
    }

    // =================================================================
    // Phase 4: Morton code computation (for BVH paths).
    // =================================================================

    // Compute combined bounding box of all samples.
    if (tid < 3) { s_bbox_min[tid] = 1e30f; s_bbox_max[tid] = -1e30f; }
    __syncthreads();

    {
        float tlo[3] = { 1e30f, 1e30f, 1e30f };
        float thi[3] = { -1e30f, -1e30f, -1e30f };

        // Samples A.
        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float x = samples_a[i * 3], y = samples_a[i * 3 + 1], z = samples_a[i * 3 + 2];
            tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
            tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
            tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
        }
        // Samples B.
        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float x = samples_b[i * 3], y = samples_b[i * 3 + 1], z = samples_b[i * 3 + 2];
            tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
            tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
            tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
        }
        // Warp reduce.
        int lane = tid & 31;
        for (int ax = 0; ax < 3; ax++) {
            tlo[ax] = warp_min_f(tlo[ax]);
            thi[ax] = warp_max_f(thi[ax]);
        }
        if (lane == 0) {
            atomicMinF(&s_bbox_min[0], tlo[0]); atomicMaxF(&s_bbox_max[0], thi[0]);
            atomicMinF(&s_bbox_min[1], tlo[1]); atomicMaxF(&s_bbox_max[1], thi[1]);
            atomicMinF(&s_bbox_min[2], tlo[2]); atomicMaxF(&s_bbox_max[2], thi[2]);
        }
    }
    __syncthreads();

    float bb_min[3] = { s_bbox_min[0], s_bbox_min[1], s_bbox_min[2] };
    float bb_ext[3] = {
        s_bbox_max[0] - s_bbox_min[0],
        s_bbox_max[1] - s_bbox_min[1],
        s_bbox_max[2] - s_bbox_min[2]
    };
    // Prevent division by zero.
    for (int ax = 0; ax < 3; ax++)
        if (bb_ext[ax] < 1e-20f) bb_ext[ax] = 1e-20f;

    // Determine which directions need BVH.
    bool need_bvh_b = (b->nt > HD_BRUTE_THRESH);  // for A->B direction
    bool need_bvh_a = (a->nt > HD_BRUTE_THRESH);  // for B->A direction

    // Allocate morton arrays for whichever we need.
    if (tid == 0) {
        void *p1 = NULL, *p2 = NULL;
        s_alloc_ok = 1;
        if (need_bvh_b) {
            if (heap_alloc(scratch_heap, (unsigned int)(n_sb * (int)sizeof(MortonPoint)), &p1) != HEAP_OK)
                { s_alloc_ok = 0; p1 = NULL; }
        }
        if (need_bvh_a) {
            if (heap_alloc(scratch_heap, (unsigned int)(n_sa * (int)sizeof(MortonPoint)), &p2) != HEAP_OK)
                { s_alloc_ok = 0; p2 = NULL; }
        }
        s_morton_b = (MortonPoint*)p1;  // BVH over B's samples (for A->B query)
        s_morton_a = (MortonPoint*)p2;  // BVH over A's samples (for B->A query)
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
        return 0.0f;
    }

    // Compute Morton codes.
    PC_BUF(MortonPoint, morton_b, s_morton_b, need_bvh_b ? n_sb : 0);
    PC_BUF(MortonPoint, morton_a, s_morton_a, need_bvh_a ? n_sa : 0);
    if (need_bvh_b) {
        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float x = samples_b[i * 3], y = samples_b[i * 3 + 1], z = samples_b[i * 3 + 2];
            unsigned int mx = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (x - bb_min[0]) / bb_ext[0] * 1023.0f));
            unsigned int my = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (y - bb_min[1]) / bb_ext[1] * 1023.0f));
            unsigned int mz = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (z - bb_min[2]) / bb_ext[2] * 1023.0f));
            morton_b[i].code = hd_morton3D(mx, my, mz);
            morton_b[i].index = i;
        }
    }
    if (need_bvh_a) {
        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float x = samples_a[i * 3], y = samples_a[i * 3 + 1], z = samples_a[i * 3 + 2];
            unsigned int mx = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (x - bb_min[0]) / bb_ext[0] * 1023.0f));
            unsigned int my = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (y - bb_min[1]) / bb_ext[1] * 1023.0f));
            unsigned int mz = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (z - bb_min[2]) / bb_ext[2] * 1023.0f));
            morton_a[i].code = hd_morton3D(mx, my, mz);
            morton_a[i].index = i;
        }
    }
    __syncthreads();

    // =================================================================
    // Phase 5: Cooperative sort of Morton codes.
    //
    // 8 warps total. Warps 0-3 handle the first needed sort,
    // warps 4-7 handle the second (if both needed).
    // If only one direction needs BVH, all 8 warps sort that one set.
    // =================================================================

    {
        // Determine which sets to sort and with which warps.
        MortonPoint* sort_data[2] = { NULL, NULL };
        int sort_n[2] = { 0, 0 };
        int n_sets = 0;
        if (need_bvh_b) { sort_data[n_sets] = s_morton_b; sort_n[n_sets] = n_sb; n_sets++; }
        if (need_bvh_a) { sort_data[n_sets] = s_morton_a; sort_n[n_sets] = n_sa; n_sets++; }

        // Allocate sort scratch.
        int max_n = 0;
        for (int i = 0; i < n_sets; i++)
            if (sort_n[i] > max_n) max_n = sort_n[i];

        int scratch_per_set = max_n * (int)sizeof(MortonPoint) +
                              4 * WS_MAX_STACK * 2 * (int)sizeof(int);
        int total_scratch = scratch_per_set * n_sets;

        if (tid == 0) {
            void* p = NULL;
            s_alloc_ok = 1;
            if (heap_alloc(scratch_heap, (unsigned int)total_scratch, &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_sort_scratch = (char*)p;
        }
        __syncthreads();
        if (!s_alloc_ok) {
            if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
            return 0.0f;
        }

        int warp_id = tid / WARP_SIZE;
        int lane    = tid & (WARP_SIZE - 1);

        if (n_sets == 1) {
            // All 8 warps cooperate on a single sort: 2 partitions -> 4 segments -> 4 warps sort.
            // (Warps 4-7 idle during partition, active during sort of segments 4-7 if needed.)
            MortonPoint* data = sort_data[0];
            int n = sort_n[0];
            char* scratch_base = s_sort_scratch;
            MortonPoint* tmp = (MortonPoint*)scratch_base;
            int* stacks = (int*)(scratch_base + n * (int)sizeof(MortonPoint));

            if (n <= 32) {
                if (warp_id == 0) {
                    MortonPoint val = (lane < n) ? data[lane] : MortonPointCmp::sentinel();
                    val = warp_bitonic32_t<MortonPoint, MortonPointCmp>(val, n, lane);
                    if (lane < n) data[lane] = val;
                }
            } else {
                // Warp 0: partition twice to get 4 segments.
                if (warp_id == 0) {
                    MortonPoint piv1 = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, 0, n, lane);
                    int ml1, mh1;
                    warp_partition<MortonPoint, MortonPointCmp>(data, tmp, 0, n, piv1, &ml1, &mh1, lane);

                    int ml2 = 0, mh2 = 0;
                    if (ml1 > 1) {
                        MortonPoint pivL = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, 0, ml1, lane);
                        warp_partition<MortonPoint, MortonPointCmp>(data, tmp, 0, ml1, pivL, &ml2, &mh2, lane);
                    }

                    int ml3 = mh1, mh3 = mh1;
                    if (n - mh1 > 1) {
                        MortonPoint pivR = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, mh1, n - mh1, lane);
                        warp_partition<MortonPoint, MortonPointCmp>(data, tmp, mh1, n, pivR, &ml3, &mh3, lane);
                    }

                    if (lane == 0) {
                        // 4 segments: [0,ml2), [mh2,ml1), [mh1,ml3), [mh3,n)
                        s_seg_a[0] = 0;     s_seg_a[1] = ml2;
                        s_seg_a[2] = mh2;   s_seg_a[3] = ml1;
                        s_seg_a[4] = mh1;   s_seg_a[5] = ml3;
                        // Segment 3 is [mh3, n) — store mh3 in s_seg_b[0] as overflow.
                        s_seg_b[0] = mh3;   s_seg_b[1] = n;
                    }
                }
            }
            __syncthreads();

            if (n > 32) {
                // 4 segments stored as pairs: (s_seg_a[0],s_seg_a[1]), (s_seg_a[2],s_seg_a[3]),
                //                             (s_seg_a[4],s_seg_a[5]), (s_seg_b[0],s_seg_b[1])
                int seg_lo, seg_hi;
                if (warp_id == 0)      { seg_lo = s_seg_a[0]; seg_hi = s_seg_a[1]; }
                else if (warp_id == 1) { seg_lo = s_seg_a[2]; seg_hi = s_seg_a[3]; }
                else if (warp_id == 2) { seg_lo = s_seg_a[4]; seg_hi = s_seg_a[5]; }
                else if (warp_id == 3) { seg_lo = s_seg_b[0]; seg_hi = s_seg_b[1]; }
                else { seg_lo = 0; seg_hi = 0; } // warps 4-7 idle

                if (warp_id < 4 && seg_hi - seg_lo > 1) {
                    int* my_stack_lo = stacks + warp_id * WS_MAX_STACK * 2;
                    int* my_stack_hi = my_stack_lo + WS_MAX_STACK;
                    int serr = warp_sort_inner<MortonPoint, MortonPointCmp>(
                        data, tmp, my_stack_lo, my_stack_hi, seg_lo, seg_hi, lane);
                    if (serr && lane == 0) atomicOr(kernel_error, HD_KERR_SORT_ERR);
                }
            }
        } else {
            // n_sets == 2: warps 0-3 sort set 0, warps 4-7 sort set 1.
            int set_idx = (warp_id < 4) ? 0 : 1;
            int local_warp = warp_id & 3;
            MortonPoint* data = sort_data[set_idx];
            int n = sort_n[set_idx];
            char* scratch_base = s_sort_scratch + set_idx * scratch_per_set;
            MortonPoint* tmp = (MortonPoint*)scratch_base;
            int* stacks = (int*)(scratch_base + sort_n[set_idx] * (int)sizeof(MortonPoint));
            int* my_segs = (set_idx == 0) ? s_seg_a : s_seg_b;

            if (n <= 32) {
                if (local_warp == 0) {
                    MortonPoint val = (lane < n) ? data[lane] : MortonPointCmp::sentinel();
                    val = warp_bitonic32_t<MortonPoint, MortonPointCmp>(val, n, lane);
                    if (lane < n) data[lane] = val;
                }
            } else {
                // Leader warp partitions twice.
                if (local_warp == 0) {
                    MortonPoint piv1 = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, 0, n, lane);
                    int ml1, mh1;
                    warp_partition<MortonPoint, MortonPointCmp>(data, tmp, 0, n, piv1, &ml1, &mh1, lane);

                    int ml2 = 0, mh2 = 0;
                    if (ml1 > 1) {
                        MortonPoint pivL = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, 0, ml1, lane);
                        warp_partition<MortonPoint, MortonPointCmp>(data, tmp, 0, ml1, pivL, &ml2, &mh2, lane);
                    }

                    int ml3 = mh1, mh3 = mh1;
                    if (n - mh1 > 1) {
                        MortonPoint pivR = warp_pick_pivot<MortonPoint, MortonPointCmp>(data, mh1, n - mh1, lane);
                        warp_partition<MortonPoint, MortonPointCmp>(data, tmp, mh1, n, pivR, &ml3, &mh3, lane);
                    }

                    if (lane == 0) {
                        my_segs[0] = 0;     my_segs[1] = ml2;
                        my_segs[2] = mh2;   my_segs[3] = ml1;
                        my_segs[4] = mh1;   my_segs[5] = ml3;
                        // Stash mh3, n in shared memory — use reduce_i as temp storage.
                        // (reduce_i is not in use right now.)
                        if (set_idx == 0) {
                            s_reduce_i[0] = mh3;
                            s_reduce_i[1] = n;
                        } else {
                            s_reduce_i[2] = mh3;
                            s_reduce_i[3] = n;
                        }
                    }
                }
            }
            __syncthreads();

            if (n > 32) {
                int seg_lo, seg_hi;
                int mh3_val, n_val;
                if (set_idx == 0) { mh3_val = s_reduce_i[0]; n_val = s_reduce_i[1]; }
                else              { mh3_val = s_reduce_i[2]; n_val = s_reduce_i[3]; }

                if (local_warp == 0)      { seg_lo = my_segs[0]; seg_hi = my_segs[1]; }
                else if (local_warp == 1) { seg_lo = my_segs[2]; seg_hi = my_segs[3]; }
                else if (local_warp == 2) { seg_lo = my_segs[4]; seg_hi = my_segs[5]; }
                else                      { seg_lo = mh3_val;    seg_hi = n_val; }

                if (seg_hi - seg_lo > 1) {
                    int* my_stack_lo = stacks + local_warp * WS_MAX_STACK * 2;
                    int* my_stack_hi = my_stack_lo + WS_MAX_STACK;
                    int serr = warp_sort_inner<MortonPoint, MortonPointCmp>(
                        data, tmp, my_stack_lo, my_stack_hi, seg_lo, seg_hi, lane);
                    if (serr && lane == 0) atomicOr(kernel_error, HD_KERR_SORT_ERR);
                }
            }
        }
    }
    __syncthreads();

    // Free sort scratch.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_sort_scratch); s_sort_scratch = NULL;
    }
    __syncthreads();

    // =================================================================
    // Phase 6: Linear BVH construction (Karras 2012).
    // Build BVH(s) sequentially (all 256 threads on each).
    // =================================================================

    // Helper lambda-like: build BVH over morton[] with n elements, using
    // tri_ids[] and target_mesh for leaf AABBs. Writes into bvh_nodes[].
    // bvh_counters[] has n-1 ints for bottom-up propagation.

    // We'll do up to 2 BVH builds. Allocate the larger one first.
    // BVH for B's samples (if need_bvh_b): used for A->B direction.
    // BVH for A's samples (if need_bvh_a): used for B->A direction.

    if (need_bvh_b && n_sb > 1) {
        int bvh_nodes_count = 2 * n_sb - 1;
        int counter_count   = n_sb - 1;

        if (tid == 0) {
            void *p1 = NULL, *p2 = NULL;
            s_alloc_ok = 1;
            if (heap_alloc(scratch_heap, (unsigned int)(bvh_nodes_count * (int)sizeof(BVHNode)), &p1) != HEAP_OK)
                { s_alloc_ok = 0; p1 = NULL; }
            if (heap_alloc(scratch_heap, (unsigned int)(counter_count * (int)sizeof(int)), &p2) != HEAP_OK)
                { s_alloc_ok = 0; p2 = NULL; }
            s_bvh_b = (BVHNode*)p1;
            s_bvh_counters = (int*)p2;
        }
        __syncthreads();
        if (!s_alloc_ok) {
            if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
            return 0.0f;
        }

        int n = n_sb;
        BVHNode* bvh = s_bvh_b;
        MortonPoint* morton = s_morton_b;
        int* counters = s_bvh_counters;

        // Init counters and parent pointers.
        for (int i = tid; i < n - 1; i += HD_BLOCK) {
            counters[i] = 0;
            bvh[i].parent = -1;
        }
        // Init leaf parents.
        for (int i = tid; i < n; i += HD_BLOCK) {
            bvh[(n - 1) + i].parent = -1;
            bvh[(n - 1) + i].left = -1;
            bvh[(n - 1) + i].right = -1;
        }
        __syncthreads();

        // Step A: Initialize leaf AABBs.
        for (int i = tid; i < n; i += HD_BLOCK) {
            int leaf_idx = (n - 1) + i;
            int tri = tri_ids_b[morton[i].index];
            int ia = bt[tri * 3 + 0], ib_t = bt[tri * 3 + 1], ic = bt[tri * 3 + 2];
            float ax = bv[ia*3], ay = bv[ia*3+1], az = bv[ia*3+2];
            float bx = bv[ib_t*3], by = bv[ib_t*3+1], bz = bv[ib_t*3+2];
            float cx = bv[ic*3], cy = bv[ic*3+1], cz = bv[ic*3+2];
            bvh[leaf_idx].bmin[0] = fminf(ax, fminf(bx, cx));
            bvh[leaf_idx].bmin[1] = fminf(ay, fminf(by, cy));
            bvh[leaf_idx].bmin[2] = fminf(az, fminf(bz, cz));
            bvh[leaf_idx].bmax[0] = fmaxf(ax, fmaxf(bx, cx));
            bvh[leaf_idx].bmax[1] = fmaxf(ay, fmaxf(by, cy));
            bvh[leaf_idx].bmax[2] = fmaxf(az, fmaxf(bz, cz));
            bvh[leaf_idx].tri_idx = tri;
        }
        __syncthreads();

        // Step B: Build internal nodes (Karras 2012).
        for (int i = tid; i < n - 1; i += HD_BLOCK) {
            int d_val = hd_delta(morton, n, i, i + 1) - hd_delta(morton, n, i, i - 1);
            int d = (d_val > 0) ? 1 : -1;
            int delta_min = hd_delta(morton, n, i, i - d);

            int l_max = 2;
            while (hd_delta(morton, n, i, i + l_max * d) > delta_min)
                l_max <<= 1;

            int l = 0;
            for (int t = l_max >> 1; t >= 1; t >>= 1) {
                if (hd_delta(morton, n, i, i + (l + t) * d) > delta_min)
                    l += t;
            }
            int j = i + l * d;

            int delta_node = hd_delta(morton, n, i, j);
            int s = 0;
            int max_len = (j > i) ? (j - i) : (i - j);
            // Start t at the largest power of 2 <= max_len so the binary
            // search can express every integer in [0, max_len].  The old
            // (max_len+1)>>1 start skipped valid positions when max_len
            // was not a power of two.
            int t_start = 1;
            while (t_start * 2 <= max_len) t_start *= 2;
            for (int t = t_start; t >= 1; t >>= 1) {
                if (hd_delta(morton, n, i, i + (s + t) * d) > delta_node)
                    s += t;
            }
            int gamma = i + s * d + min(d, 0);

            int left_child, right_child;
            int range_min = min(i, j);
            int range_max = max(i, j);
            left_child  = (range_min == gamma)     ? (n - 1) + gamma     : gamma;
            right_child = (range_max == gamma + 1) ? (n - 1) + gamma + 1 : gamma + 1;

            bvh[i].left  = left_child;
            bvh[i].right = right_child;
            bvh[left_child].parent  = i;
            bvh[right_child].parent = i;
        }
        __syncthreads();

        // Step C: Bottom-up AABB propagation.
        for (int i = tid; i < n; i += HD_BLOCK) {
            int node = bvh[(n - 1) + i].parent;
            while (node >= 0) {
                int old = atomicAdd(&counters[node], 1);
                if (old == 0) break;  // first visitor, second child not ready
                // Second visitor: merge.
                int lc = bvh[node].left;
                int rc = bvh[node].right;
                for (int ax = 0; ax < 3; ax++) {
                    bvh[node].bmin[ax] = fminf(bvh[lc].bmin[ax], bvh[rc].bmin[ax]);
                    bvh[node].bmax[ax] = fmaxf(bvh[lc].bmax[ax], bvh[rc].bmax[ax]);
                }
                __threadfence_block();
                node = bvh[node].parent;
            }
        }
        __syncthreads();

        // Free counters.
        if (tid == 0) {
            heap_free(scratch_heap, (void*)s_bvh_counters); s_bvh_counters = NULL;
        }
        __syncthreads();

    }

    if (need_bvh_a && n_sa > 1) {
        int bvh_nodes_count = 2 * n_sa - 1;
        int counter_count   = n_sa - 1;

        if (tid == 0) {
            void *p1 = NULL, *p2 = NULL;
            s_alloc_ok = 1;
            if (heap_alloc(scratch_heap, (unsigned int)(bvh_nodes_count * (int)sizeof(BVHNode)), &p1) != HEAP_OK)
                { s_alloc_ok = 0; p1 = NULL; }
            if (heap_alloc(scratch_heap, (unsigned int)(counter_count * (int)sizeof(int)), &p2) != HEAP_OK)
                { s_alloc_ok = 0; p2 = NULL; }
            s_bvh_a = (BVHNode*)p1;
            s_bvh_counters = (int*)p2;
        }
        __syncthreads();
        if (!s_alloc_ok) {
            if (tid == 0) { HD_FREE_ALL_SCRATCH(); atomicOr(kernel_error, HD_KERR_SCRATCH_OOM); }
            return 0.0f;
        }

        int n = n_sa;
        BVHNode* bvh = s_bvh_a;
        MortonPoint* morton = s_morton_a;
        int* counters = s_bvh_counters;

        for (int i = tid; i < n - 1; i += HD_BLOCK) {
            counters[i] = 0;
            bvh[i].parent = -1;
        }
        for (int i = tid; i < n; i += HD_BLOCK) {
            bvh[(n - 1) + i].parent = -1;
            bvh[(n - 1) + i].left = -1;
            bvh[(n - 1) + i].right = -1;
        }
        __syncthreads();

        // Leaf AABBs — from mesh A's triangles.
        for (int i = tid; i < n; i += HD_BLOCK) {
            int leaf_idx = (n - 1) + i;
            int tri = tri_ids_a[morton[i].index];
            int ia = at[tri * 3 + 0], ib_t = at[tri * 3 + 1], ic = at[tri * 3 + 2];
            float ax = av[ia*3], ay = av[ia*3+1], az = av[ia*3+2];
            float bx = av[ib_t*3], by = av[ib_t*3+1], bz = av[ib_t*3+2];
            float cx = av[ic*3], cy = av[ic*3+1], cz = av[ic*3+2];
            bvh[leaf_idx].bmin[0] = fminf(ax, fminf(bx, cx));
            bvh[leaf_idx].bmin[1] = fminf(ay, fminf(by, cy));
            bvh[leaf_idx].bmin[2] = fminf(az, fminf(bz, cz));
            bvh[leaf_idx].bmax[0] = fmaxf(ax, fmaxf(bx, cx));
            bvh[leaf_idx].bmax[1] = fmaxf(ay, fmaxf(by, cy));
            bvh[leaf_idx].bmax[2] = fmaxf(az, fmaxf(bz, cz));
            bvh[leaf_idx].tri_idx = tri;
        }
        __syncthreads();

        // Build internal nodes (serial — see note in first BVH block).
        for (int i = tid; i < n - 1; i += HD_BLOCK) {
            int d_val = hd_delta(morton, n, i, i + 1) - hd_delta(morton, n, i, i - 1);
            int d = (d_val > 0) ? 1 : -1;
            int delta_min = hd_delta(morton, n, i, i - d);

            int l_max = 2;
            while (hd_delta(morton, n, i, i + l_max * d) > delta_min)
                l_max <<= 1;

            int l = 0;
            for (int t = l_max >> 1; t >= 1; t >>= 1) {
                if (hd_delta(morton, n, i, i + (l + t) * d) > delta_min)
                    l += t;
            }
            int j = i + l * d;

            int delta_node = hd_delta(morton, n, i, j);
            int s = 0;
            int max_len = (j > i) ? (j - i) : (i - j);
            // Start t at the largest power of 2 <= max_len so the binary
            // search can express every integer in [0, max_len].  The old
            // (max_len+1)>>1 start skipped valid positions when max_len
            // was not a power of two.
            int t_start = 1;
            while (t_start * 2 <= max_len) t_start *= 2;
            for (int t = t_start; t >= 1; t >>= 1) {
                if (hd_delta(morton, n, i, i + (s + t) * d) > delta_node)
                    s += t;
            }
            int gamma = i + s * d + min(d, 0);

            int left_child, right_child;
            int range_min = min(i, j);
            int range_max = max(i, j);
            left_child  = (range_min == gamma)     ? (n - 1) + gamma     : gamma;
            right_child = (range_max == gamma + 1) ? (n - 1) + gamma + 1 : gamma + 1;

            bvh[i].left  = left_child;
            bvh[i].right = right_child;
            bvh[left_child].parent  = i;
            bvh[right_child].parent = i;
        }
        __syncthreads();

        // Bottom-up AABB propagation.
        for (int i = tid; i < n; i += HD_BLOCK) {
            int node = bvh[(n - 1) + i].parent;
            while (node >= 0) {
                int old = atomicAdd(&counters[node], 1);
                if (old == 0) break;
                int lc = bvh[node].left;
                int rc = bvh[node].right;
                for (int ax = 0; ax < 3; ax++) {
                    bvh[node].bmin[ax] = fminf(bvh[lc].bmin[ax], bvh[rc].bmin[ax]);
                    bvh[node].bmax[ax] = fmaxf(bvh[lc].bmax[ax], bvh[rc].bmax[ax]);
                }
                __threadfence_block();
                node = bvh[node].parent;
            }
        }
        __syncthreads();

        if (tid == 0) {
            heap_free(scratch_heap, (void*)s_bvh_counters); s_bvh_counters = NULL;
        }
        __syncthreads();

    }

    // Free morton arrays and tri_ids — no longer needed.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_morton_a); s_morton_a = NULL;
        heap_free(scratch_heap, (void*)s_morton_b); s_morton_b = NULL;
        // s_tri_ids_a/b are part of the single samples allocation — not freed separately
    }
    __syncthreads();

    // =================================================================
    // Phase 7: BVH traversal — nearest-neighbor queries.
    // =================================================================

    // Direction A->B (query samples_a against BVH built on B's samples).
    if (need_bvh_b && n_sb > 1) {
        BVHNode* bvh = s_bvh_b;
        float local_max = 0.0f;

        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float qx = samples_a[i * 3], qy = samples_a[i * 3 + 1], qz = samples_a[i * 3 + 2];
            float best_sq = 1e30f;

            int stack[HD_BVH_STACK];
            int sp = 0;
            stack[sp++] = 0;  // root

            while (sp > 0) {
                int idx = stack[--sp];
                BVHNode* nd = &bvh[idx];

                float aabb_d = hd_pt_aabb_dist_sq(qx, qy, qz, nd->bmin, nd->bmax);
                if (aabb_d >= best_sq) continue;

                if (nd->left == -1) {
                    // Leaf: exact distance to source triangle.
                    int ti = nd->tri_idx;
                    int i0 = bt[ti*3+0], i1 = bt[ti*3+1], i2 = bt[ti*3+2];
                    float d = hd_dist_pt_tri_sq(qx,qy,qz,
                        bv[i0*3],bv[i0*3+1],bv[i0*3+2],
                        bv[i1*3],bv[i1*3+1],bv[i1*3+2],
                        bv[i2*3],bv[i2*3+1],bv[i2*3+2]);
                    best_sq = fminf(best_sq, d);
                } else {
                    // Push children, farther first.
                    float dL = hd_pt_aabb_dist_sq(qx, qy, qz, bvh[nd->left].bmin, bvh[nd->left].bmax);
                    float dR = hd_pt_aabb_dist_sq(qx, qy, qz, bvh[nd->right].bmin, bvh[nd->right].bmax);
                    if (dL < dR) {
                        if (dR < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->right;
                        if (dL < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->left;
                    } else {
                        if (dL < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->left;
                        if (dR < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->right;
                    }
                }
            }
            local_max = fmaxf(local_max, best_sq);
        }

        float dir_ab = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ab = dir_ab;
        __syncthreads();
    } else if (need_bvh_b && n_sb <= 1) {
        // Too few samples for BVH — brute force against all target triangles.
        float local_max = 0.0f;
        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float qx = samples_a[i * 3], qy = samples_a[i * 3 + 1], qz = samples_a[i * 3 + 2];
            float best = 1e30f;
            for (int t = 0; t < b->nt; t++) {
                int i0 = bt[t*3+0], i1 = bt[t*3+1], i2 = bt[t*3+2];
                float d = hd_dist_pt_tri_sq(qx,qy,qz,
                    bv[i0*3],bv[i0*3+1],bv[i0*3+2],
                    bv[i1*3],bv[i1*3+1],bv[i1*3+2],
                    bv[i2*3],bv[i2*3+1],bv[i2*3+2]);
                best = fminf(best, d);
            }
            local_max = fmaxf(local_max, best);
        }
        float dir_ab = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ab = dir_ab;
        __syncthreads();
    }

    // Direction B->A (query samples_b against BVH built on A's samples).
    if (need_bvh_a && n_sa > 1) {
        BVHNode* bvh = s_bvh_a;
        float local_max = 0.0f;

        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float qx = samples_b[i * 3], qy = samples_b[i * 3 + 1], qz = samples_b[i * 3 + 2];
            float best_sq = 1e30f;

            int stack[HD_BVH_STACK];
            int sp = 0;
            stack[sp++] = 0;

            while (sp > 0) {
                int idx = stack[--sp];
                BVHNode* nd = &bvh[idx];

                float aabb_d = hd_pt_aabb_dist_sq(qx, qy, qz, nd->bmin, nd->bmax);
                if (aabb_d >= best_sq) continue;

                if (nd->left == -1) {
                    int ti = nd->tri_idx;
                    int i0 = at[ti*3+0], i1 = at[ti*3+1], i2 = at[ti*3+2];
                    float d = hd_dist_pt_tri_sq(qx,qy,qz,
                        av[i0*3],av[i0*3+1],av[i0*3+2],
                        av[i1*3],av[i1*3+1],av[i1*3+2],
                        av[i2*3],av[i2*3+1],av[i2*3+2]);
                    best_sq = fminf(best_sq, d);
                } else {
                    float dL = hd_pt_aabb_dist_sq(qx, qy, qz, bvh[nd->left].bmin, bvh[nd->left].bmax);
                    float dR = hd_pt_aabb_dist_sq(qx, qy, qz, bvh[nd->right].bmin, bvh[nd->right].bmax);
                    if (dL < dR) {
                        if (dR < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->right;
                        if (dL < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->left;
                    } else {
                        if (dL < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->left;
                        if (dR < best_sq && sp < HD_BVH_STACK) stack[sp++] = nd->right;
                    }
                }
            }
            local_max = fmaxf(local_max, best_sq);
        }

        float dir_ba = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ba = dir_ba;
        __syncthreads();
    } else if (need_bvh_a && n_sa <= 1) {
        // Too few samples for BVH — brute force against all target triangles.
        float local_max = 0.0f;
        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float qx = samples_b[i * 3], qy = samples_b[i * 3 + 1], qz = samples_b[i * 3 + 2];
            float best = 1e30f;
            for (int t = 0; t < a->nt; t++) {
                int i0 = at[t*3+0], i1 = at[t*3+1], i2 = at[t*3+2];
                float d = hd_dist_pt_tri_sq(qx,qy,qz,
                    av[i0*3],av[i0*3+1],av[i0*3+2],
                    av[i1*3],av[i1*3+1],av[i1*3+2],
                    av[i2*3],av[i2*3+1],av[i2*3+2]);
                best = fminf(best, d);
            }
            local_max = fmaxf(local_max, best);
        }
        float dir_ba = block_reduce_max(local_max, s_reduce, tid);
        if (tid == 0) s_dir_ba = dir_ba;
        __syncthreads();
    }

    // =================================================================
    // Cleanup and return.
    // =================================================================

    float result = sqrtf(fmaxf(s_dir_ab, s_dir_ba));

    if (tid == 0) {
        HD_FREE_ALL_SCRATCH();
    }

    return result;
}
