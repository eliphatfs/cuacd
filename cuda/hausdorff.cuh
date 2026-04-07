// hausdorff.cuh — GPU bidirectional Hausdorff distance (sampling + linear BVH).
//
// One block (256 threads = 8 warps) computes the bidirectional Hausdorff
// distance between two meshes (typically a convex hull and its source part).
//
// Uses sampled point-point distance (matching CoACD): sample points on both
// meshes proportional to triangle area, then compute max-min point-point
// distance in both directions via linear BVH nearest-neighbor queries.
//
// Algorithm:
//   1  Compute per-triangle areas, prefix-sum sample counts   (256 threads)
//   2  Generate sample points via barycentric sampling         (256 threads)
//   3  Morton code computation                                 (256 threads)
//   4  Cooperative 4-warp sort of Morton codes (x2 sets)       (8 warps)
//   5  Linear BVH construction (Karras 2012)                   (256 threads)
//   6  BVH traversal — nearest-neighbor point queries          (256 threads)
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
#define HD_RESOLUTION  2000    // target sample count per mesh (CoACD default)
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
// BVH node — point-based (no triangle references)
// ============================================================================

struct BVHNode {
    float bmin[3];  // AABB min; for leaves, stores the sample point xyz
    float bmax[3];  // AABB max; for leaves, same as bmin (point AABB)
    int left;       // child index; -1 = leaf marker
    int right;
    int parent;
    int _pad;       // pad to 40 bytes (10 ints) for alignment
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
// Block-level prefix sum.
// counts[] is in global (scratch heap); smem_i is shared [HD_BLOCK].
// After call, counts[i] contains exclusive prefix sum (offset), and
// *total is the total count.
// ============================================================================

__device__ inline void hd_block_prefix_sum(
    int* counts, int n, int* smem_i, int tid, int* total)
{
    // Phase 1: each thread sums its strided chunk.
    int local_sum = 0;
    for (int i = tid; i < n; i += HD_BLOCK)
        local_sum += counts[i];
    smem_i[tid] = local_sum;
    __syncthreads();

    // Phase 2: exclusive scan over the 256 partial sums (Blelloch).
    // Up-sweep.
    for (int d = 1; d < HD_BLOCK; d <<= 1) {
        int ai = (tid + 1) * (d << 1) - 1;
        if (ai < HD_BLOCK)
            smem_i[ai] += smem_i[ai - d];
        __syncthreads();
    }
    if (tid == 0) {
        *total = smem_i[HD_BLOCK - 1];
        smem_i[HD_BLOCK - 1] = 0;
    }
    __syncthreads();
    // Down-sweep.
    for (int d = HD_BLOCK >> 1; d >= 1; d >>= 1) {
        int ai = (tid + 1) * (d << 1) - 1;
        if (ai < HD_BLOCK) {
            int tmp = smem_i[ai - d];
            smem_i[ai - d] = smem_i[ai];
            smem_i[ai] += tmp;
        }
        __syncthreads();
    }
    // smem_i[tid] now holds the exclusive prefix sum of the chunk sums.

    // Phase 3: each thread converts its chunk's counts to exclusive offsets.
    int chunk_base = smem_i[tid];
    int running = 0;
    for (int i = tid; i < n; i += HD_BLOCK) {
        int c = counts[i];
        counts[i] = chunk_base + running;
        running += c;
    }
    __syncthreads();
}

// ============================================================================
// Free-all macro for scratch cleanup on error.
// ============================================================================

#define HD_FREE_ALL_SCRATCH() do { \
    heap_free(scratch_heap, (void*)s_counts_a); \
    heap_free(scratch_heap, (void*)s_counts_b); \
    heap_free(scratch_heap, (void*)s_samples_a); \
    heap_free(scratch_heap, (void*)s_samples_b); \
    heap_free(scratch_heap, (void*)s_morton_a); \
    heap_free(scratch_heap, (void*)s_morton_b); \
    heap_free(scratch_heap, (void*)s_sort_scratch); \
    heap_free(scratch_heap, (void*)s_bvh_a); \
    heap_free(scratch_heap, (void*)s_bvh_b); \
    heap_free(scratch_heap, (void*)s_bvh_counters); \
} while(0)

// ============================================================================
// BVH build — NVIDIA blog approach (generateHierarchy + findSplit).
// Builds into bvh[] of size (2*n-1). Leaves at [n-1, 2n-2], internals at [0, n-2].
// ============================================================================

// findSplit: binary search for the highest object that shares more than
// commonPrefix bits with the first one (directly from NVIDIA blog).
__device__ inline int hd_find_split(unsigned int* sortedCodes, int first, int last) {
    unsigned int firstCode = sortedCodes[first];
    unsigned int lastCode  = sortedCodes[last];

    // Identical Morton codes => split in the middle.
    if (firstCode == lastCode)
        return (first + last) >> 1;

    int commonPrefix = __clz(firstCode ^ lastCode);

    int split = first;
    int step  = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            unsigned int splitCode = sortedCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit;
        }
    } while (step > 1);

    return split;
}

// determineRange: find which range of objects internal node idx covers.
// Returns (first, last) with first <= last.
__device__ inline void hd_determine_range(
    const MortonPoint* morton, int n, int idx, int* out_first, int* out_last)
{
    // Determine direction of the range.
    int d_val = hd_delta(morton, n, idx, idx + 1) - hd_delta(morton, n, idx, idx - 1);
    int d = (d_val > 0) ? 1 : -1;
    int delta_min = hd_delta(morton, n, idx, idx - d);

    // Compute upper bound for range length.
    int l_max = 2;
    while (hd_delta(morton, n, idx, idx + l_max * d) > delta_min)
        l_max <<= 1;

    // Binary search for exact range end.
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (hd_delta(morton, n, idx, idx + (l + t) * d) > delta_min)
            l += t;
    }
    int j = idx + l * d;

    *out_first = min(idx, j);
    *out_last  = max(idx, j);
}

__device__ inline void hd_bvh_build(
    BVHNode* bvh, MortonPoint* morton, float* samples,
    int* counters, int n, int tid)
{
    // Extract sorted Morton codes into a contiguous unsigned int array
    // (needed by findSplit which takes unsigned int*). We reuse the
    // morton[].code field but need a contiguous code array. Since
    // MortonPoint is {code, index}, morton codes are at stride 2 ints.
    // For findSplit we need codes[i] = morton[i].code.
    // We can just access morton[i].code directly in a modified findSplit.
    // But the NVIDIA blog findSplit takes unsigned int*. Let's just
    // access morton[].code inline.

    // Init counters and leaf markers.
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

    // Step A: Initialize leaf AABBs from sample points.
    for (int i = tid; i < n; i += HD_BLOCK) {
        int leaf_idx = (n - 1) + i;
        int si = morton[i].index;
        float px = samples[si * 3 + 0];
        float py = samples[si * 3 + 1];
        float pz = samples[si * 3 + 2];
        bvh[leaf_idx].bmin[0] = px; bvh[leaf_idx].bmax[0] = px;
        bvh[leaf_idx].bmin[1] = py; bvh[leaf_idx].bmax[1] = py;
        bvh[leaf_idx].bmin[2] = pz; bvh[leaf_idx].bmax[2] = pz;
    }
    __syncthreads();

    // Step B: Build internal nodes (NVIDIA blog parallel approach).
    for (int idx = tid; idx < n - 1; idx += HD_BLOCK) {
        int first, last;
        hd_determine_range(morton, n, idx, &first, &last);

        // Find split using NVIDIA blog's binary search on Morton codes.
        unsigned int firstCode = morton[first].code;
        unsigned int lastCode  = morton[last].code;
        int split;

        if (firstCode == lastCode) {
            split = (first + last) >> 1;
        } else {
            int commonPrefix = __clz(firstCode ^ lastCode);
            split = first;
            int step = last - first;
            do {
                step = (step + 1) >> 1;
                int newSplit = split + step;
                if (newSplit < last) {
                    unsigned int splitCode = morton[newSplit].code;
                    int splitPrefix = __clz(firstCode ^ splitCode);
                    if (splitPrefix > commonPrefix)
                        split = newSplit;
                }
            } while (step > 1);
        }

        // Select children (directly from NVIDIA blog).
        int left_child  = (split == first)     ? (n - 1) + split     : split;
        int right_child = (split + 1 == last)  ? (n - 1) + split + 1 : split + 1;

        bvh[idx].left  = left_child;
        bvh[idx].right = right_child;
        bvh[left_child].parent  = idx;
        bvh[right_child].parent = idx;
    }
    __syncthreads();

    // Step C: Bottom-up AABB propagation.
    for (int i = tid; i < n; i += HD_BLOCK) {
        int node = bvh[(n - 1) + i].parent;
        while (node >= 0) {
            int old = atomicAdd(&counters[node], 1);
            if (old == 0) break;  // first visitor, second child not ready
            // Second visitor: both children ready.
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
}

// ============================================================================
// Brute-force point-point max-min squared distance.
// ============================================================================

__device__ inline float hd_brute_query_max_min_sq(
    float* query_samples, int n_query,
    float* target_samples, int n_target,
    float* s_reduce, int tid)
{
    float local_max = 0.0f;
    for (int i = tid; i < n_query; i += HD_BLOCK) {
        float qx = query_samples[i * 3];
        float qy = query_samples[i * 3 + 1];
        float qz = query_samples[i * 3 + 2];
        float best_sq = 1e30f;
        for (int j = 0; j < n_target; j++) {
            float dx = qx - target_samples[j * 3];
            float dy = qy - target_samples[j * 3 + 1];
            float dz = qz - target_samples[j * 3 + 2];
            best_sq = fminf(best_sq, dx*dx + dy*dy + dz*dz);
        }
        local_max = fmaxf(local_max, best_sq);
    }
    return block_reduce_max(local_max, s_reduce, tid);
}

// ============================================================================
// BVH traversal helper — point-point nearest-neighbor.
// query_samples: query points (float*3, n_query elements).
// bvh: BVH over target points (built by hd_bvh_build).
// Returns the max over all query points of (min distance to any target point).
// Result is squared distance; caller takes sqrt.
// ============================================================================

__device__ inline float hd_bvh_query_max_min_sq(
    float* query_samples, int n_query,
    BVHNode* bvh, int n_leaves,
    float* s_reduce, int tid)
{
    float local_max = 0.0f;

    for (int i = tid; i < n_query; i += HD_BLOCK) {
        float qx = query_samples[i * 3];
        float qy = query_samples[i * 3 + 1];
        float qz = query_samples[i * 3 + 2];
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
                // Leaf: point-point distance.
                float dx = qx - nd->bmin[0];
                float dy = qy - nd->bmin[1];
                float dz = qz - nd->bmin[2];
                best_sq = fminf(best_sq, dx*dx + dy*dy + dz*dz);
            } else {
                // Push children, farther first for better pruning.
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

    return block_reduce_max(local_max, s_reduce, tid);
}

// Linear leaf scan (for debugging)
__device__ inline float hd_leaf_scan_max_min_sq(
    float* query_samples, int n_query,
    BVHNode* bvh, int n_leaves,
    float* s_reduce, int tid)
{
    float local_max = 0.0f;
    for (int i = tid; i < n_query; i += HD_BLOCK) {
        float qx = query_samples[i * 3];
        float qy = query_samples[i * 3 + 1];
        float qz = query_samples[i * 3 + 2];
        float best_sq = 1e30f;
        for (int j = 0; j < n_leaves; j++) {
            int leaf = (n_leaves - 1) + j;
            float dx = qx - bvh[leaf].bmin[0];
            float dy = qy - bvh[leaf].bmin[1];
            float dz = qz - bvh[leaf].bmin[2];
            best_sq = fminf(best_sq, dx*dx + dy*dy + dz*dz);
        }
        local_max = fmaxf(local_max, best_sq);
    }
    return block_reduce_max(local_max, s_reduce, tid);
}

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

    // Scratch pointers (all init to NULL for safe cleanup).
    __shared__ int*          s_counts_a;
    __shared__ int*          s_counts_b;
    __shared__ float*        s_samples_a;
    __shared__ float*        s_samples_b;
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
    __shared__ float s_bbox_min[3];
    __shared__ float s_bbox_max[3];

    // Sort coordination.
    __shared__ int s_seg_a[6];
    __shared__ int s_seg_b[6];

    if (tid == 0) {
        s_counts_a = NULL; s_counts_b = NULL;
        s_samples_a = NULL; s_samples_b = NULL;
        s_morton_a = NULL; s_morton_b = NULL;
        s_sort_scratch = NULL;
        s_bvh_a = NULL; s_bvh_b = NULL;
        s_bvh_counters = NULL;
    }
    __syncthreads();

    // Early exit: degenerate meshes.
    if (a->nt <= 0 || a->nv <= 0 || b->nt <= 0 || b->nv <= 0)
        return 0.0f;

    // =================================================================
    // Phase 1: Compute triangle areas and sample counts.
    // =================================================================

    // --- Mesh A ---
    {
        float local_area = 0.0f;
        for (int t = tid; t < a->nt; t += HD_BLOCK) {
            int ia = a->tris[t * 3 + 0], ib = a->tris[t * 3 + 1], ic = a->tris[t * 3 + 2];
            local_area += hd_tri_area(
                a->verts[ia*3], a->verts[ia*3+1], a->verts[ia*3+2],
                a->verts[ib*3], a->verts[ib*3+1], a->verts[ib*3+2],
                a->verts[ic*3], a->verts[ic*3+1], a->verts[ic*3+2]);
        }
        float ta = block_reduce_sum(local_area, s_reduce, tid);
        if (tid == 0) {
            s_total_area_a = ta;
            // Fixed resolution (CoACD uses 2000); area only distributes
            // samples across triangles, not the total count.
            s_resolution_a = HD_RESOLUTION;
        }
    }
    __syncthreads();

    // --- Mesh B ---
    {
        float local_area = 0.0f;
        for (int t = tid; t < b->nt; t += HD_BLOCK) {
            int ia = b->tris[t * 3 + 0], ib = b->tris[t * 3 + 1], ic = b->tris[t * 3 + 2];
            local_area += hd_tri_area(
                b->verts[ia*3], b->verts[ia*3+1], b->verts[ia*3+2],
                b->verts[ib*3], b->verts[ib*3+1], b->verts[ib*3+2],
                b->verts[ic*3], b->verts[ic*3+1], b->verts[ic*3+2]);
        }
        float tb = block_reduce_sum(local_area, s_reduce, tid);
        if (tid == 0) {
            s_total_area_b = tb;
            s_resolution_b = HD_RESOLUTION;
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
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); }
        return 0.0f;
    }

    // Compute per-triangle sample counts for mesh A.
    for (int t = tid; t < a->nt; t += HD_BLOCK) {
        int ia = a->tris[t * 3 + 0], ib = a->tris[t * 3 + 1], ic = a->tris[t * 3 + 2];
        float ta = hd_tri_area(
            a->verts[ia*3], a->verts[ia*3+1], a->verts[ia*3+2],
            a->verts[ib*3], a->verts[ib*3+1], a->verts[ib*3+2],
            a->verts[ic*3], a->verts[ic*3+1], a->verts[ic*3+2]);
        int area_count = (area_a > 1e-20f) ? (int)((float)res_a / area_a * ta) : 0;
        int N;
        if (a->nt > res_a) {
            int step = a->nt / res_a;
            N = (area_count > 0) ? area_count : ((t % step == 0) ? 1 : 0);
        } else {
            N = (area_count > 0) ? area_count : ((t % 2 == 0) ? 1 : 0);
        }
        s_counts_a[t] = N;
    }

    // Compute per-triangle sample counts for mesh B.
    for (int t = tid; t < b->nt; t += HD_BLOCK) {
        int ia = b->tris[t * 3 + 0], ib = b->tris[t * 3 + 1], ic = b->tris[t * 3 + 2];
        float tb = hd_tri_area(
            b->verts[ia*3], b->verts[ia*3+1], b->verts[ia*3+2],
            b->verts[ib*3], b->verts[ib*3+1], b->verts[ib*3+2],
            b->verts[ic*3], b->verts[ic*3+1], b->verts[ic*3+2]);
        int area_count = (area_b > 1e-20f) ? (int)((float)res_b / area_b * tb) : 0;
        int N;
        if (b->nt > res_b) {
            int step = b->nt / res_b;
            N = (area_count > 0) ? area_count : ((t % step == 0) ? 1 : 0);
        } else {
            N = (area_count > 0) ? area_count : ((t % 2 == 0) ? 1 : 0);
        }
        s_counts_b[t] = N;
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

    if (n_sa <= 0 || n_sb <= 0) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); }
        return 0.0f;
    }

    // =================================================================
    // Phase 2: Allocate sample buffers and generate sample points.
    // =================================================================

    if (tid == 0) {
        void *p1 = NULL, *p3 = NULL;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(n_sa * 3 * (int)sizeof(float)), &p1) != HEAP_OK)
            { s_alloc_ok = 0; p1 = NULL; }
        if (heap_alloc(scratch_heap, (unsigned int)(n_sb * 3 * (int)sizeof(float)), &p3) != HEAP_OK)
            { s_alloc_ok = 0; p3 = NULL; }
        s_samples_a  = (float*)p1;
        s_samples_b  = (float*)p3;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); }
        return 0.0f;
    }

    // Generate samples for mesh A.
    for (int t = tid; t < a->nt; t += HD_BLOCK) {
        int off = s_counts_a[t];
        int next_off = (t + 1 < a->nt) ? s_counts_a[t + 1] : n_sa;
        int N = next_off - off;
        int ia = a->tris[t * 3], ib = a->tris[t * 3 + 1], ic = a->tris[t * 3 + 2];
        float p0x = a->verts[ia*3], p0y = a->verts[ia*3+1], p0z = a->verts[ia*3+2];
        float p1x = a->verts[ib*3], p1y = a->verts[ib*3+1], p1z = a->verts[ib*3+2];
        float p2x = a->verts[ic*3], p2y = a->verts[ic*3+1], p2z = a->verts[ic*3+2];
        for (int k = 0; k < N; k++) {
            unsigned int ha = hd_wang_hash((unsigned int)t * 65537u + (unsigned int)k);
            unsigned int hb = hd_wang_hash(ha + 0x9e3779b9u);
            float ra = (float)(ha & 0xFFFFFFu) / (float)0xFFFFFFu;
            float rb = (float)(hb & 0xFFFFFFu) / (float)0xFFFFFFu;
            float sqa = sqrtf(ra);
            int idx = off + k;
            s_samples_a[idx * 3 + 0] = (1.0f - sqa) * p0x + sqa * (1.0f - rb) * p1x + sqa * rb * p2x;
            s_samples_a[idx * 3 + 1] = (1.0f - sqa) * p0y + sqa * (1.0f - rb) * p1y + sqa * rb * p2y;
            s_samples_a[idx * 3 + 2] = (1.0f - sqa) * p0z + sqa * (1.0f - rb) * p1z + sqa * rb * p2z;
        }
    }

    // Generate samples for mesh B.
    for (int t = tid; t < b->nt; t += HD_BLOCK) {
        int off = s_counts_b[t];
        int next_off = (t + 1 < b->nt) ? s_counts_b[t + 1] : n_sb;
        int N = next_off - off;
        int ia = b->tris[t * 3], ib = b->tris[t * 3 + 1], ic = b->tris[t * 3 + 2];
        float p0x = b->verts[ia*3], p0y = b->verts[ia*3+1], p0z = b->verts[ia*3+2];
        float p1x = b->verts[ib*3], p1y = b->verts[ib*3+1], p1z = b->verts[ib*3+2];
        float p2x = b->verts[ic*3], p2y = b->verts[ic*3+1], p2z = b->verts[ic*3+2];
        for (int k = 0; k < N; k++) {
            unsigned int ha = hd_wang_hash((unsigned int)t * 65537u + (unsigned int)k);
            unsigned int hb = hd_wang_hash(ha + 0x9e3779b9u);
            float ra = (float)(ha & 0xFFFFFFu) / (float)0xFFFFFFu;
            float rb = (float)(hb & 0xFFFFFFu) / (float)0xFFFFFFu;
            float sqa = sqrtf(ra);
            int idx = off + k;
            s_samples_b[idx * 3 + 0] = (1.0f - sqa) * p0x + sqa * (1.0f - rb) * p1x + sqa * rb * p2x;
            s_samples_b[idx * 3 + 1] = (1.0f - sqa) * p0y + sqa * (1.0f - rb) * p1y + sqa * rb * p2y;
            s_samples_b[idx * 3 + 2] = (1.0f - sqa) * p0z + sqa * (1.0f - rb) * p1z + sqa * rb * p2z;
        }
    }
    __syncthreads();

    // Free count arrays.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_counts_a); s_counts_a = NULL;
        heap_free(scratch_heap, (void*)s_counts_b); s_counts_b = NULL;
    }
    __syncthreads();

    // =================================================================
    // Phase 3: Morton code computation.
    // =================================================================

    // Compute combined bounding box of all samples.
    if (tid < 3) { s_bbox_min[tid] = 1e30f; s_bbox_max[tid] = -1e30f; }
    __syncthreads();

    {
        float tlo[3] = { 1e30f, 1e30f, 1e30f };
        float thi[3] = { -1e30f, -1e30f, -1e30f };
        for (int i = tid; i < n_sa; i += HD_BLOCK) {
            float x = s_samples_a[i * 3], y = s_samples_a[i * 3 + 1], z = s_samples_a[i * 3 + 2];
            tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
            tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
            tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
        }
        for (int i = tid; i < n_sb; i += HD_BLOCK) {
            float x = s_samples_b[i * 3], y = s_samples_b[i * 3 + 1], z = s_samples_b[i * 3 + 2];
            tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
            tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
            tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
        }
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
    for (int ax = 0; ax < 3; ax++)
        if (bb_ext[ax] < 1e-20f) bb_ext[ax] = 1e-20f;

    // Allocate morton arrays.
    if (tid == 0) {
        void *p1 = NULL, *p2 = NULL;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(n_sb * (int)sizeof(MortonPoint)), &p1) != HEAP_OK)
            { s_alloc_ok = 0; p1 = NULL; }
        if (heap_alloc(scratch_heap, (unsigned int)(n_sa * (int)sizeof(MortonPoint)), &p2) != HEAP_OK)
            { s_alloc_ok = 0; p2 = NULL; }
        s_morton_b = (MortonPoint*)p1;
        s_morton_a = (MortonPoint*)p2;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { HD_FREE_ALL_SCRATCH(); }
        return 0.0f;
    }

    // Compute Morton codes for B samples (BVH for A->B queries).
    for (int i = tid; i < n_sb; i += HD_BLOCK) {
        float x = s_samples_b[i * 3], y = s_samples_b[i * 3 + 1], z = s_samples_b[i * 3 + 2];
        unsigned int mx = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (x - bb_min[0]) / bb_ext[0] * 1023.0f));
        unsigned int my = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (y - bb_min[1]) / bb_ext[1] * 1023.0f));
        unsigned int mz = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (z - bb_min[2]) / bb_ext[2] * 1023.0f));
        s_morton_b[i].code = hd_morton3D(mx, my, mz);
        s_morton_b[i].index = i;
    }
    // Compute Morton codes for A samples (BVH for B->A queries).
    for (int i = tid; i < n_sa; i += HD_BLOCK) {
        float x = s_samples_a[i * 3], y = s_samples_a[i * 3 + 1], z = s_samples_a[i * 3 + 2];
        unsigned int mx = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (x - bb_min[0]) / bb_ext[0] * 1023.0f));
        unsigned int my = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (y - bb_min[1]) / bb_ext[1] * 1023.0f));
        unsigned int mz = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (z - bb_min[2]) / bb_ext[2] * 1023.0f));
        s_morton_a[i].code = hd_morton3D(mx, my, mz);
        s_morton_a[i].index = i;
    }
    __syncthreads();

    // =================================================================
    // Phase 4: Cooperative sort of Morton codes.
    //
    // 8 warps: warps 0-3 sort morton_b, warps 4-7 sort morton_a.
    // =================================================================

    {
        int max_n = (n_sa > n_sb) ? n_sa : n_sb;
        int scratch_per_set = max_n * (int)sizeof(MortonPoint) +
                              4 * WS_MAX_STACK * 2 * (int)sizeof(int);
        int total_scratch = scratch_per_set * 2;

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

        // Warps 0-3: sort morton_b; warps 4-7: sort morton_a.
        int set_idx = (warp_id < 4) ? 0 : 1;
        int local_warp = warp_id & 3;
        MortonPoint* data = (set_idx == 0) ? s_morton_b : s_morton_a;
        int n = (set_idx == 0) ? n_sb : n_sa;
        char* scratch_base = s_sort_scratch + set_idx * scratch_per_set;
        MortonPoint* tmp = (MortonPoint*)scratch_base;
        int* stacks = (int*)(scratch_base + ((set_idx == 0) ? n_sb : n_sa) * (int)sizeof(MortonPoint));
        int* my_segs = (set_idx == 0) ? s_seg_a : s_seg_b;

        if (n <= 32) {
            if (local_warp == 0) {
                MortonPoint val = (lane < n) ? data[lane] : MortonPointCmp::sentinel();
                val = warp_bitonic32_t<MortonPoint, MortonPointCmp>(val, n, lane);
                if (lane < n) data[lane] = val;
            }
        } else {
            // Leader warp partitions twice to get 4 segments.
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
                    if (set_idx == 0) {
                        s_reduce_i[0] = mh3; s_reduce_i[1] = n;
                    } else {
                        s_reduce_i[2] = mh3; s_reduce_i[3] = n;
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
    __syncthreads();

    // Free sort scratch.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_sort_scratch); s_sort_scratch = NULL;
    }
    __syncthreads();

    // =================================================================
    // Phase 5: Linear BVH construction (Karras 2012).
    // Build both BVHs sequentially (all 256 threads on each).
    // =================================================================

    // BVH for B's samples (for A->B queries).
    {
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

        hd_bvh_build(s_bvh_b, s_morton_b, s_samples_b, s_bvh_counters, n_sb, tid);

        if (tid == 0) {
            heap_free(scratch_heap, (void*)s_bvh_counters); s_bvh_counters = NULL;
        }
        __syncthreads();
    }

    // BVH for A's samples (for B->A queries).
    {
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

        hd_bvh_build(s_bvh_a, s_morton_a, s_samples_a, s_bvh_counters, n_sa, tid);

        if (tid == 0) {
            heap_free(scratch_heap, (void*)s_bvh_counters); s_bvh_counters = NULL;
        }
        __syncthreads();
    }

    // Free morton arrays — no longer needed.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_morton_a); s_morton_a = NULL;
        heap_free(scratch_heap, (void*)s_morton_b); s_morton_b = NULL;
    }
    __syncthreads();

    // =================================================================
    // Phase 6: BVH traversal — point-point nearest-neighbor queries.
    // =================================================================

    // Direction A->B: query A samples against B's BVH.
    float dir_ab = hd_bvh_query_max_min_sq(s_samples_a, n_sa, s_bvh_b, n_sb, s_reduce, tid);
    __syncthreads();

    // Direction B->A: query B samples against A's BVH.
    float dir_ba = hd_bvh_query_max_min_sq(s_samples_b, n_sb, s_bvh_a, n_sa, s_reduce, tid);
    __syncthreads();

    // =================================================================
    // Cleanup and return.
    // =================================================================

    float result = sqrtf(fmaxf(dir_ab, dir_ba));

    if (tid == 0) {
        HD_FREE_ALL_SCRATCH();
    }

    return result;
}
