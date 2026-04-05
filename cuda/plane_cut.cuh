// plane_cut.cu — GPU plane cut kernel with cap triangulation.
//
// One block (64 threads = 2 warps) handles one plane cut.
// Produces two closed meshes (positive/negative halves) with
// triangulated caps. Supports simple, ring, and multi-hole topologies.
//
// Scratch memory is allocated from scratch_heap and freed as soon as each
// buffer is no longer needed.  Output mesh data goes into heap (persistent).
//
// Phases:
//   1-2  Classify vertices, copy to scratch           (parallel, 64 threads)
//   3    Collect crossing edges                        (parallel, 64 threads)
//   4    Sort crossing edges                           (warp 0)
//   5    Dedup + create intersection vertices          (thread 0)
//        Free sort_scratch; alloc pos_tris, neg_tris.
//   6    Split triangles                               (parallel, 64 threads)
//        Free signs, cross_edges, isect_idx.
//   7    Collect directed edges from pos_tris          (parallel, 64 threads)
//   8    Sort directed edges                           (warp 0)
//        Free dir_sort.
//   9    Binary search for boundary edges              (parallel, 64 threads)
//  10-12 Compact boundary → loops → polygon → ear-clip (thread 0)
//  13    Compact verts per side, heap-alloc output     (parallel, 64 threads)
//        Free all remaining scratch.
//
#pragma once
#include "common.cuh"
#include "allocator.cuh"
#include "structs.cuh"
#include "warp_sort.cuh"

#define PC_BLOCK 64
#define PC_EPS   1e-6f
#define PC_ALIGN16(x) (((x) + 15) & ~15)

// ============================================================================
// Edge2i — compact 8-byte edge struct (two vertex indices)
// ============================================================================

struct Edge2i {
    int a, b;
};

struct Edge2iCmp {
    static __device__ inline int cmp(Edge2i x, Edge2i y) {
        if (x.a != y.a) return (x.a < y.a) ? -1 : 1;
        if (x.b != y.b) return (x.b < y.b) ? -1 : 1;
        return 0;
    }
    static __device__ inline Edge2i sentinel() {
        Edge2i s; s.a = 0x7fffffff; s.b = 0x7fffffff; return s;
    }
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ inline int pc_edge_bsearch(const Edge2i* arr, int n, int key_a, int key_b) {
    Edge2i key = {key_a, key_b};
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = Edge2iCmp::cmp(arr[mid], key);
        if (c < 0) lo = mid + 1; else if (c > 0) hi = mid - 1; else return mid;
    }
    return -1;
}

__device__ inline float pc_cross2d(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
}

__device__ inline int pc_pt_in_tri(float px, float py,
    float ax, float ay, float bx, float by, float cx, float cy)
{
    float d1 = pc_cross2d(ax,ay,bx,by,px,py);
    float d2 = pc_cross2d(bx,by,cx,cy,px,py);
    float d3 = pc_cross2d(cx,cy,ax,ay,px,py);
    return !((d1<-1e-10f||d2<-1e-10f||d3<-1e-10f) &&
             (d1> 1e-10f||d2> 1e-10f||d3> 1e-10f));
}

__device__ inline float pc_signed_area(const float* verts, const int* poly, int n, int pu, int pv) {
    float area = 0.0f;
    for (int i = 0; i < n; i++) {
        int j = (i+1) % n;
        area += verts[poly[i]*3+pu]*verts[poly[j]*3+pv]
              - verts[poly[j]*3+pu]*verts[poly[i]*3+pv];
    }
    return area * 0.5f;
}

__device__ inline void pc_reverse(int* arr, int n) {
    for (int i = 0; i < n/2; i++) { int t = arr[i]; arr[i] = arr[n-1-i]; arr[n-1-i] = t; }
}

__device__ inline void pc_intersect(
    const float* verts, int va, int vb,
    float pa, float pb, float pc_n, float pd, float* out)
{
    float d0 = pa*verts[va*3]+pb*verts[va*3+1]+pc_n*verts[va*3+2]+pd;
    float d1 = pa*verts[vb*3]+pb*verts[vb*3+1]+pc_n*verts[vb*3+2]+pd;
    float t = d0 / (d0 - d1);
    out[0] = verts[va*3+0] + t*(verts[vb*3+0]-verts[va*3+0]);
    out[1] = verts[va*3+1] + t*(verts[vb*3+1]-verts[va*3+1]);
    out[2] = verts[va*3+2] + t*(verts[vb*3+2]-verts[va*3+2]);
}

// ============================================================================
// Kernel
// ============================================================================

#define PC_KERR_SCRATCH_OOM 1
#define PC_KERR_POOL_OOM    2
#define PC_KERR_SORT_ERR    4

__device__ inline void pc_zero_part(Part* p) {
    p->mesh.verts = NULL; p->mesh.tris = NULL; p->mesh.nv = 0; p->mesh.nt = 0; p->mesh.refcount = NULL;
    p->hull.verts = NULL; p->hull.tris = NULL; p->hull.nv = 0; p->hull.nt = 0; p->hull.refcount = NULL;
    p->mesh_vol = 0.0f; p->hull_vol = 0.0f; p->hausdorff = 0.0f;
}

// Free all shared scratch pointers (NULL-safe; call from thread 0 only).
// Used on every early-exit path.
#define PC_FREE_ALL_SHARED_SCRATCH() do {                           \
    heap_free(scratch_heap, (void*)s_signs);                        \
    heap_free(scratch_heap, (void*)s_all_verts);                    \
    heap_free(scratch_heap, (void*)s_cross_edges);                  \
    heap_free(scratch_heap, (void*)s_sort_scratch);                 \
    heap_free(scratch_heap, (void*)s_isect_idx);                    \
    heap_free(scratch_heap, (void*)s_pos_tris);                     \
    heap_free(scratch_heap, (void*)s_neg_tris);                     \
    heap_free(scratch_heap, (void*)s_dir_edges);                    \
    heap_free(scratch_heap, (void*)s_dir_sort);                     \
    heap_free(scratch_heap, (void*)s_boundary_flags);               \
} while(0)

__device__ inline PartPair plane_cut_block(
    // Input
    const Mesh* __restrict__ mesh,
    float pa, float pb, float pc_n, float pd,
    DeviceHeap* heap,          // output heap (persistent mesh data)
    DeviceHeap* scratch_heap,  // scratch heap (all allocs freed within this call)
    int* __restrict__ kernel_error)
{
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;

    const float* vertices  = mesh->verts;
    const int*   triangles = mesh->tris;
    int          n_verts   = mesh->nv;
    int          n_tris    = mesh->nt;

    // Upper-bound sizes (all derived from n_verts / n_tris)
    int max_new_verts      = n_tris * 2;
    int total_verts_cap    = n_verts + max_new_verts;
    int max_cross_edges    = n_tris * 2;
    int max_out_tris       = n_tris * 3;   // phase-6 upper bound; also sizes dir_edges
    // Cap triangles (ear-clip) are appended to pos_tris/neg_tris at index n_pos/n_neg.
    // poly_n <= 2*n_boundary <= 4*n_tris, so n_cap <= 4*n_tris + 1024.
    int max_cap_tris       = n_tris * 4 + 1024;
    // sort_scratch_bytes computed after phase 3 when n_cross is known.
    // dir_edges/dir_sort/boundary_flags are now sized from actual n_trace after phase 6.

    // s_counters: [0]=n_cross, [1]=n_all_verts, [2]=n_pos, [3]=n_neg
    __shared__ int s_counters[4];
    __shared__ int s_alloc_ok;

    // Shared scratch pointers — all initialised to NULL so heap_free is always safe.
    __shared__ int*    s_signs;
    __shared__ float*  s_all_verts;
    __shared__ Edge2i* s_cross_edges;
    __shared__ char*   s_sort_scratch;
    __shared__ int*    s_isect_idx;
    __shared__ int*    s_pos_tris;
    __shared__ int*    s_neg_tris;
    __shared__ Edge2i* s_dir_edges;
    __shared__ char*   s_dir_sort;
    __shared__ int*    s_boundary_flags;

    // Return value — written by thread 0, returned by all threads.
    __shared__ PartPair s_result;

    // Phase-13 broadcast and parallel-compaction scratch
    __shared__ int    s_total_pos, s_total_neg;
    __shared__ int    s_pnv, s_nnv;
    __shared__ int*   s_pr_ptr;
    __shared__ int*   s_nr_ptr;
    __shared__ float* s_pvp;
    __shared__ int*   s_ptp;
    __shared__ float* s_nvp;
    __shared__ int*   s_ntp;
    __shared__ int    s_chunk_ok;
    __shared__ int    s_pcount[PC_BLOCK];
    __shared__ int    s_ncount[PC_BLOCK];
    __shared__ int    s_pbase[PC_BLOCK];
    __shared__ int    s_nbase[PC_BLOCK];

    if (tid == 0) {
        s_signs = NULL; s_all_verts = NULL;
        s_cross_edges = NULL; s_sort_scratch = NULL; s_isect_idx = NULL;
        s_pos_tris = NULL; s_neg_tris = NULL;
        s_dir_edges = NULL; s_dir_sort = NULL; s_boundary_flags = NULL;
        s_counters[0] = 0; s_counters[1] = n_verts;
        s_counters[2] = 0; s_counters[3] = 0;
        pc_zero_part(&s_result.pos);
        pc_zero_part(&s_result.neg);
    }
    __syncthreads();

    // =========================================================================
    // Alloc signs + all_verts  (needed: phases 1-6)
    // =========================================================================
    if (tid == 0) {
        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(n_verts * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_signs = (int*)p;

        if (heap_alloc(scratch_heap, (unsigned int)(total_verts_cap * 3 * (int)sizeof(float)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_all_verts = (float*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(int,   signs,     s_signs,     n_verts);
    PC_BUF(float, all_verts, s_all_verts, total_verts_cap * 3);

    // === Phase 1: Classify vertices ===
    for (int v = tid; v < n_verts; v += PC_BLOCK) {
        float val = pa*vertices[v*3] + pb*vertices[v*3+1] + pc_n*vertices[v*3+2] + pd;
        signs[v] = (val > PC_EPS) ? 1 : ((val < -PC_EPS) ? -1 : 0);
    }

    // === Phase 2: Copy original vertices ===
    for (int i = tid; i < n_verts * 3; i += PC_BLOCK)
        all_verts[i] = vertices[i];
    __syncthreads();

    // =========================================================================
    // Alloc cross_edges + sort_scratch + isect_idx  (needed: phases 3-6)
    // =========================================================================
    if (tid == 0) {
        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(max_cross_edges * (int)sizeof(Edge2i)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_cross_edges = (Edge2i*)p;

        // sort_scratch deferred to after phase 3 when n_cross is known.

        if (heap_alloc(scratch_heap, (unsigned int)(max_cross_edges * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_isect_idx = (int*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(Edge2i, cross_edges, s_cross_edges, max_cross_edges);
    PC_BUF(int,    isect_idx,   s_isect_idx,   max_cross_edges);

    // === Phase 3: Collect crossing edges ===
    for (int t = tid; t < n_tris; t += PC_BLOCK) {
        int i0 = triangles[t*3], i1 = triangles[t*3+1], i2 = triangles[t*3+2];
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];
        int edges[3][2]  = {{i0,i1},{i1,i2},{i2,i0}};
        int esigns[3][2] = {{s0,s1},{s1,s2},{s2,s0}};
        for (int e = 0; e < 3; e++) {
            int sa = esigns[e][0], sb = esigns[e][1];
            if ((sa > 0 && sb < 0) || (sa < 0 && sb > 0)) {
                int va = edges[e][0], vb = edges[e][1];
                int mn = (va < vb) ? va : vb, mx = (va < vb) ? vb : va;
                int ci = atomicAdd(&s_counters[0], 1);
                if (ci < max_cross_edges) { cross_edges[ci].a = mn; cross_edges[ci].b = mx; }
            }
        }
    }
    __syncthreads();

    __shared__ int s_n_cross;
    if (tid == 0) {
        s_n_cross = s_counters[0];
        if (s_n_cross > max_cross_edges) s_n_cross = max_cross_edges;
    }
    __syncthreads();
    int n_cross = s_n_cross;

    if (n_cross == 0) {
        // No crossing — entire mesh on one side.
        if (tid == 0) {
            int any_pos = 0, any_neg = 0;
            for (int v = 0; v < n_verts; v++) {
                if (signs[v] > 0) any_pos = 1;
                if (signs[v] < 0) any_neg = 1;
            }
            int is_neg = (any_neg && !any_pos);
            unsigned int vb = (unsigned int)PC_ALIGN16(n_verts * 3 * (int)sizeof(float));
            unsigned int tb = (unsigned int)PC_ALIGN16(n_tris  * 3 * (int)sizeof(int));
            unsigned int rb = (unsigned int)PC_ALIGN16((int)sizeof(int));
            unsigned int sz = vb + tb + rb; if (!sz) sz = 1;
            void* chunk = NULL;
            if (heap_alloc(heap, sz, &chunk) != HEAP_OK) {
                atomicOr(kernel_error, PC_KERR_POOL_OOM);
            } else {
                float* pv = (float*)chunk;
                int*   pt = (int*)((char*)chunk + vb);
                int*   rc = (int*)((char*)chunk + vb + tb);
                *rc = 1;
                for (int i = 0; i < n_verts * 3; i++) pv[i] = vertices[i];
                for (int i = 0; i < n_tris  * 3; i++) pt[i] = triangles[i];
                Part* side  = is_neg ? &s_result.neg : &s_result.pos;
                Part* empty = is_neg ? &s_result.pos : &s_result.neg;
                pc_zero_part(side);
                side->mesh.verts = pv; side->mesh.tris = pt;
                side->mesh.nv = n_verts; side->mesh.nt = n_tris;
                side->mesh.refcount = rc;
                pc_zero_part(empty);
            }
            PC_FREE_ALL_SHARED_SCRATCH();
        }
        __syncthreads();
        return s_result;
    }

    // Alloc sort_scratch now that n_cross is known.
    if (tid == 0) {
        int sort_scratch_bytes = n_cross * (int)sizeof(Edge2i) + WS_MAX_STACK * 2 * (int)sizeof(int);
        void* p;
        if (heap_alloc(scratch_heap, (unsigned int)sort_scratch_bytes, &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_sort_scratch = (char*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    // === Phase 4: Sort crossing edges (warp 0) ===
    if (warp_id == 0) {
        int serr = warp_sort_t<Edge2i, Edge2iCmp>(cross_edges.raw(), s_sort_scratch, n_cross, lane);
        if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
    }
    __syncthreads();

    // =========================================================================
    // Phase 5: Dedup + intersection vertices  (thread 0)
    // Also: free sort_scratch (done); alloc pos_tris + neg_tris.
    // =========================================================================
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_sort_scratch); s_sort_scratch = NULL;

        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)((max_out_tris + max_cap_tris) * 3 * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_pos_tris = (int*)p;

        if (heap_alloc(scratch_heap, (unsigned int)((max_out_tris + max_cap_tris) * 3 * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_neg_tris = (int*)p;

        int cur_isect = n_verts;
        for (int i = 0; i < n_cross; i++) {
            if (i == 0 || cross_edges[i].a != cross_edges[i-1].a || cross_edges[i].b != cross_edges[i-1].b) {
                int va = cross_edges[i].a, vb = cross_edges[i].b;
                pc_intersect(all_verts.raw(), va, vb, pa, pb, pc_n, pd, &all_verts[cur_isect * 3]);
                isect_idx[i] = cur_isect++;
            } else {
                isect_idx[i] = isect_idx[i - 1];
            }
        }
        s_counters[1] = cur_isect;  // n_all_verts
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    int  n_all    = s_counters[1];
    int  tris_cap = (max_out_tris + max_cap_tris) * 3;
    PC_BUF(int, pos_tris, s_pos_tris, tris_cap);
    PC_BUF(int, neg_tris, s_neg_tris, tris_cap);

    // === Phase 6: Split triangles ===
    for (int t = tid; t < n_tris; t += PC_BLOCK) {
        int i0 = triangles[t*3], i1 = triangles[t*3+1], i2 = triangles[t*3+2];
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];
        int hp = (s0>0)||(s1>0)||(s2>0), hn = (s0<0)||(s1<0)||(s2<0);

        if (!hp || !hn) {
            if (hn) {
                int ni = atomicAdd(&s_counters[3], 1);
                neg_tris[ni*3]=i0; neg_tris[ni*3+1]=i1; neg_tris[ni*3+2]=i2;
            } else if (hp) {
                int pi = atomicAdd(&s_counters[2], 1);
                pos_tris[pi*3]=i0; pos_tris[pi*3+1]=i1; pos_tris[pi*3+2]=i2;
            } else {
                float e1x=all_verts[i1*3]-all_verts[i0*3], e1y=all_verts[i1*3+1]-all_verts[i0*3+1], e1z=all_verts[i1*3+2]-all_verts[i0*3+2];
                float e2x=all_verts[i2*3]-all_verts[i0*3], e2y=all_verts[i2*3+1]-all_verts[i0*3+1], e2z=all_verts[i2*3+2]-all_verts[i0*3+2];
                float dot = pa*(e1y*e2z-e1z*e2y)+pb*(e1z*e2x-e1x*e2z)+pc_n*(e1x*e2y-e1y*e2x);
                if (dot > 0) { int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=i0;pos_tris[pi*3+1]=i1;pos_tris[pi*3+2]=i2; }
                else         { int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=i0;neg_tris[ni*3+1]=i1;neg_tris[ni*3+2]=i2; }
            }
            continue;
        }

        int vi[3] = {i0,i1,i2}, si[3] = {s0,s1,s2};

        #define FIND_ISECT(va, vb) ({ \
            int _mn=((va)<(vb))?(va):(vb), _mx=((va)<(vb))?(vb):(va); \
            int _fi=pc_edge_bsearch(cross_edges.raw(), n_cross, _mn, _mx); \
            (_fi>=0)?isect_idx[_fi]:-1; \
        })

        int on = -1;
        for (int k = 0; k < 3; k++) if (si[k] == 0) { on = k; break; }
        if (on >= 0) {
            int ov=vi[on], a_i=vi[(on+1)%3], b_i=vi[(on+2)%3], sa=si[(on+1)%3];
            int nvi = FIND_ISECT(a_i, b_i); if (nvi < 0) continue;
            if (sa > 0) {
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov;pos_tris[pi*3+1]=a_i;pos_tris[pi*3+2]=nvi;
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov;neg_tris[ni*3+1]=nvi;neg_tris[ni*3+2]=b_i;
            } else {
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov;neg_tris[ni*3+1]=a_i;neg_tris[ni*3+2]=nvi;
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov;pos_tris[pi*3+1]=nvi;pos_tris[pi*3+2]=b_i;
            }
        } else {
            int lone = -1;
            for (int k = 0; k < 3; k++)
                if (si[k]!=si[(k+1)%3] && si[k]!=si[(k+2)%3]) { lone=k; break; }
            if (lone < 0) continue;
            int lv=vi[lone], ov1=vi[(lone+1)%3], ov2=vi[(lone+2)%3], ls=si[lone];
            int nv1=FIND_ISECT(lv,ov1), nv2=FIND_ISECT(lv,ov2);
            if (nv1<0||nv2<0) continue;
            if (ls > 0) {
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=lv;pos_tris[pi*3+1]=nv1;pos_tris[pi*3+2]=nv2;
                int ni=atomicAdd(&s_counters[3],2);
                neg_tris[ni*3]=ov1;neg_tris[ni*3+1]=nv2;neg_tris[ni*3+2]=nv1;
                neg_tris[(ni+1)*3]=ov1;neg_tris[(ni+1)*3+1]=ov2;neg_tris[(ni+1)*3+2]=nv2;
            } else {
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=lv;neg_tris[ni*3+1]=nv1;neg_tris[ni*3+2]=nv2;
                int pi=atomicAdd(&s_counters[2],2);
                pos_tris[pi*3]=ov1;pos_tris[pi*3+1]=nv2;pos_tris[pi*3+2]=nv1;
                pos_tris[(pi+1)*3]=ov1;pos_tris[(pi+1)*3+1]=ov2;pos_tris[(pi+1)*3+2]=nv2;
            }
        }
        #undef FIND_ISECT
    }
    __syncthreads();

    // =========================================================================
    // Free signs, cross_edges, isect_idx — done after phase 6.
    // Alloc dir_edges + dir_sort + boundary_flags  (needed: phases 7-9)
    // =========================================================================
    __shared__ int s_n_pos, s_n_neg, s_n_trace;
    __shared__ int* s_trace_tris;  // points to the smaller side's triangle buffer
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_signs);       s_signs       = NULL;
        heap_free(scratch_heap, (void*)s_cross_edges); s_cross_edges = NULL;
        heap_free(scratch_heap, (void*)s_isect_idx);   s_isect_idx   = NULL;

        s_n_pos = s_counters[2];
        s_n_neg = s_counters[3];
        // Trace the smaller side to find boundary edges — saves ~half space+time.
        // Boundary edges are identical regardless of which side is traced.
        if (s_n_pos <= s_n_neg) {
            s_n_trace = s_n_pos; s_trace_tris = s_pos_tris;
        } else {
            s_n_trace = s_n_neg; s_trace_tris = s_neg_tris;
        }

        int n_de_alloc = s_n_trace * 3;
        int de_sort_bytes = n_de_alloc * (int)sizeof(Edge2i) + WS_MAX_STACK * 2 * (int)sizeof(int);

        void* p;
        s_alloc_ok = 1;
        if (n_de_alloc > 0) {
            if (heap_alloc(scratch_heap, (unsigned int)(n_de_alloc * (int)sizeof(Edge2i)), &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_dir_edges = (Edge2i*)p;

            if (heap_alloc(scratch_heap, (unsigned int)de_sort_bytes, &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_dir_sort = (char*)p;

            if (heap_alloc(scratch_heap, (unsigned int)(n_de_alloc * (int)sizeof(int)), &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_boundary_flags = (int*)p;
        }
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    int n_pos = s_n_pos, n_neg = s_n_neg;
    int n_trace = s_n_trace;
    int* trace_tris = s_trace_tris;
    int n_de = n_trace * 3;
    PC_BUF(Edge2i, dir_edges,      s_dir_edges,      n_de);
    PC_BUF(int,    boundary_flags, s_boundary_flags,  n_de);

    // === Phase 7: Collect directed edges from the smaller side ===
    for (int t = tid; t < n_trace; t += PC_BLOCK) {
        int a = trace_tris[t*3], b = trace_tris[t*3+1], c = trace_tris[t*3+2];
        int bi = t * 3;
        dir_edges[bi  ] = (Edge2i){a, b};
        dir_edges[bi+1] = (Edge2i){b, c};
        dir_edges[bi+2] = (Edge2i){c, a};
    }
    __syncthreads();

    // === Phase 8: Sort directed edges (warp 0) ===
    if (warp_id == 0) {
        int serr = warp_sort_t<Edge2i, Edge2iCmp>(dir_edges.raw(), s_dir_sort, n_de, lane);
        if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
    }
    __syncthreads();

    // Free dir_sort — not needed after sort.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_dir_sort); s_dir_sort = NULL;
    }

    // === Phase 9: Binary search for boundary edges ===
    for (int i = tid; i < n_de; i += PC_BLOCK) {
        int a = dir_edges[i].a, b = dir_edges[i].b;
        boundary_flags[i] = (pc_edge_bsearch(dir_edges.raw(), n_de, b, a) < 0) ? 1 : 0;
    }
    __syncthreads();

    // =========================================================================
    // Phases 10-12: thread 0 only.
    //
    // All local scratch is void* locals (init NULL), freed inline as early as
    // possible.  kern_ok gates work after the first failure.
    // At the end, kern_ok + total_pos/neg + remap ptrs are broadcast to all
    // threads via shared memory for the parallel phase 13.
    // =========================================================================
    if (tid == 0) {
        int kern_ok    = 1;
        int n_cap      = 0;
        int n_boundary = 0;

        // Thread-0-only scratch — all NULL so heap_free is always safe.
        void* lv_ptr   = NULL;  // boundary edge pairs (be_a), freed after loop recon
        void* lv2_ptr  = NULL;  // loop vertex sequence (lv)
        void* ls_ptr   = NULL;  // loop_starts
        void* lsz_ptr  = NULL;  // loop_sizes
        void* poly_ptr = NULL;  // polygon
        void* cap_ptr  = NULL;  // cap_tris (ear-clip output)
        void* ep_ptr   = NULL;  // ear_prevnext
        void* sb_ptr   = NULL;  // inner sort buf (inner_idx + inner_max_u, 256 each)

        // --- Compact boundary edges into lv_ptr (packed a,b pairs) ---
        if (n_de > 0) {
            if (heap_alloc(scratch_heap, (unsigned int)(n_de * 2 * (int)sizeof(int)), &lv_ptr) != HEAP_OK) {
                kern_ok = 0; atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);
            }
        }

        if (kern_ok) {
            PC_BUF(int, be_a, lv_ptr, n_de * 2);
            for (int i = 0; i < n_de; i++) {
                if (boundary_flags[i]) {
                    be_a[n_boundary*2]   = dir_edges[i].a;
                    be_a[n_boundary*2+1] = dir_edges[i].b;
                    n_boundary++;
                }
            }
            // dir_edges and boundary_flags no longer needed.
            heap_free(scratch_heap, (void*)s_dir_edges);       s_dir_edges      = NULL;
            heap_free(scratch_heap, (void*)s_boundary_flags);  s_boundary_flags = NULL;

            // --- Phase 10-12: loop reconstruction + polygon + ear-clip ---
            if (n_boundary > 0) {
                if (heap_alloc(scratch_heap, (unsigned int)(n_boundary * (int)sizeof(int)), &ls_ptr)  != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)(n_boundary * (int)sizeof(int)), &lsz_ptr) != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)(n_boundary * (int)sizeof(int)), &lv2_ptr) != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)((n_boundary*4+64) * (int)sizeof(int)), &poly_ptr) != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)((n_boundary*4+64) * (int)sizeof(int)), &cap_ptr)  != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)((n_boundary*4+64) * 2 * (int)sizeof(int)), &ep_ptr)   != HEAP_OK) { kern_ok = 0; }
                if (kern_ok && heap_alloc(scratch_heap, (unsigned int)(256 * (int)sizeof(int) + 256 * (int)sizeof(float)), &sb_ptr) != HEAP_OK) { kern_ok = 0; }
                if (!kern_ok) atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);

                if (kern_ok) {
                    int poly_cap = n_boundary * 4 + 64;
                    PC_BUF(int,   loop_starts,  ls_ptr,   n_boundary);
                    PC_BUF(int,   loop_sizes,   lsz_ptr,  n_boundary);
                    PC_BUF(int,   lv,           lv2_ptr,  n_boundary);
                    PC_BUF(int,   polygon,      poly_ptr, poly_cap);
                    PC_BUF(int,   cap_tris,     cap_ptr,  poly_cap * 3);
                    PC_BUF(int,   ear_prevnext, ep_ptr,   poly_cap * 2);
                    PC_BUF(int,   inner_idx,    sb_ptr,   256);
                    float* inner_max_u  = (float*)((int*)sb_ptr + 256);

                    // Phase 10: reconstruct loops from boundary edge pairs
                    PC_BUF(int, be_a2, lv_ptr, n_de * 2);
                    int* be_used = loop_starts.raw();  // borrow before it's filled
                    for (int i = 0; i < n_boundary; i++) be_used[i] = 0;

                    int n_loops = 0, lvi = 0;
                    for (int start = 0; start < n_boundary; start++) {
                        if (be_used[start]) continue;
                        int lvi_start = lvi;
                        int first = be_a2[start*2], cur = be_a2[start*2+1];
                        be_used[start] = 1;  // aliases loop_starts[start]; loop_starts[n_loops] set below
                        if (lvi >= n_boundary) break;
                        lv[lvi++] = first;
                        int safety = n_boundary + 2;
                        while (cur != first && safety-- > 0) {
                            if (lvi >= n_boundary) break;
                            lv[lvi++] = cur;
                            int found = 0;
                            for (int i = 0; i < n_boundary; i++) {
                                if (!be_used[i] && be_a2[i*2] == cur) {
                                    cur = be_a2[i*2+1]; be_used[i] = 1; found = 1; break;
                                }
                            }
                            if (!found) break;
                        }
                        loop_starts[n_loops] = lvi_start;  // set after be_used writes (avoids alias clobber)
                        loop_sizes[n_loops] = lvi - lvi_start;
                        if (loop_sizes[n_loops] > 0) n_loops++;
                    }

                    // be_a (lv_ptr) no longer needed — free to reclaim memory.
                    heap_free(scratch_heap, lv_ptr); lv_ptr = NULL;

                    // 2D projection axes (drop largest normal component)
                    int pu, pv_ax;
                    {
                        float anx=fabsf(pa), any=fabsf(pb), anz=fabsf(pc_n);
                        if (anx>=any && anx>=anz)  { pu=1; pv_ax=2; }
                        else if (any>=anz)          { pu=0; pv_ax=2; }
                        else                        { pu=0; pv_ax=1; }
                    }

                    // Phase 11-12: For each group of loops, build polygon
                    // (bridging true holes) and ear-clip.
                    //
                    // Loops that are not contained inside any other loop are
                    // independent islands — ear-clipped separately.  Loops
                    // contained inside another are holes that get bridged
                    // into their enclosing loop before ear-clipping.

                    // Orient all loops: positive area = CCW (outer), negative = CW (hole).
                    for (int i = 0; i < n_loops; i++) {
                        float a = pc_signed_area(all_verts.raw(), lv.raw()+loop_starts[i], loop_sizes[i], pu, pv_ax);
                        if (a < 0) pc_reverse(lv.raw()+loop_starts[i], loop_sizes[i]);
                    }

                    // Classify: is_hole[i] = 1 if loop i's first vertex is inside some other loop.
                    // parent[i] = enclosing loop index (smallest enclosing area), or -1.
                    // Reuse inner_idx as parent[], inner_max_u's int-alias as is_hole[].
                    // Both are 256-element buffers from sb_ptr.
                    // First 128 entries: parent/is_hole classification (read-only after fill).
                    // Second 128 entries: per-outer hole sort scratch.
                    int*   parent    = inner_idx.raw();
                    int*   is_hole   = inner_idx.raw() + 128;
                    int*   h_idx     = (int*)inner_max_u;          // hole sort: indices
                    float* h_max_u   = inner_max_u + 128;          // hole sort: max-u values
                    int max_loops = 128;  // limit loops to fit classification arrays
                    for (int i = 0; i < n_loops && i < max_loops; i++) { parent[i] = -1; is_hole[i] = 0; }

                    for (int i = 0; i < n_loops && i < max_loops; i++) {
                        if (loop_sizes[i] < 3) continue;
                        // Test first vertex of loop i against all other loops
                        int vi = lv[loop_starts[i]];
                        float pu_ = all_verts[vi*3+pu], pv_ = all_verts[vi*3+pv_ax];
                        float best_area = 1e30f;
                        for (int j = 0; j < n_loops && j < max_loops; j++) {
                            if (j == i || loop_sizes[j] < 3) continue;
                            // Ray-casting point-in-polygon (2D, along +u axis)
                            int* lp = lv.raw() + loop_starts[j];
                            int ls_ = loop_sizes[j];
                            int crossings = 0;
                            for (int k = 0; k < ls_; k++) {
                                int kn = (k+1) % ls_;
                                float ay = all_verts[lp[k]*3+pv_ax], by = all_verts[lp[kn]*3+pv_ax];
                                if ((ay <= pv_) == (by <= pv_)) continue;
                                float ax = all_verts[lp[k]*3+pu], bx = all_verts[lp[kn]*3+pu];
                                float t = (pv_ - ay) / (by - ay);
                                if (ax + t * (bx - ax) > pu_) crossings++;
                            }
                            if (crossings & 1) {
                                // i is inside j — pick smallest enclosing
                                float a = fabsf(pc_signed_area(all_verts.raw(), lp, ls_, pu, pv_ax));
                                if (a < best_area) { best_area = a; parent[i] = j; }
                            }
                        }
                        if (parent[i] >= 0) is_hole[i] = 1;
                    }

                    // Reverse hole loops to CW winding (they were oriented CCW above).
                    for (int i = 0; i < n_loops && i < max_loops; i++) {
                        if (is_hole[i]) pc_reverse(lv.raw()+loop_starts[i], loop_sizes[i]);
                    }

                    // Process each outer loop (non-hole) and its direct children.
                    for (int oi = 0; oi < n_loops && oi < max_loops; oi++) {
                        if (is_hole[oi] || loop_sizes[oi] < 3) continue;

                        // Copy outer loop into polygon
                        int poly_n = loop_sizes[oi];
                        for (int i = 0; i < poly_n; i++)
                            polygon[i] = lv[loop_starts[oi]+i];

                        // Collect direct holes (parent == oi), sorted by rightmost u descending
                        int n_inner = 0;
                        for (int i = 0; i < n_loops && n_inner < 128; i++) {
                            if (parent[i] != oi || loop_sizes[i] < 3) continue;
                            float mu = -1e30f;
                            for (int j = 0; j < loop_sizes[i]; j++) {
                                float u = all_verts[lv[loop_starts[i]+j]*3+pu];
                                if (u > mu) mu = u;
                            }
                            h_idx[n_inner] = i; h_max_u[n_inner] = mu; n_inner++;
                        }
                        for (int i = 0; i < n_inner-1; i++) {
                            int best = i;
                            for (int j = i+1; j < n_inner; j++)
                                if (h_max_u[j] > h_max_u[best]) best = j;
                            if (best != i) {
                                int ti=h_idx[i]; h_idx[i]=h_idx[best]; h_idx[best]=ti;
                                float tf=h_max_u[i]; h_max_u[i]=h_max_u[best]; h_max_u[best]=tf;
                            }
                        }

                        // Bridge each hole into polygon
                        for (int ii = 0; ii < n_inner; ii++) {
                            int hi = h_idx[ii], hs = loop_sizes[hi];
                            int* hole = lv.raw() + loop_starts[hi];
                            int m_idx = 0; float max_u = -1e30f;
                            for (int j = 0; j < hs; j++) {
                                float u = all_verts[hole[j]*3+pu];
                                if (u > max_u) { max_u=u; m_idx=j; }
                            }
                            float mu_ = all_verts[hole[m_idx]*3+pu];
                            float mv_ = all_verts[hole[m_idx]*3+pv_ax];
                            float best_t = 1e30f; int best_edge = -1;
                            for (int j = 0; j < poly_n; j++) {
                                int jn = (j+1)%poly_n;
                                float au=all_verts[polygon[j]*3+pu],  av2=all_verts[polygon[j]*3+pv_ax];
                                float bu=all_verts[polygon[jn]*3+pu], bv2=all_verts[polygon[jn]*3+pv_ax];
                                float dv = bv2-av2; if (fabsf(dv)<1e-10f) continue;
                                float s = (mv_-av2)/dv;
                                if (s<-1e-10f||s>1.0f+1e-10f) continue;
                                float tv = au+s*(bu-au)-mu_;
                                if (tv>1e-10f && tv<best_t) { best_t=tv; best_edge=j; }
                            }
                            if (best_edge < 0) continue;
                            int jn=(best_edge+1)%poly_n;
                            int p_pos = (all_verts[polygon[best_edge]*3+pu] >= all_verts[polygon[jn]*3+pu])
                                        ? best_edge : jn;
                            int k = 0;
                            for (int j=0; j<=p_pos; j++) cap_tris[k++]=polygon[j];
                            for (int j=0; j<hs; j++) cap_tris[k++]=hole[(m_idx+j)%hs];
                            cap_tris[k++]=hole[m_idx]; cap_tris[k++]=polygon[p_pos];
                            for (int j=p_pos+1; j<poly_n; j++) cap_tris[k++]=polygon[j];
                            for (int j=0; j<k; j++) polygon[j]=cap_tris[j];
                            poly_n=k;
                        }

                        // Ear-clip this polygon
                        if (poly_n >= 3) {
                            int* prev_a = ear_prevnext.raw();
                            int* next_a = ear_prevnext.raw() + poly_n;
                            for (int i = 0; i < poly_n; i++) {
                                prev_a[i]=(i+poly_n-1)%poly_n; next_a[i]=(i+1)%poly_n;
                            }
                            int remaining=poly_n, cur=0, max_iter=poly_n*poly_n, iter=0;
                            while (remaining > 3 && iter < max_iter) {
                                iter++;
                                int p=prev_a[cur], n=next_a[cur];
                                int vp=polygon[p], vc=polygon[cur], vn=polygon[n];
                                float up=all_verts[vp*3+pu],  vp_=all_verts[vp*3+pv_ax];
                                float uc=all_verts[vc*3+pu],  vc_=all_verts[vc*3+pv_ax];
                                float un=all_verts[vn*3+pu],  vn_=all_verts[vn*3+pv_ax];
                                float cross=(uc-up)*(vn_-vp_)-(vc_-vp_)*(un-up);
                                if (cross<=1e-10f) { cur=next_a[cur]; continue; }
                                int ear=1, chk=next_a[n];
                                while (chk != p) {
                                    int vi_=polygon[chk];
                                    float cu=all_verts[vi_*3+pu], cv=all_verts[vi_*3+pv_ax];
                                    if (fabsf(cu-up)+fabsf(cv-vp_)>1e-6f &&
                                        fabsf(cu-uc)+fabsf(cv-vc_)>1e-6f &&
                                        fabsf(cu-un)+fabsf(cv-vn_)>1e-6f &&
                                        pc_pt_in_tri(cu,cv,up,vp_,uc,vc_,un,vn_))
                                        { ear=0; break; }
                                    chk=next_a[chk];
                                }
                                if (ear) {
                                    cap_tris[n_cap*3]=vp; cap_tris[n_cap*3+1]=vc; cap_tris[n_cap*3+2]=vn;
                                    n_cap++; next_a[p]=n; prev_a[n]=p; remaining--; cur=n; iter=0;
                                } else { cur=next_a[cur]; }
                            }
                            if (remaining == 3) {
                                int p=prev_a[cur], n=next_a[cur];
                                cap_tris[n_cap*3]=polygon[p]; cap_tris[n_cap*3+1]=polygon[cur]; cap_tris[n_cap*3+2]=polygon[n];
                                n_cap++;
                            }
                        }
                    } // end for each outer loop

                    // lv, loop_starts, loop_sizes, sort buf no longer needed
                    heap_free(scratch_heap, lv2_ptr);   lv2_ptr = NULL;
                    heap_free(scratch_heap, ls_ptr);     ls_ptr  = NULL;
                    heap_free(scratch_heap, lsz_ptr);    lsz_ptr = NULL;
                    heap_free(scratch_heap, sb_ptr);      sb_ptr  = NULL;

                    // polygon and ear_prevnext no longer needed
                    heap_free(scratch_heap, poly_ptr); poly_ptr = NULL;
                    heap_free(scratch_heap, ep_ptr);   ep_ptr   = NULL;

                    // Cap winding + append to pos_tris / neg_tris
                    float n2d[3]={0,0,0};
                    if      (pu==0 && pv_ax==1) n2d[2]=1.0f;
                    else if (pu==1 && pv_ax==2) n2d[0]=1.0f;
                    else                        n2d[1]=-1.0f;
                    int flip_pos = (n2d[0]*(-pa)+n2d[1]*(-pb)+n2d[2]*(-pc_n) < 0) ? 1 : 0;
                    for (int i = 0; i < n_cap; i++) {
                        int pi=n_pos+i;
                        pos_tris[pi*3]=cap_tris[i*3];
                        if (flip_pos) { pos_tris[pi*3+1]=cap_tris[i*3+2]; pos_tris[pi*3+2]=cap_tris[i*3+1]; }
                        else          { pos_tris[pi*3+1]=cap_tris[i*3+1]; pos_tris[pi*3+2]=cap_tris[i*3+2]; }
                        int ni=n_neg+i;
                        neg_tris[ni*3]=cap_tris[i*3];
                        if (!flip_pos) { neg_tris[ni*3+1]=cap_tris[i*3+2]; neg_tris[ni*3+2]=cap_tris[i*3+1]; }
                        else           { neg_tris[ni*3+1]=cap_tris[i*3+1]; neg_tris[ni*3+2]=cap_tris[i*3+2]; }
                    }

                    // cap_tris no longer needed
                    heap_free(scratch_heap, cap_ptr); cap_ptr = NULL;
                } // kern_ok after loop-phase alloc
            } // n_boundary > 0
        } // kern_ok after lv alloc

        // Broadcast phase-13 state.  Local scratch lv_ptr..sb_ptr are all NULL
        // at this point (freed inline above); heap_free is NULL-safe.
        s_total_pos = n_pos + n_cap;
        s_total_neg = n_neg + n_cap;
        s_alloc_ok  = kern_ok;   // repurpose flag for phase-13 gate
        s_pr_ptr = NULL; s_nr_ptr = NULL;
        if (kern_ok) {
            void* p;
            if (heap_alloc(scratch_heap, (unsigned int)(n_all * (int)sizeof(int)), &p) == HEAP_OK)
                s_pr_ptr = (int*)p;
            else { s_alloc_ok = 0; atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
            if (s_alloc_ok) {
                if (heap_alloc(scratch_heap, (unsigned int)(n_all * (int)sizeof(int)), &p) == HEAP_OK)
                    s_nr_ptr = (int*)p;
                else { s_alloc_ok = 0; atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
            }
        }
        heap_free(scratch_heap, lv_ptr);
        heap_free(scratch_heap, lv2_ptr);
        heap_free(scratch_heap, ls_ptr);
        heap_free(scratch_heap, lsz_ptr);
        heap_free(scratch_heap, poly_ptr);
        heap_free(scratch_heap, cap_ptr);
        heap_free(scratch_heap, ep_ptr);
        heap_free(scratch_heap, sb_ptr);
    } // end if (tid == 0) phases 10-12
    __syncthreads();

    // =========================================================================
    // Phase 13: compact verts per side, heap-alloc output, fill PartPair.
    //           Parallel across all 64 threads.
    //
    // Steps:
    //   A  Init remap[-1]           (strided, all threads)
    //   B  Mark used verts [→ 0]    (strided, atomicMax, all threads)
    //   C  Count per-thread chunk   (contiguous chunks, all threads)
    //   D  Exclusive prefix scan    (thread 0, 64 iterations)
    //   E  Assign new indices        (contiguous chunks, all threads)
    //   F  Remap tris in-place      (strided, all threads)
    //   G  Heap alloc + metadata    (thread 0)
    //   H  Scatter verts to output  (strided, all threads)
    //   I  Copy tris to output      (strided, all threads)
    // =========================================================================
    if (!s_alloc_ok) {
        if (tid == 0) {
            heap_free(scratch_heap, (void*)s_pr_ptr);
            heap_free(scratch_heap, (void*)s_nr_ptr);
            PC_FREE_ALL_SHARED_SCRATCH();
        }
        return s_result;
    }

    {
        int   total_pos = s_total_pos;
        int   total_neg = s_total_neg;
        PC_BUF(int, pos_remap, s_pr_ptr, n_all);
        PC_BUF(int, neg_remap, s_nr_ptr, n_all);

        // Step A: init remap to -1 (parallel)
        for (int i = tid; i < n_all; i += PC_BLOCK) { pos_remap[i] = -1; neg_remap[i] = -1; }
        __syncthreads();

        // Step B: mark vertices used by each side (atomicMax -1 → 0, parallel)
        for (int t = tid; t < total_pos; t += PC_BLOCK) {
            int vi0=pos_tris[t*3+0], vi1=pos_tris[t*3+1], vi2=pos_tris[t*3+2];
            atomicMax(&pos_remap[vi0], 0);
            atomicMax(&pos_remap[vi1], 0);
            atomicMax(&pos_remap[vi2], 0);
        }
        for (int t = tid; t < total_neg; t += PC_BLOCK) {
            int vi0=neg_tris[t*3+0], vi1=neg_tris[t*3+1], vi2=neg_tris[t*3+2];
            atomicMax(&neg_remap[vi0], 0);
            atomicMax(&neg_remap[vi1], 0);
            atomicMax(&neg_remap[vi2], 0);
        }
        __syncthreads();

        // Step C: each thread counts marked verts in its contiguous chunk
        {
            int chunk = (n_all + PC_BLOCK - 1) / PC_BLOCK;
            int lo = tid * chunk, hi = lo + chunk; if (hi > n_all) hi = n_all;
            int pc = 0, nc = 0;
            for (int i = lo; i < hi; i++) {
                pc += (pos_remap[i] == 0);
                nc += (neg_remap[i] == 0);
            }
            s_pcount[tid] = pc;
            s_ncount[tid] = nc;
        }
        __syncthreads();

        // Step D: exclusive prefix scan over 64 counts (thread 0, O(64))
        if (tid == 0) {
            int sp = 0, sn = 0;
            for (int i = 0; i < PC_BLOCK; i++) {
                s_pbase[i] = sp; sp += s_pcount[i];
                s_nbase[i] = sn; sn += s_ncount[i];
            }
            s_pnv = sp; s_nnv = sn;
        }
        __syncthreads();

        // Step E: assign new sequential indices within each thread's chunk
        {
            int chunk = (n_all + PC_BLOCK - 1) / PC_BLOCK;
            int lo = tid * chunk, hi = lo + chunk; if (hi > n_all) hi = n_all;
            int pb = s_pbase[tid], nb = s_nbase[tid];
            for (int i = lo; i < hi; i++) {
                if (pos_remap[i] == 0) pos_remap[i] = pb++;
                if (neg_remap[i] == 0) neg_remap[i] = nb++;
            }
        }
        __syncthreads();

        // Step F: remap triangle indices in place (parallel)
        for (int i = tid; i < total_pos * 3; i += PC_BLOCK) {
            int vi = pos_tris[i];
            pos_tris[i] = pos_remap[vi];
        }
        for (int i = tid; i < total_neg * 3; i += PC_BLOCK) {
            int vi = neg_tris[i];
            neg_tris[i] = neg_remap[vi];
        }
        __syncthreads();

        // Step G: two separate heap allocs (pos and neg), each with own refcount.
        // Layout: [verts | tris | refcount(16B)], 16-byte aligned sections.
        if (tid == 0) {
            int pnv = s_pnv, nnv = s_nnv;
            unsigned int pv_b=(unsigned int)PC_ALIGN16(pnv      *3*(int)sizeof(float));
            unsigned int pt_b=(unsigned int)PC_ALIGN16(total_pos*3*(int)sizeof(int));
            unsigned int nv_b=(unsigned int)PC_ALIGN16(nnv      *3*(int)sizeof(float));
            unsigned int nt_b=(unsigned int)PC_ALIGN16(total_neg*3*(int)sizeof(int));
            unsigned int rc_b=(unsigned int)PC_ALIGN16((int)sizeof(int));
            unsigned int psz = pv_b+pt_b+rc_b; if (!psz) psz=1;
            unsigned int nsz = nv_b+nt_b+rc_b; if (!nsz) nsz=1;
            void* pchunk = NULL; void* nchunk = NULL;
            s_chunk_ok = 0;
            if (heap_alloc(heap, psz, &pchunk) == HEAP_OK &&
                heap_alloc(heap, nsz, &nchunk) == HEAP_OK) {
                s_chunk_ok = 1;
                char* pcb = (char*)pchunk;
                char* ncb = (char*)nchunk;
                s_pvp = (float*)(pcb);
                s_ptp = (int*)  (pcb+pv_b);
                int* rcp = (int*)(pcb+pv_b+pt_b); *rcp = 1;
                s_nvp = (float*)(ncb);
                s_ntp = (int*)  (ncb+nv_b);
                int* rcn = (int*)(ncb+nv_b+nt_b); *rcn = 1;
                pc_zero_part(&s_result.pos);
                s_result.pos.mesh.verts=s_pvp; s_result.pos.mesh.tris=s_ptp;
                s_result.pos.mesh.nv=pnv;      s_result.pos.mesh.nt=total_pos;
                s_result.pos.mesh.refcount=rcp;
                pc_zero_part(&s_result.neg);
                s_result.neg.mesh.verts=s_nvp; s_result.neg.mesh.tris=s_ntp;
                s_result.neg.mesh.nv=nnv;      s_result.neg.mesh.nt=total_neg;
                s_result.neg.mesh.refcount=rcn;
            } else {
                atomicOr(kernel_error, PC_KERR_POOL_OOM);
                if (pchunk) heap_free(heap, pchunk);
                s_pvp=NULL; s_ptp=NULL; s_nvp=NULL; s_ntp=NULL;
            }
        }
        __syncthreads();

        // Steps H-I: scatter verts and copy tris (parallel)
        if (s_chunk_ok) {
            float* pvp = s_pvp; int* ptp = s_ptp;
            float* nvp = s_nvp; int* ntp = s_ntp;

            for (int i = tid; i < n_all; i += PC_BLOCK) {
                int ni = pos_remap[i];
                if (ni >= 0) { pvp[ni*3+0]=all_verts[i*3+0]; pvp[ni*3+1]=all_verts[i*3+1]; pvp[ni*3+2]=all_verts[i*3+2]; }
            }
            for (int i = tid; i < n_all; i += PC_BLOCK) {
                int ni = neg_remap[i];
                if (ni >= 0) { nvp[ni*3+0]=all_verts[i*3+0]; nvp[ni*3+1]=all_verts[i*3+1]; nvp[ni*3+2]=all_verts[i*3+2]; }
            }
            for (int i = tid; i < total_pos * 3; i += PC_BLOCK) ptp[i] = pos_tris[i];
            for (int i = tid; i < total_neg * 3; i += PC_BLOCK) ntp[i] = neg_tris[i];
        }
        __syncthreads();
    }

    // Cleanup: free remap scratch and remaining shared scratch (NULL-safe).
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_pr_ptr);
        heap_free(scratch_heap, (void*)s_nr_ptr);
        PC_FREE_ALL_SHARED_SCRATCH();
    }
    __syncthreads();
    return s_result;
}
