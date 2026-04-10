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
//        Free sort_scratch; alloc all_verts (n_verts+n_cross, tighter bound).
//   4b   Copy original vertices into all_verts         (parallel, 64 threads)
//   5    Dedup + create intersection vertices          (thread 0)
//        Alloc pos_tris, neg_tris (phase-6 only).
//   6    Split triangles                               (parallel, 64 threads)
//        Free signs, cross_edges, isect_idx.
//   7    Collect directed edges from pos_tris          (parallel, 64 threads)
//   8    Sort directed edges                           (warp 0)
//        Free dir_sort.
//   9    Binary search for boundary edges              (parallel, 64 threads)
//  10-12 Compact boundary → loops → polygon → ear-clip (thread 0)
//        Cap tris kept in separate cap_tris buffer (sized from n_boundary).
//  13    Compact verts per side, heap-alloc output,    (parallel, 64 threads)
//        merge cap tris with winding + remap.
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

// Comparator for directed edges sorted by unordered (min,max) key.
// Used in phase 8 so that (a,b) and (b,a) land adjacent after sort.
struct Edge2iNormCmp {
    static __device__ inline int cmp(Edge2i x, Edge2i y) {
        int xlo = min(x.a, x.b), xhi = max(x.a, x.b);
        int ylo = min(y.a, y.b), yhi = max(y.a, y.b);
        if (xlo != ylo) return (xlo < ylo) ? -1 : 1;
        if (xhi != yhi) return (xhi < yhi) ? -1 : 1;
        return 0;
    }
    static __device__ inline Edge2i sentinel() {
        Edge2i s; s.a = 0x7fffffff; s.b = 0x7fffffff; return s;
    }
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ inline int pc_edge_bsearch(const Edge2i* __restrict__ arr, int n, int key_a, int key_b) {
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

__device__ inline float pc_signed_area(const float* __restrict__ verts, const int* __restrict__ poly, int n, int pu, int pv) {
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

__device__ inline void pc_zero_part(Part* __restrict__ p) {
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
    heap_free(scratch_heap, (void*)s_be_a);                        \
    heap_free(scratch_heap, (void*)s_cap_tris);                    \
} while(0)

__device__ inline PartPair plane_cut_block(
    // Input
    const Mesh* __restrict__ mesh,
    float pa, float pb, float pc_n, float pd,
    DeviceHeap* __restrict__ heap,          // output heap (persistent mesh data)
    DeviceHeap* __restrict__ scratch_heap,  // scratch heap (all allocs freed within this call)
    int* __restrict__ kernel_error)
{
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;
    long long t_start = clock64();

    const float* vertices  = mesh->verts;
    const int*   triangles = mesh->tris;
    int          n_verts   = mesh->nv;
    int          n_tris    = mesh->nt;

    // Upper-bound sizes (all derived from n_verts / n_tris)
    int max_cross_edges    = n_tris * 2;
    // Cap triangles are kept in a separate cap_tris buffer (sized from n_boundary after phase 9b)
    // and merged into the output during phase 13.
    // sort_scratch_bytes computed after phase 3 when n_cross is known.
    // dir_edges/dir_sort/boundary_flags are now sized from actual n_trace after phase 6.

    // s_counters: [0]=n_cross, [1]=n_all_verts, [2]=n_pos, [3]=n_neg
    __shared__ int s_counters[4];
    __shared__ int s_alloc_ok;

    // Shared scratch pointers — all initialised to NULL so heap_free is always safe.
    __shared__ signed char* s_signs;
    __shared__ float*  s_all_verts;
    __shared__ Edge2i* s_cross_edges;
    __shared__ char*   s_sort_scratch;
    __shared__ int*    s_isect_idx;
    __shared__ int*    s_pos_tris;
    __shared__ int*    s_neg_tris;
    __shared__ Edge2i* s_dir_edges;
    __shared__ char*   s_dir_sort;
    __shared__ char*   s_boundary_flags;
    __shared__ int*    s_be_a;
    __shared__ int*    s_cap_tris;
    __shared__ int     s_n_boundary;
    __shared__ int     s_n_cap;
    __shared__ int     s_flip_pos;

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

    // Warp-cooperative phases 10-12: shared broadcast slots for heap pointers
    __shared__ void*  s_w_ptrs[2];   // loop_blk, cap_ptr
    __shared__ int    s_w_kern_ok;
    __shared__ int    s_w_n_loops;
    __shared__ int    s_mid_lo, s_mid_hi;
    if (tid == 0) {
        s_signs = NULL; s_all_verts = NULL;
        s_cross_edges = NULL; s_sort_scratch = NULL; s_isect_idx = NULL;
        s_pos_tris = NULL; s_neg_tris = NULL;
        s_dir_edges = NULL; s_dir_sort = NULL; s_boundary_flags = NULL; s_be_a = NULL; s_cap_tris = NULL;
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
        if (heap_alloc(scratch_heap, (unsigned int)(n_verts * (int)sizeof(signed char)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_signs = (signed char*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(signed char, signs, s_signs, n_verts);

    // === Phase 1: Classify vertices ===
    for (int v = tid; v < n_verts; v += PC_BLOCK) {
        float val = pa*vertices[v*3] + pb*vertices[v*3+1] + pc_n*vertices[v*3+2] + pd;
        signs[v] = (val > PC_EPS) ? 1 : ((val < -PC_EPS) ? -1 : 0);
    }
    __syncthreads();

    // =========================================================================
    // Alloc cross_edges (needed: phase 3; sort_scratch + isect_idx deferred)
    // =========================================================================
    if (tid == 0) {
        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(max_cross_edges * (int)sizeof(Edge2i)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_cross_edges = (Edge2i*)p;

        // sort_scratch and isect_idx deferred to after phase 3 when n_cross is known.
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(Edge2i, cross_edges, s_cross_edges, max_cross_edges);

    // === Phase 3: Collect crossing edges ===
    // An edge crosses the plane if its effective signs differ.
    // Sign=0 (on-plane) is treated as positive for crossing detection,
    // matching CoACD's convention: edges (0,-1) and (-1,0) are crossing
    // but (0,+1) and (+1,0) are not.
    for (int t = tid; t < n_tris; t += PC_BLOCK) {
        int i0 = triangles[t*3], i1 = triangles[t*3+1], i2 = triangles[t*3+2];
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];
        int edges[3][2]  = {{i0,i1},{i1,i2},{i2,i0}};
        int esigns[3][2] = {{s0,s1},{s1,s2},{s2,s0}};
        for (int e = 0; e < 3; e++) {
            int sa = esigns[e][0], sb = esigns[e][1];
            int ea = (sa >= 0) ? 1 : -1;
            int eb = (sb >= 0) ? 1 : -1;
            if (ea != eb) {
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
    // Each crossing triangle contributes ≤1 extra tri per side; a tri with 1 crossing edge
    // (vertex on plane) can count as 1 crossing, so use n_cross*2 for safety.
    int max_out_tris = min(n_tris + n_cross * 2, n_tris * 3);

    if (n_cross == 0) {
        // No crossing edges.  Check whether vertices exist on both sides
        // (disjoint components separated by the plane).
        if (tid == 0) {
            int any_pos = 0, any_neg = 0;
            for (int v = 0; v < n_verts; v++) {
                if (signs[v] > 0) any_pos = 1;
                if (signs[v] < 0) any_neg = 1;
            }
            if (!any_pos || !any_neg) {
                // All on one side — copy entire mesh.
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
            } else {
                // Disjoint components on both sides — separate triangles.
                // Classify each triangle by its vertices' signs.
                // On-plane vertices (sign=0): triangle goes to whichever side
                // it has a non-zero vertex on; all-zero triangles go to pos.

                // Count triangles per side.
                int n_pos_t = 0, n_neg_t = 0;
                for (int t = 0; t < n_tris; t++) {
                    int s0 = signs[triangles[t*3]], s1 = signs[triangles[t*3+1]], s2 = signs[triangles[t*3+2]];
                    if (s0 < 0 || s1 < 0 || s2 < 0) n_neg_t++;
                    else n_pos_t++;
                }

                // Build vertex remap: vertex → new index per side.
                // Use scratch heap for the remap array (-1 = not used on this side).
                int* vmap = NULL;
                void* vmap_p;
                if (heap_alloc(scratch_heap, (unsigned int)(n_verts * (int)sizeof(int)), &vmap_p) != HEAP_OK) {
                    atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);
                    PC_FREE_ALL_SHARED_SCRATCH();
                } else {
                    vmap = (int*)vmap_p;

                    // --- Positive side ---
                    for (int v = 0; v < n_verts; v++) vmap[v] = -1;
                    int pnv = 0;
                    for (int t = 0; t < n_tris; t++) {
                        int i0=triangles[t*3], i1=triangles[t*3+1], i2=triangles[t*3+2];
                        int s0=signs[i0], s1=signs[i1], s2=signs[i2];
                        if (s0 < 0 || s1 < 0 || s2 < 0) continue;
                        if (vmap[i0]<0) vmap[i0]=pnv++;
                        if (vmap[i1]<0) vmap[i1]=pnv++;
                        if (vmap[i2]<0) vmap[i2]=pnv++;
                    }
                    unsigned int pvb=(unsigned int)PC_ALIGN16(pnv*3*(int)sizeof(float));
                    unsigned int ptb=(unsigned int)PC_ALIGN16(n_pos_t*3*(int)sizeof(int));
                    unsigned int prc=(unsigned int)PC_ALIGN16((int)sizeof(int));
                    unsigned int psz=pvb+ptb+prc; if (!psz) psz=1;
                    void* pchunk=NULL;
                    if (heap_alloc(heap, psz, &pchunk) != HEAP_OK) {
                        atomicOr(kernel_error, PC_KERR_POOL_OOM);
                    } else {
                        float* opv=(float*)pchunk;
                        int*   opt=(int*)((char*)pchunk+pvb);
                        int*   orc=(int*)((char*)pchunk+pvb+ptb);
                        *orc=1;
                        for (int v=0; v<n_verts; v++) {
                            if (vmap[v]>=0) { int j=vmap[v]; opv[j*3]=vertices[v*3]; opv[j*3+1]=vertices[v*3+1]; opv[j*3+2]=vertices[v*3+2]; }
                        }
                        int ti=0;
                        for (int t=0; t<n_tris; t++) {
                            int i0=triangles[t*3], i1=triangles[t*3+1], i2=triangles[t*3+2];
                            int s0=signs[i0], s1=signs[i1], s2=signs[i2];
                            if (s0 < 0 || s1 < 0 || s2 < 0) continue;
                            opt[ti*3]=vmap[i0]; opt[ti*3+1]=vmap[i1]; opt[ti*3+2]=vmap[i2]; ti++;
                        }
                        pc_zero_part(&s_result.pos);
                        s_result.pos.mesh.verts=opv; s_result.pos.mesh.tris=opt;
                        s_result.pos.mesh.nv=pnv; s_result.pos.mesh.nt=n_pos_t;
                        s_result.pos.mesh.refcount=orc;
                    }

                    // --- Negative side ---
                    for (int v = 0; v < n_verts; v++) vmap[v] = -1;
                    int nnv = 0;
                    for (int t = 0; t < n_tris; t++) {
                        int i0=triangles[t*3], i1=triangles[t*3+1], i2=triangles[t*3+2];
                        int s0=signs[i0], s1=signs[i1], s2=signs[i2];
                        if (!(s0 < 0 || s1 < 0 || s2 < 0)) continue;
                        if (vmap[i0]<0) vmap[i0]=nnv++;
                        if (vmap[i1]<0) vmap[i1]=nnv++;
                        if (vmap[i2]<0) vmap[i2]=nnv++;
                    }
                    unsigned int nvb=(unsigned int)PC_ALIGN16(nnv*3*(int)sizeof(float));
                    unsigned int ntb=(unsigned int)PC_ALIGN16(n_neg_t*3*(int)sizeof(int));
                    unsigned int nrc=(unsigned int)PC_ALIGN16((int)sizeof(int));
                    unsigned int nsz=nvb+ntb+nrc; if (!nsz) nsz=1;
                    void* nchunk=NULL;
                    if (heap_alloc(heap, nsz, &nchunk) != HEAP_OK) {
                        atomicOr(kernel_error, PC_KERR_POOL_OOM);
                    } else {
                        float* onv=(float*)nchunk;
                        int*   ont=(int*)((char*)nchunk+nvb);
                        int*   orc=(int*)((char*)nchunk+nvb+ntb);
                        *orc=1;
                        for (int v=0; v<n_verts; v++) {
                            if (vmap[v]>=0) { int j=vmap[v]; onv[j*3]=vertices[v*3]; onv[j*3+1]=vertices[v*3+1]; onv[j*3+2]=vertices[v*3+2]; }
                        }
                        int ti=0;
                        for (int t=0; t<n_tris; t++) {
                            int i0=triangles[t*3], i1=triangles[t*3+1], i2=triangles[t*3+2];
                            int s0=signs[i0], s1=signs[i1], s2=signs[i2];
                            if (!(s0 < 0 || s1 < 0 || s2 < 0)) continue;
                            ont[ti*3]=vmap[i0]; ont[ti*3+1]=vmap[i1]; ont[ti*3+2]=vmap[i2]; ti++;
                        }
                        pc_zero_part(&s_result.neg);
                        s_result.neg.mesh.verts=onv; s_result.neg.mesh.tris=ont;
                        s_result.neg.mesh.nv=nnv; s_result.neg.mesh.nt=n_neg_t;
                        s_result.neg.mesh.refcount=orc;
                    }

                    heap_free(scratch_heap, vmap_p);
                }
            }
            PC_FREE_ALL_SHARED_SCRATCH();
        }
        __syncthreads();
        return s_result;
    }

    // Alloc sort_scratch + isect_idx now that n_cross is known.
    if (tid == 0) {
        void* p;
        int sort_scratch_bytes = n_cross * (int)sizeof(Edge2i) + WS_MAX_STACK * 2 * (int)sizeof(int);
        if (heap_alloc(scratch_heap, (unsigned int)sort_scratch_bytes, &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_sort_scratch = (char*)p;

        if (heap_alloc(scratch_heap, (unsigned int)(n_cross * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_isect_idx = (int*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(int, isect_idx, s_isect_idx, n_cross);

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
    // =========================================================================
    // Alloc all_verts now that n_cross is known; tighter bound than max_new_verts.
    // =========================================================================
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_sort_scratch); s_sort_scratch = NULL;

        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)((n_verts + n_cross) * 3 * (int)sizeof(float)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_all_verts = (float*)p;
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
        return s_result;
    }

    PC_BUF(float, all_verts, s_all_verts, (n_verts + n_cross) * 3);

    // === Phase 4b: Copy original vertices ===
    for (int i = tid; i < n_verts * 3; i += PC_BLOCK)
        all_verts[i] = vertices[i];
    __syncthreads();

    // =========================================================================
    // Phase 5: Dedup + intersection vertices  (thread 0)
    // Also: alloc pos_tris + neg_tris.
    // =========================================================================
    if (tid == 0) {
        void* p;
        s_alloc_ok = 1;
        if (heap_alloc(scratch_heap, (unsigned int)(max_out_tris * 3 * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_pos_tris = (int*)p;

        if (heap_alloc(scratch_heap, (unsigned int)(max_out_tris * 3 * (int)sizeof(int)), &p) != HEAP_OK)
            { s_alloc_ok = 0; p = NULL; }
        s_neg_tris = (int*)p;

        int cur_isect = n_verts;
        for (int i = 0; i < n_cross; i++) {
            if (i == 0 || cross_edges[i].a != cross_edges[i-1].a || cross_edges[i].b != cross_edges[i-1].b) {
                int va = cross_edges[i].a, vb = cross_edges[i].b;
                // For edges where one endpoint is on the plane (sign=0),
                // the intersection is that vertex itself — reuse its index
                // to avoid creating a duplicate that would break topology.
                float d0 = pa*all_verts[va*3]+pb*all_verts[va*3+1]+pc_n*all_verts[va*3+2]+pd;
                float d1 = pa*all_verts[vb*3]+pb*all_verts[vb*3+1]+pc_n*all_verts[vb*3+2]+pd;
                if (fabsf(d0) <= PC_EPS && fabsf(d1) > PC_EPS) {
                    isect_idx[i] = va;  // va is on-plane → intersection is va
                } else if (fabsf(d1) <= PC_EPS && fabsf(d0) > PC_EPS) {
                    isect_idx[i] = vb;  // vb is on-plane → intersection is vb
                } else {
                    pc_intersect(all_verts.raw(), va, vb, pa, pb, pc_n, pd, &all_verts[cur_isect * 3]);
                    isect_idx[i] = cur_isect++;
                }
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
    int  tris_cap = max_out_tris * 3;
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
            int ov=vi[on], a_i=vi[(on+1)%3], b_i=vi[(on+2)%3], sa=si[(on+1)%3], sb=si[(on+2)%3];
            if (sa == sb) {
                // Both non-on-plane vertices on same side → triangle stays whole on that side.
                // The on-plane vertex is part of the boundary but contributes zero area
                // on the other side (CoACD convention: on-plane goes with the off-plane side).
                if (sa > 0) {
                    int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=i0;pos_tris[pi*3+1]=i1;pos_tris[pi*3+2]=i2;
                } else {
                    int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=i0;neg_tris[ni*3+1]=i1;neg_tris[ni*3+2]=i2;
                }
            } else {
                // Non-on-plane vertices have opposite signs — split at on-plane vertex.
                // The intersection of the a_i-b_i edge with the plane divides the triangle.
                int nvi = FIND_ISECT(a_i, b_i); if (nvi < 0) continue;
                if (sa > 0) {
                    int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov;pos_tris[pi*3+1]=a_i;pos_tris[pi*3+2]=nvi;
                    int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov;neg_tris[ni*3+1]=nvi;neg_tris[ni*3+2]=b_i;
                } else {
                    int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov;neg_tris[ni*3+1]=a_i;neg_tris[ni*3+2]=nvi;
                    int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov;pos_tris[pi*3+1]=nvi;pos_tris[pi*3+2]=b_i;
                }
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
        // 4 stack arrays (lo+hi for each of 2 warps) instead of 2.
        int de_sort_bytes = n_de_alloc * (int)sizeof(Edge2i) + WS_MAX_STACK * 4 * (int)sizeof(int);

        void* p;
        s_alloc_ok = 1;
        if (n_de_alloc > 0) {
            if (heap_alloc(scratch_heap, (unsigned int)(n_de_alloc * (int)sizeof(Edge2i)), &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_dir_edges = (Edge2i*)p;

            if (heap_alloc(scratch_heap, (unsigned int)de_sort_bytes, &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_dir_sort = (char*)p;

            if (heap_alloc(scratch_heap, (unsigned int)(n_de_alloc * (int)sizeof(char)), &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; }
            s_boundary_flags = (char*)p;
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
    PC_BUF(char,   boundary_flags, s_boundary_flags,  n_de);

    // === Phase 7: Collect directed edges from the smaller side ===
    for (int t = tid; t < n_trace; t += PC_BLOCK) {
        int a = trace_tris[t*3], b = trace_tris[t*3+1], c = trace_tris[t*3+2];
        int bi = t * 3;
        dir_edges[bi  ] = (Edge2i){a, b};
        dir_edges[bi+1] = (Edge2i){b, c};
        dir_edges[bi+2] = (Edge2i){c, a};
    }
    __syncthreads();

    // === Phase 8: Sort directed edges — two-warp parallel quicksort ===
    // Scratch layout: tmp (n_de * 8B) | stack_lo_w0 | stack_hi_w0 | stack_lo_w1 | stack_hi_w1
    // (each stack array: WS_MAX_STACK ints)
    {
        Edge2i* dir_tmp   = (Edge2i*)s_dir_sort;
        int* stack_lo_w0  = (int*)(s_dir_sort + n_de * (int)sizeof(Edge2i));
        int* stack_hi_w0  = stack_lo_w0 + WS_MAX_STACK;
        int* stack_lo_w1  = stack_hi_w0 + WS_MAX_STACK;
        int* stack_hi_w1  = stack_lo_w1 + WS_MAX_STACK;

        if (n_de <= 32) {
            // Small: single bitonic sort by warp 0, nothing left for phase 8b.
            if (warp_id == 0) {
                int serr = warp_sort_t<Edge2i, Edge2iNormCmp>(dir_edges.raw(), s_dir_sort, n_de, lane);
                if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
                if (lane == 0) { s_mid_lo = 0; s_mid_hi = n_de; }
            }
        } else {
            // Phase 8a: warp 0 does one pivot+partition pass.
            if (warp_id == 0) {
                Edge2i pivot = warp_pick_pivot<Edge2i, Edge2iNormCmp>(dir_edges.raw(), 0, n_de, lane);
                int mid_lo, mid_hi;
                warp_partition<Edge2i, Edge2iNormCmp>(
                    dir_edges.raw(), dir_tmp, 0, n_de, pivot, &mid_lo, &mid_hi, lane);
                if (lane == 0) { s_mid_lo = mid_lo; s_mid_hi = mid_hi; }
            }
        }
        __syncthreads();

        // Phase 8b: warp 0 sorts left half [0..mid_lo), warp 1 sorts right half [mid_hi..n_de).
        if (n_de > 32) {
            int mid_lo = s_mid_lo, mid_hi = s_mid_hi;
            if (warp_id == 0 && mid_lo > 1) {
                int serr = warp_sort_inner<Edge2i, Edge2iNormCmp>(
                    dir_edges.raw(), dir_tmp, stack_lo_w0, stack_hi_w0, 0, mid_lo, lane);
                if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
            }
            if (warp_id == 1 && (n_de - mid_hi) > 1) {
                int serr = warp_sort_inner<Edge2i, Edge2iNormCmp>(
                    dir_edges.raw(), dir_tmp, stack_lo_w1, stack_hi_w1, mid_hi, n_de, lane);
                if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
            }
        }
    }
    __syncthreads();

    // Free dir_sort — not needed after sort.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_dir_sort); s_dir_sort = NULL;
    }

    // === Phase 9: Detect boundary edges via neighbor scan ===
    // After sorting by (min,max) key, internal edge pairs (a,b)+(b,a) are adjacent.
    // A boundary edge has no normalized-equal neighbor on either side.
    for (int i = tid; i < n_de; i += PC_BLOCK) {
        Edge2i e = dir_edges[i];
        int elo = min(e.a, e.b), ehi = max(e.a, e.b);
        bool prev_dup = false, next_dup = false;
        if (i > 0) {
            Edge2i p = dir_edges[i-1];
            prev_dup = (min(p.a,p.b)==elo && max(p.a,p.b)==ehi);
        }
        if (i < n_de-1) {
            Edge2i n = dir_edges[i+1];
            next_dup = (min(n.a,n.b)==elo && max(n.a,n.b)==ehi);
        }
        boundary_flags[i] = (prev_dup || next_dup) ? 0 : 1;
    }
    __syncthreads();

    // === Phase 9b: Parallel compact boundary edges ===
    if (tid == 0) {
        s_n_boundary = 0;
        s_be_a = NULL;
        if (n_de > 0) {
            void* p;
            if (heap_alloc(scratch_heap, (unsigned int)(n_de * 2 * (int)sizeof(int)), &p) != HEAP_OK)
                { s_alloc_ok = 0; p = NULL; atomicOr(kernel_error, PC_KERR_SCRATCH_OOM); }
            s_be_a = (int*)p;
        }
    }
    __syncthreads();
    if (!s_alloc_ok) {
        if (tid == 0) { PC_FREE_ALL_SHARED_SCRATCH(); }
        return s_result;
    }
    for (int i = tid; i < n_de; i += PC_BLOCK) {
        if (boundary_flags[i]) {
            int wi = atomicAdd(&s_n_boundary, 1);
            s_be_a[wi*2]   = dir_edges[i].a;
            s_be_a[wi*2+1] = dir_edges[i].b;
        }
    }
    __syncthreads();
    // dir_edges and boundary_flags no longer needed.
    if (tid == 0) {
        heap_free(scratch_heap, (void*)s_dir_edges);       s_dir_edges      = NULL;
        heap_free(scratch_heap, (void*)s_boundary_flags);  s_boundary_flags = NULL;
    }
    __syncthreads();

    // =========================================================================
    // Phases 10-12: warp 0 cooperative.
    //
    // Lane 0 handles heap alloc/free and serial control flow.
    // All 32 lanes cooperate on inner scans (edge chain search, ray-cast
    // point-in-polygon, ear-clip point-in-triangle) for ~32× speedup on
    // those hot loops.
    // At the end, kern_ok + total_pos/neg + remap ptrs are broadcast to all
    // threads via shared memory for the parallel phase 13.
    // =========================================================================
    if (warp_id == 0) {
        int kern_ok    = 1;
        int n_cap      = 0;
        int n_boundary = s_n_boundary;
        if (lane == 0) s_w_n_loops = 0;
        void* lv_ptr   = (void*)s_be_a;  // boundary edge pairs (be_a), freed after loop recon

        // Lane-0-only scratch — all NULL so heap_free is always safe.
        void* loop_blk = NULL;  // merged block for ls/lsz/lv2/poly/ep/sb
        void* cap_ptr  = NULL;  // cap_tris (separate: ownership transfers to s_cap_tris)

            // --- Phase 10-12: loop reconstruction + polygon + ear-clip ---
            if (n_boundary > 0) {
                if (lane == 0) {
                    // Merged allocation: ls + lsz + lv2 + poly + ep + sb
                    // All int-sized, so natural alignment within the block is fine.
                    int poly_cap_ = n_boundary * 4 + 64;
                    int sb_ints = 256 + 256;  // inner_idx(256 int) + inner_max_u(256 float)
                    unsigned int loop_blk_sz = (unsigned int)(
                        (n_boundary * 3            // ls + lsz + lv2
                         + poly_cap_               // polygon
                         + poly_cap_ * 2           // ear_prevnext
                         + sb_ints                 // inner sort buf
                        ) * (int)sizeof(int));
                    if (heap_alloc(scratch_heap, loop_blk_sz, &loop_blk) != HEAP_OK) { kern_ok = 0; }
                    if (kern_ok && heap_alloc(scratch_heap, (unsigned int)(poly_cap_ * (int)sizeof(int)), &cap_ptr) != HEAP_OK) { kern_ok = 0; }
                    if (!kern_ok) atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);
                    s_w_ptrs[0] = loop_blk; s_w_ptrs[1] = cap_ptr;
                    s_w_kern_ok = kern_ok;
                }
                __syncwarp();
                kern_ok = s_w_kern_ok;
                loop_blk = s_w_ptrs[0]; cap_ptr = s_w_ptrs[1];

                if (kern_ok) {
                    int poly_cap = n_boundary * 4 + 64;
                    // Carve sub-buffers from merged block
                    int* blk_base       = (int*)loop_blk;
                    int* ls_raw         = blk_base;
                    int* lsz_raw        = ls_raw   + n_boundary;
                    int* lv2_raw        = lsz_raw  + n_boundary;
                    int* poly_raw       = lv2_raw  + n_boundary;
                    int* ep_raw         = poly_raw + poly_cap;
                    int* sb_raw         = ep_raw   + poly_cap * 2;
                    PC_BUF(int,   loop_starts,  ls_raw,   n_boundary);
                    PC_BUF(int,   loop_sizes,   lsz_raw,  n_boundary);
                    PC_BUF(int,   lv,           lv2_raw,  n_boundary);
                    PC_BUF(int,   polygon,      poly_raw, poly_cap);
                    PC_BUF(int,   cap_tris,     cap_ptr,  poly_cap * 3);
                    PC_BUF(int,   ear_prevnext, ep_raw,   poly_cap * 2);
                    PC_BUF(int,   inner_idx,    sb_raw,   256);
                    float* inner_max_u  = (float*)(sb_raw + 256);

                    // Phase 10: reconstruct loops from boundary edge pairs
                    // Lane 0 drives chain-following; all lanes help with inner search.
                    PC_BUF(int, be_a2, lv_ptr, n_boundary * 2);
                    int* be_used = loop_starts.raw();  // borrow before it's filled
                    for (int i = lane; i < n_boundary; i += 32) be_used[i] = 0;
                    __syncwarp();

                    int n_loops = 0, lvi = 0;
                    for (int start = 0; start < n_boundary; start++) {
                        if (be_used[start]) continue;
                        int lvi_start = lvi;
                        int first = be_a2[start*2], cur = be_a2[start*2+1];
                        be_used[start] = 1;  // aliases loop_starts[start]; loop_starts[n_loops] set below
                        if (lvi >= n_boundary) break;
                        if (lane == 0) lv[lvi] = first;
                        lvi++;
                        int safety = n_boundary + 2;
                        while (cur != first && safety-- > 0) {
                            if (lvi >= n_boundary) break;
                            if (lane == 0) lv[lvi] = cur;
                            lvi++;
                            // Warp-parallel edge search: each lane checks a stride
                            int my_found_i = -1;
                            for (int i = lane; i < n_boundary; i += 32) {
                                if (!be_used[i] && be_a2[i*2] == cur) {
                                    my_found_i = i; break;
                                }
                            }
                            unsigned mask = __ballot_sync(0xFFFFFFFF, my_found_i >= 0);
                            if (mask == 0) break;  // not found
                            int winner = __ffs(mask) - 1;  // lowest lane with a match
                            int found_i = __shfl_sync(0xFFFFFFFF, my_found_i, winner);
                            cur = be_a2[found_i*2+1];
                            be_used[found_i] = 1;
                        }
                        if (lane == 0) {
                            loop_starts[n_loops] = lvi_start;
                            loop_sizes[n_loops] = lvi - lvi_start;
                        }
                        __syncwarp();
                        if (loop_sizes[n_loops] > 0) n_loops++;
                    }
                    if (lane == 0) s_w_n_loops = n_loops;

                    // be_a (lv_ptr) no longer needed — free to reclaim memory.
                    if (lane == 0) { heap_free(scratch_heap, lv_ptr); lv_ptr = NULL; s_be_a = NULL; }
                    __syncwarp();

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
                        if (lane == 0) {
                            float a = pc_signed_area(all_verts.raw(), lv.raw()+loop_starts[i], loop_sizes[i], pu, pv_ax);
                            if (a < 0) pc_reverse(lv.raw()+loop_starts[i], loop_sizes[i]);
                        }
                    }
                    __syncwarp();

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
                            // Warp-parallel ray-casting point-in-polygon (2D, along +u axis)
                            int* lp = lv.raw() + loop_starts[j];
                            int ls_ = loop_sizes[j];
                            int my_crossings = 0;
                            for (int k = lane; k < ls_; k += 32) {
                                int kn = (k+1) % ls_;
                                float ay = all_verts[lp[k]*3+pv_ax], by = all_verts[lp[kn]*3+pv_ax];
                                if ((ay <= pv_) == (by <= pv_)) continue;
                                float ax = all_verts[lp[k]*3+pu], bx = all_verts[lp[kn]*3+pu];
                                float t = (pv_ - ay) / (by - ay);
                                if (ax + t * (bx - ax) > pu_) my_crossings++;
                            }
                            // Warp-reduce crossings
                            for (int s = 16; s > 0; s >>= 1)
                                my_crossings += __shfl_xor_sync(0xFFFFFFFF, my_crossings, s);
                            if (my_crossings & 1) {
                                // i is inside j — pick smallest enclosing
                                float a = fabsf(pc_signed_area(all_verts.raw(), lp, ls_, pu, pv_ax));
                                if (a < best_area) { best_area = a; if (lane == 0) parent[i] = j; }
                            }
                        }
                        __syncwarp();
                        if (parent[i] >= 0) is_hole[i] = 1;
                    }

                    // Reverse hole loops to CW winding (they were oriented CCW above).
                    for (int i = 0; i < n_loops && i < max_loops; i++) {
                        if (lane == 0 && is_hole[i]) pc_reverse(lv.raw()+loop_starts[i], loop_sizes[i]);
                    }
                    __syncwarp();

                    // Process each outer loop (non-hole) and its direct children.
                    for (int oi = 0; oi < n_loops && oi < max_loops; oi++) {
                        if (is_hole[oi] || loop_sizes[oi] < 3) continue;

                        // Copy outer loop into polygon (lane 0)
                        int poly_n = loop_sizes[oi];
                        if (lane == 0) {
                            for (int i = 0; i < poly_n; i++)
                                polygon[i] = lv[loop_starts[oi]+i];
                        }
                        __syncwarp();

                        // Collect direct holes (parent == oi), sorted by rightmost u descending
                        // (lane 0 only — small n_inner)
                        int n_inner = 0;
                        if (lane == 0) {
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
                        }
                        n_inner = __shfl_sync(0xFFFFFFFF, n_inner, 0);

                        // Bridge each hole into polygon (lane 0 — modifies polygon)
                        if (lane == 0) {
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
                                    float s_ = (mv_-av2)/dv;
                                    if (s_<-1e-10f||s_>1.0f+1e-10f) continue;
                                    float tv = au+s_*(bu-au)-mu_;
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
                        }
                        poly_n = __shfl_sync(0xFFFFFFFF, poly_n, 0);
                        __syncwarp();

                        // Ear-clip this polygon (warp-cooperative)
                        if (poly_n >= 3) {
                            int* prev_a = ear_prevnext.raw();
                            int* next_a = ear_prevnext.raw() + poly_n;
                            for (int i = lane; i < poly_n; i += 32) {
                                prev_a[i]=(i+poly_n-1)%poly_n; next_a[i]=(i+1)%poly_n;
                            }
                            __syncwarp();
                            int remaining=poly_n, cur=0, iter=0;
                            while (remaining > 3 && iter < remaining) {
                                iter++;
                                int p=prev_a[cur], n=next_a[cur];
                                int vp=polygon[p], vc=polygon[cur], vn=polygon[n];
                                float up=all_verts[vp*3+pu],  vp_=all_verts[vp*3+pv_ax];
                                float uc=all_verts[vc*3+pu],  vc_=all_verts[vc*3+pv_ax];
                                float un=all_verts[vn*3+pu],  vn_=all_verts[vn*3+pv_ax];
                                float cross=(uc-up)*(vn_-vp_)-(vc_-vp_)*(un-up);
                                if (cross<=1e-10f) { cur=next_a[cur]; continue; }
                                // Warp-parallel point-in-triangle test: scan all
                                // polygon indices, skip removed vertices.
                                int ear=1;
                                for (int idx = lane; idx < poly_n; idx += 32) {
                                    if (idx == p || idx == cur || idx == n) continue;
                                    // Check if vertex is still active (not removed)
                                    if (next_a[prev_a[idx]] != idx) continue;
                                    int vi_=polygon[idx];
                                    float cu=all_verts[vi_*3+pu], cv=all_verts[vi_*3+pv_ax];
                                    if (fabsf(cu-up)+fabsf(cv-vp_)>1e-6f &&
                                        fabsf(cu-uc)+fabsf(cv-vc_)>1e-6f &&
                                        fabsf(cu-un)+fabsf(cv-vn_)>1e-6f &&
                                        pc_pt_in_tri(cu,cv,up,vp_,uc,vc_,un,vn_))
                                        { ear=0; }
                                }
                                if (__ballot_sync(0xFFFFFFFF, !ear))
                                    ear = 0;
                                else
                                    ear = 1;
                                if (ear) {
                                    if (lane == 0) {
                                        cap_tris[n_cap*3]=vp; cap_tris[n_cap*3+1]=vc; cap_tris[n_cap*3+2]=vn;
                                        n_cap++; next_a[p]=n; prev_a[n]=p;
                                    }
                                    remaining--; cur=n; iter=0;
                                    __syncwarp();  // ensure prev_a/next_a visible
                                } else { cur=next_a[cur]; }
                            }
                            if (remaining == 3 && lane == 0) {
                                int p=prev_a[cur], n=next_a[cur];
                                cap_tris[n_cap*3]=polygon[p]; cap_tris[n_cap*3+1]=polygon[cur]; cap_tris[n_cap*3+2]=polygon[n];
                                n_cap++;
                            }
                            __syncwarp();
                        }
                    } // end for each outer loop

                    // Merged loop block no longer needed
                    if (lane == 0) {
                        heap_free(scratch_heap, loop_blk); loop_blk = NULL;

                        // Compute cap winding flip; broadcast cap state to shared memory.
                        // Cap tris stay in cap_ptr and are merged into output during phase 13.
                        {
                            float n2d[3]={0,0,0};
                            if      (pu==0 && pv_ax==1) n2d[2]=1.0f;
                            else if (pu==1 && pv_ax==2) n2d[0]=1.0f;
                            else                        n2d[1]=-1.0f;
                            s_flip_pos = (n2d[0]*(-pa)+n2d[1]*(-pb)+n2d[2]*(-pc_n) < 0) ? 1 : 0;
                        }
                        s_cap_tris = (int*)cap_ptr; cap_ptr = NULL;  // ownership transferred to shared
                    }
                } // kern_ok after loop-phase alloc
            } // n_boundary > 0
        // Broadcast phase-13 state.  Local scratch lv_ptr..sb_ptr are all NULL
        // at this point (freed inline above); heap_free is NULL-safe.
        if (lane == 0) {
            s_n_cap     = n_cap;
            s_total_pos = n_pos;
            s_total_neg = n_neg;
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
            heap_free(scratch_heap, lv_ptr); s_be_a = NULL;
            heap_free(scratch_heap, loop_blk);
            // cap_ptr ownership transferred to s_cap_tris; freed via PC_FREE_ALL_SHARED_SCRATCH
        }
    } // end if (warp_id == 0) phases 10-12
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
        int n_cap_b = s_n_cap;
        int* cap_b  = s_cap_tris;
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
        // Cap tris share the same vertices on both sides
        for (int t = tid; t < n_cap_b; t += PC_BLOCK) {
            int vi0=cap_b[t*3+0], vi1=cap_b[t*3+1], vi2=cap_b[t*3+2];
            atomicMax(&pos_remap[vi0], 0); atomicMax(&pos_remap[vi1], 0); atomicMax(&pos_remap[vi2], 0);
            atomicMax(&neg_remap[vi0], 0); atomicMax(&neg_remap[vi1], 0); atomicMax(&neg_remap[vi2], 0);
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
        // Output tri count includes cap tris.
        if (tid == 0) {
            int pnv = s_pnv, nnv = s_nnv;
            int out_pos_nt = total_pos + n_cap_b;
            int out_neg_nt = total_neg + n_cap_b;
            unsigned int pv_b=(unsigned int)PC_ALIGN16(pnv       *3*(int)sizeof(float));
            unsigned int pt_b=(unsigned int)PC_ALIGN16(out_pos_nt*3*(int)sizeof(int));
            unsigned int nv_b=(unsigned int)PC_ALIGN16(nnv       *3*(int)sizeof(float));
            unsigned int nt_b=(unsigned int)PC_ALIGN16(out_neg_nt*3*(int)sizeof(int));
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
                s_result.pos.mesh.nv=pnv;      s_result.pos.mesh.nt=out_pos_nt;
                s_result.pos.mesh.refcount=rcp;
                pc_zero_part(&s_result.neg);
                s_result.neg.mesh.verts=s_nvp; s_result.neg.mesh.tris=s_ntp;
                s_result.neg.mesh.nv=nnv;      s_result.neg.mesh.nt=out_neg_nt;
                s_result.neg.mesh.refcount=rcn;
            } else {
                atomicOr(kernel_error, PC_KERR_POOL_OOM);
                if (pchunk) heap_free(heap, pchunk);
                s_pvp=NULL; s_ptp=NULL; s_nvp=NULL; s_ntp=NULL;
            }
        }
        __syncthreads();

        // Steps H-I: scatter verts, copy phase-6 tris, copy cap tris with winding (parallel)
        if (s_chunk_ok) {
            float* pvp = s_pvp; int* ptp = s_ptp;
            float* nvp = s_nvp; int* ntp = s_ntp;
            int flip = s_flip_pos;

            // H: scatter verts
            for (int i = tid; i < n_all; i += PC_BLOCK) {
                int ni = pos_remap[i];
                if (ni >= 0) { pvp[ni*3+0]=all_verts[i*3+0]; pvp[ni*3+1]=all_verts[i*3+1]; pvp[ni*3+2]=all_verts[i*3+2]; }
            }
            for (int i = tid; i < n_all; i += PC_BLOCK) {
                int ni = neg_remap[i];
                if (ni >= 0) { nvp[ni*3+0]=all_verts[i*3+0]; nvp[ni*3+1]=all_verts[i*3+1]; nvp[ni*3+2]=all_verts[i*3+2]; }
            }
            // I: copy phase-6 tris (already remapped in step F)
            for (int i = tid; i < total_pos * 3; i += PC_BLOCK) ptp[i] = pos_tris[i];
            for (int i = tid; i < total_neg * 3; i += PC_BLOCK) ntp[i] = neg_tris[i];
            // I2: copy cap tris with winding adjustment + remap
            for (int t = tid; t < n_cap_b; t += PC_BLOCK) {
                int v0 = cap_b[t*3], v1 = cap_b[t*3+1], v2 = cap_b[t*3+2];
                int po = (total_pos + t) * 3;
                ptp[po]   = pos_remap[v0];
                ptp[po+1] = flip ? pos_remap[v2] : pos_remap[v1];
                ptp[po+2] = flip ? pos_remap[v1] : pos_remap[v2];
                int no = (total_neg + t) * 3;
                ntp[no]   = neg_remap[v0];
                ntp[no+1] = flip ? neg_remap[v1] : neg_remap[v2];
                ntp[no+2] = flip ? neg_remap[v2] : neg_remap[v1];
            }
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
    if (tid == 0) DPRINTF("[pc] block=%d nv=%d nt=%d n_cross=%d n_boundary=%d n_loops=%d dt=%lld\n",
        blockIdx.x, n_verts, n_tris, n_cross, s_n_boundary, s_w_n_loops, clock64() - t_start);
    return s_result;
}
