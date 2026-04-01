// plane_cut.cu — GPU plane cut kernel with cap triangulation.
//
// One block (64 threads = 2 warps) handles one plane cut.
// Produces two closed meshes (positive/negative halves) with
// triangulated caps. Supports simple, ring, and multi-hole topologies.
//
// Phases:
//   1-2  Classify vertices, copy to scratch           (parallel, 64 threads)
//   3    Collect crossing edges                        (parallel, 64 threads)
//   4    Sort crossing edges                           (warp 0)
//   5    Dedup + create intersection vertices          (thread 0)
//   6    Split triangles                               (parallel, 64 threads)
//   7    Collect directed edges from pos_tris          (parallel, 64 threads)
//   8    Sort directed edges                           (warp 0)
//   9    Binary search for boundary edges              (parallel, 64 threads)
//  10-12 Chain loops, bridge holes, ear clip cap       (thread 0)
//  13    Compact verts per side, heap-alloc one chunk,
//        materialise output into PartPair              (thread 0)
//
#pragma once
#include "common.cuh"
#include "heap_arena.cuh"
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
        Edge2i s;
        s.a = 0x7fffffff; s.b = 0x7fffffff;
        return s;
    }
};

// ============================================================================
// Device helpers
// ============================================================================

// Binary search for (key_a, key_b) in sorted Edge2i array.
// Returns index of first match, or -1.
__device__ inline int pc_edge_bsearch(const Edge2i* arr, int n, int key_a, int key_b) {
    Edge2i key = {key_a, key_b};
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = Edge2iCmp::cmp(arr[mid], key);
        if (c < 0)       lo = mid + 1;
        else if (c > 0)  hi = mid - 1;
        else              return mid;
    }
    return -1;
}

// 2D cross product (b-a) x (c-a)
__device__ inline float pc_cross2d(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Point-in-triangle test (2D). Returns 1 if strictly or weakly inside.
__device__ inline int pc_pt_in_tri(float px, float py,
    float ax, float ay, float bx, float by, float cx, float cy)
{
    float d1 = pc_cross2d(ax,ay,bx,by,px,py);
    float d2 = pc_cross2d(bx,by,cx,cy,px,py);
    float d3 = pc_cross2d(cx,cy,ax,ay,px,py);
    return !((d1<-1e-10f||d2<-1e-10f||d3<-1e-10f) &&
             (d1> 1e-10f||d2> 1e-10f||d3> 1e-10f));
}

// Signed area of 2D polygon
__device__ inline float pc_signed_area(const float* verts, const int* poly, int n, int pu, int pv) {
    float area = 0.0f;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += verts[poly[i]*3+pu] * verts[poly[j]*3+pv]
              - verts[poly[j]*3+pu] * verts[poly[i]*3+pv];
    }
    return area * 0.5f;
}

// Reverse an int array in place
__device__ inline void pc_reverse(int* arr, int n) {
    for (int i = 0; i < n / 2; i++) {
        int t = arr[i]; arr[i] = arr[n-1-i]; arr[n-1-i] = t;
    }
}

// Edge intersection: compute point on edge va-vb where plane = 0
__device__ inline void pc_intersect(
    const float* verts, int va, int vb,
    float pa, float pb, float pc_n, float pd,
    float* out)
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

// Helper: zero a Part's fields (empty mesh, no hull, no costs).
__device__ inline void pc_zero_part(Part* p) {
    p->mesh.verts = NULL; p->mesh.tris = NULL; p->mesh.nv = 0; p->mesh.nt = 0;
    p->hull.verts = NULL; p->hull.tris = NULL; p->hull.nv = 0; p->hull.nt = 0;
    p->mesh_vol = 0.0f; p->hull_vol = 0.0f; p->hausdorff = 0.0f;
}

__device__ inline void plane_cut_block(
    // Input
    const Mesh* __restrict__ mesh,
    float pa, float pb, float pc_n, float pd,
    // Output
    PartPair* __restrict__ out,
    DeviceHeap* heap,
    // Scratch
    DevicePool scratch,
    int* __restrict__ kernel_error)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    const float* vertices  = mesh->verts;
    const int*   triangles = mesh->tris;
    int          n_verts   = mesh->nv;
    int          n_tris    = mesh->nt;

    // === Shared state ===
    // s_counters: [0]=n_cross, [1]=n_all_verts, [2]=n_pos, [3]=n_neg
    __shared__ int   s_counters[4];
    __shared__ char* s_base;
    __shared__ int   s_ok;

    // === Scratch allocation (thread 0) ===
    int max_new_verts       = n_tris * 2;
    int total_verts_cap     = n_verts + max_new_verts;
    int max_cross_edges     = n_tris * 2;
    int max_out_tris        = n_tris * 3;
    int sort_scratch_bytes  = max_cross_edges * (int)sizeof(Edge2i) + WS_MAX_STACK * 2 * (int)sizeof(int);
    int max_dir_edges       = max_out_tris * 3;
    int dir_sort_bytes      = max_dir_edges * (int)sizeof(Edge2i) + WS_MAX_STACK * 2 * (int)sizeof(int);

    int scratch_bytes =
        PC_ALIGN16(n_verts          * (int)sizeof(int))   +   // signs
        PC_ALIGN16(total_verts_cap  * 3 * (int)sizeof(float)) + // all_verts
        PC_ALIGN16(max_cross_edges  * (int)sizeof(Edge2i)) +  // crossing_edges
        PC_ALIGN16(sort_scratch_bytes)                      +  // sort scratch (cross)
        PC_ALIGN16(max_cross_edges  * (int)sizeof(int))    +  // isect_idx
        PC_ALIGN16(max_out_tris * 3 * (int)sizeof(int))    +  // pos_tris scratch
        PC_ALIGN16(max_out_tris * 3 * (int)sizeof(int))    +  // neg_tris scratch
        PC_ALIGN16(max_dir_edges    * (int)sizeof(Edge2i)) +  // dir_edges
        PC_ALIGN16(dir_sort_bytes)                          +  // sort scratch (dir)
        PC_ALIGN16(max_dir_edges    * (int)sizeof(int))    +  // boundary_flags
        PC_ALIGN16(max_dir_edges    * (int)sizeof(int))    +  // loop_verts (packed pairs)
        PC_ALIGN16(max_dir_edges    * (int)sizeof(int))    +  // loop_starts
        PC_ALIGN16(max_dir_edges    * (int)sizeof(int))    +  // loop_sizes
        PC_ALIGN16((max_dir_edges + 64) * (int)sizeof(int)) + // polygon
        PC_ALIGN16(max_dir_edges * 3 * (int)sizeof(int))   +  // cap_tris
        PC_ALIGN16(max_dir_edges * 2 * (int)sizeof(int));     // ear_prevnext

    if (tid == 0) {
        s_base = (char*)pool_alloc(&scratch, scratch_bytes);
        s_ok = (s_base != NULL) ? 1 : 0;
        s_counters[0] = 0;
        s_counters[1] = n_verts;  // will become n_all_verts after phase 5
        s_counters[2] = 0;
        s_counters[3] = 0;
    }
    __syncthreads();
    if (!s_ok) {
        if (tid == 0) atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);
        return;
    }

    // Carve scratch into sub-arrays
    char* base = s_base;
    int off = 0;

    int*    signs          = (int*)(base + off);    off += PC_ALIGN16(n_verts * (int)sizeof(int));
    float*  all_verts      = (float*)(base + off);  off += PC_ALIGN16(total_verts_cap * 3 * (int)sizeof(float));
    Edge2i* cross_edges    = (Edge2i*)(base + off); off += PC_ALIGN16(max_cross_edges * (int)sizeof(Edge2i));
    char*   sort_scratch_buf = base + off;           off += PC_ALIGN16(sort_scratch_bytes);
    int*    isect_idx      = (int*)(base + off);    off += PC_ALIGN16(max_cross_edges * (int)sizeof(int));
    int*    pos_tris       = (int*)(base + off);    off += PC_ALIGN16(max_out_tris * 3 * (int)sizeof(int));
    int*    neg_tris       = (int*)(base + off);    off += PC_ALIGN16(max_out_tris * 3 * (int)sizeof(int));
    Edge2i* dir_edges      = (Edge2i*)(base + off); off += PC_ALIGN16(max_dir_edges * (int)sizeof(Edge2i));
    char*   dir_sort_buf   = base + off;             off += PC_ALIGN16(dir_sort_bytes);
    int*    boundary_flags = (int*)(base + off);    off += PC_ALIGN16(max_dir_edges * (int)sizeof(int));
    int*    loop_verts     = (int*)(base + off);    off += PC_ALIGN16(max_dir_edges * (int)sizeof(int));
    int*    loop_starts    = (int*)(base + off);    off += PC_ALIGN16(max_dir_edges * (int)sizeof(int));
    int*    loop_sizes     = (int*)(base + off);    off += PC_ALIGN16(max_dir_edges * (int)sizeof(int));
    int*    polygon        = (int*)(base + off);    off += PC_ALIGN16((max_dir_edges + 64) * (int)sizeof(int));
    int*    cap_tris       = (int*)(base + off);    off += PC_ALIGN16(max_dir_edges * 3 * (int)sizeof(int));
    int*    ear_prevnext   = (int*)(base + off);    // last; no advance needed

    // === Phase 1: Classify vertices ===
    for (int v = tid; v < n_verts; v += PC_BLOCK) {
        float val = pa*vertices[v*3] + pb*vertices[v*3+1] + pc_n*vertices[v*3+2] + pd;
        signs[v] = (val > PC_EPS) ? 1 : ((val < -PC_EPS) ? -1 : 0);
    }

    // === Phase 2: Copy original vertices ===
    for (int i = tid; i < n_verts * 3; i += PC_BLOCK)
        all_verts[i] = vertices[i];
    __syncthreads();

    // === Phase 3: Collect crossing edges ===
    for (int t = tid; t < n_tris; t += PC_BLOCK) {
        int i0 = triangles[t*3], i1 = triangles[t*3+1], i2 = triangles[t*3+2];
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];
        int edges[3][2] = {{i0,i1},{i1,i2},{i2,i0}};
        int esigns[3][2] = {{s0,s1},{s1,s2},{s2,s0}};
        for (int e = 0; e < 3; e++) {
            int sa = esigns[e][0], sb = esigns[e][1];
            if ((sa > 0 && sb < 0) || (sa < 0 && sb > 0)) {
                int va = edges[e][0], vb = edges[e][1];
                int mn = (va < vb) ? va : vb;
                int mx = (va < vb) ? vb : va;
                int ci = atomicAdd(&s_counters[0], 1);
                if (ci < max_cross_edges) {
                    cross_edges[ci].a = mn;
                    cross_edges[ci].b = mx;
                }
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
        // No crossing — entire mesh is on one side.
        // Thread 0 allocates from heap and fills PartPair.
        if (tid == 0) {
            int any_pos = 0, any_neg = 0;
            for (int v = 0; v < n_verts; v++) {
                if (signs[v] > 0) any_pos = 1;
                if (signs[v] < 0) any_neg = 1;
            }
            int is_neg = (any_neg && !any_pos);

            unsigned int vb = (unsigned int)PC_ALIGN16(n_verts * 3 * (int)sizeof(float));
            unsigned int tb = (unsigned int)PC_ALIGN16(n_tris  * 3 * (int)sizeof(int));
            unsigned int chunk_sz = vb + tb;
            if (!chunk_sz) chunk_sz = 1;

            void* chunk_ptr = NULL;
            int herr = heap_alloc(heap, chunk_sz, &chunk_ptr);
            if (herr) {
                atomicOr(kernel_error, PC_KERR_POOL_OOM);
            } else {
                char*  cb = (char*)chunk_ptr;
                float* pv = (float*)cb;
                int*   pt = (int*)(cb + vb);
                for (int i = 0; i < n_verts * 3; i++) pv[i] = vertices[i];
                for (int i = 0; i < n_tris  * 3; i++) pt[i] = triangles[i];

                Part* side  = is_neg ? &out->neg : &out->pos;
                Part* empty = is_neg ? &out->pos : &out->neg;
                pc_zero_part(side);
                side->mesh.verts = pv; side->mesh.tris = pt;
                side->mesh.nv = n_verts; side->mesh.nt = n_tris;
                pc_zero_part(empty);
            }
        }
        __syncthreads();
        return;
    }

    // === Phase 4: Sort crossing edges (warp 0) ===
    if (warp_id == 0) {
        int serr = warp_sort_t<Edge2i, Edge2iCmp>(cross_edges, sort_scratch_buf, n_cross, lane);
        if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
    }
    __syncthreads();

    // === Phase 5: Dedup + create intersection vertices (thread 0) ===
    __shared__ int s_n_unique;
    if (tid == 0) {
        int n_unique = 0;
        int cur_isect = n_verts;
        for (int i = 0; i < n_cross; i++) {
            if (i == 0 || cross_edges[i].a != cross_edges[i-1].a || cross_edges[i].b != cross_edges[i-1].b) {
                int va = cross_edges[i].a, vb = cross_edges[i].b;
                int idx = cur_isect++;
                pc_intersect(all_verts, va, vb, pa, pb, pc_n, pd, &all_verts[idx * 3]);
                isect_idx[i] = idx;
                n_unique++;
            } else {
                isect_idx[i] = isect_idx[i - 1];
            }
        }
        s_counters[1] = cur_isect;  // n_all_verts
        s_n_unique = n_unique;
    }
    __syncthreads();

    int n_all = s_counters[1];

    // === Phase 6: Split triangles ===
    for (int t = tid; t < n_tris; t += PC_BLOCK) {
        int i0 = triangles[t*3], i1 = triangles[t*3+1], i2 = triangles[t*3+2];
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];
        int hp = (s0>0)||(s1>0)||(s2>0);
        int hn = (s0<0)||(s1<0)||(s2<0);

        if (!hp || !hn) {
            if (hn) {
                int ni = atomicAdd(&s_counters[3], 1);
                neg_tris[ni*3]=i0; neg_tris[ni*3+1]=i1; neg_tris[ni*3+2]=i2;
            } else if (hp) {
                int pi = atomicAdd(&s_counters[2], 1);
                pos_tris[pi*3]=i0; pos_tris[pi*3+1]=i1; pos_tris[pi*3+2]=i2;
            } else {
                // All on plane: assign by normal direction
                float e1x=all_verts[i1*3]-all_verts[i0*3], e1y=all_verts[i1*3+1]-all_verts[i0*3+1], e1z=all_verts[i1*3+2]-all_verts[i0*3+2];
                float e2x=all_verts[i2*3]-all_verts[i0*3], e2y=all_verts[i2*3+1]-all_verts[i0*3+1], e2z=all_verts[i2*3+2]-all_verts[i0*3+2];
                float dot = pa*(e1y*e2z-e1z*e2y)+pb*(e1z*e2x-e1x*e2z)+pc_n*(e1x*e2y-e1y*e2x);
                if (dot > 0) { int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=i0;pos_tris[pi*3+1]=i1;pos_tris[pi*3+2]=i2; }
                else         { int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=i0;neg_tris[ni*3+1]=i1;neg_tris[ni*3+2]=i2; }
            }
            continue;
        }

        // Triangle spans both sides
        int vi[3] = {i0,i1,i2};
        int si[3] = {s0,s1,s2};

        #define FIND_ISECT(va, vb) ({ \
            int _mn = ((va)<(vb))?(va):(vb), _mx = ((va)<(vb))?(vb):(va); \
            int _fi = pc_edge_bsearch(cross_edges, n_cross, _mn, _mx); \
            (_fi >= 0) ? isect_idx[_fi] : -1; \
        })

        int on = -1;
        for (int k = 0; k < 3; k++) if (si[k] == 0) { on = k; break; }

        if (on >= 0) {
            int ov = vi[on], a_i = vi[(on+1)%3], b_i = vi[(on+2)%3];
            int sa = si[(on+1)%3];
            int nvi = FIND_ISECT(a_i, b_i);
            if (nvi < 0) continue;
            if (sa > 0) {
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov; pos_tris[pi*3+1]=a_i; pos_tris[pi*3+2]=nvi;
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov; neg_tris[ni*3+1]=nvi; neg_tris[ni*3+2]=b_i;
            } else {
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=ov; neg_tris[ni*3+1]=a_i; neg_tris[ni*3+2]=nvi;
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=ov; pos_tris[pi*3+1]=nvi; pos_tris[pi*3+2]=b_i;
            }
        } else {
            int lone = -1;
            for (int k = 0; k < 3; k++)
                if (si[k] != si[(k+1)%3] && si[k] != si[(k+2)%3]) { lone = k; break; }
            if (lone < 0) continue;

            int lv = vi[lone], ov1 = vi[(lone+1)%3], ov2 = vi[(lone+2)%3];
            int ls = si[lone];
            int nv1 = FIND_ISECT(lv, ov1);
            int nv2 = FIND_ISECT(lv, ov2);
            if (nv1 < 0 || nv2 < 0) continue;

            if (ls > 0) {
                int pi=atomicAdd(&s_counters[2],1); pos_tris[pi*3]=lv; pos_tris[pi*3+1]=nv1; pos_tris[pi*3+2]=nv2;
                int ni=atomicAdd(&s_counters[3],2);
                neg_tris[ni*3]=ov1; neg_tris[ni*3+1]=nv2; neg_tris[ni*3+2]=nv1;
                neg_tris[(ni+1)*3]=ov1; neg_tris[(ni+1)*3+1]=ov2; neg_tris[(ni+1)*3+2]=nv2;
            } else {
                int ni=atomicAdd(&s_counters[3],1); neg_tris[ni*3]=lv; neg_tris[ni*3+1]=nv1; neg_tris[ni*3+2]=nv2;
                int pi=atomicAdd(&s_counters[2],2);
                pos_tris[pi*3]=ov1; pos_tris[pi*3+1]=nv2; pos_tris[pi*3+2]=nv1;
                pos_tris[(pi+1)*3]=ov1; pos_tris[(pi+1)*3+1]=ov2; pos_tris[(pi+1)*3+2]=nv2;
            }
        }
        #undef FIND_ISECT
    }
    __syncthreads();

    __shared__ int s_n_pos, s_n_neg;
    if (tid == 0) {
        s_n_pos = s_counters[2];
        s_n_neg = s_counters[3];
    }
    __syncthreads();
    int n_pos = s_n_pos, n_neg = s_n_neg;

    // === Phase 7: Collect directed edges from pos_tris ===
    int n_de = n_pos * 3;
    for (int t = tid; t < n_pos; t += PC_BLOCK) {
        int a = pos_tris[t*3], b = pos_tris[t*3+1], c = pos_tris[t*3+2];
        int bi = t * 3;
        dir_edges[bi  ] = (Edge2i){a, b};
        dir_edges[bi+1] = (Edge2i){b, c};
        dir_edges[bi+2] = (Edge2i){c, a};
    }
    __syncthreads();

    // === Phase 8: Sort directed edges (warp 0) ===
    if (warp_id == 0) {
        int serr = warp_sort_t<Edge2i, Edge2iCmp>(dir_edges, dir_sort_buf, n_de, lane);
        if (serr && lane == 0) atomicOr(kernel_error, PC_KERR_SORT_ERR);
    }
    __syncthreads();

    // === Phase 9: Binary search for boundary edges ===
    for (int i = tid; i < n_de; i += PC_BLOCK) {
        int a = dir_edges[i].a, b = dir_edges[i].b;
        int rev = pc_edge_bsearch(dir_edges, n_de, b, a);
        boundary_flags[i] = (rev < 0) ? 1 : 0;
    }
    __syncthreads();

    // Thread 0: compact boundary edges into loop_verts (packed a,b pairs)
    __shared__ int s_n_boundary;
    if (tid == 0) {
        int nb = 0;
        for (int i = 0; i < n_de; i++) {
            if (boundary_flags[i]) {
                loop_verts[nb*2]   = dir_edges[i].a;
                loop_verts[nb*2+1] = dir_edges[i].b;
                nb++;
            }
        }
        s_n_boundary = nb;
    }
    __syncthreads();

    int n_boundary = s_n_boundary;

    // === Phases 10-12 + compact + heap alloc (thread 0) ===
    if (tid == 0) {
        int n_cap = 0;

        if (n_boundary > 0) {
            // Phase 10: Reconstruct boundary loops
            int* be_a    = loop_verts;      // packed (a,b) pairs
            int* be_used = boundary_flags;  // reuse as used-flags
            for (int i = 0; i < n_boundary; i++) be_used[i] = 0;

            int* lv = isect_idx;  // reuse for loop vertex storage
            int n_loops = 0, lvi = 0;

            for (int start = 0; start < n_boundary; start++) {
                if (be_used[start]) continue;
                loop_starts[n_loops] = lvi;
                int first = be_a[start*2], cur = be_a[start*2+1];
                be_used[start] = 1;
                lv[lvi++] = first;
                int safety = n_boundary + 2;
                while (cur != first && safety-- > 0) {
                    lv[lvi++] = cur;
                    int found = 0;
                    for (int i = 0; i < n_boundary; i++) {
                        if (!be_used[i] && be_a[i*2] == cur) {
                            cur = be_a[i*2+1]; be_used[i] = 1; found = 1; break;
                        }
                    }
                    if (!found) break;
                }
                loop_sizes[n_loops] = lvi - loop_starts[n_loops];
                n_loops++;
            }

            // 2D projection axes (drop largest normal component)
            int pu, pv_ax;
            {
                float anx = fabsf(pa), any = fabsf(pb), anz = fabsf(pc_n);
                if (anx >= any && anx >= anz) { pu = 1; pv_ax = 2; }
                else if (any >= anz)          { pu = 0; pv_ax = 2; }
                else                          { pu = 0; pv_ax = 1; }
            }

            // Phase 11: Build polygon (bridge holes if needed)
            int poly_n = 0;

            if (n_loops == 1) {
                for (int i = 0; i < loop_sizes[0]; i++)
                    polygon[i] = lv[loop_starts[0] + i];
                poly_n = loop_sizes[0];
                if (pc_signed_area(all_verts, polygon, poly_n, pu, pv_ax) < 0)
                    pc_reverse(polygon, poly_n);
            } else if (n_loops > 1) {
                // Find outer loop (largest |area|)
                int outer_i = 0;
                float max_abs = 0;
                for (int i = 0; i < n_loops; i++) {
                    float a = pc_signed_area(all_verts, lv + loop_starts[i], loop_sizes[i], pu, pv_ax);
                    if (fabsf(a) > max_abs) { max_abs = fabsf(a); outer_i = i; }
                }
                // Ensure outer CCW, inner CW
                if (pc_signed_area(all_verts, lv + loop_starts[outer_i], loop_sizes[outer_i], pu, pv_ax) < 0)
                    pc_reverse(lv + loop_starts[outer_i], loop_sizes[outer_i]);
                for (int i = 0; i < n_loops; i++) {
                    if (i == outer_i) continue;
                    if (pc_signed_area(all_verts, lv + loop_starts[i], loop_sizes[i], pu, pv_ax) > 0)
                        pc_reverse(lv + loop_starts[i], loop_sizes[i]);
                }

                // Sort inner loops by rightmost vertex (descending u), use boundary_flags as scratch
                int n_inner = 0;
                int*   inner_idx   = boundary_flags;
                float* inner_max_u = (float*)(boundary_flags + 256);
                for (int i = 0; i < n_loops && n_inner < 256; i++) {
                    if (i == outer_i || loop_sizes[i] < 3) continue;
                    float mu = -1e30f;
                    for (int j = 0; j < loop_sizes[i]; j++) {
                        float u = all_verts[lv[loop_starts[i]+j]*3+pu];
                        if (u > mu) mu = u;
                    }
                    inner_idx[n_inner]   = i;
                    inner_max_u[n_inner] = mu;
                    n_inner++;
                }
                // Selection sort descending
                for (int i = 0; i < n_inner - 1; i++) {
                    int best = i;
                    for (int j = i+1; j < n_inner; j++)
                        if (inner_max_u[j] > inner_max_u[best]) best = j;
                    if (best != i) {
                        int ti = inner_idx[i]; inner_idx[i] = inner_idx[best]; inner_idx[best] = ti;
                        float tf = inner_max_u[i]; inner_max_u[i] = inner_max_u[best]; inner_max_u[best] = tf;
                    }
                }

                // Start with outer loop
                for (int i = 0; i < loop_sizes[outer_i]; i++)
                    polygon[i] = lv[loop_starts[outer_i] + i];
                poly_n = loop_sizes[outer_i];

                // Bridge each inner loop
                for (int ii = 0; ii < n_inner; ii++) {
                    int hi  = inner_idx[ii];
                    int* hole = lv + loop_starts[hi];
                    int hs  = loop_sizes[hi];

                    int m_idx = 0; float max_u = -1e30f;
                    for (int j = 0; j < hs; j++) {
                        float u = all_verts[hole[j]*3+pu];
                        if (u > max_u) { max_u = u; m_idx = j; }
                    }
                    float mu = all_verts[hole[m_idx]*3+pu];
                    float mv = all_verts[hole[m_idx]*3+pv_ax];

                    float best_t = 1e30f; int best_edge = -1;
                    for (int j = 0; j < poly_n; j++) {
                        int jn = (j+1) % poly_n;
                        float au = all_verts[polygon[j]*3+pu],  av2 = all_verts[polygon[j]*3+pv_ax];
                        float bu = all_verts[polygon[jn]*3+pu], bv2 = all_verts[polygon[jn]*3+pv_ax];
                        float dv = bv2 - av2;
                        if (fabsf(dv) < 1e-10f) continue;
                        float s = (mv - av2) / dv;
                        if (s < -1e-10f || s > 1.0f + 1e-10f) continue;
                        float tv = au + s * (bu - au) - mu;
                        if (tv > 1e-10f && tv < best_t) { best_t = tv; best_edge = j; }
                    }
                    if (best_edge < 0) continue;

                    int jn = (best_edge+1) % poly_n;
                    float eu = all_verts[polygon[best_edge]*3+pu];
                    float fu = all_verts[polygon[jn]*3+pu];
                    int p_pos = (eu >= fu) ? best_edge : jn;

                    // Insert bridge into cap_tris temporarily
                    int k = 0;
                    for (int j = 0; j <= p_pos; j++) cap_tris[k++] = polygon[j];
                    for (int j = 0; j < hs; j++) cap_tris[k++] = hole[(m_idx+j) % hs];
                    cap_tris[k++] = hole[m_idx];
                    cap_tris[k++] = polygon[p_pos];
                    for (int j = p_pos+1; j < poly_n; j++) cap_tris[k++] = polygon[j];
                    for (int j = 0; j < k; j++) polygon[j] = cap_tris[j];
                    poly_n = k;
                }
            }

            // Phase 12: Ear clipping
            if (poly_n >= 3) {
                int* prev_a = ear_prevnext;
                int* next_a = ear_prevnext + poly_n;
                for (int i = 0; i < poly_n; i++) {
                    prev_a[i] = (i + poly_n - 1) % poly_n;
                    next_a[i] = (i + 1) % poly_n;
                }
                int remaining = poly_n, cur = 0;
                int max_iter = poly_n * poly_n;
                int iter = 0;
                while (remaining > 3 && iter < max_iter) {
                    iter++;
                    int p = prev_a[cur], n = next_a[cur];
                    int vp = polygon[p], vc = polygon[cur], vn = polygon[n];
                    float up = all_verts[vp*3+pu],  vp_ = all_verts[vp*3+pv_ax];
                    float uc = all_verts[vc*3+pu],  vc_ = all_verts[vc*3+pv_ax];
                    float un = all_verts[vn*3+pu],  vn_ = all_verts[vn*3+pv_ax];
                    float cross = (uc-up)*(vn_-vp_) - (vc_-vp_)*(un-up);
                    if (cross <= 1e-10f) { cur = next_a[cur]; continue; }
                    int ear = 1;
                    int chk = next_a[n];
                    while (chk != p) {
                        int vi_ = polygon[chk];
                        float cu = all_verts[vi_*3+pu], cv = all_verts[vi_*3+pv_ax];
                        float d1 = fabsf(cu-up)+fabsf(cv-vp_);
                        float d2 = fabsf(cu-uc)+fabsf(cv-vc_);
                        float d3 = fabsf(cu-un)+fabsf(cv-vn_);
                        if (d1 > 1e-6f && d2 > 1e-6f && d3 > 1e-6f &&
                            pc_pt_in_tri(cu, cv, up, vp_, uc, vc_, un, vn_)) {
                            ear = 0; break;
                        }
                        chk = next_a[chk];
                    }
                    if (ear) {
                        cap_tris[n_cap*3]   = vp;
                        cap_tris[n_cap*3+1] = vc;
                        cap_tris[n_cap*3+2] = vn;
                        n_cap++;
                        next_a[p] = n; prev_a[n] = p;
                        remaining--; cur = n; iter = 0;
                    } else {
                        cur = next_a[cur];
                    }
                }
                if (remaining == 3) {
                    int p = prev_a[cur], n = next_a[cur];
                    cap_tris[n_cap*3]   = polygon[p];
                    cap_tris[n_cap*3+1] = polygon[cur];
                    cap_tris[n_cap*3+2] = polygon[n];
                    n_cap++;
                }
            }

            // Determine cap winding and append to pos_tris / neg_tris
            float n2d[3] = {0, 0, 0};
            if (pu == 0 && pv_ax == 1)      n2d[2] = 1.0f;
            else if (pu == 1 && pv_ax == 2) n2d[0] = 1.0f;
            else                            n2d[1] = -1.0f;
            float dot = n2d[0]*(-pa) + n2d[1]*(-pb) + n2d[2]*(-pc_n);
            int flip_pos = (dot < 0) ? 1 : 0;

            for (int i = 0; i < n_cap; i++) {
                int pi = n_pos + i;
                pos_tris[pi*3] = cap_tris[i*3];
                if (flip_pos) {
                    pos_tris[pi*3+1] = cap_tris[i*3+2];
                    pos_tris[pi*3+2] = cap_tris[i*3+1];
                } else {
                    pos_tris[pi*3+1] = cap_tris[i*3+1];
                    pos_tris[pi*3+2] = cap_tris[i*3+2];
                }
                int ni = n_neg + i;
                neg_tris[ni*3] = cap_tris[i*3];
                if (!flip_pos) {
                    neg_tris[ni*3+1] = cap_tris[i*3+2];
                    neg_tris[ni*3+2] = cap_tris[i*3+1];
                } else {
                    neg_tris[ni*3+1] = cap_tris[i*3+1];
                    neg_tris[ni*3+2] = cap_tris[i*3+2];
                }
            }
        } // end if (n_boundary > 0)

        int total_pos = n_pos + n_cap;
        int total_neg = n_neg + n_cap;

        // === Phase 13: Compact vertices per side, single heap alloc, fill PartPair ===

        // Remap arrays: allocated from the remaining scratch pool capacity.
        // n_all ints each; pool has ample headroom beyond the upfront scratch_bytes.
        int* pos_remap = (int*)pool_alloc(&scratch, (unsigned int)(n_all * (int)sizeof(int)));
        int* neg_remap = (int*)pool_alloc(&scratch, (unsigned int)(n_all * (int)sizeof(int)));

        if (!pos_remap || !neg_remap) {
            atomicOr(kernel_error, PC_KERR_SCRATCH_OOM);
        } else {
            // --- Compact pos side ---
            for (int i = 0; i < n_all; i++) pos_remap[i] = -1;
            for (int t = 0; t < total_pos; t++) {
                pos_remap[pos_tris[t*3+0]] = 0;
                pos_remap[pos_tris[t*3+1]] = 0;
                pos_remap[pos_tris[t*3+2]] = 0;
            }
            int pnv = 0;
            for (int i = 0; i < n_all; i++) if (pos_remap[i] == 0) pos_remap[i] = pnv++;
            for (int t = 0; t < total_pos; t++) {
                pos_tris[t*3+0] = pos_remap[pos_tris[t*3+0]];
                pos_tris[t*3+1] = pos_remap[pos_tris[t*3+1]];
                pos_tris[t*3+2] = pos_remap[pos_tris[t*3+2]];
            }

            // --- Compact neg side ---
            for (int i = 0; i < n_all; i++) neg_remap[i] = -1;
            for (int t = 0; t < total_neg; t++) {
                neg_remap[neg_tris[t*3+0]] = 0;
                neg_remap[neg_tris[t*3+1]] = 0;
                neg_remap[neg_tris[t*3+2]] = 0;
            }
            int nnv = 0;
            for (int i = 0; i < n_all; i++) if (neg_remap[i] == 0) neg_remap[i] = nnv++;
            for (int t = 0; t < total_neg; t++) {
                neg_tris[t*3+0] = neg_remap[neg_tris[t*3+0]];
                neg_tris[t*3+1] = neg_remap[neg_tris[t*3+1]];
                neg_tris[t*3+2] = neg_remap[neg_tris[t*3+2]];
            }

            // --- Single combined heap allocation ---
            // Layout: [pos_verts | pos_tris | neg_verts | neg_tris], each 16-byte aligned.
            unsigned int pv_bytes = (unsigned int)PC_ALIGN16(pnv      * 3 * (int)sizeof(float));
            unsigned int pt_bytes = (unsigned int)PC_ALIGN16(total_pos * 3 * (int)sizeof(int));
            unsigned int nv_bytes = (unsigned int)PC_ALIGN16(nnv      * 3 * (int)sizeof(float));
            unsigned int nt_bytes = (unsigned int)PC_ALIGN16(total_neg * 3 * (int)sizeof(int));
            unsigned int chunk_sz = pv_bytes + pt_bytes + nv_bytes + nt_bytes;
            if (!chunk_sz) chunk_sz = 1;  // heap_alloc requires > 0

            void* chunk_ptr = NULL;
            int herr = heap_alloc(heap, chunk_sz, &chunk_ptr);
            if (herr) {
                atomicOr(kernel_error, PC_KERR_POOL_OOM);
            } else {
                char*  cb  = (char*)chunk_ptr;
                float* pvp = (float*)(cb);
                int*   ptp = (int*)  (cb + pv_bytes);
                float* nvp = (float*)(cb + pv_bytes + pt_bytes);
                int*   ntp = (int*)  (cb + pv_bytes + pt_bytes + nv_bytes);

                // Materialise pos verts (scatter by new index)
                for (int i = 0; i < n_all; i++) {
                    if (pos_remap[i] >= 0) {
                        int ni = pos_remap[i];
                        pvp[ni*3+0] = all_verts[i*3+0];
                        pvp[ni*3+1] = all_verts[i*3+1];
                        pvp[ni*3+2] = all_verts[i*3+2];
                    }
                }
                // Copy remapped pos tris
                for (int i = 0; i < total_pos * 3; i++) ptp[i] = pos_tris[i];

                // Materialise neg verts
                for (int i = 0; i < n_all; i++) {
                    if (neg_remap[i] >= 0) {
                        int ni = neg_remap[i];
                        nvp[ni*3+0] = all_verts[i*3+0];
                        nvp[ni*3+1] = all_verts[i*3+1];
                        nvp[ni*3+2] = all_verts[i*3+2];
                    }
                }
                // Copy remapped neg tris
                for (int i = 0; i < total_neg * 3; i++) ntp[i] = neg_tris[i];

                // Fill PartPair
                pc_zero_part(&out->pos);
                out->pos.mesh.verts = pvp; out->pos.mesh.tris = ptp;
                out->pos.mesh.nv    = pnv; out->pos.mesh.nt   = total_pos;

                pc_zero_part(&out->neg);
                out->neg.mesh.verts = nvp; out->neg.mesh.tris = ntp;
                out->neg.mesh.nv    = nnv; out->neg.mesh.nt   = total_neg;
            }
        }
    } // end if (tid == 0)
    __syncthreads();
}
