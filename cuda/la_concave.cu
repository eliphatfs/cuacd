// la_concave.cu — Concave edge detection and plane generation kernel.
//
// la_find_concave_edges: one block per cutting part (32 threads = 1 warp).
// Detects concave edges in the part's mesh, samples up to n_concave_edges
// of them, and generates up to 4 plane coefficients per sampled edge.

#include <cstdio>
#include "la_common.cuh"
#include "warp_sort.cuh"
#include "warp_common.cuh"

// ============================================================================
// Edge with face index for sort+scan deduplication
// ============================================================================
struct EdgeFace {
    int lo;        // min(v0, v1)
    int hi;        // max(v0, v1)
    int face_idx;  // which triangle this directed edge came from
    int _pad;
};

struct EdgeFaceCmp {
    static __device__ inline int cmp(EdgeFace a, EdgeFace b) {
        if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
        if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
        if (a.face_idx != b.face_idx) return (a.face_idx < b.face_idx) ? -1 : 1;
        return 0;
    }
    static __device__ inline EdgeFace sentinel() {
        EdgeFace s;
        s.lo = 0x7fffffff; s.hi = 0x7fffffff;
        s.face_idx = 0x7fffffff; s._pad = 0;
        return s;
    }
};

// ============================================================================
// Simple LCG RNG for reservoir sampling
// ============================================================================
static __device__ inline unsigned int lcg_next(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

// ============================================================================
// la_find_concave_edges: <<<n_cutting, 32>>>
// ============================================================================
extern "C" __global__ void la_find_concave_edges(
    LaDecompState*     decomp,
    int*               cutting_indices,
    int                n_cutting,
    int                n_concave_edges,
    float              concave_eps,
    float              concave_threshold,
    ConcaveEdgePlane*  edge_planes,    // output: [n_cutting * 4 * n_concave_edges]
    int*               n_edge_cuts,    // output per block: actual planes written
    DevicePool*        pool,
    int*               err)
{
    if (*err) return;

    int i = blockIdx.x;
    if (i >= n_cutting) return;

    int lane = threadIdx.x;
    if (lane >= 32) return;

    int part_idx = cutting_indices[i];
    Part* part   = &decomp->parts[part_idx];
    Mesh* mesh   = &part->mesh;

    int nv = mesh->nv;
    int nt = mesh->nt;
    if (nv <= 0 || nt <= 0 || nt < 2) {
        if (lane == 0) n_edge_cuts[i] = 0;
        return;
    }

    int n_dir_edges = 3 * nt;  // total directed edges

    // Allocate scratch from scratch heap for sort + reservoir.
    // Layout: [EdgeFace array][sort tmp][sort stacks][reservoir]
    size_t ef_bytes = (size_t)n_dir_edges * sizeof(EdgeFace);
    size_t sort_tmp = ef_bytes;  // warp_sort needs a second copy for partitioning
    size_t sort_stacks = (size_t)WS_MAX_STACK * 2 * sizeof(int);
    size_t reservoir_bytes = (size_t)n_concave_edges * 4 * sizeof(int);
    size_t total_scratch = ef_bytes + sort_tmp + sort_stacks + reservoir_bytes;

    // Lane 0 allocates, then broadcasts pointer
    __shared__ void* s_scratch_base;
    __shared__ int   s_alloc_ok;
    if (lane == 0) {
        s_alloc_ok = (heap_alloc(&pool->scratch, (unsigned int)total_scratch, &s_scratch_base) == HEAP_OK) ? 1 : 0;
        if (!s_alloc_ok) atomicOr(err, KERR_LA_SORT_OOM);
    }
    __syncwarp();
    if (!s_alloc_ok) {
        if (lane == 0) n_edge_cuts[i] = 0;
        return;
    }

    char* scratch = (char*)s_scratch_base;
    EdgeFace* ef_arr    = (EdgeFace*)(scratch);
    char*     sort_scratch = scratch + ef_bytes;
    int*      reservoir = (int*)(scratch + ef_bytes + sort_tmp + sort_stacks);

    // Build directed-edge array: each lane processes a strided subset
    for (int e = lane; e < n_dir_edges; e += 32) {
        int fi = e / 3;       // face index
        int ei = e % 3;       // edge index within face (0,1,2)
        int v0 = mesh->tris[3 * fi + ei];
        int v1 = mesh->tris[3 * fi + (ei + 1) % 3];
        ef_arr[e].lo = min(v0, v1);
        ef_arr[e].hi = max(v0, v1);
        ef_arr[e].face_idx = fi;
        ef_arr[e]._pad = 0;
    }
    __syncwarp();

    // Sort by (lo, hi, face_idx)
    int sort_rc = warp_sort_t<EdgeFace, EdgeFaceCmp>(ef_arr, sort_scratch, n_dir_edges, lane);
    if (sort_rc != 0) {
        if (lane == 0) {
            atomicOr(err, KERR_LA_SORT_STACK);
            heap_free(&pool->scratch, s_scratch_base);
            n_edge_cuts[i] = 0;
        }
        return;
    }
    __syncwarp();

    // Initialize reservoir to -1
    for (int j = lane; j < n_concave_edges * 4; j += 32)
        reservoir[j] = -1;
    __syncwarp();

    __shared__ int s_n_seen;   // total concave edges seen
    __shared__ int s_n_res;    // items currently in reservoir
    if (lane == 0) { s_n_seen = 0; s_n_res = 0; }
    __syncwarp();

    // RNG state per block
    __shared__ unsigned int s_rng;
    if (lane == 0) s_rng = 1u;
    __syncwarp();

    // Scan sorted edges for shared edges — warp-parallel detection, lane 0 reservoir sampling.
    // Each batch of 32 lanes evaluates 32 consecutive edges in parallel up to the
    // dihedral threshold test. Results are staged in shared memory and lane 0
    // consumes them in order to preserve reservoir-sampling determinism.
    __shared__ int s_batch_valid[32];
    __shared__ int s_batch_v0[32];
    __shared__ int s_batch_v1[32];
    __shared__ int s_batch_f0[32];
    __shared__ int s_batch_f1[32];

    int total_edges = n_dir_edges - 1;
    for (int batch = 0; batch < total_edges; batch += 32) {
        int e = batch + lane;
        int valid = 0;
        int v0 = -1, v1 = -1, f0 = -1, f1 = -1;

        if (e < total_edges) {
            EdgeFace ef0 = ef_arr[e];
            EdgeFace ef1 = ef_arr[e + 1];

            // First occurrence of this (lo, hi) in sorted array — prevents
            // double-counting on non-manifold edges (3+ faces sharing an edge).
            bool is_first = (e == 0) ||
                            (ef_arr[e - 1].lo != ef0.lo) ||
                            (ef_arr[e - 1].hi != ef0.hi);
            bool is_shared = (ef0.lo == ef1.lo) && (ef0.hi == ef1.hi) &&
                             (ef0.face_idx != ef1.face_idx);

            if (is_first && is_shared) {
                v0 = ef0.lo;
                v1 = ef0.hi;
                f0 = ef0.face_idx;
                f1 = ef1.face_idx;

                float3 n0, n1;
                bool ok = true;

                {
                    int t0a = mesh->tris[3*f0+0], t0b = mesh->tris[3*f0+1], t0c = mesh->tris[3*f0+2];
                    float3 p0a = {mesh->verts[3*t0a+0], mesh->verts[3*t0a+1], mesh->verts[3*t0a+2]};
                    float3 p0b = {mesh->verts[3*t0b+0], mesh->verts[3*t0b+1], mesh->verts[3*t0b+2]};
                    float3 p0c = {mesh->verts[3*t0c+0], mesh->verts[3*t0c+1], mesh->verts[3*t0c+2]};
                    float3 e1v = {p0b.x-p0a.x, p0b.y-p0a.y, p0b.z-p0a.z};
                    float3 e2v = {p0c.x-p0a.x, p0c.y-p0a.y, p0c.z-p0a.z};
                    n0.x = e1v.y*e2v.z - e1v.z*e2v.y;
                    n0.y = e1v.z*e2v.x - e1v.x*e2v.z;
                    n0.z = e1v.x*e2v.y - e1v.y*e2v.x;
                    float len0 = sqrtf(n0.x*n0.x + n0.y*n0.y + n0.z*n0.z);
                    if (len0 < 1e-12f) ok = false;
                    else { n0.x /= len0; n0.y /= len0; n0.z /= len0; }
                }
                if (ok) {
                    int t1a = mesh->tris[3*f1+0], t1b = mesh->tris[3*f1+1], t1c = mesh->tris[3*f1+2];
                    float3 p1a = {mesh->verts[3*t1a+0], mesh->verts[3*t1a+1], mesh->verts[3*t1a+2]};
                    float3 p1b = {mesh->verts[3*t1b+0], mesh->verts[3*t1b+1], mesh->verts[3*t1b+2]};
                    float3 p1c = {mesh->verts[3*t1c+0], mesh->verts[3*t1c+1], mesh->verts[3*t1c+2]};
                    float3 e1v = {p1b.x-p1a.x, p1b.y-p1a.y, p1b.z-p1a.z};
                    float3 e2v = {p1c.x-p1a.x, p1c.y-p1a.y, p1c.z-p1a.z};
                    n1.x = e1v.y*e2v.z - e1v.z*e2v.y;
                    n1.y = e1v.z*e2v.x - e1v.x*e2v.z;
                    n1.z = e1v.x*e2v.y - e1v.y*e2v.x;
                    float len1 = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z);
                    if (len1 < 1e-12f) ok = false;
                    else { n1.x /= len1; n1.y /= len1; n1.z /= len1; }
                }

                if (ok) {
                    float cos_a = n0.x*n1.x + n0.y*n1.y + n0.z*n1.z;
                    cos_a = fmaxf(-1.0f, fminf(1.0f, cos_a));
                    float alpha = acosf(cos_a);

                    int v_other = -1;
                    for (int vi = 0; vi < 3; vi++) {
                        int vv = mesh->tris[3*f0+vi];
                        if (vv != v0 && vv != v1) { v_other = vv; break; }
                    }

                    if (v_other >= 0) {
                        float3 edge_v = {mesh->verts[3*v0+0], mesh->verts[3*v0+1], mesh->verts[3*v0+2]};
                        float3 other_v = {mesh->verts[3*v_other+0], mesh->verts[3*v_other+1], mesh->verts[3*v_other+2]};
                        float3 diff = {other_v.x - edge_v.x, other_v.y - edge_v.y, other_v.z - edge_v.z};
                        float d = diff.x*n1.x + diff.y*n1.y + diff.z*n1.z;

                        const float pi = 3.14159265358979f;
                        float dihedral = (d > 0.0f) ? (pi + alpha) : (pi - alpha);

                        if (dihedral > concave_threshold) valid = 1;
                    }
                }
            }
        }

        s_batch_valid[lane] = valid;
        s_batch_v0[lane]    = v0;
        s_batch_v1[lane]    = v1;
        s_batch_f0[lane]    = f0;
        s_batch_f1[lane]    = f1;
        __syncwarp();

        // Lane 0 consumes the batch in lane-order (= edge order) to preserve
        // the deterministic reservoir-sampling behavior of the sequential version.
        if (lane == 0) {
            for (int k = 0; k < 32; k++) {
                if (!s_batch_valid[k]) continue;

                s_n_seen++;
                if (s_n_res < n_concave_edges) {
                    int slot = s_n_res;
                    reservoir[4*slot+0] = s_batch_v0[k];
                    reservoir[4*slot+1] = s_batch_v1[k];
                    reservoir[4*slot+2] = s_batch_f0[k];
                    reservoir[4*slot+3] = s_batch_f1[k];
                    s_n_res++;
                } else {
                    unsigned int rng = lcg_next(&s_rng);
                    unsigned int threshold_r = (unsigned int)((float)n_concave_edges / (float)s_n_seen * 4294967295.0f);
                    if (rng < threshold_r) {
                        unsigned int r2 = lcg_next(&s_rng);
                        int slot = (int)(r2 % (unsigned int)n_concave_edges);
                        reservoir[4*slot+0] = s_batch_v0[k];
                        reservoir[4*slot+1] = s_batch_v1[k];
                        reservoir[4*slot+2] = s_batch_f0[k];
                        reservoir[4*slot+3] = s_batch_f1[k];
                    }
                }
            }
        }
        __syncwarp();
    }

    // Generate planes from reservoir entries — warp-parallel across slots.
    int n_res = s_n_res;
    __shared__ int s_plane_count;
    if (lane == 0) s_plane_count = 0;
    __syncwarp();

    ConcaveEdgePlane* out = &edge_planes[i * 4 * n_concave_edges];

    for (int r = lane; r < n_res; r += 32) {
        int v0 = reservoir[4*r+0];
        int v1 = reservoir[4*r+1];
        int f0 = reservoir[4*r+2];
        int f1 = reservoir[4*r+3];
        if (v0 < 0) continue;

        float3 n0, n1;
        {
            int t0a = mesh->tris[3*f0+0], t0b = mesh->tris[3*f0+1], t0c = mesh->tris[3*f0+2];
            float3 p0a = {mesh->verts[3*t0a+0], mesh->verts[3*t0a+1], mesh->verts[3*t0a+2]};
            float3 p0b = {mesh->verts[3*t0b+0], mesh->verts[3*t0b+1], mesh->verts[3*t0b+2]};
            float3 p0c = {mesh->verts[3*t0c+0], mesh->verts[3*t0c+1], mesh->verts[3*t0c+2]};
            float3 e1 = {p0b.x-p0a.x, p0b.y-p0a.y, p0b.z-p0a.z};
            float3 e2 = {p0c.x-p0a.x, p0c.y-p0a.y, p0c.z-p0a.z};
            n0.x = e1.y*e2.z - e1.z*e2.y;
            n0.y = e1.z*e2.x - e1.x*e2.z;
            n0.z = e1.x*e2.y - e1.y*e2.x;
            float len0 = sqrtf(n0.x*n0.x + n0.y*n0.y + n0.z*n0.z);
            if (len0 < 1e-12f) continue;
            n0.x /= len0; n0.y /= len0; n0.z /= len0;
        }
        {
            int t1a = mesh->tris[3*f1+0], t1b = mesh->tris[3*f1+1], t1c = mesh->tris[3*f1+2];
            float3 p1a = {mesh->verts[3*t1a+0], mesh->verts[3*t1a+1], mesh->verts[3*t1a+2]};
            float3 p1b = {mesh->verts[3*t1b+0], mesh->verts[3*t1b+1], mesh->verts[3*t1b+2]};
            float3 p1c = {mesh->verts[3*t1c+0], mesh->verts[3*t1c+1], mesh->verts[3*t1c+2]};
            float3 e1 = {p1b.x-p1a.x, p1b.y-p1a.y, p1b.z-p1a.z};
            float3 e2 = {p1c.x-p1a.x, p1c.y-p1a.y, p1c.z-p1a.z};
            n1.x = e1.y*e2.z - e1.z*e2.y;
            n1.y = e1.z*e2.x - e1.x*e2.z;
            n1.z = e1.x*e2.y - e1.y*e2.x;
            float len1 = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z);
            if (len1 < 1e-12f) continue;
            n1.x /= len1; n1.y /= len1; n1.z /= len1;
        }

        float midx = (mesh->verts[3*v0+0] + mesh->verts[3*v1+0]) * 0.5f;
        float midy = (mesh->verts[3*v0+1] + mesh->verts[3*v1+1]) * 0.5f;
        float midz = (mesh->verts[3*v0+2] + mesh->verts[3*v1+2]) * 0.5f;
        float eps  = concave_eps;

        int idx = atomicAdd(&s_plane_count, 2);
        out[idx].pa   = n1.x;
        out[idx].pb   = n1.y;
        out[idx].pc   = n1.z;
        out[idx].pd   = -(n1.x*midx + n1.y*midy + n1.z*midz + eps);
        out[idx+1].pa = n0.x;
        out[idx+1].pb = n0.y;
        out[idx+1].pc = n0.z;
        out[idx+1].pd = -(n0.x*midx + n0.y*midy + n0.z*midz + eps);

        float bx = n0.x + n1.x;
        float by = n0.y + n1.y;
        float bz = n0.z + n1.z;
        float blen = sqrtf(bx*bx + by*by + bz*bz);
        if (blen >= 1e-4f) {
            bx /= blen; by /= blen; bz /= blen;
            int idx2 = atomicAdd(&s_plane_count, 2);
            out[idx2].pa   = bx;
            out[idx2].pb   = by;
            out[idx2].pc   = bz;
            out[idx2].pd   = -(bx*midx + by*midy + bz*midz + eps);
            out[idx2+1].pa = bx;
            out[idx2+1].pb = by;
            out[idx2+1].pc = bz;
            out[idx2+1].pd = -(bx*midx + by*midy + bz*midz - eps);
        }
    }
    __syncwarp();

    if (lane == 0) n_edge_cuts[i] = s_plane_count;
    __syncwarp();

    // Free scratch memory
    if (lane == 0) {
        heap_free(&pool->scratch, s_scratch_base);
    }
}
