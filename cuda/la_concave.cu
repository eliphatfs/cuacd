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

    // Allocate scratch for sort: EdgeFace array + sort workspace
    // scratch = n_dir_edges * sizeof(EdgeFace) + n_dir_edges * sizeof(EdgeFace)
    //         + WS_MAX_STACK * 2 * sizeof(int)
    size_t ef_bytes   = (size_t)n_dir_edges * sizeof(EdgeFace);
    size_t sort_extra = ef_bytes + (size_t)WS_MAX_STACK * 2 * sizeof(int);
    size_t scratch_needed = ef_bytes + sort_extra;

    // Allocate from global pool (all lanes must see the pointer)
    WarpPool wp;
    if (lane == 0) {
        if (warppool_from_global(pool, (int)scratch_needed, &wp) != 0) {
            wp.base = NULL;
        }
    }
    long long wp_base = (long long)wp.base;
    wp_base = __shfl_sync(WARP_MASK, wp_base, 0);
    wp.base = (char*)wp_base;
    if (wp_base == 0) {
        if (lane == 0) { n_edge_cuts[i] = 0; }
        return;
    }

    EdgeFace* ef_arr = (EdgeFace*)wp.base;
    char* sort_scratch = wp.base + ef_bytes;

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
        if (lane == 0) { n_edge_cuts[i] = 0; }
        return;
    }
    __syncwarp();

    // Scan sorted edges: find shared edges (consecutive pairs with same lo,hi)
    // and compute dihedral angle.  Count concave edges.
    // We need: face normals.  Compute on the fly.

    // For each shared edge, we need:
    //   - The two face indices
    //   - The two vertex indices of the edge
    //   - The "other" vertex of face 1 (to determine concave vs convex)
    //
    // We'll use a second scratch region to store sampled concave edge info.
    // Each sampled edge produces up to 4 ConcaveEdgePlane values.
    // Maximum output per block: 4 * n_concave_edges planes.
    // We need scratch for storing (v0, v1, face0, face1) of concave edges
    // as we find them during the scan.  Max concave edges = n_dir_edges/2.
    // But we only keep n_concave_edges via reservoir sampling.

    // Reservoir sampling state (lane 0 manages):
    // reservoir[0..n_concave_edges-1] stores concave edge info
    // n_seen = total concave edges encountered so far

    // Scratch for reservoir: n_concave_edges * 4 ints (v0, v1, f0, f1)
    size_t reservoir_bytes = (size_t)n_concave_edges * 4 * sizeof(int);
    int* reservoir = NULL;
    if (n_concave_edges > 0 && lane == 0) {
        reservoir = (int*)warp_pool_alloc(&wp, (int)reservoir_bytes, lane);
    }
    long long res_ptr = (long long)reservoir;
    res_ptr = __shfl_sync(WARP_MASK, res_ptr, 0);
    reservoir = (int*)res_ptr;

    // Initialize reservoir
    if (reservoir) {
        for (int j = lane; j < n_concave_edges * 4; j += 32)
            reservoir[j] = -1;
    }
    __syncwarp();

    __shared__ int s_n_seen;   // total concave edges seen
    __shared__ int s_n_res;    // items currently in reservoir
    if (lane == 0) { s_n_seen = 0; s_n_res = 0; }
    __syncwarp();

    // RNG state per block
    __shared__ unsigned int s_rng;
    if (lane == 0) s_rng = (unsigned int)(blockIdx.x * 2654435761u + 1u);
    __syncwarp();

    // Scan sorted edges for shared edges
    // Each lane checks e and e+1; but to avoid races, lane 0 does it sequentially.
    // For small meshes this is fine; for large meshes we could parallelize.
    // Since the kernel is 1 warp, lane 0 sequential scan is adequate.
    if (lane == 0) {
        for (int e = 0; e < n_dir_edges - 1; e++) {
            EdgeFace* ef0 = &ef_arr[e];
            EdgeFace* ef1 = &ef_arr[e + 1];
            if (ef0->lo != ef1->lo || ef0->hi != ef1->hi) continue;
            if (ef0->face_idx == ef1->face_idx) continue;  // shouldn't happen

            // Shared edge between ef0->face_idx and ef1->face_idx
            int v0 = ef0->lo;
            int v1 = ef0->hi;
            int f0 = ef0->face_idx;
            int f1 = ef1->face_idx;

            // Compute face normals
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

            // Compute dihedral angle
            float cos_a = n0.x*n1.x + n0.y*n1.y + n0.z*n1.z;
            cos_a = fmaxf(-1.0f, fminf(1.0f, cos_a));
            float alpha = acosf(cos_a);

            // Find "other" vertex of f0 not on the edge
            int v_other = -1;
            for (int vi = 0; vi < 3; vi++) {
                int vv = mesh->tris[3*f0+vi];
                if (vv != v0 && vv != v1) { v_other = vv; break; }
            }
            if (v_other < 0) continue;

            // Determine concave vs convex using dot product
            float3 edge_v = {mesh->verts[3*v0+0], mesh->verts[3*v0+1], mesh->verts[3*v0+2]};
            float3 other_v = {mesh->verts[3*v_other+0], mesh->verts[3*v_other+1], mesh->verts[3*v_other+2]};
            float3 diff = {other_v.x - edge_v.x, other_v.y - edge_v.y, other_v.z - edge_v.z};
            float d = diff.x*n1.x + diff.y*n1.y + diff.z*n1.z;

            const float pi = 3.14159265358979f;
            float dihedral = (d > 0.0f) ? (pi + alpha) : (pi - alpha);

            if (dihedral <= concave_threshold) continue;  // not concave enough

            // This edge is concave — reservoir sampling
            s_n_seen++;
            if (s_n_res < n_concave_edges) {
                // Fill reservoir
                int slot = s_n_res;
                reservoir[4*slot+0] = v0;
                reservoir[4*slot+1] = v1;
                reservoir[4*slot+2] = f0;
                reservoir[4*slot+3] = f1;
                s_n_res++;
            } else {
                // Replace with probability n_concave_edges / s_n_seen
                unsigned int rng = lcg_next(&s_rng);
                unsigned int threshold_r = (unsigned int)((float)n_concave_edges / (float)s_n_seen * 4294967295.0f);
                if (rng < threshold_r) {
                    // Pick random slot to replace
                    unsigned int r2 = lcg_next(&s_rng);
                    int slot = (int)(r2 % (unsigned int)n_concave_edges);
                    reservoir[4*slot+0] = v0;
                    reservoir[4*slot+1] = v1;
                    reservoir[4*slot+2] = f0;
                    reservoir[4*slot+3] = f1;
                }
            }

            // Skip past the pair (e and e+1 are same edge)
            // Check if e+2 is also same edge (3+ faces sharing an edge — skip extra)
            while (e + 2 < n_dir_edges &&
                   ef_arr[e+2].lo == ef0->lo && ef_arr[e+2].hi == ef0->hi) {
                e++;  // skip non-manifold edge (shared by 3+ faces)
            }
            e++;  // skip ef1
        }
    }
    __syncwarp();

    // Now generate planes from reservoir entries (lane 0)
    int n_res = s_n_res;
    if (lane == 0) {
        int plane_count = 0;
        ConcaveEdgePlane* out = &edge_planes[i * 4 * n_concave_edges];

        for (int r = 0; r < n_res; r++) {
            int v0 = reservoir[4*r+0];
            int v1 = reservoir[4*r+1];
            int f0 = reservoir[4*r+2];
            int f1 = reservoir[4*r+3];
            if (v0 < 0) continue;  // unused slot

            // Compute face normals again
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

            // Edge midpoint
            float midx = (mesh->verts[3*v0+0] + mesh->verts[3*v1+0]) * 0.5f;
            float midy = (mesh->verts[3*v0+1] + mesh->verts[3*v1+1]) * 0.5f;
            float midz = (mesh->verts[3*v0+2] + mesh->verts[3*v1+2]) * 0.5f;

            float eps = concave_eps;

            // Plane 1: face-offset using n1 (offset eps outside face 1)
            out[plane_count].pa = n1.x;
            out[plane_count].pb = n1.y;
            out[plane_count].pc = n1.z;
            out[plane_count].pd = -(n1.x*midx + n1.y*midy + n1.z*midz + eps);
            plane_count++;

            // Plane 2: face-offset using n2 (offset eps outside face 2)
            out[plane_count].pa = n0.x;
            out[plane_count].pb = n0.y;
            out[plane_count].pc = n0.z;
            out[plane_count].pd = -(n0.x*midx + n0.y*midy + n0.z*midz + eps);
            plane_count++;

            // Bisector planes (skip if ||n1+n2|| < 1e-4)
            float bx = n0.x + n1.x;
            float by = n0.y + n1.y;
            float bz = n0.z + n1.z;
            float blen = sqrtf(bx*bx + by*by + bz*bz);
            if (blen >= 1e-4f) {
                bx /= blen; by /= blen; bz /= blen;

                // Plane 3: bisector + eps
                out[plane_count].pa = bx;
                out[plane_count].pb = by;
                out[plane_count].pc = bz;
                out[plane_count].pd = -(bx*midx + by*midy + bz*midz + eps);
                plane_count++;

                // Plane 4: bisector - eps
                out[plane_count].pa = bx;
                out[plane_count].pb = by;
                out[plane_count].pc = bz;
                out[plane_count].pd = -(bx*midx + by*midy + bz*midz - eps);
                plane_count++;
            }
        }

        n_edge_cuts[i] = plane_count;
    }
    __syncwarp();

    // Note: WarpPool memory is reclaimed automatically (bump alloc from global pool).
    // It will be reclaimed when the pool offset is reset or the pool is destroyed.
}
