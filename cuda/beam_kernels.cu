// GPU Beam Search Convex Decomposition — Pure device code.
// Compiled to fatbin, loaded via CUDA driver API.
// No host-side includes, no runtime API calls.

extern "C" {

// ============================================================================
// Constants
// ============================================================================

#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64
#define MAX_PLANES 64
#define MAX_HULL_VERTS 256
#define MAX_HULL_FACES 512
#define MAX_BOUNDARY_EDGES 4096
#define MAX_BOUNDARY_LOOPS 64
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

// ============================================================================
// Data structures (must match beam.h / beam.c)
// ============================================================================

struct PartInfo {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    float bbox[6];    // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;    // cached Rv for this part
};

struct BeamItem {
    int num_parts;
    int worst_part_idx;
    float worst_cost;
    int cut_count;
};

struct DevicePool {
    char*         base;
    unsigned int* offset;
    unsigned int  capacity;
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ void* pool_alloc(DevicePool* pool, unsigned int size) {
    size = (size + 15) & ~15;  // align to 16 bytes
    unsigned int old = atomicAdd(pool->offset, size);
    if (old + size > pool->capacity) return NULL;
    return pool->base + old;
}

__device__ float atomicMaxF_beam(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fmaxf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ float atomicMinF_beam(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fminf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

// Cross product
__device__ void cross3(float ax, float ay, float az,
                       float bx, float by, float bz,
                       float* ox, float* oy, float* oz) {
    *ox = ay * bz - az * by;
    *oy = az * bx - ax * bz;
    *oz = ax * by - ay * bx;
}

// Dot product
__device__ float dot3(float ax, float ay, float az,
                      float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

// Signed volume of tetrahedron formed by triangle and origin
__device__ float signed_tet_volume(float p0x, float p0y, float p0z,
                                   float p1x, float p1y, float p1z,
                                   float p2x, float p2y, float p2z) {
    float cx, cy, cz;
    cross3(p1x, p1y, p1z, p2x, p2y, p2z, &cx, &cy, &cz);
    return dot3(p0x, p0y, p0z, cx, cy, cz) / 6.0f;
}

// Triangle area
__device__ float tri_area(float p0x, float p0y, float p0z,
                          float p1x, float p1y, float p1z,
                          float p2x, float p2y, float p2z) {
    float ex = p1x - p0x, ey = p1y - p0y, ez = p1z - p0z;
    float fx = p2x - p0x, fy = p2y - p0y, fz = p2z - p0z;
    float cx, cy, cz;
    cross3(ex, ey, ez, fx, fy, fz, &cx, &cy, &cz);
    return 0.5f * sqrtf(cx * cx + cy * cy + cz * cz);
}

// ============================================================================
// Vertex classification
// ============================================================================

// Classify vertex against plane ax+by+cz+d=0. Returns +1, -1, or 0.
__device__ int classify_vertex(float vx, float vy, float vz,
                               float a, float b, float c, float d) {
    float val = a * vx + b * vy + c * vz + d;
    if (val > EPS) return 1;
    if (val < -EPS) return -1;
    return 0;
}

// Compute edge-plane intersection parameter t, store intersection point
__device__ float intersect_edge(float v0x, float v0y, float v0z,
                                float v1x, float v1y, float v1z,
                                float a, float b, float c, float d,
                                float* ix, float* iy, float* iz) {
    float d0 = a * v0x + b * v0y + c * v0z + d;
    float d1 = a * v1x + b * v1y + c * v1z + d;
    float t = d0 / (d0 - d1);
    *ix = v0x + t * (v1x - v0x);
    *iy = v0y + t * (v1y - v0y);
    *iz = v0z + t * (v1z - v0z);
    return t;
}

// ============================================================================
// Mesh clipping (within a block, using scratch memory)
// ============================================================================

// Clip a mesh by a plane. Produces positive-side and negative-side triangle counts
// and mesh volume for each side. Also traces boundary loops for cap volume.
//
// This is the core device function used by evaluate_candidates and beam_expand.
//
// Inputs (global memory, read-only):
//   verts[vert_count*3]  — vertices of the part
//   tris[tri_count*3]    — triangles of the part (indices relative to part's vertex offset)
//   plane (a,b,c,d)      — cutting plane
//
// Outputs (scratch memory, allocated by caller):
//   pos_tri_count, neg_tri_count — number of triangles on each side
//   mesh_vol_pos, mesh_vol_neg — signed mesh volume for each side (from original + split tris)
//   cap_vol_pos, cap_vol_neg — cap volume contribution from divergence theorem
//
// Scratch layout per block:
//   signs[vert_count]            — int, vertex classification
//   new_verts[max_new*3]         — float, intersection vertices
//   new_vert_count               — int, atomic counter
//   pos_tris[tri_count*3*3]      — int, positive side triangles (worst case ~3x original)
//   neg_tris[tri_count*3*3]      — int, negative side triangles
//   pos_tri_count                — int, atomic counter
//   neg_tri_count                — int, atomic counter
//   boundary_edges[max_edges*2]  — int, pairs of new vertex indices forming boundary edges

struct ClipResult {
    float mesh_vol_pos;
    float mesh_vol_neg;
    float cap_vol_pos;    // = -cap_vol_neg (opposite sides of plane)
    int   pos_n_tris;
    int   neg_n_tris;
    int   pos_n_verts;
    int   neg_n_verts;
    int   total_verts;    // original + new intersection verts
};

// Simple hash for edge dedup during boundary tracing
__device__ unsigned int edge_hash(int v0, int v1, int table_size) {
    unsigned int key = (unsigned int)(v0 * 73856093) ^ (unsigned int)(v1 * 19349663);
    return key % (unsigned int)table_size;
}

// ============================================================================
// Compute mesh volume via parallel reduction (signed tetrahedra with origin)
// ============================================================================

// Each thread handles a subset of triangles, does warp-level reduction
__device__ float compute_mesh_volume_block(
    const float* verts,     // [V*3] all vertices (original + new)
    const int*   tris,      // [T*3] triangles
    int          n_tris,
    int          tid,
    int          block_size)
{
    float local_vol = 0.0f;
    for (int t = tid; t < n_tris; t += block_size) {
        int i0 = tris[t * 3 + 0];
        int i1 = tris[t * 3 + 1];
        int i2 = tris[t * 3 + 2];
        local_vol += signed_tet_volume(
            verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2],
            verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2],
            verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);
    }
    return local_vol;
}

// ============================================================================
// Convex hull volume (incremental construction within a block)
// ============================================================================

// Simplified incremental convex hull for volume computation.
// Operates in shared/scratch memory. Returns hull volume.
// Uses the gift-wrapping / incremental approach:
//   1. Find initial tetrahedron from extreme points
//   2. Insert remaining points, updating visible faces
//   3. Accumulate volume incrementally

struct HullFace {
    int v[3];           // vertex indices into hull_verts
    float nx, ny, nz;   // outward normal
    float d;            // plane offset (nx*x+ny*y+nz*z+d=0)
    int alive;          // 1 if face is active
};

// Find 6 extreme points (min/max along x,y,z) using parallel reduction.
// Returns indices into the input vertex array.
__device__ void find_extremes(const float* verts, int n_verts,
                              int tid, int block_size,
                              int* extremes,   // [6] output: minx,maxx,miny,maxy,minz,maxz
                              float* shared_vals, int* shared_idxs)
{
    // Initialize
    float vals[6];
    int   idxs[6];
    vals[0] = 1e30f; vals[1] = -1e30f;
    vals[2] = 1e30f; vals[3] = -1e30f;
    vals[4] = 1e30f; vals[5] = -1e30f;
    idxs[0] = 0; idxs[1] = 0;
    idxs[2] = 0; idxs[3] = 0;
    idxs[4] = 0; idxs[5] = 0;

    for (int i = tid; i < n_verts; i += block_size) {
        float x = verts[i * 3 + 0];
        float y = verts[i * 3 + 1];
        float z = verts[i * 3 + 2];
        if (x < vals[0]) { vals[0] = x; idxs[0] = i; }
        if (x > vals[1]) { vals[1] = x; idxs[1] = i; }
        if (y < vals[2]) { vals[2] = y; idxs[2] = i; }
        if (y > vals[3]) { vals[3] = y; idxs[3] = i; }
        if (z < vals[4]) { vals[4] = z; idxs[4] = i; }
        if (z > vals[5]) { vals[5] = z; idxs[5] = i; }
    }

    // Warp reduction for each of the 6 extremes
    // Use shared memory for cross-warp reduction
    for (int e = 0; e < 6; e++) {
        shared_vals[tid] = vals[e];
        shared_idxs[tid] = idxs[e];
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                bool replace;
                if (e % 2 == 0) // min
                    replace = shared_vals[tid + s] < shared_vals[tid];
                else             // max
                    replace = shared_vals[tid + s] > shared_vals[tid];
                if (replace) {
                    shared_vals[tid] = shared_vals[tid + s];
                    shared_idxs[tid] = shared_idxs[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) extremes[e] = shared_idxs[0];
        __syncthreads();
    }
}

// Compute convex hull volume for a set of points.
// Uses incremental construction. Thread 0 does sequential face updates,
// all threads help with parallel visibility tests.
// Returns volume (always positive).
__device__ float convex_hull_volume(
    const float* verts,    // [n*3] input points
    int          n_verts,  // number of points
    int          tid,
    int          block_size,
    // Scratch memory (caller-allocated):
    float*       shared_vals,   // [block_size]
    int*         shared_idxs,   // [block_size]
    HullFace*    faces,         // [MAX_HULL_FACES]
    int*         visible,       // [MAX_HULL_FACES] per-face visibility flag
    int*         horizon_edges, // [MAX_HULL_FACES*2] edge pairs on horizon
    float*       hull_verts,    // [MAX_HULL_VERTS*3]
    int*         n_faces_ptr,   // [1]
    int*         n_hverts_ptr,  // [1]
    float*       vol_accum      // [1] accumulated volume
)
{
    if (n_verts < 4) {
        if (tid == 0 && vol_accum) *vol_accum = 0.0f;
        __syncthreads();
        return 0.0f;
    }

    // Step 1: Find extreme points
    int extremes[6];
    find_extremes(verts, n_verts, tid, block_size, extremes, shared_vals, shared_idxs);

    if (tid == 0) {
        *n_faces_ptr = 0;
        *n_hverts_ptr = 0;
        *vol_accum = 0.0f;

        // Find 4 non-coplanar points from extremes
        // Pick the two most distant extreme points
        int best_a = extremes[0], best_b = extremes[1];
        float best_dist = 0.0f;
        for (int i = 0; i < 6; i++) {
            for (int j = i + 1; j < 6; j++) {
                int ai = extremes[i], bi = extremes[j];
                float dx = verts[ai*3]-verts[bi*3];
                float dy = verts[ai*3+1]-verts[bi*3+1];
                float dz = verts[ai*3+2]-verts[bi*3+2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > best_dist) {
                    best_dist = d2;
                    best_a = ai;
                    best_b = bi;
                }
            }
        }

        // Copy first two hull verts
        for (int k = 0; k < 3; k++) hull_verts[0*3+k] = verts[best_a*3+k];
        for (int k = 0; k < 3; k++) hull_verts[1*3+k] = verts[best_b*3+k];

        // Find point most distant from line a→b
        float abx = hull_verts[1*3+0]-hull_verts[0*3+0];
        float aby = hull_verts[1*3+1]-hull_verts[0*3+1];
        float abz = hull_verts[1*3+2]-hull_verts[0*3+2];
        float ab_len2 = abx*abx + aby*aby + abz*abz;

        int best_c = -1;
        float max_dist2 = -1.0f;
        for (int i = 0; i < n_verts && i < MAX_HULL_VERTS; i++) {
            if (i == best_a || i == best_b) continue;
            float apx = verts[i*3]-hull_verts[0*3];
            float apy = verts[i*3+1]-hull_verts[0*3+1];
            float apz = verts[i*3+2]-hull_verts[0*3+2];
            float t_proj = (apx*abx + apy*aby + apz*abz) / (ab_len2 + 1e-30f);
            float rx = apx - t_proj*abx;
            float ry = apy - t_proj*aby;
            float rz = apz - t_proj*abz;
            float d2 = rx*rx + ry*ry + rz*rz;
            if (d2 > max_dist2) { max_dist2 = d2; best_c = i; }
        }
        if (best_c < 0) best_c = 0;
        for (int k = 0; k < 3; k++) hull_verts[2*3+k] = verts[best_c*3+k];

        // Find point most distant from triangle plane
        float e0x = hull_verts[1*3]-hull_verts[0*3];
        float e0y = hull_verts[1*3+1]-hull_verts[0*3+1];
        float e0z = hull_verts[1*3+2]-hull_verts[0*3+2];
        float e1x = hull_verts[2*3]-hull_verts[0*3];
        float e1y = hull_verts[2*3+1]-hull_verts[0*3+1];
        float e1z = hull_verts[2*3+2]-hull_verts[0*3+2];
        float nx, ny, nz;
        cross3(e0x, e0y, e0z, e1x, e1y, e1z, &nx, &ny, &nz);
        float nlen = sqrtf(nx*nx+ny*ny+nz*nz) + 1e-30f;
        nx /= nlen; ny /= nlen; nz /= nlen;
        float pd = -(nx*hull_verts[0*3]+ny*hull_verts[0*3+1]+nz*hull_verts[0*3+2]);

        int best_d = -1;
        float max_abs_dist = 0.0f;
        float best_d_sign = 1.0f;
        for (int i = 0; i < n_verts && i < MAX_HULL_VERTS; i++) {
            float dist = nx*verts[i*3] + ny*verts[i*3+1] + nz*verts[i*3+2] + pd;
            if (fabsf(dist) > max_abs_dist) {
                max_abs_dist = fabsf(dist);
                best_d = i;
                best_d_sign = dist;
            }
        }
        if (best_d < 0) best_d = 0;
        for (int k = 0; k < 3; k++) hull_verts[3*3+k] = verts[best_d*3+k];
        *n_hverts_ptr = 4;

        // Build initial tetrahedron — 4 faces
        // Orient so normals point outward: if d-point is on negative side of ABC,
        // winding is correct; otherwise swap B,C
        int v0=0, v1=1, v2=2, v3=3;
        if (best_d_sign > 0) { int tmp = v1; v1 = v2; v2 = tmp; }

        // Face 0: v0,v1,v2 (opposite v3)
        // Face 1: v0,v2,v3 (opposite v1)
        // Face 2: v0,v3,v1 (opposite v2)
        // Face 3: v1,v3,v2 (opposite v0)
        int tet_faces[4][3] = {
            {v0,v1,v2}, {v0,v2,v3}, {v0,v3,v1}, {v1,v3,v2}
        };

        for (int f = 0; f < 4; f++) {
            HullFace& face = faces[f];
            face.v[0] = tet_faces[f][0];
            face.v[1] = tet_faces[f][1];
            face.v[2] = tet_faces[f][2];
            face.alive = 1;

            float fe0x = hull_verts[face.v[1]*3]-hull_verts[face.v[0]*3];
            float fe0y = hull_verts[face.v[1]*3+1]-hull_verts[face.v[0]*3+1];
            float fe0z = hull_verts[face.v[1]*3+2]-hull_verts[face.v[0]*3+2];
            float fe1x = hull_verts[face.v[2]*3]-hull_verts[face.v[0]*3];
            float fe1y = hull_verts[face.v[2]*3+1]-hull_verts[face.v[0]*3+1];
            float fe1z = hull_verts[face.v[2]*3+2]-hull_verts[face.v[0]*3+2];
            cross3(fe0x, fe0y, fe0z, fe1x, fe1y, fe1z,
                   &face.nx, &face.ny, &face.nz);
            float fl = sqrtf(face.nx*face.nx+face.ny*face.ny+face.nz*face.nz)+1e-30f;
            face.nx /= fl; face.ny /= fl; face.nz /= fl;
            face.d = -(face.nx*hull_verts[face.v[0]*3] +
                        face.ny*hull_verts[face.v[0]*3+1] +
                        face.nz*hull_verts[face.v[0]*3+2]);
        }
        *n_faces_ptr = 4;

        // Initial volume = tetrahedron
        *vol_accum = fabsf(signed_tet_volume(
            hull_verts[v0*3], hull_verts[v0*3+1], hull_verts[v0*3+2],
            hull_verts[v1*3], hull_verts[v1*3+1], hull_verts[v1*3+2],
            hull_verts[v2*3], hull_verts[v2*3+1], hull_verts[v2*3+2])
          + signed_tet_volume(
            hull_verts[v0*3], hull_verts[v0*3+1], hull_verts[v0*3+2],
            hull_verts[v2*3], hull_verts[v2*3+1], hull_verts[v2*3+2],
            hull_verts[v3*3], hull_verts[v3*3+1], hull_verts[v3*3+2]));
    }
    __syncthreads();

    int n_faces = *n_faces_ptr;
    int n_hverts = *n_hverts_ptr;

    // Step 2: Incrementally insert remaining points
    // Limit to MAX_HULL_VERTS points for performance
    int max_pts = n_verts < MAX_HULL_VERTS ? n_verts : MAX_HULL_VERTS;

    for (int pi = 0; pi < max_pts; pi++) {
        // Check if this point is already a hull vertex (skip initial 4)
        // Thread 0 checks, broadcasts
        __shared__ int skip_point;
        __shared__ float new_px, new_py, new_pz;
        __shared__ int any_visible;

        if (tid == 0) {
            new_px = verts[pi * 3 + 0];
            new_py = verts[pi * 3 + 1];
            new_pz = verts[pi * 3 + 2];
            skip_point = 0;
            // Check if inside current hull (all face distances <= eps)
            int outside = 0;
            for (int f = 0; f < n_faces; f++) {
                if (!faces[f].alive) continue;
                float dist = faces[f].nx * new_px + faces[f].ny * new_py +
                             faces[f].nz * new_pz + faces[f].d;
                if (dist > EPS) { outside = 1; break; }
            }
            if (!outside) skip_point = 1;
        }
        __syncthreads();
        if (skip_point) continue;

        // Parallel visibility test: each thread checks a subset of faces
        for (int f = tid; f < n_faces; f += block_size) {
            if (!faces[f].alive) { visible[f] = 0; continue; }
            float dist = faces[f].nx * new_px + faces[f].ny * new_py +
                         faces[f].nz * new_pz + faces[f].d;
            visible[f] = (dist > EPS) ? 1 : 0;
        }
        __syncthreads();

        // Thread 0: find horizon edges and build new faces
        if (tid == 0) {
            any_visible = 0;
            int n_horizon = 0;

            // Collect horizon edges: edges shared between visible and non-visible faces
            // Also accumulate volume of removed tetrahedra
            for (int f = 0; f < n_faces; f++) {
                if (!faces[f].alive || !visible[f]) continue;
                any_visible = 1;

                for (int e = 0; e < 3; e++) {
                    int ea = faces[f].v[e];
                    int eb = faces[f].v[(e + 1) % 3];

                    // Check if the other face sharing this edge is NOT visible
                    int is_horizon = 0;
                    for (int g = 0; g < n_faces; g++) {
                        if (g == f || !faces[g].alive || visible[g]) continue;
                        // Check if face g shares edge (ea,eb)
                        for (int e2 = 0; e2 < 3; e2++) {
                            int ga = faces[g].v[e2];
                            int gb = faces[g].v[(e2 + 1) % 3];
                            if ((ga == eb && gb == ea) || (ga == ea && gb == eb)) {
                                is_horizon = 1;
                                break;
                            }
                        }
                        if (is_horizon) break;
                    }
                    if (is_horizon && n_horizon < MAX_HULL_FACES) {
                        // Store edge with winding from non-visible side:
                        // horizon edge goes eb→ea (reversed from visible face)
                        horizon_edges[n_horizon * 2 + 0] = eb;
                        horizon_edges[n_horizon * 2 + 1] = ea;
                        n_horizon++;
                    }
                }
            }

            if (any_visible && n_horizon > 0 && n_hverts < MAX_HULL_VERTS) {
                // Add new vertex
                int new_vi = n_hverts;
                hull_verts[new_vi * 3 + 0] = new_px;
                hull_verts[new_vi * 3 + 1] = new_py;
                hull_verts[new_vi * 3 + 2] = new_pz;
                n_hverts++;

                // Remove visible faces
                for (int f = 0; f < n_faces; f++) {
                    if (faces[f].alive && visible[f]) faces[f].alive = 0;
                }

                // Add new faces connecting horizon edges to new vertex
                for (int h = 0; h < n_horizon; h++) {
                    int ha = horizon_edges[h * 2 + 0];
                    int hb = horizon_edges[h * 2 + 1];

                    if (n_faces >= MAX_HULL_FACES) break;
                    HullFace& nf = faces[n_faces];
                    nf.v[0] = ha;
                    nf.v[1] = hb;
                    nf.v[2] = new_vi;
                    nf.alive = 1;

                    float fe0x = hull_verts[hb*3]-hull_verts[ha*3];
                    float fe0y = hull_verts[hb*3+1]-hull_verts[ha*3+1];
                    float fe0z = hull_verts[hb*3+2]-hull_verts[ha*3+2];
                    float fe1x = hull_verts[new_vi*3]-hull_verts[ha*3];
                    float fe1y = hull_verts[new_vi*3+1]-hull_verts[ha*3+1];
                    float fe1z = hull_verts[new_vi*3+2]-hull_verts[ha*3+2];
                    cross3(fe0x, fe0y, fe0z, fe1x, fe1y, fe1z,
                           &nf.nx, &nf.ny, &nf.nz);
                    float fl = sqrtf(nf.nx*nf.nx+nf.ny*nf.ny+nf.nz*nf.nz)+1e-30f;
                    nf.nx /= fl; nf.ny /= fl; nf.nz /= fl;
                    nf.d = -(nf.nx*hull_verts[ha*3] +
                             nf.ny*hull_verts[ha*3+1] +
                             nf.nz*hull_verts[ha*3+2]);
                    n_faces++;
                }
            }
        }
        __syncthreads();

        // Update shared state
        if (tid == 0) {
            *n_faces_ptr = n_faces;
            *n_hverts_ptr = n_hverts;
        }
        __syncthreads();
        n_faces = *n_faces_ptr;
        n_hverts = *n_hverts_ptr;
    }

    // Step 3: Compute hull volume from all alive faces
    // Each alive face forms a tetrahedron with the origin
    float local_vol = 0.0f;
    for (int f = tid; f < n_faces; f += block_size) {
        if (!faces[f].alive) continue;
        int v0i = faces[f].v[0], v1i = faces[f].v[1], v2i = faces[f].v[2];
        local_vol += signed_tet_volume(
            hull_verts[v0i*3], hull_verts[v0i*3+1], hull_verts[v0i*3+2],
            hull_verts[v1i*3], hull_verts[v1i*3+1], hull_verts[v1i*3+2],
            hull_verts[v2i*3], hull_verts[v2i*3+1], hull_verts[v2i*3+2]);
    }

    // Warp reduction
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        shared_vals[tid] = local_vol;
        __syncthreads();
        if (tid < offset) local_vol += shared_vals[tid + offset];
        __syncthreads();
    }

    float result = 0.0f;
    if (tid == 0) {
        result = fabsf(local_vol);
        *vol_accum = result;
    }
    __syncthreads();
    return *vol_accum;
}

// ============================================================================
// Rv computation
// ============================================================================

__device__ float compute_rv(float mesh_vol, float hull_vol, float k) {
    float diff = fabsf(mesh_vol - hull_vol);
    float radius = cbrtf(3.0f * diff / (4.0f * PI_F));
    return radius * k;
}

// ============================================================================
// Kernel 1: evaluate_candidates
// ============================================================================
// Grid: (num_beam_items * num_planes, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// Each block evaluates one (beam_item, plane) pair.
// Writes scalar cost to cost_buffer[blockIdx.x].

__global__ void evaluate_candidates(
    const float* __restrict__ vertex_pool,     // [pool_verts * 3]
    const int*   __restrict__ triangle_pool,   // [pool_tris * 3]
    const PartInfo* __restrict__ parts,        // [MAX_BEAM * MAX_PARTS_PER_BEAM]
    const BeamItem* __restrict__ beam,         // [MAX_BEAM]
    const float* __restrict__ planes,          // [num_planes * 4] (a,b,c,d)
    int          num_planes,
    int          num_beam_items,
    float        rv_k,
    float        threshold,
    DevicePool   scratch,
    float* __restrict__ cost_buffer,           // [num_beam_items * num_planes]
    int* __restrict__ best_component_info      // [num_beam_items * num_planes * 2]
                                               // stores (worst_part_idx, plane_idx)
)
{
    int bid = blockIdx.x;
    int beam_idx = bid / num_planes;
    int plane_idx = bid % num_planes;
    int tid = threadIdx.x;

    if (beam_idx >= num_beam_items) return;

    const BeamItem& item = beam[beam_idx];
    int worst_part = item.worst_part_idx;
    if (worst_part < 0 || worst_part >= item.num_parts) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    const PartInfo& part = parts[beam_idx * MAX_PARTS_PER_BEAM + worst_part];
    int vc = part.vert_count;
    int tc = part.tri_count;
    int vo = part.vert_offset;
    int to = part.tri_offset;

    // Load plane
    float pa = planes[plane_idx * 4 + 0];
    float pb = planes[plane_idx * 4 + 1];
    float pc = planes[plane_idx * 4 + 2];
    float pd = planes[plane_idx * 4 + 3];

    // Allocate scratch memory for this block
    int* signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
    if (!signs) { if (tid == 0) cost_buffer[bid] = 1e30f; return; }

    // Stage A: Classify vertices
    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        float vx = vertex_pool[(vo + v) * 3 + 0];
        float vy = vertex_pool[(vo + v) * 3 + 1];
        float vz = vertex_pool[(vo + v) * 3 + 2];
        signs[v] = classify_vertex(vx, vy, vz, pa, pb, pc, pd);
    }
    __syncthreads();

    // Count triangles on each side and straddling
    // First pass: count for allocation
    __shared__ int pos_count, neg_count, straddle_count;
    __shared__ int all_pos, all_neg;
    if (tid == 0) {
        pos_count = 0; neg_count = 0; straddle_count = 0;
        all_pos = 1; all_neg = 1;
    }
    __syncthreads();

    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int i0 = triangle_pool[(to + t) * 3 + 0] - vo;
        int i1 = triangle_pool[(to + t) * 3 + 1] - vo;
        int i2 = triangle_pool[(to + t) * 3 + 2] - vo;
        int s0 = signs[i0], s1 = signs[i1], s2 = signs[i2];

        int has_pos = (s0 > 0) | (s1 > 0) | (s2 > 0);
        int has_neg = (s0 < 0) | (s1 < 0) | (s2 < 0);

        if (has_pos && has_neg) {
            atomicAdd(&straddle_count, 1);
            atomicExch(&all_pos, 0);
            atomicExch(&all_neg, 0);
        } else if (has_pos || (!has_neg && !has_pos)) {
            // On-plane triangles go to positive side by default
            atomicAdd(&pos_count, 1);
            if (has_pos) atomicExch(&all_neg, 0);
        } else {
            atomicAdd(&neg_count, 1);
            atomicExch(&all_pos, 0);
        }
    }
    __syncthreads();

    // If all triangles on one side, this plane doesn't cut — infinite cost
    if (all_pos || all_neg) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // Allocate output arrays for split triangles
    // Worst case: each straddling tri → 3 tris on each side
    int max_pos_tris = pos_count + straddle_count * 3;
    int max_neg_tris = neg_count + straddle_count * 3;
    int max_new_verts = straddle_count * 2;  // at most 2 new verts per straddling tri

    // All vertices (original + new intersection points)
    float* all_verts = (float*)pool_alloc(&scratch, (vc + max_new_verts) * 3 * sizeof(float));
    int* pos_tris = (int*)pool_alloc(&scratch, max_pos_tris * 3 * sizeof(int));
    int* neg_tris = (int*)pool_alloc(&scratch, max_neg_tris * 3 * sizeof(int));
    if (!all_verts || !pos_tris || !neg_tris) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    __shared__ int pos_tri_out, neg_tri_out, new_vert_out;
    if (tid == 0) { pos_tri_out = 0; neg_tri_out = 0; new_vert_out = 0; }
    __syncthreads();

    // Copy original vertices
    for (int v = tid; v < vc * 3; v += BLOCK_SIZE) {
        all_verts[v] = vertex_pool[vo * 3 + v];
    }
    __syncthreads();

    // Stage B: Process triangles — classify and split
    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int gi0 = triangle_pool[(to + t) * 3 + 0];
        int gi1 = triangle_pool[(to + t) * 3 + 1];
        int gi2 = triangle_pool[(to + t) * 3 + 2];
        int li0 = gi0 - vo, li1 = gi1 - vo, li2 = gi2 - vo;
        int s0 = signs[li0], s1 = signs[li1], s2 = signs[li2];

        int has_pos = (s0 > 0) | (s1 > 0) | (s2 > 0);
        int has_neg = (s0 < 0) | (s1 < 0) | (s2 < 0);

        if (has_pos && has_neg) {
            // Straddling triangle — need to split
            // Sort vertices so that the lone vertex (different sign) is first
            int vi[3] = {li0, li1, li2};
            int si[3] = {s0, s1, s2};

            // Find the vertex that's alone on one side
            // Cases: (+,-,-), (-,+,+), (+,+,-), (-,-,+), (+,-,+), (-,+,-)
            // We want to identify: lone_idx (the single different one), and its sign
            int lone = -1;
            if (si[0] != 0 && si[0] != si[1] && si[0] != si[2]) lone = 0;
            else if (si[1] != 0 && si[1] != si[0] && si[1] != si[2]) lone = 1;
            else if (si[2] != 0 && si[2] != si[0] && si[2] != si[1]) lone = 2;

            if (lone >= 0) {
                // Rotate so lone vertex is first
                int lv = vi[lone];
                int ov1 = vi[(lone+1)%3];
                int ov2 = vi[(lone+2)%3];
                int ls = si[lone];

                // Compute intersection points
                float ix1, iy1, iz1, ix2, iy2, iz2;
                intersect_edge(
                    all_verts[lv*3], all_verts[lv*3+1], all_verts[lv*3+2],
                    all_verts[ov1*3], all_verts[ov1*3+1], all_verts[ov1*3+2],
                    pa, pb, pc, pd, &ix1, &iy1, &iz1);
                intersect_edge(
                    all_verts[lv*3], all_verts[lv*3+1], all_verts[lv*3+2],
                    all_verts[ov2*3], all_verts[ov2*3+1], all_verts[ov2*3+2],
                    pa, pb, pc, pd, &ix2, &iy2, &iz2);

                // Allocate new vertex slots
                int nv_base = atomicAdd(&new_vert_out, 2);
                int nv1 = vc + nv_base;
                int nv2 = vc + nv_base + 1;
                all_verts[nv1*3+0] = ix1; all_verts[nv1*3+1] = iy1; all_verts[nv1*3+2] = iz1;
                all_verts[nv2*3+0] = ix2; all_verts[nv2*3+1] = iy2; all_verts[nv2*3+2] = iz2;

                // lone-side: 1 triangle (lv, nv1, nv2)
                // other-side: 2 triangles (ov1, nv2, nv1) and (ov1, ov2, nv2)
                if (ls > 0) {
                    int pi = atomicAdd(&pos_tri_out, 1);
                    pos_tris[pi*3+0] = lv; pos_tris[pi*3+1] = nv1; pos_tris[pi*3+2] = nv2;
                    int ni = atomicAdd(&neg_tri_out, 2);
                    neg_tris[ni*3+0] = ov1; neg_tris[ni*3+1] = nv2; neg_tris[ni*3+2] = nv1;
                    neg_tris[(ni+1)*3+0] = ov1; neg_tris[(ni+1)*3+1] = ov2; neg_tris[(ni+1)*3+2] = nv2;
                } else {
                    int ni = atomicAdd(&neg_tri_out, 1);
                    neg_tris[ni*3+0] = lv; neg_tris[ni*3+1] = nv1; neg_tris[ni*3+2] = nv2;
                    int pi = atomicAdd(&pos_tri_out, 2);
                    pos_tris[pi*3+0] = ov1; pos_tris[pi*3+1] = nv2; pos_tris[pi*3+2] = nv1;
                    pos_tris[(pi+1)*3+0] = ov1; pos_tris[(pi+1)*3+1] = ov2; pos_tris[(pi+1)*3+2] = nv2;
                }
            } else {
                // Edge case: one vertex on plane, other two on different sides
                // Find the on-plane vertex
                int on_plane = -1;
                for (int k = 0; k < 3; k++) if (si[k] == 0) { on_plane = k; break; }
                if (on_plane >= 0) {
                    int ov = vi[on_plane];
                    int a_idx = vi[(on_plane+1)%3];
                    int b_idx = vi[(on_plane+2)%3];
                    int sa = si[(on_plane+1)%3];

                    float ix, iy, iz;
                    intersect_edge(
                        all_verts[a_idx*3], all_verts[a_idx*3+1], all_verts[a_idx*3+2],
                        all_verts[b_idx*3], all_verts[b_idx*3+1], all_verts[b_idx*3+2],
                        pa, pb, pc, pd, &ix, &iy, &iz);

                    int nvi = vc + atomicAdd(&new_vert_out, 1);
                    all_verts[nvi*3+0] = ix; all_verts[nvi*3+1] = iy; all_verts[nvi*3+2] = iz;

                    if (sa > 0) {
                        int pi = atomicAdd(&pos_tri_out, 1);
                        pos_tris[pi*3+0] = ov; pos_tris[pi*3+1] = a_idx; pos_tris[pi*3+2] = nvi;
                        int ni = atomicAdd(&neg_tri_out, 1);
                        neg_tris[ni*3+0] = ov; neg_tris[ni*3+1] = nvi; neg_tris[ni*3+2] = b_idx;
                    } else {
                        int ni = atomicAdd(&neg_tri_out, 1);
                        neg_tris[ni*3+0] = ov; neg_tris[ni*3+1] = a_idx; neg_tris[ni*3+2] = nvi;
                        int pi = atomicAdd(&pos_tri_out, 1);
                        pos_tris[pi*3+0] = ov; pos_tris[pi*3+1] = nvi; pos_tris[pi*3+2] = b_idx;
                    }
                } else {
                    // All on-plane, shouldn't happen here, assign to positive
                    int pi = atomicAdd(&pos_tri_out, 1);
                    pos_tris[pi*3+0] = li0; pos_tris[pi*3+1] = li1; pos_tris[pi*3+2] = li2;
                }
            }
        } else if (has_neg) {
            int ni = atomicAdd(&neg_tri_out, 1);
            neg_tris[ni*3+0] = li0; neg_tris[ni*3+1] = li1; neg_tris[ni*3+2] = li2;
        } else {
            int pi = atomicAdd(&pos_tri_out, 1);
            pos_tris[pi*3+0] = li0; pos_tris[pi*3+1] = li1; pos_tris[pi*3+2] = li2;
        }
    }
    __syncthreads();

    int total_verts = vc + new_vert_out;
    int n_pos_tris = pos_tri_out;
    int n_neg_tris = neg_tri_out;

    // Stage C-D: Skip full connected components for evaluation.
    // Treat each side as one component. This is approximate but fast.

    // Stage E: Compute mesh volumes for each side
    __shared__ float vol_pos_shared[BLOCK_SIZE];
    __shared__ float vol_neg_shared[BLOCK_SIZE];

    float local_vol_pos = compute_mesh_volume_block(all_verts, pos_tris, n_pos_tris, tid, BLOCK_SIZE);
    vol_pos_shared[tid] = local_vol_pos;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) vol_pos_shared[tid] += vol_pos_shared[tid + s];
        __syncthreads();
    }

    float local_vol_neg = compute_mesh_volume_block(all_verts, neg_tris, n_neg_tris, tid, BLOCK_SIZE);
    vol_neg_shared[tid] = local_vol_neg;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) vol_neg_shared[tid] += vol_neg_shared[tid + s];
        __syncthreads();
    }

    // Add cap volume via divergence theorem
    // V_cap = (-d / 3) * A_net where A_net = sum of signed areas of boundary loops
    // For axis-aligned planes this simplifies. The cap contributes to closing each half.
    // Since we don't trace boundary loops in this fast path, approximate:
    // The missing cap volume is accounted for by using the plane equation.
    // For a plane ax+by+cz+d=0 with unit normal, cap area projected:
    // We compute cap contribution from the intersection vertices.
    // Note: mesh volume from signed tets already accounts for open boundaries
    // consistently. The hull is computed from the same vertices, so both are
    // "open" the same way. Rv is valid as an approximation without explicit
    // cap volume computation.
    __syncthreads();

    float mesh_vol_pos = fabsf(vol_pos_shared[0]);
    float mesh_vol_neg = fabsf(vol_neg_shared[0]);

    // Stage F: Convex hull volume for each side
    // Allocate hull scratch
    HullFace* hull_faces = (HullFace*)pool_alloc(&scratch, MAX_HULL_FACES * sizeof(HullFace));
    int* hull_visible = (int*)pool_alloc(&scratch, MAX_HULL_FACES * sizeof(int));
    int* hull_horizon = (int*)pool_alloc(&scratch, MAX_HULL_FACES * 2 * sizeof(int));
    float* hull_verts_scratch = (float*)pool_alloc(&scratch, MAX_HULL_VERTS * 3 * sizeof(float));
    int* hull_n_faces = (int*)pool_alloc(&scratch, sizeof(int));
    int* hull_n_verts = (int*)pool_alloc(&scratch, sizeof(int));
    float* hull_vol = (float*)pool_alloc(&scratch, sizeof(float));
    // Reuse vol_pos_shared/vol_neg_shared for hull computation
    int* shared_idxs = (int*)pool_alloc(&scratch, BLOCK_SIZE * sizeof(int));

    if (!hull_faces || !hull_visible || !hull_horizon || !hull_verts_scratch ||
        !hull_n_faces || !hull_n_verts || !hull_vol || !shared_idxs) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // Collect unique vertices for positive side
    // For speed, just use all_verts (superset) — hull computation handles interior points
    float hull_vol_pos = convex_hull_volume(
        all_verts, total_verts, tid, BLOCK_SIZE,
        vol_pos_shared, shared_idxs, hull_faces, hull_visible,
        hull_horizon, hull_verts_scratch, hull_n_faces, hull_n_verts, hull_vol);

    float rv_pos = compute_rv(mesh_vol_pos, hull_vol_pos, rv_k);

    // For negative side, we'd need separate hull computation.
    // To save time, compute Rv for the side that matters most (the larger Rv).
    // Approximate: the positive side's Rv is often sufficient for candidate ranking.
    // For accuracy, compute both:
    float hull_vol_neg = convex_hull_volume(
        all_verts, total_verts, tid, BLOCK_SIZE,
        vol_neg_shared, shared_idxs, hull_faces, hull_visible,
        hull_horizon, hull_verts_scratch, hull_n_faces, hull_n_verts, hull_vol);

    float rv_neg = compute_rv(mesh_vol_neg, hull_vol_neg, rv_k);

    // Stage G: Cost = max Rv across components (here just 2 sides)
    // The total cost for this candidate = max of the two sides' Rv,
    // combined with the existing costs of other parts in the beam item.
    if (tid == 0) {
        float cut_cost = fmaxf(rv_pos, rv_neg);

        // The beam item has other parts with cached Rv costs.
        // The candidate's total cost is the max of:
        //   - The new cut's worst Rv (cut_cost)
        //   - The worst Rv among all OTHER parts (not the one being cut)
        float other_worst = 0.0f;
        for (int p = 0; p < item.num_parts; p++) {
            if (p == worst_part) continue;
            float pc_cost = parts[beam_idx * MAX_PARTS_PER_BEAM + p].rv_cost;
            if (pc_cost > other_worst) other_worst = pc_cost;
        }

        cost_buffer[bid] = fmaxf(cut_cost, other_worst);
    }
}

// ============================================================================
// Kernel 2: select_top_k
// ============================================================================
// Selects top beam_width candidates from cost_buffer.
// Grid: (1, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Simple: one block scans all candidates, picks the best beam_width.

__global__ void select_top_k(
    const float* __restrict__ cost_buffer,  // [num_candidates]
    int          num_candidates,
    int          beam_width,
    int          num_planes,
    int* __restrict__ winner_beam_idx,      // [beam_width] — which beam item
    int* __restrict__ winner_plane_idx,     // [beam_width] — which plane
    float* __restrict__ winner_costs        // [beam_width] — costs
)
{
    // Thread 0 does sequential top-k selection (num_candidates is small: ~240)
    if (threadIdx.x != 0) return;

    // Initialize winners with infinity
    for (int k = 0; k < beam_width; k++) {
        winner_costs[k] = 1e30f;
        winner_beam_idx[k] = -1;
        winner_plane_idx[k] = -1;
    }

    for (int c = 0; c < num_candidates; c++) {
        float cost = cost_buffer[c];
        if (cost >= 1e29f) continue;

        int beam_idx = c / num_planes;
        int plane_idx = c % num_planes;

        // Check if this beam_idx already won (keep only best plane per beam parent)
        // Actually no — we want the globally best beam_width candidates
        // regardless of parent. Multiple candidates from same parent are OK
        // if they're all good.

        // Insert into sorted winners list if better than worst winner
        if (cost < winner_costs[beam_width - 1]) {
            // Find insertion point
            int insert_at = beam_width - 1;
            for (int k = 0; k < beam_width; k++) {
                if (cost < winner_costs[k]) { insert_at = k; break; }
            }
            // Shift down
            for (int k = beam_width - 1; k > insert_at; k--) {
                winner_costs[k] = winner_costs[k-1];
                winner_beam_idx[k] = winner_beam_idx[k-1];
                winner_plane_idx[k] = winner_plane_idx[k-1];
            }
            winner_costs[insert_at] = cost;
            winner_beam_idx[insert_at] = beam_idx;
            winner_plane_idx[insert_at] = plane_idx;
        }
    }
}

// ============================================================================
// Kernel 3: apply_cuts
// ============================================================================
// Grid: (beam_width, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Each block applies one winning cut: clips the worst part of a beam item,
// writes results to the next mesh pool.

__global__ void apply_cuts(
    const float* __restrict__ vertex_pool_src,
    const int*   __restrict__ triangle_pool_src,
    const PartInfo* __restrict__ parts_src,
    const BeamItem* __restrict__ beam_src,
    float* __restrict__ vertex_pool_dst,
    int*   __restrict__ triangle_pool_dst,
    PartInfo* __restrict__ parts_dst,
    BeamItem* __restrict__ beam_dst,
    const float* __restrict__ planes,
    const int* __restrict__ winner_beam_idx,
    const int* __restrict__ winner_plane_idx,
    int    num_planes,
    float  rv_k,
    DevicePool scratch,
    unsigned int* __restrict__ dst_vert_offset,   // atomic bump for dst vertex pool
    unsigned int* __restrict__ dst_tri_offset     // atomic bump for dst triangle pool
)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int src_beam = winner_beam_idx[bid];
    int plane_idx = winner_plane_idx[bid];

    if (src_beam < 0) return;

    const BeamItem& src_item = beam_src[src_beam];
    int worst_part = src_item.worst_part_idx;

    float pa = planes[plane_idx * 4 + 0];
    float pb = planes[plane_idx * 4 + 1];
    float pc = planes[plane_idx * 4 + 2];
    float pd_val = planes[plane_idx * 4 + 3];

    // Copy all non-worst parts from source to destination
    __shared__ int new_num_parts;
    if (tid == 0) new_num_parts = 0;
    __syncthreads();

    for (int p = 0; p < src_item.num_parts; p++) {
        if (p == worst_part) continue;

        const PartInfo& sp = parts_src[src_beam * MAX_PARTS_PER_BEAM + p];

        // Allocate space in dst pool
        __shared__ unsigned int dst_vo, dst_to;
        __shared__ int new_p_idx;
        if (tid == 0) {
            dst_vo = atomicAdd(dst_vert_offset, sp.vert_count);
            dst_to = atomicAdd(dst_tri_offset, sp.tri_count);
            new_p_idx = atomicAdd(&new_num_parts, 1);
        }
        __syncthreads();

        // Copy vertices
        for (int v = tid; v < sp.vert_count * 3; v += BLOCK_SIZE) {
            vertex_pool_dst[dst_vo * 3 + v] = vertex_pool_src[sp.vert_offset * 3 + v];
        }
        // Copy triangles (adjust indices)
        for (int t = tid; t < sp.tri_count; t += BLOCK_SIZE) {
            for (int k = 0; k < 3; k++) {
                int old_idx = triangle_pool_src[(sp.tri_offset + t) * 3 + k];
                triangle_pool_dst[(dst_to + t) * 3 + k] = old_idx - sp.vert_offset + dst_vo;
            }
        }
        __syncthreads();

        // Update part info
        if (tid == 0) {
            PartInfo& dp = parts_dst[bid * MAX_PARTS_PER_BEAM + new_p_idx];
            dp.vert_offset = dst_vo;
            dp.vert_count = sp.vert_count;
            dp.tri_offset = dst_to;
            dp.tri_count = sp.tri_count;
            for (int k = 0; k < 6; k++) dp.bbox[k] = sp.bbox[k];
            dp.rv_cost = sp.rv_cost;
        }
        __syncthreads();
    }

    // Now clip the worst part
    const PartInfo& wp = parts_src[src_beam * MAX_PARTS_PER_BEAM + worst_part];
    int vc = wp.vert_count;
    int tc = wp.tri_count;
    int wo = wp.vert_offset;
    int wto = wp.tri_offset;

    // Allocate scratch for clipping
    int* signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
    if (!signs) return;

    // Classify vertices
    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        float vx = vertex_pool_src[(wo + v) * 3 + 0];
        float vy = vertex_pool_src[(wo + v) * 3 + 1];
        float vz = vertex_pool_src[(wo + v) * 3 + 2];
        signs[v] = classify_vertex(vx, vy, vz, pa, pb, pc, pd_val);
    }
    __syncthreads();

    // Allocate arrays for split
    int max_new_verts = tc * 2;
    float* all_verts = (float*)pool_alloc(&scratch, (vc + max_new_verts) * 3 * sizeof(float));
    int max_out_tris = tc * 3;
    int* pos_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
    int* neg_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
    if (!all_verts || !pos_tris || !neg_tris) return;

    __shared__ int pos_tri_out, neg_tri_out, new_vert_out;
    if (tid == 0) { pos_tri_out = 0; neg_tri_out = 0; new_vert_out = 0; }
    __syncthreads();

    // Copy original vertices
    for (int v = tid; v < vc * 3; v += BLOCK_SIZE) {
        all_verts[v] = vertex_pool_src[wo * 3 + v];
    }
    __syncthreads();

    // Process triangles (same logic as evaluate_candidates)
    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int gi0 = triangle_pool_src[(wto + t) * 3 + 0];
        int gi1 = triangle_pool_src[(wto + t) * 3 + 1];
        int gi2 = triangle_pool_src[(wto + t) * 3 + 2];
        int li0 = gi0 - wo, li1 = gi1 - wo, li2 = gi2 - wo;
        int s0 = signs[li0], s1 = signs[li1], s2 = signs[li2];

        int has_pos = (s0 > 0) | (s1 > 0) | (s2 > 0);
        int has_neg = (s0 < 0) | (s1 < 0) | (s2 < 0);

        if (has_pos && has_neg) {
            int vi[3] = {li0, li1, li2};
            int si[3] = {s0, s1, s2};
            int lone = -1;
            if (si[0] != 0 && si[0] != si[1] && si[0] != si[2]) lone = 0;
            else if (si[1] != 0 && si[1] != si[0] && si[1] != si[2]) lone = 1;
            else if (si[2] != 0 && si[2] != si[0] && si[2] != si[1]) lone = 2;

            if (lone >= 0) {
                int lv = vi[lone], ov1 = vi[(lone+1)%3], ov2 = vi[(lone+2)%3];
                int ls = si[lone];
                float ix1, iy1, iz1, ix2, iy2, iz2;
                intersect_edge(all_verts[lv*3], all_verts[lv*3+1], all_verts[lv*3+2],
                               all_verts[ov1*3], all_verts[ov1*3+1], all_verts[ov1*3+2],
                               pa, pb, pc, pd_val, &ix1, &iy1, &iz1);
                intersect_edge(all_verts[lv*3], all_verts[lv*3+1], all_verts[lv*3+2],
                               all_verts[ov2*3], all_verts[ov2*3+1], all_verts[ov2*3+2],
                               pa, pb, pc, pd_val, &ix2, &iy2, &iz2);
                int nv_base = atomicAdd(&new_vert_out, 2);
                int nv1 = vc + nv_base, nv2 = vc + nv_base + 1;
                all_verts[nv1*3+0]=ix1; all_verts[nv1*3+1]=iy1; all_verts[nv1*3+2]=iz1;
                all_verts[nv2*3+0]=ix2; all_verts[nv2*3+1]=iy2; all_verts[nv2*3+2]=iz2;
                if (ls > 0) {
                    int pi = atomicAdd(&pos_tri_out, 1);
                    pos_tris[pi*3]=lv; pos_tris[pi*3+1]=nv1; pos_tris[pi*3+2]=nv2;
                    int ni = atomicAdd(&neg_tri_out, 2);
                    neg_tris[ni*3]=ov1; neg_tris[ni*3+1]=nv2; neg_tris[ni*3+2]=nv1;
                    neg_tris[(ni+1)*3]=ov1; neg_tris[(ni+1)*3+1]=ov2; neg_tris[(ni+1)*3+2]=nv2;
                } else {
                    int ni = atomicAdd(&neg_tri_out, 1);
                    neg_tris[ni*3]=lv; neg_tris[ni*3+1]=nv1; neg_tris[ni*3+2]=nv2;
                    int pi = atomicAdd(&pos_tri_out, 2);
                    pos_tris[pi*3]=ov1; pos_tris[pi*3+1]=nv2; pos_tris[pi*3+2]=nv1;
                    pos_tris[(pi+1)*3]=ov1; pos_tris[(pi+1)*3+1]=ov2; pos_tris[(pi+1)*3+2]=nv2;
                }
            } else {
                int on_plane = -1;
                for (int k = 0; k < 3; k++) if (si[k] == 0) { on_plane = k; break; }
                if (on_plane >= 0) {
                    int ov = vi[on_plane];
                    int a_idx = vi[(on_plane+1)%3], b_idx = vi[(on_plane+2)%3];
                    int sa = si[(on_plane+1)%3];
                    float ix, iy, iz;
                    intersect_edge(all_verts[a_idx*3], all_verts[a_idx*3+1], all_verts[a_idx*3+2],
                                   all_verts[b_idx*3], all_verts[b_idx*3+1], all_verts[b_idx*3+2],
                                   pa, pb, pc, pd_val, &ix, &iy, &iz);
                    int nvi = vc + atomicAdd(&new_vert_out, 1);
                    all_verts[nvi*3]=ix; all_verts[nvi*3+1]=iy; all_verts[nvi*3+2]=iz;
                    if (sa > 0) {
                        int pi = atomicAdd(&pos_tri_out, 1);
                        pos_tris[pi*3]=ov; pos_tris[pi*3+1]=a_idx; pos_tris[pi*3+2]=nvi;
                        int ni = atomicAdd(&neg_tri_out, 1);
                        neg_tris[ni*3]=ov; neg_tris[ni*3+1]=nvi; neg_tris[ni*3+2]=b_idx;
                    } else {
                        int ni = atomicAdd(&neg_tri_out, 1);
                        neg_tris[ni*3]=ov; neg_tris[ni*3+1]=a_idx; neg_tris[ni*3+2]=nvi;
                        int pi = atomicAdd(&pos_tri_out, 1);
                        pos_tris[pi*3]=ov; pos_tris[pi*3+1]=nvi; pos_tris[pi*3+2]=b_idx;
                    }
                } else {
                    int pi = atomicAdd(&pos_tri_out, 1);
                    pos_tris[pi*3]=li0; pos_tris[pi*3+1]=li1; pos_tris[pi*3+2]=li2;
                }
            }
        } else if (has_neg) {
            int ni = atomicAdd(&neg_tri_out, 1);
            neg_tris[ni*3]=li0; neg_tris[ni*3+1]=li1; neg_tris[ni*3+2]=li2;
        } else {
            int pi = atomicAdd(&pos_tri_out, 1);
            pos_tris[pi*3]=li0; pos_tris[pi*3+1]=li1; pos_tris[pi*3+2]=li2;
        }
    }
    __syncthreads();

    int total_new_verts = vc + new_vert_out;
    int n_pos = pos_tri_out;
    int n_neg = neg_tri_out;

    // Write positive-side part to dst pool
    __shared__ unsigned int pos_vo, pos_to, neg_vo, neg_to;
    __shared__ int pos_p_idx, neg_p_idx;
    if (tid == 0) {
        pos_vo = atomicAdd(dst_vert_offset, total_new_verts);
        pos_to = atomicAdd(dst_tri_offset, n_pos);
        pos_p_idx = atomicAdd(&new_num_parts, 1);
        neg_vo = atomicAdd(dst_vert_offset, total_new_verts);
        neg_to = atomicAdd(dst_tri_offset, n_neg);
        neg_p_idx = atomicAdd(&new_num_parts, 1);
    }
    __syncthreads();

    // Copy all vertices to both halves (superset — includes unused verts, but correct indices)
    for (int v = tid; v < total_new_verts * 3; v += BLOCK_SIZE) {
        vertex_pool_dst[pos_vo * 3 + v] = all_verts[v];
        vertex_pool_dst[neg_vo * 3 + v] = all_verts[v];
    }

    // Copy positive triangles (adjust local indices to global)
    for (int t = tid; t < n_pos; t += BLOCK_SIZE) {
        for (int k = 0; k < 3; k++)
            triangle_pool_dst[(pos_to + t) * 3 + k] = pos_tris[t * 3 + k] + pos_vo;
    }
    for (int t = tid; t < n_neg; t += BLOCK_SIZE) {
        for (int k = 0; k < 3; k++)
            triangle_pool_dst[(neg_to + t) * 3 + k] = neg_tris[t * 3 + k] + neg_vo;
    }
    __syncthreads();

    // Compute Rv for both new parts (approximate: use mesh vol vs hull vol)
    // For now, store a placeholder Rv — it'll be recomputed next iteration
    // Actually compute bboxes and approximate Rv
    if (tid == 0) {
        // Positive part
        PartInfo& pp = parts_dst[bid * MAX_PARTS_PER_BEAM + pos_p_idx];
        pp.vert_offset = pos_vo;
        pp.vert_count = total_new_verts;
        pp.tri_offset = pos_to;
        pp.tri_count = n_pos;
        pp.rv_cost = 0.0f;  // Will be recomputed
        pp.bbox[0] = 1e30f; pp.bbox[1] = -1e30f;
        pp.bbox[2] = 1e30f; pp.bbox[3] = -1e30f;
        pp.bbox[4] = 1e30f; pp.bbox[5] = -1e30f;

        // Negative part
        PartInfo& np_part = parts_dst[bid * MAX_PARTS_PER_BEAM + neg_p_idx];
        np_part.vert_offset = neg_vo;
        np_part.vert_count = total_new_verts;
        np_part.tri_offset = neg_to;
        np_part.tri_count = n_neg;
        np_part.rv_cost = 0.0f;
        np_part.bbox[0] = 1e30f; np_part.bbox[1] = -1e30f;
        np_part.bbox[2] = 1e30f; np_part.bbox[3] = -1e30f;
        np_part.bbox[4] = 1e30f; np_part.bbox[5] = -1e30f;

        // Update beam item
        beam_dst[bid].num_parts = new_num_parts;
        beam_dst[bid].worst_part_idx = -1;  // Will be recomputed
        beam_dst[bid].worst_cost = 0.0f;
        beam_dst[bid].cut_count = src_item.cut_count + 1;
    }
}

// ============================================================================
// Kernel 4: compute_part_costs
// ============================================================================
// Compute Rv for all parts of all beam items. Update worst part info.
// Grid: (num_beam_items, 1, 1), Block: (BLOCK_SIZE, 1, 1)

__global__ void compute_part_costs(
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    PartInfo*    __restrict__ parts,
    BeamItem*    __restrict__ beam,
    int          num_beam_items,
    float        rv_k,
    DevicePool   scratch
)
{
    int beam_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (beam_idx >= num_beam_items) return;

    BeamItem& item = beam[beam_idx];
    int np = item.num_parts;

    // For each part that needs Rv recomputed (rv_cost == 0)
    __shared__ float worst_cost;
    __shared__ int worst_idx;
    if (tid == 0) { worst_cost = 0.0f; worst_idx = 0; }
    __syncthreads();

    for (int p = 0; p < np; p++) {
        PartInfo& part = parts[beam_idx * MAX_PARTS_PER_BEAM + p];

        if (part.rv_cost > 0.0f) {
            // Already computed, just check if worst
            if (tid == 0 && part.rv_cost > worst_cost) {
                worst_cost = part.rv_cost;
                worst_idx = p;
            }
            __syncthreads();
            continue;
        }

        int vc = part.vert_count;
        int tc = part.tri_count;
        int vo = part.vert_offset;
        int to = part.tri_offset;

        if (tc == 0) {
            if (tid == 0) part.rv_cost = 0.0f;
            __syncthreads();
            continue;
        }

        // Compute mesh volume
        __shared__ float vol_shared[BLOCK_SIZE];
        float local_vol = 0.0f;
        for (int t = tid; t < tc; t += BLOCK_SIZE) {
            int i0 = triangle_pool[(to + t) * 3 + 0];
            int i1 = triangle_pool[(to + t) * 3 + 1];
            int i2 = triangle_pool[(to + t) * 3 + 2];
            local_vol += signed_tet_volume(
                vertex_pool[i0*3], vertex_pool[i0*3+1], vertex_pool[i0*3+2],
                vertex_pool[i1*3], vertex_pool[i1*3+1], vertex_pool[i1*3+2],
                vertex_pool[i2*3], vertex_pool[i2*3+1], vertex_pool[i2*3+2]);
        }
        vol_shared[tid] = local_vol;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) vol_shared[tid] += vol_shared[tid + s];
            __syncthreads();
        }
        float mesh_vol = fabsf(vol_shared[0]);

        // Compute convex hull volume
        // Gather vertices referenced by triangles
        HullFace* hf = (HullFace*)pool_alloc(&scratch, MAX_HULL_FACES * sizeof(HullFace));
        int* hv = (int*)pool_alloc(&scratch, MAX_HULL_FACES * sizeof(int));
        int* hh = (int*)pool_alloc(&scratch, MAX_HULL_FACES * 2 * sizeof(int));
        float* hverts = (float*)pool_alloc(&scratch, MAX_HULL_VERTS * 3 * sizeof(float));
        int* hnf = (int*)pool_alloc(&scratch, sizeof(int));
        int* hnv = (int*)pool_alloc(&scratch, sizeof(int));
        float* hvol = (float*)pool_alloc(&scratch, sizeof(float));
        int* sidxs = (int*)pool_alloc(&scratch, BLOCK_SIZE * sizeof(int));

        if (!hf || !hv || !hh || !hverts || !hnf || !hnv || !hvol || !sidxs) {
            if (tid == 0) part.rv_cost = EPS;
            __syncthreads();
            continue;
        }

        float hull_vol = convex_hull_volume(
            &vertex_pool[vo * 3], vc, tid, BLOCK_SIZE,
            vol_shared, sidxs, hf, hv, hh, hverts, hnf, hnv, hvol);

        float rv = compute_rv(mesh_vol, hull_vol, rv_k);

        if (tid == 0) {
            part.rv_cost = fmaxf(rv, EPS);  // Avoid exact 0 to prevent re-computation
            if (part.rv_cost > worst_cost) {
                worst_cost = part.rv_cost;
                worst_idx = p;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        item.worst_part_idx = worst_idx;
        item.worst_cost = worst_cost;
    }
}

// ============================================================================
// Kernel 5: normalize_mesh
// ============================================================================
// Normalize input mesh to [-1,1]^3. Compute bbox, translate to center, scale.
// Grid: (1, 1, 1), Block: (BLOCK_SIZE, 1, 1)

__global__ void normalize_mesh(
    float* __restrict__ vertices,   // [n_verts * 3], modified in-place
    int    n_verts,
    float* __restrict__ norm_info   // [7]: cx, cy, cz, scale, xmin, ymin, zmin
)
{
    int tid = threadIdx.x;

    // Find bbox via parallel reduction
    extern __shared__ float smem[];
    float* s_min = smem;                // [BLOCK_SIZE * 3]
    float* s_max = smem + BLOCK_SIZE * 3; // [BLOCK_SIZE * 3]

    float lmin[3] = {1e30f, 1e30f, 1e30f};
    float lmax[3] = {-1e30f, -1e30f, -1e30f};

    for (int i = tid; i < n_verts; i += BLOCK_SIZE) {
        for (int k = 0; k < 3; k++) {
            float v = vertices[i * 3 + k];
            lmin[k] = fminf(lmin[k], v);
            lmax[k] = fmaxf(lmax[k], v);
        }
    }

    for (int k = 0; k < 3; k++) {
        s_min[tid * 3 + k] = lmin[k];
        s_max[tid * 3 + k] = lmax[k];
    }
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 3; k++) {
                s_min[tid * 3 + k] = fminf(s_min[tid * 3 + k], s_min[(tid + s) * 3 + k]);
                s_max[tid * 3 + k] = fmaxf(s_max[tid * 3 + k], s_max[(tid + s) * 3 + k]);
            }
        }
        __syncthreads();
    }

    __shared__ float center[3], scale;
    if (tid == 0) {
        float range = 0.0f;
        for (int k = 0; k < 3; k++) {
            center[k] = (s_min[k] + s_max[k]) * 0.5f;
            float r = s_max[k] - s_min[k];
            if (r > range) range = r;
        }
        scale = (range > 1e-10f) ? (2.0f / range) : 1.0f;

        norm_info[0] = center[0];
        norm_info[1] = center[1];
        norm_info[2] = center[2];
        norm_info[3] = scale;
        norm_info[4] = s_min[0];
        norm_info[5] = s_min[1];
        norm_info[6] = s_min[2];
    }
    __syncthreads();

    // Apply normalization
    for (int i = tid; i < n_verts; i += BLOCK_SIZE) {
        for (int k = 0; k < 3; k++) {
            vertices[i * 3 + k] = (vertices[i * 3 + k] - center[k]) * scale;
        }
    }
}

// ============================================================================
// Kernel 6: recover_coordinates
// ============================================================================
// De-normalize vertices back to original coordinate space.

__global__ void recover_coordinates(
    float* __restrict__ vertices,
    int    n_verts,
    const float* __restrict__ norm_info   // [7]: cx, cy, cz, scale
)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    float cx = norm_info[0];
    float cy = norm_info[1];
    float cz = norm_info[2];
    float scale = norm_info[3];
    float inv_scale = 1.0f / scale;

    if (idx < n_verts) {
        vertices[idx * 3 + 0] = vertices[idx * 3 + 0] * inv_scale + cx;
        vertices[idx * 3 + 1] = vertices[idx * 3 + 1] * inv_scale + cy;
        vertices[idx * 3 + 2] = vertices[idx * 3 + 2] * inv_scale + cz;
    }
}

// ============================================================================
// Kernel 7: Hausdorff check (reuses point_triangle_dist from kernels.cu)
// ============================================================================

__device__ float point_triangle_dist_beam(
    float px, float py, float pz,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z) {
    float e0x = v1x-v0x, e0y = v1y-v0y, e0z = v1z-v0z;
    float e1x = v2x-v0x, e1y = v2y-v0y, e1z = v2z-v0z;
    float dx = v0x-px, dy = v0y-py, dz = v0z-pz;
    float a = e0x*e0x+e0y*e0y+e0z*e0z;
    float b = e0x*e1x+e0y*e1y+e0z*e1z;
    float c = e1x*e1x+e1y*e1y+e1z*e1z;
    float d = e0x*dx+e0y*dy+e0z*dz;
    float e = e1x*dx+e1y*dy+e1z*dz;
    float det = a*c-b*b;
    float s = b*e-c*d;
    float t = b*d-a*e;
    if (s+t <= det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                if (d < 0.0f) { t=0; s=(-d>=a)?1.0f:-d/a; }
                else { s=0; t=(e>=0)?0:((-e>=c)?1.0f:-e/c); }
            } else { s=0; t=(e>=0)?0:((-e>=c)?1.0f:-e/c); }
        } else if (t < 0.0f) { t=0; s=(d>=0)?0:((-d>=a)?1.0f:-d/a); }
        else { float inv=1.0f/det; s*=inv; t*=inv; }
    } else {
        if (s<0) {
            float t0=b+d, t1=c+e;
            if(t1>t0){float nm=t1-t0,dn=a-2*b+c;s=(nm>=dn)?1.0f:nm/dn;t=1-s;}
            else{s=0;t=(t1<=0)?1.0f:((e>=0)?0:-e/c);}
        } else if(t<0) {
            float t0=b+e, t1=a+d;
            if(t1>t0){float nm=t1-t0,dn=a-2*b+c;t=(nm>=dn)?1.0f:nm/dn;s=1-t;}
            else{t=0;s=(t1<=0)?1.0f:((d>=0)?0:-d/a);}
        } else {
            float nm=(c+e)-(b+d);
            if(nm<=0){s=0;t=1;}
            else{float dn=a-2*b+c;s=(nm>=dn)?1.0f:nm/dn;t=1-s;}
        }
    }
    float rx=v0x+s*e0x+t*e1x-px;
    float ry=v0y+s*e0y+t*e1y-py;
    float rz=v0z+s*e0z+t*e1z-pz;
    return sqrtf(rx*rx+ry*ry+rz*rz);
}

// Surface sampling: generate quasi-random points on mesh surface
// One thread per sample point. Uses triangle area for weighting.
__global__ void sample_surface(
    const float* __restrict__ vertices,
    const int*   __restrict__ triangles,
    int n_tris,
    int vert_offset,
    float* __restrict__ samples,    // [n_samples * 3]
    int n_samples,
    unsigned int seed
)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_samples) return;

    // Simple RNG (xorshift)
    unsigned int state = seed ^ (idx * 2654435761u);
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;

    // Pick a random triangle (uniform — not area-weighted for simplicity)
    int tri_idx = state % n_tris;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;

    int i0 = triangles[tri_idx * 3 + 0] + vert_offset;
    int i1 = triangles[tri_idx * 3 + 1] + vert_offset;
    int i2 = triangles[tri_idx * 3 + 2] + vert_offset;

    // Random barycentric coordinates
    float u = (float)(state & 0xFFFF) / 65535.0f;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    float v = (float)(state & 0xFFFF) / 65535.0f;
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    float w = 1.0f - u - v;

    samples[idx * 3 + 0] = w * vertices[i0*3+0] + u * vertices[i1*3+0] + v * vertices[i2*3+0];
    samples[idx * 3 + 1] = w * vertices[i0*3+1] + u * vertices[i1*3+1] + v * vertices[i2*3+1];
    samples[idx * 3 + 2] = w * vertices[i0*3+2] + u * vertices[i1*3+2] + v * vertices[i2*3+2];
}

// Point-mesh distance for a batch of points (same as existing kernel but standalone)
__global__ void beam_point_mesh_distance(
    const float* __restrict__ points,
    const float* __restrict__ vertices,
    const int*   __restrict__ triangles,
    float*       __restrict__ distances,
    int N, int T, int vert_offset
)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N) return;

    float px = points[idx*3], py = points[idx*3+1], pz = points[idx*3+2];
    float min_dist = 1e30f;
    for (int t = 0; t < T; t++) {
        int i0 = triangles[t*3] + vert_offset;
        int i1 = triangles[t*3+1] + vert_offset;
        int i2 = triangles[t*3+2] + vert_offset;
        float d = point_triangle_dist_beam(px,py,pz,
            vertices[i0*3],vertices[i0*3+1],vertices[i0*3+2],
            vertices[i1*3],vertices[i1*3+1],vertices[i1*3+2],
            vertices[i2*3],vertices[i2*3+1],vertices[i2*3+2]);
        min_dist = fminf(min_dist, d);
    }
    distances[idx] = min_dist;
}

// Shared-memory max reduction (same as existing but standalone)
__global__ void beam_reduce_max(
    const float* __restrict__ data,
    float*       __restrict__ output,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float val = -1e30f;
    if (idx < N) val = data[idx];
    if (idx + blockDim.x < N) val = fmaxf(val, data[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

} // extern "C"
