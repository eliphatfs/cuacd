// hull_warp.cuh — Warp-based (32-thread) convex hull volume algorithms.
//
// Two algorithms:
//   hull_quickhull_warp  — QuickHull with per-face conflict lists in scratch
//   hull_dandc_warp      — Preparata-Hong bottom-up divide-and-conquer
//
// Both use 1 warp (32 threads).  Thread 0 handles sequential topology work;
// all 32 threads participate in parallel reductions / point classification.
// All scratch in caller-provided WarpPool (global memory).
// No shared memory used.
//
// Requires: common.cuh (EPS, signed_tet_volume from geometry.cuh)

#ifndef HULL_WARP_CUH
#define HULL_WARP_CUH

#define WARP_SIZE 32
#define WARP_MASK 0xffffffffu

// ============================================================================
// WarpPool — bump allocator for warp-scope scratch
// ============================================================================

struct WarpPool {
    char*         base;
    int           offset;
    int           capacity;
    int           error;      // 1 = OOM set by warp_pool_alloc
};

// Allocate `bytes` (16-byte aligned) from pool.
// Thread 0 bumps the pointer; result is broadcast to all lanes via __shfl_sync.
// Returns NULL on OOM (pool->error is set).
__device__ inline void* warp_pool_alloc(WarpPool* pool, int bytes, int lane) {
    long long ptr_ll = 0LL;
    if (lane == 0) {
        int aligned = (bytes + 15) & ~15;
        if (pool->offset + aligned > pool->capacity) {
            pool->error = 1;
            ptr_ll = 0LL;
        } else {
            ptr_ll = (long long)(pool->base + pool->offset);
            pool->offset += aligned;
        }
    }
    ptr_ll = __shfl_sync(WARP_MASK, ptr_ll, 0);
    return (void*)ptr_ll;
}

// ============================================================================
// Warp reductions
// ============================================================================

// Reduce maximum float value across all 32 lanes.
__device__ inline float warp_max_f(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, off));
    return val;
}

// OR reduction: returns nonzero iff any lane passes nonzero flag.
__device__ inline int warp_any_i(int flag) {
    return (int)__any_sync(WARP_MASK, flag);
}

// (value, index) argmax across all 32 lanes.
// Each lane passes its local (val, idx); after the call every lane sees the
// global winner.
__device__ inline void warp_argmax_f(float* val, int* idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float v2 = __shfl_xor_sync(WARP_MASK, *val, off);
        int   i2 = __shfl_xor_sync(WARP_MASK, *idx, off);
        if (v2 > *val) { *val = v2; *idx = i2; }
    }
}

// Broadcast a float from lane 0 to all lanes.
__device__ inline float warp_bcast_f(float val) {
    return __shfl_sync(WARP_MASK, val, 0);
}

// Broadcast an int from lane 0 to all lanes.
__device__ inline int warp_bcast_i(int val) {
    return __shfl_sync(WARP_MASK, val, 0);
}

// ============================================================================
// Plane helpers (thread 0 only)
// ============================================================================

// Compute outward face plane (nx,ny,nz,nd) for CCW triangle (a,b,c).
// Plane equation: nx*x + ny*y + nz*z + nd = 0.
// Points outside (front) have positive distance.
__device__ inline void qh_set_face_plane(
    const float* pts,
    float* nx, float* ny, float* nz, float* nd,
    int a, int b, int c)
{
    float e1x = pts[b*3  ] - pts[a*3  ];
    float e1y = pts[b*3+1] - pts[a*3+1];
    float e1z = pts[b*3+2] - pts[a*3+2];
    float e2x = pts[c*3  ] - pts[a*3  ];
    float e2y = pts[c*3+1] - pts[a*3+1];
    float e2z = pts[c*3+2] - pts[a*3+2];
    *nx = e1y*e2z - e1z*e2y;
    *ny = e1z*e2x - e1x*e2z;
    *nz = e1x*e2y - e1y*e2x;
    *nd = -((*nx)*pts[a*3] + (*ny)*pts[a*3+1] + (*nz)*pts[a*3+2]);
}

// Signed distance from point (px,py,pz) to plane.  Positive = outside.
__device__ inline float qh_face_dist(
    float fnx, float fny, float fnz, float fnd,
    float px, float py, float pz)
{
    return fnx*px + fny*py + fnz*pz + fnd;
}

// ============================================================================
// Algorithm 1: hull_quickhull_warp
// ============================================================================
//
// QuickHull with per-face conflict lists stored as linked lists in pool
// scratch.  The outer loop repeatedly picks the point farthest from any face
// (the apex of the conflict list with the largest stored distance), tests
// face visibility, collects horizon edges, and distributes the former conflict
// points to the new fan faces.
//
// Volume is accumulated incrementally by tracking the signed-tet contribution
// of every face added or removed.
//
// All 32 threads must call with the same arguments.
// Returns hull volume (>= 0) or -1.0f on error.

__device__ float hull_quickhull_warp(
    const float* pts,   // [n_pts * 3] float xyz
    int          n_pts,
    int          lane,
    WarpPool*    pool,
    int*         error)
{
    *error = 0;

    if (n_pts < 4) {
        *error = 2;
        return -1.0f;
    }

    // ------------------------------------------------------------------
    // Scratch layout (all allocated from pool via warp_pool_alloc)
    // ------------------------------------------------------------------
    int max_faces = n_pts * 2 + 8;

    // Face vertex arrays
    int*   fv0 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   fv1 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   fv2 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    // Face adjacency (edge 0..2)
    int*   fadj0 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   fadj1 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   fadj2 = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    // Face plane
    float* fnx = (float*)warp_pool_alloc(pool, max_faces * (int)sizeof(float), lane);
    float* fny = (float*)warp_pool_alloc(pool, max_faces * (int)sizeof(float), lane);
    float* fnz = (float*)warp_pool_alloc(pool, max_faces * (int)sizeof(float), lane);
    float* fnd = (float*)warp_pool_alloc(pool, max_faces * (int)sizeof(float), lane);
    // Conflict list heads (one per face) and linked-list next per point
    int*   pts_head = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   pt_next  = (int*)warp_pool_alloc(pool, n_pts    * (int)sizeof(int), lane);
    // Atomic argmax scratch: (float_bits<<32 | pt_idx) packed as ull
    unsigned long long* pts_max_ull =
        (unsigned long long*)warp_pool_alloc(
            pool, max_faces * (int)sizeof(unsigned long long), lane);
    // Decoded argmax results
    int*   pts_max_i = (int*)  warp_pool_alloc(pool, max_faces * (int)sizeof(int),   lane);
    float* pts_max_d = (float*)warp_pool_alloc(pool, max_faces * (int)sizeof(float), lane);
    // Horizon edges
    int*   hz_a   = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   hz_b   = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    int*   hz_nbr = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    // Face stack (indices of faces with non-empty conflict lists)
    int*   face_stack = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    // Visibility flags
    int*   visible = (int*)warp_pool_alloc(pool, max_faces * (int)sizeof(int), lane);
    // Dead point buffer
    int*   dead_pts = (int*)warp_pool_alloc(pool, n_pts * (int)sizeof(int), lane);
    // Scalar state: [n_faces, n_stack, n_horizon, n_dead]
    int*   scalars = (int*)warp_pool_alloc(pool, 8 * (int)sizeof(int), lane);

    __syncwarp(WARP_MASK);

    // OOM check — all lanes see the same pool->error via the broadcast in alloc
    if (pool->error) {
        *error = 1;
        return -1.0f;
    }

    // Convenience aliases into scalars[]
    // scalars[0] = n_faces
    // scalars[1] = n_stack
    // scalars[2] = n_horizon
    // scalars[3] = n_dead

    // ------------------------------------------------------------------
    // Step 1: Find initial tetrahedron
    // ------------------------------------------------------------------
    // Warp-parallel: find extreme points along each axis.

    // Each lane scans its stride of points and tracks local extremes.
    float lox= 1e30f, hix=-1e30f;
    float loy= 1e30f, hiy=-1e30f;
    float loz= 1e30f, hiz=-1e30f;
    int ilox=lane, ihix=lane;
    int iloy=lane, ihiy=lane;
    int iloz=lane, ihiz=lane;

    // Seed with lane's first point if available; use lane 0's value otherwise
    if (lane < n_pts) {
        lox = hix = pts[lane*3  ]; ilox = ihix = lane;
        loy = hiy = pts[lane*3+1]; iloy = ihiy = lane;
        loz = hiz = pts[lane*3+2]; iloz = ihiz = lane;
    }

    for (int i = lane; i < n_pts; i += WARP_SIZE) {
        float x = pts[i*3  ];
        float y = pts[i*3+1];
        float z = pts[i*3+2];
        if (x < lox) { lox = x; ilox = i; }
        if (x > hix) { hix = x; ihix = i; }
        if (y < loy) { loy = y; iloy = i; }
        if (y > hiy) { hiy = y; ihiy = i; }
        if (z < loz) { loz = z; iloz = i; }
        if (z > hiz) { hiz = z; ihiz = i; }
    }

    // Warp-reduce argmin-x (use -lox as positive value for argmax)
    {
        float mv = -lox; int mi = ilox; warp_argmax_f(&mv, &mi);
        if (lane == 0) scalars[4] = mi;
    }
    {
        float mv =  hix; int mi = ihix; warp_argmax_f(&mv, &mi);
        if (lane == 0) scalars[5] = mi;
    }
    {
        float mv = -loy; int mi = iloy; warp_argmax_f(&mv, &mi);
        if (lane == 0) scalars[6] = mi;
    }
    {
        float mv =  hiy; int mi = ihiy; warp_argmax_f(&mv, &mi);
        if (lane == 0) scalars[7] = mi;
    }
    // We only need two extreme x candidates to seed the first edge.
    // Thread 0 picks the farthest pair from all 6 candidates.
    __syncwarp(WARP_MASK);

    // Thread 0: build the initial tetrahedron
    if (lane == 0) {
        scalars[0] = 0; // n_faces = 0
        scalars[1] = 0; // n_stack = 0

        // Six extreme point candidates (min/max per axis)
        int cands[6];
        cands[0] = scalars[4]; // min-x
        cands[1] = scalars[5]; // max-x
        cands[2] = scalars[6]; // min-y
        cands[3] = scalars[7]; // max-y
        // For z we'll compute inline (only needed here)
        {
            float best_loz =  1e30f; int best_iloz = 0;
            float best_hiz = -1e30f; int best_ihiz = 0;
            for (int i = 0; i < n_pts; i++) {
                float z = pts[i*3+2];
                if (z < best_loz) { best_loz = z; best_iloz = i; }
                if (z > best_hiz) { best_hiz = z; best_ihiz = i; }
            }
            cands[4] = best_iloz;
            cands[5] = best_ihiz;
        }

        // Pick pair with maximum separation distance
        int p0 = cands[0], p1 = cands[1];
        float best_d2 = -1.0f;
        for (int a = 0; a < 6; a++) {
            for (int b = a+1; b < 6; b++) {
                int ca = cands[a], cb = cands[b];
                if (ca == cb) continue;
                float dx = pts[ca*3  ]-pts[cb*3  ];
                float dy = pts[ca*3+1]-pts[cb*3+1];
                float dz = pts[ca*3+2]-pts[cb*3+2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > best_d2) { best_d2 = d2; p0 = ca; p1 = cb; }
            }
        }

        // Find p2: farthest from line p0-p1
        float dx01 = pts[p1*3  ]-pts[p0*3  ];
        float dy01 = pts[p1*3+1]-pts[p0*3+1];
        float dz01 = pts[p1*3+2]-pts[p0*3+2];
        float len2 = dx01*dx01 + dy01*dy01 + dz01*dz01;
        if (len2 < 1e-30f) len2 = 1e-30f;
        int p2 = -1;
        float best_r2 = -1.0f;
        for (int i = 0; i < n_pts; i++) {
            if (i == p0 || i == p1) continue;
            float ex = pts[i*3  ]-pts[p0*3  ];
            float ey = pts[i*3+1]-pts[p0*3+1];
            float ez = pts[i*3+2]-pts[p0*3+2];
            float t  = (ex*dx01 + ey*dy01 + ez*dz01) / len2;
            float rx = ex - t*dx01, ry = ey - t*dy01, rz = ez - t*dz01;
            float r2 = rx*rx + ry*ry + rz*rz;
            if (r2 > best_r2) { best_r2 = r2; p2 = i; }
        }

        if (p2 < 0 || best_r2 < 1e-20f) {
            // Degenerate (collinear)
            scalars[0] = -1; // signal degenerate
        } else {
            // Find p3: farthest from plane (p0,p1,p2)
            float e1x = pts[p1*3  ]-pts[p0*3  ];
            float e1y = pts[p1*3+1]-pts[p0*3+1];
            float e1z = pts[p1*3+2]-pts[p0*3+2];
            float e2x = pts[p2*3  ]-pts[p0*3  ];
            float e2y = pts[p2*3+1]-pts[p0*3+1];
            float e2z = pts[p2*3+2]-pts[p0*3+2];
            float pnx = e1y*e2z - e1z*e2y;
            float pny = e1z*e2x - e1x*e2z;
            float pnz = e1x*e2y - e1y*e2x;
            int p3 = -1;
            float best_ad = -1.0f;
            for (int i = 0; i < n_pts; i++) {
                if (i == p0 || i == p1 || i == p2) continue;
                float ex = pts[i*3  ]-pts[p0*3  ];
                float ey = pts[i*3+1]-pts[p0*3+1];
                float ez = pts[i*3+2]-pts[p0*3+2];
                float ad = fabsf(ex*pnx + ey*pny + ez*pnz);
                if (ad > best_ad) { best_ad = ad; p3 = i; }
            }

            if (p3 < 0 || best_ad < 1e-20f) {
                scalars[0] = -1; // coplanar
            } else {
                // Build 4 faces CCW (outward normals verified against centroid)
                float cx = (pts[p0*3  ]+pts[p1*3  ]+pts[p2*3  ]+pts[p3*3  ])*0.25f;
                float cy = (pts[p0*3+1]+pts[p1*3+1]+pts[p2*3+1]+pts[p3*3+1])*0.25f;
                float cz = (pts[p0*3+2]+pts[p1*3+2]+pts[p2*3+2]+pts[p3*3+2])*0.25f;

                int ff[4][3] = {{p0,p1,p2},{p0,p3,p1},{p1,p3,p2},{p0,p2,p3}};
                for (int f = 0; f < 4; f++) {
                    int a = ff[f][0], b = ff[f][1], c = ff[f][2];
                    float tnx, tny, tnz, tnd;
                    qh_set_face_plane(pts, &tnx, &tny, &tnz, &tnd, a, b, c);
                    float dot = tnx*(cx-pts[a*3  ])
                              + tny*(cy-pts[a*3+1])
                              + tnz*(cz-pts[a*3+2]);
                    if (dot > 0.0f) {
                        // Flip winding so normal points outward
                        fv0[f]=a; fv1[f]=c; fv2[f]=b;
                        qh_set_face_plane(pts, &fnx[f],&fny[f],&fnz[f],&fnd[f], a, c, b);
                    } else {
                        fv0[f]=a; fv1[f]=b; fv2[f]=c;
                        fnx[f]=tnx; fny[f]=tny; fnz[f]=tnz; fnd[f]=tnd;
                    }
                    pts_head[f] = -1;
                    pts_max_ull[f] = 0ULL;
                    fadj0[f] = fadj1[f] = fadj2[f] = -1;
                }
                scalars[0] = 4; // n_faces

                // Recompute adjacency for 4 initial faces
                for (int f = 0; f < 4; f++) {
                    for (int e = 0; e < 3; e++) {
                        int* adj_e = (e==0)?fadj0:(e==1)?fadj1:fadj2;
                        if (adj_e[f] >= 0) continue;
                        int va = (e==0)?fv0[f]:(e==1)?fv1[f]:fv2[f];
                        int vb = (e==0)?fv1[f]:(e==1)?fv2[f]:fv0[f];
                        for (int g = 0; g < 4; g++) {
                            if (g == f) continue;
                            // Check if g has the reverse edge (vb->va)
                            for (int e2 = 0; e2 < 3; e2++) {
                                int gva = (e2==0)?fv0[g]:(e2==1)?fv1[g]:fv2[g];
                                int gvb = (e2==0)?fv1[g]:(e2==1)?fv2[g]:fv0[g];
                                if (gva == vb && gvb == va) {
                                    adj_e[f] = g;
                                    int* gadj = (e2==0)?fadj0:(e2==1)?fadj1:fadj2;
                                    gadj[g] = f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __syncwarp(WARP_MASK);

    // Check for degenerate initial tetrahedron
    int n_faces = warp_bcast_i(scalars[0]);
    if (n_faces < 0) {
        *error = 2;
        return -1.0f;
    }

    // ------------------------------------------------------------------
    // Step 2: Initial conflict assignment (warp-parallel)
    // ------------------------------------------------------------------
    // Each thread handles pts[lane::32].  Each point is assigned to the
    // face with the maximum positive signed distance.

    for (int i = lane; i < n_pts; i += WARP_SIZE) {
        float px = pts[i*3  ];
        float py = pts[i*3+1];
        float pz = pts[i*3+2];
        int   best_f = -1;
        float best_d = EPS; // must exceed epsilon to be "outside"
        for (int f = 0; f < 4; f++) {
            float d = qh_face_dist(fnx[f],fny[f],fnz[f],fnd[f], px,py,pz);
            if (d > best_d) { best_d = d; best_f = f; }
        }
        if (best_f >= 0) {
            // Prepend i to face's conflict linked list
            int old = atomicExch(&pts_head[best_f], i);
            pt_next[i] = old;
            // Atomic argmax: pack (dist_bits, pt_idx) into ull
            unsigned long long key =
                ((unsigned long long)__float_as_uint(best_d) << 32) |
                (unsigned long long)(unsigned int)i;
            atomicMax(&pts_max_ull[best_f], key);
        } else {
            pt_next[i] = -1;
        }
    }
    __syncwarp(WARP_MASK);

    // Thread 0: decode pts_max_ull -> pts_max_i/d; build initial face stack
    float accumulated_vol = 0.0f;
    if (lane == 0) {
        // Compute initial volume from tetrahedron faces
        for (int f = 0; f < 4; f++) {
            accumulated_vol += signed_tet_volume(
                pts[fv0[f]*3  ], pts[fv0[f]*3+1], pts[fv0[f]*3+2],
                pts[fv1[f]*3  ], pts[fv1[f]*3+1], pts[fv1[f]*3+2],
                pts[fv2[f]*3  ], pts[fv2[f]*3+1], pts[fv2[f]*3+2]);
        }

        // Decode argmax entries
        for (int f = 0; f < 4; f++) {
            unsigned long long u = pts_max_ull[f];
            pts_max_i[f] = (int)(u & 0xffffffffULL);
            pts_max_d[f] = __uint_as_float((unsigned int)(u >> 32));
        }

        int n_stack = 0;
        for (int f = 0; f < 4; f++) {
            if (pts_head[f] >= 0)
                face_stack[n_stack++] = f;
        }
        scalars[0] = 4;       // n_faces
        scalars[1] = n_stack; // n_stack
        // Store volume in scalars as float bits
        ((float*)scalars)[4] = accumulated_vol; // reuse scalars[4] (already done with extremes)
    }
    __syncwarp(WARP_MASK);

    // ------------------------------------------------------------------
    // Main QuickHull loop
    // ------------------------------------------------------------------
    // Invariant: scalars[0]=n_faces, scalars[1]=n_stack, scalars[4..]=vol(float)

    while (1) {
        int n_stack = warp_bcast_i(scalars[1]);
        if (n_stack <= 0) break;
        n_faces = warp_bcast_i(scalars[0]);
        if (n_faces >= max_faces) break; // guard against overflow

        // Thread 0: pop a face with a non-empty conflict list
        int apex_idx = -1;
        if (lane == 0) {
            int f = face_stack[--scalars[1]];
            // If this face's list is now empty (stale entry), skip it by
            // marking apex as -1; the broadcast below will cause a continue.
            if (pts_head[f] >= 0) {
                apex_idx = pts_max_i[f];
            }
        }
        apex_idx = warp_bcast_i(apex_idx);
        __syncwarp(WARP_MASK);
        if (apex_idx < 0) continue;

        float ax = pts[apex_idx*3  ];
        float ay = pts[apex_idx*3+1];
        float az = pts[apex_idx*3+2];

        // Visibility test (warp-parallel)
        int local_any = 0;
        for (int f = lane; f < n_faces; f += WARP_SIZE) {
            float d = qh_face_dist(fnx[f],fny[f],fnz[f],fnd[f], ax,ay,az);
            visible[f] = (d > EPS) ? 1 : 0;
            if (d > EPS) local_any = 1;
        }
        __syncwarp(WARP_MASK);
        int any_vis = warp_any_i(local_any);
        if (!any_vis) continue; // apex is interior

        // Thread 0: horizon collection + dead-point gathering + topology update
        if (lane == 0) {
            int n_horizon = 0;
            int n_dead    = 0;
            float vol_delta = 0.0f;

            // --- Horizon edges ---
            for (int f = 0; f < n_faces; f++) {
                if (!visible[f]) continue;
                int fv[3] = {fv0[f], fv1[f], fv2[f]};
                int fa[3] = {fadj0[f], fadj1[f], fadj2[f]};
                for (int e = 0; e < 3; e++) {
                    int nb = fa[e];
                    if (nb >= 0 && !visible[nb]) {
                        hz_a[n_horizon]   = fv[e];
                        hz_b[n_horizon]   = fv[(e+1)%3];
                        hz_nbr[n_horizon] = nb;
                        n_horizon++;
                    }
                }
            }

            // --- Collect dead conflict points from visible faces ---
            for (int f = 0; f < n_faces; f++) {
                if (!visible[f]) continue;
                int idx = pts_head[f];
                while (idx >= 0) {
                    if (idx != apex_idx)
                        dead_pts[n_dead++] = idx;
                    int nxt = pt_next[idx];
                    pt_next[idx] = -1;
                    idx = nxt;
                }
                pts_head[f] = -1;
                pts_max_ull[f] = 0ULL;
            }
            pt_next[apex_idx] = -1;

            // --- Remove visible faces (volume delta, compaction) ---
            int dst = 0;
            for (int f = 0; f < n_faces; f++) {
                if (visible[f]) {
                    vol_delta -= signed_tet_volume(
                        pts[fv0[f]*3  ], pts[fv0[f]*3+1], pts[fv0[f]*3+2],
                        pts[fv1[f]*3  ], pts[fv1[f]*3+1], pts[fv1[f]*3+2],
                        pts[fv2[f]*3  ], pts[fv2[f]*3+1], pts[fv2[f]*3+2]);
                } else {
                    if (dst != f) {
                        fv0[dst]=fv0[f]; fv1[dst]=fv1[f]; fv2[dst]=fv2[f];
                        fnx[dst]=fnx[f]; fny[dst]=fny[f];
                        fnz[dst]=fnz[f]; fnd[dst]=fnd[f];
                        pts_head[dst]=pts_head[f];
                        pts_max_ull[dst]=pts_max_ull[f];
                        pts_max_i[dst]=pts_max_i[f];
                        pts_max_d[dst]=pts_max_d[f];
                    }
                    dst++;
                }
            }

            // --- Add new fan faces from horizon ---
            int first_new = dst;
            for (int h = 0; h < n_horizon && dst < max_faces; h++) {
                int a = hz_a[h], b = hz_b[h];
                fv0[dst]=a; fv1[dst]=b; fv2[dst]=apex_idx;
                qh_set_face_plane(pts,
                    &fnx[dst],&fny[dst],&fnz[dst],&fnd[dst], a, b, apex_idx);
                // Ensure outward orientation (centroid of the 3 verts vs normal)
                // The horizon edge (a,b) was on the boundary between visible and
                // non-visible faces.  The new face connects (a,b) to the apex.
                // We DON'T flip here because the horizon edge orientation already
                // encodes the correct winding relative to the hull interior.
                pts_head[dst] = -1;
                pts_max_ull[dst] = 0ULL;
                vol_delta += signed_tet_volume(
                    pts[a*3  ], pts[a*3+1], pts[a*3+2],
                    pts[b*3  ], pts[b*3+1], pts[b*3+2],
                    ax, ay, az);
                dst++;
            }

            n_faces = dst;
            scalars[0] = n_faces;
            scalars[2] = n_horizon;
            scalars[3] = n_dead;

            // Update accumulated volume
            float vol = ((float*)scalars)[4] + vol_delta;
            ((float*)scalars)[4] = vol;

            // Recompute adjacency for all faces (O(F²))
            for (int f = 0; f < n_faces; f++)
                fadj0[f] = fadj1[f] = fadj2[f] = -1;
            for (int f = 0; f < n_faces; f++) {
                for (int e = 0; e < 3; e++) {
                    int* adj_e_arr = (e==0)?fadj0:(e==1)?fadj1:fadj2;
                    if (adj_e_arr[f] >= 0) continue;
                    int va = (e==0)?fv0[f]:(e==1)?fv1[f]:fv2[f];
                    int vb = (e==0)?fv1[f]:(e==1)?fv2[f]:fv0[f];
                    for (int g = 0; g < n_faces; g++) {
                        if (g == f) continue;
                        for (int e2 = 0; e2 < 3; e2++) {
                            int gva = (e2==0)?fv0[g]:(e2==1)?fv1[g]:fv2[g];
                            int gvb = (e2==0)?fv1[g]:(e2==1)?fv2[g]:fv0[g];
                            if (gva == vb && gvb == va) {
                                adj_e_arr[f] = g;
                                int* gadj = (e2==0)?fadj0:(e2==1)?fadj1:fadj2;
                                gadj[g] = f;
                            }
                        }
                    }
                }
            }

            // Initialize pts_max_ull for new faces (already set to 0 above)
        }
        __syncwarp(WARP_MASK);

        // Redistribution of dead conflict points (warp-parallel)
        int n_dead   = warp_bcast_i(scalars[3]);
        n_faces = warp_bcast_i(scalars[0]);

        for (int dp = lane; dp < n_dead; dp += WARP_SIZE) {
            int pt_i = dead_pts[dp];
            float px = pts[pt_i*3  ];
            float py = pts[pt_i*3+1];
            float pz = pts[pt_i*3+2];
            int   best_f = -1;
            float best_d = EPS;
            // Check ALL faces: with single-face assignment, a dead point may
            // still be above an old non-visible face (it was assigned to the
            // face with maximum positive distance, not all faces it's above).
            for (int f = 0; f < n_faces; f++) {
                float d = qh_face_dist(fnx[f],fny[f],fnz[f],fnd[f], px,py,pz);
                if (d > best_d) { best_d = d; best_f = f; }
            }
            if (best_f >= 0) {
                int old = atomicExch(&pts_head[best_f], pt_i);
                pt_next[pt_i] = old;
                unsigned long long key =
                    ((unsigned long long)__float_as_uint(best_d) << 32) |
                    (unsigned long long)(unsigned int)pt_i;
                atomicMax(&pts_max_ull[best_f], key);
            }
            // else: point is inside hull, discard
        }
        __syncwarp(WARP_MASK);

        // Thread 0: decode pts_max_ull for all faces and rebuild the face stack.
        // We must scan all faces (not just new ones) because dead points may have
        // been assigned to old non-visible faces during redistribution above.
        if (lane == 0) {
            int n_stack = 0;
            for (int f = 0; f < n_faces; f++) {
                unsigned long long u = pts_max_ull[f];
                pts_max_i[f] = (int)(u & 0xffffffffULL);
                pts_max_d[f] = __uint_as_float((unsigned int)(u >> 32));
                if (pts_head[f] >= 0 && n_stack < max_faces)
                    face_stack[n_stack++] = f;
            }
            scalars[1] = n_stack;
        }
        __syncwarp(WARP_MASK);
    }

    // Return absolute volume
    float vol = warp_bcast_f(((float*)scalars)[4]);
    return fabsf(vol);
}

// ============================================================================
// Algorithm 2: hull_dandc_warp
// ============================================================================
//
// Preparata-Hong bottom-up divide-and-conquer convex hull.
// Points are sorted along the longest AABB axis and merged pairwise.
// The half-edge graph is allocated entirely in pool scratch.
//
// All 32 threads must call with the same arguments.
// Returns hull volume (>= 0) or -1.0f on error.

// ---------------------------------------------------------------------------
// Half-edge data structure helpers (SOA in pool memory, thread 0 only)
// ---------------------------------------------------------------------------

// Edge e goes from source vertex src(e) to target vertex e_target[e].
// src(e) = e_target[e_reverse[e]]  (implicit)
// Ring around source vertex: e, e_next[e], e_next[e_next[e]], ... (CW)
// Face containing e (left of e): e -> face_next(e) -> face_next(...) until e
//   where face_next(e) = e_next[e_reverse[e]]

// Allocate a new edge slot from the free list; returns -1 on error.
__device__ inline int dnc_alloc_edge(int* e_free, int* n_free, int* n_alloc, int max_e) {
    if (*n_free > 0) {
        return e_free[--(*n_free)];
    }
    if (*n_alloc >= max_e) return -1; // out of slots
    return (*n_alloc)++;
}

// Link edge e into the CW ring of vertex v after edge e_after.
// If e_after < 0 (empty ring), start a singleton ring.
__device__ inline void dnc_link_after(int* e_next, int* e_prev, int* v_edge,
                                       int v, int e, int e_after) {
    if (e_after < 0) {
        // Singleton
        e_next[e] = e;
        e_prev[e] = e;
        v_edge[v] = e;
    } else {
        int e_before = e_next[e_after];
        e_next[e_after] = e;
        e_prev[e]       = e_after;
        e_next[e]       = e_before;
        e_prev[e_before]= e;
    }
}

// Unlink edge e from the CW ring of its source vertex.
// Does NOT update v_edge — caller must do that if needed.
__device__ inline void dnc_unlink(int* e_next, int* e_prev, int e) {
    int en = e_next[e];
    int ep = e_prev[e];
    e_next[ep] = en;
    e_prev[en] = ep;
    e_next[e]  = -1;
    e_prev[e]  = -1;
}

// Create a new directed edge e: src -> tgt.
// Links e after e_after_src in src's ring (-1 = empty ring).
// Links reverse r after e_after_tgt in tgt's ring (-1 = empty ring).
// Returns edge index or -1 on failure.
__device__ inline int dnc_make_edge(
    int* e_target, int* e_reverse, int* e_next, int* e_prev, int* e_copy,
    int* v_edge,
    int* e_free, int* n_free, int* n_alloc, int max_e,
    int src, int tgt, int stamp,
    int e_after_src, int e_after_tgt)
{
    int e = dnc_alloc_edge(e_free, n_free, n_alloc, max_e);
    int r = dnc_alloc_edge(e_free, n_free, n_alloc, max_e);
    if (e < 0 || r < 0) return -1;

    e_target[e] = tgt;
    e_target[r] = src;
    e_reverse[e] = r;
    e_reverse[r] = e;
    e_copy[e]    = stamp;
    e_copy[r]    = stamp;

    dnc_link_after(e_next, e_prev, v_edge, src, e, e_after_src);
    dnc_link_after(e_next, e_prev, v_edge, tgt, r, e_after_tgt);
    return e;
}

// Delete edge e (and its reverse) and return slots to free list.
__device__ inline void dnc_delete_edge(
    int* e_target, int* e_reverse, int* e_next, int* e_prev,
    int* v_edge,
    int* e_free, int* n_free,
    int e)
{
    int r = e_reverse[e];
    int src = e_target[r];
    int tgt = e_target[e];

    // Fix v_edge pointers before unlinking
    if (v_edge[src] == e) {
        int nxt = e_next[e];
        v_edge[src] = (nxt != e) ? nxt : -1;
    }
    if (v_edge[tgt] == r) {
        int nxt = e_next[r];
        v_edge[tgt] = (nxt != r) ? nxt : -1;
    }

    dnc_unlink(e_next, e_prev, e);
    dnc_unlink(e_next, e_prev, r);

    e_target[e] = -1;
    e_target[r] = -1;
    e_free[(*n_free)++] = e;
    e_free[(*n_free)++] = r;
}

// face_next(e): next half-edge on the same face.
// In this half-edge convention: face_next(e) = e_next[e_reverse[e]]
__device__ inline int dnc_face_next(const int* e_next, const int* e_reverse, int e) {
    return e_next[e_reverse[e]];
}

// Source vertex of edge e: e_target[e_reverse[e]]
__device__ inline int dnc_src(const int* e_target, const int* e_reverse, int e) {
    return e_target[e_reverse[e]];
}

// ---------------------------------------------------------------------------
// 2D merge: find upper common tangent (max-axis projection)
// ---------------------------------------------------------------------------
// Projects onto (med_axis, max_axis) plane and finds the lower tangent point
// that forms the first bridge between hull h0 (left) and hull h1 (right).
//
// h_head[h]: linked vertex list head (via v_next)
// sorted_pts[v*3 + axis]: coordinate of vertex v along axis
// max_axis, med_axis: projection axes
//
// Returns (c0_out, c1_out) — the initial bridge vertices.

__device__ void dnc_find_bridge(
    const int* v_next, const float* sorted_pts,
    int max_ax, int med_ax,
    int h0_head, int h1_head,
    int* c0_out, int* c1_out)
{
    // Find rightmost vertex of h0 and leftmost vertex of h1 in max_axis
    // (v_next lists are -1 terminated)
    int c0 = h0_head;
    for (int cur = h0_head; cur >= 0; cur = v_next[cur]) {
        if (sorted_pts[cur*3+max_ax] > sorted_pts[c0*3+max_ax]) c0 = cur;
    }
    int c1 = h1_head;
    for (int cur = h1_head; cur >= 0; cur = v_next[cur]) {
        if (sorted_pts[cur*3+max_ax] < sorted_pts[c1*3+max_ax]) c1 = cur;
    }

    // Iteratively improve to lower tangent in 2D
    // The lower tangent has h0 and h1 entirely above the bridge line.
    bool changed = true;
    while (changed) {
        changed = false;

        // Advance c0: step to neighbor that makes the line slope decrease
        // (in the 2D projection: med_ax vs max_ax)
        // For lower tangent: c0 is the rightmost vertex of h0 such that
        //   no vertex of h0 is below the line (c0, c1).
        // We check adjacent vertices via e_next ring... but we don't have
        // edge info here.  Use a full scan instead (N² but N is small).
        // Advance c0: find vertex in h0 minimizing (med - slope * max)
        {
            float dx = sorted_pts[c1*3+max_ax] - sorted_pts[c0*3+max_ax];
            float dy = sorted_pts[c1*3+med_ax]  - sorted_pts[c0*3+med_ax];
            if (fabsf(dx) > 1e-12f) {
                float slope = dy / dx;
                int   best  = c0;
                float best_v = sorted_pts[c0*3+med_ax] - slope*sorted_pts[c0*3+max_ax];
                for (int vv = h0_head; vv >= 0; vv = v_next[vv]) {
                    float val = sorted_pts[vv*3+med_ax] - slope*sorted_pts[vv*3+max_ax];
                    if (val < best_v) { best_v = val; best = vv; }
                }
                if (best != c0) { c0 = best; changed = true; }
            }
        }
        // Advance c1: find vertex in h1 minimizing (med - slope * max)
        {
            float dx = sorted_pts[c1*3+max_ax] - sorted_pts[c0*3+max_ax];
            float dy = sorted_pts[c1*3+med_ax]  - sorted_pts[c0*3+med_ax];
            if (fabsf(dx) > 1e-12f) {
                float slope = dy / dx;
                int   best  = c1;
                float best_v = sorted_pts[c1*3+med_ax] - slope*sorted_pts[c1*3+max_ax];
                for (int vv = h1_head; vv >= 0; vv = v_next[vv]) {
                    float val = sorted_pts[vv*3+med_ax] - slope*sorted_pts[vv*3+max_ax];
                    if (val < best_v) { best_v = val; best = vv; }
                }
                if (best != c1) { c1 = best; changed = true; }
            }
        }
    }

    *c0_out = c0;
    *c1_out = c1;
}

// ---------------------------------------------------------------------------
// 3D Preparata-Hong merge (thread 0 only)
// ---------------------------------------------------------------------------
// Merges hull h0 and hull h1 into a single hull by walking the seam.
// Returns the head of the merged vertex list.

__device__ int dnc_merge_hulls(
    float*       sorted_pts,   // [n_pts * 3]
    int          n_pts,
    int          max_ax,       // sort axis (longest extent)
    int          med_ax,       // medium axis
    int          min_ax,       // shortest axis
    // Half-edge arrays (all in pool)
    int*         v_edge,       // [n_pts] first outgoing edge from vertex
    int*         v_next,       // [n_pts] vertex linked-list next (within hull)
    int*         e_target,     // [max_e]
    int*         e_reverse,    // [max_e]
    int*         e_next,       // [max_e]
    int*         e_prev,       // [max_e]
    int*         e_copy,       // [max_e] merge stamp (for new edges)
    int*         e_free,       // free list
    int*         n_free,       // free count
    int*         n_alloc,      // allocated count
    int          max_e,
    int          h0_head,      // head of h0 vertex list
    int          h1_head,      // head of h1 vertex list
    int          stamp,        // current merge stamp
    int*         out_error)
{
    // Find initial bridge
    int c0, c1;
    dnc_find_bridge(v_next, sorted_pts, max_ax, med_ax, h0_head, h1_head, &c0, &c1);

    // The bridge vector and reference plane for findMaxAngle
    // s = c1 - c0
    float sx = sorted_pts[c1*3  ] - sorted_pts[c0*3  ];
    float sy = sorted_pts[c1*3+1] - sorted_pts[c0*3+1];
    float sz = sorted_pts[c1*3+2] - sorted_pts[c0*3+2];

    // prevPoint = a point "below" the first bridge.
    // Use the lower-tangent reference: a point in the direction -max_ax
    // from c0.  We use c0 itself shifted slightly as the initial prevPoint;
    // the algorithm handles the first step specially.
    // For simplicity, use the approach from Preparata-Hong:
    //   prevPoint = c0 - (0, 0, large) (below the plane)
    // i.e., a virtual point below both hulls.
    float rpx = sorted_pts[c0*3  ];
    float rpy = sorted_pts[c0*3+1];
    float rpz = sorted_pts[c0*3+2] - 1000.0f; // virtual prev point below

    // Current bridge: c0 (h0) -- c1 (h1)
    // Walk the seam by advancing the bridge.

    // Seam has at most n_pts edges total.
    int max_iters = n_pts + 4;
    int iter = 0;

    // Save initial bridge endpoints to detect seam completion.
    int c0_init = c0;
    int c1_init = c1;

    // Create the first bridge edge c0 -> c1
    int bridge_first = dnc_make_edge(e_target, e_reverse, e_next, e_prev, e_copy,
                                      v_edge, e_free, n_free, n_alloc, max_e,
                                      c0, c1, stamp,
                                      (v_edge[c0] >= 0) ? e_prev[v_edge[c0]] : -1,
                                      (v_edge[c1] >= 0) ? e_prev[v_edge[c1]] : -1);
    if (bridge_first < 0) { *out_error = 1; return h0_head; }

    int prev_bridge = bridge_first;

    while (iter++ < max_iters) {
        // r = c0 - prevPoint (reference direction for dihedral angle)
        float rx = sorted_pts[c0*3  ] - rpx;
        float ry = sorted_pts[c0*3+1] - rpy;
        float rz = sorted_pts[c0*3+2] - rpz;

        // rxs = r x s
        float rxsx = ry*sz - rz*sy;
        float rxsy = rz*sx - rx*sz;
        float rxsz = rx*sy - ry*sx;

        // sxrxs = s x (r x s)
        float sxrxsx = sy*rxsz - sz*rxsy;
        float sxrxsy = sz*rxsx - sx*rxsz;
        float sxrxsz = sx*rxsy - sy*rxsx;

        // findMaxAngle for c0 side (h0 vertices)
        float min_cot0 = 1e30f;
        int   best_e0  = -1;

        if (v_edge[c0] >= 0) {
            int e_start = v_edge[c0];
            int e_cur   = e_start;
            do {
                // Skip edges created in this merge (stamp == stamp)
                if (e_copy[e_cur] < stamp) {
                    int w = e_target[e_cur];
                    float dirx = sorted_pts[w*3  ] - sorted_pts[c0*3  ];
                    float diry = sorted_pts[w*3+1] - sorted_pts[c0*3+1];
                    float dirz = sorted_pts[w*3+2] - sorted_pts[c0*3+2];
                    float num = dirx*sxrxsx + diry*sxrxsy + dirz*sxrxsz;
                    float den = dirx*rxsx   + diry*rxsy   + dirz*rxsz;
                    float cot;
                    if (fabsf(den) > 1e-30f) cot = num/den;
                    else cot = (num >= 0.0f) ? 1e30f : -1e30f;
                    if (cot < min_cot0) { min_cot0 = cot; best_e0 = e_cur; }
                }
                e_cur = e_next[e_cur];
            } while (e_cur != e_start);
        }

        // findMaxAngle for c1 side (h1 vertices)
        // Mirror: use -s as the bridge direction for c1's perspective
        float min_cot1 = 1e30f;
        int   best_e1  = -1;
        {
            // For c1: we look at edges leaving c1 toward h1
            // The reference: r1 = c1 - prevPoint (from c1's perspective)
            float r1x = sorted_pts[c1*3  ] - rpx;
            float r1y = sorted_pts[c1*3+1] - rpy;
            float r1z = sorted_pts[c1*3+2] - rpz;
            // The bridge direction from c1 is -s (toward c0)
            float nsx = -sx, nsy = -sy, nsz = -sz;
            // r1 x (-s)
            float r1xnsx = r1y*nsz - r1z*nsy;
            float r1xnsy = r1z*nsx - r1x*nsz;
            float r1xnsz = r1x*nsy - r1y*nsx;
            // (-s) x (r1 x (-s))
            float sxr1xnsx = nsy*r1xnsz - nsz*r1xnsy;
            float sxr1xnsy = nsz*r1xnsx - nsx*r1xnsz;
            float sxr1xnsz = nsx*r1xnsy - nsy*r1xnsx;

            if (v_edge[c1] >= 0) {
                int e_start = v_edge[c1];
                int e_cur   = e_start;
                do {
                    if (e_copy[e_cur] < stamp) {
                        int w = e_target[e_cur];
                        float dirx = sorted_pts[w*3  ] - sorted_pts[c1*3  ];
                        float diry = sorted_pts[w*3+1] - sorted_pts[c1*3+1];
                        float dirz = sorted_pts[w*3+2] - sorted_pts[c1*3+2];
                        float num = dirx*sxr1xnsx + diry*sxr1xnsy + dirz*sxr1xnsz;
                        float den = dirx*r1xnsx   + diry*r1xnsy   + dirz*r1xnsz;
                        float cot;
                        if (fabsf(den) > 1e-30f) cot = num/den;
                        else cot = (num >= 0.0f) ? 1e30f : -1e30f;
                        if (cot < min_cot1) { min_cot1 = cot; best_e1 = e_cur; }
                    }
                    e_cur = e_next[e_cur];
                } while (e_cur != e_start);
            }
        }

        // Termination: no advance possible
        if (best_e0 < 0 && best_e1 < 0) break;
        if (min_cot0 >= 1e29f && min_cot1 >= 1e29f) break;

        // Advance the bridge
        // We advance whichever side has the smaller cotangent (larger angle).
        bool advance_c0 = (best_e0 >= 0 && min_cot0 <= min_cot1);
        bool advance_c1 = (best_e1 >= 0 && !advance_c0);

        // Handle coplanar (both equal): advance both sides
        bool coplanar = (best_e0 >= 0 && best_e1 >= 0 &&
                         fabsf(min_cot0 - min_cot1) < 1e-8f);

        float new_rpx, new_rpy, new_rpz;

        if (advance_c0 || coplanar) {
            int new_c0 = e_target[best_e0];
            new_rpx = sorted_pts[c0*3  ];
            new_rpy = sorted_pts[c0*3+1];
            new_rpz = sorted_pts[c0*3+2];
            // New bridge: new_c0 -> c1
            // Before creating the new edge, we may need to handle the
            // triangle (c0, new_c0, c1) by deleting the crossing edge if any.
            // Simple approach: just add bridge edge new_c0 -> c1
            if (!coplanar || advance_c0) {
                int new_br = dnc_make_edge(
                    e_target, e_reverse, e_next, e_prev, e_copy,
                    v_edge, e_free, n_free, n_alloc, max_e,
                    new_c0, c1, stamp,
                    (v_edge[new_c0] >= 0) ? e_prev[v_edge[new_c0]] : -1,
                    e_reverse[prev_bridge]);
                if (new_br < 0) { *out_error = 1; break; }
                prev_bridge = new_br;
            }
            sx = sorted_pts[c1*3  ] - sorted_pts[new_c0*3  ];
            sy = sorted_pts[c1*3+1] - sorted_pts[new_c0*3+1];
            sz = sorted_pts[c1*3+2] - sorted_pts[new_c0*3+2];
            rpx = new_rpx; rpy = new_rpy; rpz = new_rpz;
            c0 = new_c0;
        }

        if (advance_c1 || coplanar) {
            int new_c1 = e_target[best_e1];
            if (!coplanar || advance_c1) {
                new_rpx = sorted_pts[c1*3  ];
                new_rpy = sorted_pts[c1*3+1];
                new_rpz = sorted_pts[c1*3+2];
            }
            // New bridge: c0 -> new_c1
            int new_br = dnc_make_edge(
                e_target, e_reverse, e_next, e_prev, e_copy,
                v_edge, e_free, n_free, n_alloc, max_e,
                c0, new_c1, stamp,
                e_prev[prev_bridge],
                (v_edge[new_c1] >= 0) ? e_prev[v_edge[new_c1]] : -1);
            if (new_br < 0) { *out_error = 1; break; }
            prev_bridge = new_br;
            sx = sorted_pts[new_c1*3  ] - sorted_pts[c0*3  ];
            sy = sorted_pts[new_c1*3+1] - sorted_pts[c0*3+1];
            sz = sorted_pts[new_c1*3+2] - sorted_pts[c0*3+2];
            if (!coplanar || advance_c1) {
                rpx = new_rpx; rpy = new_rpy; rpz = new_rpz;
            }
            c1 = new_c1;
        }

        // Termination: bridge has returned to its initial position — seam complete.
        if (c0 == c0_init && c1 == c1_init) break;
    }

    // Merge vertex lists: concatenate h1's list into h0's list (-1 terminated).
    if (h1_head >= 0) {
        // Find tails of the two linear lists
        int tail0 = h0_head;
        while (v_next[tail0] >= 0) tail0 = v_next[tail0];
        int tail1 = h1_head;
        while (v_next[tail1] >= 0) tail1 = v_next[tail1];
        // Splice: ...tail0 -> h1_head ... tail1 -> -1
        v_next[tail0] = h1_head;
        v_next[tail1] = -1;
    }

    return h0_head;
}

// ---------------------------------------------------------------------------
// hull_dandc_warp implementation
// ---------------------------------------------------------------------------

__device__ float hull_dandc_warp(
    const float* pts,
    int          n_pts,
    int          lane,
    WarpPool*    pool,
    int*         error)
{
    *error = 0;

    if (n_pts < 4) {
        *error = 2;
        return -1.0f;
    }

    // ------------------------------------------------------------------
    // Step 1: Find sort axis (warp-parallel AABB)
    // ------------------------------------------------------------------
    float lo[3] = {1e30f, 1e30f, 1e30f};
    float hi[3] = {-1e30f, -1e30f, -1e30f};

    for (int i = lane; i < n_pts; i += WARP_SIZE) {
        float x = pts[i*3  ], y = pts[i*3+1], z = pts[i*3+2];
        lo[0] = fminf(lo[0], x); hi[0] = fmaxf(hi[0], x);
        lo[1] = fminf(lo[1], y); hi[1] = fmaxf(hi[1], y);
        lo[2] = fminf(lo[2], z); hi[2] = fmaxf(hi[2], z);
    }

    // Reduce per-axis
    for (int k = 0; k < 3; k++) {
        for (int off = 16; off > 0; off >>= 1) {
            lo[k] = fminf(lo[k], __shfl_xor_sync(WARP_MASK, lo[k], off));
            hi[k] = fmaxf(hi[k], __shfl_xor_sync(WARP_MASK, hi[k], off));
        }
    }
    __syncwarp(WARP_MASK);

    // Thread 0 picks axes
    int max_ax = 0, med_ax = 1, min_ax = 2;
    if (lane == 0) {
        float ext[3] = {hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]};
        // Sort axes by descending extent
        for (int a = 0; a < 3; a++) {
            for (int b = a+1; b < 3; b++) {
                if (ext[b] > ext[a]) {
                    float tmp = ext[a]; ext[a] = ext[b]; ext[b] = tmp;
                    int   ti  = a;   a   = b;   b   = ti;
                    // Hmm, can't swap loop vars; use index sort instead
                }
            }
        }
        // Redo: explicit index sort
        int order[3] = {0, 1, 2};
        float exts[3] = {hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]};
        // Insertion sort descending
        for (int a = 1; a < 3; a++) {
            for (int b = a; b > 0 && exts[order[b]] > exts[order[b-1]]; b--) {
                int t = order[b]; order[b] = order[b-1]; order[b-1] = t;
            }
        }
        max_ax = order[0];
        med_ax = order[1];
        min_ax = order[2];
    }
    max_ax = warp_bcast_i(max_ax);
    med_ax = warp_bcast_i(med_ax);
    min_ax = warp_bcast_i(min_ax);

    // ------------------------------------------------------------------
    // Step 2: Allocate all scratch arrays from pool
    // ------------------------------------------------------------------
    int max_e = n_pts * 8 + 32; // upper bound on half-edges (O(n log n) with proper termination)

    // Sorted points (copy + sort)
    float* sorted_pts = (float*)warp_pool_alloc(pool, n_pts * 3 * (int)sizeof(float), lane);
    float* temp_pts   = (float*)warp_pool_alloc(pool, n_pts * 3 * (int)sizeof(float), lane);
    int*   sort_idx   = (int*)  warp_pool_alloc(pool, n_pts     * (int)sizeof(int),   lane);
    int*   temp_idx   = (int*)  warp_pool_alloc(pool, n_pts     * (int)sizeof(int),   lane);

    // Vertex arrays
    int* v_edge = (int*)warp_pool_alloc(pool, n_pts * (int)sizeof(int), lane);
    int* v_next = (int*)warp_pool_alloc(pool, n_pts * (int)sizeof(int), lane);

    // Half-edge arrays
    int* e_target  = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);
    int* e_reverse = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);
    int* e_next    = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);
    int* e_prev    = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);
    int* e_copy    = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);

    // Free list
    int* e_free = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);

    // Per-merge hull head array: n_hulls at each level <= n_pts
    int* hull_heads = (int*)warp_pool_alloc(pool, n_pts * (int)sizeof(int), lane);

    // Visited flags for volume traversal
    int* e_visited = (int*)warp_pool_alloc(pool, max_e * (int)sizeof(int), lane);

    // Scalars: [n_free, n_alloc, merge_stamp, out_error_flag, n_hulls]
    int* scalars = (int*)warp_pool_alloc(pool, 8 * (int)sizeof(int), lane);

    __syncwarp(WARP_MASK);

    if (pool->error) {
        *error = 1;
        return -1.0f;
    }

    // ------------------------------------------------------------------
    // Step 3: Copy and sort points (thread 0 — merge sort O(N log N))
    // ------------------------------------------------------------------
    if (lane == 0) {
        // Copy pts into sorted_pts; initialize sort_idx
        for (int i = 0; i < n_pts; i++) {
            sorted_pts[i*3  ] = pts[i*3  ];
            sorted_pts[i*3+1] = pts[i*3+1];
            sorted_pts[i*3+2] = pts[i*3+2];
            sort_idx[i] = i;
        }

        // Bottom-up merge sort by sorted_pts[i*3 + max_ax]
        for (int width = 1; width < n_pts; width *= 2) {
            for (int lo_i = 0; lo_i < n_pts; lo_i += 2*width) {
                int mid  = lo_i + width;       if (mid  > n_pts) mid  = n_pts;
                int hi_i = lo_i + 2*width;     if (hi_i > n_pts) hi_i = n_pts;
                // Merge sorted_pts[lo_i..mid) and [mid..hi_i) into temp
                int i = lo_i, j = mid, k = lo_i;
                while (i < mid && j < hi_i) {
                    if (sorted_pts[i*3+max_ax] <= sorted_pts[j*3+max_ax]) {
                        temp_pts[k*3  ]=sorted_pts[i*3  ];
                        temp_pts[k*3+1]=sorted_pts[i*3+1];
                        temp_pts[k*3+2]=sorted_pts[i*3+2];
                        temp_idx[k] = sort_idx[i];
                        i++; k++;
                    } else {
                        temp_pts[k*3  ]=sorted_pts[j*3  ];
                        temp_pts[k*3+1]=sorted_pts[j*3+1];
                        temp_pts[k*3+2]=sorted_pts[j*3+2];
                        temp_idx[k] = sort_idx[j];
                        j++; k++;
                    }
                }
                while (i < mid) {
                    temp_pts[k*3  ]=sorted_pts[i*3  ];
                    temp_pts[k*3+1]=sorted_pts[i*3+1];
                    temp_pts[k*3+2]=sorted_pts[i*3+2];
                    temp_idx[k] = sort_idx[i];
                    i++; k++;
                }
                while (j < hi_i) {
                    temp_pts[k*3  ]=sorted_pts[j*3  ];
                    temp_pts[k*3+1]=sorted_pts[j*3+1];
                    temp_pts[k*3+2]=sorted_pts[j*3+2];
                    temp_idx[k] = sort_idx[j];
                    j++; k++;
                }
            }
            // Swap pointers (just copy back)
            for (int i = 0; i < n_pts; i++) {
                sorted_pts[i*3  ] = temp_pts[i*3  ];
                sorted_pts[i*3+1] = temp_pts[i*3+1];
                sorted_pts[i*3+2] = temp_pts[i*3+2];
                sort_idx[i] = temp_idx[i];
            }
        }

        // Initialize half-edge data
        for (int i = 0; i < max_e; i++) {
            e_target[i]  = -1;
            e_reverse[i] = -1;
            e_next[i]    = -1;
            e_prev[i]    = -1;
            e_copy[i]    = -1;
        }
        for (int i = 0; i < n_pts; i++) {
            v_edge[i] = -1;
            v_next[i] = -1;
        }

        scalars[0] = 0;  // n_free
        scalars[1] = 0;  // n_alloc
        scalars[2] = 0;  // merge_stamp
        scalars[3] = 0;  // error_flag
    }
    __syncwarp(WARP_MASK);

    // ------------------------------------------------------------------
    // Step 4: Initialize level-0 hulls (one vertex each)
    // ------------------------------------------------------------------
    if (lane == 0) {
        for (int i = 0; i < n_pts; i++) {
            hull_heads[i] = i;
            v_next[i] = -1;
            v_edge[i] = -1;
        }
        scalars[4] = n_pts; // n_hulls
    }
    __syncwarp(WARP_MASK);

    // ------------------------------------------------------------------
    // Step 5: Bottom-up pairwise merge
    // ------------------------------------------------------------------
    if (lane == 0) {
        int n_hulls  = n_pts;
        int n_free   = 0;
        int n_alloc  = 0;
        int stamp    = 0;
        int err_flag = 0;

        while (n_hulls > 1 && !err_flag) {
            int new_n = 0;
            for (int h = 0; h + 1 < n_hulls; h += 2) {
                stamp++;
                int merged = dnc_merge_hulls(
                    sorted_pts, n_pts,
                    max_ax, med_ax, min_ax,
                    v_edge, v_next,
                    e_target, e_reverse, e_next, e_prev, e_copy,
                    e_free, &n_free, &n_alloc, max_e,
                    hull_heads[h], hull_heads[h+1],
                    stamp, &err_flag);
                hull_heads[new_n++] = merged;
            }
            // If odd number of hulls, carry the last one unchanged
            if (n_hulls & 1)
                hull_heads[new_n++] = hull_heads[n_hulls - 1];
            n_hulls = new_n;
        }
        scalars[3] = err_flag;
        scalars[1] = n_alloc;
    }
    __syncwarp(WARP_MASK);

    if (warp_bcast_i(scalars[3])) {
        *error = 1;
        return -1.0f;
    }

    // ------------------------------------------------------------------
    // Step 6: Compute volume from half-edge graph (thread 0)
    // ------------------------------------------------------------------
    float total_vol = 0.0f;
    if (lane == 0) {
        int n_alloc = scalars[1];

        // Clear visited flags
        for (int e = 0; e < n_alloc; e++) e_visited[e] = 0;

        // Traverse all faces: each face is identified by a canonical edge
        // (the minimum-index edge in its face cycle that has not been visited).
        for (int e0 = 0; e0 < n_alloc; e0++) {
            if (e_target[e0] < 0) continue; // free slot
            if (e_visited[e0])    continue;

            // Trace the face cycle starting at e0
            // face_next(e) = e_next[e_reverse[e]]
            int face_verts[64];
            int nfv = 0;
            int e_cur = e0;
            do {
                if (e_reverse[e_cur] < 0 || e_reverse[e_cur] >= n_alloc) break;
                int src_v = e_target[e_reverse[e_cur]];
                if (nfv < 62) face_verts[nfv++] = src_v;
                e_visited[e_cur] = 1;
                int rev = e_reverse[e_cur];
                if (e_next[rev] < 0 || e_next[rev] >= n_alloc) break;
                e_cur = e_next[rev];
                if (nfv > 62) break; // safety
            } while (e_cur != e0);

            if (nfv < 3) continue;

            // Fan-triangulate from face_verts[0]
            float* p0 = sorted_pts + face_verts[0]*3;
            for (int k = 1; k < nfv - 1; k++) {
                float* p1 = sorted_pts + face_verts[k  ]*3;
                float* p2 = sorted_pts + face_verts[k+1]*3;
                total_vol += signed_tet_volume(
                    p0[0], p0[1], p0[2],
                    p1[0], p1[1], p1[2],
                    p2[0], p2[1], p2[2]);
            }
        }
    }
    __syncwarp(WARP_MASK);

    total_vol = warp_bcast_f(total_vol);
    return fabsf(total_vol);
}

#endif // HULL_WARP_CUH
