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
// Algorithm 2: hull_dandc_warp — Preparata-Hong D&C convex hull
// ============================================================================
//
// Ported from btConvexHullComputer by Ole Kniemeyer (MAXON, zlib license).
// Adapted for GPU: array-based half-edge structure, iterative bottom-up merge,
// all scratch in WarpPool. Uses int32 coordinates with exact int64/int128
// predicates for robustness.
//
// Points are sorted along the longest AABB axis and merged pairwise.
// The half-edge graph is allocated entirely in pool scratch.
//
// All 32 threads must call with the same arguments.
// Returns hull volume (>= 0) or -1.0f on error.

// ---------------------------------------------------------------------------
// Int128 — 128-bit signed integer for exact geometric predicates
// ---------------------------------------------------------------------------

struct GpuInt128 {
    unsigned long long low;
    unsigned long long high;
};

__device__ inline GpuInt128 int128_umul(unsigned long long a, unsigned long long b) {
    unsigned long long a_lo = a & 0xffffffffULL, a_hi = a >> 32;
    unsigned long long b_lo = b & 0xffffffffULL, b_hi = b >> 32;
    unsigned long long p00 = a_lo * b_lo;
    unsigned long long p01 = a_lo * b_hi;
    unsigned long long p10 = a_hi * b_lo;
    unsigned long long p11 = a_hi * b_hi;
    unsigned long long mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    GpuInt128 r;
    r.low = (p00 & 0xffffffffULL) | ((mid & 0xffffffffULL) << 32);
    r.high = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

__device__ inline int int128_ucmp(GpuInt128 a, GpuInt128 b) {
    if (a.high < b.high) return -1;
    if (a.high > b.high) return 1;
    if (a.low < b.low) return -1;
    if (a.low > b.low) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Rational64 — exact rational comparison for cotangent
// ---------------------------------------------------------------------------

struct GpuRational64 {
    unsigned long long numerator;
    unsigned long long denominator;
    int sign;
};

__device__ inline GpuRational64 rational64_make(long long num, long long den) {
    GpuRational64 r;
    r.sign = (num > 0) ? 1 : (num < 0) ? -1 : 0;
    r.numerator = (r.sign >= 0) ? (unsigned long long)num : (unsigned long long)(-num);
    if (den > 0) { r.denominator = (unsigned long long)den; }
    else if (den < 0) { r.sign = -r.sign; r.denominator = (unsigned long long)(-den); }
    else { r.denominator = 0; }
    return r;
}

__device__ inline bool rational64_is_nan(GpuRational64 r) {
    return (r.sign == 0) && (r.denominator == 0);
}

__device__ inline bool rational64_is_neg_inf(GpuRational64 r) {
    return (r.sign < 0) && (r.denominator == 0);
}

__device__ inline int rational64_compare(GpuRational64 a, GpuRational64 b) {
    if (a.sign != b.sign) return a.sign - b.sign;
    if (a.sign == 0) return 0;
    return a.sign * int128_ucmp(int128_umul(a.numerator, b.denominator),
                                 int128_umul(a.denominator, b.numerator));
}

// ---------------------------------------------------------------------------
// Half-edge data structure helpers (SOA in pool memory, thread 0 only)
// ---------------------------------------------------------------------------

// Edge e goes from source vertex src(e) to target vertex e_target[e].
// src(e) = e_target[e_reverse[e]]  (implicit)
// Ring around source vertex: e, e_next[e], ... (circular doubly-linked list)

__device__ inline int dnc_alloc_edge(int* e_free, int* n_free, int* n_alloc, int max_e) {
    if (*n_free > 0) return e_free[--(*n_free)];
    if (*n_alloc >= max_e) return -1;
    return (*n_alloc)++;
}

// Link: a->next = b, b->prev = a  (Bullet's Edge::link)
__device__ inline void dnc_edge_link(int* e_next, int* e_prev, int a, int b) {
    e_next[a] = b;
    e_prev[b] = a;
}

// Create edge pair (no ring linking — caller links into rings).
// Returns forward edge index; reverse is e_reverse[returned].
__device__ inline int dnc_new_edge_pair(
    int* e_target, int* e_reverse, int* e_next, int* e_prev, int* e_copy,
    int* e_free, int* n_free, int* n_alloc, int max_e,
    int from, int to, int stamp)
{
    int e = dnc_alloc_edge(e_free, n_free, n_alloc, max_e);
    int r = dnc_alloc_edge(e_free, n_free, n_alloc, max_e);
    if (e < 0 || r < 0) return -1;
    e_target[e] = to;   e_target[r] = from;
    e_reverse[e] = r;   e_reverse[r] = e;
    e_copy[e] = stamp;  e_copy[r] = stamp;
    e_next[e] = -1; e_prev[e] = -1;
    e_next[r] = -1; e_prev[r] = -1;
    return e;
}

// Remove edge pair (Bullet's removeEdgePair): unlinks from both vertex rings, frees.
__device__ inline void dnc_remove_edge_pair(
    int* e_target, int* e_reverse, int* e_next, int* e_prev,
    int* v_edge, int* e_free, int* n_free, int edge)
{
    int n = e_next[edge];
    int r = e_reverse[edge];
    int src = e_target[r];
    int tgt = e_target[edge];

    if (n != edge) {
        e_prev[n] = e_prev[edge];
        e_next[e_prev[edge]] = n;
        v_edge[src] = n;
    } else {
        v_edge[src] = -1;
    }

    n = e_next[r];
    if (n != r) {
        e_prev[n] = e_prev[r];
        e_next[e_prev[r]] = n;
        v_edge[tgt] = n;
    } else {
        v_edge[tgt] = -1;
    }

    e_target[edge] = -1; e_target[r] = -1;
    e_free[(*n_free)++] = edge;
    e_free[(*n_free)++] = r;
}

// ---------------------------------------------------------------------------
// getOrientation — determines CW/CCW/NONE relationship between two edges
// ---------------------------------------------------------------------------
// Returns: 1=CCW, -1=CW, 0=NONE
// s, t define a reference plane via n = t cross s.

__device__ int dnc_get_orientation(
    int prev_e, int next_e,
    const int* e_next, const int* e_prev, const int* e_reverse, const int* e_target,
    const int* pts,
    int sx, int sy, int sz,
    int tx, int ty, int tz)
{
    if (e_next[prev_e] == next_e) {
        if (e_prev[prev_e] == next_e) {
            // Only 2 edges: use face normal test
            long long nx = (long long)ty*sz - (long long)tz*sy;
            long long ny = (long long)tz*sx - (long long)tx*sz;
            long long nz = (long long)tx*sy - (long long)ty*sx;
            int src = e_target[e_reverse[next_e]];
            int pt = e_target[prev_e], nt = e_target[next_e];
            int ax = pts[pt*3]-pts[src*3], ay = pts[pt*3+1]-pts[src*3+1], az = pts[pt*3+2]-pts[src*3+2];
            int bx = pts[nt*3]-pts[src*3], by = pts[nt*3+1]-pts[src*3+1], bz = pts[nt*3+2]-pts[src*3+2];
            long long mx = (long long)ay*bz - (long long)az*by;
            long long my = (long long)az*bx - (long long)ax*bz;
            long long mz = (long long)ax*by - (long long)ay*bx;
            long long dot = nx*mx + ny*my + nz*mz;
            return (dot > 0) ? 1 : -1;
        }
        return 1; // CCW
    }
    if (e_prev[prev_e] == next_e) return -1; // CW
    return 0; // NONE
}

// ---------------------------------------------------------------------------
// findMaxAngle — find edge with largest dihedral angle from vertex
// ---------------------------------------------------------------------------
// ccw: false for c0 side, true for c1 side.
// Returns edge index or -1.

__device__ int dnc_find_max_angle(
    bool ccw, int start_v,
    int sx, int sy, int sz,
    long long rxsx, long long rxsy, long long rxsz,
    long long sxrxsx, long long sxrxsy, long long sxrxsz,
    const int* pts,
    const int* v_edge, const int* e_next, const int* e_prev,
    const int* e_reverse, const int* e_target, const int* e_copy,
    int merge_stamp, GpuRational64* out_min_cot)
{
    int min_edge = -1;
    int e_start = v_edge[start_v];
    if (e_start < 0) return -1;

    int e = e_start;
    do {
        if (e_copy[e] > merge_stamp) {  // old edge (not from current merge)
            int w = e_target[e];
            int tx = pts[w*3]-pts[start_v*3];
            int ty = pts[w*3+1]-pts[start_v*3+1];
            int tz = pts[w*3+2]-pts[start_v*3+2];
            long long num = (long long)tx*sxrxsx + (long long)ty*sxrxsy + (long long)tz*sxrxsz;
            long long den = (long long)tx*rxsx   + (long long)ty*rxsy   + (long long)tz*rxsz;
            GpuRational64 cot = rational64_make(num, den);
            if (!rational64_is_nan(cot)) {
                if (min_edge < 0) {
                    *out_min_cot = cot;
                    min_edge = e;
                } else {
                    int cmp = rational64_compare(cot, *out_min_cot);
                    if (cmp < 0) {
                        *out_min_cot = cot;
                        min_edge = e;
                    } else if (cmp == 0) {
                        int orient = dnc_get_orientation(min_edge, e,
                            e_next, e_prev, e_reverse, e_target, pts,
                            sx, sy, sz, tx, ty, tz);
                        if (ccw == (orient == 1))
                            min_edge = e;
                    }
                }
            }
        }
        e = e_next[e];
    } while (e != e_start);

    return min_edge;
}

// ---------------------------------------------------------------------------
// findEdgeForCoplanarFaces — advance edges along coplanar face
// ---------------------------------------------------------------------------

__device__ void dnc_find_edge_coplanar(
    int c0, int c1, int* e0_io, int* e1_io, int stop0, int stop1,
    const int* pts,
    const int* v_edge, int* e_next, int* e_prev, int* e_reverse,
    int* e_target, const int* e_copy, int merge_stamp)
{
    int start0 = *e0_io, start1 = *e1_io;
    int et0 = (start0 >= 0) ? e_target[start0] : c0;
    int et1 = (start1 >= 0) ? e_target[start1] : c1;

    int sx = pts[c1*3]-pts[c0*3], sy = pts[c1*3+1]-pts[c0*3+1], sz = pts[c1*3+2]-pts[c0*3+2];
    int ref = (start0 >= 0) ? e_target[start0] : e_target[start1];
    int dx = pts[ref*3]-pts[c0*3], dy = pts[ref*3+1]-pts[c0*3+1], dz = pts[ref*3+2]-pts[c0*3+2];
    long long nx = (long long)dy*sz-(long long)dz*sy;
    long long ny = (long long)dz*sx-(long long)dx*sz;
    long long nz = (long long)dx*sy-(long long)dy*sx;
    long long dist = (long long)pts[c0*3]*nx + (long long)pts[c0*3+1]*ny + (long long)pts[c0*3+2]*nz;
    long long px = (long long)sy*nz-(long long)sz*ny;
    long long py = (long long)sz*nx-(long long)sx*nz;
    long long pz = (long long)sx*ny-(long long)sy*nx;

    // Advance e0 along coplanar face
    long long maxDot0 = (long long)pts[et0*3]*px + (long long)pts[et0*3+1]*py + (long long)pts[et0*3+2]*pz;
    if (*e0_io >= 0) {
        for (int iter = 0; iter < 1000; iter++) {
            if (e_target[*e0_io] == stop0) break;
            int e = e_prev[e_reverse[*e0_io]];
            int w = e_target[e];
            long long dn = (long long)pts[w*3]*nx+(long long)pts[w*3+1]*ny+(long long)pts[w*3+2]*nz;
            if (dn < dist) break;
            if (e_copy[e] == merge_stamp) break;
            long long dot = (long long)pts[w*3]*px+(long long)pts[w*3+1]*py+(long long)pts[w*3+2]*pz;
            if (dot <= maxDot0) break;
            maxDot0 = dot; *e0_io = e; et0 = w;
        }
    }

    // Advance e1 along coplanar face
    long long maxDot1 = (long long)pts[et1*3]*px + (long long)pts[et1*3+1]*py + (long long)pts[et1*3+2]*pz;
    if (*e1_io >= 0) {
        for (int iter = 0; iter < 1000; iter++) {
            if (e_target[*e1_io] == stop1) break;
            int e = e_next[e_reverse[*e1_io]];
            int w = e_target[e];
            long long dn = (long long)pts[w*3]*nx+(long long)pts[w*3+1]*ny+(long long)pts[w*3+2]*nz;
            if (dn < dist) break;
            if (e_copy[e] == merge_stamp) break;
            long long dot = (long long)pts[w*3]*px+(long long)pts[w*3+1]*py+(long long)pts[w*3+2]*pz;
            if (dot <= maxDot1) break;
            maxDot1 = dot; *e1_io = e; et1 = w;
        }
    }

    // Tangent finding within coplanar face
    long long dxp = maxDot1 - maxDot0;
    if (dxp > 0) {
        for (int iter = 0; iter < 1000; iter++) {
            long long dyp = (long long)(pts[et1*3]-pts[et0*3])*sx
                          + (long long)(pts[et1*3+1]-pts[et0*3+1])*sy
                          + (long long)(pts[et1*3+2]-pts[et0*3+2])*sz;
            bool advanced = false;
            if (*e0_io >= 0 && e_target[*e0_io] != stop0) {
                int f0 = e_reverse[e_next[*e0_io]];
                if (e_copy[f0] > merge_stamp) {
                    int w = e_target[f0];
                    long long dx0 = (long long)(pts[w*3]-pts[et0*3])*px
                                  + (long long)(pts[w*3+1]-pts[et0*3+1])*py
                                  + (long long)(pts[w*3+2]-pts[et0*3+2])*pz;
                    long long dy0 = (long long)(pts[w*3]-pts[et0*3])*sx
                                  + (long long)(pts[w*3+1]-pts[et0*3+1])*sy
                                  + (long long)(pts[w*3+2]-pts[et0*3+2])*sz;
                    if ((dx0 == 0) ? (dy0 < 0) : ((dx0 < 0) && (rational64_compare(rational64_make(dy0,dx0), rational64_make(dyp,dxp)) >= 0))) {
                        et0 = w;
                        long long newDot0 = (long long)pts[et0*3]*px+(long long)pts[et0*3+1]*py+(long long)pts[et0*3+2]*pz;
                        dxp = maxDot1 - newDot0;
                        *e0_io = (*e0_io == start0) ? -1 : f0;
                        advanced = true;
                    }
                }
            }
            if (!advanced && *e1_io >= 0 && e_target[*e1_io] != stop1) {
                int f1 = e_next[e_reverse[*e1_io]];
                if (e_copy[f1] > merge_stamp) {
                    int w = e_target[f1];
                    long long dn = (long long)(pts[w*3]-pts[et1*3])*nx
                                 + (long long)(pts[w*3+1]-pts[et1*3+1])*ny
                                 + (long long)(pts[w*3+2]-pts[et1*3+2])*nz;
                    if (dn == 0) {
                        long long dx1 = (long long)(pts[w*3]-pts[et1*3])*px
                                      + (long long)(pts[w*3+1]-pts[et1*3+1])*py
                                      + (long long)(pts[w*3+2]-pts[et1*3+2])*pz;
                        long long dy1 = (long long)(pts[w*3]-pts[et1*3])*sx
                                      + (long long)(pts[w*3+1]-pts[et1*3+1])*sy
                                      + (long long)(pts[w*3+2]-pts[et1*3+2])*sz;
                        long long dxn = (long long)(pts[w*3]-pts[et0*3])*px
                                      + (long long)(pts[w*3+1]-pts[et0*3+1])*py
                                      + (long long)(pts[w*3+2]-pts[et0*3+2])*pz;
                        if ((dxn > 0) && ((dx1 == 0) ? (dy1 < 0) : ((dx1 < 0) && (rational64_compare(rational64_make(dy1,dx1), rational64_make(dyp,dxp)) > 0)))) {
                            *e1_io = f1; et1 = w; dxp = dxn;
                            advanced = true;
                        }
                    }
                }
            }
            if (!advanced) break;
        }
    } else if (dxp < 0) {
        for (int iter = 0; iter < 1000; iter++) {
            long long dyp = (long long)(pts[et1*3]-pts[et0*3])*sx
                          + (long long)(pts[et1*3+1]-pts[et0*3+1])*sy
                          + (long long)(pts[et1*3+2]-pts[et0*3+2])*sz;
            bool advanced = false;
            if (*e1_io >= 0 && e_target[*e1_io] != stop1) {
                int f1 = e_prev[e_reverse[*e1_io]];
                if (e_copy[f1] > merge_stamp) {
                    int w = e_target[f1];
                    long long dx1 = (long long)(pts[w*3]-pts[et1*3])*px
                                  + (long long)(pts[w*3+1]-pts[et1*3+1])*py
                                  + (long long)(pts[w*3+2]-pts[et1*3+2])*pz;
                    long long dy1 = (long long)(pts[w*3]-pts[et1*3])*sx
                                  + (long long)(pts[w*3+1]-pts[et1*3+1])*sy
                                  + (long long)(pts[w*3+2]-pts[et1*3+2])*sz;
                    if ((dx1 == 0) ? (dy1 > 0) : ((dx1 < 0) && (rational64_compare(rational64_make(dy1,dx1), rational64_make(dyp,dxp)) <= 0))) {
                        et1 = w;
                        long long newDot1 = (long long)pts[et1*3]*px+(long long)pts[et1*3+1]*py+(long long)pts[et1*3+2]*pz;
                        dxp = newDot1 - maxDot0;
                        *e1_io = (*e1_io == start1) ? -1 : f1;
                        advanced = true;
                    }
                }
            }
            if (!advanced && *e0_io >= 0 && e_target[*e0_io] != stop0) {
                int f0 = e_prev[e_reverse[*e0_io]];
                if (e_copy[f0] > merge_stamp) {
                    int w = e_target[f0];
                    long long dn = (long long)(pts[w*3]-pts[et0*3])*nx
                                 + (long long)(pts[w*3+1]-pts[et0*3+1])*ny
                                 + (long long)(pts[w*3+2]-pts[et0*3+2])*nz;
                    if (dn == 0) {
                        long long dx0 = (long long)(pts[w*3]-pts[et0*3])*px
                                      + (long long)(pts[w*3+1]-pts[et0*3+1])*py
                                      + (long long)(pts[w*3+2]-pts[et0*3+2])*pz;
                        long long dy0 = (long long)(pts[w*3]-pts[et0*3])*sx
                                      + (long long)(pts[w*3+1]-pts[et0*3+1])*sy
                                      + (long long)(pts[w*3+2]-pts[et0*3+2])*sz;
                        long long dxn = (long long)(pts[et1*3]-pts[w*3])*px
                                      + (long long)(pts[et1*3+1]-pts[w*3+1])*py
                                      + (long long)(pts[et1*3+2]-pts[w*3+2])*pz;
                        if ((dxn < 0) && ((dx0 == 0) ? (dy0 > 0) : ((dx0 < 0) && (rational64_compare(rational64_make(dy0,dx0), rational64_make(dyp,dxp)) < 0)))) {
                            *e0_io = f0; et0 = w; dxp = dxn;
                            advanced = true;
                        }
                    }
                }
            }
            if (!advanced) break;
        }
    }
}

// ---------------------------------------------------------------------------
// mergeProjection — find initial bridge via 2D convex hull merge
// ---------------------------------------------------------------------------
// Returns true (normal case) or false (degenerate same-x,y case).
// h0 is updated to the merged 2D hull.

__device__ bool dnc_merge_projection(
    const int* pts, int* v_next, int* v_prev, const int* v_edge,
    const int* e_next, const int* e_target,
    int* h0_minXy, int* h0_maxXy, int* h0_minYx, int* h0_maxYx,
    int h1_minXy, int h1_maxXy, int h1_minYx, int h1_maxYx,
    int* c0_out, int* c1_out)
{
    int v0 = *h0_maxYx;
    int v1 = h1_minYx;

    if (pts[v0*3] == pts[v1*3] && pts[v0*3+1] == pts[v1*3+1]) {
        // Degenerate: same (x,y)
        int v1p = v_prev[v1];
        if (v1p == v1) {
            *c0_out = v0;
            if (v_edge[v1] >= 0) v1 = e_target[v_edge[v1]];
            *c1_out = v1;
            return false;
        }
        int v1n = v_next[v1];
        v_next[v1p] = v1n;
        v_prev[v1n] = v1p;
        if (v1 == h1_minXy) {
            h1_minXy = ((pts[v1n*3] < pts[v1p*3]) || (pts[v1n*3]==pts[v1p*3] && pts[v1n*3+1]<pts[v1p*3+1])) ? v1n : v1p;
        }
        if (v1 == h1_maxXy) {
            h1_maxXy = ((pts[v1n*3] > pts[v1p*3]) || (pts[v1n*3]==pts[v1p*3] && pts[v1n*3+1]>pts[v1p*3+1])) ? v1n : v1p;
        }
    }

    v0 = *h0_maxXy;
    v1 = h1_maxXy;
    int v00 = -1, v10 = -1;
    int sign = 1;

    for (int side = 0; side <= 1; side++) {
        int dxv = (pts[v1*3] - pts[v0*3]) * sign;
        if (dxv > 0) {
            for (int iter = 0; iter < 10000; iter++) {
                int dy = pts[v1*3+1] - pts[v0*3+1];
                int w0 = side ? v_next[v0] : v_prev[v0];
                if (w0 != v0) {
                    int dx0 = (pts[w0*3]-pts[v0*3])*sign;
                    int dy0 = pts[w0*3+1]-pts[v0*3+1];
                    if ((dy0<=0)&&((dx0==0)||((dx0<0)&&((long long)dy0*dxv<=(long long)dy*dx0)))) {
                        v0=w0; dxv=(pts[v1*3]-pts[v0*3])*sign; continue;
                    }
                }
                int w1 = side ? v_next[v1] : v_prev[v1];
                if (w1 != v1) {
                    int dx1=(pts[w1*3]-pts[v1*3])*sign;
                    int dy1=pts[w1*3+1]-pts[v1*3+1];
                    int dxn=(pts[w1*3]-pts[v0*3])*sign;
                    if ((dxn>0)&&(dy1<0)&&((dx1==0)||((dx1<0)&&((long long)dy1*dxv<(long long)dy*dx1)))) {
                        v1=w1; dxv=dxn; continue;
                    }
                }
                break;
            }
        } else if (dxv < 0) {
            for (int iter = 0; iter < 10000; iter++) {
                int dy = pts[v1*3+1] - pts[v0*3+1];
                int w1 = side ? v_prev[v1] : v_next[v1];
                if (w1 != v1) {
                    int dx1=(pts[w1*3]-pts[v1*3])*sign;
                    int dy1=pts[w1*3+1]-pts[v1*3+1];
                    if ((dy1>=0)&&((dx1==0)||((dx1<0)&&((long long)dy1*dxv<=(long long)dy*dx1)))) {
                        v1=w1; dxv=(pts[v1*3]-pts[v0*3])*sign; continue;
                    }
                }
                int w0 = side ? v_prev[v0] : v_next[v0];
                if (w0 != v0) {
                    int dx0=(pts[w0*3]-pts[v0*3])*sign;
                    int dy0=pts[w0*3+1]-pts[v0*3+1];
                    int dxn=(pts[v1*3]-pts[w0*3])*sign;
                    if ((dxn<0)&&(dy0>0)&&((dx0==0)||((dx0<0)&&((long long)dy0*dxv<(long long)dy*dx0)))) {
                        v0=w0; dxv=dxn; continue;
                    }
                }
                break;
            }
        } else {
            int x = pts[v0*3];
            int y0 = pts[v0*3+1];
            for (int iter = 0; iter < 10000; iter++) {
                int t = side ? v_next[v0] : v_prev[v0];
                if (t==v0 || pts[t*3]!=x || pts[t*3+1]>y0) break;
                v0=t; y0=pts[t*3+1];
            }
            int y1 = pts[v1*3+1];
            for (int iter = 0; iter < 10000; iter++) {
                int t = side ? v_prev[v1] : v_next[v1];
                if (t==v1 || pts[t*3]!=x || pts[t*3+1]<y1) break;
                v1=t; y1=pts[t*3+1];
            }
        }

        if (side == 0) {
            v00 = v0; v10 = v1;
            v0 = *h0_minXy; v1 = h1_minXy;
            sign = -1;
        }
    }

    // Connect circular lists
    v_prev[v0] = v1; v_next[v1] = v0;
    v_next[v00] = v10; v_prev[v10] = v00;

    // Update h0 extremes
    if (pts[h1_minXy*3] < pts[(*h0_minXy)*3] ||
        (pts[h1_minXy*3]==pts[(*h0_minXy)*3] && pts[h1_minXy*3+1]<pts[(*h0_minXy)*3+1]))
        *h0_minXy = h1_minXy;
    if (pts[h1_maxXy*3] > pts[(*h0_maxXy)*3] ||
        (pts[h1_maxXy*3]==pts[(*h0_maxXy)*3] && pts[h1_maxXy*3+1]>=pts[(*h0_maxXy)*3+1]))
        *h0_maxXy = h1_maxXy;
    *h0_maxYx = h1_maxYx;

    *c0_out = v00;
    *c1_out = v10;
    return true;
}

// ---------------------------------------------------------------------------
// merge — full 3D Preparata-Hong merge with interior edge deletion
// ---------------------------------------------------------------------------

__device__ void dnc_merge(
    const int* pts, int* v_next, int* v_prev, int* v_edge,
    int* e_target, int* e_reverse, int* e_next, int* e_prev, int* e_copy,
    int* e_free, int* n_free, int* n_alloc, int max_e,
    int* h0_minXy, int* h0_maxXy, int* h0_minYx, int* h0_maxYx,
    int h1_minXy, int h1_maxXy, int h1_minYx, int h1_maxYx,
    int merge_stamp, int* out_error)
{
    if (h1_maxXy < 0) return;
    if (*h0_maxXy < 0) {
        *h0_minXy=h1_minXy; *h0_maxXy=h1_maxXy;
        *h0_minYx=h1_minYx; *h0_maxYx=h1_maxYx;
        return;
    }

    int c0, c1;
    int toPrev0=-1, firstNew0=-1, pendingHead0=-1, pendingTail0=-1;
    int toPrev1=-1, firstNew1=-1, pendingHead1=-1, pendingTail1=-1;
    int prevPt_x, prevPt_y, prevPt_z;

    bool proj_ok = dnc_merge_projection(pts, v_next, v_prev, v_edge, e_next, e_target,
                                         h0_minXy, h0_maxXy, h0_minYx, h0_maxYx,
                                         h1_minXy, h1_maxXy, h1_minYx, h1_maxYx,
                                         &c0, &c1);

    if (proj_ok) {
        // Check for coplanar start edges on the bottom face
        int sx=pts[c1*3]-pts[c0*3], sy=pts[c1*3+1]-pts[c0*3+1], sz=pts[c1*3+2]-pts[c0*3+2];
        // normal = (0,0,-1) cross s = (sy, -sx, 0)
        long long bnx=sy, bny=-sx, bnz=0;
        // t = s cross normal
        long long btx=(long long)sz*sx, bty=(long long)sz*sy, btz=-((long long)sx*sx+(long long)sy*sy);

        int start0=-1;
        if (v_edge[c0]>=0) {
            int e_st=v_edge[c0], e=e_st;
            do {
                int w=e_target[e];
                long long dn=(long long)(pts[w*3]-pts[c0*3])*bnx
                            +(long long)(pts[w*3+1]-pts[c0*3+1])*bny;
                if (dn==0) {
                    long long dt=(long long)(pts[w*3]-pts[c0*3])*btx
                                +(long long)(pts[w*3+1]-pts[c0*3+1])*bty
                                +(long long)(pts[w*3+2]-pts[c0*3+2])*btz;
                    if (dt>0) {
                        if (start0<0) { start0=e; }
                        else {
                            int orient=dnc_get_orientation(start0,e,e_next,e_prev,e_reverse,e_target,pts,
                                sx,sy,sz,0,0,-1);
                            if (orient==-1) start0=e; // CLOCKWISE
                        }
                    }
                }
                e=e_next[e];
            } while (e!=e_st);
        }

        int start1=-1;
        if (v_edge[c1]>=0) {
            int e_st=v_edge[c1], e=e_st;
            do {
                int w=e_target[e];
                long long dn=(long long)(pts[w*3]-pts[c1*3])*bnx
                            +(long long)(pts[w*3+1]-pts[c1*3+1])*bny;
                if (dn==0) {
                    long long dt=(long long)(pts[w*3]-pts[c1*3])*btx
                                +(long long)(pts[w*3+1]-pts[c1*3+1])*bty
                                +(long long)(pts[w*3+2]-pts[c1*3+2])*btz;
                    if (dt>0) {
                        if (start1<0) { start1=e; }
                        else {
                            int orient=dnc_get_orientation(start1,e,e_next,e_prev,e_reverse,e_target,pts,
                                sx,sy,sz,0,0,-1);
                            if (orient==1) start1=e; // CCW
                        }
                    }
                }
                e=e_next[e];
            } while (e!=e_st);
        }

        if (start0>=0 || start1>=0) {
            dnc_find_edge_coplanar(c0,c1,&start0,&start1,-1,-1,
                pts,v_edge,e_next,e_prev,e_reverse,e_target,e_copy,merge_stamp);
            if (start0>=0) c0=e_target[start0];
            if (start1>=0) c1=e_target[start1];
        }

        prevPt_x=pts[c1*3]; prevPt_y=pts[c1*3+1]; prevPt_z=pts[c1*3+2]+1;
    } else {
        prevPt_x=pts[c1*3]+1; prevPt_y=pts[c1*3+1]; prevPt_z=pts[c1*3+2];
    }

    int first0=c0, first1=c1;
    bool firstRun=true;

    for (int mainIter=0; mainIter<10000; mainIter++) {
        int sx=pts[c1*3]-pts[c0*3], sy=pts[c1*3+1]-pts[c0*3+1], sz=pts[c1*3+2]-pts[c0*3+2];
        int rx=prevPt_x-pts[c0*3], ry=prevPt_y-pts[c0*3+1], rz=prevPt_z-pts[c0*3+2];
        long long rxsx=(long long)ry*sz-(long long)rz*sy;
        long long rxsy=(long long)rz*sx-(long long)rx*sz;
        long long rxsz=(long long)rx*sy-(long long)ry*sx;
        long long sxrxsx=(long long)sy*rxsz-(long long)sz*rxsy;
        long long sxrxsy=(long long)sz*rxsx-(long long)sx*rxsz;
        long long sxrxsz=(long long)sx*rxsy-(long long)sy*rxsx;

        GpuRational64 minCot0={0,0,0};
        int min0=dnc_find_max_angle(false,c0,sx,sy,sz,rxsx,rxsy,rxsz,sxrxsx,sxrxsy,sxrxsz,
            pts,v_edge,e_next,e_prev,e_reverse,e_target,e_copy,merge_stamp,&minCot0);
        GpuRational64 minCot1={0,0,0};
        int min1=dnc_find_max_angle(true,c1,sx,sy,sz,rxsx,rxsy,rxsz,sxrxsx,sxrxsy,sxrxsz,
            pts,v_edge,e_next,e_prev,e_reverse,e_target,e_copy,merge_stamp,&minCot1);

        if (min0<0 && min1<0) {
            // Both have no old edges — create simple edge pair
            int e=dnc_new_edge_pair(e_target,e_reverse,e_next,e_prev,e_copy,
                                     e_free,n_free,n_alloc,max_e,c0,c1,merge_stamp);
            if (e<0) { *out_error=1; return; }
            int r=e_reverse[e];
            e_next[e]=e; e_prev[e]=e; v_edge[c0]=e;
            e_next[r]=r; e_prev[r]=r; v_edge[c1]=r;
            return;
        }

        int cmp = (min0<0) ? 1 : (min1<0) ? -1 : rational64_compare(minCot0,minCot1);

        if (firstRun || ((cmp>=0) ? !rational64_is_neg_inf(minCot1) : !rational64_is_neg_inf(minCot0))) {
            int e=dnc_new_edge_pair(e_target,e_reverse,e_next,e_prev,e_copy,
                                     e_free,n_free,n_alloc,max_e,c0,c1,merge_stamp);
            if (e<0) { *out_error=1; return; }
            int r=e_reverse[e];
            // Add e to pending0 (backward: tail→...→head via next)
            if (pendingTail0>=0) e_prev[pendingTail0]=e; else pendingHead0=e;
            e_next[e]=pendingTail0; pendingTail0=e;
            // Add r to pending1 (forward: head→...→tail via next)
            if (pendingTail1>=0) e_next[pendingTail1]=r; else pendingHead1=r;
            e_prev[r]=pendingTail1; pendingTail1=r;
        }

        int e0=min0, e1=min1;
        if (cmp==0) {
            dnc_find_edge_coplanar(c0,c1,&e0,&e1,-1,-1,
                pts,v_edge,e_next,e_prev,e_reverse,e_target,e_copy,merge_stamp);
        }

        // Advance c1 side
        if ((cmp>=0) && e1>=0) {
            if (toPrev1>=0) {
                int e=e_next[toPrev1];
                while (e!=min1) { int n=e_next[e]; dnc_remove_edge_pair(e_target,e_reverse,e_next,e_prev,v_edge,e_free,n_free,e); e=n; }
            }
            if (pendingTail1>=0) {
                if (toPrev1>=0) { dnc_edge_link(e_next,e_prev,toPrev1,pendingHead1); }
                else { dnc_edge_link(e_next,e_prev,e_prev[min1],pendingHead1); firstNew1=pendingHead1; }
                dnc_edge_link(e_next,e_prev,pendingTail1,min1);
                pendingHead1=-1; pendingTail1=-1;
            } else if (toPrev1<0) { firstNew1=min1; }
            prevPt_x=pts[c1*3]; prevPt_y=pts[c1*3+1]; prevPt_z=pts[c1*3+2];
            c1=e_target[e1]; toPrev1=e_reverse[e1];
        }

        // Advance c0 side
        if ((cmp<=0) && e0>=0) {
            if (toPrev0>=0) {
                int e=e_prev[toPrev0];
                while (e!=min0) { int n=e_prev[e]; dnc_remove_edge_pair(e_target,e_reverse,e_next,e_prev,v_edge,e_free,n_free,e); e=n; }
            }
            if (pendingTail0>=0) {
                if (toPrev0>=0) { dnc_edge_link(e_next,e_prev,pendingHead0,toPrev0); }
                else { dnc_edge_link(e_next,e_prev,pendingHead0,e_next[min0]); firstNew0=pendingHead0; }
                dnc_edge_link(e_next,e_prev,min0,pendingTail0);
                pendingHead0=-1; pendingTail0=-1;
            } else if (toPrev0<0) { firstNew0=min0; }
            prevPt_x=pts[c0*3]; prevPt_y=pts[c0*3+1]; prevPt_z=pts[c0*3+2];
            c0=e_target[e0]; toPrev0=e_reverse[e0];
        }

        // Termination: seam closed
        if (c0==first0 && c1==first1) {
            if (toPrev0<0) {
                dnc_edge_link(e_next,e_prev,pendingHead0,pendingTail0);
                v_edge[c0]=pendingTail0;
            } else {
                int e=e_prev[toPrev0];
                while (e!=firstNew0) { int n=e_prev[e]; dnc_remove_edge_pair(e_target,e_reverse,e_next,e_prev,v_edge,e_free,n_free,e); e=n; }
                if (pendingTail0>=0) {
                    dnc_edge_link(e_next,e_prev,pendingHead0,toPrev0);
                    dnc_edge_link(e_next,e_prev,firstNew0,pendingTail0);
                }
            }
            if (toPrev1<0) {
                dnc_edge_link(e_next,e_prev,pendingTail1,pendingHead1);
                v_edge[c1]=pendingTail1;
            } else {
                int e=e_next[toPrev1];
                while (e!=firstNew1) { int n=e_next[e]; dnc_remove_edge_pair(e_target,e_reverse,e_next,e_prev,v_edge,e_free,n_free,e); e=n; }
                if (pendingTail1>=0) {
                    dnc_edge_link(e_next,e_prev,toPrev1,pendingHead1);
                    dnc_edge_link(e_next,e_prev,pendingTail1,firstNew1);
                }
            }
            return;
        }

        firstRun = false;
    }
}

// ---------------------------------------------------------------------------
// hull_dandc_warp — main entry point
// ---------------------------------------------------------------------------

__device__ float hull_dandc_warp(
    const float* pts,
    int          n_pts,
    int          lane,
    WarpPool*    pool,
    int*         error)
{
    *error = 0;
    if (n_pts < 4) { *error = 2; return -1.0f; }

    // Step 1: AABB (warp-parallel)
    float lo[3]={1e30f,1e30f,1e30f}, hi[3]={-1e30f,-1e30f,-1e30f};
    for (int i=lane; i<n_pts; i+=WARP_SIZE) {
        float x=pts[i*3],y=pts[i*3+1],z=pts[i*3+2];
        lo[0]=fminf(lo[0],x); hi[0]=fmaxf(hi[0],x);
        lo[1]=fminf(lo[1],y); hi[1]=fmaxf(hi[1],y);
        lo[2]=fminf(lo[2],z); hi[2]=fmaxf(hi[2],z);
    }
    for (int k=0;k<3;k++) {
        for (int off=16;off>0;off>>=1) {
            lo[k]=fminf(lo[k],__shfl_xor_sync(WARP_MASK,lo[k],off));
            hi[k]=fmaxf(hi[k],__shfl_xor_sync(WARP_MASK,hi[k],off));
        }
    }
    __syncwarp(WARP_MASK);

    // Thread 0 picks axes
    int max_ax=0, med_ax=1, min_ax=2;
    if (lane==0) {
        int order[3]={0,1,2};
        float exts[3]={hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]};
        for (int a=1;a<3;a++)
            for (int b=a; b>0 && exts[order[b]]>exts[order[b-1]]; b--)
                { int t=order[b]; order[b]=order[b-1]; order[b-1]=t; }
        max_ax=order[0]; med_ax=order[1]; min_ax=order[2];
    }
    max_ax=warp_bcast_i(max_ax); med_ax=warp_bcast_i(med_ax); min_ax=warp_bcast_i(min_ax);

    // Step 2: Allocate
    int max_e = n_pts*12+32;

    float* sorted_pts=(float*)warp_pool_alloc(pool,n_pts*3*(int)sizeof(float),lane);
    int*   int_pts   =(int*)  warp_pool_alloc(pool,n_pts*3*(int)sizeof(int),lane);
    float* temp_pts  =(float*)warp_pool_alloc(pool,n_pts*3*(int)sizeof(float),lane);
    int*   sort_idx  =(int*)  warp_pool_alloc(pool,n_pts*(int)sizeof(int),lane);
    int*   temp_idx  =(int*)  warp_pool_alloc(pool,n_pts*(int)sizeof(int),lane);
    int*   v_edge    =(int*)  warp_pool_alloc(pool,n_pts*(int)sizeof(int),lane);
    int*   v_next    =(int*)  warp_pool_alloc(pool,n_pts*(int)sizeof(int),lane);
    int*   v_prev    =(int*)  warp_pool_alloc(pool,n_pts*(int)sizeof(int),lane);
    int*   e_target  =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    int*   e_reverse =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    int*   e_next    =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    int*   e_prev    =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    int*   e_copy    =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    int*   e_free    =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    // IntermediateHull: [minXy, maxXy, minYx, maxYx] * n_pts
    int*   hull_data =(int*)  warp_pool_alloc(pool,n_pts*4*(int)sizeof(int),lane);
    int*   e_visited =(int*)  warp_pool_alloc(pool,max_e*(int)sizeof(int),lane);
    // scalars[0..7] ints, scalars[8..10] floats (scale factors)
    int*   scalars   =(int*)  warp_pool_alloc(pool,16*(int)sizeof(int),lane);

    __syncwarp(WARP_MASK);
    if (pool->error) { *error=1; return -1.0f; }

    // Step 3: Convert to int32, sort (thread 0)
    if (lane==0) {
        float center[3], inv_scale[3], scale_f[3];
        for (int k=0;k<3;k++) {
            center[k]=(lo[k]+hi[k])*0.5f;
            float ext=hi[k]-lo[k];
            scale_f[k] = (ext>0.0f) ? ext/10216.0f : 1.0f;
            inv_scale[k] = (ext>0.0f) ? 10216.0f/ext : 1.0f;
        }
        ((float*)scalars)[8]=scale_f[0];
        ((float*)scalars)[9]=scale_f[1];
        ((float*)scalars)[10]=scale_f[2];

        for (int i=0;i<n_pts;i++) {
            sorted_pts[i*3]=pts[i*3]; sorted_pts[i*3+1]=pts[i*3+1]; sorted_pts[i*3+2]=pts[i*3+2];
            // Bullet layout: x=med, y=max, z=min
            int_pts[i*3]  =(int)((pts[i*3+med_ax]-center[med_ax])*inv_scale[med_ax]);
            int_pts[i*3+1]=(int)((pts[i*3+max_ax]-center[max_ax])*inv_scale[max_ax]);
            int_pts[i*3+2]=(int)((pts[i*3+min_ax]-center[min_ax])*inv_scale[min_ax]);
            sort_idx[i]=i;
        }

        // Merge sort by (y=max, x=med, z=min)
        for (int width=1; width<n_pts; width*=2) {
            for (int lo_i=0; lo_i<n_pts; lo_i+=2*width) {
                int mid=lo_i+width; if(mid>n_pts) mid=n_pts;
                int hi_i=lo_i+2*width; if(hi_i>n_pts) hi_i=n_pts;
                int i=lo_i, j=mid, k=lo_i;
                while (i<mid && j<hi_i) {
                    int a=sort_idx[i], b=sort_idx[j];
                    bool af;
                    if (int_pts[a*3+1]!=int_pts[b*3+1]) af=(int_pts[a*3+1]<int_pts[b*3+1]);
                    else if (int_pts[a*3]!=int_pts[b*3]) af=(int_pts[a*3]<int_pts[b*3]);
                    else af=(int_pts[a*3+2]<=int_pts[b*3+2]);
                    temp_idx[k++] = af ? sort_idx[i++] : sort_idx[j++];
                }
                while (i<mid) temp_idx[k++]=sort_idx[i++];
                while (j<hi_i) temp_idx[k++]=sort_idx[j++];
            }
            for (int i=0;i<n_pts;i++) sort_idx[i]=temp_idx[i];
        }

        // Rearrange sorted_pts by sort order
        for (int i=0;i<n_pts;i++) {
            int s=sort_idx[i];
            temp_pts[i*3]=sorted_pts[s*3]; temp_pts[i*3+1]=sorted_pts[s*3+1]; temp_pts[i*3+2]=sorted_pts[s*3+2];
        }
        for (int i=0;i<n_pts*3;i++) sorted_pts[i]=temp_pts[i];

        // Rearrange int_pts (reuse temp_pts as int scratch — same size)
        int* ti=(int*)temp_pts;
        for (int i=0;i<n_pts;i++) {
            int s=sort_idx[i];
            ti[i*3]=int_pts[s*3]; ti[i*3+1]=int_pts[s*3+1]; ti[i*3+2]=int_pts[s*3+2];
        }
        for (int i=0;i<n_pts*3;i++) int_pts[i]=ti[i];

        // Initialize half-edge arrays
        for (int i=0;i<max_e;i++) {
            e_target[i]=-1; e_reverse[i]=-1; e_next[i]=-1; e_prev[i]=-1; e_copy[i]=0;
        }

        // Initialize per-vertex: single-vertex hulls
        for (int i=0;i<n_pts;i++) {
            v_edge[i]=-1;
            v_next[i]=i; v_prev[i]=i; // circular self-loop
            hull_data[i*4+0]=i; // minXy
            hull_data[i*4+1]=i; // maxXy
            hull_data[i*4+2]=i; // minYx
            hull_data[i*4+3]=i; // maxYx
        }

        // Step 4: Bottom-up pairwise merge
        int n_hulls=n_pts;
        int merge_stamp=0;
        int n_free_val=0, n_alloc_val=0;
        int err_flag=0;

        while (n_hulls>1 && !err_flag) {
            int new_n=0;
            for (int h=0; h+1<n_hulls; h+=2) {
                merge_stamp--;
                int h0m=hull_data[h*4+0], h0M=hull_data[h*4+1];
                int h0y=hull_data[h*4+2], h0Y=hull_data[h*4+3];
                int h1m=hull_data[(h+1)*4+0], h1M=hull_data[(h+1)*4+1];
                int h1y=hull_data[(h+1)*4+2], h1Y=hull_data[(h+1)*4+3];

                dnc_merge(int_pts, v_next, v_prev, v_edge,
                    e_target, e_reverse, e_next, e_prev, e_copy,
                    e_free, &n_free_val, &n_alloc_val, max_e,
                    &h0m, &h0M, &h0y, &h0Y,
                    h1m, h1M, h1y, h1Y,
                    merge_stamp, &err_flag);

                hull_data[new_n*4+0]=h0m; hull_data[new_n*4+1]=h0M;
                hull_data[new_n*4+2]=h0y; hull_data[new_n*4+3]=h0Y;
                new_n++;
            }
            if (n_hulls&1) {
                hull_data[new_n*4+0]=hull_data[(n_hulls-1)*4+0];
                hull_data[new_n*4+1]=hull_data[(n_hulls-1)*4+1];
                hull_data[new_n*4+2]=hull_data[(n_hulls-1)*4+2];
                hull_data[new_n*4+3]=hull_data[(n_hulls-1)*4+3];
                new_n++;
            }
            n_hulls=new_n;
        }

        scalars[0]=err_flag;
        scalars[1]=n_alloc_val;
    }
    __syncwarp(WARP_MASK);

    if (warp_bcast_i(scalars[0])) { *error=1; return -1.0f; }

    // Step 5: Compute volume from half-edge graph (thread 0)
    float total_vol=0.0f;
    if (lane==0) {
        int n_alloc_val=scalars[1];
        for (int e=0;e<n_alloc_val;e++) e_visited[e]=0;

        for (int e0=0; e0<n_alloc_val; e0++) {
            if (e_target[e0]<0) continue;
            if (e_visited[e0]) continue;

            // Trace face cycle: face_next(e) = e_next[e_reverse[e]]
            int face_verts[64];
            int nfv=0;
            int e_cur=e0;
            do {
                if (e_reverse[e_cur]<0 || e_reverse[e_cur]>=n_alloc_val) break;
                int src_v=e_target[e_reverse[e_cur]];
                if (nfv<62) face_verts[nfv++]=src_v;
                e_visited[e_cur]=1;
                int rev=e_reverse[e_cur];
                if (e_next[rev]<0 || e_next[rev]>=n_alloc_val) break;
                e_cur=e_next[rev];
                if (nfv>62) break;
            } while (e_cur!=e0);

            if (nfv<3) continue;

            // Fan-triangulate from face_verts[0] using FLOAT coordinates
            float* p0=sorted_pts+face_verts[0]*3;
            for (int k=1;k<nfv-1;k++) {
                float* p1=sorted_pts+face_verts[k]*3;
                float* p2=sorted_pts+face_verts[k+1]*3;
                total_vol+=signed_tet_volume(
                    p0[0],p0[1],p0[2],
                    p1[0],p1[1],p1[2],
                    p2[0],p2[1],p2[2]);
            }
        }
    }
    __syncwarp(WARP_MASK);

    total_vol=warp_bcast_f(total_vol);
    return fabsf(total_vol);
}

#endif // HULL_WARP_CUH
