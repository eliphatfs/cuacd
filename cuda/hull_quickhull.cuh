// hull_quickhull.cuh — Warp-based QuickHull convex hull volume.
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
//
// Requires: hull_warp_common.cuh, geometry.cuh (signed_tet_volume, EPS)

#ifndef HULL_QUICKHULL_CUH
#define HULL_QUICKHULL_CUH

#include "hull_warp_common.cuh"

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
// hull_quickhull_warp
// ============================================================================

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

#endif // HULL_QUICKHULL_CUH
