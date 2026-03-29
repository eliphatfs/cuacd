// Incremental convex hull with SIMT-parallel visibility tests.
// Computes volume of the 3D convex hull of a point set.
//
// Algorithm:
//   1. Thread 0: find 4 non-coplanar extreme points -> initial tetrahedron
//   2. For each remaining point:
//      a. ALL threads: visibility test on assigned faces (parallel)
//      b. Thread 0: if any visible -> find horizon edges via adjacency (O(F)),
//         remove visible faces, add new faces, recompute adjacency, update volume
//   3. Return |volume|
//
// Requires: common.cuh (BLOCK_SIZE), geometry.cuh (signed_tet_volume)

#ifndef HULL_CUH
#define HULL_CUH

#define MAX_HULL_VERTS 256
#define MAX_HULL_FACES 512  // 2*MAX_HULL_VERTS - 4

struct HullWorkspace {
    int fv0[MAX_HULL_FACES];         // face vertex 0
    int fv1[MAX_HULL_FACES];         // face vertex 1
    int fv2[MAX_HULL_FACES];         // face vertex 2
    int visible[MAX_HULL_FACES];     // per-face visibility flag
    int horizon_a[MAX_HULL_FACES];   // horizon edge endpoint A
    int horizon_b[MAX_HULL_FACES];   // horizon edge endpoint B
    int n_faces;
    int n_horizon;
    float volume;
    int any_visible;                 // block-level OR reduction
    // adjacency arrays add 4*512*4 = 8KB, total ~20KB shared memory
    int fadj0[MAX_HULL_FACES];       // adjacent face index across edge (fv0->fv1)
    int fadj1[MAX_HULL_FACES];       // adjacent face index across edge (fv1->fv2)
    int fadj2[MAX_HULL_FACES];       // adjacent face index across edge (fv2->fv0)
    int horizon_nbr[MAX_HULL_FACES]; // for each horizon edge: the non-visible neighbor face
};

// ---- Adjacency helper device functions (thread 0 only) ----

__device__ inline int hull_face_v(HullWorkspace* ws, int f, int i) {
    return i==0 ? ws->fv0[f] : i==1 ? ws->fv1[f] : ws->fv2[f];
}

__device__ inline int hull_face_adj(HullWorkspace* ws, int f, int e) {
    return e==0 ? ws->fadj0[f] : e==1 ? ws->fadj1[f] : ws->fadj2[f];
}

__device__ inline void hull_set_adj(HullWorkspace* ws, int f, int e, int nb) {
    if      (e==0) ws->fadj0[f]=nb;
    else if (e==1) ws->fadj1[f]=nb;
    else           ws->fadj2[f]=nb;
}

// Find edge e in face f where (fv[e], fv[(e+1)%3]) == (va, vb). Returns -1 if not found.
__device__ inline int hull_find_edge(HullWorkspace* ws, int f, int va, int vb) {
    for (int e = 0; e < 3; e++)
        if (hull_face_v(ws,f,e)==va && hull_face_v(ws,f,(e+1)%3)==vb) return e;
    return -1;
}

// Compute adjacency from scratch for faces [0..nf). O(nf²). Thread 0 only.
__device__ inline void hull_recompute_adj(HullWorkspace* ws, int nf) {
    for (int f = 0; f < nf; f++) ws->fadj0[f]=ws->fadj1[f]=ws->fadj2[f]=-1;
    for (int f = 0; f < nf; f++) {
        for (int e = 0; e < 3; e++) {
            if (hull_face_adj(ws,f,e) >= 0) continue;
            int ea = hull_face_v(ws,f,e), eb = hull_face_v(ws,f,(e+1)%3);
            for (int g = 0; g < nf; g++) {
                if (g == f) continue;
                int e2 = hull_find_edge(ws,g,eb,ea);
                if (e2 >= 0) { hull_set_adj(ws,f,e,g); hull_set_adj(ws,g,e2,f); }
            }
        }
    }
}

// Compute convex hull volume of a point set.
// points:   [n_points * 3] float coords (global or pool memory)
// n_points: number of input points (<= MAX_HULL_VERTS)
// tid:      threadIdx.x
// ws:       shared-memory HullWorkspace
// Returns hull volume (non-negative). All threads return the same value.
// ALL threads in the block must call this (contains __syncthreads).
__device__ float compute_hull_volume(
    const float* __restrict__ points,
    int n_points,
    int tid,
    HullWorkspace* ws)
{
    if (n_points < 4) {
        if (tid == 0) ws->volume = 0.0f;
        __syncthreads();
        return 0.0f;
    }

    // ---- Step 1: initial tetrahedron (thread 0) ----
    if (tid == 0) {
        ws->n_faces = 0;
        ws->volume = 0.0f;

        // Find two points maximally separated along x
        int p0 = 0, p1 = 0;
        float lo = points[0], hi = points[0];
        for (int i = 1; i < n_points; i++) {
            float x = points[i * 3];
            if (x < lo) { lo = x; p0 = i; }
            if (x > hi) { hi = x; p1 = i; }
        }
        if (p0 == p1) p1 = (p0 + 1) % n_points;

        // Point farthest from line p0-p1
        float dx = points[p1*3]-points[p0*3];
        float dy = points[p1*3+1]-points[p0*3+1];
        float dz = points[p1*3+2]-points[p0*3+2];
        float len2 = dx*dx + dy*dy + dz*dz;
        if (len2 < 1e-30f) len2 = 1e-30f;
        int p2 = -1;
        float best = -1.0f;
        for (int i = 0; i < n_points; i++) {
            if (i == p0 || i == p1) continue;
            float ex = points[i*3]-points[p0*3];
            float ey = points[i*3+1]-points[p0*3+1];
            float ez = points[i*3+2]-points[p0*3+2];
            float t = (ex*dx + ey*dy + ez*dz) / len2;
            float rx = ex-t*dx, ry = ey-t*dy, rz = ez-t*dz;
            float d2 = rx*rx + ry*ry + rz*rz;
            if (d2 > best) { best = d2; p2 = i; }
        }

        if (p2 < 0 || best < 1e-20f) {
            ws->n_faces = 0;  // collinear
        } else {
            // Point farthest from plane (p0,p1,p2)
            float e1x = points[p1*3]-points[p0*3], e1y = points[p1*3+1]-points[p0*3+1], e1z = points[p1*3+2]-points[p0*3+2];
            float e2x = points[p2*3]-points[p0*3], e2y = points[p2*3+1]-points[p0*3+1], e2z = points[p2*3+2]-points[p0*3+2];
            float nx = e1y*e2z-e1z*e2y, ny = e1z*e2x-e1x*e2z, nz = e1x*e2y-e1y*e2x;
            int p3 = -1;
            float best_ad = -1.0f;
            for (int i = 0; i < n_points; i++) {
                if (i == p0 || i == p1 || i == p2) continue;
                float ex = points[i*3]-points[p0*3], ey = points[i*3+1]-points[p0*3+1], ez = points[i*3+2]-points[p0*3+2];
                float ad = fabsf(ex*nx + ey*ny + ez*nz);
                if (ad > best_ad) { best_ad = ad; p3 = i; }
            }

            if (p3 < 0 || best_ad < 1e-20f) {
                ws->n_faces = 0;  // coplanar
            } else {
                // Build 4 faces with outward normals (verified against centroid)
                float cx = (points[p0*3]+points[p1*3]+points[p2*3]+points[p3*3])*0.25f;
                float cy = (points[p0*3+1]+points[p1*3+1]+points[p2*3+1]+points[p3*3+1])*0.25f;
                float cz = (points[p0*3+2]+points[p1*3+2]+points[p2*3+2]+points[p3*3+2])*0.25f;

                int ff[4][3] = {{p0,p1,p2},{p0,p3,p1},{p1,p3,p2},{p0,p2,p3}};
                for (int f = 0; f < 4; f++) {
                    int a = ff[f][0], b = ff[f][1], c = ff[f][2];
                    float fe1x = points[b*3]-points[a*3], fe1y = points[b*3+1]-points[a*3+1], fe1z = points[b*3+2]-points[a*3+2];
                    float fe2x = points[c*3]-points[a*3], fe2y = points[c*3+1]-points[a*3+1], fe2z = points[c*3+2]-points[a*3+2];
                    float fnx = fe1y*fe2z-fe1z*fe2y, fny = fe1z*fe2x-fe1x*fe2z, fnz = fe1x*fe2y-fe1y*fe2x;
                    float dot = (cx-points[a*3])*fnx + (cy-points[a*3+1])*fny + (cz-points[a*3+2])*fnz;
                    if (dot > 0) {
                        ws->fv0[f]=a; ws->fv1[f]=c; ws->fv2[f]=b;  // flip
                    } else {
                        ws->fv0[f]=a; ws->fv1[f]=b; ws->fv2[f]=c;
                    }
                }
                ws->n_faces = 4;

                float vol = 0.0f;
                for (int f = 0; f < 4; f++) {
                    int a=ws->fv0[f], b=ws->fv1[f], c=ws->fv2[f];
                    vol += signed_tet_volume(
                        points[a*3],points[a*3+1],points[a*3+2],
                        points[b*3],points[b*3+1],points[b*3+2],
                        points[c*3],points[c*3+1],points[c*3+2]);
                }
                ws->volume = vol;

                hull_recompute_adj(ws, 4);
            }
        }
    }
    __syncthreads();

    if (ws->n_faces == 0) return 0.0f;

    // ---- Step 2: incremental insertion ----
    for (int p = 0; p < n_points; p++) {
        float px = points[p*3], py = points[p*3+1], pz = points[p*3+2];
        int nf = ws->n_faces;

        // -- Parallel visibility test --
        int local_any = 0;
        for (int f = tid; f < nf; f += BLOCK_SIZE) {
            int a=ws->fv0[f], b=ws->fv1[f], c=ws->fv2[f];
            float e1x = points[b*3]-points[a*3], e1y = points[b*3+1]-points[a*3+1], e1z = points[b*3+2]-points[a*3+2];
            float e2x = points[c*3]-points[a*3], e2y = points[c*3+1]-points[a*3+1], e2z = points[c*3+2]-points[a*3+2];
            float fnx = e1y*e2z-e1z*e2y, fny = e1z*e2x-e1x*e2z, fnz = e1x*e2y-e1y*e2x;
            float dot = (px-points[a*3])*fnx + (py-points[a*3+1])*fny + (pz-points[a*3+2])*fnz;
            ws->visible[f] = (dot > EPS) ? 1 : 0;
            if (dot > EPS) local_any = 1;
        }
        __syncthreads();

        // -- Block-level OR: any face visible? --
        if (tid == 0) ws->any_visible = 0;
        __syncthreads();
        if (local_any) atomicExch(&ws->any_visible, 1);
        __syncthreads();

        if (!ws->any_visible) continue;  // interior point

        // -- Thread 0: horizon edges via adjacency (O(F)) + topology update --
        if (tid == 0) {
            // Horizon edges: for each visible face, check each edge's adjacent face.
            // If the adjacent face is non-visible, this edge is a horizon edge.
            ws->n_horizon = 0;
            for (int f = 0; f < nf; f++) {
                if (!ws->visible[f]) continue;
                for (int e = 0; e < 3; e++) {
                    int nb = hull_face_adj(ws, f, e);
                    if (nb >= 0 && !ws->visible[nb]) {
                        ws->horizon_a[ws->n_horizon]   = hull_face_v(ws, f, e);
                        ws->horizon_b[ws->n_horizon]   = hull_face_v(ws, f, (e+1)%3);
                        ws->horizon_nbr[ws->n_horizon] = nb;
                        ws->n_horizon++;
                    }
                }
            }

            // Remove visible faces, track volume delta
            float vol_delta = 0.0f;
            int dst = 0;
            for (int f = 0; f < nf; f++) {
                if (ws->visible[f]) {
                    int a=ws->fv0[f], b=ws->fv1[f], c=ws->fv2[f];
                    vol_delta -= signed_tet_volume(
                        points[a*3],points[a*3+1],points[a*3+2],
                        points[b*3],points[b*3+1],points[b*3+2],
                        points[c*3],points[c*3+1],points[c*3+2]);
                } else {
                    if (dst != f) {
                        ws->fv0[dst]=ws->fv0[f];
                        ws->fv1[dst]=ws->fv1[f];
                        ws->fv2[dst]=ws->fv2[f];
                    }
                    dst++;
                }
            }

            // Add new faces from horizon edges + point p
            for (int h = 0; h < ws->n_horizon && dst < MAX_HULL_FACES; h++) {
                int a = ws->horizon_a[h], b = ws->horizon_b[h];
                ws->fv0[dst] = a;
                ws->fv1[dst] = b;
                ws->fv2[dst] = p;
                vol_delta += signed_tet_volume(
                    points[a*3],points[a*3+1],points[a*3+2],
                    points[b*3],points[b*3+1],points[b*3+2],
                    px, py, pz);
                dst++;
            }
            ws->n_faces = dst;
            ws->volume += vol_delta;

            // Recompute full adjacency for the updated face set.
            // O(F_new²) but F_new is small (typically <100) and avoids
            // all index-remapping complexity from face compaction.
            hull_recompute_adj(ws, dst);
        }
        __syncthreads();
    }

    return fabsf(ws->volume);
}

// Pool-allocated hull for large meshes (no MAX_HULL_VERTS cap).
// Workspace arrays are in global memory (pool), sized to max_faces.
// Slower than shared-memory version but handles any vertex count.
// ALL threads must call (contains __syncthreads).
__device__ float compute_hull_volume_pool(
    const float* __restrict__ points, int n_points, int tid,
    int* fv0, int* fv1, int* fv2,
    int* visible, int* horizon_a, int* horizon_b,
    int max_faces)
{
    // Shared scalars for the hull state
    __shared__ int  hps_n_faces, hps_n_horizon, hps_any_visible;
    __shared__ float hps_volume;

    if (n_points < 4) {
        if (tid == 0) hps_volume = 0.0f;
        __syncthreads();
        return 0.0f;
    }

    // ---- Initial tetrahedron (thread 0) — same as shared-memory version ----
    if (tid == 0) {
        hps_n_faces = 0; hps_volume = 0.0f;
        int p0=0,p1=0; float lo_x=points[0],hi_x=points[0];
        for(int i=1;i<n_points;i++){float x=points[i*3];if(x<lo_x){lo_x=x;p0=i;}if(x>hi_x){hi_x=x;p1=i;}}
        if(p0==p1) p1=(p0+1)%n_points;
        float dx=points[p1*3]-points[p0*3],dy=points[p1*3+1]-points[p0*3+1],dz=points[p1*3+2]-points[p0*3+2];
        float len2=dx*dx+dy*dy+dz*dz; if(len2<1e-30f)len2=1e-30f;
        int p2=-1; float best=-1.0f;
        for(int i=0;i<n_points;i++){if(i==p0||i==p1)continue;
            float ex=points[i*3]-points[p0*3],ey=points[i*3+1]-points[p0*3+1],ez=points[i*3+2]-points[p0*3+2];
            float t=(ex*dx+ey*dy+ez*dz)/len2; float rx=ex-t*dx,ry=ey-t*dy,rz=ez-t*dz;
            float d2=rx*rx+ry*ry+rz*rz; if(d2>best){best=d2;p2=i;}}
        if(p2<0||best<1e-20f){hps_n_faces=0;}
        else{
            float e1x=points[p1*3]-points[p0*3],e1y=points[p1*3+1]-points[p0*3+1],e1z=points[p1*3+2]-points[p0*3+2];
            float e2x=points[p2*3]-points[p0*3],e2y=points[p2*3+1]-points[p0*3+1],e2z=points[p2*3+2]-points[p0*3+2];
            float nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x;
            int p3=-1; float best_ad=-1.0f;
            for(int i=0;i<n_points;i++){if(i==p0||i==p1||i==p2)continue;
                float ex=points[i*3]-points[p0*3],ey=points[i*3+1]-points[p0*3+1],ez=points[i*3+2]-points[p0*3+2];
                float ad=fabsf(ex*nx+ey*ny+ez*nz);if(ad>best_ad){best_ad=ad;p3=i;}}
            if(p3<0||best_ad<1e-20f){hps_n_faces=0;}
            else{
                float cx=(points[p0*3]+points[p1*3]+points[p2*3]+points[p3*3])*0.25f;
                float cy=(points[p0*3+1]+points[p1*3+1]+points[p2*3+1]+points[p3*3+1])*0.25f;
                float cz=(points[p0*3+2]+points[p1*3+2]+points[p2*3+2]+points[p3*3+2])*0.25f;
                int ff[4][3]={{p0,p1,p2},{p0,p3,p1},{p1,p3,p2},{p0,p2,p3}};
                for(int f=0;f<4;f++){
                    int a=ff[f][0],b=ff[f][1],c=ff[f][2];
                    float fe1x=points[b*3]-points[a*3],fe1y=points[b*3+1]-points[a*3+1],fe1z=points[b*3+2]-points[a*3+2];
                    float fe2x=points[c*3]-points[a*3],fe2y=points[c*3+1]-points[a*3+1],fe2z=points[c*3+2]-points[a*3+2];
                    float fnx=fe1y*fe2z-fe1z*fe2y,fny=fe1z*fe2x-fe1x*fe2z,fnz=fe1x*fe2y-fe1y*fe2x;
                    float dot=(cx-points[a*3])*fnx+(cy-points[a*3+1])*fny+(cz-points[a*3+2])*fnz;
                    if(dot>0){fv0[f]=a;fv1[f]=c;fv2[f]=b;}
                    else     {fv0[f]=a;fv1[f]=b;fv2[f]=c;}}
                hps_n_faces=4;
                float vol=0.0f;
                for(int f=0;f<4;f++){int a=fv0[f],b=fv1[f],c=fv2[f];
                    vol+=signed_tet_volume(points[a*3],points[a*3+1],points[a*3+2],
                        points[b*3],points[b*3+1],points[b*3+2],points[c*3],points[c*3+1],points[c*3+2]);}
                hps_volume=vol;
            }
        }
    }
    __syncthreads();
    if(hps_n_faces==0) return 0.0f;

    // ---- Incremental insertion ----
    for(int p=0;p<n_points;p++){
        float px=points[p*3],py=points[p*3+1],pz=points[p*3+2];
        int nf=hps_n_faces;
        int local_any=0;
        for(int f=tid;f<nf;f+=BLOCK_SIZE){
            int a=fv0[f],b=fv1[f],c=fv2[f];
            float e1x=points[b*3]-points[a*3],e1y=points[b*3+1]-points[a*3+1],e1z=points[b*3+2]-points[a*3+2];
            float e2x=points[c*3]-points[a*3],e2y=points[c*3+1]-points[a*3+1],e2z=points[c*3+2]-points[a*3+2];
            float fnx=e1y*e2z-e1z*e2y,fny=e1z*e2x-e1x*e2z,fnz=e1x*e2y-e1y*e2x;
            float dot=(px-points[a*3])*fnx+(py-points[a*3+1])*fny+(pz-points[a*3+2])*fnz;
            visible[f]=(dot>EPS)?1:0;
            if(dot>EPS) local_any=1;
        }
        __syncthreads();
        if(tid==0) hps_any_visible=0;
        __syncthreads();
        if(local_any) atomicExch(&hps_any_visible,1);
        __syncthreads();
        if(!hps_any_visible) continue;

        if(tid==0){
            hps_n_horizon=0;
            for(int f=0;f<nf;f++){
                if(!visible[f]) continue;
                int fvl[3]={fv0[f],fv1[f],fv2[f]};
                for(int e=0;e<3;e++){
                    int ea=fvl[e],eb=fvl[(e+1)%3];
                    int is_int=0;
                    for(int g=0;g<nf;g++){
                        if(g==f||!visible[g]) continue;
                        int gv[3]={fv0[g],fv1[g],fv2[g]};
                        for(int e2=0;e2<3;e2++){
                            if(gv[e2]==eb&&gv[(e2+1)%3]==ea){is_int=1;break;}}
                        if(is_int) break;}
                    if(!is_int&&hps_n_horizon<max_faces){
                        horizon_a[hps_n_horizon]=ea;horizon_b[hps_n_horizon]=eb;hps_n_horizon++;}
                }
            }
            float vol_delta=0.0f; int dst=0;
            for(int f=0;f<nf;f++){
                if(visible[f]){
                    int a=fv0[f],b=fv1[f],c=fv2[f];
                    vol_delta-=signed_tet_volume(points[a*3],points[a*3+1],points[a*3+2],
                        points[b*3],points[b*3+1],points[b*3+2],points[c*3],points[c*3+1],points[c*3+2]);
                } else { if(dst!=f){fv0[dst]=fv0[f];fv1[dst]=fv1[f];fv2[dst]=fv2[f];} dst++; }
            }
            for(int h=0;h<hps_n_horizon&&dst<max_faces;h++){
                int a=horizon_a[h],b=horizon_b[h];
                fv0[dst]=a;fv1[dst]=b;fv2[dst]=p;
                vol_delta+=signed_tet_volume(points[a*3],points[a*3+1],points[a*3+2],
                    points[b*3],points[b*3+1],points[b*3+2],px,py,pz);
                dst++;}
            hps_n_faces=dst; hps_volume+=vol_delta;
        }
        __syncthreads();
    }
    return fabsf(hps_volume);
}

#endif // HULL_CUH
