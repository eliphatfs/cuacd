// Batch test kernels for scalar device functions.
// Each kernel: one thread per element, reads inputs, calls device function, writes output.
// Grid: ((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1), Block: (BLOCK_SIZE, 1, 1)

// --------------------------------------------------------------------------
// batch_signed_tet_volume
// --------------------------------------------------------------------------
// Input:  tets[N*9] — 3 vertices per tet, each 3 floats (p0, p1, p2)
// Output: volumes[N] — signed tet volume for each
__global__ void batch_signed_tet_volume(
    const float* __restrict__ tets,    // [N * 9]
    float* __restrict__ volumes,       // [N]
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* t = tets + idx * 9;
    volumes[idx] = signed_tet_volume(
        t[0], t[1], t[2],
        t[3], t[4], t[5],
        t[6], t[7], t[8]);
}

// --------------------------------------------------------------------------
// batch_point_triangle_dist
// --------------------------------------------------------------------------
// Input:  points[N*3], triangles[N*9] — one point and one triangle per element
// Output: dists[N] — minimum distance from point to triangle
__global__ void batch_point_triangle_dist(
    const float* __restrict__ points,     // [N * 3]
    const float* __restrict__ triangles,  // [N * 9]
    float* __restrict__ dists,            // [N]
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* p = points + idx * 3;
    const float* tri = triangles + idx * 9;
    dists[idx] = point_triangle_dist(
        p[0], p[1], p[2],
        tri[0], tri[1], tri[2],
        tri[3], tri[4], tri[5],
        tri[6], tri[7], tri[8]);
}

// --------------------------------------------------------------------------
// batch_intersect_edge
// --------------------------------------------------------------------------
// Input:  segments[N*6] — (v0x,v0y,v0z, v1x,v1y,v1z) per segment
//         planes[N*4]   — (a,b,c,d) per plane
// Output: results[N*3]  — intersection point (ix,iy,iz) per pair
__global__ void batch_intersect_edge(
    const float* __restrict__ segments,   // [N * 6]
    const float* __restrict__ planes,     // [N * 4]
    float* __restrict__ results,          // [N * 3]
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* seg = segments + idx * 6;
    const float* pl = planes + idx * 4;
    float* out = results + idx * 3;
    intersect_edge(
        seg[0], seg[1], seg[2],
        seg[3], seg[4], seg[5],
        pl[0], pl[1], pl[2], pl[3],
        &out[0], &out[1], &out[2]);
}

// --------------------------------------------------------------------------
// batch_rv_from_volumes
// --------------------------------------------------------------------------
// Input:  mesh_vols[N], hull_vols[N], rv_k (scalar)
// Output: rvs[N]
__global__ void batch_rv_from_volumes(
    const float* __restrict__ mesh_vols,  // [N]
    const float* __restrict__ hull_vols,  // [N]
    float* __restrict__ rvs,              // [N]
    float rv_k,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    rvs[idx] = rv_from_volumes(mesh_vols[idx], hull_vols[idx], rv_k);
}
