// GPU Beam Search Convex Decomposition — Pure device code.
// Compiled to fatbin, loaded via CUDA driver API.
// No host-side includes, no runtime API calls.
//
// Concavity metric: bbox concavity = 1 - V_mesh / V_bbox.
// Zero for axis-aligned convex shapes, positive for non-convex.
// Fast to compute (no hull needed). Hausdorff validates final parts.

extern "C" {

// ============================================================================
// Constants
// ============================================================================

#define BLOCK_SIZE 256
#define MAX_BEAM 16
#define MAX_PARTS_PER_BEAM 64
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

// ============================================================================
// Data structures (must match beam.c)
// ============================================================================

struct PartInfo {
    int vert_offset, vert_count;
    int tri_offset, tri_count;
    float bbox[6];    // xmin,xmax,ymin,ymax,zmin,zmax
    float rv_cost;
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
    size = (size + 15) & ~15;
    unsigned int old = atomicAdd(pool->offset, size);
    if (old + size > pool->capacity) return NULL;
    return pool->base + old;
}

__device__ float signed_tet_volume(float p0x, float p0y, float p0z,
                                   float p1x, float p1y, float p1z,
                                   float p2x, float p2y, float p2z) {
    // V = p0 . (p1 x p2) / 6
    float cx = p1y * p2z - p1z * p2y;
    float cy = p1z * p2x - p1x * p2z;
    float cz = p1x * p2y - p1y * p2x;
    return (p0x * cx + p0y * cy + p0z * cz) / 6.0f;
}

__device__ void intersect_edge(float v0x, float v0y, float v0z,
                               float v1x, float v1y, float v1z,
                               float a, float b, float c, float d,
                               float* ix, float* iy, float* iz) {
    float d0 = a * v0x + b * v0y + c * v0z + d;
    float d1 = a * v1x + b * v1y + c * v1z + d;
    float t = d0 / (d0 - d1);
    *ix = v0x + t * (v1x - v0x);
    *iy = v0y + t * (v1y - v0y);
    *iz = v0z + t * (v1z - v0z);
}

// Atomic float min/max via CAS
__device__ float atomicMinF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fminf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ float atomicMaxF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fmaxf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

// Shared-memory parallel reduction (sum)
__device__ float block_reduce_sum(float val, float* smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

// Shared-memory parallel reduction (min/max for bbox)
__device__ void block_reduce_bbox(const float* verts, int n_verts, int vert_offset,
                                  int tid, float* smem,
                                  float* out_min, float* out_max) {
    // Compute bbox per axis
    for (int axis = 0; axis < 3; axis++) {
        float lo = 1e30f, hi = -1e30f;
        for (int i = tid; i < n_verts; i += BLOCK_SIZE) {
            float v = verts[(vert_offset + i) * 3 + axis];
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
        // Reduce min
        smem[tid] = lo;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fminf(smem[tid], smem[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out_min[axis] = smem[0];
        // Reduce max
        smem[tid] = hi;
        __syncthreads();
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out_max[axis] = smem[0];
        __syncthreads();
    }
}

// Compute concavity for a mesh part.
// Uses bbox max dimension as a proxy: larger parts are "more concave" and
// need further splitting. The beam search terminates based on Hausdorff.
// Returns the max bbox dimension of the part (range [0, 2] for normalized mesh).
__device__ float compute_concavity_tris(
    const float* verts,      // all vertices (global pool)
    const int*   tris,       // triangle indices (global)
    int          n_tris,
    int          tid,
    float*       smem)       // [BLOCK_SIZE]
{
    // Compute bbox from triangle vertices only (not all part vertices)
    float lo[3] = {1e30f, 1e30f, 1e30f};
    float hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int t = tid; t < n_tris; t += BLOCK_SIZE) {
        for (int e = 0; e < 3; e++) {
            int vi = tris[t*3+e];
            for (int k = 0; k < 3; k++) {
                float v = verts[vi*3+k];
                lo[k] = fminf(lo[k], v);
                hi[k] = fmaxf(hi[k], v);
            }
        }
    }

    // Reduce bbox across threads
    __shared__ float s_lo[3], s_hi[3];
    for (int k = 0; k < 3; k++) {
        smem[tid] = lo[k];
        __syncthreads();
        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fminf(smem[tid], smem[tid+s]);
            __syncthreads();
        }
        if (tid == 0) s_lo[k] = smem[0];

        smem[tid] = hi[k];
        __syncthreads();
        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
            __syncthreads();
        }
        if (tid == 0) s_hi[k] = smem[0];
        __syncthreads();
    }

    __shared__ float result;
    if (tid == 0) {
        float dims[3];
        for (int k = 0; k < 3; k++)
            dims[k] = fmaxf(s_hi[k] - s_lo[k], 0.0f);
        float vol = dims[0] * dims[1] * dims[2];
        result = cbrtf(fmaxf(vol, 0.0f));
    }
    __syncthreads();
    return result;
}

// ============================================================================
// Kernel: evaluate_candidates
// ============================================================================
// Grid: (num_beam_items * num_planes, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Each block: clip worst part of a beam item by one plane, compute concavity.

__global__ void evaluate_candidates(
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    const PartInfo* __restrict__ parts,
    const BeamItem* __restrict__ beam,
    const float* __restrict__ planes,     // [num_planes * 4]
    int          num_planes,
    int          num_beam_items,
    float        rv_k,                    // unused, kept for API compat
    float        threshold,
    DevicePool   scratch,
    float* __restrict__ cost_buffer,
    int* __restrict__ best_component_info // unused
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

    float pa = planes[plane_idx * 4 + 0];
    float pb = planes[plane_idx * 4 + 1];
    float pc = planes[plane_idx * 4 + 2];
    float pd = planes[plane_idx * 4 + 3];

    // --- Allocate scratch (thread 0 only) ---
    __shared__ int* s_signs;
    __shared__ float* s_all_verts;
    __shared__ int *s_pos_tris, *s_neg_tris;
    if (tid == 0) {
        s_signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
        int max_new_verts = tc * 2;
        int max_out_tris = tc * 3;
        s_all_verts = (float*)pool_alloc(&scratch, (vc + max_new_verts) * 3 * sizeof(float));
        s_pos_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
        s_neg_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
    }
    __syncthreads();
    int* signs = s_signs;
    float* all_verts = s_all_verts;
    int* pos_tris = s_pos_tris;
    int* neg_tris = s_neg_tris;
    if (!signs || !all_verts || !pos_tris || !neg_tris) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // --- Classify vertices ---
    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        float vx = vertex_pool[(vo + v) * 3 + 0];
        float vy = vertex_pool[(vo + v) * 3 + 1];
        float vz = vertex_pool[(vo + v) * 3 + 2];
        float val = pa * vx + pb * vy + pc * vz + pd;
        signs[v] = (val > EPS) ? 1 : ((val < -EPS) ? -1 : 0);
    }
    __syncthreads();

    // --- Quick check: does this plane actually cut? ---
    __shared__ int has_pos_s, has_neg_s;
    if (tid == 0) { has_pos_s = 0; has_neg_s = 0; }
    __syncthreads();

    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        if (signs[v] > 0) atomicExch(&has_pos_s, 1);
        if (signs[v] < 0) atomicExch(&has_neg_s, 1);
    }
    __syncthreads();

    if (!has_pos_s || !has_neg_s) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    __shared__ int pos_cnt, neg_cnt, new_v_cnt;
    if (tid == 0) { pos_cnt = 0; neg_cnt = 0; new_v_cnt = 0; }
    __syncthreads();

    // Copy original vertices
    for (int v = tid; v < vc * 3; v += BLOCK_SIZE)
        all_verts[v] = vertex_pool[vo * 3 + v];
    __syncthreads();

    // Process each triangle
    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int li0 = triangle_pool[(to + t) * 3 + 0] - vo;
        int li1 = triangle_pool[(to + t) * 3 + 1] - vo;
        int li2 = triangle_pool[(to + t) * 3 + 2] - vo;
        int s0 = signs[li0], s1 = signs[li1], s2 = signs[li2];

        int hp = (s0 > 0) | (s1 > 0) | (s2 > 0);
        int hn = (s0 < 0) | (s1 < 0) | (s2 < 0);

        if (!hp || !hn) {
            // Entirely on one side (or on-plane → positive)
            if (hn) {
                int ni = atomicAdd(&neg_cnt, 1);
                neg_tris[ni*3]=li0; neg_tris[ni*3+1]=li1; neg_tris[ni*3+2]=li2;
            } else {
                int pi = atomicAdd(&pos_cnt, 1);
                pos_tris[pi*3]=li0; pos_tris[pi*3+1]=li1; pos_tris[pi*3+2]=li2;
            }
        } else {
            // Straddling — find the lone vertex
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
                               pa, pb, pc, pd, &ix1, &iy1, &iz1);
                intersect_edge(all_verts[lv*3], all_verts[lv*3+1], all_verts[lv*3+2],
                               all_verts[ov2*3], all_verts[ov2*3+1], all_verts[ov2*3+2],
                               pa, pb, pc, pd, &ix2, &iy2, &iz2);
                int base = vc + atomicAdd(&new_v_cnt, 2);
                int nv1 = base, nv2 = base + 1;
                all_verts[nv1*3]=ix1; all_verts[nv1*3+1]=iy1; all_verts[nv1*3+2]=iz1;
                all_verts[nv2*3]=ix2; all_verts[nv2*3+1]=iy2; all_verts[nv2*3+2]=iz2;

                // Lone side: 1 tri.  Other side: 2 tris.
                if (ls > 0) {
                    int pi = atomicAdd(&pos_cnt, 1);
                    pos_tris[pi*3]=lv; pos_tris[pi*3+1]=nv1; pos_tris[pi*3+2]=nv2;
                    int ni = atomicAdd(&neg_cnt, 2);
                    neg_tris[ni*3]=ov1; neg_tris[ni*3+1]=nv2; neg_tris[ni*3+2]=nv1;
                    neg_tris[(ni+1)*3]=ov1; neg_tris[(ni+1)*3+1]=ov2; neg_tris[(ni+1)*3+2]=nv2;
                } else {
                    int ni = atomicAdd(&neg_cnt, 1);
                    neg_tris[ni*3]=lv; neg_tris[ni*3+1]=nv1; neg_tris[ni*3+2]=nv2;
                    int pi = atomicAdd(&pos_cnt, 2);
                    pos_tris[pi*3]=ov1; pos_tris[pi*3+1]=nv2; pos_tris[pi*3+2]=nv1;
                    pos_tris[(pi+1)*3]=ov1; pos_tris[(pi+1)*3+1]=ov2; pos_tris[(pi+1)*3+2]=nv2;
                }
            } else {
                // On-plane vertex case
                int on = -1;
                for (int k = 0; k < 3; k++) if (si[k] == 0) { on = k; break; }
                if (on >= 0) {
                    int ov = vi[on];
                    int a_i = vi[(on+1)%3], b_i = vi[(on+2)%3];
                    int sa = si[(on+1)%3];
                    float ix, iy, iz;
                    intersect_edge(all_verts[a_i*3], all_verts[a_i*3+1], all_verts[a_i*3+2],
                                   all_verts[b_i*3], all_verts[b_i*3+1], all_verts[b_i*3+2],
                                   pa, pb, pc, pd, &ix, &iy, &iz);
                    int nvi = vc + atomicAdd(&new_v_cnt, 1);
                    all_verts[nvi*3]=ix; all_verts[nvi*3+1]=iy; all_verts[nvi*3+2]=iz;
                    if (sa > 0) {
                        int pi = atomicAdd(&pos_cnt, 1);
                        pos_tris[pi*3]=ov; pos_tris[pi*3+1]=a_i; pos_tris[pi*3+2]=nvi;
                        int ni = atomicAdd(&neg_cnt, 1);
                        neg_tris[ni*3]=ov; neg_tris[ni*3+1]=nvi; neg_tris[ni*3+2]=b_i;
                    } else {
                        int ni = atomicAdd(&neg_cnt, 1);
                        neg_tris[ni*3]=ov; neg_tris[ni*3+1]=a_i; neg_tris[ni*3+2]=nvi;
                        int pi = atomicAdd(&pos_cnt, 1);
                        pos_tris[pi*3]=ov; pos_tris[pi*3+1]=nvi; pos_tris[pi*3+2]=b_i;
                    }
                } else {
                    // Degenerate — assign to positive
                    int pi = atomicAdd(&pos_cnt, 1);
                    pos_tris[pi*3]=li0; pos_tris[pi*3+1]=li1; pos_tris[pi*3+2]=li2;
                }
            }
        }
    }
    __syncthreads();

    int total_verts = vc + new_v_cnt;
    int n_pos = pos_cnt;
    int n_neg = neg_cnt;

    // Reject degenerate cuts (one side empty)
    if (n_pos == 0 || n_neg == 0) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // --- Compute cost: max bbox dimension of the two halves ---
    __shared__ float smem[BLOCK_SIZE];

    float pos_cost = compute_concavity_tris(all_verts, pos_tris, n_pos, tid, smem);
    float neg_cost = compute_concavity_tris(all_verts, neg_tris, n_neg, tid, smem);

    if (tid == 0) {
        float cut_cost = fmaxf(pos_cost, neg_cost);

        // Combine with costs of other (unchanged) parts
        float other_worst = 0.0f;
        for (int p = 0; p < item.num_parts; p++) {
            if (p == worst_part) continue;
            float pc = parts[beam_idx * MAX_PARTS_PER_BEAM + p].rv_cost;
            if (pc > other_worst) other_worst = pc;
        }
        cost_buffer[bid] = fmaxf(cut_cost, other_worst);
    }
}

// ============================================================================
// Kernel: select_top_k
// ============================================================================

__global__ void select_top_k(
    const float* __restrict__ cost_buffer,
    int num_candidates, int beam_width, int num_planes,
    int* __restrict__ winner_beam_idx,
    int* __restrict__ winner_plane_idx,
    float* __restrict__ winner_costs)
{
    if (threadIdx.x != 0) return;

    for (int k = 0; k < beam_width; k++) {
        winner_costs[k] = 1e30f;
        winner_beam_idx[k] = -1;
        winner_plane_idx[k] = -1;
    }

    for (int c = 0; c < num_candidates; c++) {
        float cost = cost_buffer[c];
        if (cost >= 1e29f) continue;

        int bi = c / num_planes;
        int pi = c % num_planes;

        if (cost < winner_costs[beam_width - 1]) {
            int at = beam_width - 1;
            for (int k = 0; k < beam_width; k++) {
                if (cost < winner_costs[k]) { at = k; break; }
            }
            for (int k = beam_width - 1; k > at; k--) {
                winner_costs[k] = winner_costs[k-1];
                winner_beam_idx[k] = winner_beam_idx[k-1];
                winner_plane_idx[k] = winner_plane_idx[k-1];
            }
            winner_costs[at] = cost;
            winner_beam_idx[at] = bi;
            winner_plane_idx[at] = pi;
        }
    }
}

// ============================================================================
// Kernel: apply_cuts
// ============================================================================
// Grid: (beam_width, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Each block applies one winning cut to pool_dst.

__global__ void apply_cuts(
    const float* __restrict__ vp_src,
    const int*   __restrict__ tp_src,
    const PartInfo* __restrict__ parts_src,
    const BeamItem* __restrict__ beam_src,
    float* __restrict__ vp_dst,
    int*   __restrict__ tp_dst,
    PartInfo* __restrict__ parts_dst,
    BeamItem* __restrict__ beam_dst,
    const float* __restrict__ planes,
    const int* __restrict__ winner_beam,
    const int* __restrict__ winner_plane,
    int num_planes, float rv_k, DevicePool scratch,
    unsigned int* __restrict__ dst_voff,
    unsigned int* __restrict__ dst_toff)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int src_beam = winner_beam[bid];
    int plane_idx = winner_plane[bid];
    if (src_beam < 0) return;

    const BeamItem& si = beam_src[src_beam];
    int worst = si.worst_part_idx;
    float pa = planes[plane_idx*4], pb = planes[plane_idx*4+1];
    float pc = planes[plane_idx*4+2], pd = planes[plane_idx*4+3];

    // --- Copy unchanged parts ---
    __shared__ int out_np;
    if (tid == 0) out_np = 0;
    __syncthreads();

    for (int p = 0; p < si.num_parts; p++) {
        if (p == worst) continue;
        const PartInfo& sp = parts_src[src_beam * MAX_PARTS_PER_BEAM + p];

        __shared__ unsigned int dvo, dto;
        __shared__ int np_idx;
        if (tid == 0) {
            dvo = atomicAdd(dst_voff, sp.vert_count);
            dto = atomicAdd(dst_toff, sp.tri_count);
            np_idx = atomicAdd(&out_np, 1);
        }
        __syncthreads();

        for (int v = tid; v < sp.vert_count * 3; v += BLOCK_SIZE)
            vp_dst[dvo * 3 + v] = vp_src[sp.vert_offset * 3 + v];
        for (int t = tid; t < sp.tri_count; t += BLOCK_SIZE)
            for (int k = 0; k < 3; k++)
                tp_dst[(dto+t)*3+k] = tp_src[(sp.tri_offset+t)*3+k] - sp.vert_offset + dvo;
        __syncthreads();

        if (tid == 0) {
            PartInfo& dp = parts_dst[bid * MAX_PARTS_PER_BEAM + np_idx];
            dp.vert_offset = dvo; dp.vert_count = sp.vert_count;
            dp.tri_offset = dto;  dp.tri_count = sp.tri_count;
            for (int k = 0; k < 6; k++) dp.bbox[k] = sp.bbox[k];
            dp.rv_cost = sp.rv_cost;
        }
        __syncthreads();
    }

    // --- Clip worst part ---
    const PartInfo& wp = parts_src[src_beam * MAX_PARTS_PER_BEAM + worst];
    int vc = wp.vert_count, tc = wp.tri_count;
    int wo = wp.vert_offset, wto = wp.tri_offset;

    // Allocate scratch (thread 0 only, broadcast via shared mem)
    __shared__ int* s_signs;
    __shared__ float* s_av;
    __shared__ int *s_pt, *s_nt;
    if (tid == 0) {
        s_signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
        int max_nv = tc * 2;
        s_av = (float*)pool_alloc(&scratch, (vc + max_nv) * 3 * sizeof(float));
        s_pt = (int*)pool_alloc(&scratch, tc * 3 * 3 * sizeof(int));
        s_nt = (int*)pool_alloc(&scratch, tc * 3 * 3 * sizeof(int));
    }
    __syncthreads();
    int* signs = s_signs;
    float* av = s_av;
    int* pt = s_pt;
    int* nt = s_nt;
    if (!signs || !av || !pt || !nt) return;

    __shared__ int pc_s, nc_s, nvc_s;
    if (tid == 0) { pc_s = 0; nc_s = 0; nvc_s = 0; }
    __syncthreads();

    for (int v = tid; v < vc * 3; v += BLOCK_SIZE)
        av[v] = vp_src[wo * 3 + v];
    __syncthreads();

    // Split triangles (same logic as evaluate_candidates)
    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int li0 = tp_src[(wto+t)*3]-wo, li1 = tp_src[(wto+t)*3+1]-wo, li2 = tp_src[(wto+t)*3+2]-wo;
        int s0 = signs[li0], s1 = signs[li1], s2 = signs[li2];
        int hp = (s0>0)|(s1>0)|(s2>0);
        int hn = (s0<0)|(s1<0)|(s2<0);
        if (!hp || !hn) {
            if (hn) { int ni=atomicAdd(&nc_s,1); nt[ni*3]=li0; nt[ni*3+1]=li1; nt[ni*3+2]=li2; }
            else    { int pi=atomicAdd(&pc_s,1); pt[pi*3]=li0; pt[pi*3+1]=li1; pt[pi*3+2]=li2; }
        } else {
            int vi_a[3]={li0,li1,li2}; int si_a[3]={s0,s1,s2};
            int lone=-1;
            if(si_a[0]!=0&&si_a[0]!=si_a[1]&&si_a[0]!=si_a[2]) lone=0;
            else if(si_a[1]!=0&&si_a[1]!=si_a[0]&&si_a[1]!=si_a[2]) lone=1;
            else if(si_a[2]!=0&&si_a[2]!=si_a[0]&&si_a[2]!=si_a[1]) lone=2;
            if(lone>=0){
                int lv=vi_a[lone],ov1=vi_a[(lone+1)%3],ov2=vi_a[(lone+2)%3];
                int ls=si_a[lone];
                float ix1,iy1,iz1,ix2,iy2,iz2;
                intersect_edge(av[lv*3],av[lv*3+1],av[lv*3+2],av[ov1*3],av[ov1*3+1],av[ov1*3+2],pa,pb,pc,pd,&ix1,&iy1,&iz1);
                intersect_edge(av[lv*3],av[lv*3+1],av[lv*3+2],av[ov2*3],av[ov2*3+1],av[ov2*3+2],pa,pb,pc,pd,&ix2,&iy2,&iz2);
                int base=vc+atomicAdd(&nvc_s,2); int nv1=base,nv2=base+1;
                av[nv1*3]=ix1;av[nv1*3+1]=iy1;av[nv1*3+2]=iz1;
                av[nv2*3]=ix2;av[nv2*3+1]=iy2;av[nv2*3+2]=iz2;
                if(ls>0){ int pi=atomicAdd(&pc_s,1); pt[pi*3]=lv;pt[pi*3+1]=nv1;pt[pi*3+2]=nv2;
                          int ni=atomicAdd(&nc_s,2); nt[ni*3]=ov1;nt[ni*3+1]=nv2;nt[ni*3+2]=nv1; nt[(ni+1)*3]=ov1;nt[(ni+1)*3+1]=ov2;nt[(ni+1)*3+2]=nv2;
                } else  { int ni=atomicAdd(&nc_s,1); nt[ni*3]=lv;nt[ni*3+1]=nv1;nt[ni*3+2]=nv2;
                          int pi=atomicAdd(&pc_s,2); pt[pi*3]=ov1;pt[pi*3+1]=nv2;pt[pi*3+2]=nv1; pt[(pi+1)*3]=ov1;pt[(pi+1)*3+1]=ov2;pt[(pi+1)*3+2]=nv2;
                }
            } else {
                int on=-1; for(int k=0;k<3;k++) if(si_a[k]==0){on=k;break;}
                if(on>=0){
                    int ov=vi_a[on],a_i=vi_a[(on+1)%3],b_i=vi_a[(on+2)%3]; int sa=si_a[(on+1)%3];
                    float ix,iy,iz;
                    intersect_edge(av[a_i*3],av[a_i*3+1],av[a_i*3+2],av[b_i*3],av[b_i*3+1],av[b_i*3+2],pa,pb,pc,pd,&ix,&iy,&iz);
                    int nvi=vc+atomicAdd(&nvc_s,1); av[nvi*3]=ix;av[nvi*3+1]=iy;av[nvi*3+2]=iz;
                    if(sa>0){ int pi=atomicAdd(&pc_s,1);pt[pi*3]=ov;pt[pi*3+1]=a_i;pt[pi*3+2]=nvi;
                              int ni=atomicAdd(&nc_s,1);nt[ni*3]=ov;nt[ni*3+1]=nvi;nt[ni*3+2]=b_i;
                    } else  { int ni=atomicAdd(&nc_s,1);nt[ni*3]=ov;nt[ni*3+1]=a_i;nt[ni*3+2]=nvi;
                              int pi=atomicAdd(&pc_s,1);pt[pi*3]=ov;pt[pi*3+1]=nvi;pt[pi*3+2]=b_i;
                    }
                } else { int pi=atomicAdd(&pc_s,1);pt[pi*3]=li0;pt[pi*3+1]=li1;pt[pi*3+2]=li2; }
            }
        }
    }
    __syncthreads();

    int tv = vc + nvc_s;

    // Write positive half to dst pool
    __shared__ unsigned int pos_vo, pos_to, neg_vo, neg_to;
    __shared__ int pos_pi, neg_pi;
    if (tid == 0) {
        if (pc_s > 0) {
            pos_vo = atomicAdd(dst_voff, tv);
            pos_to = atomicAdd(dst_toff, pc_s);
            pos_pi = atomicAdd(&out_np, 1);
        }
        if (nc_s > 0) {
            neg_vo = atomicAdd(dst_voff, tv);
            neg_to = atomicAdd(dst_toff, nc_s);
            neg_pi = atomicAdd(&out_np, 1);
        }
    }
    __syncthreads();

    if (pc_s > 0) {
        for (int v = tid; v < tv * 3; v += BLOCK_SIZE)
            vp_dst[pos_vo * 3 + v] = av[v];
        for (int t = tid; t < pc_s; t += BLOCK_SIZE)
            for (int k = 0; k < 3; k++)
                tp_dst[(pos_to+t)*3+k] = pt[t*3+k] + pos_vo;
    }
    if (nc_s > 0) {
        for (int v = tid; v < tv * 3; v += BLOCK_SIZE)
            vp_dst[neg_vo * 3 + v] = av[v];
        for (int t = tid; t < nc_s; t += BLOCK_SIZE)
            for (int k = 0; k < 3; k++)
                tp_dst[(neg_to+t)*3+k] = nt[t*3+k] + neg_vo;
    }
    __syncthreads();

    if (tid == 0) {
        if (pc_s > 0) {
            PartInfo& dp = parts_dst[bid * MAX_PARTS_PER_BEAM + pos_pi];
            dp.vert_offset = pos_vo; dp.vert_count = tv;
            dp.tri_offset = pos_to; dp.tri_count = pc_s;
            dp.rv_cost = 0.0f;
        }
        if (nc_s > 0) {
            PartInfo& dp = parts_dst[bid * MAX_PARTS_PER_BEAM + neg_pi];
            dp.vert_offset = neg_vo; dp.vert_count = tv;
            dp.tri_offset = neg_to; dp.tri_count = nc_s;
            dp.rv_cost = 0.0f;
        }
        beam_dst[bid].num_parts = out_np;
        beam_dst[bid].worst_part_idx = -1;
        beam_dst[bid].worst_cost = 0.0f;
        beam_dst[bid].cut_count = si.cut_count + 1;
    }
}

// ============================================================================
// Kernel: compute_part_costs
// ============================================================================
// Grid: (num_beam_items, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Compute bbox concavity for parts that need it; find worst per beam item.

__global__ void compute_part_costs(
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    PartInfo*    __restrict__ parts,
    BeamItem*    __restrict__ beam,
    int          num_beam_items,
    float        rv_k,
    DevicePool   scratch)
{
    int beam_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (beam_idx >= num_beam_items) return;

    BeamItem& item = beam[beam_idx];
    int np = item.num_parts;

    __shared__ float worst_cost;
    __shared__ int worst_idx;
    __shared__ float smem[BLOCK_SIZE];
    if (tid == 0) { worst_cost = 0.0f; worst_idx = 0; }
    __syncthreads();

    for (int p = 0; p < np; p++) {
        PartInfo& part = parts[beam_idx * MAX_PARTS_PER_BEAM + p];

        if (part.rv_cost > 0.0f) {
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
            if (tid == 0) part.rv_cost = EPS;
            __syncthreads();
            continue;
        }

        float conc = compute_concavity_tris(
            vertex_pool, &triangle_pool[to * 3], tc, tid, smem);

        if (tid == 0) {
            part.rv_cost = fmaxf(conc, EPS);
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
// Kernel: normalize_mesh
// ============================================================================

__global__ void normalize_mesh(
    float* __restrict__ vertices, int n_verts,
    float* __restrict__ norm_info)   // [7]: cx,cy,cz,scale,...
{
    int tid = threadIdx.x;
    extern __shared__ float smem_norm[];
    float* s_min = smem_norm;
    float* s_max = smem_norm + BLOCK_SIZE * 3;

    float lmin[3] = {1e30f, 1e30f, 1e30f};
    float lmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = tid; i < n_verts; i += BLOCK_SIZE)
        for (int k = 0; k < 3; k++) {
            float v = vertices[i*3+k];
            lmin[k] = fminf(lmin[k], v);
            lmax[k] = fmaxf(lmax[k], v);
        }
    for (int k = 0; k < 3; k++) {
        s_min[tid*3+k] = lmin[k];
        s_max[tid*3+k] = lmax[k];
    }
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) for (int k = 0; k < 3; k++) {
            s_min[tid*3+k] = fminf(s_min[tid*3+k], s_min[(tid+s)*3+k]);
            s_max[tid*3+k] = fmaxf(s_max[tid*3+k], s_max[(tid+s)*3+k]);
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
        norm_info[0]=center[0]; norm_info[1]=center[1]; norm_info[2]=center[2];
        norm_info[3]=scale;
    }
    __syncthreads();
    for (int i = tid; i < n_verts; i += BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            vertices[i*3+k] = (vertices[i*3+k] - center[k]) * scale;
}

// ============================================================================
// Kernel: recover_coordinates
// ============================================================================

__global__ void recover_coordinates(
    float* __restrict__ vertices, int n_verts,
    const float* __restrict__ norm_info)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_verts) return;
    float inv = 1.0f / norm_info[3];
    vertices[idx*3+0] = vertices[idx*3+0] * inv + norm_info[0];
    vertices[idx*3+1] = vertices[idx*3+1] * inv + norm_info[1];
    vertices[idx*3+2] = vertices[idx*3+2] * inv + norm_info[2];
}

// ============================================================================
// Kernel: surface sampling + point-mesh distance + reduction (for Hausdorff)
// ============================================================================

__global__ void sample_surface(
    const float* __restrict__ vertices, const int* __restrict__ triangles,
    int n_tris, int vert_offset,
    float* __restrict__ samples, int n_samples, unsigned int seed)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_samples) return;
    unsigned int st = seed ^ (idx * 2654435761u);
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    int tri = st % n_tris;
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    int i0=triangles[tri*3]+vert_offset, i1=triangles[tri*3+1]+vert_offset, i2=triangles[tri*3+2]+vert_offset;
    float u = (float)(st & 0xFFFF)/65535.0f;
    st ^= st<<13; st ^= st>>17; st ^= st<<5;
    float v = (float)(st & 0xFFFF)/65535.0f;
    if(u+v>1.0f){u=1.0f-u;v=1.0f-v;}
    float w=1.0f-u-v;
    samples[idx*3]=w*vertices[i0*3]+u*vertices[i1*3]+v*vertices[i2*3];
    samples[idx*3+1]=w*vertices[i0*3+1]+u*vertices[i1*3+1]+v*vertices[i2*3+1];
    samples[idx*3+2]=w*vertices[i0*3+2]+u*vertices[i1*3+2]+v*vertices[i2*3+2];
}

__device__ float point_triangle_dist_beam(
    float px,float py,float pz,
    float v0x,float v0y,float v0z,float v1x,float v1y,float v1z,float v2x,float v2y,float v2z) {
    float e0x=v1x-v0x,e0y=v1y-v0y,e0z=v1z-v0z;
    float e1x=v2x-v0x,e1y=v2y-v0y,e1z=v2z-v0z;
    float dx=v0x-px,dy=v0y-py,dz=v0z-pz;
    float a=e0x*e0x+e0y*e0y+e0z*e0z, b=e0x*e1x+e0y*e1y+e0z*e1z;
    float c=e1x*e1x+e1y*e1y+e1z*e1z, d=e0x*dx+e0y*dy+e0z*dz, e=e1x*dx+e1y*dy+e1z*dz;
    float det=a*c-b*b, s=b*e-c*d, t=b*d-a*e;
    if(s+t<=det){if(s<0){if(t<0){if(d<0){t=0;s=(-d>=a)?1.0f:-d/a;}else{s=0;t=(e>=0)?0:((-e>=c)?1.0f:-e/c);}}else{s=0;t=(e>=0)?0:((-e>=c)?1.0f:-e/c);}}else if(t<0){t=0;s=(d>=0)?0:((-d>=a)?1.0f:-d/a);}else{float inv=1.0f/det;s*=inv;t*=inv;}}
    else{if(s<0){float t0=b+d,t1=c+e;if(t1>t0){float nm=t1-t0,dn=a-2*b+c;s=(nm>=dn)?1.0f:nm/dn;t=1-s;}else{s=0;t=(t1<=0)?1.0f:((e>=0)?0:-e/c);}}else if(t<0){float t0=b+e,t1=a+d;if(t1>t0){float nm=t1-t0,dn=a-2*b+c;t=(nm>=dn)?1.0f:nm/dn;s=1-t;}else{t=0;s=(t1<=0)?1.0f:((d>=0)?0:-d/a);}}else{float nm=(c+e)-(b+d);if(nm<=0){s=0;t=1;}else{float dn=a-2*b+c;s=(nm>=dn)?1.0f:nm/dn;t=1-s;}}}
    float rx=v0x+s*e0x+t*e1x-px,ry=v0y+s*e0y+t*e1y-py,rz=v0z+s*e0z+t*e1z-pz;
    return sqrtf(rx*rx+ry*ry+rz*rz);
}

__global__ void beam_point_mesh_distance(
    const float* __restrict__ points, const float* __restrict__ vertices,
    const int* __restrict__ triangles, float* __restrict__ distances,
    int N, int T, int vert_offset)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N) return;
    float px=points[idx*3],py=points[idx*3+1],pz=points[idx*3+2];
    float md = 1e30f;
    for (int t = 0; t < T; t++) {
        int i0=triangles[t*3]+vert_offset, i1=triangles[t*3+1]+vert_offset, i2=triangles[t*3+2]+vert_offset;
        md = fminf(md, point_triangle_dist_beam(px,py,pz,
            vertices[i0*3],vertices[i0*3+1],vertices[i0*3+2],
            vertices[i1*3],vertices[i1*3+1],vertices[i1*3+2],
            vertices[i2*3],vertices[i2*3+1],vertices[i2*3+2]));
    }
    distances[idx] = md;
}

__global__ void beam_reduce_max(const float* __restrict__ data, float* __restrict__ output, int N) {
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
