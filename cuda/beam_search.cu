// Beam search kernels: evaluate candidates, select top-k, apply cuts,
// compute part costs.
// Requires: common.cuh, reduce.cuh, geometry.cuh, hull.cuh

// ============================================================================
// Helper: compute Rv for a triangle set
// ============================================================================
// Full Rv computation: mesh volume (signed tet) + optional cap volume
// (divergence theorem) + convex hull volume -> Rv formula.
//
// ALL threads must call. Returns Rv (same value for all threads).
//
// For a closed mesh, pass plane_d = 0 and n_boundary = 0.
// For an open mesh from a plane cut, pass the plane and boundary edges.

__device__ float compute_rv_for_tris(
    const float* verts,            // vertex coordinates
    const int*   tris,             // triangle indices into verts
    int          n_tris,
    int          total_verts,      // LOCAL vertex count (for flag array sizing)
    int          vert_offset,      // subtracted from tris indices for local flags
    float        plane_a,          // cutting plane (0 if closed)
    float        plane_b,
    float        plane_c,
    float        plane_d,          // 0 if closed mesh
    const int*   boundary_a,       // boundary edge endpoints [n_boundary]
    const int*   boundary_b,
    int          n_boundary,
    int          tid,
    HullWorkspace* hull_ws,        // shared memory
    float*       smem,             // shared memory [BLOCK_SIZE]
    DevicePool*  scratch,
    float        rv_k)
{
    if (n_tris == 0) {
        __syncthreads();
        return 0.0f;
    }

    // --- 1. Mesh volume via parallel signed-tet reduction ---
    float local_vol = 0.0f;
    for (int t = tid; t < n_tris; t += BLOCK_SIZE) {
        int v0 = tris[t*3], v1 = tris[t*3+1], v2 = tris[t*3+2];
        local_vol += signed_tet_volume(
            verts[v0*3], verts[v0*3+1], verts[v0*3+2],
            verts[v1*3], verts[v1*3+1], verts[v1*3+2],
            verts[v2*3], verts[v2*3+1], verts[v2*3+2]);
    }
    float mesh_vol_open = block_reduce_sum(local_vol, smem, tid);

    // --- 2. Cap volume via divergence theorem (thread 0) ---
    //
    // Trace boundary loops from edge pairs, compute 2D signed area
    // via shoelace in the plane's projection, then:
    //   V_cap_pos = (d / 3) * |A_net|
    //   V_cap_neg = -(d / 3) * |A_net|
    // We always compute for the POSITIVE half here; caller negates for neg.

    // Allocate boundary-used flags from pool (non-destructive marking)
    __shared__ int* s_be_used;
    __shared__ float s_cap_vol;
    if (tid == 0) {
        s_be_used = (n_boundary > 0) ?
            (int*)pool_alloc(scratch, n_boundary * sizeof(int)) : NULL;
        s_cap_vol = 0.0f;
    }
    __syncthreads();

    // Clear used flags (parallel)
    if (n_boundary > 0 && s_be_used) {
        for (int i = tid; i < n_boundary; i += BLOCK_SIZE)
            s_be_used[i] = 0;
    }
    __syncthreads();

    // Cap volume computation (thread 0)
    if (tid == 0) {
        if (n_boundary > 0 && s_be_used &&
            (plane_a != 0.0f || plane_b != 0.0f || plane_c != 0.0f)) {
            // Choose 2D projection axes (drop dominant normal axis)
            int pu, pv;
            float anx = fabsf(plane_a), any = fabsf(plane_b), anz = fabsf(plane_c);
            if (anx >= any && anx >= anz)      { pu = 1; pv = 2; }
            else if (any >= anz)               { pu = 0; pv = 2; }
            else                               { pu = 0; pv = 1; }

            // Trace boundary loop(s) — chain following with separate used flags
            float area_total = 0.0f;

            for (int start_e = 0; start_e < n_boundary; start_e++) {
                if (s_be_used[start_e]) continue;

                int first_v = boundary_a[start_e];
                int cur = boundary_b[start_e];
                s_be_used[start_e] = 1;

                float loop_area = 0.0f;
                float fu = verts[first_v*3+pu], fv_coord = verts[first_v*3+pv];
                float prev_u = fu, prev_v = fv_coord;

                int safety = n_boundary + 2;
                while (cur != first_v && safety-- > 0) {
                    float cu = verts[cur*3+pu], cv = verts[cur*3+pv];
                    loop_area += prev_u * cv - cu * prev_v;
                    prev_u = cu; prev_v = cv;

                    int next = -1;
                    for (int i = 0; i < n_boundary; i++) {
                        if (s_be_used[i]) continue;
                        if (boundary_a[i] == cur) {
                            next = boundary_b[i];
                            s_be_used[i] = 1;
                            break;
                        }
                        if (boundary_b[i] == cur) {
                            next = boundary_a[i];
                            s_be_used[i] = 1;
                            break;
                        }
                    }
                    if (next < 0) break;
                    cur = next;
                }
                // Close the loop
                loop_area += prev_u * fv_coord - fu * prev_v;
                area_total += loop_area * 0.5f;
            }

            // V_cap = (d / 3) * |A_net|
            s_cap_vol = (plane_d / 3.0f) * fabsf(area_total);
        }
    }
    __syncthreads();

    __shared__ float s_mesh_vol;
    if (tid == 0) s_mesh_vol = mesh_vol_open + s_cap_vol;
    __syncthreads();

    // --- 3. Collect unique vertices from triangles for hull ---
    __shared__ int* s_flags;
    __shared__ float* s_hull_pts;
    __shared__ int s_hull_n;

    if (tid == 0) {
        s_flags = (int*)pool_alloc(scratch, total_verts * sizeof(int));
        s_hull_pts = (float*)pool_alloc(scratch, MAX_HULL_VERTS * 3 * sizeof(float));
        s_hull_n = 0;
    }
    __syncthreads();
    if (!s_flags || !s_hull_pts) {
        // Pool exhausted — fall back to 0
        __syncthreads();
        return 0.0f;
    }

    // Clear flags
    for (int v = tid; v < total_verts; v += BLOCK_SIZE)
        s_flags[v] = 0;
    __syncthreads();

    // Mark vertices referenced by triangles (use local index = global - vert_offset)
    for (int t = tid; t < n_tris; t += BLOCK_SIZE) {
        int i0 = tris[t*3+0] - vert_offset;
        int i1 = tris[t*3+1] - vert_offset;
        int i2 = tris[t*3+2] - vert_offset;
        if (i0 >= 0 && i0 < total_verts) s_flags[i0] = 1;
        if (i1 >= 0 && i1 < total_verts) s_flags[i1] = 1;
        if (i2 >= 0 && i2 < total_verts) s_flags[i2] = 1;
    }
    __syncthreads();

    // Gather marked vertices (capped at MAX_HULL_VERTS)
    for (int v = tid; v < total_verts; v += BLOCK_SIZE) {
        if (s_flags[v]) {
            int idx = atomicAdd(&s_hull_n, 1);
            if (idx < MAX_HULL_VERTS) {
                int gv = vert_offset + v;
                s_hull_pts[idx*3+0] = verts[gv*3+0];
                s_hull_pts[idx*3+1] = verts[gv*3+1];
                s_hull_pts[idx*3+2] = verts[gv*3+2];
            }
        }
    }
    __syncthreads();

    int n_hull = s_hull_n;
    if (n_hull > MAX_HULL_VERTS) n_hull = MAX_HULL_VERTS;

    // --- 4. Convex hull volume ---
    float hull_vol = compute_hull_volume(s_hull_pts, n_hull, tid, hull_ws);

    // --- 5. Rv formula ---
    __shared__ float s_rv;
    if (tid == 0) {
        float diff = fabsf(s_mesh_vol) - hull_vol;
        // hull_vol >= |mesh_vol| always (convex hull encloses mesh)
        // but floating point can invert; take abs of difference
        if (diff < 0.0f) diff = -diff;
        s_rv = cbrtf(3.0f * diff / (4.0f * PI_F)) * rv_k;
    }
    __syncthreads();
    return s_rv;
}

// ============================================================================
// evaluate_candidates
// ============================================================================
// Grid: (num_beam_items * num_planes, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Each block: clip worst part of a beam item by one plane, compute Rv.

__global__ void evaluate_candidates(
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    const PartInfo* __restrict__ parts,
    const BeamItem* __restrict__ beam,
    const float* __restrict__ planes,     // [num_planes * 4]
    int          num_planes,
    int          num_beam_items,
    float        rv_k,
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
    __shared__ int *s_be_a, *s_be_b;  // boundary edges
    if (tid == 0) {
        s_signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
        int max_new_verts = tc * 2;
        int max_out_tris = tc * 3;
        s_all_verts = (float*)pool_alloc(&scratch, (vc + max_new_verts) * 3 * sizeof(float));
        s_pos_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
        s_neg_tris = (int*)pool_alloc(&scratch, max_out_tris * 3 * sizeof(int));
        s_be_a = (int*)pool_alloc(&scratch, tc * sizeof(int));
        s_be_b = (int*)pool_alloc(&scratch, tc * sizeof(int));
    }
    __syncthreads();
    int* signs = s_signs;
    float* all_verts = s_all_verts;
    int* pos_tris = s_pos_tris;
    int* neg_tris = s_neg_tris;
    int* be_a = s_be_a;
    int* be_b = s_be_b;
    if (!signs || !all_verts || !pos_tris || !neg_tris || !be_a || !be_b) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // --- Classify vertices (inlined — see CLAUDE.md deviation #4) ---
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

    __shared__ int pos_cnt, neg_cnt, new_v_cnt, be_cnt;
    if (tid == 0) { pos_cnt = 0; neg_cnt = 0; new_v_cnt = 0; be_cnt = 0; }
    __syncthreads();

    // Copy original vertices
    for (int v = tid; v < vc * 3; v += BLOCK_SIZE)
        all_verts[v] = vertex_pool[vo * 3 + v];
    __syncthreads();

    // Process each triangle — split + collect boundary edges
    for (int t = tid; t < tc; t += BLOCK_SIZE) {
        int li0 = triangle_pool[(to + t) * 3 + 0] - vo;
        int li1 = triangle_pool[(to + t) * 3 + 1] - vo;
        int li2 = triangle_pool[(to + t) * 3 + 2] - vo;
        int s0 = signs[li0], s1 = signs[li1], s2 = signs[li2];

        int hp = (s0 > 0) | (s1 > 0) | (s2 > 0);
        int hn = (s0 < 0) | (s1 < 0) | (s2 < 0);

        if (!hp || !hn) {
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

                // Boundary edge: the two intersection points
                int bei = atomicAdd(&be_cnt, 1);
                be_a[bei] = nv1; be_b[bei] = nv2;

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

                    // Boundary edge: on-plane vertex to intersection point
                    int bei = atomicAdd(&be_cnt, 1);
                    be_a[bei] = ov; be_b[bei] = nvi;

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
    int n_be = be_cnt;

    if (n_pos == 0 || n_neg == 0) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        return;
    }

    // --- Compute Rv for each half ---
    __shared__ float smem[BLOCK_SIZE];
    __shared__ HullWorkspace hull_ws;

    // Positive half: cap_vol uses plane_d directly
    float pos_rv = compute_rv_for_tris(
        all_verts, pos_tris, n_pos, total_verts, 0,
        pa, pb, pc, pd,
        be_a, be_b, n_be,
        tid, &hull_ws, smem, &scratch, rv_k);

    // Negative half: cap_vol negated (negate plane_d)
    float neg_rv = compute_rv_for_tris(
        all_verts, neg_tris, n_neg, total_verts, 0,
        pa, pb, pc, -pd,    // negate d for negative half
        be_a, be_b, n_be,   // same boundary, but divergence theorem sign flips via -pd
        tid, &hull_ws, smem, &scratch, rv_k);

    if (tid == 0) {
        float cut_cost = fmaxf(pos_rv, neg_rv);

        float other_worst = 0.0f;
        for (int p = 0; p < item.num_parts; p++) {
            if (p == worst_part) continue;
            float pc_val = parts[beam_idx * MAX_PARTS_PER_BEAM + p].rv_cost;
            if (pc_val > other_worst) other_worst = pc_val;
        }
        cost_buffer[bid] = fmaxf(cut_cost, other_worst);
    }
}

// ============================================================================
// select_top_k
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
// apply_cuts
// ============================================================================
// Grid: (beam_width, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Each block applies one winning cut to pool_dst, adds cap triangles to close
// meshes, and computes Rv for each new half.

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

    __shared__ int* s_signs;
    __shared__ float* s_av;
    __shared__ int *s_pt, *s_nt;
    __shared__ int *s_be_a, *s_be_b;
    if (tid == 0) {
        s_signs = (int*)pool_alloc(&scratch, vc * sizeof(int));
        int max_nv = tc * 2;
        int max_cap_tris = tc;  // upper bound on boundary loop length
        s_av = (float*)pool_alloc(&scratch, (vc + max_nv) * 3 * sizeof(float));
        s_pt = (int*)pool_alloc(&scratch, (tc * 3 + max_cap_tris) * 3 * sizeof(int));
        s_nt = (int*)pool_alloc(&scratch, (tc * 3 + max_cap_tris) * 3 * sizeof(int));
        s_be_a = (int*)pool_alloc(&scratch, tc * sizeof(int));
        s_be_b = (int*)pool_alloc(&scratch, tc * sizeof(int));
    }
    __syncthreads();
    int* signs = s_signs;
    float* av = s_av;
    int* pt = s_pt;
    int* nt = s_nt;
    int* be_a = s_be_a;
    int* be_b = s_be_b;
    if (!signs || !av || !pt || !nt || !be_a || !be_b) return;

    __shared__ int pc_s, nc_s, nvc_s, be_cnt;
    if (tid == 0) { pc_s = 0; nc_s = 0; nvc_s = 0; be_cnt = 0; }
    __syncthreads();

    for (int v = tid; v < vc * 3; v += BLOCK_SIZE)
        av[v] = vp_src[wo * 3 + v];
    __syncthreads();

    // Classify vertices (inlined)
    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        float vx = av[v * 3 + 0];
        float vy = av[v * 3 + 1];
        float vz = av[v * 3 + 2];
        float val = pa * vx + pb * vy + pc * vz + pd;
        signs[v] = (val > EPS) ? 1 : ((val < -EPS) ? -1 : 0);
    }
    __syncthreads();

    // Split triangles + collect boundary edges
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

                int bei=atomicAdd(&be_cnt,1);
                be_a[bei]=nv1; be_b[bei]=nv2;

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

                    int bei=atomicAdd(&be_cnt,1);
                    be_a[bei]=ov; be_b[bei]=nvi;

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
    int n_be = be_cnt;

    // --- Fan cap triangulation: close both halves ---
    // Thread 0: trace boundary loop, add fan triangles to both sides.
    // Determines winding via shoelace signed area.
    __shared__ int* s_loop_v;
    if (tid == 0) {
        s_loop_v = (n_be > 0) ? (int*)pool_alloc(&scratch, 512 * sizeof(int)) : NULL;
    }
    __syncthreads();

    if (tid == 0 && n_be > 0 && s_loop_v) {
        int* loop_v = s_loop_v;

        // Choose projection axes
        int pu, pv;
        float anx = fabsf(pa), any = fabsf(pb), anz = fabsf(pc);
        if (anx >= any && anx >= anz)      { pu = 1; pv = 2; }
        else if (any >= anz)               { pu = 0; pv = 2; }
        else                               { pu = 0; pv = 1; }

        // Trace boundary loop(s)
        for (int start_e = 0; start_e < n_be; start_e++) {
            if (be_a[start_e] < 0) continue;  // used
            int loop_len = 0;
            int first_v = be_a[start_e];
            int cur = be_b[start_e];
            be_a[start_e] = -1;
            loop_v[loop_len++] = first_v;

            int safety = n_be + 2;
            while (cur != first_v && loop_len < 510 && safety-- > 0) {
                loop_v[loop_len++] = cur;
                int next = -1;
                for (int i = 0; i < n_be; i++) {
                    if (be_a[i] < 0) continue;
                    if (be_a[i] == cur) { next = be_b[i]; be_a[i] = -1; break; }
                    if (be_b[i] == cur) { next = be_a[i]; be_a[i] = -1; break; }
                }
                if (next < 0) break;
                cur = next;
            }

            if (loop_len < 3) continue;

            // Compute shoelace signed area (from +n direction)
            float area2 = 0.0f;
            for (int i = 0; i < loop_len; i++) {
                int j = (i + 1) % loop_len;
                float ui = av[loop_v[i]*3+pu], vi_c = av[loop_v[i]*3+pv];
                float uj = av[loop_v[j]*3+pu], vj = av[loop_v[j]*3+pv];
                area2 += ui * vj - uj * vi_c;
            }
            // area2 > 0 => CCW from +n => fan normal is +n
            // Positive half cap needs normal -n, negative needs +n

            // Add fan cap triangles
            for (int i = 1; i < loop_len - 1; i++) {
                // Positive half: need normal -n
                // If area2 > 0 (CCW/+n), flip winding
                int pi_idx = atomicAdd(&pc_s, 1);
                if (area2 > 0) {
                    pt[pi_idx*3+0]=loop_v[0]; pt[pi_idx*3+1]=loop_v[i+1]; pt[pi_idx*3+2]=loop_v[i];
                } else {
                    pt[pi_idx*3+0]=loop_v[0]; pt[pi_idx*3+1]=loop_v[i]; pt[pi_idx*3+2]=loop_v[i+1];
                }

                // Negative half: need normal +n (opposite of positive)
                int ni_idx = atomicAdd(&nc_s, 1);
                if (area2 > 0) {
                    nt[ni_idx*3+0]=loop_v[0]; nt[ni_idx*3+1]=loop_v[i]; nt[ni_idx*3+2]=loop_v[i+1];
                } else {
                    nt[ni_idx*3+0]=loop_v[0]; nt[ni_idx*3+1]=loop_v[i+1]; nt[ni_idx*3+2]=loop_v[i];
                }
            }
        }
    }
    __syncthreads();

    // --- Compute Rv for new halves ---
    __shared__ float smem_ac[BLOCK_SIZE];
    __shared__ HullWorkspace hull_ws_ac;
    __shared__ float pos_rv_val, neg_rv_val;

    // Positive half Rv (mesh is now closed thanks to cap tris)
    if (pc_s > 0) {
        float prv = compute_rv_for_tris(
            av, pt, pc_s, tv, 0,
            0,0,0,0,  // closed mesh — no cap correction needed
            (int*)0, (int*)0, 0,
            tid, &hull_ws_ac, smem_ac, &scratch, rv_k);
        if (tid == 0) pos_rv_val = prv;
    } else {
        if (tid == 0) pos_rv_val = 0.0f;
    }
    __syncthreads();

    // Negative half Rv
    if (nc_s > 0) {
        float nrv = compute_rv_for_tris(
            av, nt, nc_s, tv, 0,
            0,0,0,0,
            (int*)0, (int*)0, 0,
            tid, &hull_ws_ac, smem_ac, &scratch, rv_k);
        if (tid == 0) neg_rv_val = nrv;
    } else {
        if (tid == 0) neg_rv_val = 0.0f;
    }
    __syncthreads();

    // Write halves to dst pool
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
            dp.rv_cost = fmaxf(pos_rv_val, EPS);
        }
        if (nc_s > 0) {
            PartInfo& dp = parts_dst[bid * MAX_PARTS_PER_BEAM + neg_pi];
            dp.vert_offset = neg_vo; dp.vert_count = tv;
            dp.tri_offset = neg_to; dp.tri_count = nc_s;
            dp.rv_cost = fmaxf(neg_rv_val, EPS);
        }
        beam_dst[bid].num_parts = out_np;
        beam_dst[bid].worst_part_idx = -1;
        beam_dst[bid].worst_cost = 0.0f;
        beam_dst[bid].cut_count = si.cut_count + 1;
    }
}

// ============================================================================
// compute_part_costs
// ============================================================================
// Grid: (num_beam_items, 1, 1), Block: (BLOCK_SIZE, 1, 1)
// Computes Rv for parts with rv_cost == 0, finds worst part per beam item.

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
    __shared__ HullWorkspace hull_ws;
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

        int tc = part.tri_count;
        int to = part.tri_offset;

        if (tc == 0) {
            if (tid == 0) part.rv_cost = EPS;
            __syncthreads();
            continue;
        }

        // Compute Rv for this part (mesh is closed — initial mesh or capped)
        float rv = compute_rv_for_tris(
            vertex_pool, &triangle_pool[to * 3], tc,
            part.vert_count,   // total_verts for flag array
            part.vert_offset,  // subtract from global indices for local flags
            0, 0, 0, 0,        // closed mesh, no cap
            (int*)0, (int*)0, 0,
            tid, &hull_ws, smem, &scratch, rv_k);

        if (tid == 0) {
            part.rv_cost = fmaxf(rv, EPS);
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
