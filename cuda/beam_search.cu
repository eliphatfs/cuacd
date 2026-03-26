// Beam search kernels: evaluate candidates, select top-k, apply cuts,
// compute part costs.
// Requires: common.cuh, geometry.cuh

// ============================================================================
// evaluate_candidates
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

    // Classify vertices (inlined)
    for (int v = tid; v < vc; v += BLOCK_SIZE) {
        float vx = av[v * 3 + 0];
        float vy = av[v * 3 + 1];
        float vz = av[v * 3 + 2];
        float val = pa * vx + pb * vy + pc * vz + pd;
        signs[v] = (val > EPS) ? 1 : ((val < -EPS) ? -1 : 0);
    }
    __syncthreads();

    // Split triangles
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
// compute_part_costs
// ============================================================================
// Grid: (num_beam_items, 1, 1), Block: (BLOCK_SIZE, 1, 1)

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

        int tc = part.tri_count;
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
