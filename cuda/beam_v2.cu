// beam_v2.cu — New 3-kernel beam search architecture.
//
// Kernel 1: beam_expansion     (EXPANSION_BLOCK_SIZE=64 threads, 2 warps)
// Kernel 2: beam_hausdorff_parts (HAUSDORFF_BLOCK_SIZE=256 threads)
// Kernel 3: beam_termination   (32 threads, 1 warp)
//
// Uses PartInfoV2 + WorkItem from common.cuh (V2 structs).
// Requires: common.cuh, reduce.cuh, geometry.cuh, hull_warp_common.cuh, hull_dandc.cuh

// ============================================================================
// Helper: allocate a region from DevicePool (lane 0 only, broadcast to warp)
// ============================================================================
__device__ inline void* global_alloc_warp(DevicePool* gpool, int bytes, int lane) {
    long long ptr_ll = 0LL;
    if (lane == 0) {
        unsigned int aligned = ((unsigned int)bytes + 15) & ~15;
        unsigned int old = atomicAdd(gpool->offset, aligned);
        if (old + aligned <= gpool->capacity)
            ptr_ll = (long long)(gpool->base + old);
    }
    ptr_ll = __shfl_sync(WARP_MASK, ptr_ll, 0);
    return (void*)ptr_ll;
}

// Helper: thread-0-only alloc from DevicePool (for block-level use, NOT warp)
__device__ inline void* global_alloc_t0(DevicePool* gpool, int bytes) {
    unsigned int aligned = ((unsigned int)bytes + 15) & ~15;
    unsigned int old = atomicAdd(gpool->offset, aligned);
    if (old + aligned > gpool->capacity) return NULL;
    return gpool->base + old;
}

// ============================================================================
// Kernel 1: beam_expansion
// ============================================================================
// Grid: num_work_items * 3 * cuts_per_axis blocks.
// Block: EXPANSION_BLOCK_SIZE (64) threads = 2 warps.
//
// Each block handles one candidate cut of one work item's worst part.
// Warp 0 -> positive half D&C hull, Warp 1 -> negative half D&C hull.
//
// All scratch memory is allocated from a global DevicePool via atomicAdd.
// No pre-sized per-block scratch — each warp grabs what it needs.

// Kernel error bit flags (written via atomicOr)
#define KERR_SCRATCH_OOM  1   // warppool_from_global OOM
#define KERR_SPLIT_OOM    2   // split scratch alloc failed
#define KERR_HULL_PTS_OOM 4   // hull vertex collect alloc failed
#define KERR_POOL_OOM     8   // vert/tri/part output pool overflow
#define KERR_HULL_ERR    16   // D&C hull internal error

__global__ void beam_expansion(
    // Input pools (read-only)
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    const PartInfoV2* __restrict__ part_pool,
    const WorkItem*   __restrict__ work_items,
    int          num_work_items,
    int          cuts_per_axis,
    float        rv_k,
    // Output pools (atomicAdd allocation)
    float*       __restrict__ out_vertex_pool,
    int*         __restrict__ out_triangle_pool,
    PartInfoV2*  __restrict__ out_part_pool,
    unsigned int* __restrict__ vert_counter,
    unsigned int* __restrict__ tri_counter,
    unsigned int* __restrict__ part_counter,
    // Output pool capacities for bounds checking
    unsigned int vert_pool_cap,
    unsigned int tri_pool_cap,
    unsigned int part_pool_cap,
    // Global scratch pool for split + D&C
    DevicePool   scratch,
    // Candidate output
    WorkItem*    __restrict__ work_scratch,   // [grid_size]
    float*       __restrict__ cost_buffer,    // [grid_size]
    // Selection output
    WorkItem*    __restrict__ out_work_items, // [beam_width]
    int          beam_width,
    // Synchronization
    unsigned int* __restrict__ done_counter,
    // Error output (atomicOr)
    int*         __restrict__ kernel_error)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_planes = 3 * cuts_per_axis;
    int grid_size = num_work_items * num_planes;
    int work_idx = bid / num_planes;
    int plane_idx = bid % num_planes;

    if (bid >= grid_size || work_idx >= num_work_items) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        goto selection;
    }

    {
    const WorkItem& item = work_items[work_idx];
    int worst_local = item.worst_part_idx;
    if (worst_local < 0 || worst_local >= item.num_parts) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        goto selection;
    }

    int worst_part_global = item.part_indices[worst_local];
    const PartInfoV2& part = part_pool[worst_part_global];
    int vc = part.vert_count;
    int tc = part.tri_count;
    int vo = part.vert_offset;
    int to = part.tri_offset;

    // --- Compute per-part bbox from TRIANGLE vertices ---
    __shared__ float part_lo[3], part_hi[3];
    {
        __shared__ float bbox_smem[EXPANSION_BLOCK_SIZE];
        for (int k = 0; k < 3; k++) {
            float lo = 1e30f, hi = -1e30f;
            for (int t = tid; t < tc; t += EXPANSION_BLOCK_SIZE) {
                for (int e = 0; e < 3; e++) {
                    float val = vertex_pool[triangle_pool[(to + t) * 3 + e] * 3 + k];
                    lo = fminf(lo, val); hi = fmaxf(hi, val);
                }
            }
            float rlo = twowarp_reduce_max(-lo, bbox_smem, tid);
            if (tid == 0) part_lo[k] = -rlo;
            float rhi = twowarp_reduce_max(hi, bbox_smem, tid);
            if (tid == 0) part_hi[k] = rhi;
            __syncthreads();
        }
    }

    // Reconstruct plane
    int axis = plane_idx / cuts_per_axis;
    int cut_i = plane_idx % cuts_per_axis;
    float rlo = part_lo[axis], rhi = part_hi[axis];
    float pos = rlo + (rhi - rlo) * (float)(cut_i + 1) / (float)(cuts_per_axis + 1);

    float pa = (axis == 0) ? 1.0f : 0.0f;
    float pb = (axis == 1) ? 1.0f : 0.0f;
    float pc = (axis == 2) ? 1.0f : 0.0f;
    float pd = -pos;

    // --- Allocate split scratch from global pool (thread 0) ---
    int max_new_verts = tc * 2;
    int max_out_tris = tc * 3;
    int split_bytes =
        ((vc * (int)sizeof(int)) + 15) / 16 * 16 +                    // signs
        (((vc + max_new_verts) * 3 * (int)sizeof(float)) + 15) / 16 * 16 + // all_verts
        ((max_out_tris * 3 * (int)sizeof(int)) + 15) / 16 * 16 +      // pos_tris
        ((max_out_tris * 3 * (int)sizeof(int)) + 15) / 16 * 16 +      // neg_tris
        ((tc * (int)sizeof(int)) + 15) / 16 * 16 +                    // be_a
        ((tc * (int)sizeof(int)) + 15) / 16 * 16 +                    // be_b
        ((tc * (int)sizeof(int)) + 15) / 16 * 16;                     // be_used (for cap)

    __shared__ char* s_split_base;
    __shared__ int s_split_ok;
    if (tid == 0) {
        s_split_base = (char*)global_alloc_t0(&scratch, split_bytes);
        s_split_ok = (s_split_base != NULL) ? 1 : 0;
    }
    __syncthreads();

    if (!s_split_ok) {
        if (tid == 0) { cost_buffer[bid] = 1e30f; atomicOr(kernel_error, KERR_SPLIT_OOM); }
        goto selection;
    }

    // Carve split scratch into sub-arrays
    char* sbase = s_split_base;
    int off = 0;
    int* signs = (int*)(sbase + off);         off += ((vc * (int)sizeof(int)) + 15) & ~15;
    float* all_verts = (float*)(sbase + off); off += (((vc + max_new_verts) * 3 * (int)sizeof(float)) + 15) & ~15;
    int* pos_tris = (int*)(sbase + off);      off += ((max_out_tris * 3 * (int)sizeof(int)) + 15) & ~15;
    int* neg_tris = (int*)(sbase + off);      off += ((max_out_tris * 3 * (int)sizeof(int)) + 15) & ~15;
    int* be_a = (int*)(sbase + off);          off += ((tc * (int)sizeof(int)) + 15) & ~15;
    int* be_b = (int*)(sbase + off);          off += ((tc * (int)sizeof(int)) + 15) & ~15;
    int* be_used = (int*)(sbase + off);

    // --- Classify vertices (inlined) ---
    for (int v = tid; v < vc; v += EXPANSION_BLOCK_SIZE) {
        float vx = vertex_pool[(vo + v) * 3 + 0];
        float vy = vertex_pool[(vo + v) * 3 + 1];
        float vz = vertex_pool[(vo + v) * 3 + 2];
        float val = pa * vx + pb * vy + pc * vz + pd;
        signs[v] = (val > EPS) ? 1 : ((val < -EPS) ? -1 : 0);
    }
    __syncthreads();

    // Quick cut check
    __shared__ int has_pos_s, has_neg_s;
    if (tid == 0) { has_pos_s = 0; has_neg_s = 0; }
    __syncthreads();
    for (int v = tid; v < vc; v += EXPANSION_BLOCK_SIZE) {
        if (signs[v] > 0) atomicExch(&has_pos_s, 1);
        if (signs[v] < 0) atomicExch(&has_neg_s, 1);
    }
    __syncthreads();

    if (!has_pos_s || !has_neg_s) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        goto selection;
    }

    // Copy original vertices
    for (int v = tid; v < vc * 3; v += EXPANSION_BLOCK_SIZE)
        all_verts[v] = vertex_pool[vo * 3 + v];
    __syncthreads();

    __shared__ int pos_cnt, neg_cnt, new_v_cnt, be_cnt;
    if (tid == 0) { pos_cnt = 0; neg_cnt = 0; new_v_cnt = 0; be_cnt = 0; }
    __syncthreads();

    // --- Split triangles ---
    for (int t = tid; t < tc; t += EXPANSION_BLOCK_SIZE) {
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
            } else if (hp) {
                int pi = atomicAdd(&pos_cnt, 1);
                pos_tris[pi*3]=li0; pos_tris[pi*3+1]=li1; pos_tris[pi*3+2]=li2;
            } else {
                float e1x=all_verts[li1*3]-all_verts[li0*3],e1y=all_verts[li1*3+1]-all_verts[li0*3+1],e1z=all_verts[li1*3+2]-all_verts[li0*3+2];
                float e2x=all_verts[li2*3]-all_verts[li0*3],e2y=all_verts[li2*3+1]-all_verts[li0*3+1],e2z=all_verts[li2*3+2]-all_verts[li0*3+2];
                float nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x;
                float dot_fn=pa*nx+pb*ny+pc*nz;
                if(dot_fn>0){ int pi=atomicAdd(&pos_cnt,1);pos_tris[pi*3]=li0;pos_tris[pi*3+1]=li1;pos_tris[pi*3+2]=li2;}
                else        { int ni=atomicAdd(&neg_cnt,1);neg_tris[ni*3]=li0;neg_tris[ni*3+1]=li1;neg_tris[ni*3+2]=li2;}
            }
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
                intersect_edge(all_verts[lv*3],all_verts[lv*3+1],all_verts[lv*3+2],all_verts[ov1*3],all_verts[ov1*3+1],all_verts[ov1*3+2],pa,pb,pc,pd,&ix1,&iy1,&iz1);
                intersect_edge(all_verts[lv*3],all_verts[lv*3+1],all_verts[lv*3+2],all_verts[ov2*3],all_verts[ov2*3+1],all_verts[ov2*3+2],pa,pb,pc,pd,&ix2,&iy2,&iz2);
                int base=vc+atomicAdd(&new_v_cnt,2); int nv1=base,nv2=base+1;
                all_verts[nv1*3]=ix1;all_verts[nv1*3+1]=iy1;all_verts[nv1*3+2]=iz1;
                all_verts[nv2*3]=ix2;all_verts[nv2*3+1]=iy2;all_verts[nv2*3+2]=iz2;
                int bei=atomicAdd(&be_cnt,1); be_a[bei]=nv1; be_b[bei]=nv2;
                if(ls>0){ int pi=atomicAdd(&pos_cnt,1);pos_tris[pi*3]=lv;pos_tris[pi*3+1]=nv1;pos_tris[pi*3+2]=nv2;
                          int ni=atomicAdd(&neg_cnt,2);neg_tris[ni*3]=ov1;neg_tris[ni*3+1]=nv2;neg_tris[ni*3+2]=nv1;neg_tris[(ni+1)*3]=ov1;neg_tris[(ni+1)*3+1]=ov2;neg_tris[(ni+1)*3+2]=nv2;
                } else  { int ni=atomicAdd(&neg_cnt,1);neg_tris[ni*3]=lv;neg_tris[ni*3+1]=nv1;neg_tris[ni*3+2]=nv2;
                          int pi=atomicAdd(&pos_cnt,2);pos_tris[pi*3]=ov1;pos_tris[pi*3+1]=nv2;pos_tris[pi*3+2]=nv1;pos_tris[(pi+1)*3]=ov1;pos_tris[(pi+1)*3+1]=ov2;pos_tris[(pi+1)*3+2]=nv2;
                }
            } else {
                int on=-1; for(int k=0;k<3;k++) if(si_a[k]==0){on=k;break;}
                if(on>=0){
                    int ov=vi_a[on],a_i=vi_a[(on+1)%3],b_i=vi_a[(on+2)%3]; int sa=si_a[(on+1)%3];
                    float ix,iy,iz;
                    intersect_edge(all_verts[a_i*3],all_verts[a_i*3+1],all_verts[a_i*3+2],all_verts[b_i*3],all_verts[b_i*3+1],all_verts[b_i*3+2],pa,pb,pc,pd,&ix,&iy,&iz);
                    int nvi=vc+atomicAdd(&new_v_cnt,1);all_verts[nvi*3]=ix;all_verts[nvi*3+1]=iy;all_verts[nvi*3+2]=iz;
                    int bei=atomicAdd(&be_cnt,1);be_a[bei]=ov;be_b[bei]=nvi;
                    if(sa>0){ int pi=atomicAdd(&pos_cnt,1);pos_tris[pi*3]=ov;pos_tris[pi*3+1]=a_i;pos_tris[pi*3+2]=nvi;
                              int ni=atomicAdd(&neg_cnt,1);neg_tris[ni*3]=ov;neg_tris[ni*3+1]=nvi;neg_tris[ni*3+2]=b_i;
                    } else  { int ni=atomicAdd(&neg_cnt,1);neg_tris[ni*3]=ov;neg_tris[ni*3+1]=a_i;neg_tris[ni*3+2]=nvi;
                              int pi=atomicAdd(&pos_cnt,1);pos_tris[pi*3]=ov;pos_tris[pi*3+1]=nvi;pos_tris[pi*3+2]=b_i;
                    }
                } else { int pi=atomicAdd(&pos_cnt,1);pos_tris[pi*3]=li0;pos_tris[pi*3+1]=li1;pos_tris[pi*3+2]=li2; }
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
        goto selection;
    }

    // --- Mesh volumes for both halves ---
    __shared__ float smem_v2[EXPANSION_BLOCK_SIZE];

    float local_vol_pos = 0.0f;
    for (int t = tid; t < n_pos; t += EXPANSION_BLOCK_SIZE) {
        int v0 = pos_tris[t*3], v1 = pos_tris[t*3+1], v2 = pos_tris[t*3+2];
        local_vol_pos += signed_tet_volume(
            all_verts[v0*3],all_verts[v0*3+1],all_verts[v0*3+2],
            all_verts[v1*3],all_verts[v1*3+1],all_verts[v1*3+2],
            all_verts[v2*3],all_verts[v2*3+1],all_verts[v2*3+2]);
    }
    float mesh_vol_pos_open = twowarp_reduce_sum(local_vol_pos, smem_v2, tid);

    float local_vol_neg = 0.0f;
    for (int t = tid; t < n_neg; t += EXPANSION_BLOCK_SIZE) {
        int v0 = neg_tris[t*3], v1 = neg_tris[t*3+1], v2 = neg_tris[t*3+2];
        local_vol_neg += signed_tet_volume(
            all_verts[v0*3],all_verts[v0*3+1],all_verts[v0*3+2],
            all_verts[v1*3],all_verts[v1*3+1],all_verts[v1*3+2],
            all_verts[v2*3],all_verts[v2*3+1],all_verts[v2*3+2]);
    }
    float mesh_vol_neg_open = twowarp_reduce_sum(local_vol_neg, smem_v2, tid);

    // --- Cap volume (thread 0) ---
    __shared__ float s_mesh_vol_pos, s_mesh_vol_neg;
    if (tid == 0) {
        float cap_vol_pos = 0.0f, cap_vol_neg = 0.0f;
        if (n_be > 0 && (pa != 0.0f || pb != 0.0f || pc != 0.0f)) {
            int pu, pv;
            float anx = fabsf(pa), any = fabsf(pb), anz = fabsf(pc);
            if (anx >= any && anx >= anz)      { pu = 1; pv = 2; }
            else if (any >= anz)               { pu = 0; pv = 2; }
            else                               { pu = 0; pv = 1; }
            for (int i = 0; i < n_be; i++) be_used[i] = 0;
            float area_total = 0.0f;
            float tol = 1e-4f;
            for (int start_e = 0; start_e < n_be; start_e++) {
                if (be_used[start_e]) continue;
                int first_v = be_a[start_e]; int cur = be_b[start_e];
                be_used[start_e] = 1;
                float loop_area = 0.0f;
                float fu = all_verts[first_v*3+pu], fv_coord = all_verts[first_v*3+pv];
                float prev_u = fu, prev_v = fv_coord;
                int safety = n_be + 2;
                while (safety-- > 0) {
                    float cu = all_verts[cur*3+pu], cv = all_verts[cur*3+pv];
                    loop_area += prev_u * cv - cu * prev_v;
                    prev_u = cu; prev_v = cv;
                    float dx = fabsf(all_verts[cur*3]-all_verts[first_v*3])
                             + fabsf(all_verts[cur*3+1]-all_verts[first_v*3+1])
                             + fabsf(all_verts[cur*3+2]-all_verts[first_v*3+2]);
                    if (dx < tol && safety < n_be) break;
                    float cx3=all_verts[cur*3],cy3=all_verts[cur*3+1],cz3=all_verts[cur*3+2];
                    int next = -1;
                    for (int i = 0; i < n_be; i++) {
                        if (be_used[i]) continue;
                        float da=fabsf(all_verts[be_a[i]*3]-cx3)+fabsf(all_verts[be_a[i]*3+1]-cy3)+fabsf(all_verts[be_a[i]*3+2]-cz3);
                        if(da<tol){next=be_b[i];be_used[i]=1;break;}
                        float db=fabsf(all_verts[be_b[i]*3]-cx3)+fabsf(all_verts[be_b[i]*3+1]-cy3)+fabsf(all_verts[be_b[i]*3+2]-cz3);
                        if(db<tol){next=be_a[i];be_used[i]=1;break;}
                    }
                    if (next < 0) break;
                    cur = next;
                }
                loop_area += prev_u * fv_coord - fu * prev_v;
                area_total += loop_area * 0.5f;
            }
            cap_vol_pos = (pd / 3.0f) * fabsf(area_total);
            cap_vol_neg = (-pd / 3.0f) * fabsf(area_total);
        }
        if (cap_vol_pos != 0.0f && mesh_vol_pos_open != 0.0f)
            s_mesh_vol_pos = fabsf(mesh_vol_pos_open + copysignf(fabsf(cap_vol_pos), -mesh_vol_pos_open));
        else
            s_mesh_vol_pos = fabsf(mesh_vol_pos_open);
        if (cap_vol_neg != 0.0f && mesh_vol_neg_open != 0.0f)
            s_mesh_vol_neg = fabsf(mesh_vol_neg_open + copysignf(fabsf(cap_vol_neg), -mesh_vol_neg_open));
        else
            s_mesh_vol_neg = fabsf(mesh_vol_neg_open);
    }
    __syncthreads();

    // --- Collect ALL vertices referenced by each half's triangles ---
    // Allocate flag + hull_pts arrays from global pool (thread 0)
    __shared__ int* s_flags;
    __shared__ float* s_hull_pts_pos;
    __shared__ float* s_hull_pts_neg;
    __shared__ int s_n_hull_pos, s_n_hull_neg;

    if (tid == 0) {
        int flag_bytes = ((total_verts * (int)sizeof(int)) + 15) & ~15;
        // Max hull pts per half = total_verts (no limit!)
        int pts_bytes = ((total_verts * 3 * (int)sizeof(float)) + 15) & ~15;
        int collect_bytes = flag_bytes + pts_bytes * 2;
        char* cb = (char*)global_alloc_t0(&scratch, collect_bytes);
        if (cb) {
            s_flags = (int*)cb;
            s_hull_pts_pos = (float*)(cb + flag_bytes);
            s_hull_pts_neg = (float*)(cb + flag_bytes + pts_bytes);
        } else {
            s_flags = NULL;
        }
        s_n_hull_pos = 0;
        s_n_hull_neg = 0;
    }
    __syncthreads();

    if (!s_flags) {
        if (tid == 0) { cost_buffer[bid] = 1e30f; atomicOr(kernel_error, KERR_HULL_PTS_OOM); }
        goto selection;
    }

    // Collect positive half vertices (thread 0 sequential — no limit)
    if (tid == 0) {
        for (int v = 0; v < total_verts; v++) s_flags[v] = 0;
        for (int t = 0; t < n_pos; t++) {
            int i0=pos_tris[t*3],i1=pos_tris[t*3+1],i2=pos_tris[t*3+2];
            if(i0>=0&&i0<total_verts)s_flags[i0]=1;
            if(i1>=0&&i1<total_verts)s_flags[i1]=1;
            if(i2>=0&&i2<total_verts)s_flags[i2]=1;
        }
        for (int v = 0; v < total_verts; v++) {
            if (s_flags[v]) {
                s_hull_pts_pos[s_n_hull_pos*3+0]=all_verts[v*3+0];
                s_hull_pts_pos[s_n_hull_pos*3+1]=all_verts[v*3+1];
                s_hull_pts_pos[s_n_hull_pos*3+2]=all_verts[v*3+2];
                s_n_hull_pos++;
            }
        }
        // Collect negative half vertices
        for (int v = 0; v < total_verts; v++) s_flags[v] = 0;
        for (int t = 0; t < n_neg; t++) {
            int i0=neg_tris[t*3],i1=neg_tris[t*3+1],i2=neg_tris[t*3+2];
            if(i0>=0&&i0<total_verts)s_flags[i0]=1;
            if(i1>=0&&i1<total_verts)s_flags[i1]=1;
            if(i2>=0&&i2<total_verts)s_flags[i2]=1;
        }
        for (int v = 0; v < total_verts; v++) {
            if (s_flags[v]) {
                s_hull_pts_neg[s_n_hull_neg*3+0]=all_verts[v*3+0];
                s_hull_pts_neg[s_n_hull_neg*3+1]=all_verts[v*3+1];
                s_hull_pts_neg[s_n_hull_neg*3+2]=all_verts[v*3+2];
                s_n_hull_neg++;
            }
        }
    }
    __syncthreads();

    // --- D&C hull for both halves (2 warps, each grabs its own WarpPool) ---
    int warp_id = tid >> 5;
    int lane = tid & 31;

    __shared__ float hull_vol_pos, hull_vol_neg;
    __shared__ int hull_err_pos, hull_err_neg;
    __shared__ int hull_nv_pos, hull_nt_pos, hull_nv_neg, hull_nt_neg;
    __shared__ unsigned int hull_vo_pos, hull_to_pos, hull_vo_neg, hull_to_neg;

    __shared__ int s_hull_pool_ok;
    if (tid == 0) {
        hull_vol_pos = 0.0f; hull_vol_neg = 0.0f;
        hull_err_pos = 0; hull_err_neg = 0;
        hull_nv_pos = 0; hull_nt_pos = 0;
        hull_nv_neg = 0; hull_nt_neg = 0;
        // Allocate hull mesh output space using natural Euler bounds:
        // n input points -> max n hull verts, max 2n-4 hull tris
        int max_hv_pos = s_n_hull_pos;
        int max_ht_pos = (s_n_hull_pos >= 4) ? (2 * s_n_hull_pos - 4) : 0;
        int max_hv_neg = s_n_hull_neg;
        int max_ht_neg = (s_n_hull_neg >= 4) ? (2 * s_n_hull_neg - 4) : 0;
        hull_vo_pos = atomicAdd(vert_counter, max_hv_pos + max_hv_neg);
        hull_to_pos = atomicAdd(tri_counter, max_ht_pos + max_ht_neg);
        hull_vo_neg = hull_vo_pos + max_hv_pos;
        hull_to_neg = hull_to_pos + max_ht_pos;
        // Bounds check
        s_hull_pool_ok = 1;
        if (hull_vo_pos + (unsigned int)(max_hv_pos + max_hv_neg) > vert_pool_cap ||
            hull_to_pos + (unsigned int)(max_ht_pos + max_ht_neg) > tri_pool_cap) {
            s_hull_pool_ok = 0;
            atomicOr(kernel_error, KERR_POOL_OOM);
        }
    }
    __syncthreads();

    // Warp 0: positive half D&C hull
    if (warp_id == 0 && s_n_hull_pos >= 4 && s_hull_pool_ok) {
        // Allocate WarpPool from global scratch
        int needed = dandc_scratch_bytes(s_n_hull_pos);
        WarpPool wp;
        if (lane == 0) warppool_from_global(&scratch, needed, &wp);
        // Broadcast wp fields
        {
            long long b = __shfl_sync(WARP_MASK, (long long)wp.base, 0);
            int o = __shfl_sync(WARP_MASK, wp.offset, 0);
            int c = __shfl_sync(WARP_MASK, wp.capacity, 0);
            int e = __shfl_sync(WARP_MASK, wp.error, 0);
            wp.base = (char*)b; wp.offset = o; wp.capacity = c; wp.error = e;
        }
        if (wp.error) {
            if (lane == 0) atomicOr(kernel_error, KERR_SCRATCH_OOM);
        } else {
            int err0 = 0;
            float* hv = out_vertex_pool + (long long)hull_vo_pos * 3;
            int* ht = out_triangle_pool + (long long)hull_to_pos * 3;
            int nhv = 0, nht = 0;
            int max_hv_p = s_n_hull_pos;
            int max_ht_p = (s_n_hull_pos >= 4) ? (2 * s_n_hull_pos - 4) : 0;
            float vol = hull_dandc_warp_mesh(
                s_hull_pts_pos, s_n_hull_pos, lane, &wp, &err0,
                hv, ht, max_hv_p, max_ht_p,
                &nhv, &nht);
            if (lane == 0) {
                hull_vol_pos = vol;
                hull_err_pos = wp.error ? wp.error : err0;
                hull_nv_pos = nhv; hull_nt_pos = nht;
                if (hull_err_pos) atomicOr(kernel_error, KERR_HULL_ERR);
            }
        }
    }

    // Warp 1: negative half D&C hull
    if (warp_id == 1 && s_n_hull_neg >= 4 && s_hull_pool_ok) {
        int needed = dandc_scratch_bytes(s_n_hull_neg);
        WarpPool wp;
        if (lane == 0) warppool_from_global(&scratch, needed, &wp);
        {
            long long b = __shfl_sync(WARP_MASK, (long long)wp.base, 0);
            int o = __shfl_sync(WARP_MASK, wp.offset, 0);
            int c = __shfl_sync(WARP_MASK, wp.capacity, 0);
            int e = __shfl_sync(WARP_MASK, wp.error, 0);
            wp.base = (char*)b; wp.offset = o; wp.capacity = c; wp.error = e;
        }
        if (wp.error) {
            if (lane == 0) atomicOr(kernel_error, KERR_SCRATCH_OOM);
        } else {
            int err1 = 0;
            float* hv = out_vertex_pool + (long long)hull_vo_neg * 3;
            int* ht = out_triangle_pool + (long long)hull_to_neg * 3;
            int nhv = 0, nht = 0;
            int max_hv_n = s_n_hull_neg;
            int max_ht_n = (s_n_hull_neg >= 4) ? (2 * s_n_hull_neg - 4) : 0;
            float vol = hull_dandc_warp_mesh(
                s_hull_pts_neg, s_n_hull_neg, lane, &wp, &err1,
                hv, ht, max_hv_n, max_ht_n,
                &nhv, &nht);
            if (lane == 0) {
                hull_vol_neg = vol;
                hull_err_neg = wp.error ? wp.error : err1;
                hull_nv_neg = nhv; hull_nt_neg = nht;
                if (hull_err_neg) atomicOr(kernel_error, KERR_HULL_ERR);
            }
        }
    }
    __syncthreads();

    // --- Rebase hull triangle indices from 0-based to absolute pool indices ---
    // hull_dandc_warp_mesh writes 0-based tri indices, but Hausdorff needs absolute.
    if (s_hull_pool_ok) {
        for (int i = tid; i < hull_nt_pos * 3; i += EXPANSION_BLOCK_SIZE)
            out_triangle_pool[(long long)hull_to_pos * 3 + i] += (int)hull_vo_pos;
        for (int i = tid; i < hull_nt_neg * 3; i += EXPANSION_BLOCK_SIZE)
            out_triangle_pool[(long long)hull_to_neg * 3 + i] += (int)hull_vo_neg;
    }
    __syncthreads();

    // --- Compute Rv ---
    __shared__ float pos_rv, neg_rv;
    if (tid == 0) {
        float hvp = (hull_err_pos == 0 && hull_vol_pos > 0.0f) ? hull_vol_pos : s_mesh_vol_pos;
        float hvn = (hull_err_neg == 0 && hull_vol_neg > 0.0f) ? hull_vol_neg : s_mesh_vol_neg;
        pos_rv = rv_from_volumes(s_mesh_vol_pos, hvp, rv_k);
        neg_rv = rv_from_volumes(s_mesh_vol_neg, hvn, rv_k);
    }
    __syncthreads();

    // --- Copy mesh to output pool, create PartInfoV2 entries ---
    __shared__ unsigned int out_vo_pos, out_to_pos, out_vo_neg, out_to_neg;
    __shared__ unsigned int out_pi_pos, out_pi_neg;
    __shared__ int s_mesh_pool_ok;

    if (tid == 0) {
        out_vo_pos = atomicAdd(vert_counter, total_verts);
        out_to_pos = atomicAdd(tri_counter, n_pos);
        out_vo_neg = atomicAdd(vert_counter, total_verts);
        out_to_neg = atomicAdd(tri_counter, n_neg);
        out_pi_pos = atomicAdd(part_counter, 2);
        out_pi_neg = out_pi_pos + 1;
        s_mesh_pool_ok = 1;
        if (out_vo_pos + (unsigned int)total_verts > vert_pool_cap ||
            out_vo_neg + (unsigned int)total_verts > vert_pool_cap ||
            out_to_pos + (unsigned int)n_pos > tri_pool_cap  ||
            out_to_neg + (unsigned int)n_neg > tri_pool_cap  ||
            out_pi_pos + 1u >= part_pool_cap) {
            s_mesh_pool_ok = 0;
            atomicOr(kernel_error, KERR_POOL_OOM);
        }
    }
    __syncthreads();

    if (!s_mesh_pool_ok) {
        if (tid == 0) cost_buffer[bid] = 1e30f;
        goto selection;
    }

    for (int v = tid; v < total_verts * 3; v += EXPANSION_BLOCK_SIZE) {
        out_vertex_pool[out_vo_pos * 3 + v] = all_verts[v];
        out_vertex_pool[out_vo_neg * 3 + v] = all_verts[v];
    }
    for (int t = tid; t < n_pos; t += EXPANSION_BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            out_triangle_pool[(out_to_pos+t)*3+k] = pos_tris[t*3+k] + out_vo_pos;
    for (int t = tid; t < n_neg; t += EXPANSION_BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            out_triangle_pool[(out_to_neg+t)*3+k] = neg_tris[t*3+k] + out_vo_neg;
    __syncthreads();

    if (tid == 0) {
        PartInfoV2* pp = &out_part_pool[out_pi_pos];
        pp->vert_offset = out_vo_pos; pp->vert_count = total_verts;
        pp->tri_offset = out_to_pos;  pp->tri_count = n_pos;
        pp->hull_vert_offset = hull_vo_pos; pp->hull_vert_count = hull_nv_pos;
        pp->hull_tri_offset = hull_to_pos;  pp->hull_tri_count = hull_nt_pos;
        for (int k = 0; k < 6; k++) pp->bbox[k] = 0.0f;
        pp->rv_cost = fmaxf(pos_rv, EPS);
        pp->hausdorff = -1.0f;
        pp->mesh_volume = s_mesh_vol_pos;
        pp->hull_volume = hull_vol_pos;

        PartInfoV2* pn = &out_part_pool[out_pi_neg];
        pn->vert_offset = out_vo_neg; pn->vert_count = total_verts;
        pn->tri_offset = out_to_neg;  pn->tri_count = n_neg;
        pn->hull_vert_offset = hull_vo_neg; pn->hull_vert_count = hull_nv_neg;
        pn->hull_tri_offset = hull_to_neg;  pn->hull_tri_count = hull_nt_neg;
        for (int k = 0; k < 6; k++) pn->bbox[k] = 0.0f;
        pn->rv_cost = fmaxf(neg_rv, EPS);
        pn->hausdorff = -1.0f;
        pn->mesh_volume = s_mesh_vol_neg;
        pn->hull_volume = hull_vol_neg;

        // Build candidate WorkItem
        WorkItem* cand = &work_scratch[bid];
        cand->num_parts = 0;
        for (int p = 0; p < item.num_parts; p++) {
            if (p == worst_local) continue;
            cand->part_indices[cand->num_parts++] = item.part_indices[p];
        }
        cand->part_indices[cand->num_parts++] = out_pi_pos;
        cand->part_indices[cand->num_parts++] = out_pi_neg;

        // Cost = max of all parts' Rv
        float cut_cost = fmaxf(pos_rv, neg_rv);
        float other_worst = 0.0f;
        for (int p = 0; p < item.num_parts; p++) {
            if (p == worst_local) continue;
            float pc_val = part_pool[item.part_indices[p]].rv_cost;
            if (pc_val > other_worst) other_worst = pc_val;
        }
        cost_buffer[bid] = fmaxf(cut_cost, other_worst);

        // Find worst part for the candidate
        float worst = 0.0f; int widx = 0;
        for (int p = 0; p < cand->num_parts; p++) {
            int gi = cand->part_indices[p];
            float rv = out_part_pool[gi].rv_cost;
            if (rv > worst) { worst = rv; widx = p; }
        }
        cand->worst_part_idx = widx;
        cand->worst_metric = worst;
    }
    }  // end of main scope

selection:
    __syncthreads();

    // Last-block selection via atomicAdd on done_counter
    if (tid == 0) {
        unsigned int count = atomicAdd(done_counter, 1) + 1;
        if (count == (unsigned int)grid_size) {
            // Select top beam_width candidates by insertion sort
            for (int k = 0; k < beam_width; k++) {
                out_work_items[k].num_parts = 0;
                out_work_items[k].worst_metric = 1e30f;
            }
            for (int c = 0; c < grid_size; c++) {
                float cost = cost_buffer[c];
                if (cost >= 1e29f) continue;
                if (cost < out_work_items[beam_width - 1].worst_metric) {
                    int at = beam_width - 1;
                    for (int k = 0; k < beam_width; k++) {
                        if (cost < out_work_items[k].worst_metric) { at = k; break; }
                    }
                    for (int k = beam_width - 1; k > at; k--)
                        out_work_items[k] = out_work_items[k-1];
                    out_work_items[at] = work_scratch[c];
                }
            }
            *done_counter = 0;
        }
    }
}

// ============================================================================
// Kernel 2: beam_hausdorff_parts
// ============================================================================
__global__ void beam_hausdorff_parts(
    const float* __restrict__ vertex_pool,
    const int*   __restrict__ triangle_pool,
    PartInfoV2*  __restrict__ part_pool,
    const int*   __restrict__ part_indices,
    int          total_parts,
    float        threshold,
    int*         __restrict__ kernel_error)
{
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    if (pid >= total_parts) return;

    int part_idx = part_indices[pid];
    PartInfoV2& part = part_pool[part_idx];

    if (part.hausdorff >= 0.0f) return;
    if (part.rv_cost > threshold * 2.0f) {
        if (tid == 0) part.hausdorff = part.rv_cost;
        return;
    }

    int tc = part.tri_count, to = part.tri_offset;
    int hull_tc = part.hull_tri_count, hull_to = part.hull_tri_offset;

    if (tc == 0 || hull_tc == 0) {
        if (tid == 0) part.hausdorff = 0.0f;
        return;
    }

    __shared__ float smem[HAUSDORFF_BLOCK_SIZE];
    float my_max_dist = 0.0f;

    // Forward: original mesh vertices -> hull mesh
    for (int t = tid; t < tc; t += HAUSDORFF_BLOCK_SIZE) {
        for (int e = 0; e < 3; e++) {
            int vi = triangle_pool[(to + t) * 3 + e];
            float px=vertex_pool[vi*3],py=vertex_pool[vi*3+1],pz=vertex_pool[vi*3+2];
            float min_d = 1e30f;
            for (int ht = 0; ht < hull_tc; ht++) {
                int hv0=triangle_pool[(hull_to+ht)*3],hv1=triangle_pool[(hull_to+ht)*3+1],hv2=triangle_pool[(hull_to+ht)*3+2];
                float d = point_triangle_dist(px,py,pz,
                    vertex_pool[hv0*3],vertex_pool[hv0*3+1],vertex_pool[hv0*3+2],
                    vertex_pool[hv1*3],vertex_pool[hv1*3+1],vertex_pool[hv1*3+2],
                    vertex_pool[hv2*3],vertex_pool[hv2*3+1],vertex_pool[hv2*3+2]);
                min_d = fminf(min_d, d);
            }
            my_max_dist = fmaxf(my_max_dist, min_d);
        }
    }

    // Backward: hull mesh vertices -> original mesh
    for (int t = tid; t < hull_tc; t += HAUSDORFF_BLOCK_SIZE) {
        for (int e = 0; e < 3; e++) {
            int vi = triangle_pool[(hull_to + t) * 3 + e];
            float px=vertex_pool[vi*3],py=vertex_pool[vi*3+1],pz=vertex_pool[vi*3+2];
            float min_d = 1e30f;
            for (int ot = 0; ot < tc; ot++) {
                int ov0=triangle_pool[(to+ot)*3],ov1=triangle_pool[(to+ot)*3+1],ov2=triangle_pool[(to+ot)*3+2];
                float d = point_triangle_dist(px,py,pz,
                    vertex_pool[ov0*3],vertex_pool[ov0*3+1],vertex_pool[ov0*3+2],
                    vertex_pool[ov1*3],vertex_pool[ov1*3+1],vertex_pool[ov1*3+2],
                    vertex_pool[ov2*3],vertex_pool[ov2*3+1],vertex_pool[ov2*3+2]);
                min_d = fminf(min_d, d);
            }
            my_max_dist = fmaxf(my_max_dist, min_d);
        }
    }

    float hausdorff = block_reduce_max(my_max_dist, smem, tid);
    if (tid == 0) part.hausdorff = hausdorff;
}

// ============================================================================
// Kernel 3: beam_termination
// ============================================================================
__global__ void beam_termination(
    const PartInfoV2* __restrict__ part_pool,
    WorkItem*         __restrict__ work_items,
    int               num_work_items,
    float             threshold,
    int*              __restrict__ result,
    int*              __restrict__ kernel_error)
{
    int wid = blockIdx.x;
    int lane = threadIdx.x;
    if (wid >= num_work_items) return;

    WorkItem& item = work_items[wid];
    int np = item.num_parts;

    float my_worst = 0.0f;
    int my_worst_idx = 0;
    int my_all_ok = 1;

    for (int p = lane; p < np; p += WARP_SIZE) {
        int gi = item.part_indices[p];
        float rv = part_pool[gi].rv_cost;
        float hd = part_pool[gi].hausdorff;
        float metric = (hd >= 0.0f) ? fmaxf(rv, hd) : rv;
        if (metric > my_worst) { my_worst = metric; my_worst_idx = p; }
        if (metric >= threshold) my_all_ok = 0;
    }

    int terminated = __all_sync(WARP_MASK, my_all_ok);

    if (terminated) {
        if (lane == 0) atomicExch(result, wid);
    } else {
        warp_argmax_f(&my_worst, &my_worst_idx);
        if (lane == 0) {
            item.worst_part_idx = my_worst_idx;
            item.worst_metric = my_worst;
        }
    }
}
