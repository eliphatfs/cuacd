// beam.cu — beam search expansion kernels.
//
// beam_expansion: 3*cuts_per_axis*current->nitems blocks, PC_BLOCK threads each.
// Each block cuts the last part of one WorkItem along an axis-aligned plane,
// adds the two halves to a new WorkItem in next.
//
// beam_hull: 2*current->nitems blocks, 32 threads each.
// Each block computes the convex hull of one of the last two parts of each WorkItem.

#include "plane_cut.cuh"
#include "warp_common.cuh"
#include "hull_dandc.cuh"
#include "kdop_hull.cuh"
#include "mesh_volume.cuh"
#include "warp_sort.cuh"
#include "hausdorff.cuh"

// Error code for exceeding WORK_ITEM_MAX_PARTS (distinct from plane_cut errors).
#define BEAM_ERR_OVERFLOW      0x10000
// Error codes for beam_sort.
#define BEAM_ERR_SORT_OOM      0x20000  // scratch heap allocation failed
#define BEAM_ERR_SORT_STACK    0x40000  // warp_sort_t stack overflow
// Error codes for beam_finalize.
#define BEAM_ERR_FINALIZE_OOM  0x80000  // scratch heap allocation failed

// CoACD k_rv weight: scales the concavity radius rv = cbrt(3/(4pi)*(hull_vol-mesh_vol)).
#define PART_COST_K_RV 0.3f

// Cost of a Part: max(k_rv * rv, hausdorff) where rv converts volume difference to distance.
static __device__ inline float part_cost(const Part& p) {
    const float pi = 3.14159265358979f;
    float rv = cbrtf((3.0f / (4.0f * pi)) * fmaxf(p.hull_vol - p.mesh_vol, 0.0f));
    return fmaxf(PART_COST_K_RV * rv, p.hausdorff);
}

// Comparator for sorting Parts by part_cost, ascending.
struct PartKeyCmp {
    static __device__ inline float key(const Part& p) { return part_cost(p); }
    static __device__ inline int cmp(Part a, Part b) {
        float ka = key(a), kb = key(b);
        if (ka < kb) return -1;
        if (ka > kb) return  1;
        return 0;
    }
    static __device__ inline Part sentinel() {
        Part s = {};
        s.hausdorff = 1e30f;
        s.mesh_vol  = 1e30f;
        s.hull_vol  = 0.0f;
        return s;
    }
};

// beam_initialize: <<<3, 32>>>
// Block 0: compute mesh volume via mesh_volume_warp, store in part 0 mesh_vol.
// Block 1: compute hull volume via mesh_volume_warp, store in part 0 hull_vol.
// Block 2: thread 0 sets up WorkItem 0 part 0 metadata and nitems/nparts.
// Caller must launch block 2 before blocks 0/1 when ordering matters — or use
// a dependency-free layout where blocks 0/1 receive args directly (current impl).
extern "C" __global__ void beam_initialize(
    float*      verts,
    int*        tris,
    int         nv,
    int         nt,
    float*      hull_verts,
    int*        hull_tris,
    int         hull_nv,
    int         hull_nt,
    AlgoState*  current)
{
    int lane = threadIdx.x;  // 0..31

    if (blockIdx.x == 0) {
        Mesh m;
        m.verts    = verts;
        m.tris     = tris;
        m.nv       = nv;
        m.nt       = nt;
        m.refcount = NULL;
        float vol = mesh_volume_warp(&m, lane);
        if (lane == 0) {
            DPRINTF("[init] mesh nv=%d nt=%d verts=%p tris=%p vol=%.6f\n",
                   m.nv, m.nt, m.verts, m.tris, vol);
            current->items[0].parts[0].mesh_vol = vol;
        }
    } else if (blockIdx.x == 1) {
        Mesh h;
        h.verts    = hull_verts;
        h.tris     = hull_tris;
        h.nv       = hull_nv;
        h.nt       = hull_nt;
        h.refcount = NULL;
        float vol = mesh_volume_warp(&h, lane);
        if (lane == 0) {
            DPRINTF("[init] hull nv=%d nt=%d verts=%p tris=%p vol=%.6f\n",
                   h.nv, h.nt, h.verts, h.tris, vol);
            current->items[0].parts[0].hull_vol = vol;
        }
    } else {
        if (lane == 0) {
            Part* p          = &current->items[0].parts[0];
            p->mesh.verts    = verts;
            p->mesh.tris     = tris;
            p->mesh.nv       = nv;
            p->mesh.nt       = nt;
            p->mesh.refcount = NULL;
            p->hull.verts    = hull_verts;
            p->hull.tris     = hull_tris;
            p->hull.nv       = hull_nv;
            p->hull.nt       = hull_nt;
            p->hull.refcount = NULL;
            p->hausdorff     = 0.0f;
            current->items[0].nparts = 1;
            current->nitems          = 1;
        }
    }
}

extern "C" __global__ void beam_expansion(
    AlgoState*  current,
    AlgoState*  next,
    DevicePool* pool,
    int         cuts_per_axis,
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int cpa3     = 3 * cuts_per_axis;
    int item_idx = blockIdx.x / cpa3;
    int cut_idx  = blockIdx.x % cpa3;
    int axis     = cut_idx / cuts_per_axis;
    int slice    = cut_idx % cuts_per_axis;  // 0-based, gives (slice+1)/(cuts_per_axis+1) fraction

    WorkItem* wi   = &current->items[item_idx];
    int       np   = wi->nparts;
    Mesh*     mesh = &wi->parts[np - 1].mesh;

    // Validate mesh before use — detect freed/corrupted meshes early.
    if (tid == 0 && (mesh->nv <= 0 || mesh->nt <= 0 ||
                     mesh->nv > 1000000 || mesh->nt > 1000000 ||
                     mesh->verts == NULL || mesh->tris == NULL)) {
        DPRINTF("[expand] BAD MESH block=%d item=%d np=%d nv=%d nt=%d verts=%p tris=%p rc=%p\n",
               blockIdx.x, item_idx, np, mesh->nv, mesh->nt,
               mesh->verts, mesh->tris, mesh->refcount);
        if (mesh->refcount) DPRINTF("[expand]   *refcount=%d\n", *mesh->refcount);
        atomicOr(err, 0x100000);  // custom error code
        return;
    }

    // Compute bounding box — three-stage reduction:
    //   1. Each thread accumulates its own lo/hi over its strided vertices.
    //   2. warp_min_f / warp_max_f reduces to a per-warp result (lane 0 holds it).
    //   3. Lane 0 of each warp atomicMin/Max into shared memory.
    __shared__ float s_lo[3], s_hi[3];
    if (tid < 3) { s_lo[tid] = 1e30f; s_hi[tid] = -1e30f; }
    __syncthreads();

    int  lane    = tid & 31;
    float tlo[3] = { 1e30f,  1e30f,  1e30f};
    float thi[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = tid; i < mesh->nv; i += blockDim.x) {
        float x = mesh->verts[3*i + 0];
        float y = mesh->verts[3*i + 1];
        float z = mesh->verts[3*i + 2];
        tlo[0] = fminf(tlo[0], x); thi[0] = fmaxf(thi[0], x);
        tlo[1] = fminf(tlo[1], y); thi[1] = fmaxf(thi[1], y);
        tlo[2] = fminf(tlo[2], z); thi[2] = fmaxf(thi[2], z);
    }
    // Warp reduction
    for (int a = 0; a < 3; a++) {
        tlo[a] = warp_min_f(tlo[a]);
        thi[a] = warp_max_f(thi[a]);
    }
    // First lane of each warp writes to shared
    if (lane == 0) {
        atomicMinF(&s_lo[0], tlo[0]); atomicMaxF(&s_hi[0], thi[0]);
        atomicMinF(&s_lo[1], tlo[1]); atomicMaxF(&s_hi[1], thi[1]);
        atomicMinF(&s_lo[2], tlo[2]); atomicMaxF(&s_hi[2], thi[2]);
    }
    __syncthreads();

    // Axis-aligned plane at (slice+1)/(cuts_per_axis+1) fraction of bbox extent
    float pa = (axis == 0) ? 1.f : 0.f;
    float pb = (axis == 1) ? 1.f : 0.f;
    float pc = (axis == 2) ? 1.f : 0.f;
    float pd = -(s_lo[axis] + (slice + 1.f) / (cuts_per_axis + 1.f) * (s_hi[axis] - s_lo[axis]));

    PartPair pp = plane_cut_block(mesh, pa, pb, pc, pd,
                                  &pool->heap, &pool->scratch, err);
    __syncthreads();

    // If either side is empty, discard the cut result and exit
    if (pp.pos.mesh.nv == 0 || pp.neg.mesh.nv == 0) {
        if (tid == 0) {
            if (pp.pos.mesh.verts) heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
            if (pp.neg.mesh.verts) heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
        }
        return;
    }

    // Guard against exceeding part capacity
    if (np + 1 > WORK_ITEM_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, BEAM_ERR_OVERFLOW);
            heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
            heap_free(&pool->heap, (void*)pp.neg.mesh.verts);
        }
        return;
    }

    // Claim a slot in next
    __shared__ int s_idx;
    if (tid == 0) s_idx = atomicAdd(&next->nitems, 1);
    __syncthreads();
    WorkItem* wo = &next->items[s_idx];

    // Copy parts[0..np-2] in parallel (all except the last, which is being replaced)
    int copy_ints = (np - 1) * (int)(sizeof(Part) / sizeof(int));
    int* src = (int*)wi->parts;
    int* dst = (int*)wo->parts;
    for (int i = tid; i < copy_ints; i += blockDim.x)
        dst[i] = src[i];
    __syncthreads();

    // Increment refcounts for the copied parts (parallel over parts)
    for (int i = tid; i < np - 1; i += blockDim.x) {
        if (wo->parts[i].mesh.refcount) atomicAdd(wo->parts[i].mesh.refcount, 1);
        if (wo->parts[i].hull.refcount) atomicAdd(wo->parts[i].hull.refcount, 1);
    }

    // Thread 0 writes the two new parts and nparts
    if (tid == 0) {
        wo->parts[np - 1] = pp.pos;
        wo->parts[np]     = pp.neg;
        wo->nparts        = np + 1;
    }
    __syncthreads();

    // Warp 0 computes pos mesh volume, warp 1 computes neg mesh volume.
    int warp_id = tid / WARP_SIZE;
    int wlane   = tid & (WARP_SIZE - 1);
    if (warp_id == 0) {
        float vol = mesh_volume_warp(&wo->parts[np - 1].mesh, wlane);
        if (wlane == 0) wo->parts[np - 1].mesh_vol = vol;
    } else if (warp_id == 1) {
        float vol = mesh_volume_warp(&wo->parts[np].mesh, wlane);
        if (wlane == 0) wo->parts[np].mesh_vol = vol;
    }
}

// beam_hull: <<<2*current->nitems, KDOP_BLOCK>>>
// Each block computes the approximate convex hull (k-DOP) of one of the last
// two parts of a WorkItem and stores it in Part.hull + Part.hull_vol.
// Block b: item_idx = b / 2, part_offset = b % 2 (0 = nparts-2, 1 = nparts-1).
// Early-exits (no error) if item_idx >= nitems or hull already computed.
extern "C" __global__ void beam_hull(
    AlgoState*  current,
    DevicePool* pool,
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int item_idx = blockIdx.x / 2;
    int part_off = blockIdx.x % 2;  // 0 = second-to-last, 1 = last

    if (item_idx >= current->nitems) return;

    WorkItem* wi = &current->items[item_idx];
    int       np = wi->nparts;
    int part_idx = np - 2 + part_off;
    if (part_idx < 0 || part_idx >= np) return;

    Part* p = &wi->parts[part_idx];
    if (p->hull.verts != NULL) return;  // already computed

    float hvol = 0.0f;
    Mesh hull = kdop_hull_block(
        p->mesh.verts, p->mesh.nv,
        &pool->heap, &pool->scratch,
        &hvol, err);

    if (tid == 0) {
        p->hull     = hull;
        p->hull_vol = hvol;
    }
}

// beam_hausdorff: <<<2*max_expand, HD_BLOCK>>>
// Each block (256 threads) computes the bidirectional Hausdorff distance
// between one part's hull and its mesh. Same block->item mapping as beam_hull.
extern "C" __global__ void beam_hausdorff(
    AlgoState*  current,
    DevicePool* pool,
    int*        err)
{
    if (*err) return;

    int tid      = threadIdx.x;
    int item_idx = blockIdx.x / 2;
    int part_off = blockIdx.x % 2;

    if (item_idx >= current->nitems) return;

    WorkItem* wi = &current->items[item_idx];
    int       np = wi->nparts;
    int part_idx = np - 2 + part_off;
    if (part_idx < 0 || part_idx >= np) return;

    Part* p = &wi->parts[part_idx];
    if (p->hull.verts == NULL) return;  // no hull yet

    float h = hausdorff_block(&p->hull, &p->mesh, &pool->scratch, err);

    if (tid == 0) {
        p->hausdorff = h;
    }
}

// beam_sort: <<<current->nitems, 32>>>
// Each block (one warp) sorts the parts of one WorkItem in ascending order of
// max(hausdorff, mesh_vol / (hull_vol + eps)) using warp_sort_t.
// Scratch is allocated from pool->scratch and freed before return.
// Sets BEAM_ERR_SORT_OOM if scratch allocation fails, BEAM_ERR_SORT_STACK on
// warp_sort_t stack overflow.
extern "C" __global__ void beam_sort(
    AlgoState*  current,
    DevicePool* pool,
    int*        err)
{
    if (*err) return;

    int lane     = threadIdx.x;  // 0..31
    int item_idx = blockIdx.x;

    if (item_idx >= current->nitems) return;

    WorkItem* wi = &current->items[item_idx];
    int np = wi->nparts;
    if (np <= 1) return;

    // Allocate sort scratch from scratch heap (lane 0 only).
    __shared__ char* s_scratch;
    if (lane == 0) {
        int nb = np * (int)sizeof(Part) + WS_MAX_STACK * 2 * (int)sizeof(int);
        if (heap_alloc(&pool->scratch, nb, (void**)&s_scratch) != 0) {
            s_scratch = NULL;
            atomicOr(err, BEAM_ERR_SORT_OOM);
        }
    }
    __syncwarp();
    if (s_scratch == NULL) return;

    int rc = warp_sort_t<Part, PartKeyCmp>(wi->parts, s_scratch, np, lane);
    if (rc != 0 && lane == 0)
        atomicOr(err, BEAM_ERR_SORT_STACK);

    if (lane == 0)
        heap_free(&pool->scratch, s_scratch);
}

// WorkItem sort key: sort by cost of last (worst) part, break ties by index.
struct WIKey { float cost; int idx; };
struct WIKeyCmp {
    static __device__ inline int cmp(WIKey a, WIKey b) {
        if (a.cost < b.cost) return -1;
        if (a.cost > b.cost) return  1;
        return (a.idx < b.idx) ? -1 : (a.idx > b.idx) ? 1 : 0;
    }
    static __device__ inline WIKey sentinel() {
        WIKey s; s.cost = 1e30f; s.idx = 0x7fffffff; return s;
    }
};

// beam_finalize: <<<max_keep, 1024>>>
// Clears prev AlgoState and compacts current to the best max_keep WorkItems.
//
// Phase 1 (all blocks): clear prev->items[blockIdx.x] — decrement mesh/hull
//   refcounts and heap_free chunks that reach zero; thread 0 sets nparts=0.
// Phase 2 (block 0 only):
//   a. Allocate WIKey sort buffer + scratch; fill with {cost-of-last-part, idx};
//      warp 0 sorts ascending. Free sort scratch.
//   b. Allocate WorkItem scratch (max_keep items). Each warp moves one top-k
//      source item (by sorted order) into scratch; set source nparts=0.
//   c. Thread 0 checks best cost vs threshold; sets *finish=1 if below.
//      Frees key buffer.
//   d. Each warp clears one item in current (decrement refcounts, free if 0,
//      set nparts=0) — moved items are already empty (nparts=0), no-op.
//   e. Each warp moves one scratch item back to current->items[0..k-1].
//   f. Thread 0 frees scratch; sets prev->nitems=0, current->nitems=k.
//
// Error codes: BEAM_ERR_FINALIZE_OOM on scratch OOM, BEAM_ERR_SORT_STACK on
// sort stack overflow (both atomicOr'd into *err; early-exits on any error).
extern "C" __global__ void beam_finalize(
    AlgoState*  prev,
    AlgoState*  current,
    DevicePool* pool,
    int*        finish,
    int*        err,
    int         max_keep,
    float       threshold)
{
    if (*err) return;

    int tid       = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;
    int wlane     = tid & (WARP_SIZE - 1);
    int num_warps = blockDim.x / WARP_SIZE;

    // =========================================================================
    // Phase 1: clear prev->items[blockIdx.x]
    // =========================================================================
    {
        WorkItem* wi = &prev->items[blockIdx.x];
        int np = wi->nparts;
        for (int i = tid; i < np; i += blockDim.x) {
            Part* p = &wi->parts[i];
            if (p->mesh.refcount) {
                int old = atomicAdd(p->mesh.refcount, -1);
                if (old <= 0)
                    DPRINTF("[fin-p1] REFCOUNT BUG: prev block=%d part=%d mesh old_rc=%d verts=%p\n",
                           blockIdx.x, i, old, p->mesh.verts);
                if (old == 1) heap_free(&pool->heap, (void*)p->mesh.verts);
            }
            if (p->hull.refcount) {
                int old = atomicAdd(p->hull.refcount, -1);
                if (old <= 0)
                    DPRINTF("[fin-p1] REFCOUNT BUG: prev block=%d part=%d hull old_rc=%d verts=%p\n",
                           blockIdx.x, i, old, p->hull.verts);
                if (old == 1) heap_free(&pool->heap, (void*)p->hull.verts);
            }
        }
        __syncthreads();
        if (tid == 0) wi->nparts = 0;
    }

    if (blockIdx.x != 0) return;
    __syncthreads();  // fence before block-0-only work

    // =========================================================================
    // Block 0 only
    // =========================================================================

    __shared__ int      s_nitems;
    __shared__ int      s_k;
    __shared__ WIKey*   s_keys;
    __shared__ char*    s_key_scratch;
    __shared__ WorkItem* s_scratch_items;

    // Phase 2a: allocate sort buffers
    if (tid == 0) {
        s_nitems = current->nitems;
        s_k      = (max_keep < s_nitems) ? max_keep : s_nitems;
        DPRINTF("[finalize-enter] nitems=%d items_ptr=%p\n", s_nitems, current->items);
        if (s_nitems > 0) {
            WorkItem* wi0 = &current->items[0];
            DPRINTF("[finalize-enter] wi0.nparts=%d wi0.parts[0].mesh_vol=%.6f wi0.parts[0].hull_vol=%.6f\n",
                   wi0->nparts, wi0->parts[0].mesh_vol, wi0->parts[0].hull_vol);
        }
        if (s_nitems == 0) { s_keys = NULL; s_key_scratch = NULL; }
        else {
            int key_bytes  = s_nitems * (int)sizeof(WIKey);
            int sort_bytes = key_bytes + WS_MAX_STACK * 2 * (int)sizeof(int);
            void *p1 = NULL, *p2 = NULL;
            if (heap_alloc(&pool->scratch, (unsigned int)key_bytes,  &p1) != HEAP_OK ||
                heap_alloc(&pool->scratch, (unsigned int)sort_bytes, &p2) != HEAP_OK) {
                if (p1) heap_free(&pool->scratch, p1);
                if (p2) heap_free(&pool->scratch, p2);
                s_keys = NULL; s_key_scratch = NULL;
                atomicOr(err, BEAM_ERR_FINALIZE_OOM);
            } else {
                s_keys        = (WIKey*)p1;
                s_key_scratch = (char*)p2;
            }
        }
    }
    __syncthreads();
    if (*err) return;

    int nitems = s_nitems;
    int k      = s_k;

    if (nitems == 0) {
        if (tid == 0) { prev->nitems = 0; current->nitems = 0; }
        return;
    }

    // Fill key buffer
    for (int i = tid; i < nitems; i += blockDim.x) {
        WorkItem* wi = &current->items[i];
        int np = wi->nparts;
        float cost = (np > 0) ? part_cost(wi->parts[np - 1]) : 0.0f;
        s_keys[i].cost = cost;
        s_keys[i].idx  = i;
    }
    __syncthreads();

    // Sort with warp 0
    if (warp_id == 0) {
        int rc = warp_sort_t<WIKey, WIKeyCmp>(s_keys, s_key_scratch, nitems, wlane);
        if (rc != 0 && wlane == 0)
            atomicOr(err, BEAM_ERR_SORT_STACK);
    }
    __syncthreads();
    if (*err) {
        if (tid == 0) {
            heap_free(&pool->scratch, (void*)s_keys);
            heap_free(&pool->scratch, (void*)s_key_scratch);
        }
        return;
    }

    // Free sort scratch (done sorting)
    if (tid == 0) heap_free(&pool->scratch, (void*)s_key_scratch);

    // Allocate WorkItem scratch
    if (tid == 0) {
        void* p = NULL;
        if (heap_alloc(&pool->scratch, (unsigned int)(max_keep * (int)sizeof(WorkItem)), &p) != HEAP_OK) {
            heap_free(&pool->scratch, (void*)s_keys);
            s_scratch_items = NULL;
            atomicOr(err, BEAM_ERR_FINALIZE_OOM);
        } else {
            s_scratch_items = (WorkItem*)p;
        }
    }
    __syncthreads();
    if (*err) return;

    // Phase 2b: move top-k items from current to scratch (each warp = one item)
    for (int wi_local = warp_id; wi_local < k; wi_local += num_warps) {
        int src_idx     = s_keys[wi_local].idx;
        WorkItem* src   = &current->items[src_idx];
        WorkItem* dst   = &s_scratch_items[wi_local];
        int np          = src->nparts;
        if (wlane == 0) dst->nparts = np;
        __syncwarp();
        int copy_ints   = np * (int)(sizeof(Part) / sizeof(int));
        int* isrc       = (int*)src->parts;
        int* idst       = (int*)dst->parts;
        for (int i = wlane; i < copy_ints; i += WARP_SIZE)
            idst[i] = isrc[i];
        if (wlane == 0) src->nparts = 0;
        __syncwarp();
    }
    __syncthreads();

    // Phase 2c: threshold check + free key buffer
    if (tid == 0) {
        if (k > 0) {
            // Read from scratch (phase 2b already zeroed current->items[s_keys[0].idx])
            WorkItem* best = &s_scratch_items[0];
            int best_np = best->nparts;
            Part* best_last = &best->parts[best_np - 1];
            const float kpi = 3.14159265358979f;
            float best_rv = cbrtf((3.0f / (4.0f * kpi)) * fmaxf(best_last->hull_vol - best_last->mesh_vol, 0.0f));
            DPRINTF("[finalize] nitems=%d k=%d best_idx=%d best_nparts=%d best_cost=%.6f threshold=%.6f "
                   "last_mesh_vol=%.6f last_hull_vol=%.6f last_hausdorff=%.6f last_rv=%.6f\n",
                   nitems, k, s_keys[0].idx, best_np, s_keys[0].cost, threshold,
                   best_last->mesh_vol, best_last->hull_vol, best_last->hausdorff, best_rv);
            for (int pi = 0; pi < best_np; pi++) {
                Part* pp = &best->parts[pi];
                float pp_rv = cbrtf((3.0f / (4.0f * kpi)) * fmaxf(pp->hull_vol - pp->mesh_vol, 0.0f));
                float pp_cost = fmaxf(PART_COST_K_RV * pp_rv, pp->hausdorff);
                DPRINTF("  part[%d]: mesh_vol=%.6f hull_vol=%.6f hausdorff=%.6f rv=%.6f cost=%.6f\n",
                       pi, pp->mesh_vol, pp->hull_vol, pp->hausdorff, pp_rv, pp_cost);
            }
            if (s_keys[0].cost < threshold)
                *finish = 1;
        }
        heap_free(&pool->scratch, (void*)s_keys);
    }

    // Phase 2d: clear all items in current (decrement refcounts; moved items
    // already have nparts=0 so their inner loop is a no-op)
    for (int wi_local = warp_id; wi_local < nitems; wi_local += num_warps) {
        WorkItem* wi = &current->items[wi_local];
        int np = wi->nparts;
        for (int i = wlane; i < np; i += WARP_SIZE) {
            Part* p = &wi->parts[i];
            if (p->mesh.refcount) {
                int old = atomicAdd(p->mesh.refcount, -1);
                if (old <= 0)
                    DPRINTF("[fin-p2d] REFCOUNT BUG: item=%d part=%d mesh old_rc=%d verts=%p\n",
                           wi_local, i, old, p->mesh.verts);
                if (old == 1) heap_free(&pool->heap, (void*)p->mesh.verts);
            }
            if (p->hull.refcount) {
                int old = atomicAdd(p->hull.refcount, -1);
                if (old <= 0)
                    DPRINTF("[fin-p2d] REFCOUNT BUG: item=%d part=%d hull old_rc=%d verts=%p\n",
                           wi_local, i, old, p->hull.verts);
                if (old == 1) heap_free(&pool->heap, (void*)p->hull.verts);
            }
        }
        if (wlane == 0) wi->nparts = 0;
        __syncwarp();
    }
    __syncthreads();

    // Phase 2e: move top-k items from scratch back to current->items[0..k-1]
    for (int wi_local = warp_id; wi_local < k; wi_local += num_warps) {
        WorkItem* src = &s_scratch_items[wi_local];
        WorkItem* dst = &current->items[wi_local];
        int np        = src->nparts;
        if (wlane == 0) dst->nparts = np;
        __syncwarp();
        int copy_ints = np * (int)(sizeof(Part) / sizeof(int));
        int* isrc     = (int*)src->parts;
        int* idst     = (int*)dst->parts;
        for (int i = wlane; i < copy_ints; i += WARP_SIZE)
            idst[i] = isrc[i];
        __syncwarp();
    }
    __syncthreads();

    // Phase 2f: free scratch, update counts
    if (tid == 0) {
        heap_free(&pool->scratch, (void*)s_scratch_items);
        prev->nitems    = 0;
        current->nitems = k;
    }
}
