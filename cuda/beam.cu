// beam.cu — beam search expansion kernel.
//
// beam_expansion: 3*cuts_per_axis*current->nitems blocks, PC_BLOCK threads each.
// Each block cuts the last part of one WorkItem along an axis-aligned plane,
// adds the two halves to a new WorkItem in next.

#include "plane_cut.cuh"
#include "warp_common.cuh"

// Error code for exceeding WORK_ITEM_MAX_PARTS (distinct from plane_cut errors).
#define BEAM_ERR_OVERFLOW 0x10000

extern "C" __global__ void beam_expansion(
    AlgoState*  current,
    AlgoState*  next,
    DevicePool* pool,
    int         cuts_per_axis,
    int*        err)
{
    int tid      = threadIdx.x;
    int cpa3     = 3 * cuts_per_axis;
    int item_idx = blockIdx.x / cpa3;
    int cut_idx  = blockIdx.x % cpa3;
    int axis     = cut_idx / cuts_per_axis;
    int slice    = cut_idx % cuts_per_axis;  // 0-based, gives (slice+1)/(cuts_per_axis+1) fraction

    WorkItem* wi   = &current->items[item_idx];
    int       np   = wi->nparts;
    Mesh*     mesh = &wi->parts[np - 1].mesh;

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
            void* chunk = pp.pos.mesh.verts
                          ? (void*)pp.pos.mesh.verts
                          : (void*)pp.neg.mesh.verts;
            heap_free(&pool->heap, chunk);
        }
        return;
    }

    // Guard against exceeding part capacity
    if (np + 1 > WORK_ITEM_MAX_PARTS) {
        if (tid == 0) {
            atomicOr(err, BEAM_ERR_OVERFLOW);
            heap_free(&pool->heap, (void*)pp.pos.mesh.verts);
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

    // Thread 0 writes the two new parts and nparts
    if (tid == 0) {
        wo->parts[np - 1] = pp.pos;
        wo->parts[np]     = pp.neg;
        wo->nparts        = np + 1;
    }
}
