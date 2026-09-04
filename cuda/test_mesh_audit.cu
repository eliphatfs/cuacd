// test_mesh_audit.cu — __global__ wrapper for mesh_audit_block.
//
// One block per mesh. Caller supplies a scratch-heap region sized for this
// mesh's edge table; verdict is written to out_flags[blockIdx.x].
#include "common.cuh"
#include "mesh_audit.cuh"

extern "C" __global__ void mesh_audit_kernel(
    DevicePool* __restrict__ pool,
    const float* __restrict__ verts,
    const int*   __restrict__ tris,
    int nv, int nt,
    unsigned int* __restrict__ out_flags,
    int* __restrict__ kernel_error)
{
    __shared__ AuditEdgeRec* s_recs;
    __shared__ unsigned int* s_slots;
    __shared__ unsigned int* s_counters;

    const unsigned int cap = mesh_audit_edge_capacity(nt);
    if (threadIdx.x == 0) {
        void* p1 = NULL; void* p2 = NULL; void* p3 = NULL;
        int rc1 = heap_alloc(&pool->scratch, cap * sizeof(AuditEdgeRec), &p1);
        int rc2 = heap_alloc(&pool->scratch, cap * sizeof(unsigned int), &p2);
        int rc3 = heap_alloc(&pool->scratch, 8u * sizeof(unsigned int), &p3);
        if (rc1 != HEAP_OK || rc2 != HEAP_OK || rc3 != HEAP_OK || !p1 || !p2 || !p3) {
            if (kernel_error) atomicOr(kernel_error, 1);
            s_recs = NULL; s_slots = NULL; s_counters = NULL;
        } else {
            s_recs     = (AuditEdgeRec*)p1;
            s_slots    = (unsigned int*)p2;
            s_counters = (unsigned int*)p3;
        }
    }
    __syncthreads();
    if (!s_recs) return;

    // Zero slots + counters cooperatively.
    for (unsigned int i = threadIdx.x; i < cap; i += blockDim.x) s_slots[i] = 0u;
    if (threadIdx.x == 0) for (int i = 0; i < 8; i++) s_counters[i] = 0u;
    __syncthreads();

    mesh_audit_block(verts, tris, nv, nt, s_recs, s_slots, s_counters, cap,
                     out_flags + blockIdx.x);

    __syncthreads();
    if (threadIdx.x == 0) {
        heap_free(&pool->scratch, s_recs);
        heap_free(&pool->scratch, s_slots);
        heap_free(&pool->scratch, s_counters);
    }
}
