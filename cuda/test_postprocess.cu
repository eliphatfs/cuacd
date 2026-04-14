// test_postprocess.cu — Test kernel wrapping decompose_components_block.
//
// Kernels:
//   test_postprocess_dc_kernel — one block (128 threads)
//
// Builds a heap-allocated input Part from raw host-supplied verts/tris,
// calls decompose_components_block, copies each component's mesh to flat
// output buffers, then frees the component allocations.

#include "postprocess.cuh"

// Maximum components the test kernel can handle (shared-memory array).
#define DC_MAX_COMP 32

extern "C" __global__ void test_postprocess_dc_kernel(
    const float* __restrict__ in_verts,
    const int*   __restrict__ in_tris,
    int          n_verts,
    int          n_tris,
    float*       __restrict__ out_verts,         // flat [max_comp * max_verts_per * 3]
    int*         __restrict__ out_tris,          // flat [max_comp * max_tris_per * 3]
    int*         __restrict__ out_nv,            // [max_comp]
    int*         __restrict__ out_nt,            // [max_comp]
    int*         __restrict__ out_n_components,  // [1]
    int          max_components,
    int          max_verts_per,
    int          max_tris_per,
    DeviceHeap*  heap,
    DeviceHeap*  scratch_heap,
    int*         __restrict__ kernel_error)
{
    __shared__ Part  s_parts[DC_MAX_COMP];
    __shared__ Part  s_input_part;

    int tid = threadIdx.x;

    // -------------------------------------------------------------------------
    // Step 1: Build a heap-allocated input Part (thread 0).
    // Layout: [verts (16-byte aligned) | tris (16-byte aligned) | refcount (16B)]
    // -------------------------------------------------------------------------
    if (tid == 0) {
        size_t vb = (size_t)n_verts * 3 * sizeof(float);
        size_t va = DC_ALIGN16(vb);
        size_t tb = (size_t)n_tris  * 3 * sizeof(int);
        size_t ta = DC_ALIGN16(tb);
        size_t rb = 16;

        void* chunk = NULL;
        if (heap_alloc(heap, (unsigned int)(va + ta + rb), &chunk) != HEAP_OK) {
            if (kernel_error) atomicMax(kernel_error, 1);
        } else {
            float* ov = (float*)chunk;
            int*   ot = (int*)((char*)chunk + va);
            int*   rc = (int*)((char*)chunk + va + ta);

            // Copy verts and tris into the heap allocation
            for (int i = 0; i < n_verts * 3; i++) ov[i] = in_verts[i];
            for (int i = 0; i < n_tris  * 3; i++) ot[i] = in_tris[i];
            *rc = 1;

            s_input_part.mesh.verts    = ov;
            s_input_part.mesh.tris     = ot;
            s_input_part.mesh.nv       = n_verts;
            s_input_part.mesh.nt       = n_tris;
            s_input_part.mesh.refcount = rc;

            s_input_part.hull.verts    = NULL;
            s_input_part.hull.tris     = NULL;
            s_input_part.hull.nv       = 0;
            s_input_part.hull.nt       = 0;
            s_input_part.hull.refcount = NULL;

            s_input_part.mesh_vol  = 0.0f;
            s_input_part.hull_vol  = 0.0f;
            s_input_part.hausdorff = 0.0f;
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 3: Call decompose_components_block.
    // -------------------------------------------------------------------------
    int max_out = (max_components < DC_MAX_COMP) ? max_components : DC_MAX_COMP;
    int n_comp = decompose_components_block(
        &s_input_part, s_parts, max_out,
        heap, scratch_heap, kernel_error);
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 5: Write counts (thread 0).
    // -------------------------------------------------------------------------
    if (tid == 0) {
        *out_n_components = n_comp;
        for (int c = 0; c < n_comp; c++) {
            out_nv[c] = s_parts[c].mesh.nv;
            out_nt[c] = s_parts[c].mesh.nt;
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 7: Parallel copy output data (all DC_BLOCK threads).
    // Each component is copied sequentially; within a component all DC_BLOCK
    // threads stride-copy non-overlapping elements in parallel.
    // -------------------------------------------------------------------------
    for (int c = 0; c < n_comp; c++) {
        int verts_off = c * max_verts_per * 3;
        int tris_off  = c * max_tris_per  * 3;

        for (int i = tid; i < s_parts[c].mesh.nv * 3; i += DC_BLOCK)
            out_verts[verts_off + i] = s_parts[c].mesh.verts[i];

        for (int i = tid; i < s_parts[c].mesh.nt * 3; i += DC_BLOCK)
            out_tris[tris_off + i] = s_parts[c].mesh.tris[i];
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 8: Free heap allocations (thread 0).
    // For n_comp == 1 the block function returns the original input part
    // unchanged (no new allocation). For n_comp > 1 new meshes were allocated
    // and the original was freed inside the block function.
    // In both cases s_parts[c].mesh.verts is the base of a valid heap chunk
    // that this kernel owns and must free.
    // -------------------------------------------------------------------------
    if (tid == 0) {
        for (int c = 0; c < n_comp; c++)
            heap_free(heap, s_parts[c].mesh.verts);
    }
}
