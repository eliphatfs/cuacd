// Root compilation unit — includes all GPU kernel modules.
// Compiled to fatbin via nvcc, loaded at runtime via CUDA driver API.
// No host-side includes, no runtime API calls.

extern "C" {

// Shared foundations
#include "common.cuh"
#include "reduce.cuh"
#include "geometry.cuh"

// Kernel modules
#include "mesh_transform.cu"
#include "hausdorff.cu"
#include "hull_batch.cu"
#include "test_warp_sort.cu"
#include "beam_v2.cu"

} // extern "C"
