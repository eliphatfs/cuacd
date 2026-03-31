// Root compilation unit — includes all GPU kernel modules.
// Compiled to fatbin via nvcc, loaded at runtime via CUDA driver API.
// Each module is self-contained (carries its own #include directives).

#include "mesh_transform.cu"
#include "hull_batch.cu"
#include "test_warp_sort.cu"
#include "plane_cut.cu"
