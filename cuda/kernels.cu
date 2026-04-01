// Root compilation unit — includes all GPU kernel modules.
// Compiled to fatbin via nvcc, loaded at runtime via CUDA driver API.
// Each module is self-contained (carries its own #include directives).

#include "heap_arena.cuh"   // large-object heap (compile-check; not yet wired up)
#include "hull_batch.cu"
#include "test_warp_sort.cu"
#include "test_hull_dandc.cu"
#include "test_plane_cut.cu"
