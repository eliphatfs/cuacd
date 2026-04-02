// mm.cu — Memory management kernel: heap_compact_kernel.
//
// Kernels:
//   heap_compact_kernel — one block (64 threads) wrapping heap_compact.
//
// Launch config: <<<1, 64, 0, stream>>>.
// No concurrent heap ops on other blocks allowed while this runs.

#include "heap_arena.cuh"

extern "C" __global__ void heap_compact_kernel(DeviceHeap* heap) {
    heap_compact(heap);
}
