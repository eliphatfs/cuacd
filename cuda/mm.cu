// mm.cu — Memory management kernels.
//
// Kernels:
//   heap_init_kernel — initialise both heaps embedded in a DevicePool.
//
// Launch config: <<<128, 32, 0, stream>>>
//   Blocks   0..63  initialise pool->heap.arenas[blockIdx.x]
//   Blocks  64..127 initialise pool->scratch.arenas[blockIdx.x - 64]
// Each block is one warp (32 threads) for cooperative zeroing of heads/tails.
// The pool back-pointer (heap.pool / scratch.pool) is set by block 0 of each heap.

#include "heap_arena.cuh"   // transitively includes allocator.cuh

extern "C" __global__ void heap_init_kernel(DevicePool* pool) {
    int is_scratch = (blockIdx.x >= (unsigned int)HEAP_NUM_ARENAS) ? 1 : 0;
    int arena_idx  = (int)(blockIdx.x % (unsigned int)HEAP_NUM_ARENAS);
    DeviceHeap* h  = is_scratch ? &pool->scratch : &pool->heap;
    HeapArena*  arena = &h->arenas[arena_idx];

    int lane = (int)threadIdx.x;  // 0..31

    // Thread 0 of block 0 (for each heap) sets the pool back-pointer.
    // Only needs to happen once per heap, but doing it from every block's lane 0
    // with the same value is idempotent and avoids extra coordination.
    if (lane == 0) {
        h->pool       = pool;
        arena->bitmap = 0;
        arena->lock   = 0;
        arena->_pad   = 0;
    }

    // All lanes cooperatively zero heads[] and tails[] (64 entries each).
    for (int i = lane; i < HEAP_NUM_SUBBINS; i += 32) {
        arena->heads[i] = 0;
        arena->tails[i] = 0;
    }
}
