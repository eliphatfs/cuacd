// Host-side data structures for GPU kernels.
// Included by beam.c and test_beam.c.
// DevicePool, DeviceHeap, HeapArena must stay in sync with cuda/allocator.cuh.
//
// NOTE: beam.c update pending (next phase) to match the new DevicePool layout:
//   - DevicePool now embeds both DeviceHeap instances as direct fields.
//   - DeviceHeap has a DevicePool* pool back-pointer (set by heap_init_kernel).
//   - sizeof(DevicePool) is now much larger; allocate d_pool_struct accordingly.
//   - heap_compact_kernel removed; replaced by heap_init_kernel<<<128,32>>>.
//   - d_heap, d_scratch, d_heap_compact_buf, d_scratch_compact_buf removed from beam_ctx.

#ifndef STRUCTS_H
#define STRUCTS_H

#include <cuda.h>

// ---------------------------------------------------------------------------
// Constants (must match cuda/allocator.cuh)
// ---------------------------------------------------------------------------
#define HEAP_NUM_ARENAS  64
#define HEAP_NUM_SUBBINS 64

// ---------------------------------------------------------------------------
// Block header/footer (16 bytes each, same layout)
// ---------------------------------------------------------------------------
struct HeapBlockHdr {
    unsigned int   data_size;
    unsigned short arena_idx;
    unsigned char  is_free;
    unsigned char  _reserved;
    unsigned int   _pad;
};
typedef struct HeapBlockHdr HeapBlockFtr;

// ---------------------------------------------------------------------------
// Per-arena free lists (1040 bytes each)
// ---------------------------------------------------------------------------
struct HeapArena {
    unsigned long long bitmap;                   // sub-bin occupancy bitmap
    int                lock;                     // spin-lock: 0=unlocked
    int                _pad;
    unsigned long long heads[HEAP_NUM_SUBBINS];  // free-list heads
    unsigned long long tails[HEAP_NUM_SUBBINS];  // free-list tails
};

// Forward declaration for circular reference.
struct DevicePool;

// ---------------------------------------------------------------------------
// Heap allocator: pool back-pointer + 64 arenas
// ---------------------------------------------------------------------------
struct DeviceHeap {
    struct DevicePool* pool;                     // back-pointer set by heap_init_kernel
    struct HeapArena   arenas[HEAP_NUM_ARENAS];  // 64 * 1040 = 66560 bytes
};

// ---------------------------------------------------------------------------
// DevicePool: bump allocator + two embedded heap allocators
// ---------------------------------------------------------------------------
struct DevicePool {
    char*               base;      // bump alloc base (device ptr)
    unsigned long long* offset;    // bump alloc offset counter (device ptr)
    unsigned long long  capacity;  // pool capacity in bytes
    struct DeviceHeap   heap;      // output heap  (heap.pool = this)
    struct DeviceHeap   scratch;   // scratch heap (scratch.pool = this)
};

// ---------------------------------------------------------------------------
// Main GPU context  (beam.c update pending — see note above)
// ---------------------------------------------------------------------------
struct beam_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context;

    CUfunction fn_test_warp_sort;
    CUfunction fn_hull_dandc;
    CUfunction fn_query_dandc_scratch;
    CUfunction fn_mesh_volume;
    CUfunction fn_batch_mesh_volume;
    CUfunction fn_plane_cut;
    CUfunction fn_heap_init;      // replaces fn_heap_compact; beam.c update pending

    // Heaps embedded in d_pool_struct (beam.c update pending).
    // Fields d_heap, d_scratch, d_heap_compact_buf, d_scratch_compact_buf removed.
    CUdeviceptr d_pool_mem;       // pool backing memory (user allocations)
    CUdeviceptr d_pool_off;       // unsigned long long offset counter (device)
    CUdeviceptr d_pool_struct;    // struct DevicePool on device (much larger now)

    char last_error[256];
};

#endif // STRUCTS_H
