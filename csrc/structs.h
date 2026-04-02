// Host-side data structures for GPU kernels.
// Included by beam.c and test_beam.c.
// DevicePool, HeapArena, DeviceHeap must stay in sync with device-side definitions
// in cuda/allocator.cuh and cuda/heap_arena.cuh.

#ifndef STRUCTS_H
#define STRUCTS_H

#include <cuda.h>

// ---------------------------------------------------------------------------
// GPU scratch pool (host-side view — base/offset are device pointers)
// ---------------------------------------------------------------------------
struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
};

// ---------------------------------------------------------------------------
// Heap arena free list (must match cuda/heap_arena.cuh HeapArena)
// ---------------------------------------------------------------------------
#define HEAP_NUM_ARENAS        64
#define HEAP_COMPACT_BUF_BYTES (16 << 20)   // 16 MB

struct HeapArena {
    unsigned long long head;   // free-list head (device ptr), 0 = empty
    int                lock;   // spin-lock: 0 = unlocked
    int                _pad;
};

// ---------------------------------------------------------------------------
// Device heap (host-side mirror of cuda/heap_arena.cuh DeviceHeap)
// ---------------------------------------------------------------------------
struct DeviceHeap {
    struct DevicePool*  pool;                        // device ptr to DevicePool
    struct HeapArena    arenas[HEAP_NUM_ARENAS];     // zero = empty, unlocked
    unsigned long long* compact_buf;                 // device ptr to compact buffer
};

// ---------------------------------------------------------------------------
// Main GPU context
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

    char last_error[256];
};

#endif // STRUCTS_H
