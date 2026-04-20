// Host-side data structures for GPU kernels.
// Included by heap.c, lookahead.c, and test.c.
// DevicePool, DeviceHeap, HeapArena must stay in sync with cuda/allocator.cuh.

#ifndef STRUCTS_H
#define STRUCTS_H

#include <cuda.h>

// ---------------------------------------------------------------------------
// Constants (must match cuda/allocator.cuh)
// ---------------------------------------------------------------------------
#ifndef HEAP_NUM_ARENAS
#define HEAP_NUM_ARENAS  64
#endif
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
    unsigned long long outstanding_bytes;        // diagnostic: live alloc user-bytes
    unsigned long long alloc_count;              // diagnostic: cumulative alloc calls
    unsigned long long free_count;               // diagnostic: cumulative free calls
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
// Main GPU context
// ---------------------------------------------------------------------------
struct gpu_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context;

    CUfunction fn_test_warp_sort;
    CUfunction fn_hull_dandc;
    CUfunction fn_mesh_volume;
    CUfunction fn_batch_mesh_volume;
    CUfunction fn_plane_cut;
    CUfunction fn_heap_init;      // replaces fn_heap_compact; beam.c update pending
    CUfunction fn_kdop_hull;
    CUfunction fn_hausdorff;

    // Lookahead decomposition kernels
    CUfunction fn_la_init;
    CUfunction fn_la_hausdorff_parts;
    CUfunction fn_la_sort_and_count_cutting;
    CUfunction fn_la_seed_tree;
    CUfunction fn_la_expand;
    CUfunction fn_la_expand_quick;
    CUfunction fn_la_hull_la;
    CUfunction fn_la_sort_and_record;
    CUfunction fn_la_evaluate;
    CUfunction fn_la_apply_cuts;
    CUfunction fn_la_hull_decomp;
    CUfunction fn_la_cleanup_tree;
    CUfunction fn_la_cleanup_tree3;
    CUfunction fn_la_free_decomp;
    CUfunction fn_la_decompose_components;
    CUfunction fn_la_find_concave_edges;
    CUfunction fn_la_compute_best_ub;

    CUfunction fn_test_postprocess_dc;   // test kernel

    CUdeviceptr d_pool_mem;       // pool backing memory (user allocations)
    CUdeviceptr d_pool_off;       // unsigned long long offset counter (device)
    CUdeviceptr d_pool_struct;    // struct DevicePool on device (much larger now)

    char last_error[256];
};

#endif // STRUCTS_H
