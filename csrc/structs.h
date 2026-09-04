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
    CUfunction fn_mesh_audit;

    // PDMC (UDF→surface) kernels
    CUfunction fn_pdmc_count_used_cells;
    CUfunction fn_pdmc_index_used_cells;
    CUfunction fn_pdmc_count_cell_mc_verts;
    CUfunction fn_pdmc_index_cell_mc_verts;
    CUfunction fn_pdmc_count_cell_patches;
    CUfunction fn_pdmc_create_dmc_verts;
    CUfunction fn_pdmc_create_quads;
    CUfunction fn_pdmc_count_div_quads;
    CUfunction fn_pdmc_divide_quads;

    // cumesh2sdf (mesh→SDF grid) kernels
    CUfunction fn_c2s_fill_f32;
    CUfunction fn_c2s_fill_u32;
    CUfunction fn_c2s_fill_collide;
    CUfunction fn_c2s_arange;
    CUfunction fn_c2s_build_trisoup;
    CUfunction fn_c2s_rlayer_p1;
    CUfunction fn_c2s_rlayer_f1;
    CUfunction fn_c2s_rlayer_p2;
    CUfunction fn_c2s_rlayer_f2;
    CUfunction fn_c2s_rlayer_p3;
    CUfunction fn_c2s_rlayer_f3;
    CUfunction fn_c2s_rlayer_p4;
    CUfunction fn_c2s_rlayer_f4;
    CUfunction fn_c2s_rlayer_p5;
    CUfunction fn_c2s_rlayer_f5;
    CUfunction fn_c2s_rlayer_p6;
    CUfunction fn_c2s_rlayer_f6;
    CUfunction fn_c2s_rlayer_p7;
    CUfunction fn_c2s_rlayer_f7;
    CUfunction fn_c2s_rlayer_p8;
    CUfunction fn_c2s_rlayer_f8;
    CUfunction fn_c2s_rasterize_reduce;
    CUfunction fn_c2s_volume_sign_prescan;
    CUfunction fn_c2s_volume_cts;
    CUfunction fn_c2s_volume_apply_sign;
    CUfunction fn_c2s_sdf_shift;

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

    // Merge-hulls postprocess kernels
    CUfunction fn_la_merge_cost_matrix;
    CUfunction fn_la_merge_hausdorff;
    CUfunction fn_la_merge_match;
    CUfunction fn_la_merge_apply;
    CUfunction fn_la_merge_free_unused;
    CUfunction fn_la_merge_compact;

    CUfunction fn_test_postprocess_dc;   // test kernel

    CUdeviceptr d_pool_mem;       // pool backing memory (user allocations)
    CUdeviceptr d_pool_off;       // unsigned long long offset counter (device)
    CUdeviceptr d_pool_struct;    // struct DevicePool on device (much larger now)

    char last_error[256];
};

#endif // STRUCTS_H
