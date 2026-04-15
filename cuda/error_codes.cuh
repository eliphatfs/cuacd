// error_codes.cuh — Centralized GPU kernel error flags.
//
// All codes are single-bit flags combined via atomicOr into a shared int.
// HEAP_OK in allocator.cuh remains 0 (success); the HEAP error codes here
// replace the old HEAP_ERR_OOM/2/3 so heap_alloc/heap_free return values
// can be directly atomicOr'd into the kernel error word.
//
// Usage:
//   Device:  atomicOr(err, KERR_XX_YY);
//   Host:    kerr_decode(err, buf, sizeof(buf));  // csrc/error_codes.h
#pragma once

// --- hull_dandc (D&C convex hull) ---
#define KERR_BT_WARP_OOM       (1 <<  0)  // WarpPool allocation failed (bt_alloc)
#define KERR_BT_SORT_STACK     (1 <<  1)  // warp_sort stack overflow
#define KERR_BT_DC_STACK       (1 <<  2)  // D&C recursion stack overflow
#define KERR_BT_POOL_EXHAUST   (1 <<  3)  // edge pool block limit reached
#define KERR_BT_HEAP_TO_WARP   (1 <<  4)  // heap_alloc for WarpPool backing failed
#define KERR_BT_HEAP_OUTPUT    (1 <<  5)  // heap_alloc for output mesh failed
#define KERR_BT_EXTRACT_FAIL   (1 <<  6)  // bt_extractMesh returned error

// --- plane_cut ---
#define KERR_PC_SCRATCH_OOM    (1 <<  7)  // scratch heap alloc failed
#define KERR_PC_POOL_OOM       (1 <<  8)  // heap_alloc for output mesh failed
#define KERR_PC_SORT_ERR       (1 <<  9)  // warp_sort error

// --- hausdorff ---
#define KERR_HD_SCRATCH_OOM    (1 << 10)  // scratch heap alloc failed
#define KERR_HD_SORT_ERR       (1 << 11)  // warp_sort error

// --- postprocess (connected components) ---
#define KERR_DC_OOM            (1 << 12)  // component decomposition OOM

// --- kdop_hull ---
#define KERR_KDOP_SCRATCH_OOM  (1 << 13)  // scratch heap alloc failed in kdop_hull

// --- lookahead ---
#define KERR_LA_OVERFLOW       (1 << 14)  // exceeded LA_MAX_PARTS
#define KERR_LA_SORT_OOM       (1 << 15)  // scratch heap alloc failed in sort
#define KERR_LA_SORT_STACK     (1 << 16)  // warp_sort stack overflow
#define KERR_LA_EVAL_OOM       (1 << 17)  // scratch heap alloc failed in evaluate

// --- mesh validation ---
#define KERR_MESH_INVALID      (1 << 18)  // mesh validation failed (bad nv/nt/null ptrs)

// --- heap allocator (return codes from heap_alloc / heap_free) ---
#define KERR_HEAP_OOM          (1 << 19)  // heap_alloc out of memory
#define KERR_HEAP_CORRUPT      (1 << 20)  // heap_free detected block corruption
#define KERR_HEAP_CORRUPT_PREV (1 << 21)  // heap_free detected previous-block corruption

// Total error types and mask for unknown-bit detection on host side.
#define KERR_COUNT     22
#define KERR_ALL_BITS  ((1 << KERR_COUNT) - 1)
