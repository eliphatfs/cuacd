// hull_warp.cuh — Warp-based (32-thread) convex hull volume algorithms.
//
// Two algorithms:
//   hull_quickhull_warp  — QuickHull with per-face conflict lists in scratch
//   hull_dandc_warp      — Preparata-Hong bottom-up divide-and-conquer
//
// Both use 1 warp (32 threads).  Thread 0 handles sequential topology work;
// all 32 threads participate in parallel reductions / point classification.
// All scratch in caller-provided WarpPool (global memory).
// No shared memory used.
//
// Requires: common.cuh (EPS, signed_tet_volume from geometry.cuh)

#ifndef HULL_WARP_CUH
#define HULL_WARP_CUH

#include "hull_warp_common.cuh"  // WarpPool, warp reductions
#include "hull_quickhull.cuh"    // hull_quickhull_warp
#include "hull_dandc.cuh"        // hull_dandc_warp

#endif // HULL_WARP_CUH
