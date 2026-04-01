// Host-side data structures for GPU kernels.
// Included by beam.c and test_beam.c.
// DevicePool must stay in sync with cuda/structs.cuh (device-side mirror).

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
// Main GPU context
// ---------------------------------------------------------------------------
struct beam_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context;

    CUfunction fn_test_warp_sort;

    CUfunction fn_batch_hull_dandc;
    CUfunction fn_batch_hull_dandc_mesh;
    CUfunction fn_query_dandc_scratch;
    CUfunction fn_batch_mesh_volume;
    CUfunction fn_plane_cut;

    struct DevicePool scratch;

    char last_error[256];
};

#endif // STRUCTS_H
