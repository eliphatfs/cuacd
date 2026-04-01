// Host-side data structures for GPU beam search convex decomposition.
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
// Extracted convex hull part (CPU memory, filled by beam_get_part)
// ---------------------------------------------------------------------------
struct OutputPart {
    float* vertices;
    int*   triangles;
    int    n_verts;
    int    n_tris;
};

// ---------------------------------------------------------------------------
// Main GPU context
// ---------------------------------------------------------------------------
struct beam_ctx {
    CUdevice   device;
    CUcontext  cuda_ctx;
    CUmodule   module;
    int        owns_context;

    CUfunction fn_normalize_mesh;
    CUfunction fn_recover_coordinates;
    CUfunction fn_test_warp_sort;

    CUfunction fn_batch_hull_dandc;
    CUfunction fn_batch_hull_dandc_mesh;
    CUfunction fn_query_dandc_scratch;
    CUfunction fn_batch_mesh_volume;
    CUfunction fn_plane_cut;
    CUfunction fn_batch_plane_cut;
    CUfunction fn_batch_compact_mesh;
    CUfunction fn_batch_bbox;

    struct DevicePool scratch;

    struct OutputPart* output_parts;
    int num_output_parts;

    char last_error[256];
};

// ---------------------------------------------------------------------------
// beam_decompose internal structures (GPU-resident decomposition state)
// ---------------------------------------------------------------------------

#define DC_MAX_PARTS     16384
#define DC_MAX_ITEMS      8192
#define DC_MAX_ITEM_PARTS  256
#define DC_MAX_EPOCHS      256

typedef struct {
    CUdeviceptr d_verts;    // absolute GPU pointer (into epoch buf or own alloc)
    CUdeviceptr d_tris;
    int n_verts, n_tris;
    float hull_vol, mesh_vol, rv;
    float bbox[6];          // xmin,xmax,ymin,ymax,zmin,zmax
    int epoch_idx;          // -1 = own GPU alloc; >= 0 = points into epoch buffer
    int ref_count;          // # of items referencing this part
    int alive;
} DCPart;

typedef struct {
    CUdeviceptr d_pos_verts, d_pos_tris;  // compact pos halves for all cuts
    CUdeviceptr d_neg_verts, d_neg_tris;  // compact neg halves for all cuts
    int ref_count;  // # of DCParts pointing into this epoch
    int alive;
} DCEpoch;

typedef struct {
    int part_ids[DC_MAX_ITEM_PARTS];
    int n_parts;
    float worst_rv;
    int worst_local;
} DCItem;

#endif // STRUCTS_H
