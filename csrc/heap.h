// GPU context and heap/pool lifecycle management.
// Uses CUDA driver API internally — no CUDA runtime dependency.
// Linked into a CPython extension module; not a standalone shared library.

#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>
#include <stddef.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to GPU context
typedef struct gpu_ctx* gpu_ctx_t;

// pool_bytes: backing memory for both heaps combined.
//   0 = auto: use 70% of free device memory at init time.
int  gpu_init(gpu_ctx_t* ctx, int device_ordinal, size_t pool_bytes);
void gpu_destroy(gpu_ctx_t ctx);

const char* gpu_last_error(gpu_ctx_t ctx);

// gpu_heap_compact — coalesce free blocks in both persistent heaps.
// Call periodically to recover fragmented memory between kernel launches.
// Blocks until compaction is complete.
int  gpu_heap_compact(gpu_ctx_t ctx);

// gpu_pool_usage — read back how many bytes have been bump-allocated from
// the shared pool (= peak live device memory used by both heaps combined).
// The offset never decreases; freed blocks go to heap free-lists, not back
// to the pool. Returns 0 on error.
size_t gpu_pool_usage(gpu_ctx_t ctx);

// Host-side mirrors of CUDA device structs.
// Pointer fields use CUdeviceptr (uint64) to match 64-bit device pointers.
struct Mesh_h {
    CUdeviceptr verts;     // float* on device
    CUdeviceptr tris;      // int*   on device
    int         nv;
    int         nt;
    CUdeviceptr refcount;  // int*   on device
};

struct Part_h {
    struct Mesh_h mesh;
    struct Mesh_h hull;
    float mesh_vol;
    float hull_vol;
    float hausdorff;
};

struct gpu_part_result {
    float* verts;       // malloc'd, nv*3 floats
    int*   tris;        // malloc'd, nt*3 ints
    int    nv;
    int    nt;
    float* hull_verts;  // malloc'd, hull_nv*3 floats
    int*   hull_tris;   // malloc'd, hull_nt*3 ints
    int    hull_nv;
    int    hull_nt;
    float  mesh_vol;
    float  hull_vol;
    float  hausdorff;
};

struct gpu_result {
    struct gpu_part_result* parts;  // malloc'd array of nparts entries
    int nparts;
};

void gpu_result_free(struct gpu_result* result);

#ifdef __cplusplus
}
#endif

#endif // HEAP_H
