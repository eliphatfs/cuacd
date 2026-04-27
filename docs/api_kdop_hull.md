# Convex Hull via Extreme-Point Prefilter — `kdop_hull_block`

## Overview

`kdop_hull_block` (`cuda/kdop_hull.cuh`) computes an exact convex hull of a point cloud using a two-stage algorithm:

- **Fast path** (`nv ≤ 1024`): runs `hull_dandc_warp_mesh` directly on the input.
- **Main path** (`nv > 1024`): uses 40 icosphere-L1 axes to identify up to 80 extreme vertices, builds a rough inner hull from those extremes, filters interior points via a face half-space test, then runs `hull_dandc_warp_mesh` on the survivor set.

**Single-warp**: all `KDOP_BLOCK` (32) threads — i.e. exactly one warp — must call with identical arguments.

## Signature

```c
__device__ inline Mesh kdop_hull_block(
    const float* verts, int nv,
    DeviceHeap*  heap,           // output: final mesh chunk
    DeviceHeap*  scratch_heap,   // temporaries (freed on return)
    float*       out_volume,     // written on all threads
    int*         kernel_error)   // atomicOr'd on failure
```

Returns a heap-allocated `Mesh`. Returns `{NULL,NULL,0,0}` if `nv < 4`.

## Algorithm

### Fast path — nv ≤ 1024

Call `hull_dandc_warp_mesh(verts, nv, lane, heap, scratch_heap, &err, &s_result)` directly, then `mesh_volume_warp`. Returns the exact D&C hull.

### Main path — nv > 1024

#### Step 1 — Extreme vertices (warp argmax/argmin with index)
For each of 40 axes, all lanes scan their assigned vertices and maintain a local `(value, index)` pair for max and min of `axis · vert`. Warp-shuffle reduction yields the global argmax and argmin per axis. Lane 0 stores the 80 result indices in `s_extreme_idx[80]`.

#### Step 2 — Deduplicated extreme point array (lane 0)
Collect unique vertex indices from `s_extreme_idx` (O(80²) dedup). Copy original coordinates into scratch-heap buffer `s_extreme_pts` (≤ 80 × 3 floats).

#### Step 3 — Rough inner hull (all lanes)
`hull_dandc_warp_mesh(s_extreme_pts, n_extreme, lane, scratch_heap, scratch_heap, err, &s_ext_hull)` — exact D&C hull of the extreme points. Result is an inner approximation of the true convex hull.

#### Step 4 — Allocate filtered buffer (lane 0)
Scratch-heap buffer for up to `nv + ext_hull.nv` float3 entries.

#### Step 5 — Filter original vertices (all lanes, ballot/popcount)
Lane 0 computes the centroid of the rough-hull vertices as an interior reference point. For each original vertex `v`, check all faces of the rough hull: compute normal `n = cross(b−a, c−a)`, orient outward (away from interior reference), test `n·v > n·a`. Keep vertex if ANY face says outside. Collect survivors using warp ballot/popcount into the filtered buffer.

#### Step 6 — Append rough-hull vertices (all lanes)
Copy all vertices of the rough hull into the filtered buffer (ensures boundary coverage regardless of floating-point sign at hull faces).

Free the rough hull from scratch_heap.

#### Step 7 — Final hull (all lanes)
`hull_dandc_warp_mesh(s_filtered, n_filtered, lane, heap, scratch_heap, err, &s_result)` — exact D&C hull of the filtered + boundary set.

#### Step 8 — Volume
`mesh_volume_warp(&s_result, lane)`.

## Memory Usage (scratch heap)

| Allocation | Size | Lifetime |
|---|---|---|
| Extreme pts | `80 × 3 × 4` = 960 B | Steps 3–cleanup |
| Rough hull mesh | `hull_dandc_warp_mesh` scratch + output | Freed after step 7 |
| Filtered buffer | `(nv + ext_hull.nv) × 3 × 4` | Steps 5–cleanup |

All scratch freed before return.

## Axes

40 normalized face normals of a level-1 icosphere (icosahedron subdivided once → 80 faces, antipodal pairs merged). Defined as `__constant__ float KDOP_AXES[40][3]` in `cuda/kdop_const.cu` (declared `extern __constant__` in `kdop_hull.cuh`); constant memory gives broadcast-cached reads since all warp lanes access the same axis index each iteration. Used only for finding extreme-point indices; not used to define the output hull geometry.

## Correctness

The filtered set contains all true convex hull vertices because the rough hull (convex hull of extreme points) is an inner approximation of the true hull. Any true hull vertex that is not one of the 80 extreme points lies outside the rough hull and therefore passes the filter. The extreme-hull vertices are explicitly appended in step 7.

## Python API

```python
with cuacd.Context() as ctx:
    results = ctx.batch_kdop_hull_mesh(pts_list)
    # results[i] = (verts: np.ndarray (nv,3), tris: np.ndarray (nt,3), volume: float)
```

Output buffer bounds: `max_hv=4096`, `max_ht=8192`.
