# k-DOP Approximate Convex Hull — `kdop_hull_block`

## Overview

`kdop_hull_block` (`cuda/kdop_hull.cuh`) computes an approximate convex hull of a point cloud using a Discrete Oriented Polytope (k-DOP) with 40 icosphere-level-1 face normals as axes (80 half-spaces total). The result is a superset of the true convex hull — it contains all input points but may have larger volume.

**Block-level**: all `KDOP_BLOCK` (64) threads in the block must call with identical arguments.

## Signature

```c
__device__ inline Mesh kdop_hull_block(
    const float* verts, int nv,
    DeviceHeap* heap,          // output: final mesh chunk
    DeviceHeap* scratch_heap,  // temporaries (freed on return)
    float* out_volume,         // result written to *out_volume on all threads
    int* kernel_error)         // atomicOr'd on failure
```

Returns a heap-allocated `Mesh`. Returns `{NULL,NULL,0,0}` if `nv < 4`.

## Algorithm

### Step 1 — Centroid
Block-parallel sum over all vertices for x, y, z. Warp shuffle within each warp, then accumulate via shared memory across warps. Divide by `nv`. O(nv / KDOP_BLOCK) work per thread.

### Step 2 — k-DOP Extremes
For each of 40 axes (loop), all threads scan their assigned vertices and compute local max/min of the dot product with `vert - centroid`. Warp reduce, then inter-warp reduce via shared memory → `s_max[40]` and `s_min[40]`. Gives 80 half-spaces: `axis_i · x ≤ s_max[i]` and `(-axis_i) · x ≤ -s_min[i]`.

### Step 3 — Polar Dual Vertices (thread 0)
For each half-space with outward normal `n` and offset `h > ε`, the dual point is `n / h`. Up to 80 dual points stored in scratch heap (`KDOP_MAX_DUAL_PTS * 3 * sizeof(float)`). Degenerate half-spaces (`|h| < KDOP_EPS = 1e-7`) are skipped.

### Step 4 — Convex Hull of Dual Points (warp 0)
`hull_dandc_warp_mesh(dual_pts, n_dual, lane, scratch_heap, scratch_heap, err)`. The dual hull is allocated on `scratch_heap` (temporary).

### Step 5 — Primal Vertices (thread 0)
Each triangle `(d_a, d_b, d_c)` of the dual hull corresponds to one primal vertex: solve the 3×3 system `[d_a; d_b; d_c] · x = (1, 1, 1)` via Cramer's rule. Degenerate triangles (near-singular matrix) are skipped. Primal points allocated on scratch heap (`nt_dual * 3 * sizeof(float)`).

The dual hull mesh is freed immediately after primal vertices are extracted.

### Step 6 — Convex Hull of Primal Vertices (warp 0)
`hull_dandc_warp_mesh(primal_pts, n_primal, lane, heap, scratch_heap, err)`. Output allocated on `heap` (persistent).

### Step 6b — Translate Back (block-parallel)
Add centroid to all output vertices (the centroid was subtracted before step 2).

### Step 7 — Volume (warp 0)
`mesh_volume_warp(&s_result, lane)`.

## Memory Usage (scratch heap)

| Allocation | Size | Lifetime |
|---|---|---|
| Dual points | `KDOP_MAX_DUAL_PTS * 3 * 4` = 960 B | Steps 3–5 |
| Dual hull mesh | `hull_dandc_warp_mesh` scratch + output | Freed end of step 5 |
| Primal points | `nt_dual * 3 * 4` ≤ 2 KB | Steps 5–6 |

All scratch freed before return.

## Axes

40 normalized face normals of a level-1 icosphere (icosahedron subdivided once → 80 faces, antipodal pairs merged). Stored as `__device__ static const float KDOP_AXES[40][3]`.

## Known Limitation: Volume Overestimate

The k-DOP is a bounding volume — it always **contains** the true convex hull. When used in beam search decomposition, `hull_vol - mesh_vol` is inflated vs. the exact D&C hull, causing the cost function to remain high and triggering more splits than intended. Mitigation options: scale the k-DOP volume down by an empirical factor, or switch to exact hull once parts are small enough.

## Python API

```python
with coacd_gpu.Context() as ctx:
    results = ctx.batch_kdop_hull_mesh(pts_list)
    # results[i] = (verts: np.ndarray (nv,3), tris: np.ndarray (nt,3), volume: float)
```

Output buffer bounds: `max_hv=160`, `max_ht=320` (derived from Euler's formula on the dual hull of ≤80 points).
