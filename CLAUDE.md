# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition (reference C++ implementation, SIGGRAPH 2022).
2. **coacd_gpu** — GPU-accelerated convex decomposition via beam search. Single CPython extension (abi3, cp310+) via CUDA driver API. Includes beam search decomposition, Hausdorff distance, pairwise merge cost.

## Repository Layout

```
setup.py              # Build config: setuptools builds _gpu extension
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device code (compiled to single fatbin)
  kernels.cu          #   Root compilation unit — includes all modules
  common.cuh          #   Constants, data structures, pool allocator, atomics
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   Edge intersection, point-triangle distance, concavity metrics
  hull_warp_common.cuh#   WarpPool allocator + warp reductions (used by D&C)
  hull_dandc.cuh      #   Preparata-Hong D&C hull volume (Bullet port, warp sort + lane-0 D&C)
  warp_sort.cuh       #   Warp-cooperative quicksort for BtPoint32 (bitonic ≤32, partitioned >32)
  hull_batch.cu       #   batch_hull_dandc + batch_hull_dandc_mesh + batch_mesh_volume kernels
  test_warp_sort.cu   #   Test kernel for warp_sort_bp32
  beam_v2.cu          #   V2 beam search: beam_expansion, beam_hausdorff_parts, beam_termination
  hausdorff.cu        #   Hausdorff kernels (point_mesh_distance, reduce_max, pairwise)
  mesh_transform.cu   #   Normalize/recover coordinate kernels
csrc/                 # C host code
  beam.h/.c           #   Host implementation (CUDA driver API) — beam search + Hausdorff
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (Hausdorff API) + re-exports BeamContext
  beam.py             #   Python API for beam search (imports _gpu extension)
tests/                # All tests
  test_extension.py   #   GPU smoke tests (Hausdorff, pairwise — standalone script)
  test_beam_v2.py     #   GPU beam search V2 tests (3-kernel architecture)
  test_hull.py        #   Hull volume + mesh volume tests (CPU ref, GPU D&C vs scipy, noisy icosphere)
  test_hull_mesh.py   #   D&C hull mesh extraction tests (batch_hull_dandc_mesh)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling (gaussian points)
  test_v2_kernels.py  #   Per-kernel tests for beam_expansion, beam_hausdorff_parts, beam_termination
  test_warp_sort.py   #   Tests for warp_sort_bp32 (bitonic + quicksort paths)
CoACD/                # Reference C++ CoACD (submodule/external)
```

## Build Commands

```bash
# Install everything (requires: CUDA toolkit with nvcc, C compiler)
pip install -e .

# Override GPU architectures
COACD_GPU_ARCHS="80;86" pip install -e .

# Run GPU smoke tests
python tests/test_extension.py

# Run all tests
python -m pytest tests/ -v

# Build with V2 beam loop debug output (iter/metric/kernel error per iteration)
COACD_V2_DEBUG=1 pip install -e .

# D&C hull benchmark (standalone, for profiling)
python tests/bench_dandc.py --n_pts 200 --n_hulls 8

# Profile D&C hull with NCU
ncu --set full -o dandc_profile python tests/bench_dandc.py --n_pts 200 --n_hulls 8
```

Dependencies: `numpy`, `pytest` (test only), `trimesh` (comparison only).

### CoACD (C++ / Python)

```bash
cd CoACD && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make main -j$(nproc)
./build/main -i examples/SnowFlake.obj -o output.obj -t 0.05
cd CoACD && pip install -e .
```

## coacd_gpu Architecture

### Build System

One extension is built by `setup.py`:

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/kernels.cu` (which `#include`s all `.cuh`/`.cu` modules) -> fatbin -> xxd-style C header, then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake involved. Fatbin is compiled with `--generate-line-info` for NCU source-level profiling.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **Single extension** — all GPU functionality (beam search + Hausdorff + merge cost) in one `_gpu` module. No cmake, no ctypes.
- **Native CPython extension, not ctypes** — ctypes has fragile import path resolution, no type safety, no proper Python object lifecycle. The torchoptix pattern (native CPython extension with embedded fatbin) is the standard approach.
- **No artificial limits** — Do not impose arbitrary caps (e.g., max hull verts/tris per part) that are not inherent to the algorithm. Convex hull output is bounded by Euler's formula (n verts → max 2n-4 triangles); use natural bounds from input size, not hardcoded constants.
- **Evidence-based debugging** — Do not guess errors from partial or truncated output or code. Design and run experiments to confirm root causes with proof before making any fixes. When diagnosing a bug, first write a minimal reproducer or add instrumentation to observe the actual failure.
- **No divergence without consent** — Do not change the algorithm design or add workarounds without explicit approval. If a fix requires a design change, describe the proposed change and wait for approval.
- **Hull algorithm** — D&C (Preparata-Hong) is the primary hull algorithm, exposed via `batch_hull_volume` and `batch_hull_dandc_mesh` (volume + mesh extraction). `hull_dandc_warp_mesh` returns both volume and the extracted hull mesh (vertices + triangles). Volume and mesh extraction use BFS over the half-edge graph (Bullet getVertexCopy pattern) — no DFS stacks, no stack overflow. BFS queue allocated from pool (n pointers), rewound between mesh extraction and volume. Sort scratch and D&C stack are also rewound. `dandc_scratch_bytes` reports `presort + max(sort_scratch, persistent + max(dc_stack, bfs_queue))`. Edge pool sized at `6n` half-edges (exact: max live = `6n-12` by Euler's formula for triangulated convex hulls, confirmed empirically).

### Python API

```python
# Hausdorff distance computation
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    dists = ctx.point_mesh_distances(points, vertices, triangles)
    h = ctx.hausdorff(sa, va, ta, sb, vb, tb)
    cost = ctx.pairwise_hausdorff(samples, s_off, verts, tris, t_off, v_off)

# Beam search decomposition (V2 — 3-kernel, D&C hull + Hausdorff)
from coacd_gpu.beam import BeamContext, run_beam_coacd_v2
with BeamContext(device=0) as ctx:
    parts = ctx.run_v2(vertices, triangles, threshold=0.05)
# or:
parts = run_beam_coacd_v2(vertices, triangles, threshold=0.05)
```

Both `Context` and `BeamContext` share the same underlying `_gpu` extension and CUDA context.

## Beam Search Algorithm Design

### Overview

Replace MCTS with beam search over decomposition states. Search space: 3xN axis-aligned cuts (N per axis, default N=10 -> 30 total candidates). Beam width X (default 30). 3-kernel architecture (V2):

**Kernel 1 — `beam_expansion`**: For X work items × 30 candidates = 30X blocks. Each block clips the worst part of a work item along a candidate plane, computes Rv via D&C hull, writes new PartInfoV2 entries. Last block (via atomicAdd counter) selects top X candidates by insertion sort.

**Kernel 2 — `beam_hausdorff_parts`**: For each unique part (across all work items). Bidirectional Hausdorff between part mesh and its hull mesh. Skips parts already computed or with Rv > 2×threshold.

**Kernel 3 — `beam_termination`**: One warp per work item. Uses `__all_sync` to check if all parts are below threshold. `warp_argmax_f` to find worst part. Writes winning item to `result` if all items are done.

No connected component analysis — each cut produces exactly two parts (positive/negative half), matching CoACD's approach. Metrics are computed per-part, not per-component.

### Concavity Metric: Rv (Volume-Ratio)

Rv = `(3 * |V_mesh - V_hull| / (4*pi))^(1/3) * k` where k = rv_k (default 0.3).

- **Mesh volume**: parallel signed-tetrahedra reduction `V = (1/6) * sum p0.(p1 x p2)`. For open meshes after clipping, cap volume correction via divergence theorem: `V_cap = (d/3) * |A_boundary|` from boundary edge loop shoelace signed area.
- **Hull volume**: D&C hull (`hull_dandc_warp_mesh`) — warp-parallel Preparata-Hong, unlimited vertices, exact Int128 arithmetic. Returns both volume and extracted mesh for Hausdorff.
- **Cap triangulation**: fan cap triangles close meshes after each clip for correct signed-tet volumes in subsequent iterations.
- **Threshold**: compatible with CoACD semantics. Convex shape -> Rv ~ 0. Threshold 0.05 typical.
- **Scoring a cut**: `max(Rv_positive_half, Rv_negative_half)`. Beam search minimizes worst-case concavity.
- **Per-part bounding box planes**: cutting planes uniformly distributed within each part's triangle-vertex bbox (not all-vertex bbox, not global). Odd cuts_per_axis (e.g., 15) ensures midpoint is always a candidate. No snapping to vertex coordinates.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_width` (X) | 30 | Number of beam items to keep |
| `cuts_per_axis` (N) | 10 | Planes per axis direction (30 total) |
| `threshold` | 0.05 | Concavity threshold for termination |
| `rv_k` | 0.3 | Scaling factor in Rv formula |
| `max_parts` | 64 | Max parts per beam item |
| `max_iterations` | 64 | Max decomposition steps |
| `hausdorff_samples` | 1000 | Samples for Hausdorff validation |

### Pool Allocator Pattern

All scratch memory in kernels is allocated from a global bump pool. **Critical pattern**: only thread 0 calls `pool_alloc()`, stores pointer in `__shared__` memory, then all threads read after `__syncthreads()`:

```cuda
__shared__ int* s_signs;
__shared__ float* s_verts;
if (tid == 0) {
    s_signs = (int*)pool_alloc(&scratch, n * sizeof(int));
    s_verts = (float*)pool_alloc(&scratch, m * sizeof(float));
}
__syncthreads();
int* signs = s_signs;  // all threads see same pointer
```

If all 256 threads call `pool_alloc`, each gets a different offset (256x memory waste) and threads write to different arrays, causing data corruption.

## Utility Function Inventory

### A. Scalar Device Functions (per-thread, no sync)

| ID | Function | File |
|----|----------|------|
| A1 | `signed_tet_volume(p0, p1, p2) -> float` | geometry.cuh |
| A2 | `intersect_edge(v0, v1, plane) -> (ix, iy, iz)` | geometry.cuh |
| A3 | `point_triangle_dist(point, v0, v1, v2) -> float` | geometry.cuh |

### B. Block-Level Reductions (all threads call, __syncthreads)

| ID | Function | File |
|----|----------|------|
| B1 | `block_reduce_sum(val, smem, tid) -> float` | reduce.cuh |
| B2 | `block_reduce_bbox(verts, n, offset, tid, smem, out_lo, out_hi)` | reduce.cuh |
| B3 | `block_reduce_max(val, smem, tid) -> float` | reduce.cuh |
| B4 | `block_reduce_count(flag, smem_i, tid) -> int` | reduce.cuh |

### F. Hausdorff (kernels + host helpers)

| ID | Function | File |
|----|----------|------|
| F1 | `sample_surface` kernel | hausdorff.cu — area-weighted surface sampling |
| F2 | `point_mesh_distance` kernel | hausdorff.cu — brute-force point-to-triangle, one thread per point |
| F3 | `reduce_max` kernel | hausdorff.cu — shared-memory parallel max reduction |
| F5 | `pairwise_hausdorff` kernel | hausdorff.cu — batch merge cost matrix |

### G. Mesh Transform (kernels)

| ID | Function | File |
|----|----------|------|
| G1 | `normalize_mesh` kernel | mesh_transform.cu |
| G2 | `recover_coordinates` kernel | mesh_transform.cu |

### I. Memory / Infrastructure

| ID | Function | File |
|----|----------|------|
| I1 | `pool_alloc(pool, size) -> void*` | common.cuh |
| I2 | `atomicMinF / atomicMaxF` | common.cuh |

### J. Dead Code

| ID | What | Why dead |
|----|------|----------|
| J1 | `compute_concavity_tris` in geometry.cuh | Bbox cube-root proxy, superseded by Rv |

## Beam Search Architecture Detail

### Data Structures (common.cuh)

- **PartInfoV2**: `hull_vert_offset`, `hull_vert_count`, `hull_tri_offset`, `hull_tri_count`, `hausdorff` (-1 = not computed), `mesh_volume`, `hull_volume`.
- **WorkItem**: `part_indices[MAX_PARTS_PER_BEAM]` for indirect indexing into a global PartInfoV2 pool (no per-item part copies). `worst_part_idx` indexes into `part_indices`, `worst_metric` = max(rv, hausdorff) of worst part.

### Kernels (beam_v2.cu)

**Kernel 1 — `beam_expansion`** (64 threads = 2 warps per block):
- Grid: `num_work_items * 3 * cuts_per_axis`. Each block = one candidate cut.
- Split scratch allocated from global DevicePool via `global_alloc_t0` (thread 0 atomicAdd).
- Each warp allocates its own D&C WarpPool from the global DevicePool via `warppool_from_global` (lane 0 atomicAdd). No shared memory for pointers.
- No hull vertex limit — all unique triangle-referenced vertices fed to D&C.
- Hull mesh extracted via `hull_dandc_warp_mesh` into the output vertex/triangle pool.
- Last-block selection: `atomicAdd(done_counter)`, last block does insertion sort.

**Kernel 2 — `beam_hausdorff_parts`** (256 threads per block):
- Grid: total unique parts. Bidirectional Hausdorff between part mesh and hull mesh.
- Skips parts where `hausdorff >= 0` (already computed) or `rv_cost > 2*threshold` (too far).

**Kernel 3 — `beam_termination`** (32 threads = 1 warp per block):
- Grid: `num_work_items`. Uses `__all_sync` for termination check, `warp_argmax_f` for worst part.
- Writes winning work item index to `result` via `atomicExch`.

### Host Loop (beam_run_v2 in beam.c)

1. Upload mesh + scipy ConvexHull to vertex/triangle pools
2. Normalize both mesh and hull together
3. Compute initial mesh_vol and hull_vol via `batch_mesh_volume` kernel
4. Compute initial Rv; early exit if convex
5. Loop: Hausdorff → Termination check → Expansion → swap buffers

### Scratch Rewind and BFS (hull_dandc.cuh)

Three rewind points in the D&C hull minimize peak scratch:
1. **Sort scratch**: rewound after `warp_sort_bp32` completes.
2. **D&C stack**: rewound in `bt_compute_postsort` after `bt_computeInternal`.
3. **BFS queue**: `bt_extractMesh`'s BFS queue rewound in `hull_dandc_warp_mesh` before `bt_computeVolume`.

Volume and mesh extraction use BFS (Bullet `getVertexCopy` pattern) instead of DFS — no stack overflow possible. BFS queue size = n vertex pointers, bounded by hull vertex count. `dandc_scratch_bytes` = `presort + max(sort_scratch, persistent + max(dc_stack, bfs_queue))`.

Edge pool: `6n` half-edges. Empirically confirmed exact: max live edge pairs = `3n - 6` (Euler, triangulated hull), peak never exceeds final count during merge. `COACD_TRACK_EDGES=1 pip install -e .` enables `TRACK_MAX_EDGE_PAIRS` compile macro for printf tracking.

### Global DevicePool Scratch Allocation

V2 kernels allocate all scratch from a single global DevicePool (70% of free VRAM, capped at 4GB). Each warp calls `warppool_from_global(&scratch, dandc_scratch_bytes(n), &wp)` from lane 0, which does an `atomicAdd` on the global offset. No pre-sized per-block scratch arrays.

### V2 Status

- **Working**: Hull mesh extraction (`batch_hull_dandc_mesh`), cube convexity (early exit), scratch rewinds (sort/D&C/BFS), BFS graph traversal (no DFS stack overflow).
- **Bug (WIP)**: `beam_run_v2` has an illegal memory access during the initial `batch_mesh_volume` call for hull volume computation. The hull triangle indices in the triangle pool are 0-based but need rebasing by `n_verts` since hull vertices are stored after original vertices. The `vert_offsets` fix to `{n_verts, n_verts}` was applied but the crash persists — needs further debugging.
- **Not yet working**: Full L-shape decomposition via V2 path.

## Current Implementation vs. Design Discrepancies

### No Hausdorff in beam loop

**Design**: Dedicated kernel — Hausdorff for parts where Rv < eps, skipping parts with Rv > 2×threshold.

**Current**: `beam_hausdorff_parts` kernel exists and skips expensive cases, but host wiring to integrate it into the termination loop is incomplete. Termination currently uses Rv alone.

## Implementation Lessons

### DevicePool offset/capacity are 64-bit

`DevicePool.offset` and `DevicePool.capacity` are `unsigned long long`. Previously they were `unsigned int` (32-bit), causing silent wraparound when >4GB scratch was consumed by concurrent expansion blocks (480 blocks × ~15MB each = ~7GB). The wraparound made the OOM check pass on wrapped offsets, giving blocks overlapping memory regions → corrupted D&C edge pool free lists → illegal memory access. Fixed by widening to 64-bit; the 4GB scratch cap was also removed.

### D&C: Faithful Port of Bullet's btConvexHullComputer

The D&C algorithm is a faithful port of Bullet's `btConvexHullComputer` (Ole Kniemeyer, MAXON, zlib license). Three phases: (1) warp-parallel pre-sort (AABB via warp min/max reduction, Point32 conversion strided across lanes), (2) warp-cooperative sort via `warp_sort_bp32` (all 32 lanes), (3) D&C merge + volume extraction on lane 0. Key elements: int32 coordinates with exact Int128/Rational64/Rational128 predicates, iterative `computeInternal` D&C with explicit `BtDCStackItem` stack (`BT_DC_MAX_STACK=4096`), `mergeProjection` for 2D bridge finding, `findMaxAngle` with exact cotangent comparison, `findEdgeForCoplanarFaces` for coplanar handling, and the full `merge` function with interior edge deletion via `removeEdgePair`. Memory is allocated from WarpPool bump allocator with warp-parallel free-list init in `btpool_init` (all 32 lanes build the free list, lane 0 sets the head) and free-list recycling for edges. Volume and mesh extraction use BFS over the half-edge graph (Bullet `getVertexCopy` pattern) — no DFS, no stack overflow. BFS queue allocated from pool (n pointers). Volume sums signed tetrahedra in integer coordinates, converts back via scaling factor. Edge pool sized at `6n` half-edges (exact tight bound: max live = `6n-12` by Euler). Scratch per hull computed by `dandc_scratch_bytes(n)` (queried from device via `query_dandc_scratch`). Error codes: 1=pool OOM, 2=sort stack overflow (`WS_MAX_STACK`), 4=D&C stack overflow (`BT_DC_MAX_STACK`), 5=pool exhausted (`BT_ERR_POOL_EXHAUST`). Requires 8KB thread stack (via `cuCtxSetLimit`), block size 32 (1 warp per block, DANDC_BLOCK_SIZE).

### Warp Sort (warp_sort.cuh)

Warp-cooperative quicksort for BtPoint32 arrays. All 32 lanes participate. Uses a global-memory workspace of the same size as the input. Segments ≤32 elements use a bitonic sorting network; larger segments use quicksort partitioning with a median-of-32-samples pivot. Three-way partition via `ws_cmp` (returns -1/0/1): items < pivot go left, items > pivot go right, items == pivot are filled in the middle gap by warp-parallel fill. This avoids worst-case O(n²) on all-equal or many-duplicate inputs. Uses `__ballot_sync`/`__popc` for warp-wide prefix sums. Explicit stack (max depth 2048) in global memory, `sp` in lane-0 register broadcast via `__shfl_sync`.

### btpool: Eager Warp-Parallel Init, No Lazy Allocation

`btpool_init(p, wp, blockSize, objSize, lane)` eagerly allocates the pool block and uses all 32 warp lanes to build the free list in parallel (each lane handles every 32nd slot). `btpool_new` only pops from the free list; if exhausted, it sets `BT_ERR_POOL_EXHAUST` (error code 5) and returns NULL — no lazy block allocation. `objSize` is padded to multiple of 4; zero-init in `btpool_new` uses `int*` with `objSize/4` iterations. Only the edge pool (BtEdge=40B) is used; vertex and face pools were removed as unused.

### pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

### nvcc Crashes with --generate-line-info and Deep Recursion

nvcc crashed when compiling with `--generate-line-info` while `bt_computeInternal` was recursive. Converting to an iterative explicit stack resolved the crash. Line info is now always enabled for NCU profiling.

### scipy ConvexHull.simplices Winding Is Not Consistent

`scipy.spatial.ConvexHull.simplices` does not guarantee outward-facing normals. For signed-tet volume computation, inconsistent winding causes triangle contributions to cancel, yielding ≈0 instead of the true volume. Fix: for each simplex `[a,b,c]`, compute `cross(b-a, c-a)` and check `dot(normal, va - centroid) < 0`; flip `b,c` if so. Apply in `beam.py` after remapping hull triangle indices.

### V2 Pool Capacity Formula

V2's `beam_expansion` creates 2 new parts per block per iteration. Hull output uses natural Euler bounds from actual hull input size (n input points → max n hull verts, max 2n-4 hull tris) — no artificial limits. Pool capacity is sized from available GPU memory (65% of free VRAM split 45/45/10 between vert/tri/part pools) — no arbitrary caps.

### V2 Kernel Error Reporting Pattern

All three V2 kernels (`beam_expansion`, `beam_hausdorff_parts`, `beam_termination`) take `int* kernel_error`. Errors are written via `atomicOr(kernel_error, KERR_*)`. Host zeros `d_kerr` before each launch, then issues `cuMemcpyDtoHAsync` immediately after the `cuLaunchKernel` call (before `cuStreamSynchronize`) so the download overlaps with kernel execution. Bit flags: `KERR_SCRATCH_OOM=1`, `KERR_SPLIT_OOM=2`, `KERR_HULL_PTS_OOM=4`, `KERR_POOL_OOM=8`, `KERR_HULL_ERR=16`.

### Scratch Query Ordering Does Not Help batch_hull_volume

Moving `query_dandc_scratch` before the large `cuMemcpyHtoDAsync` calls (to avoid syncing after copies) was tested and made performance slightly worse. The `cuMemAlloc` calls themselves may already serialize, so reordering provides no benefit. Keep the query after copies for now.

## Current Status

### Working (V2 — partial)
- D&C hull mesh extraction (`bt_extractMesh`, `hull_dandc_warp_mesh`, `batch_hull_dandc_mesh`) — tested (cube 8v/12t, tetra 4v/4t, gaussian)
- V2 data structures (PartInfoV2, WorkItem) — defined in common.cuh, beam.h
- Two-warp reductions (`twowarp_reduce_sum/max/count`) — in reduce.cuh
- Scratch rewinds (sort/D&C/BFS) + BFS graph traversal replacing DFS in D&C hull
- Global DevicePool scratch allocation (`warppool_from_global`) — in hull_warp_common.cuh
- V2 kernel compilation: beam_expansion, beam_hausdorff_parts, beam_termination — compiles and individually tested
- V2 host code (`beam_run_v2`) and Python API (`run_v2`, `run_beam_coacd_v2`) — compiles
- V2 cube convexity early-exit path — **working**
- V2 L-shape decomposition — **working** (7 parts at threshold=0.05)
- V2 Octocat-v2 (20k verts) — **working** (37 parts at threshold=0.05, ~6s)
- Per-kernel test infrastructure: `test_termination`, `test_hausdorff_parts`, `test_expansion` C API + Python wrappers (tests/test_v2_kernels.py — 14 tests pass)
- Kernel-side error reporting: `KERR_*` bit flags in all 3 V2 kernels (`KERR_SCRATCH_OOM=1`, `KERR_SPLIT_OOM=2`, `KERR_HULL_PTS_OOM=4`, `KERR_POOL_OOM=8`, `KERR_HULL_ERR=16`); host async-downloads and checks after each launch
- Pool OOM bounds checking in `beam_expansion` (hull pre-alloc and mesh copy both checked)
- `CHECK_CU` macro now includes `beam.c:LINE` in error messages for faster diagnosis

### V2 Bugs Fixed
- **Scipy hull winding**: `hull.simplices` have inconsistent winding → signed-tet sum ≈ 0 → rv non-zero for convex shapes → no early exit → pool overflow. Fixed in `beam.py`: orient each hull triangle outward using centroid dot-product check.
- **Triangle pool overflow in `beam_expansion`**: Fixed by removing artificial `max_hull_verts_per_part`/`max_hull_tris_per_part` limits and using natural Euler bounds from actual hull input size. Pool capacities computed from worst-case per-block allocation.
- Root cause of illegal memory access confirmed via `compute-sanitizer`: `bt_extractMesh` writing to `out_tris` past end of triangle pool in block 324/480 (second iteration).

### V2 Bugs Fixed (cont.)
- **Hull triangle index rebasing**: `hull_dandc_warp_mesh` writes 0-based triangle indices, but the Hausdorff kernel needs absolute pool indices. Missing rebase (`+= hull_vo_pos/neg`) caused Hausdorff to read wrong vertices → absurd distances (1.6 on normalized mesh) → termination never fired. Fixed by adding rebase loop in `beam_expansion` after D&C hull extraction.
- **DevicePool 32-bit offset wraparound**: `DevicePool.offset` was `unsigned int`, causing silent wraparound when >4GB scratch was consumed by concurrent expansion blocks (480 blocks × ~15MB D&C scratch each). Blocks got overlapping memory → corrupted edge pool free lists → illegal memory access in `btpool_new`. Fixed by widening offset/capacity to `unsigned long long`; removed 4GB scratch cap.

### V2 Bugs Remaining
- Hull mesh winding: `bt_extractMesh` produces mixed winding (mesh volume via signed tet = 0.667 for unit cube instead of 1.0). The D&C volume (via int128 arithmetic) is correct. Winding consistency in extracted mesh needs investigation.
- Hausdorff kernel skipped in beam loop (debugging Rv-only termination quality first).

### V2 Scaling Note

Every expansion block allocates O(n_verts) pool space for mesh copies + hull output, but only beam_width results survive selection. For large meshes (20k+ verts), this wastes significant pool space per iteration. A GPU-side pool compaction kernel (run between iterations) would reclaim dead space. Currently this is not a problem because pools are sized from available GPU memory (65% of free VRAM).

### Not Yet Implemented
- Pool compaction kernel (GPU-side, between iterations — reclaim dead pool space from non-winning expansion blocks)
- Merge post-processing
- Vertex compaction (parts carry superset of vertices)
- `__cuda_array_interface__` support for GPU tensor input
