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
  hull.cuh            #   Incremental convex hull (shared mem, ≤256 verts, used by beam search)
  hull_warp_common.cuh#   WarpPool allocator + warp reductions (used by D&C)
  hull_dandc.cuh      #   Preparata-Hong D&C hull volume (Bullet port, warp sort + lane-0 D&C)
  warp_sort.cuh       #   Warp-cooperative quicksort for BtPoint32 (bitonic ≤32, partitioned >32)
  hull_batch.cu       #   batch_hull_dandc + batch_hull_dandc_mesh + batch_mesh_volume kernels
  test_warp_sort.cu   #   Test kernel for warp_sort_bp32
  beam_search.cu      #   V1 beam search kernels + compute_rv_for_tris device function
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
  test_beam.py        #   GPU beam search tests (cube convexity, L-shape decomposition) — V1
  test_beam_v2.py     #   GPU beam search V2 tests (3-kernel architecture)
  test_hull_mesh.py   #   D&C hull mesh extraction tests (batch_hull_dandc_mesh)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling (gaussian points)
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

# Run GPU beam search tests (ALWAYS ignore test_beam_v2 — it's under development
# and its CUDA errors poison subsequent tests in the same pytest session)
python -m pytest tests/ -v --ignore=tests/test_beam_v2.py

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
- **Hull algorithm** — D&C (Preparata-Hong) is the primary hull algorithm, exposed via `batch_hull_volume` and `batch_hull_dandc_mesh` (volume + mesh extraction). `hull_dandc_warp_mesh` returns both volume and the extracted hull mesh (vertices + triangles). Volume and mesh extraction use BFS over the half-edge graph (Bullet getVertexCopy pattern) — no DFS stacks, no stack overflow. BFS queue allocated from pool (n pointers), rewound between mesh extraction and volume. Sort scratch and D&C stack are also rewound. `dandc_scratch_bytes` reports `presort + max(sort_scratch, persistent + max(dc_stack, bfs_queue))`. Edge pool sized at `6n` half-edges (exact: max live = `6n-12` by Euler's formula for triangulated convex hulls, confirmed empirically). The incremental shared-memory hull (`hull.cuh`) is still used internally by V1 beam search for Rv computation.

### Python API

```python
# Hausdorff distance computation
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    dists = ctx.point_mesh_distances(points, vertices, triangles)
    h = ctx.hausdorff(sa, va, ta, sb, vb, tb)
    cost = ctx.pairwise_hausdorff(samples, s_off, verts, tris, t_off, v_off)

# Beam search decomposition (V1 — 5-kernel, incremental hull)
from coacd_gpu.beam import BeamContext, run_beam_coacd
with BeamContext(device=0) as ctx:
    parts = ctx.run(vertices, triangles, threshold=0.05)
# or:
parts = run_beam_coacd(vertices, triangles, threshold=0.05)

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

Replace MCTS with beam search over decomposition states. Search space: 3xN axis-aligned cuts (N per axis, default N=10 -> 30 total candidates). Beam width X (default 8). Each kernel does one job, with maximum SIMT parallelism within blocks.

### Algorithm Steps

No connected component analysis — each cut produces exactly two parts (positive/negative half), matching CoACD's approach. Metrics are computed per-part, not per-component.

**Step 0 — Initialize:**
Upload mesh, normalize to [-1,1]^3. Generate 3xN cutting planes. Start with all 30 cuts in work list.

**Step 1 — Compute Rv per part (1 kernel):**
Compute Rv for each part in each candidate, with 1 part per block. Build index of parts satisfying Rv < epsilon.

**Step 2 — Hausdorff validation (1 kernel):**
Compute Hausdorff distance for parts with Rv < epsilon. Build index of parts where max(Rv, Hausdorff) < epsilon.

**Step 3 — Worst selection + termination check (1 kernel):**
For each work list item, choose the part with the worst metric max(Rv, Hausdorff). If all parts in all candidates satisfy -> pick the one with fewest parts and terminate.

**Step 4 — Beam expansion (1 kernel):**
For X items in the work list, the worst part leads to 30 next-step candidates -> 30X total. Each thread block handles one candidate: clip, compute Rv, record cost scalar. Block 0 waits via atomic counter and selects the best X candidates.

Memory strategy: Keep at most max(30, X) meshes in global memory. Use references and only add 'cut plane intersection' to memory if needed. Trade recomputation for memory. Rv can be computed without storing convex hull (hull volume accumulated incrementally, only scalar kept). Hull approximated to ~256 vertices during search.

**Step 5 — Apply cuts + update (1 kernel):**
Track the indices to find which cuts were taken. Update meshes with the cuts. The work list now contains X items. Update bounding boxes and other metadata. Repeat from step 1.

### Concavity Metric: Rv (Volume-Ratio)

Rv = `(3 * |V_mesh - V_hull| / (4*pi))^(1/3) * k` where k = rv_k (default 0.3).

- **Mesh volume**: parallel signed-tetrahedra reduction `V = (1/6) * sum p0.(p1 x p2)`. For open meshes after clipping, cap volume correction via divergence theorem: `V_cap = (d/3) * |A_boundary|` from boundary edge loop shoelace signed area.
- **Hull volume**: incremental 3D convex hull with parallel visibility tests (all 256 threads test assigned faces) and thread-0 sequential topology updates. ~12KB shared memory workspace, capped at 256 input vertices for search. Full vertex count for termination checks only when approximate Rv is near threshold.
- **Cap triangulation**: fan cap triangles close meshes after each clip for correct signed-tet volumes in subsequent iterations.
- **Threshold**: compatible with CoACD semantics. Convex shape -> Rv ~ 0. Threshold 0.05 typical.
- **Scoring a cut**: `max(Rv_positive_half, Rv_negative_half)`. Beam search minimizes worst-case concavity.
- **Per-part bounding box planes**: cutting planes uniformly distributed within each part's triangle-vertex bbox (not all-vertex bbox, not global). Odd cuts_per_axis (e.g., 15) ensures midpoint is always a candidate. No snapping to vertex coordinates.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_width` (X) | 8 | Number of beam items to keep |
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

The beam search kernels should be high-level logic calling these reusable utilities. Each utility is a clean device function or kernel with a well-defined interface.

### A. Scalar Device Functions (per-thread, no sync)

| ID | Function | Status | File |
|----|----------|--------|------|
| A1 | `signed_tet_volume(p0, p1, p2) -> float` | EXISTS | geometry.cuh |
| A2 | `intersect_edge(v0, v1, plane) -> (ix, iy, iz)` | EXISTS | geometry.cuh |
| A3 | `point_triangle_dist(point, v0, v1, v2) -> float` | EXISTS | geometry.cuh |
| A4 | `rv_from_volumes(mesh_vol, hull_vol, rv_k) -> float` | EXTRACT — inlined as `cbrtf(3*|diff|/(4*pi))*k` in 3 places | beam_search.cu |

### B. Block-Level Reductions (all threads call, __syncthreads)

| ID | Function | Status | File |
|----|----------|--------|------|
| B1 | `block_reduce_sum(val, smem, tid) -> float` | EXISTS | reduce.cuh |
| B2 | `block_reduce_bbox(verts, n, offset, tid, smem, out_lo, out_hi)` | EXISTS | reduce.cuh |
| B3 | `block_reduce_max(val, smem, tid) -> float` | EXISTS | reduce.cuh |
| B4 | `block_reduce_count(flag, smem_i, tid) -> int` | EXISTS | reduce.cuh |

### C. Block-Level Mesh Operations (device functions, all threads, __syncthreads)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| C1 | `classify_vertices` | EXTRACT — inlined identically in evaluate_candidates and apply_cuts | `(verts, vo, vc, plane_abcd, signs, tid)` -> writes `signs[0..vc)` as +1/0/-1 |
| C2 | `split_triangles` | EXTRACT — ~110 lines duplicated in evaluate_candidates and apply_cuts | `(vp, tp, to, tc, vo, signs, plane, av, pos_tris, neg_tris, be_a, be_b, &pos_cnt, &neg_cnt, &new_v_cnt, &be_cnt, tid)` -> classifies + splits straddling triangles + collects boundary edges |
| C3 | `compute_tri_bbox` | EXTRACT — ~25 lines duplicated in evaluate_candidates and apply_cuts | `(vp, tp, tri_offset, tri_count, tid, smem, out_lo[3], out_hi[3])` -> per-part bbox from triangle vertex references |
| C4 | `compute_mesh_volume` | EXTRACT — embedded in compute_rv_for_tris and compute_part_costs | `(verts, tris, n_tris, tid, smem) -> float` — parallel signed-tet reduction, returns absolute volume |
| C5 | `collect_hull_vertices` | EXTRACT — flag + count + strided sample, ~50 lines in compute_rv_for_tris | `(verts, tris, n_tris, vert_offset, total_verts, flags, out_pts, max_pts, tid, smem, scratch) -> int n_hull` — flags triangle-referenced vertices, strides to <=MAX_HULL_VERTS |
| C6 | `compute_hull_volume` (shared memory) | EXISTS | hull.cuh — `(points, n_points, tid, ws) -> float`, ≤256 verts, ~12KB shared memory; used by beam search Rv |
| C7 | `compute_hull_volume_pool` (pool memory) | EXISTS | hull.cuh — `(points, n_points, tid, fv0..., max_faces) -> float`, unlimited verts, pool-allocated face arrays |

### D. Block-Level Boundary / Cap Operations (device functions)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| D1 | `compute_cap_volume` | EXTRACT — thread-0 loop tracing + shoelace, ~70 lines in compute_rv_for_tris | `(verts, be_a, be_b, n_be, plane, tid, scratch) -> float` — divergence theorem cap volume. Uses pool-allocated `used` flags for non-destructive tracing. Position-based matching (L1 tol 1e-4) |
| D2 | `trace_boundary_loops` | WRITE — currently loop tracing is embedded in D1 and D3, duplicated with differences | `(verts, be_a, be_b, n_be, out_loops, out_loop_lens, tid) -> int n_loops` — clean loop tracer by position matching. Consumable by both cap volume and fan cap |
| D3 | `fan_cap_triangulate` | EXTRACT — ~60 lines in apply_cuts | `(verts, be_a, be_b, n_be, plane, pos_tris, neg_tris, &pc, &nc, tid)` — traces loops, adds fan triangles to both halves with correct winding |

### E. Rv Computation (composite device functions)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| E1 | `compute_rv_closed` | WRITE — for closed meshes (no cap) | `(verts, tris, n_tris, vert_offset, total_verts, tid, hull_ws, smem, scratch, rv_k) -> float` — composes C4 + C5 + C6 + A4 |
| E2 | `compute_rv_open` | REFACTOR — current `compute_rv_for_tris` | `(verts, tris, n_tris, total_verts, vert_offset, plane, be_a, be_b, n_be, tid, hull_ws, smem, scratch, rv_k) -> float` — composes C4 + D1 + C5 + C6 + A4 |

### F. Hausdorff (kernels + host helpers)

| ID | Function | Status | File |
|----|----------|--------|------|
| F1 | `sample_surface` kernel | EXISTS | hausdorff.cu — area-weighted surface sampling |
| F2 | `point_mesh_distance` kernel | EXISTS | hausdorff.cu — brute-force point-to-triangle, one thread per point |
| F3 | `reduce_max` kernel | EXISTS | hausdorff.cu — shared-memory parallel max reduction |
| F4 | `compute_hausdorff_pair` (host orchestration) | WRITE — wire F1+F2+F3 into beam loop for step 2 | Bidirectional: sample A->mesh B, sample B->mesh A, take max |
| F5 | `pairwise_hausdorff` kernel | EXISTS | hausdorff.cu — batch merge cost matrix |

### G. Mesh Transform (kernels)

| ID | Function | Status | File |
|----|----------|--------|------|
| G1 | `normalize_mesh` kernel | EXISTS | mesh_transform.cu |
| G2 | `recover_coordinates` kernel | EXISTS | mesh_transform.cu |

### H. Selection / Search

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| H1 | `select_top_k` kernel | EXISTS | beam_search.cu — thread-0 insertion sort, single block |
| H2 | `block_select_top_k` | NEEDED if fusing into expansion kernel | Block-0 selection pattern for step 4 |

### I. Memory / Infrastructure

| ID | Function | Status | File |
|----|----------|--------|------|
| I1 | `pool_alloc(pool, size) -> void*` | EXISTS | common.cuh |
| I2 | `atomicMinF / atomicMaxF` | EXISTS | common.cuh |

### J. Legacy / Dead Code (to remove)

| ID | What | Why dead |
|----|------|----------|
| J1 | `compute_concavity_tris` in geometry.cuh | Bbox cube-root proxy, superseded by Rv |
| J2 | Global `planes` array generation + upload in beam.c | Planes derived from per-part bbox; global array allocated/uploaded but never read |
| J3 | `test_hull_volume` kernel + Python wiring | Debug-only diagnostic |

### Kernel Composition from Utilities

**Step 1 kernel** (Rv per part): `E1` or `E2` (one block per part)

**Step 2 kernel** (Hausdorff): `F1 -> F2 -> F3` per part (host orchestration via F4)

**Step 3 kernel** (worst + termination): `B3` reduction over parts per item

**Step 4 kernel** (expand): `C3 -> C1 -> C2 -> E2(pos) -> E2(neg) -> write cost` + block-0 `H2`

**Step 5 kernel** (apply cuts): `C3 -> C1 -> C2 -> D3 -> copy to pool`

## V2 Beam Search Architecture (3-Kernel, WIP)

V2 replaces V1's 5-kernel pipeline with 3 kernels using D&C hull everywhere, PartInfoV2/WorkItem data structures, and global DevicePool scratch allocation.

### Data Structures (common.cuh)

- **PartInfoV2**: extends PartInfo with `hull_vert_offset`, `hull_vert_count`, `hull_tri_offset`, `hull_tri_count`, `hausdorff` (-1 = not computed), `mesh_volume`, `hull_volume`.
- **WorkItem**: replaces BeamItem. Uses `part_indices[MAX_PARTS_PER_BEAM]` for indirect indexing into a global PartInfoV2 pool (no per-beam-item part copies). `worst_part_idx` indexes into `part_indices`, `worst_metric` = max(rv, hausdorff) of worst part.
- Old V1 structs (PartInfo, BeamItem) retained for backward compat.

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

- **Working**: Hull mesh extraction (`batch_hull_dandc_mesh`), cube convexity (early exit), scratch rewinds (sort/D&C/BFS), BFS graph traversal (no DFS stack overflow), all V1 tests still pass (122 tests).
- **Bug (WIP)**: `beam_run_v2` has an illegal memory access during the initial `batch_mesh_volume` call for hull volume computation. The hull triangle indices in the triangle pool are 0-based but need rebasing by `n_verts` since hull vertices are stored after original vertices. The `vert_offsets` fix to `{n_verts, n_verts}` was applied but the crash persists — needs further debugging.
- **Not yet working**: Full L-shape decomposition via V2 path.

## Current Implementation vs. Design Discrepancies

### 1. No Hausdorff in beam loop (design step 2)

**Design**: Dedicated kernel — Hausdorff for parts where Rv < eps.

**Current**: Hausdorff kernels exist but are not wired into the beam loop. Termination uses Rv alone.

### 2. Worst selection / termination (design step 3)

**Design**: Per-item worst part based on max(Rv, Hausdorff). Single kernel.

**Current**: `compute_part_costs` finds worst per beam item (Rv only, sequential per-part loop, 1 block per beam item). Termination on host.

### 3. Expansion kernel architecture (design step 4)

**Design**: Single fused kernel — 30X blocks, block-0 selects top X.

**Current**: Three separate kernels with host sync: evaluate_candidates, select_top_k, apply_cuts.

### 4. Memory model (design step 4)

**Design**: Keep at most max(30, X) meshes. Use references. Trade recomputation for memory.

**Current**: Double-buffered pools with full vertex/triangle copies per part. Each part copies ALL parent vertices (superset). Scratch materializes full clipped meshes for all 30X candidates.

### 5. Global planes array is dead code

`generate_planes()` uploads a global planes array. Both `evaluate_candidates` and `apply_cuts` reconstruct planes from per-part bounding boxes and ignore the global array. The plane_idx is used only to derive axis + cut index for the per-part bbox formula.

### 6. compute_part_costs hangs on large meshes

`evaluate_candidates` uses 256-vert approximate hull (correct). `compute_part_costs` attempts full-precision hull with ALL vertices unconditionally — hangs on 20K+ vertices because thread-0 sequential hull is O(N^2 * F).

Design intent: approximate (256 verts) during search, full precision only when approximate Rv is near threshold for termination check.

## Implementation Lessons

### Vertex Classification Must Be Inlined

The `classify_vertex()` function call produced incorrect results in `apply_cuts` context — returning 0 (on-plane) for vertices clearly not on the plane (val=1.45). Root cause unconfirmed (suspected nvcc optimization issue). Inlining resolved it. Keep classification inlined at call sites rather than extracting to C1.

### Position-Based Boundary Edge Matching

Parallel triangle processing creates duplicate intersection points for shared edges — same 3D position but different vertex indices. Boundary loop tracing must match by 3D position (L1 distance < 1e-4f tolerance), not vertex index.

### Cap Volume Sign Correction

Cap always reduces V_open magnitude (closes a hole). Must oppose V_open sign: `|V_open + copysignf(|V_cap|, -V_open)|`. Using `|V_open| + |V_cap|` double-counts.

### On-Plane Triangle Assignment

When cutting plane passes through vertices (sign=0), triangles with ALL vertices on-plane must be assigned by `dot(face_normal, plane_normal)`, not defaulting to positive half.

### Per-Part Bbox from Triangle Vertices

After apply_cuts, parts carry ALL parent vertices including unreferenced ones. Bbox must iterate over triangle vertex references `triangle_pool[(to+t)*3+e]`, not all vertices `vertex_pool[(vo+v)*3+k]`.

### DevicePool Capacity is 32-bit

`DevicePool.capacity` is `unsigned int`. With >4GB free GPU memory, 70% exceeds 4GB and wraps. Cap scratch at 4GB.

### Convex Hull: Shared vs Pool Memory

Shared-memory hull (`compute_hull_volume`): ~12KB, capped at 256 verts, fast. Pool-memory hull (`compute_hull_volume_pool`): unlimited verts, slower due to global memory latency. Use shared-memory version for search, pool version only for final validation of borderline cases.

### Strided Hull Vertex Sampling

First-256-by-atomicAdd is biased by thread order, missing extreme vertices. Strided sampling (count flagged, compute stride = total/MAX_HULL_VERTS, collect every K-th) gives uniform spatial coverage.

### QuickHull: Dead Point Redistribution Must Check All Faces

With single-face assignment (each point assigned to the face with maximum positive distance), dead points from visible faces may still be above old non-visible faces after apex insertion. Checking only new faces (the "optimization") incorrectly discards those points, producing a significantly smaller hull volume. Fix: check all faces during redistribution, then rebuild the entire face stack by scanning all faces.

### QuickHull: `max_faces - WARP_SIZE` Guard Underflows for Small n

`if (n_faces >= max_faces - WARP_SIZE) break` with WARP_SIZE=32: for small n (e.g., n=8: max_faces=24, 24-32=-8), this evaluates as n_faces >= negative number — always true — causing immediate loop exit after the initial tetrahedron. Fix: use `max_faces` directly.

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

V2's `beam_expansion` creates 2 new parts per block per iteration. Across `max_iterations × beam_width × num_planes` total expansion blocks, the original V1-style formula `(n_verts + 4096) × beam_width × 8` is far too small (e.g., 480 blocks × 4096 hull tris/block >> tri_capacity). New formula: `total_blocks × (max_hull_tris_per_part × 2 + n_tris × 3)`, capped at 4M entries. Hull output limits reduced to `max_hull_verts_per_part=64`, `max_hull_tris_per_part=128`.

### V2 Kernel Error Reporting Pattern

All three V2 kernels (`beam_expansion`, `beam_hausdorff_parts`, `beam_termination`) take `int* kernel_error`. Errors are written via `atomicOr(kernel_error, KERR_*)`. Host zeros `d_kerr` before each launch, then issues `cuMemcpyDtoHAsync` immediately after the `cuLaunchKernel` call (before `cuStreamSynchronize`) so the download overlaps with kernel execution. Bit flags: `KERR_SCRATCH_OOM=1`, `KERR_SPLIT_OOM=2`, `KERR_HULL_PTS_OOM=4`, `KERR_POOL_OOM=8`, `KERR_HULL_ERR=16`.

### Scratch Query Ordering Does Not Help batch_hull_volume

Moving `query_dandc_scratch` before the large `cuMemcpyHtoDAsync` calls (to avoid syncing after copies) was tested and made performance slightly worse. The `cuMemAlloc` calls themselves may already serialize, so reordering provides no benefit. Keep the query after copies for now.

## Current Status

### Working (V1 — production path)
- Full V1 beam search pipeline: init -> normalize -> evaluate -> select -> apply -> recover -> download
- Rv concavity metric with GPU convex hull (incremental hull, parallel visibility)
- Fan cap triangulation closes meshes after each clip
- Multi-iteration decomposition with double-buffered pools
- Single CPython extension (abi3 cp310+) with all GPU functionality
- Cube correctly identified as convex (Rv ~ 0, 1 part)
- L-shape decomposed into exactly 2 convex boxes at threshold 0.05
- GPU beam search tests pass (cube convexity, L-shape decomposition, beam params)
- `batch_mesh_volume` GPU kernel (divergence theorem, watertight meshes) — tested
- `batch_hull_volume` algo=2 (D&C warp) — tested and passing
- Full V1 pytest suite: 118 passed, 0 failures

### Working (V2 — new architecture, partial)
- D&C hull mesh extraction (`bt_extractMesh`, `hull_dandc_warp_mesh`, `batch_hull_dandc_mesh`) — tested (cube 8v/12t, tetra 4v/4t, gaussian)
- V2 data structures (PartInfoV2, WorkItem) — defined in common.cuh, beam.h
- Two-warp reductions (`twowarp_reduce_sum/max/count`) — in reduce.cuh
- Scratch rewinds (sort/D&C/BFS) + BFS graph traversal replacing DFS in D&C hull
- Global DevicePool scratch allocation (`warppool_from_global`) — in hull_warp_common.cuh
- V2 kernel compilation: beam_expansion, beam_hausdorff_parts, beam_termination — compiles
- V2 host code (`beam_run_v2`) and Python API (`run_v2`, `run_beam_coacd_v2`) — compiles
- V2 cube convexity early-exit path — **working** (5/6 tests pass)
- Kernel-side error reporting: `KERR_*` bit flags in all 3 V2 kernels (`KERR_SCRATCH_OOM=1`, `KERR_SPLIT_OOM=2`, `KERR_HULL_PTS_OOM=4`, `KERR_POOL_OOM=8`, `KERR_HULL_ERR=16`); host async-downloads and checks after each launch
- Pool OOM bounds checking in `beam_expansion` (hull pre-alloc and mesh copy both checked)
- `CHECK_CU` macro now includes `beam.c:LINE` in error messages for faster diagnosis

### V2 Bugs Fixed
- **Scipy hull winding**: `hull.simplices` have inconsistent winding → signed-tet sum ≈ 0 → rv non-zero for convex shapes → no early exit → pool overflow. Fixed in `beam.py`: orient each hull triangle outward using centroid dot-product check.
- **Triangle pool overflow in `beam_expansion`**: `max_hull_tris_per_part=2048` × 480 blocks/iteration exhausted `tri_capacity`. Fixed by reducing to 128 and computing pool capacities from expected total expansion blocks (`max_iterations × beam_width × num_planes`) capped at 4M entries.
- Root cause of illegal memory access confirmed via `compute-sanitizer`: `bt_extractMesh` writing to `out_tris` past end of triangle pool in block 324/480 (second iteration).

### V2 Bugs Remaining
- `test_decomposition` (L-shape, `len(parts) >= 2`) returns 0 parts — likely a kernel error or the download/result path is broken after one expansion iteration. Needs investigation with the new kernel error codes.
- Hull mesh winding: `bt_extractMesh` produces mixed winding (mesh volume via signed tet = 0.667 for unit cube instead of 1.0). The D&C volume (via int128 arithmetic) is correct. Winding consistency in extracted mesh needs investigation.

### Not Yet Implemented
- V2 L-shape decomposition end-to-end (expansion path has a bug, see above)
- Hausdorff validation in V2 beam loop (kernel exists, host wiring incomplete)
- Merge post-processing
- Vertex compaction (parts carry superset of vertices)
- Large mesh support (V1 compute_part_costs hangs on 20K+ vertices; V2 uses D&C with no vertex limit)
- Utility function extraction (C1-C5, D1-D3 still inlined/duplicated in V1 kernels)
- `__cuda_array_interface__` support for GPU tensor input
- Old V1 cleanup (remove old kernels after V2 is fully working)
