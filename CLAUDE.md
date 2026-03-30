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
  hull_warp.cuh       #   Umbrella include for warp-parallel hull algorithms
  hull_warp_common.cuh#   WarpPool allocator + warp reductions (shared by QuickHull & D&C)
  hull_quickhull.cuh  #   QuickHull warp algorithm (algo=1)
  hull_dandc.cuh      #   Preparata-Hong D&C hull volume (algo=2, Bullet port, warp sort + lane-0 D&C)
  warp_sort.cuh       #   Warp-cooperative quicksort for BtPoint32 (bitonic ≤32, partitioned >32)
  hull_batch.cu       #   batch_hull_volume kernel dispatcher (algo 0/1/2)
  test_warp_sort.cu   #   Test kernel for warp_sort_bp32
  beam_search.cu      #   Beam search kernels + compute_rv_for_tris device function
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
  test_beam.py        #   GPU beam search tests (cube convexity, L-shape decomposition)
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

# Run GPU beam search tests
python -m pytest tests/ -v

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

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/kernels.cu` (which `#include`s all `.cuh`/`.cu` modules) -> fatbin -> xxd-style C header, then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake involved.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **Single extension** — all GPU functionality (beam search + Hausdorff + merge cost) in one `_gpu` module. No cmake, no ctypes.
- **Native CPython extension, not ctypes** — ctypes has fragile import path resolution, no type safety, no proper Python object lifecycle. The torchoptix pattern (native CPython extension with embedded fatbin) is the standard approach.

### Python API

```python
# Hausdorff distance computation
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    dists = ctx.point_mesh_distances(points, vertices, triangles)
    h = ctx.hausdorff(sa, va, ta, sb, vb, tb)
    cost = ctx.pairwise_hausdorff(samples, s_off, verts, tris, t_off, v_off)

# Beam search decomposition
from coacd_gpu.beam import BeamContext, run_beam_coacd
with BeamContext(device=0) as ctx:
    parts = ctx.run(vertices, triangles, threshold=0.5)
# or:
parts = run_beam_coacd(vertices, triangles, threshold=0.5)
```

Both `Context` and `BeamContext` share the same underlying `_gpu` extension and CUDA context.

## Beam Search Algorithm Design

### Overview

Replace MCTS with beam search over decomposition states. Search space: 3xN axis-aligned cuts (N per axis, default N=10 -> 30 total candidates). Beam width X (default 8). Each kernel does one job, with maximum SIMT parallelism within blocks.

### Algorithm Steps

**Step 0 — Initialize:**
Upload mesh, normalize to [-1,1]^3. Generate 3xN cutting planes. Start with all 30 cuts in work list.

**Step 1 — Connected Components (1 kernel):**
For each item in the work list, scan and compute connected components. 1 candidate per block. Uses parallel union-find with path compression via atomicMin.

**Step 2 — Compute Rv per component (1 kernel):**
Compute Rv for each component in each candidate, with 1 component per block. Use atomic operations to build an index of components satisfying Rv < epsilon.

**Step 3 — Hausdorff validation (1 kernel):**
Compute Hausdorff distance for components with Rv < epsilon. Build an index of components where max(Rv, Hausdorff) < epsilon.

**Step 4 — Worst selection + termination check (1 kernel):**
For each work list item, choose the component with the worst metric max(Rv, Hausdorff). If all components in all candidates satisfy -> pick the one with fewest components and terminate.

**Step 5 — Beam expansion (1 kernel):**
For X items in the work list, the worst part leads to 30 next-step candidates -> 30X total. Each thread block handles one candidate: clip, compute Rv, record cost scalar. Block 0 waits via atomic counter and selects the best X candidates.

Memory strategy: Keep at most max(30, X) meshes in global memory. Use references and only add 'cut plane intersection' to memory if needed. Trade recomputation for memory. Rv can be computed without storing convex hull (hull volume accumulated incrementally, only scalar kept). Hull approximated to ~256 vertices during search.

**Step 6 — Apply cuts + update (1 kernel):**
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
| C6 | `compute_hull_volume` (shared memory) | EXISTS | hull.cuh — `(points, n_points, tid, ws) -> float`, <=256 verts, ~12KB shared memory |
| C7 | `compute_hull_volume_pool` (pool memory) | EXISTS | hull.cuh — `(points, n_points, tid, fv0..., max_faces) -> float`, unlimited verts, pool-allocated face arrays |

### D. Block-Level Boundary / Cap Operations (device functions)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| D1 | `compute_cap_volume` | EXTRACT — thread-0 loop tracing + shoelace, ~70 lines in compute_rv_for_tris | `(verts, be_a, be_b, n_be, plane, tid, scratch) -> float` — divergence theorem cap volume. Uses pool-allocated `used` flags for non-destructive tracing. Position-based matching (L1 tol 1e-4) |
| D2 | `trace_boundary_loops` | WRITE — currently loop tracing is embedded in D1 and D3, duplicated with differences | `(verts, be_a, be_b, n_be, out_loops, out_loop_lens, tid) -> int n_loops` — clean loop tracer by position matching. Consumable by both cap volume and fan cap |
| D3 | `fan_cap_triangulate` | EXTRACT — ~60 lines in apply_cuts | `(verts, be_a, be_b, n_be, plane, pos_tris, neg_tris, &pc, &nc, tid)` — traces loops, adds fan triangles to both halves with correct winding |

### E. Connected Components (device functions, all threads)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| E1 | `build_edge_adjacency` | WRITE | `(tris, n_tris, edge_table, table_size, tid)` — hash table: canonical edge -> (tri_a, tri_b). Parallel insert via atomicCAS |
| E2 | `union_find_init` | WRITE | `(labels, n, tid)` — `labels[i] = i` for all i |
| E3 | `union_find_iterate` | WRITE | `(labels, tris, n_tris, edge_table, tid) -> bool changed` — one pass of parallel union with path compression via atomicMin |
| E4 | `compact_components` | WRITE | `(labels, n_tris, comp_offsets, comp_counts, tid) -> int n_components` — per-component triangle index lists |

### F. Rv Computation (composite device functions)

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| F1 | `compute_rv_closed` | WRITE — for closed meshes (no cap) | `(verts, tris, n_tris, vert_offset, total_verts, tid, hull_ws, smem, scratch, rv_k) -> float` — composes C4 + C5 + C6 + A4 |
| F2 | `compute_rv_open` | REFACTOR — current `compute_rv_for_tris` | `(verts, tris, n_tris, total_verts, vert_offset, plane, be_a, be_b, n_be, tid, hull_ws, smem, scratch, rv_k) -> float` — composes C4 + D1 + C5 + C6 + A4 |

### G. Hausdorff (kernels + host helpers)

| ID | Function | Status | File |
|----|----------|--------|------|
| G1 | `sample_surface` kernel | EXISTS | hausdorff.cu — area-weighted surface sampling |
| G2 | `point_mesh_distance` kernel | EXISTS | hausdorff.cu — brute-force point-to-triangle, one thread per point |
| G3 | `reduce_max` kernel | EXISTS | hausdorff.cu — shared-memory parallel max reduction |
| G4 | `compute_hausdorff_pair` (host orchestration) | WRITE — wire G1+G2+G3 into beam loop for step 3 | Bidirectional: sample A->mesh B, sample B->mesh A, take max |
| G5 | `pairwise_hausdorff` kernel | EXISTS | hausdorff.cu — batch merge cost matrix |

### H. Mesh Transform (kernels)

| ID | Function | Status | File |
|----|----------|--------|------|
| H1 | `normalize_mesh` kernel | EXISTS | mesh_transform.cu |
| H2 | `recover_coordinates` kernel | EXISTS | mesh_transform.cu |

### I. Selection / Search

| ID | Function | Status | Signature |
|----|----------|--------|-----------|
| I1 | `select_top_k` kernel | EXISTS | beam_search.cu — thread-0 insertion sort, single block |
| I2 | `block_select_top_k` | NEEDED if fusing into expansion kernel | Block-0 selection pattern for step 5 |

### J. Memory / Infrastructure

| ID | Function | Status | File |
|----|----------|--------|------|
| J1 | `pool_alloc(pool, size) -> void*` | EXISTS | common.cuh |
| J2 | `atomicMinF / atomicMaxF` | EXISTS | common.cuh |

### K. Legacy / Dead Code (to remove)

| ID | What | Why dead |
|----|------|----------|
| K1 | `compute_concavity_tris` in geometry.cuh | Bbox cube-root proxy, superseded by Rv |
| K2 | Global `planes` array generation + upload in beam.c | Planes derived from per-part bbox; global array allocated/uploaded but never read |
| K3 | `test_hull_volume` kernel + Python wiring | Debug-only diagnostic |

### Kernel Composition from Utilities

**Step 1 kernel** (CC): `E1 -> E2 -> repeat(E3) -> E4`

**Step 2 kernel** (Rv per component): `F1` (one block per component)

**Step 3 kernel** (Hausdorff): `G1 -> G2 -> G3` per component (host orchestration via G4)

**Step 4 kernel** (worst + termination): `B3` reduction over components per item

**Step 5 kernel** (expand): `C3 -> C1 -> C2 -> F2(pos) -> F2(neg) -> write cost` + block-0 `I2`

**Step 6 kernel** (apply cuts): `C3 -> C1 -> C2 -> D3 -> copy to pool`

## Current Implementation vs. Design Discrepancies

### 1. No Connected Components (design step 1)

**Design**: Dedicated kernel — for each item, compute CC. 1 candidate per block.

**Current**: Completely absent. Each clip half is one part regardless of connectivity.

### 2. Rv is per-half, not per-component (design step 2)

**Design**: Separate kernel — Rv for each connected component, 1 component per block.

**Current**: Rv computed per clip-half (fused into `evaluate_candidates`), not per component.

### 3. No Hausdorff in beam loop (design step 3)

**Design**: Dedicated kernel — Hausdorff for components where Rv < eps.

**Current**: Hausdorff kernels exist but are not wired into the beam loop. Termination uses Rv alone.

### 4. Worst selection / termination (design step 4)

**Design**: Per-item worst component based on max(Rv, Hausdorff). Single kernel.

**Current**: `compute_part_costs` finds worst per beam item (Rv only, sequential per-part loop, 1 block per beam item). Termination on host.

### 5. Expansion kernel architecture (design step 5)

**Design**: Single fused kernel — 30X blocks, block-0 selects top X.

**Current**: Three separate kernels with host sync: evaluate_candidates, select_top_k, apply_cuts.

### 6. Memory model (design step 5)

**Design**: Keep at most max(30, X) meshes. Use references. Trade recomputation for memory.

**Current**: Double-buffered pools with full vertex/triangle copies per part. Each part copies ALL parent vertices (superset). Scratch materializes full clipped meshes for all 30X candidates.

### 7. Global planes array is dead code

`generate_planes()` uploads a global planes array. Both `evaluate_candidates` and `apply_cuts` reconstruct planes from per-part bounding boxes and ignore the global array. The plane_idx is used only to derive axis + cut index for the per-part bbox formula.

### 8. compute_part_costs hangs on large meshes

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

The D&C algorithm is a faithful port of Bullet's `btConvexHullComputer` (Ole Kniemeyer, MAXON, zlib license). Three phases: (1) warp-parallel pre-sort (AABB via warp min/max reduction, Point32 conversion strided across lanes), (2) warp-cooperative sort via `warp_sort_bp32` (all 32 lanes), (3) D&C merge + volume extraction on lane 0. Key elements: int32 coordinates with exact Int128/Rational64/Rational128 predicates, recursive `computeInternal` D&C, `mergeProjection` for 2D bridge finding, `findMaxAngle` with exact cotangent comparison, `findEdgeForCoplanarFaces` for coplanar handling, and the full `merge` function with interior edge deletion via `removeEdgePair`. Memory is allocated from WarpPool bump allocator with free-list recycling for edges. Volume is extracted by walking the half-edge graph and summing signed tetrahedra in integer coordinates, then converting back via the scaling factor. Requires 2MB scratch per hull, 32KB thread stack (via `cuCtxSetLimit`), and block size 64 (DANDC_BLOCK_SIZE).

### Warp Sort (warp_sort.cuh)

Warp-cooperative quicksort for BtPoint32 arrays. All 32 lanes participate. Uses a global-memory workspace of the same size as the input. Segments ≤32 elements use a bitonic sorting network; larger segments use quicksort partitioning with a median-of-32-samples pivot. Three-way partition via `ws_cmp` (returns -1/0/1): items < pivot go left, items > pivot go right, items == pivot are filled in the middle gap by warp-parallel fill. This avoids worst-case O(n²) on all-equal or many-duplicate inputs. Uses `__ballot_sync`/`__popc` for warp-wide prefix sums. Explicit stack (max depth 2048) in global memory, `sp` in lane-0 register broadcast via `__shfl_sync`.

### pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

## Current Status

### Working
- Full beam search pipeline: init -> normalize -> evaluate -> select -> apply -> recover -> download
- Rv concavity metric with GPU convex hull (incremental hull, parallel visibility)
- Fan cap triangulation closes meshes after each clip
- Multi-iteration decomposition with double-buffered pools
- Single CPython extension (abi3 cp310+) with all GPU functionality
- Cube correctly identified as convex (Rv ~ 0, 1 part)
- L-shape decomposed into exactly 2 convex boxes at threshold 0.05
- GPU beam search tests pass (cube convexity, L-shape decomposition, beam params)
- `batch_mesh_volume` GPU kernel (divergence theorem, watertight meshes) — tested
- `batch_hull_volume` algo=0 (incremental, ≤256 pts), algo=1 (QuickHull warp), algo=2 (D&C warp) — all tested and passing
- Full pytest suite: 121 passed (2 pre-existing failures in algo=0/1 gaussian)

### Not Yet Implemented
- Connected components after clipping (design step 1)
- Hausdorff validation in beam loop (design step 3; kernels exist, not wired in)
- Fused expansion kernel with block-0 selection (design step 5)
- Merge post-processing
- Vertex compaction (parts carry superset of vertices)
- Large mesh support (compute_part_costs hangs on 20K+ vertices)
- Utility function extraction (C1-C5, D1-D3 still inlined/duplicated in monolithic kernels)
