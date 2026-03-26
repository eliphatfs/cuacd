# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition (reference C++ implementation, SIGGRAPH 2022).
2. **coacd_gpu** — GPU-accelerated convex decomposition. Two components:
   - **GPU extension** (`cuda/beam_kernels.cu`, `csrc/beam.c`, `csrc/beam_module.c`) — All GPU functionality: beam search decomposition, Hausdorff distance, pairwise merge cost. Single CPython extension (abi3, cp310+) via CUDA driver API.
   - **Pure Python CoACD** (`coacd_gpu/coacd/`) — Pure Python reimplementation using scipy/triangle. No C++ build required.

## Repository Layout

```
setup.py              # Build config: setuptools builds _gpu extension
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device code (compiled to single fatbin)
  kernels.cu          #   Root compilation unit — includes all modules
  common.cuh          #   Constants, data structures, pool allocator, atomics
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   Edge intersection, point-triangle distance, concavity metrics
  hull.cuh            #   Incremental convex hull with parallel visibility tests
  beam_search.cu      #   Beam search kernels + compute_rv_for_tris device function
  hausdorff.cu        #   Hausdorff kernels (point_mesh_distance, reduce_max, pairwise)
  mesh_transform.cu   #   Normalize/recover coordinate kernels
csrc/                 # C host code
  beam.h/.c           #   Host implementation (CUDA driver API) — beam search + Hausdorff
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (Hausdorff API) + re-exports BeamContext
  beam.py             #   Python API for beam search (imports _gpu extension)
  coacd/              #   Pure Python CoACD implementation
tests/                # All tests
  test_extension.py   #   GPU smoke tests (Hausdorff, pairwise — standalone script)
  test_beam.py        #   GPU beam search tests (cube convexity, L-shape decomposition)
  test_clip.py ...    #   Pure Python CoACD unit tests
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

# Run pure Python CoACD tests (fast)
python -m pytest tests/ -v

# Run all tests including slow MCTS/pipeline tests
python -m pytest tests/ -v --slow

# Run a specific test
python -m pytest tests/test_clip.py -v

# Compare C++ vs Python CoACD on Octocat
python compare_octocat.py
```

Dependencies: `numpy`, `scipy`, `triangle`, `pytest` (test only), `trimesh` (comparison only).

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

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/kernels.cu` (which `#include`s all `.cuh`/`.cu` modules) → fatbin → xxd-style C header, then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake involved.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x–12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **Single extension** — all GPU functionality (beam search + Hausdorff + merge cost) in one `_gpu` module. No cmake, no ctypes.

### GPU Kernels (`cuda/`)

CUDA device code organized into modular files, compiled as a single fatbin via `cuda/kernels.cu`:

- **`common.cuh`** — Constants (`BLOCK_SIZE`, `MAX_BEAM`, etc.), data structures (`PartInfo`, `BeamItem`, `DevicePool`), `pool_alloc`, `atomicMinF`/`atomicMaxF`.
- **`reduce.cuh`** — `block_reduce_sum`, `block_reduce_bbox`.
- **`geometry.cuh`** — `signed_tet_volume`, `intersect_edge`, `point_triangle_dist` (Eberly's method), `compute_concavity_tris` (legacy bbox metric).
- **`hull.cuh`** — `HullWorkspace`, `compute_hull_volume` — incremental 3D convex hull with SIMT-parallel visibility tests and thread-0 topology updates. Returns hull volume in ~12KB of shared memory.

**Beam search kernels (`beam_search.cu`):**
- **`compute_rv_for_tris`** — Device function: full Rv computation. Parallel signed-tet mesh volume + divergence theorem cap volume + parallel convex hull volume → Rv formula.
- **`evaluate_candidates`** — Grid `(beams × planes)`. Per block: classify vertices against plane, split triangles, collect boundary edges, compute Rv for each half via `compute_rv_for_tris`.
- **`select_top_k`** — Single block, thread 0 only. Insertion-sort to pick best `beam_width` candidates.
- **`apply_cuts`** — Grid `(winners)`. Copy unchanged parts, re-clip worst part, add fan cap triangles to close meshes, compute and cache Rv for new parts, write to double-buffered pool.
- **`compute_part_costs`** — Grid `(beams)`. Compute Rv (mesh volume + hull volume) for parts needing it (initial mesh), find worst per beam item.
**Hausdorff/merge kernels (`hausdorff.cu`):**
- **`sample_surface`** — Area-weighted surface sampling (for Hausdorff validation, not yet wired in).
- **`point_mesh_distance`** — brute-force point-to-triangle (Eberly's method), one thread per point. Has `vert_offset` parameter for pool-based usage.
- **`reduce_max`** — shared-memory parallel max reduction.
- **`pairwise_hausdorff`** — batch merge cost matrix with `atomicMaxF` float CAS.

**Mesh transform kernels (`mesh_transform.cu`):**
- **`normalize_mesh`** / **`recover_coordinates`** — Normalize to [-1,1]³ and recover.

### Host Orchestration (`csrc/beam.c`)

Single context manages both beam search and Hausdorff functionality. All kernel function handles resolved from one fatbin module at init time.

**Beam search flow:**
```
Input mesh → Upload to pool_a → Normalize to [-1,1]³
  → Generate 3×N axis-aligned cutting planes
  → Compute initial Rv (compute_part_costs: mesh volume + hull volume)
  → If already below threshold → return single part
  → Beam loop:
      1. evaluate_candidates: clip worst part by each plane, compute Rv for both halves
      2. select_top_k: pick best beam_width candidates
      3. apply_cuts: materialize cuts, add cap triangles, compute+cache Rv
      4. Swap pool_cur ↔ pool_nxt, swap parts/beam buffers
      5. compute_part_costs: read cached Rv, find worst part per beam item
      6. If best beam item's worst_cost ≤ threshold → done
  → Recover coordinates → Download parts
```

**Memory management:**
- **Double-buffered mesh pools**: pool_a and pool_b alternate each iteration. Read from current, write to next, swap.
- **Bump-allocated scratch pool**: single large device buffer (~50% of free GPU memory). Reset offset to 0 between kernel launches. Thread-0-only allocation with shared memory broadcast.
- **Per-iteration alloc/free**: parts and beam metadata arrays are freshly allocated each iteration (old ones freed after swap).

### Concavity Metric: Rv (Volume-Ratio)

The beam search uses Rv = `(3 × |V_mesh - V_hull| / (4π))^(1/3) × k` where k = rv_k (default 0.3). This measures the difference between the mesh volume and its convex hull volume.

- **Mesh volume**: parallel signed-tetrahedra reduction. For open meshes (after clipping), a cap volume correction is added via the divergence theorem: `V_cap = (d/3) × |A_boundary|`, computed from boundary edge loop shoelace signed area.
- **Hull volume**: incremental 3D convex hull with parallel visibility tests (all 256 threads test their assigned faces) and thread-0 sequential topology updates (horizon edge finding, face removal/addition). Runs in ~12KB of shared memory (`HullWorkspace`), capped at 256 input vertices.
- **Cap triangulation**: `apply_cuts` adds fan cap triangles after each clip to close meshes, ensuring correct signed-tet volumes in subsequent iterations.
- **Threshold meaning**: compatible with original CoACD threshold semantics. For meshes normalized to [-1,1]³, a convex shape has Rv ≈ 0. Threshold 0.05 is typical.
- **Scoring a cut**: `max(Rv_positive_half, Rv_negative_half)`. The beam search minimizes worst-case concavity.

### Pool Allocator Pattern

All scratch memory in kernels is allocated from a global bump pool. **Critical pattern**: only thread 0 calls `pool_alloc()`, stores the pointer in `__shared__` memory, then all threads read it after `__syncthreads()`:

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

If all 256 threads call `pool_alloc`, each gets a different offset (256× memory waste) and threads write to different arrays, causing data corruption. This was a critical bug that was fixed.

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

## Pure Python CoACD (`coacd_gpu/coacd/`)

Pure Python reimplementation of CoACD. Uses `scipy` (Qhull), `triangle` (CDT), optionally `coacd_gpu` for GPU Hausdorff.

| File | Role |
|------|------|
| `_geometry.py` | `Plane`, `mesh_volume`, `normalize`, `recover`, `pca_align` |
| `_mesh.py` | `Mesh` class wrapping vertices + triangles + `convex_hull()` via scipy |
| `_sampling.py` | Area-weighted surface sampling |
| `_cost.py` | `compute_rv` (volume), `compute_hb` (Hausdorff via KD-tree), `compute_hcost` |
| `_clip.py` | Plane-mesh clipping with CDT cap triangulation |
| `_mcts.py` | MCTS search (Node/State/Part), UCB1, Rv-only rollout, ternary refinement |
| `_merge.py` | Greedy agglomerative merge |
| `_pipeline.py` | `run_coacd()` orchestration |

```python
from coacd_gpu.coacd import run_coacd
parts = run_coacd(vertices, triangles, threshold=0.05)
```

## Design Deviations from Original Plan

The implementation diverges from the original GPU beam search plan in several significant ways. These are intentional engineering decisions made during implementation.

### 1. Concavity Metric: Rv via GPU Convex Hull (Implemented)

**Plan**: Rv = `(3 * |V_mesh - V_hull| / (4π))^(1/3) * k`, requiring per-part convex hull volume on GPU.

**Implementation**: Rv is computed as planned. The key design that made it work:
- **Parallel visibility, sequential topology**: All 256 threads test face visibility in parallel. Thread 0 alone does horizon edge finding, face removal/addition. This avoids `__syncthreads()` deadlocks in conditional loops.
- **Shared memory workspace**: ~12KB `HullWorkspace` in shared memory (not pool). Faces stored as SoA (fv0/fv1/fv2 arrays). Capped at 256 input vertices.
- **Cap volume via divergence theorem**: For open meshes after clipping, `V_cap = (d/3) × |A_boundary|` computed from boundary edge loop shoelace signed area. Non-destructive marking via pool-allocated flag array.
- **Fan cap triangulation**: `apply_cuts` closes meshes with fan cap triangles after each cut, ensuring correct signed-tet volume in subsequent iterations.

### 2. Cap Volume via Divergence Theorem (Implemented)

**Plan**: `V_cap = (-d/3) × A_net` from boundary loop signed areas.

**Implementation**: Implemented as planned. Boundary edges collected during triangle splitting (parallel), loop tracing and shoelace area computed by thread 0 (sequential, O(n_boundary²)). Fan cap triangulation in `apply_cuts` closes meshes for correct volumes in future iterations.

### 3. GPU Convex Hull: Incremental with Parallel Visibility (Implemented)
- Incremental hull needs sequential face updates after each vertex insertion
**Plan**: Port QuickHull to GPU.

**Implementation**: Incremental hull with SIMT-parallel visibility tests. Not QuickHull. The parallel visibility test (all threads test assigned faces) avoids `__syncthreads()` in conditional loops. Thread 0 handles sequential topology (horizon edges via brute-force search of visible faces, O(9 × n_vis²) per insertion — fast for typical n_vis < 20). Volume accumulated incrementally via signed-tet delta on face removal/addition.

### 4. Vertex Classification Inlined (Not a Function Call)

**Plan**: `classify_vertex()` device function.

**Implementation**: Vertex classification is inlined at each call site:
```cuda
float val = pa * vx + pb * vy + pc * vz + pd;
signs[v] = (val > EPS) ? 1 : ((val < -EPS) ? -1 : 0);
```

**Why changed**: The `classify_vertex()` function call produced incorrect results in the `apply_cuts` kernel context — returning 0 (on-plane) for vertices clearly not on the plane (val=1.45). Root cause unconfirmed (suspected nvcc optimization issue with function parameter passing in complex kernels). Inlining resolved it immediately.

### 5. Pool Allocator: Thread-0-Only Pattern

**Plan**: `pool_alloc()` callable from any thread, with atomic bump pointer.

**Implementation**: `pool_alloc()` is only called by thread 0 within each block. Pointers are broadcast to all threads via `__shared__` memory.

**Why changed**: When all 256 threads called `pool_alloc`, each atomically bumped the offset and received a *different* pointer. Thread 0 wrote data to its allocation, thread 1 wrote to a different allocation, etc. Later when the block cooperatively processed the data, threads read from thread 0's allocation, seeing uninitialized memory for indices > 0. This was the root cause of the signs corruption bug (7 of 8 vertices classified as zero).

### 6. Connected Components Not Implemented

**Plan**: Parallel union-find with path compression in per-block scratch memory to find disconnected pieces after clipping.

**Implementation**: Each clip half is treated as a single part regardless of connectivity.

**Why changed**: Deferred for simplicity. Most axis-aligned cuts of connected meshes produce connected halves. Disconnected pieces would just be over-segmented (functionally correct, just sub-optimal part count).

### 7. Fan Cap Triangulation (Simplified from Ear-Clipping)

**Plan**: Bridge holes to outer boundary, ear-clip merged polygon, produce closed mesh output.

**Implementation**: `apply_cuts` adds fan cap triangles (from loop vertex v0, create triangles (v0, v_i, v_{i+1})) after each clip. Winding determined by shoelace signed area. This closes meshes for correct signed-tet volume in subsequent iterations. Fan triangulation is used instead of ear-clipping — it may produce overlapping triangles for non-convex boundary polygons, but the signed volume is still correct (cancellation property).

**Output**: Final parts are returned with cap triangles included. Parts are closed meshes.

### 8. Merge Post-Processing Not Implemented

**Plan**: Greedy agglomerative merge using the existing `pairwise_hausdorff` kernel to combine over-segmented parts.

**Implementation**: Not implemented. Raw beam search output is returned.

**Why changed**: Deferred. The beam search already optimizes for fewest parts at termination (picks beam item with minimum num_parts among those satisfying threshold). Merge would further reduce part count but isn't critical for initial functionality.

### 9. Single CPython Extension (no cmake, no ctypes)

**Plan** (original): CMake builds everything, Python loads via ctypes.CDLL.

**Implementation**: All GPU functionality (beam search + Hausdorff + merge cost) is built as a single `_gpu` CPython extension module by setuptools. `setup.py` compiles `beam_kernels.cu` → fatbin → C header, then builds `beam_module.c` + `beam.c` with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake, no ctypes, no separate shared library.

**Why changed**: ctypes has fragile import path resolution (fails when cwd shadows package name), no type safety, no proper Python object lifecycle. Having two separate extensions (`_native` via cmake + ctypes, `_beam` via setuptools) was unnecessary complexity — all kernels share the same CUDA context and can live in one fatbin/module. The torchoptix pattern (native CPython extension with embedded fatbin) is the standard approach for shipping CUDA-accelerated Python modules.

## Current Status

### Working
- Full beam search pipeline: init → normalize → evaluate → select → apply → recover → download
- **Rv concavity metric**: proper convex hull volume (incremental hull with parallel visibility) + mesh volume (signed tet + divergence theorem cap correction) → Rv formula
- **Fan cap triangulation**: `apply_cuts` closes meshes after each clip for correct volume in subsequent iterations
- Multi-iteration decomposition with double-buffered pools
- Single CPython extension (abi3 cp310+) with all GPU functionality
- Hausdorff distance, pairwise merge cost, beam search in one module
- Cube correctly identified as convex (Rv ≈ 0, 1 part)
- L-shape decomposed at threshold 0.05 (18 parts), 0.15 (2 parts)
- Pure Python CoACD (existing, unmodified)
- All 66 unit tests pass (including 10 GPU beam search tests), GPU smoke tests pass

### Not Yet Implemented
- Hausdorff validation in beam loop (kernels exist, not wired in)
- Connected components after clipping
- Merge post-processing (would reduce over-segmentation at low thresholds)
- Vertex compaction (parts carry superset of vertices, only referenced ones needed)
- Benchmark on Octocat / larger meshes
