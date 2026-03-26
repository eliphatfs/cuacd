# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition (reference C++ implementation, SIGGRAPH 2022).
2. **coacd_gpu** — GPU-accelerated convex decomposition. Three components:
   - **Hausdorff/merge kernels** (`cuda/kernels.cu`, `csrc/coacd_gpu.c`) — GPU Hausdorff distance and pairwise merge cost via CUDA driver API.
   - **Beam search decomposition** (`cuda/beam_kernels.cu`, `csrc/beam.c`, `csrc/beam_module.c`) — GPU-native beam search replacing MCTS. CPython extension (abi3, cp310+).
   - **Pure Python CoACD** (`coacd_gpu/coacd/`) — Pure Python reimplementation using scipy/triangle. No C++ build required.

## Repository Layout

```
setup.py              # Build config: CMake for _native, setuptools for _beam
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device kernels + CMake for fatbin
  CMakeLists.txt      #   Builds libcoacd_gpu.so (Hausdorff/merge only)
  kernels.cu          #   point_mesh_distance, reduce_max, pairwise_hausdorff
  beam_kernels.cu     #   Beam search: clip, bbox concavity, candidate eval, apply_cuts
csrc/                 # C host code
  coacd_gpu.h/.c      #   Hausdorff/merge host implementation (CUDA driver API)
  beam.h/.c           #   Beam search host implementation (CUDA driver API)
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   ctypes wrapper for _native + re-exports BeamContext
  beam.py             #   Python API for beam search (imports _beam extension)
  coacd/              #   Pure Python CoACD implementation
tests/                # All tests
  test_extension.py   #   GPU smoke tests (Hausdorff, pairwise)
  test_clip.py ...    #   Pure Python CoACD unit tests
CoACD/                # Reference C++ CoACD (submodule/external)
```

## Build Commands

```bash
# Install everything (requires: cmake >= 3.24, CUDA toolkit, C compiler)
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

Two extensions are built by `setup.py`:

1. **`coacd_gpu._native`** — CMake-built shared library (`libcoacd_gpu.so`). CMakeLists.txt in `cuda/` compiles `kernels.cu` → fatbin → bin2c → C array, links with `csrc/coacd_gpu.c`. Python loads via ctypes.

2. **`coacd_gpu._beam`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/beam_kernels.cu` → fatbin → xxd-style C header, then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake involved for this extension.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x–12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.

### Hausdorff/Merge Kernels (`cuda/kernels.cu`)

- **`point_mesh_distance`** — brute-force point-to-triangle (Eberly's method), one thread per point.
- **`reduce_max`** — shared-memory parallel max reduction.
- **`pairwise_hausdorff`** — batch merge cost matrix with `atomicMax` float CAS.

### Beam Search Kernels (`cuda/beam_kernels.cu`)

GPU beam search replacing MCTS. Search space: 3×N axis-aligned cuts per step (default N=10 → 30 planes).

Kernels:
- **`evaluate_candidates`** — Grid `(beams × planes)`. Per block: classify vertices against plane (inline, not function call — see implementation notes), split straddling triangles, compute `cbrt(bbox_volume)` for each half as cost proxy.
- **`select_top_k`** — Single block, thread 0 only. Insertion-sort to pick best `beam_width` candidates.
- **`apply_cuts`** — Grid `(winners)`. Copy unchanged parts, re-clip worst part, write to double-buffered pool.
- **`compute_part_costs`** — Grid `(beams)`. Compute `cbrt(bbox_volume)` for parts needing it, find worst per beam item.
- **`normalize_mesh`** / **`recover_coordinates`** — Normalize to [-1,1]³ and recover.
- **`sample_surface`**, **`beam_point_mesh_distance`**, **`beam_reduce_max`** — Hausdorff support (present but not yet wired into beam loop).

### Beam Search Host Orchestration (`csrc/beam.c`)

```
Input mesh → Upload to pool_a → Normalize to [-1,1]³
  → Generate 3×N axis-aligned cutting planes
  → Compute initial bbox concavity (compute_part_costs)
  → If already below threshold → return single part
  → Beam loop:
      1. evaluate_candidates: clip worst part of each beam item by each plane
      2. select_top_k: pick best beam_width candidates
      3. apply_cuts: materialize winning cuts into pool_nxt
      4. Swap pool_cur ↔ pool_nxt, swap parts/beam buffers
      5. compute_part_costs: recompute concavity, find worst part per beam item
      6. If best beam item's worst_cost ≤ threshold → done
  → Recover coordinates → Download parts
```

**Memory management:**
- **Double-buffered mesh pools**: pool_a and pool_b alternate each iteration. Read from current, write to next, swap.
- **Bump-allocated scratch pool**: single large device buffer (~50% of free GPU memory). Reset offset to 0 between kernel launches. Thread-0-only allocation with shared memory broadcast.
- **Per-iteration alloc/free**: parts and beam metadata arrays are freshly allocated each iteration (old ones freed after swap).

### Concavity Metric: Bbox Cube Root

The beam search uses `cbrt(bbox_volume)` of each part's triangle vertices as the concavity metric. This is a proxy for "part size" — smaller parts have lower cost.

- **Threshold meaning**: decompose until all parts fit within a box of characteristic length ≤ threshold. For meshes normalized to [-1,1]³, the initial whole-mesh cost is ~2.0 (`cbrt(8)`). A threshold of 0.5 produces moderate decomposition; lower values produce more parts.
- **Scoring a cut**: `max(cbrt(bbox_vol_positive_half), cbrt(bbox_vol_negative_half))`. The beam search minimizes the worst-case part size across all beam items.

This replaces the planned Rv (volume-ratio) metric which required convex hull computation. See "Design Deviations" below.

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
# Hausdorff (legacy ctypes wrapper)
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    dists = ctx.point_mesh_distances(points, vertices, triangles)
    h = ctx.hausdorff(sa, va, ta, sb, vb, tb)

# Beam search decomposition (native CPython extension)
from coacd_gpu.beam import BeamContext, run_beam_coacd
with BeamContext(device=0) as ctx:
    parts = ctx.run(vertices, triangles, threshold=0.5)
# or:
parts = run_beam_coacd(vertices, triangles, threshold=0.5)
```

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

### 1. Concavity Metric: Bbox Cube Root instead of Rv (Volume-Ratio)

**Plan**: Rv = `(3 * |V_mesh - V_hull| / (4π))^(1/3) * k`, requiring per-part convex hull volume computed on GPU via incremental QuickHull within each thread block.

**Implementation**: `cbrt(bbox_volume)` computed from triangle vertex bounding boxes. No convex hull, no mesh volume.

**Why changed**: The GPU incremental convex hull was the single largest source of bugs:
- `__syncthreads()` inside conditional point-insertion loops caused deadlocks
- Massive scratch allocations (HullFace arrays, horizon edges, visibility flags) exhausted the pool
- Even when it didn't crash, it produced inaccurate volumes for simple convex shapes (cubes got Rv=0.37 instead of ~0)
- The signed-tetrahedra mesh volume is unreliable for open meshes produced by clipping (no cap → volume cancels to ~0)

The bbox metric avoids all of these issues. It's geometrically meaningful for axis-aligned decomposition: a convex part that fills its bbox well has been sufficiently decomposed. The tradeoff is that non-axis-aligned concavity isn't detected — this is acceptable because the cutting planes are also axis-aligned.

### 2. No Cap Volume via Divergence Theorem

**Plan**: Close open meshes after clipping using `V_cap = (-d/3) × A_net` (divergence theorem with boundary loop signed areas via shoelace formula). This would make signed-tet volume correct for open meshes.

**Implementation**: Caps are not computed. Mesh volume is not used for scoring.

**Why changed**: The divergence theorem approach requires tracing boundary loops from intersection edges — a sequential graph traversal that's hard to parallelize within a block. Since the bbox metric doesn't need volume at all, this became unnecessary.

### 3. No GPU Convex Hull (QuickHull Port Removed)

**Plan**: Port `leomccormack/convhull_3d` (QuickHull) to GPU device functions. Parallel visibility tests, sequential vertex insertion, on-the-fly volume accumulation. Approximate hull limited to 128-256 vertices.

**Implementation**: Convex hull computation removed entirely from GPU path. ~1200 lines of hull code (find_extremes, incremental insertion, horizon edge tracing, HullFace struct) deleted.

**Why changed**: The GPU hull was fundamentally difficult to make correct:
- Incremental hull needs sequential face updates after each vertex insertion
- Parallel visibility tests help but the insertion itself serializes
- Floating-point edge cases (coplanar points, degenerate tetrahedra) caused crashes
- The `__shared__` variable scoping within the per-point loop caused `__syncthreads` issues

For future work, hull computation should be done either on CPU (download vertices, scipy qhull) or via a purpose-built parallel hull kernel separate from the evaluation kernel.

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

### 7. Ear-Clipping Cap Triangulation Deferred

**Plan**: Bridge holes to outer boundary, ear-clip merged polygon, produce closed mesh output.

**Implementation**: Clipped parts are returned as open meshes (no cap faces on cutting planes).

**Why changed**: Not needed for the bbox concavity metric (which doesn't use volume). Will be needed when Hausdorff validation is wired in (Hausdorff needs closed meshes for accurate surface sampling).

### 8. Merge Post-Processing Not Implemented

**Plan**: Greedy agglomerative merge using the existing `pairwise_hausdorff` kernel to combine over-segmented parts.

**Implementation**: Not implemented. Raw beam search output is returned.

**Why changed**: Deferred. The beam search already optimizes for fewest parts at termination (picks beam item with minimum num_parts among those satisfying threshold). Merge would further reduce part count but isn't critical for initial functionality.

### 9. Build System: CPython Extension instead of CMake + ctypes

**Plan** (original): CMake builds everything, Python loads via ctypes.CDLL.

**Implementation**: CMake builds only `_native` (Hausdorff/merge). The beam search `_beam` is built directly by setuptools as a CPython extension module (Py_LIMITED_API, slot-based module definition).

**Why changed**: ctypes has fragile import path resolution (fails when cwd shadows package name), no type safety, no proper Python object lifecycle. The torchoptix pattern (native CPython extension with embedded fatbin) is the standard approach for shipping CUDA-accelerated Python modules.

## Current Status

### Working
- Full beam search pipeline: init → normalize → evaluate → select → apply → recover → download
- Multi-iteration decomposition with double-buffered pools
- CPython extension (abi3 cp310+) builds and loads
- Cube correctly identified as convex at appropriate threshold
- L-shape decomposed into 3 parts
- Hausdorff/merge kernels (existing, unmodified)
- Pure Python CoACD (existing, unmodified)
- All 56 unit tests pass, GPU smoke tests pass

### Not Yet Implemented
- Hausdorff validation in beam loop (kernels exist, not wired in)
- Connected components after clipping
- Cap triangulation for closed mesh output
- Merge post-processing
- Vertex compaction (parts carry superset of vertices, only referenced ones needed)
- Benchmark on Octocat / larger meshes
