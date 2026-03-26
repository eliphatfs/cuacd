# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition (reference C++ implementation, SIGGRAPH 2022).
2. **coacd_gpu** — GPU-accelerated convex decomposition. Three components:
   - **Hausdorff/merge kernels** (`cuda/kernels.cu`, `csrc/coacd_gpu.c`) — GPU Hausdorff distance and pairwise merge cost via CUDA driver API.
   - **Beam search decomposition** (`cuda/beam_kernels.cu`, `csrc/beam.c`, `csrc/beam_module.c`) — Fully GPU-native beam search replacing MCTS. CPython extension (abi3, cp310+).
   - **Pure Python CoACD** (`coacd_gpu/coacd/`) — Pure Python reimplementation using scipy/triangle. No C++ build required.

## Repository Layout

```
setup.py              # Build config: CMake for _native, setuptools for _beam
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device kernels + CMake for fatbin
  CMakeLists.txt      #   Builds libcoacd_gpu.so (Hausdorff/merge only)
  kernels.cu          #   point_mesh_distance, reduce_max, pairwise_hausdorff
  beam_kernels.cu     #   Beam search: clip, hull, Rv, candidate eval, apply_cuts
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

2. **`coacd_gpu._beam`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/beam_kernels.cu` → fatbin → header (xxd-style), then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel).

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

GPU beam search replacing MCTS. Search space: 3×N axis-aligned cuts per step.

Kernels:
- **`evaluate_candidates`** — Per (beam_item, plane) block: classify vertices, split triangles, compute mesh volume (signed tets), convex hull volume (incremental), Rv cost.
- **`select_top_k`** — Pick best beam_width candidates from cost buffer.
- **`apply_cuts`** — Materialize winning cuts: copy parts to new pool, re-clip worst part.
- **`compute_part_costs`** — Recompute Rv for all parts, find worst per beam item.
- **`normalize_mesh`** / **`recover_coordinates`** — Normalize to [-1,1]³ and recover.
- **`sample_surface`**, **`beam_point_mesh_distance`**, **`beam_reduce_max`** — For Hausdorff validation.

### Beam Search Host (`csrc/beam.c`)

Orchestrates the beam loop via CUDA driver API:
1. Upload mesh, normalize
2. Generate axis-aligned cutting planes
3. Loop: evaluate → select top-K → apply cuts → recompute Rv → check termination
4. Recover coordinates, download parts

Uses double-buffered mesh pools and a bump-allocated scratch pool.

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
    parts = ctx.run(vertices, triangles, threshold=0.05)
# or:
parts = run_beam_coacd(vertices, triangles, threshold=0.05)
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

## Beam Search Progress (WIP)

### Working
- Full pipeline: init → normalize → evaluate candidates → select top-K → apply cuts → recover → download
- CPython extension builds and loads correctly (abi3, cp310+)
- End-to-end produces decomposed parts with correct coordinates

### Known Issues
- Convex hull volume computation on GPU has accuracy issues (over-estimates Rv for convex meshes like cubes), causing unnecessary decomposition
- Need to debug/replace incremental hull construction in `convex_hull_volume()` device function

### Not Yet Implemented
- Hausdorff validation pass (kernel exists but not wired into beam loop)
- Connected components (treating each clip half as single component for now)
- Ear-clipping cap triangulation for final output mesh closure
- Merge post-processing
