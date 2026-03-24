# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two projects:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition. Decomposes 3D meshes into approximate convex parts using MCTS-guided plane cutting. Published at SIGGRAPH 2022.
2. **coacd_gpu** (`coacd_gpu/`) — Standalone GPU acceleration library for CoACD's computational bottlenecks (Hausdorff distance, merge cost matrix). Uses CUDA driver API; no PyTorch or CUDA runtime dependency.

## Build Commands

### CoACD (C++ / Python)

```bash
# C++ build
cd CoACD && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make main -j$(nproc)

# Run CLI
./build/main -i examples/SnowFlake.obj -o output.obj -t 0.05

# Python package
cd CoACD && pip install -e .

# Run tests (requires: trimesh, numpy, coacd)
cd CoACD && python run_tests.py

# Run single test
python -m unittest run_tests.TestExamples.test_snowflake
```

### coacd_gpu (CUDA / Python)

```bash
# Build and install (requires: cmake >= 3.24, CUDA toolkit, C compiler)
cd coacd_gpu && pip install -e .

# Override target GPU architectures (default: 80;86;89;90)
COACD_GPU_ARCHS="80;86" pip install -e .

# Run smoke tests
cd coacd_gpu && python test_extension.py
```

## CoACD Architecture

### Algorithm Pipeline

```
Input OBJ → Normalize → Manifold Check/Repair (OpenVDB) → [PCA alignment]
    → Iterative MCTS Decomposition Loop:
        For each part with concavity > threshold:
            1. Compute convex hull
            2. Calculate cost = max(Rv, Hausdorff)
            3. MCTS tree search for best cutting plane
            4. Ternary refinement of plane position
            5. Clip mesh along plane → two halves
            6. Recurse on halves
    → Merge post-processing (greedy agglomerative)
    → [Decimate] → [Extrude] → Output
```

### Key Source Files (`CoACD/src/`)

| File | Role |
|------|------|
| `process.cpp` | Main decomposition loop, merge, decimate, extrude. Contains OpenMP parallel regions |
| `mcts.cpp` | MCTS tree search (UCB selection, random rollout, backprop) + ternary plane refinement |
| `cost.cpp` | Concavity metrics: Rv (volume-based) and Hausdorff distance computation |
| `hausdorff.h` | KD-tree nearest-neighbor queries via nanoflann for Hausdorff distance |
| `clip.cpp` | Plane-mesh clipping with CDT (Constrained Delaunay Triangulation) for cap faces |
| `model_obj.cpp` | `Model` class: mesh I/O, convex hull (QuickHull primary, btConvexHull fallback), PCA |
| `bvh.cpp` | BVH for triangle self-intersection detection (manifold checking) |
| `config.h` | `Params` struct holding all algorithm parameters |
| `preprocess.cpp` | OpenVDB-based manifold repair (compiled only with `WITH_3RD_PARTY_LIBS`) |

### Public API

- `public/coacd.h` / `coacd.cpp` — C/C++ API wrapping the decomposition pipeline
- `python/package/` — Python bindings via shared library `_coacd`

### Core Data Structures

- **`Model`** (`model_obj.h`): Primary mesh representation (vertices as `vec3d`, triangles as `vec3i`, bbox, rotation matrix). Methods: `ComputeCH()`, `ComputeAPX()`, `IsManifold()`
- **`Node`/`State`/`Part`** (`mcts.h`): MCTS tree node, decomposition state (list of parts + costs), and individual mesh partition with candidate cutting planes
- **`Plane`** (`shape.h`): Cutting plane `ax+by+cz+d=0` with intersection/side methods

### Parallelism

OpenMP is used in two places:
1. **Per-part MCTS search** (`process.cpp`): Each mesh part's decomposition runs in parallel
2. **Merge cost matrix** (`process.cpp`): Pairwise convex hull merge costs computed in parallel

### Computational Bottlenecks

1. **Hausdorff distance** — KD-tree NN queries over sampled surface points, called per candidate plane per MCTS iteration
2. **Convex hull computation** — QuickHull per part per MCTS iteration
3. **Merge phase** — O(p²) pairwise cost computations where p = number of parts
4. **Mesh clipping** — CDT triangulation of clip boundaries

### CMake Options

- `WITH_3RD_PARTY_LIBS` (default ON): Enables OpenVDB manifold preprocessing + spdlog. When OFF, preprocessing code is excluded and logging is disabled.
- `CMAKE_BUILD_TYPE`: Use `Release` for optimized builds, `Debug` enables AddressSanitizer on GCC.

## coacd_gpu Architecture

### Design Principles

- **CUDA driver API only** — links `libcuda.so` (GPU driver), NOT `libcudart.so` (runtime). One binary works across CUDA 11.x–12.x+.
- **Fatbin embedding** — kernels compiled to fatbin (cubins for sm_80/86/89/90 + PTX for forward compat), converted to a C array via `bin2c`, linked into the shared library. No external files to ship.
- **Python abi3 wheel** — uses `Py_LIMITED_API` targeting Python 3.9+. One wheel per platform.
- **No PyTorch dependency** — pure numpy arrays in/out via ctypes. Reuses existing CUDA context if available (e.g. from PyTorch) via `coacd_gpu_init(ctx, -1)`.

### File Layout

| File | Role |
|------|------|
| `kernels.cu` | Pure device code (`extern "C"`): `point_mesh_distance`, `reduce_max`, `pairwise_hausdorff` |
| `coacd_gpu.h` | Public C API header (ctypes-friendly, `COACD_GPU_API` visibility) |
| `coacd_gpu.c` | Host implementation: `cuModuleLoadFatBinary`, `cuLaunchKernel`, memory management |
| `CMakeLists.txt` | `nvcc --fatbin` → `bin2c` → `libcoacd_gpu.so`, links only `CUDA::cuda_driver` |
| `python/__init__.py` | ctypes wrapper exposing `Context` class with numpy API |
| `setup.py` | CMake-based build + abi3 wheel tagging |
| `pyproject.toml` | PEP 621 metadata |

### GPU Kernels

- **`point_mesh_distance`** — brute-force point-to-triangle distance (Eberly's method). One thread per query point, scans all triangles. Faster than BVH for CoACD's typical part sizes (<50k triangles).
- **`reduce_max`** — shared-memory parallel max reduction for computing Hausdorff from per-point distances.
- **`pairwise_hausdorff`** — batch merge cost matrix. Grid maps (pair_idx, sample_block), uses `atomicMax` float CAS to reduce per-point distances into per-pair Hausdorff values.

### Python Usage

```python
import coacd_gpu

with coacd_gpu.Context(device=0) as ctx:
    # Per-point distances
    dists = ctx.point_mesh_distances(points, vertices, triangles)

    # Hausdorff between two meshes
    h = ctx.hausdorff(samples_a, verts_a, tris_a, samples_b, verts_b, tris_b)

    # Pairwise merge cost matrix
    cost = ctx.pairwise_hausdorff(all_samples, sample_offsets,
                                   all_vertices, all_triangles,
                                   tri_offsets, vert_offsets)
```

### Planned: `__cuda_array_interface__` Support

Support the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol for zero-copy GPU memory communication. This eliminates CPU↔GPU round-trips when coacd_gpu is used alongside PyTorch, CuPy, or other GPU libraries.

**Design:**

- **Input path**: All public API methods (`point_mesh_distances`, `hausdorff`, `pairwise_hausdorff`) should detect `__cuda_array_interface__` on input arguments. When present, extract the device pointer directly (`cai['data'][0]`) and skip `cuMemAlloc` + `cuMemcpyHtoD`. Validate dtype/shape/contiguity from the interface metadata (`typestr`, `shape`, `strides`).
- **Output path**: Provide an option to return GPU-resident results wrapped in a lightweight object exposing `__cuda_array_interface__` (device pointer, shape, typestr), instead of downloading to numpy. This lets downstream GPU code consume results without a device→host copy.
- **Fallback**: Plain numpy arrays continue to work as before (upload to GPU, compute, download). The interface is additive — no breaking changes.
- **Memory ownership**: For inputs, coacd_gpu borrows the pointer (caller owns the memory). For GPU outputs, coacd_gpu allocates device memory and the returned wrapper object frees it on garbage collection (via `cuMemFree` pointers stored in the context).
- **Reference**: See [`gint/host/executor.py`](https://github.com/eliphatfs/gint/blob/main/gint/host/executor.py) `TensorInterface` class for `from_cuda_array_interface` / `__cuda_array_interface__` property patterns.
