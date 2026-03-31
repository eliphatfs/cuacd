# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains:

1. **CoACD** (`CoACD/`) — Collision-Aware Approximate Convex Decomposition (reference C++ implementation, SIGGRAPH 2022).
2. **coacd_gpu** — GPU-accelerated convex decomposition. Single CPython extension (abi3, cp310+) via CUDA driver API. Includes D&C hull volume/mesh, plane cut with cap triangulation, batch mesh volume.

## Repository Layout

```
setup.py              # Build config: setuptools builds _gpu extension
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device code (compiled to single fatbin)
  kernels.cu          #   Root compilation unit — includes all .cu modules (each is self-contained)
  common.cuh          #   Constants, data structures, pool allocator, atomics, global_alloc helpers
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   Edge intersection, point-triangle distance, concavity metrics
  hull_warp_common.cuh#   WarpPool allocator + warp reductions (used by D&C)
  hull_dandc.cuh      #   Preparata-Hong D&C hull volume (Bullet port, warp sort + lane-0 D&C)
  warp_sort.cuh       #   Generic warp-cooperative quicksort template (warp_sort_t<T,Cmp>) + BtPoint32 legacy API
  hull_batch.cu       #   batch_hull_dandc + batch_hull_dandc_mesh + batch_mesh_volume kernels
  test_warp_sort.cu   #   Test kernel for warp_sort_bp32
  plane_cut.cu        #   GPU plane cut: plane_cut_block (__device__) + plane_cut_kernel thin wrapper
  mesh_transform.cu   #   Normalize/recover coordinate kernels
csrc/                 # C host code
  beam.h/.c           #   Host implementation (CUDA driver API) — beam search
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (hull/mesh volume API)
tests/                # All tests
  test_hull.py        #   Hull volume + mesh volume tests (CPU ref, GPU D&C vs scipy, noisy icosphere)
  test_hull_mesh.py   #   D&C hull mesh extraction tests (batch_hull_dandc_mesh)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling (gaussian points)
  test_warp_sort.py   #   Tests for warp_sort_bp32 (bitonic + quicksort paths)
  test_plane_cut.py   #   Plane cut tests: simple loop, ring, multi-hole, edge cases (14 tests)
CoACD/                # Reference C++ CoACD (submodule/external)
```

## Git Usage

Only stage project source files — never use `git add -A` or `git add .`. The repo contains directories that must not be committed:
- `CoACD/` — embedded git repository (reference C++ implementation, not a submodule)
- `compare_output/`, `octocat_output/`, `tmpcompare/` — temporary output directories
- `*.ncu-rep` — NSight Compute profiling artifacts
- Build artifacts (`*.fatbin`, `*.o`, `*.so`, `build/`, `*.egg-info/`)

Before staging, always run `git status` and `git diff --stat` to verify only the expected tracked files appear as modified. Untracked files in the list above should remain unstaged. Then stage the exact files listed in `git status` as modified/deleted, e.g.:
```bash
git add CLAUDE.md coacd_gpu/__init__.py csrc/beam.c csrc/beam.h csrc/beam_module.c cuda/hull_batch.cu cuda/kernels.cu cuda/plane_cut.cu
```

## Build Commands

```bash
# Install everything (requires: CUDA toolkit with nvcc, C compiler)
pip install -e .

# Override GPU architectures
COACD_GPU_ARCHS="80;86" pip install -e .

# Run all tests
python -m pytest tests/ -v

# Build with verbose host-side debug output
COACD_DEBUG=1 pip install -e .

# D&C hull benchmark (standalone, for profiling)
python tests/bench_dandc.py --n_pts 200 --n_hulls 8

# Profile D&C hull with NCU
ncu --set full -o dandc_profile python tests/bench_dandc.py --n_pts 200 --n_hulls 8
```

Dependencies: `numpy`, `pytest` (test only), `trimesh` (comparison only), `manifold3d` (optional, ring/multi-hole plane cut tests).

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

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/kernels.cu` (which `#include`s all self-contained `.cu` modules) -> fatbin -> xxd-style C header, then builds `csrc/beam_module.c` + `csrc/beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). No cmake involved. Fatbin is compiled with `--generate-line-info` for NCU source-level profiling.

Each `.cu` kernel module is self-contained: it carries its own `#include` directives for all `.cuh` dependencies, and all `__global__` kernels are declared `extern "C"` directly on the function definition (no file-level `extern "C"` block). The `.cuh` headers also include their own dependencies (e.g. `reduce.cuh` includes `common.cuh`). This means any `.cu` file can be compiled standalone without relying on include order from `kernels.cu`.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **Single extension** — all GPU functionality (beam search + Hausdorff) in one `_gpu` module. No cmake, no ctypes.
- **Native CPython extension, not ctypes** — ctypes has fragile import path resolution, no type safety, no proper Python object lifecycle. The torchoptix pattern (native CPython extension with embedded fatbin) is the standard approach.
- **No artificial limits** — Do not impose arbitrary caps (e.g., max hull verts/tris per part) that are not inherent to the algorithm. Convex hull output is bounded by Euler's formula (n verts → max 2n-4 triangles); use natural bounds from input size, not hardcoded constants.
- **Evidence-based debugging** — Do not guess errors from partial or truncated output or code. Design and run experiments to confirm root causes with proof before making any fixes. When diagnosing a bug, first write a minimal reproducer or add instrumentation to observe the actual failure.
- **No divergence without consent** — Do not change the algorithm design or add workarounds without explicit approval. If a fix requires a design change, describe the proposed change and wait for approval.
- **Hull algorithm** — D&C (Preparata-Hong) is the primary hull algorithm, exposed via `batch_hull_volume` and `batch_hull_dandc_mesh` (volume + mesh extraction). `hull_dandc_warp_mesh` returns both volume and the extracted hull mesh (vertices + triangles). Volume and mesh extraction use BFS over the half-edge graph (Bullet getVertexCopy pattern) — no DFS stacks, no stack overflow. BFS queue allocated from pool (n pointers), rewound between mesh extraction and volume. Sort scratch and D&C stack are also rewound. `dandc_scratch_bytes` reports `presort + max(sort_scratch, persistent + max(dc_stack, bfs_queue))`. Edge pool sized at `6n` half-edges (exact: max live = `6n-12` by Euler's formula for triangulated convex hulls, confirmed empirically).

### Python API

```python
# Hull/mesh volume utilities
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
```

## Algorithm Notes

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

### Scratch Rewind and BFS (hull_dandc.cuh)

Three rewind points in the D&C hull minimize peak scratch:
1. **Sort scratch**: rewound after `warp_sort_bp32` completes.
2. **D&C stack**: rewound in `bt_compute_postsort` after `bt_computeInternal`.
3. **BFS queue**: `bt_extractMesh`'s BFS queue rewound in `hull_dandc_warp_mesh` before `bt_computeVolume`.

Volume and mesh extraction use BFS (Bullet `getVertexCopy` pattern) instead of DFS — no stack overflow possible. BFS queue size = n vertex pointers, bounded by hull vertex count. `dandc_scratch_bytes` = `presort + max(sort_scratch, persistent + max(dc_stack, bfs_queue))`.

Edge pool: `6n` half-edges. Empirically confirmed exact: max live edge pairs = `3n - 6` (Euler, triangulated hull), peak never exceeds final count during merge. `COACD_TRACK_EDGES=1 pip install -e .` enables `TRACK_MAX_EDGE_PAIRS` compile macro for printf tracking.

## Implementation Lessons

### DevicePool offset/capacity are 64-bit

`DevicePool.offset` and `DevicePool.capacity` are `unsigned long long`. Previously they were `unsigned int` (32-bit), causing silent wraparound when >4GB scratch was consumed by concurrent expansion blocks (480 blocks × ~15MB each = ~7GB). The wraparound made the OOM check pass on wrapped offsets, giving blocks overlapping memory regions → corrupted D&C edge pool free lists → illegal memory access. Fixed by widening to 64-bit; the 4GB scratch cap was also removed.

### D&C: Faithful Port of Bullet's btConvexHullComputer

The D&C algorithm is a faithful port of Bullet's `btConvexHullComputer` (Ole Kniemeyer, MAXON, zlib license). Three phases: (1) warp-parallel pre-sort (AABB via warp min/max reduction, Point32 conversion strided across lanes), (2) warp-cooperative sort via `warp_sort_bp32` (all 32 lanes), (3) D&C merge + volume extraction on lane 0. Key elements: int32 coordinates with exact Int128/Rational64/Rational128 predicates, iterative `computeInternal` D&C with explicit `BtDCStackItem` stack (`BT_DC_MAX_STACK=4096`), `mergeProjection` for 2D bridge finding, `findMaxAngle` with exact cotangent comparison, `findEdgeForCoplanarFaces` for coplanar handling, and the full `merge` function with interior edge deletion via `removeEdgePair`. Memory is allocated from WarpPool bump allocator with warp-parallel free-list init in `btpool_init` (all 32 lanes build the free list, lane 0 sets the head) and free-list recycling for edges. Volume and mesh extraction use BFS over the half-edge graph (Bullet `getVertexCopy` pattern) — no DFS, no stack overflow. BFS queue allocated from pool (n pointers). Volume sums signed tetrahedra in integer coordinates, converts back via scaling factor. Edge pool sized at `6n` half-edges (exact tight bound: max live = `6n-12` by Euler). Scratch per hull computed by `dandc_scratch_bytes(n)` (queried from device via `query_dandc_scratch`). Error codes: 1=pool OOM, 2=sort stack overflow (`WS_MAX_STACK`), 4=D&C stack overflow (`BT_DC_MAX_STACK`), 5=pool exhausted (`BT_ERR_POOL_EXHAUST`). Requires 8KB thread stack (via `cuCtxSetLimit`), block size 32 (1 warp per block, DANDC_BLOCK_SIZE).

### Warp Sort (warp_sort.cuh)

Generic warp-cooperative quicksort template. All 32 lanes participate. Uses a global-memory workspace of the same size as the input. Segments ≤32 elements use a bitonic sorting network; larger segments use quicksort partitioning with a median-of-32-samples pivot. Three-way partition via comparator (returns -1/0/1): items < pivot go left, items > pivot go right, items == pivot are filled in the middle gap by warp-parallel fill. This avoids worst-case O(n²) on all-equal or many-duplicate inputs. Uses `__ballot_sync`/`__popc` for warp-wide prefix sums. Explicit stack (max depth 2048) in global memory, `sp` in lane-0 register broadcast via `__shfl_sync`.

**Template API**: `warp_sort_t<T, Cmp>(data, scratch, n, lane)` where `T` is any POD type with `sizeof(T) % 4 == 0`, and `Cmp` is a struct with `static __device__ int cmp(T, T)` (returns -1/0/1) and `static __device__ T sentinel()`. Generic shuffles via `ws_shfl_xor_t<T>` / `ws_shfl_t<T>` use a union-based approach to shuffle each 4-byte field. Compiled as C++ throughout — no `extern "C++"` wrapper needed.

**Comparators**: `BtPoint32Cmp` sorts by (y,x,z,index) — used by D&C hull. `Edge2iCmp` (in plane_cut.cu) sorts `Edge2i` structs (8 bytes: two ints `a,b`) by (a,b) — used for crossing edge dedup and directed edge boundary detection. **Legacy API**: `warp_sort_bp32`, `ws_cmp`, `ws_min`, `ws_max`, `ws_bitonic32` are thin wrappers calling the template with `BtPoint32Cmp`.

### btpool: Eager Warp-Parallel Init, No Lazy Allocation

`btpool_init(p, wp, blockSize, objSize, lane)` eagerly allocates the pool block and uses all 32 warp lanes to build the free list in parallel (each lane handles every 32nd slot). `btpool_new` only pops from the free list; if exhausted, it sets `BT_ERR_POOL_EXHAUST` (error code 5) and returns NULL — no lazy block allocation. `objSize` is padded to multiple of 4; zero-init in `btpool_new` uses `int*` with `objSize/4` iterations. Only the edge pool (BtEdge=40B) is used; vertex and face pools were removed as unused.

### pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

### nvcc Crashes with --generate-line-info and Deep Recursion

nvcc crashed when compiling with `--generate-line-info` while `bt_computeInternal` was recursive. Converting to an iterative explicit stack resolved the crash. Line info is now always enabled for NCU profiling.

### Scratch Query Ordering Does Not Help batch_hull_volume

Moving `query_dandc_scratch` before the large `cuMemcpyHtoDAsync` calls (to avoid syncing after copies) was tested and made performance slightly worse. The `cuMemAlloc` calls themselves may already serialize, so reordering provides no benefit. Keep the query after copies for now.

## Current Status

### Working
- D&C hull volume (`batch_hull_volume`) and mesh extraction (`batch_hull_dandc_mesh`) — tested (cube 8v/12t, tetra 4v/4t, gaussian)
- Batch mesh volume (`batch_mesh_volume`) — divergence theorem, watertight meshes
- Scratch rewinds (sort/D&C/BFS) + BFS graph traversal (no DFS stack overflow)
- `CHECK_CU` macro includes `beam.c:LINE` in error messages for faster diagnosis

### Plane Cut Kernel (plane_cut.cu) — Working
- `plane_cut_block` is the `__device__` function doing the real work; `plane_cut_kernel` is a thin `__global__` wrapper that calls it (enables future multi-block use)
- GPU kernel `plane_cut_kernel`: 1 block × 64 threads (2 warps), handles one plane cut
- Parallel phases: vertex classification, crossing edge collection, triangle splitting (with sorted edge dedup), directed edge collection, boundary detection (sort + binary search)
- Warp 0 phases: sort crossing edges and directed edges via `warp_sort_t<Edge2i, Edge2iCmp>`
- Thread 0 sequential phases: boundary loop reconstruction, multi-hole bridging (sorted by rightmost vertex), ear clipping with bridge-duplicate-aware point-in-triangle
- Cap winding determined from 2D projection convention (`e_pu × e_pv` cross product direction)
- Tested: simple loop (cube cuts on all axes, off-center, sphere), non-convex cap (L-shape), ring topology (hollow tube), multi-hole (box with 2 through-holes), edge cases (plane through vertex/edge, diagonal plane) — 14 tests pass
- Host launcher in `beam_test_plane_cut` (beam.c) handles device memory allocation, kernel launch, result download

### Not Yet Implemented
- `__cuda_array_interface__` support for GPU tensor input
