# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coacd_gpu** — GPU hull volume, mesh volume, and plane cut utilities. Single CPython extension (abi3, cp310+) via CUDA driver API.

`CoACD/` — Reference C++ CoACD implementation (SIGGRAPH 2022), not part of the GPU extension.

## Repository Layout

```
setup.py              # Build config: setuptools builds _gpu extension
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device code (compiled to single fatbin)
  kernels.cu          #   Root compilation unit — includes all .cu modules (each is self-contained)
  allocator.cuh       #   DevicePool struct + pool_alloc (block) + global_alloc_warp (warp)
  heap_arena.cuh      #   DeviceHeap: 64-arena large-object heap (heap_alloc/free/compact)
  common.cuh          #   Constants, atomics (includes allocator.cuh)
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   Edge intersection, point-triangle distance, concavity metrics
  hull_warp_common.cuh#   WarpPool allocator + warp reductions (used by D&C)
  hull_dandc.cuh      #   Preparata-Hong D&C hull (Bullet port); hull_dandc_warp_mesh returns Mesh via heap
  warp_sort.cuh       #   Generic warp-cooperative quicksort template (warp_sort_t<T,Cmp>) + BtPoint32 legacy API
  plane_cut.cuh       #   plane_cut_block device function + Edge2i/Edge2iCmp structs; returns PartPair via DeviceHeap
  hull_batch.cu       #   batch_hull_dandc + query_dandc_scratch + batch_mesh_volume kernels
  test_warp_sort.cu   #   Test kernel: test_warp_sort_kernel
  test_hull_dandc.cu  #   Test kernel: batch_hull_dandc_mesh (hull volume + mesh extraction)
  test_plane_cut.cu   #   Test kernel: plane_cut_kernel (thin wrapper around plane_cut_block)
csrc/                 # C host code
  structs.h           #   Host-side structs: DevicePool, beam_ctx
  beam.h              #   Public C API (beam_ctx_t, beam_init/destroy, batch ops)
  beam.c              #   Host implementation: beam_init/destroy, beam_batch_hull_volume, beam_batch_mesh_volume
  test_beam.c         #   Test host launchers: beam_test_warp_sort, beam_batch_hull_dandc_mesh, beam_test_plane_cut, beam_set_plane_cut_ctx
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (batch_hull_volume, batch_mesh_volume, batch_hull_dandc_mesh)
tests/                # All tests
  test_hull.py        #   Hull volume + mesh volume tests (CPU ref, GPU D&C vs scipy, noisy icosphere)
  test_hull_mesh.py   #   D&C hull mesh extraction tests (batch_hull_dandc_mesh)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling (gaussian points)
  test_warp_sort.py   #   Tests for warp_sort_bp32 (bitonic + quicksort paths)
  test_plane_cut.py   #   Plane cut tests: simple loop, ring, multi-hole, edge cases (14 tests)
CoACD/                # Reference C++ CoACD (embedded repo, not a submodule)
```

## Git Usage

Only stage project source files — never use `git add -A` or `git add .`. The repo contains directories that must not be committed:
- `CoACD/` — embedded git repository (reference C++ implementation, not a submodule)
- `compare_output/`, `octocat_output/`, `tmpcompare/` — temporary output directories
- `*.ncu-rep` — NSight Compute profiling artifacts
- Build artifacts (`*.fatbin`, `*.o`, `*.so`, `build/`, `*.egg-info/`)

Before staging, always run `git status` and `git diff --stat` to verify only the expected tracked files appear as modified. Untracked files in the list above should remain unstaged. Then stage the exact files listed in `git status` as modified/deleted, e.g.:
```bash
git add CLAUDE.md coacd_gpu/__init__.py csrc/beam.c csrc/beam.h csrc/beam_module.c csrc/structs.h cuda/kernels.cu cuda/allocator.cuh cuda/common.cuh
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

## Workflow Rule

After completing any code change, always build (`pip install -e .`) and run the full test suite (`python -m pytest tests/ -v`), skipping any tests that were already failing before the change.

## Architecture

### Build System

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles `cuda/kernels.cu` (which `#include`s all self-contained `.cu` modules) → fatbin → C header, then builds `csrc/beam_module.c` + `csrc/beam.c` + `csrc/test_beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). Fatbin compiled with `--generate-line-info` for NCU source-level profiling.

**File split:**
- `cuda/hull_batch.cu` — `batch_hull_dandc`, `query_dandc_scratch`, `batch_mesh_volume`
- `cuda/test_hull_dandc.cu` — `batch_hull_dandc_mesh` (hull volume + mesh extraction)
- `cuda/test_warp_sort.cu` — `test_warp_sort_kernel`
- `cuda/test_plane_cut.cu` — `plane_cut_kernel` (thin `__global__` wrapper around `plane_cut_block`)
- `csrc/beam.c` — `beam_init/destroy`, `beam_batch_hull_volume`, `beam_batch_mesh_volume`
- `csrc/test_beam.c` — `beam_test_warp_sort`, `beam_batch_hull_dandc_mesh`, `beam_test_plane_cut`, `beam_set_plane_cut_ctx`

**Struct locations:**
- `cuda/allocator.cuh` — device-side: `DevicePool`; included by `common.cuh`
- `cuda/structs.cuh` — device-side: `Mesh`, `Part`, `PartPair`, `WorkItem`, `AlgoState`; included by `plane_cut.cuh` and future algo code
- `csrc/structs.h` — host-side: `DevicePool`, `beam_ctx`; included by `beam.c` and `test_beam.c`

Each `.cu` kernel module is self-contained: carries its own `#include` directives, all `__global__` kernels declared `extern "C"` directly on the function definition (no file-level block).

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat, embedded as C arrays.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+. One wheel per platform.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **No artificial limits** — Convex hull output is bounded by Euler's formula (n verts → max 2n-4 triangles); use natural bounds, not hardcoded constants.
- **Evidence-based debugging** — Do not guess errors from partial output. Write a minimal reproducer or add instrumentation to observe the actual failure before making fixes.

### Python API

```python
import coacd_gpu
with coacd_gpu.Context(device=0) as ctx:
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
    results = ctx.batch_hull_dandc_mesh(pts_list)  # list of (verts, tris, volume)
```

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

### I. Memory / Infrastructure

| ID | Function | File |
|----|----------|------|
| I1 | `pool_alloc(pool, size) -> void*` | allocator.cuh |
| I2 | `global_alloc_warp(pool, bytes, lane) -> void*` | allocator.cuh |
| I3 | `atomicMinF / atomicMaxF` | common.cuh |
| I4 | `heap_alloc(heap, size, out) -> int` | heap_arena.cuh |
| I5 | `heap_free(heap, ptr) -> int` | heap_arena.cuh |
| I6 | `heap_compact(heap) -> int` | heap_arena.cuh |

### J. Dead Code

| ID | What | Why dead |
|----|------|----------|
| J1 | `compute_concavity_tris` in geometry.cuh | Bbox cube-root proxy, superseded by Rv |
| J2 | `hull_dandc_warp` | Removed; use `hull_dandc_warp_mesh` for all callers |

## plane_cut_block API

`plane_cut_block(mesh, pa, pb, pc_n, pd, out, heap, scratch_heap, kernel_error)` — device function, one block (64 threads).

- **Input**: `const Mesh*` (replaces separate verts/tris/nv/nt params).
- **Output**: writes a `PartPair*` (caller-allocated device memory); `out->pos.mesh` and `out->neg.mesh` point into a **single heap chunk** allocated from `DeviceHeap* heap`.
- **Heap chunk layout** (one `heap_alloc` call on `heap`): `[pos_verts | pos_tris | neg_verts | neg_tris]`, each section 16-byte aligned.
- **Scratch**: all temporary buffers (signs, all_verts, cross_edges, sort_scratch, isect_idx, pos_tris, neg_tris, dir_edges, dir_sort, boundary_flags, and per-loop working buffers) are allocated from `DeviceHeap* scratch_heap` and freed individually as soon as each buffer's last use completes. Nothing from scratch_heap survives the call.
- **Vertex compaction** (phase 13, parallel): phases 1-12 run on thread 0 or warp 0; phase 13 runs across all 64 threads. Steps: init remap[]=-1 (strided), mark used verts via `atomicMax` (strided), count per contiguous chunk, exclusive prefix scan (thread 0, 64 iters), assign new indices per chunk, remap tris in-place (strided), heap alloc (thread 0), scatter verts + copy tris (strided). Each side's `Mesh.verts` contains only the vertices actually referenced — no loose vertices.
- **Counters** (`n_cross`, `n_all_verts`, `n_pos`, `n_neg`): stored in `__shared__ int s_counters[4]`; `atomicAdd` on shared memory. Not on heap.
- **Early exit** (`n_cross == 0`): entire input mesh goes to one side; allocates one heap chunk for verts+tris, other side gets empty `Mesh {NULL,NULL,0,0}`.
- **No-boundary / one-empty-side cases**: handled by natural fallthrough — compaction produces a 0-entry side correctly.
- **Call sites** (`test_plane_cut.cu`, `test_beam.c`): not yet updated; will be overhauled separately.

## hull_dandc_warp_mesh API

`hull_dandc_warp_mesh(pts, n, lane, heap, scratch_heap, err) -> Mesh` — warp device function (all 32 lanes call with identical args).

- **Output**: returns a `Mesh` struct directly. Allocates a single combined `[verts | tris]` chunk from `DeviceHeap* heap`. Returns `{NULL,NULL,0,0}` on error or n<4.
- **Heap chunk layout**: `[verts (nv*3 floats, 16-byte aligned) | tris (nt*3 ints)]`; exact sizes from a count pass.
- **No volume**: `bt_computeVolume` is not called; the function only produces the mesh.
- **Scratch**: `DeviceHeap* scratch_heap` backs (a) the `WarpPool` (allocated as a single heap chunk via `dandc_scratch_bytes(n)`) and (b) `BtEdge` pool slabs (`BTPOOL_BLOCK_SIZE=8192` edges each, 2 initial + dynamic expansion). All scratch is heap-freed before return — scratch_heap is clean after call.
- **WarpPool**: declared `__shared__`; backing allocated from scratch_heap by lane 0. Used for BtPoint32 array, vertex block, sort scratch, D&C stack, BFS queues (all rewound when done).
- **BtPool (edge pool)**: starts with 2 slabs (16384 edges); expands one slab at a time via `heap_alloc(scratch_heap, ...)` when exhausted. Lane 0 only; free-list setup is serial. Up to `BTPOOL_MAX_BLOCKS=32` slabs tracked for cleanup.
- **Two-pass mesh extraction**: count pass (`bt_extractMesh` with NULL buffers, counts nv/nt via fan formula) → `heap_alloc(heap, ...)` for exact output → extract pass (writes verts+tris). BFS queue rewound between passes.
- **dandc_scratch_bytes**: no longer includes the `6*n*sizeof(BtEdge)` edge pool term (pool now comes from scratch_heap separately).
- **Call sites** (`hull_batch.cu`, `test_hull_dandc.cu`): not yet updated; will be overhauled separately.

## Pool Allocator Pattern

All scratch memory in kernels is allocated from a global bump pool. **Critical**: only thread 0 calls `pool_alloc()`, stores pointer in `__shared__` memory, then all threads read after `__syncthreads()`:

```cuda
__shared__ int* s_signs;
if (tid == 0) {
    s_signs = (int*)pool_alloc(&scratch, n * sizeof(int));
}
__syncthreads();
int* signs = s_signs;  // all threads see same pointer
```

If all threads call `pool_alloc`, each gets a different offset → data corruption.

## Heap Arena Allocator (heap_arena.cuh)

`DeviceHeap` is a large-object heap backed by a `DevicePool`. Design:

- **64 arenas** (`HEAP_NUM_ARENAS`), selected by `blockIdx.x % 64`. Each arena has a singly-linked free list and a spin-lock (`int lock`).
- **Block layout**: `[HeapBlockHdr (16 B)][data (data_size B)]`. Free blocks store the next header address in `data[0..7]`.
- **Allocation** (thread 0 only): align request to 4K, first-fit walk of free list with spin-lock, split remainder if ≥ header + 4K. If free list has no fit, bump-allocate a new slab (min 128 KB or next-pow-2 of request) from the pool, split remainder into free list.
- **Free** (thread 0 only): push block to head of arena free list under spin-lock.
- **Compact** (one block, no concurrent ops): (1) drain all arenas into `compact_buf`; (2) warp 0 sorts addresses with `warp_sort_t<unsigned long long, HeapAddrCmp>`; (3) tid 0 scans sorted list, coalesces physically adjacent blocks, re-inserts by page index.
- `compact_buf` is 16 MB of dedicated device memory (not from the pool). The first half holds collected addresses; the second half is sort scratch. `HEAP_COMPACT_CAP ≈ 1M entries`.

Host init: zero-init `DeviceHeap` (zero head = empty list, zero lock = unlocked), set `pool` and `compact_buf` pointers.

## Implementation Notes

### DevicePool offset/capacity are 64-bit

`DevicePool.offset` and `DevicePool.capacity` are `unsigned long long`. Previously 32-bit, causing silent wraparound with >4GB scratch → overlapping memory regions → corrupted D&C edge pool free lists → illegal memory access.

### pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

### nvcc Crashes with --generate-line-info and Deep Recursion

nvcc crashed when compiling with `--generate-line-info` while `bt_computeInternal` was recursive. Converting to an iterative explicit stack resolved the crash. Line info is now always enabled for NCU profiling.

## Current Status

### Working
- D&C hull volume (`batch_hull_volume`) and mesh extraction (`batch_hull_dandc_mesh`) — tested (cube 8v/12t, tetra 4v/4t, gaussian); call sites for hull_dandc_warp_mesh not yet updated to new API
- Batch mesh volume (`batch_mesh_volume`) — divergence theorem, watertight meshes
- Warp sort (`test_warp_sort`) — bitonic + quicksort paths, duplicates
- Plane cut (`test_plane_cut`) — simple loop, ring, multi-hole, edge cases (14 tests); call sites not yet updated to new API

### Not Yet Implemented
- `__cuda_array_interface__` support for GPU tensor input
