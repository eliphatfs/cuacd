# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coacd_gpu** — GPU convex approximate decomposition. Single CPython extension (abi3, cp310+) via CUDA driver API. Sub-algorithms: D&C convex hull, plane cut, beam search decomposition. All run on GPU with a custom heap allocator.

`CoACD/` — Reference C++ CoACD implementation (SIGGRAPH 2022), not part of the GPU extension.

## Repository Layout

```
setup.py              # Build config: setuptools builds _gpu extension
pyproject.toml        # PEP 621 metadata
cuda/                 # CUDA device code (compiled to single fatbin)
  allocator.cuh       #   DevicePool (bump alloc + embedded DeviceHeap×2) + pool_alloc + heap_alloc/free
  common.cuh          #   Constants, atomics, CheckedBuf<T> (includes allocator.cuh)
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   signed_tet_volume (divergence theorem kernel helper)
  warp_common.cuh     #   WarpPool allocator + warp reductions (warp_min_f, warp_max_f, etc.)
  hull_dandc.cuh      #   Preparata-Hong D&C hull (Bullet port); hull_dandc_warp_mesh returns Mesh via heap
  warp_sort.cuh       #   Generic warp-cooperative quicksort template (warp_sort_t<T,Cmp>) + BtPoint32 legacy API
  plane_cut.cuh       #   plane_cut_block device function + Edge2i/Edge2iCmp structs; returns PartPair via DeviceHeap
  kdop_hull.cuh       #   kdop_hull_block: block-level (64 threads) approximate hull via k-DOP (40 icosphere axes)
  mesh_volume.cuh     #   mesh_volume_warp: per-warp divergence theorem volume of a Mesh
  structs.cuh         #   Device-side: Mesh, Part, PartPair, WorkItem, AlgoState
  mm.cu               #   heap_init_kernel
  beam.cu             #   beam_expansion, beam_hull, beam_sort, beam_finalize kernels
  test_warp_sort.cu   #   Test kernel: test_warp_sort_kernel
  test_hull_dandc.cu  #   Test kernel: hull_dandc_kernel
  test_kdop_hull.cu   #   Test kernel: kdop_hull_kernel
  test_plane_cut.cu   #   Test kernel: plane_cut_kernel
csrc/                 # C host code
  structs.h           #   Host-side structs: DevicePool, HeapArena, DeviceHeap, beam_ctx
  beam.h              #   Public C API (beam_ctx_t, beam_init/destroy/compact/pool_usage, batch ops)
  beam.c              #   Host implementation: beam_init/destroy, beam_heap_compact, beam_pool_usage
  test_beam.c         #   Test host launchers
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (batch_hull_volume, batch_mesh_volume, batch_hull_dandc_mesh, batch_kdop_hull_mesh)
tests/                # All tests
  test_hull.py        #   Hull volume + mesh volume tests
  test_hull_mesh.py   #   D&C hull mesh extraction tests + k-DOP hull tests
  test_warp_sort.py   #   Tests for warp_sort_bp32
  test_plane_cut.py   #   Plane cut tests (14 tests)
  test_decompose.py   #   beam_decompose tests (cube, lshape, octocat, octocat_debug_steps)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling
  bench_mm.py         #   Memory management benchmark
  bench_arena_sweep.py#   Arena count sweep benchmark
docs/                 # Detailed documentation
  arena_sweep.md      #   Arena count sweep results
  api_hull_dandc.md   #   D&C hull algorithm and hull_dandc_warp_mesh API
  api_plane_cut.md    #   plane_cut_block API details
  api_beam.md         #   Beam search kernel APIs (initialize, expansion, hull, sort, finalize, decompose)
  api_heap_allocator.md#  Heap arena allocator design
  api_kdop_hull.md    #   k-DOP approximate hull algorithm and kdop_hull_block API
  implementation_notes.md# Resolved bugs and implementation gotchas
CoACD/                # Reference C++ CoACD (embedded repo, not a submodule)
```

## Git Usage

Only stage project source files — never use `git add -A` or `git add .`. The repo contains directories that must not be committed:
- `CoACD/` — embedded git repository (not a submodule)
- `compare_output/`, `octocat_output/`, `tmpcompare/` — temporary output directories
- `*.ncu-rep` — NSight Compute profiling artifacts
- Build artifacts (`*.fatbin`, `*.o`, `*.so`, `build/`, `*.egg-info/`)

Before staging, always run `git status` and `git diff --stat` to verify only expected tracked files are modified.

## Build Commands

```bash
# Install (requires CUDA toolkit with nvcc, C compiler)
pip install -e .

# Fast dev build — single arch for current GPU (RTX 4090 = sm_89, ~15s)
COACD_GPU_ARCHS="89" pip install -e .

# Run all tests
python -m pytest tests/ -v

# Verbose build (see ptxas register usage)
pip install -ve .

# Debug builds
COACD_DEBUG=1 pip install -e .          # host-side debug output
COACD_BEAM_DEBUG=1 pip install -e .     # device-side DPRINTF + CheckedBuf OOB detection

# Override arena count (default 64)
COACD_GPU_ARENAS=32 pip install -e .

# NCU profiling
ncu --set full -o dandc_profile python tests/bench_dandc.py --n_pts 200 --n_hulls 8
```

Dependencies: `numpy`, `pytest` (test only), `trimesh` (comparison only), `manifold3d` (optional, ring/multi-hole plane cut tests).

## Workflow Rule

After completing any code change, always build (`pip install -e .`) and run the full test suite (`python -m pytest tests/ -v`), skipping any tests that were already failing before the change.

## Architecture

### Build System

Setuptools compiles each `.cu` module in parallel (`-rdc=true -dc`), then device-links into `kernels.fatbin` → C header, then builds the C extension with `Py_LIMITED_API` (cp310+, abi3 wheel). Fatbin includes `--generate-line-info` for NCU.

**Parallel separate compilation**: all `__device__` functions in `.cuh` headers **must** be `inline` (or `__forceinline__`/`static`) to avoid duplicate symbol errors from the device linker.

### Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **No artificial limits** — use natural bounds (e.g. Euler's formula), not hardcoded constants.
- **Evidence-based debugging** — Do not guess errors from partial output. Write a minimal reproducer or add instrumentation to observe the actual failure before making fixes.

### Python API

```python
import coacd_gpu
with coacd_gpu.Context(device=0, pool_bytes=0) as ctx:  # pool_bytes=0 → auto (70% free VRAM)
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
    results = ctx.batch_hull_dandc_mesh(pts_list)   # list of (verts, tris, volume) — exact D&C hull
    results = ctx.batch_kdop_hull_mesh(pts_list)    # list of (verts, tris, volume) — approximate k-DOP hull
    used = ctx.pool_usage()     # bytes consumed from pool (monotonic high-water mark)
    ctx.heap_compact()          # no-op (coalescing handled by heap_free)
```

### Memory Architecture

One `DevicePool` with bump allocator backs two embedded `DeviceHeap` instances (`heap` for output, `scratch` for temporaries). Both heaps use arena-based free lists with O(1) coalescing — see `docs/api_heap_allocator.md`. Pool offset never decreases; freed blocks are recycled via free-lists. Pool stabilizes after first call.

### Pool Allocator Pattern

**Critical**: only thread 0 calls `pool_alloc()`, stores pointer in `__shared__`, then all threads read after `__syncthreads()`. If all threads call `pool_alloc`, each gets a different offset → data corruption.

### Algorithm Overview

**D&C Convex Hull** (`hull_dandc.cuh`): Warp-level parallel tree merge — 16 primary lanes each build a small hull (BT_HULL_GROUPS=16), then 4 rounds of pairwise merging using 16×2 thread pairs. Within each merge, both threads in a pair call `bt_findMaxAngle` convergedly (primary for c0/h0, secondary for c1/h1); results are exchanged via warp shuffle. All `BtVertex*` pointer fields (`next`, `prev`, `BtEdge::target`, `BtIntermediateHull` extremals) are stored as `BtVIndex` (int index into the shared `vblock` array) to reduce memory and improve locality.

Memory layout in `hull_dandc_warp_mesh`: presort `BtPoint32` array is heap-allocated from `scratch_heap` (not WarpPool) and freed immediately after the vertex-init copy in postsort, before the D&C phase. Edge pools (BtPool + initial slabs) exist only for BT_HULL_GROUPS=16 primaries — secondaries never allocate edges, so `edgePool.blocks` is indexed by `group` (not `lane`). WarpPool backing covers only sort scratch + postsort persistent data + D&C stacks/BFS queues. See `docs/api_hull_dandc.md`.

**Plane Cut** (`plane_cut.cuh`): Block-level (64 threads) mesh splitting along an arbitrary plane. Produces `PartPair` with pos/neg meshes. See `docs/api_plane_cut.md`.

**k-DOP Approximate Hull** (`kdop_hull.cuh`): Block-level (64 threads) approximate convex hull using 40 icosphere-L1 face normals as k-DOP axes (80 half-spaces total). Algorithm: centroid → block-parallel k-DOP extreme projections → polar dual vertices → D&C hull of dual points (warp 0) → 3-plane intersection per dual triangle → D&C hull of primal vertices (warp 0) → volume. Exposed as `ctx.batch_kdop_hull_mesh()`. Used by `beam_hull` kernel. The k-DOP overestimates the true convex hull volume — this inflates the concavity cost `hull_vol - mesh_vol` and causes the beam search to over-decompose unless a correction factor is applied.

**Beam Search Decomposition** (`beam.cu` + `csrc/beam.c`): Iterative beam search: `finalize → expansion → hull → sort`. Pipelined in stream order, sync only after finalize. `beam_hull` uses `kdop_hull_block` (64 threads/block). See `docs/api_beam.md`.

### Utility Functions

| Function | Level | File |
|----------|-------|------|
| `signed_tet_volume` | per-thread | geometry.cuh |
| `block_reduce_sum/bbox/max/count` | block (syncthreads) | reduce.cuh |
| `mesh_volume_warp` | warp (32 lanes) | mesh_volume.cuh |
| `kdop_hull_block` | block (64 threads) | kdop_hull.cuh |
| `pool_alloc` | thread 0 only | allocator.cuh |
| `heap_alloc` / `heap_free` | thread 0 only | allocator.cuh |
| `atomicMinF` / `atomicMaxF` | per-thread | common.cuh |
| `CheckedBuf<T>` | debug-mode OOB detection | common.cuh |

## Debugging

- **CUDA error 700/716 is sticky** — once triggered, all subsequent CUDA calls fail. The *first* error is the real one.
- **Isolate tests**: run single failing test alone to avoid cascade from earlier tests.
- **compute-sanitizer**: `compute-sanitizer --tool memcheck python <script.py>` to find exact source.
- **CheckedBuf**: `COACD_BEAM_DEBUG=1 pip install -e .` enables OOB detection in device code.
- **Debug mode**: `ctx.decompose(..., debug=1)` syncs after each kernel, prints per-stage status.
- **Per-substep memory profiling**: `python -m pytest tests/test_decompose.py::test_octocat_decompose_debug_steps -v -s` — runs octocat with `debug=1`, use captured stdout/stderr to see per-substep pool usage.
- See `docs/implementation_notes.md` for resolved bugs and gotchas.

## Current Status

### Working
- D&C hull, mesh volume, warp sort, plane cut (14 tests), beam_decompose (cube/lshape/octocat) — all tests pass.
- k-DOP approximate hull (`kdop_hull_block`, `batch_kdop_hull_mesh`) — 5 tests pass. Used by `beam_hull`.

### Known Limitations
- k-DOP hull **overestimates** true convex hull volume (it is a superset). In `beam_hull` this inflates `hull_vol - mesh_vol`, causing the decomposition to split more aggressively than with exact hull. A volume correction factor or fallback to D&C hull may be needed for production quality.

### Not Yet Implemented
- `__cuda_array_interface__` support for GPU tensor input
