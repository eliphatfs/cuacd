# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coacd_gpu** — GPU convex approximate decomposition. Single CPython extension (abi3, cp310+) via CUDA driver API. Sub-algorithms: D&C convex hull, plane cut, lookahead tree search decomposition. All run on GPU with a custom heap allocator.

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
  hull_dandc.cuh      #   Preparata-Hong D&C hull (Bullet port); hull_dandc_warp_mesh writes Mesh via heap into caller's __shared__ Mesh*
  warp_sort.cuh       #   Generic warp-cooperative quicksort template (warp_sort_t<T,Cmp>, warp_sort_inner, warp_pick_pivot, warp_partition) + BtPoint32 legacy API
  plane_cut.cuh       #   plane_cut_block device function + Edge2i/Edge2iCmp structs; returns PartPair via DeviceHeap
  kdop_hull.cuh       #   kdop_hull_block: single-warp (32 threads) exact hull via extreme-point prefilter + D&C
  mesh_volume.cuh     #   mesh_volume_warp: per-warp divergence theorem volume of a Mesh
  hausdorff.cuh       #   hausdorff_block: block-level (256 threads) bidirectional Hausdorff distance via sampling + linear BVH
  postprocess.cuh     #   decompose_components_block: block-level (128 threads) connected-components via union-find
  structs.cuh         #   Device-side: Mesh, Part, PartPair, LaWorkItem, LaDecompState, LaEvalResult
  mm.cu               #   heap_init_kernel
  kdop_const.cu       #   __constant__ KDOP_AXES[40][3] definition (broadcast-cached icosphere axes)
  lookahead.cu        #   lookahead search kernels: la_initialize, la_expand, la_evaluate, la_apply_cuts, la_decompose_components, etc.
  test_warp_sort.cu   #   Test kernel: test_warp_sort_kernel
  test_hull_dandc.cu  #   Test kernel: hull_dandc_kernel
  test_mesh_volume.cu #   Test kernel: mesh_volume_kernel
  test_kdop_hull.cu   #   Test kernel: kdop_hull_kernel
  test_plane_cut.cu   #   Test kernel: plane_cut_kernel
  test_hausdorff.cu   #   Test kernel: hausdorff_kernel
  test_postprocess.cu #   Test kernel: test_postprocess_dc_kernel
csrc/                 # C host code
  structs.h           #   Host-side structs: DevicePool, HeapArena, DeviceHeap, gpu_ctx
  heap.h              #   GPU context lifecycle (gpu_ctx_t, gpu_init/destroy/compact/pool_usage) + Mesh_h/Part_h/gpu_result host mirrors
  heap.c              #   Host implementation: gpu_init/destroy, pool management, result_free
  test.h              #   Declarations for kernel host launchers (hull, volume, plane cut, kdop, hausdorff, warp sort)
  test.c              #   Host launcher implementations
  postprocess.h       #   Declarations for post-processing host launchers (decompose_components)
  postprocess.c       #   Host launcher implementations for post-processing
  lookahead.h         #   Declaration for lookahead_decompose
  lookahead.c         #   Host implementation: lookahead_decompose (full lookahead tree search with Hausdorff)
  module.c            #   CPython extension wrapping heap.h/test.h/postprocess.h/lookahead.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (batch_hull_volume, batch_mesh_volume, batch_hull_dandc_mesh, batch_kdop_hull_mesh, lookahead_decompose)
  cli.py              #   coacd-gpu console entry point (normalize → decompose → denormalize → export GLB)
tests/                # All tests
  test_hull.py        #   Hull volume + mesh volume tests
  test_hull_mesh.py   #   D&C hull mesh extraction tests + k-DOP hull tests
  test_warp_sort.py   #   Tests for warp_sort_bp32
  test_plane_cut.py   #   Plane cut tests (16 tests)
  test_hausdorff.py   #   Hausdorff distance tests (5 tests) + CoACD reference comparison
  gen_hausdorff_fixtures.py # Generates CoACD reference Hausdorff fixtures (C++ harness + .npz)
  ref_hausdorff.cpp    #   Standalone C++ CoACD Hausdorff reference harness
  test_lookahead.py   #   lookahead_decompose tests (cube, lshape, octocat, convergence, export_glb)
  test_decompose_components.py # Connected-components decomposition tests (5 tests)
  test_edge_tracking.py # Max edge pairs stress test for D&C hull (requires COACD_TRACK_EDGES=1 build)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling
  bench_mm.py         #   Memory management benchmark
  bench_arena_sweep.py#   Arena count sweep benchmark
docs/                 # Detailed documentation
  arena_sweep.md      #   Arena count sweep results
  api_hull_dandc.md   #   D&C hull algorithm and hull_dandc_warp_mesh API
  api_plane_cut.md    #   plane_cut_block API details
  api_heap_allocator.md#  Heap arena allocator design
  api_kdop_hull.md    #   kdop_hull_block algorithm and API (extreme-point prefilter + D&C)
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
- **rv-only tree search with hausdorff stop is correct by design** — Our algorithm is based on CoACD. CoACD also uses rv-only cost for tree search but checks `max(rv, hausdorff)` for the stopping criterion. CoACD works. The mismatch between search cost and stop cost is intentional and not a bug. Do not propose "fixing" this mismatch as a solution to convergence issues.

### Python API

```python
import coacd_gpu
with coacd_gpu.Context(device=0, pool_bytes=0) as ctx:  # pool_bytes=0 → auto (70% free VRAM)
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
    results = ctx.batch_hull_dandc_mesh(pts_list)   # list of (verts, tris, volume) — exact D&C hull
    results = ctx.batch_kdop_hull_mesh(pts_list)    # list of (verts, tris, volume) — approximate k-DOP hull
    parts = ctx.lookahead_decompose(verts, tris, max_iters=100, width=60, width2=5, threshold=0.05,
                                     decompose_components=False)  # list of (verts, tris, hull_verts, hull_tris)
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

**Plane Cut** (`plane_cut.cuh`): Block-level (64 threads) mesh splitting along an arbitrary plane. Produces `PartPair` with pos/neg meshes. Handles disjoint components: when the plane doesn't intersect any edge but vertices exist on both sides, triangles are separated by vertex sign with per-side vertex compaction. Cap triangulation handles multiple boundary loops with arbitrary nesting depth via parity classification (even depth = outer, odd depth = hole). See `docs/api_plane_cut.md`.

**Hull with Extreme-Point Prefilter** (`kdop_hull.cuh`): Single-warp (32 threads). Fast path: direct D&C hull for nv ≤ 1024. Main path: warp argmax/argmin over 40 icosphere axes finds up to 80 extreme vertices → D&C rough inner hull → ballot/popcount half-space filter discards interior points → D&C final hull of survivors. Result is the exact convex hull. Exposed as `ctx.batch_kdop_hull_mesh()`. Used by `la_hull` kernel. See `docs/api_kdop_hull.md`.

**Hausdorff Distance** (`hausdorff.cuh`): Block-level (256 threads, 8 warps). Computes bidirectional Hausdorff distance between two meshes via sampling + linear BVH. CoACD-matching area-proportional sampling with Wang hash pseudo-random barycentric coordinates. Brute-force path for ≤64 target triangles; linear BVH (Karras 2012 radix tree) for larger meshes with cooperative 4-warp Morton code sort. Used by `la_hausdorff_parts` kernel to fill `Part.hausdorff`.

**Lookahead Search Decomposition** (`lookahead.cu` + `csrc/lookahead.c`): Maintains a flat decomposition (LaDecompState). For each part above threshold, explores a shallow tree of candidate cuts: `depth` full expansion levels (`width` cuts at level 0, `width2` at deeper levels), then `quick_depth` levels (1 best-axis midpoint cut each). Path cost = average worst-part cost across levels; cut selection = minimum path cost per initial cut. Uses rv-only cost for tree exploration (matching CoACD), but full cost `max(rv, hausdorff)` for the stopping criterion. Default depth=2, quick_depth=0, width=60, width2=5. `la_evaluate` falls back to level-0 items when no leaf descendants exist for a cut (all deeper expansions produced empty halves because pieces were too small to cut further — such pieces are provably below threshold). Kernels: la_initialize, la_sort_parts, la_hausdorff_parts, la_count_cutting, la_seed_tree, la_expand, la_expand_quick, la_hull, la_sort_items, la_record_level_cost, la_evaluate, la_apply_cuts, la_hull_decomp, la_cleanup_tree, la_free_decomp, la_decompose_components.

**Connected-Components Decomposition** (`postprocess.cuh`): Block-level (128 threads) post-processing pass. Uses lock-free rank-based union-find over triangle adjacency to identify connected components in each decomposition part. Read-only find (no path compression) for GPU performance; rank-based union with CAS for lock-free merging. When a part has multiple components, allocates new mesh data per component from the main heap and appends extra parts to LaDecompState via atomicAdd on nparts. Controlled by `decompose_components` parameter (default False in Python API, True in CLI). Inherits hausdorff distance as upper bound; hulls are recomputed via la_hull_decomp after splitting.

### Utility Functions

| Function | Level | File |
|----------|-------|------|
| `signed_tet_volume` | per-thread | geometry.cuh |
| `block_reduce_sum/bbox/max/count` | block (syncthreads) | reduce.cuh |
| `mesh_volume_warp` | warp (32 lanes) | mesh_volume.cuh |
| `hausdorff_block` | block (256 threads) | hausdorff.cuh |
| `kdop_hull_block` | warp (32 threads) | kdop_hull.cuh |
| `decompose_components_block` | block (128 threads) | postprocess.cuh |
| `pool_alloc` | thread 0 only | allocator.cuh |
| `heap_alloc` / `heap_free` | thread 0 only | allocator.cuh |
| `atomicMinF` / `atomicMaxF` | per-thread | common.cuh |
| `CheckedBuf<T>` | debug-mode OOB detection | common.cuh |

## Debugging

- **CUDA error 700/716 is sticky** — once triggered, all subsequent CUDA calls fail. The *first* error is the real one.
- **Isolate tests**: run single failing test alone to avoid cascade from earlier tests.
- **compute-sanitizer**: `compute-sanitizer --tool memcheck python <script.py>` to find exact source.
- **COACD_BEAM_DEBUG**: `COACD_BEAM_DEBUG=1 pip install -e .` enables `DPRINTF` (device-side `printf`) and `CheckedBuf` OOB detection. Use `DPRINTF` guarded by `if (lane == 0)` or `if (tid == 0)` to add temporary device-side diagnostics in `.cuh` files — no extra includes needed, `DPRINTF` is defined in `common.cuh`.
- **Debug mode**: `ctx.lookahead_decompose(..., debug=1)` syncs after each kernel, prints per-stage status.
- **Lookahead verbose levels** (`lookahead_decompose`):
  - `verbose=1`: per-part table each iteration (nv, nt, mesh_vol, hull_vol, rv_cost, hausdorff, full_cost; `*` = above threshold), per-iteration timing and n_cutting, kernel stage OK messages.
  - `verbose=2`: additionally prints cutting_indices, la_evaluate results (best_cut_idx, best_cost per src part), leaf item details (src, cut, nparts, path_cost, level_costs), level-0 item details (per-cut per-source with per-part rv/hv/mv), per-cut best path cost summary.
- See `docs/implementation_notes.md` for resolved bugs and gotchas.

## Current Status

### Working
- D&C hull, mesh volume, warp sort, plane cut (16 tests) — all tests pass.
- `kdop_hull_block` / `batch_kdop_hull_mesh` — 5 tests pass. Produces exact hull via extreme-point prefilter + D&C.
- `hausdorff_block` — 5 tests pass. Sampling-based bidirectional Hausdorff distance with linear BVH acceleration.
- `decompose_components_block` / `la_decompose_components` — 5 tests pass. Connected-components decomposition via GPU union-find. Integrated as optional post-processing pass in lookahead_decompose (decompose_components parameter).
- `lookahead_decompose` — cube, L-shape, octocat, convergence, 49160 tests pass. Uses full cost `max(rv, hausdorff)` for stopping criterion, rv-only for tree search. Default depth=2, quick_depth=0, width=60, width2=5. `la_count_cutting` deterministically selects highest-cost parts. `la_expand`/`la_expand_quick` place cuts evenly in the valid range `[lo+min_edge_dist, hi-min_edge_dist]` (min_edge_dist = threshold/4) to prevent degenerate thin slivers. Parts too small to cut in all axes (extent ≤ 2*min_edge_dist) are recorded as extra_leaves with zero cost for remaining levels — distinguishes genuinely solved parts from failed cuts (which get infinite cost). `plane_cut_block` detects incomplete cap triangulation (ear-clip gave up) and returns the whole mesh unsplit, so the lookahead tree search treats it as a failed cut rather than producing non-watertight parts. Two heap leak fixes: (1) `la_free_decomp` kernel frees final decomp part meshes/hulls after read-back; (2) `la_cleanup_tree` is called on `d_cur` before each buffer swap so that seed/intermediate items' refcounts are decremented before the buffer is reused.

### Known Limitations
- **Lookahead residual pool growth (~1 MB/call)**: After fixing two confirmed leaks (final decomp meshes never freed; seed buffer overwritten without refcount decrement before swap), pool usage still grows ~1 MB/call over 100 repeated calls. Cause unconfirmed — may be allocator bin fragmentation (varying chunk sizes → new slabs) or a remaining minor leak. Not yet experimentally distinguished.

### Not Yet Implemented
- `__cuda_array_interface__` support for GPU tensor input
- **Ternary search refinement** for lookahead cut selection: CoACD refines the MCTS-selected cut position via ternary search (`TernaryMCTS`, up to 10 iterations, epsilon=0.0001) to find the optimal cut within ±interval of the grid point. Our implementation uses only the fixed grid (width/3 cuts per axis). Adding refinement would let the algorithm find exact structural corners (e.g. the L-shape junction) instead of relying on the nearest grid point.
- **Centroid-based mesh volume**: `mesh_volume_warp` computes volume via divergence theorem relative to origin (`signed_tet_volume` sums tetrahedra formed with the origin). For non-watertight meshes (boundary edges from plane_cut ear-clipping giving up), this effectively connects open edges to the origin, introducing volume error proportional to the distance from origin to the hole. Computing relative to the mesh centroid instead would reduce this error since the implicit triangles closing the holes would be much smaller. Low priority — the main convergence issue is rv-only tree search, not volume accuracy.
