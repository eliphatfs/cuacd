# CLAUDE.md

## Project Overview

**cuacd** — GPU convex approximate decomposition. Single CPython extension (abi3, cp310+) via CUDA driver API. Sub-algorithms: D&C convex hull, plane cut, lookahead tree search decomposition. All run on GPU with a custom heap allocator.

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
  warp_common.cuh     #   WarpPool allocator + warp reductions
  hull_dandc.cuh      #   Preparata-Hong D&C hull (Bullet port)
  warp_sort.cuh       #   Generic warp-cooperative quicksort template + BtPoint32 legacy API
  plane_cut.cuh       #   plane_cut_block device function; returns PartPair via DeviceHeap
  kdop_hull.cuh       #   kdop_hull_block: single-warp exact hull via extreme-point prefilter + D&C
  mesh_volume.cuh     #   mesh_volume_warp: per-warp divergence theorem volume
  hausdorff.cuh       #   hausdorff_block: block-level bidirectional Hausdorff via sampling + linear BVH
  postprocess.cuh     #   decompose_components_block: block-level connected-components via union-find
  merge.cuh           #   Merge-hulls helpers: flat upper-tri pair index, AABB gap, warp concat, shared-plane finder
  structs.cuh         #   Device-side: Mesh, Part, PartPair, LaWorkItem, LaDecompState, LaEvalResult, ConcaveEdgePlane
  error_codes.cuh     #   Centralized GPU kernel error flags (KERR_*)
  la_common.cuh       #   Shared lookahead constants + la_part_cost/la_part_cost_rv inline functions
  mm.cu               #   heap_init_kernel
  kdop_const.cu       #   __constant__ KDOP_AXES[40][3] (icosphere axes)
  la_expand.cu        #   la_expand, la_hull, la_compute_best_ub, la_seed_tree, la_cleanup_tree, la_cleanup_tree3
  la_refine.cu        #   la_expand_quick, la_hausdorff_parts, la_evaluate, la_sort_and_record
  la_lifecycle.cu     #   la_initialize, la_sort_and_count_cutting, la_apply_cuts, la_hull_decomp, la_decompose_components, la_free_decomp
  la_concave.cu       #   la_find_concave_edges: concave edge detection + plane generation
  postprocess_merge.cu #   Merge-hulls pass: la_merge_cost_matrix, la_merge_hausdorff, la_merge_match, la_merge_apply, la_merge_free_unused, la_merge_compact
  test_*.cu           #   Test kernels
csrc/                 # C host code
  structs.h           #   Host-side structs: DevicePool, HeapArena, DeviceHeap, gpu_ctx
  heap.h / heap.c     #   GPU context lifecycle + pool management
  test.h / test.c     #   Kernel host launchers
  postprocess.h/c     #   Post-processing host launchers
  error_codes.h       #   Host-side error decoding (kerr_decode)
  lookahead.h/c       #   lookahead_decompose host implementation
  module.c            #   CPython extension (Py_LIMITED_API cp310)
cuacd/                # Python package
  __init__.py         #   Context class
  cli.py              #   cuacd console entry point
tests/                # All tests + benchmarks
bench/                # Standalone perf experiments (not built by setup.py)
  warp_sort_bench.cu  #   std::sort vs warp_sort vs cub BlockMergeSort/BlockRadixSort (+4-pass int4 radix)
  Makefile            #   warp_sort bench build (ARCH=89 default); `make verbose` for ptxas -v
  hull_bench.cu       #   btConvexHull (CPU) vs hull_dandc vs kdop_hull (batch=256 × {512,2048,16384,65536} pts × gaussian/uniform_cube)
  Makefile.hull       #   hull bench build (rdc=true; links mm.cu, kdop_const.cu, CoACD btConvexHull .cpp)
  malloc_test_bench.cu #  mstress-style alloc/free stress: coacd heap vs CUDA __device__ malloc; sweeps grid size for arena-contention curve
  Makefile.malloc     #   malloc bench build (rdc=true; links mm.cu)
  plot_malloc_and_hausdorff.py  #   Two-panel figure: malloc throughput + Hausdorff CPU vs GPU bars
  prepare_hausdorff_data.py     #   Load meshes, normalize, convex hull → hausdorff_data.h
  hausdorff_bench.cu            #   GPU hausdorff_block benchmark (matches warp_sort_bench timing style)
  bench_cpu_hausdorff.cpp       #   CPU CoACD face_hausdorff_distance wrapper (nanoflann, no openvdb)
  Makefile.hausdorff            #   Build for hausdorff_bench (rdc=true; links mm.cu + CoACD shape.cpp)
  summarize.py        #   Pivot results.csv → text / Markdown tables (--md, --ptxas)
  parse_ptxas.py      #   Parse ptxas.log → per-kernel reg / smem / spill table
docs/                 # Documentation (see below)
CoACD/                # Reference C++ CoACD (embedded repo, not a submodule)
```

## Build Commands

```bash
pip install -e .                                    # Install (requires CUDA toolkit with nvcc)
CUACD_GPU_ARCHS="89" pip install -e .               # Fast dev build — single arch (RTX 4090 = sm_89, ~15s)
python -m pytest tests/ -v                          # Run all tests
CUACD_DEBUG=1 pip install -e .                      # Host-side debug output
CUACD_BEAM_DEBUG=1 pip install -e .                 # Device-side DPRINTF + CheckedBuf OOB detection
CUACD_GPU_ARENAS=32 pip install -e .                # Override arena count (default 64)
CUACD_MEMCHECK=1 pip install -e .                    # Device-side memory sanitizer (-fdevice-sanitize=memcheck)
CUACD_LEAK_PROBE=1 pip install -e .                 # Device-side refcount-mismatch probe in la_free_decomp
CUACD_LEAK_BISECT=1 python your_bench.py            # Runtime: per-phase heap_stats deltas around each kernel
CUACD_PARALLEL=4 pip install -e .                   # Limit parallel nvcc processes
pip install -ve .                                   # Verbose build (see ptxas register usage)
```

Dependencies: `numpy`, `pytest` (test only), `trimesh` (comparison only), `manifold3d` (optional, ring/multi-hole plane cut tests).

## Workflow Rule

After completing any code change, always build (`pip install -e .`) and run the full test suite (`python -m pytest tests/ -v`), skipping any tests that were already failing before the change.

## Design Principles

- **CUDA driver API only** — links `libcuda.so`, not `libcudart.so`. One binary across CUDA 11.x-12.x+.
- **Fatbin embedding** — cubins for sm_80/86/89/90 + PTX for forward compat.
- **abi3 wheel** — `Py_LIMITED_API` targeting Python 3.10+.
- **No PyTorch dependency** — numpy arrays in/out. Reuses existing CUDA context if available.
- **No artificial limits** — use natural bounds (e.g. Euler's formula), not hardcoded constants.
- **Parallel separate compilation**: all `__device__` functions in `.cuh` headers **must** be `inline` (or `__forceinline__`/`static`) to avoid duplicate symbol errors from the device linker.

## Python API

```python
import cuacd
with cuacd.Context(device=0, pool_bytes=0) as ctx:  # pool_bytes=0 → auto (70% free VRAM)
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
    results = ctx.batch_hull_dandc_mesh(pts_list)   # list of (verts, tris, volume) — exact D&C hull
    results = ctx.batch_kdop_hull_mesh(pts_list)    # list of (verts, tris, volume) — approximate k-DOP hull
    parts = ctx.lookahead_decompose(verts, tris, max_iters=100, width=60, width2=5, threshold=0.05,
                                     decompose_components=False,
                                     no_decompose_components_per_iter=False,
                                     n_concave_edges=32, concave_eps=0.005,
                                     concave_threshold=3.49, concave_iters=10,
                                     merge_hulls=False)  # list of (verts, tris, hull_verts, hull_tris)
    used = ctx.pool_usage()     # bytes consumed from pool (monotonic high-water mark)
    ctx.heap_compact()          # no-op (coalescing handled by heap_free)
    ho, ha, hf, so, sa, sf = ctx.heap_stats()  # diag: (live_bytes, allocs, frees) × (heap, scratch)
```

## Documentation

| Doc | Content |
|-----|---------|
| `docs/algorithms.md` | Algorithm overviews (D&C hull, plane cut, k-DOP hull, Hausdorff, lookahead, connected-components), utility function table, memory architecture |
| `docs/debugging.md` | Debugging guide: CUDA error handling, build flags, runtime debug/verbose modes, key debugging principles |
| `docs/status.md` | Current status (working features), known limitations, not-yet-implemented features |
| `docs/implementation_notes.md` | Resolved bugs and implementation gotchas |
| `docs/api_hull_dandc.md` | D&C hull algorithm details and `hull_dandc_warp_mesh` API |
| `docs/api_plane_cut.md` | `plane_cut_block` API and phase-by-phase details |
| `docs/api_kdop_hull.md` | `kdop_hull_block` algorithm and API |
| `docs/api_heap_allocator.md` | Heap arena allocator design and benchmarks |
| `docs/arena_sweep.md` | Arena count sweep benchmark results |
| `docs/dev_log_la_perf.md` | Lookahead/`la_hull` perf optimization log: each attempt, outcome, kept/reverted |
