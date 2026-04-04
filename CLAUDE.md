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
  allocator.cuh       #   DevicePool (bump alloc + embedded DeviceHeap×2) + pool_alloc + heap_alloc/free
  common.cuh          #   Constants, atomics, CheckedBuf<T> (includes allocator.cuh)
  reduce.cuh          #   Block-level parallel reductions (sum, bbox)
  geometry.cuh        #   Edge intersection, point-triangle distance, concavity metrics
  warp_common.cuh     #   WarpPool allocator + warp reductions (warp_min_f, warp_max_f, etc.)
  hull_dandc.cuh      #   Preparata-Hong D&C hull (Bullet port); hull_dandc_warp_mesh returns Mesh via heap
  warp_sort.cuh       #   Generic warp-cooperative quicksort template (warp_sort_t<T,Cmp>) + BtPoint32 legacy API
  plane_cut.cuh       #   plane_cut_block device function + Edge2i/Edge2iCmp structs; returns PartPair via DeviceHeap
  mesh_volume.cuh     #   mesh_volume_warp: per-warp divergence theorem volume of a Mesh
  mm.cu               #   heap_init_kernel: <<<2×HEAP_NUM_ARENAS,32>>> initialises both embedded heaps in DevicePool
  beam.cu             #   beam_expansion kernel: <<<3×cuts_per_axis×nitems, 64>>> cuts last part of each WorkItem; beam_hull kernel: <<<2×nitems, 32>>> fills Part.hull via D&C convex hull; beam_sort kernel: <<<nitems, 32>>> sorts parts by part_cost; beam_finalize kernel: <<<max_keep, 1024>>> clears prev AlgoState and compacts current to best max_keep items; DPRINTF macro (device printf gated on COACD_BEAM_DEBUG compile flag)
  test_warp_sort.cu   #   Test kernel: test_warp_sort_kernel
  test_hull_dandc.cu  #   Test kernel: hull_dandc_kernel (hull mesh extraction)
  test_plane_cut.cu   #   Test kernel: plane_cut_kernel (thin wrapper around plane_cut_block)
csrc/                 # C host code
  structs.h           #   Host-side structs: DevicePool, HeapArena, DeviceHeap, beam_ctx (with persistent heaps)
  beam.h              #   Public C API (beam_ctx_t, beam_init/destroy/compact/pool_usage, batch ops)
  beam.c              #   Host implementation: beam_init/destroy, beam_heap_compact, beam_pool_usage
  test_beam.c         #   Test host launchers: beam_test_warp_sort, beam_hull_dandc, beam_test_plane_cut
  beam_module.c       #   CPython extension wrapping beam.h (Py_LIMITED_API cp310)
coacd_gpu/            # Python package (import name)
  __init__.py         #   Context class (batch_hull_volume, batch_mesh_volume, batch_hull_dandc_mesh)
tests/                # All tests
  test_hull.py        #   Hull volume + mesh volume tests (CPU ref, GPU D&C vs scipy, noisy icosphere); standalone mode prints GPU benchmark table with peak pool MB
  test_hull_mesh.py   #   D&C hull mesh extraction tests (batch_hull_dandc_mesh)
  bench_dandc.py      #   D&C hull benchmark for NCU profiling (gaussian points)
  bench_mm.py         #   Memory management benchmark: compact overhead, pool growth, arena effects
  bench_arena_sweep.py#   Arena count sweep: rebuild + run benchmark for each HEAP_NUM_ARENAS value
  test_warp_sort.py   #   Tests for warp_sort_bp32 (bitonic + quicksort paths)
  test_plane_cut.py   #   Plane cut tests: simple loop, ring, multi-hole, edge cases (14 tests)
  test_decompose.py   #   beam_decompose tests (cube, lshape, octocat); volume comparison table (GPU vs trimesh vs scipy)
docs/                 # Analysis and benchmark results
  arena_sweep.md      #   Arena count sweep results (HEAP_NUM_ARENAS ∈ {32,64,128,256})
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

# Override GPU architectures (default: 80;86;89;90 — ~57s)
COACD_GPU_ARCHS="80;86" pip install -e .

# Fast development build — single arch for current GPU (RTX 4090 = sm_89, ~15s)
COACD_GPU_ARCHS="89" pip install -e .

# Run all tests
python -m pytest tests/ -v

# Build with verbose host-side debug output
COACD_DEBUG=1 pip install -e .

# Build with device-side beam debug printfs (DPRINTF in beam.cu)
COACD_BEAM_DEBUG=1 pip install -e .

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

**`coacd_gpu._gpu`** — Setuptools-built CPython extension. `setup.py` compiles each `.cu` module in parallel (`-rdc=true -dc`) using `concurrent.futures.ThreadPoolExecutor`, then runs `nvcc --device-link --fatbin` to produce `kernels.fatbin` → C header, then builds `csrc/beam_module.c` + `csrc/beam.c` + `csrc/test_beam.c` as a native Python extension with `Py_LIMITED_API` (cp310+, abi3 wheel). Fatbin compiled with `--generate-line-info` for NCU source-level profiling.

**Parallel separate compilation requirement**: because modules are compiled independently with `-rdc=true`, all `__device__` functions defined in `.cuh` headers **must** be marked `inline` (or `__forceinline__`/`static`). Without `inline`, the device linker sees duplicate external symbol definitions from each `.o` that includes the same header. 
**File split:**
- `cuda/mm.cu` — `heap_init_kernel` (2×HEAP_NUM_ARENAS blocks × 32 threads; initialises both embedded heaps in DevicePool)
- `cuda/test_hull_dandc.cu` — `hull_dandc_kernel` (hull mesh extraction)
- `cuda/test_warp_sort.cu` — `test_warp_sort_kernel`
- `cuda/test_plane_cut.cu` — `plane_cut_kernel` (thin `__global__` wrapper around `plane_cut_block`)
- `csrc/beam.c` — `beam_init/destroy`, `beam_heap_compact`, `beam_pool_usage`
- `csrc/test_beam.c` — `beam_test_warp_sort`, `beam_hull_dandc`, `beam_test_plane_cut`

**Struct locations:**
- `cuda/allocator.cuh` — device-side: `DevicePool` (with embedded `DeviceHeap heap/scratch`), `HeapArena`, `heap_alloc/free`; included by `common.cuh`
- `cuda/structs.cuh` — device-side: `Mesh` (verts, tris, nv, nt, refcount), `Part`, `PartPair`, `WorkItem`, `AlgoState`; included by `plane_cut.cuh` and future algo code
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
# pool_bytes=0 → auto (70% of free VRAM at init time)
with coacd_gpu.Context(device=0, pool_bytes=0) as ctx:
    volumes, errors = ctx.batch_hull_volume(pts_list)
    volumes = ctx.batch_mesh_volume(verts_list, tris_list)
    results = ctx.batch_hull_dandc_mesh(pts_list)  # list of (verts, tris, volume)
    used = ctx.pool_usage()     # bytes consumed from shared pool (monotonic)
    ctx.heap_compact()          # compact both heaps (see heap compact note)
```

### Persistent Heaps

`beam_init` allocates one `DevicePool` (`d_pool_struct`, ~133 KB on device) that contains both embedded `DeviceHeap` instances. Both share the same bump-allocator backing. Pool size defaults to 70% of free device memory.

- **pool->heap** — output heap: holds temporary per-kernel output (freed by kernel before return)
- **pool->scratch** — scratch heap: holds per-kernel working data (freed by kernel before return)
- **Shared pool**: one `DevicePool.base/offset/capacity` with a shared `atomicAdd` counter
- **Pool offset never decreases**: freed blocks coalesce in free-lists (not returned to pool)
- **`beam_pool_usage()`**: reads back the pool offset — shows peak (high-water mark) of live device bytes

`heap_free` now performs O(1) coalescing with adjacent free blocks (via boundary sentinels + doubly-linked free lists), so `heap_compact()` is a no-op and need not be called. Both kernels call `heap_free` on all allocations before returning; pool offset stabilizes after the first call.

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

### W. Warp-Level Functions (all 32 lanes call)

| ID | Function | File |
|----|----------|------|
| W1 | `mesh_volume_warp(mesh, lane) -> float` | mesh_volume.cuh |

### I. Memory / Infrastructure

| ID | Function | File |
|----|----------|------|
| I1 | `pool_alloc(pool, size) -> void*` | allocator.cuh |
| I2 | `global_alloc_warp(pool, bytes, lane) -> void*` | allocator.cuh |
| I3 | `atomicMinF / atomicMaxF` | common.cuh |
| I6 | `CheckedBuf<T>` — debug-mode bounds-checked buffer | common.cuh |
| I4 | `heap_alloc(heap, size, out) -> int` | allocator.cuh |
| I5 | `heap_free(heap, ptr) -> int` | allocator.cuh |

### J. Dead Code

| ID | What | Why dead |
|----|------|----------|
| J1 | `compute_concavity_tris` in geometry.cuh | Bbox cube-root proxy, superseded by Rv |
| J2 | `hull_dandc_warp` | Removed; use `hull_dandc_warp_mesh` for all callers |
| J3 | `heap_compact(heap) -> int` | Removed from allocator.cuh; `beam_heap_compact()` is now a no-op |
| J4 | `query_dandc_scratch` kernel | Removed from test_hull_dandc.cu, beam.c, structs.h, test_beam.c; scratch is heap-managed, no pre-query needed |
| J5 | `bt_computeVolume` in hull_dandc.cuh | Volume now computed via `mesh_volume_warp` on extracted triangle mesh; half-edge BFS volume never called |

## bt_findMaxAngle Profiling (Edge Degree)

`bt_findMaxAngle` iterates a vertex's circular edge list to find the best merge angle. Instrumentation (gated on `COACD_BEAM_DEBUG`) measures edges chased per call. `BtHullState` carries 4 counter fields (`fma_total_edges`, `fma_min_edges`, `fma_max_edges`, `fma_calls`); lane 0 accumulates after shuffling lane 1's count.

Results on octocat mesh (20k vertices, realistic distribution):
- **Average degree: 3.0–3.9** across all hull sizes (86 to 20k points)
- **Max degree: 20–42** (rare outliers)
- **Calls per hull: ~180k** for 20k-point hulls

**Conclusion**: parallelizing the inner edge loop (batch K edges across K lanes) is not viable — average degree ~3.5 means most lanes would be idle. The bottleneck is the sheer number of `bt_findMaxAngle` calls, not work per call. Alternative data structures (linear arrays replacing the linked list) are also not justified at this degree.

## plane_cut_block API

`plane_cut_block(mesh, pa, pb, pc_n, pd, heap, scratch_heap, kernel_error) -> PartPair` — device function, one block (64 threads).

- **Input**: `const Mesh*` (replaces separate verts/tris/nv/nt params).
- **Output**: returns a `PartPair` directly (stored in `__shared__ PartPair s_result`); `pos.mesh` and `neg.mesh` each have their own independent heap allocation from `DeviceHeap* heap`.
- **Heap chunk layout** (two `heap_alloc` calls on `heap`): pos chunk `[verts | tris | refcount(16B)]`, neg chunk `[verts | tris | refcount(16B)]`, each section 16-byte aligned. `pos.mesh.refcount` and `neg.mesh.refcount` point to independent `int`s in their respective chunks; each initialized to 1. `pos.mesh.verts` and `neg.mesh.verts` are both the starts of their respective allocations — `heap_free(mesh.verts)` is always valid. Early-exit (no-cross) case allocates one chunk `[verts | tris | refcount(16B)]` for the non-empty side. Input meshes not allocated via `plane_cut_block` have `refcount = NULL`.
- **Scratch**: all temporary buffers (signs, all_verts, cross_edges, sort_scratch, isect_idx, pos_tris, neg_tris, dir_edges, dir_sort, boundary_flags, and per-loop working buffers) are allocated from `DeviceHeap* scratch_heap` and freed individually as soon as each buffer's last use completes. Nothing from scratch_heap survives the call.
- **Vertex compaction** (phase 13, parallel): phases 1-12 run on thread 0 or warp 0; phase 13 runs across all 64 threads. Steps: init remap[]=-1 (strided), mark used verts via `atomicMax` (strided), count per contiguous chunk, exclusive prefix scan (thread 0, 64 iters), assign new indices per chunk, remap tris in-place (strided), heap alloc (thread 0), scatter verts + copy tris (strided). Each side's `Mesh.verts` contains only the vertices actually referenced — no loose vertices.
- **Counters** (`n_cross`, `n_all_verts`, `n_pos`, `n_neg`): stored in `__shared__ int s_counters[4]`; `atomicAdd` on shared memory. Not on heap.
- **Early exit** (`n_cross == 0`): entire input mesh goes to one side; allocates one heap chunk for verts+tris, other side gets empty `Mesh {NULL,NULL,0,0}`.
- **No-boundary / one-empty-side cases**: handled by natural fallthrough — compaction produces a 0-entry side correctly.
- **Call sites**: `test_plane_cut.cu` (thin `__global__` wrapper) and `test_beam.c` (host launcher) pass `DeviceHeap*` pointers into the embedded heaps of `DevicePool`.

## beam_initialize API

`beam_initialize<<<3, 32>>>(verts, tris, nv, nt, hull_verts, hull_tris, hull_nv, hull_nt, current)` — seeds `current` with a single WorkItem containing one Part from the input mesh and its precomputed convex hull.

- **Block 0**: all 32 lanes call `mesh_volume_warp` on the input mesh; lane 0 writes result to `current->items[0].parts[0].mesh_vol`.
- **Block 1**: all 32 lanes call `mesh_volume_warp` on the hull mesh; lane 0 writes result to `current->items[0].parts[0].hull_vol`.
- **Block 2**: thread 0 sets `parts[0].mesh` and `parts[0].hull` to point directly at the caller-supplied device pointers (both `refcount = NULL` — input arrays are not heap-allocated and will not interact with `heap_free`); sets `hausdorff = 0`; sets `nparts = 1`; sets `nitems = 1`.
- **No error argument**: initialization cannot fail; volumes are computed independently of allocation.
- **Block ordering**: blocks 0/1 build a local `Mesh` from kernel parameters directly (not reading from `current->items`) so there is no data dependency on block 2's writes. All three blocks write disjoint fields of `parts[0]`.

## beam_expansion API

`beam_expansion<<<3*cuts_per_axis*current->nitems, 64>>>(current, next, pool, cuts_per_axis, err)` — one block per (item, cut); produces candidate WorkItems in `next`. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x / (3*cuts_per_axis)`, `axis = (blockIdx.x % (3*cuts_per_axis)) / cuts_per_axis`, `slice = … % cuts_per_axis`.
- **Plane**: axis-aligned at `(slice+1)/(cuts_per_axis+1)` fraction of the last part's bounding box. Bbox via three-stage reduction: per-thread local min/max → `warp_min_f`/`warp_max_f` → lane-0-per-warp `atomicMinF`/`atomicMaxF` into shared memory.
- **Calls `plane_cut_block`** on `wi->parts[nparts-1].mesh` using `pool->heap` and `pool->scratch`.
- **Empty side**: if either `pos.mesh.nv == 0` or `neg.mesh.nv == 0`, frees both pos and neg heap chunks independently and returns — no entry written to `next`.
- **Overflow**: `BEAM_ERR_OVERFLOW = 0x10000` — `atomicOr`'d into `err` if `nparts+1 > WORK_ITEM_MAX_PARTS`. Distinct from any plane_cut error code (those are small integers).
- **Part copy**: bulk int-copy of `parts[0..nparts-2]` in parallel; thread 0 appends `pp.pos` and `pp.neg`, sets `nparts = old_nparts + 1`. After the copy, threads iterate over the `nparts-1` copied parts in parallel and `atomicAdd(+1)` on `part.mesh.refcount` and `part.hull.refcount` if non-null.
- **`next->nitems`**: incremented atomically (thread 0) only for successful cuts; caller must pre-zero it and ensure sufficient `items` capacity.
- **Mesh volumes**: after writing the new parts, warp 0 calls `mesh_volume_warp` on the pos part, warp 1 on the neg part; lane 0 of each warp writes to `Part.mesh_vol`.

## beam_hull API

`beam_hull<<<2*current->nitems, 32>>>(current, pool, err)` — one block (one warp) per part; fills `Part.hull` with the convex hull mesh. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x / 2`, `part_off = blockIdx.x % 2` → `part_idx = nparts - 2 + part_off` (0 = second-to-last, 1 = last part).
- **Early exit** (no error): if `item_idx >= nitems`, or `part_idx` out of range, or `part->hull.verts != NULL` (hull already computed).
- **Calls `hull_dandc_warp_mesh`** on `p->mesh.verts` / `p->mesh.nv` using `pool->heap` (output) and `pool->scratch` (scratch). All 32 lanes call.
- **Stores result**: lane 0 writes the returned `Mesh` into `p->hull`. Returns `{NULL,NULL,0,0,NULL}` on error or < 4 points (error code set in `*err`).
- **Hull volume**: if `hull.nt > 0`, all 32 lanes call `mesh_volume_warp(&hull, lane)`; lane 0 writes result to `p->hull_vol`.

## beam_sort API

`beam_sort<<<current->nitems, 32>>>(current, pool, err)` — one block (one warp) per WorkItem; sorts `parts[0..nparts-1]` in ascending order of `max(hausdorff, mesh_vol / (hull_vol + eps))`. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x`. Early-exit if `item_idx >= nitems` or `nparts <= 1`.
- **Sort key**: `part_cost(p) = fmaxf(PART_COST_K_RV * rv, hausdorff)` where `rv = cbrt(3/(4π) × max(hull_vol − mesh_vol, 0))` converts volume concavity to a distance scale. `PART_COST_K_RV = 0.3f` (CoACD default). `PartKeyCmp::key` delegates to `part_cost`. Uses `warp_sort_t<Part, PartKeyCmp>`; sorts `wi->parts` in-place.
- **Scratch**: lane 0 allocates `nparts * sizeof(Part) + WS_MAX_STACK * 2 * sizeof(int)` bytes from `pool->scratch` via `heap_alloc`; freed by lane 0 before return.
- **Error codes**: `BEAM_ERR_SORT_OOM = 0x20000` if scratch allocation fails; `BEAM_ERR_SORT_STACK = 0x40000` if `warp_sort_t` stack overflows (both `atomicOr`'d into `*err`).

## beam_finalize API

`beam_finalize<<<max_keep, 1024>>>(prev, current, pool, finish, err, max_keep, threshold)` — clears `prev` AlgoState and compacts `current` to the best `max_keep` WorkItems. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: each of the `max_keep` blocks clears `prev->items[blockIdx.x]`. Block 0 additionally does all compaction work on `current`.
- **Phase 1 (all blocks)**: parallel over parts (1024 threads strided); `atomicAdd(-1)` on `mesh.refcount` / `hull.refcount` if non-null; `heap_free(&pool->heap, verts)` if old value was 1; thread 0 sets `nparts = 0`.
- **Phase 2a (block 0)**: allocate `WIKey[nitems]` sort buffer + warp_sort scratch from `pool->scratch`; fill with `{part_cost(last_part), idx}` in parallel; warp 0 sorts ascending via `warp_sort_t<WIKey, WIKeyCmp>`; free sort scratch.
- **Phase 2b**: allocate `max_keep × sizeof(WorkItem)` scratch from `pool->scratch`. Each warp moves one top-k source item (`current->items[sorted_idx]`) into scratch: lane 0 writes `dst->nparts`, `__syncwarp()`, all lanes copy parts, lane 0 sets `src->nparts = 0`, `__syncwarp()`.
- **Phase 2c**: thread 0 checks `keys[0].cost < threshold`; sets `*finish = 1` if so. Frees key buffer.
- **Phase 2d**: each warp clears one item in `current` (all nitems, strided). Moved items have `nparts = 0` — inner loop is a no-op. Non-moved items: `atomicAdd(-1)` on refcounts, `heap_free` if old == 1; lane 0 sets `nparts = 0`, `__syncwarp()`.
- **Phase 2e**: each warp moves one scratch item back to `current->items[0..k-1]`: lane 0 writes `dst->nparts`, `__syncwarp()`, all lanes copy parts, `__syncwarp()`.
- **Phase 2f**: thread 0 frees WorkItem scratch; sets `prev->nitems = 0`, `current->nitems = k` where `k = min(max_keep, nitems)`.
- **`part_cost` helper**: `static __device__ inline float part_cost(const Part&)` defined in `beam.cu`; returns `fmaxf(PART_COST_K_RV * rv, hausdorff)` where `rv = cbrtf(3/(4π) × max(hull_vol − mesh_vol, 0))`. `#define PART_COST_K_RV 0.3f` (CoACD default). Used by `PartKeyCmp` and `beam_finalize`'s key-fill phase.
- **`WIKey` / `WIKeyCmp`**: sort struct `{float cost; int idx}`, ascending by cost, ties broken by index. `sentinel = {1e30f, 0x7fffffff}`.
- **Error codes**: `BEAM_ERR_FINALIZE_OOM = 0x80000` on scratch OOM; `BEAM_ERR_SORT_STACK` reused for WIKey sort stack overflow. All `atomicOr`'d into `*err`; scratch freed before early return.
- **`__syncwarp()` discipline**: every `if (wlane == 0)` write to global memory is immediately followed by `__syncwarp()` to make the write visible to all lanes before proceeding.
- **Debug printf**: DPRINTF in phase 2c logs (1) summary line with `nitems, k, best_cost, threshold` and last part's volumes, (2) per-part detail lines with `mesh_vol, hull_vol, hausdorff, rv, cost` for all parts in the best WorkItem. Gated on `COACD_BEAM_DEBUG` compile flag.

## beam_decompose (csrc/beam.c)

`beam_decompose(ctx, verts, nv, tris, nt, hull_verts, hull_nv, hull_tris, hull_nt, max_iters, cuts_per_axis, threshold, max_keep, verbose, debug, out)` — host-side full beam-search convex decomposition loop.

- **Inputs**: mesh (verts/tris) and its precomputed convex hull (hull_verts/hull_tris) as host pointers; hyperparams; `verbose` flag; `debug` flag.
- **Allocations**: all device buffers allocated with `cuMemAllocAsync` / freed with `cuMemFreeAsync` on a dedicated `CUstream` (created with `cuStreamCreate`, NOT NULL stream — async pool allocations on NULL stream have ordering issues with subsequent kernels). Double-buffered WorkItem arrays: `d_wi_a`/`d_wi_b` each holding `3*cuts_per_axis*max_keep` WorkItems.
- **Loop**: `beam_finalize → (sync + read finish/err/nitems) → beam_expansion → [swap buffers] → beam_hull → beam_sort`. No sync between expansion, hull, and sort — they pipeline in stream order. Sync only after finalize (to read finish/err/nitems via pinned-memory async D2H copies).
- **Debug mode** (`debug != 0`): inserts `cuStreamSynchronize` + error readback after each kernel (expansion, hull, sort), printing per-stage status to stderr. Useful for isolating which kernel crashes.
- **Pinned host memory**: `h_finish`, `h_err`, `h_st` are allocated via `cuMemAllocHost` so `cuMemcpyDtoHAsync` is truly async. Without pinned memory, the driver API falls back to synchronous copies to pageable memory, hiding GPU time from the sync measurement.
- **Timing** (`verbose != 0`): prints per-iteration total time, sync wait time, and final readback time to stderr via `clock_gettime(CLOCK_MONOTONIC)`.
- **Output**: `beam_result` with `nparts` parts; each `beam_part_result` has malloc'd `verts`/`tris` + scalar volumes. Free with `beam_result_free`.
- **Readback** (`decomp_read_result`): reads AlgoState + WorkItem synchronously (large struct), then issues all per-part D2H copies async, syncs once at end.
- **Python binding** (`beam_module.c → py_decompose`): returns a list of `(verts_bytes, tris_bytes, nv, nt, mesh_vol, hull_vol)` tuples. `verbose` and `debug` are optional keyword args (default 0).

## hull_dandc_warp_mesh API

`hull_dandc_warp_mesh(pts, n, lane, heap, scratch_heap, err) -> Mesh` — warp device function (all 32 lanes call with identical args).

- **Output**: returns a `Mesh` struct directly. Allocates a single combined chunk from `DeviceHeap* heap`. Returns `{NULL,NULL,0,0,NULL}` on error or n<4.
- **Heap chunk layout**: `[verts (nv*3 floats, 16-byte aligned) | tris (nt*3 ints, 16-byte aligned) | refcount(16B)]`; exact sizes from a count pass. `Mesh.refcount` points to the trailing `int`, initialized to 1.
- **No volume**: `bt_computeVolume` is not called; the function only produces the mesh.
- **Scratch**: `DeviceHeap* scratch_heap` backs (a) the `WarpPool` (allocated as a single heap chunk via `dandc_scratch_bytes(n)`) and (b) `BtEdge` pool slabs (`BTPOOL_BLOCK_SIZE=8192` edges each, 2 initial + dynamic expansion). All scratch is heap-freed before return — scratch_heap is clean after call.
- **WarpPool**: declared `__shared__`; backing allocated from scratch_heap by lane 0. Used for BtPoint32 array, vertex block, sort scratch, D&C stack, BFS queues (all rewound when done).
- **BtPool (edge pool)**: starts with 2 slabs (16384 edges); expands one slab at a time via `heap_alloc(scratch_heap, ...)` when exhausted. Lane 0 only; free-list setup is serial. Up to `BTPOOL_MAX_BLOCKS=32` slabs tracked for cleanup.
- **Two-pass mesh extraction**: count pass (`bt_extractMesh` with NULL buffers, counts nv/nt via fan formula) → `heap_alloc(heap, ...)` for exact output → extract pass (writes verts+tris). BFS queue rewound between passes.
- **dandc_scratch_bytes**: no longer includes the `6*n*sizeof(BtEdge)` edge pool term (pool now comes from scratch_heap separately).
- **Call sites**: `test_hull_dandc.cu` (kernel) and `test_beam.c` (host launcher) pass `DeviceHeap*` pointers into the embedded heaps of `DevicePool`.

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

## Heap Arena Allocator (allocator.cuh)

`DeviceHeap` is a large-object heap embedded directly in `DevicePool`. Design:

- **`HEAP_NUM_ARENAS` arenas** (default 64, overridable via `COACD_GPU_ARENAS=N pip install -e .`), selected by `blockIdx.x % HEAP_NUM_ARENAS`. Each arena has **64 sub-bin doubly-linked free lists**, a **64-bit occupancy bitmap**, and a spin-lock.
- **Sub-bins**: 32 pow-of-2 bins × 2 linear halves. Bin b = `[512·2^b, 1024·2^b)`. Sub-bin 2b = lower half `[512·2^b, 768·2^b)`, sub-bin 2b+1 = upper half `[768·2^b, 1024·2^b)`. Minimum alignment: `HEAP_ALIGN = 512` bytes.
- **Block layout**: `[HeapBlockHdr (16 B)][data (data_size B)][HeapBlockFtr (16 B)]`. `HeapBlockHdr`/`Ftr` store `data_size`, `arena_idx`, `is_free`. Free blocks store `prev`/`next` pointers in first 16 bytes of data.
- **Slab layout**: each new pool slab has `[leading sentinel 32B][free block][trailing sentinel 32B]`. Sentinels (`is_free=0`, `data_size=0`) prevent coalescing across slab boundaries.
- **Allocation** (thread 0 only): bitmap search for lowest eligible sub-bin (O(1) via `__ffsll`); pop head; split remainder if ≥ `HEAP_HDR_SIZE + HEAP_ALIGN + HEAP_FTR_SIZE`. If no sub-bin has a free block, allocate a new slab from the pool (min 128 KB or next-pow-2).
- **Free** (thread 0 only): inspect prev footer and next header; coalesce adjacent free blocks via doubly-linked O(1) removal; insert merged block into correct sub-bin. Arena for insertion = block's stored `arena_idx` (all blocks in a slab share the same arena, so coalescing is always intra-arena).
- **No compact**: `beam_heap_compact()` is a no-op — coalescing is handled in-place by `heap_free`. Pool remains stable across all calls.

**DevicePool layout** (must match `csrc/structs.h`):
```
DevicePool { base, offset*, capacity, DeviceHeap heap, DeviceHeap scratch }
DeviceHeap { DevicePool* pool (back-ptr set by heap_init_kernel), HeapArena arenas[HEAP_NUM_ARENAS] }
HeapArena  { bitmap, lock, _pad, heads[64], tails[64] }  // 1040 bytes each
```
`sizeof(DevicePool)` ≈ 133 KB at default HEAP_NUM_ARENAS=64 (bulk is the two embedded heap arrays).

Host init: allocate one `d_pool_struct` of `sizeof(struct DevicePool)`, set `base`/`offset`/`capacity`, then launch `heap_init_kernel<<<2*HEAP_NUM_ARENAS,32>>>` which sets `heap.pool = scratch.pool = pool` and zeroes all arena state.

Benchmark results (100 hulls × 2000 pts/hull, gaussian, HEAP_NUM_ARENAS=64):
- **Pool stable at ~172 MB** after first call regardless of compact frequency (compact is a no-op).
- In the old allocator, compact-every-call caused pool to grow to ~5700 MB; this is fully resolved.
- Arena count sweep (`docs/arena_sweep.md`): pool scales ~linearly with HEAP_NUM_ARENAS; throughput is unaffected. A=32 saves ~10% pool with no performance cost.

## Implementation Notes

### DevicePool offset/capacity are 64-bit

`DevicePool.offset` and `DevicePool.capacity` are `unsigned long long`. Previously 32-bit, causing silent wraparound with >4GB scratch → overlapping memory regions → corrupted D&C edge pool free lists → illegal memory access.

### pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

### nvcc Crashes with --generate-line-info and Deep Recursion

nvcc crashed when compiling with `--generate-line-info` while `bt_computeInternal` was recursive. Converting to an iterative explicit stack resolved the crash. Line info is now always enabled for NCU profiling.

### plane_cut Loop Reconstruction: be_used / loop_starts Aliasing Bug

In `plane_cut_block` phase 10, `be_used` was aliased to `loop_starts` to avoid an extra allocation. The code did:

```cuda
loop_starts[n_loops] = lvi;   // store start index
be_used[start] = 1;           // start == n_loops → overwrites loop_starts[n_loops]!
```

Because `start == n_loops` at the beginning of each outer loop iteration, `be_used[start] = 1` immediately clobbered the stored start value, making every loop appear to start one vertex late (size N-1 instead of N). Fix: save `lvi_start` in a local variable and assign `loop_starts[n_loops] = lvi_start` **after** all `be_used` writes complete.

### plane_cut cap_ptr Buffer Too Small (Bridge Swap Overflow)

`cap_ptr` (cap_tris) was allocated as `n_boundary * 3 * sizeof(int)` but is used as both:
1. A temporary swap buffer during hole-bridge insertion (max size = `poly_n + hs + 2` ≤ `n_boundary*4+64`)
2. The ear-clip output triangle buffer (max `n_cap` triangles × 3 ints)

Use-1 can exceed `n_boundary * 3` when the merged polygon grows during bridging, overwriting adjacent scratch heap block headers → corrupted `arena_idx` → OOB arena lock in `heap_free`. Fix: allocate `cap_ptr` as `(n_boundary*4+64) * sizeof(int)` to match `poly_ptr`.

### plane_cut Phase 10: lv Buffer Overflow in Loop Reconstruction

`lv` (loop vertex sequence) was allocated with `n_boundary` elements. When boundary edges don't form clean closed loops (e.g., non-manifold geometry, safety counter exhaustion), the while loop in phase 10 could write past the buffer end. Writing one element beyond `lv` corrupts the adjacent heap block's header, causing cascading failures: corrupted output meshes with -1 vertex indices → OOB `all_verts` accesses → misaligned address errors (CUDA error 716) in subsequent iterations. Fix: guard `lvi >= n_boundary` before each `lv[lvi++]` write.

### CheckedBuf<T> — Debug-Mode Bounds-Checked Buffers

`CheckedBuf<T>` in `common.cuh` wraps `T*` + element count. Enabled by `COACD_BEAM_DEBUG`:
- `operator[]` bounds-checks and prints `[OOB] name: idx=N count=M blk=B tid=T`, clamping to element 0 to prevent faults.
- `slice(offset, len)` returns a sub-buffer (also bounds-checked in debug).
- `raw()` returns the underlying pointer (for passing to sort, `pc_signed_area`, etc.).
- Release mode: struct contains only `T* ptr_`, all methods inline to raw access (zero overhead).
- `PC_BUF(type, name, ptr, count)` convenience macro constructs a `CheckedBuf` with `#name` as the label.

All scratch buffers in `plane_cut_block` are wrapped with `CheckedBuf` for OOB detection.

### Using compute-sanitizer for CUDA Memory Errors

When a CUDA kernel crashes with error 716 (misaligned address) or 700 (illegal memory access), use `compute-sanitizer --tool memcheck` to find the exact source location:

```bash
compute-sanitizer --tool memcheck python <script.py>
```

Note: compute-sanitizer serializes GPU execution and can change timing/behavior (e.g. algorithms may exit early or produce different results). Use it to find the source of crashes, not to validate correctness.

## Benchmarking

```bash
# Memory management benchmark: compact overhead, pool usage, arena effects
python tests/bench_mm.py [--n_rounds 100] [--config 100x2000pts]

# Configs (same range as test_hull, ordered fewer-large → more-small, gaussian only):
#   10x20000pts, 100x2000pts, 1000x200pts, 10000x20pts

# Arena count sweep: rebuild + benchmark for HEAP_NUM_ARENAS ∈ {32,64,128,256}
python tests/bench_arena_sweep.py [--arenas 32,64,128,256] [--output docs/arena_sweep.md]

# D&C hull benchmark for NCU profiling
python tests/bench_dandc.py --n_pts 200 --n_hulls 8
ncu --set full -o dandc_profile python tests/bench_dandc.py --n_pts 200 --n_hulls 8
```

## Current Status

### Working
- D&C hull volume (`batch_hull_volume`) and mesh extraction (`batch_hull_dandc_mesh`) — all 103 tests pass
- Batch mesh volume (`batch_mesh_volume`) — divergence theorem, watertight meshes
- Warp sort (`test_warp_sort`) — bitonic + quicksort paths, duplicates
- Plane cut (`test_plane_cut`) — simple loop, ring, multi-hole, edge cases (14 tests)
- Persistent heaps (`beam_init` with `pool_bytes`): both heaps share one pool, all memory recycled by kernels, pool stable after first call
- `ctx.pool_usage()` — peak device pool bytes (monotonic), `ctx.heap_compact()` — available but not needed normally
- `beam_decompose` (cube, lshape, octocat): cube → 1 part, lshape → 2 parts, octocat → ~14 parts ✓; test prints volume comparison table (GPU vs trimesh vs scipy)

### Not Yet Implemented
- `__cuda_array_interface__` support for GPU tensor input
