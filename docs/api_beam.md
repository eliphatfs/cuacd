# Beam Search Kernel APIs

## beam_initialize

`beam_initialize<<<3, 32>>>(verts, tris, nv, nt, hull_verts, hull_tris, hull_nv, hull_nt, current)` -- seeds `current` with a single WorkItem containing one Part from the input mesh and its precomputed convex hull.

- **Block 0**: all 32 lanes call `mesh_volume_warp` on the input mesh; lane 0 writes result to `current->items[0].parts[0].mesh_vol`.
- **Block 1**: all 32 lanes call `mesh_volume_warp` on the hull mesh; lane 0 writes result to `current->items[0].parts[0].hull_vol`.
- **Block 2**: thread 0 sets `parts[0].mesh` and `parts[0].hull` to point directly at the caller-supplied device pointers (both `refcount = NULL` -- input arrays are not heap-allocated and will not interact with `heap_free`); sets `hausdorff = 0`; sets `nparts = 1`; sets `nitems = 1`.
- **No error argument**: initialization cannot fail; volumes are computed independently of allocation.
- **Block ordering**: blocks 0/1 build a local `Mesh` from kernel parameters directly (not reading from `current->items`) so there is no data dependency on block 2's writes. All three blocks write disjoint fields of `parts[0]`.

## beam_expansion

`beam_expansion<<<3*cuts_per_axis*current->nitems, 64>>>(current, next, pool, cuts_per_axis, err)` -- one block per (item, cut); produces candidate WorkItems in `next`. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x / (3*cuts_per_axis)`, `axis = (blockIdx.x % (3*cuts_per_axis)) / cuts_per_axis`, `slice = ... % cuts_per_axis`.
- **Plane**: axis-aligned at `(slice+1)/(cuts_per_axis+1)` fraction of the last part's bounding box. Bbox via three-stage reduction: per-thread local min/max -> `warp_min_f`/`warp_max_f` -> lane-0-per-warp `atomicMinF`/`atomicMaxF` into shared memory.
- **Calls `plane_cut_block`** on `wi->parts[nparts-1].mesh` using `pool->heap` and `pool->scratch`.
- **Empty side**: if either `pos.mesh.nv == 0` or `neg.mesh.nv == 0`, frees both pos and neg heap chunks independently and returns -- no entry written to `next`.
- **Overflow**: `BEAM_ERR_OVERFLOW = 0x10000` -- `atomicOr`'d into `err` if `nparts+1 > WORK_ITEM_MAX_PARTS`. Distinct from any plane_cut error code (those are small integers).
- **Part copy**: bulk int-copy of `parts[0..nparts-2]` in parallel; thread 0 appends `pp.pos` and `pp.neg`, sets `nparts = old_nparts + 1`. After the copy, threads iterate over the `nparts-1` copied parts in parallel and `atomicAdd(+1)` on `part.mesh.refcount` and `part.hull.refcount` if non-null.
- **`next->nitems`**: incremented atomically (thread 0) only for successful cuts; caller must pre-zero it and ensure sufficient `items` capacity.
- **Mesh volumes**: after writing the new parts, warp 0 calls `mesh_volume_warp` on the pos part, warp 1 on the neg part; lane 0 of each warp writes to `Part.mesh_vol`.

## beam_hull

`beam_hull<<<2*current->nitems, 32>>>(current, pool, err)` -- one block (one warp) per part; fills `Part.hull` with the convex hull mesh. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x / 2`, `part_off = blockIdx.x % 2` -> `part_idx = nparts - 2 + part_off` (0 = second-to-last, 1 = last part).
- **Early exit** (no error): if `item_idx >= nitems`, or `part_idx` out of range, or `part->hull.verts != NULL` (hull already computed).
- **Calls `hull_dandc_warp_mesh`** on `p->mesh.verts` / `p->mesh.nv` using `pool->heap` (output) and `pool->scratch` (scratch). All 32 lanes call.
- **Stores result**: lane 0 writes the returned `Mesh` into `p->hull`. Returns `{NULL,NULL,0,0,NULL}` on error or < 4 points (error code set in `*err`).
- **Hull volume**: if `hull.nt > 0`, all 32 lanes call `mesh_volume_warp(&hull, lane)`; lane 0 writes result to `p->hull_vol`.

## beam_sort

`beam_sort<<<current->nitems, 32>>>(current, pool, err)` -- one block (one warp) per WorkItem; sorts `parts[0..nparts-1]` in ascending order of `max(hausdorff, mesh_vol / (hull_vol + eps))`. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: `item_idx = blockIdx.x`. Early-exit if `item_idx >= nitems` or `nparts <= 1`.
- **Sort key**: `part_cost(p) = fmaxf(PART_COST_K_RV * rv, hausdorff)` where `rv = cbrt(3/(4pi) * max(hull_vol - mesh_vol, 0))` converts volume concavity to a distance scale. `PART_COST_K_RV = 0.3f` (CoACD default). `PartKeyCmp::key` delegates to `part_cost`. Uses `warp_sort_t<Part, PartKeyCmp>`; sorts `wi->parts` in-place.
- **Scratch**: lane 0 allocates `nparts * sizeof(Part) + WS_MAX_STACK * 2 * sizeof(int)` bytes from `pool->scratch` via `heap_alloc`; freed by lane 0 before return.
- **Error codes**: `BEAM_ERR_SORT_OOM = 0x20000` if scratch allocation fails; `BEAM_ERR_SORT_STACK = 0x40000` if `warp_sort_t` stack overflows (both `atomicOr`'d into `*err`).

## beam_finalize

`beam_finalize<<<max_keep, 1024>>>(prev, current, pool, finish, err, max_keep, threshold)` -- clears `prev` AlgoState and compacts `current` to the best `max_keep` WorkItems. Returns immediately if `*err` is non-zero on entry.

- **Block mapping**: each of the `max_keep` blocks clears `prev->items[blockIdx.x]`. Block 0 additionally does all compaction work on `current`.
- **Phase 1 (all blocks)**: parallel over parts (1024 threads strided); `atomicAdd(-1)` on `mesh.refcount` / `hull.refcount` if non-null; `heap_free(&pool->heap, verts)` if old value was 1; thread 0 sets `nparts = 0`.
- **Phase 2a (block 0)**: allocate `WIKey[nitems]` sort buffer + warp_sort scratch from `pool->scratch`; fill with `{part_cost(last_part), idx}` in parallel; warp 0 sorts ascending via `warp_sort_t<WIKey, WIKeyCmp>`; free sort scratch.
- **Phase 2b**: allocate `max_keep * sizeof(WorkItem)` scratch from `pool->scratch`. Each warp moves one top-k source item (`current->items[sorted_idx]`) into scratch: lane 0 writes `dst->nparts`, `__syncwarp()`, all lanes copy parts, lane 0 sets `src->nparts = 0`, `__syncwarp()`.
- **Phase 2c**: thread 0 checks `keys[0].cost < threshold`; sets `*finish = 1` if so. Frees key buffer.
- **Phase 2d**: each warp clears one item in `current` (all nitems, strided). Moved items have `nparts = 0` -- inner loop is a no-op. Non-moved items: `atomicAdd(-1)` on refcounts, `heap_free` if old == 1; lane 0 sets `nparts = 0`, `__syncwarp()`.
- **Phase 2e**: each warp moves one scratch item back to `current->items[0..k-1]`: lane 0 writes `dst->nparts`, `__syncwarp()`, all lanes copy parts, `__syncwarp()`.
- **Phase 2f**: thread 0 frees WorkItem scratch; sets `prev->nitems = 0`, `current->nitems = k` where `k = min(max_keep, nitems)`.
- **`part_cost` helper**: `static __device__ inline float part_cost(const Part&)` defined in `beam.cu`; returns `fmaxf(PART_COST_K_RV * rv, hausdorff)` where `rv = cbrtf(3/(4pi) * max(hull_vol - mesh_vol, 0))`. `#define PART_COST_K_RV 0.3f` (CoACD default). Used by `PartKeyCmp` and `beam_finalize`'s key-fill phase.
- **`WIKey` / `WIKeyCmp`**: sort struct `{float cost; int idx}`, ascending by cost, ties broken by index. `sentinel = {1e30f, 0x7fffffff}`.
- **Error codes**: `BEAM_ERR_FINALIZE_OOM = 0x80000` on scratch OOM; `BEAM_ERR_SORT_STACK` reused for WIKey sort stack overflow. All `atomicOr`'d into `*err`; scratch freed before early return.
- **`__syncwarp()` discipline**: every `if (wlane == 0)` write to global memory is immediately followed by `__syncwarp()` to make the write visible to all lanes before proceeding.
- **Debug printf**: DPRINTF in phase 2c logs (1) summary line with `nitems, k, best_cost, threshold` and last part's volumes, (2) per-part detail lines with `mesh_vol, hull_vol, hausdorff, rv, cost` for all parts in the best WorkItem. Gated on `COACD_BEAM_DEBUG` compile flag.

## beam_decompose (csrc/beam.c)

`beam_decompose(ctx, verts, nv, tris, nt, hull_verts, hull_nv, hull_tris, hull_nt, max_iters, cuts_per_axis, threshold, max_keep, verbose, debug, out)` -- host-side full beam-search convex decomposition loop.

- **Inputs**: mesh (verts/tris) and its precomputed convex hull (hull_verts/hull_tris) as host pointers; hyperparams; `verbose` flag; `debug` flag.
- **Allocations**: all device buffers allocated with `cuMemAllocAsync` / freed with `cuMemFreeAsync` on a dedicated `CUstream` (created with `cuStreamCreate`, NOT NULL stream -- async pool allocations on NULL stream have ordering issues with subsequent kernels). Double-buffered WorkItem arrays: `d_wi_a`/`d_wi_b` each holding `3*cuts_per_axis*max_keep` WorkItems.
- **Loop**: `beam_finalize -> (sync + read finish/err/nitems) -> beam_expansion -> [swap buffers] -> beam_hull -> beam_sort`. No sync between expansion, hull, and sort -- they pipeline in stream order. Sync only after finalize (to read finish/err/nitems via pinned-memory async D2H copies).
- **Debug mode** (`debug != 0`): inserts `cuStreamSynchronize` + error readback after each kernel (expansion, hull, sort), printing per-stage status and pool usage (MB) to stderr. Also prints pool usage after finalize. Useful for isolating which kernel crashes and tracking memory growth.
- **Pinned host memory**: `h_finish`, `h_err`, `h_st` are allocated via `cuMemAllocHost` so `cuMemcpyDtoHAsync` is truly async. Without pinned memory, the driver API falls back to synchronous copies to pageable memory, hiding GPU time from the sync measurement.
- **Timing** (`verbose != 0`): prints per-iteration total time, sync wait time, and final readback time to stderr via `clock_gettime(CLOCK_MONOTONIC)`.
- **Output**: `beam_result` with `nparts` parts; each `beam_part_result` has malloc'd `verts`/`tris` + scalar volumes. Free with `beam_result_free`.
- **Readback** (`decomp_read_result`): reads AlgoState + WorkItem synchronously (large struct), then issues all per-part D2H copies async, syncs once at end.
- **Python binding** (`beam_module.c -> py_decompose`): returns a list of `(verts_bytes, tris_bytes, nv, nt, mesh_vol, hull_vol)` tuples. `verbose` and `debug` are optional keyword args (default 0).
