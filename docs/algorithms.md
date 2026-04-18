# Algorithm Details

## D&C Convex Hull (`hull_dandc.cuh`)

Warp-level parallel tree merge — 16 primary lanes each build a small hull (BT_HULL_GROUPS=16), then 4 rounds of pairwise merging using 16×2 thread pairs. Within each merge, both threads in a pair call `bt_findMaxAngle` convergedly (primary for c0/h0, secondary for c1/h1); results are exchanged via warp shuffle. All `BtVertex*` pointer fields (`next`, `prev`, `BtEdge::target`, `BtIntermediateHull` extremals) are stored as `BtVIndex` (int index into the shared `vblock` array) to reduce memory and improve locality.

Memory layout in `hull_dandc_warp_mesh`: presort `BtPoint32` array is heap-allocated from `scratch_heap` (not WarpPool) and freed immediately after the vertex-init copy in postsort, before the D&C phase. Edge pools (BtPool + initial slabs) exist only for BT_HULL_GROUPS=16 primaries — secondaries never allocate edges, so `edgePool.blocks` is indexed by `group` (not `lane`). WarpPool backing covers only sort scratch + postsort persistent data + D&C stacks/BFS queues. See `docs/api_hull_dandc.md`.

## Plane Cut (`plane_cut.cuh`)

Block-level (64 threads) mesh splitting along an arbitrary plane. Produces `PartPair` with pos/neg meshes. Handles disjoint components: when the plane doesn't intersect any edge but vertices exist on both sides, triangles are separated by vertex sign with per-side vertex compaction. Cap triangulation handles multiple boundary loops with arbitrary nesting depth via parity classification (even depth = outer, odd depth = hole). Cap failure detection: if loop reconstruction overflows the vertex buffer, a loop chain breaks (incomplete loop), the parent-tree traversal produces a cycle (ambiguous nesting), or hole bridging fails (no visible edge), the cap is treated as failed and the mesh is returned unsplit — so the lookahead tree search treats it as a failed cut rather than producing non-watertight parts. See `docs/api_plane_cut.md`.

## Hull with Extreme-Point Prefilter (`kdop_hull.cuh`)

Single-warp (32 threads). Fast path: direct D&C hull for nv ≤ 1024. Main path: warp argmax/argmin over 40 icosphere axes finds up to 80 extreme vertices → D&C rough inner hull → ballot/popcount half-space filter discards interior points → D&C final hull of survivors. Result is the exact convex hull. Exposed as `ctx.batch_kdop_hull_mesh()`. Used by `la_hull` kernel. See `docs/api_kdop_hull.md`.

## Hausdorff Distance (`hausdorff.cuh`)

Block-level (256 threads, 8 warps). Computes bidirectional Hausdorff distance between two meshes via sampling + linear BVH. CoACD-matching area-proportional sampling with Wang hash pseudo-random barycentric coordinates. Brute-force path for ≤64 target triangles; linear BVH (Karras 2012 radix tree) for larger meshes with cooperative 4-warp Morton code sort. Used by `la_hausdorff_parts` kernel to fill `Part.hausdorff`.

## Lookahead Search Decomposition

Source files: `la_expand.cu` + `la_refine.cu` + `la_lifecycle.cu` + `la_concave.cu` + `csrc/lookahead.c`.

Maintains a flat decomposition (LaDecompState). For each part above threshold, explores a shallow tree of candidate cuts: `depth` full expansion levels (`width` cuts at level 0, `width2` at deeper levels), then `quick_depth` levels (1 best-axis midpoint cut each). Path cost = average worst-part cost across levels; cut selection = minimum path cost per initial cut. Uses rv-only cost for tree exploration (matching CoACD), but full cost `max(rv, hausdorff)` for the stopping criterion. Default depth=2, quick_depth=0, width=60, width2=5.

### Concave Edge Sampling (`la_concave.cu`)

Optional first-layer expansion supplement. When `n_concave_edges > 0`, the `la_find_concave_edges` kernel (<<<n_cutting, 32>>>) processes each cutting part's mesh: sorts directed edges via `warp_sort_t`, scans for shared edges (consecutive pairs with same vertex key), computes dihedral angles, and reservoir-samples up to `n_concave_edges` concave edges. For each sampled edge, generates up to 4 cutting planes: two face-offset planes and two bisector planes (bisector skipped when `||n1+n2|| < 1e-4`). Total first-layer candidates = `width + n_edge_cuts` (must be < 512). Controlled by `n_concave_edges` (default 32), `concave_eps`, `concave_threshold` (radians, default 3.49 ≈ 200°), `concave_iters` (iterations to apply, default 10).

`la_evaluate` falls back to level-0 items when no leaf descendants exist for a cut (all deeper expansions produced empty halves because pieces were too small to cut further — such pieces are provably below threshold).

`la_count_cutting` deterministically selects highest-cost parts. `la_expand`/`la_expand_quick` place cuts evenly in the valid range `[lo+min_edge_dist, hi-min_edge_dist]` (min_edge_dist = min(threshold/4, 0.015)) to prevent degenerate thin slivers. Parts too small to cut in all axes (extent ≤ 2*min_edge_dist) are recorded as extra_leaves with zero cost for remaining levels — distinguishes genuinely solved parts from failed cuts (which get infinite cost).

Kernels: la_initialize, la_sort_parts, la_hausdorff_parts, la_count_cutting, la_seed_tree, la_expand, la_expand_quick, la_hull, la_sort_items, la_record_level_cost, la_evaluate, la_apply_cuts, la_hull_decomp, la_cleanup_tree, la_free_decomp, la_decompose_components.

## Connected-Components Decomposition (`postprocess.cuh`)

Block-level (128 threads) post-processing pass. Uses lock-free rank-based union-find over triangle adjacency to identify connected components in each decomposition part. Read-only find (no path compression) for GPU performance; rank-based union with CAS for lock-free merging. Rank bump guards against concurrent re-rooting (winner must still be a root/self-pointing before its rank is incremented). When a part has multiple components, allocates new mesh data per component from the main heap and appends extra parts to LaDecompState via atomicAdd on nparts; computes mesh_vol per component via `mesh_volume_warp`. Controlled by `decompose_components` parameter (default False in Python API, True in CLI). `no_decompose_components_per_iter` (default False = per-iter enabled) controls whether decompose-components runs each iteration (before evaluation and after applying cuts) instead of only at the end. Inherits hausdorff distance as upper bound; hulls are recomputed via la_hull_decomp after splitting.

## Utility Functions

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

## Memory Architecture

One `DevicePool` with bump allocator backs two embedded `DeviceHeap` instances (`heap` for output, `scratch` for temporaries). Both heaps use arena-based free lists with O(1) coalescing — see `docs/api_heap_allocator.md`. Pool offset never decreases; freed blocks are recycled via free-lists. Pool stabilizes after first call.

### Pool Allocator Pattern

**Critical**: only thread 0 calls `pool_alloc()`, stores pointer in `__shared__`, then all threads read after `__syncthreads()`. If all threads call `pool_alloc`, each gets a different offset → data corruption.
