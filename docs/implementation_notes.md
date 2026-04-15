# Implementation Notes

## Avoid Local Arrays with Runtime-Variable Indices in Warp Code

In `hull_dandc_warp_mesh`, point conversion originally used a local `float p[3]` array indexed by runtime variables (`medAx`, `maxAx`, `minAx`). Because the compiler cannot prove the indices are compile-time constants, it cannot keep `p` in registers and must spill to local (per-thread stack) memory, adding load/store traffic. Fix: unroll to three direct inline expressions, eliminating the array entirely.

## DevicePool offset/capacity are 64-bit

`DevicePool.offset` and `DevicePool.capacity` are `unsigned long long`. Previously 32-bit, causing silent wraparound with >4GB scratch -> overlapping memory regions -> corrupted D&C edge pool free lists -> illegal memory access.

## pyproject.toml license Field Format

PEP 621 requires `license = {text = "MIT"}` or `license = {file = "LICENSE"}`. The bare string form `license = "MIT"` fails `setuptools` validation and prevents `build_ext` from running.

## nvcc Crashes with --generate-line-info and Deep Recursion

nvcc crashed when compiling with `--generate-line-info` while `bt_computeInternal` was recursive. Converting to an iterative explicit stack resolved the crash. Line info is now always enabled for NCU profiling.

## plane_cut Loop Reconstruction: be_used / loop_starts Aliasing Bug

In `plane_cut_block` phase 10, `be_used` was aliased to `loop_starts` to avoid an extra allocation. The code did:

```cuda
loop_starts[n_loops] = lvi;   // store start index
be_used[start] = 1;           // start == n_loops -> overwrites loop_starts[n_loops]!
```

Because `start == n_loops` at the beginning of each outer loop iteration, `be_used[start] = 1` immediately clobbered the stored start value, making every loop appear to start one vertex late (size N-1 instead of N). Fix: save `lvi_start` in a local variable and assign `loop_starts[n_loops] = lvi_start` **after** all `be_used` writes complete.

## plane_cut cap_ptr Buffer Too Small (Bridge Swap Overflow)

`cap_ptr` (cap_tris) was allocated as `n_boundary * 3 * sizeof(int)` but is used as both:
1. A temporary swap buffer during hole-bridge insertion (max size = `poly_n + hs + 2` <= `n_boundary*4+64`)
2. The ear-clip output triangle buffer (max `n_cap` triangles x 3 ints)

Use-1 can exceed `n_boundary * 3` when the merged polygon grows during bridging, overwriting adjacent scratch heap block headers -> corrupted `arena_idx` -> OOB arena lock in `heap_free`. Fix: allocate `cap_ptr` as `(n_boundary*4+64) * sizeof(int)` to match `poly_ptr`.

## plane_cut Phase 10: lv Buffer Overflow in Loop Reconstruction

`lv` (loop vertex sequence) was allocated with `n_boundary` elements. When boundary edges don't form clean closed loops (e.g., non-manifold geometry, safety counter exhaustion), the while loop in phase 10 could write past the buffer end. Writing one element beyond `lv` corrupts the adjacent heap block's header, causing cascading failures: corrupted output meshes with -1 vertex indices -> OOB `all_verts` accesses -> misaligned address errors (CUDA error 716) in subsequent iterations. Fix: guard `lvi >= n_boundary` before each `lv[lvi++]` write.

## hausdorff BVH: Karras Split Binary Search Off-by-One

The Karras 2012 linear BVH construction finds the split point gamma via a power-of-2 binary search. The original implementation started `t` at `(max_len + 1) >> 1`, which for non-power-of-two `max_len` produces a halving sequence (e.g. 10, 5, 2, 1 for max_len=19) whose subset sums cannot express every integer in [0, max_len]. This caused some internal nodes to never be assigned as children, leaving orphan subtrees with uninitialized AABBs. The traversal then missed entire subtrees, returning inflated Hausdorff distances. Fix: start `t` at the largest power of 2 ≤ max_len (e.g. 16, 8, 4, 2, 1 for max_len=19), guaranteeing full coverage.

## hausdorff BVH: Heap Free-List Overlap on Sample Buffers

Four separate `heap_alloc` calls for `s_samples_a`, `s_tri_ids_a`, `s_samples_b`, `s_tri_ids_b` could return overlapping regions when the free-list had recently freed blocks of similar size. Observed: last 64 bytes of `s_samples_a` aliased first 64 bytes of `s_samples_b`, corrupting ~5 sample positions. This silently corrupted Morton codes and BVH leaf placement, causing the B→A traversal to miss correct triangles. Fix: allocate all four arrays as a single contiguous `heap_alloc` block and compute sub-pointers manually.

## warp_sort PartKeyCmp: Equal-Cost Parts Cause Refcount Corruption

`warp_partition` fills the "equal-to-pivot" region by writing the pivot value into every equal-key slot: `data[mid_lo + i] = pivot`. When `PartKeyCmp::cmp` returned 0 for two different Parts with the same cost, the sort duplicated one Part struct over multiple array slots — including its `mesh.verts`, `hull.verts`, and `refcount` pointers. Now multiple slots reference the same heap chunk, but expansion only incremented refcounts for the original copy count. When finalize decremented each slot independently, it double-freed the duplicated chunk and leaked the overwritten ones. Manifested as massively negative refcounts (e.g. -1095744947), use-after-free in hull/expansion kernels, and CUDA illegal memory access errors. Fix: tiebreak `PartKeyCmp::cmp` on `mesh.verts` pointer to ensure a total order — every Part has a unique mesh allocation.

## la_expand_quick: __shared__ Memory Aliasing with plane_cut_block's s_result

`la_expand_quick` tried 3 axis-aligned midpoint cuts in a loop, storing the best result in `__shared__ PartPair s_best_pp`. The loop body called `plane_cut_block`, which is inlined and declares `__shared__ PartPair s_result`. The CUDA compiler determined that `s_best_pp` and `s_result` are never simultaneously live (since `s_result` is only used inside `plane_cut_block` and `s_best_pp` is only used between calls), and placed them at the same shared memory offset. When `plane_cut_block` for axis=1 executed `pc_zero_part(&s_result.pos)` at its entry, it also zeroed `s_best_pp`, destroying the best cut from axis=0. Subsequent `mesh_volume_warp` calls then read garbage from the freed/zeroed mesh pointers, causing CUDA error 700 (illegal memory access). Only manifested with enough concurrent blocks (>576) because the compiler only performs this optimization at higher optimization levels with sufficient register pressure.

Fix: replaced `s_best_pp` with per-axis arrays (`s_costs[3]`, `s_vptrs[3][2]`, etc.) that store each axis's result immediately after `plane_cut_block` returns, before the next iteration can overwrite shared memory. After the loop, non-best cuts are freed and the best is reconstructed from the per-axis arrays. This avoids any `__shared__` variable persisting across `plane_cut_block` calls.

**General rule**: Never store data in `__shared__` variables that must survive across calls to inlined `__device__` functions that also use `__shared__` memory. The compiler is free to alias `__shared__` variables with non-overlapping liveness.

## la_expand: Level-0 Items Share Mesh Memory with Leaf Items (Double-Decrement Bug)

`la_expand` copies the output work item (`wo`) to the level-0 buffer (`l0`), creating two references to the same `mesh.verts` device memory (from `plane_cut_block`) but without incrementing the refcount. During cleanup, both `la_cleanup_tree` on `d_cur` (leaf items) and on `d_level0` decrement the same refcount. With initial refcount=1, the first decrement frees the memory, and the second is a double-free. Worse, when `la_apply_cuts` adopts the level-0 item's parts into the decomp and increments the refcount to 2, both cleanup passes decrement it to 0 and free it — leaving the decomp with a dangling pointer. The next `la_hull_decomp` call allocates new hull memory over the freed mesh, corrupting vertex data. Observed as mesh_vol changing from 0.566 to 0.352 between kernel launches.

Fix: (1) In `la_expand`, increment the mesh refcount when writing to the level-0 item (`atomicAdd(l0->parts[np-1].mesh.refcount, 1)` and same for `np`). This accounts for the second reference. (2) In `la_apply_cuts`, null out both hull AND mesh pointers in the adopted level-0 item so that `la_cleanup_tree` doesn't decrement the refcounts for the decomp-owned copies.

## warp_sort_t Corrupts Large Structs (Part)

`warp_sort_t<Part, LAPartKeyCmpRV>` corrupted Part data during sorting. The `warp_partition` step writes the pivot value into all "equal-key" slots, but for large structs like Part (80 bytes), this overwrites the entire struct — including `mesh.verts` and `refcount` pointers — with the pivot's values. This causes duplicate mesh references, double-frees, and NULL pointer dereferences at depth>1 expansion. Fix: replaced `warp_sort_t<Part>` with single-thread insertion sort (adequate for `LA_MAX_PARTS=16` and `LA_MAX_DECOMP=1024`).

## plane_cut Hole-Bridge Overwrites Earlier Outer Loop Cap Triangles

When `plane_cut_block` encounters multiple boundary loops on the cap plane (from cutting genus>0 meshes or meshes with self-overlapping cross-sections), it processes each outer loop and its holes sequentially. The hole-bridging step (which inserts hole vertices into the outer polygon) used `cap_tris` as scratch to rebuild the polygon:

```cuda
for (int j=0; j<=p_pos; j++) cap_tris[k++]=polygon[j];
for (int j=0; j<hs; j++)     cap_tris[k++]=hole[(m_idx+j)%hs];
cap_tris[k++]=hole[m_idx]; cap_tris[k++]=polygon[p_pos];
for (int j=p_pos+1; j<poly_n; j++) cap_tris[k++]=polygon[j];
for (int j=0; j<k; j++) polygon[j]=cap_tris[j];
```

But `cap_tris` is also the output buffer for ear-clip triangles (written at `cap_tris[n_cap*3..]`). When a second outer loop has holes, the bridging writes to `cap_tris[0..]`, overwriting cap triangles already emitted by the first outer loop's ear-clip. The corrupted entries contain vertex indices from the wrong loop, producing non-manifold edges (valence 3) and boundary edges (valence 1) in the output. The triangle count is correct but the content is wrong, so the cap-fail detection (`n_cap < n_cap_expected`) doesn't catch it.

Triggered when: (1) the cut produces 3+ boundary loops, (2) at least 2 are independent outer loops, and (3) at least one outer loop has a hole. Common with genus>0 parts created by cutting through geometrically overlapping body sections (e.g., between an elephant's legs — the positive half has an annular cross-section creating genus-1 topology; a subsequent cut through the annulus produces 3 loops).

Fix: use `ear_prevnext` (which is reinitialized before each ear-clip) as the bridging scratch buffer instead of `cap_tris`. `ear_prevnext` has `poly_cap * 2` ints, more than enough for the bridged polygon.

## plane_cut Loop Classification: Nested Holes (Depth > 1) Misclassified as Holes

The loop classification in `plane_cut_block` phase 10 used a simple rule: any loop with a parent (contained inside another loop) is a hole (`is_hole[i] = parent[i] >= 0`). This fails for loops at nesting depth ≥ 2.

On high-genus meshes (e.g. hero.obj, genus 56), a cut through nested handles produces boundary loops at multiple nesting depths:
- Depth 0: outer loop (cap independently)
- Depth 1: hole inside an outer loop (bridge into parent, then ear-clip)
- Depth 2: island inside a hole — geometrically an independent outer region

The old code classified depth-2 loops as holes with `parent = depth-1 loop`. But the depth-1 loop is itself a hole, so when processing outer loops and collecting "direct children" (`parent[i] == oi`), the depth-2 loop was never collected by any outer loop. It was orphaned — never capped, producing boundary edges (valence 1) on both output halves.

Example: hero.obj cut at y=0.148 produces 4 boundary loops. Loop 0 (47 edges, outer) contains loop 1 (102 edges, hole) which contains loop 3 (17 edges, island). Loop 3 was classified as a hole with parent=1, but loop 1 is a hole — so loop 3 was never capped. Both output halves had 17 boundary edges. Over 100 iterations, the uncapped holes accumulated, producing 549+ parts that never converged.

Fix: after computing parent pointers, walk the parent chain to compute nesting depth. Even depth (0, 2, 4...) = outer loop; odd depth (1, 3, 5...) = hole. Depth-2+ islands get `parent = -1` to be processed as independent outer loops. This matches CDT's flood-fill parity approach (the library CoACD uses), where even-depth triangles are erased and odd-depth are kept.

## plane_cut LVI Overflow and Broken Loop Chains Now Signal Cap Failure

Previously, when the loop vertex buffer (`lv`) overflowed during loop reconstruction (phase 10), the code only printed a diagnostic but continued, producing corrupted cap geometry. Similarly, if a loop chain broke (didn't return to its starting vertex due to safety counter expiration or boundary edge anomalies), the code continued with a partial loop, producing non-watertight caps.

Fix: both conditions now set `s_n_cap = -1` (cap failure), causing `plane_cut_block` to return the mesh unsplit. This ensures the lookahead tree search treats such cuts as failed rather than producing non-watertight parts that corrupt downstream processing.

## plane_cut Hole-Bridge Failure Now Signals Cap Failure

When the hole-bridge step fails to find a visible edge to connect a hole into its parent outer polygon, the hole's boundary edges were previously left uncapped (the code `continue`d to the next hole), producing boundary edges in the output mesh. Fix: bridge failure now sets `s_n_cap = -1` (cap failure), returning the mesh unsplit. This is consistent with the other cap failure modes (cycle detection, buffer overflow, broken loops).

## decompose_components Union-Find: Rank Bump Must Verify Winner Is Still Root

In `dc_union`, after a successful CAS merging loser→winner, the code bumps the winner's rank when `rx == ry`. However, a concurrent thread may have already merged winner into another node Z (changing `parents[winner]` from `pack(r, winner)` to `pack(r, Z)`). The old code unconditionally did `atomicCAS(&parents[winner], vw, pack(r+1, winner))`, which could re-root winner and silently undo the concurrent merge (leaving Z with a dangling child). Fix: check `dc_getId(vw) == winner` (winner is still a self-pointing root) before attempting the rank-bump CAS.

## decompose_components Off-by-One: nparts Overflow Check

`la_decompose_components` checked `s_base_idx + n_comp - 1 > LA_MAX_DECOMP` but the correct check is `>= LA_MAX_DECOMP` (array is indexed 0..LA_MAX_DECOMP-1). The off-by-one allowed writing to `parts[LA_MAX_DECOMP]`, one past the end.

## decompose_components_per_iter: Per-Iteration Connected-Components Splitting

Added `decompose_components_per_iter` parameter to `lookahead_decompose`. When enabled, runs `la_decompose_components` + `la_hull_decomp` at two points each iteration: (1) before evaluation (splits multi-component input or previously-cut parts for more accurate cost assessment), and (2) after applying cuts (splits multi-component results before the next iteration). This is useful for multi-component input meshes where splitting early allows the lookahead tree search to evaluate each component independently. The existing `decompose_components` parameter (end-of-decomposition only) remains the default.

## CLI: Refactored Normalize/Decompose/Denormalize into _decompose_mesh

Extracted the normalize→decompose→denormalize pipeline from `_processor_worker` into `_decompose_mesh()`, reused by the new `--serial` mode. Added `--serial` flag for single-threaded load-process-save (easier debugging). Added `--decompose-components-per-iter` CLI flag.

## decompose_components_block Now Computes mesh_vol per Component

Added phase 11b to `decompose_components_block`: after scattering vertices and triangles into per-component output meshes, each component's `mesh_vol` is computed via `mesh_volume_warp` (4 warps stride over components in DC_BLOCK=128 threads). This replaces the inherited parent volume, giving accurate per-component volumes for downstream cost evaluation.

## L-shape Test Fixture Must Be Watertight

The `_make_lshape()` fixture in `test_lookahead.py` was changed from the original `_merge_meshes([box_a, box_b])` approach to a hand-coded vertex/index list that was NOT watertight (euler_number=-1, is_watertight=False). Non-watertight input causes `mesh_volume_warp` to return incorrect volumes, making rv cost unreliable and preventing convergence. Fix: restored the original `_box()` + `_merge_meshes()` approach which produces a watertight L-shape (two overlapping boxes with outward-facing triangles). The merged mesh is watertight with volume=4 (unnormalized).

## decompose_components: Inner Shell Filtering

Meshes with thick shells (e.g., the Stanford bunny, which has an outer surface and an inner cavity surface) decompose into multiple connected components where the inner cavity surface has negative signed volume as a standalone mesh (its normals point inward toward the cavity center). When `decompose_components_per_iter=True`, these inner-shell components were passed to `plane_cut`, which produced non-manifold output — 73 duplicated directed edges (same-direction edge appearing twice instead of once each way).

Root cause: the bunny is a thick shell — outer surface + inner cavity surface. Together they form one valid volume. After splitting, the inner surface has flipped winding (normals point into the cavity, correct from the solid's perspective but inverted as a standalone mesh). `plane_cut` operating on this flipped-winding input produces broken edge consistency.

Fix: in `decompose_components_block` Phase 11b, compute signed volume via `mesh_signed_volume_warp` instead of `mesh_volume_warp`. Components with negative signed volume are inner shells — they are compacted out of the output array and their mesh memory is freed. At least one component is always kept (if all have negative signed volume, filtering is skipped). Plane cut cannot produce inner shells from outer shells, so this filtering is only needed in decompose_components.
