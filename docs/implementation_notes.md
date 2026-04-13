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

## beam_sort PartKeyCmp: Equal-Cost Parts Cause Refcount Corruption

`warp_partition` fills the "equal-to-pivot" region by writing the pivot value into every equal-key slot: `data[mid_lo + i] = pivot`. When `PartKeyCmp::cmp` returned 0 for two different Parts with the same cost, the sort duplicated one Part struct over multiple array slots — including its `mesh.verts`, `hull.verts`, and `refcount` pointers. Now multiple slots reference the same heap chunk, but `beam_expansion` only incremented refcounts for the original copy count. When `beam_finalize` decremented each slot independently, it double-freed the duplicated chunk and leaked the overwritten ones. Manifested as massively negative refcounts (e.g. -1095744947), use-after-free in hull/expansion kernels, and CUDA illegal memory access errors 20-50 iterations into beam_decompose. Fix: tiebreak `PartKeyCmp::cmp` on `mesh.verts` pointer to ensure a total order — every Part has a unique mesh allocation.

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

## L-shape Test Fixture Must Be Watertight

The `_make_lshape()` fixture in `test_lookahead.py` was changed from the original `_merge_meshes([box_a, box_b])` approach to a hand-coded vertex/index list that was NOT watertight (euler_number=-1, is_watertight=False). Non-watertight input causes `mesh_volume_warp` to return incorrect volumes, making rv cost unreliable and preventing convergence. Fix: restored the original `_box()` + `_merge_meshes()` approach which produces a watertight L-shape (two overlapping boxes with outward-facing triangles). The merged mesh is watertight with volume=4 (unnormalized).
