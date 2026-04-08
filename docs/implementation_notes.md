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
