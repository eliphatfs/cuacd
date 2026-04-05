# Implementation Notes

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
