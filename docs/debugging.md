# Debugging Guide

## General CUDA Debugging

- **CUDA error 700/716 is sticky** — once triggered, all subsequent CUDA calls fail. The *first* error is the real one.
- **Isolate tests**: run single failing test alone to avoid cascade from earlier tests.
- **compute-sanitizer**: `compute-sanitizer --tool memcheck python <script.py>` to find exact source.

## Build-Time Debug Flags

- `COACD_DEBUG=1 pip install -e .` — host-side debug output.
- `COACD_BEAM_DEBUG=1 pip install -e .` — device-side `DPRINTF` + `CheckedBuf` OOB detection. Use `DPRINTF` guarded by `if (lane == 0)` or `if (tid == 0)` to add temporary device-side diagnostics in `.cuh` files — no extra includes needed, `DPRINTF` is defined in `common.cuh`.
- `COACD_MEMCHECK=1 pip install -e .` — compiles with `-fdevice-sanitize=memcheck` for device-side memory error detection (requires compute-sanitizer support in nvcc).

## Runtime Debug Modes

- **Debug mode**: `ctx.lookahead_decompose(..., debug=1)` syncs after each kernel, prints per-stage status.
- **Lookahead verbose levels** (`lookahead_decompose`):
  - `verbose=1`: per-part table each iteration (nv, nt, mesh_vol, hull_vol, rv_cost, hausdorff, full_cost; `*` = above threshold), per-iteration timing and n_cutting, kernel stage OK messages.
  - `verbose=2`: additionally prints cutting_indices, la_evaluate results (best_cut_idx, best_cost per src part), leaf item details (src, cut, nparts, path_cost, level_costs), level-0 item details (per-cut per-source with per-part rv/hv/mv), per-cut best path cost summary.
- **`COACD_DUMP_PARTS_DIR=<dir>`** (runtime env): dumps every part at every iteration to `<dir>/iter%03d_part%03d.bin`. Binary layout: header `int[2] {nv, nt}` + `float[2] {mesh_vol, hull_vol}` + verts (`float32 nv*3`) + tris (`int32 nt*3`). Useful for stuck-part investigation: cross-reference the dumped files with the verbose `[la]` table to identify which parts are *truly* stuck (same `(nv, nt, mesh_vol, hull_vol)` row appearing with `*` across consecutive iterations), which sidesteps run-to-run index non-determinism.

## Key Principles

- **Evidence-based debugging** — Do not guess errors from partial output. Write a minimal reproducer or add instrumentation to observe the actual failure before making fixes.
- **rv-only tree search with hausdorff stop is correct by design** — Our algorithm is based on CoACD. CoACD also uses rv-only cost for tree search but checks `max(rv, hausdorff)` for the stopping criterion. CoACD works. The mismatch between search cost and stop cost is intentional and not a bug. Do not propose "fixing" this mismatch as a solution to convergence issues.

## Resolved Bugs and Gotchas

See `docs/implementation_notes.md` for the full list of resolved bugs and implementation gotchas, including:
- `__shared__` memory aliasing between inlined device functions
- warp_sort corrupting large structs
- refcount double-decrement in level-0 items
- plane_cut cap failure modes and detection
- union-find rank-bump race condition
- Karras BVH split binary search off-by-one
