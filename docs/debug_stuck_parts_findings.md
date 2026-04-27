# Debugging: Stuck Decomposition Parts on High-Resolution Meshes

## Introduction

When running `cuacd` lookahead decomposition on high-resolution remeshed meshes (R256 — 256× face count), some parts get **"stuck"**: all 94 candidate plane cuts fail (CAP_INCOMPLETE), so the part remains over the concavity threshold (`*`) with identical `(nv, nt, rv_cost)` across consecutive iterations. The decomposition runs to `max_iters` without converging those parts.

**Affected mesh:** `tests/data/remesh_R256/block.obj` (184,600 verts, 369,208 tris).

**Parameters:** CLI defaults — `width=30, width2=5, depth=2, quick_depth=0, threshold=0.05, n_concave_edges=16, concave_eps=0.005, concave_threshold=3.49, concave_iters=10, max_iters=100, merge_hulls=True`.

**Reproduction command:**
```bash
cuacd tests/data/remesh_R256/block.obj decomp_output/test --verbose 1 --serial
```

The stuck behavior is non-deterministic — sometimes appears, sometimes doesn't (depends on atomic operation ordering).

## Failure Chain (Root Cause)

```
1. Plane cut generates intersection polygon loops on the mesh
2. Ear-clipping triangulation fails on complex/near-degenerate polygons
   → EARCLIP_FAIL DPRINTF diagnostic (remaining > 3 vertices un-triangulated)
3. Cap triangulation is incomplete: n_cap < n_cap_expected
   → s_n_cap = -1 set in shared memory
4. CAP_INCOMPLETE path in plane_cut_block (line 1274):
   - Allocates new heap memory, copies ENTIRE original mesh to positive side
   - Returns empty negative part (pc_zero_part)
5. la_expand.cu line 204: sees neg.mesh.nv == 0, FREES both sides, returns
   → No level-0 LaWorkItem is created for this cut
6. la_evaluate sees best_wi with nparts=0
7. la_apply_cuts skips (line 171: if nparts < 2 return)
   → Part stays unchanged in decomposition state
8. If ALL 94 candidates fail for a part → part is "stuck"
```

## The Central Mystery

**Same part, same inputs, same algorithm — different results.**

| Scenario | n_cutting | CUT_OK count | Result |
|----------|-----------|-------------|--------|
| Full run (iters 4-7) | 10-16 | **0** | Stuck |
| Isolation | 1 | ~40 | Converges (12-15 parts) |

The identical part geometry (verified by mesh_vol, centroid, bbox) produces valid cuts in isolation but not when processed concurrently with 10-16 other parts. No OOB memory corruption, no heap OOM, no use-after-free, no RNG seed difference.

### Verified Non-Causes

1. **RNG seed difference:** The concave edge RNG was seeded by `blockIdx.x` (`la_concave.cu:144`). Changed to fixed `1u` — behavior unchanged (both full run and isolation now use same seed).

2. **OOB memory corruption:** BEAM_DEBUG CheckedBuf found 0 violations.

3. **Heap OOM:** 0 KERR_PC_* errors, allocs/frees balanced.

4. **Use-after-free:** la_expand.cu line 204-209 frees the NEWLY ALLOCATED copy (from CAP_INCOMPLETE), not the original part data in decomp->parts. Original pointers intact.

5. **Part extraction mismatch:** Mesh volumes computed from saved .npz files match the verbose log exactly (e.g., nv=41596 nt=83188 mesh_vol=0.295733). Parts are truly identical.

6. **Re-normalization:** `lookahead_decompose` does not normalize input; verts passed directly to `la_initialize`.

7. **BRIDGE_FAIL:** 0 events.

8. **Text truncation:** Verbose output lines are complete; mesh_vol cross-checks confirm.

## Key Files Involved

| File | Role |
|------|------|
| `cuda/plane_cut.cuh` | Ear-clipping + cap triangulation + CAP_INCOMPLETE no-op return |
| `cuda/la_expand.cu` | Expands candidate cuts, line 204 discards cuts with empty child |
| `cuda/la_concave.cu` | Concave edge detection + reservoir sampling (line 144: RNG seed) |
| `cuda/la_lifecycle.cu` | la_apply_cuts: replaces old parts with children (line 171: skip if nparts<2) |
| `cuda/la_refine.cu` | la_evaluate: selects best cut by min path cost |
| `csrc/lookahead.c` | Host-side orchestration (concave_iters check at line 441) |
| `cuacd/cli.py` | CLI entry: normalize mesh → decompose → denormalize output |

## Code Changes Made

### `cuda/la_concave.cu` line 144
```cuda
// Before:
if (lane == 0) s_rng = (unsigned int)(blockIdx.x * 2654435761u + 1u);
// After:
if (lane == 0) s_rng = 1u;
```
**Status:** Committed. Makes concave edge sampling deterministic. Does not fix stuck behavior.

### `cuda/plane_cut.cuh` — Enhanced DPRINTF diagnostics
- CAP_INCOMPLETE now prints `nv, nt, a, b, c, d` (plane coefficients)
- Added `CUT_FAIL_NOOP` DPRINTF at the failure return (line ~1300)
- Changed final `[pc]` line to `CUT_OK` with `n_pos, n_neg`
**Status:** Only active with `CUACD_BEAM_DEBUG=1`. Useful for diagnostics.

### `scripts/debug_stuck_parts.py` — Improved matching
- Changed stuck signature from `(nv, nt)` to `(nv, nt, rv_cost)` triple
- Updated regex: `r'\[la\]\s+\d+\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+'`
**Status:** Better robustness against coincidental (nv, nt) collisions.

## Scripts Created/Used

### `scripts/debug_stuck_parts.py`
Purpose: Detect stuck parts from verbose log, save as GLB.
Usage: `python scripts/debug_stuck_parts.py <mesh.obj> [output_dir] --max-iters 8`

### `docs/debugging_stuck_parts.md`
Brief procedure document (4-step method: find, isolate, BEAM_DEBUG, explore).

## Specific Observations

### BEAM_DEBUG run (fixed seed, max_iters=8)
9 true stuck parts detected with triple (nv, nt, rv_cost) matching:

```
nv= 3019 nt=  6034 rv_cost=0.020191  spans iters 6..7
nv= 3547 nt=  7090 rv_cost=0.021532  spans iters 6..7
nv= 4081 nt=  8158 rv_cost=0.031990  spans iters 6..7
nv= 4472 nt=  8940 rv_cost=0.023330  spans iters 6..7
nv= 5959 nt= 11914 rv_cost=0.033551  spans iters 6..7
nv= 8085 nt= 16166 rv_cost=0.022309  spans iters 5..7
nv=10576 nt= 21148 rv_cost=0.034536  spans iters 4..7
nv=22015 nt= 44030 rv_cost=0.089016  spans iters 4..7
nv=41596 nt= 83188 rv_cost=0.104584  spans iters 4..7   ← largest stuck part
```

### Cut outcomes for nv=41596 (fixed seed, BEAM_DEBUG)
- Iter 2: 37 CUT_OK (blocks 379-467) — a DIFFERENT part with coincidental nv=41596, successfully cut
- Iter 3: 0 CUT_OK — the ACTUAL stuck nv=41596 part newly created here
- Iter 4: 0 CUT_OK — persists
- Iter 5: 0 CUT_OK — persists
- Iter 6: 0 CUT_OK — persists
- Iter 7: 0 CUT_OK — persists

### Without BEAM_DEBUG (original build, max_iters=8)
Stuck part nv=45245 nt=90486 detected. Isolated → 12 parts (converges).

### DPRINTF statistics (full run, 12 iters)
- EARCLIP_FAIL: 10,952 (avg remaining=150, min=4, max=3679)
- CAP_INCOMPLETE: 324,352
- BRIDGE_FAIL: 0
- CUT_OK: 9,469 (per-block count in [pc] lines)
- [OOB]: 0

## Data Files

| Path | Content |
|------|---------|
| `debug_stuck_out/block/fixedseed_stdout.log` | DPRINTF from fixed-seed BEAM_DEBUG full run |
| `debug_stuck_out/block/fixedseed_stderr2.log` | Verbose log from fixed-seed BEAM_DEBUG full run |
| `debug_stuck_out/block/fixedseed_stderr.log` | Verbose log from first fixed-seed run |
| `debug_stuck_out/block/beam_full.log` | Partial stderr from earlier BEAM_DEBUG run |
| `debug_stuck_out/block/fixedseed_part*.npz` | Saved part meshes (nv>5000) from fixed-seed run |
| `debug_stuck_out/stuck_v2.npz` | Saved stuck part from non-BEAM_DEBUG run |
| `debug_stuck_out/block/block_stuck.glb` | Stuck parts visualization (from debug_stuck_parts.py) |

## Resolution

The "concurrency" framing turned out to be a red herring — full-run vs isolation differed because the full run reached the part with the *concave-edge planes from prior iterations*, while isolation regenerated planes from a fresh seed. The real failures were two independent bugs that combined to produce the stuck behavior:

### Fix 1 — Greedy level-0 fallback in `la_evaluate` (`cuda/la_refine.cu` + `csrc/lookahead.c`)

`la_hull` runs branch-and-bound at the last full expansion level using one global `best_ub` shared across all `src_part_idx`. When one part's leaves are very cheap, B&B prunes every leaf from the *other* parts' subtrees. `la_evaluate` then sees zero surviving leaves for those parts, and `la_apply_cuts` defaults to `cut_idx=0` (often a low-quality slice) — leaving the part stuck.

The fix preserves B&B but adds a greedy fallback: when `la_evaluate` finds no leaf for a `src_part`, it falls back to the best level-0 cut for that part by its depth-0 cost. To support this, `la_sort_and_record` was extended to mirror its computed `level_costs[0]` back into the `level0_items` buffer at `d == 0`, so `la_evaluate` always has a populated cost array for the fallback even when downstream B&B kills the leaves. The host call (`csrc/lookahead.c`) passes `d_level0` + `d_total_width` only on the d=0 launch; the quick path passes NULL.

### Fix 2 — Degenerate-triangle ear-clipping (`cuda/plane_cut.cuh`)

Cap polygons at T-junctions and along long shared edges have collinear or duplicate consecutive vertices (`|cross| <= 1e-9f`). The old code treated these as non-ears and advanced past them, eventually hitting the "no ear found in one full pass" termination → `n_cap < n_cap_expected` → `CAP_INCOMPLETE` → empty negative part.

A first attempt removed the collinear vertex from the polygon and decremented `n_cap_expected`. This produced "slightly incomplete caps" — the original mesh tris that split at that vertex still referenced it, leaving non-watertight cuts. Volume errors compounded into 188 spurious parts on `block.obj`.

The current fix emits a **degenerate (zero-area) ear triangle** at collinear vertices: vertex `c` stays referenced by exactly one cap triangle `(p, c, n)`, matching the original split tris. Watertightness is preserved, the triangle contributes zero volume, and ear-clipping makes progress. The point-in-triangle test is skipped for degenerate ears (it gives spurious positives on zero-area triangles).

### Verification (`tests/data/remesh_R256/block.obj`, CLI defaults)

| | Before | After |
|---|---|---|
| Final parts | 73 (max_iters reached) | 41 (converged) |
| Iterations | 100 (cap) | 9 |
| Stuck pairs | 28 (truly stuck per dup-row-with-`*` definition) | 0 |
| Wall time | — | 3.17 s |

`pytest tests/ -v`: 138 / 138 pass.

### Tooling added

- `CUACD_DUMP_PARTS_DIR=<dir>` runtime env (`csrc/lookahead.c`): dumps all parts at every iteration as `iter%03d_part%03d.bin` (header `[nv, nt, mesh_vol, hull_vol]` then verts then tris). Lets you cross-reference the verbose log to identify which dumped parts are *truly* stuck (the same `(nv, nt, mesh_vol, hull_vol)` row appearing across consecutive iters with `*`), independent of run-to-run non-determinism in the indexing.
- `cuda/la_concave.cu`: RNG seed pinned to `1u` (was `blockIdx.x * 2654435761u + 1u`) — makes concave-edge sampling reproducible across blocks. Did not by itself fix the stuck behavior, but is required to make stuck-part repro deterministic.