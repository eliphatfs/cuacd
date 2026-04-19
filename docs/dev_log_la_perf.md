# Lookahead Performance Optimization — Dev Log

Goal: Reduce end-to-end wall time of the lookahead pipeline (`coacd-gpu
tests/data/vhacd2_data_r0.1_mv10k/bunny.obj decomp_output/ --pool 10G --serial`),
focused on the `la_hull` kernel which dominates the profile (≥ 50% of total
GPU time per `pipeline_overview_bunny4_ncu.csv`).

## Baseline (before any changes)

End-to-end (5 runs, --serial, --pool 10G):

```
bunny.obj  0.40s  46 parts
bunny.obj  0.39s  50 parts
bunny.obj  0.33s  41 parts
bunny.obj  0.34s  46 parts
bunny.obj  0.35s  45 parts
```

Mean ≈ 0.36 s, range 0.33–0.40 s, 41–50 parts.

Test suite: `python -m pytest tests/ -v` → 138 passed.

Per-kernel takeaways from the NCU overview:
- `la_hull` is the single largest contributor (~7–33 ms per launch, many launches).
- Hot source line in `la_hull` is `bt_findMaxAngle` and `bt_findEdgeForCoplanarFaces`,
  dominated by `vblock[e->target].point` random global loads.
- `la_expand` is second largest (~1–11 ms each, fewer launches).

Most experiments below target the inner loops of those two routines.

---

## Approach log

(Each entry: change, rationale, result, decision.)

In-process bench: `python /tmp/bench_la.py <n>` runs lookahead_decompose on the
normalized bunny, 2 warmups + N timed; reports mean/median/std. Baseline across
100 runs: mean ≈ 347 ms, median ≈ 341 ms.

### A1. `__align__(16)` on BtPoint32 (already present)

`warp_sort.cuh` already tags `BtPoint32` with `__align__(16)` so nvcc can fold
the 4×int32 point load into a single `LD.E.128`. `hull_dandc.cuh` re-defines
the struct under a `#ifndef BT_POINT32_DEFINED` guard, so whoever includes
first wins — in practice `warp_sort.cuh` is pulled in first through
`hull_dandc.cuh`, so the aligned variant is what compiles. No change needed.

### A2. Cache `target_point` inside `BtEdge` (REVERTED)

Hypothesis: the dominant latency in `bt_findMaxAngle` is the random
`vblock[e->target].point` load. Caching the destination point inside the edge
struct (with `__align__(16)`) would turn it into a sequential load relative to
`e`, avoiding a dependent random gather.

Implementation: added `BtPoint32 target_point` to `BtEdge`, initialized in
`bt_newEdgePair`, replaced ~28 call sites of `vblock[X->target].point` with
`X->target_point`.

Result: `bench_la 100` → mean 383.5 ms, median 374.0 ms (≈ +10% regression).

Why it backfires:
1. `sizeof(BtEdge)` doubles from 32 → 48 B. Ring traversal in
   `bt_findMaxAngle` / `bt_merge_pair` now pulls 2 L1 sectors per edge instead
   of 1, inflating cache footprint across a ring that often has tens of edges.
2. `bt_newEdgePair` does two extra random `vblock` loads at edge *creation*
   time to snapshot the point. Edge creation is on the merge critical path
   (every new edge blocks progress until the loads return), so moving the
   gather earlier does not erase its cost.

Reverted.

### A3. PTX `atom.acquire.gpu.cas` / `st.release.gpu` lock (REVERTED)

NCU attributes the single largest stall bucket in `la_hull` to the two
`__threadfence()` calls inside `arena_lock` / `arena_unlock` (~4.9 M sampled
stall cycles, vs ~0.4 M each for the hottest `bt_findMaxAngle` lines). A
device-wide sequentially-consistent fence is stronger than a mutex needs —
acquire-CAS on lock and release-store on unlock express the same visibility
contract while only ordering operations reachable via the lock word.

Implementation: replaced `atomicCAS + __threadfence` with
`atom.acquire.gpu.cas.b32` spin, and `__threadfence + atomicExch` with
`st.release.gpu.b32`.

Result: all 138 unit tests pass; short (100-run) bench looked neutral to
slightly better; but the 200-run bench crashed with CUDA 719 (unspecified
launch failure) on the way into `cuMemcpyDtoH(&h_n_extra, ...)`.

Diagnosis: the crash is timing-dependent and only appears under sustained
load, suggesting a real (latent) race — most likely some non-lock-protected
memory op elsewhere in the pipeline that implicitly relied on the SC fence
semantics of the original `__threadfence()` to order against an unrelated
arena or pool field. Proving it safe would require auditing every implicit
memory ordering assumption across allocator + heap init + kernel interaction.
Not worth the risk for a speculative win on a stall bucket that is not
wall-time critical (see A5).

Reverted.

### A4. `HEAP_NUM_ARENAS=128` (NOT kept)

With 64 arenas and `la_hull` grid sizes up to 3376 blocks, each arena is hit
by ~53 blocks → serialized mutex contention per arena. Doubling to 128 halves
the contention.

Result: `bench_la 100` → mean 340 ms, median 334 ms (≈ 2% faster).  256
arenas: 350/345 — slightly *worse* than 64, likely from increased pool-slab
fragmentation and wider arena bitmap scans.

The tiny gain is within run-to-run variance and inconsistent with the
pre-existing `docs/arena_sweep.md` (which shows 64 is near-optimal for the
batch-hull workload). Keeping the default at 64 to avoid pessimizing other
workloads; left env override `COACD_GPU_ARENAS=128` for users who want to try
it on their own pipelines.

### A5. Hoist `e->next` load above body in `bt_findMaxAngle` (KEPT)

In the inner ring loop:
```cpp
do {
    if (e->copy > mergeStamp) {
        BtPoint32 t = bp32_sub(vblock[e->target].point, start_point);
        ... body ...
    }
    e = e->next;          // issued at the END of the iteration
} while (e != start_edges);
```
the dependent load chain per iteration is
`e->next → e->target → vblock[e->target].point → body`. Scheduling `e->next`
at the end of the body forces the compiler to serialize each iteration on
this chain.

Hoisting `BtEdge* e_next = e->next;` to the top of the body lets nvcc issue
the `e->next` load early, overlapping its latency with the ~tens of
instructions of body work (the `br64_cmp`, the 128-bit `bt128_umul`, etc.).
No semantic change — still advancing the same ring.

Result: `bench_la 100` (two independent runs): 333.3 / 323.4 ms and 338.4 /
328.7 ms mean/median. ≈ 4–5% wall-time improvement, consistent across runs.
138 tests pass. ptxas reports +1 live register for `la_hull` (141 → 142), no
spills.

Kept.

### A6. Hoist repeated `vblock[e->target].point` reads in `bt_findEdgeForCoplanarFaces` (KEPT)

The two "slide along the coplanar boundary" while-loops at the top of
`bt_findEdgeForCoplanarFaces` read `vblock[e->target].point` up to three
times per iteration (for the normal test, perp dot, and et0/et1 update).
Under `__restrict__` the compiler likely CSEs those already, but writing it
as an explicit local makes the invariant obvious and guarantees the hoist.

Result: `bench_la 100` (two independent runs): 331.8 / 325.0 ms and 329.7 /
323.8 ms mean/median. Essentially neutral vs A5 alone — the compiler was
already CSE-ing. Kept for code clarity; no regression.

---

## Results summary

End-to-end CLI (`coacd-gpu tests/data/vhacd2_data_r0.1_mv10k/bunny.obj
decomp_output/ --pool 10G --serial`), 5 runs each:

| Variant | run1 | run2 | run3 | run4 | run5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| Baseline              | 0.40 | 0.39 | 0.33 | 0.34 | 0.35 | 0.36 |
| + A5 + A6 (this PR)   | 0.31 | 0.37 | 0.32 | 0.34 | 0.32 | 0.33 |

In-process timer (`bench_la 100`, mean/median in ms):

| Variant | mean | median |
|---|---:|---:|
| Baseline               | 347.1 | 340.9 |
| + A5                   | 333.3 | 323.4 |
| + A5 + A6              | 329.7 | 323.8 |
| + A5 + A6 + A9 (5-run avg) | 326.4 | 321.9 |

A later 4-cell compositional sweep (5 runs × 100 samples each, fresh build per
cell) refined this picture: A9's pipelined prologue + carried-state loop
inherently contains A5's `e->next` hoist, so A5 is strictly subsumed. The
final kept configuration is **A6 + A9** (no A5):

| Variant | mean of medians (ms) | vs NONE |
|---|---:|---:|
| NONE          | 339.0 |     —  |
| A5 + A6       | 325.5 | −4.0 % |
| A9 only       | 323.7 | −4.5 % |
| **A6 + A9**   | **320.2** | **−5.5 %** |

≈ 5–6 % on the in-process measurement, ≈ 8 % on the CLI measurement. A6 and
A9 compose additively (different functions, no shared state). No regressions;
138/138 tests pass.

### A7. Warp-cooperative kdop centroid (REVERTED)

`kdop_hull_block` step 5 has lane 0 sum the rough-hull vertex coordinates
(up to 80 floats) for a face-orientation reference point, broadcasting via
shared memory. Replaced with warp-strided sum + 5-step `__shfl_xor_sync`
reduction.

Result: `bench_la 100` two runs → 335.0 / 327.0 ms, 336.6 / 331.6 ms, vs the
A5+A6 baseline of 329.7 / 323.8 ms. Within run-to-run noise (std ~30–50 ms),
or marginally worse. The loop runs once per kdop_hull invocation and `nv_ext`
is small (≤ 80), so the warp version's extra shuffle overhead (~30 inst.) is
not amortized.

Reverted.

### A8. Reorder `BtEdge` fields to single-sector hot loop (REVERTED)

`bt_findMaxAngle`'s inner ring loop reads only `e->next` (offset 0–7),
`e->target` (24–27), `e->copy` (28–31). With the original Bullet field order
(next, prev, reverse, target, copy) those reads span both 16 B sectors of
the 32 B struct. Reordering to (next, target, copy, prev, reverse) +
`__align__(16)` puts the three hot fields in a single 16 B sector, which
should let the ring walk issue one LD.E.128 per edge instead of two.

Result: `bench_la 100` two runs → 338.6 / 333.3 ms, 336.3 / 329.4 ms, vs A5+A6
baseline 329.7 / 323.8 ms. Neutral within noise. Register pressure shifted:
`la_hull_decomp` 144 → 140, `kdop_hull_kernel` 142 → 140, but
`hull_dandc_kernel` 128 → 132 (worse). `la_hull` itself stayed at 142.

Why no win: the dominant cost in this loop is the dependent random gather
`vblock[e->target].point`, not the BtEdge fetch itself. Whether the BtEdge
read takes 1 or 2 LD.E.128 makes ~8 cycles of difference per iteration,
dwarfed by the 200–300 cycles for the random vblock load (which A5 already
overlaps with body work).

Reverted to keep the field order consistent with the upstream Bullet port.

### A9. Software-pipeline `bt_findMaxAngle` ring walk (KEPT)

A5 hoists `e->next` early so its load overlaps the body, but the inner-loop
body still serializes on `vblock[e->target].point` — a random gather whose
~200–300-cycle latency is much longer than the ~30–50 cycles of body work
(`bp32_sub` / `bp32_dot64` / `br64_make` / `br64_cmp`). Single-stage software
pipelining issues the *next* iteration's `e_next->next`, `->target`, `->copy`,
and the dependent `vblock[next_target].point` at the top of the body, so they
are in flight while this iteration's body executes.

Implementation: prologue loads iter 0's edge fields and `vblock[e->target].point`
into carried registers; each loop body issues iter N+1's prefetch, processes
iter N from the carried registers, then advances the pipeline. The defensive
`!e_next` bug bailout is preserved (predicates the prefetch on non-NULL).

Result: `bench_la 100`, fresh side-by-side with `git stash` for parity (5
runs each):

| | mean (ms) | median (ms) |
|---|---:|---:|
| A5+A6 baseline           | 333.0 | 324.8 |
| A5+A6 + A9 pipeline      | 326.4 | 321.9 |

≈ 2 % on mean, ≈ 1 % on median. Modest but consistent — every one of the 5
pipelined runs has a lower median than the *highest* baseline median. ptxas
shows `la_hull` register count unchanged at 142 (compiler reuses regs across
the carried state); `hull_dandc_kernel` 128 → 130. No spills. 138/138 tests
pass.

Why the win is small: a single pipeline stage hides one load latency behind
one body's worth of compute (~30–50 cycles compute against ~200–300 cycles
of latency), so steady-state still spends most of its time waiting. A 2- or
3-stage pipeline would hide more, but each extra stage costs ~6 carried
registers (BtEdge*, target, copy, BtPoint32) and `la_hull` is already at 142
— risk of occupancy regression outweighs the projected gain. Single stage is
the risk-adjusted sweet spot.

Kept.

### A5 / A6 / A9 compositional bench

To verify A6 and A9 compose additively (and that A9 truly subsumes A5 — A9's
prologue + carried-state loop *contains* the `e->next` hoist that A5 added),
ran a 4-cell sweep. Each cell: rebuild from scratch, `bench_la 100` × 5 runs.

| Variant | median range (ms) | mean of medians | vs NONE |
|---|---:|---:|---:|
| V1 NONE (no A5/A6/A9)   | 336.5 – 345.5 | 339.0 |    —    |
| V2 A5 + A6              | 323.4 – 327.7 | 325.5 | −4.0 % |
| V3 A9 only              | 319.0 – 326.9 | 323.7 | −4.5 % |
| V4 **A6 + A9** (kept)   | **315.8 – 325.1** | **320.2** | **−5.5 %** |

Reads:

- **A9 alone ≥ A5+A6**: A9's prologue + carried `e_next` give the same overlap
  A5 was buying, plus the extra latency-hiding from prefetched `e_next->target`
  and `vblock[next_target].point`. So A5 is strictly subsumed.
- **A6 composes additively on top of A9**: V4 (A6+A9) is consistently ~3 ms
  median below V3 (A9 only). A6 hoists `vblock[e->target].point` in
  `bt_findEdgeForCoplanarFaces`'s two slide loops — a different function from
  A9, so no interference is expected and none is observed.
- **Repeatability**: within-variant median spread is ~5–10 ms. The V4
  distribution sits below all V1 runs and below all but one V2 run. Signal
  is small (~5 ms = ~1.5 %) but consistent across 5 reps.

Net: A6 + A9 is the kept configuration; A5 is removed (replaced by A9).

### A10. `kdop_hull_block` plane filter — float4 + drop redundant pd==0 (KEPT)

The plane filter in `kdop_hull_block` step 5 runs `nv × nt_ext` plane checks
per kdop call (≈ 5000 × 156 = 800K iterations on bunny). Two cheap edits:

1. **Vectorize plane storage as `float4`**. Precompute writes `make_float4`
   instead of four scalar stores; filter loop reads one `float4` instead of
   four scalar `pl[t*4+k]` indexes. heap_alloc returns 16 B-aligned, and
   `t*4` floats == `t*16` B, so the cast is safe. ptxas reg count unchanged
   (compiler had already coalesced) but the source is shorter and the read
   issues as a single LD.E.128.
2. **Drop the `if (pd == 0.0f) continue;` early-out**. Degenerate triangles
   give `pnx=pny=pnz=0` and `pd=0` from the cross product, so the outward
   test `0 > 0` is naturally false — the explicit skip was redundant. It
   was also subtly wrong: a *valid* face passing exactly through the origin
   would have `pd ≈ 0` and would be incorrectly skipped. Removing it both
   simplifies and slightly broadens correctness.

Bench (`bench_la 100`, 5 runs, mean of medians):

| | mean of medians (ms) |
|---|---:|
| A6+A9 baseline (V4) | 320.2 |
| + A10               | 318.6 |

≈ 0.5 %. Modest but every A10 run is below 322 ms vs V4's 315.8–325.1 spread.
138/138 tests pass; `kdop_hull_kernel` regs unchanged at 142.

A companion attempt (call it A10b): parallelize the 6 lane-0 random gathers
in step 1 across lanes 0-5 (using the uniform `lmax_i / lmin_i` after the
warp argmax/argmin reductions). Theoretically saves ~50 K cycles per kdop
call, but bench was a wash (mean of medians 319.5 ms vs A10's 318.6 ms — a
slight regression within noise). The 6-gather tail is dwarfed by the two
inlined `hull_dandc_warp_mesh` calls. Reverted.

Kept (A10 only).

## Candidates considered but not pursued

- **Broadcast the merged point inside `bt_merge_pair`**: both primary and
  secondary independently load `vblock[c0].point` / `vblock[c1].point` each
  loop iteration after a __shfl_sync broadcast of the indices — 4 random
  loads that could collapse to 2 + a shuffle. Material refactor, left for a
  follow-up.
- **Multi-stage (2–3 stage) software pipeline for `bt_findMaxAngle`**: A9
  is single-stage. Each extra stage hides another ~30–50 cycles of latency
  but adds ~6 carried registers to a 142-reg kernel. Worth revisiting if
  ptxas headroom opens up elsewhere.
- **Reduce heap_alloc/free frequency in `kdop_hull_block`**: fast path does
  ~5 allocs/frees per block; could be fused into a single up-front scratch
  reservation. Significant rewrite of the allocation protocol; deferred.
- **Fuse `la_expand` + `la_hull` into a single kernel**: would cut launch
  overhead but requires reworking the per-block grid math and refcounting.
  Out of scope for this pass.

