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

**Update after A12 + A13** (vblock hoists in `bt_mergeProjection` and
`bt_findEdgeForCoplanarFaces`), 5×100 bench, mean of medians:

| Variant | mean of medians (ms) | vs NONE |
|---|---:|---:|
| NONE (pre-perf)         | 339.0  |     —  |
| A6 + A9 + A10           | 322.9  | −4.8 % |
| **A6 + A9 + A10 + A12 + A13** | **317.6** | **−6.3 %** |

A12 gives the material win (~6 ms / ~2 %); A13 is noise-level but kept
for code-style consistency with A12. A14/A15 (la_expand barrier /
metadata parallelization) were tried and reverted — no measurable win.

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

### A11. `kdop_hull_block` step 1 — 2x axis tiling (REVERTED)

Hypothesis: process two axes per outer iteration so each `verts[i*3+{0,1,2}]`
load feeds two dot products. Halves the number of address calculations and
load instructions issued, and pairs naturally with `KDOP_N_AXES = 40`.

Implementation: carry two sets of `(dx,dy,dz)` and two sets of
`(lmax,lmax_i,lmin,lmin_i)`; one inner-loop iteration computes both `d0` and
`d1` from the same `(vx,vy,vz)`; followed by four (instead of two) interleaved
warp argmax/argmin reductions per outer iter; lane-0 stores write 12 floats
(instead of 6).

Result: `bench_la 100`, 5 runs, mean of medians **319.0 ms** vs A10's
**318.6 ms** — within noise (Δ ≈ 0.4 ms, < 0.15 %). `kdop_hull_kernel` reg
count unchanged at 142 (compiler reused regs across the carried state, since
total kernel reg count is dominated by the inlined `hull_dandc_warp_mesh`).

Why the predicted ~1 ms/call savings did not materialize: at typical
`nv ≈ 5000`, the entire `verts[]` buffer is ~60 KB and fits in L1 (128 KB on
sm_89). After the first axis primes the cache, every subsequent axis hits L1
— so the "halved memory accesses" were already cache hits, not DRAM round-
trips. The remaining savings (halved address compute + load issue rate) are
swamped by the two inlined `hull_dandc_warp_mesh` calls in steps 3 and 7.

Reverted. Worth revisiting only if step 1 ever shows up as a hot region in a
profile, or if `nv` regularly exceeds the L1 capacity (~10K+ verts).

### A12. Hoist `vblock[v].point` in `bt_mergeProjection` (KEPT)

Ncu 2026.1 source-page analysis (summed across all invocations of `la_hull`
over one bunny end-to-end run):

| Top SASS site in `la_hull` | Samples | Stall class |
|---|---:|---|
| `IMAD.WIDE R*, R*, 0x28, R*`  @ 0x50efe0 | 1.82 M | long_sb (vblock gather) |
| `IMAD.WIDE R*, R*, 0x28, R*`  @ 0x52bee0 | 1.80 M | long_sb |
| `LDG.E.128 R*, [R*]`          @ 0x52bee8 | 1.25 M | long_sb |

The `0x28` (=40) stride is `sizeof(BtVertex)`. These are still the same kind
of random gather A5/A6/A9 were chasing — but the top two PCs localize to
`bt_mergeProjection`'s side-walking while loops (primary/secondary walk two
chains of `vblock[v].point` advancing by `.next` / `.prev`).

In the loop, every iteration reloads `vblock[v0].point`, `vblock[v1].point`,
`vblock[w0].point`, `vblock[w1].point` to compute `dx`, `dy`, candidates, and
advance. The compiler doesn't CSE across iterations because `v0/v1/w0/w1`
change every iteration. But when `v0 = w0`, the "new" `v0_pt` is literally
the `w0_pt` we just loaded.

Implementation: hoist `v0_pt`, `v1_pt` at the top of each `for (side)`
iteration. Inside the while loop, load `w0_pt = vblock[w0].point` /
`w1_pt = vblock[w1].point` once per iteration, reuse for dx0/dy0/dxn/dyn
tests, and on the `v0 = w0` / `v1 = w1` advance, also carry `v0_pt = w0_pt`
(skipping the reload). Same pattern in the `dx==0` inner loops (new
`t_pt = vblock[t].point` local).

Result: `bench_la 100`, 5 runs each, mean of medians:

| Variant | mean of medians (ms) | vs prior |
|---|---:|---:|
| Prior baseline (A6+A9+A10)   | 322.9 | — |
| + A12                         | 316.8 | −1.9 % |

138/138 tests pass. `la_hull` register count unchanged at 142 (compiler
reused the registers that previously held temp `bp32` fields).

Kept.

### A13. Hoist `vblock[f].point` in `bt_findEdgeForCoplanarFaces` (KEPT)

Companion to A12, targeting the two "slide along coplanar boundary" loops at
the top of `bt_findEdgeForCoplanarFaces`. A6 already hoisted the per-body
copy, but each iteration re-evaluates `vblock[f0->target].point` and
`vblock[f1->target].point` for both the `dy`/`dxn` test and the `et0 = ...`
assignment — reading the same gather twice per iteration.

Implementation: inside the dx>0 and dx<0 while loops, load `f0_pt` /
`f1_pt` once per iteration, share across the test and the `et0 = f0_pt;` /
`et1 = f1_pt;` update. Also share the `d0 = bp32_sub(f0_pt, et0)` compute
feeding both `dxn` and the angle test.

Result: `bench_la 100`, 5 runs, mean of medians:

| Variant | mean of medians (ms) | vs A12 |
|---|---:|---:|
| A12 alone                     | 316.8 |     — |
| + A13                         | 317.6 | +0.2 % |

Noise-level (Δ ≈ 0.8 ms within a run-to-run std of ~30 ms). The compiler
was already doing most of the CSE A13 forces by hand, so the code-level
change is a wash. Kept for consistency with A12's style (explicit hoist),
no regression, 138/138 tests pass, `la_hull` regs still 142.

### A14. Parallelize `tid==0` metadata write in `la_expand` (REVERTED)

The 24-line `if (tid == 0)` block at la_expand.cu:250 serializes on tid 0:
two full `Part` struct copies (`wo->parts[np-1] = pp.pos;`
`wo->parts[np] = pp.neg;`), scalar metadata writes, and a short
`level_costs` copy loop. Each `Part` is 76 B (20 ints), so striped int
writes across 64 threads could do 1 ST/thread instead of 40 ST/thread for
both copies.

Implementation: treat `(int*)&pp.pos`, `(int*)&wo->parts[np-1]` as int
arrays and stride the copy across `tid`. The `pp` returned by
`plane_cut_block` is thread-local but identical across threads
(`plane_cut_block` returns from shared `s_result`), so striped writes from
different threads reconstruct the struct consistently. Move `level_costs`
copy out into its own parallel loop; keep the 4 scalar metadata writes on
tid 0.

Result: A12+A13 + A14 + A15 bench, 5 runs, mean of medians:

| Variant | mean of medians (ms) | vs A12+A13 |
|---|---:|---:|
| A12+A13                 | 317.6 |     — |
| + A14 + A15             | 319.4 | +0.6 % |
| + A15 only              | 321.2 | +1.1 % |

All values within run-to-run std (~30 ms) but trending neutral/slightly
worse, not better. Hypothesis for why A14 doesn't win: taking `&pp.pos`
forces `pp` into addressable (local) memory, so the "serial" struct copy
the compiler previously emitted from registers now becomes a local-memory
load → global store. The issue-rate savings from parallelizing 40 STs
across 64 threads is smaller than the added local-memory-read latency.

138/138 tests pass under A14+A15. Reverted both to keep A12+A13 as the
final kept state.

### A15. Remove redundant `__syncthreads()` after `plane_cut_block` (REVERTED)

`plane_cut_block` ends with `__syncthreads()` at plane_cut.cuh:1491 before
`return s_result;`. The following `__syncthreads()` at la_expand.cu:201
immediately after `pp = plane_cut_block(...)` is semantically redundant —
all threads already synced before reading s_result.

Implementation: replaced the redundant barrier with a comment.

Result: 5×100 bench first pass put solo A15 at 321.2 ms vs A12+A13 baseline
317.6 ms, suggesting a slight regression. Because the 5-run delta was
inside the run-to-run std (~30 ms), re-ran head-to-head 10×100:

| Variant | medians (10 runs, ms) | mean of medians | median of medians |
|---|---|---:|---:|
| A12+A13 baseline | 318.7 321.5 315.0 316.2 317.3 320.4 315.8 316.0 319.9 314.9 | **317.57** | 316.75 |
| + A15            | 320.9 320.2 319.1 321.5 309.7 312.7 320.4 323.2 315.0 316.3 | **317.90** | 319.65 |

Mean-of-medians Δ = 0.33 ms — indistinguishable from zero over a per-sample
std of ~35 ms. Median-of-medians leans baseline-ward by ~3 ms but the
ten-run sets overlap heavily. So A15 is zero-to-slightly-negative at the
noise floor, not a win.

Hypothesis for why the "redundant" barrier isn't free when removed: the
subsequent empty-mesh branch reads `pp.pos.mesh.nv` / `pp.neg.mesh.nv`
(local registers populated from the shared `s_result` inside
`plane_cut_block`). Without the outer sync the two warps may drift out of
phase at this branch, increasing divergence-related scheduler overhead in
the tid==0 heap_free path that follows. Whatever the micro-reason, the
measurement does not justify the change.

Reverted. 138/138 tests pass under A15 alone; kept configuration remains
A12+A13.

### A16. Branch-and-bound pruning of lookahead paths (REVERTED)

Each lookahead leaf's path cost is `sum(level_costs[0..total_levels-1]) /
total_levels` (averaged with a fixed divisor — `all_small` early-exits pad
the rest with 0). Between expansion levels we now know `level_costs[0..k-1]`
for each item. For each item at level k:

- **LB**: `sum_i = Σ level_costs[0..k-1]` (future costs ≥ 0).
- **UB**: `sum_i + worst_i * (total - k)`, where `worst_i =
  level_costs[k-1]`. Valid because `sort_and_record` records the max-rv
  part, and cutting that worst part into two halves plus retaining
  inherited parts (all ≤ worst by sort order) can only keep or decrease
  the next level's worst — so `level_costs` is non-increasing.

If `LB_i > min_j UB_j` over siblings sharing the same `src_part_idx`,
item i cannot beat some cousin even under best-case completion → mark
`src_part_idx = -1`. `la_expand`/`la_expand_quick` then early-return on
pruned parents; `la_evaluate` naturally skips items whose src doesn't
match the active block (via `src != my_idx`).

Implementation: two new single-warp kernels `la_prune_reduce` (reduce min
UB per src via atomicMinF) and `la_prune_apply` (mark prunes). Invoked
after each `la_sort_and_record` in the full- and quick-expansion loops,
guarded to skip the last level (where `k == total_levels` would be a
no-op).

Result at default `depth=2, quick_depth=1` (total_levels=3): **0 prunes
out of 6541 candidate items** on the bunny workload. The UB is too loose
at low total depth:

- k=1: UB_i = `level_costs_i[0] * 3`, prune needs
  `level_costs_i[0] > 3 × min_j level_costs_j[0]` → requires 3× spread
  across (cut × src) pairs, which initial cuts don't produce (they give
  narrowly clustered rv costs).
- k=2: UB_i = `level_costs_i[0] + 2 × level_costs_i[1]`, needs similarly
  large spread that doesn't materialize.
- k=3 (last level before evaluate): guard skips — no pruning opportunity.

5×100 bench with pruning: mean of medians **318.9 ms** vs **318.6 ms**
baseline — the reduce+apply launch overhead is within noise (~0.3 ms),
but there is no offsetting win.

Conclusion: algorithm is correct, infrastructure works, but the BnB bound
is too weak at shallow search depth. Reverted. Potentially worth
revisiting if the default depth grows (k ≥ 3 with total_levels ≥ 5 would
have tight UBs), or paired with a second-worst tracker to tighten the
projection.

138/138 tests passed under A16 before revert.

---

### A17. Rough-hull lower-bound pruning via probe pass (REVERTED)

Hypothesis: in `kdop_hull_block`'s slow path (nv > 1024), step 3 builds an
extreme-point rough hull as a strict superset of the exact hull. We can
extract its volume and convert to a lower bound on `la_part_cost_rv`,
giving a true LB on the part's contribution to the path cost. Prior offline
analysis on bunny showed ~67% of slow-path `la_hull` calls at d=1, iters
0–2, would be prunable if the threshold came from a probe-derived per-src
attainable best.

Implementation:
- `kdop_hull_block` accepts pruning params; after step 3 computes
  `rough_vol = mesh_volume_warp(rough_hull)`, derives
  `rv_lower = 0.3·cbrt((3/4π)·max(rough_vol − mesh_vol, 0))`,
  computes `path_cost_LB = (partial_sum + max(max_inh_rv, rv_lower)) /
  total_levels`, and if it exceeds best returns a sentinel hull whose
  volume gives `la_part_cost_rv == level_cost_LB`.
- New `la_best_by_src` kernel reduces probe per-item path costs into
  `best_by_src[]` via `atomicMinF`.
- `lookahead.c` runs a probe pass before the final full-expand level
  (only when `quick_depth==0 && d==depth-1`):
  `la_expand_quick → la_hull → la_sort_and_record → la_best_by_src →
  la_cleanup_tree`. Then the main `la_hull` consumes `best_by_src[]`.
- Refcounting: pruned hulls leave `refcount=NULL` so `la_cleanup_tree`
  skips heap_free.

Toggle: `COACD_LA_ROUGH_PRUNE` env var (default 1).

Benchmark (mirrors the canonical `/tmp/bench_la.py`: bunny normalized,
2 warmups + 100 timed runs, single Context, threshold=0.05 default).

Baseline (`COACD_LA_ROUGH_PRUNE=0`) confirms dev-log baseline:

| n   | mean   | median | min    | max    | std   | parts |
|-----|--------|--------|--------|--------|-------|-------|
| 100 | 322 ms | 316 ms | 273 ms | 441 ms | 27 ms | 43    |

`COACD_LA_ROUGH_PRUNE=1`: **fails with `plane_cut:pool_oom` at iter 5**
on bunny default config. The probe pass leaks pool memory across runs
(`la_expand_quick` writes child items whose Mesh.verts heap allocations
are not freed before the main full-expand overwrites the buffer / before
the next outer iteration). When the prune path runs for ~5 iterations of
repeated end-to-end calls in one Context, the bump allocator is
exhausted.

Coarse 5-run smoke before the OOM was hit (n_warmup=2, n_runs=5,
single-mesh-per-run in fresh subprocess on canonical normalized meshes):

| mesh | prune=0 (min) | prune=1 (min) | delta |
|------|---------------|---------------|-------|
| bunny  | 272 ms | 378 ms | -39% |
| dragon | 318 ms | 398 ms | -25% |
| camel  | 343 ms | 426 ms | -24% |
| hand   | 250 ms | 312 ms | -25% |
| teapot | 406 ms | 508 ms | -25% |

Uniform 24–39% regression on top of being broken.

Why it fails: the probe runs `la_hull` on the 1-child quick-expand items
(1× cur_n_cutting), while the main pass runs it on width-child items
(60× cur_n_cutting at d=0, width2=5× at d=1). The probe's relative cost
is therefore highest at d=1, exactly where pruning is supposed to fire.
Empirically, even on bunny — where prior offline analysis suggested
~67% of slow-path la_hull calls at d=1 iters 0-2 were prunable — the
total wall time grows by 39%, meaning the probe cost dwarfs the saved
slow-path work and/or pruning fires far less often once a real probe
(rather than oracle best) drives it.

Also observed: under the wrong-mesh / wrong-threshold debug runs
(non-normalized meshes), several `prune=1` runs followed by a `prune=0`
run on hand@0.05 hit `plane_cut:pool_oom` at iter 0 — likely pool
fragmentation/pressure from probe-pass allocations leaving residue.

Conclusion: correct, but net-negative. Reverted (changes confined to
`kdop_hull.cuh`, `la_expand.cu`, `la_refine.cu`, `la_lifecycle.cu` (no-op),
`structs.h`, `heap.c`, `lookahead.c`).

138/138 tests passed before revert.

---

### A18. Rough-hull BnB pruning — no probe pass (KEPT)

A17 revisited, minus the probe pass that sank it. The rough-hull LB from A17
is sound; the probe's 24–39% regression came from running a miniature
la_hull pass just to derive a threshold. A18 sources the threshold from
A16's branch-and-bound UB instead — free to compute, and tight enough at the
last la_hull call to actually fire.

**Idea.** At the LAST la_hull invocation (`quick_depth==0 && d==depth-1`, or
`q==quick_depth-1`), for each candidate item *i*:

- UB_j on the true total path sum: `partial_sum_j + level_costs_j[n_levels-1] *
  (total_levels - n_levels)`. Sound because `sort_and_record`'s max-rv
  recording guarantees `level_costs` is non-increasing (cutting the worst
  part can only keep or shrink the next-level worst).
- `best_UB = min_j UB_j` — an upper bound on the optimum.
- In `kdop_hull_block`'s slow path, after the extreme-point rough inner hull
  is built (step 3), compute
  `rv_LB = 0.3 · cbrt((3/4π) · max(rough_vol - mesh_vol, 0))`. Rough hull ⊂
  exact hull so `rv_LB` is a true LB on the part's rv cost.
- If `partial_sum_i + rv_LB > best_UB`, skip step 4–8 (the filter + exact
  D&C — the expensive half of the slow path). Return `verts=NULL` plus
  `hull_vol = rough_vol`.

**Why last-call-only.** Recording an underestimated `level_cost[d]` breaks
the non-increasing invariant that *deeper* levels' UBs rely on — a shallower
level's recorded cost can no longer upper-bound deeper level costs once it
itself is an underestimate. At the last level there are no deeper levels, so
this is safe.

**Sentinel handling.** Pruned hull returns with `verts=NULL` and
`refcount=NULL`, so `la_cleanup_tree` naturally skips heap_free.
`la_sort_and_record` reads only `hull_vol`/`mesh_vol` through `la_part_cost_rv`,
so the LB flows through as an underestimated `level_cost[d]`. Because that
underestimate would make provably-dominated items look *better* than their
true cost in `la_evaluate` (which picks the per-initial-cut min), every
la_hull block that prunes also writes `wi->src_part_idx = -1`, which makes
`la_evaluate`'s `src_part_idx != my_idx` check filter the item out.

**Threshold derivation.** Caller converts rv-space to vol-gap space (the
only form `kdop_hull_block` can test cheaply against `rough_vol - mesh_vol`):
given `rv_thr = best_UB - partial_sum`, pass
`vol_gap_thr = (rv_thr / 0.3)³ · (4π/3)`. `rv_thr ≤ 0` collapses to
`vol_gap_thr = -1` (always prune). `rv_thr = +inf` (sentinel `1e30f`)
disables pruning entirely, which is what non-last la_hull calls pass.

**Implementation.**
- `kdop_hull.cuh`: added optional `prune_vol_gap_threshold` and `mesh_vol`
  params (default `1e30f` / `0` = disabled). Slow path runs a
  `mesh_volume_warp` on the rough hull and early-exits if the LB exceeds
  the caller's threshold.
- `la_expand.cu`: la_hull now takes `const float* best_ub_sum_d` and
  `int total_levels`. On prune (hull.verts==NULL) skips `LA_REFCOUNT_HEAP`
  assignment and sets `wi->src_part_idx = -1`.
- New kernel `la_compute_best_ub`: single warp min-reduce of UB_j over
  items with `n_levels > 0`.
- `structs.h`/`heap.c`: register `fn_la_compute_best_ub`.
- `lookahead.c`: one extra `cuMemAllocAsync(d_best_ub, 4)`. Before the last
  la_hull call, launches `la_compute_best_ub` on `d_cur` (whose
  `level_costs` are the same as `d_next`'s since `la_expand` copies them);
  passes `&d_best_ub` as the new kernel arg. All other la_hull calls pass
  `NULL`.

**Benchmark** (canonical `/tmp/bench_la.py`: normalized bunny,
`lookahead_decompose(width=60, width2=5, depth=2, quick_depth=0,
threshold=0.05, max_iters=100)`, 2 warmups + 100 timed, 5 reps, same
binary — toggle via short-circuited flag):

| Variant           | means (ms, 5 reps)               | mean  | medians (ms, 5 reps)             | mean of medians |
|-------------------|----------------------------------|------:|----------------------------------|----------------:|
| Baseline (off)    | 319.5, 325.9, 328.6, 320.4, 325.2 | 323.9 | 314.9, 320.9, 320.3, 316.1, 319.0 | 318.3 |
| **A18 (on)**      | 311.5, 311.7, 314.0, 312.4, 309.6 | **311.8** | 309.6, 306.3, 309.8, 310.4, 309.7 | **309.1** |

Δ ≈ **-12 ms on mean / -9 ms on median (~3 % speedup)** — the A18 median
set sits strictly below every baseline median. Parts: 42–51 (baseline
41–51) — decomposition quality preserved.

138/138 tests pass.

Kept.

---

## Ablation: `max_n_cutting` (runtime knob, default 16)

`max_n_cutting` is the per-iteration cap on cutting candidates processed
in parallel; it also sets the breadth of the tree-search leaf fan-out
(`max_leaf_items = max_n_cutting` in `csrc/lookahead.c:183`). Distinct
from the compile-time `LA_MAX_PARTS` (parts array *inside one* LaWorkItem
— `cuda/structs.cuh:55`).

Ran canonical bench (`/tmp/bench_la_pool.py`: normalized bunny,
`width=60, width2=5, threshold=0.05`) with 2 warmups + 100 timed, 5 reps
each config. Pool HWM read back via `ctx.pool_usage()`.

| `max_n_cutting` | mean (5-run avg) | median-of-medians | pool HWM (max) | parts |
|-----------------|-----------------:|------------------:|---------------:|-------|
| **16** (default) | 311.9 ms | 311.3 ms | 7773 MB | 44–50 |
| 20               | 313.8 ms | 310.4 ms | 7800 MB | 42–50 |
| 24               | 315.8 ms | 311.3 ms | 7851 MB | 39–50 |
| 32               | 313.0 ms | 310.3 ms | 7773 MB | 40–48 |

All variants within run-to-run noise (σ ≈ 25 ms per 100-run block). No
speedup from raising the cap: bunny with `width=60` converges before
hitting 16 simultaneous cutting candidates, so higher mnc adds search
breadth that is never consumed. Pool HWM rises slightly at mnc=24 (+80
MB) and falls back at mnc=32 — likewise within noise.

Conclusion: leave default at 16. No action.

---

## Heap-leak fix: `la_apply_cuts` refcount double-increment

### Symptom

Across a 100-iter lookahead_decompose on normalized bunny (width=60, width2=5,
threshold=0.05), `pool_usage()` kept climbing even though the tree-search
intermediates were supposedly freed by `la_cleanup_tree` / `la_cleanup_tree3`.
Pool HWM was stable-ish but `DeviceHeap` had a growing number of live blocks
that were never freed — classic refcount leak.

### Instrumentation added

Three small pieces of diagnostic plumbing (kept in-tree, all low overhead):

1. **Per-heap counters** in `DeviceHeap` (`cuda/allocator.cuh`):
   `outstanding_bytes`, `alloc_count`, `free_count`. `atomicAdd` on every
   `heap_alloc` / `heap_free`. Initialized in `heap_init_kernel`
   (`cuda/mm.cu`).
2. **C API** `gpu_heap_stats(ctx, out[6])` in `csrc/heap.c` + Python wrapper
   `ctx.heap_stats()` returning `(heap_outstanding, heap_allocs, heap_frees,
   scratch_outstanding, scratch_allocs, scratch_frees)`. Use
   `heap_allocs - heap_frees` as a live-chunk count; stable across iterations
   ⇒ no leak, growing ⇒ leak.
3. **Env-gated per-phase leak bisector** in `csrc/lookahead.c`:
   set `COACD_LEAK_BISECT=1` and the host snaps both heap counters after
   every kernel launch in the iter loop, printing
   `[bisect iter=N] <phase> heap_diff=X (+Δ) scratch_diff=Y (+Δ)`. Makes
   it trivial to localize which kernel is the source of the drift.
4. **Optional device-side assertion** via `COACD_LEAK_PROBE` build flag:
   when enabled, `la_free_decomp` prints any part whose refcount was not
   1 at the moment of final free (i.e. had stragglers).

### Root cause

`la_apply_cuts` (`cuda/la_lifecycle.cu`) was double-incrementing the
mesh/hull refcounts of the two adopted halves when it wrote them into
the decomp slot. The level-0 slot that previously held those halves is
about to be nulled (nparts=0), so `cleanup_tree3` will *not* decrement
for it; meanwhile, the chunk already came in with a refcount balanced for
the cleanup-of-d_cur-leaves sequence. Adding +1 here left each iter's
adopted halves with refcount one too high → orphaned heap blocks every
iteration.

### Fix

`cuda/la_lifecycle.cu:211-224` — remove the four `atomicAdd(...->refcount, 1)`
calls; replace with an explanatory comment noting the ownership-transfer
semantics. No new code path; just a deletion.

### Measured impact

| Iters | Live heap chunks after `la_free_decomp` |
|-------|---------------------------------------:|
|   1   |  1 |
|   2   |  3 |
|   5   | 16 |
|  10   | 33 |
|  20   | 68 |
|  50   | 71 |
| 100   | 69 |

Prior to the fix, the 100-iter default run leaked ~141 chunks; after the
fix, 68 — roughly a 50% reduction. Scratch heap is clean (0 live) in all
runs, confirming the scratch-vs-main separation is airtight. **The leak
is bounded and self-terminating**: bunny converges near iter ~20
(`n_cutting` drops to 0) and the live count stops growing — indeed at
iter 50 it's 71 and at iter 100 it's 69, within noise. During productive
iters the leak grows at ~3.5 chunks per iter; once no more cutting
happens per-iter, it flattens.

A follow-up bisect run with
`lookahead_decompose(..., no_decompose_components_per_iter=True)` showed
**0 live chunks after `la_free_decomp`** for 1 iter. The residual leak
is therefore confined to the `la_decompose_components` + post-iter
`la_hull_decomp` path that runs when
`decompose_components_per_iter=True` (the default). With multi-component
splitting disabled, the tree-search core is completely balanced.

### Residual leak: localized but not fully pinned

Per-kernel bisect on iter 19 shows allocs/frees balance to within +3-4/iter
while cutting, and 0/iter after convergence. The localization is now tight:
the leak ONLY happens when `decompose_components_per_iter` is on (default).
A 1-iter bisect with the flag off shows 0 live chunks after `la_free_decomp`.
The per-kernel alloc/free flow during iter 19 (after convergence this will
not run, but during productive iters it looks like):

| phase (iter 19) | alloc delta | free delta | net |
|-----------------|-----:|-----:|-----:|
| expand_d0       | +480 |      |      |
| hull_d0         | +480 |      |      |
| expand_d1       | +1437 | -5 |      |
| hull_d1         | +1371 |     |      |
| cleanup_tree_d1 |       | -480 |     |
| apply_cuts      |       | -4 |      |
| hull_decomp     | +8   |     |      |
| cleanup_tree3   |       | -3275 | +12 |

All kernels are accounted for — no hidden launch — but the net is +12 not
0. Candidates: non-chosen cuts whose hulls never get swept by
cleanup_tree3; inherited-part refcount asymmetry between `la_expand` and
`la_cleanup_tree_item`; `la_hull_decomp` allocating without a matching
free path. A rigorous re-derivation of the refcount invariants per
kernel would likely pin it, but the impact is small (68 live blocks /
~7 GB pool) and non-fatal — pool_usage tracks HWM so fragmentation is
bounded. Deferred.

The "mystery +2 at `pre_hausdorff` at iter 0" from earlier notes is not
actually mysterious: `la_decompose_components` + `la_hull_decomp` are
launched inside the per-iter block (guarded by
`decompose_components_per_iter`, default on) between `iter_start` and
`pre_hausdorff`. For bunny (which trimesh splits into 2 components,
confirmed via `mesh.split(only_watertight=False)`), the flow is:
Phase 8 allocates 2 output meshes (+2), Phase 11 detects one inner shell
and frees it (-1), Phase 12 skips freeing the input (refcount=NULL,
owned externally), then `la_hull_decomp` allocates 1 hull for the
surviving outer shell (+1). Total `+3 allocs, +1 free, net +2` — matches
the bisector exactly. No bug there; just a case of "the kernel between
the snaps wasn't recognized as producing heap allocs."

---

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

