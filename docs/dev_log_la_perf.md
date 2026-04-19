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

≈ 5% on the in-process measurement, ≈ 8% on the CLI measurement. No
regressions; 138/138 tests pass.

## Candidates considered but not pursued

- **Broadcast the merged point inside `bt_merge_pair`**: both primary and
  secondary independently load `vblock[c0].point` / `vblock[c1].point` each
  loop iteration after a __shfl_sync broadcast of the indices — 4 random
  loads that could collapse to 2 + a shuffle. Material refactor, left for a
  follow-up.
- **Reduce heap_alloc/free frequency in `kdop_hull_block`**: fast path does
  ~5 allocs/frees per block; could be fused into a single up-front scratch
  reservation. Significant rewrite of the allocation protocol; deferred.
- **Fuse `la_expand` + `la_hull` into a single kernel**: would cut launch
  overhead but requires reworking the per-block grid math and refcounting.
  Out of scope for this pass.

