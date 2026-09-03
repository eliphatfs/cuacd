"""bench_mm.py — Memory management benchmarks for persistent heaps.

Investigates:
  1. Compact-after-every-call vs no compact: wall time comparison.
  2. Optimal compact frequency (every N calls).
  3. Peak device pool usage (pool offset read-back; never decreases).
  4. How heap compacting affects peak memory usage across repeated calls.
  5. Notes on HEAP_NUM_ARENAS (compile-time constant, 64) effect on
     fragmentation and performance.

NOTE (results reflect an earlier heap_compact implementation):
  These findings were measured when heap_compact() re-inserted free blocks
  by address hash (heap_arena_for_addr) rather than by blockIdx.x % 64 —
  an arena mismatch that made routine compaction grow the pool on every call.
  heap_compact has since been changed to a no-op (free-block coalescing is
  handled inside heap_free), so run it only as a historical reference for the
  allocator behavior, not as guidance on a live compact() API.

  Original observation: pool usage is flat after the first call when blocks
  are freed to the correct arena by blockIdx.x % HEAP_NUM_ARENAS and reused
  in subsequent calls; the pool offset never grows after warm-up.

  heap_compact() is only useful in the (currently unused) case where you want
  to hold heap-allocated data alive across calls (e.g. a streaming pipeline
  that builds up results over multiple launches).

Run:
    python tests/bench_mm.py [--n_pts N] [--n_hulls H] [--n_rounds R]

Defaults: 200 points/hull, 64 hulls/call, 100 rounds.
"""

import argparse
import time
import numpy as np
import cuacd
import cuacd._gpu as _gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gaussian_pts(n, rng):
    return rng.standard_normal((n, 3)).astype(np.float32)


def run_hull_batch(ctx, pts_list):
    """Run one batch_hull_volume call; return elapsed seconds."""
    t0 = time.perf_counter()
    ctx.batch_hull_volume(pts_list)
    t1 = time.perf_counter()
    return t1 - t0


def run_compact(ctx):
    """Run heap_compact; return elapsed seconds."""
    t0 = time.perf_counter()
    ctx.heap_compact()
    t1 = time.perf_counter()
    return t1 - t0


def fmt_mb(n_bytes):
    return f"{n_bytes / (1 << 20):.1f} MB"


# ---------------------------------------------------------------------------
# Experiment 1: Compact every call vs never compact
# ---------------------------------------------------------------------------

def exp1_compact_every_vs_never(n_pts, n_hulls, n_rounds, rng):
    print("\n" + "=" * 70)
    print("Experiment 1: Compact after every call vs no compact")
    print(f"  {n_hulls} hulls/call × {n_pts} pts/hull × {n_rounds} rounds")
    print("=" * 70)

    pts_list = [make_gaussian_pts(n_pts, rng) for _ in range(n_hulls)]

    for label, compact_freq in [("never", 0), ("every call", 1)]:
        ctx = cuacd.Context(device=0)
        try:
            # Warm-up
            ctx.batch_hull_volume(pts_list)
            if compact_freq:
                ctx.heap_compact()

            pool_samples = []
            call_times = []
            compact_times = []

            for i in range(n_rounds):
                call_times.append(run_hull_batch(ctx, pts_list))
                pool_samples.append(ctx.pool_usage())

                if compact_freq and (i % compact_freq == 0):
                    compact_times.append(run_compact(ctx))

            peak_pool = max(pool_samples)
            total_call_s = sum(call_times)
            total_compact_s = sum(compact_times)
            mean_call_ms = np.mean(call_times) * 1000
            mean_compact_ms = (np.mean(compact_times) * 1000) if compact_times else 0.0

            print(f"\n  [{label}]")
            print(f"    hull_dandc mean:   {mean_call_ms:7.2f} ms/call")
            if compact_times:
                print(f"    heap_compact mean: {mean_compact_ms:7.2f} ms/call")
            print(f"    total wall time:   {(total_call_s + total_compact_s) * 1000:7.1f} ms")
            print(f"    peak pool usage:   {fmt_mb(peak_pool)} (after {n_rounds} rounds)")
        finally:
            ctx.close()


# ---------------------------------------------------------------------------
# Experiment 2: Compact frequency sweep
# ---------------------------------------------------------------------------

def exp2_compact_frequency(n_pts, n_hulls, n_rounds, rng):
    print("\n" + "=" * 70)
    print("Experiment 2: Compact frequency sweep")
    print(f"  {n_hulls} hulls/call × {n_pts} pts/hull × {n_rounds} rounds")
    print("=" * 70)

    pts_list = [make_gaussian_pts(n_pts, rng) for _ in range(n_hulls)]
    freqs = [1, 5, 10, 25, 50, n_rounds]  # every N calls; n_rounds = "once at end"

    print(f"\n  {'freq':>6}  {'call ms':>9}  {'compact ms':>11}  "
          f"{'total ms':>10}  {'peak pool':>12}  {'compacts':>9}")
    print("  " + "-" * 64)

    for freq in freqs:
        ctx = cuacd.Context(device=0)
        try:
            ctx.batch_hull_volume(pts_list)
            ctx.heap_compact()

            call_times = []
            compact_times = []
            pool_samples = []

            for i in range(n_rounds):
                call_times.append(run_hull_batch(ctx, pts_list))
                pool_samples.append(ctx.pool_usage())
                if (i + 1) % freq == 0:
                    compact_times.append(run_compact(ctx))

            peak = max(pool_samples)
            mean_call = np.mean(call_times) * 1000
            mean_cmp  = (np.mean(compact_times) * 1000) if compact_times else 0.0
            total_ms  = (sum(call_times) + sum(compact_times)) * 1000
            n_cmp     = len(compact_times)
            freq_label = f"1/{freq}" if freq < n_rounds else "once"

            print(f"  {freq_label:>6}  {mean_call:>9.2f}  {mean_cmp:>11.2f}  "
                  f"{total_ms:>10.1f}  {fmt_mb(peak):>12}  {n_cmp:>9}")
        finally:
            ctx.close()


# ---------------------------------------------------------------------------
# Experiment 3: Pool usage trajectory over many calls
# ---------------------------------------------------------------------------

def exp3_pool_usage_trajectory(n_pts, n_hulls, n_rounds, rng):
    print("\n" + "=" * 70)
    print("Experiment 3: Pool usage trajectory")
    print(f"  {n_hulls} hulls/call × {n_pts} pts/hull × {n_rounds} rounds")
    print("=" * 70)

    pts_list = [make_gaussian_pts(n_pts, rng) for _ in range(n_hulls)]

    ctx = cuacd.Context(device=0)
    try:
        usages_no_compact = []
        for _ in range(n_rounds):
            ctx.batch_hull_volume(pts_list)
            usages_no_compact.append(ctx.pool_usage())

        step = max(1, n_rounds // 10)
        print("\n  Pool usage (no compact) — sampled every", step, "calls:")
        for i in range(0, n_rounds, step):
            u = usages_no_compact[i]
            bar = "#" * int(u / max(usages_no_compact) * 40)
            print(f"    call {i+1:4d}: {fmt_mb(u):>10}  {bar}")

        delta = usages_no_compact[-1] - usages_no_compact[0]
        print(f"\n  Per-call pool growth: {fmt_mb(delta / n_rounds)} / call")
        print(f"  Final pool usage:     {fmt_mb(usages_no_compact[-1])}")
    finally:
        ctx.close()

    # Same with compact every call
    ctx = cuacd.Context(device=0)
    try:
        usages_compact = []
        for i in range(n_rounds):
            ctx.batch_hull_volume(pts_list)
            ctx.heap_compact()
            usages_compact.append(ctx.pool_usage())

        final_no = usages_no_compact[-1]
        final_cmp = usages_compact[-1]
        print(f"\n  Pool usage comparison after {n_rounds} rounds:")
        print(f"    No compact:   {fmt_mb(final_no)}")
        print(f"    With compact: {fmt_mb(final_cmp)}")
        print(f"    Delta:        {fmt_mb(final_no - final_cmp)} saved by compact")
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Experiment 4: Arena and compact interaction (analytical)
# ---------------------------------------------------------------------------

def exp4_arena_analysis():
    print("\n" + "=" * 70)
    print("Experiment 4: HEAP_NUM_ARENAS and compaction (analysis)")
    print("=" * 70)
    print("""
  HEAP_NUM_ARENAS = 64 (compile-time constant in cuda/heap_arena.cuh).

  Arena mismatch (root cause of compact causing pool growth):
    - heap_alloc:   arena = blockIdx.x % 64       (block-index based)
    - heap_free:    arena = blockIdx.x % 64       (block-index based)
    - heap_compact: re-inserts by heap_arena_for_addr(ptr) (address hash)
    After compact, blocks land in address-hash arenas, NOT blockIdx arenas.
    The next kernel call (same blockIdx) finds preferred arena empty →
    allocates a new slab from the pool. Pool grows indefinitely with compact.

  Effect on fragmentation without compact:
    - For plane_cut (1 block, blockIdx.x=0): all allocs/frees go to arena 0.
      No cross-arena fragmentation; memory is perfectly reused call-to-call.
    - For hull_dandc (N hulls, blockIdx.x = 0..N-1): each block owns its
      arena (for N ≤ 64). Memory cycles correctly without compact.

  Effect on performance (hull_dandc, N hulls):
    - N ≤ 64: each hull uses its own arena, zero lock contention.
    - N > 64: arenas shared by multiple warps → spin-lock contention.
    - Recommendation: HEAP_NUM_ARENAS ≥ typical n_hulls_per_batch.

  Compact performance (when actually needed):
    - Phase 1 (drain arenas): O(n_free_blocks), parallelized across 64 threads.
    - Phase 2 (sort by address): O(n * log²n) via warp_sort.
    - Phase 3 (coalesce + re-insert): O(n_free_blocks), serial in thread 0.
    - ~0.1 ms for 64-hull workloads. Compact overhead is not the concern;
      the arena mismatch pool growth is.
    """)


# ---------------------------------------------------------------------------
# Sweep: run all four experiments across the test_hull config space.
# Configs ordered from fewer hulls/large clouds → more hulls/small clouds
# (same range as test_hull.py benchmarks, gaussian distribution only).
# ---------------------------------------------------------------------------

# (n_hulls, n_pts, label) — ordered: fewer larger → more smaller
CONFIGS = [
    (   10, 20000, "10x20000pts"),
    (  100,  2000, "100x2000pts"),
    ( 1000,   200, "1000x200pts"),
    (10000,    20, "10000x20pts"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, default=100,
                        help="number of hull_dandc calls per config")
    parser.add_argument("--config", type=str, default=None,
                        help="run a single config by label (e.g. 100x2000pts)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    n_rounds = args.n_rounds

    configs = CONFIGS
    if args.config:
        configs = [c for c in CONFIGS if c[2] == args.config]
        if not configs:
            print(f"Unknown config {args.config!r}. Available: {[c[2] for c in CONFIGS]}")
            raise SystemExit(1)

    for n_hulls, n_pts, label in configs:
        print(f"\n{'#' * 70}")
        print(f"# Config: {label}  ({n_hulls} hulls × {n_pts} pts, {n_rounds} rounds, gaussian)")
        print(f"{'#' * 70}")

        exp1_compact_every_vs_never(n_pts, n_hulls, n_rounds, rng)
        exp2_compact_frequency(n_pts, n_hulls, n_rounds, rng)
        exp3_pool_usage_trajectory(n_pts, n_hulls, n_rounds, rng)

    exp4_arena_analysis()
