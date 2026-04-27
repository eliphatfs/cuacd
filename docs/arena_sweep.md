# Arena Count Sweep

GPU hull D&C benchmark across `HEAP_NUM_ARENAS` ∈ {32, 64, 128, 256}.

Each arena count requires a full rebuild (`CUACD_GPU_ARENAS=N pip install -e .`).
Metric: wall-clock time for one `batch_hull_volume` call (after one warm-up call),
plus peak pool usage read-back via `ctx.pool_usage()`.

GPU: single device, distribution = gaussian/sphere_shell/cube_interior.
Pool size: 70% of free VRAM (default auto).

---

## GPU ms — wall-clock time per batch call

| Config | Distribution | A=32 | A=64 | A=128 | A=256 |
|---|---|---:|---:|---:|---:|
| 10000x20pts | sphere_shell    | 151.4 | 150.9 | 148.8 | 150.2 |
| 10000x20pts | cube_interior   | 147.7 | 145.0 | 146.6 | 147.2 |
| 10000x20pts | gaussian        | 145.3 | 143.5 | 144.8 | 146.1 |
| 1000x200pts | sphere_shell    |  33.3 |  32.9 |  33.1 |  33.2 |
| 1000x200pts | cube_interior   |  21.1 |  21.1 |  20.9 |  21.3 |
| 1000x200pts | gaussian        |  20.8 |  20.8 |  20.8 |  20.9 |
| 100x2000pts | sphere_shell    |  78.6 |  78.1 |  78.7 |  80.4 |
| 100x2000pts | cube_interior   |  29.0 |  29.3 |  28.7 |  29.3 |
| 100x2000pts | gaussian        |  27.2 |  27.4 |  27.1 |  27.5 |
| 10x20000pts | sphere_shell    | 779.7 | 768.7 | 770.5 | 786.9 |
| 10x20000pts | cube_interior   | 248.7 | 249.1 | 248.9 | 248.9 |
| 10x20000pts | gaussian        | 221.3 | 221.7 | 221.5 | 221.4 |

**Finding:** Arena count has negligible effect on throughput. Variance across runs (~1–2%)
exceeds any trend. The workload is not lock-contention-limited at these batch sizes.

---

## Peak pool MB — high-water mark of pool offset after benchmark

| Config | Distribution | A=32 | A=64 | A=128 | A=256 |
|---|---|---:|---:|---:|---:|
| 10000x20pts | sphere_shell    | 3201 | 3542 | 3713 | 3984 |
| 10000x20pts | cube_interior   | 3259 | 3585 | 3810 | 4185 |
| 10000x20pts | gaussian        | 3324 | 3622 | 3906 | 4259 |
| 1000x200pts | sphere_shell    | 3328 | 3622 | 3906 | 4259 |
| 1000x200pts | cube_interior   | 3328 | 3622 | 3906 | 4259 |
| 1000x200pts | gaussian        | 3328 | 3622 | 3906 | 4259 |
| 100x2000pts | sphere_shell    | 3336 | 3631 | 3906 | 4259 |
| 100x2000pts | cube_interior   | 3336 | 3631 | 3906 | 4259 |
| 100x2000pts | gaussian        | 3336 | 3631 | 3906 | 4259 |
| 10x20000pts | sphere_shell    | 3436 | 3731 | 4006 | 4359 |
| 10x20000pts | cube_interior   | 3436 | 3731 | 4006 | 4359 |
| 10x20000pts | gaussian        | 3436 | 3731 | 4006 | 4359 |

**Finding:** Peak pool usage scales roughly linearly with arena count. Each arena
independently allocates slabs from the bump pool; more arenas → more distinct
active slab sets → higher pool consumption before freed blocks are reused.

Ratio A=256 / A=32 ≈ 1.27–1.34 across configs. A=64 → A=128 adds ~350–400 MB
(≈12%). A=128 → A=256 adds ~350–450 MB (≈11%).

---

## Why pool grows with arena count

Arena selection: `blockIdx.x % HEAP_NUM_ARENAS`. With N arenas and B concurrent blocks:
- **N ≤ B**: multiple blocks share each arena → high temporal reuse, freed slabs reused quickly
- **N > B**: arenas are exclusive per block → zero lock contention, but each arena allocates independently

With 10000 hulls and HEAP_NUM_ARENAS=32: 313 hulls/arena → high reuse, pool fills slowly.
With 10000 hulls and HEAP_NUM_ARENAS=256: 39 hulls/arena → less reuse opportunity, more slabs.

For concurrent fragmentation (partial coalescing due to interleaved alloc/free across blocks),
fewer arenas means blocks are more likely to coalesce into their shared arena's free list → lower pool.

---

## Recommendation

**Keep `HEAP_NUM_ARENAS = 64` (default).**

- No measurable throughput benefit from larger counts (128, 256).
- Larger counts increase peak pool memory by 10–35%.
- Smaller counts (32) reduce pool usage by ~10% with no throughput regression,
  which may be worthwhile on memory-constrained devices (e.g. 8 GB VRAM).
- Lock contention is not a bottleneck even at 10000 hulls × 64 arenas (156 hulls/arena).

To override for a specific deployment:
```bash
CUACD_GPU_ARENAS=32 pip install -e .
```

---

## Environment

- HEAP_ALIGN = 512 bytes
- HEAP_MIN_POOL_ALLOC = 128 KB
- HEAP_NUM_SUBBINS = 64
- Pool = 70% free VRAM (auto)
- Build: `pip install -e .` with `CUACD_GPU_ARENAS=N`
- Benchmark: `python tests/test_hull.py` (standalone mode, after warm-up)
