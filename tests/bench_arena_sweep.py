"""bench_arena_sweep.py — Rebuild and benchmark across different HEAP_NUM_ARENAS values.

For each arena count in [32, 64, 128, 256]:
  1. Rebuild the extension with COACD_GPU_ARENAS=N
  2. Run the GPU hull benchmark (test_hull.py standalone)
  3. Parse the benchmark table
  4. Write a summary to docs/arena_sweep.md

Usage:
    python tests/bench_arena_sweep.py [--arenas 32,64,128,256] [--output docs/arena_sweep.md]
"""

import argparse
import subprocess
import sys
import os
import re
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(n_arenas):
    """Rebuild the extension with HEAP_NUM_ARENAS=n_arenas."""
    env = os.environ.copy()
    env["COACD_GPU_ARENAS"] = str(n_arenas)
    print(f"\n[sweep] Building with HEAP_NUM_ARENAS={n_arenas} ...", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd=_ROOT, env=env,
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  BUILD FAILED (exit {result.returncode})")
        print(result.stderr[-2000:])
        return False
    print(f"  OK ({elapsed:.1f}s)", flush=True)
    return True


def run_benchmark():
    """Run test_hull.py standalone and return its stdout."""
    print(f"[sweep] Running benchmark ...", flush=True)
    result = subprocess.run(
        [sys.executable, "tests/test_hull.py"],
        cwd=_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  BENCHMARK FAILED")
        print(result.stderr[-2000:])
        return None
    return result.stdout


def parse_gpu_table(output):
    """Extract the GPU hull benchmark table rows from test_hull.py output.

    Looks for lines of the form:
      Config               Distribution     GPU ms   mean vol  peak pool MB
    """
    rows = []
    in_table = False
    for line in output.splitlines():
        if "GPU ms" in line and "mean vol" in line:
            in_table = True
            continue
        if in_table:
            line = line.strip()
            if not line or line.startswith("-"):
                continue
            # Stop at next blank section
            if line.startswith("#") or line.startswith("="):
                break
            rows.append(line)
    return rows


def fmt_table_md(arena_rows):
    """arena_rows: list of (n_arenas, rows) where rows are parsed benchmark lines."""
    # Collect all unique (config, dist) keys preserving order
    all_keys = []
    seen = set()
    for _, rows in arena_rows:
        for r in rows:
            parts = r.split()
            if len(parts) >= 5:
                key = (parts[0], parts[1])
                if key not in seen:
                    seen.add(key)
                    all_keys.append(key)

    # Build dict: (n_arenas, config, dist) -> (gpu_ms, mean_vol, peak_mb)
    data = {}
    for n, rows in arena_rows:
        for r in rows:
            parts = r.split()
            if len(parts) >= 5:
                cfg, dist = parts[0], parts[1]
                try:
                    gpu_ms   = float(parts[2])
                    mean_vol = float(parts[3])
                    peak_mb  = float(parts[4])
                    data[(n, cfg, dist)] = (gpu_ms, mean_vol, peak_mb)
                except ValueError:
                    pass

    arenas = [n for n, _ in arena_rows]

    # Header
    cols = ["Config", "Distribution"] + [f"A={n} ms" for n in arenas] + [f"A={n} pool MB" for n in arenas]
    sep  = ["---"] * len(cols)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for cfg, dist in all_keys:
        ms_cells   = []
        pool_cells = []
        for n in arenas:
            entry = data.get((n, cfg, dist))
            if entry:
                ms_cells.append(f"{entry[0]:.1f}")
                pool_cells.append(f"{entry[2]:.1f}")
            else:
                ms_cells.append("—")
                pool_cells.append("—")
        lines.append("| " + " | ".join([cfg, dist] + ms_cells + pool_cells) + " |")

    return "\n".join(lines)


def write_report(arena_rows, output_path):
    arenas = [n for n, _ in arena_rows]

    lines = []
    lines.append("# Arena Count Sweep")
    lines.append("")
    lines.append("GPU hull D&C benchmark across `HEAP_NUM_ARENAS` ∈ {" + ", ".join(str(n) for n in arenas) + "}.")
    lines.append("Metric: wall-clock time for one `batch_hull_volume` call (after warm-up) and peak pool usage.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(fmt_table_md(arena_rows))
    lines.append("")
    lines.append("## Raw Output")
    lines.append("")
    for n, rows in arena_rows:
        lines.append(f"### HEAP_NUM_ARENAS = {n}")
        lines.append("")
        lines.append("```")
        for r in rows:
            lines.append(r)
        lines.append("```")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[sweep] Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arenas", default="32,64,128,256",
                        help="comma-separated list of arena counts to sweep")
    parser.add_argument("--output", default=os.path.join(_ROOT, "docs", "arena_sweep.md"),
                        help="output markdown file")
    args = parser.parse_args()

    arena_list = [int(x.strip()) for x in args.arenas.split(",")]
    print(f"[sweep] Arena sweep: {arena_list}")

    arena_rows = []
    for n in arena_list:
        ok = build(n)
        if not ok:
            arena_rows.append((n, [f"BUILD FAILED"]))
            continue
        output = run_benchmark()
        if output is None:
            arena_rows.append((n, ["BENCHMARK FAILED"]))
            continue
        # Print raw output for visibility
        for line in output.splitlines():
            if "GPU ms" in line or "---" in line or any(
                    c in line for c in ["10000x", "1000x", "100x", "10x"]):
                print(" ", line)
        rows = parse_gpu_table(output)
        arena_rows.append((n, rows))

    write_report(arena_rows, args.output)

    # Restore default build (64 arenas)
    print(f"\n[sweep] Restoring default build (HEAP_NUM_ARENAS=64) ...")
    build(64)


if __name__ == "__main__":
    main()
