#!/usr/bin/env python3
"""Ablation study driver #2 — runs only the combinations needed for the
specified plot and table.

Plot needs: width × nc (fixed width2=5, per-iter, ci=10)
Table needs: 6 specific rows varying dc, ci, and width2 around defaults.

Output:
    decomp_output/vhacd2_data_r0.1_mv10k_ablations2/<param_encoding>/
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time


INPUT_DIR = "tests/data/vhacd2_data_r0.1_mv10k"
OUTPUT_BASE = "decomp_output/vhacd2_data_r0.1_mv10k_ablations2"


def encode_params(width, width2, n_concave_edges, dc_per_iter, concave_iters):
    dc_str = "noperiter" if dc_per_iter else "periter"
    ci_str = f"ci{concave_iters}" if n_concave_edges > 0 else "ci0"
    return f"w{width}_w2{width2}_nc{n_concave_edges}_{dc_str}_{ci_str}"


def get_required_configs():
    """Return list of (width, width2, nc, dc_per_iter, ci) dicts."""
    configs = []

    # Plot data: width × nc, fixed at default width2=5, per-iter, ci=10
    for w in (9, 15, 30, 45, 90):
        for nc in (0, 4, 8, 16, 32):
            ci = 10
            configs.append(dict(width=w, width2=5, n_concave_edges=nc,
                                no_decompose_components_per_iter=False,
                                concave_iters=ci))

    # Table rows (skip duplicates already in plot set)
    table_rows = [
        (30, 5, 16, True, 10),   # No decompose
        (30, 5, 16, False, 3),   # Fewer CI
        (30, 5, 16, False, 20),  # More CI
        (30, 3, 16, False, 10),  # Fewer width2
        (30, 15, 16, False, 10), # More width2
    ]
    seen = set()
    for c in configs:
        seen.add((c["width"], c["width2"], c["n_concave_edges"],
                  c["no_decompose_components_per_iter"], c["concave_iters"]))
    for w, w2, nc, dc, ci in table_rows:
        key = (w, w2, nc, dc, ci)
        if key not in seen:
            configs.append(dict(width=w, width2=w2, n_concave_edges=nc,
                                no_decompose_components_per_iter=dc,
                                concave_iters=ci))

    return configs


def config_already_done(config):
    name = encode_params(
        width=config["width"],
        width2=config["width2"],
        n_concave_edges=config["n_concave_edges"],
        dc_per_iter=config["no_decompose_components_per_iter"],
        concave_iters=config["concave_iters"],
    )
    output_dir = pathlib.Path(OUTPUT_BASE) / name
    return output_dir.exists() and (output_dir / "run.log").exists()


def run_config(config, dry_run=False):
    params = config
    name = encode_params(
        width=params["width"],
        width2=params["width2"],
        n_concave_edges=params["n_concave_edges"],
        dc_per_iter=params["no_decompose_components_per_iter"],
        concave_iters=params["concave_iters"],
    )
    output_dir = pathlib.Path(OUTPUT_BASE) / name

    cmd = [
        sys.executable, "-m", "cuacd.cli",
        INPUT_DIR, str(output_dir),
        "--width", str(params["width"]),
        "--width2", str(params["width2"]),
        "--n-concave-edges", str(params["n_concave_edges"]),
        "--concave-iters", str(params["concave_iters"]),
    ]
    if params["no_decompose_components_per_iter"]:
        cmd.append("--no-decompose-components-per-iter")

    if dry_run:
        print(f"  [DRY RUN] {name}")
        print(f"            {' '.join(cmd)}")
        return name, 0.0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    t0 = time.perf_counter()
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0

    meta = {
        "config": params,
        "name": name,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.datetime.now().isoformat(),
        "command": cmd,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"  [{status}] {name}  {elapsed:.1f}s")
    return name, elapsed, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Ablation study driver #2 for cuacd")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N configurations")
    parser.add_argument("--only-new", action="store_true", help="Skip configurations that already have results")
    args = parser.parse_args()

    configs = get_required_configs()

    if args.only_new:
        before = len(configs)
        configs = [c for c in configs if not config_already_done(c)]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipping {skipped} already-completed configurations")

    total = len(configs)

    print(f"Ablation study #2: {total} configurations")
    print(f"  Input:               {INPUT_DIR}")
    print(f"  Output base:         {OUTPUT_BASE}")
    print()

    if args.start_from > 0:
        print(f"Resuming from configuration {args.start_from} (skipping first {args.start_from})")
        configs = configs[args.start_from:]

    results = []
    t_start = time.perf_counter()

    for i, config in enumerate(configs, start=1):
        global_i = i + args.start_from
        name, elapsed, rc = run_config(config, dry_run=args.dry_run)
        results.append((name, elapsed, rc))

        if not args.dry_run and i > 0:
            avg_so_far = (time.perf_counter() - t_start) / i
            remaining = avg_so_far * (len(configs) - i)
            print(f"  Progress: {global_i}/{total}  ~{remaining/60:.0f} min remaining")
            print()

    total_elapsed = time.perf_counter() - t_start

    print("=" * 60)
    print("ABLATION STUDY #2 SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {len(results)}")
    print(f"Total time:           {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    successes = [r for r in results if r[2] == 0]
    failures = [r for r in results if r[2] != 0]
    print(f"Successes:            {len(successes)}")
    print(f"Failures:             {len(failures)}")

    if failures:
        print("\nFailed configurations:")
        for name, elapsed, rc in failures:
            print(f"  {name}  (rc={rc})")

    summary_path = pathlib.Path(OUTPUT_BASE) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_configs": len(results),
        "total_seconds": round(total_elapsed, 2),
        "successes": len(successes),
        "failures": len(failures),
        "results": [
            {"name": n, "elapsed": e, "returncode": rc} for n, e, rc in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
