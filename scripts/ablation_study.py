#!/usr/bin/env python3
"""Ablation study driver for coacd-gpu lookahead decomposition.

Sweeps over combinations of width, n_concave_edges, no_decompose_components_per_iter,
and concave_iters. Each configuration gets its own output directory under
decomp_output/vhacd2_data_r0.1_mv10k_ablations/<param_encoding>/ with the GLB
results and a run log.

Usage:
    python scripts/ablation_study.py [--dry-run] [--only-new] [--dc-only]
"""

import argparse
import datetime
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time


INPUT_DIR = "tests/data/vhacd2_data_r0.1_mv10k"
OUTPUT_BASE = "decomp_output/vhacd2_data_r0.1_mv10k_ablations"

WIDTHS = [15, 30, 60, 120, 240]
N_CONCAVE_EDGES = [0, 4, 16, 32, 64]
DECOMPOSE_COMPONENTS_PER_ITER = [False, True]
CONCAVE_ITERS = [1, 3, 5, 10, 20]  # only used when n_concave_edges > 0


def encode_params(width, n_concave_edges, dc_per_iter, concave_iters):
    """Encode parameters into a directory-friendly string."""
    dc_str = "dc" if dc_per_iter else "nodc"
    ci_str = f"ci{concave_iters}" if n_concave_edges > 0 else "ci0"
    return f"w{width}_nc{n_concave_edges}_{dc_str}_{ci_str}"


def build_configs(dc_only=False):
    """Build the full list of ablation configurations."""
    dc_values = [True] if dc_only else DECOMPOSE_COMPONENTS_PER_ITER
    configs = []
    for width, nce, dc in itertools.product(WIDTHS, N_CONCAVE_EDGES, dc_values):
        if nce == 0:
            configs.append({
                "width": width,
                "n_concave_edges": nce,
                "no_decompose_components_per_iter": dc,
                "concave_iters": 1,  # irrelevant when nce=0, use default
            })
        else:
            for ci in CONCAVE_ITERS:
                configs.append({
                    "width": width,
                    "n_concave_edges": nce,
                    "no_decompose_components_per_iter": dc,
                    "concave_iters": ci,
                })
    return configs


def config_already_done(config):
    """Check if a config's output directory already exists with a run.log."""
    name = encode_params(
        width=config["width"],
        n_concave_edges=config["n_concave_edges"],
        dc_per_iter=config["no_decompose_components_per_iter"],
        concave_iters=config["concave_iters"],
    )
    output_dir = pathlib.Path(OUTPUT_BASE) / name
    return output_dir.exists() and (output_dir / "run.log").exists()


def run_config(config, dry_run=False):
    """Run a single ablation configuration. Returns (config_name, elapsed_seconds, returncode)."""
    params = config
    name = encode_params(
        width=params["width"],
        n_concave_edges=params["n_concave_edges"],
        dc_per_iter=params["no_decompose_components_per_iter"],
        concave_iters=params["concave_iters"],
    )
    output_dir = pathlib.Path(OUTPUT_BASE) / name

    cmd = [
        sys.executable, "-m", "coacd_gpu.cli",
        INPUT_DIR, str(output_dir),
        "--width", str(params["width"]),
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

    # Write a metadata file alongside the results
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
    parser = argparse.ArgumentParser(description="Ablation study driver for coacd-gpu")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N configurations (for resuming)")
    parser.add_argument("--only-new", action="store_true", help="Skip configurations that already have results")
    parser.add_argument("--dc-only", action="store_true", help="Only run no-decompose-components-per-iter configs")
    args = parser.parse_args()

    configs = build_configs(dc_only=args.dc_only)

    if args.only_new:
        before = len(configs)
        configs = [c for c in configs if not config_already_done(c)]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipping {skipped} already-completed configurations")

    total = len(configs)

    print(f"Ablation study: {total} configurations")
    print(f"  Widths:              {WIDTHS}")
    print(f"  N concave edges:     {N_CONCAVE_EDGES}")
    print(f"  DC per iter:         {DECOMPOSE_COMPONENTS_PER_ITER}")
    print(f"  Concave iters:       {CONCAVE_ITERS} (only when n_concave_edges > 0)")
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

        # Progress estimate
        if not args.dry_run and i > 0:
            avg_so_far = (time.perf_counter() - t_start) / i
            remaining = avg_so_far * (len(configs) - i)
            print(f"  Progress: {global_i}/{total}  ~{remaining/60:.0f} min remaining")
            print()

    total_elapsed = time.perf_counter() - t_start

    # Summary
    print("=" * 60)
    print("ABLATION STUDY SUMMARY")
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

    # Write overall summary
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
