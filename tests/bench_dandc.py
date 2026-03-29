#!/usr/bin/env python3
"""Standalone D&C hull benchmark for NCU profiling.

Usage:
    # Basic run:
    python tests/bench_dandc.py

    # Custom params:
    python tests/bench_dandc.py --n_pts 500 --n_hulls 32 --seed 42

    # Profile with NCU:
    ncu --set full -o dandc_profile python tests/bench_dandc.py --n_pts 200 --n_hulls 8
"""
import argparse
import numpy as np
import coacd_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pts", type=int, default=200, help="Points per hull")
    parser.add_argument("--n_hulls", type=int, default=8, help="Number of hulls in batch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pts_list = [rng.standard_normal((args.n_pts, 3)).astype(np.float32)
                for _ in range(args.n_hulls)]

    ctx = coacd_gpu.Context(device=0)

    # Warmup (context init, module load, etc.)
    ctx.batch_hull_volume(pts_list[:1], algo=2)

    # Profiled run
    vols, errs = ctx.batch_hull_volume(pts_list, algo=2)

    print(f"n_hulls={args.n_hulls}  n_pts={args.n_pts}")
    for i in range(args.n_hulls):
        print(f"  hull {i}: vol={vols[i]:.4f}  err={errs[i]}")

    ctx.close()

if __name__ == "__main__":
    main()
