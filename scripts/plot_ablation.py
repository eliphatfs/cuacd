#!/usr/bin/env python3
"""Plot ablation study results: average parts and average time vs width.

Generates a single PDF with 2×2 subplots: rows = parts / time, cols = nodc / dc,
with lines for each (n_concave_edges, concave_iters) combination.
"""

import json
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = pathlib.Path("decomp_output/vhacd2_data_r0.1_mv10k_ablations")
WIDTHS = [15, 30, 60, 120, 240]
N_CONCAVE_EDGES = [0, 4, 16, 32, 64]
CONCAVE_ITERS = [1, 3, 5, 10, 20]
DC_VALUES = [False, True]

LINE_PATTERN = re.compile(r"^(.+\.obj)\s+([\d.]+)s\s+(\d+)\s+parts$", re.MULTILINE)


def parse_run_log(log_path):
    """Parse a run.log, return list of (mesh_name, seconds, n_parts)."""
    text = log_path.read_text()
    return [(m.group(1), float(m.group(2)), int(m.group(3)))
            for m in LINE_PATTERN.finditer(text)]


def gather_data():
    """Return dict: (dc, nce, ci) -> {width: (avg_parts, avg_time, n_meshes)}"""
    data = {}
    for dc in DC_VALUES:
        dc_tag = "dc" if dc else "nodc"
        for nce in N_CONCAVE_EDGES:
            ci_list = [1] if nce == 0 else CONCAVE_ITERS
            for ci in ci_list:
                key = (dc, nce, ci)
                data[key] = {}
                ci_tag = f"ci{ci}" if nce > 0 else "ci0"
                for w in WIDTHS:
                    dirname = f"w{w}_nc{nce}_{dc_tag}_{ci_tag}"
                    meta_path = BASE / dirname / "meta.json"
                    log_path = BASE / dirname / "run.log"
                    if not meta_path.exists() or not log_path.exists():
                        continue
                    entries = parse_run_log(log_path)
                    if not entries:
                        continue
                    avg_parts = np.mean([e[2] for e in entries])
                    avg_time = np.mean([e[1] for e in entries])
                    data[key][w] = (avg_parts, avg_time, len(entries))
    return data


def make_label(nce, ci):
    if nce == 0:
        return "nc=0"
    return f"nc={nce}, ci={ci}"


def plot_all(data, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)

    metrics = [(0, "Avg. number of parts"), (1, "Avg. time (s)")]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h", "<", ">", "d"]
    linestyles = ["-", "--", "-.", ":",
                  (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)),
                  (0, (3, 2, 1, 2)), (0, (5, 1)), (0, (3, 1)),
                  (0, (1, 2)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    for row, (metric_idx, ylabel) in enumerate(metrics):
        for col, dc in enumerate(DC_VALUES):
            ax = axes[row, col]
            line_idx = 0
            for nce in N_CONCAVE_EDGES:
                ci_list = [1] if nce == 0 else CONCAVE_ITERS
                for ci in ci_list:
                    key = (dc, nce, ci)
                    if key not in data:
                        continue
                    ws = sorted(data[key].keys())
                    ys = [data[key][w][metric_idx] for w in ws]
                    label = make_label(nce, ci)
                    ax.plot(ws, ys,
                            label=label,
                            marker=markers[line_idx % len(markers)],
                            linestyle=linestyles[line_idx % len(linestyles)],
                            markersize=6, linewidth=1.5)
                    line_idx += 1

            ax.set_xlabel("Width")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS)
            ax.set_xticklabels([str(w) for w in WIDTHS])
            dc_label = "decompose-components-per-iter" if dc else "no decompose-components-per-iter"
            ax.set_title(f"{ylabel} — {dc_label}")
            ax.legend(fontsize=8, ncol=2, loc="best")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved {filename}")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    data = gather_data()
    plot_all(data, BASE / "ablation.pdf")


if __name__ == "__main__":
    main()
