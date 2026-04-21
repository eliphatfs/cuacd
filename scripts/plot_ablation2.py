#!/usr/bin/env python3
"""Plot ablation study #2 results and generate markdown table.

Plot: width (x-axis) vs average time / average number of parts (y-axis, 2
subplots). Curves represent different nc values. Fixed for the plot:
width2=5, dc=periter (default), ci=10.

Also emits a markdown table comparing a handful of specific settings.
"""

import pathlib
import re
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


BASE = pathlib.Path("decomp_output/vhacd2_data_r0.1_mv10k_ablations2")
WIDTHS = [9, 15, 30, 45, 90]
N_CONCAVE_EDGES = [0, 4, 8, 16, 32]

sns.set_theme()
plt.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
})

LINE_PATTERN = re.compile(
    r"^(.+\.(?:obj|stl|ply|off|glb|gltf))\s+([\d.]+)s\s+(\d+)\s+parts$",
    re.MULTILINE,
)


def parse_run_log(log_path):
    text = log_path.read_text()
    return [(m.group(1), float(m.group(2)), int(m.group(3))) for m in LINE_PATTERN.finditer(text)]


def gather_plot_data():
    """Return dict: nc -> {width: (avg_parts, avg_time, n_meshes)}."""
    width2 = 5
    dc_tag = "periter"
    ci = 10
    data = {}
    for nc in N_CONCAVE_EDGES:
        data[nc] = {}
        ci_tag = f"ci{ci}" if nc > 0 else "ci0"
        for w in WIDTHS:
            dirname = f"w{w}_w2{width2}_nc{nc}_{dc_tag}_{ci_tag}"
            meta_path = BASE / dirname / "meta.json"
            log_path = BASE / dirname / "run.log"
            if not meta_path.exists() or not log_path.exists():
                continue
            entries = parse_run_log(log_path)
            if not entries:
                continue
            avg_parts = float(np.mean([e[2] for e in entries]))
            avg_time = float(np.mean([e[1] for e in entries]))
            data[nc][w] = (avg_parts, avg_time, len(entries))
    return data


def plot_all(data, filename):
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.0))

    metrics = [
        (1, "Avg. time (s)"),
        (0, "Avg. # of parts"),
    ]

    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for idx, (metric_idx, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for line_idx, nc in enumerate(N_CONCAVE_EDGES):
            if nc not in data:
                continue
            ws = sorted(data[nc].keys())
            if not ws:
                continue
            ys = [data[nc][w][metric_idx] for w in ws]
            label = f"$N_{{ce}}={nc}$"
            ax.plot(
                ws, ys,
                label=label,
                marker=markers[line_idx % len(markers)],
                linestyle=linestyles[line_idx % len(linestyles)],
                markersize=5,
                linewidth=1.5,
            )
        ax.set_xlabel("Width")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WIDTHS)
        ax.set_xticklabels([str(w) for w in WIDTHS])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, ncol=2, loc="best")
        ax.text(0.5, -0.28, f"({chr(ord('a') + idx)})", transform=ax.transAxes,
                ha="center", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved {filename}")


def get_setting_data(width, width2, nc, dc_per_iter, ci):
    dc_tag = "noperiter" if dc_per_iter else "periter"
    ci_tag = f"ci{ci}" if nc > 0 else "ci0"
    dirname = f"w{width}_w2{width2}_nc{nc}_{dc_tag}_{ci_tag}"
    log_path = BASE / dirname / "run.log"
    meta_path = BASE / dirname / "meta.json"
    if not log_path.exists() or not meta_path.exists():
        return None, None, 0
    entries = parse_run_log(log_path)
    if not entries:
        return None, None, 0
    avg_time = float(np.mean([e[1] for e in entries]))
    avg_parts = float(np.mean([e[2] for e in entries]))
    return avg_time, avg_parts, len(entries)


def generate_markdown_table():
    rows = [
        ("Default", 30, 5, 16, False, 10),
        ("No decompose components per iter", 30, 5, 16, True, 10),
        ("Fewer CI (3)", 30, 5, 16, False, 3),
        ("More CI (20)", 30, 5, 16, False, 20),
        ("Fewer width2 (3)", 30, 3, 16, False, 10),
        ("More width2 (15)", 30, 15, 16, False, 10),
    ]

    lines = []
    lines.append("| Setting | Avg. Time (s) | Avg. Parts | N meshes |")
    lines.append("|---------|---------------|------------|----------|")
    for name, w, w2, nc, dc, ci in rows:
        t, p, n = get_setting_data(w, w2, nc, dc, ci)
        t_str = f"{t:.2f}" if t is not None else "N/A"
        p_str = f"{p:.2f}" if p is not None else "N/A"
        n_str = str(n)
        lines.append(f"| {name} | {t_str} | {p_str} | {n_str} |")

    return "\n".join(lines)


def main():
    data = gather_plot_data()
    if not data or all(not v for v in data.values()):
        print("No plot data found.", file=sys.stderr)
        sys.exit(1)

    pdf_path = BASE / "ablation2.pdf"
    plot_all(data, pdf_path)

    table_md = generate_markdown_table()
    md_path = BASE / "ablation2_table.md"
    md_path.write_text(table_md + "\n")
    print(f"Saved {md_path}")
    print()
    print(table_md)


if __name__ == "__main__":
    main()
