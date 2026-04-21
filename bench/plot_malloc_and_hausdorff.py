#!/usr/bin/env python3
"""Two-panel figure:
   Left:  coacd heap vs CUDA device-malloc throughput (small-medium regime).
   Right: Hausdorff batch=100 CPU vs GPU bars for bunny and 49160, with speedup labels.

Output: malloc_and_hausdorff.pdf
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()
plt.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
})

fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.0))

# ---------------------------------------------------------------------------
# Subplot 0: malloc throughput
# ---------------------------------------------------------------------------
grids = [1, 4, 16, 64, 256, 1024]
ours_mops = [0.3521, 1.407, 5.540, 21.32, 21.91, 24.47]
cuda_mops = [0.0705, 0.4044, 1.161, 1.846, 2.271, 2.397]

ax0 = axes[0]
ax0.plot(grids, ours_mops, marker="o", label="Ours")
ax0.plot(grids, cuda_mops, marker="s", label="CUDA")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlabel("Blocks")
ax0.set_ylabel("Mops / s")
ax0.set_xticks(grids)
ax0.set_xticklabels([str(g) for g in grids])
ax0.legend(frameon=False)
ax0.text(0.5, -0.28, "(a)", transform=ax0.transAxes, ha="center", va="top", fontsize=11)

# ---------------------------------------------------------------------------
# Subplot 1: Hausdorff batch=100 CPU vs GPU
# ---------------------------------------------------------------------------
ax1 = axes[1]
datasets = ["bunny", "vase"]
cpu_ms = [860.9756, 397.2055]
gpu_ms = [29.9832, 7.9617]
speedups = [c / g for c, g in zip(cpu_ms, gpu_ms)]

x = np.arange(len(datasets))
width = 0.35

bars_cpu = ax1.bar(x - width/2, cpu_ms, width, label="CPU")
bars_gpu = ax1.bar(x + width/2, gpu_ms, width, label="GPU")

ax1.set_ylabel("Time (ms)")
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.legend(frameon=False)
ax1.set_yscale("log")

# Annotate with speedup ratio directly above the GPU bar.
for i, (xi, su) in enumerate(zip(x, speedups)):
    ax1.annotate(f"↓{su:.1f}×",
                 xy=(xi + width / 2, gpu_ms[i]),
                 xytext=(0, 4),
                 textcoords="offset points",
                 ha="center", va="bottom",
                 fontsize=9, color="#333333")

ax1.text(0.5, -0.28, "(b)", transform=ax1.transAxes, ha="center", va="top", fontsize=11)

fig.tight_layout()
fig.savefig("malloc_and_hausdorff.pdf", bbox_inches="tight")
print("wrote malloc_and_hausdorff.pdf")
