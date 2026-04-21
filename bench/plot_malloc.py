#!/usr/bin/env python3
"""Plot coacd heap vs CUDA device-malloc throughput (small-medium regime).

Output: malloc_throughput.pdf, sized for ~80% of a column in a two-column
letter paper (col ~3.375in -> ~2.7in wide).
"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
})

grids = [1, 4, 16, 64, 256, 1024]
# ops/s from ./malloc_test_bench (size=[4096, 262144], slots=16, iters=1024)
ours_mops = [0.3521, 1.407, 5.540, 21.32, 21.91, 24.47]
cuda_mops = [0.0705, 0.4044, 1.161, 1.846, 2.271, 2.397]

fig, ax = plt.subplots(figsize=(3.04, 2.35))
ax.plot(grids, ours_mops, marker="o", label="Ours")
ax.plot(grids, cuda_mops, marker="s", label="CUDA")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Blocks")
ax.set_ylabel("Mops / s")
ax.set_xticks(grids)
ax.set_xticklabels([str(g) for g in grids])
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("malloc_throughput.pdf", bbox_inches="tight")
print("wrote malloc_throughput.pdf")
