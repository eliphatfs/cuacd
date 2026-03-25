"""Greedy agglomerative merge of convex hulls.

Translates CoACD/src/process.cpp MergeConvexHulls.
"""

import math
import numpy as np

from ._mesh import Mesh
from ._cost import (
    compute_hcost, compute_hcost_3, mesh_dist, merge_meshes,
    INF,
)


def _merge_ch(ch1: Mesh, ch2: Mesh) -> Mesh:
    """Merge two convex hulls and recompute hull."""
    merged = merge_meshes(ch1, ch2)
    return merged.convex_hull()


def _tri_index(p1, p2):
    """Map (p1, p2) to flat upper-triangle index. p1 > p2."""
    return (p1 * (p1 - 1)) // 2 + p2


def merge_convex_hulls(meshs, cvxs, params, gpu_ctx=None):
    """Greedy agglomerative merge.

    Args:
        meshs: list of Mesh (original part meshes, for pre-cost)
        cvxs: list of Mesh (convex hulls to merge)
        params: dict with threshold, rv_k, resolution, seed, max_convex_hull
        gpu_ctx: optional coacd_gpu.Context

    Returns:
        list of Mesh (merged convex hulls)
    """
    threshold = params["threshold"]
    rv_k = params["rv_k"]
    resolution = params["resolution"]
    seed = params["seed"]
    max_ch = params.get("max_convex_hull", -1)

    n = len(cvxs)
    if n <= 1:
        return cvxs

    # Flatten upper triangle cost matrix
    bound = (n * (n - 1)) // 2
    cost_matrix = np.full(bound, INF)
    pre_cost_matrix = np.full(bound, 0.0)

    for idx in range(bound):
        p1 = int((math.sqrt(8 * idx + 1) - 1) / 2)
        s = (p1 * (p1 + 1)) // 2
        p2 = idx - s
        p1 += 1

        dist = mesh_dist(cvxs[p1], cvxs[p2])
        if dist < threshold:
            combined = _merge_ch(cvxs[p1], cvxs[p2])
            cost_matrix[idx] = compute_hcost_3(
                cvxs[p1], cvxs[p2], combined, rv_k, resolution, seed, gpu_ctx)
            pre1 = compute_hcost(meshs[p1], cvxs[p1], rv_k, 3000, seed, gpu_ctx)
            pre2 = compute_hcost(meshs[p2], cvxs[p2], rv_k, 3000, seed, gpu_ctx)
            pre_cost_matrix[idx] = max(pre1, pre2)

    cost_size = n

    while True:
        # Find minimum cost
        if len(cost_matrix) == 0:
            break
        addr = int(np.argmin(cost_matrix))
        best_cost = cost_matrix[addr]

        if best_cost >= INF:
            break

        # Stopping criteria
        if max_ch <= 0:
            if best_cost > threshold:
                break
            if best_cost > max(threshold - pre_cost_matrix[addr], 0.01):
                cost_matrix[addr] = INF
                continue
        else:
            if len(cvxs) <= max_ch and best_cost > threshold:
                break
            if len(cvxs) <= max_ch and best_cost > max(threshold - pre_cost_matrix[addr], 0.01):
                cost_matrix[addr] = INF
                continue

        # Decode p1, p2
        addr_i = int((math.sqrt(1 + 8 * addr) - 1) / 2)
        p1 = addr_i + 1
        p2 = addr - (addr_i * (addr_i + 1)) // 2

        # Merge p1 and p2 (keep p2, remove p1)
        combined = _merge_ch(cvxs[p1], cvxs[p2])
        cvxs[p2] = combined

        # Swap p1 with last and pop
        last = len(cvxs) - 1
        cvxs[p1] = cvxs[last]
        cvxs.pop()
        meshs[p1] = meshs[last]
        meshs.pop()

        cost_size -= 1

        # Recompute costs for p2 row/column
        # Row part (i < p2)
        row_idx = (p2 * (p2 - 1)) // 2
        for i in range(p2):
            dist = mesh_dist(cvxs[p2], cvxs[i])
            if dist < threshold:
                cch = _merge_ch(cvxs[p2], cvxs[i])
                cost_matrix[row_idx] = compute_hcost_3(
                    cvxs[p2], cvxs[i], cch, rv_k, resolution, seed, gpu_ctx)
                pre_cost_matrix[row_idx] = max(pre_cost_matrix[p2] + best_cost, pre_cost_matrix[i])
            else:
                cost_matrix[row_idx] = INF
            row_idx += 1

        # Column part (i > p2)
        row_idx = (p2 * (p2 - 1)) // 2 + p2
        for i in range(p2 + 1, cost_size):
            dist = mesh_dist(cvxs[p2], cvxs[i])
            if dist < threshold:
                cch = _merge_ch(cvxs[p2], cvxs[i])
                cost_matrix[row_idx] = compute_hcost_3(
                    cvxs[p2], cvxs[i], cch, rv_k, resolution, seed, gpu_ctx)
                pre_cost_matrix[row_idx] = max(pre_cost_matrix[p2] + best_cost, pre_cost_matrix[i])
            else:
                cost_matrix[row_idx] = INF
            row_idx += i

        # Move top row/column into p1's slot
        erase_idx = (cost_size * (cost_size - 1)) // 2
        if p1 < cost_size:
            row_idx = (addr_i * p1) // 2
            top_row = erase_idx
            for i in range(p1):
                if i != p2:
                    cost_matrix[row_idx] = cost_matrix[top_row]
                    pre_cost_matrix[row_idx] = pre_cost_matrix[top_row]
                row_idx += 1
                top_row += 1

            top_row += 1
            row_idx += p1
            for i in range(p1 + 1, cost_size):
                cost_matrix[row_idx] = cost_matrix[top_row]
                pre_cost_matrix[row_idx] = pre_cost_matrix[top_row]
                top_row += 1
                row_idx += i

        cost_matrix = cost_matrix[:erase_idx]
        pre_cost_matrix = pre_cost_matrix[:erase_idx]

    return cvxs
