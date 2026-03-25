"""Top-level decomposition pipeline.

Translates CoACD/src/process.cpp Compute() and the Python API from
CoACD/python/package/__init__.py.
"""

import logging
import numpy as np

from ._geometry import normalize, recover, pca_align, revert_pca, compute_bbox
from ._mesh import Mesh
from ._cost import compute_hcost
from ._clip import clip
from ._mcts import monte_carlo_tree_search, ternary_refine
from ._merge import merge_convex_hulls

logger = logging.getLogger("coacd")


def run_coacd(
    vertices: np.ndarray,
    triangles: np.ndarray,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 150,
    mcts_max_depth: int = 3,
    pca: bool = False,
    merge: bool = True,
    seed: int = 0,
    rv_k: float = 0.3,
    gpu_ctx=None,
):
    """Run approximate convex decomposition.

    Args:
        vertices: (N, 3) float64 array
        triangles: (M, 3) int32 array
        threshold: concavity threshold (lower = finer decomposition)
        max_convex_hull: max number of output parts (-1 = no limit)
        resolution: surface sampling resolution
        mcts_nodes: number of candidate planes per axis
        mcts_iterations: MCTS iterations per part
        mcts_max_depth: max MCTS tree depth
        pca: whether to PCA-align before decomposition
        merge: whether to run merge post-processing
        seed: random seed
        rv_k: Rv cost weight
        gpu_ctx: optional coacd_gpu.Context for GPU-accelerated Hausdorff

    Returns:
        list of (vertices, triangles) tuples — each a convex hull.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(triangles, dtype=np.int32)

    mesh = Mesh(vertices, triangles)

    # Normalize to [-1, 1]
    orig_bbox = mesh.bbox
    new_verts, new_bbox, orig_bbox = normalize(mesh.vertices, mesh.bbox)
    mesh = Mesh(new_verts, mesh.triangles, new_bbox)

    # Optional PCA alignment
    rot = None
    if pca:
        aligned_verts, rot, pca_bbox = pca_align(mesh.vertices)
        mesh = Mesh(aligned_verts, mesh.triangles, pca_bbox)

    params = {
        "threshold": threshold,
        "max_convex_hull": max_convex_hull,
        "resolution": resolution,
        "mcts_nodes": mcts_nodes,
        "mcts_iteration": mcts_iterations,
        "mcts_max_depth": mcts_max_depth,
        "rv_k": rv_k,
        "seed": seed,
    }

    # Main decomposition loop
    input_parts = [mesh]
    final_parts = []   # convex hulls
    final_meshs = []   # original part meshes (for merge)

    iteration = 0
    while len(input_parts) > 0:
        logger.info("iter %d ---- waiting pool: %d", iteration, len(input_parts))
        next_parts = []

        for p_idx, pmesh in enumerate(input_parts):
            rng = np.random.RandomState(seed)

            pCH = pmesh.convex_hull()
            h = compute_hcost(pmesh, pCH, rv_k, resolution, seed, gpu_ctx)

            if h > threshold:
                # Run MCTS
                best_plane, best_path, best_quality = monte_carlo_tree_search(pmesh, params, rng)

                if best_plane is None:
                    final_parts.append(pCH)
                    final_meshs.append(pmesh)
                else:
                    # Ternary refinement
                    best_plane = ternary_refine(pmesh, best_plane, best_path, best_quality, params)

                    ok, pos, neg = clip(pmesh, best_plane)
                    if not ok:
                        logger.error("Clip failed for part %d", p_idx)
                        final_parts.append(pCH)
                        final_meshs.append(pmesh)
                    else:
                        if pos is not None and pos.n_triangles > 0:
                            next_parts.append(pos)
                        if neg is not None and neg.n_triangles > 0:
                            next_parts.append(neg)
            else:
                final_parts.append(pCH)
                final_meshs.append(pmesh)

        input_parts = next_parts
        iteration += 1

    # Merge phase
    if merge and len(final_parts) > 1:
        final_parts = merge_convex_hulls(final_meshs, final_parts, params, gpu_ctx)

    # Recover original coordinates
    result = []
    for part in final_parts:
        v = part.vertices.copy()
        if pca and rot is not None:
            v = revert_pca(v, rot)
        v = recover(v, orig_bbox)
        result.append((v, part.triangles.copy()))

    logger.info("# Convex Hulls: %d", len(result))
    return result
