"""End-to-end regression for warp-cooperative cut-boundary traversal."""

import numpy as np
import pytest
import trimesh

import cuacd


def _l_prism():
    xy = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    vertices = np.concatenate([np.column_stack([xy, np.zeros(6)]),
                               np.column_stack([xy, np.ones(6)])])
    top = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    faces = list(top[:, [0, 2, 1]]) + list(top + 6)
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    return trimesh.Trimesh(vertices, faces, process=True)


@pytest.mark.parametrize("depth", [1, 2])
def test_l_prism_boundary_traversal(depth):
    """Reconstruct valid cut loops and preserve a closed input's volume."""
    mesh = _l_prism()
    lo, hi = mesh.bounds
    vertices = np.ascontiguousarray(
        (mesh.vertices - (lo + hi) / 2) / ((hi - lo).max() / 2), dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    original = trimesh.Trimesh(vertices, faces, process=False)
    with cuacd.Context(pool_bytes=1024**3) as ctx:
        assert not ctx.check_mesh(vertices, faces)['needs_remesh']
        # Keep this regression independent of non-divisible axis-candidate widths.
        parts = ctx.lookahead_decompose(
            vertices, faces, threshold=0.05, width=30, width2=3, depth=depth,
            n_concave_edges=16, merge_hulls=False, max_iters=100)
    assert len(parts) >= 2
    total_volume = 0.0
    for pv, pt, hv, ht in parts:
        for v, t in ((pv, pt), (hv, ht)):
            assert np.isfinite(v).all()
            assert (t >= 0).all() and (t < len(v)).all()
            result = trimesh.Trimesh(v, t, process=False)
            assert result.is_watertight and result.is_winding_consistent
            assert result.volume > 0
        total_volume += trimesh.Trimesh(pv, pt, process=False).volume
    assert total_volume == pytest.approx(original.volume, rel=1e-4, abs=1e-6)
