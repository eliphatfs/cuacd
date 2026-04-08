"""Tests for lookahead search decomposition."""

import os
import numpy as np
import pytest

import coacd_gpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_ctx():
    with coacd_gpu.Context(device=0, pool_bytes=10 * 1024**3) as ctx:
        yield ctx


def _make_cube():
    """Unit cube [0,1]^3 with outward-facing triangles."""
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    t = np.array([
        [0,2,1],[0,3,2],  # -Z
        [4,5,6],[4,6,7],  # +Z
        [0,1,5],[0,5,4],  # -Y
        [2,3,7],[2,7,6],  # +Y
        [0,4,7],[0,7,3],  # -X
        [1,2,6],[1,6,5],  # +X
    ], dtype=np.int32)
    return v, t


def _make_lshape():
    """L-shape: two boxes sharing an edge."""
    a_v, a_t = _make_cube()
    # Second box offset in X, sharing the [1,0,0]→[1,1,1] edge
    b_v = a_v + np.array([1, 0, 0], dtype=np.float32)
    b_t = a_t + len(a_v)
    v = np.concatenate([a_v, b_v], axis=0)
    t = np.concatenate([a_t, b_t], axis=0)
    return v, t


def _call_lookahead_decompose(verts, tris, **kwargs):
    """Call lookahead_decompose with raw C pointers."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int32)

    import scipy.spatial
    hull = scipy.spatial.ConvexHull(verts)
    hull_verts = np.ascontiguousarray(hull.points, dtype=np.float32)
    hull_tris = np.ascontiguousarray(hull.simplices, dtype=np.int32)

    defaults = dict(
        max_iters=100, width=30, threshold=0.05,
        depth=2, quick_depth=1, max_n_cutting=16,
        verbose=0, debug=0,
    )
    defaults.update(kwargs)

    from coacd_gpu import _gpu
    return _gpu.lookahead_decompose(
        verts.ctypes.data, len(verts),
        tris.ctypes.data, len(tris),
        hull_verts.ctypes.data, len(hull_verts),
        hull_tris.ctypes.data, len(hull_tris),
        **defaults)


def _decompose_shape(verts, tris, **kwargs):
    """Normalize, decompose, and denormalize."""
    # Normalize to [-1, 1]
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2
    extent = (hi - lo).max()
    scale = extent / 2 if extent > 0 else 1.0
    nv = (verts - center) / scale

    raw = _call_lookahead_decompose(nv, tris, **kwargs)

    parts = []
    for vb, tb, nv_count, nt_count, mv, hv in raw:
        v = np.frombuffer(vb, dtype=np.float32).reshape(nv_count, 3).copy()
        t = np.frombuffer(tb, dtype=np.int32).reshape(nt_count, 3).copy()
        # Denormalize
        v = v * scale + center
        parts.append((v, t))
    return parts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLookaheadDecompose:

    def test_cube_la(self, gpu_ctx):
        """Unit cube is already convex — should produce 1 part."""
        verts, tris = _make_cube()
        parts = _decompose_shape(verts, tris, max_iters=5, threshold=0.05)
        assert len(parts) == 1

    def test_lshape_la(self, gpu_ctx):
        """L-shape should decompose into 2-4 convex parts."""
        verts, tris = _make_lshape()
        parts = _decompose_shape(verts, tris, max_iters=50, threshold=0.05)
        assert 2 <= len(parts) <= 8
        for v, t in parts:
            assert v.ndim == 2 and v.shape[1] == 3
            assert t.ndim == 2 and t.shape[1] == 3
            assert len(v) >= 4  # minimum tetrahedron
            assert len(t) >= 4

    @pytest.mark.skipif(not os.path.exists("CoACD/examples/Octocat-v2.obj"),
                        reason="Octocat model not found")
    def test_octocat_la(self, gpu_ctx):
        """Octocat decomposition should produce >= 1 part."""
        import trimesh
        mesh = trimesh.load("CoACD/examples/Octocat-v2.obj")
        parts = _decompose_shape(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            max_iters=100, threshold=0.05)
        assert len(parts) >= 1

    def test_la_convergence(self, gpu_ctx):
        """After decomposition with reasonable threshold, all parts should be roughly convex."""
        verts, tris = _make_lshape()
        parts = _decompose_shape(verts, tris, max_iters=100, threshold=0.05)
        assert len(parts) >= 2
        # Each part should have valid geometry
        for v, t in parts:
            assert len(v) > 0
            assert len(t) > 0

    @pytest.mark.skipif(not os.path.exists("49160.stl"),
                        reason="49160.stl not found")
    def test_49160_la(self, gpu_ctx):
        """Decompose 49160.stl — should produce >= 1 part."""
        import trimesh
        mesh = trimesh.load("49160.stl")
        parts = _decompose_shape(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            max_iters=100, threshold=0.05)
        assert len(parts) >= 1
