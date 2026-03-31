"""Tests for V2 beam search (3-kernel architecture).

Tests the run_v2 path which uses:
- scipy ConvexHull for initial hull
- beam_expansion kernel (D&C hull per candidate)
- beam_hausdorff_parts kernel
- beam_termination kernel
"""

import numpy as np
import pytest

try:
    from coacd_gpu.beam import BeamContext, run_beam_coacd_v2
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

pytestmark = pytest.mark.skipif(
    not _HAS_GPU or not _HAS_SCIPY,
    reason="GPU extension or scipy not available")


def _cube_mesh():
    """Closed unit cube [0,1]^3."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return v, t


def _l_shape_mesh():
    """Closed L-shaped mesh."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0], [0.5, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 0.5, 1], [0.5, 0.5, 1], [0.5, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5],
        [6, 8, 7], [6, 9, 8], [6, 11, 9], [9, 11, 10],
        [0, 6, 7], [0, 7, 1],
        [1, 7, 8], [1, 8, 2],
        [2, 8, 9], [2, 9, 3],
        [3, 9, 10], [3, 10, 4],
        [4, 10, 11], [4, 11, 5],
        [5, 11, 6], [5, 6, 0],
    ], dtype=np.int32)
    return v, t


class TestCubeConvexityV2:
    """Cube is convex — V2 should return it as a single part."""

    def test_single_part(self):
        v, t = _cube_mesh()
        parts = run_beam_coacd_v2(v, t, threshold=0.05)
        assert len(parts) == 1

    def test_valid_geometry(self):
        v, t = _cube_mesh()
        parts = run_beam_coacd_v2(v, t, threshold=0.05)
        pv, pt = parts[0]
        assert pv.shape[1] == 3
        assert pt.shape[1] == 3
        assert pt.min() >= 0
        assert pt.max() < len(pv)


class TestLShapeV2:
    """L-shape should decompose into 2+ parts."""

    def test_decomposition(self):
        v, t = _l_shape_mesh()
        parts = run_beam_coacd_v2(v, t, threshold=0.05,
                                  beam_width=8, cuts_per_axis=10)
        assert len(parts) >= 2

    def test_convex_at_high_threshold(self):
        v, t = _l_shape_mesh()
        parts = run_beam_coacd_v2(v, t, threshold=0.3)
        assert len(parts) == 1

    def test_valid_parts(self):
        v, t = _l_shape_mesh()
        parts = run_beam_coacd_v2(v, t, threshold=0.05,
                                  beam_width=8, cuts_per_axis=10)
        for pv, pt in parts:
            assert pv.ndim == 2 and pv.shape[1] == 3
            assert pt.ndim == 2 and pt.shape[1] == 3
            assert pt.min() >= 0
            assert pt.max() < len(pv)


class TestBeamV2Context:
    def test_context_manager(self):
        v, t = _cube_mesh()
        with BeamContext(device=0) as ctx:
            parts = ctx.run_v2(v, t, threshold=0.05)
        assert len(parts) == 1
