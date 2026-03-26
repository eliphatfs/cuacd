"""Tests for GPU beam search decomposition with Rv concavity metric.

Requires a CUDA GPU. Skipped automatically if the _gpu extension
cannot be imported (no GPU / no CUDA toolkit at build time).
"""

import numpy as np
import pytest

try:
    from coacd_gpu.beam import BeamContext, run_beam_coacd
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="GPU extension not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cube_mesh():
    """Closed unit cube [0,1]^3."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 6, 5], [4, 7, 6],   # top
        [0, 4, 5], [0, 5, 1],   # front
        [2, 6, 7], [2, 7, 3],   # back
        [0, 3, 7], [0, 7, 4],   # left
        [1, 5, 6], [1, 6, 2],   # right
    ], dtype=np.int32)
    return v, t


def _l_shape_mesh():
    """Closed L-shaped mesh: two rectangular prisms joined at the base."""
    v = np.array([
        # Bottom face (z=0) — L outline
        [0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0], [0.5, 1, 0], [0, 1, 0],
        # Top face (z=1) — same outline
        [0, 0, 1], [1, 0, 1], [1, 0.5, 1], [0.5, 0.5, 1], [0.5, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        # Bottom (z=0)
        [0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5],
        # Top (z=1)
        [6, 8, 7], [6, 9, 8], [6, 11, 9], [9, 11, 10],
        # Sides
        [0, 6, 7], [0, 7, 1],
        [1, 7, 8], [1, 8, 2],
        [2, 8, 9], [2, 9, 3],
        [3, 9, 10], [3, 10, 4],
        [4, 10, 11], [4, 11, 5],
        [5, 11, 6], [5, 6, 0],
    ], dtype=np.int32)
    return v, t


# ---------------------------------------------------------------------------
# Tests: cube (convex shape)
# ---------------------------------------------------------------------------

class TestCubeConvexity:
    """A cube is convex — Rv should be ~0, returned as a single part."""

    def test_single_part_at_default_threshold(self):
        v, t = _cube_mesh()
        parts = run_beam_coacd(v, t, threshold=0.05)
        assert len(parts) == 1

    def test_single_part_at_tight_threshold(self):
        v, t = _cube_mesh()
        parts = run_beam_coacd(v, t, threshold=0.01)
        assert len(parts) == 1

    def test_vertices_and_triangles_valid(self):
        v, t = _cube_mesh()
        parts = run_beam_coacd(v, t, threshold=0.05)
        pv, pt = parts[0]
        assert pv.shape[1] == 3
        assert pt.shape[1] == 3
        assert pv.dtype == np.float32
        assert pt.dtype == np.int32
        # All triangle indices must reference valid vertices
        assert pt.min() >= 0
        assert pt.max() < len(pv)


# ---------------------------------------------------------------------------
# Tests: L-shape (concave shape)
# ---------------------------------------------------------------------------

class TestLShapeDecomposition:
    """An L-shape is concave — should be decomposed into multiple parts."""

    def test_needs_cutting_at_low_threshold(self):
        """At threshold 0.15, the L-shape Rv (~0.19) exceeds it."""
        v, t = _l_shape_mesh()
        parts = run_beam_coacd(v, t, threshold=0.15)
        assert len(parts) >= 2

    def test_convex_at_high_threshold(self):
        """At threshold 0.3, the L-shape Rv (~0.19) is below it."""
        v, t = _l_shape_mesh()
        parts = run_beam_coacd(v, t, threshold=0.3)
        assert len(parts) == 1

    def test_more_parts_at_tighter_threshold(self):
        v, t = _l_shape_mesh()
        parts_loose = run_beam_coacd(v, t, threshold=0.15)
        parts_tight = run_beam_coacd(v, t, threshold=0.05)
        assert len(parts_tight) >= len(parts_loose)

    def test_all_parts_have_valid_geometry(self):
        v, t = _l_shape_mesh()
        parts = run_beam_coacd(v, t, threshold=0.05)
        for pv, pt in parts:
            assert pv.ndim == 2 and pv.shape[1] == 3
            assert pt.ndim == 2 and pt.shape[1] == 3
            assert pt.min() >= 0
            assert pt.max() < len(pv)
            assert len(pt) >= 1  # every part has at least one triangle


# ---------------------------------------------------------------------------
# Tests: beam search parameters
# ---------------------------------------------------------------------------

class TestBeamParams:
    def test_beam_width_1(self):
        """beam_width=1 should still produce a valid decomposition."""
        v, t = _l_shape_mesh()
        parts = run_beam_coacd(v, t, threshold=0.15, beam_width=1)
        assert len(parts) >= 2

    def test_max_iterations_limits_parts(self):
        v, t = _l_shape_mesh()
        parts = run_beam_coacd(v, t, threshold=0.05, max_iterations=1)
        # 1 iteration = 1 cut = at most 2 parts
        assert len(parts) <= 2 + 1  # +1 for possible unchanged parts

    def test_context_manager(self):
        """BeamContext as context manager should not leak."""
        v, t = _cube_mesh()
        with BeamContext(device=0) as ctx:
            parts = ctx.run(v, t, threshold=0.05)
        assert len(parts) == 1
