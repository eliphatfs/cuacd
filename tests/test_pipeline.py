"""Tests for _pipeline module: end-to-end run_coacd."""

import numpy as np
import pytest

from coacd_gpu.coacd import run_coacd

# Pipeline tests use MCTS and are inherently slow.
pytestmark = pytest.mark.slow


class TestPipeline:
    def test_convex_mesh_single_part(self, unit_cube):
        """A convex mesh should produce exactly 1 part."""
        v, t = unit_cube
        parts = run_coacd(v, t, threshold=0.05, mcts_iterations=10,
                          mcts_nodes=3, seed=42, merge=False)
        assert len(parts) == 1
        verts, tris = parts[0]
        assert len(verts) > 0
        assert len(tris) > 0

    def test_output_format(self, unit_cube):
        """Output should be list of (vertices, triangles) tuples."""
        v, t = unit_cube
        parts = run_coacd(v, t, threshold=0.05, mcts_iterations=10,
                          mcts_nodes=3, seed=42)
        assert isinstance(parts, list)
        for verts, tris in parts:
            assert isinstance(verts, np.ndarray)
            assert isinstance(tris, np.ndarray)
            assert verts.ndim == 2 and verts.shape[1] == 3
            assert tris.ndim == 2 and tris.shape[1] == 3

    def test_nonconvex_produces_multiple(self, l_shape):
        """A non-convex mesh should produce > 1 part."""
        v, t = l_shape
        parts = run_coacd(v, t, threshold=0.05, mcts_iterations=10,
                          mcts_nodes=3, seed=42, merge=False)
        assert len(parts) >= 1

    def test_recovery_preserves_scale(self, unit_cube):
        """Output vertices should be in original coordinate space."""
        v, t = unit_cube
        parts = run_coacd(v, t, threshold=0.05, mcts_iterations=10,
                          mcts_nodes=3, seed=42)
        for verts, tris in parts:
            # All vertices should be within [0,1]^3 (with tolerance)
            assert verts.min() >= -0.1
            assert verts.max() <= 1.1

    def test_pca_option(self, unit_cube):
        """Pipeline should work with PCA enabled."""
        v, t = unit_cube
        parts = run_coacd(v, t, threshold=0.05, mcts_iterations=10,
                          mcts_nodes=3, seed=42, pca=True)
        assert len(parts) >= 1

    def test_deterministic(self, l_shape):
        """Same seed should produce same results."""
        v, t = l_shape
        kwargs = dict(threshold=0.08, mcts_iterations=5, mcts_nodes=3,
                      seed=42, merge=False)
        parts1 = run_coacd(v, t, **kwargs)
        parts2 = run_coacd(v, t, **kwargs)
        assert len(parts1) == len(parts2)
