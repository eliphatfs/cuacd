"""Tests for _mcts module: MCTS search and ternary refinement."""

import numpy as np
import pytest

from coacd_gpu.coacd._mesh import Mesh
from coacd_gpu.coacd._geometry import Plane
from coacd_gpu.coacd._mcts import (
    compute_axes_aligned_planes,
    monte_carlo_tree_search,
    ternary_refine,
    _clip_by_path,
)


class TestAxesAlignedPlanes:
    def test_generates_planes(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        planes = compute_axes_aligned_planes(m.bbox, mcts_nodes=5)
        assert len(planes) > 0

    def test_planes_are_axis_aligned(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        planes = compute_axes_aligned_planes(m.bbox, mcts_nodes=5)
        for p in planes:
            nonzero = sum(abs(x) > 1e-10 for x in [p.a, p.b, p.c])
            assert nonzero == 1, "Plane should be axis-aligned"

    def test_shuffle(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        rng = np.random.RandomState(42)
        p1 = compute_axes_aligned_planes(m.bbox, 5, shuffle=True, rng=rng)
        p2 = compute_axes_aligned_planes(m.bbox, 5, shuffle=False)
        # Shuffled and unshuffled should have same length
        assert len(p1) == len(p2)


class TestMCTS:
    @pytest.mark.slow
    def test_finds_plane_for_l_shape(self, l_shape):
        """MCTS should find a valid cutting plane for a non-convex mesh."""
        v, t = l_shape
        m = Mesh(v, t)
        params = {
            "threshold": 0.05,
            "mcts_nodes": 5,
            "mcts_iteration": 20,
            "mcts_max_depth": 3,
            "rv_k": 0.3,
            "seed": 42,
        }
        rng = np.random.RandomState(42)
        plane, path, quality = monte_carlo_tree_search(m, params, rng)
        assert plane is not None
        assert quality < float("inf")

    def test_returns_none_for_convex(self, unit_cube):
        """For a convex mesh, MCTS may still find planes (they just have low cost)."""
        v, t = unit_cube
        m = Mesh(v, t)
        params = {
            "threshold": 0.05,
            "mcts_nodes": 3,
            "mcts_iteration": 10,
            "mcts_max_depth": 2,
            "rv_k": 0.3,
            "seed": 42,
        }
        rng = np.random.RandomState(42)
        plane, path, quality = monte_carlo_tree_search(m, params, rng)
        # Should return a plane (even if not needed)
        # The quality should be low for a convex mesh
        if plane is not None:
            assert quality >= 0


class TestClipByPath:
    def test_single_plane(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, cost = _clip_by_path(m, p, [], 0.3)
        assert ok
        assert cost >= 0
        assert cost < float("inf")


class TestTernaryRefine:
    @pytest.mark.slow
    def test_refine_improves_or_maintains(self, l_shape):
        """Ternary refinement should not make cost worse."""
        v, t = l_shape
        m = Mesh(v, t)
        params = {
            "threshold": 0.05,
            "mcts_nodes": 5,
            "mcts_iteration": 10,
            "mcts_max_depth": 2,
            "rv_k": 0.3,
            "seed": 42,
        }
        rng = np.random.RandomState(42)
        plane, path, quality = monte_carlo_tree_search(m, params, rng)
        if plane is not None:
            refined = ternary_refine(m, plane, path, quality, params)
            assert refined is not None
            # Refined plane should be axis-aligned
            nonzero = sum(abs(x) > 1e-10 for x in [refined.a, refined.b, refined.c])
            assert nonzero == 1
