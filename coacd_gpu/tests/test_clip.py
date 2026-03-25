"""Tests for _clip module: plane-mesh clipping."""

import numpy as np
import pytest

from coacd_gpu.coacd._mesh import Mesh
from coacd_gpu.coacd._geometry import Plane
from coacd_gpu.coacd._clip import clip


class TestClip:
    def test_clip_cube_x_axis(self, unit_cube):
        """Clip unit cube at x=0.5 should produce two halves."""
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pos, neg = clip(m, p)
        assert ok
        assert pos is not None
        assert neg is not None
        assert pos.n_triangles > 0
        assert neg.n_triangles > 0

    def test_clip_volume_conservation(self, unit_cube):
        """Sum of clipped volumes should equal original volume."""
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pos, neg = clip(m, p)
        assert ok
        total = pos.volume() + neg.volume()
        assert abs(total - 1.0) < 1e-6

    def test_clip_equal_halves(self, unit_cube):
        """Cutting cube at midplane should give two equal halves."""
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pos, neg = clip(m, p)
        assert ok
        assert abs(pos.volume() - 0.5) < 1e-6
        assert abs(neg.volume() - 0.5) < 1e-6

    def test_clip_y_axis(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(0.0, 1.0, 0.0, -0.3)
        ok, pos, neg = clip(m, p)
        assert ok
        total = pos.volume() + neg.volume()
        assert abs(total - 1.0) < 1e-4

    def test_clip_z_axis(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(0.0, 0.0, 1.0, -0.7)
        ok, pos, neg = clip(m, p)
        assert ok
        total = pos.volume() + neg.volume()
        assert abs(total - 1.0) < 1e-4

    def test_clip_outside_fails(self, unit_cube):
        """Plane outside mesh should fail or produce one empty half."""
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -5.0)  # x=5, far outside
        ok, pos, neg = clip(m, p)
        # Either fails or one side is empty
        if ok:
            assert pos.n_triangles == 0 or neg.n_triangles == 0

    def test_clip_creates_cap(self, unit_cube):
        """Clipped halves should have more triangles than original halves
        (because of cap triangulation)."""
        v, t = unit_cube
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pos, neg = clip(m, p)
        assert ok
        # Each half should be a closed mesh (12 original tris / 2 = 6 side +
        # cap tris)
        assert pos.n_triangles > 6
        assert neg.n_triangles > 6

    def test_clip_l_shape(self, l_shape):
        """Clip an L-shape should produce valid parts."""
        v, t = l_shape
        m = Mesh(v, t)
        p = Plane(1.0, 0.0, 0.0, -1.0)  # x=1 boundary between the two boxes
        ok, pos, neg = clip(m, p)
        assert ok
        assert pos.n_triangles > 0
        assert neg.n_triangles > 0
