"""Tests for _mesh module: Mesh class and convex hull."""

import numpy as np
import pytest

from coacd_gpu.coacd._mesh import Mesh


class TestMesh:
    def test_construction(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        assert m.n_vertices == 8
        assert m.n_triangles == 12

    def test_volume(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        assert abs(m.volume() - 1.0) < 1e-10

    def test_area(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        assert abs(m.area() - 6.0) < 1e-10

    def test_bbox(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        assert m.bbox == pytest.approx((0, 1, 0, 1, 0, 1), abs=1e-10)

    def test_copy(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        m2 = m.copy()
        assert m2.n_vertices == m.n_vertices
        m2.vertices[0, 0] = 999
        assert m.vertices[0, 0] != 999  # independent copy


class TestConvexHull:
    def test_cube_hull_volume(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        assert abs(ch.volume() - 1.0) < 1e-6

    def test_offset_cube_hull_volume(self, offset_cube):
        v, t = offset_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        assert abs(ch.volume() - 1.0) < 1e-6

    def test_hull_is_convex(self, unit_cube):
        """Convex hull of convex shape should have same volume."""
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        ch2 = ch.convex_hull()
        assert abs(ch.volume() - ch2.volume()) < 1e-6

    def test_hull_encloses_original(self, l_shape):
        """Hull volume >= original volume."""
        v, t = l_shape
        m = Mesh(v, t)
        ch = m.convex_hull()
        assert ch.volume() >= abs(m.volume()) - 1e-6

    def test_hull_positive_volume(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        assert ch.volume() > 0
