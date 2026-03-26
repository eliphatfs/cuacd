"""Tests for _geometry module: Plane, mesh_volume, mesh_area, normalize, PCA."""

import numpy as np
import pytest

from coacd_gpu.coacd._geometry import (
    Plane,
    mesh_volume,
    mesh_area,
    triangle_areas,
    normalize,
    recover,
    pca_align,
    revert_pca,
    compute_bbox,
)


class TestPlane:
    def test_side_one(self):
        p = Plane(1.0, 0.0, 0.0, -0.5)
        assert p.side_one(np.array([0.0, 0.0, 0.0])) == -1
        assert p.side_one(np.array([1.0, 0.0, 0.0])) == 1
        assert p.side_one(np.array([0.5, 0.0, 0.0])) == 0

    def test_sides_vectorised(self):
        p = Plane(0.0, 1.0, 0.0, 0.0)
        pts = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64)
        s = p.sides(pts)
        np.testing.assert_array_equal(s, [-1, 0, 1])

    def test_intersect_segment_hit(self):
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pi = p.intersect_segment(np.array([0.0, 0.5, 0.5]),
                                     np.array([1.0, 0.5, 0.5]))
        assert ok
        np.testing.assert_allclose(pi, [0.5, 0.5, 0.5], atol=1e-10)

    def test_intersect_segment_miss(self):
        p = Plane(1.0, 0.0, 0.0, -2.0)
        ok, pi = p.intersect_segment(np.array([0.0, 0.0, 0.0]),
                                     np.array([1.0, 0.0, 0.0]))
        assert not ok

    def test_intersect_segment_endpoints(self):
        p = Plane(1.0, 0.0, 0.0, -0.5)
        ok, pi = p.intersect_segment(np.array([0.5, 0.0, 0.0]),
                                     np.array([1.0, 0.0, 0.0]))
        assert ok
        np.testing.assert_allclose(pi, [0.5, 0.0, 0.0], atol=1e-10)

    def test_cut_side(self):
        p = Plane(0.0, 0.0, 1.0, 0.0)
        # Triangle with normal pointing in +z
        p0 = np.array([0, 0, 0.0])
        p1 = np.array([1, 0, 0.0])
        p2 = np.array([0, 1, 0.0])
        side = p.cut_side(p0, p1, p2)
        assert side in (-1, 1)


class TestMeshMetrics:
    def test_unit_cube_volume(self, unit_cube):
        v, t = unit_cube
        assert abs(mesh_volume(v, t) - 1.0) < 1e-10

    def test_unit_cube_area(self, unit_cube):
        v, t = unit_cube
        assert abs(mesh_area(v, t) - 6.0) < 1e-10

    def test_offset_cube_volume(self, offset_cube):
        v, t = offset_cube
        assert abs(mesh_volume(v, t) - 1.0) < 1e-10

    def test_triangle_areas_sum(self, unit_cube):
        v, t = unit_cube
        areas = triangle_areas(v, t)
        assert abs(areas.sum() - 6.0) < 1e-10
        assert len(areas) == len(t)

    def test_scaled_cube(self, unit_cube):
        v, t = unit_cube
        v2 = v * 2.0
        assert abs(mesh_volume(v2, t) - 8.0) < 1e-10
        assert abs(mesh_area(v2, t) - 24.0) < 1e-10


class TestNormalize:
    def test_roundtrip(self, unit_cube):
        v, t = unit_cube
        bbox = compute_bbox(v)
        new_v, new_bbox, orig_bbox = normalize(v, bbox)
        recovered = recover(new_v, orig_bbox)
        np.testing.assert_allclose(recovered, v, atol=1e-10)

    def test_normalized_range(self, unit_cube):
        v, t = unit_cube
        bbox = compute_bbox(v)
        new_v, new_bbox, orig_bbox = normalize(v, bbox)
        assert new_v.min() >= -1.0 - 1e-10
        assert new_v.max() <= 1.0 + 1e-10


class TestPCA:
    def test_roundtrip(self, unit_cube):
        v, t = unit_cube
        new_v, rot, new_bbox = pca_align(v)
        recovered = revert_pca(new_v, rot)
        # PCA centers the data, so recovered is centered at origin
        # The original center is [0.5, 0.5, 0.5]
        np.testing.assert_allclose(recovered + v.mean(axis=0), v, atol=1e-10)

    def test_preserves_volume(self, unit_cube):
        v, t = unit_cube
        new_v, rot, _ = pca_align(v)
        vol_before = abs(mesh_volume(v, t))
        vol_after = abs(mesh_volume(new_v, t))
        assert abs(vol_before - vol_after) < 1e-8
