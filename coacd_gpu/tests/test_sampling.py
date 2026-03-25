"""Tests for _sampling module: surface sampling."""

import numpy as np
import pytest

from coacd_gpu.coacd._sampling import sample_surface
from coacd_gpu.coacd._geometry import Plane


class TestSampling:
    def test_sample_count(self, unit_cube):
        v, t = unit_cube
        rng = np.random.RandomState(42)
        samples, ids = sample_surface(v, t, resolution=500, rng=rng)
        assert len(samples) > 0
        assert len(samples) == len(ids)

    def test_samples_on_surface(self, unit_cube):
        """All samples should lie on the cube surface (one coord is 0 or 1)."""
        v, t = unit_cube
        rng = np.random.RandomState(42)
        samples, ids = sample_surface(v, t, resolution=500, rng=rng)
        for pt in samples:
            on_face = any(abs(pt[k]) < 1e-10 or abs(pt[k] - 1.0) < 1e-10
                         for k in range(3))
            assert on_face, f"Point {pt} not on cube surface"

    def test_samples_within_bbox(self, unit_cube):
        v, t = unit_cube
        rng = np.random.RandomState(42)
        samples, ids = sample_surface(v, t, resolution=1000, rng=rng)
        assert samples[:, 0].min() >= -1e-10
        assert samples[:, 0].max() <= 1.0 + 1e-10
        assert samples[:, 1].min() >= -1e-10
        assert samples[:, 1].max() <= 1.0 + 1e-10
        assert samples[:, 2].min() >= -1e-10
        assert samples[:, 2].max() <= 1.0 + 1e-10

    def test_tri_ids_valid(self, unit_cube):
        v, t = unit_cube
        rng = np.random.RandomState(42)
        samples, ids = sample_surface(v, t, resolution=500, rng=rng)
        assert ids.min() >= 0
        assert ids.max() < len(t)

    def test_zero_resolution(self, unit_cube):
        v, t = unit_cube
        samples, ids = sample_surface(v, t, resolution=0)
        assert len(samples) == 0

    def test_exclude_plane(self, unit_cube):
        """Samples should skip triangles coplanar with the exclusion plane."""
        v, t = unit_cube
        plane = Plane(0.0, 0.0, 1.0, 0.0)  # z=0 plane
        rng = np.random.RandomState(42)
        samples, ids = sample_surface(v, t, resolution=500, rng=rng,
                                      exclude_plane=plane)
        # No samples should be on the z=0 face
        if len(samples) > 0:
            z_vals = samples[:, 2]
            # Allow samples from non-z=0 faces that happen to be near z=0
            # but the bottom face triangles should be excluded
            bottom_tris = {0, 1}  # first two triangles are z=0 face
            for tri_id in ids:
                assert tri_id not in bottom_tris

    def test_deterministic(self, unit_cube):
        v, t = unit_cube
        s1, _ = sample_surface(v, t, 200, np.random.RandomState(42))
        s2, _ = sample_surface(v, t, 200, np.random.RandomState(42))
        np.testing.assert_array_equal(s1, s2)
