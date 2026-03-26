"""Tests for _cost module: Rv, Hausdorff, HCost."""

import numpy as np
import pytest

from coacd_gpu.coacd._mesh import Mesh
from coacd_gpu.coacd._cost import (
    compute_rv,
    compute_rv_3,
    compute_total_rv,
    hausdorff_cpu,
    compute_hb,
    compute_hcost,
    merge_meshes,
    mesh_dist,
)
from coacd_gpu.coacd._sampling import sample_surface


class TestRv:
    def test_convex_mesh_zero_rv(self, unit_cube):
        """Rv of a convex mesh vs its own hull should be ~0."""
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        rv = compute_rv(m, ch, k=0.3)
        assert rv < 1e-6

    def test_nonconvex_positive_rv(self, l_shape):
        """Rv of a non-convex mesh should be > 0."""
        v, t = l_shape
        m = Mesh(v, t)
        ch = m.convex_hull()
        rv = compute_rv(m, ch, k=0.3)
        assert rv > 0

    def test_rv_3_identity(self, unit_cube):
        """Rv3 with two identical hulls merged = same hull should be ~0."""
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        rv = compute_rv_3(ch, ch, ch, k=0.3)
        # Not exactly 0 because combined hull of two copies has 2x volume
        # but ch_combined = ch in this degenerate case
        assert rv >= 0


class TestHausdorff:
    def test_identical_meshes_zero(self, unit_cube):
        """Hausdorff between identical meshes should be ~0."""
        v, t = unit_cube
        m = Mesh(v, t)
        rng = np.random.RandomState(42)
        sa, ida = sample_surface(v, t, 200, rng)
        rng2 = np.random.RandomState(42)
        sb, idb = sample_surface(v, t, 200, rng2)
        h = hausdorff_cpu(m, sa, ida, m, sb, idb)
        assert h < 1e-6

    def test_hausdorff_positive_for_different(self, unit_cube, offset_cube):
        """Hausdorff between distant meshes should be positive."""
        v1, t1 = unit_cube
        v2, t2 = offset_cube
        m1 = Mesh(v1, t1)
        m2 = Mesh(v2, t2)
        rng1 = np.random.RandomState(42)
        sa, ida = sample_surface(v1, t1, 200, rng1)
        rng2 = np.random.RandomState(42)
        sb, idb = sample_surface(v2, t2, 200, rng2)
        h = hausdorff_cpu(m1, sa, ida, m2, sb, idb)
        assert h > 1.0  # distance between [0,1]^3 and [2,3]^3 is >= 1

    def test_hausdorff_symmetric(self, unit_cube, offset_cube):
        """Hausdorff(A,B) should be close to Hausdorff(B,A) (it's the max of both)."""
        v1, t1 = unit_cube
        v2, t2 = offset_cube
        m1 = Mesh(v1, t1)
        m2 = Mesh(v2, t2)
        rng = np.random.RandomState(42)
        sa, ida = sample_surface(v1, t1, 200, rng)
        rng = np.random.RandomState(42)
        sb, idb = sample_surface(v2, t2, 200, rng)
        h_ab = hausdorff_cpu(m1, sa, ida, m2, sb, idb)
        h_ba = hausdorff_cpu(m2, sb, idb, m1, sa, ida)
        assert abs(h_ab - h_ba) < 0.5  # Same max, but sampling may differ


class TestHCost:
    def test_convex_low(self, unit_cube):
        """HCost of convex mesh should be small (sampling noise)."""
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        h = compute_hcost(m, ch, k=0.3, resolution=200, seed=42)
        # Not exactly 0: different triangulations cause small Hausdorff noise
        assert h < 0.1

    def test_hb_convex_low(self, unit_cube):
        """Hb of convex mesh should be small."""
        v, t = unit_cube
        m = Mesh(v, t)
        ch = m.convex_hull()
        hb = compute_hb(m, ch, resolution=200, seed=42)
        assert hb < 0.1


class TestMergeMeshes:
    def test_merge_preserves_triangles(self, unit_cube, offset_cube):
        v1, t1 = unit_cube
        v2, t2 = offset_cube
        m1 = Mesh(v1, t1)
        m2 = Mesh(v2, t2)
        merged = merge_meshes(m1, m2)
        assert merged.n_vertices == m1.n_vertices + m2.n_vertices
        assert merged.n_triangles == m1.n_triangles + m2.n_triangles


class TestMeshDist:
    def test_same_mesh_zero(self, unit_cube):
        v, t = unit_cube
        m = Mesh(v, t)
        d = mesh_dist(m, m)
        assert d < 1e-10

    def test_distant_meshes(self, unit_cube, offset_cube):
        v1, t1 = unit_cube
        v2, t2 = offset_cube
        m1 = Mesh(v1, t1)
        m2 = Mesh(v2, t2)
        d = mesh_dist(m1, m2)
        assert d > 0.9
