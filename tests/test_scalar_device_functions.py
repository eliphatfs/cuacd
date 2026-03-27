"""Tests for scalar device functions (A1-A4) against reference implementations.

Compares GPU results against trimesh/scipy for 1000-element random batches.
"""

import numpy as np
import pytest

# Reference libraries
trimesh = pytest.importorskip("trimesh")
scipy_spatial = pytest.importorskip("scipy.spatial")

import coacd_gpu._gpu as _gpu


N = 1000
SEED = 42


@pytest.fixture(scope="module", autouse=True)
def gpu_ctx():
    _gpu.init()
    yield
    _gpu.destroy()


def _ptr(arr):
    return arr.ctypes.data


# -------------------------------------------------------------------------
# A1: signed_tet_volume
# -------------------------------------------------------------------------

class TestSignedTetVolume:
    """Test signed_tet_volume against scipy ConvexHull.volume on random tets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.RandomState(SEED)
        # Random tetrahedra: 4 points each, but signed_tet_volume takes 3 points
        # (the 4th is the origin). So we generate 3 points per tet.
        self.tets = rng.randn(N, 3, 3).astype(np.float32)

    def test_against_scipy(self):
        """Each signed_tet_volume(p0, p1, p2) = p0 . (p1 x p2) / 6.
        For a tetrahedron with vertices [origin, p0, p1, p2],
        |volume| = |p0 . (p1 x p2)| / 6 = scipy ConvexHull volume."""
        tets_flat = self.tets.reshape(N, 9).copy()
        out = np.zeros(N, dtype=np.float32)
        _gpu.batch_signed_tet_volume(_ptr(tets_flat), N, _ptr(out))

        for i in range(N):
            p0, p1, p2 = self.tets[i]
            hull_pts = np.array([[0, 0, 0], p0, p1, p2], dtype=np.float64)
            try:
                hull = scipy_spatial.ConvexHull(hull_pts)
                expected_abs_vol = hull.volume
            except scipy_spatial.QhullError:
                # Degenerate tet (coplanar), volume ~ 0
                expected_abs_vol = 0.0
            np.testing.assert_allclose(
                abs(out[i]), expected_abs_vol,
                atol=1e-4, rtol=1e-3,
                err_msg=f"Tet {i}: GPU={out[i]}, scipy={expected_abs_vol}")


# -------------------------------------------------------------------------
# A2: intersect_edge
# -------------------------------------------------------------------------

class TestIntersectEdge:
    """Test intersect_edge against trimesh.intersections.plane_lines."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.RandomState(SEED + 1)
        # Random segments: two endpoints each
        self.v0 = rng.randn(N, 3).astype(np.float32)
        self.v1 = rng.randn(N, 3).astype(np.float32)
        # Random planes: normal (unit) + offset d
        normals = rng.randn(N, 3).astype(np.float64)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals.astype(np.float32)
        self.d = rng.randn(N).astype(np.float32)

        # Filter: keep only segments that actually cross the plane
        # d0 = n.v0 + d, d1 = n.v1 + d; crossing if d0*d1 < 0
        d0 = np.sum(self.normals * self.v0, axis=1) + self.d
        d1 = np.sum(self.normals * self.v1, axis=1) + self.d
        self.crossing = (d0 * d1) < 0
        # Ensure we have enough crossing cases
        assert self.crossing.sum() > N // 4, "Too few crossing segments"

    def test_against_trimesh(self):
        segments = np.hstack([self.v0, self.v1]).astype(np.float32).copy()
        planes = np.hstack([self.normals, self.d[:, None]]).astype(np.float32).copy()
        out = np.zeros((N, 3), dtype=np.float32)
        _gpu.batch_intersect_edge(_ptr(segments), _ptr(planes), N, _ptr(out))

        for i in range(N):
            if not self.crossing[i]:
                continue
            # trimesh reference: endpoints is (2, n, 3)
            plane_origin = -self.d[i] * self.normals[i].astype(np.float64)
            plane_normal = self.normals[i].astype(np.float64)
            endpoints = np.array([[self.v0[i]], [self.v1[i]]], dtype=np.float64)
            intersections, valid = trimesh.intersections.plane_lines(
                plane_origin, plane_normal, endpoints, line_segments=True)
            if len(intersections) == 0 or not valid[0]:
                continue
            expected = intersections[0]
            np.testing.assert_allclose(
                out[i].astype(np.float64), expected,
                atol=1e-3, rtol=1e-3,
                err_msg=f"Edge {i}: GPU={out[i]}, trimesh={expected}")


# -------------------------------------------------------------------------
# A3: point_triangle_dist
# -------------------------------------------------------------------------

class TestPointTriangleDist:
    """Test point_triangle_dist against trimesh.proximity.closest_point."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.RandomState(SEED + 2)
        self.points = rng.randn(N, 3).astype(np.float32)
        # Random non-degenerate triangles
        self.triangles = rng.randn(N, 3, 3).astype(np.float32)

    def test_against_trimesh(self):
        pts_flat = self.points.reshape(N, 3).copy()
        tris_flat = self.triangles.reshape(N, 9).copy()
        out = np.zeros(N, dtype=np.float32)
        _gpu.batch_point_triangle_dist(_ptr(pts_flat), _ptr(tris_flat), N, _ptr(out))

        for i in range(N):
            tri_verts = self.triangles[i].astype(np.float64)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            mesh = trimesh.Trimesh(vertices=tri_verts, faces=faces,
                                   process=False)
            closest, distance, _ = trimesh.proximity.closest_point(
                mesh, [self.points[i].astype(np.float64)])
            expected = distance[0]
            np.testing.assert_allclose(
                float(out[i]), expected,
                atol=1e-3, rtol=1e-3,
                err_msg=f"Point {i}: GPU={out[i]}, trimesh={expected}")


# -------------------------------------------------------------------------
# A4: rv_from_volumes
# -------------------------------------------------------------------------

class TestRvFromVolumes:
    """Test rv_from_volumes with plausibility checks."""

    def test_convex_mesh_rv_zero(self):
        """When mesh_vol == hull_vol, Rv should be 0."""
        n = 100
        vols = np.full(n, 1.0, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        _gpu.batch_rv_from_volumes(_ptr(vols), _ptr(vols.copy()), n, 0.3, _ptr(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_nonconvex_rv_positive(self):
        """When mesh_vol < hull_vol, Rv should be positive."""
        n = 100
        mesh_vols = np.full(n, 0.5, dtype=np.float32)
        hull_vols = np.full(n, 1.0, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        _gpu.batch_rv_from_volumes(_ptr(mesh_vols), _ptr(hull_vols), n, 0.3, _ptr(out))
        assert np.all(out > 0), f"Expected all Rv > 0, got min={out.min()}"

    def test_formula_matches_numpy(self):
        """Check exact formula: Rv = (3*|V_mesh - V_hull|/(4*pi))^(1/3) * k"""
        rng = np.random.RandomState(SEED + 3)
        n = N
        mesh_vols = rng.uniform(0.1, 2.0, n).astype(np.float32)
        hull_vols = rng.uniform(0.1, 2.0, n).astype(np.float32)
        rv_k = 0.3
        out = np.zeros(n, dtype=np.float32)
        _gpu.batch_rv_from_volumes(
            _ptr(mesh_vols), _ptr(hull_vols), n, rv_k, _ptr(out))

        diff = np.abs(mesh_vols.astype(np.float64) - hull_vols.astype(np.float64))
        expected = np.cbrt(3.0 * diff / (4.0 * np.pi)) * rv_k
        np.testing.assert_allclose(
            out.astype(np.float64), expected,
            atol=1e-5, rtol=1e-4)

    def test_rv_increases_with_difference(self):
        """Larger volume difference should give larger Rv."""
        n = 10
        mesh_vols = np.ones(n, dtype=np.float32)
        hull_vols = np.arange(1.0, 1.0 + n * 0.1, 0.1, dtype=np.float32)[:n]
        out = np.zeros(n, dtype=np.float32)
        _gpu.batch_rv_from_volumes(_ptr(mesh_vols), _ptr(hull_vols), n, 0.3, _ptr(out))
        # Should be monotonically non-decreasing
        for i in range(1, n):
            assert out[i] >= out[i-1] - 1e-6, \
                f"Rv not monotonic: rv[{i}]={out[i]} < rv[{i-1}]={out[i-1]}"
