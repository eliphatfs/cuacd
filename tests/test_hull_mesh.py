"""Tests for D&C hull mesh extraction (hull_dandc kernel).

Verifies that the extracted hull mesh:
- Has correct vertex/triangle counts
- Produces valid triangle indices
- Has positive volume (computed via test_mesh_volume)
"""

import numpy as np
import pytest

try:
    from coacd_gpu import _gpu
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="GPU extension not available")


@pytest.fixture(autouse=True)
def gpu_ctx():
    _gpu.init(0)
    yield
    _gpu.destroy()


def _cube_points():
    """8 vertices of a unit cube."""
    return np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)


def _tetrahedron_points():
    """4 vertices of a regular tetrahedron."""
    return np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
    ], dtype=np.float32)


def _run_hull_dandc(pts, max_hv=None, max_ht=None):
    """Run hull_dandc on a single point cloud; return (verts, tris, err)."""
    n = len(pts)
    if max_hv is None:
        max_hv = n
    if max_ht is None:
        max_ht = max(2 * n - 4, 4)
    offsets = np.array([0, n], dtype=np.int32)
    out_verts  = np.zeros(max_hv * 3, dtype=np.float32)
    out_tris   = np.zeros(max_ht * 3, dtype=np.int32)
    out_nv     = np.zeros(1, dtype=np.int32)
    out_nt     = np.zeros(1, dtype=np.int32)
    out_errors = np.zeros(1, dtype=np.int32)

    _gpu.hull_dandc(
        pts.ctypes.data, n,
        offsets.ctypes.data, 1,
        n, max_hv, max_ht,
        out_verts.ctypes.data, out_tris.ctypes.data,
        out_nv.ctypes.data, out_nt.ctypes.data,
        out_errors.ctypes.data)

    nv = int(out_nv[0])
    nt = int(out_nt[0])
    verts = out_verts[:nv * 3].reshape(nv, 3).copy()
    tris  = out_tris [:nt * 3].reshape(nt, 3).copy()
    return verts, tris, int(out_errors[0])


def _mesh_volume(verts, tris):
    """CPU divergence-theorem volume."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return abs(float(np.sum(v0 * np.cross(v1, v2)))) / 6.0


def _gpu_mesh_volume(verts, tris):
    """GPU mesh volume via test_mesh_volume."""
    verts_c = np.ascontiguousarray(verts, dtype=np.float32)
    tris_c  = np.ascontiguousarray(tris,  dtype=np.int32)
    vol = np.zeros(1, dtype=np.float32)
    _gpu.test_mesh_volume(
        verts_c.ctypes.data, len(verts_c),
        tris_c.ctypes.data,  len(tris_c),
        vol.ctypes.data)
    return float(vol[0])


class TestHullMeshExtraction:
    def test_cube_counts(self):
        """Cube hull should have 8 vertices and 12 triangles."""
        pts = _cube_points()
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        assert len(verts) == 8,  f"Expected 8 hull vertices, got {len(verts)}"
        assert len(tris)  == 12, f"Expected 12 hull triangles, got {len(tris)}"

    def test_tetrahedron_counts(self):
        """Tetrahedron hull should have 4 vertices and 4 triangles."""
        pts = _tetrahedron_points()
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        assert len(verts) == 4, f"Expected 4 hull vertices, got {len(verts)}"
        assert len(tris)  == 4, f"Expected 4 hull triangles, got {len(tris)}"

    def test_cube_volume_gpu(self):
        """Hull mesh GPU volume should be ~1.0 for a unit cube."""
        pts = _cube_points()
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        vol = _gpu_mesh_volume(verts, tris)
        assert abs(vol - 1.0) < 0.01, f"GPU hull volume {vol} != 1.0"

    def test_cube_volume_cpu_matches_gpu(self):
        """CPU and GPU mesh volumes of cube hull should agree within 0.01%."""
        pts = _cube_points()
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        cpu_vol = _mesh_volume(verts, tris)
        gpu_vol = _gpu_mesh_volume(verts, tris)
        rel_err = abs(cpu_vol - gpu_vol) / max(cpu_vol, 1e-12)
        assert rel_err < 1e-4, f"cpu={cpu_vol:.6f}, gpu={gpu_vol:.6f}"

    def test_valid_triangle_indices(self):
        """Triangle indices must be in [0, nv-1]."""
        pts = _cube_points()
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        assert tris.min() >= 0
        assert tris.max() < len(verts)

    def test_gaussian_hull_mesh(self):
        """Random gaussian points should produce a valid hull mesh."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((100, 3)).astype(np.float32)
        verts, tris, err = _run_hull_dandc(pts)
        assert err == 0
        assert len(verts) >= 4
        assert len(tris)  >= 4
        assert tris.min() >= 0
        assert tris.max() < len(verts)
        vol = _gpu_mesh_volume(verts, tris)
        assert vol > 0, f"Gaussian hull volume is {vol}"

    def test_batch_multiple_hulls(self):
        """Batch of 4 hulls should all succeed and return valid meshes."""
        rng = np.random.default_rng(7)
        pts_list = [rng.standard_normal((50, 3)).astype(np.float32) for _ in range(4)]
        packed = np.concatenate(pts_list, axis=0)
        offsets = np.array([0, 50, 100, 150, 200], dtype=np.int32)
        n_hulls = 4
        max_pts = 50
        max_hv = max_pts
        max_ht = max(2 * max_pts - 4, 4)

        out_verts  = np.zeros(n_hulls * max_hv * 3, dtype=np.float32)
        out_tris   = np.zeros(n_hulls * max_ht * 3, dtype=np.int32)
        out_nv     = np.zeros(n_hulls, dtype=np.int32)
        out_nt     = np.zeros(n_hulls, dtype=np.int32)
        out_errors = np.zeros(n_hulls, dtype=np.int32)

        _gpu.hull_dandc(
            packed.ctypes.data, len(packed),
            offsets.ctypes.data, n_hulls,
            max_pts, max_hv, max_ht,
            out_verts.ctypes.data, out_tris.ctypes.data,
            out_nv.ctypes.data, out_nt.ctypes.data,
            out_errors.ctypes.data)

        for i in range(n_hulls):
            assert out_errors[i] == 0, f"Hull {i}: error {out_errors[i]}"
            nv = int(out_nv[i])
            nt = int(out_nt[i])
            assert nv >= 4, f"Hull {i}: only {nv} vertices"
            assert nt >= 4, f"Hull {i}: only {nt} triangles"
            tris_i = out_tris[i * max_ht * 3 : i * max_ht * 3 + nt * 3].reshape(nt, 3)
            assert tris_i.min() >= 0
            assert tris_i.max() < nv
