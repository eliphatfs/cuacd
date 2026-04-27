"""Tests for GPU bidirectional Hausdorff distance.

Validates hausdorff_block via the test kernel against known geometries:
- Identical meshes (hull == mesh) -> distance ~ 0
- Concentric cubes -> distance ~ half the size difference
- Hull larger than mesh -> positive distance
"""
import numpy as np
import pytest

import cuacd._gpu as _gpu


@pytest.fixture(autouse=True, scope="module")
def gpu_ctx():
    _gpu.init(0)
    yield
    _gpu.destroy()


def _hausdorff(hull_verts, hull_tris, mesh_verts, mesh_tris):
    """Call GPU Hausdorff distance kernel."""
    hv = np.ascontiguousarray(hull_verts, dtype=np.float32)
    ht = np.ascontiguousarray(hull_tris, dtype=np.int32)
    mv = np.ascontiguousarray(mesh_verts, dtype=np.float32)
    mt = np.ascontiguousarray(mesh_tris, dtype=np.int32)
    out = np.zeros(1, dtype=np.float32)
    _gpu.test_hausdorff(
        hv.ctypes.data, len(hv),
        ht.ctypes.data, len(ht),
        mv.ctypes.data, len(mv),
        mt.ctypes.data, len(mt),
        out.ctypes.data)
    return float(out[0])


def _box_mesh(lo, hi):
    """Axis-aligned box mesh with outward-facing normals."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    t = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],   # front
        [2, 3, 7], [2, 7, 6],   # back
        [0, 4, 7], [0, 7, 3],   # left
        [1, 2, 6], [1, 6, 5],   # right
    ], dtype=np.int32)
    return v, t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identical_meshes():
    """Hausdorff distance of a mesh to itself should be ~0."""
    v, t = _box_mesh([0, 0, 0], [1, 1, 1])
    h = _hausdorff(v, t, v, t)
    assert h < 0.05, f"Expected ~0, got {h}"


def test_concentric_cubes():
    """Outer cube hull, inner cube mesh. Distance should be ~0.25."""
    hull_v, hull_t = _box_mesh([-1, -1, -1], [1, 1, 1])  # side 2
    mesh_v, mesh_t = _box_mesh([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])  # side 1
    h = _hausdorff(hull_v, hull_t, mesh_v, mesh_t)
    # The max min-distance from hull sample to mesh surface should be ~0.5
    # (corner of hull at (1,1,1) to nearest point on inner cube face at z=0.5).
    # Actually the corner (1,1,1) to the nearest point on the inner cube
    # is distance sqrt(0.5^2+0.5^2+0.5^2) = ~0.866.
    # And from inner cube corner (-0.5,-0.5,-0.5) to hull surface at z=-1 is 0.5.
    # So the bidirectional Hausdorff should be ~0.866.
    assert 0.4 < h < 1.2, f"Expected ~0.87, got {h}"


def test_hull_encloses_mesh():
    """Hull is a larger box. Hausdorff should be positive."""
    hull_v, hull_t = _box_mesh([0, 0, 0], [2, 2, 2])
    mesh_v, mesh_t = _box_mesh([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    h = _hausdorff(hull_v, hull_t, mesh_v, mesh_t)
    assert h > 0.1, f"Expected positive distance, got {h}"


def test_small_meshes_brute_force():
    """Both meshes < 64 tris: exercises brute-force path."""
    # Two tetrahedra.
    v1 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    t1 = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)

    v2 = v1 * 0.5 + 0.25  # smaller, shifted
    t2 = t1.copy()

    h = _hausdorff(v1, t1, v2, t2)
    assert h > 0.0, f"Expected positive distance, got {h}"
    assert h < 2.0, f"Distance unreasonably large: {h}"


def test_symmetry():
    """Hausdorff(A, B) should approximately equal Hausdorff(B, A)."""
    hull_v, hull_t = _box_mesh([0, 0, 0], [2, 2, 2])
    mesh_v, mesh_t = _box_mesh([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    h1 = _hausdorff(hull_v, hull_t, mesh_v, mesh_t)
    h2 = _hausdorff(mesh_v, mesh_t, hull_v, hull_t)
    # Bidirectional Hausdorff is symmetric by definition.
    assert abs(h1 - h2) < 0.15 * max(h1, h2, 0.01), \
        f"Asymmetric: {h1} vs {h2}"


# ---------------------------------------------------------------------------
# CoACD reference comparison
# ---------------------------------------------------------------------------
import os

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'hausdorff_coacd_ref.npz')


@pytest.mark.skipif(
    not os.path.exists(_FIXTURE_PATH),
    reason="Run tests/gen_hausdorff_fixtures.py to generate reference data")
def test_coacd_reference_comparison():
    """Compare GPU hausdorff against CoACD reference on fixture pairs.

    GPU uses BVH-based exact triangle search; CoACD uses KD-tree with
    10-NN point lookup.  In most cases results should match closely.
    When they differ, GPU should be <= CoACD (BVH finds exact nearest
    triangle, whereas 10-NN may miss it).
    """
    data = np.load(_FIXTURE_PATH, allow_pickle=True)
    n_pairs = int(data['n_pairs'])
    names = data['names']

    for i in range(n_pairs):
        va = data[f'verts_a_{i}'].astype(np.float32)
        ta = data[f'tris_a_{i}'].astype(np.int32)
        vb = data[f'verts_b_{i}'].astype(np.float32)
        tb = data[f'tris_b_{i}'].astype(np.int32)
        ref = float(data[f'ref_{i}'])

        gpu = _hausdorff(va, ta, vb, tb)

        rel_err = abs(gpu - ref) / max(ref, 1e-6)
        # Either results match within 15%, or GPU <= CoACD (expected when
        # CoACD's 10-NN misses the nearest triangle).
        ok = rel_err < 0.15 or gpu <= ref * 1.01
        print(f"  {names[i]}: gpu={gpu:.6f}  coacd={ref:.6f}  "
              f"rel_err={rel_err:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, (
            f"{names[i]}: gpu={gpu:.6f} vs coacd={ref:.6f}, "
            f"rel_err={rel_err:.3f} and gpu > coacd")
