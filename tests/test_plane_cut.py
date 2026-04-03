"""Tests for GPU plane cut with cap triangulation.

Tests different boundary topologies:
- Simple loop (disk): cutting a solid box
- Ring (annular): cutting a hollow tube
- Multi-hole: cutting an object with multiple through-holes

Validates:
- Watertightness (via trimesh)
- Volume conservation (pos + neg = original)
- Correct volume for known geometries
"""
import numpy as np
import pytest
import trimesh

import coacd_gpu._gpu as _gpu


# ---------------------------------------------------------------------------
# Context fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def gpu_ctx():
    _gpu.init(0)
    yield
    _gpu.destroy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plane_cut(verts, tris, pa, pb, pc, pd):
    """Call GPU plane cut; return (pos_verts, pos_tris, neg_verts, neg_tris)."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris  = np.ascontiguousarray(tris,  dtype=np.int32)
    n_v, n_t = len(verts), len(tris)

    # Conservative upper bounds
    max_v = n_v + n_t * 2 + 16
    max_t = n_t * 5 + 16

    out_pv = np.zeros((max_v, 3), dtype=np.float32)
    out_pt = np.zeros((max_t, 3), dtype=np.int32)
    out_nv = np.zeros((max_v, 3), dtype=np.float32)
    out_nt = np.zeros((max_t, 3), dtype=np.int32)

    n_pv, n_pt, n_nv, n_nt = _gpu.test_plane_cut(
        verts.ctypes.data, n_v,
        tris.ctypes.data,  n_t,
        float(pa), float(pb), float(pc), float(pd),
        out_pv.ctypes.data, max_v,
        out_pt.ctypes.data, max_t,
        out_nv.ctypes.data, max_v,
        out_nt.ctypes.data, max_t)

    return (out_pv[:n_pv], out_pt[:n_pt],
            out_nv[:n_nv], out_nt[:n_nt])


def _signed_volume(verts, tris):
    """Signed volume via divergence theorem."""
    if len(tris) == 0:
        return 0.0
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return float(np.sum(v0 * np.cross(v1, v2))) / 6.0


def _check_watertight(verts, tris, label="mesh"):
    """Check mesh is watertight via trimesh."""
    if len(tris) == 0:
        return None
    m = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    assert m.is_watertight, f"{label} is not watertight (euler={m.euler_number})"
    return m


def _check_volume_conservation(pv, pt, nv, nt, expected_total, tol=0.02):
    """Check pos + neg volumes equal expected total (separate verts per side)."""
    pos_vol = abs(_signed_volume(pv, pt))
    neg_vol = abs(_signed_volume(nv, nt))
    total = pos_vol + neg_vol
    assert abs(total - expected_total) < tol * expected_total, \
        f"Volume not conserved: {pos_vol:.4f} + {neg_vol:.4f} = {total:.4f}, expected {expected_total:.4f}"
    return pos_vol, neg_vol


# ---------------------------------------------------------------------------
# Test meshes
# ---------------------------------------------------------------------------

def _cube_mesh(extents=(1, 1, 1)):
    """Trimesh box, centered at origin."""
    cube = trimesh.creation.box(extents=extents)
    return np.array(cube.vertices, dtype=np.float32), np.array(cube.faces, dtype=np.int32)


def _l_shape_mesh():
    """L-shaped mesh (non-convex): two overlapping boxes."""
    box1 = trimesh.creation.box(extents=[1, 2, 1],
        transform=trimesh.transformations.translation_matrix([0.5, 1, 0.5]))
    box2 = trimesh.creation.box(extents=[1, 1, 1],
        transform=trimesh.transformations.translation_matrix([1.5, 0.5, 0.5]))
    try:
        l_shape = box1.union(box2)
        return np.array(l_shape.vertices, dtype=np.float32), np.array(l_shape.faces, dtype=np.int32)
    except Exception:
        pytest.skip("trimesh boolean ops not available")


def _hollow_tube_mesh():
    """Rectangular tube (hollow box with through-hole in Z)."""
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box(extents=[1, 1, 4])
    try:
        tube = outer.difference(inner)
        if not tube.is_watertight:
            pytest.skip("Boolean result not watertight")
        return np.array(tube.vertices, dtype=np.float32), np.array(tube.faces, dtype=np.int32)
    except Exception:
        pytest.skip("trimesh boolean ops not available")


def _two_holes_mesh():
    """Box with two through-holes in Z direction."""
    outer = trimesh.creation.box(extents=[4, 2, 2])
    hole1 = trimesh.creation.box(extents=[0.5, 0.5, 4],
        transform=trimesh.transformations.translation_matrix([-1, 0, 0]))
    hole2 = trimesh.creation.box(extents=[0.5, 0.5, 4],
        transform=trimesh.transformations.translation_matrix([1, 0, 0]))
    try:
        result = outer.difference(hole1).difference(hole2)
        if not result.is_watertight:
            pytest.skip("Boolean result not watertight")
        return np.array(result.vertices, dtype=np.float32), np.array(result.faces, dtype=np.int32)
    except Exception:
        pytest.skip("trimesh boolean ops not available")


# ---------------------------------------------------------------------------
# Tests: Simple loop topology (disk cap)
# ---------------------------------------------------------------------------

class TestSimpleLoop:
    """Cutting a solid box produces a simple (single-loop) boundary."""

    def test_cube_midpoint_z(self):
        """Cut unit cube at z=0 → two equal halves."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)

        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        pos_vol, neg_vol = _check_volume_conservation(pv, pt, nv, nt, 1.0)
        assert abs(pos_vol - 0.5) < 0.02, f"pos vol {pos_vol} != 0.5"
        assert abs(neg_vol - 0.5) < 0.02, f"neg vol {neg_vol} != 0.5"

    def test_cube_midpoint_x(self):
        """Cut unit cube at x=0."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 1, 0, 0, 0)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, 1.0)

    def test_cube_midpoint_y(self):
        """Cut unit cube at y=0."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 1, 0, 0)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, 1.0)

    def test_cube_off_center(self):
        """Cut unit cube at z=0.25 → unequal halves."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, -0.25)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        pos_vol, neg_vol = _check_volume_conservation(pv, pt, nv, nt, 1.0)
        assert abs(pos_vol - 0.25) < 0.02, f"pos vol {pos_vol} != 0.25"
        assert abs(neg_vol - 0.75) < 0.02, f"neg vol {neg_vol} != 0.75"

    def test_rectangular_box(self):
        """Cut a 2x3x4 box at various planes."""
        v, t = _cube_mesh(extents=(2, 3, 4))
        total = 24.0

        for pa, pb, pc, pd, label in [
            (1, 0, 0, 0, "x=0"),
            (0, 1, 0, 0.5, "y=-0.5"),
            (0, 0, 1, -0.3, "z=0.3"),
        ]:
            pv, pt, nv, nt = _plane_cut(v, t, pa, pb, pc, pd)
            _check_watertight(pv, pt, f"pos ({label})")
            _check_watertight(nv, nt, f"neg ({label})")
            _check_volume_conservation(pv, pt, nv, nt, total, tol=0.02)

    def test_trimesh_volume_match(self):
        """Signed-tet volume matches trimesh.volume for both halves."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)

        pos_mesh = trimesh.Trimesh(vertices=pv, faces=pt, process=False)
        neg_mesh = trimesh.Trimesh(vertices=nv, faces=nt, process=False)

        signed_pos = abs(_signed_volume(pv, pt))
        signed_neg = abs(_signed_volume(nv, nt))

        assert abs(signed_pos - pos_mesh.volume) < 0.01, \
            f"Signed tet vol {signed_pos} != trimesh vol {pos_mesh.volume}"
        assert abs(signed_neg - neg_mesh.volume) < 0.01, \
            f"Signed tet vol {signed_neg} != trimesh vol {neg_mesh.volume}"

    def test_no_cut(self):
        """Plane that doesn't intersect mesh → one empty side."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, -2)
        assert len(pt) == 0 or len(nt) == 0, "One side should be empty"

    def test_sphere(self):
        """Cut a sphere → check volume conservation."""
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        v = np.array(sphere.vertices, dtype=np.float32)
        t = np.array(sphere.faces, dtype=np.int32)
        total = sphere.volume

        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Non-convex cap (L-shape)
# ---------------------------------------------------------------------------

class TestNonConvexCap:
    def test_l_shape_volume_conservation(self):
        """Cut L-shape → volume is conserved."""
        v, t = _l_shape_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        pv, pt, nv, nt = _plane_cut(v, t, 0, 1, 0, -0.5)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Ring topology (annular cap)
# ---------------------------------------------------------------------------

class TestRingTopology:
    def test_hollow_tube_volume(self):
        """Cut hollow tube → volume conservation with ring cap."""
        v, t = _hollow_tube_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        pos_vol, neg_vol = _check_volume_conservation(pv, pt, nv, nt, total, tol=0.05)
        assert abs(pos_vol - total / 2) < 0.1 * total, f"pos vol {pos_vol} != {total/2}"


# ---------------------------------------------------------------------------
# Tests: Multi-hole topology
# ---------------------------------------------------------------------------

class TestMultiHole:
    def test_two_holes_volume(self):
        """Cut box with two through-holes → volume conservation."""
        v, t = _two_holes_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_plane_through_vertex(self):
        """Plane passing through a mesh vertex."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 1, 1, 1, -1.5)
        if len(pt) > 0 and len(nt) > 0:
            _check_volume_conservation(pv, pt, nv, nt, 1.0, tol=0.05)

    def test_plane_through_edge(self):
        """Plane passing through a mesh edge — no crash."""
        v, t = _cube_mesh()
        pv, pt, nv, nt = _plane_cut(v, t, 0, 1, 0, 0.5)
        assert len(pt) >= 0 and len(nt) >= 0

    def test_diagonal_plane(self):
        """Non-axis-aligned plane cut."""
        v, t = _cube_mesh()
        norm = 1.0 / np.sqrt(2)
        pv, pt, nv, nt = _plane_cut(v, t, norm, norm, 0, 0)
        if len(pt) > 0 and len(nt) > 0:
            _check_watertight(pv, pt, "pos")
            _check_watertight(nv, nt, "neg")
            _check_volume_conservation(pv, pt, nv, nt, 1.0, tol=0.05)

    def test_two_disjoint_cubes(self):
        """Two disjoint cubes cut by z=0 — both split, watertight, volume conserved."""
        c = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        cv = np.array(c.vertices, dtype=np.float32)
        ct = np.array(c.faces, dtype=np.int32)
        # Cube A centered at (-1, 0, 0), Cube B at (+1, 0, 0)
        va = cv.copy(); va[:, 0] -= 1.0
        vb = cv.copy(); vb[:, 0] += 1.0
        v = np.vstack([va, vb])
        t = np.vstack([ct, ct + len(va)])
        total_vol = abs(_signed_volume(v, t))
        # Cut with z=0
        pv, pt, nv, nt = _plane_cut(v, t, 0, 0, 1, 0)
        assert len(pt) > 0 and len(nt) > 0, "Expected split on both sides"
        _check_watertight(pv, pt, "pos")
        _check_watertight(nv, nt, "neg")
        _check_volume_conservation(pv, pt, nv, nt, total_vol, tol=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
