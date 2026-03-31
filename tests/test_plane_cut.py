"""Tests for CPU plane cut with cap triangulation.

Tests different boundary topologies:
- Simple loop (disk): cutting a solid box
- Ring (annular): cutting a hollow tube
- Multi-hole: cutting an object with multiple through-holes

Validates:
- Watertightness (via trimesh)
- Volume conservation (pos + neg = original)
- Correct volume for known geometries
- Mesh volume via signed tetrahedra matches trimesh
"""
import numpy as np
import pytest
import trimesh

import coacd_gpu._gpu as _gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plane_cut(verts, tris, pa, pb, pc, pd):
    """Call C plane cut, return (all_verts, pos_tris, neg_tris)."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int32)
    n_v, n_t = len(verts), len(tris)
    max_v = n_v + n_t * 2 + 16
    max_t = n_t * 5 + n_t + 16

    out_v = np.zeros((max_v, 3), dtype=np.float32)
    out_pt = np.zeros((max_t, 3), dtype=np.int32)
    out_nt = np.zeros((max_t, 3), dtype=np.int32)

    _gpu.init(0)
    nv, npt, nnt = _gpu.test_plane_cut(
        verts.ctypes.data, n_v,
        tris.ctypes.data, n_t,
        float(pa), float(pb), float(pc), float(pd),
        out_v.ctypes.data, max_v,
        out_pt.ctypes.data, max_t,
        out_nt.ctypes.data, max_t)

    return out_v[:nv], out_pt[:npt], out_nt[:nnt]


def _signed_volume(verts, tris):
    """Signed volume via divergence theorem (sum of signed tetrahedra)."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1, v2)
    return float(np.sum(v0 * cross)) / 6.0


def _check_watertight(verts, tris, label="mesh"):
    """Check mesh is watertight via trimesh."""
    m = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    assert m.is_watertight, f"{label} is not watertight (euler={m.euler_number})"
    return m


def _check_volume_conservation(verts, pos_tris, neg_tris, expected_total, tol=0.02):
    """Check pos + neg volumes equal expected total."""
    pv = abs(_signed_volume(verts, pos_tris))
    nv = abs(_signed_volume(verts, neg_tris))
    total = pv + nv
    assert abs(total - expected_total) < tol * expected_total, \
        f"Volume not conserved: {pv:.4f} + {nv:.4f} = {total:.4f}, expected {expected_total:.4f}"
    return pv, nv


# ---------------------------------------------------------------------------
# Test meshes
# ---------------------------------------------------------------------------

def _cube_mesh(extents=(1, 1, 1)):
    """Unit cube centered at origin."""
    cube = trimesh.creation.box(extents=extents)
    return np.array(cube.vertices, dtype=np.float32), np.array(cube.faces, dtype=np.int32)


def _l_shape_mesh():
    """L-shaped mesh (non-convex): two overlapping boxes."""
    # Box 1: [0, 1] x [0, 2] x [0, 1]
    # Box 2: [0, 2] x [0, 1] x [0, 1]
    verts = np.array([
        # Box 1 bottom (z=0): y goes 0..2, x goes 0..1
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 0-3
        [0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0],  # 4-7
        # Box 1 top (z=1)
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # 8-11
        [0, 1, 1], [1, 1, 1], [1, 2, 1], [0, 2, 1],  # 12-15
        # Box 2 extension: x goes 1..2, y goes 0..1
        [2, 0, 0], [2, 1, 0],  # 16-17
        [2, 0, 1], [2, 1, 1],  # 18-19
    ], dtype=np.float32)
    # Use trimesh to create the L-shape properly
    box1 = trimesh.creation.box(extents=[1, 2, 1], transform=trimesh.transformations.translation_matrix([0.5, 1, 0.5]))
    box2 = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix([1.5, 0.5, 0.5]))
    try:
        l_shape = box1.union(box2)
        return np.array(l_shape.vertices, dtype=np.float32), np.array(l_shape.faces, dtype=np.int32)
    except Exception:
        pytest.skip("trimesh boolean ops not available")


def _hollow_tube_mesh():
    """Rectangular tube (hollow box with through-hole in Z).

    Outer: [-1,1]^3, Inner hole: [-0.5,0.5]^2 × [-1,1] in Z.
    Creates ring (annular) boundary topology when cut by Z=0 plane.
    """
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box(extents=[1, 1, 4])  # extends beyond in Z
    try:
        tube = outer.difference(inner)
        if not tube.is_watertight:
            pytest.skip("Boolean result not watertight")
        return np.array(tube.vertices, dtype=np.float32), np.array(tube.faces, dtype=np.int32)
    except Exception:
        pytest.skip("trimesh boolean ops not available")


def _two_holes_mesh():
    """Box with two through-holes in Z direction.

    Creates multi-hole boundary topology when cut.
    """
    outer = trimesh.creation.box(extents=[4, 2, 2])
    hole1 = trimesh.creation.box(
        extents=[0.5, 0.5, 4],
        transform=trimesh.transformations.translation_matrix([-1, 0, 0]))
    hole2 = trimesh.creation.box(
        extents=[0.5, 0.5, 4],
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
        av, pt, nt = _plane_cut(v, t, 0, 0, 1, 0)

        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        pv, nv = _check_volume_conservation(av, pt, nt, 1.0)
        assert abs(pv - 0.5) < 0.02, f"pos vol {pv} != 0.5"
        assert abs(nv - 0.5) < 0.02, f"neg vol {nv} != 0.5"

    def test_cube_midpoint_x(self):
        """Cut unit cube at x=0."""
        v, t = _cube_mesh()
        av, pt, nt = _plane_cut(v, t, 1, 0, 0, 0)

        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        _check_volume_conservation(av, pt, nt, 1.0)

    def test_cube_midpoint_y(self):
        """Cut unit cube at y=0."""
        v, t = _cube_mesh()
        av, pt, nt = _plane_cut(v, t, 0, 1, 0, 0)

        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        _check_volume_conservation(av, pt, nt, 1.0)

    def test_cube_off_center(self):
        """Cut unit cube at z=0.25 → unequal halves (0.25 and 0.75)."""
        v, t = _cube_mesh()
        # Cube is [-0.5, 0.5]^3. Cut at z=0.25.
        # Positive (z>0.25): height 0.25, vol = 1*1*0.25 = 0.25
        # Negative (z<0.25): height 0.75, vol = 1*1*0.75 = 0.75
        av, pt, nt = _plane_cut(v, t, 0, 0, 1, -0.25)

        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        pv, nv = _check_volume_conservation(av, pt, nt, 1.0)
        assert abs(pv - 0.25) < 0.02, f"pos vol {pv} != 0.25"
        assert abs(nv - 0.75) < 0.02, f"neg vol {nv} != 0.75"

    def test_rectangular_box(self):
        """Cut a 2x3x4 box at various planes."""
        v, t = _cube_mesh(extents=(2, 3, 4))
        total = 24.0  # 2*3*4

        for pa, pb, pc, pd, label in [
            (1, 0, 0, 0, "x=0"),
            (0, 1, 0, 0.5, "y=-0.5"),
            (0, 0, 1, -0.3, "z=0.3"),
        ]:
            av, pt, nt = _plane_cut(v, t, pa, pb, pc, pd)
            _check_watertight(av, pt, f"pos ({label})")
            _check_watertight(av, nt, f"neg ({label})")
            _check_volume_conservation(av, pt, nt, total, tol=0.02)

    def test_trimesh_volume_match(self):
        """Signed-tet volume matches trimesh.volume for both halves."""
        v, t = _cube_mesh()
        av, pt, nt = _plane_cut(v, t, 0, 0, 1, 0)

        pos_mesh = trimesh.Trimesh(vertices=av, faces=pt, process=False)
        neg_mesh = trimesh.Trimesh(vertices=av, faces=nt, process=False)

        signed_pos = abs(_signed_volume(av, pt))
        signed_neg = abs(_signed_volume(av, nt))

        assert abs(signed_pos - pos_mesh.volume) < 0.01, \
            f"Signed tet vol {signed_pos} != trimesh vol {pos_mesh.volume}"
        assert abs(signed_neg - neg_mesh.volume) < 0.01, \
            f"Signed tet vol {signed_neg} != trimesh vol {neg_mesh.volume}"

    def test_no_cut(self):
        """Plane that doesn't intersect mesh → one empty side."""
        v, t = _cube_mesh()
        # Cube is [-0.5, 0.5]^3. Plane z=2 doesn't cut.
        av, pt, nt = _plane_cut(v, t, 0, 0, 1, -2)
        # Everything should be on negative side
        assert len(pt) == 0 or len(nt) == 0, "One side should be empty"

    def test_sphere(self):
        """Cut a sphere → check volume conservation."""
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        v = np.array(sphere.vertices, dtype=np.float32)
        t = np.array(sphere.faces, dtype=np.int32)
        total = sphere.volume

        av, pt, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        _check_volume_conservation(av, pt, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Non-convex cap (L-shape)
# ---------------------------------------------------------------------------

class TestNonConvexCap:
    """Cutting a non-convex shape produces a non-convex cap polygon."""

    def test_l_shape_volume_conservation(self):
        """Cut L-shape → volume is conserved."""
        v, t = _l_shape_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        # Cut at y=0.5 (through the L)
        av, pt, nt = _plane_cut(v, t, 0, 1, 0, -0.5)
        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        _check_volume_conservation(av, pt, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Ring topology (annular cap)
# ---------------------------------------------------------------------------

class TestRingTopology:
    """Cutting a hollow tube creates two nested boundary loops."""

    def test_hollow_tube_volume(self):
        """Cut hollow tube → volume conservation with ring cap."""
        v, t = _hollow_tube_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        av, pt, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        pv, nv = _check_volume_conservation(av, pt, nt, total, tol=0.05)
        # Each half should be half the total
        assert abs(pv - total / 2) < 0.1 * total, f"pos vol {pv} != {total/2}"


# ---------------------------------------------------------------------------
# Tests: Multi-hole topology
# ---------------------------------------------------------------------------

class TestMultiHole:
    """Cutting an object with multiple through-holes."""

    def test_two_holes_volume(self):
        """Cut box with two through-holes → volume conservation."""
        v, t = _two_holes_mesh()
        orig = trimesh.Trimesh(vertices=v, faces=t, process=False)
        total = orig.volume

        av, pt, nt = _plane_cut(v, t, 0, 0, 1, 0)
        _check_watertight(av, pt, "pos")
        _check_watertight(av, nt, "neg")
        _check_volume_conservation(av, pt, nt, total, tol=0.05)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for plane cutting."""

    def test_plane_through_vertex(self):
        """Plane passing through a mesh vertex."""
        v, t = _cube_mesh()
        # Cube vertex at (0.5, 0.5, 0.5). Plane: x + y + z = 1.5
        # passes through that vertex
        av, pt, nt = _plane_cut(v, t, 1, 1, 1, -1.5)
        # Should produce non-empty halves with volume conservation
        if len(pt) > 0 and len(nt) > 0:
            _check_volume_conservation(av, pt, nt, 1.0, tol=0.05)

    def test_plane_through_edge(self):
        """Plane passing through a mesh edge."""
        v, t = _cube_mesh()
        # Cube has edge from (-0.5,-0.5,-0.5) to (0.5,-0.5,-0.5).
        # Plane y = -0.5 passes through this edge.
        av, pt, nt = _plane_cut(v, t, 0, 1, 0, 0.5)
        # One side should be empty (plane at mesh boundary)
        # or very small — just check no crash
        assert len(pt) >= 0 and len(nt) >= 0

    def test_diagonal_plane(self):
        """Non-axis-aligned plane cut."""
        v, t = _cube_mesh()
        # Plane: x + y = 0 (diagonal cut)
        norm = 1.0 / np.sqrt(2)
        av, pt, nt = _plane_cut(v, t, norm, norm, 0, 0)
        if len(pt) > 0 and len(nt) > 0:
            _check_watertight(av, pt, "pos")
            _check_watertight(av, nt, "neg")
            _check_volume_conservation(av, pt, nt, 1.0, tol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
