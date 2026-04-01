"""
Tests for GPU beam-search convex decomposition (coacd_gpu.Context.decompose).

Shapes tested:
  - Icosphere  — convex,          expect 1 part
  - Cube       — convex,          expect 1 part
  - L-shape    — mildly concave,  expect 2-4 parts
  - Octocat    — complex,         expect multiple parts

Includes CoACD-style visualization (writes PLY/OBJ per part) when VISUALIZE=1.
"""

import os
import math
import numpy as np
import pytest
import coacd_gpu

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _icosphere(subdivisions=2):
    """Generate an icosphere mesh, centered at origin, radius 1."""
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float32)
    verts /= np.linalg.norm(verts[0])

    tris = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)

    for _ in range(subdivisions):
        edge_mid = {}
        new_tris = []
        for t in tris:
            mids = []
            for i in range(3):
                a, b = int(t[i]), int(t[(i+1)%3])
                key = (min(a,b), max(a,b))
                if key not in edge_mid:
                    m = (verts[a] + verts[b]) * 0.5
                    m /= np.linalg.norm(m)
                    edge_mid[key] = len(verts)
                    verts = np.vstack([verts, m])
                mids.append(edge_mid[key])
            v0,v1,v2 = int(t[0]),int(t[1]),int(t[2])
            m0,m1,m2 = mids
            new_tris += [[v0,m0,m2],[v1,m1,m0],[v2,m2,m1],[m0,m1,m2]]
        tris = np.array(new_tris, dtype=np.int32)

    return verts.astype(np.float32), tris


def _box_mesh(x0, y0, z0, x1, y1, z1):
    """Axis-aligned box mesh with 8 verts, 12 triangles."""
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float32)
    t = np.array([
        [0,2,1],[0,3,2],  # -z
        [4,5,6],[4,6,7],  # +z
        [0,1,5],[0,5,4],  # -y
        [2,3,7],[2,7,6],  # +y
        [0,4,7],[0,7,3],  # -x
        [1,2,6],[1,6,5],  # +x
    ], dtype=np.int32)
    return v, t


def _merge_meshes(meshes):
    """Union of meshes that don't overlap (just vertex-pack + triangle-offset)."""
    all_v, all_t = [], []
    offset = 0
    for v, t in meshes:
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    return np.concatenate(all_v), np.concatenate(all_t)


def _l_shape():
    """L-shaped mesh: union of two non-overlapping boxes."""
    b1 = _box_mesh(-1, -1, -0.5, 1, 0, 0.5)   # horizontal arm
    b2 = _box_mesh(-1,  0, -0.5, 0, 1, 0.5)   # vertical arm
    return _merge_meshes([b1, b2])


def _signed_mesh_volume(verts, tris):
    v = verts[tris]
    return abs(np.sum(np.einsum('ij,ij->i',
                                v[:,0], np.cross(v[:,1], v[:,2]))) / 6.0)


# ---------------------------------------------------------------------------
# Visualization helper (CoACD-style)
# ---------------------------------------------------------------------------

def _visualize(parts, name, out_dir="decomp_output"):
    """
    Save each convex part as a colored OBJ.
    Mimics CoACD's visualization in coacd#L174.
    """
    try:
        import trimesh
    except ImportError:
        return

    os.makedirs(out_dir, exist_ok=True)
    meshes = []
    rng = np.random.default_rng(42)
    for i, (hv, ht) in enumerate(parts):
        color = (rng.random(3) * 255).astype(np.uint8)
        m = trimesh.Trimesh(vertices=hv, faces=ht, process=False)
        m.visual.face_colors = np.tile(
            np.append(color, 200), (len(ht), 1))
        meshes.append(m)
        m.export(os.path.join(out_dir, f"{name}_part_{i:03d}.obj"))

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(os.path.join(out_dir, f"{name}_all.obj"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx():
    with coacd_gpu.Context() as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIcosphere:
    def test_convex_single_part(self, ctx):
        """Icosphere is convex — should produce exactly 1 part."""
        verts, tris = _icosphere(subdivisions=2)
        parts = ctx.decompose(verts, tris, threshold=0.05,
                              beam_width=30, cuts_per_axis=10)
        _visualize(parts, "icosphere")
        assert len(parts) == 1, f"Expected 1 part, got {len(parts)}"
        hv, ht = parts[0]
        assert hv.shape[1] == 3 and ht.shape[1] == 3
        assert len(hv) >= 4 and len(ht) >= 4

    def test_volume_preserved(self, ctx):
        """Hull volume of result should approximate input mesh volume."""
        verts, tris = _icosphere(subdivisions=2)
        parts = ctx.decompose(verts, tris, threshold=0.05)
        assert len(parts) == 1
        hv, ht = parts[0]
        vol_input = _signed_mesh_volume(verts, tris)
        vol_hull  = _signed_mesh_volume(hv, ht)
        assert abs(vol_hull - vol_input) / max(vol_input, 1e-8) < 0.05


class TestCube:
    def test_convex_single_part(self, ctx):
        """Cube is convex — should produce exactly 1 part."""
        verts, tris = _box_mesh(-1, -1, -1, 1, 1, 1)
        parts = ctx.decompose(verts, tris, threshold=0.05,
                              beam_width=30, cuts_per_axis=10)
        _visualize(parts, "cube")
        assert len(parts) == 1, f"Expected 1 part, got {len(parts)}"


class TestLShape:
    def test_concave_multiple_parts(self, ctx):
        """L-shape is concave — should require 2-4 parts."""
        verts, tris = _l_shape()
        parts = ctx.decompose(verts, tris, threshold=0.05,
                              beam_width=30, cuts_per_axis=10)
        _visualize(parts, "l_shape")
        assert 1 < len(parts) <= 5, (
            f"Expected 2-5 parts for L-shape, got {len(parts)}")

    def test_watertight_parts(self, ctx):
        """Each returned hull mesh should be watertight (trimesh check)."""
        trimesh = pytest.importorskip("trimesh")
        verts, tris = _l_shape()
        parts = ctx.decompose(verts, tris, threshold=0.05)
        for i, (hv, ht) in enumerate(parts):
            m = trimesh.Trimesh(vertices=hv, faces=ht, process=False)
            assert m.is_watertight, f"Part {i} is not watertight"


class TestOctocat:
    def test_decomposes(self, ctx):
        """Octocat is complex — should produce multiple parts with threshold=0.05."""
        obj_path = os.path.join(
            os.path.dirname(__file__), "..", "CoACD", "examples", "Octocat-v2.obj")
        if not os.path.exists(obj_path):
            pytest.skip("Octocat-v2.obj not found")

        trimesh_mod = pytest.importorskip("trimesh")
        mesh = trimesh_mod.load(obj_path, force="mesh")
        verts = np.array(mesh.vertices, dtype=np.float32)
        tris  = np.array(mesh.faces,    dtype=np.int32)

        # Normalize to [-1,1]
        center = (verts.max(axis=0) + verts.min(axis=0)) * 0.5
        scale  = (verts.max(axis=0) - verts.min(axis=0)).max()
        if scale > 1e-8:
            verts = (verts - center) / (scale * 0.5)

        parts = ctx.decompose(verts, tris, threshold=0.05,
                              beam_width=30, cuts_per_axis=10, max_iters=20)
        _visualize(parts, "octocat")
        assert len(parts) >= 2, (
            f"Expected multiple parts for Octocat, got {len(parts)}")
        for hv, ht in parts:
            assert hv.shape[1] == 3 and ht.shape[1] == 3
            assert len(hv) >= 4 and len(ht) >= 4
