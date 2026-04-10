"""Test beam_decompose on a cube, an L-shape, and the Octocat mesh.

Shapes:
  - Cube [0,1]^3: already convex → should produce 1 part
  - L-shape (two boxes joined at a corner): non-convex → 2-4 parts
  - Octocat-v2.obj: complex organic shape → several parts

Each shape is normalized to bbox [-1, 1] before decomposition, then
rescaled back.
"""
import os
import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull

import coacd_gpu._gpu as _gpu

OCTOCAT_OBJ  = os.path.join(os.path.dirname(__file__),
                            "../CoACD/examples/Octocat-v2.obj")
STL_49160    = os.path.join(os.path.dirname(__file__),
                            "data/49160.stl")


@pytest.fixture(autouse=True, scope="module")
def gpu_ctx():
    _gpu.init(0, 10_000_000_000)
    yield
    _gpu.destroy()


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _make_cube():
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    t = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,4,7],[0,7,3],[1,2,6],[1,6,5],
    ], dtype=np.int32)
    return v, t


def _make_lshape():
    """Manifold L-shape = boolean union of two boxes (watertight, no internal faces)."""
    box_a = trimesh.creation.box(extents=[2, 1, 1],
        transform=trimesh.transformations.translation_matrix([1, 0.5, 0.5]))
    box_b = trimesh.creation.box(extents=[1, 2, 1],
        transform=trimesh.transformations.translation_matrix([0.5, 2, 0.5]))
    l_shape = box_a.union(box_b)
    return (np.ascontiguousarray(l_shape.vertices, dtype=np.float32),
            np.ascontiguousarray(l_shape.faces, dtype=np.int32))


def _scipy_hull(verts):
    """Compute convex hull via scipy; return (hull_verts float32, hull_tris int32).

    Reorient each triangle so its normal points outward (away from centroid),
    ensuring consistent winding for the divergence theorem volume computation.
    """
    ch = ConvexHull(verts)
    hull_verts = verts[ch.vertices].astype(np.float32)
    remap = {old: new for new, old in enumerate(ch.vertices)}
    tris = np.array([[remap[i] for i in tri] for tri in ch.simplices],
                    dtype=np.int32)

    centroid = hull_verts.mean(axis=0)
    v0 = hull_verts[tris[:, 0]]
    v1 = hull_verts[tris[:, 1]]
    v2 = hull_verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    outward = v0 - centroid
    inward  = (normals * outward).sum(axis=1) < 0
    tris[inward] = tris[inward][:, [0, 2, 1]]

    return hull_verts, tris


def _normalize(verts):
    """Scale verts so bbox spans [-1, 1] on all axes. Returns (verts_norm, center, scale)."""
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2.0
    scale  = (hi - lo).max() / 2.0
    if scale == 0:
        scale = 1.0
    return ((verts - center) / scale).astype(np.float32), center, float(scale)


def _denormalize(verts, center, scale):
    """Undo _normalize."""
    return (verts * scale + center).astype(np.float32)


def _call_decompose(verts, tris, hull_verts, hull_tris,
                    max_iters=100, cuts_per_axis=10,
                    threshold=0.05, max_keep=32, verbose=0, debug=0):
    verts      = np.ascontiguousarray(verts,      dtype=np.float32)
    tris       = np.ascontiguousarray(tris,       dtype=np.int32)
    hull_verts = np.ascontiguousarray(hull_verts, dtype=np.float32)
    hull_tris  = np.ascontiguousarray(hull_tris,  dtype=np.int32)

    parts = _gpu.decompose(
        verts.ctypes.data,      len(verts),
        tris.ctypes.data,       len(tris),
        hull_verts.ctypes.data, len(hull_verts),
        hull_tris.ctypes.data,  len(hull_tris),
        max_iters, cuts_per_axis, threshold, max_keep,
        verbose, debug,
    )
    return parts


def _decompose_shape(verts, tris, label, **kwargs):
    """Normalize, decompose, denormalize. Returns list of trimesh.Trimesh."""
    verts_n, center, scale = _normalize(verts)
    hull_verts, hull_tris  = _scipy_hull(verts_n)

    raw = _call_decompose(verts_n, tris, hull_verts, hull_tris, **kwargs)

    print(f"\n{label}: {len(raw)} parts")
    print(f"  {'':>6s}  {'gpu_mesh':>10s} {'tm_mesh':>10s} {'gpu_hull':>10s} {'tm_hull':>10s} {'scipy_hull':>10s}")
    meshes = []
    for i, (vb, tb, nv, nt, mv, hv) in enumerate(raw):
        pv = np.frombuffer(vb, dtype=np.float32).reshape(nv, 3)
        pt = np.frombuffer(tb, dtype=np.int32).reshape(nt, 3)
        pv_world = _denormalize(pv, center, scale)
        # Compare volumes in normalized space
        tm = trimesh.Trimesh(pv.copy(), pt.copy(), process=False)
        tm_mesh_vol = abs(tm.volume) if tm.is_volume else float('nan')
        # Convex hull of the part mesh via scipy
        try:
            sc_hull = ConvexHull(pv)
            scipy_hull_vol = sc_hull.volume
        except Exception:
            scipy_hull_vol = float('nan')
        # trimesh convex hull volume (uses its own hull)
        try:
            tm_hull_vol = abs(tm.convex_hull.volume)
        except Exception:
            tm_hull_vol = float('nan')
        print(f"  part {i:2d}: {mv:10.6f} {tm_mesh_vol:10.6f} {hv:10.6f} {tm_hull_vol:10.6f} {scipy_hull_vol:10.6f}")
        meshes.append(trimesh.Trimesh(pv_world.copy(), pt.copy()))
    print("Memory Usage:", '%.1f MB' % (_gpu.pool_usage() / 1e6))
    return meshes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cube_decompose():
    verts, tris = _make_cube()
    parts = _decompose_shape(verts, tris, "cube", verbose=1)
    assert len(parts) == 1, f"Cube is convex, expected 1 part, got {len(parts)}"


def test_lshape_decompose():
    verts, tris = _make_lshape()
    parts = _decompose_shape(verts, tris, "lshape", verbose=1)
    assert 2 <= len(parts) <= 4, f"Expected 2-4 parts, got {len(parts)}"
    for p in parts:
        assert p.vertices.shape[1] == 3
        assert p.faces.shape[1] == 3


@pytest.mark.skipif(not os.path.exists(OCTOCAT_OBJ),
                    reason="Octocat-v2.obj not found")
def test_octocat_decompose():
    mesh = trimesh.load(OCTOCAT_OBJ, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    tris  = np.array(mesh.faces,    dtype=np.int32)
    parts = _decompose_shape(verts, tris, "octocat",
                             max_iters=100, cuts_per_axis=10,
                             threshold=0.05, max_keep=32,
                             verbose=1)
    assert len(parts) >= 1


@pytest.mark.skipif(not os.path.exists(OCTOCAT_OBJ),
                    reason="Octocat-v2.obj not found")
def test_octocat_decompose_debug_steps():
    mesh = trimesh.load(OCTOCAT_OBJ, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    tris  = np.array(mesh.faces,    dtype=np.int32)
    parts = _decompose_shape(verts, tris, "octocat",
                             max_iters=100, cuts_per_axis=10,
                             threshold=0.05, max_keep=32,
                             verbose=1, debug=1)
    assert len(parts) >= 1


@pytest.mark.skipif(not os.path.exists(STL_49160),
                    reason="49160.stl not found")
def test_49160_decompose():
    mesh = trimesh.load(STL_49160, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    tris  = np.array(mesh.faces,    dtype=np.int32)
    parts = _decompose_shape(verts, tris, "49160",
                             max_iters=100, cuts_per_axis=10,
                             threshold=0.05, max_keep=32,
                             verbose=1)
    assert len(parts) >= 1
