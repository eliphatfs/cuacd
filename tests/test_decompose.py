"""Test beam_decompose on an L-shaped mesh.

The L-shape is built from two axis-aligned boxes joined at a corner:
  - Box A: [0,2] x [0,1] x [0,1]   (horizontal bar)
  - Box B: [0,1] x [1,3] x [0,1]   (vertical bar)
Their union is non-convex.  With threshold=0.05 the decomposition should
produce 2-4 convex pieces.
"""
import numpy as np
import pytest
from scipy.spatial import ConvexHull

import coacd_gpu._gpu as _gpu


@pytest.fixture(autouse=True, scope="module")
def gpu_ctx():
    _gpu.init(0)
    yield
    _gpu.destroy()


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _box(lo, hi):
    """Return (verts float32[N,3], tris int32[M,3]) for an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tris = np.array([
        [0,2,1],[0,3,2],  # -Z
        [4,5,6],[4,6,7],  # +Z
        [0,1,5],[0,5,4],  # -Y
        [2,3,7],[2,7,6],  # +Y
        [0,4,7],[0,7,3],  # -X
        [1,2,6],[1,6,5],  # +X
    ], dtype=np.int32)
    return verts, tris


def _merge_meshes(meshes):
    """Concatenate a list of (verts, tris) into one mesh (no deduplication)."""
    all_v, all_t = [], []
    offset = 0
    for v, t in meshes:
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    return np.concatenate(all_v), np.concatenate(all_t)


def _make_lshape():
    """L-shape = box_A ∪ box_B (two boxes sharing a face-edge)."""
    box_a = _box([0, 0, 0], [2, 1, 1])  # horizontal bar
    box_b = _box([0, 1, 0], [1, 3, 1])  # vertical bar
    return _merge_meshes([box_a, box_b])


def _scipy_hull(verts):
    """Compute convex hull via scipy; return (hull_verts float32, hull_tris int32)."""
    ch = ConvexHull(verts)
    hull_verts = verts[ch.vertices].astype(np.float32)
    # Remap simplices to hull_verts indices
    remap = {old: new for new, old in enumerate(ch.vertices)}
    hull_tris = np.array([[remap[i] for i in tri] for tri in ch.simplices],
                         dtype=np.int32)
    return hull_verts, hull_tris


def _call_decompose(verts, tris, hull_verts, hull_tris,
                    max_iters=100, cuts_per_axis=10,
                    threshold=0.05, max_keep=32, verbose=1):
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
        verbose,
    )
    return parts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lshape_decompose():
    verts, tris = _make_lshape()
    hull_verts, hull_tris = _scipy_hull(verts)

    parts = _call_decompose(verts, tris, hull_verts, hull_tris)

    print(f"\nL-shape decomposition: {len(parts)} parts")
    for i, (vb, tb, nv, nt, mv, hv) in enumerate(parts):
        print(f"  part {i}: nv={nv} nt={nt} mesh_vol={mv:.4f} hull_vol={hv:.4f}")

    assert 2 <= len(parts) <= 4, f"Expected 2-4 parts, got {len(parts)}"

    # Each part should have valid geometry
    for vb, tb, nv, nt, mv, hv in parts:
        assert nv >= 4
        assert nt >= 4
        part_verts = np.frombuffer(vb, dtype=np.float32).reshape(nv, 3)
        part_tris  = np.frombuffer(tb, dtype=np.int32).reshape(nt, 3)
        assert part_verts.shape == (nv, 3)
        assert part_tris.shape  == (nt, 3)
        assert part_tris.min() >= 0
        assert part_tris.max() < nv
