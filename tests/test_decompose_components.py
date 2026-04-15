"""Tests for GPU connected components decomposition.

Validates:
- Single-component meshes return 1 component unchanged.
- Disjoint meshes are correctly split into separate components.
- Volume conservation (sum of component volumes = original volume).
- Vertex/triangle counts are correct per component.
"""
import numpy as np
import pytest

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

def _make_cube(center=(0, 0, 0), size=1.0):
    """Create a watertight cube mesh (8 verts, 12 tris)."""
    cx, cy, cz = center
    h = size / 2
    verts = np.array([
        [cx-h, cy-h, cz-h], [cx+h, cy-h, cz-h],
        [cx+h, cy+h, cz-h], [cx-h, cy+h, cz-h],
        [cx-h, cy-h, cz+h], [cx+h, cy-h, cz+h],
        [cx+h, cy+h, cz+h], [cx-h, cy+h, cz+h],
    ], dtype=np.float32)
    tris = np.array([
        [0,2,1], [0,3,2],  # -Z
        [4,5,6], [4,6,7],  # +Z
        [0,1,5], [0,5,4],  # -Y
        [2,3,7], [2,7,6],  # +Y
        [0,4,7], [0,7,3],  # -X
        [1,2,6], [1,6,5],  # +X
    ], dtype=np.int32)
    return verts, tris


def _make_tetrahedron(center=(0, 0, 0), size=1.0):
    """Create a tetrahedron (4 verts, 4 tris)."""
    cx, cy, cz = center
    s = size
    verts = np.array([
        [cx + s, cy + s, cz + s],
        [cx + s, cy - s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
    ], dtype=np.float32)
    tris = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int32)
    return verts, tris


def _merge_meshes(meshes):
    """Merge a list of (verts, tris) into a single combined mesh."""
    all_verts = []
    all_tris = []
    offset = 0
    for v, t in meshes:
        all_verts.append(v)
        all_tris.append(t + offset)
        offset += len(v)
    return (np.concatenate(all_verts).astype(np.float32),
            np.concatenate(all_tris).astype(np.int32))


def _decompose(verts, tris):
    """Call GPU decompose_components; return list of (verts, tris)."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int32)
    nv, nt = len(verts), len(tris)

    max_comp = 32
    max_vp = nv + 16
    max_tp = nt + 16

    out_verts = np.zeros((max_comp * max_vp, 3), dtype=np.float32)
    out_tris = np.zeros((max_comp * max_tp, 3), dtype=np.int32)
    out_nv = np.zeros(max_comp, dtype=np.int32)
    out_nt = np.zeros(max_comp, dtype=np.int32)

    n_comp = _gpu.test_postprocess_dc(
        verts.ctypes.data, nv,
        tris.ctypes.data, nt,
        max_comp, max_vp, max_tp,
        out_verts.ctypes.data,
        out_tris.ctypes.data,
        out_nv.ctypes.data,
        out_nt.ctypes.data)

    components = []
    for c in range(n_comp):
        cnv = out_nv[c]
        cnt = out_nt[c]
        cv = out_verts[c * max_vp: c * max_vp + cnv].copy()
        ct = out_tris[c * max_tp: c * max_tp + cnt].copy()
        components.append((cv, ct))
    return components


def _signed_volume(verts, tris):
    """Signed volume via divergence theorem."""
    if len(tris) == 0:
        return 0.0
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return float(np.sum(v0 * np.cross(v1, v2))) / 6.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_cube():
    """A single connected cube should produce 1 component."""
    verts, tris = _make_cube()
    comps = _decompose(verts, tris)
    assert len(comps) == 1
    cv, ct = comps[0]
    assert cv.shape[0] == 8
    assert ct.shape[0] == 12


def test_two_disjoint_cubes():
    """Two disjoint cubes should produce 2 components."""
    cube1 = _make_cube(center=(0, 0, 0), size=1.0)
    cube2 = _make_cube(center=(5, 0, 0), size=1.0)
    verts, tris = _merge_meshes([cube1, cube2])

    comps = _decompose(verts, tris)
    assert len(comps) == 2

    # Each component should have 8 verts and 12 tris
    nvs = sorted([c[0].shape[0] for c in comps])
    nts = sorted([c[1].shape[0] for c in comps])
    assert nvs == [8, 8]
    assert nts == [12, 12]


def test_three_disjoint_tetrahedra():
    """Three disjoint tetrahedra should produce 3 components."""
    t1 = _make_tetrahedron(center=(0, 0, 0))
    t2 = _make_tetrahedron(center=(10, 0, 0))
    t3 = _make_tetrahedron(center=(0, 10, 0))
    verts, tris = _merge_meshes([t1, t2, t3])

    comps = _decompose(verts, tris)
    assert len(comps) == 3

    nvs = sorted([c[0].shape[0] for c in comps])
    nts = sorted([c[1].shape[0] for c in comps])
    assert nvs == [4, 4, 4]
    assert nts == [4, 4, 4]


def test_connected_cubes_shared_vertex():
    """Two cubes sharing vertices should remain 1 component."""
    # Create two adjacent cubes that share a face (4 vertices)
    verts = np.array([
        # First cube (0-7)
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        # Second cube extends from x=1 to x=2, sharing verts 1,2,5,6
        [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
    ], dtype=np.float32)
    tris = np.array([
        # First cube
        [0,2,1], [0,3,2], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [0,4,7], [0,7,3], [1,2,6], [1,6,5],
        # Second cube (shares verts 1,2,5,6)
        [1,9,8], [1,2,9], [5,10,11], [5,11,6],
        [1,8,10], [1,10,5], [9,2,6], [9,6,11],
        [1,5,6], [1,6,2],  # shared face (interior, but still connected)
        [8,9,11], [8,11,10],
    ], dtype=np.int32)

    comps = _decompose(verts, tris)
    assert len(comps) == 1


def test_volume_conservation():
    """Sum of component volumes should equal original volume."""
    cube1 = _make_cube(center=(0, 0, 0), size=2.0)
    cube2 = _make_cube(center=(10, 0, 0), size=3.0)
    verts, tris = _merge_meshes([cube1, cube2])

    original_vol = abs(_signed_volume(verts, tris))
    comps = _decompose(verts, tris)

    comp_vol_sum = sum(abs(_signed_volume(cv, ct)) for cv, ct in comps)
    assert abs(comp_vol_sum - original_vol) < 1e-3, \
        f"Volume mismatch: {comp_vol_sum} vs {original_vol}"


def test_33_disjoint_cubes_clamp_to_32():
    """33 disjoint unit cubes exceed DC_MAX_OUT=32.

    The clamp merges the last two components into one output slot,
    producing 32 output parts: 31 with volume=1 and 1 with volume=2.
    All must be watertight with positive volume.
    """
    cubes = [_make_cube(center=(i * 3, 0, 0), size=1.0) for i in range(33)]
    verts, tris = _merge_meshes(cubes)

    comps = _decompose(verts, tris)
    assert len(comps) == 32, \
        f"Expected 32 components, got {len(comps)}"

    vols = sorted([abs(_signed_volume(cv, ct)) for cv, ct in comps])
    # 31 components with vol≈1, 1 component with vol≈2
    vol1_count = sum(1 for v in vols if abs(v - 1.0) < 1e-3)
    vol2_count = sum(1 for v in vols if abs(v - 2.0) < 1e-3)
    assert vol1_count == 31, \
        f"Expected 31 components with volume 1.0, got {vol1_count}; vols={vols}"
    assert vol2_count == 1, \
        f"Expected 1 component with volume 2.0, got {vol2_count}; vols={vols}"

    # All components must be watertight (positive signed volume)
    for i, (cv, ct) in enumerate(comps):
        sv = _signed_volume(cv, ct)
        assert sv > 0, f"Component {i} has non-positive signed volume {sv}"
