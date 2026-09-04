"""
Tests for GPU mesh preprocess (PaMO stage-1 port: cumesh2sdf SDF + PDMC
Dual Marching Cubes).

Sections:
1. Topology healing: non-watertight / non-manifold / misoriented inputs
2. Geometry fidelity: volume/extent inflation matches upstream semantics
3. Robustness: degenerate inputs, resolution ladder

Run as pytest:
    pytest tests/test_preprocess.py -v
"""

import numpy as np
import pytest

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

try:
    import cuacd
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="cuacd extension not built")


@pytest.fixture(scope="module")
def ctx():
    c = cuacd.Context(device=0, pool_bytes=0)
    yield c
    c.close()


def _cube(scale=1.0, offset=(0.0, 0.0, 0.0)):
    """Unit cube scaled/offset, consistently outward-oriented."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32) * scale + np.asarray(offset, np.float32)
    t = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return v, t


# ---------------------------------------------------------------------------
# 1. Topology healing
# ---------------------------------------------------------------------------

class TestTopologyHealing:
    def test_open_cube_healed(self, ctx):
        """Cube with a removed face (open edge) remeshes to watertight manifold."""
        v, t = _cube()
        open_t = t[:10]  # drop the two "+x" faces
        audit = ctx.check_mesh(v, open_t)
        assert not audit["watertight"]

        vv, tt = ctx.preprocess(v, open_t, resolution=32)
        out = ctx.check_mesh(vv, tt)
        assert out["watertight"] and out["manifold"] and out["oriented"]
        assert out["flags"] == 0

    def test_reversed_winding_healed(self, ctx):
        """Globally inverted winding remeshes to the same geometry as correct input."""
        v, t = _cube()
        vv_a, tt_a = ctx.preprocess(v, t, resolution=32)
        vv_b, tt_b = ctx.preprocess(v, t[:, ::-1].copy(), resolution=32)
        # Orientation-robust sign fill: both give the same shape (volume match).
        vol_a = _volume(vv_a, tt_a)
        vol_b = _volume(vv_b, tt_b)
        assert abs(vol_a - vol_b) < 0.05 * vol_a

    @pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
    def test_nonmanifold_duplicate_slab_healed(self, ctx):
        """Two coincident slabs with duplicated vertices heal into one solid."""
        v, t = _cube()
        dup_v = np.vstack([v, v + np.float32([0, 0, 1e-4])])
        dup_t = np.vstack([t, t[:, ::-1] + len(v)])  # opposite winding pair

        vv, tt = ctx.preprocess(dup_v, dup_t, resolution=32)
        out = ctx.check_mesh(vv, tt)
        assert out["watertight"] and out["manifold"] and out["flags"] == 0

    def test_degenerate_flat_triangle(self, ctx):
        """A single zero-volume triangle remeshes without error into a thin pillow."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        t = np.array([[0, 1, 2]], dtype=np.int32)
        vv, tt = ctx.preprocess(v, t, resolution=32)
        out = ctx.check_mesh(vv, tt)
        assert out["watertight"] and out["manifold"] and out["flags"] == 0
        assert len(vv) > 0


# ---------------------------------------------------------------------------
# 2. Geometry fidelity (upstream-faithful semantics)
# ---------------------------------------------------------------------------

class TestGeometryFidelity:
    @pytest.mark.parametrize("R", [32, 64, 128])
    def test_cube_extent_inflation_matches_shift(self, ctx, R):
        """Upstream extracts at iso +0.9/R: the surface dilates by
        ~2*(0.9/R)*margin*extent per axis (pamo/CDF semantics)."""
        v, t = _cube()
        vv, tt = ctx.preprocess(v, t, resolution=R)
        extent = float(vv[:, 0].max() - vv[:, 0].min())
        margin = 1.0 + 2.0 * 3.0 / R
        expected = 1.0 + 2.0 * (0.9 / R) * margin
        assert extent == pytest.approx(expected, rel=0.15)

    def test_resolution_rejection(self, ctx):
        v, t = _cube()
        with pytest.raises(ValueError):
            ctx.preprocess(v, t, resolution=48)  # not a power of two
        with pytest.raises(ValueError):
            ctx.preprocess(v, t, resolution=4)   # below minimum

    def test_translation_scale_invariance(self, ctx):
        """Output volume scales cubically with input scale, is translation-invariant."""
        v, t = _cube(scale=5.0, offset=(7.0, -3.0, 2.0))
        vv, tt = ctx.preprocess(v, t, resolution=32)
        vol_big = _volume(vv, tt)
        v2, t2 = _cube()
        vv2, tt2 = ctx.preprocess(v2, t2, resolution=32)
        vol_small = _volume(vv2, tt2)
        assert vol_big / vol_small == pytest.approx(125.0, rel=0.05)


# ---------------------------------------------------------------------------
# 3. Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_high_resolution_icosphere(self, ctx):
        """Dense curved surface at R=128 stays watertight and near unit volume."""
        try:
            m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
            v = np.ascontiguousarray(m.vertices, np.float32)
            t = np.ascontiguousarray(m.faces, np.int32)
        except Exception:
            pytest.skip("trimesh not available")
        vv, tt = ctx.preprocess(v, t, resolution=128)
        out = ctx.check_mesh(vv, tt)
        assert out["watertight"] and out["manifold"] and out["flags"] == 0
        vol = _volume(vv, tt)
        assert vol == pytest.approx(4.19, rel=0.15)  # sphere vol + dilation


def _volume(v, t):
    if _HAS_TRIMESH:
        return float(trimesh.Trimesh(v, t, process=False).volume)
    # Divergence-theorem fallback
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
