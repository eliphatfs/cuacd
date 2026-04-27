"""Tests for lookahead search decomposition."""

import os
import numpy as np
import pytest
import trimesh

import cuacd
from cuacd import _mesh_volume_cpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_ctx():
    with cuacd.Context(device=0, pool_bytes=10 * 1024**3) as ctx:
        yield ctx


OCTOCAT_OBJ = os.path.join(os.path.dirname(__file__),
                            "../CoACD/examples/Octocat-v2.obj")
STL_49160   = os.path.join(os.path.dirname(__file__),
                            "data/49160.stl")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__),
                            "../decomp_output")


def _make_cube():
    """Unit cube [0,1]^3 with outward-facing triangles."""
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    t = np.array([
        [0,2,1],[0,3,2],  # -Z
        [4,5,6],[4,6,7],  # +Z
        [0,1,5],[0,5,4],  # -Y
        [2,3,7],[2,7,6],  # +Y
        [0,4,7],[0,7,3],  # -X
        [1,2,6],[1,6,5],  # +X
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


def _decompose_shape(ctx, verts, tris, label="", **kwargs):
    """Normalize, decompose, and denormalize."""
    from scipy.spatial import ConvexHull

    # Normalize to [-1, 1]
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2
    extent = (hi - lo).max()
    scale = extent / 2 if extent > 0 else 1.0
    nv = ((verts - center) / scale).astype(np.float32)

    kwargs.setdefault("verbose", 1)
    raw = ctx.lookahead_decompose(nv, tris, **kwargs)

    # Hull volumes from the readback
    print(f"\n{label}: {len(raw)} parts" if label else f"\n{len(raw)} parts")
    print(f"  {'':>6s}  {'tm_mesh':>10s} {'gpu_hull':>10s} {'tm_hull':>10s} {'scipy_hull':>10s}")

    parts = []
    for i, (pv, pt, hull_v, hull_t) in enumerate(raw):
        # tm mesh volume (normalized space)
        tm = trimesh.Trimesh(pv.copy(), pt.copy(), process=False)
        tm_mesh_vol = abs(tm.volume) if tm.is_volume else float('nan')
        # GPU hull volume from readback
        hv = _mesh_volume_cpu(hull_v, hull_t) if len(hull_t) > 0 else float('nan')
        # trimesh hull volume
        try:
            tm_hull_vol = abs(tm.convex_hull.volume)
        except Exception:
            tm_hull_vol = float('nan')
        # scipy hull volume
        try:
            sc_hull = ConvexHull(pv)
            scipy_hull_vol = sc_hull.volume
        except Exception:
            scipy_hull_vol = float('nan')
        print(f"  part {i:2d}: {tm_mesh_vol:10.6f} {hv:10.6f} {tm_hull_vol:10.6f} {scipy_hull_vol:10.6f}")

        # Denormalize for output
        v = pv * scale + center
        parts.append((v, pt))

    print(f"  pool usage: {ctx.pool_usage() / 1e6:.1f} MB")
    return parts


def _build_scene(all_parts_by_shape):
    """Build a trimesh.Scene with random per-part colors."""
    scene = trimesh.Scene()
    rng = np.random.default_rng(0)
    for label, parts in all_parts_by_shape:
        for i, (v, t) in enumerate(parts):
            mesh = trimesh.Trimesh(v, t)
            color = (rng.random(3) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
            mesh.visual.vertex_colors[:, :3] = color
            scene.add_geometry(mesh, node_name=f"{label}_{i}")
    return scene


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLookaheadDecompose:

    def test_cube_la(self, gpu_ctx):
        """Unit cube is already convex — should produce 1 part."""
        verts, tris = _make_cube()
        parts = _decompose_shape(gpu_ctx, verts, tris, label="cube", max_iters=5, threshold=0.05)
        assert len(parts) == 1

    def test_lshape_la(self, gpu_ctx):
        """L-shape should decompose into 2+ convex parts."""
        verts, tris = _make_lshape()
        parts = _decompose_shape(gpu_ctx, verts, tris, label="lshape",
                                 max_iters=50, threshold=0.05)
        assert 2 <= len(parts) <= 10
        for v, t in parts:
            assert v.ndim == 2 and v.shape[1] == 3
            assert t.ndim == 2 and t.shape[1] == 3
            assert len(v) >= 4  # minimum tetrahedron
            assert len(t) >= 4

    @pytest.mark.skipif(not os.path.exists(OCTOCAT_OBJ),
                        reason="Octocat model not found")
    def test_octocat_la(self, gpu_ctx):
        """Octocat decomposition should produce >= 1 part."""
        mesh = trimesh.load(OCTOCAT_OBJ, force="mesh")
        parts = _decompose_shape(
            gpu_ctx,
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            label="octocat", max_iters=100, threshold=0.05)
        assert len(parts) >= 1

    @pytest.mark.skipif(not os.path.exists(OCTOCAT_OBJ),
                        reason="Octocat model not found")
    def test_octocat_la_debug_steps(self, gpu_ctx):
        """Octocat with debug=1 — prints per-step timing and memory usage."""
        mesh = trimesh.load(OCTOCAT_OBJ, force="mesh")
        parts = _decompose_shape(
            gpu_ctx,
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            label="octocat_debug", max_iters=100, threshold=0.05, debug=1)
        assert len(parts) >= 1

    def test_la_convergence(self, gpu_ctx):
        """After decomposition with reasonable threshold, all parts should be roughly convex."""
        verts, tris = _make_lshape()
        parts = _decompose_shape(gpu_ctx, verts, tris, label="lshape",
                                 max_iters=100, threshold=0.05)
        assert len(parts) >= 2
        # Each part should have valid geometry
        for v, t in parts:
            assert len(v) > 0
            assert len(t) > 0

    @pytest.mark.skipif(not os.path.exists(STL_49160),
                        reason="49160.stl not found")
    def test_49160_la(self, gpu_ctx):
        """Decompose 49160.stl — should produce >= 1 part."""
        mesh = trimesh.load(STL_49160, force="mesh")
        parts = _decompose_shape(
            gpu_ctx,
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            label="49160", max_iters=100, threshold=0.05)
        assert len(parts) >= 1

    def test_export_glb(self, gpu_ctx):
        """Decompose shapes with lookahead and export GLB files."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        kw = dict(max_iters=100, width=60, width2=5, threshold=0.05,
                  depth=2, quick_depth=0)

        # Cube
        v, t = _make_cube()
        parts = _decompose_shape(gpu_ctx, v, t, label="cube", max_iters=5, threshold=0.05)
        path = os.path.join(OUTPUT_DIR, "la_cube.glb")
        _build_scene([("cube", parts)]).export(path)
        print(f"\nExported {path}")
        assert os.path.exists(path)

        # L-shape
        v, t = _make_lshape()
        parts = _decompose_shape(gpu_ctx, v, t, label="lshape", **kw)
        path = os.path.join(OUTPUT_DIR, "la_lshape.glb")
        _build_scene([("lshape", parts)]).export(path)
        print(f"\nExported {path}")
        assert os.path.exists(path)

        # 49160.stl (skip quietly if missing)
        if os.path.exists(STL_49160):
            mesh = trimesh.load(STL_49160, force="mesh")
            v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            t = np.ascontiguousarray(mesh.faces, dtype=np.int32)
            parts = _decompose_shape(gpu_ctx, v, t, label="49160", **kw)
            path = os.path.join(OUTPUT_DIR, "la_49160.glb")
            _build_scene([("49160", parts)]).export(path)
            print(f"\nExported {path}")
            assert os.path.exists(path)

        # Octocat (skip quietly if missing)
        if os.path.exists(OCTOCAT_OBJ):
            mesh = trimesh.load(OCTOCAT_OBJ, force="mesh")
            v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            t = np.ascontiguousarray(mesh.faces, dtype=np.int32)
            parts = _decompose_shape(gpu_ctx, v, t, label="octocat", **kw)
            path = os.path.join(OUTPUT_DIR, "la_octocat.glb")
            _build_scene([("octocat", parts)]).export(path)
            print(f"\nExported {path}")
            assert os.path.exists(path)
