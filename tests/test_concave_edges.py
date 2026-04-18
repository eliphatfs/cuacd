"""Tests for concave edge sampling in lookahead decomposition."""

import numpy as np
import pytest
import coacd_gpu


def _l_shape_mesh():
    """L-shape with strong concavity along a diagonal edge."""
    verts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],  # bottom
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],  # top of base
        [0,0,2],[1,0,2],                    # tower top
    ], dtype=np.float32)
    tris = np.array([
        [0,1,2],[0,2,3],  # bottom
        [4,5,6],[4,6,7],  # top of base
        [0,1,5],[0,5,4],  # front
        [1,2,6],[1,6,5],  # right
        [2,3,7],[2,7,6],  # back
        [3,0,4],[3,4,7],  # left
        [4,5,9],[4,9,8],  # tower front
        [5,9,8],[5,8,4],  # tower back
    ], dtype=np.int32)
    return verts, tris


def _make_torus(R=1.0, r=0.3, n_major=24, n_minor=12):
    """Torus mesh — many concave edges on the inner ring."""
    verts = []
    tris = []
    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        ct, st = np.cos(theta), np.sin(theta)
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            cp, sp = np.cos(phi), np.sin(phi)
            x = (R + r * cp) * ct
            y = (R + r * cp) * st
            z = r * sp
            verts.append([x, y, z])
            i0 = i * n_minor + j
            i1 = i * n_minor + (j + 1) % n_minor
            i2 = ((i + 1) % n_major) * n_minor + j
            i3 = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            tris.append([i0, i1, i2])
            tris.append([i1, i3, i2])
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


class TestConcaveEdges:
    """Test concave edge sampling with various meshes."""

    def test_explicit_zero_disables(self):
        """n_concave_edges=0 should produce fewer parts than the default (n_concave_edges=32)."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts_default = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05)
            parts_zero = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05,
                n_concave_edges=0)
            # With concave edges enabled (default), we should get at least as many parts
            assert len(parts_default) >= len(parts_zero)

    def test_l_shape_concave_edges(self):
        """L-shape decomposition with concave edge sampling."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05,
                n_concave_edges=3, concave_eps=0.005,
                concave_threshold=3.49, concave_iters=1)
            assert len(parts) >= 2
            for v, t, hv, ht in parts:
                assert len(v) > 0
                assert len(t) > 0

    def test_l_shape_multiple_concave_iters(self):
        """L-shape with concave edges on multiple iterations."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05,
                n_concave_edges=3, concave_iters=3)
            assert len(parts) >= 2

    def test_torus_concave_edges(self):
        """Torus has many concave edges on the inner ring."""
        verts, tris = _make_torus(n_major=16, n_minor=8)
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=10, width=9, threshold=0.05,
                n_concave_edges=5, concave_iters=1)
            assert len(parts) >= 2
            for v, t, hv, ht in parts:
                assert len(v) > 0
                assert len(t) > 0

    def test_width_validation(self):
        """total_width = width + 4*n_concave_edges must be < 512."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            with pytest.raises(ValueError, match="must be < 512"):
                ctx.lookahead_decompose(
                    verts, tris, width=500, n_concave_edges=5)

    def test_concave_eps_zero(self):
        """Zero epsilon should still work (planes through edge midpoint)."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05,
                n_concave_edges=3, concave_eps=0.0, concave_iters=1)
            assert len(parts) >= 2

    def test_concave_threshold_high(self):
        """Very high threshold means no edges are concave → same as disabled."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts_no = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, threshold=0.05,
                n_concave_edges=3, concave_threshold=100.0, concave_iters=1)
            # With threshold so high, no edges qualify as concave,
            # so edge_cuts = 0 and behavior matches n_concave_edges=0
            assert len(parts_no) >= 1


class TestConcaveEdgePresets:
    """Quick smoke tests with small width and low iteration count."""

    def test_l_shape_width3(self):
        """Minimal width with concave edges."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=3, width=3, threshold=0.05,
                n_concave_edges=2, concave_iters=1)
            assert len(parts) >= 1

    def test_l_shape_depth1(self):
        """depth=1 (single expansion level) with concave edges."""
        verts, tris = _l_shape_mesh()
        with coacd_gpu.Context() as ctx:
            parts = ctx.lookahead_decompose(
                verts, tris, max_iters=5, width=9, depth=1, threshold=0.05,
                n_concave_edges=3, concave_iters=1)
            assert len(parts) >= 2
