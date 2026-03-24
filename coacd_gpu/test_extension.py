"""
Smoke test for coacd_gpu.

    pip install -e .
    python test_extension.py
"""
import numpy as np
import coacd_gpu


def make_triangle():
    """Single triangle at z=0: (0,0,0), (1,0,0), (0,1,0)."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, tris


def test_point_mesh_distances(ctx):
    verts, tris = make_triangle()
    points = np.array([
        [0.25, 0.25, 0.0],   # on triangle -> 0
        [0.25, 0.25, 1.0],   # 1 unit above -> 1
        [0.0,  0.0,  0.0],   # on vertex -> 0
    ], dtype=np.float32)

    dists = ctx.point_mesh_distances(points, verts, tris)
    np.testing.assert_allclose(dists[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(dists[1], 1.0, atol=1e-5)
    np.testing.assert_allclose(dists[2], 0.0, atol=1e-5)
    print("PASS: point_mesh_distances")


def test_hausdorff(ctx):
    va, ta = make_triangle()
    vb = va.copy(); vb[:, 2] += 0.5
    tb = ta.copy()
    sa = np.array([[0.25, 0.25, 0.0]], dtype=np.float32)
    sb = np.array([[0.25, 0.25, 0.5]], dtype=np.float32)

    h = ctx.hausdorff(sa, va, ta, sb, vb, tb)
    np.testing.assert_allclose(h, 0.5, atol=1e-5)
    print("PASS: hausdorff")


def test_pairwise_hausdorff(ctx):
    v0 = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
    v1 = np.array([[0,0,1],[1,0,1],[0,1,1]], dtype=np.float32)
    t0 = np.array([[0,1,2]], dtype=np.int32)
    t1 = np.array([[0,1,2]], dtype=np.int32)
    s0 = np.array([[0.25, 0.25, 0.0]], dtype=np.float32)
    s1 = np.array([[0.25, 0.25, 1.0]], dtype=np.float32)

    cost = ctx.pairwise_hausdorff(
        all_samples=np.vstack([s0, s1]),
        sample_offsets=np.array([0, 1, 2], dtype=np.int32),
        all_vertices=np.vstack([v0, v1]),
        all_triangles=np.vstack([t0, t1]),
        tri_offsets=np.array([0, 1, 2], dtype=np.int32),
        vert_offsets=np.array([0, 3, 6], dtype=np.int32),
    )
    np.testing.assert_allclose(cost[1, 0], 1.0, atol=1e-4)
    print("PASS: pairwise_hausdorff")


if __name__ == "__main__":
    with coacd_gpu.Context(device=0) as ctx:
        test_point_mesh_distances(ctx)
        test_hausdorff(ctx)
        test_pairwise_hausdorff(ctx)
    print("\nAll tests passed!")
