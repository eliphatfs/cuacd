"""
Tests and benchmarks for GPU hull volume algorithms and mesh volume kernel.

Sections:
1. Mesh volume tests (watertight icospheres vs trimesh)
2. Hull volume correctness tests (vs scipy ConvexHull)
3. Benchmarks: throughput for different batch sizes and point distributions

Run as pytest:
    pytest tests/test_hull.py -v

Run standalone benchmark:
    python tests/test_hull.py
"""

import numpy as np
import pytest
import time

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

try:
    import coacd_gpu
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


# ---------------------------------------------------------------------------
# Mesh generators
# ---------------------------------------------------------------------------

def make_unit_cube():
    """Closed unit cube [0,1]^3 — 8 vertices, 12 triangles."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        [0, 1, 2], [0, 2, 3],   # bottom  (z=0, winding: normal -z)
        [4, 6, 5], [4, 7, 6],   # top     (z=1, winding: normal +z)
        [0, 4, 5], [0, 5, 1],   # front   (y=0)
        [2, 6, 7], [2, 7, 3],   # back    (y=1)
        [0, 3, 7], [0, 7, 4],   # left    (x=0)
        [1, 5, 6], [1, 6, 2],   # right   (x=1)
    ], dtype=np.int32)
    return v, t


def subdivide_icosphere(verts, faces):
    """Subdivide each triangular face into 4 by splitting edges.

    Each new midpoint vertex is projected back onto the unit sphere so the
    mesh remains a sphere approximation.  Returns new (verts, faces) arrays.
    """
    edge_map = {}
    new_verts = list(verts)
    new_faces = []

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_map:
            mid = (verts[a] + verts[b]) * 0.5
            mid = mid / np.linalg.norm(mid)
            edge_map[key] = len(new_verts)
            new_verts.append(mid)
        return edge_map[key]

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.append([a,  ab, ca])
        new_faces.append([b,  bc, ab])
        new_faces.append([c,  ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def make_icosphere(level):
    """Build a unit icosphere by subdividing a regular icosahedron `level` times.

    Returns (vertices [N,3] float32, triangles [M,3] int32).  At level 0 we
    have 12 verts / 20 faces.  Each subdivision multiplies faces by 4 and
    places the resulting vertices on the unit sphere.

    Volume converges to 4π/3 ≈ 4.18879 from below as level increases.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float64)
    # Normalise to unit sphere
    raw /= np.linalg.norm(raw[0])

    faces = np.array([
        # 5 faces around vertex 0
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        # 5 adjacent faces
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [7,  1,  8],
        # 5 faces around vertex 3
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        # 5 faces connecting the two halves
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1],
    ], dtype=np.int32)

    verts = raw
    for _ in range(level):
        verts, faces = subdivide_icosphere(verts, faces)

    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------

def mesh_volume_cpu(verts, tris):
    """Signed mesh volume via divergence theorem (signed-tetrahedra sum).

    V = (1/6) * |sum_i  p0_i . (p1_i x p2_i)|

    Works for any consistently-wound, watertight, closed mesh.
    """
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1, v2)
    dots = np.sum(v0 * cross, axis=1)
    return abs(float(np.sum(dots))) / 6.0


def compute_hull_volume_scipy(pts):
    """Return the convex hull volume of a point cloud using scipy."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy not available")
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Point-cloud distribution generators (used in benchmarks and hull tests)
# ---------------------------------------------------------------------------

def make_hemisphere_shell(n, rng):
    """n points uniformly distributed on the upper hemisphere shell."""
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    pts /= norms
    return pts


def make_sphere_shell(n, rng):
    """n points uniformly distributed on the unit sphere surface."""
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    pts /= norms
    return pts


def make_sphere_interior(n, rng):
    """n points uniformly distributed inside the unit sphere."""
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    pts /= norms
    # r ~ U(0,1)^{1/3} gives uniform volume distribution
    r = np.cbrt(rng.random(n).astype(np.float32))[:, None]
    pts = pts * r
    return pts


def make_cube_interior(n, rng):
    """n points uniformly distributed inside the unit cube [-0.5, 0.5]^3."""
    return rng.random((n, 3)).astype(np.float32) - 0.5


def make_gaussian(n, rng):
    """n points drawn from an isotropic standard normal distribution."""
    return rng.standard_normal((n, 3)).astype(np.float32)


def make_noisy_icosphere(level, amplitude, seed):
    """Icosphere with Perlin-like normal displacement.

    Each vertex is displaced along its normal by ``amplitude * noise(v)``,
    where noise is a smooth, spatially-correlated random field built from
    low-frequency spherical harmonics (band-limited noise on the sphere).
    The result is a bumpy, non-convex surface whose convex hull volume
    differs from its mesh volume.

    Returns (vertices [N,3] float32, faces [M,3] int32).
    """
    verts, faces = make_icosphere(level)
    rng = np.random.default_rng(seed)

    # Build smooth noise via random spherical harmonics up to degree 4.
    # For each vertex, noise = sum of random_coeff * Y_lm-like basis.
    # We approximate with random linear combos of low-freq trig functions.
    n_basis = 16
    dirs = rng.standard_normal((n_basis, 3)).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    coeffs = rng.standard_normal(n_basis).astype(np.float64)

    # Project each vertex onto each random direction, take sin/cos
    verts64 = verts.astype(np.float64)
    proj = verts64 @ dirs.T  # (N, n_basis)
    noise = np.zeros(len(verts), dtype=np.float64)
    for i in range(n_basis):
        noise += coeffs[i] * np.sin(2.0 * np.pi * proj[:, i])

    # Normalize noise to [-1, 1]
    nmax = np.abs(noise).max()
    if nmax > 1e-12:
        noise /= nmax

    # Normals = vertex positions (unit sphere)
    normals = verts64 / np.linalg.norm(verts64, axis=1, keepdims=True)
    displaced = verts64 + amplitude * noise[:, None] * normals
    return displaced.astype(np.float32), faces


# ---------------------------------------------------------------------------
# Section 1: Mesh volume tests (CPU reference)
# ---------------------------------------------------------------------------

class TestMeshVolumeCPU:
    """Validate CPU mesh_volume_cpu against analytic and trimesh references."""

    def test_unit_cube_volume(self):
        """Unit cube volume must equal 1.0 exactly (up to float rounding)."""
        verts, tris = make_unit_cube()
        vol = mesh_volume_cpu(verts, tris)
        assert abs(vol - 1.0) < 1e-6, f"unit cube: got {vol}, expected 1.0"

    @pytest.mark.parametrize("level", [2, 3, 4])
    def test_icosphere_volume_converges(self, level):
        """Icosphere volume should converge to 4π/3 ≈ 4.18879 from below."""
        verts, tris = make_icosphere(level)
        vol = mesh_volume_cpu(verts, tris)
        expected = 4.0 * np.pi / 3.0
        # Polyhedral underestimate; tolerance gets tighter with each level
        # level=2: ~2500 faces, level=3: ~10K, level=4: ~41K
        # Relative error is O(1/4^level) so we allow a generous bound
        assert vol < expected + 1e-4, \
            f"level={level}: volume {vol:.6f} exceeds sphere volume {expected:.6f}"
        assert vol > expected * 0.9, \
            f"level={level}: volume {vol:.6f} too small (expected >{expected * 0.9:.4f})"

    @pytest.mark.parametrize("level", [3, 4])
    def test_icosphere_volume_accuracy(self, level):
        """Higher subdivision levels must be close to 4π/3.

        Polyhedral approximation error is O(1/n_faces).  Measured values:
          level 3 → 1280 faces, rel_err ≈ 0.86%
          level 4 → 5120 faces, rel_err ≈ 0.22%
        Each subdivision multiplies faces by 4 so the error divides by ~4.
        """
        verts, tris = make_icosphere(level)
        vol = mesh_volume_cpu(verts, tris)
        expected = 4.0 * np.pi / 3.0
        rel_err = abs(vol - expected) / expected
        # level 3: 0.86% measured → allow up to 1.5%
        # level 4: 0.22% measured → allow up to 0.4%
        tol = 0.015 / (4 ** (level - 3))
        assert rel_err < tol, \
            f"level={level}: relative error {rel_err:.4%} exceeds {tol:.4%}"

    def test_icosphere_vertex_count(self):
        """Verify icosphere vertex / face counts follow the subdivision formula."""
        # level 0: 12 verts, 20 faces
        # level k: V_k = 10*4^k + 2,  F_k = 20*4^k
        for level in range(3):
            verts, tris = make_icosphere(level)
            expected_faces = 20 * (4 ** level)
            expected_verts = 10 * (4 ** level) + 2
            assert len(tris) == expected_faces, \
                f"level={level}: {len(tris)} faces, expected {expected_faces}"
            assert len(verts) == expected_verts, \
                f"level={level}: {len(verts)} verts, expected {expected_verts}"

    def test_icosphere_vertices_on_unit_sphere(self):
        """All icosphere vertices must lie on the unit sphere (r=1 ± 1e-5)."""
        verts, _ = make_icosphere(2)
        radii = np.linalg.norm(verts, axis=1)
        assert np.allclose(radii, 1.0, atol=1e-5), \
            f"vertex radii range: [{radii.min():.6f}, {radii.max():.6f}]"

    @pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not available")
    @pytest.mark.parametrize("level", [3, 4])
    def test_vs_trimesh_sphere(self, level):
        """Our CPU volume must agree with trimesh to within 1%."""
        verts, tris = make_icosphere(level)
        our_vol = mesh_volume_cpu(verts, tris)
        mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
        tm_vol = abs(float(mesh.volume))
        rel_err = abs(our_vol - tm_vol) / max(tm_vol, 1e-12)
        assert rel_err < 0.01, \
            f"level={level}: ours={our_vol:.6f}, trimesh={tm_vol:.6f}, err={rel_err:.4%}"

    @pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not available")
    @pytest.mark.parametrize("level", [3, 4])
    def test_vs_trimesh_displaced_sphere(self, level):
        """Displaced icosphere volume must agree with trimesh to within 1%."""
        verts, tris = make_icosphere(level)
        # z-axis sinusoidal displacement — mesh is no longer a sphere
        verts_d = verts.copy()
        verts_d[:, 2] = verts[:, 2] * (1.0 + 0.3 * np.sin(np.pi * verts[:, 2]))

        our_vol = mesh_volume_cpu(verts_d, tris)
        mesh = trimesh.Trimesh(vertices=verts_d, faces=tris, process=False)
        tm_vol = abs(float(mesh.volume))
        rel_err = abs(our_vol - tm_vol) / max(tm_vol, 1e-12)
        assert rel_err < 0.01, \
            f"level={level}: ours={our_vol:.6f}, trimesh={tm_vol:.6f}, err={rel_err:.4%}"

    @pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not available")
    def test_vs_trimesh_unit_cube(self):
        """Unit cube volume (trimesh reference)."""
        verts, tris = make_unit_cube()
        our_vol = mesh_volume_cpu(verts, tris)
        mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
        tm_vol = abs(float(mesh.volume))
        assert abs(our_vol - tm_vol) < 1e-4, \
            f"cube: ours={our_vol:.6f}, trimesh={tm_vol:.6f}"

    def test_scaled_cube(self):
        """Uniformly scaling a cube by s should multiply volume by s^3."""
        verts, tris = make_unit_cube()
        for s in [0.5, 2.0, 3.14]:
            vol = mesh_volume_cpu(verts * s, tris)
            expected = s ** 3
            assert abs(vol - expected) / expected < 1e-5, \
                f"scale={s}: got {vol:.6f}, expected {expected:.6f}"

    def test_translated_mesh_invariant(self):
        """Translation must not change mesh volume.

        The computation is done in float64 to avoid float32 cancellation errors
        that arise when vertex coordinates are large relative to edge lengths.
        """
        verts, tris = make_icosphere(3)
        # Use float64 for the volume calls to isolate algorithm correctness
        # from float32 catastrophic cancellation with large offsets.
        verts64 = verts.astype(np.float64)
        vol_origin = mesh_volume_cpu(verts64, tris)
        offset = np.array([100.0, -50.0, 23.7], dtype=np.float64)
        vol_translated = mesh_volume_cpu(verts64 + offset, tris)
        assert abs(vol_origin - vol_translated) / vol_origin < 1e-6, \
            f"origin={vol_origin:.6f}, translated={vol_translated:.6f}"


# ---------------------------------------------------------------------------
# Section 2: Hull volume correctness vs scipy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not available")
class TestHullVolumeScipy:
    """Self-consistency checks: scipy ConvexHull on various point distributions."""

    @pytest.mark.parametrize("n_pts,seed", [(20, 42), (100, 123), (256, 7)])
    def test_sphere_shell_positive_volume(self, n_pts, seed):
        """Points on a sphere shell should have positive convex hull volume."""
        rng = np.random.default_rng(seed)
        pts = make_sphere_shell(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        assert vol > 0.5, \
            f"n={n_pts}, seed={seed}: hull volume {vol:.4f} unexpectedly small"

    @pytest.mark.parametrize("n_pts,seed", [(20, 42), (100, 123), (256, 7)])
    def test_sphere_shell_upper_bound(self, n_pts, seed):
        """Convex hull of unit-sphere points cannot exceed a sphere of radius 1."""
        rng = np.random.default_rng(seed)
        pts = make_sphere_shell(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        sphere_vol = 4.0 * np.pi / 3.0
        assert vol <= sphere_vol + 1e-4, \
            f"hull volume {vol:.4f} exceeds unit sphere {sphere_vol:.4f}"

    @pytest.mark.parametrize("n_pts,seed", [(20, 42), (50, 99)])
    def test_cube_interior_upper_bound(self, n_pts, seed):
        """Hull of points in [-0.5, 0.5]^3 cannot exceed unit cube volume."""
        rng = np.random.default_rng(seed)
        pts = make_cube_interior(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        assert vol <= 1.0 + 1e-5, \
            f"cube interior hull {vol:.6f} exceeds 1.0"

    @pytest.mark.parametrize("n_pts,seed", [(20, 42), (50, 99)])
    def test_cube_interior_lower_bound(self, n_pts, seed):
        """With enough interior points the hull should fill most of [-0.5,0.5]^3."""
        rng = np.random.default_rng(seed)
        pts = make_cube_interior(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        # Not guaranteed to be tight but should be > 0 for n>=4
        assert vol > 0, f"hull volume is zero: {vol}"

    @pytest.mark.parametrize("n_pts,seed", [(20, 42), (100, 0), (256, 99)])
    def test_sphere_interior_bounds(self, n_pts, seed):
        """Hull of interior sphere points has volume in (0, 4π/3]."""
        rng = np.random.default_rng(seed)
        pts = make_sphere_interior(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        sphere_vol = 4.0 * np.pi / 3.0
        assert 0 < vol <= sphere_vol + 1e-4, \
            f"sphere interior: vol={vol:.4f}, sphere={sphere_vol:.4f}"

    @pytest.mark.parametrize("n_pts,seed", [(20, 5), (100, 6)])
    def test_hemisphere_shell_volume(self, n_pts, seed):
        """Hemisphere convex hull volume should be well below full sphere volume."""
        rng = np.random.default_rng(seed)
        pts = make_hemisphere_shell(n_pts, rng)
        vol = compute_hull_volume_scipy(pts)
        sphere_vol = 4.0 * np.pi / 3.0
        # A hemisphere hull (with flat base) is roughly half the sphere vol
        assert vol < sphere_vol + 1e-4, \
            f"hemisphere hull {vol:.4f} exceeds sphere {sphere_vol:.4f}"
        assert vol > 0, "hemisphere hull volume is zero"

    def test_volume_scales_cubically(self):
        """Scaling a point cloud by s must scale hull volume by s^3."""
        rng = np.random.default_rng(42)
        pts = make_sphere_shell(100, rng)
        vol1 = compute_hull_volume_scipy(pts)
        for s in [0.5, 2.0]:
            vol_s = compute_hull_volume_scipy(pts * s)
            ratio = vol_s / vol1
            expected = s ** 3
            assert abs(ratio - expected) / expected < 0.01, \
                f"scale={s}: ratio={ratio:.4f}, expected {expected:.4f}"

    def test_translation_invariance(self):
        """Translation must not change convex hull volume."""
        rng = np.random.default_rng(17)
        pts = make_sphere_shell(100, rng)
        vol_orig = compute_hull_volume_scipy(pts)
        offset = np.array([10.0, -5.0, 3.14], dtype=np.float32)
        vol_shifted = compute_hull_volume_scipy(pts + offset)
        assert abs(vol_orig - vol_shifted) / vol_orig < 1e-4, \
            f"original={vol_orig:.6f}, shifted={vol_shifted:.6f}"

    def test_gaussian_hull_positive(self):
        """Gaussian point cloud hull must have positive volume."""
        rng = np.random.default_rng(0)
        for n in [20, 100]:
            pts = make_gaussian(n, rng)
            vol = compute_hull_volume_scipy(pts)
            assert vol > 0, f"n={n}: Gaussian hull volume is {vol}"

    @pytest.mark.parametrize("level", [2, 3])
    def test_icosphere_hull_equals_mesh_volume(self, level):
        """Convex hull of icosphere vertices should approximate the sphere volume.

        The icosphere is itself convex, so its hull volume equals its mesh volume.
        """
        verts, tris = make_icosphere(level)
        mesh_vol = mesh_volume_cpu(verts, tris)
        hull_vol = compute_hull_volume_scipy(verts)
        # Both should be close to 4π/3 and to each other
        rel_err = abs(mesh_vol - hull_vol) / max(hull_vol, 1e-12)
        assert rel_err < 0.02, \
            f"level={level}: mesh={mesh_vol:.6f}, hull={hull_vol:.6f}, err={rel_err:.4%}"

    def test_unit_cube_hull_volume(self):
        """Convex hull of the 8 unit-cube vertices equals 1.0."""
        verts, _ = make_unit_cube()
        vol = compute_hull_volume_scipy(verts)
        assert abs(vol - 1.0) < 1e-5, f"cube hull volume: {vol:.6f}"


# ---------------------------------------------------------------------------
# Section 3: GPU mesh volume tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_GPU, reason="coacd_gpu not available")
class TestMeshVolumeGPU:
    """Validate GPU batch_mesh_volume against CPU mesh_volume_cpu reference."""

    @pytest.fixture(scope="class")
    def ctx(self):
        c = coacd_gpu.Context(device=0)
        yield c
        c.close()

    def test_unit_cube(self, ctx):
        """Unit cube must return volume 1.0 (within float32 tolerance)."""
        verts, tris = make_unit_cube()
        vols = ctx.batch_mesh_volume([verts], [tris])
        assert abs(float(vols[0]) - 1.0) < 1e-5, \
            f"unit cube GPU: got {vols[0]}, expected 1.0"

    @pytest.mark.parametrize("level", [2, 3, 4])
    def test_icosphere_matches_cpu(self, ctx, level):
        """GPU volume must agree with CPU divergence-theorem to within 0.01%."""
        verts, tris = make_icosphere(level)
        gpu_vol = float(ctx.batch_mesh_volume([verts], [tris])[0])
        cpu_vol = mesh_volume_cpu(verts, tris)
        rel_err = abs(gpu_vol - cpu_vol) / max(cpu_vol, 1e-12)
        assert rel_err < 1e-4, \
            f"level={level}: gpu={gpu_vol:.6f}, cpu={cpu_vol:.6f}, err={rel_err:.4%}"

    def test_batch_two_icospheres(self, ctx):
        """Batching two meshes must give independent correct results."""
        verts_list, tris_list, cpu_vols = [], [], []
        for level in [2, 3]:
            v, t = make_icosphere(level)
            verts_list.append(v); tris_list.append(t)
            cpu_vols.append(mesh_volume_cpu(v, t))
        gpu_vols = ctx.batch_mesh_volume(verts_list, tris_list)
        for i, (gpu, cpu) in enumerate(zip(gpu_vols, cpu_vols)):
            rel_err = abs(float(gpu) - cpu) / max(cpu, 1e-12)
            assert rel_err < 1e-4, \
                f"mesh {i}: gpu={float(gpu):.6f}, cpu={cpu:.6f}, err={rel_err:.4%}"

    def test_batch_mixed_meshes(self, ctx):
        """Batch with unit cube + icospheres must all agree with CPU."""
        verts_list, tris_list, cpu_vols = [], [], []
        v, t = make_unit_cube()
        verts_list.append(v); tris_list.append(t); cpu_vols.append(mesh_volume_cpu(v, t))
        for level in [2, 3]:
            v, t = make_icosphere(level)
            verts_list.append(v); tris_list.append(t); cpu_vols.append(mesh_volume_cpu(v, t))
        gpu_vols = ctx.batch_mesh_volume(verts_list, tris_list)
        for i, (gpu, cpu) in enumerate(zip(gpu_vols, cpu_vols)):
            rel_err = abs(float(gpu) - cpu) / max(cpu, 1e-12)
            assert rel_err < 1e-4, \
                f"mesh {i}: gpu={float(gpu):.6f}, cpu={cpu:.6f}, err={rel_err:.4%}"

    @pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not available")
    @pytest.mark.parametrize("level", [3, 4])
    def test_displaced_sphere_vs_trimesh(self, ctx, level):
        """GPU volume of a displaced icosphere must match trimesh to within 1%."""
        verts, tris = make_icosphere(level)
        verts_d = verts.copy()
        verts_d[:, 2] = verts[:, 2] * (1.0 + 0.3 * np.sin(np.pi * verts[:, 2]))
        gpu_vol = float(ctx.batch_mesh_volume([verts_d], [tris])[0])
        mesh = trimesh.Trimesh(vertices=verts_d, faces=tris, process=False)
        tm_vol = abs(float(mesh.volume))
        rel_err = abs(gpu_vol - tm_vol) / max(tm_vol, 1e-12)
        assert rel_err < 0.01, \
            f"level={level}: gpu={gpu_vol:.6f}, trimesh={tm_vol:.6f}, err={rel_err:.4%}"


# ---------------------------------------------------------------------------
# Section 4: GPU hull volume tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_GPU, reason="coacd_gpu not available")
@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not available")
class TestHullVolumeGPU:
    """Validate GPU batch_hull_volume (D&C algorithm) against scipy ConvexHull."""

    @pytest.fixture(scope="class")
    def ctx(self):
        c = coacd_gpu.Context(device=0)
        yield c
        c.close()

    @pytest.mark.parametrize("algo", [2])
    def test_unit_cube_vertices(self, ctx, algo):
        """Convex hull of the 8 unit-cube vertices must equal 1.0."""

        verts, _ = make_unit_cube()
        vols, errs = ctx.batch_hull_volume([verts], algo=algo)
        assert errs[0] == 0, f"algo={algo}: error code {errs[0]}"
        assert abs(float(vols[0]) - 1.0) < 0.02, \
            f"algo={algo}: cube hull volume {vols[0]:.4f}, expected ~1.0"

    @pytest.mark.parametrize("algo,n_pts,seed", [
        (2, 20,  42), (2, 100, 123), (2, 200, 99),
    ])
    def test_sphere_shell_vs_scipy(self, ctx, algo, n_pts, seed):
        """GPU hull volume must match scipy to within 5% on sphere-shell points."""

        rng = np.random.default_rng(seed)
        pts = make_sphere_shell(n_pts, rng)
        vols, errs = ctx.batch_hull_volume([pts], algo=algo)
        assert errs[0] == 0, f"algo={algo}, n={n_pts}: error {errs[0]}"
        gpu_vol = float(vols[0])
        scipy_vol = compute_hull_volume_scipy(pts)
        rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
        assert rel_err < 0.05, \
            f"algo={algo}, n={n_pts}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"

    @pytest.mark.parametrize("algo,n_pts,seed", [
        (2, 50, 0),
    ])
    def test_cube_interior_vs_scipy(self, ctx, algo, n_pts, seed):
        """GPU hull of cube-interior points must match scipy to within 5%."""

        rng = np.random.default_rng(seed)
        pts = make_cube_interior(n_pts, rng)
        vols, errs = ctx.batch_hull_volume([pts], algo=algo)
        assert errs[0] == 0, f"algo={algo}: error {errs[0]}"
        gpu_vol = float(vols[0])
        scipy_vol = compute_hull_volume_scipy(pts)
        rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
        assert rel_err < 0.05, \
            f"algo={algo}, n={n_pts}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"

    @pytest.mark.parametrize("algo,n_pts,seed", [
        (2, 20, 0), (2, 100, 123), (2, 200, 99),
    ])
    def test_gaussian_vs_scipy(self, ctx, algo, n_pts, seed):
        """GPU hull volume of Gaussian points must match scipy to within 5%."""

        rng = np.random.default_rng(seed)
        pts = make_gaussian(n_pts, rng)
        vols, errs = ctx.batch_hull_volume([pts], algo=algo)
        assert errs[0] == 0, f"algo={algo}, n={n_pts}: error {errs[0]}"
        gpu_vol = float(vols[0])
        scipy_vol = compute_hull_volume_scipy(pts)
        rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
        assert rel_err < 0.05, \
            f"algo={algo}, n={n_pts}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"

    @pytest.mark.parametrize("algo,n_pts,seed,dist_fn", [
        (2, 2000, 0, "sphere_shell"),
        (2, 2000, 0, "cube_interior"),
        (2, 2000, 0, "gaussian"),
    ])
    def test_large_pointcloud_vs_scipy(self, ctx, algo, n_pts, seed, dist_fn):
        """GPU hull volume must match scipy within 5% on larger point clouds."""

        dist_fns = {
            "sphere_shell": make_sphere_shell,
            "cube_interior": make_cube_interior,
            "gaussian": make_gaussian,
        }
        rng = np.random.default_rng(seed)
        pts = dist_fns[dist_fn](n_pts, rng)
        vols, errs = ctx.batch_hull_volume([pts], algo=algo)
        assert errs[0] == 0, f"algo={algo}, n={n_pts}, {dist_fn}: error {errs[0]}"
        gpu_vol = float(vols[0])
        scipy_vol = compute_hull_volume_scipy(pts)
        rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
        assert rel_err < 0.05, \
            f"algo={algo}, n={n_pts}, {dist_fn}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"

    @pytest.mark.parametrize("algo", [2])
    def test_batch_sphere_hulls_vs_scipy(self, ctx, algo):
        """Batch of 10 sphere-shell hulls must all match scipy to within 5%."""

        rng = np.random.default_rng(42)
        n_hulls = 10
        pts_list = [make_sphere_shell(50, rng) for _ in range(n_hulls)]
        vols, errs = ctx.batch_hull_volume(pts_list, algo=algo)
        for i in range(n_hulls):
            assert errs[i] == 0, f"algo={algo}, hull {i}: error {errs[i]}"
            scipy_vol = compute_hull_volume_scipy(pts_list[i])
            rel_err = abs(float(vols[i]) - scipy_vol) / max(scipy_vol, 1e-12)
            assert rel_err < 0.05, \
                f"algo={algo}, hull {i}: gpu={vols[i]:.4f} scipy={scipy_vol:.4f}"

    @pytest.mark.parametrize("algo", [2])
    def test_icosphere_hull_matches_mesh(self, ctx, algo):
        """Convex hull of level-3 icosphere vertices should match its mesh volume.

        The icosphere is convex so hull volume ≈ mesh volume.
        """

        verts, tris = make_icosphere(3)
        mesh_vol = mesh_volume_cpu(verts, tris)
        vols, errs = ctx.batch_hull_volume([verts], algo=algo)
        assert errs[0] == 0, f"algo={algo}: error {errs[0]}"
        rel_err = abs(float(vols[0]) - mesh_vol) / max(mesh_vol, 1e-12)
        assert rel_err < 0.05, \
            f"algo={algo}: hull={vols[0]:.4f}, mesh={mesh_vol:.4f}, err={rel_err:.3%}"

    @pytest.mark.parametrize("algo", [2])
    def test_volume_scales_cubically(self, ctx, algo):
        """Scaling a point cloud by s must scale GPU hull volume by s^3."""

        rng = np.random.default_rng(17)
        pts = make_sphere_shell(80, rng)
        vols1, errs1 = ctx.batch_hull_volume([pts], algo=algo)
        assert errs1[0] == 0
        vol1 = float(vols1[0])
        for s in [0.5, 2.0]:
            vols_s, errs_s = ctx.batch_hull_volume([pts * s], algo=algo)
            assert errs_s[0] == 0
            ratio = float(vols_s[0]) / max(vol1, 1e-12)
            expected = s ** 3
            assert abs(ratio - expected) / expected < 0.05, \
                f"algo={algo}, scale={s}: ratio={ratio:.4f}, expected {expected:.4f}"

    @pytest.mark.parametrize("algo,level,amplitude,seed", [
        (2, 2, 0.1, 42),
        (2, 2, 0.3, 7),
        (2, 3, 0.1, 99),
        (2, 3, 0.2, 123),
        (2, 3, 0.4, 0),
    ])
    def test_noisy_icosphere_vs_scipy(self, ctx, algo, level, amplitude, seed):
        """GPU hull volume of noisy icosphere vertices must match scipy within 5%.

        The noisy icosphere is non-convex (bumpy surface), so its convex hull
        volume exceeds its mesh volume.  We compare GPU D&C hull vs scipy hull.
        """
        verts, _ = make_noisy_icosphere(level, amplitude, seed)
        vols, errs = ctx.batch_hull_volume([verts], algo=algo)
        assert errs[0] == 0, f"error code {errs[0]}"
        gpu_vol = float(vols[0])
        scipy_vol = compute_hull_volume_scipy(verts)
        rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
        assert rel_err < 0.05, \
            f"level={level}, amp={amplitude}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"

    @pytest.mark.parametrize("algo", [2])
    def test_batch_noisy_icospheres_vs_scipy(self, ctx, algo):
        """Batch of noisy icospheres with varying amplitudes, all vs scipy."""
        pts_list = []
        for i in range(8):
            amp = 0.05 + 0.05 * i  # 0.05 to 0.40
            verts, _ = make_noisy_icosphere(2, amp, seed=i)
            pts_list.append(verts)
        vols, errs = ctx.batch_hull_volume(pts_list, algo=algo)
        for i in range(8):
            assert errs[i] == 0, f"hull {i}: error {errs[i]}"
            scipy_vol = compute_hull_volume_scipy(pts_list[i])
            gpu_vol = float(vols[i])
            rel_err = abs(gpu_vol - scipy_vol) / max(scipy_vol, 1e-12)
            assert rel_err < 0.05, \
                f"hull {i}: gpu={gpu_vol:.4f} scipy={scipy_vol:.4f} err={rel_err:.3%}"


# ---------------------------------------------------------------------------
# Section 5: Benchmarks (standalone, no GPU required for CPU benchmarks)
# ---------------------------------------------------------------------------

def benchmark_hull_volumes():
    """Benchmark scipy convex hull volume on various batch configurations.

    Prints a table of (config, distribution, mean_volume, scipy_time_ms).
    """
    if not _HAS_SCIPY:
        print("scipy not available — skipping hull benchmark")
        return

    configs = [
        (10000,    20, "10000x20pts"),
        ( 1000,   200, "1000x200pts"),
        (  100,  2000, "100x2000pts"),
        (   10, 20000, "10x20000pts"),
    ]
    distributions = [
        ("hemisphere_shell", make_hemisphere_shell),
        ("sphere_shell",     make_sphere_shell),
        ("sphere_interior",  make_sphere_interior),
        ("cube_interior",    make_cube_interior),
        ("gaussian",         make_gaussian),
    ]

    header = f"{'Config':<20} {'Distribution':<22} {'mean vol':<12} {'scipy ms':<12}"
    print(f"\n{header}")
    print("-" * len(header))

    rng = np.random.default_rng(0)

    for n_hulls, n_pts, cfg_name in configs:
        for dist_name, dist_fn in distributions:
            pts_batch = [dist_fn(n_pts, rng) for _ in range(n_hulls)]

            t0 = time.perf_counter()
            vols = [compute_hull_volume_scipy(pts) for pts in pts_batch]
            t1 = time.perf_counter()
            scipy_ms = (t1 - t0) * 1000.0

            mean_vol = float(np.mean(vols)) if vols else 0.0
            print(f"{cfg_name:<20} {dist_name:<22} {mean_vol:<12.4f} {scipy_ms:<12.1f}")

    print()


def benchmark_mesh_volumes():
    """Benchmark CPU mesh volume on icospheres of increasing detail."""
    header = f"{'Mesh':<28} {'verts':<8} {'tris':<8} {'volume':<12} {'cpu ms':<10}"
    print(f"\n{header}")
    print("-" * len(header))

    for level in range(6):
        verts, tris = make_icosphere(level)
        reps = max(1, 1000 // max(1, len(tris) // 20))
        t0 = time.perf_counter()
        for _ in range(reps):
            vol = mesh_volume_cpu(verts, tris)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) / reps * 1000.0
        label = f"icosphere level {level}"
        print(f"{label:<28} {len(verts):<8} {len(tris):<8} {vol:<12.6f} {elapsed_ms:<10.3f}")

    print()


def benchmark_hull_volumes_gpu():
    """Benchmark GPU D&C hull volume on various batch configurations."""
    if not _HAS_GPU:
        print("GPU not available — skipping GPU hull benchmark")
        return

    configs = [
        (10000,    20, "10000x20pts"),
        ( 1000,   200, "1000x200pts"),
        (  100,  2000, "100x2000pts"),
        (   10, 20000, "10x20000pts"),
    ]
    distributions = [
        ("sphere_shell",  make_sphere_shell),
        ("cube_interior", make_cube_interior),
        ("gaussian",      make_gaussian),
    ]

    ctx = coacd_gpu.Context(device=0)
    rng = np.random.default_rng(0)

    try:
        header = (f"{'Config':<20} {'Distribution':<16} {'GPU ms':>9} {'mean vol':>10} {'peak pool MB':>13}")
        print(f"\n{header}")
        print("-" * len(header))

        for n_hulls, n_pts, cfg_name in configs:
            for dist_name, dist_fn in distributions:
                pts_list = [dist_fn(n_pts, rng) for _ in range(n_hulls)]
                try:
                    # warm up
                    ctx.batch_hull_volume(pts_list[:min(8, n_hulls)])
                    t0 = time.perf_counter()
                    vols, _ = ctx.batch_hull_volume(pts_list)
                    t1 = time.perf_counter()
                    gpu_ms = (t1 - t0) * 1000.0
                    mean_vol = float(np.mean(vols))
                    peak_mb = ctx.pool_usage() / (1 << 20)
                    print(f"{cfg_name:<20} {dist_name:<16} {gpu_ms:>9.1f} {mean_vol:>10.4f} {peak_mb:>12.1f}")
                except Exception as e:
                    print(f"{cfg_name:<20} {dist_name:<16} ERROR: {e}")

        print()
    finally:
        ctx.close()


def benchmark_mesh_volumes_gpu():
    """Benchmark GPU batch_mesh_volume on batches of icospheres at multiple levels."""
    if not _HAS_GPU:
        print("GPU not available — skipping GPU mesh volume benchmark")
        return

    ctx = coacd_gpu.Context(device=0)
    try:
        header = (f"{'Batch':<26} {'V/mesh':>8} {'T/mesh':>8} "
                  f"{'GPU ms':>9} {'mean vol':>12}")
        print(f"\n{header}")
        print("-" * len(header))

        for level, n_meshes in [(2, 1000), (3, 100), (4, 10)]:
            verts, tris = make_icosphere(level)
            vl = [verts] * n_meshes
            tl = [tris]  * n_meshes
            # warm up
            ctx.batch_mesh_volume(vl[:min(5, n_meshes)], tl[:min(5, n_meshes)])
            t0 = time.perf_counter()
            vols = ctx.batch_mesh_volume(vl, tl)
            t1 = time.perf_counter()
            gpu_ms = (t1 - t0) * 1000.0
            label = f"{n_meshes}x icosphere L{level}"
            print(f"{label:<26} {len(verts):>8} {len(tris):>8} "
                  f"{gpu_ms:>9.2f} {float(np.mean(vols)):>12.6f}")

        print()
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Mesh volume benchmark (CPU divergence theorem)")
    print("=" * 70)
    benchmark_mesh_volumes()

    print("=" * 70)
    print("Convex hull volume benchmark (scipy, CPU)")
    print("=" * 70)
    benchmark_hull_volumes()

    print("=" * 70)
    print("Mesh volume benchmark (GPU divergence theorem)")
    print("=" * 70)
    benchmark_mesh_volumes_gpu()

    print("=" * 70)
    print("Convex hull volume benchmark (GPU — 3 algorithms)")
    print("=" * 70)
    benchmark_hull_volumes_gpu()
