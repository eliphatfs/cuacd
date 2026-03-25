"""Cost functions: Rv (volume-based), Hb (Hausdorff-based), HCost (combined)."""

import numpy as np
from scipy.spatial import cKDTree

from ._geometry import mesh_volume, mesh_area, Plane
from ._mesh import Mesh
from ._sampling import sample_surface

_PI = 3.14159265
INF = float("inf")


def compute_rv(mesh: Mesh, ch: Mesh, k: float = 0.3) -> float:
    """Volume-based concavity between mesh and its convex hull."""
    v1 = mesh.volume()
    v2 = ch.volume()
    d = (3.0 * abs(v1 - v2) / (4.0 * _PI)) ** (1.0 / 3.0) * k
    return d


def compute_rv_3(ch1: Mesh, ch2: Mesh, ch_combined: Mesh, k: float = 0.3) -> float:
    """Volume-based concavity for merge cost (three meshes)."""
    v1 = ch1.volume()
    v2 = ch2.volume()
    v3 = ch_combined.volume()
    d = (3.0 * abs(v1 + v2 - v3) / (4.0 * _PI)) ** (1.0 / 3.0) * k
    return d


def compute_total_rv(pos: Mesh, pos_ch: Mesh, neg: Mesh, neg_ch: Mesh,
                     k: float = 0.3) -> float:
    """Max Rv of two halves."""
    return max(compute_rv(pos, pos_ch, k), compute_rv(neg, neg_ch, k))


# ---------------------------------------------------------------------------
# Hausdorff distance (CPU fallback via KD-tree, matching C++ logic)
# ---------------------------------------------------------------------------

def _dist_point2segment(pt, s0, s1):
    """Distance from point to line segment, INF if projection outside."""
    ba = pt - s1
    bc = s0 - s1
    bc_len = np.linalg.norm(bc)
    if bc_len < 1e-30:
        return np.linalg.norm(pt - s0)
    proj = np.dot(ba, bc) / bc_len
    if proj < 0 or proj > bc_len:
        return INF
    return np.sqrt(max(0.0, np.dot(ba, ba) - proj * proj))


def _dist_point2triangle(pt, t0, t1, t2):
    """Exact point-to-triangle distance (Eberly-style)."""
    v = t1 - t0
    w = t2 - t0
    raw_n = np.cross(v, w)
    n_len = np.linalg.norm(raw_n)
    if n_len < 1e-30:
        # Degenerate triangle
        return min(np.linalg.norm(pt - t0), np.linalg.norm(pt - t1), np.linalg.norm(pt - t2))
    n = raw_n / n_len
    d_val = -np.dot(n, t0)

    dist_plane = abs(np.dot(n, pt) + d_val)
    side = np.dot(n, pt) + d_val
    if side > 1e-8:
        proj = pt - dist_plane * n
    elif side < -1e-8:
        proj = pt + dist_plane * n
    else:
        proj = pt.copy()

    # Inside-triangle test via cross products
    ab = t1 - t0
    bc = t2 - t1
    ca = t0 - t2
    ap = proj - t0
    bp = proj - t1
    cp = proj - t2

    if (np.dot(np.cross(ab, ap), raw_n) >= 0 and
            np.dot(np.cross(bc, bp), raw_n) >= 0 and
            np.dot(np.cross(ca, cp), raw_n) >= 0):
        return dist_plane

    # Outside: check edges and vertices
    d1 = _dist_point2segment(pt, t0, t1)
    d2 = _dist_point2segment(pt, t1, t2)
    d3 = _dist_point2segment(pt, t2, t0)
    d4 = np.linalg.norm(pt - t0)
    d5 = np.linalg.norm(pt - t1)
    d6 = np.linalg.norm(pt - t2)
    return min(d1, d2, d3, d4, d5, d6)


def hausdorff_cpu(mesh_a: Mesh, samples_a: np.ndarray, tri_ids_a: np.ndarray,
                  mesh_b: Mesh, samples_b: np.ndarray, tri_ids_b: np.ndarray) -> float:
    """CPU Hausdorff distance matching C++ face_hausdorff_distance."""
    if len(samples_a) == 0 or len(samples_b) == 0:
        return INF

    tree_a = cKDTree(samples_a)
    tree_b = cKDTree(samples_b)
    cmax = 0.0

    # B -> A direction
    dists_sq, idxs = tree_a.query(samples_b, k=min(10, len(samples_a)))
    if idxs.ndim == 1:
        idxs = idxs[:, None]
        dists_sq = dists_sq[:, None]
    for i in range(len(samples_b)):
        cmin = INF
        for j in range(idxs.shape[1]):
            nn_idx = idxs[i, j]
            ti = tri_ids_a[nn_idx]
            t0 = mesh_a.vertices[mesh_a.triangles[ti, 0]]
            t1 = mesh_a.vertices[mesh_a.triangles[ti, 1]]
            t2 = mesh_a.vertices[mesh_a.triangles[ti, 2]]
            d = _dist_point2triangle(samples_b[i], t0, t1, t2)
            if d < cmin:
                cmin = d
                if cmin < 1e-14:
                    break
        if cmin > 10:
            cmin = np.sqrt(dists_sq[i, 0]) if dists_sq.shape[1] > 0 else INF
        if cmin > cmax and cmin < INF:
            cmax = cmin

    # A -> B direction
    dists_sq, idxs = tree_b.query(samples_a, k=min(10, len(samples_b)))
    if idxs.ndim == 1:
        idxs = idxs[:, None]
        dists_sq = dists_sq[:, None]
    for i in range(len(samples_a)):
        cmin = INF
        for j in range(idxs.shape[1]):
            nn_idx = idxs[i, j]
            ti = tri_ids_b[nn_idx]
            t0 = mesh_b.vertices[mesh_b.triangles[ti, 0]]
            t1 = mesh_b.vertices[mesh_b.triangles[ti, 1]]
            t2 = mesh_b.vertices[mesh_b.triangles[ti, 2]]
            d = _dist_point2triangle(samples_a[i], t0, t1, t2)
            if d < cmin:
                cmin = d
                if cmin < 1e-14:
                    break
        if cmin > 10:
            cmin = np.sqrt(dists_sq[i, 0]) if dists_sq.shape[1] > 0 else INF
        if cmin > cmax and cmin < INF:
            cmax = cmin

    return cmax


# ---------------------------------------------------------------------------
# High-level cost functions
# ---------------------------------------------------------------------------

def compute_hb(mesh: Mesh, ch: Mesh, resolution: int = 2000, seed: int = 1234,
               gpu_ctx=None) -> float:
    """Hausdorff-based concavity between mesh and its convex hull."""
    rng = np.random.RandomState(seed)
    sa, ida = sample_surface(mesh.vertices, mesh.triangles, resolution, rng)
    rng2 = np.random.RandomState(seed)
    sb, idb = sample_surface(ch.vertices, ch.triangles, resolution, rng2)

    if len(sa) == 0 or len(sb) == 0:
        return INF

    if gpu_ctx is not None:
        return gpu_ctx.hausdorff(
            sa.astype(np.float32), mesh.vertices.astype(np.float32), mesh.triangles,
            sb.astype(np.float32), ch.vertices.astype(np.float32), ch.triangles,
        )
    return hausdorff_cpu(mesh, sa, ida, ch, sb, idb)


def compute_hcost(mesh: Mesh, ch: Mesh, k: float = 0.3,
                  resolution: int = 2000, seed: int = 1234,
                  gpu_ctx=None) -> float:
    """Combined cost = max(Rv, Hb)."""
    h1 = compute_rv(mesh, ch, k)
    h2 = compute_hb(mesh, ch, resolution, seed, gpu_ctx)
    return max(h1, h2)


def merge_meshes(m1: Mesh, m2: Mesh) -> Mesh:
    """Concatenate two meshes into one."""
    verts = np.vstack([m1.vertices, m2.vertices])
    tris2 = m2.triangles + len(m1.vertices)
    tris = np.vstack([m1.triangles, tris2])
    return Mesh(verts, tris)


def mesh_dist(ch1: Mesh, ch2: Mesh) -> float:
    """Minimum vertex-to-vertex distance between two meshes (for merge pre-filter)."""
    if len(ch1.vertices) == 0 or len(ch2.vertices) == 0:
        return INF
    tree = cKDTree(ch2.vertices)
    dists, _ = tree.query(ch1.vertices, k=1)
    return float(dists.min())


def compute_hb_3(ch1: Mesh, ch2: Mesh, ch_combined: Mesh,
                 resolution: int = 2000, seed: int = 1234,
                 gpu_ctx=None) -> float:
    """Hausdorff for merge cost (three-mesh variant)."""
    if ch1.n_vertices + ch2.n_vertices == ch_combined.n_vertices:
        return 0.0

    merged = merge_meshes(ch1, ch2)
    rng1 = np.random.RandomState(seed)
    sa, ida = sample_surface(merged.vertices, merged.triangles, resolution, rng1)
    rng2 = np.random.RandomState(seed)
    sb, idb = sample_surface(ch_combined.vertices, ch_combined.triangles, resolution, rng2)

    if len(sa) == 0 or len(sb) == 0:
        return INF

    if gpu_ctx is not None:
        return gpu_ctx.hausdorff(
            sa.astype(np.float32), merged.vertices.astype(np.float32), merged.triangles,
            sb.astype(np.float32), ch_combined.vertices.astype(np.float32), ch_combined.triangles,
        )
    return hausdorff_cpu(merged, sa, ida, ch_combined, sb, idb)


def compute_hcost_3(ch1: Mesh, ch2: Mesh, ch_combined: Mesh,
                    k: float = 0.3, resolution: int = 2000, seed: int = 1234,
                    gpu_ctx=None) -> float:
    """Combined merge cost = max(Rv, Hb) for three-mesh variant."""
    h1 = compute_rv_3(ch1, ch2, ch_combined, k)
    h2 = compute_hb_3(ch1, ch2, ch_combined, resolution + 2000, seed, gpu_ctx)
    return max(h1, h2)
