"""Plane, mesh volume/area, normalize, PCA utilities."""

import numpy as np

_PI = 3.14159265


class Plane:
    """Half-space plane ax + by + cz + d = 0."""

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    # -- vectorised helpers --------------------------------------------------

    def side_values(self, pts: np.ndarray) -> np.ndarray:
        """Return a*x + b*y + c*z + d for each point (N, 3)."""
        return pts[:, 0] * self.a + pts[:, 1] * self.b + pts[:, 2] * self.c + self.d

    def side_one(self, pt, eps: float = 1e-6) -> int:
        """Return -1, 0, or 1 for a single point (3,)."""
        r = pt[0] * self.a + pt[1] * self.b + pt[2] * self.c + self.d
        if r > eps:
            return 1
        elif r < -eps:
            return -1
        return 0

    def sides(self, pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Return -1/0/1 array for (N, 3) points."""
        v = self.side_values(pts)
        out = np.zeros(len(v), dtype=np.int8)
        out[v > eps] = 1
        out[v < -eps] = -1
        return out

    def intersect_segment(self, p1, p2, eps: float = 1e-6):
        """Compute intersection of plane with segment p1-p2.

        Returns (True, intersection_point) or (False, None).
        """
        denom = (self.a * (p2[0] - p1[0]) +
                 self.b * (p2[1] - p1[1]) +
                 self.c * (p2[2] - p1[2]))
        if abs(denom) < 1e-30:
            return False, None
        numer = -(self.a * p1[0] + self.b * p1[1] + self.c * p1[2] + self.d)
        t = numer / denom
        pi = np.array([
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2]),
        ])
        # Check within segment bounding box
        for k in range(3):
            lo = min(p1[k], p2[k]) - eps
            hi = max(p1[k], p2[k]) + eps
            if pi[k] < lo or pi[k] > hi:
                return False, None
        return True, pi

    def cut_side(self, p0, p1, p2):
        """Determine which side a coplanar triangle should go to.

        Uses face normal vs plane normal dot product.
        """
        normal = _face_normal(p0, p1, p2)
        if (normal[0] * self.a > 0 or
                normal[1] * self.b > 0 or
                normal[2] * self.c > 0):
            return -1
        return 1


def _face_normal(p0, p1, p2):
    """Unnormalised face normal via cross product."""
    v = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])
    w = np.array([p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]])
    return np.cross(v, w)


# ---------------------------------------------------------------------------
# Mesh metrics (vectorised)
# ---------------------------------------------------------------------------

def mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Signed volume of a triangle mesh (sum of signed tetrahedron volumes)."""
    p1 = vertices[triangles[:, 0]]
    p2 = vertices[triangles[:, 1]]
    p3 = vertices[triangles[:, 2]]
    # Scalar triple product with origin
    vol = np.sum(
        p1[:, 0] * (p2[:, 1] * p3[:, 2] - p2[:, 2] * p3[:, 1]) +
        p1[:, 1] * (p2[:, 2] * p3[:, 0] - p2[:, 0] * p3[:, 2]) +
        p1[:, 2] * (p2[:, 0] * p3[:, 1] - p2[:, 1] * p3[:, 0])
    ) / 6.0
    return vol


def mesh_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Total surface area of a triangle mesh."""
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.sum(np.linalg.norm(cross, axis=1))


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per-triangle area array."""
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


# ---------------------------------------------------------------------------
# Normalize / Recover / PCA
# ---------------------------------------------------------------------------

def normalize(vertices: np.ndarray, bbox):
    """Scale mesh to [-1, 1] cube.  Returns (new_vertices, new_bbox, orig_bbox)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    m_len = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
    new_verts = 2.0 * (vertices - mid) / m_len
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    new_bbox = (-x_len / m_len, x_len / m_len,
                -y_len / m_len, y_len / m_len,
                -z_len / m_len, z_len / m_len)
    orig_bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    return new_verts, new_bbox, orig_bbox


def recover(vertices: np.ndarray, orig_bbox):
    """Reverse normalisation."""
    x_min, x_max, y_min, y_max, z_min, z_max = orig_bbox
    m_len = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
    return vertices / 2.0 * m_len + mid


def pca_align(vertices: np.ndarray):
    """Align vertices to principal axes.  Returns (new_vertices, rotation_matrix, new_bbox)."""
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = (centered.T @ centered) / len(vertices)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvecs columns are principal axes, sorted ascending eigenvalue
    rot = eigvecs.T  # rows are principal axes
    new_verts = centered @ rot.T  # same as rot @ (centered.T)  per-point
    bbox = compute_bbox(new_verts)
    return new_verts, rot, bbox


def revert_pca(vertices: np.ndarray, rot: np.ndarray):
    """Reverse PCA rotation: rot is the rotation applied during PCA."""
    return vertices @ rot  # rot.T.T = rot


def compute_bbox(vertices: np.ndarray):
    """Return (x_min, x_max, y_min, y_max, z_min, z_max)."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
