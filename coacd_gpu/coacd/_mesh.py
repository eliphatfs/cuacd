"""Lightweight mesh wrapper with convex hull support."""

import numpy as np
from scipy.spatial import ConvexHull

from ._geometry import compute_bbox, mesh_volume, mesh_area


class Mesh:
    """Triangle mesh: vertices (N,3) float64, triangles (M,3) int32."""

    __slots__ = ("vertices", "triangles", "bbox")

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, bbox=None):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int32)
        self.bbox = bbox if bbox is not None else compute_bbox(self.vertices)

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_triangles(self):
        return len(self.triangles)

    def volume(self) -> float:
        return mesh_volume(self.vertices, self.triangles)

    def area(self) -> float:
        return mesh_area(self.vertices, self.triangles)

    def convex_hull(self) -> "Mesh":
        """Compute convex hull via scipy (Qhull)."""
        if len(self.vertices) < 4:
            return Mesh(self.vertices.copy(), self.triangles.copy())
        try:
            hull = ConvexHull(self.vertices)
        except Exception:
            # Degenerate: return copy of self
            return Mesh(self.vertices.copy(), self.triangles.copy())
        verts = self.vertices[hull.vertices]
        # Remap simplices to new vertex indices
        remap = np.full(len(self.vertices), -1, dtype=np.int32)
        remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
        tris = remap[hull.simplices].copy()
        # Fix orientation: scipy simplices may not be consistently wound.
        # Use hull.equations normals to ensure outward-pointing cross products.
        for i in range(len(tris)):
            p0 = verts[tris[i, 0]]
            p1 = verts[tris[i, 1]]
            p2 = verts[tris[i, 2]]
            face_normal = np.cross(p1 - p0, p2 - p0)
            eq_normal = hull.equations[i, :3]
            if np.dot(face_normal, eq_normal) < 0:
                tris[i, 0], tris[i, 2] = tris[i, 2], tris[i, 0]
        # Ensure positive volume (outward normals -> positive signed volume)
        ch = Mesh(verts, tris)
        if ch.volume() < 0:
            tris = tris[:, ::-1].copy()
            ch = Mesh(verts, tris)
        return ch

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.triangles.copy(), self.bbox)
