"""Plane-mesh clipping with CDT cap triangulation.

Translates CoACD/src/clip.cpp to Python using the `triangle` library for CDT.
"""

import numpy as np
import triangle as tr

from ._geometry import Plane, compute_bbox
from ._mesh import Mesh

_SAME_PT_EPS = 1e-4


def _same_point(a, b):
    return (abs(a[0] - b[0]) < _SAME_PT_EPS and
            abs(a[1] - b[1]) < _SAME_PT_EPS and
            abs(a[2] - b[2]) < _SAME_PT_EPS)


class _BorderCollector:
    """Collects border vertices and edges during clipping."""

    def __init__(self):
        self.points = []          # list of (x,y,z) border points
        self.edges = []           # list of (i, j) 0-based border edge pairs
        self.overlap = []         # coplanar triangle vertices
        self.vertex_map = {}      # original_vertex_id -> border_index
        self.edge_map = {}        # (id1, id2) -> border_index

    def _find_existing(self, pt):
        for i, bp in enumerate(self.points):
            if _same_point(bp, pt):
                return i
        return -1

    def add_vertex(self, pt, vid):
        """Add a vertex that lies on the plane."""
        if vid not in self.vertex_map:
            existing = self._find_existing(pt)
            if existing == -1:
                idx = len(self.points)
                self.vertex_map[vid] = idx
                self.points.append(pt.copy())
            else:
                self.vertex_map[vid] = existing
        return self.vertex_map[vid]

    def add_edge_point(self, pt, id1, id2):
        """Add an intersection point on edge (id1, id2)."""
        key1 = (id1, id2)
        key2 = (id2, id1)
        if key1 not in self.edge_map and key2 not in self.edge_map:
            existing = self._find_existing(pt)
            if existing == -1:
                idx = len(self.points)
                self.edge_map[key1] = idx
                self.edge_map[key2] = idx
                self.points.append(pt.copy())
            else:
                self.edge_map[key1] = existing
                self.edge_map[key2] = existing
        if key1 in self.edge_map:
            return self.edge_map[key1]
        return self.edge_map[key2]

    def add_border_edge(self, i, j):
        if i != j:
            self.edges.append((i, j))


def clip(mesh: Mesh, plane: Plane):
    """Clip mesh along plane.

    Returns (success, pos_mesh, neg_mesh) where pos is on the positive side
    of the plane and neg on the negative side.
    """
    verts = mesh.vertices
    tris = mesh.triangles
    n_verts = len(verts)

    bc = _BorderCollector()

    # Track which original vertices end up in pos/neg
    pos_used = np.zeros(n_verts, dtype=bool)
    neg_used = np.zeros(n_verts, dtype=bool)

    # Temporary triangle lists (indices reference original mesh or encoded border points)
    # Positive indices = original vertex ids
    # Negative indices = encoded border point: -(border_idx + 1)
    pos_tris = []
    neg_tris = []

    for ti in range(len(tris)):
        id0, id1, id2 = int(tris[ti, 0]), int(tris[ti, 1]), int(tris[ti, 2])
        p0, p1, p2 = verts[id0], verts[id1], verts[id2]
        s0 = plane.side_one(p0)
        s1 = plane.side_one(p1)
        s2 = plane.side_one(p2)
        s_sum = s0 + s1 + s2

        # Coplanar triangle
        if s0 == 0 and s1 == 0 and s2 == 0:
            cs = plane.cut_side(p0, p1, p2)
            s0 = s1 = s2 = cs
            s_sum = s0 + s1 + s2
            bc.overlap.extend([p0.copy(), p1.copy(), p2.copy()])

        # Entirely on positive side
        if s_sum == 3 or s_sum == 2 or (s_sum == 1 and _one_vertex_pos(s0, s1, s2)):
            pos_used[id0] = pos_used[id1] = pos_used[id2] = True
            pos_tris.append((id0, id1, id2))
            if s_sum == 1:
                _add_border_for_touching(bc, plane, s0, s1, s2, p0, p1, p2, id0, id1, id2, positive=True)
        # Entirely on negative side
        elif s_sum == -3 or s_sum == -2 or (s_sum == -1 and _one_vertex_neg(s0, s1, s2)):
            neg_used[id0] = neg_used[id1] = neg_used[id2] = True
            neg_tris.append((id0, id1, id2))
            if s_sum == -1:
                _add_border_for_touching(bc, plane, s0, s1, s2, p0, p1, p2, id0, id1, id2, positive=False)
        # Straddling
        else:
            ok = _handle_straddling(
                bc, plane, verts,
                id0, id1, id2, p0, p1, p2, s0, s1, s2,
                pos_tris, neg_tris, pos_used, neg_used
            )
            if not ok:
                return False, None, None

    # Triangulate the cap
    border_verts, cap_tris = _triangulate_cap(bc, plane)
    if border_verts is None:
        border_verts = np.array(bc.points) if bc.points else np.empty((0, 3))
        cap_tris = []

    # Build output meshes
    pos_mesh = _build_output(verts, pos_tris, pos_used, border_verts, cap_tris, flip_cap=False)
    neg_mesh = _build_output(verts, neg_tris, neg_used, border_verts, cap_tris, flip_cap=True)

    if pos_mesh is None or neg_mesh is None:
        return False, None, None
    if pos_mesh.n_triangles == 0 or neg_mesh.n_triangles == 0:
        return False, None, None

    return True, pos_mesh, neg_mesh


def _one_vertex_pos(s0, s1, s2):
    return ((s0 == 1 and s1 == 0 and s2 == 0) or
            (s0 == 0 and s1 == 1 and s2 == 0) or
            (s0 == 0 and s1 == 0 and s2 == 1))


def _one_vertex_neg(s0, s1, s2):
    return ((s0 == -1 and s1 == 0 and s2 == 0) or
            (s0 == 0 and s1 == -1 and s2 == 0) or
            (s0 == 0 and s1 == 0 and s2 == -1))


def _add_border_for_touching(bc, plane, s0, s1, s2, p0, p1, p2, id0, id1, id2, positive):
    """Handle triangles touching the plane at one edge (sum=±1)."""
    if positive:
        if s0 == 1 and s1 == 0 and s2 == 0:
            i1 = bc.add_vertex(p1, id1)
            i2 = bc.add_vertex(p2, id2)
            bc.add_border_edge(i1, i2)
        elif s0 == 0 and s1 == 1 and s2 == 0:
            i2 = bc.add_vertex(p2, id2)
            i0 = bc.add_vertex(p0, id0)
            bc.add_border_edge(i2, i0)
        elif s0 == 0 and s1 == 0 and s2 == 1:
            i0 = bc.add_vertex(p0, id0)
            i1 = bc.add_vertex(p1, id1)
            bc.add_border_edge(i0, i1)
    else:
        if s0 == -1 and s1 == 0 and s2 == 0:
            i2 = bc.add_vertex(p2, id2)
            i1 = bc.add_vertex(p1, id1)
            bc.add_border_edge(i2, i1)
        elif s0 == 0 and s1 == -1 and s2 == 0:
            i0 = bc.add_vertex(p0, id0)
            i2 = bc.add_vertex(p2, id2)
            bc.add_border_edge(i0, i2)
        elif s0 == 0 and s1 == 0 and s2 == -1:
            i1 = bc.add_vertex(p1, id1)
            i0 = bc.add_vertex(p0, id0)
            bc.add_border_edge(i1, i0)


def _handle_straddling(bc, plane, verts, id0, id1, id2, p0, p1, p2, s0, s1, s2,
                       pos_tris, neg_tris, pos_used, neg_used):
    """Handle triangle straddling the plane. Returns True on success."""
    f0_ok, pi0 = plane.intersect_segment(p0, p1)
    f1_ok, pi1 = plane.intersect_segment(p1, p2)
    f2_ok, pi2 = plane.intersect_segment(p2, p0)

    if f0_ok and f1_ok and not f2_ok:
        fi0 = bc.add_edge_point(pi0, id0, id1)
        fi1 = bc.add_edge_point(pi1, id1, id2)
        _split_two_intersections(bc, id0, id1, id2, fi0, fi1, s1,
                                 pos_tris, neg_tris, pos_used, neg_used,
                                 # isolated vertex is p1 (between the two cut edges)
                                 iso_orig=id1, other1=id0, other2=id2)
    elif f1_ok and f2_ok and not f0_ok:
        fi1 = bc.add_edge_point(pi1, id1, id2)
        fi2 = bc.add_edge_point(pi2, id2, id0)
        _split_two_intersections(bc, id1, id2, id0, fi1, fi2, s2,
                                 pos_tris, neg_tris, pos_used, neg_used,
                                 iso_orig=id2, other1=id1, other2=id0)
    elif f2_ok and f0_ok and not f1_ok:
        fi2 = bc.add_edge_point(pi2, id2, id0)
        fi0 = bc.add_edge_point(pi0, id0, id1)
        _split_two_intersections(bc, id2, id0, id1, fi2, fi0, s0,
                                 pos_tris, neg_tris, pos_used, neg_used,
                                 iso_orig=id0, other1=id2, other2=id1)
    elif f0_ok and f1_ok and f2_ok:
        _handle_three_intersections(bc, plane, verts, id0, id1, id2, p0, p1, p2,
                                    s0, s1, s2, pi0, pi1, pi2,
                                    pos_tris, neg_tris, pos_used, neg_used)
    else:
        # No valid intersections — degenerate, skip
        pass
    return True


def _encode(border_idx):
    """Encode a border point index as negative for triangle storage."""
    return -(border_idx + 1)


def _split_two_intersections(bc, idA, idB, idC, fiA, fiB, sB,
                             pos_tris, neg_tris, pos_used, neg_used,
                             iso_orig, other1, other2):
    """Split when two edges are cut. iso_orig is the isolated vertex (idB), sB is its side."""
    eA = _encode(fiA)
    eB = _encode(fiB)

    if fiA == fiB:
        # Degenerate: intersection points coincide
        if sB == 1:
            neg_used[other1] = neg_used[other2] = True
            neg_tris.append((eB, other2, other1))
        else:
            pos_used[other1] = pos_used[other2] = True
            pos_tris.append((eB, other2, other1))
        return

    if sB == 1:
        bc.add_border_edge(fiB, fiA)
        pos_used[iso_orig] = True
        neg_used[other1] = neg_used[other2] = True
        pos_tris.append((iso_orig, eB, eA))
        neg_tris.append((other1, eA, eB))
        neg_tris.append((eB, other2, other1))
    else:
        bc.add_border_edge(fiA, fiB)
        neg_used[iso_orig] = True
        pos_used[other1] = pos_used[other2] = True
        neg_tris.append((iso_orig, eB, eA))
        pos_tris.append((other1, eA, eB))
        pos_tris.append((eB, other2, other1))


def _handle_three_intersections(bc, plane, verts, id0, id1, id2, p0, p1, p2,
                                s0, s1, s2, pi0, pi1, pi2,
                                pos_tris, neg_tris, pos_used, neg_used):
    """Handle case where plane intersects all three edges (vertex on plane)."""
    if s0 == 0 or (s0 != 0 and s1 != 0 and s2 != 0 and _same_point(pi0, pi2)):
        # Intersection at p0
        bi0 = bc.add_vertex(p0, id0)
        bc.edge_map[(id0, id1)] = bi0
        bc.edge_map[(id1, id0)] = bi0
        bc.edge_map[(id2, id0)] = bi0
        bc.edge_map[(id0, id2)] = bi0
        fi1 = bc.add_edge_point(pi1, id1, id2)
        eI0 = _encode(bi0)
        eI1 = _encode(fi1)
        if bi0 != fi1:
            if s1 == 1:
                bc.add_border_edge(fi1, bi0)
                pos_used[id1] = True
                neg_used[id2] = True
                pos_tris.append((id1, eI1, eI0))
                neg_tris.append((id2, eI0, eI1))
            else:
                bc.add_border_edge(bi0, fi1)
                neg_used[id1] = True
                pos_used[id2] = True
                neg_tris.append((id1, eI1, eI0))
                pos_tris.append((id2, eI0, eI1))
    elif s1 == 0 or (s0 != 0 and s1 != 0 and s2 != 0 and _same_point(pi0, pi1)):
        # Intersection at p1
        bi1 = bc.add_vertex(p1, id1)
        bc.edge_map[(id0, id1)] = bi1
        bc.edge_map[(id1, id0)] = bi1
        bc.edge_map[(id1, id2)] = bi1
        bc.edge_map[(id2, id1)] = bi1
        fi2 = bc.add_edge_point(pi2, id2, id0)
        eI1 = _encode(bi1)
        eI2 = _encode(fi2)
        if bi1 != fi2:
            if s0 == 1:
                bc.add_border_edge(bi1, fi2)
                pos_used[id0] = True
                neg_used[id2] = True
                pos_tris.append((id0, eI1, eI2))
                neg_tris.append((id2, eI2, eI1))
            else:
                bc.add_border_edge(fi2, bi1)
                neg_used[id0] = True
                pos_used[id2] = True
                neg_tris.append((id0, eI1, eI2))
                pos_tris.append((id2, eI2, eI1))
    elif s2 == 0 or (s0 != 0 and s1 != 0 and s2 != 0 and _same_point(pi1, pi2)):
        # Intersection at p2
        bi2 = bc.add_vertex(p2, id2)
        bc.edge_map[(id1, id2)] = bi2
        bc.edge_map[(id2, id1)] = bi2
        bc.edge_map[(id2, id0)] = bi2
        bc.edge_map[(id0, id2)] = bi2
        fi0 = bc.add_edge_point(pi0, id0, id1)
        eI2 = _encode(bi2)
        eI0 = _encode(fi0)
        if fi0 != bi2:
            if s0 == 1:
                bc.add_border_edge(fi0, bi2)
                pos_used[id0] = True
                neg_used[id1] = True
                pos_tris.append((id0, eI0, eI2))
                neg_tris.append((id1, eI2, eI0))
            else:
                bc.add_border_edge(bi2, fi0)
                neg_used[id0] = True
                pos_used[id1] = True
                neg_tris.append((id0, eI0, eI2))
                pos_tris.append((id1, eI2, eI0))
    # else: degenerate, silently skip


def _triangulate_cap(bc, plane):
    """Triangulate the cap polygon using constrained Delaunay."""
    if len(bc.points) < 3 or len(bc.edges) < 3:
        if bc.points:
            return np.array(bc.points), []
        return None, []

    border_pts = np.array(bc.points)  # (N, 3)

    # Find 3 non-collinear points for projection basis
    idx0 = 0
    idx1 = None
    for i in range(1, len(border_pts)):
        if np.linalg.norm(border_pts[i] - border_pts[idx0]) > 0.01:
            idx1 = i
            break
    if idx1 is None:
        return border_pts, []

    idx2 = None
    ab = border_pts[idx1] - border_pts[idx0]
    ab_len = np.linalg.norm(ab)
    for i in range(2, len(border_pts)):
        if i == idx1:
            continue
        bc_vec = border_pts[i] - border_pts[idx1]
        bc_len = np.linalg.norm(bc_vec)
        if bc_len < 1e-10:
            continue
        cos_val = np.dot(ab, bc_vec) / (ab_len * bc_len)
        if abs(abs(cos_val) - 1) > 1e-6:
            idx2 = i
            break
    if idx2 is None:
        return border_pts, []

    # Build rotation matrix to project to 2D
    p0, p1_pt, p2_pt = border_pts[idx0], border_pts[idx1], border_pts[idx2]
    # Align cap normal with plane normal
    normal = np.cross(p1_pt - p0, p2_pt - p0)
    pn = np.array([plane.a, plane.b, plane.c])
    if np.dot(normal, pn) > 0:
        p0, p2_pt = p2_pt, p0

    e0 = p0 - p1_pt
    e0 = e0 / np.linalg.norm(e0)
    e2_raw = np.cross(p2_pt - p0, e0)
    e2 = e2_raw / np.linalg.norm(e2_raw)
    e1 = np.cross(e2, e0)
    e1 = e1 / np.linalg.norm(e1)

    T = p0
    R = np.array([e0, e1, e2])  # 3x3

    # Project to 2D
    centered = border_pts - T
    projected = centered @ R.T  # (N, 3) -> use first 2 cols
    pts_2d = projected[:, :2]

    # Build segment constraints (0-based)
    segments = np.array(bc.edges, dtype=np.int32)

    try:
        result = tr.triangulate({
            'vertices': pts_2d,
            'segments': segments,
        }, 'p')  # p = use PSLG (segments as constraints)
    except Exception:
        return border_pts, []

    cap_tris = result.get('triangles', np.empty((0, 3), dtype=np.int32))
    if len(cap_tris) == 0:
        return border_pts, []

    # Additional vertices from Steiner points
    new_verts_2d = result.get('vertices', pts_2d)
    if len(new_verts_2d) > len(border_pts):
        extra_2d = new_verts_2d[len(border_pts):]
        # Back-project to 3D
        extra_3d = np.zeros((len(extra_2d), 3))
        extra_3d[:, :2] = extra_2d
        extra_3d_world = extra_3d @ R + T
        border_pts = np.vstack([border_pts, extra_3d_world])

    # Filter cap triangles: keep only those reachable from border edges (BFS)
    cap_tris = _filter_cap_triangles(cap_tris, bc.edges, bc.overlap, border_pts)

    return border_pts, cap_tris


def _filter_cap_triangles(cap_tris, border_edges, overlap_pts, border_pts):
    """BFS from border edges to select interior cap triangles, removing outliers."""
    if len(cap_tris) == 0:
        return cap_tris

    # Build edge -> triangle adjacency
    edge_to_tris = {}
    for ti, tri in enumerate(cap_tris):
        for j in range(3):
            e = (int(tri[j]), int(tri[(j + 1) % 3]))
            e_rev = (e[1], e[0])
            if e not in edge_to_tris:
                edge_to_tris[e] = []
            edge_to_tris[e].append(ti)
            if e_rev not in edge_to_tris:
                edge_to_tris[e_rev] = []
            edge_to_tris[e_rev].append(ti)

    # Build overlap vertex set
    overlap_set = set()
    for op in overlap_pts:
        for bi, bp in enumerate(border_pts):
            if _same_point(op, bp):
                overlap_set.add(bi)

    # Build set of same-direction edge pairs (edges that appear in both directions)
    same_edge = set()
    edge_set = set()
    for e in border_edges:
        if (e[1], e[0]) in edge_set:
            same_edge.add(e)
            same_edge.add((e[1], e[0]))
        edge_set.add(e)

    # Border edges (non-same) form the BFS boundary
    border_set = set()
    for e in border_edges:
        if (e[1], e[0]) not in edge_set:
            border_set.add(e)
            border_set.add((e[1], e[0]))

    # BFS from border edges inward
    visited = set()
    result = []
    queue = list(border_edges)

    for e in queue:
        for adj_e in [e, (e[1], e[0])]:
            for ti in edge_to_tris.get(adj_e, []):
                if ti in visited:
                    continue
                tri = cap_tris[ti]
                # Skip if all vertices are overlap vertices
                if (int(tri[0]) in overlap_set and
                        int(tri[1]) in overlap_set and
                        int(tri[2]) in overlap_set):
                    visited.add(ti)
                    continue
                visited.add(ti)
                result.append(tri)
                # Add non-border edges to queue for continued BFS
                for j in range(3):
                    ne = (int(tri[j]), int(tri[(j + 1) % 3]))
                    if ne not in border_set and (ne[1], ne[0]) not in border_set:
                        queue.append(ne)

    if result:
        return np.array(result, dtype=np.int32)
    return np.empty((0, 3), dtype=np.int32)


def _build_output(orig_verts, tri_list, used_mask, border_verts, cap_tris, flip_cap):
    """Build output Mesh from collected triangles and border."""
    n_orig = len(orig_verts)
    n_border = len(border_verts)

    # Compact original vertices
    orig_indices = np.where(used_mask)[0]
    if len(orig_indices) == 0 and len(cap_tris) == 0 and len(tri_list) == 0:
        return None

    # Remap: old_index -> new_index for original verts
    remap = np.full(n_orig, -1, dtype=np.int32)
    remap[orig_indices] = np.arange(len(orig_indices), dtype=np.int32)

    out_verts_list = [orig_verts[orig_indices]]
    border_offset = len(orig_indices)

    if n_border > 0:
        out_verts_list.append(border_verts)

    if not out_verts_list:
        return None

    out_verts = np.vstack(out_verts_list) if out_verts_list else np.empty((0, 3))

    # Remap triangles
    out_tris = []
    for tri in tri_list:
        new_tri = [0, 0, 0]
        for k in range(3):
            v = tri[k]
            if v >= 0:
                new_tri[k] = int(remap[v])
            else:
                # Encoded border point
                border_idx = -(v + 1)
                new_tri[k] = border_offset + border_idx
        if -1 in new_tri:
            continue
        out_tris.append(new_tri)

    # Add cap triangles
    for tri in cap_tris:
        if flip_cap:
            out_tris.append([
                border_offset + int(tri[2]),
                border_offset + int(tri[1]),
                border_offset + int(tri[0]),
            ])
        else:
            out_tris.append([
                border_offset + int(tri[0]),
                border_offset + int(tri[1]),
                border_offset + int(tri[2]),
            ])

    if not out_tris:
        return None

    out_tris_arr = np.array(out_tris, dtype=np.int32)

    # Validate indices
    max_idx = len(out_verts) - 1
    if out_tris_arr.max() > max_idx or out_tris_arr.min() < 0:
        # Remove invalid triangles
        valid = np.all((out_tris_arr >= 0) & (out_tris_arr <= max_idx), axis=1)
        out_tris_arr = out_tris_arr[valid]
        if len(out_tris_arr) == 0:
            return None

    return Mesh(out_verts, out_tris_arr)
