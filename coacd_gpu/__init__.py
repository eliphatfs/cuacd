"""
coacd_gpu — GPU hull volume, mesh volume, and plane cut utilities.

Uses CUDA driver API via a native CPython extension. No PyTorch or CUDA runtime dependency.
Only requires an NVIDIA GPU driver (libcuda.so / nvcuda.dll).
"""

import numpy as np
from coacd_gpu import _gpu


def _as_f32(arr):
    return np.ascontiguousarray(arr, dtype=np.float32)

def _as_i32(arr):
    return np.ascontiguousarray(arr, dtype=np.int32)


def _mesh_volume_cpu(verts, tris):
    """Signed mesh volume via divergence theorem (signed-tetrahedra sum)."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1, v2)
    return abs(float(np.sum(v0 * cross))) / 6.0


class Context:
    """GPU context wrapping a CUDA device and loaded kernels.

    Usage::

        with coacd_gpu.Context() as ctx:
            vols, errs = ctx.batch_hull_volume(pts_list)
            vols = ctx.batch_mesh_volume(verts_list, tris_list)

    Parameters
    ----------
    device : int
        GPU device ordinal. -1 reuses an existing CUDA context if present.
    pool_bytes : int
        Bytes to reserve for the shared device heap pool (output + scratch).
        0 (default) → 70% of free device memory at init time.
    """

    def __init__(self, device=-1, pool_bytes=0):
        _gpu.init(device, pool_bytes)
        self._alive = True

    def close(self):
        if self._alive:
            _gpu.destroy()
            self._alive = False

    def heap_compact(self):
        """Compact both persistent heaps (output + scratch).

        Coalesces adjacent freed blocks so they can be reused for larger
        allocations. Call periodically when running many kernel launches
        to prevent fragmentation from exhausting the pool.
        """
        _gpu.heap_compact()

    def pool_usage(self):
        """Return bytes consumed from the shared device pool.

        This is the peak (high-water mark) of device memory allocated from
        the bump pool. It never decreases; freed blocks return to heap
        free-lists rather than back to the pool.
        """
        return _gpu.pool_usage()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # batch_hull_volume
    # ------------------------------------------------------------------

    def batch_hull_volume(self, pts_list, algo=2):
        """Compute convex hull volume for a batch of point clouds.

        Uses D&C hull extraction followed by divergence-theorem volume.
        The ``algo`` parameter is accepted for API compatibility but ignored
        (only D&C is supported).
        """
        n_hulls = len(pts_list)
        if n_hulls == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

        pts_arrays = [_as_f32(p) for p in pts_list]
        packed = np.concatenate([p.reshape(-1, 3) for p in pts_arrays], axis=0)
        offsets = np.zeros(n_hulls + 1, dtype=np.int32)
        for i, p in enumerate(pts_arrays):
            offsets[i + 1] = offsets[i] + len(p)

        ns = [len(p) for p in pts_arrays]
        max_pts = max(ns)
        max_hv = max_pts
        max_ht = max(max(2 * n - 4 for n in ns), 4)
        total_pts = int(offsets[-1])

        out_verts  = np.empty(n_hulls * max_hv * 3, dtype=np.float32)
        out_tris   = np.empty(n_hulls * max_ht * 3, dtype=np.int32)
        out_nv     = np.empty(n_hulls, dtype=np.int32)
        out_nt     = np.empty(n_hulls, dtype=np.int32)
        errors     = np.empty(n_hulls, dtype=np.int32)

        _gpu.hull_dandc(
            packed.ctypes.data, total_pts,
            offsets.ctypes.data, n_hulls,
            max_pts, max_hv, max_ht,
            out_verts.ctypes.data, out_tris.ctypes.data,
            out_nv.ctypes.data, out_nt.ctypes.data,
            errors.ctypes.data)

        volumes = np.empty(n_hulls, dtype=np.float32)
        for i in range(n_hulls):
            nv = int(out_nv[i])
            nt = int(out_nt[i])
            if nt > 0 and nv > 0:
                hv = out_verts[i * max_hv * 3 : i * max_hv * 3 + nv * 3].reshape(nv, 3)
                ht = out_tris [i * max_ht * 3 : i * max_ht * 3 + nt * 3].reshape(nt, 3)
                volumes[i] = _mesh_volume_cpu(hv, ht)
            else:
                volumes[i] = 0.0

        return volumes, errors

    # ------------------------------------------------------------------
    # batch_mesh_volume
    # ------------------------------------------------------------------

    def batch_mesh_volume(self, verts_list, tris_list):
        """Compute mesh volume for a batch of watertight meshes."""
        n_meshes = len(verts_list)
        v_arrays = [_as_f32(v) for v in verts_list]
        t_arrays = [_as_i32(t) for t in tris_list]

        all_verts = np.concatenate([v.reshape(-1, 3) for v in v_arrays], axis=0)
        all_tris  = np.concatenate([t.reshape(-1, 3) for t in t_arrays], axis=0)

        v_off = np.zeros(n_meshes + 1, dtype=np.int32)
        t_off = np.zeros(n_meshes + 1, dtype=np.int32)
        for i in range(n_meshes):
            v_off[i + 1] = v_off[i] + len(v_arrays[i])
            t_off[i + 1] = t_off[i] + len(t_arrays[i])

        volumes = np.empty(n_meshes, dtype=np.float32)
        _gpu.batch_mesh_volume(
            all_verts.ctypes.data, int(v_off[-1]),
            all_tris.ctypes.data,  int(t_off[-1]),
            t_off.ctypes.data,
            v_off.ctypes.data,
            n_meshes, volumes.ctypes.data)
        return volumes

    # ------------------------------------------------------------------
    # batch_hull_dandc_mesh — hull mesh extraction
    # ------------------------------------------------------------------

    def batch_kdop_hull_mesh(self, pts_list):
        """Compute convex hull mesh for a batch of point clouds.

        For meshes with ≤1024 vertices, runs the exact D&C hull directly.
        For larger meshes, finds 80 extreme vertices (40 icosphere axes ×
        max/min), builds a rough inner hull, filters interior points via
        half-space test, then runs the exact D&C hull on the survivor set.

        Returns list of (hull_verts, hull_tris, hull_volume) per input.
        """
        n_hulls = len(pts_list)
        if n_hulls == 0:
            return []
        pts_arrays = [_as_f32(p) for p in pts_list]
        packed = np.concatenate([p.reshape(-1, 3) for p in pts_arrays], axis=0)
        offsets = np.zeros(n_hulls + 1, dtype=np.int32)
        for i, p in enumerate(pts_arrays):
            offsets[i + 1] = offsets[i] + len(p)

        # Output bounds: the final hull is an exact D&C hull of the filtered
        # point set. In the worst case the hull has O(n) verts/tris, but we
        # bound by a generous fixed cap sufficient for typical meshes.
        max_hv = 4096
        max_ht = 8192
        total_pts = int(offsets[-1])

        out_verts   = np.empty(n_hulls * max_hv * 3, dtype=np.float32)
        out_tris    = np.empty(n_hulls * max_ht * 3, dtype=np.int32)
        out_nv      = np.empty(n_hulls, dtype=np.int32)
        out_nt      = np.empty(n_hulls, dtype=np.int32)
        out_volumes = np.empty(n_hulls, dtype=np.float32)
        errors      = np.empty(n_hulls, dtype=np.int32)

        _gpu.kdop_hull(
            packed.ctypes.data, total_pts,
            offsets.ctypes.data, n_hulls,
            max_hv, max_ht,
            out_verts.ctypes.data, out_tris.ctypes.data,
            out_nv.ctypes.data, out_nt.ctypes.data,
            out_volumes.ctypes.data, errors.ctypes.data)

        results = []
        for i in range(n_hulls):
            nv = int(out_nv[i])
            nt = int(out_nt[i])
            hv = out_verts[i * max_hv * 3 : i * max_hv * 3 + nv * 3].reshape(nv, 3).copy()
            ht = out_tris [i * max_ht * 3 : i * max_ht * 3 + nt * 3].reshape(nt, 3).copy()
            vol = float(out_volumes[i]) if nt > 0 else 0.0
            results.append((hv, ht, vol))
        return results

    def batch_hull_dandc_mesh(self, pts_list):
        """Compute D&C hull mesh for a batch of point clouds.

        Returns list of (hull_verts, hull_tris, hull_volume) per input.
        Natural bounds: max_hull_verts = n_pts, max_hull_tris = 2*n_pts - 4.
        """
        n_hulls = len(pts_list)
        if n_hulls == 0:
            return []
        pts_arrays = [_as_f32(p) for p in pts_list]
        packed = np.concatenate([p.reshape(-1, 3) for p in pts_arrays], axis=0)
        offsets = np.zeros(n_hulls + 1, dtype=np.int32)
        for i, p in enumerate(pts_arrays):
            offsets[i + 1] = offsets[i] + len(p)

        ns = [len(p) for p in pts_arrays]
        max_pts = max(ns)
        max_hv = max_pts
        max_ht = max(max(2 * n - 4 for n in ns), 4)
        total_pts = int(offsets[-1])

        out_verts  = np.empty(n_hulls * max_hv * 3, dtype=np.float32)
        out_tris   = np.empty(n_hulls * max_ht * 3, dtype=np.int32)
        out_nv     = np.empty(n_hulls, dtype=np.int32)
        out_nt     = np.empty(n_hulls, dtype=np.int32)
        errors     = np.empty(n_hulls, dtype=np.int32)

        _gpu.hull_dandc(
            packed.ctypes.data, total_pts,
            offsets.ctypes.data, n_hulls,
            max_pts, max_hv, max_ht,
            out_verts.ctypes.data, out_tris.ctypes.data,
            out_nv.ctypes.data, out_nt.ctypes.data,
            errors.ctypes.data)

        results = []
        for i in range(n_hulls):
            nv = int(out_nv[i])
            nt = int(out_nt[i])
            hv = out_verts[i * max_hv * 3 : i * max_hv * 3 + nv * 3].reshape(nv, 3).copy()
            ht = out_tris [i * max_ht * 3 : i * max_ht * 3 + nt * 3].reshape(nt, 3).copy()
            vol = _mesh_volume_cpu(hv, ht) if nt > 0 else 0.0
            results.append((hv, ht, float(vol)))
        return results

    # ------------------------------------------------------------------
    # lookahead_decompose
    # ------------------------------------------------------------------

    def lookahead_decompose(self, verts, tris, *,
                            max_iters=100, width=60, width2=5, threshold=0.05,
                            depth=2, quick_depth=0, max_n_cutting=16,
                            verbose=0, debug=0, decompose_components=False,
                            no_decompose_components_per_iter=False,
                            n_concave_edges=32, concave_eps=0.005,
                            concave_threshold=3.49, concave_iters=10):
        """Decompose a mesh into convex parts using lookahead tree search.

        Parameters
        ----------
        verts : array_like, shape (N, 3), float32
            Mesh vertices.
        tris : array_like, shape (M, 3), int32
            Triangle indices.
        max_iters : int
            Maximum outer iterations.
        width : int
            Number of candidate cuts per expansion level.
        threshold : float
            Stop when all parts have cost below this value.
        depth : int
            Number of full expansion levels in the lookahead tree.
        quick_depth : int
            Number of quick expansion levels (1 child per item, best-axis midpoint).
        max_n_cutting : int
            Maximum parts processed in parallel per iteration.
        verbose : int
            Print timing info if nonzero.

        Returns
        -------
        list of (verts, tris, hull_verts, hull_tris) tuples, one per output part.
        """
        import scipy.spatial

        verts = _as_f32(verts).reshape(-1, 3)
        tris = _as_i32(tris).reshape(-1, 3)

        # Compute convex hull for the input mesh
        hull = scipy.spatial.ConvexHull(verts)
        hull_verts = _as_f32(hull.points)
        hull_tris = _as_i32(hull.simplices)

        # Reorient hull triangles so normals point outward (away from centroid).
        # scipy's ConvexHull does not guarantee consistent winding.
        centroid = hull_verts.mean(axis=0)
        v0 = hull_verts[hull_tris[:, 0]]
        v1 = hull_verts[hull_tris[:, 1]]
        v2 = hull_verts[hull_tris[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        inward = (normals * (v0 - centroid)).sum(axis=1) < 0
        hull_tris[inward] = hull_tris[inward][:, [0, 2, 1]]

        if width2 is None:
            width2 = width
        total_width = width + 4 * n_concave_edges
        if total_width >= 512:
            raise ValueError(f"width + 4*n_concave_edges must be < 512 (got {total_width}); "
                             f"la_evaluate uses a fixed shared buffer of 512 entries")
        raw = _gpu.lookahead_decompose(
            verts.ctypes.data, len(verts),
            tris.ctypes.data, len(tris),
            hull_verts.ctypes.data, len(hull_verts),
            hull_tris.ctypes.data, len(hull_tris),
            max_iters, width, width2, threshold,
            depth, quick_depth, max_n_cutting,
            verbose, debug, decompose_components=int(decompose_components),
            no_decompose_components_per_iter=int(no_decompose_components_per_iter),
            n_concave_edges=n_concave_edges,
            concave_eps=concave_eps,
            concave_threshold=concave_threshold,
            concave_iters=concave_iters)

        results = []
        for vb, tb, nv, nt, hvb, htb, hnv, hnt, mv, hv in raw:
            v = np.frombuffer(vb, dtype=np.float32).reshape(nv, 3).copy()
            t = np.frombuffer(tb, dtype=np.int32).reshape(nt, 3).copy()
            hull_v = np.frombuffer(hvb, dtype=np.float32).reshape(hnv, 3).copy()
            hull_t = np.frombuffer(htb, dtype=np.int32).reshape(hnt, 3).copy()
            results.append((v, t, hull_v, hull_t))
        return results
