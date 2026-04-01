"""
coacd_gpu — GPU-accelerated convex decomposition.

Uses CUDA driver API via a native CPython extension. No PyTorch or CUDA runtime dependency.
Only requires an NVIDIA GPU driver (libcuda.so / nvcuda.dll).
"""

import numpy as np
from coacd_gpu import _gpu


def _as_f32(arr):
    return np.ascontiguousarray(arr, dtype=np.float32)

def _as_i32(arr):
    return np.ascontiguousarray(arr, dtype=np.int32)


class Context:
    """GPU context wrapping a CUDA device and loaded kernels.

    Usage::

        with coacd_gpu.Context() as ctx:
            vols, errs = ctx.batch_hull_volume(pts_list)
            parts = ctx.decompose(verts, tris, threshold=0.05)
    """

    def __init__(self, device=-1):
        _gpu.init(device)
        self._alive = True

    def close(self):
        if self._alive:
            _gpu.destroy()
            self._alive = False

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
        """Compute convex hull volume for a batch of point clouds."""
        n_hulls = len(pts_list)
        pts_arrays = [_as_f32(p) for p in pts_list]
        packed = np.concatenate([p.reshape(-1, 3) for p in pts_arrays], axis=0)
        offsets = np.zeros(n_hulls + 1, dtype=np.int32)
        for i, p in enumerate(pts_arrays):
            offsets[i + 1] = offsets[i] + len(p)

        max_pts = int(max(len(p) for p in pts_arrays)) if n_hulls else 1
        total_pts = int(offsets[-1])
        volumes = np.empty(n_hulls, dtype=np.float32)
        errors  = np.empty(n_hulls, dtype=np.int32)

        _gpu.batch_hull_volume(
            packed.ctypes.data, total_pts,
            offsets.ctypes.data, n_hulls,
            algo, max_pts,
            volumes.ctypes.data, errors.ctypes.data)
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
    # batch_hull_dandc_mesh — hull volume + mesh extraction
    # ------------------------------------------------------------------

    def batch_hull_dandc_mesh(self, pts_list):
        """Compute D&C hull volume + mesh for a batch of point clouds.

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

        volumes    = np.empty(n_hulls, dtype=np.float32)
        errors     = np.empty(n_hulls, dtype=np.int32)
        out_verts  = np.empty(n_hulls * max_hv * 3, dtype=np.float32)
        out_tris   = np.empty(n_hulls * max_ht * 3, dtype=np.int32)
        out_vc     = np.empty(n_hulls, dtype=np.int32)
        out_tc     = np.empty(n_hulls, dtype=np.int32)

        _gpu.batch_hull_dandc_mesh(
            packed.ctypes.data, total_pts,
            offsets.ctypes.data, n_hulls,
            max_pts, max_hv, max_ht,
            volumes.ctypes.data, errors.ctypes.data,
            out_verts.ctypes.data, out_tris.ctypes.data,
            out_vc.ctypes.data, out_tc.ctypes.data)

        results = []
        for i in range(n_hulls):
            nv = int(out_vc[i])
            nt = int(out_tc[i])
            hv = out_verts[i * max_hv * 3 : i * max_hv * 3 + nv * 3].reshape(nv, 3).copy()
            ht = out_tris [i * max_ht * 3 : i * max_ht * 3 + nt * 3].reshape(nt, 3).copy()
            results.append((hv, ht, float(volumes[i])))
        return results

    # ------------------------------------------------------------------
    # decompose — beam-search convex decomposition
    # ------------------------------------------------------------------

    def decompose(self, verts, tris, threshold=0.05,
                  beam_width=30, cuts_per_axis=10, max_iters=64):
        """Beam-search approximate convex decomposition (GPU-resident).

        Args:
            verts:         (N, 3) float32 vertex array
            tris:          (T, 3) int32 triangle array (0-based)
            threshold:     Rv concavity threshold (hull_vol - mesh_vol) / hull_vol
            beam_width:    Maximum beam items kept per iteration (default 30)
            cuts_per_axis: Candidate planes per axis (default 10)
            max_iters:     Maximum iterations

        Returns:
            list of (hull_verts, hull_tris) for each decomposed part
        """
        verts = _as_f32(verts).reshape(-1, 3)
        tris  = _as_i32(tris).reshape(-1, 3)

        n_parts = _gpu.beam_decompose(
            verts.ctypes.data, len(verts),
            tris.ctypes.data,  len(tris),
            float(threshold), int(beam_width),
            int(cuts_per_axis), int(max_iters))

        results = []
        for i in range(n_parts):
            nv, nt = _gpu.get_part_sizes(i)
            if nv < 4 or nt < 4:
                continue
            hv = np.empty(nv * 3, dtype=np.float32)
            ht = np.empty(nt * 3, dtype=np.int32)
            _gpu.get_part(i, hv.ctypes.data, nv, ht.ctypes.data, nt)
            results.append((hv.reshape(nv, 3), ht.reshape(nt, 3)))
        return results
