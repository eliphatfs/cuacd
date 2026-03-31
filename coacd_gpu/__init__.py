"""
coacd_gpu — GPU-accelerated convex decomposition, Hausdorff distance, and merge cost.

Uses CUDA driver API via a native CPython extension. No PyTorch or CUDA runtime dependency.
Only requires an NVIDIA GPU driver (libcuda.so / nvcuda.dll).

Includes:
- Hausdorff distance computation (point_mesh_distances, hausdorff, pairwise_hausdorff)
- GPU beam search convex decomposition V2 (BeamContext, run_beam_coacd_v2)
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

        ctx = coacd_gpu.Context(device=0)
        dists = ctx.point_mesh_distances(points, vertices, triangles)
        h = ctx.hausdorff(sa, va, ta, sb, vb, tb)
        ctx.close()

    Also usable as a context manager::

        with coacd_gpu.Context() as ctx:
            ...
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

    def point_mesh_distances(self, points, vertices, triangles):
        """Compute min distance from each point to a triangle mesh.

        Args:
            points:    (N, 3) float array — query points
            vertices:  (V, 3) float array — mesh vertices
            triangles: (T, 3) int array   — mesh face indices

        Returns:
            (N,) float array of distances.
        """
        points = _as_f32(points)
        vertices = _as_f32(vertices)
        triangles = _as_i32(triangles)
        distances = np.empty(len(points), dtype=np.float32)

        _gpu.point_mesh_distances(
            points.ctypes.data, len(points),
            vertices.ctypes.data, len(vertices),
            triangles.ctypes.data, len(triangles),
            distances.ctypes.data)
        return distances

    def hausdorff(self, samples_a, vertices_a, triangles_a,
                        samples_b, vertices_b, triangles_b):
        """Compute Hausdorff distance between two sampled meshes.

        Returns:
            float — the Hausdorff distance.
        """
        sa = _as_f32(samples_a);  va = _as_f32(vertices_a);  ta = _as_i32(triangles_a)
        sb = _as_f32(samples_b);  vb = _as_f32(vertices_b);  tb = _as_i32(triangles_b)

        return _gpu.hausdorff(
            sa.ctypes.data, len(sa), va.ctypes.data, len(va), ta.ctypes.data, len(ta),
            sb.ctypes.data, len(sb), vb.ctypes.data, len(vb), tb.ctypes.data, len(tb))

    def pairwise_hausdorff(self, all_samples, sample_offsets,
                                 all_vertices, all_triangles,
                                 tri_offsets, vert_offsets):
        """Compute pairwise Hausdorff cost matrix for merge phase.

        Args:
            all_samples:    (total_samples, 3) float — packed sample points
            sample_offsets: (n_parts+1,) int — prefix sums
            all_vertices:   (total_verts, 3) float — packed vertices
            all_triangles:  (total_tris, 3) int — packed triangles (local idx)
            tri_offsets:    (n_parts+1,) int — prefix sums
            vert_offsets:   (n_parts+1,) int — prefix sums

        Returns:
            (n_parts, n_parts) float array — lower triangle filled.
        """
        samples = _as_f32(all_samples)
        s_off = _as_i32(sample_offsets)
        verts = _as_f32(all_vertices)
        tris = _as_i32(all_triangles)
        t_off = _as_i32(tri_offsets)
        v_off = _as_i32(vert_offsets)

        n_parts = len(s_off) - 1
        cost = np.zeros((n_parts, n_parts), dtype=np.float32)

        _gpu.pairwise_hausdorff(
            samples.ctypes.data, s_off.ctypes.data,
            verts.ctypes.data, tris.ctypes.data,
            t_off.ctypes.data, v_off.ctypes.data,
            n_parts, cost.ctypes.data)
        return cost


    def batch_hull_volume(self, pts_list, algo=2):
        """Compute convex hull volume for a batch of point clouds.

        Args:
            pts_list: list of (N_i, 3) float32 arrays — one per hull
            algo:     2=dandc (D&C Preparata-Hong; only supported algorithm)

        Returns:
            (volumes, errors) — float32[n_hulls], int32[n_hulls]
            errors: 0=ok  1=OOM  2=sort_stack_overflow  3=dfs_stack_overflow  4=dc_stack_overflow  5=pool_exhaust
        """
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

    def batch_mesh_volume(self, verts_list, tris_list):
        """Compute mesh volume for a batch of watertight meshes.

        Args:
            verts_list: list of (V_i, 3) float32 vertex arrays
            tris_list:  list of (T_i, 3) int32 triangle arrays (0-based per mesh)

        Returns:
            float32[n_meshes] volumes
        """
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


from coacd_gpu.beam import BeamContext, run_beam_coacd_v2
