"""
coacd_gpu — GPU-accelerated convex decomposition, Hausdorff distance, and merge cost.

Uses CUDA driver API via a native CPython extension. No PyTorch or CUDA runtime dependency.
Only requires an NVIDIA GPU driver (libcuda.so / nvcuda.dll).

Includes:
- Hausdorff distance computation (point_mesh_distances, hausdorff, pairwise_hausdorff)
- GPU beam search convex decomposition (BeamContext, run_beam_coacd)
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


from coacd_gpu.beam import BeamContext, run_beam_coacd
