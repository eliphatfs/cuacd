"""
GPU Beam Search Convex Decomposition.

High-level Python API wrapping the native _beam CPython extension.
"""

import numpy as np
from coacd_gpu import _gpu


class BeamContext:
    """GPU context for beam search convex decomposition.

    Usage::

        with BeamContext(device=0) as ctx:
            parts = ctx.run(vertices, triangles, threshold=0.05)

    Or::

        ctx = BeamContext()
        parts = ctx.run(vertices, triangles)
        ctx.close()
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

    def run(self, vertices, triangles, *,
            beam_width=16,
            cuts_per_axis=15,
            threshold=0.05,
            rv_k=0.3,
            max_parts=64,
            max_iterations=64,
            hausdorff_samples=1000):
        """Run beam search convex decomposition.

        Args:
            vertices:    (V, 3) float array -- mesh vertices
            triangles:   (T, 3) int array -- mesh face indices
            beam_width:  Number of beam items to maintain (default 8)
            cuts_per_axis: Planes per axis (30 total by default)
            threshold:   Concavity threshold (default 0.05)
            rv_k:        Rv scaling factor (default 0.3)
            max_parts:   Maximum parts per beam item (default 64)
            max_iterations: Max decomposition steps (default 64)
            hausdorff_samples: Samples for Hausdorff check (default 1000)

        Returns:
            List of (vertices, triangles) tuples -- each part's mesh.
        """
        verts = np.ascontiguousarray(vertices, dtype=np.float32)
        tris = np.ascontiguousarray(triangles, dtype=np.int32)

        num_parts = _gpu.run(
            verts.ctypes.data, len(verts),
            tris.ctypes.data, len(tris),
            beam_width, cuts_per_axis,
            threshold, rv_k,
            max_parts, max_iterations, hausdorff_samples)

        parts = []
        for i in range(num_parts):
            nv, nt = _gpu.get_part_sizes(i)
            if nv <= 0 or nt <= 0:
                continue

            out_v = np.empty((nv, 3), dtype=np.float32)
            out_t = np.empty((nt, 3), dtype=np.int32)
            actual_nv, actual_nt = _gpu.get_part(
                i, out_v.ctypes.data, nv, out_t.ctypes.data, nt)
            parts.append((out_v[:actual_nv], out_t[:actual_nt]))

        return parts


def run_beam_coacd(vertices, triangles, **kwargs):
    """Convenience function: run GPU beam search decomposition.

    Args:
        vertices:  (V, 3) float array -- mesh vertices
        triangles: (T, 3) int array -- mesh face indices
        **kwargs:  Passed to BeamContext.run()

    Returns:
        List of (vertices, triangles) tuples.
    """
    device = kwargs.pop('device', -1)
    with BeamContext(device=device) as ctx:
        return ctx.run(vertices, triangles, **kwargs)
