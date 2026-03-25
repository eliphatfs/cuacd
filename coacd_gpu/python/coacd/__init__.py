"""Pure Python CoACD — approximate convex decomposition with GPU acceleration.

Usage::

    from coacd_gpu.coacd import run_coacd

    parts = run_coacd(vertices, triangles, threshold=0.05)
    # parts is a list of (vertices, triangles) tuples
"""

from ._pipeline import run_coacd
from ._mesh import Mesh

__all__ = ["run_coacd", "Mesh"]
