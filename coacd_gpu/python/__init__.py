"""
coacd_gpu — GPU-accelerated convex decomposition, Hausdorff distance, and merge cost.

Uses CUDA driver API via ctypes. No PyTorch or CUDA runtime dependency.
Only requires an NVIDIA GPU driver (libcuda.so / nvcuda.dll).

Includes:
- Hausdorff distance computation (point_mesh_distances, hausdorff, pairwise_hausdorff)
- GPU beam search convex decomposition (BeamContext, run_beam_coacd)
"""

import ctypes
import os
import sys
import numpy as np
from ctypes import c_int, c_float, c_void_p, POINTER, byref

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------

def _find_lib():
    """Locate the coacd_gpu shared library next to this file."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    # Exact names first (CMake output), then fallback to setuptools-generated names
    if sys.platform == "win32":
        candidates = ["coacd_gpu.dll", "libcoacd_gpu.dll"]
        fallback_ext = ".pyd"
    elif sys.platform == "darwin":
        candidates = ["libcoacd_gpu.dylib"]
        fallback_ext = ".so"
    else:
        candidates = ["libcoacd_gpu.so"]
        fallback_ext = ".so"

    for name in candidates:
        path = os.path.join(pkg_dir, name)
        if os.path.isfile(path):
            return path

    # setuptools editable installs may name it _native.abi3.so or similar
    for entry in os.listdir(pkg_dir):
        if entry.startswith("_native") and (entry.endswith(fallback_ext) or ".abi3." in entry):
            return os.path.join(pkg_dir, entry)

    raise RuntimeError(
        f"Cannot find coacd_gpu shared library in {pkg_dir}. "
        "Make sure the package was built with `pip install .`"
    )


_lib = ctypes.CDLL(_find_lib())

# ---------------------------------------------------------------------------
# C API bindings
# ---------------------------------------------------------------------------

_ctx_t = c_void_p  # opaque handle

_lib.coacd_gpu_init.argtypes = [POINTER(_ctx_t), c_int]
_lib.coacd_gpu_init.restype = c_int

_lib.coacd_gpu_destroy.argtypes = [_ctx_t]
_lib.coacd_gpu_destroy.restype = None

_lib.coacd_gpu_last_error.argtypes = [_ctx_t]
_lib.coacd_gpu_last_error.restype = ctypes.c_char_p

_lib.coacd_gpu_point_mesh_distances.argtypes = [
    _ctx_t,
    POINTER(c_float), c_int,   # points, n_points
    POINTER(c_float), c_int,   # vertices, n_verts
    POINTER(c_int),   c_int,   # triangles, n_tris
    POINTER(c_float),          # distances (output)
    c_void_p,                  # stream
]
_lib.coacd_gpu_point_mesh_distances.restype = c_int

_lib.coacd_gpu_hausdorff.argtypes = [
    _ctx_t,
    POINTER(c_float), c_int,   # samples_a, n_sa
    POINTER(c_float), c_int,   # vertices_a, n_va
    POINTER(c_int),   c_int,   # triangles_a, n_ta
    POINTER(c_float), c_int,   # samples_b, n_sb
    POINTER(c_float), c_int,   # vertices_b, n_vb
    POINTER(c_int),   c_int,   # triangles_b, n_tb
    POINTER(c_float),          # result (output)
    c_void_p,                  # stream
]
_lib.coacd_gpu_hausdorff.restype = c_int

_lib.coacd_gpu_pairwise_hausdorff.argtypes = [
    _ctx_t,
    POINTER(c_float),          # all_samples
    POINTER(c_int),            # sample_offsets
    POINTER(c_float),          # all_vertices
    POINTER(c_int),            # all_triangles
    POINTER(c_int),            # tri_offsets
    POINTER(c_int),            # vert_offsets
    c_int,                     # n_parts
    POINTER(c_float),          # cost_matrix (output)
    c_void_p,                  # stream
]
_lib.coacd_gpu_pairwise_hausdorff.restype = c_int


def _as_f32(arr):
    return np.ascontiguousarray(arr, dtype=np.float32)

def _as_i32(arr):
    return np.ascontiguousarray(arr, dtype=np.int32)

def _fptr(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def _iptr(arr):
    return arr.ctypes.data_as(POINTER(c_int))

def _check(ctx, rc):
    if rc != 0:
        msg = _lib.coacd_gpu_last_error(ctx)
        raise RuntimeError(
            f"coacd_gpu error {rc}: {msg.decode() if msg else 'unknown'}")


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------

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
        """Initialize GPU context.

        Args:
            device: GPU ordinal, or -1 to reuse current CUDA context.
        """
        self._handle = _ctx_t()
        rc = _lib.coacd_gpu_init(byref(self._handle), c_int(device))
        if rc != 0:
            msg = _lib.coacd_gpu_last_error(self._handle)
            raise RuntimeError(
                f"Failed to initialize coacd_gpu (error {rc}): "
                f"{msg.decode() if msg else 'no CUDA-capable GPU?'}")

    def close(self):
        if self._handle:
            _lib.coacd_gpu_destroy(self._handle)
            self._handle = None

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
        n_points = len(points)

        distances = np.empty(n_points, dtype=np.float32)
        rc = _lib.coacd_gpu_point_mesh_distances(
            self._handle,
            _fptr(points), n_points,
            _fptr(vertices), len(vertices),
            _iptr(triangles), len(triangles),
            _fptr(distances),
            None)
        _check(self._handle, rc)
        return distances

    def hausdorff(self, samples_a, vertices_a, triangles_a,
                        samples_b, vertices_b, triangles_b):
        """Compute Hausdorff distance between two sampled meshes.

        Returns:
            float — the Hausdorff distance.
        """
        sa = _as_f32(samples_a);  va = _as_f32(vertices_a);  ta = _as_i32(triangles_a)
        sb = _as_f32(samples_b);  vb = _as_f32(vertices_b);  tb = _as_i32(triangles_b)

        result = c_float(0.0)
        rc = _lib.coacd_gpu_hausdorff(
            self._handle,
            _fptr(sa), len(sa), _fptr(va), len(va), _iptr(ta), len(ta),
            _fptr(sb), len(sb), _fptr(vb), len(vb), _iptr(tb), len(tb),
            byref(result), None)
        _check(self._handle, rc)
        return result.value

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

        rc = _lib.coacd_gpu_pairwise_hausdorff(
            self._handle,
            _fptr(samples), _iptr(s_off),
            _fptr(verts), _iptr(tris),
            _iptr(t_off), _iptr(v_off),
            n_parts, _fptr(cost), None)
        _check(self._handle, rc)
        return cost


# ---------------------------------------------------------------------------
# Beam search API
# ---------------------------------------------------------------------------

from coacd_gpu.beam import BeamContext, run_beam_coacd
