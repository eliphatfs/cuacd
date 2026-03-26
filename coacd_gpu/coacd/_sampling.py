"""Area-weighted surface sampling with mixed random/quasi-random strategy."""

import numpy as np

from ._geometry import triangle_areas, Plane


def sample_surface(vertices: np.ndarray, triangles: np.ndarray,
                   resolution: int = 2000, rng=None,
                   base_area: float = 0.0,
                   exclude_plane: Plane = None):
    """Sample points on mesh surface.

    Returns (samples (K,3), sample_tri_ids (K,)) arrays.
    """
    if resolution == 0 or len(triangles) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int32)

    if rng is None:
        rng = np.random.RandomState(1234)

    areas = triangle_areas(vertices, triangles)
    total_area = areas.sum()

    if total_area < 1e-30:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int32)

    if base_area > 0:
        resolution = max(1000, int(resolution * (total_area / base_area)))

    n_tris = len(triangles)

    # Pre-compute per-triangle sample counts
    counts = np.zeros(n_tris, dtype=np.int32)
    for i in range(n_tris):
        # Optionally skip coplanar triangles
        if exclude_plane is not None:
            p0 = vertices[triangles[i, 0]]
            p1 = vertices[triangles[i, 1]]
            p2 = vertices[triangles[i, 2]]
            if (exclude_plane.side_one(p0, 1e-3) == 0 and
                    exclude_plane.side_one(p1, 1e-3) == 0 and
                    exclude_plane.side_one(p2, 1e-3) == 0):
                continue

        area = areas[i]
        if n_tris > resolution and resolution > 0:
            counts[i] = max(
                int(i % (n_tris // resolution) == 0),
                int(resolution / total_area * area)
            )
        else:
            counts[i] = max(int(i % 2 == 0), int(resolution / total_area * area))

    total_samples = int(counts.sum())
    if total_samples == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int32)

    samples = np.empty((total_samples, 3), dtype=np.float64)
    tri_ids = np.empty(total_samples, dtype=np.int32)

    idx = 0
    # Sobol-like quasi-random: use golden ratio based low-discrepancy
    # (approximating the C++ i4_sobol for 2D)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    sobol_idx = 0

    for i in range(n_tris):
        n = counts[i]
        if n == 0:
            continue
        p0 = vertices[triangles[i, 0]]
        p1 = vertices[triangles[i, 1]]
        p2 = vertices[triangles[i, 2]]

        for k in range(n):
            if k % 3 == 0:
                a = rng.uniform(0.0, 1.0)
                b = rng.uniform(0.0, 1.0)
            else:
                sobol_idx += 1
                a = (sobol_idx * phi) % 1.0
                b = (sobol_idx * phi * phi) % 1.0

            sqrt_a = np.sqrt(a)
            w0 = 1.0 - sqrt_a
            w1 = sqrt_a * (1.0 - b)
            w2 = sqrt_a * b

            samples[idx] = w0 * p0 + w1 * p1 + w2 * p2
            tri_ids[idx] = i
            idx += 1

    return samples[:idx], tri_ids[:idx]
