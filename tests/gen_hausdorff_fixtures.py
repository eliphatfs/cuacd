#!/usr/bin/env python3
"""Generate Hausdorff distance comparison fixtures.

Creates mesh pairs (non-convex mesh + convex hull), computes CoACD reference
Hausdorff distances via a compiled C++ harness, and saves everything as an
.npz fixture file for use in test_hausdorff.py.

Usage:
    python tests/gen_hausdorff_fixtures.py
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Mesh generators (adapted from tests/test_hull.py)
# ---------------------------------------------------------------------------

def subdivide_icosphere(verts, faces):
    edge_map = {}
    new_verts = list(verts)
    new_faces = []

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_map:
            mid = (verts[a] + verts[b]) * 0.5
            mid = mid / np.linalg.norm(mid)
            edge_map[key] = len(new_verts)
            new_verts.append(mid)
        return edge_map[key]

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def make_icosphere(level):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    raw /= np.linalg.norm(raw[0])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    verts = raw
    for _ in range(level):
        verts, faces = subdivide_icosphere(verts, faces)
    return verts, faces


def make_noisy_icosphere(level, amplitude, seed):
    verts, faces = make_icosphere(level)
    rng = np.random.default_rng(seed)
    n_basis = 16
    dirs = rng.standard_normal((n_basis, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    coeffs = rng.standard_normal(n_basis)
    proj = verts @ dirs.T
    noise = np.zeros(len(verts))
    for i in range(n_basis):
        noise += coeffs[i] * np.sin(2.0 * np.pi * proj[:, i])
    nmax = np.abs(noise).max()
    if nmax > 1e-12:
        noise /= nmax
    normals = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    displaced = verts + amplitude * noise[:, None] * normals
    return displaced, faces


def convex_hull_mesh(verts):
    """Compute convex hull of a point set, return (hull_verts, hull_tris)."""
    hull = ConvexHull(verts)
    # Re-index: hull.vertices are indices into original verts
    hull_verts = verts[hull.vertices]
    vertex_map = {old: new for new, old in enumerate(hull.vertices)}
    hull_tris = np.array(
        [[vertex_map[s[0]], vertex_map[s[1]], vertex_map[s[2]]]
         for s in hull.simplices],
        dtype=np.int32)
    return hull_verts, hull_tris


# ---------------------------------------------------------------------------
# Binary I/O
# ---------------------------------------------------------------------------

def write_mesh_pair(path, va, ta, vb, tb):
    """Write mesh pair in binary format matching ref_hausdorff.cpp."""
    va = np.ascontiguousarray(va, dtype=np.float64)
    ta = np.ascontiguousarray(ta, dtype=np.int32)
    vb = np.ascontiguousarray(vb, dtype=np.float64)
    tb = np.ascontiguousarray(tb, dtype=np.int32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiii', len(va), len(ta), len(vb), len(tb)))
        f.write(va.tobytes())
        f.write(ta.tobytes())
        f.write(vb.tobytes())
        f.write(tb.tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'tests', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # --- Generate mesh pairs ---
    # Each pair is (noisy icosphere, its convex hull).
    # Noisy icospheres: closed meshes with Perlin-like normal displacement.
    pairs = []
    names = []

    for level, amp, seed in [
        (2, 0.20, 42),
        (2, 0.30, 100),
        (3, 0.15, 99),
        (3, 0.25, 200),
        (4, 0.10, 777),
        (4, 0.20, 300),
    ]:
        name = f"icosphere_L{level}_a{amp}_s{seed}"
        v, t = make_noisy_icosphere(level, amp, seed)
        hv, ht = convex_hull_mesh(v)
        pairs.append((v, t, hv, ht))
        names.append(name)
        print(f"  {name}: mesh {len(v)}v/{len(t)}t, hull {len(hv)}v/{len(ht)}t")

    # --- Compile C++ harness ---
    cpp_src = os.path.join(project_root, 'tests', 'ref_hausdorff.cpp')
    sobol_src = os.path.join(project_root, 'CoACD', 'src', 'sobol.cpp')
    harness_bin = os.path.join(project_root, 'ref_hausdorff')
    include_dir = os.path.join(project_root, 'CoACD', 'src')

    compile_cmd = [
        'g++', '-O2', '-std=c++17',
        '-I', include_dir,
        cpp_src, sobol_src,
        '-o', harness_bin,
    ]
    print(f"\nCompiling: {' '.join(compile_cmd)}")
    rc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        print("COMPILE FAILED:")
        print(rc.stderr)
        sys.exit(1)
    print("Compiled OK.")

    # --- Run C++ harness on each pair ---
    ref_distances = []
    tmpfiles = []
    try:
        for i, (va, ta, vb, tb) in enumerate(pairs):
            tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
            tmpfiles.append(tmp.name)
            tmp.close()
            write_mesh_pair(tmp.name, va, ta, vb, tb)

            result = subprocess.run(
                [harness_bin, tmp.name],
                capture_output=True, text=True)
            if result.returncode != 0:
                print(f"HARNESS FAILED for pair {i} ({names[i]}):")
                print(result.stderr)
                sys.exit(1)
            dist = float(result.stdout.strip())
            ref_distances.append(dist)
            print(f"  {names[i]}: CoACD hausdorff = {dist:.6f}")
            if result.stderr.strip():
                for line in result.stderr.strip().split('\n'):
                    print(f"    {line}")
    finally:
        for f in tmpfiles:
            os.unlink(f)

    # --- Save fixture ---
    save_dict = {'n_pairs': len(pairs), 'names': np.array(names)}
    for i, (va, ta, vb, tb) in enumerate(pairs):
        save_dict[f'verts_a_{i}'] = np.asarray(va, dtype=np.float64)
        save_dict[f'tris_a_{i}'] = np.asarray(ta, dtype=np.int32)
        save_dict[f'verts_b_{i}'] = np.asarray(vb, dtype=np.float64)
        save_dict[f'tris_b_{i}'] = np.asarray(tb, dtype=np.int32)
        save_dict[f'ref_{i}'] = np.float64(ref_distances[i])

    out_path = os.path.join(data_dir, 'hausdorff_coacd_ref.npz')
    np.savez(out_path, **save_dict)
    print(f"\nSaved {len(pairs)} pairs to {out_path}")

    # Clean up harness binary
    os.unlink(harness_bin)


if __name__ == '__main__':
    main()
