"""coacd-gpu — CLI for lookahead convex decomposition of mesh files."""

import argparse
import os
import sys
import time
import pathlib
import numpy as np
import trimesh
import coacd_gpu


def _find_meshes(input_path, recursive):
    """Return list of (abs_path, rel_path) for all mesh files under input_path."""
    mesh_exts = {'.obj', '.stl', '.ply', '.off', '.glb', '.gltf'}
    p = pathlib.Path(input_path).resolve()
    if p.is_file():
        return [(p, pathlib.Path(p.name))]
    base = p
    if recursive:
        files = [f for f in base.rglob('*') if f.is_file() and f.suffix.lower() in mesh_exts]
    else:
        files = [f for f in base.iterdir() if f.is_file() and f.suffix.lower() in mesh_exts]
    files = sorted(files)
    return [(f, f.relative_to(base)) for f in files]


def _load_mesh(path):
    """Load mesh using trimesh. Returns (verts, tris) as numpy arrays."""
    mesh = trimesh.load(str(path), force='mesh')
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    tris = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    return verts, tris


def _save_parts(parts, output_dir, rel_path):
    """Save decomposition as a single GLB with random per-part colors."""
    rng = np.random.default_rng(0)
    scene = trimesh.Scene()
    stem = pathlib.Path(rel_path).stem
    for i, (verts, tris) in enumerate(parts):
        mesh = trimesh.Trimesh(verts, tris)
        color = (rng.random(3) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        mesh.visual.vertex_colors[:, :3] = color
        scene.add_geometry(mesh, node_name=f"{stem}_{i}")
    out_dir = pathlib.Path(output_dir) / pathlib.Path(rel_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{stem}.glb'
    scene.export(str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog='coacd-gpu',
        description='GPU lookahead convex decomposition of mesh files.')
    parser.add_argument('input', help='Input mesh file or directory.')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('-r', '--recursive', action='store_true', default=False,
                        help='Recurse into subdirectories (default: false).')
    parser.add_argument('--width', type=int, default=60,
                        help='Number of candidate cuts per expansion level (default: 60).')
    parser.add_argument('--width2', type=int, default=5,
                        help='Cuts at deeper expansion levels (default: 5).')
    parser.add_argument('--depth', type=int, default=2,
                        help='Full expansion depth (default: 2).')
    parser.add_argument('--quick-depth', type=int, default=0,
                        help='Quick expansion depth (default: 0).')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Convergence threshold (default: 0.05).')
    parser.add_argument('--max-iters', type=int, default=100,
                        help='Max outer iterations (default: 100).')
    parser.add_argument('--device', type=int, default=-1,
                        help='GPU device ordinal (default: -1 = reuse existing context).')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level (0/1/2).')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0/1).')
    parser.add_argument('--parts', action='store_true', default=False,
                        help='Save cut parts instead of convex hulls (default: hulls).')
    args = parser.parse_args()

    meshes = _find_meshes(args.input, args.recursive)
    if not meshes:
        print(f'No mesh files found in: {args.input}', file=sys.stderr)
        sys.exit(1)

    with coacd_gpu.Context(device=args.device) as ctx:
        for abs_path, rel_path in meshes:
            try:
                verts, tris = _load_mesh(abs_path)
            except Exception as e:
                print(f'{rel_path}: load error — {e}', file=sys.stderr)
                continue

            # Normalize to [-1, 1]
            lo = verts.min(axis=0)
            hi = verts.max(axis=0)
            center = (lo + hi) / 2
            extent = float((hi - lo).max())
            scale = extent / 2 if extent > 0 else 1.0
            norm_verts = ((verts - center) / scale).astype(np.float32)

            t0 = time.perf_counter()
            try:
                parts = ctx.lookahead_decompose(
                    norm_verts, tris,
                    max_iters=args.max_iters,
                    width=args.width,
                    width2=args.width2,
                    depth=args.depth,
                    quick_depth=args.quick_depth,
                    threshold=args.threshold,
                    verbose=args.verbose,
                    debug=args.debug)
            except Exception as e:
                print(f'{rel_path}: decompose error — {e}', file=sys.stderr)
                continue
            elapsed = time.perf_counter() - t0

            if not args.parts:
                hulls = ctx.batch_kdop_hull_mesh([pv for pv, _ in parts])
                parts = [(hv * scale + center, ht) for hv, ht, _ in hulls]
            else:
                parts = [(pv * scale + center, pt) for pv, pt in parts]

            _save_parts(parts, args.output, rel_path)
            print(f'{rel_path}  {elapsed:.2f}s  {len(parts)} parts')


if __name__ == '__main__':
    main()
