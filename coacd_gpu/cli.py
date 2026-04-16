"""coacd-gpu — CLI for lookahead convex decomposition of mesh files."""

import argparse
import multiprocessing
import os
import re
import sys
import time
import pathlib
import numpy as np
import trimesh
import coacd_gpu


def _parse_pool_bytes(s):
    """Parse pool size string with optional K/M/G suffix to bytes. '0' means auto."""
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([KMG])?', s, re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid pool size: {s!r} (use e.g. 512M, 2G, or 0 for auto)")
    val = float(m.group(1))
    suffix = (m.group(2) or '').upper()
    mult = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3}[suffix]
    return int(val * mult)


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


def _decompose_mesh(ctx, verts, tris, args):
    """Normalize, decompose, and denormalize a mesh. Returns list of (verts, tris)."""
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2
    extent = float((hi - lo).max())
    scale = extent / 2 if extent > 0 else 1.0
    norm_verts = ((verts - center) / scale).astype(np.float32)

    parts = ctx.lookahead_decompose(
        norm_verts, tris,
        max_iters=args.max_iters,
        width=args.width,
        width2=args.width2,
        depth=args.depth,
        quick_depth=args.quick_depth,
        threshold=args.threshold,
        verbose=args.verbose,
        debug=args.debug,
        decompose_components=not args.no_decompose_components,
        decompose_components_per_iter=args.decompose_components_per_iter,
        n_concave_edges=args.n_concave_edges,
        concave_eps=args.concave_eps,
        concave_threshold=args.concave_threshold,
        concave_iters=args.concave_iters)

    if not args.parts:
        return [(hv * scale + center, ht) for _, _, hv, ht in parts]
    else:
        return [(pv * scale + center, pt) for pv, pt, _, _ in parts]


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


# -- Pipeline workers for multiprocessing --

_SENTINEL = None  # signals end-of-stream


def _loader_worker(mesh_list, load_queue, print_lock):
    """Load meshes and push (rel_path, verts, tris) into load_queue."""
    for abs_path, rel_path in mesh_list:
        try:
            verts, tris = _load_mesh(abs_path)
        except Exception as e:
            with print_lock:
                print(f'{rel_path}: load error — {e}', file=sys.stderr, flush=True)
            continue
        load_queue.put((rel_path, verts, tris))
    load_queue.put(_SENTINEL)


def _processor_worker(load_queue, save_queue, print_lock, args):
    """Decompose meshes from load_queue, push results into save_queue."""
    ctx = coacd_gpu.Context(device=args.device, pool_bytes=args.pool)
    try:
        while True:
            item = load_queue.get()
            if item is _SENTINEL:
                break
            rel_path, verts, tris = item

            t0 = time.perf_counter()
            out_parts = None
            for attempt in range(2):
                try:
                    out_parts = _decompose_mesh(ctx, verts, tris, args)
                    break
                except Exception as e:
                    if attempt == 0:
                        with print_lock:
                            print(f'{rel_path}: decompose error — {e}, recreating context and retrying',
                                  file=sys.stderr, flush=True)
                        try:
                            ctx.close()
                        except Exception:
                            pass
                        ctx = coacd_gpu.Context(device=args.device, pool_bytes=args.pool)
                    else:
                        with print_lock:
                            print(f'{rel_path}: decompose error on retry — {e}',
                                  file=sys.stderr, flush=True)

            if out_parts is None:
                continue

            elapsed = time.perf_counter() - t0
            save_queue.put((rel_path, out_parts, elapsed))
    finally:
        ctx.close()
    save_queue.put(_SENTINEL)


def _saver_worker(save_queue, output_dir, print_lock):
    """Save decomposed parts from save_queue."""
    while True:
        item = save_queue.get()
        if item is _SENTINEL:
            break
        rel_path, parts, elapsed = item
        _save_parts(parts, output_dir, rel_path)
        with print_lock:
            print(f'{rel_path}  {elapsed:.2f}s  {len(parts)} parts', flush=True)


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
    parser.add_argument('--pool', type=_parse_pool_bytes, default=0,
                        help='GPU pool size in bytes (supports K/M/G suffix, e.g. 512M, 2G; 0 = auto).')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level (0/1/2).')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0/1).')
    parser.add_argument('--parts', action='store_true', default=False,
                        help='Save cut parts instead of convex hulls (default: hulls).')
    parser.add_argument('--no-decompose-components', action='store_true', default=False,
                        help='Skip connected components decomposition (default: enabled).')
    parser.add_argument('--decompose-components-per-iter', action='store_true', default=False,
                        help='Run connected components decomposition each iteration (default: off).')
    parser.add_argument('--n-concave-edges', type=int, default=0,
                        help='Max concave edges to sample per cutting part (default: 0 = disabled).')
    parser.add_argument('--concave-eps', type=float, default=0.005,
                        help='Epsilon offset for edge-based planes (default: 0.005).')
    parser.add_argument('--concave-threshold', type=float, default=3.49,
                        help='Dihedral angle threshold in radians for concavity (default: 3.49 ~200deg).')
    parser.add_argument('--concave-iters', type=int, default=1,
                        help='Number of first iterations to include concave edge sampling (default: 1).')
    parser.add_argument('--serial', action='store_true', default=False,
                        help='Serial load-process-save instead of pipelined workers (easier debugging).')
    args = parser.parse_args()

    meshes = _find_meshes(args.input, args.recursive)
    if not meshes:
        print(f'No mesh files found in: {args.input}', file=sys.stderr)
        sys.exit(1)

    if args.serial:
        ctx = coacd_gpu.Context(device=args.device, pool_bytes=args.pool)
        try:
            for abs_path, rel_path in meshes:
                try:
                    verts, tris = _load_mesh(abs_path)
                except Exception as e:
                    print(f'{rel_path}: load error — {e}', file=sys.stderr, flush=True)
                    continue

                t0 = time.perf_counter()
                try:
                    out_parts = _decompose_mesh(ctx, verts, tris, args)
                except Exception as e:
                    print(f'{rel_path}: decompose error — {e}', file=sys.stderr, flush=True)
                    continue

                elapsed = time.perf_counter() - t0
                _save_parts(out_parts, args.output, rel_path)
                print(f'{rel_path}  {elapsed:.2f}s  {len(out_parts)} parts', flush=True)
        finally:
            ctx.close()
        return

    mp_ctx = multiprocessing.get_context('spawn')
    print_lock = mp_ctx.Lock()
    load_queue = mp_ctx.Queue(maxsize=2)
    save_queue = mp_ctx.Queue(maxsize=2)

    loader = mp_ctx.Process(target=_loader_worker, args=(meshes, load_queue, print_lock))
    processor = mp_ctx.Process(target=_processor_worker, args=(load_queue, save_queue, print_lock, args))
    saver = mp_ctx.Process(target=_saver_worker, args=(save_queue, args.output, print_lock))

    loader.start()
    processor.start()
    saver.start()

    loader.join()
    processor.join()
    saver.join()


if __name__ == '__main__':
    main()
