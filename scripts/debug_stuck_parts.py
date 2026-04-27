"""Identify and save "stuck" parts from lookahead decomposition.

A part is stuck when its (nv, nt) stays unchanged while over the concavity
threshold for two consecutive iterations — meaning every attempted plane cut
produced invalid geometry (degenerate splits).

Detection works by parsing the verbose stderr output from lookahead_decompose,
which prints per-part costs at the start of each iteration.  Parts marked with
a trailing '*' are over threshold.  An (nv, nt) pair appearing over threshold
in two consecutive iterations is stuck.

Usage:
    python scripts/debug_stuck_parts.py <mesh> [output_dir]

Options mirror the cuacd CLI decomposition flags; defaults are set to
reproduce the pamo_abl_s squirrel.obj test case.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import numpy as np
import trimesh
import cuacd


def _parse_pool_bytes(s):
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([KMG])?', s, re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid pool size: {s!r}")
    val = float(m.group(1))
    suffix = (m.group(2) or '').upper()
    mult = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3}[suffix]
    return int(val * mult)


def _find_stuck_parts(log_lines):
    """Parse verbose log and return set of (nv, nt, rv_cost) signatures that are stuck."""
    iter_over = {}  # iter_num -> set of (nv, nt, rv_cost) over threshold
    current_iter = 0
    in_listing = False

    for line in log_lines:
        if 'part   nv   nt' in line:
            in_listing = True
            iter_over.setdefault(current_iter, set())
            continue
        if in_listing and line.startswith('[la]'):
            m = re.match(r'\[la\]\s+\d+\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+', line)
            if m:
                nv, nt = int(m.group(1)), int(m.group(2))
                rv_cost = float(m.group(3))
                if line.rstrip().endswith('*'):
                    iter_over[current_iter].add((nv, nt, rv_cost))
        # Timing line ("iter N: X ms") marks end of a listing; next listing is iter+1.
        if 'iter' in line and 'ms' in line and 'n_cutting' not in line:
            current_iter += 1
            in_listing = False

    stuck = set()
    for it in sorted(iter_over.keys()):
        if it == 0:
            continue
        stuck |= (iter_over[it - 1] & iter_over[it])
    return stuck, iter_over


def main():
    p = argparse.ArgumentParser(
        description="Find and save stuck (uncuttable) parts from decomposition.")
    p.add_argument("input", help="Input mesh file.")
    p.add_argument("output", nargs="?", default=None,
                   help="Output directory (default: ./debug_stuck).")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--width", type=int, default=30)
    p.add_argument("--width2", type=int, default=5)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--quick-depth", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--device", type=int, default=-1)
    p.add_argument("--pool", type=str, default="0",
                   help="Pool size (supports K/M/G suffix, 0=auto).")
    p.add_argument("--n-concave-edges", type=int, default=128,
                   help="Max concave-edge plane samples; the pamo_abl_s squirrel "
                        "test case passes 128 via the R256 sweep script. "
                        "(default: 128)")
    p.add_argument("--concave-eps", type=float, default=0.005)
    p.add_argument("--concave-threshold", type=float, default=3.49)
    p.add_argument("--concave-iters", type=int, default=10)
    p.add_argument("--no-decompose-components", action="store_true")
    p.add_argument("--no-decompose-components-per-iter", action="store_true")
    p.add_argument("--no-merge-hulls", action="store_true")
    p.add_argument("--parts", action="store_true",
                   help="Save cut parts instead of convex hulls.")
    args = p.parse_args()

    pool_bytes = _parse_pool_bytes(args.pool)

    # Load and normalize (same as cli.py _decompose_mesh)
    mesh = trimesh.load(args.input, force='mesh')
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    tris = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2
    extent = float((hi - lo).max())
    scale = extent / 2 if extent > 0 else 1.0
    norm_verts = ((verts - center) / scale).astype(np.float32)

    # Run decomposition and capture verbose stderr via a helper subprocess.
    # We cannot reliably capture C fprintf(stderr) through Python's sys.stderr
    # redirection, so we re-invoke this same process to collect the log.
    script = pathlib.Path(__file__).resolve()
    tmp_log = pathlib.Path(tempfile.mktemp(suffix=".log", prefix="la_stuck_"))

    # Build the inner-run command
    inner_cmd = [
        sys.executable, "-c", f"""
import numpy as np, trimesh, cuacd
mesh = trimesh.load({str(args.input)!r}, force='mesh')
verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
tris = np.ascontiguousarray(mesh.faces, dtype=np.int32)
lo = verts.min(axis=0); hi = verts.max(axis=0)
center = (lo + hi) / 2
extent = float((hi - lo).max())
scale = extent / 2 if extent > 0 else 1.0
norm_verts = ((verts - center) / scale).astype(np.float32)
with cuacd.Context(device={args.device}, pool_bytes={pool_bytes}) as ctx:
    parts = ctx.lookahead_decompose(
        norm_verts, tris,
        max_iters={args.max_iters},
        width={args.width},
        width2={args.width2},
        depth={args.depth},
        quick_depth={args.quick_depth},
        threshold={args.threshold},
        verbose=1,
        decompose_components={not args.no_decompose_components},
        no_decompose_components_per_iter={args.no_decompose_components_per_iter},
        n_concave_edges={args.n_concave_edges},
        concave_eps={args.concave_eps},
        concave_threshold={args.concave_threshold},
        concave_iters={args.concave_iters},
        merge_hulls={not args.no_merge_hulls})
print(f'DONE: {{len(parts)}} parts')
"""
    ]

    result = subprocess.run(
        inner_cmd,
        capture_output=True, text=True, timeout=120)
    log_lines = result.stderr.split('\n')

    # Parse stuck parts
    stuck_sigs, iter_over = _find_stuck_parts(log_lines)

    # Print summary
    print("Over-threshold parts per iteration:")
    for it in sorted(iter_over.keys()):
        s = iter_over[it]
        print(f"  iter {it}: {len(s)} over-threshold", end="")
        if len(s) <= 5:
            for nv, nt in sorted(s):
                print(f" (nv={nv}, nt={nt})", end="")
        print()

    if not stuck_sigs:
        print("\nNo stuck parts detected (no (nv,nt) pair appeared over-threshold "
              "in two consecutive iterations).")
        return

    print(f"\nStuck parts: {len(stuck_sigs)}")
    for nv, nt in sorted(stuck_sigs):
        print(f"  nv={nv}, nt={nt}")

    # Now run again (quiet) to get the actual part meshes
    with cuacd.Context(device=args.device, pool_bytes=pool_bytes) as ctx:
        all_parts = ctx.lookahead_decompose(
            norm_verts, tris,
            max_iters=args.max_iters,
            width=args.width,
            width2=args.width2,
            depth=args.depth,
            quick_depth=args.quick_depth,
            threshold=args.threshold,
            decompose_components=not args.no_decompose_components,
            no_decompose_components_per_iter=args.no_decompose_components_per_iter,
            n_concave_edges=args.n_concave_edges,
            concave_eps=args.concave_eps,
            concave_threshold=args.concave_threshold,
            concave_iters=args.concave_iters,
            merge_hulls=not args.no_merge_hulls)

    # Match stuck signatures to final parts (using nv, nt, rv_cost)
    out_dir = pathlib.Path(args.output or "debug_stuck")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.input).stem

    rng = np.random.default_rng(42)
    saved = 0
    scene = trimesh.Scene()
    for i, (pv, pt, hv, ht) in enumerate(all_parts):
        # Compute rv_cost for this part to match against stuck signatures
        nv, nt = len(pv), len(pt)
        # rv_cost = (hull_vol - mesh_vol) / hull_vol — same as la_part_cost_rv
        # We approximate from0 hull if available, otherwise skip
        hull_v = np.abs(np.sum(np.sum(hv[ht[:,0]] * np.cross(hv[ht[:,1]], hv[ht[:,2]]), axis=1))) / 6 if len(hv) > 0 and len(ht) > 0 else 0
        mesh_v = np.abs(np.sum(np.sum(pv[pt[:,0]] * np.cross(pv[pt[:,1]], pv[pt[:,2]]), axis=1))) / 6
        rv_cost = (hull_v - mesh_v) / hull_v if hull_v > 0 else 0
        if (nv, nt, round(rv_cost, 4)) in [(s[0], s[1], round(s[2], 4)) for s in stuck_sigs]:
            saved += 1
            if args.parts:
                v = pv * scale + center
                t = pt
            else:
                v = hv * scale + center
                t = ht
            mesh_obj = trimesh.Trimesh(v, t)
            color = (rng.random(3) * 255).astype(np.uint8)
            mesh_obj.visual = trimesh.visual.ColorVisuals(mesh=mesh_obj)
            mesh_obj.visual.vertex_colors[:, :3] = color
            scene.add_geometry(mesh_obj, node_name=f"stuck_{i}")
            print(f"  part {i}: nv={len(pv)}, nt={len(pt)} — saved")

    out_path = out_dir / f"{stem}_stuck.glb"
    scene.export(str(out_path))
    print(f"\nSaved {saved} stuck part(s) to {out_path}")


if __name__ == "__main__":
    main()