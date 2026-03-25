"""Compare C++ CoACD vs pure-Python CoACD on Octocat mesh."""

import time
import numpy as np

OBJ_PATH = "CoACD/examples/Octocat-v2.obj"


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


vertices, triangles = load_obj(OBJ_PATH)
print(f"Loaded: {len(vertices)} vertices, {len(triangles)} triangles")
print()

# --- C++ CoACD ---
import coacd

cpp_mesh = coacd.Mesh(vertices, triangles)
print("=== C++ CoACD ===")
t0 = time.time()
cpp_parts = coacd.run_coacd(
    cpp_mesh,
    threshold=0.08,
    mcts_iterations=100,
    mcts_nodes=20,
    mcts_max_depth=3,
    resolution=2000,
    seed=42,
    merge=True,
    preprocess_mode="off",
)
t1 = time.time()
print(f"Time: {t1 - t0:.2f}s")
print(f"Parts: {len(cpp_parts)}")
cpp_total_verts = sum(len(v) for v, _ in cpp_parts)
cpp_total_tris = sum(len(t) for _, t in cpp_parts)
print(f"Total vertices: {cpp_total_verts}, triangles: {cpp_total_tris}")
print()

# --- Pure Python CoACD ---
from coacd_gpu.coacd import run_coacd

# Use reduced params for Python: pure-Python MCTS is ~100x slower per iteration
# than C++ with OpenMP, so we reduce iterations to keep runtime reasonable.
PY_MCTS_ITER = 30
PY_MCTS_NODES = 10
PY_RESOLUTION = 1000

print(f"=== Python CoACD (iter={PY_MCTS_ITER}, nodes={PY_MCTS_NODES}) ===")
t0 = time.time()
py_parts = run_coacd(
    vertices,
    triangles,
    threshold=0.08,
    mcts_iterations=PY_MCTS_ITER,
    mcts_nodes=PY_MCTS_NODES,
    mcts_max_depth=3,
    resolution=PY_RESOLUTION,
    seed=42,
    merge=True,
)
t1 = time.time()
print(f"Time: {t1 - t0:.2f}s")
print(f"Parts: {len(py_parts)}")
py_total_verts = sum(len(v) for v, _ in py_parts)
py_total_tris = sum(len(t) for _, t in py_parts)
print(f"Total vertices: {py_total_verts}, triangles: {py_total_tris}")
print()

# --- Summary ---
print("=== Comparison ===")
print(f"{'':20s} {'C++':>10s} {'Python':>10s}")
print(f"{'Parts':20s} {len(cpp_parts):10d} {len(py_parts):10d}")
print(f"{'Total verts':20s} {cpp_total_verts:10d} {py_total_verts:10d}")
print(f"{'Total tris':20s} {cpp_total_tris:10d} {py_total_tris:10d}")

# --- Save output meshes as OBJ for visualization ---
import os, trimesh

out_dir = "compare_output"
os.makedirs(out_dir, exist_ok=True)


def save_parts(parts, prefix):
    """Save each part as separate OBJ + a combined colored OBJ."""
    meshes = []
    for i, (v, f) in enumerate(parts):
        m = trimesh.Trimesh(vertices=v, faces=f)
        m.export(os.path.join(out_dir, f"{prefix}_part_{i:03d}.obj"))
        # Assign a random color per part for the combined mesh
        color = trimesh.visual.random_color()
        m.visual.face_colors = color
        meshes.append(m)
    combined = trimesh.util.concatenate(meshes)
    combined.export(os.path.join(out_dir, f"{prefix}_combined.obj"))
    # Also save as GLB for easy 3D viewing
    combined.export(os.path.join(out_dir, f"{prefix}_combined.glb"))
    print(f"Saved {len(parts)} parts to {out_dir}/{prefix}_*.obj")
    print(f"Saved colored combined mesh to {out_dir}/{prefix}_combined.glb")


save_parts(cpp_parts, "cpp")
save_parts(py_parts, "python")
