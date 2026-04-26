# Debugging Stuck Decomposition Parts

When a mesh gets "stuck" during decomposition: all candidate cuts fail for some part, so those parts remain over the concavity threshold (`*`) with identical (nv, nt) across consecutive iterations. The program reaches `max_iters` without converging.

## Procedure

### Step 1: Find first stuck parts and visualize

Run with `verbose=1` at low `max_iters` (e.g., 8). Identify (nv, nt) pairs that appear over-threshold (`*`) in two consecutive iterations. Use `scripts/debug_stuck_parts.py` to automatically detect and save stuck parts as GLB.

```bash
python scripts/debug_stuck_parts.py <mesh.obj> <output_dir> --max-iters 8
```

Inspect the saved GLB for weird geometry (degenerate tris, non-manifold edges, etc.).

### Step 2: Isolate and reproduce

Extract the stuck part's normalized verts/tris (after the original global normalization) and run `lookahead_decompose` on it directly with **no re-normalization**. If stuck, should remain 1 part over threshold after 2+ iters.

```python
data = np.load('stuck_part.npz')
ctx.lookahead_decompose(data['pv'], data['pt'], max_iters=4, verbose=1, ...)
```

### Step 3: BEAM_DEBUG root cause

Build with device-side debug output:
```bash
COACD_BEAM_DEBUG=1 pip install -e .
```

Run the isolated stuck part, capture stderr to a file, and grep for `DPRINTF` output to find why each candidate cut fails (degenerate split, zero-volume child, etc.).

### Step 4: Free exploration

Based on findings from steps 1-3, debug the specific failure path in the relevant CUDA kernel (typically `plane_cut.cuh` or `la_expand.cu`).