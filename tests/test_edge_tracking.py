"""Test max edge pairs in D&C hull using icospheres.

Requires: CUACD_TRACK_EDGES=1 pip install -e .
Reads EDGE_TRACK printf lines from GPU kernel output.
"""
import subprocess
import sys
import re
import numpy as np


def icosphere(subdivisions=0):
    """Generate icosphere vertices. All points lie on the unit sphere."""
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts[0])

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                e = tuple(sorted((tri[i], tri[(i+1)%3])))
                if e not in edge_midpoints:
                    mid = (verts[e[0]] + verts[e[1]]) / 2.0
                    mid /= np.linalg.norm(mid)
                    edge_midpoints[e] = len(verts)
                    verts = np.vstack([verts, mid])
                mids.append(edge_midpoints[e])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces.extend([[a,m0,m2],[b,m1,m0],[c,m2,m1],[m0,m1,m2]])
        faces = np.array(new_faces, dtype=np.int32)

    return verts.astype(np.float32), faces


def main():
    import cuacd
    with cuacd.Context(device=0) as ctx:
        for subdiv in range(6):
            verts, faces = icosphere(subdiv)
            n = len(verts)
            vols, errs = ctx.batch_hull_volume([verts])
            assert errs[0] == 0, f"subdiv={subdiv} n={n} error={errs[0]}"
            print(f"subdiv={subdiv} n={n:5d} vol={vols[0]:.4f} err={errs[0]}")


if __name__ == "__main__":
    main()
