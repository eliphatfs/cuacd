"""Shared test fixtures."""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="run slow tests (MCTS/pipeline)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (MCTS/pipeline)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="use --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def _box_mesh(lo, hi):
    """Create a box mesh from (lo) to (hi) corners."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    t = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],   # front
        [2, 3, 7], [2, 7, 6],   # back
        [0, 4, 7], [0, 7, 3],   # left
        [1, 2, 6], [1, 6, 5],   # right
    ], dtype=np.int32)
    return v, t


@pytest.fixture
def unit_cube():
    """Unit cube [0,1]^3 as (vertices, triangles)."""
    return _box_mesh([0, 0, 0], [1, 1, 1])


@pytest.fixture
def offset_cube():
    """Cube [2,3]^3 as (vertices, triangles)."""
    return _box_mesh([2, 2, 2], [3, 3, 3])


@pytest.fixture
def l_shape():
    """L-shaped mesh from two boxes joined along x-axis."""
    v1, t1 = _box_mesh([0, 0, 0], [1, 1, 1])
    v2, t2 = _box_mesh([1, 0, 0], [2, 1, 0.5])
    verts = np.vstack([v1, v2])
    tris = np.vstack([t1, t2 + len(v1)])
    return verts, tris


def load_obj(path):
    """Load a simple OBJ file (vertices + triangle faces)."""
    verts = []
    faces = []
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
