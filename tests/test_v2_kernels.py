"""Per-kernel tests for V2 beam search (beam_expansion, beam_hausdorff_parts, beam_termination).

Tests each kernel in isolation with simple, well-understood inputs to verify basic function.
"""

import numpy as np
import pytest
import ctypes

try:
    import coacd_gpu._gpu as _gpu
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="GPU extension not available")


@pytest.fixture(autouse=True)
def _ensure_gpu_ctx():
    """Ensure GPU context is initialized before each test."""
    _gpu.init(0)
    yield

# ---------------------------------------------------------------------------
# Struct dtypes matching C PartInfoV2 and WorkItem
# ---------------------------------------------------------------------------

MAX_PARTS_PER_BEAM = 64

# PartInfoV2: 4 ints (vert), 4 ints (tri), 4 ints (hull vert), 4 ints (hull tri),
#             6 floats (bbox), float rv_cost, float hausdorff, float mesh_volume, float hull_volume
# Total: 8 ints + 6 floats + 4 floats = 8*4 + 10*4 = 72 bytes
# Actually from C struct:
#   int vert_offset, vert_count, tri_offset, tri_count           (4 ints = 16B)
#   int hull_vert_offset, hull_vert_count, hull_tri_offset, hull_tri_count  (4 ints = 16B)
#   float bbox[6]                                                (24B)
#   float rv_cost, hausdorff, mesh_volume, hull_volume           (16B)
# Total = 72 bytes
PARTINFO_DTYPE = np.dtype([
    ('vert_offset', np.int32), ('vert_count', np.int32),
    ('tri_offset', np.int32), ('tri_count', np.int32),
    ('hull_vert_offset', np.int32), ('hull_vert_count', np.int32),
    ('hull_tri_offset', np.int32), ('hull_tri_count', np.int32),
    ('bbox', np.float32, (6,)),
    ('rv_cost', np.float32), ('hausdorff', np.float32),
    ('mesh_volume', np.float32), ('hull_volume', np.float32),
])

# WorkItem: int part_indices[64], int num_parts, int worst_part_idx, float worst_metric
# Total = 64*4 + 4 + 4 + 4 = 268 bytes
WORKITEM_DTYPE = np.dtype([
    ('part_indices', np.int32, (MAX_PARTS_PER_BEAM,)),
    ('num_parts', np.int32),
    ('worst_part_idx', np.int32),
    ('worst_metric', np.float32),
])


def _make_part(vert_offset=0, vert_count=0, tri_offset=0, tri_count=0,
               hull_vert_offset=0, hull_vert_count=0,
               hull_tri_offset=0, hull_tri_count=0,
               rv_cost=0.0, hausdorff=-1.0, mesh_volume=0.0, hull_volume=0.0):
    """Create a single PartInfoV2 numpy struct."""
    p = np.zeros(1, dtype=PARTINFO_DTYPE)
    p['vert_offset'] = vert_offset
    p['vert_count'] = vert_count
    p['tri_offset'] = tri_offset
    p['tri_count'] = tri_count
    p['hull_vert_offset'] = hull_vert_offset
    p['hull_vert_count'] = hull_vert_count
    p['hull_tri_offset'] = hull_tri_offset
    p['hull_tri_count'] = hull_tri_count
    p['rv_cost'] = rv_cost
    p['hausdorff'] = hausdorff
    p['mesh_volume'] = mesh_volume
    p['hull_volume'] = hull_volume
    return p[0]


def _make_work_item(part_indices, worst_part_idx=0, worst_metric=0.0):
    """Create a single WorkItem numpy struct."""
    wi = np.zeros(1, dtype=WORKITEM_DTYPE)
    for i, pi in enumerate(part_indices):
        wi['part_indices'][0, i] = pi
    wi['num_parts'] = len(part_indices)
    wi['worst_part_idx'] = worst_part_idx
    wi['worst_metric'] = worst_metric
    return wi[0]


# ---------------------------------------------------------------------------
# Kernel 3: beam_termination
# ---------------------------------------------------------------------------

class TestTermination:
    """Test beam_termination kernel in isolation."""

    def test_single_item_below_threshold(self):
        """One work item with one part below threshold → should terminate."""
        parts = np.array([_make_part(rv_cost=0.01, hausdorff=0.02)], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.02)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == 0  # work item 0 terminated

    def test_single_item_above_threshold(self):
        """One work item with one part above threshold → should NOT terminate."""
        parts = np.array([_make_part(rv_cost=0.1, hausdorff=-1.0)], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.1)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == -1  # no termination

    def test_worst_part_updated(self):
        """Termination kernel should update worst_part_idx and worst_metric."""
        parts = np.array([
            _make_part(rv_cost=0.01, hausdorff=0.01),  # part 0: good
            _make_part(rv_cost=0.2, hausdorff=-1.0),    # part 1: worst (rv only)
            _make_part(rv_cost=0.05, hausdorff=0.08),   # part 2: moderate
        ], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0, 1, 2], worst_part_idx=0, worst_metric=0.0)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == -1  # not terminated (part 1 above threshold)
        # worst_part_idx should be 1 (index into part_indices, which is part 1)
        assert items[0]['worst_part_idx'] == 1
        assert items[0]['worst_metric'] == pytest.approx(0.2, abs=1e-5)

    def test_multiple_items_one_terminates(self):
        """Two work items, one below threshold → returns the terminated one."""
        parts = np.array([
            _make_part(rv_cost=0.01, hausdorff=0.02),  # below 0.05
            _make_part(rv_cost=0.1, hausdorff=0.15),   # above 0.05
        ], dtype=PARTINFO_DTYPE)
        items = np.array([
            _make_work_item([1], worst_part_idx=0, worst_metric=0.15),  # item 0: above
            _make_work_item([0], worst_part_idx=0, worst_metric=0.02),  # item 1: below
        ], dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == 1  # item 1 terminated

    def test_hausdorff_overrides_rv(self):
        """When hausdorff >= 0, metric = max(rv, hausdorff)."""
        parts = np.array([
            _make_part(rv_cost=0.01, hausdorff=0.1),  # rv low but hausdorff high
        ], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.0)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == -1  # hausdorff 0.1 > threshold 0.05
        assert items[0]['worst_metric'] == pytest.approx(0.1, abs=1e-5)

    def test_hausdorff_negative_uses_rv_only(self):
        """When hausdorff < 0 (not computed), metric = rv only."""
        parts = np.array([
            _make_part(rv_cost=0.03, hausdorff=-1.0),
        ], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.0)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == 0  # rv 0.03 < threshold 0.05 → terminated

    def test_multi_part_all_below(self):
        """Work item with multiple parts all below threshold → terminates."""
        parts = np.array([
            _make_part(rv_cost=0.01, hausdorff=0.02),
            _make_part(rv_cost=0.03, hausdorff=0.04),
            _make_part(rv_cost=0.02, hausdorff=0.01),
        ], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0, 1, 2], worst_part_idx=0, worst_metric=0.0)],
                         dtype=WORKITEM_DTYPE)
        result = _gpu.test_termination(
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            0.05)
        assert result == 0


# ---------------------------------------------------------------------------
# Kernel 2: beam_hausdorff_parts
# ---------------------------------------------------------------------------

def _cube_mesh_pools():
    """Create vertex/triangle pools for a unit cube mesh and its hull (identical for cube)."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    # Pool: mesh verts + hull verts (same for cube)
    # Hull tris use absolute indices starting after mesh verts
    hull_tris = tris + 8  # hull verts start at offset 8
    vertex_pool = np.vstack([verts, verts]).astype(np.float32)  # [16, 3]
    triangle_pool = np.vstack([tris, hull_tris]).astype(np.int32)  # [24, 3]
    return vertex_pool, triangle_pool


class TestHausdorffParts:
    """Test beam_hausdorff_parts kernel in isolation."""

    def test_identical_mesh_and_hull(self):
        """Cube mesh == its hull → Hausdorff distance ≈ 0."""
        vp, tp = _cube_mesh_pools()
        parts = np.array([_make_part(
            vert_offset=0, vert_count=8,
            tri_offset=0, tri_count=12,
            hull_vert_offset=8, hull_vert_count=8,
            hull_tri_offset=12, hull_tri_count=12,
            rv_cost=0.01, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        indices = np.array([0], dtype=np.int32)
        rc = _gpu.test_hausdorff_parts(
            vp.ctypes.data, len(vp), tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            indices.ctypes.data, len(indices),
            0.05)
        assert rc == 0
        assert parts[0]['hausdorff'] == pytest.approx(0.0, abs=1e-4)

    def test_skips_already_computed(self):
        """Parts with hausdorff >= 0 should be skipped."""
        vp, tp = _cube_mesh_pools()
        parts = np.array([_make_part(
            vert_offset=0, vert_count=8,
            tri_offset=0, tri_count=12,
            hull_vert_offset=8, hull_vert_count=8,
            hull_tri_offset=12, hull_tri_count=12,
            rv_cost=0.01, hausdorff=0.42,  # already computed
        )], dtype=PARTINFO_DTYPE)
        indices = np.array([0], dtype=np.int32)
        _gpu.test_hausdorff_parts(
            vp.ctypes.data, len(vp), tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            indices.ctypes.data, len(indices),
            0.05)
        assert parts[0]['hausdorff'] == pytest.approx(0.42, abs=1e-6)  # unchanged

    def test_high_rv_sets_hausdorff_to_rv(self):
        """Parts with rv_cost > 2*threshold get hausdorff = rv_cost (skip expensive computation)."""
        vp, tp = _cube_mesh_pools()
        parts = np.array([_make_part(
            vert_offset=0, vert_count=8,
            tri_offset=0, tri_count=12,
            hull_vert_offset=8, hull_vert_count=8,
            hull_tri_offset=12, hull_tri_count=12,
            rv_cost=0.5, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        indices = np.array([0], dtype=np.int32)
        _gpu.test_hausdorff_parts(
            vp.ctypes.data, len(vp), tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            indices.ctypes.data, len(indices),
            0.05)
        assert parts[0]['hausdorff'] == pytest.approx(0.5, abs=1e-6)

    def test_scaled_hull_has_nonzero_hausdorff(self):
        """Hull scaled larger than mesh → nonzero Hausdorff distance."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        hull_verts = verts * 2.0  # scaled by 2
        tris = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)
        hull_tris = tris + 8
        vp = np.vstack([verts, hull_verts]).astype(np.float32)
        tp = np.vstack([tris, hull_tris]).astype(np.int32)

        parts = np.array([_make_part(
            vert_offset=0, vert_count=8,
            tri_offset=0, tri_count=12,
            hull_vert_offset=8, hull_vert_count=8,
            hull_tri_offset=12, hull_tri_count=12,
            rv_cost=0.01, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        indices = np.array([0], dtype=np.int32)
        _gpu.test_hausdorff_parts(
            vp.ctypes.data, len(vp), tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            indices.ctypes.data, len(indices),
            0.05)
        assert parts[0]['hausdorff'] > 0.5  # scaled 2x → significant distance


# ---------------------------------------------------------------------------
# Kernel 1: beam_expansion
# ---------------------------------------------------------------------------

def _l_shape_mesh():
    """L-shaped mesh (non-convex) with its hull."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0], [0.5, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 0.5, 1], [0.5, 0.5, 1], [0.5, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    t = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5],
        [6, 8, 7], [6, 9, 8], [6, 11, 9], [9, 11, 10],
        [0, 6, 7], [0, 7, 1],
        [1, 7, 8], [1, 8, 2],
        [2, 8, 9], [2, 9, 3],
        [3, 9, 10], [3, 10, 4],
        [4, 10, 11], [4, 11, 5],
        [5, 11, 6], [5, 6, 0],
    ], dtype=np.int32)
    return v, t


class TestExpansion:
    """Test beam_expansion kernel in isolation."""

    def test_basic_expansion_runs(self):
        """Run expansion on L-shape and verify it produces candidates without kernel errors."""
        v, t = _l_shape_mesh()

        # Build hull via scipy
        from scipy.spatial import ConvexHull
        hull = ConvexHull(v)
        hull_verts = v[hull.vertices].copy()
        hull_tris_raw = hull.simplices.copy()
        # Remap to hull vertex indices
        remap = np.full(len(v), -1, dtype=np.int32)
        for i, vi in enumerate(hull.vertices):
            remap[vi] = i
        hull_tris = np.array([[remap[a], remap[b], remap[c]] for a, b, c in hull_tris_raw],
                             dtype=np.int32)
        # Fix winding
        centroid = hull_verts.mean(axis=0)
        for i in range(len(hull_tris)):
            a, b, c = hull_tris[i]
            va, vb, vc = hull_verts[a], hull_verts[b], hull_verts[c]
            normal = np.cross(vb - va, vc - va)
            if np.dot(normal, va - centroid) < 0:
                hull_tris[i, 1], hull_tris[i, 2] = hull_tris[i, 2], hull_tris[i, 1]

        # Build pools: mesh + hull packed
        n_v, n_t = len(v), len(t)
        n_hv, n_ht = len(hull_verts), len(hull_tris)
        hull_tris_abs = hull_tris + n_v  # absolute indices into vertex pool
        vp = np.vstack([v, hull_verts]).astype(np.float32)
        tp = np.vstack([t, hull_tris_abs]).astype(np.int32)

        # Initial part
        parts = np.array([_make_part(
            vert_offset=0, vert_count=n_v,
            tri_offset=0, tri_count=n_t,
            hull_vert_offset=n_v, hull_vert_count=n_hv,
            hull_tri_offset=n_t, hull_tri_count=n_ht,
            rv_cost=0.2, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.2)],
                         dtype=WORKITEM_DTYPE)

        beam_width = 4
        cuts_per_axis = 3
        out_parts_cap = 1024
        out_vert_cap = 100000
        out_tri_cap = 100000

        out_wi = np.zeros(beam_width, dtype=WORKITEM_DTYPE)
        out_parts = np.zeros(out_parts_cap, dtype=PARTINFO_DTYPE)
        out_vp = np.zeros((out_vert_cap, 3), dtype=np.float32)
        out_tp = np.zeros((out_tri_cap, 3), dtype=np.int32)

        n_out_parts, n_out_verts, n_out_tris, kerr = _gpu.test_expansion(
            vp.ctypes.data, len(vp),
            tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            cuts_per_axis, 0.3, beam_width, 0,
            out_wi.ctypes.data,
            out_parts.ctypes.data, out_parts_cap,
            out_vp.ctypes.data, out_vert_cap,
            out_tp.ctypes.data, out_tri_cap)

        assert kerr == 0, f"Kernel error: 0x{kerr:x}"
        assert n_out_parts > 0, "Should create new parts"
        # Each valid cut creates 2 parts
        assert n_out_parts % 2 == 0, "Parts should come in pairs"

    def test_expansion_produces_valid_work_items(self):
        """Output work items should have valid part indices and num_parts."""
        v, t = _l_shape_mesh()
        from scipy.spatial import ConvexHull
        hull = ConvexHull(v)
        hull_verts = v[hull.vertices].copy()
        remap = np.full(len(v), -1, dtype=np.int32)
        for i, vi in enumerate(hull.vertices):
            remap[vi] = i
        hull_tris = np.array([[remap[a], remap[b], remap[c]] for a, b, c in hull.simplices],
                             dtype=np.int32)
        centroid = hull_verts.mean(axis=0)
        for i in range(len(hull_tris)):
            a, b, c = hull_tris[i]
            va, vb, vc = hull_verts[a], hull_verts[b], hull_verts[c]
            normal = np.cross(vb - va, vc - va)
            if np.dot(normal, va - centroid) < 0:
                hull_tris[i, 1], hull_tris[i, 2] = hull_tris[i, 2], hull_tris[i, 1]

        n_v, n_t = len(v), len(t)
        n_hv, n_ht = len(hull_verts), len(hull_tris)
        vp = np.vstack([v, hull_verts]).astype(np.float32)
        tp = np.vstack([t, hull_tris + n_v]).astype(np.int32)

        parts = np.array([_make_part(
            vert_offset=0, vert_count=n_v,
            tri_offset=0, tri_count=n_t,
            hull_vert_offset=n_v, hull_vert_count=n_hv,
            hull_tri_offset=n_t, hull_tri_count=n_ht,
            rv_cost=0.2, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.2)],
                         dtype=WORKITEM_DTYPE)

        beam_width = 4
        cuts_per_axis = 3
        out_parts_cap = 1024
        out_vert_cap = 100000
        out_tri_cap = 100000

        out_wi = np.zeros(beam_width, dtype=WORKITEM_DTYPE)
        out_parts = np.zeros(out_parts_cap, dtype=PARTINFO_DTYPE)
        out_vp = np.zeros((out_vert_cap, 3), dtype=np.float32)
        out_tp = np.zeros((out_tri_cap, 3), dtype=np.int32)

        n_out_parts, n_out_verts, n_out_tris, kerr = _gpu.test_expansion(
            vp.ctypes.data, len(vp),
            tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            cuts_per_axis, 0.3, beam_width, 0,
            out_wi.ctypes.data,
            out_parts.ctypes.data, out_parts_cap,
            out_vp.ctypes.data, out_vert_cap,
            out_tp.ctypes.data, out_tri_cap)

        assert kerr == 0
        # Check that at least one output work item has valid num_parts
        valid_items = [wi for wi in out_wi if wi['num_parts'] > 0]
        assert len(valid_items) > 0, "Should produce at least one valid work item"
        for wi in valid_items:
            assert wi['num_parts'] == 2  # original had 1 part, split into 2
            assert wi['worst_metric'] < 1e29  # should have a real cost
            # Part indices should be valid (>= 1 since part 0 was the input)
            for j in range(wi['num_parts']):
                pi = wi['part_indices'][j]
                assert 1 <= pi < 1 + n_out_parts

    def test_expansion_new_parts_have_rv(self):
        """Newly created parts should have rv_cost > 0."""
        v, t = _l_shape_mesh()
        from scipy.spatial import ConvexHull
        hull = ConvexHull(v)
        hull_verts = v[hull.vertices].copy()
        remap = np.full(len(v), -1, dtype=np.int32)
        for i, vi in enumerate(hull.vertices):
            remap[vi] = i
        hull_tris = np.array([[remap[a], remap[b], remap[c]] for a, b, c in hull.simplices],
                             dtype=np.int32)
        centroid = hull_verts.mean(axis=0)
        for i in range(len(hull_tris)):
            a, b, c = hull_tris[i]
            va, vb, vc = hull_verts[a], hull_verts[b], hull_verts[c]
            normal = np.cross(vb - va, vc - va)
            if np.dot(normal, va - centroid) < 0:
                hull_tris[i, 1], hull_tris[i, 2] = hull_tris[i, 2], hull_tris[i, 1]

        n_v, n_t = len(v), len(t)
        n_hv, n_ht = len(hull_verts), len(hull_tris)
        vp = np.vstack([v, hull_verts]).astype(np.float32)
        tp = np.vstack([t, hull_tris + n_v]).astype(np.int32)

        parts = np.array([_make_part(
            vert_offset=0, vert_count=n_v,
            tri_offset=0, tri_count=n_t,
            hull_vert_offset=n_v, hull_vert_count=n_hv,
            hull_tri_offset=n_t, hull_tri_count=n_ht,
            rv_cost=0.2, hausdorff=-1.0,
        )], dtype=PARTINFO_DTYPE)
        items = np.array([_make_work_item([0], worst_part_idx=0, worst_metric=0.2)],
                         dtype=WORKITEM_DTYPE)

        beam_width = 4
        cuts_per_axis = 3
        out_parts_cap = 1024
        out_vert_cap = 100000
        out_tri_cap = 100000

        out_wi = np.zeros(beam_width, dtype=WORKITEM_DTYPE)
        out_parts = np.zeros(out_parts_cap, dtype=PARTINFO_DTYPE)
        out_vp = np.zeros((out_vert_cap, 3), dtype=np.float32)
        out_tp = np.zeros((out_tri_cap, 3), dtype=np.int32)

        n_out_parts, _, _, kerr = _gpu.test_expansion(
            vp.ctypes.data, len(vp),
            tp.ctypes.data, len(tp),
            parts.ctypes.data, len(parts),
            items.ctypes.data, len(items),
            cuts_per_axis, 0.3, beam_width, 0,
            out_wi.ctypes.data,
            out_parts.ctypes.data, out_parts_cap,
            out_vp.ctypes.data, out_vert_cap,
            out_tp.ctypes.data, out_tri_cap)

        assert kerr == 0
        for i in range(n_out_parts):
            p = out_parts[i]
            assert p['tri_count'] > 0, f"Part {i} has no triangles"
            assert p['vert_count'] > 0, f"Part {i} has no vertices"
            # rv_cost should be set (>= EPS which is ~1e-6)
            assert p['rv_cost'] > 0, f"Part {i} has rv_cost=0"
            assert p['hausdorff'] == pytest.approx(-1.0), f"Part {i} hausdorff should be -1 (not computed)"
