"""Tests for block-level reductions (B1-B4) against numpy.

Each test creates data sized as a multiple of BLOCK_SIZE=256,
runs the GPU block reduction, and compares per-block results against numpy.
"""

import numpy as np
import pytest

import coacd_gpu._gpu as _gpu

BLOCK_SIZE = 256
N_BLOCKS = 50  # 50 blocks x 256 = 12800 elements
N = N_BLOCKS * BLOCK_SIZE
SEED = 99


@pytest.fixture(scope="module", autouse=True)
def gpu_ctx():
    _gpu.init()
    yield
    _gpu.destroy()


def _ptr(arr):
    return arr.ctypes.data


# -------------------------------------------------------------------------
# B1: block_reduce_sum
# -------------------------------------------------------------------------

class TestBlockReduceSum:
    """Test block_reduce_sum against numpy sum over BLOCK_SIZE chunks."""

    def test_random_data(self):
        rng = np.random.RandomState(SEED + 20)
        data = rng.randn(N).astype(np.float32)
        out = np.zeros(N_BLOCKS, dtype=np.float32)
        _gpu.test_block_reduce_sum(_ptr(data), N, _ptr(out))

        for i in range(N_BLOCKS):
            chunk = data[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            expected = np.sum(chunk)
            np.testing.assert_allclose(
                out[i], expected, atol=1e-2, rtol=1e-4,
                err_msg=f"Block {i}: GPU={out[i]}, numpy={expected}")

    def test_all_zeros(self):
        data = np.zeros(N, dtype=np.float32)
        out = np.zeros(N_BLOCKS, dtype=np.float32)
        _gpu.test_block_reduce_sum(_ptr(data), N, _ptr(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_all_ones(self):
        data = np.ones(N, dtype=np.float32)
        out = np.zeros(N_BLOCKS, dtype=np.float32)
        _gpu.test_block_reduce_sum(_ptr(data), N, _ptr(out))
        np.testing.assert_allclose(out, float(BLOCK_SIZE), atol=1e-3)

    def test_single_block(self):
        data = np.arange(1, BLOCK_SIZE + 1, dtype=np.float32)
        out = np.zeros(1, dtype=np.float32)
        _gpu.test_block_reduce_sum(_ptr(data), BLOCK_SIZE, _ptr(out))
        expected = BLOCK_SIZE * (BLOCK_SIZE + 1) / 2.0
        np.testing.assert_allclose(out[0], expected, atol=1e-1)

    def test_negative_values(self):
        data = np.full(BLOCK_SIZE, -2.0, dtype=np.float32)
        out = np.zeros(1, dtype=np.float32)
        _gpu.test_block_reduce_sum(_ptr(data), BLOCK_SIZE, _ptr(out))
        np.testing.assert_allclose(out[0], -2.0 * BLOCK_SIZE, atol=1e-3)


# -------------------------------------------------------------------------
# B2: block_reduce_bbox
# -------------------------------------------------------------------------

class TestBlockReduceBbox:
    """Test block_reduce_bbox against numpy min/max over vertex groups."""

    def test_single_group(self):
        """Single group of random vertices."""
        rng = np.random.RandomState(SEED + 30)
        n_verts = 200
        verts = rng.randn(n_verts, 3).astype(np.float32)
        offsets = np.array([0], dtype=np.int32)
        counts = np.array([n_verts], dtype=np.int32)
        out = np.zeros(6, dtype=np.float32)

        _gpu.test_block_reduce_bbox(
            _ptr(verts.ravel()), n_verts,
            _ptr(offsets), _ptr(counts), 1, _ptr(out))

        np.testing.assert_allclose(out[0], verts[:, 0].min(), atol=1e-6)
        np.testing.assert_allclose(out[1], verts[:, 1].min(), atol=1e-6)
        np.testing.assert_allclose(out[2], verts[:, 2].min(), atol=1e-6)
        np.testing.assert_allclose(out[3], verts[:, 0].max(), atol=1e-6)
        np.testing.assert_allclose(out[4], verts[:, 1].max(), atol=1e-6)
        np.testing.assert_allclose(out[5], verts[:, 2].max(), atol=1e-6)

    def test_multiple_groups(self):
        """Multiple groups packed contiguously."""
        rng = np.random.RandomState(SEED + 31)
        group_sizes = [100, 50, 300, 10]
        total = sum(group_sizes)
        verts = rng.randn(total, 3).astype(np.float32)

        offsets = np.zeros(len(group_sizes), dtype=np.int32)
        acc = 0
        for i, sz in enumerate(group_sizes):
            offsets[i] = acc
            acc += sz
        counts = np.array(group_sizes, dtype=np.int32)
        n_groups = len(group_sizes)
        out = np.zeros(n_groups * 6, dtype=np.float32)

        _gpu.test_block_reduce_bbox(
            _ptr(verts.ravel()), total,
            _ptr(offsets), _ptr(counts), n_groups, _ptr(out))

        for g in range(n_groups):
            gv = verts[offsets[g]:offsets[g] + counts[g]]
            expected_min = gv.min(axis=0)
            expected_max = gv.max(axis=0)
            np.testing.assert_allclose(
                out[g * 6:g * 6 + 3], expected_min, atol=1e-6,
                err_msg=f"Group {g} min mismatch")
            np.testing.assert_allclose(
                out[g * 6 + 3:g * 6 + 6], expected_max, atol=1e-6,
                err_msg=f"Group {g} max mismatch")

    def test_large_group_exceeding_block_size(self):
        """Group with more vertices than BLOCK_SIZE (tests strided loop)."""
        rng = np.random.RandomState(SEED + 32)
        n_verts = BLOCK_SIZE * 3 + 17  # 785 vertices
        verts = rng.randn(n_verts, 3).astype(np.float32)
        offsets = np.array([0], dtype=np.int32)
        counts = np.array([n_verts], dtype=np.int32)
        out = np.zeros(6, dtype=np.float32)

        _gpu.test_block_reduce_bbox(
            _ptr(verts.ravel()), n_verts,
            _ptr(offsets), _ptr(counts), 1, _ptr(out))

        np.testing.assert_allclose(out[0], verts[:, 0].min(), atol=1e-6)
        np.testing.assert_allclose(out[1], verts[:, 1].min(), atol=1e-6)
        np.testing.assert_allclose(out[2], verts[:, 2].min(), atol=1e-6)
        np.testing.assert_allclose(out[3], verts[:, 0].max(), atol=1e-6)
        np.testing.assert_allclose(out[4], verts[:, 1].max(), atol=1e-6)
        np.testing.assert_allclose(out[5], verts[:, 2].max(), atol=1e-6)

    def test_known_bbox(self):
        """Unit cube vertices -> bbox [0,0,0] to [1,1,1]."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float32)
        offsets = np.array([0], dtype=np.int32)
        counts = np.array([8], dtype=np.int32)
        out = np.zeros(6, dtype=np.float32)

        _gpu.test_block_reduce_bbox(
            _ptr(verts.ravel()), 8,
            _ptr(offsets), _ptr(counts), 1, _ptr(out))

        np.testing.assert_allclose(out[:3], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(out[3:], [1, 1, 1], atol=1e-6)

    def test_offset_groups(self):
        """Groups at nonzero offsets in the vertex buffer."""
        rng = np.random.RandomState(SEED + 33)
        # Padding + group A (20 verts) + gap + group B (30 verts)
        total = 100
        verts = rng.randn(total, 3).astype(np.float32)
        offsets = np.array([10, 60], dtype=np.int32)
        counts = np.array([20, 30], dtype=np.int32)
        out = np.zeros(12, dtype=np.float32)

        _gpu.test_block_reduce_bbox(
            _ptr(verts.ravel()), total,
            _ptr(offsets), _ptr(counts), 2, _ptr(out))

        for g in range(2):
            gv = verts[offsets[g]:offsets[g] + counts[g]]
            np.testing.assert_allclose(
                out[g * 6:g * 6 + 3], gv.min(axis=0), atol=1e-6)
            np.testing.assert_allclose(
                out[g * 6 + 3:g * 6 + 6], gv.max(axis=0), atol=1e-6)


# -------------------------------------------------------------------------
# B3: block_reduce_max
# -------------------------------------------------------------------------

class TestBlockReduceMax:
    """Test block_reduce_max against numpy max over BLOCK_SIZE chunks."""

    def test_random_data(self):
        rng = np.random.RandomState(SEED)
        data = rng.randn(N).astype(np.float32)
        out = np.zeros(N_BLOCKS, dtype=np.float32)
        _gpu.test_block_reduce_max(_ptr(data), N, _ptr(out))

        for i in range(N_BLOCKS):
            chunk = data[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            expected = np.max(chunk)
            np.testing.assert_allclose(
                out[i], expected, atol=1e-6,
                err_msg=f"Block {i}: GPU={out[i]}, numpy={expected}")

    def test_all_negative(self):
        data = np.full(N, -5.0, dtype=np.float32)
        data[BLOCK_SIZE * 3 + 17] = -1.0  # max in block 3
        out = np.zeros(N_BLOCKS, dtype=np.float32)
        _gpu.test_block_reduce_max(_ptr(data), N, _ptr(out))

        assert out[3] == pytest.approx(-1.0)
        for i in range(N_BLOCKS):
            if i != 3:
                assert out[i] == pytest.approx(-5.0)

    def test_single_block(self):
        data = np.arange(BLOCK_SIZE, dtype=np.float32)
        out = np.zeros(1, dtype=np.float32)
        _gpu.test_block_reduce_max(_ptr(data), BLOCK_SIZE, _ptr(out))
        assert out[0] == pytest.approx(float(BLOCK_SIZE - 1))

    def test_max_at_various_positions(self):
        """Max element at different thread positions within a block."""
        rng = np.random.RandomState(SEED + 1)
        for pos in [0, 1, 127, 128, 255]:
            data = np.full(BLOCK_SIZE, -10.0, dtype=np.float32)
            data[pos] = 42.0
            out = np.zeros(1, dtype=np.float32)
            _gpu.test_block_reduce_max(_ptr(data), BLOCK_SIZE, _ptr(out))
            assert out[0] == pytest.approx(42.0), f"Failed with max at pos {pos}"


# -------------------------------------------------------------------------
# B4: block_reduce_count
# -------------------------------------------------------------------------

class TestBlockReduceCount:
    """Test block_reduce_count against numpy sum over BLOCK_SIZE chunks."""

    def test_random_flags(self):
        rng = np.random.RandomState(SEED + 10)
        flags = rng.randint(0, 2, size=N).astype(np.int32)
        out = np.zeros(N_BLOCKS, dtype=np.int32)
        _gpu.test_block_reduce_count(_ptr(flags), N, _ptr(out))

        for i in range(N_BLOCKS):
            chunk = flags[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            expected = int(np.sum(chunk != 0))
            assert out[i] == expected, \
                f"Block {i}: GPU={out[i]}, numpy={expected}"

    def test_all_zeros(self):
        flags = np.zeros(N, dtype=np.int32)
        out = np.zeros(N_BLOCKS, dtype=np.int32)
        _gpu.test_block_reduce_count(_ptr(flags), N, _ptr(out))
        assert np.all(out == 0)

    def test_all_ones(self):
        flags = np.ones(N, dtype=np.int32)
        out = np.zeros(N_BLOCKS, dtype=np.int32)
        _gpu.test_block_reduce_count(_ptr(flags), N, _ptr(out))
        assert np.all(out == BLOCK_SIZE)

    def test_nonzero_values_count_as_one(self):
        """Flags with values > 1 should still count as 1."""
        flags = np.array([0, 1, 2, 3, 0, 5, 0, 100] + [0] * (BLOCK_SIZE - 8),
                         dtype=np.int32)
        out = np.zeros(1, dtype=np.int32)
        _gpu.test_block_reduce_count(_ptr(flags), BLOCK_SIZE, _ptr(out))
        assert out[0] == 5  # indices 1,2,3,5,7 are nonzero

    def test_single_set_bit(self):
        """One flag set per block at varying positions."""
        for pos in [0, 1, 127, 128, 255]:
            flags = np.zeros(BLOCK_SIZE, dtype=np.int32)
            flags[pos] = 1
            out = np.zeros(1, dtype=np.int32)
            _gpu.test_block_reduce_count(_ptr(flags), BLOCK_SIZE, _ptr(out))
            assert out[0] == 1, f"Failed with flag at pos {pos}"
