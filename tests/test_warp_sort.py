"""Tests for warp_sort_bp32 (GPU warp-cooperative quicksort for BtPoint32)."""
import numpy as np
import pytest

try:
    import coacd_gpu
    import coacd_gpu._gpu as _gpu
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


def _bp32_sort_key(arr):
    """NumPy sort key matching ws_less: sort by (y, x, z)."""
    # arr shape: (N, 4) int32 columns: x, y, z, index
    # Sort by y first, then x, then z
    order = np.lexsort((arr[:, 2], arr[:, 0], arr[:, 1]))
    return arr[order]


def _run_gpu_sort(ctx, arrays):
    """Sort a list of BtPoint32 arrays on GPU using test_warp_sort_kernel.

    Each array is (N, 4) int32: [x, y, z, index].
    Returns list of sorted arrays.
    """
    n_arrays = len(arrays)
    offsets = np.zeros(n_arrays + 1, dtype=np.int32)
    for i, a in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(a)
    total_pts = int(offsets[-1])

    packed = np.concatenate(arrays, axis=0).astype(np.int32).copy()
    assert packed.shape == (total_pts, 4)

    # Flatten to contiguous int32
    flat = packed.reshape(-1).copy()

    _gpu.test_warp_sort(
        flat.ctypes.data, total_pts,
        offsets.ctypes.data, n_arrays)

    packed = flat.reshape(-1, 4)
    result = []
    for i in range(n_arrays):
        s, e = int(offsets[i]), int(offsets[i + 1])
        result.append(packed[s:e].copy())
    return result


def _make_random_bp32(n, rng):
    """Generate random BtPoint32 array: (n, 4) int32."""
    arr = np.empty((n, 4), dtype=np.int32)
    arr[:, 0] = rng.integers(-10000, 10000, size=n)  # x
    arr[:, 1] = rng.integers(-10000, 10000, size=n)  # y
    arr[:, 2] = rng.integers(-10000, 10000, size=n)  # z
    arr[:, 3] = np.arange(n, dtype=np.int32)          # index
    return arr


@pytest.mark.skipif(not _HAS_GPU, reason="coacd_gpu not available")
class TestWarpSort:

    @pytest.fixture(scope="class")
    def ctx(self):
        c = coacd_gpu.Context(device=0)
        yield c
        c.close()

    @pytest.mark.parametrize("n", [1, 2, 5, 16, 31, 32])
    def test_small(self, ctx, n):
        """Sort arrays with <= 32 elements (bitonic path)."""
        rng = np.random.default_rng(42 + n)
        arr = _make_random_bp32(n, rng)
        [result] = _run_gpu_sort(ctx, [arr])
        expected = _bp32_sort_key(arr)
        np.testing.assert_array_equal(result, expected,
            err_msg=f"n={n}: GPU sort != CPU sort")

    @pytest.mark.parametrize("n", [33, 50, 100, 200, 500, 1000, 2000])
    def test_medium(self, ctx, n):
        """Sort arrays with > 32 elements (quicksort path)."""
        rng = np.random.default_rng(123 + n)
        arr = _make_random_bp32(n, rng)
        [result] = _run_gpu_sort(ctx, [arr])
        expected = _bp32_sort_key(arr)
        np.testing.assert_array_equal(result, expected,
            err_msg=f"n={n}: GPU sort != CPU sort")

    def test_batch(self, ctx):
        """Sort multiple arrays in one batch."""
        rng = np.random.default_rng(99)
        sizes = [10, 32, 33, 100, 500]
        arrays = [_make_random_bp32(s, rng) for s in sizes]
        results = _run_gpu_sort(ctx, arrays)
        for i, (arr, res) in enumerate(zip(arrays, results)):
            expected = _bp32_sort_key(arr)
            np.testing.assert_array_equal(res, expected,
                err_msg=f"batch[{i}] n={sizes[i]}: GPU sort != CPU sort")

    def test_already_sorted(self, ctx):
        """Already-sorted input should remain unchanged."""
        arr = np.empty((64, 4), dtype=np.int32)
        arr[:, 1] = np.arange(64)  # y ascending
        arr[:, 0] = 0              # x constant
        arr[:, 2] = 0              # z constant
        arr[:, 3] = np.arange(64)
        [result] = _run_gpu_sort(ctx, [arr])
        np.testing.assert_array_equal(result, arr)

    def test_reverse_sorted(self, ctx):
        """Reverse-sorted input."""
        n = 128
        arr = np.empty((n, 4), dtype=np.int32)
        arr[:, 1] = np.arange(n)[::-1]  # y descending
        arr[:, 0] = 0
        arr[:, 2] = 0
        arr[:, 3] = np.arange(n)
        [result] = _run_gpu_sort(ctx, [arr])
        expected = _bp32_sort_key(arr)
        np.testing.assert_array_equal(result, expected)

    def test_all_equal(self, ctx):
        """All elements equal — should not crash or reorder badly."""
        n = 100
        arr = np.zeros((n, 4), dtype=np.int32)
        arr[:, 3] = np.arange(n)
        [result] = _run_gpu_sort(ctx, [arr])
        # All x,y,z are 0 so order is stable-ish; just check y,x,z columns match
        np.testing.assert_array_equal(result[:, :3], arr[:, :3])

    def test_duplicates(self, ctx):
        """Many duplicate keys — sort is unstable so only check key columns."""
        rng = np.random.default_rng(7)
        n = 200
        arr = np.empty((n, 4), dtype=np.int32)
        arr[:, 0] = rng.integers(0, 3, size=n)
        arr[:, 1] = rng.integers(0, 3, size=n)
        arr[:, 2] = rng.integers(0, 3, size=n)
        arr[:, 3] = np.arange(n)
        [result] = _run_gpu_sort(ctx, [arr])
        expected = _bp32_sort_key(arr)
        # Compare only (x, y, z) — index column order may differ for equal keys
        np.testing.assert_array_equal(result[:, :3], expected[:, :3])
        # But the set of indices must be preserved
        assert sorted(result[:, 3].tolist()) == sorted(expected[:, 3].tolist())
