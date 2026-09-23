"""Tests for the CLI pipeline watchdog (no GPU needed)."""

import os
import multiprocessing as mp

from cuacd.cli import _join_workers, _set_cur_mesh, _get_cur_mesh, _CUR_MESH_BYTES


class _FakeProc:
    """Minimal Process stand-in: alive for `ticks` polls, then exits `code`."""

    def __init__(self, code, ticks_alive=0):
        self._ticks = ticks_alive
        self._code = code
        self.exitcode = None
        self.terminated = False

    def is_alive(self):
        if self._ticks > 0:
            self._ticks -= 1
            return True
        self.exitcode = self._code
        return False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        pass


def test_all_clean_returns_none():
    workers = [("loader", _FakeProc(0)), ("saver", _FakeProc(0))]
    assert _join_workers(workers) is None


def test_dead_worker_reported_and_survivors_terminated():
    survivor = _FakeProc(0, ticks_alive=10)
    dead = _FakeProc(-11)
    assert _join_workers([("processor", dead), ("loader", survivor)]) == \
        ("processor", -11)
    assert survivor.terminated is True


def test_exitcode_checked_after_last_worker_dies():
    # All workers finish on the same poll; the non-zero one must not slip past
    # the "everybody is done" check.
    workers = [("loader", _FakeProc(0)), ("saver", _FakeProc(3))]
    assert _join_workers(workers) == ("saver", 3)


# ---------------------------------------------------------------------------
# Real processes: a crashed consumer leaves its feeder blocked on a full
# bounded queue, which is exactly the deadlock _join_workers must break.
# ---------------------------------------------------------------------------

def _die_after_one_item(queue):
    queue.get()
    os._exit(3)          # hard exit: no traceback, no sentinel


def _block_forever(queue):
    queue.get()


def test_real_deadlock_is_broken_and_reported():
    ctx = mp.get_context("spawn")
    load_queue = ctx.Queue(maxsize=2)
    stall_queue = ctx.Queue(maxsize=2)

    killer = ctx.Process(target=_die_after_one_item, args=(load_queue,))
    blocker = ctx.Process(target=_block_forever, args=(stall_queue,))

    killer.start()
    blocker.start()
    load_queue.put(("mesh.obj", None, None))

    try:
        failure = _join_workers([("processor", killer), ("saver", blocker)])
    finally:
        killer.join()
        blocker.join()

    assert failure == ("processor", 3)
    assert not blocker.is_alive(), "survivor should have been terminated"


def test_cur_mesh_roundtrip():
    ctx = mp.get_context("spawn")
    buf = ctx.Array('c', _CUR_MESH_BYTES)

    _set_cur_mesh(buf, "dir/a.obj")
    assert _get_cur_mesh(buf) == "dir/a.obj"

    # Shorter path must not leave stale tail bytes behind.
    _set_cur_mesh(buf, "b.obj")
    assert _get_cur_mesh(buf) == "b.obj"

    # Over-long path is truncated, not dropped.
    _set_cur_mesh(buf, "x" * (_CUR_MESH_BYTES + 64))
    assert _get_cur_mesh(buf) == "x" * (_CUR_MESH_BYTES - 1)
