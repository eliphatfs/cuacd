"""Regression guard for host-side readbacks of the big device structs.

`struct LaDecompState_h` is ~1.25 MiB (16k parts x 80 B).  Staging it as a
stack-local copy used to overrun a Windows thread stack (1 MiB default) and
kill the process with no message — Linux got away with it only because its
stacks are 8 MiB.  This runs the readback paths on a 1 MiB stack so Linux CI
fails loudly on the same bug Windows hits in the field.
"""

import threading

import numpy as np
import pytest

import cuacd

_WINDOWS_STACK = 1 << 20  # default thread stack size on Windows


@pytest.mark.parametrize("verbose", [0, 1])
def test_readback_fits_windows_stack(l_shape, verbose):
    """lookahead_decompose must not need more than a 1 MiB stack."""
    verts, tris = l_shape
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int32)

    result = {}

    def run():
        try:
            ctx = cuacd.Context(device=0)
            try:
                parts = ctx.lookahead_decompose(
                    verts, tris, verbose=verbose, merge_hulls=1)
            finally:
                ctx.close()
            result['nparts'] = len(parts)
        except Exception as e:  # surface failures instead of dying in-thread
            result['error'] = e

    old = threading.stack_size(_WINDOWS_STACK)
    try:
        t = threading.Thread(target=run)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)

    if t.is_alive():
        pytest.fail("worker thread hung; likely a hard crash was swallowed")
    if 'error' in result:
        raise result['error']
    assert result['nparts'] >= 2
