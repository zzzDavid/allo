# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

import allo
from allo.ir.types import int32

N = 4
T = 3
M = 5


def test_static_length_slice_with_loop_carried_offset():
    def kernel(A: int32[T * N]) -> int32[T * N]:
        out: int32[T * N] = 0
        for t in range(T):
            r: int32 = t * N
            tile: int32[N] = A[r : r + N]
            out[r : r + N] = tile
        return out

    s = allo.customize(kernel)
    assert re.search(rf"memref\.subview .*\[{N}\]", str(s.module)), str(s.module)
    mod = s.build()
    A = np.random.randint(-100, 100, size=(T * N,)).astype(np.int32)
    np.testing.assert_array_equal(mod(A), A)


def test_static_length_row_slice_with_scalar_offset():
    def kernel(W: int32[T * N, M], row0: int32) -> int32[N, M]:
        tile: int32[N, M] = W[row0 : row0 + N, :]
        return tile

    s = allo.customize(kernel)
    assert re.search(rf"memref\.subview .*\[{N}, {M}\]", str(s.module)), str(
        s.module
    )
    mod = s.build()
    W = np.random.randint(-100, 100, size=(T * N, M)).astype(np.int32)
    for row0 in (0, 3, (T - 1) * N):
        np.testing.assert_array_equal(mod(W, row0), W[row0 : row0 + N, :])


if __name__ == "__main__":
    pytest.main([__file__])
