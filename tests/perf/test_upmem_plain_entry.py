# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain UPMEM entries use the same positional caller ABI as shared plans."""
import numpy as np
import pytest
import allo
from allo.ir.types import int32
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target


def add(a: int32[1024], b: int32[1024], y: int32[1024]):
    for i in range(1024):
        y[i] = a[i] + b[i]


def gemv(W: int32[96, 128], x: int32[128], y: int32[96]):
    for i in range(96):
        y[i] = 0
        for k in range(128):
            y[i] += W[i, k] * x[k]


def gemm(A: int32[24, 16], B: int32[16, 12], C: int32[24, 12]):
    for i in range(24):
        for j in range(12):
            for k in range(16):
                C[i, j] += A[i, k] * B[k, j]


@pytest.mark.parametrize(
    "workload,shapes",
    [
        (add, ((1024,), (1024,), (1024,))),
        (gemv, ((96, 128), (128,), (96,))),
        (gemm, ((24, 16), (16, 12), (24, 12))),
    ],
)
@pytest.mark.parametrize("backend", ["virtual", "simulator"])
def test_plain_entry_executes_with_positional_arguments(workload, shapes, backend):
    rng = np.random.default_rng(9215)
    a, b = [rng.integers(-9, 10, size=s, dtype=np.int32) for s in shapes[:2]]
    out = np.full(shapes[2], 37, np.int32)
    if workload is add:
        expected = a.astype(np.int64) + b
    else:
        expected = a.astype(np.int64) @ b
        if workload is gemm:
            expected += out
    # Default public execution selects uPIMulator. Virtual remains explicitly
    # labelled as a reference/analytical result, not simulated numerical work.
    options = {"backend": "virtual"} if backend == "virtual" else {}
    compiled = allo.compile(workload, build_upmem_target(), upmem_cost, **options)
    result = compiled(a, b, out)
    assert result.kind == backend
    if backend == "virtual":
        np.testing.assert_array_equal(out, np.full(shapes[2], 37, np.int32))
    else:
        np.testing.assert_array_equal(out, expected.astype(np.int32))


@allo.dataflow.region()
def reordered_region(a: int32[1024], b: int32[1024], y: int32[1024]):
    @allo.work(mapping=[1], args=[y, b, a])
    def body(dst: int32[1024], rhs: int32[1024], lhs: int32[1024]):
        for i in range(1024):
            dst[i] = lhs[i] - rhs[i]


def test_region_keeps_retained_argument_order():
    compiled = allo.compile(reordered_region, build_upmem_target(), upmem_cost)
    a = np.arange(1024, dtype=np.int32)
    b = np.full(1024, 19, np.int32)
    out = np.full(1024, -123, np.int32)
    result = compiled(a, b, out)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(out, a - b)
    np.testing.assert_array_equal(a, np.arange(1024, dtype=np.int32))
    np.testing.assert_array_equal(b, np.full(1024, 19, np.int32))
