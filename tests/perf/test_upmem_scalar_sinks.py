# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A scalar accumulator's output copy is an epilogue, not compaction."""
import numpy as np
import pytest
import allo
from allo.ir.types import int32, int64
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target


def gemv(W: int32[70, 90], x: int32[90], y: int32[70]):
    for i in range(70):
        acc: int32 = 0
        for k in range(90):
            acc += W[i, k] * x[k]
        y[i] = acc


@allo.dataflow.region()
def region_gemv(W: int32[70, 90], x: int32[90], y: int32[70]):
    @allo.work(mapping=[1], args=[y, x, W])
    def body(out: int32[70], vector: int32[90], matrix: int32[70, 90]):
        for i in range(70):
            acc: int32 = 0
            for k in range(90):
                acc += matrix[i, k] * vector[k]
            out[i] = acc


def gemm(A: int32[10, 9], B: int32[9, 7], C: int32[10, 7]):
    for i in range(10):
        for j in range(7):
            acc: int32 = 0
            for k in range(9):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def ttv(A: int32[5, 7, 19], x: int32[19], C: int32[5, 7]):
    for i in range(5):
        for j in range(7):
            acc: int32 = 0
            for k in range(19):
                acc += A[i, j, k] * x[k]
            C[i, j] = acc


@pytest.mark.parametrize(
    "workload,shapes",
    [
        (gemv, ((70, 90), (90,), (70,))),
        (region_gemv, ((70, 90), (90,), (70,))),
        (gemm, ((10, 9), (9, 7), (10, 7))),
        (ttv, ((5, 7, 19), (19,), (5, 7))),
    ],
)
def test_scalar_copy_executes_on_upimulator(workload, shapes):
    rng = np.random.default_rng(9216)
    a, b = [rng.integers(-7, 8, size=s, dtype=np.int32) for s in shapes[:2]]
    out = np.full(shapes[2], 113, np.int32)
    before = (a.copy(), b.copy())
    expected = a.astype(np.int64) @ b.astype(np.int64)
    compiled = allo.compile(workload, build_upmem_target(), upmem_cost)
    result = compiled(a, b, out)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(out, expected.astype(np.int32))
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


def scalar_sum(a: int32[137], out: int64[1]):
    acc: int64 = 0
    for i in range(137):
        acc += a[i]
    out[0] = acc


def direct_sum(a: int32[137], out: int64[1]):
    out[0] = 0
    for i in range(137):
        out[0] += a[i]


def test_scalar_sum_names_launch_domain_boundary_and_direct_form_executes():
    with pytest.raises(
        NotImplementedError, match="scalar-copy sum reduction.*single-domain"
    ):
        allo.compile(scalar_sum, build_upmem_target(), upmem_cost)
    a = np.arange(137, dtype=np.int32) - 41
    before = a.copy()
    out = np.full(1, 991, np.int64)
    result = allo.compile(direct_sum, build_upmem_target(), upmem_cost)(a, out)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(out, [a.astype(np.int64).sum()])
    np.testing.assert_array_equal(a, before)


def two_adds(a: int32[32], b: int32[32], c: int32[32], y: int32[32]):
    for i in range(32):
        y[i] = a[i] + b[i]
    for j in range(32):
        y[j] = y[j] + c[j]


def test_family_mismatch_includes_operation_counts():
    with pytest.raises(NotImplementedError, match=r"matches ops \{'ADD': 2\}"):
        allo.compile(two_adds, build_upmem_target(), upmem_cost, backend="virtual")


def transposed_sink(A: int32[10, 9], B: int32[9, 7], C: int32[7, 10]):
    for i in range(10):
        for j in range(7):
            acc: int32 = 0
            for k in range(9):
                acc += A[i, k] * B[k, j]
            C[j, i] = acc


def test_scalar_copy_cannot_change_output_coordinate_order():
    with pytest.raises(NotImplementedError, match="no UPMEM kernel family"):
        allo.compile(
            transposed_sink, build_upmem_target(), upmem_cost, backend="virtual"
        )


def early_copy(W: int32[70, 90], x: int32[90], y: int32[70]):
    for i in range(70):
        acc: int32 = 0
        y[i] = acc
        for k in range(90):
            acc += W[i, k] * x[k]


def guarded_copy(W: int32[70, 90], x: int32[90], y: int32[70]):
    for i in range(70):
        acc: int32 = 0
        for k in range(90):
            acc += W[i, k] * x[k]
        if x[0] > 0:
            y[i] = acc


@pytest.mark.parametrize("workload", [early_copy, guarded_copy])
def test_copy_must_follow_its_producer_and_be_unconditional(workload):
    with pytest.raises(NotImplementedError, match="no UPMEM kernel family"):
        allo.compile(workload, build_upmem_target(), upmem_cost, backend="virtual")
