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


def distributed_gemv(m, k, dpus):
    rows = m // dpus
    last = m - (dpus - 1) * rows

    @allo.dataflow.region()
    def region(W: int32[m, k], x: int32[k], y: int32[m]):
        @allo.work(mapping=[dpus], args=[W, x, y])
        def body(matrix: int32[m, k], vector: int32[k], out: int32[m]):
            (dpu,) = allo.get_wid()
            with allo.meta_if(dpu < dpus - 1):
                for i in range(rows):
                    acc: int32 = 0
                    for j in range(k):
                        acc += matrix[dpu * rows + i, j] * vector[j]
                    out[dpu * rows + i] = acc
            with allo.meta_else():
                for i in range(last):
                    acc: int32 = 0
                    for j in range(k):
                        acc += matrix[m - last + i, j] * vector[j]
                    out[m - last + i] = acc

    return region


@allo.dataflow.region()
def distributed_ttv(A: int32[12, 5, 17], x: int32[17], C: int32[12, 5]):
    @allo.work(mapping=[3], args=[A, x, C])
    def body(matrix: int32[12, 5, 17], vector: int32[17], out: int32[12, 5]):
        (dpu,) = allo.get_wid()
        for i in range(4):
            for j in range(5):
                acc: int32 = 0
                for k in range(17):
                    acc += matrix[dpu * 4 + i, j, k] * vector[k]
                out[dpu * 4 + i, j] = acc


@pytest.mark.parametrize(
    "workload,shapes",
    [
        (distributed_gemv(96, 40, 4), ((96, 40), (40,), (96,))),
        (distributed_gemv(77, 35, 3), ((77, 35), (35,), (77,))),
        (distributed_ttv, ((12, 5, 17), (17,), (12, 5))),
    ],
    ids=["even-gemv", "uneven-gemv", "ttv"],
)
def test_distributed_scalar_copy_binds_each_dpu_window(workload, shapes):
    # Products overflow int32: the independent oracle narrows the int64 sum.
    rng = np.random.default_rng(9217)
    a, b = [rng.integers(-(2**20), 2**20, size=s, dtype=np.int32) for s in shapes[:2]]
    before = (a.copy(), b.copy())
    out = np.full(shapes[2], 113, np.int32)
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    compiled = allo.compile(workload, build_upmem_target(), upmem_cost)
    assert compiled(a, b, out).kind == "simulator"
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


def offset_sink(W: int32[37, 53], x: int32[53], y: int32[38]):
    for i in range(37):
        acc: int32 = 0
        for k in range(53):
            acc += W[i, k] * x[k]
        y[i + 1] = acc


def test_offset_scalar_copy_preserves_untouched_output():
    rng = np.random.default_rng(9218)
    W = rng.integers(-21, 22, size=(37, 53), dtype=np.int32)
    x = rng.integers(-21, 22, size=53, dtype=np.int32)
    y = np.full(38, 113, np.int32)
    expected = np.concatenate(([113], W.astype(np.int64) @ x)).astype(np.int32)
    compiled = allo.compile(offset_sink, build_upmem_target(), upmem_cost)
    assert compiled(W, x, y).kind == "simulator"
    np.testing.assert_array_equal(y, expected)


def reversed_gemv(W: int32[37, 53], x: int32[53], y: int32[37]):
    for i in range(37):
        acc: int32 = 0
        for k in range(53):
            acc += W[i, k] * x[k]
        y[36 - i] = acc


def strided_gemv(W: int32[37, 53], x: int32[53], y: int32[74]):
    for i in range(37):
        acc: int32 = 0
        for k in range(53):
            acc += W[i, k] * x[k]
        y[2 * i] = acc


def reversed_gemm(A: int32[10, 9], B: int32[9, 7], C: int32[10, 7]):
    for i in range(10):
        for j in range(7):
            acc: int32 = 0
            for k in range(9):
                acc += A[i, k] * B[k, j]
            C[9 - i, j] = acc


def strided_gemm(A: int32[10, 9], B: int32[9, 7], C: int32[10, 14]):
    for i in range(10):
        for j in range(7):
            acc: int32 = 0
            for k in range(9):
                acc += A[i, k] * B[k, j]
            C[i, 2 * j] = acc


@pytest.mark.parametrize("backend", [None, "virtual"])
@pytest.mark.parametrize(
    "workload", [reversed_gemv, strided_gemv, reversed_gemm, strided_gemm]
)
def test_scalar_copy_requires_affine_unit_stride_window(workload, backend):
    with pytest.raises(
        NotImplementedError,
        match=r"UPMEM access dimension \d+ is not offset \+ one loop variable",
    ):
        kwargs = {} if backend is None else {"backend": backend}
        allo.compile(workload, build_upmem_target(), upmem_cost, **kwargs)


def test_output_packing_error_names_output_role():
    from allo.pim.upmem_physical import UPMEMMatrixVectorPlan

    plan = UPMEMMatrixVectorPlan(rows=37, columns=53)
    with pytest.raises(ValueError, match="output has 38 values; expected 37"):
        plan.pack_output([0] * 38)
