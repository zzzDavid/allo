# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release ledger X-42: a contraction operand access the PIM lowerings cannot
realize as written is refused before any backend selects a candidate.

The lowerings stage a contraction operand contiguously from its first
element along each loop (AiM's ``WR_GB`` preload, Samsung's bank staging,
AttAcc's GEMV-buffer write), so ``b[2*k]``, ``b[k + 1]`` or ``a[i, k + 8]``
would be read as the plain access: the reviewer's ``a = 1``, ``b``
alternating 0, 1 gives 128 in the source and 0 in the lowered program.  The
shared check (`spmw_codegen.require_contraction_accesses_realizable`) names
the function, operand, coordinate and what was read; the plain program is
unchanged."""

from __future__ import annotations

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16, float16
from allo.pim.costs import aim_cost, attacc_cost, samsung_cost
from allo.pim.targets import build_aim_target, build_attacc_target, build_samsung_target
from allo.spmw_semantics import contraction_access_problems

M, K, V = 32, 256, 512


@region()
def plain(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[i, k] * b[k]
            c[i] = acc


@region()
def strided_vector(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[i, k] * b[2 * k]
            c[i] = acc


@region()
def shifted_vector(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[i, k] * b[k + 128]
            c[i] = acc


@region()
def shifted_matrix(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(K - 8):
                acc += a[i, k + 8] * b[k]
            c[i] = acc


@region()
def strided_attacc(W: float16[M, K], x: float16[V], y: float16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: float16[M, K], b: float16[V], c: float16[M]):
        for i in range(M):
            acc: float16 = 0
            for k in range(K):
                acc += a[i, k] * b[2 * k]
            c[i] = acc


@region()
def strided_samsung(W: float16[M, K], x: float16[V], y: float16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: float16[M, K], b: float16[V], c: float16[M]):
        for i in range(M):
            acc: float16 = 0
            for k in range(K):
                acc += a[i, k] * b[2 * k]
            c[i] = acc


def _aim(workload):
    return allo.compile(
        workload,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        reduction_order="reassociate",
    )


def test_plain_access_compiles_and_its_facts_report_no_problem():
    compiled = _aim(plain)
    fact = compiled.trace.matches[0].extra["semantic_facts"]
    assert contraction_access_problems(fact, reduction_offsets=False) == []
    assert [str(c) for c in compiled.compiled.cmds][:3] == [
        "AiM WR_GB 16 0 3221225472",
        "AiM WR_BIAS 0 3221225472",
        "AiM MAC_ABK 16 3221225472 0",
    ]


def test_strided_vector_is_refused_before_selection_on_aim():
    with pytest.raises(NotImplementedError) as excinfo:
        _aim(strided_vector)
    message = str(excinfo.value)
    assert "unsupported contraction operand access" in message
    assert (
        "dot_0: input 'b' coordinate 0 reads 2*k, which is not one loop variable"
        in message
    )


def test_shifted_reduction_coordinates_are_refused_on_either_operand():
    with pytest.raises(
        NotImplementedError, match="input 'b' coordinate 0 reads k shifted by 128"
    ):
        _aim(shifted_vector)
    with pytest.raises(
        NotImplementedError, match="input 'a' coordinate 1 reads k shifted by 8"
    ):
        _aim(shifted_matrix)


def test_strided_vector_is_refused_on_attacc_and_samsung_by_the_shared_check():
    with pytest.raises(
        NotImplementedError, match="reads 2\\*k, which is not one loop variable"
    ):
        allo.compile(
            strided_attacc,
            build_attacc_target(),
            attacc_cost,
            backend="virtual",
            reduction_order="reassociate",
        )
    with pytest.raises(
        NotImplementedError, match="reads 2\\*k, which is not one loop variable"
    ):
        allo.compile(
            strided_samsung,
            build_samsung_target(),
            samsung_cost,
            backend="virtual",
            reduction_order="reassociate",
        )


PQ, QQ = 32, 64
D = PQ * QQ


@region()
def dense_flattened(x: bfloat16[D], part: bfloat16[PQ]):
    @allo.work(mapping=[1], args=[x, part])
    def rms(xs: bfloat16[D], out: bfloat16[PQ]):
        for p in range(PQ):
            acc: bfloat16 = 0
            for k in range(QQ):
                acc += xs[p * QQ + k] * xs[p * QQ + k]
            out[p] = acc


@region()
def padded_flattened(x: bfloat16[D], part: bfloat16[PQ]):
    @allo.work(mapping=[1], args=[x, part])
    def rms(xs: bfloat16[D], out: bfloat16[PQ]):
        for p in range(PQ):
            acc: bfloat16 = 0
            for k in range(QQ):
                acc += xs[p * 32 + k] * xs[p * 32 + k]
            out[p] = acc


def test_dense_flattening_is_accepted_and_a_padded_stride_is_refused():
    """``xs[p*Q + k]`` with ``Q`` the trip count of ``k`` addresses the same
    contiguous elements as a two-dimensional access (the CENT decode
    region's RMSNorm partial sums are such accesses); ``xs[p*32 + k]`` with
    ``k`` running to 64 overlaps rows and is not what the lowering stages."""
    compiled = _aim(dense_flattened)
    fact = compiled.trace.matches[0].extra["semantic_facts"]
    assert contraction_access_problems(fact, reduction_offsets=False) == []
    with pytest.raises(
        NotImplementedError,
        match="input 'xs' coordinate 0 reads 32\\*p \\+ k, which is not one loop",
    ):
        _aim(padded_flattened)
