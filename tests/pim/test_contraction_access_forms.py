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
        "dot_0: input 'b' (operand 'y') coordinate 0 reads 2*k, which is not one loop counter"
        in message
    )


def test_shifted_reduction_coordinates_are_refused_on_either_operand():
    with pytest.raises(
        NotImplementedError,
        match=r"input 'b' \(operand 'y'\) coordinate 0 reads k, an effective shift of 128 on a reduction loop",
    ):
        _aim(shifted_vector)
    with pytest.raises(
        NotImplementedError,
        match=r"input 'a' \(operand 'x'\) coordinate 1 reads k, an effective shift of 8 on a reduction loop",
    ):
        _aim(shifted_matrix)


def test_strided_vector_is_refused_on_attacc_and_samsung_by_the_shared_check():
    with pytest.raises(
        NotImplementedError, match="reads 2\\*k, which is not one loop counter"
    ):
        allo.compile(
            strided_attacc,
            build_attacc_target(),
            attacc_cost,
            backend="virtual",
            reduction_order="reassociate",
        )
    with pytest.raises(
        NotImplementedError, match="reads 2\\*k, which is not one loop counter"
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
        match=r"input 'xs' \(operand 'x'\) coordinate 0 reads 32\*p \+ k, which is not one loop counter",
    ):
        _aim(padded_flattened)


@region()
def reduction_lower_bound(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(128, K):
                acc += a[i, k] * b[k]
            c[i] = acc


@region()
def reduction_step_two(W: bfloat16[M, K], x: bfloat16[V], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[V], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(0, K, 2):
                acc += a[i, k] * b[k]
            c[i] = acc


@region()
def result_lower_bound(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[K], c: bfloat16[M]):
        for i in range(16, 32):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[i, k] * b[k]
            c[i] = acc


@region()
def result_offset_single_item(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[K], c: bfloat16[M]):
        for i in range(16):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[16 + i, k] * b[k]
            c[16 + i] = acc


@region()
def attacc_reduction_lower_bound(W: float16[M, K], x: float16[V], y: float16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: float16[M, K], b: float16[V], c: float16[M]):
        for i in range(M):
            acc: float16 = 0
            for k in range(128, K):
                acc += a[i, k] * b[k]
            c[i] = acc


@region()
def attacc_reduction_step_two(W: float16[M, K], x: float16[V], y: float16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: float16[M, K], b: float16[V], c: float16[M]):
        for i in range(M):
            acc: float16 = 0
            for k in range(0, K, 2):
                acc += a[i, k] * b[k]
            c[i] = acc


def test_loop_domains_are_substituted_before_judging_the_access():
    """The reviewer's round 95 probes: ``range(128, K)`` and ``range(0, K, 2)``
    read ``b[k]`` but mean ``b[k + 128]`` and ``b[2*k]``; AiM and AttAcc
    lowered both as the dense ``range(128)`` program (with ``a = 1`` and
    ``b[0:128] = 0``, ``b[128:256] = 1`` the source gives 128 and 64, the
    lowered program 0).  Both are refused on both backends, naming the
    loop domain."""
    with pytest.raises(
        NotImplementedError,
        match=r"input 'b' \(operand 'y'\) coordinate 0 reads k with k in \[128, 256\), an effective shift of 128",
    ):
        _aim(reduction_lower_bound)
    with pytest.raises(
        NotImplementedError,
        match=r"input 'b' \(operand 'y'\) coordinate 0 reads k with k in \[0, 256\) step 2, which is not one loop counter",
    ):
        _aim(reduction_step_two)
    for workload in (attacc_reduction_lower_bound, attacc_reduction_step_two):
        with pytest.raises(
            NotImplementedError, match="unsupported contraction operand access"
        ):
            allo.compile(
                workload,
                build_attacc_target(),
                attacc_cost,
                backend="virtual",
                reduction_order="reassociate",
            )


def test_single_item_result_offsets_are_refused_where_the_backend_stages_from_row_zero():
    """AiM stages one work item's rows from row zero (`AimContraction.row`
    counts items), so ``range(16, 32)`` and ``a[16 + i]`` / ``c[16 + i]`` on
    one item compiled to the rows 0..15 program (`r96/row_offset_probe.log`,
    `domain_probe2.log`); ``AimCtx.REALIZES_RESULT_OFFSETS`` is False and
    both are refused before selection.  Multi-item kernels (the CENT rows'
    ``lW[ch * ROWS + i, k]``) tile the result through the work grid and stay
    accepted (`tests/pim/test_aim_axis_roles.py`)."""
    with pytest.raises(
        NotImplementedError,
        match=r"input 'a' \(operand 'x'\) coordinate 0 reads i with i in \[16, 32\), an effective offset of 16 on a result loop",
    ):
        _aim(result_lower_bound)
    with pytest.raises(
        NotImplementedError,
        match=r"input 'a' \(operand 'x'\) coordinate 0 reads i, an effective offset of 16 on a result loop",
    ):
        _aim(result_offset_single_item)


GM, GR, GN = 64, 16, 2


@region()
def grid_base(W: bfloat16[GM, K], x: bfloat16[K], y: bfloat16[GM]):
    @allo.work(mapping=[GN], args=[W, x, y])
    def dot(a: bfloat16[GM, K], b: bfloat16[K], c: bfloat16[GM]):
        ch = allo.get_wid()
        for i in range(GR):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[ch * GR + i, k] * b[k]
            c[ch * GR + i] = acc


@region()
def grid_shifted(W: bfloat16[GM, K], x: bfloat16[K], y: bfloat16[GM]):
    @allo.work(mapping=[GN], args=[W, x, y])
    def dot(a: bfloat16[GM, K], b: bfloat16[K], c: bfloat16[GM]):
        ch = allo.get_wid()
        for i in range(GR):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[16 + ch * GR + i, k] * b[k]
            c[16 + ch * GR + i] = acc


@region()
def grid_lower_bound(W: bfloat16[GM, K], x: bfloat16[K], y: bfloat16[GM]):
    @allo.work(mapping=[GN], args=[W, x, y])
    def dot(a: bfloat16[GM, K], b: bfloat16[K], c: bfloat16[GM]):
        ch = allo.get_wid()
        for i in range(16, 32):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[ch * GR + i, k] * b[k]
            c[ch * GR + i] = acc


@region()
def grid_strided_items(W: bfloat16[GM, K], x: bfloat16[K], y: bfloat16[GM]):
    @allo.work(mapping=[GN], args=[W, x, y])
    def dot(a: bfloat16[GM, K], b: bfloat16[K], c: bfloat16[GM]):
        ch = allo.get_wid()
        for i in range(GR):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[ch * 32 + i, k] * b[k]
            c[ch * 32 + i] = acc


def test_work_grid_offsets_must_tile_the_realized_frontier():
    """The reviewer's round 96 probe (`reviewer-r96/grid_offsets.py`): on a
    two-item grid AiM stages the result frontier rows 0..31 in work-id
    order, so ``a[ch*R + i]`` is realized as written while
    ``a[16 + ch*R + i]`` (rows 16..47), ``range(16, 32)`` (rows 16..47 as
    well) and ``a[ch*32 + i]`` (rows 0..15 and 32..47) compiled to the same
    program.  The grid's offsets are now proven against the frontier
    (item ``w`` reads offset ``w x R``); the base grid stays accepted and
    the three others are refused before selection."""
    compiled = _aim(grid_base)
    assert len(compiled.trace.matches) == GN
    for workload, offsets in (
        (grid_shifted, r"\[16, 32\]"),
        (grid_lower_bound, r"\[16, 32\]"),
        (grid_strided_items, r"\[0, 32\]"),
    ):
        with pytest.raises(
            NotImplementedError,
            match=rf"input 'a' \(operand 'x'\) coordinate 0 reads .*; the work items' effective offsets "
            rf"{offsets} do not tile the realized frontier \[0, 16\]",
        ):
            _aim(workload)


SM, SR = 64, 16


@region()
def same_buffer_base(W: bfloat16[SM, K], y: bfloat16[SM, SR]):
    @allo.work(mapping=[1], args=[W, y])
    def dot(a: bfloat16[SM, K], c: bfloat16[SM, SR]):
        for i in range(SR):
            for j in range(SR):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[i, k] * a[j, k]
                c[i, j] = acc


@region()
def same_buffer_shifted(W: bfloat16[SM, K], y: bfloat16[SM, SR]):
    @allo.work(mapping=[1], args=[W, y])
    def dot(a: bfloat16[SM, K], c: bfloat16[SM, SR]):
        for i in range(SR):
            for j in range(SR):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[16 + i, k] * a[j, k]
                c[16 + i, j] = acc


@region()
def same_buffer_shifted_reversed(W: bfloat16[SM, K], y: bfloat16[SM, SR]):
    @allo.work(mapping=[1], args=[W, y])
    def dot(a: bfloat16[SM, K], c: bfloat16[SM, SR]):
        for i in range(SR):
            for j in range(SR):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[j, k] * a[16 + i, k]
                c[16 + i, j] = acc


@region()
def same_buffer_grid(W: bfloat16[SM, K], y: bfloat16[SM, SR]):
    @allo.work(mapping=[2], args=[W, y])
    def dot(a: bfloat16[SM, K], c: bfloat16[SM, SR]):
        ch = allo.get_wid()
        for i in range(SR):
            for j in range(SR):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[ch * SR + i, k] * a[j, k]
                c[ch * SR + i, j] = acc


@region()
def same_buffer_grid_shifted(W: bfloat16[SM, K], y: bfloat16[SM, SR]):
    @allo.work(mapping=[2], args=[W, y])
    def dot(a: bfloat16[SM, K], c: bfloat16[SM, SR]):
        ch = allo.get_wid()
        for i in range(SR):
            for j in range(SR):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[16 + ch * SR + i, k] * a[j, k]
                c[16 + ch * SR + i, j] = acc


def test_same_buffer_operands_keep_both_accesses():
    """The reviewer's round 97 probe (`reviewer-r97/access_probes2.py`): the
    result-offset records were keyed by buffer and coordinate, so the two
    operands of ``a[16 + i, k] * a[j, k]`` overwrote each other and the
    shifted operand's offset was lost in one operand order.  The records
    are keyed by operand; the shifted program is refused in both orders,
    the zero-offset neighbour is accepted, and on a grid the shifted
    operand is judged on its own against the frontier."""
    from allo.spmw_semantics import contraction_result_offsets

    compiled = _aim(same_buffer_base)
    fact = compiled.trace.matches[0].extra["semantic_facts"]
    offsets = contraction_result_offsets(fact)
    assert sorted(offsets) == [("x", "a", 0), ("y", "a", 0)]
    assert {value[1] for value in offsets.values()} == {0}
    for workload in (same_buffer_shifted, same_buffer_shifted_reversed):
        with pytest.raises(
            NotImplementedError,
            match=r"input 'a' \(operand '[xy]'\) coordinate 0 reads i, an effective offset of 16 on a result loop",
        ):
            _aim(workload)
    # On a grid AiM's own work-axis derivation refuses a same-buffer kernel
    # (it cannot tell which axis the grid partitions) with an explicit
    # error, after the shared check has judged the accesses: the tiling grid
    # reaches that derivation, the shifted grid is refused first.
    with pytest.raises(ValueError, match="cannot tell which contraction axis"):
        _aim(same_buffer_grid)
    with pytest.raises(
        NotImplementedError,
        match=r"input 'a' \(operand 'x'\) coordinate 0 reads i; the work items' effective offsets \[16, 32\] do not tile the realized frontier \[0, 16\]",
    ):
        _aim(same_buffer_grid_shifted)


DM, DK, DB = 16, 256, 4


@region()
def diagonal_vector(W: bfloat16[DM, DK], x: bfloat16[DK, DK], y: bfloat16[DM]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[DM, DK], b: bfloat16[DK, DK], c: bfloat16[DM]):
        for i in range(DM):
            acc: bfloat16 = 0
            for k in range(DK):
                acc += a[i, k] * b[k, k]
            c[i] = acc


@region()
def diagonal_matrix(W: bfloat16[DM, DM, DK], x: bfloat16[DK], y: bfloat16[DM]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[DM, DM, DK], b: bfloat16[DK], c: bfloat16[DM]):
        for i in range(DM):
            acc: bfloat16 = 0
            for k in range(DK):
                acc += a[i, i, k] * b[k]
            c[i] = acc


@region()
def batched_neighbour(W: bfloat16[DM, DK], X: bfloat16[DB, DK], Y: bfloat16[DB, DM]):
    @allo.work(mapping=[1], args=[W, X, Y])
    def dot(a: bfloat16[DM, DK], x: bfloat16[DB, DK], c: bfloat16[DB, DM]):
        for bb in range(DB):
            for i in range(DM):
                acc: bfloat16 = 0
                for k in range(DK):
                    acc += a[i, k] * x[bb, k]
                c[bb, i] = acc


def test_repeated_loop_variables_across_coordinates_are_refused():
    """The reviewer's round 98 probe (`reviewer-r98/diagonal_probe.py`):
    ``b[k, k]`` passed because each coordinate is a plain loop counter on
    its own, and the AiM lowering staged the contiguous prefix of ``b``
    instead of its diagonal (with ``a`` ones and ``b`` the identity the
    source gives 256 per row, the staged program 1).  A loop variable read
    in two coordinates of one operand is refused per operand; the ordinary
    two-dimensional vector operand of a batched GEMV (``x[bb, k]``, each
    coordinate its own loop) stays accepted with no reported problem."""
    with pytest.raises(
        NotImplementedError,
        match=r"input 'b' \(operand 'y'\) reads loop 'k' in coordinates \[0, 1\], a diagonal",
    ):
        _aim(diagonal_vector)
    with pytest.raises(
        NotImplementedError,
        match=r"input 'a' \(operand 'x'\) reads loop 'i' in coordinates \[0, 1\], a diagonal",
    ):
        _aim(diagonal_matrix)
    compiled = _aim(batched_neighbour)
    fact = compiled.trace.matches[0].extra["semantic_facts"]
    assert contraction_access_problems(fact, reduction_offsets=False) == []
    roles = compiled.trace.matches[0].extra["axis_roles"]
    assert len(roles["batch"]) == 1 and len(roles["output"]) == 1


def test_a_flat_view_of_a_multidimensional_buffer_is_bounded_by_its_element_count():
    """The Gemini-II typed ``block_sum`` record declares one coordinate
    ``b*K + k`` over its logical shape (the card test at `f6b19bd` found the
    round 99 rule comparing that flat extent with the first dimension and
    refusing a correct program; `r105/g2-card-tests.log`).  A flat view is
    bounded by the buffer's element count; one that exceeds it is refused."""
    from types import SimpleNamespace

    from allo.pim.apu_g2_typed_program import APUG2ScalarType as S
    from allo.pim.apu_g2_typed_program import APUG2TypedProgram
    from allo.pim.costs import apu_g2_cost
    from allo.pim.targets import build_apu_g2_target
    from allo.spmw_semantics import BufferAccess, LoopAxis

    program = APUG2TypedProgram(
        "block_sum", (S(8, False),), S(16, False), (1024, 256), 256, repetitions=4
    )
    compiled = allo.compile(
        program, build_apu_g2_target(), apu_g2_cost, backend="virtual"
    )
    fact = compiled.trace.matches[0].extra["semantic_facts"]
    (access,) = fact.inputs
    assert len(access.coords) == 1 and access.shape == (1024, 256)
    assert contraction_access_problems(fact, reduction_offsets=False) == []

    def flat_fact(shape, blocks, k):
        access = BufferAccess(
            role="x",
            memref_name="src",
            parameter="src",
            local=False,
            value_ref=None,
            shape=shape,
            element=None,
            coords=(((("b", k), ("k", 1)), 0),),
            dims=(None,) * len(shape),
            indices=("b*K+k",),
            index_map=None,
        )
        loops = (LoopAxis("b", "b", 0, blocks, 1), LoopAxis("k", "k", 0, k, 1))
        return SimpleNamespace(
            loops=loops,
            loop_vars=("b", "k"),
            reduction_vars=("k",),
            trips={"b": blocks, "k": k},
            loop_labels={"b": "b", "k": "k"},
            inputs=(access,),
        )

    assert (
        contraction_access_problems(
            flat_fact((1024, 256), 1024, 256), reduction_offsets=False
        )
        == []
    )
    problems = contraction_access_problems(
        flat_fact((1024, 256), 1025, 256), reduction_offsets=False
    )
    assert (
        len(problems) == 1
        and "a flat view of 262144 elements (1024x256)" in problems[0]
    )
