# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung lowering contract without the simulator.

Contract: docs/release/design/samsung-execution.md.  The emitted CRF is the
per-pass program the conductor uploads; these tests pin that it follows the
selected placement, that unrealizable levers are rejected at
materialization, and that operand roles are structural (S-05).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.schedule_search import InfeasibleSchedule
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import _samsung_enumerate
from allo.spmw_samsung import (
    SamsungUnsupported,
    samsung_reduce_plan,
    samsung_target_facts,
    samsung_vendor_gemv_layout,
)

M, K, ROWS = 4096, 256, 32
GEOMETRY = {
    "channels": 64,
    "pim_blocks": 8,
    "grf_a": 8,
    "grf_b": 8,
    "lanes": 16,
    "crf_words": 32,
    "banks": 16,
    "cols_per_row": 32,
    "rows": 16384,
}


def _workloads():
    path = Path(__file__).resolve().parent / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_lowering_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


@_df_region()
def swapped_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_x[k] * local_W[row0 + i, k]
            local_y[row0 + i] = acc


@_df_region()
def half_grid_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 4], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 4 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


@_df_region()
def dot(a: fp16[K], b: fp16[K], y: fp16[1]):
    @allo.work(mapping=[1], args=[a, b, y])
    def kernel(la: fp16[K], lb: fp16[K], ly: fp16[1]):
        for i in range(1):
            acc: fp16 = 0
            for k in range(K):
                acc += la[k] * lb[k]
            ly[i] = acc


@_df_region()
def partitioned_add(a: fp16[131072], b: fp16[131072], c: fp16[131072]):
    @allo.work(mapping=[16, 8], args=[a, b, c])
    def kernel(la: fp16[131072], lb: fp16[131072], lc: fp16[131072]):
        pid, uid = allo.get_wid()
        base = (pid * 8 + uid) * 1024
        for i in range(1024):
            lc[base + i] = la[base + i] + lb[base + i]


@_df_region()
def flat_relu(a: fp16[131072], c: fp16[131072]):
    @allo.work(mapping=[1], args=[a, c])
    def kernel(la: fp16[131072], lc: fp16[131072]):
        for i in range(131072):
            lc[i] = max(la[i], 0.0)


@_df_region()
def flat_mad(a: fp16[131072], b: fp16[131072], d: fp16[131072], c: fp16[131072]):
    @allo.work(mapping=[1], args=[a, b, d, c])
    def kernel(la: fp16[131072], lb: fp16[131072], ld: fp16[131072], lc: fp16[131072]):
        for i in range(131072):
            lc[i] = la[i] * lb[i] + ld[i]


def _words(compiled):
    return [
        (c.type_, c.dst_, c.src0_, c.src1_, c.src2_, c.loopCounter_, c.isAuto_)
        for c in compiled.compiled.cmds
    ]


VENDOR_K256 = [
    ("MAC", "GRF_B", "GRF_A", "EVEN_BANK", "A_OUT", 0, 1),
    ("JUMP", "A_OUT", "A_OUT", "A_OUT", "A_OUT", 7, 0),
    ("MAC", "GRF_B", "GRF_A", "ODD_BANK", "A_OUT", 0, 1),
    ("JUMP", "A_OUT", "A_OUT", "A_OUT", "A_OUT", 7, 0),
    ("NOP", "A_OUT", "A_OUT", "A_OUT", "A_OUT", 7, 0),
    ("EXIT", "A_OUT", "A_OUT", "A_OUT", "A_OUT", 0, 0),
]


@pytest.fixture(scope="module")
def paper():
    workload, _M, _K = _workloads()["gemv_m4096_k256"]
    # The REDUCE placements regroup the FP16 reduction (lane partial sums):
    # their lowering is exercised under explicit permission.
    return workload, allo.compile(
        workload, build_samsung_target(), samsung_cost, reduction_order="reassociate"
    )


def _forced(workload, compiled, mode, residency, issue="shared"):
    target = compiled.target
    for candidate in _samsung_enumerate(target, compiled.trace.matches[:1]):
        modes = set((candidate.extra.get("grf_residency") or {}).values()) or {"crf"}
        if (
            candidate.mode.split("+")[0] == mode
            and candidate.extra.get("crf_issue") == issue
            and modes == {residency}
            and not candidate.extra.get("stage_resident")
        ):
            n = compiled.placement_ranking.kernels[0].bucket_count
            return allo.compile(
                workload,
                target,
                samsung_cost,
                layout=[candidate] * n,
                reduction_order="reassociate",
            )
    raise KeyError((mode, residency, issue))


@pytest.fixture(scope="module")
def vendor(paper):
    workload, compiled = paper
    target = compiled.target
    return allo.compile(
        workload,
        target,
        samsung_cost,
        layout=samsung_vendor_gemv_layout(target, compiled.trace),
        reduction_order="reassociate",
    )


def test_vendor_layout_emits_the_vendor_gemv_program(vendor):
    assert _words(vendor) == VENDOR_K256
    (program,) = vendor.compiled.layout_ctx.programs
    assert program.kind == "REDUCE"
    assert (program.k_tiles, program.tile_fiber, program.rows_per_block) == (
        2,
        (0, 1),
        8,
    )
    assert program.vector_from_bank is False
    assert program.lever_record["grf_residency"] == {"local_x": "host"}
    assert program.lever_record["fibers"] == 2
    assert dict(vendor._source_binding.inputs) == {"A": "W", "B": "x"}


def test_default_selection_splits_each_tiles_rows_across_the_fibers(paper):
    # The estimate includes each work item's share of the matrix staging
    # (allo/pim/costs/samsung.py, S-16): the cost-selected program places a
    # tile's second row in the odd fiber (`split_rows`, the measured fastest
    # placement on the paper GEMVs), so each tile is two MACs, one per bank,
    # and the loop runs over half the tile count.
    _workload, compiled = paper
    assert [w[:4] + (w[5],) for w in _words(compiled)] == [
        ("MAC", "GRF_B", "GRF_A", "EVEN_BANK", 0),
        ("MAC", "GRF_B", "GRF_A", "ODD_BANK", 0),
        ("JUMP", "A_OUT", "A_OUT", "A_OUT", 7),
        ("NOP", "A_OUT", "A_OUT", "A_OUT", 7),
        ("EXIT", "A_OUT", "A_OUT", "A_OUT", 0),
    ]
    (program,) = compiled.compiled.layout_ctx.programs
    assert program.lever_record["grf_residency"] == {"local_x": "host"}
    assert program.lever_record["fibers"] == 2
    assert program.lever_record["tile_rows"] == "split_fibers"
    assert program.lever_record["mode"].split("+")[0] == "split_rows"


def test_k512_program_derives_trip_counts_from_tiles():
    workload, _M, _K = _workloads()["gemv_m4096_k512"]
    target = build_samsung_target()
    default = allo.compile(
        workload, target, samsung_cost, reduction_order="reassociate"
    )
    compiled = allo.compile(
        workload,
        target,
        samsung_cost,
        layout=samsung_vendor_gemv_layout(target, default.trace),
        reduction_order="reassociate",
    )
    words = _words(compiled)
    assert [w[5] for w in words if w[0] == "JUMP"] == [15, 15]
    (program,) = compiled.compiled.layout_ctx.programs
    assert program.tile_fiber == (0, 1, 0, 1)


def test_operand_order_does_not_change_decisions(paper):
    _workload, compiled = paper
    swapped = allo.compile(
        swapped_gemv,
        build_samsung_target(),
        samsung_cost,
        reduction_order="reassociate",
    )
    assert _words(swapped) == _words(compiled)
    assert dict(swapped._source_binding.inputs) == {"A": "W", "B": "x"}
    (a,) = compiled.compiled.layout_ctx.programs
    (b,) = swapped.compiled.layout_ctx.programs
    assert (a.k_tiles, a.tile_fiber, a.vector_from_bank, a.binds) == (
        b.k_tiles,
        b.tile_fiber,
        b.vector_from_bank,
        b.binds,
    )
    assert a.contraction.matrix_memref == b.contraction.matrix_memref == "local_W"
    assert a.contraction.vector_memref == b.contraction.vector_memref == "local_x"


def test_bank_row_and_crf_fill_change_the_emitted_program(paper):
    workload, compiled = paper
    bank_row = _forced(workload, compiled, "bank_row", "host")
    assert [w[:4] + (w[5],) for w in _words(bank_row)] == [
        ("MAC", "GRF_B", "GRF_A", "EVEN_BANK", 0),
        ("JUMP", "A_OUT", "A_OUT", "A_OUT", 15),
        ("NOP", "A_OUT", "A_OUT", "A_OUT", 7),
        ("EXIT", "A_OUT", "A_OUT", "A_OUT", 0),
    ]
    crf_fill = _forced(workload, compiled, "dual_fiber", "crf")
    assert [w[0] for w in _words(crf_fill)] == [
        "FILL",
        "MAC",
        "JUMP",
        "FILL",
        "MAC",
        "JUMP",
        "NOP",
        "EXIT",
    ]
    (program,) = crf_fill.compiled.layout_ctx.programs
    assert program.vector_from_bank is True
    assert [b[1] for b in program.binds] == [
        "vector",
        "matrix",
        "vector",
        "matrix",
        "drain",
    ]


def test_unrealizable_levers_are_rejected_at_materialization(paper):
    workload, compiled = paper
    with pytest.raises(
        InfeasibleSchedule, match="grf_staged places the matrix in a GRF"
    ):
        _forced(workload, compiled, "grf_staged", "host")
    with pytest.raises(InfeasibleSchedule, match="lockstep"):
        _forced(workload, compiled, "dual_fiber", "host", issue="per_workid")


def test_reduce_plan_places_tiles_by_selected_fibers(vendor):
    (program,) = vendor.compiled.layout_ctx.programs
    plan = samsung_reduce_plan(program, GEOMETRY, columns=1)
    assert (plan["passes"], plan["slots"], plan["padded_k"]) == (1, 4096, 256)
    lines = plan["text"].splitlines()
    assert "tile_fiber 0 1" in lines and "vector_from_bank 0" in lines
    assert [line for line in lines if line.startswith("bind")] == [
        "bind 0 matrix 0 -1 0",
        "bind 2 matrix 1 -1 0",
        "bind 4 drain 1 -1 0",
    ]


def test_ambiguous_operands_are_rejected():
    with pytest.raises(
        NotImplementedError, match="cannot tell the matrix from the vector"
    ):
        allo.compile(dot, build_samsung_target(), samsung_cost)


def test_partial_row_coverage_is_an_error():
    with pytest.raises(ValueError, match="cover 2048 output rows but the matrix"):
        allo.compile(half_grid_gemv, build_samsung_target(), samsung_cost)


def test_elementwise_needs_a_provable_element_index():
    # Work item (0, 0) reads ``la[0 + i]`` over 1024 of 131072 elements; the
    # others read ``la[base + i]``: either way the lowering cannot walk the
    # whole extent with one item and says so (resolved coordinates, C-24).
    with pytest.raises(SamsungUnsupported, match="walks the whole 1-D FP16 extent"):
        allo.compile(partitioned_add, build_samsung_target(), samsung_cost)


def test_elementwise_programs_use_declared_moves():
    relu = allo.compile(flat_relu, build_samsung_target(), samsung_cost)
    words = relu.compiled.cmds
    movs = [c for c in words if c.type_ == "MOV"]
    assert len(movs) == 16 and all(c.isRelu_ for c in movs)
    assert [c.dstIdx_ for c in movs[:8]] == list(range(8))
    assert {c.dst_ for c in movs[:8]} == {"GRF_A"} and {c.dst_ for c in movs[8:]} == {
        "GRF_B"
    }

    mad = allo.compile(flat_mad, build_samsung_target(), samsung_cost)
    (program,) = mad.compiled.layout_ctx.programs
    assert program.swap_parity == (False, False, True)
    assert [c.type_ for c in mad.compiled.cmds] == [
        "FILL",
        "FILL",
        "MAD",
        "NOP",
        "FILL",
        "FILL",
        "MAD",
        "NOP",
        "EXIT",
    ]
    assert dict(mad._source_binding.inputs) == {"IN0": "a", "IN1": "b", "IN2": "d"}


def test_target_facts_raise_instead_of_guessing():
    with pytest.raises(ValueError, match="lacks the Samsung declarations"):
        samsung_target_facts(SimpleNamespace(name="partial", _walk=lambda: []))
    facts = samsung_target_facts(build_samsung_target())
    assert (
        facts.pim_blocks,
        facts.banks_per_pim,
        facts.grf_a_entries,
        facts.lanes,
    ) == (
        8,
        2,
        8,
        16,
    )


GEMM_N = 4


@_df_region()
def gemm_vector_first(A: fp16[M, K], B: fp16[K, GEMM_N], C: fp16[M, GEMM_N]):
    @allo.work(mapping=[16, 8], args=[A, B, C])
    def kernel(lA: fp16[M, K], lB: fp16[K, GEMM_N], lC: fp16[M, GEMM_N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(GEMM_N):
                acc: fp16 = 0
                for k in range(K):
                    acc += lB[k, j] * lA[row0 + i, k]
                lC[row0 + i, j] = acc


@_df_region()
def gemm_matrix_first(A: fp16[M, K], B: fp16[K, GEMM_N], C: fp16[M, GEMM_N]):
    @allo.work(mapping=[16, 8], args=[A, B, C])
    def kernel(lA: fp16[M, K], lB: fp16[K, GEMM_N], lC: fp16[M, GEMM_N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(GEMM_N):
                acc: fp16 = 0
                for k in range(K):
                    acc += lA[row0 + i, k] * lB[k, j]
                lC[row0 + i, j] = acc


def test_gemm_operand_order_does_not_change_the_selected_program():
    """Release ledger X-40: the candidate contract named the bank-resident
    matrix after the shared facts' ``matrix`` role, which breaks the GEMM
    tie (two inputs each indexed by the reduction loop and one result loop)
    by pattern order.  With the vector written first, every REDUCE candidate
    declared the matrix packing on the GRF_A operand, the realization check
    rejected all of them ("declared packing None") and selection fell back to
    the TEMPORAL program at 9.8x the cycles.  The bank-resident operand is
    now the one the REDUCE lowering stages into the banks, so both operand
    orders select the same program."""
    from allo.spmw_candidate import candidate_effects

    target = build_samsung_target()
    compiled = {}
    for name, workload in (
        ("vector_first", gemm_vector_first),
        ("matrix_first", gemm_matrix_first),
    ):
        compiled[name] = allo.compile(
            workload, target, samsung_cost, reduction_order="reassociate"
        )
    programs = {name: c.compiled.layout_ctx.programs[0] for name, c in compiled.items()}
    kernels = {name: c.placement_ranking.kernels[0] for name, c in compiled.items()}
    for name, program in programs.items():
        assert program.kind == "REDUCE", (name, program.kind)
        assert program.contraction.matrix_memref == "lA", name
        assert program.contraction.vector_memref == "lB", name
        assert program.weight_resident is True, name
        kernel = kernels[name]
        chosen = [s for s in kernel.scores if s[0] == kernel.chosen_position]
        assert chosen and chosen[0][2].startswith("split_rows"), (name, chosen)
        # The selected candidate declares the matrix packing on the banks
        # operand, the realization check accepted it, and no candidate was
        # rejected for its declared packing.
        layout = next(
            pl for pos, pl in kernel.candidates if pos == kernel.chosen_position
        )
        effects = candidate_effects(layout)
        declared = {v.value: (v.storage, v.packing) for v in effects.value_layouts}
        realized = "half_bank:{},{}".format(*program.half_fibers)
        assert declared["lA"] == ("banks", realized), (declared, realized)
        assert declared["lB"][0] == "grf_a", declared
        assert not [r for r in kernel.rejections if "declared packing" in r[1]], name
    assert _words(compiled["vector_first"]) == _words(compiled["matrix_first"])
    a, b = programs["vector_first"], programs["matrix_first"]
    assert (a.k_tiles, a.tile_fiber, a.binds, a.half_fibers) == (
        b.k_tiles,
        b.tile_fiber,
        b.binds,
        b.half_fibers,
    )
    assert kernels["vector_first"].scores[0][1] == kernels["matrix_first"].scores[0][1]
    chosen = {
        name: [s for s in k.scores if s[0] == k.chosen_position][0][1:]
        for name, k in kernels.items()
    }
    assert chosen["vector_first"] == chosen["matrix_first"], chosen


# --------------------------------------------------------------------------- #
# Round 156 (G1 claim 8 review): a body that stores a GEMV row and then an
# elementwise product of that row is outside the released Samsung domain (a
# kernel is one contraction or one elementwise op).  The refusal must name
# that rule; before the round, role binding across the mixed matches raised
# first with an internal-inconsistency message blaming the matcher.
# --------------------------------------------------------------------------- #

_MX_M, _MX_K, _MX_R = 64, 64, 4


@_df_region()
def _mixed_gemv_then_scale(
    W: fp16[_MX_M, _MX_K],
    x: fp16[_MX_K],
    s: fp16[_MX_M],
    y: fp16[_MX_M],
    z: fp16[_MX_M],
):
    @allo.work(mapping=[16], args=[W, x, s, y, z])
    def body(
        lW: fp16[_MX_M, _MX_K],
        lx: fp16[_MX_K],
        ls: fp16[_MX_M],
        ly: fp16[_MX_M],
        lz: fp16[_MX_M],
    ):
        (c,) = allo.get_wid()
        for i in range(_MX_R):
            acc: fp16 = 0
            for k in range(_MX_K):
                acc += lW[c * _MX_R + i, k] * lx[k]
            ly[c * _MX_R + i] = acc
        for i in range(_MX_R):
            lz[c * _MX_R + i] = ly[c * _MX_R + i] * ls[c * _MX_R + i]


def test_mixed_contraction_and_elementwise_body_is_refused_by_the_domain_rule():
    with pytest.raises(SamsungUnsupported, match="mixes ops") as info:
        allo.compile(
            _mixed_gemv_then_scale,
            build_samsung_target(),
            samsung_cost,
            backend="virtual",
        )
    message = str(info.value)
    assert "MAC" in message and "MUL" in message
    assert "a kernel is one contraction or one elementwise op" in message
    assert "binds to multiple memrefs" not in message
