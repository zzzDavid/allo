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


def test_default_selection_follows_the_serialized_fiber_cost(paper):
    # Both fibers share the channel bus (allo/pim/costs/samsung.py): the
    # cost-selected program keeps every tile on the even bank.
    _workload, compiled = paper
    assert [w[:4] + (w[5],) for w in _words(compiled)] == [
        ("MAC", "GRF_B", "GRF_A", "EVEN_BANK", 0),
        ("JUMP", "A_OUT", "A_OUT", "A_OUT", 15),
        ("NOP", "A_OUT", "A_OUT", "A_OUT", 7),
        ("EXIT", "A_OUT", "A_OUT", "A_OUT", 0),
    ]
    (program,) = compiled.compiled.layout_ctx.programs
    assert program.lever_record["grf_residency"] == {"local_x": "host"}
    assert program.lever_record["fibers"] == 1


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
