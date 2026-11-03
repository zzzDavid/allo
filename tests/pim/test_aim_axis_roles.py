# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM contraction axis roles, logical-work coverage and unit binding.

Semantically identical GEMVs must lower to identical decisions whatever their
spelling (audit M-01), the emitted MAC commands must perform exactly the
matched program's work up to declared padding, and symbolic unit coordinates
are never guessed (audit C-07).
"""

from __future__ import annotations

import collections

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.aim_lowering import (
    AimContraction,
    AimCoverageError,
    _target_geometry,
    check_contraction_coverage,
)
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_codegen import _eval_sym
from allo.spmw_target import UnitId

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

M = K = 256
CH = 32
ROWS = M // CH


@region()
def gemv_row0(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[CH], args=[W, x, y])
    def mv(lW: bfloat16[M, K], lx: bfloat16[K], ly: bfloat16[M]):
        (ch,) = allo.get_wid()
        row0 = ch * ROWS
        for i in range(ROWS):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[row0 + i, j] * lx[j]
            ly[row0 + i] = acc


@region()
def gemv_inline(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[CH], args=[W, x, y])
    def mv(lW: bfloat16[M, K], lx: bfloat16[K], ly: bfloat16[M]):
        (ch,) = allo.get_wid()
        for i in range(ROWS):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[ch * ROWS + i, j] * lx[j]
            ly[ch * ROWS + i] = acc


@region()
def gemv_single(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def mv(lW: bfloat16[M, K], lx: bfloat16[K], ly: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[i, j] * lx[j]
            ly[i] = acc


@region()
def matvec_renamed(A: bfloat16[M, K], v: bfloat16[K], out: bfloat16[M]):
    @allo.work(mapping=[CH], args=[A, v, out])
    def dot_rows(mat: bfloat16[M, K], vec: bfloat16[K], res: bfloat16[M]):
        (lane,) = allo.get_wid()
        for r in range(ROWS):
            s: bfloat16 = 0
            for c in range(K):
                s += mat[lane * ROWS + r, c] * vec[c]
            res[lane * ROWS + r] = s


M96 = 96
ROWS96 = M96 // 3


@region()
def gemv_96_map3(W: bfloat16[M96, K], x: bfloat16[K], y: bfloat16[M96]):
    @allo.work(mapping=[3], args=[W, x, y])
    def mv(lW: bfloat16[M96, K], lx: bfloat16[K], ly: bfloat16[M96]):
        (ch,) = allo.get_wid()
        for i in range(ROWS96):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[ch * ROWS96 + i, j] * lx[j]
            ly[ch * ROWS96 + i] = acc


KO, KI = 16, 16


@region()
def gemv_double_reduction(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def mv(lW: bfloat16[M, K], lx: bfloat16[K], ly: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for jo in range(KO):
                for ji in range(KI):
                    acc += lW[i, jo * KI + ji] * lx[jo * KI + ji]
            ly[i] = acc


H, DH, L = 8, 64, 128


@region()
def heads_qk(q: bfloat16[H, DH], kc: bfloat16[H, L, DH], sc: bfloat16[H, L]):
    @allo.work(mapping=[4], args=[q, kc, sc])
    def qk(lq: bfloat16[H, DH], lk: bfloat16[H, L, DH], ls: bfloat16[H, L]):
        for h in range(H):
            for s in range(L):
                acc: bfloat16 = 0
                for d in range(DH):
                    acc += lq[h, d] * lk[h, s, d]
                ls[h, s] = acc


def _compile(workload, **kwargs):
    # Every AiM MAC reduces 16 BF16 products through the bank adder tree.
    kwargs.setdefault("reduction_order", "reassociate")
    return allo.compile(workload, build_aim_target(), aim_cost, **kwargs)


def _contraction(compiled):
    (entry,) = [
        e
        for e in compiled.compiled.layout_ctx.lowering_manifest
        if e["kind"] == "contraction"
    ]
    (record,) = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimContraction)
    ]
    return entry, record


def _decisions(compiled):
    """Spelling-independent summary of everything the compiler decided."""
    (kernel,) = compiled.placement_ranking.kernels
    mode = {p: m for p, _c, m in kernel.scores}[kernel.chosen_position]
    entry, record = _contraction(compiled)
    commands = tuple(str(line) for line in compiled.compiled.cmds)
    opcodes = collections.Counter(line.split()[1] for line in commands)
    return {
        "mode": mode,
        "record": (
            record.outputs,
            record.reduction,
            record.batches,
            record.replicas,
            record.channels,
            record.batch_mapping,
        ),
        "mac_abk": opcodes["MAC_ABK"],
        "issued": entry["issued_scalar_macs"],
        "program": entry["program_scalar_macs"],
        "padding": entry["padding_ratio"],
        "commands": commands,
        "estimate": compiled.estimate().cycles,
    }


GEMV_VARIANTS = {
    "row0": gemv_row0,
    "inline": gemv_inline,
    "single": gemv_single,
    "renamed": matvec_renamed,
}


@pytest.fixture(scope="module")
def gemv_decisions():
    return {
        name: _decisions(_compile(workload, backend="virtual"))
        for name, workload in GEMV_VARIANTS.items()
    }


def test_gemv_spellings_make_identical_decisions(gemv_decisions):
    reference = gemv_decisions["row0"]
    assert reference["record"][:4] == (M, K, 1, 1)
    assert reference["mac_abk"] == 1
    assert reference["program"] == M * K
    assert reference["issued"] == M * K
    assert reference["padding"] == 1.0
    for name, decisions in gemv_decisions.items():
        assert decisions == reference, name


def test_output_loop_is_not_a_batch_axis():
    compiled = _compile(gemv_inline, backend="virtual")
    match = next(
        m for m in compiled.compiled.trace.matches if m.target_op_name == "MAC"
    )
    roles = match.extra["axis_roles"]
    i, j = (loop[0] for loop in match.enclosing_loops)
    assert roles["output"] == (i,)
    assert roles["reduction"] == (j,)
    assert roles["batch"] == ()
    assert roles["work_axis"] == "output"
    assert match.extra["batch_dim"] == 1
    assert match.extra["batch_loop_var"] is None


def test_vector_indexing_result_loop_is_the_batch_axis():
    compiled = _compile(heads_qk, backend="virtual")
    match = next(
        m for m in compiled.compiled.trace.matches if m.target_op_name == "MAC"
    )
    h, s, d = (loop[0] for loop in match.enclosing_loops)
    roles = match.extra["axis_roles"]
    assert (roles["batch"], roles["output"], roles["reduction"]) == ((h,), (s,), (d,))
    assert match.extra["batch_loop_var"] == h
    assert match.extra["batch_dim"] == H
    entry, record = _contraction(compiled)
    assert (record.outputs, record.reduction, record.batches, record.replicas) == (
        L,
        DH,
        H,
        4,
    )
    assert entry["program_scalar_macs"] == 4 * H * L * DH


def test_partitioned_96_row_gemv_issues_logical_work_plus_declared_padding():
    compiled = _compile(gemv_96_map3, backend="virtual")
    entry, record = _contraction(compiled)
    logical = M96 * K
    assert (record.outputs, record.batches, record.replicas) == (M96, 1, 1)
    assert entry["program_scalar_macs"] == logical
    assert entry["logical_scalar_macs"] == logical
    # Padding is the all-bank granularity: whole 16-bank channels per launch.
    geometry = _target_geometry(build_aim_target())
    capacity = entry["active_channel_count"] * geometry.banks
    padded_rows = -(-M96 // capacity) * capacity
    assert entry["issued_scalar_macs"] == padded_rows * K
    assert entry["issued_scalar_macs"] == entry["physical_mac_slots"]
    assert entry["padding_ratio"] == padded_rows / M96


def test_coverage_check_rejects_overissued_contraction():
    entry = {
        "name": "mv_0",
        "logical_scalar_macs": 8 * M * K,
        "physical_mac_slots": 8 * M * K,
    }
    with pytest.raises(AimCoverageError, match=rf"{8 * M * K}.*{M * K}"):
        check_contraction_coverage(entry, 8 * M * K, M * K)
    entry = {"name": "mv_0", "logical_scalar_macs": M * K, "physical_mac_slots": M * K}
    with pytest.raises(AimCoverageError, match="emitted commands issue"):
        check_contraction_coverage(entry, 2 * M * K, M * K)
    assert check_contraction_coverage(entry, M * K, M * K)["padding_ratio"] == 1.0


def test_codegen_rejects_a_record_that_misstates_the_program(monkeypatch):
    import allo.spmw_aim as spmw_aim

    build = spmw_aim.build_contraction

    def eightfold(shape, knobs, **kwargs):
        return build(dict(shape, outputs=8 * shape["outputs"]), knobs, **kwargs)

    monkeypatch.setattr(spmw_aim, "build_contraction", eightfold)
    with pytest.raises(AimCoverageError, match=rf"{8 * M * K}.*{M * K}"):
        _compile(gemv_inline, backend="virtual")


def test_unsupported_reduction_nest_is_rejected():
    with pytest.raises(ValueError, match="reduces over exactly the innermost loop"):
        _compile(gemv_double_reduction, backend="virtual")


def test_gemv_spellings_simulate_identically(aim_sim_gate):
    cycles = {
        name: _compile(GEMV_VARIANTS[name]).run_backend().cycles
        for name in ("row0", "inline", "single")
    }
    assert len(set(cycles.values())) == 1, cycles
    assert next(iter(cycles.values())) > 0


# --------------------------------------------------------------------- #
# Unit coordinates (C-07) and target/simulator geometry (M-10)
# --------------------------------------------------------------------- #


def _bank_handle(target):
    bank_group = UnitId(level=1, unit=target.unit("bank_group"))
    bank = UnitId(level=2, unit=target.unit("bank"))
    return target.banks[4 * bank_group + bank]


def test_eval_sym_refuses_unbound_unit_ids():
    target = build_aim_target()
    index = _bank_handle(target).idx
    with pytest.raises(ValueError, match=r"level 1 \(unit 'bank_group'\)"):
        _eval_sym(index, {0: 3})
    assert _eval_sym(index, {0: 3}, representative=True) == 0
    assert _eval_sym(index, {1: 2, 2: 3}) == 11


# The handle-based `AimCtx.cmd` encoder that bound bank handles from work ids
# was unreachable from the matcher route and was removed (audit M-11); bank
# addresses come from the LinearLayout in `AimCtx.emit_program`/`_sbk_banks`.


def test_target_geometry_requires_a_reuse_slot():
    target = build_aim_target()
    target.mac_reg.slots = 0
    with pytest.raises(ValueError, match="slots >= 1"):
        _target_geometry(target)


def test_simulator_rejects_a_mismatched_channel_count(tmp_path):
    from allo.spmw_aim import _check_simulator_geometry

    config = tmp_path / "example.yaml"
    config.write_text(
        "MemorySystem:\n  DRAM:\n    impl: GDDR6\n    org:\n"
        "      preset: GDDR6_AiM_org\n"
    )
    _check_simulator_geometry(build_aim_target(), config)
    target = build_aim_target()
    target.unit("channel").axes["channel"] = 24
    with pytest.raises(ValueError, match=r"'channels': \(24, 32\)"):
        _check_simulator_geometry(target, config)
