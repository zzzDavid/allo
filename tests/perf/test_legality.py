# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral placement legality (release audit C-01, C-08, C-12)."""

from __future__ import annotations

import pytest

import allo
from allo import spmw_match_engine, spmw_plan
from allo.compiler import _stamp_matcher_work_scopes
from allo.customize import customize
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import (
    _ENUMERATORS,
    Placement,
    _bucket_for_autoschedule,
    _trace_memrefs_by_role,
    autoschedule,
)
from allo.spmw_codegen import compile_for_target
from allo.spmw_legality import (
    RULE_ALIASING,
    RULE_CAPACITY,
    RULE_DECLARATION,
    RULE_MOVE,
    RULE_PATTERN,
    RULE_SCOPE,
    IllegalPlacement,
    check_placement,
    check_target,
    operand_signature,
)
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchedOp, OperandBinding


# --------------------------------------------------------------------- #
# Synthetic target tree
# --------------------------------------------------------------------- #


def _synthetic_target():
    """Two sibling units under one device scope plus a host scope.

    unit_b's MAC reads `rb` (x) and the shared memory or `rb`/`r1`/`nofill`
    (y) and accumulates into `racc`. `nofill` has no declared fill move;
    `r1` holds one slot.
    """

    @allo.target("legality_probe")
    def probe():
        @allo.device
        def dev():
            @allo.unit(mapping={"a": 2})
            def unit_a():
                allo.reg(8, 16, name="ra")

            @allo.unit(mapping={"b": 2})
            def unit_b():
                shared = allo.mem(name="bmem", size_bytes=1024)
                rb = allo.reg(8, 16, name="rb")
                racc = allo.reg(8, 16, name="racc")
                r1 = allo.reg(1, 16, name="r1")
                nofill = allo.reg(8, 16, name="nofill")
                allo.move("LD_B", src=shared, dst=rb, emit=lambda ctx: None)
                allo.move("LD_R1", src=shared, dst=r1, emit=lambda ctx: None)
                allo.move("ST_ACC", src=racc, dst=shared, emit=lambda ctx: None)
                allo.op(
                    "MAC",
                    src=(allo.or_(rb, r1), allo.or_(allo.any_(shared), rb, r1, nofill)),
                    dst=racc,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                )

        @allo.unit(mode="host")
        def host():
            allo.mem(name="hmem", bytes=1 << 20)

    return probe


def _mac_match():
    return MatchedOp(
        target_op_name="MAC",
        func_name="kernel_0",
        work_id=(0,),
        enclosing_loops=[],
        operands=[
            OperandBinding("x", "X"),
            OperandBinding("y", "Y"),
            OperandBinding("acc", "A", is_loop_carried=True),
        ],
        result_memref_name="A",
        op_range=("begin", "end"),
    )


def _rules(target, placements):
    verdict = check_placement(target, [_mac_match()], Placement(placements=placements))
    return {v.rule for v in verdict.violations}, verdict


def test_legal_placement_has_no_violations():
    t = _synthetic_target()
    rules, verdict = _rules(t, {"X": t.rb, "Y": t.bmem, "A": t.racc})
    assert verdict.legal, verdict.describe()
    assert rules == set()


def test_commuted_operands_are_matched_up_to_fn_symmetry():
    # `acc + x*y` is symmetric in x, y: the matcher may bind either source to
    # either parameter, so the swapped assignment is legal.
    t = _synthetic_target()
    _, verdict = _rules(t, {"X": t.bmem, "Y": t.rb, "A": t.racc})
    assert verdict.legal, verdict.describe()


def test_sibling_register_is_unreachable():
    t = _synthetic_target()
    rules, verdict = _rules(t, {"X": t.ra, "Y": t.bmem, "A": t.racc})
    assert RULE_SCOPE in rules
    (scope,) = [v for v in verdict.violations if v.rule == RULE_SCOPE]
    assert (scope.op, scope.operand, scope.owner) == ("MAC", "x", "unit_a")
    assert scope.handle == "Register(ra)"


def test_host_memory_is_unreachable_without_a_move():
    t = _synthetic_target()
    rules, verdict = _rules(t, {"X": t.rb, "Y": t.hmem, "A": t.racc})
    assert RULE_SCOPE in rules
    assert any(v.owner == "host" and v.memref == "Y" for v in verdict.violations)


def test_src_pattern_violation():
    t = _synthetic_target()
    # x admits only rb/r1 and y the memory or registers; two memory operands
    # fit neither parameter order.
    rules, verdict = _rules(t, {"X": t.bmem, "Y": t.bmem, "A": t.racc})
    assert rules == {RULE_PATTERN}
    assert "src pattern" in verdict.describe()


def test_dst_pattern_and_accumulator_aliasing():
    t = _synthetic_target()
    rules, _verdict = _rules(t, {"X": t.rb, "Y": t.bmem, "A": t.rb})
    assert {RULE_PATTERN, RULE_ALIASING} <= rules


def test_register_without_declared_fill():
    t = _synthetic_target()
    rules, verdict = _rules(t, {"X": t.rb, "Y": t.nofill, "A": t.racc})
    assert rules == {RULE_MOVE}
    assert "no declared move fills" in verdict.describe()


def test_register_slot_capacity():
    t = _synthetic_target()
    rules, _verdict = _rules(t, {"X": t.r1, "Y": t.r1, "A": t.racc})
    assert rules == {RULE_CAPACITY}


def test_cross_subtree_declaration_is_reported():
    # Audit probe P1(a): an op and a device move in unit_b naming unit_a's
    # register are accepted at declaration; check_target reports both.
    holder = []

    @allo.target("scope_probe")
    def scope_probe():
        @allo.unit(mapping={"a": 2})
        def unit_a():
            holder.append(allo.reg(8, 16, name="ra"))

        @allo.unit(mapping={"b": 2})
        def unit_b():
            rb = allo.reg(8, 16, name="rb")
            allo.op("ADD_X", src=(holder[0], rb), dst=rb, fn=lambda x, y: x + y)
            allo.move("LD_X", src=holder[0], dst=rb, emit=lambda s, d, ctx: None)

    verdict = check_target(scope_probe)
    assert {(v.rule, v.op) for v in verdict.violations} == {
        (RULE_DECLARATION, "ADD_X"),
        (RULE_DECLARATION, "LD_X"),
    }
    assert check_target(_synthetic_target()).legal


def test_src_arity_must_fit_fn():
    t = _synthetic_target()
    op = t.op("MAC")
    sources, acc, pattern_of = operand_signature(op)
    assert (sources, acc, sorted(pattern_of)) == (["x", "y"], "acc", ["x", "y"])
    op.src = (t.rb,)
    with pytest.raises(ValueError, match="src pattern"):
        operand_signature(op)


def test_shipped_targets_declarations_are_legal():
    from allo.pim import targets

    for build in (
        targets.build_samsung_target,
        targets.build_aim_target,
        targets.build_upmem_target,
        targets.build_apu_v1_target,
        targets.build_apu_v2_target,
    ):
        verdict = check_target(build())
        assert verdict.legal, verdict.describe(None)


def test_samsung_srf_is_not_a_declared_operand():
    # The Samsung emitter encodes only GRF_A/GRF_B and the conductor has no
    # SRF upload, so `srf` is not offered as an operand (audit Q6).
    t = build_samsung_target()
    _sources, _acc, pattern_of = operand_signature(t.op("MAC"))
    from allo.spmw_target import pattern_admits

    assert pattern_admits(pattern_of["x"], t.grf_a)
    assert not pattern_admits(pattern_of["x"], t.srf)


# --------------------------------------------------------------------- #
# Samsung probe cases through compile_for_target (audit P1, P1b)
# --------------------------------------------------------------------- #

M, K = 128, 16


@_df_region()
def small_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@pytest.fixture(scope="module")
def samsung_case():
    target = build_samsung_target()
    schedule = customize(small_gemv, enable_tensor=False)
    trace = spmw_match_engine.match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)
    roles = _trace_memrefs_by_role(trace.matches)
    return target, trace, roles


def _forced(samsung_case, **handles):
    target, trace, roles = samsung_case
    placements = {roles[role]: handle for role, handle in handles.items()}
    return compile_for_target(
        target,
        trace,
        layout=Placement(placements=placements, mode="probe"),
        cost=samsung_cost.bind(target),
    )


@pytest.mark.parametrize(
    "case, rule",
    [
        ("acc_on_grf_a", RULE_PATTERN),
        ("all_on_grf_b", RULE_ALIASING),
        ("y_on_srf", RULE_PATTERN),
        ("x_on_host_dram", RULE_SCOPE),
    ],
)
def test_samsung_illegal_user_layouts_are_rejected(samsung_case, case, rule):
    t = samsung_case[0]
    handles = {
        "acc_on_grf_a": dict(x=t.grf_a, y=t.grf_a, acc=t.grf_a),
        "all_on_grf_b": dict(x=t.grf_b, y=t.grf_b, acc=t.grf_b),
        "y_on_srf": dict(x=t.grf_a, y=t.srf, acc=t.grf_b),
        "x_on_host_dram": dict(x=t.host_dram, y=t.grf_a, acc=t.grf_b),
    }[case]
    with pytest.raises(IllegalPlacement) as caught:
        _forced(samsung_case, **handles)
    assert rule in {v.rule for v in caught.value.violations}
    assert "user-supplied layout" in str(caught.value)


def test_samsung_legal_baseline_compiles(samsung_case):
    target, trace, _roles = samsung_case
    layouts = autoschedule(target, trace, samsung_cost)
    (kernel,) = layouts.placement_ranking.kernels
    # Every enumerated Samsung candidate is legal: nothing was filtered.
    assert all(position is not None for position, _reason in kernel.rejections)
    compiled = compile_for_target(
        target, trace, layout=list(layouts), cost=samsung_cost.bind(target)
    )
    assert compiled.cmds


def test_autoschedule_filters_illegal_candidates(samsung_case, monkeypatch):
    target, trace, roles = samsung_case
    original = _ENUMERATORS["samsung_hbm_pim"]

    def with_illegal(target_, matches):
        candidates = original(target_, matches)
        bad = Placement(
            placements={
                roles["x"]: target_.grf_b,
                roles["y"]: target_.grf_b,
                roles["acc"]: target_.grf_b,
            },
            mode="all_on_grf_b",
        )
        return [bad] + candidates

    monkeypatch.setitem(_ENUMERATORS, "samsung_hbm_pim", with_illegal)
    layouts = autoschedule(target, trace, samsung_cost)
    (kernel,) = layouts.placement_ranking.kernels
    illegal = [reason for position, reason in kernel.rejections if position is None]
    assert len(illegal) == 1
    assert "all_on_grf_b" in illegal[0] and RULE_ALIASING in illegal[0]
    assert all(mode != "all_on_grf_b" for _p, _c, mode in kernel.scores)
    for (_scope, matches), placement in zip(_bucket_for_autoschedule(trace), layouts):
        assert check_placement(target, matches, placement).legal


def test_autoschedule_raises_when_every_candidate_is_illegal(samsung_case, monkeypatch):
    target, trace, roles = samsung_case

    def only_illegal(target_, matches):
        return [
            Placement(
                placements={roles["x"]: target_.host_dram, roles["acc"]: target_.grf_b},
                mode="host_x",
            )
        ]

    monkeypatch.setitem(_ENUMERATORS, "samsung_hbm_pim", only_illegal)
    with pytest.raises(IllegalPlacement, match="every enumerated candidate"):
        autoschedule(target, trace, samsung_cost)


# --------------------------------------------------------------------- #
# C-08: lanes come from the placement's register
# --------------------------------------------------------------------- #


def test_lanes_metric_reads_the_placement_register(samsung_case):
    target, trace, roles = samsung_case
    match = trace.matches[0]
    mac = target.op("MAC")
    placed = Placement(
        placements={
            roles["x"]: target.grf_a,
            roles["y"]: target.grf_a,
            roles["acc"]: target.grf_b,
        }
    )
    assert spmw_plan._placement_register_lanes(mac, match, placed) == target.grf_b.lanes
    assert spmw_plan._placement_register_lanes(mac, match, Placement()) is None


def test_samsung_mac_rule_has_no_lanes_default(samsung_case):
    target, trace, roles = samsung_case
    bank = target.banks[0]
    no_register = Placement(
        placements={roles["x"]: bank, roles["y"]: bank, roles["acc"]: bank}
    )
    with pytest.raises(ValueError, match="no 'lanes' metric"):
        spmw_plan.build_execution_graph(
            target, trace, no_register, samsung_cost.bind(target)
        )


# --------------------------------------------------------------------- #
# C-12: identity layouts
# --------------------------------------------------------------------- #


def test_identity_rejects_unmatched_input_dims():
    with pytest.raises(ValueError, match=r"\['m', 'k'\] have no output dim"):
        LinearLayout.identity({"m": 4, "k": 4}, out_dims=("lane",))
    zeroed = LinearLayout.identity(
        {"m": 4, "k": 4}, out_dims=("lane",), allow_zero=True
    )
    assert zeroed.matrix().shape == (0, 4)
    assert not zeroed.is_surjective()
    # A size-1 input dim contributes no bits; it is not "unmatched".
    assert LinearLayout.identity(
        {"lane": 16, "unit": 1}, out_dims=("lane",)
    ).is_surjective()
