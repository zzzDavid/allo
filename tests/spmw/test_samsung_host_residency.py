# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lever 2 (SPEC-024) -- GRF preload as a host-vs-CRF placement choice.

The broadcastable `x`/grf_a preload can be materialised either as a
per-work-id CRF MOV (crf residency) or hoisted onto the native HAB
broadcast (host residency). The enumerator offers both as real
Placements; the cost model prices host strictly cheaper by
`n_workids * target.move("LD_A").cycles`; codegen omits the CRF MOV for
host-resident operands so the move is absent from `compiled.cmds`.

These tests assert each seam (codegen omission, cost differential +
K-scaling, perturbation direction, argmin pick) without a shape literal
in any decision path.
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import (
    Placement,
    _samsung_enumerate,
    _with_residency,
    autoschedule,
)
from allo.spmw_codegen import compile_for_target
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_samsung_target


def _samsung_trace(work_ids: list[tuple[int, ...]]) -> MatchTrace:
    """A Samsung GEMV-shaped MAC trace with one match per work_id.

    Roles: `x`/local_W -> grf_a (the broadcast vector, host-eligible),
    `y`/local_x -> bank (is_auto), `acc` -> grf_b (storeback only).
    """
    matches = []
    for wid in work_ids:
        matches.append(
            MatchedOp(
                target_op_name="MAC",
                func_name=f"gemv_{'_'.join(map(str, wid))}",
                work_id=wid,
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        )
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="synthetic", matches=matches
    )


def _candidate(target, trace, mode: str, residency: str) -> Placement:
    """Pick the `mode` candidate from the enumerator and force `x`
    (local_W) to the requested residency.

    Lever 3 (SPEC-025) appends a `+crf_*` token to `mode`; these lever-2
    receipts exercise per-work-id emission (LD_A MOV per work-id), so
    select the per_workid CRF-issue variant and match the base token.
    """
    for p in _samsung_enumerate(target, trace.matches):
        if (
            p.mode.split("+", 1)[0] == mode
            and p.extra.get("crf_issue", "per_workid") == "per_workid"
            and all(
                v == residency
                for v in p.extra.get("grf_residency", {}).values()
            )
        ):
            return p
    # Fall back: take the per_workid base mode candidate and stamp residency.
    for p in _samsung_enumerate(target, trace.matches):
        if (
            p.mode.split("+", 1)[0] == mode
            and p.extra.get("crf_issue", "per_workid") == "per_workid"
        ):
            return _with_residency(p, "local_W", residency)
    raise AssertionError(f"no {mode!r} candidate")


# --------------------------------------------------------------------- #
# §8.1 -- codegen omits the CRF MOV for host residency
# --------------------------------------------------------------------- #


def test_samsung_host_residency_omits_crf_mov():
    """A host-residency placement emits no LD_A (GRF_A) preload MOV;
    the otherwise-identical crf placement emits one LD_A MOV per work-id.
    The omission is traceable to `extra["grf_residency"]`, not a stream
    diff."""
    target = build_samsung_target()
    trace = _samsung_trace([(0, 0)])

    crf = _candidate(target, trace, "dual_fiber", "crf")
    host = _candidate(target, trace, "dual_fiber", "host")

    crf_cmds = compile_for_target(target, trace, layout=crf).cmds
    host_cmds = compile_for_target(target, trace, layout=host).cmds

    def _ld_a_movs(cmds):
        return [c for c in cmds if c.type_ == "MOV" and c.dst_ == "GRF_A"]

    assert len(_ld_a_movs(crf_cmds)) == 1, crf_cmds
    assert len(_ld_a_movs(host_cmds)) == 0, host_cmds
    # The MAC/JUMP body is identical -- only the preload differs.
    crf_macs = [c for c in crf_cmds if c.type_ == "MAC"]
    host_macs = [c for c in host_cmds if c.type_ == "MAC"]
    assert crf_macs == host_macs


def test_samsung_host_residency_omits_mov_per_workid():
    """Two work-ids: crf emits two LD_A MOVs, host emits zero."""
    target = build_samsung_target()
    trace = _samsung_trace([(0, 0), (0, 1)])

    crf = _candidate(target, trace, "dual_fiber", "crf")
    host = _candidate(target, trace, "dual_fiber", "host")

    crf_cmds = compile_for_target(target, trace, layout=crf).cmds
    host_cmds = compile_for_target(target, trace, layout=host).cmds

    ld_a = lambda cmds: sum(
        1 for c in cmds if c.type_ == "MOV" and c.dst_ == "GRF_A"
    )
    assert ld_a(crf_cmds) == 2, crf_cmds
    assert ld_a(host_cmds) == 0, host_cmds


# --------------------------------------------------------------------- #
# §8.2 -- host priced strictly cheaper, saving scales with work-ids
# --------------------------------------------------------------------- #


def test_samsung_host_residency_priced_cheaper():
    """Under `_samsung_kernel_cycles`, the host-x candidate costs strictly
    less than the crf-x candidate by `n_workids * target.move("LD_A").cycles`,
    and the differential changes when the work-id count changes."""
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    ld_a = target.move("LD_A").cycles
    # Lever 3 (SPEC-025 §4.4): the per_workid body cost is now scaled by the
    # unit-tree fanout product. The lever-2 LD_A saving rides that uniform
    # scale, so the differential is geom * n_trace_workids * LD_A; derive
    # geom from target geometry (no literal).
    from allo.spmw_cost_models import _samsung_workid_count

    geom = _samsung_workid_count(target)

    savings = {}
    for n_workids in (1, 4):
        work_ids = [(0, i) for i in range(n_workids)]
        trace = _samsung_trace(work_ids)
        crf = _candidate(target, trace, "dual_fiber", "crf")
        host = _candidate(target, trace, "dual_fiber", "host")
        crf_cost = cost_fn(trace, crf)
        host_cost = cost_fn(trace, host)
        assert host_cost < crf_cost, (host_cost, crf_cost)
        savings[n_workids] = crf_cost - host_cost
        # Saving == geom * n_workids * LD_A.cycles (no shape literal; read
        # off the move spec, the trace work-id count, and target geometry).
        assert savings[n_workids] == geom * n_workids * ld_a

    # The differential scales with the work-id count -- 4x as many
    # work-ids => 4x the saving.
    assert savings[4] == 4 * savings[1]


# --------------------------------------------------------------------- #
# §8.3 -- cost-drives-choice perturbation
# --------------------------------------------------------------------- #


def test_samsung_host_residency_perturbation():
    """Bumping `target.move("LD_A").cycles` widens the host-vs-crf cost
    gap in the modelled direction (host gets relatively cheaper)."""
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    trace = _samsung_trace([(0, 0), (0, 1)])
    crf = _candidate(target, trace, "dual_fiber", "crf")
    host = _candidate(target, trace, "dual_fiber", "host")

    base_gap = cost_fn(trace, crf) - cost_fn(trace, host)

    ld_a_move = target.move("LD_A")
    original = ld_a_move.cycles
    try:
        ld_a_move.cycles = original + 10
        # Rebuild the cost fn so it re-reads the perturbed move cost.
        cost_fn2 = allo.get_cost("kernel_cycles", target)
        bumped_gap = cost_fn2(trace, crf) - cost_fn2(trace, host)
    finally:
        ld_a_move.cycles = original

    assert bumped_gap > base_gap, (bumped_gap, base_gap)


# --------------------------------------------------------------------- #
# §8.4 -- argmin picks host residency
# --------------------------------------------------------------------- #


def test_samsung_argmin_picks_host_residency():
    """`autoschedule` over the {bank_row, grf_staged, dual_fiber} ×
    {crf, host} candidate set returns the dual_fiber + host-x placement
    as the argmin, by cost, not by a constant."""
    target = build_samsung_target()
    trace = _samsung_trace([(0, 0)])
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    chosen = layouts[0]
    # Lever 3 (SPEC-025) appends a `+crf_*` token; match the base mode.
    assert chosen.mode.split("+", 1)[0] == "dual_fiber", chosen.mode
    assert chosen.extra.get("grf_residency", {}).get("local_W") == "host", (
        chosen.extra.get("grf_residency")
    )


def test_samsung_enumerator_offers_both_residencies():
    """The enumerator emits both crf and host variants for the
    broadcastable `x` role (no pruning); argmin decides."""
    target = build_samsung_target()
    trace = _samsung_trace([(0, 0)])
    candidates = _samsung_enumerate(target, trace.matches)

    residencies = {
        c.extra.get("grf_residency", {}).get("local_W", "crf")
        for c in candidates
    }
    assert {"crf", "host"} <= residencies, residencies
    # Both residencies appear for the dual_fiber mode specifically.
    dual = [c for c in candidates if c.mode.split("+", 1)[0] == "dual_fiber"]
    dual_res = {
        c.extra.get("grf_residency", {}).get("local_W", "crf") for c in dual
    }
    assert dual_res == {"crf", "host"}, dual_res


if __name__ == "__main__":
    test_samsung_host_residency_omits_crf_mov()
    test_samsung_host_residency_omits_mov_per_workid()
    test_samsung_host_residency_priced_cheaper()
    test_samsung_host_residency_perturbation()
    test_samsung_argmin_picks_host_residency()
    test_samsung_enumerator_offers_both_residencies()
    print("ok")
