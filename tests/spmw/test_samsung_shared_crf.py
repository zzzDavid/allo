# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-025 lever 3: shared CRF + host trigger schedule vs per-work-id CRF.

Receipts (SPEC-025 §7):
  1. the enumerator offers both CRF-issue modes for the fast candidate,
  2. the cost model prices shared < per_workid by the derived gap,
  3. the work-id count is unit-tree geometry, not a literal,
  4. the ranking flips when CRF_TRIGGER.cycles is perturbed past threshold,
  5. shared codegen emits one CRF body + n_workids host triggers,
  6. full autoschedule on the GEMV trace argmins to the shared mode.

No simulator boot -- these are static-comparison / cost-ranking tests.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as _fp16
from allo.spmw_autoschedule import (
    _bucket_for_autoschedule,
    _samsung_enumerate,
    autoschedule,
)
from allo.spmw_cost import get_cost
from allo.spmw_cost_models import _samsung_workid_count
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_samsung_target


# GEMV workload shared by the codegen / e2e receipts (mapping=[16,8] ->
# 128 work-id buckets == the unit-tree fanout product).
_M, _K = 4096, 1024
_ROWS = _M // (16 * 8)


@_df_region()
def _gemv_top(W: _fp16[_M, _K], x: _fp16[_K], y: _fp16[_M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: _fp16[_M, _K], local_x: _fp16[_K], local_y: _fp16[_M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * _ROWS
        for i in range(_ROWS):
            acc: _fp16 = 0
            for k in range(_K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


def _synthetic_mac_trace() -> MatchTrace:
    """1-match GEMV-shaped trace (mirrors test_autoschedule's helper) so
    the cost tests avoid the ~3-minute MLIR pipeline. K=1024 inner loop."""
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
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
        ],
    )


def _fast_pair(candidates):
    """Return (shared, per_workid) variants of the fast dual-fiber +
    host-residency base candidate the argmin ultimately prefers."""
    def pick(issue):
        return next(
            c
            for c in candidates
            if c.mode.split("+", 1)[0] == "dual_fiber"
            and c.extra.get("crf_issue") == issue
            and c.extra.get("grf_residency", {}).get("local_W") == "host"
        )

    return pick("shared"), pick("per_workid")


def test_enumerator_emits_both_crf_modes():
    """For the fast base candidate the enumerator emits exactly one
    `shared` and one `per_workid` variant, both materialisable, no shape
    passed in (SPEC-025 §3)."""
    target = build_samsung_target()
    matches = _synthetic_mac_trace().matches
    candidates = _samsung_enumerate(target, matches)

    issues = [c.extra.get("crf_issue") for c in candidates]
    assert issues, "no candidates"
    assert all(i in ("shared", "per_workid") for i in issues), issues
    # Every base candidate is offered in both modes -> equal split.
    assert issues.count("shared") == issues.count("per_workid")

    shared, per_wid = _fast_pair(candidates)
    # The mode token is composable + legible; placements identical.
    assert shared.mode.endswith("+crf_shared"), shared.mode
    assert per_wid.mode.endswith("+crf_per_workid"), per_wid.mode
    assert shared.placements == per_wid.placements


def test_cost_prefers_shared_crf():
    """`shared` scores strictly below `per_workid`, and the gap equals the
    derived `body_cyc*(n_workids-1) - trigger_cyc*n_workids`, computed from
    `target.*` -- no literal in the assertion (SPEC-025 §7.2)."""
    target = build_samsung_target()
    trace = _synthetic_mac_trace()
    cost_fn = get_cost("kernel_cycles", target)
    candidates = _samsung_enumerate(target, trace.matches)
    shared, per_wid = _fast_pair(candidates)

    c_shared = cost_fn(trace, shared)
    c_per = cost_fn(trace, per_wid)
    assert c_shared < c_per, (c_shared, c_per)

    n = _samsung_workid_count(target)
    trigger = target.move("CRF_TRIGGER").cycles
    # SPEC-026 §3.5: the return is now P + E + R (B=1, non-resident). The
    # placement-invariant P+R offset is identical for both crf-issue modes
    # (it does not read crf_issue), so it cancels from the gap and the
    # shared-vs-per_workid structure is preserved. Strip P+R before
    # reconstructing the per-work-id body (E = body_cyc * n_workids).
    from allo.spmw_cost_models import (
        _samsung_mk,
        _samsung_preload_cycles,
        _samsung_readback_cycles,
    )

    M, K = _samsung_mk(target, trace)
    PR = _samsung_preload_cycles(target, M, K) + _samsung_readback_cycles(target, M)
    exec_per = c_per - PR
    # body_cyc = exec_per / n  (per_workid exec == body_cyc * n_workids).
    assert exec_per % n == 0, (exec_per, n)
    body_cyc = exec_per // n
    # shared == body_cyc + trigger*n + P+R; gap == body_cyc*(n-1) - trigger*n.
    assert c_shared == body_cyc + trigger * n + PR, (c_shared, body_cyc, trigger, n, PR)
    assert c_per - c_shared == body_cyc * (n - 1) - trigger * n


def test_workid_count_is_geometry_not_literal():
    """`_samsung_workid_count` is the product of the unit-tree mapping
    fanouts; mutate a fanout on a copy and the count tracks it (SPEC-025
    §4.2 / §7.3)."""
    target = build_samsung_target()
    n = _samsung_workid_count(target)

    expected = 1
    for u in target._walk():
        for f in u.mapping:
            expected *= f
    assert n == expected

    # Mutate a fixture fanout on a freshly-built target (Target's custom
    # __getattr__ makes copy.deepcopy recurse, so rebuild instead of copy)
    # -> the count tracks geometry, not a baked-in literal.
    mutated = build_samsung_target()
    for u in mutated._walk():
        if u.mapping == [8]:  # the `pim` unit's fanout
            u.mapping = [4]
            break
    assert _samsung_workid_count(mutated) == (expected // 8) * 4


def test_cost_ranking_responds_to_trigger_cycles():
    """Bumping `CRF_TRIGGER.cycles` past the flip threshold makes shared
    cost MORE than per_workid -- proving the choice is computed, not
    asserted (SPEC-025 §4.5, audit evidence #2)."""
    target = build_samsung_target()
    trace = _synthetic_mac_trace()
    candidates = _samsung_enumerate(target, trace.matches)
    shared, per_wid = _fast_pair(candidates)

    n = _samsung_workid_count(target)
    cost_fn = get_cost("kernel_cycles", target)
    # SPEC-026 §3.5: strip the placement-invariant P+R offset before
    # reconstructing the per-work-id body (the flip threshold is a property
    # of E, not of the preload/readback phases, which cancel from the
    # shared-vs-per_workid comparison).
    from allo.spmw_cost_models import (
        _samsung_mk,
        _samsung_preload_cycles,
        _samsung_readback_cycles,
    )

    M, K = _samsung_mk(target, trace)
    PR = _samsung_preload_cycles(target, M, K) + _samsung_readback_cycles(target, M)
    body_cyc = (cost_fn(trace, per_wid) - PR) // n

    # Flip threshold: shared >= per_workid when
    # body_cyc + trigger*n >= body_cyc*n  <=>  trigger >= body_cyc*(n-1)/n.
    # Push trigger past it on a freshly-built target (Target's __getattr__
    # makes copy.deepcopy recurse) and confirm the ranking inverts.
    flipped = build_samsung_target()
    flipped.move("CRF_TRIGGER").cycles = body_cyc  # >= threshold for n>=2
    cost_flipped = get_cost("kernel_cycles", flipped)
    cands_f = _samsung_enumerate(flipped, trace.matches)
    shared_f, per_f = _fast_pair(cands_f)
    assert cost_flipped(trace, shared_f) >= cost_flipped(trace, per_f), (
        "raising CRF_TRIGGER.cycles to body_cyc must flip the ranking"
    )


def test_codegen_shared_emits_one_body_plus_triggers():
    """Shared codegen emits ONE CRF body (one work-id's quad) into `cmds`
    plus `n_workids` HostTriggers; the per_workid candidate emits
    ~n_workids x more MAC records (SPEC-025 §5.3 / §7.5)."""
    target = build_samsung_target()
    schedule = allo.customize(_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    matches = _bucket_for_autoschedule(trace)[0][1]
    shared, per_wid = _fast_pair(_samsung_enumerate(target, matches))

    c_shared = allo.compile_for_target(target, trace, shared)
    c_per = allo.compile_for_target(target, trace, per_wid)

    n = _samsung_workid_count(target)
    shared_macs = sum(1 for c in c_shared.cmds if c.type_ == "MAC")
    per_macs = sum(1 for c in c_per.cmds if c.type_ == "MAC")

    # One shared body == one bucket's MAC count (2: even+odd fiber).
    assert shared_macs == per_macs // n, (shared_macs, per_macs, n)
    # Host schedule: exactly one trigger per work-id; per_workid has none.
    assert len(c_shared.host_schedule) == n
    assert all(t.tile_count >= 1 for t in c_shared.host_schedule)
    assert c_per.host_schedule == []
    # The replicated stream is genuinely n_workids x longer.
    assert per_macs == shared_macs * n


def test_argmin_picks_shared_end_to_end():
    """Full `autoschedule` on the Samsung GEMV trace returns the shared
    CRF-issue mode -- cost-driven, not forced (SPEC-025 §7.6)."""
    target = build_samsung_target()
    schedule = allo.customize(_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    layouts = autoschedule(target, trace)
    assert layouts[0].extra.get("crf_issue") == "shared", layouts[0].extra
    assert layouts[0].mode.split("+", 1)[0] == "dual_fiber", layouts[0].mode


if __name__ == "__main__":
    test_enumerator_emits_both_crf_modes()
    test_cost_prefers_shared_crf()
    test_workid_count_is_geometry_not_literal()
    test_cost_ranking_responds_to_trigger_cycles()
    test_codegen_shared_emits_one_body_plus_triggers()
    test_argmin_picks_shared_end_to_end()
    print("ALL PASSED")
