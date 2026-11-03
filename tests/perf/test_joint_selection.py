# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection integrity (release audit C-03, C-05; docs/release/design/selection.md).

Ranking considers only candidates whose code generation succeeds, whole-program
cost selects the program, and the joint evaluation label says which route ran.
"""

import math
from types import SimpleNamespace

import pytest

import allo
from allo import spmw_knobs, spmw_match_engine, spmw_samsung
from allo.compiler import _stamp_matcher_work_scopes
from allo.customize import customize
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.perf.cost import _BIND_STACK, handle_path
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import (
    SelectionPolicy,
    _bucket_decision_map,
    _bucket_for_autoschedule,
    _matcher_search_scope,
    _per_bucket_argmin_index,
    _samsung_enumerate,
    autoschedule,
    selection_policy,
)
from allo.spmw_codegen import compile_for_target
from allo.spmw_knobs import reset_active_liveness, set_active_liveness
from allo.spmw_liveness import trace_liveness

M, K, ROWS = 128, 16, 1


@_df_region()
def small_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


# Two single-item kernels; both name their vector operand `lx` so a
# cross-kernel residency pair can couple them. The reduction extents differ
# (16 vs 32) so a cost rule can tell the producer from the consumer.
CM, CK = 32, 16


@_df_region()
def chain(W: fp16[CM, CK], V: fp16[CM, CM], x: fp16[CK], y: fp16[CM], z: fp16[CM]):
    @allo.work(mapping=[1], args=[W, x, y])
    def produce(lW: fp16[CM, CK], lx: fp16[CK], ly: fp16[CM]):
        for i in range(CM):
            acc: fp16 = 0
            for k in range(CK):
                acc += lW[i, k] * lx[k]
            ly[i] = acc

    @allo.work(mapping=[1], args=[V, y, z])
    def consume(lW: fp16[CM, CM], lx: fp16[CM], ly: fp16[CM]):
        for i in range(CM):
            acc: fp16 = 0
            for k in range(CM):
                acc += lW[i, k] * lx[k]
            ly[i] = acc


_PRODUCER_RESTAGE = 100  # the producer pays this unless its resident pair holds
_CONSUMER_RESIDENT = 30  # the consumer pays this when it keeps the pair resident
_samsung_rules = samsung_cost.builder


@allo.cost(target="samsung_hbm_pim", name="samsung_residency_coupled")
def coupled_cost(target):
    """Samsung cost plus a residency coupling that separates the per-group
    minima (producer resident, consumer restage) from the whole-program
    minimum (both resident)."""
    _samsung_rules(target)
    base_mac = _BIND_STACK[-1].rules.pop(handle_path(target.op("MAC")))
    channel = target.unit("pseudo_channel")

    @allo.rule(target.op("MAC"))
    def mac(event, ctx):
        base_mac(event, ctx)
        candidate = event.metrics.get("candidate", {}) or {}
        resident = candidate.get("residency") == "resident" and bool(
            candidate.get("residency_pairs")
        )
        producer = int(event.metrics.get("reduction_extent", 0)) == CK
        if producer:
            penalty = 0 if resident else _PRODUCER_RESTAGE
        else:
            penalty = _CONSUMER_RESIDENT if resident else 0
        if penalty:
            ctx.step(
                latency=penalty,
                occupy=[ctx.use(channel, cycles=penalty)],
                name="residency_coupling",
            )


def _match(region, target, **kwargs):
    schedule = customize(region, enable_tensor=False)
    trace = spmw_match_engine.match_workload(target, schedule.module, **kwargs)
    _stamp_matcher_work_scopes(target, schedule.module, trace)
    return trace


@pytest.fixture(scope="module")
def gemv_trace():
    target = build_samsung_target()
    return target, _match(small_gemv, target, reduction_order="reassociate")


@pytest.fixture
def coupled_chain(monkeypatch):
    """The chain program with `lx` declared a cross-kernel value.

    The whole-trace liveness never reports a cross-kernel span for dataflow
    regions (each kernel argument is its own ABI value), so the coupling the
    residency knob and `_reconcile_resident_pairs` act on is declared here.
    """
    target = build_samsung_target()
    trace = _match(chain, target)
    (producer_scope, _p), (consumer_scope, _c) = _bucket_for_autoschedule(trace)
    span = SimpleNamespace(
        producer_func=producer_scope,
        consumer_func=consumer_scope,
        crosses_kernel=True,
        crosses_workid=False,
    )

    def crossing(ctx):
        if not ctx.liveness or "lx" not in getattr(ctx.base, "placements", {}):
            return []
        return [("lx", span)]

    monkeypatch.setattr(spmw_knobs, "_residency_crossing_memrefs", crossing)
    return target, trace


def _residency(placement):
    return placement.extra.get("residency")


@pytest.mark.parametrize(
    "error",
    [
        spmw_samsung.SamsungUnsupported("forced emitter bug"),
        ValueError("forced emitter bug"),
    ],
    ids=["SamsungUnsupported", "ValueError"],
)
def test_emitter_error_on_top_candidate_falls_through(gemv_trace, monkeypatch, error):
    """C-03: a non-InfeasibleSchedule emitter error on the top-ranked
    candidate is a recorded rejection; the compile selects the next
    candidate that materializes instead of aborting."""
    target, trace = gemv_trace
    original = spmw_samsung.SamsungCtx.emit_program
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return original(self, *args, **kwargs)

    monkeypatch.setattr(spmw_samsung.SamsungCtx, "emit_program", flaky)
    compiled = compile_for_target(target, trace, cost=samsung_cost)
    ranking = compiled.placement_ranking
    (kernel,) = ranking.kernels

    top = kernel.scores[0][0]
    reason = f"{type(error).__name__}: forced emitter bug"
    assert (top, reason) in kernel.rejections
    rejected = {position for position, _reason in kernel.rejections}
    assert kernel.chosen_position == next(
        position for position, _c, _m in kernel.scores if position not in rejected
    )
    assert kernel.chosen_position in kernel.materialized
    assert top not in kernel.materialized
    # Lazy contract: probing stopped at the first success.
    assert kernel.probe_count == len(rejected & {p for p, _c, _m in kernel.scores}) + 1
    assert ranking.materialization["probed"] == kernel.probe_count
    assert compiled.cmds


def test_no_materializable_candidate_is_a_hard_error_listing_rejections(
    gemv_trace, monkeypatch
):
    target, trace = gemv_trace

    def broken(self, *args, **kwargs):
        raise NotImplementedError("emitter cannot lower anything")

    monkeypatch.setattr(spmw_samsung.SamsungCtx, "emit_program", broken)
    # The backend's own diagnostic surfaces, with every rejection in a note.
    with pytest.raises(
        NotImplementedError, match="emitter cannot lower anything"
    ) as caught:
        autoschedule(target, trace, samsung_cost)
    (note,) = caught.value.__notes__
    assert "none materialized" in note
    assert note.count("NotImplementedError: emitter cannot lower anything") > 1


def test_per_bucket_fallback_uses_the_materialization_probe(gemv_trace):
    target, trace = gemv_trace
    bound = samsung_cost.bind(target)
    token = set_active_liveness(trace_liveness(trace))
    try:
        _scope, matches = _bucket_for_autoschedule(trace)[0]
        bucket_trace, candidates, _key_map = _bucket_decision_map(
            target, trace, _samsung_enumerate, matches
        )
        unprobed = _per_bucket_argmin_index(target, bucket_trace, candidates, bound)
        refused = []
        rejections = []

        def refuse_first(probe_trace, candidate):
            refused.append(candidate)
            return "refused" if len(refused) == 1 else None

        chosen = _per_bucket_argmin_index(
            target,
            bucket_trace,
            candidates,
            bound,
            probe=refuse_first,
            rejections=rejections,
        )
    finally:
        reset_active_liveness(token)
    assert refused[0] is candidates[unprobed]
    assert rejections == [(unprobed, "refused")]
    assert chosen != unprobed and refused[1] is candidates[chosen]


@_df_region()
def strip_gemv(W: fp16[64, 64], x: fp16[64], y: fp16[64]):
    @allo.work(mapping=[16], args=[W, x, y])
    def mv(lW: fp16[64, 64], lx: fp16[64], ly: fp16[64]):
        (c,) = allo.get_wid()
        for i in range(4):
            acc: fp16 = 0
            for k in range(64):
                acc += lW[c * 4 + i, k] * lx[k]
            ly[c * 4 + i] = acc


def test_per_bucket_fallback_probes_the_assembled_kernel(monkeypatch):
    """Buckets that share no decision key take their own argmins; the
    assembled kernel goes through the same code generation probe, and a
    failure is recorded and repaired instead of aborting."""
    from allo import spmw_autoschedule, spmw_codegen
    from allo.spmw_autoschedule import rank_matcher_placements

    target = build_samsung_target()
    trace = _match(strip_gemv, target)
    buckets = _bucket_for_autoschedule(trace)
    bucket_of = {id(matches[0]): index for index, (_s, matches) in enumerate(buckets)}
    decision_map = spmw_autoschedule._bucket_decision_map

    def distinct_keys(target_, trace_, enumerator, matches):
        # Tag every bucket's decision keys so no key is shared; the
        # placements themselves stay identical across buckets.
        bucket_trace, candidates, key_map = decision_map(
            target_, trace_, enumerator, matches
        )
        tag = bucket_of[id(matches[0])]
        return bucket_trace, candidates, {(tag, k): c for k, c in key_map.items()}

    monkeypatch.setattr(spmw_autoschedule, "_bucket_decision_map", distinct_keys)

    original = spmw_codegen._materialize_matcher_codegen
    calls = []

    def first_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise NotImplementedError("forced kernel probe failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(spmw_codegen, "_materialize_matcher_codegen", first_fails)
    liveness = trace_liveness(trace)
    token = set_active_liveness(liveness)
    try:
        result = rank_matcher_placements(
            target,
            trace,
            _samsung_enumerate,
            samsung_cost.bind(target),
            liveness=liveness,
        )
    finally:
        reset_active_liveness(token)
    (kernel,) = result.placement_ranking.kernels
    assert kernel.fallback == "per_bucket"
    assert (
        None,
        "per-bucket argmins: NotImplementedError: forced kernel probe failure",
    ) in kernel.rejections
    assert kernel.probe_count == len(calls) >= 2
    assert len(result) == len(buckets)


def test_joint_evaluation_beats_conflicting_per_group_minima(coupled_chain):
    """C-05: the per-group minima conflict on residency (producer resident,
    consumer restage), reconciliation reverts the producer, and the joint
    evaluation selects the cheaper whole program (both resident)."""
    target, trace = coupled_chain
    bound = coupled_cost.bind(target)

    with selection_policy(SelectionPolicy(joint_budget=256)):
        joint = compile_for_target(target, trace, cost=bound)
    with selection_policy(SelectionPolicy(joint_budget=0)):
        heuristic = compile_for_target(target, trace, cost=bound)

    ranking = joint.placement_ranking
    producer, consumer = ranking.kernels
    assert ranking.joint_evaluation == "complete"
    assert ranking.joint_assignments == math.prod(
        len([p for p, _c, _m in kernel.scores if p not in dict(kernel.rejections)])
        for kernel in ranking.kernels
    )
    assert ranking.joint_assignments <= ranking.joint_budget
    assert ranking.reconciliation.reverted == ((producer.group_id, "lx"),)

    # Stage 1 picks: the producer's minimum is resident, the consumer's restage.
    per_group = heuristic.layout
    assert [_residency(p) for p in per_group] == ["restage", "restage"]
    assert heuristic.placement_ranking.joint_evaluation == "per-group heuristic"
    assert heuristic.placement_ranking.joint is None
    assert (
        producer.per_group_position
        == heuristic.placement_ranking.kernels[0].chosen_position
    )

    # The joint route keeps both endpoints resident and is cheaper as a whole.
    assert [_residency(p) for p in joint.layout] == ["resident", "resident"]
    assert all(p.extra.get("residency_pairs") for p in joint.layout)
    assert ranking.per_group_cycles == heuristic.placement_ranking.program_cycles
    assert ranking.program_cycles < ranking.per_group_cycles
    assert (
        ranking.per_group_cycles - ranking.program_cycles
        == _PRODUCER_RESTAGE - _CONSUMER_RESIDENT
    )
    assert ranking.joint.incumbent == (
        producer.per_group_position,
        consumer.per_group_position,
    )
    assert ranking.joint.incumbent_rank > 0
    assert ranking.joint.termination == "exhausted"
    assert ranking.joint.assignments_evaluated == ranking.joint_assignments
    assert ranking.joint.chosen == (
        (producer.group_id, producer.chosen_position),
        (consumer.group_id, consumer.chosen_position),
    )
    for compiled in (joint, heuristic):
        assert compiled.placement_ranking.program_cycles == int(
            bound.evaluate(compiled.execution_graph).cycles
        )


def test_over_budget_selection_is_labelled_heuristic(coupled_chain):
    target, trace = coupled_chain
    bound = coupled_cost.bind(target)
    with selection_policy(SelectionPolicy(joint_budget=3)):
        placements = autoschedule(target, trace, bound)
    ranking = placements.placement_ranking
    assert ranking.joint_assignments > 3
    assert ranking.joint_evaluation == "per-group heuristic"
    assert ranking.joint is None
    assert ranking.selection_cycles == ranking.per_group_cycles
    # The re-ranking of the reverted producer reached a fixed point without
    # finding the joint optimum: a coordinate step cannot leave it.
    assert ranking.reconciliation.fixed_point
    assert ranking.reconciliation.remaining == ranking.reconciliation.reverted


_LM, _LK, _LR = 64, 64, 4
_hx = allo.host_xfer


@_df_region()
def two_kernels(
    W: fp16[_LM, _LK],
    V: fp16[_LM, _LK],
    x: fp16[_LK],
    y: fp16[_LM],
    z: fp16[_LM],
):
    @allo.work(mapping=[16], args=[W, x, y])
    def mv1(lW: fp16[_LM, _LK], lx: fp16[_LK], ly: fp16[_LM]):
        (c,) = allo.get_wid()
        for i in range(_LR):
            acc: fp16 = 0
            for k in range(_LK):
                acc += lW[c * _LR + i, k] * lx[k]
            ly[c * _LR + i] = acc

    @allo.work(mapping=[16], args=[V, y, z])
    def mv2(lV: fp16[_LM, _LK], ly: fp16[_LK], lz: fp16[_LM]):
        (c,) = allo.get_wid()
        for i in range(_LR):
            acc: fp16 = 0
            for k in range(_LK):
                acc += lV[c * _LR + i, k] * ly[k]
            lz[c * _LR + i] = acc


@allo.host_program(two_kernels)
def chained(W, V, x, y, z):
    _hx.scatter(W, _hx.banks)
    _hx.scatter(V, _hx.banks)
    _hx.broadcast(x, _hx.grf_a)
    allo.launch("mv1", W, x, y)
    _hx.gather(y, _hx.banks)
    _hx.broadcast(y, _hx.grf_a)
    allo.launch("mv2", V, y, z)
    _hx.gather(z, _hx.banks)


@pytest.mark.parametrize(
    "host_moves", [None, chained], ids=["no-host", "launch-schedule"]
)
def test_routes_agree_and_program_cycles_is_the_estimate(host_moves):
    """The two-kernel program of test_host_program_analysis: the joint and
    the per-group routes select the same program, the label names the route,
    and `program_cycles` is `estimate().cycles`."""
    results = {}
    for budget in (256, 0):
        with selection_policy(SelectionPolicy(joint_budget=budget)):
            compiled = allo.compile(
                two_kernels,
                build_samsung_target(),
                samsung_cost,
                backend="virtual",
                host_moves=host_moves,
            )
        ranking = compiled.compiled.placement_ranking
        assert ranking.program_cycles == compiled.estimate().cycles
        assert ranking.selection_cycles == ranking.program_cycles
        results[budget] = compiled
    joint = results[256].compiled.placement_ranking
    heuristic = results[0].compiled.placement_ranking
    assert joint.joint_evaluation == "complete"
    assert joint.joint.termination == "exhausted"
    assert joint.joint.assignments_evaluated == joint.joint_assignments
    assert heuristic.joint_evaluation == "per-group heuristic"
    assert heuristic.joint_assignments == joint.joint_assignments > 0
    assert joint.program_cycles == heuristic.program_cycles
    assert [k.chosen_position for k in joint.kernels] == [
        k.chosen_position for k in heuristic.kernels
    ]
    # Equal-cycle assignments keep the per-group pick (the incumbent).
    assert joint.joint.incumbent_rank == 0 and joint.joint.chosen_rank == 0


def test_single_group_scores_are_whole_program_scores(gemv_trace):
    target, trace = gemv_trace
    placements = autoschedule(target, trace, samsung_cost)
    ranking = placements.placement_ranking
    (kernel,) = ranking.kernels
    cycles = dict((p, c) for p, c, _m in kernel.scores)
    assert ranking.joint_evaluation == "complete"
    assert ranking.selection_cycles == cycles[kernel.chosen_position]
    assert ranking.per_group_cycles == ranking.selection_cycles
    assert kernel.chosen_position == kernel.per_group_position
    # One group: the joint evaluation reuses the group scores.
    assert ranking.joint.whole_program_evaluations == 0


def test_eager_materialization_selects_the_lazy_program():
    """Lazy and eager materialization select the same program; eager
    generates every candidate, lazy stops at the first that succeeds."""
    rankings = {}
    for mode in ("lazy", "eager"):
        with selection_policy(SelectionPolicy(materialize=mode)):
            compiled = allo.compile(
                two_kernels, build_samsung_target(), samsung_cost, backend="virtual"
            )
        rankings[mode] = compiled.compiled.placement_ranking
    lazy, eager = rankings["lazy"], rankings["eager"]
    assert eager.materialize == "eager" and lazy.materialize == "lazy"
    assert [k.chosen_position for k in eager.kernels] == [
        k.chosen_position for k in lazy.kernels
    ]
    assert eager.program_cycles == lazy.program_cycles
    assert eager.materialization["probed"] == eager.materialization["ranked"]
    assert lazy.materialization["probed"] < eager.materialization["probed"]
    assert eager.joint.chosen_rank == lazy.joint.chosen_rank
