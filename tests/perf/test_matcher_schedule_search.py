# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared schedule-search adapter gates for matcher placements."""

import warnings

import pytest
import allo.spmw_codegen as codegen

from allo.pim.costs import aim_cost, samsung_cost
from allo.pim.targets import build_aim_target, build_samsung_target
from allo.spmw_autoschedule import (
    MatcherPlacementDecision,
    MatcherWorkScope,
    Placement,
    _aim_enumerate,
    _bucket_for_autoschedule,
    _clone_placement,
    _matcher_placement_decision,
    _samsung_enumerate,
    _search_autoschedule_group,
    _search_autoschedule_program,
    autoschedule,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_codegen import compile_for_target


def _bound_mac(target_name, work_id, *, group_id=0, func_name="gemv"):
    match = MatchedOp(
        target_op_name="MAC",
        func_name=func_name,
        work_id=work_id,
        enclosing_loops=[("k", "0", "128", 1)],
        operands=[
            OperandBinding("x", "x"),
            OperandBinding("y", "W"),
            OperandBinding("acc", "acc", is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("begin", "end"),
        extra={
            "spmw_work_scope": MatcherWorkScope(
                group_id=group_id,
                work_id=work_id,
                group_shape=tuple(value + 1 for value in work_id),
            )
        },
    )
    return MatchTrace(target_name, "matcher_search", [match])


def test_samsung_shared_search_matches_legacy_stable_argmin():
    target = build_samsung_target()
    trace = _bound_mac(target.name, (0, 0))
    candidates = _samsung_enumerate(target, trace.matches)
    cost = samsung_cost.bind(target)

    result = _search_autoschedule_group(target, trace, candidates, cost)
    legacy_scores = []
    for index, candidate in enumerate(candidates):
        single = _search_autoschedule_group(
            target,
            trace,
            (candidate,),
            cost,
        )
        legacy_scores.append((single.best.objective, index))
    legacy_index = min(legacy_scores)[1]

    assert tuple(result.best.decisions) == ("placement",)
    assert isinstance(result.best.decisions["placement"], MatcherPlacementDecision)
    assert result.best.payload is candidates[legacy_index]
    assert result.best.materialized.placement is not result.best.payload
    assert result.best.score.model_fingerprint == cost.fingerprint

    scheduled = autoschedule(target, trace, cost)
    assert scheduled.schedule_activation.active is scheduled.schedule_search_result.best
    assert scheduled.fallback_reason is None


def test_matcher_incumbent_ties_and_downstream_clones_do_not_alias():
    target = build_aim_target()
    trace = _bound_mac(target.name, (0,))
    candidates = _aim_enumerate(target, trace.matches)
    candidates[0].extra["nested"] = {"values": [1, 2]}
    cost = aim_cost.bind(target)

    result = _search_autoschedule_group(
        target,
        trace,
        candidates,
        cost,
        incumbent_index=0,
    )
    retained = result.best.materialized.placement
    downstream = _clone_placement(retained)
    downstream.extra["nested"]["values"].append(3)
    downstream.extra["operation_name"] = "changed"

    assert result.best is result.best_incumbent
    assert retained.extra["nested"]["values"] == [1, 2]
    assert retained.extra["operation_name"] == "MAC_ABK"
    assert candidates[0].extra["nested"]["values"] == [1, 2]


def test_matcher_physical_decisions_ignore_operand_source_names():
    target = build_samsung_target()
    original = _bound_mac(target.name, (0, 0))
    renamed = _bound_mac(target.name, (0, 0))
    for index, operand in enumerate(renamed.matches[0].operands):
        operand.memref_name = f"renamed_operand_{index}"
    renamed.matches[0].result_memref_name = "renamed_operand_2"
    cost = samsung_cost.bind(target)

    first = _search_autoschedule_group(
        target,
        original,
        _samsung_enumerate(target, original.matches),
        cost,
    )
    second = _search_autoschedule_group(
        target,
        renamed,
        _samsung_enumerate(target, renamed.matches),
        cost,
    )

    assert tuple(candidate.decisions["placement"] for candidate in first.ranked) == (
        tuple(candidate.decisions["placement"] for candidate in second.ranked)
    )


def test_matcher_decisions_deduplicate_diagnostic_mode_and_extra_renames():
    target = build_samsung_target()
    trace = _bound_mac(target.name, (0, 0))
    original = _clone_placement(_samsung_enumerate(target, trace.matches)[0])
    renamed = _clone_placement(original)
    original.mode = "original audit label"
    renamed.mode = "renamed audit label"
    original.extra["diagnostic"] = {"label": "original", "revision": 1}
    renamed.extra["diagnostic"] = {"label": "renamed", "revision": 99}
    for placement, producer, consumer in (
        (original, "producer", "consumer"),
        (renamed, "renamed_producer", "renamed_consumer"),
    ):
        placement.extra["residency_crossing"] = {
            "x": {
                "producer_func": producer,
                "consumer_func": consumer,
                "crosses_kernel": True,
                "crosses_workid": False,
                "handle": placement.placements["x"],
            }
        }

    assert _matcher_placement_decision(trace, original) == _matcher_placement_decision(
        trace, renamed
    )

    result = _search_autoschedule_group(
        target,
        trace,
        (original, renamed),
        samsung_cost.bind(target),
    )
    assert result.stats.complete_assignments_considered == 1
    assert len(result.ranked) == 1
    assert result.best.payload is original


def test_matcher_decisions_distinguish_physical_extra_changes():
    target = build_samsung_target()
    trace = _bound_mac(target.name, (0, 0))
    original = _clone_placement(_samsung_enumerate(target, trace.matches)[0])
    changed = Placement(
        placements=dict(original.placements),
        mode=original.mode,
        extra=dict(original.extra),
        layout=original.layout,
    )
    changed.extra["stage_resident"] = not bool(
        original.extra.get("stage_resident", False)
    )

    assert _matcher_placement_decision(trace, original) != _matcher_placement_decision(
        trace, changed
    )

    result = _search_autoschedule_group(
        target,
        trace,
        (original, changed),
        samsung_cost.bind(target),
    )
    assert result.stats.complete_assignments_considered == 2
    assert len(result.ranked) == 2


def test_matcher_program_search_scores_complete_candidate_tuples_once():
    target = build_samsung_target()
    first = _bound_mac(target.name, (0, 0), group_id=0, func_name="producer").matches[0]
    second = _bound_mac(target.name, (0, 0), group_id=1, func_name="consumer").matches[
        0
    ]
    trace = MatchTrace(target.name, "joint_matcher_search", [first, second])
    candidate_groups = (
        tuple(_samsung_enumerate(target, [first])[:2]),
        tuple(_samsung_enumerate(target, [second])[:2]),
    )
    cost = samsung_cost.bind(target)

    result = _search_autoschedule_program(
        target,
        trace,
        candidate_groups,
        cost,
        incumbent_indices=(0, 0),
    )

    assert result.stats.exhaustive is True
    assert len(result.ranked) == 4
    assert tuple(result.best_incumbent.decisions) == (
        "group_0_placement",
        "group_1_placement",
    )
    assert all(
        isinstance(value, MatcherPlacementDecision)
        for value in result.best_incumbent.decisions.values()
    )
    assert result.best.objective_domain.scope == "whole_program"
    compute_groups = {
        activity.metadata["group_id"]
        for activity in result.best.materialized.execution_graph.activities
        if activity.metadata.get("phase") == "compute"
    }
    assert compute_groups == {0, 1}
    retained = result.best.materialized.placements[0]
    downstream = _clone_placement(retained)
    downstream.extra["candidate_owned"] = False
    assert "candidate_owned" not in retained.extra


def test_multigroup_autoschedule_activates_legacy_incumbent_and_exposes_shadow():
    target = build_samsung_target()
    first = _bound_mac(target.name, (0, 0), group_id=0, func_name="producer").matches[0]
    second = _bound_mac(target.name, (0, 0), group_id=1, func_name="consumer").matches[
        0
    ]
    trace = MatchTrace(target.name, "joint_matcher_activation", [first, second])

    scheduled = autoschedule(target, trace, samsung_cost.bind(target))
    result = scheduled.schedule_search_result
    incumbent = result.best_incumbent

    assert len(scheduled) == 2
    assert incumbent is not None
    assert scheduled.schedule_activation.active is incumbent
    assert scheduled.schedule_activation.recommended is result.best
    assert scheduled.fallback_reason == (
        None
        if result.best is incumbent
        else "shadow_only: promotion evidence was not requested"
    )
    assert result.best.objective_domain.scope == "whole_program"
    assert [placement.mode for placement in scheduled] == [
        placement.mode for placement in incumbent.materialized.placements
    ]


def test_matcher_program_search_is_invariant_to_digit_ending_symbol_renames():
    target = build_samsung_target()
    original_matches = [
        _bound_mac(
            target.name,
            (0, 0),
            group_id=0,
            func_name="producer_0_0",
        ).matches[0],
        _bound_mac(
            target.name,
            (0, 0),
            group_id=1,
            func_name="consumer_0_0",
        ).matches[0],
    ]
    renamed_matches = [
        _bound_mac(
            target.name,
            (0, 0),
            group_id=0,
            func_name="phase_2024",
        ).matches[0],
        _bound_mac(
            target.name,
            (0, 0),
            group_id=1,
            func_name="answer42",
        ).matches[0],
    ]
    for match in renamed_matches:
        match.work_id = (91,)

    original = MatchTrace(target.name, "rename_isomorphism", original_matches)
    renamed = MatchTrace(target.name, "rename_isomorphism", renamed_matches)
    original_groups = tuple(_bucket_for_autoschedule(original))
    renamed_groups = tuple(_bucket_for_autoschedule(renamed))
    assert tuple(key for key, _matches in original_groups) == tuple(
        key for key, _matches in renamed_groups
    )

    cost = samsung_cost.bind(target)

    def search(trace, groups):
        candidates = tuple(
            tuple(_samsung_enumerate(target, matches)[:2]) for _key, matches in groups
        )
        return _search_autoschedule_program(
            target,
            trace,
            candidates,
            cost,
            incumbent_indices=(0, 0),
        )

    first = search(original, original_groups)
    second = search(renamed, renamed_groups)
    assert tuple(
        (candidate.decisions, candidate.objective, candidate.is_incumbent)
        for candidate in first.ranked
    ) == tuple(
        (candidate.decisions, candidate.objective, candidate.is_incumbent)
        for candidate in second.ranked
    )

    def graph_signature(result):
        return tuple(
            (
                activity.id,
                activity.primitive,
                activity.latency_cycles,
                activity.depends_on,
                tuple(sorted(activity.metadata.items())),
            )
            for activity in result.best.materialized.execution_graph.activities
        )

    assert graph_signature(first) == graph_signature(second)
    assert first.best.enumeration_index == second.best.enumeration_index
    assert (
        first.best_incumbent.enumeration_index
        == second.best_incumbent.enumeration_index
    )


def test_single_group_challenger_falls_back_to_exact_legacy_incumbent(
    monkeypatch,
):
    target = build_samsung_target()
    trace = _bound_mac(target.name, (0, 0))
    candidates = _samsung_enumerate(target, trace.matches)
    cost = samsung_cost.bind(target)
    probe = _search_autoschedule_group(target, trace, candidates, cost)
    challenger_index = candidates.index(probe.best.payload)
    incumbent_index = next(
        index
        for index, candidate in enumerate(candidates)
        if index != challenger_index
        and _search_autoschedule_group(target, trace, (candidate,), cost).best.objective
        > probe.best.objective
    )
    monkeypatch.setattr(
        "allo.spmw_autoschedule._legacy_autoschedule_group_index",
        lambda *_args, **_kwargs: incumbent_index,
    )

    scheduled = autoschedule(target, trace, cost)
    result = scheduled.schedule_search_result

    assert result.best is not result.best_incumbent
    assert scheduled.schedule_activation.active is result.best_incumbent
    assert scheduled.fallback_reason == (
        "shadow_only: promotion evidence was not requested"
    )
    assert _matcher_placement_decision(
        trace, scheduled[0]
    ) == _matcher_placement_decision(trace, candidates[incumbent_index])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = compile_for_target(target, trace, cost=cost)
        incumbent = compile_for_target(
            target,
            trace,
            layout=candidates[incumbent_index],
            cost=cost,
        )

    assert compiled.cmds == incumbent.cmds
    assert compiled.host_schedule == incumbent.host_schedule
    assert compiled.execution_graph is compiled.matcher_materialization.execution_graph
    assert (
        compiled.matcher_materialization.promotion_materialization_fingerprint is None
    )
    assert "REDUCE runtime replaces" in (
        compiled.matcher_codegen_artifact.non_promotable_reason
    )
    assert (
        compiled.schedule_activation.active
        is compiled.schedule_search_result.best_incumbent
    )


def test_matcher_frozen_source_commands_and_runner_reject_stale_execution(monkeypatch):
    target = build_aim_target()
    trace = _bound_mac(target.name, (0,))
    cost = aim_cost.bind(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = compile_for_target(target, trace, cost=cost)

    materialized = compiled.matcher_materialization
    assert materialized.promotion_materialization_fingerprint is not None
    assert compiled.cmds[-1] == "AiM EOC"

    compiled.matcher_codegen_artifact.context.cmds[-1] = "AiM EOC stale"
    assert materialized.promotion_materialization_fingerprint is None
    with pytest.raises(
        RuntimeError,
        match="source/command evidence changed after scoring",
    ):
        compiled.run()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = compile_for_target(target, trace, cost=cost)
    compiled.cmds[-1] = "AiM EOC stale"
    with pytest.raises(
        RuntimeError,
        match="command/source stream changed after scoring",
    ):
        compiled.run()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = compile_for_target(target, trace, cost=cost)
    monkeypatch.setitem(
        codegen._BACKEND_RUN,
        "aim",
        lambda _compiled, **_inputs: None,
    )
    assert (
        compiled.matcher_materialization.promotion_materialization_fingerprint is None
    )
    with pytest.raises(
        RuntimeError,
        match="source/command evidence changed after scoring",
    ):
        compiled.run()
