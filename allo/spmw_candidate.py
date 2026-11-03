# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common candidate-plan contract of the shared pipeline (release gate G9).

A backend whose candidates carry an intrinsic plan (the APU v1 GVML vector
plans, the Gemini-II composed-dot launch plans) states the plan's semantic
and scheduling effects in one typed record, :class:`CandidateEffects`, kept
in the placement's ``extra["effects"]``.  The shared driver *checks* those
effects, it does not merely display them:

* :func:`validate_candidate_effects` compares a candidate's coverage,
  values, transfers, launches and ABI with the semantic program the matcher
  established for the candidate's work item (every matched store's external
  destination must be covered, every value must be one of the program's
  buffers, every ABI name one of its parameters).  ``autoschedule`` runs it
  on every enumerated candidate before costing; ``compile_for_target`` runs
  it again on the selected placements.
* :func:`rebind_supplied_layouts` is the boundary for a caller-supplied
  ``layout=`` placement: a placement is a *decision*, not an executable, so
  it is rebound to the candidate of the current program that carries the
  same physical decision (or, for a backend plan, the same plan name) after
  the current enumeration applied the target's legality and the current
  reduction-order policy.  A placement that designates no candidate of the
  current program is refused; a plan the policy pruned raises
  ``ReductionOrderError``.  The payload of a supplied placement is never
  executed as such.
* After materialization the backend context's ``check_realization``
  compares the realized program (its ABI, launches) with the effects the
  selected candidate declared.

The matcher backends (Samsung, AiM, UPMEM, AttAcc) express their candidates
as storage handles checked by ``spmw_legality``; they declare no effects
yet, so the effects validation is a no-op for them (recorded in ISSUES
X-01 as remaining migration work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateCoverage:
    """Which part of the semantic program the candidate realizes."""

    function: str
    operations: tuple[str, ...]  # matched target ops it implements
    inputs: tuple[str, ...]  # buffers it reads (memref names)
    outputs: tuple[str, ...]  # external destinations it writes (memref names)


@dataclass(frozen=True)
class CandidateValueLayout:
    value: str
    axes: tuple[str, ...]
    storage: str | None = None


@dataclass(frozen=True)
class CandidateTransfer:
    value: str
    direction: str  # "in" | "out" | "inout"
    route: tuple[str, ...]


@dataclass(frozen=True)
class CandidateLaunch:
    index: int
    kind: str
    covers: tuple  # what the launch computes (plan name, stream/tile coordinates)
    accumulator: str | None = None


@dataclass(frozen=True)
class CandidateEffects:
    coverage: CandidateCoverage
    value_layouts: tuple[CandidateValueLayout, ...]
    transfers: tuple[CandidateTransfer, ...]
    launches: tuple[CandidateLaunch, ...]
    abi_inputs: tuple[str, ...]
    abi_outputs: tuple[str, ...]
    device_scope: str

    def manifest(self) -> dict[str, Any]:
        return {
            "coverage": {
                "function": self.coverage.function,
                "operations": list(self.coverage.operations),
                "inputs": list(self.coverage.inputs),
                "outputs": list(self.coverage.outputs),
            },
            "value_layouts": [
                {"value": v.value, "axes": list(v.axes), "storage": v.storage}
                for v in self.value_layouts
            ],
            "transfers": [
                {"value": t.value, "direction": t.direction, "route": list(t.route)}
                for t in self.transfers
            ],
            "launches": [
                {
                    "index": l.index,
                    "kind": l.kind,
                    "covers": list(l.covers),
                    "accumulator": l.accumulator,
                }
                for l in self.launches
            ],
            "abi": {"inputs": list(self.abi_inputs), "outputs": list(self.abi_outputs)},
            "device_scope": self.device_scope,
        }


def candidate_effects(placement) -> CandidateEffects | None:
    extra = getattr(placement, "extra", None) or {}
    effects = extra.get("effects")
    return effects if isinstance(effects, CandidateEffects) else None


def _program_values(matches):
    """The buffers of one work item's matches: inputs, external
    destinations, local destinations, parameter names (from the facts)."""

    inputs, outputs, locals_, operations = set(), set(), set(), set()
    for match in matches:
        operations.add(match.target_op_name)
        for operand in match.operands:
            if operand.memref_name is None or operand.is_loop_carried:
                continue
            inputs.add(operand.memref_name)
        if match.result_memref_name is not None:
            if match.extra.get("result_is_local"):
                locals_.add(match.result_memref_name)
            else:
                outputs.add(match.result_memref_name)
        sinks = match.extra.get("output_sinks", ())
        refs = match.extra.get("output_sink_refs", ())
        for position, (memref, _text) in enumerate(sinks):
            local = refs[position][1] if position < len(refs) else None
            if local:
                locals_.add(memref)
            elif memref != match.result_memref_name or match.extra.get(
                "result_is_local"
            ):
                outputs.add(memref)
    facts = matches[0].extra.get("program_facts") if matches else None
    parameters = set(getattr(facts, "parameter_names", ()) or ())
    return inputs, outputs, locals_, operations, parameters


def validate_candidate_effects(matches, placement) -> list[str]:
    """Problems of a candidate's declared effects against the semantic
    program of ``matches``; empty when consistent or when the candidate
    declares no effects (a handle-only matcher placement)."""

    effects = candidate_effects(placement)
    if effects is None:
        return []
    inputs, outputs, locals_, operations, parameters = _program_values(matches)
    functions = {match.func_name for match in matches}
    problems = []
    coverage = effects.coverage
    if coverage.function not in functions:
        problems.append(
            f"covers function {coverage.function!r}, not {sorted(functions)}"
        )
    if set(coverage.operations) != operations:
        problems.append(
            f"covers operations {sorted(coverage.operations)}, the work item "
            f"matches {sorted(operations)}"
        )
    if set(coverage.outputs) != outputs:
        problems.append(
            f"writes {sorted(coverage.outputs)}, the program's external "
            f"destinations are {sorted(outputs)}"
        )
    if not set(coverage.inputs) <= inputs | outputs:
        problems.append(
            f"reads {sorted(set(coverage.inputs) - inputs - outputs)}, which the "
            "program does not read"
        )
    known = inputs | outputs | locals_
    for layout in effects.value_layouts:
        if layout.value not in known:
            problems.append(f"lays out {layout.value!r}, not a program buffer")
    for transfer in effects.transfers:
        if transfer.value not in known:
            problems.append(f"transfers {transfer.value!r}, not a program buffer")
        if transfer.direction not in ("in", "out", "inout"):
            problems.append(f"transfer of {transfer.value!r} has no direction")
    if not effects.launches:
        problems.append("declares no launch")
    if set(effects.abi_outputs) != set(coverage.outputs):
        problems.append(
            f"ABI outputs {sorted(effects.abi_outputs)} differ from the coverage "
            f"outputs {sorted(coverage.outputs)}"
        )
    if parameters and not set(effects.abi_inputs) | set(effects.abi_outputs) <= (
        parameters
    ):
        problems.append(
            "ABI names "
            f"{sorted((set(effects.abi_inputs) | set(effects.abi_outputs)) - parameters)}"
            " are not parameters of the program"
        )
    return problems


def require_candidate_effects(matches, placement, what: str) -> None:
    problems = validate_candidate_effects(matches, placement)
    if problems:
        raise ValueError(
            f"{what}: candidate {getattr(placement, 'mode', '')!r} violates the "
            "candidate contract: " + "; ".join(problems)
        )


def rebind_supplied_layouts(view, trace, supplied) -> list:
    """Bind caller-supplied placements to candidates of the current program.

    ``supplied`` is aligned with ``_bucket_for_autoschedule(trace)``.  For
    each work item the current enumeration (legality and reduction-order
    policy applied) is searched for the candidate with the supplied
    placement's physical decision: a backend plan is replaced by that
    current candidate (a fresh payload bound to this program), a handle-only
    placement is kept as supplied (its decision is current).  A backend plan
    without such a candidate is looked up by plan name (``mode``) or
    refused; a handle-only placement without one keeps the caller's
    placement (its legality was checked) unless its mode was pruned by the
    reduction-order policy, which is refused.  Returns the placements to
    compile.
    """

    from .spmw_autoschedule import (
        _ENUMERATORS,
        _bucket_for_autoschedule,
        _matcher_operand_roles,
        _matcher_placement_decision,
        _matcher_work_scope,
        _stamp_group_replication,
        select_candidates,
    )
    from .spmw_legality import check_placement
    from .spmw_match import MatchTrace
    from .spmw_reduction_policy import ReductionOrderError

    enumerator = _ENUMERATORS.get(getattr(view, "name", None))
    buckets = _bucket_for_autoschedule(trace)
    if enumerator is None or len(supplied) != len(buckets):
        return list(supplied)
    # Enumerators may read whole-group facts the ranking stamps first.
    groups = {}
    for _scope, matches in buckets:
        groups.setdefault(_matcher_work_scope(matches[0]).group_id, []).extend(matches)
    for group_matches in groups.values():
        _stamp_group_replication(group_matches)
    rebound = []
    for placement, (_scope, matches) in zip(supplied, buckets):
        candidates = enumerator(view, matches)
        legal = [c for c in candidates if check_placement(view, matches, c).legal]
        bucket_trace = MatchTrace(trace.target_name, trace.module_name, list(matches))
        roles = _matcher_operand_roles(bucket_trace, [*legal, placement])
        wanted = _matcher_placement_decision(bucket_trace, placement, roles)
        same = [
            c
            for c in legal
            if _matcher_placement_decision(bucket_trace, c, roles) == wanted
        ]
        payload = (getattr(placement, "extra", None) or {}).get("payload")
        if same:
            if payload is None:
                # A handle-only placement with a current candidate's decision:
                # the caller's placement (with the fields its own ranking
                # derived) is the program; only its decision was re-checked.
                rebound.append(placement)
                continue
            chosen = same[0]
            chosen.extra = {**chosen.extra, "candidate_selection": "forced"}
            rebound.append(chosen)
            continue
        if payload is not None:
            # A backend plan from another compilation (or edited): rebind by
            # plan name to this program's plan, or refuse.
            named = [
                c
                for c in select_candidates(legal, placement.mode, matches)
                if not c.extra.get("candidate_excluded")
            ]
            if len(named) != 1:
                raise ValueError(
                    f"layout placement {placement.mode!r} designates "
                    f"{len(named)} candidates of kernel {matches[0].func_name!r}"
                )
            rebound.append(named[0])
            continue
        pruned = {}
        for candidate in candidates:
            for name, reason in candidate.extra.get("reduction_policy_pruned", ()):
                pruned.setdefault(name, reason)
        if placement.mode in pruned:
            raise ReductionOrderError(
                f"placement {placement.mode!r} was pruned by the reduction-order "
                f"policy ({pruned[placement.mode]})"
            )
        rebound.append(placement)
    return rebound


__all__ = [
    "CandidateCoverage",
    "CandidateEffects",
    "CandidateLaunch",
    "CandidateTransfer",
    "CandidateValueLayout",
    "candidate_effects",
    "rebind_supplied_layouts",
    "require_candidate_effects",
    "validate_candidate_effects",
]
