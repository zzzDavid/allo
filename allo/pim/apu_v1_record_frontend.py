# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``APUv1Program`` records as frontends of the shared pipeline (gate G9).

An ``APUv1Program`` is a whole-function program for the GSI Gemini-I: its
one phase is an ordinary ``@allo`` function the device executes verbatim,
either as scalar ARC C emitted from the retained MLIR or as one native GVML
recipe proven over that MLIR (histogram, record frequency, bivariate
moments, squared-L2 argmin) when the phase requires vectorization.

The record's semantics come from the shared front stages, not from the
lowering:

* :func:`normalize_apu_v1_program` customizes the phase once, runs the
  matcher on the retained module, keeps every store's facts the matcher
  retained (loop nest, guards, the buffers the value and guards read,
  conversions, whether the element written is also read) as the function's
  `StoreEffect` set (`allo.spmw_semantics.analyze_program`), and marks the
  facts *verbatim*: coverage problems are not dropped effects, since the
  device runs the function itself, but every store is in the contract.
  The returned ``SourceProgram`` keeps the schedule; one anchor match of the
  target's ``ARC_SCALAR_PROGRAM`` / ``NATIVE_VECTOR_PROGRAM`` op is the work
  item the common pipeline schedules.
* The ``apu_v1`` enumerator builds the lowering from that same schedule
  (:class:`allo.pim.apu_v1_program.CompiledAPUv1Program`) and emits one
  candidate (:class:`APUv1RecordCandidate`) whose :class:`CandidateEffects`
  are derived from the facts: inputs are the buffers the function reads
  plus every argument it does not overwrite whole (its initial values
  survive), outputs its destinations, one whole-function launch advancing
  every store effect, L4 value layouts, the host transfers and scratch
  allocation the runtime performs.  The shared validator checks the
  declaration against the facts before costing.
* :class:`APUv1RecordProgram` is the realized program; the backend context
  compares the declared transfers and allocations with what the lowering's
  runtime actually moves and allocates (`realized_transfers`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

from ..spmw_match import MatchedOp, MatchTrace
from ..spmw_match_engine import coordinate_forms, match_workload, require_match_coverage
from ..spmw_semantics import analyze_program, stamp_program_facts
from .apu_v1_program import APUv1Program, CompiledAPUv1Program, customize_apu_v1_phase
from .record_frontend import RecordWorkload, operand_binding, record_ref

_LOWERING_KEY = "apu_v1_lowering"
_SCHEDULE_KEY = "apu_v1_schedule"
_REDUCTION_ORDER_MESSAGE = (
    "APUv1Program fixes its reduction grouping in the program record; "
    "reduction_order applies to source workloads"
)


def is_apu_v1_program(workload) -> bool:
    return isinstance(workload, APUv1Program)


def program_fingerprint(lowered: CompiledAPUv1Program) -> str:
    """Identity of a lowering: its route and the device source it emits."""
    digest = hashlib.sha256()
    digest.update(lowered.route.encode())
    digest.update(b"\0")
    digest.update(lowered.device_source.encode())
    return digest.hexdigest()[:16]


def _operation_name(phase) -> str:
    return (
        "NATIVE_VECTOR_PROGRAM"
        if getattr(phase, "vectorize", False) == "required"
        else "ARC_SCALAR_PROGRAM"
    )


def function_values(facts):
    """``(inputs, outputs)`` of the verbatim function, the shared
    derivation of `allo.spmw_candidate.verbatim_values`."""
    from ..spmw_candidate import verbatim_values

    inputs, outputs, _locals = verbatim_values(facts)
    return inputs, outputs


def normalize_apu_v1_program(record, target, *, reduction_order=None):
    """State an ``APUv1Program``'s semantics in the shared program contract
    through the shared front stages (customize, match, coverage with the
    verbatim exemption, semantic facts with every store's retained effect).

    Returns an ``allo.spmw_pipeline.SourceProgram`` whose trace holds the
    phase's real matches plus one anchor match of the whole-function
    operation, and whose facts (``verbatim=True``, ``frontend`` = the
    record) carry the function's `StoreEffect` set.
    """

    from ..spmw_pipeline import SourceProgram, matcher_view

    if not is_apu_v1_program(record):
        raise TypeError("normalize_apu_v1_program takes an APUv1Program record")
    if getattr(target, "name", None) != "apu_v1":
        raise ValueError("APUv1Program requires the apu_v1 target")
    if reduction_order is not None:
        # The record spells out its own grouping; the policy selects among
        # schedules of ordinary source workloads only.
        raise ValueError(_REDUCTION_ORDER_MESSAGE)
    (phase,) = record.phases
    schedule = customize_apu_v1_phase(phase)
    module = schedule.module
    function = schedule.top_func_name
    view = matcher_view(target)
    trace = match_workload(view, module)
    # Every store of the function is implemented by the function itself;
    # its coverage problems are not dropped effects, and the contracts the
    # matcher proved for the matched stores stay on their matches.
    trace.verbatim_functions = frozenset({function})
    require_match_coverage(trace, target=view, module=module)
    source_names = tuple(inspect.signature(phase.kernel).parameters)
    facts = analyze_program(
        view,
        module,
        trace,
        function=function,
        parameter_names=source_names,
        result_names=tuple(phase.result_names),
        verbatim=True,
    )
    facts = dataclasses.replace(facts, frontend=record)
    reads, writes = function_values(facts)
    if not writes:
        raise ValueError(
            f"APUv1Program {record.name!r} writes no buffer: compiling it would "
            "have no external effect"
        )
    ordinals = {p.name: p.ordinal for p in facts.parameters}
    by_name = {p.name: p for p in facts.parameters}
    result, *sinks = writes
    anchor = MatchedOp(
        target_op_name=_operation_name(phase),
        func_name=function,
        work_id=(),
        enclosing_loops=[],
        operands=[
            operand_binding(
                "x" if position == 0 else f"in{position}",
                name,
                ordinals[name],
                [],
                coordinate_forms(),
                _memref_text(by_name[name]),
            )
            for position, name in enumerate(reads)
        ],
        result_memref_name=result,
        op_range=("record", "record"),
        extra={
            "store_indices": [],
            "store_coords": coordinate_forms(),
            "result_memref_type": _memref_text(by_name[result]),
            "result_is_local": False,
            "output_sinks": [(name, "") for name in sinks],
            "output_sink_refs": [(record_ref(ordinals[name]), False) for name in sinks],
            "loop_names": {},
            "whole_function": True,
            _SCHEDULE_KEY: schedule,
        },
        result_value_ref=by_name[result].value_ref,
    )
    # The work item the common pipeline schedules is the whole function:
    # the anchor alone is enumerated, placed and launched.  The matched
    # operations stay in the facts (``facts.trace`` is the matched trace;
    # their contracts were checked by the matcher), not in the scheduled
    # trace, where their operand placements would be judged against the
    # target's op patterns the verbatim program never uses.
    scheduled = MatchTrace(
        target_name=trace.target_name,
        module_name=trace.module_name,
        matches=[anchor],
        source_value_refs=dict(trace.source_value_refs),
        coverage=trace.coverage,
    )
    scheduled.verbatim_functions = trace.verbatim_functions
    stamp_program_facts(trace, facts)
    stamp_program_facts(scheduled, facts)
    return SourceProgram(
        workload=RecordWorkload(record, source_names),
        target=target,
        match_target=view,
        schedule=schedule,
        trace=scheduled,
        facts=facts,
        reduction_order=None,
        parameter_names=tuple(facts.parameter_names),
    )


def _memref_text(parameter) -> str:
    dims = "x".join(str(int(d)) for d in parameter.shape)
    text = parameter.element.text
    return f"memref<{dims}x{text}>" if dims else f"memref<{text}>"


# --------------------------------------------------------------------------- #
# The record's candidate and realized program
# --------------------------------------------------------------------------- #


def _anchor(matches):
    anchors = [m for m in matches if m.extra.get("whole_function")]
    if len(anchors) != 1:
        raise ValueError("an APUv1Program work item has exactly one anchor match")
    return anchors[0]


def _lowering(anchor, record, target) -> CompiledAPUv1Program:
    """The lowering of the record's phase from the retained schedule the
    shared front stages customized; built once per normalized program."""
    lowered = anchor.extra.get(_LOWERING_KEY)
    if lowered is None:
        lowered = CompiledAPUv1Program(
            record, target, schedule=anchor.extra[_SCHEDULE_KEY]
        )
        anchor.extra[_LOWERING_KEY] = lowered
    return lowered


@dataclass
class APUv1RecordCandidate:
    """The payload of an ``APUv1Program`` candidate: the record, its lowering
    and the facts."""

    record: Any
    lowered: CompiledAPUv1Program
    facts: Any

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def kind(self) -> str:
        return (
            "scalar_arc_c"
            if self.lowered.native_vector_lowering is None
            else "native_vector"
        )

    @property
    def route(self) -> str:
        return self.lowered.route

    def decision(self) -> tuple:
        return (self.kind, self.route, program_fingerprint(self.lowered))

    def matches_request(self, request) -> bool:
        return isinstance(request, str) and request in (
            self.name,
            self.kind,
            self.route,
        )

    def realized_transfers(self):
        """The data movement and allocation the lowering's runtime performs
        on every call: every source argument is copied to L4 (and read back
        when it is an output), returned buffers are read back, and the
        scalar program's local arrays live in one scratch allocation."""
        from ..spmw_candidate import CandidateTransfer

        outputs = {argument.name for argument in self.lowered.outputs}
        reads, writes = function_values(self.facts)
        semantic = set(reads) | set(writes)
        transfers = []
        for argument in self.lowered.source_arguments:
            if argument.name in outputs:
                transfers.append(
                    CandidateTransfer(
                        argument.name, "inout", ("host_copy", "l4", "host_readback")
                    )
                )
            else:
                # The runtime copies every source argument to L4; one the
                # function never reads is a physical copy, not an input.
                transfers.append(
                    CandidateTransfer(
                        argument.name,
                        "in",
                        ("host_copy", "l4"),
                        physical=argument.name not in semantic,
                    )
                )
        for argument in self.lowered.outputs:
            if argument.source == "result":
                transfers.append(
                    CandidateTransfer(argument.name, "out", ("l4", "host_readback"))
                )
        allocations = (
            ((self.lowered.scratch_role, int(self.lowered.scratch_bytes), "l4"),)
            if int(self.lowered.scratch_bytes)
            else ()
        )
        return tuple(transfers), allocations

    def effects(self):
        """The record's effects in the common candidate contract, derived
        from the function's retained store effects: the buffers it reads
        (including arguments whose initial values survive a partial or
        guarded write), the buffers it writes, L4 residency of every value,
        one whole-function launch advancing every store effect, and the
        transfers and scratch allocation its runtime performs."""
        from ..spmw_candidate import (
            CandidateCoverage,
            CandidateEffects,
            CandidateLaunch,
            CandidateValueLayout,
        )

        facts = self.facts
        reads, writes = function_values(facts)
        values = tuple(dict.fromkeys(reads + writes))
        (phase,) = self.record.phases
        transfers, allocations = self.realized_transfers()
        return CandidateEffects(
            coverage=CandidateCoverage(
                facts.function, (_operation_name(phase),), reads, writes
            ),
            value_layouts=tuple(
                CandidateValueLayout(name, (), "l4", None) for name in values
            ),
            transfers=transfers,
            launches=(
                CandidateLaunch(
                    index=0,
                    kind=self.kind,
                    operations=tuple(range(len(facts.effects))),
                    instances=(0, 1),
                    reduction=None,
                    accumulator=None,
                    covers=(self.route, program_fingerprint(self.lowered)),
                    whole=True,
                ),
            ),
            abi_inputs=reads,
            abi_outputs=writes,
            device_scope="APUC 0, whole function, one launch",
            allocations=allocations,
        )


def apu_v1_program_candidates(view, matches) -> list:
    """The record's one candidate placement: every ABI buffer in L4, the
    lowering of the retained schedule as payload, its effects declared."""
    from ..spmw_autoschedule import Placement

    anchor = _anchor(matches)
    facts = anchor.extra["program_facts"]
    lowered = _lowering(anchor, facts.frontend, view)
    payload = APUv1RecordCandidate(facts.frontend, lowered, facts)
    reads, writes = function_values(facts)
    l4 = view.l4
    return [
        Placement(
            placements={name: l4 for name in dict.fromkeys(reads + writes)},
            mode=payload.name,
            extra={
                "candidate_plan": payload.route,
                "candidate_decision": payload.decision(),
                "candidate_families": (payload.kind,),
                "payload": payload,
                "whole_device_launch": True,
                "launch_count": 1,
                "effects": payload.effects(),
            },
        )
    ]


class APUv1RecordProgram:
    """The realized program of an ``APUv1Program`` candidate: the lowering,
    run on the board, as host functional C, or priced by the shared
    virtual runner."""

    INTROSPECTION = (
        "record",
        "program",
        "lowered",
        "route",
        "selection",
        "facts",
        "native_vector_lowering",
        "device_source",
        "scratch_bytes",
        "hybrid_manifest",
        "arguments",
        "source_arguments",
        "outputs",
        "artifact",
    )

    def __init__(self, candidate: APUv1RecordCandidate, selection="cost"):
        self.candidate = candidate
        self.record = candidate.record
        self.lowered = candidate.lowered
        self.program = candidate.lowered
        self.facts = candidate.facts
        self.route = candidate.route
        self.selection = selection
        self.compiled = None

    def __getattr__(self, name):
        if name in {
            "native_vector_lowering",
            "device_source",
            "scratch_bytes",
            "hybrid_manifest",
            "source_arguments",
            "outputs",
            "artifact",
        }:
            return getattr(self.lowered, name)
        raise AttributeError(name)

    @property
    def arguments(self):
        """The call's source arguments (the compiled artifact's signature);
        ``lowered.arguments`` also lists returned result buffers."""
        return self.lowered.source_arguments

    def attach(self, compiled) -> None:
        self.compiled = compiled

    def virtual_extra(self) -> dict:
        graph = None if self.compiled is None else self.compiled.execution_graph
        metadata = getattr(graph, "metadata", {}) or {}
        return {
            "outputs": {},
            "program": self.record.name,
            "scalar_arc": self.lowered.native_vector_lowering is None,
            "native_vector_route": (
                None
                if self.lowered.native_vector_lowering is None
                else self.lowered.native_vector_lowering.route
            ),
            "cost_validity": metadata.get("cost_validity"),
        }

    def check_virtual_inputs(self, inputs) -> None:
        self.lowered._checked_sources(inputs)

    def run(self, compiled, inputs):
        backend = compiled.backend
        if backend == "device":
            backend = None
        self.lowered.backend = backend
        return self.lowered.run(**inputs)


__all__ = [
    "APUv1RecordCandidate",
    "APUv1RecordProgram",
    "apu_v1_program_candidates",
    "function_values",
    "is_apu_v1_program",
    "normalize_apu_v1_program",
    "program_fingerprint",
]
