# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``APUv1Program`` records as frontends of the shared pipeline (gate G9).

An ``APUv1Program`` is a whole-function program for the GSI Gemini-I: its
one phase is lowered to scalar ARC C from the retained MLIR, or proven by
one native GVML recipe (histogram, record frequency, bivariate moments,
squared-L2 argmin) when the phase requires vectorization.  Neither form is
a contraction the matcher recognizes, so the record has no source route;
it enters the common pipeline as a frontend:

* :func:`normalize_apu_v1_program` lowers the phase once
  (:class:`allo.pim.apu_v1_program.CompiledAPUv1Program`: retained MLIR,
  portable C, scalar ABI, native recipe discovery) and states the program's
  external effects in the shared program contract: one synthetic match of
  the target's ``ARC_SCALAR_PROGRAM`` or ``NATIVE_VECTOR_PROGRAM`` op whose
  operands are the buffers the function reads and whose result and sinks
  are the buffers it writes, with ``ProgramFacts`` whose ``frontend`` is the
  record.
* The ``apu_v1`` enumerator turns the lowering into one candidate
  (:class:`APUv1RecordCandidate`) whose :class:`CandidateEffects` the shared
  driver validates (coverage, L4 value layouts, transfers, the one launch
  over every result instance, ABI); legality, pricing through the
  lowering's retained cost graph, ranking and materialization follow.
* :class:`APUv1RecordProgram` is the realized program the backend context
  materializes; ``allo.compile`` returns the shared ``CompiledCallable``
  whose call runs it (board, host functional C, or the shared virtual
  runner over the cost graph).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from ..spmw_match import MatchedOp, MatchTrace
from ..spmw_match_engine import coordinate_forms
from ..spmw_semantics import ElementType, Parameter, ProgramFacts, stamp_program_facts
from .apu_v1_program import APUv1Program, CompiledAPUv1Program
from .record_frontend import (
    RecordWorkload,
    memref_type_text,
    operand_binding,
    record_ref,
)

_LOWERING_KEY = "apu_v1_lowering"
_REDUCTION_ORDER_MESSAGE = (
    "APUv1Program fixes its reduction grouping in the program record; "
    "reduction_order applies to source workloads"
)


def is_apu_v1_program(workload) -> bool:
    return isinstance(workload, APUv1Program)


def _element(dtype: str) -> ElementType:
    """The shared element type of a scalar-ABI dtype (``i32``, ``ui16``,
    ``f32``, ``index``)."""
    if dtype == "index":
        return ElementType("index", 64, "index", None)
    if dtype.startswith("f"):
        return ElementType(dtype, int(dtype[1:]), "float", None)
    unsigned = dtype.startswith("ui")
    bits = int(dtype[2:] if unsigned else dtype[1:])
    return ElementType(f"i{bits}", bits, "int", False if bits == 1 else not unsigned)


def program_fingerprint(lowered: CompiledAPUv1Program) -> str:
    """Identity of a lowering: its route and the device source it emits."""
    digest = hashlib.sha256()
    digest.update(lowered.route.encode())
    digest.update(b"\0")
    digest.update(lowered.device_source.encode())
    return digest.hexdigest()[:16]


def _reads(lowered) -> tuple[str, ...]:
    return tuple(
        argument.name
        for argument in lowered.arguments
        if argument.source == "argument" and argument.mode in {"in", "both"}
    )


def _writes(lowered) -> tuple[str, ...]:
    return tuple(argument.name for argument in lowered.outputs)


def _operation_name(lowered) -> str:
    return (
        "ARC_SCALAR_PROGRAM"
        if lowered.native_vector_lowering is None
        else "NATIVE_VECTOR_PROGRAM"
    )


def normalize_apu_v1_program(record, target, *, reduction_order=None):
    """State an ``APUv1Program``'s semantics in the shared program contract.

    Returns an ``allo.spmw_pipeline.SourceProgram`` whose trace holds the
    record's synthetic match (the whole-function operation over the
    buffers the lowered program reads and writes) and whose facts carry the
    record as ``frontend``; the lowering itself travels on the match
    (``extra["apu_v1_lowering"]``) to the ``apu_v1`` enumerator.
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
    lowered = CompiledAPUv1Program(record, target)
    arguments = lowered.arguments
    ordinals = {argument.name: index for index, argument in enumerate(arguments)}
    by_name = {argument.name: argument for argument in arguments}
    reads, writes = _reads(lowered), _writes(lowered)
    if not writes:
        raise ValueError(
            f"APUv1Program {record.name!r} writes no buffer: compiling it would "
            "have no external effect"
        )
    operands = [
        operand_binding(
            "x" if position == 0 else f"in{position}",
            name,
            ordinals[name],
            [],
            coordinate_forms(),
            memref_type_text(by_name[name].shape, _element(by_name[name].dtype)),
        )
        for position, name in enumerate(reads)
    ]
    result, *sinks = writes
    instances = math.prod(by_name[result].shape) or 1
    match = MatchedOp(
        target_op_name=_operation_name(lowered),
        func_name=record.name,
        work_id=(),
        enclosing_loops=[],
        operands=operands,
        result_memref_name=result,
        op_range=("record", "record"),
        extra={
            "store_indices": [],
            "store_coords": coordinate_forms(),
            "result_memref_type": memref_type_text(
                by_name[result].shape, _element(by_name[result].dtype)
            ),
            "result_is_local": False,
            "output_sinks": [(name, "") for name in sinks],
            "output_sink_refs": [(record_ref(ordinals[name]), False) for name in sinks],
            "loop_names": {},
            # One whole-function launch computes every element of the
            # result buffer; the shared validator checks the candidate's
            # launch against this extent.
            "result_instances": int(instances),
            "accumulator_init": None,
            "whole_function": True,
            "apu_v1_route": lowered.route,
            _LOWERING_KEY: lowered,
        },
        result_value_ref=record_ref(ordinals[result]),
    )
    trace = MatchTrace(
        target_name="apu_v1",
        module_name=record.name,
        matches=[match],
        source_value_refs={index: record_ref(index) for index in range(len(arguments))},
        coverage=None,
    )
    parameters = tuple(
        Parameter(
            argument.name,
            index,
            tuple(argument.shape),
            _element(argument.dtype),
            record_ref(index),
        )
        for index, argument in enumerate(arguments)
    )
    facts = ProgramFacts(
        function=record.name,
        parameters=parameters,
        contractions=(),
        epilogues=(),
        other_matches=(match,),
        foreign_functions=(),
        trace=trace,
        frontend=record,
    )
    stamp_program_facts(trace, facts)
    return SourceProgram(
        workload=RecordWorkload(
            record, [argument.name for argument in lowered.source_arguments]
        ),
        target=target,
        match_target=matcher_view(target),
        schedule=None,
        trace=trace,
        facts=facts,
        reduction_order=None,
        parameter_names=tuple(argument.name for argument in arguments),
    )


# --------------------------------------------------------------------------- #
# The record's candidate and realized program
# --------------------------------------------------------------------------- #


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

    def effects(self):
        """The lowering's effects in the common candidate contract: it reads
        and writes whole L4 buffers (the scalar ABI), moves them between the
        host and L4, and runs once over every result instance."""
        from ..spmw_candidate import (
            CandidateCoverage,
            CandidateEffects,
            CandidateLaunch,
            CandidateTransfer,
            CandidateValueLayout,
        )

        lowered = self.lowered
        reads, writes = _reads(lowered), _writes(lowered)
        values = tuple(dict.fromkeys(reads + writes))
        transfers = []
        for name in values:
            if name in reads and name in writes:
                transfers.append(
                    CandidateTransfer(
                        name, "inout", ("host_copy", "l4", "host_readback")
                    )
                )
            elif name in reads:
                transfers.append(CandidateTransfer(name, "in", ("host_copy", "l4")))
            else:
                transfers.append(
                    CandidateTransfer(name, "out", ("l4", "host_readback"))
                )
        (match,) = self.facts.trace.matches
        return CandidateEffects(
            coverage=CandidateCoverage(
                self.record.name, (match.target_op_name,), reads, writes
            ),
            value_layouts=tuple(
                CandidateValueLayout(name, (), "l4", None) for name in values
            ),
            transfers=tuple(transfers),
            launches=(
                CandidateLaunch(
                    index=0,
                    kind=self.kind,
                    operations=(0,),
                    instances=(0, int(match.extra["result_instances"])),
                    reduction=None,
                    accumulator=None,
                    covers=(self.route, program_fingerprint(lowered)),
                ),
            ),
            abi_inputs=reads,
            abi_outputs=writes,
            device_scope="APUC 0, whole function, one launch",
        )


def apu_v1_program_candidates(view, matches) -> list:
    """The record's one candidate placement: every ABI buffer in L4, the
    lowering as payload, its effects declared."""
    from ..spmw_autoschedule import Placement

    (match,) = matches
    facts = match.extra["program_facts"]
    lowered = match.extra[_LOWERING_KEY]
    payload = APUv1RecordCandidate(facts.frontend, lowered, facts)
    l4 = view.l4
    names = dict.fromkeys(_reads(lowered) + _writes(lowered))
    return [
        Placement(
            placements={name: l4 for name in names},
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
    "is_apu_v1_program",
    "normalize_apu_v1_program",
    "program_fingerprint",
]
