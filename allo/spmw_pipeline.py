# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared front stages of the Tenon compilation pipeline (release gate G9).

Every ordinary source workload, whatever its target, passes through the same
three stages before any backend planner sees it:

1. **frontend normalization**: ``customize`` lowers the ``@allo`` function to
   the retained MLIR module;
2. **matching and semantic validation**: ``match_workload`` matches the
   module against the target's operation patterns and classifies every
   store; ``require_match_coverage`` rejects dropped effects, unsupported
   accumulator initializers, unproven conversions and copy contracts;
3. **semantic analysis**: :func:`allo.spmw_semantics.analyze_program`
   derives the shared program facts (parameters, contractions with loop
   roles and accesses, accumulator initialization, output sinks, guards,
   element types, reduction policy).

:class:`SourceProgram` is the result handed to the backend stages.  A target
whose matcher view differs from its hardware description (Gemini-II: the
hardware target declares only unmatchable typed primitives) names that view
through ``target.spmw_match_view``; the view is a ``Target`` whose matchable
operations are the source idioms the backend lowers.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from .customize import customize
from .spmw_match import MatchTrace
from .spmw_match_engine import match_workload, require_match_coverage
from .spmw_match import UnsupportedWorkloadError
from .spmw_semantics import ProgramFacts, analyze_program, stamp_program_facts


def matcher_view(target):
    """The target the matcher runs against: ``target.spmw_match_view`` when
    the backend declares a separate source-idiom view, else the target."""

    view = getattr(target, "spmw_match_view", None)
    return target if view is None else view


@dataclass
class SourceProgram:
    """A normalized, matched and semantically analyzed source workload."""

    workload: Any
    target: Any
    match_target: Any
    schedule: Any
    trace: MatchTrace
    facts: ProgramFacts | None = None
    reduction_order: str | None = None
    parameter_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def module(self):
        return getattr(self.schedule, "module", None)

    @property
    def entry_function(self) -> str | None:
        """The public entry function: the workload's own ``func.func``."""

        return getattr(self.schedule, "top_func_name", None)

    def program_facts(self) -> ProgramFacts:
        """The shared semantic facts of the entry function (derived once).

        The facts are stamped on the trace (``trace.semantics``) and on every
        match (``extra["program_facts"]``, and ``extra["semantic_facts"]``
        for a contraction or epilogue) so the backend enumerators and
        contexts read the same facts the semantic stage established.
        """

        if self.facts is None:
            if self.schedule is None:
                raise RuntimeError("a program without a module has no derived facts")
            self.facts = analyze_program(
                self.match_target,
                self.schedule.module,
                self.trace,
                function=self.entry_function,
                parameter_names=self.parameter_names,
            )
            stamp_program_facts(self.trace, self.facts)
            # A dataflow region's work bodies are functions of their own:
            # each matched body gets its facts (its arguments named by the
            # buffers the matches bind them to), stamped on its matches, so
            # every work item's candidates can be derived from and checked
            # against the shared facts.
            bodies = dict.fromkeys(
                m.func_name
                for m in self.trace.matches
                if m.func_name != self.entry_function
            )
            for body in bodies:
                names_by_ref = {}
                for match in self.trace.matches:
                    if match.func_name != body:
                        continue
                    for operand in match.operands:
                        if operand.value_ref is not None and operand.memref_name:
                            names_by_ref.setdefault(
                                operand.value_ref, operand.memref_name
                            )
                    if match.result_value_ref is not None and match.result_memref_name:
                        names_by_ref.setdefault(
                            match.result_value_ref, match.result_memref_name
                        )
                    sinks = match.extra.get("output_sinks", ())
                    refs = match.extra.get("output_sink_refs", ())
                    for (memref, _text), (ref, _local) in zip(sinks, refs):
                        if ref is not None:
                            names_by_ref.setdefault(ref, memref)
                # Buffers the body's stores read only through computed indices
                # or guards (a histogram's samples) bind no match operand; the
                # coverage's retained loads name them.
                coverage = getattr(self.trace, "coverage", None)
                for store in getattr(coverage, "stores", ()) or ():
                    if store.func_name != body:
                        continue
                    for load in (
                        *store.loads,
                        *store.guard_loads,
                        *store.index_loads,
                    ):
                        ref = getattr(load, "value_ref", None)
                        name = getattr(load, "memref_name", None)
                        if ref is not None and name:
                            names_by_ref.setdefault(ref, name)
                facts = analyze_program(
                    self.match_target,
                    self.schedule.module,
                    self.trace,
                    function=body,
                    parameter_names_by_ref=names_by_ref,
                )
                stamp_program_facts(self.trace, facts, primary=False)
        return self.facts


def normalize_source(
    workload, target, *, reduction_order=None, analyze: bool = False
) -> SourceProgram:
    """Run the shared front stages on an ordinary source workload.

    Raises :class:`allo.spmw_match.UnsupportedWorkloadError` when the
    matcher's coverage analysis finds an effect no target operation
    implements, an unsupported accumulator initializer, an unproven
    conversion, an alias or an unsupported copy; this happens before any
    backend planner is consulted, for every target.  The semantic facts
    (:func:`allo.spmw_semantics.analyze_program`) are derived from the same
    trace on the first ``program_facts()`` call (``analyze=True`` derives
    them here); the matcher backends consume the trace directly until their
    candidate contract is unified (gate G9, slice 3).
    """

    match_target = matcher_view(target)
    schedule = customize(workload, enable_tensor=False)
    require_single_function_or_dataflow(
        schedule.module, getattr(schedule, "top_func_name", None)
    )
    trace = match_workload(
        match_target, schedule.module, reduction_order=reduction_order
    )
    # Problems in functions a backend executes verbatim on the host are
    # exempt; when that set is not known yet the check is repeated by
    # ``compile_for_target``, failing closed.
    require_match_coverage(trace, target=match_target, defer_unknown=True)
    try:
        parameter_names = tuple(inspect.signature(workload).parameters)
    except (TypeError, ValueError):
        parameter_names = ()
    program = SourceProgram(
        workload=workload,
        target=target,
        match_target=match_target,
        schedule=schedule,
        trace=trace,
        reduction_order=reduction_order,
        parameter_names=parameter_names,
    )
    if analyze:
        program.program_facts()
    return program


def _attribute_text(operation, name):
    if name not in operation.attributes:
        return None
    text = str(operation.attributes[name])
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


def _walk(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk(child)


def require_single_function_or_dataflow(module, entry: str | None) -> None:
    """Reject interprocedural ordinary source at the shared boundary.

    The supported forms are one plain function (the entry function alone)
    and a dataflow region, whose entry calls retained ``@allo.work`` kernels
    (``df.kernel`` functions carrying the structural scope attributes).  A
    plain helper called from the entry is refused: the matcher would see the
    helper's own argument order and multiplicity, not the entry's ABI and
    call sites, so the program would be lowered with the wrong operand
    binding (release review round 12).  Inline the helper or write the
    program as a dataflow region.
    """

    functions = {}
    body = getattr(module, "body", None)
    for operation in getattr(body, "operations", ()):
        if getattr(getattr(operation, "operation", None), "name", None) != "func.func":
            continue
        name = _attribute_text(operation, "sym_name")
        if name is not None:
            functions[name] = operation
    if entry is None or entry not in functions:
        return
    calls = {}
    for name, function in functions.items():
        for operation in _walk(function):
            if operation.operation.name != "func.call":
                continue
            callee = _attribute_text(operation, "callee")
            if callee is None:
                continue
            callee = callee.lstrip("@")
            calls.setdefault(name, []).append(callee)
    problems = []
    for caller, callees in calls.items():
        for callee in dict.fromkeys(callees):
            target = functions.get(callee)
            if target is not None and "df.kernel" in target.attributes:
                continue
            count = callees.count(callee)
            problems.append(
                f"function '{caller}' calls '{callee}'"
                + (f" {count} times" if count != 1 else "")
            )
    if problems:
        raise UnsupportedWorkloadError(
            "ordinary source workloads are lowered as one function or as a "
            "dataflow region of @allo.work kernels; calls to plain helper "
            "functions are not supported because the helper's own argument "
            "order and call count would replace the entry function's ABI: "
            + "; ".join(problems)
            + ". Inline the helper into the entry function or express the "
            "program as a dataflow region."
        )


__all__ = [
    "SourceProgram",
    "matcher_view",
    "normalize_source",
    "require_single_function_or_dataflow",
]
