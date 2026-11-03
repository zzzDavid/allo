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
from .spmw_semantics import ProgramFacts, analyze_program


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
        return self.schedule.module

    def program_facts(self, *, function: str | None = None) -> ProgramFacts:
        """The shared semantic facts (derived once, cached)."""

        if self.facts is None or (
            function is not None and self.facts.function != function
        ):
            self.facts = analyze_program(
                self.match_target,
                self.schedule.module,
                self.trace,
                function=function,
                parameter_names=self.parameter_names,
            )
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
        program.program_facts(function=getattr(schedule, "top_func_name", None))
    return program


__all__ = ["SourceProgram", "matcher_view", "normalize_source"]
