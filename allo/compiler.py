# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public workload + target + cost compilation facade."""

# Public API spelling intentionally matches Python's built-in compiler verb.
# pylint: disable=redefined-builtin

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from .customize import customize
from .perf import CalibrationProfile, VirtualTarget
from .spmw_codegen import RunResult, compile_for_target
from .spmw_match_engine import match_workload


def _materialize_workload(workload):
    if callable(workload):
        return workload
    build = getattr(workload, "build", None)
    if callable(build):
        return build()
    raise TypeError("workload must be callable or expose a callable build()")


def _materialize_target(target):
    if hasattr(target, "name"):
        return target
    if callable(target):
        built = target()
        if hasattr(built, "name"):
            return built
    raise TypeError("target must be a Target or a zero-argument target builder")


def _resolve_performance_model(target, cost):
    """Normalize a cost argument to a bound ``VirtualTarget``."""
    if isinstance(cost, VirtualTarget):
        if getattr(cost.target, "name", None) != target.name:
            raise ValueError(
                f"cost model targets {getattr(cost.target, 'name', None)!r}, "
                f"not {target.name!r}"
            )
        return cost
    if isinstance(cost, (str, Path)):
        cost = CalibrationProfile.load(cost)
    if isinstance(cost, CalibrationProfile):
        from .pim.performance import virtual_target

        return virtual_target(target, cost)
    if cost is not None:
        raise TypeError(
            "cost must be a CalibrationProfile, VirtualTarget, JSON profile path, "
            "or None"
        )
    if getattr(target, "has_performance_model", False):
        from .pim.performance import virtual_target

        return virtual_target(target)
    return None


def _discover_host_moves(workload):
    direct = getattr(workload, "HOST_MOVES", None)
    if direct is not None:
        return list(direct)
    module = inspect.getmodule(workload)
    if module is not None:
        records = getattr(module, "HOST_MOVES", None)
        if records is not None:
            return list(records)
    return None


class CompiledCallable:
    """An Allo-style callable backed by a compiled PIM artifact.

    Calls accept positional or keyword NumPy operands according to the original
    workload signature and return a :class:`RunResult`. When a backend returns
    output arrays, explicitly gathered output arguments are updated in place,
    matching the mutation behavior of Allo's LLVM callable modules.
    """

    def __init__(
        self,
        workload,
        target,
        schedule,
        trace,
        compiled,
        performance_model=None,
    ):
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.trace = trace
        self.compiled = compiled
        self.performance_model = performance_model
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "compiled_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result: RunResult | None = None

    def __call__(self, *args, **kwargs) -> RunResult:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self.run_backend(**bound.arguments)

    def run_backend(self, **inputs) -> RunResult:
        """Invoke with explicit backend-role arrays.

        This compiler-testing escape hatch supports backend ABIs whose lowered
        operand roles differ from the source workload signature. Applications
        should normally call the object directly.
        """
        result = self.compiled.run(**inputs)
        self._copy_outputs(result, inputs)
        self.last_result = result
        return result

    run = __call__

    @property
    def execution_graph(self):
        return self.compiled.execution_graph

    def estimate(self):
        """Evaluate the retained candidate graph without invoking a backend."""
        if self.performance_model is None:
            raise RuntimeError("compiled workload has no analytical performance model")
        if self.execution_graph is None:
            raise RuntimeError("compiled workload has no retained execution graph")
        return self.performance_model.evaluate(self.execution_graph)

    def _copy_outputs(self, result, arguments):
        outputs = result.extra.get("outputs", {}) if result.extra else {}
        if not outputs:
            return
        gather_roles = []
        for resolved in getattr(self.compiled, "host_moves", ()):
            verb = getattr(getattr(resolved, "verb", None), "name", None)
            if verb == "gather" and resolved.buffer_role in arguments:
                gather_roles.append(resolved.buffer_role)

        assignments = {}
        lower_names = {name.lower(): name for name in arguments}
        for output_name, value in outputs.items():
            if output_name in arguments:
                assignments[output_name] = value
            elif output_name.lower() in lower_names:
                assignments[lower_names[output_name.lower()]] = value
        if len(outputs) == 1 and len(gather_roles) == 1:
            assignments.setdefault(gather_roles[0], next(iter(outputs.values())))

        for name, value in assignments.items():
            destination = arguments[name]
            if not isinstance(destination, np.ndarray):
                continue
            source = np.asarray(value)
            if source.shape != destination.shape:
                if source.size != destination.size:
                    raise ValueError(
                        f"backend output {name!r} has shape {source.shape}, "
                        f"but destination has shape {destination.shape}"
                    )
                source = source.reshape(destination.shape)
            np.copyto(destination, source, casting="same_kind")


def compile(
    workload,
    target,
    cost=None,
    *,
    backend=None,
    host_moves=None,
    layout=None,
) -> CompiledCallable:
    """Compile ``workload`` for ``target`` and return a NumPy-callable object.

    Parameters
    ----------
    workload : callable or object
        An Allo workload callable, or a module/object exposing ``build()``.
    target : Target or callable
        A built Tenon target or a zero-argument target builder.
    cost : object
        A ``CalibrationProfile``, ``VirtualTarget``, JSON profile path, or
        ``None`` for the target's default profile.
    backend : str or None
        ``None`` selects the target's normal simulator/device runner;
        ``"virtual"`` evaluates only the analytical performance graph.
    host_moves : sequence or None
        Optional explicit host-transfer records. If omitted, ``HOST_MOVES`` is
        discovered beside the workload.
    layout : object
        Optional preselected placement or placement list.
    """
    workload = _materialize_workload(workload)
    target = _materialize_target(target)
    performance_model = _resolve_performance_model(target, cost)
    if host_moves is None:
        host_moves = _discover_host_moves(workload)

    schedule = customize(workload, enable_tensor=False)
    trace = match_workload(target, schedule.module)
    compiled = compile_for_target(
        target,
        trace,
        layout=layout,
        backend=backend,
        host_moves=host_moves,
        performance_model=performance_model,
    )
    return CompiledCallable(
        workload,
        target,
        schedule,
        trace,
        compiled,
        performance_model=performance_model,
    )
