# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public workload + target + cost compilation facade."""

# Public API spelling intentionally matches Python's built-in compiler verb.
# pylint: disable=redefined-builtin

from __future__ import annotations

import inspect
import math

import numpy as np

from .customize import customize
from .perf import BoundCostSpec, CostSpec
from .pim.apu_v1_program import APUv1Program, compile_apu_v1_program
from .pim.apu_v1_vector_program import (
    APUv1VectorCallable,
    compile_apu_v1_vector_workload,
)
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


def _resolve_cost(target, cost_spec):
    """Bind an executable cost program to this target instance."""
    if cost_spec is None:
        return None
    if isinstance(cost_spec, BoundCostSpec):
        if cost_spec.target is not target:
            raise ValueError("bound cost spec belongs to a different target instance")
        return cost_spec
    if not isinstance(cost_spec, CostSpec):
        raise TypeError("cost must be an executable CostSpec")
    return cost_spec.bind(target)


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


def _buffer_metrics(workload):
    """Extract static operand geometry for executable cost rules.

    Allo workloads commonly use postponed annotations, so resolve them in the
    defining module before inspecting ``TypeAnnotation.shape`` and dtype bits.
    Unknown annotations are simply omitted; cost programs remain free to use
    other event metrics.
    """
    try:
        annotations = inspect.get_annotations(workload, eval_str=True)
    except (NameError, TypeError):
        annotations = getattr(workload, "__annotations__", {}) or {}

    metrics = {}
    for name, annotation in annotations.items():
        if name == "return":
            continue
        shape = getattr(annotation, "shape", None)
        if shape is None:
            continue
        try:
            shape = tuple(int(extent) for extent in shape)
        except (TypeError, ValueError):
            continue
        dtype = getattr(annotation, "dtype", None)
        bits = int(getattr(dtype, "bits", 0) or 0)
        elements = math.prod(shape)
        entry = {"shape": shape, "elements": elements}
        if bits > 0:
            entry.update(dtype_bits=bits, bytes=(elements * bits + 7) // 8)
        metrics[name] = entry
    return metrics


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
        cost=None,
    ):
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.trace = trace
        self.compiled = compiled
        self.cost = cost
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
        if self.cost is None:
            raise RuntimeError("compiled workload has no executable cost spec")
        if self.execution_graph is None:
            raise RuntimeError("compiled workload has no retained execution graph")
        return self.cost.evaluate(self.execution_graph)

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
) -> CompiledCallable | APUv1VectorCallable:
    """Compile ``workload`` for ``target`` and return a NumPy-callable object.

    Parameters
    ----------
    workload : callable or object
        An Allo workload callable, or a module/object exposing ``build()``.
    target : Target or callable
        A built Tenon target or a zero-argument target builder.
    cost : CostSpec
        Executable cost program used by autoscheduling and virtual execution.
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
    bound_cost = _resolve_cost(target, cost)

    if isinstance(workload, APUv1Program):
        if backend not in (None, "virtual", "functional"):
            raise ValueError(
                "APUv1Program supports the device, virtual, or functional backend"
            )
        if host_moves is not None or layout is not None:
            raise ValueError("APUv1Program owns its scalar L4 ABI")
        return compile_apu_v1_program(
            workload, target, cost=bound_cost, backend=backend
        )

    if host_moves is None:
        host_moves = _discover_host_moves(workload)

    schedule = customize(workload, enable_tensor=False)
    # Ordinary contractions use the retained-MLIR layout-plan/vector path.
    # Dataflow regions retain the grouped ``@allo.work`` implementation.
    if target.name == "apu_v1" and not hasattr(workload, "mappings"):
        from .pim.contraction_analysis import NoContractionError

        try:
            return compile_apu_v1_vector_workload(
                workload,
                target,
                schedule,
                cost=bound_cost,
                layout=layout,
                backend=backend,
            )
        except NoContractionError:
            pass
    trace = match_workload(target, schedule.module)
    if target.name == "apu_v1":
        extents = {}
        for match in trace.matches:
            if len(match.work_id) != 1:
                raise TypeError(
                    "APU v1 @allo.work requires scalar mapping=N; N controls "
                    "the number of coalesced GVML groups"
                )
            base = match.func_name
            for coordinate in reversed(match.work_id):
                suffix = f"_{coordinate}"
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            extents[base] = max(extents.get(base, 0), int(match.work_id[0]) + 1)
        for match in trace.matches:
            base = match.func_name
            for coordinate in reversed(match.work_id):
                suffix = f"_{coordinate}"
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            match.extra["spmw_group_count"] = extents[base]
            match.extra["coalesced_spmw_axis"] = "group"
    compiled = compile_for_target(
        target,
        trace,
        layout=layout,
        backend=backend,
        host_moves=host_moves,
        buffer_metrics=_buffer_metrics(workload),
        cost=bound_cost,
    )
    return CompiledCallable(
        workload,
        target,
        schedule,
        trace,
        compiled,
        cost=bound_cost,
    )
