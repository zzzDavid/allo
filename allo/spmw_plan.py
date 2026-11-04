# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower a scheduled SPMW candidate through an executable CostSpec."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .perf import BoundCostSpec, CostEvent, CostSpec, ExecutionGraph
from .spmw_autoschedule import _matcher_search_scope, _matcher_work_scope
from .spmw_fingerprint import json_digest
from .spmw_liveness import MatcherValueId, TraceLiveness, trace_liveness
from .spmw_tripcount import resolve_trip_count


def _mapping_env(target):
    env = {}
    for unit in target._walk():
        extent = 1
        for factor in unit.mapping:
            extent *= factor
        env[unit.name] = extent
    return env


def _layouts_by_scope(trace, layout):
    search_scopes = []
    for match in trace.matches:
        scope = _matcher_search_scope(match)
        if scope not in search_scopes:
            search_scopes.append(scope)
    if isinstance(layout, (list, tuple)):
        layouts = list(layout)
        if len(layouts) != len(search_scopes):
            raise ValueError(
                f"received {len(layouts)} placements for "
                f"{len(search_scopes)} matcher scopes"
            )
    else:
        layouts = [layout] * len(search_scopes)
    return dict(zip(search_scopes, layouts))


def _loop_metrics(target, match):
    env = _mapping_env(target)
    extents = []
    for axis in range(len(match.enclosing_loops)):
        extent = resolve_trip_count(match, axis, mapping_env=env)
        extents.append(1 if extent is None else max(0, int(extent)))
    reduction = extents[-1] if extents else 1
    iterations = 1
    for extent in extents:
        iterations *= extent
    return {
        "loop_extents": tuple(extents),
        "reduction_extent": reduction,
        "iterations": iterations,
    }


def _unwrap_handle(handle):
    return getattr(handle, "home_handle", handle)


def _same_handle(pattern, handle):
    from .spmw_target import Memory, MemoryRef, Register

    pattern = _unwrap_handle(pattern)
    handle = _unwrap_handle(handle)
    if pattern is handle:
        return True
    if isinstance(pattern, MemoryRef) and isinstance(handle, MemoryRef):
        return pattern.memory is handle.memory
    if isinstance(pattern, Memory) and isinstance(handle, MemoryRef):
        return pattern is handle.memory
    if isinstance(pattern, MemoryRef) and isinstance(handle, Memory):
        return pattern.memory is handle
    return isinstance(pattern, Register) and pattern is handle


def _placement_register_lanes(operation, match, placement):
    """``Register.lanes`` of the register the placement computes ``match``
    in: the destination's register, else the source registers' common lane
    count. None when the placement names no register (or registers that
    disagree), so the metric is absent rather than guessed (audit C-08)."""
    from .spmw_legality import placed_operands
    from .spmw_target import Register

    placed = placed_operands(operation, match, placement)
    for kinds in (("dst",), ("src",)):
        lanes = {
            int(_unwrap_handle(item.handle).lanes)
            for item in placed
            if item.kind in kinds and isinstance(_unwrap_handle(item.handle), Register)
        }
        if len(lanes) == 1:
            return lanes.pop()
        if lanes:
            return None
    return None


def _moves_for_handles(target, handles, *, phase):
    """Resolve structural register materialization moves for a placement."""
    from .spmw_target import Register

    out = []
    seen = set()
    for handle in handles:
        if not isinstance(_unwrap_handle(handle), Register):
            continue
        matches = []
        for unit in target._walk():
            if unit.mode == "host":
                continue
            for move in unit.moves.values():
                if move.src is move.dst:
                    continue
                endpoint = move.dst if phase == "pre" else move.src
                if _same_handle(endpoint, handle):
                    matches.append(move)
        if len(matches) > 1:
            names = sorted(move.name for move in matches)
            raise ValueError(
                f"ambiguous {phase} move for placement {handle!r}: {names}"
            )
        if matches and matches[0].name not in seen:
            seen.add(matches[0].name)
            out.append(matches[0])
    return out


def _bind_cost(cost_spec, target):
    if isinstance(cost_spec, BoundCostSpec):
        if cost_spec.target is not target:
            raise ValueError("bound cost spec belongs to a different target instance")
        return cost_spec
    if not isinstance(cost_spec, CostSpec):
        raise TypeError("cost must be an executable CostSpec")
    return cost_spec.bind(target)


def _emit_event(bound_cost, graph, primitive, event_id, work_id, metrics, attrs, deps):
    event = CostEvent.create(
        event_id,
        primitive,
        work_id=work_id,
        metrics=metrics,
        attributes=attrs,
    )
    return list(bound_cost.emit(graph, event, deps))


def _freeze_metric_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (str(key), _freeze_metric_value(item))
                        for key, item in value.items()
                    ),
                    key=lambda item: item[0],
                )
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metric_value(item) for item in value)
    raise TypeError(f"unsupported buffer metric value {type(value).__name__!r}")


def _thaw_metric_value(value):
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == "mapping"
        and isinstance(value[1], tuple)
    ):
        return {key: _thaw_metric_value(item) for key, item in value[1]}
    if isinstance(value, tuple):
        return tuple(_thaw_metric_value(item) for item in value)
    return value


@dataclass(frozen=True)
class BufferMetricEntry:
    """Immutable metrics bound to one canonical matcher value."""

    value_id: MatcherValueId
    metrics: tuple[tuple[str, object], ...]

    @classmethod
    def create(cls, value_id, metrics):
        if not isinstance(value_id, MatcherValueId):
            raise TypeError("buffer metric entries require MatcherValueId keys")
        if not isinstance(metrics, Mapping):
            raise TypeError("buffer metrics must be mappings")
        return cls(
            value_id,
            tuple(
                sorted(
                    (
                        (str(key), _freeze_metric_value(value))
                        for key, value in metrics.items()
                    ),
                    key=lambda item: item[0],
                )
            ),
        )

    def as_metrics(self) -> dict:
        return {key: _thaw_metric_value(value) for key, value in self.metrics}

    def manifest(self) -> dict:
        return {
            "value_id": self.value_id.manifest(),
            "metrics": [[key, value] for key, value in self.metrics],
        }


@dataclass(frozen=True)
class BufferMetricManifest:
    """Typed value/host-transfer geometry used by executable cost scoring."""

    entries: tuple[BufferMetricEntry, ...] = ()
    host_bindings: tuple[tuple[int, MatcherValueId], ...] = ()

    def __post_init__(self):
        value_ids = [entry.value_id for entry in self.entries]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("buffer metric manifest repeats a value identity")
        host_indices = [index for index, _value_id in self.host_bindings]
        if len(host_indices) != len(set(host_indices)):
            raise ValueError("buffer metric manifest repeats a host transfer")
        known = set(value_ids)
        if any(value_id not in known for _index, value_id in self.host_bindings):
            raise ValueError("host transfer binding lacks a metric entry")

    @classmethod
    def create(cls, entries=None, *, host_bindings=None):
        entries = entries or {}
        host_bindings = host_bindings or {}
        if not isinstance(entries, Mapping):
            raise TypeError("buffer metric manifest entries must be a mapping")
        if not isinstance(host_bindings, Mapping):
            raise TypeError("host metric bindings must be a mapping")
        frozen_entries = tuple(
            sorted(
                (
                    BufferMetricEntry.create(value_id, metrics)
                    for value_id, metrics in entries.items()
                ),
                key=lambda entry: entry.value_id,
            )
        )
        frozen_host_bindings = tuple(
            sorted(
                ((int(index), value_id) for index, value_id in host_bindings.items()),
                key=lambda item: item[0],
            )
        )
        return cls(frozen_entries, frozen_host_bindings)

    def manifest(self) -> dict:
        return {
            "schema": "matcher-buffer-metrics-v1",
            "entries": [entry.manifest() for entry in self.entries],
            "host_bindings": [
                [index, value_id.manifest()] for index, value_id in self.host_bindings
            ],
        }

    def fingerprint_data(self) -> dict:
        return self.manifest()

    @property
    def fingerprint(self) -> str:
        return json_digest(self.manifest())

    def metrics_for(self, value_id: MatcherValueId | None) -> dict | None:
        if value_id is None:
            return None
        for entry in self.entries:
            if entry.value_id == value_id:
                return entry.as_metrics()
        return None

    def metrics_for_host_transfer(self, index: int) -> dict:
        value_id = self.value_id_for_host_transfer(index)
        return dict(self.metrics_for(value_id) or {})

    def value_id_for_host_transfer(self, index: int) -> MatcherValueId:
        bindings = dict(self.host_bindings)
        if index not in bindings:
            raise ValueError(
                f"buffer metric manifest lacks host transfer binding {index}"
            )
        return bindings[index]


_MEMREF_TYPE = re.compile(r"^memref<(.+)>$")


def _memref_shape(memref_type):
    if not memref_type:
        return None
    matched = _MEMREF_TYPE.match(str(memref_type))
    if matched is None:
        return None
    dimensions = []
    for token in matched.group(1).split("x")[:-1]:
        try:
            dimensions.append(int(token))
        except ValueError:
            return None
    return tuple(dimensions)


def _value_shapes(trace, values: TraceLiveness):
    shapes = {}
    for match in trace.matches:
        for index, operand in enumerate(match.operands):
            value_id = values.value_id_for_operand(match, index)
            shape = _memref_shape(operand.memref_type)
            if value_id is not None and shape is not None:
                shapes.setdefault(value_id, set()).add(shape)
    return {value_id: frozenset(items) for value_id, items in shapes.items()}


def _coerce_buffer_metric_manifest(trace, host_moves, buffer_metrics):
    values = trace_liveness(trace)
    host_moves = tuple(host_moves)
    if isinstance(buffer_metrics, BufferMetricManifest):
        bindings = dict(buffer_metrics.host_bindings)
        missing = [index for index in range(len(host_moves)) if index not in bindings]
        if missing:
            raise ValueError(
                f"buffer metric manifest lacks host transfer bindings {missing}"
            )
        if any(index >= len(host_moves) for index in bindings):
            raise ValueError("buffer metric manifest binds an unknown host transfer")
        return buffer_metrics, values
    if buffer_metrics is None:
        buffer_metrics = {}
    if not isinstance(buffer_metrics, Mapping):
        raise TypeError("buffer_metrics must be a BufferMetricManifest or mapping")
    if not buffer_metrics:
        if host_moves:
            raise ValueError(
                "host transfers require an explicit buffer metric manifest"
            )
        return BufferMetricManifest(), values

    host_roles = {
        getattr(resolved, "buffer_role", None): index
        for index, resolved in enumerate(host_moves)
        if getattr(resolved, "buffer_role", None) is not None
    }
    value_shapes = _value_shapes(trace, values)
    assigned = set()
    entries = {}
    key_values = {}
    for ordinal, (name, metrics) in enumerate(buffer_metrics.items()):
        if not isinstance(name, str):
            raise TypeError("legacy buffer metric keys must be ABI strings")
        value_id = None
        if isinstance(metrics, Mapping) and "shape" in metrics:
            shape = tuple(int(extent) for extent in metrics["shape"])
            candidates = [
                candidate
                for candidate, shapes in value_shapes.items()
                if candidate not in assigned and shapes == frozenset((shape,))
            ]
            if len(candidates) == 1:
                value_id = candidates[0]
        if value_id is None and name in host_roles:
            value_id = MatcherValueId("abi", -1, "argument", ordinal)
        if value_id is None:
            raise ValueError(
                f"buffer metric {name!r} has no structural value binding; "
                "provide BufferMetricManifest"
            )
        existing = entries.get(value_id)
        if existing is not None and dict(existing) != dict(metrics):
            raise ValueError("one structural value has conflicting buffer metrics")
        entries[value_id] = dict(metrics)
        key_values[name] = value_id
        assigned.add(value_id)

    host_bindings = {}
    for index, resolved in enumerate(host_moves):
        role = getattr(resolved, "buffer_role", None)
        if role not in key_values:
            raise ValueError(f"host transfer {index} lacks an exact ABI metric binding")
        host_bindings[index] = key_values[role]
    return (
        BufferMetricManifest.create(
            entries,
            host_bindings=host_bindings,
        ),
        values,
    )


@dataclass(frozen=True)
class HostStep:
    """One host transfer in program order; ``host_index`` indexes ``host_moves``."""

    host_index: int


@dataclass(frozen=True)
class LaunchStep:
    """One kernel launch bound to its matcher group.

    ``operands`` are the region buffers the host program passed to
    ``allo.launch`` (base parameter names, in call order); they equal the
    kernel's declared ABI buffers (`launch_schedule` validates arity and
    binding before any backend sees the schedule).
    """

    kernel: str
    group_id: int
    ordinal: int
    operands: tuple = ()


@dataclass(frozen=True)
class LaunchSchedule:
    """Host transfers and kernel launches in recorded host-program order.

    ``resident`` names buffers already on the device before the program runs.
    ``parameters`` lists the region's ``(name, shape)`` pairs in declaration
    order; its index is the ABI ordinal of ``trace.source_value_refs``.
    """

    steps: tuple
    resident: frozenset = frozenset()
    parameters: tuple = ()

    def manifest(self) -> tuple:
        steps = tuple(
            (
                ("host", step.host_index)
                if isinstance(step, HostStep)
                else (
                    "launch",
                    step.kernel,
                    step.group_id,
                    step.ordinal,
                    tuple(step.operands),
                )
            )
            for step in self.steps
        )
        if self.resident:
            return steps + (("resident", tuple(sorted(self.resident))),)
        return steps


def _kernel_abi_names(module, parameters, entry_kernel):
    """Declared ABI buffers of every kernel of ``module``, by kernel name: the
    region parameters a dataflow kernel binds (``spmw.abi_value_ids``), or
    every parameter for the plain entry function."""
    from .spmw_match_engine import _attr_str, _parse_work_id, _retained_abi_value_ids

    names = [name for name, _shape in parameters]
    out = {}
    body = getattr(module, "body", None)
    for function in getattr(body, "operations", ()):
        if getattr(getattr(function, "operation", None), "name", None) != "func.func":
            continue
        symbol = _attr_str(function.attributes["sym_name"])
        kernel = _parse_work_id(symbol)[0]
        try:
            ids = _retained_abi_value_ids(function)
        except (KeyError, ValueError, TypeError, AttributeError):
            ids = None
        if ids is not None:
            if all(0 <= i < len(names) for i in ids):
                out.setdefault(kernel, tuple(names[i] for i in ids))
        elif kernel == entry_kernel and names:
            out.setdefault(kernel, tuple(names))
    return out


def _check_launch_binding(record, declared):
    """Arity and binding of one ``allo.launch`` against the kernel's ABI."""
    operands = tuple(getattr(token, "base", token) for token in record.operands)
    if declared is None:
        return operands
    if len(operands) != len(declared):
        raise ValueError(
            f"host program launches kernel {record.kernel!r} with "
            f"{len(operands)} operand(s) {list(operands)}; the kernel binds "
            f"{len(declared)} buffer(s) {list(declared)} and every launch must "
            "pass them (sliced references such as X[b] address their base buffer)"
        )
    if operands != tuple(declared):
        raise ValueError(
            f"host program launches kernel {record.kernel!r} with operands "
            f"{list(operands)}, which rebind its ABI {list(declared)}; a launch "
            "must pass the kernel's buffers in declaration order (permuted or "
            "substituted operands are not lowered)"
        )
    return operands


def launch_schedule(host_program, trace, *, module=None) -> LaunchSchedule:
    """Bind a recorded ``HostProgram`` to the matcher groups of ``trace``.

    Every launch's operands are checked against the kernel's declared ABI
    buffers (arity and order) when ``module`` is given; a launch that passes
    no operands, too few, or a permutation is refused here, before selection,
    because no backend lowers a rebinding of a kernel's ABI.
    """
    from .spmw_match_engine import _parse_work_id
    from .spmw_target import HostMoveRecord, LaunchRecord

    parameters = _region_parameters(getattr(host_program, "region", None))
    entry = _parse_work_id(trace.matches[0].func_name)[0] if trace.matches else None
    declared = (
        _kernel_abi_names(module, parameters, entry) if module is not None else {}
    )

    groups_by_kernel = {}
    group_order = []
    for match in trace.matches:
        group_id = _matcher_work_scope(match).group_id
        if group_id not in group_order:
            group_order.append(group_id)
        kernel = _parse_work_id(match.func_name)[0]
        groups = groups_by_kernel.setdefault(kernel, [])
        if group_id not in groups:
            groups.append(group_id)

    steps = []
    host_index = 0
    ordinals = {}
    for record in host_program.steps:
        if isinstance(record, HostMoveRecord):
            steps.append(HostStep(host_index))
            host_index += 1
        elif isinstance(record, LaunchRecord):
            groups = groups_by_kernel.get(record.kernel, [])
            if not groups:
                raise ValueError(
                    f"host program launches kernel {record.kernel!r} with no "
                    "matched implementation"
                )
            if len(groups) > 1:
                raise ValueError(
                    f"host program launch of kernel {record.kernel!r} is "
                    f"ambiguous: it matches groups {groups}"
                )
            group_id = groups[0]
            ordinal = ordinals.get(group_id, 0)
            ordinals[group_id] = ordinal + 1
            operands = _check_launch_binding(record, declared.get(record.kernel))
            steps.append(LaunchStep(record.kernel, group_id, ordinal, operands))
        else:
            raise ValueError(f"unknown host program step {record!r}")

    unlaunched = [group_id for group_id in group_order if group_id not in ordinals]
    if unlaunched and not getattr(host_program, "subset", False):
        raise ValueError(
            f"matched kernel group never launched by the host program: {unlaunched}"
        )
    return LaunchSchedule(
        tuple(steps),
        frozenset(getattr(host_program, "resident", ()) or ()),
        parameters,
    )


def _region_parameters(region) -> tuple:
    """``(name, shape)`` for each region parameter, shape None if unknown."""
    import inspect

    if region is None:
        return ()
    try:
        names = tuple(inspect.signature(region).parameters)
    except (TypeError, ValueError):
        return ()
    try:
        annotations = inspect.get_annotations(region, eval_str=True)
    except (NameError, TypeError):
        annotations = getattr(region, "__annotations__", {}) or {}
    parameters = []
    for name in names:
        shape = getattr(annotations.get(name), "shape", None)
        try:
            shape = None if shape is None else tuple(int(v) for v in shape)
        except (TypeError, ValueError):
            shape = None
        parameters.append((name, shape))
    return tuple(parameters)


def _candidate_execution_graph(target, trace, layout, bound_cost, **options):
    """Delegate pricing to the backend context class of ``target`` when it
    declares ``candidate_execution_graph`` (a classmethod returning an
    ``ExecutionGraph`` for its own candidates, None otherwise)."""

    from .spmw_codegen import _BACKENDS, backend_context_class

    name = getattr(target, "name", None)
    if name not in _BACKENDS:
        return None
    hook = getattr(backend_context_class(name), "candidate_execution_graph", None)
    if hook is None:
        return None
    graph = hook(target, trace, layout, bound_cost, **options)
    if graph is not None:
        from .spmw_candidate import check_graph_consistency

        for placement in layout if isinstance(layout, (list, tuple)) else [layout]:
            check_graph_consistency(placement, graph)
    return graph


def build_execution_graph(
    target,
    trace,
    layout,
    cost_spec,
    *,
    host_moves=(),
    buffer_metrics=None,
    launch_schedule=None,
):
    """Execute a cost program over one target-bound autoscheduler candidate.

    With ``launch_schedule`` None, host ingress is a serial prefix and gathers
    a serial suffix around the groups. With a ``LaunchSchedule``, host
    transfers and launches run serially in its program order.
    """
    bound_cost = _bind_cost(cost_spec, target)
    # A backend whose candidates carry their own realization (the APU vector
    # plans, the Gemini-II launch plans) prices them through its context's
    # ``candidate_execution_graph`` hook; the hook returns None for placements
    # that are not its candidates, which take the generic lowering below.
    hooked = _candidate_execution_graph(
        target,
        trace,
        layout,
        bound_cost,
        host_moves=host_moves,
        buffer_metrics=buffer_metrics,
        launch_schedule=launch_schedule,
    )
    if hooked is not None:
        return hooked
    buffer_manifest, trace_values = _coerce_buffer_metric_manifest(
        trace, host_moves, buffer_metrics
    )
    layouts = _layouts_by_scope(trace, layout)
    graph = ExecutionGraph(
        name=f"{trace.module_name}@{target.name}",
        metadata={
            "target": target.name,
            "module": trace.module_name,
            "cost": bound_cost.spec.name,
            "cost_fingerprint": bound_cost.fingerprint,
            "buffer_metric_fingerprint": buffer_manifest.fingerprint,
        },
    )

    group_order = []
    for match in trace.matches:
        group_id = _matcher_work_scope(match).group_id
        if group_id not in group_order:
            group_order.append(group_id)

    def _emit_host(host_index, resolved, phase_index, dependencies):
        egress = _is_gather(resolved)
        direction = "egress" if egress else "ingress"
        return tuple(
            _emit_event(
                bound_cost,
                graph,
                resolved.move,
                f"host:{direction}:{phase_index}:{resolved.move.name}",
                (),
                _host_transfer_metrics(
                    buffer_manifest.metrics_for_host_transfer(host_index),
                    resolved,
                    launch_schedule,
                ),
                {
                    "buffer_value": buffer_manifest.value_id_for_host_transfer(
                        host_index
                    ).manifest(),
                    "phase": f"host_{direction}",
                },
                dependencies,
            )
        )

    def _emit_group(group_id, dependencies_in, label_prefix):
        matches = [
            match
            for match in trace.matches
            if _matcher_work_scope(match).group_id == group_id
        ]
        streams = {}
        stream_order = []
        for match in matches:
            work_id = _matcher_work_scope(match).work_id
            if work_id not in streams:
                streams[work_id] = []
                stream_order.append(work_id)
            streams[work_id].append(match)

        if any(_matcher_work_scope(match).coalesced_axes for match in matches):
            zero = next(
                (
                    work_id
                    for work_id in stream_order
                    if all(int(value) == 0 for value in work_id)
                ),
                stream_order[0],
            )
            streams = {zero: streams[zero]}
            stream_order = [zero]

        function_terminals = []
        for work_id in stream_order:
            stream_matches = streams[work_id]
            placement = layouts[_matcher_search_scope(stream_matches[0])]
            from .spmw_autoschedule import derive_layout_properties

            placement_metrics = {
                "candidate": derive_layout_properties(target, placement)
            }
            coordinate_text = ".".join(str(value) for value in work_id)
            group_label = f"group{group_id}"
            prefix = f"{label_prefix}:{coordinate_text or 'root'}"
            dependencies = list(dependencies_in)

            operand_memrefs = []
            result_memrefs = []
            for match in stream_matches:
                for operand in match.operands:
                    if (
                        operand.memref_name is not None
                        and operand.memref_name not in operand_memrefs
                    ):
                        operand_memrefs.append(operand.memref_name)
                if (
                    match.result_memref_name is not None
                    and match.result_memref_name not in result_memrefs
                ):
                    result_memrefs.append(match.result_memref_name)
            operand_handles = [
                placement.placements[name]
                for name in operand_memrefs
                if name in placement.placements
            ]
            result_handles = [
                placement.placements[name]
                for name in result_memrefs
                if name in placement.placements
            ]

            for index, move in enumerate(
                _moves_for_handles(target, operand_handles, phase="pre")
            ):
                dependencies = _emit_event(
                    bound_cost,
                    graph,
                    move,
                    f"{prefix}:pre:{index}:{move.name}",
                    work_id,
                    placement_metrics,
                    {
                        "func_name": group_label,
                        "group_id": group_id,
                        "phase": "pre",
                    },
                    dependencies,
                )

            for index, match in enumerate(stream_matches):
                extra = derive_layout_properties(target, placement)
                operation = target.op(extra.get("operation_name", match.target_op_name))
                metrics = _loop_metrics(target, match)
                operand_shapes = {}
                for operand_index, operand in enumerate(match.operands):
                    value_id = trace_values.value_id_for_operand(match, operand_index)
                    geometry = buffer_manifest.metrics_for(value_id)
                    if geometry is not None and "shape" in geometry:
                        operand_shapes[operand.role] = geometry["shape"]
                result_geometry = buffer_manifest.metrics_for(
                    trace_values.value_id_for_result(match)
                )
                lanes = _placement_register_lanes(operation, match, placement)
                if lanes is not None:
                    metrics["lanes"] = lanes
                metrics.update(
                    n_fibers=max(1, int(extra.get("n_fibers", 1))),
                    batch=max(1, int(match.extra.get("batch_dim", 1))),
                    operand_shapes=operand_shapes,
                    result_shape=(
                        result_geometry["shape"]
                        if result_geometry is not None and "shape" in result_geometry
                        else None
                    ),
                    candidate=dict(extra),
                )
                vector_width = getattr(match, "vector_width", 1)
                if vector_width != 1:
                    metrics["vector_width"] = vector_width
                dependencies = _emit_event(
                    bound_cost,
                    graph,
                    operation,
                    f"{prefix}:op:{index}:{operation.name}",
                    work_id,
                    metrics,
                    {
                        "func_name": group_label,
                        "group_id": group_id,
                        "placement_mode": getattr(placement, "mode", ""),
                        "phase": "compute",
                    },
                    dependencies,
                )

            for index, move in enumerate(
                _moves_for_handles(target, result_handles, phase="post")
            ):
                dependencies = _emit_event(
                    bound_cost,
                    graph,
                    move,
                    f"{prefix}:post:{index}:{move.name}",
                    work_id,
                    placement_metrics,
                    {
                        "func_name": group_label,
                        "group_id": group_id,
                        "phase": "post",
                    },
                    dependencies,
                )
            function_terminals.extend(dependencies)
        return tuple(function_terminals)

    indexed_host_moves = tuple(enumerate(host_moves))

    if launch_schedule is not None:
        host_steps = [s for s in launch_schedule.steps if isinstance(s, HostStep)]
        if len(host_steps) != len(indexed_host_moves):
            raise ValueError(
                f"launch schedule has {len(host_steps)} host steps but "
                f"{len(indexed_host_moves)} host moves were supplied"
            )
        frontier = ()
        direction_counts = {False: 0, True: 0}
        for step in launch_schedule.steps:
            if isinstance(step, HostStep):
                if not 0 <= step.host_index < len(indexed_host_moves):
                    raise ValueError(
                        f"launch schedule host index {step.host_index} is out "
                        f"of range for {len(indexed_host_moves)} host moves"
                    )
                resolved = indexed_host_moves[step.host_index][1]
                egress = _is_gather(resolved)
                frontier = _emit_host(
                    step.host_index, resolved, direction_counts[egress], frontier
                )
                direction_counts[egress] += 1
            elif isinstance(step, LaunchStep):
                if step.group_id not in group_order:
                    raise ValueError(
                        f"launch schedule names group {step.group_id}, which is "
                        "not in the trace"
                    )
                label_prefix = (
                    f"group{step.group_id}"
                    if step.ordinal == 0
                    else f"group{step.group_id}#{step.ordinal}"
                )
                frontier = _emit_group(step.group_id, frontier, label_prefix)
            else:
                raise TypeError(f"unknown launch schedule step {step!r}")
        return graph

    # Host-to-device transfers are an explicit prefix. Without a launch
    # schedule, launch positions are unknown, so intermediate ingress
    # transfers are conservatively complete before device execution.
    # Gathers form an explicit suffix below.
    previous_function_terminals = ()
    ingress = [item for item in indexed_host_moves if not _is_gather(item[1])]
    egress = [item for item in indexed_host_moves if _is_gather(item[1])]
    for phase_index, (host_index, resolved) in enumerate(ingress):
        previous_function_terminals = _emit_host(
            host_index, resolved, phase_index, previous_function_terminals
        )

    for group_id in group_order:
        previous_function_terminals = _emit_group(
            group_id, previous_function_terminals, f"group{group_id}"
        )

    for phase_index, (host_index, resolved) in enumerate(egress):
        previous_function_terminals = _emit_host(
            host_index, resolved, phase_index, previous_function_terminals
        )
    return graph


def _host_transfer_metrics(metrics, resolved, launch_schedule):
    """Buffer metrics of one host transfer plus, only when they differ from
    the whole-buffer default, the moved slice and the residency of its buffer.

    ``host_slice`` is the normalized index of a sliced buffer token (an int,
    None for ``:``, or a ``(start, stop)`` pair per dimension);
    ``host_resident`` marks a buffer the schedule declares resident.
    """
    metrics = dict(metrics)
    role = getattr(resolved, "buffer_role", None)
    text = str(role) if role is not None else ""
    base = text.split("[", 1)[0]
    if "[" in text and text.endswith("]"):
        index = []
        for item in text[len(base) + 1 : -1].split(","):
            item = item.strip()
            if item == ":":
                index.append(None)
            elif ":" in item:
                start, stop = item.split(":")
                index.append((int(start or 0), int(stop)))
            else:
                try:
                    index.append(int(item))
                except ValueError:
                    index = None
                    break
        if index is not None:
            metrics["host_slice"] = tuple(index)
    resident = getattr(launch_schedule, "resident", ()) or ()
    if base and base in resident:
        metrics["host_resident"] = True
    return metrics


def _is_gather(resolved) -> bool:
    return getattr(getattr(resolved, "verb", None), "name", None) == "gather"


def estimate_candidate(target, trace, layout, cost_spec):
    """Lower and evaluate one candidate with the same executable cost program."""
    bound = _bind_cost(cost_spec, target)
    graph = build_execution_graph(target, trace, layout, bound)
    return graph, bound.evaluate(graph)


# --------------------------------------------------------------------- #
# Transfer availability: what the host program brings to the device, in
# schedule order, against what every launch reads (release gate G9).
# --------------------------------------------------------------------- #

# The ``host_xfer`` verbs that bring a buffer region onto the device; a
# candidate transfer whose route starts with one of these is host-supplied
# and must be available before the launch that reads it.
HOST_INGRESS_STEPS = frozenset({"host_scatter", "host_broadcast"})


def parse_buffer_role(text):
    """``name`` or ``name[i, :, a:b]`` -> (name, normalized index or None).

    The index is a tuple with one item per indexed dimension: an ``int`` for
    a fixed coordinate, ``(start, stop)`` for a bounded range, ``None`` for a
    whole dimension.  Text that is not an index returns the whole text.
    """
    if "[" not in text:
        return text, None
    base, rest = text.split("[", 1)
    body = rest.rstrip("]")
    index = []
    for item in body.split(","):
        item = item.strip()
        if item == ":":
            index.append(None)
        elif ":" in item:
            start, stop = item.split(":")
            index.append((int(start or 0), int(stop)))
        else:
            try:
                index.append(int(item))
            except ValueError:
                return text, None
    return base, tuple(index)


@dataclass(frozen=True)
class FlatRange:
    """``[start, stop)`` of the row-major flattening of a buffer: the region a
    single bounded range names over a multi-dimensional buffer
    (``W[0:2640]`` over ``W[3, 33, 80]``).  Distinct from a Cartesian box,
    whose every dimension is bounded (``W[0:4, 0, 0]`` is the box
    ``((0, 4), (0, 1), (0, 1))`` and covers four scattered elements)."""

    start: int
    stop: int

    @property
    def count(self) -> int:
        """Number of flattened elements in the range."""
        return self.stop - self.start

    def rows(self, shape) -> tuple:
        """The Cartesian box of whole leading-dimension rows this range
        covers, or ``None`` when it is not whole rows."""
        inner = 1
        for d in shape[1:]:
            inner *= int(d)
        if self.start % inner or self.stop % inner:
            return None
        return ((self.start // inner, self.stop // inner),) + tuple(
            (0, int(d)) for d in shape[1:]
        )


def slice_region(shape, index):
    """The region ``index`` (see `parse_buffer_role`) names in a buffer of
    ``shape``: a Cartesian box ``((lo, hi), ...)`` with one bound per
    dimension (the whole buffer for ``None``), or a `FlatRange` for a single
    bounded range over a multi-dimensional buffer (its flattened extent)."""
    shape = tuple(int(d) for d in shape)
    if index is None:
        return tuple((0, d) for d in shape)
    if len(index) == 1 and len(shape) > 1 and isinstance(index[0], tuple):
        start, stop = index[0]
        total = 1
        for d in shape:
            total *= d
        if not 0 <= start <= stop <= total:
            raise ValueError(f"slice {index} does not fit a buffer of {shape}")
        return FlatRange(int(start), int(stop))
    if len(index) != len(shape):
        raise ValueError(f"index {index} does not fit a buffer of {shape}")
    box = []
    for item, d in zip(index, shape):
        if item is None:
            box.append((0, d))
        elif isinstance(item, tuple):
            start, stop = item
            if not 0 <= start <= stop <= d:
                raise ValueError(f"slice {index} does not fit a buffer of {shape}")
            box.append((start, stop))
        else:
            if not 0 <= int(item) < d:
                raise ValueError(f"index {index} does not fit a buffer of {shape}")
            box.append((int(item), int(item) + 1))
    return tuple(box)


def flat_range(shape, region) -> tuple | None:
    """``(first, count)`` of ``region`` in the row-major flattening of a
    buffer of ``shape``: a `FlatRange` as such; a Cartesian box when it is
    one contiguous range (a scalar's offset is its row-major position), else
    None."""
    shape = tuple(int(d) for d in shape)
    if isinstance(region, FlatRange):
        return region.start, region.count
    box = region
    if len(box) != len(shape):
        return None
    strides = []
    stride = 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    strides.reverse()
    first = sum(int(lo) * s for (lo, _hi), s in zip(box, strides))
    count = 1
    seen_partial = False
    for (lo, hi), d in reversed(list(zip(box, shape))):
        extent = int(hi) - int(lo)
        if extent <= 0:
            return 0, 0
        if seen_partial and extent != 1:
            return None
        count *= extent
        if extent != d:
            seen_partial = True
    return first, count


def access_region(access, loops) -> tuple | None:
    """The box an access covers over its buffer, from its resolved affine
    coordinates and the enclosing loops (`spmw_semantics.LoopAxis`), or
    None when the buffer shape is unknown.

    A coordinate over loop variables spans their trip ranges; a coordinate
    over any other atom (a work-id derived offset, loaded data) spans the
    whole dimension: the region is never smaller than what the body may
    read.  Guards are not applied (an over-approximation).
    """
    shape = getattr(access, "shape", None)
    if not shape:
        return None
    shape = tuple(int(d) for d in shape)
    bounds = {}
    for loop in loops or ():
        trip = loop.trip
        if trip is None or trip <= 0:
            continue
        bounds[loop.var] = (loop.lower, loop.lower + (trip - 1) * loop.step)
    coords = getattr(access, "coords", None) or ()
    box = []
    for d, extent in enumerate(shape):
        coord = coords[d] if d < len(coords) else None
        if coord is None:
            box.append((0, extent))
            continue
        atoms, const = coord
        lo = hi = int(const)
        whole = False
        for atom, coefficient in atoms:
            span = bounds.get(atom)
            if span is None:
                whole = True
                break
            a, b = int(coefficient) * span[0], int(coefficient) * span[1]
            lo += min(a, b)
            hi += max(a, b)
        if whole:
            box.append((0, extent))
        else:
            box.append((max(0, lo), min(extent, hi + 1)))
    return tuple(box)


def _covered(box, boxes) -> bool:
    """Is every point of ``box`` inside some box of ``boxes``?"""
    if any(hi <= lo for lo, hi in box):
        return True
    inside = [
        b
        for b in boxes
        if all(blo < hi and bhi > lo for (lo, hi), (blo, bhi) in zip(box, b))
    ]
    if not inside:
        return False
    cuts = []
    for d, (lo, hi) in enumerate(box):
        points = {lo, hi}
        for b in inside:
            points.update(p for p in b[d] if lo < p < hi)
        cuts.append(sorted(points))
    import itertools

    for cell in itertools.product(*(zip(c, c[1:]) for c in cuts)):
        if not any(
            all(blo <= clo and chi <= bhi for (clo, chi), (blo, bhi) in zip(cell, b))
            for b in inside
        ):
            return False
    return True


def _missing_region(box, boxes) -> tuple:
    """One cell of ``box`` no box of ``boxes`` contains (for diagnostics)."""
    cuts = []
    for d, (lo, hi) in enumerate(box):
        points = {lo, hi}
        for b in boxes:
            points.update(p for p in b[d] if lo < p < hi)
        cuts.append(sorted(points))
    import itertools

    for cell in itertools.product(*(zip(c, c[1:]) for c in cuts)):
        if not any(
            all(blo <= clo and chi <= bhi for (clo, chi), (blo, bhi) in zip(cell, b))
            for b in boxes
        ):
            return cell
    return box


def _facts_accesses(facts):
    """(reads, writes) of one match's retained facts: `BufferAccess`-like
    objects with ``memref_name``, ``value_ref``, ``shape``, ``coords``."""
    reads, writes = [], []
    if facts is None:
        return reads, writes
    if hasattr(facts, "reduction_vars"):
        reads.extend(facts.inputs)
        if not facts.accumulator.local:
            writes.append(facts.accumulator)
            if facts.accumulator_init == "carried":
                reads.append(facts.accumulator)
        writes.extend(facts.sinks)
    elif hasattr(facts, "loop_vars"):
        reads.extend(a for a in facts.operands if not a.local)
        if not facts.result.local:
            writes.append(facts.result)
    return reads, writes


def _value_key(ref, name):
    return ref if ref is not None else ("memref", name)


def check_transfer_availability(trace, launch_schedule, host_moves, placements):
    """Every launch of the host program finds, on the device, the region its
    work body reads of every host-supplied input: brought by an earlier
    ``scatter``/``broadcast`` (the slice it names), declared ``resident``, or
    written by an earlier launch of this program.  Walked in schedule order
    for every launch (repeated launches included) against the retained
    accesses of the body; a missing region is an `InfeasibleSchedule` naming
    the launch, the buffer and the region.  Buffers are followed by retained
    identity (`trace.source_value_refs`, the accesses' ``value_ref``), never
    by name.  Host-supplied means the candidate's declared transfer route
    starts with a step of `HOST_INGRESS_STEPS`; a candidate without declared
    effects is not checked.
    """
    if launch_schedule is None or not trace.matches:
        return
    from .pim.schedule_search import InfeasibleSchedule
    from .spmw_autoschedule import _bucket_for_autoschedule
    from .spmw_candidate import candidate_effects

    refs = getattr(trace, "source_value_refs", None) or {}
    parameters = tuple(getattr(launch_schedule, "parameters", ()) or ())
    key_by_name, shape_by_key, name_by_key = {}, {}, {}
    for ordinal, (name, shape) in enumerate(parameters):
        ref = refs.get(ordinal)
        if ref is None:
            continue
        key_by_name[name] = ref
        name_by_key[ref] = name
        if shape is not None:
            shape_by_key[ref] = tuple(int(d) for d in shape)
    # The full allocation shape of every parameter comes from the normalized
    # program's facts (the entry function's IR types), keyed by retained
    # identity: a factory whose annotations name closure dimensions gives
    # the schedule no shape, and an accessed region is never the allocation.
    semantics = getattr(trace, "semantics", None)
    for parameter in getattr(semantics, "parameters", ()) or ():
        ref = refs.get(parameter.ordinal, parameter.value_ref)
        if ref is None or not parameter.shape:
            continue
        shape_by_key[ref] = tuple(int(d) for d in parameter.shape)
        key_by_name.setdefault(parameter.name, ref)
        name_by_key.setdefault(ref, parameter.name)

    placements = (
        list(placements) if isinstance(placements, (list, tuple)) else [placements]
    )
    groups: dict = {}
    for (_scope, bucket), placement in zip(_bucket_for_autoschedule(trace), placements):
        group_id = _matcher_work_scope(bucket[0]).group_id
        entry = groups.setdefault(group_id, {"matches": [], "effects": None})
        entry["matches"].extend(bucket)
        if entry["effects"] is None:
            entry["effects"] = candidate_effects(placement)

    def regions(group, names, which):
        """{key: (display name, [boxes])} of ``which`` ('reads'/'writes')
        accesses of the group's bodies to the values ``names``."""
        out: dict = {}
        for match in group["matches"]:
            facts = match.extra.get("semantic_facts")
            reads, writes = _facts_accesses(facts)
            for access in reads if which == "reads" else writes:
                if access.memref_name not in names:
                    continue
                key = _value_key(access.value_ref, access.memref_name)
                box = access_region(access, getattr(facts, "loops", ()))
                if box is None:
                    shape = shape_by_key.get(key)
                    if shape is None:
                        continue
                    box = tuple((0, d) for d in shape)
                out.setdefault(key, (access.memref_name, []))[1].append(box)
        return out

    available: dict = {}
    for name in getattr(launch_schedule, "resident", ()) or ():
        key = key_by_name.get(name)
        shape = shape_by_key.get(key)
        if key is not None and shape is not None:
            available.setdefault(key, []).append(tuple((0, d) for d in shape))

    moves = list(host_moves or ())
    for position, step in enumerate(launch_schedule.steps):
        if isinstance(step, HostStep):
            resolved = moves[step.host_index]
            verb = getattr(getattr(resolved, "verb", None), "name", None)
            if verb not in ("scatter", "broadcast"):
                continue
            role = getattr(resolved, "buffer_role", None)
            base = getattr(role, "base", None) if not isinstance(role, str) else None
            index = getattr(role, "index", None) if base is not None else None
            if base is None:
                base, index = parse_buffer_role(str(role))
            key = key_by_name.get(base)
            shape = shape_by_key.get(key)
            if key is None or shape is None:
                continue
            try:
                region = slice_region(shape, index)
            except ValueError as error:
                raise InfeasibleSchedule(f"host {verb} of {role}: {error}") from error
            if isinstance(region, FlatRange):
                # A flat range over a multi-dimensional buffer covers whole
                # rows of the leading dimension, or it is refused.
                box = region.rows(shape)
                if box is None:
                    inner = 1
                    for d in shape[1:]:
                        inner *= d
                    raise InfeasibleSchedule(
                        f"host {verb} of {role}: the flat range [{region.start}, "
                        f"{region.stop}) over a buffer of {shape} must cover whole "
                        f"rows of {inner} elements"
                    )
            else:
                box = region  # a Cartesian box: exactly the elements it names
            available.setdefault(key, []).append(box)
            continue
        if not isinstance(step, LaunchStep):
            continue
        group = groups.get(step.group_id)
        if group is None or group["effects"] is None:
            continue
        effects = group["effects"]
        supplied = {
            t.value
            for t in effects.transfers
            if t.direction in ("in", "inout")
            and not t.physical
            and tuple(t.route)[:1]
            and tuple(t.route)[0] in HOST_INGRESS_STEPS
        }
        for key, (name, boxes) in regions(group, supplied, "reads").items():
            have = available.get(key, [])
            for box in boxes:
                if _covered(box, have):
                    continue
                missing = _missing_region(box, have)
                label = name_by_key.get(key, name)
                raise InfeasibleSchedule(
                    f"launch {step.ordinal} of {step.kernel!r} (step {position}) "
                    f"reads {label!r} ({name!r}) over {list(box)} but the host "
                    f"program has not brought {list(missing)} to the device: no "
                    "earlier scatter/broadcast of that region, not declared "
                    "resident, not produced by an earlier launch"
                )
        produced = {
            t.value for t in effects.transfers if t.direction in ("out", "inout")
        }
        for key, (_name, boxes) in regions(group, produced, "writes").items():
            available.setdefault(key, []).extend(boxes)
