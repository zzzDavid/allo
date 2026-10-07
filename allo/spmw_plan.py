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
    """One kernel launch bound to its matcher group."""

    kernel: str
    group_id: int
    ordinal: int


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
            ("host", step.host_index)
            if isinstance(step, HostStep)
            else ("launch", step.kernel, step.group_id, step.ordinal)
            for step in self.steps
        )
        if self.resident:
            return steps + (("resident", tuple(sorted(self.resident))),)
        return steps


def launch_schedule(host_program, trace) -> LaunchSchedule:
    """Bind a recorded ``HostProgram`` to the matcher groups of ``trace``."""
    from .spmw_match_engine import _parse_work_id
    from .spmw_target import HostMoveRecord, LaunchRecord

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
            steps.append(LaunchStep(record.kernel, group_id, ordinal))
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
        _region_parameters(getattr(host_program, "region", None)),
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
                buffer_manifest.metrics_for_host_transfer(host_index),
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
                metrics.update(
                    n_fibers=max(1, int(extra.get("n_fibers", 1))),
                    batch=max(1, int(match.extra.get("batch_dim", 1))),
                    lanes=8,
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


def _is_gather(resolved) -> bool:
    return getattr(getattr(resolved, "verb", None), "name", None) == "gather"


def estimate_candidate(target, trace, layout, cost_spec):
    """Lower and evaluate one candidate with the same executable cost program."""
    bound = _bind_cost(cost_spec, target)
    graph = build_execution_graph(target, trace, layout, bound)
    return graph, bound.evaluate(graph)
