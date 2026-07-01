# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower a scheduled SPMW candidate through an executable CostSpec."""

from __future__ import annotations

from .perf import BoundCostSpec, CostEvent, CostSpec, ExecutionGraph
from .spmw_tripcount import resolve_trip_count


def _mapping_env(target):
    env = {}
    for unit in target._walk():
        extent = 1
        for factor in unit.mapping:
            extent *= factor
        env[unit.name] = extent
    return env


def _layouts_by_function(trace, layout):
    functions = []
    for match in trace.matches:
        if match.func_name not in functions:
            functions.append(match.func_name)
    if isinstance(layout, (list, tuple)):
        layouts = list(layout)
        if len(layouts) != len(functions):
            raise ValueError(
                f"received {len(layouts)} placements for {len(functions)} functions"
            )
    else:
        layouts = [layout] * len(functions)
    return dict(zip(functions, layouts))


def _logical_function_name(match):
    name = match.func_name
    for coordinate in reversed(match.work_id):
        suffix = f"_{coordinate}"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


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


def _lookup_buffer_metrics(buffer_metrics, name):
    if not name:
        return None
    candidates = [name]
    if name.startswith("local_"):
        candidates.append(name[len("local_") :])
    folded = {key.casefold(): value for key, value in buffer_metrics.items()}
    for candidate in candidates:
        if candidate in buffer_metrics:
            return buffer_metrics[candidate]
        if candidate.casefold() in folded:
            return folded[candidate.casefold()]
    return None


def _host_move_metrics(resolved, buffer_metrics):
    metrics = _lookup_buffer_metrics(buffer_metrics, resolved.buffer_role)
    return dict(metrics or {})


def build_execution_graph(
    target,
    trace,
    layout,
    cost_spec,
    *,
    host_moves=(),
    buffer_metrics=None,
):
    """Execute a cost program over one target-bound autoscheduler candidate."""
    bound_cost = _bind_cost(cost_spec, target)
    buffer_metrics = dict(buffer_metrics or {})
    layouts = _layouts_by_function(trace, layout)
    graph = ExecutionGraph(
        name=f"{trace.module_name}@{target.name}",
        metadata={
            "target": target.name,
            "module": trace.module_name,
            "cost": bound_cost.spec.name,
            "cost_fingerprint": bound_cost.fingerprint,
        },
    )

    function_order = []
    for match in trace.matches:
        logical_name = _logical_function_name(match)
        if logical_name not in function_order:
            function_order.append(logical_name)

    # Host-to-device transfers are an explicit prefix. The current recorded
    # host-move surface does not retain launch positions, so intermediate
    # ingress transfers are conservatively complete before device execution.
    # Gathers form an explicit suffix below.
    previous_function_terminals = ()
    ingress = [
        resolved
        for resolved in host_moves
        if getattr(getattr(resolved, "verb", None), "name", None) != "gather"
    ]
    egress = [
        resolved
        for resolved in host_moves
        if getattr(getattr(resolved, "verb", None), "name", None) == "gather"
    ]
    for index, resolved in enumerate(ingress):
        previous_function_terminals = tuple(
            _emit_event(
                bound_cost,
                graph,
                resolved.move,
                f"host:ingress:{index}:{resolved.move.name}",
                (),
                _host_move_metrics(resolved, buffer_metrics),
                {
                    "buffer_role": resolved.buffer_role,
                    "phase": "host_ingress",
                },
                previous_function_terminals,
            )
        )

    for function_name in function_order:
        matches = [
            match
            for match in trace.matches
            if _logical_function_name(match) == function_name
        ]
        streams = {}
        stream_order = []
        for match in matches:
            work_id = tuple(match.work_id)
            if work_id not in streams:
                streams[work_id] = []
                stream_order.append(work_id)
            streams[work_id].append(match)

        function_terminals = []
        for work_id in stream_order:
            stream_matches = streams[work_id]
            placement = layouts[stream_matches[0].func_name]
            coordinate_text = ".".join(str(value) for value in work_id)
            prefix = f"{function_name}:{coordinate_text or 'root'}"
            dependencies = list(previous_function_terminals)

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
                    {},
                    {"func_name": function_name, "phase": "pre"},
                    dependencies,
                )

            for index, match in enumerate(stream_matches):
                extra = getattr(placement, "extra", {}) or {}
                operation = target.op(
                    extra.get("operation_name", match.target_op_name)
                )
                metrics = _loop_metrics(target, match)
                operand_shapes = {}
                for operand in match.operands:
                    geometry = _lookup_buffer_metrics(
                        buffer_metrics, operand.memref_name
                    )
                    if geometry is not None:
                        operand_shapes[operand.role] = geometry["shape"]
                result_geometry = _lookup_buffer_metrics(
                    buffer_metrics, match.result_memref_name
                )
                metrics.update(
                    n_fibers=max(1, int(extra.get("n_fibers", 1))),
                    batch=max(1, int(match.extra.get("batch_dim", 1))),
                    lanes=8,
                    operand_shapes=operand_shapes,
                    result_shape=(
                        result_geometry["shape"]
                        if result_geometry is not None
                        else None
                    ),
                    candidate=dict(extra),
                )
                dependencies = _emit_event(
                    bound_cost,
                    graph,
                    operation,
                    f"{prefix}:op:{index}:{operation.name}",
                    work_id,
                    metrics,
                    {
                        "func_name": function_name,
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
                    {},
                    {"func_name": function_name, "phase": "post"},
                    dependencies,
                )
            function_terminals.extend(dependencies)
        previous_function_terminals = tuple(function_terminals)

    for index, resolved in enumerate(egress):
        previous_function_terminals = tuple(
            _emit_event(
                bound_cost,
                graph,
                resolved.move,
                f"host:egress:{index}:{resolved.move.name}",
                (),
                _host_move_metrics(resolved, buffer_metrics),
                {
                    "buffer_role": resolved.buffer_role,
                    "phase": "host_egress",
                },
                previous_function_terminals,
            )
        )
    return graph


def estimate_candidate(target, trace, layout, cost_spec):
    """Lower and evaluate one candidate with the same executable cost program."""
    bound = _bind_cost(cost_spec, target)
    graph = build_execution_graph(target, trace, layout, bound)
    return graph, bound.evaluate(graph)
