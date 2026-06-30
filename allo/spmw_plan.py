# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower an autoscheduler candidate to the canonical performance DAG.

This is intentionally upstream of both code generation and virtual execution.
The graph contains candidate decisions (placement, folding, and ordering), while
the target supplies resource bindings and analytical cycle formulas.
"""

from __future__ import annotations

from .perf import Activity, ExecutionGraph, Invocation, ResourceRequest
from .spmw_tripcount import resolve_trip_count


def _mapping_env(target) -> dict[str, int]:
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
    """Strip matcher-added work-id suffixes from one replicated kernel name."""
    name = match.func_name
    for coordinate in reversed(match.work_id):
        suffix = f"_{coordinate}"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _loop_metrics(target, match) -> dict[str, int]:
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
        "reduction_extent": reduction,
        "iterations": iterations,
    }


def _resource_requests(primitive, work_id) -> tuple[ResourceRequest, ...]:
    requests = []
    for handle in primitive.resources:
        requests.append(
            ResourceRequest(
                resource=handle.qualified_name,
                instances=(handle.instance_for(work_id),),
            )
        )
    return tuple(requests)


def _unwrap_handle(handle):
    return getattr(handle, "home_handle", handle)


def _same_handle(pattern, handle):
    """Match a concrete placement handle against a move endpoint."""
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
    """Resolve declared device moves structurally, without a backend ctx.

    Preloads end at the placed handle; storebacks start from it. Host-scope
    moves and control self-moves are excluded. Ambiguity is a plan-construction
    error rather than a silently selected backend special case.
    """
    out = []
    seen = set()
    from .spmw_target import Register

    for handle in handles:
        # A bank/memory placement is already at its storage tier. This helper
        # resolves register materialization only; explicit memory-to-memory and
        # host transfers are first-class activities supplied by their plan pass.
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
        # A placement can legitimately need no move (already in a bank). More
        # than one distinct route needs an explicit route in the candidate IR.
        if len(matches) > 1:
            names = sorted(move.name for move in matches)
            raise ValueError(
                f"ambiguous {phase} move for placement {handle!r}: {names}"
            )
        if matches and matches[0].name not in seen:
            move = matches[0]
            if move.timing_model is None:
                raise KeyError(f"target move {move.name!r} has no timing_model")
            seen.add(move.name)
            out.append(move)
    return out


def _add_primitive_activity(
    graph,
    primitive,
    activity_id,
    work_id,
    metrics,
    attributes,
    deps,
):
    graph.add(
        Activity(
            id=activity_id,
            primitive=primitive.name,
            timing_model=primitive.timing_model,
            invocation=Invocation(metrics=metrics, attributes=attributes),
            resources=_resource_requests(primitive, work_id),
            depends_on=tuple(deps),
            label=f"{attributes['func_name']}:{primitive.name}",
        )
    )
    return activity_id


def build_execution_graph(target, trace, layout) -> ExecutionGraph:
    """Build the exact resource/timing graph for one candidate placement."""
    if not getattr(target, "has_performance_model", False):
        raise ValueError(f"target {target.name!r} has no resource performance model")
    layouts = _layouts_by_function(trace, layout)
    graph = ExecutionGraph(
        name=f"{trace.module_name}@{target.name}",
        metadata={"target": target.name, "module": trace.module_name},
    )

    function_order = []
    for match in trace.matches:
        logical_name = _logical_function_name(match)
        if logical_name not in function_order:
            function_order.append(logical_name)

    previous_function_terminals: tuple[str, ...] = ()
    for func_name in function_order:
        matches = [
            match
            for match in trace.matches
            if _logical_function_name(match) == func_name
        ]
        streams: dict[tuple[int, ...], list] = {}
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
            coordinate_text = ".".join([str(coordinate) for coordinate in work_id])
            prefix = f"{func_name}:{coordinate_text or 'root'}"
            deps = list(previous_function_terminals)
            operand_memrefs = []
            for match in stream_matches:
                for operand in match.operands:
                    name = operand.memref_name
                    if name is not None and name not in operand_memrefs:
                        operand_memrefs.append(name)
            operand_handles = [
                placement.placements[name]
                for name in operand_memrefs
                if name in placement.placements
            ]
            result_memrefs = []
            for match in stream_matches:
                name = match.result_memref_name
                if name is not None and name not in result_memrefs:
                    result_memrefs.append(name)
            result_handles = [
                placement.placements[name]
                for name in result_memrefs
                if name in placement.placements
            ]

            for move_index, move in enumerate(
                _moves_for_handles(target, operand_handles, phase="pre")
            ):
                activity_id = f"{prefix}:pre:{move_index}:{move.name}"
                deps = [
                    _add_primitive_activity(
                        graph,
                        move,
                        activity_id,
                        work_id,
                        dict(move.performance_inputs),
                        {"func_name": func_name, "work_id": work_id, "phase": "pre"},
                        deps,
                    )
                ]

            for op_index, match in enumerate(stream_matches):
                primitive = target.op(match.target_op_name)
                if primitive.timing_model is None:
                    raise KeyError(
                        f"target op {match.target_op_name!r} has no timing_model"
                    )
                metrics = dict(primitive.performance_inputs)
                metrics.update(_loop_metrics(target, match))
                extra = getattr(placement, "extra", {}) or {}
                metrics["n_fibers"] = max(1, int(extra.get("n_fibers", 1)))
                metrics["batch"] = max(1, int(match.extra.get("batch_dim", 1)))
                activity_id = f"{prefix}:op:{op_index}:{primitive.name}"
                deps = [
                    _add_primitive_activity(
                        graph,
                        primitive,
                        activity_id,
                        work_id,
                        metrics,
                        {
                            "func_name": func_name,
                            "work_id": work_id,
                            "placement_mode": getattr(placement, "mode", ""),
                            "phase": "compute",
                        },
                        deps,
                    )
                ]

            for move_index, move in enumerate(
                _moves_for_handles(target, result_handles, phase="post")
            ):
                activity_id = f"{prefix}:post:{move_index}:{move.name}"
                deps = [
                    _add_primitive_activity(
                        graph,
                        move,
                        activity_id,
                        work_id,
                        dict(move.performance_inputs),
                        {"func_name": func_name, "work_id": work_id, "phase": "post"},
                        deps,
                    )
                ]
            if deps:
                function_terminals.append(deps[-1])
        previous_function_terminals = tuple(function_terminals)
    return graph


def estimate_candidate(target, trace, layout, profile=None):
    """Convenience seam shared by autoscheduling and virtual execution."""
    from .pim.performance import virtual_target

    graph = build_execution_graph(target, trace, layout)
    return graph, virtual_target(target, profile).evaluate(graph)
