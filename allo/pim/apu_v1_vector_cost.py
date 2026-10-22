# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower immutable APU v1 vector plans into executable cost graphs.

Traffic is interpreted exclusively from ``TransferRouteStep`` layout
relations.  The cost program never recognizes a kernel name, broadcast flag,
or planner-owned transfer-count annotation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from ..perf import BoundCostSpec, CostEvent, CostSpec, ExecutionGraph
from .apu_v1_layout import APUV1Plan, PlanOperation, Transfer


VR_LANES = 32768
WRITABLE_VRS = 15


def _freeze_inventory_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_inventory_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_inventory_value(item) for item in value)
    raise TypeError(
        "APU v1 operation inventory attributes must be immutable scalar data"
    )


@dataclass(frozen=True)
class APUV1OperationInventoryEntry:
    """One operand-name-free operation and its exact materialized call count."""

    opcode: str
    count: int
    dtype: str = "f16"
    attributes: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        opcode = str(self.opcode).upper()
        if not opcode or not opcode.replace("_", "").isalnum():
            raise ValueError("APU v1 inventory opcode must be an uppercase-style name")
        count = int(self.count)
        if count <= 0:
            raise ValueError("APU v1 inventory call count must be positive")
        source = (
            self.attributes.items()
            if isinstance(self.attributes, Mapping)
            else tuple(self.attributes)
        )
        attributes = tuple(
            sorted(
                (str(name), _freeze_inventory_value(value)) for name, value in source
            )
        )
        if len({name for name, _value in attributes}) != len(attributes):
            raise ValueError("APU v1 inventory attributes must have unique names")
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "dtype", str(self.dtype).lower())
        object.__setattr__(self, "attributes", attributes)

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "opcode": self.opcode,
            "count": self.count,
            "dtype": self.dtype,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class APUV1PlanEstimate:
    """One plan, its retained graph, and its evaluated analytical result."""

    plan: APUV1Plan
    graph: ExecutionGraph
    estimate: object
    operation_inventory: tuple[APUV1OperationInventoryEntry, ...] = ()

    @property
    def cycles(self) -> int:
        return int(self.estimate.cycles)


def _bound_cost(target, cost) -> BoundCostSpec:
    if isinstance(cost, BoundCostSpec):
        if cost.target is not target:
            raise ValueError("bound APU v1 cost spec belongs to another target")
        return cost
    if isinstance(cost, CostSpec):
        return cost.bind(target)
    raise TypeError("cost must be a CostSpec or BoundCostSpec")


def _ceil_div(value, divisor):
    if int(divisor) <= 0:
        raise ValueError("cost-plan divisor must be positive")
    return (int(value) + int(divisor) - 1) // int(divisor)


def _dtype_bytes(plan: APUV1Plan) -> int:
    dtype = str(plan.metadata.get("dtype", "f16")).lower()
    if dtype in {"bool", "i1", "ui1"}:
        return 1
    digits = "".join(character for character in dtype if character.isdigit())
    if not digits:
        raise ValueError(f"cannot derive element width from dtype {dtype!r}")
    return max(1, _ceil_div(int(digits), 8))


def _shape_facts(plan: APUV1Plan):
    shape = dict(plan.metadata.get("problem_shape", {}))
    tiles = dict(plan.metadata.get("tile_sizes", {}))
    axis_extents = dict(plan.metadata.get("axis_extents", {}))
    output_axes = [
        name
        for name, role in dict(plan.metadata.get("loop_roles", {})).items()
        if role == "parallel_output"
    ]
    reduction_axes = [
        name
        for name, role in dict(plan.metadata.get("loop_roles", {})).items()
        if role == "reduction"
    ]
    if not axis_extents and shape:
        axis_extents = shape
    output_tiles = (
        math.prod(
            _ceil_div(axis_extents[axis], max(1, int(tiles.get(axis, 1))))
            for axis in output_axes
        )
        if output_axes
        else 1
    )
    reduction = (
        int(axis_extents[reduction_axes[0]])
        if reduction_axes
        else int(shape.get("K", 1))
    )
    reduction_tile = max(
        1,
        int(tiles.get(reduction_axes[0], 1)) if reduction_axes else 1,
    )
    reduction_tiles = _ceil_div(reduction, reduction_tile)
    temporal = getattr(plan.reduction_strategy, "kind", "none") == "temporal_accumulate"
    vector_calls = output_tiles * (reduction if temporal else reduction_tiles)
    return {
        "output_tiles": max(1, output_tiles),
        "reduction": max(1, reduction),
        "reduction_tile": reduction_tile,
        "reduction_tiles": max(1, reduction_tiles),
        "vector_calls": max(1, vector_calls),
    }


def _operation_calls(plan: APUV1Plan, operation: PlanOperation, facts) -> int:
    recorded = dict(plan.metadata.get("operation_counts", {}))
    name = operation.name.upper()
    if name in {
        "GROUP_REDUCE",
        "REDUCE",
        "GROUP_REDUCE_ADD_F16",
        "GROUP_REDUCE_ADD_S16",
    }:
        fallback = recorded.get("group_reduce_calls", facts["vector_calls"])
    else:
        fallback = recorded.get("vector_compute_calls", facts["vector_calls"])
    scalar_items = recorded.get(
        "scalar_work_items", plan.metadata.get("scalar_work_items")
    )
    # Older plan generators stored scalar iterations in PlanOperation.count;
    # current generators store vector calls.  Retain both representations.
    if scalar_items is not None and int(operation.count) == int(scalar_items):
        return max(1, int(fallback))
    return max(1, int(operation.count or fallback))


def _plan_operation_inventory(plan, facts):
    return tuple(
        APUV1OperationInventoryEntry(
            operation.name,
            _operation_calls(plan, operation, facts),
            operation.dtype,
        )
        for operation in plan.operations
    )


def _operation_inventory(plan, facts, operation_inventory):
    if operation_inventory is None:
        return _plan_operation_inventory(plan, facts)
    inventory = tuple(operation_inventory)
    if any(
        not isinstance(operation, APUV1OperationInventoryEntry)
        for operation in inventory
    ):
        raise TypeError(
            "operation_inventory must contain APUV1OperationInventoryEntry values"
        )
    return inventory


def _operation_handle(target, operation: APUV1OperationInventoryEntry):
    name = operation.opcode
    dtype = str(operation.dtype).lower()
    floating = dtype.startswith(("f", "bf"))
    unsigned = dtype.startswith("ui")
    aliases = {
        "GROUP_REDUCE": (
            "GROUP_REDUCE_ADD_F16"
            if floating
            else "GROUP_REDUCE_ADD_U16" if unsigned else "GROUP_REDUCE_ADD_S16"
        ),
        "REDUCE": (
            "GROUP_REDUCE_ADD_F16"
            if floating
            else "GROUP_REDUCE_ADD_U16" if unsigned else "GROUP_REDUCE_ADD_S16"
        ),
        "ADD": "ADD" if floating else "ADD_U16" if unsigned else "ADD_S16",
        "SUB": "SUB" if floating else "SUB_U16" if unsigned else "SUB_S16",
        "MUL": "MUL" if floating else "MUL_U16" if unsigned else "MUL",
        "AND": "AND_16",
        "OR": "OR_16",
        "XOR": "XOR_16",
        "MUL_F16": "MUL",
        "ADD_F16": "ADD",
        "GROUP_REDUCE_F16": "GROUP_REDUCE_ADD_F16",
        "GROUP_REDUCE_U16": "GROUP_REDUCE_ADD_U16",
        "GROUP_REDUCE_S16": "GROUP_REDUCE_ADD_S16",
    }
    target_name = aliases.get(name, name)
    try:
        return target.op(target_name), target_name
    except (KeyError, ValueError) as error:
        raise ValueError(f"APU v1 plan has unknown/unpriced opcode {name!r}") from error


def _checked_layout_parameter(step, name, derived):
    """Keep optional codegen parameters consistent with the layout algebra."""

    if name in step.parameters and int(step.parameters[name]) != int(derived):
        raise ValueError(
            f"route {step.kind!r} parameter {name!r}={step.parameters[name]} "
            f"disagrees with layout-derived value {derived}"
        )
    return int(derived)


def _route_metrics(plan: APUV1Plan, step):
    relation = step.metrics()
    count = int(relation.call_count)
    source_per_call = int(relation.source_elements_per_call)
    destination_per_call = int(relation.destination_elements_per_call)
    if count <= 0 or source_per_call <= 0 or destination_per_call <= 0:
        raise ValueError("transfer-route metrics must be positive")
    element_bytes = _dtype_bytes(plan)
    return {
        "count": count,
        "source_elements": int(relation.source_elements),
        "destination_elements": int(relation.destination_elements),
        "source_elements_per_call": source_per_call,
        "destination_elements_per_call": destination_per_call,
        "expansion_factor": float(relation.expansion_factor),
        "resident_reuse_factor": int(relation.resident_reuse_factor),
        # Cost rules consume bytes per call and multiply by count.  Retaining
        # the total makes graph inspection/calibration unambiguous without
        # materializing one event for every logical transfer.
        "bytes": source_per_call * element_bytes,
        "total_bytes": source_per_call * element_bytes * count,
        "elements": source_per_call,
        "iterations": source_per_call * count,
    }


def _effective_route_metrics(plan: APUV1Plan, transfer: Transfer, step):
    """Return traffic for the concrete resident-or-streaming realization."""

    metrics = _route_metrics(plan, step)
    accumulator_block = max(1, int(getattr(plan, "accumulator_block", 1)))
    if step.kind == "duplicate_subgroup" and accumulator_block > 1:
        facts = _shape_facts(plan)
        blocked_calls = (
            _ceil_div(facts["output_tiles"], accumulator_block) * facts["reduction"]
        )
        metrics["count"] = blocked_calls
        metrics["source_elements"] = metrics["source_elements_per_call"] * blocked_calls
        metrics["destination_elements"] = (
            metrics["destination_elements_per_call"] * blocked_calls
        )
        metrics["total_bytes"] = metrics["bytes"] * blocked_calls
        metrics["iterations"] = metrics["source_elements_per_call"] * blocked_calls
        metrics["accumulator_block"] = accumulator_block
        return metrics
    if step.kind not in {"dma_l4_l1_32k", "load_vr"}:
        return metrics
    duplicate = next(
        (item for item in transfer.route if item.kind == "duplicate_subgroup"),
        None,
    )
    if duplicate is None:
        return metrics
    rows_per_vr = int(duplicate.parameters["rows_per_vr"])
    temporal_axis = str(
        duplicate.temporal_axis
        or transfer.temporal_axis
        or getattr(plan.reduction_strategy, "axis", "")
    )
    temporal_extent = int(
        dict(plan.metadata.get("axis_extents", {})).get(
            temporal_axis, dict(plan.metadata.get("problem_shape", {})).get("K", 1)
        )
    )
    if _ceil_div(temporal_extent, rows_per_vr) < WRITABLE_VRS:
        return metrics

    # The code generator reuses one VR and therefore reloads the chunk stream
    # for every output batch instead of pinning an impossible bank of VRs.
    replay = max(
        1,
        _ceil_div(int(metrics["resident_reuse_factor"]), accumulator_block),
    )
    for name in (
        "count",
        "source_elements",
        "destination_elements",
        "total_bytes",
        "iterations",
    ):
        metrics[name] *= replay
    metrics["resident_reuse_factor"] = 1
    metrics["streaming_replay_factor"] = replay
    metrics["accumulator_block"] = accumulator_block
    return metrics


def _route_step_sequence(plan, target, transfer: Transfer, step):
    """Map one explicit source/transit/compute relation to priced handles."""

    metrics = _effective_route_metrics(plan, transfer, step)
    kind = step.kind
    if kind == "dma_l4_l3":
        return ((target.move("DMA_L4_TO_L3"), "DMA_L4_TO_L3", metrics),)
    if kind == "dma_l4_l1_32k":
        return ((target.move("DMA_L4_TO_L1_32K"), "DMA_L4_TO_L1_32K", metrics),)
    if kind == "load_vr":
        return ((target.move("LOAD_L1_TO_VR16"), "LOAD_L1_TO_VR16", metrics),)
    if kind == "lookup":
        table_size = _checked_layout_parameter(
            step, "table_size", metrics["source_elements_per_call"]
        )
        vector = {
            **metrics,
            "table_size": table_size,
            "source_is_l4": int(str(step.source.storage).lower() == "l4"),
        }
        return (
            (
                target.op("CREATE_GROUP_INDEX_16"),
                "CREATE_GROUP_INDEX_16",
                vector,
            ),
            (target.op("LOOKUP_16"), "LOOKUP_16", vector),
        )
    if kind == "duplicate_subgroup":
        # One GVML invocation may duplicate one subgroup in several physical
        # groups simultaneously.  The relation's aggregate source and
        # destination images therefore determine traffic and call count;
        # planner parameters are codegen spelling, not cost inputs.
        vector = dict(metrics)
        return (
            (
                target.op("CREATE_SUBGROUP_INDEX_16"),
                "CREATE_SUBGROUP_INDEX_16",
                vector,
            ),
            (
                target.op("DUPLICATE_SUBGROUP_16"),
                "DUPLICATE_SUBGROUP_16",
                vector,
            ),
        )
    if kind == "dma_vr_l4":
        return (
            (target.move("STORE_VR16_TO_L1"), "STORE_VR16_TO_L1", metrics),
            (target.move("DMA_L1_TO_L4_32K"), "DMA_L1_TO_L4_32K", metrics),
        )
    if kind == "direct":
        egress = transfer.direction in {
            "out",
            "store",
            "vr_to_l4",
            "l4_to_host",
        }
        name = "PIO_VR16_TO_L4" if egress else "PIO_L4_TO_VR16"
        # A direct PIO ingress has no expansion primitive: the ARC must write
        # every destination lane.  Egress reads each source lane instead.
        elements = (
            metrics["source_elements_per_call"]
            if egress
            else metrics["destination_elements_per_call"]
        )
        pio = {
            **metrics,
            "elements": elements,
            "bytes": elements * _dtype_bytes(plan),
            "total_bytes": elements * _dtype_bytes(plan) * metrics["count"],
            "iterations": elements * metrics["count"],
        }
        return ((target.move(name), name, pio),)
    raise ValueError(f"APU v1 plan has unknown route kind {kind!r}")


def _transfer_sequence(plan, target, transfer):
    if not transfer.route:
        raise ValueError(
            f"APU v1 transfer {transfer.value!r} needs an explicit layout route"
        )
    sequence = []
    for step in transfer.route:
        sequence.extend(_route_step_sequence(plan, target, transfer, step))
    return tuple(sequence)


def _transfer_route_metadata(plan, transfer):
    fields = (
        "source_elements",
        "destination_elements",
        "source_elements_per_call",
        "destination_elements_per_call",
        "expansion_factor",
        "resident_reuse_factor",
        "streaming_replay_factor",
    )
    return tuple(
        {
            "kind": step.kind,
            "source_storage": step.source.storage,
            "call_count": metrics["count"],
            **{field: metrics[field] for field in fields if field in metrics},
        }
        for step in transfer.route
        for metrics in (_effective_route_metrics(plan, transfer, step),)
    )


def build_apu_v1_plan_graph(
    plan,
    target,
    bound_cost,
    *,
    operation_inventory=None,
) -> ExecutionGraph:
    """Execute one plan into a concrete target-bound dependency graph."""

    if not isinstance(plan, APUV1Plan):
        raise TypeError("plan must be an APUV1Plan")
    if getattr(target, "name", None) != "apu_v1":
        raise ValueError("APU v1 plans require the apu_v1 target")
    cost = _bound_cost(target, bound_cost)
    facts = _shape_facts(plan)
    inventory = _operation_inventory(plan, facts, operation_inventory)
    graph = ExecutionGraph(
        name=f"{plan.name}@apu_v1",
        metadata={
            "target": "apu_v1",
            "plan": plan.name,
            "cost": cost.spec.name,
            "cost_fingerprint": cost.fingerprint,
            "vector_facts": facts,
            "transfer_routes": tuple(
                {
                    "value": transfer.value,
                    "direction": transfer.direction,
                    "steps": _transfer_route_metadata(plan, transfer),
                }
                for transfer in plan.transfers
            ),
            "operation_inventory": tuple(
                operation.canonical_manifest for operation in inventory
            ),
            "analytical": True,
        },
    )
    dependencies = ()
    event_index = 0

    ingress = [
        transfer
        for transfer in plan.transfers
        if transfer.direction not in {"out", "store", "vr_to_l4", "l4_to_host", "inout"}
    ]
    egress = [
        transfer
        for transfer in plan.transfers
        if transfer.direction in {"out", "store", "vr_to_l4", "l4_to_host", "inout"}
    ]
    for transfer in ingress:
        for handle, name, metrics in _transfer_sequence(plan, target, transfer):
            event = CostEvent.create(
                f"transfer:{event_index}:{transfer.value}:{name}",
                handle,
                work_id=(0,),
                metrics=metrics,
                attributes={
                    "value": transfer.value,
                    "phase": "ingress",
                    "cost_count": int(metrics.get("count", 1)),
                },
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))
            event_index += 1

    for operation in inventory:
        handle, target_name = _operation_handle(target, operation)
        count = operation.count
        reduction = plan.reduction_strategy
        parameters = dict(operation.attributes)
        group_size = int(
            parameters.get("group_size")
            or getattr(reduction, "group_size", 0)
            or facts["reduction_tile"]
        )
        metrics = {
            "count": count,
            "iterations": count * VR_LANES,
            "reduction_extent": max(1, facts["reduction_tile"]),
            "group_size": max(1, group_size),
            "candidate": {"group_size": max(1, group_size)},
        }
        if target_name.startswith("GROUP_REDUCE_ADD_"):
            metrics["candidate"]["n_out_tiles"] = count
        event = CostEvent.create(
            f"operation:{event_index}:{target_name}",
            handle,
            work_id=(0,),
            metrics=metrics,
            attributes={
                "operation": operation.opcode,
                "dtype": operation.dtype,
                "materialized_attributes": parameters,
                "cost_count": count,
            },
        )
        dependencies = tuple(cost.emit(graph, event, dependencies))
        event_index += 1

    for transfer in egress:
        for handle, name, metrics in _transfer_sequence(plan, target, transfer):
            event = CostEvent.create(
                f"transfer:{event_index}:{transfer.value}:{name}",
                handle,
                work_id=(0,),
                metrics=metrics,
                attributes={
                    "value": transfer.value,
                    "phase": "egress",
                    "cost_count": int(metrics.get("count", 1)),
                },
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))
            event_index += 1

    if not graph.activities:
        raise ValueError(f"APU v1 plan {plan.name!r} emitted no cost activities")
    return graph


def estimate_apu_v1_plan(
    plan,
    target,
    bound_cost,
    *,
    operation_inventory=None,
) -> APUV1PlanEstimate:
    cost = _bound_cost(target, bound_cost)
    inventory = _operation_inventory(plan, _shape_facts(plan), operation_inventory)
    graph = build_apu_v1_plan_graph(
        plan,
        target,
        cost,
        operation_inventory=inventory,
    )
    return APUV1PlanEstimate(plan, graph, cost.evaluate(graph), inventory)


def materialized_apu_v1_operation_inventory(realization):
    """Freeze the exact repeated compute calls from one faithful realization."""

    plan = getattr(realization, "plan", None)
    if not isinstance(plan, APUV1Plan):
        raise TypeError("APU v1 realization must retain its APUV1Plan")
    invocations = tuple(getattr(realization, "compute_invocation_inventory", ()) or ())
    if not invocations:
        raise ValueError("APU v1 realization has no compute operation inventory")
    if any(
        not all(hasattr(invocation, field) for field in ("opcode", "count", "attrs"))
        for invocation in invocations
    ):
        raise TypeError("APU v1 realization has an invalid compute invocation entry")
    dtype = str(plan.metadata.get("dtype", "f16"))
    return tuple(
        APUV1OperationInventoryEntry(
            invocation.opcode,
            invocation.count,
            dtype,
            invocation.attrs,
        )
        for invocation in invocations
    )


def estimate_apu_v1_realization(realization, target, bound_cost) -> APUV1PlanEstimate:
    """Estimate the exact immutable operation inventory that was materialized."""

    inventory = materialized_apu_v1_operation_inventory(realization)
    return estimate_apu_v1_plan(
        realization.plan,
        target,
        bound_cost,
        operation_inventory=inventory,
    )


def rank_apu_v1_plans(plans, target, bound_cost) -> tuple[APUV1PlanEstimate, ...]:
    """Return stable ascending analytical rank; input order breaks ties."""

    ranked = [
        (index, estimate_apu_v1_plan(plan, target, bound_cost))
        for index, plan in enumerate(plans)
    ]
    ranked.sort(key=lambda item: (item[1].cycles, item[0], item[1].plan.name))
    return tuple(item[1] for item in ranked)


__all__ = [
    "APUV1OperationInventoryEntry",
    "APUV1PlanEstimate",
    "build_apu_v1_plan_graph",
    "estimate_apu_v1_realization",
    "estimate_apu_v1_plan",
    "materialized_apu_v1_operation_inventory",
    "rank_apu_v1_plans",
]
