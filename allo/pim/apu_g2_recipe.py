# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical direct-VL64 recipes for the Gemini-II backend.

This module is deliberately below workload recognition and above C++ source
emission.  A recipe records the exact issued calls, descriptor shapes, and
data dependencies that both a future emitter and a calibrated cost program
must consume.  Its execution graph uses one structural issue cycle per call;
those cycles are explicitly *not* calibrated device ticks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Mapping

from ..perf import Activity, ExecutionGraph, Occupancy
from ..perf.cost import concrete_instance, handle_path
from ..spmw_fingerprint import json_digest


APUG2_GROUPS = 16
APUG2_LANES = 65_536
APUG2_MMB_SETS = 4
APUG2_MMB_SEGMENT_BITS = 24


def _structural_digest(value) -> str:
    return json_digest(value)


class APUG2RecipeOpKind(str, Enum):
    """Scheduling class of one physical recipe call."""

    VL64 = "vl64"
    TRANSFER = "transfer"
    TRANSFORM = "transform"
    BARRIER = "barrier"
    SCALAR_CONTROL = "scalar_control"


@dataclass(frozen=True)
class APUG2Descriptor:
    """One concrete L1 or MMB vector descriptor used by a VL64 call."""

    storage: str
    num_bits: int
    num_vectors: int = APUG2_MMB_SETS
    value_type: str = "uint"
    start_row: int = 0
    segment: str | None = None
    broadcast: bool = False

    def __post_init__(self):
        if self.storage not in {"l1", "mmb", "immediate"}:
            raise ValueError(
                "APUg2 descriptor storage must be 'l1', 'mmb', or 'immediate'"
            )
        if isinstance(self.num_bits, bool) or not isinstance(self.num_bits, Integral):
            raise TypeError("APUg2 descriptor num_bits must be an integer")
        if not 1 <= int(self.num_bits) <= APUG2_MMB_SEGMENT_BITS:
            raise ValueError("APUg2 descriptor num_bits must be in [1, 24]")
        if isinstance(self.num_vectors, bool) or not isinstance(
            self.num_vectors, Integral
        ):
            raise TypeError("APUg2 descriptor num_vectors must be an integer")
        if not 1 <= int(self.num_vectors) <= APUG2_MMB_SETS:
            raise ValueError("APUg2 descriptor num_vectors must be in [1, 4]")
        if self.value_type not in {"uint", "int", "marker"}:
            raise ValueError("unsupported APUg2 descriptor value_type")
        if isinstance(self.start_row, bool) or not isinstance(self.start_row, Integral):
            raise TypeError("APUg2 descriptor start_row must be an integer")
        if int(self.start_row) < 0:
            raise ValueError("APUg2 descriptor start_row must be nonnegative")
        if self.storage in {"l1", "immediate"}:
            if self.segment is not None:
                raise ValueError(
                    f"{self.storage} descriptors do not have an MMB segment"
                )
            if self.storage == "immediate" and int(self.start_row) != 0:
                raise ValueError("immediate descriptors must have start_row=0")
        else:
            if self.segment not in {"seg0", "seg1"}:
                raise ValueError("MMB descriptors require segment 'seg0' or 'seg1'")
            lower = 0 if self.segment == "seg0" else APUG2_MMB_SEGMENT_BITS
            upper = lower + APUG2_MMB_SEGMENT_BITS
            if not lower <= int(self.start_row) < upper:
                raise ValueError(
                    f"MMB {self.segment} descriptor must start in rows "
                    f"[{lower}, {upper})"
                )
            if int(self.start_row) + int(self.num_bits) > upper:
                raise ValueError(
                    f"MMB {self.segment} descriptor crosses its {upper}-row boundary"
                )
        if self.broadcast and self.storage not in {"l1", "immediate"}:
            raise ValueError("only an L1 or immediate descriptor may broadcast")

        object.__setattr__(self, "num_bits", int(self.num_bits))
        object.__setattr__(self, "num_vectors", int(self.num_vectors))
        object.__setattr__(self, "start_row", int(self.start_row))

    def manifest(self) -> dict[str, object]:
        return {
            "storage": self.storage,
            "num_bits": self.num_bits,
            "num_vectors": self.num_vectors,
            "value_type": self.value_type,
            "start_row": self.start_row,
            "segment": self.segment,
            "broadcast": self.broadcast,
        }


@dataclass(frozen=True)
class APUG2DescriptorUse:
    """A named operand/result descriptor on one physical call."""

    role: str
    descriptor: APUG2Descriptor

    def __post_init__(self):
        if not isinstance(self.role, str) or not self.role.isidentifier():
            raise ValueError("APUg2 descriptor roles must be identifiers")
        if not isinstance(self.descriptor, APUG2Descriptor):
            raise TypeError("descriptor use requires an APUG2Descriptor")

    def manifest(self) -> dict[str, object]:
        return {"role": self.role, **self.descriptor.manifest()}


@dataclass(frozen=True)
class APUG2RecipeOperation:
    """One exactly issued transfer, transform, VL64 call, or barrier."""

    id: str
    opcode: str
    kind: APUG2RecipeOpKind | str
    dependencies: tuple[str, ...] = ()
    descriptors: tuple[APUG2DescriptorUse, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=dict)
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("APUg2 recipe operation id must be nonempty")
        opcode = str(self.opcode).upper()
        if not opcode or not opcode.replace("_", "").isalnum():
            raise ValueError("APUg2 recipe opcode must be an uppercase-style name")
        kind = APUG2RecipeOpKind(self.kind)
        dependencies = tuple(self.dependencies)
        descriptors = tuple(self.descriptors)
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"operation {self.id!r} has duplicate dependencies")
        if self.id in dependencies:
            raise ValueError(f"operation {self.id!r} cannot depend on itself")
        roles = [use.role for use in descriptors]
        if len(roles) != len(set(roles)):
            raise ValueError(f"operation {self.id!r} has duplicate descriptor roles")

        metrics = dict(self.metrics)
        metrics.setdefault(
            "vl64_calls", 0 if kind == APUG2RecipeOpKind.SCALAR_CONTROL else 1
        )
        metrics.setdefault("groups", APUG2_GROUPS)
        metrics.setdefault("physical_lanes", APUG2_LANES)
        if metrics["vl64_calls"] not in {0, 1}:
            raise ValueError(
                "one recipe operation must represent at most one VL64 call"
            )
        vector_updates = metrics.get("vector_lane_updates", 0)
        if isinstance(vector_updates, bool) or not isinstance(vector_updates, Integral):
            raise TypeError("vector_lane_updates must be an integer")
        if int(vector_updates) < 0:
            raise ValueError("vector_lane_updates must be nonnegative")
        metrics["vector_lane_updates"] = int(vector_updates)

        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.id,
            "opcode": self.opcode,
            "kind": self.kind.value,
            "dependencies": list(self.dependencies),
            "descriptors": [item.manifest() for item in self.descriptors],
            "metrics": dict(self.metrics),
            "attributes": dict(self.attributes),
        }

    def structural_manifest(self, operation_indices) -> dict[str, object]:
        """Return the physical issue shape, excluding diagnostic identities."""

        return {
            "opcode": self.opcode,
            "kind": self.kind.value,
            "dependencies": [operation_indices[item] for item in self.dependencies],
            "descriptors": [item.manifest() for item in self.descriptors],
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class APUG2VectorizationCertificate:
    """Auditable statement that tensor work has no scalar fallback."""

    vector_lane_updates: int
    scalar_control_ops: int
    scalar_tensor_updates: int
    coalesced_groups: int = APUG2_GROUPS
    direct_vl64: bool = True
    simulator_fallback: bool = False

    def __post_init__(self):
        for name in (
            "vector_lane_updates",
            "scalar_control_ops",
            "scalar_tensor_updates",
            "coalesced_groups",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.coalesced_groups != APUG2_GROUPS:
            raise ValueError("APUg2 recipes must coalesce the 16 physical groups")

    @property
    def fully_vectorized(self) -> bool:
        return (
            self.direct_vl64
            and not self.simulator_fallback
            and self.vector_lane_updates > 0
            and self.scalar_tensor_updates == 0
        )

    def manifest(self) -> dict[str, object]:
        return {
            "vector_lane_updates": self.vector_lane_updates,
            "scalar_control_ops": self.scalar_control_ops,
            "scalar_tensor_updates": self.scalar_tensor_updates,
            "coalesced_groups": self.coalesced_groups,
            "direct_vl64": self.direct_vl64,
            "simulator_fallback": self.simulator_fallback,
            "fully_vectorized": self.fully_vectorized,
        }


@dataclass(frozen=True)
class APUG2Recipe:
    """An immutable, dependency-checked physical program."""

    name: str
    operations: tuple[APUG2RecipeOperation, ...]
    certificate: APUG2VectorizationCertificate
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("APUg2 recipe name must be an identifier")
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("APUg2 recipe requires at least one operation")
        if not all(isinstance(item, APUG2RecipeOperation) for item in operations):
            raise TypeError("APUg2 recipe operations have the wrong type")
        ids = [item.id for item in operations]
        if len(ids) != len(set(ids)):
            raise ValueError("APUg2 recipe operation ids must be unique")
        seen = set()
        for operation in operations:
            missing = set(operation.dependencies) - seen
            if missing:
                raise ValueError(
                    f"operation {operation.id!r} has unknown or forward dependencies "
                    f"{sorted(missing)}"
                )
            seen.add(operation.id)

        expected_vector_updates = sum(
            item.metrics["vector_lane_updates"] for item in operations
        )
        scalar_control_ops = sum(
            item.kind == APUG2RecipeOpKind.SCALAR_CONTROL for item in operations
        )
        if self.certificate.vector_lane_updates != expected_vector_updates:
            raise ValueError(
                "vectorization certificate disagrees with recipe lane updates"
            )
        if self.certificate.scalar_control_ops != scalar_control_ops:
            raise ValueError(
                "vectorization certificate disagrees with scalar-control inventory"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def inventory(self) -> Mapping[str, int]:
        return MappingProxyType(dict(Counter(item.opcode for item in self.operations)))

    def structural_manifest(self) -> dict[str, object]:
        """Return a rename-invariant physical recipe description."""

        operation_indices = {
            operation.id: index for index, operation in enumerate(self.operations)
        }
        return {
            "operations": [
                operation.structural_manifest(operation_indices)
                for operation in self.operations
            ],
            "certificate": self.certificate.manifest(),
        }

    @property
    def structural_fingerprint(self) -> str:
        return _structural_digest(self.structural_manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "inventory": dict(self.inventory),
            "certificate": self.certificate.manifest(),
            "metadata": dict(self.metadata),
            "operations": [item.manifest() for item in self.operations],
        }


@dataclass(frozen=True)
class APUG2RecipeCalibration:
    """A measured total distributed over one exact recipe inventory.

    Attribution may be normalized when isolated instruction measurements are
    unavailable.  The operation map is explicit so a graph never silently
    invents per-call costs or loses the measured whole-pipeline total.
    """

    recipe_name: str
    total_cycles: int
    operation_cycles: Mapping[str, int]
    measured_ticks_per_pipeline: float | None = None
    repetitions: int | None = None
    basis: str = "normalized_full_pipeline_attribution"
    recipe_fingerprint: str | None = None
    operation_cycle_sequence: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.recipe_name, str) or not self.recipe_name:
            raise ValueError("calibration recipe_name must be nonempty")
        if isinstance(self.total_cycles, bool) or not isinstance(
            self.total_cycles, Integral
        ):
            raise TypeError("calibration total_cycles must be an integer")
        total_cycles = int(self.total_cycles)
        if total_cycles <= 0:
            raise ValueError("calibration total_cycles must be positive")
        operation_cycles = dict(self.operation_cycles)
        if not operation_cycles or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in operation_cycles.values()
        ):
            raise ValueError(
                "calibration operation cycles must be nonnegative integers"
            )
        operation_cycles = {
            operation: int(value) for operation, value in operation_cycles.items()
        }
        if sum(operation_cycles.values()) != total_cycles:
            raise ValueError("calibration operation cycles must sum to total_cycles")
        if (
            self.measured_ticks_per_pipeline is not None
            and float(self.measured_ticks_per_pipeline) <= 0
        ):
            raise ValueError("measured ticks per pipeline must be positive")
        if self.repetitions is not None and (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, Integral)
            or int(self.repetitions) <= 0
        ):
            raise ValueError("calibration repetitions must be a positive integer")
        recipe_fingerprint = self.recipe_fingerprint
        if recipe_fingerprint is not None and (
            not isinstance(recipe_fingerprint, str)
            or len(recipe_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in recipe_fingerprint
            )
        ):
            raise ValueError(
                "calibration recipe_fingerprint must be a SHA-256 hex digest"
            )
        operation_cycle_sequence = tuple(self.operation_cycle_sequence)
        if not operation_cycle_sequence:
            operation_cycle_sequence = tuple(operation_cycles.values())
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in operation_cycle_sequence
        ):
            raise ValueError(
                "operation_cycle_sequence must contain nonnegative integers"
            )
        operation_cycle_sequence = tuple(
            int(value) for value in operation_cycle_sequence
        )
        if len(operation_cycle_sequence) != len(operation_cycles):
            raise ValueError(
                "operation cycle sequence length must match operation cycles"
            )
        if sum(operation_cycle_sequence) != total_cycles:
            raise ValueError("operation cycle sequence must sum to total_cycles")
        object.__setattr__(self, "total_cycles", total_cycles)
        object.__setattr__(self, "operation_cycles", MappingProxyType(operation_cycles))
        object.__setattr__(self, "operation_cycle_sequence", operation_cycle_sequence)
        if self.measured_ticks_per_pipeline is not None:
            object.__setattr__(
                self,
                "measured_ticks_per_pipeline",
                float(self.measured_ticks_per_pipeline),
            )
        if self.repetitions is not None:
            object.__setattr__(self, "repetitions", int(self.repetitions))

    @classmethod
    def normalized(
        cls,
        recipe: "APUG2Recipe",
        total_cycles: int,
        *,
        measured_ticks_per_pipeline: float | None = None,
        repetitions: int | None = None,
        opcode_weights: Mapping[str, int] | None = None,
        basis: str = "normalized_full_pipeline_attribution",
    ) -> "APUG2RecipeCalibration":
        """Distribute a measured total using optional opcode weights."""

        if not isinstance(recipe, APUG2Recipe):
            raise TypeError("normalized calibration requires an APUG2Recipe")
        total_cycles = int(total_cycles)
        weights = dict(opcode_weights or {})
        operation_weights = [
            int(weights.get(item.opcode, 1)) for item in recipe.operations
        ]
        if any(value <= 0 for value in operation_weights):
            raise ValueError("calibration opcode weights must be positive")
        weight_total = sum(operation_weights)
        floors = [total_cycles * value // weight_total for value in operation_weights]
        remainder = total_cycles - sum(floors)
        # Assign the integer remainder deterministically in physical issue order.
        cycles = {
            operation.id: floors[index] + int(index < remainder)
            for index, operation in enumerate(recipe.operations)
        }
        return cls(
            recipe.name,
            total_cycles,
            cycles,
            measured_ticks_per_pipeline,
            repetitions,
            basis,
            recipe.structural_fingerprint,
            tuple(cycles.values()),
        )

    def validate_recipe(self, recipe: "APUG2Recipe") -> None:
        if not isinstance(recipe, APUG2Recipe):
            raise TypeError("calibration validation requires an APUG2Recipe")
        if self.recipe_fingerprint is None:
            raise ValueError("calibration requires a structural recipe fingerprint")
        if self.recipe_fingerprint != recipe.structural_fingerprint:
            raise ValueError("calibration physical recipe fingerprint does not match")
        if len(self.operation_cycle_sequence) != len(recipe.operations):
            raise ValueError("calibration operation count does not match the recipe")

    def cycles_for(self, recipe: "APUG2Recipe") -> Mapping[str, int]:
        self.validate_recipe(recipe)
        return MappingProxyType(
            {
                operation.id: self.operation_cycle_sequence[index]
                for index, operation in enumerate(recipe.operations)
            }
        )

    def manifest(self) -> dict[str, object]:
        return {
            "recipe_name": self.recipe_name,
            "total_cycles": self.total_cycles,
            "measured_ticks_per_pipeline": self.measured_ticks_per_pipeline,
            "repetitions": self.repetitions,
            "basis": self.basis,
            "recipe_fingerprint": self.recipe_fingerprint,
            "operation_cycle_sequence": list(self.operation_cycle_sequence),
            "operation_cycles": dict(self.operation_cycles),
        }


_TRANSFER_OPS = {
    "COPY_L1_VECTORS_TO_MMB",
    "COPY_L1_VECTOR_TO_MMB",
    "COPY_MMB_TO_L1_VECTORS",
    "COPY_IMMEDIATE_TO_MMB",
}
_VL64_OPS = {
    "MUL_U8_TO_U16",
    "GROUP_REDUCE_ADD_U16_TO_U23",
    "MUL_TYPED",
    "GROUP_REDUCE_ADD_TYPED",
    "SHIFT_LEFT_U16",
    "SHIFT_RIGHT_U16",
    "ADD_U16",
    "ADD_TYPED",
    "LT_U16",
    "MIN_U16",
    "MAX_U16",
    "DIV_U16",
    "DIV_TYPED",
    "COPY_ODD_TO_EVEN_VECTORS",
    "SUB_U16",
}
_TRANSFORM_OPS = {"SQUEEZE_ROWS_INPLACE", "SPREAD_BLOCK"}


def _target_primitive(target, operation: APUG2RecipeOperation):
    opcode = operation.opcode
    expected_kind = {
        **{name: APUG2RecipeOpKind.TRANSFER for name in _TRANSFER_OPS},
        **{name: APUG2RecipeOpKind.VL64 for name in _VL64_OPS},
        **{name: APUG2RecipeOpKind.TRANSFORM for name in _TRANSFORM_OPS},
        "SEU_BARRIER": APUG2RecipeOpKind.BARRIER,
        "ARC_SCALAR_CONTROL": APUG2RecipeOpKind.SCALAR_CONTROL,
    }.get(opcode)
    if expected_kind is not None and operation.kind != expected_kind:
        raise ValueError(
            f"physical opcode {opcode} requires kind {expected_kind.value!r}, "
            f"got {operation.kind.value!r}"
        )
    if opcode == "ARC_SCALAR_CONTROL":
        return target.op("DISPATCH")
    if opcode in {"COPY_L1_VECTORS_TO_MMB", "COPY_L1_VECTOR_TO_MMB"}:
        destination = next(
            (item.descriptor for item in operation.descriptors if item.role == "dst"),
            None,
        )
        move = (
            "L1_TO_MMB_SEG1"
            if destination is not None and destination.segment == "seg1"
            else "L1_TO_MMB_SEG0"
        )
        return target.move(move)
    if opcode == "COPY_MMB_TO_L1_VECTORS":
        return target.move("MMB_TO_L1")
    if opcode == "COPY_IMMEDIATE_TO_MMB":
        # The current structural target has no scalar-immediate move handle;
        # account it on the same singleton L1/MMB issue path until that target
        # surface is extended.  The recipe opcode remains the exact SDK call.
        return target.move("L1_TO_MMB_SEG0")
    if opcode in _VL64_OPS | _TRANSFORM_OPS | {"SEU_BARRIER"}:
        return target.op(opcode)
    raise KeyError(f"APUg2 target has no physical recipe primitive {opcode!r}")


def build_apu_g2_recipe_graph(
    recipe: APUG2Recipe,
    target,
    *,
    latency: Callable[[APUG2RecipeOperation], int] | None = None,
    calibration: APUG2RecipeCalibration | None = None,
) -> ExecutionGraph:
    """Lower a physical recipe into a dependency/resource graph.

    With no ``latency`` callback, every issued call has one structural issue
    cycle.  The graph metadata marks that unit clearly so it cannot be confused
    with a real-card tick estimate.  A future calibrated cost model can supply
    per-descriptor latencies without changing the recipe or its inventory.
    """

    if not isinstance(recipe, APUG2Recipe):
        raise TypeError("recipe must be an APUG2Recipe")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 recipes require the apu_v2 target")
    if latency is not None and calibration is not None:
        raise ValueError("pass either latency or calibration, not both")
    if calibration is not None:
        if not isinstance(calibration, APUG2RecipeCalibration):
            raise TypeError("calibration must be an APUG2RecipeCalibration")
        calibrated_cycles = calibration.cycles_for(recipe)
        latency = lambda operation: calibrated_cycles[operation.id]
    graph = ExecutionGraph(
        recipe.name,
        metadata={
            "target": "apu_v2",
            "physical_recipe": True,
            "latency_unit": "structural_issue_call" if latency is None else "cycles",
            "calibrated_device_ticks": latency is not None,
            "recipe_calibration": (
                None if calibration is None else calibration.manifest()
            ),
            "inventory": dict(recipe.inventory),
            "vectorization_certificate": recipe.certificate.manifest(),
            **dict(recipe.metadata),
        },
    )
    core = target.unit("core")
    arc = target.unit("arc")
    vector_engine = target.unit("vector_engine")
    for operation in recipe.operations:
        primitive = _target_primitive(target, operation)
        cycles = 1 if latency is None else int(latency(operation))
        if cycles < 0:
            raise ValueError("APUg2 recipe latency must be nonnegative")
        occupied = [primitive, core, arc]
        if operation.kind != APUG2RecipeOpKind.BARRIER:
            occupied.append(vector_engine)
        if operation.kind in {
            APUG2RecipeOpKind.VL64,
            APUG2RecipeOpKind.TRANSFER,
            APUG2RecipeOpKind.TRANSFORM,
        }:
            occupied.append(target.mmb)
        if (
            operation.kind
            in {
                APUG2RecipeOpKind.TRANSFER,
                APUG2RecipeOpKind.TRANSFORM,
            }
            and operation.opcode != "COPY_IMMEDIATE_TO_MMB"
        ):
            occupied.append(target.l1)
        graph.add(
            Activity(
                id=operation.id,
                primitive=handle_path(primitive),
                latency_cycles=cycles,
                occupancy=tuple(
                    Occupancy(concrete_instance(handle, ()), cycles)
                    for handle in occupied
                ),
                depends_on=operation.dependencies,
                label=operation.opcode.lower(),
                metadata={
                    "kind": operation.kind.value,
                    "metrics": dict(operation.metrics),
                    "descriptors": [
                        descriptor.manifest() for descriptor in operation.descriptors
                    ],
                    **dict(operation.attributes),
                },
            )
        )
    return graph



def _typed_call_feature_arguments(operation: APUG2RecipeOperation) -> dict:
    """Widths and depth that :func:`apu_g2_typed_call_features` prices."""

    descriptors = {item.role: item.descriptor for item in operation.descriptors}
    opcode = operation.opcode
    if opcode == "SEU_BARRIER":
        return {}
    if opcode == "MUL_TYPED":
        return {
            "lhs_bits": descriptors["lhs"].num_bits,
            "rhs_bits": descriptors["rhs"].num_bits,
        }
    if opcode == "GROUP_REDUCE_ADD_TYPED":
        attributes = dict(operation.attributes)
        if "log_reduction" in attributes:
            log_reduction = int(attributes["log_reduction"])
        else:
            extent = int(attributes["reduction_extent"])
            if extent < 2 or extent & (extent - 1):
                raise ValueError("typed reduction extent must be a power of two")
            log_reduction = extent.bit_length() - 1
        return {"log_reduction": log_reduction}
    return {"bits": descriptors["dst"].num_bits}


def emit_apu_g2_priced_recipe(
    graph: ExecutionGraph,
    recipe: APUG2Recipe,
    target,
    cost,
    *,
    prefix: str = "",
    dependencies: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Emit every issued typed/composed call through the bound cost program.

    Each recipe call becomes one :class:`~allo.perf.CostEvent` on its target
    primitive, carrying the structural features of
    :func:`~allo.pim.apu_g2_pipeline.apu_g2_typed_call_features`; the cost
    rules turn them into device ticks.  Calls run in recipe order after
    ``dependencies``.  Returns the terminal activity ids and a summary with
    the per-feature totals of the recipe.
    """

    from ..perf import CostEvent
    from .apu_g2_pipeline import APUG2_TYPED_TICK_FEATURES, apu_g2_typed_call_features

    if not isinstance(recipe, APUG2Recipe):
        raise TypeError("recipe must be an APUG2Recipe")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 recipes require the apu_v2 target")
    tick_model = apu_g2_cost_tick_model(cost)
    totals = dict.fromkeys(APUG2_TYPED_TICK_FEATURES, 0)
    validity = "calibrated"
    terminals = tuple(dependencies)
    for operation in recipe.operations:
        features = apu_g2_typed_call_features(
            operation.opcode, **_typed_call_feature_arguments(operation)
        )
        for name, value in features.items():
            totals[name] += value
        if tick_model.validity(features) != "calibrated":
            validity = "extrapolated"
        event = CostEvent.create(
            f"{prefix}{operation.id}",
            _target_primitive(target, operation),
            metrics={
                "typed_call_features": features,
                "opcode": operation.opcode,
            },
            attributes={
                "opcode": operation.opcode,
                "kind": operation.kind.value,
                "descriptors": [
                    descriptor.manifest() for descriptor in operation.descriptors
                ],
                "features": dict(features),
            },
        )
        terminals = cost.emit(graph, event, terminals)
    return terminals, {
        "features": totals,
        "calls": len(recipe.operations),
        "validity": validity,
    }


def apu_g2_cost_tick_model(cost):
    """The typed tick model carried by a bound ``build_apu_g2_cost`` program."""

    model = getattr(getattr(cost, "spec", None), "apu_g2_tick_model", None)
    if model is None:
        raise TypeError(
            "typed/composed APUg2 recipes need a cost program built by "
            "allo.pim.costs.apu_g2.build_apu_g2_cost (e.g. apu_g2_cost)"
        )
    return model


def build_apu_g2_priced_recipe_graph(recipe: APUG2Recipe, target, cost):
    """One recipe priced by ``cost`` (a target-bound ``apu_g2_cost``)."""

    graph = ExecutionGraph(
        recipe.name,
        metadata={
            "target": "apu_v2",
            "physical_recipe": True,
            "latency_unit": "device_ticks_per_pipeline_body",
            "calibrated_device_ticks": True,
            "inventory": dict(recipe.inventory),
            "vectorization_certificate": recipe.certificate.manifest(),
            **dict(recipe.metadata),
        },
    )
    _terminals, summary = emit_apu_g2_priced_recipe(graph, recipe, target, cost)
    tick_model = apu_g2_cost_tick_model(cost)
    graph.metadata.update(
        {
            "tick_features": summary["features"],
            "estimate_validity": summary["validity"],
            "cost_model": cost.spec.name,
            "tick_model_fingerprint": tick_model.fingerprint,
        }
    )
    return graph


def chain_apu_g2_recipes(
    recipes,
    *,
    name: str = "apu_g2_u16_contraction_chain",
    stage_names=None,
) -> APUG2Recipe:
    """Compose physical recipes with explicit stage-to-stage dependencies."""

    recipes = tuple(recipes)
    if not recipes or not all(isinstance(item, APUG2Recipe) for item in recipes):
        raise ValueError("recipe chains require one or more APUG2Recipe values")
    if stage_names is None:
        stage_names = tuple(f"stage{index}" for index in range(len(recipes)))
    else:
        stage_names = tuple(stage_names)
    if len(stage_names) != len(recipes):
        raise ValueError("stage_names must match the recipe count")
    if len(stage_names) != len(set(stage_names)) or any(
        not isinstance(item, str) or not item.isidentifier() for item in stage_names
    ):
        raise ValueError("recipe stage names must be unique identifiers")

    chained = []
    previous_terminals: tuple[str, ...] = ()
    for stage_index, (stage_name, recipe) in enumerate(zip(stage_names, recipes)):
        id_map = {
            operation.id: f"{stage_name}__{operation.id}"
            for operation in recipe.operations
        }
        depended_on = {
            dependency
            for operation in recipe.operations
            for dependency in operation.dependencies
        }
        terminals = tuple(
            id_map[operation.id]
            for operation in recipe.operations
            if operation.id not in depended_on
        )
        for operation in recipe.operations:
            dependencies = tuple(id_map[item] for item in operation.dependencies)
            if not dependencies:
                dependencies = previous_terminals
            chained.append(
                APUG2RecipeOperation(
                    id_map[operation.id],
                    operation.opcode,
                    operation.kind,
                    dependencies,
                    operation.descriptors,
                    operation.metrics,
                    {
                        **dict(operation.attributes),
                        "stage": stage_name,
                        "stage_index": stage_index,
                    },
                )
            )
        previous_terminals = terminals

    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=sum(
            recipe.certificate.vector_lane_updates for recipe in recipes
        ),
        scalar_control_ops=sum(
            recipe.certificate.scalar_control_ops for recipe in recipes
        ),
        scalar_tensor_updates=sum(
            recipe.certificate.scalar_tensor_updates for recipe in recipes
        ),
    )
    return APUG2Recipe(
        name,
        tuple(chained),
        certificate,
        metadata={
            "operation": "contraction_chain",
            "stage_count": len(recipes),
            "stages": tuple(
                {
                    "name": stage_name,
                    "recipe": recipe.name,
                    "output_shape": recipe.metadata.get("output_shape"),
                }
                for stage_name, recipe in zip(stage_names, recipes)
            ),
        },
    )


__all__ = [
    "APUG2Descriptor",
    "APUG2DescriptorUse",
    "APUG2Recipe",
    "APUG2RecipeCalibration",
    "APUG2RecipeOpKind",
    "APUG2RecipeOperation",
    "APUG2VectorizationCertificate",
    "apu_g2_cost_tick_model",
    "build_apu_g2_priced_recipe_graph",
    "build_apu_g2_recipe_graph",
    "chain_apu_g2_recipes",
    "emit_apu_g2_priced_recipe",
]
