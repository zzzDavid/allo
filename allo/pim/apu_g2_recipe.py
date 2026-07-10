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
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Mapping

from ..perf import Activity, ExecutionGraph, Occupancy
from ..perf.cost import concrete_instance, handle_path


APUG2_GROUPS = 16
APUG2_LANES = 65_536
APUG2_MMB_SETS = 4
APUG2_MMB_SEGMENT_BITS = 24
APUG2_MAX_REDUCTION_BLOCK = 256


def _structural_digest(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


@dataclass(frozen=True)
class APUG2RecipeMeasuredPoint:
    """One real-card repeated-throughput point for a physical tile shape."""

    log_block_size: int
    epilogue: bool
    active_outputs: int
    analytical_cycles: int
    measured_ticks_per_pipeline: float
    repetitions: int
    final_pipeline_ticks: int

    def __post_init__(self):
        for name in (
            "log_block_size",
            "active_outputs",
            "analytical_cycles",
            "repetitions",
            "final_pipeline_ticks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"measured point {name} must be an integer")
            if name == "log_block_size":
                if int(value) < 0:
                    raise ValueError(
                        "measured point log_block_size must be nonnegative"
                    )
            elif int(value) <= 0:
                raise ValueError(f"measured point {name} must be positive")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "epilogue", bool(self.epilogue))
        if float(self.measured_ticks_per_pipeline) <= 0:
            raise ValueError("measured ticks per pipeline must be positive")
        object.__setattr__(
            self,
            "measured_ticks_per_pipeline",
            float(self.measured_ticks_per_pipeline),
        )


# Real-card direct-VL64 measurements.  Keys are
# (log_block_size, epilogue_enabled, active_outputs).  Coefficient values do
# not affect the issued instruction sequence; all fused points use the same
# alpha*dot+beta*accumulator recipe inventory.
APUG2_DOT_TILE_MEASURED_POINTS = MappingProxyType(
    {
        (0, True, 120): APUG2RecipeMeasuredPoint(
            0, True, 120, 4_613, 4_612.875, 8, 3_958
        ),
        (1, True, 14400): APUG2RecipeMeasuredPoint(
            1, True, 14400, 4_925, 4_925.0, 8, 4_451
        ),
        (6, False, 2000): APUG2RecipeMeasuredPoint(
            6, False, 2000, 5_177, 5_177.125, 8, 4_802
        ),
        (7, False, 240): APUG2RecipeMeasuredPoint(
            7, False, 240, 7_474, 7_473.5, 8, 7_106
        ),
        (7, False, 1000): APUG2RecipeMeasuredPoint(
            7, False, 1000, 7_423, 7_423.125, 8, 7_165
        ),
        (8, False, 1000): APUG2RecipeMeasuredPoint(
            8, False, 1000, 12_216, 12_216.0, 8, 11_873
        ),
        (5, True, 6808): APUG2RecipeMeasuredPoint(
            5, True, 6808, 6_351, 6_351.25, 8, 5_729
        ),
        (5, True, 8192): APUG2RecipeMeasuredPoint(
            5, True, 8192, 6_369, 6_369.75, 8, 5_726
        ),
        (6, True, 2000): APUG2RecipeMeasuredPoint(
            6, True, 2000, 7_492, 7_491.875, 8, 7_013
        ),
        (6, True, 704): APUG2RecipeMeasuredPoint(
            6, True, 704, 7_607, 7_607.375, 8, 7_063
        ),
        (6, True, 2800): APUG2RecipeMeasuredPoint(
            6, True, 2800, 7_558, 7_558.375, 8, 7_079
        ),
        (6, True, 3200): APUG2RecipeMeasuredPoint(
            6, True, 3200, 7_507, 7_506.625, 8, 7_007
        ),
        (6, True, 4096): APUG2RecipeMeasuredPoint(
            6, True, 4096, 7_558, 7_558.25, 8, 6_968
        ),
        (6, True, 2304): APUG2RecipeMeasuredPoint(
            6, True, 2304, 7_559, 7_559.375, 8, 6_962
        ),
        (7, True, 104): APUG2RecipeMeasuredPoint(
            7, True, 104, 9_911, 9_911.25, 8, 9_325
        ),
        (7, True, 120): APUG2RecipeMeasuredPoint(
            7, True, 120, 9_825, 9_824.75, 8, 9_290
        ),
        (7, True, 1000): APUG2RecipeMeasuredPoint(
            7, True, 1000, 9_835, 9_835.875, 8, 9_321
        ),
        (7, True, 1452): APUG2RecipeMeasuredPoint(
            7, True, 1452, 9_846, 9_846.5, 8, 9_395
        ),
        (7, True, 2000): APUG2RecipeMeasuredPoint(
            7, True, 2000, 9_882, 9_881.875, 8, 9_242
        ),
        (7, True, 2048): APUG2RecipeMeasuredPoint(
            7, True, 2048, 9_806, 9_805.875, 8, 9_246
        ),
    }
)


def _dot_tile_active_outputs(recipe: APUG2Recipe) -> tuple[int, ...]:
    try:
        output_extent = int(recipe.metadata["output_extent"])
        output_tile_extent = int(recipe.metadata["output_tile_extent"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("recipe has no dot-tile calibration metadata") from error
    if output_extent <= 0 or output_tile_extent <= 0:
        raise ValueError("recipe output extents must be positive")
    return tuple(
        min(output_tile_extent, output_extent - offset)
        for offset in range(0, output_extent, output_tile_extent)
    )


def _interpolate_dot_tile_point(
    log_block_size: int, epilogue: bool, active_outputs: int
) -> APUG2RecipeMeasuredPoint:
    exact = APUG2_DOT_TILE_MEASURED_POINTS.get(
        (log_block_size, epilogue, active_outputs)
    )
    if exact is not None:
        return exact

    candidates = sorted(
        (
            point
            for (
                log,
                has_epilogue,
                _active,
            ), point in APUG2_DOT_TILE_MEASURED_POINTS.items()
            if log == log_block_size and has_epilogue is epilogue
        ),
        key=lambda point: point.active_outputs,
    )
    if not candidates:
        candidates = [
            point
            for (
                _log,
                has_epilogue,
                _active,
            ), point in APUG2_DOT_TILE_MEASURED_POINTS.items()
            if has_epilogue is epilogue
        ]
    if not candidates:
        raise ValueError(f"no measured APUg2 dot-tile point for epilogue={epilogue}")
    same_log = [point for point in candidates if point.log_block_size == log_block_size]
    if not same_log:
        nearest = min(
            candidates,
            key=lambda point: (
                abs(point.log_block_size - log_block_size),
                abs(point.active_outputs - active_outputs),
            ),
        )
        scale = (1 << log_block_size) / (1 << nearest.log_block_size)
        return APUG2RecipeMeasuredPoint(
            log_block_size,
            epilogue,
            active_outputs,
            max(1, round(nearest.analytical_cycles * scale)),
            nearest.measured_ticks_per_pipeline * scale,
            nearest.repetitions,
            max(1, round(nearest.final_pipeline_ticks * scale)),
        )

    lower = [point for point in same_log if point.active_outputs <= active_outputs]
    upper = [point for point in same_log if point.active_outputs >= active_outputs]
    if lower and upper:
        low = lower[-1]
        high = upper[0]
        if low.active_outputs == high.active_outputs:
            return low
        fraction = (active_outputs - low.active_outputs) / (
            high.active_outputs - low.active_outputs
        )

        def lerp(attribute):
            return getattr(low, attribute) + fraction * (
                getattr(high, attribute) - getattr(low, attribute)
            )

        return APUG2RecipeMeasuredPoint(
            log_block_size,
            epilogue,
            active_outputs,
            round(lerp("analytical_cycles")),
            lerp("measured_ticks_per_pipeline"),
            min(low.repetitions, high.repetitions),
            round(lerp("final_pipeline_ticks")),
        )

    nearest = min(
        same_log, key=lambda point: abs(point.active_outputs - active_outputs)
    )
    return APUG2RecipeMeasuredPoint(
        log_block_size,
        epilogue,
        active_outputs,
        nearest.analytical_cycles,
        nearest.measured_ticks_per_pipeline,
        nearest.repetitions,
        nearest.final_pipeline_ticks,
    )


def calibrate_apu_g2_recipe_from_measured_tile(
    recipe: APUG2Recipe,
    *,
    opcode_weights: Mapping[str, int] | None = None,
) -> APUG2RecipeCalibration:
    """Normalize a tiled contraction recipe to its real-card tile point."""

    if not isinstance(recipe, APUG2Recipe):
        raise TypeError("measured tile calibration requires an APUG2Recipe")
    try:
        log_block_size = int(recipe.metadata["log_block_size"])
        tile_count = int(recipe.metadata["output_tile_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("recipe has no dot-tile calibration metadata") from error
    has_epilogue = recipe.metadata.get("epilogue") is not None
    active_outputs = _dot_tile_active_outputs(recipe)
    if tile_count <= 0:
        raise ValueError("recipe output_tile_count must be positive")
    if tile_count != len(active_outputs):
        raise ValueError("recipe tile count does not match output extents")
    exact = True
    points = []
    for active in active_outputs:
        point = _interpolate_dot_tile_point(log_block_size, has_epilogue, active)
        points.append(point)
        exact = (
            exact
            and point.active_outputs == active
            and (
                log_block_size,
                has_epilogue,
                active,
            )
            in APUG2_DOT_TILE_MEASURED_POINTS
        )
    return APUG2RecipeCalibration.normalized(
        recipe,
        sum(point.analytical_cycles for point in points),
        measured_ticks_per_pipeline=sum(
            point.measured_ticks_per_pipeline for point in points
        ),
        repetitions=min(point.repetitions for point in points),
        opcode_weights=opcode_weights,
        basis=(
            "real_card_output_sensitive_tile_normalized_attribution"
            if exact
            else "real_card_output_sensitive_interpolated_tile_normalized_attribution"
        ),
    )


_TRANSFER_OPS = {
    "COPY_L1_VECTORS_TO_MMB",
    "COPY_L1_VECTOR_TO_MMB",
    "COPY_MMB_TO_L1_VECTORS",
    "COPY_IMMEDIATE_TO_MMB",
}
_VL64_OPS = {
    "MUL_U8_TO_U16",
    "GROUP_REDUCE_ADD_U16_TO_U23",
    "SHIFT_LEFT_U16",
    "SHIFT_RIGHT_U16",
    "ADD_U16",
    "LT_U16",
    "MIN_U16",
    "MAX_U16",
    "DIV_U16",
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


class _RecipeBuilder:
    def __init__(self):
        self.operations: list[APUG2RecipeOperation] = []
        self.last_id: str | None = None

    def add(
        self,
        opcode,
        kind,
        *,
        descriptors=(),
        metrics=None,
        attributes=None,
    ):
        identifier = f"call_{len(self.operations):04d}_{str(opcode).lower()}"
        operation = APUG2RecipeOperation(
            identifier,
            opcode,
            kind,
            dependencies=() if self.last_id is None else (self.last_id,),
            descriptors=tuple(descriptors),
            metrics=metrics or {},
            attributes=attributes or {},
        )
        self.operations.append(operation)
        self.last_id = identifier
        return operation


def _positive_extent(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _uint16_scalar(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must fit uint16")
    return value


def build_apu_g2_u16_dot_tile_recipe(
    output_extent: int,
    reduction_extent: int,
    *,
    output_tile_extent: int | None = None,
    alpha: int | None = None,
    beta: int | None = None,
    name: str = "apu_g2_u16_dot_tiles",
) -> APUG2Recipe:
    """Build the proven three-byte-product recipe for tiled uint16 dots.

    Independent outputs are striped over all four MMB sets.  Output points
    beyond one four-set carrier are temporal tiles; the reduction itself must
    fit the proven 24-bit widened-sum envelope.  Every output tile executes
    three byte products, three widened sums, two shifts, and two combine adds.
    The products use the hardware-proven MMB-segment-1 -> L1 scratch ->
    MMB-segment-0 route before ``sum``.
    """

    output_extent = _positive_extent("output_extent", output_extent)
    reduction_extent = _positive_extent("reduction_extent", reduction_extent)
    if (alpha is None) != (beta is None):
        raise ValueError("alpha and beta must be supplied together")
    has_epilogue = alpha is not None
    if has_epilogue:
        alpha = _uint16_scalar("alpha", alpha)
        beta = _uint16_scalar("beta", beta)
    padded_reduction = 1 << (reduction_extent - 1).bit_length()
    if padded_reduction > APUG2_MAX_REDUCTION_BLOCK:
        raise ValueError("padded reduction extent must not exceed 256")

    maximum_outputs = APUG2_MMB_SETS * APUG2_LANES // padded_reduction
    if output_tile_extent is None:
        output_tile_extent = min(output_extent, maximum_outputs)
    output_tile_extent = _positive_extent("output_tile_extent", output_tile_extent)
    if 1 << (output_tile_extent - 1).bit_length() > maximum_outputs:
        raise ValueError(
            "padded output tile and reduction block exceed the 65,536-lane carrier"
        )

    output_tiles = math.ceil(output_extent / output_tile_extent)
    log_block_size = padded_reduction.bit_length() - 1
    builder = _RecipeBuilder()

    l1_left_low = APUG2Descriptor("l1", 8, num_vectors=4, start_row=0)
    l1_left_high = APUG2Descriptor("l1", 8, num_vectors=4, start_row=8)
    l1_right_low = APUG2Descriptor("l1", 8, num_vectors=4, start_row=32)
    l1_right_high = APUG2Descriptor("l1", 8, num_vectors=4, start_row=40)
    l1_product_scratch = APUG2Descriptor(
        "l1", 16, num_vectors=APUG2_MMB_SETS, start_row=576
    )
    l1_output = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=768)
    l1_accumulator = APUG2Descriptor(
        "l1", 16, num_vectors=APUG2_MMB_SETS, start_row=800
    )
    l1_scaled_dot = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=832)
    l1_scaled_accumulator = APUG2Descriptor(
        "l1", 16, num_vectors=APUG2_MMB_SETS, start_row=864
    )
    operand0 = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    operand1 = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=8, segment="seg0"
    )
    product = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )
    sum_source = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    sum_destination = APUG2Descriptor(
        "mmb",
        16 + log_block_size,
        num_vectors=APUG2_MMB_SETS,
        start_row=24,
        segment="seg1",
    )
    sum_low16 = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )
    immediate_byte = APUG2Descriptor(
        "immediate", 8, num_vectors=APUG2_MMB_SETS, broadcast=True
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    def append_scalar_multiply(
        value: APUG2Descriptor,
        scaled: APUG2Descriptor,
        scalar: int,
        common: Mapping[str, object],
        label: str,
    ) -> None:
        value_low = APUG2Descriptor("l1", 8, num_vectors=4, start_row=value.start_row)
        value_high = APUG2Descriptor(
            "l1", 8, num_vectors=4, start_row=value.start_row + 8
        )
        scalar_low = scalar & 0xFF
        scalar_high = scalar >> 8
        for term, (value_byte, scalar_byte) in enumerate(
            (
                (value_low, scalar_low),
                (value_low, scalar_high),
                (value_high, scalar_low),
            )
        ):
            attributes = {
                **common,
                "phase": "epilogue",
                "epilogue_value": label,
                "scalar": scalar,
                "scalar_byte": scalar_byte,
                "term": term,
            }
            builder.add(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=value_byte, dst=operand0),
                attributes=attributes,
            )
            builder.add(
                "COPY_IMMEDIATE_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=immediate_byte, dst=operand1),
                metrics={**common, "immediate": scalar_byte},
                attributes=attributes,
            )
            builder.add(
                "MUL_U8_TO_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(lhs=operand0, rhs=operand1, dst=product),
                metrics={
                    **common,
                    "src_bits": 8,
                    "dst_bits": 16,
                    "vector_lane_updates": common["active_outputs"],
                },
                attributes=attributes,
            )
            if term:
                builder.add(
                    "SHIFT_LEFT_U16",
                    APUG2RecipeOpKind.VL64,
                    descriptors=uses(src_dst=product),
                    metrics={
                        **common,
                        "shift": 8,
                        "vector_lane_updates": common["active_outputs"],
                    },
                    attributes=attributes,
                )
                builder.add(
                    "ADD_U16",
                    APUG2RecipeOpKind.VL64,
                    descriptors=uses(lhs=scaled, rhs=product, dst=product),
                    metrics={
                        **common,
                        "vector_lane_updates": common["active_outputs"],
                    },
                    attributes={**attributes, "purpose": "combine_scalar_terms"},
                )
            builder.add(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=product, dst=scaled),
                attributes={**attributes, "purpose": "retain_scaled_value"},
            )

    for output_tile in range(output_tiles):
        output_offset = output_tile * output_tile_extent
        active_outputs = min(output_tile_extent, output_extent - output_offset)
        common = {
            "output_tile": output_tile,
            "output_offset": output_offset,
            "active_outputs": active_outputs,
            "active_reduction": reduction_extent,
            "padded_reduction": padded_reduction,
            "log_block_size": log_block_size,
            "physical_sets": APUG2_MMB_SETS,
        }
        product_lanes = active_outputs * reduction_extent
        output_lanes = active_outputs

        for term, byte_pair in enumerate(("lo_lo", "lo_hi", "hi_lo")):
            term_attributes = {**common, "byte_product": byte_pair, "term": term}
            left_byte = l1_left_high if byte_pair == "hi_lo" else l1_left_low
            right_byte = l1_right_high if byte_pair == "lo_hi" else l1_right_low
            builder.add(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=left_byte, dst=operand0),
                attributes={**term_attributes, "operand": "left"},
            )
            builder.add(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=right_byte, dst=operand1),
                attributes={**term_attributes, "operand": "right"},
            )
            builder.add(
                "MUL_U8_TO_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(lhs=operand0, rhs=operand1, dst=product),
                metrics={
                    **common,
                    "src_bits": 8,
                    "dst_bits": 16,
                    "vector_lane_updates": product_lanes,
                },
                attributes=term_attributes,
            )
            builder.add(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=product, dst=l1_product_scratch),
                attributes={**term_attributes, "segment_safe_bounce": "out"},
            )
            builder.add(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=l1_product_scratch, dst=sum_source),
                attributes={**term_attributes, "segment_safe_bounce": "in"},
            )
            builder.add(
                "GROUP_REDUCE_ADD_U16_TO_U23",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(src=sum_source, dst=sum_destination),
                metrics={
                    **common,
                    "src_bits": 16,
                    "dst_bits": 16 + log_block_size,
                    "vector_lane_updates": output_lanes,
                },
                attributes=term_attributes,
            )
            if term:
                builder.add(
                    "SHIFT_LEFT_U16",
                    APUG2RecipeOpKind.VL64,
                    descriptors=uses(src_dst=sum_low16),
                    metrics={
                        **common,
                        "shift": 8,
                        "vector_lane_updates": output_lanes,
                    },
                    attributes=term_attributes,
                )
                builder.add(
                    "ADD_U16",
                    APUG2RecipeOpKind.VL64,
                    descriptors=uses(lhs=l1_output, rhs=sum_low16, dst=sum_low16),
                    metrics={
                        **common,
                        "vector_lane_updates": output_lanes,
                    },
                    attributes={**term_attributes, "purpose": "combine_terms"},
                )
            builder.add(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=sum_low16, dst=l1_output),
                attributes={**term_attributes, "purpose": "retain_partial"},
            )

        if has_epilogue:
            append_scalar_multiply(l1_output, l1_scaled_dot, alpha, common, "dot")
            append_scalar_multiply(
                l1_accumulator,
                l1_scaled_accumulator,
                beta,
                common,
                "accumulator",
            )
            builder.add(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=l1_scaled_accumulator, dst=sum_source),
                attributes={**common, "phase": "epilogue", "purpose": "combine"},
            )
            builder.add(
                "ADD_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(lhs=l1_scaled_dot, rhs=sum_source, dst=product),
                metrics={
                    **common,
                    "vector_lane_updates": active_outputs,
                },
                attributes={**common, "phase": "epilogue", "purpose": "combine"},
            )
            builder.add(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                descriptors=uses(src=product, dst=l1_output),
                attributes={
                    **common,
                    "phase": "epilogue",
                    "purpose": "retain_epilogue_output",
                },
            )

    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "arithmetic": "modulo_2^16",
            "output_extent": output_extent,
            "reduction_extent": reduction_extent,
            "output_tile_extent": output_tile_extent,
            "output_tile_count": output_tiles,
            "dot_tile_count": output_tiles,
            "tile_capacity": maximum_outputs,
            "log_block_size": log_block_size,
            "physical_sets": APUG2_MMB_SETS,
            "segment_safe_product_bounce": True,
            "epilogue": "alpha_dot_plus_beta_accumulator" if has_epilogue else None,
            "alpha": alpha,
            "beta": beta,
        },
    )


def build_apu_g2_u16_select_lt_recipe(
    *,
    name: str = "apu_g2_u16_select_lt",
) -> APUG2Recipe:
    """Build a full-vector ``lhs < rhs ? true_value : false_value`` recipe."""

    builder = _RecipeBuilder()
    l1_lhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_rhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    l1_true = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=32)
    l1_false = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=48)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=64)
    mmb_lhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_rhs_out = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )
    mmb_mask = APUG2Descriptor(
        "mmb",
        1,
        num_vectors=APUG2_MMB_SETS,
        value_type="marker",
        start_row=23,
        segment="seg0",
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_lhs, dst=mmb_lhs),
        attributes={"operand": "lhs"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_rhs, dst=mmb_rhs_out),
        attributes={"operand": "rhs"},
    )
    builder.add(
        "LT_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(lhs=mmb_lhs, rhs=mmb_rhs_out, dst=mmb_mask),
        metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
        attributes={"predicate": "lhs_lt_rhs"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_false, dst=mmb_rhs_out),
        attributes={"operand": "false_value", "purpose": "initialize_output"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_true, dst=mmb_rhs_out, mask=mmb_mask),
        attributes={"operand": "true_value", "masked": True, "purpose": "select"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_rhs_out, dst=l1_out),
        attributes={"purpose": "store_selected_output"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "select_lt",
            "arithmetic": "unsigned_compare",
            "output_extent": APUG2_MMB_SETS * APUG2_LANES,
            "predicate": "lhs < rhs",
            "physical_sets": APUG2_MMB_SETS,
            "masked_copy": True,
        },
    )


def build_apu_g2_u16_minmax_recipe(
    *,
    name: str = "apu_g2_u16_minmax",
) -> APUG2Recipe:
    """Build a full-vector unsigned min/max recipe for two uint16 packs."""

    builder = _RecipeBuilder()
    l1_lhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_rhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    l1_min = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=32)
    l1_max = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=48)
    mmb_lhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_rhs_out = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_lhs, dst=mmb_lhs),
        attributes={"operand": "lhs"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_rhs, dst=mmb_rhs_out),
        attributes={"operand": "rhs", "purpose": "initialize_min"},
    )
    builder.add(
        "MIN_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(src_dst=mmb_rhs_out, src=mmb_lhs),
        metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
        attributes={"operation": "min(lhs,rhs)", "inplace_segment": "seg1"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_rhs_out, dst=l1_min),
        attributes={"purpose": "store_min"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_rhs, dst=mmb_rhs_out),
        attributes={"operand": "rhs", "purpose": "initialize_max"},
    )
    builder.add(
        "MAX_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(src_dst=mmb_rhs_out, src=mmb_lhs),
        metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
        attributes={"operation": "max(lhs,rhs)", "inplace_segment": "seg1"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_rhs_out, dst=l1_max),
        attributes={"purpose": "store_max"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "minmax",
            "arithmetic": "unsigned_minmax",
            "output_extent": APUG2_MMB_SETS * APUG2_LANES,
            "physical_sets": APUG2_MMB_SETS,
            "outputs": ("min", "max"),
        },
    )


def build_apu_g2_u16_div_recipe(
    *,
    name: str = "apu_g2_u16_div",
) -> APUG2Recipe:
    """Build a full-vector unsigned floor-division recipe for uint16 packs."""

    builder = _RecipeBuilder()
    l1_lhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_rhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=32)
    mmb_lhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_rhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_lhs, dst=mmb_lhs),
        attributes={"operand": "dividend"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_rhs, dst=mmb_rhs),
        attributes={"operand": "divisor"},
    )
    builder.add(
        "DIV_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(dividend=mmb_lhs, divisor=mmb_rhs, dst=l1_out),
        metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
        attributes={"operation": "lhs // rhs", "division_by_zero": "rejected"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "div",
            "arithmetic": "unsigned_floor_division",
            "output_extent": APUG2_MMB_SETS * APUG2_LANES,
            "physical_sets": APUG2_MMB_SETS,
            "division_by_zero": "rejected",
        },
    )


def build_apu_g2_u16_mul_recipe(
    *,
    name: str = "apu_g2_u16_mul",
) -> APUG2Recipe:
    """Build a full-vector modular uint16 multiplication recipe.

    Gemini-II's direct VL64 multiply produces an 8x8->16-bit byte product.
    Full uint16 multiplication modulo 2^16 is therefore:

    ``lo*lo + ((lo*hi + hi*lo) << 8)``.

    The high*high term is a multiple of 2^16 and is intentionally omitted.
    """

    builder = _RecipeBuilder()
    l1_lhs_low = APUG2Descriptor("l1", 8, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_lhs_high = APUG2Descriptor("l1", 8, num_vectors=APUG2_MMB_SETS, start_row=8)
    l1_rhs_low = APUG2Descriptor("l1", 8, num_vectors=APUG2_MMB_SETS, start_row=16)
    l1_rhs_high = APUG2Descriptor("l1", 8, num_vectors=APUG2_MMB_SETS, start_row=24)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=32)
    mmb_lhs = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_rhs = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=8, segment="seg0"
    )
    mmb_product = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    full_extent = APUG2_MMB_SETS * APUG2_LANES
    for term, (lhs_byte, rhs_byte) in enumerate(
        (
            (l1_lhs_low, l1_rhs_low),
            (l1_lhs_low, l1_rhs_high),
            (l1_lhs_high, l1_rhs_low),
        )
    ):
        attributes = {
            "term": term,
            "byte_product": ("lo_lo", "lo_hi", "hi_lo")[term],
        }
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=lhs_byte, dst=mmb_lhs),
            attributes={**attributes, "operand": "lhs"},
        )
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=rhs_byte, dst=mmb_rhs),
            attributes={**attributes, "operand": "rhs"},
        )
        builder.add(
            "MUL_U8_TO_U16",
            APUG2RecipeOpKind.VL64,
            descriptors=uses(lhs=mmb_lhs, rhs=mmb_rhs, dst=mmb_product),
            metrics={
                "src_bits": 8,
                "dst_bits": 16,
                "vector_lane_updates": full_extent,
            },
            attributes=attributes,
        )
        if term:
            builder.add(
                "SHIFT_LEFT_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(src_dst=mmb_product),
                metrics={"shift": 8, "vector_lane_updates": full_extent},
                attributes=attributes,
            )
            builder.add(
                "ADD_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(lhs=l1_out, rhs=mmb_product, dst=mmb_product),
                metrics={"vector_lane_updates": full_extent},
                attributes={**attributes, "purpose": "combine_terms"},
            )
        builder.add(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=mmb_product, dst=l1_out),
            attributes={**attributes, "purpose": "retain_partial"},
        )

    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "mul",
            "arithmetic": "unsigned_multiply_mod_2_16",
            "output_extent": full_extent,
            "physical_sets": APUG2_MMB_SETS,
            "byte_products": ("lo_lo", "lo_hi", "hi_lo"),
            "omitted_term": "hi_hi_is_zero_mod_2_16",
        },
    )


def build_apu_g2_u16_fill_recipe(
    value: int = 0,
    *,
    name: str = "apu_g2_u16_fill",
) -> APUG2Recipe:
    """Build a full-vector uint16 scalar fill recipe."""

    value = _uint16_scalar("value", value)
    builder = _RecipeBuilder()
    immediate = APUG2Descriptor(
        "immediate", 16, num_vectors=APUG2_MMB_SETS, broadcast=True
    )
    mmb_out = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    full_extent = APUG2_MMB_SETS * APUG2_LANES
    builder.add(
        "COPY_IMMEDIATE_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=immediate, dst=mmb_out),
        metrics={"immediate": value},
        attributes={"operation": "broadcast_scalar_to_mmb", "value": value},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_out, dst=l1_out),
        metrics={"vector_lane_updates": full_extent},
        attributes={"purpose": "store_filled_output"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "fill",
            "arithmetic": "scalar_broadcast",
            "output_extent": full_extent,
            "physical_sets": APUG2_MMB_SETS,
            "value": value,
        },
    )


def build_apu_g2_u16_block_sum_recipe(
    log_block_size: int,
    *,
    name: str = "apu_g2_u16_block_sum",
) -> APUG2Recipe:
    """Build a uint16 block-sum recipe.

    The hardware reduction writes each block's total to the first element of
    that block.  Sums widen to ``16 + log_block_size`` bits in MMB and are
    truncated to uint16 at the L1 store boundary.
    """

    if isinstance(log_block_size, bool) or not isinstance(log_block_size, Integral):
        raise TypeError("log_block_size must be an integer")
    log_block_size = int(log_block_size)
    if not 0 <= log_block_size <= 8:
        raise ValueError("log_block_size must be in [0, 8] for uint16 sums")

    builder = _RecipeBuilder()
    l1_src = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    mmb_src = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_sum = APUG2Descriptor(
        "mmb",
        16 + log_block_size,
        num_vectors=APUG2_MMB_SETS,
        start_row=24,
        segment="seg1",
    )
    mmb_sum_low16 = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    full_extent = APUG2_MMB_SETS * APUG2_LANES
    block_size = 1 << log_block_size
    block_outputs = full_extent // block_size
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_src, dst=mmb_src),
        attributes={"operand": "source"},
    )
    builder.add(
        "GROUP_REDUCE_ADD_U16_TO_U23",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(src=mmb_src, dst=mmb_sum),
        metrics={
            "src_bits": 16,
            "dst_bits": 16 + log_block_size,
            "log_block_size": log_block_size,
            "block_size": block_size,
            "vector_lane_updates": block_outputs,
        },
        attributes={"operation": "block_sum", "result_lane": "block_first"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_sum_low16, dst=l1_out),
        attributes={"purpose": "store_low16_block_sums"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "block_sum",
            "arithmetic": "unsigned_block_sum_low16",
            "output_extent": full_extent,
            "block_outputs": block_outputs,
            "physical_sets": APUG2_MMB_SETS,
            "log_block_size": log_block_size,
            "block_size": block_size,
            "result_lane": "block_first",
        },
    )


def build_apu_g2_u16_shift_right_recipe(
    shift: int,
    *,
    name: str = "apu_g2_u16_shift_right",
) -> APUG2Recipe:
    """Build a full-vector logical right-shift recipe for uint16 packs."""

    if isinstance(shift, bool) or not isinstance(shift, Integral):
        raise TypeError("shift must be an integer")
    shift = int(shift)
    if not 0 <= shift <= 15:
        raise ValueError("shift must be in [0, 15] for uint16")

    builder = _RecipeBuilder()
    l1_src = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    mmb_src_out = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    full_extent = APUG2_MMB_SETS * APUG2_LANES
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_src, dst=mmb_src_out),
        attributes={"operand": "src", "purpose": "initialize_inplace_output"},
    )
    builder.add(
        "SHIFT_RIGHT_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(src_dst=mmb_src_out),
        metrics={"shift": shift, "vector_lane_updates": full_extent},
        attributes={"operation": "logical_right_shift", "inplace_segment": "seg1"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_src_out, dst=l1_out),
        attributes={"purpose": "store_shifted_output"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "shift_right",
            "arithmetic": "logical_right_shift",
            "output_extent": full_extent,
            "physical_sets": APUG2_MMB_SETS,
            "shift": shift,
        },
    )


def build_apu_g2_u16_sub_recipe(
    *,
    name: str = "apu_g2_u16_sub",
) -> APUG2Recipe:
    """Build a full-vector unsigned subtraction recipe for uint16 packs."""

    builder = _RecipeBuilder()
    l1_lhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_rhs = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=16)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=32)
    mmb_lhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    mmb_rhs_out = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_lhs, dst=mmb_lhs),
        attributes={"operand": "lhs"},
    )
    builder.add(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=l1_rhs, dst=mmb_rhs_out),
        attributes={"operand": "rhs", "purpose": "initialize_inplace_output"},
    )
    builder.add(
        "SUB_U16",
        APUG2RecipeOpKind.VL64,
        descriptors=uses(lhs=mmb_lhs, rhs_dst=mmb_rhs_out),
        metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
        attributes={"operation": "lhs - rhs", "inplace_segment": "seg1"},
    )
    builder.add(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        descriptors=uses(src=mmb_rhs_out, dst=l1_out),
        attributes={"purpose": "store_difference"},
    )
    builder.add("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        item.metrics["vector_lane_updates"] for item in builder.operations
    )
    certificate = APUG2VectorizationCertificate(
        vector_lane_updates=vector_updates,
        scalar_control_ops=0,
        scalar_tensor_updates=0,
    )
    return APUG2Recipe(
        name,
        tuple(builder.operations),
        certificate,
        metadata={
            "dtype": "uint16",
            "operation": "sub",
            "arithmetic": "unsigned_subtract_mod_2_16",
            "output_extent": APUG2_MMB_SETS * APUG2_LANES,
            "physical_sets": APUG2_MMB_SETS,
        },
    )


def build_apu_g2_u16_sqrt_recipe(
    *,
    name: str = "apu_g2_u16_sqrt",
) -> APUG2Recipe:
    """Build the full-vector floor-sqrt macro recipe used by the G2 task.

    The device kernel uses an eight-step binary-restoring macro.  Each step
    forms one candidate root bit, squares the candidate through the same
    three-term byte-product expansion used by modular multiplication, then
    accepts the candidate with a marker-controlled copy when
    ``candidate**2 <= src``.
    """

    builder = _RecipeBuilder()
    l1_src = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=0)
    l1_out = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=256)
    l1_candidate = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=448)
    l1_candidate_low = APUG2Descriptor(
        "l1", 8, num_vectors=APUG2_MMB_SETS, start_row=448
    )
    l1_candidate_high = APUG2Descriptor(
        "l1", 8, num_vectors=APUG2_MMB_SETS, start_row=456
    )
    l1_square = APUG2Descriptor("l1", 16, num_vectors=APUG2_MMB_SETS, start_row=640)
    operand0 = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    operand1 = APUG2Descriptor(
        "mmb", 8, num_vectors=APUG2_MMB_SETS, start_row=8, segment="seg0"
    )
    compare_lhs = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=0, segment="seg0"
    )
    work = APUG2Descriptor(
        "mmb", 16, num_vectors=APUG2_MMB_SETS, start_row=24, segment="seg1"
    )
    predicate = APUG2Descriptor(
        "mmb",
        1,
        num_vectors=APUG2_MMB_SETS,
        value_type="marker",
        start_row=23,
        segment="seg0",
    )
    immediate16 = APUG2Descriptor(
        "immediate", 16, num_vectors=APUG2_MMB_SETS, broadcast=True
    )

    def uses(**items):
        return tuple(
            APUG2DescriptorUse(role, descriptor) for role, descriptor in items.items()
        )

    def copy_immediate_to_l1(value: int, dst: APUG2Descriptor, purpose: str) -> None:
        builder.add(
            "COPY_IMMEDIATE_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=immediate16, dst=work),
            metrics={"immediate": value & 0xFFFF},
            attributes={"purpose": purpose, "value": value & 0xFFFF},
        )
        builder.add(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=work, dst=dst),
            attributes={"purpose": purpose},
        )

    def multiply_byte_pair(
        lhs_byte: APUG2Descriptor,
        rhs_byte: APUG2Descriptor,
        *,
        bit: int,
        term: str,
    ) -> None:
        common = {"sqrt_bit": bit, "term": term}
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=lhs_byte, dst=operand0),
            attributes={**common, "operand": "lhs_byte"},
        )
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=rhs_byte, dst=operand1),
            attributes={**common, "operand": "rhs_byte"},
        )
        builder.add(
            "MUL_U8_TO_U16",
            APUG2RecipeOpKind.VL64,
            descriptors=uses(lhs=operand0, rhs=operand1, dst=work),
            metrics={
                "src_bits": 8,
                "dst_bits": 16,
                "vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES,
            },
            attributes=common,
        )

    def retain_work(dst: APUG2Descriptor, *, bit: int, purpose: str) -> None:
        builder.add(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=work, dst=dst),
            attributes={"sqrt_bit": bit, "purpose": purpose},
        )

    copy_immediate_to_l1(0, l1_out, "initialize_root")
    for bit in reversed(range(8)):
        bit_value = 1 << bit
        builder.add(
            "ARC_SCALAR_CONTROL",
            APUG2RecipeOpKind.SCALAR_CONTROL,
            metrics={"vl64_calls": 0},
            attributes={"sqrt_bit": bit, "purpose": "fixed_root_bit_loop"},
        )
        builder.add(
            "COPY_IMMEDIATE_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=immediate16, dst=work),
            metrics={"immediate": bit_value},
            attributes={"sqrt_bit": bit, "purpose": "candidate_bit"},
        )
        builder.add(
            "ADD_U16",
            APUG2RecipeOpKind.VL64,
            descriptors=uses(lhs=l1_out, rhs=work, dst=work),
            metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
            attributes={"sqrt_bit": bit, "purpose": "candidate_with_bit"},
        )
        retain_work(l1_candidate, bit=bit, purpose="retain_candidate")

        multiply_byte_pair(l1_candidate_low, l1_candidate_low, bit=bit, term="lo_lo")
        retain_work(l1_square, bit=bit, purpose="retain_square_lo_lo")
        for term, lhs_byte, rhs_byte in (
            ("lo_hi", l1_candidate_low, l1_candidate_high),
            ("hi_lo", l1_candidate_high, l1_candidate_low),
        ):
            multiply_byte_pair(lhs_byte, rhs_byte, bit=bit, term=term)
            builder.add(
                "SHIFT_LEFT_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(src_dst=work),
                metrics={
                    "shift": 8,
                    "vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES,
                },
                attributes={"sqrt_bit": bit, "term": term},
            )
            builder.add(
                "ADD_U16",
                APUG2RecipeOpKind.VL64,
                descriptors=uses(lhs=l1_square, rhs=work, dst=work),
                metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
                attributes={"sqrt_bit": bit, "term": term, "purpose": "square_add"},
            )
            retain_work(l1_square, bit=bit, purpose=f"retain_square_{term}")

        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=l1_src, dst=compare_lhs),
            attributes={"sqrt_bit": bit, "purpose": "compare_src"},
        )
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=l1_square, dst=work),
            attributes={"sqrt_bit": bit, "purpose": "compare_square"},
        )
        builder.add(
            "LT_U16",
            APUG2RecipeOpKind.VL64,
            descriptors=uses(lhs=compare_lhs, rhs=work, dst=predicate),
            metrics={"vector_lane_updates": APUG2_MMB_SETS * APUG2_LANES},
            attributes={
                "sqrt_bit": bit,
                "predicate": "src_lt_candidate_square",
            },
        )
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=l1_candidate, dst=work),
            attributes={"sqrt_bit": bit, "purpose": "accept_candidate"},
        )
        builder.add(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=uses(src=l1_out, dst=work, predicate=predicate),
            attributes={
                "sqrt_bit": bit,
                "purpose": "keep_old_root_when_candidate_too_large",
                "masked": True,
            },
        )
        retain_work(l1_out, bit=bit, purpose="retain_root")

    builder.add(
        "SEU_BARRIER",
        APUG2RecipeOpKind.BARRIER,
        descriptors=uses(src=work, dst=work),
        attributes={"purpose": "complete_sqrt_macro"},
    )
    operations = tuple(builder.operations)
    return APUG2Recipe(
        name,
        operations,
        APUG2VectorizationCertificate(
            vector_lane_updates=sum(
                item.metrics["vector_lane_updates"] for item in operations
            ),
            scalar_control_ops=sum(
                item.kind == APUG2RecipeOpKind.SCALAR_CONTROL for item in operations
            ),
            scalar_tensor_updates=0,
        ),
        metadata={
            "dtype": "uint16",
            "operation": "sqrt",
            "arithmetic": "unsigned_floor_sqrt",
            "macro": "binary_restoring_floor_sqrt_u16",
            "steps": 8,
            "output_extent": APUG2_MMB_SETS * APUG2_LANES,
            "physical_sets": APUG2_MMB_SETS,
        },
    )


def build_apu_g2_u16_contraction_recipe(
    output_shape,
    reduction_extent: int,
    *,
    batch_rank: int = 0,
    alpha: int | None = None,
    beta: int | None = None,
    output_tile_extent: int | None = None,
    name: str = "apu_g2_u16_contraction",
) -> APUG2Recipe:
    """Flatten an arbitrary-rank output domain onto four-set dot tiles."""

    if not isinstance(output_shape, tuple) or not output_shape:
        raise TypeError("output_shape must be a nonempty tuple")
    shape = tuple(
        _positive_extent("output shape extent", item) for item in output_shape
    )
    if isinstance(batch_rank, bool) or not isinstance(batch_rank, Integral):
        raise TypeError("batch_rank must be an integer")
    batch_rank = int(batch_rank)
    if not 0 <= batch_rank <= len(shape):
        raise ValueError("batch_rank must be within output_shape rank")
    recipe = build_apu_g2_u16_dot_tile_recipe(
        math.prod(shape),
        reduction_extent,
        output_tile_extent=output_tile_extent,
        alpha=alpha,
        beta=beta,
        name=name,
    )
    return APUG2Recipe(
        recipe.name,
        recipe.operations,
        recipe.certificate,
        metadata={
            **dict(recipe.metadata),
            "operation": "rank_n_contraction",
            "output_shape": shape,
            "output_rank": len(shape),
            "batch_rank": batch_rank,
            "batch_shape": shape[:batch_rank],
            "per_batch_output_shape": shape[batch_rank:],
        },
    )


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
    "APUG2_DOT_TILE_MEASURED_POINTS",
    "APUG2Descriptor",
    "APUG2DescriptorUse",
    "APUG2Recipe",
    "APUG2RecipeCalibration",
    "APUG2RecipeMeasuredPoint",
    "APUG2RecipeOpKind",
    "APUG2RecipeOperation",
    "APUG2VectorizationCertificate",
    "build_apu_g2_recipe_graph",
    "build_apu_g2_u16_block_sum_recipe",
    "build_apu_g2_u16_shift_right_recipe",
    "build_apu_g2_u16_select_lt_recipe",
    "build_apu_g2_u16_minmax_recipe",
    "build_apu_g2_u16_div_recipe",
    "build_apu_g2_u16_mul_recipe",
    "build_apu_g2_u16_fill_recipe",
    "build_apu_g2_u16_sub_recipe",
    "build_apu_g2_u16_sqrt_recipe",
    "build_apu_g2_u16_contraction_recipe",
    "build_apu_g2_u16_dot_tile_recipe",
    "calibrate_apu_g2_recipe_from_measured_tile",
    "chain_apu_g2_recipes",
]
