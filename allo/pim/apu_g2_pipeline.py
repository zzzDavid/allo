# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed structural calibrations for direct-VL64 pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
import hashlib
import json
from numbers import Integral


def _integer(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{field_name} must be {qualifier}")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _CanonicalModel:
    @property
    def canonical_manifest(self) -> dict[str, object]:
        raise NotImplementedError

    @property
    def key(self) -> str:
        return json.dumps(
            self.canonical_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()


@unique
class Opcode(str, Enum):
    COPY_L1_TO_MMB_SEGMENT_0 = "copy_l1_to_mmb_segment_0"
    COPY_L1_TO_MMB_SEGMENT_1 = "copy_l1_to_mmb_segment_1"
    COPY_MMB_TO_L1 = "copy_mmb_to_l1"
    MUL = "mul_u8_to_u16"
    REDUCE = "group_reduce_add_u16_to_u23"
    SHIFT = "shift_left_u16"
    ADD = "add_u16"
    SQUEEZE = "squeeze_rows_inplace"
    SPREAD = "spread_block"
    BARRIER = "barrier"


@unique
class PipelineTopology(str, Enum):
    STANDALONE = "standalone"
    NORMALIZED = "normalized"


@unique
class VL64Scope(str, Enum):
    CORE_WIDE = "core_wide"
    PER_GROUP = "per_group"


@dataclass(frozen=True)
class CallInventoryEntry(_CanonicalModel):
    opcode: Opcode
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, Opcode):
            raise TypeError("opcode must be an Opcode")
        object.__setattr__(self, "count", _integer(self.count, "count", minimum=1))

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {"opcode": self.opcode.value, "count": self.count}


@dataclass(frozen=True)
class ShapeValidityDomain(_CanonicalModel):
    extent_bounds: tuple[tuple[int, int], ...]
    power_of_two_padding: tuple[bool, ...] = ()
    padded_extent_limits: tuple[int | None, ...] = ()
    max_padded_volume: int | None = None

    def __post_init__(self) -> None:
        bounds = []
        for position, raw_bound in enumerate(tuple(self.extent_bounds)):
            try:
                lower, upper = tuple(raw_bound)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "each extent bound must contain a lower and upper integer"
                ) from error
            lower = _integer(lower, f"extent_bounds[{position}].lower", minimum=1)
            upper = _integer(upper, f"extent_bounds[{position}].upper", minimum=1)
            if lower > upper:
                raise ValueError("extent lower bound must not exceed its upper bound")
            bounds.append((lower, upper))
        if not bounds:
            raise ValueError("extent_bounds must not be empty")

        padding = tuple(self.power_of_two_padding)
        if not padding:
            padding = (False,) * len(bounds)
        if len(padding) != len(bounds) or any(
            type(flag) is not bool for flag in padding
        ):
            raise TypeError("power_of_two_padding must contain one boolean per extent")

        limits = tuple(self.padded_extent_limits)
        if not limits:
            limits = (None,) * len(bounds)
        if len(limits) != len(bounds):
            raise ValueError("padded_extent_limits must contain one limit per extent")
        normalized_limits = []
        for position, limit in enumerate(limits):
            normalized_limits.append(
                None
                if limit is None
                else _integer(limit, f"padded_extent_limits[{position}]", minimum=1)
            )

        volume = self.max_padded_volume
        if volume is not None:
            volume = _integer(volume, "max_padded_volume", minimum=1)

        object.__setattr__(self, "extent_bounds", tuple(bounds))
        object.__setattr__(self, "power_of_two_padding", padding)
        object.__setattr__(self, "padded_extent_limits", tuple(normalized_limits))
        object.__setattr__(self, "max_padded_volume", volume)

    def validate(self, *extents: int) -> tuple[int, ...]:
        if len(extents) == 1 and not isinstance(extents[0], Integral):
            try:
                extents = tuple(extents[0])
            except TypeError:
                pass
        if len(extents) != len(self.extent_bounds):
            raise ValueError(
                f"shape must contain exactly {len(self.extent_bounds)} extents"
            )

        validated = []
        padded = []
        for position, (raw_extent, bounds, use_padding, limit) in enumerate(
            zip(
                extents,
                self.extent_bounds,
                self.power_of_two_padding,
                self.padded_extent_limits,
            )
        ):
            extent = _integer(raw_extent, f"extent[{position}]", minimum=1)
            lower, upper = bounds
            if not lower <= extent <= upper:
                raise ValueError(f"extent[{position}] must be in [{lower}, {upper}]")
            physical_extent = _next_power_of_two(extent) if use_padding else extent
            if limit is not None and physical_extent > limit:
                raise ValueError(f"padded extent[{position}] must not exceed {limit}")
            validated.append(extent)
            padded.append(physical_extent)

        if self.max_padded_volume is not None:
            padded_volume = 1
            for extent in padded:
                padded_volume *= extent
            if padded_volume > self.max_padded_volume:
                raise ValueError("padded shape volume exceeds the structural capacity")
        return tuple(validated)

    def contains(self, *extents: int) -> bool:
        try:
            self.validate(*extents)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "extent_bounds": [list(bounds) for bounds in self.extent_bounds],
            "power_of_two_padding": list(self.power_of_two_padding),
            "padded_extent_limits": list(self.padded_extent_limits),
            "max_padded_volume": self.max_padded_volume,
        }


@dataclass(frozen=True)
class PipelineSignature(_CanonicalModel):
    topology: PipelineTopology
    coalesced_matrix_streams: int
    call_inventory: tuple[CallInventoryEntry, ...]
    resident_intermediate: bool
    vl64_scope: VL64Scope
    vl64_group_count: int
    shape_domain: ShapeValidityDomain

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PipelineTopology):
            raise TypeError("topology must be a PipelineTopology")
        if not isinstance(self.vl64_scope, VL64Scope):
            raise TypeError("vl64_scope must be a VL64Scope")
        if type(self.resident_intermediate) is not bool:
            raise TypeError("resident_intermediate must be a boolean")
        if not isinstance(self.shape_domain, ShapeValidityDomain):
            raise TypeError("shape_domain must be a ShapeValidityDomain")

        streams = _integer(
            self.coalesced_matrix_streams,
            "coalesced_matrix_streams",
            minimum=0,
        )
        groups = _integer(self.vl64_group_count, "vl64_group_count", minimum=1)
        inventory = tuple(self.call_inventory)
        if not inventory:
            raise ValueError("call_inventory must not be empty")
        seen = set()
        for entry in inventory:
            if not isinstance(entry, CallInventoryEntry):
                raise TypeError("call_inventory must contain CallInventoryEntry values")
            if entry.opcode in seen:
                raise ValueError("call_inventory must not repeat an opcode")
            seen.add(entry.opcode)

        object.__setattr__(self, "coalesced_matrix_streams", streams)
        object.__setattr__(self, "vl64_group_count", groups)
        object.__setattr__(self, "call_inventory", inventory)

    @property
    def coalesced_matrix_stream_count(self) -> int:
        return self.coalesced_matrix_streams

    def validate_shape(self, *extents: int) -> tuple[int, ...]:
        return self.shape_domain.validate(*extents)

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "topology": self.topology.value,
            "coalesced_matrix_streams": self.coalesced_matrix_streams,
            "call_inventory": [
                entry.canonical_manifest for entry in self.call_inventory
            ],
            "resident_intermediate": self.resident_intermediate,
            "vl64_scope": self.vl64_scope.value,
            "vl64_group_count": self.vl64_group_count,
            "shape_domain": self.shape_domain.canonical_manifest,
        }


@dataclass(frozen=True)
class OpAttribution(_CanonicalModel):
    opcode: Opcode
    cycles_per_call: int

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, Opcode):
            raise TypeError("opcode must be an Opcode")
        object.__setattr__(
            self,
            "cycles_per_call",
            _integer(self.cycles_per_call, "cycles_per_call", minimum=1),
        )

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "opcode": self.opcode.value,
            "cycles_per_call": self.cycles_per_call,
        }


@dataclass(frozen=True)
class PipelineCalibration(_CanonicalModel):
    signature: PipelineSignature
    attributions: tuple[OpAttribution, ...]
    total_cycles: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.signature, PipelineSignature):
            raise TypeError("signature must be a PipelineSignature")

        by_opcode = {}
        for attribution in tuple(self.attributions):
            if not isinstance(attribution, OpAttribution):
                raise TypeError("attributions must contain OpAttribution values")
            if attribution.opcode in by_opcode:
                raise ValueError("attributions must not repeat an opcode")
            by_opcode[attribution.opcode] = attribution

        inventory_opcodes = tuple(
            entry.opcode for entry in self.signature.call_inventory
        )
        if set(by_opcode) != set(inventory_opcodes):
            raise ValueError(
                "attributions must cover exactly the call inventory opcodes"
            )
        ordered_attributions = tuple(by_opcode[opcode] for opcode in inventory_opcodes)
        cycles_by_opcode = {
            attribution.opcode: attribution.cycles_per_call
            for attribution in ordered_attributions
        }
        total_cycles = sum(
            entry.count * cycles_by_opcode[entry.opcode]
            for entry in self.signature.call_inventory
        )

        object.__setattr__(self, "attributions", ordered_attributions)
        object.__setattr__(self, "total_cycles", total_cycles)

    def require_signature(self, signature: PipelineSignature) -> None:
        if not isinstance(signature, PipelineSignature):
            raise TypeError("signature must be a PipelineSignature")
        if signature != self.signature:
            raise ValueError("calibration does not match the pipeline signature")

    def cycles_for(self, signature: PipelineSignature) -> int:
        self.require_signature(signature)
        return self.total_cycles

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "signature": self.signature.canonical_manifest,
            "attributions": [
                attribution.canonical_manifest for attribution in self.attributions
            ],
            "total_cycles": self.total_cycles,
        }


_CORE_GROUP_COUNT = 16

_STANDALONE_4X64K_DOMAIN = ShapeValidityDomain(
    extent_bounds=((4, 4), (65_536, 65_536)),
)

_DIRECT_REDUCTION_DOMAIN = ShapeValidityDomain(
    extent_bounds=((1, 65_536), (1, 256)),
    power_of_two_padding=(True, True),
    padded_extent_limits=(65_536, 256),
    max_padded_volume=65_536,
)

_RESIDENT_TWO_STAGE_DOMAIN = ShapeValidityDomain(
    extent_bounds=((1, 128), (1, 128)),
)


STANDALONE_4X64K_ADD_CALIBRATION = PipelineCalibration(
    signature=PipelineSignature(
        topology=PipelineTopology.STANDALONE,
        coalesced_matrix_streams=0,
        call_inventory=(
            CallInventoryEntry(Opcode.COPY_L1_TO_MMB_SEGMENT_0, 1),
            CallInventoryEntry(Opcode.COPY_L1_TO_MMB_SEGMENT_1, 1),
            CallInventoryEntry(Opcode.ADD, 1),
            CallInventoryEntry(Opcode.COPY_MMB_TO_L1, 1),
            CallInventoryEntry(Opcode.BARRIER, 1),
        ),
        resident_intermediate=False,
        vl64_scope=VL64Scope.CORE_WIDE,
        vl64_group_count=_CORE_GROUP_COUNT,
        shape_domain=_STANDALONE_4X64K_DOMAIN,
    ),
    attributions=(
        OpAttribution(Opcode.COPY_L1_TO_MMB_SEGMENT_0, 56),
        OpAttribution(Opcode.COPY_L1_TO_MMB_SEGMENT_1, 56),
        OpAttribution(Opcode.ADD, 98),
        OpAttribution(Opcode.COPY_MMB_TO_L1, 55),
        OpAttribution(Opcode.BARRIER, 1),
    ),
)


NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION = PipelineCalibration(
    signature=PipelineSignature(
        topology=PipelineTopology.NORMALIZED,
        coalesced_matrix_streams=1,
        call_inventory=(
            CallInventoryEntry(Opcode.MUL, 3),
            CallInventoryEntry(Opcode.REDUCE, 3),
            CallInventoryEntry(Opcode.SHIFT, 2),
            CallInventoryEntry(Opcode.ADD, 3),
            CallInventoryEntry(Opcode.BARRIER, 1),
        ),
        resident_intermediate=False,
        vl64_scope=VL64Scope.CORE_WIDE,
        vl64_group_count=_CORE_GROUP_COUNT,
        shape_domain=_DIRECT_REDUCTION_DOMAIN,
    ),
    attributions=(
        OpAttribution(Opcode.MUL, 580),
        OpAttribution(Opcode.REDUCE, 1_474),
        OpAttribution(Opcode.SHIFT, 179),
        OpAttribution(Opcode.ADD, 124),
        OpAttribution(Opcode.BARRIER, 1),
    ),
)


NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION = PipelineCalibration(
    signature=PipelineSignature(
        topology=PipelineTopology.NORMALIZED,
        coalesced_matrix_streams=2,
        call_inventory=(
            CallInventoryEntry(Opcode.MUL, 6),
            CallInventoryEntry(Opcode.REDUCE, 3),
            CallInventoryEntry(Opcode.SHIFT, 4),
            CallInventoryEntry(Opcode.ADD, 5),
            CallInventoryEntry(Opcode.BARRIER, 1),
        ),
        resident_intermediate=False,
        vl64_scope=VL64Scope.CORE_WIDE,
        vl64_group_count=_CORE_GROUP_COUNT,
        shape_domain=_DIRECT_REDUCTION_DOMAIN,
    ),
    attributions=(
        OpAttribution(Opcode.MUL, 426),
        OpAttribution(Opcode.REDUCE, 1_152),
        OpAttribution(Opcode.SHIFT, 142),
        OpAttribution(Opcode.ADD, 284),
        OpAttribution(Opcode.BARRIER, 1),
    ),
)


NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION = PipelineCalibration(
    signature=PipelineSignature(
        topology=PipelineTopology.NORMALIZED,
        coalesced_matrix_streams=1,
        call_inventory=(
            CallInventoryEntry(Opcode.MUL, 15),
            CallInventoryEntry(Opcode.REDUCE, 15),
            CallInventoryEntry(Opcode.SHIFT, 10),
            CallInventoryEntry(Opcode.ADD, 16),
            CallInventoryEntry(Opcode.SQUEEZE, 20),
            CallInventoryEntry(Opcode.SPREAD, 4),
            CallInventoryEntry(Opcode.BARRIER, 1),
        ),
        resident_intermediate=True,
        vl64_scope=VL64Scope.CORE_WIDE,
        vl64_group_count=_CORE_GROUP_COUNT,
        shape_domain=_RESIDENT_TWO_STAGE_DOMAIN,
    ),
    attributions=(
        OpAttribution(Opcode.MUL, 580),
        OpAttribution(Opcode.REDUCE, 1_474),
        OpAttribution(Opcode.SHIFT, 179),
        OpAttribution(Opcode.ADD, 124),
        OpAttribution(Opcode.SQUEEZE, 1_200),
        OpAttribution(Opcode.SPREAD, 4_382),
        OpAttribution(Opcode.BARRIER, 6),
    ),
)


__all__ = [
    "CallInventoryEntry",
    "NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION",
    "NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION",
    "NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION",
    "Opcode",
    "OpAttribution",
    "PipelineCalibration",
    "PipelineSignature",
    "PipelineTopology",
    "STANDALONE_4X64K_ADD_CALIBRATION",
    "ShapeValidityDomain",
    "VL64Scope",
]
