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


# --------------------------------------------------------------------------- #
# Typed/composed direct-VL64 tick model (release audit G2-03)
# --------------------------------------------------------------------------- #
#
# The typed (``APUG2TypedProgram``) and composed (``APUG2ComposedContractionProgram``)
# routes issue width-polymorphic calls: COPY_L1_VECTORS_TO_MMB,
# COPY_MMB_TO_L1_VECTORS, MUL_TYPED, GROUP_REDUCE_ADD_TYPED, ADD_TYPED,
# COPY_ODD_TO_EVEN_VECTORS, DIV_TYPED and one SEU_BARRIER per pipeline body.
# Every call covers the whole (4, 65536) carrier, so device time depends on the
# bit widths, the reduction depth, and the number of pipeline bodies, not on
# how many logical dots occupy the carrier.
#
# The model attributes one pipeline's device ticks to these structural
# features of its issued calls.  It is a normalized attribution fitted to whole
# pipelines, not a set of isolated instruction latencies.

APUG2_TYPED_TICK_FEATURES = (
    "pipeline_bodies",
    "issued_calls",
    "row_bits",
    "multiply_bit_products",
    "reduce_levels",
    "reduce_shift_columns",
    "divide_bits_squared",
)

_TRANSFER_OPCODES = frozenset({"COPY_L1_VECTORS_TO_MMB", "COPY_MMB_TO_L1_VECTORS"})
_ROW_SWEEP_OPCODES = frozenset({"ADD_TYPED", "COPY_ODD_TO_EVEN_VECTORS"})
APUG2_TYPED_TICK_OPCODES = frozenset(
    _TRANSFER_OPCODES
    | _ROW_SWEEP_OPCODES
    | {"MUL_TYPED", "GROUP_REDUCE_ADD_TYPED", "DIV_TYPED", "SEU_BARRIER"}
)


def apu_g2_typed_call_features(
    opcode: str,
    *,
    bits: int = 0,
    lhs_bits: int = 0,
    rhs_bits: int = 0,
    log_reduction: int = 0,
) -> dict[str, int]:
    """Structural tick features of one issued typed/composed call.

    ``bits`` is the destination value width of a transfer or row-sweep call
    (each of its rows is swept across all four vectors), ``lhs_bits`` and
    ``rhs_bits`` are the multiplicand widths of MUL_TYPED (bit-serial
    multiplication touches ``lhs_bits * rhs_bits`` partial-product rows), and
    ``log_reduction`` is the depth of the GROUP_REDUCE_ADD_TYPED tree, whose
    level ``l`` shifts by ``2**l`` columns.
    """

    features = dict.fromkeys(APUG2_TYPED_TICK_FEATURES, 0)
    if opcode == "SEU_BARRIER":
        features["pipeline_bodies"] = 1
        return features
    if opcode not in APUG2_TYPED_TICK_OPCODES:
        raise KeyError(f"no typed tick features for APUg2 opcode {opcode!r}")
    features["issued_calls"] = 1
    if opcode in _TRANSFER_OPCODES or opcode in _ROW_SWEEP_OPCODES:
        features["row_bits"] = _integer(bits, "bits", minimum=1)
    elif opcode == "MUL_TYPED":
        features["multiply_bit_products"] = _integer(
            lhs_bits, "lhs_bits", minimum=1
        ) * _integer(rhs_bits, "rhs_bits", minimum=1)
    elif opcode == "GROUP_REDUCE_ADD_TYPED":
        levels = _integer(log_reduction, "log_reduction", minimum=1)
        features["reduce_levels"] = levels
        features["reduce_shift_columns"] = (1 << levels) - 1
    else:
        width = _integer(bits, "bits", minimum=1)
        features["divide_bits_squared"] = width * width
    return features


@dataclass(frozen=True)
class APUG2TypedTickMeasurement(_CanonicalModel):
    """One real-card typed/composed program measurement (ticks per body)."""

    program: str
    route: str
    ticks_per_pipeline: int
    final_pipeline_ticks: int

    def __post_init__(self) -> None:
        if self.route not in ("typed", "composed"):
            raise ValueError("measurement route must be 'typed' or 'composed'")
        for name in ("ticks_per_pipeline", "final_pipeline_ticks"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "program": self.program,
            "route": self.route,
            "ticks_per_pipeline": self.ticks_per_pipeline,
            "final_pipeline_ticks": self.final_pipeline_ticks,
        }


# Round 2 device evidence (docs/release/VALIDATION.md): host en-ec-zhang-x4,
# card apu-00 (PCI 0000:61:00.0), firmware GSI_G2_Infrastructure_V1.24.1, SDK
# /opt/gsi, 2026-11-02, checkout ffb4a78 plus the wave-2 tree.  Every value is
# round(device_pipeline_ticks / 20): twenty pipeline bodies and one completion
# barrier per timed batch, transfers outside the timer, fresh random inputs,
# all outputs bit-exact against the runtime oracle.  Raw logs:
# tenon-release-evidence/g2/step2_3.log and step4.log.  The program keys name
# records rebuilt by ``apu_g2_typed_calibration_programs`` in
# ``allo.pim.costs.apu_g2``.
APUG2_TYPED_TICK_PROVENANCE = (
    ("source", "docs/release/VALIDATION.md Round 2 device evidence"),
    ("date", "2026-11-02"),
    ("host", "en-ec-zhang-x4"),
    ("card", "apu-00 PCI 0000:61:00.0"),
    ("firmware", "GSI_G2_Infrastructure_V1.24.1"),
    ("sdk", "/opt/gsi"),
    ("metric", "round(device_pipeline_ticks / repetitions), repetitions=20"),
    ("timed_scope", "pipeline bodies plus one completion barrier; no transfers"),
    ("logs", "tenon-release-evidence/g2/step2_3.log, step4.log"),
)

APUG2_TYPED_TICK_MEASUREMENTS = (
    APUG2TypedTickMeasurement("typed_add_i16", "typed", 333, 602),
    APUG2TypedTickMeasurement("typed_mul_i8_to_i16", "typed", 312, 451),
    APUG2TypedTickMeasurement("typed_div_u16", "typed", 1_019, 1_092),
    APUG2TypedTickMeasurement("typed_block_sum_u8_k256", "typed", 2_614, 2_676),
    APUG2TypedTickMeasurement("typed_dot_i4_k128", "typed", 1_713, 1_775),
    APUG2TypedTickMeasurement("composed_gemm_i4_k128", "composed", 1_711, 1_746),
    APUG2TypedTickMeasurement(
        "composed_batched_gemm_i4_k128", "composed", 1_720, 1_750
    ),
    APUG2TypedTickMeasurement("composed_pair_affine_i2_k128", "composed", 1_876, 1_838),
    APUG2TypedTickMeasurement("composed_two_stream_i3_k256", "composed", 2_569, 2_554),
    APUG2TypedTickMeasurement(
        "composed_two_stream_accumulate_i3_k256", "composed", 2_754, 2_754
    ),
)


@dataclass(frozen=True)
class APUG2TypedTickModel(_CanonicalModel):
    """Calibrated device-tick attribution for typed/composed recipe calls.

    ``coefficients`` maps every name in :data:`APUG2_TYPED_TICK_FEATURES` to
    ticks per unit of that feature.  ``calibrated_reduce_levels``,
    ``calibrated_divide_bits`` and ``calibrated_row_bits`` bound the measured
    domain: estimates for calls outside it are still produced (the selection
    needs a total order) but are labelled ``"extrapolated"`` by
    :meth:`validity`.
    """

    coefficients: tuple[tuple[str, float], ...]
    calibrated_reduce_levels: tuple[int, ...]
    calibrated_divide_bits: tuple[int, ...]
    calibrated_row_bits: tuple[int, int]
    calibrated_multiply_bit_products: tuple[int, int]
    fit: str
    provenance: tuple[tuple[str, str], ...] = APUG2_TYPED_TICK_PROVENANCE

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.coefficients)
        if names != APUG2_TYPED_TICK_FEATURES:
            raise ValueError(
                "tick model coefficients must cover APUG2_TYPED_TICK_FEATURES in order"
            )
        for name, value in self.coefficients:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"tick coefficient {name} must be a number")
            if value < 0:
                raise ValueError(f"tick coefficient {name} must be nonnegative")

    @property
    def coefficient_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.coefficients}

    def call_ticks(self, features) -> int:
        """Integer device ticks attributed to one issued call."""

        coefficients = self.coefficient_map
        unknown = set(features) - set(coefficients)
        if unknown:
            raise KeyError(f"unknown typed tick features {sorted(unknown)}")
        value = sum(coefficients[name] * features[name] for name in features)
        return int(round(value))

    def validity(self, features) -> str:
        """``"calibrated"`` inside the measured feature ranges, else
        ``"extrapolated"``."""

        if features.get("reduce_levels", 0) and (
            features["reduce_levels"] not in self.calibrated_reduce_levels
        ):
            return "extrapolated"
        divide = features.get("divide_bits_squared", 0)
        if divide and divide not in tuple(
            bits * bits for bits in self.calibrated_divide_bits
        ):
            return "extrapolated"
        low, high = self.calibrated_row_bits
        rows = features.get("row_bits", 0)
        if rows and not low <= rows <= high:
            return "extrapolated"
        low, high = self.calibrated_multiply_bit_products
        products = features.get("multiply_bit_products", 0)
        if products and not low <= products <= high:
            return "extrapolated"
        return "calibrated"

    def scaled(self, **factors: float) -> "APUG2TypedTickModel":
        """Return a perturbed copy with selected coefficients multiplied."""

        unknown = set(factors) - set(APUG2_TYPED_TICK_FEATURES)
        if unknown:
            raise KeyError(f"unknown typed tick features {sorted(unknown)}")
        return APUG2TypedTickModel(
            coefficients=tuple(
                (name, value * float(factors.get(name, 1.0)))
                for name, value in self.coefficients
            ),
            calibrated_reduce_levels=self.calibrated_reduce_levels,
            calibrated_divide_bits=self.calibrated_divide_bits,
            calibrated_row_bits=self.calibrated_row_bits,
            calibrated_multiply_bit_products=self.calibrated_multiply_bit_products,
            fit=f"{self.fit}; perturbed {sorted(factors.items())}",
            provenance=self.provenance,
        )

    @property
    def canonical_manifest(self) -> dict[str, object]:
        return {
            "schema": "apu-g2-typed-tick-model-v1",
            "unit": "device ARC ticks per pipeline body",
            "coefficients": dict(self.coefficients),
            "calibrated_reduce_levels": list(self.calibrated_reduce_levels),
            "calibrated_divide_bits": list(self.calibrated_divide_bits),
            "calibrated_row_bits": list(self.calibrated_row_bits),
            "calibrated_multiply_bit_products": list(
                self.calibrated_multiply_bit_products
            ),
            "fit": self.fit,
            "provenance": dict(self.provenance),
            "measurements": [
                item.canonical_manifest for item in APUG2_TYPED_TICK_MEASUREMENTS
            ],
        }


# Coefficients from ``fit_apu_g2_typed_tick_model()`` (allo.pim.costs.apu_g2)
# over APUG2_TYPED_TICK_MEASUREMENTS, rounded to three decimals; the release
# test re-fits and compares.  In-sample error <= 1.9% on all ten programs;
# leave-one-out error <= 2.6% on the five composed programs (see
# docs/release/design/apu-g2-source-route.md).  The calibrated reduction
# depths are only 7 and 8; the per-level 2**l shift scaling below them is an
# extrapolation whose shape agrees with the 2026-07-08 uint16 dot-tile
# measurements (increments 2,246 ticks from depth 6 to 7 and 4,793 from 7 to
# 8; that kernel and its measured points left the release with G2-11).
APUG2_TYPED_TICK_MODEL = APUG2TypedTickModel(
    coefficients=(
        ("pipeline_bodies", 195.704),
        ("issued_calls", 0.0),
        ("row_bits", 2.235),
        ("multiply_bit_products", 0.609),
        ("reduce_levels", 84.848),
        ("reduce_shift_columns", 6.505),
        ("divide_bits_squared", 2.937),
    ),
    calibrated_reduce_levels=(7, 8),
    calibrated_divide_bits=(16,),
    calibrated_row_bits=(2, 16),
    calibrated_multiply_bit_products=(4, 64),
    fit=(
        "exhaustive active-set nonnegative least squares over the ten Round 2 "
        "programs; features from their issued recipes"
    ),
)

__all__ = [
    "APUG2_TYPED_TICK_FEATURES",
    "APUG2_TYPED_TICK_MEASUREMENTS",
    "APUG2_TYPED_TICK_MODEL",
    "APUG2_TYPED_TICK_OPCODES",
    "APUG2_TYPED_TICK_PROVENANCE",
    "APUG2TypedTickMeasurement",
    "APUG2TypedTickModel",
    "apu_g2_typed_call_features",
    "CallInventoryEntry",
    "Opcode",
    "OpAttribution",
    "PipelineCalibration",
    "PipelineSignature",
    "PipelineTopology",
    "STANDALONE_4X64K_ADD_CALIBRATION",
    "ShapeValidityDomain",
    "VL64Scope",
]
