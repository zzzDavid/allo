# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LinearLayout-aware composed integer contractions for Gemini-II.

The public program is deliberately expressed in terms of independent dots,
logical output geometry, exact scalar types, and an epilogue.  Workload names
and individual benchmark shapes are never consulted by legality, packing, or
device dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from .apu_g2_layout import APUG2DotTilePlan
from .apu_g2_typed_program import APUG2ScalarType


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _positive_shape(shape) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise TypeError("APUg2 composed output_shape must be a nonempty tuple")
    result = []
    for extent in shape:
        if isinstance(extent, (bool, np.bool_)) or not isinstance(extent, Integral):
            raise TypeError("APUg2 composed output extents must be integers")
        extent = int(extent)
        if extent <= 0:
            raise ValueError("APUg2 composed output extents must be positive")
        result.append(extent)
    return tuple(result)


def _positive_u32(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"APUg2 {name} must be an integer")
    value = int(value)
    if not 1 <= value <= np.iinfo(np.uint32).max:
        raise ValueError(f"APUg2 {name} must fit a nonzero uint32")
    return value


def _signed_bits(value: int) -> int:
    """Return the smallest two's-complement width containing ``value``."""

    if value >= 0:
        return max(2, value.bit_length() + 1)
    return max(2, (~value).bit_length() + 1)


class APUG2DotEpilogueMode(str, Enum):
    """Device-realizable post-reduction dataflow."""

    IDENTITY = "identity"
    PAIR_AFFINE = "pair_affine"
    ACCUMULATE = "accumulate"


@dataclass(frozen=True)
class APUG2DotEpilogue:
    """A small affine dataflow over resident dot results.

    ``coefficients=(1,)`` is an identity epilogue.  Two coefficients consume
    pairs of interleaved dots and combine them on the device.  Supplying an
    ``accumulator_type`` instead adds one same-layout accumulator to every dot.
    The two nontrivial forms are intentionally exclusive because they are the
    exact dataflow relations currently implemented by the target.
    """

    coefficients: tuple[int, ...] = (1,)
    accumulator_type: APUG2ScalarType | None = None

    def __post_init__(self):
        if not isinstance(self.coefficients, tuple):
            raise TypeError("APUg2 epilogue coefficients must be a tuple")
        coefficients = []
        for value in self.coefficients:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError("APUg2 epilogue coefficients must be integers")
            value = int(value)
            if not -(1 << 23) <= value < (1 << 23):
                raise ValueError("APUg2 epilogue coefficients must fit int24")
            coefficients.append(value)
        coefficients = tuple(coefficients)
        if coefficients not in ((1,),) and len(coefficients) != 2:
            raise ValueError(
                "APUg2 epilogue supports identity or two interleaved terms"
            )
        if not coefficients:
            raise ValueError("APUg2 epilogue requires at least one coefficient")
        if self.accumulator_type is not None and not isinstance(
            self.accumulator_type, APUG2ScalarType
        ):
            raise TypeError("APUg2 accumulator_type must be APUG2ScalarType")
        if self.accumulator_type is not None and coefficients != (1,):
            raise ValueError(
                "APUg2 pair-affine and accumulator epilogues are exclusive"
            )
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def mode(self) -> APUG2DotEpilogueMode:
        if self.accumulator_type is not None:
            return APUG2DotEpilogueMode.ACCUMULATE
        if len(self.coefficients) == 2:
            return APUG2DotEpilogueMode.PAIR_AFFINE
        return APUG2DotEpilogueMode.IDENTITY

    @property
    def terms_per_output(self) -> int:
        return len(self.coefficients)

    def coefficient_type(self, *, signed: bool) -> APUG2ScalarType | None:
        if self.mode is not APUG2DotEpilogueMode.PAIR_AFFINE:
            return None
        if signed:
            bits = max(_signed_bits(value) for value in self.coefficients)
            return APUG2ScalarType(bits, True)
        if any(value < 0 for value in self.coefficients):
            raise ValueError("unsigned APUg2 dots cannot use negative coefficients")
        bits = max(1, max(self.coefficients).bit_length())
        return APUG2ScalarType(bits, False)

    def manifest(self, *, signed: bool) -> dict[str, object]:
        coefficient_type = self.coefficient_type(signed=signed)
        return {
            "mode": self.mode.value,
            "coefficients": list(self.coefficients),
            "terms_per_output": self.terms_per_output,
            "coefficient_type": (
                None if coefficient_type is None else coefficient_type.manifest()
            ),
            "accumulator_type": (
                None
                if self.accumulator_type is None
                else self.accumulator_type.manifest()
            ),
        }


@dataclass(frozen=True)
class APUG2ComposedContractionProgram:
    """One full-carrier composed dot program.

    Inputs use the canonical logical shape ``(dot_count, reduction_extent)``.
    Dots belonging to the same logical result are adjacent.  Consequently,
    two-term affine programs use rows ``2*i`` and ``2*i+1``; independent
    multi-stream programs use a trailing stream axis in ``output_shape``.
    """

    input_types: tuple[APUG2ScalarType, APUG2ScalarType]
    output_type: APUG2ScalarType
    output_shape: tuple[int, ...]
    reduction_extent: int
    epilogue: APUG2DotEpilogue = field(default_factory=APUG2DotEpilogue)
    repetitions: int = 256
    name: str | None = None
    layout_plan: APUG2DotTilePlan = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.input_types, tuple)
            or len(self.input_types) != 2
            or not all(isinstance(item, APUG2ScalarType) for item in self.input_types)
        ):
            raise TypeError(
                "APUg2 composed input_types must contain two APUG2ScalarType values"
            )
        if not isinstance(self.output_type, APUG2ScalarType):
            raise TypeError("APUg2 composed output_type must be APUG2ScalarType")
        if not isinstance(self.epilogue, APUG2DotEpilogue):
            raise TypeError("APUg2 composed epilogue must be APUG2DotEpilogue")
        output_shape = _positive_shape(self.output_shape)
        reduction_extent = _positive_u32(
            "composed reduction extent", self.reduction_extent
        )
        if reduction_extent < 2 or reduction_extent & (reduction_extent - 1):
            raise ValueError(
                "APUg2 composed reduction extent must be a power of two >= 2"
            )
        repetitions = _positive_u32("composed profiling repetitions", self.repetitions)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "reduction_extent", reduction_extent)
        object.__setattr__(self, "repetitions", repetitions)

        lhs, rhs = self.input_types
        if lhs.bits < 2 or rhs.bits < 2:
            raise ValueError("APUg2 composed multiplication inputs need >= 2 bits")
        if lhs.signed != rhs.signed:
            raise ValueError("APUg2 composed multiplication needs shared signedness")
        product_bits = lhs.bits + rhs.bits
        dot_bits = product_bits + int(math.log2(reduction_extent))
        if product_bits > 23 or dot_bits > 24:
            raise ValueError("APUg2 composed dot exceeds one 24-row MMB segment")
        if self.output_type.signed != lhs.signed:
            raise ValueError("APUg2 composed output must preserve signedness")

        mode = self.epilogue.mode
        if mode is APUG2DotEpilogueMode.IDENTITY:
            expected_bits = dot_bits
        elif mode is APUG2DotEpilogueMode.PAIR_AFFINE:
            coefficient_type = self.epilogue.coefficient_type(signed=lhs.signed)
            scaled_bits = dot_bits + coefficient_type.bits
            expected_bits = scaled_bits + 1
            if scaled_bits > 24:
                raise ValueError(
                    "APUg2 pair-affine scaled branch exceeds one MMB segment"
                )
        else:
            accumulator_type = self.epilogue.accumulator_type
            if accumulator_type.signed != lhs.signed:
                raise ValueError("APUg2 accumulator must preserve signedness")
            if accumulator_type.bits != dot_bits:
                raise ValueError(
                    f"APUg2 accumulator must use the exact dot width {dot_bits}"
                )
            expected_bits = dot_bits + 1
        if expected_bits > 24:
            raise ValueError("APUg2 composed result exceeds one MMB segment")
        if self.output_type.bits != expected_bits:
            raise ValueError(
                f"APUg2 composed output must be exactly {expected_bits} bits"
            )

        plan = APUG2DotTilePlan(self.dot_count, reduction_extent)
        object.__setattr__(self, "layout_plan", plan)
        if self.name is None:
            object.__setattr__(
                self,
                "name",
                "apu_g2_composed_dot_"
                f"{lhs.name}_{rhs.name}_to_{self.output_type.name}_"
                f"{mode.value}",
            )
        elif not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("APUg2 composed program name must be an identifier")

    @property
    def dot_count(self) -> int:
        return math.prod(self.output_shape) * self.epilogue.terms_per_output

    @property
    def dot_shape(self) -> tuple[int, int]:
        return (self.dot_count, self.reduction_extent)

    @property
    def product_type(self) -> APUG2ScalarType:
        lhs, rhs = self.input_types
        return APUG2ScalarType(lhs.bits + rhs.bits, lhs.signed)

    @property
    def dot_type(self) -> APUG2ScalarType:
        return APUG2ScalarType(
            self.product_type.bits + int(math.log2(self.reduction_extent)),
            self.product_type.signed,
        )

    @property
    def auxiliary_type(self) -> APUG2ScalarType | None:
        if self.epilogue.mode is APUG2DotEpilogueMode.PAIR_AFFINE:
            return self.epilogue.coefficient_type(signed=self.input_types[0].signed)
        return self.epilogue.accumulator_type

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        shapes = [self.dot_shape, self.dot_shape]
        if self.epilogue.mode is APUG2DotEpilogueMode.ACCUMULATE:
            shapes.append(self.output_shape)
        return tuple(shapes)

    def layout_manifest(self) -> dict[str, object]:
        plan = self.layout_plan
        active_sets = sorted(
            {plan.physical_coordinate(row)[1] for row in range(plan.output_extent)}
        )
        manifest = {
            "schema": "apu-g2-dot-linear-layout-v1",
            "logical_axes": {
                "row": self.dot_count,
                "reduction": self.reduction_extent,
            },
            "validity": {
                "logical_rows": self.dot_count,
                "padded_rows": plan.padded_output_extent,
                "logical_reduction": self.reduction_extent,
                "padded_reduction": plan.padded_reduction_extent,
                "padding_value": 0,
            },
            "physical": {
                "l1_columns": 65_536,
                "mmb_sets": 4,
                "output_capacity": plan.output_capacity,
                "active_sets": active_sets,
            },
            "semantic_heads": {
                "predicate": "reduction == 0",
                "pair_affine_result_rows": (
                    "even logical dot rows"
                    if self.epilogue.mode is APUG2DotEpilogueMode.PAIR_AFFINE
                    else "all logical dot rows"
                ),
            },
            "linear_layout": plan.layout.manifest(),
        }
        return {**manifest, "fingerprint": _canonical_digest(manifest)}

    def structural_manifest(self) -> dict[str, object]:
        return {
            "schema": "apu-g2-composed-contraction-v1",
            "input_types": [item.manifest() for item in self.input_types],
            "product_type": self.product_type.manifest(),
            "dot_type": self.dot_type.manifest(),
            "output_type": self.output_type.manifest(),
            "output_shape": list(self.output_shape),
            "dot_shape": list(self.dot_shape),
            "reduction_extent": self.reduction_extent,
            "log_reduction": int(math.log2(self.reduction_extent)),
            "epilogue": self.epilogue.manifest(signed=self.input_types[0].signed),
            "layout": self.layout_manifest(),
        }

    @property
    def structural_fingerprint(self) -> str:
        return _canonical_digest(self.structural_manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "repetitions": self.repetitions,
            **self.structural_manifest(),
            "structural_fingerprint": self.structural_fingerprint,
        }

    def build(self):
        return self


__all__ = [
    "APUG2ComposedContractionProgram",
    "APUG2DotEpilogue",
    "APUG2DotEpilogueMode",
]
