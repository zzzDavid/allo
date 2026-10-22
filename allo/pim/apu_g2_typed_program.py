# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-bitwidth direct-VL64 programs for Gemini-II.

The legacy :class:`APUG2Program` surface is intentionally uint16-specific.
This module is the width-aware counterpart: lowering is selected from the
semantic operation, scalar types, and logical geometry.  Benchmark names and
individual matrix shapes never participate in legality or dispatch.
"""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np

from ..perf import BoundCostSpec
from ..spmw_codegen import RunResult
from .apu_g2_recipe import (
    APUG2Descriptor,
    APUG2DescriptorUse,
    APUG2Recipe,
    APUG2RecipeOpKind,
    APUG2RecipeOperation,
    APUG2VectorizationCertificate,
    build_apu_g2_recipe_graph,
)


APUG2_TYPED_PHYSICAL_VECTORS = 4
APUG2_TYPED_LANES_PER_VECTOR = 65_536
APUG2_TYPED_CAPACITY = APUG2_TYPED_PHYSICAL_VECTORS * APUG2_TYPED_LANES_PER_VECTOR
APUG2_TYPED_GROUP_LANES = 4_096
APUG2_TYPED_SEGMENT_BITS = 24
APUG2_TYPED_L1_BASE_ROW = 64
APUG2_TYPED_L1_SLOT_ALIGNMENT = 64
APUG2_TYPED_L1_SLOT_ROWS = APUG2_TYPED_PHYSICAL_VECTORS * APUG2_TYPED_SEGMENT_BITS
APUG2_TYPED_L1_SLOT_PITCH = (
    (APUG2_TYPED_L1_SLOT_ROWS + APUG2_TYPED_L1_SLOT_ALIGNMENT - 1)
    // APUG2_TYPED_L1_SLOT_ALIGNMENT
) * APUG2_TYPED_L1_SLOT_ALIGNMENT
APUG2_TYPED_L1_SLOT_BASES = tuple(
    APUG2_TYPED_L1_BASE_ROW + index * APUG2_TYPED_L1_SLOT_PITCH for index in range(4)
)
APUG2_TYPED_GTML_TEMP_ROW = 2_800
APUG2_TYPED_GTML_TEMP_ROWS = 128
APUG2_TYPED_GTML_INDEX_ROW = 2_928
APUG2_TYPED_GTML_INDEX_ROWS = 16


class APUG2TypedOperation(str, Enum):
    """Operations with a direct exact-width VL64 realization."""

    ADD = "add"
    MUL = "mul"
    DIV = "div"
    BLOCK_SUM = "block_sum"
    DOT = "dot"

    def __str__(self):
        return self.value


@dataclass(frozen=True, order=True)
class APUG2ScalarType:
    """Logical integer type stored in the narrowest NumPy byte container."""

    bits: int
    signed: bool

    def __post_init__(self):
        if isinstance(self.bits, (bool, np.bool_)) or not isinstance(
            self.bits, Integral
        ):
            raise TypeError("APUg2 scalar bits must be an integer")
        bits = int(self.bits)
        if not 1 <= bits <= APUG2_TYPED_SEGMENT_BITS:
            raise ValueError("APUg2 scalar bits must be in [1, 24]")
        if not isinstance(self.signed, (bool, np.bool_)):
            raise TypeError("APUg2 scalar signed flag must be boolean")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "signed", bool(self.signed))

    @property
    def name(self) -> str:
        return f"{'int' if self.signed else 'uint'}{self.bits}"

    @property
    def numpy_dtype(self) -> np.dtype:
        storage_bits = 8 if self.bits <= 8 else 16 if self.bits <= 16 else 32
        return np.dtype(f"{'int' if self.signed else 'uint'}{storage_bits}")

    @property
    def minimum(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def maximum(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bits": self.bits,
            "signed": self.signed,
            "numpy_storage_dtype": self.numpy_dtype.name,
        }


def _shape_tuple(shape) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise TypeError("APUg2 typed shape must be a nonempty tuple")
    extents = []
    for extent in shape:
        if isinstance(extent, (bool, np.bool_)) or not isinstance(extent, Integral):
            raise TypeError("APUg2 typed shape extents must be integers")
        extent = int(extent)
        if extent <= 0:
            raise ValueError("APUg2 typed shape extents must be positive")
        extents.append(extent)
    return tuple(extents)


def _positive_uint32(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"APUg2 {name} must be an integer")
    value = int(value)
    if not 1 <= value <= np.iinfo(np.uint32).max:
        raise ValueError(f"APUg2 {name} must fit a nonzero uint32")
    return value


def pack_apu_g2_gemm_as_independent_dots(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a generic rank-2 GEMM into independent row-major dot inputs.

    This is an ABI transform, not a matrix-shape specialization.  Output row
    ``m * N + n`` contains ``lhs[m, :]`` and ``rhs[:, n]``.  The returned
    arrays can be passed directly to a ``DOT`` :class:`APUG2TypedProgram`.
    """

    if not isinstance(lhs, np.ndarray) or not isinstance(rhs, np.ndarray):
        raise TypeError("APUg2 GEMM dot packing requires NumPy arrays")
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("APUg2 GEMM dot packing requires rank-2 operands")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError("APUg2 GEMM operands have incompatible reduction extents")
    if lhs.dtype != rhs.dtype:
        raise TypeError("APUg2 GEMM operands must use one storage dtype")
    rows, reduction = lhs.shape
    columns = rhs.shape[1]
    required = rows * columns * reduction
    if required > APUG2_TYPED_CAPACITY:
        raise ValueError(
            "APUg2 flattened GEMM dots exceed the four-vector VL64 carrier "
            f"({required} > {APUG2_TYPED_CAPACITY})"
        )
    left_dots = np.broadcast_to(
        lhs[:, np.newaxis, :],
        (rows, columns, reduction),
    ).reshape(rows * columns, reduction)
    right_dots = np.broadcast_to(
        rhs.T[np.newaxis, :, :],
        (rows, columns, reduction),
    ).reshape(rows * columns, reduction)
    return np.ascontiguousarray(left_dots), np.ascontiguousarray(right_dots)


def unpack_apu_g2_gemm_dots(
    dots: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Restore row-major GEMM output from compact independent dot results."""

    rows = _positive_uint32("GEMM row extent", rows)
    columns = _positive_uint32("GEMM column extent", columns)
    if not isinstance(dots, np.ndarray):
        raise TypeError("APUg2 GEMM dot output must be a NumPy array")
    if dots.shape != (rows * columns,):
        raise ValueError(f"APUg2 GEMM dot output must have shape {(rows * columns,)}")
    return np.ascontiguousarray(dots.reshape(rows, columns))


def _power_of_two(name: str, value) -> int:
    value = _positive_uint32(name, value)
    if value & (value - 1):
        raise ValueError(f"APUg2 {name} must be a power of two")
    if not 2 <= value <= APUG2_TYPED_GROUP_LANES:
        raise ValueError(f"APUg2 {name} must be in [2, {APUG2_TYPED_GROUP_LANES}]")
    return value


@dataclass(frozen=True)
class APUG2TypedProgram:
    """A generic direct-VL64 integer program.

    ``shape`` is the logical input shape.  Elementwise programs return that
    shape. ``block_sum`` returns one compact value per reduction block.
    ``dot`` requires ``shape == (output_extent, reduction_extent)`` and
    returns one compact value per independent row.
    """

    operation: APUG2TypedOperation | str
    input_types: tuple[APUG2ScalarType, ...]
    output_type: APUG2ScalarType
    shape: tuple[int, ...]
    reduction_extent: int | None = None
    repetitions: int = 256
    name: str | None = None

    def __post_init__(self):
        try:
            operation = APUG2TypedOperation(self.operation)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "APUg2 typed operation must be one of "
                + ", ".join(item.value for item in APUG2TypedOperation)
            ) from error
        object.__setattr__(self, "operation", operation)

        if not isinstance(self.input_types, tuple) or not all(
            isinstance(item, APUG2ScalarType) for item in self.input_types
        ):
            raise TypeError(
                "APUg2 input_types must be a tuple of APUG2ScalarType values"
            )
        if not isinstance(self.output_type, APUG2ScalarType):
            raise TypeError("APUg2 output_type must be an APUG2ScalarType")
        shape = _shape_tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        repetitions = _positive_uint32("profiling repetitions", self.repetitions)
        object.__setattr__(self, "repetitions", repetitions)

        numel = math.prod(shape)
        if numel > APUG2_TYPED_CAPACITY:
            raise ValueError(
                "APUg2 logical inputs exceed the four-vector VL64 carrier "
                f"({numel} > {APUG2_TYPED_CAPACITY})"
            )

        if operation in {
            APUG2TypedOperation.ADD,
            APUG2TypedOperation.MUL,
            APUG2TypedOperation.DIV,
        }:
            if self.reduction_extent is not None:
                raise ValueError(f"APUg2 {operation} does not take a reduction extent")
            if len(self.input_types) != 2:
                raise ValueError(f"APUg2 {operation} requires two input types")
        else:
            reduction = _power_of_two(
                "reduction extent",
                self.reduction_extent,
            )
            object.__setattr__(self, "reduction_extent", reduction)
            if len(self.input_types) != (
                1 if operation is APUG2TypedOperation.BLOCK_SUM else 2
            ):
                raise ValueError(
                    f"APUg2 {operation} has the wrong number of input types"
                )
            if operation is APUG2TypedOperation.DOT:
                if len(shape) != 2 or shape[1] != reduction:
                    raise ValueError(
                        "APUg2 dot shape must be " "(output_extent, reduction_extent)"
                    )
            elif numel % reduction:
                raise ValueError(
                    "APUg2 block-sum input elements must be divisible by "
                    "the reduction extent"
                )

        self._validate_type_legality()
        if self.name is None:
            type_signature = "_".join(item.name for item in self.input_types)
            object.__setattr__(
                self,
                "name",
                f"apu_g2_typed_{operation.value}_{type_signature}_to_"
                f"{self.output_type.name}",
            )
        elif not isinstance(self.name, str) or not self.name:
            raise TypeError("APUg2 typed program name must be a nonempty string")

    def _validate_type_legality(self):
        op = self.operation
        inputs = self.input_types
        output = self.output_type

        if op is APUG2TypedOperation.ADD:
            if not (inputs[0] == inputs[1] == output):
                raise ValueError(
                    "APUg2 add requires identical input and output scalar types"
                )
            return

        if op is APUG2TypedOperation.DIV:
            if any(item.signed for item in (*inputs, output)):
                raise ValueError("APUg2 direct division requires unsigned types")
            if not (inputs[0] == inputs[1] == output):
                raise ValueError(
                    "APUg2 division requires identical input and output types"
                )
            return

        if op in {APUG2TypedOperation.MUL, APUG2TypedOperation.DOT}:
            lhs, rhs = inputs
            if lhs.bits < 2 or rhs.bits < 2:
                raise ValueError("APUg2 multiplication inputs need at least 2 bits")
            if lhs.signed != rhs.signed or output.signed != lhs.signed:
                raise ValueError("APUg2 multiplication requires one shared signedness")
            product_bits = lhs.bits + rhs.bits
            if product_bits > APUG2_TYPED_SEGMENT_BITS:
                raise ValueError(
                    "APUg2 multiplication product exceeds one 24-row segment"
                )
            expected_bits = product_bits
            if op is APUG2TypedOperation.DOT:
                if product_bits > 23:
                    raise ValueError(
                        "APUg2 dot product source exceeds the sum primitive width"
                    )
                expected_bits += int(math.log2(self.reduction_extent))
            if output.bits != expected_bits:
                raise ValueError(
                    f"APUg2 {op} output must be exactly {expected_bits} bits"
                )
            if expected_bits > APUG2_TYPED_SEGMENT_BITS:
                raise ValueError(f"APUg2 {op} result exceeds one 24-row segment")
            return

        if op is APUG2TypedOperation.BLOCK_SUM:
            source = inputs[0]
            if source.signed or output.signed:
                raise ValueError("APUg2 block sum currently requires unsigned types")
            expected_bits = source.bits + int(math.log2(self.reduction_extent))
            if output.bits != expected_bits:
                raise ValueError(
                    "APUg2 block-sum output must be exactly " f"{expected_bits} bits"
                )
            if expected_bits > APUG2_TYPED_SEGMENT_BITS:
                raise ValueError("APUg2 block-sum result exceeds one 24-row segment")
            return

        raise AssertionError(f"unhandled APUg2 typed operation {op}")

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (self.shape,) * len(self.input_types)

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.operation is APUG2TypedOperation.DOT:
            return (self.shape[0],)
        if self.operation is APUG2TypedOperation.BLOCK_SUM:
            return (math.prod(self.shape) // self.reduction_extent,)
        return self.shape

    @property
    def log_reduction(self) -> int:
        return (
            0
            if self.reduction_extent is None
            else int(math.log2(self.reduction_extent))
        )

    def manifest(self) -> dict[str, object]:
        slot_types = [
            ("lhs", self.input_types[0]),
            (
                "rhs",
                (
                    self.input_types[0]
                    if len(self.input_types) == 1
                    else self.input_types[1]
                ),
            ),
            ("out", self.output_type),
            (
                "scratch",
                (
                    APUG2ScalarType(
                        self.input_types[0].bits + self.input_types[1].bits,
                        self.output_type.signed,
                    )
                    if self.operation is APUG2TypedOperation.DOT
                    else self.output_type
                ),
            ),
        ]
        l1_slots = [
            {
                "name": name,
                "start_row": start,
                "value_bits": scalar_type.bits,
                "physical_vectors": APUG2_TYPED_PHYSICAL_VECTORS,
                "live_rows": scalar_type.bits * APUG2_TYPED_PHYSICAL_VECTORS,
                "live_end_row_exclusive": (
                    start + scalar_type.bits * APUG2_TYPED_PHYSICAL_VECTORS
                ),
                "reserved_end_row_exclusive": (start + APUG2_TYPED_L1_SLOT_PITCH),
            }
            for (name, scalar_type), start in zip(slot_types, APUG2_TYPED_L1_SLOT_BASES)
        ]
        return {
            "operation": self.operation.value,
            "input_types": [item.manifest() for item in self.input_types],
            "output_type": self.output_type.manifest(),
            "shape": list(self.shape),
            "output_shape": list(self.output_shape),
            "reduction_extent": self.reduction_extent,
            "log_reduction": self.log_reduction,
            "repetitions": self.repetitions,
            "physical_vectors": APUG2_TYPED_PHYSICAL_VECTORS,
            "lanes_per_vector": APUG2_TYPED_LANES_PER_VECTOR,
            "l1_allocator": {
                "base_row": APUG2_TYPED_L1_BASE_ROW,
                "max_value_bits": APUG2_TYPED_SEGMENT_BITS,
                "slot_live_rows_at_max_width": APUG2_TYPED_L1_SLOT_ROWS,
                "slot_alignment_rows": APUG2_TYPED_L1_SLOT_ALIGNMENT,
                "slot_pitch_rows": APUG2_TYPED_L1_SLOT_PITCH,
                "slots": l1_slots,
                "gtml_reserved": {
                    "temp_start_row": APUG2_TYPED_GTML_TEMP_ROW,
                    "temp_end_row_exclusive": (
                        APUG2_TYPED_GTML_TEMP_ROW + APUG2_TYPED_GTML_TEMP_ROWS
                    ),
                    "index_start_row": APUG2_TYPED_GTML_INDEX_ROW,
                    "index_end_row_exclusive": (
                        APUG2_TYPED_GTML_INDEX_ROW + APUG2_TYPED_GTML_INDEX_ROWS
                    ),
                    "slots_end_before_temp": (
                        l1_slots[-1]["live_end_row_exclusive"]
                        <= APUG2_TYPED_GTML_TEMP_ROW
                    ),
                    "temp_end_before_index": (
                        APUG2_TYPED_GTML_TEMP_ROW + APUG2_TYPED_GTML_TEMP_ROWS
                        <= APUG2_TYPED_GTML_INDEX_ROW
                    ),
                },
            },
        }

    def build(self):
        return self


def _descriptor(
    storage: str,
    scalar_type: APUG2ScalarType,
    *,
    start_row: int,
    segment: str | None = None,
) -> APUG2Descriptor:
    return APUG2Descriptor(
        storage,
        scalar_type.bits,
        num_vectors=APUG2_TYPED_PHYSICAL_VECTORS,
        value_type="int" if scalar_type.signed else "uint",
        start_row=start_row,
        segment=segment,
    )


def _use(role: str, descriptor: APUG2Descriptor) -> APUG2DescriptorUse:
    return APUG2DescriptorUse(role, descriptor)


def build_apu_g2_typed_recipe(program: APUG2TypedProgram) -> APUG2Recipe:
    """Build the exact issued-call recipe for a typed program."""

    if not isinstance(program, APUG2TypedProgram):
        raise TypeError("typed recipe requires an APUG2TypedProgram")
    operations: list[APUG2RecipeOperation] = []
    last: str | None = None

    def issue(opcode, kind, descriptors=(), *, updates=0, **attributes):
        nonlocal last
        identifier = f"call_{len(operations):04d}_{opcode.lower()}"
        operations.append(
            APUG2RecipeOperation(
                identifier,
                opcode,
                kind,
                dependencies=() if last is None else (last,),
                descriptors=tuple(descriptors),
                metrics={
                    "vector_lane_updates": updates,
                    "logical_elements": math.prod(program.shape),
                },
                attributes=attributes,
            )
        )
        last = identifier

    lhs_type = program.input_types[0]
    # A slot's live footprint is ``num_bits * 4`` L1 rows.  The allocator
    # reserves an aligned carrier for the maximum legal 24-bit value.
    lhs_l1 = _descriptor("l1", lhs_type, start_row=APUG2_TYPED_L1_SLOT_BASES[0])
    lhs_mmb = _descriptor("mmb", lhs_type, start_row=0, segment="seg0")
    out_l1 = _descriptor(
        "l1", program.output_type, start_row=APUG2_TYPED_L1_SLOT_BASES[2]
    )

    issue(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        (_use("src", lhs_l1), _use("dst", lhs_mmb)),
    )

    if program.operation is APUG2TypedOperation.BLOCK_SUM:
        sum_mmb = _descriptor("mmb", program.output_type, start_row=24, segment="seg1")
        issue(
            "GROUP_REDUCE_ADD_TYPED",
            APUG2RecipeOpKind.VL64,
            (_use("src", lhs_mmb), _use("dst", sum_mmb)),
            reduction_extent=program.reduction_extent,
        )
        issue(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", sum_mmb), _use("dst", out_l1)),
            updates=APUG2_TYPED_CAPACITY,
        )
    else:
        rhs_type = program.input_types[1]
        rhs_l1 = _descriptor("l1", rhs_type, start_row=APUG2_TYPED_L1_SLOT_BASES[1])
        if program.operation in {
            APUG2TypedOperation.ADD,
            APUG2TypedOperation.DIV,
        }:
            rhs_mmb = _descriptor("mmb", rhs_type, start_row=24, segment="seg1")
        else:
            rhs_mmb = _descriptor(
                "mmb",
                rhs_type,
                start_row=lhs_type.bits,
                segment="seg0",
            )
        issue(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", rhs_l1), _use("dst", rhs_mmb)),
        )

        if program.operation is APUG2TypedOperation.ADD:
            out_mmb = _descriptor(
                "mmb", program.output_type, start_row=24, segment="seg1"
            )
            issue(
                "ADD_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", lhs_mmb),
                    _use("rhs", rhs_mmb),
                    _use("dst", out_mmb),
                ),
            )
            issue(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                (_use("src", out_mmb), _use("dst", out_l1)),
                updates=APUG2_TYPED_CAPACITY,
            )
        elif program.operation is APUG2TypedOperation.DIV:
            issue(
                "DIV_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", lhs_mmb),
                    _use("rhs", rhs_mmb),
                    _use("dst", out_l1),
                ),
                updates=APUG2_TYPED_CAPACITY,
            )
        else:
            product_type = APUG2ScalarType(
                lhs_type.bits + rhs_type.bits,
                lhs_type.signed,
            )
            product_mmb = _descriptor("mmb", product_type, start_row=24, segment="seg1")
            issue(
                "MUL_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", lhs_mmb),
                    _use("rhs", rhs_mmb),
                    _use("dst", product_mmb),
                ),
            )
            if program.operation is APUG2TypedOperation.MUL:
                issue(
                    "COPY_MMB_TO_L1_VECTORS",
                    APUG2RecipeOpKind.TRANSFER,
                    (_use("src", product_mmb), _use("dst", out_l1)),
                    updates=APUG2_TYPED_CAPACITY,
                )
            else:
                scratch_l1 = _descriptor(
                    "l1",
                    product_type,
                    start_row=APUG2_TYPED_L1_SLOT_BASES[3],
                )
                sum_source = _descriptor(
                    "mmb", product_type, start_row=0, segment="seg0"
                )
                sum_result = _descriptor(
                    "mmb", program.output_type, start_row=24, segment="seg1"
                )
                issue(
                    "COPY_MMB_TO_L1_VECTORS",
                    APUG2RecipeOpKind.TRANSFER,
                    (_use("src", product_mmb), _use("dst", scratch_l1)),
                )
                issue(
                    "COPY_L1_VECTORS_TO_MMB",
                    APUG2RecipeOpKind.TRANSFER,
                    (_use("src", scratch_l1), _use("dst", sum_source)),
                )
                issue(
                    "GROUP_REDUCE_ADD_TYPED",
                    APUG2RecipeOpKind.VL64,
                    (_use("src", sum_source), _use("dst", sum_result)),
                    reduction_extent=program.reduction_extent,
                )
                issue(
                    "COPY_MMB_TO_L1_VECTORS",
                    APUG2RecipeOpKind.TRANSFER,
                    (_use("src", sum_result), _use("dst", out_l1)),
                    updates=APUG2_TYPED_CAPACITY,
                )

    issue("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        operation.metrics["vector_lane_updates"] for operation in operations
    )
    return APUG2Recipe(
        name=program.name,
        operations=tuple(operations),
        certificate=APUG2VectorizationCertificate(
            vector_lane_updates=vector_updates,
            scalar_control_ops=0,
            scalar_tensor_updates=0,
        ),
        metadata={
            "operation": program.operation.value,
            "typed": True,
            "program": program.manifest(),
            "logical_output_extent": math.prod(program.output_shape),
            "physical_capacity": APUG2_TYPED_CAPACITY,
            "arithmetic": "exact_integer_width",
        },
    )


def build_apu_g2_typed_execution_graph(
    program: APUG2TypedProgram,
    target,
    cost: BoundCostSpec,
):
    if not isinstance(program, APUG2TypedProgram):
        raise TypeError("typed execution graph requires an APUG2TypedProgram")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 typed programs require the apu_v2 target")
    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("APUg2 typed programs require a target-bound cost")
    recipe = build_apu_g2_typed_recipe(program)
    graph = build_apu_g2_recipe_graph(recipe, target)
    graph.metadata.update(
        {
            "cost": cost.spec.name,
            "cost_fingerprint": cost.fingerprint,
            "recipe": recipe.manifest(),
            "recipe_fingerprint": recipe.structural_fingerprint,
            "execution": "direct_vl64",
            "analytical": True,
        }
    )
    return graph


class APUG2TypedCallable:
    """NumPy-callable exact-width Gemini-II program."""

    def __init__(self, program, target, *, cost, backend=None):
        if not isinstance(program, APUG2TypedProgram):
            raise TypeError("program must be an APUG2TypedProgram")
        if getattr(target, "name", None) != "apu_v2":
            raise ValueError("APUG2TypedProgram requires build_apu_g2_target()")
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 typed programs support device or virtual")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 typed programs require a target-bound cost")
        self.program = program
        self.target = target
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.execution_graph = build_apu_g2_typed_execution_graph(program, target, cost)
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.last_result = None
        self.__name__ = program.name
        input_names = (
            ("src",)
            if program.operation is APUG2TypedOperation.BLOCK_SUM
            else ("lhs", "rhs")
        )
        self.__signature__ = inspect.Signature(
            parameters=tuple(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in input_names
            )
            + (
                inspect.Parameter(
                    "out", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                ),
            )
        )

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(value, name, shape, scalar_type):
        array = np.asarray(value)
        if array.dtype != scalar_type.numpy_dtype:
            raise TypeError(
                f"APUg2 operand {name!r} must have storage dtype "
                f"{scalar_type.numpy_dtype.name}"
            )
        if array.shape != shape:
            raise ValueError(
                f"APUg2 operand {name!r} must have shape {shape}, " f"got {array.shape}"
            )
        if np.any(array < scalar_type.minimum) or np.any(array > scalar_type.maximum):
            raise ValueError(
                f"APUg2 operand {name!r} has values outside {scalar_type.name}"
            )
        return np.ascontiguousarray(array)

    def __call__(self, *args, **kwargs):
        out = kwargs.pop("out", None)
        if kwargs:
            raise TypeError(f"unexpected keyword argument {next(iter(kwargs))!r}")
        expected = len(self.program.input_types)
        if len(args) == expected + 1:
            if out is not None:
                raise TypeError("output supplied both positionally and by keyword")
            args, out = args[:-1], args[-1]
        if len(args) != expected:
            raise TypeError(
                f"APUg2 {self.program.operation} expects {expected} inputs "
                "and optional out"
            )
        names = (
            ("src",)
            if self.program.operation is APUG2TypedOperation.BLOCK_SUM
            else ("lhs", "rhs")
        )
        operands = tuple(
            self._array(value, name, shape, scalar_type)
            for value, name, shape, scalar_type in zip(
                args,
                names,
                self.program.input_shapes,
                self.program.input_types,
            )
        )
        if self.program.operation is APUG2TypedOperation.DIV and np.any(
            operands[1] == 0
        ):
            raise ValueError("APUg2 division requires nonzero divisors")
        if out is not None:
            if (
                not isinstance(out, np.ndarray)
                or out.dtype != self.program.output_type.numpy_dtype
            ):
                raise TypeError(
                    "APUg2 output must be a NumPy "
                    f"{self.program.output_type.numpy_dtype.name} array"
                )
            if out.shape != self.program.output_shape:
                raise ValueError(
                    f"APUg2 output must have shape {self.program.output_shape}"
                )
            if not out.flags.writeable:
                raise ValueError("APUg2 output must be writable")

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 typed cost evaluation; hardware not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "program": self.program.manifest(),
                    "recipe": self.execution_graph.metadata["recipe"],
                },
            )
        else:
            module = importlib.import_module(
                ".apu_g2_typed_runtime", package=__package__
            )
            result = module.run_apu_g2_typed(self.program, *operands)
            if out is not None:
                np.copyto(out, result.extra["outputs"]["out"])
        self.last_result = result
        return result

    run = __call__


def compile_apu_g2_typed_program(program, target, *, cost, backend=None):
    return APUG2TypedCallable(
        program,
        target,
        cost=cost,
        backend=backend,
    )


__all__ = [
    "APUG2ScalarType",
    "APUG2TypedCallable",
    "APUG2TypedOperation",
    "APUG2TypedProgram",
    "APUG2_TYPED_CAPACITY",
    "APUG2_TYPED_GROUP_LANES",
    "APUG2_TYPED_GTML_INDEX_ROW",
    "APUG2_TYPED_GTML_INDEX_ROWS",
    "APUG2_TYPED_GTML_TEMP_ROW",
    "APUG2_TYPED_GTML_TEMP_ROWS",
    "APUG2_TYPED_L1_BASE_ROW",
    "APUG2_TYPED_L1_SLOT_ALIGNMENT",
    "APUG2_TYPED_L1_SLOT_BASES",
    "APUG2_TYPED_L1_SLOT_PITCH",
    "APUG2_TYPED_L1_SLOT_ROWS",
    "APUG2_TYPED_LANES_PER_VECTOR",
    "APUG2_TYPED_PHYSICAL_VECTORS",
    "build_apu_g2_typed_execution_graph",
    "build_apu_g2_typed_recipe",
    "compile_apu_g2_typed_program",
    "pack_apu_g2_gemm_as_independent_dots",
    "unpack_apu_g2_gemm_dots",
]
