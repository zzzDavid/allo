# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side layout and ABI helpers for composed APUg2 contractions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .apu_g2_composed_contraction import (
    APUG2ComposedContractionProgram,
    APUG2DotEpilogueMode,
)
from .apu_g2_layout import APUG2_U16_SHAPE
from .apu_g2_typed_program import APUG2ScalarType


APUG2_COMPOSED_EPILOGUE_ABI = {
    APUG2DotEpilogueMode.IDENTITY: 0,
    APUG2DotEpilogueMode.PAIR_AFFINE: 1,
    APUG2DotEpilogueMode.ACCUMULATE: 2,
}


def canonicalize_apu_g2_values(values, scalar_type: APUG2ScalarType) -> np.ndarray:
    """Apply one declared logical width inside its native NumPy container."""

    if not isinstance(scalar_type, APUG2ScalarType):
        raise TypeError("scalar_type must be APUG2ScalarType")
    wide = np.asarray(values, dtype=np.int64)
    modulus = 1 << scalar_type.bits
    payload = np.mod(wide, modulus)
    if scalar_type.signed:
        sign = 1 << (scalar_type.bits - 1)
        payload = np.where(payload >= sign, payload - modulus, payload)
    return payload.astype(scalar_type.numpy_dtype)


def decode_apu_g2_u16_bitpatterns(
    values,
    scalar_type: APUG2ScalarType,
) -> np.ndarray:
    """Decode portable little-endian uint16 fixture payloads at logical width."""

    values = np.asarray(values)
    if values.dtype != np.dtype("<u2") and values.dtype != np.dtype(np.uint16):
        raise TypeError("canonical APUg2 fixtures must use uint16 containers")
    return canonicalize_apu_g2_values(values, scalar_type)


def encode_apu_g2_u16_bitpatterns(
    values,
    scalar_type: APUG2ScalarType,
) -> np.ndarray:
    """Encode exact-width signed or unsigned values as portable uint16 payloads."""

    canonical = canonicalize_apu_g2_values(values, scalar_type)
    mask = (1 << scalar_type.bits) - 1
    return np.bitwise_and(
        canonical.astype(np.int64),
        mask,
    ).astype("<u2")


def _logical_operand(
    program: APUG2ComposedContractionProgram,
    value,
    index: int,
) -> np.ndarray:
    if not isinstance(program, APUG2ComposedContractionProgram):
        raise TypeError("program must be APUG2ComposedContractionProgram")
    if index not in (0, 1):
        raise IndexError("composed dot operand index must be zero or one")
    scalar_type = program.input_types[index]
    array = np.asarray(value)
    if array.dtype != scalar_type.numpy_dtype:
        raise TypeError(
            f"operand {index} must have dtype {scalar_type.numpy_dtype.name}"
        )
    if array.shape != program.dot_shape:
        raise ValueError(f"operand {index} must have shape {program.dot_shape}")
    if np.any(array < scalar_type.minimum) or np.any(array > scalar_type.maximum):
        raise ValueError(f"operand {index} contains values outside {scalar_type.name}")
    return np.ascontiguousarray(array)


def pack_apu_g2_composed_operand(
    program: APUG2ComposedContractionProgram,
    value,
    index: int,
) -> np.ndarray:
    """Pack one logical ``[dot,reduction]`` operand through its LinearLayout."""

    logical = _logical_operand(program, value, index)
    scalar_type = program.input_types[index]
    physical = np.zeros(APUG2_U16_SHAPE, dtype=scalar_type.numpy_dtype)
    offsets = np.arange(program.reduction_extent, dtype=np.int64)
    for row in range(program.dot_count):
        column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
        physical[mmb_set, column + offsets] = logical[row]
    return physical


def _logical_accumulator(
    program: APUG2ComposedContractionProgram,
    accumulator,
) -> np.ndarray:
    if program.epilogue.mode is not APUG2DotEpilogueMode.ACCUMULATE:
        raise ValueError("only an accumulate epilogue accepts an accumulator")
    scalar_type = program.epilogue.accumulator_type
    array = np.asarray(accumulator)
    if array.dtype != scalar_type.numpy_dtype:
        raise TypeError(
            "accumulator must have dtype " f"{scalar_type.numpy_dtype.name}"
        )
    if array.shape != program.output_shape:
        raise ValueError(f"accumulator must have shape {program.output_shape}")
    if np.any(array < scalar_type.minimum) or np.any(array > scalar_type.maximum):
        raise ValueError(f"accumulator contains values outside {scalar_type.name}")
    return np.ascontiguousarray(array)


def pack_apu_g2_composed_auxiliary(
    program: APUG2ComposedContractionProgram,
    accumulator=None,
) -> np.ndarray | None:
    """Pack compile-time coefficients or a logical same-layout accumulator."""

    mode = program.epilogue.mode
    if mode is APUG2DotEpilogueMode.IDENTITY:
        if accumulator is not None:
            raise ValueError("identity epilogue does not accept an accumulator")
        return None

    scalar_type = program.auxiliary_type
    physical = np.zeros(APUG2_U16_SHAPE, dtype=scalar_type.numpy_dtype)
    if mode is APUG2DotEpilogueMode.PAIR_AFFINE:
        if accumulator is not None:
            raise ValueError("pair-affine coefficients are program constants")
        logical = np.resize(
            np.asarray(program.epilogue.coefficients, dtype=scalar_type.numpy_dtype),
            program.dot_count,
        )
    else:
        logical = _logical_accumulator(program, accumulator).reshape(-1)

    for row, value in enumerate(logical):
        column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
        physical[mmb_set, column] = value
    return physical


def gather_apu_g2_composed_output(
    program: APUG2ComposedContractionProgram,
    physical,
) -> np.ndarray:
    """Gather semantic reduction heads from one physical output carrier."""

    physical = np.asarray(physical)
    if physical.dtype != program.output_type.numpy_dtype:
        raise TypeError(
            "physical output must have dtype " f"{program.output_type.numpy_dtype.name}"
        )
    if physical.shape != APUG2_U16_SHAPE:
        raise ValueError(f"physical output must have shape {APUG2_U16_SHAPE}")
    count = math.prod(program.output_shape)
    row_step = program.epilogue.terms_per_output
    result = np.empty(count, dtype=program.output_type.numpy_dtype)
    for output_index in range(count):
        row = output_index * row_step
        column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
        result[output_index] = physical[mmb_set, column]
    return result.reshape(program.output_shape)


def expected_apu_g2_composed_output(
    program: APUG2ComposedContractionProgram,
    lhs,
    rhs,
    *,
    accumulator=None,
) -> np.ndarray:
    """Evaluate the exact semantic dot-and-epilogue program in NumPy."""

    lhs = _logical_operand(program, lhs, 0)
    rhs = _logical_operand(program, rhs, 1)
    dots = np.sum(
        lhs.astype(np.int64) * rhs.astype(np.int64),
        axis=1,
        dtype=np.int64,
    )
    mode = program.epilogue.mode
    if mode is APUG2DotEpilogueMode.PAIR_AFFINE:
        coefficients = np.asarray(program.epilogue.coefficients, dtype=np.int64)
        values = np.sum(
            dots.reshape(-1, len(coefficients)) * coefficients,
            axis=1,
            dtype=np.int64,
        )
    elif mode is APUG2DotEpilogueMode.ACCUMULATE:
        values = dots + _logical_accumulator(program, accumulator).reshape(-1)
    else:
        if accumulator is not None:
            raise ValueError("identity epilogue does not accept an accumulator")
        values = dots
    return canonicalize_apu_g2_values(
        values.reshape(program.output_shape),
        program.output_type,
    )


def pack_apu_g2_batched_gemm_dots(
    lhs,
    rhs,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn rank-N batched GEMM operands into row-major independent dots."""

    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.ndim < 2 or rhs.ndim != lhs.ndim:
        raise ValueError("batched GEMM operands must have one shared rank >= 2")
    if lhs.shape[:-2] != rhs.shape[:-2]:
        raise ValueError("batched GEMM operands need identical batch geometry")
    if lhs.shape[-1] != rhs.shape[-2]:
        raise ValueError("batched GEMM reduction extents do not match")
    if lhs.dtype != rhs.dtype:
        raise TypeError("batched GEMM operands must use one storage dtype")
    batch_shape = lhs.shape[:-2]
    rows, reduction = lhs.shape[-2:]
    columns = rhs.shape[-1]
    left = np.broadcast_to(
        lhs[..., :, np.newaxis, :],
        (*batch_shape, rows, columns, reduction),
    )
    right = np.broadcast_to(
        np.swapaxes(rhs, -1, -2)[..., np.newaxis, :, :],
        (*batch_shape, rows, columns, reduction),
    )
    return (
        np.ascontiguousarray(left.reshape(-1, reduction)),
        np.ascontiguousarray(right.reshape(-1, reduction)),
    )


def pack_apu_g2_matrix_vector_dots(
    matrix,
    vector,
    *,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one matrix-vector contraction into independent dot operands."""

    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("matrix-vector packing requires rank-2/rank-1 inputs")
    logical_matrix = matrix.T if transpose else matrix
    if logical_matrix.shape[1] != vector.shape[0]:
        raise ValueError("matrix-vector reduction extents do not match")
    if logical_matrix.dtype != vector.dtype:
        raise TypeError("matrix and vector must use one storage dtype")
    right = np.broadcast_to(vector, logical_matrix.shape)
    return (
        np.ascontiguousarray(logical_matrix),
        np.ascontiguousarray(right),
    )


def interleave_apu_g2_dot_streams(
    *streams: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Interleave equally shaped dot streams along the logical row axis."""

    if not streams:
        raise ValueError("at least one dot stream is required")
    normalized = []
    shape = None
    dtype = None
    for stream in streams:
        if not isinstance(stream, tuple) or len(stream) != 2:
            raise TypeError("each dot stream must be a (left, right) tuple")
        left, right = (np.asarray(item) for item in stream)
        if left.ndim != 2 or right.shape != left.shape:
            raise ValueError("dot-stream operands must share one rank-two shape")
        if left.dtype != right.dtype:
            raise TypeError("dot-stream operands must use one storage dtype")
        if shape is None:
            shape, dtype = left.shape, left.dtype
        if left.shape != shape or left.dtype != dtype:
            raise ValueError("all dot streams must share shape and dtype")
        normalized.append((left, right))
    left = np.stack([item[0] for item in normalized], axis=1)
    right = np.stack([item[1] for item in normalized], axis=1)
    return (
        np.ascontiguousarray(left.reshape(-1, shape[1])),
        np.ascontiguousarray(right.reshape(-1, shape[1])),
    )


def deinterleave_apu_g2_streams(values) -> np.ndarray:
    """Move a trailing physical stream axis to branch-first serialization."""

    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("interleaved stream output requires rank at least two")
    return np.ascontiguousarray(np.moveaxis(values, -1, 0))


@dataclass(frozen=True)
class APUG2ComposedHostFiles:
    lhs: Path
    rhs: Path
    auxiliary: Path | None
    output: Path


def build_apu_g2_composed_host_command(
    program: APUG2ComposedContractionProgram,
    host_binary,
    files: APUG2ComposedHostFiles,
) -> tuple[str, ...]:
    """Build the exact positional command for the generic composed host."""

    if not isinstance(files, APUG2ComposedHostFiles):
        raise TypeError("files must be APUG2ComposedHostFiles")
    host_binary = Path(host_binary)
    lhs_type, rhs_type = program.input_types
    auxiliary_type = program.auxiliary_type
    if auxiliary_type is None:
        auxiliary_path = "-"
        auxiliary_bits = 1
        auxiliary_signed = 0
    else:
        if files.auxiliary is None:
            raise ValueError("non-identity program requires an auxiliary file")
        auxiliary_path = str(files.auxiliary)
        auxiliary_bits = auxiliary_type.bits
        auxiliary_signed = int(auxiliary_type.signed)
    return tuple(
        str(item)
        for item in (
            host_binary,
            files.lhs,
            files.rhs,
            auxiliary_path,
            files.output,
            lhs_type.bits,
            int(lhs_type.signed),
            rhs_type.bits,
            int(rhs_type.signed),
            program.dot_type.bits,
            auxiliary_bits,
            auxiliary_signed,
            program.output_type.bits,
            int(program.output_type.signed),
            int(math.log2(program.reduction_extent)),
            APUG2_COMPOSED_EPILOGUE_ABI[program.epilogue.mode],
            program.repetitions,
            program.layout_manifest()["fingerprint"],
        )
    )


__all__ = [
    "APUG2_COMPOSED_EPILOGUE_ABI",
    "APUG2ComposedHostFiles",
    "build_apu_g2_composed_host_command",
    "canonicalize_apu_g2_values",
    "decode_apu_g2_u16_bitpatterns",
    "deinterleave_apu_g2_streams",
    "encode_apu_g2_u16_bitpatterns",
    "expected_apu_g2_composed_output",
    "gather_apu_g2_composed_output",
    "interleave_apu_g2_dot_streams",
    "pack_apu_g2_batched_gemm_dots",
    "pack_apu_g2_composed_auxiliary",
    "pack_apu_g2_composed_operand",
    "pack_apu_g2_matrix_vector_dots",
]
