# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realize declarative APU v1 layouts as concrete GVML vector programs.

This module is deliberately independent of the Allo compiler driver.  It
consumes an :class:`~allo.pim.apu_v1_layout.APUV1Plan`, local ordered
``VectorOp`` descriptors, and value shapes; then performs four mechanical
steps shared by a future autoscheduler and a direct plan author:

* expand transfers and neutral vector opcodes into verified GVML calls;
* allocate every live tensor and temporary onto writable ``GVML_VR16_0..14``;
* emit a build-harness-compatible C fragment; and
* pack/gather NumPy arrays using the plan's exact ``ValueLayout`` coordinates.

The MICRO'25 implementation is the source of the APIs emitted here.  Bulk
contiguous paths use ``direct_dma_l4_to_l1_32k``/``gvml_load_16`` and
``gvml_store_16``/``direct_dma_l1_to_l4_32k``.  Broadcast paths use either
``gvml_lookup_16`` or ``gvml_duplicate_subgrp_16_grp_sgidx``.  Binary dot
products expand to XOR, NOT, POPCOUNT, shift/subtract, and signed group add;
FP16 contractions expand to native multiply and FP16 group add.

Unknown operations, dtypes wider than one 16-bit lane, layouts without a
``vr_lane`` output, and VR pressure above fifteen all fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import ast
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


VR_LANES = 32768
APUC_COUNT = 4
WRITABLE_VRS = 15
VMR_COUNT = 48
# Leave one quarter of the 1 MiB per-APUC L3 available for the generated
# project, allocator metadata, and other runtime state. Near-capacity lookup
# images (the 1,024,000-byte K=2000 case) build but do not complete reliably
# on the installed G1 board.
L3_LOOKUP_BUDGET_BYTES = 3 * (1 << 18)


class UnsupportedVectorOperation(ValueError):
    """Raised when a plan asks for a vector primitive this backend cannot emit."""


class VRCapacityError(ValueError):
    """Raised when overlapping vector live ranges need too many writable VRs."""


class APULayoutPackingError(ValueError):
    """Raised when a logical value cannot be represented by its physical layout."""


def _identifier(value: str, *, label: str) -> str:
    value = str(value)
    if not value.isidentifier():
        raise ValueError(f"{label} must be a C/Python identifier, got {value!r}")
    return value


def _attrs(value) -> MappingProxyType:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class VectorValue:
    """Logical NumPy value attached to an APU plan layout."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype | str | type = np.dtype("float16")
    intent: str = "in"

    def __post_init__(self):
        object.__setattr__(self, "name", _identifier(self.name, label="value name"))
        shape = tuple(int(value) for value in self.shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("vector value shapes must be non-empty and positive")
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.itemsize > 2:
            raise TypeError(
                f"APU v1 value {self.name!r} has unsupported dtype {dtype}; "
                "one VR lane stores at most 16 bits"
            )
        if self.intent not in {"in", "out", "inout", "temporary"}:
            raise ValueError(f"unsupported vector value intent {self.intent!r}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True)
class VectorOp:
    """One ordered, GVML-neutral vector operation.

    ``opcode`` uses the exact uppercase vocabulary exported in ``OPCODES``.
    Operation-specific scalar parameters live in ``attrs``.
    """

    opcode: str
    output: str | None = None
    inputs: tuple[str, ...] = ()
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        opcode = str(self.opcode).upper()
        if opcode not in OPCODES:
            raise UnsupportedVectorOperation(
                f"unsupported APU vector opcode {opcode!r}; supported: {sorted(OPCODES)}"
            )
        if self.output is not None:
            object.__setattr__(
                self, "output", _identifier(self.output, label="operation output")
            )
        inputs = tuple(
            _identifier(value, label="operation input") for value in self.inputs
        )
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "attrs", _attrs(self.attrs))


@dataclass(frozen=True)
class VRRef:
    name: str


@dataclass(frozen=True)
class PointerRef:
    name: str
    batch_offset: bool = True
    level: str = "L4"
    batch_stride: int = VR_LANES

    def __post_init__(self):
        if self.level not in {"L3", "L4"}:
            raise ValueError(f"unsupported APU pointer level {self.level!r}")
        if int(self.batch_stride) <= 0:
            raise ValueError("pointer batch_stride must be positive")


@dataclass(frozen=True)
class PointerOffsetRef:
    """A pointer plus a backend-derived element-offset expression."""

    name: str
    offset: str
    level: str = "L4"

    def __post_init__(self):
        _identifier(self.name, label="pointer name")
        if self.level not in {"L3", "L4"}:
            raise ValueError(f"unsupported APU pointer level {self.level!r}")


@dataclass(frozen=True)
class CExpression:
    """A backend-produced C expression, never supplied by workload text."""

    value: str


@dataclass(frozen=True)
class VRBankRef:
    """Select one member of a statically allocated resident VR bank."""

    names: tuple[str, ...]
    index: str

    def __post_init__(self):
        names = tuple(_identifier(name, label="resident VR") for name in self.names)
        if not names:
            raise ValueError("resident VR bank cannot be empty")
        object.__setattr__(self, "names", names)


@dataclass(frozen=True)
class GVMLInstruction:
    """One concrete API call before VR names are substituted."""

    opcode: str
    api: str
    args: tuple[object, ...]
    definitions: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    phase: str = "compute"
    predicate: str | None = None

    def render(
        self,
        bindings: Mapping[str, "VRBinding"],
        pointer_batches: Mapping[str, str] | None = None,
    ) -> str:
        pointer_batches = pointer_batches or {}

        def argument(value):
            if isinstance(value, VRRef):
                try:
                    return bindings[value.name].c_name
                except KeyError as error:
                    raise VRCapacityError(
                        f"no concrete VR for {value.name!r}"
                    ) from error
            if isinstance(value, PointerRef):
                base = f"{value.name}_{value.level}ptr"
                batch = pointer_batches.get(value.name, "batch")
                return (
                    f"({base} + ({batch}) * {int(value.batch_stride)})"
                    if value.batch_offset
                    else base
                )
            if isinstance(value, PointerOffsetRef):
                return f"({value.name}_{value.level}ptr + ({value.offset}))"
            if isinstance(value, VRBankRef):
                # Allocation need not make a long-lived bank contiguous with
                # other live indices/accumulators.  A backend-generated
                # conditional is still a scalar enum expression and avoids
                # pinning physical registers in the programming abstraction.
                selected = bindings[value.names[-1]].c_name
                for index in reversed(range(len(value.names) - 1)):
                    selected = (
                        f"(({value.index}) == {index} ? "
                        f"{bindings[value.names[index]].c_name} : {selected})"
                    )
                return selected
            if isinstance(value, CExpression):
                return value.value
            return str(value)

        if self.api == "__pio_store_16":
            pointer, vr = (argument(value) for value in self.args)
            return (
                f"for (uint32_t pio_lane = 0; pio_lane < {VR_LANES}; "
                f"++pio_lane) {pointer}[pio_lane] = "
                f"gvml_get_entry_16({vr}, pio_lane);"
            )
        call = f"{self.api}({', '.join(argument(value) for value in self.args)});"
        return f"if ({self.predicate}) {call}" if self.predicate else call


@dataclass(frozen=True)
class VRBinding:
    logical_name: str
    concrete: int
    live_range: tuple[int, int]

    @property
    def concrete_name(self) -> str:
        return f"GVML_VR16_{self.concrete}"

    @property
    def c_name(self) -> str:
        return f"vr_{self.logical_name}"


@dataclass(frozen=True)
class TransferRecord:
    value: str
    direction: str
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class PackedVector:
    """One logical value in plan-defined per-batch/per-APUC VR images."""

    value: VectorValue
    data: np.ndarray
    valid: np.ndarray


def _enum_size(size: int) -> str:
    size = int(size)
    if size <= 0 or size > VR_LANES or size & (size - 1):
        raise ValueError(f"GVML size must be a power of two in [1, 32768], got {size}")
    suffix = f"{size // 1024}K" if size >= 1024 else str(size)
    return f"GVML_P2_{suffix}"


def _tmp(prefix: str, index: int) -> str:
    return f"__{prefix}_{index}"


def fp16_contraction_ops(
    lhs: str,
    rhs: str,
    output: str,
    *,
    group_size: int,
    accumulator: str | None = None,
) -> tuple[VectorOp, ...]:
    """Expand one grouped FP16 contraction into exact neutral primitives."""

    product_name = f"__{output}_product"
    reduction_tmp = f"__{output}_reduce_tmp"
    reduced = output if accumulator is None else f"__{output}_reduced"
    operations = [
        VectorOp("MUL_F16", product_name, (lhs, rhs)),
        VectorOp("RESET_16", reduction_tmp),
        VectorOp(
            "GROUP_REDUCE_F16",
            reduced,
            (product_name, reduction_tmp),
            {"group_size": int(group_size), "subgroup_size": 1},
        ),
    ]
    if accumulator is not None:
        operations.append(VectorOp("ADD_F16", output, (accumulator, reduced)))
    return tuple(operations)


def uint16_contraction_ops(
    lhs: str,
    rhs: str,
    output: str,
    *,
    group_size: int,
    accumulator: str | None = None,
) -> tuple[VectorOp, ...]:
    """Lower a modulo-2^16 unsigned contraction to native GVML ops."""

    product_name = f"__{output}_product"
    reduced = output if accumulator is None else f"__{output}_reduced"
    reduction_tmp = f"__{output}_reduce_tmp"
    operations = [
        VectorOp("MUL_U16", product_name, (lhs, rhs)),
        VectorOp("RESET_16", reduction_tmp),
        VectorOp(
            "GROUP_REDUCE_U16",
            reduced,
            (product_name, reduction_tmp),
            {"group_size": int(group_size), "subgroup_size": 1},
        ),
    ]
    if accumulator is not None:
        operations.append(VectorOp("ADD_U16", output, (accumulator, reduced)))
    return tuple(operations)


def xnor_popcount_ops(
    lhs: str,
    rhs: str,
    output: str,
    *,
    group_size: int | None = None,
    accumulator: str | None = None,
) -> tuple[VectorOp, ...]:
    """Lower one proven packed XNOR/popcount contribution.

    ``group_size=None`` leaves one contribution per lane for temporal
    accumulation.  A group size performs the spatial reduction used by the
    baseline mapping.  The operation is target-neutral at analysis time; this
    helper only spells its concrete GVML primitive sequence.
    """

    bits = f"__{output}_binary"
    operations = [
        VectorOp("XOR_16", bits, (lhs, rhs)),
        VectorOp("NOT_16", bits, (bits,)),
        VectorOp("POPCOUNT_16", bits, (bits,)),
    ]
    contribution = bits
    if group_size is not None:
        reduction_tmp = f"__{output}_reduce_tmp"
        contribution = output if accumulator is None else f"__{output}_reduced"
        operations.extend(
            [
                VectorOp("RESET_16", reduction_tmp),
                VectorOp(
                    "GROUP_REDUCE_S16",
                    contribution,
                    (bits, reduction_tmp),
                    {"group_size": int(group_size), "subgroup_size": 1},
                ),
            ]
        )
    if accumulator is not None:
        operations.append(VectorOp("ADD_S16", output, (accumulator, contribution)))
    return tuple(operations)


def binary_matmul_ops(
    lhs: str,
    rhs: str,
    output: str,
    *,
    group_size: int,
    packed_word_bits: int = 16,
    accumulator: str | None = None,
) -> tuple[VectorOp, ...]:
    """Expand the MICRO bipolar dot ``2*popcount(XNOR)-word_bits``."""

    bits = f"__{output}_binary"
    offset = f"__{output}_offset"
    reduction_tmp = f"__{output}_reduce_tmp"
    reduced = output if accumulator is None else f"__{output}_reduced"
    operations = (
        VectorOp("XOR_16", bits, (lhs, rhs)),
        VectorOp("NOT_16", bits, (bits,)),
        VectorOp("POPCOUNT_16", bits, (bits,)),
        VectorOp("SHL_IMM_16", bits, (bits,), {"shift": 1}),
        VectorOp("CPY_IMM_16", offset, attrs={"value": int(packed_word_bits)}),
        VectorOp("SUB_S16", bits, (bits, offset)),
        VectorOp("RESET_16", reduction_tmp),
        VectorOp(
            "GROUP_REDUCE_S16",
            reduced,
            (bits, reduction_tmp),
            {"group_size": int(group_size), "subgroup_size": 1},
        ),
    )
    if accumulator is None:
        return operations
    return operations + (VectorOp("ADD_S16", output, (accumulator, reduced)),)


OPCODES = frozenset(
    {
        "RESET_16",
        "CPY_IMM_16",
        "XOR_16",
        "AND_16",
        "OR_16",
        "NOT_16",
        "POPCOUNT_16",
        "SHL_IMM_16",
        "ADD_S16",
        "ADD_U16",
        "SUB_S16",
        "SUB_U16",
        "MUL_U16",
        "MUL_F16",
        "ADD_F16",
        "GROUP_REDUCE_S16",
        "GROUP_REDUCE_U16",
        "GROUP_REDUCE_F16",
        "LOOKUP_16",
        "CREATE_GROUP_INDEX",
        "CREATE_SUBGROUP_INDEX",
        "DUPLICATE_SUBGROUP",
    }
)


@dataclass(frozen=True)
class APUVectorComputeInvocation:
    """One operand-free compute opcode and its exact emitted call count."""

    opcode: str
    count: int
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        opcode = str(self.opcode).upper()
        if opcode not in OPCODES:
            raise UnsupportedVectorOperation(
                f"unsupported emitted APU vector opcode {opcode!r}"
            )
        count = int(self.count)
        if count <= 0:
            raise ValueError("emitted APU vector invocation count must be positive")
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "attrs", _attrs(self.attrs))


@dataclass(frozen=True)
class LoopNestEmission:
    """Physical output/reduction loop nest used by the general emitter."""

    output_batches: int
    reduction_steps: int

    def __post_init__(self):
        output_batches = int(self.output_batches)
        reduction_steps = int(self.reduction_steps)
        if output_batches <= 0 or reduction_steps <= 0:
            raise ValueError("physical loop extents must be positive")
        object.__setattr__(self, "output_batches", output_batches)
        object.__setattr__(self, "reduction_steps", reduction_steps)

    @property
    def operation_call_count(self) -> int:
        return self.output_batches * self.reduction_steps


@dataclass(frozen=True)
class SpatialResidentReductionEmission:
    """Stream output tiles across a bank of resident reduction-vector VRs."""

    streamed_value: str
    resident_value: str
    output_value: str
    output_axis: str
    reduction_axis: str
    output_extent: int
    rows_per_tile: int
    matrix_tiles: int
    reduction_tile_extent: int
    reduction_tiles: int

    def __post_init__(self):
        for field_name in (
            "streamed_value",
            "resident_value",
            "output_value",
            "output_axis",
            "reduction_axis",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), label=field_name),
            )
        for field_name in (
            "output_extent",
            "rows_per_tile",
            "matrix_tiles",
            "reduction_tile_extent",
            "reduction_tiles",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.reduction_tiles > 10:
            raise UnsupportedVectorOperation(
                "spatial resident reduction needs at most ten resident vector tiles"
            )

    @property
    def resident_names(self) -> tuple[str, ...]:
        return (
            self.resident_value,
            *(
                f"{self.resident_value}__gemv_resident{index}"
                for index in range(1, self.reduction_tiles)
            ),
        )

    @property
    def operation_call_count(self) -> int:
        return self.matrix_tiles * self.reduction_tiles

    @property
    def output_reset_call_count(self) -> int:
        return self.matrix_tiles


PhysicalEmissionDescriptor = LoopNestEmission | SpatialResidentReductionEmission


def _op_instructions(op: VectorOp, index: int) -> list[GVMLInstruction]:
    output = op.output
    inputs = op.inputs

    def unary(api, *, scalar=None):
        if output is None or len(inputs) != 1:
            raise UnsupportedVectorOperation(f"{op.opcode} needs one input and output")
        args = [VRRef(output), VRRef(inputs[0])]
        if scalar is not None:
            args.append(scalar)
        return [GVMLInstruction(op.opcode, api, tuple(args), (output,), inputs)]

    def binary(api):
        if output is None or len(inputs) != 2:
            raise UnsupportedVectorOperation(f"{op.opcode} needs two inputs and output")
        return [
            GVMLInstruction(
                op.opcode,
                api,
                (VRRef(output), VRRef(inputs[0]), VRRef(inputs[1])),
                (output,),
                inputs,
            )
        ]

    if op.opcode == "RESET_16":
        if output is None or inputs:
            raise UnsupportedVectorOperation("RESET_16 needs only an output")
        return [
            GVMLInstruction(op.opcode, "gvml_reset_16", (VRRef(output),), (output,))
        ]
    if op.opcode == "CPY_IMM_16":
        if output is None or inputs or "value" not in op.attrs:
            raise UnsupportedVectorOperation(
                "CPY_IMM_16 needs output and attrs['value']"
            )
        return [
            GVMLInstruction(
                op.opcode,
                "gvml_cpy_imm_16",
                (VRRef(output), int(op.attrs["value"])),
                (output,),
            )
        ]
    if op.opcode == "XOR_16":
        return binary("gvml_xor_16")
    if op.opcode == "AND_16":
        return binary("gvml_and_16")
    if op.opcode == "OR_16":
        return binary("gvml_or_16")
    if op.opcode == "NOT_16":
        return unary("gvml_not_16")
    if op.opcode == "POPCOUNT_16":
        return unary("gvml_popcount_16")
    if op.opcode == "SHL_IMM_16":
        return unary("gvml_sl_imm_16", scalar=int(op.attrs.get("shift", 1)))
    if op.opcode == "ADD_S16":
        return binary("gvml_add_s16")
    if op.opcode == "ADD_U16":
        return binary("gvml_add_u16")
    if op.opcode == "SUB_S16":
        return binary("gvml_sub_s16")
    if op.opcode == "SUB_U16":
        return binary("gvml_sub_u16")
    if op.opcode == "MUL_U16":
        return binary("gvml_mul_u16")
    if op.opcode == "MUL_F16":
        return binary("gvml_mul_f16")
    if op.opcode == "ADD_F16":
        return binary("gvml_add_f16")
    if op.opcode in {
        "GROUP_REDUCE_S16",
        "GROUP_REDUCE_U16",
        "GROUP_REDUCE_F16",
    }:
        if output is None or len(inputs) != 2:
            raise UnsupportedVectorOperation(
                f"{op.opcode} needs (source, temporary) and output"
            )
        group = _enum_size(int(op.attrs["group_size"]))
        subgroup = _enum_size(int(op.attrs.get("subgroup_size", 1)))
        api = {
            "GROUP_REDUCE_S16": "gvml_add_subgrps_s16_grp",
            "GROUP_REDUCE_U16": "gvml_add_subgrps_u16_grp",
            "GROUP_REDUCE_F16": "gvml_add_subgrps_f16_grp",
        }[op.opcode]
        return [
            GVMLInstruction(
                op.opcode,
                api,
                (
                    VRRef(output),
                    VRRef(inputs[0]),
                    group,
                    subgroup,
                    int(op.attrs.get("destination_subgroup", 0)),
                    f"GVML_VM_{int(op.attrs.get('temporary_vmr', VMR_COUNT - 1))}",
                    VRRef(inputs[1]),
                ),
                (output,),
                inputs,
            )
        ]
    if op.opcode == "CREATE_SUBGROUP_INDEX":
        if output is None or inputs:
            raise UnsupportedVectorOperation("CREATE_SUBGROUP_INDEX needs an output")
        group = _enum_size(int(op.attrs["group_size"]))
        subgroup = _enum_size(int(op.attrs["subgroup_size"]))
        return [
            GVMLInstruction(
                op.opcode,
                "gvml_create_subgrp_index_u16",
                (VRRef(output), group, subgroup),
                (output,),
            )
        ]
    if op.opcode == "CREATE_GROUP_INDEX":
        if output is None or inputs:
            raise UnsupportedVectorOperation("CREATE_GROUP_INDEX needs an output")
        return [
            GVMLInstruction(
                op.opcode,
                "gvml_create_grp_index_u16",
                (VRRef(output), _enum_size(int(op.attrs["group_size"]))),
                (output,),
            )
        ]
    if op.opcode == "DUPLICATE_SUBGROUP":
        if output is None or len(inputs) != 2:
            raise UnsupportedVectorOperation(
                "DUPLICATE_SUBGROUP needs (source, index) and output"
            )
        return [
            GVMLInstruction(
                op.opcode,
                "gvml_duplicate_subgrp_16_grp_sgidx",
                (
                    VRRef(output),
                    VRRef(inputs[0]),
                    VRRef(inputs[1]),
                    _enum_size(int(op.attrs["group_size"])),
                    _enum_size(int(op.attrs["subgroup_size"])),
                    str(op.attrs.get("source_subgroup", 0)),
                    f"GVML_VM_{int(op.attrs.get('temporary_vmr', VMR_COUNT - 1))}",
                ),
                (output,),
                inputs,
            )
        ]
    if op.opcode == "LOOKUP_16":
        if output is None or len(inputs) != 1:
            raise UnsupportedVectorOperation("LOOKUP_16 needs index input and output")
        pointer = op.attrs.get("pointer", PointerRef(output, False, "L3"))
        if isinstance(pointer, str):
            match = re.fullmatch(r"([A-Za-z_]\w*)_L([34])ptr", pointer)
            if not match:
                raise UnsupportedVectorOperation(
                    "LOOKUP_16 pointer must be PointerRef or '<name>_L3ptr/L4ptr'"
                )
            pointer = PointerRef(match.group(1), False, f"L{match.group(2)}")
        table_size = int(op.attrs["table_size"])
        return [
            GVMLInstruction(
                op.opcode,
                "gvml_lookup_16",
                (VRRef(output), VRRef(inputs[0]), pointer, table_size),
                (output,),
                inputs,
            )
        ]
    raise UnsupportedVectorOperation(f"no lowering for opcode {op.opcode!r}")


def _direction(value: str) -> str:
    aliases = {
        "in": "in",
        "load": "in",
        "l4_to_vr": "in",
        "host_to_l4": "host",
        "out": "out",
        "store": "out",
        "vr_to_l4": "out",
        "l4_to_host": "host",
        "inout": "inout",
        "vr_to_vr": "vr_to_vr",
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise UnsupportedVectorOperation(
            f"unsupported transfer direction {value!r}"
        ) from error


_ROUTE_KINDS = frozenset(
    {
        "dma_l4_l3",
        "lookup",
        "dma_l4_l1_32k",
        "load_vr",
        "duplicate_subgroup",
        "dma_vr_l4",
        "direct",
    }
)


def _route(transfer):
    route = tuple(getattr(transfer, "route", ()) or ())
    if not route:
        return ()
    previous = None
    for step in route:
        kind = str(getattr(step, "kind", ""))
        if kind not in _ROUTE_KINDS:
            raise UnsupportedVectorOperation(
                f"unsupported explicit APU transfer-route step {kind!r}"
            )
        source = getattr(step, "source", None)
        destination = getattr(step, "destination", None)
        if source is None or destination is None:
            raise UnsupportedVectorOperation(
                f"route step {kind!r} needs explicit source and destination layouts"
            )
        if previous is not None and source != previous:
            raise UnsupportedVectorOperation(
                f"transfer route is discontinuous before {kind!r}"
            )
        previous = destination
    return route


def _step_parameters(step) -> dict[str, object]:
    return dict(getattr(step, "parameters", {}) or {})


def _route_axis_extent(step, axis: str) -> int:
    for endpoint in (getattr(step, "source", None), getattr(step, "destination", None)):
        layout = getattr(endpoint, "layout", None)
        extents = getattr(layout, "input_extents", None)
        if extents is not None and axis in extents:
            return int(extents[axis])
        bases = getattr(layout, "bases", None)
        if bases is not None and axis in bases:
            return int(layout.size_of(axis))
    raise UnsupportedVectorOperation(
        f"route does not expose extent for temporal axis {axis!r}"
    )


def _plan_vector_batches(plan) -> int:
    metadata = _metadata(plan)
    metadata_batches = int(
        metadata.get("vr_batches", metadata.get("vector_batches", 1))
    )
    temporal = int(getattr(plan, "temporal_extent", 1))
    return max(1, metadata_batches, temporal)


def _plan_output_batches(plan) -> int:
    batching = getattr(plan, "output_batching", None)
    if batching is not None:
        return max(1, int(batching.physical_output_batches))
    return max(1, int(_metadata(plan).get("output_tiles", 1)))


def _plan_reduction_steps(plan) -> int:
    batching = getattr(plan, "output_batching", None)
    if batching is not None:
        return max(1, int(batching.work_steps_per_output_batch))
    return max(
        1,
        int(
            _metadata(plan).get(
                "temporal_steps",
                _plan_vector_batches(plan) // _plan_output_batches(plan),
            )
        ),
    )


def _route_window_signature(step, field_name: str):
    return tuple(
        (str(window.axis), int(window.tile_extent), window.extent)
        for window in tuple(getattr(step, field_name, ()) or ())
    )


def _derive_spatial_resident_reduction_emission(
    plan, values: Sequence[VectorValue]
) -> SpatialResidentReductionEmission | None:
    """Recognize the physical resident-bank topology without semantic names."""

    batching = getattr(plan, "output_batching", None)
    reduction = getattr(plan, "reduction_strategy", None)
    if (
        batching is None
        or str(getattr(reduction, "kind", "")) != "group_tree"
        or str(getattr(reduction, "axis", ""))
        != str(getattr(batching, "reduction_axis", ""))
    ):
        return None
    output_values = tuple(value for value in values if value.intent in {"out", "inout"})
    if len(output_values) != 1:
        return None
    output_value = output_values[0].name
    reduction_axis = str(reduction.axis)
    transfers = tuple(getattr(plan, "transfers", ()))

    resident_candidates = []
    for transfer in transfers:
        if transfer.value == output_value or _direction(transfer.direction) not in {
            "in",
            "inout",
        }:
            continue
        route = _route(transfer)
        if tuple(step.kind for step in route) != ("dma_l4_l1_32k", "load_vr"):
            continue
        destination = str(getattr(route[-1].destination, "storage", "")).lower()
        if destination != "resident_vr":
            continue
        executed = _route_window_signature(route[0], "executed_at")
        resident = _route_window_signature(route[0], "resident_across")
        if (
            not executed
            or tuple(axis for axis, _tile, _extent in executed) != (reduction_axis,)
            or len(resident) != 1
            or any(
                _route_window_signature(step, "executed_at") != executed
                or _route_window_signature(step, "resident_across") != resident
                for step in route[1:]
            )
        ):
            continue
        resident_candidates.append((transfer, route, resident[0][0]))
    if len(resident_candidates) != 1:
        return None
    resident_transfer, resident_route, output_axis = resident_candidates[0]

    output_extents = dict(batching.axis_extents)
    if output_axis not in output_extents or any(
        int(extent) != 1
        for axis, extent in output_extents.items()
        if axis != output_axis
    ):
        return None
    group_size = int(getattr(reduction, "group_size", 0) or 0)
    if group_size <= 0 or VR_LANES % group_size:
        return None
    rows_per_tile = VR_LANES // group_size
    output_extent = int(output_extents[output_axis])
    matrix_tiles = math.ceil(output_extent / rows_per_tile)
    reduction_tiles = int(batching.reduction_tiles)
    if matrix_tiles != int(batching.work_output_tiles):
        return None
    if any(
        int(step.metrics().call_count) != reduction_tiles
        or int(step.metrics().resident_reuse_factor) != matrix_tiles
        for step in resident_route
    ):
        return None

    expected_calls = matrix_tiles * reduction_tiles
    expected_axes = set(batching.output_axes) | {reduction_axis}
    streamed_candidates = []
    for transfer in transfers:
        if transfer.value in {output_value, resident_transfer.value}:
            continue
        if _direction(transfer.direction) not in {"in", "inout"}:
            continue
        route = _route(transfer)
        if tuple(step.kind for step in route) != ("dma_l4_l1_32k", "load_vr"):
            continue
        destination = str(getattr(route[-1].destination, "storage", "")).lower()
        if destination != "compute_vr":
            continue
        if any(
            {
                axis
                for axis, _tile, _extent in _route_window_signature(step, "executed_at")
            }
            != expected_axes
            or int(step.metrics().call_count) != expected_calls
            for step in route
        ):
            continue
        streamed_candidates.append(transfer)
    if len(streamed_candidates) != 1:
        return None

    output_transfers = tuple(
        transfer for transfer in transfers if transfer.value == output_value
    )
    if (
        not any(
            _direction(transfer.direction) in {"in", "inout"}
            for transfer in output_transfers
        )
        or not any(
            _direction(transfer.direction) in {"out", "inout"}
            for transfer in output_transfers
        )
        or any(
            tuple(step.kind for step in _route(transfer)) != ("direct",)
            for transfer in output_transfers
        )
    ):
        return None

    return SpatialResidentReductionEmission(
        str(streamed_candidates[0].value),
        str(resident_transfer.value),
        output_value,
        output_axis,
        reduction_axis,
        output_extent,
        rows_per_tile,
        matrix_tiles,
        group_size,
        reduction_tiles,
    )


def _derive_physical_emission(
    plan, values: Sequence[VectorValue]
) -> PhysicalEmissionDescriptor:
    spatial = _derive_spatial_resident_reduction_emission(plan, values)
    if spatial is not None:
        return spatial
    return LoopNestEmission(_plan_output_batches(plan), _plan_reduction_steps(plan))


def _explicit_route_instructions(
    plan,
    transfer,
    transfer_index,
    current_vm,
    physical_emission: PhysicalEmissionDescriptor,
):
    """Lower one source->transit->compute relation without metadata guesses."""

    route = _route(transfer)
    if not route:
        return None
    name = str(transfer.value)
    kinds = tuple(str(step.kind) for step in route)
    ingress: list[GVMLInstruction] = []

    lookup = next((step for step in route if step.kind == "lookup"), None)
    duplicate = next(
        (step for step in route if step.kind == "duplicate_subgroup"), None
    )
    if lookup is not None and duplicate is not None:
        raise UnsupportedVectorOperation(
            f"value {name!r} cannot use lookup and subgroup duplication in one route"
        )

    if (
        isinstance(physical_emission, SpatialResidentReductionEmission)
        and name == physical_emission.resident_value
    ):
        for index, resident in enumerate(physical_emission.resident_names):
            vm = (current_vm + index) % VMR_COUNT
            ingress.extend(
                [
                    GVMLInstruction(
                        "DMA_L4_TO_L1_32K",
                        "direct_dma_l4_to_l1_32k",
                        (
                            f"GVML_VM_{vm}",
                            PointerOffsetRef(name, f"{index} * {VR_LANES}"),
                        ),
                        phase="resident_setup",
                    ),
                    GVMLInstruction(
                        "LOAD_16",
                        "gvml_load_16",
                        (VRRef(resident), f"GVML_VM_{vm}"),
                        (resident,),
                        phase="resident_setup",
                    ),
                ]
            )
        return ingress, ("dma_l4_l1", "load_resident_vr")

    if lookup is not None:
        parameters = _step_parameters(lookup)
        table_size = int(parameters["table_size"])
        group_size = int(parameters["group_size"])
        _enum_size(group_size)
        table_slots = VR_LANES // group_size
        if table_size != table_slots:
            raise UnsupportedVectorOperation(
                f"lookup table_size={table_size} must equal the {table_slots} "
                f"physical groups induced by group_size={group_size}"
            )
        temporal_axis = str(
            getattr(lookup, "temporal_axis", None)
            or getattr(transfer, "temporal_axis", None)
            or getattr(getattr(plan, "reduction_strategy", None), "axis", "")
        )
        temporal_extent = _route_axis_extent(lookup, temporal_axis)
        batching = getattr(plan, "output_batching", None)
        physical_batches = (
            max(1, int(batching.physical_output_batches))
            if batching is not None
            else max(1, int(_metadata(plan).get("physical_output_batches", 1)))
        )
        lookup_level = str(getattr(lookup.source, "storage", "")).upper()
        if lookup_level not in {"L3", "L4"}:
            raise UnsupportedVectorOperation(
                f"lookup source for {name!r} must be L3 or L4, got "
                f"{lookup_level or '<unspecified>'}"
            )
        lookup_bytes = table_size * temporal_extent * physical_batches * 2
        if lookup_level == "L3" and lookup_bytes > L3_LOOKUP_BUDGET_BYTES:
            raise UnsupportedVectorOperation(
                f"lookup image for {name!r} needs {lookup_bytes} L3 bytes, "
                f"exceeding the safe {L3_LOOKUP_BUDGET_BYTES}-byte budget"
            )
        index_name = _tmp(f"{name}_lookup_index", transfer_index)
        ingress.extend(
            _op_instructions(
                VectorOp(
                    "CREATE_GROUP_INDEX", index_name, attrs={"group_size": group_size}
                ),
                transfer_index,
            )
        )
        ingress[-1] = GVMLInstruction(
            **{**ingress[-1].__dict__, "phase": "index_setup"}
        )
        ingress.extend(
            _op_instructions(
                VectorOp(
                    "LOOKUP_16",
                    name,
                    (index_name,),
                    {
                        "pointer": PointerRef(name, True, lookup_level, table_size),
                        "table_size": table_size,
                    },
                ),
                transfer_index,
            )
        )
        ingress[-1] = GVMLInstruction(
            **{**ingress[-1].__dict__, "phase": "transfer_in"}
        )
        selected = ["create_group_index", "lookup_16"]
        if lookup_level == "L3":
            selected.insert(0, "dma_l4_l3")
        return ingress, tuple(selected)

    if duplicate is not None:
        parameters = _step_parameters(duplicate)
        rows_per_vr = int(parameters["rows_per_vr"])
        group_size = int(parameters["group_size"])
        subgroup_size = int(parameters["subgroup_size"])
        if rows_per_vr <= 0 or rows_per_vr & (rows_per_vr - 1):
            raise UnsupportedVectorOperation(
                "rows_per_vr must be a positive power of two"
            )
        if group_size != rows_per_vr * subgroup_size:
            raise UnsupportedVectorOperation(
                "duplicate_subgroup requires group_size == rows_per_vr * subgroup_size"
            )
        _enum_size(group_size)
        _enum_size(subgroup_size)
        temporal_axis = str(
            getattr(duplicate, "temporal_axis", None)
            or getattr(transfer, "temporal_axis", None)
            or getattr(getattr(plan, "reduction_strategy", None), "axis", "")
        )
        temporal_extent = _route_axis_extent(duplicate, temporal_axis)
        resident_count = math.ceil(temporal_extent / rows_per_vr)
        if resident_count <= 0:
            raise VRCapacityError(f"resident route for {name!r} is empty")
        # A compact table may be pinned before the kernel. Large reductions
        # instead reuse one streaming VR. The old lowering tried to pin all
        # 150 PolyBench GEMM chunks at once on hardware with fifteen writable
        # VRs, forcing selection to fall back to two full-VR DMAs per k-step.
        streamed = resident_count >= WRITABLE_VRS
        allocated_resident_count = 1 if streamed else resident_count
        resident_names = tuple(
            _tmp(f"{name}_resident", index) for index in range(allocated_resident_count)
        )
        for index, resident in enumerate(resident_names):
            vm = (current_vm + index) % VMR_COUNT
            if streamed:
                pointer = PointerOffsetRef(
                    name,
                    f"(reduction_step / {rows_per_vr}) * {VR_LANES}",
                )
                phase = "transfer_in"
                predicate = f"reduction_step % {rows_per_vr} == 0"
            else:
                pointer = PointerOffsetRef(name, f"{index} * {VR_LANES}")
                phase = "resident_setup"
                predicate = None
            ingress.extend(
                [
                    GVMLInstruction(
                        "DMA_L4_TO_L1_32K",
                        "direct_dma_l4_to_l1_32k",
                        (f"GVML_VM_{vm}", pointer),
                        phase=phase,
                        predicate=predicate,
                    ),
                    GVMLInstruction(
                        "LOAD_16",
                        "gvml_load_16",
                        (VRRef(resident), f"GVML_VM_{vm}"),
                        (resident,),
                        phase=phase,
                        predicate=predicate,
                    ),
                ]
            )
        index_name = _tmp(f"{name}_subgroup_index", transfer_index)
        created = _op_instructions(
            VectorOp(
                "CREATE_SUBGROUP_INDEX",
                index_name,
                attrs={"group_size": group_size, "subgroup_size": subgroup_size},
            ),
            transfer_index,
        )[0]
        ingress.append(GVMLInstruction(**{**created.__dict__, "phase": "index_setup"}))
        ingress.append(
            GVMLInstruction(
                "DUPLICATE_SUBGROUP",
                "gvml_duplicate_subgrp_16_grp_sgidx",
                (
                    VRRef(name),
                    (
                        VRRef(resident_names[0])
                        if streamed
                        else VRBankRef(
                            resident_names, f"reduction_step / {rows_per_vr}"
                        )
                    ),
                    VRRef(index_name),
                    _enum_size(group_size),
                    _enum_size(subgroup_size),
                    CExpression(f"reduction_step % {rows_per_vr}"),
                    f"GVML_VM_{(current_vm + allocated_resident_count) % VMR_COUNT}",
                ),
                (name,),
                resident_names + (index_name,),
                phase="transfer_in",
            )
        )
        return ingress, (
            "stream_l4_l1_32k" if streamed else "dma_l4_l1",
            "load_stream_vr" if streamed else "load_resident_vr",
            "create_subgroup_index",
            "duplicate_subgroup",
        )

    # Direct explicit routes retain the proven 32K path.  Partial or unknown
    # route compositions fail instead of falling back to pre-replicated data.
    if set(kinds) <= {"dma_l4_l1_32k", "load_vr"} and "dma_l4_l1_32k" in kinds:
        ingress.extend(
            [
                GVMLInstruction(
                    "DMA_L4_TO_L1_32K",
                    "direct_dma_l4_to_l1_32k",
                    (f"GVML_VM_{current_vm}", PointerRef(name)),
                    phase="transfer_in",
                ),
                GVMLInstruction(
                    "LOAD_16",
                    "gvml_load_16",
                    (VRRef(name), f"GVML_VM_{current_vm}"),
                    (name,),
                    phase="transfer_in",
                ),
            ]
        )
        return ingress, ("dma_l4_l1", "load_16")
    if kinds == ("direct",):
        source = route[0].source
        destination = route[0].destination
        if tuple(source.axes) != tuple(destination.axes) or tuple(
            source.replica_axes
        ) != tuple(destination.replica_axes):
            raise UnsupportedVectorOperation(
                f"direct route for {name!r} must expose an explicitly expanded "
                "source with the compute axes and replicas"
            )
        ingress.append(
            GVMLInstruction(
                "DIRECT_PIO_LOAD",
                "gvml_cpy_imm_subgrp_16_grp",
                (
                    VRRef(name),
                    _enum_size(VR_LANES),
                    _enum_size(1),
                    PointerRef(name),
                    VR_LANES,
                ),
                (name,),
                phase="transfer_in",
            )
        )
        return ingress, ("pio_load_expanded",)
    raise UnsupportedVectorOperation(
        f"route for {name!r} is not a complete supported ingress path: {kinds}"
    )


def _transfer_instructions(
    plan,
    value_names: set[str],
    physical_emission: PhysicalEmissionDescriptor,
):
    ingress: list[GVMLInstruction] = []
    egress: list[GVMLInstruction] = []
    records: list[TransferRecord] = []
    vm_index = 0
    for transfer_index, transfer in enumerate(tuple(getattr(plan, "transfers", ()))):
        name = str(transfer.value)
        if name not in value_names:
            raise KeyError(f"transfer references unknown value {name!r}")
        direction = _direction(transfer.direction)
        if direction == "host":
            records.append(TransferRecord(name, direction, ("host_l4",)))
            continue
        if direction == "vr_to_vr":
            raise UnsupportedVectorOperation(
                "vr_to_vr requires an explicit COPY opcode; no implicit transfer emitted"
            )
        selected = []
        current_vm = vm_index % VMR_COUNT
        vm_index += 1
        if direction in {"in", "inout"}:
            explicit = _explicit_route_instructions(
                plan,
                transfer,
                transfer_index,
                current_vm,
                physical_emission,
            )
            if explicit is not None:
                route_instructions, route_kinds = explicit
                ingress.extend(route_instructions)
                selected.extend(route_kinds)
            elif bool(getattr(transfer, "broadcast", False)):
                geometry = dict(_metadata(plan).get("broadcast_geometry", {})).get(
                    name, {}
                )
                group_size = int(geometry.get("group_size", _plan_group_size(plan)))
                subgroup_size = int(
                    geometry.get(
                        "subgroup_size", _metadata(plan).get("subgroup_size", 1)
                    )
                )
                source_subgroup = int(geometry.get("source_subgroup", 0))
                index_name = _tmp(f"{name}_index", transfer_index)
                if geometry.get("method") == "lookup":
                    ingress.extend(
                        [
                            *_op_instructions(
                                VectorOp(
                                    "CREATE_GROUP_INDEX",
                                    index_name,
                                    attrs={"group_size": group_size},
                                ),
                                transfer_index,
                            ),
                            *_op_instructions(
                                VectorOp(
                                    "LOOKUP_16",
                                    name,
                                    (index_name,),
                                    {
                                        "pointer": PointerRef(
                                            name,
                                            True,
                                            "L3",
                                            int(
                                                geometry.get(
                                                    "lookup_table_size", group_size
                                                )
                                            ),
                                        ),
                                        "table_size": int(
                                            geometry.get(
                                                "lookup_table_size", group_size
                                            )
                                        ),
                                    },
                                ),
                                transfer_index,
                            ),
                        ]
                    )
                    selected.extend(["dma_l4_to_l3", "create_group_index", "lookup_16"])
                elif geometry.get("method") == "direct_replicated":
                    ingress.extend(
                        [
                            GVMLInstruction(
                                "DMA_L4_TO_L1_32K",
                                "direct_dma_l4_to_l1_32k",
                                (f"GVML_VM_{current_vm}", PointerRef(name)),
                                phase="transfer_in",
                            ),
                            GVMLInstruction(
                                "LOAD_16",
                                "gvml_load_16",
                                (VRRef(name), f"GVML_VM_{current_vm}"),
                                (name,),
                                phase="transfer_in",
                            ),
                        ]
                    )
                    selected.extend(["dma_l4_to_l1", "load_16", "pre_replicated"])
                elif bool(getattr(transfer, "coalesced", False)):
                    reuse_name = _tmp(f"{name}_reuse", transfer_index)
                    ingress.extend(
                        [
                            GVMLInstruction(
                                "DMA_L4_TO_L1_32K",
                                "direct_dma_l4_to_l1_32k",
                                (f"GVML_VM_{current_vm}", PointerRef(name)),
                                phase="transfer_in",
                            ),
                            GVMLInstruction(
                                "LOAD_16",
                                "gvml_load_16",
                                (VRRef(reuse_name), f"GVML_VM_{current_vm}"),
                                (reuse_name,),
                                phase="transfer_in",
                            ),
                            *_op_instructions(
                                VectorOp(
                                    "CREATE_SUBGROUP_INDEX",
                                    index_name,
                                    attrs={
                                        "group_size": group_size,
                                        "subgroup_size": subgroup_size,
                                        "source_subgroup": source_subgroup,
                                    },
                                ),
                                transfer_index,
                            ),
                            *_op_instructions(
                                VectorOp(
                                    "DUPLICATE_SUBGROUP",
                                    name,
                                    (reuse_name, index_name),
                                    {
                                        "group_size": group_size,
                                        "subgroup_size": subgroup_size,
                                        "source_subgroup": source_subgroup,
                                    },
                                ),
                                transfer_index,
                            ),
                        ]
                    )
                    selected.extend(["dma_l4_to_l1", "load_16", "duplicate_subgroup"])
                else:
                    ingress.extend(
                        [
                            *_op_instructions(
                                VectorOp(
                                    "CREATE_SUBGROUP_INDEX",
                                    index_name,
                                    attrs={
                                        "group_size": group_size,
                                        "subgroup_size": subgroup_size,
                                    },
                                ),
                                transfer_index,
                            ),
                            *_op_instructions(
                                VectorOp(
                                    "LOOKUP_16",
                                    name,
                                    (index_name,),
                                    {
                                        "pointer": PointerRef(name, False, "L3"),
                                        "table_size": int(
                                            _metadata(plan).get(
                                                f"{name}_lookup_table_size", group_size
                                            )
                                        ),
                                    },
                                ),
                                transfer_index,
                            ),
                        ]
                    )
                    selected.extend(["create_subgroup_index", "lookup_16"])
            elif bool(getattr(transfer, "coalesced", False)):
                ingress.extend(
                    [
                        GVMLInstruction(
                            "DMA_L4_TO_L1_32K",
                            "direct_dma_l4_to_l1_32k",
                            (f"GVML_VM_{current_vm}", PointerRef(name)),
                            phase="transfer_in",
                        ),
                        GVMLInstruction(
                            "LOAD_16",
                            "gvml_load_16",
                            (VRRef(name), f"GVML_VM_{current_vm}"),
                            (name,),
                            phase="transfer_in",
                        ),
                    ]
                )
                selected.extend(["dma_l4_to_l1", "load_16"])
            else:
                # Fine-grained PIO/irregular reads use indexed lookup.  An
                # explicit index vector keeps this path deterministic.
                index_name = _tmp(f"{name}_index", transfer_index)
                ingress.extend(
                    [
                        *_op_instructions(
                            VectorOp(
                                "CREATE_SUBGROUP_INDEX",
                                index_name,
                                attrs={"group_size": VR_LANES, "subgroup_size": 1},
                            ),
                            transfer_index,
                        ),
                        *_op_instructions(
                            VectorOp(
                                "LOOKUP_16",
                                name,
                                (index_name,),
                                {
                                    "pointer": PointerRef(name, False, "L3"),
                                    "table_size": int(
                                        _metadata(plan).get(
                                            f"{name}_lookup_table_size", VR_LANES
                                        )
                                    ),
                                },
                            ),
                            transfer_index,
                        ),
                    ]
                )
                selected.extend(["pio_lookup", "lookup_16"])
        if direction in {"out", "inout"}:
            if bool(getattr(transfer, "coalesced", False)):
                egress.extend(
                    [
                        GVMLInstruction(
                            "STORE_16",
                            "gvml_store_16",
                            (f"GVML_VM_{current_vm}", VRRef(name)),
                            uses=(name,),
                            phase="transfer_out",
                        ),
                        GVMLInstruction(
                            "DMA_L1_TO_L4_32K",
                            "direct_dma_l1_to_l4_32k",
                            (PointerRef(name), f"GVML_VM_{current_vm}"),
                            phase="transfer_out",
                        ),
                    ]
                )
                selected.extend(["store_16", "dma_l1_to_l4"])
            else:
                # The ARC PIO path reads individual lanes.  Source uses the
                # same layout-driven packing order as the NumPy gather ABI.
                egress.append(
                    GVMLInstruction(
                        "PIO_STORE",
                        "__pio_store_16",
                        (PointerRef(name), VRRef(name)),
                        uses=(name,),
                        phase="transfer_out",
                    )
                )
                selected.append("pio_store")
        records.append(TransferRecord(name, direction, tuple(selected)))
    return ingress, egress, tuple(records)


def _metadata(plan) -> Mapping[str, object]:
    return dict(getattr(plan, "metadata", {}) or {})


def _plan_group_size(plan) -> int:
    reduction = getattr(plan, "reduction_strategy", None)
    size = getattr(reduction, "group_size", None)
    if size is None:
        size = _metadata(plan).get("group_size", VR_LANES)
    _enum_size(int(size))
    return int(size)


def _normalize_values(plan, values) -> tuple[VectorValue, ...]:
    if values is not None:
        source = values.values() if isinstance(values, Mapping) else values
        normalized = tuple(
            value if isinstance(value, VectorValue) else VectorValue(**dict(value))
            for value in source
        )
    else:
        metadata = _metadata(plan)
        dtypes = dict(metadata.get("dtypes", {}))
        intents = dict(metadata.get("intents", {}))
        normalized_list = []
        for value_layout in tuple(getattr(plan, "value_layouts", ())):
            if getattr(value_layout, "validity", None) is not None:
                shape = tuple(value_layout.validity.logical_shape)
            else:
                layout = value_layout.effective_layout(plan.iteration_layout)
                extents = getattr(layout, "input_extents", None)
                if extents is None:
                    extents = {name: layout.size_of(name) for name in layout.bases}
                shape = tuple(int(extents[axis]) for axis in value_layout.axes)
            normalized_list.append(
                VectorValue(
                    value_layout.value,
                    shape,
                    dtypes.get(value_layout.value, "float16"),
                    intents.get(
                        value_layout.value,
                        _intent_from_transfers(plan, value_layout.value),
                    ),
                )
            )
        normalized = tuple(normalized_list)
    names = [value.name for value in normalized]
    if len(names) != len(set(names)):
        raise ValueError("vector values must have unique names")
    plan_names = {value.value for value in tuple(getattr(plan, "value_layouts", ()))}
    if set(names) != plan_names:
        raise ValueError(
            f"value descriptors {sorted(names)} do not match plan layouts {sorted(plan_names)}"
        )
    return normalized


def _intent_from_transfers(plan, value: str) -> str:
    directions = {
        _direction(transfer.direction)
        for transfer in tuple(getattr(plan, "transfers", ()))
        if transfer.value == value
    }
    if "inout" in directions or {"in", "out"} <= directions:
        return "inout"
    if "out" in directions:
        return "out"
    return "in"


def _normalize_operations(plan, operations) -> tuple[VectorOp, ...]:
    source = operations
    if source is None:
        source = _metadata(plan).get("operation_descriptors")
    if source is None:
        source = tuple(getattr(plan, "operations", ()))
    normalized = []
    for operation in tuple(source or ()):
        if isinstance(operation, VectorOp):
            normalized.append(operation)
        elif isinstance(operation, Mapping):
            normalized.append(VectorOp(**dict(operation)))
        elif all(hasattr(operation, name) for name in ("opcode", "inputs")):
            normalized.append(
                VectorOp(
                    operation.opcode,
                    getattr(operation, "output", None),
                    tuple(operation.inputs),
                    getattr(operation, "attrs", {}),
                )
            )
        else:
            raise UnsupportedVectorOperation(
                "plan operations contain counts but no operands; pass ordered "
                "VectorOp descriptors to realize_apu_v1_plan"
            )
    return tuple(normalized)


def _allocate_vrs(plan, instructions, *, capacity: int) -> tuple[VRBinding, ...]:
    if not 1 <= int(capacity) <= WRITABLE_VRS:
        raise ValueError(f"VR capacity must be in [1, {WRITABLE_VRS}]")
    positions: dict[str, list[int]] = {}
    persistent = set()
    for position, instruction in enumerate(instructions):
        for name in instruction.definitions + instruction.uses:
            positions.setdefault(name, []).append(position)
        if instruction.phase in {"resident_setup", "index_setup"} or (
            instruction.phase == "transfer_in"
            and instruction.opcode == "LOAD_16"
            and instruction.predicate is not None
        ):
            persistent.update(instruction.definitions)
    # These definitions are emitted before the output loop after phase-aware
    # code motion.  Their textual position in the neutral instruction list is
    # therefore not their liveness boundary: resident data and reusable index
    # vectors remain live through every output batch.
    for name in persistent:
        positions[name].extend((0, len(instructions)))
    intervals = {
        name: (min(points), max(points) + 1) for name, points in positions.items()
    }

    fixed = {}
    for allocation in tuple(getattr(plan, "vr_allocations", ())):
        name = str(allocation.abstract)
        if name not in intervals:
            continue
        requested = tuple(int(value) for value in allocation.live_range)
        actual = intervals[name]
        if requested[0] > actual[0] or requested[1] < actual[1]:
            raise VRCapacityError(
                f"fixed live range {requested} for {name!r} does not cover {actual}"
            )
        fixed[name] = (int(allocation.concrete), requested)

    bindings: dict[str, VRBinding] = {}
    active: list[VRBinding] = []
    for name, (start, stop) in sorted(
        intervals.items(), key=lambda item: (item[1][0], item[0])
    ):
        active = [binding for binding in active if binding.live_range[1] > start]
        occupied = {binding.concrete for binding in active}
        if name in fixed:
            concrete, live_range = fixed[name]
            if concrete >= capacity or concrete in occupied:
                raise VRCapacityError(
                    f"fixed {name!r}=GVML_VR16_{concrete} conflicts at instruction {start}"
                )
            binding = VRBinding(name, concrete, live_range)
        else:
            available = next(
                (index for index in range(capacity) if index not in occupied), None
            )
            if available is None:
                live = ", ".join(
                    f"{binding.logical_name}:VR{binding.concrete}{binding.live_range}"
                    for binding in active
                )
                raise VRCapacityError(
                    f"APU vector plan needs more than {capacity} writable VRs at "
                    f"instruction {start} while allocating {name!r}; live: {live}"
                )
            binding = VRBinding(name, available, (start, stop))
        bindings[name] = binding
        active.append(binding)
    return tuple(
        sorted(
            bindings.values(), key=lambda value: (value.concrete, value.logical_name)
        )
    )


def _reserve_accumulator_block(
    plan,
    bindings: tuple[VRBinding, ...],
    output_names: set[str],
    *,
    capacity: int,
) -> tuple[VRBinding, ...]:
    """Reserve distinct physical VRs for simultaneously live accumulators."""

    block = int(getattr(plan, "accumulator_block", 1))
    if block == 1:
        return bindings
    if len(output_names) != 1:
        raise UnsupportedVectorOperation(
            "accumulator blocking currently requires exactly one carried output"
        )
    output = next(iter(output_names))
    if not any(binding.logical_name == output for binding in bindings):
        raise VRCapacityError(f"carried output {output!r} has no allocated VR")
    occupied = {binding.concrete for binding in bindings}
    available = [index for index in range(capacity) if index not in occupied]
    if len(available) < block - 1:
        raise VRCapacityError(
            f"accumulator block {block} needs {block - 1} additional writable VRs; "
            f"only {len(available)} remain after allocating the vector program"
        )
    lifetime = (0, max((binding.live_range[1] for binding in bindings), default=1))
    extras = tuple(
        VRBinding(f"{output}__acc{slot}", available[slot - 1], lifetime)
        for slot in range(1, block)
    )
    return tuple(
        sorted(
            bindings + extras,
            key=lambda value: (value.concrete, value.logical_name),
        )
    )


class APUVectorABI:
    """Layout-driven NumPy pack/gather ABI for one realized plan."""

    def __init__(
        self,
        plan,
        values: Sequence[VectorValue],
        physical_emission: PhysicalEmissionDescriptor | None = None,
    ):
        self.plan = plan
        self.values = tuple(values)
        self.physical_emission = physical_emission or _derive_physical_emission(
            plan, self.values
        )
        self._by_name = {value.name: value for value in self.values}

    def _value_layout(self, name):
        if hasattr(self.plan, "value_layout"):
            return self.plan.value_layout(name)
        for layout in self.plan.value_layouts:
            if layout.value == name:
                return layout
        raise KeyError(name)

    def _layout_extents(self, value_layout):
        layout = value_layout.effective_layout(self.plan.iteration_layout)
        extents = getattr(layout, "input_extents", None)
        if extents is not None:
            return dict(extents)
        return {name: layout.size_of(name) for name in layout.bases}

    def _coordinates(self, value: VectorValue, logical_index):
        value_layout = self._value_layout(value.name)
        extents = self._layout_extents(value_layout)
        if len(logical_index) != len(value_layout.axes):
            raise APULayoutPackingError(
                f"{value.name}: rank {len(logical_index)} does not match axes {value_layout.axes}"
            )
        base = {name: 0 for name in extents}
        base.update(zip(value_layout.axes, logical_index))
        replica_ranges = [range(extents[name]) for name in value_layout.replica_axes]
        replicas = product(*replica_ranges) if replica_ranges else [()]
        result = []
        for replica in replicas:
            indices = dict(base)
            indices.update(zip(value_layout.replica_axes, replica))
            coordinate = value_layout.coordinate(self.plan.iteration_layout, **indices)
            lane = int(coordinate.get("vr_lane", -1))
            apuc = int(coordinate.get("apuc", 0))
            batch = int(coordinate.get("vr_batch", coordinate.get("batch", 0)))
            if not 0 <= lane < VR_LANES or not 0 <= apuc < APUC_COUNT or batch < 0:
                raise APULayoutPackingError(
                    f"{value.name}{logical_index} maps outside APU storage: {coordinate}"
                )
            result.append((batch, apuc, lane))
        return tuple(result)

    def _is_dense_batch_major(self, value: VectorValue) -> bool:
        """Prove C-order logical storage equals consecutive APUC-0 VR blocks."""

        value_layout = self._value_layout(value.name)
        if value_layout.replica_axes:
            return False
        zero = (0,) * len(value.shape)
        if self._coordinates(value, zero) != ((0, 0, 0),):
            return False
        strides = []
        trailing = 1
        for extent in reversed(value.shape):
            strides.append(trailing)
            trailing *= extent
        strides.reverse()
        for axis, extent in enumerate(value.shape):
            bit = 1
            while bit < extent:
                index = [0] * len(value.shape)
                index[axis] = bit
                linear = bit * strides[axis]
                expected = ((linear // VR_LANES, 0, linear % VR_LANES),)
                if self._coordinates(value, tuple(index)) != expected:
                    return False
                bit <<= 1
        return True

    @staticmethod
    def _bits(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
        value = np.asarray(array, dtype=dtype, order="C")
        if dtype == np.dtype(bool):
            return value.astype(np.uint16)
        if dtype.itemsize == 1:
            return value.astype(np.int16 if dtype.kind == "i" else np.uint16).view(
                np.uint16
            )
        return np.ascontiguousarray(value).view(np.uint16)

    @staticmethod
    def _from_bits(bits: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if dtype == np.dtype(bool):
            return bits != 0
        if dtype.itemsize == 1:
            return bits.astype(dtype)
        return np.ascontiguousarray(bits, dtype=np.uint16).view(dtype)

    def pack(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        values: Sequence[str] | None = None,
    ) -> dict[str, PackedVector]:
        packed = {}
        selected = None if values is None else set(values)
        for value in self.values:
            if selected is not None and value.name not in selected:
                continue
            if value.intent == "out" and value.name not in arrays:
                array = np.zeros(value.shape, dtype=value.dtype)
            else:
                if value.name not in arrays:
                    raise KeyError(f"missing APU vector value {value.name!r}")
                array = np.asarray(arrays[value.name], dtype=value.dtype)
            if array.shape != value.shape:
                raise ValueError(
                    f"{value.name}: got shape {array.shape}, expected {value.shape}"
                )
            bits = self._bits(array, value.dtype)
            if self._is_dense_batch_major(value):
                elements = bits.size
                batches = math.ceil(elements / VR_LANES)
                data = np.zeros((batches, APUC_COUNT, VR_LANES), dtype=np.uint16)
                valid = np.zeros_like(data, dtype=bool)
                flat = bits.reshape(-1)
                data[:, 0, :].reshape(-1)[:elements] = flat
                valid[:, 0, :].reshape(-1)[:elements] = True
                packed[value.name] = PackedVector(value, data, valid)
                continue
            mappings = {
                index: self._coordinates(value, index)
                for index in np.ndindex(value.shape)
            }
            batches = 1 + max(
                (
                    batch
                    for coords in mappings.values()
                    for batch, _apuc, _lane in coords
                ),
                default=0,
            )
            data = np.zeros((batches, APUC_COUNT, VR_LANES), dtype=np.uint16)
            valid = np.zeros_like(data, dtype=bool)
            owners = {}
            for index, coordinates in mappings.items():
                bit_value = int(bits[index])
                for coordinate in coordinates:
                    previous = owners.get(coordinate)
                    if previous is not None and previous != index:
                        raise APULayoutPackingError(
                            f"{value.name}: logical indices {previous} and {index} "
                            f"collide at physical {coordinate}"
                        )
                    owners[coordinate] = index
                    data[coordinate] = bit_value
                    valid[coordinate] = True
            packed[value.name] = PackedVector(value, data, valid)
        return packed

    def _transfer(self, name: str):
        matches = [
            transfer
            for transfer in tuple(getattr(self.plan, "transfers", ()))
            if transfer.value == name
            and _direction(transfer.direction) in {"in", "inout"}
        ]
        if len(matches) != 1:
            raise APULayoutPackingError(
                f"{name}: expected exactly one ingress transfer, found {len(matches)}"
            )
        return matches[0]

    def _logical_bits(self, value: VectorValue, arrays):
        if value.name not in arrays:
            raise KeyError(f"missing APU vector value {value.name!r}")
        array = np.asarray(arrays[value.name], dtype=value.dtype)
        if array.shape != value.shape:
            raise ValueError(
                f"{value.name}: got shape {array.shape}, expected {value.shape}"
            )
        return self._bits(array, value.dtype)

    def _pack_lookup_route(self, value, bits, transfer, step):
        parameters = _step_parameters(step)
        table_size = int(parameters["table_size"])
        temporal_axis = str(
            getattr(step, "temporal_axis", None)
            or getattr(transfer, "temporal_axis", None)
            or getattr(getattr(self.plan, "reduction_strategy", None), "axis", "")
        )
        value_layout = self._value_layout(value.name)
        if temporal_axis not in value_layout.axes:
            raise APULayoutPackingError(
                f"{value.name}: lookup temporal axis {temporal_axis!r} is not logical"
            )
        table_axes = [axis for axis in value_layout.axes if axis != temporal_axis]
        if len(table_axes) != 1:
            raise APULayoutPackingError(
                f"{value.name}: compact lookup currently requires one table axis, "
                f"got {table_axes}"
            )
        table_axis = table_axes[0]
        axis_positions = {axis: index for index, axis in enumerate(value_layout.axes)}
        ordered = np.transpose(
            bits, (axis_positions[table_axis], axis_positions[temporal_axis])
        )
        table_extent, temporal_extent = ordered.shape
        batches = math.ceil(table_extent / table_size)
        packed = np.zeros((batches, temporal_extent, table_size), dtype=np.uint16)
        for batch in range(batches):
            begin = batch * table_size
            end = min(begin + table_size, table_extent)
            packed[batch, :, : end - begin] = ordered[begin:end, :].T
        batching = getattr(self.plan, "output_batching", None)
        if (
            batching is not None
            and table_axis in batching.output_axes
            and batching.physical_output_batches > batches
        ):
            # A physical output stream may tile another output axis (for
            # example the columns of an outer product) more often than it
            # advances this lookup table axis.  Device code indexes compact
            # lookup data by physical output batch, so materialize the exact
            # repeated table slice for every batch instead of reading beyond
            # the shorter logical table image.
            axis_index = batching.output_axes.index(table_axis)
            expanded = np.zeros(
                (batching.physical_output_batches, temporal_extent, table_size),
                dtype=np.uint16,
            )
            for physical_batch in range(batching.physical_output_batches):
                placements = [
                    placement
                    for placement in batching.placements
                    if placement.physical_output_batch == physical_batch
                ]
                if not placements:
                    continue
                begin = min(
                    placement.logical_origin[axis_index] for placement in placements
                )
                end = min(begin + table_size, table_extent)
                expanded[physical_batch, :, : end - begin] = ordered[begin:end, :].T
            packed = expanded
        flattened = packed.reshape(-1)
        # L4->L3 uses 512-byte transactions.  Padding is physical transfer
        # validity, not host-side replication of the compute layout.
        alignment = 512 // np.dtype(np.uint16).itemsize
        if flattened.size % alignment:
            flattened = np.pad(flattened, (0, alignment - flattened.size % alignment))
        return np.ascontiguousarray(flattened, dtype=np.uint16)

    def _pack_resident_route(self, value, bits, transfer, step):
        parameters = _step_parameters(step)
        rows_per_vr = int(parameters["rows_per_vr"])
        subgroup_size = int(parameters["subgroup_size"])
        group_size = int(parameters["group_size"])
        temporal_axis = str(
            getattr(step, "temporal_axis", None)
            or getattr(transfer, "temporal_axis", None)
            or getattr(getattr(self.plan, "reduction_strategy", None), "axis", "")
        )
        value_layout = self._value_layout(value.name)
        payload_axes = [axis for axis in value_layout.axes if axis != temporal_axis]
        if temporal_axis not in value_layout.axes or len(payload_axes) != 1:
            raise APULayoutPackingError(
                f"{value.name}: resident subgroup route needs temporal plus one payload axis"
            )
        axis_positions = {axis: index for index, axis in enumerate(value_layout.axes)}
        ordered = np.transpose(
            bits, (axis_positions[temporal_axis], axis_positions[payload_axes[0]])
        )
        temporal_extent, payload_extent = ordered.shape
        if payload_extent > subgroup_size or group_size != rows_per_vr * subgroup_size:
            raise APULayoutPackingError(
                f"{value.name}: route geometry ({rows_per_vr}x{subgroup_size}) "
                f"cannot contain logical payload {payload_extent}"
            )
        if VR_LANES % group_size:
            raise APULayoutPackingError(
                f"{value.name}: resident group size {group_size} does not tile a VR"
            )
        banks = math.ceil(temporal_extent / rows_per_vr)
        packed = np.zeros((banks, VR_LANES), dtype=np.uint16)
        replicas = VR_LANES // group_size
        for bank in range(banks):
            begin = bank * rows_per_vr
            end = min(begin + rows_per_vr, temporal_extent)
            group = np.zeros((rows_per_vr, subgroup_size), dtype=np.uint16)
            group[: end - begin, :payload_extent] = ordered[begin:end, :]
            packed[bank] = np.tile(group.reshape(-1), replicas)
        return packed.reshape(-1)

    def _pack_expanded_tiles(self, value, bits, transfer):
        """Materialize the declared expanded layout with zero-filled padding.

        Logical tile extents need not divide the problem shape.  Addressing is
        taken directly from the immutable destination layout, so masked edge
        lanes remain zero and F2 batch order never relies on a row-major guess.
        """

        route = _route(transfer)
        resident_emission = (
            self.physical_emission
            if isinstance(self.physical_emission, SpatialResidentReductionEmission)
            and transfer.value == self.physical_emission.resident_value
            else None
        )
        layout = (
            route[0].source if resident_emission else route[-1].destination
        ).layout
        if resident_emission:
            reduction_tile = resident_emission.reduction_tile_extent
            if VR_LANES % reduction_tile:
                raise APULayoutPackingError(
                    f"{value.name}: GEMV reduction tile must divide one VR"
                )
            reduction_axis = resident_emission.reduction_axis
            value_layout = self._value_layout(value.name)
            reduction_position = value_layout.axes.index(reduction_axis)
            ordered = np.moveaxis(bits, reduction_position, 0)
            if math.prod(ordered.shape[1:]) != 1:
                raise APULayoutPackingError(
                    f"{value.name}: spatial GEMV vector payload must be rank one"
                )
            vector = ordered.reshape(ordered.shape[0])
            resident_count = math.ceil(vector.size / reduction_tile)
            if resident_count != resident_emission.reduction_tiles:
                raise APULayoutPackingError(
                    f"{value.name}: resident bank count disagrees with physical emission"
                )
            packed = np.zeros((resident_count, VR_LANES), dtype=np.uint16)
            row_groups = VR_LANES // reduction_tile
            for bank in range(resident_count):
                begin = bank * reduction_tile
                end = min(begin + reduction_tile, vector.size)
                group = np.zeros(reduction_tile, dtype=np.uint16)
                group[: end - begin] = vector[begin:end]
                packed[bank] = np.tile(group, row_groups)
            return packed.reshape(-1)
        metadata = _metadata(self.plan)
        roles = dict(metadata.get("loop_roles", {}))
        reduction_axes = [axis for axis, role in roles.items() if role == "reduction"]
        output_axes = [
            axis for axis, role in roles.items() if role == "parallel_output"
        ]
        iteration_axes = tuple(output_axes + reduction_axes)
        extents = dict(metadata.get("axis_extents", {}))
        tiles = dict(metadata.get("tile_sizes", {}))
        padded_full = getattr(layout, "padded_extents", None)
        padded_tiles = {
            axis: 1 << (int(tiles[axis]) - 1).bit_length()
            for axis in iteration_axes
            if axis in tiles
        }
        can_vectorize = (
            len(output_axes) == 2
            and len(reduction_axes) == 1
            and not (set(iteration_axes) - set(extents))
            and not (set(iteration_axes) - set(padded_tiles))
            and padded_full is not None
            and tuple(layout.out_dims) == ("vr_lane", "vr_batch")
            and all(axis in padded_full for axis in iteration_axes)
            and math.prod(padded_tiles.values()) == int(layout.out_sizes[0])
            and math.prod(
                int(padded_full[axis]) // padded_tiles[axis] for axis in iteration_axes
            )
            == int(layout.out_sizes[1])
        )
        if can_vectorize:
            # Materialize the F2 carrier in outer-tile/inner-lane order.  This
            # retains the fully vectorized 1K path while making masked SMALL
            # edges exact: logical values occupy only the leading hypercube;
            # all carrier padding remains zero.
            value_layout = self._value_layout(value.name)
            logical_axes = tuple(value_layout.axes)
            ordered_logical = tuple(
                axis for axis in iteration_axes if axis in logical_axes
            )
            permutation = tuple(logical_axes.index(axis) for axis in ordered_logical)
            ordered = np.transpose(bits, permutation) if permutation else bits
            logical_shape = tuple(int(extents[axis]) for axis in iteration_axes)
            padded_shape = tuple(int(padded_full[axis]) for axis in iteration_axes)
            tile_shape = tuple(padded_tiles[axis] for axis in iteration_axes)
            broadcast_shape = tuple(
                extents[axis] if axis in logical_axes else 1 for axis in iteration_axes
            )
            expanded = np.broadcast_to(ordered.reshape(broadcast_shape), logical_shape)
            carrier = np.zeros(padded_shape, dtype=np.uint16)
            carrier[tuple(slice(0, extent) for extent in logical_shape)] = expanded
            split_shape = []
            for extent, tile in zip(padded_shape, tile_shape):
                split_shape.extend((extent // tile, tile))
            outer = tuple(range(0, 2 * len(iteration_axes), 2))
            inner = tuple(range(1, 2 * len(iteration_axes), 2))
            outer_shape = tuple(
                extent // tile for extent, tile in zip(padded_shape, tile_shape)
            )
            blocks = np.transpose(carrier.reshape(split_shape), outer + inner).reshape(
                *outer_shape, math.prod(tile_shape)
            )
            valid_outer = tuple(
                slice(0, (logical + tile - 1) // tile)
                for logical, tile in zip(logical_shape, tile_shape)
            )
            # Remove padded outer coordinates from the execution stream while
            # retaining zero-filled lanes inside each edge block. This keeps
            # dense work IDs aligned with source pointer batches for arbitrary
            # non-power-of-two output and reduction extents.
            blocks = blocks[valid_outer].reshape(-1, math.prod(tile_shape))
            packed = np.zeros((blocks.shape[0], VR_LANES), dtype=np.uint16)
            packed[:, : blocks.shape[1]] = blocks
            return packed.reshape(-1)

        layout_extents = getattr(layout, "input_extents", None)
        if layout_extents is None:
            layout_extents = {axis: layout.size_of(axis) for axis in layout.bases}
        layout_extents = dict(layout_extents)
        value_layout = self._value_layout(value.name)
        if set(value_layout.axes + value_layout.replica_axes) != set(layout_extents):
            raise APULayoutPackingError(
                f"{value.name}: expanded route axes disagree with value layout"
            )
        out_dims = tuple(layout.out_dims)
        out_sizes = tuple(layout.out_sizes)
        try:
            lane_position = out_dims.index("vr_lane")
        except ValueError as error:
            raise APULayoutPackingError(
                f"{value.name}: expanded destination has no vr_lane dimension"
            ) from error
        batch_position = out_dims.index("vr_batch") if "vr_batch" in out_dims else None
        apuc_position = out_dims.index("apuc") if "apuc" in out_dims else None
        batches = out_sizes[batch_position] if batch_position is not None else 1
        packed = np.zeros((batches, APUC_COUNT, VR_LANES), dtype=np.uint16)
        owners = {}
        replica_ranges = [
            range(layout_extents[axis]) for axis in value_layout.replica_axes
        ]
        for logical_index in np.ndindex(value.shape):
            logical = dict(zip(value_layout.axes, logical_index))
            replicas = product(*replica_ranges) if replica_ranges else [()]
            for replica_index in replicas:
                indices = dict(logical)
                indices.update(zip(value_layout.replica_axes, replica_index))
                if hasattr(layout, "coordinate"):
                    raw = layout.coordinate(**indices)
                    coordinate = tuple(raw[axis] for axis in out_dims)
                else:
                    coordinate = tuple(layout.apply(**indices))
                lane = coordinate[lane_position]
                batch = coordinate[batch_position] if batch_position is not None else 0
                apuc = coordinate[apuc_position] if apuc_position is not None else 0
                physical = (batch, apuc, lane)
                previous = owners.get(physical)
                if previous is not None and previous != logical_index:
                    raise APULayoutPackingError(
                        f"{value.name}: expanded logical indices {previous} and "
                        f"{logical_index} collide at {physical}"
                    )
                owners[physical] = logical_index
                packed[physical] = int(bits[logical_index])
        return packed[:, 0, :].reshape(-1)

    def transfer_input_images(
        self, arrays: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Pack host inputs in the source/transit layouts of explicit routes."""

        compute_images = None
        result = {}
        for value in self.values:
            # Inout/output storage is allocated by the runtime from the
            # canonical compute layout.  This method owns only host input
            # representations selected by ingress routes.
            if value.intent != "in":
                continue
            bits = self._logical_bits(value, arrays)
            transfer = self._transfer(value.name)
            route = _route(transfer)
            lookup = next((step for step in route if step.kind == "lookup"), None)
            duplicate = next(
                (step for step in route if step.kind == "duplicate_subgroup"), None
            )
            if lookup is not None:
                result[value.name] = self._pack_lookup_route(
                    value, bits, transfer, lookup
                )
            elif duplicate is not None:
                result[value.name] = self._pack_resident_route(
                    value, bits, transfer, duplicate
                )
            elif route and tuple(route[0].source.replica_axes) == tuple(
                route[-1].destination.replica_axes
            ):
                result[value.name] = self._pack_expanded_tiles(value, bits, transfer)
            else:
                if compute_images is None:
                    compute_images = self.pack(arrays)
                result[value.name] = np.ascontiguousarray(
                    compute_images[value.name].data[:, 0, :], dtype=np.uint16
                ).reshape(-1)
        return result

    def gather(
        self,
        images: Mapping[str, PackedVector | np.ndarray],
        *,
        outputs: Sequence[str] | None = None,
        check_replicas: bool = True,
    ) -> dict[str, np.ndarray]:
        selected = set(
            outputs
            or [value.name for value in self.values if value.intent in {"out", "inout"}]
        )
        result = {}
        for name in selected:
            value = self._by_name[name]
            image = images[name]
            data = image.data if isinstance(image, PackedVector) else np.asarray(image)
            if self._is_dense_batch_major(value):
                flat = np.asarray(data)[:, 0, :].reshape(-1)[: math.prod(value.shape)]
                result[name] = self._from_bits(flat, value.dtype).reshape(value.shape)
                continue
            bits = np.zeros(value.shape, dtype=np.uint16)
            for index in np.ndindex(value.shape):
                coordinates = self._coordinates(value, index)
                observed = [int(data[coordinate]) for coordinate in coordinates]
                if check_replicas and any(item != observed[0] for item in observed[1:]):
                    raise APULayoutPackingError(
                        f"{name}{index}: replica values disagree: {observed}"
                    )
                bits[index] = observed[0]
            result[name] = self._from_bits(bits, value.dtype).reshape(value.shape)
        return result


@dataclass(frozen=True)
class SchedulePrologue:
    """The device translation unit's head, rendered from its fields: the
    includes, ``static int <function>_vector(<pointer params>)``, the GVML
    initialization and one register declaration per binding."""

    function: str
    pointers: tuple  # (value name, level) of every pointer the stream addresses
    registers: tuple  # (c name, concrete GVML register) per binding

    def lines(self) -> list[str]:
        params = ", ".join(
            f"uint16_t *{name}_{level}ptr" for name, level in self.pointers
        )
        out = [
            "#include <stdint.h>",
            "#include <gsi/libgvml_memory.h>",
            "#include <gsi/libgvml_element_wise.h>",
            "#include <gsi/libgvml_iv.h>",
            '#include "gsi_dma.h"',
            "",
            f"static int {self.function}_vector({params}) {{",
            "    gvml_init_once();",
        ]
        out.extend(
            f"    enum gvml_vr16 {c_name} = {concrete};"
            for c_name, concrete in self.registers
        )
        return out


@dataclass(frozen=True)
class ScheduleEpilogue:
    """``return 0;`` and the function's closing brace."""

    def lines(self) -> list[str]:
        return ["    return 0;", "}", ""]


@dataclass(frozen=True)
class ScheduleIndex:
    """``const uint32_t name = expression;`` (a batch index of the scope)."""

    name: str
    expression: str


@dataclass(frozen=True)
class ScheduleReset:
    """``gvml_reset_16(register);`` of the output's compute register."""

    register: str


@dataclass(frozen=True)
class ScheduleScatter:
    """The spatial forms' store of the output's compute register: for
    ``scatter`` in ``[0, count)`` (and, when ``bound`` is set, while
    ``destination_base + scatter < bound``) the lane ``scatter *
    lane_stride`` of ``register`` goes to ``output[destination_base +
    scatter]``; ``destination_base`` is an expression over the scope's
    indices.  Rendering derives from these fields."""

    output: str
    register: str
    destination_base: str
    count: int
    lane_stride: int
    bound: int | None = None

    def header(self) -> str:
        guard = (
            f" && {self.destination_base} + scatter < {self.bound}"
            if self.bound is not None
            else ""
        )
        return f"for (uint32_t scatter = 0; scatter < {self.count}{guard}; ++scatter)"

    def body(self) -> str:
        return (
            f"{self.output}_L4ptr[{self.destination_base} + scatter] = "
            f"gvml_get_entry_16({self.register}, scatter * {self.lane_stride});"
        )


@dataclass(frozen=True)
class ScheduleCall:
    """One instruction rendered once.  ``overrides`` are the only register
    substitutions the emission may apply, as (logical name, the logical name
    whose binding it takes): the carried output onto an accumulator slot
    (``(output, f"{output}__acc{k}")``) inside that slot's guard, or the
    resident vector onto a resident bank inside that bank's section;
    ``pointer_batches`` maps a pointer's value to the batch expression of the
    scope."""

    instruction: GVMLInstruction
    overrides: tuple = ()
    pointer_batches: Mapping[str, str] = field(default_factory=dict)

    def bindings(self, base: Mapping[str, VRBinding]) -> dict:
        bound = dict(base)
        for name, source in self.overrides:
            bound[name] = base[source]
        return bound

    def render(self, base: Mapping[str, VRBinding]) -> str:
        return self.instruction.render(self.bindings(base), dict(self.pointer_batches))

    def defined_registers(self, base) -> tuple[str, ...]:
        bound = self.bindings(base)
        return tuple(
            bound[name].c_name for name in self.instruction.definitions if name in bound
        )

    def used_registers(self, base) -> tuple[str, ...]:
        bound = self.bindings(base)
        return tuple(
            bound[argument.name].c_name
            for argument in self.instruction.args
            if isinstance(argument, VRRef)
            and argument.name in self.instruction.uses
            and argument.name in bound
        )


@dataclass(frozen=True)
class ScheduleLoop:
    """``for (uint32_t var = 0; var < count; ++var) { body }``; ``role`` is
    the plan axis the loop walks (``output_batch``, ``reduction_step``,
    ``output_block``, ``output_tile``)."""

    role: str
    var: str
    count: int
    body: tuple


@dataclass(frozen=True)
class ScheduleGuard:
    """``if (output_block * block + slot < total) { body }``: the accumulator
    slot ``slot`` of a block is live for the output batches it covers."""

    slot: int
    block: int
    total: int
    body: tuple

    @property
    def condition(self) -> str:
        return f"output_block * {self.block} + {self.slot} < {self.total}"


@dataclass(frozen=True)
class ScheduleBank:
    """The resident form's section for resident bank ``index`` (no source of
    its own; the calls inside substitute the resident vector)."""

    index: int
    body: tuple


_SCOPE_NODES = (ScheduleLoop, ScheduleGuard, ScheduleBank)


@dataclass(frozen=True)
class EmissionSchedule:
    """The structured emission of one realization: the single source the
    device source is rendered from (rounds 80 and 81, ISSUES X-36).  Loops,
    guards and bank sections carry their plan roles and counts, resets and
    scatters their registers, calls their substitution declarations; the
    validator reads these fields, never the rendered text."""

    steps: tuple

    def render(self, realization) -> str:
        base = realization.binding_map
        lines: list[str] = []

        def emit(steps, depth):
            indent = "    " * depth
            for step in steps:
                if isinstance(step, (SchedulePrologue, ScheduleEpilogue)):
                    lines.extend(step.lines())
                elif isinstance(step, ScheduleIndex):
                    lines.append(
                        f"{indent}const uint32_t {step.name} = {step.expression};"
                    )
                elif isinstance(step, ScheduleReset):
                    lines.append(f"{indent}gvml_reset_16({step.register});")
                elif isinstance(step, ScheduleScatter):
                    lines.append(indent + step.header())
                    lines.append(indent + "    " + step.body())
                elif isinstance(step, ScheduleCall):
                    lines.append(indent + step.render(base))
                elif isinstance(step, ScheduleLoop):
                    lines.append(
                        f"{indent}for (uint32_t {step.var} = 0; {step.var} < "
                        f"{step.count}; ++{step.var}) {{"
                    )
                    emit(step.body, depth + 1)
                    lines.append(f"{indent}}}")
                elif isinstance(step, ScheduleGuard):
                    lines.append(f"{indent}if ({step.condition}) {{")
                    emit(step.body, depth + 1)
                    lines.append(f"{indent}}}")
                elif isinstance(step, ScheduleBank):
                    emit(step.body, depth)
                else:
                    raise TypeError(f"unknown schedule step {type(step).__name__}")

        # The function body is one level deep; the prologue and the epilogue
        # render their own lines.
        emit(self.steps, 1)
        return "\n".join(lines)

    def calls(self) -> tuple:
        found = []

        def walk(steps):
            for step in steps:
                if isinstance(step, ScheduleCall):
                    found.append(step)
                elif isinstance(step, _SCOPE_NODES):
                    walk(step.body)

        walk(self.steps)
        return tuple(found)


def _instruction_stream(plan, normalized_values, normalized_operations):
    """The ordered instruction stream of one plan: the physical emission, the
    transfers' ingress (the carried output's first), the operations'
    compute instructions and the egress.  Shared by the realizer and by
    `validate_realization`, so a realization is checked against the same
    derivation that builds one."""
    physical_emission = _derive_physical_emission(plan, normalized_values)
    value_names = {value.name for value in normalized_values}
    ingress, egress, transfer_records = _transfer_instructions(
        plan,
        value_names,
        physical_emission,
    )
    output_names = {
        value.name for value in normalized_values if value.intent in {"out", "inout"}
    }

    def ingress_touches_output(instruction):
        pointer_names = {
            argument.name
            for argument in instruction.args
            if isinstance(argument, (PointerRef, PointerOffsetRef))
        }
        return bool(
            pointer_names & output_names
            or set(instruction.definitions + instruction.uses) & output_names
        )

    # Device lowering hoists the carried-output load outside the reduction
    # loop.  Allocate against the same order so input staging cannot alias the
    # live accumulator after that code motion.
    ingress = sorted(ingress, key=lambda item: not ingress_touches_output(item))
    compute = []
    for index, operation in enumerate(normalized_operations):
        compute.extend(_op_instructions(operation, index))
    return (
        physical_emission,
        tuple(ingress + compute + egress),
        transfer_records,
        output_names,
    )


def _argument_key(value):
    if isinstance(value, VRRef):
        return ("vr", value.name)
    if isinstance(value, PointerRef):
        return (
            "ptr",
            value.name,
            value.level,
            bool(value.batch_offset),
            int(value.batch_stride),
        )
    if isinstance(value, PointerOffsetRef):
        return ("ptr_offset", value.name, value.level, str(value.offset))
    if isinstance(value, VRBankRef):
        return ("vr_bank", tuple(value.names), str(value.index))
    if isinstance(value, CExpression):
        return ("expr", value.value)
    return ("literal", repr(value))


def instruction_key(instruction: GVMLInstruction) -> tuple:
    """A structural key of one instruction: opcode, API, every argument,
    definitions, uses, phase and predicate."""
    return (
        instruction.opcode,
        instruction.api,
        tuple(_argument_key(argument) for argument in instruction.args),
        tuple(instruction.definitions),
        tuple(instruction.uses),
        instruction.phase,
        instruction.predicate,
    )


def validate_realization(realization, operations) -> list[str]:
    """Why ``realization`` is not the faithful lowering of its plan, its
    values and the given neutral ``operations`` (derived by the caller from
    the shared contraction), or an empty list.

    Checks, each re-derived structurally and none from an approved
    fingerprint: (1) the operations; (2) the whole instruction stream,
    ingress, resident and index setup, compute and egress, instruction by
    instruction (API, every argument including the operand registers in
    order, definitions, uses, phase, predicate) against `_instruction_stream`
    of the same plan, values and operations; (3) the physical emission
    descriptor; (4) the register bindings against the allocation of that
    stream; (5) the emission schedule, executed by `_ScheduleInterpreter`
    on an abstract machine and compared with the plan's own batching
    executed on another: every store's value, destination and multiplicity,
    the scatters' coverage of the output, the compute, reset and setup
    counts (and the priced inventory), the ingress multiplicities, every
    call an instruction of the stream, every index and pointer batch
    evaluated in its scope, every substitution declared and in scope.  The
    emitted text is the schedule's rendering (`device_source`); the caller
    compares it with what it emitted as a tamper check only.
    """
    problems: list[str] = []
    plan = realization.plan
    try:
        expected_operations = _normalize_operations(plan, operations)
    except UnsupportedVectorOperation as error:
        return [f"operations cannot be normalized: {error}"]

    def op_key(op):
        return (op.opcode, op.output, tuple(op.inputs), dict(op.attrs))

    realized_ops = [op_key(op) for op in realization.operations]
    derived_ops = [op_key(op) for op in expected_operations]
    if realized_ops != derived_ops:
        return [
            f"operations differ from the contraction: realized {realized_ops}, "
            f"derived {derived_ops}"
        ]
    try:
        emission, expected, _records, output_names = _instruction_stream(
            plan, realization.values, expected_operations
        )
    except (TypeError, ValueError, KeyError, UnsupportedVectorOperation) as error:
        return [f"the plan's instruction stream cannot be derived: {error}"]
    realized = tuple(realization.instructions)
    if len(realized) != len(expected) or any(
        instruction_key(a) != instruction_key(b) for a, b in zip(realized, expected)
    ):
        position = next(
            (
                index
                for index, (a, b) in enumerate(zip(realized, expected))
                if instruction_key(a) != instruction_key(b)
            ),
            min(len(realized), len(expected)),
        )
        got = instruction_key(realized[position]) if position < len(realized) else None
        want = instruction_key(expected[position]) if position < len(expected) else None
        phase = (want or got)[5] if (want or got) else "?"
        return [
            f"instruction {position} ({phase}) differs from the plan's stream: "
            f"realized {got}, derived {want}"
        ]
    if emission != realization.physical_emission:
        problems.append(
            f"physical emission differs: realized {realization.physical_emission!r}, "
            f"derived {emission!r}"
        )
    try:
        bindings = _allocate_vrs(plan, expected, capacity=WRITABLE_VRS)
        bindings = _reserve_accumulator_block(
            plan, bindings, output_names, capacity=WRITABLE_VRS
        )
    except (ValueError, UnsupportedVectorOperation, VRCapacityError) as error:
        return problems + [f"the plan's registers cannot be allocated: {error}"]
    derived_bindings = {(b.logical_name, b.concrete) for b in bindings}
    realized_bindings = {(b.logical_name, b.concrete) for b in realization.vr_bindings}
    if derived_bindings != realized_bindings:
        problems.append(
            f"register bindings differ: realized {sorted(realized_bindings)}, "
            f"derived {sorted(derived_bindings)}"
        )
    if problems:
        return problems
    try:
        schedule = realization.emission_schedule()
    except (ValueError, UnsupportedVectorOperation, VRCapacityError) as error:
        return [f"the emission schedule cannot be built: {error}"]
    return _ScheduleInterpreter(realization, emission, output_names).check(schedule)


class _Fault(Exception):
    """One reason the schedule is not the plan's lowering."""


class _ExpressionEvaluator(ast.NodeVisitor):
    """Evaluate a backend-produced C index expression (non-negative
    integers, the indices of the scope, ``+ - * / %``, parentheses, one
    comparison) under concrete index values.  ``/`` is C's truncating
    division, which on the non-negative operands checked here is floor
    division."""

    def __init__(self, names):
        self.names = names

    def evaluate(self, expression):
        try:
            tree = ast.parse(str(expression), mode="eval")
        except SyntaxError as error:
            raise ValueError(f"unparsable expression {expression!r}") from error
        return self.visit(tree.body)

    def visit_Constant(self, node):  # noqa: N802
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(f"non-integer constant {node.value!r}")
        return node.value

    def visit_Name(self, node):  # noqa: N802
        if node.id not in self.names:
            raise ValueError(f"unknown index {node.id!r}")
        return self.names[node.id]

    def visit_BinOp(self, node):  # noqa: N802
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if right <= 0 or left < 0:
                raise ValueError("division of a negative or by zero")
            return left // right if not isinstance(node.op, ast.Mod) else left % right
        raise ValueError("unsupported operator")

    def visit_UnaryOp(self, node):  # noqa: N802
        if isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        raise ValueError("unsupported operator")

    def visit_Compare(self, node):  # noqa: N802
        if len(node.ops) != 1:
            raise ValueError("chained comparison")
        left, right = self.visit(node.left), self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise ValueError("unsupported comparison")

    def generic_visit(self, node):
        raise ValueError(f"unsupported expression node {type(node).__name__}")


class _Values:
    """Hash-consed abstract values: every distinct value is one integer id,
    so a reduction of thousands of steps compares in constant time and two
    machines sharing one table produce equal ids for equal computations."""

    def __init__(self):
        self.ids: dict = {}

    def make(self, *key):
        return self.ids.setdefault(key, len(self.ids))


def _register_key(binding) -> tuple:
    """The physical identity of a binding: the concrete VR.  Every machine
    cell is keyed by this, so logical names sharing one hardware register
    (the allocator reuses VRs across disjoint live ranges) share one cell."""
    return ("vr", int(binding.concrete))


class _Machine:
    """An abstract APU: VR registers and L1 slots hold abstract values;
    stores to L4 are recorded by destination.  An instruction's effect is
    its opcode's: a DMA into L1 places the addressed tile of its pointer in
    the slot, a load moves a slot into a register, a store moves a register
    into a slot, a DMA out of L1 (or a PIO store) records the slot's (or
    register's) value at the addressed tile, and every other instruction
    defines its output registers with the opcode applied to its operands in
    order (register values, the addressed tiles of pointer operands, the
    evaluated backend expressions, and the literal parameters)."""

    def __init__(self, values: _Values):
        self.values = values
        self.registers: dict = {}
        self.memory: dict = {}
        self.stores: dict = {}
        self.counts: dict = {}

    def count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def read(self, key, label, api):
        """The value of the physical register ``key`` (``label`` names it
        for the message).  Storage is keyed by the concrete VR, never by a
        logical name, so two logical values the allocator placed in one
        hardware register share one cell and a write to either is a write
        to both (the clobber a schedule change can introduce)."""
        if key not in self.registers:
            raise _Fault(
                f"{api} reads {label} before anything in the emission defines it"
            )
        return self.registers[key]

    def store(self, key, value):
        previous = self.stores.get(key)
        self.stores[key] = (value, 1 if previous is None else previous[1] + 1)

    def reset(self, key):
        self.count(("reset", key))
        self.registers[key] = self.values.make("zero")

    def scatter(self, output, key, label, base, count, lane_stride, bound):
        self.count(("scatter", output))
        self.store(
            ("scatter", output, int(base), int(count), int(lane_stride), bound),
            self.read(key, label, "the scatter"),
        )

    def execute(self, instruction, bound, scope, batch_of):
        """``batch_of(name)`` is the batch index a batched pointer of value
        ``name`` addresses in this context."""
        evaluator = _ExpressionEvaluator(scope)
        if instruction.predicate is not None and not evaluator.evaluate(
            instruction.predicate
        ):
            return
        self.count(instruction_key(instruction))
        api = instruction.api

        def binding(name):
            try:
                return bound[name]
            except KeyError as error:
                raise _Fault(f"{api} names {name!r}, which has no register") from error

        def register(name):
            # The physical identity after the call's substitutions.
            return _register_key(binding(name))

        def label(name):
            b = binding(name)
            return f"{b.c_name} ({b.concrete_name})"

        def pointer(argument):
            if isinstance(argument, PointerRef):
                return (
                    "ptr",
                    argument.name,
                    int(batch_of(argument.name)) if argument.batch_offset else 0,
                )
            return (
                "ptr",
                argument.name,
                ("offset", int(evaluator.evaluate(argument.offset))),
            )

        def slot(argument):
            if not isinstance(argument, str) or not argument.startswith("GVML_VM_"):
                raise _Fault(f"{api} addresses {argument!r}, which is not an L1 slot")
            return argument

        args = instruction.args
        opcode = instruction.opcode
        if opcode == "DMA_L4_TO_L1_32K":
            destination, source = args
            self.memory[slot(destination)] = self.values.make("mem", pointer(source))
            return
        if opcode == "LOAD_16":
            destination, source = args
            if slot(source) not in self.memory:
                raise _Fault(f"{api} loads {source} before any DMA fills it")
            self.registers[register(destination.name)] = self.memory[source]
            return
        if opcode == "STORE_16":
            destination, source = args
            self.memory[slot(destination)] = self.read(
                register(source.name), label(source.name), api
            )
            return
        if opcode == "DMA_L1_TO_L4_32K":
            destination, source = args
            if slot(source) not in self.memory:
                raise _Fault(f"{api} stores {source} before anything fills it")
            self.store(("store",) + pointer(destination)[1:], self.memory[source])
            return
        if opcode == "PIO_STORE":
            destination, source = args
            self.store(
                ("store",) + pointer(destination)[1:],
                self.read(register(source.name), label(source.name), api),
            )
            return
        operands = []
        for argument in args:
            if isinstance(argument, VRRef):
                if (
                    argument.name in instruction.definitions
                    and argument.name not in instruction.uses
                ):
                    operands.append(("dst",))
                else:
                    operands.append(
                        (
                            "vr",
                            self.read(
                                register(argument.name), label(argument.name), api
                            ),
                        )
                    )
            elif isinstance(argument, VRBankRef):
                index = int(evaluator.evaluate(argument.index))
                if not 0 <= index < len(argument.names):
                    raise _Fault(
                        f"{api} selects resident bank {index} of {len(argument.names)}"
                    )
                chosen = argument.names[index]
                operands.append(("vr", self.read(register(chosen), label(chosen), api)))
            elif isinstance(argument, (PointerRef, PointerOffsetRef)):
                operands.append(pointer(argument))
            elif isinstance(argument, CExpression):
                operands.append(("expr", int(evaluator.evaluate(argument.value))))
            else:
                operands.append(("lit", str(argument)))
        if not instruction.definitions:
            raise _Fault(f"{api} has no effect the contract models")
        value = self.values.make("op", opcode, api, tuple(operands))
        for name in instruction.definitions:
            self.registers[register(name)] = value


class _ScheduleInterpreter:
    """The plan-to-emission contract (ISSUES X-36): the emission schedule
    is executed on an abstract machine and compared with the plan's
    batching executed on another.

    The reference executes the plan's own instruction stream under the
    plan's batching, without the schedule's loops, guards, slots or
    indices: the setup once; for a temporal reduction, for every output
    batch ``b`` the carried output's ingress at batch ``b``, then for every
    reduction step ``s`` the operand ingress at tile ``b * steps + s`` and
    the operations, then the egress at batch ``b``; for a spatial
    reduction the same per work tile ``(b, s)`` from a reset register with
    the scatter of ``elements`` lanes to ``b * lanes + s * elements``; for
    the resident form, per matrix tile ``t`` and resident bank ``i`` the
    streamed operand at tile ``t * banks + i`` with the bank's register for
    the resident vector, then the scatter of the tile's rows.  Values are
    abstract (an addressed tile, an opcode applied to operands), so the
    reference fixes what every destination must receive and the
    destinations themselves.

    The schedule is then executed concretely: loops iterate, guards
    evaluate, indices and pointer batches evaluate under the scope, the
    declared substitutions resolve against the bindings, predicates
    evaluate, and every call is an instruction of the stream applied to the
    machine.  The run must store the reference value at every reference
    destination exactly once and nowhere else; its compute, reset and setup
    counts must equal the reference's and the priced inventory; its ingress
    must execute at least once and at most as often as the reference; a
    register read before it is defined, a call outside the stream, a
    substitution outside its scope, an index outside its loop, and a
    scatter of anything but the output's register are faults.  Any
    deviation in addressing, operand order, multiplicity, loop extent,
    coverage or substitution changes a stored value, a destination or a
    count and is reported.
    """

    def __init__(self, realization, emission, output_names):
        self.realization = realization
        self.plan = realization.plan
        self.emission = emission
        self.outputs = set(output_names)
        self.output = next(iter(sorted(output_names)), None)
        self.base = realization.binding_map
        self.values = _Values()
        self.block = int(realization.accumulator_block)
        self.resident = (
            emission if isinstance(emission, SpatialResidentReductionEmission) else None
        )
        self.spatial = (
            None
            if self.resident is not None
            else realization._spatial_scatter_elements()
        )
        self.output_batches = int(_plan_output_batches(self.plan))
        self.reduction_steps = int(_plan_reduction_steps(self.plan))
        self.stream = tuple(realization.instructions)
        self.stream_keys = {instruction_key(i): i for i in self.stream}
        self.setup = [
            i for i in self.stream if i.phase in {"resident_setup", "index_setup"}
        ]
        ingress = [i for i in self.stream if i.phase == "transfer_in"]
        self.outer_ingress = [i for i in ingress if self.touches_output(i)]
        self.inner_ingress = [i for i in ingress if not self.touches_output(i)]
        self.compute = [i for i in self.stream if i.phase == "compute"]
        self.egress = [i for i in self.stream if i.phase == "transfer_out"]

    def touches_output(self, instruction):
        pointers = {
            a.name
            for a in instruction.args
            if isinstance(a, (PointerRef, PointerOffsetRef))
        }
        return bool(
            pointers & self.outputs
            or set(instruction.definitions + instruction.uses) & self.outputs
        )

    @property
    def output_register(self):
        """The output's C register name (the schedule's reset and scatter
        nodes name it)."""
        return self.base[self.output].c_name

    @property
    def output_key(self):
        """The output's physical register, the machine cell its resets and
        scatters touch."""
        return _register_key(self.base[self.output])

    @property
    def output_label(self):
        b = self.base[self.output]
        return f"{b.c_name} ({b.concrete_name})"

    # ------------------------------------------------------------------ #
    def reference(self) -> _Machine:
        machine = _Machine(self.values)
        base = self.base
        out = self.output_key
        label = self.output_label
        for instruction in self.setup:
            machine.execute(instruction, base, {}, lambda name: 0)
        if self.resident is not None:
            form = self.resident
            banks = int(form.reduction_tiles)
            for tile in range(int(form.matrix_tiles)):
                machine.reset(out)
                for bank in range(banks):
                    bound = dict(base)
                    bound[form.resident_value] = base[form.resident_names[bank]]
                    scope = {"output_tile": tile, "bank": bank}
                    at = tile * banks + bank
                    for instruction in self.inner_ingress + self.compute:
                        machine.execute(
                            instruction, bound, scope, lambda name, at=at: at
                        )
                machine.scatter(
                    self.output,
                    out,
                    label,
                    tile * int(form.rows_per_tile),
                    int(form.rows_per_tile),
                    int(form.reduction_tile_extent),
                    int(form.output_extent),
                )
            return machine
        steps = self.reduction_steps
        for batch in range(self.output_batches):
            outer = {"output_batch": batch}
            if self.spatial is None:
                for instruction in self.outer_ingress:
                    machine.execute(instruction, base, outer, lambda name, b=batch: b)
            for step in range(steps):
                at = batch * steps + step
                scope = {"output_batch": batch, "reduction_step": step, "batch": at}
                if self.spatial is not None:
                    machine.reset(out)
                for instruction in self.inner_ingress + self.compute:
                    machine.execute(instruction, base, scope, lambda name, at=at: at)
                if self.spatial is not None:
                    machine.scatter(
                        self.output,
                        out,
                        label,
                        batch * VR_LANES + step * int(self.spatial),
                        int(self.spatial),
                        int(_plan_group_size(self.plan)),
                        None,
                    )
            if self.spatial is None:
                for instruction in self.egress:
                    machine.execute(instruction, base, outer, lambda name, b=batch: b)
        return machine

    # ------------------------------------------------------------------ #
    def check(self, schedule) -> list[str]:
        if self.output is None:
            return ["the plan has no carried output to check against"]
        steps = tuple(schedule.steps)
        if (
            len(steps) < 2
            or not isinstance(steps[0], SchedulePrologue)
            or not isinstance(steps[-1], ScheduleEpilogue)
        ):
            return [
                "the schedule does not begin with the prologue and end with the epilogue"
            ]
        pointers = sorted(
            {
                (a.name, a.level)
                for i in self.stream
                for a in i.args
                if isinstance(a, (PointerRef, PointerOffsetRef))
            }
        )
        registers = tuple(
            (b.c_name, b.concrete_name) for b in self.realization.vr_bindings
        )
        if (
            tuple(steps[0].pointers) != tuple(pointers)
            or tuple(steps[0].registers) != registers
        ):
            return [
                "the prologue's pointer parameters or register declarations differ from the realization"
            ]
        try:
            reference = self.reference()
        except (_Fault, ValueError, KeyError) as error:
            return [
                f"the plan's stream cannot be executed under the plan's batching: {error}"
            ]
        run = _Machine(self.values)
        try:
            self.run(run, steps[1:-1], {})
        except _Fault as error:
            return [str(error)]
        except ValueError as error:
            return [f"the schedule cannot be executed: {error}"]
        return self.compare(reference, run)

    def compare(self, reference, run) -> list[str]:
        problems: list[str] = []

        def describe(key):
            if key[0] == "store":
                return f"batch {key[2]} of {key[1]!r}"
            if key[0] == "scatter":
                _kind, output, base, count, stride, bound = key
                limit = "" if bound is None else f" below {bound}"
                return (
                    f"the scatter of {count} lanes (stride {stride}) of {output!r} "
                    f"to elements {base}..{base + count - 1}{limit}"
                )
            return str(key)

        for key, (value, times) in reference.stores.items():
            stored = run.stores.get(key)
            if stored is None:
                problems.append(f"{describe(key)} is never stored")
            elif stored[1] != times:
                problems.append(
                    f"{describe(key)} is stored {stored[1]} times; the plan stores it {times} time(s)"
                )
            elif stored[0] != value:
                problems.append(
                    f"{describe(key)} receives a value other than the plan's computation "
                    "(its operands, their order, their addresses or the reduction extent differ)"
                )
        for key in run.stores:
            if key not in reference.stores:
                problems.append(
                    f"{describe(key)} is stored but the plan has no such destination"
                )
        if problems:
            return problems[:6]
        # Coverage of the output elements by the scatters.
        if self.resident is not None or self.spatial is not None:
            elements: dict = {}
            for key in reference.stores:
                _kind, _output, base, count, _stride, bound = key
                for lane in range(count):
                    element = base + lane
                    if bound is not None and element >= bound:
                        break
                    elements[element] = elements.get(element, 0) + 1
            extent = (
                int(self.resident.output_extent)
                if self.resident is not None
                else self.output_batches * self.reduction_steps * int(self.spatial)
            )
            if sorted(elements) != list(range(extent)) or any(
                t != 1 for t in elements.values()
            ):
                problems.append(
                    f"the scatters cover {len(elements)} output elements; the plan has {extent}"
                )
        # Counts: compute, resets and setup exactly; ingress bounded.
        for key, times in reference.counts.items():
            emitted = run.counts.get(key, 0)
            if isinstance(key[0], str) and key[0] in {"reset", "scatter"}:
                target = (
                    f"GVML_VR16_{key[1][1]}" if isinstance(key[1], tuple) else key[1]
                )
                label = f"the {key[0]} of {target}"
                exact = True
            else:
                instruction = self.stream_keys[key]
                label = f"{instruction.api} ({instruction.phase})"
                exact = instruction.phase != "transfer_in"
            if exact and emitted != times:
                problems.append(
                    f"{label} executes {emitted} times; the plan requires {times}"
                )
            elif not exact and (emitted == 0 or emitted > times):
                problems.append(
                    f"{label} executes {emitted} times; the plan requires between 1 and {times}"
                )
        for key, emitted in run.counts.items():
            if key not in reference.counts:
                label = (
                    self.stream_keys[key].api if key in self.stream_keys else str(key)
                )
                problems.append(
                    f"{label} executes {emitted} times; the plan's batching never runs it"
                )
        if problems:
            return problems[:6]
        # The priced inventory against the executed compute.
        priced: dict = {}
        for item in self.realization.compute_invocation_inventory:
            priced[item.opcode] = priced.get(item.opcode, 0) + int(item.count)
        executed: dict = {}
        for key, times in run.counts.items():
            if key[0] == "reset":
                executed["RESET_16"] = executed.get("RESET_16", 0) + times
            elif key in self.stream_keys and self.stream_keys[key].phase == "compute":
                opcode = self.stream_keys[key].opcode
                executed[opcode] = executed.get(opcode, 0) + times
        if executed != priced:
            problems.append(
                f"the executed compute inventory {sorted(executed.items())} differs from "
                f"the priced inventory {sorted(priced.items())}"
            )
        return problems

    # ------------------------------------------------------------------ #
    def expected_loop_count(self, role):
        if role == "output_batch":
            return self.output_batches
        if role == "reduction_step":
            return self.reduction_steps
        if role == "output_block":
            return math.ceil(self.output_batches / self.block)
        if role == "output_tile" and self.resident is not None:
            return int(self.resident.matrix_tiles)
        raise _Fault(f"loop role {role!r} is not an axis of the plan")

    def run(self, machine, steps, scope):
        for step in steps:
            if isinstance(step, (SchedulePrologue, ScheduleEpilogue)):
                raise _Fault("a prologue or epilogue inside the function body")
            if isinstance(step, ScheduleLoop):
                if step.var in scope:
                    raise _Fault(
                        f"loop variable {step.var!r} shadows an index of the scope"
                    )
                expected = self.expected_loop_count(step.role)
                if int(step.count) != expected or step.var != step.role:
                    raise _Fault(
                        f"loop {step.var!r} ({step.role}) runs {step.count} times; the plan requires {expected}"
                    )
                for value in range(int(step.count)):
                    inner = dict(scope)
                    inner[step.var] = value
                    self.run(machine, step.body, inner)
            elif isinstance(step, ScheduleGuard):
                if "output_block" not in scope:
                    raise _Fault(
                        "an accumulator-slot guard outside the output-block loop"
                    )
                if (
                    int(step.block) != self.block
                    or int(step.total) != self.output_batches
                    or not 0 <= int(step.slot) < self.block
                ):
                    raise _Fault(
                        f"guard of slot {step.slot} of block {step.block} over {step.total} batches; "
                        f"the plan blocks {self.block} accumulators over {self.output_batches}"
                    )
                if (
                    scope["output_block"] * self.block + int(step.slot)
                    < self.output_batches
                ):
                    inner = dict(scope)
                    inner["__slot__"] = int(step.slot)
                    self.run(machine, step.body, inner)
            elif isinstance(step, ScheduleBank):
                if self.resident is None or not 0 <= int(step.index) < len(
                    self.resident.resident_names
                ):
                    raise _Fault(
                        f"resident bank section {step.index} outside the plan's banks"
                    )
                inner = dict(scope)
                inner["__bank__"] = int(step.index)
                self.run(machine, step.body, inner)
            elif isinstance(step, ScheduleIndex):
                if not re.fullmatch(
                    r"[A-Za-z_]\w*", str(step.name)
                ) or step.name.startswith("__"):
                    raise _Fault(f"index name {step.name!r} is not an identifier")
                if step.name in scope:
                    raise _Fault(f"index {step.name!r} redefines an index of the scope")
                scope[step.name] = int(
                    _ExpressionEvaluator(scope).evaluate(step.expression)
                )
            elif isinstance(step, ScheduleReset):
                if step.register != self.output_register:
                    raise _Fault(
                        f"reset of {step.register!r}, which is not the output's register"
                    )
                machine.reset(self.output_key)
            elif isinstance(step, ScheduleScatter):
                if step.output != self.output or step.register != self.output_register:
                    raise _Fault("a scatter of a register that is not the output's")
                base = _ExpressionEvaluator(scope).evaluate(step.destination_base)
                machine.scatter(
                    step.output,
                    self.output_key,
                    self.output_label,
                    int(base),
                    int(step.count),
                    int(step.lane_stride),
                    None if step.bound is None else int(step.bound),
                )
            elif isinstance(step, ScheduleCall):
                self.call(machine, step, scope)
            else:
                raise _Fault(f"unknown schedule step {type(step).__name__}")

    def call(self, machine, step, scope):
        instruction = step.instruction
        key = instruction_key(instruction)
        if key not in self.stream_keys:
            raise _Fault(
                f"the emission calls {instruction.api}, which is not in the plan's instruction stream"
            )
        slot = scope.get("__slot__")
        bank = scope.get("__bank__")
        for name, source in step.overrides:
            legal = (
                name == self.output
                and slot is not None
                and slot >= 1
                and source == f"{self.output}__acc{slot}"
            ) or (
                self.resident is not None
                and bank is not None
                and name == self.resident.resident_value
                and source == self.resident.resident_names[bank]
            )
            if not legal or source not in self.base:
                raise _Fault(
                    f"{instruction.api} substitutes {name!r} by the register of {source!r}, "
                    "which the plan does not allow in this scope"
                )
        if (
            slot is not None
            and slot >= 1
            and self.output in (instruction.definitions + instruction.uses)
        ):
            if (self.output, f"{self.output}__acc{slot}") not in tuple(step.overrides):
                raise _Fault(
                    f"{instruction.api} touches the output in accumulator slot {slot} without the slot's register"
                )
        if (
            bank is not None
            and self.resident is not None
            and self.resident.resident_value in instruction.uses
        ):
            if (
                self.resident.resident_value,
                self.resident.resident_names[bank],
            ) not in tuple(step.overrides):
                raise _Fault(
                    f"{instruction.api} reads the resident vector in bank {bank} without that bank's register"
                )
        evaluator = _ExpressionEvaluator(scope)
        batches = dict(step.pointer_batches)

        def batch_of(name):
            return evaluator.evaluate(batches.get(name, "batch"))

        machine.execute(instruction, step.bindings(self.base), scope, batch_of)


@dataclass(frozen=True)
class APUVectorRealization:
    plan: object
    values: tuple[VectorValue, ...]
    operations: tuple[VectorOp, ...]
    instructions: tuple[GVMLInstruction, ...]
    vr_bindings: tuple[VRBinding, ...]
    transfers: tuple[TransferRecord, ...]
    physical_emission: PhysicalEmissionDescriptor
    abi: APUVectorABI
    _runtime_artifact: object | None = field(default=None, repr=False, compare=False)

    @property
    def runtime_artifact(self):
        artifact = self._runtime_artifact
        if artifact is not None:
            return artifact
        try:
            from .apu_v1_vector_runtime import freeze_apu_v1_runtime_artifact

            artifact = freeze_apu_v1_runtime_artifact(self)
        except FileNotFoundError:
            return None
        object.__setattr__(self, "_runtime_artifact", artifact)
        return artifact

    @property
    def carries_output_ingress(self) -> bool:
        """Whether the device program accumulates onto the incoming output.

        Temporal plans load each output tile from L4 before the reduction, so
        a carried (nonzero) initial output is accumulated exactly once the
        host transfers its packed image.  Spatial group-reduction plans reset
        the output VR and scatter group heads, so they cannot carry one.
        """

        if isinstance(self.physical_emission, SpatialResidentReductionEmission):
            return False
        if self._spatial_scatter_elements() is not None:
            return False
        output_names = {
            value.name for value in self.values if value.intent in {"out", "inout"}
        }
        return any(
            isinstance(argument, (PointerRef, PointerOffsetRef))
            and argument.name in output_names
            for instruction in self.instructions
            if instruction.phase == "transfer_in"
            for argument in instruction.args
        )

    def check_initial_outputs(self, arrays) -> None:
        """Reject a nonzero initial output this plan cannot accumulate onto.

        Every route (device, functional, virtual) calls this, so a plan that
        resets its output never silently returns ``A@B`` for ``C0 + A@B``.
        """

        if self.carries_output_ingress:
            return
        for value in self.values:
            if value.intent not in {"out", "inout"} or value.name not in arrays:
                continue
            if np.any(np.asarray(arrays[value.name]) != 0):
                plan = getattr(self.plan, "name", "selected plan")
                raise ValueError(
                    f"APU v1 plan {plan!r} resets output {value.name!r} on the "
                    "device (spatial group reduction has no output ingress), so "
                    "a nonzero initial value cannot be accumulated; pass a "
                    "zero-initialized output or select a temporal plan with "
                    "layout="
                )

    @property
    def binding_map(self) -> dict[str, VRBinding]:
        return {binding.logical_name: binding for binding in self.vr_bindings}

    @property
    def vector_batches(self) -> int:
        return _plan_vector_batches(self.plan)

    @property
    def output_batches(self) -> int:
        return _plan_output_batches(self.plan)

    @property
    def reduction_steps(self) -> int:
        return _plan_reduction_steps(self.plan)

    @property
    def compute_invocation_inventory(
        self,
    ) -> tuple[APUVectorComputeInvocation, ...]:
        """Return every emitted compute opcode with its exact dynamic count."""

        compute_opcodes = tuple(
            instruction.opcode
            for instruction in self.instructions
            if instruction.phase == "compute"
        )
        operation_opcodes = tuple(operation.opcode for operation in self.operations)
        if compute_opcodes != operation_opcodes:
            raise UnsupportedVectorOperation(
                "realized operations disagree with emitted compute instructions"
            )
        emission = self.physical_emission
        reset_count = 0
        if isinstance(emission, SpatialResidentReductionEmission):
            operation_count = emission.operation_call_count
            reset_count = emission.output_reset_call_count
        else:
            operation_count = emission.operation_call_count
            if self._spatial_scatter_elements() is not None:
                reset_count = operation_count
        inventory = []
        if reset_count:
            inventory.append(APUVectorComputeInvocation("RESET_16", reset_count))
        inventory.extend(
            APUVectorComputeInvocation(
                operation.opcode,
                operation_count,
                operation.attrs,
            )
            for operation in self.operations
        )
        return tuple(inventory)

    @property
    def compute_tiles_per_output(self) -> int:
        batching = getattr(self.plan, "output_batching", None)
        if batching is None:
            return 1
        return max(1, int(batching.work_tiles_per_output_batch))

    @property
    def accumulator_block(self) -> int:
        return max(1, int(getattr(self.plan, "accumulator_block", 1)))

    def _spatial_scatter_elements(self) -> int | None:
        """Return dense elements per spatial work tile, proving placement order."""

        if self.compute_tiles_per_output == 1:
            return None
        batching = getattr(self.plan, "output_batching", None)
        if batching is None or int(batching.reduction_tiles) != 1:
            raise UnsupportedVectorOperation(
                "multi-tile output with temporal reduction needs explicit accumulation"
            )
        grouped = [[] for _ in range(self.output_batches)]
        for placement in batching.placements:
            grouped[placement.physical_output_batch].append(placement)
        first = grouped[0]
        if not first:
            raise UnsupportedVectorOperation("output batching has no placements")
        elements = math.prod(first[0].logical_shape)
        for placements in grouped:
            if len(placements) != self.compute_tiles_per_output:
                raise UnsupportedVectorOperation(
                    "output batches must contain a uniform number of work tiles"
                )
            for local, placement in enumerate(placements):
                if (
                    math.prod(placement.logical_shape) != elements
                    or placement.lane_offset != local * elements
                ):
                    raise UnsupportedVectorOperation(
                        "spatial output placements are not contiguous dense tiles"
                    )
        group_size = _plan_group_size(self.plan)
        if elements * group_size > VR_LANES:
            raise UnsupportedVectorOperation(
                "spatial group heads exceed the compute VR"
            )
        return elements

    def emission_schedule(self) -> "EmissionSchedule":
        """The structured emission the device source is rendered from
        (rounds 80 and 81, ISSUES X-36): loops with their plan roles and
        counts, accumulator-slot guards, resident-bank sections, resets,
        scatters and calls with declared substitutions.
        `validate_realization` checks the schedule's scopes, counts,
        substitutions and ordering against the plan; `device_source`
        renders it, so the emitted text is the validated schedule's."""
        bindings = self.binding_map
        pointer_values = sorted(
            {
                (argument.name, argument.level)
                for instruction in self.instructions
                for argument in instruction.args
                if isinstance(argument, (PointerRef, PointerOffsetRef))
            }
        )
        params = ", ".join(
            f"uint16_t *{name}_{level}ptr" for name, level in pointer_values
        )
        function = re.sub(r"\W", "_", str(getattr(self.plan, "name", "apu_vector")))
        steps: list = [
            SchedulePrologue(
                function,
                tuple(pointer_values),
                tuple(
                    (binding.c_name, binding.concrete_name)
                    for binding in self.vr_bindings
                ),
            )
        ]
        output_names = {
            value.name for value in self.values if value.intent in {"out", "inout"}
        }

        def touches_output(instruction):
            pointer_names = {
                argument.name
                for argument in instruction.args
                if isinstance(argument, (PointerRef, PointerOffsetRef))
            }
            return bool(
                pointer_names & output_names
                or set(instruction.definitions + instruction.uses) & output_names
            )

        resident_setup = [
            item for item in self.instructions if item.phase == "resident_setup"
        ]
        index_setup = [
            item for item in self.instructions if item.phase == "index_setup"
        ]
        ingress = [item for item in self.instructions if item.phase == "transfer_in"]
        compute = [item for item in self.instructions if item.phase == "compute"]
        egress = [item for item in self.instructions if item.phase == "transfer_out"]
        outer_ingress = [item for item in ingress if touches_output(item)]
        inner_ingress = [item for item in ingress if not touches_output(item)]
        pointer_batches = {name: "output_batch" for name in output_names}
        for instruction in ingress:
            if instruction.opcode != "LOOKUP_16":
                continue
            for argument in instruction.args:
                if isinstance(argument, PointerRef):
                    pointer_batches[argument.name] = (
                        "output_batch * " f"{self.reduction_steps} + reduction_step"
                    )

        def call(instruction, overrides=(), batches=None):
            return ScheduleCall(
                instruction,
                tuple(overrides),
                dict(pointer_batches if batches is None else batches),
            )

        # The function body: setup first.
        body: list = [call(instruction) for instruction in resident_setup + index_setup]
        epilogue = [ScheduleEpilogue()]

        emission = self.physical_emission
        if isinstance(emission, SpatialResidentReductionEmission):
            if output_names != {emission.output_value}:
                raise UnsupportedVectorOperation(
                    "spatial resident reduction requires its retained output vector"
                )
            output = emission.output_value
            vector = emission.resident_value
            reduction_tiles = emission.reduction_tiles
            resident_names = emission.resident_names
            if any(name not in bindings for name in resident_names):
                raise VRCapacityError(
                    "spatial GEMV resident vector bank was not fully allocated"
                )
            group_size = emission.reduction_tile_extent
            rows_per_tile = emission.rows_per_tile
            output_rows = emission.output_extent
            matrix_tiles = emission.matrix_tiles
            tile_body: list = [ScheduleReset(bindings[output].c_name)]
            for reduction_index, resident_name in enumerate(resident_names):
                slot_batches = dict(pointer_batches)
                for instruction in inner_ingress:
                    for argument in instruction.args:
                        if isinstance(argument, PointerRef):
                            slot_batches[argument.name] = f"batch_{reduction_index}"
                overrides = ((vector, resident_name),)
                bank: list = [
                    ScheduleIndex(
                        f"batch_{reduction_index}",
                        f"output_tile * {reduction_tiles} + {reduction_index}",
                    )
                ]
                bank.extend(
                    call(instruction, overrides, slot_batches)
                    for instruction in inner_ingress
                )
                bank.extend(
                    call(instruction, overrides, slot_batches)
                    for instruction in compute
                )
                tile_body.append(ScheduleBank(reduction_index, tuple(bank)))
            tile_body.append(
                ScheduleScatter(
                    output,
                    bindings[output].c_name,
                    f"output_tile * {rows_per_tile}",
                    rows_per_tile,
                    group_size,
                    bound=output_rows,
                )
            )
            body.append(
                ScheduleLoop(
                    "output_tile", "output_tile", matrix_tiles, tuple(tile_body)
                )
            )
            return EmissionSchedule(tuple(steps) + tuple(body) + tuple(epilogue))

        spatial_elements = self._spatial_scatter_elements()
        if self.accumulator_block > 1:
            if spatial_elements is not None:
                raise UnsupportedVectorOperation(
                    "accumulator blocking is only defined for temporal reductions"
                )
            if len(output_names) != 1:
                raise UnsupportedVectorOperation(
                    "accumulator blocking requires exactly one carried output"
                )
            output = next(iter(output_names))
            block = self.accumulator_block
            for slot in range(1, block):
                if f"{output}__acc{slot}" not in bindings:
                    raise VRCapacityError(
                        f"accumulator slot {slot} of {output!r} has no allocated VR"
                    )

            def has_batched_pointer(instruction):
                return any(
                    isinstance(argument, PointerRef) for argument in instruction.args
                )

            slot_ingress = [item for item in inner_ingress if has_batched_pointer(item)]
            shared_ingress = [
                item for item in inner_ingress if not has_batched_pointer(item)
            ]
            if not slot_ingress:
                raise UnsupportedVectorOperation(
                    "accumulator blocking needs output-batched operand ingress"
                )

            def slot_substitution(slot):
                current = f"(output_block * {block} + {slot})"
                overrides = () if slot == 0 else ((output, f"{output}__acc{slot}"),)
                slot_batches = dict(pointer_batches)
                slot_batches[output] = current
                for item in slot_ingress:
                    if item.opcode != "LOOKUP_16":
                        continue
                    for argument in item.args:
                        if isinstance(argument, PointerRef):
                            slot_batches[argument.name] = (
                                f"{current} * {self.reduction_steps} + reduction_step"
                            )
                return overrides, slot_batches

            output_blocks = math.ceil(self.output_batches / block)
            block_body: list = []
            for slot in range(block):
                overrides, batches = slot_substitution(slot)
                block_body.append(
                    ScheduleGuard(
                        slot,
                        block,
                        self.output_batches,
                        tuple(
                            call(instruction, overrides, batches)
                            for instruction in outer_ingress
                        ),
                    )
                )
            reduction_body: list = [call(instruction) for instruction in shared_ingress]
            for slot in range(block):
                overrides, batches = slot_substitution(slot)
                guarded: list = [
                    ScheduleIndex(
                        "batch",
                        f"(output_block * {block} + {slot}) * {self.reduction_steps} "
                        "+ reduction_step",
                    )
                ]
                guarded.extend(
                    call(instruction, overrides, batches)
                    for instruction in slot_ingress + compute
                )
                reduction_body.append(
                    ScheduleGuard(slot, block, self.output_batches, tuple(guarded))
                )
            block_body.append(
                ScheduleLoop(
                    "reduction_step",
                    "reduction_step",
                    self.reduction_steps,
                    tuple(reduction_body),
                )
            )
            for slot in range(block):
                overrides, batches = slot_substitution(slot)
                block_body.append(
                    ScheduleGuard(
                        slot,
                        block,
                        self.output_batches,
                        tuple(
                            call(instruction, overrides, batches)
                            for instruction in egress
                        ),
                    )
                )
            body.append(
                ScheduleLoop(
                    "output_block", "output_block", output_blocks, tuple(block_body)
                )
            )
            return EmissionSchedule(tuple(steps) + tuple(body) + tuple(epilogue))

        batch_body: list = []
        if spatial_elements is None:
            batch_body.extend(call(instruction) for instruction in outer_ingress)
        reduction_body = [
            ScheduleIndex(
                "batch", f"output_batch * {self.reduction_steps} + reduction_step"
            )
        ]
        output = next(
            (value.name for value in self.values if value.intent in {"out", "inout"}),
            None,
        )
        if spatial_elements is not None:
            reduction_body.append(ScheduleReset(bindings[output].c_name))
        reduction_body.extend(
            call(instruction) for instruction in inner_ingress + compute
        )
        if spatial_elements is not None:
            group_size = _plan_group_size(self.plan)
            reduction_body.append(
                ScheduleScatter(
                    output,
                    bindings[output].c_name,
                    f"output_batch * {VR_LANES} + reduction_step * {spatial_elements}",
                    spatial_elements,
                    group_size,
                )
            )
        batch_body.append(
            ScheduleLoop(
                "reduction_step",
                "reduction_step",
                self.reduction_steps,
                tuple(reduction_body),
            )
        )
        if spatial_elements is None:
            batch_body.extend(call(instruction) for instruction in egress)
        body.append(
            ScheduleLoop(
                "output_batch", "output_batch", self.output_batches, tuple(batch_body)
            )
        )
        return EmissionSchedule(tuple(steps) + tuple(body) + tuple(epilogue))

    def device_source(self) -> str:
        """The device translation unit: the emission schedule rendered."""
        return self.emission_schedule().render(self)

    def execute_numpy(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute high-level neutral operation semantics without board access."""

        state = {name: np.asarray(value).copy() for name, value in arrays.items()}
        for op in self.operations:
            if op.opcode in {"ADD_F16", "ADD_S16", "ADD_U16"}:
                state[op.output] = (state[op.inputs[0]] + state[op.inputs[1]]).astype(
                    state[op.inputs[0]].dtype
                )
            elif op.opcode in {"SUB_S16", "SUB_U16"}:
                state[op.output] = (state[op.inputs[0]] - state[op.inputs[1]]).astype(
                    state[op.inputs[0]].dtype
                )
            elif op.opcode == "MUL_U16":
                state[op.output] = (state[op.inputs[0]] * state[op.inputs[1]]).astype(
                    np.uint16
                )
            elif op.opcode == "MUL_F16":
                state[op.output] = state[op.inputs[0]] * state[op.inputs[1]]
            elif op.opcode == "XOR_16":
                state[op.output] = np.bitwise_xor(
                    state[op.inputs[0]], state[op.inputs[1]]
                )
            elif op.opcode == "AND_16":
                state[op.output] = np.bitwise_and(
                    state[op.inputs[0]], state[op.inputs[1]]
                )
            elif op.opcode == "OR_16":
                state[op.output] = np.bitwise_or(
                    state[op.inputs[0]], state[op.inputs[1]]
                )
            elif op.opcode == "NOT_16":
                state[op.output] = np.bitwise_not(state[op.inputs[0]])
            elif op.opcode == "POPCOUNT_16":
                source = np.asarray(state[op.inputs[0]], dtype=np.uint16)
                state[op.output] = (
                    np.unpackbits(source.view(np.uint8))
                    .reshape(source.size, 16)
                    .sum(axis=1)
                    .reshape(source.shape)
                    .astype(np.int16)
                )
            elif op.opcode == "SHL_IMM_16":
                state[op.output] = np.left_shift(
                    state[op.inputs[0]], int(op.attrs.get("shift", 1))
                )
            elif op.opcode == "CPY_IMM_16":
                exemplar = next(iter(state.values()))
                state[op.output] = np.full_like(exemplar, int(op.attrs["value"]))
            elif op.opcode == "RESET_16":
                exemplar = next(iter(state.values()))
                state[op.output] = np.zeros_like(exemplar)
            elif op.opcode in {
                "GROUP_REDUCE_F16",
                "GROUP_REDUCE_S16",
                "GROUP_REDUCE_U16",
            }:
                source = np.asarray(state[op.inputs[0]])
                group = int(op.attrs["group_size"])
                flat = source.reshape(-1)
                if flat.size % group:
                    raise ValueError(
                        f"NumPy group reduction size {flat.size} is not divisible by {group}"
                    )
                reduced = flat.reshape(-1, group).sum(axis=1)
                if op.opcode == "GROUP_REDUCE_U16":
                    reduced = reduced.astype(np.uint16)
                elif op.opcode == "GROUP_REDUCE_S16":
                    reduced = reduced.astype(np.int16)
                elif source.dtype == np.float16:
                    reduced = reduced.astype(np.float16)
                state[op.output] = reduced
            elif op.opcode in {
                "LOOKUP_16",
                "CREATE_GROUP_INDEX",
                "CREATE_SUBGROUP_INDEX",
                "DUPLICATE_SUBGROUP",
            }:
                # Layout/transfer primitives are already modeled by ABI.pack.
                continue
            else:
                raise UnsupportedVectorOperation(
                    f"no NumPy semantics for {op.opcode!r}"
                )
        return {
            value.name: np.asarray(state[value.name])
            for value in self.values
            if value.intent in {"out", "inout"} and value.name in state
        }


def realize_apu_v1_plan(
    plan,
    *,
    operations: Sequence[VectorOp | Mapping[str, object]] | None = None,
    values: (
        Sequence[VectorValue | Mapping[str, object]] | Mapping[str, object] | None
    ) = None,
    vr_capacity: int = WRITABLE_VRS,
) -> APUVectorRealization:
    """Lower one declarative APU v1 plan to GVML, allocation, and NumPy ABI."""

    normalized_values = _normalize_values(plan, values)
    normalized_operations = _normalize_operations(plan, operations)
    if not normalized_operations:
        raise UnsupportedVectorOperation(
            "APU vector plan contains no realizable compute operations"
        )
    physical_emission, instructions, transfer_records, output_names = (
        _instruction_stream(plan, normalized_values, normalized_operations)
    )
    bindings = _allocate_vrs(plan, instructions, capacity=int(vr_capacity))
    bindings = _reserve_accumulator_block(
        plan,
        bindings,
        output_names,
        capacity=int(vr_capacity),
    )
    abi = APUVectorABI(plan, normalized_values, physical_emission)
    return APUVectorRealization(
        plan=plan,
        values=normalized_values,
        operations=normalized_operations,
        instructions=instructions,
        vr_bindings=bindings,
        transfers=transfer_records,
        physical_emission=physical_emission,
        abi=abi,
    )


__all__ = [
    "APUC_COUNT",
    "APULayoutPackingError",
    "APUVectorComputeInvocation",
    "APUVectorABI",
    "APUVectorRealization",
    "GVMLInstruction",
    "OPCODES",
    "PackedVector",
    "PhysicalEmissionDescriptor",
    "LoopNestEmission",
    "SpatialResidentReductionEmission",
    "TransferRecord",
    "UnsupportedVectorOperation",
    "VRBinding",
    "VRCapacityError",
    "VR_LANES",
    "VectorOp",
    "VectorValue",
    "binary_matmul_ops",
    "fp16_contraction_ops",
    "realize_apu_v1_plan",
    "uint16_contraction_ops",
    "xnor_popcount_ops",
]
