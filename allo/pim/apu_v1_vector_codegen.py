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


def _explicit_route_instructions(plan, transfer, transfer_index, current_vm):
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
        lookup_bytes = table_size * temporal_extent * physical_batches * 2
        if lookup_bytes > L3_LOOKUP_BUDGET_BYTES:
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
                        "pointer": PointerRef(name, True, "L3", table_size),
                        "table_size": table_size,
                    },
                ),
                transfer_index,
            )
        )
        ingress[-1] = GVMLInstruction(
            **{**ingress[-1].__dict__, "phase": "transfer_in"}
        )
        return ingress, ("dma_l4_l3", "create_group_index", "lookup_16")

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
            _tmp(f"{name}_resident", index)
            for index in range(allocated_resident_count)
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


def _transfer_instructions(plan, value_names: set[str]):
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
                plan, transfer, transfer_index, current_vm
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


class APUVectorABI:
    """Layout-driven NumPy pack/gather ABI for one realized plan."""

    def __init__(self, plan, values: Sequence[VectorValue]):
        self.plan = plan
        self.values = tuple(values)
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

        destination = _route(transfer)[-1].destination
        layout = destination.layout
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
class APUVectorRealization:
    plan: object
    values: tuple[VectorValue, ...]
    operations: tuple[VectorOp, ...]
    instructions: tuple[GVMLInstruction, ...]
    vr_bindings: tuple[VRBinding, ...]
    transfers: tuple[TransferRecord, ...]
    abi: APUVectorABI

    @property
    def binding_map(self) -> dict[str, VRBinding]:
        return {binding.logical_name: binding for binding in self.vr_bindings}

    @property
    def vector_batches(self) -> int:
        metadata = _metadata(self.plan)
        metadata_batches = int(
            metadata.get("vr_batches", metadata.get("vector_batches", 1))
        )
        temporal = int(getattr(self.plan, "temporal_extent", 1))
        return max(1, metadata_batches, temporal)

    @property
    def output_batches(self) -> int:
        batching = getattr(self.plan, "output_batching", None)
        if batching is not None:
            return max(1, int(batching.physical_output_batches))
        return max(1, int(_metadata(self.plan).get("output_tiles", 1)))

    @property
    def reduction_steps(self) -> int:
        batching = getattr(self.plan, "output_batching", None)
        if batching is not None:
            return max(1, int(batching.work_steps_per_output_batch))
        return max(
            1,
            int(
                _metadata(self.plan).get(
                    "temporal_steps", self.vector_batches // self.output_batches
                )
            ),
        )

    @property
    def compute_tiles_per_output(self) -> int:
        batching = getattr(self.plan, "output_batching", None)
        if batching is None:
            return 1
        return max(1, int(batching.work_tiles_per_output_batch))

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

    def device_source(self) -> str:
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
        lines = [
            "#include <stdint.h>",
            "#include <gsi/libgvml_memory.h>",
            "#include <gsi/libgvml_element_wise.h>",
            "#include <gsi/libgvml_iv.h>",
            '#include "gsi_dma.h"',
            "",
            f"static int {function}_vector({params}) {{",
            "    gvml_init_once();",
        ]
        for binding in self.vr_bindings:
            lines.append(
                f"    enum gvml_vr16 {binding.c_name} = {binding.concrete_name};"
            )
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

        lines.extend(
            "    " + instruction.render(bindings, pointer_batches)
            for instruction in resident_setup + index_setup
        )

        spatial_elements = self._spatial_scatter_elements()
        lines.append(
            f"    for (uint32_t output_batch = 0; output_batch < {self.output_batches}; "
            "++output_batch) {"
        )
        if spatial_elements is None:
            lines.extend(
                "        " + instruction.render(bindings, pointer_batches)
                for instruction in outer_ingress
            )
        lines.append(
            f"        for (uint32_t reduction_step = 0; reduction_step < "
            f"{self.reduction_steps}; ++reduction_step) {{"
        )
        lines.append(
            f"            const uint32_t batch = output_batch * {self.reduction_steps} "
            "+ reduction_step;"
        )
        if spatial_elements is not None:
            output = next(
                value.name for value in self.values if value.intent in {"out", "inout"}
            )
            lines.append(f"            gvml_reset_16({bindings[output].c_name});")
        lines.extend(
            "            " + instruction.render(bindings, pointer_batches)
            for instruction in inner_ingress + compute
        )
        if spatial_elements is not None:
            output = next(
                value.name for value in self.values if value.intent in {"out", "inout"}
            )
            group_size = _plan_group_size(self.plan)
            lines.extend(
                [
                    f"            for (uint32_t scatter = 0; scatter < {spatial_elements}; ++scatter)",
                    f"                {output}_L4ptr[output_batch * {VR_LANES} + "
                    f"reduction_step * {spatial_elements} + scatter] = "
                    f"gvml_get_entry_16({bindings[output].c_name}, scatter * {group_size});",
                ]
            )
        lines.append("        }")
        if spatial_elements is None:
            lines.extend(
                "        " + instruction.render(bindings, pointer_batches)
                for instruction in egress
            )
        lines.append("    }")
        lines.extend(["    return 0;", "}", ""])
        return "\n".join(lines)

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
    value_names = {value.name for value in normalized_values}
    ingress, egress, transfer_records = _transfer_instructions(plan, value_names)
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
    instructions = tuple(ingress + compute + egress)
    bindings = _allocate_vrs(plan, instructions, capacity=int(vr_capacity))
    abi = APUVectorABI(plan, normalized_values)
    return APUVectorRealization(
        plan,
        normalized_values,
        normalized_operations,
        instructions,
        bindings,
        transfer_records,
        abi,
    )


__all__ = [
    "APUC_COUNT",
    "APULayoutPackingError",
    "APUVectorABI",
    "APUVectorRealization",
    "GVMLInstruction",
    "OPCODES",
    "PackedVector",
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
