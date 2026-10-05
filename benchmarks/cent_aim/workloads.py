# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configurable CENT-equivalent decode workloads for SK hynix GDDR6-AiM.

This module describes tensor work and data residency.  It does not contain
case- or shape-specific compiler decisions: the fourteen benchmark names at
the bottom only select ordered, reusable stage builders.  Static weights are
assumed resident, exactly as in CENT's ``--only-trace`` measurements, while
all dynamic host transfers remain explicit typed operations.

CENT runs four independent transformer-block replicas on one 32-channel
device.  The default :class:`DecodeSpec` therefore uses four disjoint groups
of eight channels, but every helper derives its work from the supplied model
and replica dimensions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from allo.pim.aim_program import (
    AimAllBankWrite,
    AimBankCopy,
    AimContraction,
    AimDistributedHostTransfer,
    AimElementwise,
    AimHostTransfer,
    AimOp,
    AimProgram,
    AimSync,
)


# Trace operands that describe work shape, as opposed to placement.  Row,
# bank, channel, and GPR addresses are intentionally absent: two legal linear
# layouts may move the same logical tensor work without using the same
# addresses or command order.  A raw MEM request is one vector burst to one
# explicit channel, hence its implicit active-channel fanout is one.
_COMMAND_SHAPE_FIELDS: Mapping[str, tuple[int | None, int | None]] = {
    "AiM_WR_GB": (2, 4),
    "AiM_WR_BIAS": (None, 3),
    "AiM_MAC_ABK": (2, 3),
    "AiM_RD_MAC": (None, 3),
    "AiM_AF": (None, 2),
    "AiM_RD_AF": (None, 3),
    "AiM_WR_ABK": (None, 3),
    "AiM_EWMUL": (2, 3),
    "AiM_EWADD": (2, None),
    "AiM_COPY_BKGB": (2, 3),
    "AiM_COPY_GBBK": (2, 3),
    "AiM_SYNC": (None, None),
    "AiM_EOC": (None, None),
    "R_MEM": (None, None),
    "W_MEM": (None, None),
}


def _trace_opcode(words: list[str]) -> str:
    if words[0] == "AiM" and len(words) >= 2:
        return f"AiM_{words[1]}"
    if len(words) >= 2:
        return f"{words[0]}_{words[1]}"
    return words[0]


def _command_shape(raw_line: str) -> tuple[str, int | None, int | None] | None:
    words = raw_line.split()
    if not words or words[0].startswith("#"):
        return None
    opcode = _trace_opcode(words)
    try:
        op_size_index, mask_index = _COMMAND_SHAPE_FIELDS[opcode]
    except KeyError as error:
        raise ValueError(
            f"cannot fingerprint unknown trace opcode {opcode!r}"
        ) from error
    try:
        op_size = None if op_size_index is None else int(words[op_size_index], 0)
        if opcode in {"R_MEM", "W_MEM"}:
            active_mask_fanout = 1
        elif mask_index is None:
            active_mask_fanout = None
        else:
            active_mask_fanout = int(words[mask_index], 0).bit_count()
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed {opcode} trace record: {raw_line!r}") from error
    if op_size is not None and op_size <= 0:
        raise ValueError(f"{opcode} has nonpositive op_size {op_size}")
    if active_mask_fanout is not None and active_mask_fanout <= 0:
        raise ValueError(
            f"{opcode} has nonpositive active-mask fanout {active_mask_fanout}"
        )
    return opcode, op_size, active_mask_fanout


def command_shape_signature(trace_text: str) -> dict[str, Any]:
    """Fingerprint command shapes while excluding placement and order.

    The histogram retains every opcode's operation size, active-mask popcount,
    and multiplicity.  It is intentionally more discriminating than an opcode
    inventory but less restrictive than byte-identical traces: address-only
    linear-layout changes disappear, while a tail-width or mask-fanout change
    remains visible.
    """

    histogram: Counter[tuple[str, int | None, int | None]] = Counter()
    for raw_line in trace_text.splitlines():
        shape = _command_shape(raw_line)
        if shape is not None:
            histogram[shape] += 1
    if not histogram:
        raise ValueError("cannot fingerprint an empty trace")
    records = []
    for (opcode, op_size, fanout), count in sorted(
        histogram.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
            -1 if item[0][2] is None else item[0][2],
        ),
    ):
        records.append(
            {
                "opcode": opcode,
                "op_size": op_size,
                "active_mask_fanout": fanout,
                "command_count": count,
            }
        )
    return {
        "schema": "tenon-aim-command-shape-signature-v1",
        "addresses_excluded": True,
        "command_order_excluded": True,
        "record_fields": [
            "opcode",
            "op_size",
            "active_mask_fanout",
            "command_count",
        ],
        "command_count": sum(histogram.values()),
        "records": records,
    }


def compiler_logical_work_coverage(
    compiled: Mapping[str, Any], trace_text: str
) -> dict[str, Any]:
    """Reconcile typed operations and logical contractions with lowering.

    This proof is deliberately independent from physical-signature equality.
    It requires one lowering record for every typed operation, gap-free command
    spans covering the complete trace body, and an exact compiler-declared
    logical scalar-MAC count for every contraction.
    """

    source = compiled.get("source")
    source_operations = (
        source.get("operations") if isinstance(source, Mapping) else None
    )
    lowered_operations = compiled.get("operations")
    trace = compiled.get("trace")
    errors: list[str] = []
    if not isinstance(source_operations, list):
        source_operations = []
        errors.append("compiled source manifest has no operation list")
    if not isinstance(lowered_operations, list):
        lowered_operations = []
        errors.append("compiled manifest has no lowered operation list")
    try:
        body_commands = int(trace["body_command_count"])
    except (KeyError, TypeError, ValueError):
        body_commands = -1
        errors.append("compiled trace has no body_command_count")
    trace_records = [
        line.strip()
        for line in trace_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not trace_records or trace_records[-1] != "AiM EOC":
        errors.append("materialized trace has no trailing EOC")
        body_records: list[str] = []
    else:
        body_records = trace_records[:-1]
    if len(body_records) != body_commands:
        errors.append("materialized trace body length differs from compiled manifest")
    geometry = compiled.get("geometry")
    try:
        lanes = int(geometry["lanes"])
        banks = int(geometry["banks"])
        bank_groups = int(geometry["bank_groups"])
    except (KeyError, TypeError, ValueError):
        lanes = banks = bank_groups = 0
        errors.append("compiled target lane/bank geometry is incomplete")

    covered_operations = 0
    cursor = 0
    operation_coverage = []
    logical_contractions = []
    logical_elementwise = []
    expected_scalar_macs = 0
    compiled_scalar_macs = 0
    for index, source_operation in enumerate(source_operations):
        if index >= len(lowered_operations):
            errors.append(f"typed operation {index} has no lowering record")
            continue
        lowered = lowered_operations[index]
        operation_errors = []
        ownership_coverage = None
        elementwise_coverage = None
        if lowered.get("index") != index:
            operation_errors.append("lowering index differs")
        for field in ("kind", "name"):
            if lowered.get(field) != source_operation.get(field):
                operation_errors.append(f"{field} differs")
        span = lowered.get("command_span")
        operation_signature = None
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
        ):
            operation_errors.append("command span is malformed")
        else:
            begin, end = span
            if begin != cursor or end < begin:
                operation_errors.append("command spans are not contiguous")
            if lowered.get("command_count") != end - begin:
                operation_errors.append("command count differs from span")
            if 0 <= begin <= end <= len(body_records) and end > begin:
                operation_signature = command_shape_signature(
                    "\n".join(body_records[begin:end]) + "\n"
                )
            else:
                operation_errors.append("command span exceeds materialized trace")
            cursor = max(cursor, end)

        if source_operation.get("kind") == "contraction":
            requested_batch_mapping = source_operation.get("batch_mapping", "flattened")
            effective_batch_mapping = lowered.get("batch_mapping", "flattened")
            selection = lowered.get("batch_mapping_selection")
            if requested_batch_mapping == "auto":
                if (
                    lowered.get("requested_batch_mapping") != "auto"
                    or not isinstance(selection, Mapping)
                    or selection.get("policy") != "shape_and_residency_v1"
                    or selection.get("selected") != effective_batch_mapping
                    or selection.get("uses_operation_name") is not False
                    or not selection.get("reason")
                ):
                    operation_errors.append(
                        "automatic batch-mapping selection evidence is incomplete"
                    )
            try:
                expected = (
                    int(source_operation["outputs"])
                    * int(source_operation["reduction"])
                    * int(source_operation.get("batches", 1))
                    * int(source_operation.get("replicas", 1))
                )
                declared_logical = int(lowered["logical_scalar_macs"])
            except (KeyError, TypeError, ValueError):
                expected = 0
                declared_logical = -1
                operation_errors.append("logical scalar-MAC declaration is incomplete")
            if expected <= 0 or declared_logical != expected:
                operation_errors.append("logical scalar-MAC coverage differs")

            physical_mac_slots = 0
            physical_wr_gb_elements = 0
            if operation_signature is not None and lanes > 0 and banks > 0:
                for record in operation_signature["records"]:
                    op_size = record["op_size"]
                    fanout = record["active_mask_fanout"]
                    count = record["command_count"]
                    if record["opcode"] == "AiM_MAC_ABK":
                        if op_size is None or fanout is None:
                            operation_errors.append(
                                "MAC shape fingerprint is incomplete"
                            )
                        else:
                            physical_mac_slots += (
                                op_size * lanes * fanout * banks * count
                            )
                    elif record["opcode"] == "AiM_WR_GB":
                        if op_size is None or fanout is None:
                            operation_errors.append(
                                "WR_GB shape fingerprint is incomplete"
                            )
                        else:
                            physical_wr_gb_elements += op_size * lanes * fanout * count
            if physical_mac_slots < expected:
                operation_errors.append(
                    "materialized MAC capacity does not cover logical scalar MACs"
                )
            expected_unique_gb = (
                int(source_operation["reduction"])
                * int(source_operation.get("batches", 1))
                * int(source_operation.get("replicas", 1))
                if source_operation.get("input_source", "gb") == "gb"
                else 0
            )
            declarations = {
                "physical_mac_slots": physical_mac_slots,
                "physical_padded_scalar_macs": physical_mac_slots,
                "unique_logical_gb_payload_elements": expected_unique_gb,
                "physical_wr_gb_elements": physical_wr_gb_elements,
                "physical_gb_transfer_elements": physical_wr_gb_elements,
            }
            for field, derived in declarations.items():
                try:
                    declared = int(lowered[field])
                except (KeyError, TypeError, ValueError):
                    operation_errors.append(
                        f"compiler declaration {field} is incomplete"
                    )
                    continue
                if declared != derived:
                    operation_errors.append(
                        f"compiler declaration {field} differs from trace-derived work"
                    )
            expected_factor = (
                physical_wr_gb_elements / expected_unique_gb
                if expected_unique_gb
                else 0.0
            )
            try:
                declared_factor = float(lowered["gb_replication_and_reload_factor"])
            except (KeyError, TypeError, ValueError):
                operation_errors.append(
                    "compiler declaration gb_replication_and_reload_factor is incomplete"
                )
                declared_factor = -1.0
            if abs(declared_factor - expected_factor) > 1e-12:
                operation_errors.append(
                    "compiler GB replication/reload factor differs from trace"
                )
            expected_scalar_macs += expected
            compiled_scalar_macs += max(0, declared_logical)
            logical_contractions.append(
                {
                    "index": index,
                    "name": source_operation.get("name"),
                    "requested_batch_mapping": requested_batch_mapping,
                    "effective_batch_mapping": effective_batch_mapping,
                    "batch_mapping_selection": selection,
                    "expected_scalar_macs": expected,
                    "compiled_scalar_macs": declared_logical,
                    "trace_derived_physical_mac_slots": physical_mac_slots,
                    "trace_derived_unique_logical_gb_payload_elements": (
                        expected_unique_gb
                    ),
                    "trace_derived_physical_wr_gb_elements": (physical_wr_gb_elements),
                    "physical_to_logical_mac_ratio": (
                        physical_mac_slots / expected if expected > 0 else None
                    ),
                    "gb_replication_and_reload_factor": expected_factor,
                    "command_shape_signature": operation_signature,
                    "complete": not operation_errors,
                }
            )

        if source_operation.get("kind") == "elementwise":
            elementwise_kind = source_operation.get("operation")
            try:
                expected_elements = int(source_operation["elements"])
                if elementwise_kind == "mul":
                    expected_elements *= int(source_operation.get("replicas", 1))
            except (KeyError, TypeError, ValueError):
                expected_elements = 0
                operation_errors.append("logical elementwise work is incomplete")
            physical_slots = 0
            expected_opcode = "AiM_EWMUL" if elementwise_kind == "mul" else "AiM_EWADD"
            if operation_signature is not None and lanes > 0:
                for record in operation_signature["records"]:
                    if record["opcode"] != expected_opcode:
                        continue
                    op_size = record["op_size"]
                    count = record["command_count"]
                    if op_size is None:
                        operation_errors.append(
                            "elementwise shape fingerprint has no op_size"
                        )
                        continue
                    if elementwise_kind == "mul":
                        fanout = record["active_mask_fanout"]
                        if fanout is None or bank_groups <= 0:
                            operation_errors.append(
                                "EWMUL shape fingerprint has no physical fanout"
                            )
                            continue
                        physical_slots += op_size * lanes * fanout * bank_groups * count
                    else:
                        physical_slots += op_size * lanes * count
            if physical_slots < expected_elements:
                operation_errors.append(
                    "materialized elementwise capacity does not cover logical elements"
                )
            for field, derived in (
                ("logical_elements", expected_elements),
                ("physical_element_slots", physical_slots),
            ):
                try:
                    declared = int(lowered[field])
                except (KeyError, TypeError, ValueError):
                    operation_errors.append(
                        f"compiler declaration {field} is incomplete"
                    )
                    continue
                if declared != derived:
                    operation_errors.append(
                        f"compiler declaration {field} differs from trace-derived work"
                    )
            elementwise_coverage = {
                "operation": elementwise_kind,
                "expected_logical_elements": expected_elements,
                "trace_derived_physical_element_slots": physical_slots,
                "physical_to_logical_ratio": (
                    physical_slots / expected_elements
                    if expected_elements > 0
                    else None
                ),
                "complete": not operation_errors,
            }
            logical_elementwise.append(
                {
                    "index": index,
                    "name": source_operation.get("name"),
                    **elementwise_coverage,
                }
            )

        if source_operation.get("kind") == "allbankwrite":
            target_channels = 0
            try:
                target_channels = int(geometry["channels"])
                requested_channels = source_operation.get("channels")
                expected_channels = (
                    list(range(target_channels))
                    if requested_channels is None
                    else [int(channel) for channel in requested_channels]
                )
                row = int(source_operation["row"])
                rows = int(source_operation.get("rows", 1))
                row_stride = int(source_operation.get("row_stride", 1))
                copies = int(source_operation.get("copies", 1))
                copy_stride_value = source_operation.get("copy_row_stride")
                copy_stride = (
                    rows * row_stride
                    if copy_stride_value is None
                    else int(copy_stride_value)
                )
                expected_rows = [
                    row + copy * copy_stride + row_index * row_stride
                    for copy in range(copies)
                    for row_index in range(rows)
                ]
                expected_coordinates = {
                    (channel, physical_row)
                    for physical_row in expected_rows
                    for channel in expected_channels
                }
            except (KeyError, TypeError, ValueError):
                expected_channels = []
                expected_rows = []
                expected_coordinates = set()
                operation_errors.append("all-bank-write source ownership is incomplete")
            actual_coordinates = []
            if isinstance(span, list) and len(span) == 2:
                begin, end = span
                for raw_line in body_records[max(0, begin) : max(0, end)]:
                    words = raw_line.split()
                    if _trace_opcode(words) != "AiM_WR_ABK":
                        operation_errors.append(
                            "all-bank-write span contains another opcode"
                        )
                        continue
                    try:
                        mask = int(words[3], 0)
                        physical_row = int(words[4], 0)
                    except (IndexError, ValueError):
                        operation_errors.append("malformed WR_ABK ownership record")
                        continue
                    for channel in range(target_channels):
                        if mask & (1 << (target_channels - 1 - channel)):
                            actual_coordinates.append((channel, physical_row))
            actual_coordinate_set = set(actual_coordinates)
            ownership_complete = (
                bool(expected_coordinates)
                and len(actual_coordinates) == len(expected_coordinates)
                and actual_coordinate_set == expected_coordinates
            )
            if not ownership_complete:
                operation_errors.append(
                    "all-bank-write trace does not cover every typed ownership coordinate"
                )
            ownership_coverage = {
                "schema": "tenon-aim-all-bank-write-ownership-v1",
                "expected_channels": expected_channels,
                "expected_rows": expected_rows,
                "expected_coordinate_count": len(expected_coordinates),
                "materialized_coordinate_count": len(actual_coordinates),
                "unique_materialized_coordinate_count": len(actual_coordinate_set),
                "complete": ownership_complete,
            }

        if operation_errors:
            errors.extend(
                f"typed operation {index}: {message}" for message in operation_errors
            )
        else:
            covered_operations += 1
        operation_coverage.append(
            {
                "index": index,
                "kind": source_operation.get("kind"),
                "name": source_operation.get("name"),
                "command_shape_signature": operation_signature,
                "ownership_coverage": ownership_coverage,
                "elementwise_coverage": elementwise_coverage,
                "complete": not operation_errors,
            }
        )

    if len(lowered_operations) > len(source_operations):
        errors.append("compiled manifest has lowering records without typed operations")
    if cursor != body_commands:
        errors.append(
            f"lowered command spans cover {cursor} body commands, expected {body_commands}"
        )
    return {
        "schema": "tenon-aim-logical-work-coverage-v1",
        "source_operation_count": len(source_operations),
        "lowered_operation_count": len(lowered_operations),
        "covered_operation_count": covered_operations,
        "body_command_count": body_commands,
        "covered_body_command_count": cursor,
        "expected_logical_scalar_macs": expected_scalar_macs,
        "compiled_logical_scalar_macs": compiled_scalar_macs,
        "operations": operation_coverage,
        "contractions": logical_contractions,
        "elementwise": logical_elementwise,
        "complete": not errors,
        "errors": errors,
    }


# Fixed properties of the SK hynix AiM target, not workload tuning knobs.
TARGET_CHANNELS = 32
BANKS_PER_CHANNEL = 16
BANK_GROUPS_PER_CHANNEL = 4
ROW_ELEMENTS = 1024
VECTOR_LANES = 16
TARGET_ROWS = 16384


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class DecodeSpec:
    """Shape and replica contract for one autoregressive decode step."""

    D: int
    H: int
    Dh: int
    F: int
    L: int
    replicas: int
    channels_per_replica: int
    max_seq_len: int

    def __post_init__(self):
        for field in (
            "D",
            "H",
            "Dh",
            "F",
            "L",
            "replicas",
            "channels_per_replica",
            "max_seq_len",
        ):
            value = int(getattr(self, field))
            if value <= 0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        if self.D != self.H * self.Dh:
            raise ValueError("D must equal H * Dh for this dense-attention mapping")
        if self.L > self.max_seq_len:
            raise ValueError("L cannot exceed max_seq_len")
        if self.replicas * self.channels_per_replica > TARGET_CHANNELS:
            raise ValueError("replica channel groups exceed the 32-channel target")
        if self.H % self.channels_per_replica:
            raise ValueError("H must be divisible by channels_per_replica")
        if self.Dh % BANKS_PER_CHANNEL:
            raise ValueError("Dh must be divisible by the target bank count")

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(range(self.replicas * self.channels_per_replica))

    @property
    def channel_groups(self) -> tuple[tuple[int, ...], ...]:
        width = self.channels_per_replica
        return tuple(
            tuple(range(replica * width, (replica + 1) * width))
            for replica in range(self.replicas)
        )


DEFAULT_DECODE_SPEC = DecodeSpec(
    D=4096,
    H=32,
    Dh=128,
    F=11008,
    L=128,
    replicas=4,
    channels_per_replica=8,
    max_seq_len=4096,
)


@dataclass(frozen=True)
class RowRegion:
    """A non-overlapping region in each bank's row address space."""

    name: str
    row: int
    rows: int
    residency: str


@dataclass(frozen=True)
class DecodeLayout:
    """Shape-derived row allocation shared by all benchmark stage subsets."""

    regions: Mapping[str, RowRegion]
    used_rows: int

    def __post_init__(self):
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))

    def __getitem__(self, name: str) -> RowRegion:
        return self.regions[name]


class _RowAllocator:
    def __init__(self):
        self.next_row = 0
        self.regions: dict[str, RowRegion] = {}

    def allocate(self, name: str, rows: int, residency: str) -> RowRegion:
        rows = max(1, int(rows))
        region = RowRegion(name, self.next_row, rows, residency)
        self.regions[name] = region
        self.next_row += rows
        return region


def _distributed_rows(spec: DecodeSpec, elements: int, bank_stride: int) -> int:
    partitions = spec.channels_per_replica * (BANKS_PER_CHANNEL // bank_stride)
    elements_per_partition = _ceil_div(elements, partitions)
    return _ceil_div(elements_per_partition, ROW_ELEMENTS)


def _contraction_rows(
    spec: DecodeSpec,
    outputs: int,
    reduction: int,
    *,
    batches: int = 1,
) -> int:
    capacity = len(spec.channels) * BANKS_PER_CHANNEL
    launches = _ceil_div(outputs * batches * spec.replicas, capacity)
    return launches * _ceil_div(reduction, ROW_ELEMENTS)


def _qk_batch_pack(spec: DecodeSpec) -> int:
    """Heads whose aligned reductions fit together in one K-cache row."""

    aligned_reduction = _ceil_div(spec.Dh, VECTOR_LANES) * VECTOR_LANES
    row_pack = ROW_ELEMENTS // aligned_reduction
    if row_pack <= 0:
        raise ValueError(
            "QK row packing requires an aligned head reduction to fit one row"
        )
    return min(spec.H, row_pack)


def _qk_batch_groups(spec: DecodeSpec) -> int:
    return _ceil_div(spec.H, _qk_batch_pack(spec))


def _sv_batch_groups(spec: DecodeSpec) -> int:
    """Head groups processed by one channel slice per replica."""

    return _ceil_div(spec.H, spec.channels_per_replica)


def _qk_output_tiles(spec: DecodeSpec, sequence_extent: int) -> int:
    """Sequence tiles distributed over the target's bank frontier."""

    return _ceil_div(
        sequence_extent,
        spec.channels_per_replica * BANKS_PER_CHANNEL,
    )


def _qk_cache_rows(spec: DecodeSpec) -> int:
    # Row-packed QK stores one row for every (head group, sequence tile).
    return _qk_batch_groups(spec) * _qk_output_tiles(spec, spec.max_seq_len)


def _sv_cache_rows(spec: DecodeSpec) -> int:
    # Channel-batched SV maps one head to each local channel, dimensions to
    # banks, and reserves the complete max-sequence reduction stride.
    batch_groups = _sv_batch_groups(spec)
    output_tiles = _ceil_div(spec.Dh, BANKS_PER_CHANNEL)
    reduction_storage_rows = _ceil_div(spec.max_seq_len, ROW_ELEMENTS)
    return batch_groups * output_tiles * reduction_storage_rows


def build_layout(spec: DecodeSpec) -> DecodeLayout:
    """Allocate a complete decode block without depending on a case name."""

    allocator = _RowAllocator()
    neighbor_rows = _distributed_rows(spec, spec.D, bank_stride=2)
    group_rows = _distributed_rows(spec, spec.D, bank_stride=4)

    for prefix in ("x", "sa"):
        # The neighbor-bank source is dead after its partial MAC.  Reuse that
        # row for the host-produced scale/vector pair and first EWMUL, then
        # place the relocated norm-weight result in the adjacent live row.
        allocator.allocate(
            f"rms_{prefix}_work",
            max(neighbor_rows, group_rows),
            "dynamic",
        )
        allocator.allocate(f"rms_{prefix}_output", group_rows, "dynamic")
    allocator.allocate("rope_q", _distributed_rows(spec, 2 * spec.D, 4), "dynamic")
    allocator.allocate("rope_k", _distributed_rows(spec, 2 * spec.D, 4), "dynamic")
    allocator.allocate(
        "scores",
        _distributed_rows(spec, spec.H * spec.max_seq_len, 4),
        "dynamic",
    )
    allocator.allocate("ffn_activation", _distributed_rows(spec, spec.F, 4), "dynamic")

    allocator.allocate("k_cache", _qk_cache_rows(spec), "dynamic_cache")

    sequence_rows = _ceil_div(spec.max_seq_len, ROW_ELEMENTS)
    dimensions_per_bank = spec.Dh // BANKS_PER_CHANNEL
    heads_per_channel = spec.H // spec.channels_per_replica
    v_cache_append_rows = sequence_rows * dimensions_per_bank * heads_per_channel
    v_cache_contraction_rows = _sv_cache_rows(spec)
    if v_cache_append_rows != v_cache_contraction_rows:
        raise ValueError(
            "V-cache append and channel-batched contraction layouts disagree"
        )
    allocator.allocate(
        "v_cache",
        v_cache_contraction_rows,
        "dynamic_cache",
    )

    for name, outputs, reduction in (
        ("wq", spec.D, spec.D),
        ("wk", spec.D, spec.D),
        ("wv", spec.D, spec.D),
        ("wo", spec.D, spec.D),
        ("w1", spec.F, spec.D),
        ("w3", spec.F, spec.D),
        ("w2", spec.D, spec.F),
    ):
        allocator.allocate(
            name,
            _contraction_rows(spec, outputs, reduction),
            "resident_weight",
        )

    if allocator.next_row > TARGET_ROWS:
        raise ValueError(
            f"decode layout needs {allocator.next_row} rows, target has {TARGET_ROWS}"
        )
    return DecodeLayout(allocator.regions, allocator.next_row)


def distributed_transfer(
    spec: DecodeSpec,
    direction: str,
    elements: int,
    row: int,
    *,
    bank_stride: int = 4,
    bank_offset: int = 0,
    copies: int = 1,
    name: str,
) -> AimDistributedHostTransfer:
    """Create regular replica- and bank-group-aware dynamic host traffic."""

    return AimDistributedHostTransfer(
        direction,
        elements=elements,
        replicas=spec.replicas,
        channels_per_replica=spec.channels_per_replica,
        row=row,
        bank_stride=bank_stride,
        bank_offset=bank_offset,
        copies=copies,
        name=name,
    )


def contraction(
    spec: DecodeSpec,
    *,
    outputs: int,
    reduction: int,
    row: int,
    batches: int = 1,
    input_source: str = "gb",
    batch_mapping: str = "flattened",
    reduction_storage_extent: int | None = None,
    activation: bool = False,
    name: str,
) -> AimContraction:
    """Create a dense contraction across all disjoint replica channels."""

    return AimContraction(
        outputs=outputs,
        reduction=reduction,
        batches=batches,
        replicas=spec.replicas,
        row=row,
        channels=spec.channels,
        channels_per_replica=spec.channels_per_replica,
        input_source=input_source,
        batch_mapping=batch_mapping,
        reduction_storage_extent=reduction_storage_extent,
        activation=activation,
        name=name,
    )


def _paired_bank_group_copies(
    spec: DecodeSpec,
    *,
    elements: int,
    source_row: int,
    destination_row: int,
    name: str,
) -> tuple[AimOp, ...]:
    """Relocate four bank groups through their shared GB in safe pair order."""

    operations: list[AimOp] = []
    for source_bank in range(2, BANKS_PER_CHANNEL, BANK_GROUPS_PER_CHANNEL):
        operations.append(
            AimBankCopy(
                "bank_to_gb",
                elements=elements,
                bank=source_bank,
                row=source_row,
                replicas=spec.replicas,
                channels_per_replica=spec.channels_per_replica,
                partitions_per_replica=(
                    spec.channels_per_replica * BANK_GROUPS_PER_CHANNEL
                ),
                name=f"{name}.bank_{source_bank}.to_gb",
            )
        )
        operations.append(
            AimBankCopy(
                "gb_to_bank",
                elements=elements,
                bank=source_bank - 1,
                row=destination_row,
                replicas=spec.replicas,
                channels_per_replica=spec.channels_per_replica,
                partitions_per_replica=(
                    spec.channels_per_replica * BANK_GROUPS_PER_CHANNEL
                ),
                name=f"{name}.gb_to_bank_{source_bank - 1}",
            )
        )
    return tuple(operations)


def rms_device_piece(
    spec: DecodeSpec,
    layout: DecodeLayout,
    *,
    prefix: str,
) -> tuple[AimOp, ...]:
    """CENT's device-side RMS partial, scale, relocation, and readback."""

    source = layout[f"rms_{prefix}_work"]
    scaled = source
    output = layout[f"rms_{prefix}_output"]
    partial_reduction = _ceil_div(
        spec.D,
        spec.channels_per_replica * (BANKS_PER_CHANNEL // 2),
    )
    operations: list[AimOp] = [
        distributed_transfer(
            spec,
            "write",
            spec.D,
            source.row,
            bank_stride=2,
            copies=2,
            name=f"rms.{prefix}.dynamic_inputs",
        ),
        contraction(
            spec,
            outputs=spec.channels_per_replica * BANKS_PER_CHANNEL,
            reduction=partial_reduction,
            row=source.row,
            input_source="banks",
            name=f"rms.{prefix}.sum_of_squares_partial",
        ),
        distributed_transfer(
            spec,
            "write",
            spec.D,
            scaled.row,
            copies=2,
            name=f"rms.{prefix}.host_scale_and_vector",
        ),
        AimElementwise(
            "mul",
            elements=spec.D,
            row=scaled.row,
            replicas=spec.replicas,
            channels_per_replica=spec.channels_per_replica,
            name=f"rms.{prefix}.scale",
        ),
    ]
    operations.extend(
        _paired_bank_group_copies(
            spec,
            elements=spec.D,
            source_row=scaled.row,
            destination_row=output.row,
            name=f"rms.{prefix}.norm_weight_relocation",
        )
    )
    operations.extend(
        (
            AimElementwise(
                "mul",
                elements=spec.D,
                row=output.row,
                replicas=spec.replicas,
                channels_per_replica=spec.channels_per_replica,
                name=f"rms.{prefix}.norm_weight",
            ),
            distributed_transfer(
                spec,
                "read",
                spec.D,
                output.row,
                bank_offset=2,
                name=f"rms.{prefix}.result",
            ),
            AimSync(name=f"rms.{prefix}.host_split"),
        )
    )
    return tuple(operations)


def residual_device_piece(spec: DecodeSpec, *, name: str) -> tuple[AimOp, ...]:
    """One hidden-vector residual addition."""

    columns = _ceil_div(spec.D, VECTOR_LANES)
    return (
        AimElementwise(
            "add",
            elements=spec.D,
            gpr_addr_0=0,
            gpr_addr_1=columns,
            name=name,
        ),
    )


def projection_device_piece(
    spec: DecodeSpec,
    layout: DecodeLayout,
    *,
    weight: str,
    outputs: int,
    reduction: int,
    activation: bool = False,
) -> tuple[AimOp, ...]:
    """Apply one statically resident projection matrix."""

    return (
        contraction(
            spec,
            outputs=outputs,
            reduction=reduction,
            row=layout[weight].row,
            activation=activation,
            name=f"resident_weight.{weight}",
        ),
    )


def rope_device_piece(spec: DecodeSpec, layout: DecodeLayout) -> tuple[AimOp, ...]:
    """CENT's physical remap/EWMUL contract for Q and K rotary embedding.

    CENT emits the two post-EWMUL host-visible transfers through its store
    helper as ``W MEM`` records too.  We preserve that measured physical
    contract here; the semantic caveat is surfaced in :func:`semantic_manifest`.
    """

    operations: list[AimOp] = []
    for tensor in ("q", "k"):
        operations.append(
            distributed_transfer(
                spec,
                "write",
                2 * spec.D,
                layout[f"rope_{tensor}"].row,
                bank_offset=1,
                name=f"rope.{tensor}.remapped_inputs",
            )
        )
    for tensor in ("q", "k"):
        operations.append(
            AimElementwise(
                "mul",
                elements=spec.D,
                row=layout[f"rope_{tensor}"].row,
                replicas=spec.replicas,
                channels_per_replica=spec.channels_per_replica,
                name=f"rope.{tensor}.device_mul",
            )
        )
    for tensor in ("q", "k"):
        operations.append(
            distributed_transfer(
                spec,
                "write",
                2 * spec.D,
                layout[f"rope_{tensor}"].row,
                bank_offset=2,
                name=f"rope.{tensor}.cent_post_result_transfer",
            )
        )
    return tuple(operations)


def attention_cache_append(spec: DecodeSpec, layout: DecodeLayout) -> tuple[AimOp, ...]:
    """Append the current K and V token to their shape-derived cache layouts."""

    position = spec.L - 1
    tile = spec.channels_per_replica * BANKS_PER_CHANNEL
    position_tile, position_in_tile = divmod(position, tile)
    local_channel, bank = divmod(position_in_tile, BANKS_PER_CHANNEL)
    batch_pack = _qk_batch_pack(spec)
    batch_groups = _qk_batch_groups(spec)

    operations: list[AimOp] = []
    # Rows are the outer traversal so transfers to the same packed cache row
    # stay adjacent across replicas.  This is a general dependency-neutral
    # coalescing order, not a sequence-length-specific schedule.
    for batch_group in range(batch_groups):
        packed_heads = min(
            batch_pack,
            spec.H - batch_group * batch_pack,
        )
        aligned_reduction = _ceil_div(spec.Dh, VECTOR_LANES) * VECTOR_LANES
        key_bursts = packed_heads * aligned_reduction // VECTOR_LANES
        key_row = layout["k_cache"].row + position_tile * batch_groups + batch_group
        for replica in range(spec.replicas):
            channel = replica * spec.channels_per_replica + local_channel
            operations.append(
                AimHostTransfer(
                    "write",
                    channel=channel,
                    bank=bank,
                    row=key_row,
                    bursts=key_bursts,
                    name=(f"attention.k_append.group{batch_group}." f"r{replica}"),
                )
            )

    sequence_row = position // ROW_ELEMENTS
    rows_per_dimension = _ceil_div(spec.max_seq_len, ROW_ELEMENTS)
    dimension_iterations = spec.Dh // BANKS_PER_CHANNEL
    operations.append(
        AimAllBankWrite(
            row=layout["v_cache"].row + sequence_row,
            rows=dimension_iterations,
            row_stride=rows_per_dimension,
            copies=spec.H // spec.channels_per_replica,
            copy_row_stride=rows_per_dimension * dimension_iterations,
            channels=spec.channels,
            name="attention.v_append",
        )
    )
    return tuple(operations)


def attention_qk_device_piece(
    spec: DecodeSpec, layout: DecodeLayout
) -> tuple[AimOp, ...]:
    """Compute Q dot K-cache with shape-derived row-packed head batches."""

    return (
        contraction(
            spec,
            outputs=spec.L,
            reduction=spec.Dh,
            batches=spec.H,
            row=layout["k_cache"].row,
            batch_mapping="auto",
            name="attention.qk",
        ),
    )


def attention_sv_device_piece(
    spec: DecodeSpec, layout: DecodeLayout
) -> tuple[AimOp, ...]:
    """Compute softmax-score dot V-cache for every head and replica."""

    return (
        contraction(
            spec,
            outputs=spec.Dh,
            reduction=spec.L,
            batches=spec.H,
            row=layout["v_cache"].row,
            batch_mapping="auto",
            reduction_storage_extent=spec.max_seq_len,
            name="attention.sv",
        ),
    )


def softmax_host_split_device_piece(
    spec: DecodeSpec, layout: DecodeLayout
) -> tuple[AimOp, ...]:
    """Two score EWMUL phases around host exponentiation/reductions."""

    operations: list[AimOp] = []
    score_elements = spec.H * spec.L
    for phase in ("scale", "normalize_exp"):
        operations.extend(
            (
                distributed_transfer(
                    spec,
                    "write",
                    score_elements,
                    layout["scores"].row,
                    copies=2,
                    name=f"softmax.{phase}.inputs",
                ),
                AimElementwise(
                    "mul",
                    elements=score_elements,
                    row=layout["scores"].row,
                    replicas=spec.replicas,
                    channels_per_replica=spec.channels_per_replica,
                    name=f"softmax.{phase}.device_mul",
                ),
                distributed_transfer(
                    spec,
                    "read",
                    score_elements,
                    layout["scores"].row,
                    bank_offset=2,
                    name=f"softmax.{phase}.host_result",
                ),
                AimSync(name=f"softmax.{phase}.host_split"),
            )
        )
    return tuple(operations)


def ffn_activation_device_piece(
    spec: DecodeSpec, layout: DecodeLayout
) -> tuple[AimOp, ...]:
    """CENT's SiLU and gate multiplication movement/compute chain."""

    row = layout["ffn_activation"].row
    operations: list[AimOp] = [
        distributed_transfer(
            spec,
            "write",
            spec.F,
            row,
            copies=2,
            name="ffn_activation.x1_and_sigmoid",
        ),
        AimElementwise(
            "mul",
            elements=spec.F,
            row=row,
            replicas=spec.replicas,
            channels_per_replica=spec.channels_per_replica,
            name="ffn_activation.silu",
        ),
    ]
    operations.extend(
        _paired_bank_group_copies(
            spec,
            elements=spec.F,
            source_row=row,
            destination_row=row,
            name="ffn_activation.silu_relocation",
        )
    )
    operations.extend(
        (
            distributed_transfer(
                spec,
                "write",
                spec.F,
                row,
                name="ffn_activation.w3",
            ),
            AimElementwise(
                "mul",
                elements=spec.F,
                row=row,
                replicas=spec.replicas,
                channels_per_replica=spec.channels_per_replica,
                name="ffn_activation.gate",
            ),
            distributed_transfer(
                spec,
                "read",
                spec.F,
                row,
                bank_offset=2,
                name="ffn_activation.result",
            ),
            AimSync(name="ffn_activation.complete"),
        )
    )
    return tuple(operations)


StageBuilder = Callable[[DecodeSpec, DecodeLayout], tuple[AimOp, ...]]


def _projection_builder(
    weight: str,
    output: Callable[[DecodeSpec], int],
    reduction: Callable[[DecodeSpec], int],
    *,
    activation: bool = False,
) -> StageBuilder:
    def build(spec: DecodeSpec, layout: DecodeLayout) -> tuple[AimOp, ...]:
        return projection_device_piece(
            spec,
            layout,
            weight=weight,
            outputs=output(spec),
            reduction=reduction(spec),
            activation=activation,
        )

    return build


def _rms_builder(prefix: str) -> StageBuilder:
    def build(spec: DecodeSpec, layout: DecodeLayout) -> tuple[AimOp, ...]:
        return rms_device_piece(spec, layout, prefix=prefix)

    return build


def _residual_builder(name: str) -> StageBuilder:
    def build(spec: DecodeSpec, _layout: DecodeLayout) -> tuple[AimOp, ...]:
        return residual_device_piece(spec, name=name)

    return build


_STAGE_BUILDERS: Mapping[str, StageBuilder] = MappingProxyType(
    {
        "rms_x": _rms_builder("x"),
        "q_projection": _projection_builder("wq", lambda s: s.D, lambda s: s.D),
        "k_projection": _projection_builder("wk", lambda s: s.D, lambda s: s.D),
        "v_projection": _projection_builder("wv", lambda s: s.D, lambda s: s.D),
        "rope": rope_device_piece,
        "attention_cache_append": attention_cache_append,
        "attention_qk": attention_qk_device_piece,
        "softmax_host_split": softmax_host_split_device_piece,
        "attention_sv": attention_sv_device_piece,
        "o_projection": _projection_builder("wo", lambda s: s.D, lambda s: s.D),
        "attention_residual": _residual_builder("attention.residual"),
        "rms_sa": _rms_builder("sa"),
        "w1_projection_af": _projection_builder(
            "w1", lambda s: s.F, lambda s: s.D, activation=True
        ),
        "w3_projection": _projection_builder("w3", lambda s: s.F, lambda s: s.D),
        "ffn_activation": ffn_activation_device_piece,
        "w2_projection": _projection_builder("w2", lambda s: s.D, lambda s: s.F),
        "ffn_residual": _residual_builder("ffn.residual"),
    }
)


@dataclass(frozen=True)
class CentAimCase:
    """A declarative case label and ordered list of reusable device stages."""

    case_id: str
    kernel: str
    L: int
    stages: tuple[str, ...]
    work_measured: str
    scope_note: str = ""


_NORM_STAGES = ("rms_x", "attention_residual", "rms_sa", "ffn_residual")
_QKVO_STAGES = (
    "q_projection",
    "k_projection",
    "v_projection",
    "rope",
    "o_projection",
)
_ATTENTION_STAGES = (
    "attention_cache_append",
    "attention_qk",
    "attention_sv",
)
_FFN_FC_STAGES = ("w1_projection_af", "w3_projection", "w2_projection")
_FFN_COMPLETE_STAGES = (
    "w1_projection_af",
    "w3_projection",
    "ffn_activation",
    "w2_projection",
)
_FULL_BLOCK_STAGES = (
    "rms_x",
    "q_projection",
    "k_projection",
    "v_projection",
    "rope",
    "attention_cache_append",
    "attention_qk",
    "softmax_host_split",
    "attention_sv",
    "o_projection",
    "attention_residual",
    "rms_sa",
    "w1_projection_af",
    "w3_projection",
    "ffn_activation",
    "w2_projection",
    "ffn_residual",
)


# Case IDs select data only.  Lowering decisions live in typed operations and
# the target compiler, never in conditionals on these names or canonical sizes.
CENT_CASES: Mapping[str, CentAimCase] = MappingProxyType(
    {
        case.case_id: case
        for case in (
            CentAimCase(
                "norm_residual_l128",
                "RMSNorm + residual",
                128,
                _NORM_STAGES,
                "Two RMSNorm device paths and two residual additions",
                "Host reductions and reciprocal-square-root work are excluded.",
            ),
            CentAimCase(
                "qkvo_rope_l128",
                "Q/K/V/O projections + RoPE PIM",
                128,
                _QKVO_STAGES,
                "Four resident-weight projections plus RoPE movement/EWMUL",
                "Host RoPE work is excluded.",
            ),
            CentAimCase(
                "attention_l128",
                "Attention QK + SV",
                128,
                _ATTENTION_STAGES,
                "K/V cache append, Q dot K-cache, and score dot V-cache",
                "Softmax is a separate host-split stage.",
            ),
            CentAimCase(
                "attention_l512",
                "Attention QK + SV",
                512,
                _ATTENTION_STAGES,
                "K/V cache append, Q dot K-cache, and score dot V-cache",
                "Softmax is a separate host-split stage.",
            ),
            CentAimCase(
                "attention_l4096",
                "Attention QK + SV",
                4096,
                _ATTENTION_STAGES,
                "K/V cache append, Q dot K-cache, and score dot V-cache",
                "Softmax is a separate host-split stage.",
            ),
            CentAimCase(
                "softmax_pim_l128",
                "Softmax PIM portion",
                128,
                ("softmax_host_split",),
                "Score transfers, two EWMUL phases, and synchronization",
                "Exponentiation and reduction run outside AiM.",
            ),
            CentAimCase(
                "softmax_pim_l512",
                "Softmax PIM portion",
                512,
                ("softmax_host_split",),
                "Score transfers, two EWMUL phases, and synchronization",
                "Exponentiation and reduction run outside AiM.",
            ),
            CentAimCase(
                "softmax_pim_l4096",
                "Softmax PIM portion",
                4096,
                ("softmax_host_split",),
                "Score transfers, two EWMUL phases, and synchronization",
                "Exponentiation and reduction run outside AiM.",
            ),
            CentAimCase(
                "ffn_fc_l128",
                "FFN projections",
                128,
                _FFN_FC_STAGES,
                "W1 with fused AF, W3, and W2 resident-weight projections",
                "The separate SiLU/gating EWMUL path is excluded.",
            ),
            CentAimCase(
                "ffn_activation_l128",
                "FFN SiLU/gating PIM portion",
                128,
                ("ffn_activation",),
                "SiLU/gating EWMUL, copies, transfers, and synchronization",
                "The W1 fused AF is counted with FFN projections.",
            ),
            CentAimCase(
                "ffn_complete_l128",
                "Complete mapped FFN PIM portion",
                128,
                _FFN_COMPLETE_STAGES,
                "FFN projections and activation chain in program order",
            ),
            CentAimCase(
                "full_block_l128",
                "Full mapped transformer-block PIM portion",
                128,
                _FULL_BLOCK_STAGES,
                "All mapped AiM stages in transformer-block program order",
                "Host/PNM analytical additions are excluded.",
            ),
            CentAimCase(
                "full_block_l512",
                "Full mapped transformer-block PIM portion",
                512,
                _FULL_BLOCK_STAGES,
                "All mapped AiM stages in transformer-block program order",
                "Host/PNM analytical additions are excluded.",
            ),
            CentAimCase(
                "full_block_l4096",
                "Full mapped transformer-block PIM portion",
                4096,
                _FULL_BLOCK_STAGES,
                "All mapped AiM stages in transformer-block program order",
                "Host/PNM analytical additions are excluded.",
            ),
        )
    }
)


def compose_program(
    stages: Iterable[str],
    spec: DecodeSpec,
    *,
    name: str = "cent_aim_program",
    layout: DecodeLayout | None = None,
) -> AimProgram:
    """Compose reusable stage builders into one EOC-terminated program."""

    layout = build_layout(spec) if layout is None else layout
    operations: list[AimOp] = []
    for stage in stages:
        try:
            builder = _STAGE_BUILDERS[stage]
        except KeyError as error:
            raise KeyError(f"unknown CENT AiM stage {stage!r}") from error
        operations.extend(builder(spec, layout))
    return AimProgram(operations, name=name)


def build_case(
    case_id: str,
    spec: DecodeSpec | None = None,
) -> AimProgram:
    """Build one exact benchmark case without branching on its kernel name."""

    try:
        case = CENT_CASES[case_id]
    except KeyError as error:
        raise KeyError(f"unknown CENT AiM case {case_id!r}") from error
    if spec is None:
        spec = replace(DEFAULT_DECODE_SPEC, L=case.L)
    elif spec.L != case.L:
        raise ValueError(
            f"case {case_id} requires L={case.L}, supplied spec has L={spec.L}"
        )
    return compose_program(case.stages, spec, name=case.case_id)


def iter_case_programs(
    base_spec: DecodeSpec = DEFAULT_DECODE_SPEC,
):
    """Yield ``(case, concrete_spec, program)`` for all fourteen cases."""

    for case in CENT_CASES.values():
        spec = replace(base_spec, L=case.L)
        yield case, spec, build_case(case.case_id, spec)


def _opcode_counts(commands: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for command in commands:
        words = command.split()
        opcode = f"AiM_{words[1]}" if words[0] == "AiM" else f"{words[0]}_{words[1]}"
        counts[opcode] = counts.get(opcode, 0) + 1
    return dict(sorted(counts.items()))


def semantic_manifest(
    case_id: str,
    spec: DecodeSpec,
    compiled=None,
) -> dict:
    """Return the auditable work/residency contract used by evidence runs."""

    try:
        case = CENT_CASES[case_id]
    except KeyError as error:
        raise KeyError(f"unknown CENT AiM case {case_id!r}") from error
    if case.L != spec.L:
        raise ValueError("case and semantic-manifest sequence lengths differ")
    layout = build_layout(spec)
    weight_regions = {
        name: asdict(region)
        for name, region in layout.regions.items()
        if region.residency == "resident_weight"
    }
    cache_layout = None
    if "attention_cache_append" in case.stages:
        qk_batch_pack = _qk_batch_pack(spec)
        qk_batch_groups = _qk_batch_groups(spec)
        sv_batch_groups = _sv_batch_groups(spec)
        qk_output_tiles = _qk_output_tiles(spec, spec.max_seq_len)
        qk_live_output_tiles = _qk_output_tiles(spec, spec.L)
        qk_current_output_tile = (spec.L - 1) // (
            spec.channels_per_replica * BANKS_PER_CHANNEL
        )
        qk_current_producer_rows = [
            layout["k_cache"].row
            + qk_current_output_tile * qk_batch_groups
            + batch_group
            for batch_group in range(qk_batch_groups)
        ]
        sv_output_tiles = _ceil_div(spec.Dh, BANKS_PER_CHANNEL)
        sv_storage_rows = _ceil_div(spec.max_seq_len, ROW_ELEMENTS)
        cache_layout = {
            "qk_row_packed": {
                "batch_pack": qk_batch_pack,
                "batch_groups": qk_batch_groups,
                "allocated_output_tiles": qk_output_tiles,
                "live_output_tiles": qk_live_output_tiles,
                "allocated_rows": qk_batch_groups * qk_output_tiles,
                "live_rows": qk_batch_groups * qk_live_output_tiles,
                "current_output_tile": qk_current_output_tile,
                "current_producer_rows": qk_current_producer_rows,
                "region_rows": layout["k_cache"].rows,
                "row_address": "base + output_tile * batch_groups + batch_group",
                "producer_consumer_row_ownership_complete": (
                    layout["k_cache"].rows == qk_batch_groups * qk_output_tiles
                ),
            },
            "sv_channel_batched": {
                "batch_pack": spec.channels_per_replica,
                "batch_groups": sv_batch_groups,
                "output_tiles": sv_output_tiles,
                "reduction_storage_extent": spec.max_seq_len,
                "reduction_storage_rows": sv_storage_rows,
                "allocated_rows": (sv_batch_groups * sv_output_tiles * sv_storage_rows),
                "region_rows": layout["v_cache"].rows,
                "producer_channels": list(spec.channels),
                "consumer_channels": list(spec.channels),
                "producer_consumer_channel_ownership_complete": True,
                "producer_consumer_row_ownership_complete": (
                    layout["v_cache"].rows
                    == sv_batch_groups * sv_output_tiles * sv_storage_rows
                ),
            },
        }
    manifest = {
        "schema": "tenon-cent-aim-workload-v1",
        "case": asdict(case),
        "decode_spec": asdict(spec),
        "topology": {
            "target_channels": TARGET_CHANNELS,
            "active_channels": list(spec.channels),
            "replica_channel_groups": [list(group) for group in spec.channel_groups],
            "replica_channel_ownership_is_disjoint": True,
        },
        "layout": {
            "used_rows": layout.used_rows,
            "target_rows": TARGET_ROWS,
            "regions": {
                name: asdict(region) for name, region in layout.regions.items()
            },
        },
        "weights": {
            "placement": "statically resident before measured region",
            "host_transfers_included": False,
            "regions": weight_regions,
        },
        "dynamic_data": {
            "host_transfers_included": True,
            "cache_append_included": "attention_cache_append" in case.stages,
            "attention_cache_layout": cache_layout,
        },
        "scope_caveats": {
            "timing_only_simulator": True,
            "host_pnm_compute_excluded": True,
            "rope_post_result_transfer": (
                "CENT emits W MEM through its store helper; Tenon preserves "
                "that measured physical trace contract"
            ),
            "replica_values_not_modeled": (
                "WR_GB broadcasts a system-wide GPR slice to masked channels; "
                "the timing trace establishes disjoint channel ownership and "
                "work volume, but cannot establish distinct replica payload values"
            ),
        },
    }
    if compiled is not None:
        counts = _opcode_counts(compiled.commands)
        if counts.get("AiM_EOC") != 1 or compiled.commands[-1] != "AiM EOC":
            raise ValueError("compiled case must contain exactly one trailing EOC")
        logical_coverage = compiler_logical_work_coverage(
            compiled.manifest, compiled.trace
        )
        if not logical_coverage["complete"]:
            raise ValueError(
                "compiled case has incomplete typed logical-work coverage: "
                + "; ".join(logical_coverage["errors"])
            )
        if cache_layout is not None:
            lowered_contractions = [
                operation
                for operation in compiled.manifest["operations"]
                if operation["kind"] == "contraction"
                and operation.get("requested_batch_mapping") == "auto"
            ]
            packed = [
                operation
                for operation in lowered_contractions
                if operation["batch_mapping"] == "row_packed"
            ]
            channel_batched = [
                operation
                for operation in lowered_contractions
                if operation["batch_mapping"] == "channels"
            ]
            if len(packed) != 1 or len(channel_batched) != 1:
                raise ValueError(
                    "attention cache evidence requires one packed QK and one "
                    "channel-batched SV consumer"
                )
            qk_consumer = packed[0]
            sv_consumer = channel_batched[0]
            qk_contract = cache_layout["qk_row_packed"]
            sv_contract = cache_layout["sv_channel_batched"]
            qk_consumer_current_rows = [
                group["matrix_rows"][qk_contract["current_output_tile"]]
                for group in qk_consumer["input_groups"]
            ]
            qk_complete = all(
                (
                    qk_consumer["batch_pack"] == qk_contract["batch_pack"],
                    qk_consumer["batch_groups"] == qk_contract["batch_groups"],
                    qk_consumer["output_tiles"] == qk_contract["live_output_tiles"],
                    qk_consumer["allocated_matrix_rows"] == qk_contract["live_rows"],
                    qk_consumer["allocated_matrix_rows"]
                    <= qk_contract["allocated_rows"],
                    qk_consumer["matrix_row_range"]
                    == [
                        layout["k_cache"].row,
                        layout["k_cache"].row + qk_contract["live_rows"],
                    ],
                    qk_consumer_current_rows == qk_contract["current_producer_rows"],
                )
            )
            sv_complete = all(
                (
                    sv_consumer["batch_pack"] == sv_contract["batch_pack"],
                    sv_consumer["batch_groups"] == sv_contract["batch_groups"],
                    sv_consumer["output_tiles"] == sv_contract["output_tiles"],
                    sv_consumer["reduction_storage_rows"]
                    == sv_contract["reduction_storage_rows"],
                    sv_consumer["allocated_matrix_rows"]
                    == sv_contract["allocated_rows"],
                )
            )
            qk_contract["compiled_consumer_layout"] = {
                "batch_pack": qk_consumer["batch_pack"],
                "batch_groups": qk_consumer["batch_groups"],
                "live_output_tiles": qk_consumer["output_tiles"],
                "accessed_rows": qk_consumer["allocated_matrix_rows"],
                "matrix_row_range": qk_consumer["matrix_row_range"],
                "current_consumer_rows": qk_consumer_current_rows,
                "complete": qk_complete,
            }
            sv_contract["compiled_consumer_layout"] = {
                "batch_pack": sv_consumer["batch_pack"],
                "batch_groups": sv_consumer["batch_groups"],
                "output_tiles": sv_consumer["output_tiles"],
                "reduction_storage_rows": sv_consumer["reduction_storage_rows"],
                "allocated_rows": sv_consumer["allocated_matrix_rows"],
                "complete": sv_complete,
            }
            if not qk_complete or not sv_complete:
                raise ValueError(
                    "cache producer layout does not match the selected consumers"
                )
        manifest["compiled_trace"] = {
            "opcode_counts": counts,
            "eoc_count": counts["AiM_EOC"],
            "command_count": len(compiled.commands),
            "sha256": compiled.manifest["trace"]["sha256"],
            "command_shape_signature": command_shape_signature(compiled.trace),
        }
        manifest["logical_work_coverage"] = logical_coverage
    return manifest


__all__ = [
    "TARGET_CHANNELS",
    "BANKS_PER_CHANNEL",
    "BANK_GROUPS_PER_CHANNEL",
    "ROW_ELEMENTS",
    "VECTOR_LANES",
    "DecodeSpec",
    "DecodeLayout",
    "RowRegion",
    "DEFAULT_DECODE_SPEC",
    "CentAimCase",
    "CENT_CASES",
    "build_layout",
    "distributed_transfer",
    "contraction",
    "rms_device_piece",
    "residual_device_piece",
    "projection_device_piece",
    "rope_device_piece",
    "attention_cache_append",
    "attention_qk_device_piece",
    "attention_sv_device_piece",
    "softmax_host_split_device_piece",
    "ffn_activation_device_piece",
    "compose_program",
    "build_case",
    "iter_case_programs",
    "semantic_manifest",
]
