# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, fail-closed evidence helpers for the CENT/AiM campaign."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_SIMULATOR_COMMIT = "0f28a07bdb83e42b9305ad3d45410ebd3aa2c091"
EXPECTED_CASE_IDS = (
    "norm_residual_l128",
    "qkvo_rope_l128",
    "attention_l128",
    "attention_l512",
    "attention_l4096",
    "softmax_pim_l128",
    "softmax_pim_l512",
    "softmax_pim_l4096",
    "ffn_fc_l128",
    "ffn_activation_l128",
    "ffn_complete_l128",
    "full_block_l128",
    "full_block_l512",
    "full_block_l4096",
)
TCK_NS = 0.5

_CYCLES_RE = re.compile(r"(?m)^\s*memory_system_cycles:\s*(\d+)\s*$")
_EOC_RE = re.compile(r"(?m)^\s*total_num_AiM_ISR_EOC_requests:\s*(\d+)\b")
_TOTAL_RE = re.compile(r"(?m)^\s*(total_num_[A-Za-z0-9_]+):\s*(\d+)\b")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceError(RuntimeError):
    """Raised when artifacts cannot support the claimed measurement."""


def jsonable(value: Any) -> Any:
    """Convert frozen dataclasses/mappings/tuples into plain JSON values."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: jsonable(getattr(value, name)) for name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def write_text(path: Path, value: str) -> None:
    write_bytes(path, value.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _trace_opcode(words: list[str]) -> str:
    if words[0] == "AiM" and len(words) >= 2:
        return f"AiM_{words[1]}"
    if len(words) >= 2:
        return f"{words[0]}_{words[1]}"
    return words[0]


def opcode_counts(trace_text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw_line in trace_text.splitlines():
        words = raw_line.split()
        if not words or words[0].startswith("#"):
            continue
        counts[_trace_opcode(words)] += 1
    return dict(sorted(counts.items()))


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


def _command_shape(raw_line: str) -> tuple[str, int | None, int | None] | None:
    words = raw_line.split()
    if not words or words[0].startswith("#"):
        return None
    opcode = _trace_opcode(words)
    try:
        op_size_index, mask_index = _COMMAND_SHAPE_FIELDS[opcode]
    except KeyError as error:
        raise EvidenceError(
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
        raise EvidenceError(f"malformed {opcode} trace record: {raw_line!r}") from error
    if op_size is not None and op_size <= 0:
        raise EvidenceError(f"{opcode} has nonpositive op_size {op_size}")
    if active_mask_fanout is not None and active_mask_fanout <= 0:
        raise EvidenceError(
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
        raise EvidenceError("cannot fingerprint an empty trace")
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
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def opcode_active_channel_coverage(
    trace_text: str,
    opcode: str,
    *,
    target_channels: int,
) -> dict[str, Any]:
    """Derive the union of channels selected by one masked opcode."""

    target_channels = int(target_channels)
    if target_channels <= 0:
        raise EvidenceError("target channel count must be positive")
    try:
        _op_size_index, mask_index = _COMMAND_SHAPE_FIELDS[opcode]
    except KeyError as error:
        raise EvidenceError(f"unknown trace opcode {opcode!r}") from error
    if mask_index is None:
        raise EvidenceError(f"trace opcode {opcode!r} has no channel mask")
    active: set[int] = set()
    command_count = 0
    for raw_line in trace_text.splitlines():
        words = raw_line.split()
        if not words or words[0].startswith("#") or _trace_opcode(words) != opcode:
            continue
        try:
            mask = int(words[mask_index], 0)
        except (IndexError, ValueError) as error:
            raise EvidenceError(f"malformed {opcode} record: {raw_line!r}") from error
        if mask <= 0 or mask.bit_length() > target_channels:
            raise EvidenceError(f"{opcode} mask exceeds target channel geometry")
        for channel in range(target_channels):
            if mask & (1 << (target_channels - 1 - channel)):
                active.add(channel)
        command_count += 1
    return {
        "opcode": opcode,
        "target_channels": target_channels,
        "command_count": command_count,
        "active_channels": sorted(active),
        "active_channel_count": len(active),
        "complete_target_coverage": len(active) == target_channels,
    }


def command_shape_signature_delta(
    vendor: Mapping[str, Any], tenon: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the canonical Tenon-minus-vendor physical-shape delta."""

    def unpack(
        label: str, signature: Mapping[str, Any]
    ) -> dict[tuple[str, int | None, int | None], int]:
        if signature.get("schema") != "tenon-aim-command-shape-signature-v1":
            raise EvidenceError(f"{label} command-shape signature schema differs")
        result = {}
        records = signature.get("records")
        if not isinstance(records, list):
            raise EvidenceError(f"{label} command-shape signature has no records")
        for record in records:
            try:
                key = (
                    str(record["opcode"]),
                    (None if record["op_size"] is None else int(record["op_size"])),
                    (
                        None
                        if record["active_mask_fanout"] is None
                        else int(record["active_mask_fanout"])
                    ),
                )
                count = int(record["command_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError(
                    f"{label} command-shape signature record is malformed"
                ) from error
            if key in result or count <= 0:
                raise EvidenceError(
                    f"{label} command-shape signature has duplicate/invalid records"
                )
            result[key] = count
        return result

    vendor_records = unpack("vendor", vendor)
    tenon_records = unpack("Tenon", tenon)
    records = []
    for opcode, op_size, fanout in sorted(
        set(vendor_records) | set(tenon_records),
        key=lambda key: (
            key[0],
            -1 if key[1] is None else key[1],
            -1 if key[2] is None else key[2],
        ),
    ):
        vendor_count = vendor_records.get((opcode, op_size, fanout), 0)
        tenon_count = tenon_records.get((opcode, op_size, fanout), 0)
        if vendor_count == tenon_count:
            continue
        records.append(
            {
                "opcode": opcode,
                "op_size": op_size,
                "active_mask_fanout": fanout,
                "vendor_command_count": vendor_count,
                "tenon_command_count": tenon_count,
                "delta_commands_tenon_minus_vendor": tenon_count - vendor_count,
            }
        )
    return {
        "schema": "tenon-cent-aim-command-shape-delta-v1",
        "equality_required": False,
        "equal": not records,
        "records": records,
    }


def physical_work_comparison(
    *,
    cent_source_commit: str,
    vendor_trace_sha256: str,
    vendor_opcode_counts: Mapping[str, int],
    vendor_command_shape_signature: Mapping[str, Any],
    vendor_wr_abk_channel_coverage: Mapping[str, Any],
    tenon_trace_text: str,
    target_channels: int,
    replica_count: int,
    channels_per_replica: int,
    logical_work_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the auditable physical comparison without requiring equality."""

    tenon_counts = opcode_counts(tenon_trace_text)
    tenon_signature = command_shape_signature(tenon_trace_text)
    tenon_wr_coverage = opcode_active_channel_coverage(
        tenon_trace_text,
        "AiM_WR_ABK",
        target_channels=target_channels,
    )
    replica_count = int(replica_count)
    channels_per_replica = int(channels_per_replica)

    def covered_groups(coverage: Mapping[str, Any]) -> list[int]:
        active = set(int(channel) for channel in coverage["active_channels"])
        return [
            replica
            for replica in range(replica_count)
            if set(
                range(
                    replica * channels_per_replica,
                    (replica + 1) * channels_per_replica,
                )
            ).issubset(active)
        ]

    cache_coverage = None
    if (
        int(vendor_wr_abk_channel_coverage["command_count"])
        or tenon_wr_coverage["command_count"]
    ):
        ownership = [
            operation["ownership_coverage"]
            for operation in logical_work_coverage["operations"]
            if operation["ownership_coverage"] is not None
        ]
        ownership_complete = bool(ownership) and all(
            item["complete"] for item in ownership
        )
        expected_coordinates = sum(
            int(item["expected_coordinate_count"]) for item in ownership
        )
        materialized_coordinates = sum(
            int(item["materialized_coordinate_count"]) for item in ownership
        )
        vendor_groups = covered_groups(vendor_wr_abk_channel_coverage)
        tenon_groups = covered_groups(tenon_wr_coverage)
        cache_coverage = {
            "expected_replica_groups": list(range(replica_count)),
            "vendor_covered_replica_groups": vendor_groups,
            "vendor_covered_replicas": len(vendor_groups),
            "tenon_covered_replica_groups": tenon_groups,
            "tenon_covered_replicas": len(tenon_groups),
            "expected_ownership_coordinate_count": expected_coordinates,
            "vendor_materialized_command_count": int(
                vendor_wr_abk_channel_coverage["command_count"]
            ),
            "tenon_materialized_coordinate_count": materialized_coordinates,
            "tenon_ownership_complete": ownership_complete,
            "tenon_complete": (
                len(tenon_groups) == replica_count
                and ownership_complete
                and materialized_coordinates == expected_coordinates
            ),
            "vendor_complete": (
                len(vendor_groups) == replica_count
                and int(vendor_wr_abk_channel_coverage["command_count"])
                == expected_coordinates
            ),
            "comparison_note": (
                "Tenon materializes every typed replica's V-cache append; the "
                "frozen vendor trace is retained unchanged even when its "
                "one-hot WR_ABK masks cover fewer replica channel groups."
            ),
        }
    return {
        "cent_source_commit": str(cent_source_commit),
        "vendor_trace_sha256": str(vendor_trace_sha256),
        "vendor_opcode_counts": dict(sorted(vendor_opcode_counts.items())),
        "tenon_opcode_counts": tenon_counts,
        "opcode_inventory_equal": tenon_counts
        == dict(sorted(vendor_opcode_counts.items())),
        "vendor_command_shape_signature": dict(vendor_command_shape_signature),
        "tenon_command_shape_signature": tenon_signature,
        "command_shape_delta": command_shape_signature_delta(
            vendor_command_shape_signature, tenon_signature
        ),
        "command_shape_equality_required": False,
        "cache_append_replica_coverage": cache_coverage,
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


def validate_trace_text(trace_text: str, *, label: str = "trace") -> dict[str, Any]:
    if not trace_text:
        raise EvidenceError(f"{label} is empty")
    try:
        encoded = trace_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EvidenceError(f"{label} is not UTF-8 encodable") from error
    records = [line.strip() for line in trace_text.splitlines() if line.strip()]
    if not records:
        raise EvidenceError(f"{label} contains no records")
    eoc_count = records.count("AiM EOC")
    if eoc_count != 1 or records[-1] != "AiM EOC":
        raise EvidenceError(
            f"{label} must contain exactly one trailing AiM EOC; "
            f"found count={eoc_count}, final={records[-1]!r}"
        )
    counts = opcode_counts(trace_text)
    if counts.get("AiM_EOC") != 1:
        raise EvidenceError(f"{label} opcode counts do not contain one EOC")
    return {
        "bytes": len(encoded),
        "nonempty_lines": len(records),
        "sha256": sha256_bytes(encoded),
        "opcode_counts": counts,
        "eoc_count": eoc_count,
        "terminal_record": records[-1],
    }


def validate_frozen_vendor_trace(
    case_dir: Path,
    *,
    case_id: str,
    expected_trace_sha256: str,
    expected_opcode_counts: Mapping[str, int],
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind an archived CENT trace to its frozen physical-work contract."""
    case = read_json(case_dir / "case.json")
    if case.get("case_id") != case_id or case_dir.name != case_id:
        raise EvidenceError(f"{case_id}: vendor case identity/path mismatch")
    first_path = case_dir / "input.trace"
    second_path = case_dir / "input.repeat.trace"
    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()
    if first_bytes != second_bytes:
        raise EvidenceError(f"{case_id}: vendor trace generations differ")
    try:
        trace = validate_trace_text(first_bytes.decode("utf-8"), label=case_id)
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{case_id}: vendor trace is not UTF-8") from error
    expected_counts = dict(sorted(expected_opcode_counts.items()))
    if trace["sha256"] != expected_trace_sha256:
        raise EvidenceError(f"{case_id}: vendor trace violates frozen SHA-256")
    if trace["opcode_counts"] != expected_counts:
        raise EvidenceError(f"{case_id}: vendor trace violates physical-work inventory")
    metadata = read_json(case_dir / "trace-metadata.json")
    if metadata.get("deterministic") is not True:
        raise EvidenceError(f"{case_id}: vendor trace metadata is nondeterministic")
    if metadata.get("source_commit") != expected_source_commit:
        raise EvidenceError(f"{case_id}: vendor trace source commit differs")
    if metadata.get("generator") != "unmodified CENT cent_simulation/function_sim.py":
        raise EvidenceError(f"{case_id}: vendor trace generator is not frozen CENT")
    runs = metadata.get("generator_runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise EvidenceError(f"{case_id}: vendor trace metadata lacks two generations")
    expected_names = ("input.trace", "input.repeat.trace")
    for run, expected_name in zip(runs, expected_names):
        if Path(str(run.get("path", ""))).name != expected_name:
            raise EvidenceError(f"{case_id}: vendor generator-run path differs")
        expected_run = {
            "bytes": trace["bytes"],
            "instruction_counts": expected_counts,
            "nonempty_lines": trace["nonempty_lines"],
            "sha256": trace["sha256"],
            "terminal_record": "AiM EOC",
        }
        for field, expected in expected_run.items():
            if run.get(field) != expected:
                raise EvidenceError(f"{case_id}: vendor generator-run {field} is stale")
    return case, trace


def parse_simulator_stdout(stdout: str) -> dict[str, Any]:
    cycles_matches = _CYCLES_RE.findall(stdout)
    eoc_matches = _EOC_RE.findall(stdout)
    if len(cycles_matches) != 1:
        raise EvidenceError(
            "expected exactly one global memory_system_cycles statistic, "
            f"found {cycles_matches}"
        )
    if len(eoc_matches) != 1 or int(eoc_matches[0]) != 1:
        raise EvidenceError(
            "expected exactly one completed EOC request, " f"found {eoc_matches}"
        )
    cycles = int(cycles_matches[0])
    if cycles <= 0:
        raise EvidenceError(f"memory_system_cycles must be positive, got {cycles}")
    total_pairs = _TOTAL_RE.findall(stdout)
    total_keys = [key for key, _value in total_pairs]
    if len(total_keys) != len(set(total_keys)):
        raise EvidenceError("simulator stdout contains duplicate total_num counters")
    totals = {key: int(value) for key, value in total_pairs}
    return {
        "memory_system_cycles": cycles,
        "latency_us_at_2GHz": cycles * TCK_NS / 1000.0,
        "eoc_requests": 1,
        "total_request_counts": dict(sorted(totals.items())),
    }


def request_counter_for_opcode(opcode: str) -> str:
    if opcode.startswith("AiM_"):
        return f"total_num_AiM_ISR_{opcode.removeprefix('AiM_')}_requests"
    return f"total_num_{opcode}_requests"


def reconcile_request_counts(
    trace_counts: Mapping[str, int], simulator_totals: Mapping[str, int]
) -> None:
    expected_counters = set()
    for opcode, expected in trace_counts.items():
        counter = request_counter_for_opcode(opcode)
        expected_counters.add(counter)
        if counter not in simulator_totals:
            raise EvidenceError(
                f"simulator stdout lacks request counter {counter} for {opcode}"
            )
        observed = int(simulator_totals[counter])
        if observed != int(expected):
            raise EvidenceError(
                f"{counter}={observed} does not match trace {opcode}={expected}"
            )
    for counter, observed in simulator_totals.items():
        is_command_counter = counter.startswith("total_num_AiM_ISR_") or bool(
            re.fullmatch(r"total_num_[WR]_[A-Za-z0-9_]+_requests", counter)
        )
        if is_command_counter and int(observed) and counter not in expected_counters:
            raise EvidenceError(
                f"simulator reports unexpected nonzero command counter {counter}={observed}"
            )


def write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == checksum_path.resolve():
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    write_text(checksum_path, "\n".join(entries) + "\n")
    return checksum_path


def verify_checksums(root: Path, *, require_complete: bool = True) -> None:
    root = root.resolve()
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise EvidenceError(f"missing checksum manifest {manifest}")
    expected_paths: set[Path] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read checksum manifest {manifest}") from error
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            expected_hash, relative_text = line.split("  ", 1)
        except ValueError as error:
            raise EvidenceError(
                f"malformed checksum line {manifest}:{number}"
            ) from error
        if not _SHA256_RE.fullmatch(expected_hash):
            raise EvidenceError(f"invalid SHA-256 at {manifest}:{number}")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError(f"unsafe checksum path at {manifest}:{number}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise EvidenceError(f"checksum path escapes root at {manifest}:{number}")
        if path in expected_paths:
            raise EvidenceError(f"duplicate checksum path {relative_text}")
        expected_paths.add(path)
        if not path.is_file():
            raise EvidenceError(f"missing checksummed file {relative_text}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise EvidenceError(
                f"checksum mismatch {relative_text}: "
                f"{observed_hash} != {expected_hash}"
            )
    if require_complete:
        actual_paths = {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest.resolve()
        }
        missing = expected_paths - actual_paths
        extra = actual_paths - expected_paths
        if missing:
            raise EvidenceError(
                "manifested files are absent: "
                + ", ".join(str(path.relative_to(root)) for path in sorted(missing))
            )
        if extra:
            raise EvidenceError(
                "unmanifested files are present: "
                + ", ".join(str(path.relative_to(root)) for path in sorted(extra))
            )


def comparison_row(
    *,
    case_id: str,
    kernel: str,
    sequence_length: int,
    vendor_cycles: int,
    tenon_cycles: int,
    trace_sha256: str,
) -> dict[str, Any]:
    vendor_cycles = int(vendor_cycles)
    tenon_cycles = int(tenon_cycles)
    if vendor_cycles <= 0 or tenon_cycles <= 0:
        raise EvidenceError("comparison cycle counts must be positive")
    outcome = (
        "win"
        if tenon_cycles < vendor_cycles
        else ("tie" if tenon_cycles == vendor_cycles else "loss")
    )
    return {
        "case_id": case_id,
        "kernel": kernel,
        "sequence_length": int(sequence_length),
        "vendor_cycles": vendor_cycles,
        "tenon_cycles": tenon_cycles,
        "delta_cycles_tenon_minus_vendor": tenon_cycles - vendor_cycles,
        "tenon_vs_vendor_percent": round(
            (tenon_cycles - vendor_cycles) * 100.0 / vendor_cycles, 6
        ),
        "speedup_vendor_over_tenon": round(vendor_cycles / tenon_cycles, 9),
        "outcome": outcome,
        "trace_sha256": trace_sha256,
        "compile_repeats": 2,
        "compile_deterministic": True,
        "simulator_repeats": 2,
        "simulator_deterministic": True,
        "tenon_evidence": f"cases/{case_id}/simulator/summary.json",
        "vendor_evidence": f"cases/{case_id}/vendor-evidence.json",
    }


COMPARISON_FIELDS = tuple(
    comparison_row(
        case_id="x",
        kernel="x",
        sequence_length=1,
        vendor_cycles=1,
        tenon_cycles=1,
        trace_sha256="0" * 64,
    ).keys()
)


def comparison_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["outcome"]) for row in rows)
    return {
        "case_count": len(rows),
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "matches_or_beats": counts["win"] + counts["tie"],
    }


def render_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    totals = comparison_totals(rows)
    lines = [
        "# Tenon versus CENT on SK hynix AiM",
        "",
        (
            "Every cycle value is the global `memory_system_cycles` statistic "
            "from two deterministic executions of one exact, fully materialized "
            "trace on simulator commit `0f28a07b`. Lower is better."
        ),
        "",
        (
            f"Summary: **{totals['wins']} wins, {totals['ties']} ties, "
            f"{totals['losses']} losses**; Tenon matches or beats CENT on "
            f"{totals['matches_or_beats']}/{totals['case_count']} cases."
        ),
        "",
        "| Case | Kernel | L | CENT cycles | Tenon cycles | Delta | Speedup | Result | Evidence |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        delta = int(row["delta_cycles_tenon_minus_vendor"])
        lines.append(
            f"| `{row['case_id']}` | {row['kernel']} | "
            f"{int(row['sequence_length']):,} | "
            f"{int(row['vendor_cycles']):,} | {int(row['tenon_cycles']):,} | "
            f"{delta:+,} | {float(row['speedup_vendor_over_tenon']):.3f}x | "
            f"{str(row['outcome']).upper()} | "
            f"[Tenon]({row['tenon_evidence']}) / "
            f"[CENT]({row['vendor_evidence']}) |"
        )
    lines.extend(
        [
            "",
            (
                "The SK hynix Ramulator2 model is timing-only: it consumes "
                "commands and addresses, but no tensor payloads, and produces no "
                "numerical outputs. These results therefore claim cycle fidelity "
                "and semantic/work-invariant construction, not numerical "
                "correctness. Each case retains that manifest explicitly."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_reuse_calibration_markdown(
    rows: list[dict[str, Any]], dimension: int
) -> str:
    best = min(
        rows, key=lambda row: (row["memory_system_cycles"], row["reuse_group_size"])
    )
    lines = [
        "# AiM contraction reuse calibration",
        "",
        (
            f"A strict {dimension}x{dimension} dense projection was compiled and "
            "simulated twice for every target-legal reuse group."
        ),
        "",
        "| Reuse group | Cycles | Commands | Evidence |",
        "|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['reuse_group_size']} | {row['memory_system_cycles']:,} | "
            f"{row['trace_commands']:,} | "
            f"[artifacts](candidate-{row['reuse_group_size']}/simulator/summary.json) |"
        )
    lines.extend(
        [
            "",
            f"Measured best legal candidate: **{best['reuse_group_size']}**.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "COMPARISON_FIELDS",
    "EXPECTED_CASE_IDS",
    "EXPECTED_SIMULATOR_COMMIT",
    "EvidenceError",
    "canonical_json_bytes",
    "command_shape_signature",
    "command_shape_signature_delta",
    "compiler_logical_work_coverage",
    "comparison_row",
    "comparison_totals",
    "jsonable",
    "opcode_active_channel_coverage",
    "opcode_counts",
    "parse_simulator_stdout",
    "physical_work_comparison",
    "read_json",
    "reconcile_request_counts",
    "render_comparison_markdown",
    "render_reuse_calibration_markdown",
    "sha256_bytes",
    "sha256_file",
    "validate_trace_text",
    "validate_frozen_vendor_trace",
    "verify_checksums",
    "write_checksums",
    "write_json",
    "write_text",
]
