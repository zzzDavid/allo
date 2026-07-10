#!/usr/bin/env python3
"""Check generated PIM artifacts against schedule-performance incumbents.

Hardware candidates carry measured repetitions in a ``runs``, ``samples``,
or ``repetitions`` array, either at the document level or inside ``result``.
Each item may be a complete result document or a compact metric mapping. The
discarded warm-up is separate and explicit as
``result.warmup = {"count": 1, "discarded": true}``.

APUv2 profile logs and companion JSON are parsed to normalize the schedule,
but only immutable executable, source, and build artifacts are SHA-256 pinned.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "perf"
    / "data"
    / "pim_schedule_incumbents.json"
)
DEFAULT_BASELINE_ROOT = Path.home() / "shared" / "tenon-artifacts"
SUPPORTED_BACKENDS = {
    "samsung-hbm-pim",
    "sk-hynix-aim",
    "upmem",
    "apu-v1",
    "apu-v2",
}
EVIDENCE_CLASSES = {
    "analytical",
    "cycle_simulator",
    "hardware_cycles",
    "hardware_wall",
}
DEFAULT_POLICIES = {
    "analytical": {
        "evidence_class": "analytical",
        "min_samples": 1,
        "aggregate": "max",
        "max_ratio": 1.0,
    },
    "cycle_simulator": {
        "evidence_class": "cycle_simulator",
        "min_samples": 1,
        "aggregate": "max",
        "max_ratio": 1.0,
    },
    "apu_v1_cycles": {
        "evidence_class": "hardware_cycles",
        "warmup_samples": 1,
        "min_samples": 5,
        "aggregate": "median",
        "max_ratio": 1.01,
    },
    "apu_v2_device_ticks": {
        "evidence_class": "hardware_cycles",
        "warmup_samples": 1,
        "min_samples": 7,
        "aggregate": "median",
        "max_ratio": 1.01,
    },
    "apu_v2_wall_us": {
        "evidence_class": "hardware_wall",
        "warmup_samples": 1,
        "min_samples": 7,
        "aggregate": "median",
        "max_ratio": 1.03,
        "max_sample_ratio": 1.05,
    },
}


class ManifestError(ValueError):
    pass


class RecordError(ValueError):
    pass


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    key: str
    code: str
    message: str
    severity: str = "failure"


@dataclass
class CheckReport:
    checked: int
    findings: list[Finding]

    @property
    def failures(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "failure"]

    @property
    def insufficient(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "insufficient"]

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class NormalizedRecord:
    backend: str
    workload: str | None
    dataset: Any
    dtype: Any
    shape: Any
    semantic: Mapping[str, Any]
    correctness: str
    metrics: Mapping[str, tuple[Decimal, ...]]
    metric_errors: tuple[str, ...]
    producer_revision: str | None
    warmup: Mapping[str, Any] | None


def policy_template() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(DEFAULT_POLICIES)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decimal(value: Any) -> Decimal:
    if not _is_number(value):
        raise InvalidOperation
    converted = Decimal(str(value))
    if not converted.is_finite() or converted < 0:
        raise InvalidOperation
    return converted


def _clean_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _strip_omission_suffix(value: str) -> str:
    cleaned = _clean_text(value)
    for suffix in (" on host", " host-side", " omitted"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip()
    return cleaned


def _partial_detail(detail: Any) -> tuple[str, list[str]]:
    parts = [part.strip() for part in str(detail or "").split(";") if part.strip()]
    if not parts:
        return "unspecified device core", ["unspecified workload semantics"]
    scope = _clean_text(parts[0])
    omitted = [_strip_omission_suffix(part) for part in parts[1:]]
    return scope, omitted or ["workload semantics outside measured device core"]


def _canonical_shape(backend: str, data: Mapping[str, Any]) -> Any:
    if backend == "samsung-hbm-pim":
        legs = []
        for leg in data.get("legs") or []:
            legs.append(
                {
                    "name": leg.get("name"),
                    "operation": leg.get("operation"),
                    "logical_shape": leg.get("logical_shape"),
                    "count": leg.get("count"),
                }
            )
        return {"legs": sorted(legs, key=lambda item: str(item["name"]))}
    if backend == "sk-hynix-aim":
        return data.get("shapes")
    if backend == "upmem":
        return data.get("dimensions")
    if backend == "apu-v1":
        legs = []
        for leg in (data.get("result") or {}).get("legs") or []:
            legs.append(
                {
                    "name": leg.get("name"),
                    "shape": leg.get("shape"),
                    "operation_count": leg.get("operation_count"),
                }
            )
        return {"legs": sorted(legs, key=lambda item: str(item["name"]))}
    return data.get("shape")


def _semantic(backend: str, data: Mapping[str, Any], shape: Any) -> dict[str, Any]:
    if backend == "upmem":
        raw_status = data.get("semantic_status")
    elif backend == "apu-v2":
        raw_status = data.get("semantic_status")
    else:
        raw_status = data.get("status")
    status_text = _clean_text(raw_status)
    if "partial" in status_text:
        status = "partial"
    elif "full" in status_text or status_text == "pass":
        status = "full"
    else:
        status = status_text or "unknown"

    execution_scope = (data.get("result") or {}).get("execution_scope")
    if status != "partial":
        scope: Any = "full workload"
        if backend == "apu-v1":
            scope = {"semantics": "full workload", "execution": execution_scope}
        return {"status": status, "scope": scope, "omitted": []}
    if backend == "samsung-hbm-pim":
        leg_names = [leg.get("name") for leg in (shape or {}).get("legs", [])]
        return {
            "status": "partial",
            "scope": {"device_legs": leg_names},
            "omitted": ["all workload semantics outside listed device legs"],
        }
    if backend == "upmem":
        omitted = data.get("omitted_semantics")
        return {
            "status": "partial",
            "scope": _clean_text(data.get("description")),
            "omitted": [_clean_text(omitted)] if omitted else [],
        }
    scope, omitted = _partial_detail(data.get("detail"))
    if backend == "apu-v1":
        scope = {"semantics": scope, "execution": execution_scope}
    return {"status": "partial", "scope": scope, "omitted": omitted}


def _walk_correctness(value: Any) -> tuple[bool, bool]:
    failed = False
    passed = False
    if isinstance(value, bool):
        return not value, value
    if isinstance(value, str):
        text = _clean_text(value)
        failed = any(token in text for token in ("fail", "error", "incorrect"))
        passed = any(
            token in text for token in ("pass", "bit-exact", "clean_finish", "correct")
        )
        return failed, passed
    if isinstance(value, Mapping):
        for nested in value.values():
            nested_failed, nested_passed = _walk_correctness(nested)
            failed = failed or nested_failed
            passed = passed or nested_passed
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            nested_failed, nested_passed = _walk_correctness(nested)
            failed = failed or nested_failed
            passed = passed or nested_passed
    return failed, passed


def _correctness(data: Mapping[str, Any], *, allow_status_only: bool) -> str:
    values = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key == "correctness":
                    values.append(nested)
                else:
                    collect(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                collect(nested)

    collect(data)
    failed, passed = _walk_correctness(values)
    status = _clean_text(data.get("status"))
    if any(token in status for token in ("fail", "error", "reject", "unsupported")):
        failed = True
    if failed:
        return "fail"
    if passed or (
        allow_status_only
        and status
        in {"pass", "partial", "measured_composition", "measured-composition"}
    ):
        return "pass"
    return "unknown"


def _find_runs(
    data: Mapping[str, Any], primary: Mapping[str, Any]
) -> Sequence[Any] | None:
    for owner in (data, primary):
        for key in ("runs", "samples", "repetitions", "repeat_results"):
            values = owner.get(key)
            if isinstance(values, list):
                return values
    return None


def _lookup_metric(value: Any, names: Sequence[str]) -> Any:
    if _is_number(value):
        return value
    if not isinstance(value, Mapping):
        return None
    for name in names:
        if name in value:
            return value[name]
    for container in ("result", "performance", "measurement"):
        nested = value.get(container)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested:
                    return nested[name]
    return None


def _metric_samples(
    data: Mapping[str, Any],
    primary: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[tuple[Decimal, ...], list[str]]:
    errors = []
    runs = _find_runs(data, primary)
    if runs is not None:
        raw_values = []
        for index, run in enumerate(runs, 1):
            value = _lookup_metric(run, names)
            if value is None:
                errors.append(f"run {index} is missing {names[0]}")
            else:
                raw_values.append(value)
    else:
        raw_value = _lookup_metric(primary, names)
        if isinstance(raw_value, list):
            raw_values = raw_value
        elif raw_value is None:
            raw_values = []
        else:
            raw_values = [raw_value]
    samples = []
    for index, value in enumerate(raw_values, 1):
        try:
            samples.append(_decimal(value))
        except InvalidOperation:
            errors.append(
                f"sample {index} for {names[0]} is not a finite nonnegative number"
            )
    return tuple(samples), errors


def _normalise_metrics(
    backend: str, data: Mapping[str, Any]
) -> tuple[dict[str, tuple[Decimal, ...]], tuple[str, ...]]:
    metrics: dict[str, tuple[Decimal, ...]] = {}
    errors: list[str] = []
    if backend == "samsung-hbm-pim":
        label = _clean_text(data.get("metric"))
        if "cycle" not in label or "wall" in label:
            errors.append(f"unrecognized Samsung metric {data.get('metric')!r}")
        metrics["cycles"], sample_errors = _metric_samples(
            data, data, ("cycles", "device_cycles")
        )
        errors.extend(sample_errors)
    elif backend == "sk-hynix-aim":
        primary = data.get("performance") or {}
        label = _clean_text(primary.get("metric"))
        if "wall" in label or not ("device-compute" in label or "cycle" in label):
            errors.append(f"unrecognized AiM metric {primary.get('metric')!r}")
        metrics["cycles"], sample_errors = _metric_samples(
            data, primary, ("cycles", "device_cycles")
        )
        errors.extend(sample_errors)
    elif backend == "upmem":
        label = _clean_text(data.get("metric"))
        if "cycle" not in label or "wall" in label:
            errors.append(f"unrecognized UPMEM metric {data.get('metric')!r}")
        metrics["cycles"], sample_errors = _metric_samples(
            data, data, ("cycles", "device_cycles")
        )
        errors.extend(sample_errors)
    elif backend == "apu-v1":
        primary = data.get("result") or {}
        label = _clean_text(primary.get("metric"))
        if "wall" in label or not ("device-compute" in label or "cycle" in label):
            errors.append(f"unrecognized APUv1 metric {primary.get('metric')!r}")
        metrics["cycles"], sample_errors = _metric_samples(
            data, primary, ("cycles", "device_cycles")
        )
        errors.extend(sample_errors)
    else:
        primary = data.get("result") or {}
        device_label = _clean_text(primary.get("metric"))
        wall_label = _clean_text(primary.get("wall_metric"))
        if "tick" not in device_label or "wall" in device_label:
            errors.append(f"unrecognized APUv2 device metric {primary.get('metric')!r}")
        if "wall" not in wall_label:
            errors.append(
                f"unrecognized APUv2 wall metric {primary.get('wall_metric')!r}"
            )
        metrics["device_ticks"], sample_errors = _metric_samples(
            data, primary, ("device_ticks", "cycles")
        )
        errors.extend(sample_errors)
        metrics["wall_us"], sample_errors = _metric_samples(
            data, primary, ("wall_us", "wall_microseconds")
        )
        errors.extend(sample_errors)
    return metrics, tuple(errors)


def _normalise_warmup(
    backend: str, data: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if backend not in {"apu-v1", "apu-v2"}:
        return None
    value = (data.get("result") or {}).get("warmup")
    return dict(value) if isinstance(value, Mapping) else None


def normalize_result(backend: str, data: Mapping[str, Any]) -> NormalizedRecord:
    if backend not in SUPPORTED_BACKENDS:
        raise RecordError(f"unsupported backend {backend!r}")
    if not isinstance(data, Mapping):
        raise RecordError("result root must be a JSON object")
    if backend == "samsung-hbm-pim":
        actual_backend = data.get("target")
        workload = data.get("kernel")
        revision = data.get("tenon_commit")
    elif backend == "sk-hynix-aim":
        actual_backend = data.get("backend")
        workload = data.get("benchmark")
        revision = (data.get("provenance") or {}).get("tenon_commit")
    elif backend == "upmem":
        actual_backend = "upmem"
        workload = data.get("kernel")
        revision = (data.get("provenance") or {}).get("allo_commit")
    else:
        actual_backend = data.get("backend")
        workload = data.get("kernel")
        revision = data.get("tenon_revision")
    workload = {"2mm": "two_mm", "3mm": "three_mm"}.get(workload, workload)
    shape = _canonical_shape(backend, data)
    metrics, metric_errors = _normalise_metrics(backend, data)
    return NormalizedRecord(
        backend=str(actual_backend) if actual_backend is not None else "",
        workload=str(workload) if workload is not None else None,
        dataset=data.get("dataset"),
        dtype=data.get("dtype"),
        shape=shape,
        semantic=_semantic(backend, data, shape),
        correctness=_correctness(data, allow_status_only=backend == "sk-hynix-aim"),
        metrics=metrics,
        metric_errors=metric_errors,
        producer_revision=str(revision) if revision is not None else None,
        warmup=_normalise_warmup(backend, data),
    )


def _schedule_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ScheduleError(f"{label} must be a contained relative path")
    return path.as_posix()


def _schedule_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ScheduleError(f"cannot read {label}: {error}") from error


def _schedule_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScheduleError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise ScheduleError(f"{label} must contain a JSON object")
    return value


def _recursive_numeric_values(
    value: Any, suffix: str, output: set[int | float]
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).endswith(suffix) and _is_number(nested):
                output.add(nested)
            _recursive_numeric_values(nested, suffix, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _recursive_numeric_values(nested, suffix, output)


def normalize_schedule(
    backend: str, data: Mapping[str, Any], result_path: str | Path
) -> dict[str, Any]:
    leaf = Path(result_path).parent
    if backend == "samsung-hbm-pim":
        fibers = set()
        crfs = []
        for leg in data.get("legs") or []:
            relative = _schedule_relative_path(leg.get("compiled_crf"), "compiled_crf")
            text = _schedule_text(leaf / relative, relative)
            if "EVEN_BANK" in text:
                fibers.add("even_bank")
            if "ODD_BANK" in text:
                fibers.add("odd_bank")
            crfs.append(
                {
                    "leg": leg.get("name"),
                    "compiled_crf": relative,
                    "physical_shape": leg.get("tenon_physical_shape"),
                }
            )
        if not crfs or fibers != {"even_bank", "odd_bank"}:
            raise ScheduleError("Samsung schedule must expose dual-fiber compiled CRFs")
        return {
            "kind": "dual_fiber_compiled_crf",
            "fibers": ["even_bank", "odd_bank"],
            "crfs": crfs,
        }
    if backend == "sk-hynix-aim":
        trace_relative = "compiled/aim.trace"
        trace = _schedule_text(leaf / trace_relative, trace_relative)
        if re.search(r"\bMAC_ABK\b", trace) is None:
            raise ScheduleError("AiM trace does not contain MAC_ABK")
        specification = _schedule_json(
            leaf / "workload_spec.json", "workload_spec.json"
        )
        mapping_text = str(specification.get("mapping", ""))
        mapping_match = re.search(r"mapping=\[(\d+)\]", mapping_text)
        banks_match = re.search(r"spans\s+(\d+)\s+banks", mapping_text)
        if mapping_match is None or banks_match is None:
            raise ScheduleError(
                "AiM workload mapping does not identify channel/bank scope"
            )
        return {
            "kind": "all_bank_mac_abk",
            "mapping": [int(mapping_match.group(1))],
            "opcode": "MAC_ABK",
            "bank_scope": "all_bank",
            "banks_per_channel": int(banks_match.group(1)),
            "trace": trace_relative,
        }
    if backend == "upmem":
        tiles = []
        for json_path in sorted((leaf / "compiled").glob("*.json")):
            tile = _schedule_json(json_path, json_path.name)
            source_path = json_path.with_suffix(".dpu.c")
            if not source_path.is_file():
                raise ScheduleError(f"missing compiled tile source {source_path.name}")
            tiles.append(
                {
                    **tile,
                    "path": json_path.relative_to(leaf).as_posix(),
                    "source": source_path.relative_to(leaf).as_posix(),
                }
            )
        if not tiles:
            raise ScheduleError("UPMEM schedule has no compiled tile JSON")
        return {"kind": "compiled_tile_json", "tiles": tiles}
    if backend == "apu-v1":
        profiles = []
        for leg in (data.get("result") or {}).get("legs") or []:
            for profile in leg.get("profiles") or []:
                profile_mlir = _schedule_relative_path(
                    profile.get("compiled_profile_mlir"), "compiled_profile_mlir"
                )
                gvml = _schedule_relative_path(
                    profile.get("compiled_gvml"), "compiled_gvml"
                )
                if not (leaf / profile_mlir).is_file() or not (leaf / gvml).is_file():
                    raise ScheduleError("APUv1 compiled profile evidence is missing")
                shape = profile.get("profile_shape")
                if not isinstance(shape, Mapping):
                    raise ScheduleError("APUv1 profile_shape must be an object")
                profiles.append(
                    {
                        "leg": leg.get("name"),
                        "profile": profile.get("name"),
                        "plan": profile.get("plan"),
                        "shape": shape,
                        "row_tile": shape.get("M"),
                        "row_waves": profile.get("row_waves"),
                        "profile_mlir": profile_mlir,
                        "gvml": gvml,
                    }
                )
        if not profiles:
            raise ScheduleError("APUv1 schedule has no compiled profiles")
        return {"kind": "profiled_plan", "profiles": profiles}
    if backend == "apu-v2":
        graph_relative = "compiled/execution_graph.json"
        graph = _schedule_json(leaf / graph_relative, graph_relative)
        metadata = graph.get("metadata") or {}
        transport = metadata.get("transport_schedule") or {}
        batch_columns: set[int | float] = set()
        reduction_tiles: set[int | float] = set()
        _recursive_numeric_values(graph, "batch_columns", batch_columns)
        _recursive_numeric_values(graph, "reduction_tile", reduction_tiles)
        profile_kinds = set()
        profile_schedule_kinds = set()
        profile_evidence = []
        profile_records = []
        for profile in (data.get("result") or {}).get("profiles") or []:
            if profile.get("kind"):
                profile_kinds.add(profile["kind"])
            if _is_number(profile.get("batch_columns")):
                batch_columns.add(profile["batch_columns"])
            relative = _schedule_relative_path(profile.get("profile"), "profile")
            if not (leaf / relative).is_file():
                raise ScheduleError(f"missing APUv2 profile evidence {relative}")
            profile_evidence.append(relative)
            if not relative.endswith(".board.log"):
                raise ScheduleError(
                    f"APUv2 profile evidence has an unknown name: {relative}"
                )
            record_relative = relative[: -len(".board.log")] + ".json"
            profile_record = _schedule_json(leaf / record_relative, record_relative)
            profile_records.append(record_relative)
            _recursive_numeric_values(profile_record, "batch_columns", batch_columns)
            _recursive_numeric_values(profile_record, "reduction_tile", reduction_tiles)
            profile_schedule = (profile_record.get("extra") or {}).get("schedule") or {}
            if profile_schedule.get("kind"):
                profile_schedule_kinds.add(profile_schedule["kind"])
        transport_kind = transport.get("kind") or metadata.get("program")
        if not transport_kind or not profile_evidence:
            raise ScheduleError("APUv2 schedule lacks transport or profile evidence")
        return {
            "execution_graph": graph_relative,
            "transport_kind": transport_kind,
            "profile_kinds": sorted(profile_kinds),
            "profile_schedule_kinds": sorted(profile_schedule_kinds),
            "batch_columns": sorted(batch_columns),
            "reduction_tiles": sorted(reduction_tiles),
            "profile_evidence": sorted(profile_evidence),
            "profile_records": sorted(profile_records),
        }
    raise ScheduleError(f"unsupported backend {backend!r}")


def schedule_artifact_paths(
    backend: str, schedule: Mapping[str, Any]
) -> tuple[str, ...]:
    paths = []
    if backend == "samsung-hbm-pim":
        paths.extend(item["compiled_crf"] for item in schedule.get("crfs") or [])
    elif backend == "sk-hynix-aim":
        paths.append(schedule.get("trace"))
    elif backend == "upmem":
        for tile in schedule.get("tiles") or []:
            paths.extend((tile.get("path"), tile.get("source")))
    elif backend == "apu-v1":
        for profile in schedule.get("profiles") or []:
            paths.extend((profile.get("profile_mlir"), profile.get("gvml")))
    elif backend == "apu-v2":
        paths.append(schedule.get("execution_graph"))
    normalized = [_schedule_relative_path(path, "schedule artifact") for path in paths]
    if not normalized:
        raise ScheduleError("schedule has no executable artifacts")
    return tuple(sorted(set(normalized)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_error(message: str) -> None:
    raise ManifestError(message)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        _manifest_error("manifest root must be a JSON object")
    if manifest.get("schema_version") != 1:
        _manifest_error("schema_version must be 1")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        _manifest_error("source must be an object")
    revision = source.get("artifact_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        _manifest_error("source.artifact_revision must be a 40-character lowercase SHA")
    if source.get("worktree_clean") is not True:
        _manifest_error("source.worktree_clean must be true")
    policies = manifest.get("policies")
    if not isinstance(policies, Mapping):
        _manifest_error("policies must be an object")
    for name, required in DEFAULT_POLICIES.items():
        if policies.get(name) != required:
            _manifest_error(
                f"policy {name!r} does not match the checker safety contract"
            )
    records = manifest.get("records")
    if not isinstance(records, list):
        _manifest_error("records must be an array")
    if manifest.get("record_count") != len(records):
        _manifest_error("record_count does not match records length")
    seen = set()
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            _manifest_error(f"{prefix} must be an object")
        key = record.get("key")
        if not isinstance(key, str) or key.count("/") != 1:
            _manifest_error(f"{prefix}.key must be backend/workload")
        backend, workload = key.split("/", 1)
        if backend not in SUPPORTED_BACKENDS or not workload:
            _manifest_error(
                f"{prefix}.key names an unsupported backend or empty workload"
            )
        if key in seen:
            _manifest_error(f"duplicate record key {key!r}")
        seen.add(key)
        result_path = record.get("result_path")
        if not isinstance(result_path, str):
            _manifest_error(f"{prefix}.result_path must be a string")
        pure_path = PurePosixPath(result_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            _manifest_error(f"{prefix}.result_path must be relative and contained")
        expected_prefix = PurePosixPath("polybench", backend, "tenon", workload)
        if pure_path.parent != expected_prefix or pure_path.name != "result.json":
            _manifest_error(f"{prefix}.result_path does not match its key")
        for field in ("dataset", "dtype"):
            if not isinstance(record.get(field), str) or not record[field]:
                _manifest_error(f"{prefix}.{field} must be a nonempty string")
        if not isinstance(record.get("shape"), Mapping):
            _manifest_error(f"{prefix}.shape must be an object")
        semantic = record.get("semantic")
        if not isinstance(semantic, Mapping):
            _manifest_error(f"{prefix}.semantic must be an object")
        status = semantic.get("status")
        omitted = semantic.get("omitted")
        if status not in {"full", "partial"}:
            _manifest_error(f"{prefix}.semantic.status must be full or partial")
        if not semantic.get("scope"):
            _manifest_error(f"{prefix}.semantic.scope must be explicit")
        if not isinstance(omitted, list):
            _manifest_error(f"{prefix}.semantic.omitted must be an array")
        if status == "full" and omitted:
            _manifest_error(f"{prefix} full semantics cannot omit work")
        if status == "partial" and not omitted:
            _manifest_error(f"{prefix} partial semantics must name omitted work")
        if record.get("correctness") != "pass":
            _manifest_error(f"{prefix}.correctness must preserve a passing incumbent")
        metrics = record.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            _manifest_error(f"{prefix}.metrics must be a nonempty array")
        metric_kinds = set()
        for metric_index, metric in enumerate(metrics):
            metric_prefix = f"{prefix}.metrics[{metric_index}]"
            if not isinstance(metric, Mapping):
                _manifest_error(f"{metric_prefix} must be an object")
            kind = metric.get("kind")
            if (
                kind not in {"cycles", "device_ticks", "wall_us"}
                or kind in metric_kinds
            ):
                _manifest_error(f"{metric_prefix}.kind is invalid or duplicated")
            metric_kinds.add(kind)
            try:
                baseline = _decimal(metric.get("baseline"))
            except InvalidOperation:
                _manifest_error(
                    f"{metric_prefix}.baseline must be finite and nonnegative"
                )
            if baseline <= 0:
                _manifest_error(f"{metric_prefix}.baseline must be positive")
            evidence = metric.get("evidence")
            if evidence not in EVIDENCE_CLASSES:
                _manifest_error(f"{metric_prefix}.evidence is invalid")
            policy_name = metric.get("policy")
            policy = policies.get(policy_name)
            if not isinstance(policy, Mapping):
                _manifest_error(f"{metric_prefix}.policy is unknown")
            if policy.get("evidence_class") != evidence:
                _manifest_error(f"{metric_prefix} evidence and policy disagree")
        expected_kinds = (
            {"device_ticks", "wall_us"} if backend == "apu-v2" else {"cycles"}
        )
        if metric_kinds != expected_kinds:
            _manifest_error(f"{prefix} has the wrong metric kinds for {backend}")
        producer_revision = record.get("producer_revision")
        if (
            not isinstance(producer_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", producer_revision) is None
        ):
            _manifest_error(f"{prefix}.producer_revision must be a lowercase SHA")
        if not isinstance(record.get("schedule"), Mapping) or not record["schedule"]:
            _manifest_error(f"{prefix}.schedule must be a nonempty object")
        artifacts = record.get("schedule_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            _manifest_error(f"{prefix}.schedule_artifacts must be a nonempty array")
        artifact_paths = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_prefix = f"{prefix}.schedule_artifacts[{artifact_index}]"
            if not isinstance(artifact, Mapping):
                _manifest_error(f"{artifact_prefix} must be an object")
            artifact_path = artifact.get("path")
            if not isinstance(artifact_path, str):
                _manifest_error(f"{artifact_prefix}.path must be a string")
            pure_artifact_path = PurePosixPath(artifact_path)
            if (
                pure_artifact_path.is_absolute()
                or ".." in pure_artifact_path.parts
                or not pure_artifact_path.is_relative_to(expected_prefix)
            ):
                _manifest_error(f"{artifact_prefix}.path must stay within its workload")
            if artifact_path in artifact_paths:
                _manifest_error(f"{artifact_prefix}.path is duplicated")
            artifact_paths.add(artifact_path)
            digest = artifact.get("sha256")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                _manifest_error(f"{artifact_prefix}.sha256 must be a lowercase SHA-256")
        try:
            required_paths = {
                (expected_prefix / path).as_posix()
                for path in schedule_artifact_paths(backend, record["schedule"])
            }
        except (KeyError, ScheduleError, TypeError) as error:
            _manifest_error(f"{prefix}.schedule is invalid: {error}")
        if backend == "apu-v2":
            compiled_prefix = expected_prefix / "compiled"
            if not required_paths <= artifact_paths:
                _manifest_error(
                    f"{prefix}.schedule_artifacts do not cover the schedule"
                )
            if any(
                not PurePosixPath(path).is_relative_to(compiled_prefix)
                for path in artifact_paths
            ):
                _manifest_error(
                    f"{prefix}.schedule_artifacts must stay within its compiled directory"
                )
            if any(
                PurePosixPath(path).name == "tenon_sources.json"
                for path in artifact_paths
            ):
                _manifest_error(
                    f"{prefix}.schedule_artifacts must not pin tenon_sources.json"
                )
        elif artifact_paths != required_paths:
            _manifest_error(f"{prefix}.schedule_artifacts do not cover the schedule")


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load {manifest_path}: {error}") from error
    validate_manifest(data)
    return data


def _omission_terms(value: Any) -> set[str]:
    text = json.dumps(value, sort_keys=True).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text)) - {
        "all",
        "and",
        "host",
        "listed",
        "on",
        "outside",
        "the",
    }
    return {
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in tokens
    }


def _median(samples: Sequence[Decimal]) -> Decimal:
    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _performance_findings(
    key: str,
    metric: Mapping[str, Any],
    samples: Sequence[Decimal],
    policy: Mapping[str, Any],
    validate_baseline: bool,
) -> list[Finding]:
    kind = str(metric["kind"])
    baseline = _decimal(metric["baseline"])
    if not samples:
        return [Finding(key, "metric_mismatch", f"missing {kind} samples")]
    if validate_baseline:
        if any(sample != baseline for sample in samples):
            observed = ",".join(str(sample) for sample in samples)
            return [
                Finding(
                    key,
                    "baseline_metric_mismatch",
                    f"{kind} source [{observed}] != manifest {baseline}",
                )
            ]
        return []

    findings = []
    minimum = int(policy["min_samples"])
    enough_samples = len(samples) >= minimum
    if not enough_samples:
        findings.append(
            Finding(
                key,
                "insufficient_repetitions",
                f"{kind} has {len(samples)} sample(s); policy requires {minimum}",
                "insufficient",
            )
        )
    sample_ratio = policy.get("max_sample_ratio")
    if sample_ratio is not None:
        sample_limit = baseline * Decimal(str(sample_ratio))
        worst = max(samples)
        if worst > sample_limit:
            findings.append(
                Finding(
                    key,
                    "performance_regression",
                    f"{kind} worst {worst} exceeds {sample_ratio}x incumbent {baseline}",
                )
            )
    if enough_samples:
        aggregate = max(samples) if policy["aggregate"] == "max" else _median(samples)
        ratio = Decimal(str(policy["max_ratio"]))
        if aggregate > baseline * ratio:
            findings.append(
                Finding(
                    key,
                    "performance_regression",
                    f"{kind} {policy['aggregate']} {aggregate} exceeds {ratio}x incumbent {baseline}",
                )
            )
    return findings


def _warmup_findings(
    key: str,
    expected: Mapping[str, Any],
    candidate: NormalizedRecord,
    policies: Mapping[str, Any],
    validate_baseline: bool,
) -> list[Finding]:
    if validate_baseline:
        return []
    hardware_policies = [
        policies[metric["policy"]]
        for metric in expected["metrics"]
        if str(metric["evidence"]).startswith("hardware_")
    ]
    if not hardware_policies:
        return []
    required = {int(policy["warmup_samples"]) for policy in hardware_policies}
    warmup = candidate.warmup or {}
    count = warmup.get("count")
    discarded = warmup.get("discarded")
    if (
        required != {1}
        or isinstance(count, bool)
        or count != 1
        or discarded is not True
    ):
        return [
            Finding(
                key,
                "insufficient_warmup",
                "hardware evidence requires one explicitly discarded warm-up",
                "insufficient",
            )
        ]
    return []


def _schedule_artifact_findings(
    key: str, expected: Mapping[str, Any], root: Path
) -> list[Finding]:
    findings = []
    for artifact in expected["schedule_artifacts"]:
        relative = artifact["path"]
        path = root / relative
        try:
            actual = sha256_file(path)
        except OSError as error:
            findings.append(
                Finding(
                    key,
                    "schedule_artifact_mismatch",
                    f"cannot read schedule artifact {relative}: {error}",
                )
            )
            continue
        if actual != artifact["sha256"]:
            findings.append(
                Finding(
                    key,
                    "schedule_artifact_mismatch",
                    f"schedule artifact hash changed: {relative}",
                )
            )
    return findings


def check_artifacts(
    manifest: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    validate_baseline: bool = False,
) -> CheckReport:
    if isinstance(manifest, (str, Path)):
        manifest_data = load_manifest(manifest)
    else:
        validate_manifest(manifest)
        manifest_data = dict(manifest)
    root = Path(artifacts_root)
    findings: list[Finding] = []
    for expected in manifest_data["records"]:
        key = expected["key"]
        backend, workload = key.split("/", 1)
        result_path = root / expected["result_path"]
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            findings.append(
                Finding(key, "missing_result", f"missing {expected['result_path']}")
            )
            continue
        except (OSError, json.JSONDecodeError) as error:
            findings.append(
                Finding(key, "invalid_result", f"cannot read result: {error}")
            )
            continue
        try:
            candidate = normalize_result(backend, data)
        except RecordError as error:
            findings.append(Finding(key, "invalid_result", str(error)))
            continue
        try:
            candidate_schedule = normalize_schedule(backend, data, result_path)
        except ScheduleError as error:
            findings.append(Finding(key, "schedule_mismatch", str(error)))
        else:
            if candidate_schedule != expected["schedule"]:
                findings.append(
                    Finding(key, "schedule_mismatch", "schedule descriptor changed")
                )
        findings.extend(_schedule_artifact_findings(key, expected, root))
        if candidate.backend != backend or candidate.workload != workload:
            findings.append(
                Finding(
                    key,
                    "identity_mismatch",
                    f"record identifies {candidate.backend}/{candidate.workload}",
                )
            )
        if candidate.dataset != expected["dataset"]:
            findings.append(
                Finding(
                    key,
                    "dataset_mismatch",
                    f"dataset {candidate.dataset!r} != {expected['dataset']!r}",
                )
            )
        if candidate.dtype != expected["dtype"]:
            findings.append(
                Finding(
                    key,
                    "dtype_mismatch",
                    f"dtype {candidate.dtype!r} != {expected['dtype']!r}",
                )
            )
        if candidate.shape != expected["shape"]:
            findings.append(
                Finding(key, "shape_mismatch", "logical shape or scope changed")
            )
        expected_semantic = expected["semantic"]
        if candidate.semantic.get("status") != expected_semantic["status"]:
            findings.append(
                Finding(
                    key,
                    "semantic_status_mismatch",
                    f"semantic status {candidate.semantic.get('status')!r} != {expected_semantic['status']!r}",
                )
            )
        else:
            if candidate.semantic.get("scope") != expected_semantic["scope"]:
                findings.append(
                    Finding(key, "semantic_scope_mismatch", "semantic scope changed")
                )
            candidate_omitted = candidate.semantic.get("omitted")
            expected_omitted = expected_semantic["omitted"]
            if candidate_omitted != expected_omitted:
                code = "omitted_semantics_mismatch"
                if _omission_terms(candidate_omitted) > _omission_terms(
                    expected_omitted
                ):
                    code = "omitted_semantics_expanded"
                findings.append(Finding(key, code, "omitted semantics changed"))
        if candidate.correctness != expected["correctness"]:
            findings.append(
                Finding(
                    key,
                    "correctness_downgrade",
                    f"correctness {candidate.correctness!r} != {expected['correctness']!r}",
                )
            )
        for error in candidate.metric_errors:
            findings.append(Finding(key, "metric_mismatch", error))
        findings.extend(
            _warmup_findings(
                key,
                expected,
                candidate,
                manifest_data["policies"],
                validate_baseline,
            )
        )
        for metric in expected["metrics"]:
            policy = manifest_data["policies"][metric["policy"]]
            findings.extend(
                _performance_findings(
                    key,
                    metric,
                    candidate.metrics.get(metric["kind"], ()),
                    policy,
                    validate_baseline,
                )
            )
    return CheckReport(checked=len(manifest_data["records"]), findings=findings)


def validate_source_checkout(
    manifest: Mapping[str, Any], artifacts_root: str | Path
) -> list[Finding]:
    root = Path(artifacts_root)
    findings = []
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return [
            Finding(
                "<baseline>",
                "source_checkout",
                f"cannot inspect source checkout: {error}",
            )
        ]
    expected_revision = manifest["source"]["artifact_revision"]
    if revision != expected_revision:
        findings.append(
            Finding(
                "<baseline>",
                "source_revision",
                f"checkout {revision} != manifest {expected_revision}",
            )
        )
    if dirty:
        findings.append(
            Finding("<baseline>", "source_dirty", "source checkout is not clean")
        )
    return findings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check PIM schedule artifacts for semantic or performance regressions."
    )
    parser.add_argument(
        "artifacts_root",
        nargs="?",
        help="candidate artifact root (defaults to the source checkout in baseline mode)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--validate-baseline",
        action="store_true",
        help=f"validate the clean incumbent source checkout (default: {DEFAULT_BASELINE_ROOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.validate_baseline and not args.artifacts_root:
        parser.error("artifacts_root is required unless --validate-baseline is used")
    artifacts_root = (
        Path(args.artifacts_root) if args.artifacts_root else DEFAULT_BASELINE_ROOT
    )
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as error:
        print(f"FAIL manifest: {error}", file=sys.stderr)
        return 1
    report = check_artifacts(
        manifest,
        artifacts_root,
        validate_baseline=args.validate_baseline,
    )
    if args.validate_baseline:
        report.findings.extend(validate_source_checkout(manifest, artifacts_root))
    for finding in report.findings:
        label = "INSUFFICIENT" if finding.severity == "insufficient" else "FAIL"
        print(f"{label} {finding.key}: {finding.message}", file=sys.stderr)
    if report.ok:
        mode = "baseline" if args.validate_baseline else "candidate"
        print(f"PASS {mode}: {report.checked} incumbents checked")
        return 0
    print(
        f"FAIL: {len(report.failures)} failure(s), "
        f"{len(report.insufficient)} insufficient measurement(s), "
        f"{report.checked} incumbent(s) checked",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
