#!/usr/bin/env python3
"""Report held-out transport-model error from APUv2 artifact records.

The artifact corpus contains measurements for one persistent GEMM schedule at
each structural shape.  This report therefore evaluates point predictions
across shapes.  It deliberately does not reinterpret singleton candidate sets
as within-shape schedule-ranking experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

from allo.pim.costs.apu_g2 import (
    APUG2_GEMM_MAX_BATCH_COLUMNS,
    APUG2_GEMM_MAX_REDUCTION_TILE,
    APUG2_GEMM_WALL_ANCHOR_PHASES_US,
    APUG2_GEMM_WALL_FINGERPRINT_DATA,
    APUG2_GEMM_WALL_MODEL_REVISION,
    APUG2_GEMM_WALL_SAMPLE_REVISION,
    estimate_apu_g2_u16_gemm_wall_us,
)


DEFAULT_ARTIFACTS_ROOT = Path("/home/nz264/shared/tenon-artifacts")
PROFILE_KIND = "persistent_column_batched_gemm"
SCHEDULE_KIND = "column_batched_u16_gemm"
WALL_METRIC = "measured host wall: h2d + host task dispatch + d2h"
TRAINING_ANCHORS = (
    (1000, 1200, 1100),
    (1900, 2100, 1),
)
TRAINING_ANCHOR_CASES = tuple((*shape, 1, 1) for shape in TRAINING_ANCHORS)

_APU_V2_BACKENDS = frozenset(("apu-v2", "apu_v2"))
_SCHEDULE_FIELDS = frozenset(
    (
        "kind",
        "batch_columns",
        "reduction_tile",
        "resident_accumulator",
        "contiguous_readback",
    )
)
_MODEL_PARAMETER_KEYS = (
    "cold_batch_h2d_bytes_per_us",
    "steady_batch_h2d_bytes_per_us",
    "h2d_call_us",
    "task_call_us",
    "scalar_mac_us",
    "d2h_bytes_per_us",
    "d2h_call_us",
    "physical_rows",
    "element_bytes",
    "index_uploads",
    "index_vectors_per_group",
    "max_batch_columns",
    "max_reduction_tile",
)


class ArtifactReportError(ValueError):
    """Raised when artifact evidence cannot support a trustworthy report."""


@dataclass(frozen=True)
class _Measurement:
    h2d_us: float
    host_task_us: float
    d2h_us: float
    wall_us: float

    def manifest(self) -> dict[str, float]:
        return {
            "h2d_us": self.h2d_us,
            "host_task_us": self.host_task_us,
            "d2h_us": self.d2h_us,
            "wall_us": self.wall_us,
        }


@dataclass(frozen=True)
class _Observation:
    shape: tuple[int, int, int]
    epilogue: tuple[int, int]
    schedule_json: str
    measurement: _Measurement
    measurement_fingerprint: str
    source: Mapping[str, object]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(namespace: str, value: object) -> str:
    payload = _canonical_json({"namespace": namespace, "value": value})
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ArtifactReportError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactReportError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactReportError(f"{label} {path} must contain a JSON object")
    return dict(value)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArtifactReportError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ArtifactReportError(f"{field} must be a positive integer")
    return result


def _u16_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArtifactReportError(f"{field} must be a uint16 integer")
    result = int(value)
    if not 0 <= result <= 0xFFFF:
        raise ArtifactReportError(f"{field} must be a uint16 integer")
    return result


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ArtifactReportError(f"{field} must be a positive finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ArtifactReportError(
            f"{field} must be a positive finite number"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ArtifactReportError(f"{field} must be a positive finite number")
    return result


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactReportError(f"{field} must be an object")
    return value


def _require_profiles(value: object, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactReportError(f"{field} must be an array")
    profiles = []
    for index, profile in enumerate(value):
        if not isinstance(profile, Mapping):
            raise ArtifactReportError(f"{field}[{index}] must be an object")
        kind = profile.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ArtifactReportError(
                f"{field}[{index}].kind must be a nonempty string"
            )
        profiles.append(profile)
    return tuple(profiles)


def _require_pass(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ArtifactReportError(f"{field} must prove passing correctness")
    normalized = value.strip().lower()
    if "pass" not in normalized or any(
        token in normalized for token in ("fail", "error", "incorrect")
    ):
        raise ArtifactReportError(f"{field} must prove passing correctness")


def _contained_file(base: Path, relative: object, field: str) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative:
        raise ArtifactReportError(f"{field} must be a nonempty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ArtifactReportError(f"{field} must stay within its result directory")
    path = base.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ArtifactReportError(
            f"{field} does not name a contained file: {relative}"
        ) from error
    if not resolved.is_file():
        raise ArtifactReportError(f"{field} does not name a file: {relative}")
    return path, pure.as_posix()


def _shape_manifest(shape: tuple[int, int, int]) -> dict[str, int]:
    rows, reduction, columns = shape
    return {"M": rows, "K": reduction, "N": columns}


def _shape_from_profile(profile: Mapping[str, Any], field: str) -> tuple[int, int, int]:
    shape = _require_mapping(profile.get("profile_shape"), f"{field}.profile_shape")
    if set(shape) != {"M", "K", "N"}:
        raise ArtifactReportError(
            f"{field}.profile_shape must contain exactly M, K, and N"
        )
    return tuple(
        _positive_int(shape[dimension], f"{field}.profile_shape.{dimension}")
        for dimension in ("M", "K", "N")
    )


def _schedule_from_profile_record(
    record: Mapping[str, Any],
    *,
    batch_columns: int,
    field: str,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    backend = record.get("backend")
    if backend is not None and backend not in _APU_V2_BACKENDS:
        raise ArtifactReportError(f"{field}.backend is not APUv2")
    extra = _require_mapping(record.get("extra"), f"{field}.extra")
    schedule = _require_mapping(extra.get("schedule"), f"{field}.extra.schedule")
    unknown = sorted(set(schedule) - _SCHEDULE_FIELDS)
    if unknown:
        raise ArtifactReportError(
            f"{field}.extra.schedule has unsupported fields: {', '.join(unknown)}"
        )
    if schedule.get("kind") != SCHEDULE_KIND:
        raise ArtifactReportError(
            f"{field}.extra.schedule.kind must be {SCHEDULE_KIND!r}"
        )
    recorded_batch = _positive_int(
        schedule.get("batch_columns"),
        f"{field}.extra.schedule.batch_columns",
    )
    if recorded_batch != batch_columns:
        raise ArtifactReportError(
            f"{field} batch_columns disagrees with its result profile"
        )
    reduction_tile = _positive_int(
        schedule.get("reduction_tile"),
        f"{field}.extra.schedule.reduction_tile",
    )
    if recorded_batch != APUG2_GEMM_MAX_BATCH_COLUMNS:
        raise ArtifactReportError("transport report requires the current B=31 schedule")
    if reduction_tile != APUG2_GEMM_MAX_REDUCTION_TILE:
        raise ArtifactReportError(
            "transport report requires the current reduction_tile=128 schedule"
        )

    normalized: dict[str, object] = {
        "kind": SCHEDULE_KIND,
        "batch_columns": recorded_batch,
        "reduction_tile": reduction_tile,
    }
    for flag in ("resident_accumulator", "contiguous_readback"):
        if flag not in schedule:
            continue
        value = schedule[flag]
        if value is not True:
            raise ArtifactReportError(f"{field}.extra.schedule.{flag} must be true")
        normalized[flag] = True
    return normalized, extra


def _measurement_from_profile_record(
    profile: Mapping[str, Any],
    extra: Mapping[str, Any],
    *,
    field: str,
) -> _Measurement:
    timings = _require_mapping(
        extra.get("host_timings_us"), f"{field}.extra.host_timings_us"
    )
    h2d_us = _positive_float(timings.get("h2d"), f"{field}.extra.host_timings_us.h2d")
    host_task_us = _positive_float(
        timings.get("host_task"),
        f"{field}.extra.host_timings_us.host_task",
    )
    d2h_us = _positive_float(timings.get("d2h"), f"{field}.extra.host_timings_us.d2h")
    wall_us = _positive_float(
        profile.get("profile_wall_microseconds"),
        f"{field}.profile_wall_microseconds",
    )
    phase_sum = math.fsum((h2d_us, host_task_us, d2h_us))
    if not math.isclose(phase_sum, wall_us, rel_tol=0.0, abs_tol=0.005):
        raise ArtifactReportError(
            f"{field} wall measurement does not equal h2d + host_task + d2h"
        )
    return _Measurement(h2d_us, host_task_us, d2h_us, wall_us)


def _profile_record_relative(profile_relative: str, field: str) -> str:
    suffix = ".board.log"
    if not profile_relative.endswith(suffix):
        raise ArtifactReportError(f"{field}.profile must end in {suffix}")
    return profile_relative[: -len(suffix)] + ".json"


def _contains_target_profile(data: Mapping[str, Any]) -> bool:
    result = data.get("result")
    if not isinstance(result, Mapping):
        return False
    profiles = result.get("profiles")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        return False
    return any(
        isinstance(profile, Mapping) and profile.get("kind") == PROFILE_KIND
        for profile in profiles
    )


def _prediction_for(
    shape: tuple[int, int, int], schedule: Mapping[str, object], field: str
) -> dict[str, float | int | str]:
    try:
        prediction = estimate_apu_g2_u16_gemm_wall_us(
            *shape,
            batch_columns=int(schedule["batch_columns"]),
            reduction_tile=int(schedule["reduction_tile"]),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactReportError(
            f"{field} is outside the current APUg2 estimator domain: {error}"
        ) from error
    for phase in ("h2d_us", "host_task_us", "d2h_us", "wall_us"):
        _positive_float(prediction.get(phase), f"estimator.{phase}")
    if prediction.get("calibration") != APUG2_GEMM_WALL_SAMPLE_REVISION:
        raise ArtifactReportError("current estimator returned unknown calibration")
    return prediction


def _observation_from_profile(
    root: Path,
    result_path: Path,
    result_data: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> _Observation:
    result_relative = result_path.relative_to(root).as_posix()
    field = f"{result_relative}:{profile.get('profile', '<missing>')}"
    _positive_int(profile.get("count", 1), f"{field}.count")
    _require_pass(profile.get("correctness"), f"{field}.correctness")
    shape = _shape_from_profile(profile, field)
    batch_columns = _positive_int(
        profile.get("batch_columns"), f"{field}.batch_columns"
    )
    epilogue = (
        _u16_int(profile.get("alpha"), f"{field}.alpha"),
        _u16_int(profile.get("beta"), f"{field}.beta"),
    )

    board_path, board_relative = _contained_file(
        result_path.parent,
        profile.get("profile"),
        f"{field}.profile",
    )
    profile_record_relative = _profile_record_relative(board_relative, field)
    profile_record_path, profile_record_relative = _contained_file(
        result_path.parent,
        profile_record_relative,
        f"{field}.profile_record",
    )
    profile_record = _load_json_object(profile_record_path, "APUv2 profile record")
    schedule, extra = _schedule_from_profile_record(
        profile_record,
        batch_columns=batch_columns,
        field=field,
    )
    measurement = _measurement_from_profile_record(profile, extra, field=field)
    prediction = _prediction_for(shape, schedule, field)

    profile_tasks = _positive_int(
        profile.get("profile_hardware_tasks"),
        f"{field}.profile_hardware_tasks",
    )
    record_tasks = _positive_int(
        extra.get("hardware_tasks"), f"{field}.extra.hardware_tasks"
    )
    predicted_tasks = int(prediction["hardware_tasks"])
    if profile_tasks != record_tasks or record_tasks != predicted_tasks:
        raise ArtifactReportError(
            f"{field} hardware_tasks disagrees with the structural schedule"
        )
    if "weight_uploads" in extra:
        weight_uploads = _positive_int(
            extra["weight_uploads"], f"{field}.extra.weight_uploads"
        )
        if weight_uploads != int(prediction["weight_uploads"]):
            raise ArtifactReportError(
                f"{field} weight_uploads disagrees with the structural schedule"
            )

    schedule_json = _canonical_json(schedule)
    shape_manifest = _shape_manifest(shape)
    structural_manifest = {
        "shape": shape_manifest,
        "epilogue": {"alpha": epilogue[0], "beta": epilogue[1]},
        "schedule": schedule,
    }
    measurement_manifest = {
        "structural_identity_fingerprint": _fingerprint(
            "apu-g2-transport-structural-identity-v1", structural_manifest
        ),
        "measurement": measurement.manifest(),
    }
    measurement_fingerprint = _fingerprint(
        "apu-g2-transport-measurement-v1", measurement_manifest
    )

    source_manifest: dict[str, object] = {
        "result_path": result_relative,
        "profile_record_path": profile_record_path.relative_to(root).as_posix(),
        "board_log_path": board_path.relative_to(root).as_posix(),
        "result_content_fingerprint": _fingerprint(
            "canonical-result-json-v1", result_data
        ),
        "profile_entry_fingerprint": _fingerprint(
            "canonical-result-profile-v1", profile
        ),
        "profile_record_content_fingerprint": _fingerprint(
            "canonical-profile-json-v1", profile_record
        ),
        "board_log_sha256": _sha256_file(board_path),
        "measurement_fingerprint": measurement_fingerprint,
    }
    source_manifest["provenance_fingerprint"] = _fingerprint(
        "apu-g2-transport-provenance-v1", source_manifest
    )
    return _Observation(
        shape=shape,
        epilogue=epilogue,
        schedule_json=schedule_json,
        measurement=measurement,
        measurement_fingerprint=measurement_fingerprint,
        source=source_manifest,
    )


def load_transport_observations(
    artifacts_root: str | Path,
) -> tuple[_Observation, ...]:
    """Load trustworthy persistent-GEMM observations from an artifact tree."""

    root_path = Path(artifacts_root)
    try:
        root = root_path.resolve(strict=True)
    except OSError as error:
        raise ArtifactReportError(
            f"artifact root does not exist: {root_path}"
        ) from error
    if not root.is_dir():
        raise ArtifactReportError(f"artifact root is not a directory: {root_path}")

    result_paths = sorted(
        root.rglob("result.json"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not result_paths:
        raise ArtifactReportError("artifact tree contains no result.json records")

    observations = []
    for result_path in result_paths:
        result_data = _load_json_object(result_path, "artifact result")
        backend = result_data.get("backend")
        if backend not in _APU_V2_BACKENDS:
            if _contains_target_profile(result_data):
                relative = result_path.relative_to(root).as_posix()
                raise ArtifactReportError(
                    f"{relative} declares {PROFILE_KIND!r} without APUv2 backend"
                )
            continue

        result = result_data.get("result")
        if result is None:
            continue
        result = _require_mapping(
            result,
            f"{result_path.relative_to(root).as_posix()}.result",
        )
        profiles = _require_profiles(
            result.get("profiles"),
            f"{result_path.relative_to(root).as_posix()}.result.profiles",
        )
        target_profiles = tuple(
            profile for profile in profiles if profile["kind"] == PROFILE_KIND
        )
        if not target_profiles:
            continue

        status = result_data.get("status")
        if not isinstance(status, str) or "measured" not in status.lower():
            raise ArtifactReportError(
                f"{result_path.relative_to(root).as_posix()} has target profiles "
                "without measured status"
            )
        dtype = result_data.get("dtype")
        if not isinstance(dtype, str) or "uint16" not in dtype.lower():
            raise ArtifactReportError(
                f"{result_path.relative_to(root).as_posix()} is not uint16"
            )
        if result.get("wall_metric") != WALL_METRIC:
            raise ArtifactReportError(
                f"{result_path.relative_to(root).as_posix()} has an incompatible "
                "wall metric"
            )
        observations.extend(
            _observation_from_profile(
                root,
                result_path,
                result_data,
                profile,
            )
            for profile in target_profiles
        )

    if not observations:
        raise ArtifactReportError(
            f"artifact tree contains no {PROFILE_KIND!r} APUv2 measurements"
        )
    return tuple(
        sorted(
            observations,
            key=lambda item: (
                item.shape,
                item.epilogue,
                item.schedule_json,
                str(item.source["provenance_fingerprint"]),
            ),
        )
    )


def _median_measurement(measurements: Sequence[_Measurement]) -> _Measurement:
    """Select one correlated observation at the deterministic median wall."""

    ordered = sorted(
        measurements,
        key=lambda item: (
            item.wall_us,
            item.h2d_us,
            item.host_task_us,
            item.d2h_us,
        ),
    )
    if not ordered:
        raise ArtifactReportError("measurement aggregation requires observations")
    return ordered[(len(ordered) - 1) // 2]


def _record_from_observations(
    observations: Sequence[_Observation],
) -> dict[str, object]:
    first = observations[0]
    schedule = json.loads(first.schedule_json)
    shape_manifest = _shape_manifest(first.shape)
    epilogue_manifest = {
        "alpha": first.epilogue[0],
        "beta": first.epilogue[1],
    }
    shape_fingerprint = _fingerprint("apu-g2-gemm-shape-v1", shape_manifest)
    semantic_problem_fingerprint = _fingerprint(
        "apu-g2-gemm-semantic-problem-v1",
        {"shape": shape_manifest, "epilogue": epilogue_manifest},
    )
    schedule_fingerprint = _fingerprint("apu-g2-gemm-schedule-v1", schedule)
    structural_identity_fingerprint = _fingerprint(
        "apu-g2-transport-structural-identity-v1",
        {
            "shape": shape_manifest,
            "epilogue": epilogue_manifest,
            "schedule": schedule,
        },
    )

    unique_measurements: dict[str, _Measurement] = {}
    provenance: dict[str, Mapping[str, object]] = {}
    for observation in observations:
        unique_measurements.setdefault(
            observation.measurement_fingerprint, observation.measurement
        )
        provenance_fingerprint = str(observation.source["provenance_fingerprint"])
        provenance.setdefault(provenance_fingerprint, observation.source)

    measurement_items = sorted(unique_measurements.items())
    measured = _median_measurement(
        tuple(measurement for _, measurement in measurement_items)
    )
    prediction = _prediction_for(first.shape, schedule, str(shape_manifest))
    predicted_wall = float(prediction["wall_us"])
    signed_relative_error = (predicted_wall - measured.wall_us) / measured.wall_us
    prediction_manifest = {
        "h2d_us": float(prediction["h2d_us"]),
        "host_task_us": float(prediction["host_task_us"]),
        "d2h_us": float(prediction["d2h_us"]),
        "wall_us": predicted_wall,
        "hardware_tasks": int(prediction["hardware_tasks"]),
        "weight_uploads": int(prediction["weight_uploads"]),
        "output_readbacks": int(prediction["output_readbacks"]),
        "weight_bytes": int(prediction["weight_bytes"]),
        "accumulator_bytes": int(prediction["accumulator_bytes"]),
        "index_bytes": int(prediction["index_bytes"]),
        "h2d_bytes": int(prediction["h2d_bytes"]),
        "cold_batch_bytes": int(prediction["cold_batch_bytes"]),
        "steady_batch_bytes": int(prediction["steady_batch_bytes"]),
        "h2d_calls": int(prediction["h2d_calls"]),
    }
    provenance_items = [
        dict(provenance[fingerprint]) for fingerprint in sorted(provenance)
    ]
    evidence_fingerprint = _fingerprint(
        "apu-g2-transport-evidence-v1",
        {
            "structural_identity_fingerprint": structural_identity_fingerprint,
            "measurement_fingerprints": [
                fingerprint for fingerprint, _ in measurement_items
            ],
            "provenance_fingerprints": sorted(provenance),
        },
    )
    return {
        "shape": shape_manifest,
        "shape_fingerprint": shape_fingerprint,
        "epilogue": epilogue_manifest,
        "semantic_problem_fingerprint": semantic_problem_fingerprint,
        "schedule": schedule,
        "schedule_fingerprint": schedule_fingerprint,
        "structural_identity_fingerprint": structural_identity_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "measurement_aggregation": "median_wall_correlated_observation",
        "observation_count": len(provenance_items),
        "unique_measurement_count": len(measurement_items),
        "unique_measurements": [
            {
                "measurement_fingerprint": fingerprint,
                **measurement.manifest(),
            }
            for fingerprint, measurement in measurement_items
        ],
        "measured": measured.manifest(),
        "predicted": prediction_manifest,
        "pointwise_error": {
            "signed_error_us": predicted_wall - measured.wall_us,
            "absolute_error_us": abs(predicted_wall - measured.wall_us),
            "signed_relative_error": signed_relative_error,
            "absolute_percentage_error": abs(signed_relative_error) * 100.0,
        },
        "provenance": provenance_items,
    }


def _deduplicate_observations(
    observations: Sequence[_Observation],
) -> tuple[dict[str, object], ...]:
    grouped: dict[
        tuple[tuple[int, int, int], tuple[int, int], str],
        list[_Observation],
    ] = defaultdict(list)
    for observation in observations:
        grouped[
            (
                observation.shape,
                observation.epilogue,
                observation.schedule_json,
            )
        ].append(observation)

    schedules_by_problem: dict[
        tuple[tuple[int, int, int], tuple[int, int]], set[str]
    ] = defaultdict(set)
    for shape, epilogue, schedule_json in grouped:
        schedules_by_problem[(shape, epilogue)].add(schedule_json)
    multiple = sorted(
        problem
        for problem, schedules in schedules_by_problem.items()
        if len(schedules) != 1
    )
    if multiple:
        rendered = ", ".join(
            str(
                {
                    "shape": _shape_manifest(shape),
                    "epilogue": {"alpha": epilogue[0], "beta": epilogue[1]},
                }
            )
            for shape, epilogue in multiple
        )
        raise ArtifactReportError(
            "within-problem candidate sets are mixed; expected one measured schedule "
            f"per shape and epilogue: {rendered}"
        )

    schedule_jsons = {schedule_json for _, _, schedule_json in grouped}
    if len(schedule_jsons) != 1:
        raise ArtifactReportError(
            "cross-shape report requires one common structural schedule"
        )
    return tuple(_record_from_observations(grouped[key]) for key in sorted(grouped))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _pointwise_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ArtifactReportError("pointwise metrics require at least one record")
    signed_relative = [
        float(
            _require_mapping(record["pointwise_error"], "pointwise_error")[
                "signed_relative_error"
            ]
        )
        for record in records
    ]
    absolute_percentage = [abs(value) * 100.0 for value in signed_relative]
    absolute_error = [
        float(
            _require_mapping(record["pointwise_error"], "pointwise_error")[
                "absolute_error_us"
            ]
        )
        for record in records
    ]
    count = len(records)
    return {
        "record_count": count,
        "signed_relative_bias": math.fsum(signed_relative) / count,
        "mean_absolute_percentage_error": (math.fsum(absolute_percentage) / count),
        "median_absolute_percentage_error": float(median(absolute_percentage)),
        "p90_absolute_percentage_error": _percentile(absolute_percentage, 0.9),
        "maximum_absolute_percentage_error": max(absolute_percentage),
        "mean_absolute_error_us": math.fsum(absolute_error) / count,
        "root_mean_squared_error_us": math.sqrt(
            math.fsum(value * value for value in absolute_error) / count
        ),
    }


def _solve_two_by_two(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    *,
    field: str,
) -> tuple[float, float]:
    first_x, first_y, first_value = first
    second_x, second_y, second_value = second
    determinant = first_x * second_y - first_y * second_x
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ArtifactReportError(f"{field} calibration equations are singular")
    left = (first_value * second_y - first_y * second_value) / determinant
    right = (first_x * second_value - first_value * second_x) / determinant
    if not all(math.isfinite(value) and value > 0.0 for value in (left, right)):
        raise ArtifactReportError(f"{field} calibration produced nonpositive terms")
    return left, right


def _fit_anchor_phase_parameters(
    training: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if len(training) != 2:
        raise ArtifactReportError("transport calibration requires exactly two anchors")
    multi_batch = next(
        (
            record
            for record in training
            if int(record["predicted"]["output_readbacks"]) > 1
        ),
        None,
    )
    single_batch = next(
        (
            record
            for record in training
            if int(record["predicted"]["output_readbacks"]) == 1
        ),
        None,
    )
    if multi_batch is None or single_batch is None:
        raise ArtifactReportError(
            "anchors must contain one single-batch and one multi-batch schedule"
        )

    fixed_h2d_call_us = float(APUG2_GEMM_WALL_FINGERPRINT_DATA["h2d_call_us"])

    def h2d_equation(record):
        predicted = record["predicted"]
        measured = record["measured"]
        calls = int(predicted["h2d_calls"])
        transfer_time = float(measured["h2d_us"]) - calls * fixed_h2d_call_us
        if transfer_time <= 0.0:
            raise ArtifactReportError("anchor H2D time does not exceed call overhead")
        return (
            float(predicted["cold_batch_bytes"]),
            float(predicted["steady_batch_bytes"]),
            transfer_time,
        )

    inverse_cold_bandwidth, inverse_steady_bandwidth = _solve_two_by_two(
        h2d_equation(single_batch),
        h2d_equation(multi_batch),
        field="H2D cold/steady batches",
    )

    def host_equation(record):
        shape = record["shape"]
        predicted = record["predicted"]
        measured = record["measured"]
        return (
            float(predicted["hardware_tasks"]),
            float(int(shape["N"]) * int(shape["K"])),
            float(measured["host_task_us"]),
        )

    task_call_us, scalar_mac_us = _solve_two_by_two(
        host_equation(multi_batch),
        host_equation(single_batch),
        field="host-task",
    )

    def d2h_equation(record):
        predicted = record["predicted"]
        measured = record["measured"]
        return (
            float(predicted["accumulator_bytes"]),
            float(predicted["output_readbacks"]),
            float(measured["d2h_us"]),
        )

    inverse_d2h_bandwidth, d2h_call_us = _solve_two_by_two(
        d2h_equation(multi_batch),
        d2h_equation(single_batch),
        field="D2H",
    )
    return {
        "cold_batch_h2d_bytes_per_us": 1.0 / inverse_cold_bandwidth,
        "steady_batch_h2d_bytes_per_us": 1.0 / inverse_steady_bandwidth,
        "h2d_call_us": fixed_h2d_call_us,
        "task_call_us": task_call_us,
        "scalar_mac_us": scalar_mac_us,
        "d2h_bytes_per_us": 1.0 / inverse_d2h_bandwidth,
        "d2h_call_us": d2h_call_us,
    }


def _fit_verification(
    training: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fitted = _fit_anchor_phase_parameters(training)
    current = {key: float(APUG2_GEMM_WALL_FINGERPRINT_DATA[key]) for key in fitted}
    mismatches = {
        key: {"fitted": fitted[key], "current": current[key]}
        for key in fitted
        if not math.isclose(fitted[key], current[key], rel_tol=1e-12, abs_tol=1e-9)
    }
    if mismatches:
        raise ArtifactReportError(
            "current transport constants do not reproduce the declared anchor "
            f"phase fit: {mismatches}"
        )
    return {
        "status": "matches_current_model",
        "solver": (
            "two_anchor_structural_index_cold_steady_phase_fit_with_fixed_h2d_call_overhead"
        ),
        "fitted_parameters": fitted,
    }


def _model_manifest() -> dict[str, object]:
    try:
        parameters = {
            key: APUG2_GEMM_WALL_FINGERPRINT_DATA[key] for key in _MODEL_PARAMETER_KEYS
        }
    except KeyError as error:
        raise ArtifactReportError(
            f"current estimator lacks fingerprint parameter {error.args[0]!r}"
        ) from error
    identity = {
        "qualified_name": ("allo.pim.costs.apu_g2.estimate_apu_g2_u16_gemm_wall_us"),
        "model_revision": APUG2_GEMM_WALL_MODEL_REVISION,
        "calibration_sample_revision": APUG2_GEMM_WALL_SAMPLE_REVISION,
        "fit_policy": APUG2_GEMM_WALL_FINGERPRINT_DATA["fit_policy"],
        "anchor_phases_us": APUG2_GEMM_WALL_ANCHOR_PHASES_US,
        "parameters": parameters,
    }
    return {
        **identity,
        "model_fingerprint": _fingerprint("apu-g2-transport-estimator-v1", identity),
        "identity_policy": (
            "structural estimator revision and numeric parameters only; "
            "benchmark names, workload names, and artifact paths are excluded"
        ),
    }


def build_transport_calibration_report(
    artifacts_root: str | Path,
) -> dict[str, object]:
    """Build a deterministic, structurally partitioned calibration report."""

    observations = load_transport_observations(artifacts_root)
    records = _deduplicate_observations(observations)
    by_case = {
        (
            *(int(record["shape"][key]) for key in ("M", "K", "N")),
            int(record["epilogue"]["alpha"]),
            int(record["epilogue"]["beta"]),
        ): record
        for record in records
    }
    missing = [case for case in TRAINING_ANCHOR_CASES if case not in by_case]
    if missing:
        rendered = ", ".join(
            str(
                {
                    "shape": _shape_manifest(case[:3]),
                    "epilogue": {"alpha": case[3], "beta": case[4]},
                }
            )
            for case in missing
        )
        raise ArtifactReportError(f"missing required training anchors: {rendered}")

    training = [
        dict(by_case[case], partition="training") for case in TRAINING_ANCHOR_CASES
    ]
    heldout = [
        dict(record, partition="heldout")
        for record in records
        if (
            *(int(record["shape"][key]) for key in ("M", "K", "N")),
            int(record["epilogue"]["alpha"]),
            int(record["epilogue"]["beta"]),
        )
        not in TRAINING_ANCHOR_CASES
    ]
    if not heldout:
        raise ArtifactReportError(
            "cross-shape validation requires at least one unique heldout shape"
        )

    model = {
        **_model_manifest(),
        "fit_verification": _fit_verification(training),
    }
    partition_identity = {
        "strategy": "fixed_structural_training_anchors",
        "training_structural_identity_fingerprints": [
            record["structural_identity_fingerprint"] for record in training
        ],
        "heldout_structural_identity_fingerprints": [
            record["structural_identity_fingerprint"] for record in heldout
        ],
    }
    source_result_paths = {
        source["result_path"] for record in records for source in record["provenance"]
    }
    training_shapes = {
        tuple(int(record["shape"][key]) for key in ("M", "K", "N"))
        for record in training
    }
    heldout_shapes = {
        tuple(int(record["shape"][key]) for key in ("M", "K", "N"))
        for record in heldout
    }
    all_shapes = training_shapes | heldout_shapes
    body: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "apu_g2_transport_calibration",
        "objective": {
            "metric": "host_wall_latency",
            "unit": "microseconds",
            "direction": "minimize",
            "scope": "h2d + host_task + d2h",
        },
        "model": model,
        "artifact_summary": {
            "source_result_record_count": len(source_result_paths),
            "profile_observation_count": len(observations),
            "unique_structural_record_count": len(records),
            "training_record_count": len(training),
            "heldout_record_count": len(heldout),
        },
        "partition": {
            **partition_identity,
            "partition_fingerprint": _fingerprint(
                "apu-g2-transport-partition-v1", partition_identity
            ),
            "training_anchor_shapes": [
                _shape_manifest(shape) for shape in TRAINING_ANCHORS
            ],
            "training_anchor_cases": [
                {
                    "shape": _shape_manifest(case[:3]),
                    "epilogue": {"alpha": case[3], "beta": case[4]},
                }
                for case in TRAINING_ANCHOR_CASES
            ],
        },
        "methodology": {
            "deduplication_key": (
                "structural shape plus retained alpha/beta epilogue plus complete "
                "physical schedule identity"
            ),
            "duplicate_measurement_policy": "exact normalized measurements count once",
            "measurement_aggregation": "median_wall_correlated_observation",
            "estimator_refit_performed": False,
            "declared_anchor_fit_reproduced": True,
            "validation_scope": "cross_shape_point_prediction",
            "limitations": [
                "The artifacts measure one schedule candidate per structural shape and epilogue.",
                "Cross-shape latency error does not establish within-shape candidate ranking quality.",
                "No schedule-ranking metrics are computed from singleton candidate sets.",
            ],
        },
        "training_records": training,
        "heldout_records": heldout,
        "cross_shape_prediction": {
            "status": "available",
            "training_anchor_error": _pointwise_metrics(training),
            "heldout_error": _pointwise_metrics(heldout),
            "coverage": {
                "eligible_heldout_semantic_problem_count": len(heldout),
                "predicted_heldout_semantic_problem_count": len(heldout),
                "eligible_heldout_shape_count": len(heldout_shapes),
                "predicted_heldout_shape_count": len(heldout_shapes),
                "prediction_coverage": 1.0,
                "heldout_shapes_unseen_in_training": len(
                    heldout_shapes - training_shapes
                ),
                "cross_shape_extrapolation_coverage": (
                    len(heldout_shapes - training_shapes) / len(heldout_shapes)
                ),
                "definition": (
                    "fraction of unique heldout structural shapes receiving a "
                    "finite prediction from the current estimator"
                ),
            },
        },
        "within_shape_candidate_ranking": {
            "status": "unavailable",
            "reason": (
                "each structural shape and epilogue has exactly one measured "
                "schedule candidate"
            ),
            "shape_count": len(all_shapes),
            "semantic_problem_count": len(records),
            "observed_candidate_count_per_semantic_problem": 1,
            "observed_candidate_count_per_shape": 1,
            "metrics": None,
        },
    }
    return {
        **body,
        "report_fingerprint": _fingerprint(
            "apu-g2-transport-calibration-report-v1", body
        ),
    }


def report_to_json(report: Mapping[str, object]) -> str:
    """Serialize a report deterministically for CI artifacts."""

    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic APUg2 transport-calibration artifact report."
        )
    )
    parser.add_argument(
        "artifacts_root",
        nargs="?",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help=f"artifact tree root (default: {DEFAULT_ARTIFACTS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rendered = report_to_json(
            build_transport_calibration_report(args.artifacts_root)
        )
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (ArtifactReportError, OSError) as error:
        print(f"FAIL APUg2 transport calibration: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
