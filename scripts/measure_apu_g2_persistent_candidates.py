#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure structural APUg2 persistent-GEMM schedule candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from allo.pim import apu_g2_u16_gemm_runtime as _runtime
from allo.pim.apu_g2_contraction import APUG2AffineAccessMap
from allo.pim.apu_g2_u16_gemm_runtime import (
    APUG2_U16_GEMM_BMAX,
    APUG2_U16_GEMM_CHUNK,
    freeze_apu_g2_u16_gemm_runtime_artifact,
    run_apu_g2_u16_gemm,
)
from allo.pim.apu_g2_vector_program import (
    APUG2PersistentGemmSchedule,
    _apu_g2_persistent_semantic_fingerprint,
    _materialize_apu_g2_persistent_gemm_schedule,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.schedule_search import InfeasibleSchedule
from allo.pim.targets import build_apu_g2_target


SCHEMA_VERSION = 3
RECORD_KIND = "apu_g2_persistent_gemm_candidate_measurements"
MIN_MEASURED_SAMPLES = 7
WARMUP_RUNS = 1
DEFAULT_BATCH_COLUMNS = tuple(range(1, APUG2_U16_GEMM_BMAX + 1))
_PLATFORM_SCHEMA = "tenon-promotion-platform-v1"
_PLATFORM_TARGET = "apu_v2"
_PLATFORM_HARDWARE_FAMILY = "gemini-ii"

_SCHEDULE_KIND = "column_batched_u16_gemm"
_HOST_TIMING_FIELDS = ("h2d", "host_task", "d2h", "end_to_end")
_MODEL_PREDICTION_FIELDS = (
    "h2d_us",
    "host_task_us",
    "d2h_us",
    "wall_us",
    "hardware_tasks",
    "weight_uploads",
    "output_readbacks",
)
_OBJECTIVE = {
    "metric": "host_wall_us",
    "unit": "microseconds",
    "direction": "minimize",
    "definition": "h2d_us + host_task_us + d2h_us from one invocation",
}


class CandidateMeasurementError(ValueError):
    """Raised when a campaign cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class _StructuralGemmPlan:
    dot_axes: tuple[tuple[str, int], ...]
    batch_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    reduction_axis: tuple[str, int]
    lhs: APUG2AffineAccessMap
    rhs: APUG2AffineAccessMap
    initial_output: APUG2AffineAccessMap
    output: APUG2AffineAccessMap
    epilogue: tuple[int, int]
    batch_local_accumulator: bool = False


@dataclass(frozen=True)
class _PreparedCandidate:
    materialized: Any
    schedule_manifest: Mapping[str, object]
    model_prediction: Mapping[str, object]
    identity: Mapping[str, str]


@dataclass(frozen=True)
class _PreparedCampaign:
    bound_cost: Any
    provenance: Mapping[str, object]
    candidates: tuple[_PreparedCandidate, ...]


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
        raise CandidateMeasurementError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _source_fingerprint(source_hashes: Mapping[str, str]) -> str:
    ordered = tuple(sorted(source_hashes.items()))
    return hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode()
    ).hexdigest()


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateMeasurementError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], field: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CandidateMeasurementError(
            f"{field} has malformed keys: missing={missing}, extra={extra}"
        )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CandidateMeasurementError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise CandidateMeasurementError(f"{field} must be a positive integer")
    return result


def _require_exact_int(value: object, expected: int, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) != expected
    ):
        raise CandidateMeasurementError(f"{field} must equal integer {expected}")


def _u16_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CandidateMeasurementError(f"{field} must be a uint16 integer")
    result = int(value)
    if not 0 <= result <= 0xFFFF:
        raise CandidateMeasurementError(f"{field} must be a uint16 integer")
    return result


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CandidateMeasurementError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CandidateMeasurementError(f"{field} must be a positive finite number")
    return result


def _require_digest(value: object, field: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidateMeasurementError(
            f"{field} must be a {length}-character lowercase hex digest"
        )
    return value


def _normalize_platform_contract(value: object, field: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CandidateMeasurementError(f"{field} is not a platform contract")
    parts = tuple(value)
    if len(parts) != 4:
        raise CandidateMeasurementError(f"{field} is not a platform contract")
    schema, target, hardware_family, fingerprint = parts
    if (
        schema != _PLATFORM_SCHEMA
        or target != _PLATFORM_TARGET
        or hardware_family != _PLATFORM_HARDWARE_FAMILY
    ):
        raise CandidateMeasurementError(f"{field} targets the wrong platform")
    return {
        "schema": schema,
        "target": target,
        "hardware_family": hardware_family,
        "fingerprint": _require_digest(fingerprint, f"{field}.fingerprint"),
    }


def _promotion_eligibility(
    platform_contract: Mapping[str, str] | None,
) -> dict[str, object]:
    reasons = ["built_binary_and_build_flag_fingerprint_unavailable"]
    if platform_contract is None:
        reasons.append("hardware_sdk_firmware_platform_fingerprint_unavailable")
    return {"eligible": False, "reason_codes": reasons}


_PROMOTION_ELIGIBILITY = _promotion_eligibility(None)


def _normalize_source_hashes(value: object, field: str) -> dict[str, str]:
    source_hashes = _require_mapping(value, field)
    if not source_hashes:
        raise CandidateMeasurementError(f"{field} must not be empty")
    normalized = {}
    for path, digest in source_hashes.items():
        if not isinstance(path, str) or not path:
            raise CandidateMeasurementError(f"{field} paths must be nonempty strings")
        normalized[path] = _require_digest(digest, f"{field}[{path!r}]")
    return dict(sorted(normalized.items()))


def current_runtime_provenance() -> dict[str, object]:
    """Return source provenance without touching the hardware stack."""

    try:
        artifact = freeze_apu_g2_u16_gemm_runtime_artifact()
        artifact.assert_current(run_apu_g2_u16_gemm)
    except (InfeasibleSchedule, RuntimeError) as error:
        raise CandidateMeasurementError(str(error)) from error
    source_fingerprint = artifact.source_fingerprint
    source_hashes = dict(artifact.source_hashes)
    if _source_fingerprint(source_hashes) != source_fingerprint:
        raise CandidateMeasurementError(
            "persistent runtime returned inconsistent source provenance"
        )
    runtime_path = Path(_runtime.__file__).resolve()
    return {
        "source_fingerprint": source_fingerprint,
        "source_sha256": dict(sorted(source_hashes.items())),
        "runtime_driver_sha256": _sha256_file(runtime_path),
        "platform_contract": _normalize_platform_contract(
            artifact.current_platform_fingerprint(), "runtime platform_contract"
        ),
    }


def _structural_gemm_plan(
    rows: int, reduction: int, columns: int, alpha: int, beta: int
) -> _StructuralGemmPlan:
    row_axis = "output_axis_0"
    column_axis = "output_axis_1"
    reduction_axis = "reduction_axis_0"
    domain = (row_axis, column_axis, reduction_axis)
    output_axes = (row_axis, column_axis)
    lhs = APUG2AffineAccessMap(
        "lhs_storage",
        "ui16",
        (rows, reduction),
        "read",
        domain,
        (row_axis, reduction_axis),
    )
    rhs = APUG2AffineAccessMap(
        "rhs_storage",
        "ui16",
        (reduction, columns),
        "read",
        domain,
        (reduction_axis, column_axis),
    )
    initial_output = APUG2AffineAccessMap(
        "output_storage",
        "ui16",
        (rows, columns),
        "read",
        domain,
        output_axes,
    )
    output = APUG2AffineAccessMap(
        "output_storage",
        "ui16",
        (rows, columns),
        "write",
        domain,
        output_axes,
    )
    return _StructuralGemmPlan(
        dot_axes=((row_axis, rows), (column_axis, columns)),
        batch_axes=(),
        output_axes=output_axes,
        reduction_axis=(reduction_axis, reduction),
        lhs=lhs,
        rhs=rhs,
        initial_output=initial_output,
        output=output,
        epilogue=(alpha, beta),
    )


def _normalize_batch_columns(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_BATCH_COLUMNS
    if isinstance(values, (str, bytes)):
        raise CandidateMeasurementError("batch_columns must be an integer sequence")
    normalized = []
    seen = set()
    for index, value in enumerate(values):
        batch = _positive_int(value, f"batch_columns[{index}]")
        if batch > APUG2_U16_GEMM_BMAX:
            raise CandidateMeasurementError(
                f"batch_columns[{index}] exceeds runtime maximum "
                f"{APUG2_U16_GEMM_BMAX}"
            )
        if batch in seen:
            raise CandidateMeasurementError(
                f"duplicate persistent-GEMM candidate batch_columns={batch}"
            )
        seen.add(batch)
        normalized.append(batch)
    if not normalized:
        raise CandidateMeasurementError(
            "at least one batch-column candidate is required"
        )
    return tuple(sorted(normalized))


def _schedule_manifest(batch_columns: int) -> dict[str, object]:
    return {
        "kind": _SCHEDULE_KIND,
        "batch_columns": batch_columns,
        "reduction_tile": APUG2_U16_GEMM_CHUNK,
        "resident_accumulator": True,
        "contiguous_readback": True,
    }


def _normalize_model_prediction(value: object) -> dict[str, object]:
    prediction = _require_mapping(value, "model prediction")
    normalized = {
        field: _positive_float(prediction.get(field), f"model prediction.{field}")
        for field in ("h2d_us", "host_task_us", "d2h_us", "wall_us")
    }
    normalized.update(
        {
            field: _positive_int(prediction.get(field), f"model prediction.{field}")
            for field in (
                "hardware_tasks",
                "weight_uploads",
                "output_readbacks",
            )
        }
    )
    expected_wall = (
        normalized["h2d_us"] + normalized["host_task_us"] + normalized["d2h_us"]
    )
    if normalized["wall_us"] != expected_wall:
        raise CandidateMeasurementError(
            "model prediction wall_us does not equal its correlated host phases"
        )
    return normalized


def _candidate_identity(
    materialized: object,
    schedule_manifest: Mapping[str, object],
    model_fingerprint: str,
    model_prediction: Mapping[str, object],
) -> dict[str, str]:
    identity = {
        "schedule_fingerprint": _fingerprint(
            "apu-g2-persistent-gemm-schedule-v1", schedule_manifest
        ),
        "materialization_fingerprint": (
            materialized.promotion_materialization_fingerprint
        ),
        "semantic_fingerprint": materialized.semantic_fingerprint,
        "source_fingerprint": materialized.source_fingerprint,
        "model_fingerprint": model_fingerprint,
        "prediction_fingerprint": _fingerprint(
            "apu-g2-persistent-gemm-model-prediction-v1",
            model_prediction,
        ),
    }
    return {
        "candidate_fingerprint": _fingerprint(
            "apu-g2-persistent-gemm-measurement-candidate-v1", identity
        ),
        **identity,
    }


def _prepare_campaign(
    rows: int,
    reduction: int,
    columns: int,
    alpha: int,
    beta: int,
    batch_columns: tuple[int, ...],
) -> _PreparedCampaign:
    target = build_apu_g2_target()
    bound_cost = apu_g2_cost.bind(target)
    runtime_provenance = current_runtime_provenance()
    plan = _structural_gemm_plan(rows, reduction, columns, alpha, beta)
    prepared = []
    materialization_fingerprints = set()
    candidate_fingerprints = set()
    semantic_fingerprint = None
    for batch in batch_columns:
        schedule = APUG2PersistentGemmSchedule(
            batch_columns=batch,
            reduction_tile=APUG2_U16_GEMM_CHUNK,
        )
        try:
            materialized = _materialize_apu_g2_persistent_gemm_schedule(
                plan, target, schedule
            )
        except InfeasibleSchedule as error:
            raise CandidateMeasurementError(
                f"cannot materialize batch_columns={batch}: {error}"
            ) from error
        source_hashes = dict(materialized.source_hashes)
        if (
            materialized.source_fingerprint != runtime_provenance["source_fingerprint"]
            or source_hashes != runtime_provenance["source_sha256"]
        ):
            raise CandidateMeasurementError(
                "mixed persistent-runtime source provenance while materializing "
                f"batch_columns={batch}"
            )
        materialized_platform = _normalize_platform_contract(
            materialized.promotion_platform_fingerprint,
            f"batch_columns={batch} platform_contract",
        )
        if materialized_platform != runtime_provenance["platform_contract"]:
            raise CandidateMeasurementError(
                "mixed platform provenance while materializing "
                f"batch_columns={batch}"
            )
        if semantic_fingerprint is None:
            semantic_fingerprint = materialized.semantic_fingerprint
        elif materialized.semantic_fingerprint != semantic_fingerprint:
            raise CandidateMeasurementError(
                "mixed semantic provenance across persistent-GEMM candidates"
            )
        manifest = _schedule_manifest(batch)
        try:
            scored = bound_cost.score_materialization(materialized)
        except RuntimeError as error:
            raise CandidateMeasurementError(
                "persistent-GEMM model provenance changed during preparation"
            ) from error
        prediction = _normalize_model_prediction(scored)
        identity = _candidate_identity(
            materialized,
            manifest,
            bound_cost.fingerprint,
            prediction,
        )
        if identity["materialization_fingerprint"] in materialization_fingerprints:
            raise CandidateMeasurementError(
                "duplicate persistent-GEMM materialization fingerprint"
            )
        if identity["candidate_fingerprint"] in candidate_fingerprints:
            raise CandidateMeasurementError(
                "duplicate persistent-GEMM candidate fingerprint"
            )
        materialization_fingerprints.add(identity["materialization_fingerprint"])
        candidate_fingerprints.add(identity["candidate_fingerprint"])
        prepared.append(
            _PreparedCandidate(materialized, manifest, prediction, identity)
        )
    provenance = {
        "target": "apu_v2",
        **runtime_provenance,
        "model_fingerprint": bound_cost.fingerprint,
    }
    return _PreparedCampaign(bound_cost, provenance, tuple(prepared))


def _deterministic_inputs(
    rows: int, reduction: int, columns: int, alpha: int, beta: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    try:
        lhs = (
            ((np.arange(rows * reduction, dtype=np.uint64) * 17 + 3) & 0xFFFF)
            .astype(np.uint16)
            .reshape(rows, reduction)
        )
        rhs = (
            ((np.arange(reduction * columns, dtype=np.uint64) * 29 + 5) & 0xFFFF)
            .astype(np.uint16)
            .reshape(reduction, columns)
        )
        accumulator = (
            ((np.arange(rows * columns, dtype=np.uint64) * 43 + 7) & 0xFFFF)
            .astype(np.uint16)
            .reshape(rows, columns)
        )
        expected = (
            np.uint64(alpha) * (lhs.astype(np.uint64) @ rhs.astype(np.uint64))
            + np.uint64(beta) * accumulator.astype(np.uint64)
        ) & np.uint64(0xFFFF)
        expected = np.ascontiguousarray(expected.astype(np.uint16))
    except (MemoryError, OverflowError, ValueError) as error:
        raise CandidateMeasurementError(
            f"cannot construct deterministic inputs for M={rows}, K={reduction}, "
            f"N={columns}: {error}"
        ) from error
    manifest = {
        "generator": "deterministic_linear_u16_v1",
        "lhs_sha256": _sha256_array(lhs),
        "rhs_sha256": _sha256_array(rhs),
        "accumulator_sha256": _sha256_array(accumulator),
        "expected_sha256": _sha256_array(expected),
    }
    return lhs, rhs, accumulator, expected, manifest


def _validate_runtime_result(
    result: object,
    *,
    candidate: _PreparedCandidate,
    expected: np.ndarray,
    provenance: Mapping[str, object],
    sample_label: str,
) -> dict[str, object]:
    if getattr(result, "backend", None) != "apu_v2":
        raise CandidateMeasurementError(
            f"{sample_label} has malformed backend provenance"
        )
    device_ticks = _positive_int(
        getattr(result, "cycles", None), f"{sample_label}.device_pipeline_ticks"
    )
    extra = _require_mapping(getattr(result, "extra", None), f"{sample_label}.extra")
    outputs = _require_mapping(extra.get("outputs"), f"{sample_label}.extra.outputs")
    observed = outputs.get("out")
    if not isinstance(observed, np.ndarray):
        raise CandidateMeasurementError(f"{sample_label} output must be a NumPy array")
    if observed.dtype != np.dtype(np.uint16) or observed.shape != expected.shape:
        raise CandidateMeasurementError(
            f"{sample_label} output has malformed dtype or shape"
        )
    if not np.array_equal(observed, expected):
        index = tuple(int(i) for i in np.argwhere(observed != expected)[0])
        raise CandidateMeasurementError(
            f"{sample_label} failed uint16 correctness at {index}: "
            f"observed={int(observed[index])}, expected={int(expected[index])}"
        )

    schedule = _require_mapping(extra.get("schedule"), f"{sample_label}.extra.schedule")
    for field, expected_value in candidate.schedule_manifest.items():
        if schedule.get(field) != expected_value or type(
            schedule.get(field)
        ) is not type(expected_value):
            raise CandidateMeasurementError(
                f"{sample_label} has mixed schedule provenance for {field}"
            )

    expected_tasks = candidate.model_prediction["hardware_tasks"]
    hardware_tasks = _positive_int(
        extra.get("hardware_tasks"), f"{sample_label}.extra.hardware_tasks"
    )
    weight_uploads = _positive_int(
        extra.get("weight_uploads"), f"{sample_label}.extra.weight_uploads"
    )
    if (
        hardware_tasks != expected_tasks
        or weight_uploads != candidate.model_prediction["weight_uploads"]
    ):
        raise CandidateMeasurementError(
            f"{sample_label} has counters inconsistent with its materialization"
        )

    host_timings = _require_mapping(
        extra.get("host_timings_us"), f"{sample_label}.extra.host_timings_us"
    )
    host = {
        f"{field}_us": _positive_float(
            host_timings.get(field),
            f"{sample_label}.extra.host_timings_us.{field}",
        )
        for field in _HOST_TIMING_FIELDS
    }
    host["wall_us"] = host["h2d_us"] + host["host_task_us"] + host["d2h_us"]

    project = _require_mapping(extra.get("project"), f"{sample_label}.extra.project")
    source_hashes = _normalize_source_hashes(
        project.get("source_sha256"), f"{sample_label}.extra.project.source_sha256"
    )
    if (
        source_hashes != provenance["source_sha256"]
        or _source_fingerprint(source_hashes) != provenance["source_fingerprint"]
    ):
        raise CandidateMeasurementError(
            f"{sample_label} has mixed persistent-runtime source provenance"
        )
    runtime_driver = _require_digest(
        project.get("runtime_sha256"),
        f"{sample_label}.extra.project.runtime_sha256",
    )
    if runtime_driver != provenance["runtime_driver_sha256"]:
        raise CandidateMeasurementError(
            f"{sample_label} has mixed runtime-driver provenance"
        )
    platform_contract = _normalize_platform_contract(
        project.get("promotion_platform_fingerprint"),
        f"{sample_label}.extra.project.promotion_platform_fingerprint",
    )
    if platform_contract != provenance["platform_contract"]:
        raise CandidateMeasurementError(f"{sample_label} has mixed platform provenance")
    return {
        "device_pipeline_ticks": device_ticks,
        "hardware_tasks": hardware_tasks,
        "weight_uploads": weight_uploads,
        "host_us": host,
    }


def _invoke_runner(
    runner: Callable[..., object],
    lhs: np.ndarray,
    rhs: np.ndarray,
    accumulator: np.ndarray,
    alpha: int,
    beta: int,
    candidate: _PreparedCandidate,
    expected: np.ndarray,
    provenance: Mapping[str, object],
    sample_label: str,
) -> dict[str, object]:
    try:
        result = runner(
            lhs,
            rhs,
            accumulator,
            alpha=alpha,
            beta=beta,
            batch_columns=candidate.materialized.schedule.batch_columns,
        )
    except Exception as error:
        raise CandidateMeasurementError(
            f"{sample_label} runner failed: {error}"
        ) from error
    return _validate_runtime_result(
        result,
        candidate=candidate,
        expected=expected,
        provenance=provenance,
        sample_label=sample_label,
    )


def measure_apu_g2_persistent_candidates(
    rows: int,
    reduction: int,
    columns: int,
    *,
    alpha: int = 1,
    beta: int = 1,
    batch_columns: Sequence[int] | None = None,
    samples: int = MIN_MEASURED_SAMPLES,
    runner: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Measure a structural shape with one discarded warmup per candidate."""

    rows = _positive_int(rows, "M")
    reduction = _positive_int(reduction, "K")
    columns = _positive_int(columns, "N")
    alpha = _u16_int(alpha, "alpha")
    beta = _u16_int(beta, "beta")
    samples = _positive_int(samples, "samples")
    if samples < MIN_MEASURED_SAMPLES:
        raise CandidateMeasurementError(
            f"samples must be at least {MIN_MEASURED_SAMPLES}"
        )
    batches = _normalize_batch_columns(batch_columns)
    prepared = _prepare_campaign(rows, reduction, columns, alpha, beta, batches)
    lhs, rhs, accumulator, expected, input_manifest = _deterministic_inputs(
        rows, reduction, columns, alpha, beta
    )
    input_hashes = dict(input_manifest)
    execute = run_apu_g2_u16_gemm if runner is None else runner
    measured_candidates = []
    for candidate_index, candidate in enumerate(prepared.candidates):
        batch = candidate.materialized.schedule.batch_columns
        _invoke_runner(
            execute,
            lhs,
            rhs,
            accumulator,
            alpha,
            beta,
            candidate,
            expected,
            prepared.provenance,
            f"batch_columns={batch} warmup",
        )
        retained_samples = []
        for sample_index in range(samples):
            sample = _invoke_runner(
                execute,
                lhs,
                rhs,
                accumulator,
                alpha,
                beta,
                candidate,
                expected,
                prepared.provenance,
                f"batch_columns={batch} sample={sample_index}",
            )
            retained_samples.append({"sample_index": sample_index, **sample})
        measured_candidates.append(
            {
                "candidate_index": candidate_index,
                "identity": dict(candidate.identity),
                "schedule": dict(candidate.schedule_manifest),
                "materialization": {
                    "M": candidate.materialized.rows,
                    "K": candidate.materialized.reduction,
                    "N": candidate.materialized.columns,
                    "alpha": candidate.materialized.epilogue[0],
                    "beta": candidate.materialized.epilogue[1],
                },
                "model_prediction": dict(candidate.model_prediction),
                "samples": retained_samples,
            }
        )

    final_input_manifest = {
        **input_manifest,
        "lhs_sha256": _sha256_array(lhs),
        "rhs_sha256": _sha256_array(rhs),
        "accumulator_sha256": _sha256_array(accumulator),
        "expected_sha256": _sha256_array(expected),
    }
    if final_input_manifest != input_hashes:
        raise CandidateMeasurementError("runner mutated deterministic campaign inputs")
    if current_runtime_provenance() != {
        field: prepared.provenance[field]
        for field in (
            "source_fingerprint",
            "source_sha256",
            "runtime_driver_sha256",
            "platform_contract",
        )
    }:
        raise CandidateMeasurementError(
            "persistent-runtime source provenance changed during the campaign"
        )
    for candidate in prepared.candidates:
        try:
            rescored = prepared.bound_cost.score_materialization(candidate.materialized)
        except RuntimeError as error:
            raise CandidateMeasurementError(
                "persistent-GEMM model provenance changed during the campaign"
            ) from error
        refreshed = _normalize_model_prediction(rescored)
        if refreshed != candidate.model_prediction:
            raise CandidateMeasurementError(
                "persistent-GEMM model changed during the campaign"
            )

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": RECORD_KIND,
        "objective": dict(_OBJECTIVE),
        "structural_case": {
            "M": rows,
            "K": reduction,
            "N": columns,
            "alpha": alpha,
            "beta": beta,
        },
        "candidate_domain": {
            "full_batch_columns": list(DEFAULT_BATCH_COLUMNS),
            "selected_batch_columns": list(batches),
            "runtime_min": 1,
            "runtime_max": APUG2_U16_GEMM_BMAX,
            "reduction_tile": APUG2_U16_GEMM_CHUNK,
            "full_candidate_count": len(DEFAULT_BATCH_COLUMNS),
            "selected_candidate_count": len(batches),
            "candidate_domain_complete": batches == DEFAULT_BATCH_COLUMNS,
        },
        "measurement_protocol": {
            "warmup_runs_per_candidate": WARMUP_RUNS,
            "warmup_samples_retained": 0,
            "measured_samples_per_candidate": samples,
            "correctness": "bit_exact_numpy_uint16_mod_2^16_each_invocation",
            "sample_correlation": (
                "one sample object contains device and host timings from one invocation"
            ),
        },
        "input_data": input_manifest,
        "provenance": dict(prepared.provenance),
        "candidates": measured_candidates,
        "rejections": [],
        "promotion_eligibility": _promotion_eligibility(
            prepared.provenance["platform_contract"]
        ),
    }
    report = {
        **body,
        "measurement_fingerprint": _fingerprint(
            "apu-g2-persistent-gemm-candidate-measurements-v1", body
        ),
    }
    validate_measurement_report(report)
    return report


def _recompute_materialization_fingerprint(
    candidate: Mapping[str, object],
) -> str:
    schedule = _require_mapping(candidate["schedule"], "candidate.schedule")
    materialization = _require_mapping(
        candidate["materialization"], "candidate.materialization"
    )
    identity = _require_mapping(candidate["identity"], "candidate.identity")
    payload = {
        "kind": "apu-g2-persistent-gemm-materialization-v1",
        "batch_columns": schedule["batch_columns"],
        "reduction_tile": schedule["reduction_tile"],
        "rows": materialization["M"],
        "reduction": materialization["K"],
        "columns": materialization["N"],
        "epilogue": [materialization["alpha"], materialization["beta"]],
        "semantic_fingerprint": identity["semantic_fingerprint"],
        "source_fingerprint": identity["source_fingerprint"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_measurement_report(report: object) -> None:
    """Validate the exact deterministic output schema and all identities."""

    root = _require_mapping(report, "report")
    root_keys = {
        "schema_version",
        "record_kind",
        "objective",
        "structural_case",
        "candidate_domain",
        "measurement_protocol",
        "input_data",
        "provenance",
        "candidates",
        "rejections",
        "promotion_eligibility",
        "measurement_fingerprint",
    }
    _require_exact_keys(root, root_keys, "report")
    _require_exact_int(root["schema_version"], SCHEMA_VERSION, "report.schema_version")
    if root["record_kind"] != RECORD_KIND:
        raise CandidateMeasurementError("report schema version or kind is unsupported")
    objective = _require_mapping(root["objective"], "report.objective")
    if dict(objective) != _OBJECTIVE:
        raise CandidateMeasurementError("report.objective is malformed")

    structural = _require_mapping(root["structural_case"], "report.structural_case")
    _require_exact_keys(
        structural, {"M", "K", "N", "alpha", "beta"}, "report.structural_case"
    )
    rows = _positive_int(structural["M"], "report.structural_case.M")
    reduction = _positive_int(structural["K"], "report.structural_case.K")
    columns = _positive_int(structural["N"], "report.structural_case.N")
    alpha = _u16_int(structural["alpha"], "report.structural_case.alpha")
    beta = _u16_int(structural["beta"], "report.structural_case.beta")

    domain = _require_mapping(root["candidate_domain"], "report.candidate_domain")
    _require_exact_keys(
        domain,
        {
            "full_batch_columns",
            "selected_batch_columns",
            "runtime_min",
            "runtime_max",
            "reduction_tile",
            "full_candidate_count",
            "selected_candidate_count",
            "candidate_domain_complete",
        },
        "report.candidate_domain",
    )
    _require_exact_int(domain["runtime_min"], 1, "candidate_domain.runtime_min")
    _require_exact_int(
        domain["runtime_max"],
        APUG2_U16_GEMM_BMAX,
        "candidate_domain.runtime_max",
    )
    _require_exact_int(
        domain["reduction_tile"],
        APUG2_U16_GEMM_CHUNK,
        "candidate_domain.reduction_tile",
    )
    full_batches_raw = domain["full_batch_columns"]
    if full_batches_raw != list(DEFAULT_BATCH_COLUMNS):
        raise CandidateMeasurementError(
            "candidate_domain.full_batch_columns is incomplete"
        )
    _require_exact_int(
        domain["full_candidate_count"],
        len(DEFAULT_BATCH_COLUMNS),
        "candidate_domain.full_candidate_count",
    )
    batches_raw = domain["selected_batch_columns"]
    if not isinstance(batches_raw, list):
        raise CandidateMeasurementError(
            "candidate_domain.selected_batch_columns must be an array"
        )
    batches = _normalize_batch_columns(batches_raw)
    if list(batches) != batches_raw:
        raise CandidateMeasurementError(
            "candidate_domain.selected_batch_columns must be strictly ascending"
        )
    _require_exact_int(
        domain["selected_candidate_count"],
        len(batches),
        "candidate_domain.selected_candidate_count",
    )
    if type(domain["candidate_domain_complete"]) is not bool or domain[
        "candidate_domain_complete"
    ] != (batches == DEFAULT_BATCH_COLUMNS):
        raise CandidateMeasurementError(
            "candidate_domain.candidate_domain_complete is false"
        )

    protocol = _require_mapping(
        root["measurement_protocol"], "report.measurement_protocol"
    )
    _require_exact_keys(
        protocol,
        {
            "warmup_runs_per_candidate",
            "warmup_samples_retained",
            "measured_samples_per_candidate",
            "correctness",
            "sample_correlation",
        },
        "report.measurement_protocol",
    )
    sample_count = _positive_int(
        protocol["measured_samples_per_candidate"],
        "report.measurement_protocol.measured_samples_per_candidate",
    )
    _require_exact_int(
        protocol["warmup_runs_per_candidate"],
        WARMUP_RUNS,
        "measurement_protocol.warmup_runs_per_candidate",
    )
    _require_exact_int(
        protocol["warmup_samples_retained"],
        0,
        "measurement_protocol.warmup_samples_retained",
    )
    if (
        sample_count < MIN_MEASURED_SAMPLES
        or protocol["correctness"] != "bit_exact_numpy_uint16_mod_2^16_each_invocation"
        or protocol["sample_correlation"]
        != "one sample object contains device and host timings from one invocation"
    ):
        raise CandidateMeasurementError("report.measurement_protocol is malformed")

    input_data = _require_mapping(root["input_data"], "report.input_data")
    _require_exact_keys(
        input_data,
        {
            "generator",
            "lhs_sha256",
            "rhs_sha256",
            "accumulator_sha256",
            "expected_sha256",
        },
        "report.input_data",
    )
    if input_data["generator"] != "deterministic_linear_u16_v1":
        raise CandidateMeasurementError("report.input_data.generator is malformed")
    for field in ("lhs_sha256", "rhs_sha256", "accumulator_sha256", "expected_sha256"):
        _require_digest(input_data[field], f"report.input_data.{field}")

    provenance = _require_mapping(root["provenance"], "report.provenance")
    _require_exact_keys(
        provenance,
        {
            "target",
            "source_fingerprint",
            "source_sha256",
            "runtime_driver_sha256",
            "model_fingerprint",
            "platform_contract",
        },
        "report.provenance",
    )
    if provenance["target"] != "apu_v2":
        raise CandidateMeasurementError("report.provenance.target is malformed")
    source_hashes = _normalize_source_hashes(
        provenance["source_sha256"], "report.provenance.source_sha256"
    )
    source_fingerprint = _require_digest(
        provenance["source_fingerprint"], "report.provenance.source_fingerprint"
    )
    if _source_fingerprint(source_hashes) != source_fingerprint:
        raise CandidateMeasurementError("report source fingerprint is inconsistent")
    _require_digest(
        provenance["runtime_driver_sha256"],
        "report.provenance.runtime_driver_sha256",
    )
    model_fingerprint = _require_digest(
        provenance["model_fingerprint"],
        "report.provenance.model_fingerprint",
        length=16,
    )
    prepared = _prepare_campaign(rows, reduction, columns, alpha, beta, batches)
    if dict(provenance) != dict(prepared.provenance):
        raise CandidateMeasurementError(
            "report provenance does not match the current exact campaign"
        )
    if root["promotion_eligibility"] != _promotion_eligibility(
        prepared.provenance["platform_contract"]
    ):
        raise CandidateMeasurementError("report promotion eligibility is malformed")
    rejections = root["rejections"]
    if not isinstance(rejections, list) or rejections:
        raise CandidateMeasurementError(
            "persistent-GEMM reports require zero candidate rejections"
        )
    _, _, _, _, expected_input_data = _deterministic_inputs(
        rows,
        reduction,
        columns,
        alpha,
        beta,
    )
    if dict(input_data) != expected_input_data:
        raise CandidateMeasurementError("report input data is stale")

    candidates = root["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(batches):
        raise CandidateMeasurementError(
            "report.candidates must contain exactly one record per candidate"
        )
    seen_candidates = set()
    seen_materializations = set()
    for candidate_index, (batch, candidate_value, prepared_candidate) in enumerate(
        zip(batches, candidates, prepared.candidates)
    ):
        candidate = _require_mapping(
            candidate_value, f"report.candidates[{candidate_index}]"
        )
        _require_exact_keys(
            candidate,
            {
                "candidate_index",
                "identity",
                "schedule",
                "materialization",
                "model_prediction",
                "samples",
            },
            f"report.candidates[{candidate_index}]",
        )
        _require_exact_int(
            candidate["candidate_index"],
            candidate_index,
            f"report.candidates[{candidate_index}].candidate_index",
        )
        schedule = _require_mapping(
            candidate["schedule"], f"report.candidates[{candidate_index}].schedule"
        )
        expected_schedule = _schedule_manifest(batch)
        if set(schedule) != set(expected_schedule) or any(
            schedule[field] != expected_value
            or type(schedule[field]) is not type(expected_value)
            for field, expected_value in expected_schedule.items()
        ):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} schedule is malformed"
            )
        materialization = _require_mapping(
            candidate["materialization"],
            f"report.candidates[{candidate_index}].materialization",
        )
        expected_materialization = {
            "M": rows,
            "K": reduction,
            "N": columns,
            "alpha": alpha,
            "beta": beta,
        }
        if set(materialization) != set(expected_materialization):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} materialization is malformed"
            )
        _positive_int(materialization["M"], "candidate.materialization.M")
        _positive_int(materialization["K"], "candidate.materialization.K")
        _positive_int(materialization["N"], "candidate.materialization.N")
        _u16_int(materialization["alpha"], "candidate.materialization.alpha")
        _u16_int(materialization["beta"], "candidate.materialization.beta")
        if dict(materialization) != expected_materialization:
            raise CandidateMeasurementError(
                f"candidate {candidate_index} materialization is malformed"
            )

        identity = _require_mapping(
            candidate["identity"], f"report.candidates[{candidate_index}].identity"
        )
        _require_exact_keys(
            identity,
            {
                "candidate_fingerprint",
                "schedule_fingerprint",
                "materialization_fingerprint",
                "semantic_fingerprint",
                "source_fingerprint",
                "model_fingerprint",
                "prediction_fingerprint",
            },
            f"report.candidates[{candidate_index}].identity",
        )
        for field in (
            "candidate_fingerprint",
            "schedule_fingerprint",
            "materialization_fingerprint",
            "semantic_fingerprint",
            "source_fingerprint",
            "prediction_fingerprint",
        ):
            _require_digest(identity[field], f"candidate {candidate_index}.{field}")
        _require_digest(
            identity["model_fingerprint"],
            f"candidate {candidate_index}.model_fingerprint",
            length=16,
        )
        if (
            identity["source_fingerprint"] != source_fingerprint
            or identity["model_fingerprint"] != model_fingerprint
        ):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} has mixed source or model provenance"
            )
        expected_semantic = _apu_g2_persistent_semantic_fingerprint(
            _structural_gemm_plan(rows, reduction, columns, alpha, beta)
        )
        if identity["semantic_fingerprint"] != expected_semantic:
            raise CandidateMeasurementError(
                f"candidate {candidate_index} semantic fingerprint is inconsistent"
            )
        if identity["schedule_fingerprint"] != _fingerprint(
            "apu-g2-persistent-gemm-schedule-v1", expected_schedule
        ):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} schedule fingerprint is inconsistent"
            )
        if identity[
            "materialization_fingerprint"
        ] != _recompute_materialization_fingerprint(candidate):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} materialization fingerprint is inconsistent"
            )
        identity_without_candidate = {
            field: identity[field]
            for field in (
                "schedule_fingerprint",
                "materialization_fingerprint",
                "semantic_fingerprint",
                "source_fingerprint",
                "model_fingerprint",
                "prediction_fingerprint",
            )
        }
        if identity["candidate_fingerprint"] != _fingerprint(
            "apu-g2-persistent-gemm-measurement-candidate-v1",
            identity_without_candidate,
        ):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} fingerprint is inconsistent"
            )
        if dict(identity) != dict(prepared_candidate.identity):
            raise CandidateMeasurementError(
                f"candidate {candidate_index} identity is stale"
            )
        if identity["candidate_fingerprint"] in seen_candidates:
            raise CandidateMeasurementError("duplicate candidate fingerprint")
        if identity["materialization_fingerprint"] in seen_materializations:
            raise CandidateMeasurementError("duplicate materialization fingerprint")
        seen_candidates.add(identity["candidate_fingerprint"])
        seen_materializations.add(identity["materialization_fingerprint"])

        prediction = _require_mapping(
            candidate["model_prediction"],
            f"report.candidates[{candidate_index}].model_prediction",
        )
        _require_exact_keys(
            prediction,
            set(_MODEL_PREDICTION_FIELDS),
            f"report.candidates[{candidate_index}].model_prediction",
        )
        normalized_prediction = _normalize_model_prediction(prediction)
        if dict(prediction) != normalized_prediction:
            raise CandidateMeasurementError(
                f"candidate {candidate_index} model prediction is malformed"
            )
        if normalized_prediction != prepared_candidate.model_prediction:
            raise CandidateMeasurementError(
                f"candidate {candidate_index} model prediction is stale"
            )

        sample_values = candidate["samples"]
        if not isinstance(sample_values, list) or len(sample_values) != sample_count:
            raise CandidateMeasurementError(
                f"candidate {candidate_index} has insufficient timing samples"
            )
        for sample_index, sample_value in enumerate(sample_values):
            sample = _require_mapping(
                sample_value,
                f"report.candidates[{candidate_index}].samples[{sample_index}]",
            )
            _require_exact_keys(
                sample,
                {
                    "sample_index",
                    "device_pipeline_ticks",
                    "hardware_tasks",
                    "weight_uploads",
                    "host_us",
                },
                f"report.candidates[{candidate_index}].samples[{sample_index}]",
            )
            _require_exact_int(
                sample["sample_index"], sample_index, "sample.sample_index"
            )
            _positive_int(
                sample["device_pipeline_ticks"], "sample.device_pipeline_ticks"
            )
            if (
                _positive_int(sample["hardware_tasks"], "sample.hardware_tasks")
                != normalized_prediction["hardware_tasks"]
                or _positive_int(sample["weight_uploads"], "sample.weight_uploads")
                != normalized_prediction["weight_uploads"]
            ):
                raise CandidateMeasurementError(
                    f"candidate {candidate_index} sample counters are inconsistent"
                )
            host = _require_mapping(sample["host_us"], "sample.host_us")
            _require_exact_keys(
                host,
                {"h2d_us", "host_task_us", "d2h_us", "end_to_end_us", "wall_us"},
                "sample.host_us",
            )
            normalized_host = {
                field: _positive_float(host[field], f"sample.host_us.{field}")
                for field in (
                    "h2d_us",
                    "host_task_us",
                    "d2h_us",
                    "end_to_end_us",
                    "wall_us",
                )
            }
            if normalized_host["wall_us"] != (
                normalized_host["h2d_us"]
                + normalized_host["host_task_us"]
                + normalized_host["d2h_us"]
            ):
                raise CandidateMeasurementError(
                    f"candidate {candidate_index} sample host phases are uncorrelated"
                )

    measurement_fingerprint = _require_digest(
        root["measurement_fingerprint"], "report.measurement_fingerprint"
    )
    body = {key: root[key] for key in root if key != "measurement_fingerprint"}
    if measurement_fingerprint != _fingerprint(
        "apu-g2-persistent-gemm-candidate-measurements-v1", body
    ):
        raise CandidateMeasurementError("report measurement fingerprint is stale")


def report_to_json(report: object) -> str:
    validate_measurement_report(report)
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


def write_measurement_report(
    report: object, output: str | Path, *, overwrite: bool = False
) -> None:
    output = Path(output)
    rendered = report_to_json(report)
    mode = "w" if overwrite else "x"
    try:
        with output.open(mode, encoding="utf-8") as stream:
            stream.write(rendered)
    except FileExistsError as error:
        raise CandidateMeasurementError(
            f"refusing to overwrite existing output {output}; pass --overwrite"
        ) from error


def _positive_cli_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _u16_cli_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= value <= 0xFFFF:
        raise argparse.ArgumentTypeError("must fit uint16")
    return value


def _batch_columns_cli(text: str) -> tuple[int, ...]:
    pieces = text.split(",")
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("must be a comma-separated integer list")
    try:
        return tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated integer list"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure structural APUg2 persistent-GEMM batch-column candidates."
        )
    )
    parser.add_argument("M", type=_positive_cli_int, help="output rows")
    parser.add_argument("K", type=_positive_cli_int, help="reduction extent")
    parser.add_argument("N", type=_positive_cli_int, help="output columns")
    parser.add_argument("--alpha", type=_u16_cli_int, default=1)
    parser.add_argument("--beta", type=_u16_cli_int, default=1)
    parser.add_argument(
        "--batch-columns",
        action="append",
        type=_batch_columns_cli,
        metavar="B[,B...]",
        help=(
            "candidate subset; repeat or comma-separate values "
            f"(default: full 1..{APUG2_U16_GEMM_BMAX} domain)"
        ),
    )
    parser.add_argument(
        "--samples",
        type=_positive_cli_int,
        default=MIN_MEASURED_SAMPLES,
        help=f"retained samples per candidate (minimum {MIN_MEASURED_SAMPLES})",
    )
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing --output file",
    )
    return parser


def _check_output_destination(output: Path | None, overwrite: bool) -> None:
    if overwrite and output is None:
        raise CandidateMeasurementError("--overwrite requires --output")
    if output is None:
        return
    if output.exists() and output.is_dir():
        raise CandidateMeasurementError(f"output is a directory: {output}")
    if output.exists() and not overwrite:
        raise CandidateMeasurementError(
            f"refusing to overwrite existing output {output}; pass --overwrite"
        )
    if not output.parent.is_dir():
        raise CandidateMeasurementError(
            f"output parent directory does not exist: {output.parent}"
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_output_destination(args.output, args.overwrite)
        batch_columns = None
        if args.batch_columns is not None:
            batch_columns = tuple(
                batch for group in args.batch_columns for batch in group
            )
        report = measure_apu_g2_persistent_candidates(
            args.M,
            args.K,
            args.N,
            alpha=args.alpha,
            beta=args.beta,
            batch_columns=batch_columns,
            samples=args.samples,
            runner=runner,
        )
        if args.output is None:
            sys.stdout.write(report_to_json(report))
        else:
            write_measurement_report(report, args.output, overwrite=args.overwrite)
    except (CandidateMeasurementError, OSError, RuntimeError, TypeError) as error:
        print(f"FAIL APUg2 persistent candidate measurement: {error}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "CandidateMeasurementError",
    "DEFAULT_BATCH_COLUMNS",
    "MIN_MEASURED_SAMPLES",
    "current_runtime_provenance",
    "main",
    "measure_apu_g2_persistent_candidates",
    "report_to_json",
    "validate_measurement_report",
    "write_measurement_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
