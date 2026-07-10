#!/usr/bin/env python3
"""Build a structural calibration report from APUv1 Tenon artifacts.

The retained artifacts measure only the selected row-tile candidate.  This
report therefore evaluates selected-candidate point predictions across shapes
and does not claim within-shape ranking quality for unmeasured challengers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any


DEFAULT_ARTIFACTS_ROOT = Path("/home/nz264/shared/tenon-artifacts")
CORPUS_RELATIVE_PATH = Path("polybench/apu-v1/tenon")

BACKEND = "apu-v1"
FRAMEWORK = "tenon"
DATASET = "PolyBench/C 4.2.1 LARGE"
DTYPE = "uint16 (16-bit modular integer datapath)"
HARDWARE = "GSI Leda APU v1, GSI SDK 13.7.1, 500 MHz cycle conversion"
BOARD_HARDWARE = "GSI Leda APU v1, GSI SDK 13.7.1"
RESULT_METRIC = "device-compute microseconds"
COMPOSITION = "sum of sequential exact-size contraction legs on one APUC"
EXECUTION_SCOPE = "single APUC (matched to the Exo baseline)"
ROW_TILE_CRITERION = "minimum executable-cost estimate for single-APUC composition"
BOARD_COUNTER_SOURCE = "newest matching PROF_PRINT total in board transcript"
BOARD_HEADER = "Tenon APU-v1 real-device profile"

DENSE_SEARCHED = "dense_searched"
SINGLETON_GEMV_BYPASS = "singleton_gemv_bypass"
REDUCTION_ONE_BYPASS = "reduction_extent_one_bypass"
_CATEGORIES = (
    DENSE_SEARCHED,
    SINGLETON_GEMV_BYPASS,
    REDUCTION_ONE_BYPASS,
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "kernel",
        "backend",
        "framework",
        "dataset",
        "dtype",
        "status",
        "detail",
        "result",
        "tenon_revision",
        "hardware",
    }
)
_RESULT_FIELDS = frozenset(
    {"metric", "cycles", "microseconds", "composition", "execution_scope", "legs"}
)
_LEG_FIELDS = frozenset(
    {
        "name",
        "shape",
        "row_waves",
        "column_tiles",
        "parallel_apucs",
        "composition_repeats",
        "operation_count",
        "cycles",
        "microseconds",
        "plan",
        "correctness",
        "profiles",
        "full_workload_mlir",
    }
)
_PROFILE_FIELDS = frozenset(
    {
        "name",
        "profile_shape",
        "tile_count",
        "row_waves",
        "composition_repeats",
        "profile_cycles",
        "profile_microseconds",
        "cycles",
        "microseconds",
        "plan",
        "row_tile_selection",
        "correctness",
        "profile",
        "compiled_profile_mlir",
        "compiled_gvml",
    }
)
_SELECTION_FIELDS = frozenset({"criterion", "candidates"})
_CANDIDATE_FIELDS = frozenset(
    {
        "tile_m",
        "row_waves",
        "plan",
        "estimated_shard_cycles",
        "estimated_composed_cycles",
    }
)
_SHAPE_FIELDS = frozenset({"M", "K", "N"})
_MEASURED_STATUSES = frozenset({"PASS", "PARTIAL"})
_ALL_STATUSES = frozenset({*_MEASURED_STATUSES, "NOT-APPLICABLE"})
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_PHYSICAL_SHAPE_RE = re.compile(r"M=(\d+) N=(\d+) K=(\d+)\Z")


class ArtifactReportError(ValueError):
    """Raised when artifact evidence cannot support a trustworthy report."""


@dataclass(frozen=True)
class _Observation:
    category: str
    semantic_shape: tuple[int, int, int]
    physical_schedule_json: str
    selected_materialization_json: str
    candidate_domain_json: str
    profile_cycles: int
    measurement_fingerprint: str
    source: Mapping[str, object]


@dataclass(frozen=True)
class _ProfileEvidence:
    observation: _Observation
    profile_shape: tuple[int, int, int]
    tile_count: int
    row_waves: int
    composition_repeats: int
    cycles: int
    plan: str


@dataclass(frozen=True)
class _CorpusEvidence:
    observations: tuple[_Observation, ...]
    result_sources: tuple[Mapping[str, object], ...]
    not_applicable_sources: tuple[Mapping[str, object], ...]
    status_counts: Mapping[str, int]
    tenon_revisions: tuple[str, ...]


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


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactReportError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except ArtifactReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactReportError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactReportError(f"{label} {path} must contain a JSON object")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unsupported " + ", ".join(unknown))
        raise ArtifactReportError(f"{field} has invalid fields: {'; '.join(details)}")


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactReportError(f"{field} must be an object")
    return value


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactReportError(f"{field} must be an array")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactReportError(f"{field} must be a nonempty string")
    return value


def _exact_string(value: object, expected: str, field: str) -> str:
    result = _nonempty_string(value, field)
    if result != expected:
        raise ArtifactReportError(f"{field} must be {expected!r}")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArtifactReportError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ArtifactReportError(f"{field} must be a positive integer")
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


def _require_pass(value: object, field: str) -> str:
    result = _nonempty_string(value, field)
    normalized = result.strip().lower()
    if not normalized.startswith("pass") or any(
        token in normalized for token in ("fail", "error", "incorrect")
    ):
        raise ArtifactReportError(f"{field} must prove passing correctness")
    return result


def _validate_microseconds(cycles: int, value: object, field: str) -> float:
    microseconds = _positive_float(value, field)
    expected = cycles / 500.0
    if not math.isclose(microseconds, expected, rel_tol=0.0, abs_tol=5e-7):
        raise ArtifactReportError(f"{field} disagrees with {cycles} cycles at 500 MHz")
    return microseconds


def _shape(value: object, field: str) -> tuple[int, int, int]:
    shape = _require_mapping(value, field)
    _require_exact_fields(shape, _SHAPE_FIELDS, field)
    return tuple(
        _positive_int(shape[dimension], f"{field}.{dimension}")
        for dimension in ("M", "K", "N")
    )


def _shape_manifest(shape: tuple[int, int, int]) -> dict[str, int]:
    rows, reduction, columns = shape
    return {"M": rows, "K": reduction, "N": columns}


def _classify_shape(shape: tuple[int, int, int], field: str) -> str:
    _rows, reduction, columns = shape
    if columns == 1 and reduction == 1:
        raise ArtifactReportError(
            f"{field} is ambiguously both a singleton GEMV and K==1 bypass"
        )
    if columns == 1:
        return SINGLETON_GEMV_BYPASS
    if reduction == 1:
        return REDUCTION_ONE_BYPASS
    return DENSE_SEARCHED


def _contained_file(base: Path, relative: object, field: str) -> tuple[Path, str]:
    relative_string = _nonempty_string(relative, field)
    pure = PurePosixPath(relative_string)
    if pure.is_absolute() or ".." in pure.parts:
        raise ArtifactReportError(f"{field} must stay within its result directory")
    path = base.joinpath(*pure.parts)
    try:
        resolved_base = base.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_base)
    except (OSError, ValueError) as error:
        raise ArtifactReportError(
            f"{field} does not name a contained file: {relative_string}"
        ) from error
    if not resolved.is_file():
        raise ArtifactReportError(f"{field} does not name a file: {relative_string}")
    return resolved, pure.as_posix()


def _read_text_evidence(path: Path, field: str) -> tuple[str, str]:
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactReportError(f"cannot read {field} {path}: {error}") from error
    if not content:
        raise ArtifactReportError(f"{field} {path} must not be empty")
    return text, hashlib.sha256(content).hexdigest()


def _validate_mlir(text: str, shape: tuple[int, int, int], field: str) -> None:
    rows, reduction, columns = shape
    required = (
        "func.func",
        f"memref<{rows}x{reduction}xi16>",
        f"memref<{reduction}x{columns}xi16>",
        f"memref<{rows}x{columns}xi16>",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise ArtifactReportError(
            f"{field} does not encode the declared contraction shape: {missing}"
        )


def _board_log_manifest(
    text: str,
    *,
    shape: tuple[int, int, int],
    plan: str,
    cycles: int,
    field: str,
) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != BOARD_HEADER:
        raise ArtifactReportError(f"{field} lacks the APUv1 profile header")

    required_labels = (
        "hardware",
        "kernel",
        "physical_shape",
        "selected_plan",
        "correctness",
        "cycles",
        "microseconds_at_500MHz",
        "counter_source",
    )
    values: dict[str, str] = {}
    for label in required_labels:
        prefix = f"{label}: "
        matches = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise ArtifactReportError(f"{field} must contain exactly one {label!r}")
        values[label] = matches[0]

    _exact_string(values["hardware"], BOARD_HARDWARE, f"{field}.hardware")
    board_kernel = _nonempty_string(values["kernel"], f"{field}.kernel")
    match = _PHYSICAL_SHAPE_RE.fullmatch(values["physical_shape"])
    if match is None:
        raise ArtifactReportError(f"{field}.physical_shape is malformed")
    board_shape = (int(match.group(1)), int(match.group(3)), int(match.group(2)))
    if board_shape != shape:
        raise ArtifactReportError(f"{field}.physical_shape disagrees with result.json")
    if values["selected_plan"] != plan:
        raise ArtifactReportError(f"{field}.selected_plan disagrees with result.json")
    _require_pass(values["correctness"], f"{field}.correctness")
    board_cycles = _positive_int_from_text(values["cycles"], f"{field}.cycles")
    if board_cycles != cycles:
        raise ArtifactReportError(f"{field}.cycles disagrees with result.json")
    _validate_microseconds(
        cycles,
        _positive_float_from_text(
            values["microseconds_at_500MHz"],
            f"{field}.microseconds_at_500MHz",
        ),
        f"{field}.microseconds_at_500MHz",
    )
    _exact_string(
        values["counter_source"],
        BOARD_COUNTER_SOURCE,
        f"{field}.counter_source",
    )
    counter_cycles = [int(value) for value in re.findall(r"\bcrun:\s*(\d+)", text)]
    if counter_cycles != [cycles]:
        raise ArtifactReportError(f"{field} must contain one matching raw crun counter")
    return {"declared_kernel": board_kernel}


def _positive_int_from_text(value: str, field: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ArtifactReportError(f"{field} must be a positive integer")
    return int(value)


def _positive_float_from_text(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ArtifactReportError(
            f"{field} must be a positive finite number"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ArtifactReportError(f"{field} must be a positive finite number")
    return result


def _physical_schedule(
    *,
    profile_shape: tuple[int, int, int],
    tile_m: int,
    plan: str,
    row_waves: int,
    tile_count: int,
    operation_count: int,
    composition_repeats: int,
) -> dict[str, object]:
    return {
        "profile_shape": _shape_manifest(profile_shape),
        "tile_m": tile_m,
        "historical_plan_label": plan,
        "repetitions": {
            "row_waves": row_waves,
            "column_tiles": tile_count,
            "operation_count": operation_count,
            "composition_repeats": composition_repeats,
        },
    }


def _candidate_manifest(
    raw: Mapping[str, Any],
    *,
    field: str,
    semantic_shape: tuple[int, int, int],
    selected_profile_shape: tuple[int, int, int],
    tile_count: int,
    operation_count: int,
) -> dict[str, object]:
    _require_exact_fields(raw, _CANDIDATE_FIELDS, field)
    tile_m = _positive_int(raw.get("tile_m"), f"{field}.tile_m")
    row_waves = _positive_int(raw.get("row_waves"), f"{field}.row_waves")
    plan = _nonempty_string(raw.get("plan"), f"{field}.plan")
    shard_cycles = _positive_int(
        raw.get("estimated_shard_cycles"), f"{field}.estimated_shard_cycles"
    )
    composed_cycles = _positive_int(
        raw.get("estimated_composed_cycles"),
        f"{field}.estimated_composed_cycles",
    )

    semantic_rows, semantic_reduction, _semantic_columns = semantic_shape
    _selected_rows, profile_reduction, profile_columns = selected_profile_shape
    if profile_reduction != semantic_reduction:
        raise ArtifactReportError(f"{field} mixes reduction extents")
    expected_waves = math.ceil(semantic_rows / tile_m)
    if row_waves != expected_waves:
        raise ArtifactReportError(
            f"{field}.row_waves does not exactly cover semantic M"
        )
    composition_repeats = row_waves * tile_count * operation_count
    if composed_cycles != shard_cycles * composition_repeats:
        raise ArtifactReportError(
            f"{field}.estimated_composed_cycles disagrees with repetitions"
        )

    schedule = _physical_schedule(
        profile_shape=(tile_m, profile_reduction, profile_columns),
        tile_m=tile_m,
        plan=plan,
        row_waves=row_waves,
        tile_count=tile_count,
        operation_count=operation_count,
        composition_repeats=composition_repeats,
    )
    schedule_fingerprint = _fingerprint(
        "apu-v1-opaque-historical-profile-schedule-v2", schedule
    )
    candidate_identity = {
        "semantic_shape": _shape_manifest(semantic_shape),
        "opaque_historical_schedule": schedule,
    }
    candidate_fingerprint = _fingerprint(
        "apu-v1-historical-profile-candidate-identity-v2", candidate_identity
    )
    estimate = {
        "profile_cycles": shard_cycles,
        "composed_cycles": composed_cycles,
    }
    return {
        "opaque_historical_schedule": schedule,
        "opaque_historical_schedule_fingerprint": schedule_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "identity_kind": "opaque_historical_plan_label",
        "physical_manifest_available": False,
        "promotion_eligible": False,
        "promotion_ineligibility_reason": (
            "historical challenger has no canonical physical decision or exact "
            "compiled materialization manifest"
        ),
        "analytical_estimate": estimate,
        "analytical_evidence_fingerprint": _fingerprint(
            "apu-v1-profile-candidate-analytical-evidence-v1",
            {"candidate_fingerprint": candidate_fingerprint, "estimate": estimate},
        ),
    }


def _candidate_domain(
    selection: Mapping[str, Any],
    *,
    field: str,
    category: str,
    semantic_shape: tuple[int, int, int],
    selected_schedule: Mapping[str, object],
    selected_profile_shape: tuple[int, int, int],
    tile_count: int,
    operation_count: int,
) -> dict[str, object]:
    _require_exact_fields(selection, _SELECTION_FIELDS, field)
    _exact_string(selection.get("criterion"), ROW_TILE_CRITERION, f"{field}.criterion")
    candidates_raw = _require_sequence(
        selection.get("candidates"), f"{field}.candidates"
    )

    if category != DENSE_SEARCHED:
        if candidates_raw:
            raise ArtifactReportError(
                f"{field}.candidates must be empty for structural bypasses"
            )
        return {
            "status": "not_applicable",
            "reason": category,
            "candidate_count": 0,
            "candidates": [],
        }

    if len(candidates_raw) < 2:
        raise ArtifactReportError(
            f"{field}.candidates must contain a selected candidate and challengers"
        )
    candidates = []
    for index, raw in enumerate(candidates_raw):
        candidate = _require_mapping(raw, f"{field}.candidates[{index}]")
        candidates.append(
            _candidate_manifest(
                candidate,
                field=f"{field}.candidates[{index}]",
                semantic_shape=semantic_shape,
                selected_profile_shape=selected_profile_shape,
                tile_count=tile_count,
                operation_count=operation_count,
            )
        )

    candidate_fingerprints = [
        str(candidate["candidate_fingerprint"]) for candidate in candidates
    ]
    if len(set(candidate_fingerprints)) != len(candidate_fingerprints):
        raise ArtifactReportError(f"{field}.candidates contains duplicate schedules")

    selected_schedule_json = _canonical_json(selected_schedule)
    selected = [
        candidate
        for candidate in candidates
        if _canonical_json(candidate["opaque_historical_schedule"])
        == selected_schedule_json
    ]
    if len(selected) != 1:
        raise ArtifactReportError(
            f"{field}.candidates must contain exactly one selected physical schedule"
        )
    selected_fingerprint = str(selected[0]["candidate_fingerprint"])
    minimum = min(
        int(candidate["analytical_estimate"]["composed_cycles"])
        for candidate in candidates
    )
    minimizers = [
        candidate
        for candidate in candidates
        if int(candidate["analytical_estimate"]["composed_cycles"]) == minimum
    ]
    if len(minimizers) != 1 or minimizers[0] is not selected[0]:
        raise ArtifactReportError(
            f"{field} selected schedule is not the unique analytical minimum"
        )

    rendered_candidates = [
        {**candidate, "selected": candidate is selected[0]}
        for candidate in sorted(
            candidates,
            key=lambda item: (
                int(item["opaque_historical_schedule"]["tile_m"]),
                str(item["opaque_historical_schedule"]["historical_plan_label"]),
                str(item["candidate_fingerprint"]),
            ),
        )
    ]
    body: dict[str, object] = {
        "status": "available",
        "selection_rule": "unique_minimum_estimated_composed_cycles",
        "candidate_count": len(rendered_candidates),
        "selected_candidate_fingerprint": selected_fingerprint,
        "candidates": rendered_candidates,
    }
    return {
        **body,
        "historical_candidate_domain_fingerprint": _fingerprint(
            "apu-v1-historical-profile-candidate-domain-v2",
            {
                "semantic_shape": _shape_manifest(semantic_shape),
                "domain": body,
            },
        ),
    }


def _profile_evidence(
    *,
    corpus_root: Path,
    result_path: Path,
    result_data: Mapping[str, Any],
    result_source: Mapping[str, object],
    result_content_fingerprint: str,
    leg: Mapping[str, Any],
    leg_index: int,
    semantic_shape: tuple[int, int, int],
    leg_row_waves: int,
    operation_count: int,
    full_workload_file: Mapping[str, str],
    profile: Mapping[str, Any],
    profile_index: int,
) -> _ProfileEvidence:
    result_relative = result_path.relative_to(corpus_root).as_posix()
    field = f"{result_relative}.result.legs[{leg_index}].profiles[{profile_index}]"
    _require_exact_fields(profile, _PROFILE_FIELDS, field)
    profile_name = _nonempty_string(profile.get("name"), f"{field}.name")
    profile_shape = _shape(profile.get("profile_shape"), f"{field}.profile_shape")
    semantic_rows, semantic_reduction, semantic_columns = semantic_shape
    profile_rows, profile_reduction, profile_columns = profile_shape
    if profile_reduction != semantic_reduction:
        raise ArtifactReportError(f"{field}.profile_shape.K disagrees with leg shape")
    if profile_rows > semantic_rows or profile_columns > semantic_columns:
        raise ArtifactReportError(f"{field}.profile_shape exceeds the semantic shape")

    tile_count = _positive_int(profile.get("tile_count"), f"{field}.tile_count")
    row_waves = _positive_int(profile.get("row_waves"), f"{field}.row_waves")
    if row_waves != leg_row_waves or row_waves != math.ceil(
        semantic_rows / profile_rows
    ):
        raise ArtifactReportError(
            f"{field}.row_waves does not exactly cover semantic M"
        )
    composition_repeats = _positive_int(
        profile.get("composition_repeats"), f"{field}.composition_repeats"
    )
    expected_repeats = row_waves * tile_count * operation_count
    if composition_repeats != expected_repeats:
        raise ArtifactReportError(
            f"{field}.composition_repeats disagrees with physical repetitions"
        )
    plan = _nonempty_string(profile.get("plan"), f"{field}.plan")
    profile_cycles = _positive_int(
        profile.get("profile_cycles"), f"{field}.profile_cycles"
    )
    _validate_microseconds(
        profile_cycles,
        profile.get("profile_microseconds"),
        f"{field}.profile_microseconds",
    )
    cycles = _positive_int(profile.get("cycles"), f"{field}.cycles")
    if cycles != profile_cycles * composition_repeats:
        raise ArtifactReportError(f"{field}.cycles disagrees with profile repetitions")
    _validate_microseconds(cycles, profile.get("microseconds"), f"{field}.microseconds")
    profile_correctness = _require_pass(
        profile.get("correctness"), f"{field}.correctness"
    )

    category = _classify_shape(semantic_shape, f"{field}.semantic_shape")
    if category != DENSE_SEARCHED and profile_shape != semantic_shape:
        raise ArtifactReportError(
            f"{field} bypass must profile the full semantic shape"
        )
    selected_schedule = _physical_schedule(
        profile_shape=profile_shape,
        tile_m=profile_rows,
        plan=plan,
        row_waves=row_waves,
        tile_count=tile_count,
        operation_count=operation_count,
        composition_repeats=composition_repeats,
    )
    selection = _require_mapping(
        profile.get("row_tile_selection"), f"{field}.row_tile_selection"
    )
    candidate_domain = _candidate_domain(
        selection,
        field=f"{field}.row_tile_selection",
        category=category,
        semantic_shape=semantic_shape,
        selected_schedule=selected_schedule,
        selected_profile_shape=profile_shape,
        tile_count=tile_count,
        operation_count=operation_count,
    )

    result_directory = result_path.parent
    board_path, board_relative = _contained_file(
        result_directory, profile.get("profile"), f"{field}.profile"
    )
    profile_mlir_path, profile_mlir_relative = _contained_file(
        result_directory,
        profile.get("compiled_profile_mlir"),
        f"{field}.compiled_profile_mlir",
    )
    gvml_path, gvml_relative = _contained_file(
        result_directory,
        profile.get("compiled_gvml"),
        f"{field}.compiled_gvml",
    )

    board_text, board_sha256 = _read_text_evidence(board_path, f"{field}.profile")
    board_manifest = _board_log_manifest(
        board_text,
        shape=profile_shape,
        plan=plan,
        cycles=profile_cycles,
        field=f"{field}.profile",
    )
    profile_mlir, profile_mlir_sha256 = _read_text_evidence(
        profile_mlir_path, f"{field}.compiled_profile_mlir"
    )
    _validate_mlir(profile_mlir, profile_shape, f"{field}.compiled_profile_mlir")
    gvml, gvml_sha256 = _read_text_evidence(gvml_path, f"{field}.compiled_gvml")
    if plan not in gvml or "gvml_init_once()" not in gvml:
        raise ArtifactReportError(
            f"{field}.compiled_gvml does not encode the selected plan"
        )

    source_files = {
        "board_log": {
            "path": (result_path.parent / board_relative)
            .relative_to(corpus_root)
            .as_posix(),
            "sha256": board_sha256,
        },
        "profile_mlir": {
            "path": (result_path.parent / profile_mlir_relative)
            .relative_to(corpus_root)
            .as_posix(),
            "sha256": profile_mlir_sha256,
        },
        "gvml": {
            "path": (result_path.parent / gvml_relative)
            .relative_to(corpus_root)
            .as_posix(),
            "sha256": gvml_sha256,
        },
        "full_workload_mlir": dict(full_workload_file),
    }
    selected_materialization = {
        "identity_kind": "exact_compiled_source_hashes",
        "profile_mlir_sha256": profile_mlir_sha256,
        "gvml_sha256": gvml_sha256,
        "typed_physical_manifest_available": False,
        "promotion_eligible": False,
        "promotion_ineligibility_reason": (
            "historical profile lacks repeated warmup/sample evidence and a "
            "canonical typed physical decision manifest"
        ),
    }
    historical_record_identity = {
        "semantic_shape": _shape_manifest(semantic_shape),
        "opaque_historical_schedule": selected_schedule,
        "selected_materialization": selected_materialization,
    }
    historical_record_identity_fingerprint = _fingerprint(
        "apu-v1-profile-historical-record-identity-v2",
        historical_record_identity,
    )
    measurement_fingerprint = _fingerprint(
        "apu-v1-profile-measurement-v2",
        {
            "historical_record_identity_fingerprint": (
                historical_record_identity_fingerprint
            ),
            "profile_cycles": profile_cycles,
        },
    )
    source_manifest: dict[str, object] = {
        "result_path": result_relative,
        "leg_index": leg_index,
        "profile_index": profile_index,
        "declared_names": {
            "result": result_source["declared_kernel"],
            "leg": _nonempty_string(leg.get("name"), f"{field}.leg_name"),
            "profile": profile_name,
            "board_log": board_manifest["declared_kernel"],
        },
        "result_status": result_source["status"],
        "result_detail": result_source["detail"],
        "tenon_revision": result_source["tenon_revision"],
        "declared_correctness": {
            "leg": leg["correctness"],
            "profile": profile_correctness,
            "board_log": "PASS bit-exact modulo 2^16",
        },
        "source_files": source_files,
        "result_content_fingerprint": result_content_fingerprint,
        "leg_entry_fingerprint": _fingerprint("canonical-apu-v1-result-leg-v1", leg),
        "profile_entry_fingerprint": _fingerprint(
            "canonical-apu-v1-result-profile-v1", profile
        ),
        "measurement_fingerprint": measurement_fingerprint,
    }
    source_manifest["provenance_fingerprint"] = _fingerprint(
        "apu-v1-profile-provenance-v1", source_manifest
    )
    return _ProfileEvidence(
        observation=_Observation(
            category=category,
            semantic_shape=semantic_shape,
            physical_schedule_json=_canonical_json(selected_schedule),
            selected_materialization_json=_canonical_json(selected_materialization),
            candidate_domain_json=_canonical_json(candidate_domain),
            profile_cycles=profile_cycles,
            measurement_fingerprint=measurement_fingerprint,
            source=source_manifest,
        ),
        profile_shape=profile_shape,
        tile_count=tile_count,
        row_waves=row_waves,
        composition_repeats=composition_repeats,
        cycles=cycles,
        plan=plan,
    )


def _leg_evidence(
    *,
    corpus_root: Path,
    result_path: Path,
    result_data: Mapping[str, Any],
    result_source: Mapping[str, object],
    result_content_fingerprint: str,
    leg: Mapping[str, Any],
    leg_index: int,
) -> tuple[tuple[_Observation, ...], int]:
    result_relative = result_path.relative_to(corpus_root).as_posix()
    field = f"{result_relative}.result.legs[{leg_index}]"
    _require_exact_fields(leg, _LEG_FIELDS, field)
    _nonempty_string(leg.get("name"), f"{field}.name")
    semantic_shape = _shape(leg.get("shape"), f"{field}.shape")
    row_waves = _positive_int(leg.get("row_waves"), f"{field}.row_waves")
    column_tiles = _positive_int(leg.get("column_tiles"), f"{field}.column_tiles")
    parallel_apucs = _positive_int(leg.get("parallel_apucs"), f"{field}.parallel_apucs")
    if parallel_apucs != 1:
        raise ArtifactReportError(f"{field}.parallel_apucs must be one")
    operation_count = _positive_int(
        leg.get("operation_count"), f"{field}.operation_count"
    )
    composition_repeats = _positive_int(
        leg.get("composition_repeats"), f"{field}.composition_repeats"
    )
    if composition_repeats != row_waves * column_tiles * operation_count:
        raise ArtifactReportError(
            f"{field}.composition_repeats disagrees with leg repetitions"
        )
    leg_cycles = _positive_int(leg.get("cycles"), f"{field}.cycles")
    _validate_microseconds(leg_cycles, leg.get("microseconds"), f"{field}.microseconds")
    leg_correctness = _require_pass(leg.get("correctness"), f"{field}.correctness")

    full_mlir_path, full_mlir_relative = _contained_file(
        result_path.parent,
        leg.get("full_workload_mlir"),
        f"{field}.full_workload_mlir",
    )
    full_mlir, full_mlir_sha256 = _read_text_evidence(
        full_mlir_path, f"{field}.full_workload_mlir"
    )
    _validate_mlir(full_mlir, semantic_shape, f"{field}.full_workload_mlir")
    full_workload_file = {
        "path": (result_path.parent / full_mlir_relative)
        .relative_to(corpus_root)
        .as_posix(),
        "sha256": full_mlir_sha256,
    }

    profiles_raw = _require_sequence(leg.get("profiles"), f"{field}.profiles")
    if not profiles_raw:
        raise ArtifactReportError(f"{field}.profiles must not be empty")
    profile_evidence = []
    for profile_index, profile_raw in enumerate(profiles_raw):
        profile = _require_mapping(profile_raw, f"{field}.profiles[{profile_index}]")
        profile_evidence.append(
            _profile_evidence(
                corpus_root=corpus_root,
                result_path=result_path,
                result_data=result_data,
                result_source=result_source,
                result_content_fingerprint=result_content_fingerprint,
                leg={**leg, "correctness": leg_correctness},
                leg_index=leg_index,
                semantic_shape=semantic_shape,
                leg_row_waves=row_waves,
                operation_count=operation_count,
                full_workload_file=full_workload_file,
                profile=profile,
                profile_index=profile_index,
            )
        )

    if sum(item.tile_count for item in profile_evidence) != column_tiles:
        raise ArtifactReportError(f"{field}.column_tiles disagrees with profiles")
    if (
        sum(item.profile_shape[2] * item.tile_count for item in profile_evidence)
        != semantic_shape[2]
    ):
        raise ArtifactReportError(f"{field}.profiles do not exactly cover semantic N")
    if (
        sum(item.composition_repeats for item in profile_evidence)
        != composition_repeats
    ):
        raise ArtifactReportError(
            f"{field}.composition_repeats disagrees with profiles"
        )
    if sum(item.cycles for item in profile_evidence) != leg_cycles:
        raise ArtifactReportError(f"{field}.cycles disagrees with profiles")

    plans = sorted({item.plan for item in profile_evidence})
    declared_plan = leg.get("plan")
    if len(plans) == 1:
        if declared_plan != plans[0]:
            raise ArtifactReportError(f"{field}.plan disagrees with selected profile")
    else:
        declared_plans = _require_sequence(declared_plan, f"{field}.plan")
        if list(declared_plans) != plans:
            raise ArtifactReportError(f"{field}.plan disagrees with selected profiles")
    return tuple(item.observation for item in profile_evidence), leg_cycles


def _result_source(
    *,
    corpus_root: Path,
    result_path: Path,
    data: Mapping[str, Any],
    result_content_fingerprint: str,
) -> dict[str, object]:
    relative = result_path.relative_to(corpus_root).as_posix()
    source: dict[str, object] = {
        "result_path": relative,
        "declared_kernel": _nonempty_string(data.get("kernel"), f"{relative}.kernel"),
        "status": data["status"],
        "detail": _nonempty_string(data.get("detail"), f"{relative}.detail"),
        "tenon_revision": data["tenon_revision"],
        "result_content_fingerprint": result_content_fingerprint,
    }
    source["provenance_fingerprint"] = _fingerprint(
        "apu-v1-result-provenance-v1", source
    )
    return source


def _resolve_corpus_root(artifacts_root: str | Path) -> tuple[Path, Path]:
    root_path = Path(artifacts_root)
    try:
        root = root_path.resolve(strict=True)
    except OSError as error:
        raise ArtifactReportError(
            f"artifact root does not exist: {root_path}"
        ) from error
    if not root.is_dir():
        raise ArtifactReportError(f"artifact root is not a directory: {root_path}")
    nested = root / CORPUS_RELATIVE_PATH
    if nested.is_dir():
        return root, nested.resolve(strict=True)
    if (
        root.name == "tenon"
        and root.parent.name == "apu-v1"
        and root.parent.parent.name == "polybench"
    ):
        return root, root
    raise ArtifactReportError(f"artifact root lacks {CORPUS_RELATIVE_PATH.as_posix()}")


def _load_corpus(artifacts_root: str | Path) -> _CorpusEvidence:
    _root, corpus_root = _resolve_corpus_root(artifacts_root)
    result_paths = sorted(
        corpus_root.glob("*/result.json"),
        key=lambda path: path.relative_to(corpus_root).as_posix(),
    )
    if not result_paths:
        raise ArtifactReportError(
            "APUv1 Tenon corpus contains no direct result.json records"
        )

    observations: list[_Observation] = []
    result_sources: list[Mapping[str, object]] = []
    not_applicable_sources: list[Mapping[str, object]] = []
    statuses: Counter[str] = Counter()
    revisions: set[str] = set()

    for result_path in result_paths:
        relative = result_path.relative_to(corpus_root).as_posix()
        data = _load_json_object(result_path, "APUv1 result")
        _require_exact_fields(data, _TOP_LEVEL_FIELDS, relative)
        _exact_string(data.get("backend"), BACKEND, f"{relative}.backend")
        _exact_string(data.get("framework"), FRAMEWORK, f"{relative}.framework")
        _exact_string(data.get("dataset"), DATASET, f"{relative}.dataset")
        _exact_string(data.get("dtype"), DTYPE, f"{relative}.dtype")
        _exact_string(data.get("hardware"), HARDWARE, f"{relative}.hardware")
        status = _nonempty_string(data.get("status"), f"{relative}.status")
        if status not in _ALL_STATUSES:
            raise ArtifactReportError(f"{relative}.status is unsupported: {status!r}")
        revision = _nonempty_string(
            data.get("tenon_revision"), f"{relative}.tenon_revision"
        )
        if _REVISION_RE.fullmatch(revision) is None:
            raise ArtifactReportError(f"{relative}.tenon_revision must be a Git SHA-1")
        statuses[status] += 1
        revisions.add(revision)

        result_content_fingerprint = _fingerprint(
            "canonical-apu-v1-result-json-v1", data
        )
        source = _result_source(
            corpus_root=corpus_root,
            result_path=result_path,
            data=data,
            result_content_fingerprint=result_content_fingerprint,
        )
        result_sources.append(source)

        result = data.get("result")
        if status == "NOT-APPLICABLE":
            if result is not None:
                raise ArtifactReportError(
                    f"{relative}.result must be null for NOT-APPLICABLE status"
                )
            not_applicable_sources.append(source)
            continue
        if result is None:
            raise ArtifactReportError(f"{relative}.result is missing measured evidence")
        result_mapping = _require_mapping(result, f"{relative}.result")
        _require_exact_fields(result_mapping, _RESULT_FIELDS, f"{relative}.result")
        _exact_string(
            result_mapping.get("metric"), RESULT_METRIC, f"{relative}.result.metric"
        )
        _exact_string(
            result_mapping.get("composition"),
            COMPOSITION,
            f"{relative}.result.composition",
        )
        _exact_string(
            result_mapping.get("execution_scope"),
            EXECUTION_SCOPE,
            f"{relative}.result.execution_scope",
        )
        result_cycles = _positive_int(
            result_mapping.get("cycles"), f"{relative}.result.cycles"
        )
        _validate_microseconds(
            result_cycles,
            result_mapping.get("microseconds"),
            f"{relative}.result.microseconds",
        )
        legs_raw = _require_sequence(
            result_mapping.get("legs"), f"{relative}.result.legs"
        )
        if not legs_raw:
            raise ArtifactReportError(f"{relative}.result.legs must not be empty")
        leg_cycles = 0
        for leg_index, leg_raw in enumerate(legs_raw):
            leg = _require_mapping(leg_raw, f"{relative}.result.legs[{leg_index}]")
            leg_observations, measured_cycles = _leg_evidence(
                corpus_root=corpus_root,
                result_path=result_path,
                result_data=data,
                result_source=source,
                result_content_fingerprint=result_content_fingerprint,
                leg=leg,
                leg_index=leg_index,
            )
            observations.extend(leg_observations)
            leg_cycles += measured_cycles
        if leg_cycles != result_cycles:
            raise ArtifactReportError(f"{relative}.result.cycles disagrees with legs")

    if not observations:
        raise ArtifactReportError("APUv1 Tenon corpus contains no measured profiles")
    observations.sort(
        key=lambda item: (
            item.category,
            item.semantic_shape,
            item.physical_schedule_json,
            str(item.source["provenance_fingerprint"]),
        )
    )
    result_sources.sort(key=lambda item: str(item["provenance_fingerprint"]))
    not_applicable_sources.sort(key=lambda item: str(item["provenance_fingerprint"]))
    return _CorpusEvidence(
        observations=tuple(observations),
        result_sources=tuple(result_sources),
        not_applicable_sources=tuple(not_applicable_sources),
        status_counts={
            status: statuses.get(status, 0) for status in sorted(_ALL_STATUSES)
        },
        tenon_revisions=tuple(sorted(revisions)),
    )


def load_apu_v1_profile_observations(
    artifacts_root: str | Path,
) -> tuple[_Observation, ...]:
    """Load validated APUv1 profile observations from the fixed corpus layout."""

    return _load_corpus(artifacts_root).observations


def _record_from_observations(
    observations: Sequence[_Observation],
) -> dict[str, object]:
    first = observations[0]
    semantic_shape = _shape_manifest(first.semantic_shape)
    historical_schedule = json.loads(first.physical_schedule_json)
    selected_materializations = [
        json.loads(value)
        for value in sorted(
            {observation.selected_materialization_json for observation in observations}
        )
    ]
    candidate_domain = json.loads(first.candidate_domain_json)
    semantic_shape_fingerprint = _fingerprint(
        "apu-v1-profile-semantic-shape-v1", semantic_shape
    )
    historical_schedule_fingerprint = _fingerprint(
        "apu-v1-opaque-historical-profile-schedule-v2", historical_schedule
    )
    historical_record_identity_fingerprint = _fingerprint(
        "apu-v1-profile-historical-record-identity-v2",
        {
            "semantic_shape": semantic_shape,
            "opaque_historical_schedule": historical_schedule,
            "selected_materializations": selected_materializations,
        },
    )

    unique_measurements: dict[str, int] = {}
    provenance: dict[str, Mapping[str, object]] = {}
    for observation in observations:
        unique_measurements.setdefault(
            observation.measurement_fingerprint, observation.profile_cycles
        )
        provenance_fingerprint = str(observation.source["provenance_fingerprint"])
        provenance.setdefault(provenance_fingerprint, observation.source)
    measurement_items = sorted(unique_measurements.items())
    measured_profile_cycles = float(median(cycles for _, cycles in measurement_items))
    repetitions = int(
        _require_mapping(
            historical_schedule["repetitions"],
            "opaque_historical_schedule.repetitions",
        )["composition_repeats"]
    )
    measured = {
        "profile_cycles": measured_profile_cycles,
        "composed_cycles": measured_profile_cycles * repetitions,
    }

    predicted: dict[str, int] | None = None
    pointwise_error: dict[str, float] | None = None
    if first.category == DENSE_SEARCHED:
        selected_fingerprint = candidate_domain["selected_candidate_fingerprint"]
        selected = next(
            candidate
            for candidate in candidate_domain["candidates"]
            if candidate["candidate_fingerprint"] == selected_fingerprint
        )
        estimate = selected["analytical_estimate"]
        predicted = {
            "profile_cycles": int(estimate["profile_cycles"]),
            "composed_cycles": int(estimate["composed_cycles"]),
        }
        signed_error = predicted["profile_cycles"] - measured_profile_cycles
        signed_relative = signed_error / measured_profile_cycles
        pointwise_error = {
            "signed_error_cycles": signed_error,
            "absolute_error_cycles": abs(signed_error),
            "signed_relative_error": signed_relative,
            "absolute_percentage_error": abs(signed_relative) * 100.0,
            "signed_composed_error_cycles": (
                predicted["composed_cycles"] - measured["composed_cycles"]
            ),
        }

    provenance_items = [
        dict(provenance[fingerprint]) for fingerprint in sorted(provenance)
    ]
    evidence_fingerprint = _fingerprint(
        "apu-v1-profile-evidence-v2",
        {
            "historical_record_identity_fingerprint": (
                historical_record_identity_fingerprint
            ),
            "candidate_domain": candidate_domain,
            "measurement_fingerprints": [key for key, _value in measurement_items],
            "provenance_fingerprints": sorted(provenance),
        },
    )
    return {
        "category": first.category,
        "semantic_shape": semantic_shape,
        "semantic_shape_fingerprint": semantic_shape_fingerprint,
        "opaque_historical_schedule": historical_schedule,
        "opaque_historical_schedule_fingerprint": historical_schedule_fingerprint,
        "historical_record_identity_fingerprint": (
            historical_record_identity_fingerprint
        ),
        "selected_materializations": selected_materializations,
        "candidate_domain": candidate_domain,
        "evidence_fingerprint": evidence_fingerprint,
        "measurement_aggregation": "median_of_unique_profile_cycle_measurements",
        "observation_count": len(provenance_items),
        "unique_measurement_count": len(measurement_items),
        "unique_measurements": [
            {
                "measurement_fingerprint": fingerprint,
                "profile_cycles": cycles,
                "composed_cycles": cycles * repetitions,
            }
            for fingerprint, cycles in measurement_items
        ],
        "measured": measured,
        "predicted": predicted,
        "pointwise_error": pointwise_error,
        "provenance": provenance_items,
    }


def _deduplicate_observations(
    observations: Sequence[_Observation],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, tuple[int, int, int], str], list[_Observation]] = (
        defaultdict(list)
    )
    for observation in observations:
        grouped[
            (
                observation.category,
                observation.semantic_shape,
                observation.physical_schedule_json,
            )
        ].append(observation)

    records = []
    for key in sorted(grouped):
        group = grouped[key]
        domains = {observation.candidate_domain_json for observation in group}
        if len(domains) != 1:
            raise ArtifactReportError(
                "duplicate semantic shape/physical schedule identity has mixed "
                "analytical candidate evidence"
            )
        records.append(_record_from_observations(group))
    fingerprints = [
        record["historical_record_identity_fingerprint"] for record in records
    ]
    if len(set(fingerprints)) != len(fingerprints):
        raise ArtifactReportError("historical record identity fingerprint collision")
    return tuple(records)


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
        raise ArtifactReportError(
            "selected-candidate cross-shape metrics require dense searched profiles"
        )
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
                "absolute_error_cycles"
            ]
        )
        for record in records
    ]
    count = len(records)
    return {
        "record_count": count,
        "signed_relative_bias": math.fsum(signed_relative) / count,
        "mean_absolute_percentage_error": math.fsum(absolute_percentage) / count,
        "median_absolute_percentage_error": float(median(absolute_percentage)),
        "p90_absolute_percentage_error": _percentile(absolute_percentage, 0.9),
        "maximum_absolute_percentage_error": max(absolute_percentage),
        "mean_absolute_error_cycles": math.fsum(absolute_error) / count,
        "root_mean_squared_error_cycles": math.sqrt(
            math.fsum(value * value for value in absolute_error) / count
        ),
    }


def build_apu_v1_profile_calibration_report(
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
) -> dict[str, object]:
    """Build deterministic content-addressed APUv1 profile calibration JSON."""

    corpus = _load_corpus(artifacts_root)
    records = _deduplicate_observations(corpus.observations)
    by_category = {
        category: [record for record in records if record["category"] == category]
        for category in _CATEGORIES
    }
    dense = by_category[DENSE_SEARCHED]
    if not dense:
        raise ArtifactReportError("corpus contains no dense searched profile domains")

    observation_counts = Counter(item.category for item in corpus.observations)
    unique_shape_counts = {
        category: len(
            {
                _canonical_json(record["semantic_shape"])
                for record in by_category[category]
            }
        )
        for category in _CATEGORIES
    }
    candidate_counts = [
        int(record["candidate_domain"]["candidate_count"]) for record in dense
    ]
    analytical_candidate_count = sum(candidate_counts)
    selected_candidate_count = len(dense)
    challenger_count = analytical_candidate_count - selected_candidate_count
    if challenger_count <= 0:
        raise ArtifactReportError("dense candidate domains contain no challengers")

    historical_record_identity_body = {
        category: [
            record["historical_record_identity_fingerprint"]
            for record in by_category[category]
        ]
        for category in _CATEGORIES
    }
    analytical_identity_body = [
        {
            "historical_record_identity_fingerprint": record[
                "historical_record_identity_fingerprint"
            ],
            "historical_candidate_domain_fingerprint": record["candidate_domain"][
                "historical_candidate_domain_fingerprint"
            ],
        }
        for record in dense
    ]
    dense_shape_count = unique_shape_counts[DENSE_SEARCHED]
    metrics = _pointwise_metrics(dense)

    body: dict[str, object] = {
        "schema_version": 2,
        "report_kind": "apu_v1_profile_calibration",
        "objective": {
            "metric": "device_compute_cycles",
            "unit": "cycles",
            "direction": "minimize",
            "scope": "single_APUC_exact_profile_composition",
        },
        "content_addressing": {
            "algorithm": "sha256",
            "canonical_json": "UTF-8 ASCII-safe sorted keys with compact separators",
            "identity_policy": (
                "semantic shapes plus opaque historical plan labels and exact "
                "selected MLIR/GVML hashes; challenger labels are not canonical "
                "physical manifests and are never promotion evidence"
            ),
            "historical_record_domain_fingerprint": _fingerprint(
                "apu-v1-profile-historical-record-domain-v2",
                historical_record_identity_body,
            ),
            "historical_analytical_domain_fingerprint": _fingerprint(
                "apu-v1-profile-historical-analytical-domain-v2",
                analytical_identity_body,
            ),
        },
        "evidence_domain": {
            "backend": BACKEND,
            "framework": FRAMEWORK,
            "dataset": DATASET,
            "dtype": DTYPE,
            "hardware": HARDWARE,
            "result_metric": RESULT_METRIC,
            "composition": COMPOSITION,
            "execution_scope": EXECUTION_SCOPE,
            "tenon_revisions": list(corpus.tenon_revisions),
        },
        "artifact_summary": {
            "source_result_record_count": len(corpus.result_sources),
            "measured_result_record_count": sum(
                corpus.status_counts[status] for status in _MEASURED_STATUSES
            ),
            "not_applicable_result_record_count": corpus.status_counts[
                "NOT-APPLICABLE"
            ],
            "result_status_counts": dict(corpus.status_counts),
            "profile_observation_count": len(corpus.observations),
            "unique_profile_schedule_count": len(records),
            "profile_observation_counts_by_category": {
                category: observation_counts.get(category, 0)
                for category in _CATEGORIES
            },
            "unique_profile_schedule_counts_by_category": {
                category: len(by_category[category]) for category in _CATEGORIES
            },
            "unique_semantic_shape_counts_by_category": unique_shape_counts,
            "unique_measurement_count": sum(
                int(record["unique_measurement_count"]) for record in records
            ),
        },
        "methodology": {
            "deduplication_key": (
                "semantic M/K/N plus historical profile M/K/N, tile_m, opaque plan "
                "label, repetitions, and exact selected MLIR/GVML hashes"
            ),
            "duplicate_measurement_policy": "exact profile-cycle measurements count once",
            "measurement_aggregation": "median_of_unique_profile_cycle_measurements",
            "partition_policy": {
                SINGLETON_GEMV_BYPASS: "semantic N == 1",
                REDUCTION_ONE_BYPASS: "semantic K == 1 and N > 1",
                DENSE_SEARCHED: "semantic N > 1 and K > 1",
            },
            "validation_scope": "selected_candidate_cross_shape_point_prediction",
            "limitations": [
                "Only selected dense candidates have device measurements.",
                "Analytical challengers contribute domain coverage but no ranking labels.",
                "Historical challenger plan labels are opaque: typed physical manifests and compiled sources are unavailable.",
                "Selected records bind exact compiled MLIR/GVML hashes but lack promotion-grade repeated measurements.",
                "Cross-shape point-prediction error does not establish within-shape ranking quality.",
                "Structurally bypassed singleton GEMV and K==1 profiles are excluded from prediction metrics.",
            ],
        },
        "profile_records": {
            category: by_category[category] for category in _CATEGORIES
        },
        "selected_candidate_cross_shape_prediction": {
            "status": "available",
            "error": metrics,
            "coverage": {
                "eligible_dense_profile_schedule_count": len(dense),
                "predicted_dense_profile_schedule_count": len(dense),
                "eligible_dense_semantic_shape_count": dense_shape_count,
                "predicted_dense_semantic_shape_count": dense_shape_count,
                "prediction_coverage": 1.0,
                "definition": (
                    "fraction of deduplicated dense semantic-shape/physical-schedule "
                    "records with a finite selected-candidate prediction"
                ),
            },
        },
        "candidate_domain_coverage": {
            "status": "available",
            "eligible_dense_profile_schedule_count": len(dense),
            "candidate_domain_count": len(dense),
            "candidate_domain_coverage": 1.0,
            "analytical_candidate_count": analytical_candidate_count,
            "selected_candidate_count": selected_candidate_count,
            "challenger_candidate_count": challenger_count,
            "measured_candidate_count": selected_candidate_count,
            "measured_challenger_candidate_count": 0,
            "candidate_measurement_coverage": (
                selected_candidate_count / analytical_candidate_count
            ),
            "challenger_measurement_coverage": 0.0,
            "candidates_per_domain": {
                "minimum": min(candidate_counts),
                "median": float(median(candidate_counts)),
                "maximum": max(candidate_counts),
            },
            "bypass_profile_schedule_count_excluded": (
                len(by_category[SINGLETON_GEMV_BYPASS])
                + len(by_category[REDUCTION_ONE_BYPASS])
            ),
        },
        "within_shape_candidate_ranking": {
            "status": "unavailable",
            "reason": (
                "challenger candidates have analytical estimates but no device measurements"
            ),
            "candidate_domain_count": len(dense),
            "analytical_candidate_count": analytical_candidate_count,
            "measured_selected_candidate_count": selected_candidate_count,
            "unmeasured_challenger_candidate_count": challenger_count,
            "metrics": None,
        },
        "not_applicable_result_provenance": [
            dict(source) for source in corpus.not_applicable_sources
        ],
    }
    return {
        **body,
        "report_fingerprint": _fingerprint(
            "apu-v1-profile-calibration-report-v2", body
        ),
    }


def report_to_json(report: Mapping[str, object]) -> str:
    """Serialize a calibration report deterministically."""

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
        description="Build a deterministic APUv1 profile-calibration report."
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
            build_apu_v1_profile_calibration_report(args.artifacts_root)
        )
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (ArtifactReportError, OSError) as error:
        print(f"FAIL APUv1 profile calibration: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
