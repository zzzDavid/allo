# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict uPIMulator counter ingestion and UPMEM cost-model calibration.

The archived UPMEM evaluations emit one counter file per one-DPU run.  Five
counters form the latency decomposition used here::

    Logic[0_0_0]_logic_cycle
      = ThreadScheduler[0_0_0]_breakdown_run
      + ThreadScheduler[0_0_0]_breakdown_etc
      + ThreadScheduler[0_0_0]_breakdown_dma
      + Logic[0_0_0]_backpressure

This module deliberately fails closed.  Missing, duplicate, malformed, or
inconsistent counters are rejected before a normalized candidate row can be
created.  Result and provenance JSON are parsed with duplicate-object-key
detection, checked against each other, and retained in the row for audit.

Only the Python standard library is used.  In particular, rank correlation is
implemented with average ranks for ties instead of depending on SciPy.  Top-1
regret is conservative under a prediction tie: it reports the worst measured
candidate among all candidates tied for the predicted minimum.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from statistics import median
from types import MappingProxyType
from typing import Any


COUNTER_SCHEMA = "upimulator-counter-decomposition-v1"
CANDIDATE_SCHEMA = "upmem-calibration-candidate-v1"
REPORT_SCHEMA = "upmem-calibration-report-v1"
DEFAULT_DPU_COORDINATE = "0_0_0"


class UPMEMCalibrationError(ValueError):
    """Base error for invalid UPMEM calibration evidence."""


class UPMEMCounterError(UPMEMCalibrationError):
    """Raised when a uPIMulator counter log is incomplete or inconsistent."""


class UPMEMMetadataError(UPMEMCalibrationError):
    """Raised when result or provenance metadata cannot be trusted."""


class UPMEMRankingError(UPMEMCalibrationError):
    """Raised when candidate rows cannot form a complete ranking problem."""


_COUNTER_LINE = re.compile(
    r"^\s*(?P<owner>[A-Za-z][A-Za-z0-9_]*)"
    r"\[(?P<coordinate>[^\]\s]+)\]_"
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*"
    r"(?P<value>\S(?:.*\S)?)\s*$"
)
_DECIMAL_COUNTER = re.compile(r"[0-9]+")
_REQUIRED_COUNTERS = {
    ("Logic", "logic_cycle"): "logic_cycle",
    ("ThreadScheduler", "breakdown_run"): "breakdown_run",
    ("ThreadScheduler", "breakdown_etc"): "breakdown_etc",
    ("ThreadScheduler", "breakdown_dma"): "breakdown_dma",
    ("Logic", "backpressure"): "backpressure",
}


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UPMEMCalibrationError(f"{field_name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UPMEMCalibrationError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    result = _require_nonnegative_int(value, field_name)
    if result == 0:
        raise UPMEMCalibrationError(f"{field_name} must be positive")
    return result


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UPMEMCalibrationError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise UPMEMCalibrationError(f"{field_name} must be finite")
    if positive and result <= 0.0:
        raise UPMEMCalibrationError(f"{field_name} must be positive")
    return result


@dataclass(frozen=True)
class UPMEMCounters:
    """The exact one-DPU uPIMulator latency decomposition."""

    logic_cycle: int
    breakdown_run: int
    breakdown_etc: int
    breakdown_dma: int
    backpressure: int
    dpu_coordinate: str = DEFAULT_DPU_COORDINATE

    def __post_init__(self) -> None:
        _require_nonempty_string(self.dpu_coordinate, "dpu_coordinate")
        _require_positive_int(self.logic_cycle, "logic_cycle")
        for name in (
            "breakdown_run",
            "breakdown_etc",
            "breakdown_dma",
            "backpressure",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        component_sum = self.component_sum
        if self.logic_cycle != component_sum:
            raise UPMEMCounterError(
                "inconsistent uPIMulator latency decomposition: "
                f"logic_cycle={self.logic_cycle}, but breakdown_run + "
                "breakdown_etc + breakdown_dma + backpressure="
                f"{component_sum}"
            )

    @property
    def component_sum(self) -> int:
        return (
            self.breakdown_run
            + self.breakdown_etc
            + self.breakdown_dma
            + self.backpressure
        )

    @property
    def logic_cycles(self) -> int:
        """Plural alias used by archived ``result.json`` files."""

        return self.logic_cycle

    @property
    def run_cycles(self) -> int:
        return self.breakdown_run

    @property
    def etc_cycles(self) -> int:
        return self.breakdown_etc

    @property
    def dma_cycles(self) -> int:
        return self.breakdown_dma

    @property
    def backpressure_cycles(self) -> int:
        return self.backpressure

    def manifest(self) -> dict[str, object]:
        return {
            "schema": COUNTER_SCHEMA,
            "dpu_coordinate": self.dpu_coordinate,
            "logic_cycle": self.logic_cycle,
            "breakdown_run": self.breakdown_run,
            "breakdown_etc": self.breakdown_etc,
            "breakdown_dma": self.breakdown_dma,
            "backpressure": self.backpressure,
            "decomposition_verified": True,
        }


def parse_upimulator_log(
    text: str,
    *,
    dpu_coordinate: str = DEFAULT_DPU_COORDINATE,
    reject_other_dpus: bool = True,
) -> UPMEMCounters:
    """Parse the archived uPIMulator key format and verify its decomposition.

    Unrelated uPIMulator counters are allowed.  Every required latency counter
    must occur exactly once for ``dpu_coordinate``.  By default, observing any
    of the five required counters for another coordinate is rejected because
    it would violate the one-DPU measurement contract used by the archive.
    """

    if not isinstance(text, str):
        raise TypeError("uPIMulator log text must be a string")
    _require_nonempty_string(dpu_coordinate, "dpu_coordinate")

    found: dict[str, tuple[int, int]] = {}
    other_dpus: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _COUNTER_LINE.match(line)
        if match is None:
            # A required-looking line must not degrade into a missing-counter
            # error merely because its value or syntax was malformed.
            if any(
                f"{owner}[{dpu_coordinate}]_{name}" in line
                for owner, name in _REQUIRED_COUNTERS
            ):
                raise UPMEMCounterError(
                    f"malformed required uPIMulator counter at line {line_number}: "
                    f"{line!r}"
                )
            continue

        identity = (match.group("owner"), match.group("name"))
        normalized_name = _REQUIRED_COUNTERS.get(identity)
        if normalized_name is None:
            continue
        coordinate = match.group("coordinate")
        if coordinate != dpu_coordinate:
            other_dpus.add(coordinate)
            continue
        value_text = match.group("value")
        if _DECIMAL_COUNTER.fullmatch(value_text) is None:
            raise UPMEMCounterError(
                f"counter {identity[0]}[{coordinate}]_{identity[1]} at line "
                f"{line_number} must be an unsigned decimal integer"
            )
        if normalized_name in found:
            previous_line = found[normalized_name][1]
            raise UPMEMCounterError(
                f"duplicate required counter {identity[0]}[{coordinate}]_"
                f"{identity[1]} at lines {previous_line} and {line_number}"
            )
        found[normalized_name] = (int(value_text), line_number)

    if reject_other_dpus and other_dpus:
        coordinates = ", ".join(sorted(other_dpus))
        raise UPMEMCounterError(
            "required latency counters were emitted for unexpected DPU "
            f"coordinate(s): {coordinates}; expected only {dpu_coordinate}"
        )

    missing = sorted(set(_REQUIRED_COUNTERS.values()) - set(found))
    if missing:
        raise UPMEMCounterError(
            "missing required uPIMulator counter(s): " + ", ".join(missing)
        )
    return UPMEMCounters(
        **{name: value for name, (value, _line) in found.items()},
        dpu_coordinate=dpu_coordinate,
    )


def parse_upimulator_log_file(
    path: str | Path,
    *,
    dpu_coordinate: str = DEFAULT_DPU_COORDINATE,
    reject_other_dpus: bool = True,
) -> UPMEMCounters:
    """Read and strictly parse one UTF-8 uPIMulator counter log."""

    log_path = Path(path)
    try:
        text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise UPMEMCounterError(
            f"cannot read uPIMulator counter log {log_path}: {error}"
        ) from error
    return parse_upimulator_log(
        text,
        dpu_coordinate=dpu_coordinate,
        reject_other_dpus=reject_other_dpus,
    )


# Common spelling for callers that regard this operation as loading evidence.
load_upimulator_counters = parse_upimulator_log_file


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UPMEMMetadataError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _freeze_json(value: object, path: str) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise UPMEMMetadataError(
                    f"{path} object keys must be non-empty strings"
                )
            frozen[key] = _freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UPMEMMetadataError(f"{path} must not contain NaN or infinity")
        return value
    raise UPMEMMetadataError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def parse_metadata_json(text: str, *, label: str = "metadata") -> Mapping[str, object]:
    """Parse a JSON object while rejecting duplicate keys and non-finite data."""

    if not isinstance(text, str):
        raise TypeError(f"{label} JSON text must be a string")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                UPMEMMetadataError(
                    f"{label} JSON contains non-finite constant {token!r}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise UPMEMMetadataError(f"cannot parse {label} JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise UPMEMMetadataError(f"{label} JSON root must be an object")
    return _freeze_json(value, label)  # type: ignore[return-value]


def load_metadata_json(
    path: str | Path,
    *,
    label: str = "metadata",
) -> Mapping[str, object]:
    """Read a UTF-8 metadata JSON file with duplicate-key detection."""

    metadata_path = Path(path)
    try:
        text = metadata_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise UPMEMMetadataError(
            f"cannot read {label} JSON {metadata_path}: {error}"
        ) from error
    return parse_metadata_json(text, label=label)


def _metadata_string(metadata: Mapping[str, object], name: str, label: str) -> str:
    try:
        value = metadata[name]
    except KeyError as error:
        raise UPMEMMetadataError(
            f"{label} is missing required field {name!r}"
        ) from error
    try:
        return _require_nonempty_string(value, f"{label}.{name}")
    except UPMEMCalibrationError as error:
        raise UPMEMMetadataError(str(error)) from error


def _metadata_positive_int(
    metadata: Mapping[str, object], name: str, label: str
) -> int:
    try:
        value = metadata[name]
    except KeyError as error:
        raise UPMEMMetadataError(
            f"{label} is missing required field {name!r}"
        ) from error
    try:
        return _require_positive_int(value, f"{label}.{name}")
    except UPMEMCalibrationError as error:
        raise UPMEMMetadataError(str(error)) from error


def validate_result_metadata(
    metadata: Mapping[str, object],
    *,
    counters: UPMEMCounters | None = None,
) -> Mapping[str, object]:
    """Validate and own the archived ``result.json`` metadata."""

    if not isinstance(metadata, Mapping):
        raise UPMEMMetadataError("result metadata must be an object")
    frozen = _freeze_json(metadata, "result")
    assert isinstance(frozen, Mapping)

    for name in ("schema", "compiler", "workload", "simulator"):
        _metadata_string(frozen, name, "result")
    if _metadata_string(frozen, "status", "result") != "pass":
        raise UPMEMMetadataError("result.status must be 'pass'")
    logic_cycles = _metadata_positive_int(frozen, "logic_cycles", "result")
    num_dpus = _metadata_positive_int(frozen, "num_dpus", "result")
    _metadata_positive_int(frozen, "num_tasklets", "result")
    _metadata_positive_int(frozen, "executions", "result")
    if num_dpus != 1:
        raise UPMEMMetadataError(
            "result.num_dpus must be 1 for the archived one-DPU counter contract"
        )

    try:
        frequency = _require_finite_number(
            frozen["logic_frequency_mhz"],
            "result.logic_frequency_mhz",
            positive=True,
        )
    except KeyError as error:
        raise UPMEMMetadataError(
            "result is missing required field 'logic_frequency_mhz'"
        ) from error
    except UPMEMCalibrationError as error:
        raise UPMEMMetadataError(str(error)) from error

    for exit_field in ("compile_exit_code", "simulation_exit_code"):
        if exit_field in frozen and frozen[exit_field] != 0:
            raise UPMEMMetadataError(f"result.{exit_field} must be zero")

    if counters is not None and logic_cycles != counters.logic_cycle:
        raise UPMEMMetadataError(
            "result.logic_cycles does not match the uPIMulator log: "
            f"{logic_cycles} != {counters.logic_cycle}"
        )

    if "predicted_kernel_ms" in frozen:
        try:
            reported_ms = _require_finite_number(
                frozen["predicted_kernel_ms"],
                "result.predicted_kernel_ms",
                positive=True,
            )
        except UPMEMCalibrationError as error:
            raise UPMEMMetadataError(str(error)) from error
        expected_ms = logic_cycles / (frequency * 1000.0)
        if not math.isclose(reported_ms, expected_ms, rel_tol=0.0, abs_tol=1e-12):
            raise UPMEMMetadataError(
                "result.predicted_kernel_ms is inconsistent with logic_cycles "
                "and logic_frequency_mhz"
            )

    oracle = frozen.get("oracle_validation")
    if oracle is not None:
        if not isinstance(oracle, Mapping) or oracle.get("status") != "pass":
            raise UPMEMMetadataError(
                "result.oracle_validation, when present, must report status 'pass'"
            )
    return frozen


def parse_result_metadata(
    text: str,
    *,
    counters: UPMEMCounters | None = None,
) -> Mapping[str, object]:
    """Parse and validate archived result JSON text."""

    return validate_result_metadata(
        parse_metadata_json(text, label="result"), counters=counters
    )


def load_result_metadata(
    path: str | Path,
    *,
    counters: UPMEMCounters | None = None,
) -> Mapping[str, object]:
    """Read and validate an archived ``result.json`` file."""

    return validate_result_metadata(
        load_metadata_json(path, label="result"), counters=counters
    )


def validate_provenance_metadata(
    metadata: Mapping[str, object],
    *,
    result_metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate compiler provenance and its agreement with a result record."""

    if not isinstance(metadata, Mapping):
        raise UPMEMMetadataError("provenance metadata must be an object")
    frozen = _freeze_json(metadata, "provenance")
    assert isinstance(frozen, Mapping)
    compiler = _metadata_string(frozen, "compiler", "provenance")
    workload = _metadata_string(frozen, "workload", "provenance")

    if result_metadata is not None:
        result = validate_result_metadata(result_metadata)
        result_compiler = _metadata_string(result, "compiler", "result")
        if compiler.casefold() != result_compiler.casefold():
            raise UPMEMMetadataError(
                "result and provenance compiler fields disagree: "
                f"{result_compiler!r} != {compiler!r}"
            )
        result_workload = _metadata_string(result, "workload", "result")
        # CINM records use ``null`` for unsplit workloads and a string only for
        # logical subruns such as ``gemv-dot``.  A null subrun therefore falls
        # back to the workload instead of becoming an identity itself.
        provenance_subrun = frozen.get("subrun")
        provenance_case = (
            provenance_subrun
            if isinstance(provenance_subrun, str) and provenance_subrun
            else workload
        )
        if provenance_case != result_workload:
            raise UPMEMMetadataError(
                "result workload does not match provenance workload/subrun: "
                f"{result_workload!r} != {provenance_case!r}"
            )
        comparable_fields = (
            ("num_dpus", "num_dpus"),
            ("num_tasklets", "num_tasklets"),
            ("num_executions", "executions"),
        )
        for provenance_name, result_name in comparable_fields:
            if (
                provenance_name in frozen
                and frozen[provenance_name] != result[result_name]
            ):
                raise UPMEMMetadataError(
                    f"provenance.{provenance_name} does not match "
                    f"result.{result_name}"
                )
    return frozen


def parse_provenance_metadata(
    text: str,
    *,
    result_metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Parse and validate archived provenance JSON text."""

    return validate_provenance_metadata(
        parse_metadata_json(text, label="provenance"),
        result_metadata=result_metadata,
    )


def load_provenance_metadata(
    path: str | Path,
    *,
    result_metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Read and validate an archived ``provenance.json`` file."""

    return validate_provenance_metadata(
        load_metadata_json(path, label="provenance"),
        result_metadata=result_metadata,
    )


def _freeze_named_fields(
    fields: Mapping[str, object] | None,
    label: str,
) -> Mapping[str, object]:
    if fields is None:
        return MappingProxyType({})
    if not isinstance(fields, Mapping):
        raise UPMEMMetadataError(f"{label} must be an object")
    frozen = _freeze_json(fields, label)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class UPMEMCandidateRow:
    """One normalized, auditable simulator measurement and model prediction."""

    candidate_set: str
    candidate_id: str
    counters: UPMEMCounters
    result_metadata: Mapping[str, object]
    provenance_metadata: Mapping[str, object]
    predicted_logic_cycle: float | None = None
    cost_features: Mapping[str, object] = field(default_factory=dict)
    prediction_fields: Mapping[str, object] = field(default_factory=dict)
    artifact_paths: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_set, "candidate_set")
        _require_nonempty_string(self.candidate_id, "candidate_id")
        if not isinstance(self.counters, UPMEMCounters):
            raise TypeError("counters must be a UPMEMCounters instance")

        result = validate_result_metadata(
            self.result_metadata,
            counters=self.counters,
        )
        provenance = validate_provenance_metadata(
            self.provenance_metadata,
            result_metadata=result,
        )
        cost_features = _freeze_named_fields(self.cost_features, "cost_features")
        prediction_fields = _freeze_named_fields(
            self.prediction_fields, "prediction_fields"
        )
        artifact_paths = _freeze_named_fields(self.artifact_paths, "artifact_paths")
        for name, value in artifact_paths.items():
            if not isinstance(value, str) or not value:
                raise UPMEMMetadataError(
                    f"artifact_paths.{name} must be a non-empty path string"
                )

        prediction = self.predicted_logic_cycle
        aliases = (
            "predicted_logic_cycle",
            "predicted_logic_cycles",
            "predicted_cycles",
            "logic_cycle",
        )
        embedded = [
            prediction_fields[name] for name in aliases if name in prediction_fields
        ]
        if prediction is None and embedded:
            prediction = embedded[0]  # type: ignore[assignment]
        if prediction is not None:
            prediction = _require_finite_number(
                prediction, "predicted_logic_cycle", positive=True
            )
            for value in embedded:
                embedded_prediction = _require_finite_number(
                    value, "prediction_fields logic-cycle prediction", positive=True
                )
                if embedded_prediction != prediction:
                    raise UPMEMMetadataError(
                        "prediction_fields logic-cycle value disagrees with "
                        "predicted_logic_cycle"
                    )

        object.__setattr__(self, "result_metadata", result)
        object.__setattr__(self, "provenance_metadata", provenance)
        object.__setattr__(self, "cost_features", cost_features)
        object.__setattr__(self, "prediction_fields", prediction_fields)
        object.__setattr__(self, "artifact_paths", artifact_paths)
        object.__setattr__(self, "predicted_logic_cycle", prediction)

    @property
    def benchmark(self) -> str:
        """Alias emphasizing the usual meaning of ``candidate_set``."""

        return self.candidate_set

    @property
    def measured_logic_cycle(self) -> int:
        return self.counters.logic_cycle

    @property
    def measured_cycles(self) -> int:
        return self.counters.logic_cycle

    @property
    def predicted_cycles(self) -> float | None:
        return self.predicted_logic_cycle

    @property
    def compiler(self) -> str:
        return str(self.result_metadata["compiler"])

    @property
    def compiler_revision(self) -> str | None:
        for name in (
            "compiler_commit",
            "tenon_commit",
            "source_revision",
            "revision",
        ):
            value = self.provenance_metadata.get(name)
            if isinstance(value, str) and value:
                return value
        return None

    def manifest(self) -> dict[str, object]:
        """Return the stable, flat normalized candidate-row schema."""

        return {
            "schema": CANDIDATE_SCHEMA,
            "candidate_set": self.candidate_set,
            "candidate_id": self.candidate_id,
            "logic_cycle": self.counters.logic_cycle,
            "breakdown_run": self.counters.breakdown_run,
            "breakdown_etc": self.counters.breakdown_etc,
            "breakdown_dma": self.counters.breakdown_dma,
            "backpressure": self.counters.backpressure,
            "dpu_coordinate": self.counters.dpu_coordinate,
            "decomposition_verified": True,
            "predicted_logic_cycle": self.predicted_logic_cycle,
            "cost_features": _thaw_json(self.cost_features),
            "prediction_fields": _thaw_json(self.prediction_fields),
            "result_metadata": _thaw_json(self.result_metadata),
            "provenance_metadata": _thaw_json(self.provenance_metadata),
            "artifact_paths": _thaw_json(self.artifact_paths),
        }


# Shorter public alias for callers that already operate in a UPMEM namespace.
CandidateRow = UPMEMCandidateRow


def _coerce_metadata(
    value: Mapping[str, object] | str,
    *,
    label: str,
) -> Mapping[str, object]:
    if isinstance(value, str):
        return parse_metadata_json(value, label=label)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} metadata must be a mapping or JSON text")
    return value


def build_candidate_row(
    *,
    candidate_set: str,
    candidate_id: str,
    log_text: str,
    result_metadata: Mapping[str, object] | str,
    provenance_metadata: Mapping[str, object] | str,
    predicted_logic_cycle: float | None = None,
    cost_features: Mapping[str, object] | None = None,
    prediction_fields: Mapping[str, object] | None = None,
    artifact_paths: Mapping[str, object] | None = None,
    dpu_coordinate: str = DEFAULT_DPU_COORDINATE,
) -> UPMEMCandidateRow:
    """Build one normalized row from in-memory log and metadata evidence."""

    counters = parse_upimulator_log(log_text, dpu_coordinate=dpu_coordinate)
    result = validate_result_metadata(
        _coerce_metadata(result_metadata, label="result"), counters=counters
    )
    provenance = validate_provenance_metadata(
        _coerce_metadata(provenance_metadata, label="provenance"),
        result_metadata=result,
    )
    return UPMEMCandidateRow(
        candidate_set=candidate_set,
        candidate_id=candidate_id,
        counters=counters,
        result_metadata=result,
        provenance_metadata=provenance,
        predicted_logic_cycle=predicted_logic_cycle,
        cost_features=cost_features or {},
        prediction_fields=prediction_fields or {},
        artifact_paths=artifact_paths or {},
    )


def load_candidate_row(
    *,
    log_path: str | Path,
    result_path: str | Path,
    provenance_path: str | Path,
    candidate_set: str | None = None,
    candidate_id: str | None = None,
    predicted_logic_cycle: float | None = None,
    cost_features: Mapping[str, object] | None = None,
    prediction_fields: Mapping[str, object] | None = None,
    dpu_coordinate: str = DEFAULT_DPU_COORDINATE,
) -> UPMEMCandidateRow:
    """Load one candidate from archived log/result/provenance paths."""

    counters = parse_upimulator_log_file(log_path, dpu_coordinate=dpu_coordinate)
    result = load_result_metadata(result_path, counters=counters)
    provenance = load_provenance_metadata(provenance_path, result_metadata=result)
    normalized_set = candidate_set or str(result["workload"])
    normalized_id = candidate_id or str(result["compiler"])
    return UPMEMCandidateRow(
        candidate_set=normalized_set,
        candidate_id=normalized_id,
        counters=counters,
        result_metadata=result,
        provenance_metadata=provenance,
        predicted_logic_cycle=predicted_logic_cycle,
        cost_features=cost_features or {},
        prediction_fields=prediction_fields or {},
        artifact_paths={
            "counter_log": str(Path(log_path)),
            "result": str(Path(result_path)),
            "provenance": str(Path(provenance_path)),
        },
    )


candidate_row_from_artifacts = load_candidate_row


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return tuple(ranks)


def spearman_rank_correlation(
    measured: Sequence[Real],
    predicted: Sequence[Real],
) -> float | None:
    """Return tie-aware Spearman rho, or ``None`` for an undefined ranking."""

    if isinstance(measured, (str, bytes)) or isinstance(predicted, (str, bytes)):
        raise UPMEMRankingError("rank values must be numeric sequences")
    if len(measured) != len(predicted):
        raise UPMEMRankingError("measured and predicted ranks must have equal length")
    if len(measured) < 2:
        return None
    measured_values = tuple(
        _require_finite_number(value, f"measured[{index}]")
        for index, value in enumerate(measured)
    )
    predicted_values = tuple(
        _require_finite_number(value, f"predicted[{index}]")
        for index, value in enumerate(predicted)
    )
    measured_ranks = _average_ranks(measured_values)
    predicted_ranks = _average_ranks(predicted_values)
    measured_mean = sum(measured_ranks) / len(measured_ranks)
    predicted_mean = sum(predicted_ranks) / len(predicted_ranks)
    numerator = sum(
        (left - measured_mean) * (right - predicted_mean)
        for left, right in zip(measured_ranks, predicted_ranks)
    )
    left_variance = sum((value - measured_mean) ** 2 for value in measured_ranks)
    right_variance = sum((value - predicted_mean) ** 2 for value in predicted_ranks)
    if left_variance == 0.0 or right_variance == 0.0:
        return None
    return numerator / math.sqrt(left_variance * right_variance)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise UPMEMRankingError("cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class UPMEMRankingMetrics:
    """Prediction accuracy and ranking quality for one candidate set."""

    candidate_count: int
    spearman_rank_correlation: float | None
    top_1_regret_cycles: float
    top_1_regret_percent: float
    signed_bias_cycles: float
    signed_relative_bias_percent: float
    median_absolute_percentage_error: float
    p90_absolute_percentage_error: float
    max_absolute_percentage_error: float
    measured_top_candidate_ids: tuple[str, ...] = ()
    predicted_top_candidate_ids: tuple[str, ...] = ()

    @property
    def spearman_rho(self) -> float | None:
        return self.spearman_rank_correlation

    @property
    def top1_regret(self) -> float:
        return self.top_1_regret_cycles

    @property
    def signed_bias(self) -> float:
        return self.signed_bias_cycles

    @property
    def median_ape(self) -> float:
        return self.median_absolute_percentage_error

    @property
    def p90_ape(self) -> float:
        return self.p90_absolute_percentage_error

    @property
    def max_ape(self) -> float:
        return self.max_absolute_percentage_error

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "spearman_rank_correlation": self.spearman_rank_correlation,
            "top_1_regret_cycles": self.top_1_regret_cycles,
            "top_1_regret_percent": self.top_1_regret_percent,
            "top_1_tie_policy": "worst_measured_member_of_predicted_minimum_tie",
            "signed_bias_cycles": self.signed_bias_cycles,
            "signed_relative_bias_percent": self.signed_relative_bias_percent,
            "median_absolute_percentage_error": self.median_absolute_percentage_error,
            "p90_absolute_percentage_error": self.p90_absolute_percentage_error,
            "max_absolute_percentage_error": self.max_absolute_percentage_error,
            "measured_top_candidate_ids": list(self.measured_top_candidate_ids),
            "predicted_top_candidate_ids": list(self.predicted_top_candidate_ids),
        }


RankingMetrics = UPMEMRankingMetrics


def _snapshot_rows(
    rows: Iterable[UPMEMCandidateRow],
    *,
    require_one_set: bool,
) -> tuple[UPMEMCandidateRow, ...]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise UPMEMRankingError("candidate rows must be an iterable of rows")
    try:
        snapshot = tuple(rows)
    except TypeError as error:
        raise UPMEMRankingError("candidate rows must be an iterable of rows") from error
    if not snapshot:
        raise UPMEMRankingError("candidate rows must not be empty")
    for index, row in enumerate(snapshot):
        if not isinstance(row, UPMEMCandidateRow):
            raise UPMEMRankingError(f"candidate row {index} is not a UPMEMCandidateRow")
        if row.predicted_logic_cycle is None:
            raise UPMEMRankingError(
                f"candidate {row.candidate_id!r} is missing predicted_logic_cycle"
            )
    identities = [(row.candidate_set, row.candidate_id) for row in snapshot]
    if len(set(identities)) != len(identities):
        raise UPMEMRankingError("candidate IDs must be unique within each set")
    if require_one_set and len({row.candidate_set for row in snapshot}) != 1:
        raise UPMEMRankingError(
            "ranking metrics require rows from exactly one candidate set"
        )
    return tuple(
        sorted(snapshot, key=lambda row: (row.candidate_set, row.candidate_id))
    )


def compute_ranking_metrics(
    rows: Iterable[UPMEMCandidateRow],
) -> UPMEMRankingMetrics:
    """Compute tie-aware rank, regret, bias, and APE metrics for one set."""

    candidates = _snapshot_rows(rows, require_one_set=True)
    measured = tuple(float(row.measured_logic_cycle) for row in candidates)
    predicted = tuple(float(row.predicted_logic_cycle) for row in candidates)
    measured_best = min(measured)
    predicted_best = min(predicted)
    measured_top_ids = tuple(
        row.candidate_id
        for row, value in zip(candidates, measured)
        if value == measured_best
    )
    predicted_top_indices = tuple(
        index for index, value in enumerate(predicted) if value == predicted_best
    )
    predicted_top_ids = tuple(
        candidates[index].candidate_id for index in predicted_top_indices
    )
    # A scheduler with no secondary discrimination may select any tied member;
    # use the worst member so zero regret is a genuine guarantee.
    predicted_top_worst_measured = max(
        measured[index] for index in predicted_top_indices
    )
    regret = predicted_top_worst_measured - measured_best
    errors = tuple(
        prediction - observation for prediction, observation in zip(predicted, measured)
    )
    relative_errors = tuple(
        error / observation for error, observation in zip(errors, measured)
    )
    absolute_percentage_errors = tuple(abs(value) * 100.0 for value in relative_errors)
    return UPMEMRankingMetrics(
        candidate_count=len(candidates),
        spearman_rank_correlation=spearman_rank_correlation(measured, predicted),
        top_1_regret_cycles=regret,
        top_1_regret_percent=regret / measured_best * 100.0,
        signed_bias_cycles=sum(errors) / len(errors),
        signed_relative_bias_percent=sum(relative_errors)
        / len(relative_errors)
        * 100.0,
        median_absolute_percentage_error=median(absolute_percentage_errors),
        p90_absolute_percentage_error=_percentile(absolute_percentage_errors, 0.9),
        max_absolute_percentage_error=max(absolute_percentage_errors),
        measured_top_candidate_ids=measured_top_ids,
        predicted_top_candidate_ids=predicted_top_ids,
    )


@dataclass(frozen=True)
class UPMEMCandidateSetReport:
    candidate_set: str
    candidates: tuple[UPMEMCandidateRow, ...]
    metrics: UPMEMRankingMetrics

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_set": self.candidate_set,
            "metrics": self.metrics.manifest(),
            "candidates": [candidate.manifest() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class UPMEMAggregateMetrics:
    candidate_set_count: int
    candidate_count: int
    rank_evaluable_candidate_set_count: int
    spearman_rank_correlation: float | None
    mean_top_1_regret_cycles: float
    mean_top_1_regret_percent: float
    signed_bias_cycles: float
    signed_relative_bias_percent: float
    median_absolute_percentage_error: float
    p90_absolute_percentage_error: float
    max_absolute_percentage_error: float

    @property
    def spearman_rho(self) -> float | None:
        return self.spearman_rank_correlation

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_set_count": self.candidate_set_count,
            "candidate_count": self.candidate_count,
            "rank_evaluable_candidate_set_count": self.rank_evaluable_candidate_set_count,
            "spearman_rank_correlation_macro_average": self.spearman_rank_correlation,
            "mean_top_1_regret_cycles": self.mean_top_1_regret_cycles,
            "mean_top_1_regret_percent": self.mean_top_1_regret_percent,
            "signed_bias_cycles": self.signed_bias_cycles,
            "signed_relative_bias_percent": self.signed_relative_bias_percent,
            "median_absolute_percentage_error": self.median_absolute_percentage_error,
            "p90_absolute_percentage_error": self.p90_absolute_percentage_error,
            "max_absolute_percentage_error": self.max_absolute_percentage_error,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class UPMEMCalibrationReport:
    """Deterministic report over one or more independent candidate sets."""

    candidate_sets: tuple[UPMEMCandidateSetReport, ...]
    aggregate_metrics: UPMEMAggregateMetrics
    report_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_metadata",
            _freeze_named_fields(self.report_metadata, "report_metadata"),
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "metric": "uPIMulator one-DPU cumulative logic_cycle",
            "objective": "minimize",
            "rank_correlation_aggregation": "macro_average_over_defined_candidate_sets",
            "error_aggregation": "micro_average_over_candidates",
            "candidate_sets": [
                candidate_set.manifest() for candidate_set in self.candidate_sets
            ],
            "aggregate_metrics": self.aggregate_metrics.manifest(),
            "report_metadata": _thaw_json(self.report_metadata),
        }

    @property
    def report_fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self._body()).encode("ascii")).hexdigest()

    def manifest(self) -> dict[str, object]:
        body = self._body()
        body["report_fingerprint"] = self.report_fingerprint
        return body

    def to_json(self, *, indent: int = 2) -> str:
        return (
            json.dumps(
                self.manifest(),
                indent=indent,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        lines = [
            "# UPMEM calibration report",
            "",
            f"Report fingerprint: `{self.report_fingerprint}`",
            "",
            "All measurements are cumulative one-DPU uPIMulator `logic_cycle` "
            "counts. Lower is better. Spearman correlation uses average ranks "
            "for ties; top-1 regret uses the worst measured member of a predicted "
            "minimum tie.",
            "",
            "## Summary",
            "",
            "| Candidate set | Candidates | Spearman rho | Top-1 regret (cycles) | "
            "Top-1 regret (%) | Median APE (%) | P90 APE (%) | Max APE (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for candidate_set in self.candidate_sets:
            metrics = candidate_set.metrics
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(candidate_set.candidate_set),
                        str(metrics.candidate_count),
                        _format_metric(metrics.spearman_rank_correlation),
                        _format_metric(metrics.top_1_regret_cycles),
                        _format_metric(metrics.top_1_regret_percent),
                        _format_metric(metrics.median_absolute_percentage_error),
                        _format_metric(metrics.p90_absolute_percentage_error),
                        _format_metric(metrics.max_absolute_percentage_error),
                    )
                )
                + " |"
            )

        aggregate = self.aggregate_metrics
        lines.extend(
            (
                "",
                "Aggregate Spearman rho (macro): "
                f"{_format_metric(aggregate.spearman_rank_correlation)}. "
                "Signed relative bias (micro): "
                f"{_format_metric(aggregate.signed_relative_bias_percent)}%.",
            )
        )
        for candidate_set in self.candidate_sets:
            lines.extend(
                (
                    "",
                    f"## {_markdown_heading(candidate_set.candidate_set)}",
                    "",
                    "| Candidate | Measured logic_cycle | Predicted logic_cycle | "
                    "Run | Etc | DMA | Backpressure | APE (%) |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                )
            )
            for candidate in candidate_set.candidates:
                predicted = float(candidate.predicted_logic_cycle)
                ape = (
                    abs(predicted - candidate.measured_logic_cycle)
                    / candidate.measured_logic_cycle
                    * 100.0
                )
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown_cell(candidate.candidate_id),
                            str(candidate.measured_logic_cycle),
                            _format_metric(predicted),
                            str(candidate.counters.breakdown_run),
                            str(candidate.counters.breakdown_etc),
                            str(candidate.counters.breakdown_dma),
                            str(candidate.counters.backpressure),
                            _format_metric(ape),
                        )
                    )
                    + " |"
                )
        return "\n".join(lines) + "\n"


CalibrationReport = UPMEMCalibrationReport


def _aggregate_metrics(
    candidate_sets: Sequence[UPMEMCandidateSetReport],
) -> UPMEMAggregateMetrics:
    rows = tuple(
        candidate
        for candidate_set in candidate_sets
        for candidate in candidate_set.candidates
    )
    errors = tuple(
        float(row.predicted_logic_cycle) - row.measured_logic_cycle for row in rows
    )
    relative_errors = tuple(
        error / row.measured_logic_cycle for error, row in zip(errors, rows)
    )
    apes = tuple(abs(error) * 100.0 for error in relative_errors)
    correlations = tuple(
        candidate_set.metrics.spearman_rank_correlation
        for candidate_set in candidate_sets
        if candidate_set.metrics.spearman_rank_correlation is not None
    )
    return UPMEMAggregateMetrics(
        candidate_set_count=len(candidate_sets),
        candidate_count=len(rows),
        rank_evaluable_candidate_set_count=len(correlations),
        spearman_rank_correlation=(
            sum(correlations) / len(correlations) if correlations else None
        ),
        mean_top_1_regret_cycles=sum(
            candidate_set.metrics.top_1_regret_cycles
            for candidate_set in candidate_sets
        )
        / len(candidate_sets),
        mean_top_1_regret_percent=sum(
            candidate_set.metrics.top_1_regret_percent
            for candidate_set in candidate_sets
        )
        / len(candidate_sets),
        signed_bias_cycles=sum(errors) / len(errors),
        signed_relative_bias_percent=sum(relative_errors)
        / len(relative_errors)
        * 100.0,
        median_absolute_percentage_error=median(apes),
        p90_absolute_percentage_error=_percentile(apes, 0.9),
        max_absolute_percentage_error=max(apes),
    )


def build_calibration_report(
    rows: Iterable[UPMEMCandidateRow],
    *,
    report_metadata: Mapping[str, object] | None = None,
) -> UPMEMCalibrationReport:
    """Group rows into candidate sets and build deterministic report output."""

    snapshot = _snapshot_rows(rows, require_one_set=False)
    grouped: dict[str, list[UPMEMCandidateRow]] = defaultdict(list)
    for row in snapshot:
        grouped[row.candidate_set].append(row)
    candidate_sets = tuple(
        UPMEMCandidateSetReport(
            candidate_set=name,
            candidates=tuple(grouped[name]),
            metrics=compute_ranking_metrics(grouped[name]),
        )
        for name in sorted(grouped)
    )
    return UPMEMCalibrationReport(
        candidate_sets=candidate_sets,
        aggregate_metrics=_aggregate_metrics(candidate_sets),
        report_metadata=report_metadata or {},
    )


def report_to_json(report: UPMEMCalibrationReport, *, indent: int = 2) -> str:
    if not isinstance(report, UPMEMCalibrationReport):
        raise TypeError("report must be a UPMEMCalibrationReport")
    return report.to_json(indent=indent)


def report_to_markdown(report: UPMEMCalibrationReport) -> str:
    if not isinstance(report, UPMEMCalibrationReport):
        raise TypeError("report must be a UPMEMCalibrationReport")
    return report.to_markdown()


def _format_metric(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6g}"


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _markdown_heading(value: str) -> str:
    return value.replace("\n", " ").strip()


__all__ = [
    "COUNTER_SCHEMA",
    "CANDIDATE_SCHEMA",
    "REPORT_SCHEMA",
    "DEFAULT_DPU_COORDINATE",
    "UPMEMCalibrationError",
    "UPMEMCounterError",
    "UPMEMMetadataError",
    "UPMEMRankingError",
    "UPMEMCounters",
    "UPMEMCandidateRow",
    "CandidateRow",
    "UPMEMRankingMetrics",
    "RankingMetrics",
    "UPMEMCandidateSetReport",
    "UPMEMAggregateMetrics",
    "UPMEMCalibrationReport",
    "CalibrationReport",
    "parse_upimulator_log",
    "parse_upimulator_log_file",
    "load_upimulator_counters",
    "parse_metadata_json",
    "load_metadata_json",
    "validate_result_metadata",
    "parse_result_metadata",
    "load_result_metadata",
    "validate_provenance_metadata",
    "parse_provenance_metadata",
    "load_provenance_metadata",
    "build_candidate_row",
    "load_candidate_row",
    "candidate_row_from_artifacts",
    "spearman_rank_correlation",
    "compute_ranking_metrics",
    "build_calibration_report",
    "report_to_json",
    "report_to_markdown",
]
