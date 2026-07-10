#!/usr/bin/env python3
"""Build immutable shape-stratified evaluations from candidate sweep records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from statistics import median
from typing import Any

from allo.pim.calibration_report import (
    CalibrationDataError,
    CalibrationObjectiveDomain,
    CalibrationReport,
    CalibrationSample,
    build_frozen_model_shape_stratified_report,
)


class CandidateRankingReportError(ValueError):
    """Raised when sweep records cannot support a trustworthy evaluation."""


def _load_helper(module_name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import measurement helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


APUV1_MEASURER = _load_helper(
    "tenon_measure_apu_v1_profile_candidates",
    "measure_apu_v1_profile_candidates.py",
)
APUG2_MEASURER = _load_helper(
    "tenon_measure_apu_g2_persistent_candidates",
    "measure_apu_g2_persistent_candidates.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(namespace: str, value: object) -> str:
    return hashlib.sha256(
        _canonical_json({"namespace": namespace, "value": value}).encode("ascii")
    ).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateRankingReportError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CandidateRankingReportError(f"{field} must be an array")
    return value


def _exact_int(value: object, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value != expected:
        raise CandidateRankingReportError(f"{field} must equal {expected}")


def _platform_contract(report: Mapping[str, Any], *, backend: str) -> dict[str, str]:
    provenance = _mapping(report.get("provenance"), "provenance")
    value = provenance.get("platform_contract")
    if value is None:
        raise CandidateRankingReportError(
            "candidate ranking refuses unverifiable hardware platforms"
        )
    contract = _mapping(value, "provenance.platform_contract")
    if set(contract) != {"schema", "target", "hardware_family", "fingerprint"}:
        raise CandidateRankingReportError("platform contract is malformed")
    expected = {
        "apu_v1": ("apu_v1", "gemini-i"),
        "apu_g2": ("apu_v2", "gemini-ii"),
    }[backend]
    fingerprint = contract.get("fingerprint")
    if (
        contract.get("schema") != "tenon-promotion-platform-v1"
        or (contract.get("target"), contract.get("hardware_family")) != expected
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise CandidateRankingReportError("platform contract is malformed")
    return dict(contract)


def _require_apu_v1_complete_coverage(
    report: Mapping[str, Any], candidates: Sequence[object]
) -> None:
    domain = _mapping(report.get("domain"), "domain")
    rejections = _sequence(report.get("rejections"), "rejections")
    if rejections:
        raise CandidateRankingReportError(
            "APUv1 ranking requires zero candidate rejections"
        )
    if (
        domain.get("candidate_domain_complete") is not True
        or domain.get("row_tile_domain_complete") is not True
        or domain.get("physical_plan_domain_complete") is not True
        or domain.get("selected_physical_plan_fingerprints") is not None
        or domain.get("selected_row_tiles") != domain.get("derived_row_tiles")
    ):
        raise CandidateRankingReportError(
            "APUv1 ranking requires a complete candidate domain"
        )
    candidate_count = len(candidates)
    for field in (
        "full_domain_candidate_count",
        "selected_candidate_count",
        "materialized_candidate_count",
        "measured_candidate_count",
    ):
        _exact_int(domain.get(field), candidate_count, f"domain.{field}")
    _exact_int(
        domain.get("rejected_candidate_count"), 0, "domain.rejected_candidate_count"
    )
    identities = {
        str(_mapping(candidate, "candidate").get("candidate_identity_fingerprint"))
        for candidate in candidates
    }
    if len(identities) != candidate_count:
        raise CandidateRankingReportError(
            "APUv1 report does not contain exact unique candidate coverage"
        )


def _require_apu_g2_complete_coverage(
    report: Mapping[str, Any], candidates: Sequence[object]
) -> None:
    domain = _mapping(report.get("candidate_domain"), "candidate_domain")
    rejections = _sequence(report.get("rejections"), "rejections")
    if rejections:
        raise CandidateRankingReportError(
            "APUg2 ranking requires zero candidate rejections"
        )
    full_batches = list(APUG2_MEASURER.DEFAULT_BATCH_COLUMNS)
    if (
        domain.get("candidate_domain_complete") is not True
        or domain.get("full_batch_columns") != full_batches
        or domain.get("selected_batch_columns") != full_batches
    ):
        raise CandidateRankingReportError(
            "APUg2 ranking requires the complete 1..BMAX candidate domain"
        )
    candidate_count = len(full_batches)
    _exact_int(
        domain.get("full_candidate_count"),
        candidate_count,
        "candidate_domain.full_candidate_count",
    )
    _exact_int(
        domain.get("selected_candidate_count"),
        candidate_count,
        "candidate_domain.selected_candidate_count",
    )
    if len(candidates) != candidate_count:
        raise CandidateRankingReportError(
            "APUg2 report does not contain exact candidate coverage"
        )
    identities = set()
    for index, (batch, candidate_raw) in enumerate(zip(full_batches, candidates)):
        candidate = _mapping(candidate_raw, f"candidates[{index}]")
        _exact_int(candidate.get("candidate_index"), index, "candidate.candidate_index")
        schedule = _mapping(candidate.get("schedule"), "candidate.schedule")
        _exact_int(
            schedule.get("batch_columns"), batch, "candidate.schedule.batch_columns"
        )
        identity = _mapping(candidate.get("identity"), "candidate.identity")
        identities.add(str(identity.get("candidate_fingerprint")))
    if len(identities) != candidate_count:
        raise CandidateRankingReportError(
            "APUg2 report does not contain exact unique candidate coverage"
        )


def _apu_v1_samples(report: Mapping[str, Any]) -> tuple[CalibrationSample, ...]:
    APUV1_MEASURER.validate_measurement_report(report)
    logical_shape = _mapping(report["logical_shape"], "logical_shape")
    provenance = _mapping(report["provenance"], "provenance")
    candidates = _sequence(report["candidates"], "candidates")
    _require_apu_v1_complete_coverage(report, candidates)
    platform_contract = _platform_contract(report, backend="apu_v1")
    target_revision = _fingerprint(
        "apu-v1-ranking-target-revision-v2",
        {
            "target_fingerprint": provenance["target_fingerprint"],
            "platform_contract": platform_contract,
        },
    )
    model_fingerprint = str(provenance["cost_fingerprint"])
    family = _fingerprint(
        "apu-v1-profile-ranking-family-v1",
        {
            "target_revision": target_revision,
            "schedule_family": "structural_row_tile_and_vector_plan_grid",
        },
    )
    shape = _fingerprint("apu-v1-profile-ranking-shape-v1", logical_shape)
    domain = CalibrationObjectiveDomain(
        metric="composed_device_cycles",
        unit="cycles",
        target="apu_v1",
        target_revision=target_revision,
        model_fingerprint=model_fingerprint,
        fidelity="hardware_cycles_vs_executable_cost",
        scope="whole_profile_composition",
    )
    samples = []
    for index, candidate_raw in enumerate(candidates):
        candidate = _mapping(candidate_raw, f"candidates[{index}]")
        measured_records = _sequence(
            candidate["correlated_cycle_samples"],
            f"candidates[{index}].correlated_cycle_samples",
        )
        measured = median(
            float(_mapping(item, "cycle sample")["composed_cycles"])
            for item in measured_records
        )
        prediction = _mapping(
            candidate["model_prediction"], f"candidates[{index}].model_prediction"
        )
        samples.append(
            CalibrationSample(
                sample_fingerprint=str(candidate["measurement_fingerprint"]),
                family_fingerprint=family,
                shape_fingerprint=shape,
                candidate_fingerprint=str(candidate["candidate_identity_fingerprint"]),
                objective_domain=domain,
                measured=measured,
                predicted=float(prediction["composed_cycles"]),
            )
        )
    return tuple(samples)


def _apu_g2_samples(report: Mapping[str, Any]) -> tuple[CalibrationSample, ...]:
    APUG2_MEASURER.validate_measurement_report(report)
    structural_case = _mapping(report["structural_case"], "structural_case")
    provenance = _mapping(report["provenance"], "provenance")
    candidates = _sequence(report["candidates"], "candidates")
    _require_apu_g2_complete_coverage(report, candidates)
    platform_contract = _platform_contract(report, backend="apu_g2")
    target_revision = _fingerprint(
        "apu-g2-ranking-target-revision-v2",
        {
            "source_fingerprint": provenance["source_fingerprint"],
            "runtime_driver_sha256": provenance["runtime_driver_sha256"],
            "platform_contract": platform_contract,
        },
    )
    model_fingerprint = str(provenance["model_fingerprint"])
    family = _fingerprint(
        "apu-g2-persistent-ranking-family-v1",
        {
            "target_revision": target_revision,
            "schedule_family": "persistent_column_batch_grid_k128",
        },
    )
    shape = _fingerprint("apu-g2-persistent-ranking-shape-v1", structural_case)
    domain = CalibrationObjectiveDomain(
        metric="host_wall_us",
        unit="microseconds",
        target="apu_v2",
        target_revision=target_revision,
        model_fingerprint=model_fingerprint,
        fidelity="hardware_wall_vs_transport_model",
        scope="whole_program_transport",
    )
    report_fingerprint = str(report["measurement_fingerprint"])
    samples = []
    for index, candidate_raw in enumerate(candidates):
        candidate = _mapping(candidate_raw, f"candidates[{index}]")
        identity = _mapping(candidate["identity"], f"candidates[{index}].identity")
        measured_records = _sequence(
            candidate["samples"], f"candidates[{index}].samples"
        )
        measured = median(
            float(_mapping(_mapping(item, "sample")["host_us"], "host_us")["wall_us"])
            for item in measured_records
        )
        prediction = _mapping(
            candidate["model_prediction"], f"candidates[{index}].model_prediction"
        )
        candidate_fingerprint = str(identity["candidate_fingerprint"])
        samples.append(
            CalibrationSample(
                sample_fingerprint=_fingerprint(
                    "apu-g2-ranking-sample-v1",
                    {
                        "report_fingerprint": report_fingerprint,
                        "candidate_fingerprint": candidate_fingerprint,
                    },
                ),
                family_fingerprint=family,
                shape_fingerprint=shape,
                candidate_fingerprint=candidate_fingerprint,
                objective_domain=domain,
                measured=measured,
                predicted=float(prediction["wall_us"]),
            )
        )
    return tuple(samples)


def _report_kind(report: Mapping[str, Any]) -> str:
    if report.get("schema") == APUV1_MEASURER.SCHEMA:
        return "apu_v1"
    if report.get("record_kind") == APUG2_MEASURER.RECORD_KIND:
        return "apu_g2"
    raise CandidateRankingReportError("unsupported candidate measurement schema")


@dataclass(frozen=True)
class _CandidateRankingComponents:
    backend: str
    platform_contract_json: str
    source_report_fingerprints: tuple[str, ...]
    evaluation: CalibrationReport

    def __post_init__(self) -> None:
        if self.backend not in ("apu_v1", "apu_g2"):
            raise CandidateRankingReportError("unsupported candidate ranking backend")
        if not isinstance(self.platform_contract_json, str):
            raise CandidateRankingReportError("platform contract is malformed")
        try:
            platform_contract = json.loads(
                self.platform_contract_json,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except json.JSONDecodeError as error:
            raise CandidateRankingReportError(
                "platform contract is malformed"
            ) from error
        _platform_contract(
            {"provenance": {"platform_contract": platform_contract}},
            backend=self.backend,
        )
        if (
            not self.source_report_fingerprints
            or tuple(sorted(self.source_report_fingerprints))
            != self.source_report_fingerprints
            or len(set(self.source_report_fingerprints))
            != len(self.source_report_fingerprints)
        ):
            raise CandidateRankingReportError(
                "source report fingerprints must be sorted and unique"
            )
        if not isinstance(self.evaluation, CalibrationReport):
            raise CandidateRankingReportError(
                "evaluation must be an immutable CalibrationReport"
            )
        if (
            self.evaluation.partition_strategy
            != "frozen_model_shape_stratified_evaluation"
        ):
            raise CandidateRankingReportError(
                "candidate ranking requires shape-stratified frozen-model evaluation"
            )


def _derive_candidate_ranking_components(
    normalized_reports: Sequence[Mapping[str, Any]],
) -> _CandidateRankingComponents:
    if len(normalized_reports) < 2:
        raise CandidateRankingReportError(
            "at least two measurement reports are required"
        )
    kinds = {_report_kind(report) for report in normalized_reports}
    if len(kinds) != 1:
        raise CandidateRankingReportError("measurement reports mix backend schemas")
    kind = next(iter(kinds))
    platform_contracts = {
        _canonical_json(_platform_contract(report, backend=kind))
        for report in normalized_reports
    }
    if len(platform_contracts) != 1:
        raise CandidateRankingReportError(
            "candidate ranking refuses mixed platform contracts"
        )
    platform_contract_json = next(iter(platform_contracts))
    converter = _apu_v1_samples if kind == "apu_v1" else _apu_g2_samples
    samples = tuple(
        sample for report in normalized_reports for sample in converter(report)
    )
    try:
        evaluation = build_frozen_model_shape_stratified_report(samples)
    except CalibrationDataError as error:
        raise CandidateRankingReportError(str(error)) from error
    source_fingerprints = tuple(
        sorted(
            _fingerprint("candidate-ranking-source-report-v1", report)
            for report in normalized_reports
        )
    )
    if len(set(source_fingerprints)) != len(source_fingerprints):
        raise CandidateRankingReportError("candidate ranking repeats a source report")
    return _CandidateRankingComponents(
        backend=kind,
        platform_contract_json=platform_contract_json,
        source_report_fingerprints=source_fingerprints,
        evaluation=evaluation,
    )


@dataclass(frozen=True, slots=True)
class CandidateRankingEvaluationReport(Mapping[str, object]):
    """Immutable report revalidated from frozen source records on serialization."""

    _components: _CandidateRankingComponents
    _source_report_json: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self._components, _CandidateRankingComponents):
            raise CandidateRankingReportError("ranking components are malformed")
        if not isinstance(self._source_report_json, tuple) or not all(
            isinstance(snapshot, str) for snapshot in self._source_report_json
        ):
            raise CandidateRankingReportError(
                "source report snapshots must be canonical JSON strings"
            )
        snapshots = tuple(sorted(self._source_report_json))
        if len(snapshots) < 2 or snapshots != self._source_report_json:
            raise CandidateRankingReportError(
                "source report snapshots must be sorted and contain at least two reports"
            )
        for snapshot in snapshots:
            try:
                parsed = json.loads(snapshot, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as error:
                raise CandidateRankingReportError(
                    "source report snapshot is malformed"
                ) from error
            if _canonical_json(parsed) != snapshot:
                raise CandidateRankingReportError(
                    "source report snapshot is not canonical JSON"
                )
        object.__setattr__(self, "_source_report_json", snapshots)

    def _reconstructed_components(self) -> _CandidateRankingComponents:
        reports = tuple(
            _mapping(
                json.loads(snapshot, object_pairs_hook=_reject_duplicate_keys),
                "source report snapshot",
            )
            for snapshot in self._source_report_json
        )
        reconstructed = _derive_candidate_ranking_components(reports)
        if reconstructed != self._components:
            raise CandidateRankingReportError(
                "candidate ranking report does not match its frozen source records"
            )
        return reconstructed

    def _manifest_body(self) -> dict[str, object]:
        components = self._reconstructed_components()
        return {
            "schema_version": 2,
            "report_kind": "candidate_ranking_evaluation",
            "backend": components.backend,
            "partition_strategy": "frozen_model_shape_stratified_evaluation",
            "identity_evidence": "adapter_derived_content_fingerprints",
            "candidate_coverage": "adapter_verified_complete_candidate_domains",
            "model_training_provenance": "not_provided",
            "evidence_strength": ("complete_domains_with_consistent_platform_contract"),
            "platform_contract": json.loads(components.platform_contract_json),
            "source_report_fingerprints": list(components.source_report_fingerprints),
            "promotion_eligible": False,
            "promotion_ineligibility_reason": (
                "shape-stratified frozen-model evaluation is not "
                "schedule-promotion evidence"
            ),
            "evaluation": components.evaluation.manifest(),
        }

    @property
    def report_fingerprint(self) -> str:
        return _fingerprint(
            "candidate-ranking-evaluation-v2",
            self._manifest_body(),
        )

    def manifest(self) -> dict[str, object]:
        body = self._manifest_body()
        return {
            **body,
            "report_fingerprint": _fingerprint(
                "candidate-ranking-evaluation-v2",
                body,
            ),
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.manifest(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def __getitem__(self, key: str) -> object:
        return self.manifest()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.manifest())

    def __len__(self) -> int:
        return len(self.manifest())


def build_candidate_ranking_report(
    reports: Sequence[Mapping[str, Any]],
) -> CandidateRankingEvaluationReport:
    """Evaluate one frozen model over adapter-verified complete shape domains."""

    if isinstance(reports, (str, bytes, Mapping)):
        raise CandidateRankingReportError("reports must be a sequence of objects")
    normalized_reports = tuple(_mapping(report, "report") for report in reports)
    snapshots = tuple(sorted(_canonical_json(report) for report in normalized_reports))
    return CandidateRankingEvaluationReport(
        _components=_derive_candidate_ranking_components(normalized_reports),
        _source_report_json=snapshots,
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CandidateRankingReportError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_report(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateRankingReportError(f"cannot load {path}: {error}") from error
    return _mapping(value, str(path))


def report_to_json(report: CandidateRankingEvaluationReport) -> str:
    if type(report) is not CandidateRankingEvaluationReport:
        raise CandidateRankingReportError(
            "only immutable builder-produced ranking evaluations can be serialized"
        )
    return CandidateRankingEvaluationReport.to_json(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build shape-stratified frozen-model metrics from complete candidate sweeps."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.overwrite and args.output is None:
        print("FAIL candidate ranking: --overwrite requires --output", file=sys.stderr)
        return 1
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"FAIL candidate ranking: refusing to overwrite {args.output}",
            file=sys.stderr,
        )
        return 1
    try:
        rendered = report_to_json(
            build_candidate_ranking_report(
                tuple(_load_report(path) for path in args.reports)
            )
        )
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except CandidateRankingReportError as error:
        print(f"FAIL candidate ranking: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
