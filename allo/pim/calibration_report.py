"""Workload-agnostic exploratory evaluation for autoscheduler cost models.

The generic API accepts caller-supplied family, shape, candidate, sample, and
partition identifiers.  It cannot verify that those identifiers are structural
or that the observed candidates exhaust a scheduler domain.  Generic reports
therefore remain explicitly non-promotable exploratory candidate-set metrics.
Lower values are better for ``minimize`` domains and higher values are better
for ``maximize`` domains.

Rank correlations and top-1 regret are macro-averaged over ranking problems.
Bias and percentage errors are computed over partition-evaluation candidates.
Percentage errors are reported in percentage points.  Partition-novel-shape
coverage is the fraction of evaluated problem shape IDs absent from that
partition's reference set; it is not model-training or generalization evidence.

The helpers evaluate predictions already present in each sample. They do not
fit or refit a model, and partition reference samples are not claimed to be the
model's training data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from statistics import median
from typing import Literal, TypeAlias


ObjectiveDirection: TypeAlias = Literal["minimize", "maximize"]


def _content_fingerprint(namespace: str, value: object) -> str:
    payload = _canonical_json({"namespace": namespace, "value": value}).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class CalibrationDataError(ValueError):
    """Raised when evaluation data cannot produce an internally valid report."""


class PartitionLeakageError(CalibrationDataError):
    """Raised when evaluation samples leak into a partition reference set."""


class ObjectiveDomainMismatch(CalibrationDataError):
    """Raised when values from non-comparable objective domains are mixed."""

    def __init__(
        self,
        expected: "CalibrationObjectiveDomain",
        actual: "CalibrationObjectiveDomain",
    ) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"objective domain {actual!r} does not match {expected!r}")


def _require_fingerprint(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationDataError(f"{field} must be a non-empty string")
    return value


def _require_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationDataError(f"{field} must be a non-negative integer")
    return value


def _require_finite_value(value: object, field: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationDataError(f"{field} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise CalibrationDataError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise CalibrationDataError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise CalibrationDataError(f"{field} must be positive")
    return result


@dataclass(frozen=True, order=True)
class CalibrationObjectiveDomain:
    """Comparable metric and model provenance for calibration values."""

    metric: str
    unit: str
    target: str
    target_revision: str
    model_fingerprint: str
    fidelity: str
    scope: str
    objective: ObjectiveDirection = "minimize"

    def __post_init__(self) -> None:
        for field in (
            "metric",
            "unit",
            "target",
            "target_revision",
            "model_fingerprint",
            "fidelity",
            "scope",
        ):
            _require_fingerprint(getattr(self, field), f"objective-domain {field}")
        if self.objective not in ("minimize", "maximize"):
            raise CalibrationDataError(
                "objective-domain objective must be 'minimize' or 'maximize'"
            )

    @property
    def cost_model_fingerprint(self) -> str:
        return self.model_fingerprint

    def manifest(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "target": self.target,
            "target_revision": self.target_revision,
            "model_fingerprint": self.model_fingerprint,
            "fidelity": self.fidelity,
            "scope": self.scope,
            "objective": self.objective,
        }


ObjectiveDomain = CalibrationObjectiveDomain


@dataclass(frozen=True, order=True)
class CalibrationProblemKey:
    """Caller-supplied identity of one independently ranked candidate set."""

    family_fingerprint: str
    shape_fingerprint: str

    def __post_init__(self) -> None:
        _require_fingerprint(self.family_fingerprint, "family_fingerprint")
        _require_fingerprint(self.shape_fingerprint, "shape_fingerprint")

    def manifest(self) -> dict[str, str]:
        return {
            "caller_supplied_family_id": self.family_fingerprint,
            "caller_supplied_shape_id": self.shape_fingerprint,
        }


@dataclass(frozen=True, order=True)
class CalibrationCandidateKey:
    """Caller-supplied identity of one observed candidate."""

    family_fingerprint: str
    shape_fingerprint: str
    candidate_fingerprint: str

    def __post_init__(self) -> None:
        _require_fingerprint(self.family_fingerprint, "family_fingerprint")
        _require_fingerprint(self.shape_fingerprint, "shape_fingerprint")
        _require_fingerprint(self.candidate_fingerprint, "candidate_fingerprint")

    @property
    def problem_key(self) -> CalibrationProblemKey:
        return CalibrationProblemKey(
            self.family_fingerprint,
            self.shape_fingerprint,
        )

    def manifest(self) -> dict[str, str]:
        return {
            "caller_supplied_family_id": self.family_fingerprint,
            "caller_supplied_shape_id": self.shape_fingerprint,
            "caller_supplied_candidate_id": self.candidate_fingerprint,
        }


@dataclass(frozen=True)
class CalibrationSample:
    """One observation with caller IDs and a computed immutable content hash."""

    sample_fingerprint: str
    family_fingerprint: str
    shape_fingerprint: str
    candidate_fingerprint: str
    objective_domain: CalibrationObjectiveDomain
    measured: float
    predicted: float

    def __post_init__(self) -> None:
        _require_fingerprint(self.sample_fingerprint, "sample_fingerprint")
        _require_fingerprint(self.family_fingerprint, "family_fingerprint")
        _require_fingerprint(self.shape_fingerprint, "shape_fingerprint")
        _require_fingerprint(self.candidate_fingerprint, "candidate_fingerprint")
        if not isinstance(self.objective_domain, CalibrationObjectiveDomain):
            raise CalibrationDataError(
                "objective_domain must be a CalibrationObjectiveDomain"
            )
        object.__setattr__(
            self,
            "measured",
            _require_finite_value(self.measured, "measured", positive=True),
        )
        object.__setattr__(
            self,
            "predicted",
            _require_finite_value(self.predicted, "predicted", positive=True),
        )

    @property
    def problem_key(self) -> CalibrationProblemKey:
        return CalibrationProblemKey(
            self.family_fingerprint,
            self.shape_fingerprint,
        )

    @property
    def candidate_key(self) -> CalibrationCandidateKey:
        return CalibrationCandidateKey(
            self.family_fingerprint,
            self.shape_fingerprint,
            self.candidate_fingerprint,
        )

    @property
    def content_fingerprint(self) -> str:
        return _content_fingerprint(
            "candidate-set-evaluation-sample-content-v2",
            {
                "caller_supplied_family_id": self.family_fingerprint,
                "caller_supplied_shape_id": self.shape_fingerprint,
                "caller_supplied_candidate_id": self.candidate_fingerprint,
                "objective_domain": self.objective_domain.manifest(),
                "measured": self.measured,
                "predicted": self.predicted,
            },
        )

    def manifest(self) -> dict[str, object]:
        return {
            "caller_supplied_sample_id": self.sample_fingerprint,
            "sample_content_fingerprint": self.content_fingerprint,
            "caller_supplied_family_id": self.family_fingerprint,
            "caller_supplied_shape_id": self.shape_fingerprint,
            "caller_supplied_candidate_id": self.candidate_fingerprint,
            "objective_domain": self.objective_domain.manifest(),
            "measured": self.measured,
            "predicted": self.predicted,
        }


CalibrationRecord = CalibrationSample


def _sample_sort_key(sample: CalibrationSample) -> tuple[str, str, str, str]:
    return (
        sample.family_fingerprint,
        sample.shape_fingerprint,
        sample.candidate_fingerprint,
        sample.sample_fingerprint,
    )


def _snapshot_samples(
    samples: Iterable[CalibrationSample],
    field: str,
) -> tuple[CalibrationSample, ...]:
    if isinstance(samples, (str, bytes, Mapping)):
        raise CalibrationDataError(f"{field} must be an iterable of samples")
    try:
        snapshot = tuple(samples)
    except TypeError as error:
        raise CalibrationDataError(f"{field} must be an iterable of samples") from error
    for index, sample in enumerate(snapshot):
        if not isinstance(sample, CalibrationSample):
            raise CalibrationDataError(f"{field}[{index}] must be a CalibrationSample")
    return tuple(sorted(snapshot, key=_sample_sort_key))


def _group_problems(
    samples: Sequence[CalibrationSample],
) -> dict[CalibrationProblemKey, tuple[CalibrationSample, ...]]:
    grouped: dict[CalibrationProblemKey, list[CalibrationSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.problem_key].append(sample)
    return {
        key: tuple(sorted(group, key=_sample_sort_key))
        for key, group in sorted(grouped.items())
    }


def _validate_sample_collection(
    samples: Sequence[CalibrationSample],
    field: str,
) -> CalibrationObjectiveDomain:
    if not samples:
        raise CalibrationDataError(f"{field} must not be empty")

    sample_ids: set[str] = set()
    sample_content_fingerprints: set[str] = set()
    expected_domain = samples[0].objective_domain
    for sample in samples:
        if sample.sample_fingerprint in sample_ids:
            raise CalibrationDataError(
                f"{field} contains duplicate sample ID "
                f"{sample.sample_fingerprint!r}"
            )
        sample_ids.add(sample.sample_fingerprint)
        if sample.content_fingerprint in sample_content_fingerprints:
            raise CalibrationDataError(
                f"{field} contains duplicate sample content fingerprint "
                f"{sample.content_fingerprint!r}"
            )
        sample_content_fingerprints.add(sample.content_fingerprint)
        if sample.objective_domain != expected_domain:
            raise ObjectiveDomainMismatch(expected_domain, sample.objective_domain)

    for key, candidates in _group_problems(samples).items():
        seen_candidates: set[str] = set()
        for candidate in candidates:
            if candidate.candidate_fingerprint in seen_candidates:
                raise CalibrationDataError(
                    "ranking problem "
                    f"{key!r} contains duplicate candidate identity "
                    f"{candidate.candidate_fingerprint!r}"
                )
            seen_candidates.add(candidate.candidate_fingerprint)
        if len(candidates) < 2:
            raise CalibrationDataError(
                f"ranking problem {key!r} must contain at least two candidates"
            )

    return expected_domain


@dataclass(frozen=True)
class CalibrationPartition:
    """One explicit, leakage-free comparison partition with a caller ID."""

    partition_fingerprint: str
    train: tuple[CalibrationSample, ...]
    holdout: tuple[CalibrationSample, ...]
    separation_axis: Literal["family", "shape"] = "family"

    def __post_init__(self) -> None:
        _require_fingerprint(self.partition_fingerprint, "partition_fingerprint")
        train = _snapshot_samples(self.train, "train")
        holdout = _snapshot_samples(self.holdout, "holdout")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "holdout", holdout)
        if self.separation_axis not in ("family", "shape"):
            raise CalibrationDataError(
                "partition separation_axis must be 'family' or 'shape'"
            )

        expected_domain = _validate_sample_collection(train, "train")
        holdout_domain = _validate_sample_collection(holdout, "holdout")
        if holdout_domain != expected_domain:
            raise ObjectiveDomainMismatch(expected_domain, holdout_domain)

        train_sample_ids = {sample.sample_fingerprint for sample in train}
        holdout_sample_ids = {sample.sample_fingerprint for sample in holdout}
        leaked_ids = sorted(train_sample_ids & holdout_sample_ids)
        if leaked_ids:
            raise PartitionLeakageError(
                "sample leakage by ID between train and holdout: "
                + ", ".join(repr(value) for value in leaked_ids)
            )
        train_content = {sample.content_fingerprint for sample in train}
        holdout_content = {sample.content_fingerprint for sample in holdout}
        leaked_content = sorted(train_content & holdout_content)
        if leaked_content:
            raise PartitionLeakageError(
                "sample content leakage between train and holdout: "
                + ", ".join(repr(value) for value in leaked_content)
            )

        if self.separation_axis == "family":
            train_values = {sample.family_fingerprint for sample in train}
            holdout_values = {sample.family_fingerprint for sample in holdout}
        else:
            train_values = {sample.shape_fingerprint for sample in train}
            holdout_values = {sample.shape_fingerprint for sample in holdout}
        leaked_values = sorted(train_values & holdout_values)
        if leaked_values:
            raise PartitionLeakageError(
                f"{self.separation_axis} leakage between train and holdout: "
                + ", ".join(repr(value) for value in leaked_values)
            )

    @property
    def objective_domain(self) -> CalibrationObjectiveDomain:
        return self.train[0].objective_domain

    @property
    def train_samples(self) -> tuple[CalibrationSample, ...]:
        return self.train

    @property
    def holdout_samples(self) -> tuple[CalibrationSample, ...]:
        return self.holdout

    @property
    def content_fingerprint(self) -> str:
        return _content_fingerprint(
            "candidate-set-evaluation-partition-content-v2",
            {
                "separation_axis": self.separation_axis,
                "partition_reference_samples": [
                    sample.content_fingerprint for sample in self.train
                ],
                "partition_evaluation_samples": [
                    sample.content_fingerprint for sample in self.holdout
                ],
            },
        )

    def manifest(self) -> dict[str, object]:
        return {
            "caller_supplied_partition_id": self.partition_fingerprint,
            "partition_content_fingerprint": self.content_fingerprint,
            "separation_axis": self.separation_axis,
            "partition_reference_samples": [sample.manifest() for sample in self.train],
            "partition_evaluation_samples": [
                sample.manifest() for sample in self.holdout
            ],
        }


def _validate_metric(
    value: float | None,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if value is None:
        return
    result = _require_finite_value(value, field, positive=False)
    if minimum is not None and result < minimum:
        raise CalibrationDataError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise CalibrationDataError(f"{field} must be at most {maximum}")


@dataclass(frozen=True)
class RankingMetrics:
    """Finite metrics over one or more observed candidate sets."""

    ranking_problem_count: int
    candidate_count: int
    rank_correlation_problem_count: int
    partition_novel_shape_problem_count: int
    kendall_tau_b: float | None
    spearman_rank_correlation: float | None
    top_1_regret: float
    signed_relative_bias: float
    median_absolute_percentage_error: float
    p90_absolute_percentage_error: float
    partition_novel_shape_coverage: float

    def __post_init__(self) -> None:
        for field in (
            "ranking_problem_count",
            "candidate_count",
            "rank_correlation_problem_count",
            "partition_novel_shape_problem_count",
        ):
            _require_count(getattr(self, field), field)
        if self.ranking_problem_count == 0:
            raise CalibrationDataError("ranking_problem_count must be positive")
        if self.candidate_count < 2 * self.ranking_problem_count:
            raise CalibrationDataError(
                "candidate_count must include at least two candidates per problem"
            )
        if self.rank_correlation_problem_count > self.ranking_problem_count:
            raise CalibrationDataError(
                "rank_correlation_problem_count exceeds ranking_problem_count"
            )
        if self.partition_novel_shape_problem_count > self.ranking_problem_count:
            raise CalibrationDataError(
                "partition_novel_shape_problem_count exceeds ranking_problem_count"
            )
        _validate_metric(self.kendall_tau_b, "kendall_tau_b", minimum=-1, maximum=1)
        _validate_metric(
            self.spearman_rank_correlation,
            "spearman_rank_correlation",
            minimum=-1,
            maximum=1,
        )
        _validate_metric(self.top_1_regret, "top_1_regret", minimum=0)
        _validate_metric(self.signed_relative_bias, "signed_relative_bias")
        _validate_metric(
            self.median_absolute_percentage_error,
            "median_absolute_percentage_error",
            minimum=0,
        )
        _validate_metric(
            self.p90_absolute_percentage_error,
            "p90_absolute_percentage_error",
            minimum=0,
        )
        _validate_metric(
            self.partition_novel_shape_coverage,
            "partition_novel_shape_coverage",
            minimum=0,
            maximum=1,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "observed_ranking_problem_count": self.ranking_problem_count,
            "observed_candidate_count": self.candidate_count,
            "rank_correlation_observed_problem_count": (
                self.rank_correlation_problem_count
            ),
            "partition_novel_shape_observed_problem_count": (
                self.partition_novel_shape_problem_count
            ),
            "kendall_tau_b": self.kendall_tau_b,
            "spearman_rank_correlation": self.spearman_rank_correlation,
            "observed_set_top_1_regret": self.top_1_regret,
            "signed_relative_bias": self.signed_relative_bias,
            "median_absolute_percentage_error": (self.median_absolute_percentage_error),
            "p90_absolute_percentage_error": self.p90_absolute_percentage_error,
            "partition_novel_shape_coverage": self.partition_novel_shape_coverage,
        }


def _rank_values(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return tuple(ranks)


def _finite_rank_values(values: Iterable[Real], field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise CalibrationDataError(f"{field} must be an iterable of real values")
    try:
        snapshot = tuple(values)
    except TypeError as error:
        raise CalibrationDataError(
            f"{field} must be an iterable of real values"
        ) from error
    return tuple(
        _require_finite_value(value, f"{field}[{index}]", positive=False)
        for index, value in enumerate(snapshot)
    )


def _paired_rank_values(
    measured: Iterable[Real],
    predicted: Iterable[Real],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    measured_values = _finite_rank_values(measured, "measured")
    predicted_values = _finite_rank_values(predicted, "predicted")
    if len(measured_values) != len(predicted_values):
        raise CalibrationDataError(
            "measured and predicted rankings must have equal lengths"
        )
    if len(measured_values) < 2:
        raise CalibrationDataError("rankings must contain at least two values")
    return measured_values, predicted_values


def kendall_tau_b(
    measured: Iterable[Real],
    predicted: Iterable[Real],
) -> float | None:
    """Return Kendall tau-b, or ``None`` when either ranking is constant."""

    measured_values, predicted_values = _paired_rank_values(measured, predicted)
    concordant = 0
    discordant = 0
    measured_only_ties = 0
    predicted_only_ties = 0

    for left in range(len(measured_values) - 1):
        for right in range(left + 1, len(measured_values)):
            measured_delta = measured_values[left] - measured_values[right]
            predicted_delta = predicted_values[left] - predicted_values[right]
            measured_tie = measured_delta == 0.0
            predicted_tie = predicted_delta == 0.0
            if measured_tie and predicted_tie:
                continue
            if measured_tie:
                measured_only_ties += 1
            elif predicted_tie:
                predicted_only_ties += 1
            elif (measured_delta < 0.0) == (predicted_delta < 0.0):
                concordant += 1
            else:
                discordant += 1

    ordered_pairs = concordant + discordant
    denominator = math.sqrt(
        (ordered_pairs + measured_only_ties) * (ordered_pairs + predicted_only_ties)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def spearman_rank_correlation(
    measured: Iterable[Real],
    predicted: Iterable[Real],
) -> float | None:
    """Return Pearson correlation of average ranks, with ties preserved."""

    measured_values, predicted_values = _paired_rank_values(measured, predicted)
    measured_ranks = _rank_values(measured_values)
    predicted_ranks = _rank_values(predicted_values)
    measured_mean = math.fsum(measured_ranks) / len(measured_ranks)
    predicted_mean = math.fsum(predicted_ranks) / len(predicted_ranks)
    measured_offsets = tuple(rank - measured_mean for rank in measured_ranks)
    predicted_offsets = tuple(rank - predicted_mean for rank in predicted_ranks)
    measured_variance = math.fsum(value * value for value in measured_offsets)
    predicted_variance = math.fsum(value * value for value in predicted_offsets)
    if measured_variance == 0.0 or predicted_variance == 0.0:
        return None
    covariance = math.fsum(
        measured_value * predicted_value
        for measured_value, predicted_value in zip(
            measured_offsets,
            predicted_offsets,
        )
    )
    correlation = covariance / math.sqrt(measured_variance * predicted_variance)
    return min(1.0, max(-1.0, correlation))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _mean(values: Iterable[float]) -> float:
    snapshot = tuple(values)
    return math.fsum(snapshot) / len(snapshot)


def _relative_errors(
    candidates: Sequence[CalibrationSample],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    signed = tuple(
        (candidate.predicted - candidate.measured) / candidate.measured
        for candidate in candidates
    )
    absolute_percentage = tuple(abs(value) * 100.0 for value in signed)
    return signed, absolute_percentage


def _top_candidates(
    candidates: Sequence[CalibrationSample],
) -> tuple[CalibrationSample, ...]:
    if candidates[0].objective_domain.objective == "minimize":
        best_prediction = min(candidate.predicted for candidate in candidates)
    else:
        best_prediction = max(candidate.predicted for candidate in candidates)
    return tuple(
        candidate for candidate in candidates if candidate.predicted == best_prediction
    )


def _best_measured(candidates: Sequence[CalibrationSample]) -> float:
    measured = (candidate.measured for candidate in candidates)
    if candidates[0].objective_domain.objective == "minimize":
        return min(measured)
    return max(measured)


def _top_1_regret(candidates: Sequence[CalibrationSample]) -> float:
    best = _best_measured(candidates)
    if candidates[0].objective_domain.objective == "minimize":
        return max(
            max(0.0, (candidate.measured - best) / best)
            for candidate in _top_candidates(candidates)
        )
    return max(
        max(0.0, (best - candidate.measured) / best)
        for candidate in _top_candidates(candidates)
    )


def _metrics_for_problem(
    candidates: Sequence[CalibrationSample],
    *,
    partition_novel_shape: bool,
) -> RankingMetrics:
    measured = tuple(candidate.measured for candidate in candidates)
    predicted = tuple(candidate.predicted for candidate in candidates)
    kendall = kendall_tau_b(measured, predicted)
    spearman = spearman_rank_correlation(measured, predicted)
    signed, absolute_percentage = _relative_errors(candidates)
    correlations_defined = kendall is not None and spearman is not None
    return RankingMetrics(
        ranking_problem_count=1,
        candidate_count=len(candidates),
        rank_correlation_problem_count=int(correlations_defined),
        partition_novel_shape_problem_count=int(partition_novel_shape),
        kendall_tau_b=kendall,
        spearman_rank_correlation=spearman,
        top_1_regret=_top_1_regret(candidates),
        signed_relative_bias=_mean(signed),
        median_absolute_percentage_error=median(absolute_percentage),
        p90_absolute_percentage_error=_percentile(absolute_percentage, 0.9),
        partition_novel_shape_coverage=float(partition_novel_shape),
    )


@dataclass(frozen=True)
class RankingProblemReport:
    """Metrics and auditable samples for one observed candidate set."""

    key: CalibrationProblemKey
    candidates: tuple[CalibrationSample, ...]
    partition_novel_shape: bool
    predicted_top_candidate_fingerprints: tuple[str, ...]
    measured_top_candidate_fingerprints: tuple[str, ...]
    metrics: RankingMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.key, CalibrationProblemKey):
            raise CalibrationDataError("key must be a CalibrationProblemKey")
        candidates = _snapshot_samples(self.candidates, "candidates")
        object.__setattr__(self, "candidates", candidates)
        if any(candidate.problem_key != self.key for candidate in candidates):
            raise CalibrationDataError("candidate does not belong to ranking problem")
        _validate_sample_collection(candidates, "candidates")
        if type(self.partition_novel_shape) is not bool:
            raise CalibrationDataError("partition_novel_shape must be a bool")
        predicted_tops = tuple(sorted(self.predicted_top_candidate_fingerprints))
        if not predicted_tops:
            raise CalibrationDataError(
                "predicted_top_candidate_fingerprints must not be empty"
            )
        for fingerprint in predicted_tops:
            _require_fingerprint(fingerprint, "predicted_top_candidate_fingerprint")
        object.__setattr__(
            self,
            "predicted_top_candidate_fingerprints",
            predicted_tops,
        )
        measured_tops = tuple(sorted(self.measured_top_candidate_fingerprints))
        if not measured_tops:
            raise CalibrationDataError(
                "measured_top_candidate_fingerprints must not be empty"
            )
        for fingerprint in measured_tops:
            _require_fingerprint(fingerprint, "measured_top_candidate_fingerprint")
        object.__setattr__(
            self,
            "measured_top_candidate_fingerprints",
            measured_tops,
        )
        candidate_fingerprints = {
            candidate.candidate_fingerprint for candidate in candidates
        }
        if not set(predicted_tops) <= candidate_fingerprints:
            raise CalibrationDataError("predicted top candidate is not in problem")
        if not set(measured_tops) <= candidate_fingerprints:
            raise CalibrationDataError("measured top candidate is not in problem")
        if not isinstance(self.metrics, RankingMetrics):
            raise CalibrationDataError("metrics must be RankingMetrics")
        expected_predicted_tops = tuple(
            sorted(
                candidate.candidate_fingerprint
                for candidate in _top_candidates(candidates)
            )
        )
        if predicted_tops != expected_predicted_tops:
            raise CalibrationDataError(
                "predicted top candidates do not match candidate predictions"
            )
        best_measured = _best_measured(candidates)
        expected_measured_tops = tuple(
            sorted(
                candidate.candidate_fingerprint
                for candidate in candidates
                if candidate.measured == best_measured
            )
        )
        if measured_tops != expected_measured_tops:
            raise CalibrationDataError(
                "measured top candidates do not match candidate measurements"
            )
        expected_metrics = _metrics_for_problem(
            candidates,
            partition_novel_shape=self.partition_novel_shape,
        )
        if self.metrics != expected_metrics:
            raise CalibrationDataError("ranking problem metrics are inconsistent")

    def manifest(self) -> dict[str, object]:
        return {
            **self.key.manifest(),
            "shape_absent_from_partition_reference": self.partition_novel_shape,
            "predicted_top_caller_candidate_ids": list(
                self.predicted_top_candidate_fingerprints
            ),
            "measured_top_caller_candidate_ids": list(
                self.measured_top_candidate_fingerprints
            ),
            "candidates": [candidate.manifest() for candidate in self.candidates],
            "metrics": self.metrics.manifest(),
        }


def _build_problem_report(
    key: CalibrationProblemKey,
    candidates: tuple[CalibrationSample, ...],
    *,
    partition_novel_shape: bool,
) -> RankingProblemReport:
    best_measured = _best_measured(candidates)
    measured_tops = tuple(
        candidate.candidate_fingerprint
        for candidate in candidates
        if candidate.measured == best_measured
    )
    return RankingProblemReport(
        key=key,
        candidates=candidates,
        partition_novel_shape=partition_novel_shape,
        predicted_top_candidate_fingerprints=tuple(
            candidate.candidate_fingerprint for candidate in _top_candidates(candidates)
        ),
        measured_top_candidate_fingerprints=measured_tops,
        metrics=_metrics_for_problem(
            candidates,
            partition_novel_shape=partition_novel_shape,
        ),
    )


def _aggregate_problem_metrics(
    problems: Sequence[RankingProblemReport],
) -> RankingMetrics:
    candidates = tuple(
        candidate for problem in problems for candidate in problem.candidates
    )
    signed, absolute_percentage = _relative_errors(candidates)
    correlation_count = sum(
        problem.metrics.rank_correlation_problem_count for problem in problems
    )
    correlations_complete = correlation_count == len(problems)
    kendall = (
        _mean(problem.metrics.kendall_tau_b for problem in problems)
        if correlations_complete
        else None
    )
    spearman = (
        _mean(problem.metrics.spearman_rank_correlation for problem in problems)
        if correlations_complete
        else None
    )
    partition_novel_shape_count = sum(
        problem.partition_novel_shape for problem in problems
    )
    return RankingMetrics(
        ranking_problem_count=len(problems),
        candidate_count=len(candidates),
        rank_correlation_problem_count=correlation_count,
        partition_novel_shape_problem_count=partition_novel_shape_count,
        kendall_tau_b=kendall,
        spearman_rank_correlation=spearman,
        top_1_regret=_mean(problem.metrics.top_1_regret for problem in problems),
        signed_relative_bias=_mean(signed),
        median_absolute_percentage_error=median(absolute_percentage),
        p90_absolute_percentage_error=_percentile(absolute_percentage, 0.9),
        partition_novel_shape_coverage=(partition_novel_shape_count / len(problems)),
    )


@dataclass(frozen=True)
class PartitionEvaluationReport:
    """Observed candidate sets and metrics for one comparison partition."""

    partition: CalibrationPartition
    problems: tuple[RankingProblemReport, ...]
    metrics: RankingMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.partition, CalibrationPartition):
            raise CalibrationDataError("partition must be a CalibrationPartition")
        problems = tuple(sorted(self.problems, key=lambda problem: problem.key))
        if not problems or not all(
            isinstance(problem, RankingProblemReport) for problem in problems
        ):
            raise CalibrationDataError(
                "problems must contain RankingProblemReport records"
            )
        object.__setattr__(self, "problems", problems)
        if not isinstance(self.metrics, RankingMetrics):
            raise CalibrationDataError("metrics must be RankingMetrics")
        expected_groups = _group_problems(self.partition.holdout)
        if tuple(problem.key for problem in problems) != tuple(expected_groups):
            raise CalibrationDataError(
                "holdout problems do not exactly cover the partition"
            )
        training_shapes = {sample.shape_fingerprint for sample in self.partition.train}
        for problem in problems:
            if problem.candidates != expected_groups[problem.key]:
                raise CalibrationDataError(
                    "holdout problem candidates do not match the partition"
                )
            expected_partition_novel_shape = (
                problem.key.shape_fingerprint not in training_shapes
            )
            if problem.partition_novel_shape is not expected_partition_novel_shape:
                raise CalibrationDataError("partition-novel-shape flag is inconsistent")
        if self.metrics != _aggregate_problem_metrics(problems):
            raise CalibrationDataError("holdout metrics are inconsistent")

    @property
    def partition_fingerprint(self) -> str:
        return self.partition.partition_fingerprint

    def manifest(self) -> dict[str, object]:
        return {
            "partition": self.partition.manifest(),
            "problems": [problem.manifest() for problem in self.problems],
            "metrics": self.metrics.manifest(),
        }


HoldoutReport = PartitionEvaluationReport


def _build_partition_evaluation_report(
    partition: CalibrationPartition,
) -> PartitionEvaluationReport:
    training_shapes = {sample.shape_fingerprint for sample in partition.train}
    problems = tuple(
        _build_problem_report(
            key,
            candidates,
            partition_novel_shape=key.shape_fingerprint not in training_shapes,
        )
        for key, candidates in _group_problems(partition.holdout).items()
    )
    return PartitionEvaluationReport(
        partition=partition,
        problems=problems,
        metrics=_aggregate_problem_metrics(problems),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Deterministic exploratory evaluation over caller-provided candidate sets."""

    partition_strategy: str
    objective_domain: CalibrationObjectiveDomain
    holdouts: tuple[PartitionEvaluationReport, ...]
    aggregate_metrics: RankingMetrics

    def __post_init__(self) -> None:
        _require_fingerprint(self.partition_strategy, "partition_strategy")
        if not isinstance(self.objective_domain, CalibrationObjectiveDomain):
            raise CalibrationDataError(
                "objective_domain must be a CalibrationObjectiveDomain"
            )
        holdouts = tuple(
            sorted(self.holdouts, key=lambda holdout: holdout.partition_fingerprint)
        )
        if not holdouts or not all(
            isinstance(holdout, PartitionEvaluationReport) for holdout in holdouts
        ):
            raise CalibrationDataError(
                "holdouts must contain at least one PartitionEvaluationReport"
            )
        object.__setattr__(self, "holdouts", holdouts)
        if not isinstance(self.aggregate_metrics, RankingMetrics):
            raise CalibrationDataError("aggregate_metrics must be RankingMetrics")
        if any(
            holdout.partition.objective_domain != self.objective_domain
            for holdout in holdouts
        ):
            raise ObjectiveDomainMismatch(
                self.objective_domain,
                next(
                    holdout.partition.objective_domain
                    for holdout in holdouts
                    if holdout.partition.objective_domain != self.objective_domain
                ),
            )
        problems = tuple(
            problem for holdout in holdouts for problem in holdout.problems
        )
        if self.aggregate_metrics != _aggregate_problem_metrics(problems):
            raise CalibrationDataError("aggregate metrics are inconsistent")

    def _manifest_body(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "report_kind": "exploratory_candidate_set_evaluation",
            "identity_evidence": "caller_supplied_ids_unverified",
            "candidate_coverage": "observed_candidate_sets_only",
            "model_training_provenance": "not_provided",
            "evidence_strength": (
                "exploratory_unverified_identity_and_candidate_coverage"
            ),
            "promotion_eligible": False,
            "promotion_ineligibility_reason": (
                "caller-supplied identifiers and observed candidate sets are not "
                "schedule-promotion evidence"
            ),
            "partition_strategy": self.partition_strategy,
            "objective_domain": self.objective_domain.manifest(),
            "partition_evaluations": [holdout.manifest() for holdout in self.holdouts],
            "aggregate_metrics": self.aggregate_metrics.manifest(),
        }

    @property
    def report_fingerprint(self) -> str:
        body = _canonical_json(self._manifest_body()).encode("ascii")
        return hashlib.sha256(body).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            **self._manifest_body(),
            "report_fingerprint": self.report_fingerprint,
        }

    def to_json(self) -> str:
        return _canonical_json(self.manifest())


def explicit_partition(
    train: Iterable[CalibrationSample],
    holdout: Iterable[CalibrationSample],
    *,
    partition_fingerprint: str = "explicit",
    separation_axis: Literal["family", "shape"] = "family",
) -> CalibrationPartition:
    """Create one validated reference/evaluation comparison partition."""

    return CalibrationPartition(
        partition_fingerprint=partition_fingerprint,
        train=train,
        holdout=holdout,
        separation_axis=separation_axis,
    )


def leave_one_family_out_partitions(
    samples: Iterable[CalibrationSample],
) -> tuple[CalibrationPartition, ...]:
    """Partition observations by caller-supplied family ID.

    The reference side is not asserted to be the model's training data.
    """

    snapshot = _snapshot_samples(samples, "samples")
    _validate_sample_collection(snapshot, "samples")
    families = tuple(sorted({sample.family_fingerprint for sample in snapshot}))
    if len(families) < 2:
        raise CalibrationDataError(
            "family-stratified evaluation requires at least two caller family IDs"
        )
    return tuple(
        CalibrationPartition(
            partition_fingerprint=f"family:{family}",
            train=tuple(
                sample for sample in snapshot if sample.family_fingerprint != family
            ),
            holdout=tuple(
                sample for sample in snapshot if sample.family_fingerprint == family
            ),
        )
        for family in families
    )


def leave_one_shape_out_partitions(
    samples: Iterable[CalibrationSample],
) -> tuple[CalibrationPartition, ...]:
    """Partition observations by caller-supplied shape ID.

    The reference side is not asserted to be the model's training data.
    """

    snapshot = _snapshot_samples(samples, "samples")
    _validate_sample_collection(snapshot, "samples")
    shapes = tuple(sorted({sample.shape_fingerprint for sample in snapshot}))
    if len(shapes) < 2:
        raise CalibrationDataError(
            "shape-stratified evaluation requires at least two caller shape IDs"
        )
    return tuple(
        CalibrationPartition(
            partition_fingerprint=f"shape:{shape}",
            train=tuple(
                sample for sample in snapshot if sample.shape_fingerprint != shape
            ),
            holdout=tuple(
                sample for sample in snapshot if sample.shape_fingerprint == shape
            ),
            separation_axis="shape",
        )
        for shape in shapes
    )


def _snapshot_partitions(
    partitions: Iterable[CalibrationPartition] | CalibrationPartition,
) -> tuple[CalibrationPartition, ...]:
    if isinstance(partitions, CalibrationPartition):
        snapshot = (partitions,)
    else:
        if isinstance(partitions, (str, bytes, Mapping)):
            raise CalibrationDataError(
                "partitions must be an iterable of CalibrationPartition records"
            )
        try:
            snapshot = tuple(partitions)
        except TypeError as error:
            raise CalibrationDataError(
                "partitions must be an iterable of CalibrationPartition records"
            ) from error
    if not snapshot:
        raise CalibrationDataError("partitions must not be empty")
    for index, partition in enumerate(snapshot):
        if not isinstance(partition, CalibrationPartition):
            raise CalibrationDataError(
                f"partitions[{index}] must be a CalibrationPartition"
            )
    return tuple(sorted(snapshot, key=lambda item: item.partition_fingerprint))


def build_calibration_report(
    partitions: Iterable[CalibrationPartition] | CalibrationPartition,
    *,
    partition_strategy: str = "explicit",
) -> CalibrationReport:
    """Evaluate observed sets; identity and domain completeness stay unverified."""

    snapshot = _snapshot_partitions(partitions)
    partition_fingerprints: set[str] = set()
    heldout_sample_ids: set[str] = set()
    heldout_sample_content: set[str] = set()
    expected_domain = snapshot[0].objective_domain
    for partition in snapshot:
        if partition.partition_fingerprint in partition_fingerprints:
            raise CalibrationDataError(
                "duplicate partition fingerprint "
                f"{partition.partition_fingerprint!r}"
            )
        partition_fingerprints.add(partition.partition_fingerprint)
        if partition.objective_domain != expected_domain:
            raise ObjectiveDomainMismatch(
                expected_domain,
                partition.objective_domain,
            )
        for sample in partition.holdout:
            if sample.sample_fingerprint in heldout_sample_ids:
                raise PartitionLeakageError(
                    "holdout sample ID appears in more than one partition: "
                    f"{sample.sample_fingerprint!r}"
                )
            heldout_sample_ids.add(sample.sample_fingerprint)
            if sample.content_fingerprint in heldout_sample_content:
                raise PartitionLeakageError(
                    "holdout sample content appears in more than one partition: "
                    f"{sample.content_fingerprint!r}"
                )
            heldout_sample_content.add(sample.content_fingerprint)

    holdouts = tuple(
        _build_partition_evaluation_report(partition) for partition in snapshot
    )
    problems = tuple(problem for holdout in holdouts for problem in holdout.problems)
    return CalibrationReport(
        partition_strategy=partition_strategy,
        objective_domain=expected_domain,
        holdouts=holdouts,
        aggregate_metrics=_aggregate_problem_metrics(problems),
    )


def build_explicit_calibration_report(
    train: Iterable[CalibrationSample],
    holdout: Iterable[CalibrationSample],
    *,
    partition_fingerprint: str = "explicit",
) -> CalibrationReport:
    """Build an exploratory report from one caller-supplied partition."""

    partition = explicit_partition(
        train,
        holdout,
        partition_fingerprint=partition_fingerprint,
    )
    return build_calibration_report(partition, partition_strategy="explicit")


def build_leave_one_family_out_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Compatibility alias for frozen-model family-stratified evaluation."""

    return build_frozen_model_family_stratified_report(samples)


def build_frozen_model_family_stratified_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Evaluate fixed predictions in caller-ID family strata.

    No model is fit or refit, and no model-training provenance is inferred from
    the partition reference samples.
    """

    return build_calibration_report(
        leave_one_family_out_partitions(samples),
        partition_strategy="frozen_model_family_stratified_evaluation",
    )


def build_leave_one_shape_out_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Compatibility alias for frozen-model shape-stratified evaluation."""

    return build_frozen_model_shape_stratified_report(samples)


def build_frozen_model_shape_stratified_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Evaluate fixed predictions in caller-ID shape strata.

    No model is fit or refit, and no model-training provenance is inferred from
    the partition reference samples.
    """

    return build_calibration_report(
        leave_one_shape_out_partitions(samples),
        partition_strategy="frozen_model_shape_stratified_evaluation",
    )


def build_frozen_model_leave_one_family_out_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Compatibility alias for :func:`build_frozen_model_family_stratified_report`."""

    return build_frozen_model_family_stratified_report(samples)


def build_frozen_model_leave_one_shape_out_report(
    samples: Iterable[CalibrationSample],
) -> CalibrationReport:
    """Compatibility alias for :func:`build_frozen_model_shape_stratified_report`."""

    return build_frozen_model_shape_stratified_report(samples)


CandidateSetEvaluationReport = CalibrationReport
family_stratified_partitions = leave_one_family_out_partitions
shape_stratified_partitions = leave_one_shape_out_partitions


__all__ = [
    "CalibrationCandidateKey",
    "CalibrationDataError",
    "CalibrationObjectiveDomain",
    "CalibrationPartition",
    "CalibrationProblemKey",
    "CalibrationRecord",
    "CalibrationReport",
    "CalibrationSample",
    "CandidateSetEvaluationReport",
    "HoldoutReport",
    "ObjectiveDomain",
    "ObjectiveDomainMismatch",
    "PartitionEvaluationReport",
    "PartitionLeakageError",
    "RankingMetrics",
    "RankingProblemReport",
    "build_calibration_report",
    "build_explicit_calibration_report",
    "build_frozen_model_family_stratified_report",
    "build_frozen_model_leave_one_family_out_report",
    "build_frozen_model_leave_one_shape_out_report",
    "build_frozen_model_shape_stratified_report",
    "build_leave_one_family_out_report",
    "build_leave_one_shape_out_report",
    "explicit_partition",
    "family_stratified_partitions",
    "kendall_tau_b",
    "leave_one_family_out_partitions",
    "leave_one_shape_out_partitions",
    "shape_stratified_partitions",
    "spearman_rank_correlation",
]
