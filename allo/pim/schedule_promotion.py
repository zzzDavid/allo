"""Fail-closed evidence gates for activating schedule-search challengers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, TypeAlias

from .schedule_search import (
    OpaqueScheduleIncumbent,
    PromotionDecision,
    ScheduleCandidate,
    ScheduleObjectiveDomain,
    structural_decision_key,
)


ScheduleLike: TypeAlias = (
    ScheduleCandidate[object, object, object, object]
    | OpaqueScheduleIncumbent[object, object]
)


_MISSING = object()


def _canonical_key(value: object) -> object:
    if isinstance(value, type):
        return {
            "kind": "type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) is complex:
        return {
            "kind": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is tuple:
        return {"kind": "tuple", "items": [_canonical_key(item) for item in value]}
    if type(value) is frozenset:
        items = [_canonical_key(item) for item in value]
        items.sort(key=_canonical_json)
        return {"kind": "frozenset", "items": items}
    raise TypeError(f"unsupported structural key component {type(value).__name__!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _key_digest(key: Hashable) -> str:
    payload = _canonical_json(_canonical_key(key)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _structural_manifest(value: object, *, path: str) -> object:
    return _canonical_key(structural_decision_key(value, path=path))


def _normalize_tokens(
    values: Iterable[Hashable],
    *,
    path: str,
) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{path} must be an iterable of structural tokens")
    keyed: dict[Hashable, Hashable] = {}
    for index, value in enumerate(values):
        key = structural_decision_key(value, path=f"{path}[{index}]")
        if key in keyed:
            raise ValueError(f"{path} contains duplicate structural tokens")
        keyed[key] = value
    return tuple(keyed[key] for key in sorted(keyed, key=_key_digest))


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _current_materialization_key(schedule: ScheduleLike) -> Hashable | None:
    fingerprint = getattr(
        schedule.materialized,
        "promotion_materialization_fingerprint",
        _MISSING,
    )
    if fingerprint is _MISSING:
        return None
    if callable(fingerprint):
        fingerprint = fingerprint()
    if fingerprint is None or (isinstance(fingerprint, str) and not fingerprint):
        return None
    return structural_decision_key(
        fingerprint,
        path="current promotion materialization fingerprint",
    )


def _current_platform_key(schedule: ScheduleLike) -> Hashable | None:
    fingerprint = getattr(
        schedule.materialized,
        "promotion_platform_fingerprint",
        _MISSING,
    )
    if fingerprint is _MISSING:
        return None
    if callable(fingerprint):
        fingerprint = fingerprint()
    if fingerprint is None or (isinstance(fingerprint, str) and not fingerprint):
        return None
    return structural_decision_key(
        fingerprint,
        path="current promotion platform fingerprint",
    )


@dataclass(frozen=True)
class ScheduleDecisionIdentity:
    """A deterministic structural identity for one schedule decision record."""

    kind: Literal["candidate", "opaque_incumbent"]
    structural_key: Hashable = field(repr=False)
    digest: str

    def __post_init__(self) -> None:
        if self.kind not in ("candidate", "opaque_incumbent"):
            raise ValueError("schedule identity kind is invalid")
        if self.digest != _key_digest(self.structural_key):
            raise ValueError(
                "schedule identity digest does not match its structural key"
            )

    @classmethod
    def from_schedule(cls, schedule: ScheduleLike) -> "ScheduleDecisionIdentity":
        if isinstance(schedule, OpaqueScheduleIncumbent):
            kind: Literal["candidate", "opaque_incumbent"] = "opaque_incumbent"
        elif isinstance(schedule, ScheduleCandidate):
            kind = "candidate"
        else:
            raise TypeError(
                "schedule identity requires a ScheduleCandidate or "
                "OpaqueScheduleIncumbent"
            )

        decisions = tuple(
            (
                name,
                structural_decision_key(
                    value,
                    path=f"schedule decision {name!r}",
                ),
            )
            for name, value in sorted(schedule.decisions.items())
        )
        structural_key: Hashable = (
            "schedule-decision-identity-v1",
            kind,
            decisions,
        )
        if isinstance(schedule, OpaqueScheduleIncumbent):
            structural_key = (
                *structural_key,
                structural_decision_key(
                    schedule.fingerprint,
                    path="opaque incumbent fingerprint",
                ),
            )
        return cls(kind, structural_key, _key_digest(structural_key))

    def manifest(self) -> dict[str, str]:
        return {"kind": self.kind, "sha256": self.digest}


@dataclass(frozen=True)
class CorrectnessEvidence:
    """Correctness result and the structural identity of its exact oracle."""

    passed: bool
    exact: bool
    oracle_fingerprint: Hashable

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.exact) is not bool:
            raise TypeError("correctness passed and exact flags must be bools")
        structural_decision_key(
            self.oracle_fingerprint,
            path="correctness oracle fingerprint",
        )

    @classmethod
    def exact_pass(cls, oracle_fingerprint: Hashable) -> "CorrectnessEvidence":
        return cls(True, True, oracle_fingerprint)

    @property
    def oracle_key(self) -> Hashable:
        return structural_decision_key(
            self.oracle_fingerprint,
            path="correctness oracle fingerprint",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "exact": self.exact,
            "oracle_fingerprint": _structural_manifest(
                self.oracle_fingerprint,
                path="correctness oracle fingerprint",
            ),
        }


@dataclass(frozen=True)
class SemanticScope:
    """Comparable semantic coverage without workload-name classifications."""

    domain: Hashable
    guarantees: tuple[Hashable, ...]
    omissions: tuple[Hashable, ...] = ()
    complete: bool = False

    def __post_init__(self) -> None:
        structural_decision_key(self.domain, path="semantic domain")
        object.__setattr__(
            self,
            "guarantees",
            _normalize_tokens(self.guarantees, path="semantic guarantees"),
        )
        object.__setattr__(
            self,
            "omissions",
            _normalize_tokens(self.omissions, path="semantic omissions"),
        )
        if type(self.complete) is not bool:
            raise TypeError("semantic completeness must be a bool")
        overlap = self.guarantee_keys & self.omission_keys
        if overlap:
            raise ValueError("semantic guarantees and omissions must be disjoint")
        if self.complete and self.omissions:
            raise ValueError("complete semantic scope cannot declare omissions")

    @property
    def domain_key(self) -> Hashable:
        return structural_decision_key(self.domain, path="semantic domain")

    @property
    def guarantee_keys(self) -> frozenset[Hashable]:
        return frozenset(
            structural_decision_key(item, path="semantic guarantee")
            for item in self.guarantees
        )

    @property
    def omission_keys(self) -> frozenset[Hashable]:
        return frozenset(
            structural_decision_key(item, path="semantic omission")
            for item in self.omissions
        )

    def manifest(self) -> dict[str, object]:
        return {
            "domain": _structural_manifest(self.domain, path="semantic domain"),
            "guarantees": [
                _structural_manifest(item, path="semantic guarantee")
                for item in self.guarantees
            ],
            "omissions": [
                _structural_manifest(item, path="semantic omission")
                for item in self.omissions
            ],
            "complete": self.complete,
        }


@dataclass(frozen=True)
class MaterializationEvidence:
    """Fingerprints for scored/emitted artifacts and their exact platform."""

    scored_fingerprint: Hashable
    emitted_fingerprint: Hashable
    platform_fingerprint: Hashable

    def __post_init__(self) -> None:
        structural_decision_key(
            self.scored_fingerprint,
            path="scored materialization fingerprint",
        )
        structural_decision_key(
            self.emitted_fingerprint,
            path="emitted materialization fingerprint",
        )
        structural_decision_key(
            self.platform_fingerprint,
            path="materialization platform fingerprint",
        )

    @property
    def scored_key(self) -> Hashable:
        return structural_decision_key(
            self.scored_fingerprint,
            path="scored materialization fingerprint",
        )

    @property
    def emitted_key(self) -> Hashable:
        return structural_decision_key(
            self.emitted_fingerprint,
            path="emitted materialization fingerprint",
        )

    @property
    def platform_key(self) -> Hashable:
        return structural_decision_key(
            self.platform_fingerprint,
            path="materialization platform fingerprint",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "scored_fingerprint": _structural_manifest(
                self.scored_fingerprint,
                path="scored materialization fingerprint",
            ),
            "emitted_fingerprint": _structural_manifest(
                self.emitted_fingerprint,
                path="emitted materialization fingerprint",
            ),
            "platform_fingerprint": _structural_manifest(
                self.platform_fingerprint,
                path="materialization platform fingerprint",
            ),
        }


@dataclass(frozen=True)
class ScheduleEvidence:
    """Correctness, semantics, and artifact evidence for one bound schedule."""

    identity: ScheduleDecisionIdentity
    correctness: CorrectnessEvidence
    semantic_scope: SemanticScope
    materialization: MaterializationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ScheduleDecisionIdentity):
            raise TypeError("schedule evidence requires a ScheduleDecisionIdentity")
        if not isinstance(self.correctness, CorrectnessEvidence):
            raise TypeError("schedule evidence requires CorrectnessEvidence")
        if not isinstance(self.semantic_scope, SemanticScope):
            raise TypeError("schedule evidence requires SemanticScope")
        if not isinstance(self.materialization, MaterializationEvidence):
            raise TypeError("schedule evidence requires MaterializationEvidence")

    @classmethod
    def from_schedule(
        cls,
        schedule: ScheduleLike,
        *,
        correctness: CorrectnessEvidence,
        semantic_scope: SemanticScope,
        scored_fingerprint: Hashable,
        emitted_fingerprint: Hashable,
        platform_fingerprint: Hashable,
    ) -> "ScheduleEvidence":
        return cls(
            ScheduleDecisionIdentity.from_schedule(schedule),
            correctness,
            semantic_scope,
            MaterializationEvidence(
                scored_fingerprint,
                emitted_fingerprint,
                platform_fingerprint,
            ),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "identity": self.identity.manifest(),
            "correctness": self.correctness.manifest(),
            "semantic_scope": self.semantic_scope.manifest(),
            "materialization": self.materialization.manifest(),
        }


@dataclass(frozen=True)
class WarmupEvidence:
    """Explicitly recorded warmup behavior for one measurement series."""

    count: int
    discarded: bool

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("warmup count must be an integer")
        if self.count < 0:
            raise ValueError("warmup count must be non-negative")
        if type(self.discarded) is not bool:
            raise TypeError("warmup discarded flag must be a bool")

    def manifest(self) -> dict[str, object]:
        return {"count": self.count, "discarded": self.discarded}


def _validate_measurement(value: object, *, index: int) -> None:
    if type(value) not in (int, float):
        raise TypeError(f"measurement {index} must be an int or float")
    if value <= 0 or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"measurement {index} must be finite and positive")


@dataclass(frozen=True)
class MetricMeasurements:
    """Measured values in one explicitly typed objective domain."""

    objective_domain: ScheduleObjectiveDomain
    samples: tuple[int | float, ...]
    warmup: WarmupEvidence | None
    platform_fingerprint: Hashable

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        for index, value in enumerate(self.samples):
            _validate_measurement(value, index=index)
        if self.warmup is not None and not isinstance(self.warmup, WarmupEvidence):
            raise TypeError("metric warmup must be WarmupEvidence or None")
        structural_decision_key(
            self.platform_fingerprint,
            path="measurement platform fingerprint",
        )

    @property
    def platform_key(self) -> Hashable:
        return structural_decision_key(
            self.platform_fingerprint,
            path="measurement platform fingerprint",
        )

    def manifest(self) -> dict[str, object]:
        domain = self.objective_domain
        if isinstance(domain, ScheduleObjectiveDomain):
            domain_manifest: object = {
                "metric": domain.metric,
                "target": domain.target,
                "target_revision": domain.target_revision,
                "model_fingerprint": _structural_manifest(
                    domain.model_fingerprint,
                    path="objective-domain model fingerprint",
                ),
                "fidelity": domain.fidelity,
                "scope": domain.scope,
                "unit": domain.unit,
                "direction": domain.direction,
            }
        else:
            domain_manifest = {"invalid_type": _type_name(domain)}
        return {
            "objective_domain": domain_manifest,
            "samples": list(self.samples),
            "warmup": None if self.warmup is None else self.warmup.manifest(),
            "platform_fingerprint": _structural_manifest(
                self.platform_fingerprint,
                path="measurement platform fingerprint",
            ),
        }


@dataclass(frozen=True)
class ObjectiveMetricBridge:
    """Typed provenance connecting search ranking to a measured quantity."""

    search_objective_domain: ScheduleObjectiveDomain
    measurement_objective_domain: ScheduleObjectiveDomain
    search_unit: str
    measurement_unit: str
    search_direction: Literal["minimize"]
    measurement_direction: Literal["minimize"]
    relation: Literal["identity_metric"]
    provenance_fingerprint: Hashable

    def __post_init__(self) -> None:
        if not isinstance(self.search_objective_domain, ScheduleObjectiveDomain):
            raise TypeError("bridge search domain must be ScheduleObjectiveDomain")
        if not isinstance(
            self.measurement_objective_domain,
            ScheduleObjectiveDomain,
        ):
            raise TypeError("bridge measurement domain must be ScheduleObjectiveDomain")
        for name in ("search_unit", "measurement_unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"objective bridge {name} must be non-empty")
        if self.search_direction != "minimize":
            raise ValueError("schedule search objective direction must be 'minimize'")
        if self.measurement_direction != "minimize":
            raise ValueError("promotion measurement direction must be 'minimize'")
        if self.relation != "identity_metric":
            raise ValueError("objective bridge relation must be 'identity_metric'")
        if (
            self.search_objective_domain.metric
            != self.measurement_objective_domain.metric
        ):
            raise ValueError(
                "objective bridge requires identical search and measurement metrics"
            )
        if self.search_unit != self.search_objective_domain.unit:
            raise ValueError("objective bridge search unit does not match its domain")
        if self.measurement_unit != self.measurement_objective_domain.unit:
            raise ValueError(
                "objective bridge measurement unit does not match its domain"
            )
        if self.search_unit != self.measurement_unit:
            raise ValueError("identity metric bridge requires identical units")
        if self.search_direction != self.search_objective_domain.direction:
            raise ValueError(
                "objective bridge search direction does not match its domain"
            )
        if self.measurement_direction != self.measurement_objective_domain.direction:
            raise ValueError(
                "objective bridge measurement direction does not match its domain"
            )
        structural_decision_key(
            self.provenance_fingerprint,
            path="objective bridge provenance fingerprint",
        )

    @property
    def search_domain_key(self) -> Hashable:
        return structural_decision_key(
            self.search_objective_domain,
            path="objective bridge search domain",
        )

    @property
    def measurement_domain_key(self) -> Hashable:
        return structural_decision_key(
            self.measurement_objective_domain,
            path="objective bridge measurement domain",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "search_objective_domain": self.search_objective_domain.manifest(),
            "measurement_objective_domain": (
                self.measurement_objective_domain.manifest()
            ),
            "search_unit": self.search_unit,
            "measurement_unit": self.measurement_unit,
            "search_direction": self.search_direction,
            "measurement_direction": self.measurement_direction,
            "relation": self.relation,
            "provenance_fingerprint": _structural_manifest(
                self.provenance_fingerprint,
                path="objective bridge provenance fingerprint",
            ),
        }


def _validate_positive_int(value: object, *, name: str, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_ratio(value: object, *, name: str) -> None:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float")
    if value <= 0 or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ExactCyclePolicy:
    """Strict deterministic-cycle policy with integral non-regression."""

    min_samples: int = 1

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.min_samples,
            name="exact-cycle minimum samples",
            allow_zero=False,
        )

    @property
    def warmup_samples(self) -> int:
        return 0

    @property
    def median_ratio(self) -> float:
        return 1.0

    @property
    def max_run_ratio(self) -> float:
        return 1.0

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "exact_cycle",
            "warmup_samples": self.warmup_samples,
            "min_samples": self.min_samples,
            "median_ratio": self.median_ratio,
            "max_run_ratio": self.max_run_ratio,
        }


@dataclass(frozen=True)
class NoisyHardwarePolicy:
    """Repeated-hardware policy using median and optional tail ceilings."""

    warmup_samples: int
    min_samples: int
    median_ratio: int | float
    max_run_ratio: int | float | None = None

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.warmup_samples,
            name="hardware warmup samples",
            allow_zero=False,
        )
        _validate_positive_int(
            self.min_samples,
            name="hardware minimum samples",
            allow_zero=False,
        )
        if self.min_samples < 5:
            raise ValueError("hardware minimum samples must be at least five")
        _validate_ratio(self.median_ratio, name="hardware median ratio")
        if Decimal(str(self.median_ratio)) > Decimal("1"):
            raise ValueError("hardware median ratio cannot exceed 1.0")
        if self.max_run_ratio is None:
            raise ValueError("hardware maximum-run ratio is required")
        _validate_ratio(self.max_run_ratio, name="hardware maximum-run ratio")
        if Decimal(str(self.max_run_ratio)) > Decimal("1.05"):
            raise ValueError("hardware maximum-run ratio cannot exceed 1.05")

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "noisy_hardware",
            "warmup_samples": self.warmup_samples,
            "min_samples": self.min_samples,
            "median_ratio": self.median_ratio,
            "max_run_ratio": self.max_run_ratio,
        }


MeasurementPolicy: TypeAlias = ExactCyclePolicy | NoisyHardwarePolicy


@dataclass(frozen=True)
class MetricPromotionCell:
    """Incumbent/challenger measurements and threshold policy for one metric."""

    recommended: MetricMeasurements
    incumbent: MetricMeasurements
    policy: MeasurementPolicy
    objective_bridge: ObjectiveMetricBridge

    def __post_init__(self) -> None:
        if not isinstance(self.recommended, MetricMeasurements):
            raise TypeError("recommended metric evidence must be MetricMeasurements")
        if not isinstance(self.incumbent, MetricMeasurements):
            raise TypeError("incumbent metric evidence must be MetricMeasurements")
        if not isinstance(self.policy, (ExactCyclePolicy, NoisyHardwarePolicy)):
            raise TypeError("metric policy must be an exact-cycle or hardware policy")
        if not isinstance(self.objective_bridge, ObjectiveMetricBridge):
            raise TypeError("metric promotion cell requires an ObjectiveMetricBridge")

    def manifest(self) -> dict[str, object]:
        return {
            "recommended": self.recommended.manifest(),
            "incumbent": self.incumbent.manifest(),
            "policy": self.policy.manifest(),
            "objective_bridge": self.objective_bridge.manifest(),
        }


def _cell_sort_key(cell: MetricPromotionCell) -> tuple[str, str]:
    domain = cell.recommended.objective_domain
    if not isinstance(domain, ScheduleObjectiveDomain):
        return (_type_name(domain), "")
    key = structural_decision_key(domain, path="metric objective domain")
    return (domain.metric, _key_digest(key))


@dataclass(frozen=True)
class PromotionEvidence:
    """All evidence required to promote one exact schedule over another."""

    recommended: ScheduleEvidence
    incumbent: ScheduleEvidence
    metrics: tuple[MetricPromotionCell, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.recommended, ScheduleEvidence):
            raise TypeError("recommended promotion evidence must be ScheduleEvidence")
        if not isinstance(self.incumbent, ScheduleEvidence):
            raise TypeError("incumbent promotion evidence must be ScheduleEvidence")
        metrics = tuple(self.metrics)
        if not all(isinstance(metric, MetricPromotionCell) for metric in metrics):
            raise TypeError("promotion metrics must be MetricPromotionCell records")
        object.__setattr__(self, "metrics", tuple(sorted(metrics, key=_cell_sort_key)))

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "schedule-promotion-evidence-v4",
            "recommended": self.recommended.manifest(),
            "incumbent": self.incumbent.manifest(),
            "metrics": [metric.manifest() for metric in self.metrics],
        }


def _reject(code: str, detail: str) -> PromotionDecision:
    return PromotionDecision(False, f"{code}: {detail}")


def _semantic_rejection(
    recommended: SemanticScope,
    incumbent: SemanticScope,
) -> PromotionDecision | None:
    if recommended.domain_key != incumbent.domain_key:
        return _reject(
            "semantic_domain_mismatch",
            "recommended and incumbent semantic scopes are not comparable",
        )

    weaknesses = []
    if incumbent.complete and not recommended.complete:
        weaknesses.append("complete coverage was lost")
    missing = incumbent.guarantee_keys - recommended.guarantee_keys
    if missing:
        weaknesses.append(f"{len(missing)} incumbent guarantee(s) are missing")
    added = recommended.omission_keys - incumbent.omission_keys
    if added:
        weaknesses.append(f"{len(added)} new omission(s) were introduced")
    if weaknesses:
        return _reject("semantic_scope_weaker", "; ".join(weaknesses))
    return None


def _domain_rejection(
    cell: MetricPromotionCell,
) -> tuple[str, PromotionDecision | None]:
    recommended = cell.recommended.objective_domain
    incumbent = cell.incumbent.objective_domain
    if not isinstance(recommended, ScheduleObjectiveDomain):
        return "unknown", _reject(
            "recommended_objective_domain_untyped",
            "every recommended metric requires ScheduleObjectiveDomain",
        )
    metric = recommended.metric
    if not isinstance(incumbent, ScheduleObjectiveDomain):
        return metric, _reject(
            "incumbent_objective_domain_untyped",
            f"incumbent metric {metric!r} requires ScheduleObjectiveDomain",
        )
    recommended_key = structural_decision_key(
        recommended,
        path=f"recommended objective domain {metric!r}",
    )
    incumbent_key = structural_decision_key(
        incumbent,
        path=f"incumbent objective domain {incumbent.metric!r}",
    )
    if recommended_key != incumbent_key:
        return metric, _reject(
            "objective_domain_mismatch",
            f"metric {metric!r} does not have exactly comparable domains",
        )
    return metric, None


def _warmup_rejection(
    measurements: MetricMeasurements,
    *,
    policy: MeasurementPolicy,
    side: Literal["recommended", "incumbent"],
    metric: str,
) -> PromotionDecision | None:
    required = policy.warmup_samples
    warmup = measurements.warmup
    if required == 0:
        if warmup is not None and warmup.count != 0:
            return _reject(
                f"{side}_warmup_mismatch",
                f"exact metric {metric!r} permits zero warmup samples, got {warmup.count}",
            )
        return None
    if warmup is None:
        return _reject(
            f"{side}_warmup_missing",
            f"metric {metric!r} requires {required} discarded warmup sample(s)",
        )
    if warmup.count != required:
        return _reject(
            f"{side}_warmup_mismatch",
            f"metric {metric!r} requires {required} warmup sample(s), got {warmup.count}",
        )
    if not warmup.discarded:
        return _reject(
            f"{side}_warmup_not_discarded",
            f"metric {metric!r} requires warmup samples to be discarded",
        )
    return None


def _samples_rejection(
    measurements: MetricMeasurements,
    *,
    policy: MeasurementPolicy,
    side: Literal["recommended", "incumbent"],
    metric: str,
) -> PromotionDecision | None:
    count = len(measurements.samples)
    if count == 0:
        return _reject(
            f"{side}_samples_missing",
            f"metric {metric!r} requires at least {policy.min_samples} measurement(s)",
        )
    if count < policy.min_samples:
        return _reject(
            f"{side}_samples_insufficient",
            f"metric {metric!r} has {count} measurement(s); policy requires {policy.min_samples}",
        )
    if isinstance(policy, ExactCyclePolicy) and any(
        type(sample) is not int for sample in measurements.samples
    ):
        return _reject(
            f"{side}_exact_cycles_non_integral",
            f"metric {metric!r} requires integer cycle measurements",
        )
    return None


def _as_decimal(value: int | float) -> Decimal:
    return Decimal(value) if type(value) is int else Decimal(str(value))


def _median(samples: tuple[int | float, ...]) -> Decimal:
    ordered = sorted(_as_decimal(sample) for sample in samples)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _threshold_rejection(
    cell: MetricPromotionCell,
    *,
    metric: str,
) -> PromotionDecision | None:
    recommended_median = _median(cell.recommended.samples)
    incumbent_median = _median(cell.incumbent.samples)
    median_limit = Decimal(str(cell.policy.median_ratio))
    if recommended_median > incumbent_median * median_limit:
        observed = recommended_median / incumbent_median
        return _reject(
            "median_regression",
            f"metric {metric!r} recommended median "
            f"{_decimal_text(recommended_median)} is "
            f"{_decimal_text(observed)}x incumbent median "
            f"{_decimal_text(incumbent_median)}; limit is "
            f"{_decimal_text(median_limit)}x",
        )

    maximum_limit = cell.policy.max_run_ratio
    if maximum_limit is not None:
        recommended_maximum = max(
            _as_decimal(sample) for sample in cell.recommended.samples
        )
        maximum_limit_decimal = Decimal(str(maximum_limit))
        if recommended_maximum > incumbent_median * maximum_limit_decimal:
            observed = recommended_maximum / incumbent_median
            return _reject(
                "maximum_run_regression",
                f"metric {metric!r} recommended maximum run "
                f"{_decimal_text(recommended_maximum)} is "
                f"{_decimal_text(observed)}x incumbent median "
                f"{_decimal_text(incumbent_median)}; limit is "
                f"{_decimal_text(maximum_limit_decimal)}x",
            )
    return None


@dataclass(frozen=True)
class SchedulePromotionGate:
    """Callable promotion gate bound to one immutable evidence manifest."""

    evidence: PromotionEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, PromotionEvidence):
            raise TypeError("schedule promotion gate requires PromotionEvidence")

    def __call__(
        self,
        recommended: ScheduleCandidate[object, object, object, object],
        incumbent: (
            ScheduleCandidate[object, object, object, object]
            | OpaqueScheduleIncumbent[object, object]
        ),
    ) -> PromotionDecision:
        actual_recommended = ScheduleDecisionIdentity.from_schedule(recommended)
        if actual_recommended.structural_key != (
            self.evidence.recommended.identity.structural_key
        ):
            return _reject(
                "recommended_identity_mismatch",
                "evidence binds decision digest "
                f"{self.evidence.recommended.identity.digest}, got "
                f"{actual_recommended.digest}",
            )
        actual_incumbent = ScheduleDecisionIdentity.from_schedule(incumbent)
        if (
            actual_incumbent.structural_key
            != self.evidence.incumbent.identity.structural_key
        ):
            return _reject(
                "incumbent_identity_mismatch",
                "evidence binds decision digest "
                f"{self.evidence.incumbent.identity.digest}, got "
                f"{actual_incumbent.digest}",
            )

        for side, schedule_evidence in (
            ("recommended", self.evidence.recommended),
            ("incumbent", self.evidence.incumbent),
        ):
            correctness = schedule_evidence.correctness
            if not correctness.passed:
                return _reject(
                    f"{side}_correctness_failed",
                    f"{side} schedule did not pass its correctness oracle",
                )
            if not correctness.exact:
                return _reject(
                    f"{side}_correctness_not_exact",
                    f"{side} schedule lacks exact correctness evidence",
                )
        if (
            self.evidence.recommended.correctness.oracle_key
            != self.evidence.incumbent.correctness.oracle_key
        ):
            return _reject(
                "correctness_oracle_mismatch",
                "recommended and incumbent used different exact correctness oracles",
            )

        semantic_rejection = _semantic_rejection(
            self.evidence.recommended.semantic_scope,
            self.evidence.incumbent.semantic_scope,
        )
        if semantic_rejection is not None:
            return semantic_rejection

        if (
            self.evidence.recommended.materialization.platform_key
            != self.evidence.incumbent.materialization.platform_key
        ):
            return _reject(
                "materialization_platform_mismatch",
                "recommended and incumbent were materialized for different platforms",
            )

        for side, schedule_evidence in (
            ("recommended", self.evidence.recommended),
            ("incumbent", self.evidence.incumbent),
        ):
            materialization = schedule_evidence.materialization
            if materialization.scored_key != materialization.emitted_key:
                return _reject(
                    f"{side}_materialization_fingerprint_mismatch",
                    f"{side} scored and emitted materialization fingerprints differ",
                )
            current_schedule = recommended if side == "recommended" else incumbent
            current_key = _current_materialization_key(current_schedule)
            if current_key is None:
                return _reject(
                    f"{side}_materialization_fingerprint_unavailable",
                    f"{side} current materialization has no promotion fingerprint",
                )
            if current_key != materialization.scored_key:
                return _reject(
                    f"{side}_materialization_fingerprint_stale",
                    f"{side} evidence does not bind the current materialization",
                )
            current_platform_key = _current_platform_key(current_schedule)
            if current_platform_key is None:
                return _reject(
                    f"{side}_platform_fingerprint_unavailable",
                    f"{side} current materialization has no platform fingerprint",
                )
            if current_platform_key != materialization.platform_key:
                return _reject(
                    f"{side}_platform_fingerprint_stale",
                    f"{side} evidence does not bind the current platform",
                )

        if isinstance(incumbent, OpaqueScheduleIncumbent):
            return _reject(
                "opaque_incumbent_promotion_unsupported",
                "opaque incumbents require a gate-owned artifact verifier",
            )

        if not self.evidence.metrics:
            return _reject(
                "metric_evidence_missing",
                "promotion requires at least one metric cell",
            )

        search_domain = recommended.objective_domain
        if not isinstance(search_domain, ScheduleObjectiveDomain):
            return _reject(
                "search_objective_domain_untyped",
                "the recommended candidate requires ScheduleObjectiveDomain",
            )

        seen_metrics: set[str] = set()
        search_domain_key = structural_decision_key(
            search_domain,
            path="searched objective domain",
        )
        for cell in self.evidence.metrics:
            metric, domain_rejection = _domain_rejection(cell)
            if domain_rejection is not None:
                return domain_rejection
            metric_domain = cell.recommended.objective_domain
            metric_domain_key = structural_decision_key(
                metric_domain,
                path=f"measurement objective domain {metric!r}",
            )
            bridge = cell.objective_bridge
            if bridge.search_domain_key != search_domain_key:
                return _reject(
                    "search_objective_bridge_mismatch",
                    f"metric {metric!r} bridge does not bind the searched objective",
                )
            if bridge.measurement_domain_key != metric_domain_key:
                return _reject(
                    "measurement_objective_bridge_mismatch",
                    f"metric {metric!r} bridge does not bind its measurement domain",
                )
            if metric_domain.target != search_domain.target:
                return _reject(
                    "measurement_target_mismatch",
                    f"metric {metric!r} target does not match the searched target",
                )
            if metric_domain.scope != search_domain.scope:
                return _reject(
                    "measurement_scope_mismatch",
                    f"metric {metric!r} scope does not match the searched scope",
                )
            if metric in seen_metrics:
                return _reject(
                    "duplicate_metric_evidence",
                    f"metric {metric!r} appears more than once",
                )
            seen_metrics.add(metric)

            for side, measurements in (
                ("recommended", cell.recommended),
                ("incumbent", cell.incumbent),
            ):
                schedule_evidence = (
                    self.evidence.recommended
                    if side == "recommended"
                    else self.evidence.incumbent
                )
                if measurements.platform_key != (
                    schedule_evidence.materialization.platform_key
                ):
                    return _reject(
                        f"{side}_measurement_platform_mismatch",
                        f"metric {metric!r} does not bind the materialized platform",
                    )
                warmup_rejection = _warmup_rejection(
                    measurements,
                    policy=cell.policy,
                    side=side,
                    metric=metric,
                )
                if warmup_rejection is not None:
                    return warmup_rejection
                samples_rejection = _samples_rejection(
                    measurements,
                    policy=cell.policy,
                    side=side,
                    metric=metric,
                )
                if samples_rejection is not None:
                    return samples_rejection

            threshold_rejection = _threshold_rejection(cell, metric=metric)
            if threshold_rejection is not None:
                return threshold_rejection

        return PromotionDecision(True)


def validate_schedule_promotion_gate(value):
    """Accept only an immutable evidence-backed public promotion gate."""

    if value is None:
        return None
    if not isinstance(value, SchedulePromotionGate):
        raise TypeError(
            "promotion_gate must be an evidence-backed SchedulePromotionGate"
        )
    return value


__all__ = [
    "CorrectnessEvidence",
    "ExactCyclePolicy",
    "MaterializationEvidence",
    "MeasurementPolicy",
    "MetricMeasurements",
    "MetricPromotionCell",
    "NoisyHardwarePolicy",
    "ObjectiveMetricBridge",
    "PromotionEvidence",
    "ScheduleDecisionIdentity",
    "ScheduleEvidence",
    "SchedulePromotionGate",
    "SemanticScope",
    "WarmupEvidence",
    "validate_schedule_promotion_gate",
]
