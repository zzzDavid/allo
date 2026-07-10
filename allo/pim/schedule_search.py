"""Deterministic exhaustive search over finite schedule decision domains.

The core deliberately knows nothing about workloads, targets, or cost models.
Callers provide the domains and callbacks that connect search to those layers.
Returned callback objects are retained by identity as candidate-owned artifacts;
the search never deep-copies backend objects.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar


Decisions: TypeAlias = Mapping[str, object]
DomainSource: TypeAlias = Iterable[object] | Callable[[Decisions], Iterable[object]]
ConstraintScope: TypeAlias = Literal["partial", "full"]
SearchTermination: TypeAlias = Literal["exhausted", "truncated"]
RejectionStage: TypeAlias = Literal[
    "incumbent",
    "domain",
    "partial_constraint",
    "full_constraint",
    "build",
    "materialize",
    "score",
]

PayloadT = TypeVar("PayloadT")
MaterializedT = TypeVar("MaterializedT")
ScoreT = TypeVar("ScoreT")
ObjectiveT = TypeVar("ObjectiveT")

_MISSING = object()
_UNSET = object()


class InvalidDecisionValue(TypeError):
    """Raised for mutable or identity-based values in a decision domain."""


class NonFiniteObjective(ValueError):
    """Raised when an objective contains NaN or infinity."""


class InvalidObjectiveValue(TypeError):
    """Raised when an objective is mutable or lacks structural ordering."""


class InvalidProducedAssignment(ValueError):
    """Raised when an external assignment producer violates the search problem."""


def _decision_key(
    value: object,
    *,
    path: str,
    active: set[int] | None = None,
) -> Hashable:
    """Validate one decision-like value and return its structural key."""

    if value is None:
        return (type(None), None)
    if isinstance(value, Enum):
        return ("enum", type(value), value.name)
    if type(value) in (bool, int, str, bytes):
        return (type(value), value)
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidDecisionValue(f"{path} must be finite")
        return (float, value)
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise InvalidDecisionValue(f"{path} must be finite")
        return (complex, value)

    active = active if active is not None else set()
    if type(value) is tuple:
        object_id = id(value)
        if object_id in active:
            raise InvalidDecisionValue(f"{path} contains a cycle")
        active.add(object_id)
        try:
            return (
                "tuple",
                tuple(
                    _decision_key(item, path=f"{path}[{index}]", active=active)
                    for index, item in enumerate(value)
                ),
            )
        finally:
            active.remove(object_id)

    if type(value) is frozenset:
        object_id = id(value)
        if object_id in active:
            raise InvalidDecisionValue(f"{path} contains a cycle")
        active.add(object_id)
        try:
            return (
                "frozenset",
                frozenset(
                    _decision_key(item, path=f"{path} member", active=active)
                    for item in value
                ),
            )
        finally:
            active.remove(object_id)

    if is_dataclass(value) and not isinstance(value, type):
        dataclass_type = type(value)
        if (
            "__dataclass_fields__" not in dataclass_type.__dict__
            or "__dataclass_params__" not in dataclass_type.__dict__
        ):
            raise InvalidDecisionValue(
                f"{path} must be an explicitly frozen dataclass, not an "
                "undecorated subclass"
            )
        parameters = dataclass_type.__dataclass_params__
        if not parameters.frozen or not parameters.eq:
            raise InvalidDecisionValue(
                f"{path} must be a frozen, equality-based dataclass"
            )
        object_id = id(value)
        if object_id in active:
            raise InvalidDecisionValue(f"{path} contains a cycle")
        active.add(object_id)
        try:
            return (
                "dataclass",
                dataclass_type,
                tuple(
                    (
                        field.name,
                        _decision_key(
                            getattr(value, field.name),
                            path=f"{path}.{field.name}",
                            active=active,
                        ),
                    )
                    for field in fields(value)
                ),
            )
        finally:
            active.remove(object_id)

    raise InvalidDecisionValue(
        f"{path} has unsupported mutable or identity-based type "
        f"{type(value).__name__!r}"
    )


def structural_decision_key(
    value: object,
    *,
    path: str = "decision value",
) -> Hashable:
    """Return the validated structural key used by schedule decisions."""

    return _decision_key(value, path=path)


def _freeze_decisions(decisions: Mapping[str, object]) -> Decisions:
    frozen: dict[str, object] = {}
    for name, value in decisions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("decision names must be non-empty strings")
        _decision_key(value, path=f"decision {name!r}")
        frozen[name] = value
    return MappingProxyType(frozen)


def _assignment_key(decisions: Decisions, names: Sequence[str]) -> tuple[Hashable, ...]:
    return tuple(
        _decision_key(decisions[name], path=f"decision {name!r}") for name in names
    )


def _ordered_values(name: str, values: Iterable[object]) -> tuple[object, ...]:
    if isinstance(values, (Mapping, set, frozenset)):
        raise TypeError(
            f"decision domain {name!r} must be an ordered iterable of values"
        )
    ordered = tuple(values)
    seen: dict[Hashable, int] = {}
    for index, value in enumerate(ordered):
        key = _decision_key(value, path=f"decision domain {name!r} value {index}")
        if key in seen:
            raise ValueError(
                f"decision domain {name!r} contains duplicate values at "
                f"indices {seen[key]} and {index}"
            )
        seen[key] = index
    return ordered


def _decision_order_key(value: object) -> tuple[object, ...]:
    """Return a total structural order for an already validated decision value."""

    if value is None:
        return ("none",)
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            value.name,
        )
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if type(value) is float:
        return ("float", value)
    if type(value) is complex:
        return ("complex", value.real, value.imag)
    if type(value) is tuple:
        return ("tuple", tuple(_decision_order_key(item) for item in value))
    if type(value) is frozenset:
        return (
            "frozenset",
            tuple(sorted(_decision_order_key(item) for item in value)),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _decision_order_key(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    raise AssertionError("decision values must be validated before ordering")


def _snapshot_objective(
    value: object,
    *,
    path: str = "objective",
    active: set[int] | None = None,
) -> tuple[object, tuple[object, ...]]:
    """Own and key one immutable, finite, structurally ordered objective."""

    if type(value) is int:
        return value, ("number", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise NonFiniteObjective(f"{path} must be finite, got {value!r}")
        return value, ("number", value)
    if isinstance(value, Real):
        raise InvalidObjectiveValue(
            f"{path} must use an exact int or float, not {type(value).__name__!r}"
        )
    active = active if active is not None else set()
    if type(value) is tuple:
        object_id = id(value)
        if object_id in active:
            raise InvalidObjectiveValue(f"{path} contains a cycle")
        active.add(object_id)
        try:
            snapshots = []
            order_keys = []
            for index, item in enumerate(value):
                snapshot, order_key = _snapshot_objective(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                )
                snapshots.append(snapshot)
                order_keys.append(order_key)
            return tuple(snapshots), ("tuple", tuple(order_keys))
        finally:
            active.remove(object_id)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_type = type(value)
        if (
            "__dataclass_fields__" not in dataclass_type.__dict__
            or "__dataclass_params__" not in dataclass_type.__dict__
        ):
            raise InvalidObjectiveValue(
                f"{path} must be an explicitly frozen ordered dataclass"
            )
        parameters = dataclass_type.__dataclass_params__
        if not parameters.frozen or not parameters.eq or not parameters.order:
            raise InvalidObjectiveValue(
                f"{path} dataclass must set frozen=True, eq=True, and order=True"
            )
        object_id = id(value)
        if object_id in active:
            raise InvalidObjectiveValue(f"{path} contains a cycle")
        active.add(object_id)
        try:
            snapshot_fields = []
            order_keys = []
            for field in fields(value):
                snapshot, order_key = _snapshot_objective(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                    active=active,
                )
                snapshot_fields.append((field.name, snapshot))
                order_keys.append((field.name, order_key))
            snapshot = object.__new__(dataclass_type)
            for name, field_value in snapshot_fields:
                object.__setattr__(snapshot, name, field_value)
            return snapshot, (
                "dataclass",
                dataclass_type.__module__,
                dataclass_type.__qualname__,
                tuple(order_keys),
            )
        finally:
            active.remove(object_id)
    raise InvalidObjectiveValue(
        f"{path} has unsupported mutable or non-structural type "
        f"{type(value).__name__!r}; use finite numbers, tuples, or frozen ordered "
        "dataclasses"
    )


@dataclass(frozen=True)
class DecisionDomain:
    """One named decision and its finite, ordered values.

    A callable source receives an immutable assignment containing only earlier
    decisions. Static iterables and dependent results are snapshotted, deeply
    validated as immutable values, and checked for duplicates.
    """

    name: str
    values: DomainSource

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("decision domain names must be non-empty strings")
        if not callable(self.values):
            object.__setattr__(
                self,
                "values",
                _ordered_values(self.name, self.values),
            )

    def values_for(self, decisions: Decisions) -> tuple[object, ...]:
        """Resolve this domain for one prefix while preserving source order."""

        values = self.values(decisions) if callable(self.values) else self.values
        return _ordered_values(self.name, values)


@dataclass(frozen=True)
class LegalityConstraint:
    """A named legality predicate evaluated on partial or full assignments."""

    name: str
    predicate: Callable[[Decisions], bool]
    scope: ConstraintScope = "full"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("constraint names must be non-empty strings")
        if self.scope not in ("partial", "full"):
            raise ValueError("constraint scope must be 'partial' or 'full'")
        if not callable(self.predicate):
            raise TypeError("constraint predicate must be callable")


@dataclass(frozen=True)
class AssignmentProblem:
    """Finite structural problem exposed to grid or future solver producers."""

    domains: tuple[DecisionDomain, ...]
    constraints: tuple[LegalityConstraint, ...]
    max_complete_assignments: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(self.domains))
        object.__setattr__(self, "constraints", tuple(self.constraints))


@dataclass(frozen=True)
class ProducedAssignments:
    """Complete assignments and honest producer termination status."""

    assignments: tuple[Mapping[str, object], ...]
    termination: SearchTermination = "exhausted"

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        if self.termination not in ("exhausted", "truncated"):
            raise ValueError("producer termination must be exhausted or truncated")
        if not all(isinstance(item, Mapping) for item in self.assignments):
            raise TypeError("produced assignments must be mappings")


AssignmentProducer: TypeAlias = Callable[[AssignmentProblem], ProducedAssignments]


class InfeasibleSchedule(Exception):
    """Expected, candidate-local rejection from build, materialize, or score."""


class ObjectiveDomainMismatch(ValueError):
    """Raised before ranking estimates from unlike objective domains."""

    def __init__(
        self,
        expected: Hashable,
        actual: Hashable,
        decisions: Decisions,
    ):
        self.expected = expected
        self.actual = actual
        self.decisions = _freeze_decisions(decisions)
        super().__init__(f"objective domain {actual!r} does not match {expected!r}")


@dataclass(frozen=True)
class ScheduleObjectiveDomain:
    """Typed provenance required before two schedule objectives may compare."""

    metric: str
    target: str
    target_revision: str
    model_fingerprint: Hashable
    fidelity: str
    scope: str
    unit: str
    direction: Literal["minimize"]

    def __post_init__(self) -> None:
        for name in (
            "metric",
            "target",
            "target_revision",
            "fidelity",
            "scope",
            "unit",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"objective-domain {name} must be non-empty")
        if self.direction != "minimize":
            raise ValueError("objective-domain direction must be 'minimize'")
        _decision_key(
            self.model_fingerprint,
            path="objective-domain model fingerprint",
        )

    @classmethod
    def fingerprinted_target(
        cls,
        *,
        metric: str,
        target: str,
        model_fingerprint: Hashable,
        fidelity: str,
        scope: str,
        unit: str,
        direction: Literal["minimize"],
        target_revision: str | None = None,
    ) -> "ScheduleObjectiveDomain":
        """Build a domain whose target geometry is covered by the model digest."""

        revision = target_revision or f"fingerprinted:{model_fingerprint}"
        return cls(
            metric,
            target,
            revision,
            model_fingerprint,
            fidelity,
            scope,
            unit,
            direction,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "target": self.target,
            "target_revision": self.target_revision,
            "model_fingerprint": self.model_fingerprint,
            "fidelity": self.fidelity,
            "scope": self.scope,
            "unit": self.unit,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ScheduleCandidate(Generic[PayloadT, MaterializedT, ScoreT, ObjectiveT]):
    """One feasible schedule and its candidate-owned callback artifacts.

    The record and decisions are immutable. Payload, materialized, and score
    objects are retained without copying. The objective is independently
    snapshotted into a finite immutable structural value before ranking.
    """

    decisions: Decisions
    payload: PayloadT
    materialized: MaterializedT
    score: ScoreT
    objective: ObjectiveT
    objective_domain: Hashable
    is_incumbent: bool
    enumeration_index: int
    _objective_order_key: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", _freeze_decisions(self.decisions))
        _decision_key(self.objective_domain, path="objective domain")
        objective_snapshot, objective_order_key = _snapshot_objective(self.objective)
        object.__setattr__(self, "objective", objective_snapshot)
        object.__setattr__(self, "_objective_order_key", objective_order_key)
        if type(self.is_incumbent) is not bool:
            raise TypeError("is_incumbent must be a bool")
        if (
            isinstance(self.enumeration_index, bool)
            or not isinstance(self.enumeration_index, int)
            or self.enumeration_index < 0
        ):
            raise ValueError("enumeration_index must be a non-negative integer")

    def validated_objective_order_key(self) -> tuple[object, ...]:
        """Revalidate the retained objective before ranking or reporting."""

        _, current_key = _snapshot_objective(self.objective)
        if current_key != self._objective_order_key:
            raise InvalidObjectiveValue("candidate objective mutated after scoring")
        return current_key


@dataclass(frozen=True)
class Rejection:
    """Diagnostic for one rejected assignment or pruned prefix."""

    decisions: Decisions
    stage: RejectionStage
    reason: str
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", _freeze_decisions(self.decisions))


@dataclass(frozen=True)
class SearchStats:
    """Deterministic counters and explicit search termination state.

    Build, materialize, and score attempts include an explicit incumbent;
    ``complete_assignments_considered`` excludes it.
    """

    partial_assignments: int
    complete_assignments_considered: int
    build_attempts: int
    materialize_attempts: int
    score_attempts: int
    feasible_candidates: int
    pruned_branches: int
    incumbent_evaluated: bool
    complete_assignments_covered: int
    complete_assignments_total: int
    max_complete_assignments: int | None
    termination: SearchTermination
    elapsed_seconds: float = field(compare=False)

    def __post_init__(self) -> None:
        if self.termination not in ("exhausted", "truncated"):
            raise ValueError("termination must be 'exhausted' or 'truncated'")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed search time must be finite and non-negative")
        for name in (
            "complete_assignments_covered",
            "complete_assignments_total",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.complete_assignments_covered > self.complete_assignments_total:
            raise ValueError("covered assignments cannot exceed the finite problem")
        if self.exhaustive and (
            self.complete_assignments_covered != self.complete_assignments_total
        ):
            raise ValueError("exhaustive search must cover the finite problem")

    @property
    def exhaustive(self) -> bool:
        return self.termination == "exhausted"

    @property
    def truncated(self) -> bool:
        return self.termination == "truncated"

    @property
    def candidate_coverage(self) -> float:
        """Fraction of constraint-legal complete assignments evaluated."""

        if self.complete_assignments_total == 0:
            return 1.0 if self.exhaustive else 0.0
        return self.complete_assignments_covered / self.complete_assignments_total


@dataclass(frozen=True)
class SearchResult(Generic[PayloadT, MaterializedT, ScoreT, ObjectiveT]):
    """Feasible candidates in ascending objective order plus diagnostics."""

    ranked: tuple[ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT], ...]
    rejections: tuple[Rejection, ...]
    stats: SearchStats

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranked", tuple(self.ranked))
        object.__setattr__(self, "rejections", tuple(self.rejections))
        if not self.ranked:
            raise ValueError("SearchResult requires at least one feasible candidate")
        incumbents = [candidate for candidate in self.ranked if candidate.is_incumbent]
        if len(incumbents) > 1:
            raise ValueError("SearchResult supports at most one incumbent")
        expected = self.ranked[0].objective_domain
        expected_key = _decision_key(expected, path="objective domain")
        self.ranked[0].validated_objective_order_key()
        for candidate in self.ranked[1:]:
            candidate.validated_objective_order_key()
            if (
                _decision_key(candidate.objective_domain, path="objective domain")
                != expected_key
            ):
                raise ObjectiveDomainMismatch(
                    expected,
                    candidate.objective_domain,
                    candidate.decisions,
                )

    @property
    def best(
        self,
    ) -> ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]:
        return self.ranked[0]

    @property
    def best_incumbent(
        self,
    ) -> ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT] | None:
        return next(
            (candidate for candidate in self.ranked if candidate.is_incumbent),
            None,
        )

    def report(self) -> dict[str, object]:
        """Return auditable candidate coverage, timing, and ranking telemetry."""

        objective_domain = self.best.objective_domain
        if hasattr(objective_domain, "manifest"):
            objective_domain = objective_domain.manifest()
        incumbent = self.best_incumbent
        for candidate in self.ranked:
            candidate.validated_objective_order_key()
        objective_unit = None
        objective_direction = None
        if isinstance(self.best.objective_domain, ScheduleObjectiveDomain):
            objective_unit = self.best.objective_domain.unit
            objective_direction = self.best.objective_domain.direction
        return {
            "termination": self.stats.termination,
            "elapsed_seconds": self.stats.elapsed_seconds,
            "partial_assignments": self.stats.partial_assignments,
            "complete_assignments_considered": (
                self.stats.complete_assignments_considered
            ),
            "complete_assignments_covered": (self.stats.complete_assignments_covered),
            "complete_assignments_total": self.stats.complete_assignments_total,
            "feasible_candidates": self.stats.feasible_candidates,
            "candidate_coverage": self.stats.candidate_coverage,
            "candidate_coverage_unit": (
                "fraction_of_constraint_legal_complete_assignments"
            ),
            "pruned_branches": self.stats.pruned_branches,
            "rejections_by_stage": dict(
                sorted(Counter(item.stage for item in self.rejections).items())
            ),
            "recommended_decisions": dict(self.best.decisions),
            "recommended_objective": self.best.objective,
            "incumbent_rank": (
                None if incumbent is None else self.ranked.index(incumbent)
            ),
            "objective_domain": objective_domain,
            "objective_unit": objective_unit,
            "objective_direction": objective_direction,
        }


@dataclass(frozen=True)
class OpaqueScheduleIncumbent(Generic[PayloadT, MaterializedT]):
    """A frozen incumbent retained outside the generated candidate domain."""

    decisions: Decisions
    payload: PayloadT
    materialized: MaterializedT
    fingerprint: Hashable

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", _freeze_decisions(self.decisions))
        _decision_key(self.fingerprint, path="opaque incumbent fingerprint")


@dataclass(frozen=True)
class PromotionDecision:
    """Result of an external correctness/performance promotion gate."""

    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("promotion eligibility must be a bool")
        if self.eligible and self.reason is not None:
            raise ValueError(
                "eligible promotion decisions cannot have a fallback reason"
            )
        if not self.eligible and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("rejected promotion decisions require a fallback reason")


@dataclass(frozen=True)
class ScheduleActivation(Generic[PayloadT, MaterializedT, ScoreT, ObjectiveT]):
    """Recommended and activated schedules plus fail-closed fallback evidence."""

    search_result: SearchResult[PayloadT, MaterializedT, ScoreT, ObjectiveT]
    recommended: ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]
    incumbent: (
        ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]
        | OpaqueScheduleIncumbent[PayloadT, MaterializedT]
    )
    active: (
        ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]
        | OpaqueScheduleIncumbent[PayloadT, MaterializedT]
    )
    promoted: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if self.recommended is not self.search_result.best:
            raise ValueError("recommended schedule must be the search argmin")
        if type(self.promoted) is not bool:
            raise TypeError("promoted must be a bool")
        if self.active is self.recommended:
            if self.fallback_reason is not None:
                raise ValueError(
                    "activated recommendation cannot have a fallback reason"
                )
        else:
            if self.active is not self.incumbent:
                raise ValueError("active schedule must be recommendation or incumbent")
            if not isinstance(self.fallback_reason, str) or not self.fallback_reason:
                raise ValueError("incumbent fallback requires a reason")
        if self.promoted != (
            self.active is self.recommended and self.recommended is not self.incumbent
        ):
            raise ValueError("promoted must identify an activated challenger")

    @property
    def active_materialized(self) -> MaterializedT:
        return self.active.materialized


class MissingScheduleIncumbent(RuntimeError):
    """Raised when guarded activation has no schedule to fall back to."""


def guarded_schedule_activation(
    search_result: SearchResult[PayloadT, MaterializedT, ScoreT, ObjectiveT],
    *,
    incumbent: (
        ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]
        | OpaqueScheduleIncumbent[PayloadT, MaterializedT]
        | None
    ) = None,
    promotion_gate: (
        Callable[
            [
                ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT],
                ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]
                | OpaqueScheduleIncumbent[PayloadT, MaterializedT],
            ],
            PromotionDecision,
        ]
        | None
    ) = None,
) -> ScheduleActivation[PayloadT, MaterializedT, ScoreT, ObjectiveT]:
    """Activate only a gated challenger; otherwise retain the incumbent."""

    if not isinstance(search_result, SearchResult):
        raise TypeError("guarded activation requires a SearchResult")
    selected_incumbent = incumbent or search_result.best_incumbent
    if selected_incumbent is None:
        raise MissingScheduleIncumbent(
            "guarded activation requires an explicit or searched incumbent"
        )
    if isinstance(selected_incumbent, ScheduleCandidate) and not any(
        candidate is selected_incumbent for candidate in search_result.ranked
    ):
        raise ValueError("searched incumbent must belong to the SearchResult")

    recommended = search_result.best
    if recommended is selected_incumbent:
        return ScheduleActivation(
            search_result,
            recommended,
            selected_incumbent,
            selected_incumbent,
            False,
            None,
        )
    if promotion_gate is None:
        return ScheduleActivation(
            search_result,
            recommended,
            selected_incumbent,
            selected_incumbent,
            False,
            "shadow_only: promotion evidence was not requested",
        )
    decision = promotion_gate(recommended, selected_incumbent)
    if not isinstance(decision, PromotionDecision):
        raise TypeError("promotion gate must return PromotionDecision")
    if decision.eligible:
        return ScheduleActivation(
            search_result,
            recommended,
            selected_incumbent,
            recommended,
            True,
            None,
        )
    return ScheduleActivation(
        search_result,
        recommended,
        selected_incumbent,
        selected_incumbent,
        False,
        decision.reason,
    )


class InfeasibleIncumbent(RuntimeError):
    """Raised when the explicit incumbent cannot be retained as a candidate."""

    def __init__(self, diagnostics: Sequence[Rejection]):
        self.diagnostics = tuple(diagnostics)
        self.rejections = self.diagnostics
        if not self.diagnostics:
            raise ValueError("InfeasibleIncumbent requires diagnostics")
        failure = self.diagnostics[-1]
        super().__init__(
            f"incumbent is infeasible at {failure.stage}: {failure.reason}"
        )


class NoFeasibleSchedule(RuntimeError):
    """Raised when an exhaustive search produced no candidate."""

    message_prefix = "no feasible schedule"

    def __init__(self, stats: SearchStats, rejections: tuple[Rejection, ...]):
        self.stats = stats
        self.rejections = rejections
        super().__init__(
            f"{self.message_prefix} after considering "
            f"{stats.complete_assignments_considered} complete assignment(s)"
        )


class NoFeasibleScheduleInPrefix(NoFeasibleSchedule):
    """Raised when a truncated search prefix produced no candidate."""

    message_prefix = "no feasible schedule in explored prefix"


@dataclass
class _Counters:
    partial_assignments: int = 0
    complete_assignments_considered: int = 0
    build_attempts: int = 0
    materialize_attempts: int = 0
    score_attempts: int = 0
    pruned_branches: int = 0
    termination: SearchTermination = "exhausted"


def grid_search(
    domains: Sequence[DecisionDomain],
    *,
    build: Callable[[Decisions], PayloadT],
    materialize: Callable[[PayloadT], MaterializedT],
    score: Callable[[MaterializedT], ScoreT],
    objective: Callable[[ScoreT], ObjectiveT],
    objective_domain: Hashable | Callable[[ScoreT], Hashable],
    constraints: Sequence[LegalityConstraint] = (),
    incumbent: Mapping[str, object] | None = None,
    max_complete_assignments: int | None = None,
    assignment_producer: AssignmentProducer | None = None,
) -> SearchResult[PayloadT, MaterializedT, ScoreT, ObjectiveT]:
    """Enumerate, realize, score, and rank a finite schedule grid.

    Domains are traversed depth-first in supplied order. An explicit incumbent
    is validated and evaluated first, outside the complete-assignment cap, and
    its ordinary duplicate is skipped. Equal objectives prefer that incumbent,
    then use a structural decision order independent of producer enumeration.
    The cap is checked only when another non-incumbent complete assignment
    exists, so an exact bound is exhaustive.

    ``build`` must return a candidate-owned payload, ``materialize`` a
    candidate-owned realization, and ``score`` a candidate-owned estimate.
    ``objective`` must return finite numbers, immutable tuples, or explicitly
    frozen ordered dataclasses composed from those values. The search owns an
    independent objective snapshot. Materialization always precedes scoring.

    Only :class:`InfeasibleSchedule` is converted into an ordinary rejection.
    Other callback exceptions propagate as programming or infrastructure errors.
    """

    started_at = time.perf_counter()

    ordered_domains = tuple(domains)
    ordered_constraints = tuple(constraints)
    if not all(isinstance(domain, DecisionDomain) for domain in ordered_domains):
        raise TypeError("domains must contain DecisionDomain instances")
    if not all(
        isinstance(constraint, LegalityConstraint) for constraint in ordered_constraints
    ):
        raise TypeError("constraints must contain LegalityConstraint instances")

    domain_names = tuple(domain.name for domain in ordered_domains)
    if len(domain_names) != len(set(domain_names)):
        raise ValueError("decision domain names must be unique")
    constraint_names = tuple(constraint.name for constraint in ordered_constraints)
    if len(constraint_names) != len(set(constraint_names)):
        raise ValueError("constraint names must be unique")
    if max_complete_assignments is not None:
        if isinstance(max_complete_assignments, bool) or not isinstance(
            max_complete_assignments, int
        ):
            raise TypeError("max_complete_assignments must be an integer or None")
        if max_complete_assignments < 0:
            raise ValueError("max_complete_assignments must be non-negative")
    if incumbent is not None and not isinstance(incumbent, Mapping):
        raise TypeError("incumbent must be a decision mapping or None")
    if assignment_producer is not None and not callable(assignment_producer):
        raise TypeError("assignment_producer must be callable or None")
    if not callable(objective_domain):
        _decision_key(objective_domain, path="objective domain")

    partial_constraints = tuple(
        constraint
        for constraint in ordered_constraints
        if constraint.scope == "partial"
    )
    full_constraints = tuple(
        constraint for constraint in ordered_constraints if constraint.scope == "full"
    )
    assignment: dict[str, object] = {}
    candidates: list[ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT]] = (
        []
    )
    rejections: list[Rejection] = []
    counters = _Counters()
    expected_objective_domain: Hashable | object = _UNSET
    expected_objective_domain_key: Hashable | object = _UNSET
    domain_cache: dict[tuple[int, tuple[Hashable, ...]], tuple[object, ...]] = {}
    constraint_cache: dict[
        tuple[str, tuple[Hashable, ...]],
        bool,
    ] = {}
    evaluated_assignment_keys: set[tuple[Hashable, ...]] = set()

    def rejection(
        decisions: Decisions,
        stage: RejectionStage,
        reason: str,
        *,
        name: str | None = None,
    ) -> Rejection:
        return Rejection(decisions, stage, reason, name=name)

    def constraint_rejection(
        constraint: LegalityConstraint,
        decisions: Decisions,
    ) -> Rejection:
        stage: RejectionStage = (
            "partial_constraint" if constraint.scope == "partial" else "full_constraint"
        )
        return rejection(
            decisions,
            stage,
            "constraint returned false",
            name=constraint.name,
        )

    def domain_values(domain_index: int, prefix: Decisions) -> tuple[object, ...]:
        key = (domain_index, _assignment_key(prefix, domain_names[:domain_index]))
        if key not in domain_cache:
            domain_cache[key] = ordered_domains[domain_index].values_for(prefix)
        return domain_cache[key]

    def constraint_holds(
        constraint: LegalityConstraint,
        decisions: Decisions,
    ) -> bool:
        decision_names = domain_names[: len(decisions)]
        key = (constraint.name, _assignment_key(decisions, decision_names))
        if key not in constraint_cache:
            constraint_cache[key] = bool(constraint.predicate(decisions))
        return constraint_cache[key]

    def constraint_legal_assignment_keys() -> frozenset[tuple[Hashable, ...]]:
        """Enumerate the finite structural problem without backend callbacks."""

        legal: set[tuple[Hashable, ...]] = set()
        proof_assignment: dict[str, object] = {}

        def visit_problem(domain_index: int) -> None:
            if domain_index == len(ordered_domains):
                decisions = _freeze_decisions(proof_assignment)
                if all(
                    constraint_holds(constraint, decisions)
                    for constraint in full_constraints
                ):
                    legal.add(_assignment_key(decisions, domain_names))
                return

            domain = ordered_domains[domain_index]
            prefix = _freeze_decisions(proof_assignment)
            for value in domain_values(domain_index, prefix):
                proof_assignment[domain.name] = value
                decisions = _freeze_decisions(proof_assignment)
                if all(
                    constraint_holds(constraint, decisions)
                    for constraint in partial_constraints
                ):
                    visit_problem(domain_index + 1)
                del proof_assignment[domain.name]

        visit_problem(0)
        return frozenset(legal)

    def register_objective_domain(
        estimate: ScoreT,
        decisions: Decisions,
    ) -> Hashable:
        nonlocal expected_objective_domain, expected_objective_domain_key

        candidate_domain = (
            objective_domain(estimate)
            if callable(objective_domain)
            else objective_domain
        )
        candidate_key = _decision_key(candidate_domain, path="objective domain")
        if expected_objective_domain is _UNSET:
            expected_objective_domain = candidate_domain
            expected_objective_domain_key = candidate_key
        elif candidate_key != expected_objective_domain_key:
            raise ObjectiveDomainMismatch(
                expected_objective_domain,
                candidate_domain,
                decisions,
            )
        return candidate_domain

    def evaluate_candidate(
        decisions: Decisions,
        *,
        is_incumbent: bool,
        enumeration_index: int,
    ) -> tuple[
        ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT] | None,
        Rejection | None,
    ]:
        for constraint in full_constraints:
            if not constraint_holds(constraint, decisions):
                return None, constraint_rejection(constraint, decisions)

        counters.build_attempts += 1
        try:
            payload = build(decisions)
        except InfeasibleSchedule as error:
            return None, rejection(decisions, "build", str(error))

        counters.materialize_attempts += 1
        try:
            realized = materialize(payload)
        except InfeasibleSchedule as error:
            return None, rejection(decisions, "materialize", str(error))

        counters.score_attempts += 1
        try:
            estimate = score(realized)
        except InfeasibleSchedule as error:
            return None, rejection(decisions, "score", str(error))

        objective_value, _ = _snapshot_objective(objective(estimate))
        candidate_domain = register_objective_domain(estimate, decisions)
        return (
            ScheduleCandidate(
                decisions=decisions,
                payload=payload,
                materialized=realized,
                score=estimate,
                objective=objective_value,
                objective_domain=candidate_domain,
                is_incumbent=is_incumbent,
                enumeration_index=enumeration_index,
            ),
            None,
        )

    def fail_incumbent(failure: Rejection) -> None:
        raise InfeasibleIncumbent((failure,))

    def validate_incumbent(raw_incumbent: Mapping[str, object]) -> Decisions:
        provided = dict(raw_incumbent)
        provided_names = tuple(provided)
        if set(provided_names) != set(domain_names) or len(provided_names) != len(
            domain_names
        ):
            missing = [name for name in domain_names if name not in provided]
            extra = [name for name in provided_names if name not in domain_names]
            fail_incumbent(
                rejection(
                    {},
                    "incumbent",
                    f"decision names must exactly match domains; "
                    f"missing={missing!r}, extra={extra!r}",
                )
            )

        selected: dict[str, object] = {}
        for domain_index, domain in enumerate(ordered_domains):
            prefix = _freeze_decisions(selected)
            try:
                requested_key = _decision_key(
                    provided[domain.name],
                    path=f"incumbent decision {domain.name!r}",
                )
            except InvalidDecisionValue as error:
                fail_incumbent(
                    rejection(prefix, "domain", str(error), name=domain.name)
                )
            match = next(
                (
                    value
                    for value in domain_values(domain_index, prefix)
                    if _decision_key(
                        value,
                        path=f"decision domain {domain.name!r} value",
                    )
                    == requested_key
                ),
                _MISSING,
            )
            if match is _MISSING:
                fail_incumbent(
                    rejection(
                        prefix,
                        "domain",
                        f"value {provided[domain.name]!r} is not in the dependent domain",
                        name=domain.name,
                    )
                )
            selected[domain.name] = match
            decisions = _freeze_decisions(selected)
            for constraint in partial_constraints:
                if not constraint_holds(constraint, decisions):
                    fail_incumbent(constraint_rejection(constraint, decisions))
        return _freeze_decisions(selected)

    incumbent_decisions: Decisions | None = None
    incumbent_key: tuple[Hashable, ...] | None = None
    if incumbent is not None:
        incumbent_decisions = validate_incumbent(incumbent)
        incumbent_candidate, incumbent_failure = evaluate_candidate(
            incumbent_decisions,
            is_incumbent=True,
            enumeration_index=0,
        )
        if incumbent_failure is not None:
            fail_incumbent(incumbent_failure)
        assert incumbent_candidate is not None
        candidates.append(incumbent_candidate)
        incumbent_key = _assignment_key(incumbent_decisions, domain_names)
        evaluated_assignment_keys.add(incumbent_key)

    enumeration_offset = 1 if incumbent_decisions is not None else 0

    def evaluate_exploration_assignment(decisions: Decisions) -> bool:
        if (
            incumbent_key is not None
            and _assignment_key(decisions, domain_names) == incumbent_key
        ):
            return False
        if (
            max_complete_assignments is not None
            and counters.complete_assignments_considered >= max_complete_assignments
        ):
            counters.termination = "truncated"
            return True

        counters.complete_assignments_considered += 1
        evaluated_assignment_keys.add(_assignment_key(decisions, domain_names))
        enumeration_index = (
            enumeration_offset + counters.complete_assignments_considered - 1
        )
        candidate, failure = evaluate_candidate(
            decisions,
            is_incumbent=False,
            enumeration_index=enumeration_index,
        )
        if failure is not None:
            rejections.append(failure)
        else:
            assert candidate is not None
            candidates.append(candidate)
        return False

    def visit(domain_index: int) -> bool:
        if domain_index == len(ordered_domains):
            return evaluate_exploration_assignment(_freeze_decisions(assignment))

        domain = ordered_domains[domain_index]
        prefix = _freeze_decisions(assignment)
        values = domain_values(domain_index, prefix)
        if not values:
            counters.pruned_branches += 1
            rejections.append(
                rejection(
                    prefix,
                    "domain",
                    "domain produced no values",
                    name=domain.name,
                )
            )
            return False

        for value in values:
            assignment[domain.name] = value
            counters.partial_assignments += 1
            decisions = _freeze_decisions(assignment)
            failed = next(
                (
                    constraint
                    for constraint in partial_constraints
                    if not constraint_holds(constraint, decisions)
                ),
                None,
            )
            if failed is not None:
                counters.pruned_branches += 1
                rejections.append(constraint_rejection(failed, decisions))
            elif visit(domain_index + 1):
                del assignment[domain.name]
                return True
            del assignment[domain.name]
        return False

    def normalize_produced_assignment(raw_decisions, produced_index):
        provided = dict(raw_decisions)
        if set(provided) != set(domain_names) or len(provided) != len(domain_names):
            raise InvalidProducedAssignment(
                f"produced assignment {produced_index} must exactly match "
                f"decision domains {domain_names!r}"
            )
        selected = {}
        for domain_index, domain in enumerate(ordered_domains):
            prefix = _freeze_decisions(selected)
            requested_key = _decision_key(
                provided[domain.name],
                path=(
                    f"produced assignment {produced_index} decision " f"{domain.name!r}"
                ),
            )
            match = next(
                (
                    value
                    for value in domain_values(domain_index, prefix)
                    if _decision_key(
                        value,
                        path=f"decision domain {domain.name!r} value",
                    )
                    == requested_key
                ),
                _MISSING,
            )
            if match is _MISSING:
                raise InvalidProducedAssignment(
                    f"produced assignment {produced_index} value "
                    f"{provided[domain.name]!r} is outside dependent domain "
                    f"{domain.name!r}"
                )
            selected[domain.name] = match
            counters.partial_assignments += 1
            decisions = _freeze_decisions(selected)
            failed = next(
                (
                    constraint
                    for constraint in partial_constraints
                    if not constraint_holds(constraint, decisions)
                ),
                None,
            )
            if failed is not None:
                counters.pruned_branches += 1
                rejections.append(constraint_rejection(failed, decisions))
                return None
        return _freeze_decisions(selected)

    if assignment_producer is None:
        visit(0)
    else:
        produced = assignment_producer(
            AssignmentProblem(
                ordered_domains,
                ordered_constraints,
                max_complete_assignments,
            )
        )
        if not isinstance(produced, ProducedAssignments):
            raise TypeError("assignment_producer must return ProducedAssignments")
        counters.termination = produced.termination
        seen_assignments = set()
        for produced_index, raw_decisions in enumerate(produced.assignments):
            decisions = normalize_produced_assignment(raw_decisions, produced_index)
            if decisions is None:
                continue
            key = _assignment_key(decisions, domain_names)
            if key in seen_assignments:
                raise InvalidProducedAssignment(
                    f"produced assignment {produced_index} duplicates an earlier "
                    "assignment"
                )
            seen_assignments.add(key)
            if evaluate_exploration_assignment(decisions):
                break

    problem_assignment_keys = constraint_legal_assignment_keys()
    covered_assignment_keys = problem_assignment_keys.intersection(
        evaluated_assignment_keys
    )
    if (
        assignment_producer is not None
        and counters.termination == "exhausted"
        and covered_assignment_keys != problem_assignment_keys
    ):
        missing_count = len(problem_assignment_keys - covered_assignment_keys)
        raise InvalidProducedAssignment(
            "assignment producer claimed exhausted but omitted "
            f"{missing_count} of {len(problem_assignment_keys)} "
            "constraint-legal complete assignments"
        )
    stats = SearchStats(
        partial_assignments=counters.partial_assignments,
        complete_assignments_considered=counters.complete_assignments_considered,
        build_attempts=counters.build_attempts,
        materialize_attempts=counters.materialize_attempts,
        score_attempts=counters.score_attempts,
        feasible_candidates=len(candidates),
        pruned_branches=counters.pruned_branches,
        incumbent_evaluated=incumbent_decisions is not None,
        complete_assignments_covered=len(covered_assignment_keys),
        complete_assignments_total=len(problem_assignment_keys),
        max_complete_assignments=max_complete_assignments,
        termination=counters.termination,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    rejection_tuple = tuple(rejections)
    if not candidates:
        error_type = (
            NoFeasibleScheduleInPrefix if stats.truncated else NoFeasibleSchedule
        )
        raise error_type(stats, rejection_tuple)

    def challenger_tie_key(
        candidate: ScheduleCandidate[PayloadT, MaterializedT, ScoreT, ObjectiveT],
    ) -> object:
        if assignment_producer is None:
            return candidate.enumeration_index
        return tuple(
            _decision_order_key(candidate.decisions[name]) for name in domain_names
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.validated_objective_order_key(),
            not candidate.is_incumbent,
            challenger_tie_key(candidate),
        )
    )
    return SearchResult(tuple(candidates), rejection_tuple, stats)


__all__ = [
    "AssignmentProblem",
    "AssignmentProducer",
    "ConstraintScope",
    "DecisionDomain",
    "Decisions",
    "InfeasibleIncumbent",
    "InfeasibleSchedule",
    "InvalidDecisionValue",
    "InvalidObjectiveValue",
    "InvalidProducedAssignment",
    "LegalityConstraint",
    "MissingScheduleIncumbent",
    "NoFeasibleSchedule",
    "NoFeasibleScheduleInPrefix",
    "NonFiniteObjective",
    "OpaqueScheduleIncumbent",
    "ObjectiveDomainMismatch",
    "PromotionDecision",
    "ProducedAssignments",
    "Rejection",
    "RejectionStage",
    "ScheduleActivation",
    "ScheduleCandidate",
    "ScheduleObjectiveDomain",
    "SearchResult",
    "SearchStats",
    "SearchTermination",
    "guarded_schedule_activation",
    "grid_search",
    "structural_decision_key",
]
