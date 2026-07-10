"""Standalone shadow search for APUv1 outer profile-row composition.

This module is deliberately not imported by the public compiler.  ``row_tile``
is the logical row extent of one single-APUC profile artifact; it is not the
inner VR tile recorded by :class:`APUV1PlanDecision`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace

from ..perf import BoundCostSpec, CostSpec
from . import apu_v1_vector_program as _vector_program
from .apu_v1_vector_codegen import VR_LANES
from .apu_v1_vector_cost import estimate_apu_v1_realization
from .apu_v1_vector_program import APUV1PlanDecision
from .apu_v1_vectorize import (
    ContractionAnalysis,
    IllegalContractionError,
    ValueAccess,
    generate_apu_v1_vectorization_candidates,
)
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    MissingScheduleIncumbent,
    OpaqueScheduleIncumbent,
    ScheduleCandidate,
    ScheduleObjectiveDomain,
    SearchResult,
    grid_search,
    guarded_schedule_activation,
)


_ACCUMULATOR_BLOCKS = (1, 2, 4, 8)
_PLANNER_ACCUMULATOR_BLOCK_LIMIT = 8


class UnsupportedAPUV1ProfileSearchError(ValueError):
    """The retained contraction is outside this exact composition domain."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class APUV1ProfileShape:
    """Name-free matrix-contraction shape in M, N, K order."""

    rows: int
    columns: int
    reduction: int

    def __post_init__(self) -> None:
        for label in ("rows", "columns", "reduction"):
            _positive_int(getattr(self, label), label)

    @property
    def M(self) -> int:
        return self.rows

    @property
    def N(self) -> int:
        return self.columns

    @property
    def K(self) -> int:
        return self.reduction

    def manifest(self) -> dict[str, int]:
        return {"M": self.rows, "N": self.columns, "K": self.reduction}


@dataclass(frozen=True)
class APUV1ProfileTargetProvenance:
    target: str
    vr_lanes: int
    writable_vrs: int
    accumulator_block_limit: int
    fingerprint: str


@dataclass(frozen=True)
class APUV1ProfileCostProvenance:
    fingerprint: str


@dataclass(frozen=True)
class APUV1ProfileRowDecision:
    """Outer row shard plus its dependent, name-free physical plan."""

    row_tile: int
    plan: APUV1PlanDecision

    def __post_init__(self) -> None:
        _positive_int(self.row_tile, "row_tile")
        if not isinstance(self.plan, APUV1PlanDecision):
            raise TypeError("profile plan must be an APUV1PlanDecision")


@dataclass(frozen=True)
class APUV1ProfileRowMaterialization:
    """Frozen physical evidence for one whole-profile composition."""

    decision: APUV1ProfileRowDecision
    logical_shape: APUV1ProfileShape
    shard_shape: APUV1ProfileShape
    final_partial_shape: APUV1ProfileShape | None
    row_waves: int
    column_repetitions: int
    semantic_fingerprint: str
    plan_fingerprint: str
    emitted_source_fingerprint: str
    structural_source_fingerprint: str
    plan_emitted_source_fingerprint: str
    final_plan_fingerprint: str | None
    final_emitted_source_fingerprint: str | None
    final_structural_source_fingerprint: str | None
    final_plan_emitted_source_fingerprint: str | None
    target_provenance: APUV1ProfileTargetProvenance
    cost_provenance: APUV1ProfileCostProvenance
    objective_domain: ScheduleObjectiveDomain
    realization: object = field(repr=False, compare=False)
    final_realization: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.shard_shape.rows != self.decision.row_tile:
            raise ValueError("shard shape does not match outer row_tile")
        if self.row_waves != math.ceil(self.logical_shape.rows / self.shard_shape.rows):
            raise ValueError("row_waves does not exactly cover logical M")
        if (
            self.shard_shape.columns * self.column_repetitions
            != self.logical_shape.columns
        ):
            raise ValueError("column repetitions do not exactly cover logical N")
        if self.plan_fingerprint != self.decision.plan.physical_fingerprint:
            raise ValueError("materialized plan fingerprint changed")
        expected = _digest(
            {
                "plan": self.plan_fingerprint,
                "emitted_source": self.emitted_source_fingerprint,
            }
        )
        if self.plan_emitted_source_fingerprint != expected:
            raise ValueError("plan/emitted-source fingerprint is inconsistent")
        if (
            _emitted_source_fingerprint(self.realization)
            != self.emitted_source_fingerprint
        ):
            raise ValueError("emitted-source fingerprint does not bind the realization")
        partial_fields = (
            self.final_plan_fingerprint,
            self.final_emitted_source_fingerprint,
            self.final_structural_source_fingerprint,
            self.final_plan_emitted_source_fingerprint,
            self.final_realization,
        )
        if self.final_partial_shape is None and any(
            value is not None for value in partial_fields
        ):
            raise ValueError("full waves cannot retain partial-wave evidence")
        if self.final_partial_shape is not None and any(
            value is None for value in partial_fields
        ):
            raise ValueError("partial waves require exact materialization evidence")
        if self.final_realization is not None:
            expected_final = _digest(
                {
                    "plan": self.final_plan_fingerprint,
                    "emitted_source": self.final_emitted_source_fingerprint,
                }
            )
            if self.final_plan_emitted_source_fingerprint != expected_final:
                raise ValueError(
                    "partial plan/emitted-source fingerprint is inconsistent"
                )
            if (
                _emitted_source_fingerprint(self.final_realization)
                != self.final_emitted_source_fingerprint
            ):
                raise ValueError(
                    "partial emitted-source fingerprint does not bind the realization"
                )

    @property
    def promotion_materialization_fingerprint(self) -> str | None:
        runtime_materialization = self.realization.promotion_materialization_fingerprint
        final_runtime_materialization = (
            None
            if self.final_realization is None
            else self.final_realization.promotion_materialization_fingerprint
        )
        if runtime_materialization is None or (
            self.final_realization is not None and final_runtime_materialization is None
        ):
            return None
        return _digest(
            {
                "kind": "apu-v1-whole-profile-composition-v3",
                "logical_shape": self.logical_shape.manifest(),
                "shard_shape": self.shard_shape.manifest(),
                "final_partial_shape": (
                    None
                    if self.final_partial_shape is None
                    else self.final_partial_shape.manifest()
                ),
                "row_waves": self.row_waves,
                "column_repetitions": self.column_repetitions,
                "semantic_fingerprint": self.semantic_fingerprint,
                "emitted_source_fingerprint": self.emitted_source_fingerprint,
                "structural_source_fingerprint": self.structural_source_fingerprint,
                "plan_emitted_source_fingerprint": (
                    self.plan_emitted_source_fingerprint
                ),
                "runtime_materialization_fingerprint": (runtime_materialization),
                "runtime_source_fingerprint": self.runtime_source_fingerprint,
                "final_emitted_source_fingerprint": (
                    self.final_emitted_source_fingerprint
                ),
                "final_structural_source_fingerprint": (
                    self.final_structural_source_fingerprint
                ),
                "final_plan_emitted_source_fingerprint": (
                    self.final_plan_emitted_source_fingerprint
                ),
                "final_runtime_materialization_fingerprint": (
                    final_runtime_materialization
                ),
                "final_runtime_source_fingerprint": (
                    self.final_runtime_source_fingerprint
                ),
                "target": {
                    "name": self.target_provenance.target,
                    "vr_lanes": self.target_provenance.vr_lanes,
                    "writable_vrs": self.target_provenance.writable_vrs,
                    "accumulator_block_limit": (
                        self.target_provenance.accumulator_block_limit
                    ),
                    "fingerprint": self.target_provenance.fingerprint,
                },
                "cost": {"fingerprint": self.cost_provenance.fingerprint},
                "objective_domain": self.objective_domain.manifest(),
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.promotion_materialization_fingerprint

    @property
    def runtime_source_fingerprint(self) -> str:
        artifact = getattr(self.realization, "runtime_artifact", None)
        if artifact is None:
            raise ValueError("profile realization has no complete runtime artifact")
        return artifact.source_fingerprint

    @property
    def final_runtime_source_fingerprint(self) -> str | None:
        if self.final_realization is None:
            return None
        artifact = getattr(self.final_realization, "runtime_artifact", None)
        if artifact is None:
            raise ValueError(
                "partial profile realization has no complete runtime artifact"
            )
        return artifact.source_fingerprint

    @property
    def promotion_platform_fingerprint(self):
        platforms = [self.realization.promotion_platform_fingerprint]
        if self.final_realization is not None:
            platforms.append(self.final_realization.promotion_platform_fingerprint)
        if any(platform is None for platform in platforms):
            return None
        if len(set(platforms)) != 1:
            return None
        return platforms[0]


@dataclass(frozen=True)
class APUV1ProfileCompositionScore:
    """Analytical cycles for the complete repeated profile composition."""

    shard_cycles: int
    row_waves: int
    column_repetitions: int
    composed_cycles: int
    objective_domain: ScheduleObjectiveDomain
    final_shard_cycles: int | None = None

    def __post_init__(self) -> None:
        for label in ("shard_cycles", "row_waves", "column_repetitions"):
            _positive_int(getattr(self, label), label)
        if self.final_shard_cycles is not None:
            _positive_int(self.final_shard_cycles, "final_shard_cycles")
        full_waves = self.row_waves - (self.final_shard_cycles is not None)
        expected = (
            self.shard_cycles * full_waves + (self.final_shard_cycles or 0)
        ) * self.column_repetitions
        if self.composed_cycles != expected:
            raise ValueError("composed cycles do not match shard repetitions")

    @property
    def cycles(self) -> int:
        return self.composed_cycles


@dataclass(frozen=True)
class APUV1ProfileRowSearchResult(SearchResult):
    """A SearchResult that also retains an exact or opaque frozen incumbent."""

    retained_incumbent: object

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.retained_incumbent, ScheduleCandidate):
            if not any(item is self.retained_incumbent for item in self.ranked):
                raise ValueError("searched incumbent is absent from ranked results")
        elif not isinstance(self.retained_incumbent, OpaqueScheduleIncumbent):
            raise TypeError("retained incumbent must be searched or explicitly opaque")

    @property
    def best_incumbent(self):
        return self.retained_incumbent

    @property
    def activation(self):
        """Return the mandatory no-evidence, shadow-only activation."""

        return guarded_schedule_activation(self)


def _bound_cost(target, cost) -> BoundCostSpec:
    if isinstance(cost, BoundCostSpec):
        if cost.target is not target:
            raise TypeError("APUv1 profile cost must be bound to the target")
        return cost
    if isinstance(cost, CostSpec):
        return cost.bind(target)
    raise TypeError("APUv1 profile search requires a CostSpec or BoundCostSpec")


def _target_provenance(target) -> APUV1ProfileTargetProvenance:
    if getattr(target, "name", None) != "apu_v1":
        raise UnsupportedAPUV1ProfileSearchError(
            "profile-row search requires the apu_v1 target"
        )
    walk = getattr(target, "_walk", None)
    units = tuple(walk()) if callable(walk) else ()
    registers = []
    for unit in units:
        for name, register in getattr(unit, "registers", {}).items():
            if re.fullmatch(r"vr\d+", name):
                registers.append(register)
    if not registers:
        raise UnsupportedAPUV1ProfileSearchError(
            "target does not expose writable APUv1 VR geometry"
        )
    lanes = {int(register.lanes) for register in registers}
    widths = {int(register.width) for register in registers}
    if lanes != {VR_LANES} or widths != {16}:
        raise UnsupportedAPUV1ProfileSearchError(
            "target VR geometry disagrees with the retained APUv1 realizer"
        )
    block_limit = max(
        block
        for block in (1, 2, 4, 8)
        if block <= min(len(registers), _PLANNER_ACCUMULATOR_BLOCK_LIMIT)
    )
    manifest = {
        "kind": "apu-v1-profile-target-v1",
        "target": target.name,
        "vr_lanes": VR_LANES,
        "writable_vrs": len(registers),
        "accumulator_block_limit": block_limit,
    }
    return APUV1ProfileTargetProvenance(
        target.name,
        VR_LANES,
        len(registers),
        block_limit,
        _digest(manifest),
    )


def _validate_analysis(analysis) -> APUV1ProfileShape:
    if not isinstance(analysis, ContractionAnalysis):
        raise TypeError("profile-row search requires a ContractionAnalysis")
    if len(analysis.output_axes) != 2 or analysis.parallel_axes != analysis.output_axes:
        raise UnsupportedAPUV1ProfileSearchError(
            "profile-row search requires exactly two parallel output axes"
        )
    extents = analysis.axis_extents
    row_axis, column_axis = analysis.output_axes
    if analysis.reduction_axis not in extents:
        raise UnsupportedAPUV1ProfileSearchError("reduction axis is not retained")
    for axis in analysis.axes:
        if axis.lower_bound != 0 or axis.step != 1 or axis.upper_bound != axis.extent:
            raise UnsupportedAPUV1ProfileSearchError(
                "profile-row search requires zero-based unit-step static axes"
            )
    supported = {
        ("arith.mulf", "arith.addf", "f16"),
        ("arith.mulf", "arith.addf", "bf16"),
        ("arith.muli", "arith.addi", "ui16"),
        ("arith.andi", "arith.addi", "ui16"),
        ("arith.ori", "arith.addi", "ui16"),
        ("arith.xori", "arith.addi", "ui16"),
        ("allo.xnor_popcount", "arith.addi", "ui16"),
    }
    semantics = (
        analysis.multiply_operation,
        analysis.combine_operation,
        analysis.numeric_type,
    )
    if semantics not in supported or (
        analysis.product_coefficient,
        analysis.accumulator_coefficient,
    ) != (1, 1):
        raise UnsupportedAPUV1ProfileSearchError(
            "dtype or contraction semantics have no exact APUv1 realization"
        )
    accesses = (analysis.lhs, analysis.rhs, analysis.accumulator, analysis.output)
    for access in accesses:
        if len(access.indices) != len(access.shape) or len(set(access.indices)) != len(
            access.indices
        ):
            raise UnsupportedAPUV1ProfileSearchError(
                "profile accesses must be dense, ranked axis projections"
            )
        for axis_name, extent in zip(access.indices, access.shape):
            if axis_name not in extents or int(extent) != int(extents[axis_name]):
                raise UnsupportedAPUV1ProfileSearchError(
                    "profile access shape disagrees with retained logical axes"
                )
    if (
        analysis.accumulator.value != analysis.output.value
        or analysis.accumulator.indices != analysis.output.indices
        or analysis.lhs.value in {analysis.output.value, analysis.rhs.value}
        or analysis.rhs.value == analysis.output.value
    ):
        raise UnsupportedAPUV1ProfileSearchError(
            "profile search requires distinct operands and one in-place output"
        )
    return APUV1ProfileShape(
        extents[row_axis], extents[column_axis], extents[analysis.reduction_axis]
    )


def _semantic_fingerprint(analysis, shape) -> str:
    row_axis, column_axis = analysis.output_axes
    roles = {
        row_axis: "parallel_output_0",
        column_axis: "parallel_output_1",
        analysis.reduction_axis: "reduction_0",
    }

    def access_manifest(access: ValueAccess):
        return {
            "axes": tuple(roles[axis] for axis in access.indices),
            "shape": tuple(access.shape),
            "dtype": access.dtype,
            "mode": access.mode,
        }

    return _digest(
        {
            "kind": "apu-v1-dense-profile-semantics-v1",
            "shape": shape.manifest(),
            "numeric_type": analysis.numeric_type,
            "multiply": analysis.multiply_operation,
            "combine": analysis.combine_operation,
            "packed_word_bits": analysis.packed_word_bits,
            "coefficients": (
                analysis.product_coefficient,
                analysis.accumulator_coefficient,
            ),
            "lhs": access_manifest(analysis.lhs),
            "rhs": access_manifest(analysis.rhs),
            "accumulator": access_manifest(analysis.accumulator),
            "output": access_manifest(analysis.output),
        }
    )


def _clone_analysis(analysis, rows: int, columns: int, *, canonical: bool):
    row_axis, column_axis = analysis.output_axes
    replacement_extents = {row_axis: rows, column_axis: columns}
    axis_names = {
        row_axis: "parallel_output_0",
        column_axis: "parallel_output_1",
        analysis.reduction_axis: "reduction_0",
    }

    def clone_access(access, value):
        shape = tuple(
            replacement_extents.get(axis_name, extent)
            for axis_name, extent in zip(access.indices, access.shape)
        )
        indices = tuple(
            axis_names[axis_name] if canonical else axis_name
            for axis_name in access.indices
        )
        return replace(access, value=value, indices=indices, shape=shape)

    values = (
        ("operand_0", "operand_1", "output_0")
        if canonical
        else (analysis.lhs.value, analysis.rhs.value, analysis.output.value)
    )
    axes = tuple(
        replace(
            axis,
            name=axis_names[axis.name] if canonical else axis.name,
            extent=replacement_extents.get(axis.name, axis.extent),
            upper_bound=replacement_extents.get(axis.name, axis.extent),
            ssa_name=f"%axis_{index}" if canonical else axis.ssa_name,
        )
        for index, axis in enumerate(analysis.axes)
    )
    return replace(
        analysis,
        function="profile_composition" if canonical else analysis.function,
        axes=axes,
        output_axes=(
            tuple(axis_names[name] for name in analysis.output_axes)
            if canonical
            else analysis.output_axes
        ),
        parallel_axes=(
            tuple(axis_names[name] for name in analysis.parallel_axes)
            if canonical
            else analysis.parallel_axes
        ),
        reduction_axis=(
            axis_names[analysis.reduction_axis]
            if canonical
            else analysis.reduction_axis
        ),
        lhs=clone_access(analysis.lhs, values[0]),
        rhs=clone_access(analysis.rhs, values[1]),
        accumulator=clone_access(analysis.accumulator, values[2]),
        output=clone_access(analysis.output, values[2]),
    )


def derive_apu_v1_profile_row_tiles(
    analysis,
    target,
    *,
    column_extent: int | None = None,
) -> tuple[int, ...]:
    """Derive the bounded parity domain for outer single-APUC row shards."""

    shape = _validate_analysis(analysis)
    provenance = _target_provenance(target)
    columns = (
        shape.columns
        if column_extent is None
        else _positive_int(column_extent, "column_extent")
    )
    if columns > shape.columns or shape.columns % columns:
        raise UnsupportedAPUV1ProfileSearchError(
            "column_extent must exactly divide logical N"
        )
    if shape.columns == 1 or shape.reduction == 1:
        return (shape.rows,)
    padded_columns = 1 << (columns - 1).bit_length()
    rows_per_vr = provenance.vr_lanes // padded_columns
    max_rows = rows_per_vr * provenance.accumulator_block_limit
    if rows_per_vr <= 0:
        raise UnsupportedAPUV1ProfileSearchError(
            "one padded output row exceeds the physical VR lane capacity"
        )
    rows = tuple(
        sorted(
            {
                min(shape.rows, rows_per_vr * block)
                for block in _ACCUMULATOR_BLOCKS
                if block <= provenance.accumulator_block_limit
                and rows_per_vr * block <= max_rows
            }
        )
    )
    if not rows:
        raise UnsupportedAPUV1ProfileSearchError(
            "logical shape has no bounded dense parity row candidate"
        )
    return rows


def _plan_family(decision: APUV1PlanDecision) -> tuple[object, ...]:
    return (
        decision.temporal_strategy,
        decision.reduction_kind,
        decision.reduction_group_size,
        decision.accumulator_block,
        decision.reduction_tile_extent,
        decision.transfer_routes,
    )


@dataclass(frozen=True)
class _PlanEntry:
    decision: APUV1PlanDecision
    family: tuple[object, ...]
    analysis: ContractionAnalysis = field(repr=False, compare=False)
    plan: object = field(repr=False, compare=False)
    canonical_analysis: ContractionAnalysis = field(repr=False, compare=False)
    canonical_plan: object = field(repr=False, compare=False)


class _ProfileProblem:
    def __init__(self, analysis, target, cost, column_extent):
        self.analysis = analysis
        self.logical_shape = _validate_analysis(analysis)
        self.target = target
        self.target_provenance = _target_provenance(target)
        self.cost = _bound_cost(target, cost)
        self.cost_provenance = APUV1ProfileCostProvenance(self.cost.fingerprint)
        self.columns = (
            self.logical_shape.columns
            if column_extent is None
            else _positive_int(column_extent, "column_extent")
        )
        if (
            self.columns > self.logical_shape.columns
            or self.logical_shape.columns % self.columns
        ):
            raise UnsupportedAPUV1ProfileSearchError(
                "column_extent must exactly divide logical N"
            )
        self.column_repetitions = self.logical_shape.columns // self.columns
        self.row_tiles = derive_apu_v1_profile_row_tiles(
            analysis, target, column_extent=self.columns
        )
        self.semantic_fingerprint = _semantic_fingerprint(analysis, self.logical_shape)
        model = _digest(
            {
                "kind": "apu-v1-whole-profile-cycle-composition-v1",
                "target": self.target_provenance.fingerprint,
                "cost": self.cost.fingerprint,
                "formula": "shard_cycles*row_waves*column_repetitions",
            }
        )
        self.objective_domain = ScheduleObjectiveDomain.fingerprinted_target(
            metric="cycles",
            target=target.name,
            model_fingerprint=model,
            fidelity="analytical",
            scope="whole_profile_composition",
            unit="cycles",
            direction="minimize",
        )
        self._entries: dict[int, tuple[_PlanEntry, ...]] = {}

    def entries(self, rows: int) -> tuple[_PlanEntry, ...]:
        if rows in self._entries:
            return self._entries[rows]
        local = _clone_analysis(self.analysis, rows, self.columns, canonical=False)
        canonical = _clone_analysis(self.analysis, rows, self.columns, canonical=True)
        try:
            original_candidates = generate_apu_v1_vectorization_candidates(local)
            canonical_candidates = generate_apu_v1_vectorization_candidates(canonical)
        except (IllegalContractionError, TypeError, ValueError) as error:
            raise UnsupportedAPUV1ProfileSearchError(
                "retained profile shape cannot generate physical plans"
            ) from error
        if len(original_candidates) != len(canonical_candidates):
            raise RuntimeError("canonical profile planning changed candidate structure")
        entries = []
        seen = set()
        for original, normalized in zip(original_candidates, canonical_candidates):
            decision = _vector_program._apu_v1_plan_decision(normalized.plan)
            if decision in seen:
                continue
            seen.add(decision)
            entries.append(
                _PlanEntry(
                    decision,
                    _plan_family(decision),
                    local,
                    original.plan,
                    canonical,
                    normalized.plan,
                )
            )
        self._entries[rows] = tuple(entries)
        return self._entries[rows]

    def entry(self, decision: APUV1ProfileRowDecision) -> _PlanEntry:
        return next(
            item
            for item in self.entries(decision.row_tile)
            if item.decision == decision.plan
        )


def derive_apu_v1_profile_plan_decisions(
    analysis,
    target,
    *,
    row_tile: int,
    column_extent: int | None = None,
) -> tuple[APUV1PlanDecision, ...]:
    """Expose the dependent physical domain for freezing an incumbent."""

    rows = derive_apu_v1_profile_row_tiles(
        analysis, target, column_extent=column_extent
    )
    if row_tile not in rows:
        raise ValueError("row_tile is outside the derived profile domain")
    problem = _ProfileProblem.__new__(_ProfileProblem)
    problem.analysis = analysis
    problem.logical_shape = _validate_analysis(analysis)
    problem.target = target
    problem.columns = (
        problem.logical_shape.columns if column_extent is None else int(column_extent)
    )
    problem._entries = {}
    return tuple(item.decision for item in problem.entries(row_tile))


def _realize(analysis, plan):
    return _vector_program._realize(analysis, plan)


def _emitted_source_fingerprint(realization):
    source = realization.device_source()
    if not isinstance(source, str) or not source:
        raise InfeasibleSchedule("profile realization emitted no exact device source")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _realize_entry(entry: _PlanEntry):
    realization, error = _realize(entry.analysis, entry.plan)
    if realization is None:
        raise InfeasibleSchedule(error or "profile plan is not physically realizable")
    canonical, error = _realize(entry.canonical_analysis, entry.canonical_plan)
    if canonical is None:
        raise InfeasibleSchedule(
            error or "canonical profile source is not physically realizable"
        )
    if realization.plan is not entry.plan or canonical.plan is not entry.canonical_plan:
        raise InfeasibleSchedule("profile realizer returned a different plan object")
    emitted_source_fingerprint = _emitted_source_fingerprint(realization)
    structural_source_fingerprint = _emitted_source_fingerprint(canonical)
    return (
        realization,
        emitted_source_fingerprint,
        structural_source_fingerprint,
        _digest(
            {
                "plan": entry.decision.physical_fingerprint,
                "emitted_source": emitted_source_fingerprint,
            }
        ),
    )


def _materialize(problem: _ProfileProblem, decision: APUV1ProfileRowDecision):
    entry = problem.entry(decision)
    (
        realization,
        emitted_source_fingerprint,
        structural_source_fingerprint,
        plan_emitted_source_fingerprint,
    ) = _realize_entry(entry)
    remainder = problem.logical_shape.rows % decision.row_tile
    final_shape = None
    final_realization = None
    final_plan = final_emitted = final_structural = final_plan_emitted = None
    if remainder:
        partial = next(
            (
                item
                for item in problem.entries(remainder)
                if item.family == entry.family
            ),
            None,
        )
        if partial is None:
            raise InfeasibleSchedule(
                "final partial row wave has no exact physical plan mapping"
            )
        (
            final_realization,
            final_emitted,
            final_structural,
            final_plan_emitted,
        ) = _realize_entry(partial)
        final_plan = partial.decision.physical_fingerprint
        final_shape = APUV1ProfileShape(
            remainder, problem.columns, problem.logical_shape.reduction
        )
    shard_shape = APUV1ProfileShape(
        decision.row_tile, problem.columns, problem.logical_shape.reduction
    )
    return APUV1ProfileRowMaterialization(
        decision=decision,
        logical_shape=problem.logical_shape,
        shard_shape=shard_shape,
        final_partial_shape=final_shape,
        row_waves=math.ceil(problem.logical_shape.rows / decision.row_tile),
        column_repetitions=problem.column_repetitions,
        semantic_fingerprint=problem.semantic_fingerprint,
        plan_fingerprint=decision.plan.physical_fingerprint,
        emitted_source_fingerprint=emitted_source_fingerprint,
        structural_source_fingerprint=structural_source_fingerprint,
        plan_emitted_source_fingerprint=plan_emitted_source_fingerprint,
        final_plan_fingerprint=final_plan,
        final_emitted_source_fingerprint=final_emitted,
        final_structural_source_fingerprint=final_structural,
        final_plan_emitted_source_fingerprint=final_plan_emitted,
        target_provenance=problem.target_provenance,
        cost_provenance=problem.cost_provenance,
        objective_domain=problem.objective_domain,
        realization=realization,
        final_realization=final_realization,
    )


def _score(problem: _ProfileProblem, materialized):
    if not isinstance(materialized, APUV1ProfileRowMaterialization):
        raise TypeError("profile scoring requires a frozen materialization")
    estimate = estimate_apu_v1_realization(
        materialized.realization, problem.target, problem.cost
    )
    metadata = estimate.graph.metadata
    if (
        metadata.get("target") != problem.target.name
        or metadata.get("cost_fingerprint") != problem.cost.fingerprint
    ):
        raise InfeasibleSchedule("profile estimate provenance is incomparable")
    shard_cycles = int(estimate.cycles)
    final_shard_cycles = None
    if materialized.final_realization is not None:
        final_estimate = estimate_apu_v1_realization(
            materialized.final_realization, problem.target, problem.cost
        )
        final_metadata = final_estimate.graph.metadata
        if (
            final_metadata.get("target") != problem.target.name
            or final_metadata.get("cost_fingerprint") != problem.cost.fingerprint
        ):
            raise InfeasibleSchedule("partial profile estimate is incomparable")
        final_shard_cycles = int(final_estimate.cycles)
    full_waves = materialized.row_waves - (final_shard_cycles is not None)
    composed_cycles = (
        shard_cycles * full_waves + (final_shard_cycles or 0)
    ) * materialized.column_repetitions
    return APUV1ProfileCompositionScore(
        shard_cycles,
        materialized.row_waves,
        materialized.column_repetitions,
        composed_cycles,
        materialized.objective_domain,
        final_shard_cycles,
    )


def search_apu_v1_profile_rows(
    analysis,
    target,
    cost,
    *,
    column_extent: int | None = None,
    incumbent_row_tile: int | None = None,
    incumbent_plan_decision: APUV1PlanDecision | None = None,
    opaque_incumbent: OpaqueScheduleIncumbent | None = None,
) -> APUV1ProfileRowSearchResult:
    """Shadow-search outer row shards and dependent physical APUv1 plans.

    Exactly one incumbent form is mandatory.  A reverse-mappable incumbent is
    evaluated through the same row/plan domain; otherwise callers may retain an
    explicit :class:`OpaqueScheduleIncumbent`.  The returned result's
    ``activation`` property always uses the no-evidence guarded path and cannot
    promote a recommendation.
    """

    exact_fields = (incumbent_row_tile, incumbent_plan_decision)
    exact = all(value is not None for value in exact_fields)
    incomplete = any(value is not None for value in exact_fields) and not exact
    if incomplete:
        raise ValueError("exact incumbent requires row_tile and physical plan")
    if exact == (opaque_incumbent is not None):
        if not exact:
            raise MissingScheduleIncumbent(
                "profile-row search requires an exact or opaque incumbent"
            )
        raise ValueError("provide either an exact or opaque incumbent, not both")
    if opaque_incumbent is not None and not isinstance(
        opaque_incumbent, OpaqueScheduleIncumbent
    ):
        raise TypeError("opaque_incumbent must be an OpaqueScheduleIncumbent")
    if exact and not isinstance(incumbent_plan_decision, APUV1PlanDecision):
        raise TypeError("incumbent physical plan must be an APUV1PlanDecision")

    problem = _ProfileProblem(analysis, target, cost, column_extent)

    def plan_domain(decisions):
        return tuple(item.decision for item in problem.entries(decisions["row_tile"]))

    result = grid_search(
        (
            DecisionDomain("row_tile", problem.row_tiles),
            DecisionDomain("plan", plan_domain),
        ),
        build=lambda decisions: APUV1ProfileRowDecision(
            decisions["row_tile"], decisions["plan"]
        ),
        materialize=lambda decision: _materialize(problem, decision),
        score=lambda materialized: _score(problem, materialized),
        objective=lambda score: score.composed_cycles,
        objective_domain=lambda score: score.objective_domain,
        incumbent=(
            {
                "row_tile": incumbent_row_tile,
                "plan": incumbent_plan_decision,
            }
            if exact
            else None
        ),
    )
    retained = result.best_incumbent if exact else opaque_incumbent
    assert retained is not None
    return APUV1ProfileRowSearchResult(
        result.ranked, result.rejections, result.stats, retained
    )


__all__ = [
    "APUV1ProfileCompositionScore",
    "APUV1ProfileCostProvenance",
    "APUV1ProfileRowDecision",
    "APUV1ProfileRowMaterialization",
    "APUV1ProfileRowSearchResult",
    "APUV1ProfileShape",
    "APUV1ProfileTargetProvenance",
    "UnsupportedAPUV1ProfileSearchError",
    "derive_apu_v1_profile_plan_decisions",
    "derive_apu_v1_profile_row_tiles",
    "search_apu_v1_profile_rows",
]
