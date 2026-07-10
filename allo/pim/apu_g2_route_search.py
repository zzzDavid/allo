"""Shadow search across the two proven APUg2 single-GEMV routes.

This module is intentionally not imported by the public compiler.  It records
cross-route evidence without changing dispatch, and it fails closed unless a
retained-MLIR contraction belongs to the exact direct-reduction/dot-tile
intersection described below.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..perf import BoundCostSpec
from .apu_g2_contraction import APUG2RankNContractionPlan
from .apu_g2_pipeline import NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
from .apu_g2_program import APUG2Program, build_apu_g2_execution_graph
from .apu_g2_recipe import (
    APUG2_DOT_TILE_MEASURED_POINTS,
    build_apu_g2_recipe_graph,
    build_apu_g2_u16_dot_tile_recipe,
    calibrate_apu_g2_recipe_from_measured_tile,
)
from .apu_g2_vectorize import (
    UnsupportedAPUG2ContractionError,
    plan_apu_g2_gemv,
)
from .costs.apu_g2 import APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    grid_search,
)


DIRECT_REDUCTION = "direct_reduction"
DOT_TILE = "dot_tile"
APUG2_SINGLE_CONTRACTION_ROUTES = (DIRECT_REDUCTION, DOT_TILE)
_ROUTE_EXECUTION_LOCK = threading.RLock()


class UnsupportedAPUG2RouteSearchError(ValueError):
    """The plan is outside the honest cross-route comparison domain."""


@dataclass(frozen=True)
class APUG2SingleContractionRouteDecision:
    """The only search decision: which proven physical executor to use."""

    route: str

    def __post_init__(self) -> None:
        if self.route not in APUG2_SINGLE_CONTRACTION_ROUTES:
            raise ValueError(f"unsupported APUg2 physical route {self.route!r}")


@dataclass(frozen=True)
class APUG2SingleContractionFacts:
    """Rename-free retained-MLIR facts shared by both materializations."""

    output_extent: int
    reduction_extent: int
    operand_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    operand_axis_positions: tuple[tuple[int, ...], tuple[int, ...]]
    accumulator_shape: tuple[int, ...]
    accumulator_axis_positions: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_axis_positions: tuple[int, ...]
    matrix_operand: int
    vector_operand: int
    matrix_transposed: bool
    alpha: int
    beta: int
    semantic_fingerprint: str


@dataclass(frozen=True)
class APUG2RouteProvenance:
    """Target and executable-cost identity retained with a materialization."""

    target_name: str
    target_revision: str
    cost_name: str
    cost_fingerprint: str


@dataclass(frozen=True)
class APUG2RouteMaterialization:
    """One immutable, score-ready APUg2 physical route realization."""

    route: str
    executor: Callable[..., object] = field(repr=False, compare=False)
    runtime_artifact: object = field(repr=False, compare=False)
    facts: APUG2SingleContractionFacts
    provenance: APUG2RouteProvenance
    source_hashes: tuple[tuple[str, str], ...]
    source_fingerprint: str
    physical_model_fingerprint: str
    calibration_basis: str
    calibration_repetitions: int
    measured_device_pipeline_ticks: float
    analytical_device_pipeline_ticks: int
    objective_domain: ScheduleObjectiveDomain
    execution_graph: object = field(repr=False, compare=False)

    @property
    def semantic_fingerprint(self) -> str:
        return self.facts.semantic_fingerprint

    @property
    def target_name(self) -> str:
        return self.provenance.target_name

    @property
    def target_revision(self) -> str:
        return self.provenance.target_revision

    @property
    def cost_name(self) -> str:
        return self.provenance.cost_name

    @property
    def cost_fingerprint(self) -> str:
        return self.provenance.cost_fingerprint

    @property
    def output_extent(self) -> int:
        return self.facts.output_extent

    @property
    def reduction_extent(self) -> int:
        return self.facts.reduction_extent

    @property
    def matrix_transposed(self) -> bool:
        return self.facts.matrix_transposed

    @property
    def epilogue(self) -> tuple[int, int]:
        return self.facts.alpha, self.facts.beta

    @property
    def executor_identity(self) -> tuple[str, str, str]:
        return self.runtime_artifact.executor_identity

    @property
    def executor_fingerprint(self) -> str:
        return self.executor_identity[2]

    @property
    def current_source_fingerprint(self) -> str:
        return self.runtime_artifact.current_source_fingerprint(self.executor)

    @property
    def promotion_platform_fingerprint(self):
        if self.runtime_artifact.platform_fingerprint is None:
            return None
        return self.runtime_artifact.current_platform_fingerprint()

    @property
    def promotion_materialization_fingerprint(self) -> str | None:
        current = self.runtime_artifact.promotion_source_fingerprint(self.executor)
        if current is None or current != self.source_fingerprint:
            return None
        payload = {
            "kind": "apu-g2-single-contraction-route-materialization-v1",
            "route": self.route,
            "executor": self.executor_identity,
            "semantic_fingerprint": self.semantic_fingerprint,
            "target": {
                "name": self.target_name,
                "revision": self.target_revision,
            },
            "cost": {
                "name": self.cost_name,
                "fingerprint": self.cost_fingerprint,
            },
            "source_hashes": self.source_hashes,
            "source_fingerprint": self.source_fingerprint,
            "physical_model_fingerprint": self.physical_model_fingerprint,
            "calibration": {
                "basis": self.calibration_basis,
                "repetitions": self.calibration_repetitions,
                "measured_device_pipeline_ticks": (self.measured_device_pipeline_ticks),
                "analytical_device_pipeline_ticks": (
                    self.analytical_device_pipeline_ticks
                ),
            },
        }
        return _digest(payload)

    @property
    def fingerprint(self) -> str:
        return self.promotion_materialization_fingerprint

    def execute(self, lhs, rhs, accumulator, *, repetitions=8):
        """Invoke the exact route after applying retained operand roles."""

        self.runtime_artifact.assert_current(self.executor)
        operands = (np.asarray(lhs), np.asarray(rhs))
        matrix = operands[self.facts.matrix_operand]
        vector = operands[self.facts.vector_operand]
        if self.matrix_transposed:
            matrix = matrix.T
        matrix = np.ascontiguousarray(matrix)
        vector = np.ascontiguousarray(vector)
        accumulator = np.ascontiguousarray(accumulator)

        def invoke():
            if self.route == DIRECT_REDUCTION:
                return self.executor(
                    matrix,
                    vector,
                    accumulator,
                    repetitions=repetitions,
                )
            broadcast = np.ascontiguousarray(np.broadcast_to(vector, matrix.shape))
            return self.executor(
                matrix,
                broadcast,
                accumulator=accumulator,
                alpha=self.facts.alpha,
                beta=self.facts.beta,
                repetitions=repetitions,
            )

        from . import apu_g2_runtime

        executor_globals = self.executor.__globals__
        if "_TEMPLATE" not in executor_globals:
            return invoke()
        with (
            _ROUTE_EXECUTION_LOCK
        ), self.runtime_artifact.temporary_project() as project:
            original_executor_template = executor_globals["_TEMPLATE"]
            original_runtime_template = apu_g2_runtime._TEMPLATE
            executor_globals["_TEMPLATE"] = project
            apu_g2_runtime._TEMPLATE = project
            try:
                return invoke()
            finally:
                executor_globals["_TEMPLATE"] = original_executor_template
                apu_g2_runtime._TEMPLATE = original_runtime_template


@dataclass(frozen=True)
class APUG2RouteScore:
    """A calibrated repeated-throughput device-pipeline observation."""

    device_pipeline_ticks: float
    analytical_device_pipeline_ticks: int
    estimate: object = field(repr=False, compare=False)
    objective_domain: ScheduleObjectiveDomain

    @property
    def cycles(self) -> int:
        return self.analytical_device_pipeline_ticks


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _access_semantics(access) -> dict[str, object]:
    return {
        "dtype": access.dtype,
        "shape": tuple(access.shape),
        "mode": access.mode,
        "axis_positions": tuple(access.axis_positions),
    }


def _semantic_facts(plan, direct_plan) -> APUG2SingleContractionFacts:
    operand_accesses = (plan.lhs, plan.rhs)
    matrix_operand = next(
        index for index, access in enumerate(operand_accesses) if len(access.shape) == 2
    )
    vector_operand = 1 - matrix_operand
    manifest = {
        "kind": "apu-g2-single-ui16-mul-add-semantics-v1",
        "topology": "single",
        "axes": (
            {"role": "output", "extent": direct_plan.output_extent},
            {"role": "reduction", "extent": direct_plan.reduction_extent},
        ),
        "numeric_type": plan.analysis.numeric_type,
        "multiply_operation": plan.analysis.multiply_operation,
        "combine_operation": plan.analysis.combine_operation,
        "packed_word_bits": plan.analysis.packed_word_bits,
        "operands": tuple(_access_semantics(access) for access in operand_accesses),
        "accumulator": _access_semantics(plan.initial_output),
        "output": _access_semantics(plan.output),
        "matrix_operand": matrix_operand,
        "vector_operand": vector_operand,
        "matrix_transposed": direct_plan.matrix_transposed,
        "epilogue": {"alpha": plan.epilogue[0], "beta": plan.epilogue[1]},
        "storage_relations": {
            "operands_distinct": plan.lhs.value != plan.rhs.value,
            "accumulator_is_output": (plan.initial_output.value == plan.output.value),
            "operands_do_not_alias_output": (
                plan.lhs.value != plan.output.value
                and plan.rhs.value != plan.output.value
            ),
        },
    }
    return APUG2SingleContractionFacts(
        output_extent=direct_plan.output_extent,
        reduction_extent=direct_plan.reduction_extent,
        operand_shapes=tuple(tuple(access.shape) for access in operand_accesses),
        operand_axis_positions=tuple(
            tuple(access.axis_positions) for access in operand_accesses
        ),
        accumulator_shape=tuple(plan.initial_output.shape),
        accumulator_axis_positions=tuple(plan.initial_output.axis_positions),
        output_shape=tuple(plan.output.shape),
        output_axis_positions=tuple(plan.output.axis_positions),
        matrix_operand=matrix_operand,
        vector_operand=vector_operand,
        matrix_transposed=direct_plan.matrix_transposed,
        alpha=plan.epilogue[0],
        beta=plan.epilogue[1],
        semantic_fingerprint=_digest(manifest),
    )


def _require_search_domain(plan, target, cost):
    if not isinstance(plan, APUG2RankNContractionPlan):
        raise UnsupportedAPUG2RouteSearchError(
            "APUg2 route search requires a retained-MLIR rank-N contraction plan"
        )
    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("APUg2 route search requires cost bound to the target")
    if getattr(target, "name", None) != "apu_v2":
        raise UnsupportedAPUG2RouteSearchError(
            "APUg2 route search requires the apu_v2 target"
        )
    if (
        plan.module.contraction_topology != "single"
        or len(plan.module.regions) != 1
        or plan.module.dependencies
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "chain and special multi-region routes are outside this shadow search"
        )
    if (
        plan.analysis.numeric_type != "ui16"
        or plan.analysis.multiply_operation != "arith.muli"
        or plan.analysis.combine_operation != "arith.addi"
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "route search requires one ui16 multiply/add contraction"
        )
    if (
        len(plan.output_axes) != 1
        or len(plan.dot_axes) != 1
        or plan.batch_axes
        or plan.batch_local_accumulator
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "persistent, streaming, batched, and special routes are outside the "
            "one-output-axis comparison domain"
        )
    if plan.epilogue != (1, 1):
        raise UnsupportedAPUG2RouteSearchError(
            "route search requires the exact unit epilogue alpha=1, beta=1"
        )
    if (
        plan.initial_output.value != plan.output.value
        or plan.initial_output.axis_positions != (0,)
        or plan.output.axis_positions != (0,)
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "route search requires one in-place output-axis accumulator"
        )
    if (
        plan.lhs.value == plan.rhs.value
        or plan.lhs.value == plan.output.value
        or plan.rhs.value == plan.output.value
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "route search requires distinct read operands and non-aliasing output"
        )
    try:
        direct_plan = plan_apu_g2_gemv(plan.analysis)
    except (UnsupportedAPUG2ContractionError, TypeError, ValueError) as error:
        raise UnsupportedAPUG2RouteSearchError(
            "contraction is not legal for the direct-reduction executor"
        ) from error
    if (
        plan.tiling.task_count != 1
        or plan.tiling.tile_count != 1
        or plan.tiling.reduction_tile_count != 1
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "streaming or multi-tile contractions are outside this route search"
        )
    try:
        tile = plan.tiling.tile_plan(0)
    except (TypeError, ValueError) as error:
        raise UnsupportedAPUG2RouteSearchError(
            "contraction is not legal for the dot-tile executor"
        ) from error
    if (
        tile.output_extent != direct_plan.output_extent
        or tile.reduction_extent != direct_plan.reduction_extent
    ):
        raise UnsupportedAPUG2RouteSearchError(
            "direct and dot-tile routes do not cover the same logical shape"
        )
    return direct_plan


def _dot_model_manifest() -> tuple[object, ...]:
    return tuple(
        (
            tuple(key),
            point.analytical_cycles,
            point.measured_ticks_per_pipeline,
            point.repetitions,
            point.final_pipeline_ticks,
        )
        for key, point in sorted(APUG2_DOT_TILE_MEASURED_POINTS.items())
    )


def _objective_domain(target, cost) -> ScheduleObjectiveDomain:
    model_fingerprint = _digest(
        {
            "kind": "apu-g2-cross-route-repeated-device-pipeline-model-v1",
            "cost_fingerprint": cost.fingerprint,
            "direct_calibration": (
                NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.canonical_manifest
            ),
            "direct_measured_ticks_per_pipeline": (
                APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE
            ),
            "direct_repetitions": 8,
            "dot_tile_model": _dot_model_manifest(),
            "dot_tile_interpolation": "output-sensitive-v1",
        }
    )
    return ScheduleObjectiveDomain.fingerprinted_target(
        metric="device_pipeline_ticks",
        target=target.name,
        model_fingerprint=model_fingerprint,
        fidelity="real_card_calibrated_model",
        scope="device_compute_repeated_throughput",
        unit="device_pipeline_ticks",
        direction="minimize",
    )


def _route_build_manifest(route: str):
    binary = {
        DIRECT_REDUCTION: "tenon_apu_g2_u16_gemv",
        DOT_TILE: "tenon_apu_g2_u16_dot_tile",
    }[route]
    return {
        "schema": "apu-g2-cmake-build-v1",
        "device_configure": [
            "cmake",
            "-S",
            "{project}/device",
            "-B",
            "{project}/build/device",
        ],
        "device_compile": [
            "cmake",
            "--build",
            "{project}/build/device",
            "-j",
            "{jobs}",
        ],
        "host_configure": [
            "cmake",
            "-S",
            "{project}",
            "-B",
            "{project}/build/host",
            "-DGSI_TARGET=DEVICE",
            "-DBOARD_DEVICE_LIB={device_library}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "host_compile": [
            "cmake",
            "--build",
            "{project}/build/host",
            "-j",
            "{jobs}",
        ],
        "jobs_policy": "TENON_APU_G2_BUILD_JOBS_or_min_cpu_count_8",
        "device_library": "build/device/bin/tenon_apu_g2_tasks.update.bin",
        "host_binary": f"build/host/{binary}",
    }


def _route_abi_manifest(route: str):
    if route == DIRECT_REDUCTION:
        return {
            "schema": "apu-g2-u16-gemv-abi-v1",
            "inputs": [
                ["matrix", "uint16", ["M", "K"], "c_contiguous"],
                ["vector", "uint16", ["K"], "c_contiguous"],
                ["accumulator", "uint16", ["M"], "c_contiguous"],
            ],
            "output": ["out", "uint16", ["M"], "c_contiguous"],
            "argv": [
                "matrix_path",
                "vector_path",
                "accumulator_path",
                "temporary_path",
                "output_path",
                "log_block_size",
                "repetitions",
            ],
            "arithmetic": "modulo_2^16",
        }
    return {
        "schema": "apu-g2-u16-dot-tile-abi-v1",
        "inputs": [
            ["left", "uint16", ["M", "K"], "c_contiguous"],
            ["right", "uint16", ["M", "K"], "c_contiguous"],
            ["accumulator", "uint16", ["M"], "c_contiguous"],
        ],
        "output": ["out", "uint16", ["M"], "c_contiguous"],
        "argv": [
            "left_path",
            "right_path",
            "accumulator_path",
            "output_path",
            "log_block_size",
            "alpha_u16",
            "beta_u16",
            "epilogue_enabled",
            "repetitions",
        ],
        "arithmetic": "modulo_2^16",
    }


def _route_runtime_artifact(route: str, executor):
    from .apu_g2_runtime import freeze_apu_g2_runtime_artifact

    try:
        return freeze_apu_g2_runtime_artifact(
            executor,
            build_manifest=_route_build_manifest(route),
            abi_manifest=_route_abi_manifest(route),
            dependency_modules=(
                "allo.pim.apu_g2_layout",
                "allo.pim.apu_g2_gesummv_runtime",
            ),
        )
    except RuntimeError as error:
        raise InfeasibleSchedule(str(error)) from error


def _direct_materialization(plan, target, cost, facts, objective_domain):
    from .apu_g2_gemv_runtime import run_apu_g2_u16_gemv

    program = APUG2Program(
        operation="gemv_u16",
        shape=(facts.output_extent, facts.reduction_extent),
        repetitions=8,
        name="apu_g2_route_search_direct_reduction",
    )
    graph = build_apu_g2_execution_graph(program, target, cost)
    if graph.metadata.get("execution") != "direct_vl64":
        raise InfeasibleSchedule("direct route did not materialize direct_vl64")
    runtime_artifact = _route_runtime_artifact(DIRECT_REDUCTION, run_apu_g2_u16_gemv)
    physical_model_fingerprint = _digest(
        {
            "kind": "apu-g2-direct-reduction-physical-model-v1",
            "pipeline_calibration_fingerprint": graph.metadata.get(
                "pipeline_calibration_fingerprint"
            ),
            "pipeline_signature": graph.metadata.get("pipeline_signature"),
            "layout": graph.metadata.get("layout"),
            "shape": (facts.output_extent, facts.reduction_extent),
        }
    )
    return APUG2RouteMaterialization(
        route=DIRECT_REDUCTION,
        executor=run_apu_g2_u16_gemv,
        runtime_artifact=runtime_artifact,
        facts=facts,
        provenance=APUG2RouteProvenance(
            target.name,
            objective_domain.target_revision,
            cost.spec.name,
            cost.fingerprint,
        ),
        source_hashes=runtime_artifact.source_hashes,
        source_fingerprint=runtime_artifact.source_fingerprint,
        physical_model_fingerprint=physical_model_fingerprint,
        calibration_basis=str(graph.metadata["calibration_basis"]),
        calibration_repetitions=int(graph.metadata["calibration_repetitions"]),
        measured_device_pipeline_ticks=float(
            graph.metadata["measured_ticks_per_pipeline"]
        ),
        analytical_device_pipeline_ticks=int(graph.metadata["calibration_ticks"]),
        objective_domain=objective_domain,
        execution_graph=graph,
    )


def _dot_tile_materialization(plan, target, cost, facts, objective_domain):
    from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile

    recipe = build_apu_g2_u16_dot_tile_recipe(
        facts.output_extent,
        facts.reduction_extent,
        alpha=facts.alpha,
        beta=facts.beta,
        name="apu_g2_route_search_dot_tile",
    )
    calibration = calibrate_apu_g2_recipe_from_measured_tile(recipe)
    graph = build_apu_g2_recipe_graph(
        recipe,
        target,
        calibration=calibration,
    )
    if (
        recipe.metadata.get("output_tile_count") != 1
        or calibration.measured_ticks_per_pipeline is None
        or calibration.repetitions is None
    ):
        raise InfeasibleSchedule(
            "dot-tile route lacks one-task repeated-throughput calibration"
        )
    runtime_artifact = _route_runtime_artifact(DOT_TILE, run_apu_g2_u16_dot_tile)
    physical_model_fingerprint = _digest(
        {
            "kind": "apu-g2-dot-tile-physical-model-v1",
            "recipe_fingerprint": recipe.structural_fingerprint,
            "calibration_recipe_fingerprint": calibration.recipe_fingerprint,
            "calibration_basis": calibration.basis,
            "calibration_total_cycles": calibration.total_cycles,
            "shape": (facts.output_extent, facts.reduction_extent),
        }
    )
    return APUG2RouteMaterialization(
        route=DOT_TILE,
        executor=run_apu_g2_u16_dot_tile,
        runtime_artifact=runtime_artifact,
        facts=facts,
        provenance=APUG2RouteProvenance(
            target.name,
            objective_domain.target_revision,
            cost.spec.name,
            cost.fingerprint,
        ),
        source_hashes=runtime_artifact.source_hashes,
        source_fingerprint=runtime_artifact.source_fingerprint,
        physical_model_fingerprint=physical_model_fingerprint,
        calibration_basis=calibration.basis,
        calibration_repetitions=calibration.repetitions,
        measured_device_pipeline_ticks=calibration.measured_ticks_per_pipeline,
        analytical_device_pipeline_ticks=calibration.total_cycles,
        objective_domain=objective_domain,
        execution_graph=graph,
    )


def _materialize_apu_g2_single_contraction_route(
    plan,
    target,
    cost,
    decision,
    objective_domain,
):
    """Materialize one route without scoring it."""

    if not isinstance(decision, APUG2SingleContractionRouteDecision):
        raise TypeError("route materialization requires a frozen route decision")
    direct_plan = _require_search_domain(plan, target, cost)
    facts = _semantic_facts(plan, direct_plan)
    if decision.route == DIRECT_REDUCTION:
        return _direct_materialization(plan, target, cost, facts, objective_domain)
    return _dot_tile_materialization(plan, target, cost, facts, objective_domain)


def _score_apu_g2_single_contraction_route(materialized, cost):
    """Score only calibrated, non-transport, repeated device computation."""

    if not isinstance(materialized, APUG2RouteMaterialization):
        raise TypeError("route scoring requires an APUg2 route materialization")
    try:
        materialized.runtime_artifact.assert_current(materialized.executor)
    except RuntimeError as error:
        raise InfeasibleSchedule(str(error)) from error
    if (
        not isinstance(cost, BoundCostSpec)
        or cost.fingerprint != materialized.cost_fingerprint
    ):
        raise InfeasibleSchedule(
            "materialized route cost provenance changed before score"
        )
    graph = materialized.execution_graph
    metadata = graph.metadata
    forbidden = {
        "wall_us",
        "host_wall_estimate",
        "transport_schedule",
    }
    if forbidden & set(metadata):
        raise InfeasibleSchedule(
            "wall-time and transport scores are out of the device-compute domain"
        )
    if materialized.calibration_repetitions != 8:
        raise InfeasibleSchedule(
            "cross-route score requires the shared eight-repeat calibration"
        )
    measured = materialized.measured_device_pipeline_ticks
    if not math.isfinite(measured) or measured <= 0:
        raise InfeasibleSchedule("device pipeline score must be positive and finite")
    estimate = cost.evaluate(graph)
    analytical = materialized.analytical_device_pipeline_ticks
    if estimate.cycles != analytical:
        raise InfeasibleSchedule(
            "calibrated graph does not reproduce its device-pipeline total"
        )
    return APUG2RouteScore(
        device_pipeline_ticks=measured,
        analytical_device_pipeline_ticks=analytical,
        estimate=estimate,
        objective_domain=materialized.objective_domain,
    )


def search_apu_g2_single_contraction_routes(plan, target, cost):
    """Shadow-search direct reduction versus one physical dot tile.

    Shapes, affine maps, transposition, and epilogue coefficients come only
    from ``plan``.  The route string is the sole decision.  Direct reduction
    is evaluated first as the explicit incumbent; callers must use guarded
    activation if they ever consume this evidence, because this standalone
    search intentionally has no connection to public compiler dispatch.
    """

    direct_plan = _require_search_domain(plan, target, cost)
    _semantic_facts(plan, direct_plan)
    domain = _objective_domain(target, cost)
    return grid_search(
        (DecisionDomain("route", APUG2_SINGLE_CONTRACTION_ROUTES),),
        build=lambda decisions: APUG2SingleContractionRouteDecision(decisions["route"]),
        materialize=lambda decision: (
            _materialize_apu_g2_single_contraction_route(
                plan,
                target,
                cost,
                decision,
                domain,
            )
        ),
        score=lambda materialized: _score_apu_g2_single_contraction_route(
            materialized, cost
        ),
        objective=lambda score: score.device_pipeline_ticks,
        objective_domain=lambda score: score.objective_domain,
        incumbent={"route": DIRECT_REDUCTION},
    )


__all__ = [
    "APUG2RouteMaterialization",
    "APUG2RouteProvenance",
    "APUG2RouteScore",
    "APUG2SingleContractionFacts",
    "APUG2SingleContractionRouteDecision",
    "APUG2_SINGLE_CONTRACTION_ROUTES",
    "DIRECT_REDUCTION",
    "DOT_TILE",
    "UnsupportedAPUG2RouteSearchError",
    "search_apu_g2_single_contraction_routes",
]
