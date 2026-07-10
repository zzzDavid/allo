# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public callable for MLIR-discovered APUg2 uint16 GEMV contractions."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field

import numpy as np

from ..perf import BoundCostSpec
from ..spmw_codegen import RunResult
from .apu_g2_contraction import (
    UnsupportedAPUG2RankNContractionError,
    plan_apu_g2_rank_n_contraction,
    plan_apu_g2_rank_n_contractions,
)
from .apu_g2_ir import discover_apu_g2_module_manifest
from .apu_g2_layout import APUG2DotTiling
from .apu_g2_program import APUG2Callable, APUG2Program
from .apu_g2_recipe import (
    APUG2Recipe,
    APUG2RecipeCalibration,
    build_apu_g2_recipe_graph,
    build_apu_g2_u16_contraction_recipe,
    build_apu_g2_u16_div_recipe,
    build_apu_g2_u16_dot_tile_recipe,
    build_apu_g2_u16_fill_recipe,
    build_apu_g2_u16_minmax_recipe,
    build_apu_g2_u16_mul_recipe,
    build_apu_g2_u16_select_lt_recipe,
    build_apu_g2_u16_sqrt_recipe,
    build_apu_g2_u16_sub_recipe,
    calibrate_apu_g2_recipe_from_measured_tile,
    chain_apu_g2_recipes,
)
from .apu_g2_structured_plans import (
    CenteredGramStatisticsPlan,
    NoStructuredAPUG2Plan,
    NormalizedGramStatisticsPlan,
    RankTwoUpdateGemvChainPlan,
    SymmetricContractionPlan,
    UnitDiagonalTriangularContractionPlan,
    analyze_apu_g2_structured_plan,
)
from .apu_g2_vectorize import (
    UnsupportedAPUG2ContractionError,
    plan_apu_g2_atax,
    plan_apu_g2_gemv,
)
from .contraction_analysis import NoContractionError, analyze_contractions
from .costs.apu_g2 import (
    APUG2_GEMM_MAX_BATCH_COLUMNS,
    APUG2_GEMM_MAX_REDUCTION_TILE,
    build_apu_g2_atax_graph,
    estimate_apu_g2_u16_gemm_wall_us,
)
from .schedule_promotion import validate_schedule_promotion_gate
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    grid_search,
    guarded_schedule_activation,
)


def _u16_epilogue_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"APUg2 epilogue {name} must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"APUg2 epilogue {name} must be an integer") from error
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"APUg2 epilogue {name} must fit uint16")
    return value


def _u16_array_argument(
    signature,
    values,
    position,
    semantic_name,
    shape,
    *,
    writable=False,
):
    parameters = tuple(signature.parameters)
    if not 0 <= position < len(parameters):
        raise TypeError(f"APUg2 structural plan references missing argument {position}")
    parameter = parameters[position]
    try:
        value = values[parameter]
    except KeyError as error:
        raise TypeError(
            f"APUg2 signature is missing structural operand {position}"
        ) from error
    if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
        raise TypeError(f"APUg2 operand {semantic_name!r} must be a NumPy uint16 array")
    if value.shape != shape:
        raise ValueError(
            f"APUg2 operand {semantic_name!r} must have shape {shape}, "
            f"got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"APUg2 operand {semantic_name!r} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"APUg2 output {semantic_name!r} must be writable")
    return value


def _calibrate_stage_recipe(recipe):
    calibration = calibrate_apu_g2_recipe_from_measured_tile(recipe)
    return calibration, "interpolated" in calibration.basis


def _apu_g2_pack_count(elements):
    elements = int(elements)
    if elements <= 0:
        raise ValueError("APUg2 pack extent must be positive")
    return math.ceil(elements / (4 * 65_536))


def _build_tiled_contraction_recipe(
    output_shape,
    tiling,
    *,
    batch_rank=0,
    alpha=1,
    beta=1,
    name,
):
    """Build an exact small recipe or a compact LARGE shard composition.

    LARGE task grids can contain tens of thousands of identical physical dot
    shards.  Expanding every shard into the cost graph is quadratic in the
    generic calendar evaluator and adds no physical information.  The compact
    form retains one exact recipe for every distinct
    ``(outputs, reduction chunk, epilogue)`` class and records its multiplicity;
    calibration is the measured per-class total multiplied by that count.
    """

    output_shape = tuple(int(extent) for extent in output_shape)
    if (alpha is None) != (beta is None):
        raise ValueError("alpha and beta must both be supplied or both omitted")
    if tiling.reduction_tile_count == 1 and tiling.task_count <= 64:
        recipe = build_apu_g2_u16_contraction_recipe(
            output_shape,
            tiling.reduction_extent,
            batch_rank=batch_rank,
            alpha=alpha,
            beta=beta,
            output_tile_extent=tiling.tile_capacity,
            name=name,
        )
        calibration = calibrate_apu_g2_recipe_from_measured_tile(recipe)
        return recipe, calibration, "interpolated" in calibration.basis

    classes = {}
    for output_tile in range(tiling.tile_count):
        output_begin, output_end = tiling.tile_bounds(output_tile)
        active_outputs = output_end - output_begin
        for reduction_tile in range(tiling.reduction_tile_count):
            reduction_begin, reduction_end = tiling.reduction_bounds(reduction_tile)
            chunk_extent = reduction_end - reduction_begin
            if reduction_tile == 0:
                task_alpha, task_beta = alpha, beta
            else:
                task_alpha, task_beta = 1, 1
            key = (active_outputs, chunk_extent, task_alpha, task_beta)
            classes[key] = classes.get(key, 0) + 1

    class_recipes = []
    class_calibrations = []
    class_metadata = []
    extrapolated = False
    for index, ((active, chunk, task_alpha, task_beta), count) in enumerate(
        classes.items()
    ):
        recipe = build_apu_g2_u16_contraction_recipe(
            (active,),
            chunk,
            alpha=task_alpha,
            beta=task_beta,
            output_tile_extent=active,
            name=f"{name}_task_class_{index}",
        )
        calibration = calibrate_apu_g2_recipe_from_measured_tile(recipe)
        class_recipes.append(recipe)
        class_calibrations.append(calibration)
        extrapolated = extrapolated or "interpolated" in calibration.basis
        class_metadata.append(
            {
                "active_outputs": active,
                "reduction_extent": chunk,
                "alpha": task_alpha,
                "beta": task_beta,
                "count": count,
                "cycles_per_task": calibration.total_cycles,
                "measured_ticks_per_task": calibration.measured_ticks_per_pipeline,
            }
        )

    chained = chain_apu_g2_recipes(
        class_recipes,
        name=name,
        stage_names=tuple(f"task_class_{i}" for i in range(len(class_recipes))),
    )
    recipe = APUG2Recipe(
        chained.name,
        chained.operations,
        chained.certificate,
        metadata={
            **dict(chained.metadata),
            "operation": "rank_n_contraction_composition",
            "output_shape": output_shape,
            "output_rank": len(output_shape),
            "batch_rank": int(batch_rank),
            "output_extent": tiling.output_extent,
            "reduction_extent": tiling.reduction_extent,
            "output_tile_extent": tiling.tile_capacity,
            "output_tile_count": tiling.tile_count,
            "reduction_tile_extent": tiling.reduction_tile_extent,
            "reduction_tile_count": tiling.reduction_tile_count,
            "dot_tile_count": tiling.task_count,
            "compact_task_composition": True,
            "task_classes": tuple(class_metadata),
            "logical_vector_lane_updates": sum(
                child.certificate.vector_lane_updates * item["count"]
                for child, item in zip(class_recipes, class_metadata)
            ),
            "epilogue": (
                "alpha_dot_plus_beta_accumulator"
                if alpha is not None
                else "raw_first_chunk_then_modular_accumulation"
            ),
            "alpha": alpha,
            "beta": beta,
        },
    )
    total_cycles = sum(
        calibration.total_cycles * item["count"]
        for calibration, item in zip(class_calibrations, class_metadata)
    )
    measured = [
        calibration.measured_ticks_per_pipeline * item["count"]
        for calibration, item in zip(class_calibrations, class_metadata)
        if calibration.measured_ticks_per_pipeline is not None
    ]
    repetitions = [
        calibration.repetitions
        for calibration in class_calibrations
        if calibration.repetitions is not None
    ]
    calibration = APUG2RecipeCalibration.normalized(
        recipe,
        total_cycles,
        measured_ticks_per_pipeline=sum(measured) if measured else None,
        repetitions=min(repetitions) if repetitions else None,
        basis=(
            "real_card_large_shard_composition_with_interpolation"
            if extrapolated
            else "real_card_large_shard_composition"
        ),
    )
    return recipe, calibration, extrapolated


def _topological_rank_n_plans(plans):
    if not plans:
        return ()
    module = plans[0].module
    by_region = {plan.region_id: plan for plan in plans}
    dependencies = {
        plan.region_id: {
            edge.producer_region
            for edge in module.dependencies
            if edge.consumer_region == plan.region_id
        }
        for plan in plans
    }
    ordered = []
    emitted = set()
    while len(ordered) < len(plans):
        progressed = False
        for plan in plans:
            if plan.region_id in emitted:
                continue
            unknown = dependencies[plan.region_id] - set(by_region)
            if unknown:
                raise UnsupportedAPUG2ContractionError(
                    f"APUg2 chain has unknown producer regions {sorted(unknown)}"
                )
            if dependencies[plan.region_id] <= emitted:
                ordered.append(plan)
                emitted.add(plan.region_id)
                progressed = True
        if not progressed:
            remaining = sorted(set(by_region) - emitted)
            raise UnsupportedAPUG2ContractionError(
                f"APUg2 chain has a cyclic or unsatisfied dependency: {remaining}"
            )
    return tuple(ordered)


class APUG2GemvCallable:
    """Preserve an ordinary Allo signature while executing direct VL64 GEMV."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        if cost is None:
            raise ValueError("APUg2 GEMV requires its executable cost spec")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.module_manifest = discover_apu_g2_module_manifest(self.module)
        self.plan = plan_apu_g2_gemv(self.module)
        self.analysis = self.plan.analysis
        self.program = APUG2Program(
            operation="gemv_u16",
            shape=(self.plan.output_extent, self.plan.reduction_extent),
            repetitions=8,
            name=getattr(workload, "__name__", "apu_g2_u16_gemv"),
        )
        self.kernel = APUG2Callable(self.program, target, cost=cost, backend=backend)
        self.cost = cost
        self.backend = self.kernel.backend
        self.layout = self.plan.layout
        self.execution_graph = self.kernel.execution_graph
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_gemv_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.kernel.estimate()

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        try:
            matrix = values[self.plan.matrix.value]
            vector = values[self.plan.vector.value]
            output = values[self.plan.output.value]
        except KeyError as error:
            raise TypeError(
                f"ordinary GEMV signature is missing analyzed operand {error.args[0]!r}"
            ) from error
        if self.plan.matrix_transposed:
            matrix = np.ascontiguousarray(np.asarray(matrix).T)
        result = self.kernel(matrix, vector, output, out=output)
        if result.extra is not None and self.backend != "virtual":
            result.extra["outputs"][self.plan.output.value] = np.asarray(
                result.extra["outputs"]["out"]
            )
            result.extra["contraction"] = {
                "matrix": self.plan.matrix.value,
                "vector": self.plan.vector.value,
                "output": self.plan.output.value,
                "matrix_transposed": self.plan.matrix_transposed,
            }
        self.last_result = result
        return result

    run = __call__


class APUG2AtaxCallable:
    """One ordinary Allo signature backed by one fused APUg2 ATAX task."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 ATAX requires a cost bound to the same target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.module_manifest = discover_apu_g2_module_manifest(self.module)
        self.plan = plan_apu_g2_atax(self.module)
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.execution_graph = build_apu_g2_atax_graph(
            target,
            cost,
            row_extent=self.plan.row_extent,
            column_extent=self.plan.column_extent,
        )
        self.execution_graph.metadata.update(
            {
                "program": "atax_u16",
                "work_grid": (16,),
                "vl64_calls": self.execution_graph.metadata["vl64_compute_calls"]
                + self.execution_graph.metadata["resident_transform_calls"],
                "execution": "direct_vl64_resident_fusion",
                "analytical": True,
            }
        )
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_atax_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        try:
            matrix = values[self.plan.matrix.value]
            vector = values[self.plan.vector.value]
            output = values[self.plan.output.value]
        except KeyError as error:
            raise TypeError(
                f"ordinary ATAX signature is missing analyzed operand {error.args[0]!r}"
            ) from error
        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 ATAX cost evaluation; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "resident_intermediate": True,
                    "hardware_tasks": 1,
                    "vl64_calls": self.execution_graph.metadata["vl64_calls"],
                },
            )
        else:
            from .apu_g2_atax_runtime import run_apu_g2_u16_atax

            result = run_apu_g2_u16_atax(matrix, vector, output, repetitions=4)
            result.extra["outputs"][self.plan.intermediate.value] = np.asarray(
                result.extra["outputs"]["tmp"]
            )
            result.extra["outputs"][self.plan.output.value] = np.asarray(
                result.extra["outputs"]["out"]
            )
            result.extra["contractions"] = {
                "stage_m": self.plan.stage_m.analysis.function,
                "stage_n": self.plan.stage_n.analysis.function,
                "matrix": self.plan.matrix.value,
                "vector": self.plan.vector.value,
                "intermediate": self.plan.intermediate.value,
                "output": self.plan.output.value,
            }
        self.last_result = result
        return result

    run = __call__


class APUG2IndependentContractionsCallable:
    """Fuse independent ordinary GEMVs into one four-set dot-tile task."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError(
                "APUg2 independent contractions require cost bound to the target"
            )
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        analyses = analyze_contractions(self.module)
        self.module_manifest = discover_apu_g2_module_manifest(analyses)
        if self.module_manifest.contraction_topology != "independent":
            raise UnsupportedAPUG2ContractionError(
                "independent contraction callable requires independent roots"
            )
        self.plans = tuple(plan_apu_g2_gemv(analysis) for analysis in analyses)
        self.reduction_extent = max(plan.reduction_extent for plan in self.plans)
        self.output_extent = sum(plan.output_extent for plan in self.plans)
        self.tiling = APUG2DotTiling(self.output_extent, self.reduction_extent)
        self.streaming_fusion = (
            self.tiling.tile_count != 1 or self.tiling.reduction_tile_count != 1
        )
        if self.streaming_fusion:
            self.recipe, self.calibration, _ = _build_tiled_contraction_recipe(
                (self.output_extent,),
                self.tiling,
                alpha=1,
                beta=1,
                name="apu_g2_u16_fused_independent_gemv",
            )
        else:
            self.recipe = build_apu_g2_u16_dot_tile_recipe(
                self.output_extent,
                self.reduction_extent,
                alpha=1,
                beta=1,
            )
            self.calibration = calibrate_apu_g2_recipe_from_measured_tile(self.recipe)
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "independent_contractions_u16",
                "work_grid": (16,),
                "hardware_tasks": 1,
                "contraction_topology": "independent",
                "calibration_ticks": self.calibration.total_cycles,
                "measured_ticks_per_pipeline": (
                    self.calibration.measured_ticks_per_pipeline
                ),
                "calibration_extrapolated": ("interpolated" in self.calibration.basis),
                "calibration_basis": self.calibration.basis,
            }
        )
        if self.streaming_fusion:
            from .apu_g2_u16_gemm_runtime import (
                APUG2_U16_GEMM_BMAX,
                APUG2_U16_GEMM_CHUNK,
            )

            columns = len(self.plans)
            reduction_tiles = math.ceil(self.reduction_extent / APUG2_U16_GEMM_CHUNK)
            self.execution_graph.metadata.update(
                {
                    "program": "fused_independent_gemv_u16",
                    "hardware_tasks": reduction_tiles,
                    "transport_schedule": {
                        "kind": "fused_independent_gemv",
                        "rows": self.output_extent,
                        "columns": columns,
                        "reduction": self.reduction_extent,
                        "batch_columns": APUG2_U16_GEMM_BMAX,
                        "reduction_tile": APUG2_U16_GEMM_CHUNK,
                        "hardware_tasks": reduction_tiles,
                        "weight_uploads": reduction_tiles,
                        "output_readbacks": 1,
                        "resident_accumulator": True,
                        "contiguous_readback": True,
                        "host_wall_estimate": _gemm_wall_estimate(
                            self.output_extent,
                            self.reduction_extent,
                            columns,
                            APUG2_U16_GEMM_BMAX,
                        ),
                    },
                }
            )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(
            workload, "__name__", "apu_g2_independent_contractions_workload"
        )
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(value, name, shape):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError(f"APUg2 operand {name!r} must be a NumPy uint16 array")
        if value.shape != shape:
            raise ValueError(
                f"APUg2 operand {name!r} must have shape {shape}, got {value.shape}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"APUg2 operand {name!r} must be C-contiguous")
        return value

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        left = np.zeros((self.output_extent, self.reduction_extent), dtype=np.uint16)
        right = np.zeros_like(left)
        outputs = []
        offset = 0
        for plan in self.plans:
            matrix = self._array(
                values[plan.matrix.value], plan.matrix.value, plan.matrix.shape
            )
            vector = self._array(
                values[plan.vector.value],
                plan.vector.value,
                (plan.reduction_extent,),
            )
            output = self._array(
                values[plan.output.value],
                plan.output.value,
                (plan.output_extent,),
            )
            oriented = matrix.T if plan.matrix_transposed else matrix
            stop = offset + plan.output_extent
            left[offset:stop, : plan.reduction_extent] = oriented
            right[offset:stop, : plan.reduction_extent] = vector[None, :]
            outputs.append((plan.output.value, output, offset, stop))
            offset = stop

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout=(
                    "virtual APUg2 independent-contraction cost evaluation; "
                    "hardware was not executed"
                ),
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": 1,
                    "contraction_topology": "independent",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                },
            )
        elif self.streaming_fusion:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            columns = len(self.plans)
            matrix = np.zeros(
                (self.output_extent, self.reduction_extent), dtype=np.uint16
            )
            vectors = np.zeros((self.reduction_extent, columns), dtype=np.uint16)
            accumulator = np.zeros((self.output_extent, columns), dtype=np.uint16)
            for column, plan in enumerate(self.plans):
                source = values[plan.matrix.value]
                oriented = source.T if plan.matrix_transposed else source
                begin = sum(item.output_extent for item in self.plans[:column])
                end = begin + plan.output_extent
                matrix[begin:end, : plan.reduction_extent] = oriented
                vectors[: plan.reduction_extent, column] = values[plan.vector.value]
                accumulator[begin:end, column] = values[plan.output.value]

            result = run_apu_g2_u16_gemm(
                matrix,
                vectors,
                accumulator,
                alpha=1,
                beta=1,
            )
            combined = np.asarray(result.extra["outputs"]["out"])
            output_payload = {}
            for column, plan in enumerate(self.plans):
                begin = sum(item.output_extent for item in self.plans[:column])
                end = begin + plan.output_extent
                output = values[plan.output.value]
                np.copyto(output, combined[begin:end, column])
                output_payload[plan.output.value] = output.copy()
            result.extra["outputs"] = output_payload
            result.extra.update(
                {
                    "contraction_topology": "independent",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "regions": [
                        region.manifest() for region in self.module_manifest.regions
                    ],
                    "schedule": dict(
                        self.execution_graph.metadata["transport_schedule"]
                    ),
                }
            )
        else:
            from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile

            accumulator = np.concatenate(
                [values[plan.output.value] for plan in self.plans]
            )
            result = run_apu_g2_u16_dot_tile(
                left,
                right,
                accumulator=np.ascontiguousarray(accumulator),
                alpha=1,
                beta=1,
                repetitions=8,
            )
            combined = np.asarray(result.extra["outputs"]["out"])
            for name, output, begin, end in outputs:
                np.copyto(output, combined[begin:end])
                result.extra["outputs"][name] = combined[begin:end].copy()
            result.extra.update(
                {
                    "hardware_tasks": 1,
                    "contraction_topology": "independent",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "regions": [
                        region.manifest() for region in self.module_manifest.regions
                    ],
                }
            )
        self.last_result = result
        return result

    run = __call__


class APUG2RankNContractionCallable:
    """Execute one ordinary rank-N uint16 contraction as temporal dot tiles."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError(
                "APUg2 rank-N contraction requires cost bound to the target"
            )
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan_apu_g2_rank_n_contraction(self.module)
        self.module_manifest = self.plan.module
        if self.plan.batch_local_accumulator:
            raise UnsupportedAPUG2ContractionError(
                "rank-N execution currently requires the accumulator to index "
                "every flattened output axis"
            )
        dot_axis_names = tuple(name for name, _extent in self.plan.dot_axes)
        if self.plan.initial_output.result_axes != dot_axis_names:
            raise UnsupportedAPUG2ContractionError(
                "rank-N execution requires an initial accumulator in dot-axis order"
            )
        if self.plan.output.result_axes != dot_axis_names:
            raise UnsupportedAPUG2ContractionError(
                "rank-N execution requires output storage in dot-axis order"
            )
        if (
            self.plan.lhs.value == self.plan.output.value
            or self.plan.rhs.value == self.plan.output.value
        ):
            raise UnsupportedAPUG2ContractionError(
                "rank-N execution requires read operands not to alias output storage"
            )
        self.epilogue = self.plan.epilogue

        output_shape = tuple(extent for _name, extent in self.plan.dot_axes)
        self.recipe, self.calibration, extrapolated = _build_tiled_contraction_recipe(
            output_shape,
            self.plan.tiling,
            batch_rank=len(self.plan.batch_axes),
            alpha=self.epilogue[0],
            beta=self.epilogue[1],
            name="apu_g2_u16_rank_n_contraction",
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "rank_n_contraction_u16",
                "work_grid": (16,),
                "hardware_tasks": self.plan.tiling.task_count,
                "contraction_topology": "single",
                "calibration_extrapolated": extrapolated,
                "affine_access_maps": {
                    "lhs": self.plan.lhs.manifest(),
                    "rhs": self.plan.rhs.manifest(),
                    "initial_output": self.plan.initial_output.manifest(),
                    "output": self.plan.output.manifest(),
                },
                "runtime_epilogue": {
                    "alpha": self.epilogue[0],
                    "beta": self.epilogue[1],
                },
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_rank_n_contraction")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(values, access, *, writable=False):
        try:
            value = values[access.value]
        except KeyError as error:
            raise TypeError(
                f"ordinary contraction signature is missing {access.value!r}"
            ) from error
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError(
                f"APUg2 operand {access.value!r} must be a NumPy uint16 array"
            )
        if value.shape != access.shape:
            raise ValueError(
                f"APUg2 operand {access.value!r} must have shape {access.shape}, "
                f"got {value.shape}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"APUg2 operand {access.value!r} must be C-contiguous")
        if writable and not value.flags.writeable:
            raise ValueError(f"APUg2 output {access.value!r} must be writable")
        return value

    def _pack_tile(self, values, tile, reduction_tile=0, accumulator=None):
        lhs = self._array(values, self.plan.lhs)
        rhs = self._array(values, self.plan.rhs)
        initial = self._array(values, self.plan.initial_output)
        begin, end = self.plan.tiling.tile_bounds(tile)
        reduction_begin, reduction_end = self.plan.tiling.reduction_bounds(
            reduction_tile
        )
        extent = end - begin
        reduction_extent = reduction_end - reduction_begin
        left = np.empty((extent, reduction_extent), dtype=np.uint16)
        right = np.empty_like(left)
        if accumulator is None:
            accumulator = np.empty(extent, dtype=np.uint16)
        else:
            accumulator = np.ascontiguousarray(accumulator, dtype=np.uint16)
        output_sites = []
        for local, flat in enumerate(range(begin, end)):
            dot_coordinates = self.plan.unflatten_output(flat)
            initial_domain = dot_coordinates + (0,)
            if reduction_tile == 0:
                accumulator[local] = initial[
                    self.plan.initial_output.apply(initial_domain)
                ]
            output_sites.append(self.plan.output.apply(initial_domain))
            for local_reduction, reduction in enumerate(
                range(reduction_begin, reduction_end)
            ):
                domain = dot_coordinates + (reduction,)
                left[local, local_reduction] = lhs[self.plan.lhs.apply(domain)]
                right[local, local_reduction] = rhs[self.plan.rhs.apply(domain)]
        return left, right, accumulator, tuple(output_sites)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        # Validate all arrays even in virtual mode; virtual costing must not
        # silently accept a call the hardware ABI would reject.
        self._array(values, self.plan.lhs)
        self._array(values, self.plan.rhs)
        self._array(values, self.plan.initial_output)
        output = self._array(values, self.plan.output, writable=True)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 rank-N contraction; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.plan.tiling.task_count,
                    "contraction_topology": "single",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "plan": self.plan.manifest(),
                    "epilogue": {
                        "alpha": self.epilogue[0],
                        "beta": self.epilogue[1],
                    },
                },
            )
        else:
            from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile

            tile_results = []
            observed_tiles = []
            for tile in range(self.plan.tiling.tile_count):
                accumulator = None
                output_sites = ()
                for reduction_tile in range(self.plan.tiling.reduction_tile_count):
                    left, right, accumulator, output_sites = self._pack_tile(
                        values, tile, reduction_tile, accumulator
                    )
                    tile_result = run_apu_g2_u16_dot_tile(
                        left,
                        right,
                        accumulator=accumulator,
                        alpha=self.epilogue[0],
                        beta=(self.epilogue[1] if reduction_tile == 0 else 1),
                        repetitions=8,
                    )
                    accumulator = np.asarray(tile_result.extra["outputs"]["out"])
                    observed_tiles.append(
                        {
                            "output_tile": tile,
                            "reduction_tile": reduction_tile,
                            "output": accumulator.copy(),
                        }
                    )
                    tile_results.append(tile_result)
                observed = accumulator
                for value, site in zip(observed, output_sites):
                    output[site] = value

            result = RunResult(
                cycles=sum(item.cycles for item in tile_results),
                stdout="\n".join(
                    f"tile={index}\n{item.stdout}"
                    for index, item in enumerate(tile_results)
                ),
                backend="apu_v2",
                extra={
                    "outputs": {
                        "out": output.copy(),
                        self.plan.output.value: output.copy(),
                    },
                    "hardware_tasks": len(tile_results),
                    "contraction_topology": "single",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "plan": self.plan.manifest(),
                    "per_tile_cycles": [item.cycles for item in tile_results],
                    "total_ticks": sum(
                        item.extra["total_ticks"] for item in tile_results
                    ),
                    "final_pipeline_ticks": sum(
                        item.extra["final_pipeline_ticks"] for item in tile_results
                    ),
                    "repetitions": 8,
                    "epilogue": {
                        "alpha": self.epilogue[0],
                        "beta": self.epilogue[1],
                    },
                    "tile_outputs": observed_tiles,
                    "sources": tile_results[0].extra["sources"],
                    "projects": [item.extra["project"] for item in tile_results],
                },
            )
        self.last_result = result
        return result

    run = __call__


def _matrixized_gemm_spec(plan):
    if len(plan.dot_axes) < 2 or plan.batch_axes:
        return None
    if (
        plan.initial_output.value != plan.output.value
        or plan.lhs.value == plan.output.value
        or plan.rhs.value == plan.output.value
    ):
        return None
    output_axes = tuple(name for name, _extent in plan.dot_axes)
    extents = dict(plan.dot_axes)
    reduction = plan.reduction_axis[0]
    row_axes = tuple(axis for axis in plan.lhs.result_axes if axis != reduction)
    column_axes = tuple(axis for axis in plan.rhs.result_axes if axis != reduction)
    if (
        not row_axes
        or not column_axes
        or set(row_axes) & set(column_axes)
        or row_axes + column_axes != output_axes
        or set(plan.lhs.result_axes) != set(row_axes + (reduction,))
        or set(plan.rhs.result_axes) != set((reduction,) + column_axes)
        or plan.initial_output.result_axes != output_axes
        or plan.output.result_axes != output_axes
    ):
        return None
    return {
        "row_axes": row_axes,
        "column_axes": column_axes,
        "rows": math.prod(extents[axis] for axis in row_axes),
        "columns": math.prod(extents[axis] for axis in column_axes),
        "reduction": plan.reduction_axis[1],
    }


def _matrixized_gemm_operands(plan, values):
    spec = _matrixized_gemm_spec(plan)
    if spec is None:
        raise UnsupportedAPUG2ContractionError(
            "contraction cannot be factorized into matrix rows and columns"
        )
    lhs_value = APUG2RankNContractionCallable._array(values, plan.lhs)
    rhs_value = APUG2RankNContractionCallable._array(values, plan.rhs)
    lhs_axes = spec["row_axes"] + (plan.reduction_axis[0],)
    rhs_axes = (plan.reduction_axis[0],) + spec["column_axes"]
    lhs_permutation = tuple(plan.lhs.result_axes.index(axis) for axis in lhs_axes)
    rhs_permutation = tuple(plan.rhs.result_axes.index(axis) for axis in rhs_axes)
    lhs = np.ascontiguousarray(lhs_value.transpose(lhs_permutation)).reshape(
        spec["rows"], spec["reduction"]
    )
    rhs = np.ascontiguousarray(rhs_value.transpose(rhs_permutation)).reshape(
        spec["reduction"], spec["columns"]
    )
    return spec, lhs, rhs


def _is_canonical_u16_gemm_plan(plan):
    return _matrixized_gemm_spec(plan) is not None


def _gemm_wall_estimate(
    rows,
    reduction,
    columns,
    batch_columns=APUG2_GEMM_MAX_BATCH_COLUMNS,
):
    return estimate_apu_g2_u16_gemm_wall_us(
        rows,
        reduction,
        columns,
        batch_columns=batch_columns,
    )


@dataclass(frozen=True)
class APUG2PersistentGemmSchedule:
    """Finite transport decisions for the persistent uint16 GEMM runtime."""

    batch_columns: int
    reduction_tile: int


@dataclass(frozen=True)
class APUG2PersistentGemmMaterialization:
    """Physical persistent-GEMM realization proven before wall scoring."""

    schedule: APUG2PersistentGemmSchedule
    rows: int
    reduction: int
    columns: int
    epilogue: tuple[int, int]
    semantic_fingerprint: str
    source_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    runtime_artifact: object = field(repr=False, compare=False)

    @property
    def current_source_fingerprint(self):
        from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

        return self.runtime_artifact.current_source_fingerprint(run_apu_g2_u16_gemm)

    @property
    def fingerprint(self):
        payload = {
            "kind": "apu-g2-persistent-gemm-materialization-v1",
            "batch_columns": self.schedule.batch_columns,
            "reduction_tile": self.schedule.reduction_tile,
            "rows": self.rows,
            "reduction": self.reduction,
            "columns": self.columns,
            "epilogue": list(self.epilogue),
            "semantic_fingerprint": self.semantic_fingerprint,
            "source_fingerprint": self.source_fingerprint,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def promotion_materialization_fingerprint(self):
        from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

        current = self.runtime_artifact.promotion_source_fingerprint(
            run_apu_g2_u16_gemm
        )
        if current is None or current != self.source_fingerprint:
            return None
        return self.fingerprint

    @property
    def promotion_platform_fingerprint(self):
        if self.runtime_artifact.platform_fingerprint is None:
            return None
        return self.runtime_artifact.current_platform_fingerprint()


def _apu_g2_runtime_source_fingerprint():
    from .apu_g2_u16_gemm_runtime import (
        freeze_apu_g2_u16_gemm_runtime_artifact,
    )

    try:
        artifact = freeze_apu_g2_u16_gemm_runtime_artifact()
    except RuntimeError as error:
        raise InfeasibleSchedule(str(error)) from error
    return artifact.source_fingerprint, artifact.source_hashes


def _apu_g2_physical_row_capacity(target):
    if getattr(target, "name", None) != "apu_v2":
        raise InfeasibleSchedule("persistent GEMM requires the apu_v2 target")
    try:
        l1_columns = int(target.l1.geometry["cols"])
        grouped_columns = int(target.l1.geometry["groups"]) * int(
            target.l1.geometry["cols_per_group"]
        )
        vl64_columns = int(target.rwen.axes["column"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise InfeasibleSchedule(
            "apu_v2 target is missing physical VL64/L1 row geometry"
        ) from error
    capacity = min(l1_columns, grouped_columns, vl64_columns)
    if capacity <= 0:
        raise InfeasibleSchedule("apu_v2 target has invalid VL64/L1 row capacity")
    return capacity


def _apu_g2_persistent_semantic_fingerprint(plan):
    dot_axis_positions = {
        name: index for index, (name, _extent) in enumerate(plan.dot_axes)
    }

    def access_manifest(access):
        return {
            "dtype": access.dtype,
            "shape": list(access.shape),
            "mode": access.mode,
            "axis_positions": list(access.axis_positions),
        }

    payload = {
        "kind": "apu-g2-persistent-gemm-semantics-v1",
        "dot_extents": [extent for _name, extent in plan.dot_axes],
        "batch_axis_positions": [dot_axis_positions[name] for name in plan.batch_axes],
        "output_axis_positions": [
            dot_axis_positions[name] for name in plan.output_axes
        ],
        "reduction_extent": int(plan.reduction_axis[1]),
        "epilogue": list(plan.epilogue),
        "batch_local_accumulator": bool(plan.batch_local_accumulator),
        "storage_relations": {
            "lhs_is_rhs": plan.lhs.value == plan.rhs.value,
            "accumulator_is_output": (plan.initial_output.value == plan.output.value),
            "operands_do_not_alias_output": (
                plan.lhs.value != plan.output.value
                and plan.rhs.value != plan.output.value
            ),
        },
        "accesses": [
            access_manifest(plan.lhs),
            access_manifest(plan.rhs),
            access_manifest(plan.initial_output),
            access_manifest(plan.output),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _materialize_apu_g2_persistent_gemm_schedule(plan, target, schedule):
    matrix_spec = _matrixized_gemm_spec(plan)
    if matrix_spec is None:
        raise InfeasibleSchedule(
            "persistent GEMM requires canonical A[row,k] * B[k,column] access"
        )

    from .apu_g2_u16_gemm_runtime import (
        APUG2_U16_GEMM_BMAX,
        APUG2_U16_GEMM_CHUNK,
        freeze_apu_g2_u16_gemm_runtime_artifact,
    )

    if not 1 <= int(schedule.batch_columns) <= APUG2_U16_GEMM_BMAX:
        raise InfeasibleSchedule(
            "persistent GEMM batch columns must fit the resident accumulator"
        )
    if int(schedule.reduction_tile) != APUG2_U16_GEMM_CHUNK:
        raise InfeasibleSchedule(
            "persistent GEMM reduction tile is fixed by the current runtime at "
            f"{APUG2_U16_GEMM_CHUNK}"
        )

    rows = int(matrix_spec["rows"])
    capacity = _apu_g2_physical_row_capacity(target)
    if rows > capacity:
        raise InfeasibleSchedule(
            f"persistent GEMM rows {rows} exceed physical VL64/L1 capacity "
            f"{capacity}"
        )
    try:
        runtime_artifact = freeze_apu_g2_u16_gemm_runtime_artifact()
    except RuntimeError as error:
        raise InfeasibleSchedule(str(error)) from error
    return APUG2PersistentGemmMaterialization(
        schedule=schedule,
        rows=rows,
        reduction=int(matrix_spec["reduction"]),
        columns=int(matrix_spec["columns"]),
        epilogue=tuple(plan.epilogue),
        semantic_fingerprint=_apu_g2_persistent_semantic_fingerprint(plan),
        source_fingerprint=runtime_artifact.source_fingerprint,
        source_hashes=runtime_artifact.source_hashes,
        runtime_artifact=runtime_artifact,
    )


def _score_apu_g2_persistent_gemm_materialization(materialized, cost):
    from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

    try:
        materialized.runtime_artifact.assert_current(run_apu_g2_u16_gemm)
    except RuntimeError as error:
        raise InfeasibleSchedule(str(error)) from error
    return cost.score_materialization(materialized)


def search_apu_g2_persistent_gemm_schedules(plan, target, cost):
    """Materialize and wall-score calibrated persistent-GEMM schedules."""

    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError(
            "persistent GEMM schedule search requires cost bound to the target"
        )
    if cost.spec.materialization_scorer is None:
        raise TypeError(
            "persistent GEMM schedule search requires a materialization cost scorer"
        )

    return grid_search(
        (
            DecisionDomain("batch_columns", range(1, APUG2_GEMM_MAX_BATCH_COLUMNS + 1)),
            DecisionDomain("reduction_tile", (APUG2_GEMM_MAX_REDUCTION_TILE,)),
        ),
        build=lambda decisions: APUG2PersistentGemmSchedule(
            batch_columns=decisions["batch_columns"],
            reduction_tile=decisions["reduction_tile"],
        ),
        materialize=lambda schedule: _materialize_apu_g2_persistent_gemm_schedule(
            plan, target, schedule
        ),
        score=lambda materialized: _score_apu_g2_persistent_gemm_materialization(
            materialized, cost
        ),
        objective=lambda estimate: float(estimate["wall_us"]),
        objective_domain=ScheduleObjectiveDomain.fingerprinted_target(
            metric="wall_us",
            target="apu_v2",
            model_fingerprint=cost.fingerprint,
            fidelity="calibrated",
            scope="whole_program_transport",
            unit="microseconds",
            direction="minimize",
        ),
        incumbent={
            "batch_columns": APUG2_GEMM_MAX_BATCH_COLUMNS,
            "reduction_tile": APUG2_GEMM_MAX_REDUCTION_TILE,
        },
    )


class APUG2ColumnBatchedGemmCallable(APUG2RankNContractionCallable):
    """Execute canonical rank-2 contractions with transport-aware GEMM."""

    def __init__(
        self,
        workload,
        target,
        schedule,
        *,
        cost,
        backend=None,
        promotion_gate=None,
    ):
        promotion_gate = validate_schedule_promotion_gate(promotion_gate)
        super().__init__(workload, target, schedule, cost=cost, backend=backend)
        if not _is_canonical_u16_gemm_plan(self.plan):
            raise UnsupportedAPUG2ContractionError(
                "column-batched GEMM requires A[row,k] * B[k,column]"
            )
        self.schedule_search_result = search_apu_g2_persistent_gemm_schedules(
            self.plan, target, cost
        )
        self.schedule_activation = guarded_schedule_activation(
            self.schedule_search_result,
            promotion_gate=promotion_gate,
        )
        selected = self.schedule_activation.active
        self.fallback_reason = self.schedule_activation.fallback_reason
        self.recommended_transport_schedule = (
            self.schedule_activation.recommended.materialized.schedule
        )
        self.recommended_transport_estimate = self.schedule_activation.recommended.score
        self.selected_transport_schedule = selected.materialized.schedule
        self.selected_transport_estimate = selected.score
        materialized = selected.materialized
        self.selected_transport_materialization = materialized
        rows = materialized.rows
        columns = materialized.columns
        reduction = materialized.reduction
        batch_columns = self.selected_transport_schedule.batch_columns
        reduction_tile = self.selected_transport_schedule.reduction_tile
        column_batches = (columns + batch_columns - 1) // batch_columns
        reduction_tiles = (reduction + reduction_tile - 1) // reduction_tile
        hardware_tasks = column_batches * reduction_tiles
        self.execution_graph.metadata.update(
            {
                "program": "column_batched_gemm_u16",
                "hardware_tasks": hardware_tasks,
                "transport_schedule": {
                    "kind": "column_batched_u16_gemm",
                    "rows": rows,
                    "columns": columns,
                    "reduction": reduction,
                    "batch_columns": batch_columns,
                    "column_batches": column_batches,
                    "reduction_tile": reduction_tile,
                    "reduction_tiles": reduction_tiles,
                    "weight_uploads": hardware_tasks,
                    "output_readbacks": column_batches,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                    "host_wall_estimate": dict(self.selected_transport_estimate),
                },
            }
        )

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        accumulator = self._array(values, self.plan.initial_output)
        output = self._array(values, self.plan.output, writable=True)
        _spec, lhs, rhs = _matrixized_gemm_operands(self.plan, values)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 column-batched uint16 GEMM",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "contraction_topology": "single",
                    "schedule": dict(
                        self.execution_graph.metadata["transport_schedule"]
                    ),
                    "plan": self.plan.manifest(),
                },
            )
        else:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            result = run_apu_g2_u16_gemm(
                lhs,
                rhs,
                np.ascontiguousarray(accumulator.reshape(lhs.shape[0], rhs.shape[1])),
                alpha=self.epilogue[0],
                beta=self.epilogue[1],
                batch_columns=self.selected_transport_schedule.batch_columns,
                runtime_artifact=(
                    self.selected_transport_materialization.runtime_artifact
                ),
            )
            observed = np.asarray(result.extra["outputs"]["out"])
            np.copyto(output, observed.reshape(output.shape))
            result.extra["outputs"][self.plan.output.value] = output.copy()
            result.extra.update(
                {
                    "contraction_topology": "single",
                    "compiled_workload": True,
                    "plan": self.plan.manifest(),
                    "epilogue": {
                        "alpha": self.epilogue[0],
                        "beta": self.epilogue[1],
                    },
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                }
            )
        self.last_result = result
        return result

    run = __call__


def _streaming_gemv_roles(plan):
    if len(plan.dot_axes) != 1:
        return None
    row = plan.dot_axes[0][0]
    reduction = plan.reduction_axis[0]
    accesses = (plan.lhs, plan.rhs)
    matrices = [access for access in accesses if len(access.shape) == 2]
    vectors = [access for access in accesses if len(access.shape) == 1]
    if len(matrices) != 1 or len(vectors) != 1:
        return None
    matrix, vector = matrices[0], vectors[0]
    if vector.result_axes != (reduction,):
        return None
    if matrix.result_axes == (row, reduction):
        transposed = False
    elif matrix.result_axes == (reduction, row):
        transposed = True
    else:
        return None
    if plan.initial_output.result_axes != (row,) or plan.output.result_axes != (row,):
        return None
    return matrix, vector, transposed


class APUG2StreamingGemvCallable(APUG2RankNContractionCallable):
    """Execute arbitrary-K GEMV as one-column persistent GEMM."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        super().__init__(workload, target, schedule, cost=cost, backend=backend)
        roles = _streaming_gemv_roles(self.plan)
        if roles is None:
            raise UnsupportedAPUG2ContractionError(
                "streaming GEMV requires one matrix and one reduction vector"
            )
        self.matrix, self.vector, self.matrix_transposed = roles
        from .apu_g2_u16_gemm_runtime import APUG2_U16_GEMM_CHUNK

        reduction_tiles = (
            self.plan.reduction_axis[1] + APUG2_U16_GEMM_CHUNK - 1
        ) // APUG2_U16_GEMM_CHUNK
        self.execution_graph.metadata.update(
            {
                "program": "streaming_gemv_u16",
                "hardware_tasks": reduction_tiles,
                "transport_schedule": {
                    "kind": "column_batched_u16_gemm",
                    "rows": self.plan.dot_axes[0][1],
                    "columns": 1,
                    "reduction": self.plan.reduction_axis[1],
                    "batch_columns": 1,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "reduction_tiles": reduction_tiles,
                    "weight_uploads": reduction_tiles,
                    "output_readbacks": 1,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                    "host_wall_estimate": _gemm_wall_estimate(
                        self.plan.dot_axes[0][1],
                        self.plan.reduction_axis[1],
                        1,
                        1,
                    ),
                },
            }
        )

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        matrix = self._array(values, self.matrix)
        vector = self._array(values, self.vector)
        output = self._array(values, self.plan.output, writable=True)
        if self.matrix_transposed:
            matrix = np.ascontiguousarray(matrix.T)

        if self.backend == "virtual":
            result = RunResult(
                int(self.estimate_result.cycles),
                "virtual APUg2 streaming uint16 GEMV",
                "virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "schedule": dict(
                        self.execution_graph.metadata["transport_schedule"]
                    ),
                },
            )
        else:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            result = run_apu_g2_u16_gemm(
                matrix,
                np.ascontiguousarray(vector.reshape(-1, 1)),
                np.ascontiguousarray(output.reshape(-1, 1)),
                alpha=self.epilogue[0],
                beta=self.epilogue[1],
                batch_columns=1,
            )
            observed = np.asarray(result.extra["outputs"]["out"][:, 0])
            np.copyto(output, observed)
            result.extra["outputs"] = {
                "out": output.copy(),
                self.plan.output.value: output.copy(),
            }
            result.extra["contraction_topology"] = "single"
            result.extra["plan"] = self.plan.manifest()
        self.last_result = result
        return result

    run = __call__


class APUG2ContractionChainCallable:
    """Execute linked or DAG rank-N uint16 contractions as dot-tile stages."""

    def __init__(self, workload, target, schedule, *, cost, backend=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 contraction chains require cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        plans = plan_apu_g2_rank_n_contractions(self.module)
        self.plans = _topological_rank_n_plans(plans)
        self.module_manifest = self.plans[0].module
        if self.module_manifest.contraction_topology not in {
            "linked_chain",
            "dag",
            "independent",
        }:
            raise UnsupportedAPUG2ContractionError(
                "APUg2 contraction chain requires linked, DAG, or independent "
                "contractions"
            )
        self.stage_epilogues = tuple(plan.epilogue for plan in self.plans)
        self.transport_aware = all(
            _is_canonical_u16_gemm_plan(plan) or _streaming_gemv_roles(plan) is not None
            for plan in self.plans
        )
        fused_roles = tuple(_streaming_gemv_roles(plan) for plan in self.plans)
        self.fused_independent_gemv = (
            self.module_manifest.contraction_topology == "independent"
            and len(self.plans) > 1
            and all(role is not None for role in fused_roles)
            and len({plan.reduction_axis[1] for plan in self.plans}) == 1
            and len(set(self.stage_epilogues)) == 1
            and sum(plan.dot_axes[0][1] for plan in self.plans) <= 65_536
        )
        for plan in self.plans:
            dot_axis_names = tuple(name for name, _extent in plan.dot_axes)
            if plan.batch_local_accumulator:
                raise UnsupportedAPUG2ContractionError(
                    "APUg2 chain execution currently requires each accumulator "
                    "to index every flattened output axis"
                )
            if plan.initial_output.result_axes != dot_axis_names:
                raise UnsupportedAPUG2ContractionError(
                    "APUg2 chain execution requires initial accumulators in "
                    "dot-axis order"
                )
            if plan.output.result_axes != dot_axis_names:
                raise UnsupportedAPUG2ContractionError(
                    "APUg2 chain execution requires outputs in dot-axis order"
                )

        self.stage_recipes = []
        self.stage_calibrations = []
        extrapolated = False
        for index, (plan, (alpha, beta)) in enumerate(
            zip(self.plans, self.stage_epilogues)
        ):
            output_shape = tuple(extent for _name, extent in plan.dot_axes)
            recipe, calibration, stage_extrapolated = _build_tiled_contraction_recipe(
                output_shape,
                plan.tiling,
                batch_rank=len(plan.batch_axes),
                alpha=alpha,
                beta=beta,
                name=f"apu_g2_u16_chain_stage_{index}_{plan.analysis.function}",
            )
            self.stage_recipes.append(recipe)
            self.stage_calibrations.append(calibration)
            extrapolated = extrapolated or stage_extrapolated
        self.stage_recipes = tuple(self.stage_recipes)
        self.stage_calibrations = tuple(self.stage_calibrations)
        self.recipe = chain_apu_g2_recipes(
            self.stage_recipes,
            name="apu_g2_u16_contraction_chain",
            stage_names=tuple(plan.analysis.function for plan in self.plans),
        )
        measured = [
            calibration.measured_ticks_per_pipeline
            for calibration in self.stage_calibrations
            if calibration.measured_ticks_per_pipeline is not None
        ]
        self.calibration = APUG2RecipeCalibration.normalized(
            self.recipe,
            sum(calibration.total_cycles for calibration in self.stage_calibrations),
            measured_ticks_per_pipeline=sum(measured) if measured else None,
            repetitions=8,
            basis=(
                "sum_of_stage_tile_calibrations_with_interpolation"
                if extrapolated
                else "sum_of_stage_tile_calibrations"
            ),
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "contraction_chain_u16",
                "work_grid": (16,),
                "hardware_tasks": sum(plan.tiling.task_count for plan in self.plans),
                "contraction_topology": self.module_manifest.contraction_topology,
                "calibration_extrapolated": extrapolated,
                "stage_calibrations": [
                    calibration.manifest() for calibration in self.stage_calibrations
                ],
                "stages": [
                    {
                        "name": plan.analysis.function,
                        "region_id": plan.region_id,
                        "output": plan.output.value,
                        "flat_output_extent": plan.flat_output_extent,
                        "tile_count": plan.tiling.tile_count,
                        "reduction_tile_count": plan.tiling.reduction_tile_count,
                        "hardware_tasks": plan.tiling.task_count,
                        "reduction_extent": plan.reduction_axis[1],
                        "epilogue": {"alpha": alpha, "beta": beta},
                    }
                    for plan, (alpha, beta) in zip(self.plans, self.stage_epilogues)
                ],
                "regions": [
                    region.manifest() for region in self.module_manifest.regions
                ],
            }
        )
        if self.transport_aware:
            from .apu_g2_u16_gemm_runtime import (
                APUG2_U16_GEMM_BMAX,
                APUG2_U16_GEMM_CHUNK,
            )

            stage_schedules = []
            for plan in self.plans:
                is_gemm = _is_canonical_u16_gemm_plan(plan)
                matrix_spec = _matrixized_gemm_spec(plan) if is_gemm else None
                rows = matrix_spec["rows"] if is_gemm else plan.dot_axes[0][1]
                columns = matrix_spec["columns"] if is_gemm else 1
                reduction = plan.reduction_axis[1]
                column_batches = (
                    columns + APUG2_U16_GEMM_BMAX - 1
                ) // APUG2_U16_GEMM_BMAX
                reduction_tiles = (
                    reduction + APUG2_U16_GEMM_CHUNK - 1
                ) // APUG2_U16_GEMM_CHUNK
                stage_schedules.append(
                    {
                        "name": plan.analysis.function,
                        "kind": (
                            "column_batched_u16_gemm"
                            if is_gemm
                            else "streaming_u16_gemv"
                        ),
                        "rows": rows,
                        "columns": columns,
                        "reduction": reduction,
                        "batch_columns": APUG2_U16_GEMM_BMAX,
                        "reduction_tile": APUG2_U16_GEMM_CHUNK,
                        "hardware_tasks": column_batches * reduction_tiles,
                        "output_readbacks": column_batches,
                        "host_wall_estimate": _gemm_wall_estimate(
                            rows,
                            reduction,
                            columns,
                            APUG2_U16_GEMM_BMAX,
                        ),
                    }
                )
            self.execution_graph.metadata.update(
                {
                    "program": "transport_aware_contraction_chain_u16",
                    "hardware_tasks": sum(
                        item["hardware_tasks"] for item in stage_schedules
                    ),
                    "transport_schedule": {
                        "kind": "transport_aware_u16_contraction_chain",
                        "resident_accumulator": True,
                        "contiguous_readback": True,
                        "stages": stage_schedules,
                    },
                }
            )
            if self.fused_independent_gemv:
                rows = sum(plan.dot_axes[0][1] for plan in self.plans)
                columns = len(self.plans)
                reduction = self.plans[0].reduction_axis[1]
                reduction_tiles = math.ceil(reduction / APUG2_U16_GEMM_CHUNK)
                self.execution_graph.metadata.update(
                    {
                        "program": "fused_independent_gemv_u16",
                        "hardware_tasks": reduction_tiles,
                        "transport_schedule": {
                            "kind": "fused_independent_gemv",
                            "rows": rows,
                            "columns": columns,
                            "reduction": reduction,
                            "alpha": self.stage_epilogues[0][0],
                            "beta": self.stage_epilogues[0][1],
                            "batch_columns": APUG2_U16_GEMM_BMAX,
                            "reduction_tile": APUG2_U16_GEMM_CHUNK,
                            "hardware_tasks": reduction_tiles,
                            "weight_uploads": reduction_tiles,
                            "output_readbacks": 1,
                            "resident_accumulator": True,
                            "contiguous_readback": True,
                            "host_wall_estimate": _gemm_wall_estimate(
                                rows,
                                reduction,
                                columns,
                                APUG2_U16_GEMM_BMAX,
                            ),
                        },
                    }
                )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_contraction_chain")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(values, access, *, writable=False, allocate=False):
        value = values.get(access.value)
        if value is None:
            if not allocate:
                raise TypeError(
                    f"ordinary contraction chain is missing {access.value!r}"
                )
            value = np.zeros(access.shape, dtype=np.uint16)
            values[access.value] = value
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError(
                f"APUg2 operand {access.value!r} must be a NumPy uint16 array"
            )
        if value.shape != access.shape:
            raise ValueError(
                f"APUg2 operand {access.value!r} must have shape {access.shape}, "
                f"got {value.shape}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"APUg2 operand {access.value!r} must be C-contiguous")
        if writable and not value.flags.writeable:
            raise ValueError(f"APUg2 output {access.value!r} must be writable")
        return value

    def _ensure_stage_values(self, values, plan):
        self._array(values, plan.lhs)
        self._array(values, plan.rhs)
        self._array(values, plan.output, writable=True, allocate=True)
        self._array(values, plan.initial_output, allocate=True)

    def _pack_tile(self, values, plan, tile, reduction_tile, accumulator=None):
        lhs = self._array(values, plan.lhs)
        rhs = self._array(values, plan.rhs)
        initial = self._array(values, plan.initial_output)
        begin, end = plan.tiling.tile_bounds(tile)
        reduction_begin, reduction_end = plan.tiling.reduction_bounds(reduction_tile)
        extent = end - begin
        reduction_extent = reduction_end - reduction_begin
        left = np.empty((extent, reduction_extent), dtype=np.uint16)
        right = np.empty_like(left)
        if accumulator is None:
            accumulator = np.empty(extent, dtype=np.uint16)
        else:
            accumulator = np.ascontiguousarray(accumulator, dtype=np.uint16)
        output_sites = []
        for local, flat in enumerate(range(begin, end)):
            dot_coordinates = plan.unflatten_output(flat)
            initial_domain = dot_coordinates + (0,)
            if reduction_tile == 0:
                accumulator[local] = initial[plan.initial_output.apply(initial_domain)]
            output_sites.append(plan.output.apply(initial_domain))
            for local_reduction, reduction in enumerate(
                range(reduction_begin, reduction_end)
            ):
                domain = dot_coordinates + (reduction,)
                left[local, local_reduction] = lhs[plan.lhs.apply(domain)]
                right[local, local_reduction] = rhs[plan.rhs.apply(domain)]
        return left, right, accumulator, tuple(output_sites)

    def _output_payload(self, values):
        outputs = {}
        for plan in self.plans:
            outputs[plan.output.value] = np.asarray(values[plan.output.value]).copy()
        sinks = set(self.module_manifest.sinks)
        sink_plans = [plan for plan in self.plans if plan.region_id in sinks]
        if len(sink_plans) == 1:
            outputs["out"] = outputs[sink_plans[0].output.value].copy()
        return outputs

    def _run_fused_independent_gemv(self, values):
        from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

        rows = sum(plan.dot_axes[0][1] for plan in self.plans)
        columns = len(self.plans)
        reduction = self.plans[0].reduction_axis[1]
        matrix = np.zeros((rows, reduction), dtype=np.uint16)
        vectors = np.zeros((reduction, columns), dtype=np.uint16)
        accumulator = np.zeros((rows, columns), dtype=np.uint16)
        bounds = []
        begin = 0
        for column, plan in enumerate(self.plans):
            matrix_access, vector_access, transposed = _streaming_gemv_roles(plan)
            source = self._array(values, matrix_access)
            if transposed:
                source = np.ascontiguousarray(source.T)
            end = begin + plan.dot_axes[0][1]
            matrix[begin:end] = source
            vectors[:, column] = self._array(values, vector_access)
            accumulator[begin:end, column] = self._array(values, plan.initial_output)
            bounds.append((begin, end))
            begin = end

        alpha, beta = self.stage_epilogues[0]
        result = run_apu_g2_u16_gemm(
            matrix,
            vectors,
            accumulator,
            alpha=alpha,
            beta=beta,
        )
        combined = np.asarray(result.extra["outputs"]["out"])
        for column, (plan, (begin, end)) in enumerate(zip(self.plans, bounds)):
            np.copyto(
                self._array(values, plan.output, writable=True),
                combined[begin:end, column],
            )
        result.extra.update(
            {
                "outputs": self._output_payload(values),
                "contraction_topology": "independent",
                "vectorization_certificate": self.recipe.certificate.manifest(),
                "plans": [plan.manifest() for plan in self.plans],
                "stage_epilogues": [
                    {"alpha": alpha, "beta": beta}
                    for alpha, beta in self.stage_epilogues
                ],
                "schedule": dict(self.execution_graph.metadata["transport_schedule"]),
            }
        )
        return result

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        for plan in self.plans:
            self._ensure_stage_values(values, plan)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 contraction chain; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "contraction_topology": self.module_manifest.contraction_topology,
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "plans": [plan.manifest() for plan in self.plans],
                    "stage_epilogues": [
                        {"alpha": alpha, "beta": beta}
                        for alpha, beta in self.stage_epilogues
                    ],
                },
            )
        elif self.fused_independent_gemv:
            result = self._run_fused_independent_gemv(values)
        elif self.transport_aware:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            stage_results = []
            stage_cycles = []
            for plan, (alpha, beta) in zip(self.plans, self.stage_epilogues):
                output = self._array(values, plan.output, writable=True)
                if _is_canonical_u16_gemm_plan(plan):
                    _spec, lhs, rhs = _matrixized_gemm_operands(plan, values)
                    initial = self._array(values, plan.initial_output)
                    stage_result = run_apu_g2_u16_gemm(
                        lhs,
                        rhs,
                        np.ascontiguousarray(
                            initial.reshape(lhs.shape[0], rhs.shape[1])
                        ),
                        alpha=alpha,
                        beta=beta,
                    )
                    observed = np.asarray(stage_result.extra["outputs"]["out"]).reshape(
                        output.shape
                    )
                else:
                    matrix_access, vector_access, transposed = _streaming_gemv_roles(
                        plan
                    )
                    matrix = self._array(values, matrix_access)
                    vector = self._array(values, vector_access)
                    initial = self._array(values, plan.initial_output)
                    if transposed:
                        matrix = np.ascontiguousarray(matrix.T)
                    stage_result = run_apu_g2_u16_gemm(
                        matrix,
                        np.ascontiguousarray(vector.reshape(-1, 1)),
                        np.ascontiguousarray(initial.reshape(-1, 1)),
                        alpha=alpha,
                        beta=beta,
                        batch_columns=1,
                    )
                    observed = np.asarray(stage_result.extra["outputs"]["out"][:, 0])
                np.copyto(output, observed)
                stage_results.append(stage_result)
                stage_cycles.append(stage_result.cycles)

            host_timings = {
                name: sum(item.extra["host_timings_us"][name] for item in stage_results)
                for name in ("h2d", "host_task", "d2h", "end_to_end")
            }
            result = RunResult(
                cycles=sum(stage_cycles),
                stdout="\n".join(item.stdout for item in stage_results),
                backend="apu_v2",
                extra={
                    "outputs": self._output_payload(values),
                    "hardware_tasks": sum(
                        item.extra["hardware_tasks"] for item in stage_results
                    ),
                    "host_timings_us": host_timings,
                    "contraction_topology": self.module_manifest.contraction_topology,
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "plans": [plan.manifest() for plan in self.plans],
                    "stage_epilogues": [
                        {"alpha": alpha, "beta": beta}
                        for alpha, beta in self.stage_epilogues
                    ],
                    "stage_cycles": stage_cycles,
                    "schedule": dict(
                        self.execution_graph.metadata["transport_schedule"]
                    ),
                    "sources": stage_results[0].extra["sources"],
                    "projects": [item.extra["project"] for item in stage_results],
                },
            )
        else:
            from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile

            tile_results = []
            observed_tiles = []
            stage_cycles = []
            for plan, (alpha, beta) in zip(self.plans, self.stage_epilogues):
                output = self._array(values, plan.output, writable=True)
                stage_start = len(tile_results)
                for tile in range(plan.tiling.tile_count):
                    accumulator = None
                    output_sites = ()
                    for reduction_tile in range(plan.tiling.reduction_tile_count):
                        left, right, accumulator, output_sites = self._pack_tile(
                            values, plan, tile, reduction_tile, accumulator
                        )
                        tile_result = run_apu_g2_u16_dot_tile(
                            left,
                            right,
                            accumulator=accumulator,
                            alpha=alpha,
                            beta=beta if reduction_tile == 0 else 1,
                            repetitions=8,
                        )
                        accumulator = np.asarray(tile_result.extra["outputs"]["out"])
                        observed_tiles.append(
                            {
                                "stage": plan.analysis.function,
                                "output_tile": tile,
                                "reduction_tile": reduction_tile,
                                "output": accumulator.copy(),
                            }
                        )
                        tile_results.append(tile_result)
                    observed = accumulator
                    for value, site in zip(observed, output_sites):
                        output[site] = value
                stage_cycles.append(
                    sum(item.cycles for item in tile_results[stage_start:])
                )

            result = RunResult(
                cycles=sum(item.cycles for item in tile_results),
                stdout="\n".join(
                    f"stage_tile={index}\n{item.stdout}"
                    for index, item in enumerate(tile_results)
                ),
                backend="apu_v2",
                extra={
                    "outputs": self._output_payload(values),
                    "hardware_tasks": len(tile_results),
                    "contraction_topology": self.module_manifest.contraction_topology,
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "plans": [plan.manifest() for plan in self.plans],
                    "stage_epilogues": [
                        {"alpha": alpha, "beta": beta}
                        for alpha, beta in self.stage_epilogues
                    ],
                    "per_tile_cycles": [item.cycles for item in tile_results],
                    "stage_cycles": stage_cycles,
                    "total_ticks": sum(
                        item.extra["total_ticks"] for item in tile_results
                    ),
                    "final_pipeline_ticks": sum(
                        item.extra["final_pipeline_ticks"] for item in tile_results
                    ),
                    "repetitions": 8,
                    "tile_outputs": observed_tiles,
                    "sources": tile_results[0].extra["sources"],
                    "projects": [item.extra["project"] for item in tile_results],
                },
            )
        self.last_result = result
        return result

    run = __call__


class APUG2GemverCallable:
    """Execute PolyBench Gemver as a four-stage VL64 dot-tile chain."""

    def __init__(self, workload, target, schedule, *, cost, backend=None, plan=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 Gemver requires a cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan or analyze_apu_g2_structured_plan(self.module)
        if not isinstance(self.plan, RankTwoUpdateGemvChainPlan):
            raise TypeError("APUg2 Gemver requires a rank-two update/GEMV plan")
        self.N = self.plan.extent
        rank_alpha = self.plan.rank_update_coefficients[0]
        self.stage_epilogues = (
            (rank_alpha, self.plan.accumulator_coefficients[0]),
            (1, self.plan.accumulator_coefficients[1]),
            (1, self.plan.accumulator_coefficients[2]),
            (1, self.plan.accumulator_coefficients[3]),
        )
        self.stage_tilings = (
            APUG2DotTiling(self.N * self.N, 2),
            APUG2DotTiling(self.N, self.N),
            APUG2DotTiling(self.N, 1),
            APUG2DotTiling(self.N, self.N),
        )
        stage_specs = (
            ("gemver_update_A", self.N * self.N),
            ("gemver_x_from_y", self.N),
            ("gemver_add_z", self.N),
            ("gemver_w_from_x", self.N),
        )
        built_stages = tuple(
            _build_tiled_contraction_recipe(
                (output_extent,),
                tiling,
                alpha=1,
                beta=1,
                name=stage_name,
            )
            for (stage_name, output_extent), tiling in zip(
                stage_specs, self.stage_tilings
            )
        )
        self.stage_recipes = tuple(item[0] for item in built_stages)
        self.stage_calibrations = tuple(item[1] for item in built_stages)
        self.recipe = chain_apu_g2_recipes(
            self.stage_recipes,
            name="apu_g2_u16_gemver_chain",
            stage_names=("update_A", "x_from_y", "add_z", "w_from_x"),
        )
        measured = [
            calibration.measured_ticks_per_pipeline
            for calibration in self.stage_calibrations
            if calibration.measured_ticks_per_pipeline is not None
        ]
        extrapolated = any(item[2] for item in built_stages)
        self.calibration = APUG2RecipeCalibration.normalized(
            self.recipe,
            sum(calibration.total_cycles for calibration in self.stage_calibrations),
            measured_ticks_per_pipeline=sum(measured) if measured else None,
            repetitions=8,
            basis=(
                "sum_of_gemver_stage_tile_calibrations_with_interpolation"
                if extrapolated
                else "sum_of_gemver_stage_tile_calibrations"
            ),
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "gemver_u16",
                "work_grid": (16,),
                "hardware_tasks": sum(
                    tiling.task_count for tiling in self.stage_tilings
                ),
                "contraction_topology": "gemver_affine_chain",
                "calibration_extrapolated": extrapolated,
                "stage_calibrations": [
                    calibration.manifest() for calibration in self.stage_calibrations
                ],
                "stages": [
                    {
                        "name": name,
                        "flat_output_extent": output_extent,
                        "reduction_extent": reduction_extent,
                        "tile_count": tiling.tile_count,
                        "reduction_tile_count": tiling.reduction_tile_count,
                        "hardware_tasks": tiling.task_count,
                        "epilogue": {"alpha": alpha, "beta": beta},
                    }
                    for name, output_extent, reduction_extent, (
                        alpha,
                        beta,
                    ), tiling in zip(
                        ("update_A", "x_from_y", "add_z", "w_from_x"),
                        (self.N * self.N, self.N, self.N, self.N),
                        (2, self.N, 1, self.N),
                        self.stage_epilogues,
                        self.stage_tilings,
                    )
                ],
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_gemver")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def _validate_operands(self, values):
        n = self.N
        arrays = {
            "A": _u16_array_argument(
                self.signature,
                values,
                self.plan.matrix_argument,
                "matrix",
                (n, n),
                writable=True,
            ),
            "u1": _u16_array_argument(
                self.signature,
                values,
                self.plan.first_row_factor_argument,
                "first row factor",
                (n,),
            ),
            "u2": _u16_array_argument(
                self.signature,
                values,
                self.plan.second_row_factor_argument,
                "second row factor",
                (n,),
            ),
            "v1": _u16_array_argument(
                self.signature,
                values,
                self.plan.first_column_factor_argument,
                "first column factor",
                (n,),
            ),
            "v2": _u16_array_argument(
                self.signature,
                values,
                self.plan.second_column_factor_argument,
                "second column factor",
                (n,),
            ),
            "x": _u16_array_argument(
                self.signature,
                values,
                self.plan.state_argument,
                "state",
                (n,),
                writable=True,
            ),
            "y": _u16_array_argument(
                self.signature,
                values,
                self.plan.projection_argument,
                "projection",
                (n,),
            ),
            "w": _u16_array_argument(
                self.signature,
                values,
                self.plan.output_argument,
                "output",
                (n,),
                writable=True,
            ),
            "z": _u16_array_argument(
                self.signature,
                values,
                self.plan.bias_argument,
                "bias",
                (n,),
            ),
        }
        return arrays

    def _pack_outer_update(self, arrays):
        n = self.N
        rows = np.repeat(np.arange(n), n)
        columns = np.tile(np.arange(n), n)
        left = np.empty((n * n, 2), dtype=np.uint16)
        right = np.empty_like(left)
        left[:, 0] = arrays["u1"][rows]
        left[:, 1] = arrays["u2"][rows]
        right[:, 0] = arrays["v1"][columns]
        right[:, 1] = arrays["v2"][columns]
        accumulator = arrays["A"].reshape(-1)
        return left, right, accumulator

    def _pack_transposed_gemv(self, arrays):
        left = np.ascontiguousarray(arrays["A"].T)
        right = np.broadcast_to(arrays["y"][None, :], left.shape).copy()
        accumulator = arrays["x"]
        return left, right, accumulator

    def _pack_vector_add(self, arrays):
        left = arrays["z"].reshape(self.N, 1)
        right = np.ones_like(left)
        accumulator = arrays["x"]
        return left, right, accumulator

    def _pack_forward_gemv(self, arrays):
        left = arrays["A"]
        right = np.broadcast_to(arrays["x"][None, :], left.shape).copy()
        accumulator = arrays["w"]
        return left, right, accumulator

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = self._validate_operands(bound.arguments)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 Gemver chain; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": sum(
                        tiling.task_count for tiling in self.stage_tilings
                    ),
                    "contraction_topology": "gemver_affine_chain",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "stage_epilogues": [
                        {"alpha": alpha, "beta": beta}
                        for alpha, beta in self.stage_epilogues
                    ],
                },
            )
        else:
            from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile

            tile_results = []
            observed_tiles = []
            stage_outputs = []
            for stage_name, packer, destination, tiling in (
                (
                    "update_A",
                    self._pack_outer_update,
                    arrays["A"],
                    self.stage_tilings[0],
                ),
                (
                    "x_from_y",
                    self._pack_transposed_gemv,
                    arrays["x"],
                    self.stage_tilings[1],
                ),
                ("add_z", self._pack_vector_add, arrays["x"], self.stage_tilings[2]),
                (
                    "w_from_x",
                    self._pack_forward_gemv,
                    arrays["w"],
                    self.stage_tilings[3],
                ),
            ):
                left, right, initial = packer(arrays)
                flat_destination = destination.reshape(-1)
                for output_tile in range(tiling.tile_count):
                    output_begin, output_end = tiling.tile_bounds(output_tile)
                    accumulator = np.ascontiguousarray(
                        initial.reshape(-1)[output_begin:output_end]
                    )
                    for reduction_tile in range(tiling.reduction_tile_count):
                        reduction_begin, reduction_end = tiling.reduction_bounds(
                            reduction_tile
                        )
                        tile_result = run_apu_g2_u16_dot_tile(
                            np.ascontiguousarray(
                                left[
                                    output_begin:output_end,
                                    reduction_begin:reduction_end,
                                ]
                            ),
                            np.ascontiguousarray(
                                right[
                                    output_begin:output_end,
                                    reduction_begin:reduction_end,
                                ]
                            ),
                            accumulator=accumulator,
                            alpha=1,
                            beta=1,
                            repetitions=8,
                        )
                        accumulator = np.asarray(tile_result.extra["outputs"]["out"])
                        observed_tiles.append(
                            {
                                "stage": stage_name,
                                "output_tile": output_tile,
                                "reduction_tile": reduction_tile,
                                "output": accumulator.copy(),
                            }
                        )
                        tile_results.append(tile_result)
                    flat_destination[output_begin:output_end] = accumulator
                stage_outputs.append((stage_name, destination.copy()))

            result = RunResult(
                cycles=sum(item.cycles for item in tile_results),
                stdout="\n".join(
                    f"task={index}\n{item.stdout}"
                    for index, item in enumerate(tile_results)
                ),
                backend="apu_v2",
                extra={
                    "outputs": {
                        "A": arrays["A"].copy(),
                        "x": arrays["x"].copy(),
                        "w": arrays["w"].copy(),
                        "out": arrays["w"].copy(),
                    },
                    "hardware_tasks": len(tile_results),
                    "contraction_topology": "gemver_affine_chain",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "stage_epilogues": [
                        {"alpha": alpha, "beta": beta}
                        for alpha, beta in self.stage_epilogues
                    ],
                    "stage_outputs": {name: output for name, output in stage_outputs},
                    "per_tile_cycles": [item.cycles for item in tile_results],
                    "stage_cycles": [
                        sum(
                            item.cycles
                            for item in tile_results[
                                sum(
                                    t.task_count for t in self.stage_tilings[:index]
                                ) : sum(
                                    t.task_count
                                    for t in self.stage_tilings[: index + 1]
                                )
                            ]
                        )
                        for index in range(len(self.stage_tilings))
                    ],
                    "total_ticks": sum(
                        item.extra["total_ticks"] for item in tile_results
                    ),
                    "final_pipeline_ticks": sum(
                        item.extra["final_pipeline_ticks"] for item in tile_results
                    ),
                    "repetitions": 8,
                    "tile_outputs": observed_tiles,
                    "sources": tile_results[0].extra["sources"],
                    "projects": [item.extra["project"] for item in tile_results],
                },
            )
        self.last_result = result
        return result

    run = __call__


class APUG2SymmCallable:
    """Execute PolyBench SYMM through symmetric triangular VL64 dot tiles."""

    def __init__(self, workload, target, schedule, *, cost, backend=None, plan=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 SYMM requires a cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan or analyze_apu_g2_structured_plan(self.module)
        if not isinstance(self.plan, SymmetricContractionPlan):
            raise TypeError("APUg2 SYMM requires a symmetric contraction plan")
        self.M = self.plan.rows
        self.N = self.plan.columns
        self.alpha = _u16_epilogue_scalar(
            self.plan.alpha_coefficient, "symmetric contraction alpha"
        )
        self.beta = _u16_epilogue_scalar(
            self.plan.beta_coefficient, "symmetric contraction beta"
        )
        self.output_extent = self.M * self.N
        self.reduction_extent = self.M
        self.tiling = APUG2DotTiling(self.output_extent, self.reduction_extent)
        self.recipe, self.calibration, extrapolated = _build_tiled_contraction_recipe(
            (self.M, self.N),
            self.tiling,
            alpha=self.alpha,
            beta=self.beta,
            name="apu_g2_u16_symm_lower_symmetric",
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "symm_u16",
                "work_grid": (16,),
                "hardware_tasks": self.tiling.task_count,
                "contraction_topology": "symm_lower_symmetric",
                "calibration_extrapolated": extrapolated,
                "calibration_basis": self.calibration.basis,
                "runtime_epilogue": {"alpha": self.alpha, "beta": self.beta},
                "triangular_access": {
                    "storage": "lower_with_diagonal",
                    "logical_matrix": "A[row,k] if k<=row else A[k,row]",
                },
                "stages": [
                    {
                        "name": "symm",
                        "flat_output_extent": self.output_extent,
                        "reduction_extent": self.reduction_extent,
                        "tile_count": self.tiling.tile_count,
                        "reduction_tile_count": self.tiling.reduction_tile_count,
                        "hardware_tasks": self.tiling.task_count,
                        "epilogue": {"alpha": self.alpha, "beta": self.beta},
                    }
                ],
            }
        )
        from .apu_g2_u16_gemm_runtime import (
            APUG2_U16_GEMM_BMAX,
            APUG2_U16_GEMM_CHUNK,
        )

        column_batches = (self.N + APUG2_U16_GEMM_BMAX - 1) // APUG2_U16_GEMM_BMAX
        reduction_tiles = (self.M + APUG2_U16_GEMM_CHUNK - 1) // APUG2_U16_GEMM_CHUNK
        self.execution_graph.metadata.update(
            {
                "program": "column_batched_symm_u16",
                "hardware_tasks": column_batches * reduction_tiles,
                "transport_schedule": {
                    "kind": "column_batched_u16_gemm",
                    "rows": self.M,
                    "columns": self.N,
                    "reduction": self.M,
                    "batch_columns": APUG2_U16_GEMM_BMAX,
                    "column_batches": column_batches,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "reduction_tiles": reduction_tiles,
                    "weight_uploads": column_batches * reduction_tiles,
                    "output_readbacks": column_batches,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                    "host_wall_estimate": _gemm_wall_estimate(
                        self.M, self.M, self.N, APUG2_U16_GEMM_BMAX
                    ),
                },
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_symm")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def _validate_operands(self, values):
        return {
            "A": _u16_array_argument(
                self.signature,
                values,
                self.plan.symmetric_matrix_argument,
                "symmetric matrix",
                (self.M, self.M),
            ),
            "B": _u16_array_argument(
                self.signature,
                values,
                self.plan.rhs_argument,
                "right-hand side",
                (self.M, self.N),
            ),
            "C": _u16_array_argument(
                self.signature,
                values,
                self.plan.output_argument,
                "output",
                (self.M, self.N),
                writable=True,
            ),
        }

    def _pack_tile(self, arrays, tile, reduction_tile=0, accumulator=None):
        begin, end = self.tiling.tile_bounds(tile)
        reduction_begin, reduction_end = self.tiling.reduction_bounds(reduction_tile)
        extent = end - begin
        reduction_extent = reduction_end - reduction_begin
        left = np.empty((extent, reduction_extent), dtype=np.uint16)
        right = np.empty_like(left)
        depth = np.arange(reduction_begin, reduction_end)
        for local, flat in enumerate(range(begin, end)):
            row = flat // self.N
            column = flat % self.N
            left[local] = np.where(
                depth <= row, arrays["A"][row, depth], arrays["A"][depth, row]
            )
            right[local] = arrays["B"][depth, column]
        if accumulator is None:
            accumulator = arrays["C"].reshape(-1)[begin:end]
        return left, right, accumulator, begin, end

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = self._validate_operands(bound.arguments)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 SYMM; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "contraction_topology": "symm_lower_symmetric",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "epilogue": {"alpha": self.alpha, "beta": self.beta},
                },
            )
        else:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            lower = np.tril(arrays["A"])
            logical_a = np.ascontiguousarray(
                lower + np.tril(arrays["A"], -1).T,
                dtype=np.uint16,
            )
            result = run_apu_g2_u16_gemm(
                logical_a,
                arrays["B"],
                arrays["C"],
                alpha=self.alpha,
                beta=self.beta,
            )
            observed = np.asarray(result.extra["outputs"]["out"])
            np.copyto(arrays["C"], observed)
            result.extra["outputs"] = {
                "C": arrays["C"].copy(),
                "out": arrays["C"].copy(),
            }
            result.extra["contraction_topology"] = "symm_lower_symmetric"
            result.extra["vectorization_certificate"] = (
                self.recipe.certificate.manifest()
            )
            result.extra["epilogue"] = {
                "alpha": self.alpha,
                "beta": self.beta,
            }
        self.last_result = result
        return result

    run = __call__


class APUG2TrmmCallable:
    """Execute PolyBench TRMM through upper-triangular VL64 dot tiles."""

    def __init__(self, workload, target, schedule, *, cost, backend=None, plan=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 TRMM requires a cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan or analyze_apu_g2_structured_plan(self.module)
        if not isinstance(self.plan, UnitDiagonalTriangularContractionPlan):
            raise TypeError(
                "APUg2 TRMM requires a unit-diagonal triangular contraction plan"
            )
        self.M = self.plan.rows
        self.N = self.plan.columns
        self.alpha = _u16_epilogue_scalar(
            self.plan.post_scale_coefficient, "triangular post-scale"
        )
        self.output_extent = self.M * self.N
        self.reduction_extent = self.M
        self.tiling = APUG2DotTiling(self.output_extent, self.reduction_extent)
        self.recipe, self.calibration, extrapolated = _build_tiled_contraction_recipe(
            (self.M, self.N),
            self.tiling,
            alpha=self.alpha,
            beta=0,
            name="apu_g2_u16_trmm_upper_triangular",
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "trmm_u16",
                "work_grid": (16,),
                "hardware_tasks": self.tiling.task_count,
                "contraction_topology": "trmm_upper_triangular",
                "calibration_extrapolated": extrapolated,
                "calibration_basis": self.calibration.basis,
                "runtime_epilogue": {"alpha": self.alpha, "beta": 0},
                "triangular_access": {
                    "storage": "strict_lower_column_with_unit_diagonal",
                    "logical_matrix": "1 if k==row else A[k,row] if k>row else 0",
                },
                "stages": [
                    {
                        "name": "trmm",
                        "flat_output_extent": self.output_extent,
                        "reduction_extent": self.reduction_extent,
                        "tile_count": self.tiling.tile_count,
                        "reduction_tile_count": self.tiling.reduction_tile_count,
                        "hardware_tasks": self.tiling.task_count,
                        "epilogue": {"alpha": self.alpha, "beta": 0},
                    }
                ],
            }
        )
        from .apu_g2_u16_gemm_runtime import (
            APUG2_U16_GEMM_BMAX,
            APUG2_U16_GEMM_CHUNK,
        )

        column_batches = (self.N + APUG2_U16_GEMM_BMAX - 1) // APUG2_U16_GEMM_BMAX
        reduction_tiles = (self.M + APUG2_U16_GEMM_CHUNK - 1) // APUG2_U16_GEMM_CHUNK
        self.execution_graph.metadata.update(
            {
                "program": "column_batched_trmm_u16",
                "hardware_tasks": column_batches * reduction_tiles,
                "transport_schedule": {
                    "kind": "column_batched_u16_gemm",
                    "rows": self.M,
                    "columns": self.N,
                    "reduction": self.M,
                    "batch_columns": APUG2_U16_GEMM_BMAX,
                    "column_batches": column_batches,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "reduction_tiles": reduction_tiles,
                    "weight_uploads": column_batches * reduction_tiles,
                    "output_readbacks": column_batches,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                    "host_wall_estimate": _gemm_wall_estimate(
                        self.M, self.M, self.N, APUG2_U16_GEMM_BMAX
                    ),
                },
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_trmm")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def _validate_operands(self, values):
        return {
            "A": _u16_array_argument(
                self.signature,
                values,
                self.plan.triangular_matrix_argument,
                "triangular matrix",
                (self.M, self.M),
            ),
            "B": _u16_array_argument(
                self.signature,
                values,
                self.plan.state_argument,
                "state",
                (self.M, self.N),
                writable=True,
            ),
        }

    def _pack_tile(self, arrays, original_b, tile, reduction_tile=0, accumulator=None):
        begin, end = self.tiling.tile_bounds(tile)
        reduction_begin, reduction_end = self.tiling.reduction_bounds(reduction_tile)
        extent = end - begin
        reduction_extent = reduction_end - reduction_begin
        left = np.zeros((extent, reduction_extent), dtype=np.uint16)
        right = np.empty_like(left)
        depth = np.arange(reduction_begin, reduction_end)
        for local, flat in enumerate(range(begin, end)):
            row = flat // self.N
            column = flat % self.N
            left[local, depth == row] = np.uint16(1)
            greater = depth > row
            left[local, greater] = arrays["A"][depth[greater], row]
            right[local] = original_b[depth, column]
        if accumulator is None:
            accumulator = np.zeros(extent, dtype=np.uint16)
        return left, right, accumulator, begin, end

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = self._validate_operands(bound.arguments)
        original_b = arrays["B"].copy()

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 TRMM; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "contraction_topology": "trmm_upper_triangular",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "epilogue": {"alpha": self.alpha, "beta": 0},
                },
            )
        else:
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            logical_a = np.ascontiguousarray(
                np.triu(arrays["A"].T, 1) + np.eye(self.M, dtype=np.uint16)
            )
            zeros = np.zeros_like(original_b)
            result = run_apu_g2_u16_gemm(
                logical_a,
                original_b,
                zeros,
                alpha=self.alpha,
                beta=0,
            )
            observed = np.asarray(result.extra["outputs"]["out"])
            np.copyto(arrays["B"], observed)
            result.extra["outputs"] = {
                "B": arrays["B"].copy(),
                "out": arrays["B"].copy(),
            }
            result.extra["contraction_topology"] = "trmm_upper_triangular"
            result.extra["vectorization_certificate"] = (
                self.recipe.certificate.manifest()
            )
            result.extra["epilogue"] = {"alpha": self.alpha, "beta": 0}
        self.last_result = result
        return result

    run = __call__


class APUG2CovarianceCallable:
    """Execute PolyBench covariance through compiled APUg2 primitive dispatch."""

    def __init__(self, workload, target, schedule, *, cost, backend=None, plan=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 covariance requires a cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan or analyze_apu_g2_structured_plan(self.module)
        if not isinstance(self.plan, CenteredGramStatisticsPlan):
            raise TypeError("APUg2 covariance requires a centered Gram statistics plan")
        self.M = self.plan.feature_extent
        self.N = self.plan.sample_extent
        self.mean_divisor = self.plan.mean_divisor
        self.gram_divisor = self.plan.gram_divisor
        if self.N <= 1:
            raise ValueError("APUg2 covariance requires N > 1")

        self.mean_tiling = APUG2DotTiling(self.M, self.N)
        self.gram_tiling = APUG2DotTiling(self.M * self.M, self.N)
        from .apu_g2_u16_gemm_runtime import (
            APUG2_U16_GEMM_BMAX,
            APUG2_U16_GEMM_CHUNK,
        )

        self.gram_column_batches = math.ceil(self.M / APUG2_U16_GEMM_BMAX)
        self.gram_reduction_tiles = math.ceil(self.N / APUG2_U16_GEMM_CHUNK)
        self.mean_hardware_tasks = math.ceil(self.N / APUG2_U16_GEMM_CHUNK)
        self.gram_hardware_tasks = self.gram_column_batches * self.gram_reduction_tiles
        mean_recipe, _mean_calibration, _ = _build_tiled_contraction_recipe(
            (self.M,),
            self.mean_tiling,
            alpha=None,
            beta=None,
            name="covariance_mean_sum",
        )
        gram_recipe, _gram_calibration, _ = _build_tiled_contraction_recipe(
            (self.M, self.M),
            self.gram_tiling,
            alpha=None,
            beta=None,
            name="covariance_centered_gram",
        )
        self.center_pack_count = _apu_g2_pack_count(self.M * self.N)
        self.covariance_pack_count = _apu_g2_pack_count(self.M * self.M)
        self.stage_recipes = (
            mean_recipe,
            build_apu_g2_u16_fill_recipe(self.mean_divisor, name="covariance_fill_n"),
            build_apu_g2_u16_div_recipe(name="covariance_mean_div"),
            build_apu_g2_u16_sub_recipe(name="covariance_center"),
            gram_recipe,
            build_apu_g2_u16_fill_recipe(
                self.gram_divisor, name="covariance_fill_n_minus_one"
            ),
            build_apu_g2_u16_div_recipe(name="covariance_cov_div"),
        )
        self.stage_names = (
            "mean_sum",
            "fill_n",
            "mean_div",
            "center",
            "gram",
            "fill_scale",
            "cov_div",
        )
        self.recipe = chain_apu_g2_recipes(
            self.stage_recipes,
            name="apu_g2_u16_covariance_chain",
            stage_names=self.stage_names,
        )
        self.hardware_tasks = (
            self.mean_hardware_tasks
            + 1
            + 1
            + self.center_pack_count
            + self.gram_hardware_tasks
            + 1
            + self.covariance_pack_count
        )
        self.execution_graph = build_apu_g2_recipe_graph(self.recipe, target)
        self.execution_graph.metadata.update(
            {
                "program": "transport_aware_covariance_u16",
                "work_grid": (16,),
                "hardware_tasks": self.hardware_tasks,
                "contraction_topology": "covariance_primitive_chain",
                "composite_calibration": "structural_only",
                "shape": {"M": self.M, "N": self.N},
                "primitive_programs": {
                    "dot_tile": 0,
                    "column_batched_gemm": 2,
                    "fill": 2,
                    "div": 1 + self.covariance_pack_count,
                    "sub": self.center_pack_count,
                },
                "tiling": {
                    "mean_output_extent": self.M,
                    "mean_tile_capacity": self.mean_tiling.tile_capacity,
                    "mean_tile_count": self.mean_tiling.tile_count,
                    "mean_reduction_tile_count": (
                        self.mean_tiling.reduction_tile_count
                    ),
                    "mean_hardware_tasks": self.mean_hardware_tasks,
                    "gram_output_extent": self.M * self.M,
                    "gram_batch_columns": APUG2_U16_GEMM_BMAX,
                    "gram_column_batches": self.gram_column_batches,
                    "gram_reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "gram_reduction_tile_count": self.gram_reduction_tiles,
                    "gram_hardware_tasks": self.gram_hardware_tasks,
                    "center_pack_count": self.center_pack_count,
                    "covariance_pack_count": self.covariance_pack_count,
                    "reduction_extent": self.N,
                },
                "stages": [
                    {
                        "name": stage_name,
                        "recipe": recipe.name,
                        "operation": recipe.metadata.get("operation"),
                    }
                    for stage_name, recipe in zip(self.stage_names, self.stage_recipes)
                ],
                "transport_schedule": {
                    "kind": "transport_aware_statistics",
                    "stages": [
                        {
                            "name": "mean_sum",
                            "kind": "streaming_u16_gemv",
                            "rows": self.M,
                            "columns": 1,
                            "reduction": self.N,
                            "hardware_tasks": self.mean_hardware_tasks,
                            "output_readbacks": 1,
                            "host_wall_estimate": _gemm_wall_estimate(
                                self.M, self.N, 1, 1
                            ),
                        },
                        {
                            "name": "centered_gram",
                            "kind": "column_batched_u16_gemm",
                            "rows": self.M,
                            "columns": self.M,
                            "reduction": self.N,
                            "hardware_tasks": self.gram_hardware_tasks,
                            "output_readbacks": self.gram_column_batches,
                            "host_wall_estimate": _gemm_wall_estimate(
                                self.M,
                                self.N,
                                self.M,
                                APUG2_U16_GEMM_BMAX,
                            ),
                        },
                    ],
                    "batch_columns": APUG2_U16_GEMM_BMAX,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                },
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_covariance")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def _validate_operands(self, values):
        return {
            "data": _u16_array_argument(
                self.signature,
                values,
                self.plan.data_argument,
                "data",
                (self.N, self.M),
                writable=True,
            ),
            "mean": _u16_array_argument(
                self.signature,
                values,
                self.plan.mean_argument,
                "mean",
                (self.M,),
                writable=True,
            ),
            "cov": _u16_array_argument(
                self.signature,
                values,
                self.plan.gram_argument,
                "Gram output",
                (self.M, self.M),
                writable=True,
            ),
        }

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = self._validate_operands(bound.arguments)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 covariance primitive chain; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.hardware_tasks,
                    "contraction_topology": "covariance_primitive_chain",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "primitive_programs": dict(
                        self.execution_graph.metadata["primitive_programs"]
                    ),
                    "tiling": dict(self.execution_graph.metadata["tiling"]),
                },
            )
        else:
            from .apu_g2_covariance_runtime import run_apu_g2_u16_covariance

            primitive_result = run_apu_g2_u16_covariance(arrays["data"], repetitions=2)
            primitive_outputs = primitive_result.extra["outputs"]
            np.copyto(arrays["data"], primitive_outputs["centered"])
            np.copyto(arrays["mean"], primitive_outputs["mean"])
            np.copyto(arrays["cov"], primitive_outputs["covariance"])

            extra = dict(primitive_result.extra)
            extra.update(
                {
                    "outputs": {
                        "data": arrays["data"].copy(),
                        "mean": arrays["mean"].copy(),
                        "cov": arrays["cov"].copy(),
                        "centered": arrays["data"].copy(),
                        "covariance": arrays["cov"].copy(),
                        "out": arrays["cov"].copy(),
                    },
                    "contraction_topology": "covariance_primitive_chain",
                    "compiled_workload": True,
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "execution_graph_metadata": dict(self.execution_graph.metadata),
                }
            )
            result = RunResult(
                primitive_result.cycles,
                primitive_result.stdout,
                primitive_result.backend,
                extra=extra,
            )

        self.last_result = result
        return result

    run = __call__


class APUG2CorrelationCallable:
    """Execute PolyBench correlation through compiled APUg2 primitive dispatch."""

    def __init__(self, workload, target, schedule, *, cost, backend=None, plan=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 correlation requires a cost bound to the target")
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.plan = plan or analyze_apu_g2_structured_plan(self.module)
        if not isinstance(self.plan, NormalizedGramStatisticsPlan):
            raise TypeError(
                "APUg2 correlation requires a normalized Gram statistics plan"
            )
        self.M = self.plan.feature_extent
        self.N = self.plan.sample_extent
        self.mean_divisor = self.plan.mean_divisor
        self.variance_divisor = self.plan.variance_divisor
        self.root_sample_factor = self.plan.root_sample_factor
        self.zero_replacement = self.plan.zero_replacement
        self.diagonal_value = self.plan.diagonal_value
        if self.mean_divisor != self.variance_divisor:
            raise ValueError(
                "APUg2 correlation requires a shared mean/variance divisor"
            )

        self.mean_tiling = APUG2DotTiling(self.M, self.N)
        self.variance_tiling = APUG2DotTiling(self.M, self.N)
        self.corr_tiling = APUG2DotTiling(self.M * self.M, self.N)
        from .apu_g2_u16_gemm_runtime import (
            APUG2_U16_GEMM_BMAX,
            APUG2_U16_GEMM_CHUNK,
        )

        self.corr_column_batches = math.ceil(self.M / APUG2_U16_GEMM_BMAX)
        self.corr_reduction_tiles = math.ceil(self.N / APUG2_U16_GEMM_CHUNK)
        self.mean_hardware_tasks = math.ceil(self.N / APUG2_U16_GEMM_CHUNK)
        self.corr_hardware_tasks = self.corr_column_batches * self.corr_reduction_tiles
        mean_recipe, _mean_calibration, _ = _build_tiled_contraction_recipe(
            (self.M,),
            self.mean_tiling,
            alpha=None,
            beta=None,
            name="correlation_mean_sum",
        )
        variance_recipe, _variance_calibration, _ = _build_tiled_contraction_recipe(
            (self.M,),
            self.variance_tiling,
            alpha=None,
            beta=None,
            name="correlation_variance_sum",
        )
        corr_recipe, _corr_calibration, _ = _build_tiled_contraction_recipe(
            (self.M, self.M),
            self.corr_tiling,
            alpha=None,
            beta=None,
            name="correlation_normalized_gram",
        )
        self.column_pack_count = _apu_g2_pack_count(self.M)
        self.data_pack_count = _apu_g2_pack_count(self.M * self.N)
        self.corr_pack_count = _apu_g2_pack_count(self.M * self.M)
        self.stage_recipes = (
            mean_recipe,
            build_apu_g2_u16_fill_recipe(self.mean_divisor, name="correlation_fill_n"),
            build_apu_g2_u16_div_recipe(name="correlation_mean_div"),
            build_apu_g2_u16_sub_recipe(name="correlation_std_center"),
            variance_recipe,
            build_apu_g2_u16_div_recipe(name="correlation_variance_div"),
            build_apu_g2_u16_sqrt_recipe(name="correlation_stddev_sqrt"),
            build_apu_g2_u16_fill_recipe(
                self.zero_replacement, name="correlation_fill_one"
            ),
            build_apu_g2_u16_minmax_recipe(name="correlation_stddev_clamp"),
            build_apu_g2_u16_sub_recipe(name="correlation_center"),
            build_apu_g2_u16_fill_recipe(
                self.root_sample_factor, name="correlation_fill_sqrt_n"
            ),
            build_apu_g2_u16_mul_recipe(name="correlation_denominator_mul"),
            build_apu_g2_u16_div_recipe(name="correlation_normalize_div"),
            corr_recipe,
            build_apu_g2_u16_select_lt_recipe(name="correlation_diagonal_select"),
        )
        self.stage_names = (
            "mean_sum",
            "fill_n",
            "mean_div",
            "std_center",
            "variance_sum",
            "variance_div",
            "stddev_sqrt",
            "fill_one",
            "stddev_clamp",
            "center",
            "fill_sqrt_n",
            "denominator_mul",
            "normalize_div",
            "corr_gram",
            "diagonal_select",
        )
        self.recipe = chain_apu_g2_recipes(
            self.stage_recipes,
            name="apu_g2_u16_correlation_chain",
            stage_names=self.stage_names,
        )
        self.hardware_tasks = (
            self.mean_hardware_tasks
            + 1
            + self.column_pack_count
            + self.data_pack_count
            + self.variance_tiling.task_count
            + self.column_pack_count
            + self.column_pack_count
            + 1
            + self.column_pack_count
            + self.data_pack_count
            + 1
            + self.data_pack_count
            + self.data_pack_count
            + self.corr_hardware_tasks
            + self.corr_pack_count
        )
        self.execution_graph = build_apu_g2_recipe_graph(self.recipe, target)
        self.execution_graph.metadata.update(
            {
                "program": "transport_aware_correlation_u16",
                "work_grid": (16,),
                "hardware_tasks": self.hardware_tasks,
                "contraction_topology": "correlation_primitive_chain",
                "composite_calibration": "structural_only",
                "shape": {"M": self.M, "N": self.N},
                "primitive_programs": {
                    "dot_tile": self.variance_tiling.task_count,
                    "column_batched_gemm": 2,
                    "fill": 3,
                    "div": 2 * self.column_pack_count + self.data_pack_count,
                    "sub": 2 * self.data_pack_count,
                    "sqrt": self.column_pack_count,
                    "minmax": self.column_pack_count,
                    "mul": self.data_pack_count,
                    "select_lt": self.corr_pack_count,
                },
                "tiling": {
                    "mean_output_extent": self.M,
                    "mean_tile_capacity": self.mean_tiling.tile_capacity,
                    "mean_tile_count": self.mean_tiling.tile_count,
                    "mean_reduction_tile_count": (
                        self.mean_tiling.reduction_tile_count
                    ),
                    "mean_hardware_tasks": self.mean_hardware_tasks,
                    "variance_output_extent": self.M,
                    "variance_tile_capacity": self.variance_tiling.tile_capacity,
                    "variance_tile_count": self.variance_tiling.tile_count,
                    "variance_reduction_tile_count": (
                        self.variance_tiling.reduction_tile_count
                    ),
                    "variance_hardware_tasks": self.variance_tiling.task_count,
                    "corr_output_extent": self.M * self.M,
                    "corr_batch_columns": APUG2_U16_GEMM_BMAX,
                    "corr_column_batches": self.corr_column_batches,
                    "corr_reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "corr_reduction_tile_count": self.corr_reduction_tiles,
                    "corr_hardware_tasks": self.corr_hardware_tasks,
                    "column_pack_count": self.column_pack_count,
                    "data_pack_count": self.data_pack_count,
                    "correlation_pack_count": self.corr_pack_count,
                    "reduction_extent": self.N,
                },
                "stages": [
                    {
                        "name": stage_name,
                        "recipe": recipe.name,
                        "operation": recipe.metadata.get("operation"),
                    }
                    for stage_name, recipe in zip(self.stage_names, self.stage_recipes)
                ],
                "transport_schedule": {
                    "kind": "transport_aware_statistics",
                    "stages": [
                        {
                            "name": "mean_sum",
                            "kind": "streaming_u16_gemv",
                            "rows": self.M,
                            "columns": 1,
                            "reduction": self.N,
                            "hardware_tasks": self.mean_hardware_tasks,
                            "output_readbacks": 1,
                            "host_wall_estimate": _gemm_wall_estimate(
                                self.M, self.N, 1, 1
                            ),
                        },
                        {
                            "name": "normalized_gram",
                            "kind": "column_batched_u16_gemm",
                            "rows": self.M,
                            "columns": self.M,
                            "reduction": self.N,
                            "hardware_tasks": self.corr_hardware_tasks,
                            "output_readbacks": self.corr_column_batches,
                            "host_wall_estimate": _gemm_wall_estimate(
                                self.M,
                                self.N,
                                self.M,
                                APUG2_U16_GEMM_BMAX,
                            ),
                        },
                    ],
                    "batch_columns": APUG2_U16_GEMM_BMAX,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                },
            }
        )
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_g2_correlation")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    def estimate(self):
        return self.estimate_result

    def _validate_operands(self, values):
        return {
            "data_mean": _u16_array_argument(
                self.signature,
                values,
                self.plan.mean_input_argument,
                "mean input",
                (self.N, self.M),
            ),
            "data_stddev": _u16_array_argument(
                self.signature,
                values,
                self.plan.variance_input_argument,
                "variance input",
                (self.N, self.M),
            ),
            "data_for_center": _u16_array_argument(
                self.signature,
                values,
                self.plan.center_input_argument,
                "centering input",
                (self.N, self.M),
            ),
            "corr": _u16_array_argument(
                self.signature,
                values,
                self.plan.gram_argument,
                "Gram output",
                (self.M, self.M),
                writable=True,
            ),
        }

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = self._validate_operands(bound.arguments)

        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual APUg2 correlation primitive chain; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.hardware_tasks,
                    "contraction_topology": "correlation_primitive_chain",
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "primitive_programs": dict(
                        self.execution_graph.metadata["primitive_programs"]
                    ),
                    "tiling": dict(self.execution_graph.metadata["tiling"]),
                },
            )
        else:
            from .apu_g2_correlation_runtime import run_apu_g2_u16_correlation

            primitive_result = run_apu_g2_u16_correlation(
                arrays["data_mean"],
                arrays["data_stddev"],
                arrays["data_for_center"],
                repetitions=2,
            )
            primitive_outputs = primitive_result.extra["outputs"]
            np.copyto(arrays["corr"], primitive_outputs["correlation"])

            extra = dict(primitive_result.extra)
            extra.update(
                {
                    "outputs": {
                        "corr": arrays["corr"].copy(),
                        "correlation": arrays["corr"].copy(),
                        "out": arrays["corr"].copy(),
                        **{
                            key: value
                            for key, value in primitive_outputs.items()
                            if key != "correlation"
                        },
                    },
                    "contraction_topology": "correlation_primitive_chain",
                    "compiled_workload": True,
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                    "execution_graph_metadata": dict(self.execution_graph.metadata),
                }
            )
            result = RunResult(
                primitive_result.cycles,
                primitive_result.stdout,
                primitive_result.backend,
                extra=extra,
            )

        self.last_result = result
        return result

    run = __call__


class APUG2ChunkedGesummvCallable:
    """Execute LARGE GESUMMV with streaming GEMVs and a device combine."""

    def __init__(self, program, target, *, cost, backend=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 chunked GESUMMV requires target-bound cost")
        if not isinstance(program, APUG2Program) or program.operation != "gesummv_u16":
            raise TypeError("chunked GESUMMV requires a gesummv_u16 APUG2Program")
        self.program = program
        self.target = target
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.rows, self.reduction_extent = program.shape
        self.matrix_tiling = APUG2DotTiling(2 * self.rows, self.reduction_extent)
        self.combine_tiling = APUG2DotTiling(self.rows, 2)
        from .apu_g2_u16_gemm_runtime import APUG2_U16_GEMM_CHUNK

        self.matrix_tasks = 2 * math.ceil(self.reduction_extent / APUG2_U16_GEMM_CHUNK)
        matrix_recipe, matrix_calibration, matrix_extrapolated = (
            _build_tiled_contraction_recipe(
                (2, self.rows),
                self.matrix_tiling,
                alpha=1,
                beta=0,
                name="apu_g2_u16_gesummv_large_matrix_dots",
            )
        )
        combine_recipe, combine_calibration, combine_extrapolated = (
            _build_tiled_contraction_recipe(
                (self.rows,),
                self.combine_tiling,
                alpha=1,
                beta=0,
                name="apu_g2_u16_gesummv_large_scale_combine",
            )
        )
        self.stage_recipes = (matrix_recipe, combine_recipe)
        self.stage_calibrations = (matrix_calibration, combine_calibration)
        self.recipe = chain_apu_g2_recipes(
            self.stage_recipes,
            name="apu_g2_u16_gesummv_large",
            stage_names=("matrix_dots", "scale_combine"),
        )
        measured = [
            calibration.measured_ticks_per_pipeline
            for calibration in self.stage_calibrations
            if calibration.measured_ticks_per_pipeline is not None
        ]
        self.calibration = APUG2RecipeCalibration.normalized(
            self.recipe,
            sum(item.total_cycles for item in self.stage_calibrations),
            measured_ticks_per_pipeline=sum(measured) if measured else None,
            repetitions=program.repetitions,
            basis=(
                "large_gesummv_shard_composition_with_interpolation"
                if matrix_extrapolated or combine_extrapolated
                else "large_gesummv_shard_composition"
            ),
        )
        self.execution_graph = build_apu_g2_recipe_graph(
            self.recipe, target, calibration=self.calibration
        )
        self.execution_graph.metadata.update(
            {
                "program": "transport_aware_gesummv_u16",
                "work_grid": (16,),
                "execution": "streaming_gemv_composition",
                "hardware_tasks": (self.matrix_tasks + self.combine_tiling.task_count),
                "shape": {
                    "rows": self.rows,
                    "reduction": self.reduction_extent,
                },
                "alpha": program.alpha,
                "beta": program.beta,
                "stages": (
                    {
                        "name": "matrix_dots",
                        "kind": "streaming_u16_gemv",
                        "hardware_tasks": self.matrix_tasks,
                        "host_invocations": 2,
                        "reduction_tile_count": math.ceil(
                            self.reduction_extent / APUG2_U16_GEMM_CHUNK
                        ),
                    },
                    {
                        "name": "scale_combine",
                        "hardware_tasks": self.combine_tiling.task_count,
                        "output_tile_count": self.combine_tiling.tile_count,
                        "reduction_tile_count": (
                            self.combine_tiling.reduction_tile_count
                        ),
                    },
                ),
                "transport_schedule": {
                    "kind": "streaming_u16_gesummv",
                    "rows": self.rows,
                    "columns": 1,
                    "reduction": self.reduction_extent,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "matrix_uploads": self.matrix_tasks,
                    "output_readbacks": 2 + self.combine_tiling.tile_count,
                    "host_invocations": 2 + self.combine_tiling.tile_count,
                    "resident_accumulator": True,
                    "matrix_host_wall_estimate": {
                        **_gemm_wall_estimate(self.rows, self.reduction_extent, 1, 1),
                        "stage_count": 2,
                    },
                },
            }
        )
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.last_result = None
        self.__name__ = program.name

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(value, name, shape):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError(f"APUg2 operand {name!r} must be a NumPy uint16 array")
        if value.shape != shape:
            raise ValueError(
                f"APUg2 operand {name!r} must have shape {shape}, got {value.shape}"
            )
        return np.ascontiguousarray(value)

    @staticmethod
    def _output(value, shape):
        if value is None:
            return
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError("APUg2 output must be a NumPy uint16 array")
        if value.shape != shape:
            raise ValueError(f"APUg2 output must have shape {shape}")

    @staticmethod
    def _run_dot_grid(left, right, initial, tiling, run_dot):
        output = np.empty(tiling.output_extent, dtype=np.uint16)
        runs = []
        for output_tile in range(tiling.tile_count):
            output_begin, output_end = tiling.tile_bounds(output_tile)
            accumulator = np.ascontiguousarray(initial[output_begin:output_end])
            for reduction_tile in range(tiling.reduction_tile_count):
                reduction_begin, reduction_end = tiling.reduction_bounds(reduction_tile)
                run = run_dot(
                    np.ascontiguousarray(
                        left[
                            output_begin:output_end,
                            reduction_begin:reduction_end,
                        ]
                    ),
                    np.ascontiguousarray(
                        right[
                            output_begin:output_end,
                            reduction_begin:reduction_end,
                        ]
                    ),
                    accumulator=accumulator,
                    alpha=1,
                    beta=0 if reduction_tile == 0 else 1,
                )
                accumulator = np.asarray(run.extra["outputs"]["out"])
                runs.append(run)
            output[output_begin:output_end] = accumulator
        return output, runs

    def __call__(self, A, B, x, out=None):
        A = self._array(A, "A", (self.rows, self.reduction_extent))
        B = self._array(B, "B", (self.rows, self.reduction_extent))
        x = self._array(x, "x", (self.reduction_extent,))
        self._output(out, (self.rows,))
        if self.backend == "virtual":
            result = RunResult(
                cycles=int(self.estimate_result.cycles),
                stdout="virtual chunked APUg2 GESUMMV; hardware was not executed",
                backend="virtual",
                extra={
                    "outputs": {},
                    "hardware_tasks": self.execution_graph.metadata["hardware_tasks"],
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                },
            )
        else:
            from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile
            from .apu_g2_u16_gemm_runtime import run_apu_g2_u16_gemm

            def run_dot(left, right, **kwargs):
                return run_apu_g2_u16_dot_tile(
                    left,
                    right,
                    repetitions=self.program.repetitions,
                    **kwargs,
                )

            vector = np.ascontiguousarray(x[:, None])
            accumulator = np.zeros((self.rows, 1), dtype=np.uint16)
            matrix_runs = [
                run_apu_g2_u16_gemm(A, vector, accumulator, beta=0),
                run_apu_g2_u16_gemm(B, vector, accumulator, beta=0),
            ]
            tmp_a = matrix_runs[0].extra["outputs"]["out"][:, 0]
            tmp_b = matrix_runs[1].extra["outputs"]["out"][:, 0]
            combine_left = np.stack((tmp_a, tmp_b), axis=1)
            coefficients = np.empty_like(combine_left)
            coefficients[:, 0] = np.uint16(self.program.alpha)
            coefficients[:, 1] = np.uint16(self.program.beta)
            observed, combine_runs = self._run_dot_grid(
                combine_left,
                coefficients,
                np.zeros(self.rows, dtype=np.uint16),
                self.combine_tiling,
                run_dot,
            )
            runs = matrix_runs + combine_runs
            result = RunResult(
                cycles=sum(item.cycles for item in runs),
                stdout="\n".join(
                    f"task={index}\n{item.stdout}" for index, item in enumerate(runs)
                ),
                backend="apu_v2",
                extra={
                    "outputs": {"tmp_a": tmp_a, "tmp_b": tmp_b, "out": observed},
                    "hardware_tasks": sum(
                        item.extra.get("hardware_tasks", 1) for item in runs
                    ),
                    "per_task_cycles": [item.cycles for item in runs],
                    "total_ticks": sum(
                        item.extra.get("total_ticks", item.cycles) for item in runs
                    ),
                    "final_pipeline_ticks": sum(
                        item.extra.get("final_pipeline_ticks", item.cycles)
                        for item in runs
                    ),
                    "repetitions": self.program.repetitions,
                    "sources": [item.extra["sources"] for item in runs],
                    "projects": [item.extra["project"] for item in runs],
                    "vectorization_certificate": self.recipe.certificate.manifest(),
                },
            )
            if out is not None:
                np.copyto(out, observed)
        self.last_result = result
        return result

    run = __call__


def compile_apu_g2_gemv_workload(
    workload,
    target,
    schedule,
    *,
    cost,
    backend=None,
    promotion_gate=None,
):
    promotion_gate = validate_schedule_promotion_gate(promotion_gate)
    if promotion_gate is not None:
        raise ValueError("direct APUg2 GEMV has no schedule-search activation")
    return APUG2GemvCallable(workload, target, schedule, cost=cost, backend=backend)


def compile_apu_g2_vector_workload(
    workload,
    target,
    schedule,
    *,
    cost,
    backend=None,
    promotion_gate=None,
):
    promotion_gate = validate_schedule_promotion_gate(promotion_gate)
    try:
        structured_plan = analyze_apu_g2_structured_plan(schedule.module)
    except NoStructuredAPUG2Plan:
        structured_plan = None
    if promotion_gate is not None and structured_plan is not None:
        raise ValueError(
            "the selected structured APUg2 lowering has no schedule-search activation"
        )
    if isinstance(structured_plan, RankTwoUpdateGemvChainPlan):
        return APUG2GemverCallable(
            workload,
            target,
            schedule,
            cost=cost,
            backend=backend,
            plan=structured_plan,
        )
    if isinstance(structured_plan, SymmetricContractionPlan):
        return APUG2SymmCallable(
            workload,
            target,
            schedule,
            cost=cost,
            backend=backend,
            plan=structured_plan,
        )
    if isinstance(structured_plan, UnitDiagonalTriangularContractionPlan):
        return APUG2TrmmCallable(
            workload,
            target,
            schedule,
            cost=cost,
            backend=backend,
            plan=structured_plan,
        )
    if isinstance(structured_plan, CenteredGramStatisticsPlan):
        return APUG2CovarianceCallable(
            workload,
            target,
            schedule,
            cost=cost,
            backend=backend,
            plan=structured_plan,
        )
    if isinstance(structured_plan, NormalizedGramStatisticsPlan):
        return APUG2CorrelationCallable(
            workload,
            target,
            schedule,
            cost=cost,
            backend=backend,
            plan=structured_plan,
        )
    try:
        rank_n_plans = plan_apu_g2_rank_n_contractions(schedule.module)
    except (NoContractionError, UnsupportedAPUG2RankNContractionError):
        rank_n_plans = ()
    manifest = (
        rank_n_plans[0].module
        if rank_n_plans
        else discover_apu_g2_module_manifest(schedule.module)
    )
    unit_epilogues = bool(rank_n_plans) and all(
        plan.epilogue == (1, 1) for plan in rank_n_plans
    )
    if promotion_gate is not None:
        if manifest.contraction_topology != "single":
            raise ValueError(
                "promotion evidence requires a single persistent APUg2 GEMM search"
            )
        analysis = (
            rank_n_plans[0].analysis
            if rank_n_plans
            else analyze_contractions(schedule.module)[0]
        )
        if len(analysis.output_axes) == 1:
            raise ValueError(
                "the selected APUg2 GEMV lowering has no schedule-search activation"
            )
        try:
            return APUG2ColumnBatchedGemmCallable(
                workload,
                target,
                schedule,
                cost=cost,
                backend=backend,
                promotion_gate=promotion_gate,
            )
        except UnsupportedAPUG2ContractionError as error:
            raise ValueError(
                "the selected APUg2 lowering has no persistent schedule search"
            ) from error
    if manifest.contraction_topology == "linked_chain":
        if unit_epilogues:
            try:
                return APUG2AtaxCallable(
                    workload, target, schedule, cost=cost, backend=backend
                )
            except (UnsupportedAPUG2ContractionError, ValueError):
                pass
        return APUG2ContractionChainCallable(
            workload, target, schedule, cost=cost, backend=backend
        )
    if manifest.contraction_topology == "dag":
        return APUG2ContractionChainCallable(
            workload, target, schedule, cost=cost, backend=backend
        )
    if manifest.contraction_topology == "single":
        analysis = (
            rank_n_plans[0].analysis
            if rank_n_plans
            else analyze_contractions(schedule.module)[0]
        )
        if len(analysis.output_axes) == 1:
            if rank_n_plans and not unit_epilogues:
                return APUG2RankNContractionCallable(
                    workload, target, schedule, cost=cost, backend=backend
                )
            try:
                return APUG2GemvCallable(
                    workload, target, schedule, cost=cost, backend=backend
                )
            except ValueError:
                try:
                    return APUG2StreamingGemvCallable(
                        workload, target, schedule, cost=cost, backend=backend
                    )
                except UnsupportedAPUG2ContractionError:
                    return APUG2RankNContractionCallable(
                        workload, target, schedule, cost=cost, backend=backend
                    )
        try:
            return APUG2ColumnBatchedGemmCallable(
                workload, target, schedule, cost=cost, backend=backend
            )
        except UnsupportedAPUG2ContractionError:
            return APUG2RankNContractionCallable(
                workload, target, schedule, cost=cost, backend=backend
            )
    if manifest.contraction_topology == "independent":
        if unit_epilogues:
            try:
                return APUG2IndependentContractionsCallable(
                    workload, target, schedule, cost=cost, backend=backend
                )
            except (UnsupportedAPUG2ContractionError, ValueError):
                pass
        return APUG2ContractionChainCallable(
            workload, target, schedule, cost=cost, backend=backend
        )
    raise UnsupportedAPUG2ContractionError(
        "APUg2 structured contraction lowering does not yet emit topology "
        f"{manifest.contraction_topology!r}; roots={manifest.roots}, "
        f"sinks={manifest.sinks}"
    )


__all__ = [
    "APUG2AtaxCallable",
    "APUG2ChunkedGesummvCallable",
    "APUG2ColumnBatchedGemmCallable",
    "APUG2ContractionChainCallable",
    "APUG2CorrelationCallable",
    "APUG2CovarianceCallable",
    "APUG2GemverCallable",
    "APUG2GemvCallable",
    "APUG2IndependentContractionsCallable",
    "APUG2PersistentGemmMaterialization",
    "APUG2PersistentGemmSchedule",
    "APUG2RankNContractionCallable",
    "APUG2StreamingGemvCallable",
    "APUG2SymmCallable",
    "APUG2TrmmCallable",
    "compile_apu_g2_gemv_workload",
    "compile_apu_g2_vector_workload",
    "search_apu_g2_persistent_gemm_schedules",
]
