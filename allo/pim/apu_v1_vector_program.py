# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public callable for MLIR-discovered APU v1 vector contractions.

This module is the narrow bridge between the public ``allo.compile`` facade
and the three independent APU abstractions: MLIR-based contraction analysis,
declarative layout plans, and executable target-bound costs.  Functional
execution deliberately uses the original Allo function as a NumPy reference;
the selected plan's concrete GVML realization remains attached for the device
runtime and for inspection.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..spmw_codegen import RunResult
from .apu_v1_layout import APUV1Plan
from .apu_v1_vector_codegen import (
    UnsupportedVectorOperation,
    VectorOp,
    fp16_contraction_ops,
    realize_apu_v1_plan,
    uint16_contraction_ops,
    xnor_popcount_ops,
)
from .apu_v1_vector_cost import (
    estimate_apu_v1_plan,
    estimate_apu_v1_realization,
    rank_apu_v1_plans,
)
from .apu_v1_vectorize import generate_apu_v1_vectorization_candidates
from .schedule_promotion import validate_schedule_promotion_gate
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    grid_search,
    guarded_schedule_activation,
)


_FUNCTIONAL_BACKENDS = frozenset({"functional", "virtual"})
_APU_V1_PHYSICAL_PLAN_FIELDS = (
    "iteration_layout",
    "value_layouts",
    "transfers",
    "temporal_strategy",
    "reduction_strategy",
    "operations",
    "vr_allocations",
    "output_batching",
    "accumulator_block",
)
_APU_V1_PHYSICAL_METADATA_FIELDS = (
    "dtype",
    "problem_shape",
    "axis_extents",
    "loop_roles",
    "tile_sizes",
    "operation_counts",
    "scalar_work_items",
)


@dataclass(frozen=True)
class APUV1PlanDecision:
    """Name-free physical APUv1 plan descriptor used by schedule search."""

    temporal_strategy: str
    reduction_kind: str
    reduction_group_size: int
    accumulator_block: int
    physical_iteration_shape: tuple[int, ...]
    output_tile_shape: tuple[int, ...]
    reduction_tile_extent: int
    transfer_routes: tuple[tuple[str, ...], ...]
    physical_fingerprint: str


def _canonical_apu_v1_manifest_value(value, replacements, parent_key=None):
    if isinstance(value, dict):
        semantic_key_maps = {
            "bases",
            "input_dims",
            "logical_extents",
            "padded_extents",
            "padded_tile_extents",
            "axis_extents",
            "tile_sizes",
            "loop_roles",
            "physical_tile_extents",
            "work_tile_extents",
        }
        return {
            (
                replacements.get(str(key), str(key))
                if parent_key in semantic_key_maps
                else str(key)
            ): _canonical_apu_v1_manifest_value(item, replacements, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _canonical_apu_v1_manifest_value(item, replacements, parent_key)
            for item in value
        ]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _apu_v1_physical_manifest(plan):
    manifest = plan.manifest()
    physical = {field: manifest[field] for field in _APU_V1_PHYSICAL_PLAN_FIELDS}
    metadata = manifest.get("metadata", {})
    physical["cost_metadata"] = {
        field: metadata[field]
        for field in _APU_V1_PHYSICAL_METADATA_FIELDS
        if field in metadata
    }
    return physical


def _apu_v1_plan_decision(plan):
    if not isinstance(plan, APUV1Plan):
        fingerprint = hashlib.sha256(repr(plan).encode()).hexdigest()
        return APUV1PlanDecision(
            "test-only",
            "test-only",
            0,
            1,
            (),
            (),
            1,
            (),
            fingerprint,
        )
    roles = dict(plan.metadata.get("loop_roles", {}))
    axis_counts = {}
    replacements = {}
    for axis in plan.iteration_layout.axes:
        role = str(roles.get(axis, "axis"))
        index = axis_counts.get(role, 0)
        axis_counts[role] = index + 1
        replacements[str(axis)] = f"<{role}_{index}>"
    for index, value_layout in enumerate(plan.value_layouts):
        replacements[str(value_layout.value)] = f"<operand_{index}>"
    manifest = _apu_v1_physical_manifest(plan)
    canonical = _canonical_apu_v1_manifest_value(manifest, replacements)
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reduction = plan.reduction_strategy
    batching = plan.output_batching
    return APUV1PlanDecision(
        str(plan.temporal_strategy),
        str(getattr(reduction, "kind", "none")),
        int(getattr(reduction, "group_size", 0) or 0),
        int(plan.accumulator_block),
        tuple(int(value) for value in plan.iteration_layout.layout.physical_out_sizes),
        (
            ()
            if batching is None
            else tuple(int(value) for _axis, value in batching.work_tile_extents)
        ),
        1 if batching is None else int(batching.reduction_tile_extent),
        tuple(
            tuple(step.kind for step in transfer.route) for transfer in plan.transfers
        ),
        fingerprint,
    )


def _same_apu_v1_physical_plan(lhs, rhs):
    if lhs is rhs:
        return True
    if not isinstance(lhs, APUV1Plan) or not isinstance(rhs, APUV1Plan):
        return False
    return _apu_v1_plan_decision(lhs) == _apu_v1_plan_decision(rhs)


def _estimate_for_apu_v1_plan(estimates, plan):
    return next(
        (
            estimate
            for estimate in estimates
            if _same_apu_v1_physical_plan(estimate.plan, plan)
        ),
        None,
    )


def _apu_v1_cycle_objective_domain(estimate):
    return ScheduleObjectiveDomain.fingerprinted_target(
        metric="cycles",
        target=estimate.graph.metadata["target"],
        model_fingerprint=estimate.estimate.model_fingerprint,
        fidelity="analytical",
        scope="region",
        unit="cycles",
        direction="minimize",
    )


def _candidate_by_name(candidates, name: str):
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    available = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"unknown APU v1 vector plan {name!r}; choose one of: {available}")


def _select_plan(candidates, ranked, layout):
    if isinstance(layout, str):
        return _candidate_by_name(candidates, layout).plan
    if isinstance(layout, APUV1Plan):
        return layout
    if layout is not None:
        raise TypeError("APU v1 vector layout must be an APUV1Plan, plan name, or None")
    if ranked:
        return ranked[0].plan
    # A cost program is optional for functional development.  In that case the
    # most capable MICRO recipe is the deterministic default, not a pretend
    # analytical ranking.
    return candidates[-1].plan


def _numpy_dtype(dtype: str):
    value = str(dtype).lower()
    aliases = {"f16": "float16", "bf16": "float16", "index": "int64"}
    if value in aliases:
        return np.dtype(aliases[value])
    if value.startswith("ui"):
        return np.dtype(f"uint{value[2:]}")
    if value.startswith("i"):
        return np.dtype(f"int{value[1:]}")
    return np.dtype(value)


def _realization_operations(analysis, plan):
    """Build operand-bearing ops from the plan's proven contraction roles."""

    lhs = analysis.lhs.value
    rhs = analysis.rhs.value
    output = analysis.output.value
    reduction = plan.reduction_strategy
    group_size = int(getattr(reduction, "group_size", 0) or 1)
    temporal = getattr(reduction, "kind", "none") == "temporal_accumulate"
    product = analysis.multiply_operation

    if product == "arith.mulf":
        if temporal:
            partial = f"__{output}_product"
            return (
                VectorOp("MUL_F16", partial, (lhs, rhs)),
                VectorOp("ADD_F16", output, (output, partial)),
            )
        return fp16_contraction_ops(
            lhs, rhs, output, group_size=group_size, accumulator=output
        )

    if product == "arith.muli" and analysis.numeric_type == "ui16":
        if temporal:
            partial = f"__{output}_product"
            return (
                VectorOp("MUL_U16", partial, (lhs, rhs)),
                VectorOp("ADD_U16", output, (output, partial)),
            )
        return uint16_contraction_ops(
            lhs, rhs, output, group_size=group_size, accumulator=output
        )

    if product == "allo.xnor_popcount":
        return xnor_popcount_ops(
            lhs,
            rhs,
            output,
            group_size=None if temporal else group_size,
            accumulator=output,
        )

    bitwise = {
        "arith.andi": "AND_16",
        "arith.ori": "OR_16",
        "arith.xori": "XOR_16",
    }.get(product)
    if bitwise is not None:
        contribution = f"__{output}_bitwise"
        operations = [VectorOp(bitwise, contribution, (lhs, rhs))]
        if temporal:
            operations.append(VectorOp("ADD_S16", output, (output, contribution)))
            return tuple(operations)
        temporary = f"__{output}_reduce_tmp"
        reduced = f"__{output}_reduced"
        operations.extend(
            [
                VectorOp("RESET_16", temporary),
                VectorOp(
                    "GROUP_REDUCE_S16",
                    reduced,
                    (contribution, temporary),
                    {"group_size": group_size, "subgroup_size": 1},
                ),
                VectorOp("ADD_S16", output, (output, reduced)),
            ]
        )
        return tuple(operations)

    # Other integer widths/sign conventions remain fail-closed.
    return None


def _realize(analysis, plan):
    operations = _realization_operations(analysis, plan)
    if operations is None:
        return None, (
            f"no faithful GVML realization for {analysis.multiply_operation} "
            f"with {plan.reduction_strategy.kind} reduction"
        )
    dtype = _numpy_dtype(analysis.numeric_type)
    if dtype.itemsize > 2:
        return None, f"APU VR lanes cannot represent {analysis.numeric_type} values"
    extents = analysis.axis_extents
    values = []
    for value_layout in plan.value_layouts:
        shape = tuple(extents[axis] for axis in value_layout.axes)
        intent = "inout" if value_layout.value == analysis.output.value else "in"
        values.append(
            {
                "name": value_layout.value,
                "shape": shape,
                "dtype": dtype,
                "intent": intent,
            }
        )
    try:
        realization = realize_apu_v1_plan(plan, operations=operations, values=values)
        # Allocation alone is not sufficient proof of a physical program.
        # Rendering validates scatter contiguity, lookup geometry, and every
        # concrete GVML call before the plan can win cost-based selection.
        realization.device_source()
        return realization, None
    except (TypeError, ValueError, UnsupportedVectorOperation) as error:
        # Planning and functional execution are useful even while a particular
        # concrete plan exceeds today's codegen envelope.  Preserve the exact
        # failure and make device_source()/device execution fail closed.
        return None, str(error)


def search_apu_v1_vector_plans(
    analysis,
    plans,
    target,
    cost,
    *,
    incumbent_plan=None,
):
    """Realize, score, and deterministically rank APU v1 vector plans."""

    ordered_plans = tuple(plans)
    incumbent_index = next(
        (index for index, plan in enumerate(ordered_plans) if plan is incumbent_plan),
        None,
    )
    if incumbent_index is None and incumbent_plan is not None:
        raise ValueError("incumbent_plan must be one of the searched plan objects")
    decision_to_plan = {}
    for plan in ordered_plans:
        decision_to_plan.setdefault(_apu_v1_plan_decision(plan), plan)
    plan_decisions = tuple(decision_to_plan)

    def materialize(plan):
        realization, error = _realize(analysis, plan)
        if realization is None:
            raise InfeasibleSchedule(
                error or "APU v1 plan has no faithful GVML realization"
            )
        if realization.plan is not plan:
            raise InfeasibleSchedule(
                "APU v1 materializer returned a different plan object"
            )
        return realization

    return grid_search(
        (DecisionDomain("plan", plan_decisions),),
        build=lambda decisions: decision_to_plan[decisions["plan"]],
        materialize=materialize,
        score=lambda realization: estimate_apu_v1_realization(
            realization, target, cost
        ),
        objective=lambda estimate: int(estimate.cycles),
        objective_domain=_apu_v1_cycle_objective_domain,
        incumbent=(
            None
            if incumbent_index is None
            else {"plan": _apu_v1_plan_decision(ordered_plans[incumbent_index])}
        ),
    )


def _feasible_candidate_estimates(search_result):
    return tuple(candidate.score for candidate in search_result.ranked)


def _legacy_realizable_incumbent(analysis, plans, target, cost):
    """Reproduce the pre-inventory best score among realizable plans."""

    feasible = []
    failures = []
    for plan in plans:
        realization, error = _realize(analysis, plan)
        if realization is not None:
            feasible.append(plan)
        else:
            failures.append(f"{plan.name}: {error}")
    if not feasible:
        raise InfeasibleSchedule(
            "APU v1 legacy selection has no faithful realization: "
            + "; ".join(failures)
        )
    return rank_apu_v1_plans(feasible, target, cost)[0].plan


class APUv1VectorCallable:
    """NumPy-callable APU v1 contraction plus plans, costs, and GVML artifact."""

    def __init__(
        self,
        workload,
        target,
        schedule,
        candidates,
        *,
        cost=None,
        layout=None,
        backend=None,
        promotion_gate=None,
    ):
        promotion_gate = validate_schedule_promotion_gate(promotion_gate)
        if promotion_gate is not None and (cost is None or layout is not None):
            raise ValueError(
                "promotion_gate requires automatic cost-ranked APU v1 search"
            )
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.analysis = candidates[0].analysis
        self.candidates = tuple(candidates)
        self.plans = tuple(candidate.plan for candidate in candidates)
        self.cost = cost
        self.schedule_search_result = None
        self.schedule_activation = None
        self.fallback_reason = None
        if layout is None and cost is not None:
            incumbent_plan = _legacy_realizable_incumbent(
                self.analysis,
                self.plans,
                target,
                cost,
            )
            self.schedule_search_result = search_apu_v1_vector_plans(
                self.analysis,
                self.plans,
                target,
                cost,
                incumbent_plan=incumbent_plan,
            )
            self.schedule_activation = guarded_schedule_activation(
                self.schedule_search_result,
                promotion_gate=promotion_gate,
            )
            selected = self.schedule_activation.active
            self.fallback_reason = self.schedule_activation.fallback_reason
            self.candidate_estimates = _feasible_candidate_estimates(
                self.schedule_search_result
            )
            self.selected_plan = selected.payload
            self.selected_estimate = selected.score
            self.realization = selected.materialized
            self.realization_error = None
        else:
            self.candidate_estimates = (
                rank_apu_v1_plans(self.plans, target, cost) if cost is not None else ()
            )
            self.selected_plan = _select_plan(
                self.candidates, self.candidate_estimates, layout
            )
            self.selected_estimate = (
                _estimate_for_apu_v1_plan(self.candidate_estimates, self.selected_plan)
                if cost is not None
                else None
            )
            if self.selected_estimate is None and cost is not None:
                self.selected_estimate = estimate_apu_v1_plan(
                    self.selected_plan, target, cost
                )
            self.realization, self.realization_error = _realize(
                self.analysis, self.selected_plan
            )
            if layout is None and self.realization is None:
                # A cost-free compile retains the historical deterministic
                # candidate fallback when its default cannot be realized.
                seen = {_apu_v1_plan_decision(self.selected_plan)}
                failures = [f"{self.selected_plan.name}: {self.realization_error}"]
                for candidate_plan in self.plans:
                    decision = _apu_v1_plan_decision(candidate_plan)
                    if decision in seen:
                        continue
                    seen.add(decision)
                    realization, error = _realize(self.analysis, candidate_plan)
                    if realization is not None:
                        self.selected_plan = candidate_plan
                        self.realization = realization
                        self.realization_error = None
                        break
                    failures.append(f"{candidate_plan.name}: {error}")
                else:
                    self.realization_error = "; ".join(failures)
        self.backend = "device" if backend is None else str(backend)
        if self.backend not in _FUNCTIONAL_BACKENDS | {"device"}:
            raise ValueError(
                "APU v1 vector contractions support backend=None/'device', "
                "'functional', or 'virtual'"
            )
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "apu_v1_vector_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.last_result = None

    @property
    def execution_graph(self):
        return self.selected_estimate.graph if self.selected_estimate else None

    @property
    def ranked_plans(self):
        return tuple(estimate.plan for estimate in self.candidate_estimates)

    def estimate(self, plan=None):
        """Return the analytical estimate for the selected or named plan."""

        if self.cost is None:
            raise RuntimeError("compiled APU v1 workload has no executable cost spec")
        if plan is None:
            return self.selected_estimate.estimate
        selected = (
            _candidate_by_name(self.candidates, plan).plan
            if isinstance(plan, str)
            else plan
        )
        if not isinstance(selected, APUV1Plan):
            raise TypeError("estimate plan must be an APUV1Plan or candidate name")
        estimate = _estimate_for_apu_v1_plan(self.candidate_estimates, selected)
        if estimate is not None:
            return estimate.estimate
        if self.schedule_search_result is not None:
            raise RuntimeError(
                f"APU v1 plan {selected.name!r} was rejected before scoring"
            )
        return estimate_apu_v1_plan(selected, self.target, self.cost).estimate

    def device_source(self):
        if self.realization is None:
            raise RuntimeError(self.realization_error or "plan has no GVML realization")
        return self.realization.device_source()

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self.run_backend(**bound.arguments)

    run = __call__

    def run_backend(self, **inputs):
        if self.backend in _FUNCTIONAL_BACKENDS:
            result = self._run_functional(inputs)
        else:
            result = self._run_device(inputs)
        self.last_result = result
        return result

    def _run_functional(self, inputs: Mapping[str, object]):
        arguments = dict(inputs)
        for name, value in arguments.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"APU v1 vector operand {name!r} must be a NumPy array, "
                    f"got {type(value).__name__}"
                )
        # ``allo.reduction(1-D)`` is represented by ``np.ndindex`` when an
        # Allo DSL function is executed directly, so its induction variable is
        # a one-element tuple.  NumPy currently accepts that scalar-shaped
        # access but emits a deprecation warning.  This functional reference
        # is intentionally temporary; the retained MLIR, not this warning, is
        # the compilation source of truth.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Conversion of an array with ndim > 0 to a scalar",
                category=DeprecationWarning,
            )
            returned = self.workload(**arguments)
        output_name = self.analysis.output.value
        outputs = {}
        if output_name in arguments:
            outputs[output_name] = np.asarray(arguments[output_name]).copy()
        elif returned is not None:
            outputs[output_name] = np.asarray(returned)
        cycles = (
            int(self.selected_estimate.cycles)
            if self.selected_estimate is not None
            else None
        )
        return RunResult(
            cycles,
            "functional NumPy execution of retained Allo contraction",
            f"apu_v1-{self.backend}",
            {
                "outputs": outputs,
                "plan": self.selected_plan.name,
                "analytical": self.selected_estimate is not None,
                "functional": True,
            },
        )

    def _run_device(self, inputs):
        if self.realization is None:
            raise RuntimeError(self.realization_error or "plan has no GVML realization")
        try:
            runtime = importlib.import_module(".apu_v1_vector_runtime", __package__)
            run = runtime.run_apu_v1_vector
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "APU v1 vector device runtime is unavailable; use "
                "backend='functional' or backend='virtual'"
            ) from error
        return run(self, dict(inputs))


def compile_apu_v1_vector_workload(
    workload,
    target,
    schedule,
    *,
    cost=None,
    layout=None,
    backend=None,
    promotion_gate=None,
):
    """Analyze one ordinary Allo contraction and return its public callable."""

    if getattr(target, "name", None) != "apu_v1":
        raise ValueError("APU v1 vector workloads require the apu_v1 target")
    candidates = generate_apu_v1_vectorization_candidates(schedule.module)
    return APUv1VectorCallable(
        workload,
        target,
        schedule,
        candidates,
        cost=cost,
        layout=layout,
        backend=backend,
        promotion_gate=promotion_gate,
    )


__all__ = [
    "APUv1VectorCallable",
    "compile_apu_v1_vector_workload",
    "search_apu_v1_vector_plans",
]
