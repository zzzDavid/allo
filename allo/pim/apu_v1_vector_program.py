# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public callable for MLIR-discovered APU v1 vector contractions.

This module is the narrow bridge between the public ``allo.compile`` facade
and the three independent APU abstractions: MLIR-based contraction analysis,
declarative layout plans, and executable target-bound costs.
``backend="virtual"`` evaluates the selected plan's retained cost graph only
(``kind="virtual"``, nothing executed, caller arrays untouched).
``backend="functional"`` is a host-side NumPy model of the selected plan's
arithmetic order (``kind="functional"``), not compiled device code.  The
selected plan's concrete GVML realization remains attached for the device
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
from ..spmw_reduction_policy import (
    ReductionOrderError,
    is_integer_type,
    normalize_reduction_order,
)
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
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    grid_search,
)


_HOST_BACKENDS = frozenset({"functional", "virtual"})
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


# Reduction strategies that add each product into the accumulator once, in
# source order: ``temporal_accumulate`` walks the reduction axis in time and
# adds one product per step into the output (its ``partial`` flag means the
# output is accumulated across steps, not that partial sums are regrouped).
_ORDER_PRESERVING_REDUCTIONS = frozenset({"none", "serial", "temporal_accumulate"})


def apu_v1_plan_regroups(plan) -> bool:
    """True when the plan forms partial sums of the reduction and combines
    them (a group tree, pairwise or host combination)."""
    reduction = plan.reduction_strategy
    if reduction is None:
        return False
    return reduction.kind not in _ORDER_PRESERVING_REDUCTIONS


def _tile_family(candidate_or_plan):
    plan = getattr(candidate_or_plan, "plan", candidate_or_plan)
    metadata = getattr(plan, "metadata", None) or {}
    decision = metadata.get("tile_decision") or {}
    return decision.get("family")


def _family_members(candidates, name):
    """Tile variants a family name (e.g. ``spatial_gemv_group_reduction``)
    stands for; empty when ``name`` is an exact plan name."""

    if any(candidate.name == name for candidate in candidates):
        return ()
    return tuple(
        candidate for candidate in candidates if _tile_family(candidate) == name
    )


def _apply_reduction_policy(candidates, analysis, reduction_order, layout):
    """Prune plans that regroup a strict floating-point reduction.

    Returns the surviving candidates and the ``(plan name, reason)`` record of
    the pruned ones.  Integer contractions wrap modulo 2^16 and keep every
    plan.  A ``layout`` naming a pruned plan, or a policy that leaves no plan,
    raises ReductionOrderError.
    """
    if reduction_order != "strict" or is_integer_type(analysis.numeric_type):
        return tuple(candidates), ()
    keep, pruned = [], []
    for candidate in candidates:
        if apu_v1_plan_regroups(candidate.plan):
            pruned.append(
                (
                    candidate.name,
                    f"reduction_order='strict' on a {analysis.numeric_type} "
                    f"accumulator; the {candidate.plan.reduction_strategy.kind} "
                    "reduction adds the products in subgroup-tree order",
                )
            )
        else:
            keep.append(candidate)
    requested = layout.name if isinstance(layout, APUV1Plan) else layout
    if isinstance(layout, APUV1Plan) and apu_v1_plan_regroups(layout):
        pruned_names = {requested}
    else:
        pruned_names = {name for name, _reason in pruned}
        pruned_families = {
            _tile_family(candidate)
            for candidate in candidates
            if candidate.name in pruned_names and _tile_family(candidate)
        }
        pruned_names |= {
            family
            for family in pruned_families
            if not any(_tile_family(candidate) == family for candidate in keep)
        }
    if isinstance(requested, str) and requested in pruned_names:
        raise ReductionOrderError(
            f"APU v1 plan {requested!r} regroups the {analysis.numeric_type} "
            "reduction, which reduction_order='strict' forbids; pass "
            "reduction_order='reassociate' or choose one of: "
            + ", ".join(candidate.name for candidate in keep)
        )
    if not keep:
        raise ReductionOrderError(
            "reduction_order='strict': no APU v1 plan keeps the source order of "
            f"the {analysis.numeric_type} reduction (pruned: "
            + ", ".join(name for name, _reason in pruned)
            + "); pass reduction_order='reassociate' to allow regrouping"
        )
    return tuple(keep), tuple(pruned)


def _grouped_fp16_contraction(analysis, plan, arrays):
    """FP16 result of a group-tree plan in that plan's accumulation order.

    Products round to FP16; each reduction group (``group_size`` consecutive
    reduction elements, zero padded) is summed by a halving tree whose every
    add rounds to FP16; the group sums are added to the incoming output in
    ascending group order (``fp16_contraction_ops``: MUL_F16,
    GROUP_REDUCE_F16, ADD_F16).
    """
    letters = {}

    def subscripts(indices):
        return "".join(
            letters.setdefault(axis, chr(ord("a") + len(letters))) for axis in indices
        )

    def operand(access):
        return np.reshape(
            np.asarray(arrays[access.value], dtype=np.float16), access.shape
        ).astype(np.float32)

    output = np.reshape(
        np.asarray(arrays[analysis.output.value], dtype=np.float16),
        analysis.output.shape,
    ).copy()
    spec = (
        f"{subscripts(analysis.lhs.indices)},{subscripts(analysis.rhs.indices)}"
        f"->{subscripts(analysis.output.indices)}"
        f"{subscripts((analysis.reduction_axis,))}"
    )
    # An FP16 x FP16 product is exact in FP32, so one rounding gives MUL_F16.
    products = np.einsum(spec, operand(analysis.lhs), operand(analysis.rhs)).astype(
        np.float16
    )
    depth = products.shape[-1]
    group = int(plan.reduction_strategy.group_size or 1 << (depth - 1).bit_length())
    for start in range(0, depth, group):
        chunk = np.zeros(products.shape[:-1] + (group,), dtype=np.float16)
        piece = products[..., start : start + group]
        chunk[..., : piece.shape[-1]] = piece
        width = group
        while width > 1:
            width //= 2
            chunk = (chunk[..., :width] + chunk[..., width : 2 * width]).astype(
                np.float16
            )
        output = (output + chunk[..., 0]).astype(np.float16)
    return output


def _candidate_by_name(candidates, name: str):
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    available = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"unknown APU v1 vector plan {name!r}; choose one of: {available}")


def _select_plan(candidates, ranked, layout):
    if isinstance(layout, str):
        members = _family_members(candidates, layout)
        if members:
            # A family name keeps the tile a cost-ranked decision.
            names = {candidate.name for candidate in members}
            for estimate in ranked:
                if estimate.plan.name in names:
                    return estimate.plan
            raise ValueError(
                f"APU v1 plan family {layout!r} has tile variants "
                + ", ".join(sorted(names))
                + "; pass a cost program to rank them or name one variant"
            )
        return _candidate_by_name(candidates, layout).plan
    if isinstance(layout, APUV1Plan):
        return layout
    if layout is not None:
        raise TypeError("APU v1 vector layout must be an APUV1Plan, plan name, or None")
    if ranked:
        return ranked[0].plan
    raise ValueError(
        "APU v1 vector plans are selected by cost ranking; pass an executable "
        "cost program (e.g. allo.pim.costs.apu_v1_cost) or choose a plan with "
        "layout=<name>; candidates: "
        + ", ".join(candidate.name for candidate in candidates)
    )


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
        reduction_order=None,
    ):
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.module = schedule.module
        self.analysis = candidates[0].analysis
        self.reduction_order = normalize_reduction_order(reduction_order)
        candidates, self.reduction_policy_pruned = _apply_reduction_policy(
            candidates, self.analysis, self.reduction_order, layout
        )
        self.candidates = tuple(candidates)
        self.plans = tuple(candidate.plan for candidate in candidates)
        self.cost = cost
        self.schedule_search_result = None
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
            selected = self.schedule_search_result.best
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
        self.plan_selection = "cost_ranked" if layout is None else "explicit_layout"
        self.backend = "device" if backend is None else str(backend)
        if self.backend not in _HOST_BACKENDS | {"device"}:
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

    @property
    def cost_validity(self):
        graph = self.execution_graph
        return None if graph is None else graph.metadata.get("cost_validity")

    @property
    def tile_decision(self):
        """The selected plan's tile decision and every tile candidate's
        estimate (``None`` when the candidate was not realizable or was
        pruned), or ``None`` for plans without a tile family."""

        decision = self.selected_plan.metadata.get("tile_decision")
        if not decision:
            return None
        family = decision["family"]
        scored = {
            estimate.plan.name: int(estimate.cycles)
            for estimate in self.candidate_estimates
        }
        members = [
            candidate
            for candidate in self.candidates
            if _tile_family(candidate) == family
        ]
        pruned = {name for name, _reason in self.reduction_policy_pruned}
        return {
            "family": family,
            "parameter": decision["parameter"],
            "selected_tile": decision["tile"],
            "selected_plan": self.selected_plan.name,
            "plan_selection": self.plan_selection,
            "candidates": [
                {
                    "tile": candidate.plan.metadata["tile_decision"]["tile"],
                    "plan": candidate.name,
                    "cycles": scored.get(candidate.name),
                }
                for candidate in members
            ],
            "legal_tiles": list(decision["candidates"]),
            "pruned_by_reduction_order": sorted(
                name for name in pruned if name.startswith(family)
            ),
        }

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
        if self.backend == "virtual":
            result = self._run_virtual(inputs)
        elif self.backend == "functional":
            result = self._run_functional(self._stage_output(inputs))
        else:
            result = self._run_device(self._stage_output(inputs))
        result.extra["reduction_order"] = self.reduction_order
        result.extra["plan_regroups_reduction"] = apu_v1_plan_regroups(
            self.selected_plan
        )
        self.last_result = result
        return result

    def _stage_output(self, inputs: Mapping[str, object]) -> dict:
        """Inputs as the executed program sees them.

        The device stages every input before it writes the output, so an
        input passed as (a view of) the output array keeps its pre-call
        value.  A source that fills the output with literal 0 before
        accumulating (``analysis.output_initializer == "zero"``) starts from
        zero, not from the caller's values the device would otherwise load.
        """
        arguments = dict(inputs)
        output_name = self.analysis.output.value
        output = arguments.get(output_name)
        if not isinstance(output, np.ndarray):
            return arguments
        for name, value in arguments.items():
            if (
                name != output_name
                and isinstance(value, np.ndarray)
                and np.shares_memory(value, output)
            ):
                arguments[name] = value.copy()
        if self.analysis.output_initializer == "zero":
            output[...] = 0
        return arguments

    def _run_virtual(self, inputs: Mapping[str, object]):
        """Evaluate the selected plan's retained cost graph; nothing executes.

        ``kind="virtual"``: no outputs, caller arrays are not written.
        """
        for name, value in inputs.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"APU v1 vector operand {name!r} must be a NumPy array, "
                    f"got {type(value).__name__}"
                )
        if self.cost is None or self.execution_graph is None:
            raise RuntimeError(
                "APU v1 vector backend='virtual' needs an executable cost spec"
            )
        if (
            self.realization is not None
            and getattr(self.analysis, "output_initializer", "carried") != "zero"
        ):
            # Price only inputs the plan can take: the executing routes reject
            # a nonzero initial output the plan resets; checked, not written.
            self.realization.check_initial_outputs(inputs)
        estimate = self.cost.evaluate(self.execution_graph)
        spec = getattr(self.cost, "spec", self.cost)
        return RunResult(
            estimate.cycles,
            "virtual APU v1 vector contraction cost evaluation; device not "
            "launched, nothing executed, outputs not written",
            "virtual",
            kind="virtual",
            extra={
                "critical_path": list(estimate.critical_path),
                "utilization": dict(estimate.utilization),
                "bottlenecks": list(estimate.bottlenecks),
                "model_fingerprint": estimate.model_fingerprint,
                "cost_model": getattr(spec, "name", None),
                "priced_target": self.target.name,
                "cost_validity": self.cost_validity,
                "plan": self.selected_plan.name,
                "plan_selection": self.plan_selection,
                "tile_decision": self.tile_decision,
            },
        )

    def _run_functional(self, inputs: Mapping[str, object]):
        """Host-side NumPy model of the selected plan's arithmetic order.

        Not compiled device code: neither the GVML realization nor any
        lowered program runs.  A plan that regroups an FP16 reduction is
        modelled by ``_grouped_fp16_contraction`` (FP16 products, the
        plan's FP16 group tree, groups added in ascending order); any other
        plan keeps the source order, so the original Allo function is
        executed on the host as that model.  ``kind="functional"``; the
        outputs are written to the caller's arrays and ``cycles`` is the
        selected plan's analytical estimate, not a measurement.
        """
        arguments = dict(inputs)
        for name, value in arguments.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"APU v1 vector operand {name!r} must be a NumPy array, "
                    f"got {type(value).__name__}"
                )
        if self.realization is not None:
            # Same contract as the device route: a plan without output ingress
            # rejects a nonzero initial output instead of diverging from it.
            self.realization.check_initial_outputs(arguments)
        output_name = self.analysis.output.value
        returned = None
        if (
            apu_v1_plan_regroups(self.selected_plan)
            and self.analysis.multiply_operation == "arith.mulf"
            and output_name in arguments
        ):
            # The selected plan regroups an FP16 reduction: reproduce its
            # order, not the source order.
            destination = arguments[output_name]
            np.copyto(
                destination,
                _grouped_fp16_contraction(
                    self.analysis, self.selected_plan, arguments
                ).reshape(destination.shape),
                casting="same_kind",
            )
            numerics = (
                "selected plan order: FP16 products, halving FP16 tree over each "
                f"reduction group of {self.selected_plan.reduction_strategy.group_size} "
                "(model of gvml_add_subgrps; not checked against the device), "
                "groups added to the output in ascending order"
            )
        else:
            # ``allo.reduction(1-D)`` is represented by ``np.ndindex`` when an
            # Allo DSL function is executed directly, so its induction variable
            # is a one-element tuple.  NumPy currently accepts that
            # scalar-shaped access but emits a deprecation warning.  This
            # functional reference is intentionally temporary; the retained
            # MLIR, not this warning, is the compilation source of truth.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Conversion of an array with ndim > 0 to a scalar",
                    category=DeprecationWarning,
                )
                returned = self.workload(**arguments)
            numerics = (
                "source order"
                if not apu_v1_plan_regroups(self.selected_plan)
                else "source order (integer or bitwise reduction: every grouping "
                "of the selected plan gives the same wrapped result)"
            )
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
            "host-side NumPy model of the selected APU v1 plan's arithmetic "
            "order; not compiled device code",
            "apu_v1-functional",
            kind="functional",
            extra={
                "outputs": outputs,
                "plan": self.selected_plan.name,
                "plan_selection": self.plan_selection,
                "tile_decision": self.tile_decision,
                "cost_validity": self.cost_validity,
                "analytical": self.selected_estimate is not None,
                "functional": True,
                "execution": "host NumPy model of the selected plan's arithmetic "
                "order; not compiled device code",
                "numerics": numerics,
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
        # A lifted 1-D contraction plans its values with a trailing unit axis.
        arrays = dict(inputs)
        lifted = {}
        for value in self.realization.values:
            source = arrays.get(value.name)
            if isinstance(source, np.ndarray) and tuple(source.shape) + (1,) == tuple(
                value.shape
            ):
                lifted[value.name] = source
                arrays[value.name] = np.reshape(source, tuple(value.shape))
        result = run(self, arrays)
        for value in self.realization.values:
            source = lifted.get(value.name)
            if (
                source is not None
                and value.intent in {"out", "inout"}
                and not np.shares_memory(source, arrays[value.name])
            ):
                np.copyto(
                    source,
                    np.reshape(arrays[value.name], source.shape),
                    casting="same_kind",
                )
        return result


def compile_apu_v1_vector_workload(
    workload,
    target,
    schedule,
    *,
    cost=None,
    layout=None,
    backend=None,
    reduction_order=None,
):
    """Analyze one ordinary Allo contraction and return its public callable.

    Under ``reduction_order="strict"`` (the default) a floating-point
    contraction ranks only plans that keep the source accumulation order.
    """

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
        reduction_order=reduction_order,
    )


__all__ = [
    "APUv1VectorCallable",
    "apu_v1_plan_regroups",
    "compile_apu_v1_vector_workload",
    "search_apu_v1_vector_plans",
]
