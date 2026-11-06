# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 vector contraction plans as candidates of the shared pipeline.

The GVML vector plans of a matched contraction enter ``allo.compile`` like
every other backend's candidates (release gate G9):

* :func:`apu_v1_vector_candidates` is the ``autoschedule`` enumerator for a
  plain-function contraction on ``apu_v1``.  It applies the vector plans'
  legality to the shared semantic facts
  (:func:`allo.pim.apu_v1_vectorize.apu_v1_contraction_analysis`), builds the
  recipe catalog, prunes the plans the reduction-order policy forbids,
  realizes each plan to GVML (an unrealizable plan is kept as a candidate the
  cost stage rejects with its reason) and returns one ``Placement`` per plan.
  The placement's ``extra`` states the plan's semantic coverage, value
  layouts, transfers, launch order and ABI; the ``payload``
  (:class:`APUv1VectorCandidate`) carries the ``APUV1Plan`` and its
  realization for the backend context.
* The shared driver prices every candidate through the backend's
  ``candidate_execution_graph`` hook (:mod:`allo.spmw_apu_v1`), which
  evaluates the materialized operation inventory with ``apu_v1_cost``,
  ranks them in ``rank_matcher_placements`` and materializes the winner
  through the backend context, which instantiates
  :class:`APUv1VectorProgram`, the realized program behind the public
  callable.

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
import json
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..spmw_codegen import RunResult
from ..spmw_reduction_policy import (
    ReductionOrderError,
    is_integer_type,
    reduction_policy,
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
from .apu_v1_vectorize import (
    apu_v1_contraction_analysis,
    generate_apu_v1_vectorization_candidates,
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


def _prune_by_reduction_policy(candidates, analysis, order):
    """Split the recipe catalog by the reduction-order policy.

    Returns the surviving candidates and the ``(plan name, reason)`` records
    of the pruned ones.  Integer contractions wrap modulo 2^16 and keep every
    plan.  The shared driver reports the pruned records on every surviving
    candidate (``extra["reduction_policy_pruned"]``) and a ``layout=`` that
    names a pruned plan finds no candidate.
    """
    if order != "strict" or is_integer_type(analysis.numeric_type):
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


# Directions of a vector plan's transfers that move a value off the device
# (``apu_v1_layout.Transfer.direction``); every other direction brings it in.
_OUTBOUND_TRANSFER_DIRECTIONS = frozenset({"out", "store", "vr_to_l4", "l4_to_host"})


def _candidate_transfers(plan, output):
    """The plan's transfers as contract records.  An input is moved in by
    each of its routes.  The output is moved in (its carried contents) and
    out: a route the plan uses in both directions (the baseline's ``direct``
    result transfer) is one ``inout`` record; distinct inbound and outbound
    routes (the DMA plans' result: loaded through L1 and the VRs, stored
    back by a VR-to-L4 DMA) are one ``in`` and one ``out`` record.  Never
    two records of one value and direction."""
    from ..spmw_candidate import CandidateTransfer

    records = []
    inbound, outbound = [], []
    for transfer in plan.transfers:
        route = tuple(step.kind for step in transfer.route)
        if transfer.value != output:
            records.append(CandidateTransfer(transfer.value, "in", route))
        elif transfer.direction in _OUTBOUND_TRANSFER_DIRECTIONS:
            outbound.append(route)
        else:
            inbound.append(route)
    if plan.transfers and not (inbound or outbound):
        return records
    for route in dict.fromkeys(inbound + outbound):
        if route in inbound and route in outbound:
            direction = "inout"
        elif route in inbound:
            direction = "in" if outbound else "inout"
        else:
            direction = "out" if inbound else "inout"
        records.append(CandidateTransfer(output, direction, route))
    return records


# --------------------------------------------------------------------------- #
# Candidates of the shared pipeline
# --------------------------------------------------------------------------- #


@dataclass
class APUv1VectorCandidate:
    """One vector plan of a matched contraction, realized to GVML.

    The payload of the plan's ``Placement``.  ``realization`` is None when
    the plan has no faithful GVML realization (``realization_error`` says
    why); such a candidate is rejected by the cost stage, never selected.
    ``estimate`` caches the materialized-inventory estimate the cost stage
    computed.
    """

    plan: APUV1Plan
    analysis: Any
    realization: Any
    realization_error: str | None
    estimate: Any = None

    @property
    def name(self) -> str:
        return self.plan.name

    @property
    def family(self) -> str | None:
        return _tile_family(self.plan)

    @property
    def regroups(self) -> bool:
        return apu_v1_plan_regroups(self.plan)

    def decision(self) -> tuple:
        """Name-free physical identity of the plan (schedule-search key)."""

        decision = _apu_v1_plan_decision(self.plan)
        return (
            decision.temporal_strategy,
            decision.reduction_kind,
            decision.reduction_group_size,
            decision.accumulator_block,
            decision.physical_iteration_shape,
            decision.output_tile_shape,
            decision.reduction_tile_extent,
            decision.transfer_routes,
            decision.physical_fingerprint,
        )

    def matches_request(self, request) -> bool:
        """Whether ``layout=request`` (a plan name, a family name or an
        ``APUV1Plan``) designates this candidate."""

        if isinstance(request, str):
            return request in (self.name, self.family)
        if isinstance(request, APUV1Plan):
            return _same_apu_v1_physical_plan(request, self.plan)
        return False

    def effects(self):
        """The plan's semantic and scheduling effects in the common candidate
        contract (`allo.spmw_candidate.CandidateEffects`), checked by the
        shared driver against the program and the realized artifact."""

        from ..spmw_candidate import (
            CandidateCoverage,
            CandidateEffects,
            CandidateLaunch,
            CandidateValueLayout,
        )
        from .apu_v1_vectorize import _UNIT_AXIS

        plan = self.plan
        analysis = self.analysis
        output = analysis.output.value
        extents = analysis.axis_extents
        instances = 1
        for axis in analysis.output_axes:
            instances *= int(extents[axis])
        synthetic = (
            ((_UNIT_AXIS, 1),)
            if any(axis.name == _UNIT_AXIS for axis in analysis.axes)
            else ()
        )
        inputs = (analysis.lhs.value, analysis.rhs.value)
        if analysis.output_initializer == "carried":
            # The output's incoming contents seed the reduction: a read.
            inputs = inputs + (output,)
        return CandidateEffects(
            coverage=CandidateCoverage(
                function=analysis.function,
                operations=("MAC",),
                inputs=inputs,
                outputs=(output,),
            ),
            value_layouts=tuple(
                CandidateValueLayout(
                    layout.value, tuple(layout.axes), getattr(layout, "storage", None)
                )
                for layout in plan.value_layouts
            ),
            transfers=tuple(dict.fromkeys(_candidate_transfers(plan, output))),
            launches=(
                CandidateLaunch(
                    index=0,
                    kind="gvml_program",
                    operations=(0,),
                    instances=(0, instances),
                    reduction=(0, int(extents[analysis.reduction_axis])),
                    accumulator=analysis.output_initializer,
                    covers=(
                        plan.name,
                        plan.temporal_strategy,
                        getattr(plan.reduction_strategy, "kind", "none"),
                    ),
                ),
            ),
            abi_inputs=inputs,
            abi_outputs=(output,),
            device_scope="one APUC, whole problem, single launch",
            synthetic_axes=synthetic,
        )


def _candidate_placements(target, facts, candidates, pruned):
    """One ``Placement`` per surviving recipe; the operand registers are the
    VR classes the plan's values occupy (legality checks them against the
    ``MAC`` declaration), the payload carries the plan and its realization."""

    from ..spmw_autoschedule import Placement

    (contraction,) = facts.contractions
    registers = {}
    for access, register in zip(contraction.inputs, (target.vr0, target.vr1)):
        if access.memref_name is not None:
            registers[access.memref_name] = register
    if contraction.accumulator.memref_name is not None:
        registers[contraction.accumulator.memref_name] = target.vr2
    placements = []
    for candidate in candidates:
        realization, error = _realize(candidate.analysis, candidate.plan)
        payload = APUv1VectorCandidate(
            candidate.plan, candidate.analysis, realization, error
        )
        extra = {
            "candidate_plan": payload.name,
            "candidate_decision": payload.decision(),
            "candidate_families": (payload.family,) if payload.family else (),
            "payload": payload,
            "whole_device_launch": True,
            "reduction_policy_pruned": tuple(pruned),
            "regroups_reduction": payload.regroups,
            "effects": payload.effects(),
        }
        if "tile_decision" in candidate.plan.metadata:
            extra["tile_decision"] = dict(candidate.plan.metadata["tile_decision"])
        placements.append(
            Placement(placements=dict(registers), mode=payload.name, extra=extra)
        )
    return placements


def apu_v1_vector_candidates(target, matches) -> list:
    """``autoschedule`` enumerator: the GVML vector plans of one matched
    contraction of a plain function (shared semantic facts required)."""

    facts = matches[0].extra.get("program_facts")
    if facts is None:
        raise NotImplementedError(
            "APU v1 vector plans need the shared semantic facts of the matched "
            "program (compile through allo.compile, which derives them)"
        )
    analysis = apu_v1_contraction_analysis(facts)
    recipes = generate_apu_v1_vectorization_candidates(analysis)
    policy = reduction_policy(facts.contractions[0].match)
    order = "strict" if policy is None else policy.order
    kept, pruned = _prune_by_reduction_policy(recipes, analysis, order)
    return _candidate_placements(target, facts, kept, pruned)


# --------------------------------------------------------------------------- #
# The realized program behind the public callable
# --------------------------------------------------------------------------- #


class APUv1VectorProgram:
    """The selected vector plan of one contraction, as the backend context
    materialized it; executed by the ``apu_v1`` runner.

    ``INTROSPECTION`` names the attributes :class:`allo.compiler.CompiledCallable`
    forwards (``compiled.selected_plan`` and friends); ranking-derived views
    (``candidates``, ``ranked_plans``, ``candidate_estimates``,
    ``tile_decision``) read the shared ``PlacementRanking`` of the compiled
    artifact.
    """

    INTROSPECTION = (
        "selected_plan",
        "plans",
        "candidates",
        "ranked_plans",
        "candidate_estimates",
        "realization",
        "realization_error",
        "analysis",
        "program_facts",
        "plan_selection",
        "cost_validity",
        "tile_decision",
        "device_source",
        "selected_estimate",
    )

    def __init__(self, target, candidate: APUv1VectorCandidate, trace, placement):
        self.target = target
        self.candidate = candidate
        self.trace = trace
        self.placement = placement
        self.compiled = None  # attached by the driver (CompiledCallable)

    # -- attachment to the compiled artifact ---------------------------------

    def attach(self, compiled) -> None:
        self.compiled = compiled

    # -- the selected plan -----------------------------------------------------

    @property
    def selected_plan(self) -> APUV1Plan:
        return self.candidate.plan

    @property
    def realization(self):
        return self.candidate.realization

    @property
    def realization_error(self):
        return self.candidate.realization_error

    @property
    def analysis(self):
        return self.candidate.analysis

    @property
    def program_facts(self):
        match = self.trace.matches[0] if self.trace.matches else None
        return None if match is None else match.extra.get("program_facts")

    @property
    def plan_selection(self) -> str:
        return (
            "explicit_layout"
            if self.placement.extra.get("candidate_selection") == "forced"
            else "cost_ranked"
        )

    @property
    def selected_estimate(self):
        return self.candidate.estimate

    @property
    def execution_graph(self):
        if self.compiled is not None and self.compiled.execution_graph is not None:
            return self.compiled.execution_graph
        estimate = self.candidate.estimate
        return None if estimate is None else estimate.graph

    @property
    def cost_validity(self):
        graph = self.execution_graph
        return None if graph is None else graph.metadata.get("cost_validity")

    def device_source(self):
        if self.realization is None:
            raise RuntimeError(self.realization_error or "plan has no GVML realization")
        return self.realization.device_source()

    # -- ranking-derived views ---------------------------------------------------

    def _ranking_candidates(self):
        """``[(position, cycles, payload)]`` of the kernel's scored candidates
        in cost order, from the shared ranking; the selected candidate alone
        when no ranking was run (explicit placements)."""

        ranking = getattr(self.compiled, "placement_ranking", None)
        if ranking is None or not ranking.kernels:
            return [(0, None, self.candidate)]
        kernel = ranking.kernels[0]
        by_position = {
            position: placement.extra.get("payload")
            for position, placement in kernel.candidates
        }
        out = []
        for position, cycles, _mode in kernel.scores:
            payload = by_position.get(position)
            if isinstance(payload, APUv1VectorCandidate):
                out.append((position, cycles, payload))
        return out or [(0, None, self.candidate)]

    @property
    def candidates(self) -> tuple[APUv1VectorCandidate, ...]:
        """Every enumerated candidate, in enumeration order."""

        return tuple(
            payload
            for _position, _cycles, payload in sorted(
                self._ranking_candidates(), key=lambda item: item[0]
            )
        )

    @property
    def plans(self) -> tuple[APUV1Plan, ...]:
        return tuple(candidate.plan for candidate in self.candidates)

    @property
    def ranked_plans(self) -> tuple[APUV1Plan, ...]:
        return tuple(payload.plan for _p, _c, payload in self._ranking_candidates())

    @property
    def candidate_estimates(self) -> tuple:
        return tuple(
            payload.estimate
            for _p, _c, payload in self._ranking_candidates()
            if payload.estimate is not None
        )

    @property
    def reduction_policy_pruned(self) -> tuple:
        return tuple(self.placement.extra.get("reduction_policy_pruned", ()))

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
            payload.name: int(cycles)
            for _p, cycles, payload in self._ranking_candidates()
            if cycles is not None
        }
        members = [
            candidate for candidate in self.candidates if candidate.family == family
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

    def check_virtual_inputs(self, inputs) -> None:
        """Price only inputs the plan can take: the executing routes reject a
        nonzero initial output the plan resets; checked here, not written."""
        arrays = {
            name: value
            for name, value in inputs.items()
            if name in self.program_facts.parameter_names
        }
        for name, value in arrays.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"APU v1 vector operand {name!r} must be a NumPy array, "
                    f"got {type(value).__name__}"
                )
        if self.realization is not None and self.analysis.output_initializer != "zero":
            self.realization.check_initial_outputs(arrays)

    def virtual_extra(self) -> dict:
        return {
            "plan": self.selected_plan.name,
            "plan_selection": self.plan_selection,
            "tile_decision": self.tile_decision,
            "cost_validity": self.cost_validity,
            "plan_regroups_reduction": self.candidate.regroups,
        }

    # -- execution -----------------------------------------------------------------

    def run(self, compiled, inputs: Mapping[str, object]) -> RunResult:
        backend = compiled.backend
        if backend == "functional":
            result = self._run_functional(compiled, self._stage_output(inputs))
        elif backend in {None, "device"}:
            result = self._run_device(self._stage_output(inputs))
        else:
            raise ValueError(
                "APU v1 vector contractions support backend=None/'device', "
                f"'functional', or 'virtual'; got {backend!r}"
            )
        result.extra["plan_regroups_reduction"] = self.candidate.regroups
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

    def _run_functional(self, compiled, inputs: Mapping[str, object]):
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
        arguments = {
            name: value
            for name, value in inputs.items()
            if name in self.program_facts.parameter_names
        }
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
            self.candidate.regroups
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
            workload = getattr(compiled, "source_workload", None)
            if workload is None:
                raise RuntimeError(
                    "the APU v1 functional model executes the source workload on "
                    "the host; the compiled artifact carries no source workload"
                )
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
                returned = workload(**arguments)
            numerics = (
                "source order"
                if not self.candidate.regroups
                else "source order (integer or bitwise reduction: every grouping "
                "of the selected plan gives the same wrapped result)"
            )
        outputs = {}
        if output_name in arguments:
            outputs[output_name] = np.asarray(arguments[output_name]).copy()
        elif returned is not None:
            outputs[output_name] = np.asarray(returned)
        estimate = self.candidate.estimate
        cycles = int(estimate.cycles) if estimate is not None else None
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
                "analytical": estimate is not None,
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
        arrays = {
            name: value
            for name, value in inputs.items()
            if name in self.program_facts.parameter_names
        }
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
        result.extra.setdefault("plan", self.selected_plan.name)
        result.extra.setdefault("plan_selection", self.plan_selection)
        return result


__all__ = [
    "APUv1VectorCandidate",
    "APUv1VectorProgram",
    "apu_v1_plan_regroups",
    "apu_v1_vector_candidates",
]


def _register_route_steps():
    """Route steps the GVML realizations name: the plan transfer kinds of
    `allo.pim.apu_v1_layout` (DMA, PIO and lookup steps between L4, L3, L1
    and the VRs), which the realization check compares with the plan."""
    from ..spmw_candidate import ROUTE_VOCABULARY
    from .apu_v1_layout import _TRANSFER_ROUTE_KINDS

    ROUTE_VOCABULARY.setdefault("apu_v1", set()).update(
        {str(kind) for kind in _TRANSFER_ROUTE_KINDS}
    )


_register_route_steps()
