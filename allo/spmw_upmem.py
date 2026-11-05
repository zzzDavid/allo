# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM matcher-path backend: family-plan lowering and uPIMulator GENERIC runs.

The matcher finds the ops; :func:`classify_upmem_kernel` turns one kernel
group's ``MatchedOp`` s into a kernel *family* plus geometry. ``UpmemCtx``
instantiates that family's calibrated physical plan from
``allo.pim.upmem_physical`` with the placement's physical decision; the plan's
``device_source()`` is the DPU translation unit. ``_run_upmem`` packs a
fixture-driven ``GENERIC`` run whose expected output bytes come from an
independent portable-C functional oracle of the work-id functions, and reports
uPIMulator DPU logic cycles. Spec: design_doc/compiler/upmem-matcher-track.md
(U1-U4).
"""

from __future__ import annotations

import dataclasses
import fcntl
import functools
import hashlib
import json
import math
import operator
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import spmw_simenv
from .pim.costs import upmem as upmem_costs
from .pim.upmem_physical import (
    UPMEM_ACTIVE_TASKLETS,
    UPMEMElementwisePlan,
    UPMEMGEMMPlan,
    UPMEMMatrixVectorPlan,
    UPMEMSumReductionPlan,
    masked_12_tasklet_linear_layout,
)
from .pim.upmem_irregular import (
    GradientFormula,
    SelectionPredicate,
    UPMEMFeatureGradientPlan,
    UPMEMHistogramPlan,
    UPMEMKMeansDistancesPlan,
    UPMEMSelectionFlagsPlan,
)
from .pim.upmem_physical_search import (
    UPMEMDataLayout,
    UPMEMOperandResidency,
    UPMEMPredicateLowering,
    UPMEMPhysicalDecision,
    UPMEMPhysicalLegalityError,
    UPMEMPhysicalProblem,
    UPMEMPlanBuilderRejected,
    calibration_validity,
    physical_features,
    selection_padded_elements,
)
from .spmw_autoschedule import (
    Placement,
    _matcher_search_scope,
    _matcher_work_scope,
)
from .spmw_codegen import CodegenContext, RunResult, SimulatorUnavailable
from .spmw_match import MatchedOp

_TIMING_SCOPE = "uPIMulator DPU logic_cycle only"
# Scope of a ``backend="virtual"`` result (``UpmemCtx.virtual_run_extra``).
VIRTUAL_TIMING_SCOPE = "analytical: device logic cycles incl. modeled host transfers"
_BUILD_TIMEOUT_S = 600
_SIMULATE_TIMEOUT_S = 1800
_LOGIC_CYCLE = re.compile(r"^Logic\[(\d+)_(\d+)_(\d+)\]_logic_cycle:\s*(\d+)\s*$", re.M)
_OBSERVED = re.compile(
    r"^observed_mram_heap_offset_(\d+)_execution_(\d+)_dpu_(\d+)\.raw\.bin$"
)
# Region-local buffers get identities at or above this value
# (``ir/builder.py::_spmw_source_value_id``); caller arguments are below it.
_LOCAL_ABI_ID = 1 << 62


# --------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class UpmemKernel:
    """One matched kernel group, as the family classifier sees it."""

    family: str
    group_id: int
    geometry: tuple
    anchor_op: str
    constants: tuple
    roles: tuple
    dpus: int
    # Spec 004: "host" groups run their own MLIR on the host (paper partition).
    placement: str = "dpu"
    # The matcher's proof of the accumulator's initial value
    # (``MatchedOp.extra["accumulator_init"]``): "zero", "carried" (the
    # caller's buffer is accumulated into), or "none" (no accumulator).
    accumulator_init: str = "none"
    # Per work-id geometry, in work-id order, when the work items have
    # different extents (uneven partition).  ``geometry`` is then the largest
    # one, which the shared DPU binary is built for; the others are
    # zero-padded to it (``_padded_slices``).  Empty when all are equal.
    streams: tuple = ()

    def geometry_value(self, name: str) -> int:
        return int(dict(self.geometry)[name])

    def stream_geometry(self, index: int) -> tuple:
        return self.streams[index] if self.streams else self.geometry

    def role(self, name: str) -> str:
        return dict(self.roles)[name]


@dataclass(frozen=True)
class UpmemSegment:
    """One simulator invocation."""

    kernel: UpmemKernel
    decision: UPMEMPhysicalDecision
    plan: object
    device_source: str
    executions: int
    kind: str = "dpu"  # "host": no plan; the group's oracle runs on the host


@dataclass(frozen=True)
class UpmemFixture:
    """A version-1 ``GENERIC`` fixture directory, held in memory."""

    manifest: dict
    files: tuple

    def manifest_bytes(self) -> bytes:
        return (json.dumps(self.manifest, indent=2, sort_keys=True) + "\n").encode()

    def sha256(self) -> dict:
        hashes = {"manifest.json": hashlib.sha256(self.manifest_bytes()).hexdigest()}
        for name, data in self.files:
            hashes[name] = hashlib.sha256(data).hexdigest()
        return dict(sorted(hashes.items()))

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_bytes(self.manifest_bytes())
        for name, data in self.files:
            (directory / name).write_bytes(data)


@dataclass(frozen=True)
class UpmemBundle:
    """What ``prepare`` hands to the simulator: one fixture per segment."""

    fixtures: tuple  # per segment; None for a host segment
    expected: tuple  # per segment: {(execution, dpu): (offset, bytes)}
    states: tuple = ()  # per segment: the oracle's arrays after it


# --------------------------------------------------------------------- #
# Family registry
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class UpmemFamily:
    """A kernel family: how to recognize it and which plan template lowers it.

    ``recognize(target, matches)`` sees one work-id stream and returns
    ``{"geometry", "anchor_op", "constants", "roles"}`` or None.
    ``features(kernel)`` returns the ``UPMEMPhysicalProblem`` whose
    ``physical_features(problem, decision)`` price and legality-check a decision.
    ``domain`` is the ordered tuple of physical decisions the family may take.
    """

    name: str
    recognize: Callable
    plan_for: Callable
    features: Callable
    domain: tuple
    # The one geometry key whose extent may differ between work items, padded
    # with zeros the template's result for which is discarded (a
    # padding-neutral axis); () when every work item must match exactly.
    uneven: tuple = ()
    # Whether the template implements accumulator_init='carried'.
    carried: bool = False


_UPMEM_FAMILIES: dict[str, UpmemFamily] = {}
# Family name -> ordered physical decision domain (spec U5).  Each domain is
# a finite, curated candidate catalog (the calibration campaign's search
# lists), not an exhaustive space: selection among its realizable members is
# by cost, with no optimality claim beyond the catalog.
UPMEM_DECISION_DOMAINS: dict[str, tuple] = {}


def register_upmem_family(
    name, recognize, plan_for, features, domain, *, uneven=(), carried=False
) -> UpmemFamily:
    family = UpmemFamily(
        name, recognize, plan_for, features, tuple(domain), tuple(uneven), carried
    )
    _UPMEM_FAMILIES[name] = family
    UPMEM_DECISION_DOMAINS[name] = family.domain
    return family


def upmem_family(name: str) -> UpmemFamily:
    return _UPMEM_FAMILIES[name]


def _streams(matches) -> list[list[MatchedOp]]:
    streams: dict[tuple, list[MatchedOp]] = {}
    for match in matches:
        streams.setdefault(_matcher_work_scope(match).work_id, []).append(match)
    return [streams[key] for key in sorted(streams)]


def classify_upmem_kernel(target, matches) -> UpmemKernel:
    """Classify one kernel group's matches into a registered family.

    A pure function of the ``MatchedOp`` s: it never reads the workload name.
    Every work-id stream must classify identically.
    """
    matches = list(matches)
    if not matches:
        raise NotImplementedError("UPMEM kernel group has no matched ops")
    group_ids = {_matcher_work_scope(match).group_id for match in matches}
    if len(group_ids) != 1:
        raise ValueError(f"matches span several kernel groups: {sorted(group_ids)}")
    streams = _streams(matches)
    found = None
    for family in _UPMEM_FAMILIES.values():
        shapes = [family.recognize(target, stream) for stream in streams]
        if any(shape is None for shape in shapes):
            continue
        found = (family, _merge_stream_shapes(family, shapes))
        break
    if found is None:
        ops = sorted({match.target_op_name for match in matches})
        functions = sorted({match.func_name for match in matches})
        raise NotImplementedError(
            f"no UPMEM kernel family matches ops {ops} in {functions}. "
            f"Families: {_FAMILY_FORMS}"
        )
    family, shape = found
    for stream in streams:
        anchors = [m for m in stream if m.target_op_name == shape["anchor_op"]]
        if len(anchors) != 1:
            raise NotImplementedError(
                f"UPMEM family {family.name!r} needs exactly one "
                f"{shape['anchor_op']} per work-id stream; "
                f"{stream[0].func_name} has {len(anchors)}"
            )
    init = shape.get("accumulator_init", "none")
    if init == "carried" and not family.carried:
        raise NotImplementedError(
            f"UPMEM {family.name}: the source accumulates into the caller's "
            f"{dict(shape['roles']).get('out', 'output')!r} without initializing "
            "it (accumulator_init='carried'); the "
            f"{family.name} DPU template computes the result from zero and "
            "would drop the initial value. Zero the output in the kernel "
            "before accumulating."
        )
    return UpmemKernel(
        family=family.name,
        group_id=group_ids.pop(),
        geometry=tuple(shape["geometry"]),
        anchor_op=shape["anchor_op"],
        constants=tuple(shape["constants"]),
        roles=tuple(shape["roles"]),
        dpus=len(streams),
        placement=shape.get("placement", "dpu"),
        accumulator_init=init,
        streams=tuple(shape.get("streams", ())),
    )


_FAMILY_FORMS = (
    "pointwise (one ADD/SUB/MUL/SCALE/AXPBY/AXMBY/AXPBYC over one loop), "
    "reduction (out[c] += in[i]), "
    "matrix_vector (one MAC over 2-3 loops, optionally y = alpha * acc via SCALE), "
    "gemm (C[i,j] += A[i,k] * B[k,j]), "
    "histogram (H[(x*m)>>k] += 1), "
    "selection (COMPACT + INC under one guard, or SELECT_ODD), "
    "kmeans_distances (SQDIST_ACC), kmeans_update (host argmin/scatter update), "
    "feature_gradient (GRAD_LINEAR/GRAD_LOGISTIC)"
)


def _merge_stream_shapes(family: UpmemFamily, shapes: list) -> dict:
    """One kernel shape for all work-id streams of a group.

    Identical streams share it.  Otherwise the streams may differ only in the
    family's padding-neutral key (``family.uneven``): the merged geometry
    takes its largest extent, and ``streams`` keeps every work item's own.
    """
    first = shapes[0]
    if all(shape == first for shape in shapes[1:]):
        return first
    for field in ("anchor_op", "constants", "roles", "accumulator_init"):
        values = {repr(shape.get(field)) for shape in shapes}
        if len(values) != 1:
            raise NotImplementedError(
                f"UPMEM family {family.name!r}: work-id streams of one kernel "
                f"group differ in {field}; one DPU binary serves every work item"
            )
    geometries = [dict(shape["geometry"]) for shape in shapes]
    varying = sorted(
        {key for g in geometries for key in g if g[key] != geometries[0][key]}
    )
    if not family.uneven:
        raise NotImplementedError(
            f"UPMEM family {family.name!r}: work items have different extents "
            f"({', '.join(varying)}); this template has no padding-neutral axis, "
            "so uneven partitions need equal work items"
        )
    if len(varying) != 1 or varying[0] not in family.uneven:
        raise NotImplementedError(
            f"UPMEM family {family.name!r}: work items differ in "
            f"{', '.join(varying)}; an uneven partition may vary only "
            f"{' or '.join(family.uneven)} (one axis), which the template "
            "zero-pads"
        )
    key = varying[0]
    if key == "rows" and geometries[0].get("batches", 1) != 1:
        raise NotImplementedError(
            f"UPMEM family {family.name!r}: batched work items differ in rows "
            "per batch; only the batch count may vary between batched work items"
        )
    largest = max(g[key] for g in geometries)
    merged = dict(first)
    merged["geometry"] = tuple(
        (name, largest if name == key else value) for name, value in first["geometry"]
    )
    merged["streams"] = tuple(tuple(shape["geometry"]) for shape in shapes)
    return merged


def _static_extent(loop) -> int | None:
    from .spmw_tripcount import _parse_loop_bound

    _name, lower, upper, step = loop
    if _parse_loop_bound(lower) != 0 or int(step) != 1:
        return None
    extent = _parse_loop_bound(upper)
    return extent if extent is not None and extent > 0 else None


def _memref_rank(operand) -> int | None:
    text = getattr(operand, "memref_type", None)
    match = re.fullmatch(r"memref<((?:\d+x)*)[a-z]+\d*>", text or "")
    if match is None:
        return None
    return match.group(1).count("x")


# ---------------------------- element types ---------------------------- #

# The element type of every role as the family's DPU template computes it
# (``pim/upmem_physical.py``, ``pim/upmem_irregular.py``): operands are packed
# into int32 MRAM words and the template's arithmetic and output words have
# exactly these widths. The device's int32 wraparound equals the source's
# int32 semantics only for these types, so every other type is rejected.
# Signedness is not part of MLIR's signless integer types; the final compile
# checks it on the work function's ABI (``_check_role_dtypes``).
UPMEM_ROLE_ELEMENT_TYPES = {
    "pointwise": {"lhs": "i32", "rhs": "i32", "out": "i32"},
    "reduction": {"input": "i32", "out": "i64"},
    "matrix_vector": {"matrix": "i32", "vector": "i32", "acc": "i32", "out": "i32"},
    "gemm": {"lhs": "i32", "rhs": "i32", "out": "i32"},
    "histogram": {"input": "i32", "out": "i32"},
    "selection": {"input": "i32", "out": "i32", "count": "i32"},
    "kmeans_distances": {"points": "i32", "centroids": "i32", "out": "i64"},
    "feature_gradient": {"samples": "i32", "out": "i64"},
}
_ABI_DTYPE_NAMES = {"i32": "int32", "i64": "int64"}


def _element_type(memref_type) -> str | None:
    match = re.match(r"memref<(?:\d+x)*([a-z]+\d*)", memref_type or "")
    return match.group(1) if match else None


def _supported_types(family: str) -> str:
    return ", ".join(
        f"{role}: {_ABI_DTYPE_NAMES[dtype]}"
        for role, dtype in UPMEM_ROLE_ELEMENT_TYPES[family].items()
    )


def _require_element_type(family: str, role: str, operand) -> None:
    """Fail recognition when ``operand`` (an ``OperandBinding`` or ``WLoad``)
    does not have the element type the family template computes."""
    want = UPMEM_ROLE_ELEMENT_TYPES[family][role]
    memref_type = getattr(operand, "memref_type", None)
    if _element_type(memref_type) != want:
        raise NotImplementedError(
            f"UPMEM {family}: operand {operand.memref_name!r} (role {role!r}) has "
            f"MLIR type {memref_type or 'unknown'}; the {family} DPU template "
            f"computes {_ABI_DTYPE_NAMES[want]} for this role (supported: "
            f"{_supported_types(family)}). Declare the buffer with that type."
        )


def _require_int32_constant(family: str, name: str, value) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not (-(2**31) <= value < 2**31)
    ):
        raise NotImplementedError(
            f"UPMEM {family}: coefficient {name}={value!r} is not an int32 "
            "literal; the DPU template computes in int32"
        )


def _accumulator_init(family: str, match) -> str:
    """The matcher's initialization proof for ``match``'s accumulator.

    ``spmw_match_engine`` sets ``extra["accumulator_init"]`` on every match
    with a loop-carried operand: "zero" (a dominating zero store covers the
    accumulated element) or "carried" (no initializer; the caller's value is
    accumulated into).  Anything else never reaches a backend, so a missing
    fact fails closed here.
    """
    init = match.extra.get("accumulator_init")
    if init not in ("zero", "carried"):
        raise NotImplementedError(
            f"UPMEM {family}: {match.target_op_name} in {match.func_name!r} has no "
            f"proven accumulator initialization (accumulator_init={init!r})"
        )
    return init


# ------------------------------ pointwise ------------------------------ #

# Matched op -> (plan operation, physical-model instructions per element).
# The matcher reports the source's pointwise expression as one of these
# target ops (``pim/targets.py::build_upmem_target``) with its int32 literal
# coefficients: ``x + y``, ``x - y``, ``x * y``, ``alpha x``,
# ``alpha x + beta y``, ``alpha x - beta y`` and ``alpha x + beta y + gamma``.
# The last three lower to the one int32 AXPBY template: a subtraction is
# ``alpha x + (-beta) y`` and the constant is one more int32 add.
# Instruction counts are analytical: one native add/sub, a constant multiply
# as one instruction, and a variable 32-bit multiply as the 32-step
# ``mul_step`` sequence plus one.  Other expressions are rejected by the
# matcher's coverage check before a family is chosen.
_POINTWISE_OPERATIONS = {
    "ADD": ("add", 1),
    "SUB": ("sub", 1),
    "MUL": ("mul", 33),
    "SCALE": ("scale", 1),
    "AXPBY": ("axpby", 2),
    "AXMBY": ("axpby", 2),
    "AXPBYC": ("axpby", 3),
}
# Sign the AXPBY template applies to the matched ``beta``.
_POINTWISE_BETA_SIGN = {"AXMBY": -1}
_POINTWISE_OPERAND_ROLES = {"scale": ("x",)}


def _data_operands(match) -> dict:
    """Operands bound to memref loads, by role (constant parameters excluded)."""
    return {
        operand.role: operand
        for operand in match.operands
        if operand.memref_name is not None
    }


def _loop_extents(match) -> list | None:
    extents = [_static_extent(loop) for loop in match.enclosing_loops]
    return None if None in extents else extents


def _recognize_pointwise(target, stream):
    """``out[i] = f(lhs[i], rhs[i])`` (or ``f(lhs[i])``) as one ADD, SUB, MUL,
    SCALE, AXPBY, AXMBY or AXPBYC over a single loop."""
    del target
    if len(stream) != 1 or stream[0].target_op_name not in _POINTWISE_OPERATIONS:
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 1:
        return None
    (iv,) = (loop[0] for loop in match.enclosing_loops)
    operation = _POINTWISE_OPERATIONS[match.target_op_name][0]
    operands = _data_operands(match)
    names = _POINTWISE_OPERAND_ROLES.get(operation, ("x", "y"))
    if set(operands) != set(names):
        return None
    inputs = [operands[name] for name in names]
    if any(op.is_loop_carried for op in inputs):
        return None
    if any(_memref_rank(op) != 1 or list(op.indices) != [iv] for op in inputs):
        return None
    if match.result_memref_name in (None, *(op.memref_name for op in inputs)):
        return None
    if match.extra.get("store_indices") != [iv]:
        return None
    for role, operand in zip(("lhs", "rhs"), inputs):
        _require_element_type("pointwise", role, operand)
    constants = tuple(sorted(match.extra.get("constants", {}).items()))
    for name, value in constants:
        _require_int32_constant("pointwise", name, value)
    sign = _POINTWISE_BETA_SIGN.get(match.target_op_name, 1)
    if sign != 1:
        _require_int32_constant("pointwise", "-beta", sign * dict(constants)["beta"])
    return {
        "geometry": (("elements", extents[0]),),
        "anchor_op": match.target_op_name,
        "constants": constants,
        "roles": tuple(
            (role, operand.memref_name) for role, operand in zip(("lhs", "rhs"), inputs)
        )
        + (("out", match.result_memref_name),),
    }


def _pointwise_operation(kernel: UpmemKernel) -> tuple[str, int]:
    return _POINTWISE_OPERATIONS[kernel.anchor_op]


def _pointwise_problem(kernel: UpmemKernel) -> UPMEMPhysicalProblem:
    operation, instructions = _pointwise_operation(kernel)
    return UPMEMPhysicalProblem.pointwise(
        kernel.geometry_value("elements"),
        input_operands=len(_POINTWISE_OPERAND_ROLES.get(operation, ("x", "y"))),
        operation_instructions=instructions,
    )


def _pointwise_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    interleaved = decision.layout is UPMEMDataLayout.FUSED_REPLICATED
    operation = _pointwise_operation(kernel)[0]
    coefficients = {}
    constants = dict(kernel.constants)
    if operation == "axpby":
        sign = _POINTWISE_BETA_SIGN.get(kernel.anchor_op, 1)
        coefficients = {
            "alpha": int(constants["alpha"]),
            "beta": sign * int(constants["beta"]),
            "gamma": int(constants.get("gamma", 0)),
        }
    elif operation == "scale":
        coefficients = {"alpha": int(constants["alpha"])}
    return UPMEMElementwisePlan(
        kernel.geometry_value("elements"),
        operation=operation,
        dma_bytes=decision.dma_bytes // 2 if interleaved else decision.dma_bytes,
        interleaved=interleaved,
        **coefficients,
    )


# The campaign's curated list, ``_pointwise_search_decisions``.
_POINTWISE_DOMAIN = (
    UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 64, 16),
    UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 128, 32),
    UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 256, 64),
    UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 1024, 256),
    UPMEMPhysicalDecision(UPMEMDataLayout.FUSED_REPLICATED, 256, 32),
)

register_upmem_family(
    "pointwise",
    _recognize_pointwise,
    _pointwise_plan,
    _pointwise_problem,
    _POINTWISE_DOMAIN,
    uneven=("elements",),
)


# ------------------------------ reduction ------------------------------ #


def _recognize_reduction(target, stream):
    """``out[c] = out[c] + in[i]``: one self-updating ADD over a single loop,
    into an int64 cell whose index does not depend on the loop."""
    del target
    if len(stream) != 1 or stream[0].target_op_name != "ADD":
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 1:
        return None
    (iv,) = (loop[0] for loop in match.enclosing_loops)
    operands = _data_operands(match)
    cells = [
        op for op in operands.values() if op.memref_name == match.result_memref_name
    ]
    values = [
        op for op in operands.values() if op.memref_name != match.result_memref_name
    ]
    if len(cells) != 1 or len(values) != 1:
        return None
    (cell,), (value,) = cells, values
    if iv in cell.indices or match.extra.get("store_indices") != list(cell.indices):
        return None
    if _memref_rank(value) != 1 or list(value.indices) != [iv]:
        return None
    _require_element_type("reduction", "input", value)
    _require_element_type("reduction", "out", cell)
    return {
        "geometry": (("elements", extents[0]),),
        "anchor_op": "ADD",
        "constants": (),
        "roles": (("input", value.memref_name), ("out", cell.memref_name)),
        "accumulator_init": _accumulator_init("reduction", match),
    }


def _reduction_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    return UPMEMSumReductionPlan(
        kernel.geometry_value("elements"),
        dma_bytes=decision.dma_bytes,
        carried_output=kernel.accumulator_init == "carried",
    )


# Not part of the archived physical search: one decision, the plan defaults
# (64-byte DMA, 16-element chunks). The physical model has no reduction kind,
# so ``features`` is None and the stream keeps the per-op cost rules.
_REDUCTION_DOMAIN = (UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 64, 16),)

register_upmem_family(
    "reduction",
    _recognize_reduction,
    _reduction_plan,
    None,
    _REDUCTION_DOMAIN,
    uneven=("elements",),
    carried=True,
)


# ---------------------------- matrix_vector ---------------------------- #


def _recognize_matrix_vector(target, stream):
    """A MAC over ``(outer..., reduction)`` loops: ``out[o] += M[o, k] * v[k]``.

    The outer loops flatten into MV rows (TTV). A vector indexed by the first
    outer loop makes that loop the batch axis (MMTV). The accumulator is either
    the output itself, or a local scalar that one ``SCALE`` match writes to the
    output (scaled GEMV).
    """
    del target
    macs = [m for m in stream if m.target_op_name == "MAC"]
    scales = [m for m in stream if m.target_op_name == "SCALE"]
    if len(macs) != 1 or len(scales) > 1 or len(macs) + len(scales) != len(stream):
        return None
    match = macs[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) not in (2, 3):
        return None
    ivs = [loop[0] for loop in match.enclosing_loops]
    outer, red_iv = ivs[:-1], ivs[-1]
    operands = _data_operands(match)
    acc = operands.get("acc")
    inputs = [operands.get("x"), operands.get("y")]
    if acc is None or not acc.is_loop_carried or None in inputs:
        return None
    matrix = next((op for op in inputs if list(op.indices) == outer + [red_iv]), None)
    vector = next((op for op in inputs if op is not matrix), None)
    if matrix is None or vector is None or _memref_rank(matrix) != len(ivs):
        return None
    if list(vector.indices) == [red_iv]:
        batches, rows = 1, 1
        for extent in extents[:-1]:
            rows *= extent
    elif len(outer) == 2 and list(vector.indices) == [outer[0], red_iv]:
        batches, rows = extents[0], extents[1]
    else:
        return None
    constants = ()
    if scales:
        scale = scales[0]
        scaled = _data_operands(scale).get("x")
        if (
            _memref_rank(acc) != 0
            or scaled is None
            or scaled.memref_name != acc.memref_name
            or scale.extra.get("store_indices") != outer
            or [loop[0] for loop in scale.enclosing_loops] != outer
        ):
            return None
        out = scale.result_memref_name
        constants = tuple(sorted(scale.extra.get("constants", {}).items()))
        for name, value in constants:
            _require_int32_constant("matrix_vector", name, value)
        _require_element_type("matrix_vector", "acc", acc)
    else:
        if list(acc.indices) != outer or acc.memref_name != match.result_memref_name:
            return None
        out = acc.memref_name
        _require_element_type("matrix_vector", "out", acc)
    _require_element_type("matrix_vector", "matrix", matrix)
    _require_element_type("matrix_vector", "vector", vector)
    return {
        "geometry": (("batches", batches), ("rows", rows), ("columns", extents[-1])),
        "anchor_op": "MAC",
        "constants": constants,
        "roles": (
            ("matrix", matrix.memref_name),
            ("vector", vector.memref_name),
            ("out", out),
        ),
        # A scaled accumulator is a local scalar the matcher proves zeroed.
        "accumulator_init": _accumulator_init("matrix_vector", match),
    }


def _matrix_vector_problem(kernel: UpmemKernel) -> UPMEMPhysicalProblem:
    batches = kernel.geometry_value("batches")
    return UPMEMPhysicalProblem.matrix_vector(
        batches * kernel.geometry_value("rows"),
        kernel.geometry_value("columns"),
        batches=batches,
    )


def _matrix_vector_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    if (
        decision.layout is UPMEMDataLayout.FUSED_REPLICATED
        and decision.vector_residency is UPMEMOperandResidency.FUSED_DMA_PACKET
    ):
        vector_mode = "fused_replicated"
    elif (
        decision.layout is UPMEMDataLayout.SEPARATE
        and decision.vector_residency is UPMEMOperandResidency.TASKLET_PRIVATE_WRAM
    ):
        vector_mode = "tasklet_private_wram"
    else:
        raise UPMEMPlanBuilderRejected(
            "MV lowering does not materialize the requested residency"
        )
    return UPMEMMatrixVectorPlan(
        rows=kernel.geometry_value("rows"),
        columns=kernel.geometry_value("columns"),
        batches=kernel.geometry_value("batches"),
        scale_factor=int(dict(kernel.constants).get("alpha", 1)),
        dma_bytes=decision.chunk_elements * 4,
        vector_mode=vector_mode,
        carried_output=kernel.accumulator_init == "carried",
    )


# The campaign's curated MV list, ``_matrix_vector_search_decisions`` in
# ``scripts/prepare_upmem_tenon_campaign.py``.
_MATRIX_VECTOR_DOMAIN = (
    UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        64,
        16,
        vector_residency=UPMEMOperandResidency.SHARED_WRAM,
    ),
    UPMEMPhysicalDecision(
        UPMEMDataLayout.FUSED_REPLICATED,
        128,
        16,
        vector_residency=UPMEMOperandResidency.FUSED_DMA_PACKET,
    ),
    UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        64,
        16,
        vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
    ),
)

register_upmem_family(
    "matrix_vector",
    _recognize_matrix_vector,
    _matrix_vector_plan,
    _matrix_vector_problem,
    _MATRIX_VECTOR_DOMAIN,
    uneven=("rows", "batches"),
    carried=True,
)


# -------------------------------- gemm --------------------------------- #


def _recognize_gemm(target, stream):
    """``C[i, j] += A[i, k] * B[k, j]`` as one MAC over an (i, j, k) nest."""
    del target
    if len(stream) != 1 or stream[0].target_op_name != "MAC":
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 3:
        return None
    i, j, k = (loop[0] for loop in match.enclosing_loops)
    operands = _data_operands(match)
    acc = operands.get("acc")
    inputs = [operands.get("x"), operands.get("y")]
    if acc is None or not acc.is_loop_carried or None in inputs:
        return None
    if list(acc.indices) != [i, j] or acc.memref_name != match.result_memref_name:
        return None
    lhs = next((op for op in inputs if list(op.indices) == [i, k]), None)
    rhs = next((op for op in inputs if list(op.indices) == [k, j]), None)
    if lhs is None or rhs is None or lhs is rhs:
        return None
    _require_element_type("gemm", "lhs", lhs)
    _require_element_type("gemm", "rhs", rhs)
    _require_element_type("gemm", "out", acc)
    rows, columns, reduction = extents
    return {
        "geometry": (("rows", rows), ("columns", columns), ("reduction", reduction)),
        "anchor_op": "MAC",
        "constants": (),
        "roles": (
            ("lhs", lhs.memref_name),
            ("rhs", rhs.memref_name),
            ("out", acc.memref_name),
        ),
        "accumulator_init": _accumulator_init("gemm", match),
    }


def _gemm_problem(kernel: UpmemKernel) -> UPMEMPhysicalProblem:
    return UPMEMPhysicalProblem.gemm(
        kernel.geometry_value("rows"),
        kernel.geometry_value("columns"),
        kernel.geometry_value("reduction"),
    )


def _gemm_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    if (
        decision.layout is not UPMEMDataLayout.FUSED_REPLICATED
        or decision.rhs_residency is not UPMEMOperandResidency.FUSED_DMA_PACKET
    ):
        raise UPMEMPlanBuilderRejected(
            "current GEMM lowering materializes fused RHS packets only"
        )
    return UPMEMGEMMPlan(
        rows=kernel.geometry_value("rows"),
        columns=kernel.geometry_value("columns"),
        reduction=kernel.geometry_value("reduction"),
        column_tile=int(decision.nc),
        reduction_tile=int(decision.kc),
        carried_output=kernel.accumulator_init == "carried",
    )


# The campaign's curated list, ``_gemm_search_decisions``.
_GEMM_DOMAIN = tuple(
    UPMEMPhysicalDecision(
        layout,
        dma_bytes,
        chunk,
        rhs_residency=residency,
        nc=nc,
        kc=kc,
    )
    for layout, dma_bytes, chunk, residency, nc, kc in (
        (
            UPMEMDataLayout.FUSED_REPLICATED,
            320,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
            4,
            16,
        ),
        (
            UPMEMDataLayout.FUSED_REPLICATED,
            576,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
            8,
            16,
        ),
        (
            UPMEMDataLayout.FUSED_REPLICATED,
            1088,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
            16,
            16,
        ),
        (
            UPMEMDataLayout.FUSED_REPLICATED,
            1056,
            8,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
            32,
            8,
        ),
        (
            UPMEMDataLayout.SEPARATE,
            2048,
            64,
            UPMEMOperandResidency.SHARED_WRAM,
            128,
            64,
        ),
    )
)

register_upmem_family(
    "gemm",
    _recognize_gemm,
    _gemm_plan,
    _gemm_problem,
    _GEMM_DOMAIN,
    uneven=("rows",),
    carried=True,
)


# =============================== spec 004 =============================== #
# Irregular families. Each realizes only the shapes its calibrated plan
# computes (spec 004 D4); any other matched shape raises NotImplementedError
# naming the value.


def _engine():
    from . import spmw_match_engine

    return spmw_match_engine


def _memref_shape(operand) -> tuple:
    match = re.fullmatch(r"memref<((?:\d+x)*)[a-z]+\d*>", operand.memref_type or "")
    if match is None:
        return ()
    return tuple(int(v) for v in match.group(1).split("x") if v)


def _only(stream, *names):
    return sorted(m.target_op_name for m in stream) == sorted(names)


def _guards(match) -> tuple:
    return tuple(match.extra.get("guards", ()))


def _is_const(term, value) -> bool:
    return _engine()._const_value(term) == value


def _load_is(term, operand) -> bool:
    E = _engine()
    return (
        isinstance(term, E.WLoad)
        and term.memref_name == operand.memref_name
        and list(term.indices) == list(operand.indices)
    )


def _cmp_parts(term):
    """``(pred, lhs, rhs)`` of a compare, plus its mirrored form."""
    E = _engine()
    if not isinstance(term, E.WCmp):
        return ()
    mirrored = E._MIRRORED_PREDICATE[term.pred]
    return ((term.pred, term.lhs, term.rhs), (mirrored, term.rhs, term.lhs))


# ------------------------------ histogram ------------------------------ #


def _histogram_index(term):
    """``(x_load, m, k)`` for ``(x * m) >> k``, else None."""
    E = _engine()
    if not (isinstance(term, E.WBinOp) and term.op == "shr"):
        return None
    k = E._const_value(term.rhs)
    product = term.lhs
    if k is None or not (isinstance(product, E.WBinOp) and product.op == "mul"):
        return None
    for value, factor in ((product.lhs, product.rhs), (product.rhs, product.lhs)):
        m = E._const_value(factor)
        if isinstance(value, E.WLoad) and m is not None:
            return value, m, k
    return None


def _histogram_guard_ok(guards, index, bins) -> bool:
    """Empty, or exactly ``0 <= index < bins`` (mirrors accepted)."""
    E = _engine()
    if not guards:
        return True
    if len(guards) != 1 or guards[0][0] != "if":
        return False
    cond = guards[0][1]
    if not (isinstance(cond, E.WBinOp) and cond.op == "and"):
        return False
    need = {("ge", 0), ("lt", bins)}
    seen = set()
    for side in (cond.lhs, cond.rhs):
        for pred, lhs, rhs in _cmp_parts(side):
            value = E._const_value(rhs)
            if E._term_eq(lhs, index) and (pred, value) in need:
                seen.add((pred, value))
                break
    return seen == need


def _recognize_histogram(target, stream):
    """``H[(x * m) >> k] += 1``, unguarded or guarded by ``0 <= bin < m``.

    An unguarded source has undefined out-of-range behavior in C; the plan's
    in-range guard refines it.
    """
    del target
    if not _only(stream, "INC"):
        return None
    match = stream[0]
    terms = match.extra.get("index_terms") or ()
    extents = _loop_extents(match)
    if len(terms) != 1 or extents is None or len(extents) != 1:
        return None
    E = _engine()
    term = terms[0]
    unproven_truncation = None
    if isinstance(term, E.WUnary) and term.fn.startswith(
        ("arith.trunci<", "arith.index_cast<")
    ):
        # The matcher kept a bin-index truncation it could not prove exact.
        # Recognize the shape so the element-type check names the real
        # problem for a non-int32 input; an int32 input whose index may
        # still truncate is refused below.
        unproven_truncation = term.fn
        term = term.arg
    parts = _histogram_index(term)
    if parts is None:
        return None
    x, bins, depth = parts
    (iv,) = (loop[0] for loop in match.enclosing_loops)
    if list(x.indices) != [iv]:
        return None
    (acc,) = [op for op in match.operands if op.is_loop_carried]
    _require_element_type("histogram", "input", x)
    _require_element_type("histogram", "out", acc)
    if unproven_truncation is not None:
        raise NotImplementedError(
            f"UPMEM histogram: the bin index ({x.memref_name} * {bins}) >> {depth} is "
            f"narrowed by {unproven_truncation}, which is not provably exact; the "
            "device computes the index in int32"
        )
    if _memref_shape(acc) != (bins,):
        raise NotImplementedError(
            f"histogram multiplier {bins} differs from the bin count "
            f"{_memref_shape(acc)} of {acc.memref_name}"
        )
    if not _histogram_guard_ok(_guards(match), terms[0], bins):
        raise NotImplementedError(
            "histogram guard is not exactly 0 <= bin < bins; the plan cannot "
            "realize it"
        )
    return {
        "geometry": (("elements", extents[0]), ("bins", bins), ("depth", depth)),
        "anchor_op": "INC",
        "constants": (),
        "roles": (("input", x.memref_name), ("out", acc.memref_name)),
        "accumulator_init": _accumulator_init("histogram", match),
    }


def _histogram_plan(kernel, decision):
    # The plan has one DMA policy (128-element reads); the decision's
    # dma/chunk fields are the single catalog entry and are not consulted.
    del decision
    return UPMEMHistogramPlan(
        kernel.geometry_value("elements"),
        kernel.geometry_value("bins"),
        kernel.geometry_value("depth"),
        carried_output=kernel.accumulator_init == "carried",
    )


# hist was not in the physical search: one decision, the plan defaults.
_HISTOGRAM_DOMAIN = (UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 512, 128),)

# Uneven work items are not padding-neutral here: a zero-padded element
# would be counted in bin (0 * m) >> k.
register_upmem_family(
    "histogram",
    _recognize_histogram,
    _histogram_plan,
    None,
    _HISTOGRAM_DOMAIN,
    carried=True,
)


# ------------------------------ selection ------------------------------ #


def _selection_predicate(guards):
    """``(tested_term, predicate)`` for a single ``if`` guard comparing
    ``x`` or ``x & mask`` against an int32 literal, else None.

    ``predicate`` is ``"odd"`` for ``x & 1 != 0`` / ``x & 1 == 1`` (the
    calibrated kernel), otherwise a :class:`SelectionPredicate`.
    """
    E = _engine()
    if len(guards) != 1 or guards[0][0] != "if":
        return None
    cond = guards[0][1]
    if not isinstance(cond, E.WCmp):
        return None
    unsigned = str(getattr(cond, "raw", "")).startswith("u")
    for pred, lhs, rhs in _cmp_parts(cond):
        value = E._const_value(rhs)
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        tested, mask = lhs, None
        if isinstance(lhs, E.WBinOp) and lhs.op == "and":
            for x, literal in ((lhs.lhs, lhs.rhs), (lhs.rhs, lhs.lhs)):
                if isinstance(E._const_value(literal), int):
                    tested, mask = x, E._const_value(literal)
                    break
            else:
                continue
        if not isinstance(tested, E.WLoad):
            continue
        if mask == 1 and (pred, value) in (("ne", 0), ("eq", 1)):
            return tested, "odd"
        try:
            return tested, SelectionPredicate(pred, value, mask, unsigned)
        except ValueError:
            return None
    return None


def _recognize_selection(target, stream):
    """Odd-value selection, compaction form (``COMPACT`` + ``INC`` under one
    odd guard) or flags form (one ``SELECT_ODD``)."""
    del target
    E = _engine()
    if _only(stream, "SELECT_ODD"):
        match = stream[0]
        extents = _loop_extents(match)
        if extents is None or len(extents) != 1:
            return None
        (iv,) = (loop[0] for loop in match.enclosing_loops)
        x = _data_operands(match).get("x")
        if x is None or list(x.indices) != [iv]:
            return None
        _require_element_type("selection", "input", x)
        return {
            "geometry": (("elements", extents[0]), ("compact", 0)),
            "anchor_op": "SELECT_ODD",
            "constants": (),
            "roles": (("input", x.memref_name), ("out", match.result_memref_name)),
        }
    if not _only(stream, "COMPACT", "INC"):
        return None
    compact = next(m for m in stream if m.target_op_name == "COMPACT")
    counter = next(m for m in stream if m.target_op_name == "INC")
    extents = _loop_extents(compact)
    if extents is None or len(extents) != 1:
        return None
    (iv,) = (loop[0] for loop in compact.enclosing_loops)
    value = _data_operands(compact).get("x")
    if value is None or list(value.indices) != [iv]:
        return None
    _require_element_type("selection", "input", value)
    guards = _guards(compact)
    if len(guards) != 1 or len(_guards(counter)) != 1:
        return None
    if guards[0][0] != _guards(counter)[0][0] or not E._term_eq(
        guards[0][1], _guards(counter)[0][1]
    ):
        return None
    found = _selection_predicate(guards)
    if found is None:
        raise NotImplementedError(
            "UPMEM selection: the guard is not a comparison of the copied value "
            "(or the value & an int32 mask) against an int32 literal; the "
            "selection plan supports '(x & mask) cmp k' and 'x cmp k' with cmp "
            "in ==, !=, <, <=, >, >="
        )
    tested, predicate = found
    if not _load_is(tested, value):
        raise NotImplementedError(
            "selection guard tests a value other than the copied one"
        )
    terms = compact.extra.get("index_terms") or ()
    cell = counter.result_memref_name
    if not (
        len(terms) == 1
        and isinstance(terms[0], E.WLoad)
        and terms[0].memref_name == cell
    ):
        return None
    return {
        "geometry": (("elements", extents[0]), ("compact", 1)),
        "anchor_op": "COMPACT",
        "constants": (("predicate", _predicate_constant(predicate)),),
        "roles": (
            ("input", value.memref_name),
            ("out", compact.result_memref_name),
            ("count", cell),
        ),
        # A carried count appends after the caller's first ``count`` outputs;
        # the host compaction step implements it.
        "accumulator_init": _accumulator_init("selection", counter),
    }


def _predicate_constant(predicate) -> tuple:
    if predicate == "odd":
        return ("odd",)
    return (predicate.compare, predicate.value, predicate.mask, predicate.unsigned)


def _kernel_predicate(kernel):
    spec = dict(kernel.constants).get("predicate", ("odd",))
    return "odd" if tuple(spec) == ("odd",) else SelectionPredicate(*spec)


def _selection_problem(kernel):
    return UPMEMPhysicalProblem.selection(kernel.geometry_value("elements"))


def _selection_plan(kernel, decision):
    if decision.predicate_lowering is not UPMEMPredicateLowering.CONDITIONAL_ZERO:
        raise UPMEMPlanBuilderRejected(
            "the selection-flags template materializes the conditional-zero "
            "lowering only"
        )
    return UPMEMSelectionFlagsPlan(
        selection_padded_elements(
            kernel.geometry_value("elements"),
            decision.tasklets,
            decision.chunk_elements,
        ),
        num_tasklets=decision.tasklets,
        dma_elements=decision.chunk_elements,
        predicate=_kernel_predicate(kernel),
    )


# The campaign's curated list, ``_selection_search_decisions``.
_SELECTION_DOMAIN = tuple(
    UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        elements * 4,
        elements,
        predicate_lowering=lowering,
    )
    for elements, lowering in (
        (32, UPMEMPredicateLowering.TERNARY),
        (32, UPMEMPredicateLowering.BRANCHLESS_MASK),
        (32, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        (16, UPMEMPredicateLowering.TERNARY),
        (64, UPMEMPredicateLowering.TERNARY),
        (128, UPMEMPredicateLowering.TERNARY),
        (8, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        (16, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        (64, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        (16, UPMEMPredicateLowering.BRANCHLESS_MASK),
    )
)

register_upmem_family(
    "selection",
    _recognize_selection,
    _selection_plan,
    _selection_problem,
    _SELECTION_DOMAIN,
    uneven=("elements",),
    carried=True,
)


# -------------------------- kmeans_distances --------------------------- #


def _kmeans_max_abs(dimension: int) -> int:
    """Largest input bound for which the template's int32 accumulator is
    exact: ``dimension * (2 v)**2 <= 2**31 - 1``."""
    return math.isqrt((2**31 - 1) // dimension) // 2


def _recognize_kmeans_distances(target, stream):
    """``D[p, c] += (P[p, d] - C[c, d]) * (P[p, d] - C[c, d])`` over ``(p, c, d)``."""
    del target
    if not _only(stream, "SQDIST_ACC"):
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 3:
        return None
    p, c, d = (loop[0] for loop in match.enclosing_loops)
    operands = _data_operands(match)
    x, y, acc = operands.get("x"), operands.get("y"), operands.get("acc")
    if None in (x, y, acc):
        return None
    if (
        list(x.indices) != [p, d]
        or list(y.indices) != [c, d]
        or list(acc.indices) != [p, c]
    ):
        return None
    _require_element_type("kmeans_distances", "points", x)
    _require_element_type("kmeans_distances", "centroids", y)
    _require_element_type("kmeans_distances", "out", acc)
    points, clusters, dimension = extents
    return {
        "geometry": (
            ("points", points),
            ("dimension", dimension),
            ("clusters", clusters),
        ),
        "anchor_op": "SQDIST_ACC",
        "constants": (),
        "roles": (
            ("points", x.memref_name),
            ("centroids", y.memref_name),
            ("out", acc.memref_name),
        ),
        "accumulator_init": _accumulator_init("kmeans_distances", match),
    }


def _kmeans_distances_plan(kernel, decision):
    # One fused-packet policy; the single catalog decision is not consulted.
    del decision
    dimension = kernel.geometry_value("dimension")
    return UPMEMKMeansDistancesPlan(
        kernel.geometry_value("points"),
        dimension,
        kernel.geometry_value("clusters"),
        max_abs_value=_kmeans_max_abs(dimension),
    )


_SINGLE_DOMAIN = (UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 64, 16),)

register_upmem_family(
    "kmeans_distances",
    _recognize_kmeans_distances,
    _kmeans_distances_plan,
    None,
    _SINGLE_DOMAIN,
)


# ---------------------------- kmeans_update ---------------------------- #


def _recognize_kmeans_update(target, stream):
    """One k-means centroid update, placed on the host (the paper partition).

    The matched part must be exactly: per point ``p`` an argmin over the
    clusters of ``D[p, c]`` (``MIN_SEL`` + ``ARGMIN_IDX``, the running
    minimum seeded by a ``COMPACT`` copy of ``D[p, 0]``), then
    ``counts[idx] += 1`` (``INC``) and ``sums[idx, d] += P[p, d]``
    (``SCATTER_ADD``); the only other matches are ``COMPACT`` copies of
    ``sums``/``counts`` elements (the division's operands).  The per-cluster
    division has no target op: ``_check_kmeans_update_epilogue`` proves on
    the IR that the remaining output is computed only from ``sums`` and
    ``counts`` with a division.  The host runs the group's own lowered MLIR
    and the cost program prices it (``costs/upmem.py::HOST_OP_CYCLES``).
    """
    del target
    names = sorted(m.target_op_name for m in stream)
    if not {"MIN_SEL", "ARGMIN_IDX", "SCATTER_ADD", "INC"} <= set(names):
        return None

    def reject(reason):
        raise NotImplementedError(
            f"UPMEM kmeans_update (host-placed centroid update): {reason}; the "
            "host partition implements exactly an argmin over D[p, c], "
            "counts[idx] += 1, sums[idx, d] += P[p, d], then per-cluster "
            "division of sums by counts"
        )

    def single(name):
        found = [m for m in stream if m.target_op_name == name]
        if len(found) != 1:
            reject(f"expected one {name}, found {len(found)}")
        return found[0]

    extra_ops = sorted(
        set(names) - {"MIN_SEL", "ARGMIN_IDX", "SCATTER_ADD", "INC", "COMPACT"}
    )
    if extra_ops:
        reject(f"the kernel also computes {extra_ops}")
    argmin, minimum = single("ARGMIN_IDX"), single("MIN_SEL")
    counter, scatter = single("INC"), single("SCATTER_ADD")
    distances = _data_operands(argmin).get("x")
    if distances is None or _data_operands(minimum).get("x") is None:
        reject("the argmin does not read a distance matrix")
    if _data_operands(minimum)["x"].memref_name != distances.memref_name:
        reject("MIN_SEL and ARGMIN_IDX read different distance matrices")
    loops = [loop[0] for loop in argmin.enclosing_loops]
    extents = _loop_extents(argmin)
    if (
        extents is None
        or len(loops) != 2
        or list(distances.indices) != loops
        or _memref_shape(distances) != tuple(extents)
    ):
        reject("the argmin is not over D[p, c] for every point and cluster")
    points, clusters = extents
    index_name = argmin.result_memref_name
    count_terms = counter.extra.get("index_terms") or ()
    counts = next((op for op in counter.operands if op.is_loop_carried), None)
    if (
        counts is None
        or _memref_shape(counts) != (clusters,)
        or len(count_terms) != 1
        or getattr(count_terms[0], "memref_name", None) != index_name
    ):
        reject("the counter is not counts[idx] += 1 over the argmin index")
    point_values = _data_operands(scatter).get("x")
    sums = _data_operands(scatter).get("acc")
    scatter_terms = scatter.extra.get("index_terms") or ()
    if (
        point_values is None
        or sums is None
        or len(_memref_shape(point_values)) != 2
        or _memref_shape(point_values)[0] != points
        or _memref_shape(sums) != (clusters, _memref_shape(point_values)[1])
        or not scatter_terms
        or getattr(scatter_terms[0], "memref_name", None) != index_name
    ):
        reject("the scatter is not sums[idx, d] += P[p, d]")
    dimension = _memref_shape(point_values)[1]
    copy_sources = {distances.memref_name, sums.memref_name, counts.memref_name}
    for match in stream:
        if match.target_op_name != "COMPACT":
            continue
        source = _data_operands(match).get("x")
        if source is None or source.memref_name not in copy_sources:
            reject(f"COMPACT into {match.result_memref_name!r} copies another value")
    for match in (counter,):
        if match.extra.get("accumulator_init") != "zero":
            reject("counts is not zero-initialized before the update")
    return {
        "geometry": (
            ("points", points),
            ("clusters", clusters),
            ("dimension", dimension),
        ),
        "anchor_op": "ARGMIN_IDX",
        "constants": (),
        "roles": (
            ("distances", distances.memref_name),
            ("points", point_values.memref_name),
            ("sums", sums.memref_name),
            ("counts", counts.memref_name),
        ),
        "placement": "host",
    }


register_upmem_family(
    "kmeans_update",
    _recognize_kmeans_update,
    lambda kernel, decision: None,
    None,
    _SINGLE_DOMAIN,
)


# --------------------------- feature_gradient -------------------------- #

_CONST_COLUMN = re.compile(r"->\s*\((?P<results>.*)\)>$")


def _constant_last_column(operand) -> int | None:
    """``L`` for an access ``S[s, L]`` read through an affine map."""
    match = _CONST_COLUMN.search(getattr(operand, "index_map", None) or "")
    if match is None:
        return None
    last = match.group("results").split(",")[-1].strip()
    return int(last) if last.lstrip("-").isdigit() else None


def _recognize_feature_gradient(target, stream):
    """``G[f] += grad(S[s, f], S[s, L])`` with ``L == F`` (packed records)."""
    del target
    if not (_only(stream, "GRAD_LINEAR") or _only(stream, "GRAD_LOGISTIC")):
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 2:
        return None
    loops = {loop[0]: extent for loop, extent in zip(match.enclosing_loops, extents)}
    operands = _data_operands(match)
    x, y, acc = operands.get("x"), operands.get("y"), operands.get("acc")
    if None in (x, y, acc) or len(acc.indices) != 1 or len(x.indices) != 2:
        return None
    s_iv, f_iv = x.indices
    if (
        acc.indices[0] != f_iv
        or list(y.indices) != [s_iv]
        or set(loops) != {s_iv, f_iv}
    ):
        return None
    _require_element_type("feature_gradient", "samples", x)
    _require_element_type("feature_gradient", "out", acc)
    features, samples = loops[f_iv], loops[s_iv]
    label = _constant_last_column(y)
    if label != features or _memref_shape(x) != (samples, features + 1):
        raise NotImplementedError(
            f"feature gradient reads label column {label} of {x.memref_name} "
            f"{_memref_shape(x)}; the plan needs packed [features..., label] records"
        )
    constants = dict(match.extra.get("constants", {}))
    alpha, beta = constants.get("alpha"), constants.get("beta")
    if match.target_op_name == "GRAD_LINEAR":
        shift = (
            (-alpha).bit_length() - 1 if isinstance(alpha, int) and alpha < 0 else -1
        )
        if not (
            0 <= shift <= 30
            and alpha == -(2**shift)
            and isinstance(beta, int)
            and 0 <= beta <= 62
        ):
            raise NotImplementedError(
                f"linear gradient coefficients alpha={alpha}, beta={beta} are not "
                "-(2**shift) and an overflow shift in 0..62"
            )
        formula, overflow = 0, beta
    else:
        if (alpha, beta) != (1, 2):
            raise NotImplementedError(
                f"logistic gradient coefficients ({alpha}, {beta}) differ from (1, 2)"
            )
        formula, shift, overflow = 1, 5, 8
    return {
        "geometry": (
            ("samples", samples),
            ("features", features),
            ("formula", formula),
            ("shift", shift),
            ("overflow_shift", overflow),
        ),
        "anchor_op": match.target_op_name,
        "constants": tuple(sorted(constants.items())),
        "roles": (("samples", x.memref_name), ("out", acc.memref_name)),
        "accumulator_init": _accumulator_init("feature_gradient", match),
    }


def _feature_gradient_plan(kernel, decision):
    # One record-streaming policy; the single catalog decision is not consulted.
    del decision
    formula = (
        GradientFormula.LINEAR_FIXED_POINT
        if kernel.geometry_value("formula") == 0
        else GradientFormula.LOGISTIC_ZERO
    )
    return UPMEMFeatureGradientPlan(
        kernel.geometry_value("samples"),
        kernel.geometry_value("features"),
        formula,
        shift=kernel.geometry_value("shift"),
        overflow_shift=kernel.geometry_value("overflow_shift"),
    )


register_upmem_family(
    "feature_gradient",
    _recognize_feature_gradient,
    _feature_gradient_plan,
    None,
    _SINGLE_DOMAIN,
)


# --------------------------------------------------------------------- #
# Decision <-> placement extra
# --------------------------------------------------------------------- #


def _residency(decision: UPMEMPhysicalDecision) -> str:
    for residency in (decision.vector_residency, decision.rhs_residency):
        if residency is not UPMEMOperandResidency.NOT_APPLICABLE:
            return residency.value
    return UPMEMOperandResidency.NOT_APPLICABLE.value


def decision_fields(decision: UPMEMPhysicalDecision) -> dict:
    """Knob-shaped encoding of a decision (spec U5 value encoding)."""
    return {
        "data_layout": decision.layout.value,
        "tasklets": decision.tasklets,
        "dma_bytes": decision.dma_bytes,
        "chunk": decision.chunk_elements,
        "wram_residency": _residency(decision),
        "nc": decision.nc or 0,
        "kc": decision.kc or 0,
        "predicate_lowering": decision.predicate_lowering.value,
    }


def _kernel_extra(kernel: UpmemKernel) -> dict:
    return {
        "upmem_family": kernel.family,
        "upmem_geometry": kernel.geometry,
        "upmem_anchor_op": kernel.anchor_op,
        "upmem_constants": kernel.constants,
        "upmem_placement": kernel.placement,
        "upmem_accumulator_init": kernel.accumulator_init,
    }


def placement_extra(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> dict:
    return {**_kernel_extra(kernel), **decision_fields(decision)}


def decision_from_extra(family: UpmemFamily, extra: Mapping) -> UPMEMPhysicalDecision:
    """The family-domain decision whose encoding the placement carries."""
    if extra.get("upmem_family") != family.name:
        raise ValueError(
            f"placement carries no {family.name!r} UPMEM decision "
            f"(upmem_family={extra.get('upmem_family')!r})"
        )
    for decision in family.domain:
        fields = decision_fields(decision)
        if all(extra.get(name) == value for name, value in fields.items()):
            return decision
    carried = {name: extra.get(name) for name in decision_fields(family.domain[0])}
    raise ValueError(
        f"placement decision {carried} is not in the {family.name!r} decision domain"
    )


# The F2 object a UPMEM placement carries: the masked 12-of-16 map from a
# tasklet coordinate to the index of the contiguous block it owns.  It is
# independent of the extent, so the work items of an uneven partition share
# decision keys and are ranked as one group; the block size is the realized
# plan's (``_check_tasklet_ownership`` proves the plan factors through it).
UPMEM_TASKLET_CARRIER = masked_12_tasklet_linear_layout(1)


def _plan_block(plan) -> int | None:
    """Units one tasklet owns in a row/element-blocked plan, else None."""
    layout = getattr(plan, "tasklet_layout", None)
    return None if layout is None else int(layout.block_elements)


def _carrier_projection(layout) -> dict | None:
    """The tasklet carrier inside a placement layout: the ``tasklet``/``lane``
    bases restricted to the ``element`` output (the movement search may
    compose program dimensions onto the carrier), or None when those inputs
    also drive another output."""
    if "element" not in layout.out_dims or "tasklet" not in layout.bases:
        return None
    index = layout.out_dims.index("element")
    projected = {}
    for name in ("tasklet", "lane"):
        vectors = layout.bases.get(name, [])
        if any(
            value for vector in vectors for i, value in enumerate(vector) if i != index
        ):
            return None
        projected[name] = [(vector[index],) for vector in vectors]
    return projected


def _check_tasklet_ownership(plan, layout) -> None:
    """The emitted kernel's ``tid * BLOCK + local`` ownership must be the
    carried tasklet->block map scaled by the plan's block: the carrier's
    tasklet bases are the identity on 16 coordinates (12 active, the
    template's ``NR_TASKLETS``) and the plan starts tasklet ``t``'s block at
    ``t * BLOCK``."""
    carrier = UPMEM_TASKLET_CARRIER
    projected = _carrier_projection(layout)
    if projected != {name: carrier.bases.get(name, []) for name in ("tasklet", "lane")}:
        raise ValueError(
            "UPMEM placement layout does not contain the 12-of-16 tasklet carrier "
            "on its 'element' output"
        )
    block = _plan_block(plan)
    if block is None:
        return
    owned = plan.tasklet_layout
    for tasklet in range(UPMEM_ACTIVE_TASKLETS):
        start = carrier.apply(tasklet=tasklet, lane=0)[0] * block
        if owned.coordinate(tasklet, 0) != start:
            raise ValueError(
                f"UPMEM plan ownership of tasklet {tasklet} starts at "
                f"{owned.coordinate(tasklet, 0)}, not {start}"
            )


def kernel_from_extra(extra: Mapping) -> UpmemKernel:
    """The geometry-only kernel a placement's ``extra`` describes."""
    return UpmemKernel(
        family=extra["upmem_family"],
        group_id=0,
        geometry=tuple(extra["upmem_geometry"]),
        anchor_op=extra["upmem_anchor_op"],
        constants=tuple(extra.get("upmem_constants", ())),
        roles=(),
        dpus=1,
        placement=extra.get("upmem_placement", "dpu"),
        accumulator_init=extra.get("upmem_accumulator_init", "none"),
    )


@functools.lru_cache(maxsize=4096)
def _realize(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    """``(plan, None)`` when the family template realizes ``decision`` for
    ``kernel``, else ``(None, reason)``.

    Legality is the template's: the physical model's structural checks
    (decision fields that belong to the family, DMA sizes, WRAM) and the plan
    constructor's own preconditions (DMA bounds and alignment, tasklet split,
    capacities).  Divisibility of an extent is never a legality condition:
    the templates zero-pad and mask, and the features price the padded
    program.
    """
    family = upmem_family(kernel.family)
    try:
        if family.features is not None:
            physical_features(family.features(kernel), decision)
        return family.plan_for(kernel, decision), None
    except ValueError as error:  # UPMEMPhysicalLegalityError, UPMEMPlanBuilderRejected
        return None, str(error)


def _legal(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> bool:
    return _realize(kernel, decision)[1] is None


def _legal_decisions(kernel: UpmemKernel) -> list:
    family = upmem_family(kernel.family)
    return sorted(
        (d for d in family.domain if _legal(kernel, d)),
        key=lambda d: d.deterministic_order_key,
    )


def _no_legal_decision(kernel: UpmemKernel) -> NotImplementedError:
    family = upmem_family(kernel.family)
    reasons: dict[str, list] = {}
    for decision in family.domain:
        reason = _realize(kernel, decision)[1]
        fields = decision_fields(decision)
        label = f"{fields['data_layout']}/dma{fields['dma_bytes']}"
        reasons.setdefault(reason, []).append(label)
    detail = "; ".join(
        f"{', '.join(labels)}: {reason}" for reason, labels in reasons.items()
    )
    geometry = ", ".join(f"{name}={value}" for name, value in kernel.geometry)
    return NotImplementedError(
        f"UPMEM {kernel.family} ({geometry}): no decision of the family's "
        f"catalog is realizable by its template: {detail}"
    )


def upmem_base_placements(target, matches) -> list[Placement]:
    """One base placement per data layout with a legal decision (spec U5)."""
    kernel = classify_upmem_kernel(target, matches)
    legal = _legal_decisions(kernel)
    if not legal:
        raise _no_legal_decision(kernel)
    layouts = sorted({d.layout.value for d in legal})
    return [
        Placement(
            placements={},
            mode=f"upmem:{kernel.family}:{layout}",
            extra={**_kernel_extra(kernel), "data_layout": layout},
        )
        for layout in layouts
    ]


# --------------------------------------------------------------------- #
# Candidate effects (the shared candidate contract, release gate G9)
# --------------------------------------------------------------------- #

# Residencies under which the template keeps the operand in WRAM for the
# whole launch (a transient allocation); the others stream it from MRAM or
# fuse it into the matrix rows' DMA packets.
_WRAM_RESIDENCIES = frozenset(
    {
        UPMEMOperandResidency.SHARED_WRAM.value,
        UPMEMOperandResidency.TASKLET_PRIVATE_WRAM.value,
    }
)
# The role the family's residency decision applies to.
_RESIDENT_ROLE = {"matrix_vector": "vector", "gemm": "rhs"}
# The role a carried accumulator (``accumulator_init == "carried"``) lives
# in: the output, except the selection count (carried by the host
# compaction step, never loaded by the template).
_CARRIED_ROLE = {"selection": "count"}


def upmem_family_extents(kernel: UpmemKernel, geometry=None) -> tuple:
    """``(result instances, reduction extent)`` the family template computes
    for ``geometry`` (the kernel's by default): what one launch of the
    template advances, in the shared contract's terms (a contraction's
    result trips and reduction trips; an accumulating elementwise op's
    iterations, no reduction).  The realization check compares these with
    the selected candidate's declared launch."""
    g = dict(kernel.geometry if geometry is None else geometry)
    family = kernel.family
    if family in ("pointwise", "reduction", "selection"):
        return int(g["elements"]), None
    if family == "matrix_vector":
        return int(g["batches"]) * int(g["rows"]), int(g["columns"])
    if family == "gemm":
        return int(g["rows"]) * int(g["columns"]), int(g["reduction"])
    if family == "histogram":
        return int(g["elements"]), None
    if family == "kmeans_distances":
        return int(g["points"]) * int(g["clusters"]), int(g["dimension"])
    if family == "feature_gradient":
        return int(g["features"]), int(g["samples"])
    raise NotImplementedError(
        f"UPMEM family {family!r} states no launch extents for the realization check"
    )


def upmem_candidate_effects(target, matches, placement):
    """`CandidateEffects` of one UPMEM placement for one work item, derived
    from the shared semantic facts of its matches (`Contraction`/`Epilogue`:
    loop labels, trips, access coordinates, accumulator initialization,
    sinks) and from the placement's physical decision (data layout, WRAM
    residency) and family.

    A DPU group: every value's backing storage is the DPU's ``mram`` heap
    image the runtime packs from the caller's arrays (``host_pack``) and
    decodes the outputs from (``host_readback``); the input images carry the
    decision's data layout as their packing (``separate`` or
    ``fused_replicated``, the resident operand also its residency); the
    template's WRAM use is a transient allocation (its DMA buffers, and the
    resident operand under a WRAM residency).  A carried accumulator
    (``accumulator_init == "carried"``) makes the output ``inout``: the
    template loads the initial output.  One launch advances the work item's
    result instances over the whole reduction (the family's template; the
    DPUs of a group run in lockstep, one work item each).

    A host group (``placement == "host"``, the paper's k-means update
    partition) executes the work item's whole function on the host from its
    retained IR: ``host_function`` effects with a whole-function launch over
    the function's retained store effects and values in ``host_dram``.

    `UpmemCtx.check_realization` compares the realized plan, decision,
    segment kind, role windows and host steps with the declaration.  None
    when the matches carry no facts (a trace assembled outside the shared
    front stages).
    """
    from .spmw_candidate import (
        CandidateCoverage,
        CandidateEffects,
        CandidateLaunch,
        CandidateTransfer,
        CandidateValueLayout,
        indirect_reads,
        value_axes,
        verbatim_values,
    )

    if not matches:
        return None
    program = matches[0].extra.get("program_facts")
    facts = [m.extra.get("semantic_facts") for m in matches]
    if program is None:
        return None
    extra = placement.extra or {}
    family = extra.get("upmem_family")
    if family is None:
        return None
    function = matches[0].func_name
    operations = tuple(sorted({m.target_op_name for m in matches}))
    if extra.get("upmem_placement") == "host":
        if not getattr(program, "effects", ()):
            return None
        inputs, outputs, _locals = verbatim_values(program)
        layouts = tuple(
            CandidateValueLayout(name, (), "host_dram")
            for name in dict.fromkeys((*inputs, *outputs))
        )
        transfers = []
        for name in dict.fromkeys((*inputs, *outputs)):
            if name in inputs and name in outputs:
                transfers.append(
                    CandidateTransfer(
                        name, "inout", ("host_copy", "host_dram", "host_copy")
                    )
                )
            elif name in inputs:
                transfers.append(
                    CandidateTransfer(name, "in", ("host_copy", "host_dram"))
                )
            else:
                transfers.append(
                    CandidateTransfer(name, "out", ("host_dram", "host_copy"))
                )
        launch = CandidateLaunch(
            index=0,
            kind=f"upmem_host:{family}",
            operations=tuple(range(len(program.effects))),
            instances=(0, 1),
            whole=True,
            covers=(placement.mode, family, "host"),
        )
        return CandidateEffects(
            coverage=CandidateCoverage(function, operations, inputs, outputs),
            value_layouts=layouts,
            transfers=tuple(transfers),
            launches=(launch,),
            abi_inputs=inputs,
            abi_outputs=outputs,
            device_scope=(
                "host partition: the work item's function, lowered to C from its "
                "retained IR, runs on the host (UPMEM paper partition)"
            ),
            host_function=True,
        )
    if any(f is None for f in facts):
        return None
    data_layout = str(extra.get("data_layout") or "")
    residency = str(extra.get("wram_residency") or "not_applicable")
    resident_role = _RESIDENT_ROLE.get(family)
    resident_value = None
    if resident_role is not None:
        kernel = classify_upmem_kernel(target, matches)
        resident_value = dict(kernel.roles).get(resident_role)
    labels = {}
    for fact in facts:
        labels.update(fact.loop_labels)
    inputs, outputs, layouts, transfers = [], [], {}, {}
    instances = reduction = init = None
    for match, fact in zip(matches, facts):
        if hasattr(fact, "reduction_vars"):
            trips = fact.trips
            extent = red = 1
            for var in fact.result_vars:
                extent *= int(trips.get(var) or 0)
            for var in fact.reduction_vars:
                red *= int(trips.get(var) or 0)
            if instances is None:
                instances, init = extent, fact.accumulator_init
                # An accumulate without a reduction loop (a histogram's
                # ``H[bin] += 1`` per sample) sums nothing per instance.
                reduction = red if fact.reduction_vars else None
            accesses = [a for a in fact.inputs if not a.local]
            results = []
            if not fact.accumulator.local:
                results.append(fact.accumulator)
            results.extend(fact.external_sinks)
            carried = fact.accumulator_init == "carried"
        else:
            extent = 1
            for loop in fact.loops:
                extent *= int(loop.trip or 0)
            if instances is None:
                instances = extent
            own = match.extra.get("accumulator_init")
            # An operand reading the element it writes: an input only when
            # the source carries the incoming contents (the shared rule).
            accesses = [
                a
                for a in fact.operands
                if not a.local
                and not (a.memref_name == fact.result.memref_name and own == "zero")
            ]
            results = [] if fact.result.local else [fact.result]
            for memref, _text in match.extra.get("output_sinks", ()):
                if memref != fact.result.memref_name:
                    results.append(fact.result)
            carried = own == "carried"
        for access in accesses:
            memref = access.memref_name
            if memref is None or memref in inputs:
                continue
            inputs.append(memref)
            packing = data_layout or None
            if memref == resident_value:
                packing = f"{data_layout}:{residency}"
            layouts[(memref, "in")] = CandidateValueLayout(
                memref, value_axes(access, labels), "mram", packing, image="in"
            )
            transfers[memref] = CandidateTransfer(memref, "in", ("host_pack", "mram"))
        for access in results:
            memref = access.memref_name
            if memref is None:
                continue
            if memref not in outputs:
                outputs.append(memref)
            if carried and memref not in inputs:
                inputs.append(memref)
            both = memref in inputs
            if both and (memref, "in") not in layouts:
                # A carried accumulator: the template loads the caller's
                # output from the same MRAM window it writes back to.
                layouts[(memref, "in")] = CandidateValueLayout(
                    memref,
                    value_axes(access, labels),
                    "mram",
                    data_layout or None,
                    image="in",
                )
            layouts[(memref, "out" if both else "value")] = CandidateValueLayout(
                memref,
                value_axes(access, labels),
                "mram",
                data_layout or None,
                image="out" if both else "value",
            )
            transfers[memref] = CandidateTransfer(
                memref,
                "inout" if both else "out",
                (
                    ("host_pack", "mram", "host_readback")
                    if both
                    else ("mram", "host_readback")
                ),
            )
    # Inputs read only through a computed index or a guard (the histogram's
    # samples): packed like every other input.
    for memref, access in indirect_reads(matches).items():
        if memref in inputs:
            continue
        inputs.append(memref)
        layouts[(memref, "in")] = CandidateValueLayout(
            memref, value_axes(access, labels), "mram", data_layout or None, image="in"
        )
        transfers[memref] = CandidateTransfer(memref, "in", ("host_pack", "mram"))
    allocations = [("dma_buffers", 0, "wram")]
    if resident_value is not None and residency in _WRAM_RESIDENCIES:
        allocations.append((f"{resident_role}_wram", 0, "wram"))
    launch = CandidateLaunch(
        index=0,
        kind=f"upmem:{family}",
        operations=tuple(range(len(matches))),
        instances=(0, int(instances or 0)),
        reduction=None if reduction is None else (0, int(reduction)),
        accumulator=init,
        covers=(placement.mode, family, data_layout, residency),
    )
    return CandidateEffects(
        coverage=CandidateCoverage(function, operations, tuple(inputs), tuple(outputs)),
        value_layouts=tuple(layouts.values()),
        transfers=tuple(transfers.values()),
        launches=(launch,),
        abi_inputs=tuple(inputs),
        abi_outputs=tuple(outputs),
        device_scope=(
            "one work item per DPU, the family template's tasklets sharing the "
            "DPU; the group's DPUs run the same binary in lockstep"
        ),
        allocations=tuple(allocations),
    )


def _derive_upmem_effects(view, matches, placement):
    """Late-binding entry of the effects registry (the enumerator, the
    movement-stage re-declaration and a supplied-placement rebinding use
    the same derivation)."""
    return upmem_candidate_effects(view, matches, placement)


def _register_effects_deriver():
    from .spmw_candidate import EFFECTS_DERIVERS

    EFFECTS_DERIVERS["upmem"] = _derive_upmem_effects


_register_effects_deriver()


# Knob names in registration (cross) order, spec 001 D5.  ``tasklets`` only
# ever takes 12 (every catalog decision and template is a 12-tasklet plan: a
# schedule policy, not a hardware limit; the target declares 24); it stays a
# knob so a calibrated catalog with other tasklet counts needs no enumerator
# change.  Families with a one-decision catalog (reduction, histogram,
# kmeans_distances, feature_gradient, kmeans_update) carry their single
# decision's fields; their templates have one DMA policy each.
UPMEM_KNOBS = (
    "tasklets",
    "dma_bytes",
    "chunk",
    "wram_residency",
    "nc",
    "kc",
    "predicate_lowering",
)


def upmem_knob_candidates(name: str, extra: Mapping) -> list:
    """Values of knob ``name`` that some legal domain decision takes, given
    the data layout and the knob values already chosen in ``extra``."""
    if "upmem_family" not in extra:
        return []
    fixed = ("data_layout",) + UPMEM_KNOBS[: UPMEM_KNOBS.index(name)]
    values = []
    for decision in _legal_decisions(kernel_from_extra(extra)):
        fields = decision_fields(decision)
        if all(fields[key] == extra.get(key) for key in fixed):
            if fields[name] not in values:
                values.append(fields[name])
    return values


def upmem_knob_emit(name: str, value, base: Placement) -> Placement:
    """A copy of ``base`` with ``extra[name] = value``.  The last knob checks
    the decision is realizable and attaches the F2 tasklet carrier (which
    codegen checks against the emitted plan's ownership) and the decision's
    cost validity."""
    extra = dict(base.extra or {})
    extra[name] = value
    layout = base.layout
    if name == UPMEM_KNOBS[-1]:
        family = upmem_family(extra["upmem_family"])
        decision = decision_from_extra(family, extra)
        kernel = kernel_from_extra(extra)
        plan, reason = _realize(kernel, decision)
        if reason is not None:
            raise ValueError(f"UPMEM knob emitted an unrealizable decision: {reason}")
        del plan
        layout = UPMEM_TASKLET_CARRIER
        extra["upmem_cost_validity"] = upmem_cost_validity(kernel, decision)
    return Placement(
        placements=dict(base.placements or {}),
        mode=f"{base.mode}+{name}={value}",
        extra=extra,
        layout=layout,
    )


_FEATURE_COST_KEYS = {
    "mram_read_calls": "mram_read_calls",
    "mram_read_bytes": "mram_read_bytes",
}


def upmem_kernel_features(properties: Mapping) -> tuple | None:
    """``(name, int)`` pairs of ``UPMEMPhysicalFeatures`` for a placement, or
    None for a family the physical model does not cover (reduction).

    A carried accumulator adds the template's initial-output loads, taken
    from the plan's own accounting (carried minus zero-initialized).
    """
    family = upmem_family(properties["upmem_family"])
    if family.features is None:
        return None
    kernel = kernel_from_extra(properties)
    decision = decision_from_extra(family, properties)
    features = dict(physical_features(family.features(kernel), decision).manifest())
    plan = _realize(kernel, decision)[0]
    if kernel.accumulator_init == "carried" and getattr(plan, "carried_output", False):
        carried = plan.cost_features()
        zeroed = family.plan_for(
            dataclasses.replace(kernel, accumulator_init="zero"), decision
        ).cost_features()
        for feature, plan_key in _FEATURE_COST_KEYS.items():
            features[feature] += int(carried[plan_key]) - int(zeroed[plan_key])
    return tuple(features.items())


def upmem_cost_validity(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> dict:
    """Validity label of the cost the cost program assigns this decision.

    ``calibrated``: the physical model has anchors in the decision's
    calibration stratum and one of them has exactly this problem.
    ``extrapolated``: anything else (other shape or operation in a calibrated
    stratum, an uncalibrated stratum at factor 1.0, or a family priced by the
    per-op analytical rules), with the nearest calibrated point.
    """
    family = upmem_family(kernel.family)
    if kernel.placement == "host":
        return {
            "cost_validity": "extrapolated",
            "basis": "coarse host-operation count (costs/upmem.py HOST_OP_CYCLES)",
            "nearest_calibrated_point": None,
        }
    if family.features is None:
        return {
            "cost_validity": "extrapolated",
            "basis": "per-op analytical rules (costs/upmem.py); no calibration "
            f"anchors exist for the {kernel.family} family",
            "nearest_calibrated_point": None,
        }
    validity = dict(calibration_validity(family.features(kernel), decision))
    if (
        kernel.accumulator_init == "carried"
        and validity["cost_validity"] == "calibrated"
    ):
        validity["cost_validity"] = "extrapolated"
        validity["basis"] += "; carried-output loads are outside every anchor"
    return validity


# --------------------------------------------------------------------- #
# MLIR helpers: work-id functions, access windows, coverage
# --------------------------------------------------------------------- #


def _ir():
    from ._mlir import ir

    return ir


def _attr_text(attribute) -> str:
    ir = _ir()
    try:
        return ir.StringAttr(attribute).value
    except (TypeError, ValueError):
        return str(attribute).strip('"')


def _functions(module) -> dict:
    functions = {}
    for operation in module.body.operations:
        if operation.operation.name != "func.func":
            continue
        functions[_attr_text(operation.attributes["sym_name"])] = operation
    return functions


def _walk(block, out):
    for operation in block.operations:
        out.append(operation)
        for region in operation.regions:
            for nested in region.blocks:
                _walk(nested, out)
    return out


def _owner_name(value) -> str | None:
    ir = _ir()
    if ir.OpResult.isinstance(value):
        return ir.OpResult(value).owner.name
    return None


def _func_argument_number(function, value) -> int | None:
    ir = _ir()
    if not ir.BlockArgument.isinstance(value):
        return None
    argument = ir.BlockArgument(value)
    entry = function.regions[0].blocks[0]
    if argument.owner != entry:
        return None
    return argument.arg_number


def _eval_affine(expr, dims, symbols) -> int:
    ir = _ir()
    if ir.AffineDimExpr.isinstance(expr):
        return dims[ir.AffineDimExpr(expr).position]
    if ir.AffineSymbolExpr.isinstance(expr):
        return symbols[ir.AffineSymbolExpr(expr).position]
    if ir.AffineConstantExpr.isinstance(expr):
        return ir.AffineConstantExpr(expr).value
    for kind, combine in (
        (ir.AffineAddExpr, lambda a, b: a + b),
        (ir.AffineMulExpr, lambda a, b: a * b),
        (ir.AffineModExpr, lambda a, b: a % b),
        (ir.AffineFloorDivExpr, lambda a, b: a // b),
        (ir.AffineCeilDivExpr, lambda a, b: -((-a) // b)),
    ):
        if kind.isinstance(expr):
            binary = kind(expr)
            return combine(
                _eval_affine(binary.lhs, dims, symbols),
                _eval_affine(binary.rhs, dims, symbols),
            )
    raise NotImplementedError(f"UPMEM index analysis: unsupported affine expr {expr}")


def _affine_indices(operation, operands, env, ctx) -> list[int]:
    ir = _ir()
    affine_map = ir.AffineMapAttr(operation.attributes["map"]).value
    values = [_eval_index(value, env, ctx) for value in operands]
    dims, symbols = values[: affine_map.n_dims], values[affine_map.n_dims :]
    return [_eval_affine(result, dims, symbols) for result in affine_map.results]


_CASTS = {
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.index_castui",
}


def _eval_index(value, env, ctx) -> int:
    """Concrete integer value of an index SSA value under loop assignment ``env``."""
    ir = _ir()
    function, state = ctx
    if ir.BlockArgument.isinstance(value):
        name = _value_name(value, state)
        if name not in env:
            raise NotImplementedError(f"UPMEM index analysis: unbound value {name}")
        return env[name]
    if not ir.OpResult.isinstance(value):
        raise NotImplementedError("UPMEM index analysis: unsupported value")
    operation = ir.OpResult(value).owner
    name = operation.name
    if name == "arith.constant":
        return ir.IntegerAttr(operation.attributes["value"]).value
    if name in _CASTS:
        return _eval_index(operation.operands[0], env, ctx)
    if name in ("arith.addi", "arith.subi", "arith.muli"):
        lhs = _eval_index(operation.operands[0], env, ctx)
        rhs = _eval_index(operation.operands[1], env, ctx)
        return {
            "arith.addi": lhs + rhs,
            "arith.subi": lhs - rhs,
            "arith.muli": lhs * rhs,
        }[name]
    if name == "affine.apply":
        (result,) = _affine_indices(operation, list(operation.operands), env, ctx)
        return result
    if name in ("affine.load", "memref.load") and len(operation.operands) >= 1:
        memref = operation.operands[0]
        if _owner_name(memref) != "memref.alloc":
            raise NotImplementedError("UPMEM index analysis: index loaded from memory")
        stores = [
            op
            for op in _walk(function.regions[0].blocks[0], [])
            if op.name in ("affine.store", "memref.store") and op.operands[1] == memref
        ]
        if len(stores) != 1 or len(operation.operands) != 1:
            raise NotImplementedError(
                "UPMEM index analysis: index scalar is not a single-store scalar"
            )
        return _eval_index(stores[0].operands[0], env, ctx)
    raise NotImplementedError(f"UPMEM index analysis: unsupported index op {name}")


def _access_indices(operation, env, ctx) -> list[int]:
    if operation.name in ("affine.load", "affine.store"):
        first = 2 if operation.name == "affine.store" else 1
        return _affine_indices(operation, list(operation.operands)[first:], env, ctx)
    first = 2 if operation.name == "memref.store" else 1
    return [_eval_index(value, env, ctx) for value in list(operation.operands)[first:]]


@dataclass(frozen=True)
class _Window:
    """``array[arg][offset_d : offset_d + extent_d]`` for every dimension."""

    argument: int
    offsets: tuple
    extents: tuple

    def slice(self, array: np.ndarray) -> np.ndarray:
        return array[tuple(slice(o, o + e) for o, e in zip(self.offsets, self.extents))]


def _access_window(operation, function, state, loops, matched_indices) -> _Window:
    """Prove the access is ``offset (+ iv)`` per dimension and return its window.

    Each dimension is probed under concrete loop assignments: it must be a
    constant, or a constant plus exactly one loop variable with coefficient 1.
    Where the matcher reports a plain loop variable for a dimension, the probe
    must agree with it.
    """
    memref = operation.operands[1 if operation.name.endswith("store") else 0]
    argument = _func_argument_number(function, memref)
    if argument is None:
        raise NotImplementedError(
            "UPMEM family operand is not a kernel argument (region-local buffers "
            "are not lowered to DPU MRAM)"
        )
    names = [loop[0] for loop in loops]
    extents = {loop[0]: _static_extent(loop) for loop in loops}
    ctx = (function, state)
    base_env = {name: 0 for name in names}
    offsets = _access_indices(operation, base_env, ctx)
    drivers: list = [None] * len(offsets)
    for name in names:
        for value in (1, extents[name] - 1):
            got = _access_indices(operation, dict(base_env, **{name: value}), ctx)
            for dim, (index, offset) in enumerate(zip(got, offsets)):
                if index == offset:
                    continue
                if index != offset + value or drivers[dim] not in (None, name):
                    raise NotImplementedError(
                        f"UPMEM access dimension {dim} is not offset + one loop variable"
                    )
                drivers[dim] = name
    for first in names:
        for second in names:
            if first >= second:
                continue
            env = dict(base_env, **{first: 1, second: 1})
            got = _access_indices(operation, env, ctx)
            want = [
                offset + (1 if driver in (first, second) else 0)
                for offset, driver in zip(offsets, drivers)
            ]
            if got != want:
                raise NotImplementedError("UPMEM access is not affine in its loops")
    if len(matched_indices) == len(offsets):
        for dim, index in enumerate(matched_indices):
            if index in extents and drivers[dim] != index:
                raise NotImplementedError(
                    f"UPMEM access dimension {dim} disagrees with the matcher"
                )
    shape = list(map(int, memref.type.shape))
    window_extents = tuple(1 if d is None else extents[d] for d in drivers)
    for offset, extent, size in zip(offsets, window_extents, shape):
        if offset < 0 or offset + extent > size:
            raise NotImplementedError("UPMEM access window leaves its array")
    return _Window(argument, tuple(offsets), window_extents)


def _value_name(value, state) -> str:
    from .spmw_match_engine import _value_name as engine_value_name

    return engine_value_name(value, state)


def _store_handle(operation, state) -> str:
    return f"{operation.name}@{_value_name(operation.operands[0], state)}"


_ZERO_CASTS = _CASTS | {"arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp"}


def _is_zero_constant(value) -> bool:
    ir = _ir()
    while _owner_name(value) in _ZERO_CASTS:
        value = ir.OpResult(value).owner.operands[0]
    if _owner_name(value) != "arith.constant":
        return False
    attribute = ir.OpResult(value).owner.attributes["value"]
    try:
        return ir.IntegerAttr(attribute).value == 0
    except (TypeError, ValueError):
        return ir.FloatAttr(attribute).value == 0.0


_INDEX_ARITH = {
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.floordivsi",
    "arith.ceildivsi",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
}


_CONTROL_ARITH = {"arith.cmpi", "arith.andi", "arith.ori", "arith.xori"}


def _feeds_only_addresses(value, seen=None) -> bool:
    """True when every use of ``value`` ends in an index position or in an
    ``scf.if`` condition (the matcher records both: M5 guards, M6 index
    terms)."""
    seen = set() if seen is None else seen
    for use in value.uses:
        owner, position = use.owner, use.operand_number
        name = owner.name
        if name in ("arith.index_cast", "arith.index_castui", "affine.apply"):
            continue
        if name == "scf.if" and position == 0:
            continue
        if name in ("memref.load", "affine.load") and position >= 1:
            continue
        if name in ("memref.store", "affine.store") and position >= 2:
            continue
        if name in _INDEX_ARITH or name in _CONTROL_ARITH:
            key = id(owner)
            if key in seen:
                continue
            seen.add(key)
            if all(_feeds_only_addresses(result, seen) for result in owner.results):
                continue
        return False
    return True


def _address_only_scratch(function, alloc) -> bool:
    """A local scalar whose every load feeds only address arithmetic
    (ruling 013-R3), for example ``row0 = d * ROWS``."""
    loads = [
        op
        for op in _walk(function.regions[0].blocks[0], [])
        if op.name in ("affine.load", "memref.load") and op.operands[0] == alloc
    ]
    return all(_feeds_only_addresses(load.results[0]) for load in loads)


def _op_path(function, target_op) -> tuple | None:
    def visit(block, prefix):
        for index, op in enumerate(block.operations):
            path = prefix + (index,)
            if op == target_op:
                return path
            for region_index, region in enumerate(op.regions):
                for block_index, nested in enumerate(region.blocks):
                    found = visit(nested, path + (region_index, block_index))
                    if found is not None:
                        return found
        return None

    return visit(function.regions[0].blocks[0], ())


def _feeds_matched_or_address(value, handles, state, seen=None) -> bool:
    """Every use ends in an address, a guard, or the stored value of a
    matched store (through arithmetic and casts)."""
    seen = set() if seen is None else seen
    for use in value.uses:
        owner, position = use.owner, use.operand_number
        name = owner.name
        if name in ("affine.store", "memref.store") and position == 0:
            if _store_handle(owner, state) in handles:
                continue
            return False
        if name.startswith("arith.") or name.startswith("math."):
            key = id(owner)
            if key in seen:
                continue
            seen.add(key)
            if all(
                _feeds_matched_or_address(result, handles, state, seen)
                for result in owner.results
            ):
                continue
            return False
        if not _feeds_only_addresses_use(owner, position):
            return False
    return True


def _feeds_only_addresses_use(owner, position) -> bool:
    name = owner.name
    return (
        name in ("arith.index_cast", "arith.index_castui", "affine.apply")
        or (name == "scf.if" and position == 0)
        or (name in ("memref.load", "affine.load") and position >= 1)
        or (name in ("memref.store", "affine.store") and position >= 2)
    )


def _forwarded_scalar(function, alloc, handles, state) -> bool:
    """Ruling 013-R3 as amended for 016: one store that precedes every load
    (the matcher's M3 forwarding folds it), each load folded into a matched
    store or used only for addresses. Stores consuming it are checked on
    their own."""
    body = _walk(function.regions[0].blocks[0], [])
    stores = [
        op
        for op in body
        if op.name in ("affine.store", "memref.store") and op.operands[1] == alloc
    ]
    loads = [
        op
        for op in body
        if op.name in ("affine.load", "memref.load") and op.operands[0] == alloc
    ]
    if len(stores) != 1:
        return False
    store_path = _op_path(function, stores[0])
    depth = len(store_path) - 1
    for load in loads:
        path = _op_path(function, load)
        if not (
            len(path) > depth
            and path[:depth] == store_path[:depth]
            and store_path[depth] < path[depth]
        ):
            return False
        if not _feeds_matched_or_address(load.results[0], handles, state):
            return False
    return True


def _check_coverage(function, state, matches) -> None:
    """Fail closed on a store no match covers (spec U4 coverage check)."""
    handles = {match.op_range[1] for match in matches}
    accumulators = {
        operand.memref_name
        for match in matches
        for operand in match.operands
        if operand.is_loop_carried
        or (
            operand.memref_name is not None
            and operand.memref_name == match.result_memref_name
        )
    }
    results = {match.result_memref_name for match in matches}
    for operation in _walk(function.regions[0].blocks[0], []):
        if operation.name not in ("affine.store", "memref.store"):
            continue
        if _store_handle(operation, state) in handles:
            continue
        name = (
            _attr_text(operation.attributes["to"])
            if "to" in operation.attributes
            else "<unnamed>"
        )
        destination = operation.operands[1]
        if _func_argument_number(function, destination) is not None:
            if name in accumulators and _is_zero_constant(operation.operands[0]):
                continue
        elif _owner_name(destination) == "memref.alloc":
            if name in accumulators | results:
                continue
            if _address_only_scratch(function, destination):
                continue
            if _forwarded_scalar(function, destination, handles, state):
                continue
        raise NotImplementedError(
            f"store to {name} at {operation.location} is not covered by a "
            "matched UPMEM op"
        )


_DIVISIONS = {"arith.divsi", "arith.divui", "arith.floordivsi", "arith.ceildivsi"}


def _check_kmeans_update_epilogue(function, kernel) -> None:
    """The unmatched part of a host ``kmeans_update`` group is the
    per-cluster division: every store into a caller argument other than
    ``counts`` writes one ``(clusters, dimension)`` output, and each stored
    value is computed (through local scalars, arithmetic and selects) only
    from ``sums``/``counts`` elements and constants, with a division."""
    ir = _ir()
    roles = dict(kernel.roles)
    sums, counts = roles["sums"], roles["counts"]
    shape = (kernel.geometry_value("clusters"), kernel.geometry_value("dimension"))
    body = _walk(function.regions[0].blocks[0], [])
    stores = [op for op in body if op.name in ("affine.store", "memref.store")]

    def name_of(op, key):
        return _attr_text(op.attributes[key]) if key in op.attributes else None

    outputs = {}
    for op in stores:
        if _func_argument_number(function, op.operands[1]) is None:
            continue
        target = name_of(op, "to")
        if target == counts:
            continue
        outputs.setdefault(target, []).append(op)
    if len(outputs) != 1:
        raise NotImplementedError(
            "UPMEM kmeans_update: the host update must write exactly one "
            f"centroid output besides {counts!r}; it writes {sorted(outputs)}"
        )
    ((output, output_stores),) = outputs.items()
    if tuple(map(int, output_stores[0].operands[1].type.shape)) != shape:
        raise NotImplementedError(
            f"UPMEM kmeans_update: centroid output {output!r} is not "
            f"{shape[0]}x{shape[1]}"
        )
    local_stores = {}
    for op in stores:
        if _owner_name(op.operands[1]) in ("memref.alloc", "memref.alloca"):
            local_stores.setdefault(name_of(op, "to"), []).append(op)
    seen_locals: set = set()
    divisions = []

    def derived(value) -> bool:
        if ir.BlockArgument.isinstance(value):
            return False
        owner = ir.OpResult(value).owner
        op_name = owner.name
        if op_name == "arith.constant":
            return True
        if op_name in ("affine.load", "memref.load"):
            source = name_of(owner, "from")
            if source in (sums, counts):
                return True
            if source in local_stores:
                if source in seen_locals:
                    return True
                seen_locals.add(source)
                return all(derived(st.operands[0]) for st in local_stores[source])
            return False
        if op_name.startswith("arith."):
            if op_name in _DIVISIONS:
                divisions.append(op_name)
            return all(derived(operand) for operand in owner.operands)
        return False

    for op in output_stores:
        if not derived(op.operands[0]):
            raise NotImplementedError(
                f"UPMEM kmeans_update: a value stored to {output!r} is not "
                f"computed from {sums!r} and {counts!r} alone"
            )
    if not divisions:
        raise NotImplementedError(
            f"UPMEM kmeans_update: {output!r} is not divided by the counts"
        )


def _check_role_dtypes(kernel, function, name, windows) -> None:
    """Every role's ABI element type, with signedness, must be the family's.

    Recognition sees signless MLIR types only; ``uint32`` is ``i32`` there.
    """
    from .utils import get_func_inputs_outputs

    inputs, _outputs = get_func_inputs_outputs(function)
    roles = dict(kernel.roles)
    for role, window in windows.items():
        want = UPMEM_ROLE_ELEMENT_TYPES[kernel.family][role]
        got = inputs[window.argument][0]
        if got != want:
            raise NotImplementedError(
                f"UPMEM {kernel.family}: operand {roles[role]!r} (role {role!r}, "
                f"argument {window.argument} of {name}) has element type {got}; "
                f"the {kernel.family} DPU template computes "
                f"{_ABI_DTYPE_NAMES[want]} for this role (supported: "
                f"{_supported_types(kernel.family)}). Declare the buffer with "
                "that type."
            )


def _union_windows(windows, role, name) -> "_Window":
    """The bounding window of several accesses to one argument."""
    windows = list(windows)
    if len({w.argument for w in windows}) != 1:
        raise NotImplementedError(
            f"UPMEM role {role!r} ({name}) spans several arguments"
        )
    if len(windows) == 1:
        return windows[0]
    begin = [min(w.offsets[d] for w in windows) for d in range(len(windows[0].offsets))]
    end = [
        max(w.offsets[d] + w.extents[d] for w in windows)
        for d in range(len(windows[0].offsets))
    ]
    return _Window(
        windows[0].argument, tuple(begin), tuple(e - b for b, e in zip(begin, end))
    )


def _check_preconditions(plan, kernel, slices) -> None:
    """Value ranges inside which the templates' narrow arithmetic is exact
    (spec 004 D5). Fails closed naming the offending value."""
    if isinstance(plan, UPMEMKMeansDistancesPlan):
        for role in ("points", "centroids"):
            values = np.asarray(slices[role]).reshape(-1)
            bad = np.nonzero(np.abs(values.astype(np.int64)) > plan.max_abs_value)[0]
            if bad.size:
                raise ValueError(
                    f"k-means {role} value {int(values[bad[0]])} exceeds "
                    f"{plan.max_abs_value}, the bound for an exact int32 accumulator"
                )
    elif isinstance(plan, UPMEMFeatureGradientPlan):
        samples = np.asarray(slices["samples"]).astype(object)
        x, label = (
            samples[:, : plan.features],
            samples[:, plan.features : plan.features + 1],
        )
        if plan.formula is GradientFormula.LINEAR_FIXED_POINT:
            products = abs(x * label) * (1 << plan.shift)
        else:
            products = abs(x * (1 - 2 * label))
        limit = 1 << 63
        for index, value in enumerate(products.reshape(-1)):
            if value >= limit:
                raise ValueError(
                    f"feature-gradient product {value} at flat index {index} "
                    "overflows the template's int64 arithmetic"
                )


# --------------------------------------------------------------------- #
# Packers
# --------------------------------------------------------------------- #


def _packed_int(values, bits: int, what: str) -> bytes:
    """Little-endian ``bits``-wide words; a value that does not fit raises
    instead of being truncated (the template computes in exactly this width)."""
    array = np.asarray(values)
    if array.dtype.kind not in "biuO":
        raise TypeError(
            f"UPMEM {what}: {array.dtype} values cannot be packed as int{bits} "
            "MRAM words exactly"
        )
    flat = array.reshape(-1)
    if array.dtype.kind == "O":
        flat = np.asarray(
            [operator.index(value) for value in flat.tolist()], dtype=object
        )
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if flat.size and not (low <= int(flat.min()) and int(flat.max()) <= high):
        index, value = next(
            (i, v) for i, v in enumerate(flat.tolist()) if not low <= int(v) <= high
        )
        raise ValueError(
            f"UPMEM {what}: value {value} at flat index {index} does not fit "
            f"the template's int{bits} word"
        )
    return np.asarray(flat, dtype=np.int64).astype(f"<i{bits // 8}").tobytes()


def _i32(values, what: str = "operand") -> bytes:
    return _packed_int(values, 32, what)


def _region_dict(region) -> dict:
    manifest = getattr(region, "manifest", None)
    return dict(manifest()) if callable(manifest) else dict(region)


def _region(plan, name: str) -> dict:
    return _region_dict(plan.mram_regions[name])


def _write_region(image: bytearray, region: Mapping, data: bytes, name: str) -> None:
    offset, allocation = int(region["offset"]), int(region["bytes"])
    logical = int(region.get("logical_bytes", allocation))
    if len(data) > logical or logical > allocation:
        raise ValueError(
            f"{name}: {len(data)} payload bytes do not fit the "
            f"{logical}/{allocation}-byte region"
        )
    image[offset : offset + len(data)] = data


def _output_window(plan) -> tuple[int, int]:
    regions = sorted(
        (_region_dict(region) for region in plan.output_regions.values()),
        key=lambda region: int(region["offset"]),
    )
    if not regions:
        raise ValueError("physical plan declares no output regions")
    for previous, current in zip(regions, regions[1:]):
        if int(previous["offset"]) + int(previous["bytes"]) != int(current["offset"]):
            raise ValueError("physical output regions must form one contiguous oracle")
    begin = int(regions[0]["offset"])
    return begin, int(regions[-1]["offset"]) + int(regions[-1]["bytes"])


def _flat(values) -> list:
    return np.asarray(values).reshape(-1).tolist()


def _i64(values, what: str = "operand") -> bytes:
    return _packed_int(values, 64, what)


def _pack_fused_elementwise(plan, lhs, rhs) -> list:
    """``[A_chunk, B_chunk]`` records with zero tails (campaign packing)."""
    values: list = []
    chunk = plan.chunk_elements
    for block in range(plan.chunks):
        begin = block * chunk
        stop = min(begin + chunk, plan.elements)
        values.extend(lhs[begin:stop])
        values.extend([0] * (chunk - (stop - begin)))
        values.extend(rhs[begin:stop])
        values.extend([0] * (chunk - (stop - begin)))
    return values


def _pack_fused_gemm(plan, lhs, rhs) -> list:
    """``[A[Kc], B_col0[Kc], ..., B_colNc-1[Kc]]`` packets (campaign packing)."""
    values: list = []
    for row in range(plan.padded_rows):
        for column_block in range(plan.column_tiles):
            for reduction_block in range(plan.reduction_tiles):
                begin = reduction_block * plan.reduction_tile
                for k in range(plan.reduction_tile):
                    valid = row < plan.rows and begin + k < plan.reduction
                    values.append(lhs[row * plan.reduction + begin + k] if valid else 0)
                for lane in range(plan.column_tile):
                    column = column_block * plan.column_tile + lane
                    for k in range(plan.reduction_tile):
                        index = begin + k
                        valid = (
                            row < plan.rows
                            and column < plan.columns
                            and index < plan.reduction
                        )
                        values.append(
                            rhs[index * plan.columns + column] if valid else 0
                        )
    return values


def _pad_to(values: list, size: int) -> list:
    return values + [0] * (size - len(values))


def _encode_output(plan, out) -> tuple[str, bytes]:
    """``(region, bytes)`` of logical output values in the plan's physical
    output layout (also the initial-output image of a carried accumulator)."""
    if isinstance(plan, UPMEMMatrixVectorPlan):
        return "output", _i32(plan.pack_output(out), "output")
    if isinstance(plan, UPMEMElementwisePlan):
        return "output", _i32(_pad_to(out, plan.padded_elements), "output")
    if isinstance(plan, UPMEMSumReductionPlan):
        return "output", _i64(out, "output")
    if isinstance(plan, UPMEMGEMMPlan):
        physical = [0] * (plan.padded_rows * plan.padded_columns)
        for row in range(plan.rows):
            begin = row * plan.padded_columns
            physical[begin : begin + plan.columns] = out[
                row * plan.columns : (row + 1) * plan.columns
            ]
        return "output", _i32(physical, "output")
    if isinstance(plan, UPMEMHistogramPlan):
        return "histogram", _i32(out, "histogram")
    if isinstance(plan, UPMEMKMeansDistancesPlan):
        return "distances", _i64(out, "distances")
    if isinstance(plan, UPMEMFeatureGradientPlan):
        return "gradient", _i64(out, "gradient")
    raise NotImplementedError(f"no UPMEM output packer for {type(plan).__name__}")


def pack_inputs(plan, kernel: UpmemKernel, slices: Mapping) -> bytes:
    """Full MRAM heap image for one DPU and launch.

    ``slices`` are already padded to the plan's geometry (``_padded_slices``);
    a carried accumulator's initial output arrives as ``initial_out`` and is
    placed in the output region, where the template loads it.
    """
    image = bytearray(int(plan.mram_image_bytes))
    if isinstance(plan, UPMEMMatrixVectorPlan):
        matrix, vector = _flat(slices["matrix"]), _flat(slices["vector"])
        if plan.vector_mode == "tasklet_private_wram":
            matrix_words, vector_words = plan.pack_private_inputs(matrix, vector)
            _write_region(
                image, _region(plan, "matrix"), _i32(matrix_words, "matrix"), "matrix"
            )
            _write_region(
                image, _region(plan, "vector"), _i32(vector_words, "vector"), "vector"
            )
        else:
            _write_region(
                image,
                _region(plan, "packed_matrix_vector"),
                _i32(plan.pack_fused_inputs(matrix, vector), "packed_matrix_vector"),
                "packed_matrix_vector",
            )
    elif isinstance(plan, UPMEMElementwisePlan):
        lhs = _flat(slices["lhs"])
        rhs = _flat(slices["rhs"]) if plan.operands == 2 else []
        if plan.interleaved:
            _write_region(
                image,
                _region(plan, "packed_operands"),
                _i32(_pack_fused_elementwise(plan, lhs, rhs), "packed_operands"),
                "packed_operands",
            )
        else:
            _write_region(image, _region(plan, "x"), _i32(lhs, "x"), "x")
            if plan.operands == 2:
                _write_region(image, _region(plan, "y"), _i32(rhs, "y"), "y")
        if "coefficients" in plan.mram_regions:
            _write_region(
                image,
                _region(plan, "coefficients"),
                _i32((plan.alpha, plan.beta), "coefficients"),
                "coefficients",
            )
    elif isinstance(plan, UPMEMSumReductionPlan):
        values = _flat(slices["input"])
        padded = values + [0] * (plan.padded_elements - len(values))
        _write_region(image, _region(plan, "input"), _i32(padded, "input"), "input")
    elif isinstance(plan, UPMEMGEMMPlan):
        _write_region(
            image,
            _region(plan, "packed_lhs_rhs"),
            _i32(
                _pack_fused_gemm(plan, _flat(slices["lhs"]), _flat(slices["rhs"])),
                "packed_lhs_rhs",
            ),
            "packed_lhs_rhs",
        )
    elif isinstance(plan, (UPMEMHistogramPlan, UPMEMSelectionFlagsPlan)):
        _write_region(
            image,
            _region(plan, "input"),
            _i32(_flat(slices["input"]), "input"),
            "input",
        )
    elif isinstance(plan, UPMEMKMeansDistancesPlan):
        fused = plan.pack_fused_pairs(
            _flat(slices["points"]), _flat(slices["centroids"])
        )
        _write_region(
            image,
            _region(plan, "fused_pairs"),
            _i32(fused, "fused_pairs"),
            "fused_pairs",
        )
    elif isinstance(plan, UPMEMFeatureGradientPlan):
        _write_region(
            image,
            _region(plan, "samples"),
            _i32(_flat(slices["samples"]), "samples"),
            "samples",
        )
    else:
        raise NotImplementedError(f"no UPMEM input packer for {type(plan).__name__}")
    if "initial_out" in slices:
        region, data = _encode_output(plan, _flat(slices["initial_out"]))
        _write_region(image, _region(plan, region), data, f"initial {region}")
    return bytes(image)


def pack_expected(plan, kernel: UpmemKernel, slices: Mapping) -> tuple[int, bytes]:
    """``(offset, bytes)`` of the exact output window for one DPU and launch."""
    image = bytearray(int(plan.mram_image_bytes))
    if isinstance(plan, UPMEMSelectionFlagsPlan):
        # The source never materializes flags; they come from the plan's own
        # reference. The end-to-end check against the oracle runs after the
        # host compaction (``_selection_epilogue``).
        flags = plan.reference_flags(_pad_to(_flat(slices["input"]), plan.elements))
        _write_region(image, _region(plan, "flags"), _i32(flags, "flags"), "flags")
        begin, end = _output_window(plan)
        return begin, bytes(image[begin:end])
    region, data = _encode_output(plan, _flat(slices["out"]))
    _write_region(image, _region(plan, region), data, region)
    begin, end = _output_window(plan)
    return begin, bytes(image[begin:end])


def gather_outputs(plan, kernel: UpmemKernel, window: bytes) -> dict:
    """Decode one DPU's output window into ``{role: logical values}`` at the
    plan's geometry (a padded work item keeps its leading elements)."""
    if isinstance(plan, UPMEMMatrixVectorPlan):
        values = np.frombuffer(window, dtype="<i4")
        return {"out": np.asarray(plan.gather_output(values.tolist()), dtype=np.int64)}
    if isinstance(plan, UPMEMElementwisePlan):
        values = np.frombuffer(window, dtype="<i4").astype(np.int64)
        return {"out": values[: plan.elements]}
    if isinstance(plan, UPMEMSumReductionPlan):
        return {"out": np.frombuffer(window, dtype="<i8").astype(np.int64)}
    if isinstance(plan, UPMEMGEMMPlan):
        values = np.frombuffer(window, dtype="<i4").astype(np.int64)
        grid = values.reshape(plan.padded_rows, plan.padded_columns)
        return {"out": grid[: plan.rows, : plan.columns].reshape(-1)}
    if isinstance(plan, UPMEMHistogramPlan):
        values = np.frombuffer(window, dtype="<u4").astype(np.int64)
        return {"out": values[: plan.bins]}
    if isinstance(plan, UPMEMSelectionFlagsPlan):
        return {
            "flags": np.frombuffer(window, dtype="<i4").astype(np.int64)[
                : plan.elements
            ]
        }
    if isinstance(plan, (UPMEMKMeansDistancesPlan, UPMEMFeatureGradientPlan)):
        return {"out": np.frombuffer(window, dtype="<i8").astype(np.int64)}
    raise NotImplementedError(f"no UPMEM output gather for {type(plan).__name__}")


# Roles whose flattened extent follows the family's uneven axis (the axis is
# outermost, so padding a work item is appending zeros to these roles).
_PADDED_ROLES = {
    "pointwise": ("lhs", "rhs", "out", "initial_out"),
    "reduction": ("input",),
    "matrix_vector": ("matrix", "out", "initial_out"),
    "gemm": ("lhs", "out", "initial_out"),
    "selection": ("input",),
}


def _padded_slices(kernel: UpmemKernel, stream: int, slices: Mapping) -> dict:
    """Zero-pad one work item's role slices to the plan's (largest) geometry.

    Every padded family is padding-neutral: zero inputs produce zero (or a
    discarded) result, and the padded outputs are dropped on gather.
    """
    own, full = dict(kernel.stream_geometry(stream)), dict(kernel.geometry)
    if own == full:
        return dict(slices)
    (axis,) = [key for key in full if own[key] != full[key]]
    roles = _PADDED_ROLES[kernel.family]
    if kernel.family == "matrix_vector" and axis == "batches":
        roles += ("vector",)
    padded = dict(slices)
    for role in roles:
        if role not in padded:
            continue
        values = _flat(padded[role])
        size = len(values) // own[axis] * full[axis]
        padded[role] = _pad_to(values, size)
    return padded


def padding_fraction(kernel: UpmemKernel) -> float:
    """Fraction of the DPUs' work-item extents that is zero padding."""
    if not kernel.streams:
        return 0.0
    full = dict(kernel.geometry)
    (axis,) = {
        key for stream in kernel.streams for key, value in stream if value != full[key]
    }
    logical = sum(dict(stream)[axis] for stream in kernel.streams)
    return 1.0 - logical / (full[axis] * len(kernel.streams))


_INPUT_ROLES = {
    "pointwise": ("lhs", "rhs"),
    "reduction": ("input",),
    "matrix_vector": ("matrix", "vector"),
    "gemm": ("lhs", "rhs"),
    "histogram": ("input",),
    "selection": ("input",),
    "kmeans_distances": ("points", "centroids"),
    "feature_gradient": ("samples",),
}
_OUTPUT_ROLES = {family: ("out",) for family in _INPUT_ROLES}
_OUTPUT_ROLES["selection"] = ("out", "count")


# --------------------------------------------------------------------- #
# Owner-map row placement (movement ties)
# --------------------------------------------------------------------- #

# Roles indexed by the output row: the DPU holding a row holds its matrix row
# and its output element.
_ROW_MAPPED_ROLES = ("matrix", "out")


def _row_map_digest(rows) -> str:
    """FNV-1a 64 over the flattened per-DPU row table."""
    digest = 0xCBF29CE484222325
    for value in rows:
        digest ^= int(value)
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def _owner_row_map(target, kernel, matches, layouts):
    """``(dpu_rows, program_rows, record)`` of the owner map the work items'
    placements carry, or ``(None, None, None)`` for the work grid's own map.

    ``dpu_rows[d]`` are the output rows DPU ``d`` (the ``d``-th work item in
    work-id order) holds, ascending: the rows the map gives flat owner unit
    ``d`` (``spmw_layout_problem.owner_rows``).  Realized for the
    single-batch ``matrix_vector`` family on an even work grid: every DPU
    keeps the plan's row count, so the task image is unchanged and only the
    per-DPU MRAM images and the output gather follow the map.  Raises
    InfeasibleSchedule with the reason otherwise.
    """
    from .pim.layout_problem import TargetHierarchyError
    from .pim.schedule_search import InfeasibleSchedule
    from .spmw_layout_problem import (
        LayoutDerivationError,
        derive_layout_problem,
        owner_rows,
        row_map_unrealizable_reason,
        split_movement_layout,
    )

    movements = [split_movement_layout(getattr(p, "layout", None))[1] for p in layouts]
    if all(m is None for m in movements):
        return None, None, None
    if any(m is None or m != movements[0] for m in movements):
        raise InfeasibleSchedule("UPMEM work items carry different owner maps")
    try:
        derived = derive_layout_problem(target, matches)
        matrix = derived.movement_matrix(movements[0])
    except (LayoutDerivationError, TargetHierarchyError) as error:
        raise InfeasibleSchedule(
            f"UPMEM: the carried movement layout cannot be placed: {error}"
        ) from error
    program = derived.program_layout()
    if matrix == program:
        return None, None, None
    if derived.owner_flag(matrix) != derived.owner_flag(program):
        reason = row_map_unrealizable_reason(derived, matrix)
        if reason is not None:
            raise InfeasibleSchedule(f"UPMEM owner map not realizable: {reason}")
    if (
        kernel.family != "matrix_vector"
        or kernel.geometry_value("batches") != 1
        or kernel.streams
        or kernel.placement != "dpu"
    ):
        raise InfeasibleSchedule(
            "UPMEM owner map not realizable: rows are placed by the owner map only "
            f"for the single-batch matrix_vector family on an even work grid (this "
            f"group: {kernel.family}, geometry {dict(kernel.geometry)}, "
            f"{'uneven' if kernel.streams else 'even'} work items)"
        )
    try:
        units, own = owner_rows(derived, matrix), owner_rows(derived, program)
    except LayoutDerivationError as error:
        raise InfeasibleSchedule(f"UPMEM owner map not realizable: {error}") from error
    rows = kernel.geometry_value("rows")
    if sorted(units) != list(range(kernel.dpus)) or any(
        len(units[d]) != rows for d in units
    ):
        raise InfeasibleSchedule(
            f"UPMEM owner map not realizable: it does not give each of the "
            f"{kernel.dpus} DPUs {rows} rows"
        )
    dpu_rows = tuple(tuple(units[d]) for d in range(kernel.dpus))
    program_rows = tuple(tuple(own.get(d, ())) for d in range(kernel.dpus))
    if dpu_rows == program_rows:
        return None, None, None
    owner_of = {row: d for d, held in enumerate(program_rows) for row in held}
    moved = sum(
        1 for d, held in enumerate(dpu_rows) for row in held if owner_of[row] != d
    )
    position = {
        row: (d, i) for d, held in enumerate(dpu_rows) for i, row in enumerate(held)
    }
    total = rows * kernel.dpus
    first = min(position)
    samples = sorted(
        {first, first + 1, first + rows - 1, first + rows, first + total - 1}
        & set(position)
    )
    record = {
        "map": "permuted",
        "rows_moved": moved,
        "rows_per_dpu": rows,
        "digest": _row_map_digest(row for held in dpu_rows for row in held),
        "placement": {
            row: {"dpu": position[row][0], "local_row": position[row][1]}
            for row in samples
        },
    }
    return dpu_rows, program_rows, record


def _check_row_windows(program, dpu, windows, name) -> None:
    """The IR windows of work item ``dpu`` hold exactly the rows the
    program's own map gives it: its output and matrix windows start at
    the same row, span the plan's rows, and the matrix spans whole rows."""
    from .pim.schedule_search import InfeasibleSchedule

    own = program.program_rows[dpu]
    out, matrix = windows.get("out"), windows.get("matrix")
    if (
        out is None
        or matrix is None
        or len(out.offsets) != 1
        or len(matrix.offsets) != 2
        or tuple(range(out.offsets[0], out.offsets[0] + out.extents[0])) != own
        or matrix.offsets[0] != out.offsets[0]
        or matrix.extents[0] != out.extents[0]
    ):
        raise InfeasibleSchedule(
            f"UPMEM owner map: the windows of {name} are not the rows {own[:1]}..."
            f"{own[-1:]} its work item owns; the row map cannot be applied"
        )


def _mapped_rows(program, dpu, role):
    if program.dpu_rows is None or role not in _ROW_MAPPED_ROLES:
        return None
    return program.dpu_rows[dpu]


def _row_index(window, rows) -> tuple:
    """Index of ``rows`` (dimension 0) inside ``window``'s other dimensions."""
    tail = tuple(
        slice(o, o + e) for o, e in zip(window.offsets[1:], window.extents[1:])
    )
    return (np.asarray(rows, dtype=np.intp),) + tail


def _role_slice(program, dpu, role, window, arrays) -> np.ndarray:
    """One DPU's slice of a role: its work item's window, or the owner
    map's rows when the group places rows by a carried owner map."""
    rows = _mapped_rows(program, dpu, role)
    array = arrays[window.argument]
    if rows is None:
        return window.slice(array)
    return array[_row_index(window, rows)]


# --------------------------------------------------------------------- #
# Codegen context
# --------------------------------------------------------------------- #


@dataclass
class _GroupProgram:
    kernel: UpmemKernel
    decision: UPMEMPhysicalDecision
    plan: object
    functions: list  # work-id function names, in work-id order
    device_source: str
    windows: dict  # function -> {role: _Window}
    oracles: dict  # function -> (compiled C module, abi positions)
    cost_validity: dict | None = None
    # Output rows of each DPU (work-id order) under a carried owner map other
    # than the work grid's; None keeps every work item's own window.
    dpu_rows: tuple | None = None
    program_rows: tuple | None = None
    row_map: dict | None = None


class UpmemCtx(CodegenContext):
    """Lower matched UPMEM kernel groups to calibrated physical plans."""

    def __init__(self, target):
        super().__init__(target)
        self.segments: tuple = ()
        self.groups: dict = {}
        self.arity: int | None = None
        # id(placement) -> (group id, work-item index) of the program emitted
        # for it (realization check); group id -> the work bodies' argument
        # names (from the shared facts).
        self._group_of_layout: dict = {}
        self._parameters_of_group: dict = {}

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ) -> None:
        """Classify each group, build its plan from the placement decision,
        and fold launches into simulator segments.

        With ``module`` (final compile only) the coverage check runs, each
        work-id function's operand windows are proven from the IR, and the
        portable-C functional oracle is compiled.
        """
        del host_moves
        from .pim.schedule_search import InfeasibleSchedule
        from .spmw_autoschedule import _bucket_for_autoschedule

        layout_by_scope = {
            scope: layout
            for (scope, _matches), layout in zip(
                _bucket_for_autoschedule(trace), layouts
            )
        }
        by_group: dict[int, list[MatchedOp]] = {}
        for match in trace.matches:
            by_group.setdefault(_matcher_work_scope(match).group_id, []).append(match)

        groups = {}
        for group_id, matches in by_group.items():
            kernel = classify_upmem_kernel(self.target, matches)
            family = upmem_family(kernel.family)
            scope_by_work = {
                _matcher_work_scope(match).work_id: _matcher_search_scope(match)
                for match in matches
            }
            work_ids = sorted(scope_by_work)
            chosen = {
                scope_by_work[work]: decision_from_extra(
                    family, layout_by_scope[scope_by_work[work]].extra
                )
                for work in work_ids
                if scope_by_work[work] in layout_by_scope
            }
            if not chosen:
                raise ValueError(f"UPMEM kernel group {group_id} carries no decision")
            if len(set(chosen.values())) == 1:
                (decision,) = set(chosen.values())
            elif kernel.streams:
                # Uneven work items are ranked at their own extents; the one
                # shared binary is built for the largest, so its decision is
                # the executed one (it is also the slowest DPU's).
                largest = next(
                    index
                    for index in range(len(work_ids))
                    if kernel.streams[index] == kernel.geometry
                )
                decision = chosen[scope_by_work[work_ids[largest]]]
            else:
                raise ValueError(
                    f"UPMEM kernel group {group_id} carries {len(set(chosen.values()))} "
                    "physical decisions; one is required"
                )
            plan, reason = _realize(kernel, decision)
            if reason is not None:
                raise InfeasibleSchedule(
                    f"UPMEM {kernel.family} decision {decision.manifest()} is not "
                    f"realizable: {reason}"
                )
            device_source = "" if plan is None else plan.device_source()
            dpu_rows = program_rows = row_map = None
            if plan is not None:
                self._check_carried_layout(
                    kernel, plan, decision, chosen, layout_by_scope
                )
                dpu_rows, program_rows, row_map = _owner_row_map(
                    self.target, kernel, matches, [layout_by_scope[s] for s in chosen]
                )
            functions = sorted(
                {m.func_name: _matcher_work_scope(m).work_id for m in matches}.items(),
                key=lambda item: item[1],
            )
            for position, work in enumerate(work_ids):
                scope = scope_by_work[work]
                if scope in layout_by_scope:
                    self._group_of_layout[id(layout_by_scope[scope])] = (
                        group_id,
                        position,
                    )
            self._parameters_of_group[group_id] = {
                name
                for match in matches
                for name in getattr(
                    match.extra.get("program_facts"), "parameter_names", ()
                )
            }
            groups[group_id] = _GroupProgram(
                kernel,
                decision,
                plan,
                [name for name, _ in functions],
                device_source,
                {},
                {},
                upmem_cost_validity(kernel, decision),
                dpu_rows,
                program_rows,
                row_map,
            )

        if module is not None:
            self._bind_module(module, by_group, groups)

        segments = []
        if schedule is None:
            for group_id, program in groups.items():
                segments.append((group_id, 1))
        else:
            from .spmw_plan import LaunchStep

            for step in schedule.steps:
                if not isinstance(step, LaunchStep):
                    segments.append(None)
                    continue
                if step.group_id not in groups:
                    raise ValueError(f"launch of unknown UPMEM group {step.group_id}")
                if (
                    segments
                    and segments[-1] is not None
                    and segments[-1][0] == step.group_id
                ):
                    segments[-1] = (step.group_id, segments[-1][1] + 1)
                else:
                    segments.append((step.group_id, 1))
            segments = [segment for segment in segments if segment is not None]

        self.groups = groups
        self.segments = tuple(
            UpmemSegment(
                kernel=groups[group_id].kernel,
                decision=groups[group_id].decision,
                plan=groups[group_id].plan,
                device_source=groups[group_id].device_source,
                executions=executions,
                kind=groups[group_id].kernel.placement,
            )
            for group_id, executions in segments
        )
        row_maps = {
            program.kernel: program.row_map
            for program in groups.values()
            if program.row_map is not None
        }
        self.cmds = [
            (
                f"segment {index} {segment.kernel.family} host "
                f"executions={segment.executions}"
                if segment.kind == "host"
                else f"segment {index} {segment.kernel.family} dpus={segment.kernel.dpus} "
                f"executions={segment.executions} tasklets={segment.decision.tasklets} "
                f"source_sha256={_sha256(segment.device_source.encode())}"
                + (
                    f" row_map={row_maps[segment.kernel]['digest']}"
                    if segment.kernel in row_maps
                    else ""
                )
            )
            for index, segment in enumerate(self.segments)
        ]

    @staticmethod
    def _check_carried_layout(kernel, plan, decision, chosen, layout_by_scope) -> None:
        """Every work item's placement carries the tasklet carrier the
        emitted plan's ownership factors through."""
        del kernel, decision
        for scope in chosen:
            layout = layout_by_scope[scope].layout
            if layout is not None:
                _check_tasklet_ownership(plan, layout)

    def check_realization(self, layouts) -> None:
        """The realized UPMEM program against the selected candidates'
        declared effects, per work item: the group's placement (DPU template
        or host partition) and family; the launch extents against the family
        template's geometry for that work item (`upmem_family_extents`); the
        accumulator chain (carried outputs load the initial output and are
        ``inout``); the decision's data layout and WRAM residency against the
        declared packings and allocations; the kernel's roles against the
        declared values (at the final compile, the proven role windows); and
        the host steps (every output is gathered)."""
        from .spmw_candidate import candidate_effects

        layouts = list(layouts) if isinstance(layouts, (list, tuple)) else [layouts]
        for layout in layouts:
            effects = candidate_effects(layout)
            if effects is None:
                continue
            located = self._group_of_layout.get(id(layout))
            program = self.groups.get(located[0]) if located is not None else None
            if program is None:
                raise RuntimeError(
                    "UPMEM realization lowered no program for the candidate "
                    f"{layout.mode!r} that declares effects"
                )
            kernel, decision = program.kernel, program.decision
            if len(effects.launches) != 1:
                raise RuntimeError(
                    "UPMEM realization is one template launch per work item; the "
                    f"candidate declares {len(effects.launches)} launches"
                )
            (launch,) = effects.launches
            host = kernel.placement == "host"
            expected_kind = (
                f"upmem_host:{kernel.family}" if host else f"upmem:{kernel.family}"
            )
            if launch.kind != expected_kind or effects.host_function != host:
                raise RuntimeError(
                    f"UPMEM realization runs {kernel.family!r} "
                    f"{'on the host' if host else 'on the DPUs'}; the candidate "
                    f"declares launch {launch.kind!r} (host_function="
                    f"{effects.host_function})"
                )
            roles = dict(kernel.roles)
            declared = set(effects.coverage.inputs) | set(effects.coverage.outputs)
            if host:
                # The host partition's roles may name function locals (the
                # k-means sums); every role that is an argument is declared.
                parameters = self._parameters_of_group.get(located[0], set())
                undeclared = (set(roles.values()) & parameters) - declared
            else:
                undeclared = set(roles.values()) ^ declared
            if undeclared:
                raise RuntimeError(
                    f"UPMEM realization binds the roles {roles}; the candidate "
                    f"declares the values {sorted(declared)} ({sorted(undeclared)} "
                    "differ)"
                )
            if host:
                if program.plan is not None or not launch.whole:
                    raise RuntimeError(
                        "UPMEM host partition runs the function verbatim without a "
                        f"template; the candidate declares plan {program.plan!r}, "
                        f"whole={launch.whole}"
                    )
                if self.arity is not None and not program.oracles:
                    raise RuntimeError(
                        f"UPMEM host partition {kernel.family!r} compiled no host "
                        "function for its group"
                    )
                continue
            instances, reduction = upmem_family_extents(
                kernel, kernel.stream_geometry(located[1])
            )
            realized = ((0, instances), None if reduction is None else (0, reduction))
            if (tuple(launch.instances), launch.reduction) != realized:
                raise RuntimeError(
                    f"UPMEM {kernel.family} template advances (instances, reduction) "
                    f"{realized} for work item {located[1]}; the candidate declares "
                    f"{(tuple(launch.instances), launch.reduction)}"
                )
            if launch.accumulator is not None and (
                launch.accumulator != kernel.accumulator_init
            ):
                raise RuntimeError(
                    f"UPMEM template starts the accumulator as "
                    f"{kernel.accumulator_init!r}; the candidate declares "
                    f"{launch.accumulator!r}"
                )
            carried = kernel.accumulator_init == "carried"
            loads_initial = carried and kernel.family != "selection"
            plan_carried = getattr(program.plan, "carried_output", loads_initial)
            if bool(plan_carried) != loads_initial:
                raise RuntimeError(
                    f"UPMEM plan carried_output={plan_carried!r} disagrees with the "
                    f"kernel's accumulator_init={kernel.accumulator_init!r}"
                )
            transfers = {t.value: t for t in effects.transfers}
            carried_role = _CARRIED_ROLE.get(kernel.family, "out")
            for role, value in roles.items():
                if value not in effects.coverage.outputs:
                    continue
                expected = "inout" if carried and role == carried_role else "out"
                direction = transfers[value].direction if value in transfers else None
                if direction != expected:
                    raise RuntimeError(
                        f"UPMEM template {'loads' if expected == 'inout' else 'does not load'} "
                        f"the initial output {value!r} ({role}); the candidate transfers "
                        f"it {direction!r}"
                    )
                if tuple(transfers[value].route)[-1] != "host_readback":
                    raise RuntimeError(
                        f"UPMEM gathers {value!r} from the MRAM output window; the "
                        f"candidate's route is {transfers[value].route}"
                    )
            fields = decision_fields(decision)
            images = {}
            for value in effects.value_layouts:
                images.setdefault(value.value, {})[value.image] = value
            for name in effects.coverage.inputs:
                image = images.get(name, {}).get("in") or images.get(name, {}).get(
                    "value"
                )
                if image is None or image.storage != "mram":
                    raise RuntimeError(
                        f"UPMEM packs {name!r} into the MRAM heap image; the candidate "
                        f"declared {image}"
                    )
                packing = str(image.packing or "")
                if packing.split(":", maxsplit=1)[0] != fields["data_layout"]:
                    raise RuntimeError(
                        f"UPMEM lays {name!r} out as {fields['data_layout']!r}; the "
                        f"candidate declared packing {image.packing!r}"
                    )
            resident_role = _RESIDENT_ROLE.get(kernel.family)
            resident = roles.get(resident_role) if resident_role else None
            if resident is not None:
                image = images.get(resident, {}).get("in") or images.get(
                    resident, {}
                ).get("value")
                packing = str(getattr(image, "packing", "") or "")
                if packing.rsplit(":", maxsplit=1)[-1] != fields["wram_residency"]:
                    raise RuntimeError(
                        f"UPMEM keeps {resident!r} as {fields['wram_residency']!r}; "
                        f"the candidate declared packing {packing!r}"
                    )
                wram = {
                    name
                    for name, _b, storage in effects.allocations
                    if storage == "wram"
                }
                if (f"{resident_role}_wram" in wram) != (
                    fields["wram_residency"] in _WRAM_RESIDENCIES
                ):
                    raise RuntimeError(
                        f"UPMEM residency {fields['wram_residency']!r} of {resident!r} "
                        f"disagrees with the candidate's WRAM allocations {sorted(wram)}"
                    )
            function = program.functions[located[1]]
            windows = program.windows.get(function)
            if windows is not None:
                bound = {roles[role] for role in windows if role in roles}
                if not bound <= declared:
                    raise RuntimeError(
                        f"UPMEM binds windows for {sorted(bound)} in {function}; the "
                        f"candidate declares {sorted(declared)}"
                    )

    def _bind_module(self, module, by_group, groups) -> None:
        from ._mlir.ir import AsmState
        from .backend.c import emit_c_from_mlir
        from .spmw_match_engine import ValueNames

        functions = _functions(module)
        arity = 0
        for group_id, matches in by_group.items():
            program = groups[group_id]
            roles = dict(program.kernel.roles)
            for name in program.functions:
                function = functions[name]
                state = ValueNames(function, AsmState(function))
                own = [match for match in matches if match.func_name == name]
                abi = [int(v.value) for v in function.attributes["spmw.abi_value_ids"]]
                if program.kernel.placement == "host":
                    # A host group runs its own MLIR: no template, no windows.
                    _check_kmeans_update_epilogue(function, program.kernel)
                    artifact = emit_c_from_mlir(module, name, wrap_wide_integers=True)
                    program.oracles[name] = (artifact.compile(), abi)
                    arity = max(arity, max(abi, default=-1) + 1)
                    continue
                _check_coverage(function, state, own)
                windows = {}
                body = _walk(function.regions[0].blocks[0], [])
                for role, memref_name in roles.items():
                    # A role written by a match binds through that match's
                    # store; any other role through its loads.
                    writer = next(
                        (m for m in own if m.result_memref_name == memref_name), None
                    )
                    if writer is not None and "index_terms" in writer.extra:
                        # A computed destination (spec 004 M6) may touch any
                        # element: the role binds the whole array.
                        arg = next(
                            (
                                _func_argument_number(function, op.operands[1])
                                for op in body
                                if op.name in ("affine.store", "memref.store")
                                and _store_handle(op, state) == writer.op_range[1]
                            ),
                            None,
                        )
                        if arg is None:
                            raise NotImplementedError(
                                f"UPMEM role {role!r} ({memref_name}) is not a kernel argument"
                            )
                        shape = tuple(map(int, function.arguments[arg].type.shape))
                        windows[role] = _Window(arg, (0,) * len(shape), shape)
                        continue
                    if writer is not None:
                        owner = writer
                        ivs = list(writer.extra.get("store_indices", ()))
                        accesses = [
                            op
                            for op in body
                            if op.name in ("affine.store", "memref.store")
                            and _store_handle(op, state) == writer.op_range[1]
                        ]
                    else:
                        owner, operand = next(
                            (
                                (m, operand)
                                for m in own
                                for operand in m.operands
                                if operand.memref_name == memref_name
                            ),
                            (None, None),
                        )
                        if owner is None:
                            # Read only inside a recorded index term or guard
                            # (histogram input): bind through its loads under
                            # the anchor's loops.
                            owner = next(
                                m
                                for m in own
                                if m.target_op_name == program.kernel.anchor_op
                            )
                        ivs = None
                        accesses = [
                            op
                            for op in body
                            if op.name in ("affine.load", "memref.load")
                            and "from" in op.attributes
                            and _attr_text(op.attributes["from"]) == memref_name
                        ]
                    loops = owner.enclosing_loops
                    # Each load is cross-checked against its own index operands,
                    # which is what the matcher records for that load.
                    found = {
                        _access_window(
                            op,
                            function,
                            state,
                            loops,
                            (
                                ivs
                                if ivs is not None
                                else [
                                    _value_name(v, state) for v in list(op.operands)[1:]
                                ]
                            ),
                        )
                        for op in accesses
                    }
                    if not found:
                        raise NotImplementedError(
                            f"UPMEM role {role!r} ({memref_name}) in {name} has no access"
                        )
                    windows[role] = _union_windows(found, role, memref_name)
                _check_role_dtypes(program.kernel, function, name, windows)
                if any(value >= _LOCAL_ABI_ID for value in abi):
                    raise NotImplementedError(
                        f"{name} reads a region-local buffer; the UPMEM oracle binds "
                        "only caller arguments"
                    )
                arity = max(arity, max(abi, default=-1) + 1)
                if program.dpu_rows is not None:
                    _check_row_windows(
                        program, program.functions.index(name), windows, name
                    )
                program.windows[name] = {
                    role: _Window(abi[w.argument], w.offsets, w.extents)
                    for role, w in windows.items()
                }
                artifact = emit_c_from_mlir(module, name, wrap_wide_integers=True)
                program.oracles[name] = (artifact.compile(), abi)
        self.arity = arity

    def virtual_run_extra(self) -> dict:
        """``RunResult.extra`` fields of a ``backend="virtual"`` run.

        The virtual estimate is the cost graph: DPU logic cycles from the
        physical model plus any modeled host steps (explicit host moves, host
        groups, selection compaction).  It is therefore never comparable
        one-to-one with the simulator's device-only ``cycles``.
        """
        labels = [
            {
                "group": group_id,
                "family": program.kernel.family,
                "placement": program.kernel.placement,
                **(program.cost_validity or {}),
            }
            for group_id, program in sorted(self.groups.items())
        ]
        overall = (
            "calibrated"
            if labels
            and all(label.get("cost_validity") == "calibrated" for label in labels)
            else "extrapolated"
        )
        return {
            "timing_scope": VIRTUAL_TIMING_SCOPE,
            "cost_validity": overall,
            "upmem_cost_validity": labels,
        }

    def runtime_route(self, commands):
        del commands
        return (
            "upimulator-generic-fixture",
            tuple(
                (
                    segment.kernel.family,
                    segment.kind,
                    segment.kernel.dpus,
                    segment.executions,
                    _sha256(segment.device_source.encode()),
                )
                for segment in self.segments
            ),
        )

    # ------------------------------------------------------------------ #

    def _arrays(self, inputs) -> list:
        if self.arity is None:
            raise RuntimeError(
                "UPMEM prepare needs the final compile (emit_program with module=)"
            )
        values = list(inputs.values()) if isinstance(inputs, Mapping) else list(inputs)
        if len(values) < self.arity:
            raise TypeError(
                f"UPMEM program takes {self.arity} arrays, got {len(values)}"
            )
        return values[: self.arity]

    def _launch(self, program: _GroupProgram, arrays: list) -> None:
        for name in program.functions:
            executable, abi = program.oracles[name]
            executable(*[arrays[position] for position in abi])

    def prepare(self, inputs) -> UpmemBundle:
        """Run the functional oracle and pack one ``GENERIC`` fixture per segment.

        Pure: the caller's arrays are copied, and no simulator runs.
        """
        arrays = [
            np.ascontiguousarray(np.array(value, copy=True))
            for value in self._arrays(inputs)
        ]
        program_by_kernel = {
            program.kernel: program for program in self.groups.values()
        }
        fixtures, expected = [], []
        states = []
        for segment in self.segments:
            program = program_by_kernel[segment.kernel]
            family = segment.kernel.family
            if segment.kind == "host":
                for _ in range(segment.executions):
                    self._launch(program, arrays)
                fixtures.append(None)
                expected.append({})
                states.append([array.copy() for array in arrays])
                continue
            inputs_blobs: list[bytes] = []
            expected_blobs: list[bytes] = []
            executions = []
            outputs = {}
            for execution in range(segment.executions):
                before = [array.copy() for array in arrays]
                self._launch(program, arrays)
                dpus = []
                for dpu, name in enumerate(program.functions):
                    windows = program.windows[name]
                    slices_in = {
                        r: _role_slice(program, dpu, r, windows[r], before)
                        for r in _INPUT_ROLES[family]
                        if r in windows
                    }
                    _check_preconditions(segment.plan, segment.kernel, slices_in)
                    device_in = dict(slices_in)
                    if segment.plan is not None and _carries_output_on_device(segment):
                        device_in["initial_out"] = _role_slice(
                            program, dpu, "out", windows["out"], before
                        )
                    device_in = _padded_slices(segment.kernel, dpu, device_in)
                    heap = pack_inputs(segment.plan, segment.kernel, device_in)
                    offset, oracle = pack_expected(
                        segment.plan,
                        segment.kernel,
                        _padded_slices(
                            segment.kernel,
                            dpu,
                            {
                                **slices_in,
                                **{
                                    r: _role_slice(program, dpu, r, windows[r], arrays)
                                    for r in _OUTPUT_ROLES[family]
                                    if r in windows
                                },
                            },
                        ),
                    )
                    outputs[(execution, dpu)] = (offset, oracle)
                    dpus.append(
                        {
                            "host_inputs": {},
                            "host_outputs": {},
                            "mram_heap_input": {
                                "offset": 0,
                                "file": _blob_name(inputs_blobs, heap, "mram.input"),
                            },
                            "mram_heap_output": {
                                "offset": offset,
                                "file": _blob_name(
                                    expected_blobs, oracle, "mram.output.expected"
                                ),
                            },
                        }
                    )
                executions.append({"dpus": dpus})
            manifest = {
                "executions": executions,
                "num_dpus": segment.kernel.dpus,
                "num_tasklets": segment.decision.tasklets,
                "schema_version": 1,
            }
            files = tuple(
                (_indexed_name("mram.input", i), blob)
                for i, blob in enumerate(inputs_blobs)
            ) + tuple(
                (_indexed_name("mram.output.expected", i), blob)
                for i, blob in enumerate(expected_blobs)
            )
            fixtures.append(UpmemFixture(manifest, files))
            expected.append(outputs)
            states.append([array.copy() for array in arrays])
        return UpmemBundle(tuple(fixtures), tuple(expected), tuple(states))


def _host_steps(segment) -> list:
    """Host-runtime steps of a DPU segment, as part of the retained program:
    the per-DPU output gather into the caller's arrays (priced by explicit
    ``host_moves`` GATHER steps when the program declares them) and, for
    compaction-form selection, the stable compaction of the device flags
    (priced by the cost program's ``selection`` host rule)."""
    kernel = segment.kernel
    steps = [
        {
            "kind": "gather_outputs",
            "dpus": kernel.dpus,
            "bytes_per_dpu": _output_window(segment.plan)[1]
            - _output_window(segment.plan)[0],
            "priced_by": "host_moves GATHER_MRAM (when declared)",
        }
    ]
    if kernel.family == "selection" and kernel.geometry_value("compact"):
        steps.append(
            {
                "kind": "selection_compaction",
                "elements": kernel.geometry_value("elements") * kernel.dpus,
                "modeled_host_cycles": kernel.dpus
                * upmem_costs.selection_compaction_cycles(
                    kernel.geometry_value("elements")
                ),
                "priced_by": "costs/upmem.py selection_compaction_cycles",
            }
        )
    return steps


def _carries_output_on_device(segment) -> bool:
    """A carried accumulator whose template loads the initial output (the
    selection count is carried by the host compaction step instead)."""
    return (
        segment.kernel.accumulator_init == "carried"
        and segment.kernel.family != "selection"
    )


def _indexed_name(stem: str, index: int) -> str:
    return f"{stem}.bin" if index == 0 else f"{stem}.{index}.bin"


def _blob_name(blobs: list, blob: bytes, stem: str) -> str:
    for index, existing in enumerate(blobs):
        if existing == blob:
            return _indexed_name(stem, index)
    blobs.append(blob)
    return _indexed_name(stem, len(blobs) - 1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #


def _stage_task_c(root: Path, source: str) -> None:
    generic = root / "benchmark" / spmw_simenv.upmem_benchmark_slot()
    dpu = generic / "dpu"
    for child in dpu.iterdir():
        if child.name == "CMakeLists.txt":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in generic.iterdir():
        if child.name in {"CMakeLists.txt", "README.md", "dpu"}:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (dpu / "task.c").write_text(source, encoding="utf-8")


def _log_tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _first_mismatch(segment, expected, bin_dir: Path) -> str | None:
    for path in sorted(bin_dir.glob("observed_mram_heap_offset_*.raw.bin")):
        match = _OBSERVED.match(path.name)
        if match is None:
            continue
        execution, dpu = int(match.group(2)), int(match.group(3))
        _offset, oracle = expected[(execution, dpu)]
        observed = path.read_bytes()
        if observed == oracle:
            continue
        got = gather_outputs(segment.plan, segment.kernel, observed)
        want = gather_outputs(segment.plan, segment.kernel, oracle)
        for role in want:
            diff = np.nonzero(np.asarray(got[role]) != np.asarray(want[role]))[0]
            if diff.size:
                index = int(diff[0])
                return (
                    f"execution {execution} DPU {dpu} role {role!r} element {index}: "
                    f"got {int(got[role][index])}, expected {int(want[role][index])}"
                )
        return f"execution {execution} DPU {dpu}: output bytes differ"
    return None


def _store_exact(view: np.ndarray, values, what: str) -> None:
    """Write device results into a caller view; raise rather than let the
    caller's dtype narrow or round them."""
    data = np.asarray(values, dtype=np.int64).reshape(view.shape)
    if view.dtype.kind not in "iu":
        raise TypeError(
            f"UPMEM {what}: the caller's {view.dtype} array cannot hold the "
            "device's integer results exactly"
        )
    if data.size:
        info = np.iinfo(view.dtype)
        low, high = int(data.min()), int(data.max())
        if low < info.min or high > info.max:
            raise ValueError(
                f"UPMEM {what}: device value {low if low < info.min else high} "
                f"does not fit the caller's {view.dtype} array"
            )
    view[...] = data.astype(view.dtype)


def _selection_epilogue(segment, windows, flags, arrays, oracle_state) -> None:
    """Host stable compaction of the observed DPU flags (spec 004 D1).

    Generated from the ``COMPACT`` and ``INC`` matches: keep, in index order,
    every input the device marked (``value_or_zero`` flags: the nonzero odd
    values; ``mask`` flags: the inputs whose flag is 1), write them to
    ``out[c0:c0 + k]`` and set the counter to ``c0 + k``, where ``c0`` is
    the counter's initial value (0 when the source zeroes it, the caller's
    value when it is carried).  ``out`` outside that range is untouched, as
    in the source.  The flags form (``SELECT_ODD``) writes the flags to its
    output instead.  The result must equal the functional oracle's, or the
    run raises.
    """
    flags = np.asarray(flags, dtype=np.int64)
    out_window = windows["out"]
    out = out_window.slice(arrays[out_window.argument]).reshape(-1)
    if not segment.kernel.geometry_value("compact"):
        _store_exact(out, flags, "selection flags")
        return
    if segment.plan.encoding == "value_or_zero":
        kept = flags[(flags & 1) != 0]
    else:
        input_window = windows["input"]
        values = input_window.slice(arrays[input_window.argument]).reshape(-1)
        kept = np.asarray(values, dtype=np.int64)[flags != 0]
    count_window = windows["count"]
    count = count_window.slice(arrays[count_window.argument]).reshape(-1)
    start = int(count[0]) if segment.kernel.accumulator_init == "carried" else 0
    if start < 0 or start + kept.size > out.size:
        raise RuntimeError(
            f"UPMEM selection: compacting {kept.size} values after the initial "
            f"count {start} overruns the {out.size}-element output"
        )
    _store_exact(out[start : start + kept.size], kept, "selection output")
    _store_exact(count[:1], [start + kept.size], "selection count")
    want_out = out_window.slice(oracle_state[out_window.argument]).reshape(-1)
    want_count = count_window.slice(oracle_state[count_window.argument]).reshape(-1)
    if int(want_count[0]) != start + kept.size or not np.array_equal(
        out[start : start + kept.size], want_out[start : start + kept.size]
    ):
        raise RuntimeError(
            "UPMEM selection: host compaction of the simulated flags differs from "
            f"the functional oracle (count {start + kept.size} vs {int(want_count[0])})"
        )


def _run_upmem(compiled, **inputs) -> RunResult:
    """Run every segment on the provenance uPIMulator through ``GENERIC``.

    ``RunResult.cycles`` is the sum over segments of the max DPU logic cycle
    count; host packing and transfers are excluded, as in the paper metric.
    Outputs decoded from the simulator overwrite the caller's arrays in place.
    """
    reason = spmw_simenv.upmem_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("upmem", reason)
    root = spmw_simenv.ensure_upimulator_root()
    ctx = compiled.layout_ctx
    bundle = ctx.prepare(inputs)
    caller_arrays = ctx._arrays(inputs)

    binary = spmw_simenv.upimulator_bin()
    simulator_sha = spmw_simenv.upimulator_sha256()
    snapshot_sha = (
        spmw_simenv._file_sha256(spmw_simenv.upimulator_snapshot())
        if "UPIMULATOR_ROOT" not in os.environ
        else None
    )
    slot = spmw_simenv.upmem_benchmark_slot()
    runs_root = spmw_simenv.upmem_scratch_dir() / "runs"
    program_by_kernel = {p.kernel: p for p in ctx.groups.values()}

    records = []
    total = 0
    stdout_parts = []
    for index, (segment, fixture, expected) in enumerate(
        zip(ctx.segments, bundle.fixtures, bundle.expected)
    ):
        if segment.kind == "host":
            # Spec 004 D1: the paper partition keeps this group on the host.
            # It runs its own MLIR (the functional oracle) on the live arrays,
            # after the preceding DPU segments wrote their observed outputs.
            started = time.perf_counter()
            for _ in range(segment.executions):
                ctx._launch(program_by_kernel[segment.kernel], caller_arrays)
            records.append(
                {
                    "kind": "host",
                    "family": segment.kernel.family,
                    "executions": segment.executions,
                    "host_wall_s": time.perf_counter() - started,
                    # Device cycles only (the paper metric); the host work is
                    # reported separately, in the cost program's host unit.
                    "logic_cycles": 0,
                    "modeled_host_cycles": segment.executions
                    * upmem_costs.host_group_cycles(dict(segment.kernel.geometry)),
                    "timing_scope": _TIMING_SCOPE,
                    "host_timing_scope": upmem_costs.HOST_TIMING_SCOPE,
                }
            )
            continue
        fixture_sha = fixture.sha256()
        task_sha = _sha256(segment.device_source.encode())
        run_key = _sha256(
            json.dumps([task_sha, fixture_sha, simulator_sha], sort_keys=True).encode()
        )[:16]
        run_dir = runs_root / run_key
        if run_dir.exists():
            shutil.rmtree(run_dir)
        fixture_dir, bin_dir, log_dir = (
            run_dir / "fixture",
            run_dir / "bin",
            run_dir / "log",
        )
        fixture.write(fixture_dir)
        bin_dir.mkdir(parents=True)
        tasklets = segment.decision.tasklets
        dpus = segment.kernel.dpus
        # The DPU SDK executor: the Docker image, or the host-installed SDK
        # through a user-namespace bind at the image's path
        # (`spmw_simenv.upmem_sdk_build_command`); recorded per run.
        build = spmw_simenv.upmem_sdk_build_command(
            root,
            ["--num_dpus", "1", "--num_tasklets", str(tasklets), "--benchmark", slot],
        )
        simulate = [
            str(binary),
            "--benchmark",
            slot,
            "--fixture_dir",
            str(fixture_dir),
            "--num_channels",
            "1",
            "--num_ranks_per_channel",
            "1",
            "--num_dpus_per_rank",
            str(dpus),
            "--num_tasklets",
            str(tasklets),
            "--num_simulation_threads",
            "16",
            "--logic_frequency",
            "350",
            "--skip_compilation",
            "true",
            "--root_dirpath",
            str(root),
            "--bin_dirpath",
            str(bin_dir),
            "--log_dirpath",
            str(log_dir),
        ]
        lock_path = root.parent / "run.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _stage_task_c(root, segment.device_source)
            built = subprocess.run(
                build, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S
            )
            (run_dir / "build.log").write_text(built.stdout + built.stderr)
            if built.returncode != 0:
                raise RuntimeError(
                    f"UPMEM DPU build failed (exit {built.returncode}):\n"
                    + "\n".join((built.stdout + built.stderr).splitlines()[-40:])
                )
            sdk_sha = spmw_simenv.sdk_build_manifest_sha256(root)
            # "pinned-campaign" (raises on drift) or "user-build:<manifest12>"
            # when UPIMULATOR_ROOT and UPIMULATOR_BIN name a user's own build.
            sdk_build = spmw_simenv.upmem_sdk_build_identity(sdk_sha)
            task_object = (
                root
                / "benchmark"
                / "build"
                / slot
                / "dpu"
                / "CMakeFiles"
                / f"{slot}_device.dir"
                / "task.c.o"
            )
            task_object_sha = _sha256(task_object.read_bytes())
            simulated = subprocess.run(
                simulate, capture_output=True, text=True, timeout=_SIMULATE_TIMEOUT_S
            )
        (run_dir / "simulate.log").write_text(simulated.stdout + simulated.stderr)
        log_path = bin_dir / "log.txt"
        if simulated.returncode != 0:
            mismatch = _first_mismatch(segment, expected, bin_dir)
            # A located mismatch is the whole story; the SDK's analysis
            # output stays in the run directory's log.
            raise RuntimeError(
                f"uPIMulator failed on segment {index} (exit {simulated.returncode})"
                + (
                    f"; first mismatch: {mismatch} (log: {run_dir / 'simulate.log'})"
                    if mismatch
                    else "\n"
                    + "\n".join(
                        (simulated.stdout + simulated.stderr).splitlines()[-40:]
                    )
                )
            )
        log_text = log_path.read_text(errors="replace")
        per_dpu = {}
        for channel, rank, dpu, cycles in _LOGIC_CYCLE.findall(log_text):
            per_dpu[(int(channel), int(rank), int(dpu))] = int(cycles)
        if len(per_dpu) != dpus:
            raise RuntimeError(
                f"uPIMulator reported {len(per_dpu)} DPU logic-cycle counters for "
                f"{dpus} DPUs\n{_log_tail(log_path)}"
            )
        segment_cycles = max(per_dpu.values())
        total += segment_cycles
        stdout_parts.append(simulated.stdout)

        program = program_by_kernel[segment.kernel]
        for execution in range(segment.executions):
            for dpu, name in enumerate(program.functions):
                offset, _oracle = expected[(execution, dpu)]
                observed = bin_dir / (
                    f"observed_mram_heap_offset_{offset}_execution_{execution}"
                    f"_dpu_{dpu}.raw.bin"
                )
                values = gather_outputs(
                    segment.plan, segment.kernel, observed.read_bytes()
                )
                if "flags" in values:
                    own = dict(segment.kernel.stream_geometry(dpu))["elements"]
                    _selection_epilogue(
                        segment,
                        program.windows[name],
                        values["flags"][:own],
                        caller_arrays,
                        bundle.states[index],
                    )
                    continue
                for role, data in values.items():
                    window = program.windows[name][role]
                    rows = _mapped_rows(program, dpu, role)
                    if rows is not None:
                        # The DPU holds the owner map's rows, in ascending order.
                        index = _row_index(window, rows)
                        destination = caller_arrays[window.argument][index].copy()
                        _store_exact(
                            destination,
                            np.asarray(data).reshape(-1)[: destination.size],
                            f"{segment.kernel.family} role {role!r}",
                        )
                        caller_arrays[window.argument][index] = destination
                        continue
                    destination = window.slice(caller_arrays[window.argument])
                    # A zero-padded work item keeps its leading elements.
                    _store_exact(
                        destination,
                        np.asarray(data).reshape(-1)[: destination.size],
                        f"{segment.kernel.family} role {role!r}",
                    )

        records.append(
            {
                "kind": "dpu",
                "family": segment.kernel.family,
                "task_c_sha256": task_sha,
                "task_c_o_sha256": task_object_sha,
                "fixture_sha256": fixture_sha,
                "simulator_sha256": simulator_sha,
                "snapshot_sha256": snapshot_sha,
                "sdk_build_sha256": sdk_sha,
                "sdk_build": sdk_build,
                "sdk_executor": spmw_simenv.upmem_sdk_executor_label(),
                "commands": {"compile": build, "simulate": simulate},
                "num_dpus": dpus,
                "num_tasklets": tasklets,
                "executions": segment.executions,
                "logic_cycles_per_dpu": {
                    f"{c}_{r}_{d}": cycles
                    for (c, r, d), cycles in sorted(per_dpu.items())
                },
                "logic_cycles": segment_cycles,
                "timing_scope": _TIMING_SCOPE,
                "geometry": dict(segment.kernel.geometry),
                "work_item_geometry": [
                    dict(segment.kernel.stream_geometry(d)) for d in range(dpus)
                ],
                "padding_fraction": padding_fraction(segment.kernel),
                "plan_manifest_sha256": _sha256(
                    json.dumps(
                        segment.plan.manifest(), sort_keys=True, default=str
                    ).encode()
                ),
                "host_steps": _host_steps(segment),
                **({"row_map": program.row_map} if program.row_map is not None else {}),
            }
        )
    # The run writes decoded outputs into the caller's arrays in place;
    # extra["outputs"] exposes each argument's post-run array by name.
    return RunResult(
        kind="simulator",
        cycles=total,
        stdout="".join(stdout_parts),
        backend="upmem",
        extra={
            "upmem": records,
            "timing_scope": _TIMING_SCOPE,
            "outputs": {
                name: value
                for name, value in zip(inputs, caller_arrays)
                if isinstance(value, np.ndarray)
            },
        },
    )


__all__ = [
    "UpmemBundle",
    "UpmemCtx",
    "UpmemFamily",
    "UpmemFixture",
    "UpmemKernel",
    "UpmemSegment",
    "classify_upmem_kernel",
    "decision_fields",
    "UPMEM_DECISION_DOMAINS",
    "UPMEM_KNOBS",
    "decision_from_extra",
    "kernel_from_extra",
    "upmem_base_placements",
    "upmem_kernel_features",
    "upmem_knob_candidates",
    "upmem_knob_emit",
    "gather_outputs",
    "pack_expected",
    "pack_inputs",
    "placement_extra",
    "register_upmem_family",
    "upmem_family",
]
