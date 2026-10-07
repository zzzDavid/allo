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

import fcntl
import hashlib
import json
import math
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
    physical_features,
)
from .spmw_autoschedule import (
    Placement,
    _matcher_search_scope,
    _matcher_work_scope,
)
from .spmw_codegen import CodegenContext, RunResult, SimulatorUnavailable
from .spmw_match import MatchedOp

_TIMING_SCOPE = "uPIMulator DPU logic_cycle only"
_BUILD_TIMEOUT_S = 600
_SIMULATE_TIMEOUT_S = 1800
_LOGIC_CYCLE = re.compile(r"^Logic\[(\d+)_(\d+)_(\d+)\]_logic_cycle:\s*(\d+)\s*$", re.M)
_OBSERVED = re.compile(r"^observed_mram_heap_offset_(\d+)_execution_(\d+)_dpu_(\d+)\.raw\.bin$")
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

    def geometry_value(self, name: str) -> int:
        return int(dict(self.geometry)[name])

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


_UPMEM_FAMILIES: dict[str, UpmemFamily] = {}
# Family name -> ordered physical decision domain (spec U5).
UPMEM_DECISION_DOMAINS: dict[str, tuple] = {}


def register_upmem_family(name, recognize, plan_for, features, domain) -> UpmemFamily:
    family = UpmemFamily(name, recognize, plan_for, features, tuple(domain))
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
        first = shapes[0]
        if any(shape != first for shape in shapes[1:]):
            raise NotImplementedError(
                f"UPMEM family {family.name!r}: work-id streams of one kernel "
                "group classify differently"
            )
        found = (family, first)
        break
    if found is None:
        ops = sorted({match.target_op_name for match in matches})
        functions = sorted({match.func_name for match in matches})
        raise NotImplementedError(
            f"no UPMEM kernel family matches ops {ops} in {functions}"
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
    return UpmemKernel(
        family=family.name,
        group_id=group_ids.pop(),
        geometry=tuple(shape["geometry"]),
        anchor_op=shape["anchor_op"],
        constants=tuple(shape["constants"]),
        roles=tuple(shape["roles"]),
        dpus=len(streams),
        placement=shape.get("placement", "dpu"),
    )


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


# ------------------------------ pointwise ------------------------------ #

# Matched op -> (plan operation, physical-model instructions per element).
_POINTWISE_OPERATIONS = {"ADD": ("add", 1), "AXPBY": ("axpby", 2)}


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
    """``out[i] = lhs[i] (op) rhs[i]`` as one ADD or AXPBY over a single loop."""
    del target
    if len(stream) != 1 or stream[0].target_op_name not in _POINTWISE_OPERATIONS:
        return None
    match = stream[0]
    extents = _loop_extents(match)
    if extents is None or len(extents) != 1:
        return None
    (iv,) = (loop[0] for loop in match.enclosing_loops)
    operands = _data_operands(match)
    lhs, rhs = operands.get("x"), operands.get("y")
    if lhs is None or rhs is None or any(op.is_loop_carried for op in (lhs, rhs)):
        return None
    if any(_memref_rank(op) != 1 or list(op.indices) != [iv] for op in (lhs, rhs)):
        return None
    if match.result_memref_name in (None, lhs.memref_name, rhs.memref_name):
        return None
    if match.extra.get("store_indices") != [iv]:
        return None
    constants = tuple(sorted(match.extra.get("constants", {}).items()))
    return {
        "geometry": (("elements", extents[0]),),
        "anchor_op": match.target_op_name,
        "constants": constants,
        "roles": (
            ("lhs", lhs.memref_name),
            ("rhs", rhs.memref_name),
            ("out", match.result_memref_name),
        ),
    }


def _pointwise_operation(kernel: UpmemKernel) -> tuple[str, int]:
    return _POINTWISE_OPERATIONS[kernel.anchor_op]


def _pointwise_problem(kernel: UpmemKernel) -> UPMEMPhysicalProblem:
    return UPMEMPhysicalProblem.pointwise(
        kernel.geometry_value("elements"),
        operation_instructions=_pointwise_operation(kernel)[1],
    )


def _pointwise_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    interleaved = decision.layout is UPMEMDataLayout.FUSED_REPLICATED
    operation = _pointwise_operation(kernel)[0]
    coefficients = {}
    if operation == "axpby":
        constants = dict(kernel.constants)
        coefficients = {"alpha": int(constants["alpha"]), "beta": int(constants["beta"])}
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
    cells = [op for op in operands.values() if op.memref_name == match.result_memref_name]
    values = [op for op in operands.values() if op.memref_name != match.result_memref_name]
    if len(cells) != 1 or len(values) != 1:
        return None
    (cell,), (value,) = cells, values
    if iv in cell.indices or match.extra.get("store_indices") != list(cell.indices):
        return None
    if _memref_rank(value) != 1 or list(value.indices) != [iv]:
        return None
    if not (cell.memref_type or "").endswith("xi64>"):
        return None
    if not (value.memref_type or "").endswith("xi32>"):
        return None
    return {
        "geometry": (("elements", extents[0]),),
        "anchor_op": "ADD",
        "constants": (),
        "roles": (("input", value.memref_name), ("out", cell.memref_name)),
    }


def _reduction_plan(kernel: UpmemKernel, decision: UPMEMPhysicalDecision):
    return UPMEMSumReductionPlan(
        kernel.geometry_value("elements"), dma_bytes=decision.dma_bytes
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
    matrix = next(
        (op for op in inputs if list(op.indices) == outer + [red_iv]), None
    )
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
    else:
        if list(acc.indices) != outer or acc.memref_name != match.result_memref_name:
            return None
        out = acc.memref_name
    return {
        "geometry": (("batches", batches), ("rows", rows), ("columns", extents[-1])),
        "anchor_op": "MAC",
        "constants": constants,
        "roles": (
            ("matrix", matrix.memref_name),
            ("vector", vector.memref_name),
            ("out", out),
        ),
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
        (UPMEMDataLayout.FUSED_REPLICATED, 320, 16, UPMEMOperandResidency.FUSED_DMA_PACKET, 4, 16),
        (UPMEMDataLayout.FUSED_REPLICATED, 576, 16, UPMEMOperandResidency.FUSED_DMA_PACKET, 8, 16),
        (UPMEMDataLayout.FUSED_REPLICATED, 1088, 16, UPMEMOperandResidency.FUSED_DMA_PACKET, 16, 16),
        (UPMEMDataLayout.FUSED_REPLICATED, 1056, 8, UPMEMOperandResidency.FUSED_DMA_PACKET, 32, 8),
        (UPMEMDataLayout.SEPARATE, 2048, 64, UPMEMOperandResidency.SHARED_WRAM, 128, 64),
    )
)

register_upmem_family(
    "gemm",
    _recognize_gemm,
    _gemm_plan,
    _gemm_problem,
    _GEMM_DOMAIN,
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
    parts = _histogram_index(terms[0])
    if parts is None:
        return None
    x, bins, depth = parts
    (iv,) = (loop[0] for loop in match.enclosing_loops)
    if list(x.indices) != [iv]:
        return None
    (acc,) = [op for op in match.operands if op.is_loop_carried]
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
    }


def _histogram_plan(kernel, decision):
    del decision
    return UPMEMHistogramPlan(
        kernel.geometry_value("elements"),
        kernel.geometry_value("bins"),
        kernel.geometry_value("depth"),
    )


# hist was not in the physical search: one decision, the plan defaults.
_HISTOGRAM_DOMAIN = (UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 512, 128),)

register_upmem_family(
    "histogram", _recognize_histogram, _histogram_plan, None, _HISTOGRAM_DOMAIN
)


# ------------------------------ selection ------------------------------ #


def _odd_guard_value(guards):
    """The tested term ``x`` when the single guard is ``x & 1 != 0`` or
    ``x & 1 == 1`` (mirrors accepted), else None."""
    E = _engine()
    if len(guards) != 1 or guards[0][0] != "if":
        return None
    for pred, lhs, rhs in _cmp_parts(guards[0][1]):
        if not (isinstance(lhs, E.WBinOp) and lhs.op == "and"):
            continue
        value = E._const_value(rhs)
        if not ((pred == "ne" and value == 0) or (pred == "eq" and value == 1)):
            continue
        for x, mask in ((lhs.lhs, lhs.rhs), (lhs.rhs, lhs.lhs)):
            if E._const_value(mask) == 1:
                return x
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
    guards = _guards(compact)
    if len(guards) != 1 or len(_guards(counter)) != 1:
        return None
    if guards[0][0] != _guards(counter)[0][0] or not E._term_eq(
        guards[0][1], _guards(counter)[0][1]
    ):
        return None
    tested = _odd_guard_value(guards)
    if tested is None:
        raise NotImplementedError(
            "selection predicate is not 'odd'; the selection plan supports "
            "x & 1 != 0 only"
        )
    if not _load_is(tested, value):
        raise NotImplementedError("selection guard tests a value other than the copied one")
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
        "constants": (),
        "roles": (
            ("input", value.memref_name),
            ("out", compact.result_memref_name),
            ("count", cell),
        ),
    }


def _selection_problem(kernel):
    return UPMEMPhysicalProblem.selection(kernel.geometry_value("elements"))


def _selection_plan(kernel, decision):
    if decision.predicate_lowering is not UPMEMPredicateLowering.CONDITIONAL_ZERO:
        raise UPMEMPlanBuilderRejected(
            "selection-flags source materializes conditional-zero only"
        )
    return UPMEMSelectionFlagsPlan(
        kernel.geometry_value("elements"),
        num_tasklets=decision.tasklets,
        dma_elements=decision.chunk_elements,
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
    "selection", _recognize_selection, _selection_plan, _selection_problem, _SELECTION_DOMAIN
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
    if list(x.indices) != [p, d] or list(y.indices) != [c, d] or list(acc.indices) != [p, c]:
        return None
    points, clusters, dimension = extents
    return {
        "geometry": (("points", points), ("dimension", dimension), ("clusters", clusters)),
        "anchor_op": "SQDIST_ACC",
        "constants": (),
        "roles": (("points", x.memref_name), ("centroids", y.memref_name), ("out", acc.memref_name)),
    }


def _kmeans_distances_plan(kernel, decision):
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
    """Argmin (``ARGMIN_IDX``, ``MIN_SEL``) plus centroid sums and counts
    (``SCATTER_ADD``, ``INC``): placed on the host, the paper partition.

    The rounded centroid division has no pattern (spec 004 D2): a host group
    runs its own MLIR, so no template consumes it.
    """
    del target
    names = {m.target_op_name for m in stream}
    if not {"MIN_SEL", "ARGMIN_IDX", "SCATTER_ADD", "INC"} <= names:
        return None
    return {
        "geometry": (),
        "anchor_op": "ARGMIN_IDX",
        "constants": (),
        "roles": (),
        "placement": "host",
    }


register_upmem_family(
    "kmeans_update", _recognize_kmeans_update, lambda kernel, decision: None, None, _SINGLE_DOMAIN
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
    if acc.indices[0] != f_iv or list(y.indices) != [s_iv] or set(loops) != {s_iv, f_iv}:
        return None
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
        shift = (-alpha).bit_length() - 1 if isinstance(alpha, int) and alpha < 0 else -1
        if not (0 <= shift <= 30 and alpha == -(2**shift) and isinstance(beta, int) and 0 <= beta <= 62):
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
    }


def _feature_gradient_plan(kernel, decision):
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

# Families whose plan overwrites its output: the source must zero it.
_ZERO_INIT_ROLES = {
    "histogram": ("out",),
    "selection": ("count",),
    "kmeans_distances": ("out",),
    "feature_gradient": ("out",),
}


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


def placement_extra(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> dict:
    return {
        "upmem_family": kernel.family,
        "upmem_geometry": kernel.geometry,
        "upmem_anchor_op": kernel.anchor_op,
        "upmem_constants": kernel.constants,
        "upmem_placement": kernel.placement,
        **decision_fields(decision),
    }


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




def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def _tasklet_block(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> int:
    """Consecutive elements one tasklet owns in the family plan."""
    if kernel.family == "matrix_vector":
        rows = kernel.geometry_value("batches") * kernel.geometry_value("rows")
        return _next_power_of_two(-(-rows // UPMEM_ACTIVE_TASKLETS))
    if kernel.family == "gemm":
        rows = kernel.geometry_value("rows")
        return _next_power_of_two(-(-rows // UPMEM_ACTIVE_TASKLETS))
    if kernel.family in ("pointwise", "reduction"):
        fused = decision.layout is UPMEMDataLayout.FUSED_REPLICATED
        chunk = (decision.dma_bytes // 2 if fused else decision.dma_bytes) // 4
        chunks = -(-kernel.geometry_value("elements") // chunk)
        return _next_power_of_two(-(-chunks // UPMEM_ACTIVE_TASKLETS)) * chunk
    work = {
        "histogram": "elements",
        "selection": "elements",
        "kmeans_distances": "points",
        "feature_gradient": "samples",
    }.get(kernel.family)
    if work is not None:
        return _next_power_of_two(-(-kernel.geometry_value(work) // UPMEM_ACTIVE_TASKLETS))
    if kernel.placement == "host":
        return 1
    raise NotImplementedError(f"no tasklet layout for UPMEM family {kernel.family!r}")


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
    )


def _legal(kernel: UpmemKernel, decision: UPMEMPhysicalDecision) -> bool:
    family = upmem_family(kernel.family)
    if family.features is None:
        return True
    try:
        physical_features(family.features(kernel), decision)
    except UPMEMPhysicalLegalityError:
        return False
    return True


def _legal_decisions(kernel: UpmemKernel) -> list:
    family = upmem_family(kernel.family)
    return sorted(
        (d for d in family.domain if _legal(kernel, d)),
        key=lambda d: d.deterministic_order_key,
    )


def upmem_base_placements(target, matches) -> list[Placement]:
    """One base placement per data layout with a legal decision (spec U5)."""
    kernel = classify_upmem_kernel(target, matches)
    layouts = sorted({d.layout.value for d in _legal_decisions(kernel)})
    return [
        Placement(
            placements={},
            mode=f"upmem:{kernel.family}:{layout}",
            extra={
                "upmem_family": kernel.family,
                "upmem_geometry": kernel.geometry,
                "upmem_anchor_op": kernel.anchor_op,
                "upmem_constants": kernel.constants,
                "upmem_placement": kernel.placement,
                "data_layout": layout,
            },
        )
        for layout in layouts
    ]


# Knob names in registration (cross) order, spec 001 D5.
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
    """A copy of ``base`` with ``extra[name] = value``; the last knob also
    attaches the plan family's masked 12-of-16 tasklet layout."""
    extra = dict(base.extra or {})
    extra[name] = value
    layout = base.layout
    if name == UPMEM_KNOBS[-1]:
        family = upmem_family(extra["upmem_family"])
        decision = decision_from_extra(family, extra)
        layout = masked_12_tasklet_linear_layout(
            _tasklet_block(kernel_from_extra(extra), decision)
        )
    return Placement(
        placements=dict(base.placements or {}),
        mode=f"{base.mode}+{name}={value}",
        extra=extra,
        layout=layout,
    )


def upmem_kernel_features(properties: Mapping) -> tuple | None:
    """``(name, int)`` pairs of ``UPMEMPhysicalFeatures`` for a placement, or
    None for a family the physical model does not cover (reduction)."""
    family = upmem_family(properties["upmem_family"])
    if family.features is None:
        return None
    kernel = kernel_from_extra(properties)
    features = physical_features(
        family.features(kernel), decision_from_extra(family, properties)
    )
    return tuple(features.manifest().items())


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
        return {"arith.addi": lhs + rhs, "arith.subi": lhs - rhs, "arith.muli": lhs * rhs}[
            name
        ]
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
        op for op in body
        if op.name in ("affine.store", "memref.store") and op.operands[1] == alloc
    ]
    loads = [
        op for op in body
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
        or (operand.memref_name is not None and operand.memref_name == match.result_memref_name)
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


def _check_zero_init(function, state, kernel) -> None:
    """The plan overwrites these outputs, so the source must zero them."""
    for role in _ZERO_INIT_ROLES.get(kernel.family, ()):
        name = kernel.role(role)
        zeroed = any(
            op.name in ("affine.store", "memref.store")
            and "to" in op.attributes
            and _attr_text(op.attributes["to"]) == name
            and _is_zero_constant(op.operands[0])
            for op in _walk(function.regions[0].blocks[0], [])
        )
        if not zeroed:
            raise NotImplementedError(
                f"accumulator {name} is not zero-initialized in the kernel; the "
                "UPMEM template overwrites it"
            )


def _union_windows(windows, role, name) -> "_Window":
    """The bounding window of several accesses to one argument."""
    windows = list(windows)
    if len({w.argument for w in windows}) != 1:
        raise NotImplementedError(f"UPMEM role {role!r} ({name}) spans several arguments")
    if len(windows) == 1:
        return windows[0]
    begin = [min(w.offsets[d] for w in windows) for d in range(len(windows[0].offsets))]
    end = [
        max(w.offsets[d] + w.extents[d] for w in windows)
        for d in range(len(windows[0].offsets))
    ]
    return _Window(windows[0].argument, tuple(begin), tuple(e - b for b, e in zip(begin, end)))


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
        x, label = samples[:, : plan.features], samples[:, plan.features : plan.features + 1]
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


def _i32(values) -> bytes:
    return np.asarray(values, dtype=np.int64).astype("<i4").tobytes()


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


def _i64(values) -> bytes:
    return np.asarray(values, dtype=np.int64).astype("<i8").tobytes()


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
                        values.append(rhs[index * plan.columns + column] if valid else 0)
    return values


def pack_inputs(plan, kernel: UpmemKernel, slices: Mapping) -> bytes:
    """Full MRAM heap image for one DPU and launch."""
    image = bytearray(int(plan.mram_image_bytes))
    if isinstance(plan, UPMEMMatrixVectorPlan):
        matrix, vector = _flat(slices["matrix"]), _flat(slices["vector"])
        if plan.vector_mode == "tasklet_private_wram":
            _write_region(image, _region(plan, "matrix"), _i32(matrix), "matrix")
            _write_region(image, _region(plan, "vector"), _i32(vector), "vector")
        else:
            _write_region(
                image,
                _region(plan, "packed_matrix_vector"),
                _i32(plan.pack_fused_inputs(matrix, vector)),
                "packed_matrix_vector",
            )
    elif isinstance(plan, UPMEMElementwisePlan):
        lhs, rhs = _flat(slices["lhs"]), _flat(slices["rhs"])
        if plan.interleaved:
            _write_region(
                image,
                _region(plan, "packed_operands"),
                _i32(_pack_fused_elementwise(plan, lhs, rhs)),
                "packed_operands",
            )
        else:
            _write_region(image, _region(plan, "x"), _i32(lhs), "x")
            _write_region(image, _region(plan, "y"), _i32(rhs), "y")
        if "coefficients" in plan.mram_regions:
            _write_region(
                image,
                _region(plan, "coefficients"),
                _i32((plan.alpha, plan.beta)),
                "coefficients",
            )
    elif isinstance(plan, UPMEMSumReductionPlan):
        values = _flat(slices["input"])
        padded = values + [0] * (plan.padded_elements - len(values))
        _write_region(image, _region(plan, "input"), _i32(padded), "input")
    elif isinstance(plan, UPMEMGEMMPlan):
        _write_region(
            image,
            _region(plan, "packed_lhs_rhs"),
            _i32(_pack_fused_gemm(plan, _flat(slices["lhs"]), _flat(slices["rhs"]))),
            "packed_lhs_rhs",
        )
    elif isinstance(plan, (UPMEMHistogramPlan, UPMEMSelectionFlagsPlan)):
        _write_region(image, _region(plan, "input"), _i32(_flat(slices["input"])), "input")
    elif isinstance(plan, UPMEMKMeansDistancesPlan):
        fused = plan.pack_fused_pairs(_flat(slices["points"]), _flat(slices["centroids"]))
        _write_region(image, _region(plan, "fused_pairs"), _i32(fused), "fused_pairs")
    elif isinstance(plan, UPMEMFeatureGradientPlan):
        _write_region(image, _region(plan, "samples"), _i32(_flat(slices["samples"])), "samples")
    else:
        raise NotImplementedError(f"no UPMEM input packer for {type(plan).__name__}")
    return bytes(image)


def pack_expected(plan, kernel: UpmemKernel, slices: Mapping) -> tuple[int, bytes]:
    """``(offset, bytes)`` of the exact output window for one DPU and launch."""
    image = bytearray(int(plan.mram_image_bytes))
    if isinstance(plan, UPMEMSelectionFlagsPlan):
        # The source never materializes flags; they come from the plan's own
        # reference. The end-to-end check against the oracle runs after the
        # host compaction (``_selection_epilogue``).
        flags = plan.reference_flags(_flat(slices["input"]))
        _write_region(image, _region(plan, "flags"), _i32(flags), "flags")
        begin, end = _output_window(plan)
        return begin, bytes(image[begin:end])
    out = _flat(slices["out"])
    if isinstance(plan, UPMEMHistogramPlan):
        _write_region(image, _region(plan, "histogram"), _i32(out), "histogram")
        begin, end = _output_window(plan)
        return begin, bytes(image[begin:end])
    if isinstance(plan, UPMEMKMeansDistancesPlan):
        _write_region(image, _region(plan, "distances"), _i64(out), "distances")
        begin, end = _output_window(plan)
        return begin, bytes(image[begin:end])
    if isinstance(plan, UPMEMFeatureGradientPlan):
        _write_region(image, _region(plan, "gradient"), _i64(out), "gradient")
        begin, end = _output_window(plan)
        return begin, bytes(image[begin:end])
    if isinstance(plan, UPMEMMatrixVectorPlan):
        data = _i32(plan.pack_output(out))
    elif isinstance(plan, UPMEMElementwisePlan):
        data = _i32(out + [0] * (plan.padded_elements - len(out)))
    elif isinstance(plan, UPMEMSumReductionPlan):
        data = _i64(out)
    elif isinstance(plan, UPMEMGEMMPlan):
        physical = [0] * (plan.padded_rows * plan.padded_columns)
        for row in range(plan.rows):
            begin = row * plan.padded_columns
            physical[begin : begin + plan.columns] = out[
                row * plan.columns : (row + 1) * plan.columns
            ]
        data = _i32(physical)
    else:
        raise NotImplementedError(f"no UPMEM output packer for {type(plan).__name__}")
    _write_region(image, _region(plan, "output"), data, "output")
    begin, end = _output_window(plan)
    return begin, bytes(image[begin:end])


def gather_outputs(plan, kernel: UpmemKernel, window: bytes) -> dict:
    """Decode one DPU's output window into ``{role: logical values}``."""
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
        return {"flags": np.frombuffer(window, dtype="<i4").astype(np.int64)[: plan.elements]}
    if isinstance(plan, (UPMEMKMeansDistancesPlan, UPMEMFeatureGradientPlan)):
        return {"out": np.frombuffer(window, dtype="<i8").astype(np.int64)}
    raise NotImplementedError(f"no UPMEM output gather for {type(plan).__name__}")


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


class UpmemCtx(CodegenContext):
    """Lower matched UPMEM kernel groups to calibrated physical plans."""

    def __init__(self, target):
        super().__init__(target)
        self.segments: tuple = ()
        self.groups: dict = {}
        self.arity: int | None = None

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
            for (scope, _matches), layout in zip(_bucket_for_autoschedule(trace), layouts)
        }
        by_group: dict[int, list[MatchedOp]] = {}
        for match in trace.matches:
            by_group.setdefault(_matcher_work_scope(match).group_id, []).append(match)

        groups = {}
        for group_id, matches in by_group.items():
            kernel = classify_upmem_kernel(self.target, matches)
            family = upmem_family(kernel.family)
            decisions = {
                decision_from_extra(family, layout_by_scope[scope].extra)
                for scope in {_matcher_search_scope(match) for match in matches}
                if scope in layout_by_scope
            }
            if len(decisions) != 1:
                raise ValueError(
                    f"UPMEM kernel group {group_id} carries {len(decisions)} "
                    "physical decisions; one is required"
                )
            (decision,) = decisions
            try:
                if family.features is not None:
                    physical_features(family.features(kernel), decision)
                plan = family.plan_for(kernel, decision)
                device_source = "" if plan is None else plan.device_source()
            except (UPMEMPlanBuilderRejected, UPMEMPhysicalLegalityError, ValueError) as error:
                raise InfeasibleSchedule(
                    f"UPMEM {kernel.family} decision {decision.manifest()} is not "
                    f"realizable: {error}"
                ) from error
            functions = sorted(
                {m.func_name: _matcher_work_scope(m).work_id for m in matches}.items(),
                key=lambda item: item[1],
            )
            groups[group_id] = _GroupProgram(
                kernel,
                decision,
                plan,
                [name for name, _ in functions],
                device_source,
                {},
                {},
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
                if segments and segments[-1] is not None and segments[-1][0] == step.group_id:
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
        self.cmds = [
            (
                f"segment {index} {segment.kernel.family} host "
                f"executions={segment.executions}"
                if segment.kind == "host"
                else f"segment {index} {segment.kernel.family} dpus={segment.kernel.dpus} "
                f"executions={segment.executions} tasklets={segment.decision.tasklets} "
                f"source_sha256={_sha256(segment.device_source.encode())}"
            )
            for index, segment in enumerate(self.segments)
        ]

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
                    artifact = emit_c_from_mlir(module, name, wrap_wide_integers=True)
                    program.oracles[name] = (artifact.compile(), abi)
                    arity = max(arity, max(abi, default=-1) + 1)
                    continue
                _check_coverage(function, state, own)
                _check_zero_init(function, state, program.kernel)
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
                                m for m in own if m.target_op_name == program.kernel.anchor_op
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
                            ivs
                            if ivs is not None
                            else [_value_name(v, state) for v in list(op.operands)[1:]],
                        )
                        for op in accesses
                    }
                    if not found:
                        raise NotImplementedError(
                            f"UPMEM role {role!r} ({memref_name}) in {name} has no access"
                        )
                    windows[role] = _union_windows(found, role, memref_name)
                if any(value >= _LOCAL_ABI_ID for value in abi):
                    raise NotImplementedError(
                        f"{name} reads a region-local buffer; the UPMEM oracle binds "
                        "only caller arguments"
                    )
                arity = max(arity, max(abi, default=-1) + 1)
                program.windows[name] = {
                    role: _Window(abi[w.argument], w.offsets, w.extents)
                    for role, w in windows.items()
                }
                artifact = emit_c_from_mlir(module, name, wrap_wide_integers=True)
                program.oracles[name] = (artifact.compile(), abi)
        self.arity = arity

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
            raise TypeError(f"UPMEM program takes {self.arity} arrays, got {len(values)}")
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
                        r: windows[r].slice(before[windows[r].argument])
                        for r in _INPUT_ROLES[family]
                    }
                    _check_preconditions(segment.plan, segment.kernel, slices_in)
                    heap = pack_inputs(segment.plan, segment.kernel, slices_in)
                    offset, oracle = pack_expected(
                        segment.plan,
                        segment.kernel,
                        {
                            **slices_in,
                            **{
                                r: windows[r].slice(arrays[windows[r].argument])
                                for r in _OUTPUT_ROLES[family]
                                if r in windows
                            },
                        },
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
                (_indexed_name("mram.input", i), blob) for i, blob in enumerate(inputs_blobs)
            ) + tuple(
                (_indexed_name("mram.output.expected", i), blob)
                for i, blob in enumerate(expected_blobs)
            )
            fixtures.append(UpmemFixture(manifest, files))
            expected.append(outputs)
            states.append([array.copy() for array in arrays])
        return UpmemBundle(tuple(fixtures), tuple(expected), tuple(states))


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


def _selection_epilogue(segment, windows, flags, arrays, oracle_state) -> None:
    """Host stable compaction of the observed DPU flags (spec 004 D1).

    Generated from the ``COMPACT`` and ``INC`` matches: keep every flag value
    whose low bit is set, in index order, into ``out[0:k]`` and set the counter
    cell to ``k``; ``out[k:]`` is untouched, as in the source. The flags form
    (``SELECT_ODD``) writes the flags to its output instead. The result must
    equal the functional oracle's, or the run raises.
    """
    flags = np.asarray(flags, dtype=np.int64)
    out_window = windows["out"]
    out = out_window.slice(arrays[out_window.argument]).reshape(-1)
    if not segment.kernel.geometry_value("compact"):
        out[...] = flags.astype(out.dtype)
        return
    kept = flags[(flags & 1) != 0]
    out[: kept.size] = kept.astype(out.dtype)
    count_window = windows["count"]
    count = count_window.slice(arrays[count_window.argument]).reshape(-1)
    count[0] = kept.size
    want_out = out_window.slice(oracle_state[out_window.argument]).reshape(-1)
    want_count = count_window.slice(oracle_state[count_window.argument]).reshape(-1)
    if int(want_count[0]) != kept.size or not np.array_equal(out[: kept.size], want_out[: kept.size]):
        raise RuntimeError(
            "UPMEM selection: host compaction of the simulated flags differs from "
            f"the functional oracle (count {kept.size} vs {int(want_count[0])})"
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
    image = spmw_simenv.upmem_sdk_image()
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
                    "logic_cycles": 0,
                    "timing_scope": _TIMING_SCOPE,
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
        fixture_dir, bin_dir, log_dir = run_dir / "fixture", run_dir / "bin", run_dir / "log"
        fixture.write(fixture_dir)
        bin_dir.mkdir(parents=True)
        tasklets = segment.decision.tasklets
        dpus = segment.kernel.dpus
        build = [
            "docker", "run", "--rm", "-v", f"{root}:/root/uPIMulator", image,
            "python3", "/root/uPIMulator/benchmark/build.py",
            "--num_dpus", "1", "--num_tasklets", str(tasklets), "--benchmark", slot,
        ]
        simulate = [
            str(binary), "--benchmark", slot, "--fixture_dir", str(fixture_dir),
            "--num_channels", "1", "--num_ranks_per_channel", "1",
            "--num_dpus_per_rank", str(dpus), "--num_tasklets", str(tasklets),
            "--num_simulation_threads", "16", "--logic_frequency", "350",
            "--skip_compilation", "true", "--root_dirpath", str(root),
            "--bin_dirpath", str(bin_dir), "--log_dirpath", str(log_dir),
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
            if sdk_sha != spmw_simenv.UPMEM_SDK_BUILD_MANIFEST_SHA256:
                raise RuntimeError("UPMEM SDK build drifted from the pinned campaign build")
            task_object = (
                root / "benchmark" / "build" / slot / "dpu" / "CMakeFiles"
                / f"{slot}_device.dir" / "task.c.o"
            )
            task_object_sha = _sha256(task_object.read_bytes())
            simulated = subprocess.run(
                simulate, capture_output=True, text=True, timeout=_SIMULATE_TIMEOUT_S
            )
        (run_dir / "simulate.log").write_text(simulated.stdout + simulated.stderr)
        log_path = bin_dir / "log.txt"
        if simulated.returncode != 0:
            mismatch = _first_mismatch(segment, expected, bin_dir)
            raise RuntimeError(
                f"uPIMulator failed on segment {index} (exit {simulated.returncode})"
                + (f"; first mismatch: {mismatch}" if mismatch else "")
                + "\n"
                + "\n".join((simulated.stdout + simulated.stderr).splitlines()[-40:])
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
                values = gather_outputs(segment.plan, segment.kernel, observed.read_bytes())
                if "flags" in values:
                    _selection_epilogue(
                        segment, program.windows[name], values["flags"], caller_arrays,
                        bundle.states[index],
                    )
                    continue
                for role, data in values.items():
                    window = program.windows[name][role]
                    destination = caller_arrays[window.argument]
                    view = window.slice(destination)
                    view[...] = np.asarray(data).reshape(view.shape).astype(view.dtype)

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
                "commands": {"compile": build, "simulate": simulate},
                "num_dpus": dpus,
                "num_tasklets": tasklets,
                "executions": segment.executions,
                "logic_cycles_per_dpu": {
                    f"{c}_{r}_{d}": cycles for (c, r, d), cycles in sorted(per_dpu.items())
                },
                "logic_cycles": segment_cycles,
                "timing_scope": _TIMING_SCOPE,
            }
        )
    return RunResult(
        cycles=total,
        stdout="".join(stdout_parts),
        backend="upmem",
        extra={"upmem": records, "timing_scope": _TIMING_SCOPE},
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
