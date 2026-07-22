# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic physical schedule search for one UPMEM DPU.

The older generic UPMEM search only varies the tasklet count.  This module
models the decisions that materially change emitted DPU C: MRAM transaction
size, separate versus compiler-fused storage, operand residency, and
contraction tiles.  It is intentionally independent of a particular plan
class.  A caller may pass a ``plan_builder`` closure once a physical lowering
is available.

The default model is calibrated to a small, named set of *uPIMulator*
observations.  It is not a hardware model.  For a represented decision
stratum, an analytical physical-feature estimate is multiplied by the mean
observed/analytical ratio.  Unrepresented strata retain the analytical
estimate and are marked as extrapolations.  This makes every prediction and
every in-sample correction auditable without pretending that simulator
measurements calibrate a real UPMEM DIMM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import itertools
import json
import math
import struct
from typing import Any


SEARCH_SCHEMA = "upmem-physical-search-v1"
MODEL_SCHEMA = "upmem-upimulator-physical-cost-v1"
ESTIMATE_SCHEMA = "upmem-physical-cost-estimate-v1"
EVIDENCE_SCHEMA = "upmem-physical-calibration-evidence-v1"
MMTV_ROW_LAYOUT_SEARCH_SCHEMA = "upmem-mmtv-row-layout-search-v1"
MMTV_ROW_LAYOUT_EVIDENCE_SCHEMA = "upmem-mmtv-row-layout-evidence-v1"

UPIMULATOR_BINARY_SHA256 = (
    "0ab6af912dc641d0cf54551c64eeabbae9bd50aa9fc71f25922b7832dcd3563a"
)
UPMEM_MAX_TASKLETS = 24
UPMEM_WRAM_BYTES = 64 * 1024
UPMEM_MAX_DMA_BYTES = 2048
UPMEM_DMA_ALIGNMENT = 8
UPMEM_REVOLVER_CYCLES = 11
UPMEM_PIPELINE_FILL_CYCLES = 13

UPMEM_MMTV_CANONICAL_MATRIX_SHA256 = (
    "2e16f1d967e9f05e00f5968346d51edbd3b992e123b048208101c13dc8f5ae77"
)
UPMEM_MMTV_CANONICAL_VECTORS_SHA256 = (
    "50f73abaea041fdabc4cbe6e1b6ddadf6e3df1531cd7caf5ef46cdc5c04d9880"
)
UPMEM_MMTV_ROW_LAYOUT_PHASES = (0, 1, 15, 2, 3, 5, 7, 9)

# These are analytical proxy constants, not fitted hardware latencies.  Their
# provenance is recorded in ``UPMEMPhysicalCostModel.manifest``.  Calibration
# only scales the resulting component vector by an observed simulator ratio.
ANALYTICAL_DMA_BYTES_PER_CYCLE = 8
ANALYTICAL_DMA_CALL_CYCLES = 24
ANALYTICAL_BARRIER_CYCLES = 256
ANALYTICAL_SCHEDULER_CYCLES_PER_TASKLET = 4
ANALYTICAL_SHARED_WRAM_ACCESS_DIVISOR = 4

_LOGIC_BREAKDOWN_COUNTERS = (
    "breakdown_run",
    "breakdown_dma",
    "breakdown_etc",
    "backpressure",
)
_MRAM_COUNTERS = (
    "mram_read_units",
    "mram_write_units",
    "mram_read_bytes",
    "mram_write_bytes",
    "mram_activations",
    "mram_precharges",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else str(value.value)


class UPMEMKernelKind(str, Enum):
    """Physical workload families currently understood by the estimator."""

    POINTWISE = "pointwise"
    SELECTION = "selection_flags"
    MATRIX_VECTOR = "matrix_vector"
    GEMM = "gemm"


class UPMEMDataLayout(str, Enum):
    """MRAM storage relationship between operands consumed together."""

    SEPARATE = "separate"
    FUSED_REPLICATED = "fused_replicated"


class UPMEMOwnership(str, Enum):
    """Tasklet ownership policy.

    ``CYCLIC`` is represented so unsupported candidates can be rejected and
    reported rather than silently reinterpreted as contiguous ownership.
    """

    CONTIGUOUS = "contiguous"
    CYCLIC = "cyclic"


class UPMEMOperandResidency(str, Enum):
    """Where a reused vector or GEMM right-hand side is obtained."""

    NOT_APPLICABLE = "not_applicable"
    STREAMED_MRAM = "streamed_mram"
    SHARED_WRAM = "shared_wram"
    TASKLET_PRIVATE_WRAM = "tasklet_private_wram"
    FUSED_DMA_PACKET = "fused_dma_packet"


class UPMEMPredicateLowering(str, Enum):
    """Device-C lowering choices for a selection predicate."""

    NOT_APPLICABLE = "not_applicable"
    TERNARY = "ternary"
    BRANCHLESS_MASK = "branchless_mask"
    CONDITIONAL_ZERO = "conditional_zero"


@dataclass(frozen=True)
class MaskedF2TaskletLayout:
    """Power-of-two linear-layout carrier with an explicit active mask."""

    physical_tasklets: int = 12

    def __post_init__(self) -> None:
        tasklets = _positive_integer(self.physical_tasklets, "physical_tasklets")
        if tasklets > UPMEM_MAX_TASKLETS:
            raise ValueError(f"UPMEM supports at most {UPMEM_MAX_TASKLETS} tasklets")

    @property
    def padded_tasklet_extent(self) -> int:
        return 1 << (self.physical_tasklets - 1).bit_length()

    @property
    def active_mask(self) -> int:
        return (1 << self.physical_tasklets) - 1

    @property
    def active_tasklets(self) -> tuple[int, ...]:
        return tuple(range(self.physical_tasklets))

    @property
    def masked_tasklets(self) -> tuple[int, ...]:
        return tuple(range(self.physical_tasklets, self.padded_tasklet_extent))

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "linear-layout-f2",
            "policy": "masked-padded-power-of-two",
            "physical_tasklets": self.physical_tasklets,
            "padded_tasklet_extent": self.padded_tasklet_extent,
            "active_tasklets": list(self.active_tasklets),
            "masked_tasklets": list(self.masked_tasklets),
            "active_mask": self.active_mask,
        }


@dataclass(frozen=True)
class UPMEMPhysicalProblem:
    """A finite physical search problem for a single DPU launch."""

    kind: UPMEMKernelKind
    logical_elements: int = 0
    rows: int = 0
    batches: int = 1
    columns: int = 0
    reduction: int = 0
    input_operands: int = 2
    element_bytes: int = 4
    output_element_bytes: int = 4
    accumulator_bytes: int = 8
    operation_instructions: int = 1
    wram_limit_bytes: int = UPMEM_WRAM_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.kind, UPMEMKernelKind):
            raise TypeError("kind must be a UPMEMKernelKind")
        for name in (
            "element_bytes",
            "output_element_bytes",
            "accumulator_bytes",
            "operation_instructions",
            "wram_limit_bytes",
        ):
            _positive_integer(getattr(self, name), name)
        if self.kind in (UPMEMKernelKind.POINTWISE, UPMEMKernelKind.SELECTION):
            _positive_integer(self.logical_elements, "logical_elements")
            _positive_integer(self.input_operands, "input_operands")
        elif self.kind is UPMEMKernelKind.MATRIX_VECTOR:
            _positive_integer(self.rows, "rows")
            _positive_integer(self.batches, "batches")
            _positive_integer(self.reduction, "reduction")
            if self.rows % self.batches:
                raise ValueError(
                    "matrix-vector flattened rows must divide evenly into batches"
                )
        elif self.kind is UPMEMKernelKind.GEMM:
            _positive_integer(self.rows, "rows")
            _positive_integer(self.columns, "columns")
            _positive_integer(self.reduction, "reduction")

    @classmethod
    def pointwise(
        cls,
        logical_elements: int,
        *,
        input_operands: int = 2,
        operation_instructions: int = 1,
        element_bytes: int = 4,
        output_element_bytes: int = 4,
        wram_limit_bytes: int = UPMEM_WRAM_BYTES,
    ) -> "UPMEMPhysicalProblem":
        return cls(
            UPMEMKernelKind.POINTWISE,
            logical_elements=logical_elements,
            input_operands=input_operands,
            operation_instructions=operation_instructions,
            element_bytes=element_bytes,
            output_element_bytes=output_element_bytes,
            wram_limit_bytes=wram_limit_bytes,
        )

    @classmethod
    def matrix_vector(
        cls,
        rows: int,
        reduction: int,
        *,
        batches: int = 1,
        operation_instructions: int = 33,
        element_bytes: int = 4,
        output_element_bytes: int = 4,
        accumulator_bytes: int = 8,
        wram_limit_bytes: int = UPMEM_WRAM_BYTES,
    ) -> "UPMEMPhysicalProblem":
        return cls(
            UPMEMKernelKind.MATRIX_VECTOR,
            rows=rows,
            batches=batches,
            reduction=reduction,
            operation_instructions=operation_instructions,
            element_bytes=element_bytes,
            output_element_bytes=output_element_bytes,
            accumulator_bytes=accumulator_bytes,
            wram_limit_bytes=wram_limit_bytes,
        )

    @classmethod
    def selection(
        cls,
        logical_elements: int,
        *,
        operation_instructions: int = 2,
        element_bytes: int = 4,
        output_element_bytes: int = 4,
        wram_limit_bytes: int = UPMEM_WRAM_BYTES,
    ) -> "UPMEMPhysicalProblem":
        """Build the archived stable-selection device-flags partition."""

        return cls(
            UPMEMKernelKind.SELECTION,
            logical_elements=logical_elements,
            input_operands=1,
            operation_instructions=operation_instructions,
            element_bytes=element_bytes,
            output_element_bytes=output_element_bytes,
            wram_limit_bytes=wram_limit_bytes,
        )

    @classmethod
    def gemm(
        cls,
        rows: int,
        columns: int,
        reduction: int,
        *,
        operation_instructions: int = 33,
        element_bytes: int = 4,
        output_element_bytes: int = 4,
        accumulator_bytes: int = 8,
        wram_limit_bytes: int = UPMEM_WRAM_BYTES,
    ) -> "UPMEMPhysicalProblem":
        return cls(
            UPMEMKernelKind.GEMM,
            rows=rows,
            columns=columns,
            reduction=reduction,
            operation_instructions=operation_instructions,
            element_bytes=element_bytes,
            output_element_bytes=output_element_bytes,
            accumulator_bytes=accumulator_bytes,
            wram_limit_bytes=wram_limit_bytes,
        )

    @property
    def logical_work(self) -> int:
        if self.kind in (UPMEMKernelKind.POINTWISE, UPMEMKernelKind.SELECTION):
            return self.logical_elements
        if self.kind is UPMEMKernelKind.MATRIX_VECTOR:
            return self.rows * self.reduction
        return self.rows * self.columns * self.reduction

    def manifest(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "logical_elements": self.logical_elements,
            "rows": self.rows,
            "batches": self.batches,
            "columns": self.columns,
            "reduction": self.reduction,
            "input_operands": self.input_operands,
            "element_bytes": self.element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "accumulator_bytes": self.accumulator_bytes,
            "operation_instructions": self.operation_instructions,
            "wram_limit_bytes": self.wram_limit_bytes,
            "logical_work": self.logical_work,
        }


@dataclass(frozen=True)
class UPMEMPhysicalDecision:
    """One fully specified physical UPMEM schedule decision."""

    layout: UPMEMDataLayout
    dma_bytes: int
    chunk_elements: int
    ownership: UPMEMOwnership = UPMEMOwnership.CONTIGUOUS
    tasklets: int = 12
    vector_residency: UPMEMOperandResidency = UPMEMOperandResidency.NOT_APPLICABLE
    rhs_residency: UPMEMOperandResidency = UPMEMOperandResidency.NOT_APPLICABLE
    nc: int | None = None
    kc: int | None = None
    predicate_lowering: UPMEMPredicateLowering = UPMEMPredicateLowering.NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not isinstance(self.layout, UPMEMDataLayout):
            raise TypeError("layout must be a UPMEMDataLayout")
        if not isinstance(self.ownership, UPMEMOwnership):
            raise TypeError("ownership must be a UPMEMOwnership")
        if not isinstance(self.vector_residency, UPMEMOperandResidency):
            raise TypeError("vector_residency must be a UPMEMOperandResidency")
        if not isinstance(self.rhs_residency, UPMEMOperandResidency):
            raise TypeError("rhs_residency must be a UPMEMOperandResidency")
        if not isinstance(self.predicate_lowering, UPMEMPredicateLowering):
            raise TypeError("predicate_lowering must be a UPMEMPredicateLowering")
        _positive_integer(self.dma_bytes, "dma_bytes")
        _positive_integer(self.chunk_elements, "chunk_elements")
        _positive_integer(self.tasklets, "tasklets")
        if self.nc is not None:
            _positive_integer(self.nc, "nc")
        if self.kc is not None:
            _positive_integer(self.kc, "kc")

    @property
    def tasklet_layout(self) -> MaskedF2TaskletLayout:
        return MaskedF2TaskletLayout(self.tasklets)

    @property
    def dma_elements(self) -> int:
        """Logical elements handled by one tasklet DMA chunk."""

        return self.chunk_elements

    def manifest(self) -> dict[str, object]:
        return {
            "layout": self.layout.value,
            "dma_bytes": self.dma_bytes,
            "chunk_elements": self.chunk_elements,
            "ownership": self.ownership.value,
            "tasklets": self.tasklets,
            "tasklet_layout": self.tasklet_layout.manifest(),
            "vector_residency": self.vector_residency.value,
            "rhs_residency": self.rhs_residency.value,
            "nc": self.nc,
            "kc": self.kc,
            "predicate_lowering": self.predicate_lowering.value,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.manifest())

    @property
    def deterministic_order_key(self) -> tuple[object, ...]:
        return (
            self.layout.value,
            self.dma_bytes,
            self.chunk_elements,
            self.ownership.value,
            self.tasklets,
            self.vector_residency.value,
            self.rhs_residency.value,
            -1 if self.nc is None else self.nc,
            -1 if self.kc is None else self.kc,
            self.predicate_lowering.value,
        )


def _domain_tuple(values: Iterable[Any], name: str) -> tuple[Any, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} decision domain must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} decision domain contains duplicate values")
    return result


@dataclass(frozen=True)
class UPMEMPhysicalDecisionDomain:
    """Ordered axes for a deterministic exhaustive Cartesian search."""

    layouts: tuple[UPMEMDataLayout, ...]
    dma_bytes: tuple[int, ...]
    chunk_elements: tuple[int, ...]
    ownerships: tuple[UPMEMOwnership, ...] = (UPMEMOwnership.CONTIGUOUS,)
    tasklets: tuple[int, ...] = (12,)
    vector_residencies: tuple[UPMEMOperandResidency, ...] = (
        UPMEMOperandResidency.NOT_APPLICABLE,
    )
    rhs_residencies: tuple[UPMEMOperandResidency, ...] = (
        UPMEMOperandResidency.NOT_APPLICABLE,
    )
    nc_tiles: tuple[int | None, ...] = (None,)
    kc_tiles: tuple[int | None, ...] = (None,)
    predicate_lowerings: tuple[UPMEMPredicateLowering, ...] = (
        UPMEMPredicateLowering.NOT_APPLICABLE,
    )

    def __post_init__(self) -> None:
        for name in (
            "layouts",
            "dma_bytes",
            "chunk_elements",
            "ownerships",
            "tasklets",
            "vector_residencies",
            "rhs_residencies",
            "nc_tiles",
            "kc_tiles",
            "predicate_lowerings",
        ):
            object.__setattr__(self, name, _domain_tuple(getattr(self, name), name))

    @property
    def assignment_count(self) -> int:
        return math.prod(
            len(getattr(self, name))
            for name in (
                "layouts",
                "dma_bytes",
                "chunk_elements",
                "ownerships",
                "tasklets",
                "vector_residencies",
                "rhs_residencies",
                "nc_tiles",
                "kc_tiles",
                "predicate_lowerings",
            )
        )

    def decisions(self) -> tuple[UPMEMPhysicalDecision, ...]:
        axes = (
            self.layouts,
            self.dma_bytes,
            self.chunk_elements,
            self.ownerships,
            self.tasklets,
            self.vector_residencies,
            self.rhs_residencies,
            self.nc_tiles,
            self.kc_tiles,
            self.predicate_lowerings,
        )
        return tuple(
            UPMEMPhysicalDecision(*assignment)
            for assignment in itertools.product(*axes)
        )

    def manifest(self) -> dict[str, object]:
        def values(items: tuple[Any, ...]) -> list[object]:
            return [
                _enum_value(item) if isinstance(item, Enum) else item for item in items
            ]

        return {
            "layouts": values(self.layouts),
            "dma_bytes": values(self.dma_bytes),
            "chunk_elements": values(self.chunk_elements),
            "ownerships": values(self.ownerships),
            "tasklets": values(self.tasklets),
            "vector_residencies": values(self.vector_residencies),
            "rhs_residencies": values(self.rhs_residencies),
            "nc_tiles": values(self.nc_tiles),
            "kc_tiles": values(self.kc_tiles),
            "predicate_lowerings": values(self.predicate_lowerings),
            "assignment_count": self.assignment_count,
        }


class UPMEMPhysicalLegalityError(ValueError):
    """Raised when a decision cannot be emitted faithfully for the problem."""


class UPMEMPlanBuilderRejected(ValueError):
    """A plan builder can raise this to reject a lowering fail-closed."""


class UPMEMNoLegalPhysicalPlan(RuntimeError):
    """Raised when exhaustive search produces no legal materialization."""

    def __init__(self, result: "UPMEMPhysicalSearchResult") -> None:
        self.result = result
        reasons = "; ".join(rejection.reason for rejection in result.rejected[:3])
        super().__init__(
            "no legal UPMEM physical plan" + (f": {reasons}" if reasons else "")
        )


@dataclass(frozen=True)
class UPMEMPhysicalFeatures:
    """Auditable physical quantities consumed by the cycle estimator."""

    logical_work: int
    dynamic_instructions: int
    mram_read_bytes: int
    mram_write_bytes: int
    mram_read_calls: int
    mram_write_calls: int
    barrier_count: int
    shared_wram_accesses: int
    replicated_mram_bytes: int
    backpressure_proxy_units: int
    wram_bytes: int

    def manifest(self) -> dict[str, int]:
        return {
            "logical_work": self.logical_work,
            "dynamic_instructions": self.dynamic_instructions,
            "mram_read_bytes": self.mram_read_bytes,
            "mram_write_bytes": self.mram_write_bytes,
            "mram_read_calls": self.mram_read_calls,
            "mram_write_calls": self.mram_write_calls,
            "barrier_count": self.barrier_count,
            "shared_wram_accesses": self.shared_wram_accesses,
            "replicated_mram_bytes": self.replicated_mram_bytes,
            "backpressure_proxy_units": self.backpressure_proxy_units,
            "wram_bytes": self.wram_bytes,
        }


def _require_common_legality(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> None:
    if decision.tasklets > UPMEM_MAX_TASKLETS:
        raise UPMEMPhysicalLegalityError(
            f"tasklets={decision.tasklets} exceeds the UPMEM limit of "
            f"{UPMEM_MAX_TASKLETS}"
        )
    if decision.ownership is not UPMEMOwnership.CONTIGUOUS:
        raise UPMEMPhysicalLegalityError(
            "only proven contiguous tasklet ownership is supported"
        )
    if decision.dma_bytes % UPMEM_DMA_ALIGNMENT:
        raise UPMEMPhysicalLegalityError("MRAM DMA bytes must be 8-byte aligned")
    if decision.dma_bytes > UPMEM_MAX_DMA_BYTES:
        raise UPMEMPhysicalLegalityError(
            f"MRAM DMA bytes exceed the {UPMEM_MAX_DMA_BYTES}-byte supported maximum"
        )
    MaskedF2TaskletLayout(decision.tasklets)
    if problem.element_bytes % 4 or problem.output_element_bytes % 4:
        raise UPMEMPhysicalLegalityError(
            "the physical search currently supports 32-bit-multiple element widths"
        )


def _pointwise_features(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> UPMEMPhysicalFeatures:
    if decision.predicate_lowering is not UPMEMPredicateLowering.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "pointwise plans cannot assign predicate lowering"
        )
    if decision.vector_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "pointwise plans cannot assign vector residency"
        )
    if decision.rhs_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError("pointwise plans cannot assign RHS residency")
    if decision.nc is not None or decision.kc is not None:
        raise UPMEMPhysicalLegalityError("pointwise plans cannot assign GEMM tiles")
    chunk_bytes = decision.chunk_elements * problem.element_bytes
    expected_read_dma = chunk_bytes
    if decision.layout is UPMEMDataLayout.FUSED_REPLICATED:
        expected_read_dma *= problem.input_operands
    if decision.dma_bytes != expected_read_dma:
        raise UPMEMPhysicalLegalityError(
            "DMA bytes do not match the selected pointwise chunk/layout"
        )
    if chunk_bytes > UPMEM_MAX_DMA_BYTES:
        raise UPMEMPhysicalLegalityError("pointwise output DMA exceeds 2048 bytes")
    if problem.logical_elements % decision.chunk_elements:
        raise UPMEMPhysicalLegalityError(
            "pointwise logical extent must be divisible by chunk_elements"
        )
    chunks = problem.logical_elements // decision.chunk_elements
    if decision.layout is UPMEMDataLayout.SEPARATE:
        read_calls = chunks * problem.input_operands
        read_buffer_bytes = decision.dma_bytes * problem.input_operands
    else:
        read_calls = chunks
        read_buffer_bytes = decision.dma_bytes
    write_calls = chunks
    read_bytes = (
        problem.logical_elements * problem.element_bytes * problem.input_operands
    )
    write_bytes = problem.logical_elements * problem.output_element_bytes
    wram = decision.tasklets * (
        read_buffer_bytes + decision.chunk_elements * problem.output_element_bytes
    )
    oversized = read_calls * max(0, decision.dma_bytes - 128)
    oversized += write_calls * max(0, chunk_bytes - 128)
    backpressure = oversized + max(0, read_calls + write_calls - 768) * 8
    return UPMEMPhysicalFeatures(
        logical_work=problem.logical_work,
        dynamic_instructions=problem.logical_work * problem.operation_instructions,
        mram_read_bytes=read_bytes,
        mram_write_bytes=write_bytes,
        mram_read_calls=read_calls,
        mram_write_calls=write_calls,
        barrier_count=1,
        shared_wram_accesses=0,
        replicated_mram_bytes=0,
        backpressure_proxy_units=backpressure,
        wram_bytes=wram,
    )


def _selection_features(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> UPMEMPhysicalFeatures:
    if decision.layout is not UPMEMDataLayout.SEPARATE:
        raise UPMEMPhysicalLegalityError(
            "selection-flags lowering requires a separate input/output layout"
        )
    if decision.vector_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "selection plans cannot assign vector residency"
        )
    if decision.rhs_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError("selection plans cannot assign RHS residency")
    if decision.nc is not None or decision.kc is not None:
        raise UPMEMPhysicalLegalityError("selection plans cannot assign GEMM tiles")
    if decision.predicate_lowering is UPMEMPredicateLowering.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "selection plans require an explicit predicate lowering"
        )
    expected_dma = decision.chunk_elements * problem.element_bytes
    if decision.dma_bytes != expected_dma:
        raise UPMEMPhysicalLegalityError(
            "selection DMA bytes must equal dma_elements * element_bytes"
        )
    if problem.logical_elements % decision.chunk_elements:
        raise UPMEMPhysicalLegalityError(
            "selection logical extent must be divisible by dma_elements"
        )
    chunks = problem.logical_elements // decision.chunk_elements
    read_bytes = problem.logical_elements * problem.element_bytes
    write_bytes = problem.logical_elements * problem.output_element_bytes
    wram = decision.tasklets * (
        decision.dma_bytes + decision.chunk_elements * problem.output_element_bytes
    )
    lowering_instruction_factor = {
        UPMEMPredicateLowering.TERNARY: 2,
        UPMEMPredicateLowering.BRANCHLESS_MASK: 3,
        UPMEMPredicateLowering.CONDITIONAL_ZERO: 1,
    }[decision.predicate_lowering]
    oversized = 2 * chunks * max(0, decision.dma_bytes - 128)
    branch_proxy = (
        chunks
        if decision.predicate_lowering is UPMEMPredicateLowering.CONDITIONAL_ZERO
        else 0
    )
    return UPMEMPhysicalFeatures(
        logical_work=problem.logical_work,
        dynamic_instructions=(
            problem.logical_work
            * problem.operation_instructions
            * lowering_instruction_factor
        ),
        mram_read_bytes=read_bytes,
        mram_write_bytes=write_bytes,
        mram_read_calls=chunks,
        mram_write_calls=chunks,
        barrier_count=1,
        shared_wram_accesses=0,
        replicated_mram_bytes=0,
        backpressure_proxy_units=oversized + branch_proxy,
        wram_bytes=wram,
    )


def _matrix_vector_features(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> UPMEMPhysicalFeatures:
    if decision.predicate_lowering is not UPMEMPredicateLowering.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "matrix-vector plans cannot assign predicate lowering"
        )
    if decision.rhs_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError(
            "matrix-vector plans cannot assign GEMM RHS residency"
        )
    if decision.nc is not None or decision.kc is not None:
        raise UPMEMPhysicalLegalityError("matrix-vector plans cannot assign GEMM tiles")
    if problem.reduction % decision.chunk_elements:
        raise UPMEMPhysicalLegalityError(
            "matrix-vector reduction must be divisible by chunk_elements"
        )
    if problem.rows % decision.tasklets:
        raise UPMEMPhysicalLegalityError(
            "matrix-vector rows must divide evenly across contiguous tasklets"
        )
    matrix_chunk_bytes = decision.chunk_elements * problem.element_bytes
    row_chunks = problem.reduction // decision.chunk_elements
    matrix_bytes = problem.rows * problem.reduction * problem.element_bytes
    vector_bytes_per_batch = problem.reduction * problem.element_bytes
    vector_bytes = problem.batches * vector_bytes_per_batch
    if decision.layout is UPMEMDataLayout.FUSED_REPLICATED:
        if decision.vector_residency is not UPMEMOperandResidency.FUSED_DMA_PACKET:
            raise UPMEMPhysicalLegalityError(
                "fused matrix-vector layout requires fused DMA vector residency"
            )
        expected = matrix_chunk_bytes * 2
        if decision.dma_bytes != expected:
            raise UPMEMPhysicalLegalityError(
                "DMA bytes do not match fused [matrix, vector] chunks"
            )
        read_bytes = matrix_bytes * 2
        read_calls = problem.rows * row_chunks
        replicated = matrix_bytes - vector_bytes
        shared_accesses = 0
        wram = decision.tasklets * (decision.dma_bytes + problem.accumulator_bytes)
    elif decision.layout is UPMEMDataLayout.SEPARATE:
        if decision.dma_bytes != matrix_chunk_bytes:
            raise UPMEMPhysicalLegalityError(
                "separate matrix-vector DMA must equal one matrix chunk"
            )
        if decision.vector_residency is UPMEMOperandResidency.SHARED_WRAM:
            if problem.batches != 1:
                raise UPMEMPhysicalLegalityError(
                    "shared-WRAM matrix-vector residency supports one broadcast "
                    "vector only"
                )
            read_bytes = matrix_bytes + vector_bytes
            read_calls = problem.rows * row_chunks + _ceil_div(
                vector_bytes, decision.dma_bytes
            )
            shared_accesses = problem.logical_work
            wram = vector_bytes + decision.tasklets * (
                decision.dma_bytes + problem.accumulator_bytes
            )
        elif decision.vector_residency is UPMEMOperandResidency.TASKLET_PRIVATE_WRAM:
            rows_per_batch = problem.rows // problem.batches
            if (
                problem.batches,
                rows_per_batch,
                problem.reduction,
                decision.tasklets,
                decision.dma_bytes,
                decision.chunk_elements,
            ) != (12, 16, 32, 12, 64, 16):
                raise UPMEMPhysicalLegalityError(
                    "tasklet-private matrix-vector residency is proven only for "
                    "12 batches x 16 rows x 32 columns, one tasklet per batch, "
                    "and 64-byte chunks"
                )
            output_group_bytes = rows_per_batch * problem.output_element_bytes
            read_bytes = matrix_bytes + vector_bytes
            read_calls = problem.rows * row_chunks + problem.batches * _ceil_div(
                vector_bytes_per_batch, decision.dma_bytes
            )
            shared_accesses = 0
            wram = decision.tasklets * (
                vector_bytes_per_batch + decision.dma_bytes + output_group_bytes
            )
        elif decision.vector_residency is UPMEMOperandResidency.STREAMED_MRAM:
            read_bytes = matrix_bytes * 2
            read_calls = problem.rows * row_chunks * 2
            shared_accesses = 0
            wram = decision.tasklets * (
                2 * decision.dma_bytes + problem.accumulator_bytes
            )
        else:
            raise UPMEMPhysicalLegalityError(
                "separate matrix-vector layout requires shared, tasklet-private, "
                "or streamed vector residency"
            )
        replicated = 0
    else:  # pragma: no cover - exhaustive enum guard
        raise UPMEMPhysicalLegalityError("unsupported matrix-vector layout")
    output_bytes = problem.rows * problem.output_element_bytes
    output_calls = decision.tasklets
    oversized = read_calls * max(0, decision.dma_bytes - 128)
    backpressure = oversized + shared_accesses
    return UPMEMPhysicalFeatures(
        logical_work=problem.logical_work,
        dynamic_instructions=problem.logical_work * problem.operation_instructions,
        mram_read_bytes=read_bytes,
        mram_write_bytes=output_bytes,
        mram_read_calls=read_calls,
        mram_write_calls=output_calls,
        barrier_count=1,
        shared_wram_accesses=shared_accesses,
        replicated_mram_bytes=replicated,
        backpressure_proxy_units=backpressure,
        wram_bytes=wram,
    )


def _gemm_features(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> UPMEMPhysicalFeatures:
    if decision.predicate_lowering is not UPMEMPredicateLowering.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError("GEMM plans cannot assign predicate lowering")
    if decision.vector_residency is not UPMEMOperandResidency.NOT_APPLICABLE:
        raise UPMEMPhysicalLegalityError("GEMM plans cannot assign vector residency")
    if decision.nc is None or decision.kc is None:
        raise UPMEMPhysicalLegalityError("GEMM plans require both Nc and Kc tiles")
    if decision.chunk_elements != decision.kc:
        raise UPMEMPhysicalLegalityError("GEMM chunk_elements must equal Kc")
    if problem.columns % decision.nc or problem.reduction % decision.kc:
        raise UPMEMPhysicalLegalityError(
            "GEMM dimensions must be divisible by Nc and Kc"
        )
    if problem.rows % decision.tasklets:
        raise UPMEMPhysicalLegalityError(
            "GEMM rows must divide evenly across contiguous tasklets"
        )
    element_bytes = problem.element_bytes
    output_tiles = problem.rows * (problem.columns // decision.nc)
    reduction_tiles = problem.reduction // decision.kc
    if decision.layout is UPMEMDataLayout.FUSED_REPLICATED:
        if decision.rhs_residency is not UPMEMOperandResidency.FUSED_DMA_PACKET:
            raise UPMEMPhysicalLegalityError(
                "fused GEMM layout requires fused DMA RHS residency"
            )
        packet_bytes = (decision.nc + 1) * decision.kc * element_bytes
        if decision.dma_bytes != packet_bytes:
            raise UPMEMPhysicalLegalityError(
                "DMA bytes do not match fused [A, transposed-B-tile] packet"
            )
        packets = output_tiles * reduction_tiles
        read_bytes = packets * packet_bytes
        read_calls = packets
        logical_rhs = problem.columns * problem.reduction * element_bytes
        replicated = max(0, read_bytes - logical_rhs)
        shared_accesses = 0
        wram = decision.tasklets * (
            packet_bytes + decision.nc * problem.accumulator_bytes
        )
    elif decision.layout is UPMEMDataLayout.SEPARATE:
        if decision.rhs_residency is UPMEMOperandResidency.SHARED_WRAM:
            rhs_bytes = problem.columns * problem.reduction * element_bytes
            lhs_bytes = problem.rows * problem.reduction * element_bytes
            read_bytes = lhs_bytes + rhs_bytes
            read_calls = _ceil_div(read_bytes, decision.dma_bytes)
            shared_accesses = problem.logical_work
            replicated = 0
            wram = rhs_bytes + decision.tasklets * (
                decision.kc * element_bytes + decision.nc * problem.accumulator_bytes
            )
        elif decision.rhs_residency is UPMEMOperandResidency.STREAMED_MRAM:
            lhs_bytes = (
                problem.rows
                * problem.reduction
                * element_bytes
                * (problem.columns // decision.nc)
            )
            rhs_bytes = (
                problem.rows * problem.columns * problem.reduction * element_bytes
            )
            read_bytes = lhs_bytes + rhs_bytes
            read_calls = _ceil_div(read_bytes, decision.dma_bytes)
            shared_accesses = 0
            replicated = max(
                0,
                rhs_bytes - problem.columns * problem.reduction * element_bytes,
            )
            wram = decision.tasklets * (
                2 * decision.dma_bytes + decision.nc * problem.accumulator_bytes
            )
        else:
            raise UPMEMPhysicalLegalityError(
                "separate GEMM layout requires shared or streamed RHS residency"
            )
    else:  # pragma: no cover - exhaustive enum guard
        raise UPMEMPhysicalLegalityError("unsupported GEMM layout")
    write_bytes = problem.rows * problem.columns * problem.output_element_bytes
    write_calls = output_tiles
    oversized = read_calls * max(0, decision.dma_bytes - 128)
    # Small Nc has loop/control pressure; large Nc increases live accumulator
    # pressure.  Both terms are dimensionless proxies, not cycle constants.
    underfill = max(0, 16 // decision.nc - 1)
    accumulator_pressure = max(0, decision.nc - 16) * problem.rows
    backpressure = (
        oversized
        + shared_accesses
        + output_tiles * underfill * underfill * 64
        + accumulator_pressure * problem.accumulator_bytes
    )
    return UPMEMPhysicalFeatures(
        logical_work=problem.logical_work,
        dynamic_instructions=problem.logical_work * problem.operation_instructions,
        mram_read_bytes=read_bytes,
        mram_write_bytes=write_bytes,
        mram_read_calls=read_calls,
        mram_write_calls=write_calls,
        barrier_count=1,
        shared_wram_accesses=shared_accesses,
        replicated_mram_bytes=replicated,
        backpressure_proxy_units=backpressure,
        wram_bytes=wram,
    )


def physical_features(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> UPMEMPhysicalFeatures:
    """Validate and summarize a physical decision, failing closed."""

    if not isinstance(problem, UPMEMPhysicalProblem):
        raise TypeError("problem must be a UPMEMPhysicalProblem")
    if not isinstance(decision, UPMEMPhysicalDecision):
        raise TypeError("decision must be a UPMEMPhysicalDecision")
    _require_common_legality(problem, decision)
    if problem.kind is UPMEMKernelKind.POINTWISE:
        features = _pointwise_features(problem, decision)
    elif problem.kind is UPMEMKernelKind.SELECTION:
        features = _selection_features(problem, decision)
    elif problem.kind is UPMEMKernelKind.MATRIX_VECTOR:
        features = _matrix_vector_features(problem, decision)
    elif problem.kind is UPMEMKernelKind.GEMM:
        features = _gemm_features(problem, decision)
    else:  # pragma: no cover - exhaustive enum guard
        raise UPMEMPhysicalLegalityError("unsupported UPMEM kernel kind")
    if features.wram_bytes > problem.wram_limit_bytes:
        raise UPMEMPhysicalLegalityError(
            f"physical plan requires {features.wram_bytes} WRAM bytes, exceeding "
            f"the {problem.wram_limit_bytes}-byte limit"
        )
    return features


@dataclass(frozen=True)
class UPMEMCalibrationEvidence:
    """One exact software-simulator observation used as a model anchor."""

    evidence_id: str
    candidate_set: str
    problem: UPMEMPhysicalProblem
    decision: UPMEMPhysicalDecision
    logic_cycles: int
    source: str
    evidence_role: str
    program_directory: str
    run_directory: str
    program_sha256: str
    result_sha256: str
    simulator_log_sha256: str
    oracle_validation_status: str
    raw_counters: tuple[tuple[str, int], ...]
    simulator_binary_sha256: str = UPIMULATOR_BINARY_SHA256

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.candidate_set:
            raise ValueError("calibration evidence strings must be non-empty")
        _positive_integer(self.logic_cycles, "logic_cycles")
        if len(self.simulator_binary_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.simulator_binary_sha256
        ):
            raise ValueError("simulator_binary_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "raw_counters", tuple(self.raw_counters))
        self._validate_raw_binding()
        # Evidence cannot bypass the same legality contract as searched plans.
        physical_features(self.problem, self.decision)

    def _validate_raw_binding(self) -> None:
        for name in ("source", "program_directory", "run_directory"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty descriptive string")
        if self.evidence_role not in {
            "tenon_compiler_measurement",
            "cross_compiler_conservative_reference",
        }:
            raise ValueError("calibration evidence has an unsupported evidence_role")
        for name in (
            "program_sha256",
            "result_sha256",
            "simulator_log_sha256",
        ):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if self.oracle_validation_status != "pass":
            raise ValueError("calibration evidence requires exact oracle pass")

        counters = self.raw_counters
        if len({name for name, _value in counters}) != len(counters):
            raise ValueError("raw counter names must be unique")
        for name, value in counters:
            if not name or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("raw counters require non-empty names and integers")
            if value < 0:
                raise ValueError("raw counter values must be nonnegative")
        breakdown = dict(counters)
        missing_breakdown = set(_LOGIC_BREAKDOWN_COUNTERS) - set(breakdown)
        if missing_breakdown:
            raise ValueError(
                "raw simulator binding lacks logic-cycle breakdown counters: "
                + ", ".join(sorted(missing_breakdown))
            )
        if sum(breakdown[name] for name in _LOGIC_BREAKDOWN_COUNTERS) != (
            self.logic_cycles
        ):
            raise ValueError("raw simulator breakdown must sum to logic_cycles")
        missing_mram = set(_MRAM_COUNTERS) - set(breakdown)
        if missing_mram:
            raise ValueError(
                "raw simulator binding lacks MRAM counters: "
                + ", ".join(sorted(missing_mram))
            )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "candidate_set": self.candidate_set,
            "problem": self.problem.manifest(),
            "decision": self.decision.manifest(),
            "logic_cycles": self.logic_cycles,
            "source": self.source,
            "evidence_role": self.evidence_role,
            "raw_artifact_binding": {
                "program_directory": self.program_directory,
                "run_directory": self.run_directory,
                "program_sha256": self.program_sha256,
                "result_sha256": self.result_sha256,
                "simulator_log_sha256": self.simulator_log_sha256,
                "oracle_validation_status": self.oracle_validation_status,
                "counters": dict(self.raw_counters),
            },
            "simulator": "uPIMulator Go software simulator",
            "simulator_binary_sha256": self.simulator_binary_sha256,
            "hardware_measurement": False,
        }


@dataclass(frozen=True)
class UPMEMPhysicalCostBreakdown:
    run_cycles: int
    dma_cycles: int
    etc_cycles: int
    backpressure_cycles: int

    @property
    def logic_cycles(self) -> int:
        return (
            self.run_cycles
            + self.dma_cycles
            + self.etc_cycles
            + self.backpressure_cycles
        )

    def manifest(self) -> dict[str, int]:
        return {
            "run_cycles": self.run_cycles,
            "dma_cycles": self.dma_cycles,
            "etc_cycles": self.etc_cycles,
            "backpressure_cycles": self.backpressure_cycles,
            "logic_cycles": self.logic_cycles,
        }


def _analytical_breakdown(
    decision: UPMEMPhysicalDecision, features: UPMEMPhysicalFeatures
) -> UPMEMPhysicalCostBreakdown:
    active_issue_lanes = min(decision.tasklets, UPMEM_REVOLVER_CYCLES)
    run = (
        _ceil_div(
            features.dynamic_instructions * UPMEM_REVOLVER_CYCLES,
            active_issue_lanes,
        )
        + UPMEM_PIPELINE_FILL_CYCLES
    )
    dma = _ceil_div(
        features.mram_read_bytes + features.mram_write_bytes,
        ANALYTICAL_DMA_BYTES_PER_CYCLE,
    ) + ANALYTICAL_DMA_CALL_CYCLES * (
        features.mram_read_calls + features.mram_write_calls
    )
    etc = (
        ANALYTICAL_BARRIER_CYCLES * features.barrier_count
        + ANALYTICAL_SCHEDULER_CYCLES_PER_TASKLET * decision.tasklets
    )
    backpressure = _ceil_div(
        features.backpressure_proxy_units,
        ANALYTICAL_DMA_BYTES_PER_CYCLE,
    ) + _ceil_div(
        features.shared_wram_accesses,
        ANALYTICAL_SHARED_WRAM_ACCESS_DIVISOR,
    )
    return UPMEMPhysicalCostBreakdown(run, dma, etc, backpressure)


def _calibration_key(
    problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
) -> tuple[object, ...]:
    common: tuple[object, ...] = (
        problem.kind.value,
        decision.layout.value,
        decision.tasklets,
        decision.ownership.value,
    )
    if problem.kind is UPMEMKernelKind.POINTWISE:
        return common + (decision.dma_bytes, decision.chunk_elements)
    if problem.kind is UPMEMKernelKind.SELECTION:
        return common + (
            decision.predicate_lowering.value,
            decision.dma_bytes,
            decision.chunk_elements,
        )
    if problem.kind is UPMEMKernelKind.MATRIX_VECTOR:
        batch_shape: tuple[int, ...] = (problem.batches,)
        if problem.batches > 1:
            # Batch-resident evidence is not extrapolated to a different
            # flattened-row or reduction extent.  Single-vector MTV/GEMV
            # retains the established shape-general physical stratum.
            batch_shape += (problem.rows, problem.reduction)
        return common + (
            batch_shape,
            decision.vector_residency.value,
            decision.dma_bytes,
            decision.chunk_elements,
        )
    return common + (
        decision.rhs_residency.value,
        decision.dma_bytes,
        decision.nc,
        decision.kc,
    )


def _scaled_breakdown(
    raw: UPMEMPhysicalCostBreakdown, factor: float
) -> UPMEMPhysicalCostBreakdown:
    raw_components = (
        raw.run_cycles,
        raw.dma_cycles,
        raw.etc_cycles,
        raw.backpressure_cycles,
    )
    scaled = [component * factor for component in raw_components]
    floors = [math.floor(value) for value in scaled]
    target = round(sum(scaled))
    order = sorted(
        range(len(scaled)), key=lambda index: (-(scaled[index] - floors[index]), index)
    )
    for index in order[: target - sum(floors)]:
        floors[index] += 1
    return UPMEMPhysicalCostBreakdown(*floors)


@dataclass(frozen=True)
class UPMEMPhysicalCostEstimate:
    problem: UPMEMPhysicalProblem
    decision: UPMEMPhysicalDecision
    features: UPMEMPhysicalFeatures
    raw_analytical: UPMEMPhysicalCostBreakdown
    predicted: UPMEMPhysicalCostBreakdown
    calibration_factor: float
    calibration_evidence_ids: tuple[str, ...]
    calibration_status: str
    model_fingerprint: str

    @property
    def predicted_logic_cycles(self) -> int:
        return self.predicted.logic_cycles

    @property
    def cycles(self) -> int:
        return self.predicted_logic_cycles

    def manifest(self) -> dict[str, object]:
        return {
            "schema": ESTIMATE_SCHEMA,
            "objective": {
                "metric": "uPIMulator one-DPU cumulative logic_cycle",
                "direction": "minimize",
                "scope": "device launch only",
                "excluded": ["host transfers", "host-side layout packing"],
            },
            "problem": self.problem.manifest(),
            "decision": self.decision.manifest(),
            "features": self.features.manifest(),
            "raw_analytical": self.raw_analytical.manifest(),
            "predicted": self.predicted.manifest(),
            "calibration_factor": self.calibration_factor,
            "calibration_evidence_ids": list(self.calibration_evidence_ids),
            "calibration_status": self.calibration_status,
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass(frozen=True)
class UPMEMPhysicalCostModel:
    """Physical proxy model with explicit simulator-only calibration strata."""

    evidence: tuple[UPMEMCalibrationEvidence, ...] = field(default_factory=tuple)
    name: str = "upmem-upimulator-physical-cost-v1"

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if any(not isinstance(row, UPMEMCalibrationEvidence) for row in evidence):
            raise TypeError("cost-model evidence rows must be calibration evidence")
        for row in evidence:
            # Revalidate frozen rows so a tampered or deserialized partial row
            # cannot enter a ranking model after construction.
            row._validate_raw_binding()
        identifiers = [row.evidence_id for row in evidence]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("calibration evidence IDs must be unique")
        object.__setattr__(self, "evidence", evidence)

    def _body(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "name": self.name,
            "objective": {
                "metric": "uPIMulator one-DPU cumulative logic_cycle",
                "direction": "minimize",
                "scope": "device launch only",
                "included": [
                    "DPU instructions/work",
                    "MRAM bytes and calls",
                    "barriers/scheduler proxy",
                    "DMA and shared-WRAM backpressure proxy",
                ],
                "excluded": ["host transfers", "host-side layout packing"],
            },
            "analytical_constants": {
                "revolver_cycles": UPMEM_REVOLVER_CYCLES,
                "pipeline_fill_cycles": UPMEM_PIPELINE_FILL_CYCLES,
                "dma_bytes_per_cycle_proxy": ANALYTICAL_DMA_BYTES_PER_CYCLE,
                "dma_call_cycles_proxy": ANALYTICAL_DMA_CALL_CYCLES,
                "barrier_cycles_proxy": ANALYTICAL_BARRIER_CYCLES,
                "scheduler_cycles_per_tasklet_proxy": (
                    ANALYTICAL_SCHEDULER_CYCLES_PER_TASKLET
                ),
                "shared_wram_access_divisor_proxy": (
                    ANALYTICAL_SHARED_WRAM_ACCESS_DIVISOR
                ),
                "provenance": (
                    "revolver=11 and pipeline fill=13 follow the checked-in "
                    "UPMEM analytical cost spec; remaining values are named "
                    "ranking proxies and are corrected per simulator stratum"
                ),
            },
            "calibration_method": {
                "kind": "mean observed/raw ratio within exact physical stratum",
                "stratum_fields": {
                    "pointwise": [
                        "layout",
                        "tasklets",
                        "ownership",
                        "dma_bytes",
                        "chunk_elements",
                    ],
                    "selection_flags": [
                        "layout",
                        "tasklets",
                        "ownership",
                        "predicate_lowering",
                        "dma_bytes",
                        "dma_elements",
                    ],
                    "matrix_vector": [
                        "layout",
                        "tasklets",
                        "ownership",
                        "batch_shape (exact rows/reduction when batches > 1)",
                        "vector_residency",
                        "dma_bytes",
                        "chunk_elements",
                    ],
                    "gemm": [
                        "layout",
                        "tasklets",
                        "ownership",
                        "rhs_residency",
                        "dma_bytes",
                        "Nc",
                        "Kc",
                    ],
                },
                "unrepresented_stratum": "analytical extrapolation, factor=1",
                "status_semantics": {
                    "calibrated_exact_problem_and_decision_stratum": (
                        "every contributing anchor has the identical problem and "
                        "decision fields"
                    ),
                    "calibrated_decision_stratum_shape_extrapolation": (
                        "the decision stratum is represented but at least one problem "
                        "shape or operation field differs from its anchors"
                    ),
                },
                "validation_warning": (
                    "in-sample uPIMulator ranking calibration; validate new "
                    "shapes and emitted sources with exact simulator counters"
                ),
            },
            "claim_scope": {
                "software_simulator_calibrated": bool(self.evidence),
                "hardware_calibrated": False,
                "hardware_performance_claim": False,
            },
            "evidence_roles": {
                role: sum(row.evidence_role == role for row in self.evidence)
                for role in sorted({row.evidence_role for row in self.evidence})
            },
            "evidence_binding_contract": {
                "required": [
                    "stable program/run directories",
                    "program/result/log SHA-256",
                    "exact oracle pass",
                    "four-component logic-cycle breakdown",
                    "MRAM read/write/count/activation counters",
                ],
                "partial_rows_allowed": False,
                "empty_analytical_model_allowed": True,
            },
            "evidence": [row.manifest() for row in self.evidence],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._body())

    def manifest(self) -> dict[str, object]:
        body = self._body()
        body["model_fingerprint"] = self.fingerprint
        return body

    def estimate(
        self, problem: UPMEMPhysicalProblem, decision: UPMEMPhysicalDecision
    ) -> UPMEMPhysicalCostEstimate:
        features = physical_features(problem, decision)
        raw = _analytical_breakdown(decision, features)
        key = _calibration_key(problem, decision)
        anchors = tuple(
            row
            for row in self.evidence
            if _calibration_key(row.problem, row.decision) == key
        )
        if anchors:
            ratios = []
            for anchor in anchors:
                anchor_features = physical_features(anchor.problem, anchor.decision)
                anchor_raw = _analytical_breakdown(
                    anchor.decision, anchor_features
                ).logic_cycles
                ratios.append(anchor.logic_cycles / anchor_raw)
            factor = sum(ratios) / len(ratios)
            if all(anchor.problem == problem for anchor in anchors):
                status = "calibrated_exact_problem_and_decision_stratum"
            else:
                status = "calibrated_decision_stratum_shape_extrapolation"
        else:
            factor = 1.0
            status = "uncalibrated_analytical_extrapolation"
        prediction = _scaled_breakdown(raw, factor)
        return UPMEMPhysicalCostEstimate(
            problem=problem,
            decision=decision,
            features=features,
            raw_analytical=raw,
            predicted=prediction,
            calibration_factor=factor,
            calibration_evidence_ids=tuple(row.evidence_id for row in anchors),
            calibration_status=status,
            model_fingerprint=self.fingerprint,
        )


def _pointwise_decision(
    layout: UPMEMDataLayout, dma_bytes: int, chunk_elements: int
) -> UPMEMPhysicalDecision:
    return UPMEMPhysicalDecision(layout, dma_bytes, chunk_elements)


def _selection_decision(
    dma_elements: int, predicate_lowering: UPMEMPredicateLowering
) -> UPMEMPhysicalDecision:
    return UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        dma_elements * 4,
        dma_elements,
        predicate_lowering=predicate_lowering,
    )


def _mv_decision(
    layout: UPMEMDataLayout,
    dma_bytes: int,
    residency: UPMEMOperandResidency,
) -> UPMEMPhysicalDecision:
    return UPMEMPhysicalDecision(
        layout,
        dma_bytes,
        16,
        vector_residency=residency,
    )


def _gemm_decision(
    layout: UPMEMDataLayout,
    dma_bytes: int,
    nc: int,
    kc: int,
    residency: UPMEMOperandResidency,
) -> UPMEMPhysicalDecision:
    return UPMEMPhysicalDecision(
        layout,
        dma_bytes,
        kc,
        rhs_residency=residency,
        nc=nc,
        kc=kc,
    )


_POINTWISE_REFERENCE = UPMEMPhysicalProblem.pointwise(12_288)
_SELECTION_REFERENCE = UPMEMPhysicalProblem.selection(12_288)
_MV_REFERENCE = UPMEMPhysicalProblem.matrix_vector(96, 128)
_MMTV_REFERENCE = UPMEMPhysicalProblem.matrix_vector(192, 32, batches=12)
_GEMM_REFERENCE = UPMEMPhysicalProblem.gemm(12, 128, 64)


_RAW_COUNTER_NAMES = _LOGIC_BREAKDOWN_COUNTERS + _MRAM_COUNTERS

# Frozen bindings to passing raw runs under the campaign evaluation root.  The
# hashes make this table usable after the transient run directories are moved
# into the artifact bundle; import and ranking never read these host paths.
_DEFAULT_RAW_EVIDENCE_BINDINGS = {
    "pointwise-separate-dma64": (
        ".tenon-upmem-eval-qjozJr/run-va-dma64",
        "65e0c551ee1fc943f2c9f9e946d9c41cf64e57f6583f38042d9a9b74dbe98873",
        "d80f42c628760210ae4471f2b0c63041017af5f4b50deb7580684bd120e483c7",
        "5adfa5c731a51c64ad024db6ec819684f10aafd126392c50e63f1959f08b0987",
        (59_560, 71_013, 1_303, 2_374, 12_288, 6_144, 98_304, 49_152, 2_304, 2_303),
    ),
    "pointwise-separate-dma128": (
        ".tenon-upmem-eval-qjozJr/repro-atim-va",
        "5eeac7283be523501bd1ec236c9b0e291e4459fb0d8c7d9b23aec5aed8b81fa5",
        "5f42393a8e20fef831f20da074116f20e9eb28ab05498ac9e38587e692ed7e06",
        "2b165f2d8a445b22f6a01546e28a3725146cc101721ce465a456361ce2e4f73a",
        (106_744, 4_944, 1_723, 17_755, 12_288, 6_144, 98_304, 49_152, 1_152, 1_151),
    ),
    "pointwise-separate-dma256": (
        ".tenon-upmem-eval-qjozJr/run-va-dma256",
        "590c44ad1a23bb1d63a7277317af0312516f8fd3d489cc1726cb6b18920455be",
        "d32395ede74aa2bc293a41d937b88d6459e687a66f31ccb4a2d6195e14a3db13",
        "cd596ba2d4b9473be5295371823e053ba4893834a45132a07703f60e90269679",
        (104_248, 7_741, 2_784, 19_966, 12_288, 6_144, 98_304, 49_152, 576, 575),
    ),
    "pointwise-separate-dma1024": (
        ".tenon-upmem-eval-qjozJr/run-va-dma1024",
        "b57fceff260741a743f0926aa497f52bd1b1a110685814f4b91eb91f4874e897",
        "864d443ac5a7784ac5837926780de5396562b53f4b16ed8849a42bf23009d1e3",
        "442fd406465252aebd8dc547c0d93385196517d79594decb9f540b66c301d382",
        (102_244, 28_615, 8_441, 25_321, 12_288, 6_144, 98_304, 49_152, 144, 143),
    ),
    "pointwise-fused-a32-b32-dma256": (
        ".tenon-upmem-eval-qjozJr/run-va-interleaved",
        "84dd19e4cf32ec480da4b7d3437ab8b200d15baaadd19d4c9eedb0138360fc19",
        "6e64d61f263d525d6af7c6d72e8bf02551155e4880c803e27c759129f2d1b11e",
        "01bb70f0ba9adcbe2c141264ccce529b14a7671bb4b917abd99a5dbe22da29a4",
        (91_677, 20_798, 1_701, 872, 12_288, 6_144, 98_304, 49_152, 768, 767),
    ),
    "selection-ternary-dma32-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-ternary32",
        "3523048b7cf7144b40bddbbf95f0c472abead0e242782282a58c8203671b191a",
        "8de6218bc485d17ffc3ec1d52ecb7c39d7854e256fc88be3c08951eac38602f7",
        "220b116233593917415ce8a9b86c83bdb9065620ce0faa64a0864e6821a5ca73",
        (85_453, 1_923, 1_550, 11_944, 6_144, 6_144, 49_152, 49_152, 768, 767),
    ),
    "selection-branchless-dma32-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-branchless32",
        "e98984d2f6ef6c224ba9a80ba54d5bc48c163a8e106b62e457f75fec8a838406",
        "e6a2c9d8b17bb12e97c7cf8cf1cb66cd4edb6058b095bb766788447915117a6b",
        "58d14afff044a1e7ec4195842ee6d29d2580961c9f0df738ba0b3a3d4212b31e",
        (103_965, 2_578, 1_683, 875, 6_144, 6_144, 49_152, 49_152, 768, 767),
    ),
    "selection-conditional-zero-dma32-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-if32",
        "541131b0ba311344ae9e48e8b279a24b41cf8e77f4c6d7d9ede8d19d55e9cbf2",
        "330a62f12171eec338b6f35da7405eebd23c4b708938d620d3c3fc86424d87a6",
        "d49844a9a7299db73fcfddbdfd0c02261a8a296fa1ba50456954eaf8bd63a7d4",
        (79_229, 1_857, 1_430, 11_582, 6_144, 6_144, 49_152, 49_152, 768, 767),
    ),
    "selection-ternary-dma16-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-ternary16",
        "08a80d79272695bd626e24e259ae623cf9a75368d0c3adf68b916d94d1efc2d2",
        "b43864208a8a66e1e71f5360c6e68f605821262a63f61a6c24e7efaa4d932866",
        "276638ced74a9015c79599854e3b0b564f091e25158af9262e0785de413fc45e",
        (61_833, 12_070, 1_240, 14_697, 6_144, 6_144, 49_152, 49_152, 1_536, 1_535),
    ),
    "selection-ternary-dma64-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-ternary64",
        "40094c093e87c3f5ea24ba3e27a5fd60824ba55a2d9523fe56fa1825512cbcf7",
        "e6a5ba8db6ebb127e78c57216ec59ae670c01a6cf0ffa221b80c335de18a6d93",
        "e3acca5ebb3a5199bea2dfaf2c8efe2de1358ff093b55f46338e67bd47dad383",
        (83_533, 3_818, 2_215, 11_539, 6_144, 6_144, 49_152, 49_152, 384, 383),
    ),
    "selection-ternary-dma128-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-ternary128",
        "25d85709a099589eaf44ae5e98af0119d83d4c3ca0b823a36b339b9f33a73ed8",
        "6ba7fd6b953e163623adcf5d4259364f68013f4d1f9b9c3610d11e463d8b2b2e",
        "b8a6904eca41c61dbd4105190e878081c88139557990357835ac9ee1e1348f58",
        (94_902, 5_207, 3_442, 38_528, 6_144, 6_144, 49_152, 49_152, 192, 191),
    ),
    "selection-conditional-zero-dma8-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-if8",
        "554a6f81cccd1cf204ff5d8b9c10b740e20f6a29d81ec2d2db7b3f7197a01fa9",
        "2d552712c90db21e2f2a1d9bb4d8f4c52ca07a88332bfabe2259b5e74b38b4b1",
        "2d2c5fab3a566b622ce39609291e01bb7be3c42ea4eaa75b1fc8abde22a86b0d",
        (42_487, 56_453, 1_119, 8_237, 6_144, 6_144, 49_152, 49_152, 3_072, 3_071),
    ),
    "selection-conditional-zero-dma16-elements-named": (
        ".tenon-upmem-eval-qjozJr/run-sel-if16",
        "f6dff71c5b0914bb4925e205444e1e8fffcf1a4edfcf42ed837bb138fc251def",
        "2b015535cdac9ed0fea31707e649ee34e5dbeeb7c1a7f367f5bb859cd5f85f03",
        "f0b904a1a22e0459d659ff19f22fa2c661cfede356c366f18cd7320da52d0a73",
        (37_499, 43_720, 1_150, 7_040, 6_144, 6_144, 49_152, 49_152, 1_536, 1_535),
    ),
    "selection-conditional-zero-dma16-elements-compiler-plan": (
        ".tenon-upmem-eval-qjozJr/run-sel-flags-plan-v2",
        "d30c8bf8b15942e536917d1c52a9296f204145611632dadd463116f0e7685ad7",
        "ca676c33c1c74287b26312f62ab8679038d4c8bfba9eecd81c65c92be7f311ec",
        "f6fa7e198d117e3a2365d21517b49e69e5cf741c950f48892d63080f164c4c8f",
        (37_452, 50_438, 1_176, 205, 6_144, 6_144, 49_152, 49_152, 1_536, 1_535),
    ),
    "selection-conditional-zero-dma64-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-if64",
        "52caca8497b4c2860d609dd0194d613f35b9d44e445c43d2e2be078e8869f6af",
        "5974a40e2d4d71454c62ab3155eed07643e01e20c671c90d1a80b2761d921633",
        "b5601adb8288efa1dee54f237b719c6edff55586e5ddbc47cda5828e26694dd9",
        (77_309, 3_929, 2_001, 11_303, 6_144, 6_144, 49_152, 49_152, 384, 383),
    ),
    "selection-branchless-dma16-elements": (
        ".tenon-upmem-eval-qjozJr/run-sel-branchless16",
        "9fbed9a37eed4ef9447281ad219a1800917248d915cab20feaa5c7432bebcafb",
        "3f25f05e834173c61e46ebfd306e2164437a884fbb2231e97d6af42d3b6674bb",
        "3cf7a98d144c788ad44d9041bb1be6fa1ce1e3d3450ffbfdd96cb02b2469f521",
        (68_663, 1_715, 1_260, 22_713, 6_144, 6_144, 49_152, 49_152, 1_536, 1_535),
    ),
    "mtv-shared-vector": (
        ".tenon-upmem-eval-qjozJr/run-mtv-resident",
        "87d2251b065fe26509f052f0595040c5c04f032da63ee28a25cf2258b6278cd1",
        "d6d778e3850928dff4fbacf00a056aac081f6ff9b0f7f9ec8e5ab0f04230f36d",
        "4ee36fa23d4f9a20f7afd00199b467080e09530e7f5bb241da5bd198423b775d",
        (222_163, 573, 1_715, 146_333, 6_208, 48, 49_664, 384, 770, 769),
    ),
    "gemv-shared-vector": (
        ".tenon-upmem-eval-qjozJr/run-gemv-resident",
        "e72d907c6bd60239bc02ce9ab47be8f30159f45475dc3a178b313ea8ce8335cf",
        "8cd0fb7365fd8b96c134a9103327626e57f2710771119e14eb1cebac689fe0d9",
        "aaa760b666187df1908edde191a586a63e39144132c63a51184fb72475998b4a",
        (222_259, 579, 1_723, 146_152, 6_208, 48, 49_664, 384, 770, 769),
    ),
    "mtv-fused-replicated-a16-x16": (
        ".tenon-upmem-eval-qjozJr/run-mtv-fused",
        "3437271367d02d0743dcba1f94390ec7fa10e0ea9d301391877ba453735726f5",
        "6b0aa15c0794ddea664c03666b26251a80d65402b96f7b0f6ad789ccf6d0c8ed",
        "9409698d5085ea1430c3be302df3b297f3d16e9510aebfb57af7b40470d5de93",
        (221_058, 141, 1_361, 136_850, 12_288, 48, 98_304, 384, 769, 768),
    ),
    "gemv-fused-replicated-a16-x16": (
        ".tenon-upmem-eval-qjozJr/run-gemv-fused",
        "be5cef56f4dc3197952653e77f816713187f9a27a9a8f8d4b24eb12db68576ac",
        "ec12f44edad47316909f1b6178ef723c8b5f3f33e45a8023ccd4ccc5cb87a31f",
        "efbedc585dcaffdaa0df19208fe848dc2d2a452004d9a9973eecd480529db76e",
        (221_154, 146, 1_375, 136_892, 12_288, 48, 98_304, 384, 769, 768),
    ),
    "mmtv-tasklet-private-vector-a16": (
        ".tenon-upmem-eval-qjozJr/tenon-campaign-final-runs-v4-escalated/mmtv",
        "222ab09f55249761f65f31b2d20ffbcc302a4ec8cbabb8d10dddeaa1b9ac4992",
        "6cb4e343507a49df4a886487152e6ec617def142b9a19638742564b3c76e1ab3",
        "4e2367a82d78295ab29a46f6d96e3818f1342eb8663309653be5adbc915c14e9",
        (127_139, 286, 2_381, 81_565, 3_264, 96, 26_112, 768, 405, 404),
    ),
    "mmtv-fused-replicated-identity": (
        ".tenon-upmem-eval-qjozJr/tenon-campaign-final-runs-v1/mmtv",
        "1dbc21082ea1464c33a7b2c58e020e7893f73b2f844e261569832a1a5e61310d",
        "c2bcb3dc2b52f7aaca24d1797e216e281d85b3f3bdd3365f4c1e628b4acd3bb1",
        "e6d7fd49125bb6ac8fbf03b7c9ea216367d718e3cd0215f6edd966bb8cde4b4b",
        (122_889, 520, 1_502, 91_955, 6_144, 96, 49_152, 768, 388, 387),
    ),
    "gemm-fused-nc4-kc16": (
        ".tenon-upmem-eval-qjozJr/run-gemm-fused4",
        "59823732e4d39693d3885d781b44a19e8db07dbcbe8581b0afdcdff8b89ea052",
        "179bb65a94ff5f91cdf8cb42bbe976f67452d08cececdd998e9fe06cb5c6b84c",
        "b7b14ced76976120aa0c8c6bb8afdfa59183dce8bc54a4c89964e8ed9aa63470",
        (1_944_030, 747, 4_560, 1_280_167, 61_440, 768, 491_520, 6_144, 2_301, 2_300),
    ),
    "gemm-fused-nc8-kc16": (
        ".tenon-upmem-eval-qjozJr/run-gemm-fused8",
        "dae3543a5dbeb7f693301802d952f39c52d5966c9841dd38c837057023276145",
        "b0a7f85e08d214742e51e99b9fa6ca6255fdc0abc2ce5d34160bb78afa9dd13d",
        "96b743491decc27ffdb4b556415ed9be23d077b73f759d49042b0cc5155f2037",
        (1_940_202, 1_361, 4_990, 1_192_286, 55_296, 768, 442_368, 6_144, 1_342, 1_341),
    ),
    "gemm-fused-nc16-kc16": (
        ".tenon-upmem-eval-qjozJr/run-gemm-fused16",
        "b472f6a099e5a709a8b8be404ca123b9d6f48e3c99d2b98e799dec8eb44ead42",
        "a1c9d43b549af35f9c5135707970d67e9d4d67233247ff720fbe28ddcbd27fbc",
        "5964e83a960e661d209614f9781e43d1ed1def800b237fcd8c14c986b8c98bc5",
        (1_934_634, 3_229, 6_295, 1_190_704, 52_224, 768, 417_792, 6_144, 861, 860),
    ),
    "gemm-fused-nc32-kc8": (
        ".tenon-upmem-eval-qjozJr/run-gemm-fused32x8",
        "83a8812a911f1891639774003ed8b1790d3882fbe3c19ff228d79878453df9b2",
        "4b5348ead936c0b2f444640289d5c99e10684e468772e8eaee6e1bd82c0bd91c",
        "28d8bb15169d5f26c33c2487d56c31dbfb8f8f2ea79ec9db6ec661c084b84ad5",
        (1_964_298, 3_247, 6_204, 1_212_170, 50_688, 768, 405_504, 6_144, 814, 813),
    ),
    "gemm-shared-full-rhs": (
        ".tenon-upmem-eval-qjozJr/run-gemm-shared",
        "1d054feefd71fe1dc57cfc9b8665799feef8db2bf4781d6d6632e557973a5773",
        "82fe123ac41b0dcc083f078e83ea7b92cb9dd2b57b821510ab0ba18081a0fa1b",
        "13be7a2a865ee3c1626d38f7bd47fd98ef9472016b99cf97bd2aadf4c043ea65",
        (2_491_791, 25_711, 2_664, 1_454_442, 4_480, 768, 35_840, 6_144, 47, 46),
    ),
}


def _calibration_evidence(
    evidence_id: str,
    candidate_set: str,
    problem: UPMEMPhysicalProblem,
    decision: UPMEMPhysicalDecision,
    logic_cycles: int,
) -> UPMEMCalibrationEvidence:
    capture_directory, program_sha, result_sha, log_sha, counter_values = (
        _DEFAULT_RAW_EVIDENCE_BINDINGS[evidence_id]
    )
    cross_compiler = evidence_id == "pointwise-separate-dma128"
    source = (
        "archived ATiM VA reproduction captured at " + capture_directory
        if cross_compiler
        else "passing frozen-uPIMulator capture " + capture_directory
    )
    artifact_root = f"tenon/calibration/anchors/{evidence_id}"
    return UPMEMCalibrationEvidence(
        evidence_id=evidence_id,
        candidate_set=candidate_set,
        problem=problem,
        decision=decision,
        logic_cycles=logic_cycles,
        source=source,
        evidence_role=(
            "cross_compiler_conservative_reference"
            if cross_compiler
            else "tenon_compiler_measurement"
        ),
        program_directory=artifact_root + "/program",
        run_directory=artifact_root + "/run",
        program_sha256=program_sha,
        result_sha256=result_sha,
        simulator_log_sha256=log_sha,
        oracle_validation_status="pass",
        raw_counters=tuple(zip(_RAW_COUNTER_NAMES, counter_values, strict=True)),
    )


DEFAULT_UPMEM_CALIBRATION_EVIDENCE = (
    _calibration_evidence(
        "pointwise-separate-dma64",
        "pointwise-layout-dma-sweep",
        _POINTWISE_REFERENCE,
        _pointwise_decision(UPMEMDataLayout.SEPARATE, 64, 16),
        134_250,
    ),
    _calibration_evidence(
        "pointwise-separate-dma128",
        "pointwise-layout-dma-sweep",
        _POINTWISE_REFERENCE,
        _pointwise_decision(UPMEMDataLayout.SEPARATE, 128, 32),
        131_166,
    ),
    _calibration_evidence(
        "pointwise-separate-dma256",
        "pointwise-layout-dma-sweep",
        _POINTWISE_REFERENCE,
        _pointwise_decision(UPMEMDataLayout.SEPARATE, 256, 64),
        134_739,
    ),
    _calibration_evidence(
        "pointwise-separate-dma1024",
        "pointwise-layout-dma-sweep",
        _POINTWISE_REFERENCE,
        _pointwise_decision(UPMEMDataLayout.SEPARATE, 1024, 256),
        164_621,
    ),
    _calibration_evidence(
        "pointwise-fused-a32-b32-dma256",
        "pointwise-layout-dma-sweep",
        _POINTWISE_REFERENCE,
        _pointwise_decision(UPMEMDataLayout.FUSED_REPLICATED, 256, 32),
        115_048,
    ),
    _calibration_evidence(
        "selection-ternary-dma32-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(32, UPMEMPredicateLowering.TERNARY),
        100_870,
    ),
    _calibration_evidence(
        "selection-branchless-dma32-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(32, UPMEMPredicateLowering.BRANCHLESS_MASK),
        109_101,
    ),
    _calibration_evidence(
        "selection-conditional-zero-dma32-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(32, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        94_098,
    ),
    _calibration_evidence(
        "selection-ternary-dma16-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(16, UPMEMPredicateLowering.TERNARY),
        89_840,
    ),
    _calibration_evidence(
        "selection-ternary-dma64-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(64, UPMEMPredicateLowering.TERNARY),
        101_105,
    ),
    _calibration_evidence(
        "selection-ternary-dma128-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(128, UPMEMPredicateLowering.TERNARY),
        142_079,
    ),
    _calibration_evidence(
        "selection-conditional-zero-dma8-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(8, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        108_296,
    ),
    _calibration_evidence(
        "selection-conditional-zero-dma16-elements-named",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(16, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        89_409,
    ),
    _calibration_evidence(
        "selection-conditional-zero-dma16-elements-compiler-plan",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(16, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        89_271,
    ),
    _calibration_evidence(
        "selection-conditional-zero-dma64-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(64, UPMEMPredicateLowering.CONDITIONAL_ZERO),
        94_542,
    ),
    _calibration_evidence(
        "selection-branchless-dma16-elements",
        "selection-predicate-dma-sweep",
        _SELECTION_REFERENCE,
        _selection_decision(16, UPMEMPredicateLowering.BRANCHLESS_MASK),
        94_351,
    ),
    _calibration_evidence(
        "mtv-shared-vector",
        "matrix-vector-layout-residency",
        _MV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.SEPARATE,
            64,
            UPMEMOperandResidency.SHARED_WRAM,
        ),
        370_784,
    ),
    _calibration_evidence(
        "gemv-shared-vector",
        "matrix-vector-layout-residency",
        _MV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.SEPARATE,
            64,
            UPMEMOperandResidency.SHARED_WRAM,
        ),
        370_713,
    ),
    _calibration_evidence(
        "mtv-fused-replicated-a16-x16",
        "matrix-vector-layout-residency",
        _MV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            128,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        359_410,
    ),
    _calibration_evidence(
        "gemv-fused-replicated-a16-x16",
        "matrix-vector-layout-residency",
        _MV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            128,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        359_567,
    ),
    _calibration_evidence(
        "mmtv-tasklet-private-vector-a16",
        "matrix-vector-batch-residency",
        _MMTV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.SEPARATE,
            64,
            UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
        ),
        211_371,
    ),
    _calibration_evidence(
        "mmtv-fused-replicated-identity",
        "matrix-vector-batch-residency",
        _MMTV_REFERENCE,
        _mv_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            128,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        216_866,
    ),
    _calibration_evidence(
        "gemm-fused-nc4-kc16",
        "gemm-layout-tile-residency",
        _GEMM_REFERENCE,
        _gemm_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            320,
            4,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        3_229_504,
    ),
    _calibration_evidence(
        "gemm-fused-nc8-kc16",
        "gemm-layout-tile-residency",
        _GEMM_REFERENCE,
        _gemm_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            576,
            8,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        3_138_839,
    ),
    _calibration_evidence(
        "gemm-fused-nc16-kc16",
        "gemm-layout-tile-residency",
        _GEMM_REFERENCE,
        _gemm_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1088,
            16,
            16,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        3_134_862,
    ),
    _calibration_evidence(
        "gemm-fused-nc32-kc8",
        "gemm-layout-tile-residency",
        _GEMM_REFERENCE,
        _gemm_decision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1056,
            32,
            8,
            UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        3_185_919,
    ),
    _calibration_evidence(
        "gemm-shared-full-rhs",
        "gemm-layout-tile-residency",
        _GEMM_REFERENCE,
        _gemm_decision(
            UPMEMDataLayout.SEPARATE,
            2048,
            128,
            64,
            UPMEMOperandResidency.SHARED_WRAM,
        ),
        3_974_608,
    ),
)

DEFAULT_UPMEM_PHYSICAL_COST_MODEL = UPMEMPhysicalCostModel(
    DEFAULT_UPMEM_CALIBRATION_EVIDENCE
)


@dataclass(frozen=True)
class UPMEMPhysicalCandidate:
    decision: UPMEMPhysicalDecision
    estimate: UPMEMPhysicalCostEstimate
    plan: object | None
    candidate_fingerprint: str

    @property
    def predicted_logic_cycles(self) -> int:
        return self.estimate.predicted_logic_cycles

    @property
    def fingerprint(self) -> str:
        return self.candidate_fingerprint

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "decision": self.decision.manifest(),
            "estimate": self.estimate.manifest(),
            "plan_materialized": self.plan is not None,
        }


@dataclass(frozen=True)
class UPMEMPhysicalRejection:
    decision: UPMEMPhysicalDecision
    stage: str
    reason: str

    def manifest(self) -> dict[str, object]:
        return {
            "decision_fingerprint": self.decision.fingerprint,
            "decision": self.decision.manifest(),
            "stage": self.stage,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UPMEMPhysicalSearchResult:
    problem: UPMEMPhysicalProblem
    model: UPMEMPhysicalCostModel
    ranked: tuple[UPMEMPhysicalCandidate, ...]
    rejected: tuple[UPMEMPhysicalRejection, ...]
    assignment_count: int
    search_fingerprint: str

    @property
    def best(self) -> UPMEMPhysicalCandidate:
        if not self.ranked:
            raise UPMEMNoLegalPhysicalPlan(self)
        return self.ranked[0]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": SEARCH_SCHEMA,
            "problem": self.problem.manifest(),
            "objective": {
                "metric": "uPIMulator one-DPU cumulative logic_cycle",
                "direction": "minimize",
                "scope": "device launch only",
                "excluded": ["host transfers", "host-side layout packing"],
            },
            "assignment_count": self.assignment_count,
            "legal_count": len(self.ranked),
            "rejected_count": len(self.rejected),
            "tie_break": "decision structural order, then SHA-256 fingerprint",
            "model_fingerprint": self.model.fingerprint,
            "ranked": [candidate.manifest() for candidate in self.ranked],
            "rejected": [rejection.manifest() for rejection in self.rejected],
            "search_fingerprint": self.search_fingerprint,
        }


PlanBuilder = Callable[[UPMEMPhysicalDecision], object]


def rank_upmem_physical_candidates(
    problem: UPMEMPhysicalProblem,
    decisions: Iterable[UPMEMPhysicalDecision] | None = None,
    *,
    domain: UPMEMPhysicalDecisionDomain | None = None,
    model: UPMEMPhysicalCostModel = DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
    plan_builder: PlanBuilder | None = None,
) -> UPMEMPhysicalSearchResult:
    """Exhaustively validate, materialize, estimate, and rank candidates.

    Exactly one of ``decisions`` and ``domain`` must be supplied.  Legality is
    checked before invoking ``plan_builder``.  Expected materialization
    failures are retained as rejected candidates instead of being scored.
    """

    if (decisions is None) == (domain is None):
        raise ValueError("supply exactly one of decisions or domain")
    if not isinstance(problem, UPMEMPhysicalProblem):
        raise TypeError("problem must be a UPMEMPhysicalProblem")
    if not isinstance(model, UPMEMPhysicalCostModel):
        raise TypeError("model must be a UPMEMPhysicalCostModel")
    assignments = tuple(domain.decisions() if domain is not None else decisions)
    if not assignments:
        raise ValueError("physical search requires at least one assignment")
    if any(not isinstance(item, UPMEMPhysicalDecision) for item in assignments):
        raise TypeError("all physical search assignments must be decisions")
    fingerprints = [item.fingerprint for item in assignments]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("physical search assignments must be structurally unique")

    candidates: list[UPMEMPhysicalCandidate] = []
    rejected: list[UPMEMPhysicalRejection] = []
    for decision in assignments:
        try:
            estimate = model.estimate(problem, decision)
        except UPMEMPhysicalLegalityError as error:
            rejected.append(UPMEMPhysicalRejection(decision, "legality", str(error)))
            continue
        plan: object | None = None
        if plan_builder is not None:
            try:
                plan = plan_builder(decision)
                if plan is None:
                    raise UPMEMPlanBuilderRejected("plan builder returned None")
            except (ValueError, TypeError) as error:
                rejected.append(
                    UPMEMPhysicalRejection(decision, "plan_builder", str(error))
                )
                continue
        identity = {
            "problem": problem.manifest(),
            "decision": decision.manifest(),
            "model_fingerprint": model.fingerprint,
        }
        candidates.append(
            UPMEMPhysicalCandidate(
                decision,
                estimate,
                plan,
                _fingerprint(identity),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.predicted_logic_cycles,
            candidate.decision.deterministic_order_key,
            candidate.fingerprint,
        )
    )
    rejected.sort(
        key=lambda rejection: (
            rejection.decision.deterministic_order_key,
            rejection.stage,
            rejection.reason,
        )
    )
    body = {
        "problem": problem.manifest(),
        "model_fingerprint": model.fingerprint,
        "assignment_fingerprints": fingerprints,
        "ranked_fingerprints": [candidate.fingerprint for candidate in candidates],
        "rejected": [rejection.manifest() for rejection in rejected],
    }
    return UPMEMPhysicalSearchResult(
        problem,
        model,
        tuple(candidates),
        tuple(rejected),
        len(assignments),
        _fingerprint(body),
    )


def select_upmem_physical_plan(
    problem: UPMEMPhysicalProblem,
    decisions: Iterable[UPMEMPhysicalDecision] | None = None,
    *,
    domain: UPMEMPhysicalDecisionDomain | None = None,
    model: UPMEMPhysicalCostModel = DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
    plan_builder: PlanBuilder | None = None,
) -> UPMEMPhysicalCandidate:
    """Return the deterministic minimum or fail with the complete report."""

    result = rank_upmem_physical_candidates(
        problem,
        decisions,
        domain=domain,
        model=model,
        plan_builder=plan_builder,
    )
    return result.best


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _int32_sha256(values: tuple[int, ...]) -> str:
    try:
        payload = struct.pack(f"<{len(values)}i", *values)
    except struct.error as error:
        raise ValueError("MMTV inputs must fit signed int32") from error
    return hashlib.sha256(payload).hexdigest()


def upmem_mmtv_row_costs(
    rows: int,
    columns: int,
    batches: int,
    matrix_values,
    vector_values,
) -> tuple[int, ...]:
    """Return the content-aware ``__mulsi3`` step proxy for each logical row."""

    rows = _positive_integer(rows, "rows")
    columns = _positive_integer(columns, "columns")
    batches = _positive_integer(batches, "batches")
    matrix = tuple(int(value) for value in matrix_values)
    vectors = tuple(int(value) for value in vector_values)
    total_rows = batches * rows
    if len(matrix) != total_rows * columns:
        raise ValueError("matrix_values has the wrong flattened MMTV extent")
    if len(vectors) != batches * columns:
        raise ValueError("vector_values has the wrong flattened MMTV extent")
    mask = (1 << 32) - 1
    return tuple(
        sum(
            min(
                matrix[logical_row * columns + column] & mask,
                vectors[(logical_row // rows) * columns + column] & mask,
            ).bit_length()
            for column in range(columns)
        )
        for logical_row in range(total_rows)
    )


def _balanced_mmtv_permutation(
    row_costs: tuple[int, ...],
    num_tasklets: int,
    rows_per_tasklet: int,
    phase_rotation: int,
) -> tuple[int, ...]:
    """Independent specification used to audit the physical-plan balancer."""

    buckets: list[list[int]] = [[] for _ in range(num_tasklets)]
    bucket_costs = [0] * num_tasklets
    for logical_row in sorted(
        range(len(row_costs)), key=lambda row: (-row_costs[row], row)
    ):
        eligible = [
            tasklet
            for tasklet in range(num_tasklets)
            if len(buckets[tasklet]) < rows_per_tasklet
        ]
        if not eligible:
            raise ValueError("MMTV LPT balancer exhausted all tasklet capacities")
        tasklet = min(
            eligible,
            key=lambda item: (
                bucket_costs[item],
                len(buckets[item]),
                item,
            ),
        )
        buckets[tasklet].append(logical_row)
        bucket_costs[tasklet] += row_costs[logical_row]
    physical_rows: list[int] = []
    for tasklet, bucket in enumerate(buckets):
        rotation = (phase_rotation * tasklet) % rows_per_tasklet
        physical_rows.extend(bucket[rotation:] + bucket[:rotation])
    return tuple(physical_rows)


@dataclass(frozen=True)
class UPMEMMMTVRowLayoutEvidence:
    """One exact uPIMulator observation for a canonical MMTV row layout."""

    candidate_id: str
    phase_rotation: int | None
    logic_cycles: int
    matrix_sha256: str = UPMEM_MMTV_CANONICAL_MATRIX_SHA256
    vectors_sha256: str = UPMEM_MMTV_CANONICAL_VECTORS_SHA256
    source: str = "tenon-mmtv-row-layout-sweep-2026-07-22"
    simulator_binary_sha256: str = UPIMULATOR_BINARY_SHA256

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source:
            raise ValueError("MMTV row-layout evidence strings must be non-empty")
        if self.phase_rotation is not None and (
            isinstance(self.phase_rotation, bool)
            or not isinstance(self.phase_rotation, int)
            or self.phase_rotation < 0
        ):
            raise ValueError("phase_rotation must be a nonnegative integer or None")
        _positive_integer(self.logic_cycles, "logic_cycles")
        _require_sha256(self.matrix_sha256, "matrix_sha256")
        _require_sha256(self.vectors_sha256, "vectors_sha256")
        _require_sha256(self.simulator_binary_sha256, "simulator_binary_sha256")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": MMTV_ROW_LAYOUT_EVIDENCE_SCHEMA,
            "candidate_id": self.candidate_id,
            "layout": (
                "identity" if self.phase_rotation is None else "content_aware_lpt"
            ),
            "phase_rotation": self.phase_rotation,
            "logic_cycles": self.logic_cycles,
            "shape": {"batches": 12, "rows": 16, "columns": 32},
            "canonical_inputs": {
                "matrix_sha256": self.matrix_sha256,
                "vectors_sha256": self.vectors_sha256,
            },
            "source": self.source,
            "simulator": "uPIMulator Go software simulator",
            "simulator_binary_sha256": self.simulator_binary_sha256,
            "metric": "one-DPU cumulative logic_cycle",
            "hardware_measurement": False,
        }


DEFAULT_UPMEM_MMTV_ROW_LAYOUT_EVIDENCE = (
    UPMEMMMTVRowLayoutEvidence("identity", None, 216_866),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-0", 0, 216_181),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-1", 1, 215_684),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-15", 15, 215_980),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-2", 2, 215_805),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-3", 3, 215_825),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-5", 5, 216_342),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-7", 7, 216_127),
    UPMEMMMTVRowLayoutEvidence("greedy-phase-9", 9, 216_120),
)


@dataclass(frozen=True)
class UPMEMMMTVRowLayoutCandidate:
    candidate_id: str
    phase_rotation: int | None
    physical_to_logical_rows: tuple[int, ...]
    row_costs: tuple[int, ...]
    num_tasklets: int
    rows_per_tasklet: int
    evidence: UPMEMMMTVRowLayoutEvidence

    @property
    def tasklet_cost_sums(self) -> tuple[int, ...]:
        return tuple(
            sum(
                self.row_costs[logical_row]
                for logical_row in self.physical_to_logical_rows[
                    tasklet
                    * self.rows_per_tasklet : (tasklet + 1)
                    * self.rows_per_tasklet
                ]
            )
            for tasklet in range(self.num_tasklets)
        )

    @property
    def logic_cycles(self) -> int:
        return self.evidence.logic_cycles

    @property
    def tasklet_cost_spread(self) -> int:
        return max(self.tasklet_cost_sums) - min(self.tasklet_cost_sums)

    def _body(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "layout": (
                "identity" if self.phase_rotation is None else "content_aware_lpt"
            ),
            "phase_rotation": self.phase_rotation,
            "permutation_semantics": "physical_row_to_logical_flat_row",
            "physical_to_logical_rows": list(self.physical_to_logical_rows),
            "physical_to_logical_rows_i32_sha256": _int32_sha256(
                self.physical_to_logical_rows
            ),
            "tasklet_cost_sums": list(self.tasklet_cost_sums),
            "tasklet_cost_spread": self.tasklet_cost_spread,
            "logic_cycles": self.logic_cycles,
            "evidence_id": self.evidence.candidate_id,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._body())

    def manifest(self) -> dict[str, object]:
        return {**self._body(), "candidate_fingerprint": self.fingerprint}


@dataclass(frozen=True)
class UPMEMMMTVRowLayoutSearchResult:
    batches: int
    rows: int
    columns: int
    matrix_sha256: str
    vectors_sha256: str
    row_costs: tuple[int, ...]
    ordered: tuple[UPMEMMMTVRowLayoutCandidate, ...]
    evidence: tuple[UPMEMMMTVRowLayoutEvidence, ...]

    @property
    def ranked(self) -> tuple[UPMEMMMTVRowLayoutCandidate, ...]:
        return tuple(
            sorted(
                self.ordered,
                key=lambda candidate: (
                    candidate.logic_cycles,
                    candidate.candidate_id,
                    candidate.fingerprint,
                ),
            )
        )

    @property
    def best(self) -> UPMEMMMTVRowLayoutCandidate:
        return self.ranked[0]

    def _body(self) -> dict[str, object]:
        return {
            "schema": MMTV_ROW_LAYOUT_SEARCH_SCHEMA,
            "objective": {
                "metric": "uPIMulator one-DPU cumulative logic_cycle",
                "direction": "minimize",
                "scope": "device launch only",
                "excluded": [
                    "host row packing",
                    "host output inverse permutation",
                    "host transfers",
                ],
            },
            "shape": {
                "batches": self.batches,
                "rows": self.rows,
                "columns": self.columns,
                "logical_flat_rows": self.batches * self.rows,
                "num_tasklets": self.ordered[0].num_tasklets,
                "rows_per_tasklet": self.ordered[0].rows_per_tasklet,
            },
            "canonical_inputs": {
                "matrix_sha256": self.matrix_sha256,
                "vectors_sha256": self.vectors_sha256,
            },
            "cost_formula": {
                "per_term": ("bit_length(min(lhs & 0xffffffff, vector & 0xffffffff))"),
                "per_logical_row": "sum(per_term for every reduction column)",
                "interpretation": (
                    "unsigned operand magnitude proxy for __mulsi3 early-exit steps"
                ),
                "logical_row_costs": list(self.row_costs),
            },
            "algorithm": {
                "row_order": "stable (-cost, logical_flat_row)",
                "assignment": (
                    "capacity-constrained LPT to eligible minimum "
                    "(current_cost_sum, current_len, tasklet_id)"
                ),
                "within_tasklet_order": "retain greedy assignment order",
                "phase_rotation": (
                    "rotate tasklet bucket left by "
                    "(phase * tasklet_id) % rows_per_tasklet"
                ),
                "identity_is_separate_non_greedy_candidate": True,
            },
            "ordered_phases": [None, *UPMEM_MMTV_ROW_LAYOUT_PHASES],
            "ordered_candidates": [candidate.manifest() for candidate in self.ordered],
            "ranked_candidate_ids": [
                candidate.candidate_id for candidate in self.ranked
            ],
            "selected": self.best.manifest(),
            "evidence": [row.manifest() for row in self.evidence],
            "claim_scope": {
                "software_simulator_calibrated": True,
                "hardware_calibrated": False,
                "hardware_performance_claim": False,
                "content_specific": True,
                "host_unpack_excluded": True,
            },
        }

    @property
    def search_fingerprint(self) -> str:
        return _fingerprint(self._body())

    def manifest(self) -> dict[str, object]:
        return {**self._body(), "search_fingerprint": self.search_fingerprint}


def search_upmem_mmtv_row_layout(
    matrix_values,
    vector_values,
    *,
    rows: int = 16,
    columns: int = 32,
    batches: int = 12,
    num_tasklets: int = 12,
    phases: tuple[int, ...] = UPMEM_MMTV_ROW_LAYOUT_PHASES,
    evidence: tuple[UPMEMMMTVRowLayoutEvidence, ...] = (
        DEFAULT_UPMEM_MMTV_ROW_LAYOUT_EVIDENCE
    ),
) -> UPMEMMMTVRowLayoutSearchResult:
    """Search the exact canonical MMTV layouts backed by simulator evidence."""

    rows = _positive_integer(rows, "rows")
    columns = _positive_integer(columns, "columns")
    batches = _positive_integer(batches, "batches")
    num_tasklets = _positive_integer(num_tasklets, "num_tasklets")
    matrix = tuple(int(value) for value in matrix_values)
    vectors = tuple(int(value) for value in vector_values)
    total_rows = batches * rows
    if total_rows % num_tasklets:
        raise ValueError("MMTV logical rows must divide evenly across tasklets")
    rows_per_tasklet = total_rows // num_tasklets
    phases = tuple(phases)
    if phases != UPMEM_MMTV_ROW_LAYOUT_PHASES:
        raise ValueError("MMTV evidence requires the exact ordered phase domain")
    matrix_sha256 = _int32_sha256(matrix)
    vectors_sha256 = _int32_sha256(vectors)
    if (
        matrix_sha256 != UPMEM_MMTV_CANONICAL_MATRIX_SHA256
        or vectors_sha256 != UPMEM_MMTV_CANONICAL_VECTORS_SHA256
    ):
        raise ValueError(
            "MMTV row-layout simulator evidence is bound to the canonical inputs"
        )
    if (batches, rows, columns, num_tasklets) != (12, 16, 32, 12):
        raise ValueError("MMTV row-layout evidence is bound to shape [12,16,32]")
    evidence = tuple(evidence)
    expected_ids = ("identity",) + tuple(f"greedy-phase-{phase}" for phase in phases)
    if tuple(row.candidate_id for row in evidence) != expected_ids:
        raise ValueError("MMTV row-layout evidence IDs/order do not match phase domain")
    if any(
        row.matrix_sha256 != matrix_sha256 or row.vectors_sha256 != vectors_sha256
        for row in evidence
    ):
        raise ValueError("MMTV row-layout evidence input hashes do not match inputs")

    row_costs = upmem_mmtv_row_costs(rows, columns, batches, matrix, vectors)
    candidates = [
        UPMEMMMTVRowLayoutCandidate(
            "identity",
            None,
            tuple(range(total_rows)),
            row_costs,
            num_tasklets,
            rows_per_tasklet,
            evidence[0],
        )
    ]
    from .upmem_physical import (
        balance_matrix_vector_rows,
        matrix_vector_row_mul_step_costs,
    )

    materialized_costs = matrix_vector_row_mul_step_costs(
        rows, columns, batches, matrix, vectors
    )
    if materialized_costs != row_costs:
        raise ValueError(
            "physical-plan MMTV row costs differ from search cost specification"
        )

    for index, phase in enumerate(phases, start=1):
        expected = _balanced_mmtv_permutation(
            row_costs, num_tasklets, rows_per_tasklet, phase
        )
        materialized = balance_matrix_vector_rows(
            rows,
            columns,
            batches,
            matrix,
            vectors,
            phase_rotation=phase,
        )
        if materialized != expected:
            raise ValueError(
                "physical-plan MMTV balancer differs from search cost specification"
            )
        candidates.append(
            UPMEMMMTVRowLayoutCandidate(
                f"greedy-phase-{phase}",
                phase,
                materialized,
                row_costs,
                num_tasklets,
                rows_per_tasklet,
                evidence[index],
            )
        )
    result = UPMEMMMTVRowLayoutSearchResult(
        batches,
        rows,
        columns,
        matrix_sha256,
        vectors_sha256,
        row_costs,
        tuple(candidates),
        evidence,
    )
    if result.best.candidate_id != "greedy-phase-1":
        raise ValueError("canonical MMTV evidence no longer selects greedy phase 1")
    return result


__all__ = [
    "DEFAULT_UPMEM_CALIBRATION_EVIDENCE",
    "DEFAULT_UPMEM_MMTV_ROW_LAYOUT_EVIDENCE",
    "DEFAULT_UPMEM_PHYSICAL_COST_MODEL",
    "MaskedF2TaskletLayout",
    "UPMEMCalibrationEvidence",
    "UPMEMDataLayout",
    "UPMEMKernelKind",
    "UPMEMMMTVRowLayoutCandidate",
    "UPMEMMMTVRowLayoutEvidence",
    "UPMEMMMTVRowLayoutSearchResult",
    "UPMEM_MMTV_CANONICAL_MATRIX_SHA256",
    "UPMEM_MMTV_CANONICAL_VECTORS_SHA256",
    "UPMEM_MMTV_ROW_LAYOUT_PHASES",
    "UPMEMNoLegalPhysicalPlan",
    "UPMEMOperandResidency",
    "UPMEMOwnership",
    "UPMEMPhysicalCandidate",
    "UPMEMPhysicalCostBreakdown",
    "UPMEMPhysicalCostEstimate",
    "UPMEMPhysicalCostModel",
    "UPMEMPhysicalDecision",
    "UPMEMPhysicalDecisionDomain",
    "UPMEMPhysicalFeatures",
    "UPMEMPhysicalLegalityError",
    "UPMEMPhysicalProblem",
    "UPMEMPredicateLowering",
    "UPMEMPhysicalRejection",
    "UPMEMPhysicalSearchResult",
    "UPMEMPlanBuilderRejected",
    "physical_features",
    "rank_upmem_physical_candidates",
    "search_upmem_mmtv_row_layout",
    "select_upmem_physical_plan",
    "upmem_mmtv_row_costs",
]
