# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable compute costs for Gemini-II uint16 direct-VL64 kernels.

One ``ADD_U16`` cost event represents the complete core-wide sequence: two
L1-to-MMB copies, one add, one MMB-to-L1 copy, and the mandatory SEU barrier.
The sixteen logical L1 groups are coalesced into that one event. Device ticks
remain separate from wall time; the transport-aware estimator at the end of
this module models the host upload, dispatch, and readback path explicitly.

The constants below are calibrated from real-card runs on 2026-07-08.  The
current registered task library measures 68,138 device ticks over 256 complete
vector-add pipelines, or 266.164 ticks per pipeline; the integer cost model
uses the repeated-throughput floor of 266 ticks.  Isolated phase slopes provide
the relative attribution; they are normalized so these sequential graph stages
sum to the measured overlapped pipeline throughput.  Host/PCIe time is not
mixed into this device-tick model.

The GESUMMV attribution is calibrated differently.  The production all-group
N=90 kernel measured 64,011 device ticks over eight repetitions, or 8,001.375
ticks per pipeline; its independent final pipeline took 8,243 ticks.  The
integer analytical model uses the repeated-throughput floor of 8,001 ticks.
There are not yet isolated, dependency-equivalent measurements for its
individual ``mul``, ``sum``, ``shift_left``, and ``add`` calls, so the
constants below are normalized attribution weights: their inventory-weighted
sum is exactly 8,001.  They must not be interpreted as isolated instruction
latencies.  One attributed event is one core-wide VL64 call and already
coalesces all sixteen groups (and, where applicable, both matrix streams).
"""

from __future__ import annotations

from numbers import Integral

from ...perf import BoundCostSpec, CostEvent, CostSpec, ExecutionGraph, cost, rule
from ..apu_g2_layout import APUG2ReductionPlan
from ..apu_g2_pipeline import (
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION,
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION,
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION,
    STANDALONE_4X64K_ADD_CALIBRATION,
    Opcode,
    PipelineCalibration,
    PipelineTopology,
)


def _pipeline_call_count(calibration, opcode):
    return next(
        entry.count
        for entry in calibration.signature.call_inventory
        if entry.opcode is opcode
    )


def _pipeline_attributed_cycles(calibration, opcode):
    return next(
        attribution.cycles_per_call
        for attribution in calibration.attributions
        if attribution.opcode is opcode
    )


APUG2_COPY_L1_TO_MMB_SEG0_U16_TICKS = _pipeline_attributed_cycles(
    STANDALONE_4X64K_ADD_CALIBRATION, Opcode.COPY_L1_TO_MMB_SEGMENT_0
)
APUG2_COPY_L1_TO_MMB_SEG1_U16_TICKS = _pipeline_attributed_cycles(
    STANDALONE_4X64K_ADD_CALIBRATION, Opcode.COPY_L1_TO_MMB_SEGMENT_1
)
APUG2_ADD_U16_TICKS = _pipeline_attributed_cycles(
    STANDALONE_4X64K_ADD_CALIBRATION, Opcode.ADD
)
APUG2_COPY_MMB_TO_L1_U16_TICKS = _pipeline_attributed_cycles(
    STANDALONE_4X64K_ADD_CALIBRATION, Opcode.COPY_MMB_TO_L1
)
APUG2_SEU_BARRIER_TICKS = _pipeline_attributed_cycles(
    STANDALONE_4X64K_ADD_CALIBRATION, Opcode.BARRIER
)

# Full GESUMMV real-card calibration and normalized primitive attribution.
# The relative weights are 3:8:1:2 for mul:reduce:shift:add.  The reduction
# attribution absorbs the integer-normalization remainder:
#   6*426 + 3*1152 + 4*142 + 5*284 + 1 = 8,001 ticks.
APUG2_GESUMMV_FULL_PIPELINE_TICKS = (
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION.total_cycles
)
APUG2_GESUMMV_MEASURED_TICKS_PER_PIPELINE = 8_001.375
APUG2_GESUMMV_MUL_U8_TO_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.MUL
)
APUG2_GESUMMV_GROUP_REDUCE_U16_TO_U23_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.REDUCE
)
APUG2_GESUMMV_SHIFT_LEFT_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.SHIFT
)
APUG2_GESUMMV_ADD_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.ADD
)
APUG2_GESUMMV_SEU_BARRIER_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.BARRIER
)

APUG2_GESUMMV_MUL_CALLS = _pipeline_call_count(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.MUL
)
APUG2_GESUMMV_REDUCE_CALLS = _pipeline_call_count(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.REDUCE
)
APUG2_GESUMMV_SHIFT_CALLS = _pipeline_call_count(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.SHIFT
)
APUG2_GESUMMV_ADD_CALLS = _pipeline_call_count(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.ADD
)
APUG2_GESUMMV_BARRIER_CALLS = _pipeline_call_count(
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION, Opcode.BARRIER
)

# The production accumulated-GEMV kernel measured 55,148 ticks over eight
# repetitions (6,893.5 ticks/call) on the same card/firmware.  As with the
# GESUMMV attribution above, these are normalized full-pipeline weights, not
# isolated instruction latencies.  Their inventory-weighted sum is the
# measured-throughput floor: 3*580 + 3*1474 + 2*179 + 3*124 + 1 = 6,893.
APUG2_GEMV_FULL_PIPELINE_TICKS = (
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.total_cycles
)
APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE = 6_893.5
APUG2_GEMV_MUL_U8_TO_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.MUL
)
APUG2_GEMV_GROUP_REDUCE_U16_TO_U23_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.REDUCE
)
APUG2_GEMV_SHIFT_LEFT_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.SHIFT
)
APUG2_GEMV_ADD_U16_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.ADD
)
APUG2_GEMV_SEU_BARRIER_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.BARRIER
)

APUG2_GEMV_MUL_CALLS = _pipeline_call_count(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.MUL
)
APUG2_GEMV_REDUCE_CALLS = _pipeline_call_count(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.REDUCE
)
APUG2_GEMV_SHIFT_CALLS = _pipeline_call_count(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.SHIFT
)
APUG2_GEMV_ADD_CALLS = _pipeline_call_count(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.ADD
)
APUG2_GEMV_BARRIER_CALLS = _pipeline_call_count(
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION, Opcode.BARRIER
)

# Fused PolyBench ATAX, M=116/N=124, measured 304,473 ticks over four
# repetitions (76,118.25 ticks/pipeline); the independent final pipeline was
# 73,471 ticks.  Existing GEMV primitive weights are retained, and the
# resident squeeze/spread attribution absorbs the additional GTML work.  As
# above, these are full-pipeline normalized weights, not isolated latencies.
APUG2_ATAX_FULL_PIPELINE_TICKS = (
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION.total_cycles
)
APUG2_ATAX_MEASURED_TICKS_PER_PIPELINE = 76_118.25
APUG2_ATAX_MUL_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.MUL
)
APUG2_ATAX_REDUCE_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.REDUCE
)
APUG2_ATAX_SHIFT_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.SHIFT
)
APUG2_ATAX_ADD_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.ADD
)
APUG2_ATAX_SQUEEZE_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.SQUEEZE
)
APUG2_ATAX_SPREAD_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.SPREAD
)
APUG2_ATAX_BARRIER_CALLS = _pipeline_call_count(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.BARRIER
)
APUG2_ATAX_SQUEEZE_ROWS_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.SQUEEZE
)
APUG2_ATAX_SPREAD_BLOCK_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.SPREAD
)
APUG2_ATAX_SEU_BARRIER_TICKS = _pipeline_attributed_cycles(
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION, Opcode.BARRIER
)

APUG2_U16_NUM_VECTORS = 4
APUG2_U16_LANES = 65_536
APUG2_U16_GROUPS = 16
APUG2_U16_LANES_PER_GROUP = 4_096
APUG2_U16_NUM_BITS = 16
APUG2_U16_BYTES = 2

# Gemini-II host-wall calibration for persistent column-batched GEMM. These
# coefficients fit the measured H2D, host-task, and D2H phases of the canonical
# LARGE GEMM and GEMV B=31 board runs on 2026-07-08. H2D uses one structural
# cold-start column-batch bandwidth and one steady-state column-batch bandwidth.
# The cold phase includes the runtime's timed index upload; the previously
# calibrated 50-us call overhead remains fixed. Host-task and D2H terms are the
# exact two-equation fits to the two phase measurements.
APUG2_GEMM_WALL_MODEL_REVISION = (
    "persistent_u16_gemm_wall_v4_structural_cold_steady_index_fit"
)
APUG2_GEMM_WALL_SAMPLE_REVISION = (
    "apu-00_2026-07-08_persistent_u16_gemm_b31_phase_anchors_v1"
)
APUG2_GEMM_MAX_BATCH_COLUMNS = 31
APUG2_GEMM_MAX_REDUCTION_TILE = 128
APUG2_GEMM_COLD_BATCH_H2D_BYTES_PER_US = 260.70358325664773
APUG2_GEMM_STEADY_BATCH_H2D_BYTES_PER_US = 312.18647138283796
APUG2_GEMM_H2D_CALL_US = 50.0
APUG2_GEMM_TASK_CALL_US = 716.8754153846154
APUG2_GEMM_SCALAR_MAC_US = 1.955460923076923
APUG2_GEMM_D2H_BYTES_PER_US = 368.9069149011588
APUG2_GEMM_D2H_CALL_US = 842.4398562030075

APUG2_GEMM_WALL_ANCHOR_PHASES_US = {
    "gemm_m1000_k1200_n1100_b31": {
        "h2d_us": 1_264_712.441,
        "host_task_us": 2_839_283.568,
        "d2h_us": 54_754.593,
        "wall_us": 4_158_750.602,
    },
    "gemv_m1900_k2100_n1_b31": {
        "h2d_us": 69_388.553,
        "host_task_us": 16_293.35,
        "d2h_us": 864.646,
        "wall_us": 86_546.549,
    },
}

APUG2_GEMM_WALL_FINGERPRINT_DATA = {
    "model_revision": APUG2_GEMM_WALL_MODEL_REVISION,
    "sample_revision": APUG2_GEMM_WALL_SAMPLE_REVISION,
    "cold_batch_h2d_bytes_per_us": APUG2_GEMM_COLD_BATCH_H2D_BYTES_PER_US,
    "steady_batch_h2d_bytes_per_us": APUG2_GEMM_STEADY_BATCH_H2D_BYTES_PER_US,
    "h2d_call_us": APUG2_GEMM_H2D_CALL_US,
    "task_call_us": APUG2_GEMM_TASK_CALL_US,
    "scalar_mac_us": APUG2_GEMM_SCALAR_MAC_US,
    "d2h_bytes_per_us": APUG2_GEMM_D2H_BYTES_PER_US,
    "d2h_call_us": APUG2_GEMM_D2H_CALL_US,
    "fit_policy": (
        "two_anchor_structural_index_cold_steady_phase_fit_with_fixed_h2d_call_overhead"
    ),
    "anchor_phases_us": APUG2_GEMM_WALL_ANCHOR_PHASES_US,
    "physical_rows": APUG2_U16_GROUPS * APUG2_U16_LANES_PER_GROUP,
    "element_bytes": APUG2_U16_BYTES,
    "index_uploads": 1,
    "index_vectors_per_group": 1,
    "max_batch_columns": APUG2_GEMM_MAX_BATCH_COLUMNS,
    "max_reduction_tile": APUG2_GEMM_MAX_REDUCTION_TILE,
    "direct_vl64_calibrations": [
        STANDALONE_4X64K_ADD_CALIBRATION.canonical_manifest,
        NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.canonical_manifest,
        NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION.canonical_manifest,
        NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION.canonical_manifest,
    ],
}


def estimate_apu_g2_u16_gemm_wall_us(
    rows: int,
    reduction: int,
    columns: int,
    *,
    batch_columns: int = 31,
    reduction_tile: int = 128,
) -> dict[str, float | int]:
    """Estimate measured host wall for the persistent APUg2 GEMM schedule.

    This model deliberately counts padded L1 transfer bytes and host calls;
    it is not a conversion from device ticks. The returned phases use the
    same ``h2d + host_task + d2h`` metric as the board harness.
    """

    for name, value in (
        ("rows", rows),
        ("reduction", reduction),
        ("columns", columns),
        ("batch_columns", batch_columns),
        ("reduction_tile", reduction_tile),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    rows = int(rows)
    reduction = int(reduction)
    columns = int(columns)
    batch_columns = int(batch_columns)
    reduction_tile = int(reduction_tile)
    if rows > APUG2_U16_GROUPS * APUG2_U16_LANES_PER_GROUP:
        raise ValueError("rows exceed one APUg2 VL64 vector")
    if batch_columns > APUG2_GEMM_MAX_BATCH_COLUMNS:
        raise ValueError("batch_columns exceed the resident accumulator limit")
    if reduction_tile != APUG2_GEMM_MAX_REDUCTION_TILE:
        raise ValueError("reduction_tile must equal the calibrated runtime tile")

    groups = (rows + APUG2_U16_LANES_PER_GROUP - 1) // APUG2_U16_LANES_PER_GROUP
    column_batches = (columns + batch_columns - 1) // batch_columns
    reduction_tiles = (reduction + reduction_tile - 1) // reduction_tile
    hardware_tasks = column_batches * reduction_tiles
    weight_bytes = (
        hardware_tasks
        * groups
        * reduction_tile
        * APUG2_U16_LANES_PER_GROUP
        * APUG2_U16_BYTES
    )
    accumulator_bytes = groups * columns * APUG2_U16_LANES_PER_GROUP * APUG2_U16_BYTES
    index_bytes = groups * APUG2_U16_LANES_PER_GROUP * APUG2_U16_BYTES
    h2d_bytes = index_bytes + weight_bytes + accumulator_bytes
    h2d_calls = 1 + hardware_tasks + column_batches
    weight_batch_bytes = (
        reduction_tiles
        * groups
        * reduction_tile
        * APUG2_U16_LANES_PER_GROUP
        * APUG2_U16_BYTES
    )
    first_batch_columns = min(columns, batch_columns)
    first_batch_accumulator_bytes = (
        groups * first_batch_columns * APUG2_U16_LANES_PER_GROUP * APUG2_U16_BYTES
    )
    cold_batch_bytes = index_bytes + weight_batch_bytes + first_batch_accumulator_bytes
    steady_batch_bytes = h2d_bytes - cold_batch_bytes
    h2d_us = (
        cold_batch_bytes / APUG2_GEMM_COLD_BATCH_H2D_BYTES_PER_US
        + steady_batch_bytes / APUG2_GEMM_STEADY_BATCH_H2D_BYTES_PER_US
        + h2d_calls * APUG2_GEMM_H2D_CALL_US
    )
    host_task_us = (
        hardware_tasks * APUG2_GEMM_TASK_CALL_US
        + columns * reduction * APUG2_GEMM_SCALAR_MAC_US
    )
    d2h_us = (
        accumulator_bytes / APUG2_GEMM_D2H_BYTES_PER_US
        + column_batches * APUG2_GEMM_D2H_CALL_US
    )
    return {
        "h2d_us": h2d_us,
        "host_task_us": host_task_us,
        "d2h_us": d2h_us,
        "wall_us": h2d_us + host_task_us + d2h_us,
        "hardware_tasks": hardware_tasks,
        "weight_uploads": hardware_tasks,
        "output_readbacks": column_batches,
        "weight_bytes": weight_bytes,
        "accumulator_bytes": accumulator_bytes,
        "index_bytes": index_bytes,
        "h2d_bytes": h2d_bytes,
        "cold_batch_bytes": cold_batch_bytes,
        "steady_batch_bytes": steady_batch_bytes,
        "h2d_calls": h2d_calls,
        "calibration": APUG2_GEMM_WALL_SAMPLE_REVISION,
    }


def _score_apu_g2_persistent_materialization(bound_cost, materialized):
    """Score an exact persistent transport materialization for this cost."""

    if getattr(bound_cost.target, "name", None) != "apu_v2":
        raise TypeError("persistent APUg2 materialization requires an apu_v2 cost")
    try:
        schedule = materialized.schedule
        rows = materialized.rows
        reduction = materialized.reduction
        columns = materialized.columns
    except AttributeError as error:
        raise TypeError(
            "APUg2 transport cost requires a persistent GEMM materialization"
        ) from error
    return estimate_apu_g2_u16_gemm_wall_us(
        rows,
        reduction,
        columns,
        batch_columns=schedule.batch_columns,
        reduction_tile=schedule.reduction_tile,
    )


def _require_one_vl64_call(event: CostEvent) -> None:
    """Reject scalar/group counts that would multiply one hardware call."""

    if "vl64_calls" not in event.metrics:
        raise ValueError("APUg2 cost event requires vl64_calls=1")
    value = event.metrics["vl64_calls"]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != 1:
        raise ValueError(
            "one APUg2 uint16 vector-pack add must issue exactly one VL64 "
            f"call, got vl64_calls={value!r}"
        )


_SUPPORTED_PIPELINE_CALIBRATIONS = (
    STANDALONE_4X64K_ADD_CALIBRATION,
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION,
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION,
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION,
)


def _event_pipeline_calibration(event: CostEvent) -> PipelineCalibration:
    calibration = event.metrics.get("pipeline_calibration")
    if not isinstance(calibration, PipelineCalibration):
        raise TypeError("APUg2 cost event requires a typed pipeline_calibration")
    if calibration not in _SUPPORTED_PIPELINE_CALIBRATIONS:
        raise ValueError("APUg2 cost event uses an unregistered structural calibration")
    streams = event.metrics.get("coalesced_matrix_streams")
    if streams != calibration.signature.coalesced_matrix_streams:
        raise ValueError("event stream count disagrees with its pipeline signature")
    return calibration


def _event_attributed_cycles(event: CostEvent, opcode: Opcode) -> int:
    calibration = _event_pipeline_calibration(event)
    return _pipeline_attributed_cycles(calibration, opcode)


_PIPELINE_TARGET_OPS = {
    Opcode.MUL: "MUL_U8_TO_U16",
    Opcode.REDUCE: "GROUP_REDUCE_ADD_U16_TO_U23",
    Opcode.SHIFT: "SHIFT_LEFT_U16",
    Opcode.ADD: "ADD_U16",
    Opcode.SQUEEZE: "SQUEEZE_ROWS_INPLACE",
    Opcode.SPREAD: "SPREAD_BLOCK",
    Opcode.BARRIER: "SEU_BARRIER",
}


def _pipeline_inventory(target, calibration):
    return tuple(
        (
            entry.opcode.value,
            target.op(_PIPELINE_TARGET_OPS[entry.opcode]),
            entry.count,
        )
        for entry in calibration.signature.call_inventory
    )


def _pipeline_metadata(calibration):
    return {
        "pipeline_signature": calibration.signature.canonical_manifest,
        "pipeline_calibration_fingerprint": calibration.fingerprint,
    }


def _pipeline_event_metrics(calibration, **metrics):
    return {
        **metrics,
        "pipeline_calibration": calibration,
        "coalesced_matrix_streams": (calibration.signature.coalesced_matrix_streams),
    }


def _stream_label(event: CostEvent, operation: str) -> str:
    signature = _event_pipeline_calibration(event).signature
    width = "single" if signature.coalesced_matrix_streams == 1 else "dual"
    resident = "resident_" if signature.resident_intermediate else ""
    return f"{resident}{width}_stream_{operation}"


def _emit_normalized_vl64_primitive(
    ctx, target, primitive, latency, name, *, uses_l1=False
):
    """Emit one calibrated, core-wide VL64 primitive attribution."""

    _require_one_vl64_call(ctx.event)
    occupy = [
        ctx.use(primitive),
        ctx.use(target.unit("core")),
        ctx.use(target.unit("arc")),
        ctx.use(target.unit("vector_engine")),
        ctx.use(target.mmb),
    ]
    if uses_l1:
        # The normalized attribution includes the copies around this primitive.
        occupy.append(ctx.use(target.l1))
    ctx.step(latency=latency, occupy=occupy, name=name)


@cost(
    target="apu_v2",
    fingerprint_data=APUG2_GEMM_WALL_FINGERPRINT_DATA,
    materialization_scorer=_score_apu_g2_persistent_materialization,
)
def apu_g2_cost(target):
    """Expand coalesced uint16 kernels onto core-wide VL64 resources."""

    core = target.unit("core")
    arc = target.unit("arc")
    vector_engine = target.unit("vector_engine")
    copy_segment0 = target.move("L1_TO_MMB_SEG0")
    copy_segment1 = target.move("L1_TO_MMB_SEG1")
    copy_to_l1 = target.move("MMB_TO_L1")
    add_u16 = target.op("ADD_U16")
    mul_u8_to_u16 = target.op("MUL_U8_TO_U16")
    reduce_u16_to_u23 = target.op("GROUP_REDUCE_ADD_U16_TO_U23")
    shift_left_u16 = target.op("SHIFT_LEFT_U16")
    seu_barrier = target.op("SEU_BARRIER")
    squeeze_rows = target.op("SQUEEZE_ROWS_INPLACE")
    spread_block = target.op("SPREAD_BLOCK")

    @rule(add_u16)
    def vector_add_u16(event, ctx):
        calibration = _event_pipeline_calibration(event)
        if calibration.signature.topology is PipelineTopology.NORMALIZED:
            _emit_normalized_vl64_primitive(
                ctx,
                target,
                add_u16,
                _event_attributed_cycles(event, Opcode.ADD),
                _stream_label(event, "add_u16"),
                uses_l1=True,
            )
            return
        if calibration.signature != STANDALONE_4X64K_ADD_CALIBRATION.signature:
            raise ValueError("unsupported standalone APUg2 pipeline signature")
        _require_one_vl64_call(event)
        shared_issue = (core, arc, vector_engine)

        ctx.step(
            latency=_event_attributed_cycles(event, Opcode.COPY_L1_TO_MMB_SEGMENT_0),
            occupy=[
                ctx.use(copy_segment0),
                *(ctx.use(handle) for handle in shared_issue),
                ctx.use(target.l1),
                ctx.use(target.mmb),
            ],
            name="copy_l1_to_mmb_seg0_u16",
        )
        ctx.step(
            latency=_event_attributed_cycles(event, Opcode.COPY_L1_TO_MMB_SEGMENT_1),
            occupy=[
                ctx.use(copy_segment1),
                *(ctx.use(handle) for handle in shared_issue),
                ctx.use(target.l1),
                ctx.use(target.mmb),
            ],
            name="copy_l1_to_mmb_seg1_u16",
        )
        ctx.step(
            latency=_event_attributed_cycles(event, Opcode.ADD),
            occupy=[
                ctx.use(add_u16),
                *(ctx.use(handle) for handle in shared_issue),
                ctx.use(target.mmb),
            ],
            name="add_u16",
        )
        ctx.step(
            latency=_event_attributed_cycles(event, Opcode.COPY_MMB_TO_L1),
            occupy=[
                ctx.use(copy_to_l1),
                *(ctx.use(handle) for handle in shared_issue),
                ctx.use(target.mmb),
                ctx.use(target.l1),
            ],
            name="copy_mmb_to_l1_u16",
        )
        ctx.step(
            latency=_event_attributed_cycles(event, Opcode.BARRIER),
            occupy=[ctx.use(handle) for handle in shared_issue],
            name="seu_barrier",
        )

    @rule(mul_u8_to_u16)
    def normalized_mul_u8_to_u16(event, ctx):
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            mul_u8_to_u16,
            _event_attributed_cycles(event, Opcode.MUL),
            _stream_label(event, "mul_u8_to_u16"),
            uses_l1=True,
        )

    @rule(reduce_u16_to_u23)
    def normalized_reduce_u16_to_u23(event, ctx):
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            reduce_u16_to_u23,
            _event_attributed_cycles(event, Opcode.REDUCE),
            _stream_label(event, "group_reduce_add_u16_to_u23"),
            uses_l1=True,
        )

    @rule(shift_left_u16)
    def normalized_shift_left_u16(event, ctx):
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            shift_left_u16,
            _event_attributed_cycles(event, Opcode.SHIFT),
            _stream_label(event, "shift_left_u16"),
        )

    @rule(seu_barrier)
    def normalized_seu_barrier(event, ctx):
        calibration = _event_pipeline_calibration(event)
        if calibration.signature.topology is not PipelineTopology.NORMALIZED:
            raise ValueError("standalone barriers are expanded by the ADD rule")
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            seu_barrier,
            _event_attributed_cycles(event, Opcode.BARRIER),
            "seu_barrier",
        )

    @rule(squeeze_rows)
    def atax_squeeze_rows(event, ctx):
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            squeeze_rows,
            _event_attributed_cycles(event, Opcode.SQUEEZE),
            "squeeze_rows_inplace",
            uses_l1=True,
        )

    @rule(spread_block)
    def atax_spread_block(event, ctx):
        _emit_normalized_vl64_primitive(
            ctx,
            target,
            spread_block,
            _event_attributed_cycles(event, Opcode.SPREAD),
            "spread_block",
            uses_l1=True,
        )


def _bind_cost(target, cost_spec) -> BoundCostSpec:
    if isinstance(cost_spec, BoundCostSpec):
        if cost_spec.target is not target:
            raise ValueError("bound APUg2 cost spec belongs to another target")
        return cost_spec
    if isinstance(cost_spec, CostSpec):
        return cost_spec.bind(target)
    raise TypeError("cost must be a CostSpec or BoundCostSpec")


def build_apu_g2_add_graph(
    target, cost_spec=apu_g2_cost, *, vl64_calls: int = 1
) -> ExecutionGraph:
    """Build the compute-only graph for one 4x64K uint16 vector-pack add."""

    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 vector-add graphs require the apu_v2 target")
    calibration = STANDALONE_4X64K_ADD_CALIBRATION
    calibration.signature.validate_shape(APUG2_U16_NUM_VECTORS, APUG2_U16_LANES)
    bound = _bind_cost(target, cost_spec)
    graph = ExecutionGraph(
        name="apu_g2_u16_vector_add",
        metadata={
            "target": "apu_v2",
            "cost": bound.spec.name,
            "cost_fingerprint": bound.fingerprint,
            "dtype": "uint16",
            "shape": (APUG2_U16_NUM_VECTORS, APUG2_U16_LANES),
            "coalesced_groups": APUG2_U16_GROUPS,
            **_pipeline_metadata(calibration),
        },
    )
    event = CostEvent.create(
        "apu_g2:add_u16",
        target.op("ADD_U16"),
        work_id=(),
        metrics=_pipeline_event_metrics(
            calibration,
            vl64_calls=vl64_calls,
            num_vectors=APUG2_U16_NUM_VECTORS,
            lanes=APUG2_U16_LANES,
            groups=APUG2_U16_GROUPS,
            lanes_per_group=APUG2_U16_LANES_PER_GROUP,
            num_bits=APUG2_U16_NUM_BITS,
        ),
        attributes={"phase": "compute", "coalesced_spmw_axis": "group"},
    )
    bound.emit(graph, event)
    return graph


def estimate_apu_g2_add(target, cost_spec=apu_g2_cost, *, vl64_calls: int = 1):
    """Evaluate :func:`build_apu_g2_add_graph` with its bound cost program."""

    bound = _bind_cost(target, cost_spec)
    graph = build_apu_g2_add_graph(target, bound, vl64_calls=vl64_calls)
    return bound.evaluate(graph)


def build_apu_g2_gemv_graph(
    target,
    cost_spec=apu_g2_cost,
    *,
    output_extent: int,
    reduction_extent: int,
) -> ExecutionGraph:
    """Build one accumulated modular-uint16 GEMV direct-VL64 graph."""

    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 GEMV graphs require the apu_v2 target")
    calibration = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
    calibration.signature.validate_shape(output_extent, reduction_extent)
    plan = APUG2ReductionPlan(output_extent, reduction_extent, stream_extent=1)
    if plan.log_block_size > 8:
        raise ValueError("APUg2 GEMV padded reduction extent must not exceed 256")
    bound = _bind_cost(target, cost_spec)
    inventory = _pipeline_inventory(target, calibration)
    graph = ExecutionGraph(
        name="apu_g2_u16_single_stream_reduction",
        metadata={
            "target": "apu_v2",
            "cost": bound.spec.name,
            "cost_fingerprint": bound.fingerprint,
            "dtype": "uint16",
            "shape": (plan.output_extent, plan.reduction_extent),
            "output_extent": plan.output_extent,
            "reduction_extent": plan.reduction_extent,
            "padded_output_extent": plan.padded_output_extent,
            "padded_reduction_extent": plan.padded_reduction_extent,
            "log_block_size": plan.log_block_size,
            "matrix_streams": 1,
            "coalesced_groups": APUG2_U16_GROUPS,
            "vl64_compute_calls": sum(count for _, _, count in inventory[:-1]),
            "vl64_barrier_calls": APUG2_GEMV_BARRIER_CALLS,
            "calibration_ticks": APUG2_GEMV_FULL_PIPELINE_TICKS,
            "measured_ticks_per_pipeline": APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE,
            "calibration_repetitions": 8,
            "calibration_basis": "real_card_full_pipeline_normalized_attribution",
            **_pipeline_metadata(calibration),
        },
    )
    dependencies = ()
    ordinal = 0
    for phase, primitive, count in inventory:
        for phase_index in range(count):
            event = CostEvent.create(
                f"apu_g2:single_stream_reduction:{ordinal}:{phase}",
                primitive,
                work_id=(),
                metrics=_pipeline_event_metrics(
                    calibration,
                    vl64_calls=1,
                    output_extent=plan.output_extent,
                    reduction_extent=plan.reduction_extent,
                    padded_output_extent=plan.padded_output_extent,
                    padded_reduction_extent=plan.padded_reduction_extent,
                    log_block_size=plan.log_block_size,
                    groups=APUG2_U16_GROUPS,
                    matrix_streams=1,
                ),
                attributes={
                    "phase": phase,
                    "phase_index": phase_index,
                    "coalesced_spmw_axis": "group",
                    "attribution": "normalized_full_pipeline_not_isolated",
                },
            )
            dependencies = bound.emit(graph, event, dependencies=dependencies)
            ordinal += 1
    return graph


def estimate_apu_g2_gemv(
    target,
    cost_spec=apu_g2_cost,
    *,
    output_extent: int,
    reduction_extent: int,
):
    """Evaluate a complete accumulated modular-uint16 GEMV graph."""

    bound = _bind_cost(target, cost_spec)
    graph = build_apu_g2_gemv_graph(
        target,
        bound,
        output_extent=output_extent,
        reduction_extent=reduction_extent,
    )
    return bound.evaluate(graph)


def build_apu_g2_atax_graph(
    target,
    cost_spec=apu_g2_cost,
    *,
    row_extent: int,
    column_extent: int,
) -> ExecutionGraph:
    """Build the one-task resident-intermediate uint16 ATAX graph."""

    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 ATAX graphs require the apu_v2 target")
    calibration = NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION
    calibration.signature.validate_shape(row_extent, column_extent)
    bound = _bind_cost(target, cost_spec)
    inventory = _pipeline_inventory(target, calibration)
    graph = ExecutionGraph(
        name="apu_g2_u16_resident_two_stage_reduction",
        metadata={
            "target": "apu_v2",
            "cost": bound.spec.name,
            "cost_fingerprint": bound.fingerprint,
            "dtype": "uint16",
            "shape": (row_extent, column_extent),
            "row_extent": row_extent,
            "column_extent": column_extent,
            "chunk_size": 32,
            "stage1_streams": 4,
            "coalesced_groups": APUG2_U16_GROUPS,
            "vl64_compute_calls": sum(count for _, _, count in inventory[:-3]),
            "resident_transform_calls": (
                APUG2_ATAX_SQUEEZE_CALLS + APUG2_ATAX_SPREAD_CALLS
            ),
            "vl64_barrier_calls": APUG2_ATAX_BARRIER_CALLS,
            "hardware_tasks": 1,
            "resident_intermediate": True,
            "calibration_ticks": APUG2_ATAX_FULL_PIPELINE_TICKS,
            "measured_ticks_per_pipeline": APUG2_ATAX_MEASURED_TICKS_PER_PIPELINE,
            "calibration_repetitions": 4,
            "calibration_basis": "real_card_full_pipeline_normalized_attribution",
            **_pipeline_metadata(calibration),
        },
    )
    dependencies = ()
    ordinal = 0
    for phase, primitive, count in inventory:
        for phase_index in range(count):
            event = CostEvent.create(
                f"apu_g2:resident_two_stage_reduction:{ordinal}:{phase}",
                primitive,
                work_id=(),
                metrics=_pipeline_event_metrics(
                    calibration,
                    vl64_calls=1,
                    row_extent=row_extent,
                    column_extent=column_extent,
                    groups=APUG2_U16_GROUPS,
                    chunk_size=32,
                ),
                attributes={
                    "phase": phase,
                    "phase_index": phase_index,
                    "coalesced_spmw_axis": "group",
                    "attribution": "normalized_full_pipeline_not_isolated",
                },
            )
            dependencies = bound.emit(graph, event, dependencies=dependencies)
            ordinal += 1
    return graph


def estimate_apu_g2_atax(
    target,
    cost_spec=apu_g2_cost,
    *,
    row_extent: int,
    column_extent: int,
):
    bound = _bind_cost(target, cost_spec)
    graph = build_apu_g2_atax_graph(
        target,
        bound,
        row_extent=row_extent,
        column_extent=column_extent,
    )
    return bound.evaluate(graph)


def build_apu_g2_gesummv_graph(
    target,
    cost_spec=apu_g2_cost,
    *,
    output_extent: int,
    reduction_extent: int,
) -> ExecutionGraph:
    """Build one complete direct-VL64 modular-uint16 GESUMMV graph.

    The graph records the exact call inventory of the hardware implementation.
    Calls are core-wide: neither logical rows nor the sixteen layout groups
    multiply an event.  Matrix streams A and B occupy separate MMB sets and
    are coalesced by each applicable four-set VL64 descriptor.
    """

    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 GESUMMV graphs require the apu_v2 target")
    calibration = NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION
    calibration.signature.validate_shape(output_extent, reduction_extent)
    plan = APUG2ReductionPlan(output_extent, reduction_extent, stream_extent=2)
    if plan.log_block_size > 8:
        raise ValueError("APUg2 GESUMMV padded reduction extent must not exceed 256")
    bound = _bind_cost(target, cost_spec)
    inventory = _pipeline_inventory(target, calibration)
    graph = ExecutionGraph(
        name="apu_g2_u16_dual_stream_reduction",
        metadata={
            "target": "apu_v2",
            "cost": bound.spec.name,
            "cost_fingerprint": bound.fingerprint,
            "dtype": "uint16",
            "shape": (plan.output_extent, plan.reduction_extent),
            "output_extent": plan.output_extent,
            "reduction_extent": plan.reduction_extent,
            "padded_output_extent": plan.padded_output_extent,
            "padded_reduction_extent": plan.padded_reduction_extent,
            "log_block_size": plan.log_block_size,
            "matrix_streams": 2,
            "coalesced_groups": APUG2_U16_GROUPS,
            "vl64_compute_calls": sum(count for _, _, count in inventory[:-1]),
            "vl64_barrier_calls": APUG2_GESUMMV_BARRIER_CALLS,
            "calibration_ticks": APUG2_GESUMMV_FULL_PIPELINE_TICKS,
            "measured_ticks_per_pipeline": (APUG2_GESUMMV_MEASURED_TICKS_PER_PIPELINE),
            "calibration_repetitions": 8,
            "calibration_basis": "real_card_full_pipeline_normalized_attribution",
            **_pipeline_metadata(calibration),
        },
    )
    dependencies = ()
    ordinal = 0
    for phase, primitive, count in inventory:
        for phase_index in range(count):
            event = CostEvent.create(
                f"apu_g2:dual_stream_reduction:{ordinal}:{phase}",
                primitive,
                work_id=(),
                metrics=_pipeline_event_metrics(
                    calibration,
                    vl64_calls=1,
                    output_extent=plan.output_extent,
                    reduction_extent=plan.reduction_extent,
                    padded_output_extent=plan.padded_output_extent,
                    padded_reduction_extent=plan.padded_reduction_extent,
                    log_block_size=plan.log_block_size,
                    groups=APUG2_U16_GROUPS,
                    matrix_streams=2,
                ),
                attributes={
                    "phase": phase,
                    "phase_index": phase_index,
                    "coalesced_spmw_axis": "group",
                    "attribution": "normalized_full_pipeline_not_isolated",
                },
            )
            dependencies = bound.emit(graph, event, dependencies=dependencies)
            ordinal += 1
    return graph


def estimate_apu_g2_gesummv(
    target,
    cost_spec=apu_g2_cost,
    *,
    output_extent: int,
    reduction_extent: int,
):
    """Evaluate a complete direct-VL64 modular-uint16 GESUMMV graph."""

    bound = _bind_cost(target, cost_spec)
    graph = build_apu_g2_gesummv_graph(
        target,
        bound,
        output_extent=output_extent,
        reduction_extent=reduction_extent,
    )
    return bound.evaluate(graph)


__all__ = [
    "APUG2_ATAX_FULL_PIPELINE_TICKS",
    "APUG2_ATAX_MEASURED_TICKS_PER_PIPELINE",
    "APUG2_ADD_U16_TICKS",
    "APUG2_COPY_L1_TO_MMB_SEG0_U16_TICKS",
    "APUG2_COPY_L1_TO_MMB_SEG1_U16_TICKS",
    "APUG2_COPY_MMB_TO_L1_U16_TICKS",
    "APUG2_GEMV_ADD_CALLS",
    "APUG2_GEMV_FULL_PIPELINE_TICKS",
    "APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE",
    "APUG2_GEMV_MUL_CALLS",
    "APUG2_GEMV_REDUCE_CALLS",
    "APUG2_GEMV_SHIFT_CALLS",
    "APUG2_GESUMMV_ADD_CALLS",
    "APUG2_GESUMMV_ADD_U16_TICKS",
    "APUG2_GESUMMV_BARRIER_CALLS",
    "APUG2_GESUMMV_FULL_PIPELINE_TICKS",
    "APUG2_GESUMMV_GROUP_REDUCE_U16_TO_U23_TICKS",
    "APUG2_GESUMMV_MEASURED_TICKS_PER_PIPELINE",
    "APUG2_GESUMMV_MUL_CALLS",
    "APUG2_GESUMMV_MUL_U8_TO_U16_TICKS",
    "APUG2_GESUMMV_REDUCE_CALLS",
    "APUG2_GESUMMV_SEU_BARRIER_TICKS",
    "APUG2_GESUMMV_SHIFT_CALLS",
    "APUG2_GESUMMV_SHIFT_LEFT_U16_TICKS",
    "APUG2_SEU_BARRIER_TICKS",
    "apu_g2_cost",
    "build_apu_g2_add_graph",
    "build_apu_g2_atax_graph",
    "build_apu_g2_gemv_graph",
    "build_apu_g2_gesummv_graph",
    "estimate_apu_g2_add",
    "estimate_apu_g2_atax",
    "estimate_apu_g2_gemv",
    "estimate_apu_g2_gesummv",
]
