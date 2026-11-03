# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable, real-device-calibrated cycle program for GSI APU v1.

The structural target contains no timing.  This ordinary Python module is the
calibration surface: an agent can rerun GVML microprofiles on the installed
Gemini board and edit the measured constants or group-size formulas here.

``CRUN`` is the end-to-end ARC counter used for latency. ``SEU`` is retained as
resource occupancy because GVML fragments execute asynchronously from ARC
control. All numbers are cycles; there is no clock conversion.

Sources:

* local GSI GVML/GAL manuals and profiling tutorial;
* APU-Performance-Modeling commit 20161ca64985b234b35b0028b633cbea21c922f4;
* real-device microprofiles on 2026-07-01 for FP16 add/sub/mul and FP16 group
  reduction at group sizes 64, 128, and 256;
* MICRO'25 APU artifact uint16 profiles in
  ``micro-experiments/power-profile/latencies.csv``;
* the fitted affine event model of the paper's agent-assisted calibration
  (``APU_V1_AFFINE_EVENT_MODEL`` below), which prices the aggregated events of
  the native record-frequency and histogram recipes.

Every estimate this program prices for a native recipe carries a validity
label (``apu_v1_cost_validity``): ``calibrated`` inside the plan families and
shapes the fitted model was validated on, ``extrapolated`` elsewhere.  See
``docs/release/design/apu-v1-cost.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

from ...perf import cost, rule


VR_LANES = 32768

# Amortized over 100 calls on real silicon.
F16_ADD_SEU = 204
F16_ADD_CRUN = 264
F16_SUB_SEU = 209
F16_SUB_CRUN = 290
F16_MUL_SEU = 196
F16_MUL_CRUN = 254

# Full 32K helper measurements from the supplied performance-modeling sweep.
DMA_L4_TO_L1_SEU = 81
DMA_L4_TO_L1_CRUN = 22731
DMA_L1_TO_L4_SEU = 81
DMA_L1_TO_L4_CRUN = 21886
GVML_LOAD_SEU = 8
GVML_LOAD_CRUN = 77
GVML_STORE_SEU = 10
GVML_STORE_CRUN = 71

# Whole grouped-kernel calibration on the four-APUC launcher. Holding the
# generated body constant and changing only its VR-batch count measured
# b=1:183149 and b=5:625524 CRUN, versus primitive sums of 95943 and 479715.
# The two points separate launch/GVML initialization from per-batch ARC loop
# and command overhead without folding either into arithmetic/DMA primitives.
GROUPED_KERNEL_STARTUP_CRUN = 72_555
GROUPED_BATCH_CONTROL_CRUN = 14_651

# Scalar ARC fallback. Retained MLIR operation counts are higher-level than ARC
# instructions. Canonical SMALL GEMM measured 140,556,657 CRUN versus 2,730,367
# retained operations, yielding 51.47 CRUN/op; use the conservative integer 52
# until the complete PolyBench corpus supplies an operation-mix regression.
ARC_SCALAR_CPI = 52
ARC_SCALAR_STARTUP_CRUN = 72_555

# MICRO'25 Tables 4 and 5.  These model the integer/logical and data-layout
# operations used by the binary-matmul vectorization candidates.  They remain
# separate from the newer FP16 microprofiles above because the measurements
# use different GVML operations and amortization protocols.
MICRO_DMA_L4_L3_SLOPE = 0.19
MICRO_DMA_L4_L3_STARTUP = 41_164
MICRO_DMA_L4_L2_SLOPE = 0.63
MICRO_DMA_L4_L2_STARTUP = 548
MICRO_DMA_L2_L1_32K = 386
MICRO_DMA_L4_L1_32K = 22_272
MICRO_DMA_L1_L4_32K = 22_186
MICRO_PIO_LOAD_PER_ELEMENT = 57
MICRO_PIO_STORE_PER_ELEMENT = 61
# Table 4's per-element PIO fit omits ARC/GVML issue setup because its sweep
# amortizes that setup inside one call.  Full-board layout experiments expose
# it: spatial reduction issues 32 large operand calls while scalar-vector
# temporal accumulation issues 2,048 32K calls.  Fitting the measured
# baseline/SVP delta after the calibrated vector primitives gives 3,365 CRUN
# per call.  This is a route-call cost, independent of kernel name and shape.
MICRO_PIO_CALL_STARTUP = 3_365
MICRO_VR_LOAD_STORE = 29
MICRO_LOOKUP_SLOPE = 7.15
MICRO_LOOKUP_STARTUP = 629
# Direct-L4 lookup is legal but pays the backing-memory access on every table
# entry.  Two route-controlled board pairs isolate that increment while
# keeping the GVML loop identical: M256xN160xK160 (128-entry tables) measured
# 1,170,053 CRUN from L3 versus 2,862,521 from L4; M256xN1024xK1200
# (32-entry tables) measured 10,760,275 versus 23,778,223.  Adding back the
# one-time L4->L3 copy predicts 42.71 and 42.89 CRUN per lookup-table entry.
MICRO_LOOKUP_L4_SLOPE = 42.8
MICRO_COPY_IMMEDIATE = 13
MICRO_RESET = 16
MICRO_XOR = 12
MICRO_NOT = 10
MICRO_POPCOUNT = 23
MICRO_SHIFT = 15
MICRO_ADD_S16 = 13
MICRO_SUB_S16 = 16
MICRO_ADD_U16 = 12
MICRO_SUB_U16 = 13
MICRO_MUL_U16 = 114
MICRO_SUM_U16_G128 = 527
MICRO_CREATE_GROUP_INDEX = 33
MICRO_CREATE_SUBGROUP_INDEX = 37
MICRO_DUPLICATE_SUBGROUP_8K_1K = 1_915
# Native marker/scalar calls used by complete reduction recipes. Equality is
# the MICRO'25 ``gvml_eq_16`` measurement; immediate equality uses the same
# element-wise fragment. Marker AND shares the measured 16-bit logical issue
# path. The supplied analytical operation table records 245 cycles for a 32K
# marker count and 60 cycles for one VR extraction. The documented fast paired
# count traverses the same 32K frontier once while returning two counts, so it
# inherits the single-traversal calibration. Direct scalar L4 stores use the
# conservative measured per-element PIO store cost.
MICRO_EQ_IMM_16 = 13
MICRO_AND_M = 13
MICRO_COUNT_M_G32K = 245
MICRO_FAST_COUNT_2M_G32K = MICRO_COUNT_M_G32K
MICRO_GET_ENTRY_16 = 60
MICRO_ARC_STORE_L4_U16 = MICRO_PIO_STORE_PER_ELEMENT


@dataclass(frozen=True)
class APUv1AffineEvent:
    """``round(activation + slope * count)`` CRUN cycles for one aggregated event.

    The activation is charged once per aggregated event of a native recipe
    graph (one event per GVML primitive family per launch); a zero count costs
    nothing.  ``low_*``/``high_*`` name the two microbenchmarks whose median
    CRUN the pair was solved from; each median includes the empty-route
    intercept, so ``intercept + cycles(count)`` reproduces it.
    """

    feature: str
    primitive: str
    label: str
    activation_cycles: float
    slope_cycles_per_call: float
    low_case: str
    low_count: int
    high_case: str
    high_count: int
    measured_medians_crun: tuple[float, float]

    def cycles(self, count: int) -> int:
        if count <= 0:
            return 0
        return max(
            1, int(round(self.activation_cycles + self.slope_cycles_per_call * count))
        )

    def manifest(self) -> dict:
        return {
            "feature": self.feature,
            "primitive": self.primitive,
            "activation_cycles": self.activation_cycles,
            "slope_cycles_per_call": self.slope_cycles_per_call,
            "low_case": self.low_case,
            "low_count": self.low_count,
            "high_case": self.high_case,
            "high_count": self.high_count,
            "measured_medians_crun": list(self.measured_medians_crun),
            "unit": "ARC CRUN device cycles",
        }


@dataclass(frozen=True)
class APUv1AffineEventModel:
    """The fitted event model with its provenance and validated domain."""

    route_intercept_cycles: float
    l4_l3_activation_cycles: float
    l4_l3_cycles_per_byte: float
    events: tuple[APUv1AffineEvent, ...]
    # Native routes whose graphs are priced by the model (one aggregated
    # event per primitive family).
    priced_routes: frozenset[str]
    # route -> {parameter: (min, max)} of the shapes the model was validated
    # on against physical measurements without refitting.
    validated_envelopes: MappingProxyType
    provenance: MappingProxyType

    def event(self, primitive: str) -> APUv1AffineEvent | None:
        return next((e for e in self.events if e.primitive == primitive), None)

    def l4_l3_cycles(self, transactions: int, byte_count: int) -> int:
        if transactions <= 0:
            return 0
        return max(
            1,
            int(
                round(
                    self.l4_l3_activation_cycles * transactions
                    + self.l4_l3_cycles_per_byte * byte_count
                )
            ),
        )

    def manifest(self) -> dict:
        return {
            "route_intercept_cycles": self.route_intercept_cycles,
            "l4_l3_activation_cycles": self.l4_l3_activation_cycles,
            "l4_l3_cycles_per_byte": self.l4_l3_cycles_per_byte,
            "events": [event.manifest() for event in self.events],
            "priced_routes": sorted(self.priced_routes),
            "validated_envelopes": {
                route: {name: list(bounds) for name, bounds in envelope.items()}
                for route, envelope in self.validated_envelopes.items()
            },
            "provenance": dict(self.provenance),
        }


RECORD_FREQUENCY_ROUTE = "gvml_record_frequency_resident_chunks_pair_count"
DENSE_HISTOGRAM_ROUTE = "gvml_dense_histogram_pair_count"

# Fitted by the agent-assisted calibration study (paper Section 5, "Agent-Assisted
# Cost Calibration") from 15 single-primitive microbenchmarks on one Gemini-I
# board.  Values are copied verbatim (full precision) from
# tenon-artifacts/agent-cost-calibration/2026-07-27-gemini-i/
# confirmatory-affine-event-v1/freeze/affine-event-model.json (model fingerprint
# 0f26de9d...dcfef); the rebuttal transfer studies reuse the same coefficients
# (frozen-affine-model.json sha256 f89d8de6...be16).  Each pair is solved from the
# median CRUN (7 measured runs after one warm-up) of a low/high call-count case.
# The route intercept is the empty-route case q00; the L4->L3 terms come from
# q01/q02.  The 15-observation fit has 15 parameters (saturated); adequacy was
# tested only on held-out compositions (validated_envelopes).
APU_V1_AFFINE_EVENT_MODEL = APUv1AffineEventModel(
    route_intercept_cycles=2558.0,
    l4_l3_activation_cycles=1842.3333333333333,
    l4_l3_cycles_per_byte=0.7766927083333334,
    events=(
        APUv1AffineEvent(
            "dma_l4_l1_32k",
            "DMA_L4_TO_L1_32K",
            "direct_dma_l4_to_l1_32k",
            8477.0,
            21808.0,
            "q03-l4-l1-r1",
            1,
            "q04-l4-l1-r4",
            4,
            (32843.0, 98267.0),
        ),
        APUv1AffineEvent(
            "gvml_load_16",
            "LOAD_L1_TO_VR16",
            "gvml_load_16",
            1871.3650793650793,
            25.634920634920636,
            "q05-load-r1",
            1,
            "q06-load-r64",
            64,
            (4455.0, 6070.0),
        ),
        APUv1AffineEvent(
            "gvml_eq_imm_16",
            "EQ_IMM_16",
            "gvml_eq_imm_16",
            1880.7019607843138,
            24.298039215686273,
            "q07-eq-r1",
            1,
            "q08-eq-r256",
            256,
            (4463.0, 10659.0),
        ),
        APUv1AffineEvent(
            "gvml_and_m",
            "AND_M",
            "gvml_and_m",
            908.7905759162304,
            27.209424083769633,
            "q09-and-r1",
            1,
            "q10-and-r192",
            192,
            (3494.0, 8691.0),
        ),
        # The paper's example: "the fitted count rule adds a 5,887.79-cycle
        # start-up cost to 876.21 cycles per call".
        APUv1AffineEvent(
            "gvml_2_fast_count_m_g32k",
            "FAST_COUNT_2M_G32K",
            "gvml_2_fast_count_m_g32k",
            5887.793650793651,
            876.2063492063492,
            "q11-paired-count-r1",
            1,
            "q12-paired-count-r64",
            64,
            (9322.0, 64523.0),
        ),
        APUv1AffineEvent(
            "scalar_l4_store_u16",
            "ARC_STORE_L4_U16",
            "scalar_l4_store_u16",
            1235.1259842519685,
            17.874015748031496,
            "q13-store-r1",
            1,
            "q14-store-r128",
            128,
            (3811.0, 6081.0),
        ),
    ),
    priced_routes=frozenset({RECORD_FREQUENCY_ROUTE, DENSE_HISTOGRAM_ROUTE}),
    validated_envelopes=MappingProxyType(
        {
            # Confirmatory shapes (C, Q) = (2,160), (6,64), (9,24) and rebuttal
            # plans (1,128), (3,64), (5,32), (8,16); 32,768 records; even Q.
            RECORD_FREQUENCY_ROUTE: MappingProxyType(
                {"chunks": (1, 9), "queries": (16, 160)}
            ),
            # Rebuttal histogram plans: 256 bins over one 32K frontier,
            # 1..128 frontier reloads (the released recipe loads it once).
            DENSE_HISTOGRAM_ROUTE: MappingProxyType(
                {"bins": (256, 256), "frontier_loads": (1, 128)}
            ),
        }
    ),
    provenance=MappingProxyType(
        {
            "unit": "ARC CRUN device cycles (PROF total, no clock conversion)",
            "algorithm": "paired-affine-event-v1",
            "model_fingerprint": (
                "0f26de9d9286e4efd1d880d70d947fd3000ea60b1a6dfb9548a5af027b6dcfef"
            ),
            "model_file": (
                "tenon-artifacts/agent-cost-calibration/2026-07-27-gemini-i/"
                "confirmatory-affine-event-v1/freeze/affine-event-model.json"
            ),
            "microbenchmarks": (
                "tenon-artifacts/agent-cost-calibration/2026-07-27-gemini-i/"
                "negative-pilot-c4q96-v3/measurements/calibration/q00..q14*.json "
                "(15 cases, 1 warm-up + 7 runs each, median CRUN)"
            ),
            "measured_utc": "2026-07-27T22:31Z..22:57Z",
            "board": (
                "GSI Gemini-I PCI 1e4c:0010 rev 00 on zhang-capra-xcel, APUC 0; "
                "platform fingerprint 1c28bbaf8ca49bdd6445a94a288bdc8f2f7414e5966493a5542f148ae6213974"
            ),
            "software": (
                "GSI APU SDK 13.7.1, ARC GNU 2021.09, ledag-ssh 100.12.0.1.1000.25; "
                "board firmware version not recorded in the archive"
            ),
            "fitted_on": (
                "single-primitive microbenchmarks inside the native "
                "record-frequency launch harness"
            ),
            "validated_on": (
                "record frequency: 3 confirmatory shapes (2026-07-27) and 8 rebuttal "
                "plans (2026-09-29, tenon-artifacts/rebuttal/reviewer-a/q1/evidence/"
                "results/2026-09-29-gemini-calibration-v4); dense histogram: 8 "
                "residency plans (2026-09-29, ...-gemini-histogram-validation-v1)"
            ),
        }
    ),
)


def _affine_priced(event) -> bool:
    attributes = event.attributes
    return bool(attributes.get("native_aggregate")) and (
        attributes.get("route") in APU_V1_AFFINE_EVENT_MODEL.priced_routes
    )


def _route_entry_step(event, ctx, apuc, arc):
    if event.attributes.get("route_entry"):
        ctx.step(
            latency=max(
                1, int(round(APU_V1_AFFINE_EVENT_MODEL.route_intercept_cycles))
            ),
            occupy=[ctx.use(apuc), ctx.use(arc)],
            name="calibrated_route_intercept",
        )


def apu_v1_cost_validity(route, inventory, cost=None) -> dict:
    """Validity label of a native-recipe estimate under ``cost``.

    ``calibrated``: ``cost`` carries ``APU_V1_AFFINE_EVENT_MODEL``, the route is
    one of the validated families, every event has a fitted pair (no odd-count
    ``gvml_count_m_g32k``), and the shape lies inside the validated envelope.
    Everything else is ``extrapolated``, with the reasons listed.
    """

    spec = getattr(cost, "spec", cost)
    model = getattr(spec, "apu_v1_calibration", None)
    inventory = dict(inventory or {})
    reasons = []
    shape = {}
    if model is None:
        reasons.append("cost program carries no APU v1 calibration record")
    elif route not in model.validated_envelopes:
        reasons.append(
            f"route {route!r} is outside the fitted families "
            f"({', '.join(sorted(model.validated_envelopes))})"
        )
    else:
        if int(inventory.get("gvml_count_m_g32k", 0)):
            reasons.append(
                "odd pair count issues gvml_count_m_g32k, which has no fitted "
                "activation/slope pair"
            )
        if route == RECORD_FREQUENCY_ROUTE:
            shape = {
                "chunks": int(inventory.get("dma_l4_to_l1_32k", 0)),
                "queries": int(inventory.get("scalar_l4_store_u16", 0)),
            }
        elif route == DENSE_HISTOGRAM_ROUTE:
            shape = {
                "bins": int(inventory.get("scalar_l4_store_u16", 0)),
                "frontier_loads": int(inventory.get("dma_l4_to_l1_32k", 0)),
            }
        for name, (low, high) in model.validated_envelopes[route].items():
            value = shape.get(name)
            if value is None or not low <= value <= high:
                reasons.append(f"{name}={value} outside validated [{low}, {high}]")
    return {
        "cost_validity": "extrapolated" if reasons else "calibrated",
        "route": route,
        "shape": shape,
        "basis": (
            "fitted affine event model (APU_V1_AFFINE_EVENT_MODEL) inside its "
            "validated plan family and shape envelope"
            if not reasons
            else "; ".join(reasons)
        ),
        "validated_envelope": (
            None
            if model is None or route not in model.validated_envelopes
            else {
                name: list(bounds)
                for name, bounds in model.validated_envelopes[route].items()
            }
        ),
        "model_fingerprint": (
            None if model is None else model.provenance["model_fingerprint"]
        ),
    }


def _ceil_div(value, divisor):
    if divisor <= 0:
        raise ValueError("cost divisor must be positive")
    return math.ceil(value / divisor)


def _metric(event, name, default=1):
    return max(0, int(event.metrics.get(name, default)))


def _count(event):
    return max(1, int(event.metrics.get("count", 1)))


def _group_size(event):
    candidate = event.metrics.get("candidate", {}) or {}
    value = int(candidate.get("group_size", event.metrics.get("group_size", 1)))
    if value <= 0 or value > VR_LANES or value & (value - 1):
        raise ValueError(f"APU v1 group size must be a power of two <=32K, got {value}")
    return value


def _vector_batches(event, *, grouped=False):
    candidate = event.metrics.get("candidate", {}) or {}
    if grouped and "n_out_tiles" in candidate:
        return max(1, int(candidate["n_out_tiles"]))
    iterations = _metric(event, "iterations")
    if not grouped:
        return max(1, _ceil_div(iterations, VR_LANES))
    reduction = max(1, _metric(event, "reduction_extent"))
    outputs = max(1, _ceil_div(iterations, reduction))
    groups_per_vr = VR_LANES // _group_size(event)
    return max(1, _ceil_div(outputs, groups_per_vr))


def _reduce_cycles(group_size):
    """Fit the real FP16 group-reduction measurements.

    Measured per-call points were:
    ``g64=(1724,4931)``, ``g128=(1962,5308)``, and
    ``g256=(2213,5652)`` for ``(SEU, CRUN)``. A linear fit in log2(group)
    reflects the reduction tree and interpolates other legal powers of two.
    """
    measured = {
        64: (1724, 4931),
        128: (1962, 5308),
        256: (2213, 5652),
    }
    if group_size in measured:
        return measured[group_size]
    level = math.log2(group_size)
    seu = max(1, round(244.5 * level + 257.0))
    crun = max(1, round(360.5 * level + 2768.0))
    return seu, crun


def _reduce_u16_cycles(group_size):
    """Scale the measured uint16 128-way reduction by tree depth."""

    return max(1, round(MICRO_SUM_U16_G128 * math.log2(group_size) / 7))


@cost(target="apu_v1")
def apu_v1_cost(target):
    """Interpret GVML operations over APUC, ARC, SEU, and DMA resources."""
    apuc = target.unit("apuc")
    arc = target.unit("arc")
    seu = target.unit("seu")
    dma = target.unit("dma")

    def micro_vector_rule(operation, per_call, label):
        @rule(operation)
        def implementation(event, ctx):
            if _affine_priced(event):
                _route_entry_step(event, ctx, apuc, arc)
                affine = APU_V1_AFFINE_EVENT_MODEL.event(operation.name)
                if affine is not None:
                    cycles = affine.cycles(_count(event))
                else:
                    cycles = max(1, int(round(per_call(event)))) * _count(event)
            else:
                cycles = max(1, int(round(per_call(event)))) * _count(event)
            ctx.step(
                latency=cycles,
                occupy=[
                    ctx.use(event.primitive, cycles=cycles),
                    ctx.use(apuc, cycles=cycles),
                    ctx.use(arc, cycles=cycles),
                    ctx.use(seu, cycles=cycles),
                ],
                name=label,
            )

        return implementation

    for op_name, cycles, label in (
        ("RESET_16", MICRO_RESET, "gvml_reset_16"),
        ("CPY_IMM_16", MICRO_COPY_IMMEDIATE, "gvml_cpy_imm_16"),
        ("XOR_16", MICRO_XOR, "gvml_xor_16"),
        # The MICRO artifact measures gvml_and_16 at 13 cycles.  OR uses the
        # same element-wise logical datapath; retain that explicit inference
        # until a dedicated microprofile supersedes it.
        ("AND_16", 13, "gvml_and_16"),
        ("OR_16", 13, "gvml_or_16_inferred"),
        ("NOT_16", MICRO_NOT, "gvml_not_16"),
        ("POPCOUNT_16", MICRO_POPCOUNT, "gvml_popcount_16"),
        ("SHL_IMM_16", MICRO_SHIFT, "gvml_sl_imm_16"),
        ("ADD_U16", MICRO_ADD_U16, "gvml_add_u16"),
        ("ADD_S16", MICRO_ADD_S16, "gvml_add_s16"),
        ("SUB_U16", MICRO_SUB_U16, "gvml_sub_u16"),
        ("SUB_S16", MICRO_SUB_S16, "gvml_sub_s16"),
        ("MUL_U16", MICRO_MUL_U16, "gvml_mul_u16"),
        (
            "CREATE_GROUP_INDEX_16",
            MICRO_CREATE_GROUP_INDEX,
            "gvml_create_grp_index_u16",
        ),
        (
            "CREATE_SUBGROUP_INDEX_16",
            MICRO_CREATE_SUBGROUP_INDEX,
            "gvml_create_subgrp_index_u16",
        ),
        (
            "DUPLICATE_SUBGROUP_16",
            MICRO_DUPLICATE_SUBGROUP_8K_1K,
            "gvml_duplicate_subgrp_16_grp_sgidx",
        ),
    ):
        micro_vector_rule(target.op(op_name), lambda _event, value=cycles: value, label)

    for op_name, cycles, label in (
        ("EQ_IMM_16", MICRO_EQ_IMM_16, "gvml_eq_imm_16"),
        ("AND_M", MICRO_AND_M, "gvml_and_m"),
        ("COUNT_M_G32K", MICRO_COUNT_M_G32K, "gvml_count_m_g32k"),
        (
            "FAST_COUNT_2M_G32K",
            MICRO_FAST_COUNT_2M_G32K,
            "gvml_2_fast_count_m_g32k",
        ),
        ("GET_ENTRY_16", MICRO_GET_ENTRY_16, "gvml_get_entry_16"),
    ):
        micro_vector_rule(target.op(op_name), lambda _event, value=cycles: value, label)

    micro_vector_rule(
        target.op("LOOKUP_16"),
        lambda event: MICRO_LOOKUP_STARTUP
        + MICRO_LOOKUP_SLOPE * _metric(event, "table_size")
        + MICRO_LOOKUP_L4_SLOPE
        * _metric(event, "table_size")
        * _metric(event, "source_is_l4", 0),
        "gvml_lookup_16",
    )
    micro_vector_rule(
        target.op("GROUP_REDUCE_ADD_U16"),
        lambda event: _reduce_u16_cycles(
            max(1, _metric(event, "group_size", VR_LANES))
        ),
        "gvml_add_subgrps_u16_grp",
    )
    micro_vector_rule(
        target.op("GROUP_REDUCE_ADD_S16"),
        lambda event: _reduce_cycles(max(1, _metric(event, "group_size", VR_LANES)))[1],
        "gvml_add_subgrps_s16_grp",
    )

    def gvml_step(event, ctx, *, seu_cycles, crun_cycles, name, repeats=1):
        seu_cycles *= int(repeats)
        crun_cycles *= int(repeats)
        ctx.step(
            latency=crun_cycles,
            occupy=[
                ctx.use(event.primitive, cycles=crun_cycles),
                ctx.use(apuc, cycles=crun_cycles),
                ctx.use(arc, cycles=crun_cycles),
                ctx.use(seu, cycles=min(seu_cycles, crun_cycles)),
            ],
            name=name,
        )

    @rule(target.op("ADD"))
    def add(event, ctx):
        gvml_step(
            event,
            ctx,
            seu_cycles=F16_ADD_SEU,
            crun_cycles=F16_ADD_CRUN,
            name="gvml_add_f16",
            repeats=_vector_batches(event),
        )

    @rule(target.op("SUB"))
    def subtract(event, ctx):
        gvml_step(
            event,
            ctx,
            seu_cycles=F16_SUB_SEU,
            crun_cycles=F16_SUB_CRUN,
            name="gvml_sub_f16",
            repeats=_vector_batches(event),
        )

    @rule(target.op("MUL"))
    def multiply(event, ctx):
        gvml_step(
            event,
            ctx,
            seu_cycles=F16_MUL_SEU,
            crun_cycles=F16_MUL_CRUN,
            name="gvml_mul_f16",
            repeats=_vector_batches(event),
        )

    @rule(target.op("GROUP_REDUCE_ADD_F16"))
    def group_reduce(event, ctx):
        seu_cycles, crun_cycles = _reduce_cycles(_group_size(event))
        gvml_step(
            event,
            ctx,
            seu_cycles=seu_cycles,
            crun_cycles=crun_cycles,
            name="gvml_add_subgrps_f16_grp",
            repeats=_vector_batches(event, grouped=True),
        )

    @rule(target.op("MAC"))
    def grouped_mac(event, ctx):
        batches = _vector_batches(event, grouped=True)
        reduce_seu, reduce_crun = _reduce_cycles(_group_size(event))
        ctx.step(
            latency=GROUPED_KERNEL_STARTUP_CRUN,
            occupy=[ctx.use(apuc), ctx.use(arc)],
            name="grouped_kernel_startup",
        )
        with ctx.repeat(batches):
            ctx.step(
                latency=GROUPED_BATCH_CONTROL_CRUN,
                occupy=[ctx.use(apuc), ctx.use(arc)],
                name="grouped_batch_control",
            )
            gvml_step(
                event,
                ctx,
                seu_cycles=F16_MUL_SEU,
                crun_cycles=F16_MUL_CRUN,
                name="gvml_mul_f16",
            )
            gvml_step(
                event,
                ctx,
                seu_cycles=reduce_seu,
                crun_cycles=reduce_crun,
                name="gvml_add_subgrps_f16_grp",
            )

    @rule(target.op("SCALAR_C"))
    def scalar_c(event, ctx):
        instructions = max(1, int(event.metrics.get("instructions", 1)))
        cycles = ARC_SCALAR_STARTUP_CRUN + instructions * ARC_SCALAR_CPI
        ctx.step(
            latency=cycles,
            occupy=[
                ctx.use(event.primitive, cycles=cycles),
                ctx.use(apuc, cycles=cycles),
                ctx.use(arc, cycles=cycles),
            ],
            name="scalar_arc_c",
        )

    def load_rule(move):
        @rule(move)
        def load(event, ctx):
            with ctx.repeat(_vector_batches(event, grouped=True)):
                ctx.step(
                    latency=DMA_L4_TO_L1_CRUN,
                    occupy=[
                        ctx.use(event.primitive),
                        ctx.use(apuc),
                        ctx.use(arc),
                        ctx.use(dma),
                        ctx.use(target.l4),
                    ],
                    name="direct_dma_l4_to_l1_32k",
                )
                ctx.step(
                    latency=GVML_LOAD_CRUN,
                    occupy=[ctx.use(apuc), ctx.use(arc), ctx.use(seu)],
                    name="gvml_load_16",
                )

    for move_name in ("LD_X_L4_TO_VR", "LD_Y_L4_TO_VR", "LD_ACC_L4_TO_VR"):
        load_rule(target.move(move_name))

    def micro_move_rule(move, latency_fn, label, resources):
        @rule(move)
        def implementation(event, ctx):
            if _affine_priced(event):
                _route_entry_step(event, ctx, apuc, arc)
                affine = APU_V1_AFFINE_EVENT_MODEL.event(move.name)
                if move.name == "DMA_L4_TO_L3":
                    cycles = APU_V1_AFFINE_EVENT_MODEL.l4_l3_cycles(
                        _count(event), _metric(event, "bytes", 0)
                    )
                elif affine is not None:
                    cycles = affine.cycles(_count(event))
                else:
                    cycles = max(1, int(round(latency_fn(event)))) * _count(event)
            else:
                cycles = max(1, int(round(latency_fn(event)))) * _count(event)
            ctx.step(
                latency=cycles,
                occupy=[
                    ctx.use(event.primitive, cycles=cycles),
                    ctx.use(apuc, cycles=cycles),
                    ctx.use(arc, cycles=cycles),
                    ctx.use(dma, cycles=cycles),
                    *(ctx.use(resource, cycles=cycles) for resource in resources),
                ],
                name=label,
            )

        return implementation

    micro_move_rule(
        target.move("DMA_L4_TO_L3"),
        lambda event: MICRO_DMA_L4_L3_STARTUP
        + MICRO_DMA_L4_L3_SLOPE * _metric(event, "bytes"),
        "dma_l4_l3",
        (target.l4, target.l3),
    )
    micro_move_rule(
        target.move("DMA_L4_TO_L2"),
        lambda event: MICRO_DMA_L4_L2_STARTUP
        + MICRO_DMA_L4_L2_SLOPE * _metric(event, "bytes"),
        "dma_l4_l2",
        (target.l4, target.l2),
    )
    for move_name, cycles, label, resources in (
        (
            "DMA_L2_TO_L1_32K",
            MICRO_DMA_L2_L1_32K,
            "direct_dma_l2_to_l1_32k",
            (target.l2, target.vmrs),
        ),
        (
            "DMA_L4_TO_L1_32K",
            MICRO_DMA_L4_L1_32K,
            "direct_dma_l4_to_l1_32k",
            (target.l4, target.vmrs),
        ),
        (
            "DMA_L1_TO_L4_32K",
            MICRO_DMA_L1_L4_32K,
            "direct_dma_l1_to_l4_32k",
            (target.vmrs, target.l4),
        ),
        (
            "LOAD_L1_TO_VR16",
            MICRO_VR_LOAD_STORE,
            "gvml_load_16",
            (target.vmrs,),
        ),
        (
            "STORE_VR16_TO_L1",
            MICRO_VR_LOAD_STORE,
            "gvml_store_16",
            (target.vmrs,),
        ),
    ):
        micro_move_rule(
            target.move(move_name),
            lambda _event, value=cycles: value,
            label,
            resources,
        )
    micro_move_rule(
        target.move("PIO_L4_TO_VR16"),
        lambda event: MICRO_PIO_CALL_STARTUP
        + MICRO_PIO_LOAD_PER_ELEMENT * _metric(event, "elements"),
        "pio_l4_to_vr16",
        (target.l4,),
    )
    micro_move_rule(
        target.move("PIO_VR16_TO_L4"),
        lambda event: MICRO_PIO_CALL_STARTUP
        + MICRO_PIO_STORE_PER_ELEMENT * _metric(event, "elements"),
        "pio_vr16_to_l4",
        (target.l4,),
    )

    @rule(target.move("ARC_STORE_L4_U16"))
    def arc_store_l4_u16(event, ctx):
        if _affine_priced(event):
            _route_entry_step(event, ctx, apuc, arc)
            cycles = APU_V1_AFFINE_EVENT_MODEL.event("ARC_STORE_L4_U16").cycles(
                _count(event)
            )
        else:
            cycles = MICRO_ARC_STORE_L4_U16 * _count(event)
        ctx.step(
            latency=cycles,
            occupy=[
                ctx.use(event.primitive, cycles=cycles),
                ctx.use(apuc, cycles=cycles),
                ctx.use(arc, cycles=cycles),
                ctx.use(target.l4, cycles=cycles),
            ],
            name="scalar_l4_store_u16",
        )

    @rule(target.move("ST_ACC_VR_TO_L4"))
    def store(event, ctx):
        with ctx.repeat(_vector_batches(event, grouped=True)):
            ctx.step(
                latency=GVML_STORE_CRUN,
                occupy=[ctx.use(apuc), ctx.use(arc), ctx.use(seu)],
                name="gvml_store_16",
            )
            ctx.step(
                latency=DMA_L1_TO_L4_CRUN,
                occupy=[
                    ctx.use(event.primitive),
                    ctx.use(apuc),
                    ctx.use(arc),
                    ctx.use(dma),
                    ctx.use(target.l4),
                ],
                name="direct_dma_l1_to_l4_32k",
            )


# Read by the native graph builder for the estimate validity label.
apu_v1_cost.apu_v1_calibration = APU_V1_AFFINE_EVENT_MODEL


__all__ = [
    "APU_V1_AFFINE_EVENT_MODEL",
    "APUv1AffineEvent",
    "APUv1AffineEventModel",
    "apu_v1_cost",
    "apu_v1_cost_validity",
]
