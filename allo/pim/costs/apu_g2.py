# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable compute costs for Gemini-II direct-VL64 kernels.

Typed and composed recipes (``APUG2TypedProgram``,
``APUG2ComposedContractionProgram`` and the source route built on them) emit
one cost event per issued call carrying ``typed_call_features``; the rules
registered by :func:`build_apu_g2_cost` price them with an
:class:`~allo.pim.apu_g2_pipeline.APUG2TypedTickModel` (device ticks per
pipeline body, calibrated on the 2026-11-02 real-card runs; provenance,
validity domain and fit in ``allo/pim/apu_g2_pipeline.py`` and
``docs/release/design/apu-g2-source-route.md``).  The uint16 vector add
(``allo.pim.apu_g2_runtime.run_apu_g2_u16_add``) keeps its pipeline
calibration.

One ``ADD_U16`` cost event represents the complete core-wide sequence: two
L1-to-MMB copies, one add, one MMB-to-L1 copy, and the mandatory SEU barrier.
The sixteen logical L1 groups are coalesced into that one event. Device ticks
remain separate from wall time.

The constants below are calibrated from real-card runs on 2026-07-08.  The
current registered task library measures 68,138 device ticks over 256 complete
vector-add pipelines, or 266.164 ticks per pipeline; the integer cost model
uses the repeated-throughput floor of 266 ticks.  Isolated phase slopes provide
the relative attribution; they are normalized so these sequential graph stages
sum to the measured overlapped pipeline throughput.  Host/PCIe time is not
mixed into this device-tick model.
"""

from __future__ import annotations

from numbers import Integral

from ...perf import BoundCostSpec, CostEvent, CostSpec, ExecutionGraph, rule
from ..apu_g2_pipeline import (
    APUG2_TYPED_TICK_FEATURES,
    APUG2_TYPED_TICK_MEASUREMENTS,
    APUG2_TYPED_TICK_MODEL,
    APUG2TypedTickModel,
    apu_g2_typed_call_features,
    STANDALONE_4X64K_ADD_CALIBRATION,
    Opcode,
    PipelineCalibration,
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

APUG2_U16_NUM_VECTORS = 4
APUG2_U16_LANES = 65_536
APUG2_U16_GROUPS = 16
APUG2_U16_LANES_PER_GROUP = 4_096
APUG2_U16_NUM_BITS = 16


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


_SUPPORTED_PIPELINE_CALIBRATIONS = (STANDALONE_4X64K_ADD_CALIBRATION,)


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


def _typed_call_features(event: CostEvent):
    features = event.metrics.get("typed_call_features")
    if features is None:
        return None
    if not isinstance(features, dict):
        raise TypeError("typed_call_features must be a feature mapping")
    return features


def _emit_typed_call(ctx, target, primitive, tick_model, features, name):
    """One issued typed/composed call priced by the typed tick model."""

    kind_occupancy = [ctx.use(primitive), ctx.use(target.unit("core"))]
    kind_occupancy.append(ctx.use(target.unit("arc")))
    if not features["pipeline_bodies"]:
        kind_occupancy.append(ctx.use(target.unit("vector_engine")))
        kind_occupancy.append(ctx.use(target.mmb))
    if features["row_bits"] and name.startswith("copy_"):
        kind_occupancy.append(ctx.use(target.l1))
    ctx.step(
        latency=tick_model.call_ticks(features),
        occupy=kind_occupancy,
        name=name,
    )


def _apu_g2_cost_builder(tick_model: APUG2TypedTickModel):
    def apu_g2_cost(target):
        """Expand Gemini-II kernels onto core-wide VL64 resources.

        uint16 events carry a ``pipeline_calibration``; typed/composed recipe
        calls carry ``typed_call_features`` and are priced by ``tick_model``.
        """

        _register_apu_g2_rules(target, tick_model)

    return apu_g2_cost


def build_apu_g2_cost(
    tick_model: APUG2TypedTickModel = APUG2_TYPED_TICK_MODEL,
    *,
    name: str = "apu_g2_cost",
) -> CostSpec:
    """The Gemini-II executable cost program for one typed tick model.

    ``apu_g2_cost`` is this program with :data:`APUG2_TYPED_TICK_MODEL`.  A
    perturbed model (``APUG2_TYPED_TICK_MODEL.scaled(...)``) yields a distinct
    cost program and fingerprint; legality and lowering do not depend on it.
    """

    if not isinstance(tick_model, APUG2TypedTickModel):
        raise TypeError("tick_model must be an APUG2TypedTickModel")
    spec = CostSpec(
        _apu_g2_cost_builder(tick_model),
        target="apu_v2",
        name=name,
        fingerprint_data={
            "direct_vl64_calibrations": [
                STANDALONE_4X64K_ADD_CALIBRATION.canonical_manifest,
            ],
            "typed_tick_model": tick_model.canonical_manifest,
        },
    )
    # Read by the typed/composed graph builders for the estimate validity.
    spec.apu_g2_tick_model = tick_model
    return spec


def _register_apu_g2_rules(target, tick_model):
    core = target.unit("core")
    arc = target.unit("arc")
    vector_engine = target.unit("vector_engine")
    copy_segment0 = target.move("L1_TO_MMB_SEG0")
    copy_segment1 = target.move("L1_TO_MMB_SEG1")
    copy_to_l1 = target.move("MMB_TO_L1")
    add_u16 = target.op("ADD_U16")
    seu_barrier = target.op("SEU_BARRIER")

    @rule(add_u16)
    def vector_add_u16(event, ctx):
        calibration = _event_pipeline_calibration(event)
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

    @rule(seu_barrier)
    def typed_seu_barrier(event, ctx):
        features = _typed_call_features(event)
        if features is None:
            raise ValueError("standalone barriers are expanded by the ADD rule")
        _emit_typed_call(ctx, target, seu_barrier, tick_model, features, "seu_barrier")

    def typed_rule(primitive, label):
        def price(event, ctx):
            features = _typed_call_features(event)
            if features is None:
                raise ValueError(
                    f"{label} is a typed/composed call and requires "
                    "typed_call_features"
                )
            _emit_typed_call(ctx, target, primitive, tick_model, features, label)

        rule(primitive)(price)

    for move_name in ("L1_TO_MMB_SEG0", "L1_TO_MMB_SEG1", "MMB_TO_L1"):
        typed_rule(target.move(move_name), f"copy_{move_name.lower()}")
    for op_name in (
        "MUL_TYPED",
        "GROUP_REDUCE_ADD_TYPED",
        "ADD_TYPED",
        "COPY_ODD_TO_EVEN_VECTORS",
        "DIV_TYPED",
    ):
        typed_rule(target.op(op_name), op_name.lower())


apu_g2_cost = build_apu_g2_cost()


def apu_g2_typed_calibration_programs():
    """The ten Round 2 programs of :data:`APUG2_TYPED_TICK_MEASUREMENTS`.

    Returns ``{measurement program key: recipe}``; the recipes are rebuilt from
    the public typed/composed program records, so a recipe change shows up as a
    feature change in the calibration fit.
    """

    from ..apu_g2_composed_contraction import (
        APUG2ComposedContractionProgram,
        APUG2DotEpilogue,
    )
    from ..apu_g2_composed_program import build_apu_g2_composed_recipe
    from ..apu_g2_typed_program import (
        APUG2ScalarType,
        APUG2TypedProgram,
        build_apu_g2_typed_recipe,
    )

    def scalar(bits, signed=True):
        return APUG2ScalarType(bits, signed)

    full = (4, 65_536)
    typed = {
        "typed_add_i16": APUG2TypedProgram(
            "add", (scalar(16), scalar(16)), scalar(16), full
        ),
        "typed_mul_i8_to_i16": APUG2TypedProgram(
            "mul", (scalar(8), scalar(8)), scalar(16), full
        ),
        "typed_div_u16": APUG2TypedProgram(
            "div", (scalar(16, False), scalar(16, False)), scalar(16, False), full
        ),
        "typed_block_sum_u8_k256": APUG2TypedProgram(
            "block_sum", (scalar(8, False),), scalar(16, False), full, 256
        ),
        "typed_dot_i4_k128": APUG2TypedProgram(
            "dot", (scalar(4), scalar(4)), scalar(15), (1985, 128), 128
        ),
    }
    composed = {
        "composed_gemm_i4_k128": APUG2ComposedContractionProgram(
            (scalar(4), scalar(4)), scalar(15), (64, 31), 128
        ),
        "composed_batched_gemm_i4_k128": APUG2ComposedContractionProgram(
            (scalar(4), scalar(4)), scalar(15), (4, 32, 16), 128
        ),
        "composed_pair_affine_i2_k128": APUG2ComposedContractionProgram(
            (scalar(2), scalar(2)),
            scalar(15),
            (1024,),
            128,
            APUG2DotEpilogue((3, 2)),
        ),
        "composed_two_stream_i3_k256": APUG2ComposedContractionProgram(
            (scalar(3), scalar(3)), scalar(14), (256, 2), 256
        ),
        "composed_two_stream_accumulate_i3_k256": APUG2ComposedContractionProgram(
            (scalar(3), scalar(3)),
            scalar(15),
            (256, 2),
            256,
            APUG2DotEpilogue(accumulator_type=scalar(14)),
        ),
    }
    recipes = {key: build_apu_g2_typed_recipe(item) for key, item in typed.items()}
    recipes.update(
        {key: build_apu_g2_composed_recipe(item) for key, item in composed.items()}
    )
    return recipes


def apu_g2_recipe_tick_features(recipe) -> dict[str, int]:
    """Per-feature totals of every issued call of one recipe."""

    from ..apu_g2_recipe import _typed_call_feature_arguments

    totals = dict.fromkeys(APUG2_TYPED_TICK_FEATURES, 0)
    for operation in recipe.operations:
        features = apu_g2_typed_call_features(
            operation.opcode, **_typed_call_feature_arguments(operation)
        )
        for name, value in features.items():
            totals[name] += value
    return totals


def _nonnegative_least_squares(matrix, values):
    """Exact NNLS by enumerating active sets (seven features: 128 sets)."""

    import itertools

    import numpy as np

    columns = matrix.shape[1]
    best = None
    for size in range(columns + 1):
        for active in itertools.combinations(range(columns), size):
            solution = np.zeros(columns)
            if active:
                fitted, *_ = np.linalg.lstsq(
                    matrix[:, list(active)], values, rcond=None
                )
                if np.any(fitted < 0):
                    continue
                solution[list(active)] = fitted
            residual = float(np.sum((matrix @ solution - values) ** 2))
            if best is None or residual < best[0] - 1e-9:
                best = (residual, solution)
    return best[1]


def fit_apu_g2_typed_tick_model(measurements=APUG2_TYPED_TICK_MEASUREMENTS):
    """Re-derive the typed tick coefficients from device measurements.

    Returns ``(coefficients, report)``: the NNLS coefficients per feature and,
    per program, the in-sample and leave-one-out relative errors.
    """

    import numpy as np

    recipes = apu_g2_typed_calibration_programs()
    names = [item.program for item in measurements]
    matrix = np.array(
        [
            [
                apu_g2_recipe_tick_features(recipes[name])[feature]
                for feature in APUG2_TYPED_TICK_FEATURES
            ]
            for name in names
        ],
        dtype=float,
    )
    values = np.array([item.ticks_per_pipeline for item in measurements], float)
    solution = _nonnegative_least_squares(matrix, values)
    report = {}
    for index, name in enumerate(names):
        keep = [row for row in range(len(names)) if row != index]
        held_out = _nonnegative_least_squares(matrix[keep], values[keep])
        report[name] = {
            "measured": float(values[index]),
            "in_sample_relative_error": float(
                matrix[index] @ solution / values[index] - 1
            ),
            "leave_one_out_relative_error": float(
                matrix[index] @ held_out / values[index] - 1
            ),
        }
    return dict(zip(APUG2_TYPED_TICK_FEATURES, map(float, solution))), report


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


__all__ = [
    "APUG2_ADD_U16_TICKS",
    "APUG2_COPY_L1_TO_MMB_SEG0_U16_TICKS",
    "APUG2_COPY_L1_TO_MMB_SEG1_U16_TICKS",
    "APUG2_COPY_MMB_TO_L1_U16_TICKS",
    "APUG2_SEU_BARRIER_TICKS",
    "apu_g2_cost",
    "apu_g2_recipe_tick_features",
    "apu_g2_typed_calibration_programs",
    "build_apu_g2_add_graph",
    "build_apu_g2_cost",
    "estimate_apu_g2_add",
    "fit_apu_g2_typed_tick_model",
]
