# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provenance and accuracy of the fitted Gemini-I cost model (release V1-05).

Compile-only: nothing launches on the board.  The measured medians below are
the archived physical results the paper's "Agent-Assisted Cost Calibration"
reports; the released ``apu_v1_cost`` is re-evaluated on the same plans
(record frequency through the public compile path; histogram residency plans
through the native graph builder with the archived inventories, since the
released histogram recipe materializes only the one-frontier-load plan).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.perf import cost, rule
import allo.pim.costs.apu_v1 as apu_v1_costs
from allo.pim.apu_v1_native_cost import APUv1NativeCostGraphBuilder
from allo.pim.costs import apu_v1_cost
from allo.pim.costs.apu_v1 import (
    APU_V1_AFFINE_EVENT_MODEL,
    DENSE_HISTOGRAM_ROUTE,
    RECORD_FREQUENCY_ROUTE,
    apu_v1_cost_validity,
)
from allo.pim.targets import build_apu_v1_target


TARGET = build_apu_v1_target()
_REPO = Path(__file__).resolve().parents[4]
_ARCHIVE = _REPO / "tenon-artifacts"

# (case, chunks, queries, measured median CRUN, frozen calibrated prediction)
# rebuttal/reviewer-a/q1/evidence/results/2026-09-29-gemini-calibration-v4.
RECORD_FREQUENCY_PLANS = (
    ("c1-q128-vr0", 1, 128, 105132, 107459),
    ("c1-q128-vr8", 1, 128, 104476, 107459),
    ("c3-q64-vr0", 3, 64, 128408, 127890),
    ("c3-q64-vr8", 3, 64, 126997, 127890),
    ("c5-q32-vr0", 5, 32, 156721, 156189),
    ("c5-q32-vr8", 5, 32, 155280, 156189),
    ("c8-q16-vr0", 8, 16, 210131, 213180),
    ("c8-q16-vr4", 8, 16, 210459, 213180),
)
# agent-cost-calibration/2026-07-27-gemini-i/confirmatory-affine-event-v1.
CONFIRMATORY_SHAPES = (
    ("c2-q160", 2, 160, 156182, 154209),
    ("c6-q64", 6, 64, 209012, 203678),
    ("c9-q24", 9, 24, 242858, 242977),
)
# (frontier reload block, measured median, frozen calibrated prediction):
# rebuttal/.../2026-09-29-gemini-histogram-verification-v1/verified.json.
HISTOGRAM_PLANS = (
    (2, 2945843, 2939566),
    (4, 1547981, 1542213),
    (8, 845002, 843537),
    (16, 491238, 494199),
    (32, 322504, 319529),
    (64, 232363, 232195),
    (128, 187137, 188528),
    (256, 166366, 166694),
)


def _mape(pairs):
    pairs = list(pairs)
    return 100.0 * sum(abs(p - m) / m for p, m in pairs) / len(pairs)


@pytest.fixture(scope="module")
def make_kernel(tmp_path_factory):
    root = tmp_path_factory.mktemp("apu_v1_cost_kernels")

    def make(source, name):
        text = "import allo\nfrom allo.ir.types import uint16\n\n" + textwrap.dedent(
            source
        )
        path = root / f"k_{name}_{hashlib.sha256(text.encode()).hexdigest()[:12]}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return getattr(module, name)

    return make


def _record_frequency(make_kernel, chunks, queries):
    body = [
        "    " * (3 + c) + f"if records[{c}, record] == queries[{c}, query]:"
        for c in range(chunks)
    ]
    body.append("    " * (3 + chunks) + "counts[query] += 1")
    source = (
        f"""
def rf(records: uint16[{chunks}, 32768], queries: uint16[{chunks}, {queries}]) -> uint16[{queries}]:
    counts: uint16[{queries}] = 0
    for query in range({queries}):
        for record in allo.reduction(32768):
"""
        + "\n".join(body)
        + "\n    return counts\n"
    )
    return make_kernel(source, "rf")


def _histogram(make_kernel, bins=256):
    return make_kernel(
        f"""
def hist(values: uint16[32768]) -> uint16[{bins}]:
    counts: uint16[{bins}] = 0
    for b in range({bins}):
        for i in allo.reduction(32768):
            if values[i] == b:
                counts[b] += 1
    return counts
""",
        "hist",
    )


def _compile(kernel, cost_spec=apu_v1_cost):
    phase = allo.APUv1Phase(kernel, result_names=("counts",), vectorize="required")
    return allo.compile(
        allo.APUv1Program((phase,)), TARGET, cost_spec, backend="virtual"
    )


def _histogram_plan_graph(block, cost_spec=apu_v1_cost):
    loads = 256 // block
    inventory = {
        "dma_l4_to_l1_32k": loads,
        "gvml_load_16": loads,
        "gvml_eq_imm_16": 256,
        "gvml_2_fast_count_m_g32k": 128,
        "gvml_count_m_g32k": 0,
        "scalar_l4_store_u16": 256,
    }
    bound = cost_spec.bind(TARGET)
    graph = APUv1NativeCostGraphBuilder(DENSE_HISTOGRAM_ROUTE, inventory, TARGET, bound)
    graph.move("DMA_L4_TO_L1_32K", "dma_l4_to_l1_32k", count=loads)
    graph.move("LOAD_L1_TO_VR16", "gvml_load_16", count=loads)
    graph.op("EQ_IMM_16", "gvml_eq_imm_16", count=256)
    graph.op("FAST_COUNT_2M_G32K", "gvml_2_fast_count_m_g32k", count=128)
    graph.move("ARC_STORE_L4_U16", "scalar_l4_store_u16", count=256)
    finished = graph.finish()
    return bound.evaluate(finished).cycles, finished


@pytest.fixture
def uncalibrated(monkeypatch):
    """The released program with the fitted model disabled: every event is
    priced by the per-call defaults (the pre-port cost program)."""

    monkeypatch.setattr(
        apu_v1_costs,
        "APU_V1_AFFINE_EVENT_MODEL",
        dataclasses.replace(APU_V1_AFFINE_EVENT_MODEL, priced_routes=frozenset()),
    )


# --------------------------------------------------------------------------- #
# Coefficients and provenance
# --------------------------------------------------------------------------- #


def test_count_rule_is_the_papers_fitted_rule():
    rule_ = APU_V1_AFFINE_EVENT_MODEL.event("FAST_COUNT_2M_G32K")
    assert round(rule_.activation_cycles, 2) == 5887.79
    assert round(rule_.slope_cycles_per_call, 2) == 876.21
    # With the empty-route intercept the pair reproduces its two
    # microbenchmark medians exactly.
    intercept = APU_V1_AFFINE_EVENT_MODEL.route_intercept_cycles
    for count, median in zip(
        (rule_.low_count, rule_.high_count), rule_.measured_medians_crun
    ):
        assert rule_.cycles(count) + intercept == median
    assert rule_.cycles(0) == 0


def test_every_coefficient_carries_provenance_units_and_measurements():
    model = APU_V1_AFFINE_EVENT_MODEL
    assert {event.primitive for event in model.events} == {
        "DMA_L4_TO_L1_32K",
        "LOAD_L1_TO_VR16",
        "EQ_IMM_16",
        "AND_M",
        "FAST_COUNT_2M_G32K",
        "ARC_STORE_L4_U16",
    }
    intercept = model.route_intercept_cycles
    for event in model.events:
        assert event.low_case.startswith("q") and event.high_case.startswith("q")
        low, high = event.measured_medians_crun
        assert event.cycles(event.low_count) + intercept == low
        assert event.cycles(event.high_count) + intercept == high
        assert event.manifest()["unit"] == "ARC CRUN device cycles"
    for key in (
        "unit",
        "model_fingerprint",
        "model_file",
        "microbenchmarks",
        "measured_utc",
        "board",
        "software",
        "fitted_on",
        "validated_on",
    ):
        assert model.provenance[key]
    assert set(model.validated_envelopes) == {
        RECORD_FREQUENCY_ROUTE,
        DENSE_HISTOGRAM_ROUTE,
    }


def test_released_coefficients_match_the_archived_frozen_model():
    path = (
        _ARCHIVE
        / "agent-cost-calibration/2026-07-27-gemini-i/confirmatory-affine-event-v1"
        / "freeze/affine-event-model.json"
    )
    if not path.is_file():
        pytest.skip("tenon-artifacts calibration archive not present")
    frozen = json.loads(path.read_text())
    model = APU_V1_AFFINE_EVENT_MODEL
    assert frozen["model_fingerprint"] == model.provenance["model_fingerprint"]
    assert frozen["route"]["route_intercept_cycles"] == model.route_intercept_cycles
    assert frozen["route"]["l4_l3_activation_cycles"] == model.l4_l3_activation_cycles
    assert frozen["route"]["l4_l3_slope_cycles_per_byte"] == model.l4_l3_cycles_per_byte
    for event in model.events:
        family = frozen["families"][event.feature]
        assert family["activation_cycles"] == event.activation_cycles
        assert family["slope_cycles_per_call"] == event.slope_cycles_per_call
        assert (family["low_case"], family["high_case"]) == (
            event.low_case,
            event.high_case,
        )


def test_cost_fingerprint_covers_the_fitted_model(monkeypatch):
    before = apu_v1_cost.bind(TARGET).fingerprint
    monkeypatch.setattr(
        apu_v1_costs,
        "APU_V1_AFFINE_EVENT_MODEL",
        dataclasses.replace(APU_V1_AFFINE_EVENT_MODEL, route_intercept_cycles=1.0),
    )
    assert apu_v1_cost.bind(TARGET).fingerprint != before


# --------------------------------------------------------------------------- #
# Accuracy on the archived plans (before / after)
# --------------------------------------------------------------------------- #


def test_record_frequency_plans_reproduce_the_frozen_predictions(make_kernel):
    estimates = {}
    for case, chunks, queries, _measured, frozen in RECORD_FREQUENCY_PLANS:
        key = (chunks, queries)
        if key not in estimates:
            compiled = _compile(_record_frequency(make_kernel, chunks, queries))
            graph = compiled.execution_graph
            assert graph.metadata["estimate_validity"] == "calibrated"
            estimates[key] = compiled.estimate().cycles
        assert estimates[key] == frozen, case
    after = _mape(
        (estimates[(c, q)], measured)
        for _case, c, q, measured, _frozen in RECORD_FREQUENCY_PLANS
    )
    assert after == pytest.approx(1.2304894823069576, abs=1e-9)


def test_record_frequency_before_calibration_is_the_papers_baseline(
    make_kernel, uncalibrated
):
    pairs = []
    for _case, chunks, queries, measured, _frozen in RECORD_FREQUENCY_PLANS:
        cycles = _compile(_record_frequency(make_kernel, chunks, queries)).estimate()
        pairs.append((cycles.cycles, measured))
    assert _mape(pairs) == pytest.approx(7.406257665760589, abs=1e-9)


def test_confirmatory_shapes_before_and_after(make_kernel, monkeypatch):
    after = []
    for _case, chunks, queries, measured, frozen in CONFIRMATORY_SHAPES:
        cycles = _compile(_record_frequency(make_kernel, chunks, queries)).estimate()
        assert cycles.cycles == frozen
        after.append((cycles.cycles, measured))
    assert _mape(after) == pytest.approx(1.2880920618065275, abs=1e-9)
    monkeypatch.setattr(
        apu_v1_costs,
        "APU_V1_AFFINE_EVENT_MODEL",
        dataclasses.replace(APU_V1_AFFINE_EVENT_MODEL, priced_routes=frozenset()),
    )
    before = [
        (
            _compile(_record_frequency(make_kernel, chunks, queries)).estimate().cycles,
            measured,
        )
        for _case, chunks, queries, measured, _frozen in CONFIRMATORY_SHAPES
    ]
    assert _mape(before) == pytest.approx(10.66504465127327, abs=1e-9)


def test_histogram_plans_before_and_after(make_kernel, monkeypatch):
    after = []
    for block, measured, frozen in HISTOGRAM_PLANS:
        cycles, graph = _histogram_plan_graph(block)
        assert cycles == frozen, block
        assert graph.metadata["estimate_validity"] == "calibrated"
        after.append((cycles, measured))
    assert _mape(after) == pytest.approx(0.41213259043634626, abs=1e-9)
    # The released recipe is the one-load plan; its public estimate agrees.
    released = _compile(_histogram(make_kernel)).estimate().cycles
    assert released == HISTOGRAM_PLANS[-1][2]

    monkeypatch.setattr(
        apu_v1_costs,
        "APU_V1_AFFINE_EVENT_MODEL",
        dataclasses.replace(APU_V1_AFFINE_EVENT_MODEL, priced_routes=frozenset()),
    )
    before = [(_histogram_plan_graph(block)[0], m) for block, m, _f in HISTOGRAM_PLANS]
    assert _mape(before) == pytest.approx(25.91816139467001, abs=1e-9)


# --------------------------------------------------------------------------- #
# Validity labels
# --------------------------------------------------------------------------- #


def test_validity_is_calibrated_only_inside_the_validated_envelope(make_kernel):
    paper = _compile(_record_frequency(make_kernel, 3, 128))
    validity = paper.execution_graph.metadata["cost_validity"]
    assert validity["cost_validity"] == "calibrated"
    assert validity["shape"] == {"chunks": 3, "queries": 128}
    result = paper(*(np.zeros(arg.shape, arg.numpy_dtype) for arg in paper.arguments))
    assert result.extra["cost_validity"]["cost_validity"] == "calibrated"

    odd = _compile(_record_frequency(make_kernel, 3, 127))
    odd_validity = odd.execution_graph.metadata["cost_validity"]
    assert odd_validity["cost_validity"] == "extrapolated"
    assert "gvml_count_m_g32k" in odd_validity["basis"]

    wide = _compile(_record_frequency(make_kernel, 10, 16))
    assert wide.execution_graph.metadata["estimate_validity"] == "extrapolated"
    many = _compile(_record_frequency(make_kernel, 1, 256))
    assert "queries=256" in many.execution_graph.metadata["cost_validity"]["basis"]

    bins = _compile(_histogram(make_kernel, bins=128))
    assert bins.execution_graph.metadata["estimate_validity"] == "extrapolated"


def test_other_routes_and_foreign_cost_programs_are_extrapolated():
    moments = apu_v1_cost_validity(
        "gvml_bivariate_moments_fused_group_reduce", {}, apu_v1_cost
    )
    assert moments["cost_validity"] == "extrapolated"
    assert "outside the fitted families" in moments["basis"]

    @cost(target="apu_v1", name="uncalibrated_test_cost")
    def foreign(target):
        @rule(target.op("EQ_IMM_16"))
        def eq(event, ctx):
            ctx.step(latency=1, occupy=[ctx.use(event.primitive)], name="eq")

    validity = apu_v1_cost_validity(
        RECORD_FREQUENCY_ROUTE,
        {"dma_l4_to_l1_32k": 3, "scalar_l4_store_u16": 128},
        foreign.bind(TARGET),
    )
    assert validity["cost_validity"] == "extrapolated"
    assert "no APU v1 calibration record" in validity["basis"]
