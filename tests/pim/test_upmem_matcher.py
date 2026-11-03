# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM matcher path: decision domains, cost-model equality, ranking flip.

Spec: design_doc/compiler/upmem-matcher-track.md, U5 and U6.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import int32
from allo.perf import CostEvent, ExecutionGraph
from allo.pim.costs import upmem as upmem_costs
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.pim.upmem_physical_search import (
    DEFAULT_UPMEM_CALIBRATION_EVIDENCE,
    DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
    UPMEMPhysicalLegalityError,
    _GEMM_REFERENCE,
    _MMTV_REFERENCE,
    _MV_REFERENCE,
    _POINTWISE_REFERENCE,
    _SELECTION_REFERENCE,
    physical_features,
)
from allo.spmw_upmem import (
    UPMEM_DECISION_DOMAINS,
    UPMEM_KNOBS,
    _realize,
    decision_fields,
    kernel_from_extra,
    upmem_knob_candidates,
)

_KIND_FAMILY = {
    "pointwise": "pointwise",
    "selection_flags": "selection",
    "matrix_vector": "matrix_vector",
    "gemm": "gemm",
}


def _geometry(problem):
    if _KIND_FAMILY[problem.kind.value] == "matrix_vector":
        return (
            ("batches", problem.batches),
            ("rows", problem.rows // problem.batches),
            ("columns", problem.reduction),
        )
    return (("elements", problem.logical_work),)


def _candidate(problem, decision):
    """Placement ``extra`` the UPMEM enumerator produces for this decision."""
    family = _KIND_FAMILY[problem.kind.value]
    return {
        "upmem_family": family,
        "upmem_geometry": _geometry(problem),
        "upmem_anchor_op": "MAC" if family in ("matrix_vector", "gemm") else "ADD",
        "upmem_constants": (),
        **decision_fields(decision),
        "predicate_lowering": decision.predicate_lowering.value,
        "upmem_features": tuple(
            physical_features(problem, decision).manifest().items()
        ),
    }


def _stream_cycles(bound, target, candidate) -> int:
    graph = ExecutionGraph(name="one-stream")
    event = CostEvent.create(
        "kernel",
        target.op(candidate["upmem_anchor_op"]),
        work_id=(0,),
        metrics={"candidate": candidate},
    )
    bound.emit(graph, event, ())
    return int(bound.evaluate(graph).cycles)


def _knob_fanout(base):
    current = [base]
    for name in UPMEM_KNOBS:
        current = [
            {**extra, name: value}
            for extra in current
            for value in upmem_knob_candidates(name, extra)
        ]
    return current


def test_upmem_knob_fanout_is_the_realizable_catalog():
    """The knob fan-out over each data layout reproduces exactly the catalog
    decisions the family template realizes (physical legality plus the plan
    builder), in deterministic_order_key order; every excluded decision has
    a stated reason."""
    for family, reference, anchor in (
        ("pointwise", _POINTWISE_REFERENCE, "ADD"),
        ("matrix_vector", _MV_REFERENCE, "MAC"),
        ("matrix_vector", _MMTV_REFERENCE, "MAC"),
    ):
        geometry = _geometry(reference)
        base = {
            "upmem_family": family,
            "upmem_geometry": geometry,
            "upmem_anchor_op": anchor,
            "upmem_constants": (),
            "upmem_accumulator_init": "zero" if family == "matrix_vector" else "none",
        }
        kernel = kernel_from_extra(base)
        legal = []
        for decision in UPMEM_DECISION_DOMAINS[family]:
            reason = _realize(kernel, decision)[1]
            if reason is None:
                legal.append(decision)
            else:
                assert reason.strip(), decision
        legal.sort(key=lambda decision: decision.deterministic_order_key)
        fanned = []
        for layout in sorted({decision.layout.value for decision in legal}):
            fanned.extend(_knob_fanout({**base, "data_layout": layout}))
        assert [
            {name: extra[name] for name in ("data_layout",) + UPMEM_KNOBS}
            for extra in fanned
        ] == [decision_fields(decision) for decision in legal], family
    # The MV catalog's shared-WRAM entry has no template: priced never.
    mv = kernel_from_extra(
        {
            "upmem_family": "matrix_vector",
            "upmem_geometry": (("batches", 1), ("rows", 96), ("columns", 128)),
            "upmem_anchor_op": "MAC",
            "upmem_accumulator_init": "zero",
        }
    )
    shared = UPMEM_DECISION_DOMAINS["matrix_vector"][0]
    assert "does not materialize" in _realize(mv, shared)[1]


def test_upmem_cost_equals_physical_model():
    target = build_upmem_target()
    bound = upmem_cost.bind(target)
    cases = [(row.problem, row.decision) for row in DEFAULT_UPMEM_CALIBRATION_EVIDENCE]
    for reference, decisions in (
        (_POINTWISE_REFERENCE, UPMEM_DECISION_DOMAINS["pointwise"]),
        (_MV_REFERENCE, UPMEM_DECISION_DOMAINS["matrix_vector"]),
        (_MMTV_REFERENCE, UPMEM_DECISION_DOMAINS["matrix_vector"]),
        (_GEMM_REFERENCE, UPMEM_DECISION_DOMAINS["gemm"]),
        (_SELECTION_REFERENCE, UPMEM_DECISION_DOMAINS["selection"]),
    ):
        cases.extend((reference, decision) for decision in decisions)
    checked = 0
    for problem, decision in cases:
        try:
            expected = DEFAULT_UPMEM_PHYSICAL_COST_MODEL.estimate(problem, decision)
        except UPMEMPhysicalLegalityError:
            continue
        candidate = _candidate(problem, decision)
        assert _stream_cycles(bound, target, candidate) == (
            expected.predicted_logic_cycles
        ), (problem.kind.value, decision.manifest())
        checked += 1
    assert checked >= len(DEFAULT_UPMEM_CALIBRATION_EVIDENCE)


VA_ELEMENTS = 12288


@_df_region()
def va(A: int32[VA_ELEMENTS], B: int32[VA_ELEMENTS], C: int32[VA_ELEMENTS]):
    @allo.work(mapping=[1], args=[A, B, C])
    def add(lA: int32[VA_ELEMENTS], lB: int32[VA_ELEMENTS], lC: int32[VA_ELEMENTS]):
        for i in range(VA_ELEMENTS):
            lC[i] = lA[i] + lB[i]


def _winner(compiled):
    return compiled.compiled.layout.extra


def test_upmem_calibration_flips_ranking(monkeypatch):
    target = build_upmem_target()
    chosen = _winner(allo.compile(va, target, upmem_cost, backend="virtual"))
    assert (chosen["data_layout"], chosen["dma_bytes"]) == ("fused_replicated", 256)

    before = upmem_cost.bind(target).fingerprint
    key = ("pointwise", "fused_replicated", 12, "contiguous", 256, 32)
    factors = dict(upmem_costs.PHYSICAL_CALIBRATION_FACTORS)
    factors[key] = 10 * factors[key]
    monkeypatch.setattr(upmem_costs, "PHYSICAL_CALIBRATION_FACTORS", factors)
    assert upmem_cost.bind(target).fingerprint != before

    flipped = _winner(allo.compile(va, target, upmem_cost, backend="virtual"))
    assert (flipped["data_layout"], flipped["dma_bytes"]) != ("fused_replicated", 256)


# --------------------------------------------------------------------- #
# Host fan-out (spec 003 U7, ruling 013-R1): uPIMulator, paper_full
# --------------------------------------------------------------------- #

FANOUT_DPUS = (1, 2, 4, 8, 16, 24, 32)


@pytest.mark.paper_full
def test_upmem_fanout_gemv():
    import conftest

    from benchmarks.upmem.workloads import gemv_host_program, gemv_workload

    conftest._simulator_gate("upmem")
    target = build_upmem_target()
    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(1152, 128), dtype=np.int32)
    x = rng.integers(-8, 8, size=128, dtype=np.int32)
    cycles = {}
    for count in FANOUT_DPUS:
        region = gemv_workload(1152, 128, count)
        compiled = allo.compile(
            region, target, upmem_cost, host_moves=gemv_host_program(region)
        )
        y = np.zeros(1152, dtype=np.int32)
        result = compiled(W, x, y)
        np.testing.assert_array_equal(y, W @ x)
        assert result.extra["upmem"][0]["num_dpus"] == count
        cycles[count] = result.cycles
    assert cycles[32] < cycles[1]


@pytest.mark.paper_full
def test_upmem_fanout_reduction():
    import conftest

    from benchmarks.upmem.workloads import reduction_host_program, reduction_workload

    conftest._simulator_gate("upmem")
    target = build_upmem_target()
    rng = np.random.default_rng(1)
    A = rng.integers(-(1 << 20), 1 << 20, size=12288, dtype=np.int32)
    for count in FANOUT_DPUS:
        region = reduction_workload(12288, count)
        compiled = allo.compile(
            region, target, upmem_cost, host_moves=reduction_host_program(region)
        )
        partials = np.zeros(count, dtype=np.int64)
        result = compiled(A, partials)
        assert int(partials.sum()) == int(A.astype(np.int64).sum())
        assert result.extra["upmem"][0]["num_dpus"] == count
