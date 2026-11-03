# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost-ranked tile decisions on the APU v1 contraction route (release V1-06).

Compile-only.  The paper's "Shared Selection" GEMV (FP16 4096x256 times a
256-vector, reassociation permitted) offers four Gemini-I reduction tiles;
each is a separate cost-ranked candidate, and the tile decision is recorded
with every candidate's estimate.  The measured 3.845 percent (tile 64 versus
tile 256) is a historical device result and is not asserted here.
"""

from __future__ import annotations

import pytest

import allo
from allo.ir.types import float16
from allo.pim.apu_v1_vectorize import (
    GEMV_TILE_FAMILY,
    apu_v1_contraction_analysis,
    gemv_reduction_tile_domain,
)
from allo.spmw_pipeline import normalize_source
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target

TARGET = build_apu_v1_target()


def paper_gemv(A: float16[4096, 256], B: float16[256, 1], out: float16[4096, 1]):
    for i, j in allo.grid(4096, 1):
        for k in allo.reduction(256):
            out[i, j] += A[i, k] * B[k, j]


def layout_gemm(A: float16[1024, 64], B: float16[64, 1024], C: float16[1024, 1024]):
    for i, j in allo.grid(1024, 1024):
        for k in allo.reduction(64):
            C[i, j] += A[i, k] * B[k, j]


def _compile(workload, **kwargs):
    kwargs.setdefault("reduction_order", "reassociate")
    return allo.compile(workload, TARGET, apu_v1_cost, backend="virtual", **kwargs)


@pytest.fixture(scope="module")
def gemv():
    return _compile(paper_gemv)


def test_paper_gemv_ranks_all_four_tiles(gemv):
    family = [c.name for c in gemv.candidates if c.name.startswith(GEMV_TILE_FAMILY)]
    assert sorted(family) == sorted(
        f"{GEMV_TILE_FAMILY}_k{tile}" for tile in (32, 64, 128, 256)
    )
    decision = gemv.tile_decision
    assert decision["family"] == GEMV_TILE_FAMILY
    assert sorted(decision["legal_tiles"]) == [32, 64, 128, 256]
    by_tile = {entry["tile"]: entry["cycles"] for entry in decision["candidates"]}
    assert set(by_tile) == {32, 64, 128, 256}
    assert all(cycles is not None and cycles > 0 for cycles in by_tile.values())
    # The ranked minimum is selected; nothing fixes the tile beforehand.
    best = min(by_tile, key=by_tile.get)
    assert decision["selected_tile"] == best
    assert gemv.selected_plan.name == f"{GEMV_TILE_FAMILY}_k{best}"
    assert gemv.selected_estimate.cycles == by_tile[best]
    # The cost program prices tile 64 above tile 256 (the device measured the
    # same order; the magnitude is reported in docs/release/design/apu-v1-cost.md).
    assert by_tile[64] > by_tile[256]
    # The decision is recorded on the estimate graph and in run results.
    assert gemv.execution_graph.metadata["tile_decision"]["tile"] == best
    assert gemv.execution_graph.metadata["estimate_validity"] == "extrapolated"


def test_tile_domain_is_derived_from_lanes_and_residency():
    analysis = apu_v1_contraction_analysis(
        normalize_source(paper_gemv, TARGET).program_facts()
    )
    domain = gemv_reduction_tile_domain(analysis)
    for tile, padded, resident, rows_per_vr, matrix_tiles in domain:
        assert padded * rows_per_vr == 32768
        assert resident == -(-256 // padded) <= 10
        assert matrix_tiles == -(-4096 // rows_per_vr) * resident


def test_tile_can_be_forced_or_requested_as_a_family(gemv):
    forced = _compile(paper_gemv, layout=f"{GEMV_TILE_FAMILY}_k64")
    assert forced.selected_plan.name == f"{GEMV_TILE_FAMILY}_k64"
    assert forced.tile_decision["selected_tile"] == 64
    assert forced.tile_decision["plan_selection"] == "explicit_layout"
    family = _compile(paper_gemv, layout=GEMV_TILE_FAMILY)
    assert family.selected_plan.name == gemv.selected_plan.name


def test_layout_gemm_catalog_and_paper_plans():
    compiled = _compile(layout_gemm, reduction_order="strict")
    names = [candidate.name for candidate in compiled.candidates]
    assert "temporal_dma_coalescing" in names
    assert "temporal_dma_coalescing_broadcast_friendly" in names
    assert not any(name.startswith(GEMV_TILE_FAMILY) for name in names)
    assert compiled.tile_decision is None
    ranked = [estimate.plan.name for estimate in compiled.candidate_estimates]
    assert compiled.selected_plan.name == ranked[0]
    baseline = compiled.estimate("temporal_dma_coalescing").cycles
    broadcast = compiled.estimate("temporal_dma_coalescing_broadcast_friendly").cycles
    assert broadcast < baseline
