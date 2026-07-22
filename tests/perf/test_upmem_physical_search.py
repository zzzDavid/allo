# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-simulator tests for physical UPMEM schedule search and calibration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from allo.pim.upmem_physical import UPMEMElementwisePlan
from allo.pim.upmem_physical_search import (
    DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
    UPMEM_MMTV_ROW_LAYOUT_PHASES,
    MaskedF2TaskletLayout,
    UPMEMDataLayout,
    UPMEMCalibrationEvidence,
    UPMEMNoLegalPhysicalPlan,
    UPMEMOperandResidency,
    UPMEMOwnership,
    UPMEMPhysicalDecision,
    UPMEMPhysicalDecisionDomain,
    UPMEMPhysicalCostModel,
    UPMEMPhysicalProblem,
    UPMEMPredicateLowering,
    physical_features,
    rank_upmem_physical_candidates,
    search_upmem_mmtv_row_layout,
    select_upmem_physical_plan,
    upmem_mmtv_row_costs,
)


def pointwise_candidates():
    return (
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 64, 16),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 128, 32),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 256, 64),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 1024, 256),
        UPMEMPhysicalDecision(UPMEMDataLayout.FUSED_REPLICATED, 256, 32),
    )


def mv_candidates():
    return (
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
    )


def mmtv_candidates():
    return (
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            64,
            16,
            vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            128,
            16,
            vector_residency=UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
    )


def gemm_candidates():
    fused = UPMEMOperandResidency.FUSED_DMA_PACKET
    return (
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            320,
            16,
            rhs_residency=fused,
            nc=4,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            576,
            16,
            rhs_residency=fused,
            nc=8,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1088,
            16,
            rhs_residency=fused,
            nc=16,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1056,
            8,
            rhs_residency=fused,
            nc=32,
            kc=8,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            2048,
            64,
            rhs_residency=UPMEMOperandResidency.SHARED_WRAM,
            nc=128,
            kc=64,
        ),
    )


def selection_candidates():
    candidates = []
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
    ):
        candidates.append(
            UPMEMPhysicalDecision(
                UPMEMDataLayout.SEPARATE,
                elements * 4,
                elements,
                predicate_lowering=lowering,
            )
        )
    return tuple(candidates)


def canonical_mmtv_inputs():
    matrix = tuple(((index * 7 + 2) % 13) - 6 for index in range(12 * 16 * 32))
    vectors = tuple(((index * 9 + 4) % 17) - 8 for index in range(12 * 32))
    return matrix, vectors


def test_masked_f2_layout_represents_exactly_twelve_active_tasklets():
    layout = MaskedF2TaskletLayout(12)

    assert layout.padded_tasklet_extent == 16
    assert layout.active_tasklets == tuple(range(12))
    assert layout.masked_tasklets == (12, 13, 14, 15)
    assert layout.active_mask == 0xFFF
    assert layout.manifest()["policy"] == "masked-padded-power-of-two"
    with pytest.raises(FrozenInstanceError):
        layout.physical_tasklets = 8


def test_pointwise_search_selects_compiler_fused_a32_b32_layout():
    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.pointwise(12_288), pointwise_candidates()
    )

    assert result.best.decision.layout is UPMEMDataLayout.FUSED_REPLICATED
    assert result.best.decision.dma_bytes == 256
    assert result.best.decision.chunk_elements == 32
    assert result.best.predicted_logic_cycles == 115_048
    separate = [
        candidate
        for candidate in result.ranked
        if candidate.decision.layout is UPMEMDataLayout.SEPARATE
    ]
    assert separate[0].decision.dma_bytes == 128
    assert separate[0].predicted_logic_cycles == 131_166


def test_matrix_vector_search_prefers_fused_replicated_vector_packets():
    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.matrix_vector(96, 128), mv_candidates()
    )

    assert result.best.decision.vector_residency is (
        UPMEMOperandResidency.FUSED_DMA_PACKET
    )
    assert result.best.predicted_logic_cycles < result.ranked[1].predicted_logic_cycles
    assert result.best.estimate.calibration_evidence_ids == (
        "mtv-fused-replicated-a16-x16",
        "gemv-fused-replicated-a16-x16",
    )


def test_mmtv_search_selects_value_independent_tasklet_private_residency():
    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.matrix_vector(192, 32, batches=12),
        mmtv_candidates(),
    )

    assert result.best.decision.layout is UPMEMDataLayout.SEPARATE
    assert result.best.decision.vector_residency is (
        UPMEMOperandResidency.TASKLET_PRIVATE_WRAM
    )
    assert [candidate.predicted_logic_cycles for candidate in result.ranked] == [
        211_371,
        216_866,
    ]
    assert result.best.estimate.calibration_evidence_ids == (
        "mmtv-tasklet-private-vector-a16",
    )
    features = result.best.estimate.features.manifest()
    assert features["mram_read_bytes"] == 26_112
    assert features["mram_read_calls"] == 408
    assert features["shared_wram_accesses"] == 0
    assert features["wram_bytes"] == 3_072

    evidence = next(
        row
        for row in DEFAULT_UPMEM_PHYSICAL_COST_MODEL.manifest()["evidence"]
        if row["evidence_id"] == "mmtv-tasklet-private-vector-a16"
    )
    assert evidence["raw_artifact_binding"]["oracle_validation_status"] == "pass"
    assert evidence["raw_artifact_binding"]["counters"] == {
        "breakdown_run": 127_139,
        "breakdown_dma": 286,
        "breakdown_etc": 2_381,
        "backpressure": 81_565,
        "mram_read_bytes": 26_112,
        "mram_write_bytes": 768,
        "mram_read_units": 3_264,
        "mram_write_units": 96,
        "mram_activations": 405,
        "mram_precharges": 404,
    }


def test_gemm_search_selects_fused_private_nc16_kc16_packet():
    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.gemm(12, 128, 64), gemm_candidates()
    )

    assert (result.best.decision.nc, result.best.decision.kc) == (16, 16)
    assert result.best.decision.rhs_residency is (
        UPMEMOperandResidency.FUSED_DMA_PACKET
    )
    assert result.best.predicted_logic_cycles == 3_134_862
    assert result.ranked[-1].decision.rhs_residency is (
        UPMEMOperandResidency.SHARED_WRAM
    )
    assert result.ranked[-1].predicted_logic_cycles == 3_974_608


def test_selection_search_jointly_selects_conditional_zero_and_dma16_elements():
    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.selection(12_288), selection_candidates()
    )

    assert result.best.decision.predicate_lowering is (
        UPMEMPredicateLowering.CONDITIONAL_ZERO
    )
    assert result.best.decision.dma_elements == 16
    # The two exact repetitions are deliberately averaged within one physical
    # stratum; either remains below ternary/16 (89,840 cycles).
    assert result.best.predicted_logic_cycles == 89_340
    assert result.best.estimate.calibration_evidence_ids == (
        "selection-conditional-zero-dma16-elements-named",
        "selection-conditional-zero-dma16-elements-compiler-plan",
    )


def test_physical_features_expose_bytes_calls_work_barriers_and_backpressure():
    decision = pointwise_candidates()[-1]
    features = physical_features(
        UPMEMPhysicalProblem.pointwise(12_288), decision
    ).manifest()

    assert features["logical_work"] == 12_288
    assert features["dynamic_instructions"] == 12_288
    assert features["mram_read_bytes"] == 98_304
    assert features["mram_write_bytes"] == 49_152
    assert features["mram_read_calls"] == 384
    assert features["mram_write_calls"] == 384
    assert features["barrier_count"] == 1
    assert features["backpressure_proxy_units"] > 0
    assert features["wram_bytes"] < 64 * 1024


def test_ordered_domain_is_exhaustive_and_search_fingerprints_are_deterministic():
    domain = UPMEMPhysicalDecisionDomain(
        layouts=(UPMEMDataLayout.SEPARATE, UPMEMDataLayout.FUSED_REPLICATED),
        dma_bytes=(128, 256),
        chunk_elements=(32,),
    )
    problem = UPMEMPhysicalProblem.pointwise(12_288)
    first = rank_upmem_physical_candidates(problem, domain=domain)
    second = rank_upmem_physical_candidates(problem, domain=domain)

    assert domain.assignment_count == 4
    assert first.assignment_count == 4
    assert len(first.ranked) == 2
    assert len(first.rejected) == 2
    assert first.search_fingerprint == second.search_fingerprint
    assert [candidate.fingerprint for candidate in first.ranked] == [
        candidate.fingerprint for candidate in second.ranked
    ]
    assert len(first.best.fingerprint) == 64


def test_unsupported_ownership_and_wram_illegal_plan_are_rejected_fail_closed():
    cyclic = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        128,
        32,
        ownership=UPMEMOwnership.CYCLIC,
    )
    cyclic_result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.pointwise(12_288), (cyclic,)
    )
    assert not cyclic_result.ranked
    assert "only proven contiguous" in cyclic_result.rejected[0].reason
    with pytest.raises(UPMEMNoLegalPhysicalPlan):
        select_upmem_physical_plan(UPMEMPhysicalProblem.pointwise(12_288), (cyclic,))

    shared_full_rhs = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        2048,
        256,
        rhs_residency=UPMEMOperandResidency.SHARED_WRAM,
        nc=256,
        kc=256,
    )
    wram_result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.gemm(12, 256, 256), (shared_full_rhs,)
    )
    assert not wram_result.ranked
    assert "WRAM bytes" in wram_result.rejected[0].reason


def test_plan_builder_runs_only_after_legality_and_retains_built_plan():
    legal = pointwise_candidates()[-1]
    illegal = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        128,
        32,
        ownership=UPMEMOwnership.CYCLIC,
    )
    calls = []

    def build(decision):
        calls.append(decision.fingerprint)
        return UPMEMElementwisePlan(
            12_288,
            dma_bytes=decision.chunk_elements * 4,
            interleaved=decision.layout is UPMEMDataLayout.FUSED_REPLICATED,
        )

    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.pointwise(12_288),
        (legal, illegal),
        plan_builder=build,
    )

    assert calls == [legal.fingerprint]
    assert isinstance(result.best.plan, UPMEMElementwisePlan)
    assert result.best.plan.dma_bytes == 128
    assert result.best.plan.interleaved is True
    assert result.rejected[0].stage == "legality"


def test_model_and_search_manifests_bound_objective_scope_and_evidence():
    model_manifest = DEFAULT_UPMEM_PHYSICAL_COST_MODEL.manifest()
    assert model_manifest["claim_scope"] == {
        "software_simulator_calibrated": True,
        "hardware_calibrated": False,
        "hardware_performance_claim": False,
    }
    assert model_manifest["objective"]["scope"] == "device launch only"
    assert "host transfers" in model_manifest["objective"]["excluded"]
    assert len(model_manifest["model_fingerprint"]) == 64
    assert all(not row["hardware_measurement"] for row in model_manifest["evidence"])
    assert model_manifest["evidence_roles"] == {
        "cross_compiler_conservative_reference": 1,
        "tenon_compiler_measurement": 26,
    }
    assert model_manifest["evidence_binding_contract"]["partial_rows_allowed"] is False

    result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.pointwise(12_288), pointwise_candidates()
    )
    manifest = result.manifest()
    assert manifest["objective"]["scope"] == "device launch only"
    assert manifest["tie_break"].startswith("decision structural order")
    assert json.dumps(manifest, sort_keys=True, allow_nan=False)


def test_default_calibration_rows_are_fully_bound_to_passing_raw_runs():
    rows = DEFAULT_UPMEM_PHYSICAL_COST_MODEL.evidence
    assert len(rows) == 27
    expected_counters = {
        "breakdown_run",
        "breakdown_dma",
        "breakdown_etc",
        "backpressure",
        "mram_read_units",
        "mram_write_units",
        "mram_read_bytes",
        "mram_write_bytes",
        "mram_activations",
        "mram_precharges",
    }
    for row in rows:
        binding = row.manifest()["raw_artifact_binding"]
        anchor_root = f"tenon/calibration/anchors/{row.evidence_id}"
        assert binding["program_directory"] == anchor_root + "/program"
        assert binding["run_directory"] == anchor_root + "/run"
        assert len(binding["program_sha256"]) == 64
        assert len(binding["result_sha256"]) == 64
        assert len(binding["simulator_log_sha256"]) == 64
        assert binding["oracle_validation_status"] == "pass"
        assert set(binding["counters"]) == expected_counters
        assert (
            sum(
                binding["counters"][name]
                for name in (
                    "breakdown_run",
                    "breakdown_dma",
                    "breakdown_etc",
                    "backpressure",
                )
            )
            == row.logic_cycles
        )

    cross_compiler = next(
        row for row in rows if row.evidence_id == "pointwise-separate-dma128"
    )
    assert cross_compiler.evidence_role == "cross_compiler_conservative_reference"
    assert "ATiM" in cross_compiler.source
    assert all(
        row.evidence_role == "tenon_compiler_measurement"
        for row in rows
        if row is not cross_compiler
    )


def test_calibration_evidence_and_model_reject_absent_or_corrupt_raw_binding():
    row = DEFAULT_UPMEM_PHYSICAL_COST_MODEL.evidence[0]
    with pytest.raises(TypeError):
        UPMEMCalibrationEvidence(
            "unbound",
            "test",
            row.problem,
            row.decision,
            row.logic_cycles,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(row, result_sha256="")
    with pytest.raises(ValueError, match="exact oracle pass"):
        replace(row, oracle_validation_status="pending")
    with pytest.raises(ValueError, match="logic-cycle breakdown"):
        replace(
            row,
            raw_counters=tuple(
                item for item in row.raw_counters if item[0] != "breakdown_dma"
            ),
        )
    with pytest.raises(ValueError, match="lacks MRAM counters"):
        replace(
            row,
            raw_counters=tuple(
                item for item in row.raw_counters if item[0] != "mram_read_bytes"
            ),
        )

    tampered = replace(row)
    object.__setattr__(tampered, "oracle_validation_status", "pending")
    with pytest.raises(ValueError, match="exact oracle pass"):
        UPMEMPhysicalCostModel((tampered,))

    assert UPMEMPhysicalCostModel().evidence == ()


def test_unrepresented_but_legal_stratum_is_explicitly_analytical_not_hardware():
    decision = UPMEMPhysicalDecision(UPMEMDataLayout.FUSED_REPLICATED, 128, 16)
    estimate = DEFAULT_UPMEM_PHYSICAL_COST_MODEL.estimate(
        UPMEMPhysicalProblem.pointwise(12_288), decision
    )

    assert estimate.calibration_status == "uncalibrated_analytical_extrapolation"
    assert estimate.calibration_factor == 1.0
    assert estimate.calibration_evidence_ids == ()
    assert estimate.predicted_logic_cycles == estimate.predicted.logic_cycles


def test_mmtv_content_aware_row_layout_search_selects_measured_phase_one():
    matrix, vectors = canonical_mmtv_inputs()
    result = search_upmem_mmtv_row_layout(matrix, vectors)

    expected_measurements = {
        "identity": 216_866,
        "greedy-phase-0": 216_181,
        "greedy-phase-1": 215_684,
        "greedy-phase-15": 215_980,
        "greedy-phase-2": 215_805,
        "greedy-phase-3": 215_825,
        "greedy-phase-5": 216_342,
        "greedy-phase-7": 216_127,
        "greedy-phase-9": 216_120,
    }
    assert [candidate.candidate_id for candidate in result.ordered] == list(
        expected_measurements
    )
    assert {
        candidate.candidate_id: candidate.logic_cycles for candidate in result.ordered
    } == expected_measurements
    assert result.best.candidate_id == "greedy-phase-1"
    assert result.best.phase_rotation == 1
    assert result.best.logic_cycles == 215_684
    assert sorted(result.best.physical_to_logical_rows) == list(range(192))
    assert result.best.physical_to_logical_rows != tuple(range(192))
    assert result.best.manifest()["physical_to_logical_rows_i32_sha256"] == (
        "14df9fb1c954296c4937691472c66cce0617bc5788459c33d77e79bf2769206c"
    )
    assert result.best.tasklet_cost_spread == 20


def test_mmtv_row_cost_formula_manifest_and_search_are_deterministic():
    matrix, vectors = canonical_mmtv_inputs()
    row_costs = upmem_mmtv_row_costs(16, 32, 12, matrix, vectors)
    manual_row_zero = sum(
        min(matrix[column] & 0xFFFFFFFF, vectors[column] & 0xFFFFFFFF).bit_length()
        for column in range(32)
    )
    assert len(row_costs) == 192
    assert row_costs[0] == manual_row_zero

    first = search_upmem_mmtv_row_layout(matrix, vectors)
    second = search_upmem_mmtv_row_layout(matrix, vectors)
    assert first.search_fingerprint == second.search_fingerprint
    assert first.manifest() == second.manifest()
    manifest = first.manifest()
    assert manifest["ordered_phases"] == [None, *UPMEM_MMTV_ROW_LAYOUT_PHASES]
    assert manifest["cost_formula"]["logical_row_costs"] == list(row_costs)
    assert manifest["selected"]["physical_to_logical_rows"] == list(
        first.best.physical_to_logical_rows
    )
    assert manifest["claim_scope"] == {
        "software_simulator_calibrated": True,
        "hardware_calibrated": False,
        "hardware_performance_claim": False,
        "content_specific": True,
        "host_unpack_excluded": True,
    }
    assert "host output inverse permutation" in manifest["objective"]["excluded"]
    assert all(not row["hardware_measurement"] for row in manifest["evidence"])


def test_mmtv_row_layout_evidence_rejects_noncanonical_content_and_phase_domain():
    matrix, vectors = canonical_mmtv_inputs()
    changed = list(matrix)
    changed[0] += 1

    with pytest.raises(ValueError, match="bound to the canonical inputs"):
        search_upmem_mmtv_row_layout(changed, vectors)
    with pytest.raises(ValueError, match="exact ordered phase domain"):
        search_upmem_mmtv_row_layout(matrix, vectors, phases=(0, 1))
