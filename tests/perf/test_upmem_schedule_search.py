# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned schedule-search migration and physical UPMEM tile legality tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.ir.types import int32
from allo.pim import upmem_program as upmem_program_module
from allo.pim.costs.upmem import upmem_cost
from allo.pim.schedule_promotion import (
    CorrectnessEvidence,
    ExactCyclePolicy,
    MetricMeasurements,
    MetricPromotionCell,
    ObjectiveMetricBridge,
    PromotionEvidence,
    ScheduleEvidence,
    SchedulePromotionGate,
    SemanticScope,
)
from allo.pim.schedule_search import (
    DecisionDomain,
    ScheduleObjectiveDomain,
    grid_search,
)
from allo.pim.targets import build_upmem_target
from allo.pim.upmem_program import (
    CompiledUPMEMProgram,
    UPMEMDenseTile,
    UPMEMDotTile,
    UPMEMRank1Tile,
    UPMEMScheduleUnavailable,
    compile_upmem_program,
    search_upmem_program_schedule,
)


def _copy(a: int32[16], out: int32[16]):
    for i in range(16):
        out[i] = a[i]


def _wide_copy(a: int32[128], out: int32[128]):
    for i in range(128):
        out[i] = a[i]


def _isomorphic_copy(source: int32[16], destination: int32[16]):
    for lane in range(16):
        destination[lane] = source[lane]


def _rank2_copy(a: int32[4, 4], out: int32[4, 4]):
    for i in range(4):
        for j in range(4):
            out[i, j] = a[i, j]


def _serial_accumulate(a: int32[16], out: int32[16]):
    out[0] = 0
    for i in range(16):
        out[0] += a[i]


def _modulo_collision(a: int32[16], out: int32[2]):
    for i in range(16):
        out[i % 2] = a[i]


def _scaled_index(a: int32[16], out: int32[32]):
    for i in range(16):
        out[2 * i] = a[i]


def _program(num_tasklets=16):
    return allo.UPMEMProgram(
        [allo.UPMEMPhase(_copy)],
        name="pinned_upmem_program",
        arrays=(allo.UPMEMArray("a"), allo.UPMEMArray("out")),
        num_tasklets=num_tasklets,
    )


def _promotion_evidence(result):
    platform = ("platform", "upmem", "hardware-campaign-1")
    domain = ScheduleObjectiveDomain(
        metric="cycles",
        target="upmem",
        target_revision="hardware-campaign-1",
        model_fingerprint=("repeated-runs", 1),
        fidelity="hardware",
        scope="whole_program",
        unit="cycles",
        direction="minimize",
    )
    correctness = CorrectnessEvidence.exact_pass(("int32-oracle", 1))
    scope = SemanticScope(
        ("upmem-program-semantics", 1),
        (("complete-output", True),),
        complete=True,
    )

    def schedule_evidence(candidate):
        fingerprint = candidate.materialized.promotion_materialization_fingerprint
        return ScheduleEvidence.from_schedule(
            candidate,
            correctness=correctness,
            semantic_scope=scope,
            scored_fingerprint=fingerprint,
            emitted_fingerprint=fingerprint,
            platform_fingerprint=platform,
        )

    return PromotionEvidence(
        schedule_evidence(result.best),
        schedule_evidence(result.best_incumbent),
        (
            MetricPromotionCell(
                MetricMeasurements(domain, (90,), None, platform),
                MetricMeasurements(domain, (100,), None, platform),
                ExactCyclePolicy(),
                ObjectiveMetricBridge(
                    result.best.objective_domain,
                    domain,
                    result.best.objective_domain.unit,
                    domain.unit,
                    result.best.objective_domain.direction,
                    domain.direction,
                    "identity_metric",
                    ("upmem-cycle-model-validation", 1),
                ),
            ),
        ),
    )


def test_incomplete_generic_lowering_fails_before_candidate_scoring(monkeypatch):
    program = allo.UPMEMProgram(
        [allo.UPMEMPhase(_rank2_copy)],
        arrays=(allo.UPMEMArray("a"), allo.UPMEMArray("out")),
        num_tasklets=13,
    )
    target = build_upmem_target()
    bound_cost = upmem_cost.bind(target)
    evaluations = []
    monkeypatch.setattr(
        type(bound_cost),
        "evaluate",
        lambda self, graph: evaluations.append(graph),
    )

    with pytest.raises(
        UPMEMScheduleUnavailable,
        match="complete candidate-specific UPMEM device source",
    ) as failure:
        search_upmem_program_schedule(program, target, bound_cost)

    assert evaluations == []
    assert failure.value.incumbent.program is program
    assert failure.value.incumbent.schedule_realizable is False
    with pytest.raises(RuntimeError, match="no complete candidate-specific"):
        failure.value.incumbent.promotion_materialization_fingerprint


@pytest.mark.parametrize(
    "result_expression",
    (
        "d0 mod 2",
        "d0 floordiv 2",
        "2 * d0",
        "d0 + d1",
        "d0 + s0",
        "-d0",
    ),
)
def test_noninjective_or_unproven_affine_coordinates_are_rejected(
    result_expression,
):
    induction = object()
    operation = SimpleNamespace(
        attributes={"map": f"affine_map<(d0, d1)[s0] -> ({result_expression})>"}
    )

    assert not upmem_program_module._direct_affine_induction(
        operation,
        (induction, object()),
        induction,
    )


def test_exact_unit_affine_coordinate_proves_disjoint_iterations():
    induction = object()
    operation = SimpleNamespace(attributes={"map": "affine_map<(d0, d1) -> (d1, d0)>"})

    assert upmem_program_module._direct_affine_induction(
        operation,
        (induction, object()),
        induction,
    )


@pytest.mark.parametrize(
    "kernel,arrays",
    (
        (
            _modulo_collision,
            (allo.UPMEMArray("a"), allo.UPMEMArray("out")),
        ),
        (
            _scaled_index,
            (allo.UPMEMArray("a"), allo.UPMEMArray("out")),
        ),
    ),
)
def test_colliding_or_unproven_store_indices_expose_no_parallel_region(
    kernel,
    arrays,
):
    compiled = CompiledUPMEMProgram(
        allo.UPMEMProgram([allo.UPMEMPhase(kernel)], arrays=arrays),
        build_upmem_target(),
    )

    assert compiled.phases[0].abi.parallel_regions == ()
    assert compiled.schedule_realizable is False


def test_search_rejects_unbound_or_wrong_target_cost():
    program = _program()
    target = build_upmem_target()

    with pytest.raises(TypeError, match="cost bound to the target"):
        search_upmem_program_schedule(program, target, object())

    other_target = build_upmem_target()
    with pytest.raises(TypeError, match="cost bound to the target"):
        search_upmem_program_schedule(
            program,
            target,
            upmem_cost.bind(other_target),
        )


def test_dependence_carrying_phase_keeps_exact_legacy_incumbent():
    program = allo.UPMEMProgram(
        [allo.UPMEMPhase(_serial_accumulate)],
        name="serial_upmem_program",
        arrays=(allo.UPMEMArray("a"), allo.UPMEMArray("out")),
        num_tasklets=7,
    )
    target = build_upmem_target()
    bound_cost = upmem_cost.bind(target)

    legacy = CompiledUPMEMProgram(program, target, cost=bound_cost)
    compiled = compile_upmem_program(program, target, cost=bound_cost)

    assert compiled.schedule_search_result is None
    assert compiled.schedule_activation is None
    assert compiled.fallback_reason.startswith("autoschedule_unavailable:")
    assert compiled.compiled.c_source == legacy.c_source
    assert (
        compiled.compiled.legacy_materialization_fingerprint
        == legacy.legacy_materialization_fingerprint
    )
    assert compiled.execution_graph.metadata["layout_parallelism"] == [
        {"phase": "_serial_accumulate", "dpu": 16, "tasklet": 1}
    ]


def test_structural_region_identity_and_tasklet_domain_ignore_names():
    target = build_upmem_target()
    bound_cost = upmem_cost.bind(target)
    first = _program(num_tasklets=7)
    renamed = allo.UPMEMProgram(
        [allo.UPMEMPhase(_isomorphic_copy, name="arbitrary_phase_label")],
        name="arbitrary_workload_label",
        arrays=(
            allo.UPMEMArray("source"),
            allo.UPMEMArray("destination"),
        ),
        num_tasklets=7,
    )

    first_result = search_upmem_program_schedule(first, target, bound_cost)
    renamed_result = search_upmem_program_schedule(renamed, target, bound_cost)

    first_region = first_result.best_incumbent.materialized.phases[
        0
    ].abi.parallel_regions
    renamed_region = renamed_result.best_incumbent.materialized.phases[
        0
    ].abi.parallel_regions
    assert [region.site_id for region in first_region] == [
        region.site_id for region in renamed_region
    ]
    assert {
        candidate.decisions["num_tasklets"] for candidate in first_result.ranked
    } == set(range(1, 25))
    assert {
        candidate.decisions["num_tasklets"] for candidate in renamed_result.ranked
    } == set(range(1, 25))


def test_costed_compile_keeps_candidate_phase_identity_and_matches_direct_legacy(
    monkeypatch,
):
    program = _program(num_tasklets=7)
    target = build_upmem_target()
    bound_cost = upmem_cost.bind(target)
    legacy = CompiledUPMEMProgram(program, target, cost=bound_cost)
    original_compile_phase = upmem_program_module._compile_phase
    phase_compiles = []

    def track_phase_compile(phase):
        phase_compiles.append(phase)
        return original_compile_phase(phase)

    monkeypatch.setattr(
        upmem_program_module,
        "_compile_phase",
        track_phase_compile,
    )

    migrated = compile_upmem_program(program, target, cost=bound_cost)
    incumbent = migrated.schedule_search_result.best_incumbent

    assert phase_compiles == list(program.phases) * 24
    assert len(migrated.schedule_search_result.ranked) == 24
    materialized_phases = [
        candidate.materialized.phases[0]
        for candidate in migrated.schedule_search_result.ranked
    ]
    assert len({id(phase) for phase in materialized_phases}) == 24
    assert len({id(phase.artifact) for phase in materialized_phases}) == 24
    assert len({id(phase.executable) for phase in materialized_phases}) == 24
    source_manifests = [
        candidate.materialized.device_source_manifests[0]
        for candidate in migrated.schedule_search_result.ranked
    ]
    assert len({manifest.source_fingerprint for manifest in source_manifests}) == 24
    assert all(manifest.sdk_complete for manifest in source_manifests)
    for candidate in migrated.schedule_search_result.ranked:
        tasklets = candidate.decisions["num_tasklets"]
        source = candidate.materialized.device_c_source[0]
        manifest = candidate.materialized.device_source_manifests[0]
        assert f"#define TENON_NUM_TASKLETS {tasklets}" in source
        assert manifest.num_tasklets == tasklets
        assert manifest.compile_flags == (f"-DNR_TASKLETS={tasklets}",)
        assert candidate.materialized.device_abi.num_tasklets == tasklets
        assert manifest.active_tasklets == (
            candidate.materialized.device_abi.launches[0].tasklet_parallelism("out")
        )
        assert f"#define TENON_ACTIVE_TASKLETS {manifest.active_tasklets}" in source
    materialization_fingerprints = {
        candidate.materialized.promotion_materialization_fingerprint
        for candidate in migrated.schedule_search_result.ranked
    }
    assert len(materialization_fingerprints) == 24
    assert all(len(fingerprint) == 64 for fingerprint in materialization_fingerprints)
    assert migrated.schedule_search_result.stats.complete_assignments_considered == 23
    assert incumbent is not None
    assert migrated.schedule_activation.active is incumbent
    assert migrated.schedule_activation.recommended is (
        migrated.schedule_search_result.best
    )
    assert migrated.fallback_reason == (
        None
        if migrated.schedule_search_result.best is incumbent
        else "shadow_only: promotion evidence was not requested"
    )
    assert incumbent.materialized is migrated.compiled
    assert incumbent.payload is program
    assert dict(incumbent.decisions) == {"num_tasklets": 7}
    assert migrated.compiled.c_source == legacy.c_source
    assert (
        migrated.compiled.promotion_materialization_fingerprint
        == legacy.promotion_materialization_fingerprint
    )
    assert migrated.compiled.device_c_abi == legacy.device_c_abi
    assert migrated.abi_manifest == legacy.abi_manifest
    assert (
        migrated.estimate().cycles == bound_cost.evaluate(legacy.execution_graph).cycles
    )
    values = np.arange(16, dtype=np.int32)
    output = np.zeros_like(values)
    migrated(values, output)
    np.testing.assert_array_equal(output, values)


def test_evidence_gate_fails_closed_without_upmem_platform_fingerprint(
    monkeypatch,
):
    program = _program(num_tasklets=7)
    target = build_upmem_target()
    bound_cost = upmem_cost.bind(target)
    programs = {
        tasklets: upmem_program_module._with_upmem_tasklets(program, tasklets)
        for tasklets in (7, 8)
    }
    result = grid_search(
        (DecisionDomain("num_tasklets", (7, 8)),),
        build=lambda decisions: programs[decisions["num_tasklets"]],
        materialize=lambda selected: CompiledUPMEMProgram(
            selected,
            target,
            cost=bound_cost,
        ),
        score=lambda compiled: SimpleNamespace(
            cycles=80 if compiled.program.num_tasklets == 8 else 100
        ),
        objective=lambda estimate: estimate.cycles,
        objective_domain=ScheduleObjectiveDomain.fingerprinted_target(
            metric="cycles",
            target="upmem",
            model_fingerprint=bound_cost.fingerprint,
            fidelity="analytical",
            scope="whole_program",
            unit="cycles",
            direction="minimize",
        ),
        incumbent={"num_tasklets": 7},
    )
    assert result.best.payload.num_tasklets == 8
    monkeypatch.setattr(
        upmem_program_module,
        "search_upmem_program_schedule",
        lambda selected, selected_target, cost, **kwargs: result,
    )

    compiled = compile_upmem_program(
        program,
        target,
        cost=bound_cost,
        promotion_gate=SchedulePromotionGate(_promotion_evidence(result)),
    )

    assert compiled.schedule_activation.promoted is False
    assert compiled.compiled is result.best_incumbent.materialized
    assert compiled.program.num_tasklets == 7
    assert compiled.fallback_reason.startswith(
        "recommended_platform_fingerprint_unavailable"
    )
    assert (
        compiled.schedule_activation.recommended.materialized.promotion_materialization_fingerprint
        == result.best.materialized.promotion_materialization_fingerprint
    )


def test_cost_none_bypasses_search_and_preserves_legacy_callable(monkeypatch):
    monkeypatch.setattr(
        upmem_program_module,
        "search_upmem_program_schedule",
        lambda *_args, **_kwargs: pytest.fail("cost-free compile invoked search"),
    )

    compiled = compile_upmem_program(_program(), build_upmem_target(), cost=None)

    assert compiled.schedule_search_result is None
    assert compiled.cost is None
    with pytest.raises(RuntimeError, match="no executable cost spec"):
        compiled.estimate()


def test_measured_physical_tiles_remain_legal():
    dense = UPMEMDenseTile(16, 16, 1200, 60, 8)
    dot = UPMEMDotTile(2048, 16, 128)
    rank1 = UPMEMRank1Tile(2048, 16)

    assert dense.num_tasklets == 8
    assert dense.wram_bytes < 64 * 1024
    assert dot.num_tasklets == 16
    assert dot.wram_bytes < 64 * 1024
    assert rank1.num_tasklets == 16
    assert rank1.wram_bytes < 64 * 1024


def test_shared_wram_overflows_are_rejected():
    with pytest.raises(ValueError, match="shared 64 KiB WRAM"):
        UPMEMDenseTile(16, 16, 1200, 60, 16)
    with pytest.raises(ValueError, match="shared 64 KiB WRAM"):
        UPMEMDotTile(8192, 16, 512)
    with pytest.raises(ValueError, match="shared 64 KiB WRAM"):
        UPMEMRank1Tile(8192, 16)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UPMEMDenseTile(2, 2, 2, 2, 1),
        lambda: UPMEMDenseTile(1, 512, 2, 2, 1),
        lambda: UPMEMDenseTile(1, 2, 512, 512, 1),
        lambda: UPMEMDotTile(2, 1, 2),
        lambda: UPMEMDotTile(512, 1, 512),
        lambda: UPMEMRank1Tile(2, 1),
        lambda: UPMEMRank1Tile(512, 1),
    ],
)
def test_dma_size_boundaries_are_legal(factory):
    tile = factory()
    assert all(offset % 8 == 0 for offset in tile.mram_offsets)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UPMEMDenseTile(1, 1, 2, 2, 1),
        lambda: UPMEMDenseTile(1, 3, 2, 2, 1),
        lambda: UPMEMDenseTile(1, 514, 2, 2, 1),
        lambda: UPMEMDenseTile(1, 2, 1, 1, 1),
        lambda: UPMEMDenseTile(1, 2, 3, 3, 1),
        lambda: UPMEMDenseTile(1, 2, 514, 514, 1),
        lambda: UPMEMDotTile(1, 1, 1),
        lambda: UPMEMDotTile(3, 1, 3),
        lambda: UPMEMDotTile(514, 1, 514),
        lambda: UPMEMRank1Tile(1, 1),
        lambda: UPMEMRank1Tile(3, 1),
        lambda: UPMEMRank1Tile(514, 1),
    ],
)
def test_invalid_dma_sizes_are_rejected(factory):
    with pytest.raises(ValueError, match="MRAM DMA size"):
        factory()


def test_mram_images_cannot_exceed_64_mib():
    with pytest.raises(ValueError, match="64 MiB MRAM"):
        UPMEMDenseTile(16, 16, 1_048_576, 2, 8)
    with pytest.raises(ValueError, match="64 MiB MRAM"):
        UPMEMDotTile(8_388_608, 16, 128)
