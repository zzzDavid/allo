# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software-only tests for calibrated compiler-owned UPMEM physical plans."""

import json

import pytest

from allo.pim.upmem_physical import (
    MaskedTaskletLayout,
    UPMEMElementwisePlan,
    UPMEMGEMMPlan,
    UPMEMMatrixVectorPlan,
    UPMEMSumReductionPlan,
    balance_matrix_vector_rows,
    masked_12_tasklet_linear_layout,
    matrix_vector_row_mul_step_costs,
)


def test_primary_upmem_compiler_apis_are_exported_from_allo_pim():
    import allo.pim as pim

    expected = {
        "UPMEMElementwisePlan",
        "UPMEMSumReductionPlan",
        "UPMEMMatrixVectorPlan",
        "UPMEMGEMMPlan",
        "UPMEMHistogramPlan",
        "UPMEMStableSelectionPlan",
        "UPMEMSelectionFlagsPlan",
        "UPMEMKMeansDistancesPlan",
        "UPMEMKMeansPlan",
        "UPMEMFeatureGradientPlan",
        "UPMEMPhysicalProblem",
        "UPMEMPhysicalDecision",
        "balance_matrix_vector_rows",
        "matrix_vector_row_mul_step_costs",
        "rank_upmem_physical_candidates",
        "select_upmem_physical_plan",
        "UPMEMCalibrationReport",
        "build_upmem_calibration_report",
    }

    assert expected <= set(pim.__all__)
    assert pim.UPMEMElementwisePlan is UPMEMElementwisePlan
    assert pim.UPMEMKMeansDistancesPlan(120, 8, 4).packet_bytes == 64


def test_masked_tasklet_layout_is_padded_f2_and_contiguous():
    masked = MaskedTaskletLayout(32)

    assert masked.active_tasklets == 12
    assert masked.padded_tasklets == 16
    assert masked.layout.output_size("element") == 16 * 32
    assert masked.layout.size_of("tasklet") == 16
    assert masked.layout.size_of("lane") == 32
    assert masked.coordinate(0, 31) == 31
    assert masked.coordinate(1, 0) == 32
    assert masked.coordinate(11, 31) == 12 * 32 - 1
    assert masked_12_tasklet_linear_layout(32).manifest() == masked.layout.manifest()
    with pytest.raises(IndexError, match="masked 12-tasklet"):
        masked.coordinate(12, 0)
    with pytest.raises(ValueError, match="power of two"):
        MaskedTaskletLayout(3)


def test_elementwise_default_is_measured_fused_a32_b32_packet():
    plan = UPMEMElementwisePlan(12 * 1024)
    manifest = plan.manifest()

    assert plan.compile_flags == ("-DNR_TASKLETS=12",)
    assert plan.num_tasklets == plan.active_tasklets == 12
    assert plan.chunk_elements == 32
    assert plan.chunks == 384
    assert plan.chunks_per_tasklet == 32
    assert manifest["operand_dma_bytes"] == 128
    assert manifest["fused_dma_bytes"] == 256
    assert manifest["interleaved"] is True
    assert manifest["mram_regions"]["packed_operands"]["packet_order"] == [
        "x",
        "y",
    ]
    assert manifest["mram_regions"]["x"]["offset"] == 0
    assert manifest["mram_regions"]["y"]["offset"] == 128
    assert manifest["mram_regions"]["output"]["offset"] == 98304
    assert manifest["output_regions"]["output"]["bytes"] == 49152
    assert plan.cost_features() == {
        "mram_read_calls": 384,
        "mram_write_calls": 384,
        "mram_calls": 768,
        "mram_read_bytes": 98304,
        "mram_write_bytes": 49152,
        "mram_bytes": 147456,
        "barriers": 0,
        "wram_bytes": 3072,
        "int32_adds": 12288,
        "int32_subtracts": 0,
        "int32_shifts": 0,
        "int32_multiplies": 0,
    }


def test_elementwise_axpby_coefficients_are_runtime_mram_inputs():
    plan = UPMEMElementwisePlan(12 * 1024, operation="axpby", runtime_coefficients=True)
    regions = plan.mram_regions
    source = plan.device_source()

    assert regions["coefficients"]["offset"] == 98304
    assert regions["coefficients"]["bytes"] == 8
    assert regions["output"]["offset"] == 98312
    assert "mram_read" in source
    assert "tenon_coefficients[0] * packet[i]" in source
    assert "tenon_coefficients[1] * packet[i + TENON_CHUNK_ELEMENTS]" in source
    assert "barrier_wait(&tenon_barrier);" in source
    assert plan.cost_features()["mram_read_calls"] == 385
    assert plan.cost_features()["int32_adds"] == 12_288
    assert plan.cost_features()["int32_subtracts"] == 0
    assert plan.cost_features()["int32_shifts"] == 0
    assert plan.cost_features()["int32_multiplies"] == 2 * 12_288


def test_elementwise_axpby_defaults_to_calibrated_compile_time_coefficients():
    plan = UPMEMElementwisePlan(12 * 1024, operation="axpby")
    source = plan.device_source()

    assert "coefficients" not in plan.mram_regions
    assert plan.mram_regions["output"]["offset"] == 98_304
    assert plan.manifest()["coefficients"] == {
        "alpha": 2,
        "beta": -1,
        "binding": "compile_time",
    }
    assert "(packet[i] << 1) - packet[i + TENON_CHUNK_ELEMENTS]" in source
    assert "barrier_wait" not in source
    assert plan.cost_features()["int32_adds"] == 0
    assert plan.cost_features()["int32_subtracts"] == 12_288
    assert plan.cost_features()["int32_shifts"] == 12_288
    assert plan.cost_features()["int32_multiplies"] == 0


def test_elementwise_separate_layout_remains_a_legal_search_candidate():
    plan = UPMEMElementwisePlan(1000, interleaved=False)

    assert plan.padded_elements == 1536
    assert "packed_operands" not in plan.mram_regions
    assert plan.mram_regions["x"]["offset"] == 0
    assert plan.mram_regions["y"]["offset"] == 6144
    assert plan.cost_features()["mram_read_calls"] == 64
    assert plan.cost_features()["mram_write_calls"] == 32


def test_reduction_matches_measured_register_partial_serial_merge_schedule():
    plan = UPMEMSumReductionPlan(12 * 1024)
    source = plan.device_source()

    assert plan.chunk_elements == 16
    assert plan.chunks == 768
    assert plan.chunks_per_tasklet == 64
    assert plan.mram_regions["input"]["dtype"] == "int32"
    assert plan.output_region["dtype"] == "int64"
    assert plan.output_region["offset"] == 49152
    assert plan.cost_features()["mram_read_calls"] == 768
    assert plan.cost_features()["mram_write_calls"] == 1
    assert plan.cost_features()["barriers"] == 1
    assert "int64_t accumulator = 0;" in source
    assert source.count("tenon_partials[tid] = accumulator;") == 1
    assert source.index("tenon_partials[tid] = accumulator;") < source.index(
        "barrier_wait(&tenon_barrier);"
    )
    assert "for (uint32_t tasklet = 0; tasklet < TENON_ACTIVE_TASKLETS;" in source


def test_mv_default_is_fused_replicated_a16_x16():
    plan = UPMEMMatrixVectorPlan(rows=96, columns=64)
    manifest = plan.manifest()

    assert plan.chunk_elements == 16
    assert plan.packet_bytes == 128
    assert plan.column_chunks == 4
    assert plan.rows_per_tasklet == 8
    assert manifest["packet_order"] == ["matrix_chunk", "vector_chunk"]
    assert manifest["mram_regions"]["packed_matrix_vector"]["bytes"] == 49152
    assert manifest["mram_regions"]["output"]["offset"] == 49152
    assert manifest["mram_regions"]["output"]["tasklet_group_bytes"] == 32
    assert plan.cost_features()["mram_read_calls"] == 384
    assert plan.cost_features()["mram_write_calls"] == 12


def test_scaled_mv_specializes_known_factor_and_retains_fused_packets():
    plan = UPMEMMatrixVectorPlan(rows=96, columns=64, scale=True)
    source = plan.device_source()

    assert "scale" not in plan.mram_regions
    assert plan.mram_regions["output"]["offset"] == 49152
    assert "accumulator = 2 * accumulator;" in source
    assert plan.effective_scale_factor == 2
    assert plan.cost_features()["barriers"] == 0
    assert plan.cost_features()["mram_read_bytes"] == 384 * 128


def test_batched_mv_flattens_batch_rows_before_tasklet_ownership():
    plan = UPMEMMatrixVectorPlan(rows=16, columns=64, batches=12)

    assert plan.total_rows == 192
    assert plan.rows_per_tasklet == 16
    assert plan.padded_rows == 192
    assert plan.output_dma_bytes == 64
    assert plan.output_region["physical_shape"] == [192]
    assert plan.cost_features()["mram_write_calls"] == 12


def test_mmtv_batch_resident_plan_has_separate_heap_and_private_wram_abi():
    plan = UPMEMMatrixVectorPlan(
        rows=16,
        columns=32,
        batches=12,
        vector_mode="tasklet_private_wram",
    )
    manifest = plan.manifest()
    source = plan.device_source()

    assert plan.mram_offsets == {"matrix": 0, "vector": 24_576, "output": 26_112}
    assert plan.mram_image_bytes == 26_880
    assert "packed_matrix_vector" not in plan.mram_regions
    assert manifest["layout"] == "separate"
    assert manifest["vector_residency"] == "tasklet_private_wram"
    assert manifest["row_ownership"] == "tasklet-id-is-batch-id"
    assert manifest["heap_abi_order"] == ["matrix", "vector", "output"]
    assert manifest["row_layout"]["identity"] is True
    assert manifest["wram_allocation"] == {
        "kind": "per-tasklet-mem-alloc",
        "calls_per_tasklet": 3,
        "vector_bytes_per_tasklet": 128,
        "matrix_chunk_bytes_per_tasklet": 64,
        "output_bytes_per_tasklet": 64,
        "total_payload_bytes": 3_072,
    }
    assert plan.cost_features() == {
        "mram_read_calls": 408,
        "mram_write_calls": 12,
        "mram_calls": 420,
        "mram_read_bytes": 26_112,
        "mram_write_bytes": 768,
        "mram_bytes": 26_880,
        "barriers": 1,
        "wram_bytes": 3_072,
        "int32_macs": 6_144,
        "int32_multiplies": 0,
        "wram_allocation_calls": 36,
        "wram_allocation_bytes": 3_072,
        "static_wram_bytes": 0,
        "mem_reset_calls": 1,
    }

    assert source.count("mem_alloc(") == 3
    assert "mem_alloc(TENON_COLUMNS * sizeof(int32_t))" in source
    assert "mem_alloc(TENON_CHUNK_BYTES)" in source
    assert "mem_alloc(TENON_OUTPUT_BYTES)" in source
    assert source.index("mem_reset();") < source.index("barrier_wait(&tenon_barrier);")
    assert "(uintptr_t)(heap + 24576u)" in source
    assert "(uintptr_t)(heap + 26112u)" in source
    assert "__mram_noinit" not in source


def test_mmtv_batch_resident_plan_is_exact_shape_and_identity_only():
    with pytest.raises(ValueError, match="calibrated only for 12 batches"):
        UPMEMMatrixVectorPlan(
            rows=8,
            columns=32,
            batches=12,
            vector_mode="tasklet_private_wram",
        )
    with pytest.raises(ValueError, match="value-independent identity"):
        UPMEMMatrixVectorPlan(
            rows=16,
            columns=32,
            batches=12,
            vector_mode="tasklet_private_wram",
            physical_to_logical_rows=tuple(reversed(range(192))),
        )


def test_mv_row_permutation_composes_with_masked_carrier_without_changing_source():
    identity = UPMEMMatrixVectorPlan(rows=24, columns=3)
    permutation = tuple(reversed(range(identity.total_rows)))
    permuted = UPMEMMatrixVectorPlan(
        rows=24,
        columns=3,
        physical_to_logical_rows=permutation,
    )

    assert identity.physical_to_logical_rows == tuple(range(24))
    assert permuted.logical_to_physical_rows == permutation
    assert permuted.is_identity_row_layout is False
    assert permuted.device_source() == identity.device_source()
    manifest = permuted.manifest()["row_layout"]
    assert manifest["physical_to_logical_rows"] == list(permutation)
    assert manifest["logical_to_physical_rows"] == list(permutation)
    assert manifest["carrier"] == permuted.tasklet_layout.manifest()
    assert manifest["padding"]["policy"] == "implicit-fixed-zero-tail"
    assert permuted.output_region["flattened_order"] == "physical-tasklet-slot-order"

    with pytest.raises(ValueError, match="exactly one entry"):
        UPMEMMatrixVectorPlan(
            rows=24, columns=3, physical_to_logical_rows=permutation[:-1]
        )
    with pytest.raises(ValueError, match="must be a permutation"):
        UPMEMMatrixVectorPlan(
            rows=24,
            columns=3,
            physical_to_logical_rows=(0,) * 24,
        )


def test_mv_permuted_packing_and_output_gather_preserve_logical_reference():
    permutation = tuple(reversed(range(24)))
    plan = UPMEMMatrixVectorPlan(
        rows=24,
        columns=3,
        scale_factor=2,
        physical_to_logical_rows=permutation,
    )
    matrix = tuple(value for row in range(24) for value in (row, row + 1, row + 2))
    vector = (2, 3, 4)

    packed = plan.pack_fused_inputs(matrix, vector)
    first_logical_row = permutation[0]
    assert (
        packed[:16]
        == matrix[first_logical_row * 3 : (first_logical_row + 1) * 3] + (0,) * 13
    )
    assert packed[16:32] == vector + (0,) * 13

    logical = plan.reference_outputs(matrix, vector)
    physical = plan.pack_output(logical)
    assert len(physical) == plan.padded_rows
    assert physical[0] == logical[permutation[0]]
    assert plan.gather_output(physical) == logical


def test_mv_input_aware_balancer_has_stable_lpt_ties_and_phase_rotation():
    matrix = (1,) * 24
    vector = (1,)

    assert matrix_vector_row_mul_step_costs(24, 1, 1, matrix, vector) == (1,) * 24
    phase_zero = balance_matrix_vector_rows(24, 1, 1, matrix, vector, phase_rotation=0)
    assert phase_zero == tuple(
        logical_row for tasklet in range(12) for logical_row in (tasklet, tasklet + 12)
    )
    phase_one = balance_matrix_vector_rows(24, 1, 1, matrix, vector, phase_rotation=1)
    assert phase_one == tuple(
        logical_row
        for tasklet in range(12)
        for logical_row in (
            (tasklet, tasklet + 12) if tasklet % 2 == 0 else (tasklet + 12, tasklet)
        )
    )

    # Zero-cost rows distinguish the row-count tie-break from filling the
    # lowest-id zero-load bucket before moving to the next tasklet.
    assert (
        balance_matrix_vector_rows(24, 1, 1, (0,) * 24, vector, phase_rotation=0)
        == phase_zero
    )

    # Cost uses uint32 operands: min(uint32(-1), uint32(2)) has two bits.
    assert matrix_vector_row_mul_step_costs(24, 1, 1, (-1,) * 24, (2,)) == (2,) * 24
    with pytest.raises(ValueError, match="capacity-equal"):
        balance_matrix_vector_rows(25, 1, 1, (1,) * 25, (1,))


@pytest.mark.parametrize("column_tile", [4, 8, 16])
def test_gemm_nc_domain_has_exact_fused_packet_and_output_dma(column_tile):
    plan = UPMEMGEMMPlan(
        rows=12,
        columns=64,
        reduction=128,
        column_tile=column_tile,
        reduction_tile=16,
    )

    assert plan.packet_elements == (1 + column_tile) * 16
    assert plan.packet_bytes == (1 + column_tile) * 16 * 4
    assert plan.output_tile_bytes == column_tile * 4
    assert plan.packet_bytes % 8 == 0
    assert 8 <= plan.packet_bytes <= 2048
    assert 8 <= plan.output_tile_bytes <= 2048
    assert plan.wram_bytes <= 64 * 1024


def test_gemm_default_matches_measured_nc16_kc16_packet():
    plan = UPMEMGEMMPlan(rows=12, columns=64, reduction=128)
    manifest = plan.manifest()
    source = plan.device_source()

    assert plan.packet_elements == 272
    assert plan.packet_bytes == 1088
    assert plan.output_tile_bytes == 64
    assert plan.column_tiles == 4
    assert plan.reduction_tiles == 8
    assert manifest["packet_order"] == ["lhs_chunk", "rhs_column_chunks"]
    assert manifest["mram_regions"]["packed_lhs_rhs"]["bytes"] == 417792
    assert manifest["mram_regions"]["output"]["offset"] == 417792
    assert manifest["output_regions"]["output"]["bytes"] == 3072
    assert plan.cost_features()["mram_read_calls"] == 384
    assert plan.cost_features()["mram_write_calls"] == 48
    assert "#define TENON_PACKET_ELEMENTS 272u" in source
    assert "#define TENON_PACKET_BYTES 1088u" in source
    assert "column * TENON_REDUCTION_TILE + k" in source
    assert "mram_write(accumulator" in source


def test_gemm_1mm_uses_tasklet_private_mem_alloc_fast_path_with_same_mram_abi():
    plan = UPMEMGEMMPlan(rows=12, columns=128, reduction=64)
    manifest = plan.manifest()
    features = plan.cost_features()
    source = plan.device_source()

    assert plan.exact_row_fast_path is True
    assert plan.codegen_path == "tasklet-private-mem-alloc-exact-row"
    assert plan.mram_offsets == {
        "packed_lhs_rhs": 0,
        "lhs": 0,
        "rhs": 64,
        "output": 417_792,
    }
    assert plan.mram_regions["output"]["bytes"] == 6_144
    assert plan.mram_image_bytes == 423_936
    assert manifest["wram_allocation"] == {
        "kind": "per-tasklet-mem-alloc",
        "calls_per_tasklet": 2,
        "packet_bytes_per_tasklet": 1_088,
        "accumulator_bytes_per_tasklet": 64,
        "total_payload_bytes": 13_824,
    }
    assert manifest["mram_addressing"] == "heap-relative-typed-bases"
    assert manifest["row_ownership"] == "tasklet-id-is-output-row"
    assert features["barriers"] == 1
    assert features["wram_allocation_calls"] == 24
    assert features["wram_allocation_bytes"] == 13_824
    assert features["static_wram_bytes"] == 0
    assert features["mem_reset_calls"] == 1

    assert "BARRIER_INIT(tenon_barrier, NR_TASKLETS);" in source
    assert source.index("mem_reset();") < source.index("barrier_wait(&tenon_barrier);")
    assert source.count("mem_alloc(") == 2
    assert "mem_alloc(TENON_PACKET_BYTES)" in source
    assert "mem_alloc(TENON_OUTPUT_TILE_BYTES)" in source
    assert "__dma_aligned int32_t tenon_packets" not in source
    assert "__dma_aligned int32_t tenon_accumulators" not in source
    assert "local_row" not in source
    assert "TENON_ROWS_PER_TASKLET" not in source
    assert "(tid * TENON_COLUMN_TILES + column_block)" in source
    assert "(uintptr_t)(heap + 417792u)" in source


def test_gemm_multirow_shape_retains_static_generic_fallback():
    plan = UPMEMGEMMPlan(rows=192, columns=64, reduction=128)
    source = plan.device_source()
    features = plan.cost_features()

    assert plan.exact_row_fast_path is False
    assert plan.codegen_path == "static-wram-generic-row-loop"
    assert plan.manifest()["wram_allocation"]["kind"] == (
        "static-tasklet-indexed-arrays"
    )
    assert "mem_alloc(" not in source
    assert "__dma_aligned int32_t tenon_packets[NR_TASKLETS]" in source
    assert "local_row" in source
    assert features["barriers"] == 0
    assert features["wram_allocation_calls"] == 0
    assert features["wram_allocation_bytes"] == 0
    assert features["static_wram_bytes"] == plan.wram_bytes


@pytest.mark.parametrize("rows", [12, 192])
def test_gemm_eliminates_row_guard_when_tasklet_rows_are_exact(rows):
    plan = UPMEMGEMMPlan(rows=rows, columns=64, reduction=128)
    source = plan.device_source()

    assert plan.padded_rows == rows
    assert "if (row < TENON_ROWS)" not in source
    assert "for (uint32_t reduction_block = 0;" in source


def test_gemm_retains_row_guard_for_padded_tasklet_ownership():
    plan = UPMEMGEMMPlan(rows=13, columns=64, reduction=128)
    source = plan.device_source()

    assert plan.padded_rows == 24
    assert "if (row < TENON_ROWS)" in source
    assert source.index("if (row < TENON_ROWS)") < source.index(
        "for (uint32_t reduction_block = 0;"
    )


def test_every_source_is_deterministic_complete_and_runtime_backed():
    plans = (
        UPMEMElementwisePlan(1024),
        UPMEMSumReductionPlan(1024),
        UPMEMMatrixVectorPlan(96, 64, scale=True),
        UPMEMGEMMPlan(12, 64, 128),
    )

    for plan in plans:
        source = plan.device_source()
        assert source == plan.device_source()
        assert "#include <defs.h>" in source
        assert "#include <mram.h>" in source
        assert "#if NR_TASKLETS != 12" in source
        assert "int main(void)" in source
        assert "DPU_MRAM_HEAP_POINTER" in source
        assert "mram_read" in source
        assert "mram_write" in source
        assert plan.manifest()["sdk_complete"] is True
        assert plan.manifest()["runtime_mram_inputs"] is True
        assert len(plan.manifest()["source_sha256"]) == 64


def test_manifests_are_json_serializable_and_regions_are_exactly_aligned():
    plans = (
        UPMEMElementwisePlan(17, operation="axpby"),
        UPMEMSumReductionPlan(17),
        UPMEMMatrixVectorPlan(13, 17, scale=True),
        UPMEMGEMMPlan(13, 17, 17),
    )

    for plan in plans:
        json.dumps(plan.manifest(), sort_keys=True)
        assert plan.mram_image_bytes <= 64 * 1024 * 1024
        for region in plan.mram_regions.values():
            assert region["offset"] % 8 == 0
            assert region["bytes"] % 8 == 0
        for region in plan.output_regions.values():
            assert region["direction"] == "output"


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: UPMEMElementwisePlan(128, dma_bytes=12), "8-byte aligned"),
        (lambda: UPMEMElementwisePlan(128, dma_bytes=2048), "fused operand DMA"),
        (lambda: UPMEMSumReductionPlan(128, dma_bytes=4), "between 8 and 2048"),
        (
            lambda: UPMEMMatrixVectorPlan(
                12, 64, dma_bytes=128, vector_mode="shared_wram"
            ),
            "fused_replicated",
        ),
        (lambda: UPMEMGEMMPlan(12, 64, 128, column_tile=32), "between 8 and 2048"),
    ],
)
def test_illegal_dma_or_unsupported_physical_modes_fail_closed(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()


def test_mram_capacity_is_enforced_before_source_export():
    with pytest.raises(ValueError, match="MRAM bytes"):
        UPMEMElementwisePlan(32 * 1024 * 1024)
