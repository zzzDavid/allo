# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-simulator tests for irregular UPMEM physical plans."""

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from allo.pim.upmem_irregular import (
    GradientFormula,
    UPMEMFeatureGradientPlan,
    UPMEMHistogramPlan,
    UPMEMKMeansDistancesPlan,
    UPMEMKMeansPlan,
    UPMEMSelectionFlagsPlan,
    UPMEMStableSelectionPlan,
)


def _canonical_plans():
    return (
        UPMEMHistogramPlan(12_288, 128, 12),
        UPMEMStableSelectionPlan(12_288),
        UPMEMSelectionFlagsPlan(12_288),
        UPMEMKMeansDistancesPlan(120, 8, 4),
        UPMEMKMeansPlan(120, 8, 4),
        UPMEMFeatureGradientPlan(120, 8, "linear_fixed_point", 5, 8),
        UPMEMFeatureGradientPlan(120, 8, "logistic_zero"),
    )


@pytest.mark.parametrize("plan", _canonical_plans())
def test_every_plan_is_a_complete_frozen_12_tasklet_translation_unit(plan):
    source = plan.device_source()
    manifest = plan.manifest()

    assert plan.tasklets == plan.num_tasklets == 12
    assert plan.compile_flags == ("-DNR_TASKLETS=12",)
    assert manifest["compile_flags"] == ["-DNR_TASKLETS=12"]
    assert manifest["sdk_compilable_translation_unit"] is True
    assert manifest["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert "#if !defined(NR_TASKLETS) || NR_TASKLETS != 12" in source
    assert "#define TENON_NUM_TASKLETS 12" in source
    assert "#define TENON_LAYOUT_TASKLETS 16" in source
    assert "BARRIER_INIT" in source
    assert "int main(void)" in source
    assert "DPU_MRAM_HEAP_POINTER" in source
    assert "mram_read" in source
    assert "mram_write" in source
    assert "__mram_noinit" not in source
    assert plan.wram_bytes <= 64 * 1024
    with pytest.raises(FrozenInstanceError):
        plan.num_tasklets = 8


@pytest.mark.parametrize("plan", _canonical_plans())
def test_mram_regions_are_aligned_non_overlapping_and_outputs_are_contiguous(plan):
    regions = list(plan.mram_regions.values())
    assert regions[0].offset == 0
    for index, region in enumerate(regions):
        assert region.offset % 8 == 0
        assert region.bytes >= 8 and region.bytes % 8 == 0
        assert 0 <= region.logical_bytes <= region.bytes
        if index:
            previous = regions[index - 1]
            assert region.offset >= previous.offset + previous.bytes
    assert plan.mram_image_bytes == regions[-1].offset + regions[-1].bytes

    outputs = list(plan.output_regions.values())
    for previous, current in zip(outputs, outputs[1:]):
        assert current.offset == previous.offset + previous.bytes
    contract = plan.output_contract
    assert contract["combined_offset"] == outputs[0].offset
    assert contract["combined_bytes"] == (
        outputs[-1].offset + outputs[-1].bytes - outputs[0].offset
    )
    assert manifest_regions(plan) == plan.mram_offsets


def manifest_regions(plan):
    return {
        name: region["offset"]
        for name, region in plan.manifest()["mram_regions"].items()
    }


@pytest.mark.parametrize("plan", _canonical_plans())
def test_layout_is_compiler_owned_masked_padded_16_f2(plan):
    layout = plan.linear_layout
    manifest = plan.layout_manifest

    assert manifest["kind"] == "linear-layout-f2"
    assert manifest["owner"] == "compiler"
    assert manifest["policy"] == "masked-padded-16"
    assert manifest["physical_tasklets"] == 12
    assert manifest["padded_tasklet_extent"] == 16
    assert manifest["active_tasklets"] == list(range(12))
    assert manifest["masked_tasklets"] == [12, 13, 14, 15]
    assert manifest["active_mask"] == 0xFFF
    assert manifest["output_dims"][0] == {"name": "tasklet", "size": 16}
    assert layout.output_size("tasklet") == 16
    assert layout.apply(tasklet_block=11, tasklet_local=0)[0] == 11
    # F2 retains all 16 coordinates; the compiler-owned mask, not a false
    # eight-tasklet subgroup, marks the four non-physical coordinates invalid.
    assert layout.apply(tasklet_block=12, tasklet_local=0)[0] == 12


def test_canonical_offsets_and_explicit_multi_output_padding():
    histogram = UPMEMHistogramPlan(12_288, 128, 12)
    selection = UPMEMStableSelectionPlan(12_288)
    distances = UPMEMKMeansDistancesPlan(120, 8, 4)
    kmeans = UPMEMKMeansPlan(120, 8, 4)
    gradient = UPMEMFeatureGradientPlan(120, 8, GradientFormula.LINEAR_FIXED_POINT)

    assert histogram.mram_offsets == {"input": 0, "histogram": 49_152}
    assert selection.mram_offsets == {
        "input": 0,
        "selected_count": 49_152,
        "selected_values": 49_160,
    }
    assert selection.mram_regions["selected_count"].logical_bytes == 4
    assert selection.mram_regions["selected_count"].padding_bytes == 4
    assert selection.output_contract["comparison_bytes"] == ("8 + 4 * selected_count")
    assert distances.mram_offsets == {"fused_pairs": 0, "distances": 30_720}
    assert distances.mram_regions["distances"].bytes == 3_840
    assert kmeans.mram_offsets == {
        "points": 0,
        "initial_centroids": 3_840,
        "centroids": 3_968,
        "counts": 4_096,
    }
    assert kmeans.output_contract["region_order"] == ["centroids", "counts"]
    assert gradient.mram_offsets == {"samples": 0, "gradient": 4_320}

    padded_counts = UPMEMKMeansPlan(3, 2, 3).mram_regions["counts"]
    assert (padded_counts.logical_bytes, padded_counts.bytes) == (12, 16)


def test_histogram_has_private_wram_merge_and_exact_reference():
    plan = UPMEMHistogramPlan(8, 4, 3, dma_elements=2)
    source = plan.device_source()

    assert plan.reference_histogram(range(8)) == (2, 2, 2, 2)
    assert "tenon_private_hist[12][4]" in source
    assert "tenon_merged_hist" in source
    assert "total += tenon_private_hist[owner][bin]" in source
    assert plan.cost_features["primitive_iterations"]["MUL"] == 8
    assert plan.cost_features["dma_bytes"] == {
        "LD_MRAM": 32,
        "ST_MRAM": 16,
    }


def test_selection_is_stable_and_emits_prefix_bulk_and_boundary_writes():
    values = (2, -3, 4, 7, 0, -8, 9)
    plan = UPMEMStableSelectionPlan(len(values), dma_elements=2)
    source = plan.device_source()

    assert plan.reference_selection(values) == ((-3, 7, 9), 3)
    assert "tenon_private_values[12][1]" in source
    assert "tenon_prefixes[owner] = prefix" in source
    assert "tenon_write_private" in source
    assert "Stitch every odd prefix" in source
    assert "__dma_aligned uint32_t header[2] = {total, 0u}" in source
    assert plan.output_contract["order"] == "stable-input-order"


def test_selection_flags_preserve_the_archived_device_partition():
    values = (2, -3, 4, 7, 0, -8, 9, 12, 13, 14, 15, 16) * 2
    plan = UPMEMSelectionFlagsPlan(len(values), dma_elements=2)
    source = plan.device_source()

    expected = (0, -3, 0, 7, 0, 0, 9, 0, 13, 0, 15, 0) * 2
    assert plan.reference_flags(values) == expected
    assert plan.output_contract["host_compaction"] == (
        "excluded-to-match-archived-device-kernel-scope"
    )
    assert "if ((buffer[i] & 1) == 0)" in source
    assert plan.mram_offsets == {"input": 0, "flags": 96}


def test_kmeans_returns_full_assignment_counts_and_signed_rounded_centroids():
    plan = UPMEMKMeansPlan(4, 2, 2)
    points = (-2, -1, 2, 1, 9, 11, 11, 9)
    initial = (0, 0, 10, 10)

    assert plan.reference_iteration(points, initial) == ((0, 0, 10, 10), (2, 2))
    assert plan.signed_round_closest(1, 2) == 1
    assert plan.signed_round_closest(-1, 2) == -1
    assert plan.signed_round_closest(-4, 3) == -1

    source = plan.device_source()
    assert "distance < best_distance" in source
    assert "tenon_private_sums[12][2][2]" in source
    assert "tenon_private_counts[12][2]" in source
    assert "tenon_div_round_closest" in source
    assert "count == 0u ? 0" in source
    assert plan.output_contract["assignment_tie_break"] == "lowest-cluster-index"
    assert plan.output_contract["centroid_rounding"] == (
        "signed-round-closest-ties-away-from-zero"
    )


def test_kmeans_distances_matches_the_archived_device_partition_and_reference():
    canonical = UPMEMKMeansDistancesPlan(120, 8, 4)
    source = canonical.device_source()

    assert canonical.total_pairs == 480
    assert canonical.pairs_per_tasklet == 40
    assert canonical.packet_words == 16
    assert canonical.packet_bytes == 64
    assert canonical.grouped_output_bytes == 320
    assert canonical.distance_upper_bound == 80_000
    assert canonical.mram_offsets == {"fused_pairs": 0, "distances": 30_720}
    assert canonical.output_contract["combined_bytes"] == 3_840
    assert canonical.output_contract["order"] == "point-major-cluster-major"
    assert canonical.output_contract["host_postprocessing"] == [
        "argmin",
        "counts",
        "centroid_update",
    ]
    assert canonical.manifest()["comparison_partition"]["full_iteration_plan"] == (
        "UPMEMKMeansPlan"
    )
    assert canonical.manifest()["distance_bound"] == {
        "input_dtype": "int32",
        "input_range": [-50, 50],
        "accumulator_dtype": "int32",
        "upper_bound": 80_000,
        "required_maximum": (1 << 31) - 1,
    }
    assert "#define TENON_PACKET_BYTES 64u" in source
    assert "#define TENON_LOCAL_PAIRS 40u" in source
    assert "#define TENON_GROUPED_OUTPUT_BYTES 320u" in source
    assert "int32_t distance = 0" in source
    assert "distances[local_pair] = (int64_t)distance" in source
    assert source.count("mram_write(") == 1
    assert canonical.cost_features["dma_calls"] == {
        "LD_MRAM": 480,
        "ST_MRAM": 12,
    }

    small = UPMEMKMeansDistancesPlan(6, 2, 2, max_abs_value=10)
    points = (0, 0, 1, 0, 0, 1, 10, 10, -1, -1, 2, 2)
    centroids = (0, 0, 1, 1)
    fused = small.pack_fused_pairs(points, centroids)
    expected = (0, 2, 1, 1, 1, 1, 200, 162, 2, 8, 8, 2)

    assert fused[:8] == (0, 0, 0, 0, 0, 0, 1, 1)
    assert len(fused) == small.total_pairs * small.packet_words
    assert small.reference_distances(fused) == expected
    assert small.reference_distances(points, centroids) == expected


def test_feature_gradient_formulas_match_small_exact_examples():
    # Rows are [x0, x1, label].
    packed = (2, -3, 1, -4, 5, -2)
    linear = UPMEMFeatureGradientPlan(2, 2, "linear", shift=1, overflow_shift=2)
    logistic = UPMEMFeatureGradientPlan(2, 2, "logistic_at_zero")

    assert linear.formula is GradientFormula.LINEAR_FIXED_POINT
    assert logistic.formula is GradientFormula.LOGISTIC_ZERO
    assert linear.reference_gradient(packed) == (-5, 6)
    assert logistic.reference_gradient(packed) == (-22, 28)
    assert "(product * -((int64_t)1 << 1)) >> 2" in linear.device_source()
    assert linear.manifest()["fixed_point"]["lowering"] == (
        "upmem-signed-arithmetic-right-shift"
    )
    assert "(int64_t)1 - (int64_t)2 * (int64_t)label" in (logistic.device_source())
    assert "floor_div_pow2" in linear.output_contract["equation"]
    assert logistic.output_contract["equation"] == (
        "gradient[j] = sum_i x[i,j] * (1 - 2*y[i])"
    )


@pytest.mark.parametrize("plan", _canonical_plans())
def test_generation_and_json_manifest_are_deterministic(plan):
    assert plan.device_source() == plan.device_source()
    first = json.dumps(plan.manifest(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(plan.manifest(), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_legality_checks_fail_before_emitting_an_invalid_translation_unit():
    with pytest.raises(ValueError, match="exactly 12 tasklets"):
        UPMEMHistogramPlan(128, 8, 3, num_tasklets=8)
    with pytest.raises(ValueError, match="8-byte-aligned DMA"):
        UPMEMHistogramPlan(128, 8, 3, dma_elements=3)
    with pytest.raises(ValueError, match="64 MiB MRAM"):
        UPMEMHistogramPlan(20_000_000, 8, 3)
    with pytest.raises(ValueError, match="64 KiB WRAM"):
        UPMEMStableSelectionPlan(20_000)
    with pytest.raises(ValueError, match="predicate='odd'"):
        UPMEMStableSelectionPlan(128, predicate="positive")
    with pytest.raises(ValueError, match="exactly one k-means iteration"):
        UPMEMKMeansPlan(120, 8, 4, iterations=2)
    with pytest.raises(ValueError, match="dimension must be even"):
        UPMEMKMeansPlan(120, 7, 4)
    with pytest.raises(ValueError, match="divide evenly across 12 tasklets"):
        UPMEMKMeansDistancesPlan(5, 2, 2)
    with pytest.raises(ValueError, match="fused packet DMA"):
        UPMEMKMeansDistancesPlan(12, 257, 1, max_abs_value=0)
    with pytest.raises(ValueError, match="grouped output DMA"):
        UPMEMKMeansDistancesPlan(12 * 257, 1, 1, max_abs_value=0)
    with pytest.raises(ValueError, match="does not fit signed int32"):
        UPMEMKMeansDistancesPlan(12, 256, 1, max_abs_value=2_000)
    with pytest.raises(ValueError, match="exceeds the declared max_abs_value"):
        UPMEMKMeansDistancesPlan(6, 2, 2, max_abs_value=2).reference_distances(
            (3,) * 48
        )
    with pytest.raises(ValueError, match="2048-byte DMA"):
        UPMEMFeatureGradientPlan(120, 512, "linear")
    with pytest.raises(ValueError, match="gradient formula"):
        UPMEMFeatureGradientPlan(120, 8, "softmax")
    with pytest.raises(ValueError, match="overflow_shift"):
        UPMEMFeatureGradientPlan(120, 8, "linear", overflow_shift=63)
