# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import allo
from allo.ir.types import uint16
from allo.pim.apu_v1_histogram import (
    APUv1DenseHistogramLowering,
    analyze_apu_v1_dense_histogram,
)
from allo.pim.apu_v1_moments import (
    APUv1BivariateMomentsLowering,
    analyze_apu_v1_bivariate_moments,
)
from allo.pim.apu_v1_record_frequency import (
    APUv1RecordFrequencyLowering,
    analyze_apu_v1_record_frequency,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def dense_histogram(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for bin_index in range(256):
        for item in allo.reduction(32768):
            if values[item] == bin_index:
                counts[bin_index] += 1
    return counts


def record_frequency(records: uint16[3, 32768], queries: uint16[3, 128]) -> uint16[128]:
    counts: uint16[128] = 0
    for query in range(128):
        for record in allo.reduction(32768):
            if records[0, record] == queries[0, query]:
                if records[1, record] == queries[1, query]:
                    if records[2, record] == queries[2, query]:
                        counts[query] += 1
    return counts


def bivariate_moments(first: uint16[8, 4096], second: uint16[8, 4096]) -> uint16[8, 5]:
    result: uint16[8, 5] = 0
    for group in range(8):
        for lane in allo.reduction(4096):
            result[group, 0] += first[group, lane]
            result[group, 1] += second[group, lane]
            result[group, 2] += first[group, lane] * first[group, lane]
            result[group, 3] += second[group, lane] * second[group, lane]
            result[group, 4] += first[group, lane] * second[group, lane]
    return result


def _compile(kernel, result_name):
    phase = allo.APUv1Phase(
        kernel,
        result_names=(result_name,),
        vectorize="required",
    )
    return allo.compile(
        allo.APUv1Program((phase,), name="native_reduction_test"),
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    ).compiled


def _assert_native_cost_graph(
    compiled, route, *, cycle_bounds, required_activity_labels
):
    graph = compiled.execution_graph
    assert graph.name == route
    assert graph.metadata["route"] == route
    assert graph.metadata["estimator"] == "native_gvml_structural_cost"
    assert graph.metadata["operation_inventory"] == (
        compiled.native_vector_lowering.operation_inventory()
    )
    labels = {activity.label for activity in graph.activities}
    assert set(required_activity_labels) <= labels
    assert "scalar_arc_c" not in labels
    estimate = compiled.cost.evaluate(graph)
    assert cycle_bounds[0] < estimate.cycles < cycle_bounds[1]


def test_dense_histogram_selects_paired_immediate_count_route():
    compiled = _compile(dense_histogram, "counts")
    lowering = compiled.native_vector_lowering
    assert isinstance(lowering, APUv1DenseHistogramLowering)
    assert lowering.route == "gvml_dense_histogram_pair_count"
    assert lowering.analysis.values_role == "values"
    assert lowering.analysis.counts_role == "counts"

    source = compiled.device_source
    assert source.count("direct_dma_l4_to_l1_32k(") == 1
    assert "for (uint16_t bin = 0; bin < HIST_BINS; bin += 2)" in source
    assert source.count("gvml_eq_imm_16(") == 2
    assert "gvml_2_fast_count_m_g32k(" in source
    assert "data->mem_hndl_counts" in source
    assert lowering.operation_inventory()["gvml_eq_imm_16"] == 256
    assert lowering.operation_inventory()["gvml_2_fast_count_m_g32k"] == 128
    _assert_native_cost_graph(
        compiled,
        lowering.route,
        cycle_bounds=(50_000, 250_000),
        required_activity_labels=(
            "direct_dma_l4_to_l1_32k",
            "gvml_eq_imm_16",
            "gvml_2_fast_count_m_g32k",
            "scalar_l4_store_u16",
        ),
    )

    values = np.arange(32768, dtype=np.uint16)
    packed = lowering.pack_inputs({"values": values})
    assert packed["values"] is values


def test_record_frequency_keeps_chunks_resident_and_counts_query_pairs():
    compiled = _compile(record_frequency, "counts")
    lowering = compiled.native_vector_lowering
    assert isinstance(lowering, APUv1RecordFrequencyLowering)
    assert lowering.analysis.chunks == 3
    assert lowering.analysis.queries == 128
    source = compiled.device_source
    assert "gvml_2_fast_count_m_g32k(" in source
    assert "gvml_eq_imm_16(" in source
    assert "gvml_and_m(GVML_MRK0, GVML_MRK0, GVML_MRK2);" in source
    assert "data->mem_hndl_counts" in source

    records = np.arange(3 * 32768, dtype=np.uint16).reshape(3, 32768)
    queries = np.arange(3 * 128, dtype=np.uint16).reshape(3, 128)
    packed = lowering.pack_inputs({"records": records, "queries": queries})
    assert packed["records"].shape == records.shape
    assert packed["queries"].shape == (512,)
    np.testing.assert_array_equal(packed["queries"][: 3 * 128], queries.reshape(-1))
    assert not np.any(packed["queries"][3 * 128 :])
    assert lowering.operation_inventory()["gvml_2_fast_count_m_g32k"] == 64
    assert lowering.operation_inventory()["gvml_eq_imm_16"] == 384
    _assert_native_cost_graph(
        compiled,
        lowering.route,
        cycle_bounds=(100_000, 500_000),
        required_activity_labels=(
            "dma_l4_l3",
            "direct_dma_l4_to_l1_32k",
            "gvml_eq_imm_16",
            "gvml_and_m",
            "gvml_2_fast_count_m_g32k",
            "scalar_l4_store_u16",
        ),
    )


def test_bivariate_moments_selects_one_load_five_reduction_route():
    compiled = _compile(bivariate_moments, "result")
    lowering = compiled.native_vector_lowering
    assert isinstance(lowering, APUv1BivariateMomentsLowering)
    assert lowering.analysis.batches == 8
    assert lowering.analysis.points_per_batch == 4096
    source = compiled.device_source
    assert source.count("direct_dma_l4_to_l1_32k(") == 2
    assert source.count("gvml_mul_u16(") == 3
    assert source.count("gvml_add_subgrps_u16_grp(") == 5
    assert "GVML_P2_4K" in source
    assert "data->mem_hndl_result" in source
    _assert_native_cost_graph(
        compiled,
        lowering.route,
        cycle_bounds=(40_000, 200_000),
        required_activity_labels=(
            "direct_dma_l4_to_l1_32k",
            "gvml_mul_u16",
            "gvml_add_subgrps_u16_grp",
            "gvml_get_entry_16",
            "scalar_l4_store_u16",
        ),
    )


def test_native_reduction_recognizers_fail_closed_on_semantic_mutations():
    histogram = _compile(dense_histogram, "counts")
    mutated_histogram = histogram.artifact.source_mlir.replace(
        "arith.cmpi eq", "arith.cmpi ne", 1
    )
    assert (
        analyze_apu_v1_dense_histogram(
            mutated_histogram,
            histogram.arguments,
            function=histogram.schedule.top_func_name,
        )
        is None
    )

    frequency = _compile(record_frequency, "counts")
    mutated_frequency = frequency.artifact.source_mlir.replace(
        "arith.cmpi eq", "arith.cmpi ne", 1
    )
    assert (
        analyze_apu_v1_record_frequency(
            mutated_frequency,
            frequency.arguments,
            function=frequency.schedule.top_func_name,
        )
        is None
    )

    moments = _compile(bivariate_moments, "result")
    mutated_moments = moments.artifact.source_mlir.replace(
        "arith.muli", "arith.addi", 1
    )
    assert (
        analyze_apu_v1_bivariate_moments(
            mutated_moments,
            moments.arguments,
            function=moments.schedule.top_func_name,
        )
        is None
    )
