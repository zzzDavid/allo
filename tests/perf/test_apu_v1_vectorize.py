# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import allo
import pytest
from allo.ir.types import float16, int16, int32

from allo.pim.apu_v1_vectorize import (
    ContractionAnalysisError,
    IllegalContractionError,
    NoContractionError,
    analyze_apu_v1_contraction,
    analyze_apu_v1_contractions,
    generate_apu_v1_plans,
    generate_apu_v1_vectorization_candidates,
)
from allo.pim.apu_v1_vector_program import _realize


# The deliberately unrelated function names prove that recognition is based on
# MLIR loop/access structure, never a benchmark-name allowlist.
def weather_forecast(left: float16[4, 8], right: float16[8, 6], result: float16[4, 6]):
    for row, column in allo.grid(4, 6):
        for depth in allo.reduction(8):
            result[row, column] += left[row, depth] * right[depth, column]


def integer_combine(left: int32[3, 5], right: int32[5, 7], result: int32[3, 7]):
    for x, y in allo.grid(3, 7):
        for z in allo.reduction(5):
            result[x, y] += right[z, y] * left[x, z]


def binary_bit_combine(left: int32[3, 5], right: int32[5, 7], result: int32[3, 7]):
    for x, y in allo.grid(3, 7):
        for z in allo.reduction(5):
            result[x, y] += left[x, z] & right[z, y]


def packed_xnor_popcount(
    left: int16[32, 8], right: int16[8, 1024], result: int16[32, 1024]
):
    for x, y in allo.grid(32, 1024):
        for z in allo.reduction(8):
            result[x, y] += allo.popcount(~(left[x, z] ^ right[z, y]))


def non_power_of_two_micro(
    left: float16[60, 80], right: float16[80, 70], result: float16[60, 70]
):
    for m, n in allo.grid(60, 70):
        for k in allo.reduction(80):
            result[m, n] += left[m, k] * right[k, n]


def matrix_vector(left: float16[8, 16], right: float16[16], result: float16[8]):
    for row in allo.grid(8):
        for depth in allo.reduction(16):
            result[row] += left[row, depth] * right[depth]


def batched_contraction(
    left: float16[3, 4, 8], right: float16[8, 5], result: float16[3, 4, 5]
):
    for batch, row, column in allo.grid(3, 4, 5):
        for depth in allo.reduction(8):
            result[batch, row, column] += left[batch, row, depth] * right[depth, column]


def full_problem_micro(
    left: float16[1024, 64],
    right: float16[64, 1024],
    result: float16[1024, 1024],
):
    for m, n in allo.grid(1024, 1024):
        for k in allo.reduction(64):
            result[m, n] += left[m, k] * right[k, n]


def not_a_contraction(source: float16[4, 6], result: float16[4, 6]):
    for i, j in allo.grid(4, 6):
        result[i, j] = source[i, j] + source[i, j]


def _module(function):
    return allo.customize(function, enable_tensor=False).module


def test_recovers_fp16_axes_accesses_and_roles_without_kernel_names():
    analysis = analyze_apu_v1_contraction(_module(weather_forecast))

    assert analysis.function == "weather_forecast"
    assert analysis.output_axes == ("row", "column")
    assert analysis.parallel_axes == ("row", "column")
    assert analysis.reduction_axis == "depth"
    assert analysis.axis_extents == {"row": 4, "column": 6, "depth": 8}
    assert analysis.lhs.value == "left"
    assert analysis.lhs.indices == ("row", "depth")
    assert analysis.rhs.value == "right"
    assert analysis.rhs.indices == ("depth", "column")
    assert analysis.accumulator.value == "result"
    assert analysis.output.value == "result"
    assert analysis.output.indices == ("row", "column")
    assert analysis.numeric_type == "f16"
    assert analysis.multiply_operation == "arith.mulf"
    assert analysis.combine_operation == "arith.addf"


def test_accepts_commuted_binary_integer_mul_add_combine():
    analysis = analyze_apu_v1_contraction(_module(integer_combine))
    assert analysis.output_axes == ("x", "y")
    assert analysis.reduction_axis == "z"
    assert {analysis.lhs.value, analysis.rhs.value} == {"left", "right"}
    assert analysis.numeric_type == "i32"
    assert analysis.multiply_operation == "arith.muli"
    assert analysis.combine_operation == "arith.addi"


def test_accepts_binary_matmul_style_and_accumulate():
    analysis = analyze_apu_v1_contraction(_module(binary_bit_combine))
    assert analysis.output_axes == ("x", "y")
    assert analysis.reduction_axis == "z"
    assert analysis.multiply_operation == "arith.andi"
    assert analysis.combine_operation == "arith.addi"


_PACKED_XNOR_POPCOUNT = r"""
module {
  func.func @packed_similarity(%A: memref<2x8xi16>, %B: memref<8x4xi16>, %C: memref<2x4xi16>) {
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 8 {
          %a = affine.load %A[%i, %k] {from = "A"} : memref<2x8xi16>
          %b = affine.load %B[%k, %j] {from = "B"} : memref<8x4xi16>
          %xor = arith.xori %a, %b : i16
          %ones = arith.constant -1 : i16
          %xnor = arith.xori %xor, %ones : i16
          %bits = math.ctpop %xnor : i16
          %c = affine.load %C[%i, %j] {from = "C"} : memref<2x4xi16>
          %sum = arith.addi %c, %bits : i16
          affine.store %sum, %C[%i, %j] {to = "C"} : memref<2x4xi16>
        } {loop_name = "k", reduction}
      } {loop_name = "j"}
    } {loop_name = "i"}
    return
  }
}
"""


@pytest.mark.parametrize("spelling", ["math.ctpop", "allo.popcount", "llvm.ctpop"])
def test_recognizes_target_neutral_xnor_popcount_dataflow(spelling):
    module = _PACKED_XNOR_POPCOUNT.replace("math.ctpop", spelling)
    analysis = analyze_apu_v1_contraction(module)
    assert analysis.multiply_operation == "allo.xnor_popcount"
    assert analysis.packed_word_bits == 16
    assert analysis.numeric_type == "i16"
    assert analysis.lhs.value == "A"
    assert analysis.rhs.value == "B"

    plans = generate_apu_v1_plans(analysis)
    assert [operation.name for operation in plans[-1].operations] == [
        "XOR_16",
        "NOT_16",
        "POPCOUNT_16",
        "ADD",
    ]
    if spelling == "math.ctpop":
        realization, error = _realize(analysis, plans[2])
        assert error is None
        source = realization.device_source()
        for api in (
            "gvml_xor_16",
            "gvml_not_16",
            "gvml_popcount_16",
            "gvml_add_s16",
        ):
            assert api in source
        assert "gvml_add_subgrps_s16_grp" not in source


def test_popcount_without_all_ones_inversion_is_not_xnor_contraction():
    module = _PACKED_XNOR_POPCOUNT.replace(
        "%ones = arith.constant -1 : i16", "%ones = arith.constant 7 : i16"
    )
    with pytest.raises(NoContractionError, match="no supported"):
        analyze_apu_v1_contraction(module)


def test_allo_popcount_dsl_reaches_mlir_analysis_and_gvml_lowering():
    module = _module(packed_xnor_popcount)
    text = str(module)
    assert "math.ctpop" in text
    analysis = analyze_apu_v1_contraction(module)
    assert analysis.multiply_operation == "allo.xnor_popcount"
    assert analysis.packed_word_bits == 16

    plan = generate_apu_v1_plans(analysis)[-1]
    realization, error = _realize(analysis, plan)
    assert error is None
    source = realization.device_source()
    assert "gvml_popcount_16" in source
    assert "gvml_add_s16" in source


def test_discovers_one_dimensional_polybench_style_reduction_but_plans_fail_closed():
    analysis = analyze_apu_v1_contraction(_module(matrix_vector))
    assert analysis.output_axes == ("row",)
    assert analysis.reduction_axis == "depth"
    assert analysis.lhs.indices == ("row", "depth")
    assert analysis.rhs.indices == ("depth",)
    with pytest.raises(
        IllegalContractionError, match="matrix-vector reductions are the next"
    ):
        generate_apu_v1_plans(analysis)


def test_discovers_doitgen_style_batched_contraction_but_plans_fail_closed():
    analysis = analyze_apu_v1_contraction(_module(batched_contraction))
    assert analysis.output_axes == ("batch", "row", "column")
    assert analysis.reduction_axis == "depth"
    with pytest.raises(IllegalContractionError, match="Batched-output contractions"):
        generate_apu_v1_plans(analysis)


def test_rejects_elementwise_code_without_reduction():
    with pytest.raises(NoContractionError, match="no supported"):
        analyze_apu_v1_contraction(_module(not_a_contraction))


_REDUCTION_INDEXES_OUTPUT = r"""
module {
  func.func @unsafe(%A: memref<4x8xf16>, %B: memref<8x6xf16>, %C: memref<4x8xf16>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 6 {
        affine.for %k = 0 to 8 {
          %a = affine.load %A[%i, %k] {from = "A"} : memref<4x8xf16>
          %b = affine.load %B[%k, %j] {from = "B"} : memref<8x6xf16>
          %p = arith.mulf %a, %b : f16
          %c = affine.load %C[%i, %k] {from = "C"} : memref<4x8xf16>
          %v = arith.addf %c, %p : f16
          affine.store %v, %C[%i, %k] {to = "C"} : memref<4x8xf16>
        } {loop_name = "k", reduction}
      } {loop_name = "j"}
    } {loop_name = "i"}
    return
  }
}
"""


_MISSING_REDUCTION_OPERAND = r"""
module {
  func.func @unsafe(%A: memref<4x8xf16>, %B: memref<4x6xf16>, %C: memref<4x6xf16>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 6 {
        affine.for %k = 0 to 8 {
          %a = affine.load %A[%i, %k] {from = "A"} : memref<4x8xf16>
          %b = affine.load %B[%i, %j] {from = "B"} : memref<4x6xf16>
          %p = arith.mulf %a, %b : f16
          %c = affine.load %C[%i, %j] {from = "C"} : memref<4x6xf16>
          %v = arith.addf %c, %p : f16
          affine.store %v, %C[%i, %j] {to = "C"} : memref<4x6xf16>
        } {loop_name = "k", reduction}
      } {loop_name = "j"}
    } {loop_name = "i"}
    return
  }
}
"""


def test_rejects_reduction_axis_in_output_address():
    with pytest.raises(IllegalContractionError, match="reduction axis.*output"):
        analyze_apu_v1_contraction(_REDUCTION_INDEXES_OUTPUT)


def test_rejects_operand_that_does_not_reference_reduction_axis():
    with pytest.raises(IllegalContractionError, match="both multiplicands"):
        analyze_apu_v1_contraction(_MISSING_REDUCTION_OPERAND)


def test_multiple_contractions_require_explicit_function_selection():
    left = str(_module(weather_forecast)).replace("@weather_forecast", "@first")
    right = str(_module(integer_combine)).replace("@integer_combine", "@second")
    combined = left[:-2] + right.split("module {", 1)[1]
    analyses = analyze_apu_v1_contractions(combined)
    assert [analysis.function for analysis in analyses] == ["first", "second"]
    with pytest.raises(ContractionAnalysisError, match="exactly one"):
        analyze_apu_v1_contraction(combined)
    assert analyze_apu_v1_contraction(combined, function="second").numeric_type == "i32"


def test_generates_four_named_micro_candidates_with_layout_metadata():
    analysis = analyze_apu_v1_contraction(_module(weather_forecast))
    candidates = generate_apu_v1_vectorization_candidates(analysis)

    assert [candidate.name for candidate in candidates] == [
        "baseline_spatial_reduction",
        "temporal_svp",
        "temporal_dma_coalescing",
        "temporal_dma_coalescing_broadcast_friendly",
    ]
    assert all(candidate.analysis is analysis for candidate in candidates)
    baseline, temporal, coalesced, broadcast = [item.plan for item in candidates]
    assert baseline.name == "baseline_spatial_reduction"
    assert baseline.reduction_strategy.axis == "depth"
    assert temporal.temporal_strategy == "svp"
    assert tuple(axis.name for axis in temporal.iteration_layout.temporal_axes) == (
        "depth",
    )
    assert any(transfer.coalesced for transfer in coalesced.transfers)
    assert any(transfer.broadcast for transfer in broadcast.transfers)
    assert broadcast.metadata["dtype"] == "f16"
    assert broadcast.metadata["axis_extents"] == {
        "row": 4,
        "column": 6,
        "depth": 8,
    }
    assert [operation.name for operation in baseline.operations] == [
        "MUL",
        "GROUP_REDUCE",
    ]
    assert [operation.name for operation in broadcast.operations] == ["MUL", "ADD"]
    assert [operation.count for operation in baseline.operations] == [1, 1]
    assert [operation.count for operation in broadcast.operations] == [8, 8]
    assert broadcast.metadata["scalar_work_items"] == 4 * 6 * 8
    assert broadcast.metadata["output_tiles"] == 1
    assert broadcast.metadata["vr_batches"] == 8
    assert broadcast.metadata["vector_compute_calls"] == 8
    assert broadcast.metadata["group_reduce_calls"] == 0
    assert broadcast.metadata["temporal_steps"] == 8
    assert broadcast.metadata["transfer_facts"]["coalesced"]
    for transfer in baseline.transfers[:2]:
        assert transfer.route[0].kind == "direct"
        assert transfer.source_layout.storage == "l4_expanded"
        assert transfer.route[0].metrics().expansion_factor == 1
    assert set(broadcast.metadata["transfer_facts"]["broadcast_values"]) == {
        "left",
        "right",
    }
    assert "broadcast_geometry" not in broadcast.metadata
    left_transfer = next(item for item in broadcast.transfers if item.value == "left")
    right_transfer = next(item for item in broadcast.transfers if item.value == "right")
    assert [step.kind for step in left_transfer.route] == [
        "dma_l4_l3",
        "lookup",
    ]
    assert [step.kind for step in right_transfer.route] == [
        "dma_l4_l1_32k",
        "load_vr",
        "duplicate_subgroup",
    ]
    assert left_transfer.route[-1].parameters["table_size"] == 32
    assert right_transfer.route[-1].parameters["rows_per_vr"] == 4
    assert len([item for item in broadcast.transfers if item.value == "result"]) == 2
    assert generate_apu_v1_plans(analysis) == tuple(
        candidate.plan
        for candidate in generate_apu_v1_vectorization_candidates(analysis)
    )


def test_non_power_of_two_micro_shape_is_validity_tiled_within_32k_lanes():
    analysis = analyze_apu_v1_contraction(_module(non_power_of_two_micro))
    plans = [
        candidate.plan
        for candidate in generate_apu_v1_vectorization_candidates(analysis)
    ]
    baseline, temporal, _, broadcast = plans

    assert baseline.metadata["problem_shape"] == {"M": 60, "N": 70, "K": 80}
    assert baseline.iteration_layout.layout.out_sizes[0] <= 32768
    assert baseline.reduction_strategy.kind == "group_tree"
    assert not baseline.reduction_strategy.partial
    assert baseline.metadata["output_tiles"] == 1
    assert baseline.metadata["physical_output_batches"] == 1
    assert baseline.metadata["work_output_tiles"] == 30
    assert baseline.metadata["compute_tiles_per_output"] == 30
    assert baseline.metadata["vr_batches"] == 30
    assert baseline.metadata["vector_compute_calls"] == 30
    assert baseline.metadata["group_reduce_calls"] == 30
    assert [operation.count for operation in baseline.operations] == [30, 30]
    reduction_low = baseline.iteration_layout.layout.coordinate(m=0, n=0, k=0)
    reduction_high = baseline.iteration_layout.layout.coordinate(m=0, n=0, k=32)
    assert reduction_low["vr_lane"] != reduction_high["vr_lane"]
    assert reduction_low["vr_batch"] == reduction_high["vr_batch"]
    assert temporal.iteration_layout.layout.input_extents == {
        "m": 60,
        "n": 70,
        "k": 80,
    }
    assert temporal.iteration_layout.layout.out_sizes == (8192, 128)
    assert temporal.temporal_extent == 80
    assert temporal.metadata["temporal_trip_count"] == 80
    assert temporal.metadata["output_tiles"] == 1
    assert temporal.metadata["work_output_tiles"] == 1
    assert temporal.metadata["vr_batches"] == 80
    assert temporal.metadata["vector_compute_calls"] == 80
    left_route = next(
        item.route for item in broadcast.transfers if item.value == "left"
    )
    right_route = next(
        item.route for item in broadcast.transfers if item.value == "right"
    )
    assert left_route[-1].parameters["table_size"] == 64
    assert right_route[-1].parameters["group_size"] == 8 * 128
    assert right_route[-1].parameters["subgroup_size"] == 128


def test_full_micro_problem_uses_lane_low_bits_and_distinct_vr_batches():
    analysis = analyze_apu_v1_contraction(_module(full_problem_micro))
    plans = generate_apu_v1_plans(analysis)
    optimized = plans[-1]
    assert "broadcast_geometry" not in optimized.metadata
    left_transfer = next(item for item in optimized.transfers if item.value == "left")
    right_transfer = next(item for item in optimized.transfers if item.value == "right")
    assert [step.kind for step in left_transfer.route] == [
        "dma_l4_l3",
        "lookup",
    ]
    assert [step.kind for step in right_transfer.route] == [
        "dma_l4_l1_32k",
        "load_vr",
        "duplicate_subgroup",
    ]
    lhs_dma, lhs_lookup = left_transfer.route
    rhs_dma, rhs_load, rhs_duplicate = right_transfer.route
    assert lhs_dma.metrics().call_count == 1
    assert lhs_lookup.metrics().call_count == 32 * 64
    assert lhs_lookup.metrics().source_elements_per_call == 32
    assert lhs_lookup.metrics().destination_elements_per_call == 32768
    assert rhs_dma.metrics().call_count == 8
    assert rhs_dma.metrics().resident_reuse_factor == 32
    assert rhs_load.metrics().expansion_factor == 4
    assert rhs_duplicate.metrics().call_count == 32 * 64
    assert rhs_duplicate.metrics().source_elements_per_call == 4 * 1024
    assert rhs_duplicate.metrics().destination_elements_per_call == 32768
    assert rhs_duplicate.metrics().expansion_factor == 8
    assert rhs_duplicate.parameters == {
        "rows_per_vr": 8,
        "group_size": 8192,
        "subgroup_size": 1024,
        "resident_vr_extent": 32768,
        "active_vr_extent": 32768,
        "replication_factor": 4,
    }
    assert optimized.metadata["output_tiles"] == 32
    for plan in plans:
        batching = plan.output_batching
        assert batching.physical_output_batches == 32
        assert plan.metadata["physical_output_batches"] == 32
        assert plan.value_layout("result").layout.out_sizes[1] == 32
        persistence = plan.metadata["accumulator_persistence"]
        assert persistence["reduction_tiles"] == plan.output_batching.reduction_tiles
    assert plans[0].metadata["accumulator_persistence"] == {
        "scope": "work_tile",
        "reduction_tiles": 1,
        "commit": "scatter_to_physical_output",
    }
    for temporal_plan in plans[1:]:
        assert temporal_plan.metadata["accumulator_persistence"] == {
            "scope": "physical_output_batch",
            "reduction_tiles": 64,
            "commit": "egress_after_work_steps",
        }
    baseline_batching = plans[0].output_batching
    assert baseline_batching.work_output_tiles == 2048
    assert baseline_batching.work_tiles_per_output_batch == 64
    assert baseline_batching.reduction_tiles == 1
    assert baseline_batching.work_steps_per_output_batch == 64
    assert baseline_batching.placement(0).physical_output_batch == 0
    assert baseline_batching.placement(0).lane_offset == 0
    assert baseline_batching.placement(2047).physical_output_batch == 31
    assert baseline_batching.placement(2047).lane_offset == 32256
    for temporal_plan in plans[1:]:
        batching = temporal_plan.output_batching
        assert batching.work_output_tiles == 32
        assert batching.work_tiles_per_output_batch == 1
        assert batching.reduction_tiles == 64
        assert batching.work_steps_per_output_batch == 64
    output_transfers = [item for item in optimized.transfers if item.value == "result"]
    assert [item.direction for item in output_transfers] == ["in", "out"]
    assert all(item.route[0].metrics().call_count == 32 for item in output_transfers)

    for temporal_plan in plans[1:]:
        first_slice = temporal_plan.iteration_layout.layout.coordinate(m=0, n=0, k=0)
        next_slice = temporal_plan.iteration_layout.layout.coordinate(m=0, n=0, k=1)
        assert first_slice["vr_lane"] == next_slice["vr_lane"]
        assert first_slice["vr_batch"] != next_slice["vr_batch"]

    for plan in plans:
        left = plan.value_layout("left")
        right = plan.value_layout("right")
        output = plan.value_layout("result")
        assert left.layout.input_extents == {"m": 1024, "n": 1024, "k": 64}
        assert right.layout.input_extents == {"m": 1024, "n": 1024, "k": 64}
        assert output.layout.input_extents == {"m": 1024, "n": 1024}
        assert left.layout.out_dims == ("vr_lane", "vr_batch")
        assert right.layout.out_dims == ("vr_lane", "vr_batch")
        assert output.layout.out_dims == ("vr_lane", "vr_batch")
        assert output.layout.out_sizes == (32768, 32)

        for layout, low, high in (
            (
                left.layout,
                {"m": 0, "k": 0, "n": 0},
                {"m": 1023, "k": 63, "n": 1023},
            ),
            (
                right.layout,
                {"k": 0, "n": 0, "m": 0},
                {"k": 63, "n": 1023, "m": 1023},
            ),
            (
                output.layout,
                {"m": 0, "n": 0},
                {"m": 1023, "n": 1023},
            ),
        ):
            for coordinate in (layout.coordinate(**low), layout.coordinate(**high)):
                assert 0 <= coordinate["vr_lane"] < layout.out_sizes[0]
                assert 0 <= coordinate["vr_batch"] < layout.out_sizes[1]

        first = output.layout.coordinate(m=0, n=0)
        next_batch = None
        for axis in ("m", "n"):
            for value in (1 << bit for bit in range(10)):
                indices = {"m": 0, "n": 0}
                indices[axis] = value
                candidate = output.layout.coordinate(**indices)
                if (
                    candidate["vr_lane"] == first["vr_lane"]
                    and candidate["vr_batch"] != first["vr_batch"]
                ):
                    next_batch = candidate
                    break
            if next_batch is not None:
                break
        assert next_batch is not None
        # Full-domain F2 rank proves every logical output has a distinct packed
        # coordinate without enumerating one million points in the unit test.
        assert output.layout.carrier.image_size(varying_inputs=("m", "n")) == (
            1024 * 1024
        )
        assert output.layout.out_sizes[0] * output.layout.out_sizes[1] == (1024 * 1024)
