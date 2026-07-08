# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import allo
import numpy as np
import pytest
from allo.ir.types import float16, int16, uint16
from allo.pim.apu_v1_vector_program import APUv1VectorCallable
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def ordinary_contraction(
    left: float16[4, 8], right: float16[8, 6], result: float16[4, 6]
):
    for row, column in allo.grid(4, 6):
        for depth in allo.reduction(8):
            result[row, column] += left[row, depth] * right[depth, column]


def packed_similarity(
    left: int16[32, 8], right: int16[8, 1024], result: int16[32, 1024]
):
    for row, column in allo.grid(32, 1024):
        for depth in allo.reduction(8):
            result[row, column] += allo.popcount(
                ~(left[row, depth] ^ right[depth, column])
            )


def singleton_output_gemv(
    left: uint16[17, 19], right: uint16[19, 1], result: uint16[17, 1]
):
    for row, column in allo.grid(17, 1):
        for depth in allo.reduction(19):
            result[row, column] += left[row, depth] * right[depth, column]


def padded_outer_product(
    left: uint16[64, 1], right: uint16[1, 1024], result: uint16[64, 1024]
):
    for row, column in allo.grid(64, 1024):
        for depth in allo.reduction(1):
            result[row, column] += left[row, depth] * right[depth, column]


def wide_outer_product(
    left: uint16[32, 1], right: uint16[1, 2000], result: uint16[32, 2000]
):
    for row, column in allo.grid(32, 2000):
        for depth in allo.reduction(1):
            result[row, column] += left[row, depth] * right[depth, column]


def streamed_dense_tile(
    left: uint16[256, 128],
    right: uint16[128, 1024],
    result: uint16[256, 1024],
):
    for row, column in allo.grid(256, 1024):
        for depth in allo.reduction(128):
            result[row, column] += left[row, depth] * right[depth, column]


def polybench_gemm_edge_tile(
    left: uint16[256, 1200],
    right: uint16[1200, 76],
    result: uint16[256, 76],
):
    for row, column in allo.grid(256, 76):
        for depth in allo.reduction(1200):
            result[row, column] += left[row, depth] * right[depth, column]


def l3_over_capacity_dense_tile(
    left: uint16[256, 2000],
    right: uint16[2000, 1024],
    result: uint16[256, 1024],
):
    for row, column in allo.grid(256, 1024):
        for depth in allo.reduction(2000):
            result[row, column] += left[row, depth] * right[depth, column]


def test_public_compile_generates_ranks_and_executes_four_apu_plans():
    target = build_apu_v1_target()
    compiled = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="functional",
    )

    assert isinstance(compiled, APUv1VectorCallable)
    assert inspect.signature(compiled) == inspect.signature(ordinary_contraction)
    assert [candidate.name for candidate in compiled.candidates] == [
        "baseline_spatial_reduction",
        "temporal_svp",
        "temporal_dma_coalescing",
        "temporal_dma_coalescing_broadcast_friendly",
    ]
    assert {plan.name for plan in compiled.ranked_plans} == {
        candidate.name for candidate in compiled.candidates
    }
    assert compiled.selected_plan is compiled.ranked_plans[0]
    assert compiled.estimate().cycles > 0
    assert compiled.execution_graph.metadata["plan"] == compiled.selected_plan.name
    assert compiled.realization is not None, compiled.realization_error
    assert "gvml_" in compiled.device_source()

    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 8)).astype(np.float16)
    right = rng.normal(size=(8, 6)).astype(np.float16)
    result = np.zeros((4, 6), dtype=np.float16)
    reference = left @ right
    run = compiled(left, right, result)

    np.testing.assert_allclose(result, reference, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(run.extra["outputs"]["result"], result)
    assert run.cycles == compiled.estimate().cycles
    assert run.extra["plan"] == compiled.selected_plan.name


def test_public_compile_accepts_candidate_name_and_plan_object():
    target = build_apu_v1_target()
    named = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="virtual",
        layout="baseline_spatial_reduction",
    )
    assert named.selected_plan.name == "baseline_spatial_reduction"
    assert named.estimate("temporal_svp").cycles > 0

    explicit = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="functional",
        layout=named.plans[1],
    )
    assert explicit.selected_plan.name == "temporal_svp"


def test_public_compile_rejects_unknown_apu_vector_plan_name():
    with pytest.raises(ValueError, match="unknown APU v1 vector plan"):
        allo.compile(
            ordinary_contraction,
            build_apu_v1_target(),
            apu_v1_cost,
            backend="functional",
            layout="not_a_plan",
        )


def test_public_compile_consumes_target_neutral_xnor_popcount_mlir():
    compiled = allo.compile(
        packed_similarity,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )
    assert compiled.analysis.multiply_operation == "allo.xnor_popcount"
    assert compiled.analysis.packed_word_bits == 16
    assert compiled.realization_error is None
    source = compiled.device_source()
    assert "gvml_xor_16" in source
    assert "gvml_not_16" in source
    assert "gvml_popcount_16" in source
    assert "gvml_add_s16" in source


def test_native_spatial_gemv_realizes_for_singleton_output_axis():
    compiled = allo.compile(
        singleton_output_gemv,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        layout="spatial_gemv_group_reduction",
    )

    assert compiled.realization is not None, compiled.realization_error
    assert compiled.selected_estimate.plan.name == compiled.selected_plan.name
    assert compiled.selected_plan.name == "spatial_gemv_group_reduction"
    source = compiled.device_source()
    assert "gvml_mul_u16" in source
    assert "gvml_add_subgrps_u16_grp" in source
    assert "for (uint32_t output_tile" in source
    assert "for (uint32_t reduction_step" not in source
    assert source.index("right_L4ptr") < source.index("for (uint32_t output_tile")
    vector = np.arange(1, 20, dtype=np.uint16).reshape(19, 1)
    images = compiled.realization.abi.transfer_input_images(
        {
            "left": np.ones((17, 19), dtype=np.uint16),
            "right": vector,
            "result": np.zeros((17, 1), dtype=np.uint16),
        }
    )
    groups = images["right"].reshape(-1, 32)
    np.testing.assert_array_equal(groups[:, :19], np.tile(vector.T, (1024, 1)))
    assert np.count_nonzero(groups[:, 19:]) == 0


def test_lookup_ingress_repeats_table_slices_for_each_physical_output_batch():
    compiled = allo.compile(
        padded_outer_product,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )
    realization = compiled.realization
    assert realization.output_batches > 1
    arrays = {
        "left": np.ones((64, 1), dtype=np.uint16),
        "right": np.ones((1, 1024), dtype=np.uint16),
        "result": np.zeros((64, 1024), dtype=np.uint16),
    }
    images = realization.abi.transfer_input_images(arrays)
    transfer = next(
        item for item in compiled.selected_plan.transfers if item.value == "left"
    )
    lookup = next(step for step in transfer.route if step.kind == "lookup")
    table_size = lookup.parameters["table_size"]
    required = realization.output_batches * table_size

    assert images["left"].size >= required
    tables = images["left"][:required].reshape(realization.output_batches, table_size)
    assert np.all(np.count_nonzero(tables, axis=1) > 0)


def test_wide_outer_product_derives_sixteen_entry_lookup_geometry():
    compiled = allo.compile(
        wide_outer_product,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )

    assert compiled.realization is not None, compiled.realization_error
    transfer = next(
        item for item in compiled.selected_plan.transfers if item.value == "left"
    )
    lookup = next(step for step in transfer.route if step.kind == "lookup")
    assert lookup.parameters["group_size"] == 2048
    assert lookup.parameters["table_size"] == 16
    assert "gvml_lookup_16" in compiled.device_source()


def test_large_broadcast_route_streams_reduction_chunks_through_one_vr():
    compiled = allo.compile(
        streamed_dense_tile,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )

    assert compiled.selected_plan.name == (
        "temporal_dma_coalescing_broadcast_friendly_acc8"
    )
    source = compiled.device_source()
    assert "if (reduction_step % 8 == 0) direct_dma_l4_to_l1_32k" in source
    assert "(reduction_step / 8) * 32768" in source
    assert source.count("right_resident") == 3  # declaration, load, and duplicate
    assert "for (uint32_t output_block = 0;" in source
    assert "result__acc7" in source
    assert "left_L3ptr" in source
    # The streamed RHS is outside the eight statically unrolled accumulator
    # updates and is therefore loaded/expanded once per reduction step.
    reduction_loop = source.split("for (uint32_t reduction_step", 1)[1]
    first_accumulator = reduction_loop.index("const uint32_t batch")
    assert reduction_loop.index("direct_dma_l4_to_l1_32k") < first_accumulator
    assert (
        reduction_loop.index("gvml_duplicate_subgrp_16_grp_sgidx") < first_accumulator
    )
    bindings = compiled.realization.binding_map
    resident = next(
        binding.concrete
        for name, binding in bindings.items()
        if "right_resident" in name
    )
    assert resident != bindings["__result_product"].concrete


def test_non_power_of_two_gemm_edge_tile_has_a_realizable_plan():
    compiled = allo.compile(
        polybench_gemm_edge_tile,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )

    assert compiled.realization is not None, compiled.realization_error
    assert compiled.selected_plan.name == ("temporal_dma_coalescing_broadcast_friendly")


def test_large_lookup_plan_uses_l4_instead_of_exhausting_runtime_l3():
    compiled = allo.compile(
        l3_over_capacity_dense_tile,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )

    assert compiled.realization is not None, compiled.realization_error
    assert compiled.selected_plan.name == (
        "temporal_dma_coalescing_broadcast_friendly_acc8_l4"
    )
    assert "left_L4ptr" in compiled.device_source()
    assert "left_L3ptr" not in compiled.device_source()
