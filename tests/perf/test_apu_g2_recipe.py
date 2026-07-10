# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural gates for generic Gemini-II physical recipes."""

from collections import Counter
from dataclasses import replace

import pytest

from allo.pim.apu_g2_recipe import (
    APUG2Descriptor,
    APUG2DescriptorUse,
    APUG2Recipe,
    APUG2RecipeCalibration,
    APUG2RecipeOpKind,
    APUG2RecipeOperation,
    APUG2VectorizationCertificate,
    build_apu_g2_recipe_graph,
    build_apu_g2_u16_block_sum_recipe,
    build_apu_g2_u16_contraction_recipe,
    build_apu_g2_u16_div_recipe,
    build_apu_g2_u16_dot_tile_recipe,
    build_apu_g2_u16_fill_recipe,
    build_apu_g2_u16_minmax_recipe,
    build_apu_g2_u16_mul_recipe,
    build_apu_g2_u16_select_lt_recipe,
    build_apu_g2_u16_shift_right_recipe,
    build_apu_g2_u16_sub_recipe,
    chain_apu_g2_recipes,
    calibrate_apu_g2_recipe_from_measured_tile,
)
from allo.pim.targets import build_apu_g2_target


def test_descriptor_rejects_mmb_segment_crossing_and_l1_segments():
    with pytest.raises(ValueError, match="crosses"):
        APUG2Descriptor("mmb", 16, start_row=16, segment="seg0")
    with pytest.raises(ValueError, match="start in rows"):
        APUG2Descriptor("mmb", 8, start_row=0, segment="seg1")
    with pytest.raises(ValueError, match="do not have"):
        APUG2Descriptor("l1", 8, segment="seg0")


def test_recipe_rejects_forward_dependencies_and_false_certificate():
    first = APUG2RecipeOperation(
        "first",
        "ADD_U16",
        APUG2RecipeOpKind.VL64,
        dependencies=("later",),
        metrics={"vector_lane_updates": 16},
    )
    later = APUG2RecipeOperation(
        "later",
        "SEU_BARRIER",
        APUG2RecipeOpKind.BARRIER,
    )
    certificate = APUG2VectorizationCertificate(16, 0, 0)
    with pytest.raises(ValueError, match="forward dependencies"):
        APUG2Recipe("bad_dependency", (first, later), certificate)

    operation = APUG2RecipeOperation(
        "only",
        "ADD_U16",
        APUG2RecipeOpKind.VL64,
        metrics={"vector_lane_updates": 8},
    )
    with pytest.raises(ValueError, match="lane updates"):
        APUG2Recipe("bad_certificate", (operation,), certificate)


def test_single_dot_tile_matches_hardware_proven_three_byte_inventory():
    recipe = build_apu_g2_u16_dot_tile_recipe(90, 90)

    assert recipe.metadata["dot_tile_count"] == 1
    assert recipe.metadata["log_block_size"] == 7
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 9,
        "MUL_U8_TO_U16": 3,
        "COPY_MMB_TO_L1_VECTORS": 6,
        "GROUP_REDUCE_ADD_U16_TO_U23": 3,
        "SHIFT_LEFT_U16": 2,
        "ADD_U16": 2,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.scalar_control_ops == 0
    assert recipe.certificate.fully_vectorized
    assert recipe.certificate.vector_lane_updates > 0

    reductions = [
        operation
        for operation in recipe.operations
        if operation.opcode == "GROUP_REDUCE_ADD_U16_TO_U23"
    ]
    assert len(reductions) == 3
    for operation in reductions:
        assert operation.metrics["log_block_size"] == 7
        assert operation.metrics["dst_bits"] == 23
        destination = next(
            item.descriptor for item in operation.descriptors if item.role == "dst"
        )
        assert destination.segment == "seg1"
        assert destination.start_row == 24
        assert destination.num_bits == 23

    bounces = [
        operation.attributes.get("segment_safe_bounce")
        for operation in recipe.operations
        if "segment_safe_bounce" in operation.attributes
    ]
    assert bounces == ["out", "in"] * 3

    operand_rows = [
        next(
            item.descriptor.start_row
            for item in operation.descriptors
            if item.role == "src"
        )
        for operation in recipe.operations
        if operation.opcode == "COPY_L1_VECTORS_TO_MMB"
        and "byte_product" in operation.attributes
        and "segment_safe_bounce" not in operation.attributes
    ]
    assert operand_rows == [0, 32, 0, 40, 8, 32]


def test_dot_builder_temporally_tiles_outputs_across_all_four_sets():
    recipe = build_apu_g2_u16_dot_tile_recipe(2500, 180)

    assert recipe.metadata["tile_capacity"] == 1024
    assert recipe.metadata["output_tile_count"] == 3
    assert recipe.metadata["dot_tile_count"] == 3
    assert recipe.metadata["physical_sets"] == 4
    inventory = recipe.inventory
    assert inventory["MUL_U8_TO_U16"] == 9
    assert inventory["GROUP_REDUCE_ADD_U16_TO_U23"] == 9
    assert inventory["SHIFT_LEFT_U16"] == 6
    assert inventory["ADD_U16"] == 6
    assert inventory["SEU_BARRIER"] == 1

    final_tile_operations = [
        operation
        for operation in recipe.operations
        if operation.attributes.get("output_tile") == 2
    ]
    assert final_tile_operations
    assert all(
        operation.attributes["active_outputs"] == 452
        for operation in final_tile_operations
    )
    assert all(
        operation.attributes["active_reduction"] == 180
        for operation in final_tile_operations
    )
    assert recipe.certificate.scalar_tensor_updates == 0


def test_recipe_graph_has_identical_inventory_dependencies_and_descriptors():
    target = build_apu_g2_target()
    recipe = build_apu_g2_u16_dot_tile_recipe(90, 90)
    graph = build_apu_g2_recipe_graph(recipe, target)

    graph_inventory = Counter(activity.label.upper() for activity in graph.activities)
    assert graph_inventory == Counter(recipe.inventory)
    assert graph.metadata["inventory"] == dict(recipe.inventory)
    assert graph.metadata["latency_unit"] == "structural_issue_call"
    assert graph.metadata["calibrated_device_ticks"] is False
    assert graph.metadata["vectorization_certificate"] == recipe.certificate.manifest()
    assert len(graph.activities) == len(recipe.operations)

    for operation, activity in zip(recipe.operations, graph.activities):
        assert activity.id == operation.id
        assert activity.depends_on == operation.dependencies
        assert activity.metadata["metrics"] == dict(operation.metrics)
        assert activity.metadata["descriptors"] == [
            descriptor.manifest() for descriptor in operation.descriptors
        ]
        assert not any("/pe" in item.handle.path for item in activity.occupancy)

    assert graph.topological_order() == graph.activities


def test_transfer_transform_and_barrier_are_first_class_recipe_calls():
    l1 = APUG2Descriptor("l1", 16, start_row=0)
    mmb = APUG2Descriptor("mmb", 16, start_row=0, segment="seg0")
    operations = (
        APUG2RecipeOperation(
            "load",
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            descriptors=(
                APUG2DescriptorUse("src", l1),
                APUG2DescriptorUse("dst", mmb),
            ),
        ),
        APUG2RecipeOperation(
            "spread",
            "SPREAD_BLOCK",
            APUG2RecipeOpKind.TRANSFORM,
            dependencies=("load",),
            descriptors=(APUG2DescriptorUse("src_dst", l1),),
            metrics={"log_block_size": 7},
        ),
        APUG2RecipeOperation(
            "done",
            "SEU_BARRIER",
            APUG2RecipeOpKind.BARRIER,
            dependencies=("spread",),
        ),
    )
    recipe = APUG2Recipe(
        "transfer_transform",
        operations,
        APUG2VectorizationCertificate(0, 0, 0),
    )
    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())

    assert [activity.metadata["kind"] for activity in graph.activities] == [
        "transfer",
        "transform",
        "barrier",
    ]
    assert [activity.depends_on for activity in graph.activities] == [
        (),
        ("load",),
        ("spread",),
    ]


def test_select_lt_recipe_has_compare_marker_and_masked_copy():
    recipe = build_apu_g2_u16_select_lt_recipe()

    assert recipe.metadata["operation"] == "select_lt"
    assert recipe.metadata["predicate"] == "lhs < rhs"
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 4,
        "LT_U16": 1,
        "COPY_MMB_TO_L1_VECTORS": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    compare = [
        operation for operation in recipe.operations if operation.opcode == "LT_U16"
    ]
    assert len(compare) == 1
    marker = next(
        item.descriptor for item in compare[0].descriptors if item.role == "dst"
    )
    assert marker.value_type == "marker"
    assert marker.segment == "seg0"
    assert marker.start_row == 23

    masked = [
        operation
        for operation in recipe.operations
        if operation.attributes.get("masked") is True
    ]
    assert len(masked) == 1
    assert {item.role for item in masked[0].descriptors} == {"src", "dst", "mask"}

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert "lt_u16" in [activity.label for activity in graph.activities]


def test_minmax_recipe_uses_inplace_unsigned_mmb_primitives():
    recipe = build_apu_g2_u16_minmax_recipe()

    assert recipe.metadata["operation"] == "minmax"
    assert recipe.metadata["outputs"] == ("min", "max")
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 3,
        "MIN_U16": 1,
        "COPY_MMB_TO_L1_VECTORS": 2,
        "MAX_U16": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    for opcode in ("MIN_U16", "MAX_U16"):
        operation = next(item for item in recipe.operations if item.opcode == opcode)
        src_dst = next(
            item.descriptor for item in operation.descriptors if item.role == "src_dst"
        )
        src = next(
            item.descriptor for item in operation.descriptors if item.role == "src"
        )
        assert src_dst.segment == "seg1"
        assert src_dst.start_row == 24
        assert src.segment == "seg0"
        assert src.start_row == 0

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    labels = [activity.label for activity in graph.activities]
    assert "min_u16" in labels
    assert "max_u16" in labels


def test_div_recipe_rejects_zero_by_contract_and_writes_l1_result():
    recipe = build_apu_g2_u16_div_recipe()

    assert recipe.metadata["operation"] == "div"
    assert recipe.metadata["division_by_zero"] == "rejected"
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 2,
        "DIV_U16": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    div_op = next(
        operation for operation in recipe.operations if operation.opcode == "DIV_U16"
    )
    roles = {item.role: item.descriptor for item in div_op.descriptors}
    assert roles["dividend"].segment == "seg0"
    assert roles["dividend"].start_row == 0
    assert roles["divisor"].segment == "seg1"
    assert roles["divisor"].start_row == 24
    assert roles["dst"].storage == "l1"

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert "div_u16" in [activity.label for activity in graph.activities]


def test_sub_recipe_uses_inplace_unsigned_mmb_primitive():
    recipe = build_apu_g2_u16_sub_recipe()

    assert recipe.metadata["operation"] == "sub"
    assert recipe.metadata["arithmetic"] == "unsigned_subtract_mod_2_16"
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 2,
        "SUB_U16": 1,
        "COPY_MMB_TO_L1_VECTORS": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    sub_op = next(
        operation for operation in recipe.operations if operation.opcode == "SUB_U16"
    )
    roles = {item.role: item.descriptor for item in sub_op.descriptors}
    assert roles["lhs"].segment == "seg0"
    assert roles["lhs"].start_row == 0
    assert roles["rhs_dst"].segment == "seg1"
    assert roles["rhs_dst"].start_row == 24

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert "sub_u16" in [activity.label for activity in graph.activities]


def test_mul_recipe_expands_to_three_byte_products_and_two_combines():
    recipe = build_apu_g2_u16_mul_recipe()

    assert recipe.metadata["operation"] == "mul"
    assert recipe.metadata["arithmetic"] == "unsigned_multiply_mod_2_16"
    assert recipe.metadata["byte_products"] == ("lo_lo", "lo_hi", "hi_lo")
    assert recipe.metadata["omitted_term"] == "hi_hi_is_zero_mod_2_16"
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 6,
        "MUL_U8_TO_U16": 3,
        "COPY_MMB_TO_L1_VECTORS": 3,
        "SHIFT_LEFT_U16": 2,
        "ADD_U16": 2,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    products = [
        operation
        for operation in recipe.operations
        if operation.opcode == "MUL_U8_TO_U16"
    ]
    assert [item.attributes["byte_product"] for item in products] == [
        "lo_lo",
        "lo_hi",
        "hi_lo",
    ]
    assert all(
        next(
            descriptor.descriptor
            for descriptor in operation.descriptors
            if descriptor.role == "dst"
        ).segment
        == "seg1"
        for operation in products
    )

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    labels = [activity.label for activity in graph.activities]
    assert labels.count("mul_u8_to_u16") == 3
    assert labels.count("shift_left_u16") == 2
    assert labels.count("add_u16") == 2


def test_fill_recipe_broadcasts_uint16_immediate_to_l1_pack():
    recipe = build_apu_g2_u16_fill_recipe(0xA55A)

    assert recipe.metadata["operation"] == "fill"
    assert recipe.metadata["arithmetic"] == "scalar_broadcast"
    assert recipe.metadata["value"] == 0xA55A
    assert dict(recipe.inventory) == {
        "COPY_IMMEDIATE_TO_MMB": 1,
        "COPY_MMB_TO_L1_VECTORS": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    immediate_copy = next(
        item for item in recipe.operations if item.opcode == "COPY_IMMEDIATE_TO_MMB"
    )
    roles = {item.role: item.descriptor for item in immediate_copy.descriptors}
    assert roles["src"].storage == "immediate"
    assert roles["src"].num_bits == 16
    assert roles["dst"].segment == "seg1"
    assert immediate_copy.metrics["immediate"] == 0xA55A

    with pytest.raises(ValueError, match="fit uint16"):
        build_apu_g2_u16_fill_recipe(65536)

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    labels = [activity.label for activity in graph.activities]
    assert labels == [
        "copy_immediate_to_mmb",
        "copy_mmb_to_l1_vectors",
        "seu_barrier",
    ]


def test_block_sum_recipe_widens_and_stores_block_first_results():
    recipe = build_apu_g2_u16_block_sum_recipe(8)

    assert recipe.metadata["operation"] == "block_sum"
    assert recipe.metadata["arithmetic"] == "unsigned_block_sum_low16"
    assert recipe.metadata["log_block_size"] == 8
    assert recipe.metadata["block_size"] == 256
    assert recipe.metadata["result_lane"] == "block_first"
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 1,
        "GROUP_REDUCE_ADD_U16_TO_U23": 1,
        "COPY_MMB_TO_L1_VECTORS": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    reduce_op = next(
        item
        for item in recipe.operations
        if item.opcode == "GROUP_REDUCE_ADD_U16_TO_U23"
    )
    roles = {item.role: item.descriptor for item in reduce_op.descriptors}
    assert roles["src"].segment == "seg0"
    assert roles["dst"].segment == "seg1"
    assert roles["dst"].num_bits == 24
    assert reduce_op.metrics["vector_lane_updates"] == (4 * 65536) // 256

    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        build_apu_g2_u16_block_sum_recipe(9)

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert "group_reduce_add_u16_to_u23" in [
        activity.label for activity in graph.activities
    ]


def test_shift_right_recipe_uses_inplace_unsigned_mmb_primitive():
    recipe = build_apu_g2_u16_shift_right_recipe(3)

    assert recipe.metadata["operation"] == "shift_right"
    assert recipe.metadata["arithmetic"] == "logical_right_shift"
    assert recipe.metadata["shift"] == 3
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 1,
        "SHIFT_RIGHT_U16": 1,
        "COPY_MMB_TO_L1_VECTORS": 1,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized

    shift_op = next(
        item for item in recipe.operations if item.opcode == "SHIFT_RIGHT_U16"
    )
    src_dst = next(
        item.descriptor for item in shift_op.descriptors if item.role == "src_dst"
    )
    assert src_dst.segment == "seg1"
    assert src_dst.start_row == 24
    assert shift_op.metrics["shift"] == 3

    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        build_apu_g2_u16_shift_right_recipe(16)

    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())
    assert graph.metadata["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert "shift_right_u16" in [activity.label for activity in graph.activities]


def test_scalar_control_is_modeled_but_never_counts_as_a_tensor_update():
    control = APUG2RecipeOperation(
        "tile_loop",
        "ARC_SCALAR_CONTROL",
        APUG2RecipeOpKind.SCALAR_CONTROL,
        metrics={"trip_count": 3},
        attributes={"purpose": "advance_output_tile"},
    )
    recipe = APUG2Recipe(
        "scalar_control",
        (control,),
        APUG2VectorizationCertificate(0, 1, 0),
    )
    graph = build_apu_g2_recipe_graph(recipe, build_apu_g2_target())

    assert recipe.certificate.scalar_control_ops == 1
    assert recipe.certificate.scalar_tensor_updates == 0
    assert graph.activities[0].primitive.endswith("/arc/DISPATCH")
    assert graph.activities[0].metadata["metrics"]["vl64_calls"] == 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"output_tile_extent": 4096}, "carrier"),
    ],
)
def test_dot_tile_builder_rejects_illegal_physical_shapes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_apu_g2_u16_dot_tile_recipe(90, 90, **kwargs)


def test_dot_tile_builder_rejects_reductions_wider_than_the_24_bit_sum():
    with pytest.raises(ValueError, match="padded reduction extent"):
        build_apu_g2_u16_dot_tile_recipe(90, 257)


def test_gemm_rank_two_recipe_tiles_outputs_and_emits_full_alpha_beta_epilogue():
    recipe = build_apu_g2_u16_contraction_recipe(
        (60, 70),
        80,
        alpha=1,
        beta=5,
        name="gemm_u16",
    )

    assert recipe.metadata["output_shape"] == (60, 70)
    assert recipe.metadata["output_extent"] == 4200
    assert recipe.metadata["reduction_extent"] == 80
    assert recipe.metadata["log_block_size"] == 7
    assert recipe.metadata["tile_capacity"] == 2048
    assert recipe.metadata["output_tile_count"] == 3
    assert recipe.metadata["epilogue"] == "alpha_dot_plus_beta_accumulator"
    assert recipe.metadata["alpha"] == 1
    assert recipe.metadata["beta"] == 5

    # Per output tile: raw three-byte dot plus two full u16 scalar multiplies
    # and one final alpha/beta combine.
    assert dict(recipe.inventory) == {
        "COPY_L1_VECTORS_TO_MMB": 48,
        "MUL_U8_TO_U16": 27,
        "COPY_MMB_TO_L1_VECTORS": 39,
        "GROUP_REDUCE_ADD_U16_TO_U23": 9,
        "SHIFT_LEFT_U16": 18,
        "ADD_U16": 21,
        "COPY_IMMEDIATE_TO_MMB": 18,
        "SEU_BARRIER": 1,
    }
    assert recipe.certificate.scalar_tensor_updates == 0
    assert recipe.certificate.fully_vectorized
    epilogue_ops = [
        operation
        for operation in recipe.operations
        if operation.attributes.get("phase") == "epilogue"
    ]
    assert epilogue_ops
    assert {operation.attributes.get("output_tile") for operation in epilogue_ops} == {
        0,
        1,
        2,
    }
    calibration = calibrate_apu_g2_recipe_from_measured_tile(recipe)
    assert calibration.total_cycles == 9_806 + 9_806 + 9_911
    assert calibration.measured_ticks_per_pipeline == pytest.approx(
        9_805.875 + 9_805.875 + 9_911.25
    )
    assert calibration.basis == "real_card_output_sensitive_tile_normalized_attribution"


def test_doitgen_rank_three_batches_flatten_into_temporal_dot_tiles():
    recipe = build_apu_g2_u16_contraction_recipe(
        (25, 20, 30),
        30,
        batch_rank=2,
        name="doitgen_u16",
    )

    assert recipe.metadata["output_rank"] == 3
    assert recipe.metadata["batch_shape"] == (25, 20)
    assert recipe.metadata["per_batch_output_shape"] == (30,)
    assert recipe.metadata["output_extent"] == 15_000
    assert recipe.metadata["tile_capacity"] == 8192
    assert recipe.metadata["output_tile_count"] == 2
    assert recipe.metadata["epilogue"] is None
    assert recipe.inventory["MUL_U8_TO_U16"] == 6
    assert recipe.inventory["GROUP_REDUCE_ADD_U16_TO_U23"] == 6
    assert recipe.inventory["ADD_U16"] == 4
    assert recipe.inventory["SEU_BARRIER"] == 1
    assert recipe.certificate.scalar_tensor_updates == 0


def test_output_sensitive_calibration_matches_2mm_3mm_tile_shapes():
    two_mm_stage2 = build_apu_g2_u16_contraction_recipe(
        (40, 80),
        50,
        alpha=5,
        beta=1,
        name="two_mm_stage2_u16",
    )
    three_mm_cd = build_apu_g2_u16_contraction_recipe(
        (50, 70),
        80,
        alpha=1,
        beta=1,
        name="three_mm_cd_u16",
    )
    doitgen = build_apu_g2_u16_contraction_recipe(
        (25, 20, 30),
        30,
        alpha=1,
        beta=1,
        batch_rank=2,
        name="doitgen_full_output_u16",
    )

    two_mm_calibration = calibrate_apu_g2_recipe_from_measured_tile(two_mm_stage2)
    three_mm_calibration = calibrate_apu_g2_recipe_from_measured_tile(three_mm_cd)
    doitgen_calibration = calibrate_apu_g2_recipe_from_measured_tile(doitgen)

    assert two_mm_calibration.total_cycles == 7_507
    assert two_mm_calibration.measured_ticks_per_pipeline == pytest.approx(7_506.625)
    assert three_mm_calibration.total_cycles == 9_806 + 9_846
    assert three_mm_calibration.measured_ticks_per_pipeline == pytest.approx(
        9_805.875 + 9_846.5
    )
    assert doitgen_calibration.total_cycles == 6_369 + 6_351
    assert doitgen_calibration.measured_ticks_per_pipeline == pytest.approx(
        6_369.75 + 6_351.25
    )
    assert {
        two_mm_calibration.basis,
        three_mm_calibration.basis,
        doitgen_calibration.basis,
    } == {"real_card_output_sensitive_tile_normalized_attribution"}


@pytest.mark.parametrize("stages", [2, 3])
def test_contraction_recipe_chains_have_explicit_2mm_3mm_dependencies(stages):
    shapes = ((60, 70), (60, 80), (60, 50))
    reductions = (80, 70, 80)
    recipes = tuple(
        build_apu_g2_u16_contraction_recipe(
            shapes[index],
            reductions[index],
            name=f"mm_stage_{index}",
        )
        for index in range(stages)
    )
    names = tuple(f"mm{index + 1}" for index in range(stages))
    chain = chain_apu_g2_recipes(recipes, name=f"chain_{stages}mm", stage_names=names)

    assert chain.metadata["stage_count"] == stages
    assert [item["name"] for item in chain.metadata["stages"]] == list(names)
    assert len(chain.operations) == sum(len(item.operations) for item in recipes)
    assert dict(chain.inventory) == dict(
        sum((Counter(item.inventory) for item in recipes), Counter())
    )
    assert chain.certificate.scalar_tensor_updates == 0

    by_stage = {
        stage: [
            operation
            for operation in chain.operations
            if operation.attributes["stage"] == stage
        ]
        for stage in names
    }
    for index in range(1, stages):
        assert by_stage[names[index]][0].dependencies == (
            by_stage[names[index - 1]][-1].id,
        )
    graph = build_apu_g2_recipe_graph(chain, build_apu_g2_target())
    assert graph.topological_order() == graph.activities


def test_calibrated_recipe_graph_preserves_measured_total_and_attribution():
    recipe = build_apu_g2_u16_dot_tile_recipe(1000, 180, name="dot_k180")
    calibration = APUG2RecipeCalibration.normalized(
        recipe,
        12_177,
        measured_ticks_per_pipeline=12_177.5,
        repetitions=8,
        opcode_weights={"GROUP_REDUCE_ADD_U16_TO_U23": 4, "MUL_U8_TO_U16": 2},
        basis="real_card_dot_tile_normalized_attribution",
    )
    graph = build_apu_g2_recipe_graph(
        recipe,
        build_apu_g2_target(),
        calibration=calibration,
    )

    assert sum(activity.latency_cycles for activity in graph.activities) == 12_177
    assert graph.metadata["calibrated_device_ticks"] is True
    assert graph.metadata["recipe_calibration"] == calibration.manifest()
    assert (
        graph.metadata["recipe_calibration"]["measured_ticks_per_pipeline"] == 12_177.5
    )
    with pytest.raises(ValueError, match="either latency or calibration"):
        build_apu_g2_recipe_graph(
            recipe,
            build_apu_g2_target(),
            latency=lambda _operation: 1,
            calibration=calibration,
        )


def test_recipe_calibration_is_alpha_rename_invariant_but_structure_sensitive():
    recipe = build_apu_g2_u16_dot_tile_recipe(1000, 180, name="original_recipe")
    calibration = APUG2RecipeCalibration.normalized(recipe, 12_177)
    renamed_ids = {
        operation.id: f"renamed_operation_{index}"
        for index, operation in enumerate(recipe.operations)
    }
    renamed_operations = tuple(
        APUG2RecipeOperation(
            renamed_ids[operation.id],
            operation.opcode,
            operation.kind,
            dependencies=tuple(renamed_ids[item] for item in operation.dependencies),
            descriptors=operation.descriptors,
            metrics=dict(operation.metrics),
            attributes={"diagnostic": f"unrelated_{index}"},
        )
        for index, operation in enumerate(recipe.operations)
    )
    renamed = APUG2Recipe(
        "renamed_recipe",
        renamed_operations,
        recipe.certificate,
        metadata={"diagnostic": "not_cost_visible"},
    )

    assert renamed.structural_fingerprint == recipe.structural_fingerprint
    calibration.validate_recipe(renamed)
    original_graph = build_apu_g2_recipe_graph(
        recipe, build_apu_g2_target(), calibration=calibration
    )
    renamed_graph = build_apu_g2_recipe_graph(
        renamed, build_apu_g2_target(), calibration=calibration
    )
    assert [item.latency_cycles for item in renamed_graph.activities] == [
        item.latency_cycles for item in original_graph.activities
    ]
    assert sum(item.latency_cycles for item in renamed_graph.activities) == 12_177

    changed_metrics = dict(renamed.operations[0].metrics)
    changed_metrics["groups"] = 8
    structurally_changed = APUG2Recipe(
        "changed_recipe",
        (replace(renamed.operations[0], metrics=changed_metrics),)
        + renamed.operations[1:],
        renamed.certificate,
    )
    assert structurally_changed.structural_fingerprint != recipe.structural_fingerprint
    with pytest.raises(ValueError, match="physical recipe fingerprint"):
        calibration.validate_recipe(structurally_changed)
