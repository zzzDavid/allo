# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural coverage tests for the configurable CENT/AiM workloads."""

from dataclasses import replace

import pytest

import allo
from allo.pim.aim_program import (
    AimAllBankWrite,
    AimContraction,
    AimDistributedHostTransfer,
    AimHostTransfer,
)
from allo.pim.targets import build_aim_target
from benchmarks.cent_aim.evidence import compiler_logical_work_coverage
from benchmarks.cent_aim.vendor_contract import VENDOR_TRACE_CONTRACTS
from benchmarks.cent_aim.workloads import (
    CENT_CASES,
    DEFAULT_DECODE_SPEC,
    DecodeSpec,
    build_case,
    build_layout,
    compose_program,
    iter_case_programs,
    semantic_manifest,
)


EXPECTED_CASE_IDS = (
    "norm_residual_l128",
    "qkvo_rope_l128",
    "attention_l128",
    "attention_l512",
    "attention_l4096",
    "softmax_pim_l128",
    "softmax_pim_l512",
    "softmax_pim_l4096",
    "ffn_fc_l128",
    "ffn_activation_l128",
    "ffn_complete_l128",
    "full_block_l128",
    "full_block_l512",
    "full_block_l4096",
)


@pytest.mark.parametrize("case,spec,program", list(iter_case_programs()))
def test_tenon_program_preserves_vendor_inventory_except_complete_cache_append(
    case, spec, program
):
    compiled = allo.compile(program, build_aim_target())
    manifest = semantic_manifest(case.case_id, spec, compiled)
    expected = dict(VENDOR_TRACE_CONTRACTS[case.case_id].opcode_counts)

    observed = manifest["compiled_trace"]["opcode_counts"]
    append_operations = [
        operation
        for operation in program.operations
        if isinstance(operation, AimAllBankWrite)
    ]
    if append_operations:
        expected_tenon_wr_abk = sum(
            len(operation.channels) * operation.rows * operation.copies
            for operation in append_operations
        )
        assert observed["AiM_WR_ABK"] == expected_tenon_wr_abk
        assert expected["AiM_WR_ABK"] < expected_tenon_wr_abk
        assert {
            opcode: count
            for opcode, count in observed.items()
            if opcode != "AiM_WR_ABK"
        } == {
            opcode: count
            for opcode, count in expected.items()
            if opcode != "AiM_WR_ABK"
        }
    else:
        assert observed == expected
    assert compiled.commands[-1] == "AiM EOC"
    assert compiled.commands.count("AiM EOC") == 1
    assert len(compiled.commands) == sum(observed.values())
    assert compiled.compiled.runtime_segments[0][1] == 1

    # CENT multiplexes four independent blocks over disjoint 8-channel slices.
    assert manifest["topology"]["replica_channel_groups"] == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert manifest["topology"]["replica_channel_ownership_is_disjoint"] is True
    assert "replica_values_not_modeled" in manifest["scope_caveats"]


def test_case_table_is_exactly_the_fourteen_archived_cases():
    assert tuple(CENT_CASES) == EXPECTED_CASE_IDS
    assert tuple(VENDOR_TRACE_CONTRACTS) == EXPECTED_CASE_IDS


@pytest.mark.parametrize("case,spec,program", list(iter_case_programs()))
def test_static_weights_are_resident_and_dynamic_traffic_is_explicit(
    case, spec, program
):
    manifest = semantic_manifest(case.case_id, spec)
    assert manifest["weights"]["placement"] == (
        "statically resident before measured region"
    )
    assert manifest["weights"]["host_transfers_included"] is False
    assert set(manifest["weights"]["regions"]) == {
        "wq",
        "wk",
        "wv",
        "wo",
        "w1",
        "w3",
        "w2",
    }
    assert manifest["dynamic_data"]["host_transfers_included"] is True

    transfers = tuple(
        operation
        for operation in program.operations
        if isinstance(
            operation,
            (AimHostTransfer, AimDistributedHostTransfer),
        )
    )
    assert all("weight" not in (operation.name or "") for operation in transfers)
    expected = VENDOR_TRACE_CONTRACTS[case.case_id].opcode_counts
    if expected.get("W_MEM", 0) + expected.get("R_MEM", 0):
        assert transfers
    else:
        assert not transfers


def test_full_block_preserves_transformer_program_order():
    names = tuple(
        operation.name
        for operation in build_case("full_block_l128").operations
        if operation.name
    )

    def position(prefix: str) -> int:
        return next(
            index for index, name in enumerate(names) if name.startswith(prefix)
        )

    assert position("resident_weight.wq") < position("rope.q")
    assert position("rope.q") < position("attention.k_append")
    assert position("attention.k_append") < position("attention.qk")
    assert position("attention.qk") < position("softmax.scale")
    assert position("softmax.normalize_exp") < position("attention.sv")
    assert position("attention.sv") < position("resident_weight.wo")
    assert position("resident_weight.w3") < position("ffn_activation")
    assert position("ffn_activation") < position("resident_weight.w2")


def test_shared_global_buffer_relocations_are_pairwise_ordered():
    compiled = allo.compile(
        build_case("ffn_activation_l128"),
        build_aim_target(),
    )
    copies = tuple(
        command.split()[1]
        for command in compiled.commands
        if command.startswith("AiM COPY_")
    )
    assert copies == ("COPY_BKGB", "COPY_GBBK") * 4


def test_noncanonical_shapes_use_the_same_stage_builders_and_layout_rules():
    spec = DecodeSpec(
        D=1024,
        H=8,
        Dh=128,
        F=2816,
        L=257,
        replicas=2,
        channels_per_replica=4,
        max_seq_len=512,
    )
    program = compose_program(
        CENT_CASES["full_block_l128"].stages,
        spec,
        name="shape_perturbation",
    )
    compiled = allo.compile(program, build_aim_target())

    contractions = {
        operation.name: operation
        for operation in program.operations
        if isinstance(operation, AimContraction)
    }
    assert contractions["resident_weight.wq"].outputs == 1024
    assert contractions["resident_weight.wq"].reduction == 1024
    assert contractions["resident_weight.w1"].outputs == 2816
    assert contractions["resident_weight.w2"].reduction == 2816
    assert contractions["attention.qk"].batch_mapping == "auto"
    assert contractions["attention.qk"].outputs == spec.L
    assert contractions["attention.sv"].batch_mapping == "auto"
    assert contractions["attention.sv"].reduction_storage_extent == spec.max_seq_len
    lowered = {
        operation["name"]: operation
        for operation in compiled.manifest["operations"]
        if operation["kind"] == "contraction"
    }
    assert lowered["attention.qk"]["batch_mapping"] == "row_packed"
    assert lowered["attention.sv"]["batch_mapping"] == "channels"
    assert (
        lowered["attention.qk"]["batch_mapping_selection"]["uses_operation_name"]
        is False
    )
    assert (
        lowered["attention.sv"]["batch_mapping_selection"]["uses_operation_name"]
        is False
    )
    assert compiled.commands[-1] == "AiM EOC"
    assert compiled.commands.count("AiM EOC") == 1


def test_layout_regions_are_nonoverlapping_and_shape_derived():
    small = build_layout(DecodeSpec(1024, 8, 128, 2816, 128, 2, 4, 512))
    large = build_layout(replace(DEFAULT_DECODE_SPEC, L=4096))

    for layout in (small, large):
        regions = sorted(layout.regions.values(), key=lambda region: region.row)
        assert all(
            left.row + left.rows <= right.row
            for left, right in zip(regions, regions[1:])
        )
        assert regions[-1].row + regions[-1].rows == layout.used_rows
        assert layout.used_rows <= 16384
    assert large["k_cache"].rows > small["k_cache"].rows
    assert large["w1"].rows > small["w1"].rows


@pytest.mark.parametrize(
    "spec",
    [
        DEFAULT_DECODE_SPEC,
        DecodeSpec(1024, 8, 128, 2816, 257, 2, 4, 512),
        DecodeSpec(1152, 8, 144, 2816, 257, 2, 4, 512),
        DecodeSpec(1152, 8, 144, 2816, 81, 2, 4, 512),
    ],
)
def test_attention_cache_producers_cover_packed_consumers(spec):
    program = compose_program(
        ("attention_cache_append", "attention_qk", "attention_sv"),
        spec,
        name="cache_ownership",
    )
    compiled = allo.compile(program, build_aim_target())
    layout = build_layout(spec)
    contractions = {
        operation.name: operation
        for operation in program.operations
        if isinstance(operation, AimContraction)
    }
    qk = contractions["attention.qk"]
    sv = contractions["attention.sv"]
    aligned_reduction = (spec.Dh + 15) // 16 * 16
    qk_batch_pack = min(spec.H, 1024 // aligned_reduction)
    qk_batch_groups = (spec.H + qk_batch_pack - 1) // qk_batch_pack
    sv_batch_groups = (spec.H + spec.channels_per_replica - 1) // (
        spec.channels_per_replica
    )
    qk_tiles = (spec.max_seq_len + spec.channels_per_replica * 16 - 1) // (
        spec.channels_per_replica * 16
    )
    sv_tiles = (spec.Dh + 15) // 16
    storage_rows = (spec.max_seq_len + 1023) // 1024

    assert qk.batch_mapping == "auto"
    assert layout["k_cache"].rows == qk_batch_groups * qk_tiles
    assert sv.batch_mapping == "auto"
    assert sv.reduction_storage_extent == spec.max_seq_len
    assert layout["v_cache"].rows == sv_batch_groups * sv_tiles * storage_rows
    lowered = {
        operation["name"]: operation
        for operation in compiled.manifest["operations"]
        if operation["kind"] == "contraction"
    }
    for name, selected in (
        ("attention.qk", "row_packed"),
        ("attention.sv", "channels"),
    ):
        operation = lowered[name]
        assert operation["requested_batch_mapping"] == "auto"
        assert operation["batch_mapping"] == selected
        assert operation["batch_mapping_selection"] == {
            "policy": "shape_and_residency_v1",
            "selected": selected,
            "reason": operation["batch_mapping_selection"]["reason"],
            "uses_operation_name": False,
        }
        assert operation["batch_mapping_selection"]["reason"]
    qk_lowered = lowered["attention.qk"]
    assert qk_lowered["batch_pack"] == qk_batch_pack
    assert qk_lowered["batch_groups"] == qk_batch_groups
    assert lowered["attention.sv"]["batch_pack"] == spec.channels_per_replica
    assert lowered["attention.sv"]["batch_groups"] == sv_batch_groups

    position = spec.L - 1
    position_tile, position_in_tile = divmod(position, spec.channels_per_replica * 16)
    local_channel, bank = divmod(position_in_tile, 16)
    consumer_k_coordinates = set()
    for batch_group in range(qk_batch_groups):
        packed_heads = min(
            qk_batch_pack,
            spec.H - batch_group * qk_batch_pack,
        )
        consumer_row = qk_lowered["input_groups"][batch_group]["matrix_rows"][
            position_tile
        ]
        assert consumer_row == (
            layout["k_cache"].row + position_tile * qk_batch_groups + batch_group
        )
        for replica in range(spec.replicas):
            physical_channel = qk_lowered["replica_channel_groups"][replica][
                local_channel
            ]
            assert (
                physical_channel
                in qk_lowered["output_tile_layouts"][position_tile]["active_channels"]
            )
            consumer_k_coordinates.add(
                (
                    physical_channel,
                    bank,
                    consumer_row,
                    packed_heads * aligned_reduction // 16,
                )
            )
    actual_k_writes = {
        (operation.channel, operation.bank, operation.row, operation.bursts)
        for operation in program.operations
        if isinstance(operation, AimHostTransfer)
    }
    assert actual_k_writes == consumer_k_coordinates

    v_append = next(
        operation
        for operation in program.operations
        if isinstance(operation, AimAllBankWrite)
    )
    assert v_append.channels == spec.channels
    coverage = compiler_logical_work_coverage(compiled.manifest, compiled.trace)
    ownership = [
        operation["ownership_coverage"]
        for operation in coverage["operations"]
        if operation["ownership_coverage"] is not None
    ]
    assert coverage["complete"] is True
    assert len(ownership) == 1
    assert ownership[0]["complete"] is True
    assert ownership[0]["expected_coordinate_count"] == (
        spec.replicas * spec.channels_per_replica * sv_batch_groups * sv_tiles
    )


def test_qk_row_pack_fails_closed_when_a_head_exceeds_one_row():
    spec = DecodeSpec(4160, 4, 1040, 2816, 1, 1, 4, 1)
    with pytest.raises(ValueError, match="aligned head reduction to fit one row"):
        build_layout(spec)
