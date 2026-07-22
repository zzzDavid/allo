# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed SK hynix AiM whole-program lowering tests."""

import json

import pytest

import allo
from allo.pim import aim_program as aim_program_module
from allo.pim.aim_program import (
    AimActivation,
    AimAllBankWrite,
    AimBankCopy,
    AimContraction,
    AimDistributedHostTransfer,
    AimElementwise,
    AimHostTransfer,
    AimProgram,
    AimProgramCallable,
    AimSync,
    compile_aim_program,
)
from allo.pim.targets import build_aim_target
from allo.spmw_codegen import RunResult


def _compile(*operations, name="test"):
    return compile_aim_program(
        AimProgram(operations, name=name),
        build_aim_target(),
    )


def _physical_contraction_work(commands):
    """Reconstruct padded work from trace fields, independent of manifests."""
    mac_slots = 0
    wr_gb_elements = 0
    for command in commands:
        words = command.split()
        if command.startswith("AiM MAC_ABK"):
            mac_slots += int(words[2]) * 16 * int(words[3], 0).bit_count() * 16
        elif command.startswith("AiM WR_GB"):
            wr_gb_elements += int(words[2]) * 16 * int(words[4], 0).bit_count()
    return mac_slots, wr_gb_elements


def test_contraction_derives_geometry_row_splits_and_reuse_order():
    compiled = _compile(
        AimContraction(
            outputs=582,
            reduction=1030,
            row=11,
            channels=(0, 1, 2),
            input_gpr=4,
            reuse_group_size=2,
            reuse_group_candidates=(1, 2),
        )
    )

    commands = compiled.commands
    # 582 results / (3 channels * 16 banks) -> 13 physical launches.
    # K=1030 splits at the target's 1024-element row boundary.
    assert len(commands) == 81
    assert commands[0] == "AiM WR_GB 64 4 3758096384"
    assert commands[1:7] == (
        "AiM WR_BIAS 0 3758096384",
        "AiM WR_BIAS 0 3758096384",
        "AiM MAC_ABK 64 3758096384 11",
        "AiM MAC_ABK 64 3758096384 13",
        "AiM RD_MAC 0 3758096384",
        "AiM RD_MAC 0 3758096384",
    )
    assert commands[40] == "AiM WR_GB 1 68 3758096384"
    assert "AiM MAC_ABK 1 3758096384 36" in commands
    assert commands[-1] == "AiM EOC"
    assert commands.count("AiM EOC") == 1

    lowering = compiled.manifest["operations"][0]
    assert lowering["total_outputs"] == 582
    assert lowering["output_capacity_per_launch"] == 3 * 16
    assert lowering["output_launches"] == 13
    assert lowering["reduction_rows"] == 2
    assert lowering["row_chunks"] == [
        {
            "index": 0,
            "elements": 1024,
            "op_size": 64,
            "input_gpr_addresses": [4],
        },
        {
            "index": 1,
            "elements": 6,
            "op_size": 1,
            "input_gpr_addresses": [68],
        },
    ]
    assert lowering["reuse_group_candidates"] == [1, 2]
    assert lowering["reuse_group_size"] == 2
    assert lowering["runtime_repeat"] == 1


def test_flattened_batches_fail_closed_before_distinct_gb_payload_loss():
    with pytest.raises(ValueError, match="distinct batch GB payloads"):
        _compile(
            AimContraction(
                outputs=128,
                reduction=65,
                batches=32,
                row=4,
            )
        )
    # Even when both batches fit within the derived channels-per-replica,
    # one K-element WR_GB cannot encode two independent logical inputs.
    with pytest.raises(ValueError, match="distinct batch GB payloads"):
        _compile(
            AimContraction(
                outputs=128,
                reduction=65,
                batches=2,
                channels=tuple(range(8)),
                channels_per_replica=8,
            )
        )


def test_row_packed_batches_match_canonical_score_layout_and_scoped_reuse():
    compiled = _compile(
        AimContraction(
            outputs=512,
            reduction=128,
            batches=32,
            replicas=4,
            row=389,
            channels=tuple(range(32)),
            channels_per_replica=8,
            batch_mapping="auto",
        )
    )

    body = compiled.commands[:-1]
    wr_gb_indices = [
        index for index, command in enumerate(body) if command.startswith("AiM WR_GB")
    ]
    assert len(wr_gb_indices) == 4 * 4
    assert len(body) == 16 * (1 + 8 * 3)
    rows = []
    for scope_index, begin in enumerate(wr_gb_indices):
        end = (
            wr_gb_indices[scope_index + 1]
            if scope_index + 1 < len(wr_gb_indices)
            else len(body)
        )
        scope = body[begin:end]
        assert scope[0] == "AiM WR_GB 64 0 4294967295"
        assert [command.split()[1] for command in scope] == (
            ["WR_GB"] + ["WR_BIAS"] * 8 + ["MAC_ABK"] * 8 + ["RD_MAC"] * 8
        )
        scope_rows = [
            int(command.split()[-1])
            for command in scope
            if command.startswith("AiM MAC_ABK")
        ]
        assert len(set(scope_rows)) == 1
        rows.append(scope_rows[0])
    # Input-group-major traversal with row = base + tile * groups + group.
    assert rows == [
        389,
        393,
        397,
        401,
        390,
        394,
        398,
        402,
        391,
        395,
        399,
        403,
        392,
        396,
        400,
        404,
    ]

    lowering = compiled.manifest["operations"][0]
    assert lowering["batch_mapping"] == "row_packed"
    assert lowering["requested_batch_mapping"] == "auto"
    assert lowering["batch_mapping_selection"] == {
        "policy": "shape_and_residency_v1",
        "selected": "row_packed",
        "reason": (
            "independent GB batches fit complete aligned reductions within "
            "one bank row"
        ),
        "uses_operation_name": False,
    }
    assert lowering["batch_pack"] == 1024 // 128 == 8
    assert lowering["batch_groups"] == 4
    assert lowering["output_tiles"] == 4
    assert lowering["allocated_matrix_rows"] == 16
    assert lowering["reuse_group_size"] == 8
    assert lowering["layout"]["reuse_crosses_wr_gb"] is False
    storage = lowering["layout"]["matrix_storage_contract"]
    assert storage["row"] == ("row_base + output_tile * batch_groups + batch_group")
    assert storage["replica_channel_groups"] == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert lowering["logical_scalar_macs"] == 4 * 32 * 512 * 128
    assert lowering["unique_logical_gb_payload_elements"] == 4 * 32 * 128
    assert _physical_contraction_work(body) == (
        lowering["physical_mac_slots"],
        lowering["physical_wr_gb_elements"],
    )
    assert lowering["physical_mac_slots"] == lowering["logical_scalar_macs"]
    assert compiled.compiled.runtime_segments[0][1] == 1


def test_row_packed_batches_derive_batch_and_output_tails_without_overcompute_channels():
    compiled = _compile(
        AimContraction(
            outputs=49,
            reduction=70,
            batches=25,
            replicas=2,
            row=7,
            channels=tuple(range(6)),
            channels_per_replica=3,
            batch_mapping="row_packed",
        )
    )
    lowering = compiled.manifest["operations"][0]

    # Aligned K=80 gives P=floor(1024/80)=12 and a 1-batch tail group.
    assert lowering["aligned_reduction"] == 80
    assert lowering["batch_pack"] == 12
    assert [group["batches"] for group in lowering["input_groups"]] == [12, 12, 1]
    assert lowering["output_tiles"] == 2
    assert lowering["output_tile_layouts"] == [
        {
            "index": 0,
            "output_begin": 0,
            "valid_outputs_per_replica": 48,
            "active_channels_per_replica": 3,
            "active_channels": [0, 1, 2, 3, 4, 5],
            "channel_mask": 4227858432,
            "physical_output_slots_per_replica": 48,
            "padded_output_slots_per_replica": 0,
        },
        {
            "index": 1,
            "output_begin": 48,
            "valid_outputs_per_replica": 1,
            "active_channels_per_replica": 1,
            "active_channels": [0, 3],
            "channel_mask": 2415919104,
            "physical_output_slots_per_replica": 16,
            "padded_output_slots_per_replica": 15,
        },
    ]

    body = compiled.commands[:-1]
    wr_gb_indices = [
        index for index, command in enumerate(body) if command.startswith("AiM WR_GB")
    ]
    scopes = [
        body[
            begin : (
                wr_gb_indices[index + 1]
                if index + 1 < len(wr_gb_indices)
                else len(body)
            )
        ]
        for index, begin in enumerate(wr_gb_indices)
    ]
    assert [int(scope[0].split()[2]) for scope in scopes] == [60, 60, 60, 60, 5, 5]
    assert [
        sum(command.startswith("AiM MAC_ABK") for command in scope) for scope in scopes
    ] == [12, 12, 12, 12, 1, 1]
    # Every WR_GB scope consumes its own payload before the next write.
    assert all(
        any(command.startswith("AiM MAC_ABK") for command in scope) for scope in scopes
    )
    assert [
        int(
            next(
                command for command in scope if command.startswith("AiM MAC_ABK")
            ).split()[-1]
        )
        for scope in scopes
    ] == [7, 10, 8, 11, 9, 12]
    assert lowering["logical_scalar_macs"] == 2 * 25 * 49 * 70
    assert lowering["physical_mac_slots"] == 25 * (96 + 32) * 80
    assert lowering["physical_wr_gb_elements"] == 25 * (6 + 2) * 80
    assert _physical_contraction_work(body) == (
        lowering["physical_mac_slots"],
        lowering["physical_wr_gb_elements"],
    )


def test_channel_batches_match_canonical_output_layout_and_storage_stride():
    compiled = _compile(
        AimContraction(
            outputs=128,
            reduction=512,
            batches=32,
            replicas=4,
            row=521,
            channels=tuple(range(32)),
            channels_per_replica=8,
            batch_mapping="auto",
            reduction_storage_extent=4096,
        )
    )
    body = compiled.commands[:-1]
    wr_gb_indices = [
        index for index, command in enumerate(body) if command.startswith("AiM WR_GB")
    ]
    assert len(wr_gb_indices) == 4
    assert len(body) == 4 * (1 + 8 * 3)
    rows = []
    for scope_index, begin in enumerate(wr_gb_indices):
        end = (
            wr_gb_indices[scope_index + 1]
            if scope_index + 1 < len(wr_gb_indices)
            else len(body)
        )
        scope = body[begin:end]
        assert scope[0] == "AiM WR_GB 32 0 4294967295"
        assert [command.split()[1] for command in scope] == (
            ["WR_GB"] + ["WR_BIAS"] * 8 + ["MAC_ABK"] * 8 + ["RD_MAC"] * 8
        )
        rows.append(
            [
                int(command.split()[-1])
                for command in scope
                if command.startswith("AiM MAC_ABK")
            ]
        )
    assert rows == [
        list(range(521, 550, 4)),
        list(range(553, 582, 4)),
        list(range(585, 614, 4)),
        list(range(617, 646, 4)),
    ]

    lowering = compiled.manifest["operations"][0]
    assert lowering["batch_mapping"] == "channels"
    assert lowering["requested_batch_mapping"] == "auto"
    assert lowering["batch_mapping_selection"] == {
        "policy": "shape_and_residency_v1",
        "selected": "channels",
        "reason": (
            "reserved reduction storage requires channel-batched output strides"
        ),
        "uses_operation_name": False,
    }
    assert lowering["batch_groups"] == 4
    assert lowering["output_tiles"] == 8
    assert lowering["reduction_rows"] == 1
    assert lowering["reduction_storage_extent"] == 4096
    assert lowering["reduction_storage_rows"] == 4
    assert lowering["allocated_matrix_rows"] == 4 * 8 * 4
    assert lowering["accessed_matrix_rows"] == 4 * 8
    assert lowering["reuse_group_size"] == 8
    assert lowering["layout"]["reuse_crosses_wr_gb"] is False
    storage = lowering["layout"]["matrix_storage_contract"]
    assert storage["row"] == (
        "row_base + batch_group * output_tiles * reduction_storage_rows + "
        "output_tile * reduction_storage_rows + reduction_row"
    )
    assert storage["required_replica_coverage"] == [0, 1, 2, 3]
    assert storage["producer_must_populate_all_replica_channel_groups"] is True
    assert lowering["logical_scalar_macs"] == 4 * 32 * 128 * 512
    assert lowering["unique_logical_gb_payload_elements"] == 4 * 32 * 512
    assert _physical_contraction_work(body) == (
        lowering["physical_mac_slots"],
        lowering["physical_wr_gb_elements"],
    )
    assert lowering["physical_mac_slots"] == lowering["logical_scalar_macs"]


def test_channel_batches_derive_reduction_and_batch_tails_with_reserved_rows():
    compiled = _compile(
        AimContraction(
            outputs=18,
            reduction=1030,
            batches=5,
            replicas=2,
            row=100,
            channels=tuple(range(6)),
            channels_per_replica=3,
            batch_mapping="channels",
            reduction_storage_extent=2050,
            activation=True,
        )
    )
    lowering = compiled.manifest["operations"][0]

    assert lowering["batch_groups"] == 2
    assert lowering["output_tiles"] == 2
    assert lowering["output_tile_layouts"] == [
        {
            "index": 0,
            "output_begin": 0,
            "valid_outputs_per_batch": 16,
            "physical_output_slots_per_batch": 16,
            "padded_output_slots_per_batch": 0,
        },
        {
            "index": 1,
            "output_begin": 16,
            "valid_outputs_per_batch": 2,
            "physical_output_slots_per_batch": 16,
            "padded_output_slots_per_batch": 14,
        },
    ]
    assert lowering["reduction_rows"] == 2
    assert lowering["reduction_storage_rows"] == 3
    assert lowering["allocated_matrix_rows"] == 2 * 2 * 3
    assert lowering["accessed_matrix_rows"] == 2 * 2 * 2
    assert [group["batches"] for group in lowering["groups"]] == [3, 2]
    assert [group["active_channels"] for group in lowering["groups"]] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 3, 4],
    ]
    assert [
        chunk["matrix_rows"]
        for group in lowering["groups"]
        for chunk in group["row_chunks"]
    ] == [
        [100, 103],
        [101, 104],
        [106, 109],
        [107, 110],
    ]
    assert [
        chunk["op_size"]
        for group in lowering["groups"]
        for chunk in group["row_chunks"]
    ] == [64, 1, 64, 1]

    body = compiled.commands[:-1]
    wr_gb_indices = [
        index for index, command in enumerate(body) if command.startswith("AiM WR_GB")
    ]
    scopes = [
        body[
            begin : (
                wr_gb_indices[index + 1]
                if index + 1 < len(wr_gb_indices)
                else len(body)
            )
        ]
        for index, begin in enumerate(wr_gb_indices)
    ]
    assert len(scopes) == 4
    # Activation is legal only on the final reduction chunk in each batch
    # group, and its paired readbacks remain inside that payload's scope.
    assert [
        sum(command.startswith("AiM AF") for command in scope) for scope in scopes
    ] == [0, 2, 0, 2]
    assert [
        sum(command.startswith("AiM RD_AF") for command in scope) for scope in scopes
    ] == [0, 2, 0, 2]
    assert lowering["reuse_window_entries_per_launch"] == 2
    assert lowering["logical_scalar_macs"] == 2 * 5 * 18 * 1030
    assert lowering["physical_mac_slots"] == 2 * 5 * 2 * 16 * 1040
    assert lowering["physical_wr_gb_elements"] == 2 * 5 * 1040
    assert _physical_contraction_work(body) == (
        lowering["physical_mac_slots"],
        lowering["physical_wr_gb_elements"],
    )


def test_contraction_activation_is_grouped_on_only_the_final_k_row():
    compiled = _compile(
        AimContraction(
            outputs=50,
            reduction=17,
            row=7,
            channels=(0, 1),
            reuse_group_size=2,
            activation=True,
        )
    )

    assert compiled.commands == (
        "AiM WR_GB 2 0 3221225472",
        "AiM WR_BIAS 0 3221225472",
        "AiM WR_BIAS 0 3221225472",
        "AiM MAC_ABK 2 3221225472 7",
        "AiM MAC_ABK 2 3221225472 8",
        "AiM AF 3221225472",
        "AiM RD_AF 0 3221225472",
        "AiM AF 3221225472",
        "AiM RD_AF 0 3221225472",
        "AiM RD_MAC 0 3221225472",
        "AiM RD_MAC 0 3221225472",
        "AiM EOC",
    )


def test_reuse_groups_are_target_derived_balanced_and_activation_aware():
    target = build_aim_target()
    assert target.mac_reg.slots == 32

    # 86 launches need three non-activation groups under the target's 32-entry
    # reuse window, so the generic balanced schedule uses 29/29/28.  Activation
    # budgets two entries per live launch and therefore uses 15x5 plus 11.
    outputs = 86 * 32 * 16
    plain = compile_aim_program(
        AimProgram([AimContraction(outputs=outputs, reduction=16)]),
        target,
    )
    activated = compile_aim_program(
        AimProgram([AimContraction(outputs=outputs, reduction=16, activation=True)]),
        target,
    )

    plain_lowering = plain.manifest["operations"][0]
    assert plain_lowering["reuse_window_capacity"] == 32
    assert plain_lowering["reuse_window_entries_per_launch"] == 1
    assert plain_lowering["effective_reuse_capacity"] == 32
    assert plain_lowering["configured_reuse_group_size"] == 32
    assert plain_lowering["reuse_group_size"] == 29

    activated_lowering = activated.manifest["operations"][0]
    assert activated_lowering["reuse_window_entries_per_launch"] == 2
    assert activated_lowering["effective_reuse_capacity"] == 16
    assert activated_lowering["configured_reuse_group_size"] == 16
    assert activated_lowering["reuse_group_size"] == 15

    def bias_runs(commands):
        runs = []
        current = 0
        for command in commands:
            if command.startswith("AiM WR_BIAS"):
                current += 1
            elif current:
                runs.append(current)
                current = 0
        return runs

    assert bias_runs(plain.commands) == [29, 29, 28]
    assert bias_runs(activated.commands) == [15, 15, 15, 15, 15, 11]


def test_mixed_typed_operations_form_one_continuous_trace():
    compiled = _compile(
        AimHostTransfer("write", channel=3, bank=5, row=9, bursts=2),
        AimElementwise("mul", elements=513, row=3, channels=(0, 1)),
        AimElementwise(
            "add",
            elements=513,
            gpr_addr_0=10,
            gpr_addr_1=100,
        ),
        AimBankCopy(
            "bank_to_gb",
            elements=2050,
            bank=6,
            row=7,
            channels=(0, 1),
        ),
        AimActivation(groups=2),
        AimSync(),
        AimHostTransfer("read", channel=4, bank=6, row=10),
    )

    assert compiled.commands == (
        "W MEM 3 5 9",
        "W MEM 3 5 9",
        "AiM EWMUL 5 3221225472 3",
        "AiM EWADD 33 10 100",
        "AiM COPY_BKGB 64 3221225472 6 7",
        "AiM COPY_BKGB 1 3221225472 6 8",
        "AiM AF 4294967295",
        "AiM RD_AF 0 4294967295",
        "AiM AF 4294967295",
        "AiM RD_AF 0 4294967295",
        "AiM SYNC",
        "R MEM 4 6 10",
        "AiM EOC",
    )
    spans = [entry["command_span"] for entry in compiled.manifest["operations"]]
    assert spans == [[0, 2], [2, 3], [3, 4], [4, 6], [6, 10], [10, 11], [11, 12]]
    mul = compiled.manifest["operations"][1]
    add = compiled.manifest["operations"][2]
    assert mul["logical_elements"] == 513
    assert mul["physical_element_slots"] == 5 * 16 * 2 * 4
    assert add["logical_elements"] == 513
    assert add["physical_element_slots"] == 33 * 16
    assert compiled.trace.endswith("AiM EOC\n")
    assert compiled.manifest["trace"]["eoc_count"] == 1


def test_bank_group_layout_drives_elementwise_and_copy_span():
    compiled = _compile(
        AimElementwise(
            "mul",
            elements=11008,
            row=40,
            replicas=4,
            channels_per_replica=8,
        ),
        AimBankCopy(
            "bank_to_gb",
            elements=11008,
            bank=2,
            row=41,
            copies=4,
            bank_stride=4,
            replicas=4,
            channels_per_replica=8,
        ),
    )

    assert compiled.commands == (
        "AiM EWMUL 22 4294967295 40",
        "AiM COPY_BKGB 22 4294967295 2 41",
        "AiM COPY_BKGB 22 4294967295 6 41",
        "AiM COPY_BKGB 22 4294967295 10 41",
        "AiM COPY_BKGB 22 4294967295 14 41",
        "AiM EOC",
    )
    mul, copy = compiled.manifest["operations"]
    assert mul["partitions_per_replica"] == 8 * 4
    assert mul["elements_per_partition"] == 344
    assert mul["logical_elements"] == 11008 * 4
    assert mul["physical_element_slots"] == 22 * 16 * 32 * 4
    assert copy["partitions_per_replica"] == 8 * 4
    assert copy["elements_per_partition"] == 344


def test_bank_bank_contraction_and_all_bank_write_are_typed():
    compiled = _compile(
        AimAllBankWrite(
            row=10,
            rows=2,
            row_stride=3,
            channels=(0, 2),
            gpr_addr=7,
        ),
        AimContraction(
            outputs=1,
            reduction=64,
            row=20,
            channels=(0,),
            input_source="banks",
        ),
    )

    assert compiled.commands == (
        "AiM WR_ABK 7 2147483648 10",
        "AiM WR_ABK 7 536870912 10",
        "AiM WR_ABK 7 2147483648 13",
        "AiM WR_ABK 7 536870912 13",
        "AiM WR_BIAS 0 2147483648",
        "AiM MAC_ABK 4 2147483648 20",
        "AiM RD_MAC 0 2147483648",
        "AiM EOC",
    )
    contraction = compiled.manifest["operations"][1]
    assert contraction["input_source"] == "banks"
    assert contraction["wr_gb_launches_per_reduction_row"] == 0
    assert contraction["unique_logical_gb_payload_elements"] == 0
    assert contraction["physical_wr_gb_elements"] == 0
    assert contraction["logical_scalar_macs"] == 64
    assert contraction["physical_mac_slots"] == 16 * 64
    assert not any(command.startswith("AiM WR_GB") for command in compiled.commands)


def test_distributed_host_transfer_expands_linear_layout_and_burst_tail():
    compiled = _compile(
        AimDistributedHostTransfer(
            "write",
            elements=8193,
            replicas=1,
            channels_per_replica=2,
            row=30,
            bank_stride=4,
            bank_offset=1,
        )
    )

    lowering = compiled.manifest["operations"][0]
    assert lowering["partitions_per_replica"] == 2 * (16 // 4)
    assert lowering["elements_per_partition"] == 1025
    assert lowering["rows_per_partition"] == 2
    assert lowering["bursts"] == 8 * 64 + 7
    assert lowering["iteration_order"] == [
        "row",
        "burst",
        "partition",
        "replica",
        "copy",
    ]
    assert compiled.commands[:4] == (
        "W MEM 0 1 30",
        "W MEM 0 5 30",
        "W MEM 0 9 30",
        "W MEM 0 13 30",
    )
    assert compiled.commands[-8:-1] == (
        "W MEM 0 1 31",
        "W MEM 0 5 31",
        "W MEM 0 9 31",
        "W MEM 0 13 31",
        "W MEM 1 1 31",
        "W MEM 1 5 31",
        "W MEM 1 9 31",
    )
    assert compiled.commands[-1] == "AiM EOC"


def test_reuse_and_storage_are_validated_against_target_capacity():
    with pytest.raises(ValueError, match="reuse-window capacity"):
        _compile(
            AimContraction(
                outputs=512 * 17,
                reduction=16,
                channels=tuple(range(32)),
                reuse_group_size=33,
            )
        )

    with pytest.raises(ValueError, match="bank rows"):
        _compile(
            AimContraction(
                outputs=1024,
                reduction=2048,
                row=16382,
                channels=tuple(range(32)),
            )
        )


def test_specialized_contractions_reject_unsafe_or_misleading_layouts():
    with pytest.raises(ValueError, match="batch_mapping must be"):
        AimContraction(outputs=16, reduction=16, batch_mapping="score")

    with pytest.raises(ValueError, match="cannot be smaller"):
        AimContraction(
            outputs=16,
            reduction=33,
            batch_mapping="channels",
            reduction_storage_extent=32,
        )

    with pytest.raises(ValueError, match="only meaningful"):
        _compile(
            AimContraction(
                outputs=16,
                reduction=16,
                reduction_storage_extent=32,
            )
        )

    with pytest.raises(ValueError, match="requires input_source='gb'"):
        _compile(
            AimContraction(
                outputs=16,
                reduction=16,
                batch_mapping="channels",
                input_source="banks",
            )
        )

    # P=floor(row_elements/aligned_K) is zero.  Silent fallback to flattened
    # batches would stage four payloads and overwrite the first three.
    with pytest.raises(ValueError, match="fits in one bank row"):
        _compile(
            AimContraction(
                outputs=64,
                reduction=1030,
                batches=16,
                channels=(0, 1, 2, 3),
                channels_per_replica=4,
                batch_mapping="row_packed",
            )
        )

    with pytest.raises(ValueError, match="distinct batch GB payloads"):
        _compile(
            AimContraction(
                outputs=64,
                reduction=128,
                batches=16,
                channels=(0, 1, 2, 3),
                channels_per_replica=4,
                batch_mapping="flattened",
            )
        )

    with pytest.raises(ValueError, match="found no dependency-safe layout"):
        _compile(
            AimContraction(
                outputs=64,
                reduction=1030,
                batches=2,
                channels=(0, 1, 2, 3),
                channels_per_replica=4,
                batch_mapping="auto",
            )
        )


def test_auto_mapping_selects_safe_single_payload_and_bank_resident_layouts():
    compiled = _compile(
        AimContraction(
            outputs=16,
            reduction=16,
            row=0,
            channels=(0,),
            batch_mapping="auto",
        ),
        AimContraction(
            outputs=16,
            reduction=16,
            batches=2,
            row=2,
            channels=(0, 1),
            input_source="banks",
            batch_mapping="auto",
        ),
    )

    gb, banks = compiled.manifest["operations"]
    assert gb["batch_mapping"] == "flattened"
    assert gb["batch_mapping_selection"] == {
        "policy": "shape_and_residency_v1",
        "selected": "flattened",
        "reason": "one GB payload needs no batch packing",
        "uses_operation_name": False,
    }
    assert banks["batch_mapping"] == "flattened"
    assert banks["batch_mapping_selection"] == {
        "policy": "shape_and_residency_v1",
        "selected": "flattened",
        "reason": "bank-resident input needs no GB batch layout",
        "uses_operation_name": False,
    }


def test_negative_channel_indices_fail_before_mask_encoding():
    # The simulator only examines its 32 legal mask bits.  Encoding channel
    # -1 would otherwise set bit 32 and silently dispatch to zero channels.
    with pytest.raises(ValueError, match="channel indices must be nonnegative"):
        AimContraction(outputs=16, reduction=16, channels=(-1,))


def test_public_compile_returns_inspectable_no_argument_runner(monkeypatch):
    compiled = allo.compile(
        AimProgram([AimSync()], name="public_dispatch"),
        build_aim_target(),
    )
    assert isinstance(compiled, AimProgramCallable)
    assert compiled.commands == ("AiM SYNC", "AiM EOC")
    assert compiled.manifest["name"] == "public_dispatch"
    assert compiled.manifest["trace"]["command_count"] == 2
    assert json.loads(json.dumps(compiled.manifest))["name"] == "public_dispatch"

    seen = {}

    def fake_run(materialization, **_inputs):
        seen["commands"] = materialization.commands
        return RunResult(123, "measured", "aim")

    monkeypatch.setattr(aim_program_module, "_run_aim", fake_run)
    result = compiled()
    assert result.cycles == 123
    assert compiled.last_result is result
    assert seen["commands"] == compiled.commands
