# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural coverage tests for the configurable CENT/AiM workloads."""

import pytest

import allo
from allo.pim.aim_program import AimAllBankWrite
from allo.pim.targets import build_aim_target
from benchmarks.cent_aim.vendor_contract import VENDOR_TRACE_CONTRACTS
from benchmarks.cent_aim.workloads import (
    CENT_CASES,
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
