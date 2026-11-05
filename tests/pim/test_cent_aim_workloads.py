# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CENT/AiM cases through the matcher path: trace identity and inventories."""

from dataclasses import replace

import pytest

import allo
from allo.pim.aim_lowering import AimAllBankWrite, AimContraction
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_aim import _aim_runtime_segments
from allo.spmw_autoschedule import _bucket_for_autoschedule
from allo.spmw_codegen import compile_for_target
from benchmarks.cent_aim.reference_traces import TYPED_TRACE_SHA256
from benchmarks.cent_aim.vendor_contract import VENDOR_TRACE_CONTRACTS
from benchmarks.cent_aim.workloads import (
    CENT_CASES,
    build_layout,
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

# Same default/paper_full split as the AiM golden rows.
_DEFAULT_CASES = frozenset({"norm_residual_l128", "attention_l128", "softmax_pim_l128"})


def _compile(case, backend=None):
    return allo.compile(
        case.region,
        build_aim_target(),
        aim_cost,
        host_moves=case.host_program,
        backend=backend,
        # Every AiM MAC reduces 16 BF16 products through the bank adder tree.
        reduction_order="reassociate",
    )


@pytest.mark.parametrize(
    "case,spec,cent",
    [
        (
            params
            if params[0].case_id in _DEFAULT_CASES
            else pytest.param(*params, marks=pytest.mark.paper_full)
        )
        for params in iter_case_programs()
    ],
    ids=EXPECTED_CASE_IDS,
)
def test_matcher_trace_is_typed_trace_and_vendor_inventory(case, spec, cent):
    compiled = _compile(cent)
    manifest = semantic_manifest(case.case_id, spec, compiled)
    commands = [str(line) for line in compiled.compiled.cmds]

    # Byte identity with the pinned trace (the typed-route trace, or the
    # corrected program's trace where the benchmark program was corrected).
    from benchmarks.cent_aim.reference_traces import reference_trace_sha256

    assert manifest["compiled_trace"]["sha256"] == reference_trace_sha256(case.case_id)

    # Vendor opcode inventory, except the complete V-cache append.
    expected = dict(VENDOR_TRACE_CONTRACTS[case.case_id].opcode_counts)
    observed = manifest["compiled_trace"]["opcode_counts"]
    appends = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimAllBankWrite)
    ]
    if appends:
        tenon_wr_abk = sum(len(op.channels) * op.rows * op.copies for op in appends)
        assert observed["AiM_WR_ABK"] == tenon_wr_abk
        assert expected["AiM_WR_ABK"] < tenon_wr_abk
        assert {k: v for k, v in observed.items() if k != "AiM_WR_ABK"} == {
            k: v for k, v in expected.items() if k != "AiM_WR_ABK"
        }
    else:
        assert observed == expected
    assert commands[-1] == "AiM EOC" and commands.count("AiM EOC") == 1
    # One pre-terminated segment, measured once (no repeat multiplier).
    assert [repeat for _label, repeat, _lines in _aim_runtime_segments(commands)] == [1]

    # Every contraction kernel scored both bank scopes and chose ABK.
    contraction_kernels = 0
    for kernel in compiled.placement_ranking.kernels:
        modes = {position: mode for position, _c, mode in kernel.scores}
        if not any("+bank_scope=sbk" in mode for mode in modes.values()):
            assert all("+bank_scope=none" in mode for mode in modes.values())
            continue
        contraction_kernels += 1
        assert any("+bank_scope=abk" in mode for mode in modes.values())
        assert "+bank_scope=abk" in modes[kernel.chosen_position]
    assert contraction_kernels == 11

    # The allocator reproduces the shape-derived decode layout.
    layout = build_layout(spec)
    rows = {
        tuple(region["buffers"]): (region["row"], region["rows"])
        for region in manifest["compiled_trace"]["row_regions"]
    }
    assert sorted(rows.values()) == sorted(
        (region.row, region.rows) for region in layout.regions.values()
    )

    assert manifest["topology"]["replica_channel_groups"] == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    for caveat in (
        "replica_values_not_modeled",
        "residuals_counted_once",
        "rope_post_result_transfer",
        "complete_v_cache_append_wr_abk",
    ):
        assert caveat in manifest["scope_caveats"]


def test_probe_and_emit_derive_the_same_replicated_contraction():
    """C1: the module-free feasibility probe and the final emit agree."""
    case_id = "attention_l128"
    _case, spec, cent = next(
        params for params in iter_case_programs() if params[0].case_id == case_id
    )
    compiled = _compile(cent, backend="virtual")
    emitted = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimContraction) and op.batch_mapping == "row_packed"
    ]
    trace = compiled.compiled.trace
    buckets = _bucket_for_autoschedule(trace)
    layouts = compiled.compiled.layout
    qk = next(
        index
        for index, (_scope, matches) in enumerate(buckets)
        if matches[0].func_name.startswith("attention_qk")
    )
    group_id = buckets[qk][1][0].extra["spmw_work_scope"].group_id
    indices = [
        index
        for index, (_scope, matches) in enumerate(buckets)
        if matches[0].extra["spmw_work_scope"].group_id == group_id
    ]
    sub_trace = type(trace)(
        target_name=trace.target_name,
        module_name=trace.module_name,
        matches=[match for index in indices for match in buckets[index][1]],
    )
    probe = compile_for_target(
        build_aim_target(), sub_trace, layout=[layouts[index] for index in indices]
    )
    probed = list(probe.layout_ctx.lowered_operations)
    assert len(emitted) == len(probed) == 1
    # Rows are allocated over the whole region at emit and over one kernel in
    # the probe; every other derived field is identical.
    assert replace(probed[0], row=0) == replace(emitted[0], row=0)
    # Shape and placement fields equal the QK record the removed typed
    # builder produced at tenon@ebb90a6 (the matcher additionally pins the
    # mapping the typed record left to the lowerer's "auto" rule).
    fields = (
        "outputs",
        "reduction",
        "batches",
        "replicas",
        "row",
        "channels",
        "channels_per_replica",
        "input_source",
        "reduction_storage_extent",
    )
    assert {f: getattr(emitted[0], f) for f in fields} == {
        "outputs": 128,
        "reduction": 128,
        "batches": 32,
        "replicas": 4,
        "row": 11,
        "channels": tuple(range(32)),
        "channels_per_replica": 8,
        "input_source": "gb",
        "reduction_storage_extent": None,
    }


def test_case_table_is_exactly_the_fourteen_archived_cases():
    assert tuple(CENT_CASES) == EXPECTED_CASE_IDS
    assert tuple(VENDOR_TRACE_CONTRACTS) == EXPECTED_CASE_IDS
    assert tuple(TYPED_TRACE_SHA256) == EXPECTED_CASE_IDS


# --------------------------------------------------------------------- #
# X-19: the RoPE multiply covers the whole 2D vector at the device minimum
# --------------------------------------------------------------------- #


def _rope_width(spec):
    """Columns per EWMUL the geometry allows for the full 2D vector: one
    bank pair per bank group on each channel of the replica, 16 lanes."""
    from benchmarks.cent_aim.workloads import BANK_GROUPS_PER_CHANNEL, VECTOR_LANES

    partitions = spec.channels_per_replica * BANK_GROUPS_PER_CHANNEL
    return -(-(2 * spec.D) // (partitions * VECTOR_LANES))


@pytest.mark.parametrize(
    "D, H, F",
    [(4096, 32, 11008), pytest.param(2048, 16, 5504, marks=pytest.mark.paper_full)],
    ids=["paper-D4096", "neighbour-D2048"],
)
def test_rope_ewmul_width_is_the_device_minimum_for_the_full_vector(D, H, F):
    """The corrected RoPE kernels multiply every element the host scatters
    (2D per replica) and lower to one EWMUL each whose op size is the
    smallest the device geometry allows for that vector; the width scales
    with D (X-19: the paper's digits came from a half-width multiply).
    The selection is the cost minimum of the enumerated replica partitions
    and every projection candidate ties on an identical command stream."""
    from dataclasses import replace

    from benchmarks.cent_aim.workloads import DEFAULT_DECODE_SPEC, build_case

    spec = replace(DEFAULT_DECODE_SPEC, D=D, H=H, F=F, L=128)
    case = build_case("qkvo_rope_l128", spec)
    compiled = _compile(case)
    commands = [str(line) for line in compiled.compiled.cmds]
    ewmul = [line for line in commands if line.startswith("AiM EWMUL ")]
    assert len(ewmul) == 2
    width = _rope_width(spec)
    assert [int(line.split()[2]) for line in ewmul] == [width, width]
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    for name in ("rope_q", "rope_k"):
        entry = next(e for e in manifest if e.get("name") == name)
        assert entry["logical_elements"] == 2 * D * spec.replicas
        assert entry["physical_element_slots"] >= entry["logical_elements"]
        assert entry["row_op_sizes"] == [width]
    # The elementwise selection is the cheapest enumerated replica partition
    # and the projections' tied candidates lower to the same stream.
    ranking = compiled.compiled.placement_ranking
    for kernel in ranking.kernels:
        chosen = next(s for s in kernel.scores if s[0] == kernel.chosen_position)
        assert chosen[1] == kernel.scores[0][1]
