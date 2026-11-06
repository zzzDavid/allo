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
    # Since round 71 a replicated elementwise grid is realized once over the
    # device, so its operands are written, its multiplies and adds issued and
    # its result read once instead of once per replica: Tenon's host writes
    # and reads are the lowered transfers' bursts and its EWMUL / EWADD
    # counts the lowered elementwise records' commands, never more than the
    # vendor's.
    lowered = {"W_MEM": 0, "R_MEM": 0, "AiM_EWMUL": 0, "AiM_EWADD": 0}
    for entry in compiled.compiled.layout_ctx.lowering_manifest:
        if entry.get("kind") in ("hosttransfer", "distributedhosttransfer"):
            key = "W_MEM" if entry["direction"] == "write" else "R_MEM"
            lowered[key] += int(entry.get("bursts") or 0)
        elif entry.get("kind") == "elementwise" and entry.get("active_channels"):
            key = {"mul": "AiM_EWMUL", "add": "AiM_EWADD"}[entry["operation"]]
            lowered[key] += int(entry["command_count"])
    for key in lowered:
        if observed.get(key, 0) != expected.get(key, 0):
            assert observed.get(key, 0) == lowered[key], (key, observed, lowered)
            assert observed.get(key, 0) < expected.get(key, 0), (
                key,
                observed,
                expected,
            )
    observed_rest = {k: v for k, v in observed.items() if k not in lowered}
    expected_rest = {k: v for k, v in expected.items() if k not in lowered}
    if appends:
        tenon_wr_abk = sum(len(op.channels) * op.rows * op.copies for op in appends)
        assert observed["AiM_WR_ABK"] == tenon_wr_abk
        assert expected["AiM_WR_ABK"] < tenon_wr_abk
        assert {k: v for k, v in observed_rest.items() if k != "AiM_WR_ABK"} == {
            k: v for k, v in expected_rest.items() if k != "AiM_WR_ABK"
        }
    else:
        assert observed_rest == expected_rest
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
        # 11 at tenon@ebb90a6; 8 since round 71, when the replicated
        # elementwise groups allocated ahead of it shrank to one copy.
        "row": 8,
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


def _rope_width(spec, channels=None):
    """Columns per EWMUL the geometry allows for the full 2D vector: one
    bank pair per bank group on each of ``channels`` (the replica's channels
    by default; all 32 for the single-copy realization), 16 lanes."""
    from benchmarks.cent_aim.workloads import BANK_GROUPS_PER_CHANNEL, VECTOR_LANES

    channels = spec.channels_per_replica if channels is None else channels
    partitions = channels * BANK_GROUPS_PER_CHANNEL
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
    Also checked: the manifest's declared slot capacity covers the logical
    elements and each kernel group's chosen candidate has the minimum
    estimated cost of its ranking.  Not checked here: address-level
    coverage of the scattered elements or the command streams of the other
    projection candidates (they are not lowered by this test)."""
    from dataclasses import replace

    from benchmarks.cent_aim.workloads import DEFAULT_DECODE_SPEC, build_case

    spec = replace(DEFAULT_DECODE_SPEC, D=D, H=H, F=F, L=128)
    case = build_case("qkvo_rope_l128", spec)
    compiled = _compile(case)
    commands = [str(line) for line in compiled.compiled.cmds]
    ewmul = [line for line in commands if line.startswith("AiM EWMUL ")]
    assert len(ewmul) == 2
    # Since round 71 the replicated multiply is realized once over all 32
    # channels (one copy of the 2D vector): the device minimum for that
    # layout; the per-work-item realization keeps the replica's width.
    width = _rope_width(spec, channels=32)
    assert [int(line.split()[2]) for line in ewmul] == [width, width]
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    for name in ("rope_q", "rope_k"):
        entry = next(e for e in manifest if e.get("name") == name)
        assert entry["logical_elements"] == 2 * D
        assert entry["physical_element_slots"] >= entry["logical_elements"]
        assert entry["row_op_sizes"] == [width]
    per_item = allo.compile(
        case.region,
        build_aim_target(),
        aim_cost,
        host_moves=case.host_program,
        reduction_order="reassociate",
        layout=_force_replica_partitions(
            compiled, ("rope_q", "rope_k"), spec.channels_per_replica
        ),
    )
    per_item_width = _rope_width(spec)
    assert [
        int(l.split()[2])
        for l in (str(x) for x in per_item.compiled.cmds)
        if l.startswith("AiM EWMUL ")
    ] == [per_item_width, per_item_width]
    assert per_item_width == width * (32 // spec.channels_per_replica)
    # Each kernel group's chosen candidate has the minimum estimated cost
    # of its ranking (ties included); an estimate ranking, not a simulation.
    ranking = compiled.compiled.placement_ranking
    for kernel in ranking.kernels:
        chosen = next(s for s in kernel.scores if s[0] == kernel.chosen_position)
        assert chosen[1] == kernel.scores[0][1]


# --------------------------------------------------------------------------- #
# A replicated elementwise grid realized once over the device (X-19, round 71)
# --------------------------------------------------------------------------- #


def _rope_group_records(compiled):
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    return {
        entry["name"]: (
            entry.get("replicas"),
            entry.get("channels_per_replica"),
            entry.get("bursts"),
        )
        for entry in manifest
        if entry.get("kind") in ("elementwise", "distributedhosttransfer")
        and entry.get("name") in ("m_q", "m_k", "rope_q", "rope_k", "r_q", "r_k")
    }


def _force_replica_partitions(compiled, prefixes, partitions):
    """The compiled program's placements with the kernel groups whose function
    names start with ``prefixes`` replaced by their candidate of the given
    ``replica_partitions`` (from the ranking's choice set)."""
    from allo.spmw_autoschedule import _matcher_work_scope

    inner = compiled.compiled
    buckets = _bucket_for_autoschedule(inner.trace)
    kernels = {k.group_id: k for k in inner.placement_ranking.kernels}
    layouts = list(inner.layout)
    for index, (_scope, matches) in enumerate(buckets):
        if not matches[0].func_name.startswith(prefixes):
            continue
        group_id = _matcher_work_scope(matches[0]).group_id
        layouts[index] = next(
            p
            for _pos, p in kernels[group_id].candidates
            if (p.extra or {}).get("replica_partitions") == partitions
        )
    return layouts


def test_replicated_elementwise_groups_are_realized_once_over_the_device():
    """The RoPE multiplies of ``qkvo_rope_l128`` are replicated grids (every
    work item multiplies the same 2D-element vectors), so the enumerator
    offers, and the default selection takes, one copy over all 32 channels:
    each operand and the result are transferred once (512 bursts, not one
    copy per work item) and each EWMUL spans 4 columns; the per-work-item
    alternative (8 channels per copy) is estimated and simulated slower and
    stays selectable."""
    from benchmarks.cent_aim.workloads import build_case

    case = build_case("qkvo_rope_l128")
    compiled = _compile(case)
    records = _rope_group_records(compiled)
    assert records == {
        "m_q": (1, 32, 512),
        "m_k": (1, 32, 512),
        "rope_q": (1, 32, None),
        "rope_k": (1, 32, None),
        "r_q": (1, 32, 512),
        "r_k": (1, 32, 512),
    }
    commands = [str(c) for c in compiled.compiled.cmds]
    assert [c for c in commands if "EWMUL" in c] == [
        "AiM EWMUL 4 4294967295 4",
        "AiM EWMUL 4 4294967295 5",
    ]
    assert sum(c.startswith("W MEM") for c in commands) == 2048
    layouts = _force_replica_partitions(compiled, ("rope_q", "rope_k"), 8)
    per_item = allo.compile(
        case.region,
        build_aim_target(),
        aim_cost,
        host_moves=case.host_program,
        reduction_order="reassociate",
        layout=layouts,
    )
    assert _rope_group_records(per_item)["m_q"] == (4, 8, 2048)
    assert [c for c in (str(x) for x in per_item.compiled.cmds) if "EWMUL" in c][
        0
    ].startswith("AiM EWMUL 16 ")
    assert per_item.estimate().cycles > compiled.estimate().cycles


def test_reduced_copy_elementwise_needs_readers_of_the_same_layout():
    """``rms_scale`` writes ``t`` that ``rms_norm`` reads on the device: both
    replicated grids take one copy by default; forcing the producer to one
    copy and the reader to per-work-item copies is refused at materialization,
    not lowered with a reader of another layout."""
    from benchmarks.cent_aim.workloads import build_case

    case = build_case("norm_residual_l128")
    compiled = _compile(case)
    manifest = {
        e["name"]: e.get("replicas")
        for e in compiled.compiled.layout_ctx.lowering_manifest
        if e.get("kind") == "elementwise"
    }
    assert manifest["rms_scale_x"] == 1 and manifest["rms_norm_x"] == 1
    layouts = _force_replica_partitions(compiled, ("rms_norm",), 8)
    with pytest.raises(Exception, match="needs readers of the same layout"):
        allo.compile(
            case.region,
            build_aim_target(),
            aim_cost,
            host_moves=case.host_program,
            reduction_order="reassociate",
            layout=layouts,
        )


@pytest.mark.paper_full
def test_single_copy_elementwise_on_the_neighbouring_rope_shape():
    from benchmarks.cent_aim.workloads import DEFAULT_DECODE_SPEC, build_case

    spec = replace(DEFAULT_DECODE_SPEC, D=2048, H=16, F=5504, L=128)
    compiled = _compile(build_case("qkvo_rope_l128", spec))
    assert _rope_group_records(compiled)["m_q"] == (1, 32, 256)
    assert [c for c in (str(x) for x in compiled.compiled.cmds) if "EWMUL" in c][
        0
    ].startswith("AiM EWMUL 2 ")
