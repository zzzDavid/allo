# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AttAcc trace structure check (``benchmarks/attacc/trace_check.py``):
ordering and address mapping of a trace against the selected program, not
just opcode totals.  The DCC-free tests build traces from the generator's
documented structure; the last test runs DCC's own generator when a
checkout is available (``ATTACC_DCC_ROOT``)."""

import os
from pathlib import Path

import pytest

from allo.pim.attacc_tiling import (
    AttAccKernel,
    AttAccTiling,
    attacc_command_counts,
    attacc_geometry,
    attacc_program,
)
from allo.pim.targets import build_attacc_target

from benchmarks.attacc import trace_check


@pytest.fixture(scope="module")
def geometry():
    return attacc_geometry(build_attacc_target())


QKV_NH8 = [[1, 8, 1], [8, 8, 80], [2, 1, 192]]


def _synthetic_trace(path, program, geometry, *, mutate=None):
    """A trace with the generator's ordering and address mapping, written
    from the program's counts (the structure the checker expects)."""
    g = trace_check.granules(geometry)
    counts = program.counts
    n, it = counts.iterations, counts.rows_per_iteration
    ch, ba = counts.channels_used, counts.units_used
    s1, s2 = counts.serial_reduction, counts.serial_outputs
    lanes, bpg = geometry.lanes, geometry.banks_per_group
    lines = []

    def barrier():
        lines.extend(f"PIM_BARRIER 0x{c * g['ch']:08x}" for c in range(geometry.channels))

    for _itr in range(n):
        for pos in range(-(-it * s1 // lanes)):
            for b in range(ba):
                for c in range(ch):
                    lines.append(f"ST 0x{c * g['ch'] + b * g['ba'] + pos * g['col']:08x}")
        barrier()
        mac = []
        for pos2 in range(s2):
            for pos1 in range(-(-s1 // lanes)):
                for c in range(ch):
                    mac.append(f"PIM_MAC_AB 0x{c * g['ch'] + (pos1 * s2 + pos2) * g['col']:08x}")
            if pos2 % lanes == lanes - 1 or pos2 == s2 - 1:
                for bg in range(-(-ba // bpg)):
                    for c in range(ch):
                        mac.append(f"PIM_MV_SB 0x{c * g['ch'] + bg * bpg * g['ba']:08x}")
        for j in range(it):
            first = -(-(j * s1) // lanes) * lanes
            for pos in range(first, (j + 1) * s1, lanes):
                for b in range(ba):
                    for c in range(ch):
                        lines.append(f"PIM_WR_GB 0x{c * g['ch'] + b * g['ba'] + pos:08x}")
            barrier()
            lines.extend(mac)
            barrier()
        barrier()
        for pos in range(-(-it * s2 // lanes)):
            for b in range(ba):
                for c in range(ch):
                    lines.append(f"ST 0x{c * g['ch'] + b * g['ba'] + pos * g['col']:08x}")
    if mutate:
        lines = mutate(lines)
    Path(path).write_text("\n".join(lines))
    return path


def test_synthetic_trace_in_the_generator_structure_passes(tmp_path, geometry):
    program = attacc_program(AttAccKernel("gemv", 8, 5120, 384), AttAccTiling.coerce(QKV_NH8), geometry)
    trace = _synthetic_trace(tmp_path / "ok.trace", program, geometry)
    report = trace_check.check_trace(trace, program)
    assert report["totals_match"] and report["ordering_checked"] and report["ordering_match"]
    assert report["mapping_match"], report["mapping"]
    assert report["commands"] == sum(dict(program.counts.commands()).values())
    assert report["mapping"]["channels_addressed"] == 16 and report["mapping"]["units_addressed"] == 64
    assert report["mapping"]["mac_columns_per_channel"] == [192 * 5]


def test_reordered_commands_fail_the_ordering_check_with_equal_totals(tmp_path, geometry):
    """Moving one WR_GB after the MACs keeps every opcode total and fails."""
    program = attacc_program(AttAccKernel("gemv", 8, 5120, 384), AttAccTiling.coerce(QKV_NH8), geometry)

    def swap(lines):
        i = next(k for k, l in enumerate(lines) if l.startswith("PIM_WR_GB"))
        j = next(k for k, l in enumerate(lines) if l.startswith("PIM_MAC_AB"))
        lines[i], lines[j] = lines[j], lines[i]
        return lines

    trace = _synthetic_trace(tmp_path / "swapped.trace", program, geometry, mutate=swap)
    report = trace_check.check_trace(trace, program)
    assert report["totals_match"] and not report["ordering_match"]
    assert report["first_difference"]["expected"] == "PIM_WR_GB" and report["first_difference"]["actual"] == "PIM_MAC_AB"


def test_wrong_channel_mapping_fails_with_equal_totals_and_order(tmp_path, geometry):
    """Addressing every MAC at channel 0 keeps totals and order and fails
    the mapping check (one channel addressed, sixteen declared)."""
    program = attacc_program(AttAccKernel("gemv", 8, 5120, 384), AttAccTiling.coerce(QKV_NH8), geometry)
    g = trace_check.granules(geometry)

    def collapse(lines):
        return [f"PIM_MAC_AB 0x{int(l.split()[1], 16) % g['ch']:08x}" if l.startswith("PIM_MAC_AB") else l for l in lines]

    trace = _synthetic_trace(tmp_path / "onechannel.trace", program, geometry, mutate=collapse)
    report = trace_check.check_trace(trace, program)
    assert report["totals_match"] and report["ordering_match"] and not report["mapping_match"]
    assert report["mapping"]["channels_addressed"] == 1 and report["mapping"]["channels_declared"] == 16


def test_partial_barrier_group_is_an_error(tmp_path, geometry):
    program = attacc_program(AttAccKernel("gemv", 8, 5120, 384), AttAccTiling.coerce(QKV_NH8), geometry)
    trace = _synthetic_trace(tmp_path / "barrier.trace", program, geometry, mutate=lambda lines: lines[:-1] if False else [l for k, l in enumerate(lines) if not (l.startswith("PIM_BARRIER") and k == next(i for i, x in enumerate(lines) if x.startswith("PIM_BARRIER")))])
    with pytest.raises(ValueError, match="barrier group"):
        trace_check.check_trace(trace, program)


def _attention_rle(program, geometry):
    seq = trace_check.expected_attention_sequence(program.counts, geometry)
    rle = []
    for op in seq:
        if rle and rle[-1][0] == op:
            rle[-1][1] += 1
        else:
            rle.append([op, 1])
    return [tuple(x) for x in rle]


def test_attention_sequence_has_the_generator_structure(geometry):
    """One row block of the paper's worst-loss shape: vector stores, barrier,
    the GB writes, the score MACs, the context MACs with their drains, the
    softmax, the context vector loads, barrier, result stores; totals equal
    the counts."""
    program = attacc_program(AttAccKernel("attention", 8, 128, 480), AttAccTiling.coerce([[8, 1, 1], [2, 4, 16], [1, 16, 30]]), geometry)
    rle = _attention_rle(program, geometry)
    assert rle == [("ST", 1024), ("B", 1), ("PIM_WR_GB", 1024), ("PIM_MAC_AB", 992), ("PIM_MV_SB", 256), ("PIM_SFM", 16), ("PIM_MV_GB", 1024), ("B", 1), ("ST", 1024)]
    import collections
    totals = collections.Counter(op for op in trace_check.expected_attention_sequence(program.counts, geometry) if op != "B")
    assert dict(totals) == dict(program.counts.commands())


def test_attention_sequence_pairs_two_row_blocks(geometry):
    """A program with two row blocks: the generator interleaves the pair
    (both vector loads, one barrier, block 0's rows, block 1's GB writes,
    barrier, block 1's MAC streams, block 0's softmax and loads, barrier,
    block 1's softmax, barrier, block 1's loads, barrier, both result
    stores); totals still equal the counts."""
    from allo.pim.attacc_tiling import attacc_command_counts, AttAccCounts

    kernel = AttAccKernel("attention", 32, 128, 4096)
    tiling = AttAccTiling.coerce([[1, 1, 32], [1, 4, 32], [16, 16, 16]])
    counts = attacc_command_counts(kernel, tiling, geometry)
    if counts.iterations < 2:
        pytest.skip("this tiling fits one row block in the bank")
    import collections
    seq = trace_check.expected_attention_sequence(counts, geometry)
    assert seq.count("B") == 5 * (counts.iterations // 2) + 2 * (counts.iterations % 2)
    totals = collections.Counter(op for op in seq if op != "B")
    assert dict(totals) == dict(counts.commands())


@pytest.mark.skipif(not os.environ.get("ATTACC_DCC_ROOT"), reason="needs a DCC checkout (ATTACC_DCC_ROOT)")
def test_dcc_generator_trace_matches_the_selected_qkv_program(tmp_path, geometry):
    """DCC's own GEMV trace for the paper's QKV batch-1 selection has the
    ordering and mapping the compiled program declares."""
    from benchmarks.attacc.scripts.run_ramulator import _dcc_root, write_trace

    program = attacc_program(AttAccKernel("gemv", 8, 5120, 384), AttAccTiling.coerce(QKV_NH8), geometry)
    trace = write_trace(_dcc_root(), "gemv", program.tiling.dcc_tiling(), tmp_path)
    report = trace_check.check_trace(trace, program)
    assert report["totals_match"] and report["ordering_match"] and report["mapping_match"], report
    assert dict(program.counts.commands()) == {"PIM_WR_GB": 5120, "PIM_MAC_AB": 15360, "PIM_MV_SB": 3072, "ST": 17408}


@pytest.mark.skipif(not os.environ.get("ATTACC_DCC_ROOT"), reason="needs a DCC checkout (ATTACC_DCC_ROOT)")
@pytest.mark.parametrize("dims, tiling", [((8, 128, 480), [[8, 1, 1], [2, 4, 16], [1, 16, 30]]), ((32, 128, 4096), [[1, 32, 1], [1, 2, 64], [16, 1, 256]]), ((8, 128, 144), [[8, 1, 1], [2, 4, 16], [1, 16, 9]])])
def test_dcc_generator_attention_trace_matches_the_program(tmp_path, geometry, dims, tiling):
    """DCC's own attention traces have the ordering and mapping the compiled
    program declares (the paper's worst-loss shape, a 32-head golden shape
    and the shortest length)."""
    from benchmarks.attacc.scripts.run_ramulator import _dcc_root, write_trace

    program = attacc_program(AttAccKernel("attention", *dims), AttAccTiling.coerce(tiling), geometry)
    trace = write_trace(_dcc_root(), "attention", program.tiling.dcc_tiling(), tmp_path)
    report = trace_check.check_trace(trace, program)
    assert report["totals_match"] and report["ordering_match"] and report["mapping_match"], report
