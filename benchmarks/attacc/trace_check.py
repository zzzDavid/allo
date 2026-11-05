# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural check of an AttAcc command trace against the selected program.

EVALUATION ONLY (imports nothing from DCC).  The compiler's selected AttAcc
program is an :class:`allo.pim.attacc_tiling.AttAccProgram` (kernel, tiling,
command inventory); the executable the simulator times is the trace the
external DCC generator writes for that tiling.  Equal opcode totals do not
establish that the trace executes the selected program: this module checks
the trace's *ordering* and *address mapping* against what the program
declares.

* Ordering: the trace is the generator's phase structure (phases separated
  by one ``PIM_BARRIER`` per channel): per row block, the vector stores, a
  barrier, then for every row of the block the ``PIM_WR_GB`` group of that
  row, a barrier, the ``PIM_MAC_AB`` stream with its ``PIM_MV_SB`` drains, a
  barrier; then a barrier and the result stores.  The expected opcode
  sequence is re-derived here from the program's counts record
  (``iterations``, ``rows_per_iteration``, ``channels_used``,
  ``units_used``, ``serial_reduction``, ``serial_outputs``) and compared
  with the trace command by command.
* Mapping: the addresses are decoded with the target geometry's granules
  (column, row, bank, channel); the distinct channels the MACs address,
  the distinct banks the GB writes address, the distinct matrix columns per
  channel and the distinct vector positions must equal the program's
  fan-outs and serial extents.

The attention (scoring) trace has its own phase structure (the generator
pairs consecutive row blocks; per block the vector stores, a barrier, then
per row the ``PIM_WR_GB`` group followed by the score ``PIM_MAC_AB`` stream
and the context ``PIM_MAC_AB`` stream with its ``PIM_MV_SB`` drains, then
the ``PIM_SFM`` of every row-unit slot on every channel, the ``PIM_MV_GB``
groups, a barrier and the result stores); it is re-derived the same way.
Both sequences describe the generator's emission order, including its
quirks (the context MACs precede the softmax; the score phase's bank-group
drains are not emitted), because that order is what the simulator times.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

PIM_OPCODES = ("PIM_WR_GB", "PIM_MAC_AB", "PIM_MV_SB", "PIM_MV_GB", "PIM_SFM", "ST", "LD")
BARRIER = "PIM_BARRIER"


def granules(geometry) -> dict:
    """Byte granules of the trace address space (DCC's ``HBM_GS``), from the
    target geometry: column, row, bank, channel."""
    col = geometry.column_bytes
    row = geometry.columns * col
    bank = geometry.rows * row
    channel = geometry.units * bank
    return {"col": col, "row": row, "ba": bank, "ch": channel}


def parse_trace(path) -> list[tuple[str, int]]:
    out = []
    with Path(path).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            opcode = fields[0]
            addr = int(fields[1], 16) if len(fields) > 1 else 0
            out.append((opcode, addr))
    return out


def collapse_barriers(commands, channels) -> list[str]:
    """Opcode sequence with each barrier group (one ``PIM_BARRIER`` per
    channel) collapsed to a single ``B`` token; a partial group is an error."""
    out = []
    run = 0
    for opcode, _addr in commands:
        if opcode == BARRIER:
            run += 1
            if run == channels:
                out.append("B")
                run = 0
            continue
        if run:
            raise ValueError(f"barrier group of {run} lines (expected {channels})")
        out.append(opcode)
    if run:
        raise ValueError(f"trailing barrier group of {run} lines (expected {channels})")
    return out


def _ceil(a, b):
    return -(-a // b)


def expected_gemv_sequence(counts, geometry) -> list[str]:
    """The generator's opcode sequence (inference mode) for a GEMV program,
    from the counts record; ``B`` marks a barrier."""
    n, it = counts.iterations, counts.rows_per_iteration
    ch, ba = counts.channels_used, counts.units_used
    s1, s2 = counts.serial_reduction, counts.serial_outputs
    lanes, bpg = geometry.lanes, geometry.banks_per_group
    seq: list[str] = []
    for _itr in range(n):
        # vector load (inference mode: stores from the host side)
        seq += ["ST"] * (_ceil(it * s1, lanes) * ba * ch)
        seq.append("B")
        mac_block: list[str] = []
        for pos2 in range(s2):
            mac_block += ["PIM_MAC_AB"] * (_ceil(s1, lanes) * ch)
            if pos2 % lanes == lanes - 1 or pos2 == s2 - 1:
                mac_block += ["PIM_MV_SB"] * (_ceil(ba, bpg) * ch)
        for j in range(it):
            first = _ceil(j * s1, lanes) * lanes
            positions = len(range(first, (j + 1) * s1, lanes))
            seq += ["PIM_WR_GB"] * (positions * ba * ch)
            seq.append("B")
            seq += mac_block
            seq.append("B")
        seq.append("B")
        seq += ["ST"] * (_ceil(it * s2, lanes) * ba * ch)
    return seq


def expected_attention_sequence(counts, geometry) -> list[str]:
    """The generator's opcode sequence (inference mode) for an attention
    scoring program, from the counts record; ``B`` marks a barrier."""
    n, it = counts.iterations, counts.rows_per_iteration
    ch, ba = counts.channels_used, counts.units_used
    s1, s2 = counts.serial_reduction, counts.serial_outputs
    lanes, bpg = geometry.lanes, geometry.banks_per_group
    channels = geometry.channels
    vector_stores = _ceil(it * s1, lanes) * ba * ch

    def positions(j):
        first = _ceil(j * s1, lanes) * lanes
        return len(range(first, (j + 1) * s1, lanes)) * ba * ch

    mac_list: list[str] = []
    mac_list += ["PIM_MAC_AB"] * (s2 * _ceil(s1, lanes) * ch)  # score (its drains are not emitted)
    for pos2 in range(s1):  # context
        mac_list += ["PIM_MAC_AB"] * (_ceil(s2, lanes) * ch)
        if pos2 % lanes == lanes - 1 or pos2 == s1 - 1:
            mac_list += ["PIM_MV_SB"] * (_ceil(ba, bpg) * ch)
    softmax = ["PIM_SFM"] * (counts.sfm // n) if n else []
    seq: list[str] = []
    for itr in range(0, n, 2):
        has_next = itr + 1 < n
        seq += ["ST"] * vector_stores
        if has_next:
            seq += ["ST"] * vector_stores
        seq.append("B")
        for j in range(it):
            seq += ["PIM_WR_GB"] * positions(j)
            seq += mac_list
        if has_next:
            for j in range(it):
                seq += ["PIM_WR_GB"] * positions(j)
            seq.append("B")
            for _j in range(it):
                seq += mac_list
        seq += softmax
        for j in range(it):
            seq += ["PIM_MV_GB"] * positions(j)
        if has_next:
            seq.append("B")
            seq += softmax
            seq.append("B")
            for j in range(it):
                seq += ["PIM_MV_GB"] * positions(j)
        seq.append("B")
        seq += ["ST"] * vector_stores
        if has_next:
            seq += ["ST"] * vector_stores
    return seq


def _first_difference(expected, actual):
    for index, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            return {"index": index, "expected": a, "actual": b}
    if len(expected) != len(actual):
        return {"index": min(len(expected), len(actual)), "expected": expected[len(actual)] if len(expected) > len(actual) else None, "actual": actual[len(expected)] if len(actual) > len(expected) else None}
    return None


def mapping_report(commands, counts, geometry, kind="gemv") -> dict:
    """Decoded address fan-outs of the trace beside the program's."""
    g = granules(geometry)
    macs = [addr for opcode, addr in commands if opcode == "PIM_MAC_AB"]
    gbs = [addr for opcode, addr in commands if opcode == "PIM_WR_GB"]
    channels = {addr // g["ch"] for addr in macs}
    banks = {(addr % g["ch"]) // g["ba"] for addr in gbs}
    columns_per_channel = collections.defaultdict(set)
    for addr in macs:
        columns_per_channel[addr // g["ch"]].add((addr % g["ba"]) // g["col"])
    positions = {addr % g["ba"] for addr in gbs}
    lanes = geometry.lanes
    expected_columns = counts.serial_outputs * _ceil(counts.serial_reduction, lanes)
    if kind == "attention":
        # score columns and context columns share the matrix base address
        expected_columns = max(expected_columns, counts.serial_reduction * _ceil(counts.serial_outputs, lanes))
    expected_positions = _ceil(counts.rows_per_iteration * counts.serial_reduction, lanes)
    return {
        "channels_addressed": len(channels),
        "channels_declared": counts.channels_used,
        "units_addressed": len(banks),
        "units_declared": counts.units_used,
        "mac_columns_per_channel": sorted({len(v) for v in columns_per_channel.values()}),
        "mac_columns_declared": expected_columns,
        "vector_positions_addressed": len(positions),
        "vector_positions_declared": expected_positions,
        "fan_out_exceeds_device": counts.channels_used > geometry.channels or counts.units_used > geometry.units,
    }


def check_trace(path, program) -> dict:
    """Check the trace at ``path`` against ``program`` (an AttAccProgram)."""
    commands = parse_trace(path)
    counts, geometry = program.counts, program.geometry
    totals = collections.Counter(opcode for opcode, _ in commands if opcode != BARRIER)
    declared = dict(counts.commands())
    report = {
        "totals": dict(totals),
        "totals_match": {k: v for k, v in totals.items() if k != "LD"} == declared,
        "declared": declared,
    }
    kind = program.kernel.kind
    actual = collapse_barriers(commands, geometry.channels)
    expected = (expected_gemv_sequence if kind == "gemv" else expected_attention_sequence)(counts, geometry)
    difference = _first_difference(expected, actual)
    mapping = mapping_report(commands, counts, geometry, kind)
    mapping_match = (
        mapping["channels_addressed"] == mapping["channels_declared"]
        and mapping["units_addressed"] == mapping["units_declared"]
        and mapping["mac_columns_per_channel"] == [mapping["mac_columns_declared"]]
        and mapping["vector_positions_addressed"] == mapping["vector_positions_declared"]
    )
    report.update({
        "ordering_checked": True,
        "ordering_match": difference is None,
        "first_difference": difference,
        "phases": actual.count("B") + 1,
        "commands": len(actual) - actual.count("B"),
        "mapping": mapping,
        "mapping_match": mapping_match,
    })
    return report


__all__ = ["check_trace", "collapse_barriers", "expected_attention_sequence", "expected_gemv_sequence", "granules", "mapping_report", "parse_trace"]
