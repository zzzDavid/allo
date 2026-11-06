# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check an AttAcc command trace, command by command, against the selected
program.

EVALUATION ONLY (imports nothing from DCC).  The compiler's selected AttAcc
program is an :class:`allo.pim.attacc_tiling.AttAccProgram` (kernel, tiling,
command inventory, geometry); the executable the simulator times is the
trace the external DCC generator writes for that tiling.  Equal opcode
totals, or equal numbers of distinct addresses, do not establish that the
trace executes the selected program.  This module derives, from the program
and the target geometry alone, the complete command stream the program
implies, opcode *and address* for every command including the stores, the
bank-group drains, the softmax and context moves and the per-channel
barriers, and compares it with the trace line by line.

What the derivation encodes (the AttAcc trace format and the device's
address granules, Ramulator2-AttAcc ``HBM3-PIM`` address mapping):

* Address granules from the geometry: column (32 B), row, bank, bank group
  (``banks_per_group`` banks), channel.  A command's address names the
  channel (``lch * channel``), the bank (``bank * bank``) or bank group
  (``group * bank_group``) and a byte offset inside the bank.
* Per row block (``iterations`` blocks of ``rows_per_iteration`` rows) the
  bank holds three regions sized to whole columns: the vector (one row
  block of the reduction extent), the matrix tile (``rows x reduction x
  outputs``) and the result (``rows x outputs`` for a GEMV, ``rows x
  reduction`` for attention scoring).  The generator places the vector at
  offset 0, the matrix at ``matrix_bytes`` and the result at ``matrix_bytes
  + vector_bytes``, so the result region lies inside the address range the
  matrix columns walk; that placement is what the simulator timed and is
  reported as such (``regions``).  The block must fit the bank
  (``iterations`` is the smallest count for which the three sizes do; the
  same rule the compiler's counts use).
* The commands of one row block, in emission order, with the fan-out
  ``channels_used x units_used`` of the (power-of-two aligned) tiling:
  vector stores (``ST``, one per column of the block's vector per unit per
  channel), the per-row ``PIM_WR_GB`` groups (one per 16-lane position of
  the row's reduction, addressed at the vector region plus the position),
  the ``PIM_MAC_AB`` stream (one per output per 16-lane reduction column,
  addressed at the matrix region column by column) with a ``PIM_MV_SB``
  drain per bank group after every 16 outputs, and the result stores.
  Attention scoring adds the context stream (``PIM_MAC_AB`` per reduction
  element per 16-lane output column with its drains at the result region),
  the ``PIM_SFM`` of every row-unit slot on every channel, and the
  ``PIM_MV_GB`` loads of the probabilities (vector region, per position).
  A barrier is one ``PIM_BARRIER`` per channel, channels in order.
* The generator's composition of the blocks, including its quirks: the
  vector stores precede a barrier; each row's GB write group is followed
  by a barrier and the whole MAC stream and another barrier (GEMV); for
  attention the GB group and the MAC stream of a row follow without a
  barrier, the score phase's drains are not emitted, the context MACs
  precede the softmax, and two consecutive row blocks are interleaved.
  That order is what the simulator times, so it is what is checked.

The check reports the first differing command (index, expected and actual
opcode and address), the region layout and whether it fits the bank, the
count of commands addressing a channel or bank outside the device, and a
summary of the decoded fan-outs.  ``check_trace`` passes only when every
command matches.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

BARRIER = "PIM_BARRIER"


def _ceil(a, b):
    return -(-a // b)


def granules(geometry) -> dict:
    """Byte granules of the trace address space from the target geometry:
    column, row, bank, bank group, channel."""
    col = geometry.column_bytes
    row = geometry.columns * col
    bank = geometry.rows * row
    group = geometry.banks_per_group * bank
    channel = geometry.units * bank
    return {"col": col, "row": row, "ba": bank, "bg": group, "ch": channel}


def regions(program) -> dict:
    """Per row block: the vector, matrix and result regions (base, bytes)
    inside a bank, rounded to whole columns, and whether they fit."""
    counts, geometry, kind = program.counts, program.geometry, program.kernel.kind
    it = counts.rows_per_iteration
    s1, s2 = counts.serial_reduction, counts.serial_outputs
    col, size = geometry.column_bytes, geometry.element_bytes
    vector = _ceil(it * s1 * size, col) * col
    matrix = _ceil(it * s1 * s2 * size, col) * col
    result_elements = s2 if kind == "gemv" else s1
    result = _ceil(it * result_elements * size, col) * col
    return {
        "vector": (0, vector),
        "matrix": (matrix, matrix),
        "result": (matrix + vector, result),
        "footprint": vector + matrix + result,
        "bank_bytes": geometry.bank_bytes,
        "fits_bank": vector + matrix + result <= geometry.bank_bytes,
        "result_inside_matrix_range": matrix + vector < 2 * matrix,
    }


def parse_trace(path) -> list[tuple[str, int]]:
    out = []
    with Path(path).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            out.append((fields[0], int(fields[1], 16) if len(fields) > 1 else 0))
    return out


def expected_trace(program) -> list[tuple[str, int]]:
    """The complete ``(opcode, address)`` stream the selected program implies
    (see the module docstring)."""
    counts, geometry, kind = program.counts, program.geometry, program.kernel.kind
    g = granules(geometry)
    lanes, bpg = geometry.lanes, geometry.banks_per_group
    n, it = counts.iterations, counts.rows_per_iteration
    ch, ba = counts.channels_used, counts.units_used
    s1, s2 = counts.serial_reduction, counts.serial_outputs
    reg = regions(program)
    vec_base, mat_base, ret_base = reg["vector"][0], reg["matrix"][0], reg["result"][0]
    channels = range(ch)
    units = range(ba)
    groups = range(_ceil(ba, bpg))

    def barrier():
        return [(BARRIER, lch * g["ch"]) for lch in range(geometry.channels)]

    def stores(base, elements):
        return [
            ("ST", base + lch * g["ch"] + unit * g["ba"] + pos * g["col"])
            for pos in range(_ceil(it * elements, lanes))
            for unit in units
            for lch in channels
        ]

    def positions(opcode, row):
        out = []
        for pos2 in range(s1):
            pos = row * s1 + pos2
            if pos % lanes == 0:
                out += [
                    (opcode, vec_base + lch * g["ch"] + unit * g["ba"] + pos)
                    for unit in units
                    for lch in channels
                ]
        return out

    def drains(base):
        return [
            ("PIM_MV_SB", base + lch * g["ch"] + group * g["bg"])
            for group in groups
            for lch in channels
        ]

    if kind == "gemv":
        mac = []
        for pos2 in range(s2):
            for pos1 in range(_ceil(s1, lanes)):
                pos = pos1 * s2 + pos2
                mac += [
                    ("PIM_MAC_AB", mat_base + lch * g["ch"] + pos * g["col"])
                    for lch in channels
                ]
            if pos2 % lanes == lanes - 1 or pos2 == s2 - 1:
                mac += drains(ret_base)
        seq: list[tuple[str, int]] = []
        for _block in range(n):
            seq += stores(vec_base, s1)
            seq += barrier()
            for row in range(it):
                seq += positions("PIM_WR_GB", row)
                seq += barrier()
                seq += mac
                seq += barrier()
            seq += barrier()
            seq += stores(ret_base, s2)
        return seq

    # attention scoring
    mac = []
    for pos2 in range(s2):  # score: q . K^T, drains not emitted
        for pos1 in range(_ceil(s1, lanes)):
            pos = pos1 + pos2 * _ceil(s1, lanes)
            mac += [
                ("PIM_MAC_AB", mat_base + lch * g["ch"] + pos * g["col"])
                for lch in channels
            ]
    for pos2 in range(s1):  # context: p . V
        for pos1 in range(_ceil(s2, lanes)):
            pos = pos1 + pos2 * _ceil(s2, lanes)
            mac += [
                ("PIM_MAC_AB", mat_base + lch * g["ch"] + pos * g["col"])
                for lch in channels
            ]
        if pos2 % lanes == lanes - 1 or pos2 == s1 - 1:
            mac += drains(ret_base)
    row_units = counts.sfm // n // geometry.channels if n else 0
    softmax = [
        ("PIM_SFM", lch * g["ch"] + slot * (geometry.units // row_units) * g["ba"])
        for slot in range(row_units)
        for lch in range(geometry.channels)
    ]
    seq = []
    for block in range(0, n, 2):
        has_next = block + 1 < n
        seq += stores(vec_base, s1)
        if has_next:
            seq += stores(vec_base, s1)
        seq += barrier()
        for row in range(it):
            seq += positions("PIM_WR_GB", row)
            seq += mac
        if has_next:
            for row in range(it):
                seq += positions("PIM_WR_GB", row)
            seq += barrier()
            for _row in range(it):
                seq += mac
        seq += softmax
        for row in range(it):
            seq += positions("PIM_MV_GB", row)
        if has_next:
            seq += barrier()
            seq += softmax
            seq += barrier()
            for row in range(it):
                seq += positions("PIM_MV_GB", row)
        seq += barrier()
        seq += stores(ret_base, s1)
        if has_next:
            seq += stores(ret_base, s1)
    return seq


def write_expected_trace(program, path) -> Path:
    """Write the derived stream in the trace format (for tests and probes)."""
    path = Path(path)
    path.write_text(
        "\n".join(f"{opcode} 0x{addr:08x}" for opcode, addr in expected_trace(program))
    )
    return path


def _summary(commands, geometry) -> dict:
    g = granules(geometry)
    channels = collections.Counter()
    out_of_device = 0
    for opcode, addr in commands:
        lch = addr // g["ch"]
        unit = (addr % g["ch"]) // g["ba"]
        channels[lch] += 1
        if lch >= geometry.channels or unit >= geometry.units:
            out_of_device += 1
    return {
        "channels_addressed": len(channels),
        "units_addressed": len(
            {
                (addr % g["ch"]) // g["ba"]
                for opcode, addr in commands
                if opcode in ("PIM_WR_GB", "ST", "PIM_MV_GB")
            }
        ),
        "out_of_device_commands": out_of_device,
    }


def check_trace(path, program) -> dict:
    """Compare the trace at ``path`` with the stream ``program`` implies.

    ``match`` is true only when every command's opcode and address equal
    the derivation's; ``totals_match`` and the first difference are
    reported for diagnosis."""
    actual = parse_trace(path)
    expected = expected_trace(program)
    declared = dict(program.counts.commands())
    totals = collections.Counter(opcode for opcode, _ in actual if opcode != BARRIER)
    first = None
    for index, (exp, act) in enumerate(zip(expected, actual)):
        if exp != act:
            first = {
                "index": index,
                "expected": [exp[0], f"0x{exp[1]:08x}"],
                "actual": [act[0], f"0x{act[1]:08x}"],
            }
            break
    if first is None and len(expected) != len(actual):
        index = min(len(expected), len(actual))
        first = {
            "index": index,
            "expected": (
                [expected[index][0], f"0x{expected[index][1]:08x}"]
                if index < len(expected)
                else None
            ),
            "actual": (
                [actual[index][0], f"0x{actual[index][1]:08x}"]
                if index < len(actual)
                else None
            ),
        }
    reg = regions(program)
    return {
        "match": first is None,
        "commands": len(actual),
        "expected_commands": len(expected),
        "first_difference": first,
        "totals": dict(totals),
        "declared": declared,
        "totals_match": dict(totals) == declared,
        "regions": {k: v for k, v in reg.items()},
        "summary": _summary(actual, program.geometry),
        "declared_fan_out": {
            "channels": program.counts.channels_used,
            "units": program.counts.units_used,
        },
    }


__all__ = [
    "check_trace",
    "expected_trace",
    "granules",
    "parse_trace",
    "regions",
    "write_expected_trace",
]
