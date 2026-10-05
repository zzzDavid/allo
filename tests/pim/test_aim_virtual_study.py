# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM virtual-architecture study (5-evaluation.tex:285-305): compile-only opcode counts."""
from collections import Counter

import pytest

import allo
from allo.pim.targets import build_aim_target
from benchmarks.cent_aim.workloads import build_case

ROWS = [  # banks.rows, banks.cols, gb entries, MAC_ABK, WR_GB, trace lines
    pytest.param(16384, 1024, 64, 1040, 19, 3312, id="row2KiB_gb2KiB"),
    pytest.param(8192, 2048, 128, 536, 10, 1791, id="row4KiB_gb4KiB"),
    pytest.param(4096, 4096, 256, 268, 5, 982, id="row8KiB_gb8KiB"),
]


@pytest.mark.parametrize("rows,cols,entries,mac,wr_gb,lines", ROWS)
def test_aim_geometry_variant_opcode_counts(rows, cols, entries, mac, wr_gb, lines):
    target = build_aim_target()
    banks, gb = target._handles["banks"], target._handles["gb"]
    banks.geometry["rows"] = banks.rows = rows
    banks.geometry["cols"] = banks.cols = cols
    gb.geometry["entries"] = entries
    trace = str(allo.compile(build_case("ffn_fc_l128"), target).trace).splitlines()
    ops = Counter(line.split()[1] for line in trace if len(line.split()) > 1)
    assert (ops["MAC_ABK"], ops["WR_GB"], len(trace)) == (mac, wr_gb, lines)
