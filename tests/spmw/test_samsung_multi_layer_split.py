# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static check for the Samsung cmd-stream layer-split helper.

`_split_samsung_layers` groups a flat Samsung GEMV cmd stream into per-work-id
chunks (closed by the storeback MOV). The legacy multi-layer GEMV RUN path that
consumed it was DELETED (SPEC-05, 2026-06-30: all MAC -> GENERIC REDUCE); the
helper is kept as a pure cmd-stream analysis used by the FFN cost-model tests
(test_samsung_loop_012). This test pins the split behaviour itself. The three
former back-compat tests (no-kwargs / wrong-layer-count / W-x diagnostics) were
DELETED with the legacy GEMV run path they exercised.
"""

from __future__ import annotations

import allo
from allo.spmw_codegen import PIMCmd

from _fixtures import build_mlp_workload, build_samsung_target


def _compile_mlp_for_samsung():
    target = build_samsung_target()
    workload = build_mlp_workload()
    schedule = allo.customize(workload, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)
    compiled = allo.compile_for_target(target, trace)
    return compiled


def test_mlp_cmd_stream_splits_into_two_layers():
    """The compiled MLP must split into two per-layer groups even though
    lever 1's dual-fiber placement emits two MACs (EVEN + ODD) per layer.
    Layer boundaries are the per-work-id storeback MOVs, not JUMPs.
    """
    from allo.spmw_codegen import _split_samsung_layers

    compiled = _compile_mlp_for_samsung()
    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]
    macs = [c for c in pim_cmds if c.type_ == "MAC"]
    jumps = [c for c in pim_cmds if c.type_ == "JUMP"]
    # Dual-fiber: 2 fibers x 2 layers = 4 MACs, 4 JUMPs.
    assert len(macs) == 4, f"Expected 4 MAC ops (2 fibers x 2 layers), got {len(macs)}"
    assert len(jumps) == len(macs), (len(jumps), len(macs))
    # The layer split yields exactly 2 layers (one per work-id), both fibers
    # of a layer kept together.
    groups = _split_samsung_layers(pim_cmds)
    assert len(groups) == 2, (
        f"Expected 2 layer groups, got {len(groups)}; "
        f"each group must hold both bank fibers of one layer"
    )
    for g in groups:
        assert sum(1 for c in g if c.type_ == "MAC") == 2, (
            "each layer group must keep both fibers' MACs together"
        )


if __name__ == "__main__":
    test_mlp_cmd_stream_splits_into_two_layers()
    print("ALL PASSED")
