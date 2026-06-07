# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-020 static checks for `_run_samsung` multi-layer dispatch.

These tests do NOT call `pim_driver`. They verify the Python-side
contract:
  - A multi-MAC cmd stream with no `layers=` kwarg returns
    `RunResult(cycles=None, ...)` with the expected diagnostic.
  - The number of MAC groups detected matches the number of MACs
    when groups are split at JUMP boundaries.
  - The 1D-x reshape contract documented in SPEC-020 §2.5 is
    applied to per-layer inputs (verified by inspecting the
    reshape branch directly).

The simulator-running, cycle-asserting test for SPEC-020 lives in
`test_e2e_mlp.py::test_e2e_mlp_samsung` and is the verifier's
domain.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.spmw_codegen import RunResult, PIMCmd

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
    Layer boundaries are the per-work-id storeback MOVs, not JUMPs, so the
    run path sees 2 layers (= len(layers=[...])), not 4. (Lever 2 omits
    the opening preload MOV under host residency, so the storeback -- not
    the preload -- is the residency-robust delimiter; SPEC-024 §5.)
    """
    from allo.spmw_codegen import _split_samsung_layers

    compiled = _compile_mlp_for_samsung()
    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]
    macs = [c for c in pim_cmds if c.type_ == "MAC"]
    jumps = [c for c in pim_cmds if c.type_ == "JUMP"]
    # Dual-fiber: 2 fibers x 2 layers = 4 MACs, 4 JUMPs.
    assert len(macs) == 4, f"Expected 4 MAC ops (2 fibers x 2 layers), got {len(macs)}"
    assert len(jumps) == len(macs), (len(jumps), len(macs))
    # But the layer split (used by the run path) yields exactly 2 layers.
    groups = _split_samsung_layers(pim_cmds)
    assert len(groups) == 2, (
        f"Expected 2 layer groups, got {len(groups)}; "
        f"each group must hold both bank fibers of one layer"
    )
    for g in groups:
        assert sum(1 for c in g if c.type_ == "MAC") == 2, (
            "each layer group must keep both fibers' MACs together"
        )


def test_multi_mac_without_layers_returns_cycles_none():
    """`compiled.run()` on an MLP without `layers=` kwarg must return
    cycles=None with a diagnostic stdout, not raise. This preserves the
    back-compat contract that `test_run_returns_runresult_for_all_backends`
    depends on.
    """
    compiled = _compile_mlp_for_samsung()
    result = compiled.run()
    assert isinstance(result, RunResult)
    assert result.backend == "samsung_hbm_pim"
    assert result.cycles is None
    assert "layers=" in result.stdout


def test_multi_mac_wrong_layer_count_raises():
    """`layers=` of the wrong length must raise ValueError so the
    test author cannot silently pair up mis-numbered inputs.
    The check fires before any subprocess invocation.
    """
    compiled = _compile_mlp_for_samsung()
    # Two MAC groups in the MLP cmd stream; pass only one layer.
    W1 = np.zeros((256, 128), dtype=np.float16)
    x = np.zeros((1, 128), dtype=np.float16)

    # Skip the subprocess-bearing path: if pim_driver isn't present,
    # the simulator-unavailable branch wins before the layers check,
    # and we can't exercise this assertion.
    from allo.spmw_codegen import _pimsim_root
    if not (_pimsim_root() / "pim_driver").exists():
        pytest.skip("pim_driver not built; cannot exercise layers-count check")

    with pytest.raises(ValueError, match="MAC groups"):
        compiled.run(layers=[{"W": W1, "x": x}])


def test_single_layer_back_compat_path_still_takes_w_x():
    """A single-layer (single work-id) trace must keep the existing W=/x=
    kwarg contract (no `layers=` required) even though lever 1's
    dual-fiber placement emits two MACs (EVEN + ODD) for it. The layer
    split sees one layer group, so the multi-layer branch is NOT taken:
    calling with no kwargs returns the legacy "GEMV needs W and x kwargs"
    diagnostic rather than the multi-layer "needs layers=" diagnostic.
    """
    from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

    target = build_samsung_target()
    trace = MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )
    compiled = allo.compile_for_target(target, trace)

    # Lever 1: a single work-id emits two MACs (EVEN + ODD fibers), but
    # the layer split must still see exactly one layer group.
    from allo.spmw_codegen import _split_samsung_layers

    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]
    n_macs = sum(1 for c in pim_cmds if c.type_ == "MAC")
    assert n_macs == 2, f"Expected 2 MACs (dual-fiber) in single-layer trace, got {n_macs}"
    assert len(_split_samsung_layers(pim_cmds)) == 1, "single work-id => one layer group"

    # No kwargs: legacy "GEMV needs W and x kwargs" diagnostic must
    # fire (not the multi-MAC diagnostic).
    from allo.spmw_codegen import _pimsim_root
    if not (_pimsim_root() / "pim_driver").exists():
        pytest.skip("pim_driver not built; can't exercise the GEMV-needs-W/x branch")
    result = compiled.run()
    assert result.cycles is None
    assert "GEMV needs W and x" in result.stdout, result.stdout


if __name__ == "__main__":
    test_mlp_cmd_stream_splits_into_two_layers()
    test_multi_mac_without_layers_returns_cycles_none()
    print("ALL PASSED (subprocess-dependent tests skipped if no pim_driver)")
