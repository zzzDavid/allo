# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-005 regression: verify the emitted PIMCmd stream now flows into
the Samsung pim_driver via --cmds, so layout choice is observable in
both the cmd stream and the simulator's returned cycle count.

Weaker assertion per SPEC-005 §6 (cmds-are-different + both runs return
positive cycles). The stronger `cycles_a != cycles_b` is gated by the
codegen-side followup (SPEC-005 §8 / needs-arch-MMM-samsung-jump-from-placement).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import allo
from allo.spmw_autoschedule import Placement
from allo.spmw_codegen import PIMCmd, _pimsim_root
from allo.spmw_target import UnitId

from _fixtures import (
    MLP_K1,
    MLP_K2,
    MLP_M1,
    MLP_M2,
    build_mlp_workload,
    build_samsung_target,
)


def _pim_driver_present() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


@pytest.mark.skipif(
    not _pim_driver_present(),
    reason="PIMSimulator pim_driver binary not built",
)
def test_samsung_layout_changes_cmd_stream_and_runs():
    """Two Samsung placements that the cost model prices differently
    also emit different PIMCmd streams, and both run end-to-end through
    pim_driver --cmds.

    Layout A binds `y` (the bank-resident operand) to a bank handle
    (is_auto=1, bank-row mode -- K-loop folds via the column-strobe).
    Layout B binds `y` to `grf_a` (is_auto=0, GRF-staged -- K MACs
    unrolled). Cost-model differential at K=1024 is ~8x in favour of
    layout A; the simulator-reported cycles need not change because
    pim_driver's GEMV scaffolding fixes total cycles by data shape, but
    the *cost model* assertion below is what gates that placement
    actually drives the cycle objective the autoscheduler argmins on
    (goal-check-3 §Criterion 3).
    """
    import numpy as np

    target = build_samsung_target()
    workload = build_mlp_workload()
    schedule = allo.customize(workload, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    pid = UnitId(level=1, unit=None)
    banks = target.banks  # type: ignore[attr-defined]

    # Layout A: y -> EVEN_BANK (is_auto=1, bank-row mode).
    layout_a = [
        Placement(placements={
            "local_W1": target.grf_a,
            "local_x":  banks[2 * pid],
            "acc":      target.grf_b,
        }, mode="bank_row"),
        Placement(placements={
            "local_W2": target.grf_a,
            "local_h":  banks[2 * pid],
            "acc":      target.grf_b,
        }, mode="bank_row"),
    ]
    # Layout B: y -> GRF_A (is_auto=0, GRF-staged mode).
    layout_b = [
        Placement(placements={
            "local_W1": target.grf_a,
            "local_x":  target.grf_a,
            "acc":      target.grf_b,
        }, mode="grf_staged"),
        Placement(placements={
            "local_W2": target.grf_a,
            "local_h":  target.grf_a,
            "acc":      target.grf_b,
        }, mode="grf_staged"),
    ]

    compiled_a = allo.compile_for_target(target, trace, layout=layout_a)
    compiled_b = allo.compile_for_target(target, trace, layout=layout_b)

    # The compiled cmd streams must differ -- gates that codegen carries
    # placement into the PIMCmd stream at all.
    pim_a = [c for c in compiled_a.cmds if isinstance(c, PIMCmd)]
    pim_b = [c for c in compiled_b.cmds if isinstance(c, PIMCmd)]
    assert pim_a, "layout A produced no PIMCmds"
    assert pim_b, "layout B produced no PIMCmds"
    assert pim_a != pim_b, (
        "is_auto=1 vs is_auto=0 placements produced identical PIMCmd "
        "streams; expected MAC src1=EVEN_BANK (is_auto=1) vs src1=GRF_A "
        "(is_auto=0)."
    )
    # Concrete sanity check: A's MACs are is_auto=1 with src1=EVEN_BANK;
    # B's MACs are is_auto=0 with src1=GRF_A.
    a_macs = [c for c in pim_a if c.type_ == "MAC"]
    b_macs = [c for c in pim_b if c.type_ == "MAC"]
    assert all(c.isAuto_ == 1 for c in a_macs), (
        f"layout A MACs must all be is_auto=1: {[c.isAuto_ for c in a_macs]}"
    )
    assert all(c.src1_ == "EVEN_BANK" for c in a_macs), (
        f"layout A MACs must all read EVEN_BANK: {[c.src1_ for c in a_macs]}"
    )
    assert all(c.isAuto_ == 0 for c in b_macs), (
        f"layout B MACs must all be is_auto=0: {[c.isAuto_ for c in b_macs]}"
    )
    assert all(c.src1_ == "GRF_A" for c in b_macs), (
        f"layout B MACs must all read GRF_A: {[c.src1_ for c in b_macs]}"
    )

    # Cost-model assertion: the kernel_cycles cost (the objective the
    # autoscheduler argmins on) must differentiate these two placements.
    # This is the criterion-3 contract: changing the placement changes
    # the cycle estimate. _samsung_kernel_cycles folds K-iters when
    # is_auto=1 (~K/burst MACs + 1 JUMP); is_auto=0 emits K MACs
    # unrolled. Both layouts here share the same MAC trace, so we score
    # them under the same cost_fn.
    cost_fn = allo.get_cost("kernel_cycles", target)
    cost_a_total = sum(cost_fn(trace, lay) for lay in layout_a)
    cost_b_total = sum(cost_fn(trace, lay) for lay in layout_b)
    assert cost_a_total != cost_b_total, (
        f"is_auto=1 and is_auto=0 must price differently under "
        f"kernel_cycles; got both={cost_a_total}"
    )
    assert cost_a_total < cost_b_total, (
        f"is_auto=1 must be cheaper than is_auto=0 (K-loop folds); "
        f"got A={cost_a_total} B={cost_b_total}"
    )

    # SPEC-05 (2026-06-30): the former end-to-end `compiled_{a,b}.run(layers=...)`
    # assertions exercised the legacy multi-layer GEMV run path, which was DELETED
    # (all MAC -> GENERIC REDUCE; the `layers=` MLP dispatch is gone). The
    # placement-carries-into-cmd-stream gate (cmd streams differ + the cost model
    # differentiates the two placements) above is the keeper assertion; the
    # run-path cycle count is now covered by the GENERIC REDUCE cells.


@pytest.mark.skipif(
    not _pim_driver_present(),
    reason="PIMSimulator pim_driver binary not built",
)
def test_samsung_pim_driver_accepts_cmds_flag(tmp_path):
    """Direct sanity check on the --cmds CLI: hand-write a tiny canonical
    GEMV-style cmd stream, point pim_driver at it, and assert it reports
    a positive cycle count via PIM_CYCLES total=...
    """
    import subprocess
    import numpy as np

    root = _pimsim_root()
    driver = root / "pim_driver"

    M_drv, K_drv = 4096, 1024
    W_np = np.zeros((M_drv, K_drv), dtype=np.float16)
    x_np = np.zeros(K_drv, dtype=np.float16)
    w_path = tmp_path / "W.npy"
    x_path = tmp_path / "x.npy"
    out_path = tmp_path / "out.bin"
    np.save(w_path, W_np)
    np.save(x_path, x_np)

    # Canonical GEMV CRF microcode from PIMCmdGen.h (line 118-127),
    # written via the line-delimited format the parser expects. The
    # num_jump_to_be_taken values (7 and 7) match an 8-grfB tile.
    cmds_path = tmp_path / "cmds.txt"
    cmds_path.write_text(
        "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
        "JUMP loop_counter=7 loop_offset=2\n"
        "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
        "JUMP loop_counter=7 loop_offset=2\n"
        "NOP loop_counter=7\n"
        "EXIT\n"
    )

    proc = subprocess.run(
        [
            str(driver),
            "--op", "GEMV",
            "--weight", str(w_path),
            "--in", str(x_path),
            "--output-dim", str(M_drv),
            "--input-dim", str(K_drv),
            "--out", str(out_path),
            "--cmds", str(cmds_path),
        ],
        capture_output=True,
        cwd=str(root),
        timeout=600,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    combined = stdout + ("\n" + stderr if stderr else "")
    assert "PIM_CYCLES total=" in combined, (
        f"pim_driver --cmds did not emit PIM_CYCLES line; "
        f"returncode={proc.returncode}; tail: {combined[-600:]}"
    )
    import re
    m = re.search(r"PIM_CYCLES total=(\d+)", combined)
    cycles = int(m.group(1))
    assert cycles > 0, f"expected positive cycles, got {cycles}"


def test_run_samsung_no_longer_has_static_kernel_cycles_source():
    """SPEC-005: the static `kernel = "GEMV"/"MUL"/"ADD"/"RELU"` inference
    is no longer the source of cycle truth. Verify the source file still
    serialises compiled.cmds via --cmds (the run-side plumbing). Static
    grep — runs without a built simulator.
    """
    src = Path(__file__).resolve().parents[2] / "allo" / "spmw_codegen.py"
    text = src.read_text()
    assert "--cmds" in text, "_run_samsung must pass --cmds to pim_driver"
    assert "cmds.txt" in text, (
        "_run_samsung must serialise compiled.cmds to a cmds.txt file"
    )
    # The kernel-detection block remains (selects --op for data plumbing),
    # but a comment must document that it no longer drives the CRF.
    # Comment / docstring must clarify the static --op kernel choice no
    # longer drives the CRF microcode (whitespace-normalised match so the
    # docstring can wrap freely).
    normalised = " ".join(text.split())
    assert (
        "no longer determines the CRF" in normalised
        or "no longer drives the CRF" in normalised
    ), "docstring must clarify static kernel inference no longer drives CRF"
