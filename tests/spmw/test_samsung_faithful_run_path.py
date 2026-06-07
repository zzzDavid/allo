# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-021 task 025 acceptance gate: the faithful Samsung run path.

Verdict (a) from SPEC-021: the legacy `--cmds` path (executeGemvWithCmds)
fixes cycles by data shape -- two streams at the same (M,K) yield identical
getCycle(). The faithful path (`--faithful` -> executeGemvFaithful) makes the
issued PIM transaction queue track the emitted cmd stream, so:

  * a redundant stream (extra MAC bodies / host-preload MOVs) costs MORE,
  * the folded/native stream costs the minimum (== the reference WithCmds
    queue, byte-for-byte),
  * numerics are unchanged (faithful output == reference WithCmds output).

These tests drive `pim_driver` directly so they exercise the C++ faithful
routine the Python run path now selects.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from allo.spmw_codegen import _pimsim_root


def _pim_driver_present() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


pytestmark = pytest.mark.skipif(
    not _pim_driver_present(),
    reason="PIMSimulator pim_driver binary not built",
)

# Folded / native canonical GEMV CRF: one even MAC + one odd MAC, each folded
# by a JUMP. This is the minimum-work stream.
_FOLDED = (
    "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
    "JUMP loop_counter=7 loop_offset=2\n"
    "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
    "JUMP loop_counter=7 loop_offset=2\n"
    "NOP loop_counter=7\n"
    "EXIT\n"
)

# Redundant stream: same logical GEMV but re-MACs each bank path twice and
# adds host-preload MOVs the native HAB-broadcast path does not pay as CRF
# instructions. Strictly more issued PIM work.
_REDUNDANT = (
    "MOV dst=GRF_A src0=EVEN_BANK\n"
    "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
    "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
    "JUMP loop_counter=7 loop_offset=2\n"
    "MOV dst=GRF_A src0=ODD_BANK\n"
    "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
    "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
    "JUMP loop_counter=7 loop_offset=2\n"
    "NOP loop_counter=7\n"
    "EXIT\n"
)

M, K = 4096, 1024


def _run(tmp_path: Path, cmds_text: str, faithful: bool):
    import numpy as np

    root = _pimsim_root()
    driver = root / "pim_driver"
    w_path = tmp_path / "W.npy"
    x_path = tmp_path / "x.npy"
    out_path = tmp_path / "out.bin"
    cmds_path = tmp_path / "cmds.txt"
    np.save(w_path, np.zeros((M, K), dtype=np.float16))
    np.save(x_path, np.zeros((1, K), dtype=np.float16))
    cmds_path.write_text(cmds_text)

    argv = [
        str(driver),
        "--op", "GEMV",
        "--weight", str(w_path),
        "--in", str(x_path),
        "--output-dim", str(M),
        "--input-dim", str(K),
        "--out", str(out_path),
        "--cmds", str(cmds_path),
    ]
    if faithful:
        argv.append("--faithful")

    proc = subprocess.run(
        argv, capture_output=True, cwd=str(root), timeout=600, check=False
    )
    combined = proc.stdout.decode() + proc.stderr.decode()
    m = re.search(r"PIM_CYCLES total=(\d+)", combined)
    assert m, f"no PIM_CYCLES line; tail: {combined[-400:]}"
    return int(m.group(1)), out_path.read_bytes()


def test_faithful_redundant_stream_costs_more(tmp_path):
    """Acceptance gate: two streams at the SAME (M,K), faithful path, yield
    DIFFERENT cycles -- the optimized (folded) stream strictly lower."""
    folded_cyc, _ = _run(_mk(tmp_path, "folded"), _FOLDED, True)
    redundant_cyc, _ = _run(_mk(tmp_path, "redundant"), _REDUNDANT, True)

    assert folded_cyc != redundant_cyc, (
        "faithful path must differentiate streams at one shape; "
        f"both = {folded_cyc}"
    )
    assert folded_cyc < redundant_cyc, (
        f"optimized/folded stream must cost fewer cycles; "
        f"folded={folded_cyc} redundant={redundant_cyc}"
    )


def test_legacy_path_is_shape_fixed(tmp_path):
    """SPEC-021 verdict (a): WITHOUT --faithful, the same two streams yield
    identical cycles (shape-fixed). This is the baseline the faithful path
    breaks out of."""
    folded_cyc, _ = _run(_mk(tmp_path, "folded"), _FOLDED, False)
    redundant_cyc, _ = _run(_mk(tmp_path, "redundant"), _REDUNDANT, False)
    assert folded_cyc == redundant_cyc, (
        "legacy --cmds path should be shape-fixed (verdict a); "
        f"folded={folded_cyc} redundant={redundant_cyc}"
    )


def test_faithful_native_matches_reference_queue(tmp_path):
    """The faithful path on the folded/native stream issues the byte-for-byte
    same transaction queue as the reference executeGemvWithCmds -- so the
    native baseline is unchanged (no flattering), and numerics are identical."""
    ref_cyc, ref_out = _run(_mk(tmp_path, "ref"), _FOLDED, False)
    faith_cyc, faith_out = _run(_mk(tmp_path, "faith"), _FOLDED, True)
    assert ref_cyc == faith_cyc, (
        "faithful path must not change the native folded-stream cycle count; "
        f"reference={ref_cyc} faithful={faith_cyc}"
    )
    assert ref_out == faith_out, (
        "faithful path must produce byte-identical GEMV output to the "
        "reference executeGemvWithCmds (numerics untouched)"
    )


def _mk(base: Path, name: str) -> Path:
    p = base / name
    p.mkdir(parents=True, exist_ok=True)
    return p
