# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung F2 bank-swizzle probe: autoscheduled XOR layout vs identity.

The paper cycles for the identity and autoscheduled XOR layouts (two banks,
two tiles, one column) are the ``xor_probe_*`` rows of ``PAPER_GOLDEN``.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from allo.pim.samsung_layout_swizzle import (
    SamsungBankGeometry,
    build_identity_layout,
    format_simulator_trace,
    select_autoscheduled_xor_layout,
)
from allo.spmw_simenv import pimsim_root as _pimsim_root

GEOMETRY = SamsungBankGeometry(bank_count=2, tile_count=2, column_count=1)


def _paper_golden_rows():
    path = Path(__file__).resolve().parents[2] / "test_paper_golden.py"
    name = "layout_swizzle_paper_golden"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GOLDEN = _paper_golden_rows()
EXPECTED_CYCLES = {
    "identity": _GOLDEN.golden_row("xor_probe_identity").expected,
    "xor": _GOLDEN.golden_row("xor_probe_synthesized").expected,
}

_ALLO_ROOT = Path(__file__).resolve().parents[4]
_PROBE_SCRIPT = _ALLO_ROOT / "scripts" / "run_pimsim_layout_trace.py"
_CYCLES = re.compile(r"^PIM_LAYOUT_CYCLES\s+cycles=(\d+)", re.MULTILINE)


def test_autoscheduled_xor_layout_is_conflict_free():
    started = time.perf_counter()
    selection = select_autoscheduled_xor_layout(GEOMETRY)
    assert time.perf_counter() - started < 120.0
    assert selection.layout.describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    )
    assert not build_identity_layout(GEOMETRY).describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    )


def _run_probe(root: Path, trace: Path) -> int:
    completed = subprocess.run(
        [sys.executable, str(_PROBE_SCRIPT), "--pimsim-root", str(root), str(trace)],
        cwd=_ALLO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined[-2000:]
    match = _CYCLES.search(combined)
    assert match is not None, combined[-2000:]
    return int(match.group(1))


def test_identity_and_xor_traces_run_on_pimsimulator(tmp_path):
    root = _pimsim_root()
    library = root / "libdramsim" / "libdramsim2.a"
    if not library.is_file():
        pytest.skip(f"environment: PIMSimulator static library missing at {library}")
    if shutil.which("g++") is None:
        pytest.skip("environment: g++ is not on PATH")

    layouts = {
        "identity": build_identity_layout(GEOMETRY),
        "xor": select_autoscheduled_xor_layout(GEOMETRY).layout,
    }
    cycles = {}
    for name, layout in layouts.items():
        trace = tmp_path / f"{name}.trace"
        trace.write_text(format_simulator_trace(layout, GEOMETRY), encoding="utf-8")
        cycles[name] = _run_probe(root, trace)

    assert cycles == EXPECTED_CYCLES
