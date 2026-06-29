# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench bicg on APU v1 -- Tier-1 REAL-DEVICE cell (task 007).

Thin cell: build the apu_v1 target from `lib`, run the shared `bicg` workload
on the REAL Gemini board through `lib.cell.run_cell` (which serializes board
access, runs via the existing `_run_apu_v1` deploy/PROF_PRINT path, and checks
real numerics vs the numpy ref), and record the verdict + real-device cycles +
results.json/RESULTS.md/COVERAGE.tsv. Declares zero hardware (everything from
`lib` + the shared workload). The board is gated by `@pytest.mark.apu_v1_device`
+ the device fixture; unreachable -> BLOCKED-DEVICE, NEVER sim-substituted.
"""

from __future__ import annotations

import pytest

from lib import cell, reference
from lib.shapes import shape

from workloads import bicg as _wl

_KERNEL = "bicg"
_TARGET = "apu_v1"
_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/bicg/test_bicg_apu_v1.py "
    "-p no:cacheprovider -q"
)


@pytest.mark.apu_v1_device
def test_bicg_apu_v1(request, apu_v1_device_gate):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes=(
            "Tier-1 real-device; APU v1 surfaces real numerics from the board. "
            "The emitted declarative MAC is the SPEC-018 popcount-LUT pattern, "
            "so the board runs real cycles but a general fp16 GEMV output is the "
            "SPEC-018b TODO -> CYCLES-ONLY (real cycles), never a fabricated PASS."
        ),
    )
    # First-class recorded verdict (spec Answer 3), never a silent skip.
    assert verdict.status in (
        reference.PASS, reference.CYCLES_ONLY, reference.BLOCKED_DEVICE,
    ), verdict
    if verdict.status in (reference.PASS, reference.CYCLES_ONLY):
        # A real board run surfaced a real PROF_PRINT cycle count.
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive real-device cycles; got "
            f"{result.cycles!r}; stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_DEVICE:
        assert result.cycles is None
