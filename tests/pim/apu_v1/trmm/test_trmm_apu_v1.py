# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench trmm on APU v1 -- Tier-2 triangular REAL-DEVICE cell (task 008).

Thin cell: build the apu_v1 target from `lib`, run the shared `trmm` workload
(the bare MAC-reduction core; triangular/orthogonalization access applied
host-side against the numpy ref, mirroring syrk) on the REAL Gemini board through
`lib.cell.run_cell` (serializes board access, runs via `_run_apu_v1`, checks real
numerics vs the ref), and record the verdict + real-device cycles + results
artifacts. Declares zero hardware. Board gated by `@pytest.mark.apu_v1_device` +
the device fixture; unreachable -> BLOCKED-DEVICE, NEVER sim-substituted.
"""

from __future__ import annotations

import pytest

from lib import cell, reference
from lib.shapes import shape

from workloads import trmm as _wl

_KERNEL = "trmm"
_TARGET = "apu_v1"
_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/trmm/test_trmm_apu_v1.py "
    "-p no:cacheprovider -q"
)


@pytest.mark.apu_v1_device
def test_trmm_apu_v1(request, apu_v1_device_gate):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-2 triangular real-device; APU v1 surfaces real numerics from the board. The emitted declarative MAC is the SPEC-018 popcount-LUT pattern, so the board runs real cycles but a general fp16 product output is the SPEC-018b TODO -> CYCLES-ONLY (real cycles), never a fabricated PASS.",
    )
    # First-class recorded verdict (spec Answer 3), never a silent skip.
    assert verdict.status in (reference.PASS, reference.CYCLES_ONLY, reference.BLOCKED_DEVICE), verdict
    if verdict.status in (reference.PASS, reference.CYCLES_ONLY):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive real-device cycles; got "
            f"{result.cycles!r}; stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_DEVICE:
        assert result.cycles is None
