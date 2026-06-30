# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench bicg on samsung_hbm_pim -- Tier-1 multi-output cell (task 006).

Thin cell: build the target from `lib`, run the shared multi-stage `bicg`
workload through `lib.cell.run_cell`, record verdict + cycles + results.json +
RESULTS.md + COVERAGE.tsv. Declares zero hardware. Verdict from what the run
surfaces: Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM.
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape


_KERNEL = "bicg"
_TARGET = "samsung_hbm_pim"
_RUN_CMD = (
    "python -m pytest tests/pim/samsung_hbm_pim/bicg/test_bicg_samsung_hbm_pim.py "
    "-p no:cacheprovider -q"
)


def test_bicg_samsung_hbm_pim(request):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET,
        folder=request.path.parent, stages=None, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-1 multi-output; Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM.",
    )
    assert verdict.status in (
        reference.PASS, reference.CYCLES_ONLY, reference.BLOCKED_SIM,
    ), verdict
    if verdict.status in (reference.CYCLES_ONLY, reference.PASS) and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
