# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench 3mm on aim -- Tier-1 single-output cell (task 005).

Thin cell: build the target from `lib`, run the shared `three_mm` workload through
`lib.cell.run_cell`, record the verdict + cycles + results.json/RESULTS.md +
COVERAGE.tsv. Declares zero hardware (everything from `lib` + the shared
workload). Verdict derived from what the run surfaces: AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

from workloads import three_mm as _wl

_KERNEL = "3mm"
_TARGET = "aim"
_RUN_CMD = (
    "python -m pytest tests/pim/aim/3mm/test_three_mm_aim.py "
    "-p no:cacheprovider -q"
)


def test_three_mm_aim(request):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-1 single-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).",
    )
    # First-class recorded verdict (spec Answer 3), never a skip.
    assert verdict.status in (reference.CYCLES_ONLY,), verdict
    if verdict.status in (reference.CYCLES_ONLY, reference.PASS) and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
