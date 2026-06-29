# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench atax on aim -- Tier-1 multi-output cell (task 006).

Thin cell: build the target from `lib`, run the shared multi-stage `atax`
workload through `lib.cell.run_cell`, record verdict + cycles + results.json +
RESULTS.md + COVERAGE.tsv. Declares zero hardware. Verdict from what the run
surfaces: AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

from workloads import atax as _wl

_KERNEL = "atax"
_TARGET = "aim"
_RUN_CMD = (
    "python -m pytest tests/pim/aim/atax/test_atax_aim.py "
    "-p no:cacheprovider -q"
)


def test_atax_aim(request):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-1 multi-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).",
    )
    assert verdict.status in (reference.CYCLES_ONLY,), verdict
    if verdict.status in (reference.CYCLES_ONLY, reference.PASS) and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
