# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench doitgen on upmem -- Tier-1 single-output cell (task 005).

Thin cell: build the target from `lib`, run the shared `doitgen` workload through
`lib.cell.run_cell`, record the verdict + cycles + results.json/RESULTS.md +
COVERAGE.tsv. Declares zero hardware (everything from `lib` + the shared
workload). Verdict derived from what the run surfaces: UPMEM GEMV-host slot verifies W@x internally (PASS w/ cycles); a VA-slot route is CYCLES-ONLY.
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

from workloads import doitgen as _wl

_KERNEL = "doitgen"
_TARGET = "upmem"
_RUN_CMD = (
    "python -m pytest tests/pim/upmem/doitgen/test_doitgen_upmem.py "
    "-p no:cacheprovider -q"
)


def test_doitgen_upmem(request):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-1 single-output; UPMEM GEMV-host slot verifies W@x internally (PASS w/ cycles); a VA-slot route is CYCLES-ONLY.",
    )
    # First-class recorded verdict (spec Answer 3), never a skip.
    assert verdict.status in (reference.PASS, reference.CYCLES_ONLY), verdict
    if verdict.status in (reference.CYCLES_ONLY, reference.PASS) and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
