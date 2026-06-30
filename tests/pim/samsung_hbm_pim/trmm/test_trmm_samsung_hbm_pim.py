# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench trmm on samsung_hbm_pim -- Tier-2 triangular extension cell (task 008).

Thin cell: build the target from `lib`, run the shared `trmm` workload (the
bare MAC-reduction core; triangular/orthogonalization access applied host-side
against the numpy ref, mirroring syrk) through `lib.cell.run_cell`, record the
verdict + cycles + results.json/RESULTS.md + COVERAGE.tsv. Declares zero hardware
(everything from `lib` + the shared workload). The bare contraction matches MAC
on the existing matcher -- no matcher change needed.
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

_KERNEL = "trmm"
_TARGET = "samsung_hbm_pim"
_RUN_CMD = (
    "python -m pytest tests/pim/samsung_hbm_pim/trmm/test_trmm_samsung_hbm_pim.py "
    "-p no:cacheprovider -q"
)


def test_trmm_samsung_hbm_pim(request):
    # Task 006: workload loaded from the leaf dir
    # (samsung_hbm_pim/trmm/workload.py, slice form) by run_cell; STAGES re-read
    # from it.
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET,
        folder=request.path.parent, stages=None, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-2 triangular; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.",
    )
    # First-class recorded verdict (spec Answer 3), never a skip.
    assert verdict.status in (reference.CYCLES_ONLY, reference.BLOCKED_SIM), verdict
    if verdict.status in (reference.CYCLES_ONLY, reference.PASS) and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"{_TARGET}: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
