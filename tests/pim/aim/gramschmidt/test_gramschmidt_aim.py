# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gramschmidt on aim -- Tier-2 triangular extension cell (task 008).

Thin cell: build the target from `lib`, run the shared `gramschmidt` workload (the
bare MAC-reduction core; triangular/orthogonalization access applied host-side
against the numpy ref, mirroring syrk) through `lib.cell.run_cell`, record the
verdict + cycles + results.json/RESULTS.md + COVERAGE.tsv. Declares zero hardware
(everything from `lib` + the shared workload). The bare contraction matches MAC
on the existing matcher -- no matcher change needed.
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

from workloads import gramschmidt as _wl

_KERNEL = "gramschmidt"
_TARGET = "aim"
_RUN_CMD = (
    "python -m pytest tests/pim/aim/gramschmidt/test_gramschmidt_aim.py "
    "-p no:cacheprovider -q"
)


def test_gramschmidt_aim(request):
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET, workload=_wl.build(),
        folder=request.path.parent, stages=_wl.STAGES, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="Tier-2 triangular; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).",
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
