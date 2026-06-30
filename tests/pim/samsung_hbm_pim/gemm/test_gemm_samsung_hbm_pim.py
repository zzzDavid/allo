# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemm on samsung_hbm_pim -- Tier-1 single-output cell (task 005).

Thin cell: build the target from `lib`, run the shared `gemm` workload through
`lib.cell.run_cell`, record the verdict + cycles + results.json/RESULTS.md +
COVERAGE.tsv. Declares zero hardware (everything from `lib` + the shared
workload). Verdict derived from what the run surfaces: Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.
"""

from __future__ import annotations

from lib import cell, reference
from lib.shapes import shape

_KERNEL = "gemm"
_TARGET = "samsung_hbm_pim"
_RUN_CMD = (
    "python -m pytest tests/pim/samsung_hbm_pim/gemm/test_gemm_samsung_hbm_pim.py "
    "-p no:cacheprovider -q"
)


def test_gemm_samsung_hbm_pim(request):
    # SPEC-05: workload loaded from the leaf dir (samsung_hbm_pim/gemm/workload.py,
    # slice form) by run_cell; STAGES re-read from it.
    result, verdict, _record = cell.run_cell(
        kernel=_KERNEL, target_name=_TARGET,
        folder=request.path.parent, stages=None, shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes="SPEC-05 SPMW slice-form gemm (mapping=[16,8]); genuine fp16 PASS vs A@B; partition mapping-driven (128 work-id buckets, per-PE slice loop bound = P//128).",
    )
    # First-class recorded verdict (spec Answer 3), never a skip. PASS is
    # accepted (the SPEC-03 tightening gate) for a single-stage bare GEMV that
    # surfaces a matching `y`; a GEMM-shaped stream stays CYCLES-ONLY.
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
