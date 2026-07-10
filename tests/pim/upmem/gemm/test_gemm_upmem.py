# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Faithful PolyBench leaf compiled through MLIR for a full UPMEM rank."""

from __future__ import annotations

from pathlib import Path

import allo

from lib import upmem
from lib.cost import bind_cost
from lib.targets import build_target

_KERNEL = "gemm"
_RUN_CMD = (
    f"python -m pytest tests/pim/upmem/{_KERNEL}/{Path(__file__).name} "
    "-p no:cacheprovider -q"
)
_NOTES = (
    "The complete canonical Allo kernel is lowered through MLIR to portable C. "
    "Its NumPy-visible results are checked against the repository reference; "
    "the UPMEM ABI uses 64 DPUs and retains the declarative partition, barrier, "
    "collective, temporal, pivot, or wavefront orchestration plan. Reported "
    "cycles are analytical cost-program estimates, not simulator measurements."
)


def test_upmem_polybench_leaf(request):
    workload = upmem.load_leaf_workload(request.path.parent)
    target = build_target("upmem")
    cost = bind_cost(target)

    compiled = allo.compile(workload.build(), target, cost)

    _result, estimate, verdict, _record = upmem.run_polybench_case(
        compiled,
        workload.CASE,
        folder=request.path.parent,
        run_cmd=_RUN_CMD,
        notes=_NOTES,
    )
    assert verdict.status == "PASS"
    assert estimate.cycles > 0
