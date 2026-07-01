# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemver compiled explicitly for Samsung HBM-PIM."""

from __future__ import annotations

import allo

from lib import samsung
from lib.cost import bind_cost
from lib.shapes import shape
from lib.targets import build_target

_KERNEL = "gemver"
_TARGET = "samsung_hbm_pim"
_RUN_CMD = "python -m pytest tests/pim/samsung_hbm_pim/gemver/test_gemver_samsung_hbm_pim.py -p no:cacheprovider -q"
_NOTES = (
    "Mixed ELTWISE and REDUCE stages; real cycles with honest CYCLES-ONLY numerics."
)


def test_gemver_samsung_hbm_pim(request):
    workload = samsung.load_workload(request.path.parent, _KERNEL)
    target = build_target(_TARGET)

    compiled = allo.compile(
        workload.build(),
        target,
        bind_cost(target),
        host_moves=workload.HOST_MOVES,
    )

    result, verdict, _record = samsung.run_compiled(
        compiled,
        workload,
        kernel=_KERNEL,
        folder=request.path.parent,
        shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes=_NOTES,
    )
    samsung.assert_result(result, verdict, allow_pass=False)
