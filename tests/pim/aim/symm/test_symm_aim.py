# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench symm compiled explicitly for SK hynix GDDR6-AiM."""

from __future__ import annotations

import allo

from lib import aim
from lib.cost import bind_cost
from lib.shapes import shape
from lib.targets import build_target

_KERNEL = "symm"
_TARGET = "aim"
_RUN_CMD = (
    "python -m pytest tests/pim/aim/symm/test_symm_aim.py " "-p no:cacheprovider -q"
)
_NOTES = (
    "AiM-local SPMW workload; ramulator2 trace simulation; "
    "cycles only because the simulator exposes no functional numerics."
)


def test_symm_aim(request):
    workload = aim.load_workload(request.path.parent, _KERNEL)
    target = build_target(_TARGET)

    compiled = allo.compile(workload.build(), target, bind_cost(target))

    result, verdict, _record = aim.run_compiled(
        compiled,
        workload,
        kernel=_KERNEL,
        folder=request.path.parent,
        shapes=shape(_KERNEL),
        run_cmd=_RUN_CMD,
        notes=_NOTES,
    )
    aim.assert_result(result, verdict)
