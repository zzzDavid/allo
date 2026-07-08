# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical syr2k on the real APU v1 scalar ARC fallback."""

import pytest

import allo
from lib import apu_v1, cell
from lib.cost import bind_cost
from lib.targets import build_target

_KERNEL = "syr2k"
_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/syr2k/test_syr2k_apu_v1.py "
    "-p no:cacheprovider -q"
)


@pytest.mark.apu_v1_device
def test_syr2k_apu_v1(request, apu_v1_device_gate):
    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    result, verdict, _record = apu_v1.run_compiled(
        compiled,
        _KERNEL,
        folder=request.path.parent,
        run_cmd=_RUN_CMD,
    )
    apu_v1.assert_result(result, verdict)
