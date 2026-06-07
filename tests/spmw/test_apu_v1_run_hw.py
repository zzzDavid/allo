# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated hardware test for the APU v1 run hook.

Skipped on any host that's missing the ARC GNU toolchain, the
example-gvml template directory, or the GSI PCI sysfs node -- the
same condition `_apu_v1_unavailable_reason` checks. On the APU host
this exercises the full build-and-run path and asserts `cycles > 0`.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.spmw_codegen import RunResult, _apu_v1_unavailable_reason
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_apu_v1_target


def _trace() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


_SKIP_REASON = _apu_v1_unavailable_reason()


@pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=str(_SKIP_REASON),
)
def test_apu_v1_run_returns_cycles():
    """On a real Gemini host the run path must build, execute, and
    report a positive cycle count."""
    target = build_apu_v1_target()
    compiled = allo.compile_for_target(target, _trace())

    N = 32 * 1024
    W = np.zeros(N, dtype=np.uint16)
    x = np.zeros(N, dtype=np.uint16)
    result = compiled.run(local_W=W, local_x=x)

    assert isinstance(result, RunResult)
    assert result.backend == "apu_v1"
    if result.extra.get("returncode") == 0:
        assert result.cycles is not None and result.cycles > 0, (
            f"expected positive cycles, got {result.cycles!r}; "
            f"stdout tail:\n{result.stdout[-400:]}"
        )


if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"SKIP: {_SKIP_REASON}")
    else:
        test_apu_v1_run_returns_cycles()
        print("ALL PASSED")
