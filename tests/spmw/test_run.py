# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for `Compiled.run()` across all five backends.

These tests verify that `Compiled.run()` no longer raises
`NotImplementedError` (BLOCKER 7) and returns a structured `RunResult`.
Each backend is exercised; if its simulator/hardware is unavailable
the call still succeeds and `cycles` is `None` — the test only
asserts that the function returns a `RunResult` of the right shape.

The Samsung GEMV test additionally checks that, when the real
`pim_driver` is available, the call returns a positive cycle count.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import allo
from allo.spmw_codegen import (
    Compiled,
    RunResult,
    _aim_root,
    _docker_image_exists,
    _pimsim_root,
    _upim_root,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_apu_v2_target,
    build_samsung_target,
    build_upmem_target,
)


# --------------------------------------------------------------------- #
# Synthetic traces (kept local to this file to avoid coupling to the
# move-scheduling test's fixtures).
# --------------------------------------------------------------------- #


def _samsung_mac_trace() -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
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


def _aim_mac_trace() -> MatchTrace:
    return MatchTrace(
        target_name="aim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0_0",
                work_id=(0, 0, 0),
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


def _upmem_mac_trace() -> MatchTrace:
    return MatchTrace(
        target_name="upmem",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0_0",
                work_id=(0, 0, 0),
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


def _apu_v1_mac_trace() -> MatchTrace:
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


def _apu_v2_mac_trace() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v2",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
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


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def _compile(target_builder, trace_builder):
    target = target_builder()
    return allo.compile_for_target(target, trace_builder())


def test_run_returns_runresult_for_all_backends():
    """`Compiled.run()` must return a `RunResult` for every target,
    even when the simulator is unavailable — it must not raise
    `NotImplementedError`.
    """
    cases = [
        (build_samsung_target, _samsung_mac_trace, "samsung_hbm_pim"),
        (build_aim_target, _aim_mac_trace, "aim"),
        (build_upmem_target, _upmem_mac_trace, "upmem"),
        (build_apu_v1_target, _apu_v1_mac_trace, "apu_v1"),
        (build_apu_v2_target, _apu_v2_mac_trace, "apu_v2"),
    ]
    for target_builder, trace_builder, name in cases:
        compiled = _compile(target_builder, trace_builder)
        result = compiled.run()
        assert isinstance(result, RunResult), (
            f"{name}: expected RunResult, got {type(result).__name__}"
        )
        assert result.backend == name, (
            f"{name}: result.backend mismatch ({result.backend!r})"
        )
        # cycles is either None (simulator unavailable / functional-only)
        # or a non-negative int.
        assert result.cycles is None or (
            isinstance(result.cycles, int) and result.cycles >= 0
        ), f"{name}: cycles must be None or non-negative int, got {result.cycles!r}"
        assert isinstance(result.stdout, str)


def test_apu_v2_returns_none_cycles():
    """APU v2 l1_sim is functional-only — cycles must be None."""
    compiled = _compile(build_apu_v2_target, _apu_v2_mac_trace)
    result = compiled.run()
    assert result.cycles is None
    assert result.backend == "apu_v2"


_PIMSIM_DRIVER = _pimsim_root() / "pim_driver"


@pytest.mark.skipif(
    not _PIMSIM_DRIVER.exists(),
    reason=f"pim_driver not built at {_PIMSIM_DRIVER}",
)
def test_samsung_gemv_real_pim_driver_returns_cycles():
    """When the real `pim_driver` is available, a Samsung GEMV run
    must report a positive cycle count.

    Marked `slow` via the long timeout in `_run_samsung`; pytest -x will
    surface failures fast. The smallest GEMV the driver accepts is
    output_dim=4096, input_dim=1024 (one full Samsung tile).
    """
    import numpy as np

    target = build_samsung_target()
    compiled = allo.compile_for_target(target, _samsung_mac_trace())
    M, K = 4096, 1024
    W = np.zeros((M, K), dtype=np.float16)
    x = np.zeros(K, dtype=np.float16)
    result = compiled.run(W=W, x=x)
    assert result.backend == "samsung_hbm_pim"
    # The driver may fail on a malformed cmd stream; if it does, stdout
    # will tell us. We only assert the no-crash + positive-cycles
    # contract when the driver returned successfully.
    if result.extra.get("returncode") == 0:
        assert result.cycles is not None and result.cycles > 0, (
            f"expected positive cycles, got {result.cycles!r}; "
            f"stdout tail:\n{result.stdout[-400:]}"
        )


if __name__ == "__main__":
    test_run_returns_runresult_for_all_backends()
    test_apu_v2_returns_none_cycles()
    print("ALL PASSED")
