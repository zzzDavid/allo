# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Virtual-backend runner (design 04 §2).

`compile_for_target(target, trace, backend="virtual")` produces a
`Compiled` whose `run()` dispatches to the sim-free `_run_virtual`, which
delegates to `spmw_cost_model.evaluate`. These are STATIC assertions (no
simulator): the virtual runner returns a `RunResult` carrying the
CostModel's `cycles` + `phases` + `confidence`, by construction without
booting any simulator.
"""
from __future__ import annotations

import allo
from allo.spmw_codegen import RunResult, _run_virtual
from allo.spmw_match import MatchedOp, MatchTrace, OperandBinding

from _fixtures import build_samsung_target, build_apu_v2_target


def _samsung_gemv_trace() -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%i", "0", "32", 1),
                    ("%k", "0", "1024", 1),
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


def _apu_v2_trace() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v2",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[("%k", "0", "1024", 1)],
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


def test_virtual_backend_selector_returns_compiled():
    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    compiled = allo.compile_for_target(target, trace, backend="virtual")
    assert compiled.backend == "virtual"
    assert compiled.cost_flavor == "faithful"


def test_virtual_run_returns_cycles_phases_confidence():
    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    compiled = allo.compile_for_target(target, trace, backend="virtual")
    result = compiled.run()
    assert isinstance(result, RunResult)
    assert result.backend == "virtual"
    assert result.cycles is not None and result.cycles > 0
    # confidence + phases ride extra (RunResult is not structurally changed).
    assert result.extra["confidence"] == "calibrated"
    assert set(("preload", "exec", "readback")) <= set(result.extra["phases"])
    assert result.extra["cost_model"] == "samsung_faithful"
    assert result.extra["priced_target"] == "samsung_hbm_pim"


def test_virtual_run_matches_kernel_cycles_argmin():
    """The virtual runner's cycle count is exactly the bound CostModel's
    whole-program estimate -- the same number the autoschedule argmin
    rides for the chosen placement."""
    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    compiled = allo.compile_for_target(target, trace, backend="virtual")
    result = compiled.run()
    cost_fn = allo.get_cost("kernel_cycles", target)
    assert cost_fn(trace, compiled.layout) == result.cycles


def test_virtual_run_flavor_selects_costmodel():
    """A different `cost_flavor` prices the same structural target with a
    different CostModel -- different estimate, zero device edit."""
    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    faithful = allo.compile_for_target(target, trace, backend="virtual").run()
    optimistic = allo.compile_for_target(
        target, trace, backend="virtual", cost_flavor="optimistic"
    ).run()
    assert optimistic.extra["cost_model"] == "samsung_optimistic"
    assert optimistic.cycles != faithful.cycles


def test_virtual_run_carries_placeholder_confidence():
    target = build_apu_v2_target()
    trace = _apu_v2_trace()
    result = allo.compile_for_target(
        target, trace, backend="virtual"
    ).run()
    assert result.extra["confidence"] == "placeholder"
    assert result.extra["cost_model"] == "apu_v2_placeholder"


def test_default_backend_unchanged():
    """`backend=None` (the default) leaves dispatch on `target.name`; the
    virtual selector is purely additive."""
    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    compiled = allo.compile_for_target(target, trace)
    assert compiled.backend is None


def test_run_virtual_is_sim_free():
    """`_run_virtual` reaches no simulator-launch helper -- its only inputs
    are the bound CostModel and the in-memory trace (design 04 §2.3). We
    assert it produces a result even with the sim root env vars pointed at
    a nonexistent path, proving it never touches them."""
    import os

    target = build_samsung_target()
    trace = _samsung_gemv_trace()
    compiled = allo.compile_for_target(target, trace, backend="virtual")
    saved = os.environ.get("PIMSIMULATOR_ROOT")
    os.environ["PIMSIMULATOR_ROOT"] = "/nonexistent/sim/root"
    try:
        result = _run_virtual(compiled)
        assert result.cycles is not None
    finally:
        if saved is None:
            os.environ.pop("PIMSIMULATOR_ROOT", None)
        else:
            os.environ["PIMSIMULATOR_ROOT"] = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("STATIC PASSED")
