# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-op corpus harness for the SPEC-023 schedule-search validation.

Provides the wiring the verifier (009) consumes to produce a REAL-sim cycle
comparison of:
  - the GROUP-LOCAL baseline (the placement-realization autoscheduler, with the
    three SPEC-023 schedule-search knobs DISABLED), vs
  - the SEARCH-enabled schedule (global liveness + cross-op residency D1/T6 +
    capacity-bounded tile/fold D2 + double-buffer depth D3).

Both schedules compile from the SAME workload through the SAME pipeline
(workload -> customize -> match_workload -> compile_for_target -> run) on the
SAME simulator; the ONLY difference is the `SPMW_DISABLE_SCHEDULE_SEARCH` env
toggle (`spmw_knobs.cross_with_knobs` skips the three search knobs when set).
This is the honest baseline-vs-search comparison the verifier needs -- a
real-sim cycle delta, NOT a cost self-compare.

NO win-assertion lives here (that is the verifier's oracle, task 009); this is
harness + fixtures only. The multi-op workload reuses `build_mlp_workload`
(two `@allo.work` MAC layers whose inter-layer activation `local_h` is the
cross-kernel value the residency knob keys on). GEMV single-op is the
regression floor (the search is a no-op there: one op, nothing crosses, so all
three knobs are 1x fans -> baseline == search).
"""

from __future__ import annotations

import os
import contextlib

import allo
from allo.spmw_codegen import RunResult
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_mlp_workload


_SEARCH_OFF_ENV = "SPMW_DISABLE_SCHEDULE_SEARCH"


@contextlib.contextmanager
def schedule_search_disabled():
    """Context manager: disable the SPEC-023 schedule-search knobs (group-local
    baseline) for the duration, restoring the prior env on exit."""
    prev = os.environ.get(_SEARCH_OFF_ENV)
    os.environ[_SEARCH_OFF_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_SEARCH_OFF_ENV, None)
        else:
            os.environ[_SEARCH_OFF_ENV] = prev


def _compile(target, workload):
    """workload -> customize -> match -> autoschedule+codegen. Returns
    (trace, compiled)."""
    schedule = allo.customize(workload, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)
    compiled = allo.compile_for_target(target, trace)
    return trace, compiled


def compile_baseline_and_search(target, workload):
    """Compile `workload` for `target` TWICE on one pipeline: the group-local
    baseline (search knobs off) and the search-enabled schedule (knobs on).

    Returns `(trace, baseline_compiled, search_compiled)`. The trace is shared
    (matching is identical; only autoschedule differs). The verifier runs both
    `Compiled` artifacts on the same sim and compares cycles.
    """
    with schedule_search_disabled():
        trace, baseline = _compile(target, workload)
    # Search enabled (default; ensure the env is clear in case a caller set it).
    prev = os.environ.pop(_SEARCH_OFF_ENV, None)
    try:
        _trace2, search = _compile(target, workload)
    finally:
        if prev is not None:
            os.environ[_SEARCH_OFF_ENV] = prev
    return trace, baseline, search


def run_baseline_vs_search(target, workload):
    """Compile + RUN both schedules on the bound simulator. Returns
    `(baseline_result, search_result)` -- two `RunResult`s the verifier's
    sim-win oracle compares (`search_result.cycles <= baseline_result.cycles`
    where the search pays off; equal at the GEMV floor). No assertion here."""
    _trace, baseline, search = compile_baseline_and_search(target, workload)
    return baseline.run(), search.run()


def sim_unavailable(result: RunResult) -> bool:
    """True iff the simulator/hardware was not reachable (the graceful-skip
    signal the verifier honours)."""
    return result.cycles is None and "simulator unavailable" in (result.stdout or "")


# --------------------------------------------------------------------- #
# Multi-op workload (MLP: two MAC layers, cross-kernel activation local_h)
# --------------------------------------------------------------------- #


def multi_op_workload():
    """The multi-op MLP corpus workload: layer1 writes `local_h`, layer2 reads
    it -- the cross-kernel activation the search keeps resident. Reuses the
    landed two-`@allo.work` MLP fixture."""
    return build_mlp_workload()


# --------------------------------------------------------------------- #
# GEMV single-op regression floor (the search is a no-op here)
# --------------------------------------------------------------------- #


def gemv_floor_trace(target_name: str, k: int = 1024, m: int = 32) -> MatchTrace:
    """A single-op GEMV `MatchTrace` -- one MAC, one work-id, nothing crosses a
    boundary. The schedule search must be a NO-OP (residency/tile/double_buffer
    all 1x fans), so the search schedule is byte-identical to the baseline at
    this floor. The regression anchor the verifier checks stays unchanged."""
    return MatchTrace(
        target_name=target_name, module_name="gemv_floor", matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
                enclosing_loops=[("%m", "0", str(m), 1), ("%k", "0", str(k), 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b"))])
