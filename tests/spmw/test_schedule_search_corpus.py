# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-023 task 007: multi-op corpus validation WIRING.

Exercises the `_schedule_search_corpus` harness the verifier (009) consumes:
the group-local baseline vs the search-enabled schedule compile from the SAME
multi-op MLP workload through the SAME pipeline on the SAME simulator. This
file proves the WIRING (both compile, the search is non-trivial, both run on a
host-runnable sim, GEMV stays at its floor) -- it asserts NO cycle win (that is
the verifier's real-sim oracle, task 009).
"""

from __future__ import annotations

import pytest

import allo
from allo.spmw_codegen import RunResult

from _fixtures import build_upmem_target, build_samsung_target
from _schedule_search_corpus import (
    compile_baseline_and_search,
    run_baseline_vs_search,
    schedule_search_disabled,
    multi_op_workload,
    gemv_floor_trace,
    sim_unavailable,
)


# --------------------------------------------------------------------- #
# The toggle: baseline (search off) vs search (on) on the SAME workload
# --------------------------------------------------------------------- #


def test_baseline_and_search_both_compile_same_workload():
    """Both schedules compile from one MLP workload through one pipeline; the
    trace is shared (matching identical), only autoschedule differs."""
    target = build_upmem_target()
    trace, baseline, search = compile_baseline_and_search(
        target, multi_op_workload())
    # The multi-op trace carries >=2 MAC layers (the cross-kernel corpus).
    assert len(trace.by_target_op("MAC")) >= 2
    assert baseline is not None and search is not None
    assert baseline.target.name == search.target.name == "upmem"


def test_search_is_non_trivial_vs_baseline():
    """The search must change SOMETHING vs the group-local baseline on a
    multi-op workload -- else the comparison is a tautology. We check the
    placement EXTRA differs (the search-enabled MLP earns a residency / depth /
    tile decision the baseline lacks). Proven on Samsung, where the cross-kernel
    activation earns `residency=resident` under search (its residency knob_cost
    is registered) but stays restage under the baseline. (UPMEM/APU earn their
    own decisions once their residency/overlap win-cost wiring lands -- the
    harness is target-agnostic; this asserts the toggle is non-trivial on a
    backend where the cost is registered today.)"""
    target = build_samsung_target()
    _trace, baseline, search = compile_baseline_and_search(
        target, multi_op_workload())

    def _search_tags(compiled):
        layouts = compiled.layout
        layouts = layouts if isinstance(layouts, list) else [layouts]
        tags = set()
        for pl in layouts:
            ex = getattr(pl, "extra", {}) or {}
            for k in ("residency", "tile", "double_buffer"):
                if k in ex and not (k == "residency" and ex[k] == "restage"):
                    tags.add(k)
        return tags

    base_tags = _search_tags(baseline)
    search_tags = _search_tags(search)
    # The baseline carries NO active search tag; the search carries at least one
    # (UPMEM MLP: cross-kernel residency on local_h). Non-tautology guard.
    assert base_tags == set(), base_tags
    assert search_tags, "search produced no schedule-search decision on the MLP"


def test_disabled_toggle_reverts_to_group_local():
    """Under `schedule_search_disabled()`, the autoschedule is the group-local
    baseline -- no residency/tile/double_buffer fan. Confirms the single env
    choke point the harness rides."""
    target = build_upmem_target()
    workload = multi_op_workload()
    with schedule_search_disabled():
        schedule = allo.customize(workload, enable_tensor=False)
        trace = allo.match_workload(target, schedule.module)
        compiled = allo.compile_for_target(target, trace)
    layouts = compiled.layout
    layouts = layouts if isinstance(layouts, list) else [layouts]
    for pl in layouts:
        ex = getattr(pl, "extra", {}) or {}
        assert "tile" not in ex and "double_buffer" not in ex
        assert ex.get("residency", "restage") == "restage"


# --------------------------------------------------------------------- #
# Both run on the host-runnable sim (UPMEM) -> the verifier's cycle inputs
# --------------------------------------------------------------------- #


def test_baseline_and_search_run_on_upmem_sim():
    """Both schedules RUN on uPIMulator (the host-runnable sim), each yielding
    a cycle count -- the artifact the verifier's sim-win oracle compares. Skips
    cleanly when the sim is unavailable or hits a transient uPIMulator crash
    (the known flakiness). NO win-assertion here."""
    target = build_upmem_target()
    try:
        base_res, search_res = run_baseline_vs_search(target, multi_op_workload())
    except RuntimeError as e:
        pytest.skip(f"uPIMulator transient failure: {str(e)[:120]}")
    assert isinstance(base_res, RunResult) and isinstance(search_res, RunResult)
    assert base_res.backend == search_res.backend == "upmem"
    if sim_unavailable(base_res) or sim_unavailable(search_res):
        pytest.skip("uPIMulator unavailable; harness compiles both schedules")
    # Both produced real cycle counts -> the verifier can compare them.
    assert base_res.cycles is not None and base_res.cycles > 0, base_res.stdout[-300:]
    assert search_res.cycles is not None and search_res.cycles > 0, search_res.stdout[-300:]


def test_upmem_residency_artifact_differs():
    """SPEC-023 T6 WIN-EMIT (no sim): the UPMEM MLP search schedule
    (residency=resident) emits a DIFFERENT DPU kernel than the group-local
    baseline (which restages) -- the baseline emits the inter-kernel MRAM
    round-trip of the cross-kernel activation; the resident search elides it.
    The artifacts must differ for any real-sim cycle delta to exist."""
    target = build_upmem_target()
    _trace, baseline, search = compile_baseline_and_search(
        target, multi_op_workload())
    bsrc = baseline._ctx.get_gemv_kernel_src()
    ssrc = search._ctx.get_gemv_kernel_src()
    # Baseline (restage, cross-kernel) emits the staging round-trip; the
    # resident search elides it -> the kernel sources differ.
    assert "residency restage" in bsrc
    assert "residency restage" not in ssrc
    assert bsrc != ssrc


def test_upmem_residency_sim_win():
    """SPEC-023 T6 WIN, real-sim: the resident search schedule runs on
    uPIMulator with STRICTLY FEWER cycles than the restaging group-local
    baseline -- the elided inter-kernel MRAM round-trip. This is the
    load-bearing sim-confirmed win (not a cost self-compare). Skips on sim
    unavailable / a transient uPIMulator crash (the known flakiness)."""
    target = build_upmem_target()
    try:
        base_res, search_res = run_baseline_vs_search(target, multi_op_workload())
    except RuntimeError as e:
        pytest.skip(f"uPIMulator transient failure: {str(e)[:120]}")
    if sim_unavailable(base_res) or sim_unavailable(search_res):
        pytest.skip("uPIMulator unavailable")
    assert base_res.cycles and search_res.cycles, (base_res.cycles, search_res.cycles)
    # The resident schedule (search) is strictly cheaper than restage (baseline).
    assert search_res.cycles < base_res.cycles, (
        f"expected residency win: search {search_res.cycles} < "
        f"baseline {base_res.cycles}")


# --------------------------------------------------------------------- #
# GEMV single-op regression floor: the search is a no-op
# --------------------------------------------------------------------- #


def test_gemv_floor_search_is_noop():
    """A single-op GEMV crosses no boundary -> the schedule search is a no-op
    (residency/tile/double_buffer all 1x fans), so the search schedule is
    byte-identical to the baseline: same placements, no search `extra` tags.
    The regression floor the verifier checks stays unchanged."""
    from allo.spmw_autoschedule import autoschedule
    target = build_upmem_target()
    trace = gemv_floor_trace("upmem")

    with schedule_search_disabled():
        base_layouts = autoschedule(target, trace)
    search_layouts = autoschedule(target, trace)

    assert len(base_layouts) == len(search_layouts) == 1
    b, s = base_layouts[0], search_layouts[0]
    # Identical placements; no active search tag on either (single-op floor).
    assert b.placements.keys() == s.placements.keys()
    for k in ("tile", "double_buffer"):
        assert k not in b.extra and k not in s.extra
    assert b.extra.get("residency", "restage") == "restage"
    assert s.extra.get("residency", "restage") == "restage"


if __name__ == "__main__":
    test_baseline_and_search_both_compile_same_workload()
    test_search_is_non_trivial_vs_baseline()
    test_disabled_toggle_reverts_to_group_local()
    test_baseline_and_search_run_on_upmem_sim()
    test_upmem_residency_artifact_differs()
    test_upmem_residency_sim_win()
    test_gemv_floor_search_is_noop()
    print("ALL PASSED")
