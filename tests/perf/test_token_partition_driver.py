# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token-partitioning reproduction driver (release gate G10): the
headline plan is the normal shared selection under the compiler's resource
budget, the forced alternatives are explicit layouts under the same budget,
every plan keeps its provenance and artifacts, a failed plan is recorded
without discarding the others, a run of the wrong kind is refused, and the
summary reports the study's figures as a diagnostic against the paper's."""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.cent_aim import token_partition as tp


def test_shapes_and_summary_figures():
    assert tp._parse_shapes("1x128,8x512") == ((1, 128), (8, 512))
    records = [
        {"heads": 1, "tokens": 4096, "selected_partitions": 1, "best_partitions": 8,
         "gap": 0.0134, "head_only_over_selected": 2.17, "distinct_plans": 5,
         "split_plans_checked": 3, "failed_plans": {}},
        {"heads": 8, "tokens": 128, "selected_partitions": 1, "best_partitions": 1,
         "gap": 0.0, "head_only_over_selected": 1.0, "distinct_plans": 4,
         "split_plans_checked": 3, "failed_plans": {"P8": {"error": "x"}}},
        {"heads": 32, "tokens": 128, "error": "RuntimeError: simulator"},
    ]
    summary = tp.summarize(records)
    assert summary["workloads_completed"] == 2
    assert summary["minima_reached"] == 1
    assert summary["max_gap"] == 0.0134
    assert summary["head_only_over_selected_1x4096"] == 2.17
    assert (summary["distinct_plans"], summary["split_plans_checked"], summary["failed_plans"]) == (9, 6, 1)
    assert summary["comparable_with_paper"] is False and summary["paper"] == tp.PAPER


def test_estimates_only_run_keeps_every_plan_and_its_provenance(tmp_path):
    result = tp.run_workload(1, 128, simulate=False, artifacts=str(tmp_path))
    rows = result.rows
    assert not result.failures
    assert {"default"} | {f"P{p}" for p in tp.PARTITIONS} <= set(rows)
    for label, row in rows.items():
        assert row.device is None and row.total is None and row.execution is None
        assert row.estimate > 0
        # Every plan, the normal selection included, fits the budget.
        assert all(tp.within_budget(mode) for mode in row.modes)
        assert row.partitions == tp.partitions_of(row.mode)
        placements = row.provenance["placements"]
        assert placements and all(p["effects"] is not None for p in placements)
        assert row.provenance["resource_budget"] == tp.BUDGET
        assert row.provenance["lowering_manifest"] and row.provenance["sv_lowering"]["kind"] == "contraction"
        assert row.provenance["command_count"] > 0
        if row.partitions > 1:
            assert row.host > 0 and row.bf16_error is not None and row.provenance["host_steps"]
        else:
            assert row.host == 0 and row.bf16_error is None
    record = result.record
    # The normal selection's P is the selected one; the forced plan with the
    # same P is the ranking's cheapest for it, and the two agree.
    assert result.selected == rows["default"].partitions
    assert record["default_equals_forced_choice"] is True
    assert record["gap"] is None and record["resource_budget"] == tp.BUDGET
    assert record["compiler"]["allo"] and record["default_selection_unrestricted"]
    assert record["plans"]["default"]["merge_check"]["applicable"] == (result.selected > 1)
    assert record["distinct_plans"] >= len(tp.PARTITIONS)
    json.dumps(record, default=str)  # JSON-ready
    # Artifacts were written per plan before the next one ran.
    for label in record["plans"]:
        folder = os.path.join(str(tmp_path), "h1_l128", label)
        assert os.path.isfile(os.path.join(folder, "plan.json")), label


def test_a_failed_plan_is_recorded_and_the_others_kept(monkeypatch, tmp_path):
    from types import SimpleNamespace

    original = tp.compile_program

    def wrong_kind_for_P2(attention, program, target, *, layouts=None, budget=tp.BUDGET, virtual=False):
        built = original(attention, program, target, layouts=layouts, budget=budget, virtual=virtual)
        if not virtual and layouts is not None:
            modes = [p.mode for p in (layouts if isinstance(layouts, (list, tuple)) else [layouts])]
            if any(tp.partitions_of(m) == 2 for m in modes):
                built.run_backend = lambda: SimpleNamespace(kind="virtual", cycles=1, extra={}, stdout="")
            else:
                built.run_backend = lambda m=modes: SimpleNamespace(
                    kind="simulator-timing-only", cycles=100 + 10 * max(tp.partitions_of(x) for x in m),
                    extra={"simulator": {"kind": "stub"}, "segment_records": [], "invocations": []}, stdout="stub",
                )
        elif not virtual:
            built.run_backend = lambda: SimpleNamespace(
                kind="simulator-timing-only", cycles=100,
                extra={"simulator": {"kind": "stub"}, "segment_records": [], "invocations": []}, stdout="stub",
            )
        return built

    monkeypatch.setattr(tp, "compile_program", wrong_kind_for_P2)
    result = tp.run_workload(1, 128, simulate=True, artifacts=str(tmp_path))
    assert set(result.failures) == {"P2"}
    assert "simulator-timing-only" in result.failures["P2"]["error"]
    assert "P2" not in result.rows and {"default", "P1", "P4", "P8"} <= set(result.rows)
    assert result.gap is not None and result.best is not None
    assert os.path.isfile(os.path.join(str(tmp_path), "h1_l128", "P2", "plan.json"))
    assert os.path.isfile(os.path.join(str(tmp_path), "h1_l128", "P1", "stdout.txt"))
