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


def _stub_run(kind="simulator-timing-only", cycles=100):
    from types import SimpleNamespace

    return lambda: SimpleNamespace(
        kind=kind, cycles=cycles,
        extra={"simulator": {"kind": "stub"}, "segment_records": [], "invocations": []},
        stdout="stub",
    )


def _inject(monkeypatch, failing):
    """Compile through the real pipeline; stub the simulator so that the plan
    ``failing(modes, layouts)`` says returns a run of the wrong kind and
    every other plan returns a positive cycle count that grows with P."""
    original = tp.compile_program

    def compile_program(attention, program, target, *, layouts=None, budget=tp.BUDGET, virtual=False):
        built = original(attention, program, target, layouts=layouts, budget=budget, virtual=virtual)
        if virtual:
            return built
        modes = [p.mode for p in tp._placements(built)]
        if failing(modes, layouts):
            built.run_backend = _stub_run(kind="virtual", cycles=1)
        else:
            built.run_backend = _stub_run(cycles=100 + 10 * max(tp.partitions_of(m) for m in modes))
        return built

    monkeypatch.setattr(tp, "compile_program", compile_program)


def test_a_failed_forced_alternative_keeps_its_record_and_voids_the_comparison(monkeypatch, tmp_path):
    _inject(monkeypatch, lambda modes, layouts: layouts is not None and any(tp.partitions_of(m) == 2 for m in modes))
    result = tp.run_workload(1, 128, simulate=True, artifacts=str(tmp_path))
    assert set(result.failures) == {"P2"}
    failure = result.failures["P2"]
    assert failure["stage"] == "simulate" and "simulator-timing-only" in failure["error"]
    # The failed plan's record is retained in full, in memory and on disk.
    kept = result.record["plans"]["P2"]
    assert kept["modes"] and kept["provenance"]["placements"] and kept["provenance"]["lowering_manifest"]
    assert kept["provenance"]["graph"]["activities"] and kept["estimate_cycles"] > 0
    assert kept["completed"] == ["provenance", "host_price", "estimate", "merge_check"]
    on_disk = json.load(open(os.path.join(str(tmp_path), "h1_l128", "P2", "plan.json")))
    assert on_disk["modes"] == kept["modes"] and on_disk["failure"]["stage"] == "simulate"
    assert on_disk["provenance"]["placements"][0]["effects"]["value_layouts"]
    # The minima comparison is not made from a partial set.
    assert "P2" not in result.rows and {"default", "P1", "P4", "P8"} <= set(result.rows)
    assert result.gap is None and result.best is None and not result.comparison_complete
    assert result.record["comparison"] == {
        "complete": False, "required": ["default", "P1", "P2", "P4", "P8"], "missing": ["P2"],
        "controls_complete": True,
    }
    with pytest.raises(AssertionError, match="P2 failed at simulate.*incomplete"):
        tp.assert_complete(result)
    # The graph record carries what the cost accounting needs.
    activity = kept["provenance"]["graph"]["activities"][0]
    assert {"primitive", "occupancy", "metadata", "latency_cycles", "depends_on"} <= set(activity)


def test_a_failed_distinct_control_is_never_replaced_by_an_alias(monkeypatch, tmp_path):
    """1 head x 4096 tokens: the head-only control (one head per channel,
    unsplit) is a different plan from P1 (the flattened mapping); when it
    fails, no alias stands in for it and the head-only ratio is absent,
    while the complete partition comparison still stands."""
    _inject(
        monkeypatch,
        lambda modes, layouts: layouts is not None
        and tp.knob(modes[-1], "batch_mapping") == "channels"
        and tp.partitions_of(modes[-1]) == 1,
    )
    result = tp.run_workload(1, 4096, simulate=True, artifacts=str(tmp_path))
    assert "head-only" in result.expected and set(result.failures) == {"head-only"}
    assert "head-only" not in result.record["aliases"] and "head-only" not in result.rows
    assert result.record["head_only_over_selected"] is None
    assert result.comparison_complete and result.gap is not None
    assert not result.controls_complete
    with pytest.raises(AssertionError, match="head-only"):
        tp.assert_complete(result)
    # The alias that exists was decided before execution, from the plans.
    assert result.record["aliases"] == {"fixed-8": "P8"}


def test_aliases_are_decided_before_execution():
    result = tp.run_workload(1, 128, simulate=False)
    # 1 x 128: the ranking's cheapest unsplit plan is the head-only mapping,
    # so the control is P1 by construction, not by a missing row.
    assert result.record["aliases"] == {"fixed-8": "P8", "head-only": "P1"}
    assert "head-only" not in result.expected


def test_summary_counts_only_complete_comparisons():
    records = [
        {"heads": 1, "tokens": 4096, "selected_partitions": 1, "best_partitions": 8, "gap": 0.0134,
         "head_only_over_selected": 2.17, "distinct_plans": 5, "split_plans_checked": 3, "failed_plans": {},
         "comparison": {"complete": True, "missing": [], "controls_complete": True}},
        {"heads": 1, "tokens": 128, "selected_partitions": 2, "best_partitions": None, "gap": None,
         "head_only_over_selected": None, "distinct_plans": 4, "split_plans_checked": 3,
         "failed_plans": {"P2": {"error": "x"}},
         "comparison": {"complete": False, "missing": ["P2"], "controls_complete": True}},
        {"heads": 32, "tokens": 128, "error": "RuntimeError: simulator"},
    ]
    summary = tp.summarize(records)
    assert summary["workloads_completed"] == 1 and summary["minima_reached"] == 0
    assert [(w["heads"], w["tokens"], w["missing"], w["failed_plans"]) for w in summary["incomplete_workloads"]] == [
        (1, 128, ["P2"], ["P2"]), (32, 128, None, []),
    ]
    assert summary["failed_plans"] == 1
    # A control that did not complete gives no head-only ratio.
    records[0]["comparison"]["controls_complete"] = False
    assert tp.summarize(records)["head_only_over_selected_1x4096"] is None


def test_simulator_failure_evidence_is_structured_and_retained(monkeypatch, tmp_path):
    """A ramulator2 invocation that exits nonzero raises with the exact
    segment trace, executor identity, command, return code and output; the
    driver keeps that evidence in the plan's retained record."""
    import subprocess

    from allo import spmw_aim

    binary = tmp_path / "ramulator2"
    binary.write_text("#!/bin/sh\nexit 7\n")
    binary.chmod(0o755)
    monkeypatch.setenv("TENON_AIM_RAMULATOR2", str(binary))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 7, stdout=b"partial stats\n", stderr=b"segmentation fault\n")

    monkeypatch.setattr(spmw_aim.subprocess, "run", fake_run)
    result = tp.run_workload(1, 128, simulate=True, artifacts=str(tmp_path / "artifacts"))
    assert set(result.failures) == set(result.expected)
    failure = result.record["plans"]["default"]["failure"]
    assert failure["stage"] == "simulate" and "status 7" in failure["error"]
    evidence = failure["evidence"]
    assert evidence["returncode"] == 7 and evidence["stderr"].startswith("segmentation fault")
    assert evidence["stdout"].startswith("partial stats")
    assert evidence["executor"]["kind"] == "native" and evidence["executor"]["binary"] == str(binary)
    assert evidence["executor"]["config"]["sha256"] and evidence["executor"]["sha256"]
    assert evidence["segment"]["lines"] > 0 and evidence["segment"]["trace"][-1].strip() == "AiM EOC"
    assert evidence["segment"]["trace_sha256"] and evidence["command"][0] == str(binary)
    assert evidence["completed_segments"] == []
    on_disk = json.load(open(tmp_path / "artifacts" / "h1_l128" / "default" / "plan.json"))
    assert on_disk["failure"]["evidence"]["segment"]["trace"] == evidence["segment"]["trace"]
    assert on_disk["provenance"]["lowering_manifest"]
    # No timing was reported for any plan.
    assert all(r.get("device_cycles") is None for r in result.record["plans"].values())
