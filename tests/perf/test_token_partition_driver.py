# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token-partitioning reproduction driver (release gate G10): its
compared plans come from the shared ranking inside the paper's channel
budget, each is compiled as an explicit layout with its provenance kept,
a run of the wrong kind is refused, and the summary reports the study's
figures against the paper's."""

from __future__ import annotations

import json

import pytest

from benchmarks.cent_aim import token_partition as tp


def test_shapes_and_summary_figures():
    assert tp._parse_shapes("1x128,8x512") == ((1, 128), (8, 512))
    records = [
        {"heads": 1, "tokens": 4096, "selected_partitions": 1, "best_partitions": 8,
         "gap": 0.0134, "head_only_over_selected": 2.17},
        {"heads": 8, "tokens": 128, "selected_partitions": 1, "best_partitions": 1,
         "gap": 0.0, "head_only_over_selected": 1.0},
        {"heads": 32, "tokens": 128, "error": "RuntimeError: simulator"},
    ]
    summary = tp.summarize(records)
    assert summary["workloads_completed"] == 2
    assert summary["minima_reached"] == 1
    assert summary["max_gap"] == 0.0134
    assert summary["head_only_over_selected_1x4096"] == 2.17
    assert summary["paper"] == tp.PAPER


def test_estimates_only_run_keeps_every_compared_plan_and_its_provenance():
    result = tp.run_workload(1, 128, simulate=False)
    rows = result.rows
    assert {f"P{p}" for p in tp.PARTITIONS} <= set(rows)
    for label, row in rows.items():
        assert row.device is None and row.total is None and row.execution is None
        assert row.estimate > 0 and row.ranked_cycles > 0
        assert tp.within_budget(row.mode)
        assert row.partitions == tp.partitions_of(row.mode)
        placements = row.provenance["placements"]
        assert placements and all(p["effects"] is not None for p in placements)
        assert row.provenance["sv_lowering"]["kind"] == "contraction"
        if row.partitions > 1:
            assert row.host > 0 and row.bf16_error is not None
            assert row.provenance["host_steps"]
        else:
            assert row.host == 0 and row.bf16_error is None
    assert result.selected == min(
        tp.PARTITIONS, key=lambda p: (rows[f"P{p}"].ranked_cycles, p)
    )
    record = result.record
    assert record["gap"] is None and record["channel_budget"] == tp.CHANNEL_BUDGET
    assert record["default_selection_unrestricted"]
    json.dumps(record, default=str)  # JSON-ready


def test_a_run_of_the_wrong_kind_is_refused(monkeypatch):
    from types import SimpleNamespace

    def wrong_kind(self):
        return SimpleNamespace(kind="virtual", cycles=1, extra={})

    from allo.spmw_codegen import Compiled

    monkeypatch.setattr(tp, "compile_variant", _compile_then(wrong_kind))
    with pytest.raises(RuntimeError, match="simulator-timing-only"):
        tp.run_workload(1, 128, simulate=True)


def _compile_then(run_backend):
    original = tp.compile_variant

    def compile_variant(*args, **kwargs):
        built = original(*args, **kwargs)
        built.run_backend = run_backend.__get__(built, type(built))
        return built

    return compile_variant
