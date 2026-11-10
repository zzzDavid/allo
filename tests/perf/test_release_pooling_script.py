# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative checks of the release pooling script `docs/release/scripts/g2_pool.py`
(round 108): a session is pooled only when it is complete and consistent;
missing evidence never implies exactness; incompatible sessions are refused."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "release" / "scripts" / "g2_pool.py"


@pytest.fixture(scope="module")
def pool():
    spec = importlib.util.spec_from_file_location("g2_pool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROUNDS = 4
REPETITIONS = 20
CASES = {"gemm": (100.0, 110.0), "mvt": (200.0, 230.0)}  # paper tenon, paper gtml
FIXTURE = {"a": "a" * 64, "b": "b" * 64, "oracle": "c" * 64}


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def make_session(
    tenon_offset=0.0,
    *,
    allo="/venv/allo/__init__.py",
    program="prog-",
    layout="lay-",
    plan="single/k1",
    fixture=None,
    launches_per_plan=1,
):
    """A session shaped like `benchmarks.apu_g2.paper_rows` writes it: the
    derived sample of every measured record is the sum of its raw repeated
    batch ticks over the repetitions, the Tenon launch items carry the same
    raw pipeline ticks, the lane summaries and the paired ratios are derived
    from those records by round, and both lanes record the fixture hashes."""
    fixture = dict(FIXTURE if fixture is None else fixture)
    cases, launches = {}, []
    for name, (paper_t, paper_g) in CASES.items():
        tenon = [paper_t - 2.0 + i + tenon_offset for i in range(ROUNDS)]
        gtml = [paper_g + i for i in range(ROUNDS)]
        for lane, samples in (("tenon", tenon), ("vendor", gtml)):
            for r, value in enumerate([samples[0]] + samples):
                kind = "warmup" if r == 0 else "measured"
                if lane == "tenon":
                    batch = [
                        value * REPETITIONS / launches_per_plan
                    ] * launches_per_plan
                else:
                    batch = [value * REPETITIONS]
                record = {
                    "case": name,
                    "lane": lane,
                    "kind": kind,
                    "round": r,
                    "outputs_exact": True,
                    "timing_evidence_problem": None,
                    "repeated_batch_ticks": batch,
                    "amortized_ticks_per_pipeline": sum(batch) / REPETITIONS,
                }
                if lane == "tenon":
                    record["expected_launches"] = launches_per_plan
                    record["launches"] = [
                        {
                            "index": i,
                            "program_fingerprint": program + name,
                            "layout_fingerprint": layout + name,
                            "packing_attestation": "ok",
                            "raw_ticks": {
                                "pipeline": batch[i],
                                "final_pipeline": value,
                            },
                        }
                        for i in range(launches_per_plan)
                    ]
                else:
                    record["host_sha256"] = "h"
                    record["device_library_sha256"] = "d"
                launches.append(record)
        ratios = [g / t for g, t in zip(gtml, tenon)]
        cases[name] = {
            "case_id": name + "_case",
            "paper_tenon": paper_t,
            "paper_gtml": paper_g,
            "lanes": {
                "tenon": {
                    "plan": plan,
                    "inputs_sha256": dict(fixture),
                    "amortized_ticks": list(tenon),
                    "statistics": {"median": _median(tenon)},
                },
                "vendor": {
                    "inputs_sha256": dict(fixture),
                    "amortized_ticks": list(gtml),
                    "statistics": {"median": _median(gtml)},
                },
            },
            "comparison": {
                "paired_round_ratios_vendor_over_tenon": ratios,
                "paired_ratio_statistics": {"median": _median(ratios)},
            },
        }
    return {
        "host": "card-host",
        "date": "2026-11-10",
        "allo": allo,
        "status": "pass",
        "problems": [],
        "protocol": {"paired_rounds": ROUNDS, "repetitions": REPETITIONS},
        "device_lock": "/tmp/lock",
        "cases": cases,
        "launches": launches,
    }


def measured(session, lane, round_id=None, case="gemm"):
    return [
        l
        for l in session["launches"]
        if l["case"] == case
        and l["lane"] == lane
        and l["kind"] == "measured"
        and (round_id is None or l["round"] == round_id)
    ]


def run(pool, tmp_path, sessions, *extra):
    files = []
    for i, s in enumerate(sessions):
        f = tmp_path / f"s{i}.json"
        f.write_text(json.dumps(s))
        files.append(str(f))
    out = tmp_path / "pooled.json"
    code = pool.main(["--out", str(out), *extra, *files])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_complete_sessions_pool_under_the_rule(pool, tmp_path):
    code, result = run(pool, tmp_path, [make_session(), make_session(tenon_offset=0.5)])
    assert code == 0 and result["rejected_sessions"] == []
    gemm = result["rows"]["gemm"]
    assert gemm["pooled_tenon"]["samples"] == 2 * ROUNDS
    assert gemm["status"] == "REPRODUCED"  # pooled median 99.75 <= 100
    assert len(gemm["session_medians"]) == 2 and gemm["outputs_exact"] is True
    assert gemm["program_fingerprint"] == "prog-gemm"


def test_nominal_and_regression_verdicts(pool, tmp_path):
    code, result = run(
        pool, tmp_path, [make_session(tenon_offset=1.0)]
    )  # samples 99..102, median 100.5, q1 99.5
    assert code == 0 and result["rows"]["gemm"]["status"] == "OPEN (nominal)"
    code, result = run(pool, tmp_path, [make_session(tenon_offset=10.0)])
    assert code == 0 and result["rows"]["gemm"]["status"] == "OPEN (regression)"


@pytest.mark.parametrize(
    "damage, reason",
    [
        (lambda s: s.update(status="fail"), "status is 'fail'"),
        (lambda s: s["problems"].append("timing"), "problems recorded"),
        (lambda s: s.update(launches=[]), "no launch records"),
        (
            lambda s: s["launches"].pop(
                next(i for i, l in enumerate(s["launches"]) if l["kind"] == "measured")
            ),
            "measured launches",
        ),
        (
            lambda s: [
                l.pop("outputs_exact")
                for l in s["launches"]
                if l["lane"] == "tenon" and l["round"] == 2
            ],
            "outputs_exact is None",
        ),
        (
            lambda s: [
                l.update(outputs_exact=False)
                for l in s["launches"]
                if l["lane"] == "vendor" and l["round"] == 3
            ],
            "outputs_exact is False",
        ),
        (
            lambda s: [
                l.update(timing_evidence_problem="clock")
                for l in s["launches"]
                if l["lane"] == "tenon" and l["round"] == 1
            ],
            "timing evidence problem",
        ),
        (
            lambda s: s["cases"]["gemm"]["lanes"]["tenon"][
                "amortized_ticks"
            ].__setitem__(0, 1.0),
            "summarized samples do not equal",
        ),
        (
            lambda s: [
                l.update(round=ROUNDS + 5)
                for l in s["launches"]
                if l["lane"] == "tenon" and l["round"] == ROUNDS
            ],
            "measured rounds",
        ),
        (
            lambda s: [
                l["launches"][0].pop("packing_attestation")
                for l in s["launches"]
                if l["lane"] == "tenon" and l["kind"] == "measured" and l["round"] == 2
            ],
            "without packing_attestation",
        ),
        (
            lambda s: [
                l["launches"][0].update(program_fingerprint="other")
                for l in s["launches"]
                if l["lane"] == "tenon" and l["kind"] == "measured" and l["round"] == 2
            ],
            "program fingerprints in one session",
        ),
        (
            lambda s: [
                l.pop("device_library_sha256")
                for l in s["launches"]
                if l["lane"] == "vendor" and l["round"] == 1
            ],
            "baseline binaries not recorded",
        ),
        (lambda s: s.pop("device_lock"), "missing field 'device_lock'"),
    ],
)
def test_incomplete_or_inconsistent_sessions_are_rejected(
    pool, tmp_path, damage, reason, capsys
):
    session = make_session()
    damage(session)
    code, result = run(pool, tmp_path, [session])
    assert code == 2 and result is None
    assert reason in capsys.readouterr().err


def test_a_rejected_session_fails_the_pool_unless_excluded_explicitly(
    pool, tmp_path, capsys
):
    bad = make_session()
    bad["status"] = "fail"
    code, result = run(pool, tmp_path, [make_session(), bad])
    assert code == 2 and result is None
    code, result = run(pool, tmp_path, [make_session(), bad], "--exclude-invalid")
    assert code == 0
    assert len(result["sessions"]) == 1 and len(result["rejected_sessions"]) == 1
    assert result["rows"]["gemm"]["pooled_tenon"]["samples"] == ROUNDS


@pytest.mark.parametrize(
    "other, reason",
    [
        (lambda: make_session(allo="/other-venv/allo/__init__.py"), "allo differs"),
        (lambda: make_session(program="other-"), "program_fingerprints differs"),
        (lambda: make_session(layout="other-"), "layout_fingerprints differs"),
        (lambda: make_session(plan="single/k2"), "plan differs"),
    ],
)
def test_incompatible_provenance_is_refused(pool, tmp_path, other, reason, capsys):
    code, result = run(pool, tmp_path, [make_session(), other()])
    assert code == 2 and result is None
    assert reason in capsys.readouterr().err


def test_missing_evidence_never_implies_exactness(pool, tmp_path, capsys):
    session = make_session()
    for launch in session["launches"]:
        launch.pop("outputs_exact", None)
    code, result = run(pool, tmp_path, [session])
    assert code == 2 and result is None
    assert "outputs_exact is None" in capsys.readouterr().err


def test_complete_sessions_with_several_launches_per_plan_pool(pool, tmp_path):
    code, result = run(pool, tmp_path, [make_session(launches_per_plan=3)])
    assert code == 0 and result["rows"]["gemm"]["pooled_tenon"]["samples"] == ROUNDS
    assert result["rows"]["gemm"]["inputs_sha256"] == FIXTURE


def _set_derived_only(session):
    """The reviewer's raw-disagreement probe: derived samples and summaries
    changed to 1.0, every raw tick unchanged."""
    for l in measured(session, "tenon"):
        l["amortized_ticks_per_pipeline"] = 1.0
    lane = session["cases"]["gemm"]["lanes"]["tenon"]
    lane["amortized_ticks"] = [1.0] * ROUNDS
    lane["statistics"]["median"] = 1.0


def _shift_vendor_rounds(session):
    """The reviewer's unpaired-rounds probe: every vendor measured round moved by +100."""
    for l in measured(session, "vendor"):
        l["round"] += 100


def _ratios_contradict(session):
    session["cases"]["gemm"]["comparison"]["paired_round_ratios_vendor_over_tenon"][
        0
    ] *= 2


def _ratio_median_contradicts(session):
    session["cases"]["gemm"]["comparison"]["paired_ratio_statistics"]["median"] = 9.0


def _raw_zero(session):
    l = measured(session, "tenon", 2)[0]
    l["repeated_batch_ticks"] = [0.0]
    l["launches"][0]["raw_ticks"]["pipeline"] = 0.0
    l["amortized_ticks_per_pipeline"] = 0.0


def _raw_not_finite(session):
    l = measured(session, "vendor", 2)[0]
    l["repeated_batch_ticks"] = [float("inf")]
    l["amortized_ticks_per_pipeline"] = float("inf")


def _raw_missing(session):
    measured(session, "tenon", 3)[0].pop("repeated_batch_ticks")


def _item_pipeline_disagrees(session):
    measured(session, "tenon", 1)[0]["launches"][0]["raw_ticks"]["pipeline"] += 1


def _item_without_raw_ticks(session):
    measured(session, "tenon", 1)[0]["launches"][0].pop("raw_ticks")


def _batch_count_disagrees(session):
    l = measured(session, "tenon", 4)[0]
    l["repeated_batch_ticks"] = l["repeated_batch_ticks"] + [1.0]


def _timing_field_absent(session):
    measured(session, "vendor", 1)[0].pop("timing_evidence_problem")


def _contradictory_launch_error(session):
    # a recorded problem next to otherwise complete evidence is a rejection
    measured(session, "tenon", 2)[0][
        "timing_evidence_problem"
    ] = "1 launch record(s) for a plan of 2 launch(es)"


def _lanes_differ_in_fixture(session):
    session["cases"]["gemm"]["lanes"]["tenon"]["inputs_sha256"] = {
        "a": "e" * 64,
        "b": "b" * 64,
        "oracle": "c" * 64,
    }


def _fixture_without_oracle(session):
    for lane in session["cases"]["gemm"]["lanes"].values():
        lane["inputs_sha256"] = {"changed-input": "completely-different-fixture"}


def _fixture_missing(session):
    session["cases"]["gemm"]["lanes"]["vendor"].pop("inputs_sha256")


@pytest.mark.parametrize(
    "damage, reason",
    [
        (_set_derived_only, "derived sample 1.0 disagrees with the raw ticks"),
        (_shift_vendor_rounds, "gemm/vendor: measured rounds [101"),
        (
            _ratios_contradict,
            "paired vendor/tenon ratios do not equal the ratios re-derived by round",
        ),
        (_ratio_median_contradicts, "paired-ratio median differs"),
        (_raw_zero, "raw device ticks 0.0 are not a positive finite count"),
        (_raw_not_finite, "raw device ticks inf are not a positive finite count"),
        (_raw_missing, "no raw repeated-batch ticks"),
        (_item_pipeline_disagrees, "raw_ticks.pipeline"),
        (_item_without_raw_ticks, "raw_ticks.pipeline None does not match"),
        (_batch_count_disagrees, "batch records for a plan of 1 launch"),
        (_timing_field_absent, "timing-evidence field absent"),
        (_contradictory_launch_error, "timing evidence problem '1 launch record(s)"),
        (_lanes_differ_in_fixture, "different fixtures (inputs_sha256)"),
        (_fixture_without_oracle, "inputs_sha256 records no oracle"),
        (_fixture_missing, "no inputs_sha256 recorded"),
    ],
)
def test_raw_evidence_pairing_and_fixture_inconsistencies_are_rejected(
    pool, tmp_path, damage, reason, capsys
):
    session = make_session()
    damage(session)
    code, result = run(pool, tmp_path, [session])
    assert code == 2 and result is None
    assert reason in capsys.readouterr().err


def test_sessions_on_different_fixtures_do_not_pool(pool, tmp_path, capsys):
    other = make_session(fixture={"a": "f" * 64, "b": "b" * 64, "oracle": "c" * 64})
    code, result = run(pool, tmp_path, [make_session(), other])
    assert code == 2 and result is None
    assert "inputs differs from the first session" in capsys.readouterr().err


def test_pooled_samples_are_the_raw_derived_values(pool, tmp_path):
    session = make_session()
    # the raw ticks decide: change a raw record and its derivation consistently
    l = measured(session, "tenon", 1)[0]
    l["repeated_batch_ticks"] = [REPETITIONS * 50.0]
    l["launches"][0]["raw_ticks"]["pipeline"] = REPETITIONS * 50.0
    l["amortized_ticks_per_pipeline"] = 50.0
    lane = session["cases"]["gemm"]["lanes"]["tenon"]
    lane["amortized_ticks"][0] = 50.0
    lane["statistics"]["median"] = _median(lane["amortized_ticks"])
    comp = session["cases"]["gemm"]["comparison"]
    comp["paired_round_ratios_vendor_over_tenon"][0] = (
        session["cases"]["gemm"]["lanes"]["vendor"]["amortized_ticks"][0] / 50.0
    )
    comp["paired_ratio_statistics"]["median"] = _median(
        comp["paired_round_ratios_vendor_over_tenon"]
    )
    code, result = run(pool, tmp_path, [session])
    assert code == 0
    assert result["rows"]["gemm"]["pooled_tenon"]["min"] == 50.0
