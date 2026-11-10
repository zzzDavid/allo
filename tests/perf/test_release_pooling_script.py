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
CASES = {"gemm": (100.0, 110.0), "mvt": (200.0, 230.0)}  # paper tenon, paper gtml


def make_session(
    tenon_offset=0.0,
    *,
    allo="/venv/allo/__init__.py",
    program="prog-",
    layout="lay-",
    plan="single/k1",
):
    cases, launches = {}, []
    for name, (paper_t, paper_g) in CASES.items():
        tenon = [paper_t - 2.0 + i + tenon_offset for i in range(ROUNDS)]
        gtml = [paper_g + i for i in range(ROUNDS)]
        for lane, samples in (("tenon", tenon), ("vendor", gtml)):
            launches.append(
                {
                    "case": name,
                    "lane": lane,
                    "kind": "warmup",
                    "round": 0,
                    "outputs_exact": True,
                    "timing_evidence_problem": None,
                    "amortized_ticks_per_pipeline": samples[0],
                    "expected_launches": 1,
                    "launches": [],
                    "host_sha256": "h",
                    "device_library_sha256": "d",
                }
            )
            for r, value in enumerate(samples, start=1):
                record = {
                    "case": name,
                    "lane": lane,
                    "kind": "measured",
                    "round": r,
                    "outputs_exact": True,
                    "timing_evidence_problem": None,
                    "amortized_ticks_per_pipeline": value,
                }
                if lane == "tenon":
                    record["expected_launches"] = 1
                    record["launches"] = [
                        {
                            "program_fingerprint": program + name,
                            "layout_fingerprint": layout + name,
                            "packing_attestation": "ok",
                        }
                    ]
                else:
                    record["host_sha256"] = "h"
                    record["device_library_sha256"] = "d"
                launches.append(record)
        cases[name] = {
            "case_id": name + "_case",
            "paper_tenon": paper_t,
            "paper_gtml": paper_g,
            "lanes": {
                "tenon": {
                    "plan": plan,
                    "amortized_ticks": list(tenon),
                    "statistics": {
                        "median": sorted(tenon)[ROUNDS // 2 - 1] / 2
                        + sorted(tenon)[ROUNDS // 2] / 2
                    },
                },
                "vendor": {"amortized_ticks": list(gtml)},
            },
            "comparison": {
                "paired_round_ratios_vendor_over_tenon": [
                    g / t for g, t in zip(gtml, tenon)
                ]
            },
        }
    return {
        "host": "card-host",
        "date": "2026-11-10",
        "allo": allo,
        "status": "pass",
        "problems": [],
        "protocol": {"paired_rounds": ROUNDS},
        "device_lock": "/tmp/lock",
        "cases": cases,
        "launches": launches,
    }


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
            "measured rounds covered",
        ),
        (
            lambda s: [
                l["launches"][0].pop("packing_attestation")
                for l in s["launches"]
                if l["lane"] == "tenon" and l["kind"] == "measured" and l["round"] == 2
            ],
            "launch without packing_attestation",
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
