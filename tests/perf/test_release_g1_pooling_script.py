# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative and positive checks of the maintained Gemini-I pooler
`docs/release/scripts/g1_pool.py` (round 142, ISSUES X-46): samples come from
the launch records, never from cached counts; a row is complete only with
its warm-ups, its measured rounds once each in every required lane, exact
outputs, positive agreeing profiler counters, consistent identities and no
recorded problem; rows pool per row (Gemini-I's rule); duplicate or
incompatible sessions are refused; the quota governs the exit status."""

from __future__ import annotations

import copy
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "release" / "scripts" / "g1_pool.py"
LANES = ("tenon", "expert", "historical_binary")
PAPER = {
    "kmeans": (624677, 641471),
    "word_count": (171988, 175397.5),
    "linear_regression": (94618.5, 95785),
}
BIN = {"expert": "b" * 64, "historical_binary": "0" * 64}
SOURCE = "4" * 64
ORACLE = "c" * 64
LOCK = "/tmp/tenon-apu-v1-0000:41:00.0.lock"


@pytest.fixture(scope="module")
def pool():
    spec = importlib.util.spec_from_file_location("g1_pool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch(case, lane, kind, rnd, crun, session_tag):
    launch = {
        "crun": crun,
        "counters": {
            "hits": 1,
            "seu": 100 + rnd,
            "crun": crun,
            "iall": 1000 + rnd,
            "icm": 5,
            "dcm": 1,
        },
        "outputs_match_oracle": True,
        "binary_sha256": BIN.get(lane, f"{rnd:064x}" if lane == "tenon" else ""),
        "binary_bytes": 1000,
        "board_lock": LOCK,
        "launch_dir": f"/evidence/{session_tag}/{case}/{lane}-launch-{rnd:03d}",
        "lane": lane,
        "kind": kind,
        "round": rnd,
        "outputs_sha256": {"out": ORACLE},
    }
    if lane == "tenon":
        launch["launched_source_sha256"] = SOURCE
        launch["route"] = "gvml_route"
    return launch


def make_session(
    tag,
    *,
    offset=0,
    cases=("kmeans", "word_count", "linear_regression"),
    measured=10,
    allo="/r/pkg/venv/lib/python3.12/site-packages/allo/__init__.py",
    label="default_selection",
    perturbation=None,
):
    """A session in the driver's own shape: for every row, the lanes'
    warm-ups in protocol order, then rounds 1..N with the lanes in that
    order on odd rounds and reversed on even rounds (paper_rows.py)."""
    records = []
    jitter = sum(map(ord, tag)) % 7  # sessions differ, as real ones do
    for case in cases:
        paper, paper_expert = PAPER[case]

        def crun_of(lane, rnd):
            base = int(paper) + offset
            base += (
                20000
                if lane == "expert"
                else (500 if lane == "historical_binary" else 0)
            )
            return base + ((rnd + jitter) % 3) * 40 - 40 + jitter

        launches = [
            _launch(case, lane, "warmup", 0, crun_of(lane, 0) + 7, tag)
            for lane in LANES
        ]
        for rnd in range(1, measured + 1):
            order = list(LANES) if rnd % 2 == 1 else list(reversed(LANES))
            for lane in order:
                launches.append(
                    _launch(case, lane, "measured", rnd, crun_of(lane, rnd), tag)
                )
        samples = {
            lane: [
                l["crun"]
                for l in launches
                if l["lane"] == lane and l["kind"] == "measured"
            ]
            for lane in LANES
        }
        lanes = {
            "tenon": {
                "lane": "tenon",
                "native_vector_route": "gvml_route",
                "launched_source_sha256": SOURCE,
                "device_source_sha256": SOURCE,
                "substituted_source": None,
                "cost_fingerprint": "0c6e5955e1663302",
                "inputs_sha256": {"points": "1" * 64},
                "oracle_sha256": ORACLE,
            },
            "expert": {
                "lane": "expert",
                "binary_sha256": BIN["expert"],
                "binary_matches_archive": True,
                "inputs_sha256": {"points": "2" * 64},
                "oracle_sha256": ORACLE,
            },
            "historical_binary": {
                "lane": "historical_binary",
                "binary_sha256": BIN["historical_binary"],
                "binary_matches_archive": True,
                "inputs_sha256": {"points": "3" * 64},
                "oracle_sha256": ORACLE,
            },
        }
        for lane in LANES:
            lanes[lane]["measured_crun"] = list(samples[lane])
            lanes[lane]["statistics"] = {
                "samples": len(samples[lane]),
                "median": statistics.median(samples[lane]),
            }
        records.append(
            {
                "case": case,
                "label": label,
                "backend": "device",
                "paper_expert_cycles": paper_expert,
                "paper_tenon_cycles": paper,
                "protocol": {
                    "warmups_per_lane": 1,
                    "measured_per_lane": measured,
                    "interleaved_with_expert": True,
                    "lanes": list(LANES),
                },
                "lanes": lanes,
                "launches": launches,
                "problems": [],
                "board_lock": LOCK,
            }
        )
    return {
        "host": "board-host",
        "date": f"2026-11-11T{tag}",
        "python": "/r/pkg",
        "allo": allo,
        "perturbation": perturbation,
        "records": records,
        "status": "pass",
    }


def _write(tmp_path, sessions):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for i, s in enumerate(sessions, 1):
        f = tmp_path / f"s{i}.json"
        f.write_text(json.dumps(s))
        files.append(str(f))
    return files


def _run(pool, tmp_path, sessions, *extra):
    files = _write(tmp_path, sessions)
    out = tmp_path / "pooled.json"
    code = pool.main(["--out", str(out), "--quota", "3", *extra, *files])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_complete_sessions_pool_per_row_under_the_rule(pool, tmp_path):
    sessions = [
        make_session("s1", offset=-100),
        make_session("s2", offset=0),
        make_session("s3", offset=100),
    ]
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 0 and pooled["campaign_complete"]
    row = pooled["rows"]["kmeans"]
    assert row["complete_sessions"] == 3 and row["pooled_tenon"]["samples"] == 30
    expected = sorted(
        c
        for s in sessions
        for l in s["records"][0]["launches"]
        if l["lane"] == "tenon" and l["kind"] == "measured"
        for c in [l["crun"]]
    )
    assert row["pooled_tenon"]["median"] == statistics.median(expected)
    assert row["status"] in ("REPRODUCED", "OPEN (nominal)", "OPEN (regression)")
    assert (
        row["outputs_exact"] is True
        and row["paired_expert_over_tenon"]["samples"] == 30
    )
    # verdicts follow the paper's position against the pooled samples
    below = [
        make_session("a", offset=-200),
        make_session("b", offset=-200),
        make_session("c", offset=-200),
    ]
    assert (
        _run(pool, tmp_path / "below", below)[1]["rows"]["kmeans"]["status"]
        == "REPRODUCED"
    )
    above = [
        make_session("a", offset=500),
        make_session("b", offset=500),
        make_session("c", offset=500),
    ]
    assert (
        _run(pool, tmp_path / "above", above)[1]["rows"]["kmeans"]["status"]
        == "OPEN (regression)"
    )
    inside = [
        make_session("a", offset=10),
        make_session("b", offset=10),
        make_session("c", offset=10),
    ]
    assert (
        _run(pool, tmp_path / "inside", inside)[1]["rows"]["kmeans"]["status"]
        == "OPEN (nominal)"
    )


def test_a_row_lost_in_one_session_excludes_that_row_only(pool, tmp_path):
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    record = sessions[0]["records"][2]  # linear regression of session 1
    lost = [
        l
        for l in record["launches"]
        if l["lane"] == "historical_binary" and l["round"] == 6
    ][0]
    lost.clear()
    lost.update(
        {
            "error": "APUV1LaunchFailure: lost profile",
            "reason": "missing_or_stale_profile",
            "crun": None,
            "lane": "historical_binary",
            "kind": "measured",
            "round": 6,
        }
    )
    record["problems"] = [
        "historical_binary measured 6: APUV1LaunchFailure: lost profile"
    ]
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and not pooled["campaign_complete"]
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 3
    assert pooled["rows"]["linear_regression"]["complete_sessions"] == 2
    assert "s1.json" in "".join(
        pooled["rows"]["linear_regression"]["incomplete_sessions"]
    )


def test_cached_claims_without_launches_are_not_evidence(pool, tmp_path):
    """The round 141 review's first probe: no launch records, cached ten-sample claims."""
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    for s in sessions:
        s["status"] = "fail (3 case(s))"
        for record in s["records"]:
            record["launches"] = []
            record["lanes"]["tenon"]["measured_crun"] = [1] * 10
            record["lanes"]["tenon"]["statistics"] = {"samples": 10, "median": 1}
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and not pooled["campaign_complete"]
    assert all(
        r["complete_sessions"] == 0 and r["status"] == "no complete session"
        for r in pooled["rows"].values()
    )
    assert "no launch records" in json.dumps(
        pooled["rows"]["kmeans"]["incomplete_sessions"]
    )


def test_cached_samples_disagreeing_with_the_launch_records_are_rejected(
    pool, tmp_path
):
    """The round 141 review's second probe: real launch counters, cached one-cycle samples."""
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    for s in sessions:
        for record in s["records"]:
            record["lanes"]["tenon"]["measured_crun"] = [1] * 10
            record["lanes"]["tenon"]["statistics"] = {"samples": 10, "median": 1}
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and all(
        r["complete_sessions"] == 0 for r in pooled["rows"].values()
    )
    assert "cached measured_crun differs from the launch records" in json.dumps(
        pooled["rows"]
    )
    # a cached statistics block that disagrees with the launches is refused too
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    sessions[1]["records"][0]["lanes"]["tenon"]["statistics"]["median"] = 1
    code, pooled = _run(pool, tmp_path / "stats", sessions)
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 2
    assert "cached statistics differ" in json.dumps(
        pooled["rows"]["kmeans"]["incomplete_sessions"]
    )


@pytest.mark.parametrize(
    "damage, reason",
    [
        (
            lambda r: [
                l
                for l in r["launches"]
                if l["kind"] == "warmup" and l["lane"] == "tenon"
            ][0].update(kind="measured", round=1),
            "not the driver's interleaved order",
        ),
        (
            lambda r: r["launches"].remove(
                [
                    l
                    for l in r["launches"]
                    if l["kind"] == "warmup" and l["lane"] == "expert"
                ][0]
            ),
            "not the driver's interleaved order",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "tenon" and l["round"] == 4
            ][0].update(outputs_match_oracle=False),
            "outputs_match_oracle is False",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "expert" and l["round"] == 2
            ][0]["counters"].update(seu=0),
            "profiler seu 0",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "expert" and l["round"] == 2
            ][0]["counters"].update(hits=2),
            "profiler hits 2",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "tenon" and l["round"] == 9
            ][0]["counters"].update(crun=5),
            "differs from the recorded crun",
        ),
        (
            lambda r: [
                l
                for l in r["launches"]
                if l["lane"] == "historical_binary" and l["round"] == 3
            ][0].update(binary_sha256="f" * 64),
            "several binaries launched",
        ),
        (
            lambda r: r["lanes"]["expert"].update(binary_matches_archive=False),
            "binary_matches_archive is False",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "tenon" and l["round"] == 1
            ][0].update(launched_source_sha256="e" * 64),
            "not the lane's declared launched source",
        ),
        (
            lambda r: r["lanes"]["historical_binary"].update(oracle_sha256="d" * 64),
            "records a different oracle",
        ),
        (
            lambda r: [
                l for l in r["launches"] if l["lane"] == "tenon" and l["round"] == 2
            ][0].pop("board_lock"),
            "no board lock recorded",
        ),
        (lambda r: r["problems"].append("tenon measured 3: lost"), "problems recorded"),
    ],
)
def test_damaged_rows_are_incomplete(pool, tmp_path, damage, reason):
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    damage(sessions[2]["records"][1])  # word count of session 3
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3
    assert pooled["rows"]["word_count"]["complete_sessions"] == 2
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 3
    assert reason in json.dumps(pooled["rows"]["word_count"]["incomplete_sessions"])


def test_duplicate_and_incompatible_sessions_are_refused(pool, tmp_path, capsys):
    base = make_session("s1")
    duplicate = copy.deepcopy(base)
    duplicate["date"] = "later"
    code, pooled = _run(pool, tmp_path / "dup", [base, duplicate, make_session("s3")])
    assert code == 2 and pooled is None
    assert "duplicate evidence" in capsys.readouterr().err
    # the same evidence under renamed directories is still a duplicate
    renamed = copy.deepcopy(base)
    for record in renamed["records"]:
        for l in record["launches"]:
            l["launch_dir"] = l["launch_dir"].replace("/s1/", "/copy/")
    code, _ = _run(pool, tmp_path / "dup2", [base, renamed, make_session("s3")])
    assert code == 2 and "same measured evidence" in capsys.readouterr().err
    for key, mutate in (
        ("allo", lambda s: s.update(allo="/other/venv/allo/__init__.py")),
        (
            "archived binary",
            lambda s: (
                s["records"][0]["lanes"]["expert"].update(binary_sha256="a" * 64),
                [
                    l.update(binary_sha256="a" * 64)
                    for l in s["records"][0]["launches"]
                    if l["lane"] == "expert"
                ],
            ),
        ),
        (
            "tenon_source",
            lambda s: (
                s["records"][0]["lanes"]["tenon"].update(
                    launched_source_sha256="9" * 64, device_source_sha256="9" * 64
                ),
                [
                    l.update(launched_source_sha256="9" * 64)
                    for l in s["records"][0]["launches"]
                    if l["lane"] == "tenon"
                ],
            ),
        ),
        (
            "oracle",
            lambda s: [
                lane.update(oracle_sha256="8" * 64)
                for lane in s["records"][0]["lanes"].values()
            ],
        ),
        ("paper_tenon", lambda s: s["records"][0].update(paper_tenon_cycles=1)),
    ):
        other = make_session("s2", offset=50)
        mutate(other)
        code, _ = _run(
            pool,
            tmp_path / key.replace(" ", "_"),
            [make_session("s1"), other, make_session("s3", offset=20)],
        )
        assert code == 2, key
        assert "differs from" in capsys.readouterr().err
    # --exclude-invalid pools the rest and lists the rejected file
    other = make_session("s2", offset=50)
    other["allo"] = "/other/allo"
    code, pooled = _run(
        pool,
        tmp_path / "excl",
        [make_session("s1"), other, make_session("s3", offset=20)],
        "--exclude-invalid",
    )
    assert (
        code == 3
        and len(pooled["rejected_sessions"]) == 1
        and pooled["rows"]["kmeans"]["complete_sessions"] == 2
    )


def test_quota_governs_the_exit_status_not_the_statistics(pool, tmp_path):
    sessions = [make_session("s1"), make_session("s2")]
    files = _write(tmp_path, sessions)
    out = tmp_path / "pooled.json"
    assert pool.main(["--out", str(out), "--quota", "3", *files]) == 3
    assert pool.main(["--out", str(out), "--quota", "2", *files]) == 0
    pooled = json.loads(out.read_text())
    assert (
        pooled["rows"]["kmeans"]["complete_sessions"] == 2
        and pooled["rows"]["kmeans"]["quota"] == 2
    )


PRESERVED = Path("/work/shared/users/phd/nz264/tenon-release-evidence/r141/campaign")


def test_the_preserved_round_141_campaign_pools_only_its_complete_rows(pool, tmp_path):
    files = [PRESERVED / f"apu_v1_campaign_s{i}.json" for i in (1, 2, 3)]
    if not all(f.is_file() for f in files):
        pytest.skip("the round 141 campaign evidence is not on this host")
    out = tmp_path / "pooled.json"
    code = pool.main(["--out", str(out), "--quota", "3", *map(str, files)])
    pooled = json.loads(out.read_text())
    assert code == 3 and not pooled["campaign_complete"]
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 1
    assert pooled["rows"]["kmeans"]["pooled_tenon"]["median"] == 624678.0
    assert pooled["rows"]["kmeans"]["status"] == "OPEN (nominal)"
    assert pooled["rows"]["word_count"]["pooled_tenon"]["median"] == 172199.0
    assert pooled["rows"]["word_count"]["status"] == "OPEN (nominal)"
    assert pooled["rows"]["linear_regression"]["complete_sessions"] == 0
    assert pooled["plan"]["remaining_complete_rows"] == {
        "kmeans": 2,
        "word_count": 2,
        "linear_regression": 3,
    }


def test_lane_major_or_reordered_records_are_refused(pool, tmp_path):
    """The round 142 review's fourth probe: the real session's kmeans row
    with its launches grouped by lane (cycles, counters and identities
    unchanged) violates the interleaved protocol and is not evidence."""
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    sessions[1]["records"][0]["launches"].sort(key=lambda l: l["lane"])
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and pooled["rows"]["kmeans"]["complete_sessions"] == 2
    assert "not the driver's interleaved order" in json.dumps(
        pooled["rows"]["kmeans"]["incomplete_sessions"]
    )
    # swapping two lanes inside one round is refused too (the alternating order is the protocol)
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    launches = sessions[2]["records"][1]["launches"]
    i = next(
        k for k, l in enumerate(launches) if l["kind"] == "measured" and l["round"] == 2
    )
    launches[i], launches[i + 1] = launches[i + 1], launches[i]
    code, pooled = _run(pool, tmp_path / "swap", sessions)
    assert pooled["rows"]["word_count"]["complete_sessions"] == 2
    assert pool.expected_launch_sequence(("a", "b"), 2, 1) == [
        ("a", "warmup", 0),
        ("b", "warmup", 0),
        ("a", "measured", 1),
        ("b", "measured", 1),
        ("b", "measured", 2),
        ("a", "measured", 2),
    ]


def test_missing_tenon_identities_are_not_accepted(pool, tmp_path):
    """The round 142 review's second probe: None equal to None is not an identity."""
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    for s in sessions:
        for r in s["records"]:
            for key in (
                "launched_source_sha256",
                "native_vector_route",
                "device_source_sha256",
                "cost_fingerprint",
            ):
                r["lanes"]["tenon"].pop(key, None)
            for l in r["launches"]:
                l.pop("launched_source_sha256", None)
                l.pop("route", None)
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and all(
        r["complete_sessions"] == 0 for r in pooled["rows"].values()
    )
    text = json.dumps(pooled["rows"]["kmeans"]["incomplete_sessions"])
    assert "not the lane's declared launched source" in text or "no valid" in text
    # each identity alone is required
    for key, reason in (
        ("device_source_sha256", "no valid compiler-selected source"),
        ("cost_fingerprint", "no cost fingerprint"),
        ("native_vector_route", "exactly one route form"),
    ):
        sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
        sessions[0]["records"][0]["lanes"]["tenon"].pop(key)
        if key == "native_vector_route":
            for l in sessions[0]["records"][0]["launches"]:
                l.pop("route", None)
        code, pooled = _run(pool, tmp_path / key, sessions)
        assert pooled["rows"]["kmeans"]["complete_sessions"] == 2, key
        assert reason in json.dumps(
            pooled["rows"]["kmeans"]["incomplete_sessions"]
        ), key
    # a vector plan row (selected_plan, launches without a native route) is a valid route form
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    for s in sessions:
        r = s["records"][0]
        r["lanes"]["tenon"].pop("native_vector_route")
        r["lanes"]["tenon"][
            "selected_plan"
        ] = "temporal_dma_coalescing_broadcast_friendly_acc8"
        for l in r["launches"]:
            if l["lane"] == "tenon":
                l["route"] = None
    code, pooled = _run(pool, tmp_path / "plan", sessions)
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 3
    assert (
        pooled["rows"]["kmeans"]["tenon_route"]
        == "temporal_dma_coalescing_broadcast_friendly_acc8"
    )


def test_substituted_program_controls_are_reported_not_pooled(pool, tmp_path):
    """The round 142 review's third probe: a historical-source control (the
    driver's --device-source / --control-source format) is a measurement of
    another program; it is kept as a labelled control and never counts as
    the default selection's reproduction."""
    sessions = [
        make_session("s1"),
        make_session("s2"),
        make_session("s3", label="historical_program"),
    ]
    control = sessions[2]
    for r in control["records"]:
        lane = r["lanes"]["tenon"]
        lane["substituted_source"] = {"path": "/archive/old.c", "sha256": "e" * 64}
        lane["launched_source_sha256"] = "e" * 64
        for l in r["launches"]:
            if l["lane"] == "tenon":
                l["launched_source_sha256"] = "e" * 64
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 3 and pooled["rows"]["kmeans"]["complete_sessions"] == 2
    controls = pooled["rows"]["kmeans"]["controls"]
    assert len(controls) == 1
    (control_record,) = controls.values()
    assert (
        control_record["label"] == "historical_program"
        and "substituted source" in control_record["reason"]
    )
    assert (
        control_record["launched_source_sha256"] == "e" * 64
        and control_record["selected_source_sha256"] == SOURCE
    )
    # a default-labelled row whose launched source is not the selected source is not evidence either
    sessions = [make_session("s1"), make_session("s2"), make_session("s3")]
    r = sessions[0]["records"][0]
    r["lanes"]["tenon"]["launched_source_sha256"] = "e" * 64
    for l in r["launches"]:
        if l["lane"] == "tenon":
            l["launched_source_sha256"] = "e" * 64
    code, pooled = _run(pool, tmp_path / "bound", sessions)
    assert pooled["rows"]["kmeans"]["complete_sessions"] == 2
    assert "not the compiler-selected source" in json.dumps(
        pooled["rows"]["kmeans"]["incomplete_sessions"]
    )
    # a forced-plan label and a perturbed session are controls as well
    sessions = [
        make_session("s1"),
        make_session("s2", label="forced:acc4"),
        make_session(
            "s3",
            perturbation={
                "constant": "MICRO_DMA_L4_L1_32K",
                "stock": 22272,
                "perturbed": 222,
            },
        ),
    ]
    code, pooled = _run(pool, tmp_path / "labels", sessions)
    assert (
        pooled["rows"]["kmeans"]["complete_sessions"] == 1
        and len(pooled["rows"]["kmeans"]["controls"]) == 2
    )


def test_rows_are_compared_with_their_first_complete_occurrence(pool, tmp_path, capsys):
    """The round 142 review's first probe: with the first session's linear
    regression incomplete, later complete linear regression rows carrying
    different sources, fixtures or paper values are refused against each
    other, whatever the input order."""
    sessions = [make_session(f"s{i}", offset=-100 + i) for i in range(1, 5)]
    sessions[0]["records"][2]["problems"] = ["lost archived launch"]
    for i, s in enumerate(sessions[2:], 2):
        r = s["records"][2]
        r["paper_tenon_cycles"] *= 10**i
        r["lanes"]["tenon"]["launched_source_sha256"] = str(i) * 64
        r["lanes"]["tenon"]["device_source_sha256"] = str(i) * 64
        r["lanes"]["tenon"]["inputs_sha256"] = {"points": str(i) * 64}
        for l in r["launches"]:
            if l["lane"] == "tenon":
                l["launched_source_sha256"] = str(i) * 64
    code, pooled = _run(pool, tmp_path, sessions)
    assert code == 2 and pooled is None
    err = capsys.readouterr().err
    assert (
        "linear_regression" in err
        and "first complete occurrence" in err
        and "s2.json" in err
    )
    # the same sessions in another order are refused the same way
    code, _ = _run(pool, tmp_path / "reversed", list(reversed(sessions)))
    assert code == 2 and "first complete occurrence" in capsys.readouterr().err


def test_per_row_quota_and_the_remaining_plan(pool, tmp_path):
    """The campaign's remaining per-row plan counts existing complete rows:
    with one complete kmeans and word count row and no linear regression
    row, two more kmeans and word count rows and three linear regression
    rows are owed; a session is planned only for rows still owed."""
    first = make_session("s1")
    first["records"][2]["problems"] = ["historical_binary measured 6: lost"]
    files = _write(tmp_path, [first])
    out = tmp_path / "pooled.json"
    plan = tmp_path / "plan.json"
    code = pool.main(
        [
            "--out",
            str(out),
            "--row-quota",
            "kmeans=3,word_count=3,linear_regression=3",
            "--plan-out",
            str(plan),
            *files,
        ]
    )
    assert code == 3
    p = json.loads(plan.read_text())
    assert p["remaining_complete_rows"] == {
        "kmeans": 2,
        "word_count": 2,
        "linear_regression": 3,
    }
    assert sorted(p["next_session_cases"]) == [
        "kmeans",
        "linear_regression",
        "word_count",
    ]
    # two more complete sessions of every row: kmeans and word count reach their quota, linear regression still owes one
    files = _write(tmp_path / "more", [first, make_session("s2"), make_session("s3")])
    code = pool.main(
        [
            "--out",
            str(out),
            "--row-quota",
            "kmeans=3,word_count=3,linear_regression=3",
            "--plan-out",
            str(plan),
            *files,
        ]
    )
    p = json.loads(plan.read_text())
    assert (
        code == 3
        and p["remaining_complete_rows"] == {"linear_regression": 1}
        and p["next_session_cases"] == ["linear_regression"]
    )
    # a session of only that row completes the campaign; extra complete rows are not needed and not planned
    files = _write(
        tmp_path / "done",
        [
            first,
            make_session("s2"),
            make_session("s3"),
            make_session("s4", cases=("linear_regression",)),
        ],
    )
    code = pool.main(
        [
            "--out",
            str(out),
            "--row-quota",
            "kmeans=3,word_count=3,linear_regression=3",
            "--plan-out",
            str(plan),
            *files,
        ]
    )
    assert code == 0 and json.loads(plan.read_text())["remaining_complete_rows"] == {}
    pooled = json.loads(out.read_text())
    assert (
        pooled["rows"]["kmeans"]["complete_sessions"] == 3
        and pooled["rows"]["linear_regression"]["complete_sessions"] == 3
    )


# --------------------------------------------------------------------------- #
# Round 159: installation equivalence.  Sessions of two installed packages pool
# only through a manifest that names both installations (wheel identities,
# every installed file's hash) and the reviewed compiler delta; the pooler
# re-hashes the live installations and the wheels at use.  Symlinked paths
# resolve to the same installation.
# --------------------------------------------------------------------------- #


def _install(root, name, *, module_text="CHECK = 'old'\n", build_tag="pkgA", drift=None):
    """A fake installed package: allo/__init__.py, two modules, a fake binding
    whose bytes embed a build path, and a wheel file.  Returns (allo_init, wheel)."""
    lib = root / name / "venv" / "lib" / "allo"
    (lib / "pim").mkdir(parents=True)
    (lib / "__init__.py").write_text("")
    (lib / "pim" / "__init__.py").write_text("PIM = 1\n")
    (lib / "spmw_autoschedule.py").write_text(module_text)
    (lib / "_binding.so").write_bytes(b"\x7fELF/build/" + build_tag.encode() + b"/src")
    if drift:
        drift(lib)
    wheel = root / name / "wheel" / "allo-0.5.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel-" + name.encode())
    return str(lib / "__init__.py"), str(wheel)


def _manifest(pool, out, a, b, allowed=(("spmw_autoschedule.py", "reviewed"), ("_binding.so", "build path")), subs=(("pkgA", "pkgB"),)):
    return pool.write_equivalence(str(out), [(a[0], a[1], "revA"), (b[0], b[1], "revB")], list(allowed), list(subs), "test decision")


def test_sessions_of_two_installations_pool_only_through_a_verified_manifest(pool, tmp_path):
    a = _install(tmp_path, "pkgA")
    b = _install(tmp_path, "pkgB", module_text="CHECK = 'new'\n", build_tag="pkgB")
    sa = make_session("sa", allo=a[0])
    sb = make_session("sb", allo=b[0], offset=20)
    files = []
    for i, s in enumerate((sa, sb), 1):
        f = tmp_path / f"s{i}.json"
        f.write_text(json.dumps(s))
        files.append(str(f))
    # without a manifest: refused as before
    pool.set_equivalence_manifest(None)
    rc = pool.main(["--out", str(tmp_path / "o.json"), "--row-quota", "kmeans=3,word_count=3,linear_regression=3", *files])
    assert rc == 2
    # with a manifest that lists both installations and the reviewed delta: pooled
    manifest = tmp_path / "eq.json"
    _manifest(pool, manifest, a, b)
    rc = pool.main(["--equivalence", str(manifest), "--out", str(tmp_path / "o2.json"), "--row-quota", "kmeans=3,word_count=3,linear_regression=3", *files])
    assert rc == 3  # two complete sessions per row, below the quota of three
    out = json.loads((tmp_path / "o2.json").read_text())
    assert out["rows"]["kmeans"]["complete_sessions"] == 2
    # preflight: a third installation identical to B by symlink resolves to B
    link = tmp_path / "linkB"
    link.symlink_to(tmp_path / "pkgB")
    linked = str(link / "venv" / "lib" / "allo" / "__init__.py")
    rc = pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", linked, *files])
    assert rc == 0
    # preflight of an installation the manifest does not list: refused
    c = _install(tmp_path, "pkgC", build_tag="pkgC")
    rc = pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", c[0], *files])
    assert rc == 2


def test_manifest_refusals(pool, tmp_path):
    a = _install(tmp_path, "pkgA")
    b = _install(tmp_path, "pkgB", module_text="CHECK = 'new'\n", build_tag="pkgB")
    manifest = tmp_path / "eq.json"
    _manifest(pool, manifest, a, b)
    loaded = pool.load_equivalence(str(manifest))
    assert pool.installations_equivalent(loaded, a[0], b[0]) is True
    # (1) a file outside the reviewed delta differs: refused
    pim_b = Path(b[0]).parent / "pim" / "__init__.py"
    original = pim_b.read_text()
    pim_b.write_text("PIM = 2\n")
    with pytest.raises(pool.SessionInvalid, match="differs from the manifest's recorded files"):
        pool.installations_equivalent(loaded, a[0], b[0])
    # the same drift recorded in a fresh manifest is still refused: outside the allowed set
    fresh = tmp_path / "eq2.json"
    _manifest(pool, fresh, a, b)
    with pytest.raises(pool.SessionInvalid, match="outside the reviewed delta"):
        pool.installations_equivalent(pool.load_equivalence(str(fresh)), a[0], b[0])
    pim_b.write_text(original)
    # (2) the allowed module changed after the manifest was written (an unreviewed delta): refused
    mod_b = Path(b[0]).parent / "spmw_autoschedule.py"
    mod_b.write_text("CHECK = 'newer'\n")
    with pytest.raises(pool.SessionInvalid, match="differs from the manifest's recorded files"):
        pool.installations_equivalent(loaded, a[0], b[0])
    mod_b.write_text("CHECK = 'new'\n")
    # (3) the binding differs beyond the recorded substitution: refused
    bind_b = Path(b[0]).parent / "_binding.so"
    bind_b.write_bytes(b"\x7fELF/build/pkgB/src-extra")
    fresh3 = tmp_path / "eq3.json"
    _manifest(pool, fresh3, a, b)
    with pytest.raises(pool.SessionInvalid, match="beyond the recorded substitutions"):
        pool.installations_equivalent(pool.load_equivalence(str(fresh3)), a[0], b[0])
    bind_b.write_bytes(b"\x7fELF/build/pkgB/src")
    # (4) the wheel no longer matches the manifest: refused
    Path(b[1]).write_bytes(b"another wheel")
    with pytest.raises(pool.SessionInvalid, match="does not match the manifest's sha256"):
        pool.installations_equivalent(loaded, a[0], b[0])
    Path(b[1]).write_bytes(b"wheel-pkgB")
    # (5) a manifest without a decision or with another schema is refused outright
    bad = json.loads(manifest.read_text())
    bad["decision"] = ""
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(pool.SessionInvalid, match="no reviewed decision"):
        pool.load_equivalence(str(tmp_path / "bad.json"))
    bad["schema"] = "other"
    (tmp_path / "bad2.json").write_text(json.dumps(bad))
    with pytest.raises(pool.SessionInvalid, match="unknown schema"):
        pool.load_equivalence(str(tmp_path / "bad2.json"))
    # (6) an installation the manifest does not list
    c = _install(tmp_path, "pkgC", build_tag="pkgC")
    with pytest.raises(pool.SessionInvalid, match="not listed in the equivalence manifest"):
        pool.installations_equivalent(loaded, a[0], c[0])
    # the untouched pair still verifies
    assert pool.installations_equivalent(loaded, a[0], b[0]) is True


# --------------------------------------------------------------------------- #
# Round 160: under a manifest every referenced installation is verified before
# any identity shortcut (the round 159 review's three cases); without a
# manifest the identity behaviour is unchanged.
# --------------------------------------------------------------------------- #


def _single_install_setup(pool, tmp_path):
    a = _install(tmp_path, "pkgA")
    b = _install(tmp_path, "pkgB", module_text="CHECK = 'new'\n", build_tag="pkgB")
    manifest = tmp_path / "eq.json"
    _manifest(pool, manifest, a, b)
    session = tmp_path / "s1.json"
    session.write_text(json.dumps(make_session("one", allo=a[0])))
    return a, b, manifest, session


def test_same_installation_drift_is_refused_under_a_manifest(pool, tmp_path):
    a, b, manifest, session = _single_install_setup(pool, tmp_path)
    quota = ["--row-quota", "kmeans=3,word_count=3,linear_regression=3"]
    # the untouched installation: preflight and the single-session pool pass under the manifest
    assert pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", a[0], str(session)]) == 0
    assert pool.main(["--equivalence", str(manifest), "--out", str(tmp_path / "o.json"), *quota, str(session)]) == 3
    # a compiler file of that same installation changes after the manifest was written
    (Path(a[0]).parent / "pim" / "__init__.py").write_text("PIM = 99\n")
    # (1) the same-installation preflight is refused
    assert pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", a[0], str(session)]) == 2
    # (2) the single-session pool is refused
    assert pool.main(["--equivalence", str(manifest), "--out", str(tmp_path / "o2.json"), *quota, str(session)]) == 2
    # without a manifest the identity behaviour is unchanged: the same path pools and preflights as before
    assert pool.main(["--out", "/dev/null", "--preflight-install", a[0], str(session)]) == 0
    assert pool.main(["--out", str(tmp_path / "o3.json"), *quota, str(session)]) == 3


def test_unlisted_same_installation_is_refused_under_a_manifest(pool, tmp_path):
    a, b, manifest, _session = _single_install_setup(pool, tmp_path)
    c = _install(tmp_path, "pkgC", build_tag="pkgC")
    session_c = tmp_path / "sc.json"
    session_c.write_text(json.dumps(make_session("c", allo=c[0])))
    # (3) the preflight of an installation the manifest does not list, against its own session
    assert pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", c[0], str(session_c)]) == 2
    # and the single-session pool of that unlisted installation under the manifest
    assert pool.main(["--equivalence", str(manifest), "--out", str(tmp_path / "o.json"), "--row-quota", "kmeans=3,word_count=3,linear_regression=3", str(session_c)]) == 2
    # the symlink alias of a listed installation is verified as that installation
    link = tmp_path / "aliasA"
    link.symlink_to(tmp_path / "pkgA")
    alias = str(link / "venv" / "lib" / "allo" / "__init__.py")
    assert pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", alias, str(_session)]) == 0
    (Path(a[0]).parent / "spmw_autoschedule.py").write_text("CHECK = 'tampered'\n")
    assert pool.main(["--equivalence", str(manifest), "--out", "/dev/null", "--preflight-install", alias, str(_session)]) == 2


def test_verification_cache_is_scoped_to_the_manifest_invocation(pool, tmp_path):
    a, b, manifest, session = _single_install_setup(pool, tmp_path)
    loaded = pool.load_equivalence(str(manifest))
    pool.set_equivalence_manifest(loaded)
    assert pool.verify_installation_under_manifest(a[0], "first") is not None
    assert Path(a[0]).parent.resolve().as_posix() in {Path(k).parent.resolve().as_posix() for k in pool._EQUIVALENCE["verified_installations"]}
    pool.set_equivalence_manifest(loaded)  # a new invocation starts empty
    assert pool._EQUIVALENCE["verified_installations"] == {} and pool._EQUIVALENCE["verified_pairs"] == set()
    pool.set_equivalence_manifest(None)
    assert pool.verify_installation_under_manifest(a[0], "none") is None
