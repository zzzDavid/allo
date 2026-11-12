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
