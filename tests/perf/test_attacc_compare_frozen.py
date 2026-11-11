# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative and positive checks of the release comparator
`docs/release/scripts/attacc_compare_frozen.py` (round 115): incomplete,
relabelled, duplicated or wrong-row evidence is refused by name; a regression
against the archived per-shape timing fails; an improvement passes."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "release" / "scripts" / "attacc_compare_frozen.py"


@pytest.fixture(scope="module")
def comparator():
    spec = importlib.util.spec_from_file_location("attacc_compare_frozen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# a small archived manifest: the four GEMV rows and two attention shapes per head count
MANIFEST = [
    ("qkv_gemv_nh8", "GEMV", "8", "5120", "384", "0.0088", "0.0280", "0.0088"),
    ("qkv_gemv_nh32", "GEMV", "32", "5120", "384", "0.0336", "0.1720", "0.0336"),
    ("proj_gemv_nh8", "GEMV", "8", "128", "5120", "0.0026", "0.0026", "0.0026"),
    ("proj_gemv_nh32", "GEMV", "32", "128", "5120", "0.0102", "0.0103", "0.0102"),
    ("score_nh8_a", "attention-QK^T", "8", "128", "144", "0.0010", "0.0010", "0.0010"),
    ("score_nh8_b", "attention-QK^T", "8", "128", "160", "0.0011", "0.0011", "0.0011"),
    (
        "score_nh32_a",
        "attention-QK^T",
        "32",
        "128",
        "144",
        "0.0040",
        "0.0046",
        "0.0040",
    ),
    (
        "score_nh32_b",
        "attention-QK^T",
        "32",
        "128",
        "160",
        "0.0042",
        "0.0047",
        "0.0042",
    ),
]
PAPER = {
    "attacc_kernel_qkv_b1": "0.00880",
    "attacc_kernel_qkv_b4": "0.03360",
    "attacc_kernel_proj_b1": "0.00260",
    "attacc_kernel_proj_b4": "0.01020",
    "attacc_kernel_score_b1": "Tenon 1.00x over fixed (figure)",
    "attacc_kernel_score_b4": "Tenon 1.13x over fixed (figure)",
    "attacc_e2e_in128_out2048_bs1": "base 1.47x, full 6.5x",
    "attacc_e2e_in2048_out128_bs1": "base 1.56x, full 6.95x",
    "attacc_e2e_in2048_out2048_bs1": "base 1.66x, full 7.4x",
    "attacc_e2e_in128_out2048_bs4": "base 1.34x, full 3.09x",
    "attacc_e2e_in2048_out128_bs4": "base 1.5x, full 3.48x",
    "attacc_e2e_in2048_out2048_bs4": "base 1.65x, full 3.87x",
}
GEMV_SHAPES = {
    "qkv_b1": (8, 5120, 384),
    "qkv_b4": (32, 5120, 384),
    "proj_b1": (8, 128, 5120),
    "proj_b4": (32, 128, 5120),
}
E2E_KEYS = [
    (128, 2048, 1),
    (2048, 128, 1),
    (2048, 2048, 1),
    (128, 2048, 4),
    (2048, 128, 4),
    (2048, 2048, 4),
]
E2E_SPEEDUPS = {
    (128, 2048, 1): (1.47, 6.50),
    (2048, 128, 1): (1.56, 6.95),
    (2048, 2048, 1): (1.66, 7.40),
    (128, 2048, 4): (1.34, 3.09),
    (2048, 128, 4): (1.50, 3.48),
    (2048, 2048, 4): (1.65, 3.87),
}


def run_record(shape, ms, selection="shared-pipeline", trace=True):
    record = {
        "kind": "simulator-timing-only",
        "shape": list(shape),
        "selection": selection,
        "milliseconds": ms,
        "tiling": [[1, 1, 1]],
    }
    if trace:
        record["trace_check"] = {"match": True, "totals_match": True}
    return record


def make_sessions(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    sim = {"name": "Ramulator2-AttAcc (test)", "binary_sha256": "s" * 64}
    for row, shape in GEMV_SHAPES.items():
        key = tuple(str(x) for x in shape)
        archived = next(m for m in MANIFEST if (m[2], m[3], m[4]) == key)
        runs = [
            run_record(shape, float(archived[7])),
            run_record(
                shape, float(archived[6]), selection="fixed-form-baseline", trace=False
            ),
        ]
        json.dump(
            {"simulator": sim, "failures": [], "runs": runs},
            open(d / f"attacc_{row}.json", "w"),
        )
    for row, nhead in (("score_b1", "8"), ("score_b4", "32")):
        runs = [
            run_record((int(m[2]), int(m[3]), int(m[4])), float(m[7]))
            for m in MANIFEST
            if m[1] != "GEMV" and m[2] == nhead
        ]
        json.dump(
            {"simulator": sim, "failures": [], "runs": runs},
            open(d / f"attacc_{row}.json", "w"),
        )
    configs = [
        {
            "lin": k[0],
            "lout": k[1],
            "batch": k[2],
            "variants": {
                "tenon": {
                    "base_speedup": E2E_SPEEDUPS[k][0],
                    "full_speedup": E2E_SPEEDUPS[k][1],
                }
            },
        }
        for k in E2E_KEYS
    ]
    json.dump(
        {"kind": "analytical-composition", "configs": configs},
        open(d / "attacc_e2e.json", "w"),
    )
    return d


@pytest.fixture
def inputs(tmp_path):
    manifest = tmp_path / "tenon_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "id",
                "family",
                "nhead",
                "input_dim",
                "output_dim",
                "dcc_our_time_ms",
                "fixed_baseline_ms",
                "tenon_time_ms",
            ]
        )
        w.writerows(MANIFEST)
    inventory = tmp_path / "inventory.csv"
    with open(inventory, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "paper_value"])
        w.writeheader()
        for case, value in PAPER.items():
            w.writerow({"case_id": case, "paper_value": value})
    return make_sessions(tmp_path), manifest, inventory


def run(comparator, inputs, tmp_path):
    sessions, manifest, inventory = inputs
    out = tmp_path / "report.json"
    code = comparator.main(
        [
            "--sessions-dir",
            str(sessions),
            "--inventory",
            str(inventory),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ]
    )
    return code, json.loads(out.read_text())


def mutate(sessions, name, fn):
    path = sessions / f"attacc_{name}.json"
    data = json.load(open(path))
    fn(data)
    json.dump(data, open(path, "w"))


def test_complete_consistent_evidence_reproduces(comparator, inputs, tmp_path):
    code, report = run(comparator, inputs, tmp_path)
    assert code == 0 and report["all_reproduced"] and len(report["verdicts"]) == 12
    assert report["problems"] == []
    assert (
        report["diagnostics"]["tally_vs_dcc"]["matches_text"] is False
    )  # 8 kernels, a diagnostic only


def test_empty_decode_configurations_fail(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]
    mutate(sessions, "e2e", lambda e: e.update(configs=[]))
    code, report = run(comparator, inputs, tmp_path)
    assert code == 1 and not report["all_reproduced"] and report["decode"] == {}
    err = capsys.readouterr().err
    assert (
        "configuration (128, 2048, 1) missing" in err
        and "6 of 12 result verdicts missing" in err
    )


def test_duplicate_or_wrong_kind_decode_fails(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]
    mutate(
        sessions,
        "e2e",
        lambda e: e.update(
            kind="virtual", configs=e["configs"][:5] + [e["configs"][0]]
        ),
    )
    code, report = run(comparator, inputs, tmp_path)
    assert code == 1
    err = capsys.readouterr().err
    assert (
        "duplicate configuration (128, 2048, 1)" in err
        and "analytical-composition required" in err
        and "missing" in err
    )


def test_relabelled_score_records_without_traces_fail(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]

    def relabel(d):
        for r in d["runs"]:
            r["selection"] = "forced-tiling"
            r.pop("trace_check", None)

    mutate(sessions, "score_b1", relabel)
    code, report = run(comparator, inputs, tmp_path)
    assert code == 1 and report["score_rows"]["score_b1"]["shapes_present"] == 0
    err = capsys.readouterr().err
    assert (
        "record of selection 'forced-tiling'" in err
        and "2 of 2 archived shapes missing" in err
    )


def test_missing_wrong_row_and_duplicate_score_shapes_fail(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]

    def damage(d):
        d["runs"] = [
            d["runs"][0],
            copy.deepcopy(d["runs"][0]),
            run_record((32, 128, 144), 0.004),
        ]

    mutate(sessions, "score_b1", damage)
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert code == 1
    assert "duplicate record for shape [8, 128, 144]" in err
    assert (
        "shape [32, 128, 144] is not one of the row's 2 archived shapes (nhead 8)"
        in err
    )
    assert "1 of 2 archived shapes missing" in err


def test_score_trace_check_required(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]
    mutate(sessions, "score_b4", lambda d: d["runs"][0].pop("trace_check"))
    code, _ = run(comparator, inputs, tmp_path)
    assert (
        code == 1
        and "trace check missing or not matched for shape [32, 128, 144]"
        in capsys.readouterr().err
    )


def test_gemv_regression_fails_and_improvement_passes(comparator, inputs, tmp_path):
    sessions = inputs[0]

    def slower(d):
        for r in d["runs"]:
            if r["selection"] == "shared-pipeline":
                r["milliseconds"] = 0.03368  # archived 0.0336

    mutate(sessions, "qkv_b4", slower)
    code, report = run(comparator, inputs, tmp_path)
    assert (
        code == 1 and report["kernel_rows"]["qkv_b4"]["verdict"] == "DIFFERS (slower)"
    )
    assert report["problems"] == []  # a performance regression, not an evidence problem

    def faster(d):
        for r in d["runs"]:
            if r["selection"] == "shared-pipeline":
                r["milliseconds"] = 0.0330

    mutate(sessions, "qkv_b4", faster)
    code, report = run(comparator, inputs, tmp_path)
    assert code == 0 and report["kernel_rows"]["qkv_b4"]["verdict"] == "REPRODUCED"


def test_score_shape_slower_than_archived_fails(comparator, inputs, tmp_path):
    sessions = inputs[0]
    mutate(sessions, "score_b4", lambda d: d["runs"][0].update(milliseconds=0.0041))
    code, report = run(comparator, inputs, tmp_path)
    assert (
        code == 1
        and "slower than the archived" in report["score_rows"]["score_b4"]["verdict"]
    )


def test_gemv_evidence_shape_and_selection_requirements(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]
    mutate(
        sessions,
        "proj_b1",
        lambda d: d["runs"].__setitem__(0, run_record((32, 128, 5120), 0.0026)),
    )
    mutate(sessions, "proj_b4", lambda d: d["runs"].pop(1))
    mutate(sessions, "qkv_b1", lambda d: d["runs"][0].pop("trace_check"))
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert code == 1
    assert (
        "proj_b1: selected run shape [32, 128, 5120] is not the row's shape ('8', '128', '5120')"
        in err
    )
    assert "proj_b4: no fixed-form-baseline run" in err
    assert "qkv_b1: trace check missing or not matched" in err


def test_missing_session_and_recorded_failures_fail(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]
    (sessions / "attacc_qkv_b4.json").unlink()
    mutate(sessions, "proj_b1", lambda d: d.update(failures=["simulator exited 1"]))
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert (
        code == 1
        and "qkv_b4: session missing" in err
        and "proj_b1: session records failures" in err
    )


def test_half_ulp_comes_from_the_printed_digits(comparator):
    assert comparator.half_ulp("0.00881") == pytest.approx(5e-6)
    assert comparator.half_ulp("6.5") == pytest.approx(0.05)
    assert comparator.half_ulp("1.47") == pytest.approx(0.005)
    assert comparator.first_number("base 1.5x, full 3.48x", after="full") == "3.48"
