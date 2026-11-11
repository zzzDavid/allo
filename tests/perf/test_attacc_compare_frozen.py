# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative and positive checks of the release comparator
`docs/release/scripts/attacc_compare_frozen.py` (round 115): incomplete,
relabelled, duplicated or wrong-row evidence is refused by name; a regression
against the archived per-shape timing fails; an improvement passes."""

from __future__ import annotations

import copy
import csv
import hashlib
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


# a small archived manifest: the four GEMV rows and two attention shapes per head
# count, with every timing an exact multiple of the simulator clock so that the
# cycle / milliseconds agreement the comparator enforces holds for the fixture
TCK_NS = 0.769


def ms_of(cycles: int) -> str:
    return repr(cycles * TCK_NS / 1e6)


def _row(ident, family, nhead, dim, out, dcc_cycles, fixed_cycles, tenon_cycles):
    return (
        ident,
        family,
        nhead,
        dim,
        out,
        ms_of(dcc_cycles),
        ms_of(fixed_cycles),
        ms_of(tenon_cycles),
    )


MANIFEST = [
    _row("qkv_gemv_nh8", "GEMV", "8", "5120", "384", 11444, 36410, 11444),
    _row("qkv_gemv_nh32", "GEMV", "32", "5120", "384", 43693, 223667, 43693),
    _row("proj_gemv_nh8", "GEMV", "8", "128", "5120", 3381, 3381, 3381),
    _row("proj_gemv_nh32", "GEMV", "32", "128", "5120", 13264, 13394, 13264),
    _row("score_nh8_a", "attention-QK^T", "8", "128", "144", 1300, 1300, 1300),
    _row("score_nh8_b", "attention-QK^T", "8", "128", "160", 1430, 1430, 1430),
    _row("score_nh32_a", "attention-QK^T", "32", "128", "144", 5202, 5982, 5202),
    _row("score_nh32_b", "attention-QK^T", "32", "128", "160", 5462, 6112, 5462),
]
PAPER = {
    "attacc_kernel_qkv_b1": "0.00880",
    "attacc_kernel_qkv_b4": "0.03360",
    "attacc_kernel_proj_b1": "0.00260",
    "attacc_kernel_proj_b4": "0.01020",
    "attacc_kernel_score_b1": "Tenon 1.00x over fixed (figure)",
    "attacc_kernel_score_b4": "Tenon 1.13x over fixed (figure)",
    "attacc_e2e_in128_out2048_bs1": "base 1.47x, full 6.50x",
    "attacc_e2e_in2048_out128_bs1": "base 1.56x, full 6.95x",
    "attacc_e2e_in2048_out2048_bs1": "base 1.66x, full 7.40x",
    "attacc_e2e_in128_out2048_bs4": "base 1.34x, full 3.09x",
    "attacc_e2e_in2048_out128_bs4": "base 1.50x, full 3.48x",
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


def run_record(
    shape,
    ms,
    selection="shared-pipeline",
    trace=True,
    family="attention",
    artifacts=None,
):
    cycles = round(ms * 1e6 / TCK_NS)
    record = {
        "kind": "simulator-timing-only",
        "family": family,
        "shape": list(shape),
        "selection": selection,
        "milliseconds": cycles * TCK_NS / 1e6,
        "memory_system_cycles": cycles,
        "tiling": [[1, 1, 1]],
    }
    if trace:
        record["trace_check"] = {"match": True, "totals_match": True}
    if artifacts is not None:
        record["artifacts"] = str(artifacts)
        record["trace_sha256"] = retain_raw_output(artifacts, cycles, shape)
    return record


def set_timing(record, ms):
    """Change a record's timing consistently: cycles, milliseconds and, for a
    selected run, the retained simulator output's cycle line."""
    cycles = round(ms * 1e6 / TCK_NS)
    record["memory_system_cycles"] = cycles
    record["milliseconds"] = cycles * TCK_NS / 1e6
    if record.get("artifacts"):
        stdout = Path(record["artifacts"]) / "stdout.txt"
        stdout.write_text(
            f"MemorySystem:\n  impl: PIMDRAM\n  memory_system_cycles: {cycles}\n"
        )


def retain_raw_output(directory, cycles, shape):
    """A retained raw-output directory as run_ramulator keeps it: the
    simulator's stdout with its cycle line and the trace; returns the trace sha."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stdout.txt").write_text(
        f"MemorySystem:\n  impl: PIMDRAM\n  memory_system_cycles: {cycles}\n"
    )
    trace = directory / "kernel.trace"
    trace.write_text("PIM_WR_GB 0\nPIM_MAC_AB 1\n" * (1 + sum(shape) % 5))
    return hashlib.sha256(trace.read_bytes()).hexdigest()


def make_sessions(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    sim = {
        "name": "Ramulator2-AttAcc (test)",
        "binary_sha256": "s" * 64,
        "tck_ns": TCK_NS,
    }
    for row, shape in GEMV_SHAPES.items():
        key = tuple(str(x) for x in shape)
        archived = next(m for m in MANIFEST if (m[2], m[3], m[4]) == key)
        runs = [
            run_record(
                shape,
                float(archived[7]),
                family="gemv",
                artifacts=d / f"{row}.artifacts" / "selected",
            ),
            run_record(
                shape,
                float(archived[6]),
                selection="fixed-form-baseline",
                trace=False,
                family="gemv",
            ),
        ]
        json.dump(
            {"simulator": sim, "failures": [], "runs": runs},
            open(d / f"attacc_{row}.json", "w"),
        )
    for row, nhead in (("score_b1", "8"), ("score_b4", "32")):
        runs = [
            run_record(
                (int(m[2]), int(m[3]), int(m[4])),
                float(m[7]),
                artifacts=d
                / f"{row}.artifacts"
                / f"attention_{m[2]}x{m[3]}x{m[4]}"
                / "selected",
            )
            for m in MANIFEST
            if m[1] != "GEMV" and m[2] == nhead
        ]
        json.dump(
            {"simulator": sim, "failures": [], "runs": runs},
            open(d / f"attacc_{row}.json", "w"),
        )
    configs = []
    for k in E2E_KEYS:
        base, full = E2E_SPEEDUPS[k]
        roofline = 10.0
        configs.append(
            {
                "lin": k[0],
                "lout": k[1],
                "batch": k[2],
                "gpu_roofline_ms": roofline,
                "variants": {
                    "tenon": {
                        "base_ms": roofline / base,
                        "full_ms": roofline / full,
                        "base_speedup": base,
                        "full_speedup": full,
                    }
                },
            }
        )
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
        and "score_b4: shape [32, 128, 144]: trace check missing or not matched"
        in capsys.readouterr().err
    )


def test_gemv_regression_fails_and_improvement_passes(comparator, inputs, tmp_path):
    sessions = inputs[0]

    def slower(d):
        for r in d["runs"]:
            if r["selection"] == "shared-pipeline":
                set_timing(r, 0.03368)  # archived 0.0336

    mutate(sessions, "qkv_b4", slower)
    code, report = run(comparator, inputs, tmp_path)
    assert (
        code == 1 and report["kernel_rows"]["qkv_b4"]["verdict"] == "DIFFERS (slower)"
    )
    assert report["problems"] == []  # a performance regression, not an evidence problem

    def faster(d):
        for r in d["runs"]:
            if r["selection"] == "shared-pipeline":
                set_timing(r, 0.0330)

    mutate(sessions, "qkv_b4", faster)
    code, report = run(comparator, inputs, tmp_path)
    assert code == 0 and report["kernel_rows"]["qkv_b4"]["verdict"] == "REPRODUCED"


def test_score_shape_slower_than_archived_fails(comparator, inputs, tmp_path):
    sessions = inputs[0]
    mutate(sessions, "score_b4", lambda d: set_timing(d["runs"][0], 0.0041))
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
    assert "qkv_b1: shape [8, 5120, 384]: trace check missing or not matched" in err


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


def _set_decode(cfg_index, **fields):
    def mutate_e2e(e):
        tenon = e["configs"][cfg_index]["variants"]["tenon"]
        tenon.update(fields)

    return mutate_e2e


def test_decode_speedup_below_printed_precision_fails_consistently(
    comparator, inputs, tmp_path
):
    """The frozen table prints 6.50 and 1.50; a consistent 6.46 or 1.46 (latency and
    speedup agreeing) is below the printed value less half an ulp and fails."""
    sessions = inputs[0]
    mutate(sessions, "e2e", _set_decode(0, full_ms=10.0 / 6.46, full_speedup=6.46))
    code, report = run(comparator, inputs, tmp_path)
    assert code == 1 and report["problems"] == []
    assert report["decode"]["attacc_e2e_in128_out2048_bs1"][
        "full_bound"
    ] == pytest.approx(6.495)
    assert report["decode"]["attacc_e2e_in128_out2048_bs1"]["verdict"].startswith(
        "DIFFERS"
    )
    mutate(sessions, "e2e", _set_decode(0, full_ms=10.0 / 6.50, full_speedup=6.50))
    mutate(sessions, "e2e", _set_decode(4, base_ms=10.0 / 1.46, base_speedup=1.46))
    code, report = run(comparator, inputs, tmp_path)
    assert code == 1
    assert report["decode"]["attacc_e2e_in2048_out128_bs4"][
        "base_bound"
    ] == pytest.approx(1.495)
    mutate(sessions, "e2e", _set_decode(4, base_ms=10.0 / 1.495, base_speedup=1.495))
    code, report = run(comparator, inputs, tmp_path)
    assert code == 0  # at the printed value less half an ulp


def test_decode_fields_must_agree_and_be_finite(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]
    mutate(
        sessions, "e2e", _set_decode(0, full_ms=10.0 / 6.50 * 10)
    )  # latency tenfold, speedup unchanged
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert (
        code == 1
        and "recorded speedups" in err
        and "do not equal gpu_roofline_ms over the recorded latencies" in err
    )
    mutate(
        sessions, "e2e", _set_decode(0, full_ms=10.0 / 6.50, full_speedup=float("inf"))
    )
    code, _ = run(comparator, inputs, tmp_path)
    assert (
        code == 1 and "not positive finite: ['full_speedup']" in capsys.readouterr().err
    )


def test_kernel_cycles_clock_and_raw_output_must_agree(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]
    mutate(
        sessions,
        "qkv_b4",
        lambda d: d["runs"][0].update(
            memory_system_cycles=d["runs"][0]["memory_system_cycles"] * 10
        ),
    )
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert code == 1
    assert (
        "do not equal memory_system_cycles" in err
        and "does not report memory_system_cycles" in err
    )


def test_retained_raw_output_is_required_and_checked(
    comparator, inputs, tmp_path, capsys
):
    sessions = inputs[0]
    mutate(sessions, "proj_b1", lambda d: d["runs"][0].pop("artifacts"))
    (sessions / "qkv_b1.artifacts" / "selected" / "kernel.trace").write_text(
        "tampered\n"
    )
    code, _ = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert code == 1
    assert "proj_b1: shape [8, 128, 5120]: retained raw output directory missing" in err
    assert "qkv_b1: shape [8, 5120, 384]: retained trace sha256" in err


def test_baselines_must_be_the_rows_workload(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]

    def wrong_shape(d):
        for r in d["runs"]:
            if r["selection"] == "fixed-form-baseline":
                r["shape"] = [1, 1, 1]

    mutate(sessions, "qkv_b4", wrong_shape)
    code, report = run(comparator, inputs, tmp_path)
    err = capsys.readouterr().err
    assert code == 1 and "qkv_b4" not in report["kernel_rows"]
    assert (
        "fixed-form baseline shape [1, 1, 1] is not the row's shape ('32', '5120', '384'); record rejected"
        in err
    )
    assert "qkv_b4: no fixed-form baseline of the row's shape" in err
    mutate(sessions, "proj_b1", lambda d: d["runs"][1].update(family="attention"))
    code, _ = run(comparator, inputs, tmp_path)
    assert (
        code == 1 and "family 'attention', 'gemv' required" in capsys.readouterr().err
    )


def test_missing_simulator_clock_fails(comparator, inputs, tmp_path, capsys):
    sessions = inputs[0]
    mutate(sessions, "score_b1", lambda d: d["simulator"].pop("tck_ns"))
    code, _ = run(comparator, inputs, tmp_path)
    assert code == 1 and "no positive finite simulator clock" in capsys.readouterr().err


def test_inventory_carries_the_tables_printed_precision():
    """The frozen decode table prints two decimals for every speedup; the
    inventory's paper values must carry them (round 116: 6.5, 7.4 and 1.5 had
    been stored, widening the half-ulp bound tenfold)."""
    inventory = REPO / "docs" / "release" / "paper_reproduction" / "inventory.csv"
    rows = {r["case_id"]: r for r in csv.DictReader(open(inventory))}
    import re

    for case in PAPER:
        if case.startswith("attacc_e2e_"):
            for number in re.findall(r"\d+\.\d+", rows[case]["paper_value"]):
                assert len(number.split(".")[1]) == 2, (case, number)
        elif case.startswith("attacc_kernel_") and "score" not in case:
            assert len(rows[case]["paper_value"].split(".")[1]) == 5, case
