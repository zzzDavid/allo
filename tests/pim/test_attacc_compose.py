# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode composition's input discipline (``compose_e2e.py``): only
shared-pipeline records feed the tables, conflicting measurements of one
shape are rejected, references never replace a selected timing, and rows
without a measurement become NaN (never a substituted baseline)."""

import csv
import json
from pathlib import Path

import pytest

from benchmarks.attacc.scripts import compose_e2e as compose


def _session(path, runs):
    path.write_text(json.dumps({"compiler": {"git_revision": "x"}, "simulator": {"binary_sha256": "y", "tck_ns": 0.769}, "runs": runs, "failures": []}))
    return str(path)


def _run(family, shape, cycles, selection="shared-pipeline"):
    return {"family": family, "shape": list(shape), "memory_system_cycles": cycles, "selection": selection}


def test_only_shared_pipeline_records_are_used(tmp_path):
    a = _session(tmp_path / "a.json", [_run("gemv", (8, 5120, 384), 11454), _run("gemv", (8, 5120, 384), 36504, "fixed-form-baseline"), _run("gemv", (8, 5120, 384), 99999, "forced-tiling")])
    times, provenance = compose.selected_times([a])
    assert times == {("gemv", 8, 5120, 384): 11454}
    assert provenance["ignored_reference_records"] == {"fixed-form-baseline": 1, "forced-tiling": 1}


def test_duplicate_measurements_must_agree(tmp_path):
    a = _session(tmp_path / "a.json", [_run("attention", (8, 128, 480), 953)])
    b = _session(tmp_path / "b.json", [_run("attention", (8, 128, 480), 953)])
    c = _session(tmp_path / "c.json", [_run("attention", (8, 128, 480), 954)])
    times, _ = compose.selected_times([a, b])
    assert times == {("attention", 8, 128, 480): 953}
    with pytest.raises(ValueError, match="conflicting measurements"):
        compose.selected_times([a, c])


def test_tables_mark_unmeasured_rows_nan_and_list_missing_required(tmp_path):
    root = tmp_path / "dcc"
    (root / "temp" / "simulation").mkdir(parents=True)
    gemv_rows = [{"batch_size": "1", "#head": "8", "input_dim": "5120", "output_dim": "384", "our_time(ms)": "0.0088", "baseline(ms)": "0.028", "Speedup": "3.19"},
                 {"batch_size": "1", "#head": "11", "input_dim": "5120", "output_dim": "384", "our_time(ms)": "0.01", "baseline(ms)": "0.03", "Speedup": "3"}]
    attn_rows = [{"batch_size": "1", "#head": "8", "input_dim": "128", "output_dim": str(L), "our_time(ms)": "0.001", "baseline(ms)": "0.001", "Speedup": "1"} for L in (128, 144, 160)]
    for name, rows in (("@attacc-gemv_inference.csv", gemv_rows), ("@attacc-attn_inference.csv", attn_rows)):
        with (root / "temp" / "simulation" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    out = tmp_path / "tables"
    out.mkdir()
    times = {("gemv", 8, 5120, 384): 11454, ("attention", 8, 128, 144): 900}
    report = compose.write_tables(root, times, out)
    assert report["measured_rows"] == 2 and report["nan_rows"] == 3
    written = {(r["#head"], r["input_dim"], r["output_dim"]): r[compose.LABEL] for r in csv.DictReader((out / "@attacc-gemv_inference.csv").open())}
    assert written[("8", "5120", "384")] == repr(11454 * compose.TCK_NS / 1e6) and written[("11", "5120", "384")] == "nan"
    attn = {r["output_dim"]: r[compose.LABEL] for r in csv.DictReader((out / "@attacc-attn_inference.csv").open())}
    assert attn["144"] != "nan" and attn["128"] == "nan" and attn["160"] == "nan"
    # every GPT-13B row the model reads that has no measurement is named
    assert "('attention', 8, 128, 160)" in report["missing_required"] and "('gemv', 32, 5120, 384)" in report["missing_required"]
    assert len(report["missing_required"]) == len(compose.required_rows()) - 2


def test_required_rows_are_the_paper_kernels():
    rows = compose.required_rows()
    assert len(rows) == 4 + 2 * 248
    assert ("gemv", 8, 5120, 384) in rows and ("gemv", 32, 128, 5120) in rows
    assert ("attention", 32, 128, 4096) in rows and ("attention", 8, 128, 128) not in rows
