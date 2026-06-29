# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: results.json + RESULTS.md + COVERAGE.tsv writers.

The schema is spec Answer 4. A per-kernel `results.json` carries the verdict +
the validated-reference provenance + the run-stamped performance provenance
(sim/device source, run_cmd, timestamp, tenon_commit). `RESULTS.md` is a
human-readable mirror. `COVERAGE.tsv` is the top-level aggregate, one row per
(kernel, target), generated from the per-folder results.json -- never
hand-edited.

These writers are suite infrastructure (written once); a kernel test calls
`write_results(...)` and `append_coverage(...)`, declaring no schema itself.
"""

from __future__ import annotations

import json
import pathlib

# tests/pim/lib/results.py -> parents[1] == tests/pim
_PIM_ROOT = pathlib.Path(__file__).resolve().parents[1]
_COVERAGE = _PIM_ROOT / "COVERAGE.tsv"

_COVERAGE_COLS = [
    "kernel", "target", "dataset", "correctness_status", "correctness_detail",
    "metric", "value", "source", "tenon_commit", "timestamp",
]


def build_record(
    *, kernel, target, dataset, shapes, verdict, reference_provenance,
    metric, value, source, run_cmd, timestamp, tenon_commit, notes="",
) -> dict:
    """Assemble the results.json record (spec Answer 4 schema). `verdict` is a
    `reference.Verdict`; `value` is the measured number (cycles/ns) or None."""
    return {
        "kernel": kernel,
        "target": target,
        "dataset": dataset,
        "shapes": dict(shapes),
        "correctness": {
            "status": verdict.status,
            "detail": verdict.detail,
            "reference": reference_provenance,
        },
        "performance": {
            "metric": metric,
            "value": value,
            "source": source,
            "run_cmd": run_cmd,
            "timestamp": timestamp,
        },
        "tenon_commit": tenon_commit,
        "notes": notes,
    }


def write_results(folder, record: dict) -> pathlib.Path:
    """Write `record` to `<folder>/results.json` (pretty-printed)."""
    path = pathlib.Path(folder) / "results.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def write_results_md(folder, record: dict) -> pathlib.Path:
    """Write a human-readable `<folder>/RESULTS.md` mirroring results.json."""
    c = record["correctness"]
    p = record["performance"]
    val = p["value"]
    lines = [
        f"# {record['kernel']} on {record['target']} ({record['dataset']})",
        "",
        f"- correctness: **{c['status']}** -- {c['detail']}",
        f"- reference: {c['reference']['source']} {c['reference']['size_class']} "
        f"(validated by `{c['reference']['validated_by']}`)",
        f"- shapes: {record['shapes']}",
        f"- {p['metric']}: {val if val is not None else 'N/A'}",
        f"- source: `{p['source']}`",
        f"- run_cmd: `{p['run_cmd']}`",
        f"- timestamp: {p['timestamp']}",
        f"- tenon_commit: `{record['tenon_commit']}`",
    ]
    if record.get("notes"):
        lines += ["", record["notes"]]
    path = pathlib.Path(folder) / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _tsv_safe(value) -> str:
    """One TSV cell: collapse newlines/tabs to spaces so a multi-line detail
    (e.g. a sim crash message) stays on ONE row."""
    return " ".join(str(value).split())


def regenerate_coverage() -> pathlib.Path:
    """Regenerate COVERAGE.tsv from every results.json under tests/pim/ (one
    row per (kernel, target)). Generated, never hand-edited (spec Answer 4)."""
    rows = []
    for rj in sorted(_PIM_ROOT.rglob("results.json")):
        rec = json.loads(rj.read_text())
        c, p = rec["correctness"], rec["performance"]
        rows.append([
            rec["kernel"], rec["target"], rec["dataset"],
            c["status"], c["detail"],
            p["metric"], "" if p["value"] is None else str(p["value"]),
            p["source"], rec["tenon_commit"], p["timestamp"],
        ])
    lines = ["\t".join(_COVERAGE_COLS)]
    lines += ["\t".join(_tsv_safe(cell) for cell in r) for r in rows]
    _COVERAGE.write_text("\n".join(lines) + "\n")
    return _COVERAGE
