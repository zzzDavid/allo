# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPT-13B decode latency from Ramulator2-AttAcc kernel times (DCC composition).

EVALUATION ONLY, external dependencies, not part of the installed package.
The result is an analytical composition (``kind: "analytical-composition"``):
per-kernel PIM times measured by Ramulator2-AttAcc (``run_ramulator.py``
session files) are combined with DCC's analytical A100 roofline and decode
pipeline model (DCC ``inference/system.py``), as in the paper ("GPT-13B
decode latency combines Ramulator2-AttAcc PIM timings with DCC's analytical
GPU roofline, excluding prefill"; FFN parallelism and head-wise pipelining
are modelled).  Nothing here is a measurement of a decode step.

DCC (github.com/SPIN-Research-Group/DCC, study commit
``212f86ea7f08da059836e3a1311d3a11afae8dbe``) has no license; its
``inference`` package is imported at run time from ``ATTACC_DCC_ROOT``.

Kernel times.  Only ``run_ramulator.py`` records whose ``selection`` is
``shared-pipeline`` are used; forced tilings and fixed-form baselines are
never substituted for a selected timing.  Two records of the same kernel
shape must agree on their cycles (a conflict is an error; an exact
duplicate is accepted once).  Every GPT-13B PIM kernel row DCC's model reads
(QKV and projection at 8 and 32 head instances, attention scoring at the
16-aligned context lengths) must have a measured Tenon time: unmeasured
rows receive NaN, so a composition that needs one fails instead of silently
using another number; ``--missing=baseline`` no longer exists.

Variants.  For each of the paper's six configurations (prompt 128 / 2048,
generation 2048 / 128 / 2048, batch 1 and 4) the script composes the GPU
roofline alone and three PIM variants, each as ``base`` (no pipelining, no
FFN parallelism) and ``full`` (both): Tenon's times and DCC's recorded
selection (``our_time(ms)``) with QKV, projection, attention and softmax
on PIM (the paper's "same variant scheduled by DCC and by Tenon"), and the
fixed tiling (``baseline(ms)``) with only attention and softmax on PIM,
which is how DCC's ``run_inference.py`` composes its AttAcc baseline
(``modelinfos["DCC"]`` false).  It reports the speedups over the GPU
roofline with the per-stage times DCC's model exposes (QKV, projection,
attention matmul, softmax).  The DCC and fixed variants are re-run through
the same composition as baselines; the paper's values are not copied in.

Usage::

    ATTACC_DCC_ROOT=<DCC>/src/src python -m benchmarks.attacc.scripts.compose_e2e \\
        --sessions r51/attacc_qkv_b1.json ... r51/attacc_score_b4.json --out e2e.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TCK_NS = 0.769  # HBM3_5.2Gbps; DCC predictor.get_running_time_ms
LABEL = "tenon_time(ms)"
CONFIGS = [(lin, lout, batch) for batch in (1, 4) for lin, lout in ((128, 2048), (2048, 128), (2048, 2048))]
# DCC's write_csv column order: tag (6) + config (12) + s_perf (7) + g_perf
# (13: all, matmul, fc, comm, etc, qkv, prj, ff, g2g, x2g, softmax, act, norm).
G_PERF = 18 + 7
G_COLUMNS = ["all", "matmul", "fc", "comm", "etc", "qkv", "prj", "ff", "g2g", "x2g", "softmax", "act", "norm"]
# GPT-13B rows DCC's model reads from the PIM tables (per head instance count
# 8 at batch 1, 32 at batch 4): the QKV and projection GEMVs and the scoring
# kernel at every 16-aligned context length up to the longest configuration.
GPT13B_HEADS = (8, 32)
GPT13B_GEMV = ((5120, 384), (128, 5120))
GPT13B_LENGTHS = range(144, 4096 + 1, 16)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def selected_times(session_paths) -> tuple[dict, dict]:
    """``(times, provenance)``: cycles per kernel key from the sessions'
    shared-pipeline records, conflicts rejected, references ignored."""
    times, sources, ignored = {}, {}, {}
    for path in session_paths:
        session = json.load(open(path))
        for record in session["runs"]:
            key = (record["family"], *map(int, record["shape"]))
            if record["selection"] != "shared-pipeline":
                ignored[record["selection"]] = ignored.get(record["selection"], 0) + 1
                continue
            cycles = int(record["memory_system_cycles"])
            if key in times and times[key] != cycles:
                raise ValueError(f"conflicting measurements for {key}: {times[key]} in {sources[key]} and {cycles} in {path}")
            times[key] = cycles
            sources.setdefault(key, str(path))
    provenance = {
        "sessions": [
            {"path": str(p), "compiler": json.load(open(p)).get("compiler"), "simulator": {k: v for k, v in (json.load(open(p)).get("simulator") or {}).items() if k in ("binary_sha256", "library_sha256", "tck_ns")}, "runs": len(json.load(open(p))["runs"]), "failures": len(json.load(open(p)).get("failures", []))}
            for p in session_paths
        ],
        "selected_records": len(times),
        "ignored_reference_records": ignored,
    }
    return times, provenance


def required_rows() -> set:
    rows = set()
    for heads in GPT13B_HEADS:
        for reduction, outputs in GPT13B_GEMV:
            rows.add(("gemv", heads, reduction, outputs))
        for length in GPT13B_LENGTHS:
            rows.add(("attention", heads, 128, length))
    return rows


def write_tables(dcc_root: Path, times: dict, out: Path) -> dict:
    """DCC's per-kernel tables with a ``tenon_time(ms)`` column: measured
    rows converted at tCK, every other row NaN."""
    report = {"measured_rows": 0, "nan_rows": 0, "missing_required": []}
    filled = set()
    for name, family in (("@attacc-gemv_inference.csv", "gemv"), ("@attacc-attn_inference.csv", "attention")):
        rows = list(csv.DictReader((dcc_root / "temp" / "simulation" / name).open()))
        header = list(rows[0]) + ([LABEL] if LABEL not in rows[0] else [])
        for row in rows:
            key = (family, int(row["#head"]), int(row["input_dim"]), int(row["output_dim"]))
            if key in times:
                row[LABEL] = repr(times[key] * TCK_NS / 1e6)
                report["measured_rows"] += 1
                filled.add(key)
            else:
                row[LABEL] = "nan"
                report["nan_rows"] += 1
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    report["missing_required"] = sorted(str(k) for k in required_rows() - filled)
    report["measured_outside_tables"] = sorted(str(k) for k in set(times) - filled)
    return report


def decode(table_dir: Path, batch, lin, lout, *, label, gpu_only, pipe, parallel, qkv_proj_on_pim=True) -> dict:
    from inference.config import (  # DCC, imported from ATTACC_DCC_ROOT
        DataType,
        DeviceType,
        GPUType,
        InterfaceType,
        PIMType,
        make_model_config,
        make_pim_config,
        make_xpu_config,
    )
    from inference.system import System

    model = make_model_config("GPT-13B", DataType.W16A16)
    # DCC's placement switch: with it the QKV and projection GEMVs run on
    # PIM (the DCC and Tenon variants); without it only attention does (the
    # original AttAcc placement DCC uses for its fixed-tiling baseline,
    # ``run_inference.py``: ``modelinfos["DCC"] = (res_label == "our_time(ms)")``).
    model["DCC"] = (not gpu_only) and qkv_proj_on_pim
    xpu = make_xpu_config(GPUType.A100a, num_gpu=1, mem_cap=80 * 1024**3)
    system = System(xpu["GPU"], model)
    if not gpu_only:
        pim = make_pim_config(PIMType.BA, InterfaceType.NVLINK3, num_attacc=1, power_constraint=False, table_path=str(table_dir) + "/")
        pim["RES_ATTACC_LABEL"] = label
        system.set_accelerator(model, DeviceType.PIM, pim)
    perfs = []
    system.simulate(batch, lin, lout, perfs=perfs, pipe=pipe, parallel_ff=parallel, power_constraint=False)
    tag, config, perf = perfs[0]
    flat = tag + config + perf
    stages = {name: float(flat[G_PERF + i]) for i, name in enumerate(G_COLUMNS)}
    if not math.isfinite(stages["all"]):
        raise ValueError(f"the composition read an unmeasured kernel row (label {label!r}, config {(lin, lout, batch)})")
    return {"g_time_ms": stages["all"], "stages_ms": stages}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="+", required=True, help="run_ramulator.py session files")
    parser.add_argument("--out", help="write the composition record (JSON)")
    args = parser.parse_args(argv)
    root = os.environ.get("ATTACC_DCC_ROOT")
    if not root or not (Path(root) / "inference" / "system.py").is_file():
        raise SystemExit("set ATTACC_DCC_ROOT to a DCC checkout's src/src directory")
    dcc_root = Path(root).resolve()
    sys.path.insert(0, str(dcc_root))
    times, provenance = selected_times(args.sessions)
    record = {
        "kind": "analytical-composition",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "tck_ns": TCK_NS,
        "model": "DCC inference/system.py: GPT-13B, A100 roofline, PIMType.BA, NVLINK3, one AttAcc device; prefill excluded; base = no pipelining / no FFN parallelism, full = both",
        "dcc": {"root": str(dcc_root), "git_revision": _git(dcc_root, "rev-parse", "HEAD"), "files_sha256": {name: _sha256(dcc_root / name) for name in ("inference/system.py", "inference/devices.py", "inference/config.py", "inference/model.py", "temp/simulation/@attacc-gemv_inference.csv", "temp/simulation/@attacc-attn_inference.csv") if (dcc_root / name).is_file()}},
        "kernel_times": provenance,
        "configs": [],
    }
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="tenon-attacc-e2e-") as tmp:
        table_dir = Path(tmp)
        record["tables"] = write_tables(dcc_root, times, table_dir)
        if record["tables"]["missing_required"]:
            raise SystemExit(f"unmeasured GPT-13B kernel rows: {record['tables']['missing_required'][:5]} ... ({len(record['tables']['missing_required'])})")
        os.chdir(dcc_root)
        try:
            for lin, lout, batch in CONFIGS:
                gpu = decode(table_dir, batch, lin, lout, label="baseline(ms)", gpu_only=True, pipe=0, parallel=False)
                entry = {"lin": lin, "lout": lout, "batch": batch, "gpu_roofline_ms": gpu["g_time_ms"], "variants": {}}
                for name, label, on_pim in (("tenon", LABEL, True), ("dcc", "our_time(ms)", True), ("fixed", "baseline(ms)", False)):
                    base = decode(table_dir, batch, lin, lout, label=label, gpu_only=False, pipe=0, parallel=False, qkv_proj_on_pim=on_pim)
                    full = decode(table_dir, batch, lin, lout, label=label, gpu_only=False, pipe=True, parallel=True, qkv_proj_on_pim=on_pim)
                    entry["variants"][name] = {
                        "placement": "QKV, projection, attention and softmax on PIM" if on_pim else "attention and softmax on PIM, QKV and projection on the GPU (AttAcc's original placement, DCC's fixed baseline)",
                        "base_ms": base["g_time_ms"], "base_speedup": gpu["g_time_ms"] / base["g_time_ms"], "base_stages_ms": base["stages_ms"],
                        "full_ms": full["g_time_ms"], "full_speedup": gpu["g_time_ms"] / full["g_time_ms"], "full_stages_ms": full["stages_ms"],
                    }
                record["configs"].append(entry)
                print(f"{lin:5d} {lout:5d} bs{batch}: gpu {gpu['g_time_ms']:.4f} ms | " + " | ".join(f"{n} base {v['base_speedup']:.2f}x full {v['full_speedup']:.2f}x" for n, v in entry["variants"].items()), flush=True)
        finally:
            os.chdir(cwd)
    text = json.dumps(record, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
