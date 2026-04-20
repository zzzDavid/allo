"""APU v1 linear-layout A/B experiment.

Problem:  y = a + b elementwise on N = 4 * 32K = 131072 uint16 elements.

Layout A ("element-only axis"): single task with a loop of 4 iterations,
    reusing VR0/VR1/VR2 each iter. The only layout input dim used is `element`.

Layout B ("element + vr axis"): single task fully unrolled across 4 VR
    triples — VR0..VR11 — so the body is straight-line 4*(2 loads)
    + 4*(add) + 4*(stores). Uses layout input dims `element` x `vr`
    (2 bits of the vr axis, i.e. log2(4) = 2).

Both are built by the DSL codegen (`pimdsl.runtime.apu_v1_codegen`), executed
on the real APU v1 hardware, and their cycle counts are pulled from the
`flo` device log via the DSL-assigned banner tag.

Writes `apu_v1_layout_ab_results.json` in this directory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make `pimdsl` importable when run as a plain script.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from allo.pim.runtime.apu_v1_codegen import (
    gen_apu_v1_project, build_and_run_apu_v1, capture_apu_log)


N = 4 * 32768  # 131072 elements, 4 VR-worths
LAYOUTS = [
    dict(name="element_only",    n_vrs_per_body=1),
    dict(name="element_plus_vr2", n_vrs_per_body=2),
    dict(name="element_plus_vr",  n_vrs_per_body=4),
]
OP = "add"


def run_one(layout: dict, workdir: Path) -> dict:
    run_tag = f"APU_AB_{layout['name']}_{int(time.time())}"
    project = workdir / layout["name"]
    lab_name = f"apu_ab_{layout['name']}"
    gen_apu_v1_project(
        str(project), N=N, op=OP,
        n_vrs_per_body=layout["n_vrs_per_body"],
        lab_name=lab_name, run_tag=run_tag)

    r = build_and_run_apu_v1(str(project), lab_name=lab_name, timeout_s=600)
    host_ok = r.get("returncode") == 0 and "PASS" in (r.get("stdout") or "")

    # Let the APU finish writing its log buffer before we drain it.
    time.sleep(2)
    counters = capture_apu_log(run_tag, timeout_s=60)

    return {
        "layout": layout["name"],
        "n_vrs_per_body": layout["n_vrs_per_body"],
        "run_tag": run_tag,
        "host_ok": host_ok,
        "host_stdout_tail": "\n".join(
            (r.get("stdout") or "").splitlines()[-8:]),
        "counters": counters,
        "build_errors": r.get("stderr") if not host_ok else None,
    }


def main():
    workdir = Path("/tmp") / f"apu_v1_layout_ab_{int(time.time())}"
    workdir.mkdir()
    print(f"workdir: {workdir}")

    results = []
    for layout in LAYOUTS:
        print(f"\n=== layout {layout['name']}  (n_vrs_per_body={layout['n_vrs_per_body']}) ===")
        r = run_one(layout, workdir)
        print(f"  host_ok: {r['host_ok']}")
        for sec, fields in r["counters"].items():
            if isinstance(fields, dict) and "crun" in fields:
                print(f"  {sec:<12} crun={fields['crun']:>8}  iall={fields['iall']:>6}  {fields.get('microsec500', 0):>4} us")
        results.append(r)

    # build a cross-layout summary table
    summary = {"N": N, "op": OP, "runs": results, "comparison": {}}
    sections = ["total", "l4_to_l1", "calc", "l1_to_l4"]
    by_layout = {r["layout"]: r["counters"] for r in results}
    baseline = "element_only"
    for sec in sections:
        a = by_layout.get(baseline, {}).get(sec, {}).get("crun", 0)
        row = {f"{baseline}_crun": a}
        for other in ("element_plus_vr2", "element_plus_vr"):
            b = by_layout.get(other, {}).get(sec, {}).get("crun", 0)
            row[f"{other}_crun"] = b
            row[f"speedup_{other}"] = (a / b) if b > 0 else None
        summary["comparison"][sec] = row

    out_path = HERE / "apu_v1_layout_ab_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # Pretty print
    print("\n=== APU v1 layout sweep (crun cycles) ===")
    print(f"{'section':<12}  {'A (loop)':>10}  {'B (nvr=2)':>10}  {'C (nvr=4)':>10}  {'B/A':>6}  {'C/A':>6}")
    for sec in sections:
        c = summary["comparison"][sec]
        a = c.get("element_only_crun", 0)
        b = c.get("element_plus_vr2_crun", 0)
        d = c.get("element_plus_vr_crun", 0)
        ba = f"{c.get('speedup_element_plus_vr2') or 0:.2f}x"
        ca = f"{c.get('speedup_element_plus_vr') or 0:.2f}x"
        print(f"{sec:<12}  {a:>10}  {b:>10}  {d:>10}  {ba:>6}  {ca:>6}")


if __name__ == "__main__":
    main()
