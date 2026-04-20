"""APU v1 two-matvec chain — linear-layout A/B on real hardware.

Workload (MLP-block shape):
    y1 = W1 @ x1           W1[M=32768, K=4] uint16 (M the spatial / VR-lane
                           axis, K the reduction / temporal axis — the
                           communication-aware SVP mapping of Zhang et al.
                           MICRO'25 §4.2).
    y2 = W2 @ x2           Same shape.
    z  = y1 + y2           Elementwise skip-combine (32K VR).

Linear-layout A/B (SVP form held constant, only intermediate persistence
differs):

    A  `(k, stage) -> time`   : one VR bank reused between stages.
         Consequences: reload W2 over W1 VRs; spill y1 to L4 and reload it
         before the final combine.

    B  `(k, stage) -> vr`     : `stage` is lifted to a 1-bit VR axis.
         Consequences: W1 and W2 co-resident in disjoint VRs; y1 stays
         resident in a reserved VR across the stage boundary.

Correctness: the emitted host.c checks every one of the 32 768 output
elements against a CPU-side `z[m] = sum_k W1[k,m]*X1[k] + W2[k,m]*X2[k]`
reference. The runner fails loudly if either layout reports a single mismatch.

Writes `apu_v1_matmul_chain_ab_results.json` in this directory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from allo.pim.runtime.apu_v1_codegen import (
    gen_apu_v1_matmul_chain_project,
    build_and_run_apu_v1,
    capture_apu_log,
)

K = 4
LAYOUTS = ["A", "B"]


def run_one(layout: str, workdir: Path) -> dict:
    tag = f"APU_CHAIN_{layout}_{int(time.time())}"
    lab  = f"apu_chain_{layout.lower()}"
    proj = workdir / layout
    gen_apu_v1_matmul_chain_project(
        str(proj), K=K, layout=layout,
        lab_name=lab, run_tag=tag)
    r = build_and_run_apu_v1(str(proj), lab_name=lab, timeout_s=600)
    passed = r.get("returncode") == 0 and "PASS" in (r.get("stdout") or "")
    time.sleep(2)
    counters = capture_apu_log(tag, timeout_s=60)
    return {
        "layout": layout,
        "run_tag": tag,
        "host_ok": passed,
        "host_stdout_tail": "\n".join(
            (r.get("stdout") or "").splitlines()[-8:]),
        "counters": counters,
        "build_errors": r.get("stderr") if not passed else None,
    }


def main():
    workdir = Path("/tmp") / f"apu_v1_matmul_chain_{int(time.time())}"
    workdir.mkdir()
    print(f"workdir: {workdir}\nK={K}, M=32768")

    results = []
    for layout in LAYOUTS:
        print(f"\n=== layout {layout} ===")
        r = run_one(layout, workdir)
        print(f"  host_ok: {r['host_ok']}")
        if not r["host_ok"]:
            print(f"  stdout tail: {r['host_stdout_tail']}")
            if r["build_errors"]:
                print(f"  stderr: {r['build_errors'][:1500]}")
        for sec, v in r["counters"].items():
            if isinstance(v, dict) and "crun" in v:
                print(f"  {sec:<14} crun={v['crun']:>8} iall={v['iall']:>6}"
                      f" {v.get('microsec500', 0):>4} us")
        results.append(r)

    # summary
    by = {r["layout"]: r["counters"] for r in results}
    sections = ["total", "load_W", "stage1", "stage1_spill", "load_W2",
                "stage2", "combine", "store_z"]
    summary = {"K": K, "M": 32768, "runs": results, "comparison": {}}
    for sec in sections:
        a = by.get("A", {}).get(sec, {}).get("crun", 0)
        b = by.get("B", {}).get(sec, {}).get("crun", 0)
        summary["comparison"][sec] = {
            "A_crun": a, "B_crun": b,
            "saved": a - b,
            "speedup": (a / b) if b > 0 else None,
        }

    out = HERE / "apu_v1_matmul_chain_ab_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")

    print("\n=== Two-matvec chain: A vs B (crun cycles) ===")
    print(f"{'section':<14}  {'A':>10}  {'B':>10}  {'saved':>8}  {'speedup':>8}")
    for sec in sections:
        c = summary["comparison"][sec]
        sp = f"{c['speedup']:.2f}x" if c["speedup"] else "--"
        print(f"{sec:<14}  {c['A_crun']:>10}  {c['B_crun']:>10}  "
              f"{c['saved']:>8}  {sp:>8}")


if __name__ == "__main__":
    main()
