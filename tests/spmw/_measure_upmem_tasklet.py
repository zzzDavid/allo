# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manual measurement driver for the UPMEM tasklet-tiling lever (design 02).

Not a pytest test -- a standalone script the coder/verifier runs to obtain
the real uPIMulator cycle counts at nt=1 vs the autoscheduled nt=T_max, so
the lever can be logged to baselines/tenon-progress.tsv as a ratio. The
lever is offset-invariant (design 02 §3.5), so the ratio is the claim.

Run:
    python tests/spmw/_measure_upmem_tasklet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import allo
from allo.spmw_autoschedule import Placement
from _fixtures import build_upmem_target, build_mlp_workload


def _compile(target, trace, layout=None):
    return allo.compile_for_target(target, trace, layout)


def main():
    target = build_upmem_target()
    workload = build_mlp_workload()
    schedule = allo.customize(workload, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    # Autoscheduled compile -- argmin should pick the parallel placement.
    compiled_auto = _compile(target, trace)
    auto_ctx = compiled_auto._ctx
    print(f"autoscheduled n_tasklets = {auto_ctx.n_tasklets}, "
          f"reduction_trip = {auto_ctx.reduction_trip}")

    res_auto = compiled_auto.run()
    print(f"[auto nt={auto_ctx.n_tasklets}] cycles={res_auto.cycles} "
          f"backend={res_auto.backend}")
    if "simulator unavailable" in res_auto.stdout:
        print("SIM UNAVAILABLE:", res_auto.stdout[:200])
        return

    # Forced nt=1 baseline: re-stage the ctx field after compile and re-run.
    compiled_one = _compile(target, trace)
    compiled_one._ctx.n_tasklets = 1
    res_one = compiled_one.run()
    print(f"[forced nt=1] cycles={res_one.cycles}")

    if res_auto.cycles and res_one.cycles:
        lever = res_one.cycles / res_auto.cycles
        print(f"MEASURED LEVER nt=1/nt={auto_ctx.n_tasklets}: {lever:.3f}x")


if __name__ == "__main__":
    main()
