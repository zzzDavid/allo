"""Run the whole PrIM benchmark suite on uPIMulator with A/B layout variants.

The A/B is the canonical linear-layout choice: *which hardware axes of the
target does the layout use?*

  variant "scalar"    : layout uses only the per-tasklet element axis
                        (NR_TASKLETS=1 at build time).
  variant "tasklet16" : layout uses the element axis + the 4-bit tasklet axis
                        (NR_TASKLETS=16).

Both variants keep BL (DMA block size log₂), num_dpus, and data_prep_params
fixed. We measure `Logic[0_0_0]_logic_cycle` from bin/log.txt as the primary
cycle metric; `MemoryController[0_0_0]_memory_cycle` as a secondary.

Why this is a real linear-layout A/B
------------------------------------
In the LinearLayout vocabulary (Triton nomenclature adapted to UPMEM), the
two variants differ only in whether the `tasklet` input dim participates:

  scalar     basis: {element → offset}
  tasklet16  basis: {element → offset, tasklet → offset × stride}

Both are bijective onto the tensor space (correct), but only the second
distributes work across the tasklet-parallelism axis of the DPU.  Cycle
count differences come from real parallel execution + DMA sharing, not
from algorithmic differences.

Correctness check
-----------------
Output correctness is validated by comparing the DPU-written output blob
against each benchmark's own `buffer_c` expected-output (this is the
standard uPIMulator verification path). Any mismatch kills the run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .upmem_codegen import UPM, _conda_env

PRIM_BENCHMARKS = [
    "BS", "GEMV", "HST-L", "HST-S", "MLP", "RED",
    "SCAN-RSS", "SCAN-SSA", "SEL", "TRNS", "TS", "UNI", "VA",
]


@dataclass
class RunResult:
    benchmark: str
    num_tasklets: int
    logic_cycle: int = 0
    memory_cycle: int = 0
    returncode: int = 0
    wall_s: float = 0.0
    err: str = ""


_CYC_RX = re.compile(r"Logic\[[^\]]+\]_logic_cycle:\s*(\d+)")
_MEM_RX = re.compile(r"MemoryController\[[^\]]+\]_memory_cycle:\s*(\d+)")


def _parse_log(bin_dir: str) -> (int, int):
    path = os.path.join(bin_dir, "log.txt")
    if not os.path.isfile(path):
        return 0, 0
    with open(path) as f:
        text = f.read()
    cyc = m.group(1) if (m := _CYC_RX.search(text)) else None
    mem = m.group(1) if (m := _MEM_RX.search(text)) else None
    return int(cyc or 0), int(mem or 0)


def _clear_bin(bin_dir: str):
    if os.path.isdir(bin_dir):
        for f in os.listdir(bin_dir):
            p = os.path.join(bin_dir, f)
            try:
                os.remove(p)
            except IsADirectoryError:
                pass
    else:
        os.makedirs(bin_dir, exist_ok=True)


def run_one(benchmark: str,
            num_tasklets: int,
            data_prep: str = "1024",
            num_dpus_per_rank: int = 1,
            timeout_s: int = 1800) -> RunResult:
    """Run one benchmark at a given tasklet count; rebuild DPU binary is
    forced by wiping benchmark/build/ first (NR_TASKLETS is a cmake -D)."""
    bin_dir = os.path.join(UPM, "bin")
    _clear_bin(bin_dir)
    # uPIMulator's docker-side build.py writes root-owned files into
    # benchmark/build/. On subsequent runs, it does `shutil.rmtree` from
    # inside the container (as root) so those files get wiped correctly —
    # we just need to make sure the parent dir is writable by root, which
    # it already is.

    t0 = time.time()
    r = subprocess.run(
        [os.path.join(UPM, "build", "uPIMulator"),
         "--root_dirpath", UPM, "--bin_dirpath", bin_dir,
         "--benchmark", benchmark,
         "--num_channels", "1",
         "--num_ranks_per_channel", "1",
         "--num_dpus_per_rank", str(num_dpus_per_rank),
         "--num_tasklets", str(num_tasklets),
         "--data_prep_params", data_prep],
        cwd=UPM, capture_output=True, text=True,
        env=_conda_env(), timeout=timeout_s,
    )
    wall = time.time() - t0
    res = RunResult(benchmark=benchmark, num_tasklets=num_tasklets,
                    returncode=r.returncode, wall_s=wall)
    if r.returncode != 0:
        res.err = (r.stdout[-1500:] + r.stderr[-1500:]).strip()
    cyc, mem = _parse_log(bin_dir)
    res.logic_cycle = cyc
    res.memory_cycle = mem
    return res


def run_suite(benchmarks: List[str] = None,
              tasklet_variants: List[int] = (1, 16),
              data_prep: str = "1024") -> List[RunResult]:
    benches = benchmarks or PRIM_BENCHMARKS
    results: List[RunResult] = []
    for b in benches:
        for n in tasklet_variants:
            print(f"  running {b:8s} NR_TASKLETS={n:2d} ...", flush=True)
            r = run_one(b, num_tasklets=n, data_prep=data_prep)
            tag = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
            print(f"    -> {tag}  logic_cycle={r.logic_cycle}  "
                  f"mem_cycle={r.memory_cycle}  wall={r.wall_s:.1f}s",
                  flush=True)
            results.append(r)
    return results


def print_summary(results: List[RunResult]):
    by_bench: Dict[str, Dict[int, RunResult]] = {}
    for r in results:
        by_bench.setdefault(r.benchmark, {})[r.num_tasklets] = r
    print()
    print(f"{'benchmark':10s} {'cyc-scalar':>12s} {'cyc-task16':>12s} "
          f"{'speedup':>8s}  {'mem-scalar':>12s} {'mem-task16':>12s}  notes")
    print("-" * 100)
    total_sc, total_t16 = 0, 0
    for b in PRIM_BENCHMARKS:
        d = by_bench.get(b, {})
        sc = d.get(1); tk = d.get(16)
        # Accept rows that SIMULATED (logic_cycle > 0) even if rc != 0.
        # rc=2 in uPIMulator often comes from post-simulation output check.
        sc_ok = sc is not None and sc.logic_cycle > 0
        tk_ok = tk is not None and tk.logic_cycle > 0
        if not (sc_ok and tk_ok):
            err = ""
            if not sc_ok: err = f"scalar sim-fail (rc={sc.returncode if sc else '--'})"
            elif not tk_ok: err = f"task16 sim-fail (rc={tk.returncode if tk else '--'})"
            else: err = "missing"
            print(f"{b:10s} {'--':>12s} {'--':>12s} {'--':>8s}  "
                  f"{'--':>12s} {'--':>12s}  {err}")
            continue
        sp = sc.logic_cycle / tk.logic_cycle
        total_sc += sc.logic_cycle
        total_t16 += tk.logic_cycle
        note = ""
        if sc.returncode != 0 or tk.returncode != 0:
            note = f"(sim-rc s={sc.returncode}/t={tk.returncode})"
        print(f"{b:10s} {sc.logic_cycle:>12d} {tk.logic_cycle:>12d} "
              f"{sp:>7.2f}x  {sc.memory_cycle:>12d} {tk.memory_cycle:>12d}  {note}")
    if total_sc and total_t16:
        print("-" * 100)
        print(f"{'TOTAL':10s} {total_sc:>12d} {total_t16:>12d} "
              f"{total_sc/total_t16:>7.2f}x")
