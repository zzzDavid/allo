# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Close-floor §6c measurement: run BOTH the Tenon and Exo gemv kernels
through the in-tree bespoke GEMV host at one shape-derived (m_size,
n_size) footprint, so the common-mode ~1.13x harness offset cancels.

Asserts the design 02 §6c PASS gate:
    abs(tenon_cyc - exo_cyc) / exo_cyc <= 0.0      (deterministic Go sim)
    tenon_run == exo_run                            (instruction-count parity)

Not a pytest test (it drives the live uPIMulator, minutes per run, and
mutates the GEMV slot task.c). Run:
    python tests/spmw/_measure_upmem_gemv_floor.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import allo
from allo.spmw_codegen import _upim_root
from _fixtures import build_upmem_target, build_mlp_workload


def _parse(combined: str, key: str) -> int | None:
    m = re.search(rf"{key}:\s*(\d+)", combined)
    return int(m.group(1)) if m else None


def _run_gemv_host(root: Path, binary: Path, num_tasklets: int,
                   m_size: int, n_size: int) -> tuple[int, int]:
    """Invoke the GEMV host on whatever task.c is currently in the slot."""
    with tempfile.TemporaryDirectory() as td:
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        proc = subprocess.run(
            [str(binary), "--root_dirpath", str(root),
             "--bin_dirpath", str(bin_dir), "--benchmark", "GEMV",
             "--num_channels", "1", "--num_dpus_per_rank", "1",
             "--num_tasklets", str(num_tasklets),
             "--data_prep_params", f"{m_size},{n_size}"],
            capture_output=True, timeout=600, check=False,
        )
        combined = proc.stdout.decode("utf-8", "replace")
        log = bin_dir / "log.txt"
        if log.exists():
            combined += "\n" + log.read_text(errors="replace")
        cyc = _parse(combined, "logic_cycle")
        run = _parse(combined, "breakdown_run")
        if cyc is None:
            raise RuntimeError("no logic_cycle; tail:\n" + combined[-500:])
        return cyc, run


def main():
    root = _upim_root()
    binary = root / "build" / "uPIMulator"
    slot = root / "benchmark" / "GEMV" / "dpu" / "task.c"
    exo_baseline = slot.with_suffix(".c.exo-baseline")
    if not binary.exists():
        print("SIM UNAVAILABLE:", binary)
        return
    if not exo_baseline.exists():
        # First-run snapshot of the stock (Exo) gemv kernel.
        exo_baseline.write_text(slot.read_text(), encoding="utf-8")

    target = build_upmem_target()
    schedule = allo.customize(build_mlp_workload(), enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)
    compiled = allo.compile_for_target(target, trace)
    ctx = compiled._ctx
    nt = ctx.n_tasklets
    m_size, n_size = ctx.row_count, ctx.reduction_trip
    print(f"shape-derived: m_size={m_size} n_size={n_size} nt={nt}")

    # --- Tenon column: compiled.run() routes to the GEMV host and writes
    #     the Tenon gemv envelope into the slot. ---
    res = compiled.run()
    if "simulator unavailable" in res.stdout:
        print("SIM UNAVAILABLE:", res.stdout[:200])
        return
    tenon_cyc = res.cycles
    tenon_run = _parse(res.stdout, "breakdown_run")
    print(f"[tenon] benchmark={res.extra.get('benchmark')} "
          f"data_prep={res.extra.get('data_prep_params')} "
          f"logic_cycle={tenon_cyc} breakdown_run={tenon_run}")

    # --- Exo column: restore the stock gemv kernel, run same footprint. ---
    slot.write_text(exo_baseline.read_text(), encoding="utf-8")
    exo_cyc, exo_run = _run_gemv_host(root, binary, nt, m_size, n_size)
    print(f"[exo]   logic_cycle={exo_cyc} breakdown_run={exo_run}")

    # --- §6c gate ---
    delta = abs(tenon_cyc - exo_cyc) / exo_cyc if exo_cyc else float("nan")
    print(f"cycle parity: |{tenon_cyc}-{exo_cyc}|/{exo_cyc} = {delta:.6f}")
    print(f"instruction parity: tenon_run={tenon_run} exo_run={exo_run} "
          f"-> {'MATCH' if tenon_run == exo_run else 'MISMATCH'}")
    ok = (delta <= 0.0) and (tenon_run == exo_run)
    print("§6c GATE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
