"""Run the full PrIM benchmark suite on uPIMulator with A/B layout variants."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from dataclasses import asdict
from allo.pim.runtime.upmem_prim_suite import run_suite, print_summary, PRIM_BENCHMARKS


if __name__ == "__main__":
    subset = None
    if len(sys.argv) > 1:
        subset = sys.argv[1].split(",")

    # Use data_prep_params=256 (4x smaller than PrIM default 1024) so the
    # scalar (NR_TASKLETS=1) runs of matmul-style benchmarks finish in
    # minutes, not hours. Both A and B variants use the same size.
    # Import run_one directly so we can checkpoint after each benchmark.
    from allo.pim.runtime.upmem_prim_suite import (
        run_one, print_summary, PRIM_BENCHMARKS)

    out = os.path.join(HERE, "upmem_prim_suite_results.json")
    benches = subset or PRIM_BENCHMARKS
    results = []
    for b in benches:
        for n in (1, 16):
            print(f"  running {b:8s} NR_TASKLETS={n:2d} ...", flush=True)
            r = run_one(b, num_tasklets=n, data_prep="256")
            tag = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
            print(f"    -> {tag}  logic_cycle={r.logic_cycle}  "
                  f"mem_cycle={r.memory_cycle}  wall={r.wall_s:.1f}s",
                  flush=True)
            results.append(r)
            # checkpoint
            with open(out, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
    print_summary(results)
    print(f"\nfull results saved to {out}")
