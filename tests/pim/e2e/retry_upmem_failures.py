"""After the main UPMEM suite run, retry any benchmarks that failed.

BS and TS had broken include paths (fixed). GEMV task16 failed fast after a
slow scalar run (suspected bin/ dir staleness). Re-run these one at a time
with a clean slate."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from allo.pim.runtime.upmem_prim_suite import run_one

TARGETS = [
    ("BS",   1), ("BS",   16),
    ("TS",   1), ("TS",   16),
    ("GEMV", 16),
]

if __name__ == "__main__":
    out_path = os.path.join(HERE, "upmem_prim_suite_results.json")
    existing = []
    if os.path.isfile(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    # index existing by (benchmark, num_tasklets)
    idx = {(r["benchmark"], r["num_tasklets"]): i for i, r in enumerate(existing)}

    for bench, n in TARGETS:
        print(f"retry {bench} NR_TASKLETS={n} ...", flush=True)
        r = run_one(bench, num_tasklets=n, data_prep="1024")
        tag = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        print(f"  -> {tag}  logic_cycle={r.logic_cycle}  "
              f"mem_cycle={r.memory_cycle}  wall={r.wall_s:.1f}s")
        from dataclasses import asdict
        rd = asdict(r)
        if (bench, n) in idx:
            existing[idx[(bench, n)]] = rd
        else:
            existing.append(rd)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nupdated {out_path}")
