# VERIFIER-OWNED probe script — do not edit source files
# Probes the multi-op residency sim run directly, bypassing pytest skip-path.
# Catches and prints the full exception when uPIMulator crashes.
import sys, os

# Resolve to the allo submodule root
_here = os.path.dirname(os.path.abspath(__file__))
_allo_root = os.path.join(_here, "..", "..")
_allo_root = os.path.normpath(_allo_root)
os.chdir(_allo_root)
sys.path.insert(0, _allo_root)

# Add tests/spmw to path for fixture imports
sys.path.insert(0, os.path.join(_allo_root, "tests", "spmw"))

from _fixtures import build_upmem_target
from _schedule_search_corpus import (
    run_baseline_vs_search,
    multi_op_workload,
    sim_unavailable,
)

import traceback

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

results = []
for i in range(1, N + 1):
    print(f"\n=== attempt {i}/{N} ===", flush=True)
    target = build_upmem_target()
    workload = multi_op_workload()
    try:
        base_res, search_res = run_baseline_vs_search(target, workload)
        base_unavail = sim_unavailable(base_res)
        search_unavail = sim_unavailable(search_res)
        if base_unavail or search_unavail:
            print(f"  SKIP: sim_unavailable  base={base_unavail}  search={search_unavail}", flush=True)
            print(f"  base stdout tail: {base_res.stdout[-300:] if base_res.stdout else '<none>'}", flush=True)
            results.append("skip-unavail")
        else:
            win = search_res.cycles is not None and base_res.cycles is not None and search_res.cycles < base_res.cycles
            print(f"  PASS: base={base_res.cycles}  search={search_res.cycles}  win={win}", flush=True)
            results.append("pass" if win else "fail-no-win")
    except RuntimeError as e:
        print(f"  SKIP via RuntimeError: {str(e)[:400]}", flush=True)
        results.append("skip-runtimeerror")
    except Exception as e:
        print(f"  EXCEPTION ({type(e).__name__}): {str(e)[:400]}", flush=True)
        traceback.print_exc()
        results.append(f"exception-{type(e).__name__}")

print(f"\n=== summary: {results} ===", flush=True)
pass_count = results.count("pass")
skip_count = sum(1 for r in results if r.startswith("skip"))
print(f"pass={pass_count}  skip={skip_count}  other={N - pass_count - skip_count}", flush=True)
