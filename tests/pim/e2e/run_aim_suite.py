import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from dataclasses import asdict
from allo.pim.runtime.aim_suite import run_suite, print_summary


if __name__ == "__main__":
    results = run_suite()
    out = os.path.join(HERE, "aim_suite_results.json")
    with open(out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print_summary(results)
    print(f"\nresults saved to {out}")
