"""Quick probe: does NR_TASKLETS=1 vs 16 give measurably different cycles
on VA?  This is the smoke test before running the whole PrIM suite."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from allo.pim.runtime.upmem_prim_suite import run_one

if __name__ == "__main__":
    r1 = run_one("VA", num_tasklets=1, data_prep="1024")
    r16 = run_one("VA", num_tasklets=16, data_prep="1024")
    print()
    print(f"VA: scalar  logic_cycle={r1.logic_cycle}  mem={r1.memory_cycle}")
    print(f"VA: task16  logic_cycle={r16.logic_cycle}  mem={r16.memory_cycle}")
    if r16.logic_cycle > 0:
        print(f"  speedup = {r1.logic_cycle/r16.logic_cycle:.2f}x")
    if r1.returncode != 0 or r16.returncode != 0:
        print("FAIL — one of the runs failed")
        if r1.err: print("scalar err:", r1.err[-400:])
        if r16.err: print("task16 err:", r16.err[-400:])
