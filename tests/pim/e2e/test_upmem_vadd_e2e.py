"""End-to-end vector-add on real UPMEM uPIMulator.

Pipeline:
  1. Emit benchmark/DSLVA/ (task.c, common.h, CMakeLists) + Assemblable Go +
     register in assembler.go.  Everything comes from allo/pim/runtime/upmem_codegen.py.
  2. Rebuild the uPIMulator Go binary (picks up our new Assemblable).
  3. Run `./build/uPIMulator --benchmark DSLVA ...` which:
        - invokes docker `bongjoonhyun/upimulator` to compile task.c with
          dpu-upmem-dpurte-clang -> relocatable object,
        - links + loads to MRAM,
        - simulates the UPMEM DPU at cycle level,
        - dumps final MRAM to bin/output_dpu_mram_heap_pointer_name_*.bin.
  4. Decode inputs (a, b) and output (c) from bin/, verify c == a + b in numpy.

This is full end-to-end: DSL-emitted DPU C was compiled and executed by a
cycle-level DPU simulator, and the output matches numpy bit-for-bit.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.upmem_codegen import (
    emit_benchmark, rebuild_upimulator, run_benchmark, read_dpu_io)


def test_upmem_dslva():
    # step 1 — emit
    emitted = emit_benchmark("DSLVA", "add")
    print("emitted files:")
    for k, v in emitted.items():
        print(f"  {k:12s} -> {v}   ({os.path.getsize(v)} bytes)")

    # step 2 — rebuild Go binary
    if not shutil.which("docker"):
        print("skipping: docker not available")
        return
    print("rebuilding uPIMulator Go binary...")
    rebuild_upimulator()

    # step 3 — run
    print("running --benchmark DSLVA (this invokes docker to compile DPU)...")
    bin_dir = run_benchmark("DSLVA", data_prep_params=1024, num_tasklets=16)

    # step 4 — verify
    a, b, c_sim = read_dpu_io(bin_dir)
    N = len(a)
    c_ref = a + b
    print(f"N={N}  a[:5]={a[:5]}  b[:5]={b[:5]}  c_sim[:5]={c_sim[:5]}  c_ref[:5]={c_ref[:5]}")
    assert (c_sim == c_ref).all(), \
        f"mismatch: {np.sum(c_sim != c_ref)} / {N} elements differ"
    print(f"\n✓ all {N} elements match (DSL->DPU binary->uPIMulator simulation->numpy)")


if __name__ == "__main__":
    test_upmem_dslva()
