"""Second-op E2E on UPMEM: proves the codegen is parametric, not a one-off.

Same pipeline as test_upmem_vadd_e2e but op="mul" -> DSLVM benchmark. Verifies
the DSL-emitted task.c for a different kernel still compiles and runs.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from allo.pim.runtime.upmem_codegen import (
    emit_benchmark, rebuild_upimulator, run_benchmark, read_dpu_io)


def test_upmem_dslvm():
    emit_benchmark("DSLVM", "mul")
    if not shutil.which("docker"):
        print("skip: docker not available")
        return
    rebuild_upimulator()
    bin_dir = run_benchmark("DSLVM", data_prep_params=1024, num_tasklets=16)
    a, b, c_sim = read_dpu_io(bin_dir)
    # Go Assemblable generated a*b as int64, stored to int32 buffer -> truncate
    c_ref = (a.astype("int64") * b.astype("int64")).astype("int32")
    n_eq = (c_sim == c_ref).sum()
    print(f"DSLVM: {n_eq}/{len(a)} elements match")
    assert n_eq == len(a)
    print("ok  test_upmem_dslvm")


if __name__ == "__main__":
    test_upmem_dslvm()
