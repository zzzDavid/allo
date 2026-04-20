"""End-to-end vector-add on SK-Hynix AiM.

Pipeline:
  1. Generate random fp16 inputs a, b (length 16, one AiM burst).
  2. Emit ISR trace: stage a into GPR 0, b into GPR 1; W GPR 0/1; EWADD.
  3. Run the real aim_simulator (ramulator2) on the trace via docker — verifies
     the trace is syntactically accepted and returns cycle counts.
  4. Run the same trace through the Python AiM shadow with real tensor data —
     verifies the *numerical* output against numpy a + b.

What's real here vs. what isn't
-------------------------------
- The trace is real AiM ISR format; ramulator2 accepts it (check: exit code 0,
  nonzero memory_system_cycles).
- The ramulator2 simulator has no functional model, so it cannot check output
  values. That is the reason the Python shadow exists.
- The shadow implements AiM ISR semantics per the public AiM paper / README,
  not the RTL. Correctness vs. real silicon would require a verilog co-sim.
  What we prove: "the ISR sequence, under our understanding of AiM semantics,
  computes numpy's a+b exactly."
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.aim_shadow import AiMShadow, LANES


REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
AIM = os.path.join(REPO, "experiments", "simulators", "aim_simulator")


def emit_vadd_trace(a: np.ndarray, b: np.ndarray):
    """Return (trace_lines, staging) for AiM EWADD(a, b) -> GPR 1."""
    assert a.shape == (LANES,) and b.shape == (LANES,)
    lines = [
        "# DSL-emitted AiM vector-add",
        "W GPR 0",
        "W GPR 1",
        "AiM EWADD 1 0 1",     # opsize=1, gpr0=0, gpr1=1 -> gpr1 += gpr0
        "AiM SYNC",
        "AiM EOC",
    ]
    staging = {1: [(0, a)], 2: [(1, b)]}   # index-of-line -> [(gpr,data)]
    return lines, staging


def run_shadow(a, b):
    lines, staging = emit_vadd_trace(a, b)
    sh = AiMShadow()
    for i, ln in enumerate(lines):
        if i in staging:
            for g, d in staging[i]:
                sh.set_gpr(g, d)
        sh.step(ln)
    return sh.read_gpr(1)


def run_ramulator2(lines):
    """Write trace to AIM/test/dsl_vadd.trace, run ramulator2 in docker, and
    return (exit_code, memory_system_cycles, ewadd_cycles)."""
    trace_path = os.path.join(AIM, "test", "dsl_vadd.trace")
    # we only write ISRs the simulator's parser expects (no W GPR/CFR headers
    # that carry no payload in its format; the readme shows bare W GPR works)
    with open(trace_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         "cd /work && ./build/ramulator2 -f test/example.yaml -t test/dsl_vadd.trace"],
        capture_output=True, text=True, timeout=180,
    )
    mem = 0
    ewadd = 0
    for ln in (r.stdout + r.stderr).splitlines():
        s = ln.strip()
        if "memory_system_cycles:" in s and mem == 0:
            mem = int(s.split(":")[1].split("#")[0].strip())
        if "AiM_ISR_EWADD_cycles:" in s:
            ewadd += int(s.split(":")[1].split("#")[0].strip())
    return r.returncode, mem, ewadd


def test_aim_vadd_e2e():
    np.random.seed(42)
    a = np.random.randn(LANES).astype(np.float32)
    b = np.random.randn(LANES).astype(np.float32)
    ref = a + b

    # 1. Python shadow: prove numerics match numpy.
    out = run_shadow(a, b)
    err = np.max(np.abs(out - ref))
    print(f"shadow max_abs_err = {err:.3e}")
    assert np.allclose(out, ref, rtol=1e-6, atol=1e-7), \
        f"shadow diverges: max_abs_err={err:.3e}"

    # 2. ramulator2: prove trace is accepted and runs.
    lines, _ = emit_vadd_trace(a, b)
    have_docker = shutil.which("docker") is not None
    img_ok = False
    if have_docker:
        img_ok = subprocess.run(["docker", "image", "inspect",
                                 "aim-simulator-build"],
                                capture_output=True).returncode == 0
    if not (have_docker and img_ok):
        print("ramulator2 skipped (no docker or image)")
        return
    rc, mem_cycles, ewadd_cycles = run_ramulator2(lines)
    assert rc == 0, "ramulator2 failed to accept the emitted trace"
    assert mem_cycles > 0, "ramulator2 reported zero memory_system_cycles"
    print(f"ramulator2: mem_cycles={mem_cycles}  ewadd_cycles={ewadd_cycles}  rc={rc}")
    print("ok  test_aim_vadd_e2e")


if __name__ == "__main__":
    test_aim_vadd_e2e()
