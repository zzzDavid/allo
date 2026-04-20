"""Cross-check: feed the DSL-emitted AiM trace into the real AiM simulator.

The AiM emitter already produces `AiM ...` text. Here we:
  1. Compile the self-attention program for the AiM target.
  2. Prepend the boilerplate (GPR writes) that any AiM trace needs.
  3. Drop `host.softmax` emissions (those run off-chip) and append EOC.
  4. Run the resulting trace through the aim_simulator docker image.
  5. Assert the simulator terminates cleanly and reports nonzero MAC cycles.

This does NOT verify numerical correctness on AiM — the AiM simulator is
trace-only and has no functional model (see report 06). It DOES verify that
the emitter produces a trace the real simulator accepts, which is the honest
boundary of what we can check without a functional AiM model.
"""
import os
import shutil
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np

from allo.pim import lower
from allo.pim.backends import build_aim
sys.path.insert(0, HERE)
from test_self_attention import make_attention


REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
AIM = os.path.join(REPO, "experiments", "simulators", "aim_simulator")


def emit_aim_trace(emitted_lines, out_path):
    lines = []
    lines.append("# DSL-compiled self-attention — AiM trace")
    # preload some GPRs so WR_GB / WR_BIAS have source data
    for gpr in range(8):
        lines.append(f"W GPR {gpr}")
    # broadcast mode for MAC_ABK from GB
    lines.append("W CFR 0 1")
    for ln in emitted_lines:
        if ln.startswith("host."):
            lines.append(f"# {ln}  (runs on host)")
            continue
        lines.append(ln)
    lines.append("AiM SYNC")
    lines.append("AiM EOC")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_aim_trace_runs_in_simulator():
    if not shutil.which("docker"):
        print("docker not available — skip")
        return
    # quick check the image exists
    r = subprocess.run(["docker", "image", "inspect", "aim-simulator-build"],
                       capture_output=True)
    if r.returncode != 0:
        print("aim-simulator-build image not built — skip")
        return

    prog = make_attention(M=8, S=16, D=8)
    res = lower(prog, build_aim())
    assert not res.unlowered

    trace_name = "attention_dsl.trace"
    trace_host = os.path.join(AIM, "test", trace_name)
    emit_aim_trace(res.emitted, trace_host)

    # show trace to aid debugging
    with open(trace_host) as f:
        lines = f.readlines()
    print(f"\nemitted {len(lines)} trace lines; first 8:")
    for ln in lines[:8]:
        print("   ", ln.rstrip())
    print("   ...")

    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         f"cd /work && ./build/ramulator2 -f test/example.yaml -t test/{trace_name}"],
        capture_output=True, text=True, timeout=180,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"simulator failed:\n{out[-2000:]}"

    # parse: look for any nonzero MAC_ABK cycle line
    mac_cycles = 0
    mem_cycles = 0
    for line in out.splitlines():
        s = line.strip()
        if "AiM_ISR_MAC_ABK_cycles:" in s:
            v = int(s.split(":")[1].split("#")[0].strip())
            mac_cycles += v
        elif "memory_system_cycles:" in s and mem_cycles == 0:
            mem_cycles = int(s.split(":")[1].split("#")[0].strip())
    print(f"aim sim: memory_system_cycles={mem_cycles}  total_mac_abk_cycles={mac_cycles}")
    assert mac_cycles > 0, "AiM simulator reported zero MAC_ABK cycles"
    assert mem_cycles > 0


if __name__ == "__main__":
    test_aim_trace_runs_in_simulator()
    print("\nok  test_aim_trace_runs_in_simulator")
