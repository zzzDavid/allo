"""End-to-end AiM MAC_ABK dot-product test.

This exercises the *real* AiM compute unit (per-bank MAC), not just GPR math.
Mapping:
  - a has length 16*rows: splits into `rows` burst-rows.
  - For one channel, we put each burst-row of `a` in bank 0 at row=r.
  - We put each burst-row of `b` in the channel's global buffer (opsize=rows).
  - WR_BIAS zeros. MAC_SBK opsize=rows bank=0 row=0 -> accumulator =
    sum_r a_row_r * b_row_r elementwise across 16 lanes.
  - RD_SBK at row 0 (after writing MAC result back via COPY)... actually we
    use RD_MAC to pull from the MAC register. The first lane of the MAC reg
    ends up being the first-lane dot; to get the sum, we compute all 16 lanes
    and sum in numpy (that matches how real AiM kernels use MAC-ABK for GEMV:
    each bank produces 16 partial sums which the host tree-reduces).

Verification: dot(a,b) -- returned as a 16-lane vector whose sum == numpy.
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
CHAN = 0   # use channel 0 throughout


def emit_mac_trace(a: np.ndarray, b: np.ndarray):
    """Emit ISRs to compute per-lane a*b element-wise-summed over `rows`
    burst-rows. a.shape == b.shape == (rows*16,)."""
    assert a.shape == b.shape and a.shape[0] % LANES == 0
    rows = a.shape[0] // LANES
    a_rows = a.reshape(rows, LANES)
    b_rows = b.reshape(rows, LANES)

    lines = [f"# DSL-emitted AiM MAC_SBK dot, rows={rows}"]
    staging = {}
    # 1. put a_rows into bank 0 rows 0..rows-1 via WR_SBK
    next_idx = 0
    for r in range(rows):
        # stage GPR r with a_rows[r]
        staging.setdefault(next_idx, []).append((r, a_rows[r]))
        lines.append(f"W GPR {r}")                 ; next_idx += 1
        lines.append(f"AiM WR_SBK {r} {1 << CHAN} 0 {r}") ; next_idx += 1
    # 2. put b_rows into GB via WR_GB
    gb_base = rows
    for r in range(rows):
        staging.setdefault(next_idx, []).append((gb_base + r, b_rows[r]))
        lines.append(f"W GPR {gb_base + r}")       ; next_idx += 1
    lines.append(f"AiM WR_GB {rows} {gb_base} {1 << CHAN}") ; next_idx += 1
    # 3. zero MAC bias via WR_BIAS from a zero-GPR
    zero_gpr = gb_base + rows
    staging.setdefault(next_idx, []).append((zero_gpr, np.zeros(LANES, dtype=np.float32)))
    lines.append(f"W GPR {zero_gpr}")              ; next_idx += 1
    lines.append(f"AiM WR_BIAS {zero_gpr} {1 << CHAN}") ; next_idx += 1
    # 4. broadcast mode + MAC_SBK (all `rows` bursts, bank 0, row 0)
    lines.append("W CFR 0 1")                      ; next_idx += 1
    lines.append(f"AiM MAC_SBK {rows} {1 << CHAN} 0 0") ; next_idx += 1
    # 5. read MAC back to GPR 100
    lines.append(f"AiM RD_MAC 100 {1 << CHAN}")    ; next_idx += 1
    lines.append("AiM SYNC")                       ; next_idx += 1
    lines.append("AiM EOC")

    return lines, staging


def run_shadow(a, b):
    lines, staging = emit_mac_trace(a, b)
    sh = AiMShadow()
    for i, ln in enumerate(lines):
        for g, d in staging.get(i, []):
            sh.set_gpr(g, d)
        sh.step(ln)
    # MAC register for (ch,0) holds element-wise sum across rows.
    # Real AiM sums these 16 lanes via host tree reduction.
    mac = sh.mac_reg[(CHAN, 0)]
    return mac, float(mac.sum())


def run_ramulator2(lines):
    trace_path = os.path.join(AIM, "test", "dsl_macsbk.trace")
    with open(trace_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         "cd /work && ./build/ramulator2 -f test/example.yaml -t test/dsl_macsbk.trace"],
        capture_output=True, text=True, timeout=180,
    )
    out = r.stdout + r.stderr
    mem = next((int(ln.split(":")[1].split("#")[0].strip())
                for ln in out.splitlines()
                if "memory_system_cycles:" in ln), 0)
    mac = sum(int(ln.split(":")[1].split("#")[0].strip())
              for ln in out.splitlines()
              if "AiM_ISR_MAC_SBK_cycles:" in ln)
    return r.returncode, mem, mac


def test_aim_macabk_dot():
    np.random.seed(7)
    N = 64   # 4 burst rows
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    ref_full_dot = float(a @ b)
    ref_per_lane = (a.reshape(-1, LANES) * b.reshape(-1, LANES)).sum(axis=0)

    mac_vec, mac_dot = run_shadow(a, b)
    print(f"per-lane err = {np.max(np.abs(mac_vec - ref_per_lane)):.3e}")
    assert np.allclose(mac_vec, ref_per_lane, rtol=1e-6, atol=1e-6), \
        f"shadow per-lane mac diverges"
    err = abs(mac_dot - ref_full_dot)
    print(f"host-summed dot err = {err:.3e}   (ref={ref_full_dot:.6f})")
    assert err < 1e-5

    # real sim run
    have_docker = shutil.which("docker") is not None
    img_ok = False
    if have_docker:
        img_ok = subprocess.run(["docker", "image", "inspect",
                                 "aim-simulator-build"],
                                capture_output=True).returncode == 0
    if not (have_docker and img_ok):
        print("ramulator2 skipped (no docker or image)")
        return
    lines, _ = emit_mac_trace(a, b)
    rc, mem, mac_cyc = run_ramulator2(lines)
    print(f"ramulator2: rc={rc}  mem_cycles={mem}  mac_sbk_cycles={mac_cyc}")
    assert rc == 0 and mem > 0 and mac_cyc > 0
    print("ok  test_aim_macabk_dot")


if __name__ == "__main__":
    test_aim_macabk_dot()
