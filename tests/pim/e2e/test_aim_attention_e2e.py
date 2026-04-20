"""Full self-attention on SK-Hynix AiM.

Layout (single channel, single query):
  - 1 burst = 16 fp16.  Use D = 16 so one K[s] or V[s] fits in one burst.
  - S = 16 keys/values -> fits in 1 bank row per (ch, bk) pair, 16 banks total.
  - Q in GB row 0.
  - K[s] written to bank s, row 0.   (MAC_ABK at row 0 broadcasts Q against
    all 16 banks at once -> produces 16 lane-wise dot-products; each lane
    holds dot(Q, K[s]).)
  - V[s] written to bank s, row 1.   (second MAC_ABK broadcasts probs
    against V rows -> per-lane output sum.)

Softmax: done on host, matching AttAcc-style split; AiM's AF is a pointwise
LUT (no exp/div/reduce-max).

For each step we:
  1. Emit ISR trace (with data staged to GPRs).
  2. Run through the Python AiM shadow -> numerical output.
  3. Run the same trace through the real aim_simulator (ramulator2) -> cycle
     counts. Ramulator2 is timing-only, so it can't verify numerics.
  4. Compare shadow numerics against numpy reference attention.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import math
import numpy as np

from allo.pim.runtime.aim_shadow import AiMShadow, LANES

CH = 0
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
AIM = os.path.join(REPO, "experiments", "simulators", "aim_simulator")


def emit_qkt_trace(Q, K, S):
    """Compute scores[s] = dot(Q, K[s]) for s in 0..S-1, all in one channel.
    Returns (lines, staging) where `scores` will be at shadow.mac_reg[(CH,s)]
    for s in 0..15 (S<=16)."""
    assert S <= 16, f"S must fit in 16 banks, got {S}"
    assert Q.shape == (LANES,), Q.shape
    assert K.shape == (S, LANES), K.shape

    lines = ["# AiM attention QKt"]
    staging = {}
    idx = 0

    # 1. K[s] -> bank s, row 0
    for s in range(S):
        staging.setdefault(idx, []).append((s, K[s]))
        lines.append(f"W GPR {s}"); idx += 1
        lines.append(f"AiM WR_SBK {s} {1 << CH} {s} 0"); idx += 1

    # 2. Q -> GB row 0
    staging.setdefault(idx, []).append((100, Q))
    lines.append("W GPR 100"); idx += 1
    lines.append(f"AiM WR_GB 1 100 {1 << CH}"); idx += 1

    # 3. WR_BIAS (zero)
    staging.setdefault(idx, []).append((101, np.zeros(LANES, dtype=np.float32)))
    lines.append("W GPR 101"); idx += 1
    lines.append(f"AiM WR_BIAS 101 {1 << CH}"); idx += 1

    # 4. broadcast mode + MAC_ABK at row 0
    lines.append("W CFR 0 1"); idx += 1
    lines.append(f"AiM MAC_ABK 1 {1 << CH} 0"); idx += 1

    # 5. RD_MAC into GPR 200
    lines.append(f"AiM RD_MAC 200 {1 << CH}"); idx += 1
    lines.append("AiM SYNC")
    lines.append("AiM EOC")
    return lines, staging


def emit_av_trace(probs, V, S):
    """out[d] = sum_s probs[s] * V[s, d] for d in 0..LANES-1. One channel."""
    assert S <= 16
    assert probs.shape == (S,), probs.shape
    assert V.shape == (S, LANES), V.shape

    lines = ["# AiM attention A@V"]
    staging = {}
    idx = 0

    # 1. V[s] -> bank s, row 1 (different row than K to avoid collision)
    for s in range(S):
        staging.setdefault(idx, []).append((s, V[s]))
        lines.append(f"W GPR {s}"); idx += 1
        lines.append(f"AiM WR_SBK {s} {1 << CH} {s} 1"); idx += 1

    # 2. "probs" -> GB row 0, but with probs broadcast to 16 lanes per bank.
    # AiM MAC_ABK multiplies GB[r] (16 lanes) element-wise with bank row (16 lanes).
    # To compute out[d] = sum_s probs[s] * V[s,d], we need bank s's MAC to
    # accumulate probs[s] * V[s, lane=d]. MAC operates elementwise, so the
    # operand pair per bank is (GB[0], V[s]). If GB[0] is same for all banks
    # but we want a different scalar probs[s] per bank, we can't do that
    # directly in one MAC_ABK.
    #
    # Trick: replicate probs[s] to all 16 lanes and put it as scaled V:
    #   pre-scale V[s] by probs[s] on host (write K_s -> V[s] * probs[s]),
    # then MAC_ABK with GB = all-ones. But that defeats the purpose.
    #
    # Simplest correct path: per-bank scalar multiply via MAC_SBK per s.
    # We do S separate MAC_SBK calls instead of one MAC_ABK.
    # GB <- probs[s] broadcast (16 lanes of same scalar)
    # MAC_SBK bank=s, row=1 -> MAC[s] += probs[s] * V[s]
    # host tree-reduces the S MAC registers at the end.
    #
    # WR_BIAS zero
    staging.setdefault(idx, []).append((101, np.zeros(LANES, dtype=np.float32)))
    lines.append("W GPR 101"); idx += 1
    lines.append(f"AiM WR_BIAS 101 {1 << CH}"); idx += 1
    lines.append("W CFR 0 1"); idx += 1

    for s in range(S):
        scalar = np.full(LANES, probs[s], dtype=np.float32)
        staging.setdefault(idx, []).append((102, scalar))
        lines.append("W GPR 102"); idx += 1
        lines.append(f"AiM WR_GB 1 102 {1 << CH}"); idx += 1
        lines.append(f"AiM MAC_SBK 1 {1 << CH} {s} 1"); idx += 1

    lines.append(f"AiM RD_MAC 200 {1 << CH}"); idx += 1
    lines.append("AiM SYNC")
    lines.append("AiM EOC")
    return lines, staging


def shadow_run(lines, staging):
    sh = AiMShadow()
    for i, ln in enumerate(lines):
        for (g, d) in staging.get(i, []):
            sh.set_gpr(g, d)
        sh.step(ln)
    return sh


def ramulator2_run(lines, tag):
    path = os.path.join(AIM, "test", tag + ".trace")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         f"cd /work && ./build/ramulator2 -f test/example.yaml -t test/{tag}.trace"],
        capture_output=True, text=True, timeout=300)
    mem = 0
    for ln in (r.stdout + r.stderr).splitlines():
        s = ln.strip()
        if "memory_system_cycles:" in s and mem == 0:
            mem = int(s.split(":")[1].split("#")[0].strip())
    return r.returncode, mem


def test_aim_attention_e2e():
    np.random.seed(8)
    S, D = 16, 16   # one row of banks, one burst per key
    Q = np.random.randn(D).astype(np.float32)
    K = np.random.randn(S, D).astype(np.float32)
    V = np.random.randn(S, D).astype(np.float32)

    # -- step 1: Q·K^T on AiM --
    lines1, stage1 = emit_qkt_trace(Q, K, S)
    sh1 = shadow_run(lines1, stage1)
    # AiM mac_reg[(CH, s)] contains lane-wise q*k_s; sum across lanes -> dot
    scores_shadow = np.array(
        [sh1.mac_reg[(CH, s)].sum() for s in range(S)], dtype=np.float32)
    scores_ref = K @ Q
    err1 = np.max(np.abs(scores_shadow - scores_ref))
    print(f"  Q·K^T (AiM shadow)  max_abs_err = {err1:.3e}")
    assert err1 < 1e-5

    # -- host softmax --
    scaled = scores_ref / math.sqrt(D)
    shifted = scaled - scaled.max()
    ex = np.exp(shifted); probs = ex / ex.sum()

    # -- step 2: probs @ V on AiM --
    lines2, stage2 = emit_av_trace(probs, V, S)
    sh2 = shadow_run(lines2, stage2)
    # out[d] = sum_s mac_reg[(CH, s)][d]   (MAC_SBK accumulated probs[s]*V[s])
    out_shadow = np.zeros(D, dtype=np.float32)
    for s in range(S):
        out_shadow += sh2.mac_reg[(CH, s)]
    out_ref = probs @ V
    err2 = np.max(np.abs(out_shadow - out_ref))
    print(f"  A·V   (AiM shadow)  max_abs_err = {err2:.3e}")
    assert err2 < 1e-5

    # -- cross-check the SAME traces on real ramulator2 for timing --
    have_docker = shutil.which("docker") is not None
    img_ok = False
    if have_docker:
        img_ok = subprocess.run(["docker", "image", "inspect",
                                 "aim-simulator-build"],
                                capture_output=True).returncode == 0
    if have_docker and img_ok:
        rc1, mem1 = ramulator2_run(lines1, "dsl_attn_qkt")
        rc2, mem2 = ramulator2_run(lines2, "dsl_attn_av")
        assert rc1 == 0 and rc2 == 0 and mem1 > 0 and mem2 > 0
        print(f"  ramulator2: QK^T mem_cycles={mem1}  A·V mem_cycles={mem2}")
    else:
        print("  ramulator2 skipped (no docker)")

    print("ok  test_aim_attention_e2e")


if __name__ == "__main__":
    test_aim_attention_e2e()
