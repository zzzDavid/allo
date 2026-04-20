"""E7 — compile-side case study for the Tenon-PIM-v0 target.

Covers success criteria from experiments/E7_tenon_pim_v0_compile/TASK.md:
  (2) vadd numerics,
  (3) GEMV numerics,
  (4) attention numerics,
  (5) predicted cycles monotonic in N,
  (6) softmax policy is on-device (LUT) and visibly labeled.

The pattern-level interpreter executes each op's `src.compute(state)` as
NumPy, so 'pattern-level-correct' means the lowered schedule reproduces
the math of the source op. Numerics here come from the NumPy compute
itself; the DSL's role is to prove the backend lowered every op to a
legal sequence (no unlowered ops, no cost-model crash).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import math
import numpy as np

from allo.pim import SrcProgram, Add, Matmul, Scale, Softmax, lower, execute
from allo.pim.backends import build_tenon_pim_v0


# ----------------------- benchmark factories --------------------------------

def _vadd_prog(N):
    return SrcProgram().add(Add(shape=(N,), inputs=("a", "b"), output="c"))


def _gemv_prog(M, K):
    # Matmul kind defaults to "gemv"; lowered through the gemv->tp_mac pattern.
    return SrcProgram().add(
        Matmul(shape=(M, K), inputs=("W", "x"), output="y",
               attrs={"op": "matmul"})
    )


def _attn_prog(M, S, D):
    return (SrcProgram()
            .add(Matmul(shape=(M, S), inputs=("Q", "K"), output="scores",
                        attrs={"op": "Q@K.T"}))
            .add(Scale (shape=(M, S), inputs=("scores",), output="scaled",
                        attrs={"scale": 1.0 / math.sqrt(D)}))
            .add(Softmax(shape=(M, S), inputs=("scaled",), output="probs"))
            .add(Matmul(shape=(M, D), inputs=("probs", "V"), output="O",
                        attrs={"op": "S@V"})))


# ----------------------- assertions on lowering ------------------------------

# -------------------------- tests --------------------------------------------

def _run_and_get(prog, state, out_key):
    t = build_tenon_pim_v0()
    res = lower(prog, t)
    assert not res.unlowered, f"unlowered ops: {res.unlowered}"
    execute(res, prog, t, state)
    return res, state[out_key], t


def test_vadd_numerics():
    """Criterion 2."""
    N = 131072
    rng = np.random.default_rng(0)
    a = rng.standard_normal(N).astype(np.float16)
    b = rng.standard_normal(N).astype(np.float16)
    ref = a + b
    res, out, _ = _run_and_get(_vadd_prog(N), {"a": a.copy(), "b": b.copy()},
                               out_key="c")
    err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    assert err == 0.0, f"vadd fp16 not exact: max_abs_err={err}"
    print(f"[vadd N={N}] cycles={res.total_cycles} max_abs_err={err}")


def test_gemv_numerics():
    """Criterion 3: GEMV 4096x1024, max_abs_err <= 1e-2."""
    M, K = 4096, 1024
    rng = np.random.default_rng(1)
    W = rng.standard_normal((M, K)).astype(np.float16) * np.float16(0.02)
    x = rng.standard_normal(K).astype(np.float16) * np.float16(0.02)
    # fp32 reference, then cast to fp16 target precision for comparison.
    ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)
    res, out, _ = _run_and_get(_gemv_prog(M, K), {"W": W.copy(), "x": x.copy()},
                               out_key="y")
    err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    assert err <= 1e-2, f"gemv fp16 precision: max_abs_err={err}"
    print(f"[gemv {M}x{K}] cycles={res.total_cycles} max_abs_err={err:.3e}")


def test_attention_numerics():
    """Criterion 4: self-attention M=128 S=128 D=64, relative err <= 1e-3."""
    M, S, D = 128, 128, 64
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((S, D)).astype(np.float32)
    V = rng.standard_normal((S, D)).astype(np.float32)
    # Independent NumPy reference (not touched by the interpreter).
    sc = Q @ K.T / math.sqrt(D)
    sc = sc - sc.max(axis=-1, keepdims=True)
    p = np.exp(sc)
    p = p / p.sum(axis=-1, keepdims=True)
    ref = p @ V

    res, out, _ = _run_and_get(_attn_prog(M, S, D),
                               {"Q": Q.copy(), "K": K.copy(), "V": V.copy()},
                               out_key="O")
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-12))
    assert rel <= 1e-3, f"attention relative err: {rel}"
    print(f"[attn M={M} S={S} D={D}] cycles={res.total_cycles} rel_err={rel:.3e}")


def test_predicted_cycles_monotonic_in_N():
    """Criterion 5: vadd cycles(N) strictly increasing at N=65536,131072,262144."""
    cycles = []
    for N in (65536, 131072, 262144):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(N).astype(np.float16)
        b = rng.standard_normal(N).astype(np.float16)
        res, _, _ = _run_and_get(_vadd_prog(N), {"a": a, "b": b}, out_key="c")
        cycles.append(res.total_cycles)
        print(f"[vadd N={N}] cycles={res.total_cycles}")
    for i in range(1, len(cycles)):
        assert cycles[i] > cycles[i - 1], \
            f"cycles non-monotonic: {cycles}"


def test_softmax_policy_is_on_device_via_lut():
    """Criterion 6: softmax is on-device, lands on tp.lut_act, is visible."""
    M, S, D = 128, 128, 64
    prog = _attn_prog(M, S, D)
    t = build_tenon_pim_v0()
    res = lower(prog, t)
    sm_sched = next(s for s in res.schedule if s["src"].kind == "softmax")
    assert sm_sched["where"] == "pim", \
        f"tenon_pim_v0 must run softmax on-device, got {sm_sched['where']}"
    sm_perf = next(p for p in res.perf_ir if p["src"].kind == "softmax")
    opnames = [e["op"] for e in sm_perf["per_op"]]
    assert "tp.lut_act" in opnames, f"softmax missing LUT step: {opnames}"
    assert len(opnames) >= 3, f"softmax lowered to <3 ops: {opnames}"

    # Print a labeled step list so the policy is visible in test output.
    print("\n[softmax-policy] ON-DEVICE via on-die LUT")
    print("  expanded to:")
    for line in res.emitted:
        if line.startswith("TP REDUCE") or line.startswith("TP LUT_ACT") \
                or line.startswith("TP SUB") or line.startswith("TP DIV"):
            print(f"    {line}")


def _pretty(name, res):
    print(f"\n--- {name} ({res.target}) ---")
    print(res.summary())
    for sch, pir in zip(res.schedule, res.perf_ir):
        src = sch["src"]
        where = sch["where"].upper()
        steps = [e["op"] for e in pir.get("per_op", [])]
        print(f"  [{where:4}] {src.kind}{tuple(src.shape)}  "
              f"pat={sch['pattern']}  cycles={pir.get('cycles',0)}  "
              f"emit={steps}")


def test_pretty_print_attention_schedule():
    """Human-readable dump of the lowered attention program."""
    prog = _attn_prog(128, 128, 64)
    t = build_tenon_pim_v0()
    res = lower(prog, t)
    _pretty("tenon_pim_v0", res)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall tenon_pim_v0 tests passed")
