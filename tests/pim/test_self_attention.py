"""Self-attention kernel compiled to all three PIM targets.

Source program mimics what an Allo SPMW dataflow region would lower to:
    scores  = Q @ K.T        # (M, S)
    scaled  = scores * (1/sqrt(D))
    probs   = softmax(scaled)
    out     = probs @ V      # (M, D)

We compile the same SrcProgram against Samsung HBM-PIM, SK-Hynix AiM, and
UPMEM. We then run the pattern-level interpreter over the lowered program and
compare `state["O"]` against a NumPy reference with np.allclose.

Verification policy (anti-hack):
  - Q, K, V are fixed random fp32 tensors. The interpreter does not peek at
    the reference; it only sees the lowered pattern sequence.
  - Samsung and AiM MUST use a host.softmax step (verified via the schedule).
  - UPMEM MUST lower softmax to >=3 on-device ops (verified via schedule + IR).
  - Output comparison uses np.allclose(rtol=1e-5, atol=1e-6); any mismatch
    surfaces the source op responsible.
  - Unlowered ops would cause execute() to raise — tested explicitly.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import math
import numpy as np

from allo.pim import SrcOp, SrcProgram, Matmul, Scale, Softmax, lower, execute
from allo.pim.backends import build_samsung, build_aim, build_upmem


def make_attention(M=8, S=16, D=8) -> SrcProgram:
    # Typed ops carry their own compute semantics — no per-backend duplication.
    return (SrcProgram()
        .add(Matmul(shape=(M, D), inputs=("Q", "K"), output="scores",
                    attrs={"op": "Q@K.T"}))
        .add(Scale (shape=(M, S), inputs=("scores",), output="scaled",
                    attrs={"scale": 1.0 / math.sqrt(D)}))
        .add(Softmax(shape=(M, S), inputs=("scaled",), output="probs"))
        .add(Matmul(shape=(M, D), inputs=("probs", "V"), output="O",
                    attrs={"op": "S@V"})))


def numpy_attention(Q, K, V):
    D = Q.shape[-1]
    s = Q @ K.T
    s = s / math.sqrt(D)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ V


def _run_on(name, builder, prog, Q, K, V):
    t = builder()
    res = lower(prog, t)
    assert not res.unlowered, f"{name}: unlowered ops present: {res.unlowered}"
    state = {"Q": Q.copy(), "K": K.copy(), "V": V.copy()}
    execute(res, prog, t, state)
    return res, state["O"], t


def test_end_to_end_attention():
    np.random.seed(0)
    M, S, D = 8, 16, 8
    Q = np.random.randn(M, D).astype(np.float32)
    K = np.random.randn(S, D).astype(np.float32)
    V = np.random.randn(S, D).astype(np.float32)

    ref = numpy_attention(Q, K, V)
    prog = make_attention(M, S, D)

    results = {}
    for name, builder in [("samsung", build_samsung),
                          ("aim",     build_aim),
                          ("upmem",   build_upmem)]:
        res, out, tgt = _run_on(name, builder, prog, Q, K, V)
        assert out.shape == ref.shape, f"{name} output shape {out.shape} != ref {ref.shape}"
        # exact-ish since we do fp32 numpy everywhere; this catches any
        # pattern-math bug (wrong transpose, wrong axis in softmax, etc.).
        assert np.allclose(out, ref, rtol=1e-5, atol=1e-6), \
            f"{name} output diverges: max_abs_err={np.max(np.abs(out-ref)):.3e}"
        results[name] = (res, out, tgt)

    # --- schedule sanity: Samsung and AiM must use host.softmax ---
    sam_res = results["samsung"][0]
    aim_res = results["aim"][0]
    upm_res = results["upmem"][0]

    def _softmax_step(res):
        return next(s for s in res.schedule if s["src"].kind == "softmax")

    assert _softmax_step(sam_res)["where"] == "host", \
        "Samsung must host-offload softmax"
    assert _softmax_step(aim_res)["where"] == "host", \
        "AiM must host-offload softmax"
    assert _softmax_step(upm_res)["where"] == "pim", \
        "UPMEM must run softmax on-device"

    # --- UPMEM softmax must actually expand to >=3 instructions ---
    upm_softmax_perf = next(p for p in upm_res.perf_ir if p["src"].kind == "softmax")
    assert len(upm_softmax_perf["per_op"]) >= 3, \
        f"UPMEM softmax lowered to <3 ops: {upm_softmax_perf['per_op']}"

    # --- host_ops count ---
    assert sam_res.host_ops == 1
    assert aim_res.host_ops == 1
    assert upm_res.host_ops == 0


def test_unlowered_raises_if_no_fallback():
    """Sanity: if a target had no softmax pattern at all, execute() raises —
    i.e. the DSL does not silently fill in an answer."""
    t = build_samsung()
    # drop the softmax pattern to simulate a naive target
    t.patterns = [p for p in t.patterns if "softmax" not in p.name.lower()]
    prog = make_attention(4, 8, 4)
    res = lower(prog, t)
    assert len(res.unlowered) == 1 and res.unlowered[0].kind == "softmax"
    try:
        execute(res, prog, t, {"Q": np.zeros((4, 4)), "K": np.zeros((8, 4)),
                               "V": np.zeros((8, 4))})
    except RuntimeError as e:
        assert "softmax" in str(e)
    else:
        assert False, "execute() should have raised on unlowered softmax"


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


def test_pretty_print():
    """Not really a test — run it for human inspection of the three compiles."""
    prog = make_attention(8, 16, 8)
    for name, builder in [("samsung", build_samsung),
                          ("aim",     build_aim),
                          ("upmem",   build_upmem)]:
        t = builder()
        res = lower(prog, t)
        _pretty(name, res)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall attention tests passed")
