"""E10 — capability gaps surface structurally (UPMEM softmax removal).

Counterpart to ``test_unlowered_raises_if_no_fallback`` in
``test_self_attention.py``. That test shows the *negative* path: drop a
pattern with no fallback and the compiler refuses (no silent answer). This
test shows the *positive* path: drop the device-side softmax decomposition
on UPMEM, register a host fallback for softmax, and the compiler
auto-lifts softmax to the host without any code change in
``lowering.py`` / ``target.py``.

Why this matters (paper claim C1): capability gaps fall out of the
structure of the target description. The compiler must not branch on the
op name "softmax"; it must just observe that the only matching pattern
emits a ``host.*`` instruction.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import math
import numpy as np

from allo.pim import lower, execute
from allo.pim.backends import build_upmem

# Reuse the attention program + numpy reference from the sibling test so we
# verify exactly the same kernel.
from test_self_attention import make_attention, numpy_attention


def _strip_device_softmax_add_host_fallback(t):
    """Mutate the UPMEM target: remove softmax->dpu_fused, register
    host.softmax + a softmax->HOST pattern. Returns (t, removed_names)."""
    removed = [p.name for p in t.patterns if p.name == "softmax->dpu_fused"]
    t.patterns = [p for p in t.patterns if p.name != "softmax->dpu_fused"]

    # Register the host fallback op (the prefix "host." is what the
    # lowering pass uses to label a step as host-placed).
    t.op("host.softmax", lanes=1, latency=500, energy_pJ=20.0,
         emit="host.call softmax axis=-1 shape={shape}")

    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [("host.softmax", {"shape": s.shape})],
        name="softmax->HOST",
    )
    return t, removed


def test_softmax_lifts_to_host_when_device_pattern_removed():
    t = build_upmem()

    # Sanity: stock UPMEM keeps softmax on-device.
    stock = lower(make_attention(4, 8, 4), t)
    stock_softmax = next(s for s in stock.schedule if s["src"].kind == "softmax")
    assert stock_softmax["where"] == "pim", \
        f"stock UPMEM should run softmax on-device, got {stock_softmax['where']}"

    # Mutate: drop the device decomposition, register host fallback.
    t, removed = _strip_device_softmax_add_host_fallback(t)
    assert removed == ["softmax->dpu_fused"], \
        f"expected to remove softmax->dpu_fused, removed {removed}"

    prog = make_attention(M=4, S=8, D=4)
    res = lower(prog, t)

    # 1. Softmax is realized — not unlowered.
    assert not res.unlowered, \
        f"softmax should be lowered via host fallback, unlowered={res.unlowered}"

    # 2. The softmax step is placed on the host, via softmax->HOST.
    softmax_step = next(s for s in res.schedule if s["src"].kind == "softmax")
    assert softmax_step["where"] == "host", \
        f"softmax must be host-placed after device pattern removal, got {softmax_step['where']}"
    assert softmax_step["pattern"] == "softmax->HOST", \
        f"unexpected pattern {softmax_step['pattern']!r}"

    # 3. Cost rolls up via host.softmax exactly once.
    softmax_perf = next(p for p in res.perf_ir if p["src"].kind == "softmax")
    emitted_ops = [e["op"] for e in softmax_perf["per_op"]]
    assert emitted_ops == ["host.softmax"], \
        f"expected single host.softmax step, got {emitted_ops}"

    # 4. The rest of the program (Q@K.T, scale, A@V) still lowers to PIM.
    other_steps = [s for s in res.schedule if s["src"].kind != "softmax"]
    assert all(s["where"] == "pim" for s in other_steps), \
        f"non-softmax ops must stay on PIM, got {[(s['src'].kind, s['where']) for s in other_steps]}"

    pim_kinds = {s["src"].kind for s in other_steps}
    assert pim_kinds == {"gemv", "scale"}, \
        f"expected gemv+scale on PIM, got {pim_kinds}"

    # 5. host_ops counter sees exactly one host op (softmax).
    assert res.host_ops == 1, f"host_ops={res.host_ops}, expected 1"

    # 6. Numerics still match — host fallback runs the typed Softmax.compute().
    np.random.seed(0)
    M, S, D = 4, 8, 4
    Q = np.random.randn(M, D).astype(np.float32)
    K = np.random.randn(S, D).astype(np.float32)
    V = np.random.randn(S, D).astype(np.float32)
    state = {"Q": Q, "K": K, "V": V}
    execute(res, prog, t, state)
    ref = numpy_attention(Q, K, V)
    assert np.allclose(state["O"], ref, rtol=1e-5, atol=1e-6), \
        f"output diverges after lift; max_abs_err={np.max(np.abs(state['O']-ref)):.3e}"

    # Print the lowered schedule for the proof-of-criterion log.
    print("UPMEM (softmax device pattern REMOVED) schedule:")
    for sch, pir in zip(res.schedule, res.perf_ir):
        steps = [e["op"] for e in pir.get("per_op", [])]
        print(f"  [{sch['where']:4}] {sch['src'].kind:8} pattern={sch['pattern']:24} emit={steps}")


def test_compiler_does_not_branch_on_op_name_strings():
    """Structural property: lowering.py / target.py must not contain
    op-name string comparisons. The only acceptable string match is the
    "host." prefix used to label placement (a namespace, not an op name)."""
    import re

    # tests/pim/ -> tests/ -> allo (repo root of allo-package) -> allo/pim
    pim_dsl_root = os.path.join(HERE, "..", "..", "allo", "pim")
    forbidden_names = ["softmax", "exp", "div", "reduce_max",
                       "reduce_sum", "matmul", "gemv", "relu",
                       "scale", "fmac", "ewmul"]

    leaks = []
    for fname in ("lowering.py", "target.py"):
        path = os.path.join(pim_dsl_root, fname)
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                for name in forbidden_names:
                    # Look for op-name string literals "name" or 'name'.
                    if re.search(rf'["\']{name}["\']', line):
                        leaks.append((fname, lineno, line.rstrip(),
                                      f"matched name {name!r}"))
    assert not leaks, ("op-name string special cases leaked into compiler:\n"
                       + "\n".join(f"  {f}:{ln}: {src}  ({why})"
                                   for f, ln, src, why in leaks))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall capability-lift tests passed")
