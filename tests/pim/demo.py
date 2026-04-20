"""Demo: one Allo-level program, three targets, three backends emitted.

Usage:
    python tests/demo.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim import SrcOp, SrcProgram, lower
from allo.pim.backends import build_samsung, build_aim, build_upmem


def make_attention_ish() -> SrcProgram:
    """A tiny program standing in for the prefill-step of a single attention head:
       scores = Q @ K^T            (gemv-ish)
       probs  = softmax(scores)
       out    = probs @ V          (gemv-ish)
    """
    p = SrcProgram()
    p.add(SrcOp(kind="gemv",    shape=(128, 64), name="QKt"))
    p.add(SrcOp(kind="softmax", shape=(128,),    name="softmax"))
    p.add(SrcOp(kind="gemv",    shape=(128, 64), name="AV"))
    return p


def make_mlp_block() -> SrcProgram:
    p = SrcProgram()
    p.add(SrcOp(kind="gemv", shape=(4096, 1024), name="fc1"))
    p.add(SrcOp(kind="add",  shape=(4096,),       name="bias1"))
    p.add(SrcOp(kind="relu", shape=(4096,),       name="act"))
    p.add(SrcOp(kind="gemv", shape=(1024, 4096), name="fc2"))
    return p


def run(name, prog):
    print(f"\n==== {name} ====")
    for builder in (build_samsung, build_aim, build_upmem):
        t = builder()
        res = lower(prog, t)
        print(res.summary())
        # show one emitted line per src-op as evidence
        for i, s in enumerate(prog.ops):
            hdr = f"  src[{i}] {s.kind}{s.shape}"
            if s in res.unlowered:
                print(f"{hdr}  -> UNLOWERED ({t.name} lacks a pattern)")
                continue
            # grab the last matching emitted lines for this src op via per_op ordering
            p_ir = res.perf_ir[i]
            ops_emitted = [e["op"] for e in p_ir.get("per_op", [])]
            print(f"{hdr}  -> [{', '.join(ops_emitted)}]   "
                  f"cycles={p_ir.get('cycles',0)}")


if __name__ == "__main__":
    run("Attention (QK^T / softmax / AV)", make_attention_ish())
    run("MLP block", make_mlp_block())
