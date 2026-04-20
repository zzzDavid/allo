"""Sanity tests for the DSL:
  - same SrcProgram lowers against all three targets,
  - softmax goes to host on Samsung/AiM and on-device on UPMEM,
  - per-op cost is captured in the perf-IR,
  - emitted text matches each target's native format.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim import SrcOp, SrcProgram, lower
from allo.pim.backends import build_samsung, build_aim, build_upmem


def _prog():
    return SrcProgram().add(SrcOp(kind="gemv", shape=(64, 64), inputs=("Q","K"), output="S")) \
                       .add(SrcOp(kind="softmax", shape=(64,), inputs=("S",), output="P")) \
                       .add(SrcOp(kind="add", shape=(64,), inputs=("P","B"), output="R"))


def test_samsung_softmax_host():
    res = lower(_prog(), build_samsung())
    assert res.unlowered == []
    assert res.host_ops == 1
    softmax = next(s for s in res.schedule if s["src"].kind == "softmax")
    assert softmax["where"] == "host"


def test_aim_softmax_host():
    res = lower(_prog(), build_aim())
    assert res.host_ops == 1


def test_upmem_all_on_device():
    res = lower(_prog(), build_upmem())
    assert res.host_ops == 0
    softmax_ir = next(p for p in res.perf_ir if p["src"].kind == "softmax")
    assert len(softmax_ir["per_op"]) >= 3


def test_perf_ir_rollup_matches_sum():
    t = build_upmem()
    res = lower(_prog(), t)
    rollup = sum(p["cycles"] for p in res.perf_ir)
    assert rollup == res.total_cycles


def test_emitted_text_is_target_native():
    samsung = lower(_prog(), build_samsung())
    aim     = lower(_prog(), build_aim())
    upmem   = lower(_prog(), build_upmem())
    # Samsung emits PIMCmd(...) plus 1 host.call for softmax
    assert any(line.startswith("PIMCmd(") for line in samsung.emitted)
    assert any(line.startswith("host.call") for line in samsung.emitted)
    assert any(line.startswith("AiM ")     for line in aim.emitted)
    assert any(line.startswith("host.call") for line in aim.emitted)
    assert any("for(" in line for line in upmem.emitted)
    assert not any(line.startswith("host.call") for line in upmem.emitted)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall tests passed")
