"""APU v2 (G2 L1 simulator) end-to-end demo.

Target: `gsi_apu_v2_g2`. Lowers an elementwise vector add against the APU v2
target description, emits a G2Gtml .cc, builds and runs it inside the
`gsi-g2-l1sim` docker image.

Limitation: the l1_sim backend is a *functional* simulator — no cycle counts.
Performance comparisons for APU v2 would require the apu_sim backend or real
G2 hardware, neither of which is available on this server.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from allo.pim.backends import build_apu_v2
from allo.pim.ops import SrcProgram, Add, Mul
from allo.pim.lowering import lower
from allo.pim.runtime.apu_v2_codegen import gen_apu_v2_cc, build_and_run_apu_v2


def _run(op_kind: str):
    print(f"\n=== APU v2 / G2 L1 sim — elementwise {op_kind} ===")
    target = build_apu_v2()
    op_cls = {"add": Add, "mul": Mul}[op_kind]
    prog = SrcProgram().add(
        op_cls(shape=(4, 4096), inputs=("a", "b"), output="c"))
    res = lower(prog, target)
    print(res.summary())
    for line in res.emitted:
        print(f"  {line}")

    # 2) codegen the .cc file
    cc_path = Path(tempfile.mkdtemp(prefix="apu_v2_")) / "prog.cc"
    gen_apu_v2_cc(str(cc_path), op=op_kind)
    print(f"\nemitted: {cc_path}")

    # 3) build + run in the l1_sim container
    r = build_and_run_apu_v2(str(cc_path))
    passed = r["returncode"] == 0 and "PASSED" in (r.get("stdout") or "")
    print(f"build/run rc={r['returncode']}  PASSED={passed}")
    for line in (r.get("stdout") or "").splitlines()[-6:]:
        print(f"  {line}")
    if not passed:
        print("--- stderr tail ---")
        for line in (r.get("stderr") or "").splitlines()[-15:]:
            print(f"  {line}")
    return passed


if __name__ == "__main__":
    ok_add = _run("add")
    ok_mul = _run("mul")
    if ok_add and ok_mul:
        print("\nOK: APU v2 add + mul lowered, compiled, and ran on l1_sim.")
        sys.exit(0)
    print("\nFAIL: at least one APU v2 kernel did not pass.")
    sys.exit(1)
