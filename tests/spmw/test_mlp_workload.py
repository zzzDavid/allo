# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for the two-layer MLP workload (Task 008).

Asserts that:
  1. The MLP `@allo.work` kernel compiles through Allo's frontend to MLIR.
  2. `allo.match_workload` against the Samsung HBM-PIM target spec returns
     at least two distinct `MatchedOp` entries — one per layer's MAC site.

End-to-end simulator runs are the verifier's job; this test stays at the
MLIR-matcher level so it runs in the same ~6 minute budget as the other
SPMW tests.
"""

from __future__ import annotations

import allo
from allo.spmw_match import MatchTrace

from _fixtures import build_mlp_workload, build_samsung_target


def test_mlp_workload_compiles_and_matches_two_macs():
    mlp_top = build_mlp_workload()
    schedule = allo.customize(mlp_top, enable_tensor=False)

    # MLIR module exists and is non-empty.
    module_str = str(schedule.module)
    assert module_str, "Allo frontend produced an empty MLIR module"
    assert "mlp" in module_str, (
        "expected the work-item function name to appear in the MLIR module"
    )

    target = build_samsung_target()
    trace: MatchTrace = allo.match_workload(target, schedule.module)

    assert trace.target_name == "samsung_hbm_pim"

    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, (
        f"expected at least 2 MAC matches (one per MLP layer), got {len(macs)}"
    )

    # The two MAC sites must come from different work-item funcs —
    # the layers are split into `mlp_layer1` and `mlp_layer2` so each
    # gets its own SSA-name scope (see comment in `_fixtures.py`).
    funcs = {m.func_name for m in macs}
    assert len(funcs) >= 2, (
        f"expected MACs from at least 2 distinct functions; got {funcs!r}"
    )
    assert any(name.startswith("mlp_layer1") for name in funcs), funcs
    assert any(name.startswith("mlp_layer2") for name in funcs), funcs

    # Each MAC's accumulator must be loop-carried and bind the W/x or
    # W/h operands as the two non-carried inputs.
    for m in macs:
        roles = [b.role for b in m.operands]
        assert roles == ["x", "y", "acc"], roles
        carried = [b for b in m.operands if b.is_loop_carried]
        assert len(carried) == 1 and carried[0].role == "acc", m.operands

    # Layer 1 MAC must consume W1 + x; layer 2 MAC must consume W2 + h.
    by_func = {m.func_name: m for m in macs}
    l1 = next(m for f, m in by_func.items() if f.startswith("mlp_layer1"))
    l2 = next(m for f, m in by_func.items() if f.startswith("mlp_layer2"))
    assert l1.operands[0].memref_name == "local_W1", l1.operands[0]
    assert l1.operands[1].memref_name == "local_x", l1.operands[1]
    assert l2.operands[0].memref_name == "local_W2", l2.operands[0]
    assert l2.operands[1].memref_name == "local_h", l2.operands[1]


if __name__ == "__main__":
    test_mlp_workload_compiles_and_matches_two_macs()
    print("ALL PASSED")
