# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 D3 acceptance: banks_per_pim > 2 materializes and emits.

The load-bearing proof that the `range(stride)` fiber walk in
`_bank_fiber_class` generalizes beyond Samsung's factor-2 even/odd swizzle. On
a Mortise-WIDE structural target (16 banks, pim fanout 4 -> banks_per_pim == 4):

  - the promoted F2 `LinearLayout` materializes FOUR fiber handles
    (`size_of("tile") == 4`), not two;
  - codegen emits a 4-way `(MAC, JUMP)` fiber walk classifying the per-fiber
    banks as `BANK_0..BANK_3` -- a layout class that was UN-EMITTABLE before
    D3 (the deleted two-class `_bank_parity` only knew `EVEN_BANK`/`ODD_BANK`);
  - the cost model ranks the wide-fiber candidate (it is scorable);
  - and the Samsung even/odd path stays byte-identical (the stride-2 case of
    the same walk).

No simulator runs this -- PIMSimulator's `PIMOpdType` enum has only the two
Samsung bank names, and no real silicon has banks_per_pim > 2. The claim is a
property of the codegen + F2 layout algebra, proven on a structural fixture.
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import _samsung_enumerate
from allo.spmw_codegen import PIMCmd, compile_for_target
import allo.spmw_codegen as cg
from allo.spmw_cost import get_cost
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _mortise_wide_target import build_mortise_wide_target
from _fixtures import build_samsung_target


def _gemv_match(target_name: str) -> MatchedOp:
    return MatchedOp(
        target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
        enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", "1024", 1)],
        operands=[
            OperandBinding(role="x", memref_name="local_W"),
            OperandBinding(role="y", memref_name="local_x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc", op_range=("%a", "%b"),
    )


def _dual_fiber(cands):
    return next(c for c in cands if c.mode.split("+", 1)[0] == "dual_fiber")


# --------------------------------------------------------------------- #
# The banks_per_pim == 4 proof
# --------------------------------------------------------------------- #


def test_wide_fibers_materialize_and_emit():
    """The acceptance proof: 4 fibers materialize through the carried layout
    and codegen emits a 4-way range(stride) walk with BANK_0..BANK_3, no
    _bank_parity in the source path."""
    target = build_mortise_wide_target()
    assert target.banks.banks == 16
    matches = [_gemv_match("mortise_wide")]
    trace = MatchTrace(
        target_name="mortise_wide", module_name="syn", matches=matches
    )

    cands = _samsung_enumerate(target, matches)
    dual = _dual_fiber(cands)

    # (1) The promoted LinearLayout materializes FOUR fibers (not two).
    assert dual.layout.size_of("tile") == 4, dual.layout.bases
    fibers = dual.extra["fibers"]
    assert dual.extra["n_fibers"] == len(fibers) == 4, dual.extra
    # Each fiber idx is `4*pid + r` for r in range(4) (banks_per_pim stride 4).
    classes = [cg._bank_fiber_class(f.idx) for f in fibers]
    assert classes == ["BANK_0", "BANK_1", "BANK_2", "BANK_3"], classes

    # (2) Codegen emits a 4-way (MAC, JUMP) fiber walk; the four MACs read the
    # four per-fiber bank classes. Un-emittable before D3.
    compiled = compile_for_target(target, trace, layout=dual)
    macs = [c for c in compiled.cmds if isinstance(c, PIMCmd) and c.type_ == "MAC"]
    jumps = [c for c in compiled.cmds if isinstance(c, PIMCmd) and c.type_ == "JUMP"]
    assert len(macs) == 4, [c.type_ for c in compiled.cmds]
    assert len(jumps) == 4, [c.type_ for c in compiled.cmds]
    assert [m.src1_ for m in macs] == ["BANK_0", "BANK_1", "BANK_2", "BANK_3"], (
        [m.src1_ for m in macs]
    )

    # (3) No _bank_parity in the source path (deleted in D3).
    assert not hasattr(cg, "_bank_parity")

    # (4) The cost model ranks the wide-fiber candidate (it is scorable, and
    # the 4-fiber dual_fiber is far cheaper than the grf-staged unroll).
    cost_fn = get_cost("kernel_cycles", target)
    kc_dual = cost_fn(trace, dual)
    grf_staged = next(c for c in cands if c.mode.split("+", 1)[0] == "grf_staged")
    kc_grf = cost_fn(trace, grf_staged)
    assert kc_dual < kc_grf, (kc_dual, kc_grf)


def test_samsung_even_odd_still_byte_identical():
    """The stride-2 Samsung case of the same range(stride) walk: exactly two
    fibers, EVEN_BANK / ODD_BANK -- byte-identical to the pre-D3 path."""
    target = build_samsung_target()
    matches = [_gemv_match("samsung_hbm_pim")]
    trace = MatchTrace(
        target_name="samsung_hbm_pim", module_name="syn", matches=matches
    )
    cands = _samsung_enumerate(target, matches)
    dual = _dual_fiber(cands)

    assert dual.layout.size_of("tile") == 2
    fibers = dual.extra["fibers"]
    assert dual.extra["n_fibers"] == len(fibers) == 2
    assert [cg._bank_fiber_class(f.idx) for f in fibers] == ["EVEN_BANK", "ODD_BANK"]

    compiled = compile_for_target(target, trace, layout=dual)
    macs = [c for c in compiled.cmds if isinstance(c, PIMCmd) and c.type_ == "MAC"]
    assert len(macs) == 2
    assert [m.src1_ for m in macs] == ["EVEN_BANK", "ODD_BANK"]


if __name__ == "__main__":
    test_wide_fibers_materialize_and_emit()
    test_samsung_even_odd_still_byte_identical()
    print("ALL PASSED")
