# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 D3: LinearLayout promoted onto Placement; _bank_parity deleted.

The chosen swizzled `LinearLayout` is carried on `Placement.layout` and
consumed by codegen directly (the `range(stride)` fiber walk reads
`layout.size_of(fiber_axis)`), making the F2 algebra load-bearing instead of
a single Samsung swizzle generator. The two-class `_bank_parity` SymExpr
matcher is replaced by `_bank_fiber_class(idx, stride)`, byte-identical for
Samsung's stride-2 even/odd and generalizing to a `range(stride)` walk.

The full `banks_per_pim > 2` proof fixture is task 008; here we prove the
mechanism: the layout is carried, `_bank_fiber_class` generalizes, and
`_bank_parity` is gone.
"""

from __future__ import annotations

import inspect

import allo
from allo.spmw_autoschedule import Placement, _samsung_enumerate
import allo.spmw_codegen as cg
from allo.spmw_codegen import _bank_fiber_class, _parse_fiber_idx
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import UnitId

from _fixtures import build_samsung_target


def _mac_trace(k: int = 1024) -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="synthetic",
        matches=[MatchedOp(
            target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
            enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", str(k), 1)],
            operands=[
                OperandBinding(role="x", memref_name="local_W"),
                OperandBinding(role="y", memref_name="local_x"),
                OperandBinding(role="acc", memref_name="acc",
                               is_loop_carried=True),
            ],
            result_memref_name="acc", op_range=("%a", "%b"))])


# --------------------------------------------------------------------- #
# _bank_parity is deleted; _bank_fiber_class replaces it
# --------------------------------------------------------------------- #


def test_bank_parity_symbol_is_deleted():
    assert not hasattr(cg, "_bank_parity"), (
        "_bank_parity must be deleted (replaced by _bank_fiber_class)"
    )
    # And it must not appear in the codegen source path.
    src = inspect.getsource(cg)
    assert "def _bank_parity" not in src


def test_bank_fiber_class_byte_identical_for_stride_2():
    """The Samsung even/odd path: stride*pid + r with stride==2 maps to
    exactly EVEN_BANK (r=0) / ODD_BANK (r=1) -- byte-identical to the old
    two-class matcher."""
    pid = UnitId(level=1, unit=None)
    even = 2 * pid          # SymExpr(mul, 2, pid)
    odd = 2 * pid + 1       # SymExpr(add, mul, 1)
    assert _bank_fiber_class(even) == "EVEN_BANK"
    assert _bank_fiber_class(odd) == "ODD_BANK"
    # Stride supplied explicitly (from a carried layout) gives the same.
    assert _bank_fiber_class(even, stride=2) == "EVEN_BANK"
    assert _bank_fiber_class(odd, stride=2) == "ODD_BANK"


def test_bank_fiber_class_generalizes_to_stride_gt_2():
    """A wide-fiber index (stride*pid + r, stride>2) maps to a per-fiber
    BANK_<r> class -- the range(stride) generalization (the no-sim wide
    fixture in task 008 consumes these names)."""
    pid = UnitId(level=1, unit=None)
    assert _bank_fiber_class(4 * pid) == "BANK_0"
    assert _bank_fiber_class(4 * pid + 1) == "BANK_1"
    assert _bank_fiber_class(4 * pid + 2) == "BANK_2"
    assert _bank_fiber_class(4 * pid + 3) == "BANK_3"
    # Self-describing: stride read off the coefficient when not supplied.
    assert _parse_fiber_idx(4 * pid + 3) == (4, 3)


def test_bank_fiber_class_rejects_non_fiber_form():
    """A non `stride*pid + r` index returns None (the _opd hard-error
    signal the old _bank_parity preserved)."""
    assert _bank_fiber_class(7) is None
    pid = UnitId(level=1, unit=None)
    # remainder out of [0, stride) is not a valid fiber.
    assert _bank_fiber_class(2 * pid + 5, stride=2) is None


# --------------------------------------------------------------------- #
# The layout is carried on Placement and survives lever crossing
# --------------------------------------------------------------------- #


def test_enumerator_carries_swizzled_layout():
    target = build_samsung_target()
    cands = _samsung_enumerate(target, _mac_trace().matches)
    # Every Samsung candidate carries the swizzled LinearLayout.
    assert cands, "no candidates"
    for c in cands:
        assert isinstance(c.layout, LinearLayout), (c.mode, c.layout)
        # The segment (fiber) axis size IS the banks-per-pim stride (== 2).
        assert c.layout.size_of("tile") == 2


def test_layout_survives_autoschedule_to_codegen():
    """The carried layout reaches the compiled placement (through the
    enumerator -> lever crossing -> regalloc round-trip)."""
    target = build_samsung_target()
    trace = _mac_trace()
    compiled = allo.compile_for_target(target, trace)
    # compile_for_target stores the chosen Placement; it must carry the layout.
    layout_obj = compiled.layout
    placements = layout_obj if isinstance(layout_obj, list) else [layout_obj]
    assert any(
        isinstance(getattr(p, "layout", None), LinearLayout) for p in placements
    ), [getattr(p, "layout", None) for p in placements]


def test_codegen_reads_stride_from_carried_layout():
    """SamsungCtx._fiber_stride derives the stride from the carried layout's
    fiber axis -- the F2 object is load-bearing, not the index pattern."""
    target = build_samsung_target()
    ctx = cg.SamsungCtx(target)
    # A dual_fiber placement with a carried layout + fiber_axis.
    cands = _samsung_enumerate(target, _mac_trace().matches)
    dual = next(c for c in cands if c.mode.split("+", 1)[0] == "dual_fiber")
    ctx._active_placement = dual
    assert ctx._fiber_stride() == 2  # from layout.size_of("tile")


if __name__ == "__main__":
    test_bank_parity_symbol_is_deleted()
    test_bank_fiber_class_byte_identical_for_stride_2()
    test_bank_fiber_class_generalizes_to_stride_gt_2()
    test_bank_fiber_class_rejects_non_fiber_form()
    test_enumerator_carries_swizzled_layout()
    test_layout_survives_autoschedule_to_codegen()
    test_codegen_reads_stride_from_carried_layout()
    print("ALL PASSED")
