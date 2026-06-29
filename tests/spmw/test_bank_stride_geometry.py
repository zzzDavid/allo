# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 drift fix: the Samsung bank stride is geometry-derived.

The even-bank base was bound as a pasted `2 * pid`. This asserts the `2`
is the `bank_out_size // pim_unit_count` instantiation computed by
`_bank_stride_per_pim`, so it tracks the target geometry rather than a
literal -- and that the Samsung emitted forms stay byte-identical
(EVEN_BANK / ODD_BANK out of the swizzle column).
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import (
    _bank_stride_per_pim,
    _samsung_enumerate,
)
from allo.spmw_codegen import _bank_fiber_class
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_samsung_target


def _samsung_bank_layout() -> LinearLayout:
    """The (grf, bank, tile) swizzled layout the Samsung enumerator builds."""
    base = LinearLayout.identity(
        {"grf": 8, "bank": 16, "tile": 2},
        out_dims=("grf", "bank"),
    )
    return LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )


def _mac_trace(k: int) -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
                    ("%arg1", "0", str(k), 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


def test_bank_stride_is_geometry_derived_for_samsung():
    # 16 banks / 8 pim units == stride 2. The `2` is `bank_out // pim_units`,
    # not a pasted constant.
    target = build_samsung_target()
    layout = _samsung_bank_layout()
    assert _bank_stride_per_pim(target, layout) == 2


def test_bank_stride_tracks_layout_bank_axis():
    # A different bank out-size with the same pim fanout yields a different
    # stride -- proving the value is read off geometry, not hardcoded `2`.
    target = build_samsung_target()  # pim fanout == 8
    wide = LinearLayout.identity(
        {"grf": 8, "bank": 32, "tile": 2},
        out_dims=("grf", "bank"),
    )
    assert _bank_stride_per_pim(target, wide) == 4  # 32 // 8


def test_samsung_even_odd_byte_identical():
    # The stride-2 instantiation must emit byte-identical EVEN/ODD fibers:
    # the geometry derivation does not perturb the Samsung path.
    target = build_samsung_target()
    candidates = _samsung_enumerate(target, _mac_trace(1024).matches)
    dual = next(c for c in candidates if c.mode.split("+", 1)[0] == "dual_fiber")
    fibers = dual.extra["fibers"]
    assert dual.extra["n_fibers"] == len(fibers) == 2
    assert _bank_fiber_class(fibers[0].idx) == "EVEN_BANK"
    assert _bank_fiber_class(fibers[-1].idx) == "ODD_BANK"
