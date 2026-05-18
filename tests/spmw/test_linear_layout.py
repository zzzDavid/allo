# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPMW linear-layout algebra.

Covers per spec 013 §G:
  1. identity_apply           — identity layout evaluates the input verbatim
  2. compose_is_matrix_mul    — `apply(self) -> apply(other)` matches compose
  3. product_is_block_diagonal— product of identities is bitwise-disjoint
  4. invert_roundtrip         — `L.compose(L.invert())` is identity
  5. optimal_swizzle_samsung  — base + segment yields tile→bank-bit-0 swizzle
  6. materialise_samsung_bank — fixed tile=0 yields `idx = 2*pid`
  7. autoschedule_picks_canonical_samsung — autoschedule produces the same
                                          placement the old enumerator's
                                          argmin would have picked.
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import autoschedule
from allo.spmw_codegen import _bank_parity
from allo.spmw_linear_layout import LinearLayout, materialise_handle
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import MemoryRef, Register, SymExpr, UnitId

from _fixtures import build_samsung_target


# --------------------------------------------------------------------- #
# Core F2 algebra
# --------------------------------------------------------------------- #


def test_identity_apply():
    L = LinearLayout.identity({"a": 4})
    assert L.out_dims == ("a",)
    assert L.out_sizes == (4,)
    assert L.apply(a=3) == (3,)
    assert L.apply(a=0) == (0,)
    assert L.apply(a=2) == (2,)


def test_compose_is_matrix_multiply():
    # Two layouts (a, b) -> (p, q): self maps a->p,b->q; other rotates.
    self_L = LinearLayout(
        bases={"a": [(1, 0), (2, 0)], "b": [(0, 1), (0, 2), (0, 4)]},
        out_dims=("p", "q"),
        out_sizes=(4, 8),
    )
    # other: (p, q) -> (q, p) (swap output dims)
    other_L = LinearLayout(
        bases={"p": [(0, 1), (0, 2)], "q": [(1, 0), (2, 0), (4, 0)]},
        out_dims=("q", "p"),
        out_sizes=(8, 4),
    )
    composed = self_L.compose(other_L)
    assert composed.out_dims == ("q", "p")

    for a in range(4):
        for b in range(8):
            seq_intermediate = self_L.apply(a=a, b=b)
            kw = dict(zip(self_L.out_dims, seq_intermediate))
            via_sequence = other_L.apply(**kw)
            via_compose = composed.apply(a=a, b=b)
            assert via_compose == via_sequence, (a, b)


def test_product_is_block_diagonal():
    A = LinearLayout.identity({"a": 4})
    B = LinearLayout.identity({"b": 8})
    P = A.product(B)
    assert P.out_dims == ("a", "b")
    assert P.out_sizes == (4, 8)
    for a in range(4):
        for b in range(8):
            assert P.apply(a=a, b=b) == (a, b)


def test_invert_roundtrip():
    # 3-bit invertible layout with an off-diagonal swap-style entry.
    L = LinearLayout(
        bases={
            "x": [(1, 0, 0)],
            "y": [(0, 1, 0)],
            "z": [(1, 1, 1)],
        },
        out_dims=("p", "q", "r"),
        out_sizes=(2, 2, 2),
    )
    Linv = L.invert()
    # L composed with its inverse should be identity over its own dims.
    RT = L.compose(Linv)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                assert RT.apply(x=x, y=y, z=z) == (x, y, z)


# --------------------------------------------------------------------- #
# Samsung-specific algebra: swizzle + materialisation
# --------------------------------------------------------------------- #


def test_optimal_swizzle_samsung_gemv():
    base = LinearLayout.identity(
        {"grf": 8, "bank": 16, "tile": 2},
        out_dims=("grf", "bank"),
    )

    # Base layout: tile contributes nothing — conflict.
    assert base.bases["tile"] == [(0, 0)]
    assert not base.describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    )

    swizzled = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )
    # The XOR-augmentation adds tile -> bank bit 0 (mask 1).
    assert swizzled.bases["tile"] == [(0, 1)]
    assert swizzled.describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    )


def test_materialise_samsung_bank_handle():
    target = build_samsung_target()
    base = LinearLayout.identity(
        {"grf": 8, "bank": 16, "tile": 2},
        out_dims=("grf", "bank"),
    )
    swizzled = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )

    pid = UnitId(level=1, unit=None)
    h_even = materialise_handle(
        swizzled,
        target=target,
        out_dim="bank",
        fixed={"grf": 0, "tile": 0},
        symbol_table={"bank": 2 * pid},
    )
    assert isinstance(h_even, MemoryRef)
    assert h_even.memory is target.banks
    # The idx must look like `2 * pid` so `_bank_parity` lowers it to
    # EVEN_BANK — that is the canonical Samsung GEMV bank handle.
    assert _bank_parity(h_even.idx) == "EVEN_BANK"

    # Toggling tile yields ODD_BANK.
    h_odd = materialise_handle(
        swizzled,
        target=target,
        out_dim="bank",
        fixed={"grf": 0, "tile": 1},
        symbol_table={"bank": 2 * pid},
    )
    assert _bank_parity(h_odd.idx) == "ODD_BANK"


# --------------------------------------------------------------------- #
# Autoschedule integration — single-element candidate, same placement.
# --------------------------------------------------------------------- #


def _synthetic_mac_trace() -> MatchTrace:
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
                    ("%arg1", "0", "1024", 1),
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


def test_autoschedule_picks_canonical_samsung_layout():
    target = build_samsung_target()
    trace = _synthetic_mac_trace()
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    placement = layouts[0].placements

    # x lands on grf_a (Register); y is a MemoryRef whose idx is `2*pid`
    # (EVEN_BANK); acc is grf_b (MAC's dst constraint).
    x_handle = placement["local_W"]
    y_handle = placement["local_x"]
    acc_handle = placement["acc"]

    assert isinstance(x_handle, Register) and x_handle.name == "grf_a"
    assert isinstance(acc_handle, Register) and acc_handle.name == "grf_b"
    assert isinstance(y_handle, MemoryRef)
    assert _bank_parity(y_handle.idx) == "EVEN_BANK"


def test_linear_layout_exported_from_allo():
    """`allo.LinearLayout` must be reachable via the public package
    re-export (see __init__.py).
    """
    assert allo.LinearLayout is LinearLayout
    assert allo.materialise_handle is materialise_handle


if __name__ == "__main__":
    test_identity_apply()
    test_compose_is_matrix_multiply()
    test_product_is_block_diagonal()
    test_invert_roundtrip()
    test_optimal_swizzle_samsung_gemv()
    test_materialise_samsung_bank_handle()
    test_autoschedule_picks_canonical_samsung_layout()
    test_linear_layout_exported_from_allo()
    print("ALL PASSED")
