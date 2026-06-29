# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design 07 D3 (task 007): bit-serial operand width at the real APU v1 sites.

On the GSI APU (bit-serial SRAM compute) an op is computed one bit-plane at a
time, so its cycle cost scales with operand bit-width -- and a multiply's
per-bit work is far steeper than an add's (MICRO'25 Table 5: mul_u16=201 vs
add_u16=12, ~16x; mul_f16=77). The APU op-cost lambdas are now FUNCTIONS OF
`OpCostCtx.dtype_bits`, mirroring the table's shape (the `micro25_add_cost`
affine template), not its absolute magnitudes -- the claim is the THREADING
of width into the fused objective, not the table itself.

All STATIC (no simulator). What is proven:

  * width-absent (`dtype_bits=None`) -> today's constant (ADD=2, MUL=16,
    MAC=8) VERBATIM -- the byte-identical anchor (design 07 §D3.2).
  * each APU op cost is monotonic in `dtype_bits`.
  * MUL scales steeper than ADD per bit (the table's mul>>add shape).
  * the APU EXEC estimate moves monotonically with a precision change
    (s32 > s16 > s8), via the dtype the matcher stamps on `match.extra`.
  * a non-APU backend (no width threaded) is unchanged; the faithful APU
    corpus (no dtype stamped) is byte-identical.
"""
from __future__ import annotations

from allo.spmw_cost_model import ComposeCtx, OpCostCtx, combine, get_cost_model
from allo.spmw_cost_tables import APU_V1_FAITHFUL, APU_V1_BASE_BITS
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement

from _fixtures import build_apu_v1_target


# --------------------------------------------------------------------- #
# Per-op width law: anchor + monotonicity + mul-steeper-than-add
# --------------------------------------------------------------------- #


def test_width_absent_reproduces_todays_constant():
    """The D3.2 anchor: a width-absent OpCostCtx returns today's constant."""
    m = APU_V1_FAITHFUL
    assert m.op_cost("ADD", OpCostCtx("ADD")) == 2
    assert m.op_cost("MUL", OpCostCtx("MUL")) == 16
    assert m.op_cost("MAC", OpCostCtx("MAC")) == 8


def test_width_16_equals_constant():
    """At the base width (gvml_*_16) the width-aware law reproduces the
    quoted constant -- the anchor that keeps the faithful number frozen."""
    m = APU_V1_FAITHFUL
    for op, const in (("ADD", 2), ("MUL", 16), ("MAC", 8)):
        assert m.op_cost(op, OpCostCtx(op, dtype_bits=APU_V1_BASE_BITS)) == const


def test_each_op_monotonic_in_width():
    m = APU_V1_FAITHFUL
    for op in ("ADD", "MUL", "MAC"):
        c8 = m.op_cost(op, OpCostCtx(op, dtype_bits=8))
        c16 = m.op_cost(op, OpCostCtx(op, dtype_bits=16))
        c32 = m.op_cost(op, OpCostCtx(op, dtype_bits=32))
        assert c8 < c16 < c32, (op, c8, c16, c32)


def test_mul_scales_steeper_than_add():
    """The MICRO'25 Table 5 shape: a bit-serial MUL's per-bit work is far
    steeper than an ADD's. The marginal cost of doubling width is larger for
    MUL than ADD."""
    m = APU_V1_FAITHFUL
    add_slope = (m.op_cost("ADD", OpCostCtx("ADD", dtype_bits=32))
                 - m.op_cost("ADD", OpCostCtx("ADD", dtype_bits=8)))
    mul_slope = (m.op_cost("MUL", OpCostCtx("MUL", dtype_bits=32))
                 - m.op_cost("MUL", OpCostCtx("MUL", dtype_bits=8)))
    assert mul_slope > add_slope, (mul_slope, add_slope)


# --------------------------------------------------------------------- #
# The estimate moves with a precision change (the D3 demo)
# --------------------------------------------------------------------- #


def _gemv_trace(dtype_bits=None) -> MatchTrace:
    extra = {} if dtype_bits is None else {"dtype_bits": dtype_bits}
    return MatchTrace(
        target_name="apu_v1", module_name="gemv",
        matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0", work_id=(0,),
                enclosing_loops=[("%i", "0", "16", 1), ("%k", "0", "1024", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b"),
                extra=dict(extra),
            )
        ],
    )


def _exec_cycles(target, trace, mode="sv_lookup") -> int:
    model = get_cost_model("apu_v1", "faithful")
    res = model.compose(ComposeCtx(target, trace, Placement(mode=mode)))
    return res.cycles


def test_precision_change_moves_estimate_monotonically():
    """s8 < s16 < s32: a narrower datatype costs strictly fewer cycles on the
    bit-serial machine -- precision becomes a first-class cost-ranked lever
    (design 07 §D3.2). 16 outer iters x MAC(width)."""
    target = build_apu_v1_target()
    e8 = _exec_cycles(target, _gemv_trace(8))
    e16 = _exec_cycles(target, _gemv_trace(16))
    e32 = _exec_cycles(target, _gemv_trace(32))
    assert e8 < e16 < e32, (e8, e16, e32)
    # exact: 16 outer iters * MAC(width); MAC(8)=4, MAC(16)=8, MAC(32)=16.
    assert e8 == 16 * 4
    assert e16 == 16 * 8
    assert e32 == 16 * 16


def test_sv_raw_mac_is_width_aware_mul_plus_add():
    """The SV (raw MUL+ADD) MAC path is also width-aware: MAC = MUL(w)+ADD(w)."""
    target = build_apu_v1_target()
    m = get_cost_model("apu_v1", "faithful")
    for w in (8, 16, 32):
        e = _exec_cycles(target, _gemv_trace(w), mode="sv")
        per = (m.op_cost("MUL", OpCostCtx("MUL", dtype_bits=w))
               + m.op_cost("ADD", OpCostCtx("ADD", dtype_bits=w)))
        assert e == 16 * per, (w, e, per)


# --------------------------------------------------------------------- #
# Byte-identical anchor: no dtype stamped -> today's APU number
# --------------------------------------------------------------------- #


def test_apu_faithful_corpus_byte_identical_without_width():
    """A trace with NO dtype stamped -> the APU exec is exactly today's
    constant-cost number (MAC=8), and the fold reproduces .cycles."""
    target = build_apu_v1_target()
    trace = _gemv_trace(dtype_bits=None)
    e = _exec_cycles(target, trace)
    assert e == 16 * 8                      # pre-D3 constant MAC=8
    model = get_cost_model("apu_v1", "faithful")
    res = model.compose(ComposeCtx(target, trace, Placement(mode="sv_lookup")))
    assert combine(list(res.phases), overlap=False) == res.cycles
