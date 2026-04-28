# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``allo.match_workload`` against a GEMV workload
mapped on the report-16 Samsung HBM-PIM target.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_match import MatchTrace


M, K = 4096, 1024
ROWS = M // (16 * 8)  # 32 rows per work-item


# --------------------------------------------------------------------- #
# Target spec — the full Samsung HBM-PIM target from /tmp/test_step_c.py
# --------------------------------------------------------------------- #


def build_samsung_target():
    @allo.target("samsung_hbm_pim")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.memory(banks=16, rows=16384, cols=128, width=8, name="banks")

            @allo.unit(mapping=[8])
            def pim():
                _, pid = allo.get_uid()
                even_bank = banks[2 * pid]
                odd_bank = banks[2 * pid + 1]
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                allo.move("LD_A", src=even_bank, dst=grf_a)
                allo.move("LD_B", src=odd_bank, dst=grf_b)
                allo.move("ST_A", src=grf_a, dst=even_bank)
                allo.move("ST_B", src=grf_b, dst=odd_bank)

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                )

    return device


# --------------------------------------------------------------------- #
# Workload — the affine-loop GEMV from the report
# --------------------------------------------------------------------- #


@_df_region()
def gemv_top(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_commutative_matching_units():
    """Direct unification check: `acc + x*y` and `x*y + acc` should both
    unify against the workload's `addf(load_acc, mulf(load_W, load_x))`.
    """
    from allo.spmw_match_engine import (
        WBinOp,
        WLoad,
        compile_op_pattern,
        _unify,
    )

    class _Fake:
        pass

    a = _Fake()
    a.fn = lambda x, y, acc: acc + x * y
    a.pattern = None
    pat_a = compile_op_pattern(a)

    b = _Fake()
    b.fn = lambda x, y, acc: x * y + acc
    b.pattern = None
    pat_b = compile_op_pattern(b)

    # Build a synthetic term identical in shape to the IR matcher's result.
    w_load = WLoad("local_W", "%21", ["%20", "%arg4"], "memref.load")
    x_load = WLoad("local_x", "%22", ["%arg4"], "affine.load")
    acc_load = WLoad("acc", "%24", [], "affine.load")
    mul = WBinOp("mul", w_load, x_load, "%23")
    term = WBinOp("add", acc_load, mul, "%25")

    bind_a: dict = {}
    assert _unify(pat_a.body, term, bind_a)
    assert isinstance(bind_a["acc"], WLoad) and bind_a["acc"].memref_name == "acc"

    bind_b: dict = {}
    assert _unify(pat_b.body, term, bind_b)
    assert isinstance(bind_b["acc"], WLoad) and bind_b["acc"].memref_name == "acc"


def test_pattern_compilation_caches_on_op():
    target = build_samsung_target()
    mac = target.op("MAC")
    pat = allo.compile_op_pattern(mac)
    assert pat.param_names == ["x", "y", "acc"]
    # body is `acc + x*y` -> PBinOp(add, PVar(acc), PBinOp(mul, PVar(x), PVar(y)))
    assert pat.body.op == "add"
    # cached
    assert mac.pattern is pat
    # MUL: `x * y`
    mul_pat = allo.compile_op_pattern(target.op("MUL"))
    assert mul_pat.body.op == "mul"


def test_match_workload_finds_macs():
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace: MatchTrace = allo.match_workload(target, schedule.module)

    assert trace.target_name == "samsung_hbm_pim"

    macs = trace.by_target_op("MAC")
    assert macs, "expected at least one MAC match in the GEMV workload"

    # Mapping is [16, 8] -> 128 work-item functions, one MAC site per function.
    expected_funcs = 16 * 8
    func_names = {m.func_name for m in macs}
    assert len(func_names) == expected_funcs, (
        f"expected MAC matches in {expected_funcs} funcs, got {len(func_names)}"
    )

    # work_id consistency
    for m in macs:
        assert m.func_name.startswith("gemv_")
        tail = m.func_name[len("gemv_"):]
        parts = tuple(int(p) for p in tail.split("_"))
        assert m.work_id == parts, f"{m.func_name} -> {m.work_id} vs {parts}"

    # Pick gemv_0_3 specifically
    by_wid = {m.work_id: m for m in macs}
    assert (0, 3) in by_wid, "missing gemv_0_3 MAC match"
    m03 = by_wid[(0, 3)]

    assert m03.target_op_name == "MAC"
    assert m03.func_name == "gemv_0_3"
    assert len(m03.operands) == 3
    roles = [b.role for b in m03.operands]
    assert roles == ["x", "y", "acc"], roles

    # x -> local_W, y -> local_x, acc -> acc
    assert m03.operands[0].memref_name == "local_W"
    assert m03.operands[1].memref_name == "local_x"
    assert m03.operands[2].memref_name == "acc"

    # acc must be flagged as loop-carried; the inputs must not be.
    assert m03.operands[2].is_loop_carried is True
    assert m03.operands[0].is_loop_carried is False
    assert m03.operands[1].is_loop_carried is False

    # The store target should be the accumulator.
    assert m03.result_memref_name == "acc"

    # enclosing_loops: outer i, inner k.
    assert len(m03.enclosing_loops) >= 2
    iv_names = [t[0] for t in m03.enclosing_loops]
    # We only assert the two loops captured in order; allo names them %argN.
    assert all(name.startswith("%arg") for name in iv_names)
    # i loop has upper bound 32, k loop has upper bound 1024.
    ubs = [t[2] for t in m03.enclosing_loops]
    assert any("32" in ub for ub in ubs), ubs
    assert any("1024" in ub for ub in ubs), ubs


def test_no_false_matches():
    """Loops without the MAC shape must not produce MAC matches."""
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    macs = trace.by_target_op("MAC")
    # Each func must have *exactly one* MAC, never more.
    by_func: dict[str, int] = {}
    for m in macs:
        by_func[m.func_name] = by_func.get(m.func_name, 0) + 1
    assert all(c == 1 for c in by_func.values()), by_func

    # Stand-alone MUL pattern shouldn't fire on the same MAC sites: the
    # workload writes to `acc` (a binop chain), not a pure mul-store. A
    # MUL site would need a store of an `arith.mulf` result; allo emits
    # `addf(load_acc, mulf(...))` then stores. So MUL count must be zero
    # in this workload.
    muls = trace.by_target_op("MUL")
    assert muls == [], f"unexpected MUL matches: {muls!r}"


if __name__ == "__main__":
    test_commutative_matching_units()
    test_pattern_compilation_caches_on_op()
    test_match_workload_finds_macs()
    test_no_false_matches()
    print("ALL PASSED")
