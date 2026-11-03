# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release gate G2: store coverage and accumulator initialization.

Every store of a matched kernel must be implemented (matched op, copy of a
match result, supported accumulator initializer, or a local consumed only by
matched ops); anything else is an UnsupportedWorkloadError.  Backends that zero
their accumulator reject a caller-carried accumulator.  Virtual backend only.
"""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_codegen import compile_for_target
from allo.spmw_match import UnsupportedWorkloadError
from allo.spmw_match_engine import match_workload

M, K = 128, 16


@_df_region()
def gemv_small(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def renamed_small(A: fp16[M, K], v: fp16[K], out: fp16[M]):
    @allo.work(mapping=[16, 8], args=[A, v, out])
    def matvec(mat: fp16[M, K], vec: fp16[K], res: fp16[M]):
        g, l = allo.get_wid()
        r = g * 8 + l
        s: fp16 = 0
        for j in range(K):
            s += mat[r, j] * vec[j]
        res[r] = s


@_df_region()
def gemv_half_init(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0.5
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def gemv_init_in_loop(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc = 0
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def gemv_div(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc / local_x[0]


@_df_region()
def gemv_sub_after_copy(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc
        local_y[row0] = local_y[row0] - local_x[0]


@_df_region()
def gemv_carried(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        for k in range(K):
            local_y[row0] += local_W[row0, k] * local_x[k]


def _compile(workload):
    return allo.compile(
        workload, build_samsung_target(), samsung_cost, backend="virtual"
    )


def _paper_gemv():
    path = Path(__file__).resolve().parent / "samsung_hbm_pim" / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("match_coverage_samsung_gemv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"][0]


def _classification(trace):
    """Rename-invariant summary: categories per store and per-match facts."""
    stores = Counter(store.category for store in trace.coverage.stores)
    matches = Counter(
        (
            match.target_op_name,
            match.extra.get("accumulator_init"),
            tuple(index for _name, index in match.extra.get("output_sinks", ())),
        )
        for match in trace.matches
    )
    return stores, matches


def test_paper_gemv_zero_init_and_output_sink():
    compiled = _compile(_paper_gemv())
    trace = compiled.trace
    assert trace.coverage.errors == []
    assert {match.target_op_name for match in trace.matches} == {"MAC"}
    assert {match.extra["accumulator_init"] for match in trace.matches} == {"zero"}
    sinks = {sink for match in trace.matches for sink in match.extra["output_sinks"]}
    assert {name for name, _index in sinks} == {"local_y"}
    # One distinct output row base per work item.
    assert len(sinks) == len(trace.matches) == 128
    assert Counter(store.category for store in trace.coverage.stores) == {
        "local": 128,  # row0 index precomputation
        "accumulator_init": 128,  # acc = 0
        "matched": 128,  # acc += W * x
        "copy": 128,  # y[row0 + i] = acc
    }
    assert compiled.estimate().cycles > 0


def test_small_gemv_sinks_name_output_rows():
    trace = _compile(gemv_small).trace
    by_work = {match.func_name: match for match in trace.matches}
    assert by_work["gemv_0_0"].extra["output_sinks"] == [("local_y", "0")]
    assert by_work["gemv_1_3"].extra["output_sinks"] == [("local_y", "11")]


def test_rename_only_variant_has_same_classification():
    assert _classification(_compile(gemv_small).trace) == _classification(
        _compile(renamed_small).trace
    )
    sinks = {
        name
        for match in _compile(renamed_small).trace.matches
        for name, _index in match.extra["output_sinks"]
    }
    assert sinks == {"res"}


def test_nonzero_accumulator_initializer_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_half_init)
    message = str(info.value)
    assert "initializer" in message
    assert "nonzero constant 0.5" in message
    assert "accumulator 'acc' of MAC" in message
    assert "literal 0 before the reduction loop" in message


def test_initializer_inside_reduction_loop_is_rejected():
    with pytest.raises(UnsupportedWorkloadError, match="inside the reduction loop"):
        _compile(gemv_init_in_loop)


def test_unmatched_division_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_div)
    message = str(info.value)
    assert "div(acc[], local_x[0])" in message
    assert "arith.divf" in message
    assert "'local_y[0]'" in message
    assert "function 'gemv_0_0'" in message
    assert "loop nest [] (no enclosing loop)" in message
    assert "Target ops considered: ADD, MAC, MAD, MUL, RELU" in message


def test_unmatched_update_after_copy_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_sub_after_copy)
    assert "sub(local_y[0], local_x[0])" in str(info.value)
    assert "arith.subf" in str(info.value)


def test_carried_accumulator_selects_the_ordered_program_on_samsung():
    """``y[i] += W[i,k]*x[k]`` keeps the caller's initial value: the matcher
    records ``accumulator_init == "carried"`` and Samsung lowers it to the
    ordered TEMPORAL program, whose conductor loads the initial accumulator.
    The REDUCE programs (which zero GRF_B) are pruned for it."""
    target = build_samsung_target()
    trace = match_workload(
        target, allo.customize(gemv_carried, enable_tensor=False).module
    )
    assert trace.coverage.errors == []
    assert {match.extra["accumulator_init"] for match in trace.matches} == {"carried"}
    # The store itself is the output sink of a carried accumulation.
    assert all(match.extra["output_sinks"] for match in trace.matches)
    compiled = _compile(gemv_carried)
    layouts = compiled.compiled.layout
    layouts = layouts if isinstance(layouts, list) else [layouts]
    assert all(layout.mode.startswith("temporal") for layout in layouts)


def test_carried_accumulator_rejected_where_the_device_zeroes_it():
    """AiM's MAC accumulates in a bias-initialized register the lowering
    always zeroes, so a carried accumulator is a finite error there."""
    from allo.pim.costs import aim_cost
    from allo.pim.targets import build_aim_target

    with pytest.raises(NotImplementedError) as info:
        allo.compile(
            gemv_carried,
            build_aim_target(),
            aim_cost,
            backend="virtual",
            reduction_order="reassociate",
        )
    message = str(info.value)
    assert "accumulator_init='carried'" in message or "carried" in message


def test_compile_for_target_rechecks_an_unchecked_trace():
    target = build_samsung_target()
    module = allo.customize(gemv_div, enable_tensor=False).module
    trace = match_workload(target, module)
    trace.coverage = None
    with pytest.raises(UnsupportedWorkloadError, match="arith.divf"):
        compile_for_target(target, trace, backend="virtual", module=module)


# --------------------------------------------------------------------------- #
# Conversions and conditional copies (release review, round 8)
# --------------------------------------------------------------------------- #

from allo.spmw_match_engine import cast_is_exact


def test_cast_exactness_rules():
    assert cast_is_exact("arith.extf", "f16", "f32")
    assert cast_is_exact("arith.extsi", "i8", "i32")
    assert cast_is_exact("arith.sitofp", "i8", "f16")  # 7 magnitude bits <= 11
    assert cast_is_exact("arith.uitofp", "i24", "f32")
    assert cast_is_exact("arith.index_cast", "i65", "index")  # address formation
    assert not cast_is_exact("arith.truncf", "f32", "f16")
    assert not cast_is_exact("arith.trunci", "i32", "i8")
    assert not cast_is_exact("arith.fptosi", "f16", "i32")
    assert not cast_is_exact("arith.sitofp", "i32", "f16")  # 31 magnitude bits > 11
    assert not cast_is_exact("arith.sitofp", "i32", "f32")  # 31 > 24
    assert cast_is_exact("arith.sitofp", "i25", "f32")  # 24 magnitude bits


@_df_region()
def gemv_rounded_output(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
        out[row] = int(acc)


@_df_region()
def gemv_guarded_output(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
        if v[0] > 0:
            out[row] = acc


@_df_region()
def gemv_copy_inside_reduction(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
            out[row] = acc


def test_value_changing_conversion_on_the_output_is_rejected():
    """``out[row] = int(acc)`` (fptosi then sitofp) must not compile to a plain
    copy of ``acc``: the reviewer's probe returned 0.25 where the source
    produces 0. The conversion is a computation no target op declares."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_rounded_output)
    message = str(info.value)
    assert "arith.fptosi<f16->i32>" in message
    assert "arith.sitofp<i32->f16>" in message
    assert "matches no target op" in message


def test_conditional_output_copy_is_rejected():
    """``if v[0] > 0: out[row] = acc`` must not become an unconditional store:
    the reviewer's probe overwrote a sentinel when the guard was false."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_guarded_output)
    message = str(info.value)
    assert "different guard than its producer" in message
    assert "conditional output copy" in message


def test_copy_inside_the_reduction_loop_is_rejected():
    """A copy that runs once per reduction step is not a once-per-element
    output copy; the lowering writes the final accumulator only."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_copy_inside_reduction)
    assert "exactly once per produced element" in str(info.value)


def test_exact_constant_initializer_still_matches():
    """The frontend spells ``acc: fp16 = 0`` as sitofp(0); the constant is
    representable, so the initializer is still recognized as zero."""
    trace = match_workload(
        build_samsung_target(),
        allo.customize(_paper_gemv(), enable_tensor=False).module,
    )
    assert trace.coverage.errors == []
    assert {m.extra["accumulator_init"] for m in trace.matches} == {"zero"}


# --------------------------------------------------------------------------- #
# Integer and float narrowing at stores (release review, round 8)
# --------------------------------------------------------------------------- #

from allo.ir.types import bfloat16 as bf16, int4, int15, int16, int32, int64
from allo.spmw_match import select, tanh  # pattern lambdas are parsed by name
from allo.spmw_match_engine import require_match_coverage


def _int_target(*names):
    ops = {
        "MAC": dict(fn=lambda a, b, acc: acc + a * b, accumulates=True),
        "ADD": dict(fn=lambda a, b: a + b),
        "ARGMIN_IDX": dict(
            fn=lambda x, best, i, acc: select(x < best, i, acc),
            accumulates=True,
            iv_params=("i",),
        ),
        "AF": dict(fn=lambda x: 0.5 * tanh(0.5 * x) + 0.5),
        # the same activation as a lookup table: one rounding at the output
        "AF_LUT": dict(fn=lambda x: 0.5 * tanh(0.5 * x) + 0.5, intermediate="f32"),
        "MUL": dict(fn=lambda a, b: a * b),
        "HALF": dict(fn=lambda x: 0.5 * x),
    }

    @allo.target("narrowing_" + "_".join(n.lower() for n in names))
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                for name in names:
                    allo.op(name, src=r, dst=r, **ops[name])

    return t


def _trace(kernel, *names):
    return match_workload(
        _int_target(*names), allo.customize(kernel, enable_tensor=False).module
    )


def gemm_int15(A: int4[8, 16], B: int4[16, 8], C: int15[8, 8]):
    for i in range(8):
        for j in range(8):
            acc: int15 = 0
            for k in range(16):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_widened_integer_accumulation_matches_in_the_stored_width():
    """The frontend computes ``i4 * i4`` in i8 and the sum in i16, then
    truncates to the i15 accumulator.  Truncation distributes over the ring
    ops and every leaf is at most 15 bits wide, so the stored value is the
    i15 evaluation of ``acc + A * B``: the pattern matches without any cast."""
    trace = _trace(gemm_int15, "MAC")
    assert trace.coverage.errors == []
    (mac,) = trace.matches
    assert mac.extra["accumulator_init"] == "zero"


def gemm_wide_acc_fits(A: int4[8, 16], B: int4[16, 8], C: int15[8, 8]):
    for i in range(8):
        for j in range(8):
            acc: int16 = 0
            for k in range(16):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def gemm_wide_acc_overflows(A: int16[8, 16], B: int16[16, 8], C: int16[8, 8]):
    for i in range(8):
        for j in range(8):
            acc: int64 = 0
            for k in range(16):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_truncating_copy_needs_a_range_proof():
    """``C = acc`` with a wider accumulator truncates at the store.  Sixteen
    products of two int4 values lie in [-896, 1024], which int15 holds, so
    the copy is exact and accepted; sixteen int16 products can reach 2^34,
    so the same copy into an int16 output is refused with the range."""
    exact = _trace(gemm_wide_acc_fits, "MAC")
    assert exact.coverage.errors == []
    assert [s.category for s in exact.coverage.stores].count("copy") == 1

    with pytest.raises(UnsupportedWorkloadError) as info:
        require_match_coverage(_trace(gemm_wide_acc_overflows, "MAC"))
    message = str(info.value)
    assert "through a truncation" in message
    assert "[-17179344896, 17179869184]" in message  # 16 products of two int16
    assert "i16 cannot hold" in message


def argmin_rows(D: int64[12, 4], a: int32[12]):
    for p in range(12):
        best: int64 = D[p, 0]
        idx: int32 = 0
        for c in range(4):
            idx = c if D[p, c] < best else idx
            best = D[p, c] if D[p, c] < best else best
        a[p] = idx


def test_loop_index_narrowing_is_proven_from_the_loop_bounds():
    """``idx = c`` casts the index loop variable to int32; the loop runs
    over [0, 4), so the cast cannot change the value and the pattern binds
    the loop variable."""
    trace = _trace(argmin_rows, "ARGMIN_IDX")
    assert [m.target_op_name for m in trace.matches] == ["ARGMIN_IDX"]
    assert trace.matches[0].extra["iv_params"] == {"i": 1}


def activation_bf16(acc: bf16[64], sg: bf16[64]):
    for i in range(64):
        sg[i] = 0.5 * allo.tanh(0.5 * acc[i]) + 0.5


def activation_bf16_f32_constant(acc: bf16[64], sg: bf16[64]):
    for i in range(64):
        sg[i] = 0.1 * allo.tanh(0.5 * acc[i]) + 0.5


def scale_bf16(acc: bf16[64], y: bf16[64]):
    for i in range(64):
        y[i] = 0.5 * acc[i]


def product_bf16(a: bf16[64], b: bf16[64], y: bf16[64]):
    for i in range(64):
        y[i] = float(a[i]) * float(b[i])


def test_store_rounding_of_a_single_operation_is_exact():
    """``0.5 * acc[i]`` and ``float(a) * float(b)`` are evaluated in f32 and
    rounded once at the bf16 store.  For one operation on bf16 values that
    equals the correctly rounded bf16 operation a device performs (double
    rounding is innocuous when the wide type has at least 2p + 2 significand
    bits: 24 >= 2 * 8 + 2), so the cast is transparent and the op matches."""
    trace = _trace(scale_bf16, "HALF")
    assert trace.coverage.errors == []
    assert [m.target_op_name for m in trace.matches] == ["HALF"]
    trace = _trace(product_bf16, "MUL")
    assert trace.coverage.errors == []
    assert [m.target_op_name for m in trace.matches] == ["MUL"]


def test_compound_store_rounding_needs_a_declared_wide_intermediate():
    """``0.5 * tanh(0.5 * acc[i]) + 0.5`` keeps f32 intermediates in the
    source and rounds once; a device that rounds after every operation
    computes something else, so the cast stays visible and no ordinary op
    matches.  An op declaring ``intermediate="f32"`` (a lookup table rounds
    once) binds the expression.  An f32 constant the destination cannot
    represent (0.1) makes the source genuinely f32: refused either way."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        require_match_coverage(_trace(activation_bf16, "AF"))
    assert "arith.truncf<f32->bf16>" in str(info.value)

    trace = _trace(activation_bf16, "AF_LUT")
    assert trace.coverage.errors == []
    assert [m.target_op_name for m in trace.matches] == ["AF_LUT"]

    with pytest.raises(UnsupportedWorkloadError) as info:
        require_match_coverage(_trace(activation_bf16_f32_constant, "AF_LUT"))
    assert "arith.truncf<f32->bf16>" in str(info.value)


def test_single_rounding_is_exact_and_compound_rounding_is_not():
    """Numerical basis of the two rules above, independent of the matcher.

    One operation: rounding the exact result to f32 and then to f16 equals
    rounding it once to f16 (checked on random pairs against exact rational
    arithmetic).  Two operations: the f32-intermediate source and the
    per-operation f16 device differ; the reviewer's two counterexamples
    (overflow of the intermediate, and cancellation that loses the low
    product bits) are kept verbatim."""
    import numpy as np
    from fractions import Fraction

    rng = np.random.default_rng(3)
    a = (rng.standard_normal(4096) * 8).astype(np.float16)
    b = (rng.standard_normal(4096) * 8).astype(np.float16)
    ops = {
        "add": (lambda p, q: p + q, np.add),
        "sub": (lambda p, q: p - q, np.subtract),
        "mul": (lambda p, q: p * q, np.multiply),
        "div": (lambda p, q: p / q, np.divide),
    }
    for name, (exact_op, np_op) in ops.items():
        keep = b != 0 if name == "div" else np.ones_like(b, dtype=bool)
        exact = [
            exact_op(Fraction(float(x)), Fraction(float(y)))
            for x, y in zip(a[keep], b[keep])
        ]
        # float(Fraction) is correctly rounded to f64; f64 -> f16 is then a
        # single rounding of an exact value (sums and products of f16
        # values are exact in f64), or an innocuous double rounding (div).
        once = np.array([np.float16(float(e)) for e in exact], dtype=np.float16)
        via_f32 = np_op(a[keep].astype(np.float32), b[keep].astype(np.float32)).astype(
            np.float16
        )
        np.testing.assert_array_equal(via_f32, once, err_msg=name)

    def source(x, y, z):  # f32 intermediates, one rounding at the store
        return np.float16(np.float32(x) * np.float32(y) + np.float32(z))

    def device(x, y, z):  # rounds after every operation
        with np.errstate(over="ignore"):
            return np.float16(np.float16(np.float16(x) * np.float16(y)) + np.float16(z))

    assert source(256, 256, -65504) == np.float16(32)
    assert np.isinf(device(256, 256, -65504))
    tiny = 1 + 2.0**-10
    assert source(tiny, tiny, -(1 + 2.0**-9)) == np.float16(2.0**-20)
    assert device(tiny, tiny, -(1 + 2.0**-9)) == np.float16(0)


N_MAD = 4096


@_df_region()
def samsung_widened_mad(A: fp16[N_MAD], B: fp16[N_MAD], C: fp16[N_MAD], Y: fp16[N_MAD]):
    @allo.work(mapping=[1], args=[A, B, C, Y])
    def calc(a: fp16[N_MAD], b: fp16[N_MAD], c: fp16[N_MAD], y: fp16[N_MAD]):
        for i in range(N_MAD):
            y[i] = float(a[i]) * float(b[i]) + float(c[i])


def test_widened_multiply_add_is_not_lowered_to_the_fp16_mad():
    """The reviewer's probe: the source keeps the product in f32 (256 * 256
    then - 65504 is 32); the Samsung MAD rounds the product to FP16 (inf).
    No Samsung op declares a wide intermediate, so compilation fails."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(samsung_widened_mad)
    assert "arith.truncf<f32->f16>" in str(info.value)


# --------------------------------------------------------------------------- #
# Output access maps (release review, round 9)
# --------------------------------------------------------------------------- #

import numpy as np


@_df_region()
def gemv_reversed_rows(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
        out[M - 1 - row] = acc


@_df_region()
def gemv_broadcast_row(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
        out[0] = acc


@_df_region()
def gemv_identity_rows(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        row = p * 8 + u
        acc: fp16 = 0
        for k in range(K):
            acc += w[row, k] * v[k]
        out[row] = acc


@_df_region()
def gemv_hoisted_rows(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def calc(w: fp16[M, K], v: fp16[K], out: fp16[M]):
        p, u = allo.get_wid()
        base = u + p * 8
        acc: fp16 = 0
        for k in range(K):
            acc += w[base, k] * v[k]
        out[base] = acc


def test_non_identity_output_maps_are_rejected_before_execution():
    """``out[M - 1 - row]`` and ``out[0]`` write the result somewhere other
    than the matrix row it came from; the Samsung lowering writes each result
    at the row it reads (the reviewer's probe returned 0..127 for a source
    that produces 127..0), so both are refused with the resolved coordinates."""
    for workload in (gemv_reversed_rows, gemv_broadcast_row):
        with pytest.raises(Exception, match="output map is not the identity") as info:
            _compile(workload)
        assert "is read at coordinate" in str(info.value)


def test_identity_and_hoisted_row_coordinates_compile_alike():
    """A renamed and re-associated row coordinate (``base = u + p * 8``) is
    the same resolved coordinate as ``row = p * 8 + u``: both compile to the
    same program."""
    identity = _compile(gemv_identity_rows)
    hoisted = _compile(gemv_hoisted_rows)
    assert identity.estimate().cycles == hoisted.estimate().cycles
    assert [str(c) for c in identity.compiled.cmds] == [
        str(c) for c in hoisted.compiled.cmds
    ]


def test_output_rows_are_written_at_the_rows_read(samsung_sim_gate):
    """Value check with a row-identifying input: W[:, 0] = row index, x = 1,
    so y must equal 0..M-1 in order (a reversed or broadcast map would show)."""
    compiled = allo.compile(gemv_identity_rows, build_samsung_target(), samsung_cost)
    W = np.zeros((M, K), dtype=np.float16)
    W[:, 0] = np.arange(M)
    x = np.ones(K, dtype=np.float16)
    y = np.full(M, 99, dtype=np.float16)
    result = compiled(W, x, y)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(y, np.arange(M, dtype=np.float16))


def test_aim_reversed_output_rows_are_rejected():
    from allo.ir.types import bfloat16 as bf16t
    from allo.pim.costs import aim_cost
    from allo.pim.targets import build_aim_target

    AM = AK = 256
    AROWS = AM // 32

    @_df_region()
    def aim_rev(W: bf16t[AM, AK], x: bf16t[AK], y: bf16t[AM]):
        @allo.work(mapping=[32], args=[W, x, y])
        def mv(lW: bf16t[AM, AK], lx: bf16t[AK], ly: bf16t[AM]):
            (ch,) = allo.get_wid()
            row0 = ch * AROWS
            for i in range(AROWS):
                acc: bf16t = 0
                for j in range(AK):
                    acc += lW[row0 + i, j] * lx[j]
                ly[AM - 1 - (row0 + i)] = acc

    with pytest.raises(Exception, match="output map is not the identity"):
        allo.compile(
            aim_rev,
            build_aim_target(),
            aim_cost,
            backend="virtual",
            reduction_order="reassociate",
        )


N_EW = 256


@_df_region()
def add_then_copy_reversed(A: fp16[N_EW], B: fp16[N_EW], T: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, T, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], t: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            t[i] = a[i] + b[i]
            y[i] = t[N_EW - 1 - i]


def test_copy_must_read_the_element_its_producer_wrote():
    """``y[i] = t[N - 1 - i]`` in the producer's own loop reads a different
    element of the matched result than ``t[i] = a[i] + b[i]`` produced (one
    not yet written); the copy contract refuses it instead of treating it as
    a plain output copy.  (A copy in a separate loop is already refused by
    the once-per-produced-element rule.)"""
    with pytest.raises(UnsupportedWorkloadError, match="copies element") as info:
        _compile(add_then_copy_reversed)
    assert "its producer stores element" in str(info.value)


def vadd_i32(A: int32[16], B: int32[16], C: int32[16]):
    for i in range(16):
        C[i] = A[i] + B[i]


def test_pointwise_sum_is_matched_in_the_stored_width():
    """``i32 + i32`` is evaluated in i33 and truncated at the i32 store:
    this is i32 addition and matches ADD with plain operands."""
    trace = _trace(vadd_i32, "ADD")
    assert trace.coverage.errors == []
    (add,) = trace.matches
    assert {b.memref_name for b in add.operands} == {"A", "B"}


from allo.ir.types import Int, uint8

int17 = Int(17)


def gemm_u8_narrow_copy(A: uint8[8, 2], B: uint8[2, 8], C: int17[8, 8]):
    for i in range(8):
        for j in range(8):
            acc: int32 = 0
            for k in range(2):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_unsigned_loads_use_the_unsigned_range():
    """``uint8`` lowers to the signless ``i8``; the frontend zero-extends it.
    Two products of two uint8 values reach 130,050, which int17 cannot hold,
    so the truncating copy is refused.  Reading the loads as signed would
    bound the sum by 32,768 and wrongly accept the copy."""
    with pytest.raises(UnsupportedWorkloadError) as info:
        require_match_coverage(_trace(gemm_u8_narrow_copy, "MAC"))
    message = str(info.value)
    assert "[0, 130050]" in message
    assert "i17 cannot hold" in message


# --------------------------------------------------------------------------- #
# Elementwise destination maps and realized copies (release review, round 10)
# --------------------------------------------------------------------------- #


@_df_region()
def ew_reverse_add(A: fp16[N_EW], B: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            y[N_EW - 1 - i] = a[i] + b[i]


@_df_region()
def ew_shifted_add(A: fp16[N_EW], B: fp16[N_EW], Y: fp16[N_EW + 1]):
    @allo.work(mapping=[1], args=[A, B, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], y: fp16[N_EW + 1]):
        for i in range(N_EW):
            y[i + 1] = a[i] + b[i]


@_df_region()
def ew_broadcast_add(A: fp16[N_EW], B: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            y[0] = a[i] + b[i]


@_df_region()
def ew_shifted_input(A: fp16[N_EW + 1], B: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, Y])
    def calc(a: fp16[N_EW + 1], b: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            y[i] = a[i + 1] * b[i]


def test_elementwise_destination_must_be_the_element_read():
    """The Samsung elementwise program writes element i of the output from
    element i of the inputs (the reviewer's reversed destination returned
    forward-order results).  Reversed, shifted and broadcast destinations and
    a shifted input are refused with the resolved coordinates, for ADD and
    MUL alike."""
    for workload in (ew_reverse_add, ew_shifted_add, ew_broadcast_add):
        with pytest.raises(Exception, match="not at the element loop index") as info:
            _compile(workload)
        assert "reversed, shifted or broadcast destination" in str(info.value)
    with pytest.raises(Exception, match="not at the element loop"):
        _compile(ew_shifted_input)


@_df_region()
def ew_copy_reversed(A: fp16[N_EW], B: fp16[N_EW], T: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, T, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], t: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            t[i] = a[i] + b[i]
            y[N_EW - 1 - i] = t[i]


@_df_region()
def ew_copy_identity(A: fp16[N_EW], B: fp16[N_EW], T: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, T, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], t: fp16[N_EW], y: fp16[N_EW]):
        for i in range(N_EW):
            t[i] = a[i] + b[i]
            y[i] = t[i]


@_df_region()
def ew_copy_two_sinks(
    A: fp16[N_EW], B: fp16[N_EW], T: fp16[N_EW], Y: fp16[N_EW], Z: fp16[N_EW]
):
    @allo.work(mapping=[1], args=[A, B, T, Y, Z])
    def calc(a: fp16[N_EW], b: fp16[N_EW], t: fp16[N_EW], y: fp16[N_EW], z: fp16[N_EW]):
        for i in range(N_EW):
            t[i] = a[i] * b[i]
            y[i] = t[i]
            z[i] = t[i]


def test_permuted_copy_of_an_array_result_is_rejected():
    """``y[N-1-i] = t[i]`` copies the produced element to a different
    element; a copy is realized element for element, so the matcher refuses
    it before any backend (the reviewer's probe reported a simulator run and
    left y untouched)."""
    with pytest.raises(
        UnsupportedWorkloadError, match="permuted or shifted copy"
    ) as info:
        _compile(ew_copy_reversed)
    assert "copies the result produced at element" in str(info.value)


def test_identity_copies_are_required_outputs():
    """``t[i] = a[i] + b[i]; y[i] = t[i]`` has two external effects.  The
    binding lists both destinations and the Samsung elementwise program
    declares it writes both; a backend that realizes only the producer
    (AiM's elementwise stream) is refused at compile time."""
    compiled = _compile(ew_copy_identity)
    binding = compiled._source_binding
    assert binding.outputs == {"out": "T", "y": "Y"}
    (match,) = compiled.trace.matches
    assert [memref for memref, _ in match.extra["output_sinks"]] == ["y"]
    assert compiled.compiled.layout_ctx.programs[0].elementwise.sink_memrefs == ("y",)

    from allo.ir.types import bfloat16 as bf16t
    from allo.pim.costs import aim_cost
    from allo.pim.targets import build_aim_target

    @_df_region()
    def aim_copy(A: bf16t[N_EW], B: bf16t[N_EW], T: bf16t[N_EW], Y: bf16t[N_EW]):
        @allo.work(mapping=[1], args=[A, B, T, Y])
        def calc(a: bf16t[N_EW], b: bf16t[N_EW], t: bf16t[N_EW], y: bf16t[N_EW]):
            for i in range(N_EW):
                t[i] = a[i] * b[i]
                y[i] = t[i]

    with pytest.raises(NotImplementedError, match="does not realize the write") as info:
        allo.compile(aim_copy, build_aim_target(), aim_cost, backend="virtual")
    assert "'y' in function 'calc_0' (copy store)" in str(info.value)


def test_identity_copies_execute_and_write_every_destination(samsung_sim_gate):
    """Executed check (values): one producer and two copies, all external;
    every destination holds the result, none keeps its initial value."""
    rng = np.random.default_rng(7)
    a = (rng.standard_normal(N_EW) * 0.5).astype(np.float16)
    b = (rng.standard_normal(N_EW) * 0.5).astype(np.float16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    compiled = allo.compile(ew_copy_identity, build_samsung_target(), samsung_cost)
    t = np.full(N_EW, 98, np.float16)
    y = np.full(N_EW, 99, np.float16)
    result = compiled(a, b, t, y)
    assert result.kind == "simulator"
    assert result.extra["copies"] == ["y"]
    np.testing.assert_array_equal(t, expected)
    np.testing.assert_array_equal(y, expected)

    expected = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
    compiled = allo.compile(ew_copy_two_sinks, build_samsung_target(), samsung_cost)
    t = np.full(N_EW, 98, np.float16)
    y = np.full(N_EW, 99, np.float16)
    z = np.full(N_EW, 97, np.float16)
    result = compiled(a, b, t, y, z)
    assert result.kind == "simulator"
    for destination in (t, y, z):
        np.testing.assert_array_equal(destination, expected)


@_df_region()
def ew_local_then_copy(A: fp16[N_EW], B: fp16[N_EW], Y: fp16[N_EW]):
    @allo.work(mapping=[1], args=[A, B, Y])
    def calc(a: fp16[N_EW], b: fp16[N_EW], y: fp16[N_EW]):
        t: fp16[N_EW] = 0
        for i in range(N_EW):
            t[i] = a[i] + b[i]
            y[i] = t[i]


def test_local_intermediate_copied_to_an_external_output(samsung_sim_gate):
    """A function-local intermediate is not an effect; its copy into the
    argument ``y`` is the only required output, and it is written."""
    compiled = allo.compile(ew_local_then_copy, build_samsung_target(), samsung_cost)
    stores = {
        (s.memref_name, s.category, s.local) for s in compiled.trace.coverage.stores
    }
    assert ("t", "matched", True) in stores and ("y", "copy", False) in stores
    assert compiled._source_binding.outputs == {"y": "Y"}
    rng = np.random.default_rng(11)
    a = (rng.standard_normal(N_EW) * 0.5).astype(np.float16)
    b = (rng.standard_normal(N_EW) * 0.5).astype(np.float16)
    y = np.full(N_EW, 99, np.float16)
    result = compiled(a, b, y)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(
        y, (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    )
