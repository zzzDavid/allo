# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integer literal typing (audit F-02); the rule is in allo/ir/builder.py.

An integer literal is never silently truncated: it compiles to its exact value
or compilation stops with a DTypeError (reported by `allo.customize` as
SystemExit after printing the message).
"""

import re
import shutil

import numpy as np
import pytest

import allo
from allo.backend.c import emit_c_from_mlir
from allo.ir.types import int8, int32, int64, uint8, uint32, uint64

BIG = 1 << 40


def _rejects(kernel, capsys, *fragments, **kwargs):
    with pytest.raises(SystemExit):
        allo.customize(kernel, **kwargs)
    text = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    for fragment in fragments:
        assert fragment in text, text


def _run_c(kernel, *args):
    # Wide (>64-bit) add/mul temporaries keep their exact low 64 bits (U-02).
    schedule = allo.customize(kernel)
    artifact = emit_c_from_mlir(
        schedule.module, schedule.top_func_name, wrap_wide_integers=True
    )
    artifact.compile(compiler=shutil.which("clang++"))(*args)


# --------------------------------------------------------------------------- #
# The F-02 forms
# --------------------------------------------------------------------------- #


def test_negated_shift_literal_into_int64():
    def kernel(C: int64[1]):
        C[0] = -(1 << 40)

    schedule = allo.customize(kernel)
    mlir = str(schedule.module)
    assert "arith.constant 1099511627776 : i64" in mlir
    assert "arith.shli" not in mlir
    out = np.zeros(1, np.int64)
    schedule.build()(out)
    assert out[0] == -(1 << 40)
    out = np.zeros(1, np.int64)
    _run_c(kernel, out)
    assert out[0] == -(1 << 40)


def test_annotated_int64_scalar_from_wide_literal():
    def kernel(C: int64[1]):
        m: int64 = -1099511627776
        C[0] = m

    out = np.zeros(1, np.int64)
    allo.customize(kernel).build()(out)
    assert out[0] == -1099511627776
    out = np.zeros(1, np.int64)
    _run_c(kernel, out)
    assert out[0] == -1099511627776


def test_constant_shift_of_int32_value_past_its_width_is_rejected(capsys):
    def kernel(A: int32[1], C: int32[1]):
        C[0] = A[0] << 40

    _rejects(kernel, capsys, "shift amount 40", "32-bit width", "A[0] << 40")


def test_augmented_shift_past_width_is_rejected(capsys):
    def kernel(A: int32[1], C: int32[1]):
        x: int32 = A[0]
        x <<= 32
        C[0] = x

    _rejects(kernel, capsys, "shift amount 32", "x <<= 32")


def test_negative_constant_shift_is_rejected(capsys):
    def kernel(A: int32[1], C: int32[1]):
        C[0] = A[0] >> -1

    _rejects(kernel, capsys, "negative shift amount -1")


def test_huge_constant_shift_is_rejected_without_folding(capsys):
    def kernel(C: int64[1]):
        C[0] = 1 << 100000

    _rejects(kernel, capsys, "shift amount 100000")


def test_shift_below_width_still_compiles():
    def kernel(A: int64[2], C: int64[2]):
        for i in range(2):
            C[i] = (A[i] << 31) >> 63

    a = np.array([1, 3], np.int64)
    out = np.zeros(2, np.int64)
    allo.customize(kernel).build()(a, out)
    np.testing.assert_array_equal(out, (a << 31) >> 63)


def test_literal_wider_than_int32_with_int64_buffer():
    def kernel(A: int64[4], C: int64[4]):
        for i in range(4):
            C[i] = A[i] * 3000000000 + 5000000000

    a = np.array([-7, 0, 1, 3], np.int64)
    expected = a * 3000000000 + 5000000000
    out = np.zeros(4, np.int64)
    allo.customize(kernel).build()(a, out)
    np.testing.assert_array_equal(out, expected)
    out = np.zeros(4, np.int64)
    _run_c(kernel, a, out)
    np.testing.assert_array_equal(out, expected)


def test_wide_literal_multiplier_in_c_backend():
    # The U-02 regressions pass the multiplier at run time; the literal form
    # is now exact through the portable C backend too.
    def kernel(A: int64[4], C: int64[4]):
        for i in range(4):
            C[i] = A[i] * -(1 << 20) + (1 << 40)

    a = np.array([-3, 0, 5, 1 << 20], np.int64)
    out = np.zeros(4, np.int64)
    _run_c(kernel, a, out)
    np.testing.assert_array_equal(out, a * -(1 << 20) + (1 << 40))


# --------------------------------------------------------------------------- #
# (a) conversion to a declared type
# --------------------------------------------------------------------------- #


def test_literal_shift_assigned_to_int32_is_rejected(capsys):
    def kernel(C: int32[1]):
        C[0] = 1 << 40

    _rejects(
        kernel,
        capsys,
        "integer literal 1 << 40 (= 1099511627776)",
        "not representable in i32",
        "[-2147483648, 2147483647]",
    )


def test_annotated_int32_from_wide_literal_is_rejected(capsys):
    def kernel(C: int32[1]):
        y: int32 = 1 << 40
        C[0] = y

    _rejects(kernel, capsys, "1099511627776", "assigned to `y`")


@pytest.mark.parametrize(
    "dtype, literal, ok",
    [
        (int8, 127, True),
        (int8, 128, False),
        (int8, -129, False),
        (uint8, 255, True),
        (uint8, 256, False),
        (uint8, -1, False),
    ],
)
def test_store_range_check(dtype, literal, ok, capsys):
    VALUE = literal

    def kernel(C: dtype[1]):
        C[0] = VALUE

    if ok:
        np_type = np.int8 if dtype == int8 else np.uint8
        out = np.zeros(1, np_type)
        allo.customize(kernel).build()(out)
        assert out[0] == literal
    else:
        _rejects(kernel, capsys, f"VALUE (= {literal})", "not representable")


def test_return_range_check(capsys):
    def kernel(A: int8[1]) -> int8:
        return 200

    _rejects(kernel, capsys, "integer literal 200", "returned")


def test_division_by_literal_wider_than_dividend_is_rejected(capsys):
    def kernel(A: int32[1], C: int32[1]):
        C[0] = A[0] // 4294967296

    _rejects(kernel, capsys, "4294967296", "computed in i32")


# --------------------------------------------------------------------------- #
# (b) literals without context and literal-only expressions
# --------------------------------------------------------------------------- #


def test_bare_wide_literal_and_module_constant_are_int64():
    def kernel(A: int64[2], C: int64[2]):
        x = 1 << 40
        for i in range(2):
            C[i] = A[i] + x + BIG

    a = np.array([5, -9], np.int64)
    out = np.zeros(2, np.int64)
    allo.customize(kernel).build()(a, out)
    np.testing.assert_array_equal(out, a + (1 << 41))


def test_unsigned_64_bit_all_ones():
    def kernel(C: uint64[2]):
        C[0] = (1 << 64) - 1
        C[1] = 0xFFFFFFFFFFFFFFFF

    out = np.zeros(2, np.uint64)
    allo.customize(kernel).build()(out)
    assert out.tolist() == [2**64 - 1, 2**64 - 1]


def test_hex_mask_on_uint32_value():
    def kernel(A: uint32[2], C: uint32[2]):
        for i in range(2):
            C[i] = (A[i] & 0xFFFF0000) | (1 << 31)

    a = np.array([0x12345678, 0xFFFFFFFF], np.uint32)
    out = np.zeros(2, np.uint32)
    allo.customize(kernel).build()(a, out)
    np.testing.assert_array_equal(out, (a & 0xFFFF0000) | np.uint32(1 << 31))


def test_literal_wider_than_64_bits_is_rejected(capsys):
    def kernel(C: int64[1]):
        C[0] = 1 << 64

    _rejects(kernel, capsys, "18446744073709551616", "not representable in i64")


def test_unmaterializable_wide_literal_is_rejected(capsys):
    def kernel(A: int64[1], C: int64[1]):
        C[0] = A[0] * (1 << 100) >> 100

    _rejects(kernel, capsys, "at most 64 bits")


def test_wide_intermediate_of_literal_only_expression_folds():
    def kernel(C: int64[1]):
        C[0] = (1 << 100) >> 90

    out = np.zeros(1, np.int64)
    allo.customize(kernel).build()(out)
    assert out[0] == 1024


def test_in_range_literals_keep_their_lowering():
    # Blast-radius pin: int32-representable literals and literal-only
    # expressions that Allo's rules compute exactly lower exactly as before.
    def kernel(A: int32[1], C: int32[1]):
        C[0] = A[0] * -32 + (1 << 4)

    mlir = str(allo.customize(kernel).module)
    assert "arith.constant 32 : i32" in mlir
    assert "arith.shli" in mlir
    assert "i64" in mlir  # i32 * i32 -> i64 under the default rules
    assert "1099511627776" not in mlir


def test_cpp_style_wide_literal_and_unsigned_modular_operand():
    def kernel(A: uint32[2], C: int64[2]):
        for i in range(2):
            C[i] = A[i] + (-1) + 1099511627776  # cpp-style has no shift rule

    a = np.array([0, 7], np.uint32)
    schedule = allo.customize(kernel, typing_rule_set="cpp-style")
    out = np.zeros(2, np.int64)
    schedule.build()(a, out)
    # C semantics: uint32 + (-1) wraps in uint32, then widens to int64.
    expected = (a + np.uint32(0xFFFFFFFF)).astype(np.int64) + (1 << 40)
    np.testing.assert_array_equal(out, expected)
