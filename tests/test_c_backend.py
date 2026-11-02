# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the portable MLIR-to-C backend seam."""

import subprocess
import shutil
import tempfile

import allo
import numpy as np
import pytest
from allo.backend.c import (
    PortableCError,
    WideIntegerError,
    _portable_c11,
    emit_c_from_mlir,
)
from allo.ir.types import Int, float32, int8, int32, int64, uint16


def _syntax_check(source):
    with tempfile.NamedTemporaryFile(suffix=".c") as src:
        src.write(source.encode("utf-8"))
        src.flush()
        subprocess.run(
            ["clang", "-std=c11", "-Werror", "-fsyntax-only", src.name],
            check=True,
            capture_output=True,
            text=True,
        )


def test_portable_c_is_emitted_from_retained_mlir_with_control_flow():
    def kernel(a: int32[8], b: int32[8]):
        i: int32 = 0
        while i < 8:
            if a[i] > 0:
                b[i] = a[i] // 2
            else:
                b[i] = 0
            i += 1

    schedule = allo.customize(kernel)
    original_mlir = str(schedule.module)
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)

    assert artifact.source_mlir == original_mlir
    assert str(schedule.module) == original_mlir
    assert "scf.while" in artifact.source_mlir
    assert "while (true)" in artifact.source
    assert "#include <ap_int.h>" not in artifact.source
    assert "using namespace std" not in artifact.source
    assert [(arg.dtype, arg.shape, arg.mode) for arg in artifact.arguments] == [
        ("i32", (8,), "in"),
        ("i32", (8,), "out"),
    ]
    _syntax_check(artifact.source)

    executable = artifact.compile(compiler=shutil.which("clang++"))
    a = np.array([-3, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    b = np.full((8,), -1, dtype=np.int32)
    assert executable(a, b) is None
    np.testing.assert_array_equal(b, np.where(a > 0, a // 2, 0))


def test_portable_c_materializes_returned_memref_and_math():
    def kernel(a: float32[8]) -> float32[8]:
        b: float32[8] = 0.0
        for i in range(8):
            b[i] = allo.sqrt(a[i])
        return b

    schedule = allo.customize(kernel)
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)

    assert artifact.arguments[0].mode == "in"
    assert artifact.arguments[1].source == "result"
    assert artifact.arguments[1].mode == "out"
    assert "sqrt(" in artifact.source
    _syntax_check(artifact.source)

    a = np.arange(1, 9, dtype=np.float32)
    output = artifact.compile()(a)
    np.testing.assert_allclose(output, np.sqrt(a), rtol=1e-6)


def test_integer_sqrt_has_an_explicit_float_bridge_and_uint16_result():
    def kernel(a: uint16[8]) -> uint16[8]:
        b: uint16[8] = 0
        for i in range(8):
            b[i] = allo.sqrt(a[i])
        return b

    schedule = allo.customize(kernel)
    mlir = str(schedule.module)
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)

    assert "arith.uitofp" in mlir
    assert "math.sqrt" in mlir
    assert "arith.fptoui" in mlir
    _syntax_check(artifact.source)

    a = np.array([0, 1, 2, 3, 4, 15, 16, 0xFFFF], dtype=np.uint16)
    np.testing.assert_array_equal(artifact.compile()(a), np.sqrt(a).astype(np.uint16))


def test_portable_c_rejects_hls_only_integer_templates():
    int100 = Int(100)

    def kernel(a: int100[8], b: int100[8]):
        for i in range(8):
            b[i] = a[i] + 1

    schedule = allo.customize(kernel)
    try:
        emit_c_from_mlir(schedule.module, schedule.top_func_name)
    except PortableCError as error:
        assert "wider than 64" in str(error)
    else:
        raise AssertionError("expected the portable C type gate to reject ap_int")

    # Only the unary math calls upstream qualified with ``hls::`` are
    # unqualified; every other HLS library construct is still rejected.
    assert _portable_c11("float f(float x) { return hls::sqrt(x); }\n").count(
        "hls::"
    ) == 0
    try:
        _portable_c11("void f(hls::stream<int> &s) {}\n")
    except PortableCError as error:
        assert "HLS library call or type" in str(error)
    else:
        raise AssertionError("expected the portable C gate to reject hls::stream")


def test_portable_c_scalar_abi():
    def kernel(value: int32) -> int32:
        return value + 1

    schedule = allo.customize(kernel)
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)

    assert artifact.compile()(41) == 42


def test_portable_c_popcount_preserves_fixed_width_semantics():
    def signed_kernel(values: int8[6]) -> int8[6]:
        return allo.popcount(values)

    def unsigned_kernel(values: uint16[6]) -> uint16[6]:
        return allo.popcount(values)

    signed = emit_c_from_mlir(allo.customize(signed_kernel).module, "signed_kernel")
    unsigned = emit_c_from_mlir(
        allo.customize(unsigned_kernel).module, "unsigned_kernel"
    )

    assert "__builtin_popcount" in signed.source
    assert "(1u << 8) - 1u" in signed.source
    _syntax_check(signed.source)
    _syntax_check(unsigned.source)

    signed_values = np.array([-1, -128, -3, 0, 1, 0x55], dtype=np.int8)
    unsigned_values = np.array([0, 1, 0xFFFF, 0xAAAA, 0x8000, 0x1234], dtype=np.uint16)
    np.testing.assert_array_equal(
        signed.compile()(signed_values),
        np.array([8, 1, 7, 0, 1, 4], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        unsigned.compile()(unsigned_values),
        np.array([0, 1, 16, 8, 1, 5], dtype=np.uint16),
    )


# --------------------------------------------------------------------------- #
# wrap_wide_integers (audit U-02): >64-bit temporaries are kept as their low
# 64 bits, which is exact only for ops whose low result bits depend only on
# the operands' low bits.
# --------------------------------------------------------------------------- #

_WIDE_A = np.array([-3, -5, 7, -(1 << 40), 1 << 40, -1, 2, -(1 << 62)], dtype=np.int64)
_WIDE_B = np.array([5, -7, 11, 1 << 30, -(1 << 30), 1 << 62, 3, 2], dtype=np.int64)


def _wrap(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _wide(kernel):
    schedule = allo.customize(kernel, enable_tensor=False)
    return emit_c_from_mlir(
        schedule.module, schedule.top_func_name, wrap_wide_integers=True
    )


def test_wide_add_mul_shl_truncated_to_int32_keeps_exact_low_bits():
    def kernel(A: int64[8], B: int64[8], C: int32[8]):
        for i in range(8):
            C[i] = ((A[i] * B[i] + A[i]) << 3) - B[i]

    artifact = _wide(kernel)
    assert "arith.muli" in artifact.lowered_mlir and "i128" in artifact.lowered_mlir
    assert "uint64_t" in artifact.source
    out = np.zeros(8, np.int32)
    artifact.compile()(_WIDE_A.copy(), _WIDE_B.copy(), out)
    expected = [
        _wrap(((int(a) * int(b) + int(a)) << 3) - int(b), 32)
        for a, b in zip(_WIDE_A, _WIDE_B)
    ]
    assert out.tolist() == expected


def _wide_shift(A: int64[8], B: int64[8], C: int64[8]):
    for i in range(8):
        C[i] = (A[i] * B[i]) >> 8


def _wide_compare(A: int64[8], B: int64[8], C: int64[8]):
    for i in range(8):
        C[i] = 1 if A[i] * B[i] < 0 else 0


def _wide_divide(A: int64[8], B: int64[8], C: int64[8]):
    for i in range(8):
        C[i] = (A[i] * B[i]) // 16


def _wide_high_bits(A: int64[8], B: int64[8], C: int64[8]):
    for i in range(8):
        C[i] = (A[i] * B[i]) >> 64


@pytest.mark.parametrize(
    "kernel, op",
    [
        (_wide_shift, "arith.shrsi"),
        (_wide_compare, "arith.cmpi"),
        (_wide_divide, "arith.floordivsi"),
        (_wide_high_bits, "arith.shrsi"),
    ],
    ids=["ashr", "signed-compare", "floordiv", "high-bits"],
)
def test_wide_ops_that_read_high_bits_are_rejected(kernel, op):
    with pytest.raises(WideIntegerError) as error:
        _wide(kernel)
    message = str(error.value)
    assert op in message
    assert "128-bit" in message
    assert f"function {kernel.__name__}" in message
    # Callers catching either the portable-C error or NotImplementedError see it.
    assert isinstance(error.value, NotImplementedError)
    assert isinstance(error.value, PortableCError)


def test_kmeans_update_wide_epilogue_raises_instead_of_returning_wrong_values():
    # Audit U-02 (p6_hostgroup): a host-placed UPMEM group's signed wide
    # epilogue (t * scale) >> 40 went through uint64_t (logical shift of the
    # low bits). Host-placed groups run exactly this emission
    # (spmw_upmem.UpmemCtx._bind_module). The literal form of the scale is
    # covered by the test below (audit F-02).
    def update(sums: int64[4, 8], scale: int64[1], C: int64[4, 8]):
        for c in range(4):
            for d in range(8):
                t: int64 = sums[c, d]
                C[c, d] = (t * scale[0]) >> 40

    with pytest.raises(
        WideIntegerError, match="arith.shrsi on a 128-bit integer in function update"
    ):
        _wide(update)


def test_kmeans_update_wide_epilogue_with_literal_scale_raises():
    # Audit F-02: -(1 << 40) used to be typed i32 and fold to 0 (an i96
    # product); it is now the int64 constant, so the product is i128 and the
    # signed shift is rejected instead of computing (t * 0) >> 40.
    def update(sums: int64[4, 8], C: int64[4, 8]):
        for c in range(4):
            for d in range(8):
                t: int64 = sums[c, d]
                C[c, d] = (t * -(1 << 40)) >> 40

    assert "arith.constant 1099511627776 : i64" in str(
        allo.customize(update, enable_tensor=False).module
    )
    with pytest.raises(
        WideIntegerError, match="arith.shrsi on a 128-bit integer in function update"
    ):
        _wide(update)


def test_wide_literal_multiplier_keeps_exact_low_bits():
    # Audit F-02: the literal forms of the U-02 multipliers reach the
    # backend exactly; add/mul on the low 64 bits are exact.
    def kernel(A: int64[8], C: int64[8]):
        for i in range(8):
            m: int64 = -1099511627776
            C[i] = A[i] * m + A[i] * -(1 << 40) + 3000000000

    artifact = _wide(kernel)
    out = np.zeros(8, np.int64)
    artifact.compile()(_WIDE_A.copy(), out)
    expected = [
        _wrap(int(a) * -(1 << 40) * 2 + 3000000000, 64) for a in _WIDE_A
    ]
    assert out.tolist() == expected


def test_wide_integer_abi_is_rejected():
    int96 = Int(96)

    def kernel(a: int96[4], b: int32[4]):
        for i in range(4):
            b[i] = a[i]

    with pytest.raises(WideIntegerError, match="96-bit integer through the ABI"):
        _wide(kernel)
