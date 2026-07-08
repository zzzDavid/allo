# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the portable MLIR-to-C backend seam."""

import subprocess
import shutil
import tempfile

import allo
import numpy as np
from allo.backend.c import PortableCError, emit_c_from_mlir
from allo.ir.types import Int, float32, int8, int32, uint16


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
