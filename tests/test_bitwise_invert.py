# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral fixed-width complement and XNOR lowering."""

import numpy as np
import pytest

import allo
from allo.backend.c import emit_c_from_mlir
from allo.ir.types import Int, UInt, int8, int16, int32, uint8, uint16, uint32


@pytest.mark.parametrize(
    "dtype,np_dtype,value",
    [
        (int8, np.int8, 0x35),
        (int16, np.int16, 0x1234),
        (int32, np.int32, 0x12345678),
        (uint8, np.uint8, 0xA5),
        (uint16, np.uint16, 0xA5C3),
        (uint32, np.uint32, 0xA5C31234),
    ],
)
def test_scalar_invert_preserves_integer_width(dtype, np_dtype, value):
    def kernel(x: dtype) -> dtype:
        return ~x

    schedule = allo.customize(kernel)
    retained = str(schedule.module)
    assert "arith.xori" in retained
    assert "arith.sitofp" not in retained
    result = schedule.build()(np_dtype(value))
    expected = np.bitwise_not(np_dtype(value)).item()
    assert result == expected


@pytest.mark.parametrize("enable_tensor", [False, True])
def test_whole_tensor_invert_lowers_elementwise(enable_tensor):
    def kernel(a: uint8[17]) -> uint8[17]:
        return ~a

    schedule = allo.customize(kernel, enable_tensor=enable_tensor)
    retained = str(schedule.module)
    assert "linalg.generic" in retained
    assert "arith.xori" in retained

    values = np.arange(17, dtype=np.uint8) * np.uint8(7)
    np.testing.assert_array_equal(schedule.build()(values), np.bitwise_not(values))


def test_xnor_is_composable_from_xor_and_invert():
    def kernel(a: uint16[19], b: uint16[19]) -> uint16[19]:
        result: uint16[19] = 0
        for i in range(19):
            result[i] = ~(a[i] ^ b[i])
        return result

    schedule = allo.customize(kernel, enable_tensor=False)
    retained = str(schedule.module)
    assert retained.count("arith.xori") >= 2

    lhs = np.arange(19, dtype=np.uint16) * np.uint16(129)
    rhs = np.arange(19, dtype=np.uint16)[::-1] * np.uint16(73)
    expected = np.bitwise_not(np.bitwise_xor(lhs, rhs))
    np.testing.assert_array_equal(schedule.build()(lhs, rhs), expected)


def test_public_scalar_xnor_has_canonical_retained_mlir():
    def kernel(lhs: uint8, rhs: uint8) -> uint8:
        return allo.xnor(lhs, rhs)

    schedule = allo.customize(kernel)
    retained = str(schedule.module)
    assert retained.count("arith.xori") == 2
    assert retained.count("unsigned") >= 2
    assert schedule.build()(np.uint8(0xA5), np.uint8(0x3C)) == np.uint8(
        ~(0xA5 ^ 0x3C) & 0xFF
    )


@pytest.mark.parametrize("enable_tensor", [False, True])
def test_public_shaped_xnor_is_elementwise(enable_tensor):
    def kernel(lhs: uint16[19], rhs: uint16[19]) -> uint16[19]:
        return allo.xnor(lhs, rhs)

    schedule = allo.customize(kernel, enable_tensor=enable_tensor)
    retained = str(schedule.module)
    assert "linalg.generic" in retained
    assert retained.count("arith.xori") == 2

    lhs = np.arange(19, dtype=np.uint16) * np.uint16(129)
    rhs = np.arange(19, dtype=np.uint16)[::-1] * np.uint16(73)
    np.testing.assert_array_equal(
        schedule.build()(lhs, rhs), np.bitwise_not(np.bitwise_xor(lhs, rhs))
    )


def test_arbitrary_width_invert_retains_width_and_unsigned_metadata():
    unsigned5 = UInt(5)
    signed13 = Int(13)

    def invert_u5(x: unsigned5) -> unsigned5:
        return ~x

    def invert_i13(x: signed13) -> signed13:
        return ~x

    unsigned_mlir = str(allo.customize(invert_u5).module)
    signed_mlir = str(allo.customize(invert_i13).module)
    assert "arith.xori" in unsigned_mlir and "i5" in unsigned_mlir
    assert "unsigned" in unsigned_mlir
    assert "arith.xori" in signed_mlir and "i13" in signed_mlir
    assert "unsigned" not in signed_mlir


def test_xnor_lowers_through_portable_c_backend():
    def kernel(lhs: uint16[19], rhs: uint16[19]) -> uint16[19]:
        return allo.xnor(lhs, rhs)

    schedule = allo.customize(kernel, enable_tensor=False)
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)
    assert "^" in artifact.source

    lhs = np.arange(19, dtype=np.uint16) * np.uint16(129)
    rhs = np.arange(19, dtype=np.uint16)[::-1] * np.uint16(73)
    np.testing.assert_array_equal(
        artifact.compile()(lhs, rhs),
        np.bitwise_not(np.bitwise_xor(lhs, rhs)),
    )


def test_invert_rejects_non_integer_values():
    from allo.ir.types import float32

    def kernel(x: float32) -> float32:
        return ~x

    with pytest.raises(SystemExit):
        allo.customize(kernel)
