# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral tests for the allo.popcount programming abstraction."""

import numpy as np
import pytest

import allo
from allo.ir.types import float32, int8, uint8, uint16, uint32


def scalar_u16(value: uint16) -> uint16:
    return allo.popcount(value)


def scalar_i8(value: int8) -> int8:
    return allo.popcount(value)


def shaped_u8(values: uint8[2, 4]) -> uint8[2, 4]:
    return allo.popcount(values)


def shaped_u32(values: uint32[2, 3]) -> uint32[2, 3]:
    return allo.popcount(values)


def invalid_float(value: float32) -> float32:
    return allo.popcount(value)


def _expected(values):
    values = np.asarray(values)
    width = values.dtype.itemsize * 8
    mask = (1 << width) - 1
    return np.asarray(
        [(int(value) & mask).bit_count() for value in values.flat],
        dtype=values.dtype,
    ).reshape(values.shape)


def test_scalar_popcount_retains_math_op_and_operand_width():
    unsigned = allo.customize(scalar_u16)
    signed = allo.customize(scalar_i8)

    assert "math.ctpop" in str(unsigned.module)
    assert "math.ctpop" in str(signed.module)
    assert "math.ctpop %arg0 : i16" in str(unsigned.module)
    assert "math.ctpop %arg0 : i8" in str(signed.module)

    unsigned_cpu = unsigned.build()
    signed_cpu = signed.build()
    assert unsigned_cpu(np.uint16(0xA55A)) == 8
    # Signed values count their fixed-width two's-complement representation.
    assert signed_cpu(np.int8(-1)) == 8


@pytest.mark.parametrize("enable_tensor", [False, True])
def test_shaped_popcount_is_elementwise_typed_and_cpu_executable(enable_tensor):
    schedule = allo.customize(shaped_u8, enable_tensor=enable_tensor)
    module = str(schedule.module)

    assert "linalg.generic" in module
    assert "math.ctpop" in module
    assert "memref<2x4xi8>" in module or "tensor<2x4xi8>" in module

    values = np.asarray([[0, 1, 3, 0xFF], [0x55, 0x80, 0x7F, 0xA5]], dtype=np.uint8)
    np.testing.assert_array_equal(schedule.build()(values), _expected(values))


def test_wide_shaped_popcount_preserves_result_element_width():
    schedule = allo.customize(shaped_u32, enable_tensor=False)
    module = str(schedule.module)
    assert "math.ctpop" in module
    assert "memref<2x3xi32>" in module

    values = np.asarray(
        [[0, 0xFFFFFFFF, 0xAAAAAAAA], [1 << 31, 0xF0F0F0F0, 7]],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(schedule.build()(values), _expected(values))


def test_popcount_rejects_non_integer_operands_during_inference():
    # customize reports inference diagnostics and exits consistently with all
    # other front-end type errors.
    with pytest.raises(SystemExit):
        allo.customize(invalid_float)


def test_python_semantics_match_compiled_width_contract():
    signed = np.asarray([-1, -128, 0, 3], dtype=np.int8)
    np.testing.assert_array_equal(allo.popcount(signed), _expected(signed))
    assert allo.popcount(np.int8(-1)) == 8
