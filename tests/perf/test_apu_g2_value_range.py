# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini-II readback range checks (G2-09) and DIV carrier padding (G2-10).

Synthetic carriers only; nothing here launches the card.
"""

import numpy as np
import pytest

import allo
from allo.pim.apu_g2_composed_layout import (
    canonicalize_apu_g2_values,
    decode_apu_g2_u16_bitpatterns,
    encode_apu_g2_u16_bitpatterns,
    gather_apu_g2_composed_output,
)
from allo.pim.apu_g2_layout import APUG2_U16_SHAPE
from allo.pim.apu_g2_typed_program import (
    APUG2_TYPED_CAPACITY,
    APUG2ScalarType,
    APUG2TypedProgram,
)
from allo.pim.apu_g2_typed_runtime import (
    _expected_output,
    _logical_output,
    pack_apu_g2_typed_operands,
)

I2 = APUG2ScalarType(2, True)
I4 = APUG2ScalarType(4, True)
I8 = APUG2ScalarType(8, True)
I15 = APUG2ScalarType(15, True)
I16 = APUG2ScalarType(16, True)
U16 = APUG2ScalarType(16, False)


def test_decode_rejects_payload_bits_above_the_declared_width():
    # The audit probe: these decoded silently to [-1, -1, 0].
    for payload in (0xFFFF, 0x8003, 0x0004):
        with pytest.raises(ValueError, match="bits above the 2-bit int2"):
            decode_apu_g2_u16_bitpatterns(np.array([payload], dtype=np.uint16), I2)
    decoded = decode_apu_g2_u16_bitpatterns(np.array([3, 1, 2, 0], dtype=np.uint16), I2)
    np.testing.assert_array_equal(decoded, [-1, 1, -2, 0])


def test_decode_and_encode_keep_the_exact_width_paths():
    every = np.arange(1 << 16, dtype=np.uint16)
    np.testing.assert_array_equal(decode_apu_g2_u16_bitpatterns(every, U16), every)
    signed = decode_apu_g2_u16_bitpatterns(every, I16)
    np.testing.assert_array_equal(signed, every.view(np.int16))
    np.testing.assert_array_equal(encode_apu_g2_u16_bitpatterns(signed, I16), every)
    values = np.arange(I15.minimum, I15.maximum + 1, dtype=np.int64)
    round_trip = decode_apu_g2_u16_bitpatterns(
        encode_apu_g2_u16_bitpatterns(values, I15), I15
    )
    np.testing.assert_array_equal(round_trip, values)


def test_canonicalize_rejects_out_of_range_values_unless_wrapping_is_asked():
    with pytest.raises(ValueError, match=r"16384 at \(0,\) is outside int15"):
        canonicalize_apu_g2_values([16384], I15)
    with pytest.raises(ValueError, match="outside int15"):
        encode_apu_g2_u16_bitpatterns([-16385], I15)
    assert canonicalize_apu_g2_values([16384], I15, wrap=True).tolist() == [-16384]
    exact = canonicalize_apu_g2_values([I15.minimum, -1, 0, I15.maximum], I15)
    assert exact.dtype == np.int16
    assert exact.tolist() == [-16384, -1, 0, 16383]


def _composed_program():
    return allo.APUG2ComposedContractionProgram((I4, I4), I15, (64, 31), 128)


def test_composed_readback_rejects_a_device_value_outside_the_output_type():
    program = _composed_program()
    carrier = np.zeros(APUG2_U16_SHAPE, dtype=np.int16)
    column, mmb_set = program.layout_plan.physical_coordinate(5 * 31 + 7, 0)
    carrier[mmb_set, column] = 16383
    gathered = gather_apu_g2_composed_output(program, carrier)
    assert gathered[5, 7] == 16383 and np.count_nonzero(gathered) == 1
    # Bit 14 set without sign extension: an int16 container value that no
    # int15 result has (it would have wrapped to -16384 before).
    carrier[mmb_set, column] = 16384
    with pytest.raises(ValueError, match="device output 16384 .* outside int15"):
        gather_apu_g2_composed_output(program, carrier)


def test_typed_readback_rejects_a_device_value_outside_the_output_type():
    program = APUG2TypedProgram("dot", (I4, I4), I15, (8, 128), reduction_extent=128)
    carrier = np.zeros(APUG2_TYPED_CAPACITY, dtype=np.int16)
    carrier[3 * 128] = -16384
    assert _logical_output(program, carrier).tolist() == [0, 0, 0, -16384, 0, 0, 0, 0]
    carrier[3 * 128] = -16385
    with pytest.raises(ValueError, match="device output -16385 .* outside int15"):
        _logical_output(program, carrier)


def test_typed_oracle_wraps_only_the_modular_add():
    add = APUG2TypedProgram("add", (I16, I16), I16, (2,))
    lhs = np.array([32767, -32768], dtype=np.int16)
    rhs = np.array([1, -1], dtype=np.int16)
    assert _expected_output(add, (lhs, rhs)).tolist() == [-32768, 32767]
    mul = APUG2TypedProgram("mul", (I8, I8), I16, (2,))
    edge = np.array([-128, 127], dtype=np.int8)
    assert _expected_output(mul, (edge, edge)).tolist() == [16384, 16129]


def test_partial_shape_div_pads_the_divisor_with_one():
    program = APUG2TypedProgram("div", (U16, U16), U16, (3, 100))
    numerator = np.arange(300, dtype=np.uint16).reshape(3, 100)
    divisor = np.arange(1, 301, dtype=np.uint16).reshape(3, 100)
    lhs, rhs = pack_apu_g2_typed_operands(program, (numerator, divisor))
    lhs, rhs = lhs.reshape(-1), rhs.reshape(-1)
    assert lhs.size == rhs.size == APUG2_TYPED_CAPACITY
    np.testing.assert_array_equal(lhs[:300], numerator.reshape(-1))
    np.testing.assert_array_equal(rhs[:300], divisor.reshape(-1))
    assert not np.any(lhs[300:])
    assert np.all(rhs[300:] == 1)
    assert np.all(rhs != 0)


def test_other_typed_operations_keep_zero_padding():
    program = APUG2TypedProgram("mul", (I8, I8), I16, (3, 100))
    operand = np.ones((3, 100), dtype=np.int8)
    for physical in pack_apu_g2_typed_operands(program, (operand, operand)):
        flat = physical.reshape(-1)
        assert np.all(flat[:300] == 1) and not np.any(flat[300:])
