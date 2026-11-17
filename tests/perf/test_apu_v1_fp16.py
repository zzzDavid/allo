# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target primitive checks independent of compilation and target scheduling."""
import numpy as np
import pytest

from allo.pim.apu_v1_fp16 import add_f16, sub_f16, mul_f16, add_subgrps_f16_grp
from benchmarks.apu_v1.fp16_reference import Reference, ReferenceUnavailable, compare


def test_installed_sdk_reference(tmp_path):
    try:
        ref = Reference(tmp_path / "reference")
    except ReferenceUnavailable as error:
        pytest.skip(str(error))
    rng = np.random.default_rng(9214)
    all_bits = np.arange(65536, dtype=np.uint16)
    anchors = np.array(
        [
            0,
            0x8000,
            1,
            0x8001,
            0x3FF,
            0x400,
            0x3C00,
            0xBC00,
            0x7BFF,
            0xFBFF,
            0x7C00,
            0xFC00,
            0x7C01,
            0xFC01,
            0x7FFF,
            0xFFFF,
        ],
        dtype=np.uint16,
    )
    left = rng.integers(0, 65536, 1048576, dtype=np.uint16)
    right = rng.integers(0, 65536, left.size, dtype=np.uint16)
    for operation in range(3):
        for a, b in (
            (left, right),
            (all_bits[:, None], anchors[None, :]),
            (anchors[:, None], all_bits[None, :]),
        ):
            result = compare(ref, a, b, operation)
            assert result["mismatches"] == 0, result
    for offset in (0, 32768):
        bits = all_bits[offset : offset + 32768]
        for power in range(16):
            group = 1 << power
            expected = ref.groups(bits, group)
            actual = add_subgrps_f16_grp(bits.view(np.float16), group)
            np.testing.assert_array_equal(actual.view(np.uint16), expected)


def test_ieee_product_difference_and_exact_product_paper_inputs():
    from benchmarks.portability.gemv_source_order import (
        historical_inputs,
        products_are_exact,
        source_order_fp16,
    )

    # An independent IEEE oracle remains visible next to target arithmetic.
    rng = np.random.default_rng(300)
    left = (rng.standard_normal((1024, 1)) * 0.5).astype(np.float16)
    right = (rng.standard_normal(1) * 0.5).astype(np.float16)
    ieee = (left.astype(np.float64) * right.astype(np.float64)).astype(np.float16)
    target = mul_f16(left, right)
    assert np.count_nonzero(ieee.view(np.uint16) != target.view(np.uint16)) == 66
    A, B = historical_inputs()
    assert products_are_exact(A, B)
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float16)
    for k in range(A.shape[1]):
        result = add_f16(result, mul_f16(A[:, k, None], B[k, None, :]))
    np.testing.assert_array_equal(
        result.view(np.uint16), source_order_fp16(A, B).view(np.uint16)
    )


def test_group_tree_is_adjacent_and_requires_complete_power_of_two_groups():
    values = np.array([2048, 1, -2048, 1], dtype=np.float16)
    assert add_subgrps_f16_grp(values, 4)[0] == 1
    assert add_f16(add_f16(values[0], values[2]), add_f16(values[1], values[3])) == 2
    for group in (0, 3, 65536):
        with pytest.raises(ValueError, match="power of two"):
            add_subgrps_f16_grp(values, group)
    with pytest.raises(ValueError, match="complete groups"):
        add_subgrps_f16_grp(values[:3], 4)
    negative_zero = np.array(0x8000, dtype=np.uint16).view(np.float16)
    assert add_f16(negative_zero, negative_zero).view(np.uint16) == 0x8000
    assert sub_f16(negative_zero, np.float16(0)).view(np.uint16) == 0x8000
