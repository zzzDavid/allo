# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact numeric contract for all-suite APUg2 lowering."""

import numpy as np
import pytest

from allo.pim.apu_g2_numeric import APUG2_U16_POLICY, specialize_integer_ratio


def test_polybench_fractional_coefficients_get_nonzero_ratio_specializations():
    gesummv = specialize_integer_ratio({"alpha": 1.5, "beta": 1.2})
    two_mm = specialize_integer_ratio({"alpha": 0.1, "beta": 0.5})
    signed = specialize_integer_ratio({"negative": -0.5, "positive": 0.5})

    assert dict(gesummv.signed) == {"alpha": 5, "beta": 4}
    assert dict(gesummv.encoded) == {"alpha": 5, "beta": 4}
    assert dict(two_mm.signed) == {"alpha": 1, "beta": 5}
    assert dict(two_mm.encoded) == {"alpha": 1, "beta": 5}
    assert dict(signed.signed) == {"negative": -1, "positive": 1}
    assert dict(signed.encoded) == {"negative": 65535, "positive": 1}
    assert all(value != 0 for value in two_mm.encoded.values())


def test_u16_policy_wraps_each_logical_store_boundary():
    policy = APUG2_U16_POLICY
    lhs = np.array([65535, 40000, 9], dtype=np.uint16)
    rhs = np.array([2, 40000, 7], dtype=np.uint16)

    np.testing.assert_array_equal(policy.add(lhs, rhs), [1, 14464, 16])
    np.testing.assert_array_equal(policy.sub(lhs, rhs), [65533, 0, 2])
    np.testing.assert_array_equal(policy.mul(lhs, rhs), [65534, 4096, 63])
    np.testing.assert_array_equal(
        policy.sum(np.stack([lhs, rhs]), axis=0), [1, 14464, 16]
    )


def test_u16_policy_defines_unsigned_division_and_floor_sqrt():
    policy = APUG2_U16_POLICY

    np.testing.assert_array_equal(
        policy.div(
            np.array([65535, 100, 9], dtype=np.uint16),
            np.array([255, 9, 2], dtype=np.uint16),
        ),
        [257, 11, 4],
    )
    np.testing.assert_array_equal(
        policy.sqrt(np.array([0, 1, 2, 15, 16, 17, 65535], dtype=np.uint16)),
        [0, 1, 1, 3, 4, 4, 255],
    )
    with pytest.raises(ZeroDivisionError):
        policy.div(np.array([1], dtype=np.uint16), np.array([0], dtype=np.uint16))
