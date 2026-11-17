# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy semantics of the Gemini-I GVML binary16 arithmetic.

This is a host model, not device execution or an IEEE product fallback.
The external GSI reference library is used only by SDK-optional validation;
neither it nor its headers are distributed with this module.

Multiplication normalizes finite operands (including subnormals), rounds the
integer significand product to nearest-even at bit 10, then truncates any
normalization/underflow shift. Overflow produces signed infinity. NaNs and
zero-times-infinity produce an all-ones NaN magnitude with the XOR sign.
Addition/subtraction round finite results to nearest-even. NaNs have an
all-ones payload and the sign of the larger-magnitude operand; opposite-sign
ties have positive sign. Subtraction negates the second operand first.
See docs/release/design/gemini-i-fp16.md for the derivation and evidence.
"""

import numpy as np


# A binary16 subnormal has ten explicit significand bits. This table derives
# the left shift that restores its leading one; zero is handled separately.
_SUBNORMAL_SHIFT = np.array(
    [0] + [11 - value.bit_length() for value in range(1, 1024)], dtype=np.int32
)

NUMERICS = (
    "Gemini-I GVML binary16: multiply rounds its significand before a "
    "truncating normalization/underflow shift; add/subtract use nearest-even; "
    "NaNs use the target's canonical payload/sign rules; subgroup sums use "
    "adjacent-pair addition"
)


def _bits(value):
    return np.asarray(value, dtype=np.float16).view(np.uint16)


def _finite_parts(bits):
    exponent = ((bits >> 10) & 31).astype(np.int32)
    fraction = (bits & 1023).astype(np.int32)
    shift = np.where(exponent == 0, _SUBNORMAL_SHIFT[fraction], 0)
    significand = np.where(exponent == 0, fraction << shift, fraction | 1024)
    return significand, np.maximum(exponent, 1) - shift


def mul_f16(left, right):
    """Broadcasting GVML multiply, including every binary16 bit pattern."""
    a, b = np.broadcast_arrays(_bits(left), _bits(right))
    ma, ea = _finite_parts(a)
    mb, eb = _finite_parts(b)
    product = ma * mb  # exact, at most 22 bits
    quotient, remainder = product >> 10, product & 1023
    quotient += (remainder > 512) | ((remainder == 512) & ((quotient & 1) != 0))
    normalize = quotient >= 2048
    quotient = np.where(normalize, quotient >> 1, quotient)
    exponent = ea + eb - 15 + normalize
    # Once the pre-normalization rounding is done, these shifts truncate.
    shift = np.clip(1 - exponent, 0, 31)
    magnitude = np.where(
        exponent <= 0, quotient >> shift, (exponent << 10) | (quotient & 1023)
    )
    magnitude = np.where(exponent >= 31, 0x7C00, magnitude)
    aa, bb = a & 0x7FFF, b & 0x7FFF
    magnitude = np.where((aa == 0) | (bb == 0), 0, magnitude)
    magnitude = np.where((aa >= 0x7C00) | (bb >= 0x7C00), 0x7C00, magnitude)
    invalid = (
        (aa > 0x7C00)
        | (bb > 0x7C00)
        | ((aa == 0) & (bb == 0x7C00))
        | ((bb == 0) & (aa == 0x7C00))
    )
    bits = np.where(invalid, 0x7FFF, magnitude) | ((a ^ b) & 0x8000)
    return bits.astype(np.uint16).view(np.float16)


def add_f16(left, right):
    """Broadcasting nearest-even addition with GVML NaN payload/sign rules."""
    a, b = np.broadcast_arrays(_bits(left), _bits(right))
    # Every finite binary16 sum is exact in binary64, including the largest
    # exponent separation. The final cast performs the single RNE rounding.
    with np.errstate(all="ignore"):
        result = (
            (
                a.view(np.float16).astype(np.float64)
                + b.view(np.float16).astype(np.float64)
            )
            .astype(np.float16)
            .view(np.uint16)
        )
    aa, bb = a & 0x7FFF, b & 0x7FFF
    invalid = (
        (aa > 0x7C00)
        | (bb > 0x7C00)
        | ((aa == 0x7C00) & (bb == 0x7C00) & (((a ^ b) & 0x8000) != 0))
    )
    sign = np.where(
        aa > bb, a & 0x8000, np.where(bb > aa, b & 0x8000, (a & b) & 0x8000)
    )
    return np.where(invalid, 0x7FFF | sign, result).astype(np.uint16).view(np.float16)


def sub_f16(left, right):
    """GVML subtraction, including signed zero and NaN signs."""
    return add_f16(left, (_bits(right) ^ np.uint16(0x8000)).view(np.float16))


def add_subgrps_f16_grp(values, group_size):
    """Return subgroup-zero sums for the instruction form emitted by Tenon.

    Subgroups contain one element; each power-of-two group reduces adjacent
    pairs at each level. The last array axis contains complete groups and is
    replaced by the group count. Only the selected subgroup's value is
    returned: other destination lanes are scratch, not part of this ABI.
    Caller-side padding and addition of consecutive groups remain explicit.
    """
    values = np.asarray(values, dtype=np.float16)
    if (
        not isinstance(group_size, (int, np.integer))
        or not 1 <= group_size <= 32768
        or group_size & (group_size - 1)
    ):
        raise ValueError("GVML FP16 group size must be a power of two in [1, 32768]")
    if not values.ndim or values.shape[-1] % group_size:
        raise ValueError("GVML FP16 reduction needs complete groups on the last axis")
    tree = values.reshape(values.shape[:-1] + (-1, group_size))
    while tree.shape[-1] > 1:
        tree = add_f16(tree[..., 0::2], tree[..., 1::2])
    return tree[..., 0]
