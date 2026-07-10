# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit modular uint16 semantics for the APUg2 PolyBench backend."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from types import MappingProxyType
from typing import Mapping

import numpy as np


_MASK = np.uint64(0xFFFF)


@dataclass(frozen=True)
class APUG2CoefficientSet:
    """Integer specialization of one group of source-level coefficients."""

    source: Mapping[str, Fraction]
    signed: Mapping[str, int]
    encoded: Mapping[str, int]

    def __post_init__(self):
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))
        object.__setattr__(self, "signed", MappingProxyType(dict(self.signed)))
        object.__setattr__(self, "encoded", MappingProxyType(dict(self.encoded)))


def specialize_integer_ratio(
    bindings: Mapping[str, int | float]
) -> APUG2CoefficientSet:
    """Preserve the exact ratio of finite decimal coefficients as integers.

    For example, ``{alpha: 1.5, beta: 1.2}`` becomes signed integers ``5:4``
    and ``{alpha: 0.1, beta: 0.5}`` becomes ``1:5``. Negative integers use
    their two's-complement uint16 encoding. A coefficient group that cannot
    fit in one signed/modular uint16 scalar is rejected and requires an
    explicit workload specialization.
    """

    if not bindings:
        return APUG2CoefficientSet({}, {}, {})
    fractions = {}
    for name, value in bindings.items():
        if not isinstance(name, str) or not name:
            raise TypeError("coefficient names must be nonempty strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"coefficient {name!r} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"coefficient {name!r} must be finite")
        fractions[name] = Fraction(str(value))

    denominator = 1
    for value in fractions.values():
        denominator = lcm(denominator, value.denominator)
    integers = {
        name: value.numerator * (denominator // value.denominator)
        for name, value in fractions.items()
    }
    divisor = 0
    for value in integers.values():
        divisor = gcd(divisor, abs(value))
    divisor = max(divisor, 1)
    integers = {name: value // divisor for name, value in integers.items()}
    oversized = {name: value for name, value in integers.items() if abs(value) > 65535}
    if oversized:
        raise ValueError(
            "coefficient ratio does not fit uint16 specialization: "
            + ", ".join(f"{name}={value}" for name, value in oversized.items())
        )
    encoded = {name: value & 0xFFFF for name, value in integers.items()}
    return APUG2CoefficientSet(fractions, integers, encoded)


@dataclass(frozen=True)
class APUG2U16Policy:
    """Reference semantics mirrored by direct-VL64 recipes."""

    name: str = "u16_modular"
    storage_dtype: np.dtype = np.dtype(np.uint16)
    reduction_bits: int = 24
    sqrt_rounding: str = "floor"
    division: str = "unsigned"

    @staticmethod
    def _u64(value) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind not in "biu":
            raise TypeError("APUg2 modular arithmetic requires integer operands")
        return array.astype(np.uint64)

    def store(self, value) -> np.ndarray:
        return np.asarray(self._u64(value) & _MASK, dtype=np.uint16)

    def add(self, lhs, rhs) -> np.ndarray:
        return self.store(self._u64(lhs) + self._u64(rhs))

    def sub(self, lhs, rhs) -> np.ndarray:
        return self.store(self._u64(lhs) - self._u64(rhs))

    def mul(self, lhs, rhs) -> np.ndarray:
        return self.store(self._u64(lhs) * self._u64(rhs))

    def sum(self, value, *, axis=None) -> np.ndarray:
        return self.store(np.sum(self._u64(value), axis=axis, dtype=np.uint64))

    def div(self, numerator, denominator) -> np.ndarray:
        numerator = self._u64(numerator)
        denominator = self._u64(denominator)
        if np.any(denominator == 0):
            raise ZeroDivisionError("APUg2 uint16 division by zero is undefined")
        return self.store(numerator // denominator)

    def sqrt(self, value) -> np.ndarray:
        value = self._u64(value)
        # uint16 inputs are small enough for float64 sqrt to identify either
        # the exact floor or an adjacent integer. Correct both directions so
        # this remains an integer definition rather than a floating contract.
        root = np.floor(np.sqrt(value.astype(np.float64))).astype(np.uint64)
        root = np.where(root * root > value, root - 1, root)
        root = np.where((root + 1) * (root + 1) <= value, root + 1, root)
        return self.store(root)


APUG2_U16_POLICY = APUG2U16Policy()


__all__ = [
    "APUG2CoefficientSet",
    "APUG2U16Policy",
    "APUG2_U16_POLICY",
    "specialize_integer_ratio",
]
