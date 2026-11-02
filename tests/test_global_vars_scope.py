# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape and type names in annotations resolve from the defining scope only.

Regression for the frontend taking annotation names from *every* frame on
the call stack: an unrelated ``M`` in the calling script silently changed a
kernel's buffer geometry. Resolution must follow Python scoping: module
globals of the defining module, enclosing function locals (nearest first),
closure cells. Callers and harness frames contribute nothing.
"""

from __future__ import annotations

import pytest

import allo
from allo.ir.types import int32
from allo.ir.utils import get_global_vars

M = 4
N = 8


def module_kernel(A: int32[M, N]) -> int32[M, N]:
    B: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        B[i, j] = A[i, j] + 1
    return B


def _caller_with_unrelated_names():
    # These must NOT leak into module_kernel's annotation resolution.
    M = 256  # noqa: F841  pylint: disable=redefined-outer-name,unused-variable
    N = 512  # noqa: F841  pylint: disable=redefined-outer-name,unused-variable
    return allo.customize(module_kernel)


def test_module_level_kernel_ignores_caller_locals():
    s = _caller_with_unrelated_names()
    ir = str(s.module)
    assert "memref<4x8xi32>" in ir
    assert "256" not in ir and "512" not in ir


def test_enclosing_function_locals_resolve():
    K = 16  # pylint: disable=invalid-name

    def kernel(A: int32[K]) -> int32[K]:
        B: int32[K] = 0
        for i in allo.grid(K):
            B[i] = A[i] * 2
        return B

    s = allo.customize(kernel)
    assert "memref<16xi32>" in str(s.module)


def test_nearest_enclosing_scope_wins():
    K = 16  # pylint: disable=invalid-name

    def outer():
        K = 32  # pylint: disable=invalid-name,redefined-outer-name

        def kernel(A: int32[K]) -> int32[K]:
            B: int32[K] = 0
            for i in allo.grid(K):
                B[i] = A[i] - 1
            return B

        return allo.customize(kernel)

    assert "memref<32xi32>" in str(outer().module)
    del K


def test_collected_names_exclude_non_enclosing_frames():
    def kernel(A: int32[M]) -> int32[M]:
        return A

    UNRELATED_LOCAL = 7  # noqa: F841  pylint: disable=unused-variable,invalid-name

    def harness():
        LEAK = 99  # noqa: F841  pylint: disable=unused-variable,invalid-name
        return get_global_vars(kernel)

    names = harness()
    assert names["M"] == 4
    # The test function encloses ``kernel``; the harness does not.
    assert "UNRELATED_LOCAL" in names
    assert "LEAK" not in names


def test_unresolved_name_is_an_error():
    def kernel(A: int32[UNDEFINED_EXTENT]) -> int32[4]:  # noqa: F821
        return A

    # The frontend reports the unresolved name and exits (upstream Allo
    # reports compile errors with ``sys.exit(1)``; see docs/release/ISSUES.md
    # F-03). Either way the name must not be resolved from a caller frame.
    with pytest.raises((Exception, SystemExit)):
        allo.customize(kernel)
