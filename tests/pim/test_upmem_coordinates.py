# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM family recognition reads the matcher's resolved access coordinates
(ISSUES U-16, rounds 68-69): equivalent spellings of an access classify and
lower alike, and a guard that tests another element than the copied one is
refused before selection."""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import int32
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target

N = 64


@_df_region()
def select_inline(A: int32[128], O: int32[128], C: int32[2]):
    @allo.work(mapping=[1], args=[A, O, C])
    def work(lA: int32[128], lO: int32[128], lC: int32[2]):
        lC[0] = 0
        for i in range(N):
            if lA[64 + i] > 0:
                lO[lC[0]] = lA[64 + i]
                lC[0] += 1


@_df_region()
def select_hoisted(A: int32[128], O: int32[128], C: int32[2]):
    @allo.work(mapping=[1], args=[A, O, C])
    def work(lA: int32[128], lO: int32[128], lC: int32[2]):
        row0 = 64
        lC[0] = 0
        for i in range(N):
            if lA[row0 + i] > 0:
                lO[lC[0]] = lA[row0 + i]
                lC[0] += 1


@_df_region()
def select_shifted_guard(A: int32[128], O: int32[128], C: int32[2]):
    """The guard tests ``A[i]`` while ``A[64 + i]`` is copied: not the
    selection template (the predicate is over another element)."""

    @allo.work(mapping=[1], args=[A, O, C])
    def work(lA: int32[128], lO: int32[128], lC: int32[2]):
        lC[0] = 0
        for i in range(N):
            if lA[i] > 0:
                lO[lC[0]] = lA[64 + i]
                lC[0] += 1


BINS, SHIFT = 256, 24


@_df_region()
def hist_inline(X: int32[128], H: int32[BINS]):
    @allo.work(mapping=[1], args=[X, H])
    def work(lX: int32[128], lH: int32[BINS]):
        for i in range(N):
            b = (lX[64 + i] * BINS) >> SHIFT
            if b >= 0 and b < BINS:
                lH[b] += 1


@_df_region()
def hist_hoisted(X: int32[128], H: int32[BINS]):
    @allo.work(mapping=[1], args=[X, H])
    def work(lX: int32[128], lH: int32[BINS]):
        row0 = 64
        for i in range(N):
            b = (lX[row0 + i] * BINS) >> SHIFT
            if b >= 0 and b < BINS:
                lH[b] += 1


def _segment(compiled):
    (segment,) = compiled.compiled.layout_ctx.segments
    return segment


def test_selection_spellings_classify_and_lower_alike():
    inline = allo.compile(
        select_inline, build_upmem_target(), upmem_cost, backend="virtual"
    )
    hoisted = allo.compile(
        select_hoisted, build_upmem_target(), upmem_cost, backend="virtual"
    )
    a, b = _segment(inline), _segment(hoisted)
    assert a.kernel.family == b.kernel.family == "selection"
    assert dict(a.kernel.geometry) == dict(b.kernel.geometry)
    assert a.kernel.constants == b.kernel.constants
    assert a.device_source == b.device_source
    assert inline.estimate().cycles == hoisted.estimate().cycles


def test_selection_guard_over_another_element_is_refused_before_selection():
    with pytest.raises(
        NotImplementedError, match="guard tests a value other than the copied one"
    ):
        allo.compile(
            select_shifted_guard, build_upmem_target(), upmem_cost, backend="virtual"
        )


def test_histogram_spellings_classify_and_lower_alike():
    inline = allo.compile(
        hist_inline, build_upmem_target(), upmem_cost, backend="virtual"
    )
    hoisted = allo.compile(
        hist_hoisted, build_upmem_target(), upmem_cost, backend="virtual"
    )
    a, b = _segment(inline), _segment(hoisted)
    assert a.kernel.family == b.kernel.family == "histogram"
    assert dict(a.kernel.geometry) == dict(b.kernel.geometry)
    assert a.device_source == b.device_source


def test_selection_spellings_execute_alike():
    """Both spellings run on the simulator (through the configured SDK
    executor) and select the positive elements of the second half."""
    import conftest

    conftest._simulator_gate("upmem")
    rng = np.random.default_rng(5)
    A = rng.integers(-50, 50, size=128, dtype=np.int32)
    expected = A[64:][A[64:] > 0]
    results = {}
    for name, region in (("inline", select_inline), ("hoisted", select_hoisted)):
        compiled = allo.compile(region, build_upmem_target(), upmem_cost)
        O = np.zeros(128, np.int32)
        C = np.zeros(2, np.int32)
        result = compiled(A.copy(), O, C)
        assert result.kind == "simulator"
        assert C[0] == expected.size
        np.testing.assert_array_equal(O[: expected.size], expected)
        results[name] = result.cycles
    assert results["inline"] == results["hoisted"]
