# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blocker #2: tests for the grid-fitting pass.

Covers:
  * positional mapping against Samsung's 4-level tree
  * named mapping (bind-by-name against the target axes)
  * rejection paths (mapping too big, not divisible)
  * baseline (mapping=None) produces an empty fit
  * UPMEM and APU v1 targets to exercise non-Samsung trees
  * end-to-end: allo.compile(work, target).grid_fit is non-None and
    carries the expected assignments, without regressing the existing
    emit path
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

# conftest.py in this directory ensures a synthetic ``allo`` package is
# installed when the real ``import allo`` fails (no MLIR C-extension).
# The synthetic package exposes ``allo.work``, ``allo.compile``,
# ``allo.unit``, and ``allo.pim.*``.

from allo.work import Work, work as _work_decorator  # noqa: E402
from allo.compile import compile as _compile  # noqa: E402
from allo.pim.grid_fit import (  # noqa: E402
    grid_fit, GridFitResult, AxisAssignment, GridFitError,
)
from allo.pim.backends import (  # noqa: E402
    build_samsung, build_upmem, build_apu_v1,
)


# ---------------------------------------------------------------------------
# Helpers: construct a ``Work`` without actually running @allo.work (we
# do not need a real kernel body — grid_fit only reads ``work.mapping``).
# ---------------------------------------------------------------------------


def _work(mapping):
    """Cheap ``Work`` stub. ``grid_fit`` never touches ``.func`` / .shapes."""
    return Work(
        func=lambda: None,
        name="stub",
        shapes={},
        dtype="fp16",
        mapping=mapping,
    )


# ===========================================================================
# Samsung: axes = [channel=16, bg=4, bank=4, lane=16], product = 4096
# ===========================================================================


def test_samsung_mapping_full_grid_consumes_all_axes():
    t = build_samsung()
    # 16 * 4 * 4 * 16 = 4096 — the full Samsung PU count.
    r = grid_fit(_work([4096]), t)
    assert isinstance(r, GridFitResult)
    assert r.target_name == "samsung_hbm_pim"
    assert len(r.assignments) == 1
    a = r.assignments[0]
    assert a.workload_axis == 0
    assert a.workload_extent == 4096
    assert a.target_axes == ["channel", "bg", "bank", "lane"]
    assert a.target_extents == [16, 4, 4, 16]
    assert a.tile == 1
    # lane is SIMD -> strictest mode on this axis is "simd".
    assert a.mode == "simd"
    # Nothing left over — all four axes consumed.
    assert r.leftover_target_axes == []
    assert r.mode_annotations == {0: "simd"}


def test_samsung_mapping_partial_grid_splits_last_axis():
    """mapping=[1024] only needs 16*4*4*4 -> splits lane (16 -> 4 + 4)."""
    t = build_samsung()
    r = grid_fit(_work([1024]), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    assert a.target_axes == ["channel", "bg", "bank", "lane"]
    assert a.target_extents == [16, 4, 4, 4]
    assert a.tile == 1
    # Residue of lane (16/4 = 4) is still in the leftover list.
    assert r.leftover_target_axes == ["lane"]


def test_samsung_mapping_two_axes_16_then_64():
    """mapping=[16, 64]: first workload axis takes ``channel``, second
    takes bg * bank * lane = 4 * 4 * 16 = 256... which overshoots 64.

    Expected: first takes channel (16), second greedily eats bg + bank
    = 16 and needs another factor 4 from lane (split lane 16 -> 4).
    """
    t = build_samsung()
    r = grid_fit(_work([16, 64]), t)
    assert len(r.assignments) == 2
    a0, a1 = r.assignments
    assert a0.target_axes == ["channel"]
    assert a0.target_extents == [16]
    assert a0.tile == 1
    assert a0.mode == "mimd"
    # Second workload axis: bg*bank = 16, plus a 4-slice of lane.
    assert a1.target_axes == ["bg", "bank", "lane"]
    assert a1.target_extents == [4, 4, 4]
    assert a1.tile == 1
    # Spans a simd level -> annotation escalates.
    assert a1.mode == "simd"
    # Lane has a residue of 16/4 = 4 still on the queue.
    assert r.leftover_target_axes == ["lane"]


def test_samsung_mapping_exceeds_total_rejects():
    t = build_samsung()
    # 32768 > 16*4*4*16 = 16384 -> reject.
    with pytest.raises(GridFitError) as exc:
        grid_fit(_work([32768]), t)
    msg = str(exc.value)
    # Error message should be actionable: show workload extent and the
    # target product.
    assert "32768" in msg
    assert "remaining target" in msg or "exceeds" in msg


def test_samsung_mapping_named_binds_channel_only():
    t = build_samsung()
    r = grid_fit(_work({"channel": 16}), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    assert a.workload_axis == "channel"
    assert a.target_axes == ["channel"]
    assert a.target_extents == [16]
    assert a.tile == 1
    # bg, bank, lane remain unconsumed.
    assert r.leftover_target_axes == ["bg", "bank", "lane"]
    assert r.mode_annotations == {"channel": "mimd"}


def test_samsung_mapping_none_is_baseline():
    """No fitting work is done when mapping is None."""
    t = build_samsung()
    r = grid_fit(_work(None), t)
    assert r.assignments == []
    # All four target axes reported as leftover.
    assert r.leftover_target_axes == ["channel", "bg", "bank", "lane"]


def test_samsung_mapping_int_treated_as_single_axis():
    t = build_samsung()
    r = grid_fit(_work(1024), t)
    assert len(r.assignments) == 1
    assert r.assignments[0].workload_extent == 1024


def test_samsung_mapping_not_divisible_rejects():
    """A workload axis that cannot cleanly split the next target axis
    must raise with a helpful message."""
    t = build_samsung()
    with pytest.raises(GridFitError) as exc:
        # channel=16 * bg=4 = 64; 100 is not 16*k for any k in {1,4,16,64}.
        grid_fit(_work([100]), t)
    assert "100" in str(exc.value)


# ===========================================================================
# UPMEM: axes = [rank=1, dpu=2048, tasklet=24]
# ===========================================================================


def test_upmem_full_dpu_axis():
    t = build_upmem()
    # mapping=[2048] should consume rank (trivially, since rank=1) then dpu.
    r = grid_fit(_work([2048]), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    # The degenerate rank=1 level gets carried along; dpu provides the
    # real mass. Both axes appear on ``target_axes``.
    assert "dpu" in a.target_axes
    # rank=1 or dpu=2048 must appear; degenerate axes don't add mass.
    prod = 1
    for ext in a.target_extents:
        prod *= ext
    assert prod == 2048
    assert a.tile == 1


def test_upmem_named_mapping_binds_dpu():
    t = build_upmem()
    r = grid_fit(_work({"dpu": 32}), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    assert a.workload_axis == "dpu"
    assert a.target_axes == ["dpu"]
    assert a.target_extents == [32]
    # 2048/32 = 64 residue -> still counted as leftover.
    assert "dpu" in r.leftover_target_axes


def test_upmem_oversubscribe_dpu_rejects_when_not_divisible():
    t = build_upmem()
    # 2048 is not divisible by 48. The greedy walker should reject.
    with pytest.raises(GridFitError):
        grid_fit(_work([48]), t)


# ===========================================================================
# APU v1: axes = [apuc=4, vr=16, element=32768 (simd)]
# ===========================================================================


def test_apu_v1_simd_mode_annotation_surfaces():
    t = build_apu_v1()
    # Full product: 4 * 16 * 32768 = 2,097,152. Pick 2 * 32768 so we
    # definitely bite into the simd element axis but still fit cleanly.
    r = grid_fit(_work([4 * 16 * 32768]), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    # element is the SIMD level — the composed mode must escalate.
    assert a.mode == "simd"
    assert r.mode_annotations == {0: "simd"}


def test_apu_v1_apuc_only_stays_mimd():
    t = build_apu_v1()
    r = grid_fit(_work([4]), t)
    assert len(r.assignments) == 1
    a = r.assignments[0]
    assert a.target_axes == ["apuc"]
    assert a.mode == "mimd"
    # vr and element leftover.
    assert r.leftover_target_axes == ["vr", "element"]


# ===========================================================================
# compile() integration
# ===========================================================================


# We build the kernel inside a function so its source is readable via
# ``inspect.getsource`` — module-level decorated kernels also work, but
# inline keeps the test's dependencies explicit.


def test_compile_attaches_grid_fit_when_mapping_given():
    N = 131072  # Samsung eltwise path requires >= 131072 (see docs)

    @_work_decorator(shapes={"A": (N,), "B": (N,), "C": (N,)},
                     dtype="fp16", mapping=[1024])
    def vadd(A, B, C):
        C[:] = A + B

    t = build_samsung()
    result = _compile(vadd, target=t)
    assert result.grid_fit is not None
    assert isinstance(result.grid_fit, GridFitResult)
    assert len(result.grid_fit.assignments) == 1
    assert result.grid_fit.assignments[0].target_axes == \
        ["channel", "bg", "bank", "lane"]
    # Emit path must stay intact.
    assert len(result.emitted) > 0
    assert len(result.unlowered) == 0


def test_compile_without_mapping_has_no_grid_fit():
    N = 131072

    @_work_decorator(shapes={"A": (N,), "B": (N,), "C": (N,)}, dtype="fp16")
    def vadd_nomap(A, B, C):
        C[:] = A + B

    t = build_samsung()
    result = _compile(vadd_nomap, target=t)
    # Baseline: no fit attached when mapping is absent.
    assert result.grid_fit is None
    # Emit path identical.
    assert len(result.emitted) > 0
    assert len(result.unlowered) == 0


def test_compile_propagates_grid_fit_error():
    """Invalid mapping (extent > target) must raise GridFitError, and
    the error must propagate out of ``allo.compile``."""
    N = 131072

    @_work_decorator(shapes={"A": (N,), "B": (N,), "C": (N,)},
                     dtype="fp16", mapping=[9999])
    def vadd_bad(A, B, C):
        C[:] = A + B

    t = build_samsung()
    with pytest.raises(GridFitError):
        _compile(vadd_bad, target=t)


def test_summary_format_readable():
    t = build_samsung()
    r = grid_fit(_work([1024]), t)
    s = r.summary()
    assert "samsung_hbm_pim" in s
    assert "W[0]" in s
    assert "channel" in s


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"]))
