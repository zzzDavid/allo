# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 008 / T-NONPOW2 (design 05 §1, §Q2, §8): the non-pow2 host fan-out.

UPMEM racks expose a NON-power-of-2 DPU count (2560, or 2552 masked). That
count is a HOST-SIDE data-partition fan-out, not a device-layout axis: a
`scatter(W, over=dpu)` carries it as a PLAIN INT (`prod(mapping)`), and the
host_staging cost prices it with NO `LinearLayout` -- so the strict GF(2)
`LinearLayout` (pow2-only) never sees it.

Asserts (both halves of T-NONPOW2):
  * positive: `resolve_collective_axis` returns fan==2560 (layout None) and
    the UPMEM host_staging compose prices a `STAGE_SCATTER` with fan==2560
    (M=5120, K=1024, 2 rows/DPU) with NO `ValueError` from spmw_linear_layout;
  * negative guard: `LinearLayout.identity({"dpu": 2560})` STILL raises
    (spmw_linear_layout is unchanged, strict GF(2)).
"""
from __future__ import annotations

import pytest

import allo
from allo.spmw_cost_model import get_cost_model, ComposeCtx
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchTrace

from _fixtures import build_upmem_target


# The non-pow2 stress shape: M=5120 rows over 2560 DPUs = 2 rows/DPU, K=1024.
M, K, DPU_FAN = 5120, 1024, 2560


def _upmem_with_dpu_fan(fan: int):
    """Build the UPMEM target and set the DPU unit mapping to `fan`. The DPU
    count is a host-side fan-out, so a non-pow2 value is legal here (it never
    reaches LinearLayout)."""
    target = build_upmem_target()
    dpu = next(u for u in target._walk() if u.name == "dpu")
    dpu.mapping = [fan]
    return target


# --------------------------------------------------------------------- #
# negative guard: LinearLayout stays strict GF(2) (unchanged this cycle)
# --------------------------------------------------------------------- #


def test_linear_layout_nonpow2_still_raises():
    """spmw_linear_layout is FORBIDDEN to change (design 05 §1): a non-pow2
    out-size still raises at the constructor chokepoint."""
    with pytest.raises(ValueError):
        LinearLayout.identity({"dpu": DPU_FAN})
    # 2552 (masked DPU count) likewise.
    with pytest.raises(ValueError):
        LinearLayout.identity({"dpu": 2552})


# --------------------------------------------------------------------- #
# positive: the host layer carries the non-pow2 fan as a plain int
# --------------------------------------------------------------------- #


def test_resolve_axis_returns_plain_int_fan_no_layout():
    target = _upmem_with_dpu_fan(DPU_FAN)
    unit, fan, layout = allo.resolve_collective_axis(target, over="dpu")
    assert unit.name == "dpu"
    assert fan == DPU_FAN  # plain int prod(mapping), NOT through LinearLayout
    assert isinstance(fan, int)
    assert layout is None  # the host fan-out is NEVER an F2 layout


def test_resolve_axis_masked_2552():
    target = _upmem_with_dpu_fan(2552)
    _unit, fan, layout = allo.resolve_collective_axis(target, over="dpu")
    assert fan == 2552 and layout is None


def test_scatter_fan_2560_constructs_and_prices_no_layout_error():
    """A `STAGE_SCATTER` over fan==2560 constructs + prices through the
    host_staging cost with NO ValueError from spmw_linear_layout."""
    target = _upmem_with_dpu_fan(DPU_FAN)
    _unit, fan, _layout = allo.resolve_collective_axis(target, over="dpu")

    hs = get_cost_model("upmem", "faithful", concern="host_staging")
    layout = allo.Placement(
        placements={}, extra={"host_stage": [("scatter", fan)]}
    )
    trace = MatchTrace(target_name="upmem", module_name="nonpow2", matches=[])
    # Must NOT raise (esp. no ValueError from the GF(2) layout module).
    res = hs.compose(ComposeCtx(target, trace, layout))
    per_dpu = hs.move_cost("STAGE_XFER")
    assert res.cycles == fan * per_dpu  # 2560 * 1000
    assert res.cycles > 0


def test_gather_fan_2560_prices():
    target = _upmem_with_dpu_fan(DPU_FAN)
    _u, fan, _l = allo.resolve_collective_axis(target, over="dpu")
    hs = get_cost_model("upmem", "faithful", concern="host_staging")
    layout = allo.Placement(placements={}, extra={"host_stage": [("gather", fan)]})
    trace = MatchTrace(target_name="upmem", module_name="nonpow2", matches=[])
    res = hs.compose(ComposeCtx(target, trace, layout))
    assert res.cycles == fan * hs.move_cost("GATHER_XFER")


def test_nonpow2_fan_never_touches_linear_layout(monkeypatch):
    """Defense-in-depth: if any part of the resolve+price path routed the
    fan degree through LinearLayout.identity, it would raise on 2560. Patch
    `identity` to flag ANY call, then run the full resolve+price path and
    assert it was never invoked."""
    calls = []
    orig = LinearLayout.identity

    @classmethod
    def _spy(cls, *a, **k):
        calls.append((a, k))
        return orig.__func__(cls, *a, **k)

    monkeypatch.setattr(LinearLayout, "identity", _spy)

    target = _upmem_with_dpu_fan(DPU_FAN)
    _u, fan, _l = allo.resolve_collective_axis(target, over="dpu")
    hs = get_cost_model("upmem", "faithful", concern="host_staging")
    layout = allo.Placement(
        placements={}, extra={"host_stage": [("scatter", fan), ("gather", fan)]}
    )
    trace = MatchTrace(target_name="upmem", module_name="nonpow2", matches=[])
    hs.compose(ComposeCtx(target, trace, layout))
    assert calls == [], (
        "the non-pow2 host fan-out routed through LinearLayout.identity "
        "(design 05 §1 invariant violated)"
    )
