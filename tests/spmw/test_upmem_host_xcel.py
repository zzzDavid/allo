# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 008 (Phase 5) -- generality proof: the HostXcel collective interface
on a SECOND target (UPMEM), with a PARTIAL basis implementation.

UPMEM declares `scatter`/`gather`/`broadcast` but NO device `reduce` (it has
no cross-DPU reduction primitive). This exercises the coverage check
(design 05 §2/§7):
  * `all_gather` (= gather + broadcast) is covered via the default
    composition;
  * `all_reduce` / `reduce_scatter` (need a `reduce` basis) raise the hard
    coverage error.

Also proves the additive host node is parity-neutral: it is a SIBLING of
`rank`, so the UPMEM device tree / `kernel_cycles` is byte-for-byte
unchanged (the host_staging cost is 0 for traces with no staging signal).
"""
from __future__ import annotations

import pytest

import allo
from allo.spmw_host import NotSupported
from allo.spmw_match import MatchTrace

from _fixtures import build_upmem_target


def _host_xcel(target):
    for u in target._walk():
        if getattr(u, "mode", None) == "host":
            return u.host_xcel
    raise AssertionError("no mode='host' unit on the UPMEM target")


# --------------------------------------------------------------------- #
# host node + partial basis
# --------------------------------------------------------------------- #


def test_upmem_host_node_present_and_sibling():
    target = build_upmem_target()
    host = next(u for u in target._walk() if getattr(u, "mode", None) == "host")
    rank = next(u for u in target._walk() if u.name == "rank")
    # SIBLING of rank (both direct children of synthetic root) -> device
    # tree depth + get_uid() unchanged.
    assert host.parent is target.root
    assert rank.parent is target.root
    assert host.mapping == []  # no fan-out: contributes 1 to any product
    assert "host_dram" in host.memories


def test_upmem_partial_basis_covered_set():
    hx = _host_xcel(build_upmem_target())
    assert isinstance(hx, allo.HostXcel)
    assert hx.covered == {"scatter", "gather", "broadcast"}
    assert not hx.covers("reduce")  # no device reduce


def test_upmem_all_gather_covered_via_composition():
    """all_gather = gather + broadcast -> covered on a basis-only target."""
    hx = _host_xcel(build_upmem_target())
    assert hx.covers("all_gather")
    lowered = hx.lower("all_gather", buf=object(), over="dpu")
    assert [name for name, _ in lowered] == ["gather", "broadcast"]


def test_upmem_all_reduce_raises_coverage_error():
    """all_reduce needs a `reduce` basis UPMEM does not implement -> hard
    error naming the missing basis + the target."""
    hx = _host_xcel(build_upmem_target())
    assert not hx.covers("all_reduce")
    with pytest.raises(NotSupported) as exc:
        hx.require("all_reduce")
    msg = str(exc.value)
    assert "all_reduce" in msg
    assert "upmem" in msg
    assert "reduce" in msg  # names the missing basis primitive


def test_upmem_reduce_scatter_also_raises():
    hx = _host_xcel(build_upmem_target())
    with pytest.raises(NotSupported):
        hx.require("reduce_scatter")


def test_upmem_scatter_emit_shape():
    """Each covered primitive returns a `lambda ctx, t: ...` emit closure
    (design 05 §2 / §7: scatter -> dpu_prepare_xfer+dpu_push_xfer, gather ->
    dpu_copy_from, broadcast -> dpu_broadcast_to). Emit present, not wired."""
    hx = _host_xcel(build_upmem_target())
    for name in ("scatter", "gather", "broadcast"):
        emit = getattr(hx, name)(buf=object(), over="dpu")
        assert callable(emit)
        assert emit.__code__.co_argcount == 2  # (ctx, t)


# --------------------------------------------------------------------- #
# parity-neutral: host node adds 0 to UPMEM kernel_cycles
# --------------------------------------------------------------------- #


def test_upmem_host_node_is_cost_neutral():
    """The whole-program seam (kernel_cycles + host_staging) returns the SAME
    value as kernel_cycles alone for a UPMEM trace with no staging signal --
    the host_staging compose is 0, so existing UPMEM cost invariants hold."""
    from allo.spmw_cost_model import get_cost_model, ComposeCtx

    target = build_upmem_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    empty = MatchTrace(target_name="upmem", module_name="empty", matches=[])
    # empty trace cost stays 0 (the test_target_upmem invariant).
    assert cost_fn(empty, allo.Placement(placements={})) == 0
    # host_staging compose itself is 0 with no staging signal.
    hs = get_cost_model("upmem", "faithful", concern="host_staging")
    res = hs.compose(ComposeCtx(target, empty, allo.Placement(placements={})))
    assert res.cycles == 0
