# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 012 (re-scoped 005) -- Samsung host node, PARITY-NEUTRAL ADDITIVE.

Per design 05 §9.1 the host node is a PARALLEL ADDITIVE structure: the
`@allo.unit(mode="host")` node + `@allo.host_xcel` class with
broadcast/scatter/gather primitives are added as a SIBLING of the device
subtree, with emit closures present but NOT wired into the run path. The
`PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER` self-moves and `_samsung_compose_with`
are untouched, so the report-18 invariants (B=1 == 15251 single-GEMV cost,
B*=2 crossover, 3.93x asymptote) and `_samsung_workid_count` must NOT move.

This test pins both halves: (a) the host node + host-xcel build and cover
the three primitives; (b) the addition is parity-neutral.
"""
from __future__ import annotations

import allo
from allo.spmw_autoschedule import _bucket_for_autoschedule, _samsung_enumerate
from allo.spmw_cost_models import (
    _samsung_mk,
    _samsung_preload_cycles,
    _samsung_readback_cycles,
)
from allo.spmw_cost_tables import _samsung_workid_count
from allo.spmw_match import MatchTrace

from _fixtures import build_samsung_target
from test_samsung_batched_gemv import single_gemv_top, M, K


def _host_unit(target):
    for u in target._walk():
        if getattr(u, "mode", None) == "host":
            return u
    raise AssertionError("no mode='host' unit on the Samsung target")


# --------------------------------------------------------------------- #
# (a) host node + host-xcel build and cover the three primitives
# --------------------------------------------------------------------- #


def test_samsung_host_node_present():
    target = build_samsung_target()
    host = _host_unit(target)
    assert host.name == "host"
    assert host.mode == "host"
    assert host.mapping == []  # no fan-out -> contributes 1 to workid count
    # carries the host_dram staging memory
    assert "host_dram" in host.memories


def test_samsung_host_node_is_sibling_of_device_subtree():
    """The host node is a SIBLING of pseudo_channel (a direct child of the
    synthetic root), NOT its parent -- so the device tree depth and the
    get_uid() chains are byte-identical (design 05 §9.1 task 012)."""
    target = build_samsung_target()
    host = _host_unit(target)
    pch = next(u for u in target._walk() if u.name == "pseudo_channel")
    assert host.parent is target.root
    assert pch.parent is target.root
    # pseudo_channel -> pim depth unchanged (pim still 2 levels under root)
    pim = next(u for u in target._walk() if u.name == "pim")
    assert pim.parent is pch
    assert pim.level == 2


def test_samsung_host_xcel_covers_three_primitives():
    target = build_samsung_target()
    hx = _host_unit(target).host_xcel
    assert isinstance(hx, allo.HostXcel)
    assert hx.covers("broadcast")
    assert hx.covers("scatter")
    assert hx.covers("gather")
    # no reduce basis declared
    assert not hx.covers("reduce")
    assert hx.covered == {"broadcast", "scatter", "gather"}


def test_samsung_host_xcel_emit_shape():
    """Each primitive returns a `lambda ctx, t: ...` emit closure (the
    allo.move emit shape / design 05 §2). Emit is present but NOT wired."""
    target = build_samsung_target()
    hx = _host_unit(target).host_xcel
    for name in ("broadcast", "scatter", "gather"):
        emit = getattr(hx, name)(buf=object(), over="pseudo_channel")
        assert callable(emit)
        assert emit.__code__.co_argcount == 2  # (ctx, t)


# --------------------------------------------------------------------- #
# (b) parity-neutral: no number moved
# --------------------------------------------------------------------- #


def test_workid_count_unchanged():
    """The host node has no `mapping`, so `_samsung_workid_count`
    (product over every unit's mapping) is the device 16*8 = 128."""
    target = build_samsung_target()
    assert _samsung_workid_count(target) == 16 * 8


def test_preload_readback_anchors_untouched():
    """The 369/1/2/4096/181 calibration anchors still resolve through the
    UNTOUCHED PRELOAD_*/READBACK_* self-moves -> P=11368, R=181 @4096x1024."""
    target = build_samsung_target()
    sch = allo.customize(single_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    sub = MatchTrace(
        target_name=trace.target_name, module_name=trace.module_name, matches=matches
    )
    M_, K_ = _samsung_mk(target, sub)
    assert (M_, K_) == (M, K)
    assert _samsung_preload_cycles(target, M_, K_) == 11368
    assert _samsung_readback_cycles(target, M_) == 181


def test_single_gemv_b1_argmin_byte_identical():
    """The B=1 single-GEMV kernel_cycles candidate cost vector is
    byte-identical to the pre-012 device-only tree -- the host-node
    addition moved NO number.

    The full sorted cost vector (not just min) is pinned: the argmin
    (dual_fiber + host-residency) is 12037 and the worst candidate is
    535837 on this target. The report-18 NON-resident floor (P+E+R=15251)
    is asserted in the existing parity suite via `_best(resident=False)`
    filtering; here we prove the host node perturbs neither end of the
    distribution. (Confirmed equal with the host child removed from
    `root.children`: same (min, max, workids).)
    """
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    sch = allo.customize(single_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    sub = MatchTrace(
        target_name=trace.target_name, module_name=trace.module_name, matches=matches
    )
    cands = _samsung_enumerate(target, sub.matches)
    costs = sorted(cost_fn(sub, c) for c in cands)

    # Cross-check against the device-only tree (host sibling dropped):
    # the cost vector must be identical, proving parity-neutrality.
    bare = build_samsung_target()
    bare.root.children = [u for u in bare.root.children if u.mode != "host"]
    bare_cost_fn = allo.get_cost("kernel_cycles", bare)
    bare_costs = sorted(bare_cost_fn(sub, c) for c in _samsung_enumerate(bare, sub.matches))
    assert costs == bare_costs, (costs[:3], bare_costs[:3])
    assert min(costs) == 12037, min(costs)
    assert max(costs) == 535837, max(costs)
