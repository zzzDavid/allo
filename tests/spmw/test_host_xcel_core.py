# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 static assertions for the HostXcel core (design 05 §2, task 003).

Pins the interface contract the later phases build on: the basis/derived
vocabulary, @allo.host_xcel / @allo.primitive registration on a
mode="host" unit, the covered-set bookkeeping, the hard coverage error for
an uncovered collective, and default-composition lowering. No target
behavior is exercised here -- migration is Phase 2.
"""
from __future__ import annotations

import pytest

import allo
from allo.spmw_host import NotSupported


def _build_partial_target():
    """A target whose host-xcel covers broadcast/scatter/gather but NOT reduce."""
    calls = []

    @allo.target("toy_partial")
    def device():
        @allo.unit(mode="host")
        def host():  # noqa: D401
            @allo.host_xcel
            class hx(allo.HostXcel):
                @allo.primitive
                def broadcast(self, buf, *, over):
                    return lambda ctx, t: calls.append(("broadcast", over))

                @allo.primitive
                def scatter(self, buf, *, over):
                    return lambda ctx, t: calls.append(("scatter", over))

                @allo.primitive
                def gather(self, buf, *, over):
                    return lambda ctx, t: calls.append(("gather", over))

                # no @allo.primitive reduce -> reduce uncovered

        @allo.unit(mapping=[16])
        def pseudo_channel():
            pass

    return device, calls


def _host_xcel_of(target):
    for u in target._walk():
        if getattr(u, "mode", None) == "host":
            return u.host_xcel
    raise AssertionError("no host unit found")


def test_unit_mode_host_kwarg_additive():
    target, _ = _build_partial_target()
    host_units = [u for u in target._walk() if getattr(u, "mode", None) == "host"]
    assert len(host_units) == 1
    # device units default to mode=None (additive, no behavior change)
    pch = next(u for u in target._walk() if u.name == "pseudo_channel")
    assert pch.mode is None
    assert pch.mapping == [16]


def test_host_xcel_registers_and_records_covered():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    assert isinstance(hx, allo.HostXcel)
    assert hx.covered == {"broadcast", "scatter", "gather"}
    assert hx._target_name == "toy_partial"


def test_covers_basis_and_derived():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    # basis: only decorated ones covered
    assert hx.covers("broadcast")
    assert hx.covers("scatter")
    assert hx.covers("gather")
    assert not hx.covers("reduce")
    # derived all_gather = gather + broadcast -> covered
    assert hx.covers("all_gather")
    # derived all_reduce / reduce_scatter need reduce -> NOT covered
    assert not hx.covers("all_reduce")
    assert not hx.covers("reduce_scatter")


def test_uncovered_collective_hard_error_names_missing_basis():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    with pytest.raises(NotSupported) as exc:
        hx.require("all_reduce")
    msg = str(exc.value)
    assert "all_reduce" in msg
    assert "toy_partial" in msg
    assert "reduce" in msg  # names the missing basis primitive


def test_uncovered_basis_hard_error():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    with pytest.raises(NotSupported) as exc:
        hx.require("reduce")
    assert "reduce" in str(exc.value)
    assert "toy_partial" in str(exc.value)


def test_default_composition_lowers_to_basis_in_order():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    lowered = hx.lower("all_gather", buf=object(), over="PCH")
    # all_gather = gather then broadcast (design 05 §2 order)
    assert [name for name, _ in lowered] == ["gather", "broadcast"]


def test_basis_collective_lowers_to_single_emit():
    target, _ = _build_partial_target()
    hx = _host_xcel_of(target)
    lowered = hx.lower("scatter", buf=object(), over="PCH")
    assert [name for name, _ in lowered] == ["scatter"]


def test_host_xcel_outside_host_unit_raises():
    with pytest.raises(RuntimeError):

        @allo.target("toy_bad")
        def device():
            @allo.unit(mapping=[16])  # NOT a host unit
            def pseudo_channel():
                @allo.host_xcel
                class hx(allo.HostXcel):
                    @allo.primitive
                    def broadcast(self, buf, *, over):
                        return lambda ctx, t: None


def test_host_xcel_requires_subclass():
    with pytest.raises(TypeError):

        @allo.target("toy_bad2")
        def device():
            @allo.unit(mode="host")
            def host():
                @allo.host_xcel
                class hx:  # not a HostXcel subclass
                    pass


def test_reduce_override_forwards_op():
    seen = {}

    @allo.target("toy_reduce")
    def device():
        @allo.unit(mode="host")
        def host():
            @allo.host_xcel
            class hx(allo.HostXcel):
                @allo.primitive
                def broadcast(self, buf, *, over):
                    return lambda ctx, t: None

                @allo.primitive
                def reduce(self, buf, *, over, op):
                    seen["op"] = op
                    return lambda ctx, t: None

    hx = _host_xcel_of(device)
    assert hx.covers("reduce")
    assert hx.covers("all_reduce")  # reduce + broadcast both covered now
    hx.lower("reduce", buf=object(), over="PCH", op=allo.host_cpu)
    assert seen["op"] is allo.host_cpu


def test_init_exports():
    assert allo.HostXcel is not None
    assert callable(allo.host_xcel)
    assert callable(allo.primitive)
    assert allo.NotSupported is NotSupported
    assert allo.host_cpu is not None
