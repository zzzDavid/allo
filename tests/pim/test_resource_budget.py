# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resource budget (``allo.compile(..., resources={"channel": 8})``): a
search policy the enumerators, the ranking, the realization check and the
result provenance all read, never a device description."""

from __future__ import annotations

import pytest

import allo
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_resources import normalize_resource_budget
from benchmarks.cent_aim.token_partition import (
    POLICY,
    build_attention,
    kernel_choices,
    knob,
    partitions_of,
)

BUDGET = {"channel": 8}


def _layouts(compiled):
    stored = compiled.compiled.layout
    return list(stored) if isinstance(stored, (list, tuple)) else [stored]


def _channel_use(compiled):
    return {
        entry["name"]: entry["active_channel_count"]
        for entry in compiled.compiled.layout_ctx.lowering_manifest
        if isinstance(entry, dict) and "active_channel_count" in entry
    }


def test_budget_is_validated_against_the_target():
    target = build_aim_target()
    assert normalize_resource_budget(None) is None
    assert normalize_resource_budget({"channel": 8}, target) == {"channel": 8}
    for bad, reason in (
        ({"channel": 64}, "exceeds the 32 channel"),
        ({"nope": 1}, "does not declare"),
        ({"channel": 0}, "positive integer"),
        ({"channel": True}, "positive integer"),
        ("channel", "mapping"),
    ):
        with pytest.raises(ValueError, match=reason):
            normalize_resource_budget(bad, target)


def test_budget_caps_the_partition_domains_and_the_selected_program():
    target = build_aim_target()
    attention, program = build_attention(1, 512)
    free = allo.compile(
        attention, target, aim_cost, backend="virtual", host_moves=program, **POLICY
    )
    capped = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    for compiled, limit in ((free, 32), (capped, 8)):
        for _name, (_index, scores) in kernel_choices(target, compiled).items():
            replicas = {int(knob(mode, "replica_partitions", 1)) for _c, mode, _p in scores}
            parts = {partitions_of(mode) for _c, mode, _p in scores}
            assert max(replicas) == limit and max(parts) == limit
    # The unbudgeted default occupies the whole device; the budgeted one
    # selects within the cap, and the cap is on every result.
    assert any(int(knob(p.mode, "replica_partitions", 1)) > 8 for p in _layouts(free))
    assert all(int(knob(p.mode, "replica_partitions", 1)) <= 8 for p in _layouts(capped))
    assert capped.run_backend().extra["resource_budget"] == BUDGET
    assert free.run_backend().extra["resource_budget"] is None
    assert capped.resources == BUDGET
    # The lowered program (the simulator route) occupies at most the cap on
    # every record; the unbudgeted one does not.
    lowered = allo.compile(
        attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY
    )
    assert lowered.compiled.layout_ctx.resource_budget == BUDGET
    use = _channel_use(lowered)
    assert use and max(use.values()) <= 8, use
    unbudgeted = allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)
    assert max(_channel_use(unbudgeted).values()) > 8
    # A placement selected without the budget designates no candidate of the
    # budgeted enumeration: refused, not silently narrowed.
    with pytest.raises(ValueError, match="resource budget"):
        allo.compile(
            attention,
            target,
            aim_cost,
            backend="virtual",
            host_moves=program,
            layout=_layouts(free),
            resources=BUDGET,
            **POLICY,
        )


def test_realization_check_reads_the_budget(monkeypatch):
    """The realization check reads the context's budget: a lowering that
    occupies more channels than the cap is a feasibility rejection the
    ranking records, so the search moves to programs that fit (here a cap
    of one channel at the realization boundary selects one-channel programs
    where the budgeted enumeration alone would have selected 8 and 4)."""
    from allo import spmw_aim

    target = build_aim_target()
    attention, program = build_attention(1, 512)
    original = spmw_aim.AimCtx.check_realization
    refusals = []

    def tightened(self, layouts):
        self.resource_budget = {"channel": 1}
        try:
            return original(self, layouts)
        except RuntimeError as error:
            refusals.append(str(error))
            raise

    monkeypatch.setattr(spmw_aim.AimCtx, "check_realization", tightened)
    compiled = allo.compile(
        attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY
    )
    assert refusals and all("resource budget allows 1" in r for r in refusals)
    assert max(_channel_use(compiled).values()) == 1
