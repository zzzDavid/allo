# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM matcher-path knobs: registration, realizability and cost ranking."""

from __future__ import annotations

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import (
    _MATCHER_PHYSICAL_EXTRA_FIELDS,
    _aim_enumerate,
    _bucket_for_autoschedule,
)
from allo.spmw_codegen import compile_for_target
from allo.spmw_knobs import registered_knobs

AIM_KNOBS = ("bank_scope", "batch_mapping", "reuse_group", "replica_partitions")

H, DH, L = 32, 128, 512


@region()
def attention_qk(q: bfloat16[H, DH], k_cache: bfloat16[H, L, DH], sc: bfloat16[H, L]):
    @allo.work(mapping=[4], args=[q, k_cache, sc])
    def qk(
        lq: bfloat16[H, DH],
        lk: bfloat16[H, L, DH],
        ls: bfloat16[H, L],
    ):
        for h in range(H):
            for s in range(L):
                acc: bfloat16 = 0
                for d in range(DH):
                    acc += lq[h, d] * lk[h, s, d]
                ls[h, s] = acc


def _chosen(compiled):
    (kernel,) = compiled.placement_ranking.kernels
    return {position: mode for position, _c, mode in kernel.scores}[
        kernel.chosen_position
    ]


def _knob(mode, name):
    return mode.split(f"+{name}=", 1)[1].split("+", 1)[0]


def _wr_gb(compiled):
    return sum(1 for line in compiled.compiled.cmds if line.startswith("AiM WR_GB "))


def test_aim_knobs_registered_logged_and_realizable(monkeypatch):
    target = build_aim_target()
    assert tuple(knob.name for knob in registered_knobs("aim")) == AIM_KNOBS
    assert set(AIM_KNOBS) <= set(_MATCHER_PHYSICAL_EXTRA_FIELDS)

    compiled = allo.compile(attention_qk, target, aim_cost, backend="virtual")
    (kernel,) = compiled.placement_ranking.kernels
    modes = [mode for _p, _c, mode in kernel.scores]
    for mode in modes:
        assert all(f"+{name}=" in mode for name in AIM_KNOBS), mode
    assert {_knob(mode, "bank_scope") for mode in modes} == {"sbk", "abk"}
    assert {_knob(mode, "batch_mapping") for mode in modes} == {
        "row_packed",
        "channels",
    }
    assert {_knob(mode, "replica_partitions") for mode in modes} == {
        "1",
        "2",
        "4",
        "8",
    }

    # Every enumerated value combination must lower (spec 001 D5 rule 2).
    trace = compiled.compiled.trace
    module = compiled.schedule.module
    buckets = _bucket_for_autoschedule(trace)
    candidates = _aim_enumerate(target, buckets[0][1])
    assert len(candidates) == len(modes)
    for candidate in candidates:
        result = compile_for_target(
            target, trace, layout=[candidate] * len(buckets), module=module
        )
        assert result.cmds[-1] == "AiM EOC"

    # A combination the cost rule finds unrealizable is dropped, not fatal.
    import allo.pim.costs.aim as aim_costs
    from allo.pim.schedule_search import InfeasibleSchedule

    profile = aim_costs._contraction_profile_cycles

    def reject_row_packed(candidate, geometry):
        if candidate["batch_mapping"] == "row_packed":
            raise InfeasibleSchedule("injected")
        return profile(candidate, geometry)

    monkeypatch.setattr(aim_costs, "_contraction_profile_cycles", reject_row_packed)
    skipped = allo.compile(attention_qk, target, aim_cost, backend="virtual")
    assert _knob(_chosen(skipped), "batch_mapping") == "channels"


def test_aim_row_activation_cost_flips_batch_mapping(monkeypatch):
    target = build_aim_target()
    compiled = allo.compile(attention_qk, target, aim_cost, backend="virtual")
    assert _knob(_chosen(compiled), "batch_mapping") == "row_packed"
    assert _wr_gb(compiled) == 16
    fingerprint = aim_cost.bind(target).fingerprint

    monkeypatch.setattr(
        "allo.pim.costs.aim.BATCH_MAPPING_ROW_ACTIVATE_CYCLES", 0, raising=True
    )
    assert aim_cost.bind(target).fingerprint != fingerprint
    flipped = allo.compile(attention_qk, target, aim_cost, backend="virtual")
    assert _knob(_chosen(flipped), "batch_mapping") == "channels"
    assert _wr_gb(flipped) == 4
    macs = lambda c: sum(1 for x in c.compiled.cmds if x.startswith("AiM MAC_"))
    assert macs(flipped) == macs(compiled)
