# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM token (reduction-axis) partitioning: paper 5-evaluation.tex "Token
Partitioning" as a compiler mechanism.

The ``reduction_partitions`` knob splits a contraction's reduction over P
channel groups (``docs/release/design/aim-token-partition.md``).  The
default tests are compile-only: knob domain, lowering, the explicit host
reduction step, its price and the numerical check of the partial-sum merge.
``test_nine_workload_selection`` (``-m paper_full``, needs the simulator)
runs the paper's experiment through the maintained driver
``benchmarks/cent_aim/token_partition.py`` (``python -m
benchmarks.cent_aim.token_partition --out tp.json`` prints its table and
writes the record), which also defines the workloads (their region
annotations are evaluated eagerly).
"""

import math

import pytest

import allo
from allo.pim.aim_lowering import AimHostReduction, _target_geometry
from allo.pim.costs import aim_cost
from allo.pim.costs.aim import host_reduction_cycles
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import _aim_enumerate, _bucket_for_autoschedule
from allo.spmw_knobs import KnobCtx, _aim_reduction_partitions_candidates
from benchmarks.cent_aim.token_partition import (
    HEAD_DIM,
    PARTITIONS,
    POLICY as _POLICY,
    SHAPES,
    build_sv,
    check_merge_numerics,
    kernel_choices as _kernel_choices,
    knob as _knob,
    partitions_of as _partitions,
    ranked as _ranked,
    run_workload,
    sv_entry as _sv_entry,
    within_budget as _within_budget,
)

# --------------------------------------------------------------------- #
# Compile-only tests
# --------------------------------------------------------------------- #


def test_knob_domain_and_reduction_policy():
    target, _attention, _program, compiled = _ranked(1, 512, qk=False, budget=None)
    choices = _kernel_choices(target, compiled)
    modes = [mode for _c, mode, _p in choices["attention_sv"][1]]
    found = {_partitions(mode) for mode in modes}
    assert found == {1, 2, 4, 8, 16, 32}
    for mode in modes:
        partitions = _partitions(mode)
        if partitions > 1:
            assert _knob(mode, "batch_mapping") == "channels"
            assert int(_knob(mode, "replica_partitions")) >= partitions

    # The knob follows the reduction-order policy: "reassociate" permits the
    # split, a strict policy on the BF16 accumulator (also the reading of an
    # unstamped match) leaves only P = 1.
    from allo.spmw_reduction_policy import POLICY_KEY, ReductionPolicy

    (_scope, matches) = _bucket_for_autoschedule(compiled.compiled.trace)[0]
    base = next(
        c
        for c in _aim_enumerate(target, matches)
        if _knob(c.mode, "batch_mapping") == "channels" and _partitions(c.mode) == 1
    )
    ctx = KnobCtx(target=target, matches=matches, role_to_memref={}, base=base)
    saved = [m.extra.get(POLICY_KEY) for m in matches]
    try:
        for m in matches:
            m.extra[POLICY_KEY] = ReductionPolicy("reassociate", "bf16")
        assert _aim_reduction_partitions_candidates(ctx)[-1] > 1
        for m in matches:
            m.extra[POLICY_KEY] = ReductionPolicy("strict", "bf16")
        assert _aim_reduction_partitions_candidates(ctx) == [1]
        for m in matches:
            m.extra.pop(POLICY_KEY, None)
        assert _aim_reduction_partitions_candidates(ctx) == [1]
        for m in matches:
            m.extra[POLICY_KEY] = ReductionPolicy("strict", "i32")
        assert _aim_reduction_partitions_candidates(ctx)[-1] > 1
    finally:
        for m, policy in zip(matches, saved):
            if policy is None:
                m.extra.pop(POLICY_KEY, None)
            else:
                m.extra[POLICY_KEY] = policy


@pytest.mark.parametrize("partitions", (2, 4, 8))
def test_partitioned_program_host_step_cost_and_numerics(partitions):
    heads, tokens = 2, 512
    target, attention, program, compiled = _ranked(heads, tokens, qk=False)
    _index, scores = _kernel_choices(target, compiled)["attention_sv"]
    cycles, mode, placement = min(
        (s for s in scores if _partitions(s[1]) == partitions), key=lambda s: s[0]
    )
    # The decision is carried by the LinearLayout: reduction bit i < log2(P)
    # selects partition bit i (element t -> partition t % P).
    layout = placement.layout
    assert layout.out_dims[-1] == "token_partition"
    assert [layout.apply(k=t)[-1] for t in range(2 * partitions)] == [
        t % partitions for t in range(2 * partitions)
    ]

    built = allo.compile(
        attention, target, aim_cost, host_moves=program, layout=[placement], **_POLICY
    )
    ctx = built.compiled.layout_ctx
    (step,) = ctx.host_steps
    assert isinstance(step, AimHostReduction)
    assert (step.partitions, step.outputs, step.batches) == (
        partitions,
        HEAD_DIM,
        heads,
    )
    entry = _sv_entry(built)
    host_entry = next(e for e in ctx.lowering_manifest if e["kind"] == "hostreduction")
    assert host_entry["command_count"] == 0 and host_entry["executes_on"] == "host"
    assert entry["reduction_partition"]["local_reduction"] == tokens // partitions
    # Every (head, partition) accumulator owns one channel of the group.
    channels = {c for group in entry["groups"] for c in group["active_channels"]}
    assert len(channels) == min(
        heads * partitions, int(_knob(mode, "replica_partitions"))
    )
    assert entry["program_scalar_macs"] == heads * HEAD_DIM * tokens

    # The cost program prices the host step with the declared link model and
    # the virtual estimate equals the ranked score of this placement.
    geometry = _target_geometry(target)
    vectors = math.ceil(HEAD_DIM / geometry.lanes) * heads
    expected = vectors * (partitions + 1) * 32 // 32 + vectors * (partitions - 1)
    assert host_reduction_cycles(step, geometry) == expected
    virtual = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        layout=[placement],
        **_POLICY,
    )
    assert virtual.run_backend().cycles == cycles
    graph = virtual.compiled.execution_graph
    assert [
        a.latency_cycles for a in graph.activities if a.label == "host_reduction"
    ] == [expected]

    error = check_merge_numerics(built, heads, tokens)
    assert error is not None


def test_cache_append_into_partitioned_layout():
    heads, tokens, partitions = 2, 512, 4
    attention, program = build_sv(heads, tokens, append=True)
    target = build_aim_target()
    ranked = allo.compile(
        attention, target, aim_cost, backend="virtual", host_moves=program, **_POLICY
    )
    (_index, scores) = _kernel_choices(target, ranked)["attention_sv"]
    _c, mode, placement = min(
        (s for s in scores if _partitions(s[1]) == partitions and _within_budget(s[1])),
        key=lambda s: s[0],
    )
    built = allo.compile(
        attention, target, aim_cost, host_moves=program, layout=[placement], **_POLICY
    )
    channels = target.unit("channel").axes["channel"]
    writes = [
        line.split() for line in built.compiled.cmds if line.startswith("AiM WR_ABK")
    ]
    owners = {channels - 1 - int(math.log2(int(fields[3]))) for fields in writes}
    # Token L-1 lives in partition (L-1) % P: only that partition's channels
    # receive it, at local row (L-1) // P // row_elements.
    cpr = int(_knob(mode, "replica_partitions"))
    position = tokens - 1
    assert owners == {
        c
        for c in range(min(cpr, heads * partitions))
        if c % partitions == position % partitions
    }
    assert writes


# --------------------------------------------------------------------- #
# The paper's nine-workload experiment (simulator)
# --------------------------------------------------------------------- #


@pytest.mark.paper_full
def test_nine_workload_selection(aim_sim_gate):
    results = [run_workload(heads, tokens) for heads, tokens in SHAPES]
    for result in results:
        for row in result.rows.values():
            assert row.device > 0
        # The selection never loses to the controls by more than the gap
        # it reports; the count of reached minima is reported, not forced.
        assert result.gap >= 0.0
    reached = sum(result.selected == result.best for result in results)
    print(f"cost-selected P reaches {reached} of {len(results)} simulated minima")
