# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM token (reduction-axis) partitioning: paper 5-evaluation.tex "Token
Partitioning" as a compiler mechanism.

The ``reduction_partitions`` knob splits a contraction's reduction over P
channel groups (``docs/release/design/aim-token-partition.md``).  The
default tests are compile-only: knob domain, lowering, the explicit host
reduction step, its price and the numerical check of the partial-sum merge.
``test_nine_workload_selection`` (``-m paper_full``, needs the simulator)
reproduces the paper's experiment structure; ``python -m
tests.pim.test_aim_token_partition`` prints its table.

The workloads are defined here rather than in ``benchmarks/cent_aim``: their
region annotations must be evaluated eagerly (``workloads.py`` uses
postponed annotations).
"""

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.aim_lowering import (
    AimHostReduction,
    _target_geometry,
    bf16_round,
    reference_partial_sums,
)
from allo.pim.costs import aim_cost
from allo.pim.costs.aim import host_reduction_cycles
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import (
    _aim_enumerate,
    _bucket_for_autoschedule,
    _matcher_work_scope,
)
from allo.spmw_knobs import KnobCtx, _aim_reduction_partitions_candidates

# The token split reassociates a floating-point reduction: compile under the
# explicit permission when this build has the reduction-order policy.
_POLICY = (
    {"reduction_order": "reassociate"}
    if "reduction_order" in inspect.signature(allo.compile).parameters
    else {}
)

HEAD_DIM = 128
MAX_TOKENS = 4096
CHANNEL_BUDGET = 8  # the paper's eight simulated channels
PARTITIONS = (1, 2, 4, 8)
SHAPES = tuple((heads, tokens) for heads in (1, 8, 32) for tokens in (128, 512, 4096))


def build_attention(heads, tokens, head_dim=HEAD_DIM, max_tokens=MAX_TOKENS):
    """Resident-input decode attention: scores = q.K, heads_out = probs.V.

    The host softmax between the kernels is identical for every partition
    choice and is not a compiler-priced stage; ``probs`` is resident.
    """
    H, L, Dh, M = heads, tokens, head_dim, max_tokens

    @region()
    def attention(
        q: bfloat16[H, Dh],
        k_cache: bfloat16[H, M, Dh],
        scores: bfloat16[H, L],
        probs: bfloat16[H, L],
        v_cache: bfloat16[H, Dh, M],
        heads_out: bfloat16[H, Dh],
    ):
        @allo.work(mapping=[1], args=[q, k_cache, scores])
        def attention_qk(
            lq: bfloat16[H, Dh], kc: bfloat16[H, M, Dh], sc: bfloat16[H, L]
        ):
            for h in range(H):
                for s in range(L):
                    acc: bfloat16 = 0
                    for d in range(Dh):
                        acc += lq[h, d] * kc[h, s, d]
                    sc[h, s] = acc

        @allo.work(mapping=[1], args=[probs, v_cache, heads_out])
        def attention_sv(
            pr: bfloat16[H, L], vc: bfloat16[H, Dh, M], o: bfloat16[H, Dh]
        ):
            for h in range(H):
                for d in range(Dh):
                    acc: bfloat16 = 0
                    for s in range(L):
                        acc += pr[h, s] * vc[h, d, s]
                    o[h, d] = acc

    def host(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(
        attention, resident=("q", "k_cache", "probs", "v_cache")
    )(host)
    return attention, program


def build_sv(heads, tokens, head_dim=HEAD_DIM, max_tokens=MAX_TOKENS, append=False):
    """SV alone; ``append`` adds the decode step's value-cache append (CENT's
    broadcast of token L-1) before the launch."""
    H, L, Dh, M = heads, tokens, head_dim, max_tokens

    @region()
    def attention(
        probs: bfloat16[H, L], v_cache: bfloat16[H, Dh, M], heads_out: bfloat16[H, Dh]
    ):
        @allo.work(mapping=[1], args=[probs, v_cache, heads_out])
        def attention_sv(
            pr: bfloat16[H, L], vc: bfloat16[H, Dh, M], o: bfloat16[H, Dh]
        ):
            for h in range(H):
                for d in range(Dh):
                    acc: bfloat16 = 0
                    for s in range(L):
                        acc += pr[h, s] * vc[h, d, s]
                    o[h, d] = acc

    def host(probs, v_cache, heads_out):
        if append:
            hx = allo.host_xfer
            hx.broadcast(v_cache[:, :, L - 1], hx.banks)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    # The cache before this decode step is on the device in both forms: the
    # append writes token L-1 into a resident cache the kernel then reads
    # over [0, L) (availability is explicit, review round 22).
    resident = ("probs", "v_cache")
    return attention, allo.host_program(attention, resident=resident)(host)


# --------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------- #


def _knob(mode, name, default=None):
    marker = f"+{name}="
    if marker not in mode:
        return default
    return mode.split(marker, 1)[1].split("+", 1)[0]


def _partitions(mode):
    return int(_knob(mode, "reduction_partitions", 1))


def _within_budget(mode, budget=CHANNEL_BUDGET):
    return int(_knob(mode, "replica_partitions", 1)) <= budget


def _ranked(heads, tokens, *, qk=True):
    if qk:
        attention, program = build_attention(heads, tokens)
    else:
        attention, program = build_sv(heads, tokens)
    target = build_aim_target()
    compiled = allo.compile(
        attention, target, aim_cost, backend="virtual", host_moves=program, **_POLICY
    )
    return target, attention, program, compiled


def _kernel_choices(target, compiled):
    """``{kernel name: (bucket index, [(cycles, mode, placement)])}``."""
    trace = compiled.compiled.trace
    buckets = _bucket_for_autoschedule(trace)
    out = {}
    for kernel in compiled.compiled.placement_ranking.kernels:
        index, matches = next(
            (i, m)
            for i, (_scope, m) in enumerate(buckets)
            if _matcher_work_scope(m[0]).group_id == kernel.group_id
        )
        by_mode = {c.mode: c for c in _aim_enumerate(target, matches)}
        name = matches[0].func_name.rsplit("_", 1)[0]
        out[name] = (
            index,
            [(cycles, mode, by_mode[mode]) for _pos, cycles, mode in kernel.scores],
        )
    return out


def budget_variants(target, compiled, budget=CHANNEL_BUDGET):
    """The cost program's best placement per decision inside the channel
    budget: the QK choice, the SV choice for each P, and the two controls."""
    choices = _kernel_choices(target, compiled)
    qk_index, qk_scores = choices.get("attention_qk", (None, []))
    sv_index, sv_scores = choices["attention_sv"]
    qk = min((s for s in qk_scores if _within_budget(s[1], budget)), default=None)
    sv = [s for s in sv_scores if _within_budget(s[1], budget)]
    per_partition = {}
    for cycles, mode, placement in sv:
        partitions = _partitions(mode)
        if partitions in PARTITIONS and (
            partitions not in per_partition or cycles < per_partition[partitions][0]
        ):
            per_partition[partitions] = (cycles, mode, placement)
    head_only = min(
        (
            s
            for s in sv
            if _partitions(s[1]) == 1 and _knob(s[1], "batch_mapping") == "channels"
        ),
        key=lambda s: s[0],
    )
    selected = min(per_partition.values(), key=lambda s: s[0])
    return SimpleNamespace(
        qk=qk,
        qk_index=qk_index,
        sv_index=sv_index,
        per_partition=per_partition,
        head_only=head_only,
        fixed_eight=per_partition.get(8),
        selected_partitions=_partitions(selected[1]),
    )


def compile_variant(attention, program, target, variants, sv_choice):
    layouts = [None, None]
    layouts[variants.sv_index] = sv_choice[2]
    if variants.qk is not None:
        layouts[variants.qk_index] = variants.qk[2]
    layouts = [layout for layout in layouts if layout is not None]
    return allo.compile(
        attention, target, aim_cost, host_moves=program, layout=layouts, **_POLICY
    )


def _sv_entry(compiled):
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    return next(
        entry
        for entry in manifest
        if entry["kind"] == "contraction"
        and str(entry["name"]).startswith("attention_sv")
    )


def check_merge_numerics(compiled, heads, tokens, seed=0):
    """Check the lowered partition ownership and the host merge on integer
    (exact) and BF16 operands; returns the BF16 max relative error."""
    entry = _sv_entry(compiled)
    split = entry.get("reduction_partition")
    if split is None:
        return None
    host = next(
        step
        for step in compiled.compiled.layout_ctx.host_steps
        if step.source == entry["name"]
    )
    rng = np.random.default_rng(seed + heads + tokens)
    # Integer operands: exact partials and exact merge.
    values = rng.integers(-16, 17, (heads, HEAD_DIM, tokens))
    probs = rng.integers(0, 9, (heads, tokens))
    partials = reference_partial_sums(entry, values, probs, arithmetic="exact")
    merged = host.merge(partials)[0]
    np.testing.assert_array_equal(merged, np.einsum("hds,hs->hd", values, probs))
    # BF16 operands under the declared readout contract.
    values = bf16_round(
        rng.standard_normal((heads, HEAD_DIM, tokens)).astype(np.float32)
    )
    probs = bf16_round(rng.random((heads, tokens)).astype(np.float32))
    merged = host.merge(reference_partial_sums(entry, values, probs))[0]
    reference = np.einsum(
        "hds,hs->hd", values.astype(np.float64), probs.astype(np.float64)
    )
    scale = np.max(np.abs(reference))
    error = float(np.max(np.abs(merged - reference)) / scale)
    # BF16 readout per row chunk: a few units of the 2^-8 BF16 epsilon.
    assert error < 2.0**-6, error
    return error


# --------------------------------------------------------------------- #
# Compile-only tests
# --------------------------------------------------------------------- #


def test_knob_domain_and_reduction_policy():
    target, _attention, _program, compiled = _ranked(1, 512, qk=False)
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


def run_workload(heads, tokens, *, simulate=True):
    """Compile and (optionally) simulate every variant of one workload."""
    target, attention, program, ranked = _ranked(heads, tokens)
    variants = budget_variants(target, ranked)
    geometry = _target_geometry(target)
    rows = {}
    labelled = [
        (f"P{p}", choice) for p, choice in sorted(variants.per_partition.items())
    ]
    if variants.head_only[1] != variants.per_partition[1][1]:
        labelled.append(("head-only", variants.head_only))
    for label, choice in labelled:
        built = compile_variant(attention, program, target, variants, choice)
        host = sum(
            host_reduction_cycles(step, geometry)
            for step in built.compiled.layout_ctx.host_steps
        )
        estimate = (
            allo.compile(
                attention,
                target,
                aim_cost,
                backend="virtual",
                host_moves=program,
                layout=built.compiled.layout,
                **_POLICY,
            )
            .run_backend()
            .cycles
        )
        device = built.run_backend().cycles if simulate else None
        rows[label] = SimpleNamespace(
            mode=choice[1],
            partitions=_partitions(choice[1]),
            estimate=estimate,
            device=device,
            host=host,
            total=None if device is None else device + host,
            bf16_error=check_merge_numerics(built, heads, tokens),
        )
    if "head-only" not in rows:
        rows["head-only"] = rows["P1"]
    rows["fixed-8"] = rows["P8"]
    best = min((rows[f"P{p}"] for p in PARTITIONS), key=lambda r: r.total or r.estimate)
    chosen = rows[f"P{variants.selected_partitions}"]
    return SimpleNamespace(
        heads=heads,
        tokens=tokens,
        rows=rows,
        selected=variants.selected_partitions,
        best=best.partitions,
        gap=(chosen.total / best.total - 1.0) if simulate else None,
    )


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


def main():  # pragma: no cover - reproduction entry point
    import sys

    simulate = "--no-sim" not in sys.argv
    print(
        "heads tokens | P1 P2 P4 P8 (device+host) | head-only | "
        "selected P | best P | gap | merge bf16 err"
    )
    reached = 0
    for heads, tokens in SHAPES:
        result = run_workload(heads, tokens, simulate=simulate)
        rows = result.rows
        totals = " ".join(
            f"{rows[f'P{p}'].total if simulate else rows[f'P{p}'].estimate}"
            f"({rows[f'P{p}'].estimate})"
            for p in PARTITIONS
        )
        errors = max((r.bf16_error or 0.0) for r in rows.values())
        reached += result.selected == result.best
        print(
            f"{heads:5d} {tokens:6d} | {totals} | "
            f"{rows['head-only'].total if simulate else rows['head-only'].estimate} | "
            f"{result.selected} | {result.best} | "
            f"{(result.gap or 0.0) * 100:.2f}% | {errors:.2e}",
            flush=True,
        )
    print(f"selection reaches {reached} of {len(SHAPES)} minima")


if __name__ == "__main__":  # pragma: no cover
    main()
