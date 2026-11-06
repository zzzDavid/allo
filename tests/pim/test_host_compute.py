# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A declared host computation between launches (``allo.host_compute``):
recorded in the host program, ordered in the launch schedule with its
declared volumes, required by the availability check to read what an
earlier launch produced and allowed to feed a later launch's host ingress,
represented and priced in the AiM program (the HOST_COMPUTE rule: readout
and upload link bytes plus the declared host passes), bound by the
realization check, refused on backends that do not represent it, and
executed numerically by the functional oracle pipeline (QK -> global
softmax -> SV) for split and unsplit plans."""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.pim.aim_lowering import (
    AimHostStage,
    _target_geometry,
    bf16_round,
    reference_outputs,
)
from allo.pim.costs import aim_cost
from allo.pim.costs.aim import (
    HOST_LINK_BYTES_PER_CYCLE,
    HOST_OPERATION_PASSES,
    HOST_VECTOR_ADD_CYCLES,
    host_stage_cycles,
    host_stage_terms,
)
from allo.pim.targets import build_aim_target, build_samsung_target
from allo.spmw_plan import HostComputeStep, LaunchStep
from benchmarks.cent_aim.token_partition import (
    BUDGET,
    POLICY,
    ScaledSoftmax,
    build_attention,
    functional_check,
    kernel_choices,
    partitions_of,
)


def _layouts(compiled):
    stored = compiled.compiled.layout
    return list(stored) if isinstance(stored, (list, tuple)) else [stored]


def test_host_compute_is_recorded_scheduled_and_declared():
    attention, program = build_attention(1, 128)
    assert [type(s).__name__ for s in program.steps] == [
        "LaunchRecord",
        "HostComputeRecord",
        "LaunchRecord",
    ]
    (stage,) = program.computes
    assert stage.name == "softmax" and stage.operation == "softmax"
    assert isinstance(stage.fn, ScaledSoftmax) and stage.fn.head_dim == 128
    compiled = allo.compile(
        attention,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        host_moves=program,
        **POLICY,
    )
    steps = compiled.compiled.launch_schedule.steps
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["LaunchStep", "HostComputeStep", "LaunchStep"]
    step = steps[1]
    assert isinstance(step, HostComputeStep)
    assert step.inputs == (("scores", (1, 128), 16),) and step.outputs == (
        ("probs", (1, 128), 16),
    )
    assert (
        step.volume("inputs") == 256
        and step.volume("outputs") == 256
        and step.elements() == 128
    )
    assert step.function is stage.fn and step.operation == "softmax"
    manifest = compiled.compiled.launch_schedule.manifest()
    assert (
        "host_compute",
        "softmax",
        "softmax",
        (("scores", (1, 128), 16),),
        (("probs", (1, 128), 16),),
    ) in manifest


def test_host_compute_rejects_bad_declarations():
    from allo.spmw_target import BufferToken, HostComputeRecord

    fn = ScaledSoftmax(128)
    with pytest.raises(RuntimeError, match="outside a host_program"):
        allo.host_compute(
            "softmax",
            fn,
            inputs=(BufferToken("a"),),
            outputs=(BufferToken("b"),),
            operation="softmax",
        )
    with pytest.raises(ValueError, match="declares no outputs"):
        HostComputeRecord("softmax", fn, (BufferToken("a"),), (), "softmax")
    with pytest.raises(TypeError, match="whole workload buffers"):
        HostComputeRecord(
            "softmax", fn, (BufferToken("a")[0],), (BufferToken("b"),), "softmax"
        )
    with pytest.raises(TypeError, match="callable"):
        HostComputeRecord(
            "softmax", None, (BufferToken("a"),), (BufferToken("b"),), "softmax"
        )
    # The semantic operation comes from the declared domain, never the label.
    with pytest.raises(ValueError, match="not a declared host operation"):
        HostComputeRecord(
            "softmax", fn, (BufferToken("a"),), (BufferToken("b"),), "normalization"
        )
    with pytest.raises(TypeError):
        HostComputeRecord("softmax", fn, (BufferToken("a"),), (BufferToken("b"),))


def test_stage_must_read_what_an_earlier_launch_produced():
    """The schedule's availability walk: a stage placed before the launch
    that produces its input is refused; placed after, its output reaches the
    consumer through the consumer's own host ingress (the GB vector preload)."""
    from allo.pim.schedule_search import InfeasibleSchedule

    attention, _program = build_attention(1, 128)

    def early(q, k_cache, scores, probs, v_cache, heads_out):
        allo.host_compute(
            "softmax",
            ScaledSoftmax(128),
            inputs=(scores,),
            outputs=(probs,),
            operation="softmax",
        )
        allo.launch("attention_qk", q, k_cache, scores)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(early)
    with pytest.raises(
        (InfeasibleSchedule, RuntimeError),
        match="before the launch that produces it|no earlier launch",
    ):
        allo.compile(
            attention,
            build_aim_target(),
            aim_cost,
            backend="virtual",
            host_moves=program,
            **POLICY,
        )


def test_other_backends_refuse_a_host_stage():
    attention, program = build_attention(1, 128)
    with pytest.raises(
        NotImplementedError, match="does not represent declared host stages"
    ):
        allo.compile(
            attention,
            build_samsung_target(),
            None,
            backend="virtual",
            host_moves=program,
            layout="bank_row",
        )


def test_stage_is_priced_in_the_selection_graph_and_lowered():
    target = build_aim_target()
    geometry = _target_geometry(target)
    attention, program = build_attention(1, 512)
    virtual = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    graph = virtual.compiled.execution_graph
    (stage_activity,) = [a for a in graph.activities if a.label == "host_stage:softmax"]
    # Readout 1 KiB + upload 1 KiB over the 32 B/cycle link, plus four
    # vector passes over 512 / 16 vectors.
    expected = (2048 // HOST_LINK_BYTES_PER_CYCLE) + (
        512 // geometry.lanes
    ) * HOST_OPERATION_PASSES["softmax"] * HOST_VECTOR_ADD_CYCLES
    assert stage_activity.latency_cycles == expected
    assert stage_activity.primitive.endswith("HOST_COMPUTE")
    assert [
        use.handle.name for use in stage_activity.occupancy
    ] and "host" in stage_activity.occupancy[0].handle.name
    # The stage depends on the QK group and the SV group depends on it.
    ids = {a.id: a for a in graph.activities}
    assert any("group" in dep for dep in stage_activity.depends_on)
    assert any(
        stage_activity.id in a.depends_on
        for a in graph.activities
        if a.id != stage_activity.id
    )
    # The lowered program carries the same stage once, after QK's commands.
    built = allo.compile(
        attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY
    )
    ctx = built.compiled.layout_ctx
    stages = [s for s in ctx.host_steps if isinstance(s, AimHostStage)]
    assert len(stages) == 1 and stages[0].stage == "softmax"
    assert host_stage_cycles(stages[0], geometry) == expected
    entry = next(e for e in ctx.lowering_manifest if e.get("kind") == "hoststage")
    assert (
        entry["readout_bytes"] == 1024
        and entry["upload_bytes"] == 1024
        and entry["command_count"] == 0
    )
    qk = next(
        e
        for e in ctx.lowering_manifest
        if e.get("kind") == "contraction" and e["name"].startswith("attention_qk")
    )
    assert entry["after_command"] >= qk["command_span"][1]
    assert "function" not in stages[0].manifest()
    # The run result lists the stage among the host steps and never adds it to cycles.
    run = virtual.run_backend()
    assert run.extra["resource_budget"] == BUDGET


def test_an_unknown_operation_is_not_priced_at_zero_and_labels_do_not_select_the_model():
    geometry = _target_geometry(build_aim_target())
    record = AimHostStage(
        stage="softmax",
        operation="unknown",
        inputs=(("a", (4,), 16),),
        outputs=(("b", (4,), 16),),
    )
    with pytest.raises(ValueError, match="no declared host-time model"):
        host_stage_cycles(record, geometry)
    with pytest.raises(ValueError, match="declares no operation"):
        AimHostStage(
            stage="softmax", inputs=(("a", (4,), 16),), outputs=(("b", (4,), 16),)
        )
    # The label is free; the operation prices.  Two labels, one operation,
    # one price; one label, two operations, two prices.
    a = AimHostStage(
        stage="softmax",
        operation="softmax",
        inputs=(("a", (64,), 16),),
        outputs=(("b", (64,), 16),),
    )
    b = AimHostStage(
        stage="normalization",
        operation="softmax",
        inputs=(("a", (64,), 16),),
        outputs=(("b", (64,), 16),),
    )
    c = AimHostStage(
        stage="softmax",
        operation="elementwise",
        inputs=(("a", (64,), 16),),
        outputs=(("b", (64,), 16),),
    )
    assert host_stage_cycles(a, geometry) == host_stage_cycles(b, geometry)
    assert host_stage_cycles(c, geometry) < host_stage_cycles(a, geometry)
    terms = host_stage_terms(a, geometry)
    assert (
        terms["link_cycles"] == 256 // 32
        and terms["compute_cycles"] == 4 * 4
        and terms["passes"] == 4
    )


def test_a_renamed_stage_with_the_same_operation_compiles_at_the_same_price():
    attention, _program = build_attention(1, 128)
    target = build_aim_target()
    softmax = ScaledSoftmax(128)

    def renamed(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.host_compute(
            "normalization",
            softmax,
            inputs=(scores,),
            outputs=(probs,),
            operation="softmax",
        )
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(
        renamed
    )
    reference = allo.compile(
        *build_attention(1, 128)[:1],
        target,
        aim_cost,
        backend="virtual",
        host_moves=build_attention(1, 128)[1],
        resources=BUDGET,
        **POLICY,
    )
    compiled = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    assert compiled.run_backend().cycles == reference.run_backend().cycles
    (activity,) = [
        a
        for a in compiled.compiled.execution_graph.activities
        if a.label == "host_stage:normalization"
    ]
    (expected,) = [
        a
        for a in reference.compiled.execution_graph.activities
        if a.label == "host_stage:softmax"
    ]
    assert activity.latency_cycles == expected.latency_cycles
    lowered = allo.compile(
        attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY
    )
    (stage,) = [
        s for s in lowered.compiled.layout_ctx.host_steps if isinstance(s, AimHostStage)
    ]
    assert (stage.stage, stage.operation) == ("normalization", "softmax")


def test_a_stage_may_read_a_host_input_the_program_never_writes():
    """The shared availability reasoning: a stage reading the region input
    ``q`` (host data no launch writes) and producing ``probs`` is accepted by
    both the schedule and the realization."""
    attention, _program = build_attention(1, 128)
    target = build_aim_target()

    def from_host_input(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.host_compute(
            "softmax",
            ScaledSoftmax(128),
            inputs=(q,),
            outputs=(probs,),
            operation="softmax",
        )
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(
        from_host_input
    )
    compiled = allo.compile(
        attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY
    )
    (stage,) = [
        s
        for s in compiled.compiled.layout_ctx.host_steps
        if isinstance(s, AimHostStage)
    ]
    assert stage.inputs == (("q", (1, 128), 16),)


@pytest.mark.parametrize("mutation", ["early", "late", "function", "between_repeats"])
def test_realization_binds_the_stage_position_and_computation(monkeypatch, mutation):
    """Fault injection on the lowered record sequence: a stage moved before
    its producer or after its consumer, or carrying a computation other than
    the recorded one, is refused at materialization (before selection)."""
    from dataclasses import replace

    from allo import spmw_aim

    attention, program = build_attention(1, 128)
    target = build_aim_target()
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        records = original(self, host_moves, ctx)
        stages = [r for r in records if isinstance(r[0], AimHostStage)]
        rest = [r for r in records if not isinstance(r[0], AimHostStage)]
        if mutation == "early":
            return stages + rest
        if mutation == "late":
            return rest + stages
        if mutation == "function":
            return [
                (
                    (replace(op, function=lambda x: np.zeros_like(x)), banks)
                    if isinstance(op, AimHostStage)
                    else (op, banks)
                )
                for op, banks in records
            ]
        # between_repeats: the stage between two copies of the SV launch
        # records while the schedule has one SV launch after the stage.
        sv = [
            r
            for r in rest
            if getattr(r[0], "name", "") and str(r[0].name).startswith("attention_sv")
        ]
        return rest + stages + sv

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    expected = (
        "carries a computation other than"
        if mutation == "function"
        else "where the schedule expects|incomplete before the next step|after the last scheduled step"
    )
    with pytest.raises(Exception, match=expected):
        allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)


def test_realization_refuses_a_dropped_or_mismatched_stage(monkeypatch):
    from allo import spmw_aim

    attention, program = build_attention(1, 128)
    target = build_aim_target()
    original = spmw_aim._AimProgramDerivation.records

    def dropped(self, host_moves, ctx):
        return [
            r
            for r in original(self, host_moves, ctx)
            if not isinstance(r[0], AimHostStage)
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", dropped)
    with pytest.raises(
        Exception,
        match="lowered 0 host stage|where the schedule expects host stage|lowered no records for host stage",
    ):
        allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)

    def renamed(self, host_moves, ctx):
        from dataclasses import replace

        return [
            (
                (replace(r[0], outputs=(("heads_out", (1, 128), 16),)), r[1])
                if isinstance(r[0], AimHostStage)
                else r
            )
            for r in original(self, host_moves, ctx)
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", renamed)
    with pytest.raises(Exception, match="differs from the schedule"):
        allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)


def test_unsplit_oracle_matches_the_reference():
    attention, program = build_attention(1, 128)
    built = allo.compile(
        attention,
        build_aim_target(),
        aim_cost,
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    qk = next(
        e
        for e in built.compiled.layout_ctx.lowering_manifest
        if e.get("kind") == "contraction" and e["name"].startswith("attention_qk")
    )
    rng = np.random.default_rng(1)
    K = bf16_round(rng.standard_normal((1, 1, 128, 128)).astype(np.float32))
    q = bf16_round(rng.standard_normal((1, 1, 128)).astype(np.float32))
    out = reference_outputs(qk, K, q)
    reference = np.einsum("rbok,rbk->rbo", K.astype(np.float64), q.astype(np.float64))
    assert out.shape == (1, 1, 128)
    assert np.max(np.abs(out - reference)) / np.max(np.abs(reference)) < 2.0**-7
    exact = reference_outputs(
        qk,
        rng.integers(-8, 9, (1, 1, 128, 128)),
        rng.integers(-8, 9, (1, 1, 128)),
        arithmetic="exact",
    )
    assert exact.dtype == np.int64
    with pytest.raises(ValueError, match="does not issue every"):
        reference_outputs({**qk, "issued_scalar_macs": 1}, K, q)


@pytest.mark.parametrize(
    "heads,tokens,partitions", [(1, 128, 1), (1, 128, 4), (2, 256, 2)]
)
def test_functional_pipeline_for_split_and_unsplit_plans(heads, tokens, partitions):
    """QK -> global softmax (the stage's own function from the compiled
    program) -> SV through the lowered programs' functional oracles, against
    a float64 reference, for an unsplit SV, a split SV and a neighbouring
    workload; no simulator is involved."""
    target = build_aim_target()
    attention, program = build_attention(heads, tokens)
    ranking = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    _index, scores = kernel_choices(target, ranking)["attention_sv"]
    choice = min(
        (s for s in scores if partitions_of(s[1]) == partitions), key=lambda s: s[0]
    )
    sv_index = [
        type(s).__name__
        for s in ranking.compiled.launch_schedule.steps
        if isinstance(s, LaunchStep)
    ]
    layouts = [None, None]
    layouts[kernel_choices(target, ranking)["attention_sv"][0]] = choice[2]
    layouts[kernel_choices(target, ranking)["attention_qk"][0]] = min(
        kernel_choices(target, ranking)["attention_qk"][1]
    )[2]
    built = allo.compile(
        attention,
        target,
        aim_cost,
        host_moves=program,
        resources=BUDGET,
        layout=layouts,
        **POLICY,
    )
    check = functional_check(built, heads, tokens, seed=7)
    assert check["split"] == (partitions > 1)
    assert check["max_rel_error"] < check["tolerance"]
    assert check["stage"] == "softmax" and check["operation"] == "softmax"
    assert check["evidence"] == "functional oracle, not the timing-only simulator"
    # The inputs expose the scale: against an unscaled softmax the same
    # pipeline is far off, so a stage without the 1/sqrt(head_dim) scale
    # could not have passed.
    assert (
        check["head_dim"] == 128
        and check["unscaled_reference_error"] > 10 * check["tolerance"]
    )


def test_the_stage_is_the_scaled_softmax():
    """``softmax(scores / sqrt(head_dim))``, the historical host operation's
    semantics, on scores whose range makes the scale visible."""
    import math

    softmax = ScaledSoftmax(128)
    scores = bf16_round(np.linspace(-4, 4, 128, dtype=np.float32)[None])
    out = softmax(scores)
    scaled = scores / np.float32(math.sqrt(128))
    reference = np.exp(scaled - scaled.max())
    reference /= reference.sum()
    assert np.allclose(out, bf16_round(reference.astype(np.float32)), atol=2.0**-9)
    unscaled = np.exp(scores - scores.max())
    unscaled /= unscaled.sum()
    assert float(out.max()) < 0.02 < float(unscaled.max())


def _repeated_program(attention):
    """QK -> softmax -> split SV -> a stage reading heads_out and producing
    probs -> SV again: a valid program with a repeated launch and two stages."""
    softmax = ScaledSoftmax(128)

    def repeated(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.host_compute(
            "softmax", softmax, inputs=(scores,), outputs=(probs,), operation="softmax"
        )
        allo.launch("attention_sv", probs, v_cache, heads_out)
        allo.host_compute(
            "next", softmax, inputs=(heads_out,), outputs=(probs,), operation="softmax"
        )
        allo.launch("attention_sv", probs, v_cache, heads_out)

    return allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(repeated)


def _split_sv_layout(target, attention, program, partitions=2):
    ranking = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    choices = kernel_choices(target, ranking)
    sv_index, sv_scores = choices["attention_sv"]
    qk_index, qk_scores = choices["attention_qk"]
    layouts = [None, None]
    layouts[sv_index] = min(
        (s for s in sv_scores if partitions_of(s[1]) == partitions), key=lambda s: s[0]
    )[2]
    layouts[qk_index] = min(qk_scores)[2]
    return layouts


def test_repeated_launches_with_two_stages_are_accepted():
    from allo.pim.aim_lowering import AimContraction, AimHostReduction

    attention, _ = build_attention(1, 128)
    target = build_aim_target()
    program = _repeated_program(attention)
    layouts = _split_sv_layout(target, attention, program)
    built = allo.compile(
        attention,
        target,
        aim_cost,
        host_moves=program,
        resources=BUDGET,
        layout=layouts,
        **POLICY,
    )
    kinds = [type(op).__name__ for op in built.compiled.layout_ctx.lowered_operations]
    assert kinds == [
        "AimContraction",
        "AimHostStage",
        "AimContraction",
        "AimHostReduction",
        "AimHostStage",
        "AimContraction",
        "AimHostReduction",
    ]
    assert (
        sum(
            isinstance(op, AimHostReduction)
            for op in built.compiled.layout_ctx.lowered_operations
        )
        == 2
    )
    assert (
        sum(
            isinstance(op, AimContraction)
            for op in built.compiled.layout_ctx.lowered_operations
        )
        == 3
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_qk",
        "stage_before_merge",
        "drop_first_merge",
        "wrong_partitions",
        "duplicate_qk_without_stages",
    ],
)
def test_realization_binds_each_launch_occurrence_and_its_completion(
    monkeypatch, mutation
):
    """Fault injection on the lowered sequence against the inventory derived
    from the facts and the selected candidates: a duplicated primary
    record, a stage placed before the producing launch's host reduction, a
    dropped host reduction (while the repeated launch keeps its own) and a
    reduction with another P are refused at materialization; with or
    without host stages."""
    from dataclasses import replace

    from allo import spmw_aim
    from allo.pim.aim_lowering import AimContraction, AimHostReduction

    attention, plain = build_attention(
        1, 128, bridge=mutation != "duplicate_qk_without_stages"
    )
    target = build_aim_target()
    program = (
        plain
        if mutation in ("duplicate_qk", "duplicate_qk_without_stages")
        else _repeated_program(attention)
    )
    layouts = _split_sv_layout(target, attention, program)
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        records = list(original(self, host_moves, ctx))
        if mutation in ("duplicate_qk", "duplicate_qk_without_stages"):
            out = []
            for record in records:
                out.append(record)
                if isinstance(record[0], AimContraction) and record[0].name.startswith(
                    "attention_qk"
                ):
                    out.append(record)
            return out
        if mutation == "stage_before_merge":
            stage = next(
                r
                for r in records
                if isinstance(r[0], AimHostStage) and r[0].stage == "next"
            )
            records.remove(stage)
            index = next(
                i for i, r in enumerate(records) if isinstance(r[0], AimHostReduction)
            )
            records.insert(index, stage)
            return records
        if mutation == "drop_first_merge":
            index = next(
                i for i, r in enumerate(records) if isinstance(r[0], AimHostReduction)
            )
            del records[index]
            return records
        return [
            (
                (replace(op, partitions=op.partitions * 2), banks)
                if isinstance(op, AimHostReduction)
                else (op, banks)
            )
            for op, banks in records
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    expected = {
        "duplicate_qk": "primary record 'attention_qk_0' where the schedule expects host stage|twice",
        "duplicate_qk_without_stages": "primary record 'attention_qk_0' where the schedule expects|twice",
        "stage_before_merge": "incomplete before the next step: missing primary records \\[\\] and host reductions \\['attention_sv_0'\\]",
        "drop_first_merge": "incomplete before the next step",
        "wrong_partitions": "merges 4 partials of 'attention_sv_0'; the candidate declares 2",
    }[mutation]
    with pytest.raises(Exception, match=expected):
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )


@pytest.mark.parametrize(
    "field,value", [("outputs", 1), ("batches", 1), ("replicas", 2)]
)
@pytest.mark.parametrize("occurrence", ["first", "later"])
def test_reduction_work_domain_is_bound_to_its_producing_contraction(
    monkeypatch, field, value, occurrence
):
    """A host reduction whose outputs, batches or replicas differ from its
    producing contraction's is refused at materialization, whether it
    belongs to the first SV launch or to the repeated one (2 heads x 128
    tokens, P = 2); its manifest price would otherwise disagree with the
    priced graph (64 cycles here: 8, 32 or 128 with the mutation)."""
    from dataclasses import replace

    from allo import spmw_aim
    from allo.pim.aim_lowering import AimHostReduction

    attention, program = build_attention(2, 128)
    target = build_aim_target()
    if occurrence == "later":
        program = _repeated_program(attention)
    layouts = _split_sv_layout(target, attention, program, partitions=2)
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        records = list(original(self, host_moves, ctx))
        merges = [
            i for i, r in enumerate(records) if isinstance(r[0], AimHostReduction)
        ]
        target_index = merges[0] if occurrence == "first" else merges[-1]
        op, banks = records[target_index]
        records[target_index] = (replace(op, **{field: value}), banks)
        return records

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    with pytest.raises(Exception, match="its contraction produces"):
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )


def test_lowered_merge_cost_equals_the_priced_graph():
    """For the unmodified programs (single and repeated launch) the lowered
    host reductions price exactly what the selection graph priced."""
    from allo.pim.aim_lowering import AimHostReduction
    from allo.pim.costs.aim import host_reduction_cycles

    target = build_aim_target()
    geometry = _target_geometry(target)
    attention, single = build_attention(2, 128)
    for program in (single, _repeated_program(attention)):
        layouts = _split_sv_layout(target, attention, program, partitions=2)
        built = allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )
        virtual = allo.compile(
            attention,
            target,
            aim_cost,
            backend="virtual",
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )
        lowered = [
            host_reduction_cycles(op, geometry)
            for op in built.compiled.layout_ctx.lowered_operations
            if isinstance(op, AimHostReduction)
        ]
        priced = [
            a.latency_cycles
            for a in virtual.compiled.execution_graph.activities
            if a.label == "host_reduction"
        ]
        assert lowered and lowered == priced == [64] * len(lowered)


@pytest.mark.parametrize("occurrence", ["single", "repeated"])
@pytest.mark.parametrize(
    "change", [{"outputs": 256, "batches": 1}, {"replicas": 2}, {"reduction": 64}]
)
def test_contraction_dimensions_are_bound_to_the_retained_shape(
    monkeypatch, occurrence, change
):
    """A coordinated mutation of the SV contraction and its merge (2 heads x
    128 tokens, P = 2): outputs 256 x batches 1 keeps the flattened work
    equal to outputs 128 x batches 2 but not the batch and output
    addressing; it is refused dimension by dimension against the retained
    semantic shape under the selected candidate, on the single and on the
    repeated program (the program's MAC metadata is carried over, so the
    coverage check is not what refuses it)."""
    from dataclasses import replace

    from allo import spmw_aim
    from allo.pim.aim_lowering import AimContraction, AimHostReduction

    attention, program = build_attention(2, 128)
    target = build_aim_target()
    if occurrence == "repeated":
        program = _repeated_program(attention)
    layouts = _split_sv_layout(target, attention, program, partitions=2)
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        result = []
        for op, banks in original(self, host_moves, ctx):
            if (
                isinstance(op, AimContraction) and op.reduction_partitions > 1
            ) or isinstance(op, AimHostReduction):
                fields = {k: v for k, v in change.items() if hasattr(op, k)}
                old = op
                op = replace(op, **fields)
                if id(old) in self.program_macs:
                    self.program_macs[id(op)] = self.program_macs[id(old)]
                if id(old) in self.row_maps:
                    self.row_maps[id(op)] = self.row_maps[id(old)]
            result.append((op, banks))
        return result

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    with pytest.raises(
        Exception,
        match="the retained program under the selected candidate gives|the retained program under the selected decision derives otherwise|not realizable|logical scalar MACs",
    ):
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )


@pytest.mark.parametrize("partitions", [1, 2])
def test_neighbouring_head_dimension_workload_passes(partitions):
    """4 heads x 256 tokens with head dimension 64: unmodified, accepted,
    functionally checked, its merge priced as the graph prices it."""
    from allo.pim.aim_lowering import AimHostReduction
    from allo.pim.costs.aim import host_reduction_cycles

    target = build_aim_target()
    attention, program = build_attention(4, 256, head_dim=64)
    layouts = _split_sv_layout(target, attention, program, partitions=partitions)
    built = allo.compile(
        attention,
        target,
        aim_cost,
        host_moves=program,
        resources=BUDGET,
        layout=layouts,
        **POLICY,
    )
    check = functional_check(built, 4, 256, seed=3, head_dim=64)
    assert check["max_rel_error"] < check["tolerance"] and check["head_dim"] == 64
    assert check["split"] == (partitions > 1)
    lowered = [
        host_reduction_cycles(op, _target_geometry(target))
        for op in built.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimHostReduction)
    ]
    virtual = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        layout=layouts,
        **POLICY,
    )
    priced = [
        a.latency_cycles
        for a in virtual.compiled.execution_graph.activities
        if a.label == "host_reduction"
    ]
    assert lowered == priced


@pytest.mark.parametrize("occurrence", ["single", "repeated"])
@pytest.mark.parametrize(
    "change",
    [
        {"activation": True},
        {"reuse_group_size": 1},
        {"result_gpr": 7},
        {"input_gpr": 3},
        {"channels": (0, 1, 2, 3, 4, 5, 6, 8)},
    ],
)
def test_every_emitted_contraction_field_is_bound_to_the_derivation(
    monkeypatch, occurrence, change
):
    """The realized records of each launch occurrence must equal, record by
    record, what the retained program and the selected decision derive for
    one launch of the group: an added activation (AF/RD_AF commands), another
    reuse group (command order), another result or input GPR address or
    another channel set on the QK record is refused at materialization,
    with the differing fields named, on the single and the repeated program."""
    from dataclasses import replace

    from allo import spmw_aim
    from allo.pim.aim_lowering import AimContraction

    attention, program = build_attention(2, 128)
    target = build_aim_target()
    if occurrence == "repeated":
        program = _repeated_program(attention)
    layouts = _split_sv_layout(target, attention, program, partitions=2)
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        result = []
        for op, banks in original(self, host_moves, ctx):
            if isinstance(op, AimContraction) and "qk" in op.name:
                old = op
                op = replace(op, **change)
                for metadata in (self.program_macs, self.row_maps):
                    if id(old) in metadata:
                        metadata[id(op)] = metadata[id(old)]
            result.append((op, banks))
        return result

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    field = next(iter(change))
    with pytest.raises(
        Exception,
        match=f"{field}: realized|not realizable|logical scalar MACs|the retained program under the selected",
    ):
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )


def test_unmodified_programs_match_their_derivation_record_by_record():
    """The whole-record comparison accepts the unmodified single, repeated and
    neighbouring (3 heads x 192 tokens x head dimension 64) programs, and the
    neighbour's functional pipeline passes."""
    target = build_aim_target()
    attention, program = build_attention(2, 128)
    for prog in (program, _repeated_program(attention)):
        layouts = _split_sv_layout(target, attention, prog, partitions=2)
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=prog,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )
    neighbour, neighbour_program = build_attention(3, 192, head_dim=64)
    built = allo.compile(
        neighbour,
        target,
        aim_cost,
        host_moves=neighbour_program,
        resources=BUDGET,
        **POLICY,
    )
    check = functional_check(built, 3, 192, seed=19, head_dim=64)
    assert check["max_rel_error"] < check["tolerance"] and check["head_dim"] == 64


@pytest.mark.parametrize("payload", [tuple(range(16)), (0,) * 16, (0, 1, 2, 3)])
def test_the_emitted_bank_payload_is_bound_to_the_derivation(monkeypatch, payload):
    """The emission input is the record and its bank payload: an all-bank QK
    record whose returned payload is replaced by sixteen banks (MAC_SBK
    expansion), by bank 0 sixteen times or by four banks is refused at
    materialization against the derived payload; the declared scope and
    estimate would otherwise stay those of the all-bank plan."""
    from allo import spmw_aim
    from allo.pim.aim_lowering import AimContraction

    attention, program = build_attention(2, 128)
    target = build_aim_target()
    layouts = _split_sv_layout(target, attention, program, partitions=2)
    original = spmw_aim._AimProgramDerivation.records

    def mutated(self, host_moves, ctx):
        return [
            (
                op,
                (
                    payload
                    if isinstance(op, AimContraction) and "qk" in op.name
                    else banks
                ),
            )
            for op, banks in original(self, host_moves, ctx)
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", mutated)
    with pytest.raises(
        Exception, match="expands 'attention_qk_0' over banks|scalar MACs"
    ):
        allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )


def test_genuine_single_bank_and_all_bank_candidates_are_accepted():
    """An enumerated single-bank QK candidate lowers with its derived bank
    payload (MAC_SBK over distinct banks, manifest scope sbk) and an all-bank
    one without (scope abk); both pass the whole emission-input comparison."""
    target = build_aim_target()
    attention, program = build_attention(2, 128)
    ranking = allo.compile(
        attention,
        target,
        aim_cost,
        backend="virtual",
        host_moves=program,
        resources=BUDGET,
        **POLICY,
    )
    choices = kernel_choices(target, ranking)
    qk_index, qk_scores = choices["attention_qk"]
    sv_index, sv_scores = choices["attention_sv"]
    sv = min(
        (s for s in sv_scores if partitions_of(s[1]) == 2 and "bank_scope=abk" in s[1]),
        key=lambda s: s[0],
    )[2]
    for scope in ("sbk", "abk"):
        qk = min(
            (s for s in qk_scores if f"bank_scope={scope}" in s[1]), key=lambda s: s[0]
        )[2]
        layouts = [None, None]
        layouts[qk_index], layouts[sv_index] = qk, sv
        built = allo.compile(
            attention,
            target,
            aim_cost,
            host_moves=program,
            resources=BUDGET,
            layout=layouts,
            **POLICY,
        )
        entry = next(
            e
            for e in built.compiled.layout_ctx.lowering_manifest
            if e.get("kind") == "contraction" and e["name"].startswith("attention_qk")
        )
        assert entry["bank_scope"] == scope
        qk_lines = [line for line in built.compiled.cmds if "MAC_" in line][
            : 4 if scope == "abk" else 32
        ]
        if scope == "sbk":
            banks = entry["sbk_banks"]
            assert len(banks) == len(set(banks)) and all(0 <= b < 16 for b in banks)
            assert any(line.startswith("AiM MAC_SBK") for line in qk_lines)
        else:
            assert "sbk_banks" not in entry and all(
                not line.startswith("AiM MAC_SBK") for line in qk_lines
            )
