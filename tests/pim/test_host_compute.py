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
    HOST_STAGE_VECTOR_PASSES,
    HOST_VECTOR_ADD_CYCLES,
    host_stage_cycles,
)
from allo.pim.targets import build_aim_target, build_samsung_target
from allo.spmw_plan import HostComputeStep, LaunchStep
from benchmarks.cent_aim.token_partition import (
    BUDGET,
    POLICY,
    build_attention,
    functional_check,
    global_softmax,
    kernel_choices,
    partitions_of,
)


def _layouts(compiled):
    stored = compiled.compiled.layout
    return list(stored) if isinstance(stored, (list, tuple)) else [stored]


def test_host_compute_is_recorded_scheduled_and_declared():
    attention, program = build_attention(1, 128)
    assert [type(s).__name__ for s in program.steps] == [
        "LaunchRecord", "HostComputeRecord", "LaunchRecord"
    ]
    (stage,) = program.computes
    assert stage.name == "softmax" and stage.fn is global_softmax
    compiled = allo.compile(
        attention, build_aim_target(), aim_cost, backend="virtual", host_moves=program, **POLICY
    )
    steps = compiled.compiled.launch_schedule.steps
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["LaunchStep", "HostComputeStep", "LaunchStep"]
    step = steps[1]
    assert isinstance(step, HostComputeStep)
    assert step.inputs == (("scores", (1, 128), 16),) and step.outputs == (("probs", (1, 128), 16),)
    assert step.volume("inputs") == 256 and step.volume("outputs") == 256 and step.elements() == 128
    assert step.function is global_softmax
    manifest = compiled.compiled.launch_schedule.manifest()
    assert ("host_compute", "softmax", (("scores", (1, 128), 16),), (("probs", (1, 128), 16),)) in manifest


def test_host_compute_rejects_bad_declarations():
    from allo.spmw_target import BufferToken, HostComputeRecord

    with pytest.raises(RuntimeError, match="outside a host_program"):
        allo.host_compute("softmax", global_softmax, inputs=(BufferToken("a"),), outputs=(BufferToken("b"),))
    with pytest.raises(ValueError, match="declares no outputs"):
        HostComputeRecord("softmax", global_softmax, (BufferToken("a"),), ())
    with pytest.raises(TypeError, match="whole workload buffers"):
        HostComputeRecord("softmax", global_softmax, (BufferToken("a")[0],), (BufferToken("b"),))
    with pytest.raises(TypeError, match="callable"):
        HostComputeRecord("softmax", None, (BufferToken("a"),), (BufferToken("b"),))


def test_stage_must_read_what_an_earlier_launch_produced():
    """The schedule's availability walk: a stage placed before the launch
    that produces its input is refused; placed after, its output reaches the
    consumer through the consumer's own host ingress (the GB vector preload)."""
    from allo.pim.schedule_search import InfeasibleSchedule

    attention, _program = build_attention(1, 128)

    def early(q, k_cache, scores, probs, v_cache, heads_out):
        allo.host_compute("softmax", global_softmax, inputs=(scores,), outputs=(probs,))
        allo.launch("attention_qk", q, k_cache, scores)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(early)
    with pytest.raises((InfeasibleSchedule, RuntimeError), match="before the launch that produces it|no earlier launch"):
        allo.compile(attention, build_aim_target(), aim_cost, backend="virtual", host_moves=program, **POLICY)


def test_other_backends_refuse_a_host_stage():
    attention, program = build_attention(1, 128)
    with pytest.raises(NotImplementedError, match="does not represent declared host stages"):
        allo.compile(attention, build_samsung_target(), None, backend="virtual", host_moves=program, layout="bank_row")


def test_stage_is_priced_in_the_selection_graph_and_lowered():
    target = build_aim_target()
    geometry = _target_geometry(target)
    attention, program = build_attention(1, 512)
    virtual = allo.compile(attention, target, aim_cost, backend="virtual", host_moves=program, resources=BUDGET, **POLICY)
    graph = virtual.compiled.execution_graph
    (stage_activity,) = [a for a in graph.activities if a.label == "host_stage:softmax"]
    # Readout 1 KiB + upload 1 KiB over the 32 B/cycle link, plus four
    # vector passes over 512 / 16 vectors.
    expected = (2048 // HOST_LINK_BYTES_PER_CYCLE) + (512 // geometry.lanes) * HOST_STAGE_VECTOR_PASSES["softmax"] * HOST_VECTOR_ADD_CYCLES
    assert stage_activity.latency_cycles == expected
    assert stage_activity.primitive.endswith("HOST_COMPUTE")
    assert [use.handle.name for use in stage_activity.occupancy] and "host" in stage_activity.occupancy[0].handle.name
    # The stage depends on the QK group and the SV group depends on it.
    ids = {a.id: a for a in graph.activities}
    assert any("group" in dep for dep in stage_activity.depends_on)
    assert any(stage_activity.id in a.depends_on for a in graph.activities if a.id != stage_activity.id)
    # The lowered program carries the same stage once, after QK's commands.
    built = allo.compile(attention, target, aim_cost, host_moves=program, resources=BUDGET, **POLICY)
    ctx = built.compiled.layout_ctx
    stages = [s for s in ctx.host_steps if isinstance(s, AimHostStage)]
    assert len(stages) == 1 and stages[0].stage == "softmax"
    assert host_stage_cycles(stages[0], geometry) == expected
    entry = next(e for e in ctx.lowering_manifest if e.get("kind") == "hoststage")
    assert entry["readout_bytes"] == 1024 and entry["upload_bytes"] == 1024 and entry["command_count"] == 0
    qk = next(e for e in ctx.lowering_manifest if e.get("kind") == "contraction" and e["name"].startswith("attention_qk"))
    assert entry["after_command"] >= qk["command_span"][1]
    assert "function" not in stages[0].manifest()
    # The run result lists the stage among the host steps and never adds it to cycles.
    run = virtual.run_backend()
    assert run.extra["resource_budget"] == BUDGET


def test_an_unknown_stage_kind_is_not_priced_at_zero():
    record = AimHostStage(stage="unknown", inputs=(("a", (4,), 16),), outputs=(("b", (4,), 16),))
    with pytest.raises(ValueError, match="no declared host-time model"):
        host_stage_cycles(record, _target_geometry(build_aim_target()))


def test_realization_refuses_a_dropped_or_mismatched_stage(monkeypatch):
    from allo import spmw_aim

    attention, program = build_attention(1, 128)
    target = build_aim_target()
    original = spmw_aim._AimProgramDerivation.records

    def dropped(self, host_moves, ctx):
        return [r for r in original(self, host_moves, ctx) if not isinstance(r[0], AimHostStage)]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", dropped)
    with pytest.raises(Exception, match="lowered 0 host stage\\(s\\) for 1 declared"):
        allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)

    def renamed(self, host_moves, ctx):
        from dataclasses import replace

        return [
            (replace(r[0], outputs=(("heads_out", (1, 128), 16),)), r[1]) if isinstance(r[0], AimHostStage) else r
            for r in original(self, host_moves, ctx)
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", renamed)
    with pytest.raises(Exception, match="differs from the schedule"):
        allo.compile(attention, target, aim_cost, host_moves=program, **POLICY)


def test_unsplit_oracle_matches_the_reference():
    attention, program = build_attention(1, 128)
    built = allo.compile(attention, build_aim_target(), aim_cost, host_moves=program, resources=BUDGET, **POLICY)
    qk = next(e for e in built.compiled.layout_ctx.lowering_manifest if e.get("kind") == "contraction" and e["name"].startswith("attention_qk"))
    rng = np.random.default_rng(1)
    K = bf16_round(rng.standard_normal((1, 1, 128, 128)).astype(np.float32))
    q = bf16_round(rng.standard_normal((1, 1, 128)).astype(np.float32))
    out = reference_outputs(qk, K, q)
    reference = np.einsum("rbok,rbk->rbo", K.astype(np.float64), q.astype(np.float64))
    assert out.shape == (1, 1, 128)
    assert np.max(np.abs(out - reference)) / np.max(np.abs(reference)) < 2.0**-7
    exact = reference_outputs(qk, rng.integers(-8, 9, (1, 1, 128, 128)), rng.integers(-8, 9, (1, 1, 128)), arithmetic="exact")
    assert exact.dtype == np.int64
    with pytest.raises(ValueError, match="does not issue every"):
        reference_outputs({**qk, "issued_scalar_macs": 1}, K, q)


@pytest.mark.parametrize("heads,tokens,partitions", [(1, 128, 1), (1, 128, 4), (2, 256, 2)])
def test_functional_pipeline_for_split_and_unsplit_plans(heads, tokens, partitions):
    """QK -> global softmax (the stage's own function from the compiled
    program) -> SV through the lowered programs' functional oracles, against
    a float64 reference, for an unsplit SV, a split SV and a neighbouring
    workload; no simulator is involved."""
    target = build_aim_target()
    attention, program = build_attention(heads, tokens)
    ranking = allo.compile(attention, target, aim_cost, backend="virtual", host_moves=program, resources=BUDGET, **POLICY)
    _index, scores = kernel_choices(target, ranking)["attention_sv"]
    choice = min((s for s in scores if partitions_of(s[1]) == partitions), key=lambda s: s[0])
    sv_index = [type(s).__name__ for s in ranking.compiled.launch_schedule.steps if isinstance(s, LaunchStep)]
    layouts = [None, None]
    layouts[kernel_choices(target, ranking)["attention_sv"][0]] = choice[2]
    layouts[kernel_choices(target, ranking)["attention_qk"][0]] = min(kernel_choices(target, ranking)["attention_qk"][1])[2]
    built = allo.compile(attention, target, aim_cost, host_moves=program, resources=BUDGET, layout=layouts, **POLICY)
    check = functional_check(built, heads, tokens, seed=7)
    assert check["split"] == (partitions > 1)
    assert check["max_rel_error"] < check["tolerance"]
    assert check["stage"] == "softmax" and check["evidence"] == "functional oracle, not the timing-only simulator"
