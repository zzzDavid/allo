# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the executable, handle-based cost abstraction."""

import allo

from allo.perf import (
    Activity,
    CostEvent,
    Evaluator,
    ExecutionGraph,
    HandleInstance,
    Occupancy,
    cost,
    rule,
)
from allo.spmw_target import op, target, unit


def _activity(name, handle, latency=10, occupancy=None, deps=()):
    return Activity(
        id=name,
        primitive="test/op",
        latency_cycles=latency,
        occupancy=(Occupancy(handle, occupancy or latency),),
        depends_on=tuple(deps),
    )


def test_independent_handle_instances_overlap():
    first = HandleInstance("chip/engine", (0,))
    second = HandleInstance("chip/engine", (1,))
    graph = ExecutionGraph("parallel")
    graph.extend([_activity("a", first), _activity("b", second)])

    estimate = Evaluator().evaluate(graph)

    assert estimate.cycles == 10
    assert estimate.spans["a"].start_cycle == 0
    assert estimate.spans["b"].start_cycle == 0


def test_old_configuration_apis_are_removed():
    assert not hasattr(allo, "cycle_model")
    assert not hasattr(allo, "resource")
    assert not hasattr(allo, "const")


def test_shared_handle_instance_serializes():
    engine = HandleInstance("chip/engine", (0,))
    graph = ExecutionGraph("serial")
    graph.extend([_activity("a", engine), _activity("b", engine)])

    estimate = Evaluator().evaluate(graph)

    assert estimate.cycles == 20
    assert estimate.spans["b"].start_cycle == 10
    assert estimate.critical_path == ("a", "b")


def test_latency_and_handle_occupancy_are_independent():
    engine = HandleInstance("chip/pipeline", (0,))
    graph = ExecutionGraph("pipeline")
    graph.extend(
        [
            _activity("a", engine, latency=10, occupancy=2),
            _activity("b", engine, latency=10, occupancy=2),
            _activity("c", engine, latency=10, occupancy=2),
        ]
    )

    estimate = Evaluator().evaluate(graph)

    assert estimate.cycles == 14
    assert [estimate.spans[name].start_cycle for name in "abc"] == [0, 2, 4]


def test_dependency_prevents_overlap_on_independent_handles():
    graph = ExecutionGraph("dependency")
    graph.extend(
        [
            _activity("a", HandleInstance("chip/engine", (0,)), latency=7),
            _activity(
                "b",
                HandleInstance("chip/engine", (1,)),
                latency=7,
                deps=("a",),
            ),
        ]
    )

    estimate = Evaluator().evaluate(graph)

    assert estimate.cycles == 14
    assert estimate.spans["b"].start_cycle == 7


def _build_test_target():
    @target("cost_test")
    def device():
        @unit(mapping={"lane": 2})
        def lane():
            scalar = object()
            op("ADD", src=(scalar, scalar), dst=scalar, fn=lambda x, y: x + y)

    return device


@cost(target="cost_test")
def sample_cost(target_spec):
    lane = target_spec.unit("lane")

    @rule(target_spec.op("ADD"))
    def add(event, ctx):
        cycles = int(event.elements) + 1
        ctx.step(
            latency=cycles,
            occupy=[
                ctx.use(event.primitive, cycles=1),
                ctx.use(lane, cycles=cycles),
            ],
            name="add",
        )


def test_cost_spec_is_ordinary_executable_code():
    target_spec = _build_test_target()
    bound = sample_cost.bind(target_spec)
    graph = ExecutionGraph("program")
    bound.emit(
        graph,
        CostEvent.create(
            "left", target_spec.op("ADD"), work_id=(0,), metrics={"elements": 4}
        ),
    )
    bound.emit(
        graph,
        CostEvent.create(
            "right", target_spec.op("ADD"), work_id=(1,), metrics={"elements": 8}
        ),
    )

    estimate = bound.evaluate(graph)

    assert estimate.cycles == 9
    assert estimate.spans["left:cost:0:add"].latency_cycles == 5
    assert estimate.spans["right:cost:0:add"].latency_cycles == 9
    assert estimate.model_fingerprint == bound.fingerprint


def test_cost_repeat_and_parallel_are_program_constructs():
    target_spec = _build_test_target()

    @cost(target="cost_test")
    def composite(spec):
        operation = spec.op("ADD")

        @rule(operation)
        def add(_event, ctx):
            with ctx.parallel():
                with ctx.repeat(3):
                    ctx.step(cycles=2, occupy=[ctx.use(operation)], name="lhs")
                ctx.step(cycles=5, name="rhs")
            ctx.step(cycles=1, name="join")

    bound = composite.bind(target_spec)
    graph = ExecutionGraph("composite")
    bound.emit(graph, CostEvent.create("add", target_spec.op("ADD")))

    estimate = bound.evaluate(graph)

    assert estimate.cycles == 7
    assert estimate.spans["add:cost:0:lhs"].latency_cycles == 6
    assert estimate.spans["add:cost:1:rhs"].start_cycle == 0
    assert estimate.spans["add:cost:2:join"].start_cycle == 6


def test_capacity_is_declared_on_existing_target_handles():
    @target("capacity_test")
    def capacity_target():
        @unit(mapping={"lane": 1}, capacity=2)
        def lane():
            storage = allo.mem(name="storage", ports=3)
            op(
                "ADD",
                src=(storage, storage),
                dst=storage,
                fn=lambda x, y: x + y,
                capacity=2,
            )

    operation = capacity_target.op("ADD")
    event = CostEvent.create("add", operation, work_id=(0,))

    assert event.instance(capacity_target.unit("lane")).capacity == 2
    assert event.operation_instance.capacity == 2
    assert event.instance(capacity_target.storage).capacity == 3


def test_symbolic_memory_refs_become_concrete_bank_instances():
    from allo.pim.targets import build_samsung_target

    target_spec = build_samsung_target()
    even_bank = target_spec.move("LD_A").src

    first = CostEvent.create("first", target_spec.move("LD_A"), work_id=(2, 0))
    second = CostEvent.create("second", target_spec.move("LD_A"), work_id=(2, 1))

    assert first.instance(even_bank).name.endswith("banks[0][channel=2]")
    assert second.instance(even_bank).name.endswith("banks[2][channel=2]")
