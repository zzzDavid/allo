# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.perf import (
    Activity,
    CalibrationProfile,
    Evaluator,
    ExecutionGraph,
    Invocation,
    MeasurementRecord,
    ParameterValue,
    ProbeSpec,
    ResourceRequest,
    ResourceSpec,
    ResourceTopology,
    TimingLibrary,
    TimingModel,
    fit_profile,
)


def _model_and_profile(latency=10, ii=0):
    timing = TimingModel(
        "test.op",
        latency_cycles="latency",
        initiation_interval_cycles="ii",
        parameters=("latency", "ii"),
    )
    profile = CalibrationProfile(
        name="test/base",
        target="test",
        model_version="v1",
        parameters={
            "latency": ParameterValue(latency),
            "ii": ParameterValue(ii),
        },
    )
    return TimingLibrary([timing]), profile


def _activity(name, instance=0, deps=()):
    return Activity(
        id=name,
        primitive="OP",
        timing_model="test.op",
        resources=(ResourceRequest("engine", (instance,)),),
        depends_on=tuple(deps),
    )


def test_independent_instances_overlap():
    graph = ExecutionGraph("parallel")
    graph.extend([_activity("a", 0), _activity("b", 1)])
    topology = ResourceTopology([ResourceSpec("engine", instances=2)])
    timings, profile = _model_and_profile(latency=10)

    estimate = Evaluator().evaluate(graph, topology, timings, profile)

    assert estimate.cycles == 10
    assert estimate.spans["a"].start_cycle == 0
    assert estimate.spans["b"].start_cycle == 0


def test_shared_instance_serializes():
    graph = ExecutionGraph("serial")
    graph.extend([_activity("a"), _activity("b")])
    topology = ResourceTopology([ResourceSpec("engine")])
    timings, profile = _model_and_profile(latency=10)

    estimate = Evaluator().evaluate(graph, topology, timings, profile)

    assert estimate.cycles == 20
    assert estimate.spans["b"].start_cycle == 10
    assert estimate.critical_path == ("a", "b")


def test_pipelined_resource_admits_at_initiation_interval():
    graph = ExecutionGraph("pipeline")
    graph.extend([_activity("a"), _activity("b"), _activity("c")])
    topology = ResourceTopology([ResourceSpec("engine", pipelined=True)])
    timings, profile = _model_and_profile(latency=10, ii=2)

    estimate = Evaluator().evaluate(graph, topology, timings, profile)

    assert estimate.cycles == 14
    assert [estimate.spans[x].start_cycle for x in "abc"] == [0, 2, 4]


def test_dependency_prevents_overlap_on_independent_resources():
    graph = ExecutionGraph("dependency")
    graph.extend([_activity("a", 0), _activity("b", 1, deps=("a",))])
    topology = ResourceTopology([ResourceSpec("engine", instances=2)])
    timings, profile = _model_and_profile(latency=7)

    estimate = Evaluator().evaluate(graph, topology, timings, profile)

    assert estimate.cycles == 14
    assert estimate.spans["b"].start_cycle == 7


def test_analytical_shape_formula_and_uncertainty():
    model = TimingModel(
        "vector",
        inputs=("elements", "lanes"),
        parameters=("startup", "per_vector"),
        latency_cycles="startup + ceil_div(elements, lanes) * per_vector",
        initiation_interval_cycles="per_vector",
    )
    profile = CalibrationProfile(
        name="test/uncertain",
        target="test",
        model_version="v1",
        parameters={
            "startup": ParameterValue(2),
            "per_vector": ParameterValue(4, lower=3, upper=5),
        },
    )

    timing = model.evaluate({"elements": 33, "lanes": 16}, profile)

    assert timing.latency_cycles == 14
    assert timing.initiation_interval_cycles == 4
    assert timing.cycle_interval == (11, 17)
    assert timing.initiation_interval_cycle_interval == (3, 5)


def test_microprofile_fit_updates_only_named_parameter():
    base = CalibrationProfile(
        name="chip/base",
        target="chip",
        model_version="v1",
        parameters={
            "mac_cycles": ParameterValue(8),
            "dma_cycles": ParameterValue(20),
        },
    )
    probe = ProbeSpec("mac-loop", parameter="mac_cycles")
    record = MeasurementRecord(
        id="run-1",
        probe_id="mac-loop",
        target="chip",
        target_fingerprint="board-a/fw-3",
        cycles_samples=(39, 40, 41),
        normalizer=10,
    )

    fitted = fit_profile(base, [probe], [record], name="chip/board-a")

    assert fitted.get("mac_cycles").value == 4
    assert fitted.get("mac_cycles").provenance == "measured"
    assert fitted.get("mac_cycles").measurement_ids == ("run-1",)
    assert fitted.get("dma_cycles") == base.get("dma_cycles")
    assert fitted.parent == "chip/base"


def test_calibration_profile_json_roundtrip(tmp_path):
    timings, profile = _model_and_profile(latency=13, ii=2)
    del timings
    path = tmp_path / "profile.json"

    profile.save(path)
    loaded = CalibrationProfile.load(path)

    assert loaded == profile
    assert loaded.fingerprint() == profile.fingerprint()
