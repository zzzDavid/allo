# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for bound cost-model fingerprints."""

import functools
import math
import sys
from types import FunctionType, ModuleType

import pytest

import allo
from allo.perf import CostEvent, ExecutionGraph, cost, rule
from allo.spmw_target import op, target, unit


_CALIBRATION_CYCLES = 7
_MATERIALIZATION_OFFSET = 0
_CANONICAL_CALIBRATION = {"cycles": 7}


class _MutableCalibration:
    def __init__(self, cycles):
        self.cycles = cycles


class _CallableCalibration:
    def __init__(self, cycles):
        self.cycles = cycles

    def __call__(self):
        return self.cycles


class _ManifestCalibration:
    def __init__(self, cycles):
        self.cycles = cycles

    def __allo_fingerprint_manifest__(self):
        return {"cycles": self.cycles}


class _ClassCalibration:
    cycles = 7


_MUTABLE_CALIBRATION = _MutableCalibration(7)
_CALLABLE_CALIBRATION = _CallableCalibration(7)
_MANIFEST_CALIBRATION = _ManifestCalibration(7)


def _calibrated_latency():
    return _CALIBRATION_CYCLES + 1


def _alternate_latency():
    return _CALIBRATION_CYCLES + 2


def _canonical_mapping_latency():
    return _CANONICAL_CALIBRATION["cycles"]


def _mutable_object_latency():
    return _MUTABLE_CALIBRATION.cycles


def _callable_object_latency():
    return _CALLABLE_CALIBRATION()


def _manifest_object_latency():
    return _MANIFEST_CALIBRATION.cycles


def _class_latency():
    return _ClassCalibration.cycles


def _external_latency_v1():
    return 7


def _external_latency_v2():
    return 11


def _build_target(
    *, mapping_items=(("lane", 2), ("cluster", 1)), geometry_items=None, capacity=1
):
    geometry_items = geometry_items or (("rows", 32), ("width", 16), ("ports", 2))

    @target("fingerprint_target")
    def device():
        @unit(mapping=dict(mapping_items), capacity=capacity)
        def lane():
            storage = allo.mem(name="storage", **dict(geometry_items))
            registers = allo.reg(4, 16, name="registers", slots=3, ports=2)
            allo.move("LOAD", src=storage, dst=registers, capacity=2)
            op(
                "ADD",
                src=(registers, registers),
                dst=registers,
                fn=lambda left, right: left + right,
                capacity=2,
            )

    return device


@cost(target="fingerprint_target")
def _module_cost(target_spec):
    operation = target_spec.op("ADD")

    @rule(operation)
    def add(_event, ctx):
        ctx.step(cycles=_calibrated_latency(), occupy=[ctx.use(operation)])


def _score_materialization(_bound, materialization):
    return materialization["cycles"] + _MATERIALIZATION_OFFSET


def _alternate_materialization_score(_bound, materialization):
    return materialization["cycles"] + 1


def _make_cost(fingerprint_data=None, materialization_scorer=None):
    @cost(
        target="fingerprint_target",
        fingerprint_data=fingerprint_data,
        materialization_scorer=materialization_scorer,
    )
    def model(target_spec):
        operation = target_spec.op("ADD")

        @rule(operation)
        def add(_event, ctx):
            ctx.step(cycles=3, occupy=[ctx.use(operation)])

    return model


def _make_dependency_cost(latency, fingerprint_data=None):
    @cost(target="fingerprint_target", fingerprint_data=fingerprint_data)
    def model(target_spec):
        operation = target_spec.op("ADD")

        @rule(operation)
        def add(_event, ctx):
            ctx.step(cycles=latency(), occupy=[ctx.use(operation)])

    return model


def test_referenced_global_calibration_changes_fingerprint(monkeypatch):
    target_spec = _build_target()
    original = _module_cost.bind(target_spec)

    monkeypatch.setattr(sys.modules[__name__], "_CALIBRATION_CYCLES", 9)
    changed = _module_cost.bind(target_spec)

    assert changed.fingerprint != original.fingerprint
    assert original.fingerprint == original.fingerprint


def test_canonical_mutable_dependency_state_changes_fingerprint(monkeypatch):
    model = _make_dependency_cost(_canonical_mapping_latency)
    original = model.bind(_build_target())

    monkeypatch.setitem(_CANONICAL_CALIBRATION, "cycles", 11)
    changed = model.bind(_build_target())

    assert changed.fingerprint != original.fingerprint
    with pytest.raises(RuntimeError, match="changed after it was bound"):
        original.evaluate(ExecutionGraph("canonical-mutable-dependency"))


def test_referenced_helper_behavior_changes_fingerprint(monkeypatch):
    target_spec = _build_target()
    original = _module_cost.bind(target_spec).fingerprint

    monkeypatch.setattr(
        sys.modules[__name__], "_calibrated_latency", _alternate_latency
    )

    assert _module_cost.bind(target_spec).fingerprint != original


def test_external_python_function_bytecode_and_attributes_are_fingerprinted():
    external = FunctionType(
        _external_latency_v1.__code__,
        globals(),
        name="external_latency",
    )
    external.__module__ = "external_cost_fixture"
    external.calibration_revision = "rev-a"

    def latency():
        return external()

    model = _make_dependency_cost(latency)
    original = model.bind(_build_target()).fingerprint

    external.__code__ = _external_latency_v2.__code__
    bytecode_changed = model.bind(_build_target()).fingerprint
    external.__code__ = _external_latency_v1.__code__
    external.calibration_revision = "rev-b"
    attribute_changed = model.bind(_build_target()).fingerprint

    assert bytecode_changed != original
    assert attribute_changed != original


def test_module_and_native_callable_provenance_changes_fingerprint(monkeypatch):
    native_ceil = math.ceil
    native_floor = math.floor

    def module_latency():
        return math.ceil(6.1)

    module_model = _make_dependency_cost(module_latency)
    original_module = module_model.bind(_build_target()).fingerprint

    monkeypatch.setattr(math, "ceil", native_floor)
    runtime_changed = module_model.bind(_build_target()).fingerprint
    monkeypatch.setattr(math, "__file__", __file__)
    artifact_changed = module_model.bind(_build_target()).fingerprint

    assert runtime_changed != original_module
    assert artifact_changed != runtime_changed

    def make_native_model(native_callable):
        def latency():
            return native_callable(6.1)

        return _make_dependency_cost(latency)

    ceil_fingerprint = make_native_model(native_ceil).bind(_build_target()).fingerprint
    floor_fingerprint = (
        make_native_model(native_floor).bind(_build_target()).fingerprint
    )

    assert floor_fingerprint != ceil_fingerprint


def test_source_backed_class_runtime_state_changes_fingerprint(monkeypatch):
    model = _make_dependency_cost(_class_latency)
    original = model.bind(_build_target()).fingerprint

    monkeypatch.setattr(_ClassCalibration, "cycles", 11)

    assert model.bind(_build_target()).fingerprint != original


def test_module_without_source_or_binary_provenance_fails_closed():
    runtime_module = ModuleType("runtime_only_cost_dependency")
    runtime_module.cycles = 7

    def latency():
        return runtime_module.cycles

    model = _make_dependency_cost(latency)

    with pytest.raises(TypeError, match="without source or binary provenance"):
        model.bind(_build_target())


def test_target_geometry_and_capacity_change_fingerprint():
    model = _make_cost()
    baseline = model.bind(_build_target()).fingerprint
    geometry_changed = model.bind(
        _build_target(geometry_items=(("rows", 64), ("width", 16), ("ports", 2)))
    ).fingerprint
    capacity_changed = model.bind(_build_target(capacity=2)).fingerprint

    assert geometry_changed != baseline
    assert capacity_changed != baseline


def test_explicit_provenance_changes_and_is_snapshotted():
    target_spec = _build_target()
    provenance = {"firmware": "rev-a", "dataset": ["sample-1", "sample-2"]}
    bound = _make_cost(provenance).bind(target_spec)
    original = bound.fingerprint

    provenance["firmware"] = "rev-b"

    assert bound.fingerprint == original
    assert _make_cost(provenance).bind(target_spec).fingerprint != original


def test_payload_and_geometry_order_do_not_change_fingerprint():
    first_target = _build_target(
        mapping_items=(("lane", 2), ("cluster", 1)),
        geometry_items=(("rows", 32), ("width", 16), ("ports", 2)),
    )
    second_target = _build_target(
        mapping_items=(("lane", 2), ("cluster", 1)),
        geometry_items=(("ports", 2), ("width", 16), ("rows", 32)),
    )
    first_data = {
        "firmware": "rev-a",
        "table": {"read": 4, "write": 6},
        "samples": {"beta", "alpha"},
    }
    second_data = {
        "samples": {"alpha", "beta"},
        "table": {"write": 6, "read": 4},
        "firmware": "rev-a",
    }

    assert (
        _make_cost(first_data).bind(first_target).fingerprint
        == _make_cost(second_data).bind(second_target).fingerprint
    )


def test_ordered_target_axis_labels_change_fingerprint():
    model = _make_cost()
    lane_first = _build_target(mapping_items=(("lane", 2), ("cluster", 1)))
    cluster_first = _build_target(mapping_items=(("cluster", 1), ("lane", 2)))

    assert model.bind(lane_first).fingerprint != model.bind(cluster_first).fingerprint


def test_actual_mapping_extent_changes_with_stable_axis_labels():
    target_spec = _build_target(mapping_items=(("lane", 2),))
    model = _make_cost()
    original = model.bind(target_spec)
    lane = target_spec.unit("lane")

    assert lane.axes == {"lane": 2}
    lane.mapping[0] = 3
    changed = model.bind(target_spec)

    assert lane.axes == {"lane": 2}
    assert changed.fingerprint != original.fingerprint
    with pytest.raises(RuntimeError, match="changed after it was bound"):
        original.evaluate(ExecutionGraph("mapping-extent"))


def test_unsupported_explicit_payload_fails_clearly():
    model = _make_cost({"firmware": object()})

    with pytest.raises(TypeError, match="fingerprint_data.*unsupported value type"):
        model.bind(_build_target())


def test_mutable_global_dependencies_fail_closed_after_same_type_mutation(
    monkeypatch,
):
    object_model = _make_dependency_cost(_mutable_object_latency)

    with pytest.raises(
        TypeError, match="unsupported executable dependency.*_MutableCalibration"
    ):
        object_model.bind(_build_target())

    monkeypatch.setattr(
        sys.modules[__name__], "_MUTABLE_CALIBRATION", _MutableCalibration(11)
    )

    with pytest.raises(
        TypeError, match="unsupported executable dependency.*_MutableCalibration"
    ):
        object_model.bind(_build_target())


def test_empty_fingerprint_data_does_not_allow_opaque_dependency():
    model = _make_dependency_cost(_mutable_object_latency, fingerprint_data={})

    with pytest.raises(
        TypeError, match="unsupported executable dependency.*_MutableCalibration"
    ):
        model.bind(_build_target())


def test_mutable_nonlocal_closure_dependency_fails_closed():
    calibration = _MutableCalibration(7)

    def latency():
        return calibration.cycles

    model = _make_dependency_cost(latency)

    with pytest.raises(
        TypeError, match="nonlocal calibration.*unsupported executable dependency"
    ):
        model.bind(_build_target())

    calibration.cycles = 11

    with pytest.raises(
        TypeError, match="nonlocal calibration.*unsupported executable dependency"
    ):
        model.bind(_build_target())


def test_callable_instance_dependency_fails_closed(monkeypatch):
    model = _make_dependency_cost(_callable_object_latency)

    with pytest.raises(
        TypeError, match="unsupported executable dependency.*_CallableCalibration"
    ):
        model.bind(_build_target())

    monkeypatch.setattr(
        sys.modules[__name__], "_CALLABLE_CALIBRATION", _CallableCalibration(11)
    )

    with pytest.raises(
        TypeError, match="unsupported executable dependency.*_CallableCalibration"
    ):
        model.bind(_build_target())


def test_per_dependency_typed_manifest_covers_mutable_dependency(monkeypatch):
    model = _make_dependency_cost(
        _manifest_object_latency,
        fingerprint_data={},
    )
    original = model.bind(_build_target())

    monkeypatch.setattr(
        sys.modules[__name__], "_MANIFEST_CALIBRATION", _ManifestCalibration(11)
    )
    changed = model.bind(_build_target())

    assert changed.fingerprint != original.fingerprint
    with pytest.raises(RuntimeError, match="changed after it was bound"):
        original.evaluate(ExecutionGraph("mutable-dependency"))


def test_equivalent_separately_built_models_are_stable():
    first = _make_cost({"revision": "stable"}).bind(_build_target())
    second = _make_cost({"revision": "stable"}).bind(_build_target())

    assert first.fingerprint == second.fingerprint


def test_graph_estimate_retains_exact_bound_fingerprint():
    target_spec = _build_target()
    bound = _make_cost({"firmware": "rev-a"}).bind(target_spec)
    graph = ExecutionGraph("workload-and-candidate-name")
    bound.emit(
        graph,
        CostEvent.create(
            "runtime-event-name",
            target_spec.op("ADD"),
            attributes={"annotation": "not-model-identity"},
        ),
    )

    estimate = bound.evaluate(graph)

    assert graph.metadata["cost_fingerprint"] == bound.fingerprint
    assert estimate.model_fingerprint == bound.fingerprint


def test_materialization_scorer_is_bound_executable_and_fingerprinted():
    target_spec = _build_target()
    baseline = _make_cost().bind(target_spec)
    scored = _make_cost(materialization_scorer=_score_materialization).bind(target_spec)
    alternate = _make_cost(
        materialization_scorer=_alternate_materialization_score
    ).bind(target_spec)

    with pytest.raises(TypeError, match="no materialization scorer"):
        baseline.score_materialization({"cycles": 4})
    assert scored.score_materialization({"cycles": 4}) == 4
    assert scored.fingerprint != baseline.fingerprint
    assert alternate.fingerprint != scored.fingerprint


def test_cost_rejects_noncallable_materialization_scorer():
    with pytest.raises(TypeError, match="Python function"):
        _make_cost(materialization_scorer=object())

    with pytest.raises(TypeError, match="Python function"):
        _make_cost(
            materialization_scorer=functools.partial(
                _score_materialization,
            )
        )


def test_bound_materialization_score_rejects_live_dependency_mutation(monkeypatch):
    bound = _make_cost(materialization_scorer=_score_materialization).bind(
        _build_target()
    )
    assert bound.score_materialization({"cycles": 4}) == 4

    monkeypatch.setattr(sys.modules[__name__], "_MATERIALIZATION_OFFSET", 9)

    with pytest.raises(RuntimeError, match="changed after it was bound"):
        bound.score_materialization({"cycles": 4})
