# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for bound cost-model fingerprints."""

import sys

import pytest

import allo
from allo.perf import cost, rule
from allo.spmw_target import op, target, unit


_CALIBRATION_CYCLES = 7


class _MutableCalibration:
    def __init__(self, cycles):
        self.cycles = cycles


_MUTABLE_CALIBRATION = _MutableCalibration(7)


def _calibrated_latency():
    return _CALIBRATION_CYCLES + 1


def _mutable_object_latency():
    return _MUTABLE_CALIBRATION.cycles


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


def test_equivalent_separately_built_models_are_stable():
    first = _make_cost({"revision": "stable"}).bind(_build_target())
    second = _make_cost({"revision": "stable"}).bind(_build_target())

    assert first.fingerprint == second.fingerprint
