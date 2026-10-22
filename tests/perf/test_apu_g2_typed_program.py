# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software gates for the exact-width APUg2 direct-program surface."""

import inspect

import numpy as np

import allo
from allo.pim.apu_g2_typed_program import (
    APUG2ScalarType,
    APUG2TypedCallable,
    APUG2TypedOperation,
    APUG2TypedProgram,
    build_apu_g2_typed_recipe,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target
from allo.spmw_codegen import RunResult


I4 = APUG2ScalarType(4, True)
I8 = APUG2ScalarType(8, True)
U8 = APUG2ScalarType(8, False)
I15 = APUG2ScalarType(15, True)
I16 = APUG2ScalarType(16, True)
U16 = APUG2ScalarType(16, False)


def _programs():
    return (
        APUG2TypedProgram("add", (I16, I16), I16, (4, 65536)),
        APUG2TypedProgram("mul", (I8, I8), I16, (4, 65536)),
        APUG2TypedProgram("div", (U16, U16), U16, (4, 65536)),
        APUG2TypedProgram(
            "block_sum",
            (U8,),
            U16,
            (4, 65536),
            reduction_extent=256,
        ),
        APUG2TypedProgram(
            "dot",
            (I4, I4),
            I15,
            (1984, 128),
            reduction_extent=128,
        ),
    )


def test_allo_compile_builds_virtual_typed_callables_for_all_profiles():
    target = build_apu_g2_target()
    for program in _programs():
        program = APUG2TypedProgram(
            program.operation,
            program.input_types,
            program.output_type,
            program.shape,
            program.reduction_extent,
            repetitions=7,
        )
        compiled = allo.compile(
            program,
            target,
            apu_g2_cost,
            backend="virtual",
        )
        assert isinstance(compiled, APUG2TypedCallable)
        assert compiled.estimate().cycles > 0
        assert compiled.execution_graph.metadata["physical_recipe"] is True
        assert (
            compiled.execution_graph.metadata["recipe_fingerprint"]
            == build_apu_g2_typed_recipe(program).structural_fingerprint
        )
        names = tuple(inspect.signature(compiled).parameters)
        expected_names = (
            ("src", "out")
            if program.operation is APUG2TypedOperation.BLOCK_SUM
            else ("lhs", "rhs", "out")
        )
        assert names == expected_names
        operands = tuple(
            np.zeros(shape, dtype=scalar_type.numpy_dtype)
            for shape, scalar_type in zip(program.input_shapes, program.input_types)
        )
        if program.operation is APUG2TypedOperation.DIV:
            operands = (operands[0], np.ones_like(operands[1]))
        out = np.full(
            program.output_shape,
            17,
            dtype=program.output_type.numpy_dtype,
        )
        result = compiled(*operands, out)
        assert result.backend == "virtual"
        assert result.extra["outputs"] == {}
        assert np.all(out == 17)


def test_device_dispatch_preserves_native_int15_output(monkeypatch):
    target = build_apu_g2_target()
    program = APUG2TypedProgram(
        "dot",
        (I4, I4),
        I15,
        (3, 128),
        reduction_extent=128,
        repetitions=11,
    )
    compiled = allo.compile(program, target, apu_g2_cost, backend="device")
    observed = np.array([-17, 0, 2047], dtype=np.int16)
    captured = {}

    def fake_runtime(received_program, *operands):
        captured["program"] = received_program
        captured["operands"] = operands
        return RunResult(
            73,
            "software-only fake typed device",
            "apu_v2",
            extra={
                "outputs": {
                    "out": observed,
                    "physical_out": np.zeros((4, 65536), dtype=np.int16),
                }
            },
        )

    monkeypatch.setattr(
        "allo.pim.apu_g2_typed_runtime.run_apu_g2_typed",
        fake_runtime,
    )
    lhs = np.zeros((3, 128), dtype=np.int8)
    rhs = np.ones((3, 128), dtype=np.int8)
    out = np.empty((3,), dtype=np.int16)
    result = compiled(lhs, rhs, out)
    assert result.cycles == 73
    assert captured["program"] is program
    assert captured["operands"][0].dtype == np.int8
    np.testing.assert_array_equal(out, observed)
