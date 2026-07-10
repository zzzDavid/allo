# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public compile and validation gates for the hardware-only APUg2 program."""

import inspect

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_program import APUG2Callable, APUG2Operation, APUG2Program
from allo.pim.apu_g2_vector_program import APUG2ChunkedGesummvCallable
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


def test_apu_g2_program_validates_operation_and_profile_repetition_count():
    assert APUG2Program().operation is APUG2Operation.ADD_U16
    assert APUG2Program(repetitions=7).repetitions == 7
    gesummv = APUG2Program(operation="gesummv_u16", shape=(90, 90))
    assert gesummv.shape == (90, 90)
    assert (gesummv.alpha, gesummv.beta) == (5, 4)

    with pytest.raises(ValueError, match="operation must be one of"):
        APUG2Program(operation="mul_u16")
    with pytest.raises(TypeError, match="shape"):
        APUG2Program(operation="gesummv_u16")
    with pytest.raises(ValueError, match="fixed physical shape"):
        APUG2Program(shape=(4, 65536))
    with pytest.raises(ValueError, match="alpha.*uint16"):
        APUG2Program(operation="gesummv_u16", shape=(90, 90), alpha=65536)
    wide = APUG2Program(operation="gesummv_u16", shape=(1, 257))
    target = build_apu_g2_target()
    compiled = allo.compile(wide, target, apu_g2_cost, backend="virtual")
    assert isinstance(compiled, APUG2ChunkedGesummvCallable)
    assert compiled.matrix_tiling.reduction_tile_count == 2
    for repetitions in (0, -1, 2**32):
        with pytest.raises(ValueError, match="repetitions"):
            APUG2Program(repetitions=repetitions)
    for repetitions in (True, 1.5, "2"):
        with pytest.raises(TypeError, match="repetitions"):
            APUG2Program(repetitions=repetitions)


def test_public_compile_builds_a_virtual_apu_g2_callable_without_simulation():
    target = build_apu_g2_target()
    compiled = allo.compile(
        APUG2Program(repetitions=3),
        target,
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2Callable)
    assert inspect.signature(compiled) == inspect.Signature(
        parameters=(
            inspect.Parameter("lhs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("rhs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "out", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        )
    )
    assert compiled.backend == "virtual"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["coalesced_groups"] == 16
    assert compiled.execution_graph.metadata["vl64_calls"] == 1
    assert (
        sum(
            activity.label == "add_u16"
            for activity in compiled.execution_graph.activities
        )
        == 1
    )

    lhs = np.arange(4 * 65536, dtype=np.uint16).reshape(4, 65536)
    rhs = np.full((4, 65536), 65535, dtype=np.uint16)
    out = np.full((4, 65536), 17, dtype=np.uint16)
    run = compiled(lhs, rhs, out)

    assert run is compiled.last_result
    assert run.backend == "virtual"
    assert run.cycles == compiled.estimate().cycles
    assert run.cycles > 0
    assert run.extra["outputs"] == {}
    assert run.extra["vl64_calls"] == 1
    assert np.all(out == 17), "virtual costing must not masquerade as execution"


def test_public_compile_builds_polybench_gesummv_with_reduction_layout_and_cost():
    target = build_apu_g2_target()
    compiled = allo.compile(
        APUG2Program(
            operation="gesummv_u16",
            shape=(90, 90),
            alpha=5,
            beta=4,
            repetitions=3,
        ),
        target,
        apu_g2_cost,
        backend="virtual",
    )

    assert inspect.signature(compiled) == inspect.Signature(
        parameters=(
            inspect.Parameter("A", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("B", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "out", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        )
    )
    assert compiled.layout.output_extent == 90
    assert compiled.layout.reduction_extent == 90
    assert compiled.layout.padded_reduction_extent == 128
    assert compiled.layout.log_block_size == 7
    assert compiled.execution_graph.metadata["program"] == "gesummv_u16"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["execution"] == "direct_vl64"
    assert compiled.execution_graph.metadata["vl64_calls"] == 18
    assert compiled.estimate().cycles > 0

    A = np.zeros((90, 90), dtype=np.uint16)
    B = np.zeros_like(A)
    x = np.zeros(90, dtype=np.uint16)
    out = np.full(90, 17, dtype=np.uint16)
    run = compiled(A, B, x, out)
    assert run.backend == "virtual"
    assert run.extra["outputs"] == {}
    assert np.all(out == 17)


def test_public_compile_requires_cost_and_rejects_generic_compile_controls():
    target = build_apu_g2_target()
    program = APUG2Program()

    with pytest.raises(ValueError, match="requires its executable cost spec"):
        allo.compile(program, target, backend="virtual")
    with pytest.raises(ValueError, match="owns its VL64 layout"):
        allo.compile(program, target, apu_g2_cost, layout=object())
    with pytest.raises(ValueError, match="owns its VL64 layout"):
        allo.compile(program, target, apu_g2_cost, host_moves=[])
    with pytest.raises(ValueError, match="hardware device or virtual cost"):
        allo.compile(program, target, apu_g2_cost, backend="functional")


def test_apu_g2_callable_rejects_wrong_input_and_output_dtype_or_shape():
    compiled = allo.compile(
        APUG2Program(),
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )
    good = np.zeros((4, 65536), dtype=np.uint16)

    with pytest.raises(TypeError, match="operand 'lhs'.*uint16"):
        compiled(good.astype(np.int16), good)
    with pytest.raises(ValueError, match="operand 'lhs'.*shape"):
        compiled(np.zeros((4, 65535), dtype=np.uint16), good)
    with pytest.raises(TypeError, match="output.*uint16"):
        compiled(good, good, np.zeros_like(good, dtype=np.int16))
    with pytest.raises(ValueError, match="output.*shape"):
        compiled(good, good, np.zeros((4, 65535), dtype=np.uint16))

    gesummv = allo.compile(
        APUG2Program(operation="gesummv_u16", shape=(90, 90)),
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )
    A = np.zeros((90, 90), dtype=np.uint16)
    x = np.zeros(90, dtype=np.uint16)
    with pytest.raises(TypeError, match="operand 'A'.*uint16"):
        gesummv(A.astype(np.int16), A, x)
    with pytest.raises(ValueError, match="operand 'B'.*shape"):
        gesummv(A, np.zeros((89, 90), dtype=np.uint16), x)
    with pytest.raises(ValueError, match="operand 'x'.*shape"):
        gesummv(A, A, np.zeros(89, dtype=np.uint16))
    with pytest.raises(ValueError, match="output.*shape"):
        gesummv(A, A, x, np.zeros(89, dtype=np.uint16))
