# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software gates for the exact-width APUg2 direct-program surface."""

import inspect

import numpy as np
import pytest

import allo
from allo.pim.apu_g2_typed_program import (
    APUG2ScalarType,
    APUG2TypedCallable,
    APUG2TypedOperation,
    APUG2TypedProgram,
    APUG2_TYPED_CAPACITY,
    APUG2_TYPED_L1_SLOT_BASES,
    APUG2_TYPED_L1_SLOT_PITCH,
    APUG2_TYPED_L1_SLOT_ROWS,
    build_apu_g2_typed_recipe,
    pack_apu_g2_gemm_as_independent_dots,
    unpack_apu_g2_gemm_dots,
)
from allo.pim.apu_g2_typed_runtime import (
    _expected_output,
    _logical_output,
    _pack_operand,
    _parse_metrics,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target
from allo.spmw_codegen import RunResult


I4 = APUG2ScalarType(4, True)
I8 = APUG2ScalarType(8, True)
U8 = APUG2ScalarType(8, False)
I15 = APUG2ScalarType(15, True)
U15 = APUG2ScalarType(15, False)
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


def test_exact_width_profiles_are_geometry_driven_and_public():
    add, mul, div, block_sum, dot = _programs()
    assert allo.APUG2ScalarType is APUG2ScalarType
    assert allo.APUG2TypedProgram is APUG2TypedProgram
    assert add.operation is APUG2TypedOperation.ADD
    assert mul.output_type.name == "int16"
    assert div.input_types[0].name == "uint16"
    assert block_sum.output_shape == (1024,)
    assert dot.output_shape == (1984,)
    assert dot.manifest()["shape"] == [1984, 128]
    assert dot.manifest()["reduction_extent"] == 128
    assert I4.numpy_dtype == np.dtype(np.int8)
    assert I15.numpy_dtype == np.dtype(np.int16)
    assert (I4.minimum, I4.maximum) == (-8, 7)


@pytest.mark.parametrize(
    "args,match",
    (
        (
            ("mul", (I8, I8), I15, (4, 65536)),
            "output must be exactly 16 bits",
        ),
        (
            ("div", (I16, I16), I16, (4, 65536)),
            "requires unsigned",
        ),
        (
            ("block_sum", (U8,), U15, (4, 65536), 256),
            "output must be exactly 16 bits",
        ),
        (
            ("dot", (I4, I4), I15, (1984, 64), 128),
            "dot shape",
        ),
        (
            ("dot", (I4, I4), I15, (1984, 127), 127),
            "power of two",
        ),
    ),
)
def test_illegal_width_or_geometry_combinations_fail_closed(args, match):
    with pytest.raises(ValueError, match=match):
        APUG2TypedProgram(*args)


def test_typed_recipes_record_exact_descriptors_and_no_case_switches():
    expected = {
        APUG2TypedOperation.ADD: ("ADD_TYPED",),
        APUG2TypedOperation.MUL: ("MUL_TYPED",),
        APUG2TypedOperation.DIV: ("DIV_TYPED",),
        APUG2TypedOperation.BLOCK_SUM: ("GROUP_REDUCE_ADD_TYPED",),
        APUG2TypedOperation.DOT: (
            "MUL_TYPED",
            "GROUP_REDUCE_ADD_TYPED",
        ),
    }
    for program in _programs():
        recipe = build_apu_g2_typed_recipe(program)
        inventory = recipe.inventory
        assert all(inventory[opcode] == 1 for opcode in expected[program.operation])
        assert recipe.metadata["typed"] is True
        assert recipe.metadata["program"]["shape"] == list(program.shape)
        assert recipe.certificate.fully_vectorized
        assert recipe.certificate.scalar_tensor_updates == 0
        descriptors = [
            use.descriptor
            for operation in recipe.operations
            for use in operation.descriptors
        ]
        assert all(descriptor.num_vectors == 4 for descriptor in descriptors)
        if program.operation is APUG2TypedOperation.DOT:
            widths = {descriptor.num_bits for descriptor in descriptors}
            assert {4, 8, 15} <= widths
            assert inventory["MUL_TYPED"] == 1
            assert inventory["GROUP_REDUCE_ADD_TYPED"] == 1


def test_l1_allocator_proves_max_width_slots_are_disjoint():
    i24 = APUG2ScalarType(24, True)
    program = APUG2TypedProgram("add", (i24, i24), i24, (1,))
    allocator = program.manifest()["l1_allocator"]
    assert APUG2_TYPED_L1_SLOT_BASES == (64, 192, 320, 448)
    assert allocator["slot_live_rows_at_max_width"] == 4 * 24 == 96
    assert APUG2_TYPED_L1_SLOT_ROWS == 96
    assert APUG2_TYPED_L1_SLOT_PITCH == 128
    slots = allocator["slots"]
    assert [slot["start_row"] for slot in slots] == [64, 192, 320, 448]
    assert all(slot["live_rows"] == 96 for slot in slots)
    for current, following in zip(slots, slots[1:]):
        assert (
            current["live_end_row_exclusive"]
            <= current["reserved_end_row_exclusive"]
            <= following["start_row"]
        )
    reserved = allocator["gtml_reserved"]
    assert reserved["slots_end_before_temp"] is True
    assert reserved["temp_end_before_index"] is True
    assert slots[-1]["live_end_row_exclusive"] == 544
    assert reserved["temp_start_row"] == 2800
    assert reserved["temp_end_row_exclusive"] == 2928
    assert reserved["index_start_row"] == 2928
    assert reserved["index_end_row_exclusive"] == 2944

    dot = build_apu_g2_typed_recipe(_programs()[-1])
    l1_starts = {
        use.descriptor.start_row
        for operation in dot.operations
        for use in operation.descriptors
        if use.descriptor.storage == "l1"
    }
    assert l1_starts == set(APUG2_TYPED_L1_SLOT_BASES)


def test_narrow_add_oracle_wraps_at_logical_not_storage_width():
    i4 = APUG2ScalarType(4, True)
    signed_program = APUG2TypedProgram("add", (i4, i4), i4, (4,))
    signed_lhs = np.array([7, -8, -7, 3], dtype=np.int8)
    signed_rhs = np.array([7, -1, -3, -8], dtype=np.int8)
    np.testing.assert_array_equal(
        _expected_output(signed_program, (signed_lhs, signed_rhs)),
        np.array([-2, 7, 6, -5], dtype=np.int8),
    )

    u4 = APUG2ScalarType(4, False)
    unsigned_program = APUG2TypedProgram("add", (u4, u4), u4, (4,))
    unsigned_lhs = np.array([15, 14, 8, 0], dtype=np.uint8)
    unsigned_rhs = np.array([1, 3, 9, 15], dtype=np.uint8)
    np.testing.assert_array_equal(
        _expected_output(unsigned_program, (unsigned_lhs, unsigned_rhs)),
        np.array([0, 1, 1, 15], dtype=np.uint8),
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


def test_callable_validates_exact_storage_and_logical_value_widths():
    target = build_apu_g2_target()
    compiled = allo.compile(
        APUG2TypedProgram(
            "dot",
            (I4, I4),
            I15,
            (3, 128),
            reduction_extent=128,
        ),
        target,
        apu_g2_cost,
        backend="virtual",
    )
    good = np.zeros((3, 128), dtype=np.int8)
    assert compiled(good, good).backend == "virtual"
    with pytest.raises(TypeError, match="storage dtype int8"):
        compiled(good.astype(np.int16), good)
    bad = good.copy()
    bad[1, 4] = 8
    with pytest.raises(ValueError, match="outside int4"):
        compiled(bad, good)
    with pytest.raises(ValueError, match="shape"):
        compiled(good[:, :64], good)

    div = allo.compile(
        APUG2TypedProgram("div", (U16, U16), U16, (8,)),
        target,
        apu_g2_cost,
        backend="virtual",
    )
    with pytest.raises(ValueError, match="nonzero divisors"):
        div(np.ones(8, dtype=np.uint16), np.zeros(8, dtype=np.uint16))


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


def test_physical_carrier_pack_and_compact_result_mapping():
    logical = np.arange(3 * 128, dtype=np.int16).reshape(3, 128)
    physical = _pack_operand(logical, np.dtype(np.int16))
    assert physical.shape == (4, 65536)
    assert physical.size == APUG2_TYPED_CAPACITY
    np.testing.assert_array_equal(
        physical.reshape(-1)[: logical.size],
        logical.reshape(-1),
    )
    assert np.all(physical.reshape(-1)[logical.size :] == 0)

    program = APUG2TypedProgram(
        "dot",
        (I4, I4),
        I15,
        (3, 128),
        reduction_extent=128,
    )
    carrier = np.zeros((4, 65536), dtype=np.int16)
    carrier.reshape(-1)[::128][:3] = (-31, 7, 100)
    compact = _logical_output(program, carrier)
    np.testing.assert_array_equal(compact, np.array([-31, 7, 100], dtype=np.int16))


def test_generic_gemm_packing_is_row_major_independent_dot_geometry():
    lhs = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
    rhs = np.array(
        [[2, -1, 3, 0], [4, 2, -2, 1], [-3, 5, 1, -4]],
        dtype=np.int8,
    )
    left_dots, right_dots = pack_apu_g2_gemm_as_independent_dots(lhs, rhs)
    assert left_dots.shape == right_dots.shape == (8, 3)
    compact = (left_dots.astype(np.int16) * right_dots.astype(np.int16)).sum(
        axis=1, dtype=np.int16
    )
    restored = unpack_apu_g2_gemm_dots(compact, 2, 4)
    expected = lhs.astype(np.int16) @ rhs.astype(np.int16)
    np.testing.assert_array_equal(restored, expected)

    benchmark_lhs = np.zeros((64, 128), dtype=np.int8)
    benchmark_rhs = np.zeros((128, 31), dtype=np.int8)
    packed = pack_apu_g2_gemm_as_independent_dots(benchmark_lhs, benchmark_rhs)
    assert packed[0].shape == packed[1].shape == (1984, 128)


def _typed_host_output(device_library):
    return "\n".join(
        (
            "device_pipeline_ticks=1940",
            "device_final_pipeline_ticks=101",
            "h2d_us=1.250",
            "host_task_us=2.500",
            "d2h_us=0.750",
            "end_to_end_us=5.000",
            "target=hardware",
            f"device_library={device_library}",
            "task_status=0",
            "timed_scope=direct_vl64_pipeline",
            "completion_barrier_included=1",
            "independent_final_correctness_call=1",
            f"PASS physical_outputs={APUG2_TYPED_CAPACITY}",
            "typed_operation=4",
            "lhs_bits=8",
            "rhs_bits=0",
            "out_bits=16",
            "log_reduction=8",
            "",
        )
    )


def test_typed_runtime_requires_exact_hardware_child_attestations(tmp_path):
    device_library = (tmp_path / "typed.update.bin").resolve()
    ticks, host = _parse_metrics(
        _typed_host_output(device_library),
        device_library,
    )
    assert ticks == {"pipeline": 1940, "final_pipeline": 101}
    assert host == {
        "h2d": 1.25,
        "host_task": 2.5,
        "d2h": 0.75,
        "end_to_end": 5.0,
    }


@pytest.mark.parametrize(
    "original,replacement,match",
    (
        ("target=hardware", "target=64vl_sim", "target"),
        ("device_library=", "device_library=/wrong/", "device_library"),
        ("task_status=0", "task_status=7", "task_status"),
        (
            "timed_scope=direct_vl64_pipeline",
            "timed_scope=unfenced_issue_only",
            "timed_scope",
        ),
        (
            "completion_barrier_included=1",
            "completion_barrier_included=0",
            "completion_barrier_included",
        ),
        (
            "independent_final_correctness_call=1",
            "independent_final_correctness_call=0",
            "independent_final_correctness_call",
        ),
    ),
)
def test_typed_runtime_rejects_changed_child_attestation(
    tmp_path, original, replacement, match
):
    device_library = (tmp_path / "typed.update.bin").resolve()
    output = _typed_host_output(device_library).replace(original, replacement, 1)
    with pytest.raises(RuntimeError, match=match):
        _parse_metrics(output, device_library)


def test_typed_host_emits_every_required_child_attestation():
    source = (inspect.getmodule(_parse_metrics)._TEMPLATE / "host_typed.cc").read_text()
    for fragment in (
        '"target=hardware\\n"',
        '"device_library=" << DEVICE_SIDE_LIB_LOCATION',
        '"task_status=" << task_rc',
        '"timed_scope=direct_vl64_pipeline\\n"',
        '"completion_barrier_included=1\\n"',
        '"independent_final_correctness_call=1\\n"',
    ):
        assert fragment in source
