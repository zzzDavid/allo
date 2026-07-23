# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software gates for generic LinearLayout-aware APUg2 contractions."""

from pathlib import Path
import inspect

import numpy as np
import pytest

import allo
from allo.pim.apu_g2_composed_layout import (
    APUG2ComposedHostFiles,
    build_apu_g2_composed_host_command,
    canonicalize_apu_g2_values,
    decode_apu_g2_u16_bitpatterns,
    deinterleave_apu_g2_streams,
    expected_apu_g2_composed_output,
    gather_apu_g2_composed_output,
    interleave_apu_g2_dot_streams,
    pack_apu_g2_batched_gemm_dots,
    pack_apu_g2_composed_auxiliary,
    pack_apu_g2_composed_operand,
    pack_apu_g2_matrix_vector_dots,
)
from allo.pim.apu_g2_composed_program import build_apu_g2_composed_recipe
from allo.pim.apu_g2_composed_runtime import _parse_metrics
from allo.pim.apu_g2_layout import APUG2_U16_SHAPE
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


I2 = allo.APUG2ScalarType(2, True)
I3 = allo.APUG2ScalarType(3, True)
I4 = allo.APUG2ScalarType(4, True)
I11 = allo.APUG2ScalarType(11, True)
I14 = allo.APUG2ScalarType(14, True)
I15 = allo.APUG2ScalarType(15, True)


def _profiles(repetitions=7):
    return (
        allo.APUG2ComposedContractionProgram(
            (I4, I4), I15, (64, 31), 128, repetitions=repetitions
        ),
        allo.APUG2ComposedContractionProgram(
            (I4, I4), I15, (4, 32, 16), 128, repetitions=repetitions
        ),
        allo.APUG2ComposedContractionProgram(
            (I2, I2),
            I15,
            (1024,),
            128,
            allo.APUG2DotEpilogue((3, 2)),
            repetitions=repetitions,
        ),
        allo.APUG2ComposedContractionProgram(
            (I3, I3), I14, (256, 2), 256, repetitions=repetitions
        ),
        allo.APUG2ComposedContractionProgram(
            (I3, I3),
            I15,
            (256, 2),
            256,
            allo.APUG2DotEpilogue((1,), I14),
            repetitions=repetitions,
        ),
    )


def _physical_from_logical(program, logical):
    physical = np.zeros(APUG2_U16_SHAPE, dtype=program.output_type.numpy_dtype)
    for output_index, value in enumerate(logical.reshape(-1)):
        row = output_index * program.epilogue.terms_per_output
        column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
        physical[mmb_set, column] = value
    return physical


def test_frozen_profiles_are_derived_from_geometry_type_and_epilogue():
    gemm, batch, gesummv, bicg, mvt = _profiles()
    assert [item.dot_shape for item in (gemm, batch, gesummv, bicg, mvt)] == [
        (1984, 128),
        (2048, 128),
        (2048, 128),
        (512, 256),
        (512, 256),
    ]
    assert [item.dot_type for item in (gemm, batch, gesummv, bicg, mvt)] == [
        I15,
        I15,
        I11,
        I14,
        I14,
    ]
    assert gesummv.auxiliary_type == I3
    assert mvt.auxiliary_type == I14
    assert gesummv.input_shapes == (gesummv.dot_shape, gesummv.dot_shape)
    assert mvt.input_shapes == (
        mvt.dot_shape,
        mvt.dot_shape,
        mvt.output_shape,
    )
    for program in (gemm, batch, gesummv, bicg, mvt):
        manifest = program.manifest()
        layout = manifest["layout"]
        assert layout["schema"] == "apu-g2-dot-linear-layout-v1"
        assert layout["linear_layout"]["kind"] == "linear-layout-f2"
        assert len(layout["fingerprint"]) == 64
        assert len(program.structural_fingerprint) == 64
        assert manifest["structural_fingerprint"] == program.structural_fingerprint
        assert layout["validity"]["logical_rows"] == program.dot_count
        assert layout["validity"]["padded_reduction"] == program.reduction_extent


def test_epilogue_and_width_contracts_fail_closed():
    with pytest.raises(ValueError, match="identity or two"):
        allo.APUG2DotEpilogue((3,))
    with pytest.raises(ValueError, match="output must be exactly 15 bits"):
        allo.APUG2ComposedContractionProgram(
            (I3, I3),
            I14,
            (4,),
            256,
            allo.APUG2DotEpilogue((1,), I14),
        )
    with pytest.raises(ValueError, match="output must be exactly 18 bits"):
        allo.APUG2ComposedContractionProgram(
            (I3, I3),
            I14,
            (4,),
            256,
            allo.APUG2DotEpilogue((3, 2)),
        )
    with pytest.raises(ValueError, match="accumulator must use"):
        allo.APUG2ComposedContractionProgram(
            (I3, I3),
            I15,
            (4,),
            256,
            allo.APUG2DotEpilogue(
                (1,),
                allo.APUG2ScalarType(13, True),
            ),
        )


def test_generic_gemm_and_batched_gemm_dot_adapters_match_numpy():
    rng = np.random.default_rng(11)
    gemm, batch = _profiles()[:2]
    lhs = rng.integers(-8, 8, size=(64, 128), dtype=np.int8)
    rhs = rng.integers(-8, 8, size=(128, 31), dtype=np.int8)
    left_dots, right_dots = pack_apu_g2_batched_gemm_dots(lhs, rhs)
    assert left_dots.shape == right_dots.shape == gemm.dot_shape
    observed = expected_apu_g2_composed_output(gemm, left_dots, right_dots)
    expected = canonicalize_apu_g2_values(
        lhs.astype(np.int64) @ rhs.astype(np.int64),
        I15,
    )
    np.testing.assert_array_equal(observed, expected)

    lhs_batch = rng.integers(-8, 8, size=(4, 32, 128), dtype=np.int8)
    rhs_batch = rng.integers(-8, 8, size=(4, 128, 16), dtype=np.int8)
    left_dots, right_dots = pack_apu_g2_batched_gemm_dots(lhs_batch, rhs_batch)
    assert left_dots.shape == right_dots.shape == batch.dot_shape
    observed = expected_apu_g2_composed_output(batch, left_dots, right_dots)
    expected = canonicalize_apu_g2_values(
        np.matmul(lhs_batch.astype(np.int64), rhs_batch.astype(np.int64)),
        I15,
    )
    np.testing.assert_array_equal(observed, expected)


def test_interleaved_pair_bicg_and_mvt_semantics_match_numpy():
    rng = np.random.default_rng(23)
    _gemm, _batch, gesummv, bicg, mvt = _profiles()

    matrix_a = rng.integers(-2, 2, size=(1024, 128), dtype=np.int8)
    matrix_b = rng.integers(-2, 2, size=(1024, 128), dtype=np.int8)
    vector = rng.integers(-2, 2, size=(128,), dtype=np.int8)
    left, right = interleave_apu_g2_dot_streams(
        pack_apu_g2_matrix_vector_dots(matrix_a, vector),
        pack_apu_g2_matrix_vector_dots(matrix_b, vector),
    )
    observed = expected_apu_g2_composed_output(gesummv, left, right)
    expected = canonicalize_apu_g2_values(
        3 * (matrix_a.astype(np.int64) @ vector.astype(np.int64))
        + 2 * (matrix_b.astype(np.int64) @ vector.astype(np.int64)),
        I15,
    )
    np.testing.assert_array_equal(observed, expected)

    matrix = rng.integers(-4, 4, size=(256, 256), dtype=np.int8)
    p = rng.integers(-4, 4, size=(256,), dtype=np.int8)
    r = rng.integers(-4, 4, size=(256,), dtype=np.int8)
    left, right = interleave_apu_g2_dot_streams(
        pack_apu_g2_matrix_vector_dots(matrix, p),
        pack_apu_g2_matrix_vector_dots(matrix, r, transpose=True),
    )
    observed = expected_apu_g2_composed_output(bicg, left, right)
    expected = canonicalize_apu_g2_values(
        np.stack(
            (
                matrix.astype(np.int64) @ p.astype(np.int64),
                matrix.T.astype(np.int64) @ r.astype(np.int64),
            ),
            axis=1,
        ),
        I14,
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        deinterleave_apu_g2_streams(observed),
        expected.T,
    )

    y1 = rng.integers(-4, 4, size=(256,), dtype=np.int8)
    y2 = rng.integers(-4, 4, size=(256,), dtype=np.int8)
    accumulator = rng.integers(-200, 201, size=(256, 2), dtype=np.int16)
    left, right = interleave_apu_g2_dot_streams(
        pack_apu_g2_matrix_vector_dots(matrix, y1),
        pack_apu_g2_matrix_vector_dots(matrix, y2, transpose=True),
    )
    observed = expected_apu_g2_composed_output(
        mvt, left, right, accumulator=accumulator
    )
    expected = canonicalize_apu_g2_values(
        np.stack(
            (
                matrix.astype(np.int64) @ y1.astype(np.int64),
                matrix.T.astype(np.int64) @ y2.astype(np.int64),
            ),
            axis=1,
        )
        + accumulator.astype(np.int64),
        I15,
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        deinterleave_apu_g2_streams(observed),
        expected.T,
    )


def test_linear_layout_controls_operand_auxiliary_and_result_heads():
    rng = np.random.default_rng(31)
    programs = _profiles()
    for program in programs:
        lhs = rng.integers(
            program.input_types[0].minimum,
            program.input_types[0].maximum + 1,
            size=program.dot_shape,
            dtype=program.input_types[0].numpy_dtype,
        )
        rhs = rng.integers(
            program.input_types[1].minimum,
            program.input_types[1].maximum + 1,
            size=program.dot_shape,
            dtype=program.input_types[1].numpy_dtype,
        )
        accumulator = (
            rng.integers(
                -100,
                101,
                size=program.output_shape,
                dtype=np.int16,
            )
            if program.epilogue.accumulator_type is not None
            else None
        )
        physical_lhs = pack_apu_g2_composed_operand(program, lhs, 0)
        physical_rhs = pack_apu_g2_composed_operand(program, rhs, 1)
        for row in (0, program.dot_count // 2, program.dot_count - 1):
            column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
            stop = column + program.reduction_extent
            np.testing.assert_array_equal(physical_lhs[mmb_set, column:stop], lhs[row])
            np.testing.assert_array_equal(physical_rhs[mmb_set, column:stop], rhs[row])

        auxiliary = pack_apu_g2_composed_auxiliary(program, accumulator)
        if program.epilogue.mode.value == "identity":
            assert auxiliary is None
        elif program.epilogue.mode.value == "pair_affine":
            for row in (0, 1, program.dot_count - 2, program.dot_count - 1):
                column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
                assert int(auxiliary[mmb_set, column]) == (3, 2)[row % 2]
        else:
            for row in (0, program.dot_count - 1):
                column, mmb_set = program.layout_plan.physical_coordinate(row, 0)
                assert int(auxiliary[mmb_set, column]) == int(
                    accumulator.reshape(-1)[row]
                )

        oracle = expected_apu_g2_composed_output(
            program, lhs, rhs, accumulator=accumulator
        )
        physical_output = _physical_from_logical(program, oracle)
        np.testing.assert_array_equal(
            gather_apu_g2_composed_output(program, physical_output),
            oracle,
        )


def test_fixture_decoding_uses_full_u16_twos_complement_payload():
    payload = np.array([0xFFFF, 0xFFFC, 0x0003, 0x0007], dtype="<u2")
    np.testing.assert_array_equal(
        decode_apu_g2_u16_bitpatterns(payload, I3),
        np.array([-1, -4, 3, -1], dtype=np.int8),
    )


def test_recipe_inventory_and_fingerprints_cover_layout_and_epilogue():
    expected = (
        (8, {"MUL_TYPED": 1, "GROUP_REDUCE_ADD_TYPED": 1}),
        (8, {"MUL_TYPED": 1, "GROUP_REDUCE_ADD_TYPED": 1}),
        (
            16,
            {
                "MUL_TYPED": 2,
                "GROUP_REDUCE_ADD_TYPED": 1,
                "COPY_ODD_TO_EVEN_VECTORS": 1,
                "ADD_TYPED": 1,
            },
        ),
        (8, {"MUL_TYPED": 1, "GROUP_REDUCE_ADD_TYPED": 1}),
        (
            12,
            {
                "MUL_TYPED": 1,
                "GROUP_REDUCE_ADD_TYPED": 1,
                "ADD_TYPED": 1,
            },
        ),
    )
    fingerprints = set()
    for program, (operation_count, inventory) in zip(_profiles(), expected):
        recipe = build_apu_g2_composed_recipe(program)
        assert len(recipe.operations) == operation_count
        assert recipe.inventory["SEU_BARRIER"] == 1
        assert recipe.certificate.fully_vectorized
        assert recipe.certificate.vector_lane_updates == 4 * 65536
        assert recipe.metadata["layout_fingerprint"] == (
            program.layout_manifest()["fingerprint"]
        )
        for opcode, calls in inventory.items():
            assert recipe.inventory[opcode] == calls
        fingerprints.add(recipe.structural_fingerprint)
    assert len(fingerprints) == 5

    original = _profiles()[0]
    renamed = allo.APUG2ComposedContractionProgram(
        original.input_types,
        original.output_type,
        original.output_shape,
        original.reduction_extent,
        name="renamed_without_physical_effect",
    )
    assert renamed.structural_fingerprint == original.structural_fingerprint
    assert (
        build_apu_g2_composed_recipe(renamed).structural_fingerprint
        == build_apu_g2_composed_recipe(original).structural_fingerprint
    )


def test_compile_exports_and_semantic_signatures_are_exact():
    target = build_apu_g2_target()
    for program in _profiles():
        compiled = allo.compile(
            program,
            target,
            apu_g2_cost,
            backend="virtual",
        )
        assert isinstance(compiled, allo.APUG2ComposedContractionCallable)
        assert compiled.estimate().cycles > 0
        names = tuple(inspect.signature(compiled).parameters)
        if program.epilogue.mode.value == "accumulate":
            assert names == ("lhs", "rhs", "accumulator", "out")
        else:
            assert names == ("lhs", "rhs", "out")
        assert (
            compiled.execution_graph.metadata["layout_fingerprint"]
            == program.layout_manifest()["fingerprint"]
        )


def test_host_command_and_parser_bind_every_physical_field(tmp_path):
    program = _profiles(repetitions=13)[2]
    files = APUG2ComposedHostFiles(
        tmp_path / "lhs.bin",
        tmp_path / "rhs.bin",
        tmp_path / "aux.bin",
        tmp_path / "out.bin",
    )
    command = build_apu_g2_composed_host_command(program, tmp_path / "host", files)
    assert len(command) == 18
    assert command[1:5] == tuple(
        str(path) for path in (files.lhs, files.rhs, files.auxiliary, files.output)
    )
    assert command[5:17] == (
        "2",
        "1",
        "2",
        "1",
        "11",
        "3",
        "1",
        "15",
        "1",
        "7",
        "1",
        "13",
    )
    assert command[17] == program.layout_manifest()["fingerprint"]

    device = (tmp_path / "device.update.bin").resolve()
    text = "\n".join(
        (
            "device_pipeline_ticks=3900",
            "device_final_pipeline_ticks=2000",
            "h2d_us=1.25",
            "host_task_us=2.5",
            "d2h_us=0.75",
            "end_to_end_us=5.0",
            "target=hardware",
            f"device_library={device}",
            "task_status=0",
            "timed_scope=composed_dot_vl64_pipeline",
            "completion_barrier_included=1",
            "independent_final_correctness_call=1",
            f"layout_fingerprint={program.layout_manifest()['fingerprint']}",
            "epilogue_mode=1",
            "lhs_bits=2",
            "lhs_signed=1",
            "rhs_bits=2",
            "rhs_signed=1",
            "dot_bits=11",
            "auxiliary_bits=3",
            "auxiliary_signed=1",
            "out_bits=15",
            "out_signed=1",
            "log_reduction=7",
            "repetitions=13",
            "PASS physical_outputs=262144",
        )
    )
    ticks, host = _parse_metrics(text, program, device)
    assert ticks == {"pipeline": 3900, "final_pipeline": 2000}
    assert host["end_to_end"] == 5.0
    with pytest.raises(RuntimeError, match="layout_fingerprint"):
        _parse_metrics(
            text.replace(
                program.layout_manifest()["fingerprint"],
                "0" * 64,
            ),
            program,
            device,
        )


def test_templates_expose_one_generic_composed_task_and_no_case_dispatch():
    template = (
        Path(__file__).resolve().parents[2] / "allo" / "pim" / "templates" / "apu_g2"
    )
    device = (template / "device" / "apu_g2_composed_dot.cc").read_text()
    host = (template / "host_composed_dot.cc").read_text()
    cmake = (template / "CMakeLists.txt").read_text()
    device_cmake = (template / "device" / "CMakeLists.txt").read_text()
    assert "tenon_apu_g2_composed_dot" in device
    assert "APUG2_COMPOSED_DOT_PAIR_AFFINE" in device
    assert "APUG2_COMPOSED_DOT_ACCUMULATE" in device
    assert "tenon_apu_g2_composed_dot" in host
    assert "tenon_apu_g2_composed_dot" in cmake
    assert "apu_g2_composed_dot.cc" in device_cmake
    for source in (device, host):
        assert "gesummv" not in source.lower()
        assert "bicg" not in source.lower()
        assert "mvt" not in source.lower()
