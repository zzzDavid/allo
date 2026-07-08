# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buildable persistent-L4 project emitted by the public hybrid compiler."""

from pathlib import Path
import sys

import allo

from allo.pim.apu_v1_hybrid_runtime import (
    emit_apu_v1_hybrid_gemm_project,
    emit_apu_v1_hybrid_three_mm_project,
    emit_apu_v1_hybrid_two_mm_project,
)
from allo.pim.targets import build_apu_v1_target


sys.path.insert(0, str(Path(__file__).parents[1] / "pim"))
from lib.apu_v1 import (  # pylint: disable=wrong-import-position
    build_hybrid_program,
    get_case,
)


def test_canonical_gemm_project_has_one_arena_two_batches_and_one_gather(tmp_path):
    compiled = allo.compile(
        build_hybrid_program("gemm"),
        build_apu_v1_target(),
        backend="functional",
    )
    emitted = emit_apu_v1_hybrid_gemm_project(
        compiled, get_case("gemm").make_inputs(), tmp_path / "project"
    )

    assert (emitted.path / "Makefile").is_file()
    assert (emitted.path / "device.c").is_file()
    assert (emitted.path / "host.c").is_file()
    assert "gdl_mem_handle_t arena" in emitted.host_source
    assert "schedule_phase(ctx, base_cmd, 0, 4" in emitted.host_source
    assert "schedule_phase(ctx, base_cmd, 1, 1" in emitted.host_source
    assert emitted.host_source.count("gdl_mem_cpy_from_dev") == 1
    assert all(
        f"temporal_dma_coalescing_vector_apuc{apuc}" in emitted.device_source
        for apuc in range(4)
    )
    assert "ele_add(" in emitted.device_source
    assert sum(value.nbytes for value in emitted.inputs.values()) > 0


def test_two_mm_project_repackages_between_vector_batches(tmp_path):
    compiled = allo.compile(
        build_hybrid_program("2mm"), build_apu_v1_target(), backend="functional"
    )
    emitted = emit_apu_v1_hybrid_two_mm_project(
        compiled, get_case("2mm").make_inputs(), tmp_path / "project"
    )
    assert emitted.launch_plan == ((0, 4), (1, 4), (2, 4), (3, 1))
    assert emitted.host_source.count("gdl_mem_cpy_from_dev") == 1
    assert "run_repack" in emitted.device_source
    assert "region0_to_region1_apuc0_map" in emitted.inputs


def test_three_mm_project_all_gathers_join_inside_l4(tmp_path):
    compiled = allo.compile(
        build_hybrid_program("3mm"), build_apu_v1_target(), backend="functional"
    )
    emitted = emit_apu_v1_hybrid_three_mm_project(
        compiled, get_case("3mm").make_inputs(), tmp_path / "project"
    )
    assert emitted.launch_plan == ((0, 4), (1, 4), (2, 4), (3, 4), (4, 1))
    assert emitted.host_source.count("gdl_mem_cpy_from_dev") == 1
    assert "run_repack_join" in emitted.device_source
    assert "region0_to_region2_apuc0_map" in emitted.inputs
    assert "region1_to_region2_apuc0_map" in emitted.inputs
