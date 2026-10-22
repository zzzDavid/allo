# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python structural gates for the Gemini-II target and VL64 layout."""

from pathlib import Path

import numpy as np
import pytest

from allo.pim.apu_g2_layout import (
    APUG2_GROUPS,
    APUG2_LANES_PER_GROUP,
    APUG2_MMB_SETS,
    APUG2_VECTOR_LANES,
    build_apu_g2_u16_layout,
)
from allo.pim.targets import build_apu_g2_target
from allo.spmw_linear_layout import LinearLayout


def test_apu_g2_build_templates_keep_publication_optimization_parity():
    template = (
        Path(__file__).resolve().parents[2] / "allo" / "pim" / "templates" / "apu_g2"
    )
    device_cmake = (template / "device" / "CMakeLists.txt").read_text()
    typed_runtime = (
        Path(__file__).resolve().parents[2] / "allo" / "pim" / "apu_g2_typed_runtime.py"
    ).read_text()
    composed_runtime = (
        Path(__file__).resolve().parents[2]
        / "allo"
        / "pim"
        / "apu_g2_composed_runtime.py"
    ).read_text()

    assert "set(CMAKE_BUILD_TYPE ArcDebug)" in device_cmake
    assert "set(GSI_SYSTEM_CMAKE_BUILD_TYPE Release)" in device_cmake
    assert "target_compile_options(tenon_apu_g2_tasks PRIVATE" in device_cmake
    assert "\n  -O3\n  -DNDEBUG)" in device_cmake
    assert '"-DCMAKE_BUILD_TYPE=Release"' in typed_runtime
    assert '"-DCMAKE_BUILD_TYPE=Release"' in composed_runtime


def test_apu_g2_target_is_one_core_with_a_one_dimensional_16_pe_grid():
    target = build_apu_g2_target()

    assert target.name == "apu_v2"
    assert target.work_grid() == ([16], 16)
    assert target.unit("core").mode == "device"
    assert target.unit("pe").axes == {"group": 16}
    assert target.unit("pe").mapping == [16]
    assert target.unit("vector_engine").capacity == 1
    assert target.unit("arc").capacity == 1


def test_apu_g2_target_declares_real_l5_l2_l1_and_mmb_geometry():
    target = build_apu_g2_target()

    assert target.l5.geometry == {
        "size_bytes": 256 << 20,
        "alignment_bytes": 256,
    }
    for half in (target.l2a, target.l2b):
        assert half.geometry == {
            "size_bytes": 4 << 10,
            "blocks_per_half": 16,
            "block_bytes": 256,
        }
    assert target.l1.geometry == {
        "banks": 8,
        "groups": 16,
        "groups_per_bank": 2,
        "cols_per_group": 4096,
        "rows": 3072,
        "cols": 65536,
        "width": 1,
    }
    assert target.mmb.geometry == {
        "sets": 4,
        "rows_per_set": 48,
        "segment_rows": 24,
        "cols": 65536,
        "width": 1,
    }
    assert (target.rwen.lanes, target.rwen.width, target.rwen.slots) == (
        65536,
        1,
        1,
    )


def test_apu_g2_target_exposes_the_validated_direct_vl64_compute_surface():
    target = build_apu_g2_target()
    operations = {
        operation.name: operation
        for unit in target._walk()  # pylint: disable=protected-access
        for operation in unit.ops.values()
    }

    assert set(operations) == {
        "ADD_U16",
        "MUL_U8_TO_U16",
        "GROUP_REDUCE_ADD_U16_TO_U23",
        "LT_U16",
        "MIN_U16",
        "MAX_U16",
        "DIV_U16",
        "SUB_U16",
        "ADD_TYPED",
        "MUL_TYPED",
        "GROUP_REDUCE_ADD_TYPED",
        "DIV_TYPED",
        "COPY_ODD_TO_EVEN_VECTORS",
        "SHIFT_LEFT_U16",
        "SHIFT_RIGHT_U16",
        "SEU_BARRIER",
        "SQUEEZE_ROWS_INPLACE",
        "SPREAD_BLOCK",
        "DISPATCH",
    }
    assert all(not operation.matchable for operation in operations.values())
    for removed_placeholder in (
        "ADD",
        "MUL",
        "MAC",
        "MATMUL",
        "RMS_NORM",
        "SOFTMAX",
        "AF",
    ):
        with pytest.raises(KeyError):
            target.op(removed_placeholder)

    assert {
        target.move("L1_TO_MMB_SEG0").dst.idx,
        target.move("L1_TO_MMB_SEG1").dst.idx,
    } == {"seg0", "seg1"}
    assert target.move("MMB_TO_L1").src.idx == "seg1"
    assert target.move("MMB_TO_L1_BITS").src.idx == "seg1"
    with pytest.raises(KeyError):
        target.move("MMB_SEG1_TO_SEG0")


def test_apu_g2_linear_layout_maps_group_boundaries_and_four_mmb_sets():
    layout = build_apu_g2_u16_layout()

    assert isinstance(layout, LinearLayout)
    assert layout.out_dims == ("l1_column", "mmb_set")
    assert layout.out_sizes == (APUG2_VECTOR_LANES, APUG2_MMB_SETS)
    assert layout.size_of("group") == APUG2_GROUPS
    assert layout.size_of("lane_in_group") == APUG2_LANES_PER_GROUP
    assert layout.size_of("vector") == APUG2_MMB_SETS

    assert layout.apply(group=0, lane_in_group=0, vector=0) == (0, 0)
    assert layout.apply(group=0, lane_in_group=4095, vector=0) == (4095, 0)
    assert layout.apply(group=1, lane_in_group=0, vector=0) == (4096, 0)
    assert layout.apply(group=15, lane_in_group=4095, vector=3) == (65535, 3)


def test_apu_g2_linear_layout_is_bijective_over_the_complete_vector_pack():
    layout = build_apu_g2_u16_layout()
    seen = np.zeros((APUG2_MMB_SETS, APUG2_VECTOR_LANES), dtype=np.bool_)

    for vector in range(APUG2_MMB_SETS):
        for group in range(APUG2_GROUPS):
            for lane in range(APUG2_LANES_PER_GROUP):
                column, mmb_set = layout.apply(
                    group=group,
                    lane_in_group=lane,
                    vector=vector,
                )
                assert not seen[mmb_set, column]
                seen[mmb_set, column] = True

    assert np.all(seen)
    assert layout.image_size(varying_inputs=("group", "lane_in_group", "vector")) == (
        APUG2_MMB_SETS * APUG2_VECTOR_LANES
    )
    assert (
        layout.image_size(
            varying_inputs=("group", "lane_in_group"),
            output_dims=("l1_column",),
        )
        == APUG2_VECTOR_LANES
    )
    assert (
        layout.image_size(varying_inputs=("vector",), output_dims=("mmb_set",))
        == APUG2_MMB_SETS
    )
