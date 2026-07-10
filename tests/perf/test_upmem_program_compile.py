# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR-driven UPMEM program compilation and functional ABI tests."""

import importlib
import inspect
from dataclasses import dataclass

import numpy as np
import pytest

import allo
from allo.pim.upmem_program import UPMEMDenseTile, UPMEMDotTile, UPMEMRank1Tile
from allo.ir.types import int32
from allo.pim.upmem_analysis import UnsupportedDynamicBoundError
from allo.pim.costs.upmem import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.pim.upmem_abi import TensorLayout


def _parallel_add(a: int32[128], b: int32[128], out: int32[128]):
    for i in range(128):
        out[i] = a[i] + b[i]


def _non_power_of_two_copy(a: int32[120], out: int32[120]):
    for i in range(120):
        out[i] = a[i]


def _dynamic_copy(a: int32[16], out: int32[16], size: int32):
    for i in range(size):
        out[i] = a[i]


def _program():
    @dataclass(frozen=True)
    class PlannedPhase:
        name: str
        barrier: str

    return allo.UPMEMProgram(
        [
            allo.UPMEMPhase(
                _parallel_add,
                name="parallel_add",
                parallel_workers=64,
            )
        ],
        name="vector_program",
        arrays=(
            allo.UPMEMArray("a", TensorLayout.BROADCAST),
            allo.UPMEMArray("b", TensorLayout.BLOCK, partition_axis=0),
            allo.UPMEMArray("out", TensorLayout.BLOCK, partition_axis=0),
        ),
        orchestration={
            "phases": (
                PlannedPhase("scatter", "host"),
                PlannedPhase("compute", "host:gather"),
            )
        },
    )


def test_upmem_program_bypasses_matcher_and_retains_mlir_c_and_abi(monkeypatch):
    compiler_api = importlib.import_module("allo.compiler")
    monkeypatch.setattr(
        compiler_api,
        "match_workload",
        lambda *_args, **_kwargs: pytest.fail("UPMEMProgram used legacy matcher"),
    )
    target = build_upmem_target()
    module = allo.compile(_program(), target, upmem_cost, backend="functional")

    assert inspect.signature(module) == inspect.Signature(
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ("a", "b", "out")
    )
    assert module.cost.spec is upmem_cost
    assert len(module.compiled.source_mlir) == 1
    assert "affine.for" in module.compiled.source_mlir[0]
    assert len(module.compiled.lowered_mlir) == 1
    assert "#pragma omp parallel for schedule(static) num_threads(64)" in (
        module.compiled.c_source[0]
    )
    manifest = module.abi_manifest
    assert manifest["oracle_launches"][0]["num_dpus"] == 64
    assert manifest["oracle_launches"][0]["tensors"][0]["layout"] == "broadcast"
    assert manifest["oracle_launches"][0]["tensors"][1]["layout"] == "block"
    assert manifest["oracle_launches"][0]["tensors"][1]["partition_axis"] == 0
    out_manifest = manifest["oracle_launches"][0]["tensors"][2]
    assert out_manifest["linear_layout"]["kind"] == "linear-layout-f2"
    assert [axis["name"] for axis in out_manifest["linear_layout"]["output_dims"]] == [
        "dpu",
        "tasklet",
        "local",
    ]

    launch = module.abi.launches[0]
    assert launch.tensor_coordinates("a", (17,), dpu_id=37)["dpu"] == 37
    assert launch.tensor_coordinates("out", (0,)) == {
        "dpu": 0,
        "tasklet": 0,
        "local": 0,
    }
    assert launch.tensor_coordinates("out", (127,)) == {
        "dpu": 63,
        "tasklet": 1,
        "local": 0,
    }
    assert launch.spatial_parallelism("out") == 64
    assert launch.tasklet_parallelism("out") == 2
    assert module.execution_graph.metadata["layout_parallelism"] == [
        {"phase": "parallel_add", "dpu": 64, "tasklet": 2}
    ]
    assert [phase["name"] for phase in manifest["planned_orchestration"]["phases"]] == [
        "scatter",
        "compute",
    ]
    assert module.execution_graph.metadata["orchestration"] == (
        manifest["planned_orchestration"]
    )
    assert "__host tenon_upmem_launch_parallel_add_t DPU_INPUT_ARGUMENTS" in (
        module.compiled.device_c_abi[0]
    )
    artifact = module.compiled.device_c_artifacts[0]
    assert artifact.sdk_complete is True
    assert artifact.schedule_realizable is True
    assert artifact.missing_capabilities == ()
    assert manifest["device_c_artifacts"][0]["schedule_realizable"] is True
    artifact.require_schedule_realizable()
    device_source = module.compiled.device_c_source[0]
    assert "int main(void)" in device_source
    assert "#define TENON_NUM_TASKLETS 16" in device_source
    assert "#define TENON_ACTIVE_TASKLETS 2" in device_source
    assert "mram_read" in device_source
    assert "mram_write" in device_source
    assert "barrier_wait" in device_source
    assert "#pragma omp" not in device_source


def test_functional_run_mutates_output_and_round_trips_device_abi():
    module = allo.compile(_program(), build_upmem_target(), upmem_cost)
    a = np.arange(128, dtype=np.int32)
    b = np.arange(128, dtype=np.int32)[::-1].copy()
    out = np.zeros(128, dtype=np.int32)

    result = module(a, b, out)

    np.testing.assert_array_equal(out, a + b)
    assert result.cycles is None
    assert result.extra["functional_oracle"] is True
    assert result.extra["simulator_cycles"] is False
    assert result.extra["packed_launches"][0]["packed_dpus"] == 64
    assert result.extra["packed_launches"][0]["metadata_image_bytes"] % 8 == 0
    assert module.last_result is result
    estimate = module.estimate()
    assert estimate.cycles > 0
    assert module.execution_graph.metadata["simulator_cycles"] is False


def test_display_string_parallel_loop_selection_is_rejected():
    with pytest.raises(ValueError, match="display-string UPMEM parallel_loops"):
        allo.UPMEMPhase(_parallel_add, parallel_loops=("missing",))


def test_costed_compile_rejects_unproven_dynamic_loop_trip_count():
    program = allo.UPMEMProgram(
        [allo.UPMEMPhase(_dynamic_copy)],
        arrays=(allo.UPMEMArray("a"), allo.UPMEMArray("out")),
    )

    with pytest.raises(
        UnsupportedDynamicBoundError,
        match="cannot prove loop trip count",
    ):
        allo.compile(program, build_upmem_target(), upmem_cost)


class _BuiltWorkload:
    def build(self):
        return _program()


def test_workload_build_may_return_upmem_program():
    module = allo.compile(_BuiltWorkload(), build_upmem_target())
    assert module.program.name == "vector_program"


def test_plain_callable_cannot_enter_removed_upmem_matcher_path():
    with pytest.raises(TypeError, match="contraction-only UPMEM matcher.*removed"):
        allo.compile(_parallel_add, build_upmem_target(), upmem_cost)


def test_non_power_of_two_extent_uses_realized_linear_layout_fanout():
    program = allo.UPMEMProgram(
        [
            allo.UPMEMPhase(
                _non_power_of_two_copy,
            )
        ],
        arrays=(allo.UPMEMArray("a"), allo.UPMEMArray("out")),
    )
    module = allo.compile(program, build_upmem_target(), upmem_cost)
    launch = module.abi.launches[0]

    assert launch.spatial_parallelism("out") == 64
    assert launch.tensor_coordinates("out", (119,)) == {
        "dpu": 55,
        "tasklet": 1,
        "local": 0,
    }
    assert launch.tensor_slot("out").shards[56].owned_extent == 1
    assert module.execution_graph.metadata["layout_parallelism"] == [
        {"phase": "_non_power_of_two_copy", "dpu": 64, "tasklet": 2}
    ]

    a = np.arange(120, dtype=np.int32)
    out = np.zeros_like(a)
    module(a, out)
    np.testing.assert_array_equal(out, a)


def test_upmem_dense_tile_emits_complete_int32_dpu_program():
    tile = UPMEMDenseTile(16, 16, 1200, 60, 8)
    source = tile.device_source()

    assert tile.rows_per_tasklet == 2
    assert tile.reduction_tiles == 20
    assert tile.mram_offsets == (0, 76800, 153600)
    assert tile.manifest()["sdk_compilable_translation_unit"] is True
    assert "int main(void)" in source
    assert "mram_read" in source
    assert "mram_write" in source
    assert "if (tid < 8)" in source
    assert "for (uint32_t kt = 0; kt < 20; ++kt)" in source


def test_upmem_dot_tile_emits_tasklet_reduction_and_aligned_dma():
    tile = UPMEMDotTile(2048, 16, 128)
    source = tile.device_source()

    assert tile.chunks_per_tasklet == 1
    assert tile.mram_offsets == (0, 8192, 16384)
    assert "int main(void)" in source
    assert "tenon_partial[16]" in source
    assert "mram_read" in source
    assert "mram_write" in source


def test_upmem_rank1_tile_emits_disjoint_tasklet_slices():
    tile = UPMEMRank1Tile(2048, 16)
    source = tile.device_source()

    assert tile.elements_per_tasklet == 128
    assert tile.mram_offsets == (0, 8192, 16384)
    assert "int main(void)" in source
    assert "128 * tid" in source
    assert "a[i] += scalar[0] * v[i]" in source
