# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR-driven UPMEM program compilation and functional ABI tests."""

from dataclasses import dataclass

import numpy as np

import allo
from allo.pim.upmem_program import UPMEMDenseTile, UPMEMDotTile, UPMEMRank1Tile
from allo.ir.types import int32
from allo.pim.costs.upmem import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.pim.upmem_abi import TensorLayout


def _parallel_add(a: int32[128], b: int32[128], out: int32[128]):
    for i in range(128):
        out[i] = a[i] + b[i]


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
