# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AttAcc analytical backend: target, tiling domain, counts, cost, route.

No simulator runs here.  Count fixtures were produced once (2026-11-02) by
the DCC trace-count mirror (DCC commit 212f86ea7f08, see
``allo/pim/costs/attacc.py``); DCC is not imported at test time.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16, float32
from allo.pim.attacc_tiling import (
    AttAccDomainError,
    AttAccKernel,
    AttAccTiling,
    attacc_command_counts,
    attacc_geometry,
    attacc_tiling_space,
    rank_attacc_tilings,
)
from allo.pim.costs import attacc_cost
from allo.pim.costs.attacc import (
    AttAccTiming,
    attacc_cost_program,
    attacc_kernel_cycles,
)
from allo.pim.schedule_search import InfeasibleSchedule
from allo.pim.targets import build_attacc_target
from allo.spmw_legality import check_target
from benchmarks.attacc.workloads import attention_region, gemv_region

QKV_NH8 = [[1, 8, 1], [8, 8, 80], [2, 1, 192]]

# (rows, reduction, outputs) -> (paper tiling, closed-form cycles)
PAPER_GEMV = {
    (8, 5120, 384): (QKV_NH8, 7040),
    (32, 5120, 384): ([[1, 16, 2], [16, 4, 80], [1, 1, 384]], 25600),
    (8, 128, 5120): ([[1, 8, 1], [1, 4, 32], [16, 2, 160]], 2432),
    (32, 128, 5120): ([[1, 32, 1], [2, 2, 32], [8, 1, 640]], 8192),
}


def _compile(region, *, cost_spec=attacc_cost, backend="virtual"):
    return allo.compile(
        region,
        build_attacc_target(),
        cost_spec,
        backend=backend,
        reduction_order="reassociate",
    )


def _selected(compiled):
    return compiled.compiled.layout.extra["attacc_tiling"].dcc_tiling()


@pytest.fixture(scope="module")
def geometry():
    return attacc_geometry(build_attacc_target())


@pytest.fixture(scope="module")
def qkv_nh8():
    return _compile(gemv_region(8, 5120, 384))


# --------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------- #


def test_target_is_legal_and_carries_the_paper_geometry(geometry):
    target = build_attacc_target()
    assert check_target(target).violations == ()
    assert target.work_grid() == ([5, 16, 64], 5 * 16 * 64)
    assert (geometry.devices, geometry.channels, geometry.units) == (5, 16, 64)
    assert (geometry.banks_per_group, geometry.bank_groups, geometry.lanes) == (
        4,
        16,
        16,
    )
    assert geometry.bank_bytes == (1 << 14) * (1 << 5) * 32
    for name in ("MAC", "SFM_EXP", "SFM_SUM", "SFM_DIV"):
        target.op(name)
    # The max-stabilized softmax needs a -inf accumulator initializer the
    # matcher does not accept; the op is not declared.
    with pytest.raises(KeyError):
        target.op("SFM_MAX")


# --------------------------------------------------------------------- #
# Tiling domain and counts
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rows, reduction, outputs, size",
    [
        (8, 128, 5120, 1391),
        (32, 128, 5120, 2273),
        (8, 5120, 384, 3043),
        (32, 5120, 384, 4478),
    ],
)
def test_gemv_space_sizes(geometry, rows, reduction, outputs, size):
    space = attacc_tiling_space(
        AttAccKernel("gemv", rows, reduction, outputs), geometry
    )
    assert len(space) == size
    for tiling in space:
        assert tiling.channels <= geometry.channels and tiling.units <= geometry.units
        for (a, b, s), extent in zip(tiling.dims, (rows, reduction, outputs)):
            assert a * b * s >= extent


_DCC_COUNTS = [
    (
        "gemv",
        8,
        5120,
        384,
        [[1, 1, 8], [1, 1, 5120], [9, 33, 2]],
        {
            "PIM_WR_GB": 2621440,
            "PIM_MAC_AB": 81920,
            "PIM_MV_SB": 2048,
            "ST": 2622464,
            "num_itr": 1,
            "itr_size": 8,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        5120,
        384,
        [[1, 1, 8], [2, 50, 52], [7, 1, 55]],
        {
            "PIM_WR_GB": 26624,
            "PIM_MAC_AB": 28160,
            "PIM_MV_SB": 8192,
            "ST": 55296,
            "num_itr": 1,
            "itr_size": 8,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        5120,
        384,
        [[1, 2, 4], [3, 23, 75], [3, 1, 128]],
        {
            "PIM_WR_GB": 19456,
            "PIM_MAC_AB": 40960,
            "PIM_MV_SB": 8192,
            "ST": 52224,
            "num_itr": 1,
            "itr_size": 4,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        5120,
        384,
        [[8, 1, 1], [2, 4, 640], [1, 16, 24]],
        {
            "PIM_WR_GB": 40960,
            "PIM_MAC_AB": 15360,
            "PIM_MV_SB": 512,
            "ST": 43008,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        32,
        5120,
        384,
        [[1, 1, 32], [1, 50, 103], [11, 1, 35]],
        {
            "PIM_WR_GB": 210944,
            "PIM_MAC_AB": 125440,
            "PIM_MV_SB": 24576,
            "ST": 282624,
            "num_itr": 1,
            "itr_size": 32,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        32,
        5120,
        384,
        [[1, 9, 4], [2, 4, 640], [8, 1, 48]],
        {
            "PIM_WR_GB": 163840,
            "PIM_MAC_AB": 122880,
            "PIM_MV_SB": 3072,
            "ST": 176128,
            "num_itr": 1,
            "itr_size": 4,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        32,
        5120,
        384,
        [[16, 2, 1], [1, 31, 166], [1, 1, 384]],
        {
            "PIM_WR_GB": 11264,
            "PIM_MAC_AB": 67584,
            "PIM_MV_SB": 6144,
            "ST": 35840,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        128,
        5120,
        [[1, 1, 8], [1, 6, 22], [14, 9, 41]],
        {
            "PIM_WR_GB": 22528,
            "PIM_MAC_AB": 10496,
            "PIM_MV_SB": 12288,
            "ST": 65536,
            "num_itr": 1,
            "itr_size": 8,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        128,
        5120,
        [[2, 1, 4], [5, 1, 26], [1, 53, 97]],
        {
            "PIM_WR_GB": 7168,
            "PIM_MAC_AB": 12416,
            "PIM_MV_SB": 7168,
            "ST": 32768,
            "num_itr": 1,
            "itr_size": 4,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        8,
        128,
        5120,
        [[8, 1, 1], [2, 4, 16], [1, 16, 320]],
        {
            "PIM_WR_GB": 1024,
            "PIM_MAC_AB": 5120,
            "PIM_MV_SB": 5120,
            "ST": 21504,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        32,
        128,
        5120,
        [[1, 16, 2], [1, 1, 128], [11, 4, 117]],
        {
            "PIM_WR_GB": 16384,
            "PIM_MAC_AB": 29952,
            "PIM_MV_SB": 4096,
            "ST": 31744,
            "num_itr": 1,
            "itr_size": 2,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        32,
        128,
        5120,
        [[16, 2, 1], [1, 5, 26], [1, 5, 1024]],
        {
            "PIM_WR_GB": 4096,
            "PIM_MAC_AB": 32768,
            "PIM_MV_SB": 32768,
            "ST": 135168,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
        },
    ),
    (
        "gemv",
        1024,
        8,
        8192,
        [[1, 1, 1024], [16, 64, 8], [1, 1, 8192]],
        {
            "PIM_WR_GB": 532480,
            "PIM_MAC_AB": 135004160,
            "PIM_MV_SB": 135004160,
            "ST": 540549120,
            "num_itr": 10,
            "itr_size": 103,
            "ch_u": 16,
        },
    ),
    (
        "attention",
        8,
        128,
        480,
        [[1, 1, 8], [1, 7, 19], [9, 6, 9]],
        {
            "PIM_WR_GB": 10240,
            "PIM_MAC_AB": 4736,
            "PIM_MV_SB": 4096,
            "ST": 20480,
            "num_itr": 1,
            "itr_size": 8,
            "ch_u": 16,
            "PIM_MV_GB": 10240,
            "PIM_SFM": 16,
        },
    ),
    (
        "attention",
        8,
        128,
        480,
        [[1, 5, 2], [1, 8, 16], [10, 1, 48]],
        {
            "PIM_WR_GB": 2048,
            "PIM_MAC_AB": 3072,
            "PIM_MV_SB": 512,
            "ST": 4096,
            "num_itr": 1,
            "itr_size": 2,
            "ch_u": 16,
            "PIM_MV_GB": 2048,
            "PIM_SFM": 128,
        },
    ),
    (
        "attention",
        8,
        128,
        480,
        [[8, 1, 1], [2, 4, 16], [1, 16, 30]],
        {
            "PIM_WR_GB": 1024,
            "PIM_MAC_AB": 992,
            "PIM_MV_SB": 256,
            "ST": 2048,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
            "PIM_MV_GB": 1024,
            "PIM_SFM": 16,
        },
    ),
    (
        "attention",
        32,
        128,
        144,
        [[1, 6, 6], [7, 1, 19], [2, 9, 8]],
        {
            "PIM_WR_GB": 16384,
            "PIM_MAC_AB": 3360,
            "PIM_MV_SB": 6144,
            "ST": 32768,
            "num_itr": 1,
            "itr_size": 6,
            "ch_u": 16,
            "PIM_MV_GB": 16384,
            "PIM_SFM": 128,
        },
    ),
    (
        "attention",
        32,
        128,
        144,
        [[2, 16, 1], [1, 3, 43], [7, 1, 21]],
        {
            "PIM_WR_GB": 3072,
            "PIM_MAC_AB": 2384,
            "PIM_MV_SB": 768,
            "ST": 6144,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
            "PIM_MV_GB": 3072,
            "PIM_SFM": 256,
        },
    ),
    (
        "attention",
        32,
        128,
        144,
        [[16, 2, 1], [1, 5, 26], [1, 5, 29]],
        {
            "PIM_WR_GB": 4096,
            "PIM_MAC_AB": 1760,
            "PIM_MV_SB": 1024,
            "ST": 8192,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
            "PIM_MV_GB": 4096,
            "PIM_SFM": 32,
        },
    ),
    (
        "attention",
        8,
        128,
        4096,
        [[1, 1, 8], [1, 4, 32], [12, 16, 22]],
        {
            "PIM_WR_GB": 16384,
            "PIM_MAC_AB": 13824,
            "PIM_MV_SB": 4096,
            "ST": 32768,
            "num_itr": 1,
            "itr_size": 8,
            "ch_u": 16,
            "PIM_MV_GB": 16384,
            "PIM_SFM": 16,
        },
    ),
    (
        "attention",
        8,
        128,
        4096,
        [[8, 1, 1], [2, 4, 16], [1, 16, 256]],
        {
            "PIM_WR_GB": 1024,
            "PIM_MAC_AB": 8192,
            "PIM_MV_SB": 256,
            "ST": 2048,
            "num_itr": 1,
            "itr_size": 1,
            "ch_u": 16,
            "PIM_MV_GB": 1024,
            "PIM_SFM": 16,
        },
    ),
    (
        "attention",
        32,
        128,
        2048,
        [[1, 6, 6], [2, 3, 22], [6, 3, 114]],
        {
            "PIM_WR_GB": 18432,
            "PIM_MAC_AB": 38784,
            "PIM_MV_SB": 6144,
            "ST": 36864,
            "num_itr": 1,
            "itr_size": 6,
            "ch_u": 16,
            "PIM_MV_GB": 18432,
            "PIM_SFM": 128,
        },
    ),
    (
        "attention",
        32,
        128,
        2048,
        [[1, 7, 5], [1, 1, 128], [11, 9, 21]],
        {
            "PIM_WR_GB": 81920,
            "PIM_MAC_AB": 33920,
            "PIM_MV_SB": 20480,
            "ST": 163840,
            "num_itr": 1,
            "itr_size": 5,
            "ch_u": 16,
            "PIM_MV_GB": 81920,
            "PIM_SFM": 128,
        },
    ),
    (
        "attention",
        512,
        256,
        4096,
        [[1, 1, 512], [4, 4, 64], [4, 16, 256]],
        {
            "PIM_WR_GB": 2097152,
            "PIM_MAC_AB": 16777216,
            "PIM_MV_SB": 524288,
            "ST": 4194304,
            "num_itr": 2,
            "itr_size": 256,
            "ch_u": 16,
            "PIM_MV_GB": 2097152,
            "PIM_SFM": 32,
        },
    ),
]


@pytest.mark.parametrize(
    "kind, rows, reduction, outputs, tiling, expected", _DCC_COUNTS
)
def test_counts_equal_the_dcc_trace_counts(
    geometry, kind, rows, reduction, outputs, tiling, expected
):
    counts = attacc_command_counts(
        AttAccKernel(kind, rows, reduction, outputs),
        AttAccTiling.coerce(tiling),
        geometry,
    )
    mine = {
        "PIM_WR_GB": counts.wr_gb,
        "PIM_MAC_AB": counts.mac_ab,
        "PIM_MV_SB": counts.mv_sb,
        "PIM_MV_GB": counts.mv_gb,
        "PIM_SFM": counts.sfm,
        "ST": counts.st,
        "num_itr": counts.iterations,
        "itr_size": counts.rows_per_iteration,
        "ch_u": counts.channels_used,
    }
    assert {key: mine[key] for key in expected} == expected


# --------------------------------------------------------------------- #
# Public route
# --------------------------------------------------------------------- #


def test_acceptance_qkv_batch1(qkv_nh8):
    assert _selected(qkv_nh8) == QKV_NH8
    assert qkv_nh8.estimate().cycles == 7040
    ranking = qkv_nh8.placement_ranking.kernels[0]
    # Every legal tiling is a distinct decision (allowlisted knob field).
    assert len(ranking.scores) == 3043
    assert ranking.scores[0][1] == 7040
    assert qkv_nh8.compiled.cmds == [
        "PIM_WR_GB count=5120",
        "PIM_MAC_AB count=15360",
        "PIM_MV_SB count=3072",
        "ST count=17408",
    ]


@pytest.mark.parametrize("shape", sorted(PAPER_GEMV))
def test_paper_gemv_instances(shape):
    tiling, cycles = PAPER_GEMV[shape]
    compiled = _compile(gemv_region(*shape))
    assert _selected(compiled) == tiling
    assert compiled.estimate().cycles == cycles


@pytest.mark.parametrize(
    "shape, tiling, ties",
    [
        # Worst loss in the paper (953 vs 729 simulated cycles); the
        # closed-form minimum is unique.
        ((8, 128, 480), [[8, 1, 1], [2, 4, 16], [1, 16, 30]], 1),
        # The one attention shape whose closed-form minimum is a 2-way tie.
        ((32, 128, 144), [[1, 32, 1], [4, 2, 16], [3, 1, 48]], 2),
    ],
)
def test_attention_spot_checks(shape, tiling, ties):
    compiled = _compile(attention_region(*shape))
    assert _selected(compiled) == tiling
    assert len(compiled.placement_ranking.kernels[0].cost_ties) == ties
    assert compiled.trace.coverage is not None
    assert [m.target_op_name for m in compiled.trace.matches] == [
        "MAC",
        "SFM_EXP",
        "SFM_SUM",
        "SFM_DIV",
        "MAC",
    ]


def test_virtual_result_kind_and_extra(qkv_nh8):
    W = np.ones((8, 384, 5120), np.float16)
    x = np.ones((8, 5120), np.float16)
    y = np.full((8, 384), 3.0, np.float16)
    result = qkv_nh8(W, x, y)
    assert result.kind == "virtual"
    assert result.cycles == 7040
    assert np.all(y == 3.0)  # analytical: outputs are not written
    extra = result.extra
    assert extra["cost_model"] == "attacc_cost"
    assert extra["priced_target"] == "attacc"
    assert extra["attacc_tiling"] == QKV_NH8
    assert extra["attacc_kernel"] == {
        "kind": "gemv",
        "rows": 8,
        "reduction": 5120,
        "outputs": 384,
    }
    assert extra["attacc_counts"]["PIM_MAC_AB"] == 15360
    assert "ranking model" in extra["cost_validity"]
    assert extra["timing_scope"].startswith("analytical")
    assert "run_ramulator.py" in extra["simulator_timing"]


def test_backend_none_has_no_runner():
    compiled = _compile(gemv_region(4, 256, 64), backend=None)
    assert compiled.compiled.layout.extra["attacc_tiling"] is not None
    with pytest.raises(NotImplementedError, match="analytical-only") as error:
        compiled(
            np.zeros((4, 64, 256), np.float16),
            np.zeros((4, 256), np.float16),
            np.zeros((4, 64), np.float16),
        )
    assert "benchmarks/attacc/scripts/run_ramulator.py" in str(error.value)
    assert "backend='virtual'" in str(error.value)


def test_cost_perturbation_changes_the_selection(geometry):
    # A slower pseudo-channel burst (nBL 2 -> 8) makes store traffic dominate;
    # the same legal space is ranked differently.
    slow_bus = attacc_cost_program(AttAccTiming(n_bl=8), name="attacc_cost_nbl8")
    compiled = _compile(gemv_region(8, 5120, 384), cost_spec=slow_bus)
    assert _selected(compiled) == [[1, 8, 1], [4, 8, 160], [4, 1, 96]]
    assert compiled.estimate().cycles == 8576
    assert len(compiled.placement_ranking.kernels[0].scores) == 3043


def test_selection_is_the_cost_argmin_on_non_paper_shapes(geometry):
    for region, kernel in (
        (gemv_region(4, 1000, 200), AttAccKernel("gemv", 4, 1000, 200)),
        (attention_region(2, 64, 100), AttAccKernel("attention", 2, 64, 100)),
    ):
        compiled = _compile(region)
        selected = compiled.compiled.layout.extra["attacc_tiling"]
        ranked = rank_attacc_tilings(
            kernel, geometry, lambda p: int(attacc_kernel_cycles(p))
        )
        assert compiled.estimate().cycles == ranked[0][1]
        assert selected in [
            tiling for tiling, cycles in ranked if cycles == ranked[0][1]
        ]
        for (a, b, s), extent in zip(selected.dims, kernel.dims):
            assert a * b * s >= extent


_RH, _RK, _RN = 8, 5120, 384


@_df_region()
def renamed_qkv(
    vec: float16[_RH, _RK], mat: float16[_RH, _RN, _RK], out: float16[_RH, _RN]
):
    @allo.work(mapping=[1], args=[vec, mat, out])
    def projection(
        v: float16[_RH, _RK], m: float16[_RH, _RN, _RK], o: float16[_RH, _RN]
    ):
        for head in range(_RH):
            for col in range(_RN):
                partial: float16 = 0
                for red in range(_RK):
                    partial += v[head, red] * m[head, col, red]
                o[head, col] = partial


def test_renamed_workload_keeps_the_tiling(qkv_nh8):
    compiled = _compile(renamed_qkv)
    assert _selected(compiled) == _selected(qkv_nh8)
    assert compiled.estimate().cycles == qkv_nh8.estimate().cycles


_FH, _FK = 64, 32


@_df_region()
def fp32_gemv(W: float32[_FH, _FK], x: float32[_FK], y: float32[_FH]):
    @allo.work(mapping=[1], args=[W, x, y])
    def mv(lW: float32[_FH, _FK], lx: float32[_FK], ly: float32[_FH]):
        for n in range(_FH):
            acc: float32 = 0.0
            for k in range(_FK):
                acc += lW[n, k] * lx[k]
            ly[n] = acc


@_df_region()
def gemv_two_items(W: float16[_FH, _FK], x: float16[_FK], y: float16[_FH]):
    @allo.work(mapping=[2], args=[W, x, y])
    def mv(lW: float16[_FH, _FK], lx: float16[_FK], ly: float16[_FH]):
        (i,) = allo.get_wid()
        for n in range(32):
            acc: float16 = 0
            for k in range(_FK):
                acc += lW[i * 32 + n, k] * lx[k]
            ly[i * 32 + n] = acc


def test_shapes_outside_the_domain_raise(geometry):
    # One row of a 2^17 x 2^17 matrix spread over all 1,024 banks still
    # exceeds a 16 MiB bank: no legal tiling.
    with pytest.raises(AttAccDomainError, match="no legal tiling"):
        _compile(gemv_region(1, 1 << 17, 1 << 17))
    with pytest.raises(AttAccDomainError, match="FP16"):
        _compile(fp32_gemv)
    with pytest.raises(AttAccDomainError, match="mapping=\\[1\\]"):
        _compile(gemv_two_items)


def test_illegal_layout_is_rejected(qkv_nh8):
    from allo.spmw_attacc import attacc_placement

    kernel = qkv_nh8.compiled.layout.extra["attacc_kernel"]
    target = build_attacc_target()
    good = attacc_placement(kernel, QKV_NH8)
    compiled = allo.compile(
        gemv_region(8, 5120, 384),
        target,
        attacc_cost,
        backend="virtual",
        reduction_order="reassociate",
        layout=good,
    )
    assert compiled.estimate().cycles == 7040
    # 32 channels on one device of 16, and a reduction tile that misses
    # the extent: rejected at materialization.
    for bad in (
        [[2, 8, 1], [16, 8, 40], [1, 1, 384]],
        [[1, 8, 1], [8, 8, 79], [2, 1, 192]],
    ):
        with pytest.raises(InfeasibleSchedule, match="illegal tiling"):
            allo.compile(
                gemv_region(8, 5120, 384),
                target,
                attacc_cost,
                backend="virtual",
                reduction_order="reassociate",
                layout=attacc_placement(kernel, bad),
            )


def test_layout_without_a_tiling_is_refused():
    target = build_attacc_target()
    from allo.spmw_autoschedule import Placement

    with pytest.raises(InfeasibleSchedule, match="no attacc_tiling"):
        allo.compile(
            gemv_region(8, 5120, 384),
            target,
            attacc_cost,
            backend="virtual",
            reduction_order="reassociate",
            layout=Placement(placements={}),
        )


@pytest.mark.paper_full
def test_paper_selector_on_all_500_instances(geometry):
    """The port's selector (integer cycles, structural tie-break order as in
    the compiler's ranking) against the paper's selection for all 500."""
    from benchmarks.attacc.workloads import GPT13B_INSTANCES, paper_selection

    expected = paper_selection()
    mismatches = []
    for item in GPT13B_INSTANCES:
        kernel = AttAccKernel(item.family, item.rows, item.reduction, item.outputs)
        ranked = rank_attacc_tilings(
            kernel, geometry, lambda p: int(attacc_kernel_cycles(p))
        )
        best = ranked[0][1]
        chosen = min(
            (t for t, c in ranked if c == best), key=lambda t: repr(t.manifest())
        )
        if chosen.dcc_tiling() != expected[item.id]:
            mismatches.append(item.id)
    assert mismatches == []
