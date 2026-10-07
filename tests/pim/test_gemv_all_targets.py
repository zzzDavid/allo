# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One GEMV through public ``allo.compile`` on every PIM target that can run it.

Each row times the compile against a budget, then runs behind its environment
gate. Paper cycle numbers live only in ``test_paper_golden.py``.

APU v2 has no row: neither its hardware nor its simulator is present here, and
its GEMV vector path was deleted. The composed and typed APU v2 compile-only
tests keep that backend's code compiling.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16, float16, int32
from allo.pim.costs import aim_cost, apu_v1_cost, samsung_cost, upmem_cost
from allo.pim.targets import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)


def _compile_within(budget_s, *args, **kwargs):
    started = time.perf_counter()
    compiled = allo.compile(*args, **kwargs)
    elapsed = time.perf_counter() - started
    assert elapsed < budget_s, f"compile took {elapsed:.1f} s (budget {budget_s} s)"
    return compiled


# --------------------------------------------------------------------- #
# Samsung HBM-PIM
# --------------------------------------------------------------------- #


def _samsung_case():
    path = Path(__file__).resolve().parent / "samsung_hbm_pim" / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("gemv_all_targets_samsung", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"]


def test_gemv_samsung_hbm_pim(samsung_sim_gate):
    workload, M, K = _samsung_case()
    compiled = _compile_within(120.0, workload, build_samsung_target(), samsung_cost)
    assert compiled.placement_ranking.kernels[0].bucket_count == 128

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert result.cycles > 0
    np.testing.assert_allclose(
        y.astype(np.float32),
        W.astype(np.float32) @ x.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


# --------------------------------------------------------------------- #
# SK hynix AiM
# --------------------------------------------------------------------- #

AIM_M = AIM_K = 256
_AIM_MAPPING = [32]
_AIM_ROWS = AIM_M // _AIM_MAPPING[0]


@_df_region()
def aim_gemv(W: bfloat16[AIM_M, AIM_K], x: bfloat16[AIM_K], y: bfloat16[AIM_M]):
    @allo.work(mapping=_AIM_MAPPING, args=[W, x, y])
    def mv(local_W: bfloat16[AIM_M, AIM_K], local_x: bfloat16[AIM_K], local_y: bfloat16[AIM_M]):
        (channel,) = allo.get_wid()
        row0 = channel * _AIM_ROWS
        for i in range(_AIM_ROWS):
            acc: bfloat16 = 0
            for j in range(AIM_K):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


def test_gemv_aim(aim_sim_gate):
    compiled = _compile_within(120.0, aim_gemv, build_aim_target(), aim_cost)

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((AIM_M, AIM_K)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(AIM_K) * 0.1).astype(np.float32)
    y = np.zeros(AIM_M, dtype=np.float32)
    result = compiled(W, x, y)

    # The AiM run replays a ramulator2 trace and reports cycles only.
    assert result.cycles > 0
    assert result.backend == "aim"
    assert "[AiMTrace]" in result.stdout

    (kernel,) = compiled.placement_ranking.kernels
    modes = {position: mode for position, _cycles, mode in kernel.scores}
    assert {mode.split("+")[0] for mode in modes.values()} == {
        "all_bank",
        "single_bank",
    }
    assert any("+bank_scope=sbk" in mode for mode in modes.values())
    assert "+bank_scope=abk" in modes[kernel.chosen_position]
    assert compiled.compiled.layout_ctx.lowering_manifest[0]["bank_scope"] == "abk"


def test_gemv_aim_forced_sbk_expands_layout_banks():
    from allo.spmw_autoschedule import _aim_layout_candidates
    from allo.spmw_codegen import simulator_unavailable_reason

    target = build_aim_target()
    ranked = allo.compile(aim_gemv, target, aim_cost, backend="virtual")
    sbk_layout, abk_layout = _aim_layout_candidates(
        target, ranked.compiled.trace.matches
    )
    assert sbk_layout.extra["bank_fanout"] == 1
    # Both layouts carry no knob values, so they lower the same contraction.
    abk = allo.compile(aim_gemv, target, aim_cost, layout=abk_layout)
    sbk = allo.compile(aim_gemv, target, aim_cost, layout=sbk_layout)

    expected_banks = sorted(
        {
            sbk_layout.layout.apply(k=0, work_bank=w, lane_bank=0)[0]
            for w in range(sbk_layout.layout.size_of("work_bank"))
        }
    )
    abk_cmds = list(abk.compiled.cmds)
    sbk_cmds = list(sbk.compiled.cmds)
    abk_macs = [line for line in abk_cmds if line.startswith("AiM MAC_ABK ")]
    sbk_macs = [line.split() for line in sbk_cmds if line.startswith("AiM MAC_SBK ")]
    assert not any(line.startswith("AiM MAC_ABK ") for line in sbk_cmds)
    assert len(sbk_macs) == len(expected_banks) * len(abk_macs) == 16 * len(abk_macs)
    for launch, abk_line in enumerate(abk_macs):
        _aim, _op, op_size, mask, row = abk_line.split()
        group = sbk_macs[launch * 16 : (launch + 1) * 16]
        assert [int(fields[4]) for fields in group] == expected_banks
        assert {(f[2], f[3], f[5]) for f in group} == {(op_size, mask, row)}
    # Channel-scoped commands are unchanged by the expansion.
    strip = lambda cmds: [c for c in cmds if not c.startswith("AiM MAC_")]
    assert strip(sbk_cmds) == strip(abk_cmds)

    assert sbk.estimate().cycles > abk.estimate().cycles
    if simulator_unavailable_reason("aim") is None:
        operands = (
            np.zeros((AIM_M, AIM_K), np.float32),
            np.zeros(AIM_K, np.float32),
            np.zeros(AIM_M, np.float32),
        )
        assert sbk(*operands).cycles > abk(*operands).cycles


# --------------------------------------------------------------------- #
# UPMEM
# --------------------------------------------------------------------- #

# 768 x 128 over 8 DPUs gives each DPU the 96 x 128 slice of the paper's
# MTV/GEMV rows; the calibrated MV plans need rows divisible by 12 tasklets.
UPMEM_M, UPMEM_K = 768, 128
_UPMEM_MAPPING = [8]
_UPMEM_ROWS = UPMEM_M // _UPMEM_MAPPING[0]


@_df_region()
def upmem_gemv(W: int32[UPMEM_M, UPMEM_K], x: int32[UPMEM_K], y: int32[UPMEM_M]):
    @allo.work(mapping=_UPMEM_MAPPING, args=[W, x, y])
    def mv(local_W: int32[UPMEM_M, UPMEM_K], local_x: int32[UPMEM_K], local_y: int32[UPMEM_M]):
        (dpu,) = allo.get_wid()
        for i in range(_UPMEM_ROWS):
            local_y[dpu * _UPMEM_ROWS + i] = 0
            for j in range(UPMEM_K):
                local_y[dpu * _UPMEM_ROWS + i] += (
                    local_W[dpu * _UPMEM_ROWS + i, j] * local_x[j]
                )


@_df_region()
def upmem_gemv_row0(W: int32[UPMEM_M, UPMEM_K], x: int32[UPMEM_K], y: int32[UPMEM_M]):
    @allo.work(mapping=_UPMEM_MAPPING, args=[W, x, y])
    def mv(local_W: int32[UPMEM_M, UPMEM_K], local_x: int32[UPMEM_K], local_y: int32[UPMEM_M]):
        (dpu,) = allo.get_wid()
        row0 = dpu * _UPMEM_ROWS
        for i in range(_UPMEM_ROWS):
            local_y[row0 + i] = 0
            for j in range(UPMEM_K):
                local_y[row0 + i] += local_W[row0 + i, j] * local_x[j]


@_df_region()
def upmem_gemv_data_temp(W: int32[UPMEM_M, UPMEM_K], x: int32[UPMEM_K], y: int32[UPMEM_M]):
    @allo.work(mapping=_UPMEM_MAPPING, args=[W, x, y])
    def mv(local_W: int32[UPMEM_M, UPMEM_K], local_x: int32[UPMEM_K], local_y: int32[UPMEM_M]):
        (dpu,) = allo.get_wid()
        for i in range(_UPMEM_ROWS):
            t: int32 = local_W[dpu * _UPMEM_ROWS + i, 0] - 1
            local_y[dpu * _UPMEM_ROWS + i] = t
            for j in range(UPMEM_K):
                local_y[dpu * _UPMEM_ROWS + i] += (
                    local_W[dpu * _UPMEM_ROWS + i, j] * local_x[j]
                )


def test_gemv_upmem():
    from allo import spmw_simenv

    compiled = _compile_within(120.0, upmem_gemv, build_upmem_target(), upmem_cost)
    ctx = compiled.compiled.layout_ctx
    (segment,) = ctx.segments
    assert segment.kernel.family == "matrix_vector"
    assert segment.kernel.dpus == _UPMEM_MAPPING[0]
    assert dict(segment.kernel.geometry) == {
        "batches": 1,
        "rows": _UPMEM_ROWS,
        "columns": UPMEM_K,
    }
    assert segment.executions == 1
    assert compiled.compiled.cmds[0].startswith("segment 0 matrix_vector dpus=8 ")
    route, segments = ctx.runtime_route(compiled.compiled.cmds)
    assert route == "upimulator-generic-fixture" and len(segments) == 1
    assert compiled.estimate().cycles > 0

    # Ruling 013-R3: a scratch that only feeds addresses (row0) is exempt from
    # the coverage check; a data temporary feeding a stored value is rejected.
    from allo.customize import customize
    from allo.spmw_upmem import _address_only_scratch, _functions, _walk

    def address_only(workload, name):
        schedule = customize(workload, enable_tensor=False)
        function = _functions(schedule.module)["mv_0"]
        (alloc,) = [
            op.results[0]
            for op in _walk(function.regions[0].blocks[0], [])
            if op.name == "memref.alloc" and str(op.attributes["name"]) == f'"{name}"'
        ]
        return _address_only_scratch(function, alloc)

    assert address_only(upmem_gemv_row0, "row0")
    assert not address_only(upmem_gemv_data_temp, "t")
    with pytest.raises(NotImplementedError, match="store to t .* not covered"):
        allo.compile(upmem_gemv_data_temp, build_upmem_target(), upmem_cost)

    import conftest

    conftest._simulator_gate("upmem")
    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(UPMEM_M, UPMEM_K), dtype=np.int32)
    x = rng.integers(-8, 8, size=UPMEM_K, dtype=np.int32)
    y = np.zeros(UPMEM_M, dtype=np.int32)
    result = compiled(W, x, y)

    np.testing.assert_array_equal(y, W @ x)
    assert result.backend == "upmem"
    assert result.cycles > 0
    (record,) = result.extra["upmem"]
    assert record["num_dpus"] == _UPMEM_MAPPING[0]
    assert len(record["logic_cycles_per_dpu"]) == _UPMEM_MAPPING[0]
    assert result.cycles == max(record["logic_cycles_per_dpu"].values())
    assert record["simulator_sha256"] == spmw_simenv.upimulator_sha256()
    assert record["sdk_build_sha256"] == spmw_simenv.UPMEM_SDK_BUILD_MANIFEST_SHA256


# --------------------------------------------------------------------- #
# GSI APU v1
# --------------------------------------------------------------------- #

APU_M = 1024
APU_K = 64


def apu_v1_gemv(W: float16[APU_M, APU_K], x: float16[APU_K], y: float16[APU_M]):
    for i in allo.grid(APU_M):
        for k in allo.reduction(APU_K):
            y[i] += W[i, k] * x[k]


def test_gemv_apu_v1(request):
    compiled = _compile_within(300.0, apu_v1_gemv, build_apu_v1_target(), apu_v1_cost)
    assert compiled.realization is not None, compiled.realization_error

    request.getfixturevalue("apu_v1_device_gate")
    import conftest

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((APU_M, APU_K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(APU_K) * 0.1).astype(np.float16)
    y = np.zeros(APU_M, dtype=np.float16)
    with conftest.board_lock():
        result = compiled(W, x, y)

    np.testing.assert_allclose(
        y.astype(np.float32),
        W.astype(np.float32) @ x.astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    assert result.cycles > 0
