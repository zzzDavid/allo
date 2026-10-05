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
from allo.pim.upmem_abi import TensorLayout


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


# --------------------------------------------------------------------- #
# UPMEM
# --------------------------------------------------------------------- #

UPMEM_M = UPMEM_K = 256


def upmem_gemv(W: int32[UPMEM_M, UPMEM_K], x: int32[UPMEM_K], y: int32[UPMEM_M]):
    for i in range(UPMEM_M):
        acc: int32 = 0
        for j in range(UPMEM_K):
            acc += W[i, j] * x[j]
        y[i] = acc


def test_gemv_upmem():
    program = allo.UPMEMProgram(
        [allo.UPMEMPhase(upmem_gemv, name="gemv", parallel_workers=64)],
        arrays=(
            allo.UPMEMArray("W", TensorLayout.BLOCK, partition_axis=0),
            allo.UPMEMArray("x", TensorLayout.BROADCAST),
            allo.UPMEMArray("y", TensorLayout.BLOCK, partition_axis=0),
        ),
    )
    compiled = _compile_within(300.0, program, build_upmem_target(), upmem_cost)

    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(UPMEM_M, UPMEM_K), dtype=np.int32)
    x = rng.integers(-8, 8, size=UPMEM_K, dtype=np.int32)
    y = np.zeros(UPMEM_M, dtype=np.int32)
    result = compiled(W, x, y)

    np.testing.assert_array_equal(y, W @ x)
    assert result.cycles is None
    assert result.extra["functional_oracle"] is True
    assert compiled.estimate().cycles > 0


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
