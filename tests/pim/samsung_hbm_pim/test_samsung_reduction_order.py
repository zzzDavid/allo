# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung ordered program (release issue S-03) on PIMSimulator.

Under the default ``reduction_order="strict"`` the paper GEMV compiles to the
ordered program (conductor ``TEMPORAL_REDUCE``): one output row per FP16 lane,
the reduction issued in source order, no host arithmetic.  Its outputs equal,
bit for bit, a NumPy reference that rounds every multiply and add to FP16 in
source order.  The reassociating REDUCE program does not, and matches the
FP32 reference within tolerance.  Design:
docs/release/design/reduction-policy.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo import spmw_samsung
from allo.spmw_simenv import pimsim_root


def _paper_gemv():
    path = Path(__file__).resolve().parent / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_order_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"]


GM, GK, GN, GROWS = 2048, 64, 2, 16


@_df_region()
def small_gemm(A: fp16[GM, GK], B: fp16[GK, GN], C: fp16[GM, GN]):
    @allo.work(mapping=[16, 8], args=[A, B, C])
    def kernel(lA: fp16[GM, GK], lB: fp16[GK, GN], lC: fp16[GM, GN]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * GROWS
        for i in range(GROWS):
            for j in range(GN):
                acc: fp16 = 0
                for k in range(GK):
                    acc += lA[row0 + i, k] * lB[k, j]
                lC[row0 + i, j] = acc


def source_order_fp16(W, X, initial):
    """``initial + sum_k W[:, k] * X[k]`` rounding every multiply and add to
    FP16, k ascending.  ``X`` is (K,) or (K, N); ``initial`` matches the output."""
    acc = np.array(initial, dtype=np.float16, copy=True)
    vector = X.ndim == 1
    for k in range(W.shape[1]):
        term = W[:, k] * X[k] if vector else W[:, k, None] * X[k][None, :]
        acc = (acc + term.astype(np.float16)).astype(np.float16)
    return acc


@pytest.fixture(scope="module")
def paper_operands():
    _workload, M, K = _paper_gemv()
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    return W, x


@pytest.fixture(scope="module")
def strict_gemv():
    workload, _M, _K = _paper_gemv()
    return allo.compile(workload, build_samsung_target(), samsung_cost)


def test_strict_default_runs_the_ordered_program(
    samsung_sim_gate, strict_gemv, paper_operands
):
    W, x = paper_operands
    M, K = W.shape
    y = np.ones(M, dtype=np.float16)
    result = strict_gemv(W, x, y)

    assert result.kind == "simulator"
    assert result.extra["kernel"] == "GENERIC_TEMPORAL_REDUCE"
    assert result.extra["reduction_order"] == "strict"
    # Measured 2026-11-02 (allo/pim/costs/samsung.py provenance).
    assert result.cycles == 26807
    assert sum(result.extra["cycle_phases"].values()) == result.cycles
    reference = source_order_fp16(W, x, np.zeros(M, np.float16))
    np.testing.assert_array_equal(y.view(np.uint16), reference.view(np.uint16))
    receipt = result.extra["conductor"]
    assert (receipt["ordered_k"], receipt["host_arithmetic"]) == (1, 0)
    assert (receipt["rows"], receipt["depth"], receipt["columns"]) == (M, K, 1)
    # One row tile is 64 channels x 8 PIM units x 16 lanes.
    assert result.extra["logical_shape"] == {"rows": M, "reduction": K, "columns": 1}
    assert result.extra["padded_shape"] == {"rows": 8192, "reduction": K, "columns": 1}
    assert result.extra["padding_fraction"] == pytest.approx(0.5)
    assert "host performs no arithmetic" in result.extra["timing_scope"]
    assert result.extra["crf"] == spmw_samsung._samsung_cmds_text(
        strict_gemv.compiled.cmds
    )


def test_reassociating_program_differs_from_source_order(
    samsung_sim_gate, paper_operands
):
    workload, M, _K = _paper_gemv()
    W, x = paper_operands
    compiled = allo.compile(
        workload, build_samsung_target(), samsung_cost, reduction_order="reassociate"
    )
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert result.extra["kernel"] == "GENERIC_REDUCE"
    assert result.extra["reduction_order"] == "reassociate"
    reference = source_order_fp16(W, x, np.zeros(M, np.float16))
    differ = int(np.count_nonzero(y.view(np.uint16) != reference.view(np.uint16)))
    assert differ > M // 2, differ
    exact = W.astype(np.float32) @ x.astype(np.float32)
    np.testing.assert_allclose(y.astype(np.float32), exact, rtol=1e-2, atol=1e-2)
    assert result.cycles < 26807


def test_ordered_program_carries_a_nonzero_initial_value(
    samsung_sim_gate, strict_gemv, paper_operands
):
    # The conductor loads the initial accumulator from bank row 0 (the first
    # CRF word); drive the compiled program's runtime with initial 0.5.
    W, x = paper_operands
    M, K = W.shape
    (program,) = strict_gemv.compiled.layout_ctx.programs
    root = pimsim_root()
    geometry = spmw_samsung.samsung_simulator_geometry(root)
    initial = np.full((M, 1), 0.5, dtype=np.float16)
    words = list(strict_gemv.compiled.cmds[program.start : program.end])
    run = spmw_samsung._samsung_run_temporal(
        program, words, root, geometry, W, x.reshape(1, K), initial
    )
    reference = source_order_fp16(W, x, np.full(M, 0.5, np.float16))
    np.testing.assert_array_equal(
        run["rows"][:, 0].view(np.uint16), reference.view(np.uint16)
    )
    assert run["cycles"] == 26807


def test_ordered_gemm_columns(samsung_sim_gate):
    compiled = allo.compile(small_gemm, build_samsung_target(), samsung_cost)
    rng = np.random.default_rng(4)
    A = (rng.standard_normal((GM, GK)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((GK, GN)) * 0.1).astype(np.float16)
    C = np.zeros((GM, GN), dtype=np.float16)
    result = compiled(A, B, C)
    assert result.extra["kernel"] == "GENERIC_TEMPORAL_REDUCE"
    reference = source_order_fp16(A, B, np.zeros((GM, GN), np.float16))
    np.testing.assert_array_equal(C.view(np.uint16), reference.view(np.uint16))
    assert result.extra["conductor"]["columns"] == GN


def test_carried_accumulator_through_the_public_api(samsung_sim_gate):
    """``y[row] += W[row, k] * x[k]`` through ``allo.compile``: the matcher
    records a carried accumulator, the strict default selects the ordered
    program, and the simulator result equals source-order FP16 accumulation
    starting from the caller's ``y``."""
    M, K, P, U = 4096, 256, 16, 8
    ROWS = M // (P * U)

    @_df_region()
    def carried_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
        @allo.work(mapping=[P, U], args=[W, x, y])
        def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
            p, u = allo.get_wid()
            row0 = (p * U + u) * ROWS
            for r in range(ROWS):
                for k in range(K):
                    local_y[row0 + r] += local_W[row0 + r, k] * local_x[k]

    compiled = allo.compile(carried_gemv, build_samsung_target(), samsung_cost)
    assert {m.extra["accumulator_init"] for m in compiled.trace.matches} == {"carried"}
    rng = np.random.default_rng(11)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    y0 = (rng.standard_normal(M) * 0.1).astype(np.float16)
    y = y0.copy()
    result = compiled(W, x, y)
    assert result.kind == "simulator"
    assert result.extra["reduction_order"] == "strict"
    reference = source_order_fp16(W, x, y0)
    np.testing.assert_array_equal(y.view(np.uint16), reference.view(np.uint16))
