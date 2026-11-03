# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""G6 semantic probes on the APU v1 board (EVIDENCE.md gaps 1-3, 11-13).

The functional twins of these probes are in ``tests/pim/test_g6_semantics.py``
(the workloads are imported from there).  Each test here runs the compiled
GVML program on the board (``backend=None``) and checks the readback against
an oracle computed in the test from fresh random inputs.  FP16 results are
compared with FP32 at the tolerance of ``test_vector_gemm_apu_v1`` (5e-2)
except where the probe is about bit-level order (last test), and integer
results exactly.  Every test is marked ``apu_v1_device`` and skips with
``BLOCKED-DEVICE`` when the board is unreachable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.ir.types import float16
from allo.pim.apu_v1_vector_program import apu_v1_plan_regroups
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target

pytestmark = pytest.mark.apu_v1_device

_NAME = "g6_semantics_for_apu_v1_board"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, Path(__file__).resolve().parents[1] / "test_g6_semantics.py"
)
G6 = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = G6
_SPEC.loader.exec_module(G6)

TOL = {"rtol": 5e-2, "atol": 5e-2}


def _device(workload, **kwargs):
    compiled = allo.compile(workload, build_apu_v1_target(), apu_v1_cost, **kwargs)
    assert compiled.realization is not None, compiled.realization_error
    return compiled


def _run(compiled, *arrays):
    import conftest

    with conftest.board_lock():
        result = compiled(*arrays)
    assert result.kind == "device"
    print(
        "G6_BOARD_RECORD "
        + json.dumps(
            {
                "workload": compiled.__name__,
                "plan": compiled.selected_plan.name,
                "cycles": result.cycles,
                "estimate": int(compiled.estimate().cycles),
            },
            sort_keys=True,
        )
    )
    return result


def _f32(a):
    return np.asarray(a).astype(np.float32)


def test_aliased_output_reads_pre_call_values_on_board(apu_v1_device_gate):
    rng = np.random.default_rng(701)
    L0 = (rng.standard_normal((G6.AQ, G6.AQ)) * 0.3).astype(np.float16)
    R0 = (rng.standard_normal((G6.AQ, G6.AQ)) * 0.3).astype(np.float16)
    compiled = _device(G6.apu_gemm_sq)
    assert compiled.realization.carries_output_ingress
    result_is_left = L0.copy()
    _run(compiled, result_is_left, R0, result_is_left)
    np.testing.assert_allclose(
        _f32(result_is_left), _f32(L0) + _f32(L0) @ _f32(R0), **TOL
    )


def test_uint16_contraction_wraps_on_board(apu_v1_device_gate):
    rng = np.random.default_rng(702)
    top = (1 << 16) - 1
    L = rng.integers(0, 1 << 16, size=(G6.UH_M, G6.UH_K), dtype=np.int64).astype(
        np.uint16
    )
    R = rng.integers(0, 1 << 16, size=(G6.UH_K, G6.UH_N), dtype=np.int64).astype(
        np.uint16
    )
    C0 = rng.integers(
        top - 16, top + 1, size=(G6.UH_M, G6.UH_N), dtype=np.int64
    ).astype(np.uint16)
    L[0, :] = top
    R[:, 0] = top
    C = C0.copy()
    _run(_device(G6.apu_u16_gemm), L, R, C)
    want = (C0.astype(object) + L.astype(object).dot(R.astype(object))) % (1 << 16)
    np.testing.assert_array_equal(C, want.astype(np.uint16))


def test_fp16_carried_output_accumulates_on_board(apu_v1_device_gate):
    rng = np.random.default_rng(703)
    L = (rng.standard_normal((G6.ZM, G6.ZK)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((G6.ZK, G6.ZN)) * 0.3).astype(np.float16)
    C0 = (rng.standard_normal((G6.ZM, G6.ZN)) * 2.0).astype(np.float16)
    compiled = _device(G6.apu_gemm_carried)
    assert compiled.realization.carries_output_ingress
    C = C0.copy()
    _run(compiled, L, R, C)
    np.testing.assert_allclose(_f32(C), _f32(C0) + _f32(L) @ _f32(R), **TOL)


def test_zero_filled_output_overwrites_a_sentinel_on_board(apu_v1_device_gate):
    """``C[r, c] = 0`` before ``C[r, c] += ...``: the device must start from
    zero, not from the caller's array (the regression of the output
    initializer fix)."""

    rng = np.random.default_rng(704)
    L = (rng.standard_normal((G6.ZM, G6.ZK)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((G6.ZK, G6.ZN)) * 0.3).astype(np.float16)
    compiled = _device(G6.apu_gemm_zero_filled)
    assert compiled.analysis.output_initializer == "zero"

    C = np.full((G6.ZM, G6.ZN), 7, np.float16)
    _run(compiled, L, np.zeros_like(R), C)
    np.testing.assert_array_equal(C, np.zeros_like(C))

    C = np.full((G6.ZM, G6.ZN), 7, np.float16)
    _run(compiled, L, R, C)
    np.testing.assert_allclose(_f32(C), _f32(L) @ _f32(R), **TOL)


def test_odd_gemm_shape_on_board(apu_v1_device_gate):
    def odd(L: float16[37, 19], R: float16[19, 23], C: float16[37, 23]):
        for r, c in allo.grid(37, 23):
            for d in allo.reduction(19):
                C[r, c] += L[r, d] * R[d, c]

    rng = np.random.default_rng(705)
    L = (rng.standard_normal((37, 19)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((19, 23)) * 0.3).astype(np.float16)
    C = np.full((37, 23), 0, np.float16)
    _run(_device(odd), L, R, C)
    np.testing.assert_allclose(_f32(C), _f32(L) @ _f32(R), **TOL)


def _strict_and_reassociative_runs(M, K, seed):
    def gemv(W: float16[M, K], x: float16[K], y: float16[M]):
        for i in allo.grid(M):
            for k in allo.reduction(K):
                y[i] += W[i, k] * x[k]

    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.2).astype(np.float16)
    x = (rng.standard_normal(K) * 0.2).astype(np.float16)
    exact = _f32(W) @ _f32(x)

    strict = _device(gemv)
    assert not apu_v1_plan_regroups(strict.selected_plan)
    y_strict = np.zeros(M, np.float16)
    _run(strict, W, x, y_strict)

    relaxed = _device(gemv, reduction_order="reassociate")
    plan = relaxed.selected_plan
    assert apu_v1_plan_regroups(plan)
    y_relaxed = np.zeros(M, np.float16)
    _run(relaxed, W, x, y_relaxed)
    return W, x, exact, y_strict, y_relaxed, plan


@pytest.mark.parametrize(
    "M, K, seed", [(512, 256, 706), (384, 200, 707)], ids=["full_group", "padded_group"]
)
def test_strict_and_reassociative_gemv_values_on_board(apu_v1_device_gate, M, K, seed):
    """Both plans execute on the device and agree with FP32 within FP16
    tolerance; the strict plan does not regroup, the relaxed plan does."""
    _W, _x, exact, y_strict, y_relaxed, _plan = _strict_and_reassociative_runs(
        M, K, seed
    )
    np.testing.assert_allclose(_f32(y_strict), exact, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(_f32(y_relaxed), exact, rtol=1e-2, atol=1e-2)


@pytest.mark.xfail(
    strict=True,
    reason="V1-16: the Gemini-I device's FP16 results differ at the bit level "
    "from the host NumPy models of both plans (source-order loop and halving "
    "tree); about half of the outputs differ by a few ulps while agreeing with "
    "FP32 within FP16 tolerance. The functional route's models are plan-order "
    "models, not bit-exact models of the device arithmetic. See docs/release/ISSUES.md.",
)
@pytest.mark.parametrize(
    "M, K, seed", [(512, 256, 706), (384, 200, 707)], ids=["full_group", "padded_group"]
)
def test_strict_and_reassociative_gemv_bit_equal_to_models_on_board(
    apu_v1_device_gate, M, K, seed
):
    """Bit-level comparison of the device results with the functional route's
    numerics models. Expected to fail until the models reproduce the device
    arithmetic (V1-16); the mismatch counts are printed for the record."""
    W, x, _exact, y_strict, y_relaxed, plan = _strict_and_reassociative_runs(M, K, seed)
    source = G6._fp16_source_order(np.zeros(M), (W[:, k] * x[k] for k in range(K)))
    tree = G6._halving_tree_fp16(W * x, plan.reduction_strategy.group_size)
    strict_mismatch = int(
        np.count_nonzero(y_strict.view(np.uint16) != source.view(np.uint16))
    )
    relaxed_mismatch = int(
        np.count_nonzero(y_relaxed.view(np.uint16) != tree.view(np.uint16))
    )
    print(
        f"APU_V1_BIT_MODEL strict_mismatch={strict_mismatch}/{M} relaxed_mismatch={relaxed_mismatch}/{M}"
    )
    np.testing.assert_array_equal(y_strict.view(np.uint16), source.view(np.uint16))
    np.testing.assert_array_equal(y_relaxed.view(np.uint16), tree.view(np.uint16))
