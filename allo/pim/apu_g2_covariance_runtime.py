# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-orchestrated uint16 covariance for Gemini-II.

This is the first statistics-family APUg2 gate.  It composes only
hardware-validated VL64 primitives:

* dot-tile: column sums;
* persistent GEMM: centered Gram products;
* fill: scalar divisors;
* div: mean and final scaling;
* sub: mean centering.

The orchestration is still host-side; all tensor arithmetic is executed on the
card and checked against an explicit uint16 NumPy reference.
"""

from __future__ import annotations

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_div_runtime import run_apu_g2_u16_div
from .apu_g2_fill_runtime import run_apu_g2_u16_fill
from .apu_g2_runtime import _SHAPE, _validate_operand, _validate_repetitions
from .apu_g2_sub_runtime import run_apu_g2_u16_sub
from .apu_g2_u16_gemm_runtime import (
    APUG2_U16_GEMM_BMAX,
    run_apu_g2_u16_gemm,
)


def _validate_data(data) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array")
    if data.dtype != np.dtype(np.uint16):
        raise TypeError(f"data must have dtype uint16, got {data.dtype}")
    if data.ndim != 2 or data.shape[0] <= 1 or data.shape[1] <= 0:
        raise ValueError("data must have shape (N, M) with N > 1 and M > 0")
    if not data.flags.c_contiguous:
        raise ValueError("data must be C-contiguous")
    return data


def _pack_linear(values, *, fill=0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint16)
    if values.size > np.prod(_SHAPE):
        raise ValueError("linear values do not fit one APUg2 vector pack")
    packed = np.full(_SHAPE, fill, dtype=np.uint16)
    packed.reshape(-1)[: values.size] = values.reshape(-1)
    return packed


def _unpack_linear(packed, size: int) -> np.ndarray:
    packed = _validate_operand("packed", packed)
    return packed.reshape(-1)[:size].copy()


def _run_binary_chunks(lhs, rhs, runner, *, repetitions, rhs_fill=0):
    lhs = np.asarray(lhs, dtype=np.uint16).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.uint16).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError("chunked binary operands must have the same shape")
    capacity = int(np.prod(_SHAPE))
    output = np.empty_like(lhs)
    runs = []
    for begin in range(0, lhs.size, capacity):
        end = min(begin + capacity, lhs.size)
        run = runner(
            _pack_linear(lhs[begin:end]),
            _pack_linear(rhs[begin:end], fill=rhs_fill),
            repetitions=repetitions,
        )
        output[begin:end] = _unpack_linear(run.extra["outputs"]["out"], end - begin)
        runs.append(run)
    return output, runs


def covariance_u16_reference(data: np.ndarray) -> dict[str, np.ndarray]:
    """Explicit uint16 covariance reference used by the hardware gate."""

    data = _validate_data(data)
    n, _m = data.shape
    mean_sum = (
        np.sum(data.astype(np.uint64), axis=0, dtype=np.uint64) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    mean = (mean_sum.astype(np.uint32) // np.uint32(n)).astype(np.uint16)
    centered = (data.astype(np.uint32) - mean.astype(np.uint32)[None, :]).astype(
        np.uint16
    )
    gram_raw = (centered.T.astype(np.uint64) @ centered.astype(np.uint64)) & np.uint64(
        0xFFFF
    )
    covariance = (gram_raw // np.uint64(n - 1)).astype(np.uint16)
    return {
        "mean_sum": mean_sum,
        "mean": mean,
        "centered": centered,
        "gram_raw": gram_raw.astype(np.uint16),
        "covariance": covariance,
    }


def _check_equal(name: str, observed: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(observed, expected):
        mismatch = np.argwhere(observed != expected)[0]
        index = tuple(int(item) for item in mismatch)
        raise RuntimeError(
            f"APUg2 covariance {name} differs from NumPy at {index}: "
            f"hardware={int(observed[index])} numpy={int(expected[index])}"
        )


def run_apu_g2_u16_covariance(data, *, repetitions=2) -> RunResult:
    """Run complete uint16 covariance on APUg2 through validated primitives."""

    data = _validate_data(data)
    repetitions = _validate_repetitions(repetitions)
    n, m = data.shape
    reference = covariance_u16_reference(data)

    mean_sum_run = run_apu_g2_u16_gemm(
        np.ascontiguousarray(data.T),
        np.ones((n, 1), dtype=np.uint16),
        np.zeros((m, 1), dtype=np.uint16),
        beta=0,
        batch_columns=1,
    )
    mean_sum = np.asarray(mean_sum_run.extra["outputs"]["out"][:, 0])
    _check_equal("mean_sum", mean_sum, reference["mean_sum"])

    fill_n_run = run_apu_g2_u16_fill(n, repetitions=repetitions)
    divisor_n = fill_n_run.extra["outputs"]["out"]
    mean_pack = _pack_linear(mean_sum)
    mean_run = run_apu_g2_u16_div(mean_pack, divisor_n, repetitions=repetitions)
    mean = _unpack_linear(mean_run.extra["outputs"]["out"], m)
    _check_equal("mean", mean, reference["mean"])

    centered_flat, centered_runs = _run_binary_chunks(
        data.reshape(-1),
        np.tile(mean, n),
        run_apu_g2_u16_sub,
        repetitions=repetitions,
    )
    centered = centered_flat.reshape(data.shape)
    _check_equal("centered", centered, reference["centered"])

    gram_run = run_apu_g2_u16_gemm(
        np.ascontiguousarray(centered.T),
        centered,
        np.zeros((m, m), dtype=np.uint16),
        beta=0,
    )
    gram_raw = gram_run.extra["outputs"]["out"]
    _check_equal("gram_raw", gram_raw, reference["gram_raw"])

    fill_scale_run = run_apu_g2_u16_fill(n - 1, repetitions=repetitions)
    divisor_scale = fill_scale_run.extra["outputs"]["out"]
    covariance_flat, covariance_runs = _run_binary_chunks(
        gram_raw.reshape(-1),
        np.full(m * m, n - 1, dtype=np.uint16),
        run_apu_g2_u16_div,
        repetitions=repetitions,
        rhs_fill=1,
    )
    covariance = covariance_flat.reshape(m, m)
    _check_equal("covariance", covariance, reference["covariance"])

    primitive_runs = (
        mean_sum_run,
        fill_n_run,
        mean_run,
        *centered_runs,
        gram_run,
        fill_scale_run,
        *covariance_runs,
    )
    total_ticks = sum(
        item.extra.get("total_ticks", item.cycles) for item in primitive_runs
    )
    cycles = sum(item.cycles for item in primitive_runs)
    hardware_tasks = sum(item.extra.get("hardware_tasks", 1) for item in primitive_runs)
    stdout_parts = [
        "APUg2 covariance primitive composition",
        f"shape=(N={n}, M={m})",
        f"gram_tasks={gram_run.extra['hardware_tasks']}",
        f"hardware_tasks={hardware_tasks}",
    ]
    return RunResult(
        cycles,
        "\n".join(stdout_parts),
        "apu_v2",
        extra={
            "outputs": {
                "mean_sum": mean_sum,
                "mean": mean,
                "centered": centered,
                "gram_raw": gram_raw,
                "covariance": covariance,
            },
            "reference": reference,
            "hardware_tasks": hardware_tasks,
            "primitive_programs": {
                "dot_tile": 0,
                "column_batched_gemm": 2,
                "fill": 2,
                "div": 1 + len(covariance_runs),
                "sub": len(centered_runs),
            },
            "total_ticks": total_ticks,
            "per_task_cycles": [item.cycles for item in primitive_runs],
            "repetitions": repetitions,
            "tiling": {
                "gram_output_extent": m * m,
                "mean_hardware_tasks": mean_sum_run.extra["hardware_tasks"],
                "gram_batch_columns": APUG2_U16_GEMM_BMAX,
                "gram_reduction_tile": 128,
                "gram_hardware_tasks": gram_run.extra["hardware_tasks"],
                "reduction_extent": n,
            },
            "vectorization_certificate": {
                "direct_vl64": True,
                "simulator_fallback": False,
                "scalar_tensor_updates": 0,
                "scalar_control_ops": 0,
                "fully_vectorized": True,
            },
            "projects": [
                mean_sum_run.extra["project"],
                fill_n_run.extra["project"],
                mean_run.extra["project"],
                *[item.extra["project"] for item in centered_runs],
                gram_run.extra["project"],
                fill_scale_run.extra["project"],
                *[item.extra["project"] for item in covariance_runs],
            ],
        },
    )


__all__ = ["covariance_u16_reference", "run_apu_g2_u16_covariance"]
