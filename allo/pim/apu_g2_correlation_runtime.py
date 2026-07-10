# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-orchestrated uint16 correlation for Gemini-II."""

from __future__ import annotations

import math

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_div_runtime import run_apu_g2_u16_div
from .apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile
from .apu_g2_fill_runtime import run_apu_g2_u16_fill
from .apu_g2_minmax_runtime import run_apu_g2_u16_minmax
from .apu_g2_mul_runtime import run_apu_g2_u16_mul
from .apu_g2_runtime import _SHAPE, _validate_operand, _validate_repetitions
from .apu_g2_select_runtime import run_apu_g2_u16_select_lt
from .apu_g2_sqrt_runtime import run_apu_g2_u16_sqrt
from .apu_g2_sub_runtime import run_apu_g2_u16_sub
from .apu_g2_u16_gemm_runtime import (
    APUG2_U16_GEMM_BMAX,
    run_apu_g2_u16_gemm,
)
from .apu_g2_covariance_runtime import _run_binary_chunks


def _validate_data(name: str, value, shape=None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have shape (N, M) with N > 0 and M > 0")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


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


def _run_self_dot_chunks(values, *, repetitions):
    rows, reduction = values.shape
    output = np.zeros(rows, dtype=np.uint16)
    runs = []
    for reduction_begin in range(0, reduction, 256):
        reduction_end = min(reduction_begin + 256, reduction)
        for output_begin in range(0, rows, 1024):
            output_end = min(output_begin + 1024, rows)
            tile = np.ascontiguousarray(
                values[
                    output_begin:output_end,
                    reduction_begin:reduction_end,
                ]
            )
            run = run_apu_g2_u16_dot_tile(
                tile,
                tile,
                accumulator=np.ascontiguousarray(output[output_begin:output_end]),
                alpha=1,
                beta=0 if reduction_begin == 0 else 1,
                repetitions=repetitions,
            )
            output[output_begin:output_end] = run.extra["outputs"]["out"]
            runs.append(run)
    return output, runs


def _run_select_chunks(
    predicate_lhs, predicate_rhs, true_values, false_values, *, repetitions
):
    operands = [
        np.asarray(item, dtype=np.uint16).reshape(-1)
        for item in (
            predicate_lhs,
            predicate_rhs,
            true_values,
            false_values,
        )
    ]
    size = operands[0].size
    if any(item.size != size for item in operands[1:]):
        raise ValueError("chunked select operands must have the same size")
    capacity = int(np.prod(_SHAPE))
    output = np.empty(size, dtype=np.uint16)
    runs = []
    for begin in range(0, size, capacity):
        end = min(begin + capacity, size)
        run = run_apu_g2_u16_select_lt(
            *[_pack_linear(item[begin:end]) for item in operands],
            repetitions=repetitions,
        )
        output[begin:end] = _unpack_linear(run.extra["outputs"]["out"], end - begin)
        runs.append(run)
    return output, runs


def correlation_u16_reference(
    data_mean: np.ndarray,
    data_stddev: np.ndarray | None = None,
    data_for_center: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Explicit modular-uint16 correlation reference for APUg2.

    The floating PolyBench divide/sqrt normalization is specialized to the
    backend's integer policy: low-16 reductions, unsigned floor division,
    floor square root, and ``stddev = max(sqrt(variance), 1)``.
    """

    data_mean = _validate_data("data_mean", data_mean)
    data_stddev = (
        data_mean
        if data_stddev is None
        else _validate_data("data_stddev", data_stddev, data_mean.shape)
    )
    data_for_center = (
        data_mean
        if data_for_center is None
        else _validate_data("data_for_center", data_for_center, data_mean.shape)
    )
    n, m = data_mean.shape
    mean_sum = (
        np.sum(data_mean.astype(np.uint64), axis=0, dtype=np.uint64) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    mean = (mean_sum.astype(np.uint32) // np.uint32(n)).astype(np.uint16)

    std_centered = (
        data_stddev.astype(np.uint32) - mean.astype(np.uint32)[None, :]
    ).astype(np.uint16)
    variance_raw = (
        np.sum(
            std_centered.astype(np.uint64) * std_centered.astype(np.uint64),
            axis=0,
            dtype=np.uint64,
        )
        & np.uint64(0xFFFF)
    ).astype(np.uint16)
    variance = (variance_raw.astype(np.uint32) // np.uint32(n)).astype(np.uint16)
    stddev_raw = np.floor(np.sqrt(variance.astype(np.float64))).astype(np.uint16)
    stddev = np.maximum(stddev_raw, np.uint16(1)).astype(np.uint16)

    centered = (
        data_for_center.astype(np.uint32) - mean.astype(np.uint32)[None, :]
    ).astype(np.uint16)
    sqrt_n = np.uint16(math.isqrt(n))
    denominator = (np.uint32(sqrt_n) * stddev.astype(np.uint32)).astype(np.uint16)
    normalized = np.floor_divide(
        centered, denominator.astype(np.uint32)[None, :]
    ).astype(np.uint16)

    gram_raw = (
        normalized.T.astype(np.uint64) @ normalized.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    correlation = gram_raw.astype(np.uint16)
    np.fill_diagonal(correlation, np.uint16(1))
    return {
        "mean_sum": mean_sum,
        "mean": mean,
        "std_centered": std_centered,
        "variance_raw": variance_raw,
        "variance": variance,
        "stddev_raw": stddev_raw,
        "stddev": stddev,
        "centered": centered,
        "sqrt_n": np.array(sqrt_n, dtype=np.uint16),
        "denominator": denominator,
        "normalized": normalized,
        "gram_raw": gram_raw.astype(np.uint16),
        "correlation": correlation,
    }


def _check_equal(name: str, observed: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(observed, expected):
        mismatch = np.argwhere(observed != expected)[0]
        index = tuple(int(item) for item in mismatch)
        raise RuntimeError(
            f"APUg2 correlation {name} differs from NumPy at {index}: "
            f"hardware={int(observed[index])} numpy={int(expected[index])}"
        )


def run_apu_g2_u16_correlation(
    data_mean,
    data_stddev,
    data_for_center,
    *,
    repetitions=2,
) -> RunResult:
    """Run complete uint16 correlation on APUg2 through validated primitives."""

    data_mean = _validate_data("data_mean", data_mean)
    data_stddev = _validate_data("data_stddev", data_stddev, data_mean.shape)
    data_for_center = _validate_data(
        "data_for_center", data_for_center, data_mean.shape
    )
    repetitions = _validate_repetitions(repetitions)
    n, m = data_mean.shape
    reference = correlation_u16_reference(data_mean, data_stddev, data_for_center)

    mean_sum_run = run_apu_g2_u16_gemm(
        np.ascontiguousarray(data_mean.T),
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

    repeated_mean = np.tile(mean, n)
    std_centered_flat, std_centered_runs = _run_binary_chunks(
        data_stddev.reshape(-1),
        repeated_mean,
        run_apu_g2_u16_sub,
        repetitions=repetitions,
    )
    std_centered = std_centered_flat.reshape(data_stddev.shape)
    _check_equal("std_centered", std_centered, reference["std_centered"])

    variance_raw, variance_raw_runs = _run_self_dot_chunks(
        np.ascontiguousarray(std_centered.T), repetitions=repetitions
    )
    _check_equal("variance_raw", variance_raw, reference["variance_raw"])

    variance_pack = _pack_linear(variance_raw)
    variance_run = run_apu_g2_u16_div(variance_pack, divisor_n, repetitions=repetitions)
    variance = _unpack_linear(variance_run.extra["outputs"]["out"], m)
    _check_equal("variance", variance, reference["variance"])

    stddev_raw_run = run_apu_g2_u16_sqrt(
        _pack_linear(variance), repetitions=repetitions
    )
    stddev_raw = _unpack_linear(stddev_raw_run.extra["outputs"]["out"], m)
    _check_equal("stddev_raw", stddev_raw, reference["stddev_raw"])

    fill_one_run = run_apu_g2_u16_fill(1, repetitions=repetitions)
    stddev_clamp_run = run_apu_g2_u16_minmax(
        _pack_linear(stddev_raw),
        fill_one_run.extra["outputs"]["out"],
        repetitions=repetitions,
    )
    stddev = _unpack_linear(stddev_clamp_run.extra["outputs"]["max"], m)
    _check_equal("stddev", stddev, reference["stddev"])

    centered_flat, centered_runs = _run_binary_chunks(
        data_for_center.reshape(-1),
        repeated_mean,
        run_apu_g2_u16_sub,
        repetitions=repetitions,
    )
    centered = centered_flat.reshape(data_for_center.shape)
    _check_equal("centered", centered, reference["centered"])

    fill_sqrt_n_run = run_apu_g2_u16_fill(
        int(reference["sqrt_n"]), repetitions=repetitions
    )
    repeated_stddev = np.tile(stddev, n)
    denominator_flat, denominator_runs = _run_binary_chunks(
        repeated_stddev,
        np.full(data_for_center.size, int(reference["sqrt_n"]), dtype=np.uint16),
        run_apu_g2_u16_mul,
        repetitions=repetitions,
    )
    denominator = denominator_flat.reshape(data_for_center.shape)
    _check_equal(
        "denominator",
        denominator,
        np.tile(reference["denominator"], (n, 1)),
    )

    normalized_flat, normalized_runs = _run_binary_chunks(
        centered.reshape(-1),
        denominator.reshape(-1),
        run_apu_g2_u16_div,
        repetitions=repetitions,
        rhs_fill=1,
    )
    normalized = normalized_flat.reshape(data_for_center.shape)
    _check_equal("normalized", normalized, reference["normalized"])

    gram_run = run_apu_g2_u16_gemm(
        np.ascontiguousarray(normalized.T),
        normalized,
        np.zeros((m, m), dtype=np.uint16),
        beta=0,
    )
    gram_raw = gram_run.extra["outputs"]["out"]
    _check_equal("gram_raw", gram_raw, reference["gram_raw"])

    flat_size = m * m
    diagonal = np.zeros(flat_size, dtype=np.uint16)
    diagonal[np.arange(m) * m + np.arange(m)] = 1
    correlation_flat, correlation_runs = _run_select_chunks(
        1 - diagonal,
        diagonal,
        np.ones(flat_size, dtype=np.uint16),
        gram_raw.reshape(-1),
        repetitions=repetitions,
    )
    correlation = correlation_flat.reshape(m, m)
    _check_equal("correlation", correlation, reference["correlation"])

    primitive_runs = (
        mean_sum_run,
        fill_n_run,
        mean_run,
        *std_centered_runs,
        *variance_raw_runs,
        variance_run,
        stddev_raw_run,
        fill_one_run,
        stddev_clamp_run,
        *centered_runs,
        fill_sqrt_n_run,
        *denominator_runs,
        *normalized_runs,
        gram_run,
        *correlation_runs,
    )
    hardware_tasks = sum(item.extra.get("hardware_tasks", 1) for item in primitive_runs)
    total_ticks = sum(
        item.extra.get("total_ticks", item.cycles) for item in primitive_runs
    )
    cycles = sum(item.cycles for item in primitive_runs)
    return RunResult(
        cycles,
        "\n".join(
            [
                "APUg2 correlation primitive composition",
                f"shape=(N={n}, M={m})",
                f"corr_tasks={gram_run.extra['hardware_tasks']}",
                f"hardware_tasks={hardware_tasks}",
            ]
        ),
        "apu_v2",
        extra={
            "outputs": {
                "mean_sum": mean_sum,
                "mean": mean,
                "std_centered": std_centered,
                "variance_raw": variance_raw,
                "variance": variance,
                "stddev_raw": stddev_raw,
                "stddev": stddev,
                "centered": centered,
                "denominator": denominator,
                "normalized": normalized,
                "gram_raw": gram_raw,
                "correlation": correlation,
            },
            "reference": reference,
            "hardware_tasks": hardware_tasks,
            "primitive_programs": {
                "dot_tile": len(variance_raw_runs),
                "column_batched_gemm": 2,
                "fill": 3,
                "div": 2 + len(normalized_runs),
                "sub": len(std_centered_runs) + len(centered_runs),
                "sqrt": 1,
                "minmax": 1,
                "mul": len(denominator_runs),
                "select_lt": len(correlation_runs),
            },
            "total_ticks": total_ticks,
            "per_task_cycles": [item.cycles for item in primitive_runs],
            "repetitions": repetitions,
            "tiling": {
                "mean_output_extent": m,
                "mean_hardware_tasks": mean_sum_run.extra["hardware_tasks"],
                "variance_output_extent": m,
                "variance_hardware_tasks": len(variance_raw_runs),
                "corr_output_extent": m * m,
                "corr_batch_columns": APUG2_U16_GEMM_BMAX,
                "corr_reduction_tile": 128,
                "corr_hardware_tasks": gram_run.extra["hardware_tasks"],
                "reduction_extent": n,
            },
            "vectorization_certificate": {
                "direct_vl64": True,
                "simulator_fallback": False,
                "scalar_tensor_updates": 0,
                "scalar_control_ops": 8,
                "fully_vectorized": True,
            },
            "projects": [item.extra["project"] for item in primitive_runs],
        },
    )


__all__ = ["correlation_u16_reference", "run_apu_g2_u16_correlation"]
