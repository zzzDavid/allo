# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper golden table: every evaluation number Tenon reports, checked live.

``PAPER_GOLDEN`` is the only place in the repository's Python that holds a
paper number. A row changes only when the cited paper table changes; a live
mismatch is a reproduction defect to report, never a reason to edit a row.

Artifact-backed rows read ``$TENON_ARTIFACTS`` (default
``/home/nz264/shared/tenon-artifacts``) and skip when it is absent.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import apu_v1_cost, samsung_cost
from allo.pim.targets import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
)


@dataclass(frozen=True)
class GoldenRow:
    backend: str
    case_id: str
    expected: int | float
    tolerance: float
    paper_source: str


_SAMSUNG = "paper/latex/tables/eval-samsung.tex"
_AIM = "paper/latex/tables/eval-skhynix.tex"
_UPMEM = "paper/latex/tables/eval-upmem.tex"
_APU_V1 = "paper/latex/tables/eval-apuv1.tex"
_LAYOUT = "paper/latex/tables/eval-layout-mechanisms.tex"

PAPER_GOLDEN = (
    GoldenRow("samsung_hbm_pim", "gemv_m4096_k256", 4_435, 0, f"{_SAMSUNG}: M=4096, K=256, B=1"),
    GoldenRow("samsung_hbm_pim", "gemv_m4096_k512", 8_041, 0, f"{_SAMSUNG}: M=4096, K=512, B=1"),
    GoldenRow("samsung_hbm_pim", "gemv_m8192_k256", 8_247, 0, f"{_SAMSUNG}: M=8192, K=256, B=1"),
    # Checked by samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py.
    GoldenRow("samsung_hbm_pim", "xor_probe_identity", 148, 0, f"{_LAYOUT}: identity bank layout"),
    GoldenRow("samsung_hbm_pim", "xor_probe_synthesized", 116, 0, f"{_LAYOUT}: synthesized XOR layout"),
    GoldenRow("aim", "norm_residual_l128", 11_750, 0, f"{_AIM}: RMSNorm + Residual, L=128"),
    GoldenRow("aim", "qkvo_rope_l128", 129_605, 0, f"{_AIM}: QKVO + RoPE, L=128"),
    GoldenRow("aim", "attention_l128", 11_010, 0, f"{_AIM}: Attention QK + SV, L=128"),
    GoldenRow("aim", "attention_l512", 18_126, 0, f"{_AIM}: Attention QK + SV, L=512"),
    GoldenRow("aim", "attention_l4096", 95_090, 0, f"{_AIM}: Attention QK + SV, L=4096"),
    GoldenRow("aim", "softmax_pim_l128", 6_307, 0, f"{_AIM}: Softmax PIM Portion, L=128"),
    GoldenRow("aim", "softmax_pim_l512", 24_835, 0, f"{_AIM}: Softmax PIM Portion, L=512"),
    GoldenRow("aim", "softmax_pim_l4096", 198_416, 0, f"{_AIM}: Softmax PIM Portion, L=4096"),
    GoldenRow("aim", "ffn_fc_l128", 250_433, 0, f"{_AIM}: FFN Projections, L=128"),
    GoldenRow("aim", "ffn_activation_l128", 11_775, 0, f"{_AIM}: FFN SiLU + Gating, L=128"),
    GoldenRow("aim", "ffn_complete_l128", 262_269, 0, f"{_AIM}: Complete Mapped FFN, L=128"),
    GoldenRow("aim", "full_block_l128", 421_159, 0, f"{_AIM}: Mapped PIM Block, L=128"),
    GoldenRow("aim", "full_block_l512", 446_803, 0, f"{_AIM}: Mapped PIM Block, L=512"),
    GoldenRow("aim", "full_block_l4096", 697_316, 0, f"{_AIM}: Mapped PIM Block, L=4096"),
    # UPMEM rows hold uPIMulator logic cycles; the paper table prints
    # milliseconds = cycles / 350,000 (350 MHz logic clock).
    GoldenRow("upmem", "va", 114_534, 0, f"{_UPMEM}: VA, N=12288"),
    GoldenRow("upmem", "red", 83_117, 0, f"{_UPMEM}: RED, N=12288"),
    GoldenRow("upmem", "mtv", 359_135, 0, f"{_UPMEM}: MTV/MV, M=96, K=128"),
    GoldenRow("upmem", "gemv", 359_026, 0, f"{_UPMEM}: Scaled GEMV, M=96, K=128"),
    GoldenRow("upmem", "geva", 126_649, 0, f"{_UPMEM}: GEVA, N=12288"),
    GoldenRow("upmem", "ttv", 214_361, 0, f"{_UPMEM}: TTV, M=12, N=16, K=32"),
    GoldenRow("upmem", "mmtv", 211_371, 0, f"{_UPMEM}: MMTV, M=12, N=16, K=32"),
    GoldenRow("upmem", "hist", 183_782, 0, f"{_UPMEM}: Histogram, N=12288"),
    GoldenRow("upmem", "sel", 89_271, 0, f"{_UPMEM}: Selection, N=12288"),
    GoldenRow("upmem", "kmeans", 221_196, 0, f"{_UPMEM}: K-means, P=120, D=8, K=4"),
    GoldenRow("upmem", "linear_reg", 212_718, 0, f"{_UPMEM}: Linear reg., S=120, F=8"),
    GoldenRow("upmem", "logistic_reg", 208_442, 0, f"{_UPMEM}: Logistic reg., S=120, F=8"),
    GoldenRow("upmem", "1mm", 3_135_600, 0, f"{_UPMEM}: 1mm, M=12, K=64, N=128"),
    GoldenRow("upmem", "2mm", 6_271_190, 0, f"{_UPMEM}: 2mm, M=12, K=64, N=128"),
    GoldenRow("upmem", "3mm", 9_406_780, 0, f"{_UPMEM}: 3mm, M=12, K=64, N=128"),
    GoldenRow("upmem", "conv", 2_956_792, 0, f"{_UPMEM}: Conv im2col GEMM, P=192"),
    # Gemini-I rows hold the Tenon median device `crun`. Only the histogram row
    # runs on the board; the other four assert compile and GVML realization.
    GoldenRow("apu_v1", "matrix_multiply", 8_764_900, 0.02, f"{_APU_V1}: MatMul 256x1024x1024"),
    GoldenRow("apu_v1", "kmeans", 624_677, 0.02, f"{_APU_V1}: K-Means P32768,D24,K10"),
    GoldenRow("apu_v1", "histogram", 166_783, 0.02, f"{_APU_V1}: Histogram N32768,B256"),
    GoldenRow("apu_v1", "linear_regression", 94_618.5, 0.02, f"{_APU_V1}: Linear Regression 8xN4096"),
    GoldenRow("apu_v1", "word_count", 171_988, 0.02, f"{_APU_V1}: Word Count Q128,N32768"),
    # Median of 3 board runs; checked by apu_v1/vector_gemm/test_vector_gemm_apu_v1.py.
    GoldenRow("apu_v1", "vector_gemm_temporal_dma_coalescing", 93_449_834, 0.02, f"{_LAYOUT}: coalesced DMA"),
    # Re-captured 2026-10-04 by user ruling (paper table still prints 7,349,987,
    # measured 2026-07-07): median of [7145109, 7145397, 7144077] on tenon@4a5104e
    # plus the uncommitted task-009 tree, TENON_APU_V1_BUILD_MODE=release (default),
    # GSI Gemini-I at PCI 0000:41:00.0 on zhang-capra-xcel, SDK 13.7.1. A clean
    # tenon@4a5104e gives the same number.
    GoldenRow("apu_v1", "vector_gemm_temporal_dma_coalescing_broadcast_friendly", 7_145_109, 0.02, f"{_LAYOUT}: broadcast-friendly (re-captured 2026-10-04)"),
)

# Rows whose live check lives in a leaf test next to its workload.
_LEAF_CHECKED = {
    "xor_probe_identity": "samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py",
    "xor_probe_synthesized": "samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py",
    "vector_gemm_temporal_dma_coalescing": "apu_v1/vector_gemm/test_vector_gemm_apu_v1.py",
    "vector_gemm_temporal_dma_coalescing_broadcast_friendly": "apu_v1/vector_gemm/test_vector_gemm_apu_v1.py",
}

_AIM_DEFAULT_CASES = frozenset(
    {"norm_residual_l128", "attention_l128", "softmax_pim_l128"}
)
_APU_V1_BOARD_CASES = frozenset({"histogram"})
_SAMSUNG_COMPILE_BUDGET_S = 120.0
_AIM_COMPILE_BUDGET_S = 120.0
_OTHER_COMPILE_BUDGET_S = 300.0

# Compile facts the paper producers select, not paper numbers.
_APU_V1_MATRIX_PLAN = "temporal_dma_coalescing_broadcast_friendly_acc8"
_APU_V1_NATIVE_ROUTES = {
    "kmeans": "gvml_squared_l2_argmin_center_streaming",
    "histogram": "gvml_dense_histogram_pair_count",
    "linear_regression": "gvml_bivariate_moments_fused_group_reduce",
    "word_count": "gvml_record_frequency_resident_chunks_pair_count",
}

_ALLO_ROOT = Path(__file__).resolve().parents[2]


def _rows(backend):
    return [
        row
        for row in PAPER_GOLDEN
        if row.backend == backend and row.case_id not in _LEAF_CHECKED
    ]


def golden_row(case_id: str) -> GoldenRow:
    (row,) = [row for row in PAPER_GOLDEN if row.case_id == case_id]
    return row


def test_leaf_checked_rows_name_their_checking_test():
    here = Path(__file__).resolve().parent
    for case_id, leaf in _LEAF_CHECKED.items():
        golden_row(case_id)
        assert case_id in (here / leaf).read_text(encoding="utf-8"), (case_id, leaf)


def _artifacts_root() -> Path:
    return Path(os.environ.get("TENON_ARTIFACTS", "/home/nz264/shared/tenon-artifacts"))


def _require_artifacts(*parts: str) -> Path:
    path = _artifacts_root().joinpath(*parts)
    if not path.exists():
        pytest.skip(f"BLOCKED-ENV(tenon artifacts not found at {path})")
    return path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _within(observed, row: GoldenRow) -> bool:
    return abs(observed - row.expected) <= row.tolerance * row.expected


def _ids(rows):
    return [f"{row.backend}-{row.case_id}" for row in rows]


@pytest.mark.parametrize("row", _rows("samsung_hbm_pim"), ids=_ids(_rows("samsung_hbm_pim")))
def test_samsung_gemv_paper_cycles(row, samsung_sim_gate):
    workloads = _load_module(
        Path(__file__).resolve().parent / "samsung_hbm_pim" / "gemv" / "workload.py",
        "paper_golden_samsung_gemv_workload",
    )
    workload, M, K = workloads.CASES[row.case_id]

    started = time.perf_counter()
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    assert time.perf_counter() - started < _SAMSUNG_COMPILE_BUDGET_S

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert _within(result.cycles, row), (row, result.cycles)
    np.testing.assert_allclose(
        y.astype(np.float32),
        W.astype(np.float32) @ x.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "row",
    [
        row if row.case_id in _AIM_DEFAULT_CASES
        else pytest.param(row, marks=pytest.mark.paper_full)
        for row in _rows("aim")
    ],
    ids=_ids(_rows("aim")),
)
def test_aim_cent_paper_cycles(row, aim_sim_gate):
    # ramulator2 is deterministic; the campaign ran each trace twice with
    # identical results, so the row is asserted exactly.
    from benchmarks.cent_aim.workloads import build_case

    started = time.perf_counter()
    compiled = allo.compile(build_case(row.case_id), build_aim_target())
    assert time.perf_counter() - started < _AIM_COMPILE_BUDGET_S

    result = compiled()

    assert _within(result.cycles, row), (row, result.cycles)


@pytest.mark.parametrize("row", _rows("upmem"), ids=_ids(_rows("upmem")))
def test_upmem_program_matches_paper_measured_source(row):
    """uPIMulator is deterministic in (program, fixture), and the canonical
    fixture is fixed, so emitting the byte-identical DPU program that the paper
    measured implies the row's `expected` logic cycles."""
    inputs = _require_artifacts("upmem", "inputs")
    measured = _require_artifacts(
        "upmem", "tenon", "programs", row.case_id, "source", "task.c"
    )
    campaign = _load_module(
        _ALLO_ROOT / "scripts" / "prepare_upmem_tenon_campaign.py",
        "paper_golden_prepare_upmem_tenon_campaign",
    )
    assert row.case_id in campaign.WORKLOADS

    started = time.perf_counter()
    plan, *_ = campaign.build_case(campaign.CanonicalInputs(inputs), row.case_id)
    source = plan.device_source()
    assert time.perf_counter() - started < _OTHER_COMPILE_BUDGET_S

    assert source.encode("utf-8") == measured.read_bytes()


def _apu_v1_compile(case_id: str, *, backend):
    workload = _load_module(
        _require_artifacts("apuv1", "tenon", case_id, "workload.py"),
        f"paper_golden_apu_v1_{case_id}",
    )
    layout = _APU_V1_MATRIX_PLAN if case_id == "matrix_multiply" else None
    started = time.perf_counter()
    compiled = allo.compile(
        workload.build(),
        build_apu_v1_target(),
        apu_v1_cost,
        backend=backend,
        layout=layout,
    )
    assert time.perf_counter() - started < _OTHER_COMPILE_BUDGET_S
    return compiled


@pytest.mark.parametrize("row", _rows("apu_v1"), ids=_ids(_rows("apu_v1")))
def test_apu_v1_paper_kernel_compiles_to_gvml(row):
    compiled = _apu_v1_compile(row.case_id, backend="virtual")
    if row.case_id == "matrix_multiply":
        assert compiled.selected_plan.name == _APU_V1_MATRIX_PLAN
        assert compiled.realization is not None, compiled.realization_error
        assert "gvml_" in compiled.realization.device_source()
    else:
        lowering = compiled.compiled.native_vector_lowering
        assert lowering is not None
        assert lowering.route == _APU_V1_NATIVE_ROUTES[row.case_id]
        assert "gvml_" in compiled.compiled.device_source


@pytest.mark.apu_v1_device
@pytest.mark.parametrize(
    "row",
    [row for row in _rows("apu_v1") if row.case_id in _APU_V1_BOARD_CASES],
    ids=_ids([row for row in _rows("apu_v1") if row.case_id in _APU_V1_BOARD_CASES]),
)
def test_apu_v1_board_median_cycles(row, apu_v1_device_gate):
    import conftest

    case_inputs = _require_artifacts("apuv1", "inputs", row.case_id)
    compiled = _apu_v1_compile(row.case_id, backend=None)
    (argument,) = compiled.arguments
    values = np.fromfile(
        case_inputs / "values.uint16.bin", dtype=argument.numpy_dtype
    ).reshape(argument.shape)
    oracle = np.fromfile(case_inputs / "oracle.uint16.bin", dtype=np.uint16)

    cycles = []
    with conftest.board_lock():
        for _ in range(3):
            result = compiled(values)
            np.testing.assert_array_equal(
                result.extra["outputs"]["counts"].reshape(-1), oracle
            )
            cycles.append(result.cycles)

    median = statistics.median(cycles)
    assert _within(median, row), (row, cycles)
