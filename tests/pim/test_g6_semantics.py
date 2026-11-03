# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""G6 semantic probes (docs/release/EVIDENCE.md section 3, gaps 1-4 and 11-13).

Every probe draws its inputs here from a fresh ``np.random.default_rng(seed)``
(or writes extreme values by hand) and checks executed results against an
oracle written here: NumPy, Python integers reduced modulo the container width,
or an explicit source-order FP16 loop.  Nothing is compared as symbol text.
Simulator probes skip with ``BLOCKED-SIM``; the Gemini-II probes in this file
run the full host path against the software card of
``tests/perf/test_apu_g2_source_route.py`` (``_EmulatedCard``).  The card and
board versions are ``tests/pim/test_apu_g2_hardware.py`` (``apu_g2_card``) and
``tests/pim/apu_v1/test_g6_board_semantics.py`` (``apu_v1_device``).

What the implementation does, as asserted below:

* Aliases (i).  A source-level alias between a contraction operand and its own
  result at another element (``y[i] = sum_k W[i, k] * y[k]``, source order
  reads rows already overwritten) is REJECTED on every backend: by the matcher
  coverage (``spmw_match_engine._FunctionCoverage._check_result_aliases``,
  "aliases its result") for Samsung, UPMEM, AiM and Gemini-II, and by the APU
  v1 contraction analyzer ("aliases the output").  Before this probe Samsung
  compiled it and returned ``W @ y_initial`` without an error, and UPMEM failed
  only after simulation with an oracle mismatch.  Reading the element being
  written (``a[i] = a[i] + b[i]``) is not an alias: Samsung executes it (it
  used to report a simulator success and leave ``a`` untouched, because the
  output parameter was bound only when no operand read it; fixed in
  ``spmw_codegen.source_backend_binding``); UPMEM
  has no two-parameter pointwise family and rejects it as unmatched.  One
  buffer as both APU v1 multiplicands in source is rejected (each multiplicand
  owns a value layout).  At the call boundary (one NumPy array passed for two
  parameters) every executing route SUPPORTS the alias with pre-call
  semantics: Samsung, UPMEM and Gemini-II stage all inputs before writing an
  output; the APU v1 routes now copy an input that shares memory with the
  output before running (the functional route used to read the output while
  writing it, and returned neither the pre-call nor the source-order result).
* Extreme integers (e).  UPMEM int32 products and sums wrap modulo 2**32; the
  int64 reduction wraps modulo 2**64; k-means distances reject operands
  outside the exact-int32 bound with a finite error.  APU v1 uint16 wraps
  modulo 2**16.  Gemini-II int4 operands reach both ends of the exact int15
  dot range; a dot whose exact sum needs 17 bits is rejected for an int16
  accumulator.  AiM executes no numerics (``kind="simulator-timing-only"``,
  outputs untouched; ``test_gemv_all_targets.py::test_gemv_aim``), so it has
  no integer semantics to probe.
* Nonzero accumulators (f).  UPMEM carried MV, GEMM, histogram and reduction
  execute on uPIMulator in ``test_upmem_generality.py::test_sim_carried_*``;
  Gemini-II MVT carries ``x1``/``x2`` on the card in
  ``test_apu_g2_hardware.py::test_paper_source_kernel_on_card[mvt]``.  Here:
  an APU v1 FP16 carried output follows the source order bit for bit.
* Two schedules (l).  AiM SBK and ABK both simulate and the cost order equals
  the simulated order; two UPMEM pointwise decisions execute different device
  sources to identical outputs; Gemini-II BiCG fused and separate plans give
  identical outputs with one and two launches.
* All-zero results (g).  Outputs are prefilled with a sentinel; zeros must
  overwrite it.  An APU v1 source that fills its output with literal 0 before
  accumulating now starts the device program from zero (it used to transfer
  the caller's output as the initial value); any other store into the output
  is rejected.
* Masks and guards (h).  UPMEM padded and uneven partitions execute in
  ``test_upmem_generality.py::test_sim_uneven_*`` and
  ``::test_sim_selection_general_predicate_with_carried_count``.  Here: an
  APU v1 reduction tile padded to its power-of-two group and an odd GEMM
  shape; Gemini-II refuses a carrier with garbage outside the layout before
  launch.
* Strict versus reassociative (j) on the APU v1 functional route: the strict
  plan is bit-equal to the source-order FP16 loop; the reassociating plan is
  bit-equal to an FP16 halving tree over its padded group, differs from the
  source order on most rows, and stays within 1e-2 of FP32.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as df_region
from allo.ir.types import bfloat16, float16, int3, int4, int8, int15, int16, int32
from allo.ir.types import uint16
from allo.pim.apu_v1_vector_program import apu_v1_plan_regroups
from allo.pim.apu_v1_vectorize import IllegalContractionError
from allo.pim.apu_g2_source_route import APUG2SourceRouteError
from allo.pim.costs import aim_cost, apu_v1_cost, samsung_cost, upmem_cost
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import (
    build_aim_target,
    build_apu_g2_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)
from allo.spmw_match import UnsupportedWorkloadError
from benchmarks.apu_g2.workloads import PAPER_KERNELS

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
REASSOCIATE = {"reduction_order": "reassociate"}


def _record(name, **fields):
    print("G6_RECORD " + json.dumps({"probe": name, **fields}, sort_keys=True))


def _wrap(value: int, bits: int) -> int:
    value = int(value) % (1 << bits)
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def _f32(a):
    return np.asarray(a).astype(np.float32)


def _fp16_source_order(initial, products_by_k):
    """``out = initial; for k: out = fp16(out + fp16(product_k))``."""

    out = np.array(initial, dtype=np.float16, copy=True)
    for product in products_by_k:
        out = (out + np.asarray(product, dtype=np.float16)).astype(np.float16)
    return out


def _upmem_sim_gate():
    import conftest

    conftest._simulator_gate("upmem")


# --------------------------------------------------------------------- #
# Samsung HBM-PIM sources
# --------------------------------------------------------------------- #

SQ = 256
ELT = 4096


@df_region()
def samsung_gemv_sq(W: float16[SQ, SQ], x: float16[SQ], y: float16[SQ]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def kernel(lW: float16[SQ, SQ], lx: float16[SQ], ly: float16[SQ]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * 2
        for i in range(2):
            acc: float16 = 0
            for k in range(SQ):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc


@df_region()
def samsung_gemv_sq_carried(W: float16[SQ, SQ], x: float16[SQ], y: float16[SQ]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def kernel(lW: float16[SQ, SQ], lx: float16[SQ], ly: float16[SQ]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * 2
        for i in range(2):
            for k in range(SQ):
                ly[row0 + i] += lW[row0 + i, k] * lx[k]


@df_region()
def samsung_gemv_in_place(W: float16[SQ, SQ], y: float16[SQ]):
    @allo.work(mapping=[16, 8], args=[W, y])
    def kernel(lW: float16[SQ, SQ], ly: float16[SQ]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * 2
        for i in range(2):
            acc: float16 = 0
            for k in range(SQ):
                acc += lW[row0 + i, k] * ly[k]
            ly[row0 + i] = acc


@df_region()
def samsung_add(a: float16[ELT], b: float16[ELT], c: float16[ELT]):
    @allo.work(mapping=[1], args=[a, b, c])
    def kernel(la: float16[ELT], lb: float16[ELT], lc: float16[ELT]):
        for i in range(ELT):
            lc[i] = la[i] + lb[i]


@df_region()
def samsung_add_in_place(a: float16[ELT], b: float16[ELT]):
    @allo.work(mapping=[1], args=[a, b])
    def kernel(la: float16[ELT], lb: float16[ELT]):
        for i in range(ELT):
            la[i] = la[i] + lb[i]


# --------------------------------------------------------------------- #
# UPMEM sources (one DPU, reviewer-chosen shapes off the reference grid)
# --------------------------------------------------------------------- #

UM, UK = 36, 40
UN = 1000
UG = 20


@df_region()
def upmem_mv(A: int32[UM, UK], x: int32[UK], y: int32[UM]):
    @allo.work(mapping=[1], args=[A, x, y])
    def mv(lA: int32[UM, UK], lx: int32[UK], ly: int32[UM]):
        for i in range(UM):
            ly[i] = 0
            for j in range(UK):
                ly[i] += lA[i, j] * lx[j]


@df_region()
def upmem_mv_square_carried(A: int32[UM, UM], x: int32[UM], y: int32[UM]):
    @allo.work(mapping=[1], args=[A, x, y])
    def mv(lA: int32[UM, UM], lx: int32[UM], ly: int32[UM]):
        for i in range(UM):
            for j in range(UM):
                ly[i] += lA[i, j] * lx[j]


@df_region()
def upmem_mv_in_place(A: int32[UM, UM], y: int32[UM]):
    @allo.work(mapping=[1], args=[A, y])
    def mv(lA: int32[UM, UM], ly: int32[UM]):
        for i in range(UM):
            for j in range(UM):
                ly[i] += lA[i, j] * ly[j]


@df_region()
def upmem_add(A: int32[UN], B: int32[UN], C: int32[UN]):
    @allo.work(mapping=[1], args=[A, B, C])
    def f(lA: int32[UN], lB: int32[UN], lC: int32[UN]):
        for i in range(UN):
            lC[i] = lA[i] + lB[i]


@df_region()
def upmem_add_in_place(A: int32[UN], B: int32[UN]):
    @allo.work(mapping=[1], args=[A, B])
    def f(lA: int32[UN], lB: int32[UN]):
        for i in range(UN):
            lA[i] = lA[i] + lB[i]


@df_region()
def upmem_gemm_square(A: int32[UG, UG], B: int32[UG, UG], C: int32[UG, UG]):
    @allo.work(mapping=[1], args=[A, B, C])
    def mm(lA: int32[UG, UG], lB: int32[UG, UG], lC: int32[UG, UG]):
        for i in range(UG):
            for j in range(UG):
                lC[i, j] = 0
                for e in range(UG):
                    lC[i, j] += lA[i, e] * lB[e, j]


def _upmem(workload, **kwargs):
    return allo.compile(workload, build_upmem_target(), upmem_cost, **kwargs)


# --------------------------------------------------------------------- #
# AiM, APU v1 and Gemini-II sources
# --------------------------------------------------------------------- #

AIM_N = 256
_AIM_ROWS = AIM_N // 32


@df_region()
def aim_gemv(W: bfloat16[AIM_N, AIM_N], x: bfloat16[AIM_N], y: bfloat16[AIM_N]):
    @allo.work(mapping=[32], args=[W, x, y])
    def mv(lW: bfloat16[AIM_N, AIM_N], lx: bfloat16[AIM_N], ly: bfloat16[AIM_N]):
        (channel,) = allo.get_wid()
        row0 = channel * _AIM_ROWS
        for i in range(_AIM_ROWS):
            acc: bfloat16 = 0
            for j in range(AIM_N):
                acc += lW[row0 + i, j] * lx[j]
            ly[row0 + i] = acc


@df_region()
def aim_gemv_in_place(W: bfloat16[AIM_N, AIM_N], y: bfloat16[AIM_N]):
    @allo.work(mapping=[32], args=[W, y])
    def mv(lW: bfloat16[AIM_N, AIM_N], ly: bfloat16[AIM_N]):
        (channel,) = allo.get_wid()
        row0 = channel * _AIM_ROWS
        for i in range(_AIM_ROWS):
            acc: bfloat16 = 0
            for j in range(AIM_N):
                acc += lW[row0 + i, j] * ly[j]
            ly[row0 + i] = acc


AQ = 32


def apu_gemm_sq(left: float16[AQ, AQ], right: float16[AQ, AQ], result: float16[AQ, AQ]):
    for r, c in allo.grid(AQ, AQ):
        for d in allo.reduction(AQ):
            result[r, c] += left[r, d] * right[d, c]


def apu_gemv_in_place(W: float16[64, 64], y: float16[64]):
    for i in allo.grid(64):
        for k in allo.reduction(64):
            y[i] += W[i, k] * y[k]


def apu_gemm_self(L: float16[AQ, AQ], C: float16[AQ, AQ]):
    for r, c in allo.grid(AQ, AQ):
        for d in allo.reduction(AQ):
            C[r, c] += L[r, d] * L[d, c]


ZM, ZK, ZN = 32, 48, 40


def apu_gemm_zero_filled(L: float16[ZM, ZK], R: float16[ZK, ZN], C: float16[ZM, ZN]):
    for r, c in allo.grid(ZM, ZN):
        C[r, c] = 0
        for d in allo.reduction(ZK):
            C[r, c] += L[r, d] * R[d, c]


def apu_gemm_carried(L: float16[ZM, ZK], R: float16[ZK, ZN], C: float16[ZM, ZN]):
    for r, c in allo.grid(ZM, ZN):
        for d in allo.reduction(ZK):
            C[r, c] += L[r, d] * R[d, c]


def apu_gemm_one_filled(L: float16[ZM, ZK], R: float16[ZK, ZN], C: float16[ZM, ZN]):
    for r, c in allo.grid(ZM, ZN):
        C[r, c] = 1.0
        for d in allo.reduction(ZK):
            C[r, c] += L[r, d] * R[d, c]


def apu_gemm_half_filled(L: float16[ZM, ZK], R: float16[ZK, ZN], C: float16[ZM, ZN]):
    for r, c in allo.grid(ZM // 2, ZN):
        C[r, c] = 0.0
    for r, c in allo.grid(ZM, ZN):
        for d in allo.reduction(ZK):
            C[r, c] += L[r, d] * R[d, c]


UH_M, UH_K, UH_N = 48, 40, 56


def apu_u16_gemm(L: uint16[UH_M, UH_K], R: uint16[UH_K, UH_N], C: uint16[UH_M, UH_N]):
    for r, c in allo.grid(UH_M, UH_N):
        for d in allo.reduction(UH_K):
            C[r, c] += L[r, d] * R[d, c]


def _apu_v1(workload, **kwargs):
    kwargs.setdefault("backend", "functional")
    return allo.compile(workload, build_apu_v1_target(), apu_v1_cost, **kwargs)


def g2_gemv_in_place(A: int4[64, 64], y: int15[64]):
    for i in range(64):
        acc: int15 = 0
        for k in range(64):
            acc += A[i, k] * y[k]
        y[i] = acc


def g2_gemv_in_place_wide_operand(A: int4[64, 64], y: int16[64]):
    for i in range(64):
        acc: int15 = 0
        for k in range(64):
            acc += A[i, k] * y[k]
        y[i] = acc


def g2_gemv_carried_in_place(A: int4[64, 64], y: int15[64]):
    for i in range(64):
        for k in range(64):
            y[i] += A[i, k] * y[k]


def g2_gemm_sq(A: int4[32, 32], B: int4[32, 32], C: int15[32, 32]):
    for i in range(32):
        for j in range(32):
            acc: int15 = 0
            for k in range(32):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def g2_dot_i8_k2(A: int8[64, 2], x: int8[2], y: int16[64]):
    for i in range(64):
        acc: int16 = 0
        for k in range(2):
            acc += A[i, k] * x[k]
        y[i] = acc


def _g2(workload, **kwargs):
    return allo.compile(workload, build_apu_g2_target(), apu_g2_cost, **kwargs)


def _emulated_card_class():
    path = Path(__file__).resolve().parents[1] / "perf" / "test_apu_g2_source_route.py"
    spec = importlib.util.spec_from_file_location("g6_apu_g2_source_route", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._EmulatedCard


@pytest.fixture
def emulated_card(monkeypatch, tmp_path):
    """The software card of the source-route tests: cmake and the composed
    host binary are replaced; packing, attestation, the NumPy oracle check
    and the write-back are the production host path."""

    card = _emulated_card_class()()
    module = "allo.pim.apu_g2_composed_runtime"
    monkeypatch.setattr(f"{module}._require_hardware_stack", lambda: "emulated card")
    monkeypatch.setattr(f"{module}._run_command", card.run)
    monkeypatch.setenv("TENON_APU_G2_LOCK", str(tmp_path / "card.lock"))
    return card


# ===================================================================== #
# (i) Aliases
# ===================================================================== #


@pytest.mark.parametrize(
    "target, cost, workload, kwargs",
    [
        (build_samsung_target, samsung_cost, samsung_gemv_in_place, REASSOCIATE),
        (build_upmem_target, upmem_cost, upmem_mv_in_place, {}),
        (build_aim_target, aim_cost, aim_gemv_in_place, REASSOCIATE),
    ],
    ids=["samsung", "upmem", "aim"],
)
def test_source_alias_of_operand_and_result_is_rejected(target, cost, workload, kwargs):
    """``y[i] = sum_k W[i, k] * y[k]``: source order reads rows earlier
    iterations overwrote; the matched program reads y before writing it."""

    with pytest.raises(UnsupportedWorkloadError, match="aliases its result") as error:
        allo.compile(workload, target(), cost, backend="virtual", **kwargs)
    assert "Write the result to a separate buffer" in str(error.value)


def test_upmem_in_place_pointwise_is_rejected_as_unmatched():
    with pytest.raises(NotImplementedError, match="no UPMEM kernel family matches"):
        _upmem(upmem_add_in_place, backend="virtual")


def test_apu_v1_source_aliases_are_rejected():
    """The operand/result alias is the shared matcher contract (the same
    rejection as Samsung, UPMEM and AiM above); two multiplicands in one
    buffer is an APU v1 value-layout restriction."""
    with pytest.raises(UnsupportedWorkloadError, match="aliases its result"):
        _apu_v1(apu_gemv_in_place)
    with pytest.raises(IllegalContractionError, match="read one buffer"):
        _apu_v1(apu_gemm_self)


def test_apu_g2_source_aliases_are_rejected():
    for workload in (g2_gemv_in_place, g2_gemv_carried_in_place):
        with pytest.raises(UnsupportedWorkloadError, match="aliases its result"):
            _g2(workload)


def test_apu_g2_operand_wider_than_the_accumulator_is_refused_visibly():
    """``acc: int15 += A * y`` with ``y: int16`` truncates a value whose
    operand is wider than the accumulator; the device would have to narrow
    the operand, which no target op declares, so the matcher keeps the
    truncation visible and the shared coverage check fails before the
    Gemini-II planner runs."""
    with pytest.raises(UnsupportedWorkloadError, match=r"arith\.trunci<i21->i15>"):
        _g2(g2_gemv_in_place_wide_operand)


def test_samsung_in_place_elementwise_executes(samsung_sim_gate):
    """``a[i] = a[i] + b[i]`` reads only the element it writes: supported."""

    rng = np.random.default_rng(601)
    a0 = (rng.standard_normal(ELT) * 0.5).astype(np.float16)
    b = (rng.standard_normal(ELT) * 0.5).astype(np.float16)
    compiled = allo.compile(samsung_add_in_place, build_samsung_target(), samsung_cost)
    a = a0.copy()
    result = compiled(a, b)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(a, (a0 + b).astype(np.float16))


def test_samsung_call_aliases_read_pre_call_values(samsung_sim_gate):
    rng = np.random.default_rng(602)
    W0 = (rng.standard_normal((SQ, SQ)) * 0.1).astype(np.float16)
    x0 = (rng.standard_normal(SQ) * 0.1).astype(np.float16)
    target = build_samsung_target()

    # Output is the input vector: strict ordered program, bit-exact to the
    # source order started from the pre-call x.
    carried = allo.compile(samsung_gemv_sq_carried, target, samsung_cost)
    x = x0.copy()
    result = carried(W0, x, x)
    assert result.extra["kernel"] == "GENERIC_TEMPORAL_REDUCE"
    want = _fp16_source_order(x0, (W0[:, k] * x0[k] for k in range(SQ)))
    np.testing.assert_array_equal(x.view(np.uint16), want.view(np.uint16))

    # The vector is a strided view of the matrix (input/input overlap).
    W = W0.copy()
    view = W[:, 3]
    y = np.full(SQ, 7, dtype=np.float16)
    allo.compile(samsung_gemv_sq, target, samsung_cost, **REASSOCIATE)(W, view, y)
    np.testing.assert_allclose(_f32(y), _f32(W0) @ _f32(W0[:, 3]), rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(W, W0)

    # One array for both inputs and the output.
    a0 = (rng.standard_normal(ELT) * 0.5).astype(np.float16)
    a = a0.copy()
    allo.compile(samsung_add, target, samsung_cost)(a, a, a)
    np.testing.assert_array_equal(a, (a0 + a0).astype(np.float16))


def test_upmem_call_aliases_read_pre_call_values():
    _upmem_sim_gate()
    rng = np.random.default_rng(603)

    A = rng.integers(-50, 50, size=(UM, UM), dtype=np.int32)
    x0 = rng.integers(-50, 50, size=UM, dtype=np.int32)
    x = x0.copy()
    result = _upmem(upmem_mv_square_carried)(A, x, x)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(x, x0 + A.astype(np.int64) @ x0)

    a0 = rng.integers(I32_MIN, I32_MAX, size=UN, dtype=np.int64).astype(np.int32)
    a = a0.copy()
    _upmem(upmem_add)(a, a, a)
    np.testing.assert_array_equal(
        a, (2 * a0.astype(np.int64)).astype(np.uint32).astype(np.int32)
    )

    m0 = rng.integers(-20, 20, size=(UG, UG), dtype=np.int32)
    m = m0.copy()
    _upmem(upmem_gemm_square)(m, m, m)
    np.testing.assert_array_equal(m, m0.astype(np.int64) @ m0)


def test_apu_v1_functional_call_aliases_read_pre_call_values():
    rng = np.random.default_rng(604)
    L0 = (rng.standard_normal((AQ, AQ)) * 0.3).astype(np.float16)
    R0 = (rng.standard_normal((AQ, AQ)) * 0.3).astype(np.float16)
    compiled = _apu_v1(apu_gemm_sq)
    assert not apu_v1_plan_regroups(compiled.selected_plan)

    def source_order(initial, left, right):
        return _fp16_source_order(
            initial, (left[:, d : d + 1] * right[d : d + 1, :] for d in range(AQ))
        )

    shared = L0.copy()
    out = np.zeros((AQ, AQ), np.float16)
    compiled(shared, shared, out)
    np.testing.assert_array_equal(out, source_order(np.zeros_like(L0), L0, L0))

    result_is_left = L0.copy()
    compiled(result_is_left, R0, result_is_left)
    np.testing.assert_array_equal(result_is_left, source_order(L0, L0, R0))

    result_is_right = R0.copy()
    compiled(L0, result_is_right, result_is_right)
    np.testing.assert_array_equal(result_is_right, source_order(R0, L0, R0))


def test_apu_v1_device_route_stages_aliases_and_zero_fills(monkeypatch):
    """No board: the device runtime is replaced by a recorder, so this checks
    what the host hands the device for an aliased call and for a source that
    fills its output with 0 before accumulating."""

    import allo.pim.apu_v1_vector_runtime as runtime
    from allo.spmw_codegen import RunResult

    seen = []

    def record(compiled, arrays):
        seen.append(
            {name: np.array(value, copy=True) for name, value in arrays.items()}
        )
        seen[-1]["_shares"] = {
            name: np.shares_memory(value, arrays[compiled.analysis.output.value])
            for name, value in arrays.items()
        }
        return RunResult(1, "recorded", "apu_v1", kind="device", extra={})

    monkeypatch.setattr(runtime, "run_apu_v1_vector", record)
    rng = np.random.default_rng(605)

    aliased = _apu_v1(apu_gemm_sq, backend=None)
    L0 = (rng.standard_normal((AQ, AQ)) * 0.3).astype(np.float16)
    R0 = (rng.standard_normal((AQ, AQ)) * 0.3).astype(np.float16)
    buffer = L0.copy()
    aliased(buffer, R0, buffer)
    staged = seen.pop()
    assert staged["_shares"] == {"left": False, "right": False, "result": True}
    np.testing.assert_array_equal(staged["left"], L0)
    np.testing.assert_array_equal(staged["result"], L0)

    filled = _apu_v1(apu_gemm_zero_filled, backend=None)
    assert filled.analysis.output_initializer == "zero"
    L = (rng.standard_normal((ZM, ZK)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((ZK, ZN)) * 0.3).astype(np.float16)
    C = np.full((ZM, ZN), 7, np.float16)
    filled(L, R, C)
    assert not np.any(seen.pop()["C"])

    carried = _apu_v1(apu_gemm_carried, backend=None)
    assert carried.analysis.output_initializer == "carried"
    C = np.full((ZM, ZN), 7, np.float16)
    carried(L, R, C)
    assert np.all(seen.pop()["C"] == 7)


@pytest.mark.parametrize(
    "workload, what",
    [
        (apu_gemm_one_filled, "nonzero constant 1.0"),
        (apu_gemm_half_filled, "does not fill the whole buffer"),
    ],
    ids=["one_filled", "half_filled"],
)
def test_apu_v1_output_store_other_than_a_zero_fill_is_rejected(workload, what):
    """The accumulator-initialization contract is shared: a nonzero or
    partial initializer is refused by the matcher's coverage analysis before
    the APU v1 planner sees the program (previously the vector route's own
    text analysis made this judgment)."""
    with pytest.raises(UnsupportedWorkloadError, match=what):
        _apu_v1(workload, backend="virtual")


def test_apu_g2_emulated_call_aliases_read_pre_call_values(emulated_card):
    rng = np.random.default_rng(606)
    gemm = _g2(g2_gemm_sq)
    A0 = rng.integers(-8, 8, size=(32, 32)).astype(np.int16)
    B0 = rng.integers(-8, 8, size=(32, 32)).astype(np.int16)

    shared = A0.copy()
    out = np.full((32, 32), 7, np.int16)
    assert gemm(shared, shared, out).kind == "device"
    np.testing.assert_array_equal(out, A0.astype(np.int64) @ A0)

    output_is_lhs = A0.copy()
    gemm(output_is_lhs, B0, output_is_lhs)
    np.testing.assert_array_equal(output_is_lhs, A0.astype(np.int64) @ B0)

    mvt = _g2(PAPER_KERNELS["mvt"])
    M = rng.integers(-4, 4, size=(256, 256)).astype(np.int16)
    y1 = rng.integers(-4, 4, size=256).astype(np.int16)
    y2 = rng.integers(-4, 4, size=256).astype(np.int16)
    x2 = rng.integers(-1000, 1000, size=256).astype(np.int16)
    x1_is_y1 = y1.copy()
    x2_out = x2.copy()
    mvt(M, x1_is_y1, y2, x1_is_y1, x2_out)
    np.testing.assert_array_equal(x1_is_y1, y1 + M.astype(np.int64) @ y1)
    np.testing.assert_array_equal(x2_out, x2 + M.T.astype(np.int64) @ y2)


# ===================================================================== #
# (e) Extreme integers
# ===================================================================== #


def test_upmem_int32_gemv_products_and_sums_wrap():
    _upmem_sim_gate()
    rng = np.random.default_rng(611)
    A = rng.integers(I32_MIN, I32_MAX, size=(UM, UK), dtype=np.int64).astype(np.int32)
    x = rng.integers(I32_MIN, I32_MAX, size=UK, dtype=np.int64).astype(np.int32)
    A[0, :] = I32_MIN
    A[1, :] = I32_MAX
    x[::2], x[1::2] = I32_MIN, I32_MAX
    y = np.full(UM, -7, np.int32)
    result = _upmem(upmem_mv)(A, x, y)
    assert result.kind == "simulator"
    want = [_wrap(sum(int(a) * int(b) for a, b in zip(row, x)), 32) for row in A]
    np.testing.assert_array_equal(y, np.array(want, dtype=np.int32))


def test_upmem_int64_reduction_wraps_at_the_extremes():
    import test_upmem_generality as generality

    _upmem_sim_gate()
    rng = np.random.default_rng(612)
    A = rng.integers(0, I32_MAX, size=UN, dtype=np.int64).astype(np.int32)
    A[:4] = [I32_MAX, I32_MAX, I32_MIN, I32_MAX]
    start = (1 << 63) - 1000
    P = np.array([start], dtype=np.int64)
    _upmem(generality.reduction_region(UN, carried=True))(A, P)
    assert int(P[0]) == _wrap(start + sum(int(v) for v in A), 64)
    assert int(P[0]) < 0  # the carried start plus a positive sum wrapped


def test_upmem_kmeans_distances_reject_operands_beyond_the_exact_bound():
    import test_upmem_generality as generality

    _upmem_sim_gate()
    points = np.zeros((24, 8), np.int32)
    centers = np.zeros((4, 8), np.int32)
    points[0, 0] = I32_MIN
    distances = np.full((24, 4), -7, np.int64)
    compiled = _upmem(generality.kmeans_distances_region(24, 8, 4))
    with pytest.raises(ValueError, match="exceeds 8191, the bound for an exact int32"):
        compiled(points, centers, distances)
    assert np.all(distances == -7)


def test_apu_v1_uint16_contraction_wraps_at_the_extremes():
    rng = np.random.default_rng(613)
    top = (1 << 16) - 1
    L = rng.integers(0, 1 << 16, size=(UH_M, UH_K), dtype=np.int64).astype(np.uint16)
    R = rng.integers(0, 1 << 16, size=(UH_K, UH_N), dtype=np.int64).astype(np.uint16)
    C0 = rng.integers(top - 16, top + 1, size=(UH_M, UH_N), dtype=np.int64).astype(
        np.uint16
    )
    L[0, :] = top
    R[:, 0] = top
    C = C0.copy()
    compiled = _apu_v1(apu_u16_gemm)
    compiled(L, R, C)
    want = (C0.astype(object) + L.astype(object).dot(R.astype(object))) % (1 << 16)
    np.testing.assert_array_equal(C, want.astype(np.uint16))
    # 65535**2 = 1 mod 2**16, so the all-65535 row/column pair adds 40.
    assert int(C[0, 0]) == (int(C0[0, 0]) + UH_K) % (1 << 16)


def test_apu_g2_int4_gemm_reaches_both_ends_of_the_exact_range(emulated_card):
    gemm = _g2(PAPER_KERNELS["gemm"])
    A = np.full((64, 128), -8, np.int16)
    B = np.full((128, 31), -8, np.int16)
    B[:, 1::2] = 7
    C = np.full((64, 31), 7, np.int16)
    gemm(A, B, C)
    np.testing.assert_array_equal(C, A.astype(np.int64) @ B)
    assert (C.max(), C.min()) == (8192, -7168)


def test_apu_g2_dot_wider_than_its_accumulator_is_rejected():
    with pytest.raises(
        APUG2SourceRouteError, match="narrower than the exact int17 sum"
    ):
        _g2(g2_dot_i8_k2)


# ===================================================================== #
# (f) Nonzero accumulator on an executed path
# ===================================================================== #


def test_apu_v1_fp16_carried_output_follows_the_source_order():
    rng = np.random.default_rng(621)
    L = (rng.standard_normal((ZM, ZK)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((ZK, ZN)) * 0.3).astype(np.float16)
    C0 = (rng.standard_normal((ZM, ZN)) * 2.0).astype(np.float16)
    compiled = _apu_v1(apu_gemm_carried)
    assert compiled.realization.carries_output_ingress
    C = C0.copy()
    result = compiled(L, R, C)
    assert result.extra["numerics"] == "source order"
    want = _fp16_source_order(
        C0, (L[:, d : d + 1] * R[d : d + 1, :] for d in range(ZK))
    )
    np.testing.assert_array_equal(C.view(np.uint16), want.view(np.uint16))
    from_zero = _fp16_source_order(
        np.zeros_like(C0), (L[:, d : d + 1] * R[d : d + 1, :] for d in range(ZK))
    )
    assert np.count_nonzero(C != from_zero) > C.size // 2


# ===================================================================== #
# (l) Two feasible schedules traced to execution
# ===================================================================== #


def test_aim_sbk_and_abk_simulate_in_cost_order(aim_sim_gate):
    from allo.spmw_autoschedule import _aim_layout_candidates

    target = build_aim_target()
    ranked = allo.compile(aim_gemv, target, aim_cost, backend="virtual", **REASSOCIATE)
    sbk_layout, abk_layout = _aim_layout_candidates(
        target, ranked.compiled.trace.matches
    )
    rng = np.random.default_rng(631)
    operands = (
        (rng.standard_normal((AIM_N, AIM_N)) * 0.1).astype(np.float32),
        (rng.standard_normal(AIM_N) * 0.1).astype(np.float32),
        np.zeros(AIM_N, np.float32),
    )
    runs = {}
    for name, layout in (("sbk", sbk_layout), ("abk", abk_layout)):
        compiled = allo.compile(
            aim_gemv, target, aim_cost, layout=layout, **REASSOCIATE
        )
        result = compiled(*operands)
        assert result.kind == "simulator-timing-only" and result.outputs == {}
        runs[name] = (int(compiled.estimate().cycles), int(result.cycles))
        _record(
            "aim_bank_scope",
            layout=name,
            estimate=runs[name][0],
            simulated=runs[name][1],
        )
    assert runs["sbk"][0] > runs["abk"][0]
    assert runs["sbk"][1] > runs["abk"][1]


def test_upmem_two_pointwise_decisions_execute_identically():
    from allo.spmw_autoschedule import _upmem_enumerate

    _upmem_sim_gate()
    target = build_upmem_target()
    n = 3000
    import test_upmem_generality as generality

    region = generality.pointwise_region(n, "add")
    ranked = allo.compile(region, target, upmem_cost, backend="virtual")
    decisions = {
        (c.extra["data_layout"], c.extra["dma_bytes"]): c
        for c in _upmem_enumerate(target, ranked.compiled.trace.matches)
    }
    rng = np.random.default_rng(632)
    A = rng.integers(I32_MIN, I32_MAX, size=n, dtype=np.int64).astype(np.int32)
    B = rng.integers(I32_MIN, I32_MAX, size=n, dtype=np.int64).astype(np.int32)
    want = (A.astype(np.int64) + B).astype(np.uint32).astype(np.int32)
    runs = {}
    for key in (("fused_replicated", 256), ("separate", 64)):
        compiled = allo.compile(region, target, upmem_cost, layout=decisions[key])
        (segment,) = [
            s for s in compiled.compiled.layout_ctx.segments if s.kind == "dpu"
        ]
        C = np.full(n, -7, np.int32)
        result = compiled(A, B, C)
        np.testing.assert_array_equal(C, want)
        source = hashlib.sha256(segment.plan.device_source().encode()).hexdigest()
        runs[key] = (int(compiled.estimate().cycles), int(result.cycles), source)
        _record(
            "upmem_pointwise",
            decision=list(key),
            estimate=runs[key][0],
            simulated=runs[key][1],
            device_source_sha256=source,
        )
    fused, separate = runs[("fused_replicated", 256)], runs[("separate", 64)]
    assert fused[2] != separate[2]
    assert fused[1] != separate[1]
    assert (fused[0] < separate[0]) == (fused[1] < separate[1])


def test_apu_g2_fused_and_separate_bicg_plans_agree(emulated_card):
    rng = np.random.default_rng(633)
    A = rng.integers(-4, 4, size=(256, 256)).astype(np.int16)
    p = rng.integers(-4, 4, size=256).astype(np.int16)
    r = rng.integers(-4, 4, size=256).astype(np.int16)
    outputs = {}
    for layout in (None, "separate/k1"):
        compiled = _g2(PAPER_KERNELS["bicg"], layout=layout)
        q = np.full(256, 7, np.int16)
        s = np.full(256, 7, np.int16)
        result = compiled(A, p, r, q, s)
        outputs[compiled.compiled.plan.name] = (q, s, result.extra["launches"])
    fused, separate = outputs["fused/k1"], outputs["separate/k1"]
    np.testing.assert_array_equal(fused[0], A.astype(np.int64) @ p)
    np.testing.assert_array_equal(fused[1], r.astype(np.int64) @ A)
    np.testing.assert_array_equal(fused[0], separate[0])
    np.testing.assert_array_equal(fused[1], separate[1])
    assert (len(fused[2]), len(separate[2])) == (1, 2)
    assert {l["layout_fingerprint"] for l in fused[2]}.isdisjoint(
        {l["layout_fingerprint"] for l in separate[2]}
    )


# ===================================================================== #
# (g) All-zero results
# ===================================================================== #


def test_upmem_zero_vector_gemv_writes_zeros():
    _upmem_sim_gate()
    A = np.random.default_rng(641).integers(-50, 50, size=(UM, UK), dtype=np.int32)
    y = np.full(UM, -7, np.int32)
    result = _upmem(upmem_mv)(A, np.zeros(UK, np.int32), y)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(y, np.zeros(UM, np.int32))


def test_apu_v1_zero_results_overwrite_a_sentinel():
    rng = np.random.default_rng(642)
    L = (rng.standard_normal((ZM, ZK)) * 0.3).astype(np.float16)
    zero = np.zeros((ZK, ZN), np.float16)

    C = np.full((ZM, ZN), 7, np.float16)
    _apu_v1(apu_gemm_zero_filled)(L, zero, C)
    np.testing.assert_array_equal(C, np.zeros((ZM, ZN), np.float16))

    # A carried output is the initial value plus zero: the sentinel stays.
    C = np.full((ZM, ZN), 7, np.float16)
    _apu_v1(apu_gemm_carried)(L, zero, C)
    np.testing.assert_array_equal(C, np.full((ZM, ZN), 7, np.float16))


def test_apu_g2_zero_operands_give_zero_dots(emulated_card):
    rng = np.random.default_rng(643)
    gemm = _g2(PAPER_KERNELS["gemm"])
    B = rng.integers(-8, 8, size=(128, 31)).astype(np.int16)
    C = np.full((64, 31), 7, np.int16)
    gemm(np.zeros((64, 128), np.int16), B, C)
    np.testing.assert_array_equal(C, np.zeros((64, 31), np.int16))

    # Zero matrix with carried accumulators: the result is the accumulator.
    mvt = _g2(PAPER_KERNELS["mvt"])
    x1 = rng.integers(-1000, 1000, size=256).astype(np.int16)
    x2 = rng.integers(-1000, 1000, size=256).astype(np.int16)
    y = rng.integers(-4, 4, size=256).astype(np.int16)
    out1, out2 = x1.copy(), x2.copy()
    mvt(np.zeros((256, 256), np.int16), y, y, out1, out2)
    np.testing.assert_array_equal(out1, x1)
    np.testing.assert_array_equal(out2, x2)


# ===================================================================== #
# (h) Masks and guards, (j) strict versus reassociative
# ===================================================================== #


def _halving_tree_fp16(products, group):
    """FP16 halving tree over ``group``-wide blocks (zero padded), blocks
    added in ascending order into an FP16 zero."""

    rows, width = products.shape
    padded = np.zeros((rows, -(-width // group) * group), np.float16)
    padded[:, :width] = products
    out = np.zeros(rows, np.float16)
    for start in range(0, padded.shape[1], group):
        block = padded[:, start : start + group]
        while block.shape[1] > 1:
            half = block.shape[1] // 2
            block = (block[:, :half] + block[:, half : 2 * half]).astype(np.float16)
        out = (out + block[:, 0]).astype(np.float16)
    return out


@pytest.mark.parametrize(
    "M, K, seed", [(512, 256, 651), (384, 200, 652)], ids=["full_group", "padded_group"]
)
def test_apu_v1_strict_and_reassociative_gemv(M, K, seed):
    def gemv(W: float16[M, K], x: float16[K], y: float16[M]):
        for i in allo.grid(M):
            for k in allo.reduction(K):
                y[i] += W[i, k] * x[k]

    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.2).astype(np.float16)
    x = (rng.standard_normal(K) * 0.2).astype(np.float16)
    exact = _f32(W) @ _f32(x)
    source = _fp16_source_order(
        np.zeros(M, np.float16), (W[:, k] * x[k] for k in range(K))
    )

    strict = _apu_v1(gemv)
    assert not apu_v1_plan_regroups(strict.selected_plan)
    y = np.full(M, 0, np.float16)
    strict(W, x, y)
    np.testing.assert_array_equal(y.view(np.uint16), source.view(np.uint16))

    relaxed = _apu_v1(gemv, **REASSOCIATE)
    plan = relaxed.selected_plan
    assert apu_v1_plan_regroups(plan)
    group = plan.reduction_strategy.group_size
    assert group & (group - 1) == 0
    assert (K % group != 0) == (K == 200)  # 200 lanes padded inside the tree
    y = np.zeros(M, np.float16)
    result = relaxed(W, x, y)
    assert result.extra["plan_regroups_reduction"] is True
    np.testing.assert_array_equal(
        y.view(np.uint16), _halving_tree_fp16(W * x, group).view(np.uint16)
    )
    differing = int(np.count_nonzero(y.view(np.uint16) != source.view(np.uint16)))
    assert differing > M // 2
    np.testing.assert_allclose(_f32(y), exact, rtol=1e-2, atol=1e-2)
    _record(
        "apu_v1_reduction_order",
        M=M,
        K=K,
        plan=plan.name,
        group=group,
        rows_differing_from_source_order=differing,
    )


def test_apu_v1_odd_gemm_shape_follows_the_source_order():
    def odd(L: float16[37, 19], R: float16[19, 23], C: float16[37, 23]):
        for r, c in allo.grid(37, 23):
            for d in allo.reduction(19):
                C[r, c] += L[r, d] * R[d, c]

    rng = np.random.default_rng(653)
    L = (rng.standard_normal((37, 19)) * 0.3).astype(np.float16)
    R = (rng.standard_normal((19, 23)) * 0.3).astype(np.float16)
    compiled = _apu_v1(odd)
    assert compiled.realization is not None, compiled.realization_error
    C = np.zeros((37, 23), np.float16)
    compiled(L, R, C)
    want = _fp16_source_order(
        np.zeros((37, 23)), (L[:, d : d + 1] * R[d : d + 1, :] for d in range(19))
    )
    np.testing.assert_array_equal(C.view(np.uint16), want.view(np.uint16))


def test_apu_g2_garbage_outside_the_layout_is_refused_before_launch(
    monkeypatch, emulated_card
):
    import allo.pim.apu_g2_composed_runtime as runtime
    from allo.pim.apu_g2_composed_layout import apu_g2_composed_layout_image

    rng = np.random.default_rng(654)
    original = runtime.pack_apu_g2_composed_operand

    def with_garbage(program, value, index):
        carrier = np.array(original(program, value, index), copy=True)
        sets, columns = apu_g2_composed_layout_image(program)
        occupied = np.zeros(carrier.shape, dtype=bool)
        occupied[sets, columns] = True
        garbage = rng.integers(1, 7, size=carrier.shape).astype(carrier.dtype)
        carrier[~occupied] = garbage[~occupied]
        return carrier

    monkeypatch.setattr(runtime, "pack_apu_g2_composed_operand", with_garbage)
    gemm = _g2(PAPER_KERNELS["gemm"])
    A = rng.integers(-8, 8, size=(64, 128)).astype(np.int16)
    B = rng.integers(-8, 8, size=(128, 31)).astype(np.int16)
    C = np.full((64, 31), 7, np.int16)
    with pytest.raises(RuntimeError, match="nonzero lanes outside the layout"):
        gemm(A, B, C)
    assert emulated_card.launches == []
    assert np.all(C == 7)
