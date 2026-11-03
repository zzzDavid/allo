# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM element-type contract (audit U-01, U-02; gate G2).

Every family's DPU template computes fixed widths (int32 data, int64 only
for the reduction / k-means-distance / gradient outputs). Any other operand
type must fail at compile time with a message naming the operand, its type
and the supported types, never be packed through int32 and "pass".
Compile-only: no simulator runs here.
"""

import numpy as np
import pytest

import allo
from allo.backend.c import WideIntegerError
from allo.dataflow import region as df_region
from allo.ir.types import float32, int16, int32, int64, uint32
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.spmw_upmem import (
    UPMEM_ROLE_ELEMENT_TYPES,
    _i32,
    _i64,
    _store_exact,
)

N = 1536
M, K = 48, 64
GR, GK, GC = 12, 64, 128
HN, HB, HD = 1536, 128, 12
KP, KD, KK = 24, 8, 4
GS, GF, GW = 120, 8, 9


def _compile(workload, **kw):
    return allo.compile(
        workload, build_upmem_target(), upmem_cost, backend="virtual", **kw
    )


def _va(ta, tb, tc):
    @df_region()
    def va(A: ta[N], B: tb[N], C: tc[N]):
        @allo.work(mapping=[1], args=[A, B, C])
        def add(lA: ta[N], lB: tb[N], lC: tc[N]):
            for i in range(N):
                lC[i] = lA[i] + lB[i]

    return va


def _gemv(t):
    @df_region()
    def gemv(A: t[M, K], x: t[K], y: t[M]):
        @allo.work(mapping=[1], args=[A, x, y])
        def mv(lA: t[M, K], lx: t[K], ly: t[M]):
            for i in range(M):
                ly[i] = 0
                for j in range(K):
                    ly[i] += lA[i, j] * lx[j]

    return gemv


def _gemm(t):
    @df_region()
    def gemm(A: t[GR, GK], B: t[GK, GC], C: t[GR, GC]):
        @allo.work(mapping=[1], args=[A, B, C])
        def mm(lA: t[GR, GK], lB: t[GK, GC], lC: t[GR, GC]):
            for i in range(GR):
                for j in range(GC):
                    lC[i, j] = 0
                    for c in range(GK):
                        lC[i, j] += lA[i, c] * lB[c, j]

    return gemm


def _reduction(cell):
    @df_region()
    def red(A: int32[N], P: cell[1]):
        @allo.work(mapping=[1], args=[A, P])
        def reduce(lA: int32[N], lP: cell[1]):
            lP[0] = 0
            for i in range(N):
                lP[0] += lA[i]

    return red


def _histogram(t):
    @df_region()
    def hist(A: t[HN], H: int32[HB]):
        @allo.work(mapping=[1], args=[A, H])
        def histogram(lA: t[HN], lH: int32[HB]):
            for j in range(HB):
                lH[j] = 0
            for i in range(HN):
                b: int32 = (lA[i] * HB) >> HD
                if b >= 0 and b < HB:
                    lH[b] += 1

    return hist


def _selection(t, out):
    @df_region()
    def sel(A: t[N], O: out[N], count: int32[1]):
        @allo.work(mapping=[1], args=[A, O, count])
        def select_odd(lA: t[N], lout: out[N], lcount: int32[1]):
            lcount[0] = 0
            for i in range(N):
                if (lA[i] & 1) != 0:
                    lout[lcount[0]] = lA[i]
                    lcount[0] += 1

    return sel


def _kmeans_distances(out):
    @df_region()
    def km(P: int32[KP, KD], C: int32[KK, KD], D: out[KP, KK]):
        @allo.work(mapping=[1], args=[P, C, D])
        def distances(lP: int32[KP, KD], lC: int32[KK, KD], lD: out[KP, KK]):
            for p in range(KP):
                for c in range(KK):
                    lD[p, c] = 0
                    for d in range(KD):
                        lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

    return km


def _logistic(out):
    @df_region()
    def logistic(S: int32[GS, GW], G: out[GF]):
        @allo.work(mapping=[1], args=[S, G])
        def gradient(lS: int32[GS, GW], lG: out[GF]):
            for f in range(GF):
                lG[f] = 0
            for s in range(GS):
                for f in range(GF):
                    lG[f] += lS[s, f] * (1 - 2 * lS[s, GF])

    return logistic


@pytest.mark.parametrize(
    "workload, operand, mlir_type, family",
    [
        (_va(int64, int64, int64), "lA", "memref<1536xi64>", "pointwise"),
        (_va(float32, float32, float32), "lA", "memref<1536xf32>", "pointwise"),
        (_va(int16, int16, int32), "lA", "memref<1536xi16>", "pointwise"),
        (_gemv(int64), "ly", "memref<48xi64>", "matrix_vector"),
        (_gemm(int64), "lA", "memref<12x64xi64>", "gemm"),
        (_reduction(int32), "lA", None, "reduction"),
        (_histogram(int64), "lA", "memref<1536xi64>", "histogram"),
        (_selection(int64, int64), "lA", "memref<1536xi64>", "selection"),
        (_kmeans_distances(int32), "lD", "memref<24x4xi32>", "kmeans_distances"),
        (_logistic(int32), "lG", "memref<8xi32>", "feature_gradient"),
    ],
    ids=[
        "va-i64",
        "va-f32",
        "va-i16",
        "gemv-i64",
        "gemm-i64",
        "red-i32-cell",
        "hist-i64",
        "sel-i64",
        "kmeans-i32-out",
        "logistic-i32-out",
    ],
)
def test_unsupported_element_type_is_rejected_at_recognition(
    workload, operand, mlir_type, family
):
    from allo.spmw_match import UnsupportedWorkloadError

    with pytest.raises((NotImplementedError, UnsupportedWorkloadError)) as error:
        _compile(workload)
    message = str(error.value)
    if isinstance(error.value, UnsupportedWorkloadError):
        # A pointwise add of int16 operands into an int32 output evaluates in
        # a wider type than its operands: the shared precision contract
        # (C-26) refuses it before UPMEM's recognition names the family.
        assert "evaluates this operation in" in message
        assert f"'{operand}'" in message or operand in message
        return
    assert f"UPMEM {family}" in message
    if mlir_type is None:
        # The int32 reduction cell: name the cell, not the input.
        assert "'lP'" in message and "xi32>" in message
    else:
        assert f"{operand!r}" in message and mlir_type in message
    assert "supported:" in message


@pytest.mark.parametrize(
    "workload, operand, dtype",
    [
        (_va(uint32, uint32, uint32), "lA", "ui32"),
        (_va(int32, int32, int64), "lC", "i64"),
        (_selection(int32, int64), "lout", "i64"),
    ],
    ids=["va-u32", "va-i64-out", "sel-i64-out"],
)
def test_signedness_and_output_type_are_checked_on_the_abi(workload, operand, dtype):
    # Signless MLIR hides uint32 and the pointwise output's type from the
    # matcher; the final compile checks the work function's ABI.  An int32
    # add stored into an int64 output is refused earlier by the shared
    # precision contract (C-26: the source evaluates it in a wider type).
    from allo.spmw_match import UnsupportedWorkloadError

    with pytest.raises((NotImplementedError, UnsupportedWorkloadError)) as error:
        _compile(workload)
    message = str(error.value)
    if isinstance(error.value, UnsupportedWorkloadError):
        assert "evaluates this operation in" in message
        return
    assert f"{operand!r}" in message and f"element type {dtype}" in message
    assert "supported:" in message and "out: int32" in message


def test_int32_pointwise_compiles_and_rejects_a_wider_caller_array():
    compiled = _compile(_va(int32, int32, int32))
    ctx = compiled.compiled.layout_ctx
    a = np.arange(N, dtype=np.int32)
    ctx.prepare([a, a, np.zeros(N, np.int32)])
    with pytest.raises(TypeError, match="dtype int32"):
        ctx.prepare([a.astype(np.int64), a, np.zeros(N, np.int32)])


def test_role_contract_covers_every_dpu_family():
    from allo.spmw_upmem import _INPUT_ROLES, _OUTPUT_ROLES

    for family, inputs in _INPUT_ROLES.items():
        roles = set(inputs) | set(_OUTPUT_ROLES[family])
        assert roles <= set(UPMEM_ROLE_ELEMENT_TYPES[family])


def test_packers_never_truncate():
    assert _i32([1, -2]) == np.array([1, -2], "<i4").tobytes()
    assert _i64(np.array([2**62], np.int64)) == np.array([2**62], "<i8").tobytes()
    for values in ([2**31], np.array([-(2**31) - 1], np.int64), [2**70]):
        with pytest.raises(ValueError, match="does not fit"):
            _i32(values, "x")
    with pytest.raises(ValueError, match="does not fit"):
        _i64(np.array([2**63], np.uint64), "out")
    with pytest.raises(TypeError, match="float64"):
        _i32(np.array([1.5]), "x")


def test_write_back_refuses_a_lossy_caller_dtype():
    view = np.zeros(2, np.int16)
    with pytest.raises(ValueError, match="does not fit"):
        _store_exact(view, np.array([1, 2**20]), "out")
    with pytest.raises(TypeError, match="float32"):
        _store_exact(np.zeros(2, np.float32), np.array([1, 2]), "out")
    _store_exact(view, np.array([1, -3]), "out")
    assert view.tolist() == [1, -3]


def test_wide_signed_gradient_without_int64_temporary_fails_closed():
    # The natural linear-gradient source computes ((x * y) * -32) >> 8 in 96
    # bits; the portable-C oracle cannot evaluate that shift exactly, so the
    # compile stops instead of building wrong expected bytes.
    @df_region()
    def linear(S: int32[GS, GW], G: int64[GF]):
        @allo.work(mapping=[1], args=[S, G])
        def gradient(lS: int32[GS, GW], lG: int64[GF]):
            for f in range(GF):
                lG[f] = 0
            for s in range(GS):
                for f in range(GF):
                    lG[f] += ((lS[s, f] * lS[s, GF]) * -32) >> 8

    with pytest.raises(WideIntegerError, match="arith.shrsi on a 96-bit"):
        _compile(linear)


def test_kmeans_host_group_with_wide_shift_does_not_run():
    # Audit U-02 probe (p6_hostgroup): a signed >64-bit shift in the
    # host-placed update group went through uint64_t.

    @df_region()
    def kmeans(
        P: int32[KP, KD],
        C: int32[KK, KD],
        D: int64[KP, KK],
        C_new: int64[KK, KD],
        counts: int32[KK],
    ):
        @allo.work(mapping=[1], args=[P, C, D])
        def distances(lP: int32[KP, KD], lC: int32[KK, KD], lD: int64[KP, KK]):
            for p in range(KP):
                for c in range(KK):
                    lD[p, c] = 0
                    for d in range(KD):
                        lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[KP, KD], uD: int64[KP, KK], uC: int64[KK, KD], ucounts: int32[KK]
        ):
            sums: int64[KK, KD] = 0
            for c0 in range(KK):
                ucounts[c0] = 0
            for p in range(KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(KK):
                for d1 in range(KD):
                    t: int64 = sums[c1, d1]
                    uC[c1, d1] = (t * uD[0, 0]) >> 40

    @allo.host_program(kmeans)
    def host(P, C, D, C_new, counts):
        allo.launch("distances", P, C, D)
        allo.launch("update", P, D, C_new, counts)

    # Either the coverage gate or the wide-integer gate stops it; neither
    # yields a runnable program.
    with pytest.raises((NotImplementedError, ValueError)):
        _compile(kmeans, host_moves=host)


def test_kmeans_host_group_with_literal_wide_scale_does_not_run():
    # Audit F-02: the literal scale -(1 << 40) used to be typed i32 and fold
    # to 0, so the host group computed (t * 0) >> 40 in 96 bits.  It is now
    # the int64 constant; the 128-bit signed shift is rejected.

    @df_region()
    def kmeans(
        P: int32[KP, KD],
        C: int32[KK, KD],
        D: int64[KP, KK],
        C_new: int64[KK, KD],
        counts: int32[KK],
    ):
        @allo.work(mapping=[1], args=[P, C, D])
        def distances(lP: int32[KP, KD], lC: int32[KK, KD], lD: int64[KP, KK]):
            for p in range(KP):
                for c in range(KK):
                    lD[p, c] = 0
                    for d in range(KD):
                        lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[KP, KD], uD: int64[KP, KK], uC: int64[KK, KD], ucounts: int32[KK]
        ):
            sums: int64[KK, KD] = 0
            for c0 in range(KK):
                ucounts[c0] = 0
            for p in range(KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(KK):
                for d1 in range(KD):
                    t: int64 = sums[c1, d1]
                    uC[c1, d1] = (t * -(1 << 40)) >> 40

    @allo.host_program(kmeans)
    def host(P, C, D, C_new, counts):
        allo.launch("distances", P, C, D)
        allo.launch("update", P, D, C_new, counts)

    with pytest.raises(WideIntegerError, match="arith.shrsi on a 128-bit integer"):
        _compile(kmeans, host_moves=host)
