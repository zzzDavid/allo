# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The UPMEM paper cases as ``@allo.work`` sources (spec 003 U8).

Each case is a dataflow region whose ``mapping=[dpus]`` work ids are the host
fan-out. Shapes follow ``$TENON_ARTIFACTS/upmem/workloads.json``; data is
int32 and RED accumulates into int64. ``build_case`` returns the region and,
for 2mm and 3mm only, a host program that launches the GEMM kernel two or
three times. Indices are written inline (``d * ROWS + i``).
"""


import json
import os
from pathlib import Path

import numpy as np

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import int32, int64

UPMEM_CASES = (
    "va",
    "red",
    "mtv",
    "gemv",
    "geva",
    "ttv",
    "mmtv",
    "hist",
    "sel",
    "kmeans",
    "linear_reg",
    "logistic_reg",
    "1mm",
    "2mm",
    "3mm",
    "conv",
)
REGULAR_CASES = (
    "va", "red", "mtv", "gemv", "geva", "ttv", "mmtv", "1mm", "2mm", "3mm", "conv",
)

GEVA_ALPHA, GEVA_BETA = 2, -1
GEMV_ALPHA = 2


def _split(extent: int, dpus: int, what: str) -> int:
    if dpus <= 0 or extent % dpus:
        raise ValueError(f"{what}={extent} does not split evenly over {dpus} DPUs")
    return extent // dpus


# Factories rebind their shape arguments to ``_up_*`` names before the kernel
# bodies use them: Allo resolves kernel names from caller frames at customize
# time, so a caller local such as ``n`` or ``k`` would otherwise shadow them.


def pointwise_workload(n: int, dpus: int = 1, *, axpby: bool = False):
    """``C = A + B`` (VA) or ``C = 2 A - B`` (GEVA), element-partitioned."""
    _up_n = n
    _up_dpus = dpus
    _up_s = _split(_up_n, _up_dpus, "N")

    if axpby:

        @_df_region()
        def geva(A: int32[_up_n], B: int32[_up_n], C: int32[_up_n]):
            @allo.work(mapping=[_up_dpus], args=[A, B, C])
            def axpby(lA: int32[_up_n], lB: int32[_up_n], lC: int32[_up_n]):
                (d,) = allo.get_wid()
                for i in range(_up_s):
                    lC[d * _up_s + i] = GEVA_ALPHA * lA[d * _up_s + i] + GEVA_BETA * lB[d * _up_s + i]

        return geva

    @_df_region()
    def va(A: int32[_up_n], B: int32[_up_n], C: int32[_up_n]):
        @allo.work(mapping=[_up_dpus], args=[A, B, C])
        def add(lA: int32[_up_n], lB: int32[_up_n], lC: int32[_up_n]):
            (d,) = allo.get_wid()
            for i in range(_up_s):
                lC[d * _up_s + i] = lA[d * _up_s + i] + lB[d * _up_s + i]

    return va


def reduction_workload(n: int, dpus: int = 1):
    """One int64 partial sum per DPU; the host adds the partials."""
    _up_n = n
    _up_dpus = dpus
    _up_s = _split(_up_n, _up_dpus, "N")

    @_df_region()
    def red(A: int32[_up_n], partials: int64[_up_dpus]):
        @allo.work(mapping=[_up_dpus], args=[A, partials])
        def reduce(lA: int32[_up_n], lP: int64[_up_dpus]):
            (d,) = allo.get_wid()
            lP[d] = 0
            for i in range(_up_s):
                lP[d] += lA[d * _up_s + i]

    return red


def gemv_workload(m: int, k: int, dpus: int = 1, *, alpha: int | None = None):
    """Row-partitioned ``y = A x`` (MTV), or ``y = alpha A x`` (scaled GEMV)."""
    _up_dpus = dpus
    _up_m = m
    _up_k = k
    _up_alpha = alpha
    _up_r = _split(_up_m, _up_dpus, "M")

    if _up_alpha is not None:

        @_df_region()
        def gemv(A: int32[_up_m, _up_k], x: int32[_up_k], y: int32[_up_m]):
            @allo.work(mapping=[_up_dpus], args=[A, x, y])
            def mv(lA: int32[_up_m, _up_k], lx: int32[_up_k], ly: int32[_up_m]):
                (d,) = allo.get_wid()
                for i in range(_up_r):
                    acc: int32 = 0
                    for j in range(_up_k):
                        acc += lA[d * _up_r + i, j] * lx[j]
                    ly[d * _up_r + i] = _up_alpha * acc

        return gemv

    @_df_region()
    def mtv(A: int32[_up_m, _up_k], x: int32[_up_k], y: int32[_up_m]):
        @allo.work(mapping=[_up_dpus], args=[A, x, y])
        def mv(lA: int32[_up_m, _up_k], lx: int32[_up_k], ly: int32[_up_m]):
            (d,) = allo.get_wid()
            for i in range(_up_r):
                ly[d * _up_r + i] = 0
                for j in range(_up_k):
                    ly[d * _up_r + i] += lA[d * _up_r + i, j] * lx[j]

    return mtv


def ttv_workload(m: int, n: int, k: int):
    """``C[m, n] = sum_k A[m, n, k] x[k]``, flattened to M*N MV rows."""
    _up_n = n
    _up_m = m
    _up_k = k

    @_df_region()
    def ttv(A: int32[_up_m, _up_n, _up_k], x: int32[_up_k], C: int32[_up_m, _up_n]):
        @allo.work(mapping=[1], args=[A, x, C])
        def tv(lA: int32[_up_m, _up_n, _up_k], lx: int32[_up_k], lC: int32[_up_m, _up_n]):
            for a in range(_up_m):
                for b in range(_up_n):
                    lC[a, b] = 0
                    for c in range(_up_k):
                        lC[a, b] += lA[a, b, c] * lx[c]

    return ttv


def mmtv_workload(m: int, n: int, k: int):
    """``C[m, n] = sum_k A[m, n, k] B[m, k]``: one vector per batch."""
    _up_n = n
    _up_m = m
    _up_k = k

    @_df_region()
    def mmtv(A: int32[_up_m, _up_n, _up_k], B: int32[_up_m, _up_k], C: int32[_up_m, _up_n]):
        @allo.work(mapping=[1], args=[A, B, C])
        def btv(lA: int32[_up_m, _up_n, _up_k], lB: int32[_up_m, _up_k], lC: int32[_up_m, _up_n]):
            for a in range(_up_m):
                for b in range(_up_n):
                    lC[a, b] = 0
                    for c in range(_up_k):
                        lC[a, b] += lA[a, b, c] * lB[a, c]

    return mmtv


def gemm_workload(rows: int, reduction: int, columns: int):
    """``C = A B`` as one kernel named ``mm``."""
    _up_rows = rows
    _up_red = reduction
    _up_cols = columns

    @_df_region()
    def gemm(A: int32[_up_rows, _up_red], B: int32[_up_red, _up_cols], C: int32[_up_rows, _up_cols]):
        @allo.work(mapping=[1], args=[A, B, C])
        def mm(
            lA: int32[_up_rows, _up_red],
            lB: int32[_up_red, _up_cols],
            lC: int32[_up_rows, _up_cols],
        ):
            for i in range(_up_rows):
                for j in range(_up_cols):
                    lC[i, j] = 0
                    for c in range(_up_red):
                        lC[i, j] += lA[i, c] * lB[c, j]

    return gemm


# ----------------------------- irregular ------------------------------ #
# Spec 004. Module constants carry distinctive names for the same reason the
# factories rebind theirs (see above).

_HIST_N, _HIST_BINS, _HIST_DEPTH = 12288, 128, 12
_SEL_N = 12288
_KM_POINTS, _KM_DIM, _KM_K = 120, 8, 4
_GRAD_S, _GRAD_F, _GRAD_W = 120, 8, 9


@_df_region()
def hist_region(A: int32[_HIST_N], H: int32[_HIST_BINS]):
    @allo.work(mapping=[1], args=[A, H])
    def histogram(lA: int32[_HIST_N], lH: int32[_HIST_BINS]):
        for j in range(_HIST_BINS):
            lH[j] = 0
        for i in range(_HIST_N):
            b: int32 = (lA[i] * _HIST_BINS) >> _HIST_DEPTH
            if b >= 0 and b < _HIST_BINS:
                lH[b] += 1


@_df_region()
def sel_region(A: int32[_SEL_N], out: int32[_SEL_N], count: int32[1]):
    @allo.work(mapping=[1], args=[A, out, count])
    def select_odd(lA: int32[_SEL_N], lout: int32[_SEL_N], lcount: int32[1]):
        lcount[0] = 0
        for i in range(_SEL_N):
            if (lA[i] & 1) != 0:
                lout[lcount[0]] = lA[i]
                lcount[0] += 1


@_df_region()
def kmeans_region(
    P: int32[_KM_POINTS, _KM_DIM],
    C: int32[_KM_K, _KM_DIM],
    D: int64[_KM_POINTS, _KM_K],
    C_new: int32[_KM_K, _KM_DIM],
    counts: int32[_KM_K],
):
    @allo.work(mapping=[1], args=[P, C, D])
    def distances(
        lP: int32[_KM_POINTS, _KM_DIM],
        lC: int32[_KM_K, _KM_DIM],
        lD: int64[_KM_POINTS, _KM_K],
    ):
        for p in range(_KM_POINTS):
            for c in range(_KM_K):
                lD[p, c] = 0
                for d in range(_KM_DIM):
                    lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

    @allo.work(mapping=[1], args=[P, D, C_new, counts])
    def update(
        uP: int32[_KM_POINTS, _KM_DIM],
        uD: int64[_KM_POINTS, _KM_K],
        uC: int32[_KM_K, _KM_DIM],
        ucounts: int32[_KM_K],
    ):
        sums: int32[_KM_K, _KM_DIM] = 0
        for c0 in range(_KM_K):
            ucounts[c0] = 0
        for p in range(_KM_POINTS):
            best: int64 = uD[p, 0]
            idx: int32 = 0
            for c in range(_KM_K):
                idx = c if uD[p, c] < best else idx
                best = uD[p, c] if uD[p, c] < best else best
            ucounts[idx] += 1
            for d in range(_KM_DIM):
                sums[idx, d] += uP[p, d]
        # The campaign's signed round-to-closest division; empty clusters
        # get 0. A host group runs this from its own MLIR (spec 004 D2).
        for c1 in range(_KM_K):
            for d1 in range(_KM_DIM):
                if ucounts[c1] == 0:
                    uC[c1, d1] = 0
                else:
                    num: int32 = sums[c1, d1]
                    den: int32 = ucounts[c1]
                    adj: int32 = num + den // 2
                    if num < 0:
                        adj = num - den // 2
                    mag: int32 = (adj if adj >= 0 else -adj) // den
                    uC[c1, d1] = -mag if adj < 0 else mag


@allo.host_program(kmeans_region)
def kmeans_host(P, C, D, C_new, counts):
    allo.launch("distances", P, C, D)
    allo.launch("update", P, D, C_new, counts)


@_df_region()
def linear_reg_region(S: int32[_GRAD_S, _GRAD_W], G: int64[_GRAD_F]):
    @allo.work(mapping=[1], args=[S, G])
    def gradient(lS: int32[_GRAD_S, _GRAD_W], lG: int64[_GRAD_F]):
        for f in range(_GRAD_F):
            lG[f] = 0
        for s in range(_GRAD_S):
            for f in range(_GRAD_F):
                # An int64 temporary keeps the shift signed in the portable-C
                # oracle (backend/c.py stores >64-bit temporaries unsigned);
                # M3 forwarding folds it into the GRAD_LINEAR match.
                t: int64 = (lS[s, f] * lS[s, _GRAD_F]) * -32
                lG[f] += t >> 8


@_df_region()
def logistic_reg_region(S: int32[_GRAD_S, _GRAD_W], G: int64[_GRAD_F]):
    @allo.work(mapping=[1], args=[S, G])
    def gradient(lS: int32[_GRAD_S, _GRAD_W], lG: int64[_GRAD_F]):
        for f in range(_GRAD_F):
            lG[f] = 0
        for s in range(_GRAD_S):
            for f in range(_GRAD_F):
                lG[f] += lS[s, f] * (1 - 2 * lS[s, _GRAD_F])


IRREGULAR_CASES = ("hist", "sel", "kmeans", "linear_reg", "logistic_reg")


def _repeated_gemm(region, repeats: int):
    @allo.host_program(region)
    def host(A, B, C):
        for _ in range(repeats):
            allo.launch("mm", A, B, C)

    return host


def gemv_host_program(region):
    """Scatter the matrix, broadcast the vector, launch, gather the output."""
    hx = allo.host_xfer

    @allo.host_program(region)
    def host(A, x, y):
        hx.scatter(A, hx.mram)
        hx.broadcast(x, hx.mram)
        allo.launch("mv", A, x, y)
        hx.gather(y, hx.mram)

    return host


def reduction_host_program(region):
    """Scatter the input, launch, gather one int64 partial per DPU."""
    hx = allo.host_xfer

    @allo.host_program(region)
    def host(A, partials):
        hx.scatter(A, hx.mram)
        allo.launch("reduce", A, partials)
        hx.gather(partials, hx.mram)

    return host


def build_case(case_id: str, dpus: int = 1):
    """``(workload, host_program_or_None)`` for one paper case."""
    irregular = {
        "hist": (hist_region, None),
        "sel": (sel_region, None),
        "kmeans": (kmeans_region, kmeans_host),
        "linear_reg": (linear_reg_region, None),
        "logistic_reg": (logistic_reg_region, None),
    }
    if case_id in irregular:
        if dpus != 1:
            raise ValueError(f"{case_id} is a one-DPU paper case")
        return irregular[case_id]
    if case_id not in REGULAR_CASES:
        raise ValueError(f"no matcher-path source for UPMEM case {case_id!r}")
    if dpus != 1 and case_id not in ("va", "geva", "red", "mtv", "gemv"):
        raise ValueError(f"{case_id} is a one-DPU paper case")
    if case_id == "va":
        return pointwise_workload(12288, dpus), None
    if case_id == "geva":
        return pointwise_workload(12288, dpus, axpby=True), None
    if case_id == "red":
        return reduction_workload(12288, dpus), None
    if case_id == "mtv":
        return gemv_workload(96, 128, dpus), None
    if case_id == "gemv":
        return gemv_workload(96, 128, dpus, alpha=GEMV_ALPHA), None
    if case_id == "ttv":
        return ttv_workload(12, 16, 32), None
    if case_id == "mmtv":
        return mmtv_workload(12, 16, 32), None
    if case_id == "conv":
        return gemm_workload(192, 16, 32), None
    region = gemm_workload(12, 64, 128)
    repeats = int(case_id[0])
    return region, (_repeated_gemm(region, repeats) if repeats > 1 else None)


def artifacts_root() -> Path:
    override = os.environ.get("TENON_ARTIFACTS")
    return Path(override) if override else Path.home() / "shared" / "tenon-artifacts"


_ARGUMENTS = {
    "va": (("A", "a.i32.bin", (12288,)), ("B", "b.i32.bin", (12288,))),
    "geva": (("A", "a.i32.bin", (12288,)), ("B", "b.i32.bin", (12288,))),
    "red": (("A", "a.i32.bin", (12288,)),),
    "mtv": (("A", "a.i32.bin", (96, 128)), ("x", "x.i32.bin", (128,))),
    "gemv": (("A", "a.i32.bin", (96, 128)), ("x", "x.i32.bin", (128,))),
    "ttv": (("A", "a.i32.bin", (12, 16, 32)), ("x", "x.i32.bin", (32,))),
    "mmtv": (("A", "a.i32.bin", (12, 16, 32)), ("B", "b.i32.bin", (12, 32))),
    "1mm": (("A", "a.i32.bin", (12, 64)), ("B", "b.i32.bin", (64, 128))),
    "2mm": (("A", "a.i32.bin", (12, 64)), ("B", "b.i32.bin", (64, 128))),
    "3mm": (("A", "a.i32.bin", (12, 64)), ("B", "b.i32.bin", (64, 128))),
    "conv": (
        ("A", "patches.i32.bin", (192, 16)),
        ("B", "filters.i32.bin", (16, 32)),
    ),
    "hist": (("A", "input.i32.bin", (_HIST_N,)),),
    "sel": (("A", "input.i32.bin", (_SEL_N,)),),
    "kmeans": (
        ("P", "points.i32.bin", (_KM_POINTS, _KM_DIM)),
        ("C", "initial_centroids.i32.bin", (_KM_K, _KM_DIM)),
    ),
    "linear_reg": (("S", "samples.i32.bin", (_GRAD_S, _GRAD_W)),),
    "logistic_reg": (("S", "samples.i32.bin", (_GRAD_S, _GRAD_W)),),
}
_OUTPUTS = {
    "va": ("C", (12288,), np.int32),
    "geva": ("C", (12288,), np.int32),
    "red": ("partials", (1,), np.int64),
    "mtv": ("y", (96,), np.int32),
    "gemv": ("y", (96,), np.int32),
    "ttv": ("C", (12, 16), np.int32),
    "mmtv": ("C", (12, 16), np.int32),
    "1mm": ("C", (12, 128), np.int32),
    "2mm": ("C", (12, 128), np.int32),
    "3mm": ("C", (12, 128), np.int32),
    "conv": ("C", (192, 32), np.int32),
    "hist": ("H", (_HIST_BINS,), np.int32),
    "sel": ("out", (_SEL_N,), np.int32),
    "linear_reg": ("G", (_GRAD_F,), np.int64),
    "logistic_reg": ("G", (_GRAD_F,), np.int64),
}
# Extra zeroed outputs, in signature order after the inputs.
_EXTRA_OUTPUTS = {
    "sel": (("count", (1,), np.int32),),
    "kmeans": (
        ("D", (_KM_POINTS, _KM_K), np.int64),
        ("C_new", (_KM_K, _KM_DIM), np.int32),
        ("counts", (_KM_K,), np.int32),
    ),
}


def canonical_inputs(case_id: str, root: Path | None = None) -> dict:
    """The archived canonical inputs, plus a zeroed output, by parameter name."""
    base = (Path(root) if root is not None else artifacts_root()) / "upmem" / "inputs" / case_id
    metadata = json.loads((base / "metadata.json").read_text())
    inputs = {}
    for name, filename, shape in _ARGUMENTS[case_id]:
        if filename not in metadata["files"]:
            raise FileNotFoundError(f"{case_id}: {filename} missing from metadata.json")
        inputs[name] = np.fromfile(base / filename, dtype="<i4").astype(np.int32).reshape(shape)
    outputs = ((_OUTPUTS[case_id],) if case_id in _OUTPUTS else ()) + _EXTRA_OUTPUTS.get(case_id, ())
    for name, shape, dtype in outputs:
        inputs[name] = np.zeros(shape, dtype=dtype)
    return inputs


def numpy_reference(case_id: str, inputs: dict) -> np.ndarray:
    """The case's output computed in NumPy with int32 wraparound semantics."""
    def i32(value):
        return np.asarray(value, dtype=np.int64).astype(np.int32)

    if case_id == "hist":
        bins = (inputs["A"].astype(np.int64) * _HIST_BINS) >> _HIST_DEPTH
        valid = bins[(bins >= 0) & (bins < _HIST_BINS)]
        return np.bincount(valid, minlength=_HIST_BINS).astype(np.int32)
    if case_id == "sel":
        values = inputs["A"]
        kept = values[(values & 1) != 0]
        return {"out": kept.astype(np.int32), "count": np.array([kept.size], dtype=np.int32)}
    if case_id == "kmeans":
        points = inputs["P"].astype(np.int64)
        centroids = inputs["C"].astype(np.int64)
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        chosen = distances.argmin(axis=1)
        counts = np.bincount(chosen, minlength=_KM_K)
        result = np.zeros((_KM_K, _KM_DIM), dtype=np.int64)
        for cluster in range(_KM_K):
            if counts[cluster] == 0:
                continue
            for feature in range(_KM_DIM):
                numerator = int(points[chosen == cluster, feature].sum())
                denominator = int(counts[cluster])
                adjusted = (
                    numerator - denominator // 2
                    if numerator < 0
                    else numerator + denominator // 2
                )
                magnitude = abs(adjusted) // denominator
                result[cluster, feature] = -magnitude if adjusted < 0 else magnitude
        return {
            "D": distances,
            "C_new": result.astype(np.int32),
            "counts": counts.astype(np.int32),
        }
    if case_id in ("linear_reg", "logistic_reg"):
        samples = inputs["S"].astype(object)
        x, label = samples[:, :_GRAD_F], samples[:, _GRAD_F : _GRAD_F + 1]
        if case_id == "linear_reg":
            terms = (x * label * -32) // 256
        else:
            terms = x * (1 - 2 * label)
        return np.array(terms.sum(axis=0), dtype=np.int64)

    if case_id == "va":
        return i32(inputs["A"].astype(np.int64) + inputs["B"])
    if case_id == "geva":
        return i32(GEVA_ALPHA * inputs["A"].astype(np.int64) + GEVA_BETA * inputs["B"].astype(np.int64))
    if case_id == "red":
        return np.array([inputs["A"].astype(np.int64).sum()], dtype=np.int64)
    if case_id == "mtv":
        return i32(inputs["A"].astype(np.int64) @ inputs["x"].astype(np.int64))
    if case_id == "gemv":
        return i32(GEMV_ALPHA * (inputs["A"].astype(np.int64) @ inputs["x"].astype(np.int64)))
    if case_id == "ttv":
        return i32(inputs["A"].astype(np.int64) @ inputs["x"].astype(np.int64))
    if case_id == "mmtv":
        return i32(np.einsum("mnk,mk->mn", inputs["A"].astype(np.int64), inputs["B"].astype(np.int64)))
    return i32(inputs["A"].astype(np.int64) @ inputs["B"].astype(np.int64))


def check_outputs(case_id: str, outputs: dict, reference) -> None:
    """Assert the run's outputs (the caller's arrays after the run) equal
    ``numpy_reference`` computed on the inputs before it."""
    if case_id == "sel":
        count = int(reference["count"][0])
        assert int(outputs["count"][0]) == count, (outputs["count"], count)
        np.testing.assert_array_equal(outputs["out"][:count], reference["out"])
        return
    if isinstance(reference, dict):
        for name, value in reference.items():
            np.testing.assert_array_equal(outputs[name], value, err_msg=name)
        return
    name = _OUTPUTS[case_id][0]
    np.testing.assert_array_equal(
        outputs[name], np.asarray(reference).reshape(outputs[name].shape)
    )


__all__ = [
    "IRREGULAR_CASES",
    "check_outputs",
    "REGULAR_CASES",
    "UPMEM_CASES",
    "artifacts_root",
    "build_case",
    "canonical_inputs",
    "gemm_workload",
    "gemv_host_program",
    "gemv_workload",
    "mmtv_workload",
    "numpy_reference",
    "pointwise_workload",
    "reduction_host_program",
    "reduction_workload",
    "ttv_workload",
]
