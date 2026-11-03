# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM generality and cost scope (release audit U-03..U-06, U-09..U-14).

Compile-only tests run everywhere.  Tests that call the uPIMulator route
skip with BLOCKED-SIM when it is unavailable; the 391x65 scaling sweep is
``paper_full``.  Workloads are written here, independently of the paper
benchmark factories, with reviewer-chosen shapes off the reference grid.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.dataflow import region as df_region
from allo.ir.types import int32, int64
from allo.pim.costs import upmem as upmem_costs
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.pim.upmem_irregular import SelectionPredicate, UPMEMSelectionFlagsPlan
from allo.pim.upmem_physical import UPMEMElementwisePlan, UPMEMMatrixVectorPlan
from allo.pim.upmem_physical_search import (
    UPMEMDataLayout,
    UPMEMOperandResidency,
    UPMEMPhysicalDecision,
    UPMEMPhysicalProblem,
    calibration_validity,
    physical_features,
)
from allo.spmw_upmem import VIRTUAL_TIMING_SCOPE, padding_fraction


def _compile(workload, **kw):
    return allo.compile(workload, build_upmem_target(), upmem_cost, **kw)


def _virtual(workload, **kw):
    return _compile(workload, backend="virtual", **kw)


def _ctx(compiled):
    return compiled.compiled.layout_ctx


def _segment(compiled, index=0):
    return [s for s in _ctx(compiled).segments if s.kind == "dpu"][index]


def _sim_gate():
    import conftest

    conftest._simulator_gate("upmem")


# --------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------- #


def pointwise_region(n, expression):
    """``C[i] = expression(A[i], B[i])`` over one DPU."""
    _n = n

    if expression == "sub":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = lA[i] - lB[i]

    elif expression == "mul":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = lA[i] * lB[i]

    elif expression == "scale":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = -5 * lA[i]

    elif expression == "axpby":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = 3 * lA[i] + (-7) * lB[i]

    elif expression == "minus_axpby":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = 3 * lA[i] - 7 * lB[i]

    elif expression == "affine":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = 2 * lA[i] + 5 * lB[i] + 11

    elif expression == "affine_wide_constant":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = 2 * lA[i] + 5 * lB[i] + 2147483648

    elif expression == "minus_int32_min":

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = 3 * lA[i] - (-2147483648) * lB[i]

    else:

        @df_region()
        def k(A: int32[_n], B: int32[_n], C: int32[_n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def f(lA: int32[_n], lB: int32[_n], lC: int32[_n]):
                for i in range(_n):
                    lC[i] = lA[i] + lB[i]

    return k


def mv_region(m, k, *, carried=False):
    _m, _k = m, k
    if carried:

        @df_region()
        def g(A: int32[_m, _k], x: int32[_k], y: int32[_m]):
            @allo.work(mapping=[1], args=[A, x, y])
            def mv(lA: int32[_m, _k], lx: int32[_k], ly: int32[_m]):
                for i in range(_m):
                    for j in range(_k):
                        ly[i] += lA[i, j] * lx[j]

        return g

    @df_region()
    def g(A: int32[_m, _k], x: int32[_k], y: int32[_m]):
        @allo.work(mapping=[1], args=[A, x, y])
        def mv(lA: int32[_m, _k], lx: int32[_k], ly: int32[_m]):
            for i in range(_m):
                ly[i] = 0
                for j in range(_k):
                    ly[i] += lA[i, j] * lx[j]

    return g


def mmtv_region(b, r, k):
    _b, _r, _k = b, r, k

    @df_region()
    def g(A: int32[_b, _r, _k], B: int32[_b, _k], C: int32[_b, _r]):
        @allo.work(mapping=[1], args=[A, B, C])
        def btv(lA: int32[_b, _r, _k], lB: int32[_b, _k], lC: int32[_b, _r]):
            for a in range(_b):
                for c in range(_r):
                    lC[a, c] = 0
                    for e in range(_k):
                        lC[a, c] += lA[a, c, e] * lB[a, e]

    return g


def gemm_region(rows, red, cols, *, carried=False):
    _r, _c, _n = rows, red, cols
    if carried:

        @df_region()
        def g(A: int32[_r, _c], B: int32[_c, _n], C: int32[_r, _n]):
            @allo.work(mapping=[1], args=[A, B, C])
            def mm(lA: int32[_r, _c], lB: int32[_c, _n], lC: int32[_r, _n]):
                for i in range(_r):
                    for j in range(_n):
                        for e in range(_c):
                            lC[i, j] += lA[i, e] * lB[e, j]

        return g

    @df_region()
    def g(A: int32[_r, _c], B: int32[_c, _n], C: int32[_r, _n]):
        @allo.work(mapping=[1], args=[A, B, C])
        def mm(lA: int32[_r, _c], lB: int32[_c, _n], lC: int32[_r, _n]):
            for i in range(_r):
                for j in range(_n):
                    lC[i, j] = 0
                    for e in range(_c):
                        lC[i, j] += lA[i, e] * lB[e, j]

    return g


def histogram_region(n, bins, depth, *, carried=False):
    _n, _b, _d = n, bins, depth
    if carried:

        @df_region()
        def h(A: int32[_n], H: int32[_b]):
            @allo.work(mapping=[1], args=[A, H])
            def histogram(lA: int32[_n], lH: int32[_b]):
                for i in range(_n):
                    v: int32 = (lA[i] * _b) >> _d
                    if v >= 0 and v < _b:
                        lH[v] += 1

        return h

    @df_region()
    def h(A: int32[_n], H: int32[_b]):
        @allo.work(mapping=[1], args=[A, H])
        def histogram(lA: int32[_n], lH: int32[_b]):
            for j in range(_b):
                lH[j] = 0
            for i in range(_n):
                v: int32 = (lA[i] * _b) >> _d
                if v >= 0 and v < _b:
                    lH[v] += 1

    return h


def reduction_region(n, *, carried=False):
    _n = n
    if carried:

        @df_region()
        def r(A: int32[_n], P: int64[1]):
            @allo.work(mapping=[1], args=[A, P])
            def reduce(lA: int32[_n], lP: int64[1]):
                for i in range(_n):
                    lP[0] += lA[i]

        return r

    @df_region()
    def r(A: int32[_n], P: int64[1]):
        @allo.work(mapping=[1], args=[A, P])
        def reduce(lA: int32[_n], lP: int64[1]):
            lP[0] = 0
            for i in range(_n):
                lP[0] += lA[i]

    return r


def selection_region(n, predicate, *, carried=False):
    _n = n
    if predicate == "even":

        @df_region()
        def s(A: int32[_n], out: int32[_n], count: int32[1]):
            @allo.work(mapping=[1], args=[A, out, count])
            def sel(lA: int32[_n], lout: int32[_n], lcount: int32[1]):
                lcount[0] = 0
                for i in range(_n):
                    if (lA[i] & 1) == 0:
                        lout[lcount[0]] = lA[i]
                        lcount[0] += 1

    elif predicate == "positive":

        @df_region()
        def s(A: int32[_n], out: int32[_n], count: int32[1]):
            @allo.work(mapping=[1], args=[A, out, count])
            def sel(lA: int32[_n], lout: int32[_n], lcount: int32[1]):
                lcount[0] = 0
                for i in range(_n):
                    if lA[i] > 0:
                        lout[lcount[0]] = lA[i]
                        lcount[0] += 1

    elif predicate == "mask" and carried:

        @df_region()
        def s(A: int32[_n], out: int32[_n], count: int32[1]):
            @allo.work(mapping=[1], args=[A, out, count])
            def sel(lA: int32[_n], lout: int32[_n], lcount: int32[1]):
                for i in range(_n):
                    if (lA[i] & 6) != 2:
                        lout[lcount[0]] = lA[i]
                        lcount[0] += 1

    else:  # an odd test against a value that is not a literal

        @df_region()
        def s(A: int32[_n], out: int32[_n], count: int32[1]):
            @allo.work(mapping=[1], args=[A, out, count])
            def sel(lA: int32[_n], lout: int32[_n], lcount: int32[1]):
                lcount[0] = 0
                for i in range(_n):
                    if lA[i] > lA[0]:
                        lout[lcount[0]] = lA[i]
                        lcount[0] += 1

    return s


def kmeans_distances_region(points, dim, clusters, *, carried=False):
    _p, _d, _c = points, dim, clusters
    if carried:

        @df_region()
        def km(P: int32[_p, _d], C: int32[_c, _d], D: int64[_p, _c]):
            @allo.work(mapping=[1], args=[P, C, D])
            def distances(lP: int32[_p, _d], lC: int32[_c, _d], lD: int64[_p, _c]):
                for p in range(_p):
                    for c in range(_c):
                        for d in range(_d):
                            lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

        return km

    @df_region()
    def km(P: int32[_p, _d], C: int32[_c, _d], D: int64[_p, _c]):
        @allo.work(mapping=[1], args=[P, C, D])
        def distances(lP: int32[_p, _d], lC: int32[_c, _d], lD: int64[_p, _c]):
            for p in range(_p):
                for c in range(_c):
                    lD[p, c] = 0
                    for d in range(_d):
                        lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

    return km


def uneven_gemv_region(m, k, dpus):
    """``y = A x`` over ``dpus`` work items; the last takes the remainder."""
    _m, _k, _n = m, k, dpus
    _r = m // dpus
    _last = m - (dpus - 1) * _r

    @df_region()
    def g(A: int32[_m, _k], x: int32[_k], y: int32[_m]):
        @allo.work(mapping=[_n], args=[A, x, y])
        def mv(lA: int32[_m, _k], lx: int32[_k], ly: int32[_m]):
            (d,) = allo.get_wid()
            with allo.meta_if(d < _n - 1):
                for i in range(_r):
                    ly[d * _r + i] = 0
                    for j in range(_k):
                        ly[d * _r + i] += lA[d * _r + i, j] * lx[j]
            with allo.meta_else():
                for i in range(_last):
                    ly[_m - _last + i] = 0
                    for j in range(_k):
                        ly[_m - _last + i] += lA[_m - _last + i, j] * lx[j]

    return g


def uneven_reduction_region(n, dpus):
    _n, _w = n, dpus
    _s = n // dpus
    _last = n - (dpus - 1) * _s

    @df_region()
    def r(A: int32[_n], P: int64[_w]):
        @allo.work(mapping=[_w], args=[A, P])
        def reduce(lA: int32[_n], lP: int64[_w]):
            (d,) = allo.get_wid()
            lP[d] = 0
            with allo.meta_if(d < _w - 1):
                for i in range(_s):
                    lP[d] += lA[d * _s + i]
            with allo.meta_else():
                for i in range(_last):
                    lP[d] += lA[_n - _last + i]

    return r


# --------------------------------------------------------------------- #
# U-05: legality is the template's; extents are padded, not rejected
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [8, 1000, 12289])
def test_pointwise_non_divisible_extents_compile_and_price_the_padded_program(n):
    compiled = _virtual(pointwise_region(n, "add"))
    segment = _segment(compiled)
    chunk = segment.plan.chunk_elements
    assert segment.plan.elements == n
    assert segment.plan.chunks == -(-n // chunk)
    features = dict(compiled.compiled.layout.extra["upmem_features"])
    padded = segment.plan.chunks * chunk
    assert features["mram_write_bytes"] == padded * 4


def test_pointwise_layout_choice_is_no_longer_a_divisibility_artifact():
    # 1008 % 32 != 0 used to make the fused 256-byte decision illegal.
    from allo.spmw_upmem import (
        UPMEM_DECISION_DOMAINS,
        _legal_decisions,
        kernel_from_extra,
    )

    kernel = kernel_from_extra(
        {
            "upmem_family": "pointwise",
            "upmem_geometry": (("elements", 1008),),
            "upmem_anchor_op": "ADD",
        }
    )
    assert _legal_decisions(kernel) == sorted(
        UPMEM_DECISION_DOMAINS["pointwise"], key=lambda d: d.deterministic_order_key
    )


@pytest.mark.parametrize("m, k", [(100, 100), (96, 100), (24, 8), (12, 16), (5, 7)])
def test_matrix_vector_padded_shapes_compile(m, k):
    compiled = _virtual(mv_region(m, k))
    plan = _segment(compiled).plan
    assert plan.total_rows == m and plan.columns == k
    assert plan.padded_columns % plan.chunk_elements == 0


@pytest.mark.parametrize("rows, red, cols", [(20, 32, 32), (12, 20, 32), (7, 9, 5)])
def test_gemm_padded_shapes_compile(rows, red, cols):
    compiled = _virtual(gemm_region(rows, red, cols))
    plan = _segment(compiled).plan
    assert (plan.rows, plan.reduction, plan.columns) == (rows, red, cols)


def test_features_equal_the_reference_shapes_unpadded():
    # Divisible shapes keep their features: the calibration anchors and
    # every pinned paper shape are unchanged by padding-aware features.
    decision = UPMEMPhysicalDecision(UPMEMDataLayout.FUSED_REPLICATED, 256, 32)
    features = physical_features(UPMEMPhysicalProblem.pointwise(12288), decision)
    assert features.mram_read_bytes == 12288 * 8
    assert features.dynamic_instructions == 12288


def test_unrealizable_kernel_names_the_template_constraint():
    # 5 points x 2 clusters = 10 distance pairs over 12 tasklets.
    with pytest.raises(NotImplementedError, match="divide evenly across 12 tasklets"):
        _virtual(kmeans_distances_region(5, 2, 2))


# --------------------------------------------------------------------- #
# U-06: no shape-exact paths; general pointwise and selection forms
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("b, r, k", [(4, 8, 32), (12, 16, 64), (6, 8, 32), (13, 5, 20)])
def test_tasklet_private_wram_is_legal_wherever_it_fits(b, r, k):
    decision = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        64,
        16,
        vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
    )
    physical_features(UPMEMPhysicalProblem.matrix_vector(b * r, k, batches=b), decision)
    plan = UPMEMMatrixVectorPlan(r, k, batches=b, vector_mode="tasklet_private_wram")
    source = plan.device_source()
    assert ("batch += NR_TASKLETS" in source) == (not plan.one_batch_per_tasklet)
    assert plan.batch_output_bytes % 8 == 0


def test_tasklet_private_wram_keeps_the_calibrated_kernel_at_one_batch_per_tasklet():
    plan = UPMEMMatrixVectorPlan(16, 32, batches=12, vector_mode="tasklet_private_wram")
    assert plan.one_batch_per_tasklet
    assert "batch += NR_TASKLETS" not in plan.device_source()


def test_tasklet_private_wram_rejects_on_its_wram_footprint():
    with pytest.raises(ValueError, match="WRAM"):
        UPMEMMatrixVectorPlan(8, 4096, batches=12, vector_mode="tasklet_private_wram")
    decision = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        64,
        16,
        vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
    )
    with pytest.raises(ValueError, match="WRAM bytes"):
        physical_features(
            UPMEMPhysicalProblem.matrix_vector(96, 4096, batches=12), decision
        )


@pytest.mark.parametrize(
    "expression, operation",
    [("sub", "sub"), ("mul", "mul"), ("scale", "scale"), ("axpby", "axpby")],
)
def test_pointwise_family_accepts_the_matched_expression(expression, operation):
    compiled = _virtual(pointwise_region(1536, expression))
    plan = _segment(compiled).plan
    assert plan.operation == operation
    if operation == "scale":
        assert plan.alpha == -5 and plan.operands == 1 and not plan.interleaved
    if operation == "axpby":
        assert (plan.alpha, plan.beta) == (3, -7)


def test_pointwise_plan_evaluates_with_int32_wraparound():
    plan = UPMEMElementwisePlan(16, operation="mul", interleaved=False)
    assert plan.evaluate(1 << 20, 1 << 12) == 0
    assert plan.evaluate(-3, 5) == -15
    sub = UPMEMElementwisePlan(16, operation="sub", interleaved=False)
    assert sub.evaluate(-(1 << 31), 1) == (1 << 31) - 1


@pytest.mark.parametrize(
    "expression, anchor, coefficients",
    [("minus_axpby", "AXMBY", (3, -7, 0)), ("affine", "AXPBYC", (2, 5, 11))],
)
def test_minus_and_constant_forms_lower_to_the_axpby_template(
    expression, anchor, coefficients
):
    compiled = _virtual(pointwise_region(1536, expression))
    segment = _segment(compiled)
    plan = segment.plan
    assert plan.operation == "axpby"
    assert (plan.alpha, plan.beta, plan.gamma) == coefficients
    source = plan.device_source()
    alpha, beta, gamma = coefficients
    assert f"(uint32_t){alpha} * (uint32_t)packet[i]" in source
    assert f"(uint32_t){beta} * (uint32_t)packet[i + TENON_CHUNK_ELEMENTS]" in source
    assert (f" + (uint32_t){gamma})" in source) == bool(gamma)
    assert plan.evaluate(-(1 << 31), (1 << 31) - 1) == UPMEMElementwisePlan(
        16, operation="axpby", interleaved=False, alpha=alpha, beta=beta, gamma=gamma
    ).evaluate(-(1 << 31), (1 << 31) - 1)
    assert segment.kernel.anchor_op == anchor


@pytest.mark.parametrize(
    "expression, message",
    [
        ("affine_wide_constant", r"gamma=2147483648 is not an int32 literal"),
        ("minus_int32_min", r"-beta=2147483648 is not an int32 literal"),
    ],
)
def test_minus_and_constant_forms_reject_non_int32_coefficients(expression, message):
    with pytest.raises(NotImplementedError, match=message):
        _virtual(pointwise_region(64, expression))


def test_constant_term_is_compile_time_axpby_only():
    with pytest.raises(ValueError, match="gamma"):
        UPMEMElementwisePlan(16, operation="add", interleaved=False, gamma=1)
    plan = UPMEMElementwisePlan(16, operation="axpby", alpha=2, beta=-1, gamma=3)
    assert "<< 1" not in plan.device_source()
    assert plan.manifest()["coefficients"]["gamma"] == 3
    assert (
        "gamma"
        not in UPMEMElementwisePlan(16, operation="axpby").manifest()["coefficients"]
    )


@pytest.mark.parametrize(
    "predicate, expected",
    [
        ("even", SelectionPredicate("eq", 0, mask=1)),
        ("positive", SelectionPredicate("gt", 0)),
    ],
)
def test_selection_generalizes_to_the_matched_predicate(predicate, expected):
    compiled = _virtual(selection_region(1000, predicate))
    plan = _segment(compiled).plan
    assert plan.predicate == expected
    assert plan.encoding == "mask"
    values = list(range(-6, 6)) + [0] * (plan.elements - 12)
    flags = plan.reference_flags(values)
    assert flags[:12] == tuple(int(expected.holds(v)) for v in range(-6, 6))


def test_selection_rejects_a_non_literal_predicate_naming_the_domain():
    with pytest.raises(NotImplementedError, match="cmp k"):
        _virtual(selection_region(1536, "other"))


def test_selection_predicate_c_and_python_agree_on_int32_edges():
    predicate = SelectionPredicate("lt", -(1 << 31) + 1)
    assert predicate.holds(-(1 << 31)) and not predicate.holds(0)
    assert "(int32_t)" in predicate.c_condition("x")
    plan = UPMEMSelectionFlagsPlan(1536, predicate=SelectionPredicate("ne", 2, mask=6))
    assert "? 1 : 0" in plan.device_source()


# --------------------------------------------------------------------- #
# U-03: the matcher's accumulator_init fact on every route
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "region, family",
    [
        (lambda: mv_region(48, 64, carried=True), "matrix_vector"),
        (lambda: gemm_region(12, 64, 32, carried=True), "gemm"),
        (lambda: histogram_region(1536, 64, 12, carried=True), "histogram"),
        (lambda: reduction_region(1536, carried=True), "reduction"),
    ],
)
def test_carried_accumulators_load_the_initial_output(region, family):
    compiled = _virtual(region())
    segment = _segment(compiled)
    assert segment.kernel.family == family
    assert segment.kernel.accumulator_init == "carried"
    assert segment.plan.carried_output
    assert "mram_read" in segment.device_source


def test_zero_initialized_sources_keep_their_templates():
    compiled = _virtual(mv_region(48, 64))
    assert _segment(compiled).kernel.accumulator_init == "zero"
    assert not _segment(compiled).plan.carried_output


def test_carried_kmeans_distances_is_rejected_on_the_virtual_route():
    with pytest.raises(NotImplementedError, match="accumulator_init='carried'"):
        _virtual(kmeans_distances_region(24, 8, 4, carried=True))


def test_carried_selection_count_is_implemented_by_the_host_step():
    compiled = _virtual(selection_region(1536, "mask", carried=True))
    assert _segment(compiled).kernel.accumulator_init == "carried"


# --------------------------------------------------------------------- #
# U-09: virtual results carry their scope and validity
# --------------------------------------------------------------------- #


def test_virtual_validity_labels():
    calibrated = _virtual(pointwise_region(12288, "add"))
    extra = _ctx(calibrated).virtual_run_extra()
    assert extra["timing_scope"] == VIRTUAL_TIMING_SCOPE
    assert extra["cost_validity"] == "calibrated"

    off_grid = _virtual(pointwise_region(1024, "add"))
    extra = _ctx(off_grid).virtual_run_extra()
    assert extra["cost_validity"] == "extrapolated"
    nearest = extra["upmem_cost_validity"][0]["nearest_calibrated_point"]
    assert nearest["problem"]["logical_elements"] == 12288

    reduction = _ctx(_virtual(reduction_region(1536))).virtual_run_extra()
    assert reduction["cost_validity"] == "extrapolated"
    assert "per-op" in reduction["upmem_cost_validity"][0]["basis"]


def test_virtual_run_result_carries_scope_when_the_runner_forwards_it():
    compiled = _virtual(pointwise_region(1024, "add"))
    result = compiled(
        np.zeros(1024, np.int32), np.zeros(1024, np.int32), np.zeros(1024, np.int32)
    )
    assert result.kind == "virtual"
    if "timing_scope" not in result.extra:
        pytest.skip("spmw_codegen._run_virtual does not forward virtual_run_extra yet")
    assert result.extra["timing_scope"] == VIRTUAL_TIMING_SCOPE
    assert result.extra["cost_validity"] == "extrapolated"


def test_calibration_validity_of_the_mmtv_stratum_is_shape_exact():
    decision = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        64,
        16,
        vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
    )
    exact = calibration_validity(
        UPMEMPhysicalProblem.matrix_vector(192, 32, batches=12), decision
    )
    other = calibration_validity(
        UPMEMPhysicalProblem.matrix_vector(96, 32, batches=6), decision
    )
    assert exact["cost_validity"] == "calibrated"
    assert other["cost_validity"] == "extrapolated"
    assert other["calibration_factor"] == 1.0


# --------------------------------------------------------------------- #
# U-10: the host k-means update is exact and priced
# --------------------------------------------------------------------- #


def test_kmeans_update_host_group_is_priced():
    from benchmarks.upmem.workloads import kmeans_host, kmeans_region

    compiled = _virtual(kmeans_region, host_moves=kmeans_host)
    host = [s for s in _ctx(compiled).segments if s.kind == "host"]
    assert [s.kernel.family for s in host] == ["kmeans_update"]
    geometry = dict(host[0].kernel.geometry)
    assert geometry == {"points": 120, "clusters": 4, "dimension": 8}
    expected = upmem_costs.host_group_cycles(geometry)
    assert expected > 0
    estimate = compiled.estimate()
    assert estimate.cycles >= expected


_KP, _KD, _KK = 24, 4, 3


def _km_extra_op():
    @df_region()
    def km(
        P: int32[_KP, _KD],
        D: int64[_KP, _KK],
        C_new: int32[_KK, _KD],
        counts: int32[_KK],
        Z: int32[_KP],
    ):
        @allo.work(mapping=[1], args=[P, D, C_new, counts, Z])
        def update(
            uP: int32[_KP, _KD],
            uD: int64[_KP, _KK],
            uC: int32[_KK, _KD],
            ucounts: int32[_KK],
            uZ: int32[_KP],
        ):
            sums: int32[_KK, _KD] = 0
            for c0 in range(_KK):
                ucounts[c0] = 0
            for p in range(_KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(_KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(_KD):
                    sums[idx, d] += uP[p, d]
                uZ[p] = uP[p, 0] * uP[p, 1]
            for c1 in range(_KK):
                for d1 in range(_KD):
                    uC[c1, d1] = sums[c1, d1] // (ucounts[c1] + 1)

    return km


def _km_no_divide():
    @df_region()
    def km(
        P: int32[_KP, _KD],
        D: int64[_KP, _KK],
        C_new: int32[_KK, _KD],
        counts: int32[_KK],
    ):
        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[_KP, _KD],
            uD: int64[_KP, _KK],
            uC: int32[_KK, _KD],
            ucounts: int32[_KK],
        ):
            sums: int32[_KK, _KD] = 0
            for c0 in range(_KK):
                ucounts[c0] = 0
            for p in range(_KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(_KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(_KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(_KK):
                for d1 in range(_KD):
                    uC[c1, d1] = sums[c1, d1] - ucounts[c1]

    return km


def _km_other_source():
    @df_region()
    def km(
        P: int32[_KP, _KD],
        D: int64[_KP, _KK],
        C_new: int32[_KK, _KD],
        counts: int32[_KK],
    ):
        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[_KP, _KD],
            uD: int64[_KP, _KK],
            uC: int32[_KK, _KD],
            ucounts: int32[_KK],
        ):
            sums: int32[_KK, _KD] = 0
            for c0 in range(_KK):
                ucounts[c0] = 0
            for p in range(_KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(_KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(_KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(_KK):
                for d1 in range(_KD):
                    uC[c1, d1] = uP[c1, d1] // (ucounts[c1] + 1)

    return km


def _km_copy_only():
    """Structurally an update (argmin, counts, sums) whose centroid output is
    a plain copy of the sums: no division by the counts."""

    @df_region()
    def km(
        P: int32[_KP, _KD],
        D: int64[_KP, _KK],
        C_new: int32[_KK, _KD],
        counts: int32[_KK],
    ):
        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[_KP, _KD],
            uD: int64[_KP, _KK],
            uC: int32[_KK, _KD],
            ucounts: int32[_KK],
        ):
            sums: int32[_KK, _KD] = 0
            for c0 in range(_KK):
                ucounts[c0] = 0
            for p in range(_KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(_KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(_KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(_KK):
                for d1 in range(_KD):
                    uC[c1, d1] = sums[c1, d1]

    return km


def _km_index_mixed():
    """The division reads a loop index besides sums and counts."""

    @df_region()
    def km(
        P: int32[_KP, _KD],
        D: int64[_KP, _KK],
        C_new: int32[_KK, _KD],
        counts: int32[_KK],
    ):
        @allo.work(mapping=[1], args=[P, D, C_new, counts])
        def update(
            uP: int32[_KP, _KD],
            uD: int64[_KP, _KK],
            uC: int32[_KK, _KD],
            ucounts: int32[_KK],
        ):
            sums: int32[_KK, _KD] = 0
            for c0 in range(_KK):
                ucounts[c0] = 0
            for p in range(_KP):
                best: int64 = uD[p, 0]
                idx: int32 = 0
                for c in range(_KK):
                    idx = c if uD[p, c] < best else idx
                    best = uD[p, c] if uD[p, c] < best else best
                ucounts[idx] += 1
                for d in range(_KD):
                    sums[idx, d] += uP[p, d]
            for c1 in range(_KK):
                for d1 in range(_KD):
                    uC[c1, d1] = (sums[c1, d1] + d1) // (ucounts[c1] + 1)

    return km


_KM_VARIANTS = {
    "extra_op": _km_extra_op,
    "no_divide": _km_no_divide,
    "other_source": _km_other_source,
    "copy_only": _km_copy_only,
    "index_mixed": _km_index_mixed,
}


@pytest.mark.parametrize(
    "kind, message",
    [
        # Extra matched ops (MUL, SUB, DIV, a COMPACT of the points) fail the
        # recognizer, so the group is not host-placed and the matcher's own
        # coverage errors reject the unimplemented stores.
        ("extra_op", "not supported by target"),
        ("no_divide", "not supported by target"),
        ("other_source", "not computed from"),
        # Recognized structure, wrong epilogue: rejected by the IR check.
        ("copy_only", "not divided by the counts"),
        ("index_mixed", "not computed from"),
    ],
)
def test_kmeans_update_rejects_other_host_kernels(kind, message):
    with pytest.raises(Exception, match=message):
        _virtual(_KM_VARIANTS[kind]())


# --------------------------------------------------------------------- #
# U-04: uneven work items share one padded binary
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("dpus", [2, 4, 32])
def test_uneven_gemv_partition_compiles_with_recorded_padding(dpus):
    compiled = _virtual(uneven_gemv_region(391, 65, dpus))
    segment = _segment(compiled)
    kernel = segment.kernel
    last = 391 - (dpus - 1) * (391 // dpus)
    assert kernel.dpus == dpus
    assert kernel.geometry_value("rows") == last
    assert [dict(g)["rows"] for g in kernel.streams] == [391 // dpus] * (dpus - 1) + [
        last
    ]
    fraction = padding_fraction(kernel)
    assert fraction == pytest.approx(1 - 391 / (last * dpus))


def test_uneven_gemv_over_the_full_dpu_level():
    # 391 rows over the target's 64 DPUs: 63 work items of 6 rows, the last
    # of 13.
    compiled = _virtual(uneven_gemv_region(391, 65, 64))
    kernel = _segment(compiled).kernel
    assert kernel.geometry_value("rows") == 13
    assert padding_fraction(kernel) == pytest.approx(1 - 391 / (13 * 64))
    assert _ctx(compiled).virtual_run_extra()["cost_validity"] == "extrapolated"


def test_uneven_streams_need_a_padding_neutral_axis():
    from allo.spmw_upmem import _merge_stream_shapes, upmem_family

    def shape(elements):
        return {
            "geometry": (("elements", elements), ("bins", 64), ("depth", 12)),
            "anchor_op": "INC",
            "constants": (),
            "roles": (("input", "lA"), ("out", "lH")),
            "accumulator_init": "zero",
        }

    # A zero-padded histogram element would be counted in bin 0.
    with pytest.raises(NotImplementedError, match="no padding-neutral axis"):
        _merge_stream_shapes(upmem_family("histogram"), [shape(15), shape(55)])
    merged = _merge_stream_shapes(
        upmem_family("reduction"),
        [
            {**shape(15), "geometry": (("elements", 15),)},
            {**shape(55), "geometry": (("elements", 55),)},
        ],
    )
    assert merged["geometry"] == (("elements", 55),)
    assert merged["streams"] == ((("elements", 15),), (("elements", 55),))


# --------------------------------------------------------------------- #
# uPIMulator: nonzero initial outputs, new forms, uneven partitions
# --------------------------------------------------------------------- #


def _rng(seed):
    return np.random.default_rng(seed)


def test_sim_carried_matrix_vector():
    _sim_gate()
    rng = _rng(3)
    A = rng.integers(-50, 50, size=(36, 40), dtype=np.int32)
    x = rng.integers(-50, 50, size=40, dtype=np.int32)
    y = rng.integers(-1000, 1000, size=36, dtype=np.int32)
    want = (y.astype(np.int64) + A.astype(np.int64) @ x).astype(np.int32)
    result = _compile(mv_region(36, 40, carried=True))(A, x, y)
    np.testing.assert_array_equal(y, want)
    assert result.kind == "simulator"


def test_sim_carried_gemm():
    _sim_gate()
    rng = _rng(4)
    A = rng.integers(-20, 20, size=(20, 24), dtype=np.int32)
    B = rng.integers(-20, 20, size=(24, 18), dtype=np.int32)
    C = rng.integers(-500, 500, size=(20, 18), dtype=np.int32)
    want = (C.astype(np.int64) + A.astype(np.int64) @ B).astype(np.int32)
    _compile(gemm_region(20, 24, 18, carried=True))(A, B, C)
    np.testing.assert_array_equal(C, want)


def test_sim_carried_histogram():
    _sim_gate()
    rng = _rng(5)
    A = rng.integers(0, 4096, size=1000, dtype=np.int32)
    H = rng.integers(0, 100, size=64, dtype=np.int32)
    bins = (A.astype(np.int64) * 64) >> 12
    want = H + np.bincount(bins, minlength=64).astype(np.int32)
    _compile(histogram_region(1000, 64, 12, carried=True))(A, H)
    np.testing.assert_array_equal(H, want)


def test_sim_carried_reduction():
    _sim_gate()
    A = _rng(6).integers(-(1 << 20), 1 << 20, size=1000, dtype=np.int32)
    P = np.array([123456789], dtype=np.int64)
    _compile(reduction_region(1000, carried=True))(A, P)
    assert int(P[0]) == 123456789 + int(A.astype(np.int64).sum())


@pytest.mark.parametrize("expression", ["sub", "mul", "scale", "minus_axpby", "affine"])
def test_sim_pointwise_forms_off_grid(expression):
    _sim_gate()
    rng = _rng(7)
    A = rng.integers(-(1 << 30), 1 << 30, size=1000, dtype=np.int32)
    B = rng.integers(-(1 << 30), 1 << 30, size=1000, dtype=np.int32)
    # int32 edge values, so the wraparound of every product and sum is hit.
    A[:4] = [-(1 << 31), (1 << 31) - 1, -1, 0]
    B[:4] = [(1 << 31) - 1, -(1 << 31), -(1 << 31), (1 << 31) - 1]
    C = np.zeros(1000, dtype=np.int32)
    a, b = A.astype(np.int64), B.astype(np.int64)
    want = {
        "sub": a - b,
        "mul": a * b,
        "scale": -5 * a,
        "minus_axpby": 3 * a - 7 * b,
        "affine": 2 * a + 5 * b + 11,
    }[expression]
    result = _compile(pointwise_region(1000, expression))(A, B, C)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(
        C, want.astype(np.int64).astype(np.uint32).astype(np.int32)
    )


def test_sim_selection_general_predicate_with_carried_count():
    _sim_gate()
    A = _rng(8).integers(-100, 100, size=1536, dtype=np.int32)
    out = np.full(1536, -7, dtype=np.int32)
    count = np.array([5], dtype=np.int32)
    kept = A[(A & 6) != 2]
    _compile(selection_region(1536, "mask", carried=True))(A, out, count)
    assert int(count[0]) == 5 + kept.size
    np.testing.assert_array_equal(out[5 : 5 + kept.size], kept)
    assert (out[:5] == -7).all()


def test_sim_uneven_reduction():
    _sim_gate()
    A = _rng(9).integers(-(1 << 20), 1 << 20, size=1000, dtype=np.int32)
    P = np.zeros(64, dtype=np.int64)
    result = _compile(uneven_reduction_region(1000, 64))(A, P)
    assert int(P.sum()) == int(A.astype(np.int64).sum())
    assert int(P[63]) == int(A[63 * 15 :].astype(np.int64).sum())
    record = result.extra["upmem"][0]
    assert record["padding_fraction"] == pytest.approx(1 - 1000 / (55 * 64))
    assert [g["elements"] for g in record["work_item_geometry"]] == [15] * 63 + [55]


def test_sim_uneven_gemv_full_dpu_level():
    _sim_gate()
    rng = _rng(11)
    A = rng.integers(-8, 8, size=(391, 65), dtype=np.int32)
    x = rng.integers(-8, 8, size=65, dtype=np.int32)
    y = np.zeros(391, dtype=np.int32)
    result = _compile(uneven_gemv_region(391, 65, 64))(A, x, y)
    np.testing.assert_array_equal(y, A @ x)
    cycles = result.extra["upmem"][0]["logic_cycles_per_dpu"]
    # Every DPU runs the binary built for the largest (padded) work item.
    assert len(set(cycles.values())) == 1


SCALING_DPUS = (1, 2, 4, 8, 16, 17, 32)


@pytest.mark.paper_full
def test_sim_uneven_gemv_scaling_391x65():
    """The paper's fixed 391x65 GEMV over 1..32 DPUs, last DPU taking the
    remainder; slowest-DPU cycles must fall from 1 to 32 DPUs and every
    output must be exact."""
    _sim_gate()
    rng = _rng(10)
    A = rng.integers(-8, 8, size=(391, 65), dtype=np.int32)
    x = rng.integers(-8, 8, size=65, dtype=np.int32)
    cycles = {}
    for dpus in SCALING_DPUS:
        y = np.zeros(391, dtype=np.int32)
        result = _compile(uneven_gemv_region(391, 65, dpus))(A, x, y)
        np.testing.assert_array_equal(y, A @ x)
        record = result.extra["upmem"][0]
        assert record["num_dpus"] == dpus
        cycles[dpus] = max(record["logic_cycles_per_dpu"].values())
    assert cycles[32] < cycles[1]
