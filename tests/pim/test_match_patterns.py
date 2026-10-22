# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matcher unary / select / compare patterns and vector width (task 006).

Spec: design_doc/compiler/matcher-path-shared-infra.md, section D4. Each test
lowers a tiny kernel with ``customize(enable_tensor=False)`` and matches it
against a tiny target declared inline. No simulator, no production target.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32, int32, int64
from allo.perf import cost, rule
from allo.spmw_match import abs_, erf, exp, log, rsqrt, select, sqrt, tanh
from allo.spmw_match_engine import match_workload

N = 32


def _match(kernel, target):
    return match_workload(target, allo.customize(kernel, enable_tensor=False).module)


def _roles(match):
    return {b.role: b.memref_name for b in match.operands}


def k_exp(A: float32[N], B: float32[N]):
    for i in range(N):
        B[i] = allo.exp(A[i])


def k_add_exp(A: float32[N], B: float32[N], C: float32[N]):
    for i in range(N):
        C[i] = A[i] + allo.exp(B[i])


def k_max(A: float32[N], B: float32[N], C: float32[N]):
    for i in range(N):
        C[i] = max(A[i], B[i])


def k_max_reduce(A: float32[N], R: float32[1]):
    for i in range(N):
        R[0] = max(R[0], A[i])


def k_select_gt(
    A: float32[N], B: float32[N], X: float32[N], Y: float32[N], C: float32[N]
):
    for i in range(N):
        C[i] = X[i] if A[i] > B[i] else Y[i]


def k_select_lt(
    A: float32[N], B: float32[N], X: float32[N], Y: float32[N], C: float32[N]
):
    for i in range(N):
        C[i] = X[i] if A[i] < B[i] else Y[i]


def k_exp_stride2(A: float32[2 * N], B: float32[N]):
    for i in range(N):
        B[i] = allo.exp(A[2 * i])


def k_geva(A: int32[N], B: int32[N], C: int32[N]):
    for i in range(N):
        C[i] = 2 * A[i] + (-1) * B[i]


def k_geva_f32(A: float32[N], B: float32[N], C: float32[N]):
    for i in range(N):
        C[i] = 2.0 * A[i] + (-1.0) * B[i]


def k_add_neg(A: float32[N], B: float32[N], C: float32[N]):
    for i in range(N):
        C[i] = A[i] + (-B[i])


def _unary_target(vector_width=1):
    @allo.target("t006_unary")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op(
                    "EXP",
                    src=r,
                    dst=r,
                    fn=lambda x: exp(x),
                    vector_width=vector_width,
                )

    return t


def test_exp_matches_unary_op():
    trace = _match(k_exp, _unary_target())
    assert [m.target_op_name for m in trace.matches] == ["EXP"]
    assert _roles(trace.matches[0]) == {"x": "A"}
    assert trace.matches[0].result_memref_name == "B"


def test_exp_is_not_a_transparent_cast():
    @allo.target("t006_add")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("ADD", src=r, dst=r, fn=lambda x, y: x + y)

    assert _match(k_add_exp, t).matches == []

    @allo.target("t006_add_exp")
    def t2():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("ADD_EXP", src=r, dst=r, fn=lambda x, y: x + exp(y))

    trace = _match(k_add_exp, t2)
    assert [m.target_op_name for m in trace.matches] == ["ADD_EXP"]
    assert _roles(trace.matches[0]) == {"x": "A", "y": "B"}


def test_max_elementwise_and_reduction():
    @allo.target("t006_max")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op(
                    "MAX_RED",
                    src=r,
                    dst=r,
                    fn=lambda x, acc: max(acc, x),
                    accumulates=True,
                )
                allo.op("MAX", src=r, dst=r, fn=lambda x, y: max(x, y))

    elementwise = _match(k_max, t).matches
    assert [m.target_op_name for m in elementwise] == ["MAX"]
    assert set(_roles(elementwise[0]).values()) == {"A", "B"}

    reduction = _match(k_max_reduce, t).matches
    assert [m.target_op_name for m in reduction] == ["MAX_RED"]
    assert _roles(reduction[0]) == {"x": "A", "acc": "R"}
    assert [b.is_loop_carried for b in reduction[0].operands] == [False, True]


def _select_target():
    @allo.target("t006_select")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op(
                    "SELECT_GT",
                    src=r,
                    dst=r,
                    fn=lambda a, b, x, y: select(a > b, x, y),
                )

    return t


def test_select_matches_cmpf_ogt():
    trace = _match(k_select_gt, _select_target())
    assert [m.target_op_name for m in trace.matches] == ["SELECT_GT"]
    assert _roles(trace.matches[0]) == {"a": "A", "b": "B", "x": "X", "y": "Y"}


def test_compare_matches_mirrored_predicate():
    # `X if A < B else Y` lowers to `cmpf olt A, B`; the pattern `a > b`
    # matches it with the operands swapped.
    trace = _match(k_select_lt, _select_target())
    assert [m.target_op_name for m in trace.matches] == ["SELECT_GT"]
    assert _roles(trace.matches[0]) == {"a": "B", "b": "A", "x": "X", "y": "Y"}


def test_vector_width_contiguous_vs_strided():
    target = _unary_target(vector_width=16)
    contiguous = _match(k_exp, target).matches
    assert [m.vector_width for m in contiguous] == [16]
    strided = _match(k_exp_stride2, target).matches
    assert [m.target_op_name for m in strided] == ["EXP"]
    assert [m.vector_width for m in strided] == [1]
    # Default declaration reports 1 on the contiguous axis.
    assert [m.vector_width for m in _match(k_exp, _unary_target()).matches] == [1]


def test_const_params_bind_compile_time_coefficients():
    # `(-1)` lowers to `subi 0, 1` (int) and `negf 1.0` (float), not a
    # single constant; both must bind beta = -1.
    @allo.target("t025_axpby")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op(
                    "AXPBY",
                    src=r,
                    dst=r,
                    fn=lambda x, y, alpha, beta: alpha * x + beta * y,
                    const_params=("alpha", "beta"),
                )

    for kernel in (k_geva, k_geva_f32):
        matches = _match(kernel, t).matches
        assert [m.target_op_name for m in matches] == ["AXPBY"]
        assert matches[0].extra["constants"] == {"alpha": 2, "beta": -1}
        assert _roles(matches[0]) == {"x": "A", "y": "B", "alpha": None, "beta": None}

    @allo.target("t025_axpby_loads")
    def t_loads():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op(
                    "AXPBY",
                    src=r,
                    dst=r,
                    fn=lambda x, y, alpha, beta: alpha * x + beta * y,
                )

    assert _match(k_geva, t_loads).matches == []


def test_negation_is_not_a_transparent_cast():
    # Float `-B[i]` lowers to `negf`; it must not trace as `B[i]`.
    @allo.target("t025_add_neg")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("ADD", src=r, dst=r, fn=lambda x, y: x + y)

    assert _match(k_add_neg, t).matches == []

    @allo.target("t025_add_negated")
    def t2():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("ADD_NEG", src=r, dst=r, fn=lambda x, y: x + (-y))

    matches = _match(k_add_neg, t2).matches
    assert [m.target_op_name for m in matches] == ["ADD_NEG"]
    assert _roles(matches[0]) == {"x": "A", "y": "B"}


def test_pattern_helpers_are_fingerprintable():
    @allo.target("t028_helpers")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("EXP", src=r, dst=r, fn=lambda x: exp(x))
                allo.op("LOG", src=r, dst=r, fn=lambda x: log(x))
                allo.op("SQRT", src=r, dst=r, fn=lambda x: sqrt(x))
                allo.op("RSQRT", src=r, dst=r, fn=lambda x: rsqrt(x))
                allo.op("TANH", src=r, dst=r, fn=lambda x: tanh(x))
                allo.op("ERF", src=r, dst=r, fn=lambda x: erf(x))
                allo.op("ABS", src=r, dst=r, fn=lambda x: abs_(x))
                allo.op(
                    "SELECT",
                    src=r,
                    dst=r,
                    fn=lambda a, b, x, y: select(a > b, x, y),
                )

    # `bind` rejects a program with no rules; one trivial rule is enough to
    # fingerprint every op `fn` and the helpers it references.
    @cost(target="t028_helpers")
    def helper_cost(target_spec):
        @rule(target_spec.op("EXP"))
        def _(event, ctx):
            ctx.step(cycles=1, name="EXP")

    first = helper_cost.bind(t).fingerprint
    second = helper_cost.bind(t).fingerprint
    assert isinstance(first, str) and first
    assert first == second


# --------------------------------------------------------------------- #
# Spec 004 (task 016): irregular UPMEM patterns, M1 to M7
# --------------------------------------------------------------------- #

H_BINS = 8


def k_hist_guarded(A: int32[N], H: int32[H_BINS]):
    for j in range(H_BINS):
        H[j] = 0
    for i in range(N):
        b: int32 = (A[i] * 8) >> 6
        if b >= 0 and b < H_BINS:
            H[b] += 1


def k_hist_inline(A: int32[N], H: int32[H_BINS]):
    for i in range(N):
        H[(A[i] * 8) >> 6] += 1


def k_sel(A: int32[N], out: int32[N], count: int32[1]):
    count[0] = 0
    for i in range(N):
        if (A[i] & 1) != 0:
            out[count[0]] = A[i]
            count[0] += 1


def k_kmeans_distances(P: int32[12, 8], C: int32[4, 8], D: int64[12, 4]):
    for p in range(12):
        for c in range(4):
            D[p, c] = 0
            for d in range(8):
                D[p, c] += (P[p, d] - C[c, d]) * (P[p, d] - C[c, d])


def k_linear_reg(S: int32[12, 9], G: int64[8]):
    for f in range(8):
        G[f] = 0
    for s in range(12):
        for f in range(8):
            G[f] += ((S[s, f] * S[s, 8]) * -32) >> 8


def k_logistic_reg(S: int32[12, 9], G: int64[8]):
    for f in range(8):
        G[f] = 0
    for s in range(12):
        for f in range(8):
            G[f] += S[s, f] * (1 - 2 * S[s, 8])


def k_argmin(D: int64[12, 4], a: int32[12]):
    for p in range(12):
        best: int64 = D[p, 0]
        idx: int32 = 0
        for c in range(4):
            idx = c if D[p, c] < best else idx
            best = D[p, c] if D[p, c] < best else best
        a[p] = idx


def k_scatter(P: int32[12, 8], a: int32[12], sums: int32[4, 8]):
    for p in range(12):
        for d in range(8):
            sums[a[p], d] += P[p, d]


def k_scatter_mismatch(P: int32[12, 8], a: int32[12], b: int32[12], sums: int32[4, 8]):
    for p in range(12):
        for d in range(8):
            sums[a[p], d] = sums[b[p], d] + P[p, d]


_IRREGULAR_OPS = {
    "INC": dict(fn=lambda acc: acc + 1, accumulates=True, guarded=True, dst_index="any"),
    "INC_UNGUARDED": dict(fn=lambda acc: acc + 1, accumulates=True, dst_index="any"),
    "COMPACT": dict(fn=lambda x: x, guarded=True, dst_index="any"),
    "SCATTER_ADD": dict(
        fn=lambda x, acc: acc + x, accumulates=True, guarded=True, dst_index="any"
    ),
    "SQDIST_ACC": dict(fn=lambda x, y, acc: acc + (x - y) * (x - y), accumulates=True),
    "GRAD_LINEAR": dict(
        fn=lambda x, y, acc, alpha, beta: acc + (((x * y) * alpha) >> beta),
        accumulates=True,
        const_params=("alpha", "beta"),
    ),
    "GRAD_LOGISTIC": dict(
        fn=lambda x, y, acc, alpha, beta: acc + x * (alpha - beta * y),
        accumulates=True,
        const_params=("alpha", "beta"),
    ),
    "ADD": dict(fn=lambda x, y: x + y),
    "MIN_SEL": dict(fn=lambda x, acc: select(x < acc, x, acc), accumulates=True),
    "ARGMIN_IDX": dict(
        fn=lambda x, best, i, acc: select(x < best, i, acc),
        accumulates=True,
        iv_params=("i",),
    ),
    "ARGMIN_NO_IV": dict(
        fn=lambda x, best, i, acc: select(x < best, i, acc), accumulates=True
    ),
}


def _irregular_target(*names):
    @allo.target("t016_" + "_".join(n.lower() for n in names))
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                for name in names:
                    allo.op(name, src=r, dst=r, **_IRREGULAR_OPS[name])

    return t


def _ops(trace):
    return sorted(m.target_op_name for m in trace.matches)


def test_indirect_accumulate_records_index_and_guard():
    from allo.spmw_match_engine import WBinOp, WConst, WLoad

    (inc,) = _match(k_hist_guarded, _irregular_target("INC")).matches
    (index,) = inc.extra["index_terms"]
    # M3 forwards the bin scalar `b` to its single store.
    assert isinstance(index, WBinOp) and index.op == "shr"
    assert index.rhs.value == 6 and isinstance(index.lhs, WBinOp)
    assert {type(index.lhs.lhs), type(index.lhs.rhs)} == {WLoad, WConst}
    ((tag, cond),) = inc.extra["guards"]
    assert tag == "if" and isinstance(cond, WBinOp) and cond.op == "and"
    # M2: the recomputed inline index unifies with itself.
    assert _ops(_match(k_hist_inline, _irregular_target("INC"))) == ["INC"]
    # M5: an op that does not opt in never matches a guarded store.
    assert _match(k_hist_guarded, _irregular_target("INC_UNGUARDED")).matches == []


def test_select_compact_shares_one_guard():
    from allo.spmw_match_engine import WLoad, _term_eq

    trace = _match(k_sel, _irregular_target("COMPACT", "INC"))
    assert _ops(trace) == ["COMPACT", "INC"]
    compact = next(m for m in trace.matches if m.target_op_name == "COMPACT")
    counter = next(m for m in trace.matches if m.target_op_name == "INC")
    (cell,) = compact.extra["index_terms"]
    assert isinstance(cell, WLoad) and cell.memref_name == "count"
    ((tag_a, guard_a),) = compact.extra["guards"]
    ((tag_b, guard_b),) = counter.extra["guards"]
    assert tag_a == tag_b == "if" and _term_eq(guard_a, guard_b)


def test_structural_equality_binds_repeated_parameters():
    (match,) = _match(k_kmeans_distances, _irregular_target("SQDIST_ACC")).matches
    assert _roles(match) == {"x": "P", "y": "C", "acc": "D"}


def test_regression_update_constants():
    (linear,) = _match(k_linear_reg, _irregular_target("GRAD_LINEAR")).matches
    assert linear.extra["constants"] == {"alpha": -32, "beta": 8}
    assert _roles(linear)["acc"] == "G"
    (logistic,) = _match(k_logistic_reg, _irregular_target("GRAD_LOGISTIC")).matches
    assert logistic.extra["constants"] == {"alpha": 1, "beta": 2}


def test_argmin_binds_loop_index():
    trace = _match(k_argmin, _irregular_target("ARGMIN_IDX", "MIN_SEL"))
    assert _ops(trace) == ["ARGMIN_IDX", "MIN_SEL"]
    argmin = next(m for m in trace.matches if m.target_op_name == "ARGMIN_IDX")
    assert argmin.extra["iv_params"] == {"i": 1}
    minimum = next(m for m in trace.matches if m.target_op_name == "MIN_SEL")
    assert _roles(minimum)["acc"] == "best"
    # Without iv_params the loop variable is not a load, so the index store
    # does not match (the `best` update still fits the pattern with i = D).
    plain = _match(k_argmin, _irregular_target("ARGMIN_NO_IV")).matches
    assert [m.result_memref_name for m in plain] == ["best"]


def test_scatter_add_checks_accumulator_index():
    (match,) = _match(k_scatter, _irregular_target("SCATTER_ADD")).matches
    assert _roles(match) == {"x": "P", "acc": "sums"}
    assert match.extra["index_terms"][0].memref_name == "a"
    assert _match(k_scatter_mismatch, _irregular_target("SCATTER_ADD")).matches == []
    # An ordinary-index op never writes a data-dependent element.
    (scatter,) = _match(k_scatter, _irregular_target("ADD", "SCATTER_ADD")).matches
    assert scatter.target_op_name == "SCATTER_ADD"


def k_sibling_loops(A: int32[N], B: int32[N], C: int32[N], D: int32[N]):
    for i in range(N):
        C[i] = A[i] + B[i]
    for j in range(N):
        D[j] = A[j] - B[j]


def test_sibling_loops_with_equal_ssa_names_trace_their_own_values():
    # The printer restarts numbering per loop region, so both loops define
    # the same names; each store must trace its own loop's values.
    @allo.target("t016_siblings")
    def t():
        @allo.device
        def dev():
            @allo.unit(mapping={"lane": 1})
            def lane():
                r = allo.reg(16, 32, name="r")
                allo.op("ADD", src=r, dst=r, fn=lambda x, y: x + y)
                allo.op("SUB", src=r, dst=r, fn=lambda x, y: x - y)

    trace = _match(k_sibling_loops, t)
    assert sorted((m.target_op_name, m.result_memref_name) for m in trace.matches) == [
        ("ADD", "C"),
        ("SUB", "D"),
    ]
