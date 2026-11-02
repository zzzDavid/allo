# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural recognition, carried outputs, and runtime contracts for APU v1.

Compile-only: nothing here launches on the board.  Device-facing helpers are
exercised with fake logs/binaries and a temporary lock path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.ir.types import uint16
from allo.pim import apu_v1_program
from allo.pim.apu_v1_native_vector import (
    APUv1NativeRecipeMismatch,
    NoNativeVectorRouteError,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


TARGET = build_apu_v1_target()


@pytest.fixture(scope="module")
def make_kernel(tmp_path_factory):
    """Write a kernel to a real file (Allo reads source) and return it."""

    root = tmp_path_factory.mktemp("apu_v1_kernels")

    def make(source: str, name: str):
        text = "import allo\nfrom allo.ir.types import uint16, uint32\n\n" + (
            textwrap.dedent(source)
        )
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        path = root / f"k_{name}_{digest}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return getattr(module, name)

    return make


def _compile(kernel, result_names=(), bounds=None):
    phase = allo.APUv1Phase(
        kernel,
        vectorize="required",
        result_names=result_names,
        argument_bounds=bounds or {},
    )
    return allo.compile(
        allo.APUv1Program((phase,)), TARGET, apu_v1_cost, backend="virtual"
    )


def _rejects(kernel, result_names, recipe, pattern, bounds=None):
    with pytest.raises(NoNativeVectorRouteError) as info:
        _compile(kernel, result_names, bounds)
    closest = info.value.mismatches[0]
    assert closest.recipe == recipe, str(info.value)
    assert re.search(pattern, closest.reason), closest.reason
    # The native recipe's own reason is surfaced, not a contraction error.
    assert "contraction needs" not in str(info.value)
    return closest


def _normalized_source(compiled, renames):
    source = compiled.compiled.device_source
    for old, new in renames.items():
        # Role names appear inside identifiers such as `mem_hndl_<role>`.
        source = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])", new, source
        )
    return source


# --------------------------------------------------------------------------- #
# Dense histogram
# --------------------------------------------------------------------------- #

HISTOGRAM = """
def {f}({v}: uint16[{N}]) -> uint16[{B}]:
    {c}: uint16[{B}] = {init}
    for {b} in range({B}):
        for {i} in allo.reduction({N}):
            if {v}[{i}] {cmp} {b}:
                {c}[{b}] {upd}
    return {c}
"""


def _histogram(make_kernel, **overrides):
    fields = dict(
        f="histogram",
        v="values",
        c="counts",
        b="bin_index",
        i="item",
        N=32768,
        B=256,
        init="0",
        cmp="==",
        upd="+= 1",
    )
    fields.update(overrides)
    return make_kernel(HISTOGRAM.format(**fields), fields["f"]), (fields["c"],)


def test_histogram_paper_and_renamed_kernels_emit_identical_gvml(make_kernel):
    paper = _compile(*_histogram(make_kernel))
    lowering = paper.compiled.native_vector_lowering
    assert lowering.route == "gvml_dense_histogram_pair_count"
    assert (lowering.analysis.items, lowering.analysis.bins) == (32768, 256)

    renamed = _compile(
        *_histogram(make_kernel, f="tally", v="samples", c="hits", b="k", i="n")
    )
    assert renamed.compiled.native_vector_lowering.route == lowering.route
    assert renamed.estimate().cycles == paper.estimate().cycles
    assert (
        _normalized_source(renamed, {"samples": "values", "hits": "counts"})
        == paper.compiled.device_source
    )


@pytest.mark.parametrize(
    "overrides, pattern",
    [
        ({"upd": "+= 2"}, r"incremented by 2; the template implements \+= 1"),
        ({"upd": "-= 1"}, r"the template implements counts\[bin\] \+= 1"),
        ({"init": "7"}, r"output 'counts' is initialized to 7"),
        ({"cmp": "!="}, r"the guard uses !="),
        ({"cmp": "<"}, r"the guard uses <"),
    ],
)
def test_histogram_semantic_mutations_are_rejected(make_kernel, overrides, pattern):
    _rejects(*_histogram(make_kernel, **overrides), "dense_histogram", pattern)


def test_histogram_nearby_bin_counts_compile_and_out_of_domain_rejects(make_kernel):
    odd = _compile(*_histogram(make_kernel, B=255))
    analysis = odd.compiled.native_vector_lowering.analysis
    assert analysis.bins == 255
    source = odd.compiled.device_source
    assert "gvml_count_m_g32k(&count, GVML_MRK0);" in source
    assert odd.compiled.native_vector_lowering.operation_inventory() == {
        "dma_l4_to_l1_32k": 1,
        "gvml_load_16": 1,
        "gvml_eq_imm_16": 255,
        "gvml_2_fast_count_m_g32k": 127,
        "gvml_count_m_g32k": 1,
        "scalar_l4_store_u16": 255,
    }
    small = _compile(*_histogram(make_kernel, B=64))
    assert small.compiled.native_vector_lowering.analysis.bins == 64
    _rejects(
        *_histogram(make_kernel, N=16384), "dense_histogram", r"exactly one complete"
    )


# --------------------------------------------------------------------------- #
# Squared-L2 argmin
# --------------------------------------------------------------------------- #

ARGMIN = """
def {f}({p}: uint16[32768, {D}], {ce}: uint16[{K}, {D}], {l}: uint16[32768]):
    for {pt} in range(32768):
        {bd}: uint16 = {init}
        for {d} in range({D}):
            {dl}: uint16 = {p}[{pt}, {d}] - {ce}[0, {d}]
            {bd} += {dl} * {dl}
        {bc}: uint16 = {bc_init}
        for {k} in range({k0}, {K}):
            {dist}: uint16 = 0
            for {d} in range({D}):
                {dl}: uint16 = {p}[{pt}, {d}] - {ce}[{k}, {d}]
                {dist} += {dl} * {dl}
            if {cond}:
                {bd} = {dist}
                {bc} = {k}
        {l}[{pt}] = {store}
"""


def _argmin(make_kernel, **overrides):
    fields = dict(
        f="kmeans_assignment",
        p="points",
        ce="centers",
        l="labels",
        pt="point",
        d="dim",
        dl="delta",
        bd="best_distance",
        bc="best_center",
        k="center",
        dist="distance",
        D=24,
        K=10,
        k0=1,
        init="0",
        bc_init="0",
        cond=None,
        store=None,
    )
    fields.update(overrides)
    fields["cond"] = fields["cond"] or f"{fields['dist']} < {fields['bd']}"
    fields["store"] = fields["store"] or fields["bc"]
    return make_kernel(ARGMIN.format(**fields), fields["f"])


def test_argmin_paper_and_renamed_kernels_emit_identical_gvml(make_kernel):
    bounds = {"points": (0, 31), "centers": (0, 31)}
    paper = _compile(_argmin(make_kernel), bounds=bounds)
    lowering = paper.compiled.native_vector_lowering
    assert lowering.route == "gvml_squared_l2_argmin_center_streaming"
    analysis = lowering.analysis
    assert (analysis.dimensions, analysis.candidates) == (24, 10)
    assert analysis.comparison == "slt" and analysis.unsigned_order_proven

    renamed = _compile(
        _argmin(
            make_kernel,
            f="assign",
            p="xs",
            ce="mu",
            l="lab",
            pt="n",
            d="j",
            dl="df",
            bd="bestd",
            bc="besti",
            k="c",
            dist="dd",
        ),
        bounds={"xs": (0, 31), "mu": (0, 31)},
    )
    assert renamed.estimate().cycles == paper.estimate().cycles
    assert (
        _normalized_source(renamed, {"xs": "points", "mu": "centers", "lab": "labels"})
        == paper.compiled.device_source
    )

    # `best > dist` is the same strict improvement written mirrored.
    mirrored = _compile(_argmin(make_kernel, cond="best_distance > distance"))
    assert mirrored.compiled.native_vector_lowering.analysis.comparison == "slt"


@pytest.mark.parametrize(
    "overrides, pattern",
    [
        (
            {"store": "best_distance"},
            r"stores 'best_distance' \(the running minimum distance\), not the "
            r"argmin index 'best_center'",
        ),
        ({"init": "100"}, r"'best_distance' starts at 100"),
        ({"bc_init": "3"}, r"'best_center' starts at 3"),
        ({"cond": "distance <= best_distance"}, r"non-strict <="),
        ({"cond": "distance > best_distance"}, r"selects the maximum"),
        ({"k0": 2}, r"candidate loop .* runs \[2, 10\)"),
    ],
)
def test_argmin_semantic_mutations_are_rejected(make_kernel, overrides, pattern):
    _rejects(_argmin(make_kernel, **overrides), (), "squared_l2_argmin", pattern)


def test_argmin_nearby_shapes_compile_and_out_of_domain_rejects(make_kernel):
    for dims, candidates in ((23, 11), (48, 2)):
        compiled = _compile(_argmin(make_kernel, D=dims, K=candidates))
        analysis = compiled.compiled.native_vector_lowering.analysis
        assert (analysis.dimensions, analysis.candidates) == (dims, candidates)
    _rejects(_argmin(make_kernel, D=49), (), "squared_l2_argmin", r"48 VMRs")


# --------------------------------------------------------------------------- #
# Bivariate moments
# --------------------------------------------------------------------------- #

MOMENTS = """
def {f}({x}: uint16[{B}, {P}], {y}: uint16[{B}, {P}]) -> uint16[{B}, 5]:
    {s}: uint16[{B}, 5] = {init}
    for {b} in range({B}):
        for {p} in allo.reduction({P}):
            {s}[{b}, {c0}] += {x}[{b}, {p}]
            {s}[{b}, {c1}] += {y}[{b}, {p}]
            {s}[{b}, 2] += {x}[{b}, {p}] * {x}[{b}, {p}]
            {s}[{b}, 3] += {y}[{b}, {p}] * {y}[{b}, {p}]
            {s}[{b}, 4] += {x}[{b}, {p}] * {last}[{b}, {p}]
    return {s}
"""


def _moments(make_kernel, **overrides):
    fields = dict(
        f="linear_regression_statistics",
        x="x",
        y="y",
        s="stats",
        b="batch",
        p="point",
        B=8,
        P=4096,
        init="0",
        c0=0,
        c1=1,
        last=None,
    )
    fields.update(overrides)
    fields["last"] = fields["last"] or fields["y"]
    return make_kernel(MOMENTS.format(**fields), fields["f"]), (fields["s"],)


def test_moments_paper_and_renamed_kernels_emit_identical_gvml(make_kernel):
    paper = _compile(*_moments(make_kernel))
    lowering = paper.compiled.native_vector_lowering
    assert lowering.route == "gvml_bivariate_moments_fused_group_reduce"
    assert (lowering.analysis.x_role, lowering.analysis.y_role) == ("x", "y")

    renamed = _compile(
        *_moments(make_kernel, f="moments", x="u", y="v", s="acc", b="g", p="n")
    )
    assert renamed.estimate().cycles == paper.estimate().cycles
    assert (
        _normalized_source(renamed, {"u": "x", "v": "y", "acc": "stats"})
        == paper.compiled.device_source
    )


@pytest.mark.parametrize(
    "overrides, pattern",
    [
        (
            {"c0": 1, "c1": 0},
            r"column 2 accumulates \(x\[batch, point\] \* x\[batch, point\]\); "
            r"with x = 'y' \(column 0\)",
        ),
        ({"init": "3"}, r"output 'stats' is initialized to 3"),
        ({"last": "x"}, r"column 4 accumulates"),
    ],
)
def test_moments_semantic_mutations_are_rejected(make_kernel, overrides, pattern):
    _rejects(*_moments(make_kernel, **overrides), "bivariate_moments", pattern)


def test_moments_nearby_shapes_compile_and_out_of_domain_rejects(make_kernel):
    full = _compile(*_moments(make_kernel, B=16, P=2048))
    assert full.compiled.native_vector_lowering.analysis.batches == 16
    # A partial frontier is zero padded; the padding groups are never read.
    partial = _compile(*_moments(make_kernel, B=4, P=4096))
    lowering = partial.compiled.native_vector_lowering
    x = np.arange(4 * 4096, dtype=np.uint16).reshape(4, 4096)
    packed = lowering.pack_inputs({"x": x, "y": x})
    assert packed["x"].shape == (32768,)
    np.testing.assert_array_equal(packed["x"][: x.size], x.reshape(-1))
    assert not packed["x"][x.size :].any()
    _rejects(*_moments(make_kernel, B=8, P=4095), "bivariate_moments", r"power-of-two")


# --------------------------------------------------------------------------- #
# Record frequency
# --------------------------------------------------------------------------- #

WORD_COUNT = """
def {f}({t}: uint16[{C}, 32768], {q}: uint16[{C}, {Q}]) -> uint16[{Q}]:
    {c}: uint16[{Q}] = {init}
    for {qq} in range({Q}):
        for {tt} in allo.reduction(32768):
{body}
    return {c}
"""


def _word_count(make_kernel, cross=False, **overrides):
    fields = dict(
        f="word_count",
        t="tokens",
        q="queries",
        c="counts",
        qq="query",
        tt="token",
        C=3,
        Q=128,
        init="0",
        upd="+= 1",
    )
    fields.update(overrides)
    lines = []
    for chunk in range(fields["C"]):
        query_chunk = (chunk + 1) % fields["C"] if (cross and chunk == 0) else chunk
        lines.append(
            "    " * (3 + chunk) + f"if {fields['t']}[{chunk}, {fields['tt']}] == "
            f"{fields['q']}[{query_chunk}, {fields['qq']}]:"
        )
    lines.append(
        "    " * (3 + fields["C"]) + f"{fields['c']}[{fields['qq']}] {fields['upd']}"
    )
    source = WORD_COUNT.format(body="\n".join(lines), **fields)
    return make_kernel(source, fields["f"]), (fields["c"],)


def test_record_frequency_paper_and_renamed_kernels_emit_identical_gvml(make_kernel):
    paper = _compile(*_word_count(make_kernel))
    lowering = paper.compiled.native_vector_lowering
    assert lowering.route == "gvml_record_frequency_resident_chunks_pair_count"
    assert (lowering.analysis.chunks, lowering.analysis.queries) == (3, 128)

    renamed = _compile(
        *_word_count(
            make_kernel, f="freq", t="recs", q="keys", c="hits", qq="k", tt="r"
        )
    )
    assert renamed.estimate().cycles == paper.estimate().cycles
    assert (
        _normalized_source(
            renamed, {"recs": "tokens", "keys": "queries", "hits": "counts"}
        )
        == paper.compiled.device_source
    )


@pytest.mark.parametrize(
    "overrides, pattern",
    [
        ({"upd": "+= 2"}, r"incremented by 2"),
        ({"init": "5"}, r"output 'counts' is initialized to 5"),
        ({"cross": True}, r"compares records chunk 0 with queries chunk 1"),
    ],
)
def test_record_frequency_semantic_mutations_are_rejected(
    make_kernel, overrides, pattern
):
    _rejects(*_word_count(make_kernel, **overrides), "record_frequency", pattern)


def test_record_frequency_nearby_shapes_compile_and_out_of_domain_rejects(
    make_kernel,
):
    odd = _compile(*_word_count(make_kernel, Q=127))
    assert (
        odd.compiled.native_vector_lowering.operation_inventory()["gvml_count_m_g32k"]
        == 1
    )
    # Resident chunks are bounded by the target's writable VRs (one scratch).
    widest = _compile(*_word_count(make_kernel, C=14, Q=16))
    assert widest.compiled.native_vector_lowering.analysis.chunks == 14
    _rejects(
        *_word_count(make_kernel, C=15, Q=16),
        "record_frequency",
        r"15 chunks; .*at most 14",
    )


# --------------------------------------------------------------------------- #
# Semantics-preserving rewrites accepted, nearby semantic changes rejected
# --------------------------------------------------------------------------- #

VARIANTS = {
    "histogram_swapped_nest": (
        """
def h(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for i in range(32768):
        for b in range(256):
            if b == values[i]:
                counts[b] = counts[b] + 1
    return counts
""",
        ("counts",),
        "gvml_dense_histogram_pair_count",
    ),
    "moments_reordered_statements": (
        """
def m(x: uint16[8, 4096], y: uint16[8, 4096]) -> uint16[8, 5]:
    s: uint16[8, 5] = 0
    for b in range(8):
        for p in allo.reduction(4096):
            s[b, 4] += y[b, p] * x[b, p]
            s[b, 3] += y[b, p] * y[b, p]
            s[b, 0] += x[b, p]
            s[b, 2] += x[b, p] * x[b, p]
            s[b, 1] += y[b, p]
    return s
""",
        ("s",),
        "gvml_bivariate_moments_fused_group_reduce",
    ),
    "moments_roles_swapped_consistently": (
        """
def m(x: uint16[8, 4096], y: uint16[8, 4096]) -> uint16[8, 5]:
    s: uint16[8, 5] = 0
    for b in range(8):
        for p in allo.reduction(4096):
            s[b, 0] += y[b, p]
            s[b, 1] += x[b, p]
            s[b, 2] += y[b, p] * y[b, p]
            s[b, 3] += x[b, p] * x[b, p]
            s[b, 4] += x[b, p] * y[b, p]
    return s
""",
        ("s",),
        "gvml_bivariate_moments_fused_group_reduce",
    ),
    "argmin_reordered_args_reversed_difference_no_temps": (
        """
def a(centers: uint16[10, 24], labels: uint16[32768], points: uint16[32768, 24]):
    for p in range(32768):
        best: uint16 = 0
        for d in range(24):
            best += (centers[0, d] - points[p, d]) * (centers[0, d] - points[p, d])
        idx: uint16 = 0
        for k in range(1, 10):
            dist: uint16 = 0
            for d in range(24):
                dist += (points[p, d] - centers[k, d]) * (points[p, d] - centers[k, d])
            if dist < best:
                best = dist
                idx = k
        labels[p] = idx
""",
        (),
        "gvml_squared_l2_argmin_center_streaming",
    ),
    "record_frequency_permuted_chunks": (
        """
def w(tokens: uint16[3, 32768], queries: uint16[3, 128]) -> uint16[128]:
    counts: uint16[128] = 0
    for q in range(128):
        for t in allo.reduction(32768):
            if queries[2, q] == tokens[2, t]:
                if tokens[0, t] == queries[0, q]:
                    if tokens[1, t] == queries[1, q]:
                        counts[q] += 1
    return counts
""",
        ("counts",),
        "gvml_record_frequency_resident_chunks_pair_count",
    ),
}


@pytest.mark.parametrize("case", sorted(VARIANTS))
def test_semantics_preserving_variants_are_accepted(make_kernel, case):
    source, results, route = VARIANTS[case]
    name = source.split("def ", 1)[1].split("(", 1)[0]
    compiled = _compile(make_kernel(source, name), results)
    assert compiled.compiled.native_vector_lowering.route == route


REJECTED_VARIANTS = {
    "histogram_extra_store": (
        """
def h(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for b in range(256):
        for i in range(32768):
            if values[i] == b:
                counts[b] += 1
                counts[0] += 1
    return counts
""",
        ("counts",),
        "dense_histogram",
        r"exactly one store .* found 2",
    ),
    "histogram_uint32_counts": (
        """
def h(values: uint16[32768]) -> uint32[256]:
    counts: uint32[256] = 0
    for b in range(256):
        for i in range(32768):
            if values[i] == b:
                counts[b] += 1
    return counts
""",
        ("counts",),
        "dense_histogram",
        r"counts:ui32; the template implements uint16",
    ),
    "histogram_reversed_bin_index": (
        """
def h(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for b in range(256):
        for i in range(32768):
            if values[i] == b:
                counts[255 - b] += 1
    return counts
""",
        ("counts",),
        "dense_histogram",
        r"is not a loop induction variable or integer literal",
    ),
    "moments_scaled_square": (
        """
def m(x: uint16[8, 4096], y: uint16[8, 4096]) -> uint16[8, 5]:
    s: uint16[8, 5] = 0
    for b in range(8):
        for p in allo.reduction(4096):
            s[b, 0] += x[b, p]
            s[b, 1] += y[b, p]
            s[b, 2] += 2 * x[b, p] * x[b, p]
            s[b, 3] += y[b, p] * y[b, p]
            s[b, 4] += x[b, p] * y[b, p]
    return s
""",
        ("s",),
        "bivariate_moments",
        r"column 2 accumulates",
    ),
    "argmin_index_off_by_one": (
        """
def a(points: uint16[32768, 24], centers: uint16[10, 24], labels: uint16[32768]):
    for p in range(32768):
        best: uint16 = 0
        for d in range(24):
            best += (points[p, d] - centers[0, d]) * (points[p, d] - centers[0, d])
        idx: uint16 = 0
        for k in range(1, 10):
            dist: uint16 = 0
            for d in range(24):
                dist += (points[p, d] - centers[k, d]) * (points[p, d] - centers[k, d])
            if dist < best:
                best = dist
                idx = k + 1
        labels[p] = idx
""",
        (),
        "squared_l2_argmin",
        r"no local records the candidate index",
    ),
    "argmin_wrong_candidate_row": (
        """
def a(points: uint16[32768, 24], centers: uint16[10, 24], labels: uint16[32768]):
    for p in range(32768):
        best: uint16 = 0
        for d in range(24):
            best += (points[p, d] - centers[0, d]) * (points[p, d] - centers[0, d])
        idx: uint16 = 0
        for k in range(1, 10):
            dist: uint16 = 0
            for d in range(24):
                dist += (points[p, d] - centers[0, d]) * (points[p, d] - centers[0, d])
            if dist < best:
                best = dist
                idx = k
        labels[p] = idx
""",
        (),
        "squared_l2_argmin",
        r"candidate distance accumulates",
    ),
    "record_frequency_missing_chunk": (
        """
def w(tokens: uint16[3, 32768], queries: uint16[3, 128]) -> uint16[128]:
    counts: uint16[128] = 0
    for q in range(128):
        for t in allo.reduction(32768):
            if tokens[0, t] == queries[0, q]:
                if tokens[1, t] == queries[1, q]:
                    counts[q] += 1
    return counts
""",
        ("counts",),
        "record_frequency",
        r"missing \[2\]",
    ),
    "record_frequency_else_branch": (
        """
def w(tokens: uint16[1, 32768], queries: uint16[1, 128]) -> uint16[128]:
    counts: uint16[128] = 0
    for q in range(128):
        for t in allo.reduction(32768):
            if tokens[0, t] == queries[0, q]:
                pass
            else:
                counts[q] += 1
    return counts
""",
        ("counts",),
        "record_frequency",
        r"else branch",
    ),
}


@pytest.mark.parametrize("case", sorted(REJECTED_VARIANTS))
def test_nearby_semantic_changes_are_rejected(make_kernel, case):
    source, results, recipe, pattern = REJECTED_VARIANTS[case]
    name = source.split("def ", 1)[1].split("(", 1)[0]
    _rejects(make_kernel(source, name), results, recipe, pattern)


# --------------------------------------------------------------------------- #
# Error family and registry
# --------------------------------------------------------------------------- #


def test_no_native_route_reports_every_recipe_closest_first(make_kernel):
    with pytest.raises(NotImplementedError) as info:
        _compile(*_histogram(make_kernel, upd="+= 2"))
    error = info.value
    assert isinstance(error, NoNativeVectorRouteError)
    assert {item.recipe for item in error.mismatches} == {
        "dense_histogram",
        "squared_l2_argmin",
        "record_frequency",
        "bivariate_moments",
    }
    assert all(isinstance(item, APUv1NativeRecipeMismatch) for item in error.mismatches)
    assert error.mismatches[0].recipe == "dense_histogram"


# --------------------------------------------------------------------------- #
# V1-02: carried outputs agree on every route
# --------------------------------------------------------------------------- #


def carried_gemm(left: uint16[64, 32], right: uint16[32, 64], result: uint16[64, 64]):
    for row, column in allo.grid(64, 64):
        for depth in allo.reduction(32):
            result[row, column] += left[row, depth] * right[depth, column]


def _gemm_inputs():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 2**16, (64, 32), dtype=np.uint16)
    right = rng.integers(0, 2**16, (32, 64), dtype=np.uint16)
    initial = rng.integers(1, 2**16, (64, 64), dtype=np.uint16)
    expected = (
        initial.astype(np.uint64) + left.astype(np.uint64) @ right.astype(np.uint64)
    ) % 2**16
    return left, right, initial, expected.astype(np.uint16)


@pytest.mark.parametrize("backend", ["functional", "virtual"])
def test_temporal_plan_accumulates_nonzero_initial_output(backend):
    compiled = allo.compile(carried_gemm, TARGET, apu_v1_cost, backend=backend)
    realization = compiled.realization
    assert realization.carries_output_ingress
    left, right, initial, expected = _gemm_inputs()
    result = initial.copy()
    run = compiled(left, right, result)
    if backend == "virtual":
        # The virtual route prices the plan; it executes and writes nothing.
        assert run.kind == "virtual" and run.outputs == {}
        np.testing.assert_array_equal(result, initial)
    else:
        assert run.kind == "functional"
        np.testing.assert_array_equal(result, expected)

    # The device ABI transfers the carried output image instead of zeroing it.
    artifact = realization.runtime_artifact
    if artifact is None:
        pytest.skip("GSI example-gvml template not installed")
    abi = __import__("json").loads(artifact.abi_manifest_json)
    assert abi["argv"]["initial_outputs"] == ["result"]
    host = dict(artifact.project_files)["host.c"].decode()
    assert "path_init_result" in host
    assert "calloc(1, sz_result)" not in host


def test_plan_without_output_ingress_rejects_nonzero_output_on_every_route(
    monkeypatch,
):
    monkeypatch.setenv("TENON_APU_V1_PCI_NODE", "/nonexistent/apu-pci-node")
    left, right, initial, _ = _gemm_inputs()
    for backend in ("functional", "virtual", None):
        compiled = allo.compile(
            carried_gemm,
            TARGET,
            apu_v1_cost,
            backend=backend,
            layout="baseline_spatial_reduction",
        )
        assert not compiled.realization.carries_output_ingress
        with pytest.raises(ValueError, match="cannot be accumulated"):
            compiled(left, right, initial.copy())
    zero = np.zeros_like(initial)
    functional = allo.compile(
        carried_gemm,
        TARGET,
        apu_v1_cost,
        backend="functional",
        layout="baseline_spatial_reduction",
    )
    run = functional(left, right, zero)
    assert run.extra["plan_selection"] == "explicit_layout"


# --------------------------------------------------------------------------- #
# V1-10: plan selection without a cost program
# --------------------------------------------------------------------------- #


def test_vector_plan_selection_requires_cost_or_explicit_layout():
    with pytest.raises(ValueError, match="selected by cost ranking"):
        allo.compile(carried_gemm, TARGET, None, backend="functional")
    explicit = allo.compile(
        carried_gemm, TARGET, None, backend="functional", layout="temporal_svp"
    )
    assert explicit.selected_plan.name == "temporal_svp"
    ranked = allo.compile(carried_gemm, TARGET, apu_v1_cost, backend="functional")
    left, right, _, _ = _gemm_inputs()
    run = ranked(left, right, np.zeros((64, 64), np.uint16))
    assert run.extra["plan_selection"] == "cost_ranked"


# --------------------------------------------------------------------------- #
# V1-07: board lock and fresh profiles
# --------------------------------------------------------------------------- #


def test_board_lock_path_is_per_device_and_overridable(monkeypatch, tmp_path):
    monkeypatch.delenv("TENON_APU_V1_LOCK", raising=False)
    node = tmp_path / "0000:41:00.0"
    node.mkdir()
    monkeypatch.setenv("TENON_APU_V1_PCI_NODE", str(node))
    assert (
        str(apu_v1_program.apu_v1_lock_path()) == "/tmp/tenon-apu-v1-0000:41:00.0.lock"
    )
    monkeypatch.setenv("TENON_APU_V1_LOCK", str(tmp_path / "board.lock"))
    assert apu_v1_program.apu_v1_lock_path() == tmp_path / "board.lock"


def test_board_lock_is_reentrant_and_excludes_other_processes(monkeypatch, tmp_path):
    lock = tmp_path / "board.lock"
    monkeypatch.setenv("TENON_APU_V1_LOCK", str(lock))
    probe = (
        "import fcntl, os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(3)\n"
    )
    with apu_v1_program.apu_v1_board_lock() as path:
        assert path == lock
        with apu_v1_program.apu_v1_board_lock():
            held = subprocess.run([sys.executable, "-c", probe, str(lock)])
            assert held.returncode == 3
        still = subprocess.run([sys.executable, "-c", probe, str(lock)])
        assert still.returncode == 3
    free = subprocess.run([sys.executable, "-c", probe, str(lock)])
    assert free.returncode == 0


def test_launch_requires_exactly_one_fresh_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_APU_V1_LOCK", str(tmp_path / "board.lock"))
    monkeypatch.setattr(apu_v1_program.time, "sleep", lambda seconds: None)
    argv = [sys.executable, "-c", "pass"]
    stale = "ARCT[0]: ***  total - hits:1 seu:9 crun:111 iall:1 icm:0 dcm:0"
    fresh = "ARCT[0]: ***  total - hits:1 seu:9 crun:222 iall:1 icm:0 dcm:0"

    def drains(*texts):
        sequence = iter(texts)
        monkeypatch.setattr(apu_v1_program, "_ledag_drain", lambda: next(sequence, ""))

    # Each `flo` prints the whole retained log, so the post-launch read is
    # the pre-launch read plus the records this launch appended.
    drains(stale, stale + "\n" + fresh)
    _, _, cycles = apu_v1_program.launch_on_board(
        argv, tmp_path, expected_records=1, timeout=60
    )
    assert cycles == (222,)

    # A run whose PROF_PRINT never arrived must not report the stale counter.
    drains(stale, stale, stale, stale, stale)
    with pytest.raises(RuntimeError, match="found 0"):
        apu_v1_program.launch_on_board(argv, tmp_path, expected_records=1, timeout=60)
    drains("", fresh + "\n" + fresh)
    with pytest.raises(RuntimeError, match="found 2"):
        apu_v1_program.launch_on_board(argv, tmp_path, expected_records=1, timeout=60)
    # A log that rotated past the pre-launch read cannot attribute records.
    drains(stale, fresh)
    with pytest.raises(RuntimeError, match="not a prefix"):
        apu_v1_program.launch_on_board(argv, tmp_path, expected_records=1, timeout=60)


# --------------------------------------------------------------------------- #
# V1-09 / V1-14: derived output ABI and role-named readback
# --------------------------------------------------------------------------- #


def test_matcher_output_specs_derive_from_abi_or_raise():
    from allo.spmw_apu_v1 import _apu_v1_output_specs

    def compiled(memref_type):
        operand = SimpleNamespace(memref_name="out", memref_type=memref_type)
        match = SimpleNamespace(result_memref_name="out", operands=[operand])
        return SimpleNamespace(trace=SimpleNamespace(matches=[match]))

    given = np.zeros((3, 5), np.int16)
    assert _apu_v1_output_specs(compiled("memref<3x5xi16>"), {"out": given}) == {
        "out": ((3, 5), np.dtype(np.int16))
    }
    assert _apu_v1_output_specs(compiled("memref<7xf16>"), {}) == {
        "out": ((7,), np.dtype(np.float16))
    }
    with pytest.raises(NotImplementedError, match="cannot derive .* 'out'"):
        _apu_v1_output_specs(compiled("memref<7xi16>"), {"big": np.zeros(99)})


def test_short_or_missing_readback_names_the_role(tmp_path):
    path = tmp_path / "out_counts.bin"
    with pytest.raises(RuntimeError, match="output 'counts': the device run wrote no"):
        apu_v1_program.read_output_image(path, "counts", np.uint16, (256,))
    np.zeros(100, np.uint16).tofile(path)
    with pytest.raises(
        RuntimeError, match="output 'counts': .* 200 bytes, expected 512"
    ):
        apu_v1_program.read_output_image(path, "counts", np.uint16, (256,))
    np.arange(256, dtype=np.uint16).tofile(path)
    np.testing.assert_array_equal(
        apu_v1_program.read_output_image(path, "counts", np.uint16, (256,)),
        np.arange(256, dtype=np.uint16),
    )


def test_host_harness_reads_carried_output_image_with_exact_size():
    from allo.spmw_apu_v1_build import _emit_host_c

    host = _emit_host_c(
        ["left"],
        ["result", "stats"],
        {"left": 64},
        {"result": 128, "stats": 32},
        initial_output_roles=("result",),
    )
    assert "const char *path_init_result = argv[4];" in host
    assert "initial image size mismatch" in host
    assert "calloc(1, sz_result)" not in host
    assert "calloc(1, sz_stats)" in host
    with pytest.raises(ValueError, match="not outputs"):
        _emit_host_c(["a"], ["b"], {"a": 2}, {"b": 2}, initial_output_roles=("a",))


def test_device_runtimes_never_launch_without_ledag(monkeypatch, tmp_path):
    from allo.spmw_codegen import SimulatorUnavailable

    monkeypatch.setattr(apu_v1_program.shutil, "which", lambda name: None)
    assert "ledag-ssh" in apu_v1_program._ledag_available_reason()
    monkeypatch.setenv("TENON_APU_V1_PCI_NODE", os.fspath(tmp_path / "missing"))
    compiled = allo.compile(carried_gemm, TARGET, apu_v1_cost)
    left, right, _, _ = _gemm_inputs()
    with pytest.raises(SimulatorUnavailable):
        compiled(left, right, np.zeros((64, 64), np.uint16))
