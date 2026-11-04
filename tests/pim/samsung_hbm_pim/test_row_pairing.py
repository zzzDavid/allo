# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tile rows of a REDUCE program (release S-13,
``docs/release/design/samsung-bank-row-swizzle-route.md``).

A tile's G*E columns are two physical rows.  ``bank_row_rowpair`` reads
the two rows of one bank under one barrier (the identity packing's
within-group conflict); ``split_rows`` stages every tile's second row in
the odd fiber at the same column position, the F2 map ``half -> bank`` that
``LinearLayout.optimal_swizzle`` picks for a {half} segment over the two
fibers (the XOR packing), with the vendor's barrier per group;
``split_rows_rowpair`` (two banks under one barrier) is enumerated and
refused by the lowering: the memory controller may reorder the two banks'
column commands inside a barrier group while the lockstep CRF executes
fixed-bank MAC instructions, which was measured to give wrong results.
The legal ones are ordinary candidates of ``_samsung_enumerate`` with the
shared contract, priced from measured and device constants, lowered to the
vendor CRF forms and run by the conductor with ``issue_order`` /
``split_rows`` in the plan.  The GRF_B entry of a group follows the
physical row parity and the accumulation order per entry is unchanged, so
their results are bit identical to ``bank_row``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import _bucket_for_autoschedule, _samsung_enumerate
from allo.spmw_knobs import reset_active_liveness, set_active_liveness
from allo.spmw_samsung import (
    SAMSUNG_DEFAULT_GEOMETRY,
    samsung_access_stream,
    samsung_access_stream_summary,
    samsung_conductor_trace_text,
    samsung_reduce_plan,
)

_SPEC = importlib.util.spec_from_file_location(
    "samsung_gemv_workload_rowpair", Path(__file__).with_name("gemv") / "workload.py"
)
_WORKLOAD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKLOAD)
FAMILIES = ("bank_row", "dual_fiber", "bank_row_rowpair", "split_rows")


def _candidates(fn):
    target = build_samsung_target()
    default = allo.compile(
        fn, target, samsung_cost, reduction_order="reassociate", backend="virtual"
    )
    buckets = _bucket_for_autoschedule(default.trace)
    token = set_active_liveness(None)
    try:
        candidates = _samsung_enumerate(target, buckets[0][1])
    finally:
        reset_active_liveness(token)
    return target, buckets, candidates


def _vendor_levers(candidates, buckets, family, fill="host"):
    for c in candidates:
        x = c.extra
        if (
            c.mode.split("+")[0] == family
            and set((x.get("grf_residency") or {}).values()) == {fill}
            and x.get("crf_issue") == "shared"
            and not x.get("stage_resident")
        ):
            return [c] * len(buckets)
    raise AssertionError(f"no {family} candidate with fill={fill}")


def test_row_pair_candidates_are_enumerated_with_the_shared_contract():
    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]
    _target, _buckets, candidates = _candidates(fn)
    families = {c.mode.split("+")[0] for c in candidates}
    assert {"bank_row_rowpair", "split_rows", "split_rows_rowpair"} <= families
    expected = {
        "bank_row_rowpair": ("row_pair", "same_bank", False),
        "split_rows": ("tile_major", "split_fibers", True),
        "split_rows_rowpair": ("row_pair", "split_fibers", True),
    }
    for c in candidates:
        family = c.mode.split("+")[0]
        if family in expected:
            order, rows, conflict_free = expected[family]
            assert (c.extra["issue_order"], c.extra["tile_rows"]) == (order, rows)
            assert c.extra.get("effects") is not None  # the shared contract
            half = c.extra["half_bank_map"]
            assert half.describes_conflict_free(
                bank_dims=("bank",), varying_inputs=("half",)
            ) == conflict_free


def test_row_pair_lowering_and_plan():
    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]
    target, buckets, candidates = _candidates(fn)
    programs = {}
    for family in FAMILIES:
        compiled = allo.compile(
            fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
            layout=_vendor_levers(candidates, buckets, family),
        )
        programs[family] = (compiled, compiled.compiled.layout_ctx.programs[0])
    same = programs["bank_row_rowpair"][1]
    split = programs["split_rows"][1]
    # The identity pairing keeps the vendor CRF (one MAC, one JUMP); the split
    # layout is one loop body of an even and an odd MAC.
    assert [b[1:3] for b in same.binds] == [("matrix", 0), ("drain", 1)]
    assert [b[1:3] for b in split.binds] == [("matrix", 0), ("matrix", 1), ("drain", 1)]
    assert same.tile_fiber == (0, 0) and split.tile_fiber == (0, 0)
    assert (same.issue_order, same.tile_rows) == ("row_pair", "same_bank")
    assert (split.issue_order, split.tile_rows) == ("tile_major", "split_fibers")
    geometry = dict(SAMSUNG_DEFAULT_GEOMETRY)
    lines = samsung_reduce_plan(split, geometry, 1)["text"].splitlines()
    assert "issue_order 0" in lines and "split_rows 1" in lines
    lines = samsung_reduce_plan(same, geometry, 1)["text"].splitlines()
    assert "issue_order 1" in lines and "split_rows 0" in lines
    lines = samsung_reduce_plan(programs["bank_row"][1], geometry, 1)["text"].splitlines()
    assert "issue_order 0" in lines and "split_rows 0" in lines
    # Streams.  The identity pairing: a pair is one round of 16 transactions,
    # both rows in bank 0.  The split layout: 32 rounds of 8, each tile's
    # first row in bank 0 and second row in bank 1.
    stream = samsung_access_stream(programs["bank_row_rowpair"][0], geometry=geometry)
    summary = samsung_access_stream_summary(stream)
    assert summary["rounds_by_kind"] == {"DRAIN": 2, "HOST_FILL": 4, "MAC": 16}
    assert summary["rows_by_bank"] == {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [0]}
    pair = next(r for r in stream if r.kind == "MAC")
    assert len(pair.accesses) == 16 and {(a.bank, a.row) for a in pair.accesses} == {(0, 0), (0, 1)}
    stream = samsung_access_stream(programs["split_rows"][0], geometry=geometry)
    summary = samsung_access_stream_summary(stream)
    assert summary["rounds_by_kind"] == {"DRAIN": 2, "HOST_FILL": 4, "MAC": 32}
    assert summary["rows_by_bank"] == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7, 8]}
    # The loop body alternates the even MAC (first row) and the odd MAC
    # (second row), each group under its own barrier.
    macs = [r for r in stream if r.kind == "MAC"][:8]
    assert [(r.accesses[0].bank, r.accesses[0].row) for r in macs] == [(0, 0), (1, 1)] * 4
    # Cost: the measured same-bank pair cost keeps the identity pairing above
    # every other placement; the split layout pays the odd-fiber term.
    estimates = {f: programs[f][0].estimate().cycles for f in programs}
    assert estimates["bank_row_rowpair"] > max(estimates["bank_row"], estimates["dual_fiber"], estimates["split_rows"])
    assert estimates["split_rows"] > estimates["bank_row"]


def test_split_rows_rejects_a_crf_filled_vector_and_the_paired_split():
    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]
    target, buckets, candidates = _candidates(fn)
    with pytest.raises(Exception, match="no CRF-filled GRF_A variant"):
        allo.compile(
            fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
            layout=_vendor_levers(candidates, buckets, "split_rows", fill="crf"),
        )
    with pytest.raises(Exception, match="unsound under the lockstep CRF"):
        allo.compile(
            fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
            layout=_vendor_levers(candidates, buckets, "split_rows_rowpair"),
        )


@pytest.mark.parametrize("case", ["gemv_m8192_k256", "gemv_m4096_k512"])
def test_row_pair_streams_equal_the_conductor_and_results_are_bit_identical(
    case, samsung_sim_gate, tmp_path, monkeypatch
):
    fn, M, K = _WORKLOAD.CASES[case]
    target, buckets, candidates = _candidates(fn)
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    x = (rng.standard_normal(K) * 0.05).astype(np.float16)
    outputs = {}
    for family in FAMILIES:
        dump = tmp_path / f"{case}-{family}.txt"
        monkeypatch.setenv("TENON_CONDUCTOR_ADDRESS_TRACE", str(dump))
        compiled = allo.compile(
            fn, target, samsung_cost, reduction_order="reassociate",
            layout=_vendor_levers(candidates, buckets, family),
        )
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        assert result.kind == "simulator" and result.cycles > 0
        np.testing.assert_allclose(
            y.astype(np.float32), W.astype(np.float32) @ x.astype(np.float32), atol=2e-2
        )
        outputs[family] = y.copy()
        text = dump.read_text() if dump.is_file() else ""
        assert "PASS 0 0" in text, (
            "the installed pim_driver predates the row-pair conductor; rebuild it from "
            "simulators/PIMSimulator (apply.sh)"
        )
        start = text.index("PASS 0 0")
        end = text.index("readback", start)
        assert text[start:end] == samsung_conductor_trace_text(samsung_access_stream(compiled)), family
    for family in ("bank_row_rowpair", "split_rows"):
        assert np.array_equal(outputs[family].view(np.uint16), outputs["bank_row"].view(np.uint16)), family
