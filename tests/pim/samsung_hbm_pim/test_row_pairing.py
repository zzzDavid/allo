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
            # The declared matrix packing is the map's rendering.
            effects = c.extra["effects"]
            matrix = next(v for v in effects.value_layouts if v.storage == "banks" and v.value in effects.coverage.inputs)
            assert matrix.packing == ("half_bank:0,1" if conflict_free else "half_bank:0,0")


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
    assert "issue_order 0" in lines and "half_fiber 0 1" in lines
    lines = samsung_reduce_plan(same, geometry, 1)["text"].splitlines()
    assert "issue_order 1" in lines and "half_fiber 0 0" in lines
    lines = samsung_reduce_plan(programs["bank_row"][1], geometry, 1)["text"].splitlines()
    assert "issue_order 0" in lines and "half_fiber 0 0" in lines
    assert split.half_fibers == (0, 1) and same.half_fibers == (0, 0)
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
    # Cost (round 37): the estimate includes the first staging from the
    # conductor's schedule, so the split layout, whose staging has no exposed
    # row change, is priced below bank_row and dual_fiber, and the default
    # selection picks it; the measured same-bank pair cost keeps the identity
    # pairing last.  The conductor agrees on both paper GEMVs and on a
    # held-out 37x129x2 contraction (VALIDATION round 37).
    estimates = {f: programs[f][0].estimate().cycles for f in programs}
    assert estimates["bank_row_rowpair"] > max(estimates["bank_row"], estimates["dual_fiber"], estimates["split_rows"])
    assert estimates["split_rows"] < estimates["bank_row"] < estimates["dual_fiber"]
    default = allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual")
    assert default.compiled.layout_ctx.programs[0].lever_record["mode"].split("+")[0] == "split_rows"


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


def test_the_selected_map_is_the_contract_not_a_label(monkeypatch):
    """Reviewer round 36: the half -> bank map must decide the packing and
    the addressing, be part of the decision identity, and an edited map or
    packing declaration must be rejected before selection."""
    from dataclasses import replace

    from allo.spmw_autoschedule import _matcher_placement_decision
    from allo.spmw_candidate import validate_candidate_effects
    from allo.spmw_samsung import samsung_half_fibers, samsung_matrix_packing

    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]
    target, buckets, candidates = _candidates(fn)
    default = allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual")
    matches = buckets[0][1]
    split = _vendor_levers(candidates, buckets, "split_rows")[0]
    identity_map = next(c.extra["half_bank_map"] for c in candidates if c.mode.split("+")[0] == "bank_row_rowpair")
    assert samsung_half_fibers(split) == (0, 1) and samsung_matrix_packing(split) == "half_bank:0,1"

    # 1. The map is part of the decision identity.
    before = _matcher_placement_decision(default.trace, split)
    edited = replace(split) if hasattr(split, "__dataclass_fields__") else split
    original_map = split.extra["half_bank_map"]
    split.extra["half_bank_map"] = identity_map
    try:
        after = _matcher_placement_decision(default.trace, split)
        assert before != after, "changing the selected map must change the search decision"
        # 2. An edited map disagrees with the declared packing: rejected
        # before costing by the shared validation, and by the lowering.
        problems = validate_candidate_effects(matches, split, target)
        assert any("half_bank:0,1" in p and "half_bank:0,0" in p for p in problems), problems
        with pytest.raises(Exception, match="map places them|declared packing|contract"):
            allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
                         layout=[split] * len(buckets))
    finally:
        split.extra["half_bank_map"] = original_map
    # 3. A bogus matrix packing declaration is rejected before selection.
    effects = split.extra["effects"]
    bogus = replace(
        effects,
        value_layouts=tuple(
            replace(v, packing="bogus-unrealized-packing")
            if v.storage == "banks" and v.value in effects.coverage.inputs else v
            for v in effects.value_layouts
        ),
    )
    split.extra["effects"] = bogus
    try:
        problems = validate_candidate_effects(matches, split, target)
        assert any("bogus-unrealized-packing" in p for p in problems), problems
        with pytest.raises(Exception, match="bogus-unrealized-packing|contract"):
            allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
                         layout=[split] * len(buckets))
    finally:
        split.extra["effects"] = effects
    # 4. A consistent candidate still compiles and realizes its map.
    compiled = allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
                            layout=[split] * len(buckets))
    assert compiled.compiled.layout_ctx.programs[0].half_fibers == (0, 1)


def test_supplied_declarations_must_match_axes_and_a_valid_map():
    """Reviewer round 37: a supplied declaration whose matrix axes were
    emptied (coverage still complete through the other values) is rejected
    at validation and at the compile boundary, and a map that does not keep
    the bank input as the tile's base fiber is refused."""
    from dataclasses import replace

    from allo.spmw_candidate import validate_candidate_effects
    from allo.spmw_linear_layout import LinearLayout
    from allo.spmw_samsung import samsung_half_fibers

    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]
    target, buckets, candidates = _candidates(fn)
    matches = buckets[0][1]
    split = _vendor_levers(candidates, buckets, "split_rows")[0]
    effects = split.extra["effects"]
    matrix = next(v for v in effects.value_layouts if v.storage == "banks" and v.value in effects.coverage.inputs)
    split.extra["effects"] = replace(
        effects,
        value_layouts=tuple(replace(v, axes=()) if v is matrix else v for v in effects.value_layouts),
    )
    try:
        problems = validate_candidate_effects(matches, split, target)
        assert any("laid out over axes ()" in p for p in problems), problems
        with pytest.raises(Exception, match="laid out over axes"):
            allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
                         layout=[split] * len(buckets))
    finally:
        split.extra["effects"] = effects
    original_map = split.extra["half_bank_map"]
    split.extra["half_bank_map"] = LinearLayout({"bank": [(0,)], "half": [(1,)]}, ("bank",), (2,))
    try:
        with pytest.raises(ValueError, match="base fiber"):
            samsung_half_fibers(split)
        problems = validate_candidate_effects(matches, split, target)
        assert any("selected map is invalid" in p or "base fiber" in p for p in problems), problems
        with pytest.raises(Exception, match="base fiber|contract"):
            allo.compile(fn, target, samsung_cost, reduction_order="reassociate", backend="virtual",
                         layout=[split] * len(buckets))
    finally:
        split.extra["half_bank_map"] = original_map
