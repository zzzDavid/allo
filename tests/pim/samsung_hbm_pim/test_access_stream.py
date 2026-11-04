# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executable access stream of a lowered Samsung REDUCE program.

``samsung_access_stream`` derives, from the program ``allo.compile``
materialized (``binds``, ``tile_fiber``, ``k_tiles``, ``rows_per_block``)
and the simulated geometry, the transaction stream the conductor issues
during its passes: one round per barrier, every MAC group's ``E`` columns
with the row carry of ``addrGenSafe`` (32 burst columns per row, so a
64-column tile spans two physical rows).  The conductor can dump the same
stream (``TENON_CONDUCTOR_ADDRESS_TRACE``); the gated test compares the two
byte for byte for both fiber placements of two paper GEMVs.
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
    samsung_access_trace_text,
    samsung_conductor_trace_text,
    samsung_vendor_gemv_layout,
)

_SPEC = importlib.util.spec_from_file_location(
    "samsung_gemv_workload", Path(__file__).with_name("gemv") / "workload.py"
)
_WORKLOAD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKLOAD)


def _placements(fn):
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
    bank_row = next(
        c
        for c in candidates
        if c.mode.split("+")[0] == "bank_row"
        and set((c.extra.get("grf_residency") or {}).values()) == {"host"}
        and c.extra.get("crf_issue") == "shared"
        and not c.extra.get("stage_resident")
    )
    return target, {
        "dual_fiber": samsung_vendor_gemv_layout(target, default.trace),
        "bank_row": [bank_row] * len(buckets),
    }


def _compile(fn, target, layout, backend):
    return allo.compile(
        fn,
        target,
        samsung_cost,
        reduction_order="reassociate",
        backend=backend,
        layout=layout,
    )


def test_access_stream_follows_the_fiber_placement_with_the_row_carry():
    fn, _m, _k = _WORKLOAD.CASES["gemv_m8192_k256"]  # two reduction tiles
    target, layouts = _placements(fn)
    streams = {}
    for name, layout in layouts.items():
        compiled = _compile(fn, target, layout, "virtual")
        rounds = samsung_access_stream(compiled, 1, SAMSUNG_DEFAULT_GEOMETRY)
        streams[name] = (rounds, samsung_access_stream_summary(rounds))
    for name, (rounds, summary) in streams.items():
        # Two passes of 16 MAC groups (8 columns each), a host fill per tile
        # and one drain per pass; every round is closed by a barrier.
        assert summary["rounds_by_kind"] == {"DRAIN": 2, "HOST_FILL": 4, "MAC": 32}
        assert summary["transactions"] == 32 * 8 + 2 * 8
        assert all(len(r.accesses) == 8 for r in rounds if r.kind == "MAC")
    # bank_row keeps both tiles in the even fiber: the 64-column tiles span
    # rows 0-3 in pass 0 and 4-7 in pass 1 of bank 0, seven row changes.
    _rounds, bank_row = streams["bank_row"]
    assert bank_row["rows_by_bank"][0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert bank_row["mac_same_bank_row_transitions"] == {0: 7}
    # dual_fiber alternates fibers: rows 0-3 of both banks, three changes each.
    dual_rounds, dual = streams["dual_fiber"]
    assert dual["rows_by_bank"][0] == [0, 1, 2, 3]
    assert dual["rows_by_bank"][1][:4] == [0, 1, 2, 3]
    assert dual["mac_same_bank_row_transitions"] == {0: 3, 1: 3}
    first = [
        (r.fiber, r.accesses[0].row, r.accesses[0].column)
        for r in dual_rounds
        if r.kind == "MAC"
    ][:9]
    assert first == [
        (0, 0, 0), (0, 0, 8), (0, 0, 16), (0, 0, 24),
        (0, 1, 0), (0, 1, 8), (0, 1, 16), (0, 1, 24),
        (1, 0, 0),
    ]
    text = samsung_access_trace_text(dual_rounds)
    lines = text.splitlines()
    assert len(lines) == 32 * 8 and lines[0] == "0 0 0 0" and lines[8] == "1 0 0 8"
    conductor = samsung_conductor_trace_text(dual_rounds)
    assert conductor.startswith("PASS 0 0\nWhost_fill_grf_a x8\nBARRIER\nMAC 0 0 0 0\n")


@pytest.mark.parametrize("case", ["gemv_m8192_k256", "gemv_m4096_k512"])
def test_derived_stream_equals_the_conductors_dump(
    case, samsung_sim_gate, tmp_path, monkeypatch
):
    fn, M, K = _WORKLOAD.CASES[case]
    target, layouts = _placements(fn)
    rng = np.random.default_rng(11)
    W = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    x = (rng.standard_normal(K) * 0.05).astype(np.float16)
    for name, layout in layouts.items():
        dump = tmp_path / f"{case}-{name}.txt"
        monkeypatch.setenv("TENON_CONDUCTOR_ADDRESS_TRACE", str(dump))
        compiled = _compile(fn, target, layout, None)
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        assert result.kind == "simulator"
        np.testing.assert_allclose(
            y.astype(np.float32),
            W.astype(np.float32) @ x.astype(np.float32),
            atol=2e-2,
        )
        text = dump.read_text()
        if "PASS 0 0" not in text:
            pytest.skip("the installed pim_driver predates the address trace; rebuild it")
        start = text.index("PASS 0 0")
        end = text.index("readback", start)
        derived = samsung_conductor_trace_text(samsung_access_stream(compiled, 1))
        assert text[start:end] == derived, name
