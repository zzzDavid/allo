# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release ledger X-43: the AiM row allocator's footprint covers the realized
mapping, and the realization check refuses a record whose matrix rows leave
the region reserved for its buffer.

The footprint used to come from the matrix buffer's dimensions alone; a
broadcast matrix (no batch coordinate) therefore reserved one row while the
channels-mapped realization of a 7-vector contraction over 48 rows addressed
two (the reviewer's probe, `reviewer-r100/grid_allocation.py`).  The
footprint is now the larger of the buffer's extents and the realized shape
per role, both storage orientations are kept, and every lowered record's
``matrix_row_range`` is checked against the allocator's regions before a
candidate is accepted."""

from __future__ import annotations

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target

M, K, B, CH = 48, 320, 7, 3
R = M // CH


@region()
def left(W: bfloat16[M, K], X: bfloat16[B, K], Y: bfloat16[B, M]):
    @allo.work(mapping=[CH], args=[W, X, Y])
    def dot(a: bfloat16[M, K], x: bfloat16[B, K], c: bfloat16[B, M]):
        w = allo.get_wid()
        for bb in range(B):
            for i in range(R):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[w * R + i, k] * x[bb, k]
                c[bb, w * R + i] = acc


@region()
def right(W: bfloat16[M, K], X: bfloat16[B, K], Y: bfloat16[B, M]):
    @allo.work(mapping=[CH], args=[W, X, Y])
    def dot(a: bfloat16[M, K], x: bfloat16[B, K], c: bfloat16[B, M]):
        w = allo.get_wid()
        for bb in range(B):
            for i in range(R):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += x[bb, k] * a[w * R + i, k]
                c[bb, w * R + i] = acc


M2, K2 = 32, 256


@region()
def two_kernels(
    W: bfloat16[M, K],
    X: bfloat16[B, K],
    Y: bfloat16[B, M],
    W2: bfloat16[M2, K2],
    v: bfloat16[K2],
    y2: bfloat16[M2],
):
    @allo.work(mapping=[CH], args=[W, X, Y])
    def dot(a: bfloat16[M, K], x: bfloat16[B, K], c: bfloat16[B, M]):
        w = allo.get_wid()
        for bb in range(B):
            for i in range(R):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[w * R + i, k] * x[bb, k]
                c[bb, w * R + i] = acc

    @allo.work(mapping=[1], args=[W2, v, y2])
    def gemv(a2: bfloat16[M2, K2], b2: bfloat16[K2], c2: bfloat16[M2]):
        for i in range(M2):
            acc: bfloat16 = 0
            for k in range(K2):
                acc += a2[i, k] * b2[k]
            c2[i] = acc


def _compile(workload, **kwargs):
    return allo.compile(
        workload,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        reduction_order="reassociate",
        **kwargs,
    )


def _rows(compiled):
    """``(regions, [(row range, allocated rows, mapping)])`` of the artifact."""
    ctx = compiled.compiled.layout_ctx
    ranges = [
        (
            tuple(e["matrix_row_range"]),
            e["allocated_matrix_rows"],
            e.get("batch_mapping"),
        )
        for e in ctx.lowering_manifest
        if isinstance(e, dict) and e.get("matrix_row_range")
    ]
    return [dict(r) for r in ctx.row_regions], ranges


def _assert_within(regions, ranges):
    for (begin, end), allocated, _mapping in ranges:
        assert end - begin == allocated
        region = next(r for r in regions if r["row"] == begin)
        assert end <= region["row"] + region["rows"], (regions, ranges)


@pytest.mark.parametrize(
    "workload", [left, right], ids=["matrix_first", "vector_first"]
)
def test_default_x_resident_orientation_reserves_the_rows_it_addresses(workload):
    compiled = _compile(workload)
    regions, ranges = _rows(compiled)
    assert len(ranges) == 1
    (begin, end), allocated, mapping = ranges[0]
    # x resident: 48 batch vectors over the channels need two matrix-row
    # groups; the region reserves them (the probe found one reserved).
    assert mapping == "channels" and allocated == 2
    _assert_within(regions, ranges)
    assert compiled.placement_ranking.kernels[0].rejections == ()
    selected = next(
        pl
        for pos, pl in compiled.placement_ranking.kernels[0].candidates
        if pos == compiled.placement_ranking.kernels[0].chosen_position
    )
    assert selected.extra["aim_orientation"]["matrix"] == "x"


def test_forced_a_resident_orientation_reserves_its_rows_too():
    compiled = _compile(left, layout="aim_bank_operand=a")
    regions, ranges = _rows(compiled)
    assert ranges[0][1] == 3 and ranges[0][2] == "channels"
    _assert_within(regions, ranges)


def _forced(compiled, substring):
    kernel = compiled.placement_ranking.kernels[0]
    candidates = [pl for _pos, pl in kernel.candidates if substring in pl.mode]
    assert candidates, substring
    return candidates


@pytest.mark.parametrize("knob", ["batch_mapping=row_packed", "reduction_partitions=2"])
def test_other_mappings_reserve_their_rows(knob):
    """Row-packed and reduction-partitioned realizations of both orientations
    (each candidate forced through ``layout=``) allocate what they address."""
    compiled = _compile(left)
    seen = 0
    for placement in _forced(compiled, knob)[:6]:
        try:
            forced = _compile(left, layout=[placement] * CH)
        except Exception as error:  # noqa: BLE001 - an unrealizable knob mix
            assert "realization" not in str(error).lower() or "reserved" not in str(
                error
            )
            continue
        regions, ranges = _rows(forced)
        assert ranges
        _assert_within(regions, ranges)
        seen += 1
    assert seen >= 1, knob


def test_two_live_bank_allocations_do_not_overlap():
    """A second contraction in the same program gets its own region; each
    record's rows lie in its region and the regions are disjoint."""
    compiled = _compile(two_kernels)
    regions, ranges = _rows(compiled)
    assert len(ranges) == 2 and len(regions) >= 2
    _assert_within(regions, ranges)
    spans = sorted((r["row"], r["row"] + r["rows"]) for r in regions)
    for (_b0, e0), (b1, _e1) in zip(spans, spans[1:]):
        assert e0 <= b1, spans


def test_realization_check_refuses_rows_outside_the_region(monkeypatch):
    """With the footprint forced back to the buffer-derived extent, the
    realization check refuses every under-allocated channels-mapped record
    before selection: those candidates are recorded as rejections naming
    the reserved region, and the selected program is one whose rows fit."""
    from allo import spmw_aim

    original = spmw_aim._AimProgramDerivation._footprint

    def buffer_only(self, group, key, usage):
        if usage.kind != "slot" and usage.contraction["mapping"] == "channels":
            return 1
        return original(self, group, key, usage)

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "_footprint", buffer_only)
    compiled = _compile(left)
    kernel = compiled.placement_ranking.kernels[0]
    refused = [r for r in kernel.rejections if "the row allocator reserved" in r[1]]
    assert refused, kernel.rejections[:3]
    regions, ranges = _rows(compiled)
    _assert_within(regions, ranges)
