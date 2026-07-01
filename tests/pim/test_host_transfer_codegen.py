# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codegen + cost-rehome wiring for the host-transfer surface (spec 001/003).

Covers the task-003 seam:
  * `SamsungCtx.drain` emits the ISA-valid NOP+WRITE drain (a `NOP`), NOT the
    `MOV bank<-GRF` the `_crf_valid` filter rejects -- so the filter now drops
    nothing (SPEC-005 §8 followup resolved);
  * `SamsungCtx.{program_crf,host_broadcast,host_scatter,host_gather}` record
    bookkeeping into `host_moves_emitted` and emit no compute `PIMCmd`;
  * `compile_for_target(host_moves=...)` resolves the records and stores them on
    `Compiled.host_moves` (default `None` -> empty, today's inferred path);
  * the resolved moves are stamped onto the placement `extra["host_moves"]` for
    the host_staging provenance cross-check.

Pure-Python (no simulator) so it runs under pim-dev without a board.
"""

from __future__ import annotations

import allo
from allo.pim import targets as T
from allo.spmw_codegen import SamsungCtx, ResolvedHostMove
from allo.spmw_target import HostMoveRecord, broadcast, scatter, gather, move_only


def test_drain_emits_nop_not_bank_mov():
    sam = T.build_samsung_target()
    ctx = SamsungCtx(sam)
    sam.move("ST_A").emit(ctx)
    sam.move("ST_B").emit(ctx)
    # Drain is a NOP marker; the conductor issues the WRITE column command.
    assert [c.type_ for c in ctx.cmds] == ["NOP", "NOP"]
    # None of the drain cmds is the rejected MOV bank<-GRF.
    for c in ctx.cmds:
        assert not (
            c.dst_ in ("EVEN_BANK", "ODD_BANK")
            and any(s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_, c.src2_))
        )


def test_drain_is_isa_valid_so_crf_filter_drops_nothing():
    # The run-side _crf_valid filter strips MOV/FILL bank<-GRF. With drain
    # emitting NOP, the filter must find nothing to drop.
    from allo.spmw_codegen import PIMCmd

    sam = T.build_samsung_target()
    ctx = SamsungCtx(sam)
    sam.move("ST_B").emit(ctx)

    def _crf_valid(c):
        if c.type_ in ("MOV", "FILL"):
            bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
            grf_src = any(s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_, c.src2_))
            if bank_dst and grf_src:
                return False
        return True

    assert all(_crf_valid(c) for c in ctx.cmds)


def test_host_hooks_record_no_compute_cmd():
    sam = T.build_samsung_target()
    ctx = SamsungCtx(sam)
    for nm in ("PROGRAM_CRF", "SCATTER_BANKS", "BCAST_GRF_A", "GATHER_BANKS"):
        sam.move(nm).emit(ctx)
    # No compute PIMCmd was emitted by the host hooks.
    assert ctx.cmds == []
    kinds = [k for k, _ in ctx.host_moves_emitted]
    assert kinds == ["program_crf", "scatter", "broadcast", "gather"]
    # The recorded device handles are the identical tree objects.
    handles = [h for _, h in ctx.host_moves_emitted]
    assert handles[0] is sam.crf
    assert handles[1] is sam.banks
    assert handles[2] is sam.grf_a
    assert handles[3] is sam.banks


def test_compile_for_target_default_host_moves_empty():
    # Without host_moves= the Compiled carries an empty list (today's path).
    # Use a trivially-empty trace via the resolver directly to avoid a full
    # customize() in the unit test.
    from allo.spmw_codegen import _resolve_host_moves

    sam = T.build_samsung_target()
    assert _resolve_host_moves(sam, None) == []
    assert _resolve_host_moves(sam, []) == []


def test_resolve_host_moves_builds_resolved_records():
    sam = T.build_samsung_target()
    recs = [
        HostMoveRecord(scatter, ("A", allo.host_xfer.banks)),
        HostMoveRecord(broadcast, ("x", allo.host_xfer.grf_a)),
        HostMoveRecord(gather, ("out", allo.host_xfer.banks)),
    ]
    from allo.spmw_codegen import _resolve_host_moves

    resolved = _resolve_host_moves(sam, recs)
    assert [r.move.name for r in resolved] == [
        "SCATTER_BANKS",
        "BCAST_GRF_A",
        "GATHER_BANKS",
    ]
    assert [r.verb.name for r in resolved] == ["scatter", "broadcast", "gather"]
    assert [r.buffer_role for r in resolved] == ["A", "x", "out"]
    # device_handle is the identical tree object.
    assert resolved[0].device_handle is sam.banks
    assert resolved[1].device_handle is sam.grf_a
    assert resolved[2].device_handle is sam.banks


def test_stamp_host_moves_attaches_to_placement_extra():
    from allo.spmw_codegen import _stamp_host_moves
    from allo.spmw_autoschedule import Placement

    pl = Placement(placements={})
    moves = [ResolvedHostMove(scatter, object(), object(), "A")]
    _stamp_host_moves(pl, moves)
    assert pl.extra["host_moves"] is moves
