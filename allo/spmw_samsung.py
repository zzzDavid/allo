# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung HBM-PIM codegen context and PIMSimulator runtime (moved from spmw_codegen)."""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
from pathlib import Path

from .spmw_match import MatchedOp
from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    PIMCmd,
    CodegenContext,
    _bank_fiber_class,
    _parse_loop_bound,
    _bucket_by_work_id,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import pimsim_root as _pimsim_root
from .spmw_simenv import samsung_unavailable_reason as _samsung_unavailable_reason


class SamsungCtx(CodegenContext):
    """Codegen ctx for Samsung HBM-PIM.

    Translates Tenon `Register` / `MemoryRef` handles into Samsung's
    `PIMOpdType` enum names (as strings), infers the is_auto column-
    strobe flag from operand shapes, and accumulates the result as a
    `list[PIMCmd]`.

    Recognized opcodes: "MAC", "MUL", "ADD", "MAD", "MOV", "FILL",
    "NOP", "JUMP", "EXIT" — i.e. the `PIMCmdType` enum from PIMCmd.h.
    """

    _REG_TO_OPD = {"grf_a": "GRF_A", "grf_b": "GRF_B"}

    def __init__(self, target):
        super().__init__(target)
        # Host-move bookkeeping (spec 001 D5): the host-scope moves' `emit`
        # closures (`host_broadcast`/`host_scatter`/`host_gather`/`program_crf`)
        # append `(hook, device_handle)` records here -- parallel to `cmds`,
        # consumed by the cost model for `host_staging`, NOT by the driver argv
        # (Q2 option a: the driver keeps doing the transfer; these moves carry
        # cost + operand-role only). Empty for every cell that does not yet
        # express `host_xfer.*`.
        self.host_moves_emitted: list[tuple[str, object]] = []

    # ------------------------------------------------------------------ #
    # Host-transfer ctx hooks (spec 001 D5). These are bookkeeping hooks the
    # host-scope moves' `emit` closures call; none emit a compute `PIMCmd`.
    # The driver still performs the real preload/readback over PIM_REG_RA, so
    # the cmd stream that reaches `programCrf` is unchanged.
    # ------------------------------------------------------------------ #

    def drain(self, dst=None, src0=None):
        """GRF->bank writeback as a NOP+WRITE column-command drain.

        ISA-1.0 forbids a bank dst with a GRF src (PIMCmd.cpp:73-97), so this
        is NOT a `MOV ODD_BANK<-GRF_B`. The proven drain (PIMRank.cpp:474-487,
        samsung-pim-isa.md) is a `NOP` in the CRF stream; the conductor issues
        the WRITE column command host-side. Emitting the NOP form directly means
        the run-side `_crf_valid` filter (which strips the rejected MOV) now
        finds nothing to drop -- resolving the SPEC-005 §8 followup. We keep the
        canonical eight-cycle pipe hold (`NOP` plus loopCounter 7).  The
        faithful GEMV conductor passes this emitted stream directly to
        ``programCrf``; omitting the hold produces an invalid drain schedule.
        """
        self.cmds.append(PIMCmd(type_="NOP", loopCounter_=7))

    def program_crf(self, crf_handle):
        """Record that the <=32-word CRF microcode is uploaded for this group.

        Host upload bookkeeping (the `STAGE_CRF` carrier); it does NOT emit a
        compute `PIMCmd`. The CRF is already conveyed via `--cmds` (ELTWISE) or
        the REDUCE minimal `--crf`, so this is a no-op on the cmd stream.
        """
        self.host_moves_emitted.append(("program_crf", crf_handle))

    def host_broadcast(self, handle):
        """Record a host->device broadcast (GRF_A/GRF_B/SRF). Bookkeeping +
        operand-role tag; the driver performs the HAB broadcast."""
        self.host_moves_emitted.append(("broadcast", handle))

    def host_scatter(self, handle):
        """Record a host->device per-bank scatter (the weight preload)."""
        self.host_moves_emitted.append(("scatter", handle))

    def host_gather(self, handle):
        """Record a device->host gather (the output readback)."""
        self.host_moves_emitted.append(("gather", handle))

    def _fiber_stride(self):
        """Bank fiber count (= banks-per-pim stride) for the active placement,
        read from the carried `LinearLayout`'s segment axis (D3) or the
        materialised fiber list, else `None` (the index's own coefficient is
        used). The layout object is load-bearing: it is what determined how
        many fibers the bank axis was swizzled into."""
        pl = self._active_placement
        if pl is None:
            return None
        layout = getattr(pl, "layout", None)
        axis = (getattr(pl, "extra", {}) or {}).get("fiber_axis")
        if layout is not None and axis is not None:
            try:
                return layout.size_of(axis)
            except Exception:
                pass
        fibers = (getattr(pl, "extra", {}) or {}).get("fibers")
        if fibers:
            return len(fibers)
        return None

    def _opd(self, handle):
        """Return (opd_name, idx) for a Tenon handle, or ("A_OUT", 0)."""
        if handle is None:
            return ("A_OUT", 0)
        if isinstance(handle, Register):
            opd = self._REG_TO_OPD.get(handle.name)
            if opd is None:
                raise NotImplementedError(
                    f"SamsungCtx: register {handle.name!r} has no PIMOpdType "
                    "mapping. Only grf_a/grf_b are supported."
                )
            return (opd, 0)
        if isinstance(handle, MemoryRef):
            # The bank-class name comes from the `range(stride)` fiber walk
            # (SPEC-022 D3). `stride` is read off the carried LinearLayout's
            # segment (fiber) axis when one is present; otherwise it is the
            # index's own coefficient (self-describing). The F2 layout object,
            # not a pattern matcher, is what determined the fiber geometry.
            stride = self._fiber_stride()
            cls = _bank_fiber_class(handle.idx, stride)
            if cls is None:
                raise NotImplementedError(
                    f"SamsungCtx: memref index {handle.idx!r} is not the "
                    "canonical stride*pid + r fiber form. Real layout "
                    "decisions are the autoscheduler's job (Blocker 4)."
                )
            return (cls, 0)
        raise NotImplementedError(
            f"SamsungCtx: unknown handle type {type(handle).__name__}."
        )

    def cmd(self, name, dst=None, src0=None, src1=None, is_relu=0):
        dst_name, dst_idx = self._opd(dst)
        src0_name, src0_idx = self._opd(src0)
        src1_name, src1_idx = self._opd(src1)
        # Samsung's "is_auto" column-strobe modifier: set when src1 is
        # bank-shaped (the bank stream is read in burst mode while the
        # GRF source stays put). Inferred per report 16 §"is_auto".
        is_auto = 1 if isinstance(src1, MemoryRef) else 0
        self.cmds.append(
            PIMCmd(
                type_=name,
                dst_=dst_name,
                dstIdx_=dst_idx,
                src0_=src0_name,
                src0Idx_=src0_idx,
                src1_=src1_name,
                src1Idx_=src1_idx,
                isAuto_=is_auto,
                isRelu_=is_relu,
            )
        )

    # Read-only roles emit LD only; read-write roles emit LD + ST.
    # `acc` starts at 0 in the GEMV workload IR so its LD half is a
    # zero-init, not a real memory move -- see spec 009 §E rule 3.
    # TODO(task-015): revisit when accumulators carry across calls.
    _ROLE_IS_READ_ONLY = {"x": True, "y": True, "acc": False, "dst": False}

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        # dst_handle is the placement-side handle. For Samsung that's
        # either grf_a / grf_b (LD/ST target) or a bank-shaped MemoryRef
        # (in-place; no LD/ST needed).
        if isinstance(dst_handle, MemoryRef):
            return (None, None)
        if isinstance(dst_handle, Register):
            if dst_handle.name == "grf_a":
                ld, st = "LD_A", "ST_A"
            elif dst_handle.name == "grf_b":
                ld, st = "LD_B", "ST_B"
            else:
                return (None, None)
            read_only = self._ROLE_IS_READ_ONLY.get(role, False)
            if read_only:
                return (ld, None)
            # acc carries no prior value in the GEMV workload IR; emit
            # only the storeback for now (see spec 009 §E rule 3).
            if role == "acc":
                return (None, st)
            return (ld, st)
        return (None, None)

    def resolve_spill_moves(self, tier, home_handle, *, n_entries=1):
        # Samsung spills to a bank row. The LD/ST pair keys off the home
        # register side (grf_a -> LD_A/ST_A, grf_b -> LD_B/ST_B) -- the
        # same `side` switch the spill cost factory prices.
        if tier != "bank_row":
            return super().resolve_spill_moves(tier, home_handle, n_entries=n_entries)
        if isinstance(home_handle, Register) and home_handle.name == "grf_b":
            return ("LD_B", "ST_B")
        return ("LD_A", "ST_A")

    # Active placement + resolved bindings for the current match, set by
    # `_walk_and_emit` before each compute emit. Default None keeps the
    # single-fiber path (no dual-fiber state) unchanged.
    _active_placement = None
    _active_bindings = None

    def after_match(self, match, n_emitted):
        # Inner-K JUMP -- the loop counter is computed off the
        # just-emitted MAC body, so the call must stay inside the
        # work-id window between compute and storeback.
        placement = self._active_placement
        fibers = tuple(getattr(placement, "extra", {}).get("fibers", ()))
        if placement is not None and len(fibers) > 1 and match.target_op_name == "MAC":
            self._emit_dual_fiber_jumps(match, n_emitted)
            return
        _emit_inner_loop_jump(match, self, n_emitted)

    def _emit_dual_fiber_jumps(self, match, n_emitted):
        """Materialise the alternating (JUMP even, MAC odd, JUMP odd, ...)
        tail for a dual-fiber MAC.

        The walker already emitted the canonical MAC against the EVEN
        fiber (`placements[y]`). Here we close fiber 0's inner loop with
        its split JUMP, then for each later fiber emit one MAC (same
        dst/src0, src1 = that fiber's bank handle so `_bank_fiber_class`
        stamps the per-fiber bank class) followed by its own split JUMP. Trip
        counts come from `inner_ub // lanes` split across the fibers — no
        shape literal. This walks `range(n_fibers)`, so it already generalizes
        to a stride>2 fiber count with no structural change (SPEC-022 D3).
        """
        if not match.enclosing_loops:
            return
        ub = _parse_loop_bound(match.enclosing_loops[-1][2])
        if ub is None:
            return
        folded = ub // _samsung_lane_burst(self)
        fibers = self._active_placement.extra.get("fibers", [])
        n_fibers = len(fibers)
        if n_fibers <= 1:
            _emit_inner_loop_jump(match, self, n_emitted)
            return
        split = _fiber_fold_split(folded, n_fibers)

        bindings = self._active_bindings or {}
        mac_dst = bindings.get("acc")
        mac_src0 = bindings.get("x")

        def emit_jump(trips):
            # First MAC of the fiber is iteration 0; JUMP loops the rest.
            loop_counter = trips - 1
            if loop_counter <= 0:
                return
            self.cmds.append(
                PIMCmd(
                    type_="JUMP",
                    loopCounter_=loop_counter,
                    loopOffset_=n_emitted + 1,
                )
            )

        # Fiber 0: the EVEN MAC the walker already emitted; just its JUMP.
        emit_jump(split[0])
        # Fibers 1..n-1: MAC against the fiber's bank handle, then JUMP.
        for i in range(1, n_fibers):
            self.cmd("MAC", dst=mac_dst, src0=mac_src0, src1=fibers[i])
            emit_jump(split[i])


def _samsung_lane_burst(ctx) -> int:
    """Samsung GRF lane count read from the target geometry.

    Each MAC consumes this many fp16 elements per K-iteration, so the
    inner-K loop folds `K` MACs into `K // lanes`. Sourced from
    `target.grf_a.lanes` (the report-16 target spec, `lanes=8`); no
    shape literal.
    """
    return ctx.target.grf_a.lanes


def _fiber_fold_split(folded: int, n_fibers: int) -> list[int]:
    """Round-robin split of `folded` burst-tiles across `n_fibers` banks.

    Fiber `i` gets `ceil` for the first `folded % n_fibers` fibers and
    `floor` after, so the busier fiber stays bounded (matches the cost
    model's ceil `per_fiber`). For folded=128, n=2 -> [64, 64]; for an
    odd folded=127, n=2 -> [64, 63].
    """
    return [(folded + (n_fibers - 1 - i)) // n_fibers for i in range(n_fibers)]


def _emit_inner_loop_jump(
    match: MatchedOp, ctx: CodegenContext, n_emitted: int
) -> None:
    """Append a Samsung-style JUMP that folds the innermost reduction
    loop.

    Loop counter is `(inner_ub // lanes) - 1` (the first MAC counts as
    iteration 0; JUMP loops back the rest). Loop offset is the number of
    instructions that comprise one iteration body — i.e. `n_emitted` for
    the compute ops the match emitted, plus 1 for the JUMP itself.
    """
    if not match.enclosing_loops:
        return
    inner = match.enclosing_loops[-1]
    ub = _parse_loop_bound(inner[2])
    if ub is None:
        return
    loop_counter = ub // _samsung_lane_burst(ctx) - 1
    if loop_counter <= 0:
        return
    ctx.cmds.append(
        PIMCmd(
            type_="JUMP",
            loopCounter_=loop_counter,
            loopOffset_=n_emitted + 1,
        )
    )


def _is_samsung_storeback_mov(c: PIMCmd) -> bool:
    """A bank<-GRF store MOV that closes a work-id (the `ST_A`/`ST_B`
    storeback). `acc -> grf_b` is never host-eligible (lever 2), so this
    MOV is always present and is the residency-robust work-id delimiter."""
    if c.type_ not in ("MOV", "FILL"):
        return False
    bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
    grf_src = any(s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_))
    return bank_dst and grf_src


def _split_samsung_layers(cmds: list[PIMCmd]) -> list[list[PIMCmd]]:
    """Group a flat Samsung GEMV cmd stream into per-layer (work-id)
    chunks, closed by the storeback MOV that ends each work-id.

    NOTE (SPEC-05, 2026-06-30): the legacy multi-layer GEMV RUN path that consumed
    this was deleted. It is KEPT because the layer-split + FFN cost-model tests
    (test_samsung_loop_012, test_samsung_multi_layer_split) assert its behaviour;
    it is a pure cmd-stream analysis helper (no run-path coupling).

    A layer body is `[LD MOV?, (MAC, JUMP)+, ST MOV]`; under lever 1 the
    `(MAC, JUMP)` part repeats once per bank fiber (so we must NOT split
    at JUMP), and under lever 2 the opening LD MOV is *absent* for
    host-resident preloads -- so the preload MOV is no longer a reliable
    boundary. The storeback MOV (`acc -> grf_b -> bank`, never
    host-eligible) always closes a work-id, so we split *after* it.
    Trailing setup/terminator (NOP/EXIT) with no MAC attaches to the
    last group; groups with no MAC are dropped.
    """
    groups: list[list[PIMCmd]] = []
    cur: list[PIMCmd] = []
    for c in cmds:
        cur.append(c)
        if _is_samsung_storeback_mov(c) and any(g.type_ == "MAC" for g in cur):
            # This storeback closes the current work-id's body.
            groups.append(cur)
            cur = []
    if cur:
        if groups and not any(g.type_ == "MAC" for g in cur):
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return [g for g in groups if any(c.type_ == "MAC" for c in g)]


def _write_samsung_cmds(path: Path, cmds: list[PIMCmd]) -> None:
    """Serialise a list of PIMCmd to the line-delimited format pim_driver
    accepts via `--cmds`. See `_run_samsung` docstring for the grammar.
    """
    with open(path, "w") as f:
        for c in cmds:
            parts = [c.type_]
            # Only emit fields that differ from PIMCmd's default ctor so
            # the trace stays human-greppable.
            if c.dst_ != "A_OUT":
                parts.append(f"dst={c.dst_}")
            if c.src0_ != "A_OUT":
                parts.append(f"src0={c.src0_}")
            if c.src1_ != "A_OUT":
                parts.append(f"src1={c.src1_}")
            if c.src2_ != "A_OUT":
                parts.append(f"src2={c.src2_}")
            if c.loopCounter_:
                parts.append(f"loop_counter={c.loopCounter_}")
            if c.loopOffset_:
                parts.append(f"loop_offset={c.loopOffset_}")
            if c.isAuto_:
                parts.append(f"is_auto={c.isAuto_}")
            if c.dstIdx_:
                parts.append(f"dst_idx={c.dstIdx_}")
            if c.src0Idx_:
                parts.append(f"src0_idx={c.src0Idx_}")
            if c.src1Idx_:
                parts.append(f"src1_idx={c.src1Idx_}")
            if c.isRelu_:
                parts.append(f"is_relu={c.isRelu_}")
            f.write(" ".join(parts) + "\n")


def _samsung_num_pim_blocks(compiled) -> int:
    """Inner `pim`-unit fanout (num PIM blocks per pseudo-channel) for the
    Samsung tile-size derivation. Read off the unit tree's innermost mapping
    factor (8 for the canonical fixture); falls back to 8 when unavailable so
    elems_per_tile never collapses to a sub-tile size."""
    try:
        factors, _ = compiled.target.work_grid()
        if factors:
            return factors[-1]
    except Exception:  # noqa: BLE001
        pass
    return 8


def _samsung_read_generic_outbin(out_path, n):
    """Flat fp16 readback for the GENERIC (ELTWISE) interpreter path: read `n`
    fp16 elements from out.bin, NO reduce-sum (unlike the REDUCE readback's
    per-element 16-lane tree reduction). Returns {"out": <fp32 array>} or {} when
    the blob is absent/empty/all-zero (CYCLES-ONLY graceful fallback)."""
    p = Path(out_path)
    if not p.exists():
        return {}
    import numpy as np

    raw = np.fromfile(str(p), dtype=np.float16)
    if raw.size == 0:
        return {}
    out = raw[:n].astype(np.float32)
    if not np.any(out):
        return {}
    return {"out": out}


# SPEC-04: the PIMSimulator physical fabric row-tile = num_total_pim_blocks_ *
# num_grfB_ = (64 channels * 8 PIM blocks) * 8 GRF_B = 4096. preloadGemv packs the
# weight assuming M is a multiple of this (the y-loop strides output_tile_size =
# num_grfB_*num_total_pim_blocks_); an M below it underfills the GRF_B slots and
# reads OOB weight rows. The REDUCE run path pads A's rows up to this multiple
# (zeros) and slices the real M back from the readback. It is a PIMSimulator
# config constant (NUM_PIM_BLOCKS=8, 64 chans hardcoded in pim_driver make_kernel,
# NUM_GRF=8), not a workload literal.
_SAMSUNG_FABRIC_ROW_TILE = 4096


def _samsung_reduce_row_tiling(rows: int) -> tuple[int, int, int]:
    """Return ``(padded_rows, rows_per_tile, num_tiles)`` for REDUCE.

    The GENERIC conductor interprets ``out_rows`` as a *per-tile* extent and
    multiplies it by ``num_tiles`` for readback.  Keeping the logical/padded
    total in ``out_rows`` therefore over-drains every multi-tile GEMV.
    """
    rows = int(rows)
    if rows <= 0:
        raise ValueError("Samsung REDUCE row count must be positive")
    rows_per_tile = _SAMSUNG_FABRIC_ROW_TILE
    num_tiles = (rows + rows_per_tile - 1) // rows_per_tile
    return rows_per_tile * num_tiles, rows_per_tile, num_tiles

# SPEC-04: pad the reduction extent K up to a multiple of this so computeGemv's
# input-tile split (num_input_tiles = ceil(ceil(K/16)/8)) yields >= 2 tiles and
# engages both even and odd banks. K below 256 (num_input_tiles==1) degenerates:
# the odd bank's JUMP counter underflows and getResultColGemv's end_col collapses,
# leaving the output undrained. 256 elems = 16 bursts = 2 input tiles. Zero-padding
# K leaves A@B unchanged.
_SAMSUNG_REDUCE_K_TILE = 256


def _samsung_reduce_partition(compiled, inputs):
    """SPEC-05 §2.1: derive the MAPPING-DRIVEN partition for a GENERIC_REDUCE run.

    Returns `(ROWS, n_workids, K, N, M_real)`:
      - ROWS      = per-PE output-row slice = bound(MAC enclosing_loops[-3]) for a
                    3-deep slice nest [i(ROWS), j(N), k(K)], or [-2] for a 2-deep
                    GEMV slice [i(ROWS), k(K)]. This is the partition primitive --
                    it tracks `mapping` (a different mapping -> a different ROWS).
      - n_workids = number of MAC work-id buckets = the realized PE count
                    (= prod(mapping) once the dataflow expander materializes the
                    grid). Cross-checked against target.work_grid()[1].
      - K, N      = contraction / output-col loop bounds.
      - M_real    = the true logical output rows = the operand A.shape[0] (the
                    un-padded P); M_logical = ROWS * n_workids is the partition's
                    total (>= M_real when the grid over-covers, e.g. P<128).

    The PARTITION (ROWS, n_workids) is the trace's, NOT operand-shape tiling:
    partition flows mapping -> trace -> codegen. Operand arrays still supply data
    + the ground-truth K/N for the zero-pad. Returns None when no MAC match.
    """
    import numpy as np

    trace = getattr(compiled, "trace", None)
    if trace is None:
        return None
    macs = [m for m in trace.matches if m.target_op_name == "MAC"]
    if not macs or not macs[0].enclosing_loops:
        return None
    mac = macs[0]
    depth = len(mac.enclosing_loops)
    K = _parse_loop_bound(mac.enclosing_loops[-1][2])
    if K is None:
        return None
    if depth >= 3:
        ROWS = _parse_loop_bound(mac.enclosing_loops[-3][2])
        N = _parse_loop_bound(mac.enclosing_loops[-2][2])
    elif depth == 2:
        ROWS = _parse_loop_bound(mac.enclosing_loops[-2][2])
        N = 1
    else:
        ROWS, N = 1, 1
    if ROWS is None or N is None:
        return None

    # n_workids = MAC work-id bucket count (the realized PE grid). Cross-check
    # against the declared grid; warn (gap-1 discipline), do not raise.
    buckets = [
        b
        for b in _bucket_by_work_id(trace)
        if any(m.target_op_name == "MAC" for m in b[1])
    ]
    n_workids = len(buckets)
    try:
        declared = compiled.target.work_grid()[1]
        if n_workids != declared:
            warnings.warn(
                f"Samsung REDUCE partition: trace has {n_workids} MAC work-id "
                f"buckets but target.work_grid() declares {declared} PEs; using "
                f"the trace count (mapping-driven).",
                stacklevel=2,
            )
    except Exception:  # noqa: BLE001
        pass
    if n_workids <= 0:
        n_workids = 1

    # M_real (the un-padded logical rows) from the operand; cross-check vs the
    # partition product (ROWS * n_workids) -- a mismatch beyond the grid pad is
    # advisory (the operand may legitimately be padded to the grid by the harness).
    a = inputs.get("A")
    if a is None:
        a = inputs.get("a")
    M_real = None
    if a is not None:
        a = np.asarray(a)
        if a.ndim == 2:
            M_real = int(a.shape[0])
    M_logical = int(ROWS) * int(n_workids)
    if M_real is None:
        M_real = M_logical
    elif M_real > M_logical:
        warnings.warn(
            f"Samsung REDUCE partition: operand rows {M_real} exceed the grid "
            f"partition ROWS*n_workids = {ROWS}*{n_workids} = {M_logical}; the "
            f"grid under-covers the operand (check mapping/ROWS).",
            stacklevel=2,
        )
    return (int(ROWS), int(n_workids), int(K), int(N), int(M_real))


def _samsung_read_reduce_outbin(out_path, M, N, m_pad):
    """SPEC-04 §4.3: REDUCE readback. The driver writes M_pad*N partial-sum bursts
    (16 lanes each), drained N-pass-major: pass n's M_pad outputs, then pass n+1's.
    So the blob reshapes (N, M_pad, 16) -> sum the 16 lanes -> (N, M_pad) -> .T ->
    (M_pad, N) -> slice [:M, :N]. For GEMV (N==1) this collapses to (M,). Graceful:
    empty/all-zero -> {} (CYCLES-ONLY), mirroring the GEMV readback contract.
    Surfaces "out" (GEMM, (M, N)) or "y" (GEMV, (M,)) so the harness finds it."""
    p = Path(out_path)
    if not p.exists():
        return {}
    import numpy as np

    raw = np.fromfile(str(p), dtype=np.float16)
    need = m_pad * N * 16
    if raw.size < need:
        return {}
    partials = raw[:need].reshape(-1, 16).astype(np.float32).sum(axis=1)  # (N*m_pad,)
    grid = partials.reshape(N, m_pad).T[:M, :N]  # (M, N)
    if not np.any(grid):
        return {}
    if N == 1:
        return {"y": grid.reshape(-1)}
    return {"out": grid}


def _samsung_run_reduce_driver(weight, vec, M, K, N):
    """SPEC-04 cross-stage core: run ONE GENERIC REDUCE contraction `weight @ vec`
    on real PIMSimulator and return `(readback_ndarray, cycles)`.

    `weight` is the (M, K) bank-resident operand (already transposed by the caller
    for an A^T stage); `vec` is the (K,) GEMV input or (K, N) GEMM second operand.
    Pads K to a multiple of 256 and M to the fabric row-tile (zeros, leaving the
    contraction unchanged), runs the driver, tree-reduces the 16-lane partial sums,
    and slices [:M, :N]. Returns `(C, cycles)` where C is (M,) for N==1 else (M, N);
    `(None, cycles)` if the readback is empty/all-zero; `(None, None)` if the driver
    is unavailable. This is the reusable engine behind both the single-stage
    GENERIC_REDUCE run path and `samsung_reduce_chain` (multi-stage threading)."""
    import numpy as np

    root = _pimsim_root()
    driver = root / "pim_driver"
    if not driver.exists():
        return None, None

    M, K, N = int(M), int(K), int(N)
    weight = np.asarray(weight, dtype=np.float16).reshape(M, K)
    if N > 1:
        second = np.asarray(vec, dtype=np.float16).reshape(K, N)
    else:
        second = np.asarray(vec, dtype=np.float16).reshape(K, 1)

    k_tile = _SAMSUNG_REDUCE_K_TILE
    k_pad = ((K + k_tile - 1) // k_tile) * k_tile
    if k_pad != K:
        w_k = np.zeros((M, k_pad), dtype=np.float16)
        w_k[:, :K] = weight
        weight = w_k
        s_k = np.zeros((k_pad, second.shape[1]), dtype=np.float16)
        s_k[:K] = second
        second = s_k
    m_pad, rows_per_tile, num_tiles = _samsung_reduce_row_tiling(M)
    w_pad = np.zeros((m_pad, k_pad), dtype=np.float16)
    w_pad[:M] = weight
    k_bursts = ((k_pad // 16) + 7) // 8

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        a_path = td_path / "A.npy"
        b_path = td_path / "B.npy"
        out_path = td_path / "out.bin"
        crf_path = td_path / "reduce.crf"
        np.save(a_path, w_pad)
        np.save(b_path, second)
        # The REDUCE path rebuilds the canonical GEMV CRF internally (matched JUMP
        # counters), but the driver's GENERIC parser still requires a non-empty
        # CRF. Supply a minimal MAC body; its JUMP counter is inert for REDUCE.
        crf_path.write_text(
            "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
            "JUMP loop_counter=0 loop_offset=2\n"
            "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
            "JUMP loop_counter=0 loop_offset=2\n"
            "MOV dst=ODD_BANK src0=GRF_B\n"
        )
        argv = [
            str(driver),
            "--op",
            "GENERIC",
            "--op-kind",
            "REDUCE",
            "--crf",
            str(crf_path),
            "--out",
            str(out_path),
            "--inputs",
            f"{a_path},{b_path}",
            "--out-rows",
            str(rows_per_tile),
            "--out-cols",
            str(N),
            "--reduce-k",
            str(k_pad),
            "--num-tiles",
            str(num_tiles),
            "--bank-type",
            "ALL",
            "--roles",
            f"A:0:0:2:0:0:0:{k_bursts},B:0:0:2:0:0:0:0,out:0:0:1:1:0:8:0",
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, cwd=str(root), timeout=600, check=False
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"Samsung pim_driver REDUCE invocation failed: {exc}"
            ) from exc
        combined = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode(
            "utf-8", "replace"
        )
        m = re.search(r"PIM_CYCLES total=(\d+)", combined)
        if not m:
            raise RuntimeError(
                "Samsung REDUCE driver missing PIM_CYCLES; tail: " + combined[-400:]
            )
        cycles = int(m.group(1))
        outputs = _samsung_read_reduce_outbin(out_path, M, N, m_pad)
    if not outputs:
        return None, cycles
    arr = outputs.get("out")
    if arr is None:
        arr = outputs.get("y")
    return np.asarray(arr), cycles


def run_samsung_reduce(weight, vec, M, K, N):
    """Public entry for the harness cross-stage chain: run one Samsung GENERIC
    REDUCE contraction `weight @ vec` and return `(readback_ndarray, cycles)`.
    See `_samsung_run_reduce_driver`. The caller owns the numpy reference and any
    host pre/post-pass (alpha/beta, transpose, centering)."""
    return _samsung_run_reduce_driver(weight, vec, M, K, N)


def execute_host_schedule_samsung(schedule, dev_inputs):
    """Execute an analyzed host schedule on Samsung HBM-PIM (the run-path hook).

    `schedule` is a `spmw_host_program.HostSchedule`; `dev_inputs` maps each
    external-input buffer name to its float16 device array. Runs one
    `run_samsung_reduce` per group -- a **batched (coalesced) group preloads its
    resident weight ONCE** and stacks its B input vectors as the N=B columns of a
    single contraction (the `P + B*(E+R)` schedule); a singleton group is a plain
    contraction at its own N. Threads each group's real device readback forward
    so a later group can consume a device-resident intermediate.

    Returns `(dev, total_cycles)` where `dev` maps every buffer name to its
    device value, or `(None, None)` if the driver is unavailable / surfaced
    nothing. Backend execution only: NO reference composition (that is the
    caller's concern), mirroring `run_samsung_reduce`.
    """
    import numpy as np

    dev = {k: np.asarray(v, dtype=np.float16) for k, v in dev_inputs.items()}
    total = 0
    for g in schedule.groups:
        M, K = g.weight_shape
        W = dev[g.weight]
        if not g.is_batched:
            lx = g.launches[0]
            out_dev, cyc = run_samsung_reduce(W, dev[lx.vec], M=M, K=K, N=lx.N)
            if cyc is None:
                return None, None
            total += cyc
            if out_dev is None:
                return None, total
            dev[lx.out] = np.asarray(out_dev, dtype=np.float16)
        else:
            # Batched GEMV: stack the B input vectors as the N columns of ONE
            # contraction -> a single preload of the resident weight.
            second = np.stack([dev[lx.vec] for lx in g.launches], axis=1)  # (K, B)
            out_dev, cyc = run_samsung_reduce(W, second, M=M, K=K, N=len(g.launches))
            if cyc is None:
                return None, None
            total += cyc
            if out_dev is None:
                return None, total
            out_arr = np.asarray(out_dev)
            for b, lx in enumerate(g.launches):
                dev[lx.out] = out_arr[:, b].astype(np.float16)
    return dev, total


def _assert_host_move_roles(compiled, inputs) -> None:
    """Assert the operand->role binding against the recorded host moves.

    spec 001 D5 (Q2 option a): when `compiled.host_moves` is non-empty the
    operand roles are *driven by the explicit moves* rather than purely inferred
    -- the scatter buffer is the weight, the broadcast buffer is the input, the
    gather buffer is the output. The driver still performs the transfer; this is
    a consistency check that the recorded role names correspond to supplied
    inputs. Advisory (warn, never raise): a buffer role may be unbound when the
    workload passed a positional array rather than a named label, in which case
    the run path's operand-shape inference is the fallback. When `host_moves` is
    empty (today's path / non-Samsung) this is a no-op.
    """
    moves = getattr(compiled, "host_moves", None)
    if not moves:
        return
    # Map verb name -> the buffer roles the moves bind it to.
    by_verb: dict[str, list[str]] = {}
    for rhm in moves:
        verb_name = getattr(rhm.verb, "name", str(rhm.verb))
        if rhm.buffer_role is not None:
            by_verb.setdefault(verb_name, []).append(rhm.buffer_role)
    supplied = {k for k, v in inputs.items() if v is not None}
    for verb_name, roles in by_verb.items():
        # A gather binds the OUTPUT buffer, which is legitimately absent from
        # `inputs` (it is the readback target, supplied separately or inferred);
        # only the input-side verbs (scatter weight / broadcast input) name
        # buffers that must appear among the supplied inputs.
        if verb_name == "gather":
            continue
        for role in roles:
            # A named role that names no supplied input is suspicious only when
            # SOME inputs were supplied (a bare `compiled.run()` probe supplies
            # none and is exempt).
            if supplied and role not in supplied:
                warnings.warn(
                    f"Samsung host move ({verb_name}) binds buffer role "
                    f"{role!r} but no input named {role!r} was supplied "
                    f"(supplied: {sorted(supplied)}); falling back to "
                    f"operand-shape inference for that operand.",
                    stacklevel=2,
                )


def _samsung_runtime_route(commands) -> str:
    op_types = {command.type_ for command in commands if isinstance(command, PIMCmd)}
    compute_types = op_types & {"MAC", "MUL", "ADD", "RELU"}
    return "GENERIC_REDUCE" if compute_types == {"MAC"} else "GENERIC"


def _samsung_main_runtime_command_valid(command: PIMCmd) -> bool:
    if command.type_ in ("MOV", "FILL"):
        bank_dst = command.dst_ in ("EVEN_BANK", "ODD_BANK")
        grf_src = any(
            source in ("GRF_A", "GRF_B")
            for source in (command.src0_, command.src1_, command.src2_)
        )
        if bank_dst and grf_src:
            return False
    return True


def _samsung_main_runtime_commands(commands) -> tuple[PIMCmd, ...]:
    return tuple(
        command
        for command in commands
        if isinstance(command, PIMCmd) and _samsung_main_runtime_command_valid(command)
    )


def _run_samsung(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled Samsung HBM-PIM artifact via `pim_driver`.

    SPEC-05 routing (legacy GEMV/ADD/MUL/RELU paths DELETED, 2026-06-30): a MAC
    kernel routes to the mapping-driven GENERIC REDUCE conductor; everything else
    routes to the GENERIC ELTWISE conductor. There is no dedicated --op
    GEMV/ADD/MUL/RELU branch and no multi-layer GEMV split.

    Wire protocol (SPEC-005): for the GENERIC ELTWISE path the emitted PIMCmd
    stream is serialised to a line-delimited `cmds.txt` and passed via
    ``pim_driver --cmds <path>``; the C++ side uses it as the CRF microcode
    (uploaded by `programCrf`) instead of regenerating it via
    `PIMCmdGen::getPIMCmds`. The static ``--op`` choice only selects which numpy
    inputs to wire up; the kernel choice no longer determines the CRF program
    (placement does). The GENERIC REDUCE path supplies its own minimal --crf
    (rebuilt canonical GEMV body) and does not pass --cmds.

    Line-delimited cmd format:
        MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1
        JUMP loop_counter=7 loop_offset=2
        NOP loop_counter=7
        EXIT
    """
    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    root = _pimsim_root()
    driver = root / "pim_driver"

    # spec 001 D5: assert operand->role binding against the recorded host moves
    # (no-op when none recorded; advisory when bound). The driver argv below is
    # unchanged -- the moves carry cost + role, not driver args (Q2 option a).
    _assert_host_move_roles(compiled, inputs)

    # The `--op` flag still picks the data-path scaffolding (eltwise vs
    # GEMV) and which numpy inputs to wire up, but no longer determines
    # the CRF microcode -- that comes from `compiled.cmds` via `--cmds`.
    # The choice of which scaffolding to use is inferred from MAC-vs-eltwise
    # opcodes in the emitted stream.
    kernel = _samsung_runtime_route(compiled.cmds)

    # numpy is a hard Tenon dependency; "no numpy" is a setup bug, not
    # an env skip -- let the ImportError propagate.
    import numpy as np

    # ISA-valid filter (drop the ST_A/ST_B storeback MOVs the C++ validationCheck
    # rejects) -> the cmd stream that reaches programCrf via --cmds for the GENERIC
    # ELTWISE path. (The GENERIC_REDUCE path supplies its own minimal CRF and
    # ignores this; see its argv branch.)
    pim_cmds_all = list(_samsung_main_runtime_commands(compiled.cmds))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_path = td_path / "out.bin"
        # SPEC-04 §3.2: GENERIC_REDUCE is not a new --op; it reuses --op GENERIC
        # with --op-kind REDUCE (added in the branch below). All other kernels map
        # their name 1:1 to --op.
        op_arg = "GENERIC" if kernel == "GENERIC_REDUCE" else kernel
        argv = [str(driver), "--op", op_arg, "--out", str(out_path)]
        # REDUCE state threaded to the readback (set in the GENERIC_REDUCE branch).
        reduce_shape = None  # (M, K, N, m_pad)

        if kernel == "GENERIC_REDUCE":
            # SPEC-05 §2: MAPPING-DRIVEN genuine fp16 matmul via --op GENERIC
            # --op-kind REDUCE. The PARTITION (per-PE slice ROWS, PE count
            # n_workids) comes from the TRACE (the work-id buckets + the MAC slice
            # loop bound), NOT from operand-shape tiling: partition flows
            # mapping -> trace -> codegen. M_logical = ROWS * n_workids; operands
            # supply data + ground-truth K/N. A is padded to the logical grid
            # (n_workids*ROWS) then to the physical fabric tile; --num-tiles =
            # ceil(M_logical/4096) tiles the logical grid across fabric tiles.
            part = _samsung_reduce_partition(compiled, inputs)
            if part is None:
                return RunResult(
                    cycles=None,
                    stdout="Samsung GENERIC_REDUCE: no MAC partition in trace",
                    backend="samsung_hbm_pim",
                )
            ROWS, n_workids, K, N, M_real = part
            M_logical = ROWS * n_workids
            # Operand arrays (real A/B/x). Accept A/B, a/b, x/in role names.
            A = inputs.get("A")
            if A is None:
                A = inputs.get("a")
            B = inputs.get("B")
            if B is None:
                B = inputs.get("b")
            xv = inputs.get("x")
            if xv is None:
                xv = inputs.get("in")
            if A is None:
                return RunResult(
                    cycles=None,
                    stdout="Samsung GENERIC_REDUCE needs A (and B for GEMM / x "
                    "for GEMV) kwargs",
                    backend="samsung_hbm_pim",
                )
            A = np.asarray(A, dtype=np.float16).reshape(int(M_real), int(K))
            # The second operand: B[K,N] (GEMM) or x[K] (GEMV).
            if N > 1:
                if B is None:
                    return RunResult(
                        cycles=None,
                        stdout="Samsung GENERIC_REDUCE GEMM needs B kwarg",
                        backend="samsung_hbm_pim",
                    )
                second = np.asarray(B, dtype=np.float16).reshape(int(K), int(N))
            else:
                src = B if B is not None else xv
                if src is None:
                    return RunResult(
                        cycles=None,
                        stdout="Samsung GENERIC_REDUCE GEMV needs x (or B) kwarg",
                        backend="samsung_hbm_pim",
                    )
                second = np.asarray(src, dtype=np.float16).reshape(int(K), 1)
            # Partition-provenance assertion (SPEC-05 §2.3): M_logical is the grid
            # product; the operand's real rows must fit within it (the harness pads
            # the operand to the grid).
            if M_real > M_logical:
                warnings.warn(
                    f"Samsung GENERIC_REDUCE: operand rows {M_real} exceed the "
                    f"mapping partition ROWS*n_workids = {ROWS}*{n_workids} = "
                    f"{M_logical}; the grid under-covers (check mapping).",
                    stacklevel=2,
                )
            # Pad K up to a multiple of _SAMSUNG_REDUCE_K_TILE (256) so computeGemv's
            # input-tile split engages both even and odd banks cleanly (K<256 /
            # odd num_input_tiles degenerates -- zero-padding K leaves A@B unchanged).
            k_tile = _SAMSUNG_REDUCE_K_TILE
            k_pad = ((int(K) + k_tile - 1) // k_tile) * k_tile
            # Pad A's rows: first to the LOGICAL grid (n_workids*ROWS = M_logical,
            # so the partition divides evenly), then to the PHYSICAL fabric tile
            # (preloadGemv requires it). The two pads compose (logical <= physical
            # for SMALL/milestone). The readback slices [:M_real, :N].
            m_pad, rows_per_tile, num_tiles = _samsung_reduce_row_tiling(M_logical)
            A_pad = np.zeros((m_pad, k_pad), dtype=np.float16)
            A_pad[: int(M_real), : int(K)] = A  # zero-fill grid pad + K pad
            second_pad = np.zeros((k_pad, second.shape[1]), dtype=np.float16)
            second_pad[: int(K)] = second
            second = second_pad
            a_path = td_path / "A.npy"
            b_path = td_path / "B.npy"
            np.save(a_path, A_pad)
            np.save(b_path, second)
            # k_bursts on the weight role = ceil(K_pad_bursts/8); the C++ derives
            # the JUMP counters itself from the weight shape (advisory here).
            k_bursts = ((k_pad // 16) + 7) // 8
            # --num-tiles tiles the LOGICAL grid across physical fabric tiles
            # (derived from the partition, not hardcoded 1). For M_logical <= 4096
            # this is 1 (the whole grid fits one fabric tile).
            # The REDUCE path rebuilds the canonical GEMV CRF internally from the
            # weight shape (matched JUMP counters), so the EMITTED cmd stream is
            # irrelevant -- and a slice-form workload's emitted stream is
            # prod(mapping)*body cmds (e.g. 512 for the 128-PE grid), which would
            # trip the conductor's 32-cmd cap. Pass a minimal MAC CRF (inert for
            # REDUCE) and SKIP the emitted --cmds block below.
            crf_path = td_path / "reduce.crf"
            crf_path.write_text(
                "MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1\n"
                "JUMP loop_counter=0 loop_offset=2\n"
                "MAC dst=GRF_B src0=GRF_A src1=ODD_BANK is_auto=1\n"
                "JUMP loop_counter=0 loop_offset=2\n"
                "MOV dst=ODD_BANK src0=GRF_B\n"
            )
            argv += [
                "--inputs",
                f"{a_path},{b_path}",
                "--crf",
                str(crf_path),
                "--op-kind",
                "REDUCE",
                "--out-rows",
                str(rows_per_tile),
                "--out-cols",
                str(int(N)),
                "--reduce-k",
                str(k_pad),
                "--num-tiles",
                str(num_tiles),
                "--bank-type",
                "ALL",
                "--roles",
                f"A:0:0:2:0:0:0:{k_bursts},B:0:0:2:0:0:0:0,out:0:0:1:1:0:8:0",
            ]
            # readback slices the REAL P (M_real), not the grid/fabric pad.
            reduce_shape = (int(M_real), int(K), int(N), m_pad)
        else:  # GENERIC -- faithful multi-opcode interpreter (spec 01 §4)
            # Flat-CLI contract (§5.1): role .npy paths joined comma-separated
            # in role order, one flat fp16 out.bin, --num-tiles derived from
            # the operand size (never a shape literal). No --faithful: GENERIC
            # is always faithful (spec 01 §5.1).
            generic_inputs = {
                k: v
                for k, v in inputs.items()
                if v is not None and k not in ("layers",)
            }
            if not generic_inputs:
                # Back-compat: compiled.run() with no kwargs returns cycles=None.
                return RunResult(
                    cycles=None,
                    stdout="Samsung GENERIC needs operand kwargs",
                    backend="samsung_hbm_pim",
                )
            in_paths = []
            generic_n = None
            for ri, (rname, rval) in enumerate(generic_inputs.items()):
                arr = np.asarray(rval, dtype=np.float16).reshape(-1)
                if generic_n is None:
                    generic_n = int(arr.size)
                rp = td_path / f"in{ri}.npy"
                np.save(rp, arr)
                in_paths.append(str(rp))
            # elems_per_tile = 16 lanes x 8 GRF x num_pim_blocks; derive
            # num_tiles from the operand size (spec 01 §4.3 / §5.2). Fall back
            # to 1 tile for sub-tile operands.
            elems_per_tile = 16 * 8 * _samsung_num_pim_blocks(compiled)
            num_tiles = max(1, (generic_n + elems_per_tile - 1) // elems_per_tile)
            argv += [
                "--inputs",
                ",".join(in_paths),
                "--num-tiles",
                str(num_tiles),
                "--bank-type",
                "ALL",
            ]

        # SPEC-005: serialise compiled.cmds to a line-delimited file and
        # pass --cmds so the driver uses our CRF microcode instead of
        # PIMCmdGen's canonical one. Layout/placement changes show up in
        # cycles only because this file flows into programCrf().
        # The ISA-valid filter was applied up-front into `pim_cmds_all`.
        # SPEC-05: GENERIC_REDUCE supplies its OWN minimal --crf (the emitted
        # stream is canonicalized internally + a slice-form stream blows the
        # 32-cmd cap), so it skips this emitted-cmds block. The GENERIC ELTWISE
        # path uses the emitted CRF microcode via --cmds.
        if pim_cmds_all and kernel != "GENERIC_REDUCE":
            cmds_path = td_path / "cmds.txt"
            _write_samsung_cmds(cmds_path, pim_cmds_all)
            argv += ["--cmds", str(cmds_path)]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=str(root),  # pim_driver resolves ini/ paths relative to cwd
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"Samsung pim_driver invocation failed: {exc}") from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        match = re.search(r"PIM_CYCLES total=(\d+)", combined)
        if not match:
            raise RuntimeError(
                "Samsung pim_driver returned but stdout missing "
                "'PIM_CYCLES total=...' line; tail: " + combined[-400:]
            )
        cycles = int(match.group(1))
        # SPEC-004b (additive readback): surface the GEMV result the driver
        # already wrote to `out_path` into extra["outputs"], mirroring
        # _run_apu_v1. Pure post-run file I/O -- cycles (parsed above) and the
        # command stream are untouched; graceful when absent. Output role "y".
        extra = {"kernel": kernel, "returncode": proc.returncode}
        if kernel == "GENERIC_REDUCE":
            # SPEC-04 §4.3: faithful matmul readback. The REDUCE column cadence
            # drives the real PIMBlock::mac accumulator and the real drain, so
            # out.bin holds genuine numerics (no dual-run / plain rerun needed).
            M, K, N, m_pad = reduce_shape
            outputs = _samsung_read_reduce_outbin(out_path, M, N, m_pad)
            if outputs:
                extra["outputs"] = outputs
                extra["cycles_source"] = "PIMSimulator GENERIC REDUCE interpreter"
                extra["correctness_source"] = "PIMSimulator GENERIC REDUCE interpreter"
            return RunResult(
                cycles=cycles,
                stdout=combined,
                backend="samsung_hbm_pim",
                extra=extra,
            )
        # GENERIC (ELTWISE) flat fp16 readback (spec 01 §3.3): the faithful
        # interpreter writes real numerics into out.bin. Role name "out". (The
        # only two routes are GENERIC_REDUCE above and GENERIC here -- the legacy
        # GEMV/ADD/MUL/RELU readbacks were deleted with the legacy routing.)
        outputs = _samsung_read_generic_outbin(out_path, generic_n)
        if outputs:
            extra["outputs"] = outputs
            extra["cycles_source"] = "PIMSimulator GENERIC interpreter"
            extra["correctness_source"] = "PIMSimulator GENERIC interpreter"
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="samsung_hbm_pim",
            extra=extra,
        )


def _layout_stage_resident(layout) -> bool:
    """Read the chosen placement's structural `stage_resident` flag.

    Bridge option (b) (task-017): renamed off `weight_resident`; the
    host_staging cost split and this run-path preload sequencing both read
    the same flag. `Compiled.layout` is a single Placement (one kernel) or a
    list. The batched run path is single-GEMV; default-absent key -> False
    (the non-resident shape). Codegen materialises this decision; it does
    not re-decide (I5).
    """
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    if layout is None:
        return False
    return bool(getattr(layout, "extra", {}).get("stage_resident", False))


def _samsung_batched_invoke(
    driver: Path,
    root: Path,
    cmd_subset: list,
    W,
    X,
    batch: int,
    native_rebaseline: bool,
    np_mod,
) -> tuple[int, dict, str]:
    """One batched pim_driver --batch invocation (SPEC-026 §4.2).

    Returns `(total_cycles, phase_dict, combined_stdout)`. `phase_dict`
    has preload/exec/readback so the caller can assert the per-phase
    split against the cost model. `native_rebaseline=True` selects the
    re-preload-per-vector comparator loop.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_path = td_path / "out.bin"
        w_path = td_path / "W.npy"
        x_path = td_path / "X.npy"
        cmds_path = td_path / "cmds.txt"
        # The vendor GEMV conductor addresses a complete 4096-row Samsung
        # fabric tile and alternates even/odd 128-wide reduction tiles.  Small
        # or ragged PolyBench legs therefore need the same zero padding as the
        # generic REDUCE path; passing an underfilled matrix makes the reference
        # PIMSimulator preload walk beyond ``NumpyBurstType::bData`` and abort.
        # Zero padding preserves W@X and is part of the physical lowering, not
        # a workload-size change.
        m_real, k_real = int(W.shape[0]), int(W.shape[1])
        m_pad = (
            (m_real + _SAMSUNG_FABRIC_ROW_TILE - 1) // _SAMSUNG_FABRIC_ROW_TILE
        ) * _SAMSUNG_FABRIC_ROW_TILE
        k_pad = (
            (k_real + _SAMSUNG_REDUCE_K_TILE - 1) // _SAMSUNG_REDUCE_K_TILE
        ) * _SAMSUNG_REDUCE_K_TILE
        if m_pad != m_real or k_pad != k_real:
            W_physical = np_mod.zeros((m_pad, k_pad), dtype=np_mod.float16)
            W_physical[:m_real, :k_real] = W
            X_physical = np_mod.zeros((batch, k_pad), dtype=np_mod.float16)
            X_physical[:, :k_real] = X
        else:
            W_physical = W
            X_physical = X
        np_mod.save(w_path, W_physical)
        np_mod.save(x_path, X_physical)
        _write_samsung_cmds(cmds_path, cmd_subset)

        argv = [
            str(driver),
            "--op",
            "GEMV",
            "--out",
            str(out_path),
            "--weight",
            str(w_path),
            "--in",
            str(x_path),
            "--output-dim",
            str(m_pad),
            "--input-dim",
            str(k_pad),
            "--cmds",
            str(cmds_path),
            "--faithful",
            "--batch",
            str(batch),
        ]
        if native_rebaseline:
            argv.append("--native-rebaseline")

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=str(root),
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"Samsung batched pim_driver invocation failed: {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        m = re.search(
            r"PIM_CYCLES total=(\d+)\s+preload=(\d+)\s+exec=(\d+)\s+readback=(\d+)",
            combined,
        )
        if not m:
            raise RuntimeError(
                "Samsung batched pim_driver returned but stdout missing "
                "'PIM_CYCLES total=...' line; tail: " + combined[-400:]
            )
        phases = {
            "preload": int(m.group(2)),
            "exec": int(m.group(3)),
            "readback": int(m.group(4)),
        }
        return int(m.group(1)), phases, combined


def _run_samsung_batched(
    compiled: "Compiled",
    W,
    X,
    compare_native: bool = True,
) -> RunResult:
    """Batched-GEMV run path (SPEC-026 §4.2/§4.3).

    Emits the Tenon stream (preload once + B*(exec+readback), selected by
    the chosen placement's `stage_resident` flag) AND, when
    `compare_native` is set, the native rebaseline (B*(preload+exec+
    readback)) under the SAME faithful instrument, cmd stream, W, and
    X(B,K). Only the preload-loop placement differs (I2). B = X.shape[0]
    is read off the operand shape, never a literal (I3).

    `RunResult.cycles` is the Tenon total; `extra` carries the native
    total + both per-phase splits so the gate-A comparison and the
    cost-model assertion can be made by the caller.
    """
    import numpy as np

    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    root = _pimsim_root()
    driver = root / "pim_driver"

    W = np.asarray(W, dtype=np.float16)
    X = np.asarray(X, dtype=np.float16)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    B = int(X.shape[0])  # I3: B is operand geometry (X[B,K] leading dim).

    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]

    # Same ISA-valid filter the single-vector run path applies.
    def _crf_valid(c: PIMCmd) -> bool:
        # GRF_A is populated by the faithful GEMV conductor before CRF
        # execution.  ``SamsungCtx`` retains a FILL marker for the generic
        # interpreter, but forwarding it to the vendor GEMV programCrf path
        # makes PIMSimulator reject/abort the stream before reporting cycles.
        # The canonical Samsung GEMV CRF begins with MAC for the same reason.
        if c.type_ == "FILL":
            return False
        if c.type_ in ("MOV", "FILL"):
            bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
            grf_src = any(s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_, c.src2_))
            if bank_dst and grf_src:
                return False
        return True

    pim_cmds = [c for c in pim_cmds if _crf_valid(c)]

    resident = _layout_stage_resident(compiled.layout)
    # Tenon: resident mode preloads once when B>1; the driver reads the
    # resident vs native loop from --native-rebaseline (absent => resident).
    # If the chosen placement is NOT stage_resident, Tenon's own run is the
    # native (re-preload-per-vector) sequencing -- codegen materialises the
    # decision argmin made, it does not override it.
    tenon_total, tenon_phases, tenon_out = _samsung_batched_invoke(
        driver,
        root,
        pim_cmds,
        W,
        X,
        B,
        native_rebaseline=not resident,
        np_mod=np,
    )

    extra = {
        "kernel": "GEMV",
        "batch": B,
        "stage_resident": resident,
        "tenon_total": tenon_total,
        "tenon_phases": tenon_phases,
    }
    combined = tenon_out
    if compare_native:
        native_total, native_phases, native_out = _samsung_batched_invoke(
            driver,
            root,
            pim_cmds,
            W,
            X,
            B,
            native_rebaseline=True,
            np_mod=np,
        )
        extra["native_total"] = native_total
        extra["native_phases"] = native_phases
        combined = tenon_out + "\n--- native rebaseline ---\n" + native_out

    return RunResult(
        cycles=tenon_total,
        stdout=combined,
        backend="samsung_hbm_pim",
        extra=extra,
    )
