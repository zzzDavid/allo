# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-trace liveness for the SPMW autoscheduler (SPEC-023 D1).

The autoscheduler's allocator runs liveness PER GROUP (`extract_live_ranges`
takes one `func_name` bucket's matches). That cannot see a
value used across two `@allo.work` kernels (MLP `h = ReLU(W1.x)` consumed by
layer 2) or a value invariant across work-ids (GEMV `x` broadcast to every
bank): each is placed independently, with no representation of "keep it
resident across the boundary."

`trace_liveness` is the cross-group FOLD of that same per-group operand walk:
it consumes the WHOLE `MatchTrace` and reports, for each `(memref_name, role)`,
the first/last `(func_name, work_id, match_idx)` site it is live at, plus two
flags derived purely from the bucketed trace order:

  - `crosses_kernel` (T6): the value's live span spans >1 `func_name` -- the
    cross-kernel residency signal.
  - `crosses_workid` (T4): the value is live across >1 `work_id` within a
    single kernel and is invariant to the work-id axis -- the broadcast-hoist
    signal (preload once, reuse).

Both fall out of the bucketed order -- NO `allo/ir/` edit, the same additive-
trace discipline as `_trace_reduction_trip` / `batch_dim`. This module DEFINES
no placement/cost interface; it produces a liveness result the residency `Knob`
(`spmw_knobs.py`) reads via `KnobCtx.liveness`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spmw_match import MatchTrace


@dataclass
class LiveSpan:
    """Whole-trace live span of one `(memref_name, role)`.

    `first`/`last` are `(func_name, work_id, match_idx)` site keys at the span
    endpoints (trace order). `func_names`/`work_ids` are the distinct kernels /
    work-ids the value is live at. `crosses_kernel` / `crosses_workid` are the
    T6 / T4 signals the residency knob keys on.
    """

    memref_name: str
    role: str
    first: tuple = None
    last: tuple = None
    func_names: tuple = ()
    work_ids: tuple = ()
    crosses_kernel: bool = False
    crosses_workid: bool = False
    # Producer / consumer kernels for a cross-kernel value: the kernel whose
    # last write defines it vs the later kernel that reads it. For T6 these are
    # the LiveSpan endpoints' func_names; the residency knob records the
    # producer->consumer pair so codegen knows which storeback/preload to elide.
    producer_func: str = None
    consumer_func: str = None


def trace_liveness(trace: MatchTrace) -> "dict[tuple[str, str], LiveSpan]":
    """Whole-trace liveness: which `(memref, role)` is live across which
    matches, across work-ids AND across `@allo.work` kernels.

    Reuses `_bucket_for_autoschedule` for grouping and the SAME operand walk
    `extract_live_ranges` uses per group -- this is the cross-group fold of
    that per-group analysis, not a new liveness notion. The bucketed trace
    order is the only input; no front-end edit.
    """
    # Local import to avoid an import cycle (autoschedule imports nothing from
    # here at module load; this is called at autoschedule run time).
    from .spmw_autoschedule import _bucket_for_autoschedule

    spans: dict[tuple[str, str], LiveSpan] = {}
    # Global match index in trace order, so first/last endpoints order across
    # kernels (a cross-kernel value's last site is in a later bucket).
    global_idx = 0
    for func_name, matches in _bucket_for_autoschedule(trace):
        for local_idx, match in enumerate(matches):
            work_id = match.work_id
            for opb in match.operands:
                if opb.memref_name is None:
                    continue
                key = (opb.memref_name, opb.role)
                site = (func_name, work_id, global_idx)
                span = spans.get(key)
                if span is None:
                    spans[key] = LiveSpan(
                        memref_name=opb.memref_name,
                        role=opb.role,
                        first=site,
                        last=site,
                        func_names=(func_name,),
                        work_ids=(work_id,),
                    )
                else:
                    span.last = site
                    if func_name not in span.func_names:
                        span.func_names = span.func_names + (func_name,)
                    if work_id not in span.work_ids:
                        span.work_ids = span.work_ids + (work_id,)
            global_idx += 1

    # Aggregate func_names / work_ids per MEMREF across all roles: a value
    # produced under one role (layer1 `acc`-role output `h`) and consumed under
    # another (layer2 `y`-role input `h`) is ONE value crossing the kernel
    # boundary, even though it occupies two `(memref, role)` keys. The T6 flag
    # is a memref-level property; the per-(memref,role) span carries it.
    mref_funcs: dict[str, tuple] = {}
    mref_workids: dict[str, tuple] = {}
    for (mref, _role), span in spans.items():
        fs = mref_funcs.get(mref, ())
        for fn in span.func_names:
            if fn not in fs:
                fs = fs + (fn,)
        mref_funcs[mref] = fs
        ws = mref_workids.get(mref, ())
        for w in span.work_ids:
            if w not in ws:
                ws = ws + (w,)
        mref_workids[mref] = ws

    # Derive the T4 / T6 flags from the memref-level aggregation.
    for span in spans.values():
        funcs = mref_funcs.get(span.memref_name, span.func_names)
        workids = mref_workids.get(span.memref_name, span.work_ids)
        span.crosses_kernel = len(funcs) > 1
        span.crosses_workid = len(workids) > 1
        if span.crosses_kernel:
            # Producer = first kernel the memref appears in; consumer = the
            # last (bucket order = program order).
            span.producer_func = funcs[0]
            span.consumer_func = funcs[-1]
    return spans


def crosses_boundary(span: LiveSpan) -> bool:
    """True iff `span` crosses a kernel or work-id boundary -- the condition
    under which residency is POSSIBLE (the residency knob's >=2-candidate gate).
    A value that crosses neither boundary is per-kernel/per-work-id local and
    has no residency DOF (today's single-candidate behaviour)."""
    return span.crosses_kernel or span.crosses_workid


def memref_span(
    liveness: "dict[tuple[str, str], LiveSpan] | None", memref_name: str
) -> "LiveSpan | None":
    """The widest `LiveSpan` for `memref_name` across any role (a memref may be
    a `y`-role producer output and an `x`-role consumer input under different
    roles). Returns the span that crosses a boundary if any does, else any
    span, else None. `None` liveness (the default-absent per-kernel mode)
    yields None -- byte-identical to today."""
    if not liveness:
        return None
    best: LiveSpan | None = None
    for (mref, _role), span in liveness.items():
        if mref != memref_name:
            continue
        if best is None or (crosses_boundary(span) and not crosses_boundary(best)):
            best = span
    return best
