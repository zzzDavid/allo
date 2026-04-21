# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blocker #2: grid-fitting pass.

Composes a workload's ``mapping=`` grid against the target's
``@allo.unit`` tree, producing a structured :class:`GridFitResult` that
later passes (layout inference, backend emit) can consume. This is a
pure-Python pass today; it does not rewrite MLIR or emitted text.

See ``experiments/reports/11-spmw-for-target-description.md`` §3 and
§10.4 limit 3, and report 07 §2.4.

Why a separate pass (and why it lives here)
-------------------------------------------

Before this pass, ``allo.compile(work, target=...)`` ignored
``work.mapping`` entirely (see the ``mapping`` kwarg comment in
``allo/work.py`` and ``allo/compile.py``). The workload grid never got
composed against the target SPMW tree, which meant the linear-layout
catalog was not parameterized by workload axes and users passing
``mapping=...`` saw no effect. The v0 slice here produces a structured
axis assignment; a later push will teach backends (and the layout
catalog) to consume it.

Design notes
------------

* **Greedy left-to-right.** Iterate the workload axes in user order, and
  consume target axes root-to-leaf. A workload axis with extent ``Pw``
  consumes target axes until their product reaches ``Pw``. If one target
  axis is larger than the workload axis, split it (the residue becomes a
  fresh target axis at the head of the queue).
* **Named mapping binds by name first.** ``mapping={"m": M, "k": K}``
  looks for target axes with the same name and binds directly; anything
  left over runs through the greedy residue.
* **No automatic tiling search.** If the workload extent does not
  cleanly match target extents, reject with an actionable error that
  shows both sides. (Report 11 §3 explicitly wants no search here.)
* **Mode annotations are advisory.** The pass records the strictest
  mode across each workload axis' target axes (SIMD > SIMD-LOCKSTEP >
  SIMT-BARRIER > MIMD) so later passes (R4 checks, linear-layout
  selection) can read them. The pass itself does not check workload-body
  constraints — that is R4 territory.
* **Leftover target axes are not rejected.** A target with 4 levels and
  a workload with ``mapping=[P]`` legitimately leaves 3 levels idle. We
  report them on ``leftover_target_axes`` for downstream passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..work import Work
from .target import Target


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------


class GridFitError(ValueError):
    """Raised when a workload mapping cannot be composed against the
    target's ``@allo.unit`` tree. The message prints both sides so the
    user can see the disagreement at a glance."""


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AxisAssignment:
    """One workload axis -> one contiguous run of target axes.

    ``tile`` is the extra multiplicity the workload axis carries beyond
    the product of the consumed target extents (``Pw // Pt``). For a
    clean fit it is 1; for a workload axis that oversubscribes the
    target axes it is the per-target-instance workload count.
    """
    workload_axis: Union[int, str]     # positional index or user name
    workload_extent: int
    target_axes: List[str]
    target_extents: List[int]
    tile: int
    # Strictest mode among the consumed target axes, useful for R4
    # / linear-layout downstream.
    mode: str = "mimd"


@dataclass
class GridFitResult:
    """Output of :func:`grid_fit`. Informational for v0; backends are
    free to ignore it today. A later patch will wire this into the
    linear-layout catalog and ``allo.pim.mlir_emit``.

    ``workload_mapping`` is the *normalized* form of ``work.mapping``:
    an int becomes ``[int]`` and a dict is preserved as-is. ``None`` /
    empty -> ``[]``.
    """
    target_name: str
    workload_mapping: Union[List[int], Dict[str, int]]
    assignments: List[AxisAssignment] = field(default_factory=list)
    leftover_target_axes: List[str] = field(default_factory=list)
    mode_annotations: Dict[Union[int, str], str] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.assignments and not self.leftover_target_axes:
            return (f"# grid_fit(target={self.target_name}): "
                    "no mapping (baseline)")
        lines = [f"# grid_fit(target={self.target_name})"]
        for a in self.assignments:
            tgt = " x ".join(f"{ax}={ext}" for ax, ext in
                             zip(a.target_axes, a.target_extents))
            tile_note = f" tile={a.tile}" if a.tile != 1 else ""
            lines.append(
                f"  W[{a.workload_axis}] (extent={a.workload_extent}) "
                f"-> {tgt}{tile_note}  [{a.mode}]")
        if self.leftover_target_axes:
            lines.append(
                f"  leftover target axes (idle): "
                f"{', '.join(self.leftover_target_axes)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# strictness order — later passes may want the strictest mode spanned by
# a workload axis (an axis that touches a simd level must itself be
# SIMD-constrained, even if it also touches a MIMD level). This is the
# strictness order the R4/linear-layout passes care about; keep it in
# sync with ``unit.py::_VALID_MODES``.
_MODE_STRICTNESS = {
    "host": 0,
    "mimd": 1,
    "simt-barrier": 2,
    "simd-lockstep": 3,
    "simd": 4,
}


def _stricter(a: str, b: str) -> str:
    return a if _MODE_STRICTNESS.get(a, 0) >= _MODE_STRICTNESS.get(b, 0) else b


def _flatten_target_axes(target: Target):
    """Yield ``(axis_name, extent, mode)`` tuples in root-to-leaf order
    by walking ``target.tn_root``.

    The tree is a single-child chain (multi-child is rejected upstream
    in ``_to_grid``). A level whose ``extents`` has length > 1 expands
    into a chain of single-extent entries sharing the same axis name —
    that mirrors how :func:`_to_grid` would flatten them.
    """
    root = getattr(target, "tn_root", None)
    if root is None:
        # Fallback: compose from ``t.axes`` and the flat ``t.parallel_units``
        # attribute. This path is only hit by older tests that build a
        # Target directly from ``build_from_grid`` without the decorator.
        for axis in getattr(target, "axes", []):
            yield axis, 1, getattr(target, "modes", {}).get(axis, "mimd")
        return

    level = root
    while level is not None:
        for ext in level.extents:
            yield level.axis, int(ext), level.mode
        if not level.children:
            break
        # Phase-1 guarantees at most one child per level.
        level = level.children[0]


def _normalize_mapping(
        mapping: Union[None, int, List[int], Dict[str, int]]
        ) -> Union[List[int], Dict[str, int]]:
    if mapping is None:
        return []
    if isinstance(mapping, int):
        return [mapping]
    if isinstance(mapping, (list, tuple)):
        return list(mapping)
    if isinstance(mapping, dict):
        # Preserve insertion order.
        return dict(mapping)
    raise GridFitError(
        f"@allo.work(mapping=...) must be int, list[int], or "
        f"dict[str, int]; got {type(mapping).__name__}: {mapping!r}")


def _validate_mapping_values(
        mapping: Union[List[int], Dict[str, int]]) -> None:
    items = mapping.items() if isinstance(mapping, dict) else \
            enumerate(mapping)
    for k, v in items:
        if not isinstance(v, int) or v <= 0:
            raise GridFitError(
                f"@allo.work(mapping=...) extents must be positive ints; "
                f"got {k}={v!r}")


# ---------------------------------------------------------------------------
# The fitting algorithm
# ---------------------------------------------------------------------------


def _consume_positional(
        workload: List[int],
        queue: List[tuple],
        ) -> List[AxisAssignment]:
    """Greedy left-to-right consumption for a positional workload list.

    ``queue`` is a mutable list of ``(axis, extent, mode)`` tuples in
    root-to-leaf order. We pop from the left and may push a residual
    back onto the left when we split a target axis.
    """
    assignments: List[AxisAssignment] = []
    for w_idx, Pw in enumerate(workload):
        consumed_axes: List[str] = []
        consumed_extents: List[int] = []
        consumed_modes: List[str] = []
        Pt = 1
        while Pt < Pw:
            if not queue:
                raise GridFitError(
                    f"workload axis [{w_idx}] (extent={Pw}) exceeds the "
                    f"remaining target grid (product of remaining target "
                    f"extents is {Pt}). Either shrink the workload or add "
                    f"a target level.")
            axis, ext, mode = queue.pop(0)
            # Skip degenerate extent=1 target axes rather than swallowing
            # them silently; record them so the mode propagates but don't
            # consume "mass" from the workload axis. This keeps
            # ``mapping=[P]`` assignments clean on targets that have an
            # idle outer axis (e.g. UPMEM's ``rank=1``).
            if ext == 1 and Pt < Pw:
                consumed_axes.append(axis)
                consumed_extents.append(ext)
                consumed_modes.append(mode)
                continue
            if Pt * ext <= Pw:
                # Whole axis fits inside the remaining workload budget.
                consumed_axes.append(axis)
                consumed_extents.append(ext)
                consumed_modes.append(mode)
                Pt *= ext
                continue
            # Pt * ext > Pw. Can we split this axis?
            remaining = Pw // Pt  # how much workload mass is left
            if Pt * remaining != Pw or ext % remaining != 0:
                raise GridFitError(
                    f"workload axis [{w_idx}] (extent={Pw}) does not "
                    f"cleanly factor across the target grid. After "
                    f"consuming {consumed_axes or '[]'} "
                    f"(product={Pt}), the next target axis {axis!r} has "
                    f"extent {ext} but we need a factor of "
                    f"{remaining}.")
            # Split: take ``remaining`` here, push residue back.
            consumed_axes.append(axis)
            consumed_extents.append(remaining)
            consumed_modes.append(mode)
            Pt *= remaining
            residue = ext // remaining
            queue.insert(0, (axis, residue, mode))
            break
        if Pt == 0:
            # Only possible if the workload axis is 0 (already rejected).
            raise GridFitError(
                f"workload axis [{w_idx}] produced zero target product")
        if Pw % Pt != 0:
            # Unreachable via the loop above (we stop when Pt >= Pw and
            # check divisibility on the split path), but keep for safety.
            raise GridFitError(
                f"workload axis [{w_idx}] (extent={Pw}) is not a clean "
                f"multiple of consumed target extents {consumed_extents}")
        tile = Pw // Pt
        mode = "mimd"
        for m in consumed_modes:
            mode = _stricter(mode, m)
        assignments.append(AxisAssignment(
            workload_axis=w_idx,
            workload_extent=Pw,
            target_axes=consumed_axes,
            target_extents=consumed_extents,
            tile=tile,
            mode=mode,
        ))
    return assignments


def _consume_named(
        workload: Dict[str, int],
        queue: List[tuple],
        ) -> List[AxisAssignment]:
    """Named workload mapping. First pass binds by name to any target
    axis in the queue with the same name; anything that did not get a
    name match falls through to greedy consumption of the residue.
    """
    assignments: List[AxisAssignment] = []
    # Build a name -> (index, extent, mode) lookup over the current queue.
    name_to_pos = {axis: i for i, (axis, _, _) in enumerate(queue)}

    # --- pass 1: named binding -------------------------------------------
    bound_mask: List[bool] = [False] * len(queue)
    bound: Dict[str, AxisAssignment] = {}
    for w_name, Pw in workload.items():
        if w_name in name_to_pos:
            idx = name_to_pos[w_name]
            if bound_mask[idx]:
                # Shouldn't happen — each target axis name is unique per
                # Rule R2. Keep the guard for safety.
                raise GridFitError(
                    f"workload axis {w_name!r} would bind to an already-"
                    "bound target axis")
            axis, ext, mode = queue[idx]
            # Pw must divide ext (shrink) or ext divide Pw (fan out).
            if Pw == ext:
                tile = 1
                target_ext = ext
            elif ext % Pw == 0:
                # Workload is smaller: split the axis. Residue stays on
                # the queue for pass 2 to pick up (or report as leftover).
                tile = 1
                target_ext = Pw
                residue = ext // Pw
                # Replace this position with the residue *in-place* so
                # pass 2's greedy walker sees it.
                queue[idx] = (axis, residue, mode)
            elif Pw % ext == 0:
                tile = Pw // ext
                target_ext = ext
            else:
                raise GridFitError(
                    f"named workload axis {w_name!r} (extent={Pw}) does "
                    f"not cleanly match target axis {axis!r} "
                    f"(extent={ext})")
            assignments.append(AxisAssignment(
                workload_axis=w_name,
                workload_extent=Pw,
                target_axes=[axis],
                target_extents=[target_ext],
                tile=tile,
                mode=mode,
            ))
            bound[w_name] = assignments[-1]
            if tile == 1 and target_ext == ext:
                # Fully consumed: drop from queue (replace with None
                # sentinel so pass 2 positional indices don't shift).
                bound_mask[idx] = True

    # Drop fully-bound entries from the queue; pass 2 runs on the rest.
    remaining_queue = [q for i, q in enumerate(queue) if not bound_mask[i]]

    # --- pass 2: greedy over unbound named workload axes -----------------
    unbound = [(k, v) for k, v in workload.items() if k not in bound]
    if unbound:
        # Reuse positional algorithm but record the *original name* back
        # onto the result rather than the positional index.
        #
        # We do this in-line to avoid munging indices: build the
        # positional result, then rewrite ``workload_axis`` from int to
        # the corresponding name.
        positional_exts = [v for _, v in unbound]
        positional_assignments = _consume_positional(
            positional_exts, remaining_queue)
        for name_pos, a in enumerate(positional_assignments):
            a.workload_axis = unbound[name_pos][0]
            assignments.append(a)

    # Reorder assignments to match the user's dict insertion order.
    by_name = {a.workload_axis: a for a in assignments}
    ordered = [by_name[k] for k in workload if k in by_name]
    # ``remaining_queue`` already reflects any pass-2 consumption via the
    # positional walker's pop/insert on the shared list.
    return ordered, remaining_queue


def grid_fit(work: Work, target: Target) -> GridFitResult:
    """Compose ``work.mapping`` against ``target``'s ``@allo.unit`` tree.

    Returns a :class:`GridFitResult`. Raises :class:`GridFitError` if
    the mapping cannot be composed cleanly.
    """
    mapping = _normalize_mapping(work.mapping)
    _validate_mapping_values(mapping)

    target_name = getattr(target, "name", "<unnamed>")

    # Build the mutable (axis, extent, mode) queue, root->leaf.
    queue = list(_flatten_target_axes(target))
    # Remember the FULL axis order for computing ``leftover_target_axes``
    # once consumption is done.
    all_axes_seen: Dict[str, int] = {}
    for axis, _, _ in queue:
        all_axes_seen[axis] = all_axes_seen.get(axis, 0) + 1
    total_axis_entries = sum(all_axes_seen.values())

    if not mapping:
        # Baseline / MVP-today path: no fitting at all.
        return GridFitResult(
            target_name=target_name,
            workload_mapping=[] if not isinstance(mapping, dict) else {},
            assignments=[],
            leftover_target_axes=[a for a, _, _ in queue],
            mode_annotations={},
        )

    if isinstance(mapping, dict):
        assignments, leftover_queue = _consume_named(mapping, queue)
    else:
        assignments = _consume_positional(list(mapping), queue)
        leftover_queue = queue

    # Leftover = axes left in the queue after consumption + any axes
    # whose residue is still sitting in the queue. For the positional
    # path the walker pushed residues back via ``queue.insert``; for
    # the named path the caller collapses its bound_mask and returns
    # the trimmed queue directly.
    leftover = [axis for axis, _, _ in leftover_queue]

    mode_annotations = {a.workload_axis: a.mode for a in assignments}

    return GridFitResult(
        target_name=target_name,
        workload_mapping=mapping,
        assignments=assignments,
        leftover_target_axes=leftover,
        mode_annotations=mode_annotations,
    )


__all__ = ["grid_fit", "GridFitResult", "AxisAssignment", "GridFitError"]
