# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""allo.unit — decorator surface for Tenon target description.

This module implements the ``@allo.unit`` / ``@allo.target`` decorator
surface that the paper figure (``paper/latex/code/tenon_mlp.tex``) uses.
It produces the SAME ``Target`` data structure the existing backends
(``pimsim_driver.py``, ``upmem_codegen.py``, ``aim_shadow.py``, ...)
already consume in ``pim_dsl/``, so porting the five targets to the new
surface is a surface-syntax change with no backend work.

The decorator module sits next to Allo's workload-side ``@allo.work``:
``@allo.work`` names a workload, ``@allo.unit`` names a level of the
target grid. Together they expose both halves of the SPMW surface from
the ``allo`` namespace.

Phase-1 scope
-------------

Type-checker rules enforced here (Report 11 §4):

  R1  Every ``@allo.unit`` has an axis name (= ``func.__name__``). Reject
      lambdas and placeholder names like ``_``.
  R2  Axis names within one target tree are unique.
  R3  A ``@allo.unit`` body contains only structural calls
      (``allo.memory``, ``allo.op``, ``allo.stream``, ``allo.cost``, or a
      nested ``@allo.unit``). Plain Python statements (prints,
      assignments, ``if``s that do not call the ``allo.*`` API) are
      *tolerated silently* — they simply have no effect on the target
      tree. This matches the "body is a sequence of allo.* calls"
      reading of §3.1 and keeps the surface friendly for debugging.
  R4  ``@allo.unit(mode="simd" | "simd-lockstep")`` bodies may not
      contain ``allo.stream(...)`` calls. (meta_if is a future
      extension.)

Rules R5-R9 (memory visibility queries, host-root op lifting, layout
input-dim checking, stream put/get matching) are **TODO** for later
phases; the existing backends already handle memory visibility
implicitly (they iterate ``target.memories`` flat), so leaving these out
does not regress anything.

Host root
---------

``@allo.target(name, host_memories=[...], host_ops=[...])`` accepts the
same kwargs the flat-form ``build_from_grid`` already takes. This is the
simplest, least-magical surface; it avoids introducing a new ``with
allo.host(): ...`` context manager and keeps the outermost decorator
the only place a user has to reason about CPU-side fallbacks. The
synthetic host root is therefore expressed as kwargs on
``@allo.target``, not as an implicit ``@allo.unit`` at the top of the
tree.

Semantic decisions worth calling out
------------------------------------

1. The function decorated by ``@allo.target`` is REPLACED by the built
   ``Target`` object. After ``@allo.target("x")`` runs, the name
   ``channel`` (from ``def channel(): ...``) is the ``Target``. This
   matches how ``compile(..., target=allo.target("samsung_hbm_pim"))``
   reads in the paper figure — the target name is a handle.

2. Nested ``@allo.unit`` declarations INSIDE a parent ``@allo.unit``'s
   body are registered by mutating a thread-local "current level"
   stack. This is the cleanest way to make the decorators cooperate
   without requiring users to thread a builder object manually.

3. ``allo.op`` / ``allo.memory`` kwargs are passed through to
   ``Op(...)`` / ``Memory(...)`` mostly verbatim. Unknown kwargs
   (``mode=``, ``bytes=``, ``elems=``, ``bits=``) are normalized to the
   nearest existing field (``bytes`` -> ``capacity_bytes``, ``elems`` ->
   ``capacity_bytes``-in-elements, ``bits`` ->
   ``capacity_bytes``-in-bits packed to bytes, ceil). This keeps the
   decorator surface aligned with the paper figure while the underlying
   dataclasses stay stable. Unknown attributes end up in a free-form
   ``extras`` dict attached to the object (via ``__dict__``).

4. ``allo.cost(...)`` attaches ``traverse``, ``broadcast``, ``reduce``
   to the corresponding ``Grid`` node's free fields. The existing
   ``Grid`` class carries these as free attributes; we add them with
   ``setattr`` rather than extending the class signature.

Note on the ``allo.pim`` import
-------------------------------

The underlying ``Memory`` / ``Op`` / ``Target`` / ``Grid`` / ``Leaf``
dataclasses live in ``allo.pim.target`` (this was formerly the
out-of-tree ``pim_dsl`` package; it has been folded into Allo so
the decorator surface can reuse the dataclasses directly via a
relative import).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .pim.target import (
    Memory, Op, Target, Grid, Leaf, build_from_grid,
)


# ---------------------------------------------------------------------------
# Builder nodes
# ---------------------------------------------------------------------------
#
# While a @allo.target chain is executing we build a lightweight "pending
# level" tree, then flatten it to `Grid`/`Leaf` at the end. Keeping the
# intermediate form separate from `Grid`/`Leaf` lets us reshape freely
# without fighting the `Grid` constructor's shape-expansion logic.


class _PendingLevel:
    """A @allo.unit level during construction — not yet a Grid/Leaf."""

    __slots__ = ("axis", "extents", "mode", "children",
                 "memories", "ops", "streams", "cost_attrs")

    def __init__(self, axis: str, extents: List[int], mode: str):
        self.axis: str = axis
        self.extents: List[int] = list(extents)
        self.mode: str = mode
        self.children: List["_PendingLevel"] = []
        self.memories: List[Memory] = []
        self.ops: List[Op] = []
        self.streams: List[Dict[str, Any]] = []
        self.cost_attrs: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Thread-local stack of the currently-being-built level
# ---------------------------------------------------------------------------

_tls = threading.local()


def _stack() -> List[_PendingLevel]:
    st = getattr(_tls, "stack", None)
    if st is None:
        st = []
        _tls.stack = st
    return st


def _current_level() -> Optional[_PendingLevel]:
    st = _stack()
    return st[-1] if st else None


def _require_current(api_name: str) -> _PendingLevel:
    lvl = _current_level()
    if lvl is None:
        raise RuntimeError(
            f"allo.{api_name}(...) called outside any @allo.unit body. "
            "Structural calls must appear inside an @allo.unit-decorated "
            "function, which is itself inside a @allo.target-decorated "
            "function.")
    return lvl


# ---------------------------------------------------------------------------
# Normalizers for kwargs that show up in the paper figure
# ---------------------------------------------------------------------------

_VALID_MODES = {"mimd", "simd", "simd-lockstep", "simt-barrier"}

_MEMORY_KNOWN = {"capacity_bytes", "lanes", "dtype", "parallel_units", "scope"}
_OP_KNOWN = {"lanes", "latency", "throughput", "cycles_per_elem",
             "energy_pJ", "emit"}


def _normalize_memory_kwargs(name: str, kw: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    extras: Dict[str, Any] = {}
    for k, v in kw.items():
        if k in _MEMORY_KNOWN:
            out[k] = v
        elif k == "bytes":
            out["capacity_bytes"] = int(v)
        elif k == "elems":
            # elems -> bytes assumes 2 bytes per elem (fp16 default); fine
            # because backends mostly read Memory.capacity_bytes as "big
            # enough" and compare relative sizes, not absolute ones.
            out.setdefault("capacity_bytes", int(v) * 2)
        elif k == "bits":
            out.setdefault("capacity_bytes", max(1, (int(v) + 7) // 8))
        else:
            extras[k] = v
    out.setdefault("capacity_bytes", 0)
    return {"_known": out, "_extras": extras}


def _normalize_op_kwargs(name: str, kw: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    extras: Dict[str, Any] = {}
    for k, v in kw.items():
        if k in _OP_KNOWN:
            out[k] = v
        elif k == "energy":
            out["energy_pJ"] = float(v)
        else:
            extras[k] = v
    out.setdefault("lanes", 1)
    out.setdefault("latency", 0)
    return {"_known": out, "_extras": extras}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def unit(*, mapping: Union[int, List[int]], mode: str = "mimd") -> Callable:
    """Declare one level of the target grid.

    ``mapping`` is the extent(s) of this axis (int or list of ints). The
    function name is the axis name (Rule R1). ``mode`` is one of
    ``{"mimd", "simd", "simd-lockstep", "simt-barrier"}``.

    The decorated function is CALLED by the decorator to expand its
    body; any ``allo.memory / allo.op / allo.stream / allo.cost`` calls
    and nested ``@allo.unit`` decorators inside the body register
    themselves against this level.
    """
    if isinstance(mapping, int):
        extents = [mapping]
    elif isinstance(mapping, (list, tuple)) and all(
            isinstance(x, int) for x in mapping):
        extents = list(mapping)
    else:
        raise TypeError(
            f"@allo.unit(mapping=...) expects int or list[int], "
            f"got {mapping!r}")
    if not extents or any(e <= 0 for e in extents):
        raise ValueError(
            f"@allo.unit(mapping={mapping!r}) must have positive extents")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"@allo.unit(mode={mode!r}) — legal modes: "
            f"{sorted(_VALID_MODES)}")

    def decorator(func: Callable) -> _PendingLevel:
        # Rule R1: axis name from function name.
        axis = getattr(func, "__name__", "") or ""
        if not axis or axis == "<lambda>" or axis.startswith("_"):
            raise TypeError(
                "@allo.unit requires a named function (axis name = "
                f"function __name__); got {axis!r}. Use "
                "`def channel(): ...`, not a lambda or `def _():`.")

        level = _PendingLevel(axis=axis, extents=extents, mode=mode)

        # Check R2 against enclosing stack (siblings can share axis
        # names across separate subtrees in theory, but the spec says
        # unique across the whole tree — we check against all open
        # ancestors plus all registered descendants of the root).
        _check_axis_unique(level)

        # Push, run the body so nested decls register against us, pop.
        _stack().append(level)
        try:
            func()
        finally:
            _stack().pop()

        # Attach to parent if we're inside one; otherwise the
        # @allo.target wrapper will pick us up from the function return
        # value.
        parent = _current_level()
        if parent is not None:
            parent.children.append(level)
        return level

    return decorator


def target(name: str,
           *,
           host_memories: Optional[Iterable[Memory]] = None,
           host_ops: Optional[Iterable[Op]] = None,
           caps: Optional[Dict[str, bool]] = None) -> Callable:
    """Outermost decorator. Wires up the synthetic host root and
    returns a ``Target`` object compatible with the existing backends.

    ``host_memories`` / ``host_ops`` mirror ``build_from_grid``'s
    kwargs — they encode the synthetic host root level described in
    Report 11 §3.7. Any source op with no device-level implementation
    lifts to the host root at lowering time (already handled by
    ``pim_dsl/lowering.py``).
    """

    def decorator(func_or_level) -> Target:
        if isinstance(func_or_level, _PendingLevel):
            root_level = func_or_level
        elif callable(func_or_level):
            # A user wrote `@allo.target(...)` alone (without a chained
            # @allo.unit). Interpret the function as if it were an
            # unit(mapping=[1])-wrapped declaration.
            root_level = _PendingLevel(
                axis=getattr(func_or_level, "__name__", "root") or "root",
                extents=[1], mode="mimd")
            _check_axis_unique(root_level)
            _stack().append(root_level)
            try:
                func_or_level()
            finally:
                _stack().pop()
        else:
            raise TypeError(
                f"@allo.target expected a @allo.unit-decorated "
                f"function, got {type(func_or_level).__name__}")

        root_grid = _to_grid(root_level)
        t = build_from_grid(
            name, root_grid,
            caps=caps,
            host_memories=list(host_memories) if host_memories else None,
            host_ops=list(host_ops) if host_ops else None,
        )

        # Stash the pending-tree for downstream passes that want the
        # axis / mode information the flat `Target` does not carry.
        t.axes = _collect_axes(root_level)
        t.modes = _collect_modes(root_level)
        t.tn_root = root_level
        return t

    return decorator


def memory(name: str, **kw) -> Memory:
    """Declare a memory at the enclosing @allo.unit's level."""
    lvl = _require_current("memory")
    norm = _normalize_memory_kwargs(name, kw)
    m = Memory(name=name, **norm["_known"])
    # Park any unknown attributes on the instance so backends that care
    # can read them (e.g., `bits=1` on GSI VR bitlines).
    for k, v in norm["_extras"].items():
        setattr(m, k, v)
    lvl.memories.append(m)
    return m


def op(name: str, **kw) -> Op:
    """Declare an op issuable at the enclosing @allo.unit's level."""
    lvl = _require_current("op")
    norm = _normalize_op_kwargs(name, kw)
    o = Op(name=name, **norm["_known"])
    for k, v in norm["_extras"].items():
        setattr(o, k, v)
    lvl.ops.append(o)
    return o


def stream(name: str, **kw) -> Dict[str, Any]:
    """Declare a typed physical link between sibling instances of the
    enclosing @allo.unit level. Rejected at simd / simd-lockstep levels
    (Rule R4)."""
    lvl = _require_current("stream")
    if lvl.mode in ("simd", "simd-lockstep"):
        raise TypeError(
            f"allo.stream({name!r}, ...) is not legal inside a "
            f"@allo.unit(mode={lvl.mode!r}) level (Rule R4). Streams "
            "are MIMD-only constructs.")
    entry = {"name": name, **kw}
    lvl.streams.append(entry)
    return entry


def cost(**kw) -> Dict[str, Any]:
    """Attach per-level cost attributes (``traverse``, ``broadcast``,
    ``reduce``, or anything else) to the enclosing @allo.unit level."""
    lvl = _require_current("cost")
    lvl.cost_attrs.update(kw)
    return lvl.cost_attrs


# ---------------------------------------------------------------------------
# Flattening: _PendingLevel tree -> Grid/Leaf tree
# ---------------------------------------------------------------------------


def _to_grid(level: _PendingLevel) -> Union[Grid, Leaf]:
    """Turn a _PendingLevel tree into the existing ``Grid``/``Leaf``
    form so ``build_from_grid`` can consume it. The conversion rule:

    - A level with no children collapses to a single
      ``Grid(extent, axis, child=Leaf(memory=..., ops=...))``.
      Memories and ops attached to the level go on the Leaf, because
      that matches the flat backends' expectation (leaves carry the op
      set).
    - A level with one child becomes a
      ``Grid(extent, axis, child=<child Grid/Leaf>)`` with its memories
      attached to the Grid node. The child recurses.
    - A level with multiple children is not yet supported (SPMW today
      assumes regular grids, Report 11 §8 item 2). Raise.
    """
    if len(level.extents) != 1:
        # Fold multi-extent mapping into a chain of single-extent
        # levels. This mirrors `Grid`'s own tuple-expansion, but we
        # retain the axis name on the *innermost* expanded level (all
        # share the same name, which the uniqueness check tolerates
        # because we only record each once in `_collect_axes` below).
        raise NotImplementedError(
            f"@allo.unit(mapping={level.extents!r}) with rank > 1 is "
            "not supported in Phase 1. Nest @allo.unit decorators "
            "instead.")

    extent = level.extents[0]
    memories = list(level.memories)
    ops = list(level.ops)

    if not level.children:
        # Leaf: memory + ops go on the Leaf so flat backends see them.
        leaf = Leaf(memory=memories, ops=ops)
        g = Grid(extent, level.axis, leaf)
        _attach_level_meta(g, level)
        return g

    if len(level.children) > 1:
        raise NotImplementedError(
            f"@allo.unit {level.axis!r} declares {len(level.children)} "
            "nested @allo.unit children. Phase 1 supports a "
            "single-child chain; see Report 11 §8 item 2.")

    child_node = _to_grid(level.children[0])
    g = Grid(extent, level.axis, child_node, memory=memories, ops=ops)
    _attach_level_meta(g, level)
    return g


def _attach_level_meta(grid_node: Grid, level: _PendingLevel) -> None:
    """Record mode / stream / cost / extra decorator state as free
    attributes on the Grid node. Existing backends ignore these; new
    passes can read them."""
    grid_node.mode = level.mode
    grid_node.streams = list(level.streams)
    for k, v in level.cost_attrs.items():
        setattr(grid_node, f"cost_{k}", v)


# ---------------------------------------------------------------------------
# Helpers for uniqueness & axis collection
# ---------------------------------------------------------------------------


def _check_axis_unique(new_level: _PendingLevel) -> None:
    """Rule R2: axis names within one target tree are unique. Check
    against all ancestors on the stack (they're being built) and, once
    a parent exists, recursively against everything already registered
    in that parent's subtree."""
    seen: Dict[str, None] = {}
    for anc in _stack():
        if anc.axis in seen:
            # Can't actually happen: the stack reflects active
            # decorators so duplicates there would already have been
            # caught when the descendant decorator fired. Keep the
            # branch for safety.
            raise ValueError(
                f"duplicate axis name {anc.axis!r} in target tree (R2)")
        seen[anc.axis] = None
    if new_level.axis in seen:
        raise ValueError(
            f"duplicate axis name {new_level.axis!r}: already declared "
            "by an enclosing @allo.unit in this target tree (R2)")
    # Also walk already-registered children of the current level to
    # catch sibling duplicates.
    cur = _current_level()
    if cur is not None:
        for sib in _walk(cur):
            if sib.axis in seen:
                continue
            if sib.axis == new_level.axis:
                raise ValueError(
                    f"duplicate axis name {new_level.axis!r}: already "
                    "declared by a sibling @allo.unit in this target "
                    "tree (R2)")


def _walk(level: _PendingLevel):
    yield level
    for c in level.children:
        yield from _walk(c)


def _collect_axes(root: _PendingLevel) -> List[str]:
    return [lvl.axis for lvl in _walk(root)]


def _collect_modes(root: _PendingLevel) -> Dict[str, str]:
    return {lvl.axis: lvl.mode for lvl in _walk(root)}


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "target", "unit", "memory", "op", "stream", "cost",
]
