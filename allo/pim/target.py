"""Target description primitives: Memory, Op, Pattern, Target, plus the
nested grid-tree form (`Grid`, `Leaf`, `grid`, `build_from_grid`) — a view
over the same underlying `Target`. Both forms coexist; see
`backends/README.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .ops import SrcOp


# ------------------------------- Memory --------------------------------------

@dataclass
class Memory:
    name: str
    capacity_bytes: int
    lanes: int = 1
    dtype: str = "fp16"
    parallel_units: int = 1
    scope: str = "bank"              # "host" | "channel" | "bank" | "grf"


# ------------------------------- Op -------------------------------------------

@dataclass
class Op:
    """A target instruction.

    `emit` is a format-string template; `render(args)` does `.format(**args)`.
    Fields the template omits are silently ignored.

    Cost model:
      - `cycles_per_elem > 0` → calibrated model, cycles = latency + cycles_per_elem * n_elems.
        Set by E5 calibration against real simulator runs. Use for ops that have
        been measured.
      - Else → analytic fallback, latency + ceil(issues/throughput) - 1 where
        issues = ceil(n_elems / parallel_units / lanes).
    """
    name: str
    lanes: int
    latency: int
    throughput: float = 1.0
    cycles_per_elem: float = 0.0
    energy_pJ: float = 0.0
    emit: str = ""

    def render(self, args: Dict) -> str:
        try:
            return self.emit.format(**args)
        except (KeyError, IndexError):
            return self.emit

    def cycles(self, n_elems: int, parallel_units: int) -> int:
        if self.cycles_per_elem > 0.0:
            return int(round(self.latency + self.cycles_per_elem * n_elems))
        per_unit = math.ceil(n_elems / max(parallel_units, 1))
        issues = math.ceil(per_unit / max(self.lanes, 1))
        return int(self.latency + math.ceil(issues / self.throughput) - 1)


# ------------------------------- Pattern --------------------------------------

@dataclass
class Pattern:
    """Source-op -> target-instr-sequence rewrite.

    `match`  : (SrcOp) -> bool.
    `lower`  : (SrcOp, Target) -> list[(op_name, args)].
    `compute`: optional override for tensor semantics. If omitted, the
        interpreter uses `src.compute(state)` (typed subclass in `allo.pim.ops`).
        If neither is present, `execute()` raises.
    """
    match: Callable[[SrcOp], bool]
    lower: Callable[[SrcOp, "Target"], List[Tuple[str, Dict]]]
    compute: Optional[Callable[[SrcOp, Dict], None]] = None
    name: str = ""


# ------------------------------- Target ---------------------------------------

class Target:
    """A PIM target description.

    API:
      - `t.memory(name, capacity_bytes=..., ...)` declares a memory region.
      - `t.op(name, lanes=, latency=, energy_pJ=, emit="template {field}")`.
      - `t.pattern(match=, lower=, name=, compute=None)`.
      - `t.cap(has_exp=True, ...)` capability flags.
      - Cost: analytic by default (latency/throughput/lanes). Override with
        `t.set_cost_model(fn)` for learned / table-based cost.
    """

    def __init__(self, name: str, parallel_units: int = 1):
        self.name = name
        self.parallel_units = parallel_units
        self.memories: Dict[str, Memory] = {}
        self.ops: Dict[str, Op] = {}
        self.patterns: List[Pattern] = []
        self._cost_model_fn: Optional[Callable] = None
        self.caps: Dict[str, bool] = {}

    # builders -----------------------------------------------------------------

    def memory(self, name: str, **kw) -> Memory:
        m = Memory(name=name, **kw)
        self.memories[name] = m
        return m

    def op(self, name: str, **kw) -> Op:
        o = Op(name=name, **kw)
        self.ops[name] = o
        return o

    def pattern(self, match, lower, compute=None, name: str = "") -> Pattern:
        p = Pattern(match=match, lower=lower, compute=compute,
                    name=name or getattr(lower, "__name__", "pattern"))
        self.patterns.append(p)
        return p

    def cap(self, **kw):
        self.caps.update(kw)

    # queries ------------------------------------------------------------------

    def find_pattern(self, src: SrcOp) -> Optional[Pattern]:
        for p in self.patterns:
            if p.match(src):
                return p
        return None

    # cost ---------------------------------------------------------------------

    def set_cost_model(self, fn: Callable):
        """Install a non-default cost model: fn(target, instrs, src) -> dict
        with at least 'cycles' and 'energy_pJ' keys."""
        self._cost_model_fn = fn

    def analyze_cost(self, instrs, src: SrcOp) -> Dict:
        if self._cost_model_fn is not None:
            return self._cost_model_fn(self, instrs, src)
        n_elems = 1
        for d in src.shape:
            n_elems *= d
        total_cycles = 0
        total_energy = 0.0
        per_op = []
        for (opname, args) in instrs:
            op = self.ops[opname]
            elems = args.get("n_elems", n_elems)
            cyc = op.cycles(elems, self.parallel_units)
            e = op.energy_pJ * elems
            per_op.append({"op": opname, "cycles": cyc,
                           "energy_pJ": e, "elems": elems})
            total_cycles += cyc
            total_energy += e
        return {"cycles": total_cycles, "energy_pJ": total_energy,
                "per_op": per_op}


# ------------------------------- Grid tree ------------------------------------
#
# Report 07 §2: a PIM target is a nested grid. A `Grid` node declares an extent
# and optional per-level memory / sync / broadcast attributes; its single
# `child` is another `Grid` or a `Leaf`. A `Leaf` holds leaf-level memories and
# ops (GRF slots, per-bank MAC registers, the ISA that fires at the leaf).
#
# `build_from_grid` flattens a tree into the same `Target` today's compiler
# consumes: memories attached at a level get `parallel_units` = product of
# extents from the root down to that level; ops at any level are stamped into
# `target.ops`. This keeps the grid-tree form a *view*, not a fork.
# ------------------------------------------------------------------------------


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _extent(shape) -> int:
    if isinstance(shape, (tuple, list)):
        p = 1
        for s in shape:
            p *= int(s)
        return p
    return int(shape)


class Leaf:
    """Grid-tree leaf: memories and the ops that fire there."""
    def __init__(self, memory=None, ops=None):
        self.memory: List[Memory] = _as_list(memory)
        self.ops: List[Op] = _as_list(ops)


class Grid:
    """Grid-tree node. `shape` is the level's extent (int) or a tuple that
    expands into a chain of single-extent levels (e.g. `shape=(64, 16)`,
    `name=["channel", "bank"]`). `child` is the next `Grid` or a `Leaf`."""
    def __init__(self, shape, name, child,
                 memory=None, sync=None, broadcast_ops=None, ops=None):
        if isinstance(shape, (tuple, list)) and len(shape) > 1:
            names = name if isinstance(name, (tuple, list)) else [name] * len(shape)
            if len(names) != len(shape):
                raise ValueError("shape/name length mismatch")
            tail = child
            for s, n in zip(reversed(shape), reversed(names)):
                tail = Grid(int(s), n, tail)
            # The head of the chain is `tail`; adopt its fields. Any level
            # attributes (memory, sync, broadcast_ops, ops) apply to the head.
            self.shape = tail.shape
            self.name = tail.name
            self.child = tail.child
        else:
            self.shape = int(shape[0] if isinstance(shape, (tuple, list)) else shape)
            self.name = (name[0] if isinstance(name, (tuple, list)) else name)
            self.child = child
        self.memory: List[Memory] = _as_list(memory)
        self.sync: Optional[str] = sync
        self.broadcast_ops: List[str] = list(broadcast_ops) if broadcast_ops else []
        self.ops: List[Op] = _as_list(ops)


def grid(shape, name, child, **kw) -> Grid:
    """Convenience constructor: `spmw.grid(...)`. Equivalent to `Grid(...)`."""
    return Grid(shape, name, child, **kw)


def _total_extent(node) -> int:
    if isinstance(node, Leaf):
        return 1
    return _extent(node.shape) * _total_extent(node.child)


def build_from_grid(name: str, root: Union[Grid, Leaf], *,
                    caps: Optional[Dict[str, bool]] = None,
                    host_memories: Optional[Iterable[Memory]] = None,
                    host_ops: Optional[Iterable[Op]] = None) -> Target:
    """Flatten a grid tree into a `Target`. Memories attached to a tree node
    receive `parallel_units = product of extents from root to that node`;
    ops at any level are stamped into `target.ops`. `host_memories` /
    `host_ops` bypass the tree (they keep their constructed `parallel_units`,
    which defaults to 1) — use for host-fallback scopes.
    """
    total = _total_extent(root)
    t = Target(name, parallel_units=total)

    def walk(node, pu_so_far: int):
        if isinstance(node, Leaf):
            for m in node.memory:
                m.parallel_units = pu_so_far
                t.memories[m.name] = m
            for op in node.ops:
                t.ops[op.name] = op
            return
        pu_here = pu_so_far * _extent(node.shape)
        for m in node.memory:
            m.parallel_units = pu_here
            t.memories[m.name] = m
        for op in node.ops:
            t.ops[op.name] = op
        walk(node.child, pu_here)

    # Host memories are attached to the "synthetic root above the chip"
    # (report 07 §2.6) — insert them first so they sit at the top of the
    # target's memory list. Host ops append last.
    for m in (host_memories or []):
        t.memories[m.name] = m

    walk(root, 1)

    for op in (host_ops or []):
        t.ops[op.name] = op

    if caps:
        t.cap(**caps)
    return t
