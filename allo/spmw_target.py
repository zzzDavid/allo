# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW target abstraction (Tenon `@tn.target` / `@tn.unit`).

Step B of report 16: pure-Python data model for the target tree.
Decorators run their bodies at registration time and build a tree of
`Unit` objects with attached `Memory` and `Register` handles. Indexing
a `Memory` with a symbolic uid returns a `MemoryRef` whose index is a
`SymExpr`; lowering (Step E) substitutes concrete coordinates.

Step C will add `move`, `op`, `any_`, `or_` onto units.
"""


_target_stack: list = []


class SymExpr:
    """Symbolic arithmetic over `UnitId`s and ints."""

    __slots__ = ("op", "args")

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __add__(self, other):
        return SymExpr("add", self, other)

    def __radd__(self, other):
        return SymExpr("add", other, self)

    def __sub__(self, other):
        return SymExpr("sub", self, other)

    def __rsub__(self, other):
        return SymExpr("sub", other, self)

    def __mul__(self, other):
        return SymExpr("mul", self, other)

    def __rmul__(self, other):
        return SymExpr("mul", other, self)

    def __floordiv__(self, other):
        return SymExpr("floordiv", self, other)

    def __mod__(self, other):
        return SymExpr("mod", self, other)

    def __repr__(self):
        return f"SymExpr({self.op}, {', '.join(map(repr, self.args))})"


class UnitId(SymExpr):
    """Symbolic unit-id at a given level of the unit-tree (0 = outermost)."""

    __slots__ = ("level", "unit")

    def __init__(self, level, unit):
        # bypass SymExpr.__init__; UnitId is a leaf
        self.op = "uid"
        self.args = (level,)
        self.level = level
        self.unit = unit

    def __repr__(self):
        unit_name = self.unit.name if self.unit is not None else "<anon>"
        return f"UnitId(level={self.level}, unit={unit_name})"


class Memory:
    """A memory cell array attached to a unit.

    Geometry is stored as a free-form `dict` (`self.geometry`) so each
    target can declare its own axis names. For backward compat the
    Samsung-style axes (`banks`, `rows`, `cols`, `width`) are also
    bound as attributes — they are `None` when the target doesn't
    declare that axis. Indexing with `m[i]` selects bank i (or the
    target's outer axis) and returns a `MemoryRef`.
    """

    def __init__(self, owner, *, name=None, **geometry):
        self.owner = owner
        self.name = name
        self.geometry = dict(geometry)
        # Convenience attributes for Samsung-shaped memories. Other
        # targets that don't declare these axes get `None`.
        self.banks = geometry.get("banks")
        self.rows = geometry.get("rows")
        self.cols = geometry.get("cols")
        self.width = geometry.get("width")

    def __getitem__(self, idx):
        return MemoryRef(self, idx)

    def __repr__(self):
        n = self.name or "<anon>"
        geom = ", ".join(f"{k}={v}" for k, v in self.geometry.items())
        return f"Memory({n}, {geom})"


class MemoryRef:
    """A symbolic reference to one bank (or sub-region) of a `Memory`."""

    __slots__ = ("memory", "idx")

    def __init__(self, memory, idx):
        self.memory = memory
        self.idx = idx

    def __repr__(self):
        return f"MemoryRef({self.memory.name}[{self.idx!r}])"


class Register:
    """A vector register attached to a unit. `lanes` × `width` bits."""

    def __init__(self, owner, lanes, width, name=None):
        self.owner = owner
        self.lanes = lanes
        self.width = width
        self.name = name

    def __repr__(self):
        n = self.name or "<anon>"
        return f"Register({n}, lanes={self.lanes}, width={self.width})"


class AnyOf:
    """Pattern: src/dst may be any handle from `candidates`.

    `candidates` is either a `Memory` (=any bank of that memory) or a
    list of explicit handles (e.g. `[grf_a, grf_b]`).
    """

    __slots__ = ("candidates",)

    def __init__(self, candidates):
        self.candidates = candidates

    def __repr__(self):
        return f"AnyOf({self.candidates!r})"


class OrOf:
    """Pattern: matches any of the given subpatterns (handles or AnyOf)."""

    __slots__ = ("alternatives",)

    def __init__(self, alternatives):
        self.alternatives = list(alternatives)

    def __repr__(self):
        return f"OrOf({self.alternatives!r})"


class Move:
    """A data-movement primitive declared on a unit (LD/ST/etc.)."""

    def __init__(self, owner, name, src, dst, emit=None, cycles=None):
        self.owner = owner
        self.name = name
        self.src = src
        self.dst = dst
        self.emit = emit
        self.cycles = cycles  # populated by fixtures / cost modules

    def __repr__(self):
        return f"Move({self.name!r}, src={self.src!r}, dst={self.dst!r})"


class Op:
    """A compute-op primitive declared on a unit (MUL/MAC/...).

    `fn` is a Python lambda over the operand values that encodes the
    op's semantics; the lowering pass (Step E) matches workload
    expressions against these lambdas. `emit` is the codegen callback
    that backends invoke per matched site; see report 16.
    """

    def __init__(self, owner, name, src, dst, fn, accumulates=False, emit=None, cycles=None):
        self.owner = owner
        self.name = name
        self.src = src
        self.dst = dst
        self.fn = fn
        self.accumulates = accumulates
        self.emit = emit
        self.cycles = cycles  # populated by fixtures / cost modules

    def __repr__(self):
        return f"Op({self.name!r}, accumulates={self.accumulates})"


class Unit:
    """A node in the target's unit tree."""

    def __init__(self, name, mapping, parent=None):
        self.name = name
        self.mapping = list(mapping) if mapping is not None else []
        self.parent = parent
        self.children: list[Unit] = []
        self.memories: dict[str, Memory] = {}
        self.registers: dict[str, Register] = {}
        self.moves: dict[str, Move] = {}
        self.ops: dict[str, Op] = {}

    @property
    def level(self):
        n, p = 0, self.parent
        while p is not None:
            n += 1
            p = p.parent
        return n  # root has level 0

    def __repr__(self):
        return f"Unit({self.name}, mapping={self.mapping}, level={self.level})"


class Target:
    """Top-level target spec. `root` is a synthetic Unit holding the tree."""

    def __init__(self, name, root):
        self.name = name
        self.root = root
        # flat name → handle index (memories + registers, all units)
        self._handles: dict[str, object] = {}
        for u in self._walk():
            for nm, h in u.memories.items():
                self._handles[nm] = h
            for nm, h in u.registers.items():
                self._handles[nm] = h

    def _walk(self):
        stack = [self.root]
        while stack:
            u = stack.pop()
            yield u
            stack.extend(u.children)

    def __getattr__(self, name):
        # Falls back here only if normal attribute lookup fails.
        try:
            return self.__dict__["_handles"][name]
        except KeyError as e:
            raise AttributeError(
                f"Target {self.name!r} has no handle named {name!r}"
            ) from e

    def move(self, name):
        for u in self._walk():
            if name in u.moves:
                return u.moves[name]
        raise KeyError(f"Target {self.name!r} has no move named {name!r}")

    def op(self, name):
        for u in self._walk():
            if name in u.ops:
                return u.ops[name]
        raise KeyError(f"Target {self.name!r} has no op named {name!r}")

    def __repr__(self):
        return f"Target({self.name!r}, root={self.root!r})"


# ---------------- decorators / primitives ---------------- #


def target(name):
    """Decorator: builds a `Target` from the function body.

    The decorated function is run once at decoration time inside a
    fresh `_target_stack` rooted at a synthetic Unit; child units
    attach via `@unit`, and memories/registers attach via `memory`/`reg`.
    """

    def decorator(fn):
        root = Unit(name, mapping=[1])
        _target_stack.append(root)
        try:
            fn()
        finally:
            popped = _target_stack.pop()
            assert popped is root
        return Target(name, root)

    return decorator


def unit(mapping):
    """Decorator: registers a child Unit on the enclosing scope."""

    def decorator(fn):
        if not _target_stack:
            raise RuntimeError("@allo.unit must be used inside @allo.target")
        parent = _target_stack[-1]
        u = Unit(fn.__name__, mapping=mapping, parent=parent)
        parent.children.append(u)
        _target_stack.append(u)
        try:
            fn()
        finally:
            popped = _target_stack.pop()
            assert popped is u
        return u

    return decorator


def memory(*, name=None, **geometry):
    """Attach a Memory to the current unit; return a handle.

    Geometry is target-specific — Samsung HBM-PIM uses
    `banks=…, rows=…, cols=…, width=…`; other targets (AiM, UPMEM,
    APU v1/v2) pass their own axis names. Stored on `Memory.geometry`.
    """
    if not _target_stack:
        raise RuntimeError("allo.memory must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    m = Memory(cur, name=name, **geometry)
    if name is not None:
        if name in cur.memories:
            raise ValueError(f"duplicate memory name {name!r} on unit {cur.name!r}")
        cur.memories[name] = m
    return m


# Tenon-style short alias — the public name exposed as `allo.mem`. The
# longer `memory` name remains the in-module canonical (it reads more
# naturally inside a @target body) but the legacy `allo.memory` module
# attribute must not shadow it, so re-export lives under `mem`.
mem = memory


def reg(lanes, width, name=None):
    """Attach a Register to the current unit; return a handle."""
    if not _target_stack:
        raise RuntimeError("allo.reg must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    r = Register(cur, lanes, width, name=name)
    if name is not None:
        if name in cur.registers:
            raise ValueError(f"duplicate register name {name!r} on unit {cur.name!r}")
        cur.registers[name] = r
    return r


def get_uid():
    """Return a tuple of symbolic uids: one per ancestor @unit (outer→inner).

    The synthetic root from @target is not exposed.
    """
    if not _target_stack:
        raise RuntimeError("allo.get_uid must be called inside @allo.unit")
    cur = _target_stack[-1]
    chain = []
    u = cur
    while u is not None:
        chain.append(u)
        u = u.parent
    # outer-first, then strip synthetic root
    chain = list(reversed(chain))[1:]
    return tuple(UnitId(level=i, unit=chain[i]) for i in range(len(chain)))


# ---------------- step C: move / op / any_ / or_ ---------------- #


def move(name, src, dst, emit=None, cycles=None):
    """Attach a Move to the current unit; return the handle."""
    if not _target_stack:
        raise RuntimeError("allo.move must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    if name in cur.moves:
        raise ValueError(f"duplicate move name {name!r} on unit {cur.name!r}")
    m = Move(cur, name, src, dst, emit=emit, cycles=cycles)
    cur.moves[name] = m
    return m


def op(name, src, dst, fn, accumulates=False, emit=None, cycles=None):
    """Attach an Op to the current unit; return the handle."""
    if not _target_stack:
        raise RuntimeError("allo.op must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    if name in cur.ops:
        raise ValueError(f"duplicate op name {name!r} on unit {cur.name!r}")
    o = Op(cur, name, src, dst, fn, accumulates=accumulates, emit=emit, cycles=cycles)
    cur.ops[name] = o
    return o


def any_(candidates):
    """Pattern: src/dst may be any handle in `candidates`.

    The paper writes `tn.any(...)`; Python's `any` is a builtin so we
    use `any_` here. `candidates` may be a `Memory` (any of its banks)
    or a list of register/memory handles.
    """
    return AnyOf(candidates)


def or_(*alternatives):
    """Pattern: matches any of the given subpatterns.

    The paper writes `tn.or(...)`; `or` is a Python keyword so we use
    `or_` here.
    """
    return OrOf(alternatives)
