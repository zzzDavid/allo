# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW target abstraction (Tenon `@tn.target` / `@tn.unit`).

Step B of report 16: pure-Python data model for the target tree.
Decorators run their bodies at registration time and build a tree of
`Unit` objects with attached `Memory` and `Register` handles. Indexing
a `Memory` with a symbolic uid returns a `MemoryRef` whose index is a
`SymExpr`; lowering (Step E) substitutes concrete coordinates.

The target tree contains structural and functional hardware facts only. Cycle
behavior is authored separately as an executable ``CostSpec``.
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
        self.capacity = int(geometry.get("ports", 1))
        if self.capacity <= 0:
            raise ValueError("memory ports must be positive")

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
    """A vector register attached to a unit. `lanes` × `width` bits.

    `slots` is the number of addressable register entries the allocator
    may pack distinct live values into. It defaults to `lanes` (the common
    case where the SIMD-lane count equals the addressable depth, e.g.
    Samsung grf / UPMEM gprs / APU vrs). A backend whose addressable GPR
    depth is a *different* axis from the SIMD width declares `slots`
    explicitly (AiM: 31 addressable MAC-accumulator GPRs vs 16 SIMD lanes,
    JSSC 2023 §IV) so the allocator's capacity is tree-derived, not pasted.
    """

    def __init__(self, owner, lanes, width, name=None, slots=None, ports=1):
        self.owner = owner
        self.lanes = lanes
        self.width = width
        self.name = name
        # Addressable depth for register allocation; defaults to lanes.
        self.slots = lanes if slots is None else slots
        self.capacity = int(ports)
        if self.capacity <= 0:
            raise ValueError("register ports must be positive")

    def __repr__(self):
        n = self.name or "<anon>"
        return (
            f"Register({n}, lanes={self.lanes}, width={self.width}, slots={self.slots})"
        )


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


class _Verb:
    """A first-class host-transfer verb tag (broadcast / scatter / gather /
    move_only). Identity *is* the tag: the dispatcher compares verbs with
    `is`, never by name string. The `name` field is for diagnostics only.
    """

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<verb {self.name}>"


# Module-level verb sentinels (re-exported from `allo`). `move_only` is the
# default for every existing `allo.move(...)` — p2p / microcode upload with no
# collective semantics — so adding `verb=` is purely additive.
broadcast = _Verb("broadcast")
scatter = _Verb("scatter")
gather = _Verb("gather")
move_only = _Verb("move_only")


class Move:
    """A data-movement primitive declared on a unit (LD/ST/etc.).

    `verb` is a first-class :class:`_Verb` tag (default :data:`move_only`);
    the host-transfer dispatcher narrows candidate moves by `move.verb is
    record.verb`. Device moves leave it at the default.
    """

    def __init__(self, owner, name, src, dst, emit=None, verb=move_only, capacity=1):
        self.owner = owner
        self.name = name
        self.src = src
        self.dst = dst
        self.emit = emit
        self.verb = verb
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("move capacity must be positive")

    def __repr__(self):
        return f"Move({self.name!r}, src={self.src!r}, dst={self.dst!r}, verb={self.verb.name})"


class Op:
    """A compute-op primitive declared on a unit (MUL/MAC/...).

    `fn` is a Python lambda over the operand values that encodes the
    op's semantics; the lowering pass (Step E) matches workload
    expressions against these lambdas. `emit` is the codegen callback
    that backends invoke per matched site; see report 16.
    """

    def __init__(
        self,
        owner,
        name,
        src,
        dst,
        fn,
        accumulates=False,
        emit=None,
        capacity=1,
        matchable=True,
    ):
        self.owner = owner
        self.name = name
        self.src = src
        self.dst = dst
        self.fn = fn
        self.accumulates = accumulates
        self.emit = emit
        self.capacity = int(capacity)
        self.matchable = bool(matchable)
        if self.capacity <= 0:
            raise ValueError("operation capacity must be positive")

    def __repr__(self):
        return f"Op({self.name!r}, accumulates={self.accumulates})"


class Unit:
    """A node in the target's unit tree."""

    def __init__(self, name, mapping, parent=None, mode=None, capacity=1):
        self.name = name
        self.axes = dict(mapping) if isinstance(mapping, dict) else {}
        self.mapping = (
            list(mapping.values())
            if isinstance(mapping, dict)
            else (list(mapping) if mapping is not None else [])
        )
        self.parent = parent
        # `mode="host"` marks a host-side staging node (design 05 §Q3); the
        # device tree hangs under it. `None` for ordinary device units.
        self.mode = mode
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("unit capacity must be positive")
        # Host-collective interface instance attached by @allo.host_xcel
        # (only ever non-None on a mode="host" unit).
        self.host_xcel = None
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

    def unit(self, name):
        """Look up one uniquely named structural unit."""
        matches = [unit for unit in self._walk() if unit.name == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"Target {self.name!r} has no unit named {name!r}")
        raise KeyError(f"Target {self.name!r} has ambiguous unit name {name!r}")

    def work_grid(self):
        """Derive the work-item grid implied by the @allo.unit tree.

        Returns (factors, product) where `factors` is the per-unit mapping
        factor list walked outer->inner (root [1] elided; host nodes have no
        mapping and contribute nothing), and `product` is their product == the
        number of PEs == the canonical full-grid work-id count. This is the
        declarative counterpart of any cost program's spatial instance map.
        """
        factors = []
        for u in self._walk():
            if getattr(u, "mode", None) == "host":
                continue
            for f in u.mapping:
                if f != 1:  # root's synthetic [1] and unit [1]s elide
                    factors.append(f)
        product = 1
        for f in factors:
            product *= f
        return factors, product

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


def unit(mapping=None, *, mode=None, capacity=1):
    """Decorator: registers a child Unit on the enclosing scope.

    `mode="host"` (default `None`) marks a host-side staging node; the
    device unit tree hangs under it (design 05 §Q3). A host unit needs no
    `mapping`. FPGA/AIE never call `@allo.unit`, so the new kwarg is inert
    outside the spmw path.
    """

    def decorator(fn):
        if not _target_stack:
            raise RuntimeError("@allo.unit must be used inside @allo.target")
        parent = _target_stack[-1]
        u = Unit(
            fn.__name__,
            mapping=mapping,
            parent=parent,
            mode=mode,
            capacity=capacity,
        )
        parent.children.append(u)
        _target_stack.append(u)
        try:
            fn()
        finally:
            popped = _target_stack.pop()
            assert popped is u
        return u

    return decorator


def _walk_unit(u):
    stack = [u]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.children)


class DeviceScope:
    """Handle returned by `@allo.device`.

    Names a device sub-tree and exposes its declared memories/registers as
    attributes, so a sibling host scope can target device memory by name --
    e.g. `hbm_pim.banks`. The underlying `Unit` is `mode="device"`, a grouping
    node that is transparent to `get_uid()` / `work_grid()`.
    """

    def __init__(self, unit):
        self.__dict__["_unit"] = unit
        handles: dict[str, object] = {}
        for u in _walk_unit(unit):
            handles.update(u.memories)
            handles.update(u.registers)
        self.__dict__["_handles"] = handles

    def __getattr__(self, name):
        try:
            return self.__dict__["_handles"][name]
        except KeyError as e:
            raise AttributeError(
                f"device scope {self.__dict__['_unit'].name!r} has no "
                f"memory/register named {name!r}; declared: "
                f"{sorted(self.__dict__['_handles'])}"
            ) from e

    def __repr__(self):
        return f"DeviceScope({self._unit.name!r}, " f"handles={sorted(self._handles)})"


def device(fn):
    """Decorator: a named device scope under the `@allo.target` root.

    Runs the body to build the device unit sub-tree (a grouping node,
    `mapping=[1]`, `mode="device"`, transparent to `get_uid()`/`work_grid()`)
    and returns a `DeviceScope` exposing its memories/registers by name. A
    sibling `@allo.unit(mode="host")` scope then declares host<->device
    transfers as `allo.move`s naming `<device>.banks` etc. as an endpoint.
    """
    if not _target_stack:
        raise RuntimeError("@allo.device must be used inside @allo.target")
    parent = _target_stack[-1]
    u = Unit(fn.__name__, mapping=[1], parent=parent, mode="device")
    parent.children.append(u)
    _target_stack.append(u)
    try:
        fn()
    finally:
        popped = _target_stack.pop()
        assert popped is u
    return DeviceScope(u)


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


def reg(lanes, width, name=None, slots=None, ports=1):
    """Attach a Register to the current unit; return a handle.

    `slots` overrides the allocator-visible addressable depth (defaults to
    `lanes`); pass it when the GPR depth differs from the SIMD-lane count.
    """
    if not _target_stack:
        raise RuntimeError("allo.reg must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    r = Register(cur, lanes, width, name=name, slots=slots, ports=ports)
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
    # outer-first, then strip the synthetic @target root AND any @device
    # grouping scope (a device scope is transparent to the work-id chain so
    # the real mapped units -- pseudo_channel, pim, ... -- keep their depth).
    chain = [
        u
        for u in reversed(chain)
        if u.parent is not None and getattr(u, "mode", None) != "device"
    ]
    return tuple(UnitId(level=i, unit=chain[i]) for i in range(len(chain)))


# ---------------- step C: move / op / any_ / or_ ---------------- #


# Host-staging stub move-name prefixes/names: a `src==dst` move under one of
# these names was a fake host self-move (design 05 §0). After task-017 these
# are deleted; the validation rejects any re-introduction (§Q3).
_HOST_STAGING_STUB_NAMES = ("CRF_TRIGGER",)
_HOST_STAGING_STUB_PREFIXES = ("PRELOAD_", "READBACK_")


def _handle_unit(h):
    """The owning `Unit` of a concrete handle (Register / MemoryRef /
    Memory), or None for patterns (`AnyOf`/`OrOf`/tuple) that have no single
    owner."""
    if isinstance(h, Register):
        return h.owner
    if isinstance(h, MemoryRef):
        return h.memory.owner
    if isinstance(h, Memory):
        return h.owner
    return None


def _is_host_side(unit) -> bool:
    """True if `unit` or any ancestor is a `mode="host"` node."""
    while unit is not None:
        if getattr(unit, "mode", None) == "host":
            return True
        unit = unit.parent
    return False


def _is_host_staging_stub_name(name: str) -> bool:
    return name in _HOST_STAGING_STUB_NAMES or any(
        name.startswith(p) for p in _HOST_STAGING_STUB_PREFIXES
    )


def _validate_move(name, src, dst, emit, owner):
    """`Move` validation under the explicit host-transfer model.

    1. A `src is dst` self-move with `emit is None` is a no-op stub and is
       rejected (a real device control move like `JUMP` carries a real emit).
    2. A move declared on a **device** scope may not straddle the host<->device
       boundary. A move declared on an `@allo.unit(mode="host")` scope, by
       contrast, IS the host's data-transfer primitive: it legitimately names
       device memory (`hbm_pim.banks`) as one endpoint and host memory as the
       other, so cross-boundary is allowed there.
    """
    if src is dst and emit is None:
        raise ValueError(f"move {name!r} is a no-op self-move (src is dst, emit=None)")
    # Host scope: this move is the host's transfer primitive -> cross-boundary OK.
    if _is_host_side(owner):
        return
    # Device scope: a bare device move must not straddle the host boundary.
    su, du = _handle_unit(src), _handle_unit(dst)
    if su is not None and du is not None:
        if _is_host_side(su) != _is_host_side(du):
            raise ValueError(
                f"move {name!r} straddles the host<->device boundary but is "
                f"declared on a device scope; a host<->device transfer must be "
                f'declared on an @allo.unit(mode="host") scope'
            )


def move(name, src, dst, emit=None, verb=move_only, capacity=1):
    """Attach a Move to the current unit; return the handle.

    `verb` (default :data:`move_only`) is the first-class host-transfer tag the
    `BackendHandle` dispatcher narrows on. Additive: every existing call that
    omits `verb=` keeps `move_only`, so no device move changes meaning.
    """
    if not _target_stack:
        raise RuntimeError("allo.move must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    if name in cur.moves:
        raise ValueError(f"duplicate move name {name!r} on unit {cur.name!r}")
    _validate_move(name, src, dst, emit, cur)
    m = Move(cur, name, src, dst, emit=emit, verb=verb, capacity=capacity)
    cur.moves[name] = m
    return m


def op(
    name,
    src,
    dst,
    fn,
    accumulates=False,
    emit=None,
    capacity=1,
    matchable=True,
):
    """Attach an Op to the current unit; return the handle."""
    if not _target_stack:
        raise RuntimeError("allo.op must be called inside @allo.target/@allo.unit")
    cur = _target_stack[-1]
    if name in cur.ops:
        raise ValueError(f"duplicate op name {name!r} on unit {cur.name!r}")
    o = Op(
        cur,
        name,
        src,
        dst,
        fn,
        accumulates=accumulates,
        emit=emit,
        capacity=capacity,
        matchable=matchable,
    )
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


# ---------------- host-transfer dispatch surface (spec 001) ---------------- #
#
# The workload-facing `backend.*` proxy: a region-body recording singleton
# (D1) whose attribute access yields deferred device-handle tokens and whose
# verb methods (broadcast/scatter/gather/move) record `HostMoveRecord`s. At
# `compile_for_target` a `BackendHandle(target)` replays the records, resolving
# tokens by declared name to the *identical* tree handle and dispatching each
# record to the target's verb-tagged host move by (verb, direction,
# device-handle-identity) — never by string `==`.
#
# NOTE (collision flagged to architect): the spec names the proxy `allo.backend`,
# but `allo.backend` is already the established codegen-backend SUBMODULE
# (allo/backend/__init__.py — llvm/hls/ip/aie; 78 references across the FPGA/AIE
# tree). Re-exporting the proxy as `allo.backend` would shadow that submodule
# and break the FPGA/AIE blast radius the task forbids. The proxy is therefore
# exported under the non-colliding name `host_xfer` (the singleton object), with
# `record_host_moves()` as the recorder scope. See the report's open question.


class HandleToken:
    """A deferred reference to a device handle *by declared name*.

    Emitted by `RecordingBackend.__getattr__` (e.g. `host_xfer.banks` ->
    `HandleToken("banks")`). Resolved to the identical `Memory`/`Register`
    object at compile by `BackendHandle.__getattr__`. The name is an addressing
    convenience; the dispatcher keys on `id()`/`is` of the resolved object.
    """

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"HandleToken({self.name!r})"


class HostMoveRecord:
    """One recorded `host_xfer.<verb>(a, b)` call from a region body.

    `verb` is a `_Verb` sentinel (`broadcast`/`scatter`/`gather`/`move_only`, or
    an undeclared `_Verb` for an unknown verb — which resolves to the D3
    zero-candidate error). `args` is the raw `(a, b)` tuple; exactly one is a
    `HandleToken` (the device endpoint) and one is a workload buffer.
    """

    __slots__ = ("verb", "args")

    def __init__(self, verb, args):
        self.verb = verb
        self.args = tuple(args)

    def __repr__(self):
        return f"HostMoveRecord(verb={self.verb.name}, args={self.args!r})"


# The active recorder list (a stack of lists). `record_host_moves()` pushes a
# fresh list; `RecordingBackend` verb calls append to the top of the stack.
_host_move_recorder_stack: list = []

# Map known verb method names -> the sentinel they stamp.
_KNOWN_VERBS = {
    "broadcast": broadcast,
    "scatter": scatter,
    "gather": gather,
    "move": move_only,
}


class _RecorderScope:
    """Context manager returned by `record_host_moves()`.

    Pushes a fresh record list for the duration of the `with` block; the
    workload's region-body `host_xfer.*` calls append to it. After the block the
    collected records are available as `scope.records` (and the scope itself is
    truthy-iterable over them) so the harness can pass them to
    `compile_for_target(host_moves=...)`.
    """

    __slots__ = ("records",)

    def __init__(self):
        self.records = []

    def __enter__(self):
        _host_move_recorder_stack.append(self.records)
        return self

    def __exit__(self, exc_type, exc, tb):
        popped = _host_move_recorder_stack.pop()
        assert popped is self.records
        return False  # never swallow exceptions

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


def record_host_moves():
    """Open a host-move recording scope (D1 spelling (a)).

    Usage in a workload module::

        with allo.record_host_moves() as hm:
            allo.host_xfer.scatter(W, allo.host_xfer.banks)
            allo.host_xfer.broadcast(x, allo.host_xfer.grf_a)
            allo.host_xfer.gather(y, allo.host_xfer.banks)
        HOST_MOVES = hm

    The harness then threads `HOST_MOVES` into
    `compile_for_target(host_moves=HOST_MOVES)`.
    """
    return _RecorderScope()


class RecordingBackend:
    """The `allo.host_xfer` recording proxy (singleton).

    At decoration time it is unbound to any target. Attribute access returns a
    `HandleToken` (deferred device-handle reference). Calling a verb method
    appends a `HostMoveRecord` to the active recorder scope.
    """

    def __getattr__(self, name):
        # A verb method: return a recorder callable that stamps the right verb.
        if name in _KNOWN_VERBS:
            verb = _KNOWN_VERBS[name]
            return lambda *args: self._record(verb, args)
        # An unknown verb (e.g. `host_xfer.reduce`): stamp a fresh _Verb the
        # target never declares, so D3 verb-narrowing yields zero candidates and
        # raises the uniform "implements no host move for verb" error naming the
        # target. (Distinguished from a handle token by the call: a handle token
        # is never called.) We return a recorder that stamps `_Verb(name)`.
        return _VerbCallOrToken(self, name)

    def _record(self, verb, args):
        if not _host_move_recorder_stack:
            raise RuntimeError(
                "host_xfer.<verb>(...) called outside a record_host_moves() scope"
            )
        rec = HostMoveRecord(verb, args)
        _host_move_recorder_stack[-1].append(rec)
        return rec


class _VerbCallOrToken:
    """Returned for an unrecognised `host_xfer.<name>` attribute.

    It behaves as a `HandleToken` if used as a value (the common case:
    `host_xfer.banks`, `host_xfer.grf_a` — any device-handle name), and as a
    verb recorder if *called* (`host_xfer.reduce(...)` — an undeclared verb).
    This lets the proxy stay agnostic to the target's handle names: every
    non-verb attribute is a deferred handle token until it is called.
    """

    __slots__ = ("_proxy", "_name", "_token")

    def __init__(self, proxy, name):
        self._proxy = proxy
        self._name = name
        self._token = HandleToken(name)

    # --- value-use: delegate to the underlying HandleToken --- #
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"HandleToken({self._name!r})"

    # --- call-use: an undeclared verb -> record a fresh _Verb(name) --- #
    def __call__(self, *args):
        return self._proxy._record(_Verb(self._name), args)


# The single workload-facing proxy instance. Re-exported from `allo` as
# `allo.host_xfer` (see collision note above — NOT `allo.backend`).
host_xfer = RecordingBackend()


# ---------------------------------------------------------------------------
# Host program (host-xcel programming model) -- PROTOTYPE.
#
# A *host program* is a plain Python function that reads top-to-bottom as the
# host-side driver: it stages operands onto the device (`host_xfer.scatter` /
# `broadcast`), invokes the device kernels in order (`allo.launch`), and reads
# results back (`host_xfer.gather`). It is NEVER parsed to MLIR -- it is run
# once at import to *record* the orchestration, so it can hold the transfer
# calls that the MLIR-parsed region/`@allo.work` bodies cannot.
#
# Buffer operands are bound to `BufferToken`s (identity, carrying the param
# name) so moves reference the actual operands rather than string labels; the
# resolver (`spmw_codegen._resolve_host_moves`) already reads `.name` off a
# non-string buffer, so the recorded moves resolve byte-identically to the old
# string-label form. `.moves` threads into `compile_for_target(host_moves=...)`.
#
# Level 1 (this prototype): record only -- the driver still performs the
# transfer (Q2 option a). Level 2 (follow-up): `_run_samsung` consumes `.moves`
# + `.launches` to DRIVE the preload/readback, retiring shape inference.


class BufferToken:
    """Identity handle for a workload buffer inside a host program.

    Carries the operand's parameter name so `_resolve_host_moves` recovers the
    same `buffer_role` a bare string label used to. Deliberately NOT a
    `HandleToken`/`_VerbCallOrToken`, so the resolver classifies it as the
    buffer (data) side of a move, not the device endpoint.
    """

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __getitem__(self, idx):
        # A batched/sliced operand reference, e.g. `X[b]` in a batch loop ->
        # `BufferToken("X[b]")`. Keeps identity addressing while letting a host
        # program iterate over a batch dimension.
        return BufferToken(f"{self.name}[{idx}]")

    def __repr__(self):
        return f"BufferToken({self.name!r})"


class LaunchRecord:
    """One recorded `allo.launch(kernel, *operands)` -- a kernel invocation in
    the host program. `kernel` is the device kernel's name (validated against
    the region's declared kernels); `operands` are `BufferToken`s."""

    __slots__ = ("kernel", "operands")

    def __init__(self, kernel, operands):
        self.kernel = kernel
        self.operands = tuple(operands)

    def __repr__(self):
        return f"LaunchRecord({self.kernel!r}, {self.operands!r})"


# Active launch-recorder stack (parallel to `_host_move_recorder_stack`).
_launch_recorder_stack: list = []


def launch(kernel, *operands):
    """Record a kernel invocation inside a host program (the host-xcel launch).

    `kernel` is the device kernel name (a string matching an `@allo.work` in the
    region); `operands` are the `BufferToken`s it reads/writes. Ordered with the
    surrounding `host_xfer.*` moves so the host program reads as a real driver.
    """
    if not _launch_recorder_stack:
        raise RuntimeError("allo.launch(...) called outside a host_program scope")
    rec = LaunchRecord(kernel, operands)
    _launch_recorder_stack[-1].append(rec)
    return rec


class HostProgram:
    """The recorded host-xcel orchestration returned by `@allo.host_program`.

    A user's host-side driver: an ordered sequence (`.steps`) of `host_xfer.*`
    data moves (`.moves`) interleaved with `allo.launch(...)` kernel invocations
    (`.launches`). The residency analysis (`allo.spmw_host_program.analyze`)
    turns it into a host schedule that the run path executes and the cost model
    prices from one source. `.moves` also threads into `compile_for_target(
    host_moves=...)`; iterating / `len()` yields the moves, so `HOST_MOVES =
    host` works.
    """

    __slots__ = ("region", "moves", "launches", "steps")

    def __init__(self, region, fn):
        import inspect

        params = list(inspect.signature(fn).parameters)
        tokens = {p: BufferToken(p) for p in params}

        # One ordered list shared by both recorder stacks, so `host_xfer.*`
        # moves and `allo.launch(...)` calls land in `steps` in true source
        # order; `.moves`/`.launches` are derived by type.
        steps: list = []
        _host_move_recorder_stack.append(steps)
        _launch_recorder_stack.append(steps)
        try:
            fn(**tokens)
        finally:
            popped_l = _launch_recorder_stack.pop()
            popped_m = _host_move_recorder_stack.pop()
            assert popped_l is steps and popped_m is steps

        moves = [s for s in steps if isinstance(s, HostMoveRecord)]
        launches = [s for s in steps if isinstance(s, LaunchRecord)]

        # Validate launch targets against the region's declared kernels.
        known = set(getattr(region, "mappings", {}) or {})
        for lr in launches:
            if known and isinstance(lr.kernel, str) and lr.kernel not in known:
                raise ValueError(
                    f"host_program launches unknown kernel {lr.kernel!r}; "
                    f"region declares {sorted(known)}"
                )

        self.region = region
        self.moves = moves
        self.launches = launches
        self.steps = steps

    def __iter__(self):
        return iter(self.moves)

    def __len__(self):
        return len(self.moves)


def host_program(region):
    """Decorator: declare a host program (host-xcel driver) for `region`.

    Usage::

        @allo.host_program(_two_mm_top)
        def host(A, B, C, AB, D):
            allo.host_xfer.scatter(A, allo.host_xfer.banks)
            allo.host_xfer.broadcast(B, allo.host_xfer.grf_a)
            allo.launch("mm1", A, B, AB)
            ...
            allo.host_xfer.gather(D, allo.host_xfer.banks)

        HOST_MOVES = host.moves

    `host` is run once here to record; `host.moves` threads into
    `compile_for_target(host_moves=...)`.
    """

    def deco(fn):
        return HostProgram(region, fn)

    return deco


# Back-compat alias (the public name is `HostProgram`).
_HostProgram = HostProgram


def _resolve_token(token, target):
    """Resolve a `HandleToken`/`_VerbCallOrToken` to the identical tree handle.

    Looks the declared name up in `target._handles` (the flat name->handle map
    the `Target`/`DeviceScope` already build). Returns the *same object* the
    target tree holds, so the dispatcher can key on `is`/`id()`.
    """
    name = token.name
    handles = target.__dict__.get("_handles", {})
    if name not in handles:
        raise KeyError(
            f"host_xfer handle {name!r} is not a declared memory/register on "
            f"target {target.name!r}; declared: {sorted(handles)}"
        )
    return handles[name]


def _move_device_endpoint(move):
    """Whichever of `move.src`/`move.dst` is the device-side handle.

    On a host-scope move exactly one endpoint is host DRAM and one is device
    memory; this returns the device one (the non-host `Memory`/`Register`),
    compared structurally by owning-unit host-ness. Returns `(handle, position)`
    where position is "dst" if the device endpoint is the destination (into
    device) or "src" (out of device).
    """
    su, du = _handle_unit(move.src), _handle_unit(move.dst)
    src_host = su is not None and _is_host_side(su)
    dst_host = du is not None and _is_host_side(du)
    # device endpoint = the non-host side.
    if dst_host and not src_host:
        return move.src, "src"  # out-of-device (src is device, dst is host)
    if src_host and not dst_host:
        return move.dst, "dst"  # into-device (dst is device, src is host)
    return None, None


class BackendHandle:
    """Compile-time resolution of recorded host moves against a concrete target.

    Built from a `Target` at `compile_for_target`. `resolve(record)` returns the
    matching declared `Move` (D3), or raises a hard error. Token attribute
    access (`bh.banks`) resolves by declared name to the identical tree handle.
    """

    def __init__(self, target):
        self.__dict__["target"] = target
        # Pre-collect host-scope moves once.
        host_moves = []
        for u in target._walk():
            if getattr(u, "mode", None) == "host":
                host_moves.extend(u.moves.values())
        self.__dict__["_host_moves"] = host_moves

    def __getattr__(self, name):
        # Resolve a declared-handle name to the identical tree object.
        handles = self.target.__dict__.get("_handles", {})
        if name in handles:
            return handles[name]
        raise AttributeError(
            f"target {self.target.name!r} has no handle named {name!r}"
        )

    def resolve(self, record):
        """Resolve one `HostMoveRecord` to a declared `Move` (D3)."""
        target = self.target
        # 1. Verb narrowing (identity, never string ==).
        verb = record.verb
        verb_moves = [m for m in self._host_moves if m.verb is verb]
        if not verb_moves:
            raise ValueError(
                f"target {target.name!r} implements no host move for verb "
                f"{verb.name!r}"
            )
        # 2. Direction + device-handle identity. Exactly one arg is a token.
        tokens = [
            a for a in record.args if isinstance(a, (HandleToken, _VerbCallOrToken))
        ]
        buffers = [
            a for a in record.args if not isinstance(a, (HandleToken, _VerbCallOrToken))
        ]
        if len(tokens) != 1:
            raise ValueError(
                f"host_xfer.{verb.name}(...) must name exactly one device handle "
                f"(a host_xfer.<handle>); got args {record.args!r}"
            )
        device_handle = _resolve_token(tokens[0], target)
        # Device-handle identity is the discriminating key. Direction is already
        # encoded by the verb (scatter/broadcast -> into-device; gather ->
        # out-of-device), so verb narrowing has separated the two directions;
        # we match the remaining candidates on the device endpoint by `is`.
        candidates = []
        for m in verb_moves:
            dev, _pos = _move_device_endpoint(m)
            if dev is device_handle:
                candidates.append(m)
        # 3. Cardinality.
        if not candidates:
            raise ValueError(
                f"target {target.name!r} implements no host move for verb "
                f"{verb.name!r} touching device handle "
                f"{getattr(device_handle, 'name', device_handle)!r}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous host move for verb {verb.name!r} on target "
                f"{target.name!r}: colliding moves "
                f"{[m.name for m in candidates]!r}"
            )
        return candidates[0]
