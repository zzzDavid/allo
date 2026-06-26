# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side collective interface for SPMW targets (design 05).

A `HostXcel` models host<->device staging as an abstract collective
vocabulary. The basis is `broadcast`/`scatter`/`gather`/`reduce`; the
derived collectives `all_reduce`/`all_gather`/`reduce_scatter` have default
compositions in terms of the basis (overridable). A target implements the
subset it supports; requesting an uncovered collective is a hard compile
error (no silent host-CPU fallback). See design 05 §2.

Phase 1 (task 003) lands the interface, the `@allo.host_xcel` /
`@allo.primitive` decorators, the covered-set bookkeeping, the coverage
check, and the default-composition lowering. Target migration, residency,
and the host_staging cost concern arrive in later phases.

Phase 3 (task 013) adds the workload-side `StageRequest` surface (the
module functions `broadcast`/`scatter`/`gather`/`reduce` + derived, recorded
on the work) and the Exo-`@config`-style residency-hoist legality transform
(`residency="resident"` is hoistable iff its buffer is NOT kernel-written;
`resident` + kernel-written = compile error). The hoist marks a request
`hoisted=True`, reproducing the SAME `weight_resident=True` placement effect
the existing enumerator/cost branch keys on. Task 013 DELETES NOTHING -- the
existing `weight_resident` branch / `_with_weight_residency` carriers stay;
their atomic deletion + cost re-homing is task 014.

Naming-collision note (design 05 §4): the host collectives are exposed as
top-level `allo.broadcast`/`scatter`/`gather`/`reduce`/... They are NOT the
`df.gather`/`df.scatter` AIE pipe primitives, which stay under the `df`
namespace. `__init__.py` must never import gather/scatter from `dataflow`.
"""

from . import spmw_target


# The basis primitives, in declaration order. A target covers each one by
# decorating it with @allo.primitive.
_BASIS = ("broadcast", "scatter", "gather", "reduce")

# Derived collectives -> the ordered basis calls their default composition
# expands to (design 05 §2). An @allo.primitive override of a derived name
# replaces this composition with a single native emit (handled in `covers`
# and `lower`).
_DERIVED_COMPOSITION = {
    "all_reduce": ("reduce", "broadcast"),
    "all_gather": ("gather", "broadcast"),
    "reduce_scatter": ("reduce", "scatter"),
}


class NotSupported(Exception):
    """A target's host-xcel does not implement / cannot cover a collective."""


# Opt-in sentinel: a `reduce` whose `op` is `allo.host_cpu` runs the
# reduction on the host CPU (design 05 §2). Silent fallback is forbidden;
# the workload must name this sentinel.
class _HostCpu:
    def __repr__(self):
        return "allo.host_cpu"


host_cpu = _HostCpu()


class HostXcel:
    """Abstract host-side collective interface.

    A target subclasses this and decorates the basis methods it supports
    with `@allo.primitive`. The base methods raise `NotSupported`; the
    derived methods compose the basis. `covers` answers whether a named
    collective is reachable.
    """

    # --- basis: a target implements the subset it supports ---
    def broadcast(self, buf, *, over):  # 1 -> all units
        raise NotSupported("broadcast not implemented")

    def scatter(self, buf, *, over):  # partition -> units
        raise NotSupported("scatter not implemented")

    def gather(self, buf, *, over):  # units -> 1
        raise NotSupported("gather not implemented")

    def reduce(self, buf, *, over, op):
        raise NotSupported("reduce not implemented")

    # --- derived: default compositions in terms of the basis, overridable ---
    def all_reduce(self, buf, *, over, op):
        self.reduce(buf, over=over, op=op)
        self.broadcast(buf, over=over)

    def all_gather(self, buf, *, over):
        self.gather(buf, over=over)
        self.broadcast(buf, over=over)

    def reduce_scatter(self, buf, *, over, op):
        self.reduce(buf, over=over, op=op)
        self.scatter(buf, over=over)

    # --- coverage ---
    @property
    def covered(self):
        """The set of basis primitive names this instance implements.

        Populated by @allo.primitive at decoration time (and by
        @allo.host_xcel as a fallback scan).
        """
        return set(getattr(self, "_covered", ()))

    def covers(self, collective_name) -> bool:
        """Whether `collective_name` is reachable on this host-xcel.

        A collective is covered iff (a) it is a basis primitive that is
        @allo.primitive-decorated, OR (b) it is an @allo.primitive-decorated
        override of a derived name, OR (c) it is a derived collective whose
        default-composition basis calls are all covered (design 05 §2).
        """
        covered = self.covered
        if collective_name in covered:
            return True
        if collective_name in _DERIVED_COMPOSITION:
            return all(b in covered for b in _DERIVED_COMPOSITION[collective_name])
        return False

    def require(self, collective_name, *, target_name=None):
        """Hard compile error if `collective_name` is not covered.

        The message names the collective, the target, and the missing basis
        primitive(s). Fires before codegen (design 05 §2).
        """
        if self.covers(collective_name):
            return
        tgt = target_name or getattr(self, "_target_name", "<unknown>")
        if collective_name in _DERIVED_COMPOSITION:
            missing = [
                b
                for b in _DERIVED_COMPOSITION[collective_name]
                if b not in self.covered
            ]
            raise NotSupported(
                f"collective {collective_name!r} is not covered on target "
                f"{tgt!r}: its default composition needs basis primitive(s) "
                f"{missing!r}, which this host-xcel does not implement"
            )
        raise NotSupported(
            f"collective {collective_name!r} is not covered on target {tgt!r}: "
            f"it is not an @allo.primitive on this host-xcel and is not a "
            f"known derived collective"
        )

    def lower(self, collective_name, buf, *, over, op=None):
        """Expand a collective into its concrete primitive emit closure(s).

        A covered basis primitive (or a derived override) returns a single
        emit closure. A derived collective with no override expands to its
        ordered basis calls, each going through that target's concrete
        primitive (design 05 §2 default-composition lowering). Returns a list
        of `(basis_name, emit)` pairs so the cost layer can sum basis costs.
        """
        self.require(collective_name)
        if collective_name in self.covered:
            method = getattr(self, collective_name)
            emit = _call_primitive(method, buf, over=over, op=op)
            return [(collective_name, emit)]
        # derived, no override: expand in written order
        out = []
        for basis in _DERIVED_COMPOSITION[collective_name]:
            method = getattr(self, basis)
            emit = _call_primitive(method, buf, over=over, op=op)
            out.append((basis, emit))
        return out


def _call_primitive(method, buf, *, over, op):
    """Invoke a host-xcel primitive method, passing `op` only to `reduce`."""
    if getattr(method, "__name__", None) == "reduce" or _is_reduce_like(method):
        return method(buf, over=over, op=op)
    return method(buf, over=over)


def _is_reduce_like(method):
    # A reduce override registered under a derived name still takes `op`; we
    # detect by the primitive's recorded arity flag set by @allo.primitive.
    return bool(getattr(method, "_takes_op", False))


# ---------------- decorators ---------------- #


def primitive(fn=None, *, cost=None):
    """Method decorator: marks a `HostXcel` method as a concrete primitive.

    The decorated method returns an `emit` closure `lambda ctx, t: ...`
    (same emit shape as `allo.move`; report 16). `cost=` is an optional
    per-primitive cost hook (escape hatch for a primitive whose cost is not
    table-driven; canonical path is the host_staging CostModel). The method
    name is recorded in the covered set by `@allo.host_xcel`.

    Usable bare (`@allo.primitive`) or parameterised (`@allo.primitive(cost=...)`).
    """

    def decorate(method):
        method._is_primitive = True
        method._primitive_cost = cost
        # reduce / reduce_scatter / all_reduce take an `op=`; flag so
        # `_call_primitive` forwards it.
        method._takes_op = method.__name__ in ("reduce", "reduce_scatter", "all_reduce")
        return method

    if fn is not None:
        return decorate(fn)
    return decorate


def host_xcel(cls):
    """Class decorator: register a host-xcel on the enclosing host unit.

    Validates `cls` subclasses `HostXcel`, instantiates it, records the
    covered set (the basis methods marked `@allo.primitive`), attaches the
    instance to the enclosing `mode="host"` unit, and returns the instance
    handle. Mirrors the `@allo.unit`/`@allo.target` decorator shape: runs at
    decoration time and registers on the tree (design 05 §2).
    """
    if not issubclass(cls, HostXcel):
        raise TypeError(
            f"@allo.host_xcel class {cls.__name__!r} must subclass allo.HostXcel"
        )
    if not spmw_target._target_stack:
        raise RuntimeError("@allo.host_xcel must be used inside @allo.target")
    cur = spmw_target._target_stack[-1]
    if getattr(cur, "mode", None) != "host":
        raise RuntimeError(
            "@allo.host_xcel must be declared inside an @allo.unit(mode=\"host\") "
            f"node; enclosing unit {cur.name!r} has mode={getattr(cur, 'mode', None)!r}"
        )

    inst = cls()
    covered = set()
    for name in dir(cls):
        member = getattr(cls, name, None)
        if callable(member) and getattr(member, "_is_primitive", False):
            covered.add(name)
    inst._covered = covered
    # walk up to the target root for the name (used in coverage error text)
    root = cur
    while root.parent is not None:
        root = root.parent
    inst._target_name = root.name

    if cur.host_xcel is not None:
        raise ValueError(
            f"unit {cur.name!r} already has a host-xcel; only one per host node"
        )
    cur.host_xcel = inst
    return inst


# ===================================================================== #
# Phase 3 (task 013): workload-side StageRequest surface + hoist legality
# ===================================================================== #

import contextlib  # noqa: E402
from dataclasses import dataclass  # noqa: E402


# residency modifier on a collective call (design 05 §3). A Literal-style
# enum validated at the call site.
RESIDENCY = ("resident", "per_call", "readback")


@dataclass
class StageRequest:
    """A host<->device staging request recorded on a work (design 05 §3).

    `collective` is the host-collective name (`broadcast`/`scatter`/...);
    `buf` is the operand view the workload staged (a name, or any object
    with a `.name`); `over` names the unit-tree axis the destination is
    partitioned/replicated over; `residency` is the call-site modifier.
    `op` rides only for reduce-class collectives.

    `hoisted` is set by `resolve_staging`: a `residency="resident"` request
    whose buffer is read-only in the work body is hoistable out of the batch
    loop (paid once), reproducing today's `weight_resident=True` effect.
    """

    collective: str
    buf: object
    over: object = None
    residency: str = "per_call"
    op: object = None
    hoisted: bool = False


# Recording context. `allo.broadcast(...)` etc. append a StageRequest to the
# innermost open `staging_scope()`. Kept spmw-local (no MLIR / dataflow.py
# edit): the workload opens a scope around its collective calls; the
# resulting list is the work's staging-request record (design 05 §3,
# "parallel to how matches are recorded").
_staging_stack: list = []


@contextlib.contextmanager
def staging_scope():
    """Open a recording scope; yields the `list[StageRequest]` populated by
    the host-collective module functions called inside it."""
    requests: list[StageRequest] = []
    _staging_stack.append(requests)
    try:
        yield requests
    finally:
        popped = _staging_stack.pop()
        assert popped is requests


def _buf_name(buf) -> str:
    return getattr(buf, "name", buf) if buf is not None else None


def _record(collective, buf, *, over=None, residency="per_call", op=None):
    if residency not in RESIDENCY:
        raise ValueError(
            f"residency must be one of {RESIDENCY!r}, got {residency!r} "
            f"(collective {collective!r}, buf {_buf_name(buf)!r})"
        )
    req = StageRequest(
        collective=collective, buf=buf, over=over, residency=residency, op=op
    )
    if _staging_stack:
        _staging_stack[-1].append(req)
    return req


# Workload-facing module functions (design 05 §3/§4). Exposed as
# `allo.broadcast` / `allo.scatter` / `allo.gather` / `allo.reduce` (+
# derived). These do NOT collide with `df.gather`/`df.scatter` (§4): those
# stay under the `df` namespace and are never imported into `__init__.py`.


def broadcast(buf, *, over=None, residency="per_call"):
    return _record("broadcast", buf, over=over, residency=residency)


def scatter(buf, *, over=None, residency="per_call"):
    return _record("scatter", buf, over=over, residency=residency)


def gather(buf, *, over=None, residency="readback"):
    return _record("gather", buf, over=over, residency=residency)


def reduce(buf, *, over=None, residency="per_call", op=None):
    return _record("reduce", buf, over=over, residency=residency, op=op)


def all_reduce(buf, *, over=None, residency="per_call", op=None):
    return _record("all_reduce", buf, over=over, residency=residency, op=op)


def all_gather(buf, *, over=None, residency="per_call"):
    return _record("all_gather", buf, over=over, residency=residency)


def reduce_scatter(buf, *, over=None, residency="per_call", op=None):
    return _record("reduce_scatter", buf, over=over, residency=residency, op=op)


def resolve_staging(requests, *, written_buffers):
    """Apply the residency-hoist legality transform (design 05 §3).

    The legality rule (Exo `@config` idempotency, report 23 open-Q3):

        A `residency="resident"` stage is hoistable out of the batch loop
        iff its buffer is NOT kernel-written (read-only in the work body).

    `written_buffers` is the set of buffer names the work writes -- derived
    spmw-side from the match trace's `result_memref_name`s (no MLIR rescan).
    A `resident` request whose buffer IS written is a hard compile error. A
    `resident` request whose buffer is read-only is marked `hoisted=True`
    (paid once -- reproducing today's `weight_resident=True` effect). Returns
    the same list (mutated in place) for chaining.
    """
    written = {_buf_name(b) for b in written_buffers}
    for req in requests:
        if req.residency != "resident":
            continue
        if _buf_name(req.buf) in written:
            raise NotSupported(
                f"residency='resident' stage of buffer "
                f"{_buf_name(req.buf)!r} is illegal: the buffer is "
                f"kernel-written, so staging it once is not idempotent "
                f"across calls (design 05 §3 hoist legality). Use "
                f"residency='per_call' for a kernel-written buffer."
            )
        req.hoisted = True
    return requests


def weight_resident_from_staging(requests) -> bool:
    """Bridge (task 013): does the resolved staging imply `weight_resident`?

    True iff any request is a `hoisted=True` resident stage -- i.e. a
    read-only buffer staged once and reused across the batch. This is the
    SAME materialisation effect today's enumerator/cost keys on via
    `extra['weight_resident']` (`spmw_autoschedule._with_weight_residency`).
    Task 013 reproduces the effect through the language-level hoist; task 014
    makes the enumerator/cost read this instead of the structural flag and
    deletes the old branch.
    """
    return any(r.residency == "resident" and r.hoisted for r in requests)


# ===================================================================== #
# Phase 5 (task 008): collective-axis resolver (design 05 §Q2)
# ===================================================================== #


class NonPow2FanError(Exception):
    """Raised if a fan degree is (incorrectly) routed through LinearLayout.

    Not normally raised in the host layer — a fan degree is a plain int.
    Exists so a regression that re-couples the host fan-out to the F2 layout
    surfaces loudly (design 05 §1 invariant).
    """


def _unit_by_name(target, name):
    for u in target._walk():
        if u.name == name:
            return u
    return None


def _outermost_unit(target):
    """The outermost non-root @unit level (get_uid()[0] level): the first
    child of the synthetic root that has a mapping."""
    root = target.root
    for u in root.children:
        return u
    return None


def resolve_collective_axis(target, over=None):
    """Resolve a collective's `over=` axis to `(unit, fan_degree, layout)`
    (design 05 §Q2). The SINGLE place `over=` is interpreted; `emit` and the
    cost model both read its output and never re-derive the axis.

    * `over` is a unit name (str) or a `Unit`; `None` defaults to the
      outermost unit-tree axis (`get_uid()[0]` level), the weight-row
      partition axis for the corpus.
    * `fan_degree` is the PLAIN INT `prod(mapping)` of that unit level. For a
      non-pow2 host fan-out (UPMEM DPU count = 2560 / 2552) it is just the
      integer degree — **no `LinearLayout` is built**.
    * `layout` is a `LinearLayout` over the same out-dim ONLY when the fan is
      a power of two AND a device-layout axis is wanted; for the host fan-out
      path it is **always `None`**. This function NEVER routes the fan degree
      through `LinearLayout` (design 05 §1 invariant: a unit-count / fan-out
      degree must never reach `LinearLayout.identity`/`.zero`/`out_sizes`).
    """
    from math import prod

    if over is None:
        unit = _outermost_unit(target)
    elif isinstance(over, str):
        unit = _unit_by_name(target, over)
    else:
        unit = over  # a Unit handle
    if unit is None:
        raise ValueError(
            f"resolve_collective_axis: no unit-tree axis named {over!r} on "
            f"target {getattr(target, 'name', '?')!r}"
        )
    fan_degree = prod(unit.mapping) if unit.mapping else 1
    # The host layer carries the fan as a plain int — NEVER a LinearLayout.
    # (A pow2 device-axis layout, if ever needed, is the matcher/regalloc's
    # job, not the host collective's; here we always return None so a
    # non-pow2 degree like 2560 type-checks and prices unconditionally.)
    layout = None
    return unit, fan_degree, layout
