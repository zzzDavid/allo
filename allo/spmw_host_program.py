# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-program residency analysis (the host-xcel scheduling pass).

A `HostProgram` (`allo/spmw_target.py`) records a user's host-side driver: an
ordered list of `host_xfer.*` data moves interleaved with `allo.launch(...)`
kernel invocations. This module turns that recorded program into a **host
schedule** -- the backend-agnostic plan the run path executes and an executable
cost program can price from the same analysis.

The load-bearing decision is *weight residency*: a weight `scatter` hoisted out
of a batch loop is preloaded ONCE and reused across the launches that follow it
(until it is re-scattered). Consecutive GEMV launches reusing a resident weight
are **coalesced** into one batched contraction -> the preload amortizes (the
`P + B*(E+R)` schedule). A re-scatter, a different weight, or a 2-D (GEMM)
operand breaks the group. The schedule is the single source that drives the
executor's coalescing and is available to cost-program lowering.

Pure and backend-agnostic: no simulator calls, no numpy. `analyze()` is unit
testable in isolation; the Samsung execution hook lives in `spmw_codegen.py` and
the reference composition stays test-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HostLaunch:
    """One kernel invocation: `out = weight @ vec` (a GEMV/GEMM contraction)."""

    weight: str
    vec: str
    out: str
    N: int  # contraction width (1 for a GEMV vec; trailing dim for GEMM)


@dataclass
class HostGroup:
    """A resident weight + the launches that reuse it without re-staging.

    `len(launches) > 1` means the weight is preloaded once and the launches are
    coalesced (batched GEMV); a singleton group is a plain single contraction.
    """

    weight: str
    weight_shape: tuple  # (M, K) of the bank-resident weight
    launches: list = field(default_factory=list)

    @property
    def is_batched(self) -> bool:
        return len(self.launches) > 1

    @property
    def exec_N(self) -> int:
        """The contraction width to run this group at: the batch size when
        coalesced (vecs stacked as columns), else the lone launch's own N."""
        return len(self.launches) if self.is_batched else self.launches[0].N


@dataclass
class HostSchedule:
    """The analyzed plan: ordered groups + the final gathered buffer + the set of
    external input buffers (consumed before any launch produces them)."""

    groups: list
    final: str | None
    external_inputs: list

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_launches(self) -> int:
        return sum(len(g.launches) for g in self.groups)


def resolve_shapes(host, namespace) -> dict:
    """Build a `{buffer_name: shape}` map from the region's parameter annotations.

    Handles sliced operands (`X[b]` -> "X[b]"): the base buffer's shape with the
    leading (batch) dim dropped, so a row of an `[B, K]` buffer is a `[K]` vec.
    `namespace` is the workload module's globals (to evaluate PEP 563 string
    annotations like `'fp16[B, K]'`).
    """
    region = host.region
    fn = getattr(region, "__wrapped__", region)
    ann = getattr(fn, "__annotations__", {})

    cache: dict = {}

    def shape_of(name):
        if name in cache:
            return cache[name]
        base, _, rest = name.partition("[")
        t = ann[base]
        if isinstance(t, str):  # PEP 563 string annotation -> evaluate
            t = eval(t, namespace)  # pylint: disable=eval-used
        shape = tuple(int(d) for d in t.shape)
        shape = shape[1:] if rest else shape
        cache[name] = shape
        return shape

    # Materialize every buffer named anywhere in the program.
    from .spmw_target import HostMoveRecord, LaunchRecord, BufferToken

    for step in host.steps:
        if isinstance(step, HostMoveRecord):
            for a in step.args:
                if isinstance(a, BufferToken):
                    shape_of(a.name)
        elif isinstance(step, LaunchRecord):
            for o in step.operands:
                shape_of(o.name)
    return cache


def analyze(host, shapes) -> HostSchedule:
    """Analyze a `HostProgram` into a `HostSchedule` (the residency plan).

    `shapes` is a `{name: shape}` map (see `resolve_shapes`). Pure: no backend
    calls. The grouping rule -- resident weight persists across launches until
    re-scattered; consecutive 1-D-vec launches reusing it coalesce -- is the
    single source the executor and cost model both consume.
    """
    from .spmw_target import HostMoveRecord, LaunchRecord, BufferToken

    def buf_name(rec):
        return next(a.name for a in rec.args if isinstance(a, BufferToken))

    # --- Pass 1: ordered launch events + weight-residency flag. ---
    events = []  # (weight, vec, out, weight_restaged)
    resident_weight = None
    pending_vec = None
    restaged: set = set()
    produced: set = set()
    consumed_order: list = []
    final = None
    for step in host.steps:
        if isinstance(step, HostMoveRecord):
            verb = step.verb.name
            name = buf_name(step)
            if verb in ("scatter", "broadcast"):
                consumed_order.append(name)
                if verb == "scatter":
                    resident_weight = name  # persists until re-scattered
                    restaged.add(name)
                else:
                    pending_vec = name
            elif verb == "gather":
                final = name
        elif isinstance(step, LaunchRecord):
            w, vc = resident_weight, pending_vec
            out = next(o.name for o in step.operands if o.name not in (w, vc))
            events.append((w, vc, out, w in restaged))
            produced.add(out)
            restaged = set()  # resident_weight persists
            pending_vec = None

    external_inputs = [n for n in dict.fromkeys(consumed_order) if n not in produced]

    # --- Pass 2: coalesce consecutive batchable launches reusing a resident
    # weight (1-D GEMV vec; a re-scatter / different weight / 2-D operand breaks
    # the group). ---
    def batchable(name):
        return len(shapes[name]) == 1

    groups: list = []
    for w, vc, out, fresh in events:
        if (
            groups
            and not fresh
            and groups[-1].weight == w
            and batchable(vc)
            and batchable(groups[-1].launches[0].vec)
        ):
            groups[-1].launches.append(HostLaunch(w, vc, out, 1))
        else:
            vshape = shapes[vc]
            N = vshape[1] if len(vshape) == 2 else 1
            M, K = shapes[w]
            groups.append(
                HostGroup(
                    weight=w, weight_shape=(M, K), launches=[HostLaunch(w, vc, out, N)]
                )
            )
    return HostSchedule(groups=groups, final=final, external_inputs=external_inputs)


def schedule_residency(schedule) -> tuple:
    """Map a host schedule onto the cost model's `(stage_resident, batch)` levers.

    A coalesced (batched) group means the weight is preloaded once and reused ->
    `stage_resident=True`, `batch` = that group's launch count. A schedule with
    no coalescing (each launch its own preload, or distinct weights like a GEMM
    chain) -> `(False, 1)`, i.e. the existing trace-driven cost path is unchanged
    (B=1 parity). This is the single seam where the host program's residency
    decision can feed executable cost-program lowering.
    """
    for g in schedule.groups:
        if g.is_batched:
            return True, len(g.launches)
    return False, 1


def schedule_cost(schedule, preload_of, exec_readback_of) -> int:
    """Price a host schedule from its structure -- the cost-side counterpart of
    the executor, reading residency from the SAME analysis.

    `preload_of(M, K) -> P` is the one-time weight-staging cost; `exec_readback_of
    (M, K, N) -> E+R` is the per-launch exec + readback. Each GROUP pays exactly
    ONE preload (the residency win: a coalesced group preloads once), and every
    launch pays exec+readback. So a single coalesced group of B launches costs
    `P + B*(E+R)`, while B singleton groups cost `B*(P+E+R)` -- the batched-GEMV
    crossover, derived from the program structure, never a hardcoded
    `weight_resident` flag. Backend-agnostic: the caller supplies the two phase
    callables.
    """
    total = 0
    for g in schedule.groups:
        M, K = g.weight_shape
        total += preload_of(M, K)  # ONE preload per group
        for lx in g.launches:
            total += exec_readback_of(M, K, lx.N)
    return total
