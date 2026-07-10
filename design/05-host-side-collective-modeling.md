# Design 05 — Host-side collective modeling: the `HostXcel` interface, residency, layout-derived distribution, and the `host_staging` cost concern

*Architect spec — Phase-0 gate for `dev/06262026-host-side-collective-modeling`.*
*Status: LANDED. Nothing in Phases 1–5 starts until this is signed off.*

This is the design source of truth the coder implements against for tasks
003–011. It answers the five Phase-0 gating questions, pins the API
signatures and dataclass/registration changes, and is the standing
FPGA/AIE blast-radius guard for this cycle. It is written against the
already-landed CostModel reformulation (`tenon@7738fee`) and the
researcher's non-pow2 finding (report 25). The design source-of-truth for
the `@allo.target`/`@allo.unit`/`fn`/`emit`/`ctx`/`@allo.cost` *shape* is
report 16; the survey/precedent and the IREE-HAL + SimplePIM hybrid is
report 23; this doc realizes report 23 §4 concretely.

---

## 0. Problem statement

Tenon models the *device* (unit tree, memories, registers, `move`/`op`,
costs) and the *device kernel* (`@allo.work`), but has **no host model**.
Host staging is smuggled into device-internal `allo.move` as physically
meaningless self-moves (`src=even_bank, dst=even_bank, emit=lambda ctx:
None`) at `tests/spmw/_fixtures.py:172-190`, and report-18's batched-GEMV
amortization (B\*=2 crossover, 3.93× asymptote) lives only as a backend
fact read through those self-moves and the `weight_resident` placement
flag. The host-staging cost is the *dominant* term (PIM-MMU: 63.7% of
end-to-end) and the one no surveyed PIM DSL prices — Tenon's novelty
opening (report 23 §3 conclusion 5).

The task: add an abstract `HostXcel` collective interface to Allo (basis
`broadcast`/`scatter`/`gather`/`reduce` + derived
`all_reduce`/`all_gather`/`reduce_scatter`); let each target implement the
subset it supports with cost+emit; let `@allo.work` *write* host-initiated
movement with a `residency` modifier; derive the collective and its cost
from the (src,dst) **linear-layout** pair (retire the sharding
annotation); and price it through a `host_staging` **CostModel concern**
on the already-landed `(target_name, flavor, concern)` registry. Prove the
abstraction is real by reproducing report-18's invariants *from the new
path*, with byte-identical kernel-body codegen.

---

## 1. The five Phase-0 answers

### Q1 — Linear layout is strict GF(2) (pow2-only). Keep it that way; non-pow2 lives in the host collective layer.

**Finding (research report 25, with file:line).** The in-tree
`LinearLayout` (`allo/spmw_linear_layout.py`) is **strict GF(2),
power-of-2-only, enforced at the constructor**. The single chokepoint
`_ilog2_exact` (`spmw_linear_layout.py:32-39`) raises `ValueError` on any
non-power-of-2; every size flows through it (`__init__` `:104-105`,
`identity` `:158`, `zero` `:183`, `optimal_swizzle` `:264`, `matrix`
`:315`, `invert` `:428,:464,:471`). The map is pure F2-linear (`apply`,
`:288-306`) — not even affine (no constant term). Empirically
`LinearLayout.identity({"dpu": 2560})` raises (report 25 §2.2).

**Ruling.** Do **not** generalize `LinearLayout` to affine/mixed-radix.
That would break the `invert`/`matrix`/`compose`/`optimal_swizzle` algebra
the Samsung swizzle relies on, and buys nothing: every *device* axis any
backend's layout touches (Samsung GRF/bank/tile, APU v1/v2 element, AiM
bank/bg) is pow2 hardware.

The non-pow2 case — UPMEM racks expose 2560 (or 2552 masked) DPUs — is
**not a device-layout axis**. It remains a **host-side data-partition
fan-out**, exactly the axis the host collective layer owns. The power-of-two
64-DPU rank and padded tasklet subgroups are now device LinearLayout outputs;
this does not route the rack fan through F2. See
`design/09-aim-upmem-linear-layout.md`. So:

* **Sharding annotation is retired for device placement** (Q4-of-task's D4
  holds): what the layout derives, `@[S(0)]` / `allo.grid_map` no longer
  states.
* **The non-pow2 DPU fan-out re-homes into the host collective.** A
  `scatter` / `broadcast` fan-out degree is a **plain `int`**, derived from
  `target.n_units` (or the unit-tree mapping product) and operand shape; it
  is **never** passed to `LinearLayout`. A 2560-way or 2552-way split
  constructs, type-checks, and prices with no F2 constraint.

**Invariant locked:** `spmw_linear_layout.py` is **unchanged** this cycle.
The host collective layer must never route a unit-count / fan-out degree
through `LinearLayout.identity`/`.zero`/`out_sizes`. The guard test is the
existing-and-staying-raising `LinearLayout.identity({"dpu": 2560})`
`ValueError`, plus a new positive test that a `STAGE_SCATTER` with
`fan == 2560` constructs and prices without touching the layout module
(see §8 test T-NONPOW2).

### Q2 — Axis binding: one mechanism — the collective's distribution axis IS the layout out-dim that maps onto the unit-tree axis.

A collective **is a relayout** (report 23 §4.2, task D4). `scatter(buf,
over=axis)` means "source layout = host-contiguous → destination layout =
partitioned over the unit-tree axis named `axis`." Because linear layouts
compose and invert, the communication *pattern* and its *cost* are derived
from the `(src, dst)` layout pair — `dst ∘ src⁻¹` — not separately
annotated.

The binding mechanism, concretely and singular:

* The unit-tree axes are exactly the named `mapping` levels of the
  `@allo.unit` chain (`spmw_target.py:get_uid` returns one `UnitId` per
  ancestor `@unit`, outer→inner). The product of `u.mapping` is the fan-out
  degree at that level (the same `prod(u.mapping)` the autoschedule already
  computes, `spmw_autoschedule.py:456`).
* The `over=` argument of a collective names **which unit-tree axis** the
  destination is partitioned/replicated over. For a pow2 device axis it is
  *also* the name of the F2 layout out-dim (so `dst` is
  `LinearLayout.identity` over that out-dim, and `dst ∘ src⁻¹` is the
  scatter map the `emit` consumes — report 23 open-Q1: reuse the layout
  map, do not invent a second one). For a **non-pow2 host fan-out** (UPMEM
  DPU count) the `over=` axis resolves to a **plain integer fan degree**
  (`prod(mapping)` of that unit level) and **no `LinearLayout` is built**.
* `over=` defaults to the **outermost unit-tree axis** when omitted
  (`get_uid()[0]` level), which is the partition axis for weight rows in
  every workload in the corpus. This keeps the common case annotation-free.

So there is **one** axis-binding rule: *`over=` names a unit-tree axis;
its fan degree is `prod(mapping)` of that level; for a pow2 device axis the
layout out-dim of the same name carries the conflict-free placement, for a
non-pow2 host fan-out it is just the integer degree.* No second mechanism,
no sharding annotation.

**Resolver.** A small spmw-local helper `_resolve_collective_axis(work,
target, over)` returns `(unit_level, fan_degree, layout_or_None)`. It
lives in the workload/host module (see §3), not in `spmw_target.py`, and is
the single place `over=` is interpreted. The `emit` and the cost model both
read its output; neither re-derives the axis.

### Q3 — Boundary: explicit collectives own host↔device staging (outer data placement); the implicit matcher keeps owning intra-kernel loads/stores. They never overlap.

Precise, implementable split:

* **Explicit collective layer** owns every transfer that **crosses the
  host↔device boundary** — i.e. `src` is host memory (`mode="host"` unit's
  `dram`) and `dst` is a device tier (bank / MRAM / L1), or vice versa.
  These are the `HostXcel` primitives. They are *written* in `@allo.work`
  (the `residency`-tagged collective calls) or *defaulted* by grid-fitting.
  This subsumes preload (broadcast/scatter weights), per-vector input
  (scatter/broadcast x), and readback (gather y).
* **Implicit matcher** owns every transfer **internal to the device** — the
  bank↔GRF `LD_A`/`LD_B`/`ST_A`/`ST_B` moves and the op MAC loads. These
  stay exactly as today: matched against `fn`, placed by regalloc, emitted
  by the device `ctx`. `allo.move` remains the device-internal primitive.

**The non-overlap rule (the one the coder enforces):** a `Move` whose
`src.owner` and `dst.owner` are on **opposite sides of the host boundary**
(one under a `mode="host"` unit, one under a device unit) is a **host
collective** and must be declared via the `HostXcel` surface, never as a
bare `allo.move`. A `Move` whose `src` and `dst` are both device tiers is a
device move and stays on `allo.move`.

**Self-move (`src is dst`) rule — refined (task-014 ruling, 2026-06-26).**
The *fake host self-moves in device costume* (`PRELOAD_*`/`READBACK_*`/
`CRF_TRIGGER`, `emit=lambda ctx: None`) are forbidden — they are the smell
this design deletes. But a same-side `src==dst` move is NOT inherently
illegal: **synthetic control moves are exempt.** The canonical case is
`JUMP` (`_fixtures.py:142`, `src=grf_b, dst=grf_b`), a real device control
op the cost model prices (`JUMP=1`) and codegen emits — it is `src==dst`
only because a JUMP has no data operand. So the validation rule is:

> A same-side `src==dst` move is rejected **iff** it is a host-staging
> stand-in — identified by `emit is None` (or an `emit` that produces no
> instruction) **and** a name in the retired `PRELOAD_*`/`READBACK_*`/
> `CRF_TRIGGER` family. A `src==dst` move with a real `emit` (JUMP, and any
> future control op) is **allowed**. Cross-boundary moves on a bare
> `allo.move` are rejected (route through `HostXcel`).

Concretely the coder gates on "`src is dst` **and** (`emit is None` or
`is_host_staging_stub(move)`)", not on `src is dst` alone. This exempts
JUMP and any control op while still rejecting the deleted host stubs.

This boundary is checkable from the target tree alone (host-side vs
device-side ownership), so the verifier's anti-hardcoding audit (task 016)
can mechanically prove the fake host self-moves are gone — without
flagging the legitimate `JUMP` control move.

### Q4 — `reduce` / `all_reduce`'s `op` reuses the existing `fn=lambda` operator-semantics path. (Yes.)

A reduction operator is declared with the **same `fn=lambda` shape**
`allo.op` already uses (report 16: `fn` is the semantics the matcher
unifies against). `reduce(buf, over=axis, op=...)` takes `op` as either
(a) a reference to a `@allo.target` `Op` handle whose `fn` is the reduction
(`target.op("ADD")`), or (b) a bare `fn=lambda acc, x: acc + x` that the
host-xcel coverage check resolves against the target's declared ops. This
keeps **one** operator-semantics path: there is no separate "reduction
operator" concept. A target that has an `ADD` op (or a host-CPU reduce)
covers `reduce`/`all_reduce`; one that does not raises the coverage error
(Q-coverage, §2). Samsung declares no device `reduce` → its output sum runs
on the host CPU after `gather` (opt-in, see §2). This also means the
derived `all_reduce` default composition (`reduce` then `broadcast`) is
expressed purely in basis primitives whose `op` rides the existing path.

### Q5 — Blast radius: spmw-local + additive only. No FPGA/AIE regression.

Every file this cycle touches and why it is safe:

| File | Change | Additive? | FPGA/AIE risk |
|---|---|---|---|
| `allo/spmw_host.py` (**NEW**) | `HostXcel` base, `@allo.host_xcel`, `@allo.primitive`, coverage check, default compositions, `_resolve_collective_axis` | new module | none |
| `allo/spmw_target.py` | `mode="host"` kwarg on `@allo.unit` (default `None`); cross-boundary `Move` validation | additive (default-None kwarg) | none — FPGA/AIE never call `@allo.unit` |
| `allo/__init__.py` | export `HostXcel`, `host_xcel`, `primitive` (and the residency enum) from `spmw_host` | additive imports | none |
| `allo/spmw_cost_tables.py` | register a NEW `host_staging`-concern `CostModel` per target; Samsung compose reads host collectives instead of `PRELOAD_*` self-moves | additive registry key | none |
| `allo/spmw_cost.py` | (if needed) a `@allo.cost("host_staging")` factory that looks up `get_cost_model(..., concern="host_staging")` | additive name | none |
| `allo/spmw_autoschedule.py` | replace `_samsung_host_eligible_memrefs` / `grf_residency` / `_with_weight_residency` derivation with host-collective + `residency`-derived equivalents | semantics change **inside spmw Samsung path only** | none — no shared IR/dataflow edit |
| `allo/spmw_codegen.py` | replace `host_preloads` / `HostTrigger` / `host_schedule` side-lists and the `weight_resident` branch (`_layout_weight_resident`, ~2284, ~2380-2435) with host-collective emission | semantics change **inside spmw run path only** | none |
| `tests/spmw/_fixtures.py` | rewrite self-moves (`:172-190`) as host collectives | test fixture | none |

**Zero edits** to `allo/ir/*`, `allo/dataflow.py`, `allo/customize.py`,
`allo/backend/*`, `allo/_mlir/*`, `allo/dsl.py`, `allo/memory.py`. The
FPGA/AIE path reaches `LLVMModule`/`HLSModule`/`IPModule` and
`df.gather`/`df.scatter` (see the **naming collision ruling**, §4) which
are all untouched. The gating upstream tests stay green:
`tests/test_vhls.py`, `tests/test_vitis.py`, `tests/test_xls.py`,
`tests/test_catapult_hls.py`, `tests/test_pynq.py`, `tests/test_nn.py`,
and the AIE collective tests `tests/dataflow/aie/test_collective_communication.py`.

---

## 2. The `HostXcel` interface (Phase 1, task 003)

New module `allo/spmw_host.py`. The base class defines the collective
vocabulary; partial implementation + coverage are first-class.

```python
class NotSupported(Exception):
    """A target's host-xcel does not implement / cannot cover a collective."""

class HostXcel:
    # --- basis: a target implements the subset it supports ---
    def broadcast(self, buf, *, over): raise NotSupported   # 1 -> all units
    def scatter  (self, buf, *, over): raise NotSupported   # partition -> units
    def gather   (self, buf, *, over): raise NotSupported   # units -> 1
    def reduce   (self, buf, *, over, op): raise NotSupported

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
```

**Registration.** A target spec declares its host-xcel inside the
`@allo.target` body:

```python
@allo.target("samsung_hbm_pim")
def device():
    @allo.unit(mode="host")
    def host():
        dram = allo.mem(name="host_dram", bytes=...)
        @allo.host_xcel
        class hx(allo.HostXcel):
            @allo.primitive
            def broadcast(self, buf, *, over):
                return lambda ctx, t: ctx.host_broadcast(t)   # -> HAB
            @allo.primitive
            def scatter(self, buf, *, over):
                return lambda ctx, t: ctx.host_scatter(t)     # -> per-bank
            @allo.primitive
            def gather(self, buf, *, over):
                return lambda ctx, t: ctx.host_gather(t)      # readback
            # no `reduce` -> Samsung output sum runs on host CPU after gather
    @allo.unit(mapping=[16])
    def pseudo_channel(): ...
```

**Signatures the coder implements:**

* `@allo.host_xcel` — class decorator. Instantiates the class, validates it
  subclasses `HostXcel`, records which basis primitives are
  `@allo.primitive`-decorated (the **covered set**), attaches the instance
  to the enclosing `mode="host"` unit (error if not inside one). Returns the
  instance handle. Mirrors the `@allo.unit`/`@allo.target` decorator shape
  (runs at decoration time, registers on the tree).
* `@allo.primitive(cost=None)` — method decorator. Marks a `HostXcel`
  method as a concrete primitive. The decorated method **returns an `emit`
  closure** `lambda ctx, t: ...` (same `emit` shape as `allo.move`, report
  16). `cost=` is an **optional** per-primitive cost hook; the canonical
  cost path is the `host_staging` CostModel (§5), so `cost=` is the escape
  hatch for a primitive whose cost is not table-driven. Default `None`.
  Records the method name in the covered set.
* **Coverage check** — `HostXcel.covers(collective_name) -> bool`: a
  collective is covered iff (a) it is a basis primitive that is
  `@allo.primitive`-decorated, OR (b) it is a derived collective whose
  default composition's basis calls are all covered. Requesting an
  uncovered collective is a **hard compile error** with a message naming the
  collective, the target, and the missing basis primitive(s). **Host-CPU
  fallback is opt-in only** — `reduce` falling to host CPU requires the
  workload (or target) to *say so* (`op=allo.host_cpu` sentinel, or a
  `@allo.primitive` `reduce` whose emit calls `ctx.host_reduce`); a silent
  fallback is forbidden (it would hide cost and break cross-target
  fairness). The coverage error fires at compile time, before codegen.

**Default-composition lowering.** A derived collective expands to its basis
calls **in the order written in the default method**, each basis call going
through that target's concrete primitive. The cost of a derived collective
is the **sum** of its basis costs (§5). An override (a `@allo.primitive
all_reduce`) replaces the composition with a single native fast-path emit +
cost.

**Phase-1 acceptance (task 004):** (i) requesting an uncovered collective
raises a clear hard error; (ii) a derived collective on a partial target
lowers to its basis calls (composition test). No target behavior changes
yet — Samsung migration is Phase 2.

---

## 3. Workload-side: writing host movement with `residency` (Phases 2–3)

The workload writes host-initiated movement by calling the host-xcel
collectives on its operand views, with a `residency` modifier:

```python
@allo.work(mapping=[16, 8], args=[W, x, y])
def gemv(local_W, local_x, local_y):
    allo.broadcast(local_x, over=PCH, residency="per_call")
    allo.scatter(local_W,  over=PCH, residency="resident")   # preloadGEMV, written
    ...                                                      # device compute body unchanged
    allo.gather(local_y,   over=PCH, residency="readback")
```

**`residency ∈ {resident, per_call, readback}`** — a **modifier on the
collective call**, not a separate concept (task D3). It is a `Literal`
enum/string validated at the call site.

* `residency="resident"` is the language-level `preloadGEMV`: stage once,
  reuse across calls. It is the construct that makes report-18 amortization
  *legal in the language*.
* `residency="per_call"`: paid every invocation (activations).
* `residency="readback"`: device→host gather after the kernel.

**Surface naming (collision ruling — see §4).** The workload-facing
collective calls are exposed as **module functions** `allo.broadcast` /
`allo.scatter` / `allo.gather` / `allo.reduce` (and derived) that, at
`@allo.work` build time, resolve to the *current target's* host-xcel
instance and record a **staging request** on the work's IR (a small
`StageRequest(collective, buf, over, residency, op=None)` record list on
the compiled work, parallel to how matches are recorded). They are **not**
the `df.gather`/`df.scatter` AIE pipe primitives (those stay under the `df`
namespace, untouched).

**Phase-3 residency hoist (task 006).** For batched GEMV (`Y[B,M] =
X[B,K] @ W^T`), a `residency="resident"` stage of `W` is hoisted out of the
batch loop **by the language**, not by a backend `if`. The formal precedent
is Exo's `@config` hoist via idempotency (report 23 §2.6, open-Q3): staging
a `resident` buffer is idempotent across calls **iff the kernel does not
write it**. The hoist is a legality transform:

> A `residency="resident"` stage is hoistable out of the batch loop iff its
> buffer is **not kernel-written** (read-only in the `@allo.work` body).

The coder adds this check (a read/write scan of the work body for the
staged buffer; `resident` + kernel-written = compile error, the type-check
rule of report 23 open-Q3). When hoistable, the staging request is marked
`hoisted=True`, and the cost model prices it **once** rather than `B` times
(§5). This makes report-18's `weight_resident ? preload + B*(exec+readback)
: B*(preload+exec+readback)` branch a **derived fact** — and the existing
`_layout_weight_resident` / `_with_weight_residency` / `weight_resident`
branch is **deleted**.

---

## 4. Naming-collision ruling: `df.gather`/`df.scatter` vs host collectives

**Finding.** `allo/dataflow.py:32-49` already defines `gather(pipes)` and
`scatter(buffer, pipes)` — the AIE collective-communication primitives,
used as `df.gather`/`df.scatter` in
`tests/dataflow/aie/test_collective_communication.py` (10+ call sites).
They are **not** exported as top-level `allo.gather`/`allo.scatter`
(verified: `__init__.py` and `dsl.py` export neither).

**Ruling.** The host collectives are exposed as top-level
`allo.broadcast` / `allo.scatter` / `allo.gather` / `allo.reduce` /
`allo.all_reduce` / `allo.all_gather` / `allo.reduce_scatter`, imported
from `spmw_host`. This does **not** collide with `df.gather`/`df.scatter`
because:

1. The AIE primitives are accessed through the `df` module namespace
   (`from allo import dataflow as df; df.gather(...)`), never as bare
   `allo.gather`. The collision would only exist if we added
   `from .dataflow import gather, scatter` to `__init__.py` — which we do
   **not**, and which is not there today.
2. Semantically they are different layers: `df.gather` is a **device-side
   AIE pipe** collective (intra-device, inside a `@df.kernel` body);
   `allo.gather` is a **host↔device** staging collective. The boundary rule
   (§Q3) keeps them disjoint.

**Coder guard:** do **not** add `gather`/`scatter` to `__init__.py` from
`dataflow`; import them only from `spmw_host`. The AIE tests must stay green
(they use `df.*`, untouched). This is the regression line for the
collision.

---

## 5. Cost: `host_staging` as a NEW concern on the landed CostModel registry (Phase 4, task 007)

**The cost machinery already exists** (`tenon@7738fee`,
`spmw_cost_model.py`): `CostModel` is bound by `(target_name, flavor,
concern)` via `register_cost_model`, looked up by `get_cost_model(...,
concern=...)`. `host_staging` attaches as a **NEW concern** — **not a fresh
cost extraction.** The coder reuses the landed `CostModel` / `OpCost` /
`MoveCost` / `compose` / registry verbatim.

**What lands (in `spmw_cost_tables.py`):**

```python
SAMSUNG_HOST_STAGING = register_cost_model(CostModel(
    name="samsung_host_staging",
    target_name="samsung_hbm_pim",
    flavor="faithful",
    concern="host_staging",            # <-- NEW concern, same registry
    op_costs={},
    move_costs={                        # the SAME calibrated constants, re-homed
        "STAGE_BCAST":  MoveCost(lambda c: 369, note="HAB preload fan-out width"),
        "STAGE_SCATTER":MoveCost(lambda c: 1,   note="per-group column-strobe"),
        "STAGE_CRF":    MoveCost(lambda c: 2,   note="programCrf upload"),
        "GATHER_FAN":   MoveCost(lambda c: 4096,note="readback tile width"),
        "GATHER_RD":    MoveCost(lambda c: 181, note="per-tile readResult"),
    },
    constants={},
    compose=_samsung_host_staging_compose,
))
```

* The constants `369/1/2/4096/181` are the report-18 calibration anchors,
  moved verbatim off the `PRELOAD_*`/`READBACK_*` self-moves. **No new
  numbers** — this is a re-homing, the proof being byte-identical argmin
  (§8).
* `_samsung_host_staging_compose(ctx)` prices the work's `StageRequest`
  list, derived from the layout transform (§Q2): each request costs
  `(buf.numel // fan_width) * per_unit_cyc`, with `resident`/`hoisted`
  paid **once** and `per_call` paid `B` times. **Derived collectives sum
  their basis costs.**
* The existing **`kernel_cycles`** Samsung CostModel keeps the *device*
  body (MAC fold, JUMP, trigger) — but **stops reading
  `PRELOAD_*`/`READBACK_*`/`grf_residency`**. The whole-program estimate is
  `kernel_cycles + host_staging`, composed at the autoschedule/run seam (the
  same place `evaluate` already composes). The B=1 sum must equal **15251**
  (parity), crossover at **B\*=2**, asymptote **3.93×** — *emerging* from
  `host_staging(resident=once) + B*(exec+readback)`, not asserted to a
  literal (task 009 anti-tautology).

**Async-overlap accommodation (out of scope to *implement*, in scope to
*not preclude*).** The whole-program composition is `device + host` as a
**sum** today. Async staging/compute overlap turns it into a `max`. The
coder must design the composition seam so the host-staging `CostResult`
returns a **per-phase breakdown** (`phases={"stage_resident": ...,
"stage_per_call": ..., "readback": ...}`, the `CostResult.phases` field
already exists) and the whole-program combiner is a small pluggable
function (default = sum). Do **not** implement the overlap scheduler. This
keeps report 23 open-Q4 a one-function swap later.

---

## 6. Target migration: the Samsung host node (Phase 2)

**Split across two tasks by the task-005 ruling (§9.1).**

**Re-scoped 005 = task 012 (parity-neutral, additive).** Add a Samsung
`@allo.unit(mode="host")` node declaring a `@allo.host_xcel` class with
`broadcast`/`scatter`/`gather` primitives whose `emit` produces the real
`pim_driver` preload/readback host-driver calls. This is a **parallel
additive structure**: the emit closures exist, but the Samsung *run path*
still drives the old `PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER` self-moves and
the `kernel_cycles` compose is **untouched**. 012 deletes nothing and moves
no number — argmin byte-identical.

**Deletion happens in task 007 (§5, §9.1), not 012.** Task 007 re-homes
369/1/2/4096/181 into the `host_staging` CostModel, switches the Samsung
`kernel_cycles` compose to stop reading `PRELOAD_*`/`READBACK_*`, wires
012's host-node emit into the run path, deletes the self-moves +
`weight_resident` branch + host side-lists, **turns on the cross-boundary
`Move` validation (§Q3)**, and migrates the 4 constant-asserting tests — all
in one change, so parity is re-proved from the new path rather than broken
mid-flight. The `CRF_TRIGGER` "host fire latency" becomes a property of the
host node's dispatch (a `trigger` cost on the host-xcel) at that point.

The device tree (pseudo_channel → pim, banks, GRF, LD/ST/MUL/MAC) hangs
**under** the host node and is **byte-for-byte unchanged** throughout — this
is the regression guard: kernel-body codegen identical before/after (task
009).

---

## 7. Generality proof: second target (Phase 5, task 008)

Bring up the collective interface on **UPMEM** (preferred): a
`mode="host"` node with `scatter`/`gather` primitives whose emit lowers to
`dpu_prepare_xfer`+`dpu_push_xfer` (scatter) / `dpu_copy_from` (gather) and
`broadcast` → `dpu_broadcast_to`. The **non-pow2 DPU fan-out** (Q1) is the
generality stress: `scatter(W, over=DPU)` with `fan == 2560` must construct
and price with **no `LinearLayout`** (the integer fan degree from
`prod(mapping)`), proving the host layer carries what the device layout
cannot.

**Partial-implementation demonstration:** a target that implements only
part of the basis (e.g. UPMEM declares `scatter`/`gather`/`broadcast` but
no device `reduce`) must either (a) cover `all_gather` via the default
composition, or (b) raise the coverage error for `all_reduce` (no `reduce`
basis), proving the coverage check is real. AiM (`WR_SBK`/`WR_ABK`) is the
fallback second target if UPMEM scatter/gather hits a runtime wall.

---

## 8. Tests that prove it works

| Tag | Test | Asserts |
|---|---|---|
| T-COVER (004) | uncovered-collective error | requesting `all_reduce` on a target with no `reduce` basis raises a hard error naming the missing basis |
| T-COMPOSE (004) | default-composition lowering | `all_gather` on a basis-only target lowers to `gather`+`broadcast` calls in order |
| T-NONPOW2 (008) | non-pow2 host fan-out | `STAGE_SCATTER` with `fan==2560` (M=5120,K=1024, 2 rows/DPU) constructs + prices with no `ValueError` from `spmw_linear_layout`; and `LinearLayout.identity({"dpu":2560})` still raises (the negative guard) |
| T-PARITY (009) | report-18 from new path | host-staging+device sum at B=1 **== 15251** (emerges, not literal); crossover **B\*=2**; asymptote **3.93×** |
| T-BYTEID (009) | device codegen unchanged | kernel-body `cmds` byte-identical before/after migration |
| T-DELETED (010) | anti-hardcoding | grep proves `PRELOAD_*`/`READBACK_*` self-moves, `weight_resident` branch, `host_preloads`/`host_schedule`/`HostTrigger` side-lists, `_samsung_host_eligible_memrefs`/`grf_residency`/`_with_weight_residency` are **gone**; 15251/B\*=2/3.93× not produced by a shape literal or backend `if` |
| T-FPGA (011) | no blast radius | `tests/test_vhls.py`, `test_vitis.py`, `test_xls.py`, `test_catapult_hls.py`, `test_pynq.py`, `test_nn.py`, `tests/dataflow/aie/test_collective_communication.py` stay green |
| T-SUITE (011) | full suite | all `tests/spmw/` green in `pim-dev` |

---

## 9. Open design tensions (carried, not blocking)

* **T18 — async staging/compute overlap (sum→max).** Out of scope to
  implement; the per-phase `CostResult.phases` + pluggable whole-program
  combiner (§5) is the seam. Revisit when a workload with measurable
  stage/compute overlap (pipelined batched GEMV) ships. Report 23 open-Q4.
* **T19 — staging cost units (device cycles vs host wall-time).** Report 18
  folded host staging into device-cycle-equivalents (the 369/181 constants);
  that keeps argmin single-currency but understates real wall-time
  (PIM-MMU). The `host_staging` CostModel keeps the cycle-equivalent
  convention this cycle; a real host-bandwidth term is a later
  table-only refinement (the whole point of design 04's table split).
  Report 23 open-Q2.
* **T20 — intra-DPU non-pow2 row count.** UPMEM tensor layouts now pad
  non-power-of-two logical spans to an F2 domain and keep tail validity in the
  ABI. Padding never gathers back. This permits genuine `(dpu,tasklet,local)`
  layout axes without pretending that 24 tasklets or an arbitrary row count is
  itself a power of two. See design 09.
* **T21 — reduce-operator resolution path.** Q4 reuses the `fn=lambda`
  path; the open edge is a `reduce` whose `op` is a *composite* (e.g.
  max-plus) not declared as a single target `Op`. Today's corpus only needs
  `+`; revisit when a non-additive collective reduction (argmax gather)
  ships.

---

## Open question for architect (raised by coder, task 005)

**Task 005's one-liner is in tension with the doc's own phase sequencing,
and cannot be executed standalone without breaking the parity it requires.**

The 005 task file says: delete the `PRELOAD_FAN`/`PRELOAD_WR`/`PRELOAD_CRF`/
`READBACK_FAN`/`READBACK_RD`/`CRF_TRIGGER` self-moves from
`tests/spmw/_fixtures.py:156-190`, re-express GEMV via the host collectives
+ `residency`, wire D4 layout-derived distribution — and (per the
orchestrator relay) "the report-18 invariants (B=1==15251, B\*=2, 3.93x)
must stay byte-identical." But those self-moves are the *carriers the
`kernel_cycles` parity path reads by name*, and the doc explicitly homes the
replacement machinery in **later tasks**:

* The 15251 number is produced **today** by `_samsung_compose_with`
  (`spmw_cost_tables.py:494-586`), whose B=1 result is
  `B*(preload_cyc + exec_cyc + readback_cyc)` where `preload_cyc` =
  `(M*K // PRELOAD_FAN)*PRELOAD_WR + PRELOAD_CRF` = `(4096*1024//369)*1 + 2`
  = **11368** (`_samsung_preload_cycles`, `:152-158`, reads PRELOAD_FAN=369/
  PRELOAD_WR=1/PRELOAD_CRF=2) and `readback_cyc` = `ceil(M/4096)*181` =
  **181** (`_samsung_readback_cycles`, `:161-166`, reads READBACK_FAN=4096/
  READBACK_RD=181). So `exec_cyc` = 15251 − 11368 − 181 = 3702. **Every one
  of these constants is read off a move whose name is the self-move 005 is
  told to delete.**
* The `weight_resident ? preload + B*(exec+readback) : B*(preload+exec+readback)`
  branch (`spmw_cost_tables.py:572-576`) — i.e. the B\*=2 crossover and
  3.93x asymptote — is the branch **§3 says is "deleted" by task 006**
  (residency hoist), and the `host_staging`-concern CostModel that re-homes
  369/1/2/4096/181 is **§5's "Phase 4, task 007."** T-PARITY/T-BYTEID are
  **task 009**; the T-DELETED grep proving the self-moves are gone is **task
  010**. The doc nowhere authorizes task 005 to land the cost re-homing.
* Blast radius: **9 `tests/spmw/` files** reference the to-be-deleted
  symbols by name (`test_samsung_batched_gemv.py`,
  `test_samsung_host_residency.py`, `test_cost_model_swap.py`,
  `test_samsung_shared_crf.py`, `test_move_scheduling.py`,
  `test_codegen_gemv.py`, `test_autoschedule.py`,
  `test_rankpreserve_vs_sim.py`, `test_samsung_loop_012.py`).
  `test_samsung_batched_gemv.py` asserts the *exact* `nr == B*(P+E+R)` /
  `rr == P + B*(E+R)` algebra and perturbs `PRELOAD_FAN`/`PRELOAD_CRF`/
  `PRELOAD_WR` by name; `test_cost_model_swap.py:93-94` asserts
  `move_costs["PRELOAD_FAN"].fn(...) == 369`. `StageRequest` does not exist
  yet (it is the §3/task-006 surface).

If task 005 deletes the self-moves, it MUST simultaneously land the
task-007 `host_staging` CostModel re-homing AND the task-006 `StageRequest`/
residency/`weight_resident`-deletion AND migrate those 9 test files — i.e.
collapse 005+006+007 (+009/010 test updates) into one change. That is a
much larger, design-decision-bearing scope than 005's one-liner, and the
non-overlap `Move`-validation rule (§Q3) can only be turned on once *all*
self-moves are gone.

### RULING (architect, 2026-06-26): Option 1 — re-scope 005 to parity-neutral additive only.

The coder's analysis is correct and matches this doc's own stated phase
boundaries. Verified independently: `_samsung_preload_cycles`
(`spmw_cost_tables.py:152-158`) and `_samsung_readback_cycles` (`:161-166`)
read the 369/1/2/4096/181 constants **by move name** through the
`kernel_cycles` CostModel, and the `weight_resident` branch (`:572-576`) is
the B\*=2/3.93× carrier. Four test files assert these by name
(`test_samsung_batched_gemv.py`, `test_cost_model_swap.py:93-94`,
`test_samsung_shared_crf.py`, plus `_fixtures.py` itself). Deleting the
self-moves in 005 would collapse 005+006+007 and force the test migration
the doc assigns to 009/010 — a scope and parity decision the 005 one-liner
does not authorize.

**Option 1 is adopted.** This supersedes the original task-005 wording. The
self-move deletion, the `kernel_cycles` decoupling from `PRELOAD_*`, the
cross-boundary `Move` validation, and the test migration are **NOT** in
re-scoped 005. They are pinned to the phases below.

Replacement tasks (this ruling re-issues them; see §9.1):
* **012-ready-coder-phase2-samsung-host-node** — re-scoped 005.
* **006/009/010 unchanged in ownership; clarified** by §9.1 so the
  hand-off seam is explicit. 007 gains the explicit "delete last self-move
  + turn on Move-validation + migrate the 4 constant-asserting tests" duty.

---

## 9.1 Phase sequencing for the Samsung migration (binding ownership map)

This section is the authoritative ownership map after the task-005 ruling.
Each artifact is deleted/added by **exactly one** task; parity stays intact
until the task that re-homes the carriers and flips the tests in the same
change.

**Final task ids** (after the task-005 and task-014 rulings): 012 (host
node), 013 (StageRequest + hoist), **017** (cost re-home + `weight_resident`
deletion — supersedes 014), 018/019 (verifiers — supersede 015/016).
Originals 005/006/007/009/010 and 014/015/016 are in `done/` (superseded).

| Phase / task | Adds | Deletes | Parity state | Tests it owns |
|---|---|---|---|---|
| **012 (re-scoped 005)** coder | Samsung `@allo.unit(mode="host")` node + `@allo.host_xcel` class with `broadcast`/`scatter`/`gather` primitives, **as a parallel additive structure**. Emit closures present but **not yet wired** into the run path. | nothing | **unchanged** — `PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER` self-moves and `_samsung_compose_with` untouched; argmin byte-identical | construction-only test: host node + host-xcel build, `covers("broadcast"/"scatter"/"gather")` True; **no parity number touched** |
| **013 (re-scoped 006)** coder | `StageRequest` workload surface; residency-hoist legality transform (`resident` + kernel-written = error). The hoist stamps a structural resident flag on the placement (today's `weight_resident=True` effect). | nothing | **unchanged** — same placement effect the branch keys on; numbers do not move | residency-hoist legality + a test that a `resident` stage marks the placement, reproducing today's `weight_resident=True` effect |
| **017 (supersedes 014)** coder | `host_staging`-concern CostModel re-homing 369/1/2/4096/181 (§5); `kernel_cycles` Samsung compose stops reading `PRELOAD_*`/`READBACK_*`; `host_staging` reads the structural resident flag from the placement (bridge option (b)); wire 012's host-node emit into the run path; **turn on the refined `Move` validation (§Q3, JUMP-exempt)** | **`weight_resident` lever ONLY**: the `PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER` self-moves; `_layout_weight_resident` + the `weight_resident ? : ` branch; `_with_weight_residency`; `host_preloads`/`host_schedule`/`HostTrigger`. **DO NOT delete `grf_residency` / `_samsung_host_eligible_memrefs` / `_with_residency` (lever 2 — stays).** | **re-proved from the new path** — whole-program = `kernel_cycles + host_staging`; B=1 sum EMERGES as 15251; static argmin floor 12037 (lever 2 intact) unchanged | **migrate ONLY the 3 `weight_resident` tests + `_fixtures.py`**: `test_cost_model_swap.py:93-94` → `host_staging` move_costs; `test_samsung_batched_gemv.py` algebra → `kernel_cycles + host_staging`; `test_residency_hoist.py` / `test_rankpreserve_vs_sim.py` weight_resident asserts; `_fixtures.py`. **`grf_residency`/host-residency tests stay green untouched.** |
| **018 (supersedes 015)** verifier | — | — | **proves** B=1==15251 (emerges, not literal), B\*=2, 3.93×, static floor 12037; kernel-body codegen byte-identical pre/post-017 | T-PARITY, T-BYTEID |
| **019 (supersedes 016)** verifier | — | — | — | T-DELETED grep: `weight_resident` family + host self-moves + host side-lists gone after 017; **`grf_residency` STILL PRESENT** (lever 2 retained); JUMP `src==dst` exemption holds (validation does not reject JUMP) |

**Lever disposition (task-014 ruling).** Two independent residency levers
must not be conflated:
* **`weight_resident`** (SPEC-026, preload-once batching) — host staging;
  re-homed into `host_staging` by 017; deleted in 017.
* **`grf_residency`** (SPEC-024 lever 2, host-broadcast-vs-CRF-MOV LD_A
  drop) — **device-exec**; subtracts `move_cost(LD_A)` from `body_cyc`
  (`spmw_cost_tables.py:518,538`); **NO host_staging carrier**; load-bearing
  for the B=1 argmin winner (12037). **Stays intact.** Retiring it is a
  separate future spec with its own 5+-file test-migration budget — out of
  scope for host-side collective modeling.

**Why the `Move` validation lands in 017, not 012.** The host self-moves
still exist through 013; turning the guard on before 017 deletes them would
fail the suite. 017 deletes the host stubs and enables the guard in the
**same** change. The guard is **JUMP-exempt** (§Q3 refined rule): it rejects
only host-staging stubs (`emit is None` / `PRELOAD_*`/`READBACK_*`/
`CRF_TRIGGER`), never a `src==dst` move with a real `emit` like JUMP.

**The 012 emit-not-wired rule.** 012's host-xcel primitives carry their
`emit` closures (so Phase-1 coverage/composition is exercisable) but the
Samsung *run path* still drives the old `PRELOAD_*` self-moves until 017
re-homes them. This keeps 012 strictly parity-neutral.

---

## Open question for architect (raised by coder, task 014)

**Task 014's "Deletes" column conflates two independent residency levers,
and deleting one of them (`grf_residency`, lever 2) demonstrably MOVES the
argmin — contradicting 014's own "15251 must emerge / byte-identical"
guarantee. 014 cannot be executed as written.**

The §9.1 row-007/014 "Deletes" column lists, together:
`_samsung_host_eligible_memrefs` / `grf_residency` / `_with_residency` AND
`_layout_weight_resident` / `weight_resident` / `_with_weight_residency`.
These are **two different mechanisms** with different homes:

* **`weight_resident`** (SPEC-026) is the **preload-once batching** split:
  `weight_resident ? preload + B*(exec+readback) : B*(preload+exec+readback)`
  (`spmw_cost_tables.py:572-576`). This is exactly what §5 re-homes into
  `host_staging` (preload paid once vs B times). **This part of 014 is
  well-formed** — verified the decomposition is parity-exact:
  kernel_cycles(=B*exec) + host_staging(=preload@once-or-B + readback@B)
  reproduces both branches identically.
* **`grf_residency`** (SPEC-024, "lever 2") is a **device-exec** cost
  adjustment, NOT host staging: a host-broadcast-vs-CRF-MOV choice for the
  broadcastable GRF operand that **subtracts the LD_A cost from `body_cyc`**
  when host-resident (`spmw_cost_tables.py:518,531-538`). It has **no
  host_staging carrier to re-home into** (§5 re-homes only 369/1/2/4096/181,
  the preload/readback constants — nothing for the LD_A drop).

**Proof that deleting lever 2 moves the number.** The B=1 single-GEMV
`kernel_cycles` argmin winner is:

```
ARGMIN: (12037, 'dual_fiber+crf_shared', grf_residency={'local_W':'host'}, weight_resident=False)
```

The winner **uses `grf_residency={'local_W':'host'}`** — its 12037 depends
on the lever-2 LD_A subtraction. Delete `grf_residency`/`_with_residency`/
`_samsung_host_eligible_memrefs` and that subtraction vanishes, `body_cyc`
(hence `exec`, hence the whole-program sum) increases, and the argmin moves.
(Note: no static candidate equals 15251 at B=1; 15251 is the *simulator*
faithful B=1 figure in `test_batched_faithful_run`, pim_driver-gated. The
static-cost argmin floor is 12037. Both must be preserved.)
`test_samsung_argmin_picks_host_residency` (`test_samsung_host_residency.py`)
explicitly asserts the winner is host-resident — so lever 2 is load-bearing
for the corpus argmin, not dead code.

**Test-scope conflict.** 014 authorizes migrating **4** tests
(`test_cost_model_swap.py`, `test_samsung_batched_gemv.py`,
`test_samsung_shared_crf.py`, `_fixtures.py`). But deleting the lever-2
`grf_residency` machinery breaks **12** test files that reference it by name
(`test_autoschedule.py`, `test_codegen_gemv.py`, `test_move_scheduling.py`,
`test_rankpreserve_vs_sim.py`, `test_samsung_host_residency.py` [6 tests,
entirely lever-2], `test_autoschedule_samsung_dual_fiber.py`,
`test_samsung_loop_012.py`, `test_regalloc.py`, `test_residency_hoist.py`,
`test_samsung_host_node.py`, + the 2 named). §8 assigns the lever-2 /
host-residency T-tests to no 014/009/010 owner.

**Enumerator↔staging bridge gap.** Nothing outside `spmw_host.py`
references `StageRequest`/`host_staging`/`resolve_staging`; the corpus
workloads (`batched_gemv_top`) do not call `allo.scatter(...)`. So the
`host_staging` compose receives `ComposeCtx(target, trace, layout)` with no
StageRequest list. For argmin to keep choosing resident-vs-non-resident
byte-identically **without rewriting the workloads with collective calls**
(which would also break the §8 T-BYTEID kernel-body guard), 014 must either
(a) synthesize the StageRequest/hoist flag structurally from the trace
(M,K,B + read-only-weight), or (b) keep a structural resident flag on the
placement that `host_staging` reads. The doc says "the hoist becomes the
driver" but does not pin which, for the no-rewrite case.

### Decision needed (one of)

1. **Scope 014's deletions to the `weight_resident` lever ONLY** (the part
   §5 actually re-homes): re-home 369/1/2/4096/181 into `host_staging`,
   split kernel_cycles = device-exec and whole-program = kernel_cycles +
   host_staging (parity-exact, verified), delete `weight_resident` /
   `_with_weight_residency` / `_layout_weight_resident` / the
   `weight_resident?:` branch, wire host-node emit, turn on the
   cross-boundary `Move` validation, migrate the 3 `weight_resident` tests +
   `_fixtures.py`. **Leave `grf_residency` (lever 2) intact** — it is a
   device-exec lever, not host staging; its retirement (if desired) is a
   separate spec with its own test-migration budget. Pin the
   enumerator↔staging bridge to option (a) or (b) above. (Coder's
   recommendation — keeps 014 atomic AND parity-exact, matches §5's actual
   re-homing list, and respects the 4-test budget once `grf_residency` is
   excluded.)
2. **Authorize a much larger 014** that also retires lever 2: provide the
   replacement for the LD_A-drop cost (a new host_staging or kernel_cycles
   carrier that keeps 12037 byte-identical), expand the test-migration
   budget to the 12 lever-2 files, and re-assign the §8 host-residency
   T-tests. (Larger scope + a real cost-model design decision about where
   the lever-2 adjustment lives post-deletion.)

Also confirm the JUMP-move edge for the §Q3 `src==dst` validation: JUMP is
`src=grf_b, dst=grf_b` (`_fixtures.py:142`) — a same-handle device move that
the cost model prices (`JUMP`=1). The `src==dst` self-move rejection must
exempt synthetic control ops like JUMP, or JUMP must be re-expressed; the
doc's "self-moves forbidden" rule does not address JUMP.

Coder will not delete lever 2 or silently move the argmin. No source
edited (014 is atomic — a partial edit would break the suite). Awaiting the
architect's ruling.

### RULING (architect, 2026-06-26): Option 1 — delete only the `weight_resident` lever; `grf_residency` (lever 2) stays as a device-exec lever.

The coder is correct and I confirmed every claim from source:

* **`grf_residency` is a device-exec adjustment, NOT host staging.** It
  subtracts `model.move_cost(load_name)` (the LD_A/LD_B cost) from
  `body_cyc` (`spmw_cost_tables.py:518, 538`); `body_cyc` feeds `exec_cyc`,
  the **device kernel body**. §5's `host_staging` re-homing list is exactly
  369/1/2/4096/181 (preload/readback) — there is **no carrier** for the
  LD_A drop. Deleting `grf_residency` moves `exec`, hence the argmin
  (verified winner `12037, grf_residency={'local_W':'host'}`). It is
  load-bearing, asserted by `test_samsung_argmin_picks_host_residency`.
* **`weight_resident` IS what §5 re-homes** — the preload-once batching
  split (`:572-576`), parity-exact under the coder's decomposition. This
  part of 014 is well-formed.

**Ruling.** §9.1's row-007/014 "Deletes" column **erroneously conflated two
levers**. Corrected (§9.1 below): 014 deletes only the `weight_resident`
machinery (`_layout_weight_resident` / `weight_resident?:` branch /
`_with_weight_residency` + the 3 `weight_resident` tests). **`grf_residency`
/ `_samsung_host_eligible_memrefs` / `_with_residency` (SPEC-024 lever 2)
stay intact.** Lever 2 is a *device-exec* cost adjustment, not host
staging; retiring it is out of scope for "host-side collective modeling"
and would be a separate spec with its own 5+-file test-migration budget.
This keeps 014 atomic AND parity-exact AND within the authorized test
budget.

**Bridge ruling (the enumerator↔staging gap).** The corpus workloads
(`batched_gemv_top`) do not call `allo.scatter(...)`, so `host_staging`
compose gets no `StageRequest` list. To keep argmin byte-identical
**without rewriting workloads with collective calls** (which would break
the §8 T-BYTEID kernel-body guard), pin **option (b): a structural resident
flag on the placement** that `host_staging` reads. Concretely: the residency
hoist (task 013) already stamps the structural resident effect on the
placement (it reproduces today's `weight_resident=True`); 014's
`host_staging` compose reads **that same structural flag** (renamed off
`weight_resident` to a host-staging-owned key, e.g.
`layout.extra["stage_resident"]`) to decide preload-once vs preload-B.
StageRequest-from-explicit-`allo.scatter` is the *additional* path the
generality target (008) exercises; the Samsung no-rewrite corpus rides the
structural flag. The workload kernel body is untouched → T-BYTEID holds.

**JUMP / `src==dst` ruling.** Confirmed `JUMP` is `src=grf_b, dst=grf_b`
(`_fixtures.py:142`), a real priced control move. §Q3's self-move rule is
**refined** (see §Q3 above): reject `src==dst` only when it is a host-staging
stub (`emit is None` / named `PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER`); a
`src==dst` move with a real `emit` (JUMP, future control ops) is allowed.

Replacement task: **017-ready-coder-phase4-host-staging-cost** (supersedes
014). Verifier tasks 015/016 unchanged in ownership but re-pointed: gating
`...,014` → `...,017`; 016's T-DELETED grep targets the `weight_resident`
family only (NOT `grf_residency`) and asserts the JUMP `src==dst` exemption.
Re-issued as **018/019** with corrected scope. Old 014/015/016 → done.

---

## 10. Receipt for the orchestrator

Strongest concrete commitments this doc locks:

* `spmw_linear_layout.py` is **FORBIDDEN to change** this cycle (stays
  strict GF(2)); non-pow2 fan-out lives as a plain `int` in the host
  collective layer, never routed through `LinearLayout`.
* `HostXcel` basis = `broadcast`/`scatter`/`gather`/`reduce`; `@allo.host_xcel`
  (class decorator), `@allo.primitive(cost=None)` (method decorator
  returning an `emit` closure), `residency ∈ {resident,per_call,readback}`.
* `host_staging` attaches as a **NEW concern** on the landed
  `(target_name, flavor, concern)` CostModel registry — re-homing the
  369/1/2/4096/181 constants, **not** a fresh extraction; B=1 sum **== 15251**
  must emerge.
* `allo.gather`/`allo.scatter` (host) do **not** collide with
  `df.gather`/`df.scatter` (AIE); `__init__.py` must not import the latter.
* Zero edits to `allo/ir/*`, `dataflow.py`, `customize.py`, `backend/*`,
  `_mlir/*`; FPGA/AIE gating tests enumerated in §1/§8.

---

## Implemented

* **Task 003 (Phase 1, coder)** — landed `allo/spmw_host.py`: `HostXcel`
  base (basis `broadcast`/`scatter`/`gather`/`reduce`; derived
  `all_reduce`/`all_gather`/`reduce_scatter` default compositions),
  `@allo.host_xcel` class decorator (validates `HostXcel` subclass,
  records the `@allo.primitive`-covered set, attaches to enclosing
  `mode="host"` unit), `@allo.primitive(cost=None)` method decorator
  (returns `emit` closure), `HostXcel.covers`/`require`/`lower` coverage +
  default-composition lowering, `NotSupported`, `host_cpu` opt-in
  sentinel. Additive `mode=` kwarg + `Unit.host_xcel` field on
  `spmw_target.py` `@allo.unit`/`Unit`; `mapping` now optional. Exports
  `HostXcel`/`host_xcel`/`primitive`/`NotSupported`/`host_cpu` from
  `__init__.py` (did NOT import gather/scatter from `dataflow` — collision
  guard held). No target behavior change; cross-boundary `Move` validation
  and workload-side `StageRequest`/`residency` deferred to Phases 2-3 (the
  self-moves they replace still live in `_fixtures.py`). Test:
  `tests/spmw/test_host_xcel_core.py` 11 passed. No regression:
  test_target_cycles/aim/upmem + match_gemv + autoschedule 34 passed.

* **Task 012 (re-scoped 005, Phase 2, coder)** — added the Samsung host
  node as a PARALLEL ADDITIVE structure (§9.1 row 012). In
  `tests/spmw/_fixtures.py:build_samsung_target`: a
  `@allo.unit(mode="host")` node `host` (with a `host_dram` staging mem)
  declaring a `@allo.host_xcel` class with `broadcast`/`scatter`/`gather`
  `@allo.primitive`s whose emit closures are
  `lambda ctx, t: ctx.host_{broadcast,scatter,gather}(t)` (design §2 shape,
  → pim_driver preload/readback). Attached as a **SIBLING** of
  `pseudo_channel` (not its parent) so `get_uid()` chains,
  `_samsung_workid_count` (host `mapping=[]` → ×1), and the whole device
  subtree are byte-for-byte unchanged. **Deleted nothing**
  (`PRELOAD_*`/`READBACK_*`/`CRF_TRIGGER` self-moves + `_samsung_compose_with`
  untouched); **Move-validation OFF** (lands in 007). Emit present but NOT
  wired into the run path. Test: `tests/spmw/test_samsung_host_node.py`
  (8) — host node + host-xcel build, `covers(broadcast/scatter/gather)`,
  P=11368/R=181 anchors still resolve, B=1 single-GEMV cost vector
  byte-identical vs device-only tree (min 12037 / max 535837). Verified:
  `test_samsung_host_node + test_cost_model_swap + test_samsung_host_residency
  + test_host_xcel_core` 28 passed; direct cross-check
  `with host == without host: (min,max,workids)=(12037,535837,128)`.

* **Task 013 (re-scoped 006, Phase 3, coder)** — added the workload
  `StageRequest` surface + residency-hoist legality (§9.1 row 006),
  ADDITIVE, deletes nothing. In `allo/spmw_host.py`:
  `StageRequest(collective,buf,over,residency,op,hoisted)`,
  `residency ∈ {resident,per_call,readback}` (validated), `staging_scope()`
  context manager, module functions
  `broadcast`/`scatter`/`gather`/`reduce`/`all_reduce`/`all_gather`/
  `reduce_scatter` recording requests, `resolve_staging(reqs,
  written_buffers)` (the Exo-`@config` hoist: `resident` + read-only →
  `hoisted=True`; `resident` + kernel-written → hard `NotSupported`),
  `weight_resident_from_staging` (bridge reproducing today's
  `weight_resident=True`). "kernel-written" derived from the trace's
  `result_memref_name` (no MLIR rescan). Exported the collectives +
  StageRequest API from `__init__.py` (still no `dataflow` gather/scatter
  import; no `dsl`/`template` collision). The structural `weight_resident`
  enumerator branch + `_with_weight_residency` are UNTOUCHED (task 014
  deletes them). Also added the missing `return inst` to `host_xcel`.
  Test: `tests/spmw/test_residency_hoist.py` + `test_host_xcel_core.py`
  **21 passed in 184s**; the included
  `test_existing_weight_resident_branch_still_present` runs the real
  batched-GEMV enumerator and confirms both `weight_resident` variants +
  `+wresident` tokens intact (parity guard). No cost/enumerator/codegen
  file touched → report-18 invariants cannot move.

* **Task 017 (supersedes 014, Phase 4, coder)** — ATOMIC host_staging
  re-home + weight_resident-lever deletion (§9.1 row 017, §5/§6, §Q3
  refined, task-014 Option-1 ruling). Landed: NEW `host_staging`-concern
  CostModel (`SAMSUNG_HOST_STAGING` + optimistic) re-homing 369/1/2/4096/181
  VERBATIM as STAGE_BCAST/STAGE_SCATTER/STAGE_CRF/GATHER_FAN/GATHER_RD;
  `_samsung_host_staging_compose` prices preload (resident→once /
  per_call→×B) + readback (×B) reading `layout.extra["stage_resident"]`
  (bridge (b)), with per-phase `CostResult.phases` for the async-overlap
  seam. kernel_cycles split to DEVICE-exec only (`B*exec`); whole-program =
  kernel_cycles + host_staging via pluggable combiner (default sum) at
  `_kernel_cycles_factory` + `evaluate`. `_with_weight_residency` →
  `_with_stage_resident`; `_layout_weight_resident` → `_layout_stage_resident`.
  §Q3 Move validation turned on (JUMP-exempt: rejects src==dst only for
  host-staging stubs / emit-None; cross-boundary bare move rejected).
  Deleted the PRELOAD_*/READBACK_*/CRF_TRIGGER self-moves from `_fixtures.py`
  + the `weight_resident?:` branch. **grf_residency / lever 2 RETAINED
  (untouched)** — `host_preloads`/`host_schedule`/`HostTrigger` are lever-2/3
  machinery, kept (the row-017 deletion cell listing them is residual
  lever-conflation; flagged in report). Migrated the 3 weight_resident
  tests + `_fixtures.py`. Verified (serial, pim-dev): 38 + 44 + 37 =
  **119 passed, 0 failed**. Static parity: B=1 argmin **12037**
  (grf_residency='host' retained); P=11368/R=181 re-homed; B=1 nr==rr;
  B=4 resident<nonresident. **15251 EMERGES** = preload(11368) +
  readback(181) + faithful device-exec(3702). Zero edits to ir/dataflow/
  customize/backend/_mlir.

* **Task 008 (Phase 5, coder)** — generality proof: HostXcel on a SECOND
  target (UPMEM) + the non-pow2 host fan-out (§7, §1/§Q2, §8 T-NONPOW2).
  Added `resolve_collective_axis(target, over)` to `spmw_host.py` (§Q2: the
  single `over=` interpreter; returns `(unit, prod(mapping) as PLAIN INT,
  None)` — never routes the fan through `LinearLayout`). Added a UPMEM
  `@allo.unit(mode="host")` node (SIBLING of `rank`) with a `@allo.host_xcel`
  declaring `scatter`/`gather`/`broadcast` (emit → dpu_prepare_xfer+
  dpu_push_xfer / dpu_copy_from / dpu_broadcast_to) and **NO `reduce`**
  (PARTIAL basis). Added `UPMEM_HOST_STAGING` CostModel (concern
  host_staging), OPT-IN (0 unless `layout.extra["host_stage"]`), pricing the
  integer DPU fan. Coverage proof: `all_gather` covered (gather+broadcast);
  `all_reduce`/`reduce_scatter` raise the hard error naming the missing
  `reduce` basis + 'upmem'. T-NONPOW2: `resolve_collective_axis(dpu=2560)` →
  fan=2560/layout=None; host_staging prices fan=2560 (= 2,560,000) with a
  monkeypatch-spy proving `LinearLayout.identity` is never called; the
  negative guard `LinearLayout.identity({'dpu':2560})` still raises (GF(2)
  unchanged); 2552 masked also handled. `spmw_linear_layout.py` UNCHANGED
  (§1 invariant). Migrated 1 device-tree-shape assertion in
  `test_target_upmem.py` (filter to device-side children for the additive
  host sibling). Verified (serial, pim-dev): new+UPMEM-core 19 passed;
  consolidated (incl. regalloc) 66 passed; pipeline test_run+test_e2e_mlp
  13 passed; existing UPMEM cost invariants (test_upmem_cost_t1_parity s*r,
  lever 5.762, empty-trace==0) byte-identical (host_staging=0 for non-stage
  traces). FLAGGED a PRE-EXISTING latent flakiness (not task 008):
  `spmw_regalloc.py:255` `_CAPACITY_CACHE` keyed by `id(target)` collides
  after GC → intermittent stale-capacity spill miscount; recommend keying by
  `target.name`. Zero edits to ir/dataflow/customize/backend/_mlir/
  spmw_linear_layout.
