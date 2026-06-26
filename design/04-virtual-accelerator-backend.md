# Design 04 — Virtual accelerator backend: decoupled `CostModel`, virtual runner, trip-count resolution

Status: Phase-0 architect spec (gating). Authoritative for tasks 002–008
in `dev/06262026-virtual-accelerator-backend/`.
Author: architect, 2026-06-26.

This doc answers the five Phase-0 questions the task gates on:
(1) the `CostModel` abstraction, (2) the virtual-backend runner contract,
(3) the D3 trip-count strategy, (4) shared-reformulation owner/merge-order
with the host-side task, (5) the blast radius (zero FPGA/AIE regression).

It is a **spec the coder implements verbatim**, not an implementation.

---

## 0. Problem statement

A user with no PIM simulator and no PIM hardware should still be able to
develop and *evaluate* a kernel for any of the five targets, purely from
`(target spec) + (cost model)`. Two things stand between us and that:

- **Coupling.** The per-op / per-move cycle constants (`cycles=4`,
  `cycles=26`, the Samsung PRELOAD/READBACK fan-outs, UPMEM
  `revolver_latency`, …) live **on the target tree** — they are
  `Move.cycles` / `Op.cycles` / `Unit.constants` attributes set in
  `tests/spmw/_fixtures.py`. So the device description and the cost
  numbers are one artifact. You cannot ship a structural target and swap
  in a different (optimistic / pessimistic / device-calibrated) cost
  table without editing the device tree.

- **Exposure.** The analytical cost path runs every autoschedule step
  (`spmw_cost_models._kernel_cycles_factory`), but it is never reachable
  through `Compiled.run()`. The only run path is `_BACKEND_RUN`, which
  boots a simulator / Docker / board. The sim-free estimate sits right
  beside it, unexposed.

The engine is **not** the deliverable — it exists and is correct. The
deliverable is the **decoupling** (D1), the **exposure** (D2), and the
**trip-count gap-closing** (D3) that today forces some bounds to come from
the simulator. D4 (sequential, no compute↔DMA overlap) is the documented
v1 scope boundary.

### 0.1 The load-bearing design principle (coordinator hint, 2026-06-26)

> The cost spec is an **interface for specifying the cost of each
> operation** (data movement, computation, …) — *not a monolithic
> estimator*. The reference example is the **per-operation analytical
> model for each APU v1 operation from the MICRO 2025 paper**: that kind
> of per-op analytical model is what must plug in. The reason the cost
> spec must stay decoupled is so it can be **separately refined to be more
> accurate against real-device profiling**, without touching the rest of
> the compiler. **Decoupling and independent refinability are first-class
> goals.**

This sharpens D1 beyond Accelergy's arch-vs-ERT split. Accelergy's ERT is
a *table of scalars*. Ours is a **table of per-op analytical functions**:
each op/move entry is an independently-authored, independently-refinable
cost function, and the *composition* (how the per-op costs fold into a
whole-program number) is a separate, also-swappable layer. The two
layers (§1.1) are why a future researcher can replace the APU v1 MAC cost
with a regression fit against board profiling and touch **nothing else**.

---

## 1. Answer 1 — the `CostModel` abstraction

### 1.1 Two layers, both swappable, both per-op-refinable

A `CostModel` has exactly two layers. Keep them separate; this separation
*is* the independent-refinability property.

**Layer A — the op/move cost table (`OpCost` / `MoveCost` entries).**
Per *named* op and move on the target, a cost entry. An entry is **not a
scalar** — it is a callable that returns a cycle count given the op's
local context (operand shapes / fold width / lane count where the per-op
model needs them). The trivial entry is "constant N cycles"; the
APU-v1-MICRO-2025 entry is "an analytical function of the operand width
and the bit-serial lane geometry." Both satisfy the same interface, so a
researcher upgrades one op from constant to analytical without touching
any caller.

```python
# spmw_cost_model.py  (NEW module — see §1.4 for the name decision)

@dataclass(frozen=True)
class OpCost:
    """Per-op cost entry. `fn` returns cycles for ONE issued op given
    its local context. The context is a small struct (§1.3), never the
    whole trace — that keeps each entry independently authorable and
    testable. A constant-cost op is `OpCost(lambda ctx: 4)`."""
    fn: Callable[["OpCostCtx"], int | float]
    # free-form provenance string (citation), surfaced in breakdowns.
    note: str = ""

@dataclass(frozen=True)
class MoveCost:
    fn: Callable[["MoveCostCtx"], int | float]
    note: str = ""
```

**Layer B — the composition (`compose`).** Folds the per-op/per-move
costs over a match trace + a placement into a whole-program estimate plus
a per-phase breakdown. This is *exactly* today's
`_samsung_kernel_cycles` / `_aim_kernel_cycles` / `_upmem_kernel_cycles`
/ `_apu_v1_kernel_cycles` body — the phase algebra (Samsung preload→exec→
readback, UPMEM revolver `S + ceil(S·(R-1)/min(T,R))`, AiM `Σ per_op·
iters`, APU v1 outer-loop + DMA). It moves verbatim; only its *reads*
change (§1.5).

```python
@dataclass(frozen=True)
class CostModel:
    name: str                      # e.g. "samsung_faithful", "samsung_optimistic"
    target_name: str               # which structural target it prices
    op_costs:   dict[str, OpCost]  # keyed by Op.name
    move_costs: dict[str, MoveCost]# keyed by Move.name
    constants:  dict[str, int]     # the flat scalars (revolver_latency, ...)
    compose: Callable[["ComposeCtx"], "CostResult"]

    # convenience accessors the compose fn calls instead of
    # target.op(n).cycles / target.move(n).cycles:
    def op_cost(self, name: str, ctx) -> int: ...
    def move_cost(self, name: str, ctx) -> int: ...
    def const(self, name: str) -> int: ...
```

`compose` returns:

```python
@dataclass(frozen=True)
class CostResult:
    cycles: int                    # whole-program estimate
    phases: dict[str, int]         # {"preload":..,"exec":..,"readback":..}
    confidence: str                # "calibrated" | "coarse" | "placeholder"
```

`phases` is what feeds `RunResult.extra` (§2). `confidence` is a static
annotation the model author sets (APU v2's placeholder model declares
`"placeholder"`; the Samsung faithful model declares `"calibrated"`).

### 1.2 What migrates OFF the target, what STAYS

| Lives where | Items |
|---|---|
| **Stays on the target tree** (structure + semantics) | the unit tree + `mapping` fan-outs; `Memory` / `Register` geometry; **which** ops/moves exist (their `name`, `src`, `dst`, `accumulates`); each op's `fn` (the matching semantics); each move's `emit` and each op's `emit` (codegen); `OrOf`/`AnyOf` operand legality. |
| **Migrates to the `CostModel`** | every `cycles=` number now on `Move`/`Op` (the 34 in the fixture); the flat scalar constants now in `Unit.constants` (UPMEM `revolver_latency`); the composition formulas (today inside `spmw_cost_models.py`). |

The acid test the swap-test (task 008) enforces: **a target tree, after
this change, contains no number that is a *cost*.** Geometry numbers
(`mapping=[16]`, `rows=16384`, `vrs.width`, `grf_a.lanes`) stay — they
are structure, and the composition reads them off the *target*, not the
cost model (they describe the device, not its speed). The line is:
*"how many / how wide" stays on the target; "how many cycles" moves to
the CostModel.*

> Edge case the coder must get right: the Samsung
> `PRELOAD_FAN`/`READBACK_FAN` moves abuse `cycles=` as a *fan-out width*
> (369, 4096), not a cycle count (see fixture comment lines ~200–216).
> These are **geometry**, not cost — but they are currently carried as
> `Move.cycles`. Decision: **they move into the CostModel's
> `op_costs`/`move_costs` context as model parameters**, not onto the
> target, because they are *calibration anchors* of the faithful model
> (report 18 §3), not device structure. A pessimistic model that
> recalibrates the preload fan-out must be able to change 369 without
> editing the device. They live in `CostModel.move_costs["PRELOAD_FAN"]`
> as an entry whose `fn` returns the width. The closed-form
> `preload_cyc = (M*K // fan) * wr + crf` stays in `compose`, reading
> `self.move_cost("PRELOAD_FAN", …)` for `fan`.

### 1.3 The per-op cost context (the interface that makes refinement local)

Each `OpCost.fn` / `MoveCost.fn` receives a small immutable context so a
per-op analytical model has what it needs **without reaching into the
trace** (which would re-couple it to composition). Minimum fields:

```python
@dataclass(frozen=True)
class OpCostCtx:
    op_name: str
    iters: int                 # resolved inner trip count for this op site (D3)
    operand_shapes: tuple      # ((M,K),...) operand extents where known
    lane_width: int | None     # device SIMD/fold width (from target geometry)
    mode: str                  # placement mode passthrough (APU v1 sv/sv_lookup)
    extra: dict                # placement.extra passthrough (n_fibers, vr_dma,...)
```

This is the seam the MICRO-2025 per-op APU v1 model plugs into: the
`gvml_lookup_16`/`gvml_add_s16` analytical cost becomes
`OpCost(lambda c: f(c.operand_shapes, c.lane_width))` and nothing in
`compose` or the target changes. `MoveCostCtx` is the analogous struct
for data movement (burst length, src/dst tier, element count).

> v1 minimal-risk note: the existing models read only *scalar* per-op
> costs today. The coder **must still pass the full `OpCostCtx`** even
> where the v1 entries ignore everything but return a constant, so that
> the interface is real from day one and the APU v1 / future refinement
> is a one-entry edit, not an interface change. This is the difference
> between "decoupled" and "decouplable-later" — the task demands the
> former.

### 1.4 Module placement and naming

New file: **`experiments/allo/allo/spmw_cost_model.py`** (singular,
distinct from the existing plural `spmw_cost_models.py`). It owns
`CostModel`, `OpCost`, `MoveCost`, the `*Ctx` structs, and `CostResult`.

The existing `spmw_cost_models.py` (the factories) is **refactored, not
replaced**: each `_xxx_kernel_cycles` body becomes the `compose` of a
concrete `CostModel` instance for that target. The `@cost("kernel_cycles")`
registry (`spmw_cost.py`) is **kept** as the autoschedule-facing surface —
see §1.5 for how the two reconcile so the argmin path is byte-identical.

Concrete `CostModel` instances (the actual numbers) live in a new
**`experiments/allo/allo/spmw_cost_tables.py`** — the "Energy-Reference-
Table" file. One `CostModel` per `(target, flavor)`:
`samsung_faithful`, `aim_faithful`, `upmem_faithful`, `apu_v1_faithful`,
`apu_v2_placeholder`. The swap-test (task 008) ships a second flavor
(`samsung_optimistic` or `_pessimistic`) to prove zero-device-edit swap.

> Rationale for the split (cost_model = mechanism, cost_tables = numbers):
> it is the same discipline as target-structure vs target-cost, applied
> one level down. A researcher refining APU v1 against board profiling
> edits **only `spmw_cost_tables.py`**. Mechanism never moves.

### 1.5 Registration and binding — reconciling with the existing `@cost` registry

Today the autoscheduler calls `get_cost("kernel_cycles", target)(trace,
layout) -> int`. We must not break that surface (every enumerator and
report-18 number rides it). Binding rule:

- A `CostModel` is **bound to a target by name**. Add a registry to
  `spmw_cost_model.py`: `register_cost_model(model)` keyed by
  `(target_name, flavor)`, default flavor `"faithful"`. `spmw_cost_tables.py`
  registers all five faithful models at import.
- The existing `_kernel_cycles_factory(target)` (in `spmw_cost_models.py`)
  is rewritten to: look up the bound `CostModel`
  (`get_cost_model(target.name, flavor)`), and return a closure
  `cost_fn(trace, layout)` that calls `model.compose(ComposeCtx(target,
  trace, layout))` and **returns `result.cycles`** — exactly the int the
  argmin expects. So the autoschedule path is unchanged in shape; only
  *where the numbers come from* changes (the `CostModel`, not
  `target.move(...).cycles`).
- The **flavor selection** for autoschedule is `"faithful"` always (the
  argmin must stay calibrated). The virtual *runner* (§2) is what lets a
  user request a different flavor. This keeps task 008's regression
  guarantee trivial: argmin uses `faithful`, which is byte-identical to
  today's numbers.

This is the merge-safe path: the swap-test proves decoupling via the
runner (§2) and via a direct `compose` call, **without** perturbing the
autoschedule argmin that report-18 depends on.

---

## 2. Answer 2 — the virtual-backend runner contract

### 2.1 Selector: `backend="virtual"` on `compile_for_target`, NOT `run(virtual=True)`

Decision: **`compile_for_target(target, trace, backend="virtual",
cost_flavor="faithful")`**. Reject `run(virtual=True)`.

Why: the headline claim is *toolflow symmetry* — "virtual" is one more
member of `{PIMSimulator, uPIMulator, ramulator2, APU silicon, virtual}`
behind **one dispatch surface**. A `virtual=True` kwarg on `run()` makes
virtual a *modifier of a real backend's run*, which is the opposite of
the claim. A `backend=` selector at compile time makes "virtual" a peer
target. It also lets the virtual backend skip the simulator-specific
codegen ctx entirely (it needs the trace + cost model, not the emitted
`cmds`), which a run-time flag could not cleanly do.

Concretely:

- `compile_for_target` grows an optional `backend: str | None = None`
  kwarg. `None` → today's behaviour (dispatch by `target.name`).
  `"virtual"` → produce a `Compiled` whose `run()` dispatches to the
  virtual runner regardless of `target.name`. Additive; every existing
  positional caller is unaffected.
- A `cost_flavor: str = "faithful"` kwarg rides alongside, stored on the
  `Compiled`, consumed only by the virtual runner.
- `Compiled` grows two additive fields: `backend: str | None` and
  `cost_flavor: str`. `run()`'s dispatch becomes: `runner =
  _BACKEND_RUN["virtual"] if self.backend == "virtual" else
  _BACKEND_RUN.get(target_name)`.

### 2.2 `_BACKEND_RUN["virtual"]` signature and `RunResult`

```python
def _run_virtual(compiled: "Compiled", **inputs) -> RunResult:
    model = get_cost_model(compiled.target.name, compiled.cost_flavor)
    result = model.compose(ComposeCtx(compiled.target, compiled.trace,
                                      compiled.layout))
    return RunResult(
        cycles=result.cycles,
        stdout=f"virtual backend: cost model "
               f"{model.name!r} (confidence={result.confidence})",
        backend="virtual",
        extra={"phases": result.phases,
               "confidence": result.confidence,
               "cost_model": model.name,
               "priced_target": compiled.target.name},
    )
```

`RunResult` is **not** changed structurally — it already has
`cycles: int|None`, `stdout`, `backend`, `extra: dict`. The task text
asks for a `confidence` field and an `output` field; **decision: carry
both inside `extra`**, do not add top-level fields. Rationale: adding
top-level `RunResult` fields touches the dataclass that all five existing
runners construct (8+ construction sites in `spmw_codegen.py`), a larger
blast radius for zero functional gain. `extra["confidence"]` and
`extra["phases"]` are the contract; `output` is `None` for v1 (no numpy
reference eval — out of scope) and simply absent from `extra`. Document
the keys in `SPMW_ARCHITECTURE.md` §3 so callers can rely on them.

> The functional output being `None` is fine: the verifier's
> rank-preservation check (§5.5) compares the *simulator's* cycle ranking
> against the *virtual backend's* cycle ranking over a shared candidate
> set — it does NOT compare numerical output, and (critically) it does
> NOT compare the virtual runner against the autoschedule cost_fn (that
> would be a tautology — they are the same function; see §5.5). If a
> later task wants a numpy reference, it adds `extra["output"]`; the
> interface already accommodates it.

### 2.3 The virtual runner is genuinely sim-free (the anti-coupling gate)

`_run_virtual` imports nothing from the simulator paths, calls no
`subprocess`, no Docker, touches no `_pimsim_root()`/`_aim_root()`/
`_upim_root()`. Task 005's no-sim proof (run with binaries removed from
PATH) passes **by construction** because the runner's only inputs are the
`CostModel` and the in-memory trace. The coder must keep `_run_virtual`
in a region of `spmw_codegen.py` (or import it from `spmw_cost_model.py`)
that has no transitive import of the sim-launch helpers. Cleanest:
**define `_run_virtual` to delegate to a `spmw_cost_model.evaluate(target,
trace, layout, flavor)` function**, so the sim-free path lives in the
sim-free module and `_BACKEND_RUN["virtual"]` is a one-line adapter.

---

## 3. Answer 3 — D3 trip-count resolution strategy

### 3.1 The gap, precisely

`_parse_loop_bound(text)` (`spmw_cost_models.py:36`) is a regex over the
affine-upper-bound *string*. It returns `int(text)` when the bound is a
literal, the single number when there's exactly one, else **`None`**.
`None` is silently swallowed (`iters = inner_ub if inner_ub is not None
else 1`), so a symbolic bound (`M*K//16`, `512*c0`, a bound that depends
on an outer loop var or a mapping param) currently *defaults to 1
iteration* — which is exactly where the analytical estimate goes wrong
and the simulator has to be the source of truth.

### 3.2 The strategy: operand-shape + mapping symbolic resolution, as a new resolver

Build a **`resolve_trip_count(match, loop_idx, *, shapes, mapping_env)
-> int | None`** in a new module **`spmw_tripcount.py`**, replacing the
naive `_parse_loop_bound` *for the cost path* (the matcher's own bound
strings are untouched). It resolves a bound in three escalating tiers:

1. **Literal** — `int(text)` or sole-number regex (today's behaviour;
   keep it, it covers the common GEMV case `K=1024`).
2. **Affine over operand shapes** — parse the affine expr (the bound is
   an MLIR affine map; we already have the string and, via the
   `MatchedOp`, the operand `OperandBinding`s with their memref names).
   Bind affine symbols/dims to **operand extents** recovered from the
   workload's tensor shapes (the same `(M,K)` the Samsung model already
   recovers in `_samsung_mk`) and to **mapping params** (the unit-tree
   fan-outs, `target._walk()` mappings — already used by
   `_samsung_workid_count`). Evaluate. This resolves `M*K//16`,
   `512*c0` (with `c0` a bound mapping/loop var), and outer-var-dependent
   bounds.
3. **Genuinely dynamic fallback** — a bound that depends on runtime data
   (data-dependent loop). Return `None` and let `compose` apply an
   **explicit, declared** fallback (not the silent `=1`): the model
   author chooses per op via a `CostModel`-level
   `dynamic_trip_default(op_name) -> int | "unbounded"`. For v1 the
   default is `1` *but emitted into `phases` as a `"dynamic_assumed"`
   marker* and reflected in `confidence="coarse"`, so the estimate is
   never silently wrong — it is *visibly* coarse. No current corpus
   workload (gemv/FFN/batched on Samsung/UPMEM/AiM) hits tier 3; this is
   the honesty seam, not a hot path.

### 3.3 Where shapes come from

The operand extents are already available: the trace's `MatchedOp`s carry
`enclosing_loops` (var, lb, ub, step) and `operands` (`OperandBinding`
with `memref_name`). The workload's tensor shapes are recoverable from
the MLIR the matcher walked. **Scope decision:** v1 resolves bounds that
are affine in (a) other resolved loop bounds and (b) target mapping
params. It does **not** attempt general polyhedral analysis. The corpus
needs exactly tiers 1–2; tier 3 is the declared escape hatch.

### 3.4 Coupling note — this resolver is consumed by `compose`, not the matcher

`resolve_trip_count` lives in the cost layer. The matcher's
`enclosing_loops` strings are the *input*, unchanged. So D3 is **purely
additive to the cost path** and touches no shared Allo file (§5). The
existing `_parse_loop_bound` is retained as the tier-1 helper inside
`spmw_tripcount.py` (moved, not duplicated); the cost models call
`resolve_trip_count` instead.

---

## 4. Answer 4 — shared-reformulation coordination with the host-side task

### 4.1 Owner and merge order — THIS task owns it, host-side consumes it

Both `tasks/virtual-accelerator-backend.md` (D1) and
`tasks/host-side-collective-modeling.md` (D5,
`@allo.cost("host_staging")`) need the same standalone cost-spec home.
The task text says: *"Do this reformulation once, here, first; the
host-side task consumes it. If the host-side task runs first, it owns the
reformulation."* As of this cycle (06262026) the **virtual-accelerator
task is the active one** and no host-side reformulation has landed.

**Ruling: the virtual-accelerator task (this one, 002) OWNS the
`CostModel` reformulation. The host-side task CONSUMES it.**

Concrete contract the host-side task inherits:

- `@allo.cost("host_staging")` becomes a **second cost name** registered
  the same way `kernel_cycles` is, but its factory returns a closure that
  prices host collectives. Its per-op costs live in the **same
  `CostModel` two-layer shape** (op/move table + compose), so host
  staging is just another phase in `CostResult.phases` (e.g.
  `phases["host_staging"]`).
- The `CostModel` dataclass (§1.1) is therefore designed to host
  *multiple cost concerns* (kernel cycles + host staging) — not just
  kernel cycles. Decision: keep `CostModel` per-concern (one for
  `kernel_cycles`, a sibling for `host_staging`), bound to the same
  target, registered under the same `(target_name, flavor)` namespace but
  a different *concern* key. This way the host-side task adds a new
  concern without touching the kernel-cycle `CostModel`.
- **Merge order:** 002 (this) merges first, establishing
  `spmw_cost_model.py` + `spmw_cost_tables.py` + the `CostModel`/`OpCost`/
  `MoveCost`/`ComposeCtx` interface. The host-side task's reformulation
  becomes "add a `host_staging` concern to the existing CostModel
  machinery" — an additive consume, not a re-extraction.

This ruling is recorded in `SPMW_ARCHITECTURE.md` §3 (new "CostModel"
seam) and in architect memory so the host-side cycle does not
re-litigate ownership.

### 4.2 What the host-side task must NOT do (guardrail for the future cycle)

It must not create a parallel cost-spec home (`spmw_host_cost.py` with its
own table format). The whole point is one decoupled cost layer. If the
host-side cycle finds the `CostModel` interface insufficient for
collectives, that is a **`needs-arch` escalation back here**, not a fork.

---

## 5. Answer 5 — blast radius (zero FPGA/AIE regression)

### 5.1 Files touched

| File | Edit | Shared? | Justification |
|---|---|---|---|
| `allo/spmw_cost_model.py` | **NEW** | no (spmw-local) | `CostModel`/`OpCost`/`MoveCost`/`*Ctx`/`CostResult`, registry, `evaluate()`. Mechanism. |
| `allo/spmw_cost_tables.py` | **NEW** | no | the five faithful `CostModel`s + the swap-test flavor. Numbers only. |
| `allo/spmw_tripcount.py` | **NEW** | no | `resolve_trip_count` (D3). Retains old `_parse_loop_bound` as tier-1 helper. |
| `allo/spmw_cost_models.py` | edited | no (spmw-local) | factories rewritten to delegate to `CostModel.compose`; per-target cycle bodies move into `spmw_cost_tables.py` `compose` fns. Cost-path internal. |
| `allo/spmw_codegen.py` | edited, **additive** | partially-shared file, but the edited regions are all spmw-local | `compile_for_target` grows `backend=` + `cost_flavor=` kwargs (default None/`"faithful"`); `Compiled` grows `backend`/`cost_flavor` fields; `_BACKEND_RUN["virtual"] = _run_virtual` (one-line adapter to `spmw_cost_model.evaluate`). No existing runner body changes; `RunResult` dataclass unchanged. |
| `tests/spmw/_fixtures.py` | edited | no (test fixture) | the 34 `cycles=` literals and the `revolver_latency` constant are **removed from the target tree** and moved into `spmw_cost_tables.py`. The target builders keep structure + `fn` + `emit` only. |
| `experiments/allo/SPMW_ARCHITECTURE.md` | edited (architect) | doc | new "CostModel" + "virtual backend" seams; module map rows for the three new modules; this design linked. |

### 5.2 Why there is no FPGA/AIE regression

- **No `allo/ir/*`, `allo/dataflow.py`, `allo/customize.py`,
  `allo/memory.py`, `allo/__init__.py` edit.** The FPGA/AIE path reaches
  Allo through `allo.customize` / `allo.LLVMModule` / `allo.HLSModule`
  and `dataflow.build()`. None of those are touched. The three new
  modules are imported only by the SPMW cost/codegen path.
- **`spmw_codegen.py` is read by the FPGA path?** No. It is a
  Tenon-only module (no upstream Allo test imports `spmw_codegen`). Its
  edits are additive kwargs with defaults; the `_BACKEND_RUN` dict gains
  a key. Zero existing-semantics change.
- **The `cycles=` removal from the fixture** is a *test-fixture* change,
  not a shared-source change. It cannot reach `tests/dataflow/` or
  `tests/customize/` — those build different targets through different
  fixtures.

### 5.3 Upstream tests that gate the shared-ish edits

`spmw_codegen.py` and `_fixtures.py` are not imported by any upstream
test. The guards are the **SPMW** suite:
- `tests/spmw/test_autoschedule.py`, `test_codegen_gemv.py`,
  `test_match_gemv.py` — the baseline three. argmin must be byte-identical
  (faithful flavor, §1.5).
- `tests/spmw/test_e2e_mlp.py`, the Samsung batched tests, the report-18
  invariants (15251, B\*=2, 3.93×) — task 008's regression guard.
- New tests the coder adds: `tests/spmw/test_cost_model_swap.py` (two
  flavors, zero device edit, different estimate), `test_virtual_backend.py`
  (virtual runner returns cycles + phases + confidence),
  `test_tripcount_resolution.py` (symbolic bounds resolve),
  `test_rankpreserve_vs_sim.py` (§5.5: real simulator vs virtual ranking
  over per-backend candidate sets — NOT cost_fn self-comparison).

The FPGA/AIE CI suites (`tests/test_vhls.py`, `test_vitis.py`,
`test_xls.py`, `test_catapult_hls.py`, `test_pynq.py`, `test_nn.py`,
`tests/dataflow/`, `tests/customize/`) are **untouched and must stay
green** — they do not import any module this design changes.

### 5.4 Rollback story

Every edit is additive or test-local. Rollback = `git revert` of the
three new modules + the additive kwargs. The `faithful` flavor makes
argmin numbers identical to pre-change, so even a partial landing cannot
move the report-18 numbers. If the swap-test reveals the decoupling is
incomplete (a cost number still hides on the target), that is a coder
ambiguity escalated back here, not a CI break — the FPGA path never sees
any of it.

### 5.5 Rank-preservation validation — re-scoped to the REAL simulator (defect fix, task 016)

**The defect (goal-check-1 FAIL).** As originally written, §2.2 and §5.3
left the rank-preservation check under-specified: a verifier could
implement it as "virtual argmin == autoschedule argmin." That is a
**tautology** — the autoschedule argmin *is* `get_cost("kernel_cycles",
target)`, and the virtual runner calls the **same** `CostModel.compose`
(§1.5). Comparing them is `cost_fn == cost_fn`; it proves nothing about
whether the analytical model tracks reality. The validation MUST anchor
the ground-truth ranking to **real simulator cycle numbers** produced by
the existing `_BACKEND_RUN` simulator path (PIMSimulator / uPIMulator /
ramulator2), never to the cost model.

**The metric.** For a backend `b` and a candidate set
`C = [c_1 … c_n]` (n ≥ 2, §5.5.1):

1. **Ground truth:** run each `c_i` through the real simulator
   (`compile_for_target(target, trace_i, layout_i).run()` — the *non*-virtual
   path) and record `sim_cycles[i]`. These are the numbers the simulator
   actually emits, with binaries present.
2. **Virtual:** run each `c_i` through
   `compile_for_target(target, trace_i, backend="virtual").run()` and
   record `virt_cycles[i]`.
3. **Rank-agreement (headline, pass/fail):** the *argmin* of `virt_cycles`
   must equal the *argmin* of `sim_cycles` (the virtual backend picks the
   same winner the simulator would). Stronger optional report:
   Kendall-τ / Spearman over the full ordering when n ≥ 3.
4. **Absolute error (reported, NOT the bar):** per-candidate
   `|virt_cycles[i] − sim_cycles[i]| / sim_cycles[i]`. Reported for
   transparency; the task is explicit that cycle-accuracy is **not** the
   goal — bounded error + rank-preservation is.

**Accept bar:** rank-agreement on argmin = 100% across every backend's
candidate set (the winner the simulator picks is the winner the virtual
backend picks). Absolute error is reported, not gated. A *single*
candidate set per backend is not enough — see §5.5.1's "distinguishability"
requirement.

#### 5.5.1 Per-backend candidate sets (the non-trivial part)

A candidate set only tests rank-preservation if **the simulator itself
produces distinguishable cycle numbers across it.** This is the trap a
naive corpus walks into: from arch-110 / T15, Samsung *single-shape* GEMV
is at the hardware floor and placement (EVEN/ODD bank parity) gives a
**zero** simulator cycle delta — so `sim_cycles` is flat, argmin is a
tie-break, and "rank-agreement" is vacuously true. Each backend's
candidate set must be chosen so the **real simulator ranks the candidates
apart**:

| Backend | Candidate set (≥2, sim-distinguishable) | Why the simulator ranks them apart | Sim source |
|---|---|---|---|
| **Samsung** | **batched GEMV**: `weight_resident=False` vs `True` at B≥2 (SPEC-026), e.g. 4096×1024, B∈{2,4}. NOT single-shape placement parity (floor, zero delta). | resident preloads once + B·(exec+readback); non-resident re-preloads per vector. Driver `--batch`/`--native-rebaseline` emit genuinely different cycle totals (report 18: B*=2 crossover, 3.93× asymptote). | `_run_samsung_batched` (real `pim_driver`) |
| **UPMEM** | `n_tasklets ∈ {1, 11}` (revolver fill) on a fixed GEMV/VA shape, AND a shape sweep (gemv 1024²/2048²). | uPIMulator schedules round-robin tasklets; the revolver model `S + ceil(S·(R−1)/min(T,R))` is validated against the sim's tasklet scaling (design 02). Different `T` → different sim cycles. | `_run_upmem` (real uPIMulator) |
| **AiM** | shape sweep gemv 1024² / 2048² / 4096×1024 (opsize=K, SPEC-019). | ramulator2 issues `opsize` column requests per MAC_SBK; bigger K → more sim cycles, monotone. A ranking over shapes exists and the sim emits it. | `_run_aim` (real ramulator2) |

> Why a *shape sweep* counts as a candidate set: rank-preservation over
> {gemv 1024², 2048², 4096²} asks "does the virtual backend order these
> three workloads by cost the same way the simulator does?" That is a real
> simulator-vs-virtual ranking with a non-flat ground truth, and it is the
> cleanest test on backends (AiM) whose single-shape placement is also at
> a floor. The corpus shapes are exactly the task's stated set:
> gemv 1024²/2048²/4096²/4096×1024, FFN, batched.

> APU v1 / APU v2 are **excluded** from the rank-preservation gate. APU v1's
> simulator path is the real GSI board (`_run_apu_v1`), available only when
> the PCI node is present — the gate must not depend on board availability;
> if present, APU v1 SV vs SV-lookup (8 vs 18 cyc/MAC) is a *bonus*
> candidate set, reported but not required. APU v2 is functional-only
> (`perf_is_placeholder=True`, SPEC-011) — it has no simulator cycle
> ground truth, so rank-preservation is undefined there by construction.

#### 5.5.2 Skip discipline (no false PASS when a simulator is absent)

The rank-preservation test requires the **real** simulator to run. If a
backend's simulator is unavailable (`run()` returns `cycles=None`), the
per-backend rank check **skips with an explicit `pytest.skip`** naming the
missing simulator — it must NOT silently PASS, and it must NOT fall back
to comparing the virtual runner against the cost_fn (the very tautology
this fix removes). At least one backend's simulator must be present for
the gate to be meaningful; the verifier records which backends were
exercised vs skipped. This is the inverse of §2.3's no-sim test: §2.3
proves the *virtual* path needs no simulator; §5.5 proves the *virtual
path's ranking is validated against* a simulator that was actually run.

---

## 6. v1 scope boundary (D4) — state it, do not build it

v1 `compose` is **sequential**: Samsung preload→exec→readback summed;
UPMEM revolver; AiM `Σ`; APU v1 outer-loop + DMA summed. **No compute↔DMA
overlap.** What this under/over-counts:

- **Over-counts** wall-clock on backends that overlap weight-stream DMA
  with compute (APU v1 L4→VR burst behind the prior tile's MAC; UPMEM
  MRAM↔WRAM behind tasklet compute). v1 sums them; reality maxes them.
- **Faithful for** Samsung's GEMV phases, which the hardware *does*
  serialize (preload then exec then readback are genuinely sequential on
  the folded path — report 18 §3).

The v2 upgrade is **sum→max** composition. The `CostModel.compose`
interface already accommodates it: `compose` owns the fold, so a v2 model
re-implements the fold as `max` over overlapping phase groups without any
interface change. This is the *design-for-it-defer-it* the task demands.
Recorded as open tension T21 in `SPMW_ARCHITECTURE.md`.

---

## 7. Concrete commitments (the receipts)

1. **`CostModel` is a two-layer per-op interface**: Layer A = a dict of
   `OpCost`/`MoveCost` *callables* (each independently refinable against
   profiling), Layer B = a swappable `compose`. Numbers live in
   `spmw_cost_tables.py`; the target tree carries **zero cost numbers**.
2. **Selector = `compile_for_target(..., backend="virtual",
   cost_flavor=...)`**, NOT `run(virtual=True)`. `_run_virtual` is
   sim-free by construction (delegates to `spmw_cost_model.evaluate`, no
   subprocess/Docker import).
3. **`RunResult` is NOT structurally changed**; `confidence` and `phases`
   ride in `extra`; `output` is absent (None) for v1.
4. **D3 = `resolve_trip_count` (new `spmw_tripcount.py`)**, three tiers
   (literal / affine-over-shapes+mapping / declared dynamic fallback with
   `confidence="coarse"`). No matcher change; cost-path only.
5. **This task owns the cost-spec reformulation; host-side consumes it.**
   `host_staging` becomes a sibling concern in the same `CostModel`
   machinery; merge 002 first.
6. **Zero shared-Allo `.py` edits** beyond additive kwargs in
   `spmw_codegen.py` (a Tenon-only module). No `allo/ir/*`,
   `dataflow.py`, `customize.py`, `__init__.py`. FPGA/AIE CI untouched.
7. **Rank-preservation is validated against the REAL simulator, never the
   cost_fn (§5.5, defect fix task 016).** Ground truth = simulator cycle
   numbers from the non-virtual `_BACKEND_RUN` path; the gate is
   argmin-agreement (virtual winner == simulator winner) over per-backend
   candidate sets chosen so the **simulator** ranks them apart (Samsung
   batched resident-vs-not; UPMEM `n_tasklets`/shape sweep; AiM shape
   sweep). Single-shape Samsung placement parity is excluded (floor =
   flat sim ranking = vacuous). Absent simulator → explicit skip, never a
   silent PASS and never a cost_fn fallback.

---

## 8. Open tensions handed forward

- **T21** (new): v1 sequential vs v2 sum→max overlap. Trigger to revisit:
  a corpus workload where APU v1 / UPMEM DMA-behind-compute overlap makes
  the v1 over-count flip a rank-preservation decision. Until then, v1
  sequential, `confidence` annotates the coarseness.
- **T22** (new, → host-side cycle): whether the `host_staging` concern
  fits the `CostModel` interface or needs a richer collective-cost
  context. Resolved when the host-side task lands; escalate here if not.
- **APU v1 per-op refinement** (the coordinator's reference example): the
  MICRO-2025 per-op analytical model is the *first intended consumer* of
  the `OpCost.fn` seam. It is **not built in this cycle** (v1 entries are
  constants), but the interface is shaped so it is a one-entry edit in
  `spmw_cost_tables.py`. Flagged as the canonical refinability demo for
  Phase 5 / a follow-up.

---

## 9. Implemented (D1 + D2 + D3, task 009)

- D1 CostModel reformulation landed: new `allo/spmw_cost_model.py`
  (mechanism: `CostModel`/`OpCost`/`MoveCost`/`OpCostCtx`/`MoveCostCtx`/
  `ComposeCtx`/`CostResult`, `register_cost_model`/`get_cost_model`/
  `evaluate`, keyed by `(target_name, flavor, concern)`); new
  `allo/spmw_cost_tables.py` (five faithful CostModels + `samsung_optimistic`
  swap flavor, compose bodies lifted verbatim); `spmw_cost_models.py`
  factories rewritten to delegate to the bound CostModel.compose and return
  `result.cycles` (faithful argmin byte-identical: 15251 / B*=2 / 3.93x
  invariants hold). The 34 `cycles=` literals + `revolver_latency` removed
  from `tests/spmw/_fixtures.py` target builders (structure + fn + emit
  only); Samsung PRELOAD_FAN/READBACK_FAN fan-out widths moved into the
  CostModel move_costs as model params.
- D2 virtual runner landed: `compile_for_target(..., backend="virtual",
  cost_flavor=...)`; `Compiled.backend`/`cost_flavor` fields; sim-free
  `_run_virtual` delegating to `spmw_cost_model.evaluate`; `RunResult`
  unchanged (confidence/phases ride `extra`).
- D3 trip-count resolver landed: new `allo/spmw_tripcount.py`
  `resolve_trip_count` (literal / affine-over-shapes+mapping / declared
  dynamic None); `_parse_loop_bound` moved here and re-exported.
- Tests: new `tests/spmw/test_cost_model_swap.py`,
  `test_virtual_backend.py`, `test_tripcount_resolution.py`;
  `test_target_cycles.py` rewritten to assert costs at their CostModel home
  + the "target carries no cost number" acid test; cost-perturbing tests
  (batched falsifier, dual-fiber JUMP, host-residency LD_A, shared-CRF
  trigger) updated to perturb the CostModel's move_costs (restored) rather
  than `target.move().cycles`. Mechanical consequence flagged to architect:
  the fixture decoupling broke more existing SPMW tests than §5.3
  enumerated (every test reading/mutating `target.{move,op}().cycles` or
  `target.revolver_latency`); all updated to the CostModel surface, no
  invariant changed.

### 9.1 D3 tier-3 + D4 doc (task 011)

- `resolve_trip_count` finalised to the spec signature
  `(match, loop_idx=-1, *, shapes, mapping_env)`; `resolve_bound_text` is
  the string-level core for non-inner bounds. Tier-3 (None) no longer
  defaults silently to 1: `CostModel.dynamic_trip_default(op_name)` is the
  DECLARED fallback (v1=1, per-op overridable), and every `compose`
  (Samsung/AiM/UPMEM/APU v1) stamps `phases["dynamic_assumed"]=1` and sets
  `confidence="coarse"` when any tier-3 fallback fires. The corpus is fully
  tier-1/2, so the faithful argmin stays calibrated + byte-identical
  (15251, B*=2, 3.93x verified).
- D4 v1-sequential assumption documented in §6 + T21 (SPMW_ARCHITECTURE.md
  §4): compose sums phases, no compute↔DMA overlap; over-counts overlapping
  backends (APU v1, UPMEM), faithful for Samsung's serialized GEMV phases;
  v2 = sum→max with no interface change.

### 9.2 Phase-5 no-backend demo + refinability showcase (task 015, STRETCH)

- "A backend that ships with the spec": new `demo_pim` structural target
  (`tests/spmw/_demo_target.py`) -- a bit-serial PIM substrate with NO
  simulator, NO hardware, NO `_BACKEND_CTX`/enumerator entry. A kernel is
  developed + estimated purely via
  `compile_for_target(target, trace, backend="virtual")`. To enable this,
  `compile_for_target` gained an additive branch: when
  `backend=="virtual"` AND the target has no codegen ctx, it skips the
  emit walk (cmds=[]) and produces a cost-only `Compiled` (design 04 §2.1).
  Real targets and the `backend=None` path are unchanged.
- Per-op refinability showcase (the MICRO-2025 reference example): two demo
  CostModels in `spmw_cost_tables.py` (`demo_pim_constant`,
  `demo_pim_micro25`, + a `faithful`-aliased default) share ONE compose and
  differ by exactly one OpCost entry -- ADD goes from
  `OpCost(lambda c: 2)` to `OpCost(micro25_add_cost)`, an analytical
  function of the operand bit-width + bit-serial lane geometry carried in
  `OpCostCtx`. The estimate changes (2048 -> 14336 for a 1024-iter s16 add)
  with ZERO edit to compose or any caller. The compose feeds each OpCost a
  rich `OpCostCtx` (lane_width + operand element bits), which the constant
  entry ignores -- proving the interface is real from day one and the
  refinement is a one-entry edit (design 04 §0.1/§1.3, T-APU-v1 resolved as
  a demonstrated pattern).
- Corpus untouched: the five faithful/placeholder models are unchanged;
  245 spmw tests collect clean; cross-backend smoke green.
