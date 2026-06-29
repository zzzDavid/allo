# Design 07 — Cost-abstraction reshape: phase timeline, placed-trace context, knob-cost seam, provenance policy

Phase-0 architect spec for `autosched-cost-model-fidelity` (task file
`003-needs-arch-cost-abstraction-reshape-spec.task`, 15:49 rewrite). This
RESHAPES the SPMW cost *abstraction* — it is not three numeric patches. It
delivers the **seven answers** the task asks for (§A1–A5 + D1–D3 + blast
radius), commits the two dataclasses co-owned with
`autosched-placement-realization`, and proves `faithful` is the
byte-for-byte regression anchor.

Gating decision up front, so the rest reads cleanly: **this is an
objective-only change.** Zero edits to `allo/ir/*`, `dataflow.py`,
`customize.py`, `spmw_codegen.py`, and the `spmw_autoschedule.py` argmin
scoring core. The single autoschedule touch is an opt-in confidence-gate
that is inert by default. Everything else lives in
`spmw_cost_model.py` / `spmw_cost_tables.py` / `spmw_cost_models.py` /
`spmw_tripcount.py` — all SPMW-local, no FPGA/AIE blast radius.

---

## 0. Novelty framing (grounded on reports 27 + 28 — cite, do not claim)

The architect does not survey here; reports 27 (contents) and 28 (shape)
are the completed surveys this spec rides on. The §A novelty is the
*shape* counterpart of report 27's *contents* claim. State it exactly
once, in these terms, so the coder/verifier never over-claim in a
docstring or a test name:

- **A1 phase timeline** — `Phase(resource, latency, ii, count)` is a
  modulo-scheduling **reservation-table-shaped** object; `ii` is the
  initiation interval. The `sum→max` fold is ALCOP's load-compute
  max-over-stages + roofline + uiCA resource-overlap (report 27 §1.1).
  Cite modulo scheduling (Rau HPL-94-115; Lam PLDI'88) and ALCOP
  (MLSys'23). **Do not claim** per-resource accounting, the `sum→max`
  fold, or latency-hiding modeling.
- **A3 knob-cost seam** — per-decision/marginal cost is modulo
  scheduling's `ResMII = max_r uses(r)/count(r)` (per-resource
  decomposition + incremental II) and IVM/semi-naive delta-not-recompute
  (report 28 §A3.2). Cite both. The **one new sub-claim**: each schedule
  lever owns a `cost(value, ctx) -> list[Phase]` contribution that
  composes additively into the *same* phase timeline the full objective
  folds, so a marginal per-knob delta is exact (not approximate) and the
  lever's cost is co-located with its codegen materializer.
- **A4 provenance/confidence** — uncertainty-as-policy is BO/UCB
  acquisition (BaCO ASPLOS'24, BOCA ICSE'21) weighting `sigma(x)`;
  calibration-as-data is DiffTune (MICRO'20) (report 28 §A4.2). Cite
  both. The **one new sub-claim**: the uncertainty band is *symbolic*,
  derived from per-constant provenance (`measured|datasheet|assumption`),
  not a learned posterior variance or conformal residual; and it is wired
  as a trust-region commit-gate in an analytical PIM scheduler. **Do not
  claim** uncertainty-aware optimization or calibration-as-data per se.
- **D1/D2/D3 contents** — report 27: overlap (ALCOP/roofline/uiCA);
  layout-conflict (Linear Layouts ASPLOS'26 owns the GF(2) algebra + the
  conflict predicate; SAFARI PIM line owns the row-buffer penalty
  *quantity*); bit-serial width (Stripes MICRO'16 / BitFusion ISCA'18 +
  the GSI APU MICRO'25 Table 5: `mul_u16=201`, `add_u16=12`). The fusion
  into one argmin objective across five PIM substrates is the new part.

The defensible claim, verbatim for the paper, is report 27 §5 + report
28's joint framing. The coder/verifier MUST NOT add a claim beyond this.

---

## A1 — Phase-timeline keystone (answer 1)

### A1.1 The `Phase` schema and the `Resource` enum (new, in `spmw_cost_model.py`)

```python
import enum

class Resource(enum.Enum):
    COMPUTE = "compute"   # the device exec lane (MAC/MUL/ADD/AF on PIM)
    DMA     = "dma"       # on-device data movement (L4<->L1, MRAM<->WRAM, bank<->GRF)
    HOST    = "host"      # host<->device staging (the host_staging concern)
    LOCALITY = "locality" # D2 row-buffer/bank-conflict penalty (a COMPUTE-side stall,
                          # but tagged separately so it is greppable and can be zeroed)

@dataclass(frozen=True)
class Phase:
    resource: Resource
    latency: int          # pipeline fill/drain (depth); cycles before steady state
    ii: int               # initiation interval (steady-state per-iteration throughput)
    count: int            # number of iterations
    tag: str = ""         # free-form provenance label for breakdown ("exec","preload",...)
    # steady-state cycles for this phase:
    #   cyc(phase) = latency + ii * (count - 1)        when count >= 1
    #   cyc(phase) = 0                                  when count == 0
```

`Phase` is `frozen=True` (every cost dataclass here is frozen — immutable,
hashable, no accidental mutation through the fold). `tag` is the
*display* label that preserves today's phase-name breakdown
(`"exec"`, `"stage_resident"`, `"readback"`, `"dynamic_assumed"`,
`"evict_per_call"`) without re-introducing a free-form dict as the
*structural* carrier. The fold reads `resource`/`latency`/`ii`/`count`;
the breakdown surfacing reads `tag`.

**`cyc()` is a module-level pure function** `phase_cycles(phase: Phase) ->
int`, not a method, so a knob's `cost(...)` and the combiner share one
definition. Define it once in `spmw_cost_model.py`.

### A1.2 `CostResult.phases` becomes `list[Phase]`

```python
@dataclass(frozen=True)
class CostResult:
    cycles: int
    phases: list[Phase] = field(default_factory=list)   # was: dict
    confidence: str = "calibrated"
```

This is the load-bearing breaking change inside SPMW. Every `compose`
that today does `phases={"exec": total}` now does
`phases=[Phase(Resource.COMPUTE, latency=total, ii=0, count=1, tag="exec")]`
(the degenerate single-iteration form: `latency + ii*(1-1) = latency =
total`). `evaluate` (`spmw_cost_model.py:281-317`) merges two `phases`
lists by concatenation, not `dict.update`; the `"preload"` alias becomes
a derived helper, not a dict key (see A1.5).

There is exactly one consumer of the old dict shape outside the cost
modules — the `RunResult.extra` surfacing path. Provide a back-compat
shim `phases_as_dict(phases: list[Phase]) -> dict` (group by `tag`, sum
`phase_cycles`) so `RunResult.extra["phases"]` and the report harness keep
their dict view. **This shim is the ONLY place the dict shape survives**;
it is read-only and derived. (The coder must grep every `result.phases[`
/ `.phases.get(` / `.phases.update(` / `for ... in result.phases` to
confirm no other structural reader remains; convert each.)

### A1.3 The resource-aware combiner: sum-within / max-across + fill/drain

The combiner replaces `_whole_program_sum`
(`spmw_cost_models.py:103-113`). It takes `list[Phase]` (the merged
device + host_staging timeline) and a *flavor-bound fold rule*:

```python
def combine(phases: list[Phase], *, overlap: bool) -> int:
    # 1. sum within each resource class (serial issue on one resource)
    per_resource: dict[Resource, int] = {}
    for ph in phases:
        per_resource[ph.resource] = per_resource.get(ph.resource, 0) + phase_cycles(ph)
    if not overlap:
        # faithful fold: everything serial -> SUM across resources too.
        return sum(per_resource.values())
    # overlap fold: resources with no data dependency run concurrently ->
    # MAX across independent resources, plus the fill/drain of the
    # subordinate (hidden) resources that cannot be fully buried.
    return _max_across_with_fill_drain(per_resource, phases)
```

**Two named fold rules, bound per flavor (D1):**

- `faithful` binds `overlap=False` → `sum(per_resource.values())`. Because
  every faithful phase is `latency=total, ii=0, count=1`, the
  within-resource sum equals the old per-phase sum, and the across-resource
  sum equals the old `device + host` whole-program sum. **Byte-identical
  by construction** (proof in A1.6).
- `overlap` binds `overlap=True` → max-across + fill/drain (D1, design body
  in §D1). This is a *new flavor*, never the default (task scope: "make
  `overlap` the default" is explicitly out).

**The flavor→combiner binding** lives in `_kernel_cycles_factory`
(`spmw_cost_models.py:116-142`) and in `evaluate`
(`spmw_cost_model.py:281-317`): both currently call
`_whole_program_combiner(device, host)` on two scalars. They change to
(a) concatenate the device + host `phases` lists, and (b) call
`combine(merged_phases, overlap=<bound by flavor>)`. The flavor string
(`"faithful"` / `"overlap"`) is already threaded through both (the
`flavor=` arg of `get_cost_model`); the combiner is selected from it by a
small `_COMBINER_FOR_FLAVOR = {"faithful": False, "overlap": True,
"optimistic": False, "micro25": ..., ...}` table — default `False`
(serial sum) for any flavor not in the table, so every existing flavor
(`optimistic`, Mortise's three, the demo's) keeps `sum`.

**Critical invariant:** the per-flavor difference for D1 is *only* the
fold rule. There is no parallel hand-coded `overlap` compose. A
backend's `compose` produces ONE `list[Phase]`; `faithful` and `overlap`
fold the same list differently. This is the report-27/28 keystone (overlap
is a property of the timeline, not a flavor to maintain).

### A1.4 What `latency` / `ii` mean per backend (the seam contract)

Each backend's `compose` must populate `latency`/`ii`/`count` so that the
faithful fold reproduces today AND the overlap fold is meaningful. The
contract per backend:

| Backend | COMPUTE phase | DMA phase | HOST phase |
|---|---|---|---|
| Samsung | `latency=exec_cyc, ii=0, count=1` (the device-exec body; today's single `"exec"` scalar) | — (no on-device DMA phase today) | `host_staging` preload/readback, each a `Phase(HOST, latency=cyc, ii=0, count=1)` tagged `stage_resident`/`stage_per_call`/`readback` |
| AiM | `latency=total, ii=0, count=1` | — | — |
| UPMEM | `latency=cycles, ii=0, count=1` (the revolver-folded exec; the revolver math stays inside compose, see A1.7) | — (MRAM moves are folded into op cost today, not a separate phase — see D1 §UPMEM for the overlap refinement) | opt-in `host_stage` scatter/gather, `Phase(HOST,...)` |
| APU v1 | `latency=op_total, ii=0, count=1` | `latency=move_total, ii=0, count=1` tagged `vr_dma` (today summed into exec; A1 SPLITS it onto `Resource.DMA` — see A1.7) | — |
| APU v2 | `latency=n, ii=0, count=1` (placeholder) | — | — |
| demo_pim | `latency=cycles, ii=0, count=1` | — | — |
| Mortise | exec = Samsung's verbatim | — | capacity-aware host_staging (preload/evict/readback as HOST phases) |

The general rule: **today's "this loop runs `count` times at `per_op`
each" becomes `Phase(R, latency=per_op, ii=per_op, count=count)`** ⇒
`cyc = per_op + per_op*(count-1) = per_op*count`, exactly today's
`per_op * iters`. The degenerate `count=1` form (`latency=total, ii=0`)
is the equivalent collapse when a compose has already multiplied iters
into a running total (Samsung exec, AiM, UPMEM, APU). **Either encoding
reproduces today; the coder picks per backend whichever keeps the diff
smallest** (the proof in A1.6 covers both encodings).

### A1.5 Which backends already separate phases vs which must split

- **Already separable (no split needed):** Samsung — preload / exec /
  readback are *already* three phases across two concerns
  (`_samsung_compose` emits exec; `_samsung_host_staging_compose` emits
  resident/per_call/readback). A1 just makes each a typed `Phase`;
  `evaluate` already concatenates the two concerns' breakdowns. Mortise —
  same (it clones the Samsung host_staging shape + `evict_per_call`).
- **Must split (today summed into one scalar):** APU v1 — today
  `_apu_v1_compose` adds `_apu_v1_move_cycles(...)` into `total` and emits
  one `"exec"` phase (`spmw_cost_tables.py:563-564`). A1 emits TWO phases:
  `Phase(COMPUTE, exec_ops)` and `Phase(DMA, move_total, tag="vr_dma")`.
  Under `faithful` (sum-across) the total is unchanged
  (`exec_ops + move_total`, byte-identical). Under `overlap` the DMA can
  hide behind compute (D1 §APU v1). UPMEM — the revolver-folded exec is
  one COMPUTE phase today; the MRAM/WRAM moves are already inside the
  per-op cost (`LD_MRAM=1000`), not a separate phase. **A1 does NOT
  re-split UPMEM's MRAM cost** (that would change the faithful number).
  The UPMEM overlap arm (D1 §UPMEM) is documented but its phase split is a
  refinement flagged for D1, not forced by A1 — A1 keeps UPMEM as one
  COMPUTE phase so the faithful number is frozen.

**Rule the coder must hold:** A1 NEVER changes which numbers a faithful
compose produces. If splitting a phase would require re-deriving a number
(UPMEM MRAM), DO NOT split it in A1 — leave it folded, and let the D1 task
decide whether the overlap arm needs the split. A1's only job is to make
the *carrier* a `Phase` list whose faithful fold is byte-identical.

### A1.6 BYTE-FOR-BYTE proof: faithful = (latency=ii=per_op, sum) reproduces every current scalar

The claim is: for every registered faithful/non-overlap model, the new
`combine(phases, overlap=False)` returns exactly the old
`compose(...).cycles`. Proof, per the two legal encodings:

1. **Collapsed encoding** (`latency=total, ii=0, count=1`):
   `phase_cycles = total + 0*(1-1) = total`. A compose that today returns
   `cycles=total` with `phases={"exec": total}` now returns
   `phases=[Phase(R, total, 0, 1, "exec")]`. `combine(overlap=False)` =
   sum over the single phase = `total`. ∎
2. **Loop encoding** (`latency=per_op, ii=per_op, count=iters`):
   `phase_cycles = per_op + per_op*(iters-1) = per_op*iters`. A compose
   that today accumulates `total += per_op*iters` over matches becomes a
   list of one `Phase` per match; `combine(overlap=False)` sums them =
   `Σ per_op*iters` = `total`. ∎
3. **Whole-program (device + host_staging):** today
   `_whole_program_sum(device, host) = device + host`. New: concatenate
   device `phases` (all COMPUTE/DMA) + host `phases` (all HOST), then
   `combine(overlap=False)` = `Σ_resource Σ_phase phase_cycles` =
   `(device exec + APU dma) + (host preload + readback)` = `device +
   host`. ∎

**The concrete anchor numbers that must survive byte-for-byte** (the
verifier's `faithful`-anchor gate, task 009): Samsung GEMV B=1 = **15251**;
batched-GEMV crossover **B\*=2**; asymptote **3.93×**; the report-18
whole-program **114992 / 457336**. Plus every per-backend per-op number
(AiM `Σ per_op·iters`, UPMEM revolver, APU v1 SV/SV-lookup, the
demo/Mortise flavors). The coder's Phase-1 regression gate (task 004) runs
the existing `test_rankpreserve_vs_sim.py` corpus + the report-18 invariant
tests and asserts equality.

**Rollback:** the entire reshape's rollback is "select `faithful`" — it is
the frozen anchor by construction. If overlap/locality/width ever regress
a number, the flavor table reverts that flavor's combiner to `False`/no
addend; the faithful path never moves.

### A1.7 Where the revolver / move math lives (do not leak into the combiner)

The combiner is generic (knows only `Phase`). Backend-specific arithmetic
stays *inside* the backend `compose`, which emits already-resolved
`Phase`s:

- **UPMEM revolver** (`cycles = s + ceil(s*(r-1)/min(t,r))`) is computed
  inside `_upmem_compose` and emitted as one `Phase(COMPUTE,
  latency=cycles, ii=0, count=1)`. The combiner never sees `r`/`t`.
- **APU v1 vr_dma** move cost is computed inside `_apu_v1_compose` (via
  `_apu_v1_move_cycles`) and emitted as a `Phase(DMA, latency=move_total,
  ii=0, count=1)`. The combiner never sees `n_moves`/`per_move`.

This keeps the combiner substrate-agnostic and the per-backend physics
auditable in one place (the anti-hardcoding gate, task 010).

---

## A2 — Placed-trace context + shared `AccessDescr` (answer 2)

### A2.1 `compose` receives the PLACED/allocated trace, not the bare match list

Today `ComposeCtx(target, trace, layout)` carries the bare `MatchTrace`
and the `Placement` (`layout`). The placement *is* the resolved
allocation (`layout.placements: dict[memref_name -> Handle]`,
`layout.extra`, and — after placement-task D3 — the promoted
`LinearLayout`). So the placed information is *already reachable* through
`ctx.layout`; what is missing is (a) the per-op `fn` never gets it, and
(b) there is no typed `AccessDescr` derived from it.

**Decision (additive, lowest blast radius):** keep `ComposeCtx`'s three
fields. Do NOT introduce a separate "placed trace" object. The compose
ALREADY runs after `allocate(...)` returns the refined `Placement`
(`spmw_autoschedule.py:752-758`: `alloc = allocate(...); kc =
cost_fn(sub_trace, alloc.placement)`), so the trace+placement pair handed
to `compose` is already the placed/allocated view. The reshape's job is to
**thread that placement down to the per-op `fn`** by enriching
`OpCostCtx`/`MoveCostCtx`, and to **derive `AccessDescr` from the
layout** once per operand.

This is the right call because: (1) it is additive — no new top-level
type, no change to the `compose(ctx) -> CostResult` signature, no change
to the argmin call site; (2) the placement already carries everything
(handles, `extra`, layout); (3) it keeps the FPGA/AIE blast radius at
zero (these dataclasses are SPMW-local, imported by nobody upstream).

### A2.2 Enriched `OpCostCtx` / `MoveCostCtx`

Add fields (all defaulted, so every existing constructor call still
type-checks and the demo's rich construction is unchanged):

```python
@dataclass(frozen=True)
class OpCostCtx:
    op_name: str
    iters: int = 1
    operand_shapes: tuple = ()
    lane_width: int | None = None
    mode: str = ""
    extra: dict = field(default_factory=dict)
    # NEW (A2):
    placement: dict = field(default_factory=dict)   # {role -> Handle} for THIS op
    access: "AccessDescr | None" = None             # layout-derived access pattern (D2)
    live_set: int | None = None                     # register pressure here (placement task reads)
    dtype_bits: int | None = None                   # D3: operand element bit-width

@dataclass(frozen=True)
class MoveCostCtx:
    move_name: str
    iters: int = 1
    operand_shapes: tuple = ()
    elem_count: int | None = None
    src_tier: str | None = None
    dst_tier: str | None = None
    extra: dict = field(default_factory=dict)
    # NEW (A2):
    placement: dict = field(default_factory=dict)
    access: "AccessDescr | None" = None
    live_set: int | None = None
```

`dtype_bits` is the D3 width carrier. It is distinct from `lane_width`
(device SIMD geometry); `lane_width` is "how wide is the lane", `dtype_bits`
is "how many bits is this operand's element". The demo's `micro25_add_cost`
reads `operand_shapes[0][0]` for bits today; D3 (task 007) reads
`ctx.dtype_bits` at the real APU sites (the cleaner field), falling back to
the operand-shape encoding for the demo. **Default-absent
`dtype_bits=None` ⇒ today's constant** — that is the D3 back-compat anchor
(non-APU and faithful-for-APU keep flat costs unless a width is threaded).

`placement`/`access`/`live_set` default to empty/None so:
- every backend's faithful compose can keep building a *bare*
  `OpCostCtx(name)` and the faithful number is unchanged (the fields are
  ignored by every constant `OpCost.fn`);
- the enrichment is **opt-in per compose**: a compose that wants locality
  builds the rich ctx (D2); one that does not, does not. **A2 only adds the
  fields + defines `AccessDescr` + threads `placement` where a compose asks
  for it.** Population of `access` is D2's job (task 006); population of
  `dtype_bits` is D3's job (task 007). A2 (task 004) wires the *plumbing*,
  not every value — exactly the "shape now, fill as terms land" discipline.

### A2.3 `AccessDescr` — the shared dataclass (CO-OWNED with placement-realization)

**This is the first of the two co-owned dataclasses.** It is consumed by
(a) the D2 locality term here, and (b) the allocator's non-zero base
access cost in `autosched-placement-realization` D2. Both must consume ONE
notion of "access pattern", not two. Definition (placed in
`spmw_cost_model.py`, the lowest shared import point both tasks already
import):

```python
@dataclass(frozen=True)
class AccessDescr:
    """Layout-derived access pattern for one operand at one access site.

    Derived from the chosen LinearLayout / Placement handle stride. Carries
    exactly the quantities a bank/row-buffer locality cost needs, and the
    quantities the allocator's base-access cost needs — one type, two
    consumers (cost-fidelity D2 + placement-realization D2).
    """
    tier: str                       # "register" | "near_bank" | "scratchpad" | "dram"
                                    #   — the memory tier this operand lives in
    stride: int = 1                 # element stride between consecutive lane accesses
    bank_dims: tuple = ()           # out-dim names that index banks (from the layout)
    n_banks: int = 1                # bank fan-out the access touches
    conflict_free: bool = True      # describes_conflict_free over (bank_dims, varying)
    conflict_count: int = 0         # # of varying-input vectors that collide on a bank
                                    #   (0 iff conflict_free; the D2 penalty multiplier)
    row_hits: int | None = None     # resolved open-row reuse count, if derivable (else None)
```

**Provenance of each field (the anti-hardcoding discipline, task 010):**
- `tier` ← the handle type / target tree (a `Register` ⇒ `"register"`; a
  `MemoryRef` into a near-bank `Memory` ⇒ `"near_bank"`; etc.). Derived,
  never pasted.
- `stride`/`bank_dims`/`n_banks` ← the promoted `LinearLayout` on the
  `Placement` (placement-task D3 carries it). Until D3 lands, A2 derives
  them from the materialised handle's geometry; if neither is available,
  `AccessDescr` defaults to the conflict-free identity (stride=1,
  conflict_free=True, conflict_count=0) — which makes the D2 penalty 0
  (back-compat by construction).
- `conflict_free`/`conflict_count` ← `LinearLayout.describes_conflict_free(
  bank_dims=..., varying_inputs=...)` (`spmw_linear_layout.py:521`).
  `conflict_count` extends the boolean: it counts the nonzero
  `varying_inputs` vectors whose bank-projection is zero (the loop in
  `describes_conflict_free` that today early-returns `False` instead
  *counts*; a `conflict_count(...)` companion method or a `count=True`
  kwarg — see A2.4). **conflict_free=True ⇒ conflict_count=0** is an
  invariant the constructor asserts.

**`AccessDescr.identity()`** classmethod returns the back-compat default
(register tier, stride 1, conflict-free, count 0) so any compose that has
no layout to derive from gets a zero-penalty descriptor — this is the
mechanism that makes "conflict-free → 0" hold by construction for every
existing corpus shape.

### A2.4 The conflict-count companion (read-only addition to `spmw_linear_layout.py`)

D2 needs a *count*, not just a boolean. The cleanest additive change:
add a method `conflict_count(self, *, bank_dims, varying_inputs) -> int`
to `LinearLayout` that runs the SAME enumeration as
`describes_conflict_free` but counts the colliding vectors instead of
early-returning. **This is additive** (a new method; `describes_conflict_free`
is unchanged — it can even delegate: `return self.conflict_count(...) == 0`,
but to keep the existing method byte-identical for its callers, leave it
as-is and add the counter alongside). `spmw_linear_layout.py` is an
SPMW-local file (no upstream importers — verified: it is `spmw_*`); this is
a zero-blast-radius additive method. **The coder for task 006 adds it;**
A2/task 004 does not touch `spmw_linear_layout.py`. (Flagged here so the D2
coder knows the predicate is theirs to extend, additively.)

---

## A3 — `Knob.cost(value, ctx) -> list[Phase]` seam (answer 3)

### A3.1 The contract (cost side ONLY — the second co-owned dataclass)

**This is the second co-owned dataclass: the typed knob registry.** The
FULL knob is owned by `autosched-placement-realization` D4
(`candidates()` and `emit()` — the codegen materializer). **This task owns
ONLY the `cost(...)` method's contract and wires `compose` to call it.**
The shared shape (defined in a new small module `spmw_knobs.py`, or — to
minimize new files — as a `Protocol`/ABC in `spmw_cost_model.py` that the
placement task's concrete registry subclasses):

```python
class Knob(Protocol):
    name: str
    def candidates(self, ctx) -> list:            # OWNED by placement-realization D4
        ...
    def cost(self, value, ctx: "ComposeCtx") -> list["Phase"]:   # OWNED HERE
        """The chosen value's contribution to the A1 phase timeline.
        Returns a list[Phase] that compose concatenates and the combiner
        folds — the SAME timeline the base op/move phases live on."""
        ...
    def emit(self, value, ctx):                   # OWNED by placement-realization D4
        ...
```

**Decision: define the seam as a `cost(value, ctx) -> list[Phase]`
callable, registered per `(target, knob_name)`, and route `compose`
through it as an ADDITIVE term.** Rationale: a knob's cost contribution is
*exactly* a set of `Phase`s on the A1 timeline (e.g. a double-buffer-depth
knob adds a DMA `Phase` whose `ii` hides behind compute under the overlap
fold; a residency knob adds/removes a HOST preload `Phase`). So the knob
seam and the A1 timeline are the same currency — no impedance mismatch.

### A3.2 How `compose` folds base + knob phases

```python
def compose(ctx):
    base_phases = [...]                      # the op/move phases (A1)
    for knob_name, chosen in active_knobs(ctx):       # the knobs this candidate set
        base_phases += knob_cost(ctx.target, knob_name, chosen, ctx)
    cycles = combine(base_phases, overlap=<flavor>)
    return CostResult(cycles=cycles, phases=base_phases, confidence=...)
```

`active_knobs(ctx)` reads the chosen knob values off `ctx.layout`
(today `layout.extra` — `grf_residency`, `crf_issue`, `stage_resident`,
`n_tasklets`, `vr_dma`, `n_fibers`; after placement-task D4, the typed
registry). **In scope here:** define `knob_cost(target, name, value, ctx)
-> list[Phase]` registration + the `compose += Σ knob.cost` plumbing, and
demonstrate it on ONE knob (the natural choice: `stage_resident`, which
already toggles a HOST `Phase` — its cost contribution is literally
"resident ⇒ preload paid once vs per_call", which is the existing Samsung
host_staging branch re-expressed as a knob.cost). **Out of scope here:**
migrating all six levers — that is placement-task D4. This task guarantees
the *cost side* of the contract works and is exact.

### A3.3 How a search evaluates a knob delta (the marginal payoff)

Because `compose = base + Σ knob.cost(chosen)` and `combine` is a fold over
a `Phase` list, a search that changes ONE knob recomputes only that knob's
`cost(...)` and re-folds — it does NOT re-walk the trace. The marginal
delta of changing knob `k` from `v0` to `v1` is:

```python
delta = combine(base + Σ_{j≠k} knob.cost(c_j) + knob.cost_k(v1), overlap=f)
      - combine(base + Σ_{j≠k} knob.cost(c_j) + knob.cost_k(v0), overlap=f)
```

**The exactness proof (task 008's non-tautology test):** a knob-delta
re-score must equal a full `compose` recompute of the changed candidate,
**byte-for-byte**. This proves the decomposition is exact (modulo
scheduling's per-resource accounting + IVM delta-not-recompute — borrowed,
cited), not an approximation. The sibling search-space task is the
consumer (it explodes tiling × residency × depth and needs the marginal
seam to rank at scale); this task only proves the seam is exact.

**Lever migration is explicitly OUT** (placement task D4). This task wires
the seam and proves it on one knob; the other five stay as today's
`layout.extra` reads inside compose until D4 migrates them.

---

## A4 — Provenance tags + calibration record + opt-in confidence gate (answer 4)

### A4.1 Per-constant provenance tag schema

Today provenance lives in `OpCost.note` / `MoveCost.note` free-form
strings (`note="tCCDL, Samsung ISCA'21 §4.1"`), and Mortise already pastes
bracket tags (`"[sim-anchored] ..."`). A4 makes it **structured data**:

```python
class Provenance(enum.Enum):
    MEASURED   = "measured"     # anchored to a sim/HW run (the calibration record names it)
    DATASHEET  = "datasheet"    # from a published spec/paper (tCCDL, MICRO'25 Table 5)
    ASSUMPTION = "assumption"   # an engineering guess, not yet validated

@dataclass(frozen=True)
class OpCost:
    fn: Callable
    note: str = ""
    provenance: Provenance = Provenance.ASSUMPTION   # NEW; default = the honest worst case

@dataclass(frozen=True)
class MoveCost:
    fn: Callable
    note: str = ""
    provenance: Provenance = Provenance.ASSUMPTION   # NEW
```

**Default = `ASSUMPTION`** is deliberate: an untagged constant is, by
default, the least-trusted, so forgetting to tag is conservative (the gate
in A4.3 widens, never silently narrows). The coder for task 004 populates
the **sim-anchored set as `MEASURED`** (the report-18 anchors: Samsung
`STAGE_BCAST=369`, `GATHER_RD=181`, the exec tCCDL=4 anchored to the 15251
run; UPMEM `revolver_latency=11` from uPIMulator src; the APU v1 report-12
moves) and tags the **datasheet constants as `DATASHEET`** (tCCDL, the AiM
JSSC numbers, the MICRO'25 APU per-op table when D3 lands), leaving genuine
guesses (`CRF_TRIGGER`, the optimistic bands) as `ASSUMPTION`. The
verifier (task 010) greps every constant and lists the assumptions — the
"provenance is data, not comments" gate.

`Resource`-mapped: the `optimistic` flavors stop being hand-typed `×0.5`
multipliers in principle (report 28 §A4); A4 only adds the *tags* now — the
"derive the band from provenance" refinement is flagged, not built (the
task scope: "populate over time").

### A4.2 Calibration record on each `(target, flavor)` CostModel

```python
@dataclass(frozen=True)
class CalibrationRecord:
    validated_against: str = ""     # which sim/HW run ("PIMSimulator GEMV M=4096 K=1024")
    residual_error: float | None = None   # |est - measured| / measured on that run, if known
    shape_coverage: tuple = ()      # the shapes this (target,flavor) was checked at
```

Add `calibration: CalibrationRecord = field(default_factory=CalibrationRecord)`
to `CostModel`. Populate the sim-anchored models (Samsung faithful:
`validated_against="report-18 PIMSimulator GEMV M=4096 K=1024",
residual_error=0.0, shape_coverage=((4096,1024),)`; UPMEM/AiM similarly).
The placeholder/coarse models (APU v2, optimistic) carry an empty record.
This is data the confidence-gate (A4.3) and the user/optimizer read; it is
NOT consulted by the argmin scoring core.

### A4.3 The opt-in confidence-gate (THE SOLE permitted autoschedule touch)

The gate is the *only* edit to `spmw_autoschedule.py`, and it MUST default
to today's behavior (commit regardless). The objective-only invariant: the
argmin scoring core (`spmw_autoschedule.py:749-766`,
`scored.append((kc + alloc.total_cost, idx, alloc.placement)); scored.sort;
placements.append(scored[0][2])`) is **untouched**.

**Where it lands:** a new optional parameter on the autoschedule entry
(default off) + a post-argmin check, e.g.:

```python
def autoschedule(..., confidence_gate: bool = False):
    ...
    scored.sort(key=lambda t: (t[0], t[1]))
    chosen = scored[0]
    if confidence_gate:
        _check_confidence(target, sub_trace, chosen, candidates)  # advisory: warn/raise/widen
    placements.append(chosen[2])      # <- UNCHANGED commit
```

`confidence_gate=False` ⇒ `_check_confidence` is never called ⇒ the path
is byte-identical to today (the verifier proves this: every report-18
number + the full argmin with the gate off is byte-identical — task 008 +
009). When `confidence_gate=True`, `_check_confidence`:
- recomputes the chosen candidate's `CostResult.confidence` and, if it is
  `"placeholder"` or `"coarse"`, **does not silently commit** — it emits a
  diagnostic (and, per a `gate_policy` sub-option, optionally raises or
  flags for a sim check). The default policy when the gate is on is
  *warn-and-commit*; *refuse-and-error* is opt-in beyond that.
- The confidence is derived symbolically from the provenance of the
  constants that fed the candidate (report 28 §A4.3: the band is a function
  of which `MEASURED|DATASHEET|ASSUMPTION` tags participated) — but the v1
  gate reads the existing `CostResult.confidence` field (which already
  downgrades to `"coarse"` on tier-3 dynamic, via `_confidence(...)`), so
  the gate is wired without requiring the full provenance-aggregation
  first. The provenance→band aggregation is a later refinement; the gate's
  *mechanism* (read confidence, refuse-low) lands now.

**The tier-3 dynamic-trip fix (A4 ↔ A5):** today
`dynamic_trip_default` is plumbed (`spmw_cost_model.py:174-185`) and the
composes already downgrade confidence to `"coarse"` on a tier-3 fall-through
(`_confidence(model, dynamic)`), but with the gate OFF that `"coarse"` is
invisible to the committer — the silent `=1` ships. With the gate ON, a
candidate whose cost rests on a tier-3 fall-through (confidence
`"coarse"`) is no longer silently committed. **A5 closes the *other* half:
an unknown op/move is now a hard error (not silent), and a tier-3 trip is a
declared default + the coarse-confidence gate, never an invisible `1`.**

---

## A5 — Closed op/move vocabulary; hard-error on unknown (answer 5)

Today three silent fallbacks paper over an unknown op:
- AiM `_aim_compose`: `per_op = 4` when `not has_op_cost(name)`
  (`spmw_cost_tables.py:303-304`).
- UPMEM `_upmem_compose`: `per_op = gpr_cyc` (the ADD baseline) when
  unknown (`:356-359`).
- APU v1 `_apu_v1_compose`: `per_op = add_cyc` when unknown (`:543-544`).

**Decision: close the vocabulary — an op/move name not in the model's
`op_costs`/`move_costs` is a `KeyError`/hard error, not a silent default.**
The mechanism is already present: `CostModel.op_cost(name)` *already*
raises `KeyError` when the entry is missing (`spmw_cost_model.py:189-192`).
A5 simply DELETES the `if has_op_cost(...) else <silent default>` guards in
the three composes and lets `op_cost(name, ctx)` raise. The error message
must name the model and the missing op (it already does).

**Migration safety (so this does not break the corpus):** before deleting
the guards, the coder confirms every op name that the corpus traces emit IS
in each backend's `op_costs` (the GEMV/FFN/batched traces emit only
`MAC`/`MUL`/`ADD`/`AF` — all present in every faithful model). If a corpus
trace emits an op the model lacks, that is a *real* gap A5 surfaces (it was
being silently mispriced at 4 cycles); the fix is to add the op to the
model, not to keep the silent default. The verifier (task 010) confirms no
corpus shape hits a hard error AND that an injected unknown op does raise.

**Tier-3 trip policy (A5 ↔ A4):** a genuinely dynamic (tier-3) trip count
stays a *declared* `dynamic_trip_default` + a `"coarse"` confidence that
the A4 gate acts on — NOT a silent `=1`. The `_resolve_iters` /
`_confidence` machinery (`spmw_cost_tables.py:80-101`) already does the
"declared default + coarse flag"; A5/A4 make the coarse flag *consumed* by
the gate. The remaining silent path (the APU v1 outer-loop tier-3 at
`:556-561` that multiplies by `d if isinstance(d,int) else 1` and sets
`dynamic=True`) is already non-silent in the confidence sense; the gate
makes it visible to the committer.

---

## D1 — Overlap semantics (answer 5 in the task's numbering)

`overlap` is a new flavor binding `combine(..., overlap=True)`. Per backend:

### D1.1 Concurrent resources per backend

| Backend | Concurrent (can overlap) | Serial (data-dependent) |
|---|---|---|
| APU v1 | `Resource.DMA` (L4→VR weight stream) hides behind `Resource.COMPUTE` (prior tile's MAC) — the retile↔DMA overlap | the FIRST tile's DMA cannot hide (fill); the LAST compute's result drain |
| UPMEM | `Resource.DMA` (MRAM↔WRAM) behind `Resource.COMPUTE` (tasklet) — IF the MRAM cost is split into a DMA phase (flagged; A1 keeps it folded, so the UPMEM overlap arm needs the D1 coder to split `LD_MRAM` into a DMA phase first) | revolver fill |
| Samsung | NONE meaningful for GEMV — preload→exec→readback is genuinely serial (the host must finish broadcasting the weight before the device strobes). Samsung's overlap fold = its sum fold (no concurrent resources). This is why Samsung is the faithful anchor. | all three phases serial |
| AiM | none modeled (single exec phase) | — |

### D1.2 Fill/drain formula

For two resources A (dominant, e.g. COMPUTE) and B (subordinate, e.g. DMA)
that pipeline:

```
overlap_cyc(A, B) = max(cyc_A, cyc_B) + fill + drain
```

where, when B feeds A (B's tile `i` enables A's tile `i`), `fill` = the
first B iteration's `latency` (one DMA before any compute can start) and
`drain` = the last A iteration's tail beyond the last B (typically 0 if A
dominates). In the `Phase(latency, ii, count)` form, a B phase that fully
hides behind A contributes only its **fill** (`latency_B`) to the total;
its steady-state `ii_B*(count-1)` is absorbed into A's. The general
`_max_across_with_fill_drain` rule:

```
total = Σ_independent_groups max_over_resources_in_group( per_resource_cyc )
        + Σ_subordinate_phases fill_drain(phase)
```

For v1, the conservative form (honest, and matching ALCOP's "the longest
stage dominates") is:
```
total = max(cyc_COMPUTE, cyc_DMA) + latency_DMA_first   (fill)  + 0 (drain)
        + cyc_HOST   (host staging is serial vs device on every PIM backend today)
```
i.e. DMA hides behind compute up to the longer of the two, plus one DMA
fill; HOST staging does NOT overlap device exec (the host must stage before
the device runs — design 05 §9 / T18 keeps it serial this cycle).

### D1.3 Honest under/over-count (T21)

The two-class `max` fold **over-counts** when partial overlap exists (a DMA
that hides only half behind compute is priced as fully hidden) and
**under-counts** bank-arbitration contention (two consumers of one DMA
queue — explicitly out of scope, the A1 timeline models overlap, not
contention; task scope D4). The claim is "a structural max-over-resources
fold," NOT "an accurate pipeline simulator" (report 27 §1.1 T21 boundary).
The verifier's bar (task 009) is **rank-correctness of the flip**, not
cycle accuracy — a large absolute error WITH a matching argmin is the
non-tautology tell; exact equality is the red flag (the cost model checking
itself).

### D1.4 Worked example — APU v1 retile↔DMA

A GEMV tiled into `n_weight_tiles` weight tiles, each DMA'd L4→VR then
MAC'd. Today (faithful, sum): `exec_ops + move_total = Σ MAC + n_moves *
(DMA_L4_L1 + LD_VR)`. Two candidates the search would produce: (a)
`vr_dma="intra"` (re-DMA per output tile boundary), (b) `vr_dma="inter"`
(DMA once, reuse) — these already differ in `n_moves`
(`_apu_v1_move_cycles`). Under faithful both are `exec + move` (the DMA is
fully charged). Under overlap, candidate (b)'s smaller DMA hides behind
compute: `max(exec, move_b) + fill` < `exec + move_b`. The flip the demo
constructs: a double-buffered (b) schedule scored *below* the serialized
(a), where faithful scores `exec + move_a` vs `exec + move_b` (it can rank
them by `n_moves` but cannot price the *hiding* — the overlap win is
invisible to it). **This is the D1 argmin-flip candidate set (task 005 +
009).**

### D1.5 Worked example — UPMEM MRAM↔compute

A tasklet streams a tile from MRAM (`LD_MRAM=1000`) then computes on it.
Today the MRAM cost is *inside* the per-op fold, not a separate phase, so
A1 keeps it folded (faithful frozen). For the overlap arm, the D1 coder
splits the MRAM portion into a `Phase(DMA, latency=1000, ii=1000,
count=n_tiles)` and the compute into `Phase(COMPUTE, ...)`; with enough
tasklets the MRAM stream hides behind compute (`max(compute, mram) + fill`
vs `compute + mram`). **This split is the D1 coder's call (task 005), NOT
A1's** — A1 must keep UPMEM faithful byte-identical, so the split only
exists in the overlap fold path. Flag: if splitting MRAM changes the
faithful number, do NOT split it for faithful; the overlap arm gets a
parallel phase construction guarded by the overlap flavor.

---

## D2 — Locality + D3 bit-width (answer 6)

### D2.1 The locality `Phase` from `AccessDescr`

A `locality` contribution is a `Phase(Resource.LOCALITY, latency=penalty,
ii=0, count=1, tag="locality")` added in each backend's compose, computed
from `AccessDescr`:

```
penalty = access.conflict_count * row_buffer_miss_cyc[target]
```

- `access.conflict_count` is layout-derived (A2.3 / A2.4) — the count of
  varying-input vectors that collide on a bank.
- `row_buffer_miss_cyc[target]` is **one provenance-tagged
  (MEASURED|DATASHEET) per-backend row-buffer constant**, sourced like
  `tCCDL` (Samsung row-miss delta from the ISCA'21 timing; AiM from JSSC;
  UPMEM MRAM row from HPCA'24). It is a NEW entry in each backend's
  `constants` dict (`CostModel.constants`), tagged, never a pasted
  magnitude in the compose body. The SAFARI PIM line owns the *quantity*
  (report 27 §1.2 — cite, do not claim).

### D2.2 Conflict-free → 0 proof (back-compat by construction)

`optimal_swizzle` emits conflict-free layouts for every current corpus
shape ⇒ `describes_conflict_free(...) == True` ⇒ `conflict_count == 0` ⇒
`penalty == 0` ⇒ the LOCALITY phase is `latency=0` ⇒ contributes 0 to the
fold (under both faithful and overlap, since `max(x, 0) = x` and `x + 0 =
x`). **So the locality term is provably inert on the existing corpus** —
the faithful number is unchanged, and the verifier (task 006 + 010)
asserts every corpus shape yields `conflict_count == 0`. The term only
moves a number when a *worse* swizzle (a deliberately constructed
parity-colliding layout) is enumerated — the D2 argmin-flip candidate set:
two swizzles with different `conflict_count`, where faithful (no locality
phase) scores them equal and the locality term separates them. The win
that finally makes `LinearLayout` cost-relevant (today the swizzle is fixed
upstream by a boolean satisfier and never re-ranked — report 27 §4).

### D2.3 The coordination with placement-realization

The allocator's non-zero base access cost (placement-task D2) consumes the
SAME `AccessDescr.tier` + `stride` to charge register-vs-near-bank-vs-
scratchpad access. Both this task's locality term and the allocator's base
cost read one `AccessDescr` per operand — the co-design requirement. **The
allocator's base cost is NOT computed here** (it is placement-task D2); A2
only guarantees the type both consume. If the allocator wants to fold its
base cost into the same A1 timeline (a `Phase(LOCALITY,...)` per handle),
that is the placement task's choice — the seam admits it.

### D3.1 Where dtype/width enters: trace → OpCostCtx → width → cycles

Today every real-target `OpCostCtx` is built empty
(`spmw_cost_tables.py:301,354,541` etc.); only the demo populates
`operand_shapes`. D3 threads operand bit-width to the APU `OpCostCtx`:

1. **Trace → ctx.** The match carries operand dtype (a `MatchedOp` /
   operand has a dtype; if not surfaced, A2/task 004 surfaces it — possibly
   the one `spmw_match.py` touch, additive: read the operand's element bit
   width off the MLIR type and stash it on the `MatchedOp`). The APU
   compose reads it and builds `OpCostCtx(..., dtype_bits=bits)`.
2. **ctx → cycles.** The APU op-cost lambdas become FUNCTIONS OF WIDTH,
   cloning the `micro25_add_cost` template (`spmw_cost_tables.py:795-825`):
   `cyc = gap_floor + seu_per_bit * dtype_bits`. Cite MICRO'25 Table 5
   (`mul_u16=201`, `add_u16=12`, `mul_f16=77`) as the width→cycles form;
   the APU lambdas mirror that table's shape, not a per-shape literal
   (report 27 §1.3 — cite Stripes/BitFusion + the APU table, claim only the
   threading into the fused objective).

### D3.2 Default-absent width → today's constant (the D3 anchor)

`OpCostCtx.dtype_bits=None` ⇒ the APU lambda returns today's constant
(`MUL=16`, `ADD=2`). Non-APU backends never thread width, so they keep flat
costs. faithful-for-APU keeps the constants too (D3 lands a NEW width-aware
flavor arm or threads only when a width is present — the cleaner option:
the faithful APU lambda is `lambda c: width_cost(c) if c.dtype_bits else
16`, so a width-absent faithful call is byte-identical). **The win:**
datatype/mixed-precision (s16→s8) becomes a first-class, cost-ranked lever
on a bit-serial machine — a precision change moves the APU estimate
monotonically (the D3 demo, task 007 + 009).

---

## Blast radius (answer 7)

### Files touched (all SPMW-local)

| File | Change | Additive vs invasive |
|---|---|---|
| `spmw_cost_model.py` | NEW: `Resource` enum, `Phase`, `phase_cycles`, `combine`, `AccessDescr`, `Knob` protocol, `Provenance` enum, `CalibrationRecord`; `provenance` field on `OpCost`/`MoveCost`; `calibration` on `CostModel`; enriched `OpCostCtx`/`MoveCostCtx` fields; `CostResult.phases: list[Phase]`; `phases_as_dict` shim; `evaluate` re-folds via `combine` | additive fields (defaulted) + ONE breaking type change (`phases` dict→list, with derived dict shim) |
| `spmw_cost_tables.py` | every `compose` emits `list[Phase]`; A5 deletes the 3 silent op fallbacks; provenance tags on every constant; calibration records; D2 locality phase + row-buffer constants; D3 APU width lambdas; A3 one-knob cost demo | mostly mechanical (carrier swap) + additive (tags, locality, width) |
| `spmw_cost_models.py` | `_whole_program_sum` → `combine`; `_whole_program_combiner` → `_COMBINER_FOR_FLAVOR`; `_kernel_cycles_factory` concatenates phases + folds via flavor-bound combiner | invasive to the combiner only (the faithful fold is byte-identical) |
| `spmw_tripcount.py` | none structural; the tier-3 policy is consumed in compose + the gate, not in tripcount | (likely no edit — flagged "possibly" in the task; the resolver is unchanged) |
| `spmw_match.py` | POSSIBLY: surface operand dtype bit-width on `MatchedOp` for D3 (additive read of the MLIR element type) | additive only, IF needed (D3/task 007 may derive it without touching match) |
| `spmw_linear_layout.py` | D2/task 006 ONLY: additive `conflict_count(...)` method (NOT touched by A2/task 004) | additive method, zero existing-caller change |
| `spmw_autoschedule.py` | the SINGLE opt-in `confidence_gate` param + post-argmin `_check_confidence` call; default off ⇒ inert | additive param, defaulted off; argmin scoring core UNTOUCHED |

### What is FORBIDDEN to touch (the FPGA/AIE guard)

**ZERO edits to:** `allo/ir/builder.py`, `allo/ir/infer.py`, any
`allo/ir/*`, `dataflow.py`, `customize.py`, `spmw_codegen.py`, and the
`spmw_autoschedule.py` argmin scoring core (lines 749-766: the
`scored.append(...); scored.sort(...); placements.append(scored[0][2])`
sequence). These carry the FPGA/AIE backends; the cost reshape is invisible
to them by construction (no upstream file imports `spmw_cost_*`).

### Upstream tests that gate the shared-file boundary

The cost modules (`spmw_cost_*`, `spmw_linear_layout`, the autoschedule
confidence param) are NOT imported by any upstream Allo path, so the FPGA
CI gate is: **the upstream `tests/dataflow/` and `tests/customize/` suites
must stay green** (they exercise `allo/ir/*`, `dataflow.py`, `customize.py`
— none of which this task touches). The coder runs them as the FPGA-blast
guard. There is no shared `.py` edit in the upstream sense — every file
this task touches is `spmw_*`, the SPMW-local namespace, which is exactly
why this is a low-blast-radius cycle. (Contrast: a change to
`allo/ir/builder.py` would gate on specific `tests/dataflow/` cases; this
task has none.)

### Rollback story if FPGA CI breaks

Two-layer rollback: (1) since no upstream file is touched, an FPGA CI break
would have to come from a transitive import — the rollback is to confirm
`spmw_cost_*` are not on the upstream import path (they are not) and revert
the offending SPMW file; (2) the *functional* rollback for any SPMW
regression is "select `faithful`" — it is the byte-for-byte anchor, and the
overlap/locality/width arms are opt-in flavors/terms that are provably inert
on the existing corpus (conflict-free→0, width-absent→constant,
sum-fold→today). The confidence-gate's rollback is `confidence_gate=False`
(the default).

### The two co-owned dataclasses (named, as the task requires)

1. **`AccessDescr`** (defined in `spmw_cost_model.py`) — consumed by the D2
   locality term HERE and the allocator's non-zero base access cost in
   `autosched-placement-realization` D2. ONE notion of access pattern.
2. **The typed knob registry / `Knob`** (the `cost(value, ctx) ->
   list[Phase]` side defined HERE; the `candidates()`/`emit()` side owned
   by `autosched-placement-realization` D4). ONE `Knob` shape, two
   tasks' methods on it. The cost task wires `compose += Σ knob.cost`; the
   placement task wires the enumerator cross-product + the codegen
   materializer.

---

## Decision log (the strongest commitments, for the orchestrator receipt)

- **`CostResult.phases` becomes `list[Phase(resource, latency, ii, count,
  tag)]`** (frozen). The dict shape survives ONLY as a read-only derived
  `phases_as_dict` shim for `RunResult.extra`.
- **`combine(phases, overlap)` replaces `_whole_program_sum`**; `faithful`
  binds `overlap=False` (sum-within / sum-across); `overlap` binds
  `overlap=True` (max-across independent resources + fill/drain). Every
  non-overlap flavor (optimistic, Mortise×3, demo) keeps the sum fold.
- **`faithful = (latency=ii=per_op, sum)` is the byte-for-byte anchor** —
  proof in A1.6; the report-18 numbers (15251 / 114992 / 457336, B\*=2,
  3.93×) and every per-op number are frozen.
- **`AccessDescr` and `Knob` are the two dataclasses co-owned with
  `autosched-placement-realization`** — `AccessDescr` in
  `spmw_cost_model.py`, the `Knob.cost(...)` side here, `candidates()`/
  `emit()` there.
- **A5 closes the vocabulary: unknown op/move = hard `KeyError`** (delete
  the 3 silent `per_op=4`/GPR/`add_cyc` fallbacks); tier-3 trip = declared
  default + coarse-confidence gate, never silent `=1`.
- **The opt-in `confidence_gate` is the SOLE autoschedule touch**, defaults
  off, argmin scoring core untouched.
- **ZERO edits to `allo/ir/*`, `dataflow.py`, `customize.py`,
  `spmw_codegen.py`, and the argmin scoring core.** No FPGA/AIE regression;
  upstream `tests/dataflow/` + `tests/customize/` stay green as the guard.

---

## Open design tensions (link to SPMW_ARCHITECTURE.md §4)

- **T21 (resolved-in-shape, not in number):** the v1-sequential vs
  v2-sum→max overlap tension is now structurally resolved — overlap is a
  property of the `Phase` timeline + the combiner, not a parallel flavor.
  T21's "trigger to revisit" (a corpus workload where over-count flips a
  rank decision) is now the D1 argmin-flip DEMO, constructed deliberately
  (task 005/009). T21 stays open as a *calibration* question (the overlap
  fold's absolute error), not a shape question.
- **T18 (host_staging overlap):** the seam this reshape was reserving is
  now realized as `Resource.HOST` phases + the `overlap` combiner. T18
  stays open because the v1 overlap fold keeps HOST serial vs device (the
  host must stage before the device runs — design 05 §9); a future
  measured host-bandwidth overlap is the refinement.
- **NEW T25 — provenance→uncertainty-band aggregation:** A4 lands the
  provenance TAGS and a confidence-gate that reads the existing
  `CostResult.confidence`. The symbolic band *derived from which tags
  participated* (report 28 §A4.3 new sub-claim) is flagged, not built —
  the v1 gate reads the coarse/placeholder flag. Revisit when a calibration
  task wants the optimistic/pessimistic flavors to be provenance-derived
  bands rather than hand-typed ×0.5 multipliers.
- **NEW T26 — UPMEM MRAM phase split for overlap:** A1 keeps UPMEM's MRAM
  cost folded into op cost (faithful frozen). The D1 UPMEM overlap arm
  needs the MRAM cost as a separate `Resource.DMA` phase; the split lives
  ONLY in the overlap path. Open: whether the split should be promoted into
  faithful (it would change the faithful number — so NO, until a measured
  UPMEM overlap workload ships).

---

## Implemented

- **Phase-1 keystone (task 004, coder)** — A1 + A2 + A5 + A4 fields landed.
  Files: `spmw_cost_model.py` (+`Resource`/`Phase`/`phase_cycles`/`combine`/
  `phases_as_dict`/`AccessDescr`/`Provenance`/`CalibrationRecord`/`Knob`
  protocol + knob-cost registry; `provenance` on `OpCost`/`MoveCost`;
  `calibration` on `CostModel`; enriched `OpCostCtx`/`MoveCostCtx`;
  `CostResult.phases: list[Phase]`; `evaluate` re-folds via `combine` and
  surfaces the dict view at the boundary); `spmw_cost_tables.py` (every
  `compose` emits `list[Phase]`; A5 deletes the 3 silent op fallbacks
  (AiM/UPMEM/APU v1); provenance tags + calibration records on the
  sim-anchored set; the `stage_resident` knob-cost demo);
  `spmw_cost_models.py` (`_whole_program_sum`→`_COMBINER_FOR_FLAVOR` +
  `combine`); `spmw_autoschedule.py` (the SOLE touch: opt-in
  `confidence_gate=False` param + post-argmin `_check_confidence`, argmin
  scoring core lines 758-773 untouched). New test
  `tests/spmw/test_cost_phase_timeline.py` (97-test reshape acid set +
  migrated tripcount/mortise carrier reads). Test command:
  `pytest tests/spmw/test_cost_phase_timeline.py tests/spmw/test_tripcount_resolution.py tests/spmw/test_mortise_capacity.py tests/spmw/test_mortise_sweep.py tests/spmw/test_virtual_backend.py tests/spmw/test_cost_model_swap.py tests/spmw/test_apu_v1_vr_dma_cost.py`
  → 97 passed. Expected outcome: faithful fold byte-identical (combine over
  every compose's phases == its old `.cycles`); 15251 / B\*=2 / 3.93x via
  `test_samsung_batched_gemv.py` (pure-cost, no sim). One mechanical
  judgment call flagged below.
- **Spec ambiguity flagged (mechanical resolution, A1.2):** `spmw_codegen.py`
  is FORBIDDEN to edit yet is the one structural reader of the old dict shape
  (`:3009 "phases": result.phases`), and `test_virtual_backend.py:90`
  requires `extra["phases"]` to carry dict keys (`preload`/`exec`/`readback`).
  Resolution: `evaluate` (in-scope, the surfacing boundary) applies the
  `phases_as_dict` shim and returns dict-shaped `.phases`, while every
  `compose(...)` returns the structural `list[Phase]`. This honors "the shim
  is the only place the dict survives" without touching codegen. The
  existing dict-reader tests (`test_tripcount_resolution.py`,
  `test_mortise_capacity.py`) call `compose(...)` directly, so their
  carrier-shape reads were migrated to `phases_as_dict`/tag presence
  (numbers preserved byte-identical). Architect: confirm this boundary
  placement of the shim is the intended A1.2 mechanism.
