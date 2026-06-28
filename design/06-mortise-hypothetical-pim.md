# Design 06 — Mortise hypothetical PIM substrate: structure-only target, capacity-aware CostModel flavors, sweep + ablation seams

Status: Phase-0b architect spec (GATING). Authoritative for tasks 003–007 in
`dev/06282026-mortise-hypothetical-pim/`. Built on the design-04 virtual-backend
machinery (LANDED: `spmw_cost_model.py` + `spmw_cost_tables.py` +
`compile_for_target(..., backend="virtual")`) and the design-05 `host_staging`
concern (LANDED: `tenon@7738fee`). Consumes the Phase-0 committed lever +
prediction in `experiments/reports/26-mortise-weight-resident-capacity-lever.md`.
Author: architect, 2026-06-28.

This doc is a **spec the coder implements verbatim**, not an implementation. It
answers the four Phase-0b questions the task gates on:
1. the structure-only `@allo.target("mortise")` tree (no cost numbers);
2. the `CostModel` flavors (baseline + ablation + swept), every constant
   provenance-tagged;
3. the sweep + ablation seams (zero source edit for the verifier);
4. the blast radius (zero FPGA/AIE; named shared-file touches).

The Phase-0 report has already done the science (the lever, the algebra, the
falsifiable prediction, the §2.4 table, the sensitivity band, the anchor). This
doc's job is to pin the *mechanism* — exactly which file gets which entry, which
existing seam each piece reuses, and the acid tests that keep it honest.

---

## 0. The one load-bearing decision

The capacity-aware cost model from report-26 §2.1 is:

```
mortise_batched(B, C) = P + B*(E + R + (1 - phi)*P_var)        phi = min(1, C/T_w)
baseline(B)           = B*(P + E + R)                          (re-preload-every-vector)
```

The `(1-phi)*P_var` term is a **per-vector eviction re-stream**: when capacity
`C` cannot hold the full weight tile `T_w = M*K`, the evicted fraction must be
re-broadcast on every batch vector. That is *staging*, not device-exec. So:

> **DECISION (load-bearing): the capacity lever lives entirely in the Mortise
> `host_staging` concern compose, not in `kernel_cycles`, not on the target
> tree as a cost number.** The `kernel_cycles` concern (device exec `E`) is
> shape/placement-only and identical to Samsung's. The capacity term modulates
> the `host_staging` preload phase: resident pays `P` once; the evicted
> shortfall `(1-phi)*P_var` is added to the *per-call* preload bucket.

This is the cleanest seam because the design-05 `host_staging` compose
*already* splits preload into `stage_resident` (paid once) vs `stage_per_call`
(paid B times) keyed off `layout.extra["stage_resident"]`. Mortise's compose is
that exact function plus one extra additive per-call term `B*(1-phi)*P_var`. The
report-26 ablation (`C=inf` ⇒ `phi=1` ⇒ term vanishes ⇒ collapses to the
report-18 `P + B*(E+R)`) is then *algebraically forced* and trivially testable.

Rejected alternatives:
- *Capacity term in `kernel_cycles`*: would entangle exec with staging and break
  the design-05 invariant that `kernel_cycles` is the device-exec phase.
- *Capacity as a `Move.cycles` cost number on the target tree*: violates the
  design-04 acid test (the tree carries zero cost numbers). `C` is a **structural
  geometry constant** (how big the resident region is — "how many elements"), so
  it is the one capacity number allowed on the tree, exactly like
  `lanes_const`/`elem_bits_const` on `demo_pim`. The *cycles* of re-streaming the
  shortfall live in the CostModel.

---

## 1. Answer 1 — the structure-only target tree

New fixture: **`experiments/allo/tests/spmw/_mortise_target.py`**, cloning the
`_demo_target.py` pattern verbatim. Mortise is a Samsung-like near-bank-SIMD
substrate (report-26 §1.2): the *only* structural addition over a Samsung-shaped
tree is the new geometry constant `resident_cap_elems` (the capacity `C`).

```python
# experiments/allo/tests/spmw/_mortise_target.py
from __future__ import annotations
import allo

# Largest weight tile in the swept corpus = M*K at 4096x1024 = 4,194,304 u16.
# The DEFAULT capacity const equals one full tile (phi=1 at the headline shape):
# this is geometry (how many resident elements), NOT a cost number.
_T_W_FULL = 4096 * 1024   # 4,194,304

def build_mortise_target(resident_cap_elems: int = _T_W_FULL):
    """Return the `mortise` structural target (no sim/HW backend).

    `resident_cap_elems` is the swept lever `C` (report-26 §1.2): the size of
    the on-device weight-resident region in u16 weight elements. It is a
    STRUCTURAL geometry constant — read by the Mortise host_staging compose to
    compute phi = min(1, C/T_w). Default = one full tile (phi=1, the Samsung
    physical analog). The sweep harness rebuilds the target with a different C
    (the sweep seam, §3.1).
    """

    @allo.target("mortise")
    def device():
        # --- geometry constants (structure, read by the compose) ---
        allo.const("resident_cap_elems", resident_cap_elems)  # the lever C
        allo.const("lanes_const", 16)        # near-bank SIMD lanes (analog of
                                             # Samsung num_grfA folded width)
        allo.const("elem_bits_const", 16)    # u16 weight/activation elements

        mem = allo.mem(size_bytes=1 << 24, name="dram")

        @allo.unit(mapping=[16])             # 16 near-bank compute tiles
        def tile():
            vr = allo.reg(16, 16, name="vr")

            # Moves: structure + emit only (no codegen for this substrate;
            # emit=None is a complete tree, exactly like demo_pim).
            allo.move("LD_A", src=mem, dst=vr, emit=lambda ctx: None)
            allo.move("LD_B", src=mem, dst=vr, emit=lambda ctx: None)
            allo.move("ST_A", src=vr, dst=mem, emit=lambda ctx: None)

            any_vr = allo.any_([vr])
            allo.op("ADD", src=(any_vr, any_vr), dst=any_vr,
                    fn=lambda x, y: x + y,
                    emit=lambda x, y, dst, ctx: None)
            allo.op("MUL", src=(any_vr, any_vr), dst=any_vr,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: None)
            allo.op("MAC", src=(any_vr, any_vr), dst=any_vr,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: None)

    return device
```

**Acid test (design-04 §1.2, the Phase-0b PASS bar):** every `Move`/`Op` on the
tree has `emit=None` and **no `cycles=`**. `mv.cycles is None` and
`op.cycles is None` for every node — confirmed by construction (the `allo.move`
/ `allo.op` calls pass no `cycles=`). The only numbers on the tree are
*geometry* (`resident_cap_elems`, `lanes_const`, `elem_bits_const`, `mapping`,
register width/count, mem size) — "how many / how wide", never "how many cycles".

> The coder MUST add a test asserting the acid test for the Mortise tree
> (mirror of the existing demo acid test in `test_target_cycles.py`): walk every
> unit, assert all `move.cycles`/`op.cycles` are `None`. This is the structural
> half of the Phase-0b receipt.

There is **no `_BACKEND_CTX` / `_BACKEND_RUN` entry for `mortise`** beyond the
generic `"virtual"` adapter, and **no simulator** — Mortise is priced purely via
`compile_for_target(target, trace, backend="virtual")`, identical to `demo_pim`.

---

## 2. Answer 2 — the CostModel flavors (numbers, all in `spmw_cost_tables.py`)

Three flavors, registered at import on the existing `(target_name, flavor,
concern)` registry. The `kernel_cycles` concern is shared across all three; the
flavors differ only in the `host_staging` concern's treatment of `phi`.

### 2.0 What each flavor is

| flavor | concern: kernel_cycles | concern: host_staging | role |
|---|---|---|---|
| `mortise_faithful` (`flavor="faithful"`) | device exec `E` (Samsung phase algebra) | capacity-aware: `P + B*(R + (1-phi)*P_var)` resident / `B*(P+E... )` baseline | **baseline finding** |
| `mortise_unlimited` (`flavor="unlimited"`) | same | hard-wires `phi=1` (capacity term dropped) | **the ABLATION** (report-26 §3) |
| `mortise_optimistic` (`flavor="optimistic"`) | same | same compose, `P_var`/`E`/`R` scaled per §4 band | **the swept sensitivity variant** (report-26 §4) |

The **swap surface** is the design-04/`demo_pim` one verbatim: each is a
distinct `CostModel` registered under the same `(target_name="mortise")` but a
different `flavor`. The verifier picks a flavor via
`compile_for_target(target, trace, backend="virtual", cost_flavor="unlimited")`
— zero device-tree edit, zero source edit (§3.2). The faithful↔unlimited swap
is the report-26 §3 ablation; the faithful↔optimistic swap is the report-26 §4
sensitivity band; together they are the design-04 "two cost models, one target,
zero device edit" swap test.

### 2.1 The kernel_cycles concern (shared, Samsung-cloned)

Mortise's device exec is Samsung's folded-GEMV exec, so the `kernel_cycles`
compose reuses the Samsung phase algebra. The coder registers:

```python
MORTISE_FAITHFUL = register_cost_model(CostModel(
    name="mortise_faithful",
    target_name="mortise",
    flavor="faithful",
    concern="kernel_cycles",
    op_costs={
        "MAC": OpCost(lambda c: 4, note="tCCDL column-strobe; Samsung-analog, "
                      "report-18 E folded MAC=(K//8)*4 [sim-anchored]"),
        "MUL": OpCost(lambda c: 4, note="tCCDL [sim-anchored, Samsung-analog]"),
    },
    move_costs={
        "LD_A": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2 [sim-anchored]"),
        "LD_B": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2 [sim-anchored]"),
        "ST_A": MoveCost(lambda c: 14, note="tCCDL+WL+BL//2 [sim-anchored]"),
    },
    constants={},
    compose=_mortise_kernel_compose,   # = Samsung exec phase, see below
))
```

`_mortise_kernel_compose` is `_samsung_compose_with(MORTISE_FAITHFUL, ctx)` with
the preload/readback host-staging terms removed (they live in the host_staging
concern). In practice the cleanest implementation is: **reuse the Samsung
`kernel_cycles` compose body** parameterised by the bound model, exactly as
`_samsung_compose_with` is already parameterised. The coder may import and call
the Samsung exec helper; Mortise's exec IS Samsung's exec by construction
(report-26 §1.2: Mortise grafts onto the Samsung near-bank-SIMD substrate). The
exec result `E` must equal report-18's `E=3702` at 4096×1024 (the anchor in §5).

### 2.2 The host_staging concern (faithful) — the capacity lever

This is the one genuinely new compose. It is `_samsung_host_staging_compose_with`
plus the capacity term. Spelled out:

```python
def _mortise_host_staging_compose_with(hs_model, ctx):
    target, trace, layout = ctx.target, ctx.trace, ctx.layout
    M, K = _samsung_mk(target, trace)            # reuse Samsung shape recovery
    B = _trace_batch_dim(trace)                  # batched dim, default 1
    T_w = M * K                                  # weight-tile size (elements)

    # --- the capacity lever ---
    C = getattr(target, "resident_cap_elems", T_w)   # read off the TREE const
    phi = 1.0 if hs_model.unlimited else min(1.0, (C / T_w) if T_w else 1.0)

    preload_cyc = _samsung_preload_cycles(hs_model, M, K)    # = P
    readback_cyc = _samsung_readback_cycles(hs_model, M)     # = R per vector
    crf_cyc = hs_model.move_cost("STAGE_CRF", MoveCostCtx("STAGE_CRF"))
    P_var = preload_cyc - crf_cyc                # data-proportional part (§2.1)
    evict_per_call = int(round((1.0 - phi) * P_var))   # the lever's per-call cost

    resident = bool(getattr(layout, "extra", {}).get("stage_resident", False))
    if resident:
        stage_resident = preload_cyc            # P paid ONCE
        stage_per_call = B * evict_per_call      # only the evicted shortfall/vec
    else:
        stage_resident = 0
        stage_per_call = B * preload_cyc         # re-preload-every-vector baseline
    readback_total = B * readback_cyc
    cycles = stage_resident + stage_per_call + readback_total
    return CostResult(
        cycles=cycles,
        phases={"stage_resident": stage_resident,
                "stage_per_call": stage_per_call,
                "evict_per_call": B * evict_per_call,    # surfaced for the sweep
                "readback": readback_total},
        confidence=hs_model.confidence,
    )
```

Reductions that MUST hold (the algebra checks, report-26 §2.1):
- `phi=1` (C ≥ T_w) AND `resident=True`: `cycles = P + B*R` for the
  host_staging concern; whole-program (`+ kernel_cycles E`) =
  `P + B*(E+R)` = report-18 resident form. Ceiling 3.93×, B*=2.
- `phi=1`, `resident=False`: `cycles = B*(P+R)`; whole = `B*(P+E+R)` =
  the re-preload baseline. **This is the comparator** that isolates the lever.
- `phi=0.5`, `resident=True`: per-call adds `0.5*P_var ≈ 5683`; whole-program
  ceiling drops to `(P+E+R)/(E+R+0.5*P_var) = 15251/9566 = 1.594×` (report-26
  §2.3, §2.4 row 0.5).

The `hs_model.unlimited` flag is the ablation toggle (§2.4). The
`mortise_optimistic` flavor reuses this exact compose; only its move_cost
numbers (`STAGE_BCAST`, `GATHER_RD`, …) differ per §4.

### 2.3 The host_staging move_cost numbers (faithful) — provenance-tagged

Re-homed VERBATIM from the landed `SAMSUNG_HOST_STAGING` (report-18 calibration
anchors). Every constant carries a provenance tag; **none is a free fit param**
(Phase-0b PASS bar; report-26 §4 provenance table):

```python
MORTISE_HOST_STAGING = register_cost_model(CostModel(
    name="mortise_host_staging",
    target_name="mortise",
    flavor="faithful",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_BCAST":   MoveCost(lambda c: 369,  note="[sim-anchored] HAB "
            "preload fan-out width; report-18 §1 P=11368 measured on "
            "PIMSimulator; Mortise inherits Samsung HAB bcast rate "
            "(spmw_cost_tables STAGE_BCAST=369). Uncertainty: Mortise bcast "
            "width may differ -> swept 0.5x-2x in optimistic flavor."),
        "STAGE_SCATTER": MoveCost(lambda c: 1,    note="[sim-anchored] "
            "per-group column-strobe; report-18 P decomposition."),
        "STAGE_CRF":     MoveCost(lambda c: 2,    note="[assumption] programCrf "
            "upload; analog of Samsung STAGE_CRF=2, capped 4-burst per "
            "report-18 §1; swept 1-8 in optimistic flavor."),
        "GATHER_FAN":    MoveCost(lambda c: 4096, note="[sim-anchored] readback "
            "tile width; report-18 R=181 @ M=4096."),
        "GATHER_RD":     MoveCost(lambda c: 181,  note="[sim-anchored] per-tile "
            "readResult; report-18 §1; GATHER_RD=181."),
    },
    constants={},
    compose=_mortise_host_staging_compose,   # binds MORTISE_HOST_STAGING
    unlimited=False,
))
```

Provenance summary (mirrors report-26 §4 — verifier's grep guard checks every
`note=` contains one of `[sim-anchored]` / `[assumption]` / `[datasheet]` /
`[structural]`):

| constant | value | provenance | citation |
|---|---:|---|---|
| `STAGE_BCAST` (P_var fan-out) | 369 | sim-anchored | report-18 §1 (P measured on PIMSimulator) |
| `STAGE_SCATTER` | 1 | sim-anchored | report-18 P decomposition |
| `STAGE_CRF` (crf_cyc) | 2 | assumption | Samsung STAGE_CRF analog; report-18 §1 |
| `GATHER_FAN` | 4096 | sim-anchored | report-18 R |
| `GATHER_RD` (R) | 181 | sim-anchored | report-18 §1 |
| MAC/MUL (E) | 4 | sim-anchored | tCCDL; report-18 E folded MAC=(K//8)*4 |
| `resident_cap_elems` (C) | swept | structural | the lever; report-26 §1.2 (not a fit param) |

### 2.4 The ablation flavor `mortise_unlimited` (report-26 §3)

The named ablation. **DECISION: the ablation is a CostModel-swap, implemented as
a one-field flag `unlimited=True` on the host_staging CostModel that forces
`phi=1`** — the report-26 §3 "two equivalent toggles" choice (b). It is the
strongest because, per report-26 §3, the only `C`-dependent term is
`(1-phi)*P_var`; with `phi=1` it is algebraically zero for every `C`, so every
capacity arm produces the identical report-18 curve and the spread vanishes.

```python
MORTISE_UNLIMITED_KERNEL = register_cost_model(replace(
    MORTISE_FAITHFUL, name="mortise_unlimited", flavor="unlimited"))
MORTISE_UNLIMITED_HOST_STAGING = register_cost_model(CostModel(
    name="mortise_unlimited_host_staging",
    target_name="mortise", flavor="unlimited", concern="host_staging",
    op_costs={}, move_costs=MORTISE_HOST_STAGING.move_costs,  # SAME numbers
    constants={}, compose=_mortise_unlimited_host_staging_compose,
    unlimited=True,    # <-- the only difference: phi forced to 1
))
```

> Implementation note: `unlimited` is a new bool field on the `CostModel`
> dataclass **iff** it is the cleanest carrier. The dataclass is spmw-local
> (design-04), so adding `unlimited: bool = False` is additive and FPGA-safe.
> ALTERNATIVE the coder may prefer (avoids touching the dataclass): bake the
> flag into the bound compose closure (`_mortise_host_staging_compose` vs
> `_mortise_unlimited_host_staging_compose` set `phi=1`), reading nothing off
> the model. **Coder's choice; both are zero-FPGA-blast.** The spec requires
> only that the unlimited flavor force `phi=1` and reuse the faithful numbers.

The ablation acid test (report-26 §3, the Phase-3 finding-validity gate): run
the §2.4 capacity sweep twice — `cost_flavor="faithful"` (the report-26 §2.4
spread appears) and `cost_flavor="unlimited"` (EVERY `C` row collapses to the
`phi=1` row: 2.875× @ B=8 / 3.598× @ B=32 regardless of `C`). Any surviving
`C`-dependent spread under unlimited ⇒ FAIL `finding-not-feature-attributable`.

---

## 3. Answer 3 — the sweep + ablation seams (zero source edit for the verifier)

### 3.1 The sweep seam — where the harness reads the swept `C`

New harness: **`experiments/scripts/R26_mortise_capacity_sweep.py`** (the
`RNN_<slug>` convention; siblings: `R17_*`, `R19_*`, `R201_*`). The swept
parameter `C` is read off the **target const** `resident_cap_elems`. The harness
sweeps by **rebuilding the target** with a different `C` (the
`build_mortise_target(resident_cap_elems=...)` kwarg, §1) — NOT by editing any
source file:

```python
# experiments/scripts/R26_mortise_capacity_sweep.py (sketch — coder fills in)
from tests.spmw._mortise_target import build_mortise_target, _T_W_FULL
# ... build the batched-GEMV trace at 4096x1024 (corpus shape) ...
for phi in (1.0, 0.75, 0.5, 0.342, 0.1, 0.0):       # report-26 §2.4 axis
    C = int(phi * _T_W_FULL)
    target = build_mortise_target(resident_cap_elems=C)
    for B in (1, 2, 8, 32, 128):
        # argmin-select stage_resident via the autoscheduler (NOT hand-set):
        compiled = compile_for_target(target, trace_at(B), backend="virtual")
        resident = compiled.run()                    # winner (resident arm)
        baseline = run_baseline_arm(target, B)       # stage_resident=False arm
        # record speedup = baseline.cycles / resident.cycles
```

The sweep over `C` is the harness loop; the sweep over `B` rides the trace's
`batch_dim` (already carried by `MatchedOp.extra["batch_dim"]`, SPEC-026). The
harness writes the raw §2.4 table under `experiments/scripts/` (committed
artifact, Phase-2). **Zero source edit** — `C` is a constructor kwarg, `B` is a
trace property.

> The §4 sensitivity sweep (the 0.5×–2× band on `P_var`/`E`/`R`, `crf` 1–8) is
> the SAME harness with `cost_flavor="optimistic"` (and, for the full band, the
> harness may register additional optimistic-band flavors at runtime via
> `register_cost_model` — a documented, source-free use of the swap surface,
> exactly how the demo swap test builds its second flavor). The qualitative
> finding (report-26 §4) must survive the whole band.

### 3.2 The ablation seam — the CostModel swap (zero device-tree edit)

The ablation toggles entirely via `cost_flavor`:

```python
faithful = compile_for_target(t, trace, backend="virtual", cost_flavor="faithful")
ablated  = compile_for_target(t, trace, backend="virtual", cost_flavor="unlimited")
```

Same `target` object, same trace, same harness — only the flavor string changes.
This is the design-04 swap surface verbatim (proven in `test_phase5_demo.py` /
`test_cost_model_swap.py`). The verifier runs both with **zero source edit**;
the finding-validity gate (Phase-3) asserts: faithful shows the §2.4 spread,
unlimited erases it.

### 3.3 Argmin-selected, not hand-set (Phase-1 requirement)

The within-substrate comparison is `stage_resident=True` (resident) vs
`stage_resident=False` (re-preload baseline). To satisfy "layouts must be
argmin-selected via the existing autoscheduler (≥2 candidates), not hand-set,"
Mortise needs an enumerator that emits both candidates:

> **New entry: `@register_enumerator("mortise")` in `spmw_autoschedule.py`,
> emitting the `stage_resident ∈ {False, True}` candidate pair.** This is an
> additive clone of the Samsung enumerator's `_with_stage_resident(cand, False)`
> / `_with_stage_resident(cand, True)` tail (lines 358–359). It registers a new
> dict key in `_ENUMERATORS`; it changes **no existing enumerator** and is not
> read by any FPGA path (§4). The autoscheduler's argmin then *picks* resident
> at B≥2 (because the Mortise faithful cost prices it cheaper), so the harness
> reports an argmin-selected winner, not a hand-set flag.

The "baseline arm" (`stage_resident=False`) for the speedup ratio is obtained by
forcing that candidate (the harness may select it explicitly for the comparator
denominator — the baseline is by definition the non-resident schedule, report-26
§2.1; this is the comparator, not the optimized result). The *finding* (Tenon
picks resident) is the argmin output; the *ratio* is winner-over-baseline.

---

## 4. Answer 4 — blast radius

### 4.1 Files touched

| File | Edit | Shared? | Justification |
|---|---|---|---|
| `tests/spmw/_mortise_target.py` | **NEW** | no (test fixture) | structure-only Mortise tree; clone of `_demo_target.py` + the `resident_cap_elems` geometry const. No cost numbers. |
| `allo/spmw_cost_tables.py` | edited, **additive** | no (spmw-local, side-effect registration) | three new `CostModel`s (`mortise_faithful`, `mortise_unlimited`, `mortise_optimistic`) × two concerns (`kernel_cycles`, `host_staging`) + the `_mortise_*_compose` fns. New registry keys only; no existing model changed. |
| `allo/spmw_cost_model.py` | edited, **additive** *(only if the coder picks the dataclass-flag ablation, §2.4)* | no (spmw-local) | OPTIONAL `unlimited: bool = False` field on `CostModel`. Additive default; alternative is closure-baked flag (no edit). |
| `allo/spmw_autoschedule.py` | edited, **additive** | no (spmw-local; not imported by any FPGA test) | `@register_enumerator("mortise")` emitting `stage_resident ∈ {F,T}`. New `_ENUMERATORS` key; clone of Samsung tail. |
| `experiments/scripts/R26_mortise_capacity_sweep.py` | **NEW** | no | the sweep + ablation + sensitivity harness (Phase-2). |
| `tests/spmw/test_mortise_*.py` | **NEW** | no | acid test (tree carries no cost number), swap test (faithful↔unlimited differ), capacity-collapse test (§2.4 reductions), ablation test (unlimited flattens all-C). |
| `experiments/allo/SPMW_ARCHITECTURE.md` | edited (architect) | doc | new Mortise module-map row + a Mortise seam subsection + open tension. |

### 4.2 Why zero FPGA/AIE regression

- **No `allo/ir/*`, `allo/dataflow.py`, `allo/customize.py`, `allo/memory.py`,
  `allo/__init__.py` edit.** The FPGA/AIE path reaches Allo through
  `allo.customize` / `allo.LLVMModule` / `allo.HLSModule` / `dataflow.build()`.
  None is touched.
- `spmw_cost_tables.py`, `spmw_cost_model.py`, `spmw_autoschedule.py` are
  Tenon-only modules. Verified: `grep -rln spmw_autoschedule allo/` returns only
  spmw files + the (already-landed) `__init__.py` re-export — no upstream test
  imports them. The edits are purely additive (new registry/dict keys, an
  optional defaulted field), so existing-semantics change is zero.
- The new fixture cannot reach `tests/dataflow/` or `tests/customize/` — those
  build different targets through different fixtures.

### 4.3 Upstream tests that gate the (spmw-local) edits

`spmw_cost_tables.py` / `spmw_autoschedule.py` / `spmw_cost_model.py` are not
imported by any upstream test. The guards are the **SPMW** suite:
- `tests/spmw/test_autoschedule.py`, `test_codegen_gemv.py`, `test_match_gemv.py`
  — baseline three; argmin must stay byte-identical (Mortise adds keys, never
  perturbs existing models).
- `tests/spmw/test_cost_model_swap.py`, `test_target_cycles.py` (acid test),
  the report-18 invariants (15251, B*=2, 3.93×) — the Mortise faithful arm
  reproduces these by construction (§2.2 reductions; the anchor is §5).
- `tests/spmw/test_rankpreserve_vs_sim.py::test_samsung_rank_preservation_weight_residency`
  — the real-substrate anchor (§5); unchanged, must stay green.

The FPGA/AIE CI suites (`test_vhls`, `test_vitis`, `test_xls`,
`test_catapult_hls`, `test_pynq`, `test_nn`, `tests/dataflow/`,
`tests/customize/`) are **untouched and stay green** — they import nothing this
design changes.

### 4.4 Rollback story

Every edit is additive or test-local. Rollback = `git revert` of the new fixture
+ harness + tests + the additive `spmw_cost_tables.py` / `spmw_autoschedule.py`
keys (and the optional `unlimited` field). Because Mortise only *adds* registry
keys keyed on `target_name="mortise"`, no existing `(target,flavor,concern)`
lookup changes, so the report-18 numbers and every existing argmin are provably
unmoved even under a partial landing. The FPGA path never sees any of it.

---

## 5. The real-substrate anchor (report-26 §5, Phase-4 / tasks 005–006)

The credibility bridge is **already live**:
`test_samsung_rank_preservation_weight_residency` (B ∈ {2,4}) proves
virtual-backend argmin == real PIMSimulator argmin on the *physical* limit of
the Mortise lever (`C = T_w`, full residency = Samsung). Mortise's
`mortise_faithful` host_staging compose is **numerically identical** to the
Samsung resident schedule at `phi=1` (§2.2 first reduction) — same
`_samsung_preload_cycles` / `_samsung_readback_cycles`, same `P + B*(E+R)`. So
the anchor holds: the method that prices Mortise is the method already validated
against ramulator-class ground truth at the `phi=1` corner.

> **Anchor acid test the verifier must add (task 006):** assert that
> `mortise_faithful` whole-program at `C ≥ T_w`, `stage_resident=True`,
> 4096×1024, B ∈ {2,4} equals the Samsung faithful whole-program for the same
> shape/B (both = `P + B*(E+R)` = report-18). If they diverge, Mortise's exec or
> staging re-home is wrong, not Samsung. This makes the bridge a *numerical*
> identity, not an analogy (report-26 §5).

Mortise then extends that exact method to `C < T_w` — a regime the
simulator-validated method already covers structurally (the `stage_resident`
flag is the `phi=1` vs `phi<1` switch; the eviction term is its smooth
generalization).

---

## 6. Concrete commitments (the receipts)

1. **The capacity lever lives ONLY in the Mortise `host_staging` compose** as
   `B*(1-phi)*P_var`, `phi = min(1, resident_cap_elems / (M*K))`. Not in
   `kernel_cycles`, not as a `cycles=` on the tree. The tree carries `C` as a
   *geometry* const (`resident_cap_elems`); `mv.cycles`/`op.cycles` are `None`
   for every node (acid test).
2. **Three flavors on `(target_name="mortise")`**: `faithful` (baseline finding),
   `unlimited` (ablation, `phi≡1`), `optimistic` (§4 sensitivity band). Swap via
   `cost_flavor=` — zero device-tree edit, the design-04 surface verbatim.
3. **Every cost constant provenance-tagged** in its `note=` with one of
   `[sim-anchored]`/`[assumption]`/`[structural]` (§2.3 table); `C` is the swept
   structural lever, not a fit param.
4. **Sweep seam = `build_mortise_target(resident_cap_elems=C)` constructor kwarg
   + the trace's `batch_dim`**, driven by `experiments/scripts/R26_*.py`. Ablation
   seam = `cost_flavor="faithful"` vs `"unlimited"`. Both zero source edit.
5. **Argmin-selected**: new `@register_enumerator("mortise")` emits
   `stage_resident ∈ {F,T}`; the autoscheduler picks resident at B≥2.
6. **Zero FPGA/AIE blast radius**: new fixture + additive `spmw_cost_tables.py` /
   `spmw_autoschedule.py` registry keys (+ optional additive `CostModel.unlimited`
   field) + new harness + new tests. No `allo/ir/*` / `dataflow.py` /
   `customize.py` / `__init__.py` edit.
7. **Anchor = numerical identity, not analogy**: `mortise_faithful` at
   `C≥T_w`/resident == Samsung faithful (`P+B(E+R)`, report-18) at 4096×1024,
   B∈{2,4}; rides the live `test_samsung_rank_preservation_weight_residency`.

---

## 7. Open tensions handed forward

- **T23 (new, Mortise):** smooth-`phi` vs `ceil(T_w/C)`-quantized eviction
  (report-26 §7 Q2). The §2.2 compose uses the smooth `(1-phi)*P_var`; real
  eviction is tile-quantized. The harness SHOULD report both curves (smooth +
  ceil-quantized) so the finding statement can pick the conservative one. This
  refines the *number* at fractional `C`, not the qualitative collapse. Trigger
  to commit a quantized compose: a corpus shape where the coarse-`C`
  quantization flips a crossover-B decision.
- **T24 (new, Mortise):** the headline 3.93× is Samsung-preload/exec-ratio
  specific (report-26 §7 Q1, §4). A Mortise with a genuinely different MAC speed
  moves the *number* (a 2×-faster exec → ~2.6× ceiling), not the ordering. The
  `optimistic` flavor's `E`/`P_var` band quantifies this; calibrating Mortise's
  own `tCCDL` against a different analog datasheet (AiM 2 GHz) is a v2 nicety,
  out of scope.
- **T18 (inherited, design-05):** the whole-program combiner is `sum`
  (no preload↔exec overlap). report-26 §7 Q3: a substrate overlapping the
  evicted-tile re-stream with the next vector's exec would soften the collapse.
  The `host_staging` compose already returns a per-phase breakdown
  (`evict_per_call` is surfaced separately) so a future `max`-combiner is a
  one-line change. The finding is stated as "in the no-overlap model."

---

## 8. Implemented (task 003, coder, 2026-06-28)

- `tests/spmw/_mortise_target.py` (NEW): structure-only `@allo.target("mortise")`,
  built SAMSUNG-SHAPED (nested units, grf_a/grf_b, same moves/ops, `emit=None`)
  plus the `resident_cap_elems` lever const + `lanes_const`/`elem_bits_const`.
  Acid test holds (all `mv.cycles`/`op.cycles` None). [DEVIATION from §1 sketch:
  the §1 code block shows a simplified `mapping=[16]`/`vr` tree, but that tree
  cannot reuse the Samsung exec compose (no grf_a/grf_b, n_workids=16≠128) and
  would break the §5/§6.7 numerical-identity anchor. I built Samsung-shaped per
  the load-bearing §0/§5 requirement; the §1 sketch was illustrative. Confirmed:
  anchor identity holds byte-identical at B∈{2,4}.]
- `allo/spmw_cost_tables.py` (+~215 lines, additive): `MORTISE_FAITHFUL` /
  `MORTISE_UNLIMITED` / `MORTISE_OPTIMISTIC` (kernel_cycles, delegate to
  `_samsung_compose_with`) + `MORTISE_HOST_STAGING` /
  `MORTISE_UNLIMITED_HOST_STAGING` / `MORTISE_OPTIMISTIC_HOST_STAGING`
  (host_staging, the capacity compose). `unlimited` carried as the CLOSURE-BAKED
  flag (§2.4 alternative), so `spmw_cost_model.py` is UNTOUCHED. Every cost
  `note=` carries a provenance tag.
- `allo/spmw_autoschedule.py` (+~20 lines, additive): `@register_enumerator(
  "mortise")` delegates to `_samsung_enumerate` (Mortise tree IS Samsung-shaped),
  emitting the `stage_resident ∈ {F,T}` pair.
- `experiments/scripts/R26_mortise_capacity_sweep.py` (NEW): the sweep
  (constructor-kwarg `C`) + ablation (`cost_flavor` swap) + sensitivity
  (optimistic) harness; writes `R26_mortise_capacity_sweep.out.txt`.
- `tests/spmw/test_mortise_capacity.py` (NEW): 22 static tests — acid, swap,
  §2.2 reductions, ablation, argmin-selection, provenance, anchor identity.

NON-OBVIOUS PATH FLAGGED FOR ARCHITECT: the autoscheduler argmin for Mortise
requires `SPMW_DISABLE_REGALLOC=1`. `spmw_regalloc.py` (NOT in the §4.1
authorized file list) has a target-keyed capacity table with no `mortise`
entry, so `allocate` raises for every Mortise candidate. Disabling regalloc is
correct AND argmin-preserving: the `stage_resident` flag rides `extra` (not
register placements), so spill `total_cost` is identical across the candidate
pair — disabling it cannot change which arm wins. The harness + the
argmin test set the flag locally. If the architect prefers a registered
Mortise capacity builder instead, that is a `spmw_regalloc.py` edit needing
sign-off; the kill-switch keeps blast radius at the authorized files.

Receipt: `tests/spmw/test_mortise_capacity.py` 22 passed; swap test
faithful≠unlimited and anchor `mortise==samsung` byte-identical at B∈{2,4};
R26 sweep shows monotone collapse (phi=1 ceiling vs phi=0.5 <2x) + unlimited
flattens all-C resident-arm spread = True.
