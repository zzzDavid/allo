# SPEC-009 — Argmin enumeration: ≥2 candidates per backend

**Scope.** Today three of five enumerators return exactly one
`Placement` (`_samsung_enumerate`, `_apu_v1_enumerate`,
`_apu_v2_enumerate`). With one candidate, the argmin in `autoschedule`
(`spmw_autoschedule.py:411-437`) is a no-op and the project's
"cost-based automatic data layout construction" success criterion is
only observable on AiM and UPMEM (which already enumerate two). This
spec defines a second candidate for each of the three single-candidate
backends, names the `Placement` field (if any) the cost model needs to
distinguish them, and states which candidate argmin picks under the
current cost-model constants.

This spec is design-only. The coder (Task 010) executes it.

## TL;DR — what argmin sees after this spec

| Backend  | Cand 1 (label)                     | Cand 2 (label)                         | Cost(1) | Cost(2) | Argmin picks | Δ-driver |
|---       |---                                 |---                                     |---      |---      |---           |---       |
| Samsung  | `y` on bank-row (current)          | `y` GRF-staged                         | 65*N    | 512*N   | Cand 1       | `is_auto` flips: 65 vs 512 per MAC at K=128 (K=1024 → 513 vs 4096) |
| APU v1   | SV mode (raw MUL + ADD)            | SV-lookup (`gvml_lookup_16 + add_s16`) | 18*M    | 8*M     | Cand 2       | `mode` field: MUL+ADD → 16+2; lookup → 6+2 |
| APU v2   | `l1[0,1,2]` for `(x,y,acc)`        | `l1[2,1,0]` for `(x,y,acc)`            | N       | N       | Cand 1 (tied, lex-first by enumerator idx) | placeholder cost; identical; argmin still runs and is deterministic |

`N` is the per-bucket match count, `K` the innermost loop trip count.

## 0. Shared changes to `Placement`

This is the only invasive bit. Two additive fields, both default to
"unchanged behaviour":

```python
@dataclass
class Placement:
    placements: dict[str, Any] = field(default_factory=dict)
    # NEW: a free-form label string that the cost model and codegen
    # consult to disambiguate candidates whose `placements` dict is
    # identical or whose op-expansion differs. Default `""` preserves
    # current behaviour (cost models that don't read it are unaffected).
    mode: str = ""
    # NEW: free-form scratch dict for backend-specific extras the
    # enumerator wants to thread to the cost model / codegen without
    # ballooning the Placement schema. Default `{}`.
    extra: dict[str, Any] = field(default_factory=dict)
```

Constraints:

* `mode` MUST be one of a backend-declared finite set (asserted by the
  cost factory when it reads it; unknown `mode` → raise). For this
  spec the only consumer is APU v1; allowed values are `"sv"` and
  `"sv_lookup"`. Samsung and APU v2 leave `mode=""`.
* `extra` is consulted by the relevant cost model only; the regalloc
  must round-trip it untouched (`spmw_regalloc.py:512` constructs a
  new `Placement(placements=...)` — the coder updates that one site to
  preserve `mode` and `extra` on the refined placement).
* Codegen MAY read `mode` (APU v1 does — see §APU v1 below); AiM /
  UPMEM / Samsung / APU v2 codegen ignores it.

Why a field on `Placement` and not on `MatchedOp.extra`: the candidate
choice is a *placement-time* decision (the autoscheduler picks it);
the matcher does not know about op-expansion alternatives. Stashing it
on `MatchedOp.extra` would require the autoscheduler to mutate trace
state — which spec 007 (option **b** justifications, §APU v1 and §APU
v2) is explicit about avoiding.

## 1. Samsung — bank-row vs GRF-staged `y`

**File:** `spmw_autoschedule.py:_samsung_enumerate` (lines 105-161).

**Cost-model signal:** `_samsung_kernel_cycles`
(`spmw_cost_models.py:142-172`) reads
`is_auto = isinstance(y_handle, MemoryRef)`. When `y` is bank-shaped
the K-loop folds to `(K // 8)` MACs plus one JUMP; when `y` is a
`Register` it stays unrolled at K MACs. This existing branch is
exactly the cost signal we want — no cost-model change.

### Candidate 1 — bank-row `y` (current, optimal-swizzle)

Unchanged from the existing single placement
(`spmw_autoschedule.py:156-161`):

```python
y_handle = materialise_handle(
    swizzled, target=target, out_dim="bank",
    fixed={"grf": 0, "tile": 0}, symbol_table={"bank": 2 * pid},
)
Placement(placements={
    x_mref: target.grf_a,
    y_mref: y_handle,              # MemoryRef → is_auto=True
    acc_mref: target.grf_b,
}, mode="bank_row")
```

Cost at K=128: folded = 128//8 = 16, total = 16*4 + 1 = **65 cycles
per match**.

### Candidate 2 — GRF-staged `y`

Push `y` to a register file. Samsung MAC's
`src=(or_(any_bank, any_reg), or_(any_bank, any_reg))` allows `y` on
either GRF. The natural choice is `grf_a` (since `dst=grf_b` is pinned
by `accumulates=True`), giving:

```python
Placement(placements={
    x_mref: target.grf_a,
    y_mref: target.grf_a,          # Register → is_auto=False
    acc_mref: target.grf_b,
}, mode="grf_staged")
```

Cost at K=128: total = 128 * 4 = **512 cycles per match**.

**Argmin picks Candidate 1** by a factor of ~8x (4096 vs 513 for
K=1024). The selection is robust under the existing constants.

**Why this pair, not bank-row × swizzle-on/off:** the swizzle is a
sub-choice within bank-row staging; turning it off just rearranges
even/odd bank assignment, not the `is_auto` bit. There is no cost
signal for it under `_samsung_kernel_cycles`. The bank-row vs
GRF-staged pair is the one the cost model can actually rank, which is
the success criterion.

**Why not also enumerate broadcast-vs-partitioned `x`:** `x` always
lands on `grf_a` here (no MemoryRef alternative is wired in the
current target); the cost model has no `is_auto`-equivalent signal for
`x`. Out of scope for this spec; revisit if a later target spec
introduces `mem`-bound `x`.

### Samsung Placement fields used

* `placements`: unchanged shape.
* `mode`: `"bank_row"` vs `"grf_staged"` — purely a label for tests
  and logs; the Samsung cost reads `placements`, not `mode`.
* `extra`: unused.

### No new target handles

Both candidates use `target.grf_a`, `target.grf_b`, and the
materialised bank handle, all of which already exist in
`build_samsung_target()` (`tests/spmw/_fixtures.py:96-154`).

## 2. APU v1 — SV mode vs SV-lookup mode

**File:** `spmw_autoschedule.py:_apu_v1_enumerate` (lines 280-314).

**Cost-model signal:** today
`_APU_V1_OP_CYCLES["MAC"] = _APU_V1_MAC_CYCLES = LOOKUP + ADD = 6 + 2
= 8`. This is the *SV-lookup* expansion only. The spec needs the cost
model to charge a different number when MAC is expanded as raw
MUL+ADD (`_APU_V1_MUL_CYCLES + _APU_V1_ADD_CYCLES = 16 + 2 = 18`).
This is the only place a cost-model change is required.

**Cost-model change:** in `_apu_v1_kernel_cycles`
(`spmw_cost_models.py:190-217`), branch on `layout.mode`:

```python
def cost_fn(trace, layout):
    sv_mode = (layout.mode == "sv")
    total = 0
    for match in trace.matches:
        if match.target_op_name == "MAC" and sv_mode:
            per_op = _APU_V1_MUL_CYCLES + _APU_V1_ADD_CYCLES   # 18
        else:
            per_op = _APU_V1_OP_CYCLES.get(
                match.target_op_name, _APU_V1_ADD_CYCLES)      # 8 for MAC
        iters = 1
        if match.enclosing_loops:
            for (_, _, ub_text, _) in match.enclosing_loops[:-1]:
                ub = _parse_loop_bound(ub_text)
                if ub is not None:
                    iters *= ub
        total += per_op * iters
    return total
```

(Coder note: the `_APU_V1_MAC_CYCLES` constant already exists; the
new SV-mode constant `_APU_V1_MUL_CYCLES + _APU_V1_ADD_CYCLES` is just
the existing `MUL` + `ADD` entries summed inline — no new module
constant required.)

### Candidate 1 — SV mode (raw MUL+ADD on VRs)

```python
Placement(placements={
    x_mref: vrs,
    y_mref: vrs,
    acc_mref: vrs,
}, mode="sv")
```

Cost per MAC match: 18 cycles × outer-loop iters.

### Candidate 2 — SV-lookup mode (`gvml_lookup_16 + gvml_add_s16`)

```python
Placement(placements={
    x_mref: vrs,
    y_mref: vrs,
    acc_mref: vrs,
}, mode="sv_lookup")
```

Cost per MAC match: 8 cycles × outer-loop iters.

**Argmin picks Candidate 2** at 8 < 18 per MAC. With a typical GEMV
trace (one MAC per i-step × M outer iters), the gap is M*10 cycles —
clearly above noise. Matches the 19.7× SV→SV-lookup speedup recorded
in the project memory entry (`project_apu_v1_two_tier_validated.md`).

### Codegen change (APU v1 ctx)

`APUv1Ctx.emit_mac_lookup` (`spmw_codegen.py:786-801`) today
unconditionally emits the lookup expansion. With `mode` on placement,
the codegen op-emit path for MAC must branch:

* `mode == "sv_lookup"` (or `""` for back-compat) → call
  `emit_mac_lookup` (current behaviour).
* `mode == "sv"` → emit two cmds inline:
  `gvml_mul_u16(<tmp>, <x>, <y>);`
  `gvml_add_s16(<acc>, <acc>, <tmp>);`

The hook point is where codegen looks up the op's `emit` lambda for a
matched MAC. The coder threads `placement` into the codegen ctx
(it already has access — `spmw_codegen.py:982` shows
`layout.placements.get(opb.memref_name)` in scope) and dispatches on
`layout.mode` before invoking the lambda. Concretely: add a
`emit_mac_mul_add(self, acc, x, y)` method to `APUv1Ctx` (mirrors
`emit_mac_lookup`) and select between them in the MAC dispatch.

### APU v1 target handles

No new handles. The fixture's MAC op declares
`emit=lambda x, y, acc, ctx: ctx.emit_mac_lookup(...)`
(`_fixtures.py:486-488`). The mode-switched codegen path overrides
this lambda at the dispatch level — the fixture stays unchanged.

If a future cleaner design moves the mode switch onto the target
(e.g. declaring two `Op("MAC", mode="sv")` and `Op("MAC", mode="sv_lookup")`),
the fixture would gain a second `allo.op("MAC", ..., mode=...)`
declaration. **Out of scope for this spec** — keep the single Op,
branch in codegen on `layout.mode`. Architecturally cleaner pass-2
work, not blocker-cycle work.

## 3. APU v2 — two L1-row bindings (placeholder cost)

**File:** `spmw_autoschedule.py:_apu_v2_enumerate` (lines 317-351).

**Cost-model signal:** none. `_apu_v2_kernel_cycles` returns
`len(trace.matches)` (`spmw_cost_models.py:271-274`) — a structural
placeholder until Task 010 builds a real cost. The architect's
guidance from SPEC-007 §APU v2 stands: the `l1[0/1/2]` subscripts are
arbitrary symbolic slots, not derived from algebra.

Per the task file, "even if the cost model returns the same value for
both, argmin should see 2 candidates." So we enumerate two with
identical placeholder cost; argmin tie-breaks lex-first on enumerator
order (the existing `scored.sort(key=lambda t: (t[0], t[1]))` at
`spmw_autoschedule.py:435`).

### Candidate 1 — canonical L1-row binding (current)

```python
Placement(placements={
    x_mref: l1[0],
    y_mref: l1[1],
    acc_mref: l1[2],
}, mode="l1_row_canonical")
```

### Candidate 2 — reversed L1-row binding

```python
Placement(placements={
    x_mref: l1[2],
    y_mref: l1[1],
    acc_mref: l1[0],
}, mode="l1_row_reversed")
```

Both score `len(trace.matches)` under the placeholder. **Argmin picks
Candidate 1** by tie-breaking on enumerator index (lex-first).

### Why "reversed" specifically

The reversal puts `acc` on the lowest L1 row index, swapping it with
`x`. This is the smallest non-trivial perturbation: it keeps the same
set of L1 slots `{0,1,2}` (so the placeholder cost is genuinely tied)
but produces a different role-to-slot mapping. When Task 010 upgrades
the cost model — e.g. to charge L1-row distance from a hypothetical
"primary" axis, or to favour `acc` on the highest-indexed row for
spill-tier proximity — this exact pair is what gates the upgrade.
Anything more elaborate (group-major walks, an `l1_walk` LinearLayout
out_dim) is the right design at Task 010 time, not now.

### Forward-looking pointer for Task 010

The cost-model upgrade in Task 010 should consume `placement.mode`
or `placement.extra` to distinguish these two; once a real signal is
wired, this section of the spec becomes the regression test
(Candidate 1 must remain argmin-picked under whatever new cost
function is chosen, or the upgrade's win condition must be stated
explicitly: e.g. "Candidate 2 now wins because reversed binding
minimises L1-row distance from the L5 staging slot").

### APU v2 target handles

No new handles. `target.l1[i]` works for any in-range `i` — the
fixture declares `l1 = allo.mem(rows=3072, cols=65536, ...)`
(`_fixtures.py:511`); indices 0..2 are well within bounds, and the
swap to 2/1/0 is also valid.

## 4. Test inventory

The coder must keep these green; each is the regression gate for one
piece of this spec.

* `tests/spmw/test_autoschedule.py` — extend with three new tests:
  * `test_samsung_argmin_picks_bank_row` — assert the chosen
    placement has `y` on a `MemoryRef`, not a `Register`.
  * `test_apu_v1_argmin_picks_sv_lookup` — assert chosen
    `placement.mode == "sv_lookup"`.
  * `test_apu_v2_argmin_picks_canonical` — assert chosen
    `placement.mode == "l1_row_canonical"` (lex-first tie-break).
* `tests/spmw/test_target_apu_v1.py` — extend the end-to-end emit
  test with one variant that pins `mode="sv"` (override the
  autoscheduler) and asserts the emitted cmds contain `gvml_mul_u16`,
  not `gvml_lookup_16`.
* `tests/spmw/test_target_samsung.py` — no API surface change; the
  existing emit test must still pass (chosen candidate is the same as
  today's single candidate).
* `tests/spmw/test_target_apu_v2.py` — no functional change; existing
  emit must still pass with Candidate 1.
* `tests/spmw/test_target_aim.py`, `test_target_upmem.py` — unchanged
  (both already enumerate two; this spec does not touch them).
* `tests/spmw/test_linear_layout.py` — unchanged.
* Upstream FPGA / AIE tests under `tests/dataflow/`, `tests/customize/`
  — untouched (no shared-file edits).

## 5. Shared-file boundary

No `allo/ir/*` or `allo/dataflow.py` edits. All changes are confined
to:

* `experiments/allo/allo/spmw_autoschedule.py` — `Placement`
  dataclass gains `mode` and `extra`; three enumerators each return
  two candidates.
* `experiments/allo/allo/spmw_cost_models.py` —
  `_apu_v1_kernel_cycles` branches on `layout.mode`. Samsung and
  APU v2 cost factories unchanged.
* `experiments/allo/allo/spmw_codegen.py` — APU v1 MAC dispatch
  branches on `layout.mode`; new `APUv1Ctx.emit_mac_mul_add` helper.
* `experiments/allo/allo/spmw_regalloc.py` — one site
  (`spmw_regalloc.py:512`) preserves `mode` and `extra` when
  constructing the refined `Placement`.

Blast radius: SPMW-only. Rollback: revert the SPEC-010 commit; the
default `mode=""` makes the new fields no-ops for any consumer that
ignores them, so even a partial revert (e.g. backing out the cost
branch but keeping the field) is safe.

## 6. Rollback

`git revert` of the SPEC-010 commit. The defaults (`mode=""`,
`extra={}`) guarantee any consumer reading the new fields gracefully
falls back to today's behaviour: APU v1 cost still scores MAC at 8;
Samsung and APU v2 cost ignore `mode` entirely.

## 7. Open design tensions (recorded, not blockers)

1. **Should APU v1's mode live on the target (two `Op("MAC", ...)`
   declarations) instead of on `Placement`?** Architecturally cleaner
   (the target spec is where lowering choices belong), but it
   requires the matcher to emit two distinct MatchedOps for the same
   IR site — a real engine change. Spec defers this; current solution
   is the minimal additive one.
2. **Samsung swizzle-on/off as a third candidate.** Real if/when a
   target with multiple bank-bit-swizzle options ships; today the
   ASPLOS '26 swizzle is uniquely optimal and the cost model has no
   signal that differentiates swizzle alternatives.
3. **APU v2 group-major vs row-major walk.** Surfaces only when the
   cost model has a real L1-distance metric. Flagged in §3 above as
   the Task 010 win condition.

These three tensions are linked from the architect's memory entry
for this spec; do not let them creep into the coder's task.

## Implemented

* 2026-05-18 (Task 010): `Placement` gained `mode: str` and
  `extra: dict` fields; Samsung enumerator returns `[bank_row,
  grf_staged]` (argmin picks `bank_row`); APU v1 enumerator returns
  `[sv, sv_lookup]` with `_apu_v1_kernel_cycles` branching on
  `layout.mode` (argmin picks `sv_lookup`); APU v2 enumerator returns
  `[l1_row_canonical, l1_row_reversed]` (argmin picks canonical via
  lex tie-break); `APUv1Ctx.emit_mac_mul_add` added and dispatched by
  the walker on `mode == "sv"`; `spmw_regalloc.allocate` round-trips
  `mode`/`extra` onto the refined `Placement`. Tests in
  `tests/spmw/test_autoschedule.py` (6 new: enumerator counts and
  argmin selections per backend). pytest `tests/spmw/`: 84 passed
  (2 pre-existing APU v1 hardware-build failures unrelated to
  this spec).
