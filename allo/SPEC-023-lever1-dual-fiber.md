# SPEC-023 — Lever 1: even-only → dual-fiber bank placement

Status: APPROVED (architect). Implemented by coder task 031.
Prereq: SPEC-022 (symbolic fibers — layout-algebra receipt path).
Soft-gate: task 010 / SPEC-021 (faithful run path). The cost-model and
codegen changes here are valid regardless; the *observable* cycle delta
on PIMSimulator only lands once SPEC-021's faithful driver path (task
025) is in, because today `getCycle()` is shape-fixed (SPEC-021 verdict
(a)). The coder task's *measurement* step is gated on 025; the
enumerator/cost/codegen changes are not.

Scope guardrail: Samsung-only. All edits live in
`spmw_autoschedule.py::_samsung_enumerate`,
`spmw_cost_models.py::_samsung_kernel_cycles`, and the Samsung
materialisation path in `spmw_codegen.py`. No edits to AiM / UPMEM / APU
enumerators, cost branches, or ctxs. No edits to shared `ir/builder.py`,
`ir/infer.py`, `dataflow.py`, or `customize.py`.

---

## 1. Problem (diagnosis, not the fix)

`_samsung_enumerate` (spmw_autoschedule.py:138-161) builds the correct
swizzled layout (`tile_parity ⊕ bank_bit_0`, i.e. `bases["tile"]=[(0,1)]`
per `optimal_swizzle`), then **collapses it** at materialisation:

```python
y_handle = materialise_handle(
    swizzled, target=target, out_dim="bank",
    fixed={"grf": 0, "tile": 0},          # <-- tile pinned to 0
    symbol_table={"bank": 2 * pid},
)
```

`fixed={"tile": 0}` evaluates the swizzle at one fiber only, yielding
`idx = 2*pid + 0` → every MAC targets `EVEN_BANK`. The odd bank
(`2*pid + 1`) is never addressed, so 8 of 16 banks per pseudo-channel
idle. The swizzle the layout *computed* is being thrown away. ~2× loss.

The native kernel interleaves: it walks input tiles alternating
`EVEN_BANK`/`ODD_BANK`, keeping both bank halves busy
(PIMKernel.cpp:502-508 even/odd interleave; cmd gen in `PIMCmdGen.h`).

The fix is **not** to special-case the even/odd emission in codegen. The
fix is: enumerate a real `Placement` that materialises *both* fibers, let
the cost model see it is ~2× cheaper *from lane/bank counts*, let argmin
pick it, and let codegen materialise the alternating stream from fields
the placement carries.

---

## 2. The layout-algebra receipt — both fibers are already derivable

**Verdict (cross-check against SPEC-022): the existing scalar-multiplier
path in `materialise_handle` already emits both fibers from symbolic
`pid` over `fixed` `tile ∈ {0,1}`.** No new F2 machinery is required for
*lever 1* itself; SPEC-022 owns the strictly-symbolic-`tile` form (where
`tile` is also a free symbol, needed only if a future candidate wants the
parity left unbound). Lever 1 binds `tile` to each concrete fiber value,
which the helper supports today. Quoted derivation, general over M/K:

The swizzled layout has `bases["bank"]=[(0,1),(0,2),(0,4),(0,8)]`
(identity column on the `bank` output) and `bases["tile"]=[(0,1)]`
(single-bit input, mask `m=1` on the `bank` output —
spmw_linear_layout.py:186-187, optimal_swizzle docstring).

`materialise_handle(out_dim="bank", symbol_table={"bank": 2*pid},
fixed={"tile": v})` computes (spmw_linear_layout.py:662-708):

- `bank` input: scaled-identity column, `scale=1`, contribution = the
  bound symbol `2*pid`.
- `tile` input: it is in `fixed`, so contribution =
  `XOR over set bits of v of mask[bit]` = `(v & 1) * 1` (one basis
  vector, mask 1). For `v=0` → `0`; for `v=1` → `1`.
- Sum (XOR on disjoint bit positions == addition):
  - `v=0`: `idx = 2*pid + 0` → `_bank_parity` → `"EVEN_BANK"`.
  - `v=1`: `idx = 2*pid + 1` → `_bank_parity` → `"ODD_BANK"`.

Neither `2*pid` nor `2*pid+1` is a literal pasted at the call site: the
`+0` / `+1` fall out of evaluating the *same swizzled layout* at
`tile=0` vs `tile=1`. The only literal at the call site is the swizzle
construction (already present) and the fiber values `{0, 1}` — which are
the two values of the `tile` axis whose size the layout declares
(`out_sizes`/`in` size 2). The coder MUST obtain the fibers by iterating
`range(layout_tile_size)` (= 2 here, read from the layout's tile-axis
size), never by writing `[0, 1]` as a magic pair.

This is positive-evidence #4 (layout-algebra receipt). The verifier
reproduces it by calling `materialise_handle` over symbolic `pid` at each
fiber and asserting `_bank_parity` returns EVEN then ODD.

> Dependency on SPEC-022: if SPEC-022 lands an extended materialise API
> with a cleaner multi-fiber entry (e.g. returning a `list` of handles
> over an unbound `tile`), the coder SHOULD consume that instead of the
> per-fiber loop below; the candidate-struct fields in §3 are unchanged
> either way (they store the resolved handle list). Until SPEC-022 lands,
> the per-fiber `fixed={"tile": v}` loop is the committed mechanism.

---

## 3. Candidate struct — the `dual_fiber` Placement

Add one new candidate to the list `_samsung_enumerate` returns. It is a
real `Placement` (the existing dataclass, spmw_autoschedule.py:26-44),
distinguished by `mode` and carrying the per-fiber handles in `extra`.
**No new dataclass.** `Placement` already has `placements`, `mode`,
`extra` — the dual-fiber data rides in `extra`, which is exactly its
"free-form per-candidate scratch" purpose (docstring line 39, 44).

### 3.1 Fields

```python
dual_fiber = Placement(
    placements={
        x_mref:   target.grf_a,        # same as bank_row
        y_mref:   y_even,              # the EVEN handle — the
                                       # canonical / default fiber, so
                                       # any consumer that ignores `extra`
                                       # still sees a valid bank-row y.
        acc_mref: target.grf_b,        # MAC dst constraint, same as both
                                       # existing candidates
    },
    mode="dual_fiber",
    extra={
        # Ordered fiber list. Index i is fiber value i (tile=i).
        # Each entry is the materialise_handle output for that fiber,
        # i.e. a bank-shaped MemoryRef whose idx is 2*pid (+i).
        "fibers": [y_even, y_odd],     # len == layout tile-axis size
        # The parity label per fiber, for codegen's emission order and
        # for the cost model's "n_fibers" read. Derived, not asserted:
        # codegen MUST recompute via _bank_parity(handle.idx), this is a
        # convenience cache only.
        "fiber_parities": ["EVEN_BANK", "ODD_BANK"],
        # The layout's segment-axis name and size, so codegen derives
        # the per-fiber trip-count split without a literal `2`.
        "fiber_axis": "tile",
        "n_fibers": 2,                 # == len(fibers); read from layout
    },
)
```

`y_even` / `y_odd` are produced by the §2 loop:

```python
fibers = []
n_fibers = swizzled.size_of("tile")          # = 2, from the layout, NOT a literal
for v in range(n_fibers):
    fibers.append(materialise_handle(
        swizzled, target=target, out_dim="bank",
        fixed={"grf": 0, "tile": v},
        symbol_table={"bank": 2 * pid},
    ))
y_even, y_odd = fibers[0], fibers[-1]
```

(If `LinearLayout` lacks a `size_of`/axis-size accessor, the coder adds a
trivial read of the existing `out_sizes`/bases length for the tile axis —
that is a Tenon-internal read, not a shape literal. Filed as a sub-note
to SPEC-022's author; non-blocking.)

### 3.2 How it differs from `bank_row` / `grf_staged`

| field        | `bank_row`            | `grf_staged`     | `dual_fiber` (new)              |
|--------------|-----------------------|------------------|---------------------------------|
| `y` handle   | `y_handle` (EVEN only)| `grf_a`          | `y_even` in `placements`; both fibers in `extra["fibers"]` |
| `mode`       | `"bank_row"`          | `"grf_staged"`   | `"dual_fiber"`                  |
| `extra`      | `{}`                  | `{}`             | fibers + parities + axis        |
| MAC fold     | is_auto, 1 bank       | unrolled         | is_auto, 2 banks in parallel    |

`dual_fiber` is a *strict superset* of `bank_row`: drop `extra` and it
degrades to `bank_row` (EVEN-only). This is deliberate — it keeps the
candidate materialisable by any consumer that doesn't read `extra`, and
makes the additive change safe for the regression suite.

`bank_row` and `grf_staged` are **retained unchanged.** The enumerator
returns `[bank_row, grf_staged, dual_fiber]`. argmin decides among all
three. (Do not delete `bank_row`: keeping it is what lets the cost-drives-
choice probe show the winner moved from `bank_row` to `dual_fiber` when
the model is correct, and lets a perturbation flip it back.)

---

## 4. Cost model — price dual_fiber ~2× cheaper from `target.*` only

Edit `_samsung_kernel_cycles` (spmw_cost_models.py:82-122). Today the
`is_auto` branch is:

```python
folded = inner_ub // lane_burst
total += folded * mac_cyc + jump_cyc
```

This counts the *entire* K reduction as a single fiber of MACs on one
bank. The dual-fiber placement runs the two bank halves **concurrently**
(even and odd banks accept MAC commands in interleave; per-bank MAC
throughput is independent), so the modelled compute time is the work of
the *larger* fiber, not the sum. The split is by the layout's fiber axis.

### 4.1 Formula

Read `n_fibers` from the placement (`layout.extra.get("n_fibers", 1)`),
defaulting to 1 so `bank_row`/`grf_staged` keep today's behaviour
exactly (additive change — no existing candidate's cost moves):

```python
n_fibers = layout.extra.get("n_fibers", 1) if is_auto else 1
# Total folded MACs across the whole K reduction (unchanged definition).
folded = inner_ub // lane_burst
# Per-fiber MAC count: ceil-split so an odd tile count still bounds the
# busier fiber. ceil(folded / n_fibers) — expressed without `math.ceil`
# literals: (folded + n_fibers - 1) // n_fibers.
per_fiber = (folded + n_fibers - 1) // n_fibers
# One JUMP per fiber (each bank half folds its own inner loop).
total += per_fiber * mac_cyc + n_fibers * jump_cyc
```

For `bank_row`: `n_fibers=1` → `per_fiber = folded`, `+1*jump_cyc` →
**identical to today** (regression-safe).

For `dual_fiber`: `n_fibers=2` → `per_fiber = ceil(folded/2)`,
`+2*jump_cyc` → ~half the MAC cycles plus one extra JUMP. With
`mac_cyc=4`, `jump_cyc=1`, `lane_burst=8`, this is a strict win for any
`inner_ub` where `folded ≥ 2` (i.e. `K ≥ 2*lanes`), which is every real
GEMV. The cost crosses over correctly: as `K→0` the extra JUMP makes
dual_fiber not worth it and `bank_row` wins; as `K` grows the
~2× MAC saving dominates. **The output changes correctly with K** —
required by the anti-hardcoding gate (no hand-picked ranking).

### 4.2 What this prices from, line by line (provenance)

- `mac_cyc = target.op("MAC").cycles` — already read (line 89).
- `jump_cyc = target.move("JUMP").cycles` — already read (line 90).
- `lane_burst = target.grf_a.lanes` — already read (line 91).
- `n_fibers` — read from `layout.extra`, which the enumerator filled from
  the **layout's** tile-axis size, which came from the swizzle. Not a
  literal `2` in the cost model.

No shape literal enters the cost model. The only integers are the spec
constants above and the `+ n_fibers - 1` ceil idiom (arithmetic on a
layout-derived count, not a shape).

### 4.3 Generality / monotonicity probes the verifier will run

- Bump `target.move("JUMP").cycles` and confirm dual_fiber's modelled
  cost rises by exactly `n_fibers * Δjump` relative to bank_row — proving
  the number is computed.
- Halve `target.op("MAC").cycles` and confirm the dual_fiber advantage
  shrinks proportionally.
- Run K=768 (non-power-of-two, off-task) and confirm `per_fiber` uses the
  ceil split and dual_fiber still wins by cost.

---

## 5. Codegen materialisation — alternating EVEN/ODD with split trips

The **decision lives in argmin** (§3/§4). Codegen only *materialises*
`mode == "dual_fiber"`: it reads `extra["fibers"]` and emits Samsung's
alternating `(MAC EVEN, JUMP n_even, MAC ODD, JUMP n_odd)`.

### 5.1 Where it hooks

The cleanest seam is the existing per-match Samsung path:
`SamsungCtx.after_match` (spmw_codegen.py:270-274) already owns the
inner-K JUMP via `_emit_inner_loop_jump`. Lever 1 generalises that single
JUMP into a per-fiber loop. Two options, chosen below.

- **Option A (chosen): a dual-fiber-aware inner-loop emitter.** When the
  active placement's `mode == "dual_fiber"`, the Samsung MAC emission for
  this match emits one `(MAC <fiber>, JUMP n_fiber)` pair *per fiber* in
  `extra["fibers"]`, instead of one MAC + one JUMP for the single EVEN
  handle. The fiber handle is passed as the MAC `src1` so `_opd` →
  `_bank_parity` stamps `EVEN_BANK` / `ODD_BANK` from the handle's own
  idx (no parity string hardcoded in codegen).
- Option B (rejected): a separate `_run_samsung` branch that detects the
  shape and replays an even/odd template. Rejected — that is decision +
  golden-stream in codegen, exactly what the gate forbids.

Mechanism contract for Option A:

1. `SamsungCtx` gains read access to the current match's placement (the
   walker already resolves `layout` per work-id at
   spmw_codegen.py:1280; pass `layout.mode` / `layout.extra` to the ctx
   for the active work-id, or stash it on the ctx in `_walk_and_emit`
   before the match loop — a ctx field `self._active_placement`). This is
   additive ctx state; default `None` preserves single-fiber behaviour.
2. For a `dual_fiber` MAC match, emit, for each `handle` in
   `extra["fibers"]` (in fiber order):
   - `MAC dst=GRF_B src0=GRF_A src1=<handle>` — `is_auto=1` (src1 is a
     bank `MemoryRef`), parity stamped by `_bank_parity(handle.idx)`.
   - `JUMP loop_counter=n_fiber loop_offset=<body insns + 1>`.
3. For any other mode, behaviour is **unchanged** (one MAC + the single
   `_emit_inner_loop_jump`).

### 5.2 Split trip counts — DERIVED, never literal

The current emitter uses a module-level literal
`_SAMSUNG_LANE_BURST = 8` (spmw_codegen.py:1129) — a code smell flagged
here. **Lever-1 codegen MUST read the burst from
`target.grf_a.lanes`**, not the literal. (Replacing the literal with the
target read is in-scope for task 031; it is the same constant, now
sourced correctly. This also discharges the FIXME comment at
spmw_codegen.py:1124-1129.)

Per-fiber trip counts, as symbolic expressions over the *inner-K loop
bound* (`inner_ub`, parsed from `match.enclosing_loops[-1]` exactly as
today) and `target` geometry:

```
lanes    = target.grf_a.lanes                  # spec constant, e.g. 8
folded   = inner_ub // lanes                    # total in_tiles after burst fold
n        = extra["n_fibers"]                    # = len(fibers), from layout
# tiles assigned to fiber i (round-robin even/odd split):
folded_i = (folded + (n - 1 - i)) // n          # ceil for i< rem, floor after
# JUMP loop_counter for fiber i: first MAC is iter 0, JUMP loops the rest:
n_fiber_i = folded_i - 1                         # emit JUMP only if > 0
```

For the canonical M=4096,K=1024 case with `lanes=8`, `n=2`:
`folded = 1024//8 = 128`; `folded_0 = 64`, `folded_1 = 64`;
`n_even = n_odd = 63`. The literal `63` from the task description thus
**emerges** from `(K//lanes)/2 - 1`; it is never written. For K=768:
`folded = 96`, `folded_0=folded_1=48`, `n_each = 47`. For an odd
`folded` (e.g. K such that `folded=127`): `folded_0=64`, `folded_1=63`,
`n_even=63`, `n_odd=62` — the ceil split keeps the busier fiber bounded,
matching the cost model's `per_fiber` ceil in §4.1.

`loop_offset` is `body_insns + 1` exactly as `_emit_inner_loop_jump`
computes it today (one iteration body = the MACs emitted for this fiber +
the JUMP). For the canonical single-MAC body, offset = 2.

### 5.3 `in_tiles` naming

The task uses `in_tiles`; in the code this is `folded = inner_ub //
lanes` (the number of K-burst tiles). The receipt formula
`n_even = ceil(in_tiles/2)*... ` reconciles as: `in_tiles = folded`,
`n_even = ceil(in_tiles/2) - 1` (the `*lanes` in the task's loose
phrasing is absorbed because `in_tiles` is already post-burst). The
committed formula is §5.2; the coder uses that, not the task's prose
shorthand.

---

## 6. Decision vs mechanism — explicit pointer

- **Decision (argmin):** `spmw_autoschedule.py`. The enumerator returns
  `[bank_row, grf_staged, dual_fiber]`; the cost model (§4) prices each;
  the existing argmin in the autoschedule loop picks the minimum. The
  cost model is the *only* place that ranks `dual_fiber` above
  `bank_row`. If you delete the `n_fibers` read in the cost model,
  `dual_fiber` and `bank_row` tie and argmin keeps the first — i.e. the
  win provably comes from the modelled cost, not from enumeration order
  or from codegen.
- **Mechanism (codegen):** `spmw_codegen.py` `SamsungCtx`. It reads
  `placement.mode == "dual_fiber"` and `extra["fibers"]` and emits the
  alternating stream. It contains **no** comparison of shapes, no choice
  of fast vs slow path, no parity string literal (parity comes from
  `_bank_parity(handle.idx)`).

---

## 7. Anti-hardcoding self-audit (pre-cleared against the gate)

| forbidden pattern | how this spec avoids it |
|---|---|
| shape literals (4096/1024/63/127/128…) in decision paths | trip counts are `(inner_ub//lanes)` split by `n_fibers`; `63` *emerges*, never written. `n_fibers` from layout axis size. |
| test/shape/memref sniffing | no branch on shape, name, or func_name; codegen branches only on `placement.mode`. |
| precomputed answer as a cost | §4 computes from `target.op/move/grf_a` + `inner_ub`; output changes with K and with perturbed constants. |
| golden cmd-stream replayed | every record generated from the fiber handle + `_bank_parity`; no table. |
| decision logic in codegen | choice is in argmin (§6); codegen materialises only. |

---

## 8. Regression / blast-radius

Additive everywhere:
- Enumerator returns a *third* candidate; the existing two are byte-for-
  byte unchanged. AiM/UPMEM/APU enumerators untouched.
- Cost model: `n_fibers` defaults to 1 → existing candidates' costs
  unchanged; AiM/UPMEM/APU branches untouched.
- Codegen: new behaviour gated on `mode == "dual_fiber"`; all other modes
  hit the unchanged path. `_SAMSUNG_LANE_BURST` literal replaced by
  `target.grf_a.lanes` read (same value, 8).

No shared-file (`ir/builder.py`, `ir/infer.py`, `dataflow.py`,
`customize.py`) edits. No simulator-source edits.

### Gating tests (must stay green; coder runs the full SPMW suite)

- `tests/spmw/test_codegen_gemv.py::test_compile_emits_canonical_mac_jump_pair_per_match`
  — the canonical single-fiber MAC+JUMP subsequence. For `dual_fiber`,
  this becomes a per-fiber subsequence; if the test asserts *exactly one*
  pair it must be relaxed to "one pair per fiber" or split into a
  `bank_row` case + a new `dual_fiber` case. Coder: prefer adding a new
  test over weakening the existing assertion.
- `tests/spmw/test_e2e_*` — numerics (fp16 max-err ≤ 0.0156) must hold;
  the alternating stream addresses the same W/x data, so the readback is
  unchanged.
- `tests/spmw/test_samsung_placement_changes_cycles.py` — per SPEC-021
  this graduates to strict `cycles_a != cycles_b` only after task 025's
  faithful run path. Until then it stays in its current (lenient) form;
  do not strengthen it in task 031.
- New: `tests/spmw/test_autoschedule_samsung_dual_fiber.py` — assert (a)
  enumerator emits a `dual_fiber` candidate whose `extra["fibers"]` are
  EVEN then ODD by `_bank_parity`; (b) cost model ranks `dual_fiber` <
  `bank_row` at K=1024 and the ranking flips when JUMP cycles are bumped
  high; (c) the per-fiber trip counts come out as the §5.2 formula for
  two shapes incl. a non-power-of-two K (the receipt). This is the
  cost-drives-choice + layout-algebra-receipt evidence for the audit.

### Rollback story

If FPGA/AiM/UPMEM/APU CI breaks (it should not — additive), revert the
single `dual_fiber` append in the enumerator and the `n_fibers` read in
the cost model; the codegen `mode == "dual_fiber"` branch is then dead
and inert. The `_SAMSUNG_LANE_BURST` → `target.grf_a.lanes` swap is the
only non-Samsung-gated change and is value-preserving; keep it.

---

## 9. Coder contract (task 031) — checklist

1. Enumerator: append `dual_fiber` `Placement` with the §3.1 fields;
   fibers via the §2 per-fiber loop; `n_fibers` from the layout axis
   size (not literal `2`). Keep `bank_row`/`grf_staged`.
2. Cost model: §4.1 `n_fibers`-aware fold; default 1.
3. Codegen: §5 Option A — `SamsungCtx` reads active placement, emits
   per-fiber MAC+JUMP with §5.2 split trips; replace `_SAMSUNG_LANE_BURST`
   literal with `target.grf_a.lanes`.
4. Tests: new `test_autoschedule_samsung_dual_fiber.py`; relax/split the
   canonical-pair test per §8.
5. **Measurement of the cycle delta is gated on task 025** (faithful run
   path). Until 025 lands, prove the win by cost-model ranking + the
   receipt, not by `getCycle()`.

---

## Implemented (coder task 031)

- Files: `spmw_autoschedule.py` (dual_fiber Placement via per-fiber
  `fixed={"tile":v}` loop over `swizzled.size_of("tile")`),
  `spmw_cost_models.py` (`n_fibers`-aware ceil fold, default 1),
  `spmw_codegen.py` (SamsungCtx `_emit_dual_fiber_jumps`; `_SAMSUNG_LANE_BURST`
  literal replaced by `target.grf_a.lanes`; `_split_samsung_layers` so the
  SPEC-020 run path splits by work-id/preload boundary, not JUMP — required
  because a dual-fiber layer now holds 2 MAC+JUMP pairs),
  `spmw_linear_layout.py` (additive `LinearLayout.size_of`, authorized by §3.1).
- Cost @4096x1024 (inner K=1024, lanes=8): bank_row=513, grf_staged=4096,
  **dual_fiber=258** → argmin picks dual_fiber (2x under bank_row).
- Receipt: fiber0 idx `2*pid`→EVEN_BANK, fiber1 idx `2*pid+1`→ODD_BANK,
  both from `materialise_handle` over the swizzled layout (the `+0`/`+1`
  fall out of the swizzle `tile` column).
- Faithful cycles @4096x1024: native_folded=dual_fiber=15251 (parity — the
  faithful counter clamps each bank path to ≥1, so dual_fiber matches the
  native folded baseline; it differentiates *redundant* streams, not
  even-only vs dual-fiber). The lever-1 win is the cost-model ranking +
  keeping both bank halves addressed, exactly as the §Status soft-gate frames.
- Tests: new `test_autoschedule_samsung_dual_fiber.py` (6); updated the
  canonical-pair / argmin / move-scheduling / multi-layer-split tests to the
  dual-fiber stream. Full `tests/spmw/`: 121 passed.
- Flagged to architect: enumerator still binds `symbol_table={"bank": 2*pid}`
  (the pre-existing scalar-multiplier path). SPEC-022's `bank_stride(target)*pid`
  swap (task 020 not yet coded) discharges the last `2*pid` literal; until then
  SPEC-023 §2's committed `fixed={"tile":v}` mechanism is used.
