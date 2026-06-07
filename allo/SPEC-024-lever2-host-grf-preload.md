# SPEC-024 — Lever 2: GRF preload as a host-vs-CRF placement choice

Status: ruled. Gates coder task 041. Reads: SPEC-021 (faithful run
path; the measurement instrument this lever's cost is observed
through), SPEC-005 (Samsung placement-driven codegen), SPEC-009 (move
scheduling / `resolve_moves`). Scope: **Samsung HBM-PIM only**; no
other backend's enumerator/cost/codegen path changes.

## 0. The lever in one paragraph

Half of Tenon's ~512-record Samsung GEMV stream is GRF preload /
storeback emitted as **CRF MOV** instructions (`LD_A`, `LD_B`, `ST_*`).
Samsung's hand-tuned kernel does not pay for those on the CRF stream:
it loads `GRF_A` once via a **host HAB broadcast** (a host-issued
transaction, outside the per-work-id CRF microcode) and reads the
bank operand into the MAC via the column-strobe (`READ_pim`,
already modelled as `is_auto=1` from lever 1 / SPEC-005). The fix is to
make *where a preloaded operand is materialised from* a **placement
attribute** — host-loaded GRF vs CRF-MOV-loaded GRF — that the
enumerator emits as two real candidates, the cost model prices from
`target.move(...).cycles` + operand counts (no shape literals), and
codegen materialises by either hoisting the preload to a host
transaction or emitting the CRF MOV. **The decision is argmin's; codegen
only reads the chosen attribute and materialises.**

This lever composes with — and is orthogonal to — lever 1 (even/odd
bank, SPEC for task 030) and lever 3 (shared CRF, task 050). It changes
the *preload* records; lever 1 changes the *MAC/JUMP* records; lever 3
changes *replication across work-ids*. The cost terms are additive (§4).

## 1. Why this is a placement choice and not a codegen flag

A preloaded operand (an operand whose role maps to `grf_a`/`grf_b`,
i.e. `resolve_moves` returns a non-`None` load name) can be brought into
its GRF in two physically distinct, both-materialisable ways:

- **host residency** — the host writes the GRF via an HAB broadcast
  before triggering the CRF microcode. The preload is a *host
  transaction*. It does **not** occupy a CRF instruction slot and does
  **not** appear in the per-work-id CRF stream.
- **crf residency** — the preload is a `MOV GRF_x <- bank` instruction
  inside the CRF microcode, issued once per work-id, occupying a CRF
  slot and a stream record.

Both produce identical numerics (the same bytes land in the same GRF).
They differ only in *issued-instruction count*, which is exactly the
quantity SPEC-021's faithful run path makes observable as cycles. So
this is a placement degree of freedom the cost model can rank — not a
correctness fork, and not something codegen should choose.

Host residency is not free: a broadcast preload is only legal for an
operand that is **uniform across the units fed by one HAB broadcast**
(Samsung's `GRF_A` = the `x` vector, broadcast to all banks). An
operand that is *per-bank distinct* (the weight tile `y`, already
bank-resident, and the accumulator `acc` that is written per work-id)
cannot be host-broadcast preloaded. The enumerator therefore only
offers host residency for the broadcastable role(s); the cost model
prices it; for non-broadcastable roles the only candidate is crf
residency (or no preload at all, when the operand is already
bank-resident and read via `is_auto=1`). This legality test is derived
from `target` geometry + the role's home handle, never from a shape.

## 2. Placement field / mode

Extend `Placement` use (no dataclass change required — `Placement.extra`
already exists, SPEC-009) with a **per-memref residency map** carried in
`extra` under a reserved key:

```
Placement.extra["grf_residency"] : dict[str, str]
    # memref_name -> "host" | "crf"
    # absent / missing key  == "crf"  (back-compat default; preserves
    #                                   every existing Samsung placement)
```

Rationale for `extra`, not a new top-level field or a new `mode`
string:

- `mode` is already load-bearing on Samsung (`"bank_row"` vs
  `"grf_staged"`, lever 1 will add `"bank_row_dualfiber"`); overloading
  it with a residency axis would force a combinatorial mode-string
  product across levers. Residency is **per-operand**, so it belongs in
  a per-memref map, not a scalar mode.
- `extra` is already the SPMW-blessed per-candidate scratch dict and is
  read by codegen via the `Placement` it is handed. No shared-file edit,
  no dataclass migration, no blast radius to other backends (they never
  set the key, so they default to `"crf"`).
- A new top-level `Placement.residency` field was considered and
  rejected: it would touch the dataclass that all five backends
  construct, and four of them have nothing to say about GRF residency.
  `extra` keeps the change additive and Samsung-local.

Default rule (mechanically: `placement.extra.get("grf_residency",
{}).get(memref, "crf")`) guarantees that any placement built before this
spec — including the hand-built `Placement`s in
`test_samsung_placement_changes_cycles.py` — behaves exactly as today.

## 3. Enumerator: emit both as materialisable candidates

In `_samsung_enumerate` (`spmw_autoschedule.py`), for the operand(s)
eligible for host residency (the broadcastable preload role — `x`,
which maps to `target.grf_a`), emit the candidate **cross-product**
of the existing layout candidates × {host, crf} residency for that
role. Concretely, today's two candidates (`bank_row`, `grf_staged`)
each split into a host-preload and a crf-preload variant **for the
broadcastable role only**, yielding the residency choice as a real
`Placement`, distinguished by `extra["grf_residency"]`.

Eligibility (the legality test of §1) is computed, not asserted:

- A role is **host-eligible** iff (a) its placement handle is a GRF
  register that takes a preload (`resolve_moves` would return a non-`None`
  load name for it) **and** (b) the operand is uniform across the HAB
  broadcast fan-out — for GEMV this is the `x`/`grf_a` role, because `x`
  is the vector broadcast to every bank. The enumerator identifies this
  structurally: `x` is the role whose home is *not* a per-bank handle
  and whose placement is `target.grf_a`. The weight `y` (bank-resident,
  read via `is_auto`) and `acc` (`grf_b`, written per work-id) are **not**
  host-eligible.
- The number of host-eligible roles and the fan-out count come from
  `target` geometry (channel/bank/unit counts) and the role's home
  handle — never from `M`/`K`.

Enumerator contract (what 041 must produce):

```
for base in (bank_row, grf_staged):            # existing candidates
    yield base_with_residency(base, x_role -> "crf")    # == today
    yield base_with_residency(base, x_role -> "host")   # new
```

`base_with_residency` copies the base `Placement` and sets
`extra["grf_residency"][x_mref] = <mode>`; it does not touch
`placements` (the handle map is identical — residency is *how* the GRF
is filled, not *which* handle holds the value). This keeps the layout
algebra (lever 1's even/odd fibers) fully orthogonal: residency rides
alongside any layout mode.

The enumerator must **not** prune the crf variant — both are offered;
argmin decides. (Pruning host-vs-crf in the enumerator would move the
decision out of the cost model, the forbidden pattern.)

## 4. Cost model: price host-preload as fewer issued CRF instructions

In `_samsung_kernel_cycles` (`spmw_cost_models.py`), the cost of a
preload move depends on residency, priced **only** from target-spec
constants and operand counts:

- **crf residency** of role `r`: the preload is a CRF MOV issued once
  per work-id; it costs `target.move(<load_name(r)>).cycles` per
  work-id and contributes one issued CRF record per work-id.
- **host residency** of role `r`: the preload is hoisted off the CRF
  stream. It contributes **zero CRF-issued cycles** to the kernel cost
  (the host transaction overlaps the CRF program / is amortised across
  the broadcast fan-out). It does **not** add a per-work-id CRF record.

So the kernel-cycle cost must add, per work-id and per preloaded role:

```
load_name = ctx-equivalent load for the role's GRF handle
            (LD_A for grf_a, LD_B for grf_b; the same name resolve_moves
             returns — read it via target.move(load_name).cycles)
residency = placement.extra.get("grf_residency", {}).get(memref, "crf")

if residency == "host":
    preload_cost = 0                       # off the CRF stream
else:  # "crf"
    preload_cost = target.move(load_name).cycles
```

multiplied by the work-id trip count the model already derives (the
outer-loop iteration count over PIM units / output tiles — the same
count lever 1 uses; it is an expression over operand shape and
`target` unit count, **never** a literal). The storeback (`ST_*`) is
priced symmetrically: a host-resident operand's storeback, when it
exists, is a host readback transaction, not a CRF MOV, and likewise
contributes 0 CRF cycles. For GEMV the only stored role is `acc`, which
is **not** host-eligible (§3), so in practice the storeback term is
unchanged by this lever; the rule is stated for completeness and to
keep the accounting symmetric (SPEC-021 §"honest vs flattering": the
same accounting must apply to native and Tenon).

Why this is monotone and shape-responsive (audit positive-evidence #2):

- The host variant's cost is strictly less than the crf variant's by
  exactly `n_workids * target.move("LD_A").cycles` (with `LD_A.cycles
  = 26` in the fixture). `n_workids` is an expression over operand
  shape + unit count, so the differential **scales with K/M** — change
  K and the number of work-ids changes and the saving changes with it.
  This satisfies "a cost whose output doesn't change with K is a red
  flag": this one does.
- Perturbing `target.move("LD_A").cycles` upward makes the host variant
  relatively cheaper and the ranking responds in the modelled
  direction — the cost-drives-choice perturbation probe the audit
  requires.

The cost model must read the load-move name from the role's GRF handle
(grf_a→`LD_A`, grf_b→`LD_B`) the same way `SamsungCtx.resolve_moves`
does, so the priced move and the materialised move are the *same Move
on the target*. It must not hardcode `26`; it reads
`target.move(name).cycles`.

Argmin consequence: among the four Samsung candidates
({bank_row, grf_staged} × {host, crf} for `x`), the
**bank_row + host-x** candidate is the modelled minimum, because (a)
bank_row folds the K-loop (lever 1 / existing is_auto saving) and (b)
host-x removes `n_workids` LD_A MOV cycles. Both savings are computed
from spec constants; neither is asserted.

## 5. Codegen: materialise host preload vs CRF MOV (mechanism only)

In `SamsungCtx` move scheduling (`spmw_codegen.py`), the preload
materialisation reads the residency attribute from the active
`Placement` and branches **only on mechanism**:

- `resolve_moves` stays the resolver of *which* Move name applies to a
  role's handle (unchanged: grf_a→`LD_A`/`ST_A`, grf_b→`LD_B`/`ST_B`,
  MemoryRef→`(None,None)`). It already returns the right names; lever 2
  does not change the name table.
- A new residency consult sits in `_schedule_moves` (or a small
  `SamsungCtx` helper it calls): given the role and the active
  `Placement`, look up `placement.extra["grf_residency"].get(memref,
  "crf")`. For `"crf"`, emit the CRF MOV exactly as today (the move's
  `emit` lambda → `ctx.cmd("MOV", ...)`). For `"host"`, **do not emit a
  CRF record**; instead record the operand on a `SamsungCtx`
  host-preload side-list (e.g. `ctx.host_preloads.append((role, handle))`)
  that the run path turns into a host transaction.

Run-path wiring (`_run_samsung`):

- The host-resident preloads never reach `programCrf` because they are
  not in `compiled.cmds` as CRF MOV records. This is the *correct*
  generalisation of the existing `_crf_valid` workaround at
  `spmw_codegen.py:1629` (which today drops MOV-to-bank storebacks as a
  blanket run-side filter). Under this spec, host-resident GRF preloads
  are **never emitted as CRF records in the first place** — the
  side-list is the placement-driven, decided-upstream version of that
  filter. `_crf_valid` may remain as a belt-and-suspenders ISA check,
  but the residency attribute is now the authority for what is on the
  stream.
- The host preload itself is materialised by the existing GEMV
  data-path scaffolding: `pim_driver`'s `--op GEMV` path already loads
  the `x` vector into GRF_A via the host broadcast (`computeGemv`
  address generation, SPEC-021 §2 keeps this untouched). For host
  residency, Tenon emits **no** CRF MOV for `x` and lets that native
  host load stand — i.e. host residency = "let the driver's host
  broadcast fill GRF_A; don't shadow it with a CRF MOV." This is
  mechanism, and it is exactly what makes host residency cheaper under
  the faithful run path: fewer records in `stream_records`.
- `READ_pim`: the bank operand `y` is read into the MAC via the
  column-strobe (`is_auto=1`, MAC `src1=EVEN_BANK`). That is lever 1 /
  SPEC-005 and already emitted; lever 2 does not change it. The spec
  names it only to record that "native reads GRF_B via READ_pim" maps
  to the existing `is_auto=1` MAC, not to a new move.

### Faithful-run coupling (SPEC-021)

SPEC-021's faithful path derives trip counts + per-tile transaction
multiplicity from `compiled.cmds`. Because host-resident preloads are
absent from `compiled.cmds`, they are automatically excluded from
`stream_records`. **No new logic in the faithful path is required** —
the residency attribute changes the *stream*, and the faithful path
already prices the stream. This is the clean seam: lever 2 lives
entirely in (enumerator emits, cost prices, codegen omits-from-stream);
the run path is untouched beyond what SPEC-021/025 already build.

## 6. What codegen must NOT do (anti-hardcoding)

- Codegen must not decide host-vs-crf. It reads
  `placement.extra["grf_residency"]` and materialises. No
  `if shape is big` / `if role == "x" and K > N` branch.
- No shape literal in the residency branch: the only inputs are the
  role, its GRF handle, and the residency string from the chosen
  `Placement`.
- No golden stream: the host-preload omission is computed per operand
  from the attribute, not by matching a reference byte stream.

## 7. Blast-radius answers (the four required questions)

1. **Can this go entirely inside `spmw_*.py`?** Yes. All three changes
   are in `spmw_autoschedule.py` (`_samsung_enumerate`),
   `spmw_cost_models.py` (`_samsung_kernel_cycles`), and
   `spmw_codegen.py` (`SamsungCtx`/`_schedule_moves`/`_run_samsung`).
   No edit to `ir/builder.py`, `ir/infer.py`, `dataflow.py`, or any
   shared upstream file. No PIMSimulator source edit (host residency =
   *not* emitting a CRF MOV; the native host broadcast already exists).
2. **Additive or invasive?** Additive. New `extra` key with a
   back-compat default (`"crf"`); new enumerator candidates appended;
   new cost term that is 0 in the default-crf path the existing tests
   exercise; new codegen branch gated on the attribute. Every existing
   Samsung placement (including hand-built test placements) keeps its
   current cost and stream.
3. **Existing tests covering this path (must stay green):**
   - `tests/spmw/test_move_scheduling.py::test_samsung_gemv_emits_ld_mac_jump_st_per_workid`
     — default (no residency key) must still emit the `LD_A` CRF MOV.
   - `tests/spmw/test_move_scheduling.py::test_two_workids_emit_two_preload_blocks`
     — default crf residency still emits two preload MOVs.
   - `tests/spmw/test_codegen_gemv.py` (canonical MAC/JUMP pair) — the
     MAC/JUMP body is untouched by lever 2.
   - `tests/spmw/test_samsung_placement_changes_cycles.py` — its
     hand-built `Placement`s have no `grf_residency` key, so they price
     and run exactly as today; `cost_a < cost_b` still holds.
   - `tests/spmw/test_autoschedule.py` — argmin must still return one
     placement per kernel; now from a 4-candidate set, winner =
     bank_row+host-x.
   - AiM / UPMEM / APU enumerators + cost models are untouched (they
     never read `grf_residency`); their move-scheduling tests
     (`test_aim_*`, `test_apu_v1_two_stage_l4_preload`,
     `test_apu_v2_emits_no_moves`) must stay green.
4. **Rollback story.** Two-level. (a) Drop the host candidate from the
   enumerator → argmin only ever sees crf residency → byte-for-byte
   today's behaviour, because the cost term for crf residency is the
   existing implicit cost and the default key is `"crf"`. (b) The
   `SPMW_DISABLE_REGALLOC=1` kill switch (autoschedule §) still selects
   by `kernel_cycles` alone; with only crf candidates that reproduces
   the pre-lever stream. No FPGA/AIE CI path touches Samsung
   enumerator/cost; the shared-file inventory is unchanged, so FPGA CI
   cannot break from this lever.

## 8. New tests the coder must add (task 041)

- `test_samsung_host_residency_omits_crf_mov`: a `Placement` with
  `extra["grf_residency"][x_mref] = "host"` emits **no** `LD_A` MOV in
  `compiled.cmds`; the otherwise-identical `"crf"` placement emits one
  `LD_A` MOV per work-id. (Diff-provenance receipt: the omission is
  traceable to the attribute.)
- `test_samsung_host_residency_priced_cheaper`: under
  `_samsung_kernel_cycles`, the host-x candidate costs strictly less
  than the crf-x candidate by `n_workids * target.move("LD_A").cycles`,
  and the differential **changes when K changes** (parametrise two K
  values; assert the saving scales). (Cost-responds-to-K receipt.)
- `test_samsung_host_residency_perturbation`: bump
  `target.move("LD_A").cycles` and assert the host-vs-crf cost gap
  widens in the modelled direction. (Cost-drives-choice receipt.)
- `test_samsung_argmin_picks_host_residency`: `autoschedule` over the
  4-candidate set returns the `bank_row` + host-x placement as the
  argmin, by cost, not by a constant.

## 9. Acceptance gate for 041

- Enumerator emits both residencies as real `Placement`s
  distinguished by `extra["grf_residency"]`; codegen never special-cases
  the choice.
- Cost model prices host residency at 0 CRF cycles for the preload and
  crf residency at `target.move(load_name).cycles` per work-id; both
  from spec constants + work-id count, no shape literal; output changes
  with K.
- Codegen omits the CRF MOV for host-resident operands (side-list →
  host transaction via the native `--op GEMV` broadcast) and emits it
  for crf-resident operands; `READ_pim` for the bank operand is the
  existing `is_auto=1` MAC, unchanged.
- The choice that reaches codegen is argmin's output; the four new tests
  (§8) pass; the full SPMW suite + the §7.3 inventory stays green.
```
