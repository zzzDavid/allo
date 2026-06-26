# Design 02 — UPMEM tasklet-tiling: where the win lives, and the FFN scope ruling

Task: `dev/06172026-beat-cinnamon-exo-cross-target/work/queue/009-needs-arch-ruling-upmem-tasklet-tiling.task`
(initial ruling), `…/022-needs-arch-upmem-incorporate-pipeline-research.task`
(cost-coefficient fold-in).
Evidence: `dev/06172026-beat-cinnamon-exo-cross-target/work/reports/005-coder.md`,
`…/work/reports/021-research-upmem-pipeline-amortization.md`,
`experiments/baselines/cinnamon-exo/upmem/README.md`.
Mirrors the resolved Samsung precedent T16 / SPEC-026 (batched-GEMV weight
reuse): shape-derived lever on `MatchedOp.extra` → enumerated `Placement.extra`
candidates → cost model prices from `target.*` constants → codegen materializes.

**Status: fully specced.** research-021 CLOSED the cost coefficient (§3.3):
saturating divisor = revolver window `R = 11` (not pipeline depth 14),
`fill_drain = 0`, cost = `S + ceil(S*(R-1)/min(T,R))`, one new target-spec
constant `revolver_latency = 11` (coder 014, §8 Implemented). research-22 (arch
024) CLOSED the close-floor fairness path (§6c): route both columns through the
in-tree `GEMV` host at a shape-derived footprint; one Go-side `n_size` fix + the
`_run_upmem` seam — specced for coder task 025.

---

## 1. Problem statement

UPMEM is the *one* target where the abstraction is supposed to move the number
(TASK_DESCRIPTION §"indicative" note, line 104-107): the win is **tasklet
tiling** — Exo fills the DPU's 14-stage / 11-revolver pipeline with many
fine-grained tasklets (`nt=16`), a lever Cinnamon's `nt=1` cannot legalize.
Phase 0 reproduced this lever on our path: per-row n=1024 went `236,881 cyc
(nt=1) → 41,490 cyc (nt=16)`, a **5.71x lever** matching Shiran's 5.80x
(README §2). So the lever is real and live on our simulator.

The architectural problem: **today nothing in Tenon's decision path knows about
tasklet count.** Three concrete gaps, all confirmed by source read:

1. **Enumerator blind.** `_upmem_enumerate` (`spmw_autoschedule.py:421-471`)
   enumerates exactly two `Placement`s — `acc` in WRAM vs `acc` in GPR. Tasklet
   count is not a candidate axis.
2. **Cost model blind.** `_upmem_kernel_cycles`
   (`spmw_cost_models.py:408-433`) sums `per_op.cycles * inner_loop_iters` over
   matches. It has **no tasklet term** — both enumerated candidates score
   identically w.r.t. parallelism, so argmin cannot prefer the fast schedule.
3. **Runner pins the lever to its worst value.** `_run_upmem`
   (`spmw_codegen.py:2421-2422`) hardcodes `--num_tasklets 1` and
   `--data_prep_params 1024`. This is the existing open tension **T9**
   (`SPMW_ARCHITECTURE.md:365`). The simulator we beat Cinnamon with is being
   driven at Cinnamon's `nt=1` operating point.

The lever that produces the entire UPMEM win is hardcoded OFF. That is the
ruling's target.

---

## 2. Considered options for *where the tasklet lever lives*

### Option A — codegen-only: derive `num_tasklets` inside `_run_upmem`.
Compute a tasklet count from operand shapes and pass it to the CLI. Smallest
diff. **Rejected.** This puts the *decision* in codegen, which the
anti-hardcoding gate forbids (TASK_DESCRIPTION line 250-252: "Codegen may
branch on `Placement` *fields* … but never re-derive or override the
autoscheduler's choice"). It also makes the choice un-priced: argmin never sees
it, so we can't show the win is *earned* by the cost model. Fails the audit.

### Option B — enumerate tasklet count as a `Placement.extra` lever, price it in the cost model, materialize it in codegen (mirror of SPEC-026).
The lever becomes a real, ranked decision. **Chosen.** Exactly the shape of the
resolved Samsung batched-GEMV beat (T16): shape-derived attribute on the match,
≥2 enumerated candidates differing only in the lever, cost model that prices
each from `target.*` constants so argmin organically picks the parallel one,
codegen that reads the chosen field and threads it to the runner. Survives the
anti-hardcoding gate because every step is an expression over shapes + spec
constants.

### Option C — make tasklet count a LinearLayout out-dim.
Model the tasklet axis as a layout dimension and let `optimal_swizzle` assign
it. **Rejected for this cycle.** The tasklet axis is a *parallelism / scheduling*
choice (how many revolver slots to fill), not a memory-coordinate swizzle. The
existing UPMEM enumerator already documents (`spmw_autoschedule.py:442-448`)
that the DPU storage classes are "discrete named storage … not coordinates on a
linear address space, so LinearLayout has no out_dim that names this choice."
The same reasoning applies to the tasklet count. Forcing it into the layout
algebra would be a category error. Revisit only if a future workload needs
*per-tasklet data partitioning* (then the tasklet axis is genuinely a layout
shard); tracked as a new tension T19 below.

---

## 3. Chosen design (Option B), step by step

### 3.1 MATCH — stamp the reduction trip count on the match (NO `allo/ir/` edit)

The tasklet lever's denominator is the **per-DPU reduction work** — the inner-K
loop the tasklets stripe across. This is already on the trace: the same
`enclosing_loops[-1]` bound that `_upmem_kernel_cycles` reads at
`spmw_cost_models.py:426-429`, and the same bound SPEC-019 already threads to
codegen as `pending_k_bound`. So the structural quantity exists; we only need a
spmw-local resolver to name it.

- Add `def _trace_reduction_trip(matches) -> int | None` in
  `spmw_autoschedule.py` (spmw-local; **no shared-file edit**). It returns the
  innermost enclosing-loop bound of the reducing match (`accumulates=True`),
  i.e. the per-DPU K. Mirror of SPEC-026's `batch_dim(match)` structural
  resolver. If absent, returns `None` and the enumerator falls back to the
  single `nt=1` candidate (parity with today).
- **FORBIDDEN:** `allo/ir/builder.py` / `allo/ir/infer.py` edits. SPEC-026 §1.4
  proved the existing match path carries loop bounds additively; the same path
  serves here. If a future workload's reduction bound is genuinely not on
  `enclosing_loops`, that escalates back to architect as a new task — never an
  ad-hoc shared edit.

### 3.2 ENUMERATOR — add the tasklet-count lever as a `Placement.extra` cross

In `_upmem_enumerate`, cross the existing 2 acc-placement candidates with a
small set of tasklet-count candidates, producing the candidate list argmin
ranks. The candidate *set* must be derived, not a literal:

- `Placement.extra["n_tasklets"]` (int, default 1): the number of tasklets the
  C body strides over. Default 1 == today's behaviour (T9 floor preserved).
- The candidate values are `{1, T_max}` at minimum (≥2-candidate discipline,
  SPEC-009), where `T_max` is the **tasklet-fanout of the DPU unit** read from
  the target tree — i.e. the `tasklet` unit's `mapping` (`[16]` in the fixture,
  `spmw_autoschedule.py` can reach it via the same unit-tree fanout walk
  SPEC-011 used for `n_workids`). It is **not** the literal `16`; it is
  `target`-derived (`build_upmem_target` line 422 `@allo.unit(mapping=[16])`).
  Optionally also enumerate the divisors of `T_max` that evenly tile the
  reduction trip (so a short reduction does not over-subscribe), all expressed
  over `_trace_reduction_trip` and the unit fanout.
- Each candidate is a real, materializable `Placement` (a tasklet-strided C
  body the runner can drive at that `--num_tasklets`), not a label codegen
  special-cases (gate: TASK_DESCRIPTION line 192-196).

### 3.3 COST — price the tasklet lever from target-spec constants (CLOSED by research-021)

Research-021 (`work/reports/021-research-upmem-pipeline-amortization.md`) closed
the deferred coefficient by reading the uPIMulator source and decomposing the
Phase-0 per-cause cycle breakdown. **The saturating divisor is the revolver
scheduling window `R = 11`, NOT the 14-stage pipeline depth `D = 14`, and
`fill_drain = 0`.** The lever comes from the round-robin *thread scheduler*
(`thread_scheduler.go:85-112`, `dpu.go:179-182`): a tasklet that issues resets
its `issue_cycle` to 0 and cannot re-issue until it has aged `R = 11` cycles, so
a single tasklet issues at most once per 11 cycles; `T` tasklets round-robined
sustain one-issue-per-cycle once `T ≥ R`. Speedup therefore saturates at
`min(T, R)` — at the revolver window, not the tasklet fanout (16) or the
pipeline depth (14). The measurement confirms this exactly: `breakdown_etc /
num_instructions = 209,864 / 20,985 = 10.0001 = R-1` revolver-idle cycles per
issued instruction at `nt=1`, read off independently of the 5.71 ratio it then
reproduces.

`_upmem_kernel_cycles` currently returns `S = sum(per_op.cycles * inner_iters)` —
the **issue count** (instruction count), `T`-independent. The single-tasklet
wall is `S * R` (every instruction pays the full revolver window); as `T` rises,
only the hideable `(R-1)`-cycle idle shrinks, by `min(T, R)`; the irreducible
1-cycle-per-issue floor `S` never shrinks. The closed form:

```
R = target.revolver_latency                # = 11, NEW target-spec constant
T = placement.extra.get("n_tasklets", 1)
kernel_cycles = S + ceil( S * (R - 1) / min(T, R) )
```

- At `T = 1`: `S + S*(R-1) = S*R` — the full single-tasklet revolver cost.
- At `T ≥ R`: `S + ceil(S*(R-1)/R) ≈ S*(2 - 1/R)` — throughput saturates; adding
  tasklets past `R = 11` does nothing, exactly as measured.
- **`fill_drain = 0`** — no separate term. Pipeline fill/drain is an `O(D)`
  constant dwarfed by an `S = O(10^4)` reduction and changes no ranking; pricing
  it would be an unsupported fudge. `D = 14` does **not** enter the steady-state
  term at all.

**Validation (anti-hardcoding):** with `S = 20,985` and `R = 11`, the model gives
`T=1 → 230,835` (vs measured 236,881, −2.6%, the gap is `breakdown_dma` priced by
the MRAM move pass) and `T=16 → 40,063` (vs 41,490, −3.4%), for a **modelled lever
of 5.762 vs measured 5.709 (+0.9%)**. The lever is a function of `R` only —
scaling `S` by any reduction trip (256…4096) leaves it at 5.762 — so the model
contains no `1024`/`16`/`5.71` literal, only `R = 11`. The modelled-speedup knee
is at `T = 11` (1.83 at T=2, 4.89 at T=8, 5.50 at T=10, 5.762 at T≥11), confirming
the bound is the revolver, not 14 or 16.

**Target-spec constant — exactly one.** Add `revolver_latency = 11` to the UPMEM
fixture (`build_upmem_target`, on the `dpu` or `tasklet` unit), sourced to the
uPIMulator default `num_revolver_scheduling_cycles` (`src/main.go:115`) — same
provenance class as the existing `LD_MRAM cycles=1000 ← HPCA-2024 Table 2`. **Do
NOT add `pipeline_depth`**: it is unused by the lever and would be a dangling
constant inviting a future fudge.

- Parity: at `T = 1` the expression returns `S * R`, a **uniform positive scale**
  of today's `S`. The acc WRAM-vs-GPR argmin (the only existing UPMEM decision)
  is invariant under a uniform scale, so no current ranking moves; the term is
  gated inside `_upmem_kernel_cycles` so no other backend is touched. Default
  `n_tasklets = 1` preserves the T9 floor.
- Cost-fn signature: the closure must see the candidate's `n_tasklets`. Per the
  research recommendation, `cost_fn(trace, layout, placement=None)` reads
  `placement.extra.get("n_tasklets", 1)`; `placement=None` falls back to `T=1`
  (today's call shape stays valid — additive kwarg).

### 3.4 CODEGEN / RUNNER — materialize the chosen tasklet count (resolves T9)

- `UPMEMCtx` already emits a `tasklet_id = me()` strided body
  (`spmw_codegen.py:866-879`). Codegen reads `placement.extra["n_tasklets"]`
  (a *field*, not a re-derivation) and stages it onto the ctx so the runner can
  pass it.
- `_run_upmem` replaces the hardcoded `"--num_tasklets", "1"`
  (`spmw_codegen.py:2421`) with the chosen value from the compiled placement.
  `--data_prep_params` (line 2422) similarly becomes the reduction trip from
  `_trace_reduction_trip`, not the literal `1024`. **This is the concrete
  resolution of tension T9.**
- Codegen must NOT re-derive the count from the workload identity; it only reads
  the field the autoscheduler chose (anti-hardcoding gate line 250-252).

### 3.5 The +1.12x harness offset is NOT in scope for this lever

Phase-0 flagged a consistent ~1.12x *multiplicative* offset on absolute cycles
(README §2): our generic TENON drop-slot host (extra `DPU_INPUT_ARGUMENTS`
transfer + VA-shape data prep) vs Shiran's bespoke `EXO_GEMV` host. This is a
**harness artifact, not a kernel/cost-model issue**, and the **tasklet lever is
measured as a ratio** (5.71x), which is offset-invariant. Ruling: the cost-model
lever work is independent of the offset. Retiring the offset to a byte-exact
absolute claim is a *separate harness task* (port the bespoke host), tracked
under T8a/T8b. The floor gate (TASK_DESCRIPTION line 118) is `tenon_wall ≤
baseline_wall`; the offset *raises* our absolute number, so it works against us
— meaning if Tenon meets the floor despite the offset, the win is conservative.
We do not paper over it; we measure the lever as a ratio and document the offset
in every UPMEM cell, exactly as Phase 0 did.

**UPDATE (§6c, arch 024):** research-22 later showed this offset is *common-mode*
(both the Tenon AND the Exo column ride the same TENON slot), so it cancels in the
Tenon-vs-Exo verdict — the §3.5 ratio-invariance argument is confirmed. §6c then
goes further and removes it from the absolute number too, by routing both columns
through the bespoke in-tree `GEMV` host (NOT by porting Shiran's `EXO_GEMV`, and
NOT by arithmetic normalization). So §3.5's "separate harness task" is now §6c's
ruling, scoped to the in-tree GEMV host rather than a Shiran port.

---

## 4. The FFN scope ruling

**Question (from coder 005, report line 88-97):** is porting Shiran's Exo/Cinnamon
multi-phase selector hosts (`EXO_FFN_1PD`, `CINM_FFN`) in scope for this cycle,
or do the UPMEM FFN cells stay BLOCKED-with-reason?

**Ruling: the UPMEM FFN cells stay BLOCKED-ON-HARNESS, with reason, for this
cycle. Porting the selector hosts is OUT OF SCOPE.** Rationale:

1. **It is a harness port, not a Tenon mechanism.** The blocker is a missing
   *driver*: both ref FFN kernels are multi-phase MRAM-selector kernels needing
   a bespoke per-phase host (Exo `ffn_1pd_8` reads a layer selector at
   heap+32768; Cinnamon `CINM_FFN` has a 4-phase selector at heap+4096). Porting
   those hosts is uPIMulator-tree harness engineering inside
   `experiments/simulators/uPIMulator/`, not a change to any of the three
   mechanism budgets (layout / enumerator / cost model). It cannot be the source
   of an *earned* abstraction win, so it does not advance this cycle's thesis.
2. **The cycle's UPMEM thesis is already carried by gemv.** The tasklet lever —
   the whole point of "UPMEM is the one target where the abstraction moves the
   number" — is demonstrated on gemv (the §3 design). FFN would re-demonstrate
   the *same* lever through a much larger harness surface.
3. **Multi-phase selector lowering is itself an unsolved design question.**
   Driving a phase selector deterministically intersects T8a (UPMEM envelope is
   single-template, VA-shape) and T8b (`tenon.go` data-prep VA-hardcoded). A
   faithful FFN host needs the parameterized envelope T8a/T8b defer. So even the
   *harness* is blocked on unresolved tensions; porting it now would be
   premature and would likely be redone once T8a/T8b land.
4. **Never fabricate a number.** Coder 005 correctly refused to quote an FFN
   cycle count (a dry run gave 43,071 cyc on an undefined selector path). The
   BLOCKED-with-reason verdict is the honest one and is explicitly permitted by
   the exit gate (TASK_DESCRIPTION line 152-153 "If red, mark BLOCKED").

**What the FFN cells carry through the exit gate:** `BLOCKED-ON-HARNESS` in
`MANIFEST.tsv` (already recorded for `upmem-ffn-exo` and `upmem-ffn64-cinm`),
with the reason = "multi-phase selector needs bespoke host port; deferred behind
T8a/T8b envelope parameterization." This is distinct from the SDK↔VM block that
stops Shiran (we cleared that — PrIM VA byte-exact, README §1). The
unified-results table (task 017) must show the FFN cells as BLOCKED-with-reason,
not as a failure and not as a fabricated win.

**Future path (not this cycle):** when T8a/T8b land a shape-parameterized UPMEM
envelope + data-prep, a follow-up harness task can port the selector hosts and
unblock the FFN cells. Recorded as the resolution path on T8a/T8b.

---

## 5. Research-021 — CLOSED

The cost coefficient deferred here is now resolved by research-021
(`work/reports/021-research-upmem-pipeline-amortization.md`, folded into §3.3
above). The headline: the saturating divisor is the **revolver scheduling window
`R = 11`** (uPIMulator `num_revolver_scheduling_cycles`, `src/main.go:115`), NOT
the 14-stage pipeline depth; `fill_drain = 0`; exactly **one** new target-spec
constant `revolver_latency = 11`; cost form `S + ceil(S*(R-1)/min(T,R))`,
reproducing the 5.71x lever within 0.9% (model 5.762), saturating at `T = R = 11`,
invariant across reduction trips. The coder COST sub-task (§6) is now fully
specced and unblocked alongside MATCH / ENUMERATOR / runner-wiring.

---

## 6. File-level change list (for the eventual coder SPEC)

| File | Change | Additive? | Gate |
|---|---|---|---|
| `spmw_autoschedule.py` | new `_trace_reduction_trip`; `_upmem_enumerate` crosses acc-placements × `extra["n_tasklets"]` candidates from unit fanout | additive (default `n_tasklets=1` == today) | `tests/spmw/test_target_upmem.py`, new `test_upmem_tasklet_candidates` |
| `spmw_cost_models.py` | `_upmem_kernel_cycles` returns `S + ceil(S*(R-1)/min(T,R))` with `R=target.revolver_latency`, `T=placement.extra.get("n_tasklets",1)`; cost-fn grows additive `placement=None` kwarg | additive (T=1 → `S*R`, uniform scale, parity) | new `test_upmem_cost_prefers_parallel` + `test_upmem_cost_t1_parity`; must not move any other backend's argmin |
| UPMEM target spec (in user/skill fixture, `build_upmem_target`) | add **exactly one** constant `revolver_latency = 11`, sourced to uPIMulator `num_revolver_scheduling_cycles` (`src/main.go:115`); **do NOT add `pipeline_depth`** | additive new field | `test_upmem_target_builds` |
| `spmw_codegen.py` | `UPMEMCtx` reads `extra["n_tasklets"]`; `_run_upmem` threads it + reduction trip to CLI (resolves T9) | replaces 2 hardcoded literals with placement-derived values | `test_run_upmem_uses_tenon_slot_and_no_proxy` |

**Shared-file blast radius:** all four files are shared (the blast-radius rule,
TASK_DESCRIPTION line 174-178). Every iteration touching them re-runs the full
SPMW pytest suite + every other in-scope target's passing cells. The T=1 /
`n_tasklets`-default-1 parity is the rollback story: with the default, all four
backends behave byte-identically to today, so a `--num_tasklets` regression on
FPGA CI is impossible (FPGA/AIE never reach `_run_upmem`). Upstream
`tests/dataflow/` and `tests/customize/` do not import any `spmw_*` module, so
they are not gated by these edits (consistent with the existing
`spmw_linear_layout` boundary note).

**Anti-hardcoding provenance chain (for task 016 audit):** layout step =
acc-placement Placement (unchanged); enumerated candidate = `n_tasklets ∈ {1,
T_max}` where `T_max` = tasklet-unit fanout from the target tree; cost
expression = `S + ceil(S*(R-1)/min(T,R))` over `S = _trace`-derived issue count
and `R = target.revolver_latency` (= 11, from uPIMulator
`num_revolver_scheduling_cycles`); winner = argmin. The lever is a function of
`R` only, so it is identical (5.762) across every reduction trip — no
`16`, `1024`, `nt=8`, `209`, `73290`, `5.71` literal in any decision path.

---

## 6b. Open question for architect (raised by coder 014)

**The §3.4 instruction "`--data_prep_params` becomes the reduction trip"
empirically INVERTS the lever and must be re-scoped.**

Coder 014 implemented num_tasklets threading (the T9 lever) and the full
cost/enumerator/spec stack, then measured the real uPIMulator behaviour on
the `build_mlp_workload` gemv trace (reduction trip = 256):

| `--data_prep_params` | nt=1 | nt=8 | nt=16 | lever (nt1/nt16) |
|---|---|---|---|---|
| 1024 (prior literal) | 367,667 | 99,237 | 102,405 | **3.59x (saturates ~nt8)** |
| 256 (= reduction trip, per §3.4) | 92,611 | 95,154 | 99,504 | **0.93x (INVERTED)** |

Root cause (verified against `uPIMulator/src/main.go:93` + the per-PrIM
assemblers, e.g. `prim/gemv.go:34 this.m_size = DataPrepParams()[0]`):
`data_prep_params` is the **total per-DPU input buffer size** the strided
byte-loop (`spmw_codegen.py:886 byte_index += BLOCK_SIZE * NR_TASKLETS`)
partitions across tasklets — it is NOT a reduction-trip / inner-K knob.
Shrinking it to 256 drops the per-DPU input below the tasklet-amortisation
threshold, so adding tasklets only adds barrier/startup overhead. The
design's Phase-0 5.71x lever was itself measured at `data_prep 1024`
(README §2), so §3.4's "= reduction trip" coupling contradicts the very
measurement it is meant to reproduce.

**Coder decision (conservative, escalated):** thread `--num_tasklets` from
the chosen placement (the load-bearing lever, §7's stated T9 resolution)
but **hold `--data_prep_params` at the prior literal 1024**, NOT the
reduction trip. This keeps the cell GREEN (autoscheduled nt=16 beats nt=1
by 3.59x at dp=1024) and is correctness-neutral (the reduction is correct
at any input size). The reduction trip is still staged on the ctx
(`UPMEMCtx.reduction_trip`) so the architect can re-scope without another
codegen edit.

**Decision needed:** how should `--data_prep_params` be set? Options:
(a) leave at a fixed amortising literal (current coder choice);
(b) derive the per-DPU input size from the workload's *outer* (M/VA) shape,
not the reduction trip — this is the dimension the byte-loop actually
strides, and a proper shape-derived value would replace the literal without
inverting the lever;
(c) the `1024` is itself a hardcode the anti-hardcoding gate may flag —
(b) is the gate-clean resolution but needs the outer shape threaded onto
the trace (possible new MATCH sub-task). This is left for architect ruling;
the lever itself (num_tasklets) is correct and earned.

**RESOLVED by §6c (arch 024, research-22).** Option (b) is taken, in its proper
form: route the gemv cell through the in-tree bespoke **`GEMV`** host (not the
VA-shaped `TENON` slot) with a shape-derived `(m_size, n_size)` footprint. This
simultaneously retires the `1024` literal (gate-clean), fixes the comparison
fairness (§6c), and removes the lever-inversion (the GEMV host's data-prep lays
out an `m_size`-row × `n_size`-reduction matrix, so the byte-loop strides the
correct footprint and the lever no longer inverts). See §6c.

---

## 6c. Close-floor ruling — route both columns through the in-tree GEMV host (arch 024)

Research-22 (`experiments/reports/22-upmem-gemv-host-fairness.md`) decomposed the
1.234x UPMEM gemv gap from the committed, exactly-closing breakdown counters. The
load-bearing finding: the ~1.13x harness offset is **common-mode** — BOTH the
Tenon column AND the MANIFEST Exo baseline (82,980) already ride the *same*
generic `TENON` drop-slot; neither rides Shiran's bespoke host. So the offset
**cancels in the Tenon-vs-Exo verdict** and only separates our absolute yardstick
from Shiran's published 73,290. The offset lives entirely in DMA + revolver-stall
overhead (not in `breakdown_run` instructions) and is provably non-normalizable by
any constant (additive `24,310 ≠ 4,845`; multiplicative `1.1144 ≠ 1.1322`).

**Ruling (the close-floor approach):**

1. **Physical fix, not arithmetic.** No constant subtraction / division of either
   column. Arithmetic normalization is REJECTED — both because it does not work
   (the offset is non-constant) and because a divisor fit to Shiran's totals is
   exactly the benchmark-ratio back-fit the anti-hardcoding gate forbids
   (TASK_DESCRIPTION line 248-250). The fix is to run **both columns through one
   host that generates one overhead path.**

2. **Reuse the in-tree `GEMV` host — do NOT port Shiran's `EXO_GEMV`.** A bespoke
   gemv data-prep already exists and is registered in our uPIMulator
   (`benchmark/GEMV/` with `dpu/task.c` + `CMakeLists.txt` drop-slot, assembler
   map `assembler.go:41 this.assemblables["GEMV"] = new(prim.Gemv)`,
   `prim/gemv.go`). It carries the gemv argument struct (`n_size`, `nr_rows`,
   `max_rows`) and the gemv MRAM layout. Porting Shiran's *exact* host would only
   buy absolute-yardstick equality with his 73,290 — strictly more work (his host
   is not in our tree) and **not required for a fair MATCH verdict**, which needs
   only that the two columns share *a* host. Downgrade exact-73,290 to an optional
   absolute-number nicety, not a gate.

3. **This is a HARNESS/DRIVER change, NOT a `spmw_*.py` decision-path change.**
   The only Tenon-side edit is at the `_run_upmem` seam (`spmw_codegen.py:2388`):
   which benchmark slot and which data-prep params the runner invokes. The
   autoscheduler's *decision* (the `n_tasklets` lever, the acc placement) is
   untouched — codegen still only reads staged fields, never re-derives. The
   `(m_size, n_size)` footprint is read from the workload shape already staged on
   the ctx (`UPMEMCtx.reduction_trip` = `n_size`; the outer row count = `m_size`),
   so it is shape-derived, not literal. No new enumerator candidate, no cost-model
   change, no layout change. The §3 tasklet lever (and its revolver cost model)
   stands exactly as specced; §6c only changes the *measurement harness* the lever
   is measured through.

4. **The `n_size = 64` hardcode in the GEMV host is the one Go-side fix.**
   `prim/gemv.go:34-35` reads `m_size = DataPrepParams()[0]` but pins
   `n_size = 64`. Since `DataPrepParams()` already returns a comma-separated
   `[]int` (`command_line_parser.go:99-111`), the fix is to read
   `n_size = DataPrepParams()[1]` with a `len()`-guarded fallback to 64 (so the
   native PrIM GEMV invocation, which passes one param, is byte-for-byte
   unchanged). This is additive: a second optional data-prep element. No CMake or
   assembler-registry edit — `GEMV` is already registered and already has a
   `dpu/task.c` drop-target.

5. **This retires the §6b `data_prep_params = 1024` design-debt.** Under the GEMV
   host, `data_prep_params = "<m_size>,<n_size>"` carries the gemv footprint
   explicitly (m rows/DPU, n reduction), so the runner no longer abuses the VA
   input-size knob with a `1024` literal. The lever-inversion §6b documented
   (which came from shrinking the VA input below the amortization threshold) does
   not occur because the GEMV host sizes the per-DPU input from the actual matrix
   footprint, not from a flat VA buffer.

6. **The yardstick RE-FREEZES at the GEMV-host level (verifier note).** The
   MANIFEST `upmem-gemv-exo = 82,980` cyc is the *VA-slot* (TENON drop-slot) Exo
   number. Once both columns move to the GEMV host (§6c), that 82,980 is no longer
   the comparison baseline — the verdict is `tenon_cyc` vs the **Exo `task.c` run
   through the same GEMV host** at the same `(m_size, n_size)`, which coder 025
   measures fresh. The `tenon_progress.tsv` ratio must be computed against that
   re-measured Exo-on-GEMV-host number, NOT against 82,980. Coder 025 must record
   the new Exo-on-GEMV-host yardstick in the MANIFEST (a new row or an annotated
   update of `upmem-gemv-exo`, with the host noted) so the re-freeze is auditable.
   Task 025's one-liner cites the 82,980 figure; it is superseded by this
   re-measured yardstick — the `±0%` parity + instruction-count guard (the §6c
   test) is the real PASS gate, not `≤ 82,980`.

**File-level change list (coder task 025):**

| File | Change | Additive? | Gate |
|---|---|---|---|
| `uPIMulator/.../src/assembler/prim/gemv.go` | `n_size = DataPrepParams()[1]` with `len()`-guarded fallback to 64 (line 35) | additive (1-param native call unchanged) | native PrIM `--benchmark GEMV` 1-param run byte-identical (back-compat) |
| `experiments/allo/allo/spmw_codegen.py` `_run_upmem` (`:2388`) | route gemv-shaped traces through `--benchmark GEMV --data_prep_params "<m_size>,<n_size>"` (both shape-derived from ctx); VA-shaped traces keep `--benchmark TENON` | additive branch (non-gemv path unchanged) | `tests/spmw/test_target_upmem.py` (TENON path stays green); new `test_upmem_gemv_host_routing` asserts GEMV slot + shape-derived params, no `1024`/`64` literal |

**Test that proves the fair MATCH (coder task 025 acceptance, from report 22 §5.3):**
both columns MUST use the SAME `--benchmark GEMV` slot at the SAME
`(m_size, n_size)`:

```
exo_cyc   = run_gemv_host(kernel=exo_task_c,   m_size=<M/DPU>, n_size=<K>, nt=<chosen>)
tenon_cyc = run_gemv_host(kernel=tenon_task_c, m_size=<M/DPU>, n_size=<K>, nt=<chosen>)
assert abs(tenon_cyc - exo_cyc) / exo_cyc <= 0.0     # UPMEM noise band = +-0% (007: deterministic Go sim)
assert tenon_run == exo_run                          # breakdown_run instruction-count parity (anti-flattering guard)
```

The first assertion is the MATCH at the revolver floor; the second
(instruction-count parity) is the anti-flattering guard — it proves the parity is
earned by the kernel issuing the same work, not by two different overhead paths
coincidentally summing to the same total. `m_size`/`n_size`/`nt` in the test are
the values *derived from the gemv shape*, never literals keyed to the benchmark
(the test reads them from the workload, satisfying the anti-hardcoding gate).

**Anti-hardcoding note (task 016):** the `(m_size, n_size)` passed to the GEMV
host is the workload's row-count-per-DPU and reduction length, threaded from the
trace/ctx — not the literals 2/1024/64. The `n_tasklets` argmin is unchanged.
Codegen branches on the trace *shape* (gemv vs VA, a structural property), never
on a workload name/hash. The GEMV-host route is a measurement harness; it carries
no schedule decision.

**Rollback story (FPGA/AIE blast radius):** the `gemv.go` change is gated by
`len(DataPrepParams()) >= 2`, so any existing 1-param GEMV invocation is
byte-identical; the `_run_upmem` branch is gated on the gemv trace shape, so the
VA/MLP TENON path is unchanged; neither edit is in a shared `spmw` decision file's
ranking logic, and FPGA/AIE never reach `_run_upmem`. Reverting is dropping the
`[1]` read back to the `64` literal and the `_run_upmem` branch back to
unconditional TENON.

---

## 7. Concrete commitment

- The tasklet lever lives on **`Placement.extra["n_tasklets"]`**, enumerated as
  `{1, tasklet-unit-fanout}` (≥2 candidates), priced by
  **`S + ceil(S*(R-1)/min(T,R))`** with `R = target.revolver_latency = 11`
  (revolver window, NOT pipeline depth) and `fill_drain = 0`, materialized
  by `UPMEMCtx` + `_run_upmem` — which **resolves tension T9** by replacing the
  hardcoded `--num_tasklets 1` / `--data_prep_params 1024`.
- The UPMEM target spec gains **exactly one** new constant `revolver_latency =
  11` (uPIMulator `num_revolver_scheduling_cycles`, `src/main.go:115`);
  `pipeline_depth` is deliberately NOT added.
- `allo/ir/builder.py` and `allo/ir/infer.py` are **FORBIDDEN** for this lever
  (reduction trip rides existing `enclosing_loops`, per SPEC-026 §1.4 proof).
- **UPMEM FFN cells stay BLOCKED-ON-HARNESS**; porting the selector hosts is OUT
  OF SCOPE this cycle (harness engineering, blocked behind T8a/T8b, not a
  mechanism win).
- research-021 CLOSED the cost coefficient: model reproduces the measured 5.71x
  lever within 0.9% (5.762), saturates at `T = R = 11`, invariant across
  reduction trips. The COST sub-task is now fully specced for coder 014.
- **Close-floor (§6c, arch 024, research-22):** the fair Tenon-vs-Exo MATCH is
  reached by routing BOTH columns through the in-tree bespoke `GEMV` host at an
  identical shape-derived `(m_size, n_size)` footprint at the `_run_upmem` seam —
  NOT by arithmetic normalization (the 1.13x offset is common-mode and
  non-constant), and NOT by porting Shiran's `EXO_GEMV` (unnecessary for the
  verdict). One Go-side fix: `gemv.go` `n_size` from `DataPrepParams()[1]`
  (`len`-guarded fallback). Gate: `±0%` cycle parity + `breakdown_run`
  instruction-count parity. This retires the §6b `data_prep_params=1024` debt and
  is a harness/driver change carrying no schedule decision.

## 8. Implemented (coder 014)

- **MATCH**: `_trace_reduction_trip` + `_tasklet_fanout` + `_upmem_tasklet_candidates`
  in `allo/spmw_autoschedule.py` (spmw-local; no `allo/ir/` edit).
- **ENUMERATOR**: `_upmem_enumerate` crosses 2 acc-placements × `{1, T_max}`
  `extra["n_tasklets"]` (T_max = tasklet-unit fanout, target-derived).
- **COST**: `_upmem_kernel_cycles` returns `S + ceil(S*(R-1)/min(T,R))`,
  `R = target.revolver_latency`, `T = layout.extra["n_tasklets"]`; reads the
  count off the `layout` arg (the existing call shape passes the Placement
  there — same convention Samsung/APU use — so no separate `placement=` kwarg
  was needed).
- **TARGET SPEC**: new `allo.const` primitive in `spmw_target.py` (+ `__init__`
  export); UPMEM fixture gains exactly one constant `revolver_latency = 11`.
  `pipeline_depth` NOT added.
- **CODEGEN**: `UPMEMCtx.n_tasklets` / `reduction_trip` staged by
  `_walk_and_emit`; `_run_upmem` threads `--num_tasklets` from the chosen
  placement (resolves T9). **DEVIATION**: `--data_prep_params` held at 1024,
  NOT the reduction trip — §6b documents why (the trip coupling inverts the
  measured lever; `data_prep_params` is the input-buffer size, not the inner-K
  knob). Escalated for architect re-scope.
- **MEASURED** (uPIMulator, `build_mlp_workload` gemv trace): autoscheduled
  picks nt=16 → 102,405 cyc (292.6 us @350MHz); forced nt=1 → 367,667 cyc;
  **lever 3.59x**, earned by argmin. Logged to `tenon-progress.tsv` iter 7/8.
- **TESTS**: `tests/spmw/test_upmem_tasklet_tiling.py` (12 tests) green;
  full `tests/spmw/` suite 180 passed. Reproduction driver:
  `tests/spmw/_measure_upmem_tasklet.py`.

## 9. Implemented (coder 027 — FFN loop-to-floor, §4 cells)

The §4 ruling held the FFN cells BLOCKED-ON-HARNESS; task 026 then ported the
bespoke selector hosts (EXO_FFN_1PD / CINM_FFN). Task 027 ran both through
RUN→MEASURE→PROFILE→ITERATE and applied the §3 tasklet lever **where it helps**.

- **LEVER PROBE (the iterate step).** Swept `--num_tasklets` on each FFN layer-2
  basis leg. The §3 lever does **not** move either FFN number, for kernel-
  structural reasons (no shared `spmw_*.py` edit was warranted):
  * **Exo FFN** statically partitions its 1024-dot across exactly NR_TASKLETS=8
    (stride `128*tid`; `reduce1` sums the literal `partial[0..7]`). It is correct
    only at nt=8; nt<8 under-counts the dot; nt>8 strides OOB and the sweep RISES
    (42,918 @nt=8 → 53k–56k @nt=16). The parallel point is baked into the kernel
    layout, not a free runtime lever — nt=8 already IS the floor the Exo reference
    operates at.
  * **Cinnamon FFN** layer-2 is a one-output-per-PU serial 256-dot with no
    tasklet striping; nt>1 **panics** the sim. nt=1 is its correct floor —
    exactly the §1 statement "a lever Cinnamon's nt=1 cannot legalize."
- **FLOOR / VERDICT (honest, deterministic, re-runnable).** vs README §3 layer-2
  basis yardstick:
  * Exo  256-1024-256  nt=8 = **42,918 cyc / 122.6 us** vs ref 47,250 / 135 us →
    ratio **0.908 → BEAT**.
  * Cinn 64-256-64     nt=1 = **58,977 cyc / 168.5 us** vs ref 67,750 / 194 us →
    ratio **0.871 → BEAT**.
  Both legs sit at the parallel floor the reference uses and beat it. Logged to
  `tenon-progress.tsv` iter 11/12.
- **TESTS**: new `tests/spmw/test_upmem_ffn_floor.py` (4) + existing
  `test_upmem_ffn_host_port.py` (6) green; suite collects 197 (was 193).
  Live sweep driver: `tests/spmw/_measure_upmem_ffn.py`; log
  `baselines/cinnamon-exo/upmem/logs/ffn-floor-sweep-027.log`.
- **NO shared-file edit**: the lever does not help FFN, so no
  enumerator/cost/codegen change is justified (would be unearned). The gemv
  lever stack (coder 014) is untouched.

## 10. Implemented (coder 025 — close-floor gemv via shared GEMV host, §6c)

- **GO-SIDE FIX (gemv.go):** `n_size = DataPrepParams()[1]` with a
  `len(...)>=2`-guarded fallback to 64 (native 1-param PrIM GEMV byte-identical).
  **Plus one determinism fix the spec did not anticipate:** the gemv host was the
  lone PrIM kernel missing `rand.Seed(42)` (tenon.go:64 / e5vmul.go:56 / cinm_ffn.go:58
  all seed). Unseeded, two runs of the SAME kernel differed ~1% in BOTH `logic_cycle`
  AND `breakdown_run`, making the §6c ±0% gate impossible. Added `rand.Seed(42)` in
  `Gemv.Init` (same in-tree convention) → fully deterministic. Rebuilt
  `build/uPIMulator` (`go build -o build/uPIMulator ./src`).
- **CODEGEN (spmw_codegen.py):** `_run_upmem` routes a gemv-shaped trace (both
  `ctx.row_count` and `ctx.reduction_trip` staged) through `--benchmark GEMV
  --data_prep_params "<m_size>,<n_size>"` (shape-derived); VA-shaped traces keep
  TENON. New `UPMEMCtx.row_count` staged from the MAC's outer enclosing loop. New
  `UPMEMCtx.get_gemv_kernel_src()` emits a GEMV-host-compatible envelope (gemv arg
  struct + MRAM layout) wrapping Tenon's MAC — required because the VA `get_kernel_src`
  envelope is incompatible with the GEMV host's `DPU_INPUT_ARGUMENTS`. This is
  materialisation, not a decision-path change; the n_tasklets argmin + revolver cost
  model are untouched. Retires the §6b `data_prep_params=1024` debt (gemv route no
  longer uses the VA input-size knob).
- **MEASURED §6c gate (deterministic uPIMulator, MLP gemv trace, shape-derived
  m_size=64 / n_size=256 / nt=16):** BOTH columns through `--benchmark GEMV` at the
  same footprint →
  * Tenon `logic_cycle` = **624,621**; Exo `logic_cycle` = **624,621** →
    `|Δ|/exo = 0.000000` (**±0% EXACT**).
  * Tenon `breakdown_run` = **376,518** == Exo `breakdown_run` = **376,518** →
    instruction-count parity (anti-flattering guard PASS).
  * **VERDICT: fair MATCH at the revolver floor.** Yardstick RE-FROZEN at the
    GEMV-host level (the old MANIFEST 82,980 was the VA-slot number, NOT the baseline);
    new yardstick recorded as `upmem-gemv-exo-gemvhost` in MANIFEST.tsv, progress
    iter 13/14.
- **TESTS:** `tests/spmw/test_upmem_tasklet_tiling.py` grows
  `test_upmem_gemv_host_routing` + `test_upmem_va_route_keeps_tenon_slot` + a
  `row_count` assertion (20 passed with test_target_upmem). Live §6c gate driver:
  `tests/spmw/_measure_upmem_gemv_floor.py`. Exo baseline snapshot:
  `benchmark/GEMV/dpu/task.c.exo-baseline`.
