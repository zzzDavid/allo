# Design 03 — Samsung FFN 256-1024-256: runnable-path ruling

*Architect, 2026-06-17. Task 028 (needs-arch). Inputs: coder-003 report
(BLOCKED-SIM evidence), arch ruling 008 §1.4 (FFN baseline = two-leg
faithful sum), memory entry 010 (Samsung cycle model is shape-fixed),
SPMW_ARCHITECTURE.md T12/T15/T17, goal-check-1 (cell NOT MET). No code;
diagnosis is read-only (probe .npy files in /tmp, stock binary, gdb).*

## Problem statement

The Samsung FFN 256-1024-256 cell is BLOCKED-SIM. The reference
PIMSimulator (`pim_driver --op GEMV`) **cores (SIGSEGV)** at the FFN leg
shapes. Ruling 008 §1.4 specified the FFN baseline as the two-leg
faithful native cycle sum, but that path was never run — the leg shapes
abort. The goal-check requires this in-scope cell closed. Task 028 asks
me to rule one of: (a) a runnable faithful decomposition the reference
sim survives, (b) authorize a sim fix, or (c) escalate as a
genuine non-closable cell needing a user descope or out-of-Phase-0 sim
modification.

## What I verified (do not re-derive)

### The core is in the memory backend during weight PRELOAD, not in compute

gdb backtrace of `M=256 K=1024`:

```
#0 DRAMSim::Bank::write  (Bank.cpp:117)
#1 DRAMSim::Rank::writeSb
#2 DRAMSim::Rank::sendToBank (Rank.cpp:393)
#3 DRAMSim::MemoryController::update
#4 DRAMSim::MemorySystem::update (MemorySystem.cpp:247)
#5 DRAMSim::MultiChannelMemorySystem::actual_update
#6 PIMKernel::runPIM
#7 main (pim_driver.cc:332)   <-- the runPIM() right after preloadGemv(&W)
```

Frame #7 is `pim_driver.cc:332`, the `kernel->runPIM()` that drains the
**weight-preload** transaction queue (`preloadGemv(&W)` is line 331). The
abort is in `Bank::write` — a write transaction whose `(row,col)` address
lands in an inconsistent bank state. It is **not** in
`executeGemv`/`computeGemv`/`readResult`. The leg never reaches the GEMV
math; it dies uploading W.

### Root cause: `preloadGemv` address generation is validated only at the M=4096 design point

`preloadGemv` (PIMKernel.cpp:283-323) walks a column counter `col`
upward across `bShape[1]/num_grfA_ = K/8` input tiles, advancing
bank/group/rank/channel via `changeBank`, bumping `starting_row/col`
only when the channel index wraps. With `NUM_COLS=128`
(`ini/HBM2_samsung_2M_16B_x64.ini`), `num_grfA_=num_grfB_=8`,
`num_total_pim_blocks_ = 8 blocks * 64 chans * 1 rank = 512`, the layout
is closed (no out-of-window `(row,col)`) only when the output dimension
fills exactly one output tile, `output_tile_size = num_grfB_ *
num_total_pim_blocks_ = 4096`. For other M the generated column base
`col = num_output_tiles*num_input_tiles/2*num_grfA_*num_grfB_ +
(j+b)*num_grfB_` (executeGemv:413) and the preload `col` sweep step out
of the validated address region; `addrGenSafe` aliases into a bank/row
that was never opened, and `Bank::write` faults. This is the
shape-fixed-model boundary already recorded in memory entry 010 / T12 /
T15, now localized to `preloadGemv` + DRAMSim bank state.

### The runnable/core boundary, fully mapped (stock binary, this server)

```
        K=128   K=256   K=512   K=1024
M=32    -        -       -       15187
M=64    -        -       -       15251
M=128   -        -       -       CORE
M=256   2813    4435    8041    CORE
M=512   -       4435     -      CORE
M=1024  -       CORE    CORE    CORE
M=2048  -       CORE     -      CORE
M=4096   -      4435     -      15251
```

(`-` = not probed; numbers = `total` cyc; CORE = SIGSEGV.) The FFN legs
are **W1: M=1024, K=256** and **W2: M=256, K=1024** — both in the CORE
region.

### The runnable points near the legs are NON-FAITHFUL, so decomposition fails on faithfulness, not just on coring

Two facts kill every decomposition that would otherwise route around the
core:

1. **M-floor non-scaling.** `M=64 K=1024 -> 15251` is *identical* to
   `M=4096 K=1024 -> 15251`; `M=32 K=1024 -> 15187` is within 0.4%. The
   T15 work formula `ceil(M/4096)*ceil(K/8)` with the `>=1` clamp
   (PIMKernel.cpp:576-579) prices any M<=4096 as one full 4096-row output
   tile. So a 64-row sub-GEMV is billed the full 4096-row cost. Tiling M
   down to a runnable size therefore *overcounts by up to 64x* — it does
   not faithfully price the leg's MAC volume. This is the SAME
   fixed-quantum non-faithfulness fingerprint ruling 008 §1.2 rejected
   for the Cinnamon 222,919 artifact (`1024^2 == 4096x1024`). Adopting it
   for the baseline would re-introduce exactly the detuned model 008
   forbade.

2. **K-tiling does not reach the legs.** `M=256 K=512` runs (8041), so
   W2 (M=256, K=1024) could in principle be 2x K=512 sub-GEMVs +
   host-accumulate. But W1 (M=1024, K=256) **cores at K=256 already**
   (`M=1024 K=256 -> CORE`) — M=1024 cores at every K probed. K-tiling
   cannot fix a leg whose M alone cores. And M-tiling W1's 1024 rows into
   runnable <=512-row blocks re-incurs problem (1): each block is billed
   the 4096-row quantum.

There is no decomposition that is simultaneously (i) runnable and (ii)
faithful to the leg's work for *both* legs.

## Considered options

- **(a-i) M-tiling to runnable sub-GEMVs.** Rejected: non-faithful
  (M-floor over-counts; finding 1). Would re-introduce the 008-forbidden
  fixed-quantum model on the baseline column.
- **(a-ii) K-tiling to K<=512 sub-GEMVs + host accumulate.** Rejected:
  does not reach W1 (M=1024 cores independent of K). Only half-covers W2.
- **(a-iii) Pad each leg's M to 4096 and run the M=4096 design point.**
  Rejected: this prices *every* leg at the 4096-row GEMV cost (15251 for
  K=1024, 4435 for K=256). It is runnable and deterministic but it is the
  4096-row quantum, not the leg's 256/1024-row work — the same
  non-faithful over-count as (a-i), just spelled as padding. The FFN
  total would be `4435 (W1 padded to 4096) + 15251 (W2 padded to 4096) =
  19686` for cycle volume that should be ~1/4 to ~1/16 of that. Not a
  faithful baseline.
- **(b) Authorize a sim fix to `preloadGemv` / DRAMSim bank state so the
  leg shapes run.** Rejected at the architect level — out of Phase-0
  scope and crosses the FORBIDDEN line. `PIMKernel.cpp` preload/compute
  and the DRAMSim core (`Bank.cpp`, `Rank.cpp`, `MemorySystem.cpp`) are
  reference-only (T12, memory 010). A fix is not a localized additive
  flag like `--batch` (SPEC-026); it is a correctness change to the
  validated address-generation layout, which would alter the M=4096
  baseline's transaction stream and risk the frozen gemv 15251 yardstick
  + upstream gtest `PIMKernelFixture.gemv`. That is a user-approved,
  out-of-Phase-0 decision, not an architect authorization.
- **(c) Escalate: the cell cannot be closed faithfully without either a
  user descope or an out-of-Phase-0 simulator modification.** CHOSEN.

## Chosen option + why: (c) ESCALATE — no faithful runnable path exists under Phase-0 scope

The cell is genuinely non-closable as specified, for a reason that is
*structural*, not a harness gap:

- The reference sim's `preloadGemv` address layout is validated only at
  the M=4096 design point (and small toy shapes K<=512 at small M); the
  FFN legs (M=1024/K=256, M=256/K=1024) fault in DRAMSim during weight
  preload.
- Every runnable point near the legs prices work at the M-floored
  4096-row quantum, which is exactly the fixed-quantum non-faithfulness
  ruling 008 §1.2 rejected. So a decomposition that *runs* is not
  *faithful*, and ruling 008 §1.4 demanded a faithful two-leg number.
- Fixing the sim is out of Phase-0 scope and touches FORBIDDEN
  reference/DRAMSim code (T12).

This does NOT reverse ruling 008's per-cell verdict. Ruling 008 §Part 2
Cell 2 already established the FFN's *analytical* verdict: **MATCH at
floor, no fair-beat avenue** (two GEMV legs, W1!=W2, host ReLU = 0 PIM
cyc; activation-residency RESOLVED no-beat, research-019). The mechanism
(per-leg enum+cost) and the honesty argument stand. What is missing is
only the *measured baseline cycle number* for the floor — and that
number is unobtainable on the reference sim at the leg shapes.

### Escalation ask to the user (the orchestrator surfaces this; I do not decide it)

One of the following, user's call (a TASK_DESCRIPTION amendment, since
the planner cannot unilaterally shrink "every in-scope cell"):

1. **Descope Samsung FFN 256-1024-256 from the cross-target floor
   table** for this cycle, citing the non-modifiable reference-sim
   `preloadGemv` shape limit (documented here + memory 010). The Samsung
   *gemv* cell (15251, MET) carries the Samsung floor; the FFN keeps its
   analytical MATCH-at-floor verdict (008 Cell 2) without a measured
   baseline. This is the AiM-contrast-free analog of ruling 009's UPMEM
   FFN descope.
2. **Authorize an out-of-Phase-0 reference-sim fix** to `preloadGemv` /
   DRAMSim bank-state so non-4096 M with K=1024 runs, accepting the build
   + re-validation cost AND the requirement that the M=4096 baseline
   (15251) stays byte-for-byte (gtest `PIMKernelFixture.gemv` green). If
   authorized, I will then spec the fix as a separate design doc; it is
   NOT authorized by this ruling.
3. **Accept the M-padded-to-4096 over-count** (option a-iii) as an
   explicitly-labeled UPPER-BOUND baseline (W1+W2 padded = 19686 cyc),
   NOT a faithful floor, with both columns padded identically (so the
   comparison is still apples-to-apples even though the absolute number
   is loose). This is the only way to put a *runnable* number on the cell
   without a sim fix; it is honest only if labeled UPPER-BOUND and both
   sides ride it. I do not recommend it (it re-imports a fixed-quantum
   number), but it is a coherent user choice if a runnable number is
   required over a faithful one.

My recommendation: **option 1 (descope with documented sim-limit
rationale)**. It is the only choice that keeps the baseline faithful
(ruling 008's standard) and does not touch FORBIDDEN reference code.

## File-level change list

**No `.py` or simulator source changes authorized by this ruling.**

- `experiments/allo/SPMW_ARCHITECTURE.md` T17 — append the FFN
  BLOCKED-SIM sub-finding (this doc's localization + escalation). DONE in
  this task.
- `experiments/baselines/MANIFEST.tsv` — `samsung-ffn` row stays
  `BLOCKED-SIM`; the coder may annotate it with the localized reason
  (`preloadGemv addr layout validated only at M=4096; legs fault in
  DRAMSim Bank::write during weight preload`) — annotation only, no
  number fabricated.
- Pending user decision: if option 2, a new
  `design/04-samsung-preloadgemv-shape-fix.md` + a ready-coder task. If
  option 1 or 3, a TASK_DESCRIPTION amendment by the user, then a
  ready-coder task to record the descope/upper-bound row.

## Test that will prove it works (for whichever option the user picks)

- **Option 1 (descope):** the cross-target results table records
  `samsung-ffn = DESCOPED (sim-limit)` with a pointer to this doc; the
  goal-check's in-scope set is amended to exclude it. No new test; the
  guard is that the Samsung *gemv* cell stays MET (15251) and
  `tests/spmw/test_samsung_faithful_run_path.py` stays green (3/3).
- **Option 2 (sim fix):** the fix's gate is (i) `M=1024 K=256` and
  `M=256 K=1024` run without SIGSEGV and return cycles that *scale* with
  the leg work (not the 4096-floor 15251); (ii) the M=4096 baseline stays
  **byte-for-byte 15251** (regression: re-run the canonical
  `pim_driver --cmds folded_gemv.cmds --faithful`); (iii) upstream gtest
  `PIMKernelFixture.gemv` green.
- **Option 3 (upper-bound):** a test asserting BOTH columns use the same
  M-padded-to-4096 path and the row is labeled `UPPER-BOUND` (not
  `floor`/`REPRODUCED`); the per-leg numbers equal the M=4096 design
  points (4435 + 15251 = 19686).

## Status

PASS (ruling delivered). The cell is escalated to the user as
non-closable under Phase-0 scope without a descope or an out-of-scope
sim fix. No implementation queued — the next action is a user decision,
which the orchestrator surfaces via this ruling + the goal-check FAIL.
This is the honest verdict: I did not invent an FFN number.
