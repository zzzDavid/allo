# Release review: architecture, constants, specialization, gate judgments

This file records the expert review of the release candidate against gates
G1-G8 of the task, the rejected shortcuts, and the remaining limitations. It
is updated at each round; judgments name the revision they apply to.

## Method

Seven read-only source audits (`audit/*.md`) covered every public route:
core target/workload/cost languages and matcher, Samsung HBM-PIM, UPMEM,
SK-Hynix AiM and AttAcc, GSI APU v1, GSI Gemini-II, packaging. Each audit
executed probes against `8160d6d` and labels claims as executed, read, or
suspected. `ISSUES.md` consolidates the findings; `CLAIMS.md` maps the frozen
paper to symbols. Repairs are reviewed by diff before commit; this file notes
what was accepted and why.

## Invariants at stage boundaries (current candidate)

1. **Frontend to matcher.** `customize()` produces retained MLIR with
   `spmw.*` scope attributes on work-item functions. `match_workload` now
   classifies every store of every function (`matched`, `copy`,
   `accumulator_init`, `local`, `unmatched`) and raises
   `UnsupportedWorkloadError` for any unmatched effect. Every accumulating
   match carries `accumulator_init` in {`zero`, `carried`}; anything else
   raises. Invariant: a trace that reaches codegen implements every store of
   the source, and the device may zero an accumulator only when the source
   does.
2. **Matcher to autoscheduler.** Axis roles (`axis_roles`: reduction,
   output, batch, matrix/vector operand, work-axis) are stamped structurally
   from index sets including the store index. Consumers must not re-derive
   them from memref extents or target counts.
3. **Autoscheduler to codegen.** Still per-kernel-group ranking before
   materialization (C-03, C-05); no access-scope legality (C-01). Not yet an
   invariant.
4. **Codegen to runtime.** `RunResult.kind` states what a result evidences
   (`device`, `simulator`, `simulator-timing-only`, `functional`, `virtual`).
   Executed kinds must carry `extra["outputs"]` for every gathered role;
   `CompiledCallable` raises otherwise and never writes caller arrays for
   non-executed kinds. Samsung readback treats all-zero as a value and
   truncation as an error. AiM lowering checks issued MACs against program
   MACs (`AimCoverageError`). UPMEM rejects element types its templates do
   not compute and never truncates on pack or gather. Portable C rejects
   operations on >64-bit integers whose result depends on bits above 63.
   Remaining breach: the Samsung REDUCE conductor regenerates the vendor
   microcode regardless of the selected placement (S-01).

## Accepted repairs (increment 1) and review notes

- Matcher coverage (`spmw_match_engine._FunctionCoverage`). Accepted. Notes:
  the `local` category uses a greatest-fixpoint over local stores and
  requires every reader of a local that reaches a matched op as an operand
  to be a constant or copy; the whole-buffer-fill rule for zero initializers
  requires each dimension indexed by its own `[0, dim)` loop with step 1.
  The host-exempt set for UPMEM host-placed groups couples the matcher to
  `spmw_upmem.classify_upmem_kernel` (import inside the function); acceptable
  for now, flagged for a cleaner "backend executes function verbatim"
  capability declaration. A hand-built trace without `coverage` cannot be
  checked; `compile_for_target` re-matches when a module is available and
  otherwise trusts only the per-match stamps.
- `RunResult.kind`. Accepted. Default `"unspecified"` keeps construction
  sites compiling; `validate()` rejects it on every public route. Twenty-two
  construction sites updated. Known residue: `apu_v1_vector_program` labels a
  Python-source execution as `functional` (it is neither the lowered program
  nor virtual); UPMEM `extra["outputs"]` exposes every argument array;
  Samsung/AiM still tolerate a nonzero simulator exit code when a cycles
  line exists.
- Removal of the Gemini-II source route. Accepted: there was no emission and
  no execution; a finite `NotImplementedError` replaces a success-shaped
  result. `build_apu_v2_target` stays for the typed routes.
- `APUv1Program` backends. Accepted: `virtual` evaluates the retained graph;
  `functional` executes the retained portable C (the same C the ARC core
  runs for scalar programs); native-vector programs reject `functional` at
  compile time rather than silently running Python.
- AiM axis roles. Accepted. The `work_axis` fact is derived by comparing
  operand accesses across work-item functions; an undecidable case raises
  instead of guessing. Forms that used to compile but cannot be proven
  (split-K grids, uneven per-item extents, `y[row0+i] += ...`) now raise;
  this narrows accepted syntax to what the lowering can prove, which the
  task prefers over silent over-issue.
- UPMEM element types and exact packers. Accepted. The type table is a
  template fact (int32 MRAM words), not a tuning parameter.
- Portable C wide integers. Accepted with one deviation from the brief:
  `extsi`/`extui` from a wide type to a wider type is allowed because the
  low 64 bits are unchanged and any later observation of high bits is
  rejected at that op. The k-means paper kernel's oracle needs this.

## Accepted repairs (round 2) and review notes

- Defining-scope name resolution (`allo/ir/utils.py`). Shared Allo frontend
  file. Accepted because it implements Python scoping (module globals,
  lexically enclosing frames nearest first, closure cells) and the upstream
  suites that exercise `customize` and dataflow regions are unchanged; the
  remaining upstream failures pre-exist at `8160d6d` on this host.
- Section IV port. Accepted: `gf2.py` verbatim; the search, bound and masked
  mode carry their telemetry and the paper's reproduction tests. Two
  deliberate departures are documented in `design/layout-search.md`: a
  faster child generator in the scalable search (checked against brute force
  on 1,196 prefixes) and a fallback incumbent on budget exit (the snapshot
  could return none, which fails G4). Not yet a selection stage.
- Samsung selected-program execution. Accepted. Reviewer note: the old
  emitted stream was not a valid program (JUMP counts ignored lanes and GRF
  depth; roles inverted), which is why the conductor had regenerated the
  vendor microcode. The corrected MAC rule changes the default selection
  away from the vendor-equivalent program; the paper rows are pinned by an
  explicit selector rather than by weakening the model, and the default's
  cycles are recorded beside them.
- APU v1 recognizers. Accepted: a strict MLIR parser plus a symbolic
  executor replaces regex counting; every recipe documents the exact
  semantics it implements. Reviewer repair after the device run: the
  profile-freshness scheme assumed `flo` returns unread records only; it
  prints the whole retained log, so the check now diffs the before/after
  reads (`fresh_profile_records`).
- Packaging. Accepted; the wheel is host-specific (libstdc++) and the
  supported installation remains the source build, which INSTALL.md states.

## Rejected shortcuts

- Relabelling the Gemini-II emitted-only route as "functional-only" instead
  of removing it.
- Downgrading the coverage check to a warning for the CENT and UPMEM
  multi-statement kernels; instead each legitimately implemented store got
  a precise category (`copy`, whole-buffer zero fill, host-executed function).
- Editing golden rows or widening tolerances: no golden value changed in
  increment 1; the 4 upstream failures were verified to pre-exist.
- Keeping the Samsung dual-fiber cost rule because it reproduced the paper's
  default selection: rejected in round 2; the rule now matches the
  simulator and the paper placement is selected explicitly in the tests.
- Treating the Gemini-II typed-record runs as evidence for a source route:
  they validate the typed routes only; G2-02 stays open.

## Constants and specialization inventory

Per-area inventories with the task's classification are in `audit/*.md`
section 4 (core, Samsung, UPMEM, AiM, APU v1, G2). Items judged defects and
their status are in `ISSUES.md` (C-08 `lanes=8`; S-09 fallback 8 blocks;
M-06 cost geometry duplication; V1-05 flat count price; G2-11 dead uint16
templates and `APUG2_U16_GEMM_BMAX = 31`; U-06 shape-exact paths).

## Gate judgments at the current candidate

| Gate | Judgment | Basis |
|---|---|---|
| G1 correspondence | **FAIL** (improved) | Section IV mechanisms now in the package (rows 17-22) but not driving selection (row 16); rows 3, 13, 15 missing; E2, E5, E8, E11, E12, E14-E16 external |
| G2 semantic preservation / explicit failure | **FAIL** (improved) | fixed: unmatched effects, accumulator init, zero outputs, result kinds, UPMEM dtypes, wide ints, AiM over-issue, Samsung selected program, APU v1 recognizers and carried outputs, scope resolution; open: C-01, C-09, F-02 |
| G3 general mechanisms | **FAIL** (improved) | APU v1 recognizers structural; Samsung roles structural; open: U-05/U-06 shape-exact paths, G2-02 per-case adapter |
| G4 search/layout/selected-program integrity | **FAIL** (improved) | S-01 closed; C-03/C-05/C-06 open; C-11 search present but not wired |
| G5 cost models | **FAIL** (improved) | Samsung MAC rule calibrated with provenance; open: M-05/M-06, U-09, V1-05, G2-03 |
| G6 evidence beyond examples | **FAIL** (partial) | variant/rename regressions for matcher, AiM, Samsung, UPMEM, APU v1 recipes; Gemini-II fresh random fixtures; no systematic independent-oracle campaign yet |
| G7 install/interfaces/maintainability | **FAIL** (improved) | fresh-venv install verified; docs added; open: P-04/P-06/S-04 simulator reproducibility, P-13/P-14/P-17/P-18/P-19/P-20, architecture doc rewrite (P-08) |
| G8 backend execution | **PARTIAL** | simulators ran at the candidate; APU v1 device: 4 device tests pass after repairs; Gemini-II: 11 programs exact on the card; still no Gemini-II test in the suite and no negative controls |

Verdict: **NOT READY.**
