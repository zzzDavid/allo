# Release status

Updated: 2026-11-02 (round 5). Read `TASK.md` in the containing repository,
then this file, then `ISSUES.md` for the next action.

## Identity

- Release target: `experiments/allo`, branch `tenon`, publishing remote
  `fork` = `git@github.com:zzzDavid/allo.git` (`origin` is upstream Cornell
  Allo and must never be pushed to). Surveyed revision `8160d6d`
  (= `fork/tenon` at the start of round 1).
- Frozen paper: `zzzDavid/tenon-hpca27` `main` @ `30b6f49e`
  (`paper/SUBMITTED_VERSION.json`).
- Dirty state inherited at the start of round 1: six untracked
  `scripts/verifier_check_*.py` files in `experiments/allo` (left untouched,
  not committed); the containing repository has extensive uncommitted
  research history (design docs, agent memory, paper source changes) that
  is out of scope and is preserved.
- Commits so far: see `git log --oneline 8160d6d..tenon` (round 1: 33f53fb,
  7530405, ffb4a78; round 2: see below).
- Containing repository (not committed by this work, user decision pending):
  `experiments/simulators/setup/PIMSimulator/` now holds the exact overlay
  that builds the running `pim_driver` (our files in full, Samsung-licensed
  files as patches, README, apply.sh); `experiments/simulators/setup/apply.sh`
  calls it; the `dev/.../pimsim-conductor` tree mirrors the installed
  simulator sources. The installed `pim_driver` was rebuilt
  (sha256 `0ae6a3d9...`, previous binary kept as `pim_driver.bak-20261101`).

## Verdict

**NOT READY** (close). Gate judgments are in `REVIEW.md`. What still keeps
the verdict from READY:

1. User decisions that are not the compiler's to make: redistribution of
   the Samsung-derived simulator patches (P-04); whether to seek a license
   from the DCC authors for the AttAcc evaluation scripts; committing the
   simulator overlays and skill-doc corrections in the containing
   repository; running the index removals in `RELEASE_CHECKLIST.md`.
2. Movement ties are realized on Samsung only (CLAIMS row 16 partial):
   AiM, UPMEM and APU v1 reject alternative owner maps with a stated reason,
   and ties that split a row's reduction belong to the reduction policy and
   are not executed anywhere.
3. G6 gaps listed in `EVIDENCE.md` section 3 (alias cases on every backend,
   extreme-integer execution outside UPMEM and the C backend, executed
   two-schedule pairs on AiM/UPMEM/Gemini-II, cost perturbations traced to
   an executed program, target-parameter variation beyond the AiM virtual
   study).
4. Device evidence owed: see VALIDATION.md round 5 for the APU v1 tile
   ranking and retiled GEMM board check.

## Round 5 (2026-11-02): what was done

- Samsung realizes movement-tie layouts whose owner bits come from the
  output-row bits (row-to-slot placement executed by the conductor, digest
  checked at readback; vendor program unchanged; 4 realizable ties on the
  paper GEMV, identical cycles). Conductor overlay updated in the containing
  repository; installed `pim_driver` rebuilt (backup kept).
- Minor items closed: UPMEM `a*A - b*B` and `a*A + b*B + c` patterns
  (exact on the simulator); Gemini-II out-of-range readback raises instead
  of wrapping, padded divisor lanes hold 1, dead uint16 templates and
  pricing removed (card test re-run: 10 passed); APU v1 template staging
  copies only the needed GSI files with a `THIRD_PARTY.txt` record; V1-03
  documented; AiM skill-doc geometry corrected.
- `docs/release/EVIDENCE.md`: per-mechanism G6 evidence tables, execution
  paths, gaps with proposed tests.

## Next concrete actions (in order)

1. Commit and push round 5 (done when this file is committed).
2. Close the G6 gaps in `EVIDENCE.md` section 3, starting with alias cases
   and executed two-schedule pairs on AiM and UPMEM.
3. Realize alternative owner maps on UPMEM (row-to-DPU maps) and AiM
   (row-to-channel maps) as Samsung now does.
4. User decisions listed above; then re-run every suite and both hardware
   hosts at the final revision and re-judge the gates.

## Round 1 (2026-11-01): what was done

- Seven source audits with executed probes: `audit/*.md`.
- Repairs (all with tests): matcher store coverage and accumulator-init
  facts (C-02, C-14/S-08/M-09); `RunResult.kind` contract and missing-output
  errors (G2-07, M-08 part); Samsung zero/truncated readback (S-02); removal
  of the dead Gemini-II source route (G2-01); `APUv1Program` virtual and
  functional backends (V1-04); AiM structural axis roles and MAC-count
  invariant (M-01, M-10), `_eval_sym` no longer guesses unit 0 (C-07); UPMEM
  element-type enforcement and exact packing (U-01); portable C wide-integer
  soundness (U-02/A09).
- Ledger: `CLAIMS.md`, `ISSUES.md`, `REVIEW.md`, `VALIDATION.md`.
- Gemini-II host inspected (card available, FW V1.24.1); SSH IP-alias host
  key issue resolved without disabling checking.

## Next concrete actions (in order)

1. Commit and push round 1 (fast-forward to `fork/tenon`) after the final
   suite run recorded in `VALIDATION.md`.
2. F-01: resolve annotation names from the defining scope only
   (`allo/ir/utils.py`); run the upstream suites that exercise
   `customize`/dataflow to confirm no regression. F-02: make integer literal
   typing in the frontend fail on truncation.
3. S-01: carry the selected Samsung realization into the conductor (or
   document a proven lowering equivalence for the canonical route and make
   the placement levers that do not reach the executable explicit); S-04:
   commit the exact simulator overlay that builds the running `pim_driver`
   (license review first: Samsung notice, P-04).
4. C-11: port the minimum-movement algebra and bounded search per
   `audit/core.md` §3.3 into `allo/pim/layout_search.py` and wire the tie set
   into `rank_matcher_placements`; re-establish the 315/576/20,160 and
   121,968 checks as package tests.
5. C-09: explicit reassociation policy consulted by every reduction-splitting
   knob; C-01 target-neutral legality pass.
6. V1-01: replace regex recognizers with structural checks that bind the
   semantics (constants, fill values, stored locals) or reject; V1-02: carry
   the output or reject nonzero initial outputs on the vector route.
7. G8: build and run the Gemini-II typed/composed routes on zhang-x4 with a
   shared `flock` (plan in `audit/apu-g2.md` §5); re-run the APU v1 device
   rows after V1 repairs with the board lock.
8. G7: packaging (package data, extension build or documented bindings
   path, UPMEM compatible-build route), README/architecture rewrite,
   capability matrix.

## External blockers

- None hard at the moment. Gemini-II validation needs card time on
  zhang-x4 (25 users logged in; card idle at inspection).
