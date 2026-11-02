# Release status

Updated: 2026-11-02 (round 4). Read `TASK.md` in the containing repository,
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
the verdict from READY, in order of weight:

1. Movement ties are not realizable: no backend can execute an
   iteration-to-owner map other than the kernel's own work-grid map, so
   CLAIMS row 16 ("ties in movement survive to final cost ranking") holds
   only in the sense that ties reach the ranking and are rejected with a
   stated reason. Closing it means teaching at least one backend (Samsung
   row interleaving across channels, or UPMEM row-to-DPU maps) to realize an
   alternative owner map and verifying it on the simulator.
2. User decisions that are not the compiler's to make: redistribution of
   the Samsung-derived simulator patches (P-04); whether to seek a license
   from the DCC authors for the AttAcc evaluation scripts; committing the
   simulator overlays in the containing repository; running the index
   removals in `RELEASE_CHECKLIST.md`.
3. Device evidence still owed: the retiled APU v1 GEMM default
   (`..._broadcast_friendly_acc2`) and the ranked GEMV tiles have not been
   measured on the board (only compile-built); the 12.71x row's plans are
   reachable by `layout=`.
4. Minor open items: V1-03 (default tree plan under `"reassociate"` is a
   documented choice), V1-08 (GSI example template location and license
   headers), G2-09..G2-12, U-06 (`a*A - b*B` and `+ c` pointwise patterns),
   skill-doc AiM bank geometry, a consolidated G6 evidence document.

## Round 4 (2026-11-02): what was done

- Selection: lazy materialization with recorded rejections; whole-program
  re-pricing; bounded joint evaluation (`joint_evaluation` label);
  `design/selection.md` (C-03, C-05).
- AttAcc as an analytical backend: target, tiling space, closed-form cost
  with provenance, `attacc_tiling` knob; 500/500 paper tilings through
  `allo.compile`; external scripts documented (M-04).
- APU v1: fitted affine event model with provenance and validity labels
  (paper error figures reproduced); cost-ranked GEMV tiles; dead API
  removed; correct sqrt (V1-05, V1-06, V1-11, V1-12).
- Frontend: exact integer literal typing with range and shift checks
  (F-02).
- Simulators: Samsung overlay, AiM Dockerfile and the new uPIMulator patch
  each rebuilt from a fresh upstream clone and shown to reproduce the
  pinned cycles; runtime versus dev requirements split; calibration-table
  regeneration script; `RELEASE_CHECKLIST.md`.
- Evidence: artifact-free execution probe (Samsung and AiM run with the
  archive hidden); APU v1 device tests pass on the wave 4 tree (4 passed).

## Next concrete actions (in order)

1. Commit and push round 4 (done when this file is committed).
2. Realize alternative owner maps on one backend (row 16): start with
   Samsung, whose conductor already packs rows by slot; let the lowering
   accept a tie layout that permutes rows across channels, verify outputs
   and cycles on the simulator.
3. Board run for the APU v1 tile ranking and the retiled GEMM default;
   update `design/apu-v1-cost.md` with the measured order.
4. Consolidated G6 evidence document listing, per mechanism, the
   independent oracle, the perturbations and the regression tests.
5. User decisions listed above; then re-run the full suites and both
   hardware hosts at the final revision and re-judge the gates.

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
