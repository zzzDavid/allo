<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release status

Updated: 2026-11-02 (round 11: fourth reviewer rejection at `f732b22`; destination identities and arithmetic precision repaired; lint: header, black and clang-format stages pass, pylint stage carries pre-existing findings, P-23). Read `TASK.md` in the containing repository,
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

**NOT READY.** Gate judgments are in `REVIEW.md`. Blockers, in order:

1. **G9 (X-01)**: the shared compilation pipeline for all targets including
   APU v1 and Gemini-II is required by the user (2026-11-02) and not
   implemented. Design seed: `design/shared-compilation-path.md`.
2. **G2**: the independent reviewer disproved the round 7 PASS with two
   executed wrong-code probes at `e41ebbd` (C-19 casts erased, C-20 guarded
   copies), repaired in round 8 (plus C-21), and then found two more at
   `ee8f522` (C-22 float precision silently changed, C-23 output access
   maps lost), and two more at `620a230` (C-24 elementwise destination
   maps, C-25 accepted copies dropped), and two more at `f732b22` (C-25's
   fixed output key displaced by a destination named `out`, C-26 wide
   arithmetic bound to the operands'-type op). Rounds 9 to 11 repair them with
   stated contracts, an external-effect realization check before selection
   and value-checked regressions; awaiting independent review.
3. **G7/G8 evidence**: at `7bdb475` the installed artifact (wheel from the
   committed sdist, clean venv, run outside the checkout) was exercised on
   the three simulators, the APU v1 board and, from a wheel built on
   zhang-x4, the Gemini-II card (VALIDATION.md round 9). Still open for G7:
   the Samsung simulator route is not buildable from the release repository
   without Samsung's permission for the five patches (P-04); the compatible
   build path for a user without the patches is the open question the
   reviewer named.
4. Open minor/boundary items: V1-16, C-18, S-13, P-17, APU v1 owner maps.

## Round 11 (2026-11-02): what was done

- Destination identities (C-25 follow-up): outputs keyed by memref name in
  the Samsung runtime and `SourceBackendBinding.outputs`; required
  destinations derived from `external_effects` in `_copy_outputs`; output
  extent and element type checked in the Samsung elementwise classifier.
- Arithmetic precision (C-26): `arithmetic_precision_problem` enforced at
  match time; narrowing proofs retag the accepted operation; wide integer
  and float operations refused unless the op declares `intermediate`.
- STATUS line 6 corrected: the complete lint command still exits in the
  pylint stage (P-23).

## Round 10 (2026-11-02): what was done

- Elementwise destination maps (C-24): Samsung and AiM elementwise
  lowerings require the resolved identity map.
- Dropped copies (C-25): copy destination contract in the matcher;
  `external_effects` / `require_effects_realized` in `compile_for_target`;
  Samsung elementwise realizes identity copies (`sink_memrefs`,
  `SourceBackendBinding.outputs`, `_copy_outputs` requires every
  destination); other backends refuse such copies at compile time.
- Full `task_lint.sh` header check passes against `origin/main`.

## Round 9 (2026-11-02): what was done

- Float precision (C-22): `truncf` at a store is transparent only for a
  single operation on destination-typed leaves (double-rounding theorem);
  compound expressions bind only to an op declaring `intermediate="f32"`
  (AiM `AF` lookup table); `precision_notes` removed; the reviewer's
  program is rejected; numerical tests independent of the matcher.
- Output maps (C-23): resolved affine coordinates on stores, operands and
  sinks; `output_map_problem` identity contract enforced by Samsung, AiM,
  AttAcc, the layout derivation and the copy contract; Samsung's
  opaque-coordinate trust removed; the reviewer's reversed-row program is
  rejected; simulator value check with row-identifying inputs.
- License headers on all release documents; P-22 corrected.

## Round 8 (2026-11-02): what was done

- Reproduced the reviewer's two probes (`/tmp/tenon_reviewer_samsung_{cast,guard}.py`)
  and kept them as regressions in `tests/pim/test_match_coverage.py`.
- Matcher contracts (`allo/spmw_match_engine.py`): `cast_is_exact`,
  per-value constant proofs, store-side integer narrowing accepted only as
  the destination-width evaluation of fitting leaves or a proven range,
  store-side float narrowing accepted only over destination-type leaves and
  recorded in `MatchTrace.precision_notes`, every other conversion a visible
  unary term that fails coverage; `_copy_contract_problem` (guards, result
  loops, element type, truncation range proof via `_carried_interval`);
  `WLoad.signed` from the frontend's extension (C-21). UPMEM histogram names
  an unproven bin-index truncation; one Gemini-II expectation updated to the
  shared copy contract's message.
- Simulator recipes distributed in the release repository under
  `simulators/` with `tests/perf/test_simulator_recipes.py`; the Samsung
  patches are excluded by `.gitignore` and `MANIFEST.in` (P-04).
- Ledgers made consistent: ISSUES C-19/C-20/C-21/X-01; REVIEW gate table
  (G2 FAIL-then-repaired, G8 UNVERIFIED at this revision, G9 FAIL);
  CLAIMS evaluation rows updated for closed issues; CAPABILITIES
  cross-cutting status rewritten.

## Next concrete actions (in order)

1. Independent review of the round 8 matcher contracts.
2. Installed-artifact evidence: complete at `67f89fd` for virtual,
   Samsung, AiM, UPMEM, the APU v1 board and the Gemini-II card
   (VALIDATION.md round 10). Repeat after any later code commit.
3. Lint: header, black and clang-format stages pass; the pylint stage
   carries pre-existing fork-wide findings (P-23).
4. G9: refine `design/shared-compilation-path.md` from source (stage
   inventory, reusable pieces, missing contracts), then the first vertical
   slice through one matcher backend, then the APU routes; remove the
   public bypasses once covered.

## Round 6 (2026-11-02): what was done

- G6 probes (`tests/pim/test_g6_semantics.py`, `test_g6_decisions.py`,
  board and card additions) covering aliases, extreme integers, nonzero
  accumulators, executed schedule pairs, all-zero results, masks, strict
  versus reassociative pairs, cost perturbations traced to execution,
  target-parameter variation, renames with normalized decisions, op and
  constant changes at fixed shape, composed stages, archive removal and an
  sdist smoke test. Four silent-wrong-result defects found and fixed
  (C-17, S-14, V1-15 and the APU v1 zero-fill disagreement).
- Owner maps realized on UPMEM and AiM (CLAIMS row 16 now holds on three
  of four matcher backends).
- Multi-kernel Samsung programs execute on the simulator through the host
  schedule (S-15); APU v1 `backend="virtual"` is analytical only.

## Next concrete actions (in order)

1. Commit and push round 6 (done when this file is committed).
2. User decisions listed above.
3. C-18 evaluator diagnosis; M-14 confirmation; APU v1 owner maps.
4. Re-run every suite and both hardware hosts at the final revision and
   re-judge the gates for a READY verdict.

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
