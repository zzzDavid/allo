# SPEC-011 — APU v2 cost model: descope (option b)

## Decision

**Option b.** APU v2 is repositioned as a *functional-correctness target only*.
`_apu_v2_kernel_cycles` keeps the `len(trace.matches)` body, the
`RuntimeWarning` is demoted to an inline docstring note (no `warnings.warn`
call), and the autoscheduler's argmin on APU v2 is documented as a
deterministic enumerator-order tie-break, not a cost-driven choice. The two
candidates that spec 009 / task 010 add to the enumerator stay — enumeration
breadth is a separate concern from cost accuracy (TASK_DESCRIPTION §"What
done looks like" item 5 vs. item 6).

## Rationale

The candidate pair the enumerator produces — `l1[0,1,2]` (canonical) vs
`l1[2,1,0]` (reversed) — binds the same three roles (x, y, acc) to the same
three L1 rows in two different orderings. Every plausible *structural* cost
that does not depend on cycle-accurate timing degenerates on this pair:

1. **L1-row-index sum.** Both candidates use rows {0, 1, 2}, sum = 3. Tie.
2. **Per-role weighted row sum** (e.g. weight `acc` heavier than `x`/`y`).
   Defensible only if L1 rows have differential access cost — and `l1_sim`
   declares `perf_is_placeholder = True` precisely because they don't, in the
   functional model. The weight would be fiction with no ground-truth signal.
3. **Group-axis crossings.** Both candidates stay inside one 16-row L1
   group; neither crosses the group boundary. Tie.
4. **Row-to-row distance / max hop.** `|0-1| + |1-2| = 2` for canonical;
   `|2-1| + |1-0| = 2` for reversed. Tie.
5. **Lexicographic on `(memref_name → row_index)`.** This is a tiebreak
   rule, not a cost model — and the enumerator-index sort that `autoschedule`
   already does (spec 009 §3) achieves the same outcome with less code.

In other words: every structural metric we can write down today is either
(a) tied on the candidate pair, or (b) requires a physical claim that
`l1_sim` itself cannot validate. Building (b) and presenting it as
"cost-based scheduling" would be a paper claim with no measurable backing —
exactly the failure mode TASK_DESCRIPTION §"truthfulness" warns against.
Option a is therefore rejected.

The paper is already structured for option b (see §"Paper / README claims"
below): the L1 simulator is labelled functional-only in `implementation.tex`,
APU v2 appears in the "unified programming coverage" section but **not** in
"Automatic Layout Selection on DRAM-PIM" (`evaluation.tex:58`). Option b
therefore needs only a handful of surface edits — no claim is being
retracted, only one or two phrases sharpened so they cannot be
misread as a cost-based-scheduling promise.

When GTML grows cycle counters (or GSI ships an L2 simulator with timing),
this decision flips to option a; the upgrade slot is `_apu_v2_kernel_cycles`
in `spmw_cost_models.py`, and the candidate pair from spec 009 §3 becomes
the regression gate that proves the new cost discriminates.

## File-level change list (for the coder, task 012)

Lines below are at the snapshot read on 2026-05-18; the coder must re-verify
before editing.

### 1. `experiments/allo/allo/spmw_cost_models.py` lines 259–290

- **Remove** the module-level `_APU_V2_WARNED` flag (line 265).
- **Remove** the `warnings.warn(...)` call inside `_apu_v2_kernel_cycles`
  (lines 276–285) along with the `global _APU_V2_WARNED` / `if not ...` /
  `_APU_V2_WARNED = True` scaffolding.
- **Rewrite** the docstring on `_apu_v2_kernel_cycles` to:

  ```
  """Functional-only cost stub for APU v2.

  APU v2 ships as a functional-correctness target: the only available
  simulator (`l1_sim`) declares `perf_is_placeholder = True` and does
  not report cycles. The autoscheduler still runs argmin on APU v2
  candidates (so the autoschedule path stays uniform across backends),
  but the ranking is a deterministic enumerator-order tie-break, not
  a performance prediction. This stub returns `len(trace.matches)` so
  argmin is well-defined; it is intentionally non-comparative across
  placements. See SPEC-011 for the rationale and the upgrade slot.
  """
  ```

- Keep the body `return len(trace.matches)` unchanged.
- `import warnings` at the top of the file stays (other factories may
  still use it; verify with `grep '^import warnings\|warnings\.' spmw_cost_models.py`
  before removing — if `_apu_v2_register_spill` at line 441 also uses
  `warnings.warn`, the same demotion applies symmetrically, but this
  spec only mandates the `kernel_cycles` change; `register_spill` is a
  follow-up).

### 2. `experiments/allo/allo/spmw_cost_models.py` lines 431–445 (`_apu_v2_register_spill`)

- **Apply the same warn-removal** as above for symmetry. The spill cost
  is also a placeholder for the same `perf_is_placeholder = True` reason;
  the docstring should mirror the new `_apu_v2_kernel_cycles` wording
  ("functional-only stub; non-comparative; upgrade slot when GTML
  timing lands"). Keep the body (whatever constant it returns)
  unchanged.

### 3. `experiments/allo/tests/spmw/test_target_apu_v2.py` `test_apu_v2_cost_factory_warns_once`

- Lines 133–155. The test currently asserts a `RuntimeWarning` *fires*.
  After the change it must assert the opposite — no warning fires. New
  body sketch (the coder writes the exact code):

  ```python
  def test_apu_v2_cost_factory_no_warning():
      """APU v2 is functional-only: the cost factory must NOT warn at
      build time. The placeholder semantics are now documented in the
      docstring, not via a runtime warning. See SPEC-011."""
      target = build_apu_v2_target()
      with warnings.catch_warnings():
          warnings.simplefilter("error")  # any warning -> test failure
          cost_fn = allo.get_cost("kernel_cycles", target)
      assert callable(cost_fn)
      trace = _synthetic_mac_trace()
      assert cost_fn(trace, allo.Placement(placements={})) == 1
  ```

  The `cm._APU_V2_WARNED = False` line at 138–139 disappears (the flag
  no longer exists).

### 4. `experiments/allo/SPMW_ARCHITECTURE.md` lines 173–181 and 189

- Line 176–177: the entry that lists APU v2 modes `{"l1_row_canonical",
  "l1_row_reversed"}` stays — enumeration is unchanged. The
  parenthetical "label-only under the placeholder cost" should be
  reworded to "label-only; APU v2 is functional-only, see SPEC-011".
- Line 189: "APU v2 (the last is a placeholder because `l1_sim` declares
  `perf_is_placeholder = True`)" should be reworded to "APU v2 (functional-only
  target; cost stub is non-comparative — see SPEC-011)".
- Section T10 in §4 (unresolved tensions) — update from "Task 010 win
  condition" to "Resolved by SPEC-011 (option b)". The regression-gate
  slot remains documented as the future option-a trigger.
- The `architect/MEMORY.md` entry that mentions T10 (lines 217–220 of
  `MEMORY.md`) should be updated by the architect on the next memory
  pass; this spec records the resolution.

### 5. `paper/latex/` — verified claim audit

Grep result on the paper tree (run on 2026-05-18) found six APU v2
mentions across five files. Per-line verdict:

| File:line | Current text fragment | Verdict |
|---|---|---|
| `abstract.tex:18-19` | "five targets (three DRAM-PIM simulators, real GSI APU v1 silicon, the APU v2 L1 simulator)" | **Keep.** Already enumerates APU v2 distinctly from "automatic layout selection". No cost-based claim attached. |
| `background.tex:32` | "exposes a higher-level tensor API (GTML), abandoning the G1 host+device" | **Keep.** Background description, no claim. |
| `introduction.tex:124-125` | "the L1 simulator of its successor (GSI APU v2). Capability gaps between targets..." | **Keep.** Coverage claim only, no cost claim. |
| `abstraction.tex:148` | Target table row "GSI APU v2 \& 1 chip \& 1 MMB \& 16 grp \& bitline \& 64K bit-serial" | **Keep.** Architectural description, no cost claim. |
| `evaluation.tex:24` | "and GSI APU v2 (L1 simulator). The program text is the same across targets" | **Keep.** `eval-coverage` subsection — coverage claim. **Verify** the surrounding paragraph (lines 19–34) does not promise cost-based scheduling on APU v2; on inspection it claims "validate elementwise vector add and multiply on all five" — pure correctness. OK. |
| `evaluation.tex:215` | "The layout-argmin picks of \S\ref{sec:eval-dram} do not change under calibration" | **Keep.** `\ref{sec:eval-dram}` excludes APU v2 by construction (the section header at line 58 is "Automatic Layout Selection on DRAM-PIM" and the per-target paragraphs are Samsung / AiM / UPMEM only). No edit. |
| `implementation.tex:50-52` | "The GSI APU v2 backend emits a G2-GTML C++ program ... the L1 simulator is functional only." | **Keep — this is the canonical functional-only label.** Cross-reference this from the new cost-model docstring. |
| `implementation.tex:79` | "GSI APU v2 target description & 120" (LOC table) | **Keep.** Implementation-effort claim only. |

**Net paper-edit count: 0.** The paper is already option-b-consistent.
The single phrase that *could* be misread — `eval-coverage`'s wording —
already qualifies APU v2 only by correctness ("validate ... vector add
and multiply on all five ... Numerical correctness is checked against
NumPy"). The "five targets ... cost-driven pass picks by argmin"
sequence is split across two subsections (`eval-coverage` for the
five targets, `eval-dram` for the cost-driven pass on three of them);
APU v2 lives in the first and is absent from the second. This is the
right structure for option b.

**Optional sharpening (not required by spec, but cleanly defensible):**
`implementation.tex:52` could append "; \Name's cost model therefore
reports a non-comparative stub for this target, and the autoschedule
pass falls back to enumerator-order tie-break on APU v2." Coder may
include this as part of task 012 if it can be done without inflating
the section.

### 6. `README.md`

- The README is currently empty (`wc -l README.md` = 0 as of
  2026-05-18). No edits required.
- If the README is repopulated before task 012 lands, the coder must
  re-grep for `apu_v2 | APU v2 | l1_sim | five backend | cost-based`
  and re-apply this spec's "no cost-based-scheduling claim on APU v2"
  rule.

### 7. `TASK_DESCRIPTION.md` success-criterion #6

- Line 125: "APU v2's cost-model and enumeration story is internally
  consistent — either upgraded out of placeholder status, or removed
  from cost-based-scheduling claims with a noted scope change."
- Under option b this criterion is satisfied: the story is *removed*
  from cost-based-scheduling claims (in code docstrings and in
  `SPMW_ARCHITECTURE.md`), and SPEC-011 is the noted scope change.
  No edit to `TASK_DESCRIPTION.md` itself; the coder records the
  resolution in the task 012 status receipt.

## Invariants this spec locks in

- `_apu_v2_kernel_cycles` returns a non-comparative constant. Any
  future PR that introduces a non-constant body must update this spec
  (option-a upgrade path) and the corresponding paper sections.
- The autoschedule enumerator's two candidates (spec 009 §3) remain
  the regression gate for a future option-a upgrade. They do **not**
  need to produce distinct cost values under the current stub.
- No `RuntimeWarning` fires from APU v2 cost factories. The placeholder
  semantics are conveyed by docstring and architecture-doc cross-reference,
  not by a runtime channel.
- APU v2 is absent from every "cost-based scheduling" / "automatic
  layout selection on DRAM-PIM" claim in the paper. It is present in
  every "unified programming coverage" / "five targets" claim.

## Test that proves this works

1. `pytest experiments/allo/tests/spmw/test_target_apu_v2.py -x -q` —
   all seven tests pass after the rewrite of
   `test_apu_v2_cost_factory_warns_once` → `test_apu_v2_cost_factory_no_warning`.
2. `pytest experiments/allo/tests/spmw/ -x -q` — full SPMW suite stays
   green; in particular the autoschedule path on APU v2 still picks a
   placement (deterministically, via enumerator order).
3. Grep gate: `grep -n 'apu_v2.*placeholder\|placeholder.*apu_v2' paper/latex/`
   returns no matches that promise cost-based behaviour. (The
   `implementation.tex:52` "functional only" line is the *correct*
   non-promise and should be the only surviving phrase tying APU v2
   to a performance-model statement.)

## Coordination notes

- This spec resolves architect tension T10 in `SPMW_ARCHITECTURE.md`.
  Memory entry to update on next architect pass:
  `.claude/agent-memory/architect/MEMORY.md` lines 217–220 — change
  "(Task 010 win condition)" to "(resolved by SPEC-011, option b)".
- Task 012 is the coder's implementation half. It must NOT touch the
  enumerator at `spmw_autoschedule.py:340–383` (spec 009 owns that
  surface). It MAY touch `_apu_v2_register_spill` per §2 above.
- No new public APIs. No shared-file edits in `allo/ir/` or
  `allo/dataflow.py` are required by this spec. The blast radius is
  contained entirely within `allo/spmw_cost_models.py`,
  `tests/spmw/test_target_apu_v2.py`, and `SPMW_ARCHITECTURE.md`.
