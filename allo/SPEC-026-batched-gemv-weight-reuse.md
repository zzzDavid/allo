# SPEC-026 — Batched-GEMV weight-reuse schedule (enumerator + cost + codegen + faithful run)

**Date:** 2026-06-07
**Author:** architect agent (Opus 4.8 1M)
**Task:** `dev/06072026-samsung-gemv-peak/work/queue/202-needs-arch-batched-enum-cost-codegen-spec.task`
**Blockers cleared:** arch-200 (verdict (a) FEASIBLE), research-201 (B*=2, asymptote 3.93x, P/E/R separable).
**Gates:** 203 (match frontend), 204 (enumerator), 205 (cost), 206 (codegen + faithful run); verified by 207 (gate A strict-beat) + 208 (gate B re-audit anti-hardcoding).

This spec pins the contract the four coder tasks implement. It is the
quantitative continuation of arch-200 §4. Read arch-200 for the residency
proof and research-201 for the crossover; this doc owns the *signatures*.

---

## 0. Invariants carried in from 200/201 (non-negotiable)

These are the boundary conditions every section below must satisfy. 207/208
check them.

- **I1 — Frozen surface.** The five reference fns (`runPIM`, `executeGemv`,
  `executeGemvWithCmds`, `computeGemv`, `programCrf`) plus
  `executeGemvFaithful` / `countGemvStreamWork` plus the `>=1` body clamp stay
  **byte-frozen**. The batched beat lives in NEW driver-level sequencing only.
- **I2 — Same accounting both sides.** `cyc_preload` / `cyc_exec` /
  `cyc_readback` are measured by the **same** faithful instrument for native
  and Tenon. Tenon's only win is *not issuing* the redundant preloads on
  calls 2..B (skipping real transactions it does not need). No second
  accounting, no re-pricing of native.
- **I3 — B is geometry, never a literal.** `B = bShape[0]` of the input
  operand X[B,K] (≡ the leading dim of Y[B,M]). It is read off the trace the
  same class as M and K. No decision path may contain a batch/shape integer
  literal (gate B, 208).
- **I4 — B=1 parity floor preserved.** At B=1 every cost branch and every run
  branch returns exactly `P + E + R`. The iteration-2 single-shape floor
  (15251 @4096×1024; `gemv-floor-finding.md`) is unchanged. Resident vs
  non-resident **tie** at B=1, so the existing argmin winner
  (`dual_fiber+crf_shared`+host) is undisturbed for B=1 workloads.
- **I5 — Decision in argmin, codegen materialises only.** The resident-vs-
  non-resident choice is made by the cost model's argmin (205) over two real
  candidates the enumerator emits (204). Codegen (206) materialises the
  chosen candidate; it does not re-decide.

---

## 1. MATCH (task 203) — recognising batched GEMV and carrying B

### 1.1 Ruling: NO `allo/ir/` edit. B is carried on `MatchedOp.extra`, additively.

Per the arch-200 §4.3 default ruling ("forbid shared-file edits for the batch
dim unless 202 proves the match path cannot carry a leading batch dim
additively"), I have read the match path and **prove it can**:

- `MatchedOp.enclosing_loops` already carries, outer→inner, every nesting
  `affine.for` as `(loop_var_name, lower, upper, step)` — strings
  (`spmw_match.py:56-60`, populated at `spmw_match_engine.py:551`).
- `OperandBinding.indices` already carries each operand's index SSA-value
  names verbatim (`spmw_match_engine.py:448`, `444-451`).
- `MatchedOp.extra: dict[str, Any]` already exists (`spmw_match.py:78`) and is
  the sanctioned additive carrier (precedent: SPEC-009 used it for nothing,
  SPEC-023 used `Placement.extra` for fibers — same discipline, on the trace
  side here).

A batched GEMV `Y[B,M] = X[B,K] @ W[K,M]^T` lowers (allo.customize, no IR
change) to a MAC store-site nested in loops `[b, (m,) k]` where the `b` loop's
iter var is the **leading index of both the `x` operand and the `acc`/`y`
result** but is **absent from the inner reduction fold**. That structure is
fully visible in the existing `MatchedOp` fields. Therefore: **no
`allo/ir/builder.py`, no `allo/ir/infer.py`, no `allo/dataflow.py` change.**
This keeps blast radius zero — no upstream `tests/dataflow/` or
`tests/customize/` test is touched.

### 1.2 The batch-dim resolver (spmw-local, additive)

Add **one** pure helper, in `spmw_match_engine.py` (next to `match_workload`)
or `spmw_match.py` (as a `MatchTrace`/`MatchedOp` method) — coder's choice,
both are spmw-local. Signature:

```python
def batch_dim(match: MatchedOp) -> tuple[str | None, int | None]:
    """Return (batch_loop_var, B) for a batched reduction match, else (None, None).

    A loop L in match.enclosing_loops is the BATCH loop iff its iter var:
      (a) appears as the leading index of the x-role operand AND of the
          result (acc/y) operand, AND
      (b) does NOT appear in the innermost reduction loop's index position
          on the bank/weight operand (it is not a reduction axis).
    B is _parse_loop_bound(L.upper). Structural test over indices + loop
    vars only — never a shape literal, never a positional [0] assumption
    about which loop is outer.
    """
```

Rules the coder must honour:

- **Structural, not positional.** Do NOT assume `enclosing_loops[0]` is the
  batch loop. Identify it by the index-coincidence test (a)+(b) above. This is
  what keeps gate B green: B falls out of the loop/index *structure*, the same
  way `is_auto` falls out of handle type.
- **Default B=1.** If no loop satisfies (a)+(b) (the canonical single-vector
  GEMV `X[1,K]` / the eltwise paths), return `(None, None)`; downstream treats
  this as `B=1` (I4 parity).
- **Stamp it once, on the trace.** `match_workload` (or a thin post-pass it
  calls) writes, for each GEMV MAC match,
  `match.extra["batch_loop_var"] = <name or None>` and
  `match.extra["batch_dim"] = <B or 1>`. Cost (205) and codegen (206) read
  `match.extra["batch_dim"]`; they do not re-derive it. The enumerator (204)
  reads it via a trace helper (§2.2). One source of truth.

### 1.3 Recognition does NOT change the matched op-shape

The matched `target_op_name` stays `"MAC"`; the operand roles stay `x/y/acc`;
the `LinearLayout` bank algebra is unchanged. Batched GEMV is the **same MAC
match** with a B-loop wrapped around it — recognised by the resolver, not by a
new pattern. No new `OpPattern`, no matcher-engine unify change.

### 1.4 Fallback (only if 1.1 is somehow false at implementation time)

If the coder discovers the leading batch index is NOT recoverable from
`enclosing_loops` + `indices` (e.g. allo folds the B-loop away before the
matcher runs, leaving B only in the memref *type* shape), the **escalation is a
task back to the architect (via orchestrator), NOT an ad-hoc `allo/ir/` edit**.
The architect's pre-authorised fallback, in priority order:

1. **Read B from the memref result type, spmw-side.** The store's result
   memref carries its shape in the MLIR type; a spmw-local reader can pull
   `bShape[0]` without touching `allo/ir/` (it reads the type, does not change
   inference). This stays additive and is the first fallback.
2. Only if even (1) is impossible: an **additive** `allo/ir/infer.py` change
   that *annotates* the leading dim as an attribute (new attr, default-absent,
   no existing-semantics change). This must name the upstream gating tests
   (`tests/customize/test_*` covering affine-map inference,
   `tests/dataflow/test_*` covering kernel lowering) and keep them green, with
   a rollback = drop the attribute read (B falls back to 1). The architect
   writes that sub-spec; the coder does not improvise it.

Expectation per 1.1: fallback is not needed. It is documented so 203 has a
deterministic path on surprise instead of reaching into shared files.

---

## 2. ENUMERATOR (task 204) — two materialisable residency candidates

### 2.1 The new candidate: `weight_resident`

Extend `_samsung_enumerate` (`spmw_autoschedule.py:192`) so that, **after** the
existing lever-1/2/3 fan-out produces its candidate list, each candidate is
crossed with a **weight-residency** dimension carried in
`Placement.extra["weight_resident"] ∈ {False, True}`:

- `weight_resident=False` (default, == today's behaviour): W is (re)preloaded
  per input vector. This is the non-resident / native-shaped schedule.
- `weight_resident=True`: W is preloaded **once** and reused across all B
  input vectors (the Tenon schedule arch-200 proved feasible).

Both are real, materialisable `Placement`s — the same `placements` dict and
`fibers`/`crf_issue`/`grf_residency` extras; they differ only by the
`weight_resident` flag and (per §4) the driver invocation codegen selects from
it. **The enumerator emits BOTH unconditionally; it never prunes the
non-resident variant and never branches on B or any shape.** This is the
SPEC-009 ≥2-candidate discipline applied to residency: the choice is *earned*
by argmin, not sniffed.

### 2.2 Placement helper + cross (mirror `_with_residency` / `_with_crf_modes`)

Add a helper paralleling the existing lever helpers:

```python
def _with_weight_residency(base: "Placement", resident: bool) -> "Placement":
    """Copy base, setting extra['weight_resident']. placements untouched."""
    new_extra = dict(base.extra)
    new_extra["weight_resident"] = resident
    return Placement(
        placements=dict(base.placements),
        mode=_join_mode(base.mode, "wresident") if resident else base.mode,
        extra=new_extra,
    )
```

and apply it as a **final 2x fan-out** at the tail of `_samsung_enumerate`,
after the `_with_crf_modes` cross (so weight-residency is orthogonal to
y-placement / even-odd / host-GRF / crf-issue, exactly like lever 3 is
orthogonal to lever 2):

```python
out: list[Placement] = []
for cand in residency_candidates:
    for crf_cand in _with_crf_modes(cand):
        out.append(_with_weight_residency(crf_cand, False))
        out.append(_with_weight_residency(crf_cand, True))
return out
```

Result: the Samsung candidate count doubles. Argmin sees both residencies for
the winning lever-1/2/3 combo and earns the resident one for B≥2, the
non-resident one (tie → enumerator-index tie-break, I4) for B=1.

### 2.3 No batch/shape literal in the enumerator

The enumerator does **not** read B. It emits the residency *choice* as two
candidates; the *evaluation* of which wins (which depends on B) is the cost
model's job (§3). This keeps the enumerator shape-blind — the gate-B-clean
split arch-200 §3 requires. `weight_resident` is a boolean materialisation
flag, the same class as `crf_issue ∈ {shared, per_workid}`.

### 2.4 Other backends untouched

`_with_weight_residency` is applied only inside `_samsung_enumerate`. AiM /
UPMEM / APU v1 / APU v2 enumerators do not gain the flag; their cost models
never read `weight_resident` (default-absent → treated as the non-resident /
B=1 path, byte-identical to today). Zero cross-backend regression.

---

## 3. COST (task 205) — amortized preload, separable P/E/R, B from shape

### 3.1 Make `_samsung_kernel_cycles` separable into P, E, R

Today `_samsung_kernel_cycles` (`spmw_cost_models.py:98-212`) computes a single
`body_cyc` (folded MAC + JUMP − host saving) wrapped by the crf-issue
replication term, and returns one number that corresponds to **exec** of a
single vector. It models neither preload nor readback (the single-GEMV floor
work priced them implicitly via the faithful run, not the cost model).

205 must refactor `cost_fn` to compute **three named per-vector phase costs**
from target-spec constants + operand shape (M, K), then compose them with B:

```
exec_cyc      = <today's full body_cyc-after-crf-issue expression>   # unchanged formula
preload_cyc   = _samsung_preload_cycles(target, M, K)                # NEW, §3.2
readback_cyc  = _samsung_readback_cycles(target, M)                  # NEW, §3.3
B             = _trace_batch_dim(trace)                              # NEW, §3.4, default 1
weight_resident = layout.extra.get("weight_resident", False)

if weight_resident:
    return preload_cyc + B * (exec_cyc + readback_cyc)     # tenon
return B * (preload_cyc + exec_cyc + readback_cyc)         # native / non-resident
```

**Critical parity requirement (I4):** at B=1 BOTH branches algebraically reduce
to `preload_cyc + exec_cyc + readback_cyc`. The coder must keep them
syntactically distinct (so the cost *signal* exists for B≥2) but numerically
identical at B=1.

**Critical regression requirement:** `exec_cyc` must be the **same value**
today's `cost_fn` returns (the post-crf-issue body number). The refactor
factors out `exec_cyc` without changing its formula. For all *existing*
single-vector Samsung tests, the cost model must return
`preload_cyc + exec_cyc + readback_cyc` — which is a CHANGE from today's
`exec_cyc`-only return. See §3.5 for why this is correct and what guards it.

### 3.2 `preload_cyc` — closed-form from target-spec constants + (M,K)

research-201 §"B and the per-phase costs" resolved the arch-200 open question:
**`preload_cyc` is closed-form**, not a measured constant. The form:

```
preload_cyc = (M * K / write_width) * write_cyc  +  programCrf_cyc
```

where `write_width`, `write_cyc`, `programCrf_cyc` are **target-spec
constants** (the W→bank streaming width and per-write cost, plus the one-time
CRF program latency) and `M, K` are operand-shape dims. The coder must source
these from `target.*` fields the same way `exec_cyc` sources
`target.op("MAC").cycles` / `target.grf_a.lanes`. **If a needed constant is not
yet a field on the Samsung target fixture, 205 adds it to the fixture (with a
PIMSimulator citation), NOT as an inline literal in the cost path** — exactly
how `_samsung_kernel_cycles` already forbids born-in-cost constants
(`spmw_cost_models.py:50-54`).

Calibration target (research-201, 4096×1024): `preload_cyc ≈ 11368`,
`exec_cyc ≈ 3702`, `readback_cyc ≈ 181`, sum ≈ 15251. The coder tunes the
fixture constants so the closed form lands on the faithful-measured P/E/R at
4096×1024 (these are the §3.6 calibration anchors); the *form* must then track
other (M,K) shapes monotonically (verified at 256×128 / 64×256 by 207).

### 3.3 `readback_cyc` — closed-form from a target constant + M

```
readback_cyc = ceil(M / read_width) * read_cyc
```

`read_width`, `read_cyc` are target-spec constants; M is shape. Same sourcing
discipline as §3.2.

### 3.4 `_trace_batch_dim(trace)` — read B off the trace, default 1

```python
def _trace_batch_dim(trace: MatchTrace) -> int:
    """Max batch_dim across the trace's matches (operand-shape dim, default 1).

    Reads match.extra['batch_dim'] stamped by the matcher (SPEC-026 §1.2).
    NOT a literal: B is geometry, the same class as the M/K the body cost
    already reads from enclosing_loops.
    """
    return max((m.extra.get("batch_dim", 1) for m in trace.matches), default=1)
```

The cost model reads `match.extra["batch_dim"]`; it does NOT call the matcher's
`batch_dim()` resolver itself (single source of truth, §1.2). If the key is
absent (every existing trace, eltwise, single-vector GEMV), B=1 → I4 parity.

### 3.5 Why the single-vector return changes — and the guard

Today `_samsung_kernel_cycles` returns only `exec_cyc`. After 205 it returns
`P + E + R` for a single vector. This is the correct fix, not a regression:
the cost model's job is to be **monotone with the faithful run** (which clocks
all three phases). The existing argmin among lever-1/2/3 placements is
**preserved**: P and R are *placement-invariant* (they depend only on M, K, not
on which y-placement / crf-issue / host-residency wins), so adding the constant
`P + R` to every candidate's score shifts all candidates by the same amount and
does not change the argmin. Verified-by-test requirement for 205:

- **Existing single-shape argmin unchanged.** The lever-1/2/3 winner at B=1
  (`dual_fiber+crf_shared`+host) is still the winner. The `P+R` offset is
  uniform across candidates (it does not read `grf_residency` / `crf_issue` /
  `fibers`). 207 asserts the B=1 winner is unchanged.

### 3.6 Cost test matrix (handoff to 207; mirrors research-201 §Test)

1. **B=1 resident == non-resident** (`P+E+R`, ≈15251 @4096×1024).
2. **B=2 resident < non-resident** (≈19134 < 30502).
3. **speedup → 3.93 as B grows** (B=8 → ≈42432 vs ≈122008).
4. **zero-preload falsifier**: with `preload_cyc` forced 0, resident ==
   non-resident for ALL B (the entire win is the single P term).
5. **B\* invariance**: perturb the P/E ratio (scale a P constant) — the win
   *magnitude* moves but the crossover stays B\*=2.

### 3.7 No other backend's cost model changes

`preload_cyc` / `readback_cyc` / `_trace_batch_dim` / the resident branch live
**inside `_samsung_kernel_cycles`**. AiM/UPMEM/APU cost fns are untouched; they
never read `weight_resident` or `batch_dim`.

---

## 4. CODEGEN + FAITHFUL RUN (task 206)

### 4.1 Codegen: emit preload once + B exec passes; decision already made

`_run_samsung` / `_run_samsung_one` (`spmw_codegen.py:1861, 1777`) read the
chosen `Placement.extra["weight_resident"]` (threaded onto `Compiled` from the
autoschedule winner the same way `host_preloads` is, `spmw_codegen.py:94-100,
1426-1432`) and select the driver invocation. The **CRF cmd stream per vector
is unchanged** (same folded MAC+JUMP); only the driver *sequencing* differs.
Codegen materialises; it does not re-decide residency (I5).

The host-residency machinery (lever-2, SPEC-024; `host_preloads`) already
expresses "preload off the CRF stream." 206 extends the run-path meaning to
"preload once for the whole batch": the emitted stream's W-preload is hoisted
out of the B loop. This is a run-path / driver-flag change, not a cmd-emission
change.

### 4.2 The faithful run path — SAME accounting both sides, no flattering (I2)

206 adds a **batched driver path** in `pim_driver.cc` — NEW harness code, the
five frozen fns + `executeGemvFaithful` + `>=1` clamp untouched (I1). Two new
CLI flags on the existing GEMV branch:

- `--batch B` (int, default 1): number of input vectors. When `B>1`, X is fed
  as a `(B,K)` numpy so `bShape[0]=B` is read directly (matching the existing
  `num_batch = i_data->bShape[0]` contract, `PIMKernel.cpp:587`).
- `--native-rebaseline` (bool, default false): selects the native comparator
  loop (re-preload per call) instead of the resident loop.

The GEMV phase block (`pim_driver.cc:292-321`) gains a batched sequencing
wrapper. **Both modes use the identical `executeGemvFaithful` + `readResult`
per vector — the only difference is whether `preloadGemv` is inside or outside
the B-loop:**

```cpp
// TENON resident (default when --batch B>1, no --native-rebaseline):
kernel->preloadGemv(&W); kernel->runPIM();
cyc_preload = getCycle() - cyc_start;                 // preload clocked ONCE
for (b = 0; b < B; ++b) {
    t1 = getCycle();
    kernel->executeGemvFaithful(&W, &x_b, false, loaded_cmds); kernel->runPIM();
    cyc_exec += getCycle() - t1;                       // accumulate B exec
    t2 = getCycle();
    kernel->readResult(result_b, ODD_BANK, ...); kernel->runPIM();
    cyc_readback += getCycle() - t2;                   // accumulate B readback
}
// => total = preload + B*(exec + readback)

// NATIVE rebaseline (--native-rebaseline):
for (b = 0; b < B; ++b) {
    s = getCycle();
    kernel->preloadGemv(&W); kernel->runPIM();          // RE-PAY preload each call
    cyc_preload += getCycle() - s;                      // accumulate B preloads
    t1 = getCycle();
    kernel->executeGemvFaithful(&W, &x_b, false, loaded_cmds); kernel->runPIM();
    cyc_exec += getCycle() - t1;
    t2 = getCycle();
    kernel->readResult(result_b, ODD_BANK, ...); kernel->runPIM();
    cyc_readback += getCycle() - t2;
}
// => total = B*(preload + exec + readback)
```

Both loops call the **same** `executeGemvFaithful` and the **same**
`readResult` per vector (I2). The only asymmetry is preload placement — the
genuine hardware asymmetry arch-200 §1 proved. The existing
`PIM_CYCLES total=.. preload=.. exec=.. readback=..` print line
(`pim_driver.cc:325-330`) is reused; the per-phase fields now hold the
accumulated batched totals, so the existing Python parser
(`re.search(r"PIM_CYCLES total=(\d+)")`, `spmw_codegen.py:1851`) keeps working
and the harness can additionally assert the per-phase split against the cost
model.

**Per-vector input note.** `x_b` is the b-th row of the `(B,K)` input. The
cleanest realization (arch-200 §4 note) is: the resident path may instead make
a **single** `executeGemvFaithful(&W, &x_full_BxK, ...)` call whose internal
`for(b<num_batch)` loop (`PIMKernel.cpp:587,619`) already amortises preload by
construction, then `readResult` per vector. 206 owns the final call shape; both
realizations clock `preload` once and `B*(exec+readback)`. The **native** side
is always the explicit B-separate-preload loop — that is the thing Tenon beats.

### 4.3 Native batched baseline harness (the comparator 207 measures)

The native baseline is **B genuinely separate preload+exec+readback
sequences**, selected by `--native-rebaseline`. This is what a backend with no
cross-call residency analysis emits: every vector is a fresh GEMV that
re-streams W. 207's gate-A measurement runs:

```
tenon_total  = pim_driver --op GEMV --batch B --cmds <tenon.txt> --faithful
native_total = pim_driver --op GEMV --batch B --native-rebaseline --cmds <native.txt> --faithful
gate A: tenon_total <= native_total, strict (<) for B>=2.
```

Both invocations use the **same faithful instrument, same cmd stream, same W,
same X(B,K)** — only the preload-loop placement differs. The Python side
(`_run_samsung` / a new `_run_samsung_batched`) emits both invocations for the
gate, parses both `PIM_CYCLES total=` lines, and reports the pair. The expected
result (research-201 curve): B=2 → 19134 vs 30502, B=8 → 42432 vs 122008,
asymptote 3.93x.

### 4.4 Rollback story

- **Cost model:** `SPMW_DISABLE_REGALLOC=1` already exists; additionally,
  absent `weight_resident` (key default False) the cost returns the
  non-resident `B*(P+E+R)`, and at B=1 that equals `P+E+R` — the pre-026
  number. To fully revert the P/E/R split, the enumerator simply stops emitting
  the resident candidate (revert §2.2's 2x fan-out); cost then only ever sees
  `weight_resident=False`.
- **Driver:** the batched path is gated behind `--batch B>1`. With `--batch 1`
  (or no flag) the GEMV branch runs the **exact** pre-026 single-vector
  sequence (I4) — the new flags are inert. FPGA/AIE CI never reaches this
  Samsung-only C++ path. Rollback = drop the two flags; the frozen fns are
  untouched so the upstream `KernelTestCases.cpp` gtest path is unaffected.

---

## 5. Shared-file boundary (for `SPMW_ARCHITECTURE.md` §2)

| File | Touched? | Justification | Guard |
|------|----------|---------------|-------|
| `allo/ir/builder.py` | **NO** | §1.1: B carried on `MatchedOp.extra`, additive | — |
| `allo/ir/infer.py` | **NO** (fallback only, §1.4.2, architect-authored) | §1.1 | upstream `tests/customize`/`tests/dataflow` named if ever invoked |
| `allo/dataflow.py` | **NO** | §1.1 | — |
| `spmw_match.py` / `spmw_match_engine.py` | yes (spmw-local) | §1.2 `batch_dim` resolver + `extra` stamp; additive | new `tests/spmw/test_match_batched.py` |
| `spmw_autoschedule.py` | yes (spmw-local) | §2 `_with_weight_residency` + tail cross | `tests/spmw/test_autoschedule*` |
| `spmw_cost_models.py` | yes (spmw-local) | §3 P/E/R split + resident branch | new cost matrix §3.6 (207) |
| `spmw_codegen.py` | yes (spmw-local) | §4 batched run path | new `_run_samsung_batched` test (206) |
| `pim_driver.cc` | yes (NEW harness, frozen fns untouched) | §4.2 `--batch`/`--native-rebaseline` | rebuild `scons -j32`; frozen-fn diff = 0 |

Zero upstream Allo `.py` edits. The C++ change is additive (two flags, new
loop wrapper); the frozen functions' bodies have a zero diff.

---

## 6. Open tensions (carry to architect memory + `SPMW_ARCHITECTURE.md`)

- **T16-a (resident call shape).** §4.2: Tenon's resident faithful run may use
  either (i) a single `executeGemvFaithful` with `bShape[0]=B` (internal b-loop
  amortises by construction) or (ii) an explicit driver B-loop. Both are
  faithful and clock the same totals. 206 picks one; revisit only if a future
  workload needs per-vector cmd-stream variation (different folded CRF per
  vector), which the single-call form cannot express.
- **T16-b (P/R placement-invariance).** §3.5 leans on preload/readback being
  independent of the lever-1/2/3 placement. True for GEMV (W→bank streaming and
  result readback don't depend on y-placement / crf-issue). Revisit if a future
  lever changes the preload transaction count (e.g. a layout that stages W
  through GRF would make `preload_cyc` placement-dependent and break the
  uniform-offset argmin-invariance argument).

---

## 7. Receipt

SPEC pins: (1) **NO `allo/ir/` edit** — B carried additively on
`MatchedOp.extra["batch_dim"]`, resolved structurally by a new spmw-local
`batch_dim(match)` helper (leading-index coincidence test, not positional, not
a literal); fallback escalates to architect, never an ad-hoc shared edit.
(2) Enumerator gains `Placement.extra["weight_resident"] ∈ {False,True}` as a
final 2x tail cross via `_with_weight_residency`, both candidates emitted
unconditionally, no shape branch. (3) `_samsung_kernel_cycles` splits into
closed-form `preload_cyc=(M*K/write_width)*write_cyc+programCrf_cyc` /
`exec_cyc` (unchanged formula) / `readback_cyc=ceil(M/read_width)*read_cyc`,
composed `weight_resident ? P+B*(E+R) : B*(P+E+R)`, B from
`_trace_batch_dim(trace)` default 1, B=1 both branches = P+E+R (parity); P/R
placement-invariant so existing argmin winner unchanged. (4) Driver gains
`--batch B` + `--native-rebaseline`; resident clocks preload once + B*(exec+
readback), native clocks B*(preload+exec+readback) via the SAME
`executeGemvFaithful`+`readResult` per vector — only preload-loop placement
differs; five frozen fns + `>=1` clamp byte-untouched. Gate A:
`tenon_total <= native_total`, strict for B≥2 (19134<30502 @B=2, →3.93x).

---

## 8. Implemented (coder 203-206, 2026-06-07)

Landed atomically (tight match→enum→cost→codegen chain on the same files):

- **203 MATCH** — `spmw_match_engine.py`: additive `batch_dim(match)` resolver
  (structural leading-index-of-one-input / absent-from-other test, never
  positional, never a literal) + `_stamp_batch_dims` writing
  `MatchedOp.extra["batch_dim"|"batch_loop_var"]`; store indices carried on
  `extra["store_indices"]`. NO `allo/ir/` edit (1.1 held; no fallback needed).
  Verified: batched 4096×1024 → B from `%arg3`; single-vector → B=1.
- **204 ENUMERATOR** — `spmw_autoschedule.py`: `_with_weight_residency` +
  final 2× tail cross after `_with_crf_modes`; both `weight_resident∈{F,T}`
  emitted unconditionally, no shape branch.
- **205 COST** — `spmw_cost_models.py`: closed-form `_samsung_preload_cycles`
  / `_samsung_readback_cycles` / `_trace_batch_dim` / `_samsung_mk`; compose
  `weight_resident ? P+B*(E+R) : B*(P+E+R)`. New HW constants on the Samsung
  fixture (`_fixtures.py`) with citations: `PRELOAD_FAN/WR/CRF`,
  `READBACK_FAN/RD` → P=11368, R=181 @4096×1024. B=1 both branches = P+E+R
  (parity); P/R placement-invariant so existing argmin winner unchanged.
- **206 CODEGEN+RUN** — `spmw_codegen.py`: `Compiled.run_batched` +
  `_run_samsung_batched`/`_samsung_batched_invoke`; `pim_driver.cc` gains
  `--batch B` + `--native-rebaseline`, resident = preload-once + B*(exec+
  readback), native = B*(preload+exec+readback), both via the SAME frozen
  `executeGemvFaithful`+`readResult` per vector (frozen-fn diff = 0). Driver
  rebuilt with `scons pim_driver -j32`.

Measured (faithful run, 4096×1024): B=1 → 15251 (ties single-shape floor,
byte-identical to pre-026); B=2 tenon 19103 < native 30469; B=4 26816 <
60967; B=8 42273 < 121488. Tenon preload clocked once (11368) vs B× native.
Each batched vector's output blob is byte-identical to the validated
single-vector faithful path (inherits fp16 fidelity ≤0.0156).

Test: `tests/spmw/test_samsung_batched_gemv.py` (9 tests). Full
`tests/spmw/` suite re-run for regressions.

NOTE for verifier/architect: the cost model's `exec_cyc` is the existing
*coarse* per-candidate post-crf body (488 for the winner), NOT the
faithful-measured exec (3702). So the cost-model sums (P+E+R = 12037 @B=1)
differ from the §3.6 faithful anchors (15251). This is correct per §3.5
(cost model is a cheap monotone objective; the absolute 15251/19134/etc.
anchors are the *faithful-run* numbers, which the driver path reproduces
exactly). The cost-model STRUCTURE matches §3.6 (B=1 tie, B≥2 resident
strictly cheaper, zero-preload falsifier, B* invariance) — all asserted.
