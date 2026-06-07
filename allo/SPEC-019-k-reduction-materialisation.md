# SPEC-019 — K-reduction materialisation for AiM and UPMEM

Status: ready for coder
Covers: TASK_DESCRIPTION.md Task 4 (AiM) and Task 5 (UPMEM)
Affects: `experiments/allo/allo/spmw_codegen.py` (only)
Does **not** touch: shared Allo files, the matcher, the regalloc, the
cost models. No changes to `MatchedOp`'s schema.

---

## 1. Problem

Each `MatchedOp` for `target_op_name == "MAC"` represents the full
inner reduction of one work-item's GEMV row — the matcher emits one
match per layer, not one match per K-iteration (see `_fixtures.py:19`
and `spmw_match_engine.py:484`). The MLP fixture has two layers
(256×128 = 32,768 MACs and 64×256 = 16,384 MACs); the matcher
produces two MatchedOps, total semantic MACs = 49,152.

Two backends currently emit one ISR / one C statement per MatchedOp
without unrolling K:

- **AiM** (`AimCtx.cmd("MAC_SBK", ...)`, `spmw_codegen.py:419`): emits
  one positional ramulator2 trace line per call with the default
  `opsize=1` (`spmw_codegen.py:465`). ramulator2 reads opsize and
  issues exactly `opsize` DRAM column requests
  (`aim_DRAM_system.cpp:310` — `for (int i = 0; i < opsize; i++)`).
  Two MatchedOps × opsize=1 = two single-MAC simulation events, priced
  at 219 cycles total.

- **UPMEM** (`MAC` op emit lambda, `_fixtures.py:439-443`): emits
  `acc += a * b;` as a single C statement into `task.c`. uPIMulator's
  DPU simulator prices the resulting two-statement body at 12,004
  cycles, dominated by DPU startup + MRAM transfer setup, not by the
  49k MAC iterations the body should represent.

Samsung already handles this correctly via `_emit_inner_loop_jump`
(`spmw_codegen.py:1037`), which is wired through `SamsungCtx.after_match`
(`spmw_codegen.py:269`). AiM and UPMEM need analogous, but
backend-shape-appropriate, fixes.

## 2. Where the K bound lives

`MatchedOp.enclosing_loops` is a list of `(iv_name, lb, ub, step)`
tuples, outer-to-inner (see `spmw_match.py:55` and the populator at
`spmw_match_engine.py:399`). For a row-of-MAC site nested inside an
`affine.for k = 0 to K`, the innermost loop is the K reduction.

The Samsung path already uses this:

```python
# spmw_codegen.py:1048
inner = match.enclosing_loops[-1]
ub = _parse_loop_bound(inner[2])
```

`_parse_loop_bound` (`spmw_codegen.py:1020`) accepts `"K"`, `"() -> (K)"`,
or `"affine_map<() -> (K)>"` strings and returns the integer K. AiM
and UPMEM reuse this helper verbatim; **no new MatchedOp field is
needed** and `spmw_match.py` is not edited.

Edge cases the existing helper handles:
- bound text is a plain int → return it
- bound text is `affine_map<() -> (1024)>` → regex pulls `1024`
- bound text contains multiple ints or symbolic atoms → returns `None`

When `_parse_loop_bound` returns `None`, the backend MUST fall back to
the current behaviour (opsize=1 / no K loop) and emit a `/* */`
comment naming the unparseable bound so the regression is debuggable.
The fall-back is not an exception path — the matcher does not
currently guarantee parseable bounds for all workloads, and the
existing Samsung JUMP code makes the same choice
(`spmw_codegen.py:1052-1053`).

## 3. AiM — chosen path: opsize=K on a single MAC_SBK ISR

### 3.1 Why opsize=K, not K unrolled lines

Examined `experiments/simulators/aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp:240-336`.
The AiM DRAM controller processes a `MAC_SBK` request by entering the
loop at line 310:

```cpp
for (int i = 0; i < opsize; i++) {
    int64_t channel_mask = ch_mask;
    aim_req.col_addr = host_req.col_addr + i;
    for (int cnt = 0; cnt < channel_count; cnt++) {
        ...
        if (m_controllers[channel_id]->send(aim_req) == false) {
            remaining_AiM_requests[channel_id].push(aim_req);
            ...
        }
    }
}
```

Each iteration sends one column request to the DRAM controller with
the column address incremented by 1. The cycle accounting falls out of
the per-request DRAM controller pipeline (tRC, tRCD, etc. set in the
GDDR6 config) — i.e. opsize=K is priced as K column accesses, not as
one cycle.

The MAC_SBK ISR's legal field set explicitly includes `opsize`
(`request.h:331-340`):

```cpp
opcode_str_to_aim_ISR["ISR_MAC_SBK"] = AiMISR(
    Opcode::ISR_MAC_SBK,
    {AiMISR::Field::opsize, AiMISR::Field::channel_mask,
     AiMISR::Field::bank_index, AiMISR::Field::row_addr},
    ...);
```

…and the AiM ISA documentation comment in `request.h:48` describes
MAC_SBK exactly as a multi-column burst over one bank.

This is the **native** semantic — opsize is the burst length. Pricing
opsize=K therefore corresponds 1:1 to K MAC operations against
consecutive DRAM columns, which is the AiM hardware model for a row
reduction. We choose opsize=K, not K unrolled trace lines, because:

1. The trace file size stays constant (~6 lines / MatchedOp instead of
   49,152 lines).
2. We match ramulator2's expected input shape rather than synthesising
   what is effectively a software unroll the simulator was designed to
   internalise.
3. The per-request DRAM controller pipeline is the source of truth for
   cycle cost — pricing K opsize=1 requests vs one opsize=K request
   gives the same cycle total *modulo* batched request-queue
   scheduling, and the opsize=K path is what real AiM silicon issues.

### 3.2 Where the K bound is read

The AiM target's `Op("MAC")` registration in `_fixtures.py:297-307`
uses an emit lambda:

```python
emit=lambda x, y, acc, ctx: ctx.cmd(
    "MAC_SBK", dst=acc, src0=x, src1=y),
```

The lambda has no access to the current `MatchedOp` — `_walk_and_emit`
calls `emit(bindings["x"], bindings["y"], bindings["acc"], ctx)` at
`spmw_codegen.py:1203`. We do **not** change the emit lambda
signature (that would ripple through every backend's fixture).
Instead we use the `after_match` hook (already present on the base
ctx, `spmw_codegen.py:103`), which receives the `match` and can
retroactively rewrite the last-emitted MAC_SBK line's opsize field.

Alternative considered and rejected: thread `match` into `emit` as a
new positional arg. This would require updating every backend's
`Op("MAC")` registration in `_fixtures.py` and every backend's emit
lambdas — five backends × multiple ops — and breaks SPEC-009's
"emit lambdas are minimal closures" rule. The `after_match` rewrite
is local to `AimCtx`.

### 3.3 Implementation — `AimCtx`

Add an override of `after_match` on `AimCtx` (parallel to Samsung's
override at `spmw_codegen.py:269`):

```python
def after_match(self, match, n_emitted):
    """Fold the inner K reduction into the just-emitted MAC ISR's
    opsize field. ramulator2 prices MAC_SBK opsize=N by issuing N
    column-address requests; see SPEC-019 §3.1 for the simulator
    citation. No-op when the match is not a MAC, when n_emitted is
    not 1 (compound emits we don't know how to fold), or when the
    inner-loop upper bound cannot be reduced to a constant.
    """
    if match.target_op_name != "MAC":
        return
    if n_emitted != 1:
        # We only know how to fold a single MAC_SBK line. Compound
        # emits (e.g. future MAC_ABK + RD_MAC pair) need a wider
        # fold; until then, leave opsize=1.
        return
    if not match.enclosing_loops:
        return
    inner = match.enclosing_loops[-1]
    k = _parse_loop_bound(inner[2])
    if k is None or k <= 1:
        return
    # Rewrite the last emitted positional trace line: substitute the
    # opsize token (field index 0 after the "AiM MAC_SBK" prefix,
    # per _ISR_FIELDS["MAC_SBK"] = ("opsize", ...)).
    last = self.cmds[-1]
    parts = last.split(" ")
    # Expect: ["AiM", "MAC_SBK", "<opsize>", "<channel_mask>",
    #          "<bank_index>", "<row_addr>"]
    if len(parts) < 3 or parts[0:2] != ["AiM", "MAC_SBK"]:
        return
    parts[2] = str(k)
    self.cmds[-1] = " ".join(parts)
    # Mirror the human-readable annotation so test introspection
    # stays consistent with the positional trace.
    human_last = self._human_lines[-1]
    if "opsize=" in human_last:
        # Replace the existing key=value form.
        import re
        self._human_lines[-1] = re.sub(
            r"opsize=\d+", f"opsize={k}", human_last)
    else:
        self._human_lines[-1] = f"{human_last}  opsize={k}"
```

That is the entire AiM-side change. The MAC op registration in
`_fixtures.py` is untouched.

### 3.4 Side effects to confirm

- `tests/spmw/test_target_aim.py:50` asserts on the exact positional
  token list `["AiM", "MAC_SBK", "1", "1", "3", "0"]`. That test
  fabricates a MatchedOp-less ctx call (`ctx.cmd("MAC_SBK", ...)`
  with no enclosing match), so `after_match` is never invoked and
  the assertion still passes. **No test update required.**
- `tests/spmw/test_run.py` and `tests/spmw/test_target_aim.py`
  synthetic-trace tests pass MatchedOps without `enclosing_loops`
  populated — `after_match` early-returns on empty loops list, so
  these tests are unaffected.
- `test_e2e_mlp_aim` runs the real matcher, which **does** populate
  `enclosing_loops` for the MAC site (the affine.for K loop is
  guaranteed to wrap the linalg.matmul lowering); the emitted trace
  for the MLP will now contain `MAC_SBK <K> 1 <bank> 0` per layer.

### 3.5 Success criterion (AiM)

After the fix, `test_e2e_mlp_aim` produces an AiM trace whose two
`MAC_SBK` lines carry `opsize=128` and `opsize=256` respectively (the
K bounds of the MLP's two layers). ramulator2 then issues
`128 + 256 = 384 column requests per channel × NumChannels` MAC
operations. The cycle count moves from 219 to a number reflecting
that workload; the test asserts `result.cycles > 0` and the
MLP_PERFORMANCE_REPORT.md is updated by goal-check with the actual
number.

**Validation grep** (the coder runs this after the fix):

```bash
grep -E 'MAC_SBK' <generated trace file>
# Expected: "AiM MAC_SBK 128 1 0 0" and "AiM MAC_SBK 256 1 0 0"
# (NOT "AiM MAC_SBK 1 1 0 0" — that was the bug.)
```

## 4. UPMEM — chosen path: explicit `for (k=0; k<K; ++k)` wrapper

### 4.1 Why an explicit C loop, not a cost-model multiplier

TASK_DESCRIPTION.md §Task 5 names path (a) as preferred — keep
uPIMulator as ground truth, give one number not two. Path (b) — multiply
the simulator's two-statement cost by K in Python — was rejected
because (i) it adds a "validation mode" vs "perf mode" gate to the
test, (ii) it makes the cycle number a synthetic composite of
simulation + Python multiplication rather than a direct simulator
read, and (iii) it diverges from Samsung's and AiM's approach of
materialising the reduction into the trace.

The DPU C compiler will compile the loop down to its native ARC-like
instruction stream; uPIMulator prices the resulting instructions at
its per-cycle DPU model. The full 12,004-cycle startup/MRAM overhead
will *still* be reported, but it is now amortised across K × N
iterations of the actual body, which is the correct economic picture.

### 4.2 Where the K bound is read

Same source as AiM: `match.enclosing_loops[-1]` parsed via
`_parse_loop_bound`. Same fall-back: when K cannot be reduced to a
constant, emit the original single statement and a comment naming the
bound string.

### 4.3 Implementation — `UPMEMCtx`

Two coupled changes are needed because UPMEM's MAC emit lambda
(in `_fixtures.py:439-443`) currently emits one C statement directly
via `ctx.emit_c_line`:

```python
emit=lambda x, y, acc, ctx: ctx.emit_c_line(
    "{acc} += {a} * {b};".format(
        acc=ctx.handle_c_name(acc),
        a=ctx.handle_c_name(x),
        b=ctx.handle_c_name(y))),
```

We cannot leave that lambda intact and patch in `after_match` (as we
did for AiM) because the body itself must change: the `a` and `b`
references must become `a[k]` and `b[k]` so the loop has work to do.
The pattern used by Samsung and now AiM — patch the last-emitted line
in `after_match` — does not work here because the body name
references are baked in at emit time by `handle_c_name`.

Therefore we take a **two-step approach** that keeps the fixture
edit additive:

**Step 1 — add a UPMEM-specific MAC-emit helper on `UPMEMCtx`** in
`spmw_codegen.py` (parallel to APUv1Ctx's `emit_mac_lookup`,
`spmw_codegen.py:786`):

```python
def emit_mac_kreduce(self, acc, x, y, k_bound: int | None) -> None:
    """Emit a K-reduction MAC body.

    When `k_bound` is a positive int, emits:
        for (unsigned k = 0; k < <k_bound>; ++k) {
            <acc> += <x>[k] * <y>[k];
        }
    Otherwise falls back to the un-looped form `<acc> += <x> * <y>;`
    and a `/* k bound unknown */` comment. The fallback preserves the
    pre-SPEC-019 behaviour so synthetic test traces (no
    enclosing_loops) still emit a one-statement body.

    Note: x/y must be MemoryRefs (so `[k]` indexing makes C-level
    sense); when either is a bare Register, falls back to un-looped
    form as well — a register operand has no per-K subscript.
    """
    acc_c = self.handle_c_name(acc)
    x_c = self.handle_c_name(x)
    y_c = self.handle_c_name(y)
    # `handle_c_name` for a MemoryRef returns `<env_name>[<idx>]`
    # (see spmw_codegen.py:574-578). For K-reduction we need the
    # per-K subscript inside the loop, so we strip the rendered
    # `[<idx>]` suffix and append `[k]` instead. For a Register
    # or whole-Memory operand the rendered name has no `[...]`
    # suffix; in that case we cannot subscript and fall back.
    def _kify(c_name: str) -> str | None:
        if c_name.endswith("]"):
            head = c_name.rsplit("[", 1)[0]
            return f"{head}[k]"
        return None  # Register / whole-Memory — not k-indexable.
    x_k = _kify(x_c)
    y_k = _kify(y_c)
    if k_bound is None or k_bound <= 1 or x_k is None or y_k is None:
        # Fallback — keep pre-SPEC-019 behaviour.
        if k_bound is None:
            self.emit_c_line(f"/* SPEC-019: k bound unparseable; emitting un-looped MAC */")
        self.emit_c_line(f"{acc_c} += {x_c} * {y_c};")
        return
    self.emit_c_line(f"for (unsigned k = 0; k < {k_bound}; ++k) {{")
    self.emit_c_line(f"    {acc_c} += {x_k} * {y_k};")
    self.emit_c_line("}")
```

**Step 2 — switch the UPMEM target's `Op("MAC")` registration in
`tests/spmw/_fixtures.py:431-444`** to call the helper:

```python
allo.op(
    "MAC",
    src=(allo.or_(any_wram, any_gpr),
         allo.or_(any_wram, any_gpr)),
    dst=any_gpr,
    accumulates=True,
    fn=lambda x, y, acc: acc + x * y,
    cycles=2,
    emit=lambda x, y, acc, ctx: ctx.emit_mac_kreduce(
        acc=acc, x=x, y=y,
        k_bound=ctx.pending_k_bound),
)
```

The `pending_k_bound` channel: `_walk_and_emit` is the only caller
of `emit`, and it has `match` in scope (`spmw_codegen.py:1182`). We
add a one-line set/clear on the ctx **only for UPMEM** so the fixture
lambda can read K without changing the global `emit` signature.

The exact addition to `_walk_and_emit` (the only shared-walker edit
in this spec) is gated on `isinstance(ctx, UPMEMCtx)` so other
backends are not affected:

```python
# spmw_codegen.py around line 1190, inside the per-match loop —
# just before the `emit(...)` call.
if isinstance(ctx, UPMEMCtx) and match.target_op_name == "MAC":
    inner = match.enclosing_loops[-1] if match.enclosing_loops else None
    ctx.pending_k_bound = _parse_loop_bound(inner[2]) if inner else None
else:
    # Other backends ignore pending_k_bound; clear it for hygiene.
    if hasattr(ctx, "pending_k_bound"):
        ctx.pending_k_bound = None
```

…with `pending_k_bound = None` initialised in `UPMEMCtx.__init__`
(after `self._name_table = {}` at `spmw_codegen.py:540`):

```python
# SPEC-019: channel for the inner-K bound, set by _walk_and_emit
# immediately before invoking the MAC emit lambda.
self.pending_k_bound: int | None = None
```

That is the entire UPMEM-side change. AiM is **not** affected by this
walker edit (the isinstance gate keeps it inert), and the other three
backends are also inert.

### 4.4 Why a ctx attribute, not a richer emit signature

Considered: pass `match` as a 5th positional arg to every `op.emit`
lambda. Rejected — five backends, multiple ops each, and SPEC-009 §B
established the 4-arg signature as the public contract. The
ctx-attribute approach is a tactical workaround for one backend
(UPMEM), explicitly scoped to it, and documented in code as such. If
a second backend needs the same channel, the right move is to revisit
the emit signature in a future spec — not to expand the side channel.

### 4.5 Side effects to confirm

- `tests/spmw/test_run.py` UPMEM-flavoured synthetic traces pass
  MatchedOps with empty `enclosing_loops`. The new
  `emit_mac_kreduce` falls back to `acc += a * b;`, identical to the
  pre-SPEC-019 emit. **No test update required.**
- `tests/spmw/test_target_upmem.py` (if it exists) likely asserts on
  exact emitted C strings; the coder must grep for any test asserting
  on `"acc += a * b;"`-shape strings and update them only if the
  matched trace there populates `enclosing_loops` (in which case the
  new for-loop form is correct and the assertion is what needs to
  change). Current grep target:
  ```bash
  grep -rn 'acc.*+=\|emit_mac_kreduce\|emit_c_line.*\\*' \
      experiments/allo/tests/spmw/
  ```
- The fixture file `experiments/allo/tests/spmw/_fixtures.py` is the
  *only* file outside `spmw_codegen.py` that this spec edits. It is a
  test fixture (not shared with FPGA/AIE paths), so changing the
  UPMEM target tree's `MAC` op emit is in-scope for SPMW. The fixture
  edit is additive — the emit closure points at a new ctx method;
  the old behaviour is recovered by the helper's fallback branch.

### 4.6 Success criterion (UPMEM)

After the fix, `test_e2e_mlp_upmem` emits a `task.c` whose
`tenon_kernel(...)` body contains two explicit `for (unsigned k = 0;
k < <N>; ++k)` loops, with `<N>` ∈ {128, 256} matching the MLP's
layer K bounds. uPIMulator runs the resulting binary and reports a
cycle count *different* from 12,004 (specifically larger, because
49,152 iterations now actually execute on the DPU simulator's
per-instruction cost model).

**Validation grep** (the coder runs this after the fix):

```bash
grep -E 'for \(unsigned k = 0; k < (128|256); \+\+k\)' \
    <generated task.c>
# Expected: two matches, one per MLP layer.
```

## 5. Files changed by this spec

| File | Edit | Purpose |
|---|---|---|
| `experiments/allo/allo/spmw_codegen.py` | + `AimCtx.after_match` override | Fold K into opsize on the just-emitted MAC_SBK |
| `experiments/allo/allo/spmw_codegen.py` | + `UPMEMCtx.emit_mac_kreduce` method | Emit `for (k...)` wrapper around `acc += a[k] * b[k]` |
| `experiments/allo/allo/spmw_codegen.py` | + `UPMEMCtx.__init__` initialises `pending_k_bound` | Side channel for K bound |
| `experiments/allo/allo/spmw_codegen.py` | + one `isinstance(ctx, UPMEMCtx)` block in `_walk_and_emit` | Populate `pending_k_bound` before the MAC emit |
| `experiments/allo/tests/spmw/_fixtures.py` | UPMEM `Op("MAC")` `emit=` → `ctx.emit_mac_kreduce(...)` | Wire the helper in |

**No edits to**:
- `spmw_match.py` (`MatchedOp` schema unchanged)
- `spmw_match_engine.py` (matcher unchanged)
- `spmw_target.py` (target-tree API unchanged)
- `spmw_autoschedule.py`, `spmw_regalloc.py`, `spmw_cost*.py`
  (all upstream of codegen — not affected)
- Any shared Allo file outside `spmw_*.py`.

## 6. Rollback story

If FPGA / AIE CI breaks: the only shared-file edit is the
`isinstance(ctx, UPMEMCtx)` block in `_walk_and_emit`. It is guarded
by an isinstance check on a SPMW-only ctx class; FPGA/AIE paths do
not construct `UPMEMCtx` and therefore cannot enter the branch.

If AiM trace asserts in upstream tests start firing: the
`AimCtx.after_match` override is opt-in (the base ctx no-ops) and
only triggers when (i) the match is a `"MAC"`, (ii) exactly one line
was emitted, and (iii) the inner loop bound parses to a constant > 1.
Tests that build a ctx directly and call `cmd(...)` without a wrapping
match never invoke `after_match` and are unaffected.

Reverting is a clean `git revert` of the codegen + fixture commit;
no migration of artifacts is needed.

## 7. Tests the coder must keep green

Upstream (FPGA / AIE / tensor) gates — none, since no shared file is
touched semantically.

SPMW tests this spec must not regress:

- `experiments/allo/tests/spmw/test_target_aim.py`
  (`test_target_aim_mac_sbk_positional` — exact-token assert; should
  pass because `after_match` is not invoked by the direct-ctx-call
  path used in that test).
- `experiments/allo/tests/spmw/test_run.py` (synthetic AiM and UPMEM
  traces with empty `enclosing_loops`; both backends fall back to
  pre-SPEC-019 behaviour).
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_aim` —
  must continue to produce a positive cycle count; the new count will
  differ from 219.
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_upmem` —
  must continue to produce a positive cycle count; the new count will
  differ from 12,004.

## 8. Open design tensions (recorded, not blocking)

- The ctx-attribute side channel (`pending_k_bound`) is a tactical
  workaround for one backend. If a second backend later needs the
  same data, the right move is a fresh spec that revisits the
  4-arg emit signature, not adding more attributes.
- The `_kify` C-name rewrite in `emit_mac_kreduce` strips a rendered
  `[<idx>]` suffix and replaces it with `[k]`. This is correct for
  the current `handle_c_name` output shape
  (`spmw_codegen.py:574-578`) but assumes the suffix is the per-K
  index. If the autoscheduler ever produces a 2D-strided memref
  whose name renders as `bufferA[stride*k + offset]`, this rewrite
  will be wrong. The fallback comment provides a debuggable trail.
- Trace-line rewriting in `AimCtx.after_match` reaches back into
  `self.cmds[-1]` and `self._human_lines[-1]`. Cleaner alternatives
  (defer emit until `after_match` knows K, or precompute opsize
  before the emit lambda fires) would require either delaying
  emission inside `cmd()` or extending the emit signature; both
  ripple further than the current rewrite. Recorded here, not fixed.

## 9. Implemented

- 2026-05-19 (task 005, AiM half): `AimCtx.after_match` override added
  in `experiments/allo/allo/spmw_codegen.py` per §3.3. Rewrites the
  opsize token (positional field index 2) of the just-emitted
  `AiM MAC_SBK ...` line from `1` to the inner-K bound parsed via
  `_parse_loop_bound`. Mirrors the rewrite into `_human_lines[-1]`.
  Five new unit tests cover the fold path, the affine_map<()->(N)>
  shape, and the three documented no-op fall-backs (non-MAC, empty
  enclosing_loops, unparseable bound).
- 2026-05-19 (task 006, UPMEM half): `UPMEMCtx.emit_mac_kreduce` added
  in `experiments/allo/allo/spmw_codegen.py` per §4.3; emits
  `for (unsigned k = 0; k < <K>; ++k) { acc += x[k] * y[k]; }` when the
  inner-K bound parses, else falls back to the pre-SPEC-019 one-liner.
  `UPMEMCtx.__init__` initialises `pending_k_bound = None`.
  `_walk_and_emit` populates `ctx.pending_k_bound` immediately before
  the MAC emit lambda fires, gated on `isinstance(ctx, UPMEMCtx)` and
  `match.target_op_name == "MAC"` (other backends untouched). The UPMEM
  `Op("MAC")` registration in `experiments/allo/tests/spmw/_fixtures.py`
  now calls `ctx.emit_mac_kreduce(acc=acc, x=x, y=y,
  k_bound=ctx.pending_k_bound)`.
  Deviation from spec §4.3 helper-code sketch: dropped the
  `/* SPEC-019: k bound unparseable */` debug comment in the fallback
  branch — emitting it would break the §4.5 "no test update required"
  invariant for `test_target_upmem.py::test_upmem_ctx_emits_c` (asserts
  `len(ctx.cmds) == 1`). Documented in code at the fallback site.
