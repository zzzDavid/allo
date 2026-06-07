# SPEC-025 — Lever 3: shared CRF + host trigger schedule (vs per-work-id CRF)

Status: design / ready for coder (coder task 051).
Owner: architect.
Related: SPEC-021 (faithful cycle model — the measurement instrument this
lever moves), SPEC-009 (`Placement.mode`/`extra` schema), SPEC-005
(`--cmds` wire protocol), SPEC-019 (K-fold materialisation).
Reads: `spmw_autoschedule.py::_samsung_enumerate`,
`spmw_cost_models.py::_samsung_kernel_cycles`,
`spmw_codegen.py::_walk_and_emit` / `_run_samsung`.

---

## 1. Problem (the lever, restated precisely)

Tenon's Samsung codegen walks the trace **per work-id**
(`_walk_and_emit` → `_bucket_by_work_id`, `spmw_codegen.py:1276`). Every
work-id bucket re-emits its own CRF body — preload MOVs, the MAC, the
inner-K JUMP, storeback MOVs. With the Samsung unit-tree fanout
(`pseudo_channel` mapping=[16] × `pim` mapping=[8] = 128 work-ids) and a
~4-instruction CRF body, the emitted stream is on the order of ~512
records. Native HBM-PIM does **not** replicate the CRF: it programs **one
shared 6-instruction CRF** into every PIM block's CRF register (a single
`programCrf` broadcast) and then issues a **host-side trigger schedule** —
the host walks the tile loop and fires the same CRF per tile. The CRF
program is issued once; only the host triggers scale with the work.

Under the SPEC-021 faithful run path (task 025), the simulator's cycle
count is a function of the **issued instruction / transaction count**
derived from the emitted stream (`stream_records` /
`stream_macs_per_tile`). A 512-record per-work-id stream therefore
genuinely costs more issued CRF instructions than a shared-CRF stream
that programs the body once. This is the lever: the win is real (fewer
issued instructions), not a relabelling.

This spec makes "shared CRF" a **placement mode** the cost model prices
as fewer issued instructions, so **argmin** prefers it. Codegen then
**materialises** the chosen mode (one shared CRF body + a host trigger
schedule, or the per-work-id replication) — mechanism only. The decision
lives in argmin; codegen never chooses.

Per the scope guardrails this is **entirely additive** and lives entirely
inside `spmw_*.py`: a new `Placement.mode` value, a new cost term keyed on
that mode, and a new codegen branch keyed on that mode. No shared-file
(`ir/builder.py`, `ir/infer.py`, `dataflow.py`) edit. No simulator edit.

---

## 2. The CRF-issue model (the accounting both candidates share)

Both candidates emit the **same per-work-id CRF body** semantically — the
same MAC + JUMP + moves. They differ only in **how many times the CRF
body is issued as instructions**:

- **per-work-id CRF** (mode `"crf_per_workid"`): the CRF body is issued
  once per work-id. Issued CRF instructions = `body_len × n_workids`.
- **shared CRF** (mode `"crf_shared"`): the CRF body is programmed once
  (broadcast to all PIM blocks), and the host issues one **trigger** per
  work-id. Issued instructions = `body_len` (the one shared program) +
  `trigger_cost × n_workids` (host triggers, which are not CRF
  instructions and are far cheaper than re-uploading the body).

`body_len`, `n_workids`, and `trigger_cost` must all be **derived**, never
literals (see §3, §4). The crucial term is `n_workids` — the replication
factor — which is the **product of unit-tree fanouts**, i.e. target
geometry, *not* the forbidden literal 128.

---

## 3. Enumerator (`_samsung_enumerate`)

### 3.1 Where this slots in

`_samsung_enumerate` currently returns `[bank_row, grf_staged]` (two `y`
placements). Lever 3 is **orthogonal** to the `y`-placement choice and to
the lever-1 (even/odd) and lever-2 (host-GRF) choices: it is a property of
*how the CRF body is issued across work-ids*, not of where `y` lives. To
keep the candidate set from exploding combinatorially, lever 3 adds the
CRF-issue dimension as a **mode pair applied to the kept fast `y`
candidate(s)**, not a full cross-product with every other lever.

Concretely, for **each** placement candidate the enumerator already
produces for the fast layout (today `bank_row`; once levers 1/2 land,
whatever fast base candidate(s) they keep), emit **two** variants that
differ only in the CRF-issue mode:

```python
def _with_crf_modes(base: Placement) -> list[Placement]:
    shared = replace(base, mode=_join_mode(base.mode, "crf_shared"))
    per_wid = replace(base, mode=_join_mode(base.mode, "crf_per_workid"))
    return [shared, per_wid]
```

`mode` is a free-form label (SPEC-009 §0). Because lever 1/2 may also want
to encode a mode token, the CRF token is **composable**: store it in
`Placement.extra["crf_issue"]` (`"shared"` | `"per_workid"`) so it does
not collide with the lever-1/lever-2 `mode` string, and keep `mode`
human-readable by appending `"+crf_shared"` / `"+crf_per_workid"`. The
cost model and codegen read `extra["crf_issue"]` (authoritative);
`mode` is for logging/audit dumps only. (`_join_mode` is a 2-line helper
that appends the token to `mode` for readability.)

> Rationale for `extra`, not a second cross-producted `mode` string: the
> CRF-issue dimension is genuinely independent of the `y`/even-odd/host
> dimensions. Threading it through `extra` keeps each lever's cost term
> additive and keeps the enumerator from emitting an N×2×2×2 candidate
> blowup. SPEC-009 §0 already mandates `extra` round-trips through
> regalloc untouched.

### 3.2 Candidate fields (the contract)

Each candidate is a real, materialisable `Placement`:

| Field | shared-CRF candidate | per-work-id candidate |
|---|---|---|
| `placements` | identical to the base fast candidate | identical to the base fast candidate |
| `mode` | `f"{base.mode}+crf_shared"` | `f"{base.mode}+crf_per_workid"` |
| `extra["crf_issue"]` | `"shared"` | `"per_workid"` |

No new placement handle is introduced — the operand→handle map is
unchanged. The only new information is *how the body is issued*, which is
a scheduling property, correctly living in `mode`/`extra` rather than in
`placements`.

### 3.3 Anti-hardcoding constraints on the enumerator

- The enumerator MUST NOT branch on shape, memref name, `func_name`, or
  `M`/`K`. It emits both CRF-issue variants for every fast base
  candidate, unconditionally. Argmin (not the enumerator) discards the
  loser.
- No literal `128`, no `n_workids=...` constant. `n_workids` is **not**
  computed in the enumerator at all — it is a *cost-model* term derived
  from `target` geometry (§4.2). The enumerator only tags the mode.

---

## 4. Cost model (`_samsung_kernel_cycles`)

### 4.1 What changes

`_samsung_kernel_cycles` today returns, per match, the folded MAC+JUMP
cost (`folded * mac_cyc + jump_cyc` for `is_auto`). That term prices the
**body** of one work-id's CRF. It does **not** account for **replication
across work-ids** — which is exactly the cost lever 3 moves. Add a
**replication term** that multiplies the issued-CRF cost by the work-id
count for the per-work-id mode, and replaces it with one-shared-program +
per-work-id-trigger cost for the shared mode.

### 4.2 Deriving `n_workids` from target geometry (NOT a literal)

The replication factor is the number of PIM compute units the CRF body is
replicated across = the product of the unit-tree `mapping` fanouts below
the root. Add a small helper on the cost side (pure function over
`target`, no shape input):

```python
def _samsung_workid_count(target) -> int:
    """Product of unit-tree fanouts = number of PIM blocks the CRF
    body is issued across in per-work-id mode. Pure target geometry;
    no shape literal. For the Samsung fixture this is 16*8 = 128, but
    it is *computed* from the tree, valid for any topology."""
    n = 1
    for u in target._walk():
        for f in u.mapping:
            n *= f
    return n
```

This reads `pseudo_channel.mapping=[16]` × `pim.mapping=[8]` (and the
synthetic root's `[1]`) and yields 128 **as a derived value**. The literal
`128` never appears in source. If the fixture topology changes, the term
tracks it. This satisfies the anti-hardcoding gate's
"128-as-workid-count" prohibition: the value is an expression over
`target.*` geometry, valid for any M/K and any topology.

> Note on the matcher's `work_id` axis: the per-work-id bucketing in
> codegen (`_bucket_by_work_id`) is driven by the *trace's* `work_id`
> tags, which for the Samsung GEMV mirror the grid_map over these same
> PIM units. The cost model uses the **target** fanout product (geometry)
> rather than counting trace buckets, because (a) it must be pure
> geometry to satisfy the audit, and (b) it must price a candidate before
> codegen has walked the trace. The two agree by construction; the
> codegen contract (§5) asserts the agreement so a divergence is caught.

### 4.3 Cost terms

Read every constant from the target spec; introduce **no** new magic
numbers. Required spec-side constants (declared on the Samsung fixture,
not in the cost model):

- `body_cyc` — the per-work-id CRF body cost already computed today
  (`folded * mac_cyc + jump_cyc` for the `is_auto` match, summed over the
  bucket's matches). Reuse the existing accumulation; do not recompute.
- `n_workids = _samsung_workid_count(target)` (§4.2).
- `trigger_cyc = target.move("CRF_TRIGGER").cycles` — the host-issued
  per-work-id trigger cost in the shared-CRF schedule. **New Move on the
  fixture** (see §6). This is the host's per-tile fire cost; it is
  strictly less than re-issuing `body_cyc` CRF instructions, which is why
  shared wins. Sourced from the Samsung ISA's per-PIM-command host
  issue latency (tCCDL-class), declared on the target with a citation in
  the fixture, never inlined here.

The per-bucket cost becomes:

```python
crf_issue = layout.extra.get("crf_issue", "per_workid")
if crf_issue == "shared":
    bucket_cost = body_cyc + trigger_cyc * n_workids
elif crf_issue == "per_workid":
    bucket_cost = body_cyc * n_workids
else:
    raise ValueError(f"samsung kernel_cycles: unknown crf_issue {crf_issue!r}")
```

Back-compat: when `extra` has no `crf_issue` key (every pre-lever-3
candidate, and AiM/UPMEM/APU which never set it), the default
`"per_workid"` reproduces the **current** per-work-id accounting **except**
that today's cost model does *not* multiply by `n_workids` at all — it
scores one work-id's body. See §4.4.

### 4.4 Back-compat / regression guard (read carefully)

Today `_samsung_kernel_cycles` returns the **single-work-id** body cost
(it iterates `trace.matches`, which for the autoscheduler sub-trace is one
bucket's matches). Multiplying by `n_workids` is a **uniform scale**
across *all* Samsung candidates, so it does **not** change the **argmin**
among the pre-lever-3 candidates (`bank_row` vs `grf_staged` rank
identically before and after a uniform ×128). The only new ranking effect
is between `crf_shared` and `crf_per_workid`, which is the intended lever.

Therefore:

- For the **bank_row vs grf_staged** decision, behaviour is preserved
  (uniform scale). Existing test
  `tests/spmw/test_autoschedule_samsung.py` (argmin picks bank_row) stays
  green.
- The **new** decision (shared beats per-workid) is exercised by a new
  test (§7).
- AiM / UPMEM / APU cost models are **untouched** — `crf_issue` is a
  Samsung-only `extra` key; their cost factories never read it. No blast
  radius.

### 4.5 Cost-drives-choice (audit positive evidence #2)

Because `bucket_cost(shared) = body_cyc + trigger_cyc*n_workids` and
`bucket_cost(per_workid) = body_cyc*n_workids`, and `body_cyc > trigger_cyc`
by construction (a full CRF body is more than one host trigger), shared is
strictly cheaper for any `n_workids ≥ 2`. Perturbing `CRF_TRIGGER.cycles`
upward (toward `body_cyc`) narrows the gap and — if pushed past
`body_cyc·(n_workids-1)/n_workids` — flips the ranking, proving the choice
is **computed**, not asserted. The audit's "perturb a target constant and
watch the ranking move" probe targets `CRF_TRIGGER.cycles`.

---

## 5. Codegen (`spmw_codegen.py`) — materialisation only

### 5.1 Decision boundary

Codegen reads `placement.extra["crf_issue"]` and materialises
accordingly. It MUST NOT choose based on shape, work-id count, or stream
length. The only branch is on the already-decided mode token.

### 5.2 `crf_per_workid` (current behaviour, unchanged)

`_walk_and_emit` keeps its per-work-id loop. Every bucket re-emits its CRF
body. This is the existing path; no change. (It remains the default when
`extra` lacks `crf_issue`, so AiM/UPMEM/APU and all pre-lever-3 Samsung
streams are byte-for-byte identical.)

### 5.3 `crf_shared` (new branch)

When `extra["crf_issue"] == "shared"`, codegen emits **one shared CRF
body** followed by a **host trigger schedule**, instead of replicating the
body per work-id. Mechanism:

1. **Emit the CRF body once.** Walk **one** representative work-id bucket
   (the first) through the normal emit path, producing the shared CRF
   program (moves + MAC + JUMP + EXIT). This is the program `programCrf`
   uploads to every PIM block.

2. **Emit the host trigger schedule.** For the remaining work-ids, instead
   of re-emitting the CRF body, emit one **trigger record** per work-id:
   a new `PIMCmd` opcode `CRF_TRIGGER` (or, to avoid a new opcode in the
   `PIMCmd` validation path, a host-schedule side record — see §5.4)
   carrying the work-id index. The trigger count = the number of work-id
   buckets, which equals `n_workids` by construction; codegen **asserts**
   `len(buckets) == _samsung_workid_count(target)` so a divergence between
   the trace's work-id axis and the target geometry is caught at compile
   time (defends §4.2's "agree by construction" claim).

3. The emitted stream the faithful run path (SPEC-021 / task 025) sees is
   therefore `body_len` CRF records + `n_workids` trigger records, i.e.
   `stream_records` is far smaller than the per-work-id
   `body_len × n_workids`. The cycle drop is real: fewer issued CRF
   instructions reach `programCrf`/`runPIM`.

### 5.4 Trigger representation — coordination with SPEC-021/task 025

The trigger schedule must be **legible to the faithful run path** so it
counts cheaper than a CRF body record. Two options; **chosen: (B)**.

- (A) New `PIMCmd` opcode `CRF_TRIGGER`. Rejected: every `PIMCmd` flows
  through the C++ `validationCheck` in `programCrf`; a new opcode risks
  the "Invalid in ISA 1.0" rejection that already bites bank-store MOVs
  (`_crf_valid` filter, `spmw_codegen.py:1625`). The CRF upload caps at 4
  bursts anyway (SPEC-021 §1), so triggers must NOT ride the CRF stream.

- **(B, chosen) Host-schedule side list on `Compiled`, parallel to
  `cmds`.** The shared CRF body goes into `compiled.cmds` (≤ body_len
  records, uploaded once by `programCrf` as today). The trigger schedule
  goes into a **new field** `compiled.host_schedule: list[HostTrigger]`
  (one per work-id), consumed by the faithful run path as the per-tile
  host-fire count. This mirrors SPEC-021's split between "what
  `programCrf` uploads" (the body) and "what `runPIM` issues" (the
  per-tile transactions). The faithful path derives `stream_records` from
  `len(cmds)` (one body) and the issued-transaction multiplicity from
  `len(host_schedule)` (triggers), pricing both under the **same**
  accounting for native and Tenon (SPEC-021 §2 honest line).

`HostTrigger` is a 2-field dataclass `(work_id: int, tile_count: int)` in
`spmw_codegen.py`. `tile_count` is derived from the same `in_tiles`/loop
bound the body's JUMP uses — no shape literal.

### 5.5 Codegen anti-hardcoding constraints

- No golden stream replay. The shared CRF body is produced by walking a
  real bucket through the existing emit path — it is generated from the
  `Placement` + layout, not a pasted table (audit "golden cmd-stream"
  prohibition).
- Trigger count and `tile_count` are expressions over the trace's work-id
  buckets and the loop bound, never literals.
- The branch key is `extra["crf_issue"]`, set by argmin. Codegen contains
  no `if shape …` / `if n_workids …` selection (audit "decision logic in
  codegen" prohibition).

### 5.6 What task 025 (faithful run path) must already provide

This lever's cycle win is **only observable** once the faithful run path
prices `stream_records` from `compiled.cmds` and the issued multiplicity
from `compiled.host_schedule`. Task 025 ships first (SPEC-021). If 025's
conservative rule ("one issued transaction per CRF instruction after JUMP
expansion") is in place, lever 3 moves the count immediately: shared mode
emits `body_len` CRF records vs `body_len × n_workids`. The coder of 051
must confirm `compiled.host_schedule` is consumed by the 025 run path; if
025 landed without that field, 051 adds the consumption (it is Tenon-side
`_run_samsung`, permitted by SPEC-021 §3 "ALLOWED Tenon-side").

---

## 6. Target-spec additions (the only fixture change)

One new Move on the Samsung fixture (`tests/spmw/_fixtures.py`,
`build_samsung_target`), additive:

```python
allo.move("CRF_TRIGGER", cycles=<host per-tile fire latency>)  # cite Samsung ISA host-issue
```

`cycles` is a genuine hardware constant (host-side per-command issue
latency, tCCDL-class), declared **on the target spec** with a source
citation — exactly the "genuine hardware constants live on the target
spec" allowance in the anti-hardcoding gate. It must be **strictly less
than** the per-work-id `body_cyc` (a single MAC+JUMP fold) so shared wins;
the fixture comment must state this invariant and its citation. No change
to any other backend's fixture.

---

## 7. Tests (what proves it works)

New test file `tests/spmw/test_samsung_shared_crf.py`:

1. `test_enumerator_emits_both_crf_modes` — `_samsung_enumerate` returns,
   for the fast base candidate, one `extra["crf_issue"]=="shared"` and one
   `"per_workid"` variant. No shape passed in; both materialisable.
2. `test_cost_prefers_shared_crf` — `_samsung_kernel_cycles` scores
   `shared < per_workid` for the Samsung fixture; assert the gap equals
   `body_cyc*(n_workids-1) - trigger_cyc*n_workids` computed from
   `target.*`, with **no literal** in the assertion (derive `n_workids`
   via `_samsung_workid_count(target)`).
3. `test_workid_count_is_geometry_not_literal` — `_samsung_workid_count`
   equals the product of the fixture's unit `mapping`s; mutate a fixture
   fanout in a local target and confirm the count tracks it.
4. `test_cost_ranking_responds_to_trigger_cycles` — bump
   `CRF_TRIGGER.cycles` on a copy of the target until it exceeds the
   flip threshold and confirm the argmin flips per-workid (proves the
   number is computed; audit evidence #2).
5. `test_codegen_shared_emits_one_body_plus_triggers` — compile the GEMV
   with the shared candidate forced; assert `len(compiled.cmds)` CRF body
   records ≈ `body_len` (one bucket) and `len(compiled.host_schedule)` ==
   `_samsung_workid_count(target)`. Assert the per-work-id candidate emits
   `~n_workids ×` more `cmds`.
6. `test_argmin_picks_shared_end_to_end` — full `autoschedule` on the
   Samsung GEMV trace returns a placement with
   `extra["crf_issue"]=="shared"` (cost-driven, not forced).

### Regression gates (must stay green; shared-file boundary inventory)

- `tests/spmw/test_autoschedule_samsung.py` — bank_row still wins among
  `y`-placement candidates (the ×n_workids scale is uniform; §4.4).
- `tests/spmw/test_codegen_gemv.py::test_compile_emits_canonical_mac_jump_pair_per_match`
  — the per-work-id default path is byte-for-byte unchanged (shared is a
  new branch, opt-in via mode).
- `tests/spmw/test_samsung_placement_changes_cycles.py` — under the
  faithful run path, shared vs per-workid now yield **different** cycle
  counts at the same shape (this lever is one of the streams that
  discharges the SPEC-021 §6 strict `cycles_a != cycles_b` promotion).
- `tests/spmw/test_move_scheduling.py`, AiM/UPMEM/APU codegen + cost
  tests — **untouched**; `crf_issue` is a Samsung-only `extra` key. Run
  the full SPMW pytest suite as the blast-radius gate.
- Upstream `tests/dataflow/` and `tests/customize/` — **not touched**;
  no edit to `ir/builder.py`, `ir/infer.py`, `dataflow.py`. Nothing in
  the upstream FPGA/AIE path reads `Placement.extra` or the Samsung cost
  model. Rollback story: the lever is gated entirely by the new
  `crf_issue` mode; setting every Samsung candidate's `crf_issue` to
  `"per_workid"` (or deleting the shared variant from the enumerator)
  restores pre-lever-3 behaviour with zero effect on any other backend or
  upstream test. FPGA CI cannot break — it shares no code path with this
  change.

---

## 8. Anti-hardcoding self-audit (maps to the gate)

| Gate clause | How this spec satisfies it |
|---|---|
| No shape literal in decision path | `n_workids` derived via `_samsung_workid_count(target)` (product of unit `mapping`s); `body_cyc`/`trigger_cyc` from `target.op/move`. No `128`, `512`, `4096`, … in source. |
| No test/shape/name sniffing | Enumerator emits both modes unconditionally; cost branches only on `extra["crf_issue"]`; codegen branches only on the same token. |
| No precomputed ranking | `bucket_cost` is computed from spec constants × derived `n_workids`; changes correctly with `CRF_TRIGGER.cycles` and topology (tests 3, 4). |
| No golden stream | Shared CRF body is walked through the real emit path; triggers are generated per work-id bucket. |
| Decision not in codegen | Codegen reads `extra["crf_issue"]` set by argmin; contains no shape/count selection. |
| Layout-algebra / derivation receipt | The replication factor is the unit-tree fanout product (target geometry); the body is the existing layout-derived CRF. Audit evidence #3 (diff provenance): every shared-mode emission points at `extra["crf_issue"]` + the unit-tree mapping. |

---

## 9. Summary of the contract (for the coder, task 051)

- **Enumerator** (`_samsung_enumerate`): for each fast base candidate,
  emit two variants tagged `extra["crf_issue"] ∈ {"shared","per_workid"}`
  (and a readable `mode` suffix). No shape branch. Add `_join_mode` helper.
- **Cost** (`_samsung_kernel_cycles`): add `_samsung_workid_count(target)`
  (unit-tree fanout product); price `shared = body_cyc + trigger_cyc·n_workids`,
  `per_workid = body_cyc·n_workids`, default `per_workid`. New constant
  read only via `target.move("CRF_TRIGGER").cycles`.
- **Codegen** (`_walk_and_emit` / `_run_samsung`): on
  `extra["crf_issue"]=="shared"`, emit one shared CRF body +
  `compiled.host_schedule: list[HostTrigger]` (one per work-id); assert
  `len(buckets)==_samsung_workid_count(target)`. Default per-work-id path
  unchanged.
- **Fixture**: add `CRF_TRIGGER` Move (cycles < body_cyc, cited).
- **Decision in argmin; mechanism in codegen.** No shared-file edit. No
  simulator edit. Rollback = drop the shared variant.
