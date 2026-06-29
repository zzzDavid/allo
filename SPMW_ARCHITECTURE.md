# SPMW Architecture (Tenon extension of Allo)

Living design doc. Each section is updated as architect tasks land.

## 1. Module map

All Tenon code lives under `experiments/allo/allo/spmw_*.py`. No new
top-level packages.

| Module | Owns | Public to user code |
|---|---|---|
| `spmw_target.py` | `@allo.target`, `@allo.unit`, `allo.mem`, `allo.reg`, `allo.move`, `allo.op`, `allo.any_`, `allo.or_`, `allo.get_uid`. Pure-Python data model for the target tree (`Unit`, `Memory`, `Register`, `Move`, `Op`, `AnyOf`, `OrOf`, `Target`). `SymExpr`/`UnitId` for symbolic indexing. | yes |
| `spmw_match.py` | `MatchTrace`, `MatchedOp`, `OperandBinding` dataclasses. Pure data; carries the matcher's output. | yes (for tests) |
| `spmw_match_engine.py` | `compile_op_pattern`, `compile_target_patterns`, `match_workload`. AST-side compilation of target `op.fn` lambdas; MLIR-side unification against affine.store sites. | yes (`match_workload`) |
| `spmw_cost.py` | `@cost` decorator, `get_cost` registry lookup. | yes |
| `spmw_cost_models.py` | Concrete `@cost` factories. Imports `spmw_cost`; registers at import time. **Design 07 (SPEC'D, NOT YET LANDED):** `_whole_program_sum` → `combine`; `_whole_program_combiner` → a `_COMBINER_FOR_FLAVOR` table (`faithful`/`optimistic`/Mortise/demo → sum; `overlap` → max+fill/drain); `_kernel_cycles_factory` concatenates device+host `phases` and folds via the flavor-bound combiner (faithful fold is byte-identical). | no (side-effect only) |
| `spmw_autoschedule.py` | `Placement` dataclass (per-memref handle assignment), `register_enumerator`, `autoschedule`. Per-backend candidate enumerators register here. After spec 015 the per-group argmin loop invokes the regalloc to refine each candidate before scoring. | yes (`autoschedule`, `Placement`) |
| `spmw_regalloc.py` (spec 015) | `LiveRange`, `Spilled`, `CostVector`, `CapacityTable`, `AllocResult`, `extract_live_ranges`, `allocate`, `_solve` (greedy v1; PBQP seam). Five backend capacity tables and the spill-tier dispatch live here, not on `Target`. | no (internal seam of `autoschedule`) |
| `spmw_codegen.py` | `CodegenContext` base, per-backend ctx subclasses (currently `SamsungCtx`), `compile_for_target`, `Compiled` artifact, `_resolve_layout`, `_walk_and_emit`. After spec 015 `_resolve_layout` unwraps `Spilled` to its home handle and `_walk_and_emit` asks each backend ctx for spill LD/ST move names via `resolve_spill_moves(tier)`. | yes (`compile_for_target`, `Compiled`, `PIMCmd`, `SamsungCtx`) |
| `spmw_linear_layout.py` (spec 013) | `LinearLayout` F2 algebra (apply / compose / product / invert / sublayout / `optimal_swizzle`) and `materialise_handle`. Consumed inside backend enumerators; the resulting concrete handle is what `Placement` actually carries. | yes (`LinearLayout`) |
| `spmw_cost_model.py` (design 04; design 07 reshape **SPEC'D, NOT YET LANDED**) | `CostModel`/`OpCost`/`MoveCost`/`OpCostCtx`/`MoveCostCtx`/`ComposeCtx`/`CostResult` dataclasses; `register_cost_model`/`get_cost_model` registry keyed by `(target_name, flavor, concern)`; `evaluate(target, trace, layout, flavor)` (the sim-free entry the virtual runner calls). Mechanism only — no numbers. **Design 07 adds:** `Resource` enum, `Phase`, `phase_cycles`, `combine`, `AccessDescr`, the `Knob` protocol, `Provenance` enum, `CalibrationRecord`; `provenance` on `OpCost`/`MoveCost`; `calibration` on `CostModel`; enriched `OpCostCtx`/`MoveCostCtx` (placement/access/live_set/dtype_bits); `CostResult.phases: list[Phase]` + `phases_as_dict` shim. | yes (`CostModel`, `evaluate`, `Phase`, `AccessDescr`) |
| `spmw_cost_tables.py` (design 04; design 07 reshape **SPEC'D, NOT YET LANDED**) | the concrete `CostModel` instances (numbers + per-target `compose`): `samsung_faithful`, `aim_faithful`, `upmem_faithful`, `apu_v1_faithful`, `apu_v2_placeholder`, plus the swap-test + Mortise + demo flavors. The "Energy-Reference-Table" file — a profiling-refinement edits ONLY this. **Design 07:** every `compose` emits `list[Phase]`; A5 deletes the 3 silent op fallbacks (`per_op=4`/GPR/`add_cyc`); provenance tags + calibration records on every constant; D2 locality `Phase` + per-backend row-buffer constants; D3 APU width lambdas; the A3 one-knob cost demo. | no (side-effect registration) |
| `spmw_tripcount.py` (design 04) **LANDED (task 009)** | `resolve_trip_count` (D3 symbolic bound resolution); retains `_parse_loop_bound` as the tier-1 literal helper. | no (cost-path internal) |
| `tests/spmw/_mortise_target.py` (design 06) **SPEC'D, NOT YET LANDED** | structure-only `@allo.target("mortise")` fixture — a Samsung-like near-bank-SIMD hypothetical substrate (no sim, no HW) whose ONLY structural addition is the `resident_cap_elems` geometry const (the swept capacity lever `C`). Clone of `_demo_target.py`; priced purely via `backend="virtual"`. NO cost numbers on the tree (acid test). | no (test fixture) |

User-facing `allo.work` and `allo.get_wid` are aliases for
`allo.dataflow.kernel` and `allo.dataflow.get_pid` respectively (see
§2). They are reused — not reimplemented — because the SPMW pipeline
reuses Allo's parser front-end (`allo.customize`) to lower workloads
into MLIR, which the matcher then walks.

## 2. Shared-file boundary

| Shared file | Status | Justification |
|---|---|---|
| `allo/dataflow.py` | **untouched** | `kernel` / `region` / `get_pid` / `customize` are reused as the SPMW parser scaffold. SPMW tests never call the FPGA `build()` path. |
| `allo/customize.py` | **untouched** | Same. `allo.customize(top, enable_tensor=False)` lowers to MLIR; matcher consumes that. |
| `allo/memory.py` | **untouched** | Legacy `Layout`/`Memory`/`DTensor` are owned by `allo/ir/*`. SPMW reserves `allo.Layout` *only at the package-namespace level* for the future Linear-Layout symbol; this is done by renaming SPMW's own `Layout` to `Placement` (spec 001). |
| `allo/ir/*` | **untouched** | Upstream Allo's responsibility. |
| `allo/__init__.py` | **edited (spec 001)** | Resolve `Layout` shadow (rename SPMW Layout → Placement). Drop `memory` callable re-export (replace with `mem`). |
| `experiments/simulators/uPIMulator/golang/uPIMulator/` | **edited (spec 003, additive only)** | Add a permanent benchmark slot named `TENON` that `_run_upmem` overwrites with the emitted DPU `task.c` per call. Touched files: `benchmark/CMakeLists.txt` (+1 `add_subdirectory(TENON)` line), `src/assembler/assembler.go` (+1 registry line), plus four new files under `benchmark/TENON/` and `src/assembler/prim/tenon.go`. Templated from the existing `DSLVA` benchmark (which uPIMulator's own authors added as a codegen slot). No existing files are semantically changed; reverting is `git revert` plus an inert benchmark dir. No upstream Allo / FPGA tests touch this tree. See `SPEC-003-upmem-real-kernel.md`. |
| `experiments/simulators/uPIMulator/golang/uPIMulator/src/assembler/prim/gemv.go` (`Init`) | **edited (design 02 §6c, additive only) — coder task 025** | The pre-existing in-tree bespoke `GEMV` host pins `n_size = 64` (`:35`). Change to `n_size = DataPrepParams()[1]` with a `len(...) >= 2`-guarded fallback to `64`, so the gemv reduction length is shape-derived (1024) not hardcoded. `DataPrepParams()` is already a comma-separated `[]int` (`command_line_parser.go:99-111`); this reads a second optional element. The native PrIM `--benchmark GEMV` invocation (one param) is byte-for-byte unchanged by the `len` guard. Used so the UPMEM gemv MATCH runs BOTH Tenon and Exo columns through one bespoke gemv overhead path (not the VA-shaped TENON slot), making the comparison byte-fair and retiring the `data_prep_params=1024` debt. Guarded by a native 1-param GEMV back-compat run + new `tests/spmw/test_upmem_gemv_host_routing`. No CMake/assembler-registry edit (`GEMV` already registered, `assembler.go:41`). See `design/02-upmem-tasklet-tiling.md` §6c. |
| `allo/spmw_autoschedule.py` (`_samsung_enumerate`, `_bank_stride_per_pim`) | **edited (spec 022 realization)** | The even-bank base is bound as `stride*pid` where `stride = bank_out_size // pim_unit_count` is geometry-derived by `_bank_stride_per_pim(target, layout)` (bank out-axis size off the swizzled layout, pim fanout off the `pim` unit's `mapping`), retiring the pasted `2*pid` (anti-hardcoding gate evidence #4). For Samsung `16 // 8 == 2`, so the emitted `2*pid` / `2*pid+1` forms are byte-identical and `_bank_parity` is **unchanged**. **No `bank_stride()` helper and no `symbolic=` kwarg were added to `spmw_linear_layout.py`** — `materialise_handle`'s existing `symbol_table=` carries the bound `stride*pid`, and `LinearLayout.size_of` already supplies the geometry; the earlier SPEC-022 plan for a `bank_stride()` helper / `symbolic=` kwarg / `test_materialise_symbolic_fibers` was never realized and is not in the tree. Guarded by `tests/spmw/test_autoschedule_samsung_dual_fiber.py::test_enumerator_emits_dual_fiber_even_then_odd` (EVEN/ODD receipt) + `tests/spmw/test_linear_layout.py` (materialise back-compat). Upstream Allo does not import `spmw_autoschedule`, so `tests/dataflow/` and `tests/customize/` are not gated. |
| `experiments/simulators/PIMSimulator/src/pim_driver.cc` (GEMV phase block) | **edited (spec 026, additive only)** | New `--batch B` + `--native-rebaseline` flags wrap the GEMV phase block (`:292-321`) in a batched sequencing loop: resident clocks `preloadGemv` once + B*(exec+readback); native re-pays `preloadGemv` per call → B*(preload+exec+readback). Both loops call the SAME `executeGemvFaithful`+`readResult` per vector (I2, no flattering); only preload placement differs. With `--batch 1` / no flag the branch runs the byte-identical pre-026 single-vector sequence (FPGA/AIE CI never reaches this Samsung-only path). The five frozen fns + `executeGemvFaithful`/`countGemvStreamWork` + `>=1` clamp have a **zero diff**. Rebuild `scons -j32`. Guarded by upstream gtest `PIMKernelFixture.gemv` (frozen-fn bodies untouched) + new `tests/spmw/test_samsung_batched_*`. See `SPEC-026-batched-gemv-weight-reuse.md`. |
| `experiments/simulators/PIMSimulator/src/PIMKernel.cpp` (`runPIM`, `executeGemv`, `executeGemvWithCmds`, `computeGemv`, `programCrf`) | **reference-only, FORBIDDEN to edit (spec 021)** | `getCycle()` is shape-fixed: cycles advance one tick per `mem_->update()` in `runPIM` (line 21-27); the `--cmds` stream reaches only `programCrf` (line 478), which caps at 4 CRF bursts (line 220-238); the work loop and per-tile MAC volume are derived from `w_data->bShape`, not from `cmds`. The faithful run path (spec 021, task 025) must be a **new** driver symbol/flag that calls these but does not edit their bodies, plus Tenon-side `spmw_codegen._run_samsung*` changes. The native `--op GEMV` cycle output must stay byte-for-byte (task 015 baseline). Guarded by upstream gtest `PIMKernelFixture.gemv` (must stay green) + `tests/spmw/test_samsung_placement_changes_cycles.py` (promoted to strict `cycles_a != cycles_b` once 025 lands). See `SPEC-021-samsung-cycle-model.md`. |

Every shared-file edit must list the upstream tests that guard it
here. So far (spec 001): no shared-file `.py` edits beyond
`__init__.py` re-exports, which are guarded by:
- `tests/test_vhls.py`, `tests/test_vitis.py`, `tests/test_xls.py`,
  `tests/test_catapult_hls.py`, `tests/test_pynq.py` for
  `allo.LLVMModule` / `allo.HLSModule` (kept in place).
- `tests/test_memory.py` for legacy `allo.Layout` (`Shard/Replicate`)
  — kept in place by the spec.

## 3. Seams

Named extension points. Each backend (Samsung, AiM, UPMEM, APU v1,
APU v2) plugs in here.

### `CodegenContext` (in `spmw_codegen.py`)
Base class. Each backend subclasses to translate Tenon handles into
its own ISA encoding. Minimum contract every subclass must implement:

```
class XxxCtx(CodegenContext):
    def cmd(self, opcode, *, dst=None, src0=None, src1=None, **kw) -> None
    def append(self, instr) -> None     # low-level escape
```

`cmd(...)` should accept the **Tenon handles** (`Register`,
`MemoryRef`) directly as `dst`/`src0`/`src1` and encode them to the
backend's native form, appending one (or more) backend-native
instructions to `self.cmds`.

### Cost-model registration (in `spmw_cost.py`)
```
@cost("kernel_cycles")
def _factory(target):
    def cost_fn(trace, placement) -> int: ...
    return cost_fn
```
Cost names are global; the factory receives `target` so it can
dispatch on `target.name`. Concrete models live in
`spmw_cost_models.py`.

### `Phase` timeline + resource-aware combiner (in `spmw_cost_model.py`, design 07) — SPEC'D, NOT YET LANDED
The cost-abstraction keystone. `CostResult.phases` becomes a
`list[Phase(resource: Resource, latency, ii, count, tag)]` (was a
free-form dict). A phase's cycles = `latency + ii*(count-1)`;
`phase_cycles(phase)` is the one module-level definition the combiner
and every `knob.cost(...)` share. The whole-program combiner
`combine(phases, *, overlap)` replaces `_whole_program_sum`: it sums
within a `Resource` class and, per the flavor-bound fold rule, either
sums across resources (`overlap=False`, `faithful`) or takes the max
across independent resources + fill/drain (`overlap=True`, `overlap`
flavor). `faithful = (latency=ii=per_op, sum)` reproduces every current
`compose` scalar BYTE-FOR-BYTE (the regression anchor: 15251 / 114992 /
457336, B\*=2, 3.93×). Backend-specific physics (UPMEM revolver, APU
vr_dma) stays INSIDE `compose`, which emits already-resolved `Phase`s;
the combiner is substrate-agnostic. The old dict shape survives ONLY as
a read-only derived `phases_as_dict(phases)` shim for `RunResult.extra`.
`Resource = {COMPUTE, DMA, HOST, LOCALITY}` (closed enum, A5). See
design 07 §A1.

### `AccessDescr` — layout-derived access pattern (in `spmw_cost_model.py`, design 07) — SPEC'D, NOT YET LANDED
**CO-OWNED with `autosched-placement-realization` (its D2 allocator
base-access cost consumes the SAME type).** A frozen dataclass derived
from the chosen `LinearLayout` / `Placement` handle: `tier`
(register/near_bank/scratchpad/dram), `stride`, `bank_dims`, `n_banks`,
`conflict_free`, `conflict_count`, `row_hits`. `conflict_free=True ⇒
conflict_count=0` (constructor invariant). `AccessDescr.identity()` is
the back-compat zero-penalty default (stride 1, conflict-free) so any
compose with no layout to derive from gets a 0 locality penalty —
"conflict-free → 0" by construction. `conflict_count` is supplied by an
additive `LinearLayout.conflict_count(bank_dims, varying_inputs)` method
(D2/task 006 adds it; mirrors `describes_conflict_free`'s enumeration but
counts collisions instead of early-returning). See design 07 §A2.

### `Knob.cost(value, ctx) -> list[Phase]` seam (in `spmw_cost_model.py`, design 07) — SPEC'D, NOT YET LANDED
**CO-OWNED with `autosched-placement-realization` (its D4 owns
`candidates()` + the `emit()` codegen materializer; THIS task owns ONLY
the `cost(...)` side).** Each schedule lever's cost contribution is a
`list[Phase]` on the SAME A1 timeline the full objective folds, so
`compose = base op/move phases + Σ knob.cost(chosen)`. A search evaluates
a marginal knob delta by recomputing one knob's `cost(...)` and
re-folding — proven EXACT (a knob-delta re-score == a full recompute
byte-for-byte; the non-tautology test). Lever migration (the six
existing `layout.extra` levers) is placement-task D4; this task wires the
seam + proves it on one knob (`stage_resident`). See design 07 §A3.

### Provenance + calibration + opt-in confidence-gate (in `spmw_cost_model.py` + `spmw_autoschedule.py`, design 07) — SPEC'D, NOT YET LANDED
`Provenance = {MEASURED, DATASHEET, ASSUMPTION}` (default `ASSUMPTION` —
an untagged constant is the least-trusted, so the gate widens, never
silently narrows). `OpCost`/`MoveCost` carry a `provenance` field;
`CostModel` carries a `CalibrationRecord(validated_against,
residual_error, shape_coverage)`. The **opt-in `confidence_gate`** is the
SOLE permitted `spmw_autoschedule.py` touch: a new defaulted-off param +
a post-argmin `_check_confidence(...)` that, when ON, refuses to silently
commit a `placeholder`/`coarse`-confidence ranking (default policy:
warn-and-commit; refuse-and-error is opt-in). The argmin scoring core
(`spmw_autoschedule.py:749-766`) is UNTOUCHED. Also closes the silent
`dynamic_trip → 1` fall-through (the tier-3 coarse flag becomes visible
to the committer). The provenance→band aggregation (the report-28 §A4.3
symbolic-band sub-claim) is flagged T25, not built; the v1 gate reads the
existing `CostResult.confidence`. See design 07 §A4/§A5.

### `HostXcel` collective interface (in `spmw_host.py`, design 05) — SPEC'D, NOT YET LANDED
```
@allo.unit(mode="host")                 # promoted host root, explicit
def host():
    dram = allo.mem(name="host_dram", bytes=...)
    @allo.host_xcel                      # class decorator -> instance on host unit
    class hx(allo.HostXcel):
        @allo.primitive(cost=None)       # method -> returns emit closure
        def broadcast(self, buf, *, over):
            return lambda ctx, t: ctx.host_broadcast(t)
        @allo.primitive
        def scatter(self, buf, *, over): ...
        @allo.primitive
        def gather(self, buf, *, over): ...
        # reduce omitted -> host-CPU sum (opt-in) after gather
```
Basis = `broadcast`/`scatter`/`gather`/`reduce`; derived
`all_reduce`/`all_gather`/`reduce_scatter` have default compositions in
the basis (overridable for a native fast path). `HostXcel.covers(name)`
is the coverage check — requesting an uncovered collective is a hard
compile error; host-CPU fallback is opt-in only. Workload writes
host movement as `allo.broadcast`/`scatter`/`gather`(buf, over=axis,
residency=...); `residency ∈ {resident,per_call,readback}` (modifier,
not a separate concept). `over=` names a unit-tree axis (one binding
rule, design 05 §Q2): its fan degree is `prod(mapping)` of that level;
for a pow2 *device* axis the same-named F2 layout out-dim carries the
placement, for a non-pow2 *host* fan-out (UPMEM 2560 DPUs) it is a plain
`int` and **never** routed through `LinearLayout`.

**Boundary rule (design 05 §Q3):** explicit collectives own host↔device
staging; the implicit matcher keeps intra-device loads/stores. A cross-
boundary move MUST go through `HostXcel`, never a bare `allo.move`;
self-moves (`src is dst`) are forbidden after this cycle.

**Naming-collision guard (design 05 §4):** `allo.gather`/`allo.scatter`
(host collectives, from `spmw_host`) do NOT collide with
`df.gather`/`df.scatter` (AIE pipe primitives in `dataflow.py`, used as
`df.*` in `tests/dataflow/aie/test_collective_communication.py`). Do NOT
import `gather`/`scatter` into `__init__.py` from `dataflow`.

### `host_staging` cost concern (in `spmw_cost_tables.py`, design 05 §5) — SPEC'D, NOT YET LANDED
Registered as a NEW concern on the landed `(target_name, flavor,
concern)` CostModel registry (`tenon@7738fee`) — NOT a fresh extraction.
`register_cost_model(CostModel(..., concern="host_staging", ...))`;
looked up via `get_cost_model(name, flavor, concern="host_staging")`.
Re-homes the report-18 calibration constants (369/1/2/4096/181) verbatim
off the deleted `PRELOAD_*`/`READBACK_*` self-moves. Whole-program
estimate = `kernel_cycles + host_staging` (sum). Returns per-phase
`CostResult.phases` + a pluggable whole-program combiner so async
overlap (sum→max) is a one-function swap later (not implemented this
cycle). B=1 sum must EMERGE as 15251; crossover B\*=2; asymptote 3.93×.

### Mortise capacity lever (in `spmw_cost_tables.py` + `_mortise_target.py`, design 06) — SPEC'D, NOT YET LANDED
The hypothetical-substrate capacity what-if. The lever is the
on-device weight-resident capacity `C`, carried as the **geometry** const
`resident_cap_elems` on the `mortise` target tree (NOT a cost number —
acid test holds). It lives ONLY in the Mortise `host_staging` compose as
`B*(1-phi)*P_var`, `phi = min(1, C/(M*K))`: resident pays preload `P`
once, the evicted shortfall is re-streamed per vector. Three flavors on
`(target_name="mortise")`: `faithful` (the §2.4 capacity-collapse
finding), `unlimited` (the ABLATION — forces `phi≡1`, erases all `C`
dependence, collapses to report-18 `P+B(E+R)`), `optimistic` (the §4
sensitivity band). Swap via `cost_flavor=` — design-04 surface verbatim,
zero device-tree edit. Sweep seam = `build_mortise_target(resident_cap_elems=C)`
constructor kwarg + the trace's `batch_dim`, driven by
`experiments/scripts/R26_mortise_capacity_sweep.py`. Anchor: at
`C≥T_w`/resident the Mortise faithful whole-program is NUMERICALLY
IDENTICAL to Samsung faithful (`P+B(E+R)`, report-18), riding the live
`test_samsung_rank_preservation_weight_residency`. Design 06 §2–§5.

### Enumerator registration (in `spmw_autoschedule.py`)
Mortise (design 06): a new `@register_enumerator("mortise")` emits the
`stage_resident ∈ {False, True}` candidate pair (additive clone of the
Samsung tail, lines 358–359) so the resident schedule is argmin-SELECTED,
not hand-set.
```
@register_enumerator("samsung_hbm_pim")
def _enumerate(target, matches: list[MatchedOp]) -> list[Placement]: ...
```
Per-backend. Receives one autoschedule group's worth of matches
(currently bucketed by `@allo.work` kernel — i.e. by `match.func_name`)
and returns candidate placements for that group. `autoschedule` calls
the registered cost callback on each candidate and returns argmin per
group, producing a `list[Placement]` aligned with the buckets.

Note (spec 012a): the enumerator signature changed from
`(target, trace)` to `(target, matches)` so the multi-layer MLP
workload works (two `@allo.work` kernels each bind `x` to a different
weight memref). The old `(target, trace)` shape would have collapsed
both layers' role→memref maps into one and raised
`NotImplementedError`.

### `LinearLayout.optimal_swizzle` (in `spmw_linear_layout.py`, spec 013)
Constructs the conflict-free F2 layout for a backend given its
`(vec_dims, bank_dims, segment_dims)` triple. Called by every backend
enumerator. The result is materialised into a concrete `Register` /
`MemoryRef` handle by `materialise_handle` before being stored in
`Placement.placements` — codegen sees no API change. The Samsung
GEMV case recovers `bank_bit_0 ⊕ tile_parity` (report 03 §5.4)
algebraically, replacing the hand-listed `_samsung_enumerate`
candidates.

### `_resolve_layout(match, placement)` (in `spmw_codegen.py`)
Turns a `Placement` + a `MatchedOp` into a dict of role-name →
handle that the user's `emit=` callback expects. Backend-agnostic;
not a seam — but every backend's `emit` lambdas must accept
`(role_args..., ctx)` in the order the `op.fn` lambda declares.

### `CodegenContext.resolve_moves(role, src_handle, dst_handle)` (spec 009)
Per-backend hook returning `(load_move_name, store_move_name)` for one
operand role given its chosen placement handle. The walker emits
preloads at the first match of each work-id and storebacks at the
last. Default implementation raises `NotImplementedError`; each
ctx subclass (Samsung / AiM / UPMEM / APU v1 / APU v2) supplies its
own table — see spec 009 §F.

### `CodegenContext.after_match(match, n_emitted)` (spec 009)
Per-match post-emit hook. Defaults to no-op. `SamsungCtx` overrides
it to emit the inner-K `JUMP` record; previously this lived in
`_walk_and_emit` directly and leaked Samsung-specific `PIMCmd`
records into other backends' `ctx.cmds` (which are `list[str]`).

### `Placement.extra["grf_residency"]` — GRF host-vs-CRF preload (spec 024, Samsung)
Per-memref residency map `memref_name -> "host" | "crf"`; absent key
== `"crf"` (back-compat, all five backends default). Set only by the
Samsung enumerator for the broadcastable preload role (`x`/`grf_a`).
`"host"` = the GRF is filled by the native HAB broadcast (host
transaction, off the CRF stream); `"crf"` = a per-work-id `LD_A`/`LD_B`
CRF MOV. Cost model (`_samsung_kernel_cycles`) prices `"host"` at 0 CRF
cycles for the preload and `"crf"` at `target.move(load_name).cycles`
per work-id; codegen (`SamsungCtx`/`_schedule_moves`) omits the CRF MOV
for `"host"` (side-list → native host load) and emits it for `"crf"`.
The choice is argmin's, not codegen's. Riding `extra` (not `mode`, not a
new dataclass field) keeps residency per-operand and orthogonal to
lever-1 layout modes, with zero blast radius to non-Samsung backends.
See SPEC-024.

### `Placement.extra["weight_resident"]` + `MatchedOp.extra["batch_dim"]` — batched GEMV weight reuse (spec 026, Samsung)
Two additive carriers for the batched-GEMV fair beat (arch-200 (a) FEASIBLE,
research-201 B\*=2 / 3.93x asymptote):
- `MatchedOp.extra["batch_dim"]` (int, default 1): the operand-shape leading
  dim B of `Y[B,M]=X[B,K]@W^T`. Stamped by the matcher via a new spmw-local
  `batch_dim(match)` resolver that identifies the batch loop **structurally**
  (its iter var is the leading index of both `x` and the result, and is not a
  reduction axis) — never positional (`enclosing_loops[0]`), never a literal.
  No `allo/ir/` edit (SPEC-026 §1.1 proves the existing `enclosing_loops` +
  operand `indices` carry it). Cost (`_trace_batch_dim`) and codegen read this
  key; default-absent → B=1.
- `Placement.extra["weight_resident"]` (bool, default False): emitted as a
  final 2x tail cross in `_samsung_enumerate` (`_with_weight_residency`), both
  values unconditionally — argmin earns the choice. `_samsung_kernel_cycles`
  composes `weight_resident ? P + B*(E+R) : B*(P+E+R)` with separable
  closed-form `preload_cyc=(M*K/write_width)*write_cyc+programCrf_cyc` /
  `exec_cyc` (today's formula, unchanged) / `readback_cyc=ceil(M/read_width)*
  read_cyc`, all from target-spec constants + (M,K). P and R are
  placement-invariant so the existing single-shape argmin winner is unchanged;
  at B=1 both branches = P+E+R (T15 parity floor preserved). Codegen
  (`_run_samsung*`) reads the flag and selects the driver invocation:
  `--batch B` (preload once + B*(exec+readback)) vs `--native-rebaseline`
  (B*(preload+exec+readback)), same `executeGemvFaithful`+`readResult` per
  vector both sides — only preload-loop placement differs. Other backends
  never set either key. See SPEC-026.

### `Placement.extra["n_tasklets"]` — UPMEM tasklet tiling (design 02, UPMEM)
The UPMEM analog of the Samsung `weight_resident` lever — the one target where
the abstraction is meant to move the number (TASK_DESCRIPTION line 104-107). The
DPU's 11-cycle revolver thread scheduler (the throughput bound; the 14-stage
pipeline is NOT the bottleneck — research-021) is filled by fine-grained tasklets
(Exo `nt=16` beats Cinnamon `nt=1` by ~5.71x on our path, Phase-0 README §2).
Today nothing in the decision path knows tasklet count: `_upmem_enumerate`
enumerates only acc WRAM-vs-GPR, `_upmem_kernel_cycles` has no tasklet term, and
`_run_upmem` hardcodes `--num_tasklets 1` (tension T9). Design 02 makes the
lever a ranked decision:
- `MatchedOp` side: a new spmw-local `_trace_reduction_trip(matches)` names the
  per-DPU reduction trip (the innermost `enclosing_loops[-1]` bound of the
  `accumulates=True` match — the same bound SPEC-019's `pending_k_bound` already
  threads). **No `allo/ir/` edit** (SPEC-026 §1.4 proof reused).
- `Placement.extra["n_tasklets"]` (int, default 1): enumerated as a cross over
  the existing acc-placements, candidate set `{1, T_max}` (≥2, SPEC-009) where
  `T_max` = the tasklet-unit `mapping` fanout read from the target tree (not the
  literal 16). Default 1 == today (T9 floor preserved).
- Cost (CLOSED by research-021): `_upmem_kernel_cycles` becomes
  `S + ceil(S*(R-1) / min(T, R))` where `S` = today's issue count
  (`sum(per_op*iters)`) and `R = target.revolver_latency`. The saturating
  divisor is the **revolver scheduling window `R = 11`, NOT the 14-stage
  pipeline depth** — the lever comes from the round-robin thread scheduler
  (`thread_scheduler.go`): a tasklet re-issues at most once per 11 cycles, so
  speedup saturates at `min(T, R)`. `fill_drain = 0` (no separate term). At T=1
  this returns `S*R`, a uniform scale of today's `S`, so the acc argmin is
  invariant. Exactly **one** new target-spec constant `revolver_latency = 11`
  (uPIMulator `num_revolver_scheduling_cycles`, `src/main.go:115`, provenance-
  parallel to `LD_MRAM cycles=1000`); `pipeline_depth` is deliberately NOT added.
  Model reproduces the 5.71x lever within 0.9% (5.762), saturates at `T = 11`,
  invariant across reduction trips (no literal keying).
- Codegen: `UPMEMCtx` reads `extra["n_tasklets"]` (a field, no re-derivation);
  `_run_upmem` threads it + the reduction trip to the CLI, **resolving T9**.
The +1.12x harness offset (generic TENON slot vs Shiran's bespoke `EXO_GEMV`
host) is independent: the lever is a ratio, offset-invariant; retiring the
offset is a separate harness task under T8a/T8b. See `design/02-upmem-tasklet-tiling.md`.

### `Compiled.ctx` (spec 003, additive)
`Compiled.__init__` grows an optional fifth kwarg `ctx=None` that
retains the `CodegenContext` instance that produced the cmds.
`compile_for_target` passes it through. Other backends ignore the
field; UPMEM uses it so `_run_upmem` can call
`ctx.get_kernel_src()` to render the full DPU envelope rather than
duplicating the envelope logic in the runner. The field is private
(`_ctx`); only intra-`spmw_codegen.py` code reads it. Fully
back-compat for every positional caller.

### `Compiled.run(layers=[...])` for multi-MAC streams (spec 020, Samsung)
`pim_driver` accepts one GEMV per invocation; the Samsung emitted
`compiled.cmds` for a multi-layer workload (e.g. MLP) concatenates
MAC blocks across layers. `_run_samsung` therefore splits
`compiled.cmds` at JUMP boundaries (one JUMP closes one MAC's inner-K
fold), and when more than one MAC group is present accepts a new
opt-in kwarg `layers=[{"W": ..., "x": ...}, ...]`. Each MAC group
runs through one `pim_driver` invocation with its own per-layer
`W.npy` / `x.npy` / `cmds.txt`; cycles are summed. Single-layer
callers (`W=, x=` form) hit the back-compat branch unchanged. See
`SPEC-020-samsung-subshape.md`.

### `CodegenContext.get_kernel_src()` (existing; spec 003 extends UPMEM's)
Per-backend hook returning the assembled source to feed to that
backend's tool flow. APU v1 returns a flat C-call sequence (low
mode). UPMEM (post-003) returns a complete DPU `task.c` with the
PrIM-standard envelope (`__host dpu_arguments_t`, `BARRIER_INIT`,
`kernels[]` dispatch, MRAM↔WRAM staging) wrapping the body in
`self.cmds`. The envelope template is fixed per-ctx for now;
multi-shape selection (`"va"` / `"gemv"` / `"mlp"`) is deferred.

### `CodegenContext.resolve_spill_moves(tier)` (spec 015)
Per-backend hook returning `(load_move_name, store_move_name)` for one
declared spill tier (e.g. `"bank_row"` on Samsung/AiM, `"mram"` on
UPMEM, `"l1"` on APU v1, `"l2"` on APU v2). Default raises
`NotImplementedError`; each ctx subclass supplies its own one-line
table. The move scheduler reads `placement._spilled` (the regalloc's
audit list of which live ranges spilled) and prepends the LD / appends
the ST around the work-id bucket that hosts the spilled value.

### `allocate(target, matches, candidate_placement)` (spec 015)
Pure-function entry point from `spmw_regalloc.py`. Takes one
autoschedule group's matches plus one enumerator-produced candidate;
returns an `AllocResult` carrying a refined `Placement` (with
`Spilled` wrappers where applicable), the total spill cost added,
and the list of live ranges that spilled. `autoschedule()` calls
this once per (group, candidate) and argmin-s on
`kernel_cycles + allocator.total_cost`. The greedy solver follows
Chow-Hennessy 1990 (sort by cost-gap descending, pick cheapest fit);
PBQP lives at `_solve()` as a deferred upgrade slot.

### `Placement.mode` and `Placement.extra` (spec 009)
Additive fields on the `Placement` dataclass
(`spmw_autoschedule.py:Placement`). `mode: str = ""` is a free-form
label the cost model and codegen consult to disambiguate candidates
whose `placements` dict is identical or whose op-expansion differs
(e.g. APU v1 `"sv"` vs `"sv_lookup"` MAC expansion). `extra: dict =
{}` is a per-backend scratch slot. Defaults preserve current behaviour
for cost models that don't read them. Allowed `mode` values are
backend-declared (asserted in cost factory): APU v1 `{"sv",
"sv_lookup"}`; Samsung `{"bank_row", "grf_staged"}` (label-only, cost
reads `placements`); APU v2 `{"l1_row_canonical",
"l1_row_reversed"}` (label-only; APU v2 is functional-only, see SPEC-011). The
regalloc round-trips both fields untouched onto the refined
`Placement`. Codegen consumers: APU v1 `MAC` dispatch branches on
`mode` to pick between `emit_mac_lookup` and `emit_mac_mul_add`. See
`SPEC-009-argmin-enumeration.md`.

Samsung `extra["crf_issue"]` (spec 025, lever 3): `{"shared",
"per_workid"}`, a CRF-issue mode **orthogonal** to the `y`-placement
`mode`, threaded through `extra` (not a second cross-producted `mode`
string) so it stays additive against levers 1/2. Cost model
(`_samsung_kernel_cycles`) prices `shared = body_cyc +
trigger_cyc·n_workids` vs `per_workid = body_cyc·n_workids`, where
`n_workids = _samsung_workid_count(target)` (product of unit-tree
`mapping` fanouts — geometry, never the literal 128). Codegen
materialises `shared` as one shared CRF body + a host trigger schedule
(`Compiled.host_schedule`). Default (absent key) = `per_workid` =
byte-for-byte current behaviour; AiM/UPMEM/APU never read it. See
`SPEC-025-lever3-shared-crf.md`.

### `Compiled.host_schedule: list[HostTrigger]` (spec 025, Samsung, additive)
New field on `Compiled`, parallel to `cmds`. For the Samsung shared-CRF
mode, the shared CRF body goes into `cmds` (uploaded once by
`programCrf`) and the per-work-id host triggers go into `host_schedule`
(one `HostTrigger(work_id, tile_count)` per work-id). The SPEC-021
faithful run path derives `stream_records` from `len(cmds)` (one body)
and the issued-transaction multiplicity from `len(host_schedule)`,
pricing native and Tenon under the same accounting (SPEC-021 §2). Empty
for every other backend and for the `per_workid` Samsung mode. Triggers
are deliberately NOT `PIMCmd`s so they bypass the C++ `validationCheck`
+ the ≤4-burst `programCrf` cap (SPEC-021 §1). See
`SPEC-025-lever3-shared-crf.md` §5.4.

### `@allo.cost("register_spill")` (spec 015)
Per-backend factory in `spmw_cost_models.py`. Returns
`spill(reg, n_entries=1)` (UPMEM additionally takes `tier=`). The
factory mutates the target by setting `target.move(LD/ST).cycles` from
backend timing constants; the regalloc reads those when building
spill-tier cost-vector entries. Five backends ship: Samsung, AiM,
UPMEM, APU v1, APU v2 (functional-only target; cost stub is
non-comparative — see SPEC-011).

### `CostModel` — decoupled per-op cost spec (design 04, virtual backend) — SPEC'D, NOT YET LANDED
Two-layer, per-op cost interface in new `spmw_cost_model.py` (mechanism)
+ `spmw_cost_tables.py` (numbers). Layer A: `dict[str, OpCost]` /
`dict[str, MoveCost]` where each entry is a **callable** `fn(ctx) ->
cycles` (independently refinable against device profiling — the MICRO-2025
APU v1 per-op model is the canonical intended consumer of `OpCost.fn`).
Layer B: a swappable `compose(ComposeCtx) -> CostResult(cycles, phases,
confidence)`. The target tree carries **zero cost numbers** after design
04; the 34 `cycles=` fixture literals + `revolver_latency` move into the
`CostModel`. Bound by `(target_name, flavor)`; `_kernel_cycles_factory`
delegates to the bound model's `compose` (faithful flavor → argmin
byte-identical to today, report-18 invariant preserved). The host-side
task's `@allo.cost("host_staging")` is a **sibling concern in the same
machinery** (design 04 §4: this task owns the reformulation, host-side
consumes it). See `experiments/allo/design/04-virtual-accelerator-backend.md`.

### `virtual` backend (design 04 §2) — SPEC'D, NOT YET LANDED
Selector: `compile_for_target(target, trace, backend="virtual",
cost_flavor="faithful")` (NOT `run(virtual=True)` — symmetry: virtual is a
peer of the five sim/HW backends). `_BACKEND_RUN["virtual"] =
_run_virtual`, a sim-free adapter to `spmw_cost_model.evaluate` (no
subprocess/Docker/sim-root import — the no-sim gate passes by
construction). `RunResult` is structurally unchanged; `confidence` +
`phases` ride in `extra`; functional `output` is None for v1.

### `resolve_trip_count` (design 04 §3, new `spmw_tripcount.py`) — LANDED (task 011)
Replaces the silent-`None`→`1` behaviour of `_parse_loop_bound` *on the
cost path only* (matcher bound strings untouched). Signature
`resolve_trip_count(match, loop_idx=-1, *, shapes, mapping_env)`;
`resolve_bound_text` is the string-level core for non-inner bounds. Three
tiers: literal / affine-over-operand-shapes+mapping-params / dynamic
(None). On tier-3 each `compose` applies the model's DECLARED
`CostModel.dynamic_trip_default(op_name)` (v1 = 1), stamps a
`phases["dynamic_assumed"]=1` marker, and downgrades the result to
`confidence="coarse"` — never silently wrong, *visibly* coarse. The
corpus (gemv/FFN/batched on Samsung/UPMEM/AiM) is fully tier-1/2, so the
faithful argmin stays calibrated and byte-identical. Cost-path-internal —
no shared-Allo edit.

## 4. Open design tensions

### T25. Provenance → uncertainty-band aggregation (design 07 §A4, report 28 §A4.3)
Design 07 lands the provenance TAGS (`MEASURED|DATASHEET|ASSUMPTION`) and a
confidence-gate that reads the existing `CostResult.confidence`
(coarse/placeholder). The report-28 §A4.3 NEW sub-claim — the uncertainty
band *derived symbolically from which tags participated in a candidate's
cost* (the analytical-model analogue of BO/UCB's `sigma(x)`, but
provenance-derived not sample-learned) — is FLAGGED, not built. Revisit
when a calibration task wants the `optimistic`/`pessimistic` flavors to be
provenance-derived bands rather than hand-typed ×0.5 multipliers. The v1
gate's mechanism (read confidence, refuse-low) lands now; the
aggregation is the open part.

### T26. UPMEM MRAM phase split for the overlap fold (design 07 §A1.5/§D1.5)
A1 keeps UPMEM's MRAM cost (`LD_MRAM=1000`) folded INSIDE the per-op cost
(one `Resource.COMPUTE` phase), so the faithful number is frozen
byte-for-byte. The D1 UPMEM overlap arm needs MRAM as a separate
`Resource.DMA` phase so the stream can hide behind tasklet compute; that
split lives ONLY in the overlap path (guarded by the `overlap` flavor),
never in faithful. Open: whether the split should be PROMOTED into faithful
— ruled NO until a measured UPMEM overlap workload ships (promoting it
would change the frozen faithful number). See design 07 §D1.5.

### T21. Virtual cost composition: v1 sequential vs v2 sum→max overlap (design 04 §6) — RESOLVED-IN-SHAPE (design 07)
v1 `compose` summed phases (no compute↔DMA overlap). **Resolved in shape
by design 07:** overlap is now a property of the `Phase` timeline + the
`combine(phases, overlap)` fold, not a parallel flavor. `faithful` binds
`overlap=False` (sum); the `overlap` flavor binds `overlap=True`
(max-across independent resources + fill/drain). APU v1 L4→VR DMA hides
behind compute; Samsung stays genuinely serial (its overlap fold = its
sum fold, no concurrent resources). T21's old "trigger to revisit" (a
corpus workload where the over-count flips a rank decision) is now the
D1 argmin-flip DEMO, constructed deliberately (design 07 §D1.4). **Stays
open as a CALIBRATION question** (the overlap fold's absolute error /
T21 over-count of partial overlap + contention — design 07 §D1.3), not a
shape question. See design 07 §A1/§D1.

### T22. `host_staging` concern fit in the `CostModel` interface (design 04 §4.2 → design 05) — RULED (design 05)
The host-side cycle adds `host_staging` as a NEW concern on the landed
`(target_name, flavor, concern)` CostModel registry. **Ruled in design 05
§5:** the per-op `OpCost`/`MoveCost` context is sufficient — a collective's
cost is `(buf.numel // fan_width) * per_unit_cyc` over the work's
`StageRequest` list, derived from the layout transform (design 05 §Q2);
the existing `MoveCostCtx.elem_count`/`extra` carry what a collective needs.
No collective-specific cost context required. The per-phase
`CostResult.phases` + pluggable whole-program combiner keeps async overlap
(sum→max, T18) a later one-function swap. See design 05 and the new
`HostXcel`/`host_staging` Seams subsections (§3).

### T23. Mortise eviction granularity: smooth-`phi` vs `ceil(T_w/C)`-quantized (design 06 §7, report 26 §7 Q2)
The Mortise capacity compose uses the smooth `(1-phi)*P_var` eviction
term; real eviction is tile-quantized (`ceil(T_w/C)` tiles). The R26
harness SHOULD report both curves so the finding statement picks the
conservative one. Refines the *number* at fractional `C`, not the
qualitative collapse. Commit a quantized compose only when a corpus shape
flips a crossover-B decision under coarse `C`.

### T24. Mortise headline ratio is Samsung-preload/exec-ratio specific (design 06 §7, report 26 §7 Q1/§4)
The 3.93× ceiling = `P/(E+R)` is Samsung-analog-specific; a Mortise with a
different MAC speed moves the *number* (2×-faster exec → ~2.6× ceiling),
not the ordering. The `optimistic` flavor's `E`/`P_var` band quantifies
it; calibrating Mortise's own `tCCDL` against a different analog datasheet
(AiM 2 GHz) is a v2 nicety, out of scope.

### T18. Async staging/compute overlap (design 05 §9, report 23 open-Q4) — SEAM REALIZED (design 07)
The whole-program composition was `device + host_staging` as a **sum**.
**Design 07 realizes the reserved seam:** `host_staging` is now a set of
`Resource.HOST` phases on the A1 timeline, and the `overlap` combiner can
fold them. T18 **stays open** because the v1 overlap fold deliberately
keeps HOST serial vs device (the host must stage the weight before the
device strobes — design 05 §9); a measured host-bandwidth overlap is the
refinement that would let HOST hide behind device. Mortise's
`evict_per_call` phase (design 06) is already a separate `Phase` the
overlap combiner can soften. Revisit when a pipelined batched-GEMV
workload with measurable host↔device overlap ships. See design 07 §D1.2.

### T19b. Host-staging cost units: device-cycle-equiv vs host wall-time (design 05 §9, report 23 open-Q2)
`host_staging` keeps the report-18 device-cycle-equivalent convention
(369/181 constants) so argmin stays single-currency. Real host bandwidth
(PIM-MMU: 63.7% of end-to-end) is a later table-only refinement (design
04's table split is exactly for this). Open until a host-bandwidth
measurement on each backend's real host path lands.

### T1. `allo.Layout` reserved name (resolved by spec 013)
SPMW's `Placement` was originally called `Layout`. The legacy
`allo.memory.Layout` (Shard/Replicate) stays put as `allo.Layout` for
upstream-test compatibility. The new linear-layout symbol is
`allo.LinearLayout`, exported from `spmw_linear_layout.py` (spec 013
§B). Option (b) chosen — namespaced — to avoid disturbing upstream's
`DTensor` use of `Shard/Replicate`.

### T2. `allo.work` vs. a Tenon-native workload decorator
Today `allo.work = allo.dataflow.kernel`. This works because
`dataflow.kernel` is a thin annotation decorator that only registers
the mapping in a region context — no FPGA-specific side effects at
decoration time. SPMW reuses Allo's parser front-end (`customize`)
to lower the workload into MLIR.

Open question (for the workload-spec task that follows Task 008
(MLP)): should `allo.work` become a thin wrapper module in
`spmw_workload.py` so we own the symbol, or keep the alias? Reusing
the alias is fine until we need workload-spec features that
`dataflow.kernel` does not provide (e.g. multiple work-grids in one
function). Defer until a concrete need surfaces.

### T3. Memory geometry shape (spec 001 §D1)
The current `memory(banks, rows, cols, width)` four-arg shape is
Samsung-specific. Spec 001 generalizes to keyword `**geometry`. The
open question is whether `Memory` should canonicalize geometry into
a fixed schema across backends (so cost models can be cross-backend)
or keep geometry purely backend-private (each cost model knows its
own target's keys). Currently leaning toward "private, named-by-key"
because the five PIM backends are genuinely heterogeneous in their
memory hierarchy. Revisit if a cross-backend cost model emerges.

### T4. Cross-work-id move elision (spec 009 §J)
The move scheduler in spec 009 emits preloads/storebacks per work-id.
For workloads where the same operand is identical across all work-ids
(GEMV's `local_x` on Samsung — every PIM unit reads the same vector),
this re-preloads N times instead of once. A global liveness analysis
would lift the LD out of the per-work-id window. Spec 015 keeps live
ranges *within* an autoschedule group (one `@allo.work` kernel) — the
cross-work-id and cross-`@allo.work` cases (see T6) are the same
liveness pass, deferred together.

### T5. Per-bank handle collapse in `Target._handles` (spec 009 §G)
`Target._handles` is a flat `dict[str, handle]`. Where a memory
declares a name like `bias` on a `@unit(mapping=[8])` (AiM banks),
all eight `Register` objects collide in the dict and only the last
survives. Move scheduling for `WR_BIAS` works *coincidentally* today
because dedup-by-move-name folds the eight per-bank moves into one
`WR_BIAS` emit, and the cmd's `bank=` field is resolved by the
autoscheduler. A proper fix needs `_handles` to be keyed by
unit-path (`channel/bg/bank/bias`) and lookup to be either path-
aware or context-aware (current unit during emit). Open task 017.

### T6. Cross-`@allo.work` reuse (spec 015 §14)
The regalloc keeps live ranges group-local — same shape as spec 012a's
group-local placements. MLP layer1's output that becomes layer2's
input is therefore placed twice. A cross-group liveness pass would
coalesce these. Tied to T4 (cross-work-id move elision); the same
machinery (a global liveness analysis over the trace's matches) drives
both. Fold them into one pass when either lands.

### T7. Greedy regalloc vs. PBQP (spec 015 §4, §9)
Greedy v1 is the submission allocator. The PBQP slot lives at
`_solve()` in `spmw_regalloc.py`. Trigger to revisit: a workload with
cyclic register pressure where greedy spills and PBQP would coalesce
by splitting live ranges. GEMV / MLP on all five backends do not
exhibit this. The `CostVector` / `CapacityTable` / `LiveRange` data
structures are designed to feed either solver unchanged.

### T8a. UPMEM kernel envelope is fixed, not workload-shape-aware (spec 003 §6)
The DPU `task.c` envelope `UPMEMCtx.get_kernel_src()` emits is a
single template templated off `DSLVA` (binary in-place reduce
shape). GEMV / three-buffer kernels will need a different
envelope. Resolution path: `UPMEMCtx` declares a shape tag
(`"va"` / `"gemv"` / `"mlp"`); `get_kernel_src` picks one of N
templates. Triggered when the first non-VA-shape UPMEM workload
lands. The `benchmark/TENON/` shell itself stays the same — only
the rendered C and the matching `prim/tenon.go` data-prep
parameterise.

### T8b. UPMEM `tenon.go` data-prep is VA-shape-hardcoded (spec 003 §7)
Counterpart to T8a on the Go side. Today `tenon.go` is a copy of
`dslva.go` and ships VA-style host arguments. This is fine for
cycle-counting (cycles come from the linker's real asm) but wrong
for end-to-end numeric correctness. Resolution path: parameterise
`tenon.go` via a small JSON dropped next to `task.c` describing
buffer count, sizes, and types. Out of scope until a
correctness-checking task lands.

### T9. UPMEM `_run_upmem` hardcodes `num_dpus=1, num_tasklets=1` (spec 003 §7) — RULED (design 02), FULLY SPECCED
The current CLI invocation pins single-DPU, single-tasklet
(`spmw_codegen.py:2421-2422`). This pins Tenon's UPMEM run at Cinnamon's
`nt=1` operating point — the worst value of the very lever (tasklet tiling)
that is the entire source of the UPMEM win (Exo `nt=16` = 5.71x, Phase-0
README §2). **Ruled (design 02):** `num_tasklets` becomes
`Placement.extra["n_tasklets"]` — a shape-derived, enumerated, cost-priced
decision (Option B, mirror of Samsung SPEC-026 `weight_resident`). Codegen reads
the field; it never re-derives the count. **COST coefficient CLOSED
(research-021):** `kernel_cycles = S + ceil(S*(R-1)/min(T,R))` with `R =
target.revolver_latency = 11` (revolver scheduling window from
`thread_scheduler.go` / `src/main.go:115` `num_revolver_scheduling_cycles`, NOT
the 14-stage pipeline depth), `fill_drain = 0`; one new spec constant
`revolver_latency`, no `pipeline_depth`; reproduces 5.71x within 0.9%, saturates
at `T = 11`. All four coder sub-tasks (MATCH / ENUMERATOR / COST / runner)
implemented by coder 014. **`--data_prep_params` resolution (design 02 §6b/§6c,
arch 024):** coder 014 found that setting it to the *reduction trip* INVERTS the
lever (`data_prep_params` is the per-DPU *input-buffer size*, not an inner-K
knob), so it was held at the prior literal pending architect ruling. **Ruled
(§6c, research-22):** the gemv cell routes through the in-tree bespoke `GEMV` host
with a shape-derived `(m_size, n_size)` footprint (`--data_prep_params
"<m_size>,<n_size>"`), which retires the `1024` literal gate-clean AND makes the
Tenon-vs-Exo comparison byte-fair (both columns share one bespoke gemv overhead
path; the ~1.13x TENON-slot offset is common-mode and cancels). One Go-side fix
(`gemv.go` `n_size` from `DataPrepParams()[1]`) + the `_run_upmem` route, specced
for coder task 025. Default `n_tasklets=1` → `S*R` (uniform scale, parity),
preserving the acc argmin and FPGA CI (rollback story). See
`design/02-upmem-tasklet-tiling.md` §3, §6, §6c,
`work/reports/021-research-upmem-pipeline-amortization.md`, and
`experiments/reports/22-upmem-gemv-host-fairness.md`.

### T8. `LinearLayout` is enumerator-ephemeral, not `Placement`-carried (spec 013 §E)
`LinearLayout` is built inside each backend enumerator, materialised
into a concrete `Register` / `MemoryRef`, and then discarded. The
`Placement.placements` dict still maps memref-name → handle (no
LinearLayout field) so codegen's `_resolve_layout` and
`_bank_parity` are unchanged. This is deliberate for spec 013: the
F2 algebra solves the *autoschedule-side* problem (constructing the
optimal SymExpr) while `_bank_parity` continues solving the
*codegen-side* problem (classifying the SymExpr to a PIMOpdType).
Promoting `LinearLayout` into `Placement` (and deleting
`_bank_parity`) is a future spec, triggered when a target needs more
than two bank-parity classes — for which the existing SymExpr
pattern-matcher is too narrow.

### T11. APU v1 lookup-table identity is build-harness-implicit (spec 018)
`APUv1Ctx.emit_mac_lookup` emits a fixed `mac_lut_ptr, 256` literal;
`spmw_apu_v1_build.py` always emits a `uint16_t mac_lut[256]` field in
`program_data` and a `popcount(u8)` initialisation in host.c. The LUT
identity (popcount-of-byte) is **implicit in the build harness** — it
is not carried on `Placement`, `MatchedOp`, or `compiled.dtype`. This
is fine today because (i) the only `mode="sv_lookup"` consumer is
binary MAC, (ii) the popcount table is constant, and (iii) the LUT
fits inline in the cmd struct (512 B). The seam to cut when a second
LUT identity ships (e.g. s16 MAC, ~256 KB LUT that no longer fits
inline) is `Placement.extra["lut_kind"]` plus a build-harness
dispatcher that routes large LUTs through a dedicated
`mem_hndl_mac_lut` L4 role. Trigger to revisit: any workload whose
matcher emits `MAC` with `dtype != u8` and selects `sv_lookup`. See
SPEC-018 §6 (slots SPEC-018b for byte-pair packing and SPEC-018c for
variable-dtype LUTs).

### T12. Samsung cycle model is shape-fixed, not stream-driven (spec 021) — RESOLVED with a build action
Determined definitively (verdict (a)): PIMSimulator `getCycle()` is a
function of `--output-dim`/`--input-dim` only; the emitted PIMCmd stream
reaches just the 32-entry CRF via `programCrf` (capped at 4 bursts) and
never changes the transaction queue `runPIM` counts. **Until the
faithful run path lands, metric (A) `tenon_cycles < native_cycles` is
unmovable** by any enumerator/cost/codegen change — this is the DAG root
of the Samsung-GEMV-peak task. Chosen resolution: **Option B** — a new
driver symbol keeps `computeGemv` address generation (numerics/readback
unchanged) but makes outer-loop trip counts and per-tile transaction
multiplicity a function of the emitted stream (`stream_records`,
`stream_macs_per_tile` derived from `compiled.cmds`), then counts cycles
off the unmodified `runPIM`. Honest/flattering line and file boundary in
`SPEC-021`. The remaining open sub-question (exact PIMCmd-stream ->
`stream_macs_per_tile` map for JUMP-folded vs unrolled, and
MOV-vs-host-broadcast accounting) is deferred to lever specs 030/040/050
because it *is* levers 1/2; task 025 ships a conservative "one issued
transaction per CRF instruction after JUMP expansion" rule that already
moves all three levers in the right direction. The MOV-vs-host-broadcast
half is now RESOLVED by SPEC-024: host-resident GRF preloads are never
emitted as CRF MOV records, so they are absent from `compiled.cmds` and
thus from `stream_records` automatically — the faithful path needs no
new logic; the residency attribute changes the stream and the path
already prices the stream. See
`SPEC-021-samsung-cycle-model.md` and
`dev/06072026-samsung-gemv-peak/work/reports/arch-cycle-model-determination.md`.

### T13. Samsung CRF-issue mode: shared CRF vs per-work-id replication (spec 025) — RESOLVED
Lever 3. Tenon's per-work-id walk (`_walk_and_emit` →
`_bucket_by_work_id`) re-emits the CRF body once per PIM block (~128
work-ids × ~4-insn body ≈ 512 records); native programs **one** shared
CRF and issues a host trigger schedule. Resolved as an additive
`Placement.extra["crf_issue"] ∈ {"shared","per_workid"}` mode the cost
model prices (`shared = body_cyc + trigger_cyc·n_workids` vs
`per_workid = body_cyc·n_workids`) so argmin prefers shared; codegen
materialises one shared CRF body + `Compiled.host_schedule`. The
replication factor `n_workids` is `_samsung_workid_count(target)` — the
product of unit-tree `mapping` fanouts (target geometry), never the
forbidden literal 128. **Zero shared-file footprint**: the change is
entirely inside `spmw_autoschedule.py` / `spmw_cost_models.py` /
`spmw_codegen.py` plus a new `Compiled.host_schedule` field and one new
`CRF_TRIGGER` Move on the Samsung fixture; `ir/*`, `dataflow.py`,
`customize.py`, and the PIMSimulator source are untouched. The cycle win
is only observable through the SPEC-021 faithful run path (the lever
feeds it a smaller `stream_records`). Rollback = drop the shared variant
from the enumerator. See `SPEC-025-lever3-shared-crf.md`.

### T14. Samsung dual-fiber (even/odd) bank placement (spec 023) — RESOLVED
Lever 1. `_samsung_enumerate` built the correct swizzle
(`tile_parity ⊕ bank_bit_0`, `bases["tile"]=[(0,1)]`) but then collapsed
it with `fixed={"tile":0}`, so every MAC hit `EVEN_BANK` and the odd bank
half idled (~2× loss). Resolved as a third additive `Placement`,
`mode="dual_fiber"`, that materialises BOTH fibers
(`materialise_handle(fixed={"tile": v})` for `v in range(tile_axis_size)`
→ `2*pid` and `2*pid+1` over symbolic `pid`; the `+0`/`+1` fall out of
the swizzle, not a pasted pair — that IS the layout-algebra receipt,
anti-hardcoding evidence #4). Cost model splits the folded-MAC count
`ceil(folded/n_fibers)` from `extra["n_fibers"]` (default 1, so the two
existing candidates are unchanged) — output changes correctly with K and
responds to perturbing `target.move("JUMP").cycles`. Codegen emits the
alternating `(MAC EVEN, JUMP n_even, MAC ODD, JUMP n_odd)` with split
trips derived from `inner_ub//target.grf_a.lanes` partitioned by
`n_fibers` (the literal `63` *emerges*; never written), and replaces the
stale `_SAMSUNG_LANE_BURST=8` codegen literal with `target.grf_a.lanes`.
**Decision in argmin, mechanism in codegen** — codegen branches only on
`placement.mode`, never on shape. Cross-reference SPEC-022 for the
strictly-symbolic-`tile` materialise form; lever 1 only needs `tile`
bound per fiber, which the current scalar-multiplier path already
supports. The cycle win is observable only through the SPEC-021 faithful
run path (task 025); the enumerator/cost/codegen changes are valid
regardless. **Zero shared-file footprint** (`ir/*`, `dataflow.py`,
`customize.py`, PIMSimulator untouched). Rollback = drop the `dual_fiber`
append from the enumerator + the `n_fibers` read in the cost model; the
codegen branch then goes dead. See `SPEC-023-lever1-dual-fiber.md`.

### T15. Samsung GEMV is at the hardware floor — strict-beat gate retired (arch 110) — RESOLVED (CASE 2)
After levers 1+2+3 Tenon faithful *ties* native faithful at 15,251 /
114,992 / 457,336 (4096×1024, 8192×4096, 16384×8192). Task 110 asked
whether the faithful model (`countGemvStreamWork` / `executeGemvFaithful`,
SPEC-021) is too coarse and flattening a real sub-native Tenon efficiency
(CASE 1) or whether native is provably at the floor (CASE 2). **Ruled
CASE 2.** Verified independently from PIMSimulator source: the body count
`ceil(M/4096)*ceil(K/8)` is read from `w_data->bShape`
(`PIMKernel.cpp:585-586`), never from `cmds`; the per-body MAC strobe is
exactly the K columns each output's dot product requires
(`computeGemv:686-688`); the `>= 1` clamp on `even_macs`/`odd_macs`
(`PIMKernel.cpp:576-579`) is a **true floor** (one MAC body per present
tile), not a modelling coarseness — a stream below it computes a wrong dot
product. All four candidate efficiencies dispatched: C1 wider-GRF_A
forbidden by `num_grfA_=8` register-file width (`PIMKernel.h:47`) and
invariant under re-tiling (total = K columns); C2 shared-CRF and C3
host-preload reach **parity** (native's folded CRF `MAC/JUMP/MAC/JUMP/NOP`,
`PIMCmdGen.h:118-126`, has zero MOV/FILL, so `host_loads=0` already);
C4 no double-count exists. The tie is the *correct* answer — the model is
not floored at native (pre-lever redundant Tenon = 362,689 cycles), so it
would reward a real efficiency if one existed; none does. **Decision:
retire the strict performance gate (A) `tenon < native` in favor of
parity-or-beat `tenon <= native` for Samsung GEMV.** The SPEC-021 §6
`cycles_a != cycles_b` *model-faithfulness* test stays valid and passing
(it discriminates redundant vs optimized streams) — it is never promoted
against native. Paper claim: "Tenon's autoscheduler reaches the vendor
hand-tuned floor from a high-level `@allo.work` description" (23.8× over
its own naive baseline). **No source change**; the five SPEC-021 §3
reference functions AND the `>=1` clamp at `PIMKernel.cpp:576-579` are
frozen — lowering the clamp below 1 would be the flattering move (skipped
tile = wrong numerics). DAG: task 150 (document floor) fires as the CASE-2
terminal; tasks 120/130/140 (sub-floor lever + re-verify) cancelled as
un-winnable by construction. **Open for next iteration (out of scope):**
the one fair beat is *batched GEMV / weight reuse* — preload is 75% of the
4096×1024 total and native re-runs it per call; a Tenon schedule hoisting
preload out of a B-vector batch loop amortizes weight writes across B
vectors. Different workload, not a new GEMV lever; flagged, not pursued.
See `dev/06072026-samsung-gemv-peak/work/reports/arch-110-granularity-ruling.md`,
`experiments/reports/17-samsung-gemv-subfloor-probe.md`, SPEC-021 §6.

### T16. Batched-GEMV weight reuse is the fair beat — FEASIBLE (arch 200, iteration 3 DAG root)
The T15 escape hatch is taken up here. **Ruled (a) FEASIBLE.** Verified
from PIMSimulator source that W stays RESIDENT in banks across B input
vectors and preload is paid once, not per vector:
- `preloadGemv` (`PIMKernel.cpp:283-323`) is the **sole, separable**
  weight-write phase; it is the only GEMV function issuing bank writes
  (`addTransaction(true, addr, &operand->bData[d_idx])`, line 313). The
  faithful driver clocks it **alone** (`pim_driver.cc:298-300`,
  `cyc_preload`); at 4096×1024 it is 11,368 of 15,251 cyc (75%) — one-time
  weight streaming.
- The compute phase `executeGemvFaithful` → `computeGemv`
  (`PIMKernel.cpp:582-688`) **never re-writes W**; it uploads the input
  vector to GRF_A (`WRIO_TO_GRF_`) and MACs it against the **resident**
  `EVEN_BANK`/`ODD_BANK` weight (folded CRF `PIMCmdGen.h:118-126`:
  `MAC GRF_B, GRF_A, {EVEN,ODD}_BANK`). The bank operand is read in place.
- The simulator **already** loops over input vectors with zero reload:
  `for (b=0; b<num_batch; b++)` (`PIMKernel.cpp:587,619`),
  `num_batch = i_data->bShape[0]`, one `computeGemv` per (tile, b), no
  `preloadGemv` inside the loop. Residency is in-model, not inferred.

**Native batched comparator (fair):** B *independent* GEMV calls, each
re-paying preload — `native = B*(preload+exec+readback)`. **Tenon batched:**
preload once, resident — `tenon = preload + B*(exec+readback)`. **Fair beat
= `(B-1)*preload_cost > 0` for B≥2**, large at the 75% preload fraction. B
is an **operand-shape dim** (X[B,K] / Y[B,M] leading dim, = the simulator's
`i_data->bShape[0]`), NOT a literal — gate-B clean. At B=1, costs tie (T15
floor preserved, no regression). Both candidates (preload-once vs
preload-per-call) must be enumerated so argmin earns the choice (SPEC-009
≥2-candidate discipline).

**Frozen surface unchanged:** the five SPEC-021 §3 reference functions,
`executeGemvFaithful` / `countGemvStreamWork`, and the `>=1` clamp stay
frozen (T15). The batched beat lives in **new driver-level sequencing**
(`pim_driver.cc` batch mode: preload-once + B passes for Tenon;
preload-per-call B-loop for native), priced by the **same** faithful
accounting both sides — Tenon wins only by *skipping redundant preloads* on
calls 2..B, never by deleting needed work or re-pricing native. Tenon-side
codegen change is Samsung-local (`spmw_codegen.py` `_run_samsung` /
`_run_samsung_one`; the lever-2 host-residency `host_preloads` machinery is
the "preload once" seam). **Default ruling on shared files:** the batch dim
must be carried additively by the existing match path; `allo/ir/builder.py`
/ `infer.py` edits are FORBIDDEN unless task 202 proves the match path
cannot carry a leading batch dim additively (and then names the gating
`tests/dataflow/` + `tests/customize/` tests).

**Open quantitative question → research-201:** whether `preload_cost` is
closed-form from target-spec constants + (M,K) or must be faithfully
measured. The comparator structure and the `(B-1)*preload_cost` beat hold
under either. DAG: this ruling (200) unblocks 201 (research), 202 (enum/
cost/codegen SPEC), 203–208 (coders + verifiers), all already queued.
See `dev/06072026-samsung-gemv-peak/work/reports/arch-200-batched-feasibility.md`,
`.claude/agent-memory/architect/2026-06-07-batched-gemv-feasibility.md`.

**Resolved into a build contract by SPEC-026 (task 202).** research-201
closed the open question: `preload_cost` is **closed-form** from target-spec
constants + (M,K) (`(M*K/write_width)*write_cyc + programCrf_cyc`), B\*=2,
asymptote 3.93x. SPEC-026 pins the four coder tasks:
- **203 MATCH:** **NO `allo/ir/` edit** (proven: existing `enclosing_loops` +
  operand `indices` carry B). B stamped on `MatchedOp.extra["batch_dim"]` by a
  new spmw-local structural `batch_dim(match)` resolver. Fallback (if ever
  needed) escalates to architect, never an ad-hoc shared edit (SPEC-026 §1.4).
- **204 ENUMERATOR:** `Placement.extra["weight_resident"]` 2x tail cross
  (`_with_weight_residency`), both candidates unconditional, no shape branch.
- **205 COST:** `_samsung_kernel_cycles` split into closed-form P/E/R + B from
  `_trace_batch_dim`, composed `weight_resident ? P+B*(E+R) : B*(P+E+R)`; B=1
  parity; P/R placement-invariant so existing argmin winner unchanged.
- **206 CODEGEN:** `pim_driver.cc` `--batch`/`--native-rebaseline`; same
  faithful instrument both sides, only preload-loop placement differs.
- Gate A (207): `tenon_total <= native_total`, strict for B≥2; gate B (208):
  no shape/batch literal in any decision path.
New open tensions T16-a (resident call shape: single internal-b-loop call vs
explicit driver B-loop) and T16-b (P/R placement-invariance assumption) carried
in SPEC-026 §6. See `experiments/allo/allo/SPEC-026-batched-gemv-weight-reuse.md`
and `.claude/agent-memory/architect/2026-06-07-batched-gemv-enum-cost-codegen.md`.

### T17. Cross-target fixed-function yardstick — Samsung redefined to faithful native, AiM verbatim (arch 008, cycle 06172026) — RESOLVED
The beat-cinnamon-exo cross-target frame (TASKS.md, 8 in-scope cells)
raised: which cycle model is the Samsung yardstick, after coder-003 found
the published Cinnamon 222,919 (gemv 4096x1024) / 242,064 (FFN) is **not
reproducible** under Phase-0 scope (the SamsungHBM lowering + `cinm-opt`
are unbuilt and excluded).

**Ruled (arch 008, Option 2): the Samsung yardstick is the faithful
PIMSimulator native folded-GEMV cycle model, applied identically to BOTH
the baseline and Tenon columns.** gemv 4096x1024 baseline = **15,251 cyc**
(the SPMW `pim_driver --cmds --faithful` path, canonical; coder-003's
standalone driver gives 15,156, a 0.6% harness delta, kept as corroborating
evidence). NOT 222,919.

Why not reproduce 222,919: the README publishes `1024^2 == 4096x1024 ==
222,919` (identical for different shapes) — the fingerprint of a Cinnamon
lowering that pads/serialises to a fixed quantum and does NOT model the
64-channel x 8-block bank parallelism our `executeGemv` exploits
(`num_total_pim_blocks_=512`). Anchoring to it would measure Tenon against
Cinnamon's *detuned tiling*, not the hardware floor — violating the
anti-flattering discipline (report 17 §4 / T15) on the baseline side.
Building the toolchain (Option 1) reproduces a non-faithful artifact;
BLOCKED (Option 3) forfeits a cell we can measure honestly.

AiM needs no redefinition: its Cinnamon path collapses to an emittable ISR
`.trace` ramulator2 times, so coder-004 reproduced 83,775 (gemv) / 12,402
(FFN) **verbatim** — the yardstick is the published number, faithful.

Per-cell verdicts (arch 008 §Part 2):
- **Samsung gemv** — MATCH at floor (single shape, T15) + FAIR BEAT
  (batched, T16/SPEC-026, B\*=2, 3.93x). Mechanism: enum `weight_resident`
  + cost P+B(E+R).
- **Samsung FFN 256-1024-256** — MATCH at floor (two GEMV legs, W1!=W2 so
  no inter-leg weight reuse; host ReLU = 0 PIM cyc). The escalated
  exploratory sub-question — inter-leg ACTIVATION residency (keep
  h=ReLU(W1 x) PIM-resident, skip host round-trip) — is **RESOLVED
  2026-06-17 (research-019, arch-020): MATCH, no fair-beat avenue.** Host
  round-trip is 0 PIM cyc (not priced); the only skippable term (leg-1
  readback R1) is ~45 cyc / ~0.15% of FFN total (sub-1% materiality);
  leg-2 GRF_A input staging is mandatory either way; on-device ReLU
  (exists, ISCA21 §IV-C / `computeRelu`) is net-negative. NO
  `activation_resident` flag/cost/enum candidate. The honest
  abstraction-moves-the-number avenue for FFN, if exercised, is
  **batched-FFN weight residency** (each leg = batched GEMV inheriting
  T16/SPEC-026), not activation residency. See
  `dev/06172026-beat-cinnamon-exo-cross-target/work/reports/019-research-samsung-ffn-activation-residency.md`.
  **BLOCKED-SIM on the MEASURED baseline (arch 028, design 03,
  2026-06-17):** the FFN's analytical verdict (MATCH at floor) stands,
  but ruling 008 §1.4 demanded a *measured* two-leg faithful number and
  that number is unobtainable. The reference `pim_driver` **cores
  (SIGSEGV in DRAMSim `Bank::write` during weight `preloadGemv`+`runPIM`,
  pim_driver.cc:332)** at both leg shapes (W1 M=1024/K=256, W2
  M=256/K=1024). Root: `preloadGemv` (PIMKernel.cpp:283-323) address
  layout is validated only at the M=4096 design point (`output_tile_size
  = num_grfB_*num_total_pim_blocks_ = 8*512 = 4096`, `NUM_COLS=128`); for
  other M the `(row,col)` sweep aliases into an unopened bank. Boundary
  fully mapped (design 03): M in {32,64,4096} run at K=1024; M in
  {128,256,512,1024,2048} core at K=1024; M=1024 cores at every K. No
  faithful runnable decomposition exists: M-tiling/M-padding prices any
  M<=4096 as the full 4096-row quantum (the T15 `>=1` clamp makes M=64
  K=1024 return 15251, IDENTICAL to M=4096 K=1024 — the same
  fixed-quantum non-faithfulness fingerprint 008 §1.2 rejected for
  Cinnamon 222,919), and K-tiling cannot reach W1 (M=1024 cores
  independent of K). **Ruled (c): ESCALATE.** Cell is non-closable under
  Phase-0 scope without a user descope (recommended) or an out-of-scope,
  user-authorized reference-sim `preloadGemv`/DRAMSim fix (FORBIDDEN by
  T12; not authorized here). The five frozen fns + M=4096 baseline 15251
  stay byte-for-byte regardless. `samsung-ffn` MANIFEST row stays
  BLOCKED-SIM; no number fabricated. See
  `experiments/allo/design/03-samsung-ffn-runnable-path.md`.
- **AiM gemv / FFN** — MATCH at floor (documented arithmetic floor;
  SPEC-019 opsize=K MAC_ABK; argmin ABK/SBK). Batched extension to AiM NOT
  ruled this cycle (would need a separate ramulator2 weight-phase source
  read); single-shape MATCH stands.

Shared-file ruling: zero `allo/ir/` edits for all four cells. Samsung
batched carries `batch_dim`/`weight_resident` additively (SPEC-026); AiM in
`spmw_codegen.py` + fixture (SPEC-019). Five frozen Samsung fns + `>=1`
clamp stay frozen. No simulator rebuild authorized by this ruling. The
yardstick redefinition is a *measurement convention* (no code) — the
baseline column is generated through the same `pim_driver --faithful` path
the Tenon column uses (I2 fairness). Rollback: trivial (no shared-file or
FPGA-path edit). See
`dev/06172026-beat-cinnamon-exo-cross-target/work/reports/008-arch-samsung-aim-fixedfn-ruling.md`
and `.claude/agent-memory/architect/2026-06-17-cross-target-samsung-yardstick.md`.

### T17. APU v1 inter-VR/intra-VR DMA tiling is enumerator-ephemeral and cost-invisible (design 01) — RESOLVED-IN-PRINCIPLE
Phase-0 board profile (`experiments/baselines/cinnamon-exo/apu-v1/
ffn-cinm-opt.flo.log`) splits FFN 64-256-64 crun as **retile 132,890
(seu:0, pure move) + dma 103,922 = 59% data movement**, GVML compute
~24%. GEMV 4096×1024 is the opposite — compute-bound, reproduced at
−0.17%, at the floor. The APU v1 decision path today prices **only**
compute: `_apu_v1_enumerate` emits 2 candidates differing only by MAC
op-expansion (`sv` vs `sv_lookup`, identical `placements`), and
`_apu_v1_kernel_cycles` never reads `target.move(...)`. So inter-VR vs
intra-VR — the axis that dominates FFN — is **outside Tenon**: it is a
host command-line layout arg (`intra`), and the wrong choice is the
documented `FAIL 62/64` host/device-mismatch gotcha. **Ruling (design
01):** GEMV gets **no** mechanism change (do not add VR machinery that
cannot beat a floor; re-prove −0.17% as a regression gate). FFN's
data-movement bottleneck is resolved via budgets (2)+(3): the
enumerator surfaces inter-VR vs intra-VR as **two real, materialisable
`Placement` candidates** keyed on `extra["vr_dma"]` with tile count
**computed from operand shape + `target.vrs`** (not `256`/`64`); the
cost model adds a move term `n_moves * target.move(...).cycles` so
argmin organically prefers the movement-minimising layout; codegen
**materialises** the chosen `extra["vr_dma"]` (emitting one consistent
host+device layout, which dissolves the FAIL-62/64 gotcha) and may, in
a *follow-up* iteration only, double-buffer/async the retile (Option B,
pure codegen materialisation of an already-chosen placement). All edits
additive and confined to the `spmw_*` set + APU v1 target data; **no
`allo/ir/builder.py` / `infer.py` / `dataflow.py` edit**. Rollback:
absent `vr_dma` → move term 0 → today's intra-VR/SV-lookup ranking
reproduced; FPGA CI untouched. Guarding tests on the eventual coder
spec (task 015): `tests/pim/test_autoselect_bmatmul_layout.py`,
`tests/pim/test_codegen_layout_dispatch.py`,
`tests/pim/test_bmatmul_sv_lookup_low_mode.py`, full `tests/spmw/`.
Deferred, non-blocking: SV/SV-lookup cost-bucket realism (the
15,650×-vs-19.8× note) is a separate cost-realism cleanup, not part of
the FFN movement win. See `experiments/allo/design/01-apu-v1-vr-loop-strategy.md`.

### T19. UPMEM tasklet axis as scheduling lever, not LinearLayout out-dim (design 02)
(Numbered T19 — T17/T18 slugs were already taken by arch-008 Samsung yardstick
and design-01 APU v1; this doc has pre-existing duplicate T17 headers.)
The tasklet count is modelled as a parallelism / scheduling lever
(`Placement.extra["n_tasklets"]`, the T9 resolution), NOT as a LinearLayout
out-dim. Same category reasoning as the existing UPMEM acc-placement note
(`spmw_autoschedule.py:442-448`): the tasklet count fills revolver slots, it
is not a memory coordinate, so the layout algebra has no out-dim for it.
Revisit ONLY if a future workload needs genuine *per-tasklet data partitioning*
(then the tasklet axis is a real layout shard and would join `optimal_swizzle`).
Until then, keeping it off the layout algebra is deliberate. See
`design/02-upmem-tasklet-tiling.md` §2 Option C.

### T20. UPMEM FFN cells BLOCKED-ON-HARNESS — selector-host port OUT OF SCOPE (design 02) — RULED
The UPMEM Exo (`ffn_1pd_8`) and Cinnamon (`CINM_FFN`) FFN cells are
multi-phase MRAM-selector kernels needing a bespoke per-phase host that does
not exist in our tree. **Ruled (design 02 §4):** porting those selector hosts
is OUT OF SCOPE for this cycle — it is uPIMulator harness engineering, not a
change to any of the three mechanism budgets (layout / enumerator / cost), so
it cannot be the source of an *earned* abstraction win. It is additionally
blocked behind T8a/T8b (the UPMEM envelope + `tenon.go` data-prep are still
VA-shape single-template, which a deterministic phase selector requires
parameterized). The cells stay **BLOCKED-ON-HARNESS with reason** through the
exit gate (already recorded in `MANIFEST.tsv` for `upmem-ffn-exo` /
`upmem-ffn64-cinm`); this is distinct from the SDK↔VM block that stops Shiran
(we cleared that — PrIM VA byte-exact). Never fabricate an FFN number. The
cycle's UPMEM thesis (the tasklet lever) is carried by gemv. Future path: when
T8a/T8b land a shape-parameterized envelope, a follow-up harness task can port
the hosts and unblock these cells. See `design/02-upmem-tasklet-tiling.md` §4.
