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
| `spmw_cost_models.py` | Concrete `@cost` factories. Imports `spmw_cost`; registers at import time. | no (side-effect only) |
| `spmw_autoschedule.py` | `Placement` dataclass (per-memref handle assignment), `register_enumerator`, `autoschedule`. Per-backend candidate enumerators register here. After spec 015 the per-group argmin loop invokes the regalloc to refine each candidate before scoring. | yes (`autoschedule`, `Placement`) |
| `spmw_regalloc.py` (spec 015) | `LiveRange`, `Spilled`, `CostVector`, `CapacityTable`, `AllocResult`, `extract_live_ranges`, `allocate`, `_solve` (greedy v1; PBQP seam). Five backend capacity tables and the spill-tier dispatch live here, not on `Target`. | no (internal seam of `autoschedule`) |
| `spmw_codegen.py` | `CodegenContext` base, per-backend ctx subclasses (currently `SamsungCtx`), `compile_for_target`, `Compiled` artifact, `_resolve_layout`, `_walk_and_emit`. After spec 015 `_resolve_layout` unwraps `Spilled` to its home handle and `_walk_and_emit` asks each backend ctx for spill LD/ST move names via `resolve_spill_moves(tier)`. | yes (`compile_for_target`, `Compiled`, `PIMCmd`, `SamsungCtx`) |
| `spmw_linear_layout.py` (spec 013) | `LinearLayout` F2 algebra (apply / compose / product / invert / sublayout / `optimal_swizzle`) and `materialise_handle`. Consumed inside backend enumerators; the resulting concrete handle is what `Placement` actually carries. | yes (`LinearLayout`) |

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
| `allo/spmw_linear_layout.py` (`materialise_handle`, new `bank_stride`) | **edited (spec 022, additive only)** | New `symbolic=` kwarg on `materialise_handle` carries the swizzle `tile` column as a free `SymExpr`; new `bank_stride(target, *, out_dim, unit_level)` derives banks-per-unit from target geometry so the even-bank base `stride*pid` is layout/target-derived, not a pasted `2*pid` (anti-hardcoding gate evidence #4). Kwarg defaults to `None` → today's behaviour; the four non-Samsung enumerators (SPEC-007 option b) never pass it. `_bank_parity` is **unchanged** (the new path emits the same `2*pid` / `2*pid+1` forms it already classifies, for Samsung's `stride==2`). Guarded by new `tests/spmw/test_linear_layout.py::test_materialise_symbolic_fibers` + existing `test_materialise_samsung_bank_handle` (back-compat). Upstream Allo does not import `spmw_linear_layout`, so `tests/dataflow/` and `tests/customize/` are not gated. See `SPEC-022-materialise-symbolic-fibers.md`. |
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

### Enumerator registration (in `spmw_autoschedule.py`)
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

## 4. Open design tensions

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

### T9. UPMEM `_run_upmem` hardcodes `num_dpus=1, num_tasklets=1` (spec 003 §7)
The current CLI invocation pins single-DPU, single-tasklet. Once
`UPMEMCtx` can emit tasklet-strided bodies, these values must
come from the target / `compiled.layout`. Tracked by HANDOFF.md
under the UPMEM blocker family.

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
