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

### `@allo.cost("register_spill")` (spec 015)
Per-backend factory in `spmw_cost_models.py`. Returns
`spill(reg, n_entries=1)` (UPMEM additionally takes `tier=`). The
factory mutates the target by setting `target.move(LD/ST).cycles` from
backend timing constants; the regalloc reads those when building
spill-tier cost-vector entries. Five backends ship: Samsung, AiM,
UPMEM, APU v1, APU v2 (the last is a placeholder because `l1_sim`
declares `perf_is_placeholder = True`).

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
