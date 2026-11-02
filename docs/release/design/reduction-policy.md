# Reduction-order policy

Status: implemented for release issues C-09 (core), S-03 (Samsung ordered
program) and V1-03 (APU v1 plan selection). Module:
`allo/spmw_reduction_policy.py`.

## 1. Problem

The paper states that "combining partial sums in a different order requires
explicit permission" (Sec. III-D), that "requiring the original
floating-point accumulation order can rule out splitting a reduction before
search" (Sec. III-E), and reports Samsung's "ordered fallback" and the
Gemini-I run matching all 4,096 outputs "when each multiply and addition
rounds to FP16 in source order" (Sec. V-B, VI). Before this change the
package had no such permission. Samsung `dual_fiber` and the REDUCE program's
lane partial sums, UPMEM `nc`/`kc` and tasklet partials, and the APU v1
subgroup-tree plans all split reductions unconditionally. The source-order
Samsung program existed only on the simulator side (`TEMPORAL_REDUCE`) and
could not be reached from `allo.compile`.

## 2. Semantics

`allo.compile(workload, target, cost, ..., reduction_order=...)`:

| value | meaning |
|---|---|
| `"strict"` (default; also `None`) | Every accumulation keeps the source order and the source rounding. The accumulator is updated once per reduction iteration, in ascending iteration order, and every multiply and add rounds to the accumulator's element type. A schedule that forms partial sums of a floating-point reduction and combines them is illegal. |
| `"reassociate"` | Partial sums may be formed, grouped and combined in any order, and in a wider type. |

**Integer accumulators.** For `i*`, `ui*`, `si*` and `index` types,
addition modulo 2^width is associative and commutative. Every grouping
therefore gives the same wrapped result, so `"strict"` still permits
splitting an integer reduction. The check reads the MLIR element type of the
loop-carried accumulator operand (`accumulator_element_type`). If that type
cannot be read, the accumulator is treated as floating point (fail closed).
The claim assumes wrapping adds, which is what every backend with integer
reductions uses (UPMEM int32/int64, APU v1 uint16).

**Why strict is the default.** The paper requires permission for
reassociation, so silence must mean "no permission". Existing tests and the
paper's speed rows were produced with regrouping, so they now pass
`reduction_order="reassociate"` explicitly (Section 6). The paper's Samsung
comparison text says the same thing ("this comparison permits regrouping
partial sums").

**Where it is recorded.**

* Every accumulating `MatchedOp` carries `extra["reduction_policy"] =
  ReductionPolicy(order, accumulator_type)`. `match_workload(target, module,
  reduction_order=None)` stamps it, and `allo.compile` passes the user's
  value. Non-accumulating matches carry no stamp. A hand-built match without
  a stamp is read under the default (`reduction_policy(match)`).
* The compiled object: `CompiledCallable.reduction_order`, and
  `CompiledCallable.reduction_policy_pruned`, which holds the `(mode,
  reason)` pairs the enumerators pruned, also kept in the ranking record
  as `KernelRanking.policy_pruned`. On APU v1:
  `APUv1VectorCallable.reduction_order` and `.reduction_policy_pruned`.
* Every result: `RunResult.extra["reduction_order"]`, plus
  `extra["reduction_policy"]`, a list of `{"accumulator_type",
  "may_regroup"}` entries (matcher route). On the APU v1 vector route the
  result also carries `extra["plan_regroups_reduction"]` and
  `extra["numerics"]`.

**Typed program records** (`APUv1Program`, `APUG2TypedProgram`,
`APUG2ComposedContractionProgram`) spell out their own reduction grouping.
Passing `reduction_order` with one of them raises `ValueError`.

**Failure.** `spmw_reduction_policy.ReductionOrderError` (a `ValueError`)
names the policy, the target, the accumulator type and the reason, and
suggests `reduction_order="reassociate"`.

## 3. Where the policy is consulted

| Site | Split it governs | Under strict on a float accumulator |
|---|---|---|
| `spmw_match_engine.match_workload` | (stamps) | |
| `compiler.compile` -> `require_matcher_route_policy` | matcher routes whose enumerators do not consult the policy | AiM and the APU v1 matcher route raise before scheduling (below). Samsung and UPMEM are policy-aware and pass through. |
| `spmw_autoschedule._samsung_enumerate` | REDUCE program: reduction element k accumulates in FP16 lane k mod 16 of a GRF_B entry, and the host sums the 16 lane partials in FP32 (`_samsung_run_reduce`). `dual_fiber` also splits the reduction tiles over two bank fibers. | `bank_row`, `grf_staged` and `dual_fiber` are pruned **before ranking**. Each survivor records the pruned modes and reasons in `extra["reduction_policy_pruned"]`, surfaced as `KernelRanking.policy_pruned` and `CompiledCallable.reduction_policy_pruned`. Only `temporal` remains. |
| `SamsungCtx._emit_reduce` | a forced REDUCE `layout=` | `ReductionOrderError` (defence for user-supplied layouts, which bypass the enumerator) |
| `spmw_samsung.samsung_vendor_gemv_layout` | vendor (dual_fiber) layout | raises and lists the pruning reasons |
| `spmw_autoschedule._upmem_enumerate` | tasklet partials, `nc`/`kc` tiles | Every UPMEM family accumulates in int32/int64 (`UPMEM_ROLE_ELEMENT_TYPES`), so strict permits the splits. The enumerator raises `ReductionOrderError` if a non-integer accumulator ever reaches it. Classification currently rejects such types first. |
| AiM (`_REGROUPING_MATCHER_ROUTES["aim"]`) | every `MAC_ABK`/`MAC_SBK` reduces the 16 BF16 products of a burst through the bank adder tree before accumulating | `ReductionOrderError`. AiM has no source-order program. `replica_partitions` spreads a replica's *outputs* over channels and never splits the reduction axis (audit M-08). The CENT RMS bank-pair idiom (`spmw_aim._check_bank_pair_span`) does split one, but under strict it is unreachable because every AiM float reduction is refused. The `reduction_partitions` knob (token split, `spmw_knobs._aim_reduction_partitions_candidates`) reads `reduction_policy(match)` and offers only P = 1 when regrouping is forbidden. |
| APU v1 matcher route (`_REGROUPING_MATCHER_ROUTES["apu_v1"]`) | `gvml_add_subgrps` group tree | `ReductionOrderError`. The source-order plans live on the vector route. |
| `pim/apu_v1_vector_program._apply_reduction_policy` | `group_tree` / `pairwise` / `host` / `partial` reduction kinds (`apu_v1_plan_regroups`). `temporal_accumulate`, `serial` and `none` add one product per step in source order. | Regrouping plans are removed before `search_apu_v1_vector_plans` and the legacy incumbent ranking. `layout=<pruned plan>` raises. If no plan remains, the compile raises. |
| `APUv1VectorCallable._run_functional` (`backend="functional"`/`"virtual"`) | numerics of the selected plan | With a temporal plan the source program runs, which is the plan's order. With a regrouping FP16 plan (only possible with `"reassociate"`) the run uses `_grouped_fp16_contraction`: FP16 products, a halving FP16 tree over each `group_size` group, then groups added to the output in ascending order. That is the order of `fp16_contraction_ops`. The halving order is a model of `gvml_add_subgrps`; it has not been checked against the board. |

Gemini-II (`apu_v2`) has no source route. Its typed records reject the
argument.

## 4. Samsung ordered program (S-03)

**Placement.** `_samsung_enumerate` adds a `temporal` base candidate:
matrix in the even bank, vector in GRF_A, accumulator in GRF_B,
`extra["reduction_program"] = "temporal"`. Its layout is
`LinearLayout.identity({"lane": 16, "bank": 16})`: FP16 lanes index output
rows, not reduction elements. The layout also separates its decision key
from `bank_row`. Its knob domains (`spmw_knobs`) are `grf_residency =
host` only (the conductor broadcasts each element from the host) and
`stage_resident = False` (the matrix is staged again for every column).
`crf_issue = per_workid` with more than one work item is rejected at
materialization, as for REDUCE.

**CRF** (`SamsungCtx._emit_temporal`, the conductor's five-word contract in
`tests/TemporalReduce.inc`):

```
MOV  dst=GRF_B src0=EVEN_BANK          # initial accumulator (bank row 0)
MAC  dst=GRF_B src0=GRF_A src1=EVEN_BANK   # not auto: one column, one element
JUMP loop_counter=K-1 loop_offset=2
NOP                                     # drain the one GRF_B entry
EXIT
```

The conductor writes broadcast element k into GRF_A before MAC k, in
ascending k. Lane l of PIM unit b owns output row `tile*8192 + b*16 + l`.
The output is the drained FP16 GRF_B entry, and the host does no
arithmetic. The conductor contract is 1 <= K <= 4096 and one rank. Larger
K is `InfeasibleSchedule`.

**Runtime.** `SamsungCtx.runtime_route` reports `GENERIC_TEMPORAL_REDUCE`.
`_run_samsung` dispatches to `_run_samsung_temporal`, which calls
`pim_driver --op GENERIC --op-kind TEMPORAL_REDUCE --cmds crf.txt --inputs
A,B,init --out-rows M --reduce-k K --out-cols N --temporal-audit ...`. It
requires the receipt `PIM_TEMPORAL_REDUCE ordered_k=1 ...` and reports
`logical_shape`, `padded_shape` (rows padded to whole 8192-row tiles,
reduction unpadded), `padding_fraction`, `timing_scope`, `cycle_phases`,
`crf`, `conductor` and `levers`, as the REDUCE route does. GEMV and
GEMM (`N` columns) are supported.

**Conductor.** The rebuilt `pim_driver` (installed
`experiments/simulators/PIMSimulator`, identical to the overlay
`experiments/simulators/setup/PIMSimulator/src/{pim_driver.cc,
tests/TemporalReduce.inc}`) still supports `--op-kind TEMPORAL_REDUCE`.
No conductor change was needed.

**Cost rule** (`allo/pim/costs/samsung.py`, `temporal_mac`). Measured
2026-11-02, cycles per (8192-row tile, column):

| K | preload | exec | readback | total |
|---|---|---|---|---|
| 100 | 5,051 | 5,859 | 67 | 10,977 |
| 128 | 6,451 | 7,174 | 67 | 13,692 |
| 256 | 12,851 | 13,889 | 67 | 26,807 |
| 512 | 25,651 | 26,967 | 67 | 52,685 |

The execution phase fits 51.4 cycles per element plus 681 cycles per pass,
within 1.2% at every point (`TEMPORAL_STEP_CYCLES = 51`,
`TEMPORAL_PASS_CYCLES = 681`). The rule charges one pass per 16 output rows
of a work item, times the column count. Like the REDUCE rule it excludes
preload and readback. For the paper GEMV the estimate is 27,598 cycles,
against 4,389 simulated for the reassociating default. Under
`"reassociate"` both families are ranked and REDUCE wins (the ranking has
52 scores: 48 REDUCE, 4 ordered).

**Measured** (4096x256 GEMV, seed 0, data `* 0.1`):

| policy | program | cycles | bit-equal to source-order FP16 | max abs error vs FP32 |
|---|---|---|---|---|
| strict (default) | TEMPORAL_REDUCE | 26,807 | 4096/4096 | 2.24e-3 |
| strict, initial 0.5 (runtime) | TEMPORAL_REDUCE | 26,807 | 4096/4096 | |
| reassociate | REDUCE bank_row + host fill | 4,389 | 343/4096 | 4.47e-4 |

**Carried initial value.** `samsung_contraction` now accepts
`accumulator_init == "carried"` and records it in `SamsungContraction`. The
enumerator prunes the REDUCE candidates for it (they zero GRF_B).
`_run_samsung_temporal` loads the incoming output from runtime role `C`,
which `compiler._bind_carried_accumulator` binds to the source parameter.
The conductor carries the value correctly (test above). The public route
accepts carried accumulators on Samsung: `compile_for_target` no longer
refuses `carried` for `samsung_hbm_pim`, and the MAC pattern matches
`y[row] += W[row, k] * x[k]` through distinct SSA index values, so
`allo.compile` selects the ordered program and the simulator result equals
source-order FP16 accumulation from the caller's `y`
(`tests/pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_carried_accumulator_through_the_public_api`).
AiM and APU v1 matcher routes still refuse `carried` (their devices zero the
accumulator).

## 5. APU v1 (V1-03)

4096x256 FP16 GEMV, functional backend:

| policy | selected plan | pruned | bit-equal to source order |
|---|---|---|---|
| strict | `temporal_dma_coalescing` | `baseline_spatial_reduction`, `spatial_gemv_group_reduction_k<tile>` | 4096/4096 |
| reassociate | `spatial_gemv_group_reduction_k<tile>` | none | 425/4096 (plan-order model) |

## 6. Tests

New:

* `tests/pim/test_reduction_policy.py` (compile-only): policy semantics and
  integer exemption, stamping by `match_workload`, Samsung pruning before
  ranking with reasons, the exact five-word CRF, ranking of both families
  under reassociate, forced vendor layout refused under strict,
  `RunResult.extra` stamping (virtual backend), UPMEM integer split
  allowed, AiM BF16 refused naming the policy.
* `tests/pim/samsung_hbm_pim/test_samsung_reduction_order.py` (PIMSimulator):
  strict default = ordered program, 26,807 cycles, bit-equal to source-order
  FP16, padding and timing scope. Reassociate differs from source order in
  more than half the rows while matching FP32 within 1e-2. Initial 0.5
  carried by the runtime and conductor. A GEMM with two columns is bit-equal.
* `tests/perf/test_apu_v1_reduction_policy.py` (compile-only and
  functional): strict ranks only temporal plans and refuses a named tree
  plan. Reassociate admits tree plans. Functional numerics follow the
  selected plan. A uint16 contraction keeps every plan.

Existing tests that now pass `reduction_order="reassociate"` explicitly,
because they exercise regrouping programs:

* `tests/pim/samsung_hbm_pim/test_samsung_lowering.py`: the `paper` and
  `vendor` fixtures, `_forced`, k512, and the operand-order test.
* `tests/pim/samsung_hbm_pim/test_samsung_conductor_runs.py`: `_forced`,
  canonical run, tampered CRF, ragged, GEMM residency.
* `tests/pim/samsung_hbm_pim/gemv/test_gemv_samsung_hbm_pim.py`: the
  paper rows (scores 48 -> 52).
* `tests/pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py`: the
  REDUCE readback path.
* `tests/pim/test_paper_golden.py`: Samsung and AiM rows.
* `tests/pim/test_gemv_all_targets.py`: the AiM tests. The Samsung test
  keeps the strict default and now runs the ordered program.
* `tests/pim/test_aim_matcher.py`: every `allo.compile`.
* `tests/perf/test_matcher_ranked_placement.py`: `_match` and the AsmState
  reference.
* `tests/perf/test_apu_v1_vector_compile.py`: both tests. The functional
  reference became FP32 with FP16 tolerance, because the run now reproduces
  the selected tree plan.

## 7. Open

1. `spmw_codegen.compile_for_target` still rejects `carried` on Samsung.
   Removing `samsung_hbm_pim` from that gate would make the carried ordered
   program reachable. The enumerator, codegen, binding and runtime are
   ready.
2. The AiM and APU v1 matcher-route reasons live in the policy module as a
   table. They belong next to those backends' enumerators.
   `require_matcher_route_policy` runs in `allo.compile`. A direct
   `compile_for_target` call on an AiM trace with strict stamps is not
   refused.
3. The APU v1 functional tree model has not been checked against the board
   (no hardware runs in this change).
4. The policy is per compile. A per-reduction policy (one strict reduction
   and one regrouped reduction in the same workload) has no syntax yet. The
   stamp is per match, so the enumerators already read it per reduction.


## Board finding on APU v1 (round 6, V1-16)

`tests/pim/apu_v1/test_g6_board_semantics.py` ran both plans of a 512x256
and a 384x200 FP16 GEMV on the Gemini-I board. Both plans agree with FP32
within FP16 tolerance, but neither is bit-equal to its host model: about
half of the outputs differ from the source-order loop (strict plan) and
from the halving tree (reassociating plan) by a few ulps. The strict policy
on APU v1 therefore guarantees that the selected plan does not regroup the
reduction, and the functional route models the plan's order; it does not
guarantee bit-equality with an IEEE FP16 round-to-nearest model, because
the GVML FP16 multiply and add have not been characterized at that level.
The bit-equality test is kept as a strict xfail with the mismatch counts
printed.
