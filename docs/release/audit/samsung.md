# Samsung HBM-PIM backend: release audit

Audited revision: `experiments/allo` branch `tenon` @ `8160d6d82c28211900e156fcff6e0b452e94131d`
(working tree clean apart from untracked `scripts/verifier_check_*.py`).
Simulator: `experiments/simulators/PIMSimulator/pim_driver` (binary mtime 2026-09-29 06:13,
built from `src/pim_driver.cc` mtime 06:12 and `src/tests/PIMKernel.cpp` mtime 06:12).
Date: 2026-11-01. Read-only audit; probe scripts and raw logs in
`/tmp/tenon-release/probes/samsung/` (`probe_a.log`, `probe_b.log`, probe C/E/F output inline below).

Every claim is tagged **[executed]** (observed by running code here), **[read]** (source
evidence with `file:line`), or **[suspected]** (inferred, not independently executed).

Prior audit items rechecked: A03 (selected program replaced by canonical conductor) **confirmed**;
A04 (valid zero outputs dropped) **confirmed**; A05 (guessed geometry fallbacks) **partly
confirmed / partly misattributed** (see S-09).

---

## 1. Routes and entry points (API -> production symbols -> emitted code / runtime)

### 1.1 Main route: `allo.compile(workload, build_samsung_target(), samsung_cost)` **[read, executed]**

| Stage | Production symbol | What it produces |
|---|---|---|
| Target | `allo/pim/targets.py:24-245 build_samsung_target` | Unit tree `pseudo_channel{16} > pim{8}` with `banks(16)`, `crf(32)`, `grf_a/grf_b(8x256b)`, `srf`; device moves LD_A/LD_B/ST_A/ST_B/JUMP; ops MUL/ADD/MAC/MAD/RELU; host moves BCAST_GRF_A/B, BCAST_SRF, PROGRAM_CRF, SCATTER_BANKS, GATHER_BANKS. `work_grid() == ([16, 8], 128)`. |
| Frontend | `allo/compiler.py:635-740 compile` -> `customize()` -> `match_workload()` (`spmw_match_engine.py:1670`) | A `MatchTrace` of 128 `MAC` matches (one per work-id) for the paper GEMV; roles `x` (first multiplicand), `y` (second), `acc`. |
| Autoschedule | `spmw_autoschedule.py:542-672 _samsung_enumerate` + `spmw_knobs.py:450-462` knobs (`grf_residency`, `crf_issue`, `stage_resident`, `residency`, `tile`, `double_buffer`) -> `rank_matcher_placements` | 24 placements per bucket (3 bases `bank_row`, `grf_staged`, `dual_fiber` x 2 x 2 x 2), 48 scored keys with the whole-trace residency knob. Two cost classes: 226 cycles (8 `dual_fiber` keys) and 275 (16 others). Chosen position 32 = `dual_fiber+crf_shared`, `grf_residency={local_W: crf}`, `stage_resident=False` **[executed, probe_a]**. |
| Codegen | `spmw_samsung.py:30-283 SamsungCtx` via `_walk_buckets` (`spmw_codegen.py:583-790`) | `compiled.cmds`: `[FILL GRF_A<-EVEN_BANK, MAC GRF_B<-GRF_A*EVEN_BANK auto, JUMP lc=15, MAC ...ODD_BANK, JUMP lc=15, NOP lc=7]` (6 cmds, shared CRF) or 768 cmds for `crf_issue=per_workid` **[executed]**. `host_schedule` has 128 `HostTrigger`s for shared CRF. |
| Runtime | `spmw_samsung.py:833-1130 _run_samsung` | Route chosen by opcode set (`_samsung_runtime_route` `:807-810`): `{MAC}` -> `GENERIC_REDUCE`; anything else -> `GENERIC`. |
| Callable | `compiler.py:437-560 CompiledCallable` | `source_backend_binding` (`spmw_codegen.py:883-935`) maps MAC role `x`->kwarg `A`, `y`->kwarg `B` (`_SAMSUNG_MAC_ROLE_KWARGS :880`); output = the one unreferenced parameter. `_copy_outputs` writes `extra["outputs"]` into the output argument with `np.copyto(..., casting="same_kind")`. |

**What reaches the simulator on route (a), `GENERIC_REDUCE`** **[executed, probe_a]**:

```
pim_driver --op GENERIC --out out.bin --inputs A.npy,B.npy --crf reduce.crf --op-kind REDUCE
  --out-rows 4096 --out-cols 1 --reduce-k 256 --num-tiles 1 --bank-type ALL
  --roles A:0:0:2:0:0:0:2,B:0:0:2:0:0:0:0,out:0:0:1:1:0:8:0
```

The `--crf` file is **not** `compiled.cmds`; it is the fixed five-line text written at
`spmw_samsung.py:991-997` (identical copy at `:649-655`). `compiled.cmds` is explicitly skipped
(`:1069 if pim_cmds_all and kernel != "GENERIC_REDUCE"`). In the simulator, `executeGeneric`
(`PIMKernel.cpp:1043-1116`) ignores even that CRF for REDUCE and regenerates the vendor GEMV
microcode: `PIMCmdGen::getPIMCmds(KernelType::GEMV, 0, num_jump_odd, num_jump_even)` at
`PIMKernel.cpp:1085-1091`, then replays `computeGemv` (`:665-692`) through
`issueReduceColumnCommands` (`:989-1041`); the weight is placed by the vendor `preloadGemv`
(`loadGeneric :846-861`). The only compiler-derived quantities that reach the run are the
operand arrays, `M_real`, `K`, `N`, and `num_tiles` (from `_samsung_reduce_partition :483-570`).

**Placement perturbation** **[executed, probe_a]**: forcing `layout=` to `bank_row+crf_shared`
(4 cmds, JUMP lc=31), `grf_staged+crf_shared` (MAC `GRF_A*GRF_A`, no bank operand), and
`dual_fiber+crf_per_workid+wresident` (768 cmds) produced byte-identical argv, identical
`reduce.crf`, identical cycles (4435) and identical output bytes to the default. The selected
layout, GRF residency, CRF issue mode and fiber count do not change the executed program.

**What reaches the simulator on route (b), `GENERIC` (elementwise)** **[read + executed via
internal API only]**: `--op GENERIC --inputs in0.npy,in1.npy --num-tiles N --bank-type ALL
--cmds cmds.txt` where `cmds.txt` is `compiled.cmds` after `_samsung_main_runtime_commands`
(`:825-830`, drops bank<-GRF MOV/FILL). The C++ `executeGeneric` ELTWISE branch does upload and
execute this CRF (`programCrf(cmds)` `:1100`; `issueTileColumnCommands :886-987`). This is the
only route where the emitted instruction stream determines numerics, but **no `allo.compile`
workload reaches it**: the Samsung enumerator raises for any non-MAC trace (S-07).

### 1.2 Secondary routes

- `Compiled.run_batched(W, X)` (`spmw_codegen.py:1017-1034` -> `_run_samsung_batched`
  `spmw_samsung.py:1255-1353`): `--op GEMV --faithful --batch B [--native-rebaseline]` with
  `--cmds` = `compiled.cmds` after `_crf_valid` (drops FILL and storeback MOV). The C++
  `executeGemvFaithful` (`PIMKernel.cpp:585-664`) uploads the emitted CRF but drives the
  cadence with the vendor `computeGemv`; extra MAC bodies "issue the same MAC strobe against a
  scratch row (priced, not accumulated)" (`:626-631`). **[executed, probe_f]**: `bank_row`,
  `grf_staged`, `dual_fiber` all give 18095 cycles for B=4; only `stage_resident=True`
  changes the result (8741) by toggling `--native-rebaseline`. No outputs are read back
  (`extra` has no `outputs`). Not exercised by any test (`grep run_batched tests/` empty).
  Not reachable from `CompiledCallable` (no `run_batched` wrapper).
- `run_samsung_reduce` / `execute_host_schedule_samsung` (`spmw_samsung.py:707-760`): harness
  helpers that call the same REDUCE conductor per `HostSchedule` group; no compile artifact
  involved.
- `allo.pim.samsung_layout_swizzle.select_autoscheduled_xor_layout` (`:195-288`): builds a
  synthetic one-match `MatchTrace` (not a user workload), runs `autoschedule`, lifts the chosen
  `LinearLayout` bank projection. Execution of the 148/116 numbers is through
  `scripts/run_pimsim_layout_trace.py` (repo `scripts/`, not the `allo` package) compiling the
  test-tree harness `tests/pim/samsung_hbm_pim/layout_swizzle/pimsim_layout_trace.cc` against
  `libdramsim2.a`. **[executed, probe_e]**: identity 148 cycles, synthesized XOR 116 cycles;
  the selected placement mode was `bank_row+crf_shared`. The trace is 4 `round bank row col`
  lines replicated over 64 channels; it is not the GEMV program and does not alter it.
- `backend="virtual"`: `_run_virtual` (`spmw_codegen.py:845-862`) evaluates the cost graph only.

### 1.3 Simulator side: which driver actually runs **[executed: md5, strings, diff]**

| Artifact | Lines / md5 | Notes |
|---|---|---|
| Committed overlay `experiments/simulators/setup/PIMSimulator/src/pim_driver.cc` | 323 lines, `874cb669...` | Accepts `--op ADD|MUL|RELU|GEMV` only; calls `executeEltwise(WithCmds)`, `preloadNoReplacement`, `readData`. Has **no** `--op GENERIC`, `--op-kind`, `--roles`, `--reduce-k`. |
| Installed `experiments/simulators/PIMSimulator/src/pim_driver.cc` | 721 lines, `488554f9...` | `GEMV|GENERIC`, REDUCE, TEMPORAL_REDUCE, `--slice-plan`, SHA receipts. Byte-identical to `dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/pim_driver.cc`. |
| Installed `src/tests/PIMKernel.cpp/.h` | identical to the `dev/` copies (`9f483a0d...`, `ef1b565b...`) | Contain `loadGeneric/executeGeneric/readGeneric/issueReduceColumnCommands/runTemporalReduce/...SamsungSlicedReduce`. **Not present in `setup/` at all.** |
| `dev/` copies in the containing repo | tracked, but `git diff --stat` shows **450 uncommitted lines** (`pim_driver.cc +189`, `PIMKernel.cpp +220`, `PIMKernel.h +41`) | The installed = dev working tree, not any commit. `SamsungSlicePlan.{h,cpp}` differ between `dev/` and installed. `Burst.h`, `KernelAddrGen.h`, `TemporalReduce.inc` identical. |
| Installed `Sconstruct` vs `setup/` `Sconstruct` | differ (installed drops `gtest` from the driver link line) | |
| Binary `pim_driver` | `strings` contains `--op-kind`, `--reduce-k`, `--roles`, `--native-rebaseline`, `TEMPORAL_REDUCE requires GENERIC...` | Built from the installed (dev) sources. `PIMSimulator/.git` points to a missing `.git/modules` dir; no git history is available for the installed tree. |

Consequence: `bash setup/apply.sh` on a fresh clone installs a driver that cannot serve
`_run_samsung` (which always passes `--op GENERIC`), and no committed source can rebuild the
binary that produced the paper cycles (S-04).

---

## 2. Enforced vs advertised domain

| Aspect | Advertised (paper / skill / docstrings) | Enforced at compile | Enforced at run | Observed **[executed, probe_b]** |
|---|---|---|---|---|
| Kernel family | GEMV example (`3-language-and-compiler.tex:113-130`), "add/mul/relu" (`SKILL.md:87-92`), ADD/MUL/MAD/RELU declared on target | Only traces whose roles are exactly `{x, y, acc}` enumerate (`spmw_autoschedule.py:556-563`). ADD (`x,y`) and MAD-shaped `y += W*x` (`x,y,z`) raise `NotImplementedError("...missing one of x/y/acc roles")` | n/a | `add_131072`: compile fails. `yacc_4096x256`: compile fails. |
| Grid / mapping | "16x8 logical grid" | Any `@allo.work(mapping=[P,U])` with `P<=16`, `U<=8` compiles; mismatch vs `work_grid()` only **warns** (`spmw_codegen.py:772-782`, `:1172-1195`) | `_samsung_reduce_partition` warns, uses trace count | `mapping=[16,4]`: 3 warnings, runs, cycles 4435 identical to `[16,8]`. |
| M | paper 4096 / 8192 | none | Padded to multiple of 4096 (`_SAMSUNG_FABRIC_ROW_TILE :457`, `:968-970`); `M_real > ROWS*n_workids` only warns (`:952-958`) | M=2048, 128: correct outputs, **cycles 4435 = same as M=4096**. M=6144: 2 tiles, 8247 cycles (= paper M=8192 row). |
| K | paper 256 / 512 (K=1024 in paper code listing) | `_emit_inner_loop_jump` folds `K // lanes` | Padded to multiple of 256 (`_SAMSUNG_REDUCE_K_TILE :480`) | K=128, K=100: correct, cycles 4435 (= K=256). K=320: correct, cycles 8041 (= K=512). |
| N (GEMM) | not claimed | 3-deep nest accepted (`:516-519`) | `--out-cols N`, B split into N GEMV passes in C++ | N=2: 5370 cycles, N=3 (M=2048): 6638 cycles, correct. |
| Operand order | "ordinary tensor code" | `x[k] * W[i,k]` compiles; binding becomes `{A: x, B: W}` | `np.asarray(A).reshape(M_real, K)` fails | **`ValueError: cannot reshape array of size 256 into shape (4096,256)`** on a semantically identical program. |
| Output init | `y[row] = acc` with `acc: f16 = 0` | matcher requires loop-carried scalar; initializer value unchecked | REDUCE conductor zeroes GRF_B each pass (`PIMKernel.cpp:1063-1064`); `_ROLE_IS_READ_ONLY` treats `acc` LD as zero-init (`:177,196-198`) | `y_init=2.0`: overwritten (correct for this form). `acc: f16 = 0.5`: compiles, device returns `W@x` (0.5 lost). |
| Zero results | real outputs | n/a | readers return `{}` when `not np.any(...)` (`:444-445`, `:591-592`) or blob short (`:587-588`) | `x = 0`: `RunResult.cycles=4435`, no `outputs`, `y` left at its initial ones. |
| Host moves | explicit `host_xfer.*` | `host_moves` with labels not naming parameters: `ValueError: host transfer 0 lacks an exact ABI metric binding` (compile) | `_assert_host_move_roles` warns, falls back to inference | compile rejects mismatched labels before the run-time warning path is reached. |
| Ordered FP16 | "ordered fallback ... preserves source-order FP16" (`5-evaluation.tex:97-102`) | not present in package | not present | device arithmetic is per-lane fp16 with 16-way regrouping (S-03, S-05). |
| dtype | FP16 | source `fp16` annotations; runtime casts everything with `np.asarray(..., dtype=np.float16)` | | |

---

## 3. Findings

### S-01 (blocker) Selected placement/layout/schedule does not determine the executed program
- **Where**: `spmw_samsung.py:984-997, 1069` (emitted stream skipped; fixed CRF text);
  `PIMKernel.cpp:1073-1095` (`PIMCmdGen::getPIMCmds(KernelType::GEMV, ...)` regenerated),
  `:846-861` (vendor `preloadGemv` places the weight), `:989-1041` (vendor `computeGemv`
  cadence). Batched route: `PIMKernel.cpp:605-631` (redundant bodies strobe a scratch row).
- **What the code does** **[read, executed]**: the autoscheduler ranks 48 keys (226 vs 275
  cycles) over `bank_row`/`grf_staged`/`dual_fiber`, GRF residency, CRF issue, stage
  residency, then codegen emits a matching `PIMCmd` stream. `_run_samsung` discards it and
  the simulator runs the vendor GEMV. Probe A: four distinct placements (4, 6, 768 cmds;
  `GRF_A*GRF_A` vs bank operands) -> identical argv, CRF, 4435 cycles, identical `y`.
  Probe F (`run_batched`): three placements -> 18095 cycles each; only `stage_resident`
  toggles a driver flag (8741).
- **Gate**: G4 (chosen plan must survive to dispatch; "simulator conductors must not replace
  it with a canonical benchmark implementation"), G1 (`eval-samsung.tex` 1.00x rows are the
  vendor program measured twice), G5 (cost differences never reach execution).
- **Reproduce**: `python /tmp/tenon-release/probes/samsung/probe_a_route.py` -> `COMPARE[...]
  same_cycles=True same_argv=True same_crf=True same_outputs=True` for all layouts.
- **Repair**: either (i) make the REDUCE conductor consume the emitted CRF and the placement's
  operand assignment (bank-resident operand, GRF operand, fiber split, JUMP trip counts derived
  from `K // lanes`), with the Python side validating the 32-word cap instead of rebuilding
  text; or (ii) document in `CLAIMS.md`/`RunResult.extra` that Samsung execution is
  "vendor-equivalent reconstruction, placement-insensitive" and stop presenting the
  placement ranking as determining Samsung cycles. (i) is required for G4; (ii) alone fails G4.

### S-02 (blocker) Valid all-zero or truncated outputs are reported as success without outputs (A04)
- **Where**: `spmw_samsung.py:444-445` (`if not np.any(out): return {}`), `:587-588`
  (`raw.size < need: return {}`), `:591-592` (`if not np.any(grid): return {}`), `:436-437`,
  `:581-582` (missing file -> `{}`); `_run_samsung :1100-1115` returns `RunResult(cycles=...)`
  with no `outputs` and no error; `CompiledCallable._copy_outputs` (`compiler.py:512-515`)
  returns silently when `outputs` is empty.
- **Observed** **[executed, probe_a]**: `x = zeros`, `y = ones` -> `cycles=4435`,
  `extra` keys `['kernel','returncode']`, `y` unchanged (all ones; correct answer is zeros).
  Probe C: short blob -> `{}`; absent file -> `{}`.
- **Gate**: G2 ("zero-valued outputs ... No ... missing-output success").
- **Repair**: return the array unconditionally; raise on absent/short blob; make
  `_copy_outputs` fail when a bound output receives nothing.

### S-03 (blocker) The paper's "ordered fallback" (source-order FP16 execution) is not in the release
- **Paper**: `5-evaluation.tex:97-102` ("Both match all 4,096 reference outputs when each
  multiply and addition rounds to FP16 in source order ... Samsung's ordered fallback
  currently lacks cost modeling and layout search").
- **Where it actually lives** **[read]**: simulator side only, `PIMKernel.cpp:22` includes
  `tests/TemporalReduce.inc` (`runTemporalReduce`, 5-word CRF contract, one GRF slot, K
  issued in ascending order, no host lane reduction) and `pim_driver.cc:380-404`
  (`--op-kind TEMPORAL_REDUCE`). Both exist only in the installed/dev sources (S-04). The
  Python driver `allo/pim/samsung_temporal.py` and its `compiler.py` dispatch exist only in
  snapshot commit `df35093` on branch `snapshot/shared-autoscheduler-20260813`, which is not an
  ancestor of `tenon`; `design_doc/compiler/samsung-bounded-placement.md` D6 explicitly ruled
  "do not port". `grep -rn ordered|temporal|strict allo/` finds nothing Samsung-related.
- **Observed** **[executed]**: driving the installed binary directly with
  `--op-kind TEMPORAL_REDUCE` on the 4096x256 GEMV with `initial = 0.5`: `rc=0`,
  `PIM_TEMPORAL_REDUCE ordered_k=1 host_arithmetic=0 initial=1`, 26807 cycles, output matches
  strict source-order two-rounding FP16 for **4096/4096** rows and preserves the initial value.
  So the mechanism the paper describes is real but unreachable from `allo.compile` at 8160d6d.
- **Gate**: G1 ("Research code stranded in ignored `dev/`, another branch/worktree ... does
  not satisfy this gate"), G2 (strict vs reassociative reduction must be explicit).
- **Repair**: port an ordered Samsung lowering into the package as an explicit, documented
  route (e.g. a `reduction="ordered"` policy or a selected placement whose cost rule exists),
  with the TEMPORAL conductor committed in the simulator overlay; or, if dropped, the claim
  matrix must record the paper statement as unreproducible from the release.

### S-04 (blocker) The simulator driver that runs is not reconstructible from committed sources
- **Where** **[executed: md5/diff/strings]**: Section 1.3. Committed overlay = 323-line driver
  with the deleted `--op ADD|MUL|RELU` API; installed = 721-line dev conductor plus
  modified `PIMKernel.{cpp,h}` (not in `setup/`), plus `SamsungSlicePlan.*`,
  `TemporalReduce.inc`, `KernelAddrGen.h` (also not in `setup/`). The dev copies themselves
  carry 450 uncommitted lines. `pim_driver.cc:548-556` even hashes
  `../../../dev/06292026-.../SamsungSlicePlan.cpp` at run time for the `--slice-plan` receipt
  (a compiled-in dependency on the containing repo's `dev/`).
- **Gate**: G7 ("reproducible simulator sources, versions, patches, build recipes"; "Ordinary
  compilation does not import `dev/`"), G8 (evidence at the exact candidate revision).
- **Reproduce**: `diff experiments/simulators/setup/PIMSimulator/src/pim_driver.cc
  experiments/simulators/PIMSimulator/src/pim_driver.cc | wc -l` -> 599;
  `strings pim_driver | grep -- --op-kind`.
- **Repair**: commit the full patch set the Python runtime requires (`pim_driver.cc`,
  `PIMKernel.cpp/h`, `TemporalReduce.inc`, `Sconstruct`, and `SamsungSlicePlan.*` or remove
  the slice-plan path) under `setup/PIMSimulator/`, pin the upstream SAIT commit, drop the
  `dev/` hash path, rebuild from a clean clone and re-record the paper cycles.

### S-05 (major) Operand roles and placement levers depend on source operand order, not semantics
- **Where** **[read]**: matcher role `x` is the first factor of `acc + x*y`
  (`targets.py` MAC `fn=lambda x, y, acc`); `spmw_autoschedule.py:437-462
  _samsung_host_eligible_memrefs` treats role `x` as "the broadcast vector"; the enumerator
  places `x -> grf_a`, `y -> bank` (`:620-660`); `source_backend_binding` maps `x -> "A"`
  (weight kwarg) and `y -> "B"` (`spmw_codegen.py:880`); `_run_samsung :930` reshapes `A`
  to `(M_real, K)`.
- **Observed** **[executed, probe_b]**: paper form `W[i,k] * x[k]` -> placement
  `{local_W: grf_a, local_x: bank}`, `grf_residency={local_W: crf}`, binding `{A: W, B: x}`,
  runs. Swapped form `x[k] * W[i,k]` -> placement `{local_x: grf_a, local_W: bank}`, binding
  `{A: x, B: W}`, run fails `ValueError: cannot reshape array of size 256 into shape
  (4096,256)`. Note also that in the accepted form the **weight** is the operand placed in
  GRF_A and treated as host-broadcast-eligible, while the conductor actually broadcasts the
  vector into GRF_A and keeps the weight bank-resident: the lever semantics are inverted
  relative to the executed hardware program.
- **Gate**: G6 (rename/reorder-equivalent program must compile to the same decisions), G3,
  G2 (actionable error; this one is not).
- **Repair**: classify MAC operands structurally (which operand is indexed by the output row
  and reduction axis vs only the reduction axis) in the matcher/binding, and make the
  enumerator/host-eligibility use that classification; reject with a clear message if the
  classification is ambiguous.

### S-06 (major) Reported "cycles" are those of the padded canonical program, not the compiled shape
- **Where** **[read]**: `_run_samsung :959-973` pads K to 256 and M to 4096 before the run;
  `RunResult` reports the driver's `PIM_CYCLES total` with no indication of padding;
  `extra["cycles_source"]` says "PIMSimulator GENERIC REDUCE interpreter".
- **Observed** **[executed, probe_b]**: K=128, K=100, M=2048, M=128, mapping `[16,4]` all
  report exactly 4435 cycles (the 4096x256 paper number); K=320 reports 8041 (the K=512
  paper number); M=6144 reports 8247 (the M=8192 paper number). The virtual `estimate()`
  for these shapes differs from the simulator (not measured here) **[suspected]**.
- **Gate**: G5/G8 ("Distinguish simulator functional checks, simulator timing ...";
  "A model without a validated performance domain must not silently advertise performance
  predictions"). Zero-padding preserves numerics but silently changes the timing scope.
- **Repair**: record `padded_shape`, `logical_shape` and a `timing_scope` field in
  `RunResult.extra`; refuse (or clearly flag) timing claims when padding exceeds a documented
  bound.

### S-07 (major) Elementwise ops are advertised but unreachable; the GENERIC runtime is untested and mis-tiled
- **Where** **[read]**: `targets.py` declares MUL/ADD/MAD/RELU (lines ~134-205),
  `costs/samsung.py:63-69` prices them, `_run_samsung :1020-1058` has the `GENERIC` branch,
  `SKILL.md:87-92` advertises `add/mul/relu`; but `_samsung_enumerate :556-563` raises for any
  trace without an `acc` role. `grep -rn "GENERIC\b|ELTWISE|_run_samsung\b" tests/` finds
  only `test_simulator_unavailable.py:29`.
- **Observed** **[executed]**: `add_131072` compile -> `NotImplementedError: ... got ['x','y']`
  (probe_b). Driving `_run_samsung` directly with the canonical ADD CRF (probe_c): numerics
  correct, but `elems_per_tile = 16*8*num_pim_blocks = 1024` (`:1049`) vs the driver's
  `131072` (`pim_driver.cc:93`), so 131072 elements -> `--num-tiles 128`, 89963 cycles and a
  33 MB `out.bin`, where `--num-tiles 1` gives the same correct 131072 results in 1588 cycles.
- **Gate**: G1/G3 (advertised capability without a reachable route), G5 (timing of the
  reachable-by-API route would be 56x inflated), G8.
- **Repair**: either add an elementwise enumerator/placement family (roles `x,y,dst`) with
  tile derivation from `channels * pim * grf * lanes * 2 banks`, plus tests; or remove the
  elementwise ops/route from the public target and docs.

### S-08 (major) Nonzero accumulator initializers are silently dropped; accumulate-into-output form is rejected with a misleading diagnostic
- **Where** **[read, executed]**: `y[i] += W[i,k]*x[k]` matches MAD (roles `x,y,z`) and the
  enumerator raises "trace is missing one of x/y/acc roles; got ['x','y','z']"; the only
  accepted form has a scalar `acc: fp16 = 0`. The REDUCE conductor zeroes GRF_B per pass
  (`PIMKernel.cpp:1063`), and `resolve_moves` emits no LD for `acc` (`spmw_samsung.py:196-198`,
  "TODO(task-015): revisit when accumulators carry across calls"). The matcher does not check
  the `acc` initializer (`grep -n "zero|init" spmw_match_engine.py:1490-1520` finds no check).
- **Observed** **[executed, probe_g_accinit]**: the paper GEMV with `acc: fp16 = 0.5` compiles
  with no warning and runs (4435 cycles); the returned `y` equals `W@x` (max error 4.5e-4) and
  differs from the source semantics `0.5 + W@x` by 0.5 on every row. Silent wrong code.
- **Gate**: G2 ("Finding one zero store somewhere is not a proof that an accumulator is
  initialized"; nonzero initial outputs must reach the runner or be rejected).
- **Repair**: add a matcher/enumerator precondition that the loop-carried accumulator is
  initialized to the literal 0 (or lower a nonzero init as a host epilogue add); name the
  rejected `y += ...` pattern explicitly in the error.

### S-09 (minor) A05 recheck: guessed-geometry fallbacks
- `_samsung_num_pim_blocks` (`spmw_samsung.py:416-427`) **[executed]**: returns 8 for a
  target without `work_grid` (`SimpleNamespace`), and 8 for the real target; only reachable
  on the GENERIC route (unreachable from `compile`, S-07) and the derived tile size is wrong
  regardless (S-07). Replace the broad `except` with a hard error and derive the tile from
  the full geometry.
- `_eval_sym` UnitId -> 0 (`spmw_codegen.py:319-356`) **[read]**: lives in the "Samsung"
  section but is **only called from `spmw_aim.py:120,178`**; the Samsung ctx classifies bank
  indices structurally via `_bank_fiber_class` (`:394-413`) and raises on non-canonical
  forms (`spmw_samsung.py:139-146`). A05's attribution to Samsung is incorrect; the default
  remains a G2 risk on the AiM path (out of scope here).
- `_assert_host_move_roles` (`spmw_samsung.py:763-804`) **[executed]**: `compile(...,
  host_moves=records)` with labels not naming parameters fails earlier
  (`ValueError: host transfer 0 lacks an exact ABI metric binding`). The warn-and-continue
  path is reachable only through the documented escape hatch `run_backend(**kwargs)` when a
  kwarg omits a bound role **[suspected]**. Low risk; make it an error or delete.

### S-10 (minor) Work-grid and partition mismatches only warn, and do not affect execution
- **Where**: `spmw_codegen.py:772-782`, `:1172-1195`; `spmw_samsung.py:536-544, 952-958`.
- **Observed** **[executed]**: `mapping=[16,4]` -> 3 warnings, run succeeds with cycles
  identical to `[16,8]`. The mapping therefore neither changes the executed program (S-01) nor
  is validated as a legality condition.
- **Repair**: make under-coverage of the operand (`M_real > ROWS*n_workids`) an error; decide
  whether a non-full grid is legal and document it.

### S-11 (minor) Documentation, discovery and dev-path residue (G7)
- `docs/source/` has no Samsung/Tenon backend page (`grep -ri samsung docs/source` empty).
- `.claude/skills/pim-samsung/SKILL.md:68,72,107` still documents `--in0/--in1`, "GEMV
  hardcoded to output_dim = 4096", add/mul/relu via driver; all stale vs the installed driver.
- `spmw_simenv.py:22-23,55-56`: default simulator root is `<allo>/../simulators/PIMSimulator`
  (outside the release tree); `PIMSIMULATOR_ROOT` override exists. Fine if documented.
- `pim_driver.cc:548-556` hashes `../../../dev/...` files at run time (slice-plan route only).
- `tests/pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py` depends on repo
  `scripts/run_pimsim_layout_trace.py` and `g++`; not part of the installed package.

### S-12 (minor) Cost model provenance and degenerate ranking
- `costs/samsung.py:20-27` constants (`MAC_CYCLES=4`, `BANK_LOAD_CYCLES=26`, ...) carry
  "migrated from the initial PIMSimulator/datasheet model" with no units/scope/measurement
  record; `mac` rule defaults `lanes=8` (`:57`), `host_transfer` defaults `burst_bytes=32`
  (`:111`). Ranking has 16-way and 8-way ties broken by enumeration position
  (`rank_matcher_placements`), and (S-01) the winner never changes execution, so G5's
  "changing a relevant cost changes ... the selected program" cannot be demonstrated on the
  simulator.

### S-13 (minor) XOR swizzle result is a standalone microbenchmark, not a compile-route effect
- **[executed, probe_e]**: 148 -> 116 cycles reproduced through `select_autoscheduled_xor_layout`
  + `scripts/run_pimsim_layout_trace.py` + `pimsim_layout_trace.cc`. The autoscheduler is
  invoked on a synthetic `MatchedOp` (K = tile_count = 2) and the lifted layout is the single
  tile->bank-bit-0 swizzle the Samsung enumerator always produces. The production GEMV packer
  does not consume this layout (module docstring `samsung_layout_swizzle.py:23-27` says so).
  Acceptable as the paper's "isolated layout-mechanism ablation" only if `CLAIMS.md` labels it
  as such and the test tolerates environment absence (it does: skips without `libdramsim2.a`).

---

## 4. Constants / specialization inventory

| Item | Location | Category (TASK.md table) | Judgment |
|---|---|---|---|
| 16 channels, 8 pim/channel, 16 banks, 4 bank groups, 16384 rows x 128 cols, GRF 8x256b, CRF 32 words | `targets.py:62-93` | ISA / physical fact | Sourced to PIMSimulator ini + ISCA'21; OK. Note: simulator runs 64 channels (`pim_driver.cc make_kernel`), target says 16; the cost model and `work_grid()` use 16. Document the 16-vs-64 relationship **[read]**. |
| `_SAMSUNG_FABRIC_ROW_TILE = 4096` | `spmw_samsung.py:457` | Derived (64ch x 8 blocks x 8 GRF_B) hard-coded | Should be computed from simulator config (channels x pim x grf_b); currently pasted and tied to the 64-channel ini. |
| `_SAMSUNG_REDUCE_K_TILE = 256` | `:480` | Derived (2 input tiles x 8 GRF_A x 16 lanes) hard-coded | Same; also a workaround for a degenerate conductor case. |
| `k_bursts = ((k_pad // 16) + 7) // 8` | `:636, :980` | Derived | OK but "advisory" (C++ recomputes). |
| `--roles "A:0:0:2:0:0:0:{k},B:0:0:2:0:0:0:0,out:0:0:1:1:0:8:0"` | `:679, :1016` | Template / target idiom | Bank parity 2 (ALL), out base row 0, col stride 8: opaque literals; the C++ REDUCE path mostly ignores them. Derive or drop. |
| Fixed five-line `reduce.crf` text | `:649-655, :991-997` | Template | Placeholder to satisfy a parser; the real CRF is regenerated in C++ (S-01). |
| `elems_per_tile = 16 * 8 * num_pim_blocks` | `:1049` | Derived, **wrong** | Omits channels x 2 banks (driver: 131072) (S-07). |
| Fallback `8` PIM blocks | `:427` | Policy fallback | Remove (S-09). |
| `_ROLE_IS_READ_ONLY = {x:True, y:True, acc:False, dst:False}` | `:177` | Template / semantics | Encodes "acc starts at 0" without any matcher check; nonzero init is silently dropped **[executed]** (S-08). |
| `_REG_TO_OPD`, `EVEN_BANK/ODD_BANK` for stride 2, `BANK_<r>` otherwise | `:42`, `spmw_codegen.py:394-413` | ISA encoding | OK; non-stride-2 is cost-only (documented). |
| `banks_per_pim ... else 2`, `_bank_stride_per_pim` fallback 1 | `spmw_autoschedule.py:578, 974-977` | Derived with fallback | Fallback paths are silent; prefer error. |
| Cost constants 4/4/1/26/14/2/1/2 cycles | `costs/samsung.py:20-27` | Empirical | No provenance/units/validity record (S-12). |
| `lanes` default 8, `burst_bytes` default 32 | `:57, :111` | Derived defaults | Should come from target handles, not literals. |
| Candidate catalog: 3 bases x knobs {crf,host} x {shared,per_workid} x {False,True} | `spmw_autoschedule.py:617-672`, `spmw_knobs.py:152-193` | Schedule policy / finite catalog | Explicit, but none of it reaches execution (S-01); the `host` residency candidate is gated on the role-`x` operand (S-05). |
| Paper shapes 4096x256/512, 8192x256, seed 0, `*0.1` scaling, expected cycles 4435/8041/8247, 148/116 | `tests/pim/samsung_hbm_pim/gemv/workload.py`, `tests/pim/test_paper_golden.py:52-57` | Benchmark / oracle | Correctly confined to tests; **no production code reads them** (grep of `allo/` for `4435|8041|8247|gemv_m4096` is empty) **[read]**. |
| `--native-rebaseline`, `--faithful`, scratch rows 1000/2000 | `pim_driver.cc`, `PIMKernel.cpp:591-592` | Simulator harness policy | Synthetic "priced not accumulated" strobes make batched cycles for non-canonical CRFs artificial (S-01). |
| TEMPORAL contract: exactly 5 words, `depth <= 4096`, one rank, audit prefix required | `TemporalReduce.inc:11-26` | Template (ordered conductor) | Only reachable outside the package (S-03). |

Benchmark-identity / filename / seed dependence in production code: **none found** **[read]**.
The reachable production path is shape-general within the padded domain, but its decisions
(placement, residency, CRF issue, mapping) are not load-bearing at execution.

---

## 5. Open questions

1. Does the virtual `estimate()` for sub-tile shapes (K=128, M=2048) diverge from the
   padded simulator cycles, and by how much? (Not measured; needed for S-06 disclosure.)
2. Is there any intention to keep the batched `run_batched` route public? It is untested,
   returns no outputs, and its cycles depend on the synthetic scratch-row strobes.
3. Which upstream SAIT PIMSimulator commit do the installed sources derive from? The
   installed tree has no usable git metadata; the overlay must be re-derived against a pinned
   upstream before S-04 can close.
4. Should the enumerator's `host` residency lever be redefined around the structurally
   identified broadcast operand (S-05), and should `grf_staged` (MAC `GRF_A*GRF_A`) be
   declared unrealizable on this conductor rather than silently mapped to the canonical run?
5. (Resolved by probe_g: it does not; `acc: f16 = 0.5` silently loses the 0.5, S-08.) Should
   a nonzero initializer be rejected or lowered as a host epilogue add?
6. The target declares 16 pseudo-channels while the simulator runs 64; the paper and cost
   model use 16. Is the 4096-row fabric tile (which assumes 64) the intended geometry for the
   released target description?
