# SPEC-021 — Samsung PIMSimulator cycle model + faithful run path

Status: ruled. Verdict **(a)**: `PIMKernel::getCycle()` is fixed by
`--output-dim`/`--input-dim` and does NOT respond to the uploaded cmd
stream. This spec gates task 025 (faithful run path) and is its
precondition read.

Full evidence (quoted C++ source) lives in the cycle ruling report:
`dev/06072026-samsung-gemv-peak/work/reports/arch-cycle-model-determination.md`.
This file is the in-repo, committable form the coder reads.

## 1. The finding (why the cmd stream is invisible to cycles)

- `cycle_` advances only in `runPIM()` (`PIMKernel.cpp:21-27`): one tick
  per `mem_->update()` while the transaction queue is non-empty. Cycles =
  a function of the issued transaction queue, nothing else.
- The `--cmds` stream reaches the kernel via `executeGemvWithCmds`, whose
  ONLY consumer of `cmds` is `programCrf(cmds)` (`PIMKernel.cpp:478`).
- `programCrf` (`PIMKernel.cpp:220-238`) uploads **at most 4** PROGRAM_CRF
  bursts (`for i<4`, `if i*8 >= cmds.size() break`). A 512-record stream
  and a 6-record stream both upload ≤32 CRF entries; stream length is
  invisible.
- The work loop (`PIMKernel.cpp:452-453, 480-508`) and per-tile MAC volume
  (`computeGemv`, `PIMKernel.cpp:541-543`) are computed from
  `w_data->bShape` (= the dims), never from `cmds`. Author confirms:
  comment at `PIMKernel.cpp:444-448` ("Loop / tile structure is
  preserved... independent of CRF contents").

Consequence: metric (A) `tenon_cycles < native_cycles` is unmovable by any
enumerator/cost/codegen change until a faithful run path exists.

## 2. Faithful-execution run path (task 025 builds this)

Chosen: **Option B** — keep `computeGemv` address generation (numerics +
readback column math untouched) but make the outer loop trip counts and
per-tile transaction multiplicity a function of the emitted stream
(`stream_records`, `stream_macs_per_tile`) derived from `compiled.cmds`,
then count cycles off the unmodified `runPIM()`. See report §5 for the
full rationale and the rejected Option A (full CRF interpreter).

### Honest vs flattering line (mechanically checkable)
- FAITHFUL (required): make issued PIM transactions a function of the
  emitted stream, priced by the SAME accounting for native and Tenon. A
  redundant stream genuinely costs more; an optimized one genuinely costs
  less. A redundant *native* stream would also cost more under the change.
- FLATTERING (forbidden): lowering Tenon's cycles without lowering its
  issued work; deleting native transactions from the shared kernel;
  pricing native and Tenon with different transaction models.

## 3. Shared-file boundary (blast radius)

- REFERENCE-ONLY, byte-for-byte unchanged: `PIMKernel.cpp` functions
  `runPIM`, `executeGemv`, `executeGemvWithCmds`, `computeGemv`,
  `programCrf`. The coder MAY call them; MUST NOT edit their bodies.
- ALLOWED new code: a separate driver symbol/entry (e.g. a `--faithful`
  flag or new `executeGemvFaithful`) under
  `experiments/simulators/PIMSimulator/src/` — "new harness/driver code,"
  permitted by the task.
- ALLOWED Tenon-side: `experiments/allo/allo/spmw_codegen.py`
  (`_run_samsung`, `_run_samsung_one`) to select the flag and derive
  stream counts from `compiled.cmds`.
- FORBIDDEN: any change to the native (`--op GEMV`, no faithful flag)
  cycle output. Task 015 re-measures the unmodified `executeGemv`.

## 4. Anti-hardcoding constraint

`stream_records` / `stream_macs_per_tile` and trip counts come from the
cmd stream + `target` geometry, never from `M`/`K` literals. The faithful
path is the measurement instrument; it must price whatever the levers
emit, which is precisely why it cannot bake in a shape.

## 5. Open question deferred to the levers (not blocking 025)

The exact PIMCmd-stream -> `stream_macs_per_tile` map for the JUMP-folded
(`is_auto=1`) vs unrolled case, and the MOV-vs-host-broadcast accounting,
belong in lever specs 030/040/050 (they ARE levers 1/2). 025 may ship the
conservative rule "one issued transaction per CRF instruction after JUMP
expansion; host-loaded operands cost a host transaction, not a CRF MOV,"
which already makes all three levers move the count correctly. See report
§6.

## 6. Acceptance gate for 025

Two cmd streams at the same `(M,K)` yield two DIFFERENT `getCycle()`
values (optimized lower); GEMV stays fp16-correct (max-err ≤ 0.0156); no
shape literal in any decision path. Promote the
`test_samsung_placement_changes_cycles.py` assertion from "cycles need
not change" / cmds-differ to a strict `cycles_a != cycles_b` once the
faithful path lands (this discharges the SPEC-005 §6 deferral).
