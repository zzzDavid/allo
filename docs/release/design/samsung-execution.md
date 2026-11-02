# Samsung HBM-PIM: selected-program execution contract

Status: design, written before the S-01/S-05/S-06/S-07/S-09/S-10 repair
(audit `docs/release/audit/samsung.md`). Section 6 is filled in after the
executed runs.

## 1. Problem

Before this repair the Samsung compiler ranked placements over six levers,
emitted a matching `PIMCmd` stream, and then ran something else: the REDUCE
route wrote a fixed five-line CRF, the conductor regenerated the vendor GEMV
microcode (`PIMCmdGen::getPIMCmds(KernelType::GEMV, ...)`) and replayed the
vendor `computeGemv` cadence against the vendor `preloadGemv` weight layout.
Every placement produced the same argv, CRF, cycles and outputs (audit S-01).
The emitted stream was also not a valid program for the hardware it named:

* JUMP trip counts were `K // grf_a.lanes - 1`, i.e. `K / 8`. `grf_a.lanes`
  is the GRF *entry* count (8). One auto-mode MAC execution consumes 8
  column commands = 8 bursts x 16 FP16 SIMD lanes = 128 reduction elements
  for one GRF_B row; the emitted count ignored the 16 SIMD lanes and the 8
  GRF_B rows.
* Operand roles followed source order (`x` = first factor). For the paper
  form `W[i,k] * x[k]` the weight was placed in GRF_A and the vector in the
  bank, the inverse of the executed hardware program (S-05).
* With CRF-filled GRF_A (`grf_residency=crf`), one FILL preceded both fiber
  loops although each reduction tile needs its own GRF_A contents.

## 2. Hardware facts the contract relies on

From `PIMRank::doPIM`/`doPIMBlock`/`readOpd` (PIMSimulator) and the target
(`allo/pim/targets.py::build_samsung_target`):

| Fact | Value | Source |
|---|---|---|
| PIM blocks per pseudo-channel | 8 (`pim` mapping), each owning banks `2p`, `2p+1` | target, `PIMRank` `banks[pb*2 (+1)]` |
| CRF | 32 words, one per pseudo-channel, executed in lockstep by its 8 blocks | target `crf(entries=32)`, `PIMRank::doPIM` loops over all blocks |
| GRF_A / GRF_B | 8 entries x 256 bit = 8 x 16 FP16 lanes | target `reg(8, 256)` |
| Auto-mode (`is_auto`, FILL) | 8 consecutive column commands per instruction execution; GRF_A index `col & 7`, MAC GRF_B index `(row&1)<<2 | (col>>3)&3` | `doPIM` repeat logic, `getGrfIdx`, `getGrfIdxHigh` |
| JUMP | resolved inline before the next trigger; one jump register (`lastJumpIdx_`), so loops cannot nest | `doPIM` |
| NOP + WRITE | writes GRF_A (bank 0) / GRF_B (bank 1) entry `col & 7` to the bank | `doPIMBlock` |
| Mode entry `HAB -> HAB_PIM` | resets PC; zeroes GRF_B when the control burst says so | `controlPIM` |
| Column rows | 32 burst columns per DRAM row | `NUM_COLS / BL` |
| Datapath | FP16 per lane (`PIMBlock::mac`) | |
| Simulated system | 64 pseudo-channels (`system_hbm_64ch.ini`, `pim_driver` `make_kernel`) | simulator configuration, reported by `pim_driver --describe` |

The target declares one 16-pseudo-channel device. The simulator
configuration models 64 pseudo-channels. The compiler never assumes 64: the
runtime asks the conductor for its geometry and checks that the per-channel
facts (8 blocks, 8+8 GRF entries, 16 lanes, 32 CRF words) equal the target's.

## 3. Levers: intended semantics, stream effect, cost-graph effect, conductor

"Stream" is the per-pass CRF program the codegen emits; "cost graph" is
what `spmw_plan.build_execution_graph` + `allo/pim/costs/samsung.py` see
(read from the code before the repair, confirmed by the ranking dump in
probe A: only two distinct scores, 226 and 275).

### 3.1 `bank_row` / `dual_fiber` (base candidates; fiber count)

* Semantics: where the bank-resident operand (the matrix) lives. `bank_row`
  puts all K tiles of a PIM block's rows in its even bank (one fiber);
  `dual_fiber` alternates K tiles between the even and odd bank of the block
  (tile `t` on fiber `t mod 2`). The vendor GEMV layout is `dual_fiber`.
* Stream: per fiber with `T_f` tiles, `MAC GRF_B <- GRF_A * BANK_f (auto)`
  then `JUMP (G*T_f - 1)` where `G` = GRF_B entries (rows per block per
  pass). A fiber with no tiles emits nothing.
* Cost graph (before): MAC rule `ceil(K / (lanes*fibers)) * 4 + fibers`, so
  `dual_fiber` is modeled as two concurrent bank halves (about 2x faster).
  The hardware serializes both fibers' column commands on one channel bus;
  the advantage is bank/row locality, not concurrency.
* Conductor: places each tile in the fiber the plan names, issues the MAC
  column groups to that bank in CRF order.

### 3.2 `grf_staged`

* Semantics: matrix operand staged into GRF_A as well (`MAC GRF_B <- GRF_A *
  GRF_A`, not auto).
* Stream (before): `FILL GRF_A <- EVEN_BANK; MAC GRF_B <- GRF_A*GRF_A;
  JUMP 31`. A non-auto MAC reads fixed indices (`src0Idx_ = src1Idx_ =
  dstIdx_ = 0`), so every iteration computes `GRF_B[0] += GRF_A[0]^2`: the
  JUMP does not walk K and the stream is not a GEMV. A correct GRF-staged
  dot product needs a distinct instruction per (row, entry) pair, i.e. at
  least 64 MAC words per K tile, which exceeds the 32-word CRF, and FILL
  into GRF_A would overwrite the broadcast vector.
* Cost graph: identical to `bank_row` (the MAC rule does not see the
  handle), so it only ever tied.
* Conductor: cannot execute it. **Contract: rejected at materialization**
  (`InfeasibleSchedule`, recorded in `KernelRanking.rejections`).

### 3.3 `crf_issue`: `shared` / `per_workid`

* Semantics: `shared` uploads one body and fires it for every work item;
  `per_workid` concatenates one body per work item into the CRF stream.
* Stream: `shared` = one body; `per_workid` = `n_workids` bodies (768 words
  for the paper grid).
* Cost graph: no rule reads `crf_issue` (tie).
* Conductor: the 8 blocks of a pseudo-channel execute one CRF in lockstep.
  A concatenation of per-work-item bodies would be executed in full by every
  block (each block would perform `n_workids` times its work), and it
  exceeds 32 words. **Contract: `per_workid` is realizable only for a
  one-work-item group (where it equals `shared`); otherwise rejected at
  materialization.**

### 3.4 `grf_residency` (`crf` / `host`) for the broadcast vector

* Semantics: how GRF_A receives the vector. `host`: the host writes GRF_A
  through `PIM_REG_RA` columns `0x08..0x0f` (HAB broadcast) before each K
  tile. `crf`: the vector is scattered into every block's even bank and the
  CRF fills GRF_A with `FILL GRF_A <- EVEN_BANK` before each K tile.
* Stream: `host` emits no FILL. `crf` emits, per K tile, `FILL GRF_A <-
  EVEN_BANK; MAC ...; JUMP (G-1)` (loops cannot nest, so tiles are unrolled;
  `3T + 2` words, rejected when above 32).
* Cost graph (before): not read by any rule (tie broken by position). After
  the repair the Samsung rules price the two fills differently (Section 5).
* Conductor: `host` performs the broadcast writes; `crf` preloads the vector
  into the even banks and issues the FILL column reads. The vector is the
  host-eligible operand only after S-05 (before, the matrix was).

### 3.5 `stage_resident` (`wresident`)

* Semantics: for several reduction columns (GEMM `N > 1`, or `run_batched`
  with `B` vectors), preload the matrix once (resident) or re-stage it for
  every column (restage).
* Stream: none (host sequencing).
* Cost graph: no Samsung rule reads it (tie).
* Conductor: resident = one preload + one CRF upload, `N` passes;
  restage = per column: preload, CRF upload, pass. For `N = 1` both are the
  same program; the conductor runs the same invocation.

### 3.6 `residency` (`restage` / `resident`), `tile`, `double_buffer`

* `residency` crosses work items (single kernel) or kernels (multi-kernel).
  On one kernel, the vector is HAB-broadcast once per K tile per pass to all
  blocks simultaneously, so restage and resident denote the same physical
  program; the knob has no cost rule either (tie). Across kernels the
  conductor keeps nothing between invocations, so a multi-kernel program
  runs only the residency its host program spells out with transfers
  (section 9); a value left device-resident across launches is rejected.
* `tile` and `double_buffer` are 1x fans on the Samsung GEMV (no legal
  non-identity tiling, no buffer pair). The Samsung lowering does not
  realize either, so a placement carrying a non-identity value is rejected
  at materialization instead of being silently ignored.

### 3.7 Host moves (`host_xfer.*`)

* Semantics: explicit SCATTER (matrix to banks), BCAST_GRF_A (vector),
  GATHER (output), PROGRAM_CRF declarations.
* Cost graph: priced by bytes (host rule) when supplied.
* Conductor: performs exactly these transfers; the runtime checks that the
  recorded buffer roles name the structurally classified matrix / vector /
  output parameters and raises on a mismatch (it used to warn).

## 4. Execution contract (as implemented)

Code: `allo/spmw_samsung.py` (classification, `SamsungCtx.emit_program`,
plans, runtime), `allo/spmw_autoschedule.py::_samsung_enumerate`,
`allo/spmw_knobs.py` (Samsung knob entries), `allo/spmw_codegen.py::
source_backend_binding`, conductor `tests/TenonConductor.{h,inc}` +
`pim_driver.cc` (overlay `experiments/simulators/setup/PIMSimulator/`).

### 4.1 Contractions (REDUCE)

1. **Structural operand classification (S-05),
   `samsung_contraction`.** Accepted nests: `[k]` (one output row per work
   item, constant store row), `[row, k]` (GEMV), `[row, column, k]` (GEMM);
   static `[0, n)` loops; the innermost loop is the only reduction axis; the
   accumulator is initialized to literal 0; all operands FP16. Coordinates
   are read from `memref.load` indices or from `affine.load` map results
   with the map operands substituted. Each input must read the reduction
   loop unshifted in exactly one coordinate. The **vector** is the input
   with no other coordinate (GEMM: only the column loop); the **matrix** is
   the input with exactly one other coordinate (the row). Its row
   coordinate must be the row loop plus a constant equal to the store row
   (checked when affine), a constant, or an SSA value the matcher keeps
   only by name (then the store row is trusted, see 6.4). Anything else,
   e.g. `a[k] * b[k]` or the same buffer twice, is rejected with
   `SamsungUnsupported` naming the coordinates. The matcher's own
   `axis_roles.matrix` (first operand when neither index set contains the
   other) is not used. `x[k] * W[i, k]` and `W[i, k] * x[k]` produce
   identical placements, CRF and plan, and bind `A = W`, `B = x`.
2. **Coverage (S-10).** The work items' store row ranges
   (`output_sinks`) must partition the matrix rows exactly; a gap, overlap
   or partial cover is a compile error naming the rows. The work grid is
   a logical partition; physically, rows are packed into slots
   `((pass * channels + channel) * blocks + block) * grf_b + entry`. With
   the program's work-grid owner map rows occupy slots in row order (the
   vendor packing). A movement tie carried on the placement places rows
   by its owner map instead (section 8); all blocks run in lockstep, so
   the packing does not change the command stream.
3. **Per-pass CRF** (`SamsungCtx._emit_reduce`), built from the target's
   own op/move `emit` callbacks: with `T = ceil(K / (grf_a.slots *
   grf_a.width/16))` tiles and tile `t` on fiber `t mod n_fibers`, for each
   fiber with `T_f > 0` tiles: host fill: `MAC GRF_B <- GRF_A * BANK_f
   (auto); JUMP (8 * T_f - 1)`; CRF fill: per tile `FILL GRF_A <-
   EVEN_BANK; MAC ...; JUMP 7` (no nested loops on this ISA); then the
   ST_B drain `NOP 7` and `EXIT`. More than 32 words, a matrix in a GRF
   (`grf_staged`), `per_workid` with more than one work item, a
   non-identity `tile`, a `double_buffer` pair, or a fiber class without a
   PIMSimulator encoding raise `InfeasibleSchedule` (the ranking falls to
   the next candidate; a forced layout fails to compile).
4. **Program record** (`ctx.programs`, `Compiled.layout_ctx.programs`):
   word range, per-word bindings (`matrix`/`vector`/`drain`, bank fiber),
   `T`, `tile_fiber`, rows per block, GRF_A fill, `stage_resident`, the
   contraction, and the lever values.
5. **Plan** (`samsung_reduce_plan`), computed at run time from the record
   and the conductor's `--describe` geometry (whose per-channel facts must
   equal the target's): passes `J = ceil(M / (channels * blocks * 8))`,
   `T`, `tile_fiber`, rows per block, `vector_from_bank`, columns `N`,
   `weight_resident`, and the bind lines. The matrix file is written in slot
   order, `[J * channels * blocks * 8, T * 128]`; the vectors `[N, T * 128]`
   (a transposed matrix `A[k, i]` or `B[j, k]` is transposed on the host).
6. **Conductor** (`PIMKernel::tenonRunReduce`): uploads the CRF words
   verbatim (decoded through `PIMCmd::toInt/fromInt`), statically walks
   them with `tenonWalkCrf` (an exact model of `PIMRank::doPIM`), and checks
   each executed word against its binding and encoding; it then issues,
   per pass and per instruction execution, the column commands that
   execution consumes: MAC groups at `((j * T_f + s) * 8 + g) * 8` of the
   fiber's bank (`s` = tile index within the fiber, `g` = row group), the
   HAB broadcast of tile `t` before group 0 of each tile (host fill), the
   FILL reads of the tile the next MAC group needs (CRF fill), and the
   drain writes of GRF_B to the odd bank. The walk must perform exactly
   `8 * T_f` MAC groups per fiber and one drain per pass, else the run
   fails naming the word. Matrix staging follows the vendor `preloadGemv`
   order, so the canonical plan issues the vendor command sequence.
   `pimcmdgen_calls` in the receipt is 0 (the vendor generator is
   instrumented and never called on this route). Resident staging preloads
   the matrix and uploads the CRF once for all `N` columns; restaged
   staging repeats both per column.
7. **Readback**: bursts in (column, pass, block, entry) order, 16-lane
   host tree reduction (the regrouping the paper permits), slot to row.
   An all-zero result is returned as such; an absent or short `out.bin`
   raises.
8. **`RunResult.extra` (S-06)**: `logical_shape` / `padded_shape`
   (`rows` padded to whole passes of the 4096-slot fabric, `reduction`
   to whole 128-element tiles, `columns`), `padding_fraction` (share of
   issued multiply-accumulates that are padding), `timing_scope` (which
   cycles `RunResult.cycles` is), `cycle_phases`, `plan`, `crf`,
   `conductor` receipt, `simulator_geometry`, `levers`.

### 4.2 Elementwise (S-07)

`ADD`, `MUL`, `MAD`, `RELU` are reachable through `allo.compile` in this
domain: one work item (`mapping=[1]`) whose single loop walks every element
of 1-D FP16 operands of one extent, every input and the store indexed by
that loop variable (`samsung_elementwise`). Partitioned work grids index
elements through SSA expressions the matcher does not retain, so
elementwise-ness cannot be proven; they are rejected with that reason. The
single placement (`all_bank`) streams both bank fibers: for fiber `f`, the
register drained to bank `f` by the target's ST move (`ST_A`: GRF_A to
even, `ST_B`: GRF_B to odd) is filled by its LD move, the op runs auto on
`BANK_f`, and the ST move drains it. `MAD` fills the addend into the other
register through that register's LD move (the other fiber), so the addend
is stored with its fibers exchanged (`input_swap_parity`). `RELU` is the
target's non-auto `MOV` with the relu flag, one word per GRF entry. One CRF
pass per 131,072-element tile (the compiler does not know the simulated
channel count, so programs are per pass); operands are padded to whole
tiles. Results are bit-exact against NumPy FP16.

## 5. Cost model scope

`allo/pim/costs/samsung.py` prices one work item's events on the declared
16-channel device (unit: PIMSimulator memory-clock cycles of the modeled
activities, excluding host preload, CRF upload and readback unless explicit
host moves are supplied). The simulator reports preload + execution +
readback on the 64-channel configuration with padded M and K. The two are
therefore not comparable in absolute value; only the ranking of candidates
is compared, with tolerances stated before the confirmation runs
(Section 6).

## 6. Results

### 6.1 Expectations stated before the confirmation runs

Written after the conductor harness runs (hand-built plans driven directly,
`/tmp/samsung-fix/conductor/harness.py`), before the confirmation runs
through `allo.compile` with forced placements. The harness measured, for
M=4096, K=256: dual_fiber/host 4,435, bank_row/host 4,389, dual_fiber/crf
4,560, bank_row/crf 4,589 cycles, so these expectations are informed by
those exploratory numbers.

Candidates (all `crf_shared`): P0 = dual_fiber + host fill (canonical),
P1 = bank_row + host fill, P2 = dual_fiber + crf fill, P3 = bank_row + crf
fill; P4 = grf_staged, P5 = dual_fiber + per_workid; P6/P7 = GEMM N=4 with
resident / restaged matrix.

* Execution: P0 reproduces 4,435 / 8,041 / 8,247 with the vendor command
  sequence; P1-P3 and P6/P7 produce correct outputs (within 1e-2 of the FP32
  reference, the test tolerance) and cycles that differ from P0; P4 and P5
  are rejected at materialization with their reason recorded.
* Ranking metric: a pair of candidates *agrees* when the cost model and the
  simulator order it the same way. Simulated differences below 0.5% are
  treated as ties that neither confirm nor refute. Absolute cycles are not
  compared (scope mismatch, Section 5: preload and readback are about 74% of
  the simulated total and absent from the cost graph).
* Expected outcome: residency pairs (P0,P2) and (P1,P3) agree (host fill
  cheaper in both); fiber pairs (P0,P1) and (P2,P3) **disagree** (the cost
  model's concurrent-fiber MAC rule favors dual_fiber by about 23%; the
  simulator serializes both fibers and favors bank_row by about 1%);
  (P0,P3) agrees and (P1,P2) disagrees. So 3 of 6 pairs agree on K=256 and
  the same pattern on K=512. P6 < P7 in both models (resident saves N-1
  matrix stagings).

### 6.2 Executed runs (installed `pim_driver`, `allo.compile`)

Inputs: seed 0, `N(0, 1) * 0.1`, FP16. "err" is the maximum absolute error
against the FP32 product. Cycles are PIMSimulator memory-clock cycles of the
padded program (preload + exec + readback). "cost" is
`CompiledCallable.estimate()` of the same compiled placement.

| Shape | Placement | CRF words | cost | sim cycles | preload / exec / readback | err |
|---|---|---|---|---|---|---|
| 4096x256 | P0 dual_fiber, host fill (default) | 6 | 194 | **4,435** | 3116 / 1138 / 181 | 4.5e-4 |
| 4096x256 | P1 bank_row, host fill | 4 | 253 | 4,389 | 3116 / 1092 / 181 | 4.5e-4 |
| 4096x256 | P2 dual_fiber, CRF fill | 8 | 226 | 4,560 | 3209 / 1170 / 181 | 4.5e-4 |
| 4096x256 | P3 bank_row, CRF fill | 8 | 275 | 4,589 | 3230 / 1178 / 181 | 4.5e-4 |
| 4096x256 | P4 grf_staged | rejected at materialization | | | | |
| 4096x256 | P5 dual_fiber, per_workid | rejected at materialization | | | | |
| 4096x512 | P0 (default) | 6 | 254 | **8,041** | 5628 / 2232 / 181 | 1.0e-3 |
| 4096x512 | P1 | 4 | 381 | 7,947 | 5628 / 2138 / 181 | 8.2e-4 |
| 4096x512 | P2 | 14 | 276 | 8,410 | 6003 / 2226 / 181 | 1.0e-3 |
| 4096x512 | P3 | 14 | 403 | 8,458 | 6008 / 2269 / 181 | 8.2e-4 |
| 8192x256 | P0 (default) | 6 | 194 | **8,247** | 5628 / 2310 / 309 | 7.3e-4 |
| 8192x256 | P1 | 4 | 253 | 8,153 | 5628 / 2216 / 309 | 7.3e-4 |
| 8192x256 | P2 | 8 | 226 | 8,359 | 5721 / 2329 / 309 | 7.3e-4 |
| 8192x256 | P3 | 8 | 275 | 8,397 | 5742 / 2346 / 309 | 7.3e-4 |
| GEMM 4096x256, N=4 | P6 resident matrix (default) | 6 | 194 | 7,604 | 3116 / 3923 / 565 | 4.7e-4 |
| GEMM 4096x256, N=4 | P7 restaged matrix | 6 | 12,498 | 17,970 | 12481 / 4552 / 937 | 4.7e-4 |

What each lever now does on the simulator, read from the conductor
receipts: `bank_row` issues all 16 (K=256) MAC groups on the even bank
(`mac_groups_odd=0`) and drains to odd-bank row 0; `dual_fiber` splits them
8/8, the vendor order. Host fill issues 8 HAB GRF_A writes per channel per
tile (`host_vector_writes`); CRF fill replicates the vector into every
block's even bank (+93 preload cycles at K=256) and issues FILL column reads
(`fills=2`). `stage_resident` changes `matrix_preloads` (1 vs N) and CRF
uploads. `per_workid` and `grf_staged` never run. `residency`
(restage/resident), `tile` and `double_buffer` select the same program on a
single kernel.

The canonical P0 program is byte-identical to the vendor GEMV CRF and its
partial-sum grid is byte-identical to the vendor `executeGemv` run
(`pim_driver --op GEMV`, which also reports 4,435 / 8,041 / 8,247).
Other programs executed through `allo.compile`: `x[k] * W[i, k]` (identical
CRF and 4,435 cycles), `A[k, i] * x[k]` (host transpose, 4,435), GEMM with
`B[k, j] * A[i, k]` operand order (resident, correct), 2048x100 (2,813
cycles, `padding_fraction` 0.609), and elementwise ADD/MUL 1,588, RELU 1,201,
MAD 1,910 cycles for one 131,072-element tile (bit-exact; 100,000 elements
pad to one tile, `padding_fraction` 0.237).

### 6.3 Cost model versus simulator

Pairs per GEMV shape (Section 6.1 rule; sim differences below 0.5% are
ties):

| Pair | 4096x256 | 4096x512 | 8192x256 |
|---|---|---|---|
| P0-P1 (fiber, host fill) | disagree (sim -1.0%) | disagree (-1.2%) | disagree (-1.1%) |
| P0-P2 (fill, dual) | agree | agree | agree |
| P0-P3 | agree | agree | agree |
| P1-P2 | disagree | disagree | disagree |
| P1-P3 (fill, bank_row) | agree | agree | agree |
| P2-P3 (fiber, CRF fill) | agree (+0.6%) | agree (+0.6%) | tie (+0.45%) |

GEMM P6-P7 agrees. Against the prediction: 4/6, 4/6 and 3/6 (+1 tie)
pairs agree instead of 3/6 each. The miss is P2-P3: with the vector
replicated into the even bank, `bank_row` puts the vector region and the
whole matrix in the same bank, so its row-buffer conflicts outweigh the
fiber effect that favors `bank_row` under host fill. The residency term is
right in sign everywhere. The fiber term is wrong in sign under host fill:
the MAC rule models the two fibers as concurrent, the conductor (like the
hardware) serializes their column commands on the channel bus, and the
remaining difference comes from DRAM row state. As a result the default
selection (P0, the vendor program) was not the fastest executed program: P1
is 1.0-1.2% faster. Section 7 corrects the MAC rule (superseding this
paragraph); the paper rows now pin P0 explicitly.

### 6.4 Artifacts

* `pim_driver` (installed, `experiments/simulators/PIMSimulator/pim_driver`):
  sha256 `0ae6a3d9bc0aa1ed06e52eb74d81d7c2b98c0fe344e1a5591add5b05b50c2df2`,
  built 2026-11-02 with system g++ 9.4.0 / SCons from upstream
  `3703d1f19c8f027360cc33a3243eb271e3bb6898` plus the overlay. Previous
  binary kept as `pim_driver.bak-20261101` (sha256 `3c5039f1...188a`).
* Before the rebuild, the installed `src/` equaled the `dev/` conductor
  sources (`pim_driver.cc`, `PIMKernel.{cpp,h}`, `TemporalReduce.inc`,
  `KernelAddrGen.h`, `Burst.h` identical; installed `SamsungSlicePlan.*` were
  shims `#include`-ing the `dev/` copies) on top of upstream `3703d1f`
  with `PIMCmdGen.{h,cpp}` also modified. After: installed `src/` = upstream
  + `setup/PIMSimulator` overlay, byte-for-byte (checked); the `dev/`
  conductor `src/` mirrors every changed file; the unused slice-plan files
  moved to `dev/.../pimsim-conductor/retired/`.
* The overlay applied to a fresh upstream clone builds and reproduces every
  cycle count above (binary hashes differ only by embedded build paths).

### 6.5 Open

* **Ordered fallback (S-03).** `TemporalReduce.inc` and `--op-kind
  TEMPORAL_REDUCE` (5-word CRF `MOV GRF_B <- EVEN_BANK; MAC GRF_B <- GRF_A *
  EVEN_BANK; JUMP K-1; NOP; EXIT`, one GRF slot, each lane owns one output
  row, K issued in source order, initial value preserved, no host lane
  reduction) are in the overlay and build, but no `allo.compile` route
  selects them. Integrating needs an explicit reduction-order policy on the
  workload (strict vs reassociative, C-09), a placement family whose CRF is
  that program, a plan for the lane-per-row layout, and a cost rule.
* Cost model fiber term (6.3): corrected in section 7.
* An opaque matrix row coordinate (`W[row0 + i, k]` lowered to
  `memref.load` with an SSA row) cannot be checked against the store row;
  the conductor computes `W[row]` for the store row. The matcher would need
  to retain the row expression (as it does for `affine.load`).
* Multi-kernel Samsung traces compile but do not run (one REDUCE or
  ELTWISE program per artifact). Closed in section 9: they run their host
  program's launch schedule, one invocation per launch.
* Elementwise kernels require `mapping=[1]`, which triggers the generic
  `_check_work_grid` warning (owned elsewhere).
* `experiments/simulators/setup/apply.sh` now calls
  `setup/PIMSimulator/apply.sh` (section 7). The table in
  `experiments/simulators/README.md` section 2 still lists the old two-file
  copy.

## 7. Corrected MAC rule (2026-11-02, coordinator decision)

The paper's 1.00x Samsung rows are evidence to explain, not a value the
compiler must reproduce, and G5 requires a relevant cost change to reach
selection. The MAC rule is therefore corrected to what the simulator does.

### 7.1 Rule

The two bank fibers of a PIM unit share one pseudo-channel column-command
bus, so the MAC column commands of both fibers serialize: the MAC term is
`ceil(K / lanes) * MAC_CYCLES` whatever the fiber count (previously divided
by the fiber count). Each fiber keeps its JUMP, and each reduction tile the
placement puts on a fiber other than the first pays a measured row-open
overhead `FIBER_TILE_CYCLES = 47`: the execution phase of dual_fiber minus
bank_row (host fill) is 46, 94 and 94 cycles at 4096x256 (1 odd-fiber tile
per pass, 1 pass), 4096x512 (2 tiles, 1 pass) and 8192x256 (1 tile, 2
passes), i.e. 46-47 cycles per odd-fiber tile-pass. The residency term is
unchanged.

### 7.2 Expected outcome, stated before computing the new estimates

Same 4 x 3 candidate table and agreement rule as 6.1 (simulated cycles are
unchanged: the programs do not depend on the cost model).

* Fiber pairs under host fill (P0-P1) now agree in all three shapes.
* P0-P2 (fill choice), P1-P3 and P1-P2 agree.
* P0-P3 likely **disagrees**: the model's residency term (16 per work item,
  one fill) is smaller than its new fiber term (48+), so it ranks
  bank_row + CRF fill ahead of dual_fiber + host fill, whereas the simulator
  charges CRF fill about 125-370 cycles (vector replication into every bank
  is preload, outside the graph).
* P2-P3 likely disagrees at 4096x256 and 4096x512 (model: bank_row cheaper;
  simulator: bank_row + CRF fill 0.6% slower) and is a tie at 8192x256.
* Expected: 4 of 6 pairs agree at 4096x256 and 4096x512, 4 agree + 1 tie
  at 8192x256; GEMM resident/restage still agrees.
* The default selection becomes bank_row + host fill: 4,389 / 7,947 / 8,153
  cycles instead of the vendor program's 4,435 / 8,041 / 8,247.

### 7.3 Results with the corrected rule

Simulated cycles are those of 6.2 (same programs); estimates change:

| Placement | 4096x256 cost / sim | 4096x512 cost / sim | 8192x256 cost / sim |
|---|---|---|---|
| P0 dual_fiber, host fill (vendor) | 301 / 4,435 | 476 / 8,041 | 301 / 8,247 |
| P1 bank_row, host fill | **253 / 4,389** | **381 / 7,947** | **253 / 8,153** |
| P2 dual_fiber, CRF fill | 323 / 4,560 | 498 / 8,410 | 323 / 8,359 |
| P3 bank_row, CRF fill | 275 / 4,589 | 403 / 8,458 | 275 / 8,397 |
| P4 grf_staged, P5 per_workid | rejected | rejected | rejected |

GEMM 4096x256 N=4 (dual_fiber forced): resident 301 / 7,604, restaged
12,543 / 17,970.

Pairwise agreement (6.1 rule): 4096x256 4/6 (P0-P3 and P2-P3 disagree),
4096x512 4/6 (same pairs), 8192x256 4/6 + P2-P3 tie (P0-P3 disagrees);
GEMM agrees. This is exactly the outcome stated in 7.2. All fiber pairs
under host fill now agree; the remaining disagreements are the residency
term, whose dominant simulated cost (replicating the vector into every
bank, 93-375 preload cycles) is preload and outside the graph's scope.

Default (cost-selected) program: P1, bank_row + host fill, in all three
shapes, with correct outputs (max error 4.1e-4 / 8.0e-4 / 4.1e-4 against
FP32): **4,389 / 7,947 / 8,153 cycles, 1.0% / 1.2% / 1.1% fewer than the
vendor program's 4,435 / 8,041 / 8,247**. The paper rows (tests
`test_paper_golden.py` Samsung rows, `samsung_hbm_pim/gemv`) now compile
the vendor-equivalent placement explicitly with
`allo.spmw_samsung.samsung_vendor_gemv_layout(target, trace)` and assert the
paper cycles for it; they also run the default compile, check its outputs
with the same tolerance, and record its cycles (`record_property`,
`SAMSUNG_GOLDEN ...` line) without comparing them with the paper value.

### 7.4 Note for docs/release/CLAIMS.md row E1

Samsung E1 (Tab. eval-samsung): the vendor-equivalent placement (dual_fiber
bank fibers, host-broadcast GRF_A, shared CRF), selected explicitly with
`samsung_vendor_gemv_layout`, compiles to the vendor GEMV CRF word for
word; the conductor uploads that emitted CRF (no PIMCmdGen call) and
reproduces 4,435 / 8,041 / 8,247 PIMSimulator cycles (padded program:
preload, execution and readback) with outputs within 1e-2 of FP32. The
default cost-selected compile no longer picks that program: after the
Samsung MAC rule was corrected to serialize both bank fibers on the
channel bus (calibrated 2026-11-02 on PIMSimulator 3703d1f + overlay), it
selects bank_row (all reduction tiles in the even bank) with host GRF_A
fill, which runs in 4,389 / 7,947 / 8,153 cycles, 1.0-1.2% fewer than the
vendor program, with correct outputs. The paper's 1.00x rows therefore
describe the vendor-equivalent placement, not the default selection; the
default beats the vendor path by about 1% on these shapes.

## 8. Owner-map row placement (movement ties, 2026-11-02)

Claim row 16 (ties in movement survive to the final cost ranking) needs a
backend that executes a tie other than the work-grid map. Samsung REDUCE
and the ordered program now do.

### 8.1 Realizable family

The movement stage (`allo/spmw_layout_problem.py`, `layout-search.md`
section 8) states a contraction's layout as a one-to-one GF(2) map from
work bits (owner levels `pseudo_channel`, `pim`, then local bits) to
iteration bits (`work0`, `work1`, the row loop, the reduction loop). On
Samsung (`ROW_MAP_TARGETS`) a tie is realizable when
(`row_map_unrealizable_reason`):

* the kernel is a contraction (`MAC`) with one accumulator of one
  output-row coordinate (GEMM's two-coordinate output, elementwise kernels:
  only the program map);
* every bit of a variable the accumulator does not depend on (the
  reduction `k`) is local at the innermost owner level, i.e. the owner
  coordinates depend on the row bits only. A tie with an owner bit from
  `k` splits one row's reduction across units; that needs a partial-sum
  combine, which belongs to the reduction policy, and is rejected with
  that reason (all minimum ties of the paper GEMV under `reassociate`);
* the row variables are not masked (every unit then owns the same number
  of rows).

The program's own map is always realizable. Besides the classes the
search returns, the realizable members of a minimum class that group rows
differently are crossed as separate ties: the per-level BLOCK/CYCLIC
distributions of the row index (`row_distributions`; the row bits ordered
by coefficient, a BLOCK level takes the most significant remaining bits, a
CYCLIC level the least significant). For the paper GEMV (`row =
(pid * 8 + uid) * 32 + i`, 12 row bits) under the default strict order:

| Tie | pseudo_channel bits | pim bits | rows of one unit |
|---|---|---|---|
| program = block x block | row[8..11] (`work0`) | row[5..7] (`work1`) | 32 consecutive rows |
| block x cyclic | row[8..11] | row[0..2] | rows `256 p + 8 j + u`, `j < 32` |
| cyclic x block | row[0..3] | row[9..11] | 32 rows `r = c (mod 16)` inside one 512-row block |
| cyclic x cyclic | row[0..3] | row[4..6] | rows `r = u (mod 128)` |

All four have movement 72,704 B (the proven minimum; x is replicated on
every unit whichever row bits name the units), so `tie_count == 4`,
`chosen_label == "minimal"`, each tie crossed with the 4 decision keys
(16 scored candidates, all realized).

### 8.2 Row-to-slot map and plan encoding

`SamsungCtx._row_slots` derives the problem again from the group's
matches, reads the movement part of `Placement.layout`
(`split_movement_layout`, the `mm.` block of the product) and evaluates the
inverse map on every `(work id, row loop value)`: owner unit `u =
pseudo_channel * 8 + pim` (outer level most significant), `R = rows / units`
rows per unit, and the rows of unit `u` occupy slots `u * R .. u * R + R -
1` in ascending row order. The program map yields slot = row, so it is
recorded as `identity` and emits nothing new.

* REDUCE: `samsung_reduce_plan` appends `row_slots s_0 ... s_{S-1}` (the
  slot of every row of the zero-padded matrix, padding rows keep their own
  slots; a permutation of the `S` slots) only when rows move. The conductor
  (`TenonConductor.inc::tenonRunReduce`) validates the permutation, stages
  matrix row `r` at slot `row_slots[r]`, drains each slot's result to its
  row's position, and prints `TENON_ROW_MAP map=identity|permuted
  rows_moved=N digest=<FNV-1a 64 of the table>`; the runtime checks the
  digest against the plan's and records it in `RunResult.extra["conductor"]`
  (`row_map_*` keys) and `extra["row_map"]` (map, digest, physical position
  of sample rows). Addresses and command order depend on slots only, so the
  stream is unchanged.
* TEMPORAL_REDUCE: the ordered conductor (`TemporalReduce.inc`, not
  changed) gives lane slot `s` to input row `s`; the runtime stages row `r`
  at slot `row_slots[r]` (matrix and initial accumulator) and reads it back
  from there (`extra["row_map"]["applied_by"]`).
* Elementwise: a carried owner map other than the program's is rejected at
  materialization (fixed flat channel-bank order).

The installed `pim_driver` was rebuilt from the overlay in a scratch copy
of the pinned clone (`/tmp/rowmap-sim/pimsim`, `scons pim_driver`, about
10 s); stripped sha256 (`objcopy --strip-all
--remove-section=.note.gnu.build-id`) `9780b17c7396...` for both the
scratch and the installed binary. Previous binary kept as
`PIMSimulator/pim_driver.bak-20261102-rowmap` (sha256 `0ae6a3d9bc0a...`).
`pim_driver.cc` and `TenonConductor.h` are unchanged.

### 8.3 Two tied layouts: predicted versus measured

Expected before running: equal cycles. The tied layouts have equal movement
by construction; no Samsung cost rule reads the row map (the MAC, fill and
drain rules count passes, tiles and column commands, which a row
permutation does not change); and in the simulator every PIM block of every
channel executes the lockstep CRF on every slot of every pass, so the
conductor issues the same addresses in the same order and only the data in
each slot differs. Outputs must be correct for both (REDUCE within 1e-2;
the ordered program bit-identical, since each row is still reduced in
source order in FP16).

| Program (M x K = 4096 x 256) | Tie | Cost program | PIMSimulator | Max abs err | Row 1 placed at |
|---|---|---|---|---|---|
| vendor REDUCE (`dual_fiber`, host fill, shared CRF), reassociate | program (identity) | 301 | 4,435 | 4.5e-4 | channel 0, block 0, entry 1 |
| same | cyclic x cyclic (4,088 of 4,096 rows moved) | 301 | 4,435 | 4.5e-4 | channel 4, block 0, entry 0 |
| default strict (TEMPORAL, shared CRF) | program (identity) | 27,598 | 26,807 | 2.2e-3 | channel 0, block 0, lane 1 |
| same | block x cyclic / cyclic x block / cyclic x cyclic | 27,598 | 26,807 | 2.2e-3 (outputs bit-identical) | block 2 / channel 2 / channel 2 |

The cost program's REDUCE estimate (301) excludes preload and readback
(section 6.3); the comparison here is between ties, which the cost program
and the simulator both rank equal. The ranking therefore breaks the tie by
the structural key (`TIE_BREAK_RULE`), which selects the program map on
the paper GEMV, so the default program and its cycles are unchanged.

Tests: `tests/pim/samsung_hbm_pim/test_samsung_owner_maps.py` (static:
four realizable ties, identity table for the program, row table only when
rows move, reduction-split ties rejected at ranking and at a forced
emission, equal predictions; simulator: vendor REDUCE with the identity
and the cyclic map, ordered program with two ties),
`tests/pim/test_layout_selection.py` (Samsung tie count and reasons,
AiM/UPMEM keep the one-program-per-work-item reason).

### 8.4 Open

* No Samsung tie is cycle-distinguishable on this simulator. A placement
  that changes cost would need the conductor to skip unoccupied channels or
  blocks (occupancy) or a different row-open pattern per unit; neither is
  modelled.
* Reduction splits (the `reassociate` minimum, 48,128 B versus 72,704 B)
  need a partial-sum combine across PIM units: reduction-policy work.
* GEMM (two output coordinates), masked row counts and elementwise kernels
  run only the program map.

## 9. Multi-kernel programs (launch schedule, 2026-11-02)

A dataflow region with several `@allo.work` kernels lowers to one
`SamsungProgram` per kernel group (`SamsungCtx.programs`). It runs when it
was compiled with an `@allo.host_program` (`host_moves=`), whose launch
schedule (`spmw_plan.launch_schedule`: `HostStep`s and `LaunchStep`s in
program order) drives one `pim_driver` invocation per launch
(`spmw_samsung._run_samsung_launches`). The conductor is unchanged: every
invocation starts with empty banks, GRF and CRF, and ends with a readback.
Without a launch schedule, `run()` raises `NotImplementedError` naming the
missing host program.

### 9.1 Contract (`samsung_launch_plan`, pure, checked before any simulator call)

* Binding. Each kernel's matrix, vector and output are bound to workload
  parameters through `trace.source_value_refs` (operands by `value_ref`;
  the output of a contraction is the parameter on an argument of the
  kernel's own functions that no operand reads). Inputs and outputs of a
  multi-kernel run are named by workload parameter.
* Staging. `scatter(b)` / `broadcast(b)` snapshot the host copy of `b` and
  stage it for the next launch that reads `b`. The selected program decides
  the verb: matrix, carried initial accumulator and elementwise inputs are
  bank-resident (scatter, realized in that launch's preload phase); the
  vector is broadcast under host GRF_A fill (realized in its execution
  phase) and scattered under CRF fill. A staging declared before an earlier
  launch (V before mv1 in `_chained`) is placed by the invocation of the
  launch that reads it, once.
* Threading. A launch's result is the device value of its output buffer.
  `gather(b)` takes it into the host copy (FP16, the declared buffer type);
  a later `broadcast(b)` / `scatter(b)` stages that copy. The second launch
  of `_chained` therefore reads the FP16 `y` gathered from the first.
* Rejected, with a `ValueError` naming the residency: a launch operand that
  a previous launch left on the device without a gather and re-staging
  (`_legacy_order`: "keeps 'y' device-resident across the two launches");
  a staging read by one launch and needed again by a later one without a
  new staging; a buffer declared `resident=` before the program; a staging
  of a host copy older than a device result (stale broadcast); a verb that
  does not match the selected program's fill; a gather of a buffer no
  launch produced; sliced buffer transfers.
* Accounting. `RunResult.cycles` = sum of the launches' invocation cycles
  + the cost-graph latency of host steps no invocation performs (a staging
  no launch reads, a repeated gather; `extra["host_steps"][i]["realized_by"]
  == "modelled"`). `extra["launches"]` has one record per launch (kernel,
  program kind, operands, staging host steps, output, gathering host steps,
  cycles, `cycle_phases`, its own `timing_scope`, conductor receipt, CRF,
  plan, levers); `extra["timing_scope"]` states the sum; `extra["outputs"]`
  holds every gathered buffer, which `CompiledCallable` writes back.
  Launches do not overlap; host arithmetic between launches (lane sum of a
  REDUCE result, FP16 rounding of a gathered buffer) is not timed.

### 9.2 Executed (`_two_kernels`, 64 x 64 GEMVs, `z = V @ (W @ x)`)

Installed `pim_driver` (section 8.2 build), inputs `W, V ~ U(-0.5, 0.5)`,
`x ~ U(-1, 1)` in FP16 (probe seed 7; the tests draw fresh seeds):

| Host program, policy | Program | Launch cycles (preload / exec / readback) | Total | Max abs err y / z vs FP32 chain (max abs z) | Cost graph |
|---|---|---|---|---|---|
| `_chained`, `reassociate` | REDUCE (bank_row, host fill) | 2,813 (1,518 / 1,114 / 181) each | 5,626 | 1.9e-3 / 4.7e-3 (6.31) | 706 |
| `_chained`, strict (default) | TEMPORAL_REDUCE | 7,485 (3,251 / 4,167 / 67) each | 14,970 | 4.8e-3 / 9.7e-3 (6.31) | 8,530 |
| `_legacy_order`, either | | rejected before the simulator | | | 8,524 / 700 |

Every host step of `_chained` is performed inside an invocation (modelled
host-step cycles 0). The cost graph prices the REDUCE program without
preload and readback (section 6.3), hence 706 versus 5,626.

Tests: `tests/pim/test_host_program_analysis.py` (`test_samsung_launch_plan_threads_gathered_results`,
`test_samsung_legacy_order_rejects_device_resident_intermediate`,
`test_samsung_reused_staging_is_rejected`, and the simulator-gated
`test_samsung_runs_chained_launches[reassociate-REDUCE|strict-TEMPORAL_REDUCE]`,
which compare `y` and `z` with the FP32 chain within 1e-2 of the largest
reference magnitude and `z` with `V @ y_gathered`).

### 9.3 Open

* The virtual estimate of `_legacy_order` prices no hand-off of `y`
  between the kernels (8,524 / 700, below `_chained`) although no Samsung
  program can execute that order; the cost graph does not check that a
  consumer's GRF_A fill has a matching host step.
* Co-residency of buffers staged for later launches (V while mv1 runs) is
  not checked against bank capacity; each invocation places its operands at
  the conductor's fixed rows.
* Repeated launches of one kernel over slices of a buffer (`X[b]`) are
  rejected (whole-buffer transfers only); `run_batched` covers the
  resident-matrix batch.

