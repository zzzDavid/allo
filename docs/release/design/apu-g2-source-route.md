<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Gemini-II source route, typed tick model, and card operation

Status: implemented (working tree on top of `6efaca5`), compile-only and
emulated-card tests green on this server; real-card confirmation pending on
zhang-x4 (section 8). Closes, subject to that run: G2-02, G2-03, G2-05,
G2-06 (partially, section 7), G2-08, G2-04 (test written). Audit:
`docs/release/audit/apu-g2.md`.

## 1. What the paper's "Tenon workload" was

The archived campaign (`tenon-artifacts/apuv2-composed-v2`) shows that no
`@allo` source existed for the five Gemini-II kernels:

* the semantic workload is a JSON graph per case (`graph-specs/<case>.json`:
  nodes `matmul`, `batched_matmul`, `matvec`, `transposed_matvec`,
  `scalar_mul`, `add`, with logical INT widths and contracts such as
  `q[i] = sum(j, a[i,j] * p[j])`);
* the Tenon lane (`provenance/implementations/tenon/adapter.py`,
  `program_and_dots`, lines 70-134) switches on the case name, constructs one
  `APUG2ComposedContractionProgram` record and packs operands with
  `pack_apu_g2_batched_gemm_dots`, `pack_apu_g2_matrix_vector_dots`
  (`transpose=True` for the transposed products),
  `interleave_apu_g2_dot_streams`, `np.stack((x1, x2), axis=1)` for MVT, and
  transposes the gathered output of BiCG/MVT;
* `operations.json` records `allo.compile(program, ..., backend='virtual')`
  with `"cycles": 16` (the old op count).

The "Workload LoC 33/33/42/38/46" column (`tables/eval-apuv2.tex`,
"normalized standalone workload LoC") has no counting script or source file
in the artifact, the paper tree, `design_doc/` or `dev/`; its basis is not
recoverable (audit OQ-2). The source kernels below are 7-10 function lines
each (8-11 with their shape constants; section 3), so the table's numbers
are not reproduced by this release.

## 2. Route

`allo.compile(source, build_apu_g2_target(), apu_g2_cost)`:
`compiler.py` sends every non-record workload on `apu_v2` through the
common pipeline: the shared front stages, then the `autoschedule`
enumerator `apu_g2_source_candidates`
(`allo/pim/apu_g2_source_route.py`, called from `allo/spmw_autoschedule.py`),
legality, the shared ranking and the `APUG2Ctx` materialization; the result
is the common `CompiledCallable` (outputs copied into the caller's arrays for
`kind="device"`, untouched for `kind="virtual"`). The route-specific entry
`compile_apu_g2_source_workload` this section once named was removed in
round 13 (`shared-compilation-path.md`). Anything outside the family
raises `APUG2SourceRouteError` (a `NotImplementedError`) whose message names
the reason and the supported family.

### 2.1 Matching

The source is customized to MLIR and matched against
`build_apu_g2_contraction_patterns()` (`allo/pim/targets.py`, Gemini-II
section): `MAC` (`acc += x * y`, accumulating) and `AXPBY`
(`alpha*x + beta*y`, literal `const_params`). The hardware target itself is
unchanged: its typed primitives stay non-matchable (one source reduction is a
whole recipe, not one primitive), and `tests/perf/test_apu_g2_target_layout.py`
pins that surface. `require_match_coverage` must succeed, so any store the two
idioms do not explain (an extra elementwise output, an unsupported
accumulator initializer) is rejected.

### 2.2 Derivation rules (from the matches only)

| Fact | Derived from |
|---|---|
| Reduction loop and K | `axis_roles["reduction"]` (exactly one loop); K = its trip count |
| Result coordinates and their order | output-buffer subscripts: the copy sink's index text (`C[i, j] = acc`), the accumulator's own access map (`C[i, j] += ...`), or the AXPBY store indices (identity store map checked in the MLIR) |
| Operand access, transposition, broadcast | each operand's `index_map` and `indices`: one loop variable per buffer dimension; `A[i, j]` with reduction `i` is the transposed matrix-vector form; a loop absent from an operand broadcasts it |
| lhs / rhs | `axis_roles["matrix"]` / `["vector"]` |
| Accumulator | `accumulator_init`: `zero` (local or buffer) -> identity epilogue; `carried` (function argument, no initializer) -> accumulate epilogue with the caller's values |
| Pair-affine epilogue | an `AXPBY` whose `x`, `y` are the two `MAC` accumulators (by `value_ref`); coefficients from `extra["constants"]`; stream order = (x, y) |
| Types | `itypes` signedness of arguments, `unsigned` attribute of local allocs, widths from memref element types |
| Output shape | output buffer extents (must equal the result loop extents) |

Width rules (the source must not wrap where the device is exact): one
signedness for everything; K a power of two; exact dot width
`D = a + b + log2 K <= 24`; local accumulator `>= D`; identity output `>= D`;
carried output `>= D + 1` (caller values must fit `int D` at call time, else
`ValueError`); pair-affine output `>= D + bits(coefficients) + 1`.

Rejected with a finite error (tested): non-power-of-two K, mixed signedness,
three or more products, an accumulator narrower than the exact sum, float
buffers, combined products of different operand types, K > 512 (or K > 256
for carried/pair-affine epilogues), subscripts that are not plain loop
variables, extra effects, outputs that are also operands, dataflow regions.

### 2.3 Launch plans (the searched space)

`enumerate_apu_g2_launch_plans` builds every structurally legal plan:

* **fusion**: `single` (one product), `fused` (two products interleaved into
  one dot stream; mandatory for pair-affine, optional for independent
  products with equal operand types, K, extents and accumulator kind) or
  `separate` (one launch sequence per product);
* **reduction chunks** `k1` / `k2`: `k2` computes `K/2` in an identity launch
  and adds the second half in an accumulating launch whose accumulator is the
  first launch's device result (exact: `int(D-1) + int(D-1) -> int(D)`);
  legal only for identity epilogues; the only legal plan for K = 512;
* **output tiles**: `APUG2DotTiling(dot_rows, K_chunk)` splits rows beyond
  one `(4, 65536)` carrier (`4*65536/K_pad` dots) into sequential launches.

Each launch is one `APUG2ComposedContractionProgram`; a plan's execution
graph chains every launch's priced recipe. `rank_apu_g2_launch_plans` orders
by estimate, then fewer launches, then catalog order; the first is selected
(`selection="cost"`), `layout="<plan name>"` forces one
(`selection="forced"`). `compiled.compiled.placement_ranking`,
`.rejected` and `.manifest()` keep every candidate and reason.

Observed rankings (calibrated model): GEMM/batched GEMM `single/k1` 1,714 vs
`single/k2` 2,550; BiCG `fused/k1` 2,610 < `fused/k2` 3,500 <
`separate/k1` 5,220 < `separate/k2` 7,000; MVT `fused/k1` 2,740 <
`separate/k1` 5,480; GESUMMV only `fused/k1` (1,882). Fusion always wins
under nonnegative coefficients (one carrier holds both streams); the
reduction split wins for GEMM 64x128x33, where both alternatives need two
launches (`single/k2` 2,550 vs `single/k1` 3,428).

### 2.4 Execution

Host packing is a broadcast-gather (`apu_g2_stream_dots`): slice each operand
to its loop extents, transpose into (result..., reduction) order, broadcast
absent loops, reshape to `(coordinates, K)`; streams are interleaved with
`interleave_apu_g2_dot_streams`; carried accumulators are stacked per stream.
Each launch runs through `run_apu_g2_composed(..., expected_layout_fingerprint=
<selected>)`; results are scattered to the output parameters and compared
with an independent `einsum` oracle of the matched contractions before the
`RunResult(kind="device", cycles=sum of launch cycles)` is returned.
`compiled.compiled.repetitions = 20` sets the timed batch (default 256).

## 3. Source workloads and equality with the artifact

`benchmarks/apu_g2/workloads.py` (`PAPER_KERNELS`):

| Kernel | Function lines (with shape constants) | Selected plan | Packing form derived |
|---|---|---|---|
| `gemm` | 7 (8) | `single/k1`, 1 launch | `pack_apu_g2_batched_gemm_dots` |
| `batched_gemm` | 10 (11) | `single/k1` | `pack_apu_g2_batched_gemm_dots` |
| `gesummv` | 8 (10) | `fused/k1`, pair-affine (3, 2) | 2 x `pack_apu_g2_matrix_vector_dots` |
| `bicg` | 8 (9) | `fused/k1`, output (256, 2) | matrix-vector + transposed |
| `mvt` | 7 (8) | `fused/k1`, accumulate int14 -> int15 | matrix-vector + transposed |

For all five (`tests/perf/test_apu_g2_source_route.py`):

* layout, program and recipe fingerprints and the `LinearLayout` equal the
  artifact's `cases/<case>/layout.json` (layouts `20a85096...`,
  `07f71036...`, `b4afa6b9...`, `a06509fd...` x2, also the Round 2 card
  runs);
* on the artifact's own fixtures the packed lhs/rhs/auxiliary carriers are
  byte-identical (SHA-256) to the campaign's `packed_files`, and the source
  oracle equals `oracle.i16le.bin`;
* with random inputs the packing equals the adapter's helper calls.

Nearby/renamed programs compile: a renamed GESUMMV (other parameter, local
and loop names) and an in-place `C[i, j] += ...` GEMM give the identical
program fingerprints; GEMM 64x128x32/33, GESUMMV M=512, BiCG N=512 (K=512,
`fused/k2`) and a GEMM with a transposed left operand (`A[k, i]`,
`broadcast_gather`) compile.

## 4. Cost model (G2-03)

`APUG2TypedTickModel` (`allo/pim/apu_g2_pipeline.py`), used by the rules
`build_apu_g2_cost(tick_model)` registers for `L1_TO_MMB_SEG0/SEG1`,
`MMB_TO_L1`, `MUL_TYPED`, `GROUP_REDUCE_ADD_TYPED`, `ADD_TYPED`,
`COPY_ODD_TO_EVEN_VECTORS`, `DIV_TYPED` and `SEU_BARRIER`
(`apu_g2_cost = build_apu_g2_cost()`). Every issued recipe call becomes one
cost event (`emit_apu_g2_priced_recipe`) with features:

| Feature | Per call | Ticks per unit |
|---|---|---|
| `pipeline_bodies` | 1 for SEU_BARRIER (body dispatch + completion) | 195.704 |
| `issued_calls` | 1 for every other call | 0.0 |
| `row_bits` | destination width of a copy, ADD_TYPED, COPY_ODD_TO_EVEN | 2.235 |
| `multiply_bit_products` | lhs bits x rhs bits of MUL_TYPED | 0.609 |
| `reduce_levels` | log2 K of GROUP_REDUCE_ADD_TYPED | 84.848 |
| `reduce_shift_columns` | 2^log2K - 1 (level l shifts 2^l columns) | 6.505 |
| `divide_bits_squared` | bits^2 of DIV_TYPED | 2.937 |

Unit: device ARC ticks per pipeline body, the runtime's
`round(device_pipeline_ticks / repetitions)` (bodies plus one completion
barrier, transfers excluded). A plan's estimate is the sum over its launches
(no inter-launch overlap; host readback between launches is outside the unit
and counted in `host_roundtrips_between_launches`). The dot count enters only
through carriers/launches: within one carrier every call covers all 262,144
lanes (GEMM 64x31 and 64x32 both 1,714; measured GEMM 1,984 dots 1,711 vs
batched 2,048 dots 1,720).

Provenance: `APUG2_TYPED_TICK_MEASUREMENTS`, the ten Round 2 programs of
`docs/release/VALIDATION.md` (host en-ec-zhang-x4, card apu-00 PCI
0000:61:00.0, firmware GSI_G2_Infrastructure_V1.24.1, SDK /opt/gsi,
2026-11-02, `ffb4a78` + wave 2, repetitions 20, logs
`tenon-release-evidence/g2/step2_3.log`, `step4.log`). Fit:
`fit_apu_g2_typed_tick_model()` rebuilds the ten recipes from public program
records, sums their features and solves exhaustive active-set NNLS; the
stored coefficients are that solution rounded to 3 decimals (re-checked by a
test).

| Program | Measured | In-sample error | Leave-one-out error |
|---|---:|---:|---:|
| typed add i16 | 333 | +1.7% | +16.7% |
| typed mul i8->i16 | 312 | -1.8% | -17.8% |
| typed div u16 | 1,019 | 0.0% | -73.8% (only DIV point) |
| typed block_sum u8 K256 | 2,614 | -1.0% | -1.8% |
| typed dot i4 K128 | 1,713 | -0.0% | -0.0% |
| composed GEMM | 1,711 | +0.1% | +0.1% |
| composed batched GEMM | 1,720 | -0.4% | -0.6% |
| composed GESUMMV | 1,876 | +0.3% | +1.2% |
| composed BiCG | 2,569 | +1.6% | +2.6% |
| composed MVT | 2,754 | -0.5% | -1.2% |

Graph estimates add at most half a tick of rounding per call (typed add
340, +2.1%).

Validity domain (`APUG2TypedTickModel.validity`): reduction depths {7, 8},
DIV width 16, row widths 2..16, multiply bit-products 4..64. Outside it the
estimate is still produced (selection needs a total order) and labelled
`estimate_validity="extrapolated"` on the graph, plan and virtual
`RunResult`. The 2^l shift scaling used below depth 7 is an extrapolation;
its shape agrees with the 2026-07-08 uint16 dot-tile points
(+2,246 ticks from depth 6 to 7, +4,793 from 7 to 8; the uint16 dot-tile
kernel and its measured-point table left the release with G2-11 and are in
git history before that change). The leave-one-out errors show elementwise-only programs are
weakly identified (add/mul +-18%).

Stated before the confirmation run: for calibrated single-launch programs
the next card run should land within +-10% of the estimate; for
extrapolated plans (GEMM 64x128x33 `single/k2`) no tolerance is claimed,
the ratio is recorded. A failure is reported, not refitted silently.

`backend="virtual"` returns `RunResult(kind="virtual", cycles=<estimate>,
backend="virtual")` with `extra["cost_model"]`, `["latency_unit"]` and
`["estimate_validity"]`; typed and composed records label themselves the
same way. Perturbation: `build_apu_g2_cost(APUG2_TYPED_TICK_MODEL.scaled(
row_bits=20.0))` has a different fingerprint, raises the GEMM estimate
(1,714 -> 3,370) with the same program, and flips GEMM 64x128x33 from
`single/k2` to `single/k1`.

## 5. Capacity and tiling (G2-05)

A composed record still rejects more dots than one carrier holds, naming the
capacity (`(64, 33)` at K=128: "exceeds capacity 2048"). The source route
tiles instead: `single/k1` for 64x128x33 is launches over dot rows
`[0, 2048)` and `[2048, 2112)`; 64x128x32 is exactly one carrier. Reductions
up to 512 use the exact two-launch chain; beyond that the error names the
256-value launch capacity and the two-launch limit.

## 6. Layout binding (G2-08)

`run_apu_g2_composed(..., expected_layout_fingerprint=...)` refuses a
program whose layout fingerprint differs from the selected one before
probing the card or building. After packing,
`attest_apu_g2_composed_packing` recomputes the program's lane image from its
`APUG2DotTilePlan` (F2-linear: `coord(row, k) = coord(row, 0) XOR
coord(0, k)`) and checks that every logical value sits at its lane, every
other lane is zero, and auxiliary values sit only on dot heads; the record
(`lane_image_sha256` over fingerprint + image) is returned in
`extra["packing_attestation"]`. The host binary must still echo the
fingerprint (`_parse_metrics`).

## 7. Discovery and locking (G2-06)

* SDK root: both template CMakeLists take `TENON_APU_G2_SDK_ROOT` (cache
  default `/opt/gsi`); the runtimes pass `-DTENON_APU_G2_SDK_ROOT=
  spmw_simenv.apu_g2_sdk_root()` to both configures, so a non-default root is
  now accepted. Not parameterized: the `GSI_SYSTEM_g2` / `GSI_GTML_g2` CMake
  config packages (found by CMake's default search; the archived build found
  them in `/usr/lib/cmake`) and the ARC toolchain they select; setting
  `CMAKE_PREFIX_PATH` remains the user's lever.
* Device index: every host program opens `DeviceConfig{
  TENON_APU_G2_DEVICE_INDEX, 0}`; the runtimes pass the index parsed from
  `TENON_APU_G2_DEVICE=apu-NN`. That the first `DeviceConfig` field selects
  the card has not been verified (no SDK header on this host), so indices
  other than 0 are still rejected by `_require_hardware_stack`.
  `spmw_simenv.apu_g2_unsupported_override_reason` (outside this change) is
  no longer called and its docstring is stale.
* Lock: `apu_g2_device_lock()` takes an exclusive `fcntl.flock` on
  `TENON_APU_G2_LOCK` or `/tmp/tenon-apu-g2-<device>.lock` (reentrant in a
  process) around status check, build and launch in the u16-add, typed and
  composed runtimes; `extra["device_lock"]` records the path. Do not also
  wrap a run in an external `flock` on the same file: the runtime opens its
  own descriptor and would wait on the wrapper.

## 8. Card tests (G2-04)

`tests/pim/test_apu_g2_hardware.py` skips with `BLOCKED-DEVICE(<reason>)`
when the SDK files, `gsi_tool` or an Available card is missing. With a card:
one typed program per operation (add, mul, div, block_sum, dot), the five
paper kernels through `allo.compile(source, ...)`, and GEMM 64x128x33 (two
launches), all with fresh random inputs (seed printed) and
`repetitions=20`; outputs are checked against plain NumPy in addition to the
runtimes' own oracle gates; each run prints an `APU_G2_RECORD` JSON line
(cycles, estimate, fingerprints, device library SHA-256, lock path).
Negative controls: a wrong oracle (monkeypatched) must raise a mismatch on
the card; a wrong fingerprint echoed by the real host binary must be
refused; a selected/packed fingerprint mismatch is refused before any card
access (runs everywhere). Card-free equivalents run against an emulated host
binary in `tests/perf/test_apu_g2_source_route.py`.

## 9. Open

* Real-card confirmation of the source route, the K-split plan and the new
  tick model (section 4 tolerance).
* `DeviceConfig` field semantics for multi-card hosts.
* Paper LoC basis (OQ-2).
* Only two products and the `alpha*t + beta*u` / carried / identity
  epilogues; `t + u` (unit coefficients without multiplies) is not matched.
