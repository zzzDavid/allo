<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Samsung bank/row swizzle: what the production program does and the route the paper row needs

Status (round 37, 2026-11-04): the tile-row levers are implemented through
the shared pipeline with the selected map as the contract, the estimate
includes the staging and the default selection earns the swizzled rows
(sections "Round 36" and finding (4) below); the paper row `layout_samsung_xor_identity_vs_synthesized`
(identity packing 148 cycles, synthesized XOR 116, PIMSimulator, two rounds
of four requests replicated over 64 channels) is OPEN (measured with a
program-level metric), not REPRODUCED.

## Correction of the round 32 diagnosis

Round 32 claimed every reduction tile stays at physical DRAM row 0. That
was wrong (reviewer, round 33): `PIMAddrManager::addrGenSafe`
(`experiments/simulators/PIMSimulator/src/tests/KernelAddrGen.cpp`) carries
column overflow into the row, the installed configuration has `NUM_COLS=128`
and `BL=4` (32 burst columns per row), and a tile spans `G*E = 64` columns,
so each tile occupies two physical rows. The conductor overlay now dumps its
normalized stream (`TENON_CONDUCTOR_ADDRESS_TRACE=<path>`: `PASS n j`,
`<tag> ch bank row col` after the carry, `BARRIER`), and the dump of the
paper's 8192x256 GEMV (`tenon-release-evidence/r33/trace_gemv_m8192_k256_*`)
shows, per pass and channel: 16 auto-MAC groups of 8 consecutive columns,
each closed by a barrier on every channel; `bank_row` reads bank 0 at rows
0-3 (pass 0) and 4-7 (pass 1), seven same-bank row changes over the two
passes; `dual_fiber` reads banks 0 and 1 at rows 0-3, three row changes in
each; plus the host GRF_A fill per tile and one drain (8 writes to bank 1)
per pass. The matrix staging writes rows 0-7 (bank_row) or 0-3 (dual_fiber)
of every bank of channel 0.

## What the paper's experiment measures, and why the GEMV stream is not it

The probe's trace has rounds of two **concurrent** requests to the same
bank at different rows (identity) or to two banks (XOR); the conflict is
inside a barrier group. The production REDUCE stream never has two rows of
one bank inside a barrier group: a group is eight consecutive columns of
one row of one bank. Its layout-dependent cost is the row change between
successive groups of the same bank (seven for `bank_row`, three per bank
for `dual_fiber`), a different mechanism (activation after a barrier, not
a bank conflict within a round). The request multiplicity (256 reads of
eight-column groups against four single reads), ordering (sequential
groups against concurrent pairs) and timing boundary (pass phase of a
program against an isolated two-round probe) all differ, so no
compression of the GEMV stream reproduces the four-request experiment;
the frozen 148/116 remain a probe-level diagnostic.

## The production slice that landed (round 33)

`allo/spmw_samsung.py`:

* `samsung_reduce_access_stream(program, geometry, columns)` /
  `samsung_access_stream(compiled)`: the stream the conductor issues during
  the passes, derived from the lowered `SamsungProgram` (`binds` order,
  `tile_fiber`, `k_tiles`, `rows_per_block`, `vector_from_bank`) and the
  simulated geometry, one `SamsungAccessRound` per barrier (MAC, FILL,
  HOST_FILL, DRAIN) with normalized `(bank, row, column)` transactions.
* `samsung_conductor_trace_text(rounds)` renders it in the conductor's dump
  format; `tests/pim/samsung_hbm_pim/test_access_stream.py` compares it
  byte for byte with a dump captured from the running conductor for both
  fiber placements of the 8192x256 and the 4096x512 GEMVs (the latter has
  four tiles, `tile_fiber` 0,1,0,1 against 0,0,0,0), and checks the counts,
  rows and row-change figures without the simulator.
* `samsung_access_trace_text(rounds)` renders the MAC and FILL read rounds
  for the layout-trace harness; the harness now lets the memory system
  advance while a long stream is queued (`pimsim_layout_trace.cc`,
  bounded retries) so an executable stream replays, not only a four-line
  probe. The replay is the pass-phase bank reads as plain reads with a
  barrier per group: no PIM execution, writes, staging, CRF programming or
  mode changes, so it isolates the row-change cost the layout controls.

Measured (`r33/access_stream_*.json`, `samsung_fiber_ab_trace_*.json`):

| GEMV | placement | full conductor cycles | harness replay of the pass-phase reads | same-bank row changes |
|---|---|---|---|---|
| 8192x256 | dual_fiber (vendor) | 8,247 | 1,188 | 3 + 3 |
| 8192x256 | bank_row | 8,153 | 1,248 | 7 |
| 4096x512 | dual_fiber (vendor) | 8,041 | 1,228 | 3 + 3 |
| 4096x512 | bank_row | 7,947 | 1,248 | 7 |

What these numbers are and are not. The harness replay issues the
stream's MAC and FILL addresses as ordinary reads with a barrier per
group: no PIM mode, no PIM execution of the MAC, no host fills, drains,
staging, CRF programming or mode changes. Its 1,188 against 1,248 is a
diagnostic of the address sequence under the simulator's DRAM timing; it
does not establish that the conductor's own bank-read phase is faster
under `dual_fiber`, and it does not attribute the full-program difference
(8,153 against 8,247) to any component. The "row changes" column counts
address changes between successive MAC groups of one bank in the modelled
stream; they are not measured row activations. Causal attribution of the
full-program difference needs controlled simulator evidence (the
conductor's per-phase cycle breakdown or its DRAM statistics under each
placement), which this round did not produce; the cost model's
`FIBER_TILE_CYCLES` surcharge for split fibers is an empirical term, not
that attribution.

## Round 36: the tile-row levers implemented through the shared pipeline

The row pairing the paper's experiment needs cannot pair two *k-tiles*
(two tiles need two vector slices in one GRF_A, which has one), so the
production realization pairs the two physical rows of **one** tile: same
vector slice, distinct GRF_B entries (`PIMRank::getGrfIdxHigh` selects the
entry from the row's parity), accumulation order per entry unchanged, so
the results are bit identical to `bank_row`. Three placements were added
to `_samsung_enumerate` (every stage of the shared pipeline: candidate,
the shared contract, legality at lowering, the plan the conductor
executes, `check_realization` of the levers, the access-stream model):

| placement | rows of a tile | barrier | lowering | conductor |
|---|---|---|---|---|
| `bank_row_rowpair` | both in the even fiber | one per pair (g, g+G/2) | vendor CRF, conductor issues the pair | `issue_order 1` |
| `split_rows` | first in the even fiber, second in the odd fiber at the same column (the F2 `half -> bank` swizzle `optimal_swizzle` chooses) | one per group (vendor) | loop body `MAC even; MAC odd; JUMP` | `split_rows 1` |
| `split_rows_rowpair` | split, one barrier per pair | refused at lowering (below) | | `issue_order 1 split_rows 1` |

Measured on the real conductor through `allo.compile` with `layout=` and
the vendor levers (host GRF_A fill, shared CRF), `tenon-release-evidence/
r36/run2/rowpair_*.json`, outputs within 2.3e-4 of the FP32 product, the
derived access stream byte-identical to the conductor's dump in every
case, `pim_driver` sha256 `82db787a...`:

| GEMV | placement | cycles | preload / exec / readback | results |
|---|---|---|---|---|
| 8192x256 | bank_row (identity rows, per-group barriers) | 8,153 | 5,628 / 2,216 / 309 | reference |
| 8192x256 | dual_fiber (tiles alternate fibers) | 8,247 | 5,628 / 2,310 / 309 | |
| 8192x256 | bank_row_rowpair (identity rows, one barrier per pair) | 53,634 | 5,628 / 47,697 / 309 | bit identical to bank_row |
| 8192x256 | split_rows (XOR rows, per-group barriers) | 7,255 | 4,595 / 2,351 / 309 | bit identical to bank_row |
| 4096x512 | bank_row | 7,947 | 5,628 / 2,138 / 181 | reference |
| 4096x512 | dual_fiber | 8,041 | 5,628 / 2,232 / 181 | |
| 4096x512 | bank_row_rowpair | 53,506 | 5,628 / 47,697 / 181 | bit identical |
| 4096x512 | split_rows | 7,057 | 4,595 / 2,281 / 181 | bit identical |

Findings. (1) Two rows of one bank inside one barrier group (the identity
packing's conflict) cost 45,481 and 45,559 execution cycles for 16 pairs
on the two shapes, about 2,845 per pair, far beyond tRP + tRCDRD: the
barrier group waits on the row conflict across the fabric. (2) The XOR
packing with the vendor's barrier structure (`split_rows`) is the fastest
program on both shapes, 1.124x and 1.126x over `bank_row`, with the gain
in the staging phase (4,595 against 5,628) and a small execution loss
(2,351 against 2,216). (3) `split_rows_rowpair` executed but produced
wrong results (max error 0.14 and 0.18 against 1.4e-4): with two banks
inside one barrier group the memory controller may reorder the two banks'
column commands while the PIM's lockstep CRF executes fixed-bank MAC
instructions (`PIMOpdType::EVEN_BANK`/`ODD_BANK`; the simulator's
"CRF bank coding and bank id from packet are inconsistent" warning is
compiled out of the release build, so the cause is inferred from the ISA
and the symptom, not logged). The ISA has no bank-agnostic operand, so
that placement is enumerated and refused at lowering with this reason.
(4) In round 36 the cost model priced `split_rows` above `bank_row` (301
against 253 for 8192x256) because the staging saving was outside the MAC
rule; round 37 added the first staging to the estimate from the
conductor's staging schedule priced with the device ini, and the default
selection now picks `split_rows` on both paper GEMVs and on a held-out
37x129x2 contraction, the measured fastest on all three (S-16 repaired).
Round 37 also made the selected half -> bank map the contract: it names the
declared matrix packing, drives the lowering and the conductor's
addressing (`half_fiber` in the plan), sits in the decision identity and
is compared at realization; an edited map or packing is rejected before
selection.

What this means for the paper row. The paper's four-request probe
measures within-round bank conflicts on plain reads (148 against 116,
1.28x). The production program cannot issue two banks inside one barrier
group, so that exact mechanism is unreachable; what the shared pipeline
does reach is the same layout choice (identity rows in one bank against
XOR rows across the fibers) under the PIM's barrier discipline, where the
XOR packing is 1.12-1.13x faster on the whole program. The inventory row
is therefore OPEN (measured through the shared pipeline with a different,
program-level metric), not REPRODUCED.

## The route for the paper row as first specified (superseded above)

The paper's experiment needs two rows of one bank inside one barrier
group, which the REDUCE lowering never issues. Making it an ordinary-
program capability means a REDUCE variant whose per-group work reads two
tiles' rows concurrently (two auto-MAC operands of the same bank in one
group), with the `matrix_rows` placement lever (identity: both tiles in one
bank at different rows; `xor`: the second tile moved to the partner bank
by `LinearLayout.optimal_swizzle` over `{tile} -> (bank, row)`):

1. The lever is a candidate of `_samsung_enumerate` with the F2 map as its
   `layout`; `candidate_effects` declares the matrix storage with the row
   coordinate; the decision key carries `matrix_rows` and the map's
   intra-group conflict count.
2. Materialization carries `tile_row` beside `tile_fiber` in
   `SamsungProgram` and the conductor plan; the conductor stages and
   addresses tiles at `(bank, row, column)` with explicit rows instead of
   the flattened column, keeping the staging collision-free (distinct
   `(bank, row, column)` per burst), the pass offsets, the vector and result
   bases, and auto-mode register indexing: `PIMRank.h` selects the GRF_B
   entry from the physical row's low bit and column bits
   (`getGrfIdxHigh`), so a tile's two rows must keep the parity the
   accumulator expects. `check_realization` compares `(tile_fiber,
   tile_row)` with the declared map.
3. Cost: a row-activation and an intra-group conflict term derived from
   the target's timing facts (tRCD/tRP-class constants of the device ini)
   or from independent microbenchmarks on geometries other than the
   evaluation's, with the ranking validated on separate geometries; not a
   fit to 148/116.
4. Evidence: the program-derived stream (above) equals the conductor dump
   for the new variant; the harness replay and the full conductor run are
   reported side by side with the paper's 148/116 and an explicit
   statement of the request multiplicity and timing boundary; the row is
   REPRODUCED only when an ordinary program's stream has the experiment's
   multiplicity, ordering and boundary.

Out of scope: changing the paper, promoting the fabricated-match probe, or
adding another synthetic-match adapter.
