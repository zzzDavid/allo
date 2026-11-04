<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Samsung bank/row swizzle: the shared-pipeline route the paper row needs

Status (round 32, 2026-11-04): design; not implemented. Inventory row
`layout_samsung_xor_identity_vs_synthesized` (paper
`tables/eval-layout-mechanisms.tex`: identity packing 148 cycles, synthesized
XOR 116 cycles, PIMSimulator, two rounds of four requests replicated over 64
channels) stays UNVERIFIED for G10 until this route exists.

## What the paper row measures

`allo/pim/samsung_layout_swizzle.py` builds a `(bank, tile, column) ->
(bank, row, column)` F2 map for a 2-bank, 2-tile, 1-column geometry. The
identity map puts the two tiles of one logical round into two rows of the
same physical bank, so their activations serialize; the synthesized map
(`LinearLayout.optimal_swizzle`, `physical_bank = bank XOR tile`, `row =
tile`) spreads them over two banks. `format_simulator_trace` emits
`round bank row column` lines and the harness
`tests/pim/samsung_hbm_pim/layout_swizzle/pimsim_layout_trace.cc` replays
them on PIMSimulator with a barrier per round. The probe obtains the bank
projection from the production enumerator through a fabricated `MatchedOp`
(`select_autoscheduled_xor_layout`), which is why the row is not a G10
reproduction (review round 32).

## Why no ordinary program reaches it today

The production Samsung REDUCE lowering (`spmw_samsung.SamsungCtx`, the
conductor `experiments/simulators/setup/PIMSimulator/src/tests/
TenonConductor.inc`) stages every reduction tile at DRAM **row 0** of its
bank fiber and distinguishes tiles by **column** (`tile_col = (pass *
fiber_tiles + local_index) * G * E`); the only layout degree of freedom the
candidates expose is the bank fiber of each tile (`tile_fiber`: `bank_row`
keeps every tile in the even fiber, `dual_fiber` alternates fibers through
the enumerator's F2 swizzle). Row conflicts, the mechanism the paper row
exercises, cannot occur in that program, so neither the identity row packing
nor its XOR repair has an ordinary-program realization. Measured on the real
conductor through `allo.compile` with `layout=` (round 32,
`tenon-release-evidence/r32/samsung_fiber_ab_m8192_k256.json`, GEMV
8192x256, fp16, reassociate): `dual_fiber` with the vendor levers 8,247
cycles (the paper's vendor row), `bank_row` with the same levers 8,153, the
default selection 8,153; outputs within 1.5e-4 of the fp32 reference. The
fiber swizzle does not help the production program because there is
nothing to de-conflict at row 0.

## The route

1. **Candidate family.** A REDUCE placement lever `matrix_rows` with values
   `identity` and `xor`, enumerated by `_samsung_enumerate` from one
   `LinearLayout` over `{pass, tile}` into `(bank, row)`: identity maps
   `tile -> row` within one fiber; `xor` is `LinearLayout.optimal_swizzle`
   of that map with `bank_dims=("bank",)`, `segment_dims=("tile",)`. The
   placement's `layout` is that map (no lifted copy); `candidate_effects`
   declares the matrix storage with the row dimension; the decision key
   carries `matrix_rows` and the map's conflict count
   (`LinearLayout.conflict_count`).
2. **Cost.** A row-activation term in `allo/pim/costs/samsung.py` priced
   from the conflict count per round (one round per pass), calibrated on the
   harness numbers (148/116 for the two-tile geometry) and labelled with its
   provenance like the other Samsung terms.
3. **Materialization.** `SamsungProgram` gains `tile_row` beside
   `tile_fiber`; `samsung_reduce_plan` emits `tile_row`; the conductor's
   `stage_matrix`/`run_pass` use `row = tile_row[t]` in `address(...)`
   instead of the constant 0 (a plan-driven change to
   `TenonConductor.inc`, rebuilt through the overlay in
   `experiments/simulators/setup/PIMSimulator`, P-04). `check_realization`
   compares the program's `(tile_fiber, tile_row)` with the declared map.
4. **Artifact.** The compiled program exposes `bank_access_trace()`: one
   round per pass, one `(bank, row, column)` access per PIM block and tile,
   derived from the materialized `SamsungProgram` (not from a geometry
   object), in the harness's text format. The same trace drives the
   harness for the paper's micro-measurement and is the input the full
   conductor run addresses.
5. **Reproduction.** An ordinary two-tile GEMV compiled twice through
   `allo.compile` (`layout=` the identity-row candidate; default selection,
   expected to pick `xor`), the harness cycles of both traces and the full
   conductor cycles reported side by side; the paper's 148/116 compared
   with the harness numbers, the ratio being the comparable metric.

Out of scope: changing the paper, keeping the fabricated-match probe as a
production path (it stays a diagnostic), or adding another synthetic-match
adapter.
