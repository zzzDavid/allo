# Minimum-movement layout search (paper Section IV)

Status: implemented as a standalone package library (`allo/pim/gf2.py`,
`allo/pim/layout_problem.py`, `allo/pim/layout_movement.py`,
`allo/pim/layout_search.py`). Wired into placement selection through
`allo/spmw_layout_problem.py` (see Section 8). Implements steps 1 to 4 and 7 of the salvage plan in
`docs/release/audit/core.md` Section 3.3 and addresses claim rows 17 to 22 of
`docs/release/CLAIMS.md`.

Source: ported from the snapshot worktree (`.worktrees/allo-snapshot`,
`df35093`) modules `gf2`, `layout_problem`, `prelayout_movement_algebra`,
`canonical_layout_enumeration`, `paper_partial_basis_search`,
`admissible_layout_search` and `scalable_layout_synthesis`, and from the
`reviews/experiments/irregular_audit.py` oracle.

## 1. Data model

| Object | Module | Meaning |
|---|---|---|
| `GF2Matrix`, `GF2Subspace` | `gf2` | Exact GF(2) algebra. Vectors are Python ints (bit `i` = coordinate `i`); subspaces are stored as their unique RREF basis. Copied verbatim from the snapshot. |
| `OwnerAxis`, `OwnerLevel`, `OwnerHierarchy`, `LocalAxis`, `LocalSpaceFlag` | `layout_problem` | The compute hierarchy `U_1 x ... x U_h x S`. Work-coordinate bits are ordered: all axes of level 0 (outermost), then level 1, and so on, then local (storage) bits. `local_space_dimensions` gives `dim T_0 .. dim T_h`. A `LocalSpaceFlag` is a validated nested chain `I = T_0 >= T_1 >= ... >= T_h`. |
| `owner_hierarchy_from_target(target, level_names, *, local_bits=0, active_extents=None)` | `layout_problem` | Builds the hierarchy from a Tenon `Target`. A level name is a unit of the unit tree (every axis of its `mapping` becomes an owner axis, origin `MAPPED_UNIT`) or a register lane axis (`"vr0.lane"`, `"vr0"` for a single-axis register, origin `REGISTER_AXIS`). Every extent must be a power of two; otherwise `TargetHierarchyError` names the axis and the `active_extents={"unit.axis": 2**k}` escape (for example 16 of UPMEM's 24 tasklets). |
| `LayoutAccess(name, access_matrix, mode, element_bytes, cost_per_level, reassociation_allowed=True)` | `layout_movement` | One tensor access. `access_matrix` maps iteration bits (columns) to element bits (rows). `mode` is `READ` or `WRITE_ACCUMULATE`. `element_bytes` may be a `Fraction` for sub-byte types. `cost_per_level[s]` is the target's cost per byte of moving data across owner level `s`. Replaces the snapshot's `TypedOperandReuseEntry` plus `PrelayoutIncidenceKind` validation. |
| `LayoutSearchProblem(hierarchy, accesses, forced_local=(), active_points=None)` | `layout_search` | The search input. `forced_local[s]` lists directions required to stay inside one unit through level `s` (nested, `F_0 >= F_1 >= ...`). `active_points` selects a masked domain; the full range is normalized to `None`. |
| `LayoutSearchPolicy(max_carrier_bits=4, max_levels=3, max_search_nodes=20_672, enable_pruning=True, allow_scalable=True)` | `layout_search` | Search policy (Section 4). |
| `LayoutSearchResult` | `layout_search` | `ties` (work-to-iteration `GF2Matrix`, the paper's `L`), `tie_flags`, `cost`, `termination` in `{EXHAUSTED, BOUND_PROVED, NODE_BUDGET}`, `visited`, `pruned`, `remaining_lower_bounds`, `optimality_proven`, `implementation`, `ties_complete`, `tie_multiplicity`, `telemetry`, `to_linear_layouts(in_dims, out_dims)`. |

A layout is a one-to-one map `L : work bits -> iteration bits`
(`work_to_iteration`). `Q_s` keeps the owner bits through level `s` of
`L^-1`; `T_s = ker Q_s = L(span of work bits after level s)`.

## 2. Movement formula and units

For access `j` with kernel `K_j = ker A_j`:

```
r_{j,s} = rank(Q_s restricted to K_j) = dim K_j - dim(K_j intersect T_s)
C_move(L) = sum_j sum_s c_{j,s} * B_j * (2**r_{j,s} - 1)
B_j      = element_bytes_j * 2**(n - dim K_j)
```

`full_carrier_rank_incidence(kernel, local_space)` returns the rank, the
number of distinct elements `2**(n - dim K)` and the extra owner incidence
`2**(n - dim K) * (2**r - 1)`. Units: the incidence is an element count, extra
copies for a READ access and extra partial results to combine for a
WRITE_ACCUMULATE access. `scale_extra_owner_incidence` multiplies by
`element_bytes` (bytes) and by `cost_per_level[s]` (cost per byte, in the
cost program's unit, for example cycles per byte). All arithmetic is exact
(`Fraction`); floats given as coefficients are converted exactly, so ties are
exact ties. `layout_movement_cost(accesses, hierarchy, L, active_points=None)`
evaluates one layout and returns per-term records.

`r_{j,s}` is cumulative over levels (the paper's `Q_s` keeps all bits through
level `s`), so a copy that crosses an outer level is also counted at every
inner level, exactly as in Eq. pim-movement-cost.

### Reduction order (claim row 13, audit C-09)

`WRITE_ACCUMULATE` with `reassociation_allowed=False` is a strict
source-order reduction: no owner bit may split its kernel. A layout is legal
only if `K_j <= T_h` (the innermost local space), equivalently
`r_{j,s} = 0` at every level. On a full carrier the search adds the span of
all strict kernels (`strict_reduction_kernel`) to every forced-local space,
so illegal layouts never enter the domain ("strict order can rule out
splitting a reduction before search"); if that span does not fit in `T_h`
the search raises `NoLegalLayoutError`. A strict access contributes no
movement term. In the masked mode legality is checked exactly per layout
(Section 5).

## 3. Search algorithm (Algorithm 1)

* Chains are built one basis vector at a time as nested relative-RREF
  prefixes: at level `s`, rows are chosen in coordinates of a fixed
  complement of `F_s` inside the completed `T_s`, pivots increasing, each
  pivot column zero in the other rows. Levels whose local space is entirely
  forced are collapsed. Every flag has exactly one representation.
* Bound (Eq. pim-minmove-bound): for a prefix with `q` of the `ell_s` basis
  vectors of `T_s` chosen, spanning `B`,
  `r_lower = dim K - min(dim K, dim(B intersect K) + ell_s - q)`. Completed
  levels use their exact `T_s`, levels not yet started use `F_s`. The bound
  is exact at a complete flag.
* Best-first over `(lower bound, canonical prefix key)`. A node is dropped
  only when its bound strictly exceeds the best complete cost, so every
  tied layout is retained.
* A complete flag is materialized into all of its layouts (canonical
  complement of `T_{s+1}` in `T_s` per level, every ordered frame of each
  complement, every ordered frame of `T_h`): `layouts_per_flag` layouts with
  identical movement.
* Budget accounting (unchanged from the snapshot): one root node, plus one
  per evaluated prefix, plus one per materialized layout. Expansions and
  flag materializations are atomic. The default 20,672 = 1 + 511 + 20,160
  exactly covers the largest bounded domain.

Telemetry (`LayoutSearchTelemetry`) partitions the domain:
`visited_layouts + pruned_layouts + unresolved_layouts == expected_layouts`.

## 4. Domains and guarantees

| Implementation | Domain | Guarantee | On budget exit |
|---|---|---|---|
| `BOUNDED` | full carrier, at most 4 carrier bits and 3 owner levels (`MAX_BOUNDED_*`, hard limits) | complete: returns every minimum-movement layout of the legal domain (`EXHAUSTED` or `BOUND_PROVED`, `ties_complete=True`) | `NODE_BUDGET`, `optimality_proven=False`, ties = best complete layouts seen, `remaining_lower_bounds` = bounds of the unexplored frontier |
| `BOUNDED_MASKED` | active subset, at most 4 bits and 3 levels | exact only on exhaustion (`EXHAUSTED`); never prunes | `NODE_BUDGET`, remaining bound `0` per unvisited flag (the rank bound is not valid under a mask) |
| `SCALABLE` | full carrier, up to 32 bits, any number of levels | without a budget, or when the budget is not reached: every minimum flag (`EXHAUSTED` or `BOUND_PROVED`); `ties` holds one canonical representative per tied flag, each standing for `tie_multiplicity = layouts_per_flag` layouts with the same movement (`ties_complete=False`) | `NODE_BUDGET` with the best flags seen and the frontier bounds; the budget counts evaluated prefixes |

`search_min_movement_layout(problem, policy)` dispatches: masked problems go
to `BOUNDED_MASKED` (outside 4 bits / 3 levels they raise
`LayoutSearchDomainError`); full-carrier problems inside the bounded domain
go to `BOUNDED`; larger full-carrier problems go to `SCALABLE` when
`allow_scalable`, else raise `LayoutSearchDomainError`.

Feasible incumbent on a budget exit. If the budget ends the search before
any complete layout is reached, the result still carries one feasible layout:
the canonical completion (lowest eligible unit pivots, canonical complements
and RREF frames) of the lowest-bound frontier prefix, evaluated exactly and
marked `telemetry.fallback_incumbent=True`. In masked mode the first legal
layout of the remaining flags is used. This is an addition over the snapshot,
which could return no incumbent; it is outside the budget and never counted
as visited.

Scalable reach (measured, GEMV objective of `layout_fixture.py`, budget 256):
5, 6 and 8 bit one-level optima are proved; two and three levels end at the
budget; 16 and 32 bits end at the budget in under 1 s each with a fallback
incumbent and frontier bounds. The root bound at 16 and 32 bits is 0, so such
results certify nothing beyond feasibility. The snapshot timed out on every
8 to 32 bit case because its child cursor scanned every raw row
(`2**d` per expansion); the port generates only extendible rows (a row is
extendible exactly when enough unused columns remain for the later pivots),
which yields the identical child set (checked against the brute-force filter
for all 1,196 prefixes up to 6 relative bits during development).

## 5. Masked contract (paper Section VI)

For an active set `D`, the exact term is
`c_{j,s} b_j sum_e (|O_{j,s}(e)| - 1)` with
`O_{j,s}(e) = {Q_s(i) : i in D, A_j i = e}`;
`exact_active_owner_incidence` computes it directly (three active units of
one element give two extra copies, not three). The masked search enumerates
every flag and every layout of the bounded domain, evaluates each exactly,
disables pruning, and claims optimality only when it completes. A strict
reduction is legal in a masked layout when every element's active
iterations sit in one owner at every level; this admits layouts the
full-carrier rule (`K <= T_h`) would forbid. Padding is not a substitute:
on the triangular 3x3 GEMV mask with three levels the padded objective
reports 88 with 576 ties, the exact masked minimum is 22 with 64 ties, and
the padded ties really cost between 22 and 32.

`enumerate_layouts(problem)` yields every layout of the bounded domain with
its terms and cost (masked or not); it is the package-side enumerator for
independent checks.

## 6. LinearLayout bridge

`LinearLayout.to_gf2()` returns the layout as a `GF2Matrix` (columns = input
bits in insertion order, rows = output bits in `out_dims` order, low bits
first, matching `LinearLayout.matrix()`). `LinearLayout.from_gf2(matrix,
in_dims, out_dims)` is its inverse; dims are `(name, power-of-two size)`
sequences or dicts. `LayoutSearchResult.to_linear_layouts(in_dims=None,
out_dims=None)` converts every tie; the default `in_dims` are the hierarchy's
owner axes then local axes (named by their target diagnostic names, for
example `"group.group"`, `"local"`), the default `out_dims` is one
`"iteration"` dim. Example (paper Fig. ll-example, lane `l = 4m + k`):
`{"group.group": [(0, 1), (0, 2)], "local": [(1, 0), (2, 0)]}` over
`out_dims=("k", "m")`.

## 7. Reproduced numbers and tests

`$PY -m pytest tests/perf/test_gf2.py tests/perf/test_layout_*.py -q -p
no:cacheprovider`: 185 tests, about 50 s on this host.

| Paper claim | Reproduced | Test |
|---|---|---|
| 4 bits admit 20,160 one-to-one layouts | `gl_count(4)`, brute force over all 65,536 4x4 matrices, and `enumerate_layouts` all give 20,160 | `test_four_bits_admit_20160_one_to_one_layouts` |
| 315 chains with three one-bit compute levels | `expected_flags == 315`, 64 layouts per flag, 511 prefix nodes | `test_three_one_bit_levels_reduce_20160_layouts_to_315_chains` |
| pruned search visits 576 of 20,160 at two and three levels | 576 at 2 and 3 levels (4,032 at 1 level), equal to the complete tie set from an independent parity/set brute force over GL(4,2); the unpruned search agrees | `test_pruned_search_visits_576_and_matches_brute_force` |
| (interpretation, audit core.md Q3) | 576 = 9 tied flags x 64 layouts at 3 levels: the visited layouts are exactly the tie set, not 576 bound evaluations (148 prefix bounds are evaluated) | `test_576_is_nine_tied_flags_of_64_layouts_at_three_levels` |
| Sec. IV-A: x replicated 4x (16 placements, 12 extra copies), one y per group; swapped assignment reverses the counts | 12/0 and 0/12; per-group distinct elements 4 and 1; lane 6 stores W[1,2], needs x[2], feeds y[1] | `test_layout_movement.py` GEMV tests |
| budget exit returns best layout and remaining bounds | `NODE_BUDGET`, `optimality_proven=False`, non-empty bounds, feasible incumbent (searched or fallback) | `test_budget_exit_*`, `test_atomic_node_budgets_*` |
| 121,968 small masked layouts checked exhaustively | 12 cases (3 and 4 bits, 1 to 3 levels, rectangular and triangular masks); every layout checked by the ported integer-parity oracle; minima and tie counts equal the snapshot record; about 25 s, not marked slow | `test_121968_masked_layouts_match_the_integer_parity_oracle` |
| three active units need two extra copies | 2 (padding says 3) | `test_three_active_units_need_two_extra_copies` |
| strict order rules out splitting a reduction | strict `y` forces `T_1 = span(k)`: 576 legal layouts, all cost 12; no legal layout when no local bits remain | `test_strict_reduction_*` |

Snapshot telemetry witnesses (12 and 47 prefix nodes, 6 and 13 evaluated
bounds, 31 and 590 search nodes, the atomic budget table) are re-asserted to
show the traversal and budget accounting were preserved.

## 8. Wiring into placement selection, and what remains

### 8.1 What is wired (layout-construction stage)

`allo/spmw_layout_problem.py` derives the problem of one `@allo.work` kernel
group, and `spmw_autoschedule.rank_matcher_placements` runs the search once
per group and crosses its ties with the backend's decision keys.

* **Derivation** (module docstring has the full rules). The variables are the
  work-grid axes (`MatcherWorkScope.group_shape`) followed by the enclosing
  loops (`0 <= v < n`, step 1, static bound). They are named positionally
  (`work0`, `loop1`), so renaming changes nothing. A non-power-of-two extent
  pads to `ceil(log2 n)` bits and makes the problem masked (a box of active
  points). Each coordinate must be `c + sum a_v v` with `c` affine in the work
  id (fitted across the buckets and checked on every bucket). It must also be
  injective in its variables (mixed radix), so `K_j` is the coordinate
  subspace of the variables an access ignores. Inputs are `READ` and the
  loop-carried accumulator (through its output sink when it is a scalar) is
  `WRITE_ACCUMULATE`. `reassociation_allowed` comes from
  `extra["reduction_policy"]` (`may_regroup`); when no policy is present it
  defaults to `True` and records an assumption. Element bytes come from the
  memref element type. The owner levels come from `COMPUTE_LEVELS`: Samsung
  `pseudo_channel`/`pim`, AiM `channel`/`bank_group`/`bank`, UPMEM
  `dpu`/`tasklet` (an `active_extents` subgroup when the grid uses fewer
  units, for example 16 of 24 tasklets), and APU v1 `vr0.lane`. They are
  truncated to the levels the work grid covers; deeper levels are listed as
  `unmodelled_levels`. Rejected with the reason: non-affine or opaque
  coordinates, non-injective coordinates, non-unit loop steps, work
  dependence that is not affine, non-power-of-two work grids, and more than
  one matched op per work item. One recorded rule: an opaque contraction
  matrix row (`memref.load` of `row0 + i`) is taken equal to the single store
  row, the rule `spmw_samsung.samsung_contraction` already applies.
* **Weights.** `cost_per_level` is read from a cost-program hook
  `movement_cost_per_byte(level_name)`. No package cost program defines one,
  so every level weighs 1 per byte and the record says
  `movement_weights="uniform"`.
* **Search.** Problems inside the bounded domain (4 bits, 3 levels; masked
  or not) run the ported Algorithm 1 (`search_min_movement_layout`). Real
  kernels do not fit: the Samsung paper GEMV is 20 iteration bits over 4 + 3
  owner bits. `search_min_movement_layout` sends it to `SCALABLE`, which hits
  the 20,672-node budget after 25 s with only a fallback incumbent, because
  about 2^35 flags tie at the minimum. Larger problems therefore run
  `search_coordinate_min_movement` (addition to `layout_search.py`). It
  relies on a coordinate-dominance lemma (stated and proved in the module):
  with coordinate kernels, the pivot (initial) flag of any flag has no larger
  rank at any level. So a minimum is a coordinate flag, and every minimum
  movement profile is the profile of a coordinate flag. The search enumerates
  per-variable owner-bit classes. It is exact over all one-to-one layouts on
  a full carrier (`optimality_proven=True`) and exact over coordinate layouts
  only on a mask (`optimality_scope="coordinate layouts"`,
  `optimality_proven=False`). Ties are reported as movement *profiles*
  (owners per element, per access and level) with a coordinate-assignment
  multiplicity. The Samsung paper GEMV runs in 0.02 s (95 assignments visited
  under strict order, 581 under reassociation). Checked against Algorithm 1
  on random small problems: same
  minimum and same set of minimum profiles (`tests/perf/test_layout_selection_search.py`).
* **Selection.** Every tie class (cap `max_tie_candidates=64`) becomes a
  candidate for every decision key. The movement layout joins the backend's
  `LinearLayout` as a block-diagonal `product` over `mm.`-prefixed dims, so
  it enters the decision key and the backend's own dims are unchanged. When
  no tie class contains the program's layout, that layout is crossed as well,
  labelled `extra["movement"]="not-minimal"`. Unanalyzable groups keep the
  hand-built layouts, labelled `not-analyzed: <reason>`. Realizability: AiM,
  UPMEM and APU v1 emit one program per work item on the unit its work id
  names, so there a tie is realizable exactly when its owner flag equals
  the program's. Samsung (`ROW_MAP_TARGETS`) places contraction rows by the
  owner map (row-to-slot table, `samsung-execution.md` section 8): a tie is
  realizable when its owner coordinates depend on the output-row bits only
  (the accumulator's kernel lies in `T_h`, so no unit splits a row's
  reduction), the accumulator has one row coordinate and the rows are not
  masked (`row_map_unrealizable_reason`). On Samsung the realizable members
  of a minimum class that group rows differently are crossed as separate
  ties: the per-level BLOCK/CYCLIC row distributions (`row_distributions`,
  multiplicity 1, still counted in `tie_layouts`). Other ties are rejected
  before emission with the reason and recorded in
  `KernelRanking.movement_rejections`. Ties are broken by cycles, then by a
  structural key: layout text, knob and feature values, handle paths
  (`TIE_BREAK_RULE`, C-06). `KernelRanking.cost_ties` exposes the full
  equal-cycle set. `KernelRanking.movement` (`KernelMovement`) records
  status, implementation, termination, `optimality_proven`, scope,
  `incomplete`, min, program and chosen movement cost and label, tie count
  and multiplicity, visited/pruned, weights, assumptions and notes.
  `PlacementRanking.incomplete` is set on any node-budget exit.
  `MovementSelectionPolicy` (default: the ported budget of 20,672 nodes)
  is set per call or with `movement_selection_policy(...)`.

### 8.2 Findings

* **Samsung realizes ties; AiM, UPMEM and APU v1 realize only the program's
  work-grid owner map.** Samsung under strict FP16 (the default): the
  program's profile is the unique minimum (72,704 B, proven); its four row
  distributions (block/cyclic per level; the program is block x block) are
  distinct owner maps with that movement, all realizable, so
  `tie_count == 4`, `chosen_label == "minimal"` and 16 candidates are scored
  (4 ties x 4 keys). The cost program and PIMSimulator both rank the ties
  equal (vendor REDUCE 4,435 cycles, ordered program 26,807, for the
  identity and the cyclic map, outputs correct), and the structural
  tie-break selects the program map, so the default program is unchanged.
  Samsung under reassociation: the minimum is 48,128 against the program's
  72,704 (one or two `k` bits split across PIM units); both tie classes
  split a row's reduction and are rejected with that reason (reduction
  policy). AiM (`reassociate`): 5,120
  against 15,872; 2 classes are rejected. UPMEM 64 x 12 rows (masked): 14,848
  against 32,256; 2 classes are rejected. Selected programs, cycles and
  emitted commands are identical to HEAD on all five GEMVs (Samsung 253, 381,
  253; AiM 255; UPMEM 359,488 virtual cycles).
* **Backend layouts are not in the iteration-to-owner coordinate space.**
  Samsung's `swizzled` maps `grf`/`bank`/`tile` and its emitter reads fibers
  from `extra["fibers"]`, not from `Placement.layout`. AiM's layout maps
  `k`/`work_bank`/`lane_bank` to `bank` and is read for SBK bank expansion.
  UPMEM checks only that a `LinearLayout` is present. A tie can therefore
  never "equal the swizzled layout"; the product keeps both.
* A user `compile(layout=...)` whose carried movement part is not the
  program's owner map is executed by Samsung contractions (rejected at
  emission with the reason when not realizable, and for elementwise
  kernels); AiM, UPMEM and APU v1 ignore it (needs the C-01 legality pass).

### 8.3 What remains

* Samsung realizes row-placement ties, but no tie is cycle-distinguishable
  there (every slot is issued in lockstep). Ties that split `k` across PIM
  units need a partial-sum combine (reduction policy) on every backend.
  AiM, UPMEM and APU v1 would need a per-unit row re-map in their lowering;
  until then their ties reach the ranking and are rejected for the stated
  reason.
* Movement weights from the cost programs (`movement_cost_per_byte`).
* Levels below the work grid (AiM banks, UPMEM tasklets, Samsung bank
  fibers) are chosen inside the backends and are not part of the problem.
* Masked problems: the lemma is not proved for masks, so the coordinate
  search claims only coordinate optimality there.
* Kernels with several matched ops per work item (MAC + AF) and opaque
  coordinates other than the contraction matrix row are not analyzed.
* Target legality beyond the owner structure (capacity, alignment,
  contiguity: `L_legal` in Eq. pim-minmove-objective) is not modelled; the
  snapshot's SLS6 target-constraint callback protocol was not ported.

## 9. Dropped from the snapshot

* `*_sha256` fingerprints, `manifest()` replay authentication, outcome
  re-validation by re-running the search, task-number rejection enums and
  `type(x) is not T` exact-type guards.
* `TypedOperandReuseEntry` / `PrelayoutIncidenceKind` and the typed-semantics
  stack (replaced by `LayoutAccess`), `IterationCarrierAxis` (the carrier is
  the hierarchy's bit count), and `operand_offset` (an XOR offset relabels
  elements and does not change any count).
* The snapshot's 6-bit cap on the rank-incidence helpers (they are
  polynomial).
* `coordinate_flags_only`, the sub-byte `element_bits` field (use a
  `Fraction` `element_bytes`), the SLS6 target-constraint protocol, and the
  scalable masked search with an admissible coset bound
  (`scalable_masked_layout_*`): the release ships the paper's simpler masked
  contract.
* Flag ordinals and per-candidate structural keys; ties are ordered by flag
  key then matrix rows.
* `layout_problem.py` was copied nearly verbatim (only black formatting and
  the target bridge were added); `StorageAxis` and the validation codes are
  kept.
