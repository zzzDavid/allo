# Minimum-movement layout search (paper Section IV)

Status: implemented as a standalone package library (`allo/pim/gf2.py`,
`allo/pim/layout_problem.py`, `allo/pim/layout_movement.py`,
`allo/pim/layout_search.py`). Not yet wired into the autoscheduler (see
Section 8). Implements steps 1 to 4 and 7 of the salvage plan in
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

## 8. What remains

* Wiring into `spmw_autoschedule.rank_matcher_placements` as the
  layout-construction stage: every tied layout (or flag representative)
  becomes a `Placement.layout` candidate crossed with the knobs, and a
  `NODE_BUDGET` termination must mark the ranking incomplete (salvage step 6).
* Deriving `LayoutAccess` rows from `MatchedOp` (operand index maps,
  `enclosing_loops`, power-of-two trip counts via
  `spmw_tripcount.resolve_trip_count`, work-grid bits from
  `MatcherWorkScope.group_shape`) and `reassociation_allowed` from a user
  numeric policy plus the accumulator role (salvage step 5). Non-affine or
  non-power-of-two accesses must be rejected or masked explicitly.
* Deriving `cost_per_level` from the cost program (per-byte movement cost
  across each owner level) instead of passing it by hand.
* Generating `active_points` from non-power-of-two loop bounds.
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
