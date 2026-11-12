<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Evidence beyond the original examples (gate G6)

This document maps the G6 probe list in `TASK.md` onto the tests that exist in
the tree at `d13cf08` ("[Docs] Release ledger round 4") on branch `tenon`. It
was written from the test sources (docstrings, bodies, assertions) and from
`docs/release/VALIDATION.md`, `ISSUES.md`, `design/*.md` and `audit/*.md`. No
test was executed for this document; every test name below was checked against
`pytest --collect-only -q` on the same tree (948 items under `tests/pim`,
`tests/perf`, `tests/test_literal_typing.py`, `tests/test_global_vars_scope.py`,
`tests/test_c_backend.py`, of which 14 exist only in the uncommitted working
tree, see "Working-tree caveat").

The document is deliberately candid. Where a mechanism has no probe for a G6
category the cell says `none`. Where a probe exists but is weaker than the
gate asks (compile-only, stub-backed, pinned to the paper shape) the cell says
so. Probes that were executed during the audit but are not regression tests
(scripts under `/tmp/tenon-release/probes/` and `/tmp/upmem-fix/`) are listed
separately as "audit probes" and never counted as coverage.

## Working-tree caveat

Other agents were editing the tree while this document was written. At the
time of writing `git status` showed `tests/pim/test_upmem_generality.py`
modified (HEAD collects 62 tests from it, the working tree 68) and
`tests/perf/test_apu_g2_value_range.py` untracked (8 tests, G2-09/G2-10
readback range checks, synthetic carriers only). Working-tree-only tests are
marked **[WT]** and must not be counted until committed. The three
**[WT]** UPMEM tests replace `test_minus_form_is_rejected_by_the_matcher_with_its_reason`
(HEAD only).

## Legend

G6 probe categories (rows of every mechanism table):

| Key | Probe |
|---|---|
| a | Renames of functions, arguments, loops, files, diagnostic labels; normalized decisions compared |
| b | Operation or constant change at a fixed shape (exposes shape-only dispatch) |
| c | Legal shapes around tile, capacity and alignment boundaries; rectangular and irregular cases |
| d | Variation of target parameters advertised as configurable |
| e | Negative and extreme integers, overflow, shift widths |
| f | Nonzero (carried) accumulator initializers |
| g | All-zero results reported as results, not as success without outputs |
| h | Masks and guards (predicates, padding masks, padded lanes) |
| i | Alias cases (same buffer as two operands, or an input aliased with an output) |
| j | Strict versus explicitly reassociative floating-point reductions |
| k | Composed operators and producer/consumer stages (intermediates, slices, transfers, host partitions) |
| l | Two distinct feasible schedules or layouts traced through addressing, transfers and the execution path |
| m | Cost or resource perturbation whose predicted effect reaches compilation or selection |
| n | Removal of archived result directories and oracle dependencies; execution still computes results |
| o | Explicit rejection of unsupported neighbors |

Oracle abbreviations: `np` NumPy reference (FP32 or int64 unless stated);
`fp16-order` source-order FP16 model (sequential FP16 accumulation);
`int-parity` independent integer/parity oracle; `brute` brute-force
enumeration; `sim` simulator readback compared to the stated reference;
`dev` device readback compared to the stated reference; `pin` pinned golden
value (cycles, SHA, trace text); `struct` structural assertion on the
generated program (no numerics); `exc` exception expected.

Input abbreviations: `rng(s)` fresh `np.random.default_rng(s)` draw with the
stated seed (fixed seed, data independent of the plan); `fresh` seed from
`time_ns` unless `TENON_APU_G2_TEST_SEED` is set; `fixed` hand-written vectors;
`archive` inputs read from `tenon-artifacts` or benchmark tables; `none` no
data (compile or plan only).

Path abbreviations: `CO` compile-only; `V` `backend="virtual"` cost model;
`HC` host C compiled with clang; `JIT` Allo LLVM JIT on the host; `NP`
Python/NumPy execution of the workload inside the package (the APU v1 vector
"functional" route); `SIM` simulator; `EMU` monkeypatched emulated card;
`DEV` device. Gates are named in section 2.

Paths are relative to `experiments/allo/`. `pim/` means `tests/pim/`,
`perf/` means `tests/perf/`.

## 1. Mechanism tables

### 1.1 Matcher coverage and accumulator init

Files: `pim/test_match_patterns.py`, `pim/test_match_coverage.py`,
`perf/test_matcher_value_identity.py`, `perf/test_compile_interface.py`.
No test in this group uses random data or executes anything; the numeric
consequences of matcher decisions are tested per backend (1.7 to 1.11).

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `pim/test_match_coverage.py::test_rename_only_variant_has_same_classification` (store-category counter plus (op, init, sink index) equal); `perf/test_matcher_value_identity.py::test_memref_renames_preserve_value_ids_and_residency_domain`; `perf/test_compile_interface.py::test_frontend_retains_stable_scopes_across_rename_relocation_and_reorder`, `::test_duplicate_isomorphic_groups_use_explicit_abi_identity_not_occurrence`, `::test_structural_mapping_boundaries_define_relocation_invariant_apu_scopes`, `::test_non_kernel_fallback_identity_is_rename_and_reorder_invariant`. Comparison stops at classification, liveness and scope identity; the selected placement is compared per backend (1.5, 1.8, 1.10, 1.11, 1.12) | struct | none | CO, V |
| b | `pim/test_match_patterns.py::test_exp_is_not_a_transparent_cast`, `::test_negation_is_not_a_transparent_cast`, `::test_max_elementwise_and_reduction`, `::test_compare_matches_mirrored_predicate`, `::test_const_params_bind_compile_time_coefficients`, `::test_regression_update_constants`, `::test_sibling_loops_with_equal_ssa_names_trace_their_own_values`; `pim/test_match_coverage.py::test_unmatched_division_is_rejected`, `::test_unmatched_update_after_copy_is_rejected`; `perf/test_compile_interface.py::test_structural_mapping_boundaries_keep_distinct_kernel_groups`, `::test_later_pid_body_mutation_invalidates_complete_group_manifest` | struct, exc | none | CO |
| c | `pim/test_match_patterns.py::test_vector_width_contiguous_vs_strided` (stride-2 access gives width 1). No tile or alignment sweep at this layer | struct | none | CO |
| d | `pim/test_match_patterns.py::test_vector_width_contiguous_vs_strided` (op `vector_width` 16 versus 1 on an inline target). Weak: one parameter, synthetic target | struct | none | CO |
| e | `pim/test_match_patterns.py::test_const_params_bind_compile_time_coefficients` (alpha 2, beta -1), `::test_regression_update_constants` (-32, `>> 8`); binding only, no overflow | struct | none | CO |
| f | `pim/test_match_coverage.py::test_paper_gemv_zero_init_and_output_sink` (zero), `::test_nonzero_accumulator_initializer_is_rejected` (`acc = 0.5` raises), `::test_initializer_inside_reduction_loop_is_rejected`, `::test_carried_accumulator_selects_the_ordered_program_on_samsung` (structural: temporal layouts), `::test_carried_accumulator_rejected_where_the_device_zeroes_it` (AiM) | struct, exc | none | CO, V |
| g | none | | | |
| h | `pim/test_match_patterns.py::test_indirect_accumulate_records_index_and_guard`, `::test_select_compact_shares_one_guard`, `::test_select_matches_cmpf_ogt`, `::test_argmin_binds_loop_index` | struct | none | CO |
| i | Adjacent only: `pim/test_match_patterns.py::test_scatter_add_checks_accumulator_index` (read `sums[b[p]]` versus write `sums[a[p]]` gives no match); `perf/test_matcher_value_identity.py::test_ambiguous_exact_handoff_with_multiple_producers_fails_closed`. No test passes one NumPy buffer as two operands or as input and output | struct | none | CO |
| j | Policy stamping: `pim/test_reduction_policy.py::test_match_workload_stamps_accumulating_matches` (see 1.3) | struct | none | CO |
| k | `perf/test_compile_interface.py::test_compile_binds_repeated_shape_metrics_by_retained_value` (first_stage to tmp to second_stage, metrics keyed by value identity, real AiM match); `pim/test_match_patterns.py::test_select_compact_shares_one_guard` | struct | none | CO |
| l | none (not a matcher concern; see 1.6 and the backends) | | | |
| m | none | | | |
| n | none | | | |
| o | `pim/test_match_coverage.py::test_compile_for_target_rechecks_an_unchecked_trace`; `perf/test_compile_interface.py::test_structurally_ambiguous_duplicate_groups_fail_closed`, `::test_incomplete_retained_adapter_contract_fails_closed`, `::test_missing_retained_grid_coordinate_fails_closed`; `pim/test_match_patterns.py::test_indirect_accumulate_records_index_and_guard` (unguarded op rejects a guarded store) | exc | none | CO |

### 1.2 Axis roles and work grid

Files: `pim/test_aim_axis_roles.py`, `pim/test_work_grid.py`. The C-16 repair
(`store_index_vars` for `y[row0 + r]`) has no direct test; it is exercised
indirectly by `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_carried_accumulator_through_the_public_api` (1.7).

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `pim/test_aim_axis_roles.py::test_gemv_spellings_make_identical_decisions` (`row0`, inline `W[ch*ROWS+i, j]`, single-item, and a fully renamed region/kernel/args/wid/loops/accumulator: whole command tuple, record, MAC counts and estimate equal); `::test_gemv_spellings_simulate_identically` (simulated cycles equal for `row0`, inline, single; the renamed spelling is not simulated); `::test_output_loop_is_not_a_batch_axis` | struct, sim cycles | none | V; SIM (aim_sim_gate) |
| b | `pim/test_aim_axis_roles.py::test_unsupported_reduction_nest_is_rejected` (reduction split jo/ji at the same shape is rejected). No operation change | exc | none | V |
| c | `pim/test_aim_axis_roles.py::test_partitioned_96_row_gemv_issues_logical_work_plus_declared_padding` (96 rows, mapping [3], issued versus logical MACs re-derived in the test); `pim/test_work_grid.py::test_work_grid_rule_table[...]` (17 parametrizations over four targets, 9 accept, 8 reject), `::test_samsung_4x4_grid_is_an_error`, `::test_samsung_full_grid_compiles`, `::test_samsung_single_item_elementwise_compiles`, `::test_virtual_only_target_accepts_single_item` | struct, exc, MAC-count invariant | none | V, unit |
| d | `pim/test_work_grid.py::test_work_grid_rule_table[...]` (unit factors on synthetic stub targets); `pim/test_aim_axis_roles.py::test_target_geometry_requires_a_reuse_slot` (`slots=0` rejected). No shipped target is rebuilt with another geometry and compiled | struct, exc | none | unit |
| e | none | | | |
| f | none here (carried `y[row0 + r]` executed in 1.7) | | | |
| g | none | | | |
| h | none | | | |
| i | none | | | |
| j | none | | | |
| k | `pim/test_aim_axis_roles.py::test_vector_indexing_result_loop_is_the_batch_axis` (QK roles (h, s, d), MAC invariant); `pim/test_work_grid.py::test_multi_kernel_grids_are_checked_per_kernel` | struct | none | V, unit |
| l | none | | | |
| m | none | | | |
| n | none | | | |
| o | `pim/test_aim_axis_roles.py::test_coverage_check_rejects_overissued_contraction`, `::test_codegen_rejects_a_record_that_misstates_the_program` (8x over-issue raises `AimCoverageError`), `::test_eval_sym_refuses_unbound_unit_ids`; the 8 rejecting rows of `pim/test_work_grid.py::test_work_grid_rule_table[...]` | exc | none | unit, V |

### 1.3 Reduction policy

Files: `pim/test_reduction_policy.py`,
`pim/samsung_hbm_pim/test_samsung_reduction_order.py`,
`perf/test_apu_v1_reduction_policy.py`, `pim/test_aim_token_partition.py`,
`perf/test_layout_search.py`, `perf/test_layout_movement.py`.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | none | | | |
| b | none (the policy's dtype dependence is checked in `pim/test_reduction_policy.py::test_policy_semantics`, which is not an operation change) | | | |
| c | `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_ordered_gemm_columns` (2048x64, N=2), `::test_strict_default_runs_the_ordered_program` (4096 rows padded to 8192, padding fraction 0.5) | sim versus fp16-order, bit-exact | rng(4), rng(0) | SIM (samsung_sim_gate) |
| d | none | | | |
| e | none | | | |
| f | `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_ordered_program_carries_a_nonzero_initial_value` (initial 0.5, through the internal `_samsung_run_temporal`, not the compiled callable), `::test_carried_accumulator_through_the_public_api` (random nonzero `y0` through `allo.compile`) | sim versus fp16-order from the initial value, bit-exact | rng(0) + fixed 0.5; rng(11) | SIM (samsung_sim_gate) |
| g | none | | | |
| h | none | | | |
| i | none | | | |
| j | Policy semantics: `pim/test_reduction_policy.py::test_policy_semantics`, `::test_match_workload_stamps_accumulating_matches`, `::test_samsung_strict_prunes_regrouping_placements` (three families pruned with reasons, exact 5-word TEMPORAL CRF), `::test_samsung_reassociate_ranks_both_families`, `::test_result_records_the_policy`, `::test_strict_permits_splitting_an_integer_reduction` (i32 UPMEM 12-tasklet split kept), `::test_aim_float_reduction_needs_permission`. Executed: `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_strict_default_runs_the_ordered_program` (bit-exact to fp16-order, 26,807 cycles pinned), `::test_reassociating_program_differs_from_source_order` (more than M/2 rows differ bitwise from fp16-order, all within 1e-2 of FP32, fewer cycles). APU v1 (NumPy route): `perf/test_apu_v1_reduction_policy.py::test_strict_ranks_only_source_order_plans`, `::test_reassociate_admits_tree_plans`, `::test_functional_numerics_follow_the_selected_plan` (strict bit-equal to a NumPy FP16 source-order loop; reassociate bit-equal to a NumPy halving-tree model that the test itself says is not checked against the device), `::test_integer_contraction_keeps_every_plan`. AiM: `pim/test_aim_token_partition.py::test_knob_domain_and_reduction_policy` (P domain collapses to [1] under strict bf16). Layout search: `perf/test_layout_search.py::test_strict_reduction_forbids_owner_bits_that_split_its_kernel`, `::test_strict_reduction_with_no_room_inside_a_lane_has_no_legal_layout`, `perf/test_layout_movement.py::test_strict_reduction_is_illegal_when_an_owner_bit_splits_its_kernel` | struct; sim versus fp16-order and np; NP versus fp16-order and tree model; brute | none; rng(0); rng(0) | CO, V; SIM; NP; unit |
| k | none | | | |
| l | `pim/test_reduction_policy.py::test_samsung_strict_prunes_regrouping_placements` versus `::test_samsung_reassociate_ranks_both_families` (different CRF and ranking, virtual); executed difference in bytes and cycles: `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_reassociating_program_differs_from_source_order` | struct; sim | none; rng(0) | V; SIM |
| m | none | | | |
| n | none | | | |
| o | `pim/test_reduction_policy.py::test_samsung_forced_regrouping_layout_fails_under_strict`, `::test_aim_float_reduction_needs_permission`, `::test_policy_semantics` (bad names); `perf/test_apu_v1_reduction_policy.py::test_strict_ranks_only_source_order_plans` (forced tree layouts raise `ReductionOrderError`) | exc | none | V, CO |

Doc note: `design/reduction-policy.md` section 4 still says `compile_for_target`
refuses carried accumulators on Samsung; the code (`allo/spmw_codegen.py`,
"Samsung is not listed") and `pim/test_match_coverage.py::test_carried_accumulator_selects_the_ordered_program_on_samsung`
disagree. The sentence is stale.

### 1.4 Legality

Files: `perf/test_legality.py`, plus the backend-specific legality tests
named below.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | none | | | |
| b | none | | | |
| c | `perf/test_legality.py::test_register_slot_capacity`, `::test_identity_rejects_unmatched_input_dims`; `pim/test_aim_program.py::test_reuse_and_storage_are_validated_against_target_capacity` (33 reuse slots over a 32-slot window; bank row overflow at 16382); `pim/test_upmem_generality.py::test_tasklet_private_wram_rejects_on_its_wram_footprint`, `::test_unrealizable_kernel_names_the_template_constraint`; `perf/test_upmem_physical.py::test_illegal_dma_or_unsupported_physical_modes_fail_closed[...]` (5 rows); `perf/test_upmem_irregular.py::test_legality_checks_fail_before_emitting_an_invalid_translation_unit` (15 `ValueError` sites); `perf/test_apu_g2_source_route.py::test_composed_record_beyond_one_carrier_names_the_capacity` | struct, exc | synthetic | unit, CO |
| d | `perf/test_legality.py::test_lanes_metric_reads_the_placement_register`, `::test_samsung_mac_rule_has_no_lanes_default` (lanes come from the register; the value is never varied) | struct, exc | none | unit, V |
| e | none | | | |
| f | none | | | |
| g | none | | | |
| h | none | | | |
| i | Register-level aliasing only: `perf/test_legality.py::test_dst_pattern_and_accumulator_aliasing`, `::test_register_slot_capacity`, `::test_samsung_illegal_user_layouts_are_rejected[all_on_grf_b]`. No memref alias | struct, exc | synthetic | unit, CO |
| j | none (see 1.3) | | | |
| k | none | | | |
| l | none | | | |
| m | none | | | |
| n | none | | | |
| o | `perf/test_legality.py::test_sibling_register_is_unreachable`, `::test_host_memory_is_unreachable_without_a_move`, `::test_src_pattern_violation`, `::test_register_without_declared_fill`, `::test_cross_subtree_declaration_is_reported`, `::test_src_arity_must_fit_fn`, `::test_samsung_srf_is_not_a_declared_operand`, `::test_samsung_illegal_user_layouts_are_rejected[acc_on_grf_a|all_on_grf_b|y_on_srf|x_on_host_dram]`, `::test_autoschedule_filters_illegal_candidates`, `::test_autoschedule_raises_when_every_candidate_is_illegal`, `::test_shipped_targets_declarations_are_legal`; structural unit comparison across target instances (C-15): `pim/test_cent_aim_workloads.py::test_probe_and_emit_derive_the_same_replicated_contraction`; `pim/test_attacc_virtual.py::test_illegal_layout_is_rejected`; `perf/test_apu_g2_source_route.py::test_unsupported_forms_raise_a_finite_error[...]` (9 rows) | struct, exc | synthetic, none | unit, CO, V |

### 1.5 Movement stage and layout search

Files: `pim/test_layout_selection.py`, `perf/test_layout_movement.py`,
`perf/test_layout_problem.py`, `perf/test_layout_search.py`,
`perf/test_layout_search_masked.py`, `perf/test_layout_selection_search.py`,
`perf/test_gf2.py`. Important limit: every compiling test in this group asserts
that `cmds` and `estimate().cycles` are identical with the movement stage
disabled (no backend can realize an owner map other than its own work-grid
map, ISSUES C-11). The stage therefore never changes generated code.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `pim/test_layout_selection.py::test_rename_only_variant_has_the_same_movement` (compiled Samsung GEMV with renamed region, args, work function, wid, loop vars, accumulator; min cost, program cost, tie count, tie layouts, chosen label equal); `perf/test_layout_selection_search.py::test_rename_only_variant_gives_the_same_problem`; `perf/test_layout_problem.py::test_owner_and_local_manifest_is_exact_and_rename_insensitive`, `::test_forced_local_diagnostics_do_not_depend_on_names`, `::test_noncontiguous_diagnostic_is_alpha_rename_invariant_and_json_stable` | struct (normalized decision) | none | V, unit |
| b | Access-map changes only: `pim/test_layout_selection.py::test_non_affine_access_falls_back_not_analyzed` (permuted row blocks become `not-analyzed`), `perf/test_layout_selection_search.py::test_unsupported_accesses_are_rejected_with_the_reason[...]`. No operation or constant change | struct, exc | none | V, unit |
| c | `perf/test_layout_selection_search.py::test_non_power_of_two_rows_become_a_masked_problem` (96 rows on 8 of 64 DPUs); `pim/test_layout_selection.py::test_aim_and_upmem_report_movement_without_changing_cycles[upmem]` (12 rows per DPU, masked); `perf/test_layout_problem.py::test_upmem_24_tasklets_fail_closed_without_declared_subgroup`, `::test_non_power_of_two_extent_is_rejected_with_a_clear_error`; `perf/test_layout_search.py::test_problems_outside_every_domain_are_rejected` (4 to 5 bit domain boundary), `::test_scalable_budget_exit_on_32_bits_returns_a_feasible_incumbent`; rectangular and triangular masks in `perf/test_layout_search_masked.py::test_121968_masked_layouts_match_the_integer_parity_oracle`. Capacity and alignment are not modelled by the stage (`design/layout-search.md` section 8.3) | struct, brute, int-parity | exhaustive small domains, none | unit, V |
| d | `perf/test_layout_problem.py::test_target_data_adapts_to_exact_owner_bits_and_nested_dimensions[...]` (5 hand-written capability records), `::test_samsung_unit_tree_gives_channel_then_pim_levels`, `::test_apu_v1_register_lane_axis_is_a_register_owner_level`, `::test_toy_target_matches_the_paper_example_hierarchy`, `::test_non_power_of_two_extent_is_rejected_with_a_clear_error` (`active_extents` 16 of 24). No sweep of one target's parameter through compile | struct | fixed, real targets | unit |
| e | Validation only: `perf/test_layout_movement.py::test_scaling_is_exact_and_supports_sub_byte_elements` (negative and zero cost rejected), `::test_layout_access_rejects_malformed_fields[...]`; the masked oracle in `perf/test_layout_search_masked.py::test_121968_masked_layouts_match_the_integer_parity_oracle` uses signed mod-65536 partial sums | exc, int-parity | fixed | unit |
| f | none (synthetic groups always use `accumulator_init: "zero"`) | | | |
| g | none | | | |
| h | `perf/test_layout_movement.py::test_2740_raw_parity_layout_mask_cases_match_the_exact_active_count` (2,740 exhaustive cases), `::test_three_active_units_need_two_extra_copies_not_three`, `::test_masked_count_equals_the_rank_formula_on_the_full_carrier`; `perf/test_layout_search_masked.py::test_121968_masked_layouts_match_the_integer_parity_oracle` (12 cases, 3 and 4 bits, min and full tie set equal the brute force), `::test_padding_overstates_cost_and_admits_non_optimal_ties`, `::test_masked_strict_reduction_is_checked_per_layout`; `perf/test_layout_selection_search.py::test_coordinate_search_on_masks_is_exact_over_coordinate_layouts` (`random.Random(11)`, 120 draws) | brute, int-parity, hand-computed | exhaustive, random seeds | unit |
| i | none (`perf/test_layout_problem.py::test_collections_are_frozen_copies_not_mutable_aliases` is Python object aliasing, not memref aliasing) | | | |
| j | `perf/test_layout_search.py::test_strict_reduction_forbids_owner_bits_that_split_its_kernel` (brute-force min and tie set; relaxed cost below strict), `::test_strict_reduction_with_no_room_inside_a_lane_has_no_legal_layout`; `pim/test_layout_selection.py::test_samsung_paper_gemv_movement_stage` (strict, min = program = 72,704) versus `::test_samsung_unrealizable_ties_are_rejected_before_emission` (reassociate, min 48,128 versus program 72,704, all ties unrealizable); `perf/test_layout_selection_search.py::test_samsung_shaped_problem_minimum_and_ties`, `::test_coordinate_search_matches_algorithm_1_on_full_carriers` (`random.Random(7)`, 150 draws), `::test_missing_policy_assumes_reassociation_and_says_so` | brute, hand-computed, cross-implementation | exhaustive, random, none | unit, V |
| k | none (multi-op kernels are `not-analyzed`; no test of that rejection) | | | |
| l | Algebra only: `perf/test_layout_movement.py::test_paper_example_group_counts_and_replication` versus `::test_swapped_assignment_reverses_the_counts` (x rank 2 / 12 extra versus y 12). No test shows two layouts reaching different addressing, transfers or a run; the compiled tests prove the opposite (identical `cmds`) | hand-computed | fixed | unit |
| m | Pure search: `perf/test_layout_search.py::test_equal_weights_prefer_the_mixed_layout_over_both_paper_extremes` versus `::test_costly_partials_select_the_paper_layout_and_reject_the_swap` (y weight 1 versus 4 changes the brute-force-checked tie set), `::test_zero_weight_equality_exhausts_every_flag`. Budget only reaches compilation: `pim/test_layout_selection.py::test_tiny_budget_is_incomplete_with_a_feasible_incumbent`, `perf/test_layout_search.py::test_budget_exit_keeps_the_incumbent_and_reports_the_frontier`. Movement weights are always uniform in production, so no weight change reaches `allo.compile` | brute, struct | fixed | unit, V |
| n | none | | | |
| o | `perf/test_layout_search.py::test_problems_outside_every_domain_are_rejected`, `::test_policy_rejects_values_outside_the_implemented_domain[...]`; `perf/test_layout_problem.py::test_negative_geometry_cases_fail_with_exact_reason[...]`, `::test_malformed_types_fail_with_typed_structural_identity[...]`, `::test_invalid_active_subgroups_fail_closed[...]`; `pim/test_layout_selection.py::test_samsung_unrealizable_ties_are_rejected_before_emission` | exc | fixed | unit, V |

Search optimality evidence (not a G6 row but load-bearing): the search minimum
and complete tie set equal an independent brute force only on 3 and 4 carrier
bits (`perf/test_layout_search.py::test_pruned_search_visits_576_and_matches_brute_force[...]`,
`::test_nested_forced_domain_matches_brute_force`,
`perf/test_layout_search_masked.py::test_121968_masked_layouts_match_the_integer_parity_oracle`).
From 5 bits up the evidence is cross-implementation
(`::test_scalable_search_agrees_with_the_bounded_search[...]`) or hand-computed
(the 20-bit Samsung shape). The GF(2) algebra is exhaustively checked
(`perf/test_gf2.py`, all 65,536 4x4 matrices, dimensions 0 to 4).

### 1.6 Selection and materialization

Files: `perf/test_joint_selection.py`, `perf/test_matcher_ranked_placement.py`,
`perf/test_schedule_search_core.py`, `perf/test_resource_model.py`,
`perf/test_cost_fingerprint.py`, `perf/test_run_result_contract.py`,
`pim/test_host_program_analysis.py`.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `perf/test_compile_interface.py::test_public_compile_selects_matcher_adapter_from_retained_ir` (adapter chosen from retained IR, not from the `mappings` label), `::test_non_kernel_digit_suffix_is_not_a_structural_work_id`, `::test_apu_work_grid_consumes_typed_coalesced_scope_without_legacy_tags` | struct | none | stub, unit |
| b | none | | | |
| c | none | | | |
| d | `perf/test_resource_model.py::test_capacity_is_declared_on_existing_target_handles` (declared capacities read back; weak) | struct | synthetic | unit |
| e | none | | | |
| f | none | | | |
| g | `perf/test_run_result_contract.py::test_executed_all_zero_output_is_copied` (stub backend: an all-zero output is copied over a prefill of 7), `::test_non_executing_runs_leave_arrays_untouched[virtual|simulator-timing-only]` | struct | fixed | stub |
| h | none | | | |
| i | none | | | |
| j | none | | | |
| k | `perf/test_joint_selection.py::test_routes_agree_and_program_cycles_is_the_estimate[no-host|launch-schedule]` (two-kernel chain), `::test_eager_materialization_selects_the_lazy_program`; `perf/test_matcher_ranked_placement.py::test_ranked_evaluation_count_is_bounded[mvt_case]` (independent recount of graph builds); `pim/test_host_program_analysis.py::test_launch_schedule_orders_host_transfers_between_kernels`, `::test_legacy_order_schedule_matches_unscheduled_graph` | struct, independent recount | none | V |
| l | Virtual only: `perf/test_joint_selection.py::test_joint_evaluation_beats_conflicting_per_group_minima` (restage/restage versus resident/resident, 70-cycle difference, `program_cycles == evaluate(graph)`), `perf/test_matcher_ranked_placement.py::test_ranked_is_no_worse_than_uniform_per_bucket_argmin`; `pim/test_host_program_analysis.py::test_hoisted_scatter_coalesces_into_one_group` versus `::test_scatter_in_loop_is_not_coalesced`, `::test_residency_is_read_from_program_structure`, `::test_cost_and_run_share_one_analysis`. Nothing here reaches a simulator | struct, closed form | none | V, unit (the cited `test_hoisted_scatter_coalesces_into_one_group` tested the separate residency analysis removed with C-10 in release round 193; the joint-selection evidence stands on `perf/test_joint_selection.py`) |
| m | `perf/test_joint_selection.py::test_joint_evaluation_beats_conflicting_per_group_minima` (a test cost spec with declared coupling flips the chosen residency), `::test_over_budget_selection_is_labelled_heuristic` (`joint_budget=3`); `perf/test_cost_fingerprint.py::test_referenced_global_calibration_changes_fingerprint` (fingerprint only); `perf/test_resource_model.py::test_independent_handle_instances_overlap`, `::test_shared_handle_instance_serializes`, `::test_dependency_prevents_overlap_on_independent_handles` (estimate only). Backend cost perturbations: 1.8 (AiM), 1.9 (UPMEM), 1.11 (Gemini-II), 1.12 (AttAcc) | struct, closed form | none | V, unit |
| n | none | | | |
| o | `perf/test_matcher_ranked_placement.py::test_ranked_falls_through_infeasible_codegen`; `perf/test_joint_selection.py::test_emitter_error_on_top_candidate_falls_through[SamsungUnsupported|ValueError]`, `::test_no_materializable_candidate_is_a_hard_error_listing_rejections`, `::test_per_bucket_fallback_probes_the_assembled_kernel`; `perf/test_schedule_search_core.py::test_exhaustive_no_feasible_schedule_is_a_hard_failure_with_diagnostics`; `perf/test_run_result_contract.py::test_executed_run_without_gathered_output_fails[device|simulator|functional]`, `::test_executed_run_with_other_output_fails`, `::test_executed_run_without_bound_output_fails`, `::test_non_executing_result_may_not_carry_outputs`; `perf/test_simulator_unavailable.py::test_samsung_run_raises_when_pim_driver_is_missing`, `::test_docker_timeout_is_reported_separately_from_missing_image`, `::test_run_without_a_hook_raises_not_implemented` | exc | none | V, stub, unit |

### 1.7 Samsung REDUCE, TEMPORAL and elementwise

Files: `pim/samsung_hbm_pim/test_samsung_lowering.py`,
`pim/samsung_hbm_pim/test_samsung_conductor_runs.py`,
`pim/samsung_hbm_pim/test_samsung_reduction_order.py`,
`pim/samsung_hbm_pim/gemv/test_gemv_samsung_hbm_pim.py`,
`pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py`,
`pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py`,
`pim/test_host_transfer_codegen.py`, `pim/test_host_transfer_dispatch.py`,
`perf/test_samsung_performance.py`, `pim/test_gemv_all_targets.py`,
`pim/test_paper_golden.py`. Simulator tests compare readback against FP32
`W@x` (rtol/atol 1e-2) or bit-exactly against the fp16-order model or NumPy
FP16 elementwise ops; inputs are `standard_normal * 0.1` (or `* 0.5`) cast
to FP16 from a fixed seed, never taken from the plan or an archive.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `pim/samsung_hbm_pim/test_samsung_lowering.py::test_operand_order_does_not_change_decisions` (`x[k]*W[i,k]` under a different region name gives identical `cmds`, `k_tiles`, `tile_fiber`, binds; compile-only, the swapped form is not run); `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_ragged_shape_reports_padding` (renamed locals `lW/lx/ly`, executed), `::test_gemm_matrix_residency_lever_changes_staging` (vector-first `lB*lA`, executed). No test renames args, loops or files and compares the Samsung decision; the rename-invariance tests in 1.1 and 1.5 compile a Samsung GEMV but compare classification and movement only | struct; sim versus np | none; rng(1), rng(2) | CO; SIM (samsung_sim_gate) |
| b | `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_elementwise_ops_run_exactly[ADD|MUL|RELU|MAD]` (one shape, 100,000 elements, bit-exact against the NumPy FP16 op; RELU sees negatives); `pim/samsung_hbm_pim/test_samsung_lowering.py::test_elementwise_programs_use_declared_moves` (RELU versus MAD streams). No constant change (for example a different literal in MAD or a clamp constant) | sim versus np FP16, bit-exact; struct | rng(3) * 0.5; none | SIM; CO |
| c | Executed: `::test_ragged_shape_reports_padding` (2048x100 padded to 4096x128, exact `padding_fraction`), `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_strict_default_runs_the_ordered_program` (4096 rows padded to 8192), `::test_ordered_gemm_columns` (2048x64, N=2), `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_gemm_matrix_residency_lever_changes_staging` (N=4), `::test_elementwise_ops_run_exactly[...]` (100,000 is not a multiple of 131,072). Plan or fake conductor: `perf/test_samsung_performance.py::test_reduce_plan_pads_rows_to_whole_passes_and_k_to_tiles` (5000x100 gives 2 passes, 8192 rows, K 128), `::test_reduce_run_zero_pads_into_physical_slots` (1024x1408, argv only, fake conductor writes zeros), `pim/samsung_hbm_pim/test_samsung_lowering.py::test_k512_program_derives_trip_counts_from_tiles`, `::test_reduce_plan_places_tiles_by_selected_fibers`. Rejected: `::test_partial_row_coverage_is_an_error` (`mapping=[16,4]` covers 2048 of 4096 rows). All rectangular; nothing irregular beyond zero padding | sim versus np and fp16-order; struct | rng(1), rng(0), rng(4), rng(2), rng(3); `np.ones`; none | SIM; unit; CO |
| d | `pim/samsung_hbm_pim/test_samsung_lowering.py::test_target_facts_raise_instead_of_guessing` (facts (8, 2, 8, 16) read from the target; a partial target raises). No parameter is varied and compiled | struct, exc | synthetic | CO |
| e | none (FP16 only; random normals include negatives but no extremes) | | | |
| f | `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_ordered_program_carries_a_nonzero_initial_value` (initial 0.5, internal runner, 26,807 cycles), `::test_carried_accumulator_through_the_public_api` (`y[row] += W*x` with random `y0`; matcher records `carried`; public `allo.compile`) | sim versus fp16-order from the initial value, bit-exact | rng(0) + 0.5; rng(11) | SIM |
| g | `pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py::test_readback_returns_all_zero_result` (an all-zero blob is returned as zeros), `::test_readback_rejects_absent_and_truncated_blob` (absent or short `out.bin` raises), `::test_paper_gemv_zero_input_writes_zero_output` (`x = 0`, `y` preset to ones becomes exact zeros, `outputs == {"y"}`) | struct; sim | synthetic blob; rng(0) W, zero x | unit; SIM |
| h | none | | | |
| i | none (`design/samsung-execution.md` 4.1 says the same buffer twice is rejected; `pim/samsung_hbm_pim/test_samsung_lowering.py::test_ambiguous_operands_are_rejected` uses two distinct 1-D buffers, so the alias case itself is untested) | | | |
| j | See 1.3: `test_strict_default_runs_the_ordered_program`, `test_reassociating_program_differs_from_source_order`, `test_ordered_gemm_columns`; `pim/test_gemv_all_targets.py::test_gemv_samsung_hbm_pim` (strict default through `allo.compile`, `kind == "simulator"`) | sim | rng(0) | SIM |
| k | Executed: `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_gemm_matrix_residency_lever_changes_staging` (`stage_resident` True versus False: `matrix_preloads` 1 versus N, resident faster, both correct). Structural or virtual: `pim/test_host_transfer_codegen.py::test_drain_emits_nop_not_bank_mov`, `::test_host_hooks_record_no_compute_cmd`, `::test_resolve_host_moves_builds_resolved_records`, `::test_stamp_host_moves_attaches_to_placement_extra`; `pim/test_host_transfer_dispatch.py::test_samsung_host_moves_carry_expected_verbs`, `::test_resolution_picks_the_right_samsung_moves`, `::test_grf_a_grf_b_disambiguate_by_handle_identity`; `perf/test_samsung_performance.py::test_virtual_graph_includes_shape_aware_host_transfers` (SCATTER, LD_B, MAC, ST_B, GATHER order; byte-priced moves); `pim/test_host_program_analysis.py::test_launch_schedule_orders_host_transfers_between_kernels`. Multi-kernel Samsung composition (two kernels with a host transfer between them) is never executed on the simulator | sim versus np; struct | rng(2); none | SIM; unit; V |
| l | CRF side (compile): `pim/samsung_hbm_pim/test_samsung_lowering.py::test_bank_row_and_crf_fill_change_the_emitted_program` (`bank_row`/host gives a 4-word CRF, `dual_fiber`/crf gives FILL MAC JUMP FILL MAC JUMP NOP EXIT). Execution side: `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_noncanonical_placements_execute_their_own_programs` (the same two forced placements run; receipts differ, bank_row MAC groups (16, 0) versus `fills == 2`; cycles differ from each other and from 4,435; both outputs within 1e-2 of FP32), `::test_tampered_crf_is_rejected_not_canonicalized` (the uploaded CRF is what runs), `::test_gemm_matrix_residency_lever_changes_staging` (cycles and preloads differ, CRF identical). Bytes and cycles both differ: `pim/samsung_hbm_pim/test_samsung_reduction_order.py::test_reassociating_program_differs_from_source_order`. Synthetic trace: `pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py::test_autoscheduled_xor_layout_is_conflict_free`, `::test_identity_and_xor_traces_run_on_pimsimulator` (148 versus 116 cycles pinned, no data). Vendor versus default: `pim/samsung_hbm_pim/gemv/test_gemv_samsung_hbm_pim.py::test_gemv_samsung_hbm_pim[...]` and `pim/test_paper_golden.py::test_samsung_gemv_paper_cycles[...]` run both but assert only the vendor cycles (4,435 / 8,041 / 8,247) and correctness of both; they do not assert that the two differ. No single test asserts different CRF and different output bytes for two REDUCE placements (REDUCE outputs agree to FP32 tolerance by design) | struct; sim | none; rng(0); rng(2); synthetic | CO; SIM; SIM via `libdramsim2.a` + g++ |
| m | none. `pim/samsung_hbm_pim/test_samsung_lowering.py::test_default_selection_follows_the_serialized_fiber_cost` and `perf/test_samsung_performance.py::test_autoscheduler_and_virtual_backend_share_cost_program` pin the selection under the current constants; neither changes a constant. The vendor-to-bank_row flip after the MAC rule recalibration exists only as the one-off run in `design/samsung-execution.md` section 7.3 | struct | none | CO, V |
| n | none as a test. Samsung simulator tests need no archive by construction (fresh inputs, no `_require_artifacts`). The round 4 artifact-free run (`TENON_ARTIFACTS` and `TENON_UPMEM_PROVENANCE_DIR` pointed at nonexistent paths; Samsung GEMV 4,389 cycles, max error 4.2e-4) is a log in `/tmp/tenon-release/probes/round3/no_artifacts.log`, not a test | | | |
| o | `pim/samsung_hbm_pim/test_samsung_lowering.py::test_ambiguous_operands_are_rejected`, `::test_partial_row_coverage_is_an_error`, `::test_elementwise_needs_a_provable_element_index`, `::test_unrealizable_levers_are_rejected_at_materialization` (`grf_staged`, `per_workid`), `::test_target_facts_raise_instead_of_guessing`; `pim/samsung_hbm_pim/test_samsung_conductor_runs.py::test_tampered_crf_is_rejected_not_canonicalized`; `pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py::test_readback_rejects_absent_and_truncated_blob`; `pim/test_host_transfer_dispatch.py::test_verb_compared_by_identity_not_string`, `::test_unimplemented_verb_is_hard_error_naming_verb_and_target`, `::test_record_outside_scope_errors` | exc | none | CO, SIM, unit |

### 1.8 AiM contraction and token partition

Files: `pim/test_aim_axis_roles.py`, `pim/test_aim_cost_routes.py`,
`pim/test_aim_matcher.py`, `pim/test_aim_program.py`,
`pim/test_aim_token_partition.py`, `pim/test_aim_virtual_study.py`,
`pim/test_cent_aim_workloads.py`, `pim/test_gemv_all_targets.py`,
`pim/test_paper_golden.py`. The AiM simulator (ramulator2) is timing-only:
every AiM run has `kind == "simulator-timing-only"` and `outputs == {}`. No
AiM numerical result is ever read back from a simulator. The only AiM
numerics checked anywhere are the host-side merge of the token partition.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `pim/test_aim_axis_roles.py::test_gemv_spellings_make_identical_decisions`, `::test_gemv_spellings_simulate_identically` (see 1.2) | struct; sim cycles | none | V; SIM (aim_sim_gate) |
| b | `pim/test_aim_axis_roles.py::test_unsupported_reduction_nest_is_rejected` only. No test swaps the operation at a fixed AiM shape (the audit's `acc += W - x` probe P3 has no in-tree successor) | exc | none | V |
| c | `pim/test_aim_axis_roles.py::test_partitioned_96_row_gemv_issues_logical_work_plus_declared_padding`; lowerer unit tests with pinned traces plus the MAC-count invariant rebuilt from the trace text: `pim/test_aim_program.py::test_contraction_derives_geometry_row_splits_and_reuse_order` (582 outputs, K=1030 crossing the 1024 row), `::test_row_packed_batches_derive_batch_and_output_tails_without_overcompute_channels` (49x70x25), `::test_channel_batches_derive_reduction_and_batch_tails_with_reserved_rows`, `::test_distributed_host_transfer_expands_linear_layout_and_burst_tail` (8193 elements), `::test_reuse_groups_are_target_derived_balanced_and_activation_aware` (86 launches over 32 slots), `::test_bank_group_layout_drives_elementwise_and_copy_span` (11,008 elements), `::test_reuse_and_storage_are_validated_against_target_capacity` | pin, MAC-count invariant, exc | fixed records | V, unit (no simulation) |
| d | `pim/test_aim_virtual_study.py::test_aim_geometry_variant_opcode_counts[row2KiB_gb2KiB|row4KiB_gb4KiB|row8KiB_gb8KiB]` (bank rows/columns and GB entries varied on the CENT `ffn_fc_l128` case; only pinned opcode counts asserted, no cycles); `pim/test_aim_axis_roles.py::test_simulator_rejects_a_mismatched_channel_count` (24-channel target versus 32-channel simulator config raises), `::test_target_geometry_requires_a_reuse_slot`; `pim/test_aim_cost_routes.py::test_cost_reads_geometry_from_target` (closed-form host-reduction cost on a synthetic 32-lane geometry), `::test_single_isr_encoder_and_target_slot_conventions` (`ewmul_slots` moved). No AiM target with another channel, bank or slot count is compiled and simulated | pin, exc, closed form | fixed | CO, unit |
| e | `pim/test_aim_token_partition.py::test_partitioned_program_host_step_cost_and_numerics[2|4|8]` (integers in [-16, 16], exact merge); `pim/test_aim_program.py::test_negative_channel_indices_fail_before_mask_encoding`. No overflow or shift probe | np (int exact, bf16 rel < 2^-6), exc | seeded rng, fixed | CO + V, unit |
| f | Rejection only: `pim/test_match_coverage.py::test_carried_accumulator_rejected_where_the_device_zeroes_it` | exc | none | V |
| g | `pim/test_gemv_all_targets.py::test_gemv_aim` asserts `outputs == {}` and `y` unchanged under `kind == "simulator-timing-only"`; this is the honest labelling of a timing-only run, not an all-zero numeric probe. None otherwise | struct | rng(0) | SIM (aim_sim_gate) |
| h | Hardware channel masks only (pinned masks in `pim/test_aim_program.py::test_row_packed_batches_derive_batch_and_output_tails_without_overcompute_channels`; negative index guard). No source-level guard | pin, exc | fixed | unit |
| i | none | | | |
| j | `pim/test_aim_token_partition.py::test_knob_domain_and_reduction_policy`; `pim/test_reduction_policy.py::test_aim_float_reduction_needs_permission`. Structural only; no numeric strict-versus-reassociate comparison exists for AiM | struct, exc | none | V |
| k | `pim/test_aim_token_partition.py::test_partitioned_program_host_step_cost_and_numerics[2|4|8]` (token axis split over P channel groups, explicit `AimHostReduction` host step; closed-form host cost recomputed in the test equals the virtual estimate; merge numerics via `reference_partial_sums` plus `AimHostReduction.merge` versus `np.einsum`), `::test_cache_append_into_partitioned_layout` (WR_ABK owner channels decoded from the mask); `pim/test_aim_cost_routes.py::test_host_transfer_prices_the_moved_slice` (whole buffer versus `[:, 127, :]` slice versus resident); `pim/test_cent_aim_workloads.py::test_matcher_trace_is_typed_trace_and_vendor_inventory[...]` (14 CENT decode stages, 3 default and 11 `paper_full`, trace SHA pinned); `pim/test_aim_program.py::test_mixed_typed_operations_form_one_continuous_trace`, `::test_bank_bank_contraction_and_all_bank_write_are_typed`, `::test_contraction_activation_is_grouped_on_only_the_final_k_row`; `pim/test_paper_golden.py::test_aim_cent_paper_cycles[...]` (14 rows, pinned cycles, 3 default and 11 `paper_full`) | np (merge only), closed form, pin | seeded rng, archive-side builders | CO, V, SIM (cycles only) |
| l | `pim/test_gemv_all_targets.py::test_gemv_aim_forced_sbk_expands_layout_banks` (forced SBK: 16 MAC_SBK per MAC_ABK, estimate SBK > ABK; simulated SBK > ABK cycles compared only when the simulator is available, no skip otherwise); `pim/test_aim_cost_routes.py::test_layout_and_knob_paths_price_one_route` (ABK 223 and SBK 1,528 virtual; the simulated 222 / 1,527 quoted in its docstring are not asserted), `::test_tied_minima_emit_one_program` (tied modes emit byte-identical traces); `pim/test_aim_program.py::test_row_packed_batches_match_canonical_score_layout_and_scoped_reuse` versus `::test_channel_batches_match_canonical_output_layout_and_storage_stride` (two batch mappings, pinned row order and stride); `pim/test_aim_token_partition.py::test_partitioned_program_host_step_cost_and_numerics[2|4|8]` (P layouts), `::test_nine_workload_selection` (`paper_full`, simulator; asserts only `device > 0` and `gap >= 0`, the selected-versus-best count is printed, not asserted) | struct, pin, sim cycles | zeros, fixed, seeded rng | CO, V, SIM |
| m | `pim/test_aim_matcher.py::test_aim_row_activation_cost_flips_batch_mapping` (`BATCH_MAPPING_ROW_ACTIVATE_CYCLES = 0` changes the fingerprint, flips `batch_mapping` row_packed to channels, WR_GB 16 to 4, MAC lines unchanged). No perturbation moves the channel group count or the partition count P | struct | none | V |
| n | none. `pim/test_cent_aim_workloads.py::test_case_table_is_exactly_the_fourteen_archived_cases` pins the dependence on the benchmark-side tables (the inverse of the probe). The round 4 artifact-free AiM GEMV (222 cycles) is a log, not a test | | | |
| o | `pim/test_aim_axis_roles.py::test_coverage_check_rejects_overissued_contraction`, `::test_codegen_rejects_a_record_that_misstates_the_program`, `::test_unsupported_reduction_nest_is_rejected`; `pim/test_aim_cost_routes.py::test_failed_simulator_run_reports_no_cycles` (nonzero exit raises even when stdout has cycles); `pim/test_aim_program.py::test_specialized_contractions_reject_unsafe_or_misleading_layouts` (7 layouts), `::test_flattened_batches_fail_closed_before_distinct_gb_payload_loss`, `::test_reuse_and_storage_are_validated_against_target_capacity` | exc | fixed | unit, V |

### 1.9 UPMEM families

Files: `pim/test_upmem_dtypes.py`, `pim/test_upmem_generality.py`,
`pim/test_upmem_matcher.py`, `perf/test_upmem_irregular.py`,
`perf/test_upmem_physical.py`, `perf/test_upmem_physical_search.py`,
`perf/test_prepare_upmem_tenon_campaign.py`, `pim/test_gemv_all_targets.py`,
`pim/test_paper_golden.py`, `tests/test_c_backend.py`. On simulator runs
uPIMulator itself byte-compares the device output against the portable-C
oracle (`backend/c.py`), and the test then compares the read-back caller
arrays against NumPy in int64. Families: MV, GEMM, reduction, histogram,
pointwise, selection, k-means distances and update, gradients,
`tasklet_private_wram` (MMTV).

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | none in the tree. The audit probe `p9_forms.py` ("identical device source after rename: True") ran on the pre-repair tree and has no regression successor | | | |
| b | Compile: `pim/test_upmem_generality.py::test_pointwise_family_accepts_the_matched_expression[sub|mul|scale|axpby]` (N=1536), `::test_selection_generalizes_to_the_matched_predicate[even|positive]`, `::test_kmeans_update_rejects_other_host_kernels[extra_op|no_divide|other_source|copy_only|index_mixed]`. Executed: `::test_sim_pointwise_forms_off_grid[sub|mul|scale]` (N=1000, inputs up to 2^30 so products wrap, compared to int64 NumPy wrapped to int32). **[WT]** `::test_minus_and_constant_forms_lower_to_the_axpby_template[...]`, `::test_minus_and_constant_forms_reject_non_int32_coefficients[...]`, `::test_constant_term_is_compile_time_axpby_only`, and the two extra `test_sim_pointwise_forms_off_grid` rows | struct, Python reference flags; sim versus np int32-wrapped | fixed -6..5; rng(7) | V; SIM (upmem gate) |
| c | Compile: `::test_pointwise_non_divisible_extents_compile_and_price_the_padded_program[8|1000|12289]`, `::test_matrix_vector_padded_shapes_compile[100-100|96-100|24-8|12-16|5-7]`, `::test_gemm_padded_shapes_compile[20-32-32|12-20-32|7-9-5]`, `::test_tasklet_private_wram_is_legal_wherever_it_fits[...]` (4 shapes, source string only), `::test_uneven_gemv_partition_compiles_with_recorded_padding[2|4|32]`, `::test_uneven_gemv_over_the_full_dpu_level` (391x65 on 64 DPUs, last DPU 13 rows), `::test_pointwise_layout_choice_is_no_longer_a_divisibility_artifact`. Executed: `::test_sim_carried_matrix_vector` (36x40), `::test_sim_carried_gemm` (20x24x18), `::test_sim_carried_histogram` (1000 inputs, 64 bins), `::test_sim_carried_reduction` (1000), `::test_sim_pointwise_forms_off_grid[...]` (1000), `::test_sim_uneven_reduction` (1000 over 64 DPUs: 63x15 + 55), `::test_sim_uneven_gemv_full_dpu_level` (391x65 over 64 DPUs), `::test_sim_uneven_gemv_scaling_391x65` (`paper_full`; 1, 2, 4, 8, 16, 17, 32 DPUs). Uneven GEMV cases all use the paper's 391x65; only the reduction is uneven at a non-paper shape | struct; sim versus np int64 | synthetic; rng(3), rng(4), rng(5), rng(6), rng(7), rng(9), rng(11), rng(10) | V; SIM (upmem gate) |
| d | DPU count only, through the workload `mapping=[N]`: the uneven tests above plus `pim/test_upmem_matcher.py::test_upmem_fanout_gemv` (1152x128, 1 to 32 DPUs, `host_moves`) and `::test_upmem_fanout_reduction` (12288, 1 to 32 DPUs), both `paper_full`. No test changes a `build_upmem_target()` parameter (tasklets, WRAM, MRAM); tasklets appear only as a negative (`perf/test_upmem_irregular.py::test_legality_checks_fail_before_emitting_an_invalid_translation_unit`, `num_tasklets=8` rejected) | sim versus np | rng(0), rng(1) | SIM (upmem gate + paper_full) |
| e | Rejections and Python-only checks: `pim/test_upmem_dtypes.py::test_unsupported_element_type_is_rejected_at_recognition[...]` (10 rows: i64, f32, i16 operands per family), `::test_signedness_and_output_type_are_checked_on_the_abi[...]`, `::test_int32_pointwise_compiles_and_rejects_a_wider_caller_array`, `::test_packers_never_truncate` (2^31, -(2^31)-1, 2^70, 2^62, 2^63), `::test_write_back_refuses_a_lossy_caller_dtype`, `::test_wide_signed_gradient_without_int64_temporary_fails_closed` (96-bit `shrsi`), `::test_kmeans_host_group_with_wide_shift_does_not_run` (`>> 40`), `::test_kmeans_host_group_with_literal_wide_scale_does_not_run` (`-(1 << 40)`); `pim/test_upmem_generality.py::test_pointwise_plan_evaluates_with_int32_wraparound` (INT32_MIN - 1), `::test_selection_predicate_c_and_python_agree_on_int32_edges`. Executed overflow: `::test_sim_pointwise_forms_off_grid[...]` (products of values up to 2^30 wrap in int32). Host-C fallback: `tests/test_c_backend.py::test_wide_add_mul_shl_truncated_to_int32_keeps_exact_low_bits`, `::test_wide_ops_that_read_high_bits_are_rejected[ashr|signed-compare|floordiv|high-bits]`, `::test_kmeans_update_wide_epilogue_raises_instead_of_returning_wrong_values`, `::test_kmeans_update_wide_epilogue_with_literal_scale_raises`, `::test_wide_integer_abi_is_rejected`. No int64-operand or float32 kernel executes on the simulator (both are rejected at recognition); the only int64 values read back are reduction cells | exc, exact packer round-trip, Python; sim versus np; `_wrap` modular oracle | fixed extremes; rng(7); fixed | CO, unit; SIM; HC |
| f | Compile: `pim/test_upmem_generality.py::test_carried_accumulators_load_the_initial_output[...]` (MV, GEMM, histogram, reduction: `accumulator_init == "carried"`, `mram_read` of the output in the device source), `::test_zero_initialized_sources_keep_their_templates`, `::test_carried_selection_count_is_implemented_by_the_host_step`. Executed: `::test_sim_carried_matrix_vector` (`y + A@x`), `::test_sim_carried_gemm`, `::test_sim_carried_histogram` (`bincount` on top of carried H), `::test_sim_carried_reduction` (int64 cell preset to 123,456,789), `::test_sim_selection_general_predicate_with_carried_count` (count preset to 5, untouched head -7). Rejected: `::test_carried_kmeans_distances_is_rejected_on_the_virtual_route`; `pim/test_gemv_all_targets.py::test_gemv_upmem` (non-constant initializer raises `UnsupportedWorkloadError`) | struct; sim versus np int64 | synthetic; rng(3), rng(4), rng(5), rng(6), rng(8) | V; SIM |
| g | none | | | |
| h | `pim/test_upmem_generality.py::test_sim_carried_histogram` (guarded increment), `::test_sim_selection_general_predicate_with_carried_count` (`(x & 6) != 2`, host compaction), `::test_selection_generalizes_to_the_matched_predicate[...]`, `::test_pointwise_non_divisible_extents_compile_and_price_the_padded_program[...]` (tail mask), `::test_uneven_streams_need_a_padding_neutral_axis` (histogram padding is not neutral), `::test_sim_uneven_reduction` (padded last DPU) | sim versus np; struct; exc | rng(5), rng(8), fixed, synthetic, rng(9) | SIM; V; unit |
| i | none | | | |
| j | Integer only: `pim/test_reduction_policy.py::test_strict_permits_splitting_an_integer_reduction`. Float operands are rejected at recognition (`pim/test_upmem_dtypes.py::test_unsupported_element_type_is_rejected_at_recognition[va-f32]`), so no FP reduction can execute | struct, exc | none | CO |
| k | `pim/test_upmem_generality.py::test_kmeans_update_host_group_is_priced` (host `kmeans_update` step priced; paper 120x8x4 from the benchmark factory; virtual), `::test_kmeans_update_rejects_other_host_kernels[...]`; executed host stages: `::test_sim_selection_general_predicate_with_carried_count` (host compaction), `::test_sim_uneven_reduction` and `pim/test_upmem_matcher.py::test_upmem_fanout_reduction` (cross-DPU partials merged by the caller, not by a compiled host step), `::test_upmem_fanout_gemv` (`host_moves`, `paper_full`); `pim/test_paper_golden.py::test_upmem_paper_cycles[...]` (16 rows incl. 2mm, 3mm, conv, k-means, regressions; archived canonical inputs; `check_outputs` versus NumPy; 3 default rows va, red, hist, 13 `paper_full`); `perf/test_prepare_upmem_tenon_campaign.py::test_offsets_sizes_repeats_and_hash_bindings` (2mm/3mm repeat fixtures, no run). No executed k-means update or gradient kernel outside the archived golden rows | struct; sim versus np; pin | synthetic; rng(8), rng(9), rng(1), rng(0); archive | V; SIM; SIM (RA + upmem gate) |
| l | none executed. `pim/test_upmem_generality.py::test_tasklet_private_wram_is_legal_wherever_it_fits[...]` and `::test_tasklet_private_wram_keeps_the_calibrated_kernel_at_one_batch_per_tasklet` compare two device-source variants as strings; `pim/test_upmem_matcher.py::test_upmem_knob_fanout_is_the_realizable_catalog` checks the catalog; `::test_upmem_calibration_flips_ranking` changes the selected decision on the virtual route only. The remainder-split versus balanced-split comparison (49,421 versus 40,058 cycles at 32 DPUs) is a design-doc run (`design/upmem-domain.md`), not a test | struct | synthetic | V, unit |
| m | `pim/test_upmem_matcher.py::test_upmem_calibration_flips_ranking` (one `PHYSICAL_CALIBRATION_FACTORS` entry x10: fingerprint changes and the VA winner leaves `fused_replicated/256`; virtual), `::test_upmem_cost_equals_physical_model`; `pim/test_upmem_generality.py::test_virtual_validity_labels` (calibrated versus extrapolated with nearest anchor), `::test_virtual_run_result_carries_scope_when_the_runner_forwards_it`, `::test_calibration_validity_of_the_mmtv_stratum_is_shape_exact`; `perf/test_prepare_upmem_tenon_campaign.py::test_calibrated_search_evidence_is_bound_to_every_modeled_plan` (MMTV `predicted_logic_cycles == 211_371`) | struct | paper shapes, synthetic | V |
| n | `perf/test_packaging.py::test_upmem_user_build_bypasses_snapshot_and_manifest` (monkeypatched: `TENON_UPMEM_PROVENANCE_DIR` missing, `UPIMULATOR_ROOT` + `UPIMULATOR_BIN` accepted, pinned route refuses a drifted manifest; nothing runs), `::test_upmem_missing_binary_names_both_routes`; `perf/test_prepare_upmem_tenon_campaign.py::test_complete_campaign_tree_is_deterministic` (fixtures generated from synthetic vectors, no archive, no run). The simulator gate accepts the user-build route so every `test_sim_*` test would run on it, but no test asserts `sdk_build == "user-build:..."`. Execution on the user build (all 16 golden rows) is documentary (`simulators.md`) | struct | tmp | unit |
| o | `pim/test_upmem_dtypes.py` (13 rejection rows plus the three wide-shift tests); `pim/test_upmem_generality.py::test_unrealizable_kernel_names_the_template_constraint`, `::test_tasklet_private_wram_rejects_on_its_wram_footprint`, `::test_selection_rejects_a_non_literal_predicate_naming_the_domain`, `::test_kmeans_update_rejects_other_host_kernels[...]`, `::test_carried_kmeans_distances_is_rejected_on_the_virtual_route`, `::test_uneven_streams_need_a_padding_neutral_axis`; `perf/test_upmem_irregular.py::test_legality_checks_fail_before_emitting_an_invalid_translation_unit`; `perf/test_upmem_physical.py::test_illegal_dma_or_unsupported_physical_modes_fail_closed[...]`; `perf/test_upmem_physical_search.py::test_unsupported_ownership_and_wram_illegal_plan_are_rejected_fail_closed` | exc | synthetic | CO, V, unit |

### 1.10 APU v1 native recipes, vector plans and scalar route

Files: `perf/test_apu_v1_recognizers.py`, `perf/test_apu_v1_native_reductions.py`,
`perf/test_apu_v1_reduction_policy.py`, `perf/test_apu_v1_tile_selection.py`,
`perf/test_apu_v1_vector_compile.py`, `perf/test_apu_v1_scalar_surface.py`,
`perf/test_apu_v1_cost_provenance.py`, `perf/test_apu_v1_distance_argmin.py`,
`perf/test_apu_v1_performance.py`, `perf/test_run_result_contract.py`,
`pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py`, `pim/test_gemv_all_targets.py`,
`pim/test_paper_golden.py`. Two facts frame this table: native recipe routes
refuse `backend="functional"` (only the device can run them), and on the
vector route `backend="functional"` executes the workload in NumPy/Python
inside the package (`allo/pim/apu_v1_vector_program.py`), not compiled C.
Only the scalar ARC route (`APUv1Program`) compiles and runs host C.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `perf/test_apu_v1_recognizers.py::test_histogram_paper_and_renamed_kernels_emit_identical_gvml`, `::test_argmin_paper_and_renamed_kernels_emit_identical_gvml` (also the mirrored `best > dist`), `::test_moments_paper_and_renamed_kernels_emit_identical_gvml`, `::test_record_frequency_paper_and_renamed_kernels_emit_identical_gvml` (function, argument, loop and local names and the kernel file name change; `device_source` byte-identical after role-name normalization; equal estimate); `::test_semantics_preserving_variants_are_accepted[...]` (5 rewrites: swapped loop nest, reordered statements, swapped roles, reordered args, permuted chunks; route name only) | byte-identical GVML, struct | synthetic | V |
| b | `::test_histogram_semantic_mutations_are_rejected[...]` (5: `+= 2`, `-= 1`, init 7, `!=`, `<`), `::test_argmin_semantic_mutations_are_rejected[...]` (6), `::test_moments_semantic_mutations_are_rejected[...]` (3: swapped columns, init 3, `x*x` in column 4), `::test_record_frequency_semantic_mutations_are_rejected[...]` (3), `::test_nearby_semantic_changes_are_rejected[...]` (8: extra store, `uint32` counts, reversed bin index, `2*x*x`, `idx = k+1`, wrong candidate row, missing chunk, else branch); `perf/test_apu_v1_native_reductions.py::test_native_reduction_recognizers_fail_closed_on_semantic_mutations` (MLIR text edits `cmpi eq` to `ne`, `muli` to `addi`) | exc (`NoNativeVectorRouteError` with closest `APUv1NativeRecipeMismatch`, reason regex) | synthetic | CO |
| c | `::test_histogram_nearby_bin_counts_compile_and_out_of_domain_rejects` (B=255 odd tail, B=64; N=16384 rejected), `::test_argmin_nearby_shapes_compile_and_out_of_domain_rejects` ((23, 11), (48, 2); D=49 rejected at 48 VMRs), `::test_moments_nearby_shapes_compile_and_out_of_domain_rejects` (16x2048, 4x4096 partial frontier; P=4095 rejected), `::test_record_frequency_nearby_shapes_compile_and_out_of_domain_rejects` (Q=127; C=14 at the writable-VR limit, C=15 rejected); `perf/test_apu_v1_tile_selection.py::test_tile_domain_is_derived_from_lanes_and_residency`; `perf/test_apu_v1_vector_compile.py::test_public_compile_generates_ranks_and_executes_four_apu_plans` (4x8x6, NumPy route). None of these shapes executes on the device | struct, exc; NP versus np FP32 | synthetic; rng(0) | V, CO; NP |
| d | none (board lock path and PCI node env overrides are discovery, not target parameters) | | | |
| e | `perf/test_apu_v1_scalar_surface.py::test_sqrt_is_exact_at_the_audit_failure_points` (1e10, 1e12, 1e30), `::test_sqrt_double_domain_ends_within_one_ulp[...]`, `::test_sqrt_special_values`, `::test_sqrt_random_sample_across_both_ranges` (20,000 doubles and floats, seed 20261102), `::test_scalar_program_uses_the_helper_and_matches_the_functional_route[root32|root64]` (MLIR-derived C compiled and run on the host against `np.sqrt`); `perf/test_apu_v1_recognizers.py::test_temporal_plan_accumulates_nonzero_initial_output[...]` (uint16 full-range wrap, NumPy route), `::test_nearby_semantic_changes_are_rejected[...]` (`uint32` counts rejected). No shift-width probe | HC versus `math.sqrt`/`np.sqrt`; NP versus np uint64 mod 2^16 | fixed, rng(20261102); rng(7) | HC (skips without a C compiler); NP |
| f | `perf/test_apu_v1_recognizers.py::test_temporal_plan_accumulates_nonzero_initial_output[functional|virtual]` (`C0 + A@B` on the functional NumPy route; the virtual route returns `kind="virtual"` and leaves the output untouched; the ABI half asserts `initial_outputs == ["result"]` in the frozen host.c and skips without the GSI template), `::test_plan_without_output_ingress_rejects_nonzero_output_on_every_route` (spatial group reduction refuses a nonzero output on functional, virtual and device before launch), `::test_host_harness_reads_carried_output_image_with_exact_size`. No device execution of a nonzero carried output; the only APU v1 device GEMM test uses a zero `result` | NP versus np; struct; exc | rng(7) | NP, V, device pre-launch |
| g | none | | | |
| h | Structural only: `perf/test_apu_v1_native_reductions.py::test_record_frequency_keeps_chunks_resident_and_counts_query_pairs` (queries 384 padded to 512, tail asserted zero), `perf/test_apu_v1_recognizers.py::test_moments_nearby_shapes_compile_and_out_of_domain_rejects` (padding zero); executed guard on the scalar route: `perf/test_run_result_contract.py::test_apu_v1_program_functional_executes_scalar_c_on_host` (`cumsum(where(v > 3))`) | struct; HC versus np | `arange`; fixed | V; HC |
| i | none | | | |
| j | See 1.3 (`perf/test_apu_v1_reduction_policy.py`, 4 tests) and `perf/test_apu_v1_vector_compile.py::test_public_compile_generates_ranks_and_executes_four_apu_plans`. The reassociative numerics are a NumPy tree model, not device output | NP | rng(0) | NP |
| k | Structural only (fused five-column moments, chained guards in the native recipes). none executed | | | |
| l | Device: `pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py::test_vector_gemm_apu_v1[vector_gemm_temporal_dma_coalescing|vector_gemm_temporal_dma_coalescing_broadcast_friendly]` (two forced plans; broadcast structure checked: 9 DMAs, lookup, subgroup duplication; readback allclose FP32 at 5e-2; median of runs within 2% of the golden row). Estimates only: `perf/test_apu_v1_tile_selection.py::test_paper_gemv_ranks_all_four_tiles` (selected = minimum estimate, tile 64 slower than 256), `::test_tile_can_be_forced_or_requested_as_a_family`, `::test_layout_gemm_catalog_and_paper_plans`; `perf/test_apu_v1_vector_compile.py::test_public_compile_accepts_candidate_name_and_plan_object` | dev versus np FP32; struct | rng(25) (zero initial result); synthetic | DEV (apu_v1_device marker + gate + board lock); V |
| m | none reaching selection. `perf/test_apu_v1_cost_provenance.py::test_cost_fingerprint_covers_the_fitted_model` (fingerprint only), `::test_record_frequency_before_calibration_is_the_papers_baseline`, `::test_confirmatory_shapes_before_and_after`, `::test_histogram_plans_before_and_after` (MAPE of estimates before and after calibration). The winner flips (`MICRO_LOOKUP_L4_SLOPE=0`, `MICRO_DMA_L4_L1_32K=2e6`) exist only in the audit probe `p5_gemm_plans.py` | struct, closed form | constants | V |
| n | none. `perf/test_apu_v1_cost_provenance.py::test_released_coefficients_match_the_archived_frozen_model` and `pim/test_paper_golden.py::test_apu_v1_paper_kernel_compiles_to_gvml[...]` / `::test_apu_v1_board_median_cycles[apu_v1-histogram]` read the archive and skip without it (the inverse of the probe) | | | |
| o | `perf/test_apu_v1_recognizers.py::test_no_native_route_reports_every_recipe_closest_first`, `::test_vector_plan_selection_requires_cost_or_explicit_layout`, `::test_launch_requires_exactly_one_fresh_profile`, `::test_matcher_output_specs_derive_from_abi_or_raise`, `::test_short_or_missing_readback_names_the_role`, `::test_device_runtimes_never_launch_without_ledag`; `perf/test_apu_v1_performance.py::test_profile_parser_requires_one_fresh_total_per_apuc`; `perf/test_run_result_contract.py::test_apu_v1_program_virtual_never_touches_device`, `::test_apu_v1_native_program_virtual_never_touches_device`, `::test_apu_v1_program_virtual_requires_cost`; `perf/test_apu_v1_cost_provenance.py::test_other_routes_and_foreign_cost_programs_are_extrapolated` | exc, struct | fixed | unit, V |

### 1.11 Gemini-II typed, composed and source route

Files: `perf/test_apu_g2_typed_program.py`, `perf/test_apu_g2_composed_contraction.py`,
`perf/test_apu_g2_source_route.py`, `perf/test_apu_g2_target_layout.py`,
`pim/test_apu_g2_hardware.py`, `pim/test_gemv_all_targets.py`. All Gemini-II
arithmetic is integer; category j does not apply. The card is on `zhang-x4`,
so `pim/test_apu_g2_hardware.py` skips `BLOCKED-DEVICE` on this server; the
round 3 card run is recorded in `VALIDATION.md`.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `perf/test_apu_g2_source_route.py::test_rename_only_variant_compiles_to_the_same_program` (renamed GESUMMV, in-place `C += ...` GEMM, int32 containers: layout/program fingerprints equal), `perf/test_apu_g2_composed_contraction.py::test_recipe_inventory_and_fingerprints_cover_layout_and_epilogue` (program-name rename keeps the fingerprint; 5 recipes distinct). No byte-level comparison of generated device code after a rename | fingerprint equality | synthetic | CO |
| b | `perf/test_apu_g2_source_route.py::test_estimates_depend_on_shape` (int2 versus int4 at a fixed shape changes the estimate), `::test_unsupported_forms_raise_a_finite_error[three_products|mixed_signedness|...]`. No test changes a coefficient or operation and checks that the program or oracle differs | struct, exc | synthetic | V, CO |
| c | `::test_nearby_shapes_compile_and_tile_beyond_one_carrier` (GEMM n32 versus n33: `single/k1` rows [0, 2048) + [2048, 2112) versus `single/k2` chain; GESUMMV M512; BiCG N512 gives `fused/k2`; transposed lhs), `::test_composed_record_beyond_one_carrier_names_the_capacity`, `::test_emulated_device_run_writes_exact_outputs[gemm_n33]`; `pim/test_apu_g2_hardware.py::test_nearby_gemm_beyond_one_carrier_on_card` (64x128x33 two-launch chain on the card) | struct, exc; EMU and dev versus np int64 | synthetic; rng(5); fresh | CO; EMU; DEV (apu_g2_card) |
| d | none (`perf/test_apu_g2_target_layout.py` pins fixed geometry facts; `::test_cmake_arguments_follow_the_resolved_sdk_and_device`, `::test_templates_take_the_sdk_root_and_device_index`, `::test_device_lock_is_exclusive_and_reentrant` are environment discovery) | | | |
| e | `perf/test_apu_g2_composed_contraction.py::test_generic_gemm_and_batched_gemm_dot_adapters_match_numpy` (int4 negatives, I15 canonicalization), `::test_interleaved_pair_bicg_and_mvt_semantics_match_numpy`; `perf/test_apu_g2_source_route.py::test_carried_accumulator_outside_device_width_is_rejected` (`x1[0] = 1 << 13` must fit int14); `perf/test_apu_g2_typed_program.py::test_device_dispatch_preserves_native_int15_output` (-17, 0, 2047 through a fake runtime); `pim/test_apu_g2_hardware.py::test_typed_program_per_operation_on_card` (add i16 over the full range wraps as `(a+b).astype(int16)`, mul i8 to i16, div u16, block_sum u8 to u16, dot i4 to i15 at 1985x128). **[WT]** `perf/test_apu_g2_value_range.py` (8 tests: decode rejects payload bits above the declared width, canonicalize rejects out-of-range unless wrapping is asked, composed and typed readback reject a device value outside the output type, DIV pads the divisor carrier with ones). No shift-width probe | np int64; exc; dev versus np int64 and runtime oracle | rng(11), rng(23); rng(9); fixed; fresh | unit; CO; EMU; DEV |
| f | `perf/test_apu_g2_composed_contraction.py::test_interleaved_pair_bicg_and_mvt_semantics_match_numpy` (MVT accumulator in [-200, 200]); `perf/test_apu_g2_source_route.py::test_paper_kernel_packing_equals_host_helpers[mvt]` (carried 14-bit), `::test_emulated_device_run_writes_exact_outputs[mvt]`, `::test_carried_accumulator_outside_device_width_is_rejected`; `pim/test_apu_g2_hardware.py::test_paper_source_kernel_on_card[mvt]` (carried `x1`, `x2` on the card) | np int64; dev versus np int64 | rng(23), rng(11), rng(5); fresh | unit, CO, EMU; DEV |
| g | none (the audit probe_03 "all-zero dots + nonzero acc -> [-8192]" has no test successor) | | | |
| h | `perf/test_apu_g2_source_route.py::test_paper_kernel_packing_equals_host_helpers[...]` (`attest_apu_g2_composed_packing`: padded lanes zero), `::test_tampered_packing_is_refused_before_launch` (rolled packing fails lane attestation, no launch). No executed test places garbage in padded lanes | struct, exc | rng(11), rng(8) | CO, EMU |
| i | none (`design/apu-g2-source-route.md` says an output that is also an operand is rejected; `gemm_accumulate_in_place` is a carried output, not an alias) | | | |
| j | not applicable (integer only) | | | |
| k | `perf/test_apu_g2_source_route.py::test_paper_kernel_derivations` (GESUMMV `fused/k1`, BiCG `{fused, separate} x {k1, k2}`, MVT carried int14), `::test_nearby_shapes_compile_and_tile_beyond_one_carrier`, `::test_emulated_device_run_writes_exact_outputs[batched_gemm|bicg|gemm|gesummv|mvt|gemm_n33]` (full device path on an emulated card: runtime int64 oracle plus the test's own NumPy reference, per-launch fingerprint and attestation); `perf/test_apu_g2_composed_contraction.py::test_interleaved_pair_bicg_and_mvt_semantics_match_numpy` (pair-affine GESUMMV (3, 2), interleaved BiCG); `pim/test_apu_g2_hardware.py::test_paper_source_kernel_on_card[batched_gemm|bicg|gemm|gesummv|mvt]` | struct; EMU and dev versus np int64 | synthetic; rng(5); fresh | CO; EMU; DEV |
| l | `::test_nearby_shapes_compile_and_tile_beyond_one_carrier` (`single/k1` versus `single/k2` launch coordinates and accumulators), `::test_layout_forces_a_named_plan_and_rejects_unknown_names` (`separate/k1` forced, 2 launches), `::test_paper_kernel_selects_the_artifact_program[batched_gemm|bicg|gemm|gesummv|mvt]` (ranking sorted, selection "cost", fingerprints equal constants embedded in the test). Executed: only the selected plan runs, emulated and on the card; the `k2` chain executes for n33 (`::test_emulated_device_run_writes_exact_outputs[gemm_n33]`, `pim/test_apu_g2_hardware.py::test_nearby_gemm_beyond_one_carrier_on_card`). A forced non-selected plan is never executed | struct, fingerprint; EMU and dev versus np int64 | synthetic; rng(5); fresh | CO; EMU; DEV |
| m | `::test_cost_perturbation_changes_estimate_and_selection` (`row_bits` x20: fingerprint differs, GEMM estimate rises with the same program, the n33 selection flips `single/k2` to `single/k1`). Compile-only; no perturbed program is executed | struct | synthetic | CO |
| n | `::test_paper_kernel_selects_the_artifact_program[...]` and `::test_rename_only_variant_compiles_to_the_same_program` use constants, so they run without the archive; `::test_paper_kernel_packing_is_byte_identical_to_the_campaign[...]` reads `tenon-artifacts/apuv2-composed-v2` and skips without it. No test runs with `TENON_ARTIFACTS` pointed at a missing path | fingerprint; SHA versus archive | synthetic; archive | CO |
| o | `::test_unsupported_forms_raise_a_finite_error[k96|mixed_signedness|three_products|narrow_accumulator|float_gemv|mixed_combined_types|k1024|offset_access|extra_effect]`, `::test_emulated_wrong_device_result_is_a_mismatch` (card adds +1 to dot heads), `::test_emulated_wrong_oracle_is_reported`, `::test_fingerprint_mismatch_is_refused_before_launch`, `::test_tampered_packing_is_refused_before_launch`, `::test_composed_record_beyond_one_carrier_names_the_capacity`; `pim/test_apu_g2_hardware.py::test_wrong_oracle_is_reported_as_a_mismatch_on_card`, `::test_echoed_fingerprint_mismatch_is_refused_on_card`, `::test_selected_fingerprint_mismatch_is_refused_before_launch` (no card needed); `pim/test_gemv_all_targets.py::test_gemv_apu_v2_source_rejected`; `perf/test_simulator_unavailable.py::test_gemini_ii_has_no_simulator_runner`; `perf/test_apu_g2_composed_contraction.py::test_host_command_and_parser_bind_every_physical_field` | exc | rng(6), rng(7), rng(8), zeros; fresh | EMU, CO; DEV |

### 1.12 AttAcc (analytical-only backend)

File: `pim/test_attacc_virtual.py`. There is no execution path:
`backend=None` raises. Ramulator2-AttAcc timing stays an external script.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `::test_renamed_workload_keeps_the_tiling` (renamed region, args, loops, accumulator: same tiling and cycles as `qkv_nh8`) | struct | fixed | V |
| b | `::test_shapes_outside_the_domain_raise` (fp32 dtype rejected). No operation change | exc | fixed | CO |
| c | `::test_gemv_space_sizes[...]` (4 shapes: space size equals the DCC sizes; every tiling satisfies coverage and channel/unit limits), `::test_counts_equal_the_dcc_trace_counts[...]` (24 tilings, 13 GEMV and 11 attention, against frozen DCC `212f86e` trace counts embedded inline; DCC is not imported), `::test_selection_is_the_cost_argmin_on_non_paper_shapes` (GEMV 4x1000x200, attention 2x64x100; checked against the production cost code, not independent) | independent external counts; production cost | fixed | V, unit |
| d | none (only the timing constant `n_bl` is varied, see m) | | | |
| e | `::test_shapes_outside_the_domain_raise` (2^17 x 2^17), `::test_counts_equal_the_dcc_trace_counts[...]` (1024x8x8192, 10 iterations). Weak | exc, counts | fixed | CO, unit |
| f | none | | | |
| g | none (`::test_virtual_result_kind_and_extra` checks outputs untouched under `kind == "virtual"`) | | | |
| h | none | | | |
| i | none | | | |
| j | not applicable | | | |
| k | `::test_attention_spot_checks[...]` (MAC, SFM_EXP, SFM_SUM, SFM_DIV, MAC chain; paper tiling; tie counts 1 and 2) | struct | fixed | V |
| l | `::test_counts_equal_the_dcc_trace_counts[...]` (per-tiling command counts). Analytical only; nothing executes | counts | fixed | unit |
| m | `::test_cost_perturbation_changes_the_selection` (`n_bl` 2 to 8 selects tiling `[[1,8,1],[4,8,160],[4,1,96]]` at 8,576 cycles from the same 3,043 candidates) | struct | fixed | V |
| n | none. `::test_paper_selector_on_all_500_instances` (`paper_full`) needs `benchmarks/attacc/paper_selection.csv` and runs `rank_attacc_tilings` directly, not `allo.compile`; the 500/500 run through `allo.compile` is a script (`verify_paper_selection.py`, 1,565 s) recorded in `VALIDATION.md` | archive CSV | archive | unit |
| o | `::test_backend_none_has_no_runner`, `::test_shapes_outside_the_domain_raise`, `::test_illegal_layout_is_rejected`, `::test_layout_without_a_tiling_is_refused`, `::test_target_is_legal_and_carries_the_paper_geometry` (`SFM_MAX` absent) | exc | fixed | CO, V |

Default-suite independent check of the selector: paper tilings on 6 instances
(`::test_paper_gemv_instances[...]`, `::test_acceptance_qkv_batch1`,
`::test_attention_spot_checks[...]`); 500 instances only under `paper_full`.

### 1.13 Frontend scoping and literals (shared Allo infrastructure)

Files: `tests/test_literal_typing.py`, `tests/test_global_vars_scope.py`,
`tests/test_c_backend.py`. These run on the host (LLVM JIT or clang-compiled
portable C), with fixed vectors. Rows d, j, l, m are not applicable to a
frontend typing rule and are marked none.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a | `tests/test_global_vars_scope.py::test_module_level_kernel_ignores_caller_locals`, `::test_enclosing_function_locals_resolve`, `::test_nearest_enclosing_scope_wins`, `::test_collected_names_exclude_non_enclosing_frames` (an unrelated `M = 256` in the caller must not change the geometry; F-01) | IR text, dict | none | CO |
| b | none | | | |
| c | `tests/test_literal_typing.py::test_store_range_check[...]` (int8 127 / 128 / -129, uint8 255 / 256 / -1 at the type boundary) | Python int; exc | none | JIT, CO |
| d | none | | | |
| e | `tests/test_literal_typing.py` (27 tests): `::test_negated_shift_literal_into_int64`, `::test_annotated_int64_scalar_from_wide_literal`, `::test_constant_shift_of_int32_value_past_its_width_is_rejected`, `::test_augmented_shift_past_width_is_rejected`, `::test_negative_constant_shift_is_rejected`, `::test_huge_constant_shift_is_rejected_without_folding`, `::test_shift_below_width_still_compiles`, `::test_literal_wider_than_int32_with_int64_buffer`, `::test_wide_literal_multiplier_in_c_backend`, `::test_literal_shift_assigned_to_int32_is_rejected`, `::test_annotated_int32_from_wide_literal_is_rejected`, `::test_return_range_check`, `::test_division_by_literal_wider_than_dividend_is_rejected`, `::test_bare_wide_literal_and_module_constant_are_int64`, `::test_unsigned_64_bit_all_ones`, `::test_hex_mask_on_uint32_value`, `::test_literal_wider_than_64_bits_is_rejected`, `::test_unmaterializable_wide_literal_is_rejected`, `::test_wide_intermediate_of_literal_only_expression_folds`, `::test_in_range_literals_keep_their_lowering`, `::test_cpp_style_wide_literal_and_unsigned_modular_operand`; `tests/test_c_backend.py::test_wide_add_mul_shl_truncated_to_int32_keeps_exact_low_bits` (+-2^40, -2^62), `::test_wide_ops_that_read_high_bits_are_rejected[ashr|signed-compare|floordiv|high-bits]`, `::test_wide_literal_multiplier_keeps_exact_low_bits`, `::test_wide_integer_abi_is_rejected` (int96), `::test_integer_sqrt_has_an_explicit_float_bridge_and_uint16_result` (0 and 0xFFFF), `::test_portable_c_popcount_preserves_fixed_width_semantics` (negative int8) | np, Python int, `_wrap` modular oracle, IR text; exc | fixed (incl. negatives, 0xFFFFFFFF, 2^40, 2^62) | JIT, HC (clang; no skip, fails without a compiler), CO |
| f | none | | | |
| g | `tests/test_literal_typing.py::test_shift_below_width_still_compiles` expects an all-zero result but the output is prefilled with zeros, so it does not discriminate. none effective | np | fixed | JIT |
| h | `tests/test_c_backend.py::test_portable_c_is_emitted_from_retained_mlir_with_control_flow` (while and if-else in portable C, `np.where(a > 0, a // 2, 0)`, prefill -1) | np | fixed | HC |
| i | none | | | |
| j | not applicable | | | |
| k | none | | | |
| l | not applicable | | | |
| m | not applicable | | | |
| n | none | | | |
| o | `tests/test_global_vars_scope.py::test_unresolved_name_is_an_error`; the 13 rejecting literal tests above; `tests/test_c_backend.py::test_portable_c_rejects_hls_only_integer_templates` | exc | none | CO |

### 1.14 Packaging and discovery

Files: `perf/test_packaging.py`, `pim/test_import_surface.py`,
`perf/test_simulator_unavailable.py`, `perf/test_run_result_contract.py`.
Categories a to c, e to m are not applicable to packaging; they are marked
none. No packaging test executes a kernel.

| Key | Tests | Oracle | Inputs | Path |
|---|---|---|---|---|
| a to c | none | | | |
| d | `perf/test_packaging.py::test_simulator_roots_follow_tenon_simulators_dir`, `::test_aim_image_override`, `::test_apu_g2_locations_go_through_simenv`; `perf/test_simulator_unavailable.py::test_apu_v1_locations_have_env_overrides`, `::test_apu_v1_pci_node_scans_vendor_id` (environment discovery parameters, not target parameters) | struct | env, tmp | unit |
| e to m | none | | | |
| n | `perf/test_packaging.py::test_sdist_ships_every_runtime_asset` (every runtime-read asset in the sdist; at least 30 `apu_g2` template files), `::test_sdist_ships_mlir_sources_but_no_build_tree` (no `mlir/build`, `allo/_mlir`, `tests/`, `examples/`, `benchmarks/`, `playground/`), `::test_only_allo_is_a_top_level_package`, `::test_no_user_home_literals_in_the_package`, `::test_apu_g2_runtime_has_no_hard_coded_sdk_paths`, `::test_paper_golden_artifacts_default_comes_from_simenv`, `::test_upmem_user_build_bypasses_snapshot_and_manifest`, `::test_install_requires_is_the_runtime_list_only`, `::test_upmem_calibration_table_regenerates`, `::test_aim_run_never_writes_into_the_simulator_checkout`; `pim/test_import_surface.py::test_import_allo_loads_no_pim_modules`, `::test_every_lazy_export_resolves`; `perf/test_simulator_unavailable.py::test_promotion_lifecycle_is_gone`. The sdist tests skip when setuptools is missing or the sdist build fails; none installs the artifact or imports from an installed copy; none executes a kernel with archives removed (the round 4 artifact-free run is a log) | struct | repo, env | build only, unit |
| o | `perf/test_packaging.py::test_unconfigured_simulator_location_names_the_variable`, `::test_upmem_missing_binary_names_both_routes`, `::test_old_cmake_error_names_the_pip_fix`; `perf/test_simulator_unavailable.py::test_unknown_target_always_has_a_reason`, `::test_samsung_run_raises_when_pim_driver_is_missing`, `::test_docker_timeout_is_reported_separately_from_missing_image`, `::test_gemini_ii_has_no_simulator_runner`, `::test_run_without_a_hook_raises_not_implemented`; `perf/test_run_result_contract.py::test_run_result_rejects_unknown_and_unspecified_kinds` | exc, struct | env | unit |

## 2. Execution-path evidence

Which tests actually execute on a simulator or device, versus compile-only
or virtual, by backend. "Gate" is the fixture or marker that skips the test
when the environment is missing; a simulator test that does not request its
gate and hits `SimulatorUnavailable` fails rather than skips.

### Gates

| Gate | Where | What it checks | Skip label |
|---|---|---|---|
| `samsung_sim_gate` | `pim/conftest.py` | `simulator_unavailable_reason("samsung_hbm_pim")`: `pim_driver` under `experiments/simulators/PIMSimulator` (`TENON_SIMULATORS_DIR`) | `BLOCKED-SIM(...)` |
| `aim_sim_gate` | `pim/conftest.py` | docker image `aim-simulator-build` (`TENON_AIM_SIM_IMAGE`) and `aim_simulator/test/example.yaml` | `BLOCKED-SIM(...)` |
| `conftest._simulator_gate("upmem")` called inline | `pim/test_upmem_generality.py`, `pim/test_upmem_matcher.py`, `pim/test_gemv_all_targets.py`, `pim/test_paper_golden.py` | `allo/spmw_simenv.py::upmem_unavailable_reason`: uPIMulator binary (`UPIMULATOR_BIN`), docker SDK image (`TENON_UPMEM_SDK_IMAGE`), provenance snapshot unless `UPIMULATOR_ROOT` is set, benchmark slot, and the pinned SDK manifest unless the user-build route is active | `BLOCKED-SIM(...)` |
| `apu_v1_device` marker + `apu_v1_device_gate` fixture + `board_lock` | `pim/conftest.py` | `spmw_simenv.apu_v1_unavailable_reason`: ARC toolchain, GVML template, GSI PCI node (vendor `0x1e4c`), GVML SDK; `flock` on `/tmp/tenon-apu-v1-<pci>.lock` (`TENON_APU_V1_LOCK`) | `BLOCKED-DEVICE(...)` |
| `apu_g2_card` fixture | `pim/test_apu_g2_hardware.py` | SDK files under `TENON_APU_G2_SDK_ROOT`, `gsi_tool` (`TENON_APU_G2_GSI_TOOL`), `gsi_tool info <device> -v` reporting `Status : Available` (`TENON_APU_G2_DEVICE`, only `apu-00` accepted); holds `apu_g2_device_lock` (`TENON_APU_G2_LOCK`) | `BLOCKED-DEVICE(...)` |
| `_require_artifacts` | `pim/test_paper_golden.py` | `spmw_simenv.tenon_artifacts_root()` (`TENON_ARTIFACTS`) | `BLOCKED-ENV(...)` |
| `paper_full` marker | `pim/conftest.py` | runs only with `-m paper_full` | "paper_full row; select with -m paper_full" |
| library/toolchain skips | `pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py` (`libdramsim2.a`, g++), `perf/test_apu_v1_scalar_surface.py` ("no host C compiler"), `perf/test_apu_v1_recognizers.py` (GSI example-gvml template), `perf/test_apu_v1_cost_provenance.py` and `perf/test_apu_g2_source_route.py` (artifact archives), `perf/test_packaging.py` (setuptools, sdist build) | | plain `pytest.skip` |

### By backend

**Samsung HBM-PIM (PIMSimulator, `pim_driver --op GENERIC`, `kind="simulator"`, outputs read back).**
Simulator, gated by `samsung_sim_gate`:
`pim/samsung_hbm_pim/test_samsung_conductor_runs.py` (all 9 collected items, 6 test functions:
`test_canonical_run_reports_program_plan_and_padding`,
`test_noncanonical_placements_execute_their_own_programs`,
`test_tampered_crf_is_rejected_not_canonicalized`,
`test_ragged_shape_reports_padding`,
`test_gemm_matrix_residency_lever_changes_staging`,
`test_elementwise_ops_run_exactly[4]`),
`pim/samsung_hbm_pim/test_samsung_reduction_order.py` (all 5),
`pim/samsung_hbm_pim/gemv/test_gemv_samsung_hbm_pim.py::test_gemv_samsung_hbm_pim[3]`,
`pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py::test_paper_gemv_zero_input_writes_zero_output`,
`pim/test_gemv_all_targets.py::test_gemv_samsung_hbm_pim`,
`pim/test_paper_golden.py::test_samsung_gemv_paper_cycles[3]`.
Simulator through a different harness (`libdramsim2.a` + g++, no data):
`test_layout_swizzle_probe.py::test_identity_and_xor_traces_run_on_pimsimulator`.
Everything else Samsung is compile-only or virtual (`test_samsung_lowering.py`,
`perf/test_samsung_performance.py`, `pim/test_reduction_policy.py`,
`pim/test_layout_selection.py`, `perf/test_legality.py`,
`perf/test_joint_selection.py`, host-transfer tests). No Samsung test needs an
archive.

**SK-Hynix AiM (ramulator2 in docker, `kind="simulator-timing-only"`, no outputs).**
Simulator cycles only, gated by `aim_sim_gate`:
`pim/test_gemv_all_targets.py::test_gemv_aim`,
`pim/test_aim_axis_roles.py::test_gemv_spellings_simulate_identically`,
`pim/test_paper_golden.py::test_aim_cent_paper_cycles[14]` (3 default, 11 `paper_full`),
`pim/test_aim_token_partition.py::test_nine_workload_selection` (`paper_full`).
`pim/test_gemv_all_targets.py::test_gemv_aim_forced_sbk_expands_layout_banks`
compiles both layouts and compares simulated cycles only if the simulator is
available (no skip). All other AiM tests are compile-only, virtual, or lowerer
unit tests with pinned traces. AiM numerics are never read back; the merge
numerics in `test_aim_token_partition.py` are host-side NumPy.

**UPMEM (uPIMulator in docker, `kind="simulator"`, outputs read back; uPIMulator byte-compares against the portable-C oracle).**
Simulator, gated inline:
`pim/test_upmem_generality.py` `test_sim_*` (8 tests, 10 items at HEAD; 12 items **[WT]**),
`pim/test_gemv_all_targets.py::test_gemv_upmem`;
`paper_full`: `::test_sim_uneven_gemv_scaling_391x65`,
`pim/test_upmem_matcher.py::test_upmem_fanout_gemv`, `::test_upmem_fanout_reduction`,
`pim/test_paper_golden.py::test_upmem_paper_cycles[16]` (3 default rows va, red,
hist; also needs `_require_artifacts`). Host-C fallback executes with clang in
`tests/test_c_backend.py` (no skip). Everything else is compile-only, virtual
or plan-constructor level. No test asserts the user-build (`sdk_build`) label
on an executed run.

**GSI APU v1 (board on this server, `kind="device"`).**
Device: `pim/test_gemv_all_targets.py::test_gemv_apu_v1` (fetches
`apu_v1_device_gate` at run time; no marker, so `-m "not apu_v1_device"` does
not deselect it),
`pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py::test_vector_gemm_apu_v1[2]`
(marker + gate + `board_lock`),
`pim/test_paper_golden.py::test_apu_v1_board_median_cycles[apu_v1-histogram]`
(marker + gate + `_require_artifacts`, archived oracle). Round 4: these 4
passed in 199 s (`VALIDATION.md`). Host C (clang/cc):
`perf/test_apu_v1_scalar_surface.py` (helper and scalar program),
`perf/test_run_result_contract.py::test_apu_v1_program_functional_executes_scalar_c_on_host`.
NumPy "functional" route (inside the package, not compiled):
`perf/test_apu_v1_reduction_policy.py::test_functional_numerics_follow_the_selected_plan`,
`perf/test_apu_v1_vector_compile.py::test_public_compile_generates_ranks_and_executes_four_apu_plans`,
`perf/test_apu_v1_recognizers.py::test_temporal_plan_accumulates_nonzero_initial_output[2]`.
All native-recipe tests are compile-only or virtual; no native recipe
(histogram, record frequency, moments, argmin) executes on the device outside
the archived golden histogram row.

**GSI Gemini-II (card on `zhang-x4`, `kind="device"`).**
Device, gated by `apu_g2_card`: `pim/test_apu_g2_hardware.py` (9 card tests;
`test_selected_fingerprint_mismatch_is_refused_before_launch` needs no card).
Round 3: 10 passed in 90 s on `zhang-x4` (`g2/round3b-hardware.log`,
`round3c-hardware.log`); on this server they skip `BLOCKED-DEVICE`. Emulated
card (monkeypatched runtime, full host path, no gate):
`perf/test_apu_g2_source_route.py::test_emulated_*` (10 items). Virtual:
`perf/test_apu_g2_typed_program.py`, cost tests in `test_apu_g2_source_route.py`.
Compile-only: the rest of the source route and the composed tests.

**AttAcc.** Analytical only (`backend="virtual"` or direct selector calls).
No simulator or device test exists; `backend=None` raises.

**Core, layout, selection, frontend.** Pure Python units, `backend="virtual"`
compiles, stub backends, the Allo LLVM JIT (literal typing), clang-compiled
portable C (`tests/test_c_backend.py`), and one dataflow CPU simulator build
(`perf/test_compile_interface.py::test_dataflow_build_does_not_validate_retained_scopes`).
No simulator or device.

### Executed probes that are not tests

`VALIDATION.md` and the audit files record executed evidence that has no
regression test: the artifact-free Samsung (4,389 cycles) and AiM (222 cycles)
runs; the AttAcc 500-instance `allo.compile` sweep; the UPMEM user-build
reproduction of all 16 golden rows; the UPMEM remainder-split versus
balanced-split scaling table; the Samsung P1/P2/P3 placement cycle table and
the cost/simulator agreement tables; the APU v1 `p5_gemm_plans.py` cost
perturbations; the Gemini-II card estimate-versus-cycles table. They are
documentary. Scripts live under `/tmp/tenon-release/probes/`,
`/tmp/upmem-fix/`, `/tmp/samsung-fix/` and
`/work/shared/users/phd/nz264/tenon-release-evidence/` on this server.

## 3. Gaps

### Status after round 6

The numbered list below is the gap analysis written at `d13cf08`. Round 6
added `tests/pim/test_g6_semantics.py` (semantic probes), `tests/pim/test_g6_decisions.py`
(decision probes), `tests/pim/apu_v1/test_g6_board_semantics.py` (board,
`apu_v1_device`) and five card-gated tests in `tests/pim/test_apu_g2_hardware.py`,
plus `tests/pim/test_upmem_owner_maps.py` and `tests/pim/test_aim_owner_maps.py`.
Their effect on each gap:

| Gap | Status after round 6 |
|---|---|
| 1 aliases | closed: source aliases rejected on every backend ("aliases its result"); same array for two parameters executes with pre-call semantics on Samsung, UPMEM, APU v1 functional and Gemini-II (card); in-place Samsung elementwise executes (defects C-17, S-14, V1-15 fixed) |
| 2 extreme integers | closed where numerics exist: UPMEM int32 products and int64 reductions wrap as documented and k-means refuses beyond its exact bound; APU v1 uint16 wraps (functional and board); Gemini-II int4 GEMM at both range ends on the card, int8 dot needing 17 bits rejected; AiM has no numerics (documented) |
| 3 nonzero accumulators on device paths | closed: APU v1 carried FP16 output bit-equal to the source-order model (functional and board); UPMEM and Gemini-II cases referenced |
| 4 two schedules executed | closed: AiM SBK versus ABK simulated (cost order equals simulated order); UPMEM two pointwise decisions executed with identical outputs; Gemini-II fused versus separate BiCG on the card with identical outputs |
| 5 cost perturbation to execution | closed for Samsung, UPMEM and APU v1 (perturbed models select different programs; both executed; the stock model's pick is faster in each case); the shared evaluator anomaly found on the way is C-18 (xfail) |
| 6 target parameter variation | closed: Samsung 8-channel variant (grid, streams, rows per item derived; the simulator runs it, see the note in `test_g6_decisions.py`), UPMEM 32-DPU variant, AiM geometry study referenced |
| 7 renames with decision comparison | closed for Samsung and UPMEM (renamed functions, arguments, loops and module file; identical normalized decisions and cycles) |
| 8 op/constant change at fixed shape | closed: Samsung ADD/MUL/MAD microcode and bindings, literal coefficients rejected; UPMEM AXPBY coefficients reach the device source and execute; AiM MUL/ADD/MAC traces (compile-only) |
| 9 composed stages executed | closed: Samsung two-kernel host program executes on the simulator (S-15); UPMEM k-means distances plus host update on the simulator with fresh inputs |
| 10 archive removal | closed: Samsung and AiM execute with the archive variables unset; UPMEM raises naming the variables (or runs the user-build route); sdist smoke test compiles the Samsung GEMV from an unpacked sdist |
| 11 all-zero results | closed: UPMEM, APU v1 functional and Gemini-II (card) zero results overwrite a sentinel |
| 12 masks and guards executed | closed: APU v1 padded reduction and odd GEMM on the functional route (board-gated too); Gemini-II refuses garbage outside the layout and the typed div with garbage in padded lanes runs on the card |
| 13 strict versus reassociative beyond Samsung | closed on APU v1 functional (strict bit-equal to source order; reassociating plan bit-equal to its tree model, within 1e-2 of FP32); board test asserts bit equality with the models |
| 14 movement stage never changes a program | closed on Samsung, UPMEM and AiM (owner-map ties executed, see CLAIMS row 16); APU v1 still realizes only the work-grid map |
| 15 working-tree drift | closed by the round 5 and 6 commits |

The list below is kept as the record of what was missing and why each test
was written.

1. **Alias cases (i), every mechanism.** No test anywhere passes the same
   NumPy buffer as two operands, or an input aliased with the output, through
   `allo.compile(...)(...)`. The Samsung and Gemini-II design docs claim the
   case is rejected; nothing checks it. Suggested: in
   `pim/samsung_hbm_pim/test_samsung_lowering.py` add
   `test_same_buffer_as_matrix_and_vector_is_rejected` (call the compiled GEMV
   with `W` passed for both roles and with `y is x`) expecting a finite error;
   add the same shape of test to `pim/test_upmem_generality.py` (pointwise
   `A = A + B` with `A` aliased, executed on the simulator and compared to
   NumPy, or rejected) and to `perf/test_apu_g2_source_route.py` (an output that
   is also an operand raises `APUG2SourceRouteError`).

2. **Extreme-integer overflow and shift widths outside UPMEM/C (e).** Samsung
   is FP16 only (acceptable), but AiM executes nothing numerically, APU v1 has
   only the uint16 wrap on the NumPy route, and Gemini-II has only the i16 add
   wrap on the card. No test anywhere shifts by a variable width on a device
   path. Suggested: `pim/test_apu_g2_hardware.py::test_typed_mul_i8_at_the_extremes_on_card`
   with operands fixed at -128, 127 and 0 (plus the emulated twin in
   `test_apu_g2_source_route.py`); for UPMEM an executed int64-cell reduction at
   INT64 extremes (`test_sim_reduction_int64_extremes`) and a gradient kernel
   with an int64 temporary executed on the simulator; for APU v1 a device run
   of `test_temporal_plan_accumulates_nonzero_initial_output` with a uint16
   initial output near 65,535.

3. **Nonzero accumulator on the device paths (f).** Samsung and UPMEM execute
   carried accumulators on simulators; Gemini-II executes MVT's carried vector
   on the card. APU v1 never executes a nonzero carried output on the board
   (V1-02 was fixed; the fix is validated by NumPy and by the frozen ABI text
   only), and AiM only rejects it. Suggested: add a third parametrization to
   `pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py` with a random nonzero
   `result` on the temporal plan, compared to `C0 + A@B` with the FP32
   tolerance already used there.

4. **Two feasible schedules traced to execution (l).** Only Samsung
   (`test_noncanonical_placements_execute_their_own_programs`,
   `test_reassociating_program_differs_from_source_order`) and APU v1
   (`test_vector_gemm_apu_v1[2]`) execute two distinct schedules of one
   workload. AiM compares SBK/ABK cycles only opportunistically
   (`test_gemv_aim_forced_sbk_expands_layout_banks`, no gate); UPMEM has no
   executed pair; Gemini-II executes only the selected plan. The layout
   movement stage never changes generated code at all. Suggested: AiM, give
   `test_gemv_aim_forced_sbk_expands_layout_banks` the `aim_sim_gate` and
   assert `sbk_cycles > abk_cycles` unconditionally, and add a `paper_full`
   assertion in `test_nine_workload_selection` that at least 6 of 9 selected
   partitions are simulated minima (the number the design doc claims); UPMEM,
   `test_sim_two_pointwise_layouts_execute_differently` forcing
   `fused_replicated/256` and `separate/64` for one VA shape, asserting both
   read back equal to NumPy and that `cycles` and the device-source hash
   differ; Gemini-II, run the forced `separate/k1` BiCG plan on the emulated
   card and on the card next to the selected `fused/k1` and assert equal
   outputs and distinct launch counts.

5. **Cost perturbation reaching compilation (m).** Present for AiM
   (`batch_mapping`), UPMEM (virtual VA winner), Gemini-II (n33 `k2` to `k1`,
   compile-only), AttAcc (tiling) and the joint selector (residency). Absent
   for Samsung (selection pinned under current constants only), APU v1 (only
   fingerprints and MAPE change) and the layout movement stage (weights never
   leave `uniform`). Suggested: Samsung,
   `test_fiber_tile_cost_perturbation_moves_the_default` monkeypatching
   `FIBER_TILE_CYCLES` so the vendor `dual_fiber` program wins again and
   asserting the emitted CRF changes to the 6-word vendor program; APU v1,
   port `p5_gemm_plans.py` as `test_lookup_slope_perturbation_flips_the_gemm_winner`
   (`MICRO_LOOKUP_L4_SLOPE=0` selects `..._acc2_l4`) in
   `perf/test_apu_v1_tile_selection.py`; for every perturbation test, extend
   with one executed run of the perturbed program on the available simulator so
   "reaches compilation" also means "reaches execution".

6. **Target parameter variation (d).** Only AiM varies a target geometry and
   only asserts opcode counts (`test_aim_geometry_variant_opcode_counts`);
   other variations are rejections (`slots=0`, 24 channels, 8 tasklets) or
   synthetic stub targets. No shipped target is rebuilt with a different
   advertised parameter and compiled plus costed. Suggested:
   `pim/test_aim_virtual_study.py::test_geometry_variant_changes_the_estimate_monotonically`
   (bigger GB entries lower the WR_GB count and the virtual cycles);
   `pim/test_upmem_generality.py::test_tasklet_count_variation_changes_the_split`
   building a UPMEM target with 8 and 16 tasklets through the target builder
   (not just rejecting 8 at the plan constructor), compiling the reduction,
   and asserting the generated `NR_TASKLETS` and estimate follow; a Samsung
   target with 8 channels compiled to a 64-bucket program.

7. **Renames with a decision comparison on Samsung and UPMEM (a).** Samsung
   compares the operand-order swap (compile-only) and executes renamed locals,
   but no test renames args, loops and the file and compares the selected
   placement and CRF. UPMEM has no rename test at all (audit probe only).
   Suggested: `pim/samsung_hbm_pim/test_samsung_lowering.py::test_renamed_gemv_emits_the_same_crf`
   (renamed region, args, loops, accumulator; assert identical `cmds`,
   `tile_fiber`, binds); `pim/test_upmem_generality.py::test_renamed_gemv_emits_identical_device_source`
   (byte-identical `task.c` after a role-name normalization, mirroring the APU
   v1 tests).

8. **Operation or constant change at fixed shape (b) outside the matcher and
   APU v1.** Samsung changes the op (ADD/MUL/RELU/MAD) but never a constant;
   AiM and Gemini-II change nothing at a fixed shape except to reject; AttAcc
   only rejects fp32. Suggested: Samsung `test_mad_with_different_constant_runs_exactly`
   (two literal scales at the same 100,000-element shape, bit-exact FP16);
   Gemini-II `test_gesummv_coefficients_change_the_oracle_and_the_program`
   (GESUMMV (3, 2) versus (5, 1) at one shape: different packed carriers,
   emulated outputs equal NumPy); AiM `test_sub_instead_of_mac_is_rejected`
   (`acc += W - x`, the audit P3 case).

9. **Composed producer/consumer stages executed (k).** Composition is executed
   only on Samsung (GEMM staging lever, one kernel) and Gemini-II (fused
   products, k2 chains). Samsung two-kernel programs with a host transfer
   between them (`mv1` then `mv2`) are virtual only
   (`test_launch_schedule_orders_host_transfers_between_kernels`); UPMEM
   k-means update and gradients execute only in the archived golden rows; AiM
   CENT stages are compile-only plus cycles. Suggested: Samsung
   `test_two_kernel_program_executes_with_a_host_transfer_between_them`
   (`y = W1 @ x; z = W2 @ y` through `allo.compile`, simulator readback versus
   NumPy FP32); UPMEM `test_sim_kmeans_update_host_group` with fresh random
   inputs at a non-paper shape, compared to NumPy; AiM
   `test_attention_block_merge_numerics_on_a_non_cent_shape` (the host-reduce
   merge oracle on a shape outside the 14 CENT cases).

10. **Archived results and oracle dependencies removed (n).** No test points
    `TENON_ARTIFACTS` and `TENON_UPMEM_PROVENANCE_DIR` at missing paths and then
    executes a kernel; the round 4 run is a log. The sdist tests do not install
    or import the artifact. Suggested:
    `pim/test_gemv_all_targets.py::test_samsung_gemv_runs_without_artifacts`
    (monkeypatch both variables to nonexistent directories, `samsung_sim_gate`,
    assert `kind == "simulator"` and readback within 1e-2 of FP32) and a UPMEM
    twin under the user-build route asserting `extra["sdk_build"]` starts with
    `user-build:`; in `perf/test_packaging.py`, build the wheel into a temporary
    venv and run `python -c "import allo; allo.compile(...)"` on the virtual
    backend.

11. **All-zero results (g) outside Samsung and the run-result stub.** Only
    Samsung executes a zero-input kernel and checks that zeros overwrite a
    prefilled output; UPMEM, APU v1 and Gemini-II have no such test (the G2
    probe_03 zero-dot case is audit-only). Suggested: UPMEM
    `test_sim_zero_input_pointwise_writes_zeros` (prefill -7, expect exact
    zeros); Gemini-II emulated and card `test_zero_lhs_gemm_returns_zero_dots`
    with a nonzero carried accumulator so the result is the accumulator alone.

12. **Masks and guards executed (h) on AiM and APU v1.** AiM has hardware
    channel masks only; APU v1 asserts padded tails are zero in the inventory
    but never executes a padded shape on the board; Gemini-II never executes
    with garbage in padded lanes. Suggested: Gemini-II
    `test_emulated_run_ignores_garbage_in_padded_lanes` (fill padded lanes
    after packing with random bits, expect attestation to refuse or the result
    to be unchanged); APU v1 device run of the B=255 histogram
    (`test_histogram_odd_bin_count_on_board`).

13. **Strict versus reassociative FP (j) outside Samsung.** The only executed
    strict/reassociate pair with bit-level checks is Samsung. APU v1's
    reassociative numerics are checked against a NumPy tree model that the
    test itself says is unvalidated against the device; AiM has no numeric
    path; UPMEM has no FP. Suggested: APU v1
    `test_vector_gemm_apu_v1[spatial_gemv_group_reduction_k256]` with
    `reduction_order="reassociate"` on the board, comparing readback bit-wise
    to the tree model (validating the model) and allclose to FP32.

14. **Layout movement stage (l, m, k).** The stage is validated as an
    algorithm (brute force at 3 and 4 bits, 121,968 masked layouts) but its
    output never changes a program and no backend realizes an alternative
    owner map (ISSUES C-11). This is a declared limitation, not a bug; the
    document records it so a reviewer does not read the layout tests as
    end-to-end evidence. Suggested: when any backend can realize a non-work-grid
    owner map, add a test that compiles the paper and swapped layouts through
    that backend and asserts different transfers plus equal readback.

15. **Working-tree drift.** `tests/perf/test_apu_g2_value_range.py` and the
    six changed UPMEM items are uncommitted. They must be committed (and this
    document's **[WT]** marks removed) or dropped before the release is cut.

## 4. How to run the evidence

Environment (this server):

```sh
cd /work/shared/users/phd/nz264/spmw-for-pim/experiments/allo
export PYTHONPATH=$PWD:$PYTHONPATH
PY=/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python
```

Suites, markers and round 4 runtimes (`VALIDATION.md`):

| Suite | Command | Round 4 result and wall time |
|---|---|---|
| PIM non-device (simulators, no golden, no APU board rows) | `$PY -m pytest tests/pim -q -p no:cacheprovider --deselect tests/pim/test_paper_golden.py -m "not apu_v1_device" -rs` | 311 passed, 25 skipped (`paper_full`), 59 deselected, 1,776 s (29 min 36 s). Note that `tests/pim/test_gemv_all_targets.py::test_gemv_apu_v1` carries no marker and runs on the board when it is reachable |
| perf (compile-only, virtual, host C, emulated card) | `$PY -m pytest tests/perf tests/test_global_vars_scope.py tests/test_c_backend.py tests/test_literal_typing.py -q -p no:cacheprovider -m "not apu_v1_device"` | 539 passed, 614 s |
| golden (default and `paper_full` rows, APU board row deselected) | `$PY -m pytest tests/pim/test_paper_golden.py -q -p no:cacheprovider -m "paper_full or not paper_full" --deselect "tests/pim/test_paper_golden.py::test_apu_v1_board_median_cycles"` | 55 passed, 834 s |
| `paper_full` rows elsewhere (AiM nine-workload study, UPMEM 391x65 scaling and fanout, AttAcc 500 instances, CENT 11 rows) | add `-m paper_full` to the PIM command | included in the 30 min figure only as skips; the nine-workload study and the UPMEM fanout rows take several minutes each (`/tmp/upmem-fix/paperfull-gatebypass.log`: 3 UPMEM rows in 1,029 s) |
| APU v1 device (this server) | `$PY -m pytest tests/pim/test_gemv_all_targets.py::test_gemv_apu_v1 tests/pim/apu_v1 "tests/pim/test_paper_golden.py::test_apu_v1_board_median_cycles" -q -p no:cacheprovider -m "apu_v1_device or not apu_v1_device"` | 4 passed, 199 s (`apuv1/w4-device-tests.log`) |
| Gemini-II card (on `zhang-x4`, wrapped in `flock /tmp/tenon-apu-g2.lock`) | `python -m pytest tests/pim/test_apu_g2_hardware.py -q -p no:cacheprovider` with the `allo-dev` interpreter on that host | 10 passed, 90 s (`g2/round3b-hardware.log`) |
| upstream smoke | `tests/test_builder.py tests/test_types.py tests/test_linalg.py tests/test_c_backend.py tests/test_bitop.py tests/test_popcount.py tests/test_static_slice.py` | 318 passed; 3 pre-existing dataflow OpenMP JIT failures reproduce on a pristine `8160d6d` |

Reading the results: a `BLOCKED-SIM(...)` or `BLOCKED-DEVICE(...)` skip means
the environment lacked the simulator or board; it is never a pass. A
`BLOCKED-ENV(...)` skip means `TENON_ARTIFACTS` was absent; only the golden
rows and the APU v1 calibration test depend on it. AiM simulator passes
validate cycles and trace content only, never numerics (section 1.8).

To confirm this document's citations against the tree:

```sh
$PY -m pytest --collect-only -q -p no:cacheprovider tests/pim tests/perf \
  tests/test_literal_typing.py tests/test_global_vars_scope.py tests/test_c_backend.py \
  | grep '::' > /tmp/collected.txt
grep -o '::test_[A-Za-z0-9_]*' docs/release/EVIDENCE.md | tr -d ':' | sort -u | \
  while read t; do grep -q "::$t\b" /tmp/collected.txt || echo "MISSING $t"; done
```

Names proposed in section 3 (Gaps) are suggestions and will appear as
`MISSING` until implemented.
