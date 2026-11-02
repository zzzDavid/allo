# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-movement layouts in placement selection, through ``allo.compile``.

(a) Samsung paper GEMV: derived accesses, completed search, chosen layout in
    the tie set, identical program with the stage disabled.
(b) A rename-only variant has the same movement cost and tie count.
(c) A non-affine (opaque) access falls back with ``movement="not-analyzed"``.
(d) A tiny budget ends at ``NODE_BUDGET``: incomplete, feasible incumbent.
(e) AiM and UPMEM GEMVs report movement without changing cycles; their
    strictly better-movement ties split a row's reduction (or leave units
    without rows) and are rejected with that reason.
(f) Samsung places contraction rows by the owner map: the paper GEMV's
    row-distribution ties are realizable and ranked (simulator runs in
    ``samsung_hbm_pim/test_samsung_owner_maps.py``).
(g) The ranking reports how many ties the backend realizes; APU v1 still
    accepts only the work-grid owner map (UPMEM and AiM row placement:
    ``test_upmem_owner_maps.py``, ``test_aim_owner_maps.py``).
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16, int32
from allo.ir.types import float16 as fp16
from allo.pim.costs import aim_cost, samsung_cost, upmem_cost
from allo.pim.layout_movement import AccessMode
from allo.pim.layout_search import LayoutSearchPolicy
from allo.pim.targets import build_aim_target, build_samsung_target, build_upmem_target
from allo.spmw_layout_problem import MovementSelectionPolicy, movement_selection_policy
from allo.spmw_linear_layout import LinearLayout

_HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SAMSUNG = _load("layout_selection_samsung", _HERE / "samsung_hbm_pim/gemv/workload.py")
_ALL = _load("layout_selection_all_targets", _HERE / "test_gemv_all_targets.py")
_HAS_ORDER = "reduction_order" in inspect.signature(allo.compile).parameters


def _order(value):
    return {"reduction_order": value} if _HAS_ORDER else {}


def _compile(workload, target, cost, order=None, policy=None):
    kwargs = _order(order) if order else {}
    if policy is None:
        return allo.compile(workload, target, cost, backend="virtual", **kwargs)
    with movement_selection_policy(policy):
        return allo.compile(workload, target, cost, backend="virtual", **kwargs)


def _kernel(compiled):
    (kernel,) = compiled.placement_ranking.kernels
    return kernel


def _split(layout):
    """Backend and movement parts of a carried ``product`` layout."""
    k = sum(1 for name in layout.out_dims if not name.startswith("mm."))
    backend = LinearLayout(
        {
            n: [v[:k] for v in vs]
            for n, vs in layout.bases.items()
            if not n.startswith("mm.")
        },
        layout.out_dims[:k],
        layout.out_sizes[:k],
    )
    movement = LinearLayout(
        {
            n: [v[k:] for v in vs]
            for n, vs in layout.bases.items()
            if n.startswith("mm.")
        },
        layout.out_dims[k:],
        layout.out_sizes[k:],
    )
    return backend, movement


_DISABLED = MovementSelectionPolicy(enabled=False)


# --------------------------------------------------------------------- #
# (a) Samsung paper GEMV
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_ORDER, reason="allo.compile has no reduction_order")
def test_samsung_paper_gemv_movement_stage():
    target = build_samsung_target()
    compiled = _compile(_SAMSUNG.gemv_m4096_k256, target, samsung_cost, "strict")
    kernel = _kernel(compiled)
    movement = kernel.movement
    assert movement.status == "analyzed"

    accesses = {a.memref: a for a in movement.derived.accesses}
    names = [v.name for v in movement.derived.variables]
    assert names == ["work0", "work1", "loop0", "loop1"]
    # W read by (row, k), x read by k, y accumulated by row (strict FP16).
    assert accesses["local_W"].mode is AccessMode.READ
    assert accesses["local_W"].variables == (0, 1, 2, 3)
    assert accesses["local_x"].mode is AccessMode.READ
    assert accesses["local_x"].variables == (3,)
    assert accesses["local_y"].mode is AccessMode.WRITE_ACCUMULATE
    assert accesses["local_y"].variables == (0, 1, 2)
    assert accesses["local_y"].reassociation_allowed is False

    assert movement.termination == "exhausted" and movement.optimality_proven
    assert not movement.incomplete and not compiled.placement_ranking.incomplete
    assert movement.movement_weights == "uniform"
    assert movement.min_cost == movement.program_cost == 72704
    assert movement.chosen_cost == movement.min_cost
    assert movement.chosen_label == "minimal"
    # One movement profile (the program's); Samsung places rows by the owner
    # map, so its per-level block/cyclic row distributions are crossed as
    # separate realizable ties (layout-search.md section 8).
    assert movement.ties[0].description == "pseudo_channel<-work0[0,1,2,3]; pim<-work1[0,1,2]"
    assert movement.tie_count == 4 and len(movement.ties) == 4
    assert all(tie.realizable for tie in movement.ties)
    assert {tie.cost for tie in movement.ties} == {72704}
    assert not kernel.movement_rejections

    # The chosen layout is a tie: its movement part is that tie's layout.
    chosen = compiled.compiled.layout[0]
    assert chosen.extra["movement"] == "minimal"
    _backend, carried = _split(chosen.layout)
    assert carried in [tie.layout for tie in movement.ties]

    # Same program and cycles with the stage disabled.
    baseline = _compile(
        _SAMSUNG.gemv_m4096_k256, target, samsung_cost, "strict", _DISABLED
    )
    assert list(compiled.compiled.cmds) == list(baseline.compiled.cmds)
    assert compiled.estimate().cycles == baseline.estimate().cycles
    assert len(kernel.scores) == movement.tie_count * len(_kernel(baseline).scores)
    assert _split(chosen.layout)[0] == baseline.compiled.layout[0].layout


def test_samsung_unrealizable_ties_are_rejected_before_emission():
    """Reassociation lets the minimum split k across PIM units; the backend
    places rows by the owner map but does not split a row's reduction (that
    is the reduction policy's partial-sum combine), so those ties are
    rejected with the reason and the program layout ranks, marked
    not-minimal."""
    target = build_samsung_target()
    order = "reassociate" if _HAS_ORDER else None
    compiled = _compile(_SAMSUNG.gemv_m4096_k256, target, samsung_cost, order)
    kernel = _kernel(compiled)
    movement = kernel.movement
    assert movement.min_cost == 48128 and movement.program_cost == 72704
    assert movement.tie_count == 2
    assert not any(tie.realizable for tie in movement.ties)
    assert len(kernel.movement_rejections) == 2 * len(kernel.scores)
    assert all(
        "splits the reduction" in reason and "reduction policy" in reason
        for _position, reason in kernel.movement_rejections
    )
    assert movement.chosen_label == "not-minimal"
    assert compiled.compiled.layout[0].extra["movement"] == "not-minimal"
    assert _split(compiled.compiled.layout[0].layout)[1] == movement.program_layout
    baseline = _compile(
        _SAMSUNG.gemv_m4096_k256, target, samsung_cost, order, _DISABLED
    )
    assert list(compiled.compiled.cmds) == list(baseline.compiled.cmds)


# --------------------------------------------------------------------- #
# (b) rename-only variant, (c) non-affine access, (d) tiny budget
# --------------------------------------------------------------------- #

_M, _K, _ROWS = 4096, 256, 32


@_df_region()
def renamed_gemv(A: fp16[_M, _K], v: fp16[_K], out: fp16[_M]):
    @allo.work(mapping=[16, 8], args=[A, v, out])
    def mv(matrix: fp16[_M, _K], vec: fp16[_K], result: fp16[_M]):
        p, q = allo.get_wid()
        base = (p * 8 + q) * _ROWS
        for r in range(_ROWS):
            s: fp16 = 0
            for c in range(_K):
                s += matrix[base + r, c] * vec[c]
            result[base + r] = s


_AIM_M = _AIM_K = 256


@_df_region()
def permuted_rows_gemv(
    W: bfloat16[_AIM_M, _AIM_K], x: bfloat16[_AIM_K], y: bfloat16[_AIM_M]
):
    @allo.work(mapping=[32], args=[W, x, y])
    def mv(
        local_W: bfloat16[_AIM_M, _AIM_K],
        local_x: bfloat16[_AIM_K],
        local_y: bfloat16[_AIM_M],
    ):
        (channel,) = allo.get_wid()
        # A permutation of the 32 row blocks (5 is odd): a valid partition
        # whose row is not affine in the work id; the matcher keeps it opaque.
        row0 = ((channel * 5) % 32) * 8
        for i in range(8):
            acc: bfloat16 = 0
            for j in range(_AIM_K):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


def test_rename_only_variant_has_the_same_movement():
    target = build_samsung_target()
    order = "strict" if _HAS_ORDER else None
    first = _kernel(
        _compile(_SAMSUNG.gemv_m4096_k256, target, samsung_cost, order)
    ).movement
    renamed = _kernel(_compile(renamed_gemv, target, samsung_cost, order)).movement
    assert renamed.status == first.status == "analyzed"
    for name in (
        "min_cost",
        "program_cost",
        "tie_count",
        "tie_layouts",
        "chosen_label",
    ):
        assert getattr(renamed, name) == getattr(first, name), name


def test_non_affine_access_falls_back_not_analyzed():
    target = build_aim_target()
    compiled = _compile(permuted_rows_gemv, target, aim_cost, "reassociate")
    baseline = _compile(permuted_rows_gemv, target, aim_cost, "reassociate", _DISABLED)
    movement = _kernel(compiled).movement
    assert movement.status == "not-analyzed"
    assert "SSA value the matcher does not retain" in movement.reason
    assert movement.chosen_label == f"not-analyzed: {movement.reason}"
    assert not movement.optimality_proven and not movement.ties
    for placement in compiled.compiled.layout:
        assert placement.extra["movement"].startswith("not-analyzed: ")
    # The hand-built layouts are kept unchanged.
    assert compiled.estimate().cycles == baseline.estimate().cycles
    assert list(compiled.compiled.cmds) == list(baseline.compiled.cmds)


def test_tiny_budget_is_incomplete_with_a_feasible_incumbent():
    target = build_samsung_target()
    order = "strict" if _HAS_ORDER else None
    policy = MovementSelectionPolicy(search=LayoutSearchPolicy(max_search_nodes=2))
    compiled = _compile(_SAMSUNG.gemv_m4096_k256, target, samsung_cost, order, policy)
    movement = _kernel(compiled).movement
    assert movement.termination == "node-budget"
    assert movement.incomplete and not movement.optimality_proven
    assert compiled.placement_ranking.incomplete
    assert movement.ties and movement.min_cost is not None
    from allo.pim.layout_movement import layout_movement_cost

    derived = movement.derived
    for tie in movement.ties:
        total = layout_movement_cost(
            derived.layout_accesses(), derived.hierarchy, tie.work_to_iteration
        ).total
        assert total == tie.cost
    assert movement.chosen_label != "minimal"


# --------------------------------------------------------------------- #
# (e) AiM and UPMEM
# --------------------------------------------------------------------- #


_UPMEM_M, _UPMEM_K, _UPMEM_DPUS = 768, 128, 64
_UPMEM_ROWS = _UPMEM_M // _UPMEM_DPUS  # 12: a non-power-of-two (masked) loop


@_df_region()
def upmem_gemv(W: int32[_UPMEM_M, _UPMEM_K], x: int32[_UPMEM_K], y: int32[_UPMEM_M]):
    @allo.work(mapping=[_UPMEM_DPUS], args=[W, x, y])
    def mv(
        local_W: int32[_UPMEM_M, _UPMEM_K],
        local_x: int32[_UPMEM_K],
        local_y: int32[_UPMEM_M],
    ):
        (dpu,) = allo.get_wid()
        for i in range(_UPMEM_ROWS):
            local_y[dpu * _UPMEM_ROWS + i] = 0
            for j in range(_UPMEM_K):
                local_y[dpu * _UPMEM_ROWS + i] += (
                    local_W[dpu * _UPMEM_ROWS + i, j] * local_x[j]
                )


@pytest.mark.parametrize(
    "workload, build, cost, order",
    [
        (_ALL.aim_gemv, build_aim_target, aim_cost, "reassociate"),
        (upmem_gemv, build_upmem_target, upmem_cost, None),
    ],
    ids=["aim", "upmem"],
)
def test_aim_and_upmem_report_movement_without_changing_cycles(
    workload, build, cost, order, record_property
):
    target = build()
    compiled = _compile(workload, target, cost, order)
    baseline = _compile(workload, target, cost, order, _DISABLED)
    kernel = _kernel(compiled)
    movement = kernel.movement
    assert movement.status == "analyzed"
    assert compiled.estimate().cycles == baseline.estimate().cycles
    assert list(compiled.compiled.cmds) == list(baseline.compiled.cmds)
    record_property("cycles", compiled.estimate().cycles)
    record_property("min_cost", str(movement.min_cost))
    record_property("program_cost", str(movement.program_cost))
    # Both programs replicate x on every work unit; a strictly
    # better-movement owner map exists, but it splits each row's reduction
    # across units, which neither lowering combines.
    assert movement.program_cost > movement.min_cost
    assert movement.chosen_label == "not-minimal"
    assert movement.ties and not any(tie.realizable for tie in movement.ties)
    assert movement.realizable_tie_count == 0
    assert kernel.movement_rejections
    # AiM and UPMEM place contraction rows by the owner map
    # (test_upmem_owner_maps.py, test_aim_owner_maps.py); these ties are
    # reduction splits and are rejected with the backend's reason.
    reasons = [reason for _position, reason in kernel.movement_rejections]
    assert all("places contraction rows by the owner map" in r for r in reasons)
    if target.name == "upmem":
        assert all("UPMEM accumulates in int32" in r for r in reasons)
    else:
        assert all("not that split" in r and "reduction_partitions" in r for r in reasons)
    if target.name == "upmem":
        assert movement.optimality_scope == "coordinate layouts"
        assert movement.derived.is_masked
        assert movement.derived.active_extents == ()
        assert movement.derived.unmodelled_levels == ("tasklet",)
    else:
        assert movement.optimality_proven
        assert movement.derived.unmodelled_levels == ("bank_group", "bank")


# --------------------------------------------------------------------- #
# (f) Samsung row-distribution ties
# --------------------------------------------------------------------- #


def test_samsung_row_distribution_ties_are_distinct_owner_maps():
    """The four per-level block/cyclic distributions of the 12 row bits are
    distinct owner flags with the program's movement, all realizable; the
    k bits stay local in every one (the strict reduction is not split)."""
    from allo.spmw_layout_problem import row_distributions, row_map_unrealizable_reason

    compiled = _compile(_SAMSUNG.gemv_m4096_k256, build_samsung_target(), samsung_cost)
    movement = _kernel(compiled).movement
    derived = movement.derived
    names = [name for name, _assignment in row_distributions(derived)]
    assert names == [
        "rows block over pseudo_channel, block over pim",
        "rows block over pseudo_channel, cyclic over pim",
        "rows cyclic over pseudo_channel, block over pim",
        "rows cyclic over pseudo_channel, cyclic over pim",
    ]
    flags = {derived.owner_flag(tie.work_to_iteration) for tie in movement.ties}
    assert len(flags) == movement.tie_count == 4
    for tie in movement.ties:
        assert row_map_unrealizable_reason(derived, tie.work_to_iteration) is None
        # loop1 (k) carries no owner bit.
        assert "loop1" not in tie.description
    # The program is the block x block distribution.
    (_name, block_block), *_ = row_distributions(derived)
    assert block_block.matrix(derived.coordinate_problem) == derived.program_layout()


# --------------------------------------------------------------------- #
# (g) Realizable tie counts; APU v1 keeps the work-grid owner map
# --------------------------------------------------------------------- #


def test_default_compile_reports_realizable_tie_counts_and_apu_v1_rejects():
    """Samsung, UPMEM and AiM place contraction rows by the owner map
    (``ROW_MAP_TARGETS``); the record counts the ties each one executes.
    APU v1 emits one program per work item and rejects any other owner map
    with that reason (UPMEM and AiM counts: ``test_upmem_owner_maps.py``,
    ``test_aim_owner_maps.py``)."""
    from dataclasses import replace

    from allo.spmw_layout_problem import (
        ROW_MAP_TARGETS,
        _realization,
        row_distributions,
        row_map_unrealizable_reason,
    )

    assert {"samsung_hbm_pim", "upmem", "aim"} <= ROW_MAP_TARGETS
    assert "apu_v1" not in ROW_MAP_TARGETS
    compiled = _compile(_SAMSUNG.gemv_m4096_k256, build_samsung_target(), samsung_cost)
    movement = _kernel(compiled).movement
    assert movement.realizable_tie_count == movement.tie_count == 4
    assert movement.chosen_label == "minimal"

    # The same problem stated for APU v1: a row distribution other than the
    # program's is rejected with the one-program-per-work-item reason.
    derived = replace(movement.derived, target_name="apu_v1")
    program = derived.program_assignment()
    program_matrix = derived.program_layout()
    (_name, cyclic), *_ = reversed(row_distributions(derived))
    matrix = cyclic.matrix(derived.coordinate_problem)
    realizable, reason = _realization(
        derived, matrix, derived.owner_flag(program_matrix), program, "cyclic rows"
    )
    assert not realizable
    assert "apu_v1 emits one program per work item" in reason
    assert "does not re-map iterations to owners" in reason
    assert row_map_unrealizable_reason(derived, matrix) == (
        "apu_v1 does not place rows by the owner map"
    )
    assert _realization(
        derived, program_matrix, derived.owner_flag(program_matrix), program, "program"
    ) == (True, None)
