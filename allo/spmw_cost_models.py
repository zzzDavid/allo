# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete `@allo.cost(...)` factories.

These run at import time to register cost callbacks with `spmw_cost`.
Importing this module is what makes `allo.get_cost("kernel_cycles", ...)`
return a working callback.

Design 04 (decoupled CostModel): the per-target cycle bodies no longer
live here. Each `kernel_cycles` factory looks up the `CostModel` bound to
the target (faithful flavor) and returns a `(trace, layout) -> int`
closure that calls `model.compose(...).cycles` -- byte-identical to the
pre-decoupling numbers (report 18 invariants). The numbers + compose
bodies live in `spmw_cost_tables.py`; the mechanism in
`spmw_cost_model.py`. The register-spill factories still read per-move
costs, now off the bound CostModel rather than the target tree.
"""

from __future__ import annotations

from .spmw_cost import cost
from .spmw_cost_model import ComposeCtx, combine, get_cost_model
# Re-export so the few non-cost callers keep their import site. The literal
# parser moved to spmw_tripcount (tier-1 of resolve_trip_count); the APU v1
# VR-tiling helper moved to spmw_cost_tables.
from .spmw_tripcount import _parse_loop_bound  # noqa: F401
from .spmw_cost_tables import _apu_v1_vr_tiling  # noqa: F401
from . import spmw_cost_tables as _tables


# --------------------------------------------------------------------- #
# Back-compat Samsung helper shims. The report-18 invariant tests import
# these by their pre-decoupling `(target, ...)` signatures; the bodies
# live in spmw_cost_tables and read the bound faithful CostModel.
# --------------------------------------------------------------------- #


# Re-exported so codegen + tests keep their import site (used by
# spmw_codegen for the shared-CRF work-id count audit).
from .spmw_cost_tables import _samsung_workid_count  # noqa: F401,E402


def _samsung_mk(target, trace):
    return _tables._samsung_mk(target, trace)


def _samsung_preload_cycles(target, M, K):
    # Re-homed: preload constants now live on the host_staging concern
    # (task-017). The shim reads that model so callers (tests) get the same
    # 369/1/2 arithmetic from the new home.
    return _tables._samsung_preload_cycles(
        get_cost_model("samsung_hbm_pim", "faithful", concern="host_staging"),
        M, K,
    )


def _samsung_readback_cycles(target, M):
    return _tables._samsung_readback_cycles(
        get_cost_model("samsung_hbm_pim", "faithful", concern="host_staging"),
        M,
    )


def _trace_batch_dim(trace):
    return _tables._trace_batch_dim(trace)


def _apu_v1_move_cycles(target, trace, layout):
    return _tables._apu_v1_move_cycles(
        get_cost_model("apu_v1", "faithful"), target, trace, layout
    )


# Back-compat per-target kernel-cycle factory shims. A handful of tests
# build the closure directly (e.g. `_upmem_kernel_cycles(target)`); these
# now delegate to the bound CostModel, so the returned closure is the same
# `(trace, layout) -> int` the autoschedule path uses.
def _upmem_kernel_cycles(target):
    return _kernel_cycles_factory(target)


def _samsung_kernel_cycles(target):
    return _kernel_cycles_factory(target)


def _aim_kernel_cycles(target):
    return _kernel_cycles_factory(target)


def _apu_v1_kernel_cycles(target):
    return _kernel_cycles_factory(target)


def _apu_v2_kernel_cycles(target):
    return _kernel_cycles_factory(target)


# --------------------------------------------------------------------- #
# kernel_cycles: delegate to the bound CostModel (design 04 §1.5)
# --------------------------------------------------------------------- #


# The flavor->fold-rule binding (design 07 §A1.3): `True` binds the overlap
# fold (max-across independent resources + fill/drain, D1), `False` the
# faithful serial sum. Every existing flavor (faithful, optimistic, Mortise's
# three, the demo's, the placeholders) folds by sum, so the faithful number is
# frozen by construction; default is `False` for any flavor not listed. The
# `overlap` flavor (design 07 §D1, task 005) is the ONLY one binding `True` --
# overlap is a property of the Phase timeline + the fold, NOT a parallel
# hand-coded flavor.
_COMBINER_FOR_FLAVOR: dict[str, bool] = {
    "faithful": False,
    "optimistic": False,
    "constant": False,
    "micro25": False,
    "unlimited": False,
    "placeholder": False,
    "overlap": True,   # design 07 D1 / task 005: max-across + fill/drain
}


@cost("kernel_cycles")
def _kernel_cycles_factory(target):
    """Build a kernel-cycle estimator specialised to `target`.

    The autoschedule path always uses the `"faithful"` flavor so the
    argmin stays calibrated (design 04 §1.5). Whole-program estimate is
    `kernel_cycles + host_staging` (design 05 §5, task-017): the device-exec
    `list[Phase]` from the `kernel_cycles` CostModel concatenated with the
    host<->device staging `list[Phase]` from the `host_staging` CostModel,
    folded by the flavor-bound combiner (design 07 §A1.3; faithful = serial
    sum, byte-identical to the old `device + host`). Targets with no
    `host_staging` model registered (AiM, UPMEM, APU) fold the device phases
    alone.
    """
    target_name = getattr(target, "name", None)
    model = get_cost_model(target_name, "faithful")
    overlap = _COMBINER_FOR_FLAVOR.get("faithful", False)
    try:
        hs_model = get_cost_model(target_name, "faithful", concern="host_staging")
    except KeyError:
        hs_model = None

    def cost_fn(trace, layout) -> int:
        device = model.compose(ComposeCtx(target, trace, layout))
        if hs_model is None:
            return combine(list(device.phases), overlap=overlap)
        host = hs_model.compose(ComposeCtx(target, trace, layout))
        return combine(list(device.phases) + list(host.phases), overlap=overlap)

    return cost_fn


# --------------------------------------------------------------------- #
# Register-spill cost factories (spec 015)
# --------------------------------------------------------------------- #
#
# Each factory runs once per target (via `get_cost`'s cache) and returns
# a closure `spill(reg, n_entries=1)` that returns the cycles charged for
# spilling a single live range to its declared spill tier over
# `n_entries` uses. Per design 04 the per-move spill costs now come from
# the bound CostModel's `move_costs`, not `target.move(name).cycles`.


from .spmw_cost_model import MoveCostCtx  # noqa: E402


@cost("register_spill")
def _register_spill_factory(target):
    name = getattr(target, "name", None)
    if name == "samsung_hbm_pim":
        return _samsung_register_spill(target)
    if name == "aim":
        return _aim_register_spill(target)
    if name == "upmem":
        return _upmem_register_spill(target)
    if name == "apu_v1":
        return _apu_v1_register_spill(target)
    if name == "apu_v2":
        return _apu_v2_register_spill(target)
    raise NotImplementedError(
        f"register_spill: no factory for target {name!r}"
    )


def _samsung_register_spill(target):
    """Samsung HBM-PIM: GRF_A/B <-> bank-row round-trip (spec 015 §6.1)."""
    model = get_cost_model("samsung_hbm_pim", "faithful")

    def spill(reg, n_entries=1):
        side = "A" if reg is target.grf_a else "B"
        return (
            model.move_cost(f"ST_{side}", MoveCostCtx(f"ST_{side}"))
            + model.move_cost(f"LD_{side}", MoveCostCtx(f"LD_{side}"))
        ) * n_entries

    return spill


def _aim_register_spill(target):
    """AiM: GPR <-> bank-row round-trip (JSSC 2023 §IV)."""
    model = get_cost_model("aim", "faithful")

    def spill(reg, n_entries=1):
        load_cyc = model.move_cost("RD_SBK", MoveCostCtx("RD_SBK"))
        store_cyc = model.move_cost("ST_SBK", MoveCostCtx("ST_SBK"))
        return (load_cyc + store_cyc) * n_entries

    return spill


def _upmem_register_spill(target):
    """UPMEM: WRAM <-> MRAM round-trip dominates spill cost (spec 015 §6.3)."""
    model = get_cost_model("upmem", "faithful")

    def spill(reg, n_entries=1, tier="mram"):
        if tier == "wram":
            return (
                model.move_cost("ST_WRAM", MoveCostCtx("ST_WRAM"))
                + model.move_cost("LD_WRAM", MoveCostCtx("LD_WRAM"))
            ) * n_entries
        if tier == "mram":
            return (
                model.move_cost("ST_MRAM", MoveCostCtx("ST_MRAM"))
                + model.move_cost("LD_MRAM", MoveCostCtx("LD_MRAM"))
            ) * n_entries
        raise ValueError(f"upmem spill tier {tier!r}")

    return spill


def _apu_v1_register_spill(target):
    """APU v1: VR <-> L1 round-trip (spec 015 §6.4)."""
    model = get_cost_model("apu_v1", "faithful")

    def spill(reg, n_entries=1):
        return (
            model.move_cost("ST_VR", MoveCostCtx("ST_VR"))
            + model.move_cost("LD_VR", MoveCostCtx("LD_VR"))
        ) * n_entries

    return spill


def _apu_v2_register_spill(target):
    """Functional-only spill stub for APU v2 (SPEC-011).

    Mirrors the placeholder kernel_cycles model: non-comparative constant.
    """

    def spill(reg, n_entries=1):
        return 2 * n_entries

    return spill
