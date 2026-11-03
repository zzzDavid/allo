# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-form AttAcc selection cost (analytical; no simulator in the loop).

Provenance
----------
* Formulas: the paper's selection cost, ``m8`` (GEMV) and ``serial_q_cf``
  (attention scoring) of ``attacc_cost_cf.py`` (containing repository,
  ``reviews/experiments/results/2026-09-29-attacc-version-crosswalk-v3/
  sources/attacc_cost_cf.py``; paper ``5-evaluation.tex`` "Cost Modeling and
  LLM Decode on AttAcc"), re-expressed over :class:`AttAccCounts`.  The
  archived plugin's replay-based ranking is a different model the paper does
  not use for selection (``5-evaluation.tex:187-189``) and is not here.
* Command counts: :func:`allo.pim.attacc_tiling.attacc_command_counts`,
  validated on 2026-11-02 against DCC (github.com/SPIN-Research-Group/DCC,
  commit ``212f86ea7f08da059836e3a1311d3a11afae8dbe``, checkout dated
  2026-07-07) ``generate_configs_3D`` and its AttAcc trace generator: equal
  space, counts and cost on all 668,575 candidates of the paper's 500
  instances, and the first minimum equals the paper's selected tiling on
  500/500.  No DCC code is used.
* Constants: Ramulator2-AttAcc ``src/dram/impl/HBM3-PIM.cpp`` timing preset
  ``HBM3_5.2Gbps`` (MIT, SAFARI 2023): ``nCCDAB = 6`` (consecutive all-bank
  MACs, power-constrained), ``nCCDL = 4`` (consecutive WRGB/MVSB/MVGB/SFM
  in a bank group), ``nBL = 2`` (burst on the pseudo-channel bus).  The lane
  count (16) is a target fact read from :func:`attacc_geometry`.

Units, scope and validity
-------------------------
Cycles are Ramulator2 ``memory_system_cycles`` of one device (tCK 0.769 ns
at ``HBM3_5.2Gbps``).  Each form is a per-channel makespan of the whole
command inventory: counts divided by the channels the trace addresses.

* GEMV (``m8``): ``max(MAC_AB * nCCDAB + WR_GB * nCCDL,
  MV_SB * nCCDL + ST * nBL) / channels`` -- compute and movement datapaths
  overlap, the longer one bounds the kernel.
* Attention scoring (``serial_q_cf``): ``(MAC_q * nCCDAB + WR_GB + MV_GB +
  MV_SB + SFM + ST) / channels`` with ``MAC_q`` the MAC count with each
  serial extent rounded up to whole 16-lane columns -- score, softmax and
  context run serially per channel.

Validity domain: FP16 GEMV and attention scoring of the studied GPT-13B
decode kernels on one device (QKV 5120 -> 384 and projection 128 -> 5120 per
head group at 8 and 32 heads; attention with d_head 128, L 144..4096).  There
it is a ranking model, not a cycle predictor: against Ramulator2-AttAcc on
337 GEMV / 180 attention samples, Spearman 0.94 / 0.98 and MAPE 30.78% /
14.19% (paper ``5-evaluation.tex``).  Row-buffer state (activations,
precharges), refresh and command issue order are omitted, as the paper's
discussion of its losses states.  Other shapes are priced by the same
formulas without validation (extrapolated).

Accounting
----------
The kernel record is priced once per work item: on the ``MAC`` event of a
GEMV, on the ``SFM_EXP`` event of an attention kernel (the one primitive an
attention record has exactly once).  Every other primitive of the record
(the second MAC, ``SFM_SUM``, ``SFM_DIV``, ``WR_GB``, ``MV_SB``, ``MV_GB``)
is zero-latency: its commands are inside the record's counts.  Host
transfers (``H2D``/``D2H``) are refused: the model has no host link.  The
cost engine keeps integer cycles, so the fractional part of the per-channel
average is dropped (``int``); over the paper's 500 instances this
changes no selected tiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...perf import cost, rule
from ..attacc_tiling import (
    AttAccProgram,
    AttAccTiling,
    attacc_command_counts,
    attacc_geometry,
)


@dataclass(frozen=True)
class AttAccTiming:
    """HBM3-PIM command spacings in memory cycles (``HBM3-PIM.cpp``)."""

    n_ccdab: int = 6
    n_ccdl: int = 4
    n_bl: int = 2


HBM3_5_2GBPS = AttAccTiming()

ATTACC_COST_VALIDITY = (
    "ranking model (paper: Spearman 0.94 GEMV / 0.98 attention, MAPE 30.78% / "
    "14.19% vs Ramulator2-AttAcc) for FP16 GEMV and attention scoring at the "
    "studied GPT-13B per-device shapes; extrapolated elsewhere; row-buffer "
    "state, refresh and issue order are not modelled"
)


def gemv_cycles(counts, timing: AttAccTiming = HBM3_5_2GBPS) -> float:
    """``m8``: max of the compute and movement datapaths, per channel."""
    ch = max(1, counts.channels_used)
    return max(
        counts.mac_ab / ch * timing.n_ccdab + counts.wr_gb / ch * timing.n_ccdl,
        counts.mv_sb / ch * timing.n_ccdl + counts.st / ch * timing.n_bl,
    )


def attention_cycles(counts, lanes: int, timing: AttAccTiming = HBM3_5_2GBPS) -> float:
    """``serial_q_cf``: lane-quantized MACs plus data-bus moves, per channel."""
    ch = max(1, counts.channels_used)
    s1, s2 = counts.serial_reduction, counts.serial_outputs

    def columns(extent):
        return math.ceil(extent / lanes)

    mac_q = (
        counts.iterations
        * counts.rows_per_iteration
        * ch
        * (columns(s2) * lanes * columns(s1) + columns(s1) * lanes * columns(s2))
    )
    data_bus = counts.wr_gb + counts.mv_gb + counts.mv_sb + counts.sfm + counts.st
    return (mac_q * timing.n_ccdab + data_bus) / ch


def attacc_kernel_cycles(
    program: AttAccProgram, timing: AttAccTiming = HBM3_5_2GBPS
) -> float:
    """Closed-form cycles of one kernel program (the selection cost)."""
    if program.kernel.kind == "gemv":
        return gemv_cycles(program.counts, timing)
    return attention_cycles(program.counts, program.geometry.lanes, timing)


def _candidate_program(event, geometry) -> AttAccProgram:
    candidate = event.metrics.get("candidate", {}) or {}
    kernel = candidate.get("attacc_kernel")
    tiling = candidate.get("attacc_tiling")
    if kernel is None or tiling is None:
        raise ValueError(
            "attacc_cost prices the kernel record of a placement and needs "
            "extra['attacc_kernel'] and extra['attacc_tiling']: use the AttAcc "
            "enumerator, or allo.spmw_attacc.attacc_placement for layout="
        )
    tiling = AttAccTiling.coerce(tiling)
    return AttAccProgram(
        kernel, tiling, attacc_command_counts(kernel, tiling, geometry), geometry
    )


def attacc_cost_program(timing: AttAccTiming = HBM3_5_2GBPS, *, name=None):
    """The AttAcc cost program for ``timing`` (``attacc_cost`` is the default)."""

    def attacc_cost(target):
        geometry = attacc_geometry(target)
        channel = target.unit("channel")

        def record(event, ctx, kind, label):
            program = _candidate_program(event, geometry)
            if program.kernel.kind != kind:
                ctx.step(
                    cycles=0,
                    occupy=[ctx.use(event.primitive)],
                    name=f"{label}_in_record",
                )
                return
            # Integer cycle unit of the cost engine (module docstring).
            latency = int(attacc_kernel_cycles(program, timing))
            ctx.step(
                latency=latency,
                occupy=[
                    ctx.use(event.primitive, cycles=latency),
                    ctx.use(channel, cycles=latency),
                ],
                name=f"{program.kernel.kind}_record",
            )

        @rule(target.op("MAC"))
        def mac(event, ctx):
            record(event, ctx, "gemv", "mac")

        @rule(target.op("SFM_EXP"))
        def sfm_exp(event, ctx):
            record(event, ctx, "attention", "sfm_exp")

        def in_record(label):
            def priced_inside(event, ctx):
                ctx.step(
                    cycles=0,
                    occupy=[ctx.use(event.primitive)],
                    name=f"{label}_in_record",
                )

            return priced_inside

        for op_name in ("SFM_SUM", "SFM_DIV"):
            rule(target.op(op_name))(in_record(op_name.lower()))
        for move_name in ("WR_GB", "MV_SB", "MV_GB"):
            rule(target.move(move_name))(in_record(move_name.lower()))

        # The model has no host link: the studied kernels keep weights, KV
        # cache and activations resident in the banks, and the paper times
        # PIM kernels only.  A recorded host transfer is refused, not priced 0.
        def host_transfer(event, ctx):
            raise NotImplementedError(
                f"attacc_cost has no host-link model; {event.primitive.name} "
                "cannot be priced (AttAcc kernels are timed with operands "
                "resident in the banks)"
            )

        for move_name in ("H2D", "D2H"):
            rule(target.move(move_name))(host_transfer)

    attacc_cost.__name__ = name or "attacc_cost"
    return cost(attacc_cost, target="attacc", name=name or "attacc_cost")


attacc_cost = attacc_cost_program()


__all__ = [
    "ATTACC_COST_VALIDITY",
    "AttAccTiming",
    "HBM3_5_2GBPS",
    "attacc_cost",
    "attacc_cost_program",
    "attacc_kernel_cycles",
    "attention_cycles",
    "gemv_cycles",
]
