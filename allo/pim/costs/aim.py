# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cycle program for SK hynix GDDR6-AiM.

This module is intentionally ordinary Python: calibration means changing these
constants or formulas after inspecting a trace or running a microprofile.  The
structural target in :mod:`allo.pim.targets` contains no cycle information.

The initial values come from ``aim_simulator`` commit
``0f28a07bdb83e42b9305ad3d45410ebd3aa2c091``:
``src/dram/impl/GDDR6.cpp`` (``GDDR6_AiM_timing`` and command latencies) and
``src/dram/impl/GDDR6.cpp``/``request.h`` (AiM command scopes).  Cycle counts
are reported directly; no frequency or wall-time conversion is involved.
"""

from __future__ import annotations

import math

from ...perf import cost, rule
from .. import aim_lowering
from ..aim_lowering import _target_geometry
from ..schedule_search import InfeasibleSchedule


# One 256-bit DRAM column contains 16 BF16 operands.  MAC_ABK launches the
# same column operation in every one of the channel's 16 banks: those banks
# hold independent output rows, not disjoint pieces of one reduction.
LANES_PER_PU = 16
BANKS_PER_CHANNEL = 16

# GDDR6_AiM_timing (rate=2000).  Names intentionally mirror the simulator so
# an agent can compare a microprofile and edit the relevant source constant.
N_BL = 2
N_CL = 50
N_RCD_RD = 36
N_RCD_RD_MAC = 56
N_RCD_EWMUL = 25
N_RCD_RD_AF = 86
N_RCD_RDCP = 66
N_RCD_WR = 28
N_RCD_WRCP = 48
N_RP = 32
N_RAS = 54
N_CWL = 6
N_CCD = 2
N_MODCH = 32

# Per-command execution latency from ``m_command_meta``.
CMD_MAC = 1
CMD_EWMUL = 1
CMD_AF = 1
CMD_COPY = 1
CMD_WR_GB = 3
CMD_RD_REG = 2

# Knob-aware contraction pricing (spec 002 A7).  The simulator keeps rows open,
# so a MAC that targets a new (channel mask, row) pays a precharge plus the
# MAC-row activation; this term is what separates batch mappings that issue
# the same MAC count.
BATCH_MAPPING_ROW_ACTIVATE_CYCLES = N_RP + N_RCD_RD_MAC
# Each WR_BIAS run opens a new accumulator reuse group (a mode change).
REUSE_GROUP_SWITCH_CYCLES = N_MODCH

# Host DRAM traffic (spec 002 A2): the first burst of a transfer opens the
# row; each further 256-bit burst issues after nCCD.
HOST_WRITE_BURST_CYCLES = N_RCD_WR + N_CWL + N_BL
HOST_READ_BURST_CYCLES = N_RCD_RD + N_CL
HOST_BURST_BYTES = 32


def _ceil_div(value, divisor):
    if divisor <= 0:
        raise ValueError("cost divisor must be positive")
    return math.ceil(value / divisor)


def _metric(event, name, default=1):
    return max(0, int(event.metrics.get(name, default)))


def _bank_fanout(event, default):
    candidate = event.metrics.get("candidate", {}) or {}
    fanout = max(1, int(candidate.get("bank_fanout", default)))
    if fanout > BANKS_PER_CHANNEL or fanout & (fanout - 1):
        raise ValueError(
            f"AiM layout bank_fanout must be a power of two up to "
            f"{BANKS_PER_CHANNEL}, got {fanout}"
        )
    return fanout


def _columns(event):
    elements = _metric(event, "reduction_extent")
    return max(1, _ceil_div(elements, LANES_PER_PU))


def _mac_launches(event, *, default_bank_fanout=1):
    """Number of physical MAC commands represented by one loop-nest event.

    ``iterations`` contains every enclosing loop, while ``reduction_extent``
    and the matcher-stamped ``batch`` identify the reduction and independent
    vector axes.  The quotient is therefore the number of output rows owned
    by this work item.  An all-bank command computes one such row per bank in
    parallel; it does not make the reduction itself wider.
    """
    reduction = max(1, _metric(event, "reduction_extent"))
    batch = max(1, _metric(event, "batch"))
    iterations = max(1, _metric(event, "iterations", reduction * batch))
    outputs = max(1, _ceil_div(iterations, reduction * batch))
    fanout = _bank_fanout(event, default_bank_fanout)
    return batch * _ceil_div(outputs, fanout)


def _knob_candidate(event):
    """The candidate's AiM knob/shape record, or None for knob-free events."""
    candidate = event.metrics.get("candidate", {}) or {}
    if candidate.get("aim_shape") is None:
        return None
    if candidate.get("batch_mapping") in (None, "none"):
        return None
    return candidate


def _contraction_profile_cycles(candidate, geometry):
    """Serial channel cycles of one replica's lowered contraction."""
    shape = candidate["aim_shape"]
    try:
        operation = aim_lowering.build_contraction(
            shape,
            {
                "batch_mapping": candidate["batch_mapping"],
                "reuse_group": candidate.get("reuse_group"),
                "replica_partitions": candidate.get("replica_partitions"),
                "replicas": 1,
            },
            activation=shape.get("activation", False),
        )
        profile = aim_lowering.aim_contraction_profile(operation, geometry)
    except ValueError as error:
        raise InfeasibleSchedule(f"AiM contraction is not realizable: {error}") from error
    counts = profile["counts"]
    mac_lines = profile["mac_lines"]
    mac_cycles = (profile["mac_columns"] - mac_lines) * N_CCD + mac_lines * CMD_MAC
    if candidate.get("bank_scope") == "sbk":
        mac_cycles *= geometry.banks
    register_io = N_MODCH + CMD_WR_GB
    register_read = N_MODCH + CMD_RD_REG
    return (
        mac_cycles
        + profile["row_activations"] * BATCH_MAPPING_ROW_ACTIVATE_CYCLES
        + profile["reuse_groups"] * REUSE_GROUP_SWITCH_CYCLES
        + counts.get("WR_GB", 0) * register_io
        + counts.get("WR_BIAS", 0) * register_io
        + counts.get("RD_MAC", 0) * register_read
        + counts.get("AF", 0) * (N_RCD_RD_AF + CMD_AF)
        + counts.get("RD_AF", 0) * register_read
    )


def _ewmul_profile_cycles(elements, partitions, geometry):
    span = _ceil_div(max(1, elements), partitions * geometry.bank_groups)
    cycles = 0
    for row_start in range(0, span, geometry.row_elements):
        columns = _ceil_div(min(geometry.row_elements, span - row_start), geometry.lanes)
        cycles += N_RCD_EWMUL + (columns - 1) * N_CCD + CMD_EWMUL
    return cycles


@cost(target="aim")
def aim_cost(target):
    """Interpret AiM operations over target-declared spatial resources."""
    # Model revision: linear-layout-v1. The realized bank image determines
    # how many output rows one command produces, never its reduction width.
    channel = target.unit("channel")
    bank_group = target.unit("bank_group")
    bank = target.unit("bank")
    gb = target.gb
    geometry = _target_geometry(target)

    def profiled_mac(event, ctx, candidate, name):
        # The profile already prices the WR_GB/WR_BIAS/RD_MAC/AF commands the
        # contraction issues; the per-bucket move events become zero-cost.
        latency = _contraction_profile_cycles(candidate, geometry)
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(channel, cycles=latency),
                ctx.use(gb, cycles=latency),
            ],
            name=name,
        )

    def single_bank_mac(event, ctx):
        candidate = _knob_candidate(event)
        if candidate is not None:
            profiled_mac(event, ctx, candidate, "mac_sbk")
            return
        columns = _columns(event)
        launches = _mac_launches(event)
        issue = max(CMD_MAC, columns * N_CCD)
        latency = N_RCD_RD_MAC + (columns - 1) * N_CCD + CMD_MAC
        with ctx.repeat(launches):
            ctx.step(
                latency=latency,
                occupy=[
                    ctx.use(event.primitive, cycles=latency),
                    ctx.use(bank, cycles=latency),
                    ctx.use(channel, cycles=issue),
                    ctx.use(gb, cycles=issue),
                ],
                name="mac_sbk",
            )

    rule(target.op("MAC"))(single_bank_mac)

    @rule(target.op("MAC_ABK"))
    def all_bank_mac(event, ctx):
        candidate = _knob_candidate(event)
        if candidate is not None:
            profiled_mac(event, ctx, candidate, "mac_abk")
            return
        columns = _columns(event)
        launches = _mac_launches(event, default_bank_fanout=BANKS_PER_CHANNEL)
        latency = N_RCD_RD_MAC + (columns - 1) * N_CCD + CMD_MAC
        with ctx.repeat(launches):
            ctx.step(
                latency=latency,
                occupy=[
                    ctx.use(event.primitive, cycles=latency),
                    ctx.use(channel, cycles=latency),
                    ctx.use(gb, cycles=latency),
                ],
                name="mac_abk",
            )

    @rule(target.op("MUL"))
    def elementwise_mul(event, ctx):
        candidate = event.metrics.get("candidate", {}) or {}
        partitions = candidate.get("replica_partitions")
        if partitions not in (None, "none"):
            # Each partition owns ceil(elements / (p * bank_groups)) elements;
            # one EWMUL per bank row of that span (spec 002 A7).
            latency = _ewmul_profile_cycles(
                _metric(event, "iterations"), int(partitions), geometry
            )
            ctx.step(
                latency=latency,
                occupy=[
                    ctx.use(event.primitive, cycles=latency),
                    ctx.use(bank_group, cycles=latency),
                    ctx.use(channel, cycles=latency),
                ],
                name="ewmul",
            )
            return
        columns = max(1, _ceil_div(_metric(event, "iterations"), LANES_PER_PU))
        latency = N_RCD_EWMUL + (columns - 1) * N_CCD + CMD_EWMUL
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive),
                ctx.use(bank_group),
                ctx.use(channel, cycles=max(1, columns * N_CCD)),
                ctx.use(gb),
            ],
            name="ewmul",
        )

    @rule(target.op("ADD"))
    def elementwise_add(event, ctx):
        # EWADD is a GPR-vector ALU instruction and does not open a bank row.
        vectors = max(1, _ceil_div(_metric(event, "iterations"), LANES_PER_PU))
        ctx.step(
            cycles=vectors,
            occupy=[ctx.use(event.primitive), ctx.use(channel)],
            name="ewadd",
        )

    @rule(target.op("AF"))
    def activation(event, ctx):
        if _knob_candidate(event) is not None:
            ctx.step(cycles=0, occupy=[ctx.use(event.primitive)], name="af_in_profile")
            return
        vectors = max(1, _ceil_div(_metric(event, "iterations"), LANES_PER_PU))
        latency = N_RCD_RD_AF + (vectors - 1) * N_CCD + CMD_AF
        ctx.step(
            latency=latency,
            occupy=[ctx.use(event.primitive), ctx.use(channel), ctx.use(target.af_lut)],
            name="af",
        )

    def bank_move(event, ctx, latency, name):
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(bank, cycles=latency),
                ctx.use(channel, cycles=min(latency, N_CCD)),
            ],
            name=name,
        )

    @rule(target.move("WR_SBK"))
    def write_single_bank(event, ctx):
        bank_move(event, ctx, N_RCD_WR + N_CWL + N_BL, "wr_sbk")

    @rule(target.move("ST_SBK"))
    def store_single_bank(event, ctx):
        bank_move(event, ctx, N_RCD_WR + N_CWL + N_BL, "st_sbk")

    @rule(target.move("RD_SBK"))
    def read_single_bank(event, ctx):
        # ACT -> RD -> PRE completes after nRAS + nRP in the simulator.
        bank_move(event, ctx, N_RAS + N_RP, "rd_sbk")

    def channel_move(event, ctx, latency, name, *resources):
        if _knob_candidate(event) is not None and name in (
            "wr_gb",
            "wr_bias",
            "rd_mac",
            "rd_af",
        ):
            latency = 0
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(channel, cycles=latency),
                *(ctx.use(resource, cycles=latency) for resource in resources),
            ],
            name=name,
        )

    @rule(target.move("WR_ABK"))
    def write_all_banks(event, ctx):
        channel_move(event, ctx, N_CWL + N_BL + N_RP, "wr_abk")

    @rule(target.move("WR_GB"))
    def write_global_buffer(event, ctx):
        channel_move(event, ctx, N_MODCH + CMD_WR_GB, "wr_gb", gb)

    @rule(target.move("WR_BIAS"))
    def write_bias(event, ctx):
        channel_move(event, ctx, N_MODCH + CMD_WR_GB, "wr_bias", target.mac_reg)

    @rule(target.move("RD_MAC"))
    def read_mac_register(event, ctx):
        channel_move(event, ctx, N_MODCH + CMD_RD_REG, "rd_mac", target.mac_reg)

    @rule(target.move("RD_AF"))
    def read_af_register(event, ctx):
        channel_move(event, ctx, N_MODCH + CMD_RD_REG, "rd_af", target.af_reg)

    host = target.unit("host")

    def host_transfer(event, ctx, first_burst, name):
        nbytes = max(0, int(event.metrics.get("bytes", 0)))
        bursts = max(1, _ceil_div(nbytes, HOST_BURST_BYTES))
        latency = first_burst + (bursts - 1) * N_CCD
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(host, cycles=latency),
            ],
            name=name,
        )

    @rule(target.move("SCATTER_BANKS"))
    def host_scatter(event, ctx):
        host_transfer(event, ctx, HOST_WRITE_BURST_CYCLES, "host_scatter")

    @rule(target.move("BCAST_BANKS"))
    def host_broadcast(event, ctx):
        host_transfer(event, ctx, HOST_WRITE_BURST_CYCLES, "host_broadcast")

    @rule(target.move("GATHER_BANKS"))
    def host_gather(event, ctx):
        host_transfer(event, ctx, HOST_READ_BURST_CYCLES, "host_gather")

    @rule(target.move("COPY_BKGB"))
    def copy_bank_to_gb(event, ctx):
        channel_move(event, ctx, N_RCD_RDCP + CMD_COPY, "copy_bkgb", gb)

    @rule(target.move("COPY_GBBK"))
    def copy_gb_to_bank(event, ctx):
        channel_move(event, ctx, N_RCD_WRCP + CMD_COPY, "copy_gbbk", gb)
