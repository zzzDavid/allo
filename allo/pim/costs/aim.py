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


# One 256-bit DRAM column contains 16 BF16 operands.  MAC_ABK launches one
# column operation in every one of the channel's 16 banks.
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


def _columns(event, *, default_bank_fanout=1):
    elements = _metric(event, "reduction_extent")
    width = LANES_PER_PU * _bank_fanout(event, default_bank_fanout)
    return max(1, _ceil_div(elements, width))


@cost(target="aim")
def aim_cost(target):
    """Interpret AiM operations over target-declared spatial resources."""
    # Model revision: linear-layout-v1. MAC width is the realized bank image
    # supplied in candidate.bank_fanout.
    channel = target.unit("channel")
    bank_group = target.unit("bank_group")
    bank = target.unit("bank")
    gb = target.gb

    def single_bank_mac(event, ctx):
        columns = _columns(event)
        issue = max(CMD_MAC, columns * N_CCD)
        latency = N_RCD_RD_MAC + (columns - 1) * N_CCD + CMD_MAC
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
        columns = _columns(event, default_bank_fanout=BANKS_PER_CHANNEL)
        latency = N_RCD_RD_MAC + (columns - 1) * N_CCD + CMD_MAC
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

    @rule(target.move("COPY_BKGB"))
    def copy_bank_to_gb(event, ctx):
        channel_move(event, ctx, N_RCD_RDCP + CMD_COPY, "copy_bkgb", gb)

    @rule(target.move("COPY_GBBK"))
    def copy_gb_to_bank(event, ctx):
        channel_move(event, ctx, N_RCD_WRCP + CMD_COPY, "copy_gbbk", gb)
