# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung HBM-PIM cycle program.

This file is the calibration surface. Agents update ordinary Python constants,
formulas, and branches here after running microbenchmarks. The structural target
in allo.pim.targets intentionally contains no cycle information.
"""

from __future__ import annotations

import math

from ...perf import cost, rule


# Cycle constants migrated from the initial PIMSimulator/datasheet model.
# They are source code by design: calibration means editing and testing this
# program, not fitting a runtime profile object.
MAC_CYCLES = 4
ELEMENTWISE_CYCLES = 4
JUMP_CYCLES = 1
BANK_LOAD_CYCLES = 26
BANK_STORE_CYCLES = 14
HOST_SETUP_CYCLES = 2
HOST_BURST_CYCLES = 1
CRF_PROGRAM_CYCLES = 2


def _ceil_div(value, divisor):
    if divisor <= 0:
        raise ValueError("cost divisor must be positive")
    return math.ceil(value / divisor)


@cost(target="samsung_hbm_pim")
def samsung_cost(target):
    """Interpret Samsung operations over unit/op/memory/move handles."""
    channel = target.unit("pseudo_channel")
    pim = target.unit("pim")
    host = target.unit("host")

    def occupy_compute(event, ctx, latency, issue_cycles):
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=issue_cycles),
                ctx.use(pim, cycles=latency),
                ctx.use(channel, cycles=issue_cycles),
            ],
            name=event.primitive.name.lower(),
        )

    @rule(target.op("MAC"))
    def mac(event, ctx):
        fibers = max(1, int(event.metrics.get("n_fibers", 1)))
        lanes = max(1, int(event.metrics.get("lanes", 8)))
        reduction = max(0, int(event.metrics.get("reduction_extent", 1)))
        chunks = _ceil_div(reduction, lanes * fibers)
        latency = chunks * MAC_CYCLES + fibers * JUMP_CYCLES
        occupy_compute(event, ctx, latency, MAC_CYCLES)

    def elementwise(event, ctx):
        iterations = max(1, int(event.metrics.get("iterations", 1)))
        latency = iterations * ELEMENTWISE_CYCLES
        occupy_compute(event, ctx, latency, ELEMENTWISE_CYCLES)

    for operation_name in ("MUL", "ADD", "MAD", "RELU"):
        rule(target.op(operation_name))(elementwise)

    def device_move(event, ctx, latency):
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(pim, cycles=latency),
                ctx.use(channel, cycles=min(latency, ELEMENTWISE_CYCLES)),
            ],
            name=event.primitive.name.lower(),
        )

    @rule(target.move("LD_A"))
    def load_a(event, ctx):
        device_move(event, ctx, BANK_LOAD_CYCLES)

    @rule(target.move("LD_B"))
    def load_b(event, ctx):
        device_move(event, ctx, BANK_LOAD_CYCLES)

    @rule(target.move("ST_A"))
    def store_a(event, ctx):
        device_move(event, ctx, BANK_STORE_CYCLES)

    @rule(target.move("ST_B"))
    def store_b(event, ctx):
        device_move(event, ctx, BANK_STORE_CYCLES)

    @rule(target.move("JUMP"))
    def jump(event, ctx):
        ctx.step(
            cycles=JUMP_CYCLES,
            occupy=[
                ctx.use(event.primitive),
                ctx.use(channel),
            ],
            name="jump",
        )

    def host_transfer(event, ctx):
        nbytes = max(0, int(event.metrics.get("bytes", 0)))
        burst_bytes = max(1, int(event.metrics.get("burst_bytes", 32)))
        latency = HOST_SETUP_CYCLES + _ceil_div(nbytes, burst_bytes) * HOST_BURST_CYCLES
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(host, cycles=latency),
            ],
            name=event.primitive.name.lower(),
        )

    for move_name in (
        "BCAST_GRF_A",
        "BCAST_GRF_B",
        "BCAST_SRF",
        "SCATTER_BANKS",
        "GATHER_BANKS",
    ):
        rule(target.move(move_name))(host_transfer)

    @rule(target.move("PROGRAM_CRF"))
    def program_crf(event, ctx):
        ctx.step(
            cycles=CRF_PROGRAM_CYCLES,
            occupy=[
                ctx.use(event.primitive),
                ctx.use(host),
            ],
            name="program_crf",
        )
