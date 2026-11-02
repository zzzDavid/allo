# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung HBM-PIM cycle program.

This file is the calibration surface. Agents update ordinary Python constants,
formulas, and branches here after running microbenchmarks. The structural target
in allo.pim.targets intentionally contains no cycle information.

What the estimate reports. The graph (``spmw_plan.build_execution_graph``)
has one stream per work item of the declared 16 x 8 grid: the GRF fills of
its register-placed operands, its MAC (or elementwise op), the accumulator
drain, and explicit host moves when the workload records ``host_xfer.*``.
The unit is the memory-clock cycle of these modeled activities on the
declared device, scheduled over the pim / pseudo_channel / host resources.
It does not model the matrix preload, the CRF upload or the readback unless
host moves are supplied, and it prices the logical (unpadded) shape. It is
therefore *not* the quantity PIMSimulator reports for an executed program
(``RunResult.cycles``: padded preload + execution + readback on the
simulated 64-channel system, ``RunResult.extra["timing_scope"]``); the two
are compared only by candidate ranking
(``docs/release/design/samsung-execution.md`` section 6).

MAC rule provenance (2026-11-02). Measured on PIMSimulator upstream
SAITPublic/PIMSimulator 3703d1f19c8f027360cc33a3243eb271e3bb6898 plus the
Tenon overlay (experiments/simulators/setup/PIMSimulator, pim_driver sha256
0ae6a3d9...2df2), FP16 GEMV M x K = 4096x256, 4096x512, 8192x256, executing
the compiled programs of each placement:

    placement (CRF fill)   4096x256   4096x512   8192x256   (total cycles)
    dual_fiber, host        4,435      8,041      8,247
    bank_row,   host        4,389      7,947      8,153
    dual_fiber, crf         4,560      8,410      8,359
    bank_row,   crf         4,589      8,458      8,397

The two bank fibers of a PIM unit share the pseudo-channel column-command
bus, so their MAC column commands serialize: the MAC term does not shrink
with the fiber count. Each reduction tile placed on a fiber other than the
first pays FIBER_TILE_CYCLES: the execution phase of dual_fiber exceeds
bank_row (host fill) by 46, 94 and 94 cycles, i.e. 46-47 cycles per
odd-fiber tile per pass. Ranking agreement with the simulator is judged per
candidate pair: a pair agrees when both order it the same way; simulated
differences below 0.5% are ties (docs/release/design/samsung-execution.md
sections 6 and 7). Absolute values are not comparable (scope above).

Ordered (temporal) program provenance (2026-11-02). Same simulator build,
conductor ``pim_driver --op GENERIC --op-kind TEMPORAL_REDUCE`` executing
the five-word ordered CRF (``spmw_samsung.SamsungCtx._emit_temporal``), one
result column, seed-0 FP16 data. One row tile is 64 channels x 8 PIM units x
16 lanes = 8192 rows; cycles per (row tile, column):

    K      preload   exec     readback   total
    100     5,051     5,859      67      10,977
    128     6,451     7,174      67      13,692
    256    12,851    13,889      67      26,807   (M = 2048, 4096, 8192)
    512    25,651    26,967      67      52,685   (M = 4096, 8192)

(M = 16384, K = 256: two tiles, 53,615.) The execution phase is linear in K:
51.4 cycles per reduction element (one host broadcast of the element to
every channel, a barrier, one MAC column command) plus 681 cycles per pass
(least squares, within 1.2% at every point). The rule prices that phase for
each pass a work item needs (one pass per 16 output rows of a PIM unit);
like the REDUCE rule it leaves the preload and readback out of scope. The
4096x256 GEMV with initial value 0.5 measures 26,807 cycles and matches
source-order FP16 for 4096/4096 rows.
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
# Row-open cost of a reduction tile on a second bank fiber (measured, see the
# module docstring): 46-47 PIMSimulator cycles per odd-fiber tile per pass.
FIBER_TILE_CYCLES = 47
# Ordered program (module docstring): per reduction element and per pass.
TEMPORAL_STEP_CYCLES = 51
TEMPORAL_PASS_CYCLES = 681


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

    def temporal_mac(event, ctx):
        # One output row per FP16 lane; the reduction is issued element by
        # element, so each pass costs K steps whatever the fiber layout.
        reduction = max(1, int(event.metrics.get("reduction_extent", 1)))
        extents = tuple(event.metrics.get("loop_extents", ()) or ())
        rows = int(extents[0]) if len(extents) > 1 else 1
        columns = int(extents[1]) if len(extents) == 3 else 1
        lanes16 = int(target.grf_b.width) // 16
        passes = _ceil_div(rows, lanes16) * columns
        latency = passes * (reduction * TEMPORAL_STEP_CYCLES + TEMPORAL_PASS_CYCLES)
        occupy_compute(event, ctx, latency, MAC_CYCLES)

    @rule(target.op("MAC"))
    def mac(event, ctx):
        candidate = event.metrics.get("candidate", {}) or {}
        if candidate.get("reduction_program") == "temporal":
            temporal_mac(event, ctx)
            return
        fibers = max(1, int(event.metrics.get("n_fibers", 1)))
        if "lanes" not in event.metrics:
            raise ValueError(
                "Samsung MAC event has no 'lanes' metric: the placement must put "
                "the MAC in a GRF, whose Register.lanes the plan reads"
            )
        lanes = max(1, int(event.metrics["lanes"]))
        reduction = max(0, int(event.metrics.get("reduction_extent", 1)))
        # Both fibers' MAC column commands share the channel bus: no
        # division by the fiber count.
        chunks = _ceil_div(reduction, lanes)
        tile = int(target.grf_a.slots) * (int(target.grf_a.width) // 16)
        tiles = _ceil_div(reduction, tile)
        later_fiber_tiles = tiles - _ceil_div(tiles, fibers)
        latency = (
            chunks * MAC_CYCLES
            + fibers * JUMP_CYCLES
            + later_fiber_tiles * FIBER_TILE_CYCLES
        )
        occupy_compute(event, ctx, latency, MAC_CYCLES)
        # Several reduction columns (GEMM N, batched GEMV) with a restaged
        # matrix: each extra column rewrites this work item's matrix rows
        # (rows x K/16 bursts) over its pseudo-channel before its pass;
        # resident staging writes once. The first staging is outside this
        # graph's scope (module docstring), so only the extra ones appear.
        # Columns come from the nest the Samsung lowering accepts, [row, k]
        # or [row, column, k] (spmw_samsung.samsung_contraction), not from
        # the matcher's batch_dim, which depends on source operand order.
        extents = tuple(event.metrics.get("loop_extents", ()) or ())
        columns = int(extents[1]) if len(extents) == 3 else 1
        batch = max(columns, int(event.metrics.get("batch", 1)))
        candidate = event.metrics.get("candidate", {}) or {}
        if batch > 1 and not candidate.get("stage_resident", False):
            rows = int(extents[0]) if len(extents) > 1 else 1
            lanes16 = int(target.grf_a.width) // 16
            bursts = rows * _ceil_div(reduction, lanes16)
            restage = (batch - 1) * bursts * HOST_BURST_CYCLES
            ctx.step(
                latency=restage,
                occupy=[ctx.use(channel, cycles=restage)],
                name="mac_restage",
            )

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

    def grf_a_fill_is_host_broadcast(event):
        # GRF_A residency (spmw_knobs grf_residency): "host" fills GRF_A with
        # a HAB broadcast through PIM_REG_RA, "crf" with a CRF FILL from the
        # bank (after the operand was scattered to every bank).
        candidate = event.metrics.get("candidate", {}) or {}
        residency = candidate.get("grf_residency", {}) or {}
        return bool(residency) and all(mode == "host" for mode in residency.values())

    @rule(target.move("LD_A"))
    def load_a(event, ctx):
        if grf_a_fill_is_host_broadcast(event):
            # One host write per GRF_A entry on the channel bus.
            entries = int(target.grf_a.slots)
            device_move(event, ctx, HOST_SETUP_CYCLES + entries * HOST_BURST_CYCLES)
        else:
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
