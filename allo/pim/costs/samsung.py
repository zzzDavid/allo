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
It models the first matrix staging of a REDUCE placement (``mac_stage``,
from the conductor's staging schedule, round 37) and, when the placement
fills GRF_A from the bank (``grf_residency`` ``crf``), the conductor's
staging of the vector into every PIM block (``mac_vector_stage``, from its
``stage_vector`` loop, round 132), but not the CRF upload or the readback
unless host moves are supplied, and it prices the logical (unpadded)
shape. It is therefore *not* the quantity PIMSimulator reports for an executed program
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
# Tile rows (release S-13).  issue_order=row_pair with tile_rows=same_bank
# issues a tile's two rows of one bank under one barrier; measured on the
# conductor the execution phase grows by 45,481 cycles for 16 pairs on the
# 8192x256 GEMV and by 45,559 for 16 pairs on the 4096x512 GEMV (VALIDATION
# round 36), 2,843 and 2,847 per pair, far above the device's tRP + tRCDRD
# (14 + 14) because the barrier group waits on the row conflict across the
# whole fabric.  Validity domain of the constant: the 64-channel, 8-block
# geometry the conductor simulates, one result column, host GRF_A fill,
# tiles of exactly two physical rows (G*E = 64 columns, 32 per row); it was
# measured on two reduction/row shapes (16 pairs each) and is not claimed
# beyond that geometry.  tile_rows=split_fibers (any issue order) stages
# every tile's second row in the odd fiber and pays the odd-fiber row-open
# cost per tile (FIBER_TILE_CYCLES, the empirical odd-fiber term above).
ROW_PAIR_SAME_BANK_CYCLES = 2845
# First matrix staging (rounds 37-38, scope change: the estimate includes
# the first staging of the bank-resident matrix, the phase the conductor
# reports as "preload", for every family).  The staging schedule is the
# conductor's (`stage_matrix` in TenonConductor.inc; `samsung_staging_schedule`
# below reproduces its channel-0 bank/row sequence and is checked against
# the conductor's address dump): per pass, tile and PIM block, G*E = 64
# consecutive column writes to the tile's fiber, so the two physical rows
# of a tile are written back to back in one bank unless the selected half
# -> bank map places the second row in the partner fiber.  Each work item
# is charged its own rows' share (`samsung_staging_share`; the shares of a
# kernel's work items partition the staging, checked in the tests), and
# the staging precedes the MAC.  Priced from the device ini
# (HBM2_samsung_2M_16B_x64.ini): a column write every tCCDS = 2 memory
# clocks on the channel (writes alternate banks between blocks), an exposed
# in-run row change in one bank stalling the bus for the write recovery,
# the precharge and the activate less the spacing it replaces,
# tWR + tRP + tRCDWR - tCCDL = 16 + 14 + 10 - 4 = 36, and each bank, in
# parallel with the others, occupied tWR + tRP + tRCDWR = 40 per row it is
# written in.  Measured preload of the 8192x256 and 4096x512 GEMVs: 5,628
# (both rows in one bank: 32 exposed changes per channel) against 4,595
# (second row in the odd fiber: none); the model gives 5,248 against 4,096,
# the right order and a gap of 1,152 against the measured 1,033.  No
# constant is fitted.
ELEMENT_BITS = (
    16  # the PIM datapath element (FP16, PIMBlock::mac); a 256-bit burst holds 16
)
STAGE_WRITE_CYCLES = 2  # tCCDS
STAGE_ROW_CHANGE_CYCLES = 16 + 14 + 10 - 4  # tWR + tRP + tRCDWR - tCCDL
# A bank's occupancy per row it is written in: write recovery, precharge and
# activate (tWR + tRP + tRCDWR = 40); banks work in parallel, so this is a
# latency, not channel time.  The ordered program's preload writes one burst
# per PIM block per reduction element, each at a new row of the even bank
# (TemporalReduce.inc): its staging is bank-bound (K=256: modelled 10,794
# per row tile and column against the measured 12,851, 16% low) while the
# REDUCE programs' is bus-bound (modelled 5,248 / 4,096 against 5,628 /
# 4,595).  Same scope for every family: the first staging of the matrix.
STAGE_ROW_VISIT_CYCLES = 16 + 14 + 10  # tWR + tRP + tRCDWR


def samsung_staging_schedule(
    passes, tiles, fibers, half_fibers, G, E, cols_per_row, blocks
):
    """Channel-0 staging writes of the conductor as (bank, row) per burst, in
    issue order: for each pass, tile and PIM block, ``G*E`` consecutive
    columns of the tile at ``(pass * tiles_on_fiber + local) * G*E``, in the
    tile's fiber (tiles alternate the fibers round-robin) or, when the
    selected half -> bank map splits the rows, half ``h`` in
    ``half_fibers[h]``; the column carries into the row every
    ``cols_per_row`` columns."""
    tile_cols = G * E
    per_fiber = {}
    tile_fiber = []
    for t in range(tiles):
        f = t % max(1, fibers) if half_fibers[0] == half_fibers[1] else 0
        tile_fiber.append(f)
        per_fiber.setdefault(f, []).append(t)
    sequence = []
    for j in range(passes):
        for t in range(tiles):
            f = tile_fiber[t]
            local = per_fiber[f].index(t)
            tile_col = (j * len(per_fiber[f]) + local) * tile_cols
            for block in range(blocks):
                for c in range(tile_cols):
                    half = c // cols_per_row
                    fiber = half_fibers[half] if half_fibers[0] != half_fibers[1] else f
                    row, _col = divmod(tile_col + c, cols_per_row)
                    sequence.append((2 * block + fiber, row))
    return sequence


def samsung_staging_cycles(sequence) -> dict:
    """Price a staging sequence (channel 0): the channel bus takes every
    write (STAGE_WRITE_CYCLES) and stalls for a row change between
    consecutive writes to the same bank (STAGE_ROW_CHANGE_CYCLES); each
    bank, in parallel with the others, is occupied STAGE_ROW_VISIT_CYCLES
    per row it is written in plus its writes.  The staging takes the longer
    of the bus and the busiest bank."""
    exposed = 0
    previous = None
    visits = {}
    writes_per_bank = {}
    last_row = {}
    for bank, row in sequence:
        if previous is not None and previous[0] == bank and previous[1] != row:
            exposed += 1
        previous = (bank, row)
        writes_per_bank[bank] = writes_per_bank.get(bank, 0) + 1
        if last_row.get(bank) != row:
            visits[bank] = visits.get(bank, 0) + 1
            last_row[bank] = row
    bus = len(sequence) * STAGE_WRITE_CYCLES + exposed * STAGE_ROW_CHANGE_CYCLES
    bank = max(
        (
            visits[b] * STAGE_ROW_VISIT_CYCLES + writes_per_bank[b] * STAGE_WRITE_CYCLES
            for b in visits
        ),
        default=0,
    )
    return {
        "writes": len(sequence),
        "exposed_row_changes": exposed,
        "bus_cycles": bus,
        "bank_cycles": bank,
        "cycles": max(bus, bank),
    }


def samsung_staging_share(
    rows,
    reduction,
    *,
    G,
    E,
    lanes,
    cols_per_row=32,
    split_rows=False,
    temporal=False,
    columns=1,
) -> dict:
    """One work item's share of the program's first matrix staging, so that
    the shares of a kernel's work items partition the conductor's staging
    without duplication (checked against the staging dump in the tests).
    ``lanes`` is the number of elements in one burst (the GRF width over
    ``ELEMENT_BITS``, 16 for the 256-bit FP16 registers), which sizes a
    reduction tile (``E x lanes``) and the ordered program's row block; it
    is not ``Register.lanes`` (the entry count the MAC rule reads).

    REDUCE: the item's rows occupy ceil(rows / G) PIM-block row groups; each
    group receives, per reduction tile, one run of G*E consecutive column
    writes in the tile's fiber (64 columns = two physical rows), so the run
    has one exposed row change unless the selected map splits the rows over
    the two fibers (then two runs of G*E/2 in two banks, no exposed change).
    Ordered program: ceil(rows / lanes) blocks receive one write per
    reduction element (and the initial accumulator), each at a new row of
    the even bank, per column.  Returns the bus time (channel occupancy),
    the bank time (a latency, banks in parallel) and the item's latency.
    """
    if isinstance(cols_per_row, bool) or int(cols_per_row) <= 0:
        raise ValueError(
            f"cols_per_row must be a positive integer, got {cols_per_row!r}"
        )
    cols_per_row = int(cols_per_row)
    if temporal:
        blocks = _ceil_div(rows, lanes)
        writes = blocks * (reduction + 1) * columns
        exposed = 0
        bank = (reduction + 1) * columns * (STAGE_ROW_VISIT_CYCLES + STAGE_WRITE_CYCLES)
    else:
        tile = E * lanes
        tiles = max(1, _ceil_div(reduction, tile))
        groups = _ceil_div(rows, G)
        run = G * E
        writes = groups * tiles * run
        if split_rows:
            # per group, each of the two banks: `tiles` runs of run/2 writes,
            # each spanning rows_half physical rows (one at 32 columns per row)
            rows_half = max(1, _ceil_div(run // 2, cols_per_row))
            exposed = groups * tiles * 2 * (rows_half - 1)
            bank = tiles * (
                rows_half * STAGE_ROW_VISIT_CYCLES + (run // 2) * STAGE_WRITE_CYCLES
            )
        else:
            # per group, one bank: `tiles` runs, each spanning rows_per_run
            # physical rows (two at 32 columns per row), one exposed change
            # per row boundary
            rows_per_run = max(1, _ceil_div(run, cols_per_row))
            exposed = groups * tiles * (rows_per_run - 1)
            bank = tiles * (
                rows_per_run * STAGE_ROW_VISIT_CYCLES + run * STAGE_WRITE_CYCLES
            )
    bus = writes * STAGE_WRITE_CYCLES + exposed * STAGE_ROW_CHANGE_CYCLES
    return {
        "writes": writes,
        "exposed_row_changes": exposed,
        "bus_cycles": bus,
        "bank_cycles": bank,
        "latency": max(bus, bank),
    }


def samsung_vector_staging_share(
    reduction,
    *,
    E,
    burst_elements,
    cols_per_row=32,
    columns=1,
    resident=False,
) -> dict:
    """One work item's share of the conductor's vector staging for a
    bank-filled GRF_A (``TenonConductor.inc`` ``stage_vector``): the
    vector's ``k_tiles x E`` bursts of a result column are written at
    consecutive columns of the even bank of *every* PIM block, after the
    matrix (``vector_base`` is row aligned) and before the column's passes,
    once whatever the pass count; the FILL column command later loads
    GRF_A from them.  A host-filled GRF_A has no such staging (its HAB
    broadcast is the LD_A rule).  The replication is per PIM unit, not per
    matrix row, so a work item's share is its unit's copy on the declared
    grid (one unit per work item): the eight units of a channel together
    write the ``8 x k_tiles x E x columns`` bursts the conductor writes per
    channel.  ``k_tiles`` comes from the burst width (``burst_elements``,
    the GRF width over ``ELEMENT_BITS``), as the conductor's does.

    Bus: one STAGE_WRITE_CYCLES per write; consecutive writes alternate
    blocks (the loop is tile, block, entry), so no same-bank row change is
    exposed on the bus.  Bank: one row visit (STAGE_ROW_VISIT_CYCLES) each
    time the staging enters a row of the unit's bank, plus its writes.
    The conductor stages every column's vector in one phase when the
    matrix is resident (``columns`` contiguous spans, one visit per distinct
    row), and otherwise stages each column's vector in its own phase,
    separated by that column's matrix restaging and the previous column's
    passes, so a row revisited by the next column is entered again (the
    visits of the columns' spans add; a span crossing a row boundary visits
    both rows)."""
    if isinstance(cols_per_row, bool) or int(cols_per_row) <= 0:
        raise ValueError(
            f"cols_per_row must be a positive integer, got {cols_per_row!r}"
        )
    if isinstance(burst_elements, bool) or int(burst_elements) <= 0:
        raise ValueError(
            f"burst_elements must be a positive integer, got {burst_elements!r}"
        )
    cols_per_row = int(cols_per_row)
    columns = max(1, int(columns))
    tiles = max(1, _ceil_div(reduction, int(E) * int(burst_elements)))
    per_column = tiles * int(E)
    writes = per_column * columns
    if resident:
        visits = _ceil_div(writes, cols_per_row)
    else:
        visits = sum(
            ((n + 1) * per_column - 1) // cols_per_row
            - (n * per_column) // cols_per_row
            + 1
            for n in range(columns)
        )
    bus = writes * STAGE_WRITE_CYCLES
    bank = visits * STAGE_ROW_VISIT_CYCLES + writes * STAGE_WRITE_CYCLES
    return {
        "tiles": tiles,
        "writes": writes,
        "row_visits": visits,
        "phases": 1 if resident else columns,
        "exposed_row_changes": 0,
        "bus_cycles": bus,
        "bank_cycles": bank,
        "latency": max(bus, bank),
    }


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

    def stage(ctx, share):
        # The first matrix staging (the conductor's preload phase): the bus
        # part occupies the channel, the bank part is latency only (banks in
        # parallel); it precedes the MAC, as the conductor stages first.
        ctx.step(
            latency=share["latency"],
            occupy=[ctx.use(channel, cycles=share["bus_cycles"])],
            name="mac_stage",
        )

    def stage_vector(ctx, share):
        # The vector staging of a bank-filled GRF_A (the conductor's
        # stage_vector, in the preload phase after the matrix): bus part on
        # the channel, bank part latency only; before the MAC's FILLs.
        ctx.step(
            latency=share["latency"],
            occupy=[ctx.use(channel, cycles=share["bus_cycles"])],
            name="mac_vector_stage",
        )

    def temporal_mac(event, ctx):
        # One output row per FP16 lane; the reduction is issued element by
        # element, so each pass costs K steps whatever the fiber layout.
        reduction = max(1, int(event.metrics.get("reduction_extent", 1)))
        extents = tuple(event.metrics.get("loop_extents", ()) or ())
        rows = int(extents[0]) if len(extents) > 1 else 1
        columns = int(extents[1]) if len(extents) == 3 else 1
        lanes16 = int(target.grf_b.width) // 16
        stage(
            ctx,
            samsung_staging_share(
                rows,
                reduction,
                G=int(target.grf_b.slots),
                E=int(target.grf_a.slots),
                lanes=lanes16,
                temporal=True,
                columns=columns,
            ),
        )
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
        split_rows = candidate.get("tile_rows") == "split_fibers"
        if split_rows:
            # Every tile's second row lives in the odd fiber (any issue order).
            latency += tiles * FIBER_TILE_CYCLES
        if candidate.get("issue_order") == "row_pair" and not split_rows:
            groups_per_tile = int(target.grf_b.slots)
            latency += tiles * (groups_per_tile // 2) * ROW_PAIR_SAME_BANK_CYCLES
        # This work item's share of the first matrix staging (its own rows'
        # bursts; the shares of a kernel's work items partition the
        # conductor's staging), before the MAC.  Rows and columns come from
        # the nest the Samsung lowering accepts, [k], [row, k] or [row,
        # column, k] (spmw_samsung.samsung_contraction: row_var the first
        # loop, column_var the second of three, N its trip count), and only
        # from there: the matcher's `batch` metric (batch_dim) names whichever
        # operand carries the free axis in source order and is the row loop
        # when the vector is written first, so it is not read here (rounds
        # 132 to 134).
        extents = tuple(event.metrics.get("loop_extents", ()) or ())
        rows = int(extents[0]) if len(extents) > 1 else 1
        columns = int(extents[1]) if len(extents) == 3 else 1
        G = int(target.grf_b.slots)
        E = int(target.grf_a.slots)
        # Staging counts bursts: a reduction tile is E bursts of
        # grf_a.width / ELEMENT_BITS elements (16 FP16), the conductor's
        # k_tiles.  Register.lanes (the event's `lanes`, the entry count the
        # calibrated MAC term reads) is not the burst width (round 132).
        burst = max(1, int(target.grf_a.width) // ELEMENT_BITS)
        stage(
            ctx,
            samsung_staging_share(
                rows,
                reduction,
                G=G,
                E=E,
                lanes=burst,
                split_rows=split_rows,
            ),
        )
        if not grf_a_fill_is_host_broadcast(event):
            # A bank-filled GRF_A: the conductor also stages the vector (every
            # column's tiles) into every PIM unit before the passes; this
            # work item's share is its unit's copy (S-12, round 132).  A
            # host-filled GRF_A stages nothing (the broadcast is in LD_A).
            # A restaged matrix stages each column's vector in its own phase.
            stage_vector(
                ctx,
                samsung_vector_staging_share(
                    reduction,
                    E=E,
                    burst_elements=burst,
                    columns=columns,
                    resident=bool(candidate.get("stage_resident", False)),
                ),
            )
        occupy_compute(event, ctx, latency, MAC_CYCLES)
        # Several result columns (GEMM N) with a restaged matrix: each extra
        # column rewrites this work item's matrix rows (rows x K/burst
        # bursts) over its pseudo-channel before its pass; resident staging
        # writes once. The first staging is in the mac_stage step above, so
        # only the extra ones appear here.
        if columns > 1 and not candidate.get("stage_resident", False):
            bursts = rows * _ceil_div(reduction, burst)
            restage = (columns - 1) * bursts * HOST_BURST_CYCLES
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
