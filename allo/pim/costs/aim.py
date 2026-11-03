# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cycle program for SK hynix GDDR6-AiM.

This module is intentionally ordinary Python: calibration means changing these
constants or formulas after inspecting a trace or running a microprofile.  The
structural target in :mod:`allo.pim.targets` contains no cycle information and
every geometry fact (lanes, banks, channels) is read from it.

Units and scope
---------------
Cycles are ramulator2 ``memory_system_cycles`` of ``aim_simulator`` commit
``0f28a07bdb83e42b9305ad3d45410ebd3aa2c091``.  ``GDDR6.cpp`` recomputes
``tCK_ps = 1e6 / (rate / 2)`` for every preset (line 602), so with
``GDDR6_AiM_timing``'s rate 2000 one cycle is 1 ns; no other wall-time
conversion is involved.  Device terms come from ``GDDR6_AiM_timing`` and
``m_command_meta`` in ``src/dram/impl/GDDR6.cpp`` and ``request.h``.

* Contractions (MAC/MAC_ABK) are priced from the lowered command stream of the
  exact ``AimContraction`` record codegen emits (``build_contraction`` with the
  placement's knob values; absent knobs resolve as in codegen), serially per
  channel group: one cost route for enumerated and ``layout=`` placements.
  Calibration (``tests/pim/test_aim_program.py``, GEMV 256x256 mapping [32],
  simulator run): ABK 223 estimated vs 222 simulated; SBK 1528 vs 1527.
  Validity domain: single-issue command streams of the installed simulator;
  every command is serialized (no overlap of a row activation with the
  previous group's register traffic), refresh and inter-channel bus
  contention are not modelled.  Observed error band beyond the calibration
  probe: on the attention token-partition workloads
  (``tests/pim/test_aim_token_partition.py``, QK + SV, 1-32 heads, 128-4096
  tokens) estimates exceed simulated cycles by 1.25x to 1.59x; the estimate is
  a ranking model there, not a cycle prediction.
* Host link transfers (scatter/gather/broadcast) are priced per 256-bit burst
  of the *moved slice* (``host_slice`` metric when the execution graph carries
  it, the whole buffer otherwise); a transfer to a buffer declared resident
  costs nothing.  The first burst opens a row, later bursts issue every nCCD.
  This mirrors the ``R/W MEM`` lines lowering emits; it is not a model of a
  host-side PCIe/CXL link.
* Host reduction steps (token/reduction partitioning) are priced with a
  declared host-link scenario: 32 GB/s and zero fixed latency (paper
  5-evaluation.tex "Token Partitioning"; harness
  ``reviews/experiments/aim_attention_campaign.py`` ``host_link_bytes_per_ns=32``,
  ``host_link_fixed_latency_ns=0``), one 256-bit GPR entry per RD_MAC
  readout, plus an analytical one-cycle host SIMD add per 256-bit vector.
  These are scenario terms, not simulator measurements; the simulator never
  executes host steps.
"""

from __future__ import annotations

import math

from ...perf import cost, rule
from .. import aim_lowering
from ..aim_lowering import _target_geometry
from ..schedule_search import InfeasibleSchedule


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

# Contraction pricing (spec 002 A7).  The simulator keeps rows open: a MAC
# line whose banks have no open row pays the MAC-row activation only; one that
# replaces an open row also pays the precharge.  The switch term is what
# separates batch mappings that issue the same MAC count.  Calibrated against
# the simulator on the GEMV probe (see module docstring): every MAC_SBK of an
# SBK expansion targets a fresh bank and pays N_RCD_RD_MAC.
FIRST_ROW_ACTIVATE_CYCLES = N_RCD_RD_MAC
BATCH_MAPPING_ROW_ACTIVATE_CYCLES = N_RP + N_RCD_RD_MAC
# Each WR_BIAS run opens a new accumulator reuse group (a mode change).
REUSE_GROUP_SWITCH_CYCLES = N_MODCH

# Host DRAM traffic (spec 002 A2): the first burst of a transfer opens the
# row; each further 256-bit burst issues after nCCD.
HOST_WRITE_BURST_CYCLES = N_RCD_WR + N_CWL + N_BL
HOST_READ_BURST_CYCLES = N_RCD_RD + N_CL

# Host reduction scenario (paper "Token Partitioning"): 32 GB/s host link with
# zero fixed latency.  At 1 ns per cycle that is 32 bytes per cycle.
HOST_LINK_BYTES_PER_CYCLE = 32
HOST_LINK_FIXED_CYCLES = 0
# One 256-bit host SIMD add per merged GPR vector (analytical, uncalibrated).
HOST_VECTOR_ADD_CYCLES = 1


def _ceil_div(value, divisor):
    if divisor <= 0:
        raise ValueError("cost divisor must be positive")
    return math.ceil(value / divisor)


def _metric(event, name, default=1):
    return max(0, int(event.metrics.get(name, default)))


def _shaped_candidate(event):
    """The placement record of a contraction event (knobs plus the stamped
    ``aim_shape``), or None for elementwise events."""
    candidate = event.metrics.get("candidate", {}) or {}
    if candidate.get("aim_shape") is None:
        return None
    return candidate


def _contraction_record(candidate):
    """The AimContraction codegen lowers for this placement (one replica)."""
    shape = candidate["aim_shape"]
    return aim_lowering.build_contraction(
        shape,
        {
            "batch_mapping": candidate.get("batch_mapping"),
            "reuse_group": candidate.get("reuse_group"),
            "replica_partitions": candidate.get("replica_partitions"),
            "reduction_partitions": candidate.get("reduction_partitions"),
            "replicas": 1,
        },
        activation=shape.get("activation", False),
    )


def _sbk_banks(candidate, geometry):
    """Banks of a single-bank (SBK) layout, or None for an all-bank one."""
    if int(candidate.get("bank_fanout", geometry.banks)) != 1:
        return None
    banks = candidate.get("sbk_banks")
    return tuple(banks) if banks else tuple(range(geometry.banks))


def _contraction_profile_cycles(candidate, geometry):
    """Serial channel cycles of one replica's lowered contraction."""
    try:
        operation = _contraction_record(candidate)
        profile = aim_lowering.aim_contraction_profile(
            operation, geometry, _sbk_banks(candidate, geometry)
        )
    except ValueError as error:
        raise InfeasibleSchedule(
            f"AiM contraction is not realizable: {error}"
        ) from error
    counts = profile["counts"]
    mac_lines = profile["mac_lines"]
    mac_cycles = (profile["mac_columns"] - mac_lines) * N_CCD + mac_lines * CMD_MAC
    register_io = N_MODCH + CMD_WR_GB
    register_read = N_MODCH + CMD_RD_REG
    return (
        mac_cycles
        + profile["first_activations"] * FIRST_ROW_ACTIVATE_CYCLES
        + profile["row_switches"] * BATCH_MAPPING_ROW_ACTIVATE_CYCLES
        + profile["reuse_groups"] * REUSE_GROUP_SWITCH_CYCLES
        + counts.get("WR_GB", 0) * register_io
        + counts.get("WR_BIAS", 0) * register_io
        + counts.get("RD_MAC", 0) * register_read
        + counts.get("AF", 0) * (N_RCD_RD_AF + CMD_AF)
        + counts.get("RD_AF", 0) * register_read
    )


def host_reduction_cycles(record, geometry):
    """Cycles of one ``AimHostReduction`` under the declared link scenario.

    The host reads ``partitions`` partial GPR entries per merged vector, adds
    them and writes the merged entry back: (P + 1) entries cross the link per
    merged vector, plus one host add per partial beyond the first.
    """
    vectors = (
        _ceil_div(record.outputs, geometry.lanes) * record.batches * record.replicas
    )
    entry_bytes = geometry.vector_bits // 8
    link_bytes = vectors * (record.partitions + 1) * entry_bytes
    return (
        HOST_LINK_FIXED_CYCLES
        + _ceil_div(link_bytes, HOST_LINK_BYTES_PER_CYCLE)
        + vectors * (record.partitions - 1) * HOST_VECTOR_ADD_CYCLES
    )


def _slice_elements(shape, index):
    """Elements of a normalized host slice (ints, None for ``:``, (a, b))."""
    index = tuple(index)
    shape = tuple(int(extent) for extent in shape)
    if len(index) == 1 and len(shape) > 1 and isinstance(index[0], (tuple, list)):
        start, stop = index[0]
        return int(stop) - int(start)
    if len(index) != len(shape):
        raise ValueError(f"AiM host slice {index} does not fit shape {shape}")
    elements = 1
    for item, extent in zip(index, shape):
        if item is None:
            elements *= extent
        elif isinstance(item, (tuple, list)):
            elements *= int(item[1]) - int(item[0])
    return elements


def _ewmul_profile_cycles(elements, partitions, geometry):
    span = _ceil_div(max(1, elements), partitions * geometry.bank_groups)
    cycles = 0
    for row_start in range(0, span, geometry.row_elements):
        columns = _ceil_div(
            min(geometry.row_elements, span - row_start), geometry.lanes
        )
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
    host = target.unit("host")
    geometry = _target_geometry(target)

    def contraction(event, ctx, name):
        # One route: the lowered record's command stream prices the
        # WR_GB/WR_BIAS/RD_MAC/AF commands too, so the per-bucket move events
        # of a contraction are zero-cost (see channel_move).
        candidate = _shaped_candidate(event)
        if candidate is None:
            raise ValueError(
                "AiM cost prices the lowered contraction record and needs the "
                "contraction shape on the placement: build placements with the "
                "AiM enumerator or spmw_autoschedule._aim_layout_candidates"
            )
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
        operation = _contraction_record(candidate)
        if operation.reduction_partitions > 1:
            # The explicit host ADD of the partials (AimHostReduction).
            record = aim_lowering.AimHostReduction(
                partitions=operation.reduction_partitions,
                outputs=operation.outputs,
                batches=operation.batches,
                replicas=operation.replicas,
            )
            host_latency = host_reduction_cycles(record, geometry)
            ctx.step(
                latency=host_latency,
                occupy=[ctx.use(host, cycles=host_latency)],
                name="host_reduction",
            )

    @rule(target.op("MAC"))
    def single_bank_mac(event, ctx):
        contraction(event, ctx, "mac_sbk")

    @rule(target.op("MAC_ABK"))
    def all_bank_mac(event, ctx):
        contraction(event, ctx, "mac_abk")

    @rule(target.op("MUL"))
    def elementwise_mul(event, ctx):
        candidate = event.metrics.get("candidate", {}) or {}
        partitions = candidate.get("replica_partitions")
        if partitions in (None, "none"):
            raise ValueError(
                "AiM EWMUL cost needs the placement's replica_partitions (the "
                "channel group codegen lowers to); use the AiM enumerator"
            )
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

    @rule(target.op("ADD"))
    def elementwise_add(event, ctx):
        # Lowering emits one ``AiM EWADD op_size`` over the device-level GPR
        # for the whole kernel (spmw_aim._spread_elementwise_grid): op_size
        # covers elements_per_item, times work_items for a partitioned grid.
        # Work item 0 prices that command; the other items' events are
        # zero-cycle. One cycle per 256-bit GPR column is analytical:
        # ramulator2 does not time ISR_EWADD (aim_DRAM_system.cpp, "Do
        # nothing"), and EWADD opens no bank row.
        candidate = event.metrics.get("candidate", {}) or {}
        grid = candidate.get("aim_elementwise")
        if grid is None:
            raise ValueError(
                "AiM EWADD cost needs the placement's aim_elementwise record "
                "(the work grid codegen lowers); use the AiM enumerator"
            )
        if any(int(value) != 0 for value in event.work_id):
            ctx.step(
                cycles=0, occupy=[ctx.use(event.primitive)], name="ewadd_in_kernel"
            )
            return
        per_item = grid.get("elements_per_item")
        elements = _metric(event, "iterations") if per_item is None else int(per_item)
        if not grid["replicated"]:
            elements *= int(grid["work_items"])
        op_size = max(1, _ceil_div(elements, geometry.lanes))
        ctx.step(
            cycles=op_size,
            occupy=[ctx.use(event.primitive), ctx.use(target.gpr)],
            name="ewadd",
        )

    @rule(target.op("AF"))
    def activation(event, ctx):
        if _shaped_candidate(event) is not None:
            ctx.step(cycles=0, occupy=[ctx.use(event.primitive)], name="af_in_profile")
            return
        vectors = max(1, _ceil_div(_metric(event, "iterations"), geometry.lanes))
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
        if _shaped_candidate(event) is not None and name in (
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

    def host_transfer(event, ctx, first_burst, name):
        if event.metrics.get("host_resident"):
            # Resident buffers are on the device before the program starts.
            ctx.step(latency=0, occupy=[ctx.use(event.primitive, cycles=0)], name=name)
            return
        bits = int(event.metrics.get("dtype_bits", 0) or 0)
        shape = event.metrics.get("shape")
        index = event.metrics.get("host_slice")
        if index is not None and shape is not None and bits > 0:
            nbytes = _ceil_div(_slice_elements(shape, index) * bits, 8)
        else:
            nbytes = max(0, int(event.metrics.get("bytes", 0)))
        bursts = max(1, _ceil_div(nbytes, geometry.vector_bits // 8))
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
