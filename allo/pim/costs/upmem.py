# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cycle program for a UPMEM DPU rank.

This is intentionally ordinary Python and is the calibration surface for the
backend.  An agent can change the constants or formulas after microprofiling a
real DIMM without changing the structural target.

The initial instruction model follows Gomez-Luna et al. (arXiv:2105.03814):
the fine-grained DPU pipeline prevents one tasklet from issuing more often than
once every 11 cycles and reaches peak throughput with 11 runnable tasklets;
32-bit integer addition is native, while a general 32-bit multiplication can
lower to as many as 32 ``mul_step`` instructions.  Memory constants mirror
uPIMulator commit 870d916: 8-byte minimum DMA accesses, 1 KiB wordlines, and
the default tRCD/tCL/tBL/tRAS/tRP values in ``src/main.go``.

All results are cycle counts.  There is deliberately no clock-frequency or
wall-time conversion in this abstraction.
"""

from __future__ import annotations

import math

from ...perf import cost, rule


PIPELINE_STAGES = 14
REVOLVER_CYCLES = 11
MAX_TASKLETS = 24

ADD_INSTRUCTIONS = 1
SUB_INSTRUCTIONS = 1
MUL_STEP_INSTRUCTIONS = 32
# ``sdk/syslib/div32.c`` contains 32 DIV_STEP instructions plus operand-sign,
# call, and return setup.  Forty-eight is the conservative signed normal path.
SIGNED_DIV_INSTRUCTIONS = 48
# No integer sqrt primitive/runtime exists in the checked-in SDK.  Four
# division-equivalent Newton steps are the initial analytical assumption.
INTEGER_SQRT_INSTRUCTIONS = 192
INTEGER_COMPARE_INSTRUCTIONS = 1
# A C ternary/min/max generally needs one condition-setting instruction and
# one conditional move/branch.  Calibration can replace this compiler guess.
SELECT_INSTRUCTIONS = 2
BRANCH_INSTRUCTIONS = 1

# UPMEM has no floating-point datapath.  The SDK links compiler-rt software
# routines (sdk/syslib/addsf3.c, mulsf3.c, divsf3.c, comparesf2.c).  These are
# initial normal-input dynamic-instruction assumptions, deliberately exposed
# here for agent calibration.  The simulator's compiled assembly has roughly
# 132 instructions in addsf3, 104 in mulsf3, and 143 in divsf3 across all static
# paths; the estimates below choose representative dynamic paths and include
# called helpers such as 64-bit multiply.  SQRT is not supplied by the checked-
# in uPIMulator SDK and is modeled as software Newton iteration until a
# generated-kernel microprofile replaces the estimate.
FLOAT_ADD_INSTRUCTIONS = 96
FLOAT_SUB_INSTRUCTIONS = 98
FLOAT_MUL_INSTRUCTIONS = 128
FLOAT_DIV_INSTRUCTIONS = 256
FLOAT_SQRT_INSTRUCTIONS = 384
FLOAT_COMPARE_INSTRUCTIONS = 24

MIN_DMA_BYTES = 8
WORDLINE_BYTES = 1024
T_RCD = 32
T_RAS = 78
T_RP = 32
T_CL = 32
T_BL = 8

# uPIMulator's rank-channel defaults.  They price explicit host moves only;
# device computation remains independent across the 64 DPU instances.
HOST_READ_BYTES_PER_CYCLE = 1
HOST_WRITE_BYTES_PER_CYCLE = 3


def _ceil_div(value, divisor):
    if divisor <= 0:
        raise ValueError("cost divisor must be positive")
    return math.ceil(value / divisor)


def _metric(event, name, default=1):
    return max(0, int(event.metrics.get(name, default)))


def _tasklets(event):
    candidate = event.metrics.get("candidate", {}) or {}
    # Producers must prove that the materialized phase maps a parallel loop to
    # the carried LinearLayout image.  Keep legacy synthetic events compatible,
    # but let an explicit unmapped phase fail closed to one issuing tasklet.
    if not candidate.get("tasklet_mapping", True):
        return 1
    return max(
        1,
        min(
            MAX_TASKLETS,
            int(candidate.get("tasklet_fanout", 1)),
        ),
    )


def _is_floating(event):
    """Read the numeric kind attached by the MLIR C summarizer.

    ``numeric_kind='float'`` is the canonical metric.  The aliases make the
    cost program convenient to call from focused calibration scripts and older
    trace producers without coupling the target to an MLIR type object.
    """
    if "is_float" in event.metrics:
        return bool(event.metrics["is_float"])
    value = event.metrics.get(
        "numeric_kind",
        event.metrics.get("dtype", event.metrics.get("element_type", "int")),
    )
    text = str(value).lower()
    return text.startswith(("f", "bf")) or "float" in text


def _instructions(event, integer_per_iteration, floating_per_iteration=None):
    """Return aggregate dynamic instructions for one summarized event.

    A C/assembly summarizer may provide ``instruction_count`` (canonical) or
    ``instructions``.  Such a count already includes all loop iterations and
    takes precedence over analytical per-operation defaults.  Otherwise the
    event's loop-derived ``iterations`` metric scales the integer or software-
    floating implementation cost.
    """
    for name in (
        "instruction_count",
        "summarized_instruction_count",
        "dynamic_instructions",
        "instructions",
    ):
        if name in event.metrics:
            return max(0, int(event.metrics[name]))
    per_iteration = integer_per_iteration
    if _is_floating(event) and floating_per_iteration is not None:
        per_iteration = floating_per_iteration
    return _metric(event, "iterations") * int(per_iteration)


def _pipeline_cycles(instructions, tasklets):
    """Cycles for an aggregate DPU instruction stream.

    Work is divided over tasklets, but all tasklets issue through one shared
    pipeline.  Below 11 tasklets, the revolver issue gap limits throughput;
    at and above 11, the DPU can sustain one instruction per cycle.  Pipeline
    fill/drain is paid once per summarized stream.
    """
    instructions = max(1, int(instructions))
    active = max(1, min(int(tasklets), REVOLVER_CYCLES))
    issue_cycles = _ceil_div(instructions * REVOLVER_CYCLES, active)
    return issue_cycles + PIPELINE_STAGES - 1


@cost(target="upmem")
def upmem_cost(target):
    """Interpret UPMEM C-level primitives over target-declared resources."""
    # Tasklet throughput is driven by the structurally mapped fanout supplied
    # in candidate.tasklet_fanout, never by the raw candidate count alone.
    rank = target.unit("rank")
    dpu = target.unit("dpu")
    tasklet = target.unit("tasklet")
    host = target.unit("host")

    def compute(event, ctx, instructions, name):
        latency = _pipeline_cycles(instructions, _tasklets(event))
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(dpu, cycles=latency),
                ctx.use(tasklet, cycles=latency),
            ],
            name=name,
        )

    @rule(target.op("ADD"))
    def add(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, ADD_INSTRUCTIONS, FLOAT_ADD_INSTRUCTIONS),
            "add",
        )

    @rule(target.op("SUB"))
    def subtract(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, SUB_INSTRUCTIONS, FLOAT_SUB_INSTRUCTIONS),
            "sub",
        )

    @rule(target.op("MUL"))
    def multiply(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, MUL_STEP_INSTRUCTIONS, FLOAT_MUL_INSTRUCTIONS),
            "mul_steps",
        )

    @rule(target.op("DIV"))
    def divide(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, SIGNED_DIV_INSTRUCTIONS, FLOAT_DIV_INSTRUCTIONS),
            "divide",
        )

    @rule(target.op("SQRT"))
    def square_root(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, INTEGER_SQRT_INSTRUCTIONS, FLOAT_SQRT_INSTRUCTIONS),
            "sqrt",
        )

    @rule(target.op("CMP"))
    def compare(event, ctx):
        compute(
            event,
            ctx,
            _instructions(
                event, INTEGER_COMPARE_INSTRUCTIONS, FLOAT_COMPARE_INSTRUCTIONS
            ),
            "compare",
        )

    @rule(target.op("SELECT"))
    def select(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, SELECT_INSTRUCTIONS, SELECT_INSTRUCTIONS),
            "select",
        )

    def minmax(event, ctx, name):
        compare_instructions = (
            FLOAT_COMPARE_INSTRUCTIONS
            if _is_floating(event)
            else INTEGER_COMPARE_INSTRUCTIONS
        )
        compute(
            event,
            ctx,
            _instructions(
                event,
                INTEGER_COMPARE_INSTRUCTIONS + SELECT_INSTRUCTIONS,
                compare_instructions + SELECT_INSTRUCTIONS,
            ),
            name,
        )

    @rule(target.op("MIN"))
    def minimum(event, ctx):
        minmax(event, ctx, "min")

    @rule(target.op("MAX"))
    def maximum(event, ctx):
        minmax(event, ctx, "max")

    @rule(target.op("BRANCH"))
    def branch(event, ctx):
        compute(
            event,
            ctx,
            _instructions(event, BRANCH_INSTRUCTIONS, BRANCH_INSTRUCTIONS),
            "branch",
        )

    @rule(target.op("MAC"))
    def mac(event, ctx):
        # UPMEM has no fused integer MAC.  One scalar source-level MAC is a
        # worst-case 32-step multiply followed by one native addition.
        scalar_instructions = MUL_STEP_INSTRUCTIONS + ADD_INSTRUCTIONS
        float_scalar_instructions = FLOAT_MUL_INSTRUCTIONS + FLOAT_ADD_INSTRUCTIONS
        compute(
            event,
            ctx,
            _instructions(event, scalar_instructions, float_scalar_instructions),
            "mul_add",
        )

    def dma_cycles(event, *, write):
        nbytes = max(MIN_DMA_BYTES, _metric(event, "bytes", MIN_DMA_BYTES))
        chunks = _ceil_div(nbytes, MIN_DMA_BYTES)
        wordlines = _ceil_div(nbytes, WORDLINE_BYTES)
        # First command opens a row; subsequent minimum-size bursts stream.
        # A write must also satisfy tRAS before precharge.
        row = max(T_RAS, T_RCD + T_CL + T_BL) if write else T_RCD + T_CL + T_BL
        return wordlines * (row + T_RP) + max(0, chunks - wordlines) * T_BL

    def device_dma(event, ctx, *, write, name):
        latency = dma_cycles(event, write=write)
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(dpu, cycles=latency),
                ctx.use(target.mram, cycles=latency),
                ctx.use(target.wram, cycles=latency),
            ],
            name=name,
        )

    @rule(target.move("LD_MRAM"))
    def load_mram(event, ctx):
        device_dma(event, ctx, write=False, name="mram_to_wram")

    @rule(target.move("ST_MRAM"))
    def store_mram(event, ctx):
        device_dma(event, ctx, write=True, name="wram_to_mram")

    # WRAM operands become register loads/stores in compiled C.  Keep these
    # rules explicit so a future matcher placement can price them without
    # smuggling timing into the target.
    for move_name in ("LD_WRAM", "ST_WRAM"):

        @rule(target.move(move_name))
        def wram_register_move(event, ctx, _name=move_name):
            compute(
                event,
                ctx,
                _instructions(event, 1, 1),
                _name.lower(),
            )

    def host_transfer(event, ctx, *, to_dpu):
        nbytes = _metric(event, "bytes", 0)
        bandwidth = HOST_WRITE_BYTES_PER_CYCLE if to_dpu else HOST_READ_BYTES_PER_CYCLE
        # A rank fans a scatter/broadcast/gather across its DPU endpoints.
        # The shared rank channel is the limiting resource; individual DPU
        # work begins only after this explicit transfer activity completes.
        latency = max(1, _ceil_div(nbytes, bandwidth))
        ctx.step(
            latency=latency,
            occupy=[
                ctx.use(event.primitive, cycles=latency),
                ctx.use(host, cycles=latency),
                ctx.use(rank, cycles=latency),
            ],
            name=event.primitive.name.lower(),
        )

    @rule(target.move("SCATTER_MRAM"))
    def scatter(event, ctx):
        host_transfer(event, ctx, to_dpu=True)

    @rule(target.move("BCAST_MRAM"))
    def broadcast(event, ctx):
        # The SDK rank-broadcast primitive replicates one host transfer across
        # the selected DPUs; it is not 64 independent host payloads.
        host_transfer(event, ctx, to_dpu=True)

    @rule(target.move("GATHER_MRAM"))
    def gather(event, ctx):
        host_transfer(event, ctx, to_dpu=False)
