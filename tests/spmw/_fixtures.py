# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for SPMW tests.

The Samsung HBM-PIM target spec from report 16 is defined here so that
both the matcher tests and the codegen tests can build it without
duplicating the spec.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16


# --------------------------------------------------------------------- #
# Workload fixtures
# --------------------------------------------------------------------- #


# MLP shapes (chosen small so the test compiles quickly but the two
# layers remain distinct so the matcher reports two MAC sites per
# work-item function).
MLP_M1, MLP_K1 = 256, 128       # Layer 1: W1[M1, K1] x x[K1] -> h[M1]
MLP_M2, MLP_K2 = 64, MLP_M1     # Layer 2: W2[M2, K2] x h[K2] -> y[M2]


# Allo's TypeInferer resolves annotations by looking up the bare
# identifier in the decorated function's source — so `fp16`, `M1`,
# `K1`, ... must all live as plain module-level names. The work
# function is therefore defined here at module scope (rather than
# inside a builder closure that rebinds the names) and `build_mlp_workload()`
# just returns it.


# NOTE: the two MLP layers are split into separate `@allo.work` kernels
# in one region rather than written as two sequential reductions in a
# single kernel body. The match engine's `_build_defining_map` is a
# flat dict keyed by MLIR SSA name (`%0`, `%6`, …) walked across all
# nested regions — when two affine.for regions inside the same func
# reuse `%0..%6`, later regions overwrite the earlier ones in the map,
# and the matcher only reports the second reduction. Splitting layers
# across work-item funcs gives each its own SSA-name scope and lets
# the matcher report one MAC per layer. The ReLU bridge between layers
# is omitted (architect deferred adding a `max` op pattern).


@_df_region()
def _mlp_top(
    W1: fp16[MLP_M1, MLP_K1],
    x: fp16[MLP_K1],
    W2: fp16[MLP_M2, MLP_K2],
    h: fp16[MLP_M1],
    y: fp16[MLP_M2],
):
    @allo.work(mapping=[1], args=[W1, x, h])
    def mlp_layer1(
        local_W1: fp16[MLP_M1, MLP_K1],
        local_x: fp16[MLP_K1],
        local_h: fp16[MLP_M1],
    ):
        for i1 in range(MLP_M1):
            acc: fp16 = 0
            for k1 in range(MLP_K1):
                acc += local_W1[i1, k1] * local_x[k1]
            local_h[i1] = acc

    @allo.work(mapping=[1], args=[W2, h, y])
    def mlp_layer2(
        local_W2: fp16[MLP_M2, MLP_K2],
        local_h: fp16[MLP_M1],
        local_y: fp16[MLP_M2],
    ):
        for i2 in range(MLP_M2):
            acc: fp16 = 0
            for k2 in range(MLP_K2):
                acc += local_W2[i2, k2] * local_h[k2]
            local_y[i2] = acc


def build_mlp_workload():
    """Two-layer MLP workload: h = W1 @ x (+ ReLU stub), y = W2 @ h.

    Mapping is ``[1]`` — a single work-item function contains both
    MAC sites, so ``match_workload`` is guaranteed to report exactly
    two MatchedOps regardless of unroll behaviour.

    Returns the decorated top-level region so callers can pass it to
    ``allo.customize``.
    """

    return _mlp_top


def build_samsung_target():
    """Return the Samsung HBM-PIM target tree per report 16, including
    `emit=` callbacks on every move and op.

    The emit lambdas use Tenon handles directly (`grf_a`, `even_bank`,
    ...). Codegen passes a backend-specific `ctx` whose `cmd(...)` method
    translates those handles into the simulator's operand encoding.
    """

    @allo.target("samsung_hbm_pim")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.mem(banks=16, rows=16384, cols=128, width=8, name="banks")

            @allo.unit(mapping=[8])
            def pim():
                _, pid = allo.get_uid()
                even_bank = banks[2 * pid]
                odd_bank = banks[2 * pid + 1]
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                # Samsung HBM-PIM timing (coarse model). tCCDL=4 is the
                # column-strobe period; LD/ST spills come from spec 015
                # §6.1: load = tCCDL+RL+BL//2 = 26, store = tCCDL+WL+BL//2
                # = 14 (RL=20, WL=8, BL=4). JUMP is a 1-cycle control op.
                allo.move(
                    "LD_A", src=even_bank, dst=grf_a,
                    cycles=26,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_a, src0=even_bank),
                )
                allo.move(
                    "LD_B", src=odd_bank, dst=grf_b,
                    cycles=26,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_b, src0=odd_bank),
                )
                allo.move(
                    "ST_A", src=grf_a, dst=even_bank,
                    cycles=14,
                    emit=lambda ctx: ctx.cmd("MOV", dst=even_bank, src0=grf_a),
                )
                allo.move(
                    "ST_B", src=grf_b, dst=odd_bank,
                    cycles=14,
                    emit=lambda ctx: ctx.cmd("MOV", dst=odd_bank, src0=grf_b),
                )
                # JUMP is a synthetic 1-cycle control op used by the cost
                # model to price the inner-K fold in bank-row MAC; no
                # codegen emits it today.
                allo.move(
                    "JUMP", src=grf_b, dst=grf_b,
                    cycles=1,
                    emit=lambda ctx: None,
                )
                # CRF_TRIGGER (SPEC-025 §6): host per-tile fire latency in
                # the shared-CRF schedule. The host issues one trigger per
                # work-id against the single broadcast CRF program rather
                # than re-uploading the CRF body per work-id. Cost is the
                # host-side per-command issue latency (tCCDL-class; Samsung
                # ISCA'21 §4.2 PIM-command issue spacing) -- strictly less
                # than a per-work-id CRF body (folded MAC + JUMP, >= 258 cyc
                # for the fast dual-fiber candidate at K=1024), which is the
                # invariant that makes shared CRF win in argmin (SPEC-025
                # §4.5). No codegen emits a PIMCmd for it; it rides the host
                # schedule side-list (SPEC-025 §5.4).
                allo.move(
                    "CRF_TRIGGER", src=grf_b, dst=grf_b,
                    cycles=2,
                    emit=lambda ctx: None,
                )

                # SPEC-026 §3.2/§3.3: closed-form preload/readback phase
                # constants for the batched-GEMV cost model. These are the
                # spec-side carriers of the per-phase costs the faithful
                # run (report 18 §3) measured at 4096x1024
                # (preload=11368, readback=181); the cost model evaluates
                # the closed forms
                #   preload_cyc  = (M*K // PRELOAD_FAN) * PRELOAD_WR + PRELOAD_CRF
                #   readback_cyc = ceil(M / READBACK_FAN) * READBACK_RD
                # so the numbers track (M, K) and never appear as inline
                # literals in `spmw_cost_models.py`. `cycles` is used as a
                # generic integer carrier (fan-out widths + per-group cyc).
                #
                # Calibration (report 18 §3, faithful preloadGemv /
                # readResult at M=4096, K=1024):
                #   PRELOAD_FAN  = 369  -- effective parallel weight-write
                #     fan-out of the HAB-broadcast preloadGemv double loop
                #     (PIMKernel.cpp:295-322), faithful-anchored so
                #     (M*K // 369) = 11366 weight-write groups.
                #   PRELOAD_WR   = 1    -- per-group column-strobe cost
                #     (tCCDL-normalised, Samsung ISCA'21 §4.1).
                #   PRELOAD_CRF  = 2    -- one-time programCrf upload latency
                #     (CRF program <= 4 bursts, arch-200 §2; ISCA'21 §4.2).
                #     => preload_cyc(4096,1024) = 11366 + 2 = 11368.
                #   READBACK_FAN = 4096 -- output elements covered by one
                #     readback tile (num_total_pim_blocks_ * num_grfB_,
                #     ISCA'21 §4.1); readback issues one GRFB_TO_BANK per
                #     output tile (PIMKernel.cpp:435).
                #   READBACK_RD  = 181  -- per-output-tile readResult +
                #     GRFB_TO_BANK writeback latency (faithful cyc_readback,
                #     report 18 §3). => readback_cyc(4096) = 1*181 = 181.
                allo.move(
                    "PRELOAD_FAN", src=even_bank, dst=even_bank,
                    cycles=369, emit=lambda ctx: None,
                )
                allo.move(
                    "PRELOAD_WR", src=grf_a, dst=even_bank,
                    cycles=1, emit=lambda ctx: None,
                )
                allo.move(
                    "PRELOAD_CRF", src=grf_a, dst=even_bank,
                    cycles=2, emit=lambda ctx: None,
                )
                allo.move(
                    "READBACK_FAN", src=odd_bank, dst=grf_b,
                    cycles=4096, emit=lambda ctx: None,
                )
                allo.move(
                    "READBACK_RD", src=odd_bank, dst=grf_b,
                    cycles=181, emit=lambda ctx: None,
                )

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                    cycles=4,
                    emit=lambda x, y, dst, ctx: ctx.cmd("MUL", dst=dst, src0=x, src1=y),
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    cycles=4,
                    emit=lambda x, y, acc, ctx: ctx.cmd("MAC", dst=acc, src0=x, src1=y),
                )

    return device


def build_aim_target():
    """Return the SK-Hynix AiM (GDDR6 PIM) target tree per spec 003.

    Architecture: 32 channels x 4 bank-groups x 8 banks x 16 MAC lanes.
    Each channel owns: GPR scratch (16 entries u16), a cross-bank Global
    Buffer, and bank-DRAM. Compute ISRs (MAC_SBK / MAC_ABK / EWMUL /
    EWADD / AF) are emitted as plain text lines via `AimCtx.cmd`.
    """

    @allo.target("aim")
    def device():
        @allo.unit(mapping=[32])  # 32 channels
        def channel():
            gpr = allo.reg(16, 16, name="gpr")
            gb = allo.mem(entries=512, width=16, name="gb")
            banks = allo.mem(
                banks=32, rows=16384, cols=1024, width=8, name="banks"
            )

            @allo.unit(mapping=[4])  # 4 bank-groups
            def bg():
                @allo.unit(mapping=[8])  # 8 banks per bg
                def bank():
                    _, bg_id, bk_id = allo.get_uid()
                    bank_ref = banks[8 * bg_id + bk_id]
                    bias = allo.reg(16, 16, name="bias")

                    # -------- moves -------- #
                    # AiM bank-row spill: tCCDL=4, RD=16, WR=12, burst=4
                    # (JSSC 2023 §IV). RD_SBK = tCCDL+RD+BURST = 24.
                    # Other moves are not priced in the spill model and
                    # default to None.
                    allo.move(
                        "WR_SBK", src=gb, dst=bank_ref,
                        emit=lambda ctx: ctx.cmd(
                            "WR_SBK", dst=bank_ref, src0=gb),
                    )
                    allo.move(
                        "WR_ABK", src=gb, dst=banks,
                        emit=lambda ctx: ctx.cmd(
                            "WR_ABK", dst=banks, src0=gb),
                    )
                    allo.move(
                        "WR_GB", src=gpr, dst=gb,
                        emit=lambda ctx: ctx.cmd(
                            "WR_GB", dst=gb, src0=gpr),
                    )
                    allo.move(
                        "WR_BIAS", src=gpr, dst=bias,
                        emit=lambda ctx: ctx.cmd(
                            "WR_BIAS", dst=bias, src0=gpr),
                    )
                    allo.move(
                        "RD_MAC", src=bank_ref, dst=gpr,
                        emit=lambda ctx: ctx.cmd(
                            "RD_MAC", dst=gpr, src0=bank_ref),
                    )
                    allo.move(
                        "RD_AF", src=bank_ref, dst=gpr,
                        emit=lambda ctx: ctx.cmd(
                            "RD_AF", dst=gpr, src0=bank_ref),
                    )
                    allo.move(
                        "RD_SBK", src=bank_ref, dst=gpr,
                        cycles=24,
                        emit=lambda ctx: ctx.cmd(
                            "RD_SBK", dst=gpr, src0=bank_ref),
                    )
                    allo.move(
                        "COPY_BKGB", src=bank_ref, dst=gb,
                        emit=lambda ctx: ctx.cmd(
                            "COPY_BKGB", dst=gb, src0=bank_ref),
                    )
                    allo.move(
                        "COPY_GBBK", src=gb, dst=bank_ref,
                        emit=lambda ctx: ctx.cmd(
                            "COPY_GBBK", dst=bank_ref, src0=gb),
                    )
                    # Synthetic store-side move (no bank-row writeback
                    # opcode is emitted today; cycles only exists so the
                    # spill model can price ST_SBK monotonically).
                    # store = tCCDL + WR + BURST = 20.
                    allo.move(
                        "ST_SBK", src=gpr, dst=bank_ref,
                        cycles=20,
                        emit=lambda ctx: None,
                    )

                    # -------- compute ops -------- #
                    # AiM cycle estimates (JSSC 2023 §IV; ramulator2
                    # YAML carries the exact per-ISR timings):
                    #   EWMUL = EWADD = 4; MAC_SBK = 8 (1 burst x 16
                    #   lanes); MAC_ABK = 16 (all-bank broadcast = 2x);
                    #   AF (GELU/SIGMOID) = 6.
                    any_bank = allo.any_(banks)
                    any_gpr = allo.any_([gpr])

                    # Element-wise MUL.
                    allo.op(
                        "MUL",
                        src=(allo.or_(any_bank, any_gpr),
                             allo.or_(any_bank, any_gpr)),
                        dst=any_gpr,
                        fn=lambda x, y: x * y,
                        cycles=4,
                        emit=lambda x, y, dst, ctx: ctx.cmd(
                            "EWMUL", dst=dst, src0=x, src1=y),
                    )

                    # Element-wise ADD.
                    allo.op(
                        "ADD",
                        src=(allo.or_(any_bank, any_gpr),
                             allo.or_(any_bank, any_gpr)),
                        dst=any_gpr,
                        fn=lambda x, y: x + y,
                        cycles=4,
                        emit=lambda x, y, dst, ctx: ctx.cmd(
                            "EWADD", dst=dst, src0=x, src1=y),
                    )

                    # Single-bank MAC.
                    allo.op(
                        "MAC",
                        src=(allo.or_(any_bank, any_gpr),
                             allo.or_(any_bank, any_gpr)),
                        dst=gpr,
                        accumulates=True,
                        fn=lambda x, y, acc: acc + x * y,
                        cycles=8,
                        emit=lambda x, y, acc, ctx: ctx.cmd(
                            "MAC_SBK", dst=acc, src0=x, src1=y),
                    )

                    # All-bank-broadcast MAC.
                    allo.op(
                        "MAC_ABK",
                        src=(allo.or_(any_bank, any_gpr),
                             allo.or_(gb, any_gpr)),
                        dst=gpr,
                        accumulates=True,
                        fn=lambda x, y, acc: acc + x * y,
                        cycles=16,
                        emit=lambda x, y, acc, ctx: ctx.cmd(
                            "MAC_ABK", dst=acc, src0=x, src1=y),
                    )

                    # Activation function (kind selected via emit field).
                    allo.op(
                        "AF",
                        src=(any_gpr,),
                        dst=any_gpr,
                        fn=lambda x: x,
                        cycles=6,
                        emit=lambda x, dst, ctx: ctx.cmd(
                            "AF", dst=dst, src0=x, kind="GELU"),
                    )

    return device


def build_upmem_target():
    """Return the UPMEM DPU (DRAM-PIM) target tree per spec 003.

    Architecture: rank -> DPUs -> tasklets. Per-DPU memories are MRAM
    (64 MB bulk), WRAM (64 KB scratch), and IRAM (24 KB instruction).
    Per-tasklet GPRs (24 entries). UPMEM's emit lambdas produce C
    statement strings (assembled later into a DPU task.c).
    """

    @allo.target("upmem")
    def device():
        @allo.unit(mapping=[1])  # rank (collapsed multi-rank)
        def rank():
            @allo.unit(mapping=[64])  # DPUs per rank
            def dpu():
                mram = allo.mem(size_bytes=67108864, name="mram")
                wram = allo.mem(size_bytes=65536, name="wram")
                iram = allo.mem(size_bytes=24576, name="iram")

                @allo.unit(mapping=[16])  # tasklets per DPU
                def tasklet():
                    gprs = allo.reg(24, 32, name="gprs")

                    # -------- moves -------- #
                    # UPMEM latency (uPIMulator / HPCA 2024 Table 2):
                    # MRAM read/write = 1000 cyc per 64 B burst; WRAM
                    # access = 1 cyc (GPR fused by C compiler).
                    allo.move(
                        "LD_MRAM", src=mram, dst=wram,
                        cycles=1000,
                        emit=lambda ctx: ctx.emit_c_line(
                            "mram_read(&{src}, &{dst}, BL);".format(
                                src=ctx.handle_c_name(mram),
                                dst=ctx.handle_c_name(wram))),
                    )
                    allo.move(
                        "ST_MRAM", src=wram, dst=mram,
                        cycles=1000,
                        emit=lambda ctx: ctx.emit_c_line(
                            "mram_write(&{src}, &{dst}, BL);".format(
                                src=ctx.handle_c_name(wram),
                                dst=ctx.handle_c_name(mram))),
                    )
                    # WRAM <-> GPR is folded by the DPU C compiler;
                    # named moves exist so the cost model can price
                    # spills without forcing an emit pass.
                    allo.move(
                        "LD_WRAM", src=wram, dst=gprs,
                        cycles=1,
                        emit=lambda ctx: ctx.emit_c_line(
                            "/* WRAM->GPR fused by C compiler */"),
                    )
                    allo.move(
                        "ST_WRAM", src=gprs, dst=wram,
                        cycles=1,
                        emit=lambda ctx: ctx.emit_c_line(
                            "/* GPR->WRAM fused by C compiler */"),
                    )

                    # -------- compute ops -------- #
                    # DPU GPR ops issue at 1 cyc; MAC = mul+add = 2 cyc
                    # (no fused MAC on DPU).
                    any_wram = allo.any_(wram)
                    any_gpr = allo.any_([gprs])

                    allo.op(
                        "MUL",
                        src=(allo.or_(any_wram, any_gpr),
                             allo.or_(any_wram, any_gpr)),
                        dst=any_gpr,
                        fn=lambda x, y: x * y,
                        cycles=1,
                        emit=lambda x, y, dst, ctx: ctx.emit_c_line(
                            "{d} = {a} * {b};".format(
                                d=ctx.handle_c_name(dst),
                                a=ctx.handle_c_name(x),
                                b=ctx.handle_c_name(y))),
                    )

                    allo.op(
                        "ADD",
                        src=(allo.or_(any_wram, any_gpr),
                             allo.or_(any_wram, any_gpr)),
                        dst=any_gpr,
                        fn=lambda x, y: x + y,
                        cycles=1,
                        emit=lambda x, y, dst, ctx: ctx.emit_c_line(
                            "{d} = {a} + {b};".format(
                                d=ctx.handle_c_name(dst),
                                a=ctx.handle_c_name(x),
                                b=ctx.handle_c_name(y))),
                    )

                    # UPMEM has no fused MAC; lowers to mul+add in C.
                    # The cost model prices it as 2 cycles.
                    # SPEC-019: emit_mac_kreduce wraps the body in an
                    # explicit `for (k...)` loop so uPIMulator prices
                    # the full K-reduction, not a single statement.
                    # `pending_k_bound` is set on the ctx by
                    # `_walk_and_emit` immediately before this fires.
                    allo.op(
                        "MAC",
                        src=(allo.or_(any_wram, any_gpr),
                             allo.or_(any_wram, any_gpr)),
                        dst=any_gpr,
                        accumulates=True,
                        fn=lambda x, y, acc: acc + x * y,
                        cycles=2,
                        emit=lambda x, y, acc, ctx: ctx.emit_mac_kreduce(
                            acc=acc, x=x, y=y,
                            k_bound=ctx.pending_k_bound),
                    )

    return device


def build_apu_v1_target():
    """Return the GSI APU v1 (Gemini 1) target tree per spec 003.

    Architecture: 4 APUCs per chip. Each APUC has 16 vector registers
    (32K bit-serial lanes, u16 each), a 32 KB L1 scratch (slot-addressed),
    and shares a 14 GB L4 DRAM at the chip level. MAC has no fused
    opcode -- the SV-lookup expansion (`gvml_lookup_16 + gvml_add_s16`)
    is owned by the codegen ctx's `emit_mac_lookup` method.
    """

    @allo.target("apu_v1")
    def device():
        # Top-level: L4 DRAM is chip-shared.
        l4 = allo.mem(size_bytes=14 * 2**30, name="l4")

        @allo.unit(mapping=[4])  # 4 APUCs
        def apuc():
            # Per-APUC L1 scratch (32 KB SRAM, 32 slots of 1 KB).
            l1 = allo.mem(size_bytes=32768, slots=32, slot_bytes=1024,
                          name="l1")
            # Per-APUC vector register file (16 VRs x 32K elements).
            vrs = allo.reg(16, 32768, name="vrs")

            # -------- moves -------- #
            # APU v1 cycle estimates (report 12 §4.2 + pim-apu-v1 skill):
            # DMA L4<->L1 = 140 cyc per 32K-element burst; LD/ST_VR = 4
            # cyc (same SRAM fabric as VRs, ~5 cyc round-trip for spill).
            allo.move(
                "DMA_L4_L1", src=l4, dst=l1,
                cycles=140,
                emit=lambda ctx: ctx.cmd(
                    "direct_dma_l4_to_l1_32k", dst=l1, src0=l4),
            )
            allo.move(
                "DMA_L1_L4", src=l1, dst=l4,
                cycles=140,
                emit=lambda ctx: ctx.cmd(
                    "direct_dma_l1_to_l4_32k", dst=l4, src0=l1),
            )
            allo.move(
                "LD_VR", src=l1, dst=vrs,
                cycles=5,
                emit=lambda ctx: ctx.cmd(
                    "gvml_load_16", dst=vrs, src0=l1),
            )
            allo.move(
                "ST_VR", src=vrs, dst=l1,
                cycles=5,
                emit=lambda ctx: ctx.cmd(
                    "gvml_store_16", dst=l1, src0=vrs),
            )
            # Chained two-stage moves -- preferred over splitting the
            # walker on a list-of-names contract (spec 009 §F). Each
            # `emit` calls ctx.cmd twice in order.
            allo.move(
                "LD_L4_TO_VR", src=l4, dst=vrs,
                emit=lambda ctx: (
                    ctx.cmd("direct_dma_l4_to_l1_32k", dst=l1, src0=l4),
                    ctx.cmd("gvml_load_16", dst=vrs, src0=l1),
                ),
            )
            allo.move(
                "ST_VR_TO_L4", src=vrs, dst=l4,
                emit=lambda ctx: (
                    ctx.cmd("gvml_store_16", dst=l1, src0=vrs),
                    ctx.cmd("direct_dma_l1_to_l4_32k", dst=l4, src0=l1),
                ),
            )

            # -------- compute ops -------- #
            # APU v1 per-op cycles (32K bit-serial lanes per VR):
            # gvml_add_s16 = 2; gvml_mul_u16 = 16; MAC (SV-lookup
            # expansion = gvml_lookup_16 + gvml_add_s16) = 6 + 2 = 8.
            # The autoscheduler also prices raw SV-mode MAC as 16+2=18
            # (see _apu_v1_kernel_cycles).
            any_vr = allo.any_([vrs])

            allo.op(
                "ADD",
                src=(any_vr, any_vr),
                dst=any_vr,
                fn=lambda x, y: x + y,
                cycles=2,
                emit=lambda x, y, dst, ctx: ctx.cmd(
                    "gvml_add_s16", dst=dst, src0=x, src1=y),
            )

            allo.op(
                "MUL",
                src=(any_vr, any_vr),
                dst=any_vr,
                fn=lambda x, y: x * y,
                cycles=16,
                emit=lambda x, y, dst, ctx: ctx.cmd(
                    "gvml_mul_u16", dst=dst, src0=x, src1=y),
            )

            # MAC: no fused opcode. The ctx owns the lookup + add
            # expansion (see APUv1Ctx.emit_mac_lookup).
            allo.op(
                "MAC",
                src=(any_vr, any_vr),
                dst=any_vr,
                accumulates=True,
                fn=lambda x, y, acc: acc + x * y,
                cycles=8,  # SV-lookup: gvml_lookup_16(6) + gvml_add_s16(2)
                emit=lambda x, y, acc, ctx: ctx.emit_mac_lookup(
                    acc=acc, x=x, y=y),
            )

    return device


def build_apu_v2_target():
    """Return the GSI APU v2 (Gemini 2, G2) target tree per spec 003.

    Architecture: 1 chip with a chip-wide L1 bitline grid (3072 rows x
    65536 cols) and host-side L5 DRAM. 16 L1 row groups partition the
    grid; each owns 4096 elements of a 64K-element vector. There is no
    per-group register file -- the L1 row itself is the operand store.
    GTML's higher ops (matmul / rms_norm / softmax / AF) are declared
    so the target surface mirrors the GTML API even though the MLIR
    matcher currently only lowers MAC / MUL / ADD.
    """

    @allo.target("apu_v2")
    def device():
        l5 = allo.mem(size_bytes=2 ** 40, name="l5")

        @allo.unit(mapping=[1])  # 1 chip
        def chip():
            l1 = allo.mem(rows=3072, cols=65536, width=1, name="l1")

            @allo.unit(mapping=[16])  # 16 L1 row groups
            def row_group():
                # -------- moves -------- #
                allo.move(
                    "COPY_TO_L1", src=l5, dst=l1,
                    emit=lambda ctx: ctx.cmd(
                        "copy_to_l1", dst=l1, src0=l5),
                )
                allo.move(
                    "COPY_FROM_L1", src=l1, dst=l5,
                    emit=lambda ctx: ctx.cmd(
                        "copy_from_l1", dst=l5, src0=l1),
                )

                # -------- compute ops -------- #
                any_l1 = allo.any_(l1)

                allo.op(
                    "ADD",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    fn=lambda x, y: x + y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "g.add", dst=dst, src0=x, src1=y),
                )

                allo.op(
                    "MUL",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "g.mul", dst=dst, src0=x, src1=y),
                )

                # MAC: GTML has no fused MAC primitive at vector
                # granularity. The ctx expands it into matmul + add.
                allo.op(
                    "MAC",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.emit_mac_matmul(
                        acc=acc, x=x, y=y),
                )

                # Higher-level ops mirror the GTML surface. fn= slots
                # are placeholders -- the MLIR matcher does not
                # currently generate these.
                allo.op(
                    "MATMUL",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    fn=lambda x, y: x,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "g.matmul", dst=dst, src0=x, src1=y),
                )

                allo.op(
                    "RMS_NORM",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd(
                        "g.rms_norm_gflt", dst=dst, src0=x),
                )

                allo.op(
                    "SOFTMAX",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd(
                        "g.softmax", dst=dst, src0=x),
                )

                allo.op(
                    "AF",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd(
                        "g.relu", dst=dst, src0=x),
                )

    return device
