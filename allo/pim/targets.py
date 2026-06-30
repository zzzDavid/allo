# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PIM device target trees -- the importable device library.

Each ``build_<backend>_target()`` returns a fresh ``@allo.target`` device
tree (structure + ``emit=`` callbacks only). The per-op / per-move cycle
numbers live in the paired :mod:`allo.spmw_cost_tables`; codegen passes a
backend-specific ``ctx`` whose ``cmd(...)`` translates the Tenon handles
into each simulator's operand encoding.

Moved here from ``tests/spmw/_fixtures.py`` (2026-06-30): these are reusable
hardware descriptions, not test scaffolding. The five suite backends are
keyed by ``target.name`` so a kernel folder names a backend, never a tree.
"""

from __future__ import annotations

import allo


def build_samsung_target():
    """Samsung HBM-PIM target tree -- faithful to PIMSimulator + ISCA'21.

    Structure: the target `samsung_hbm_pim` contains two scopes -- a device
    scope `hbm_pim` (the near-bank PIM fabric) and a `host` scope that declares
    the explicit host<->device data-transfer primitives, naming device memory
    (`hbm_pim.banks`, `hbm_pim.grf_a`, `hbm_pim.crf`, ...) as one endpoint.

    Hardware model (design_doc/compiler/samsung-pim-isa.md; PIMSimulator
    src/{PIMCmd,PIMRank,PIMBlock}.{h,cpp}, ini/HBM2_samsung_2M_16B_x64.ini):
      * 16 pseudo-channels (paper); 16 banks/channel = 4 bank-groups x 4 banks.
      * 8 PIM execution units/channel, one per even/odd bank pair (NUM_PIM_BLOCKS=8).
      * Per PIM block: GRF_A[8] + GRF_B[8] (each 256-bit = 16 FP16), SRF (1 burst,
        split SRF_M lanes 0-7 / SRF_A lanes 8-15). CRF = 32 microcode words
        shared by the channel's 8 blocks (hard 32-instruction cap).
      * 9 real opcodes: NOP/JUMP/EXIT (flow), ADD/MUL/MAC/MAD (arith), MOV/FILL
        (data-move). ReLU is the isRelu flag on MOV/FILL, not a separate opcode.

    The emit lambdas use Tenon handles directly; codegen passes a backend
    `ctx` whose `cmd(...)` translates them to the simulator's operand encoding.
    Per-op/per-move cycle numbers live in spmw_cost_tables.SAMSUNG_FAITHFUL.
    """

    @allo.target("samsung_hbm_pim")
    def samsung_hbm_pim():
        # Analytical cycle formulas are stable target semantics. Their named
        # parameters live in allo.pim.performance's calibration profile and can
        # be updated from microprofiles without editing this target.
        mac_timing = allo.cycle_model(
            "samsung.mac",
            inputs=("reduction_extent", "lanes", "n_fibers"),
            parameters=("mac_cycles", "jump_cycles"),
            latency_cycles=(
                "ceil_div(reduction_extent, lanes * n_fibers) * mac_cycles "
                "+ n_fibers * jump_cycles"
            ),
            initiation_interval_cycles="mac_cycles",
            description="Folded reduction on one Samsung PIM block",
        )
        elementwise_timing = allo.cycle_model(
            "samsung.elementwise",
            inputs=("iterations",),
            parameters=("elementwise_cycles",),
            latency_cycles="max(1, iterations) * elementwise_cycles",
            initiation_interval_cycles="elementwise_cycles",
        )
        load_timing = allo.cycle_model(
            "samsung.bank_load",
            parameters=("bank_load_cycles",),
            latency_cycles="bank_load_cycles",
        )
        store_timing = allo.cycle_model(
            "samsung.bank_store",
            parameters=("bank_store_cycles",),
            latency_cycles="bank_store_cycles",
        )
        jump_timing = allo.cycle_model(
            "samsung.jump",
            parameters=("jump_cycles",),
            latency_cycles="jump_cycles",
        )
        host_transfer_timing = allo.cycle_model(
            "samsung.host_transfer",
            inputs=("bytes", "burst_bytes"),
            parameters=("host_setup_cycles", "host_burst_cycles"),
            latency_cycles=(
                "host_setup_cycles + ceil_div(bytes, burst_bytes) * host_burst_cycles"
            ),
        )
        crf_program_timing = allo.cycle_model(
            "samsung.crf_program",
            parameters=("crf_program_cycles",),
            latency_cycles="crf_program_cycles",
        )

        # ============================ device ============================ #
        @allo.device
        def hbm_pim():
            # 16 pseudo-channels (sim runs up to 64 logical channels).
            @allo.unit(mapping=[16])
            def pseudo_channel():
                # One command path per pseudo-channel. It is pipelined: a new
                # PIM command may issue at the operation's II while a previous
                # command is still completing in a PIM block.
                command_bus = allo.resource(
                    "command_bus", pipelined=True,
                    description="shared command issue path for one pseudo-channel",
                )
                # 16 banks/channel (4 bank-groups x 4 banks), FP16,
                # 16384 rows x 128 cols (ini NUM_BANKS/NUM_ROWS/NUM_COLS).
                banks = allo.mem(
                    banks=16, bank_groups=4, rows=16384, cols=128,
                    width=8, name="banks",
                )
                # 32-word microcode store shared by the channel's 8 PIM
                # blocks; hard cap = 32 instructions (CRF, PIMRank.h:90-98).
                crf = allo.mem(entries=32, width=32, name="crf")

                # 8 PIM execution units, one per even/odd bank pair.
                @allo.unit(mapping=[8])
                def pim():
                    pim_alu = allo.resource(
                        "alu", description="one execution engine per even/odd bank pair"
                    )
                    bank_port = allo.resource(
                        "bank_pair_port", description="bank/GRF transfer port"
                    )
                    _, pid = allo.get_uid()
                    even_bank = banks[2 * pid]
                    odd_bank = banks[2 * pid + 1]
                    # GRF_A / GRF_B: 8 entries x 256-bit (16 FP16) each.
                    grf_a = allo.reg(8, 256, name="grf_a")
                    grf_b = allo.reg(8, 256, name="grf_b")
                    # SRF: 1 burst (256-bit), SRF_M lanes 0-7 / SRF_A 8-15;
                    # a read broadcasts one scalar lane to all 16 lanes.
                    srf = allo.reg(1, 256, name="srf")

                    # ----------------- data-move opcodes ----------------- #
                    # Bank-row tile -> 8 GRF entries is FILL (8x auto-repeat
                    # sweeping column&7), NOT a single MOV (PIMRank.cpp:362-377).
                    allo.move(
                        "LD_A", src=even_bank, dst=grf_a,
                        emit=lambda ctx: ctx.cmd("FILL", dst=grf_a, src0=even_bank),
                        timing_model=load_timing,
                        resources=(command_bus, bank_port),
                    )
                    allo.move(
                        "LD_B", src=odd_bank, dst=grf_b,
                        emit=lambda ctx: ctx.cmd("FILL", dst=grf_b, src0=odd_bank),
                        timing_model=load_timing,
                        resources=(command_bus, bank_port),
                    )
                    # GRF -> bank writeback is a NOP+WRITE column-command drain,
                    # NOT a MOV: ISA-1.0 validationCheck forbids a bank dst with
                    # a GRF src (PIMCmd.cpp:73-97; PIMRank.cpp:474-487).
                    allo.move(
                        "ST_A", src=grf_a, dst=even_bank,
                        emit=lambda ctx: ctx.drain(dst=even_bank, src0=grf_a),
                        timing_model=store_timing,
                        resources=(command_bus, bank_port),
                    )
                    allo.move(
                        "ST_B", src=grf_b, dst=odd_bank,
                        emit=lambda ctx: ctx.drain(dst=odd_bank, src0=grf_b),
                        timing_model=store_timing,
                        resources=(command_bus, bank_port),
                    )
                    # JUMP: real backward-branch flow opcode for the inner-K
                    # fold (PIMCmd.cpp:39-42; PIMRank.cpp:344-359).
                    allo.move(
                        "JUMP", src=grf_b, dst=grf_b,
                        emit=lambda ctx: ctx.cmd("JUMP"),
                        timing_model=jump_timing,
                        resources=(command_bus,),
                    )

                    # ------------------- compute opcodes ----------------- #
                    # Legal operand sources: GRF_A/GRF_B, SRF_M/SRF_A,
                    # EVEN/ODD_BANK (PIMCmd.h:45-55). dst forbids a bank when
                    # the src is a GRF (ISA-1.0); enforced at emit/codegen.
                    any_bank = allo.any_(banks)
                    any_reg = allo.any_([grf_a, grf_b, srf])
                    operand = allo.or_(any_bank, any_reg)
                    allo.op(
                        "MUL",
                        src=(operand, operand),
                        dst=any_reg,
                        fn=lambda x, y: x * y,
                        emit=lambda x, y, dst, ctx: ctx.cmd("MUL", dst=dst, src0=x, src1=y),
                        timing_model=elementwise_timing,
                        resources=(command_bus, pim_alu),
                        performance_inputs={"iterations": 1},
                    )
                    allo.op(
                        "ADD",
                        src=(operand, operand),
                        dst=any_reg,
                        fn=lambda x, y: x + y,
                        emit=lambda x, y, dst, ctx: ctx.cmd("ADD", dst=dst, src0=x, src1=y),
                        timing_model=elementwise_timing,
                        resources=(command_bus, pim_alu),
                        performance_inputs={"iterations": 1},
                    )
                    # MAC reads its dst as the accumulator (dst += src0*src1);
                    # GEMV uses dst=GRF_B (PIMBlock.cpp:64-85).
                    allo.op(
                        "MAC",
                        src=(operand, operand),
                        dst=grf_b,
                        accumulates=True,
                        fn=lambda x, y, acc: acc + x * y,
                        emit=lambda x, y, acc, ctx: ctx.cmd("MAC", dst=acc, src0=x, src1=y),
                        timing_model=mac_timing,
                        resources=(command_bus, pim_alu),
                        performance_inputs={"lanes": 8},
                    )
                    # MAD: explicit 3-source multiply-add (dst = s0*s1 + s2);
                    # only opcode with a real src2 (PIMBlock.cpp:87-107). No
                    # PolyBench kernel emits it yet -- declared for ISA parity.
                    allo.op(
                        "MAD",
                        src=(operand, operand, operand),
                        dst=any_reg,
                        fn=lambda x, y, z: x * y + z,
                        emit=lambda x, y, z, dst, ctx: ctx.cmd(
                            "MAD", dst=dst, src0=x, src1=y, src2=z),
                        timing_model=elementwise_timing,
                        resources=(command_bus, pim_alu),
                        performance_inputs={"iterations": 1},
                    )
                    # ReLU rides the isRelu flag on MOV (PIMRank.cpp:427-428),
                    # not a separate opcode.
                    allo.op(
                        "RELU",
                        src=(operand,),
                        dst=any_reg,
                        fn=lambda x: max(x, 0),
                        emit=lambda x, dst, ctx: ctx.cmd("MOV", dst=dst, src0=x, is_relu=1),
                        timing_model=elementwise_timing,
                        resources=(command_bus, pim_alu),
                        performance_inputs={"iterations": 1},
                    )

        # ============================= host ============================= #
        # The host's data-transfer primitives. Each names device memory
        # (hbm_pim.*) as one endpoint and host DRAM as the other; they are
        # the faithful preload/readback the pim_driver performs over
        # PIM_REG_RA (row 0x3fff) and ordinary bank RD/WR column commands.
        @allo.unit(mode="host")
        def host():
            host_link = allo.resource(
                "host_link", description="serialized host/HBM transfer path"
            )
            host_dram = allo.mem(name="host_dram", bytes=1 << 34)

            # HAB broadcast: ONE host write replicated to GRF_A across all 8
            # PIM blocks (PIM_REG_RA cols 0x08-0x0f, PIMRank.cpp:183-187).
            allo.move(
                "BCAST_GRF_A", src=host_dram, dst=hbm_pim.grf_a,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.grf_a),
                timing_model=host_transfer_timing,
                resources=(host_link,),
            )
            allo.move(
                "BCAST_GRF_B", src=host_dram, dst=hbm_pim.grf_b,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.grf_b),
                timing_model=host_transfer_timing,
                resources=(host_link,),
            )
            # Scalar broadcast into SRF (col 0x01, PIMRank.cpp:196-201). A
            # scalar-into-vector replicate is still a broadcast; SRF is
            # disambiguated from GRF by device-handle identity, not by verb.
            allo.move(
                "BCAST_SRF", src=host_dram, dst=hbm_pim.srf,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.srf),
                timing_model=host_transfer_timing,
                resources=(host_link,),
            )
            # CRF program upload: <=32 microcode words (cols 0x04-0x07,
            # PIMRank.cpp:190-195). Microcode upload, no collective verb.
            allo.move(
                "PROGRAM_CRF", src=host_dram, dst=hbm_pim.crf,
                verb=allo.move_only,
                emit=lambda ctx: ctx.program_crf(hbm_pim.crf),
                timing_model=crf_program_timing,
                resources=(host_link,),
            )
            # Per-bank operand preload (scatter): host writes operand rows
            # into bank DRAM (BANK_TO_PIM, PIMRank.cpp:154-158).
            allo.move(
                "SCATTER_BANKS", src=host_dram, dst=hbm_pim.banks,
                verb=allo.scatter,
                emit=lambda ctx: ctx.host_scatter(hbm_pim.banks),
                timing_model=host_transfer_timing,
                resources=(host_link,),
            )
            # Result readback (gather): host reads result bursts from bank
            # DRAM (PIMRank.cpp:247-253). Samsung output sum finishes on the
            # host CPU after gather (no cross-bank device reduce).
            allo.move(
                "GATHER_BANKS", src=hbm_pim.banks, dst=host_dram,
                verb=allo.gather,
                emit=lambda ctx: ctx.host_gather(hbm_pim.banks),
                timing_model=host_transfer_timing,
                resources=(host_link,),
            )

    return samsung_hbm_pim


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
            # AiM GPR: 31 addressable MAC-accumulator GPRs (JSSC 2023 §IV
            # datasheet) is the allocator SLOT axis; the 16-lane SIMD width
            # (`lanes=16`) is a DIFFERENT axis and must not be conflated
            # with slot count. The allocator reads `slots` (tree-derived),
            # not a pasted 31 (SPEC-022 D2 reconciliation).
            gpr = allo.reg(16, 16, name="gpr", slots=31)
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
                    # AiM bank-row spill numbers (RD_SBK=24, ST_SBK=20)
                    # migrated to spmw_cost_tables.AIM_FAITHFUL per design
                    # 04; the target keeps structure + emit only.
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
                    # opcode is emitted today; exists so the spill model
                    # has a named ST_SBK carrier to price = 20).
                    allo.move(
                        "ST_SBK", src=gpr, dst=bank_ref,
                        emit=lambda ctx: None,
                    )

                    # -------- compute ops -------- #
                    # AiM cycle estimates (JSSC 2023 §IV) migrated to
                    # spmw_cost_tables.AIM_FAITHFUL: EWMUL=EWADD=4,
                    # MAC_SBK=8, MAC_ABK=16, AF=6.
                    any_bank = allo.any_(banks)
                    any_gpr = allo.any_([gpr])

                    # Element-wise MUL.
                    allo.op(
                        "MUL",
                        src=(allo.or_(any_bank, any_gpr),
                             allo.or_(any_bank, any_gpr)),
                        dst=any_gpr,
                        fn=lambda x, y: x * y,
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
                        emit=lambda x, y, acc, ctx: ctx.cmd(
                            "MAC_ABK", dst=acc, src0=x, src1=y),
                    )

                    # Activation function (kind selected via emit field).
                    allo.op(
                        "AF",
                        src=(any_gpr,),
                        dst=any_gpr,
                        fn=lambda x: x,
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
    def upmem():
        # ============================ device ============================ #
        @allo.device
        def dram_pim():
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

                        # revolver_latency (=11) and all move/op cycle numbers
                        # migrated to spmw_cost_tables.UPMEM_FAITHFUL per design
                        # 04 (revolver_latency rides CostModel.constants). The
                        # target keeps move/op structure + emit only.

                        # -------- moves -------- #
                        allo.move(
                            "LD_MRAM", src=mram, dst=wram,
                            emit=lambda ctx: ctx.emit_c_line(
                                "mram_read(&{src}, &{dst}, BL);".format(
                                    src=ctx.handle_c_name(mram),
                                    dst=ctx.handle_c_name(wram))),
                        )
                        allo.move(
                            "ST_MRAM", src=wram, dst=mram,
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
                            emit=lambda ctx: ctx.emit_c_line(
                                "/* WRAM->GPR fused by C compiler */"),
                        )
                        allo.move(
                            "ST_WRAM", src=gprs, dst=wram,
                            emit=lambda ctx: ctx.emit_c_line(
                                "/* GPR->WRAM fused by C compiler */"),
                        )

                        # -------- compute ops -------- #
                        any_wram = allo.any_(wram)
                        any_gpr = allo.any_([gprs])

                        allo.op(
                            "MUL",
                            src=(allo.or_(any_wram, any_gpr),
                                 allo.or_(any_wram, any_gpr)),
                            dst=any_gpr,
                            fn=lambda x, y: x * y,
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
                            emit=lambda x, y, acc, ctx: ctx.emit_mac_kreduce(
                                acc=acc, x=x, y=y,
                                k_bound=ctx.pending_k_bound),
                        )

        # ============================= host ============================= #
        # Explicit host<->device transfer primitives over the UPMEM SDK; each
        # names device memory (dram_pim.mram) as one endpoint.
        @allo.unit(mode="host")
        def host():
            host_dram = allo.mem(name="host_dram", bytes=1 << 34)
            # Partition buf across DPUs: prepare per-DPU buffers + push the
            # transfer (dpu_prepare_xfer + dpu_push_xfer to DPU_MRAM_HEAP).
            allo.move(
                "SCATTER_MRAM", src=host_dram, dst=dram_pim.mram,
                emit=lambda ctx: (
                    ctx.dpu_prepare_xfer(dram_pim.mram),
                    ctx.dpu_push_xfer("DPU_XFER_TO_DPU"),
                ),
            )
            # Replicate buf to every DPU (dpu_broadcast_to).
            allo.move(
                "BCAST_MRAM", src=host_dram, dst=dram_pim.mram,
                emit=lambda ctx: ctx.dpu_broadcast_to(dram_pim.mram),
            )
            # Readback per-DPU results to the host (dpu_copy_from). UPMEM has
            # no cross-DPU device reduce -> the output sum finishes host-side.
            allo.move(
                "GATHER_MRAM", src=dram_pim.mram, dst=host_dram,
                emit=lambda ctx: ctx.dpu_copy_from(dram_pim.mram),
            )

    return upmem

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
            # APU v1 cycle estimates (report 12 §4.2) migrated to
            # spmw_cost_tables.APU_V1_FAITHFUL per design 04: DMA L4<->L1 =
            # 140 cyc/32K burst; LD/ST_VR = 5 cyc. Target keeps structure
            # + emit only.
            allo.move(
                "DMA_L4_L1", src=l4, dst=l1,
                emit=lambda ctx: ctx.cmd(
                    "direct_dma_l4_to_l1_32k", dst=l1, src0=l4),
            )
            allo.move(
                "DMA_L1_L4", src=l1, dst=l4,
                emit=lambda ctx: ctx.cmd(
                    "direct_dma_l1_to_l4_32k", dst=l4, src0=l1),
            )
            allo.move(
                "LD_VR", src=l1, dst=vrs,
                emit=lambda ctx: ctx.cmd(
                    "gvml_load_16", dst=vrs, src0=l1),
            )
            allo.move(
                "ST_VR", src=vrs, dst=l1,
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
            # APU v1 per-op cycles (gvml_add_s16=2, gvml_mul_u16=16, MAC
            # SV-lookup = gvml_lookup_16(6)+gvml_add_s16(2)=8; raw SV-mode
            # MAC priced as 16+2=18) migrated to
            # spmw_cost_tables.APU_V1_FAITHFUL per design 04.
            any_vr = allo.any_([vrs])

            allo.op(
                "ADD",
                src=(any_vr, any_vr),
                dst=any_vr,
                fn=lambda x, y: x + y,
                emit=lambda x, y, dst, ctx: ctx.cmd(
                    "gvml_add_s16", dst=dst, src0=x, src1=y),
            )

            allo.op(
                "MUL",
                src=(any_vr, any_vr),
                dst=any_vr,
                fn=lambda x, y: x * y,
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
