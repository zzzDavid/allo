# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PIM device target trees -- the importable device library.

Each ``build_<backend>_target()`` returns a fresh ``@allo.target`` device tree
containing structure, functional semantics, and ``emit=`` callbacks only.
Executable cycle programs are separate modules under :mod:`allo.pim.costs`.
Codegen passes a backend-specific context whose ``cmd(...)`` translates Tenon
handles into each simulator's operand encoding.

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
    Cycle behavior lives exclusively in ``allo.pim.costs.samsung``.
    """

    @allo.target("samsung_hbm_pim")
    def samsung_hbm_pim():
        # ============================ device ============================ #
        @allo.device
        def hbm_pim():
            # 16 pseudo-channels (sim runs up to 64 logical channels).
            @allo.unit(mapping={"channel": 16})
            def pseudo_channel():
                # 16 banks/channel (4 bank-groups x 4 banks), FP16,
                # 16384 rows x 128 cols (ini NUM_BANKS/NUM_ROWS/NUM_COLS).
                banks = allo.mem(
                    banks=16,
                    bank_groups=4,
                    rows=16384,
                    cols=128,
                    width=8,
                    name="banks",
                )
                # 32-word microcode store shared by the channel's 8 PIM
                # blocks; hard cap = 32 instructions (CRF, PIMRank.h:90-98).
                crf = allo.mem(entries=32, width=32, name="crf")

                # 8 PIM execution units, one per even/odd bank pair.
                @allo.unit(mapping={"pim": 8})
                def pim():
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
                        "LD_A",
                        src=even_bank,
                        dst=grf_a,
                        emit=lambda ctx: ctx.cmd("FILL", dst=grf_a, src0=even_bank),
                    )
                    allo.move(
                        "LD_B",
                        src=odd_bank,
                        dst=grf_b,
                        emit=lambda ctx: ctx.cmd("FILL", dst=grf_b, src0=odd_bank),
                    )
                    # GRF -> bank writeback is a NOP+WRITE column-command drain,
                    # NOT a MOV: ISA-1.0 validationCheck forbids a bank dst with
                    # a GRF src (PIMCmd.cpp:73-97; PIMRank.cpp:474-487).
                    allo.move(
                        "ST_A",
                        src=grf_a,
                        dst=even_bank,
                        emit=lambda ctx: ctx.drain(dst=even_bank, src0=grf_a),
                    )
                    allo.move(
                        "ST_B",
                        src=grf_b,
                        dst=odd_bank,
                        emit=lambda ctx: ctx.drain(dst=odd_bank, src0=grf_b),
                    )
                    # JUMP: real backward-branch flow opcode for the inner-K
                    # fold (PIMCmd.cpp:39-42; PIMRank.cpp:344-359).
                    allo.move(
                        "JUMP",
                        src=grf_b,
                        dst=grf_b,
                        emit=lambda ctx: ctx.cmd("JUMP"),
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
                        emit=lambda x, y, dst, ctx: ctx.cmd(
                            "MUL", dst=dst, src0=x, src1=y
                        ),
                    )
                    allo.op(
                        "ADD",
                        src=(operand, operand),
                        dst=any_reg,
                        fn=lambda x, y: x + y,
                        emit=lambda x, y, dst, ctx: ctx.cmd(
                            "ADD", dst=dst, src0=x, src1=y
                        ),
                    )
                    # MAC reads its dst as the accumulator (dst += src0*src1);
                    # GEMV uses dst=GRF_B (PIMBlock.cpp:64-85).
                    allo.op(
                        "MAC",
                        src=(operand, operand),
                        dst=grf_b,
                        accumulates=True,
                        fn=lambda x, y, acc: acc + x * y,
                        emit=lambda x, y, acc, ctx: ctx.cmd(
                            "MAC", dst=acc, src0=x, src1=y
                        ),
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
                            "MAD", dst=dst, src0=x, src1=y, src2=z
                        ),
                    )
                    # ReLU rides the isRelu flag on MOV (PIMRank.cpp:427-428),
                    # not a separate opcode.
                    allo.op(
                        "RELU",
                        src=(operand,),
                        dst=any_reg,
                        fn=lambda x: max(x, 0),
                        emit=lambda x, dst, ctx: ctx.cmd(
                            "MOV", dst=dst, src0=x, is_relu=1
                        ),
                    )

        # ============================= host ============================= #
        # The host's data-transfer primitives. Each names device memory
        # (hbm_pim.*) as one endpoint and host DRAM as the other; they are
        # the faithful preload/readback the pim_driver performs over
        # PIM_REG_RA (row 0x3fff) and ordinary bank RD/WR column commands.
        @allo.unit(mode="host")
        def host():
            host_dram = allo.mem(name="host_dram", bytes=1 << 34)

            # HAB broadcast: ONE host write replicated to GRF_A across all 8
            # PIM blocks (PIM_REG_RA cols 0x08-0x0f, PIMRank.cpp:183-187).
            allo.move(
                "BCAST_GRF_A",
                src=host_dram,
                dst=hbm_pim.grf_a,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.grf_a),
            )
            allo.move(
                "BCAST_GRF_B",
                src=host_dram,
                dst=hbm_pim.grf_b,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.grf_b),
            )
            # Scalar broadcast into SRF (col 0x01, PIMRank.cpp:196-201). A
            # scalar-into-vector replicate is still a broadcast; SRF is
            # disambiguated from GRF by device-handle identity, not by verb.
            allo.move(
                "BCAST_SRF",
                src=host_dram,
                dst=hbm_pim.srf,
                verb=allo.broadcast,
                emit=lambda ctx: ctx.host_broadcast(hbm_pim.srf),
            )
            # CRF program upload: <=32 microcode words (cols 0x04-0x07,
            # PIMRank.cpp:190-195). Microcode upload, no collective verb.
            allo.move(
                "PROGRAM_CRF",
                src=host_dram,
                dst=hbm_pim.crf,
                verb=allo.move_only,
                emit=lambda ctx: ctx.program_crf(hbm_pim.crf),
            )
            # Per-bank operand preload (scatter): host writes operand rows
            # into bank DRAM (BANK_TO_PIM, PIMRank.cpp:154-158).
            allo.move(
                "SCATTER_BANKS",
                src=host_dram,
                dst=hbm_pim.banks,
                verb=allo.scatter,
                emit=lambda ctx: ctx.host_scatter(hbm_pim.banks),
            )
            # Result readback (gather): host reads result bursts from bank
            # DRAM (PIMRank.cpp:247-253). Samsung output sum finishes on the
            # host CPU after gather (no cross-bank device reduce).
            allo.move(
                "GATHER_BANKS",
                src=hbm_pim.banks,
                dst=host_dram,
                verb=allo.gather,
                emit=lambda ctx: ctx.host_gather(hbm_pim.banks),
            )

    return samsung_hbm_pim


def build_aim_target():
    """Return the structural SK hynix GDDR6-AiM target.

    The topology and storage geometry follow ``aim_simulator``'s
    ``GDDR6_AiM_org`` preset: 32 channels, four bank groups per channel,
    four banks per group, 16K rows, and 1K columns.  One 256-bit column is
    the 16-lane BF16 vector consumed by a processing unit.  The Hot Chips
    34 system design exposes a 512 KiB host-side GPR SRAM, a 2 KiB Global
    Buffer per channel, and bank-resident activation LUT rows.

    This target deliberately contains no timing.  Simulator-derived cycle
    behavior lives in :mod:`allo.pim.costs.aim`.
    """

    @allo.target("aim")
    def aim():
        @allo.device
        def gddr6_aim():
            # System Architecture and Software Stack for GDDR6-AiM, slide 10:
            # the Operations Engine owns a 512 KiB GPR SRAM.  A 256-bit entry
            # is the granularity exposed by the simulator's GPR commands.
            gpr = allo.mem(
                size_bytes=512 * 1024,
                entries=(512 * 1024) // 32,
                width=256,
                name="gpr",
            )

            @allo.unit(mapping={"channel": 32})
            def channel():
                gb = allo.mem(entries=64, width=256, name="gb")
                mac_reg = allo.reg(16, 16, name="mac_reg")
                af_reg = allo.reg(16, 16, name="af_reg")
                # The JSSC design stores activation lookup tables in a
                # reserved word line in every DRAM bank.
                af_lut = allo.mem(
                    banks=16,
                    rows=1,
                    cols=1024,
                    width=16,
                    name="af_lut",
                )
                banks = allo.mem(
                    banks=16,
                    bank_groups=4,
                    rows=16384,
                    cols=1024,
                    width=16,
                    name="banks",
                )

                # Channel-wide data paths and instructions.  These are not
                # declared inside a bank: a single ISR channel mask launches
                # them over the channel's shared GB or all 16 banks.
                allo.move(
                    "WR_ABK",
                    src=gpr,
                    dst=banks,
                    emit=lambda ctx: ctx.cmd("WR_ABK", dst=banks, src0=gpr),
                )
                allo.move(
                    "WR_GB",
                    src=gpr,
                    dst=gb,
                    emit=lambda ctx: ctx.cmd("WR_GB", dst=gb, src0=gpr),
                )
                allo.move(
                    "WR_BIAS",
                    src=gpr,
                    dst=mac_reg,
                    emit=lambda ctx: ctx.cmd("WR_BIAS", dst=mac_reg, src0=gpr),
                )
                allo.move(
                    "RD_MAC",
                    src=mac_reg,
                    dst=gpr,
                    emit=lambda ctx: ctx.cmd("RD_MAC", dst=gpr, src0=mac_reg),
                )
                allo.move(
                    "RD_AF",
                    src=af_reg,
                    dst=gpr,
                    emit=lambda ctx: ctx.cmd("RD_AF", dst=gpr, src0=af_reg),
                )
                allo.move(
                    "COPY_BKGB",
                    src=banks,
                    dst=gb,
                    emit=lambda ctx: ctx.cmd("COPY_BKGB", dst=gb, src0=banks),
                )
                allo.move(
                    "COPY_GBBK",
                    src=gb,
                    dst=banks,
                    emit=lambda ctx: ctx.cmd("COPY_GBBK", dst=banks, src0=gb),
                )

                any_bank = allo.any_(banks)
                allo.op(
                    "MAC_ABK",
                    src=(banks, gb),
                    dst=mac_reg,
                    accumulates=True,
                    matchable=False,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.cmd(
                        "MAC_ABK", dst=acc, src0=x, src1=y
                    ),
                )
                allo.op(
                    "ADD",
                    src=(gpr, gpr),
                    dst=gpr,
                    fn=lambda x, y: x + y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "EWADD", dst=dst, src0=x, src1=y
                    ),
                )
                allo.op(
                    "AF",
                    src=(gpr,),
                    dst=af_reg,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd("AF", dst=dst, src0=x),
                )

                @allo.unit(mapping={"bank_group": 4})
                def bank_group():
                    # EWMUL operates on the selected four-bank group.
                    allo.op(
                        "MUL",
                        src=(any_bank, gb),
                        dst=gpr,
                        fn=lambda x, y: x * y,
                        emit=lambda x, y, dst, ctx: ctx.cmd(
                            "EWMUL", dst=dst, src0=x, src1=y
                        ),
                    )

                    @allo.unit(mapping={"bank": 4})
                    def bank():
                        _, bg_id, bank_id = allo.get_uid()
                        bank_ref = banks[4 * bg_id + bank_id]

                        allo.move(
                            "WR_SBK",
                            src=gpr,
                            dst=bank_ref,
                            emit=lambda ctx: ctx.cmd(
                                "WR_SBK", dst=bank_ref, src0=gpr
                            ),
                        )
                        allo.move(
                            "RD_SBK",
                            src=bank_ref,
                            dst=gpr,
                            emit=lambda ctx: ctx.cmd(
                                "RD_SBK", dst=gpr, src0=bank_ref
                            ),
                        )
                        allo.move(
                            "ST_SBK",
                            src=gpr,
                            dst=bank_ref,
                            emit=lambda ctx: ctx.cmd(
                                "WR_SBK", dst=bank_ref, src0=gpr
                            ),
                        )
                        allo.op(
                            "MAC",
                            src=(bank_ref, gb),
                            dst=mac_reg,
                            accumulates=True,
                            fn=lambda x, y, acc: acc + x * y,
                            emit=lambda x, y, acc, ctx: ctx.cmd(
                                "MAC_SBK", dst=acc, src0=x, src1=y
                            ),
                        )

    return aim


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

                        # -------- moves -------- #
                        allo.move(
                            "LD_MRAM",
                            src=mram,
                            dst=wram,
                            emit=lambda ctx: ctx.emit_c_line(
                                "mram_read(&{src}, &{dst}, BL);".format(
                                    src=ctx.handle_c_name(mram),
                                    dst=ctx.handle_c_name(wram),
                                )
                            ),
                        )
                        allo.move(
                            "ST_MRAM",
                            src=wram,
                            dst=mram,
                            emit=lambda ctx: ctx.emit_c_line(
                                "mram_write(&{src}, &{dst}, BL);".format(
                                    src=ctx.handle_c_name(wram),
                                    dst=ctx.handle_c_name(mram),
                                )
                            ),
                        )
                        # WRAM <-> GPR is folded by the DPU C compiler; these
                        # declarations retain the legal movement structure.
                        allo.move(
                            "LD_WRAM",
                            src=wram,
                            dst=gprs,
                            emit=lambda ctx: ctx.emit_c_line(
                                "/* WRAM->GPR fused by C compiler */"
                            ),
                        )
                        allo.move(
                            "ST_WRAM",
                            src=gprs,
                            dst=wram,
                            emit=lambda ctx: ctx.emit_c_line(
                                "/* GPR->WRAM fused by C compiler */"
                            ),
                        )

                        # -------- compute ops -------- #
                        any_wram = allo.any_(wram)
                        any_gpr = allo.any_([gprs])

                        allo.op(
                            "MUL",
                            src=(
                                allo.or_(any_wram, any_gpr),
                                allo.or_(any_wram, any_gpr),
                            ),
                            dst=any_gpr,
                            fn=lambda x, y: x * y,
                            emit=lambda x, y, dst, ctx: ctx.emit_c_line(
                                "{d} = {a} * {b};".format(
                                    d=ctx.handle_c_name(dst),
                                    a=ctx.handle_c_name(x),
                                    b=ctx.handle_c_name(y),
                                )
                            ),
                        )

                        allo.op(
                            "ADD",
                            src=(
                                allo.or_(any_wram, any_gpr),
                                allo.or_(any_wram, any_gpr),
                            ),
                            dst=any_gpr,
                            fn=lambda x, y: x + y,
                            emit=lambda x, y, dst, ctx: ctx.emit_c_line(
                                "{d} = {a} + {b};".format(
                                    d=ctx.handle_c_name(dst),
                                    a=ctx.handle_c_name(x),
                                    b=ctx.handle_c_name(y),
                                )
                            ),
                        )

                        # UPMEM has no fused MAC; lowers to mul+add in C.
                        # SPEC-019: emit_mac_kreduce wraps the body in an
                        # explicit `for (k...)` loop so uPIMulator prices
                        # the full K-reduction, not a single statement.
                        # `pending_k_bound` is set on the ctx by
                        # `_walk_and_emit` immediately before this fires.
                        allo.op(
                            "MAC",
                            src=(
                                allo.or_(any_wram, any_gpr),
                                allo.or_(any_wram, any_gpr),
                            ),
                            dst=any_gpr,
                            accumulates=True,
                            fn=lambda x, y, acc: acc + x * y,
                            emit=lambda x, y, acc, ctx: ctx.emit_mac_kreduce(
                                acc=acc, x=x, y=y, k_bound=ctx.pending_k_bound
                            ),
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
                "SCATTER_MRAM",
                src=host_dram,
                dst=dram_pim.mram,
                emit=lambda ctx: (
                    ctx.dpu_prepare_xfer(dram_pim.mram),
                    ctx.dpu_push_xfer("DPU_XFER_TO_DPU"),
                ),
            )
            # Replicate buf to every DPU (dpu_broadcast_to).
            allo.move(
                "BCAST_MRAM",
                src=host_dram,
                dst=dram_pim.mram,
                emit=lambda ctx: ctx.dpu_broadcast_to(dram_pim.mram),
            )
            # Readback per-DPU results to the host (dpu_copy_from). UPMEM has
            # no cross-DPU device reduce -> the output sum finishes host-side.
            allo.move(
                "GATHER_MRAM",
                src=dram_pim.mram,
                dst=host_dram,
                emit=lambda ctx: ctx.dpu_copy_from(dram_pim.mram),
            )

    return upmem


def build_apu_v1_target():
    """Return the structural target for one GSI Gemini APU v1 device.

    The official GVML abstraction exposes four APUCs.  Each APUC has fifteen
    writable 32K x 16-bit VRs plus the read-only index VR, forty-eight 32K x
    16-bit VMRs in L1, one VIOR in L2, eight writable marker vectors, an ARC
    controller, an asynchronous section-execution unit (SEU), and a DMA engine.
    L4 is shared by the four APUCs.

    A group is not a fixed hardware child unit: it is a power-of-two partition
    view of a 32K-lane VR selected by each GVML call.  Workload LinearLayouts
    map logical ``(group, lane_in_group)`` coordinates to the target-declared
    ``lane`` axis.  This keeps group size in the programming abstraction while
    the target remains an accurate description of available storage and
    execution resources.

    No performance numbers live here.  Real-device calibrated formulas are in
    :mod:`allo.pim.costs.apu_v1`.
    """

    @allo.target("apu_v1")
    def device():
        # L4 is one shared address space, but the four APUCs have independent
        # DMA paths into disjoint slices. Real multicore profiling confirms
        # that one 32K stream per APUC overlaps rather than serializing.
        l4 = allo.mem(size_bytes=14 * 2**30, ports=4, name="l4")

        @allo.unit(mapping={"apuc": 4})
        def apuc():
            lane_axis = {"lane": 32768}
            # Per-APUC memory hierarchy from MICRO'25 Fig. 3 / Sec. 2.1.
            # L3 is the ARC control-processor cache used by indexed lookup;
            # L2 is the 32K x 16-bit DMA scratchpad.  The 48 VMRs below are
            # the 3 MiB L1 vector register file.  These are capacities only;
            # all transfer timing belongs to the executable cost spec.
            l3 = allo.mem(size_bytes=1 << 20, ports=2, name="l3")
            l2 = allo.mem(size_bytes=64 << 10, ports=2, name="l2")
            # VR16_0..VR16_14 are writable. VR16_IDX is library-owned.
            vrs = tuple(
                allo.reg(32768, 16, slots=1, axes=lane_axis, name=f"vr{i}")
                for i in range(15)
            )
            index_vr = allo.reg(32768, 16, slots=1, axes=lane_axis, name="index_vr")
            vmrs = allo.reg(32768, 16, slots=48, axes=lane_axis, name="vmrs")
            vior = allo.reg(32768, 16, slots=1, axes=lane_axis, name="vior")
            markers = allo.reg(32768, 1, slots=8, axes=lane_axis, name="markers")

            @allo.unit(capacity=1)
            def dma():
                any_vr = allo.any_(vrs)
                # Target-neutral movement vocabulary used by the vector-plan
                # realizer.  The older role-specific combined moves remain
                # below for the grouped-FP16 compatibility path.
                allo.move("DMA_L4_TO_L3", src=l4, dst=l3)
                allo.move("DMA_L4_TO_L2", src=l4, dst=l2)
                allo.move("DMA_L2_TO_L1_32K", src=l2, dst=vmrs)
                allo.move("DMA_L4_TO_L1_32K", src=l4, dst=vmrs)
                allo.move("DMA_L1_TO_L4_32K", src=vmrs, dst=l4)
                allo.move("LOAD_L1_TO_VR16", src=vmrs, dst=any_vr)
                allo.move("STORE_VR16_TO_L1", src=any_vr, dst=vmrs)
                allo.move("PIO_L4_TO_VR16", src=l4, dst=any_vr)
                allo.move("PIO_VR16_TO_L4", src=any_vr, dst=l4)

                # Role-specific combined helpers match the actual public
                # direct_dma_l4_to_l1_32k + gvml_load/store sequence.
                for role, vr, vm_index in (
                    ("X", vrs[0], 0),
                    ("Y", vrs[1], 1),
                    ("ACC", vrs[2], 2),
                ):
                    allo.move(
                        f"LD_{role}_L4_TO_VR",
                        src=l4,
                        dst=vr,
                        emit=lambda ctx, r=role.lower(), v=vr, m=vm_index: (
                            ctx.emit_l4_to_vr(r, v, m)
                        ),
                    )
                allo.move(
                    "ST_ACC_VR_TO_L4",
                    src=vrs[2],
                    dst=l4,
                    emit=lambda ctx: ctx.emit_vr_to_l4("acc", vrs[2], 2),
                )

            @allo.unit(capacity=1)
            def seu():
                any_vr = allo.any_(vrs)
                allo.op(
                    "ADD",
                    src=(any_vr, any_vr),
                    dst=any_vr,
                    fn=lambda x, y: x + y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "gvml_add_f16", dst=dst, src0=x, src1=y
                    ),
                )
                allo.op(
                    "SUB",
                    src=(any_vr, any_vr),
                    dst=any_vr,
                    fn=lambda x, y: x - y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "gvml_sub_f16", dst=dst, src0=x, src1=y
                    ),
                )
                allo.op(
                    "MUL",
                    src=(any_vr, any_vr),
                    dst=any_vr,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "gvml_mul_f16", dst=dst, src0=x, src1=y
                    ),
                )
                allo.op(
                    "GROUP_REDUCE_ADD_F16",
                    src=(any_vr,),
                    dst=any_vr,
                    matchable=False,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.emit_group_reduce_f16(dst, x),
                )
                # A source-level reduction maps to one full-VR multiply and
                # one group reduction, not a binary lookup-table surrogate.
                allo.op(
                    "MAC",
                    src=(any_vr, any_vr),
                    dst=any_vr,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.emit_grouped_f16_mac(
                        acc=acc, x=x, y=y
                    ),
                )

                # Integer/logical and layout-realization primitives used by
                # the MICRO'25 binary-matmul plans.  They are deliberately
                # non-matchable: the region vectorizer recognizes a complete
                # loop/access pattern and emits these typed plan operations.
                for name, fn in (
                    ("RESET_16", lambda x: 0),
                    ("CPY_IMM_16", lambda x: x),
                    ("NOT_16", lambda x: ~x),
                    ("POPCOUNT_16", lambda x: x),
                    ("SHL_IMM_16", lambda x: x),
                    ("CREATE_GROUP_INDEX_16", lambda x: x),
                    ("CREATE_SUBGROUP_INDEX_16", lambda x: x),
                    ("LOOKUP_16", lambda x: x),
                ):
                    allo.op(
                        name,
                        src=(any_vr,),
                        dst=any_vr,
                        fn=fn,
                        matchable=False,
                    )
                for name, fn in (
                    ("XOR_16", lambda x, y: x ^ y),
                    ("AND_16", lambda x, y: x & y),
                    ("OR_16", lambda x, y: x | y),
                    ("ADD_U16", lambda x, y: x + y),
                    ("ADD_S16", lambda x, y: x + y),
                    ("SUB_U16", lambda x, y: x - y),
                    ("SUB_S16", lambda x, y: x - y),
                    ("MUL_U16", lambda x, y: x * y),
                    ("DUPLICATE_SUBGROUP_16", lambda x, y: x),
                ):
                    allo.op(
                        name,
                        src=(any_vr, any_vr),
                        dst=any_vr,
                        fn=fn,
                        matchable=False,
                    )
                allo.op(
                    "GROUP_REDUCE_ADD_U16",
                    src=(any_vr,),
                    dst=any_vr,
                    fn=lambda x: x,
                    matchable=False,
                )
                allo.op(
                    "GROUP_REDUCE_ADD_S16",
                    src=(any_vr,),
                    dst=any_vr,
                    fn=lambda x: x,
                    matchable=False,
                )

            @allo.unit(capacity=1)
            def arc():
                # The ARC issues GVML/DMA calls and can execute scalar control
                # while an asynchronous SEU fragment is in flight.
                allo.op(
                    "SCALAR_C",
                    src=(vior,),
                    dst=vior,
                    fn=lambda x: x,
                    matchable=False,
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
        l5 = allo.mem(size_bytes=2**40, name="l5")

        @allo.unit(mapping=[1])  # 1 chip
        def chip():
            l1 = allo.mem(rows=3072, cols=65536, width=1, name="l1")

            @allo.unit(mapping=[16])  # 16 L1 row groups
            def row_group():
                # -------- moves -------- #
                allo.move(
                    "COPY_TO_L1",
                    src=l5,
                    dst=l1,
                    emit=lambda ctx: ctx.cmd("copy_to_l1", dst=l1, src0=l5),
                )
                allo.move(
                    "COPY_FROM_L1",
                    src=l1,
                    dst=l5,
                    emit=lambda ctx: ctx.cmd("copy_from_l1", dst=l5, src0=l1),
                )

                # -------- compute ops -------- #
                any_l1 = allo.any_(l1)

                allo.op(
                    "ADD",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    fn=lambda x, y: x + y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "g.add", dst=dst, src0=x, src1=y
                    ),
                )

                allo.op(
                    "MUL",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "g.mul", dst=dst, src0=x, src1=y
                    ),
                )

                # MAC: GTML has no fused MAC primitive at vector
                # granularity. The ctx expands it into matmul + add.
                allo.op(
                    "MAC",
                    src=(any_l1, any_l1),
                    dst=any_l1,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.emit_mac_matmul(acc=acc, x=x, y=y),
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
                        "g.matmul", dst=dst, src0=x, src1=y
                    ),
                )

                allo.op(
                    "RMS_NORM",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd(
                        "g.rms_norm_gflt", dst=dst, src0=x
                    ),
                )

                allo.op(
                    "SOFTMAX",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd("g.softmax", dst=dst, src0=x),
                )

                allo.op(
                    "AF",
                    src=(any_l1,),
                    dst=any_l1,
                    fn=lambda x: x,
                    emit=lambda x, dst, ctx: ctx.cmd("g.relu", dst=dst, src0=x),
                )

    return device
