# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ PERFORMANCE variant — monolithic ring_core (Allo dataflow).

This is the performance-oriented port of the best-cosim HLS reference
(`feather_plus_16x16` commit 1ded6de, monolithic `ring_core`, ~2,970 cosim,
RTL ref 3,025). The central architectural lever of that reference arc was the
*monolithic ring_core*: the entire ring (weight fetch + MAC array + BIRRD
reduction + tile accumulation + writeback) collapsed into ONE flat, II=1
software-pipelined process, replacing the multi-process dataflow ring of the
baseline (`feather_plus.py`: a_loader / w_loader / w_broadcast[AW] /
pe_array[AH+1,AW] / inst_rw / BIRRD[P0,P1] / output_accum — hundreds of HLS
processes after `mapping` unfold, each pair separated by a streaming handshake).

Collapsing the ring into a single `df.kernel(mapping=[1])` eliminates the
inter-process FIFO handshake latency that the HLS arc measured at ~48 cycles per
tile across the ~370-process dataflow ring (FINAL_SUMMARY: 13,227 -> 7,855, the
single biggest step). That step is what this variant captures in pure Allo.

What is ported (HLS optimization -> Allo construct):
  * Monolithic single process              -> ONE `df.kernel(mapping=[1])`; Allo
                                              emits exactly one HLS process per
                                              single-instance kernel, so the whole
                                              ring is one process (no FIFOs).
  * Flat II=1 software-pipelined ring loop  -> `s.pipeline(..., II=1)` on the flat
                                              per-op trip loop.
  * Complete-partitioned PE-state RFs       -> `s.partition(..., Complete)` on the
                                              kernel-local `pe`, `acc`, `bcur`,
                                              `bnxt`, `insts` register files.
  * MAC array (AH*AW MACs / step, unrolled) -> `allo.meta_for(AW)` x
                                              `allo.meta_for(AH)` inside the loop.
  * BIRRD butterfly co-located in the ring  -> straight-line register butterfly
                                              over `range(P0)` (compile-time
                                              unrolled stages) x `meta_for(P1)`,
                                              `reverse_bits(...)` indices resolved
                                              at compile time. No inter-stage
                                              streams.
  * Wide A/W partitioning for parallel reads -> `s.partition(Cyclic/Complete)` on
                                              the DRAM-backed `A_pe`/`B_pe` args.

What is NOT expressible in Allo (documented as gap notes in the task dir, NOT
patched into this source):
  * m_axi `num_read_outstanding` / `latency` / per-arg `bundle` tuning (the final
    3,523 -> 2,970 HLS increment) — backend-fixed; no Schedule primitive.
  * The reference's hand-skewed prologue/drain overlap + round-robin weight-cache
    eviction + fast/fallback dual decode path + `dependence inter=false` on the
    fused weight prefetch — heavy, brittle, latency-hiding machinery that only
    pays off in concert with the m_axi tuning Allo cannot emit. This variant uses
    a clean, bit-exact monolithic loop instead and documents the skew as a gap.

MINISA programmability is UNCHANGED: the kernel decodes the same instruction
stream at runtime (SetIVNLayout / SetWVNLayout / SetOVNLayout / SetMapping via
the per-tile inst rows + precomputed BIRRD/col-map tables passed as DRAM args),
with no hard-coded dims, tile counts, mappings, Gr/Gc/sr/sc, K-passes, or base
addresses beyond the compile-time array sizes (AH = AW = 16). The `FeatherModule`
API and the 4-instruction ISA semantics are reused verbatim from the baseline.
"""

from math import log2

import allo
from allo.ir.types import int8, int32, ConstExpr
import allo.dataflow as df
from allo.customize import Partition

# Reuse the shared helpers + the FeatherModule host wrapper from the baseline so
# the perf variant stays bit-identical at the API/host level. Only the on-chip
# dataflow region is restructured.
from feather_plus import (
    PS, AR, AL, SW,            # noqa: F401  (kept for parity / readability)
    compute_birrd_params,
    FeatherModule,
)


def reverse_bits(data: int, bit_range: int) -> int:
    """Reverse the lower ``bit_range`` bits of ``data`` (butterfly routing).

    Defined locally (a verbatim copy of the baseline helper) so Allo's kernel
    tracer can resolve its source when it is called inside the traced ring loop.
    The baseline keeps the same helper in-module for the same reason.
    """
    data = int(data)
    bit_range = int(bit_range)
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return int((data & ~mask) | reversed_bits)


def _birrd_width(stage: int, LOG2_AW: int) -> int:
    """Reverse-bits routing width for BIRRD ``stage`` (compile-time, pure Python).

    Module-level so Allo's `ASTResolver.resolve` can evaluate it when it is used
    inside a `ConstExpr[int32]` butterfly-routing index (chained ConstExpr *local*
    variables do NOT resolve, but nested module-function calls do).
    """
    if stage == 0:
        return 2
    return int(min(LOG2_AW, 2 + stage, 2 * LOG2_AW - stage))


def _birrd_dst(stage: int, pos: int, P0: int, LOG2_AW: int) -> int:
    """Next-stage destination index of butterfly output ``pos`` at ``stage``.

    Mirrors the baseline BIRRD routing (feather_plus.py / feather.py): every
    stage except the LAST routes through the reverse-bits permutation; the final
    stage (stage == P0-1) writes straight to ``pos`` (the baseline's meta_else
    branch that puts to connection[P0, 2*j(+1)] without reversal). Pure Python /
    module-level so it is evaluated at type-inference time inside a ConstExpr.
    """
    pos = int(pos)
    if stage == P0 - 1:
        return pos
    return reverse_bits(pos, _birrd_width(stage, LOG2_AW))


def get_feather_perf_top(M, K, N, AW, AH, Ty, num_inst,
                         n_inner=1, k_passes=1, Nt_local=None):
    """Create the monolithic FEATHER+ performance dataflow region.

    Single-process ring: one `df.kernel(mapping=[1])` reads every DRAM buffer
    directly and writes C directly. The flat trip loop runs one *op* per
    iteration; inside it the full MAC -> BIRRD -> accumulate pipeline executes,
    pipelined II=1 with all PE-state register files completely partitioned.

    Signature is byte-for-byte the SAME 14-argument layout as the baseline
    `get_feather_full_matrix_top`, so `FeatherModule` drives it unchanged.
    """
    if Nt_local is None:
        Nt_local = AH
    TyOut = int32
    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3
    total_ops = num_tiles * n_inner
    num_blocks = num_tiles // k_passes
    num_accum_params = 2 + num_tiles  # quant_scale, quant_zp, sr[0..num_tiles-1]

    @df.region()
    def perf_top(
        A_pe: int32[M, K],
        B_pe: int32[K, N],
        inst_pe: int32[num_inst, 13],
        loader_m_start: int32[total_ops],
        inst_w: int32[num_inst, 13],
        loader_n_start: int32[total_ops],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        output_n_base: int32[num_tiles, AW],
        accum_m_start: int32[total_ops],
        accum_n_start: int32[total_ops],
        accum_params: int32[num_accum_params],
        C: int32[M, N],
    ):
        """Monolithic ring: one process holds the entire FEATHER+ ring."""

        # NOTE: the `args=[...]` first-seen order fixes the generated top-level
        # argument order. It MUST match the order `FeatherModule.__call__`
        # passes them (A, inst_pe, loader_m_start, B, inst_w, loader_n_start,
        # birrd, col_map, num_m, n_base, accum_m_start, accum_n_start,
        # accum_params, C) so the shared host wrapper drives this variant
        # unchanged — identical to the baseline's 14-arg layout.
        @df.kernel(
            mapping=[1],
            args=[A_pe, inst_pe, loader_m_start, B_pe, inst_w, loader_n_start,
                  birrd_inst, output_col_map, output_num_m, output_n_base,
                  accum_m_start, accum_n_start, accum_params, C],
        )
        def ring_core(
            local_A: int32[M, K],
            local_inst: int32[num_inst, 13],
            local_loader_m_start: int32[total_ops],
            local_B: int32[K, N],
            local_inst_w: int32[num_inst, 13],
            local_loader_n_start: int32[total_ops],
            local_birrd_inst: int8[num_tiles, P0, P1],
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_output_n_base: int32[num_tiles, AW],
            local_accum_m_start: int32[total_ops],
            local_accum_n_start: int32[total_ops],
            local_accum_params: int32[num_accum_params],
            local_C: int32[M, N],
        ):
            iacts_zp: int32 = local_inst[0, 6]
            weights_zp: int32 = local_inst_w[1, 6]
            quant_scale: int32 = local_accum_params[0]
            quant_zp: int32 = local_accum_params[1]

            # === PE-state register files (kept on-chip, fully partitioned) ===
            # pe[col, d]  : MAC result of PE (row=d, col) for the current op.
            # acc[col, d] : per-block tile accumulator (BIRRD-reduced).
            pe: int32[AW, AH]
            acc: int32[AW, AH]
            # BIRRD straight-line double buffer (one AW-wide value vector per
            # butterfly round). bcur holds stage-i values, bnxt stage-(i+1).
            bcur: int32[AW]
            bnxt: int32[AW]

            for block in range(num_blocks):
                # zero the block accumulator
                with allo.meta_for(AW) as _ci:
                    for _di in range(AH):
                        acc[_ci, _di] = 0

                base_tile: int32 = block * k_passes

                # --- accumulate over the k_passes tiles of this block ---
                for k in range(k_passes):
                    tile: int32 = base_tile + k
                    inst_idx: int32 = tile + 3
                    Gr: int32 = local_inst[inst_idx, 3]
                    Gc: int32 = local_inst_w[inst_idx, 4]
                    sr: int32 = local_inst_w[inst_idx, 5]
                    sc: int32 = local_inst_w[inst_idx, 6]
                    k_start_tile: int32 = local_inst[inst_idx, 11]

                    log2_Gr: int32 = 0
                    if Gr >= 2:
                        log2_Gr = 1
                    if Gr >= 4:
                        log2_Gr = 2
                    if Gr >= 8:
                        log2_Gr = 3
                    if Gr >= 16:
                        log2_Gr = 4
                    mask_Gr: int32 = Gr - 1
                    mask_Gc: int32 = Gc - 1

                    for inner in range(n_inner):
                        op_idx: int32 = tile * n_inner + inner
                        m_start: int32 = local_loader_m_start[op_idx]
                        n_start: int32 = local_loader_n_start[op_idx]

                        # ---- (1) MAC ARRAY: pe[col, d] = sum_nk A*W ----
                        # Mirrors baseline pe_array: PE(row=d, col) accumulates
                        # over nk the product A[m_idx, k] * W[k, wn_idx].
                        with allo.meta_for(AW) as col:
                            with allo.meta_for(AH) as d:
                                pe[col, d] = 0
                        for nk in range(AH):
                            with allo.meta_for(AW) as col:
                                m_idx: int32 = m_start + (col & mask_Gr)
                                k_idx: int32 = (
                                    k_start_tile + nk + (col >> log2_Gr) * AH
                                )
                                a_val: int32 = local_A[m_idx, k_idx] - iacts_zp
                                with allo.meta_for(AH) as d:
                                    wn_idx: int32 = (
                                        n_start + sr * d + sc * (col & mask_Gc)
                                    )
                                    w_val: int32 = (
                                        local_B[k_idx, wn_idx] - weights_zp
                                    )
                                    pe[col, d] = pe[col, d] + a_val * w_val

                        # ---- (2) BIRRD: butterfly-reduce across columns ----
                        # For each row index d, take the AW column values
                        # pe[*, d] and run the per-tile butterfly network; the
                        # final-stage column ordering matches the baseline
                        # connection[P0, col] consumption, accumulated into acc.
                        for d in range(AH):
                            # load stage-0 vector (one value per column)
                            with allo.meta_for(AW) as col0:
                                bcur[col0] = pe[col0, d]

                            # P0 butterfly stages, compile-time unrolled.
                            # `stage` is a Python loop var, so the reverse-bits
                            # width is a pure compile-time constant per stage.
                            # Both `stage` and the switch index `j` are Python
                            # loop variables, so every routing index
                            # reverse_bits(2*j, w) is a compile-time literal and
                            # `bnxt[<literal>]` is a fixed register write. The
                            # P0*P1 switches fully unroll into a combinational
                            # butterfly under the single II=1 pipeline.
                            with allo.meta_for(P0) as stage:
                                with allo.meta_for(P1) as j:
                                    # compile-time-evaluated routing indices.
                                    # Both `stage` and `j` are meta_for (constant)
                                    # indices; `reverse_bits`/`_birrd_width` are
                                    # module-level so ASTResolver evaluates the
                                    # whole index at type-inference time.
                                    dst_l: ConstExpr[int32] = _birrd_dst(
                                        stage, 2 * j, P0, LOG2_AW
                                    )
                                    dst_r: ConstExpr[int32] = _birrd_dst(
                                        stage, 2 * j + 1, P0, LOG2_AW
                                    )
                                    sw_inst: int8 = local_birrd_inst[tile, stage, j]
                                    in_left: TyOut = bcur[2 * j]
                                    in_right: TyOut = bcur[2 * j + 1]
                                    out_left: TyOut = 0
                                    out_right: TyOut = 0
                                    if sw_inst == 0:        # PS: pass
                                        out_left = in_left
                                        out_right = in_right
                                    elif sw_inst == 1:      # AR: add-right
                                        out_left = in_left
                                        out_right = in_left + in_right
                                    elif sw_inst == 2:      # AL: add-left
                                        out_left = in_left + in_right
                                        out_right = in_right
                                    else:                   # SW: swap
                                        out_left = in_right
                                        out_right = in_left
                                    bnxt[dst_l] = out_left
                                    bnxt[dst_r] = out_right
                                # copy nxt -> cur for the next stage
                                with allo.meta_for(AW) as colc:
                                    bcur[colc] = bnxt[colc]

                            # accumulate final stage into the block accumulator
                            with allo.meta_for(AW) as colf:
                                acc[colf, d] = acc[colf, d] + bcur[colf]

                # --- writeback: apply col->m mapping and write C ---
                num_m: int32 = local_output_num_m[base_tile]
                sr_val: int32 = local_accum_params[2 + base_tile]
                wb_m_start: int32 = local_accum_m_start[base_tile * n_inner]
                wb_n_start: int32 = local_accum_n_start[base_tile * n_inner]
                for col in range(AW):
                    m_pos: int32 = local_output_col_map[base_tile, col]
                    n_base_col: int32 = local_output_n_base[base_tile, col]
                    col_mask: int32 = 0
                    m_safe: int32 = 0
                    if m_pos < num_m:
                        col_mask = 1
                        m_safe = m_pos
                    for on in range(AH):
                        sr_mask: int32 = 0
                        if sr_val != 0:
                            sr_mask = 1
                        if on == 0:
                            sr_mask = 1
                        n_off: int32 = sr_val * on + n_base_col
                        val: int32 = acc[col, on] * col_mask * sr_mask
                        if quant_scale != 0:
                            val = (val * quant_scale + quant_zp) & 255
                        local_C[wb_m_start + m_safe, wb_n_start + n_off] = val

    return perf_top


def schedule_perf_hls(s, K, N, AH, AW):
    """Apply HLS scheduling for the monolithic ring (perf variant).

    The single ring_core process is pipelined and every PE-state register file
    is completely partitioned so the unrolled MAC/BIRRD lanes get parallel HW.
    The big DRAM-backed A/B operands are partitioned for parallel column reads.
    """
    # Complete-partition the PE-state register files (registers, not BRAM)
    s.partition("ring_core_0:pe", partition_type=Partition.Complete)
    s.partition("ring_core_0:acc", partition_type=Partition.Complete)
    s.partition("ring_core_0:bcur", partition_type=Partition.Complete)
    s.partition("ring_core_0:bnxt", partition_type=Partition.Complete)

    # Output C: partition along N (dim=2) for parallel column writes.
    s.partition("perf_top:C", dim=2, partition_type=Partition.Complete)

    # A operand: cyclic on M for AW-parallel row reads + complete on K for
    # parallel column reads (Gr<AW makes meta_for instances hit distinct k_idx).
    s.partition("ring_core_0:local_A", dim=1, factor=AW,
                partition_type=Partition.Cyclic)
    s.partition("ring_core_0:local_A", dim=2, partition_type=Partition.Complete)

    # B operand: small -> Complete (registers); large -> Cyclic to bound mux.
    if K * N <= 256:
        s.partition("ring_core_0:local_B", dim=1, partition_type=Partition.Complete)
        s.partition("ring_core_0:local_B", dim=2, partition_type=Partition.Complete)
    else:
        s.partition("ring_core_0:local_B", dim=2, factor=AH,
                    partition_type=Partition.Cyclic)

    # Pipeline the flat per-op trip loop (the innermost op driver) at II=1.
    # Targets the k_passes loop which carries the MAC->BIRRD->accum body.
    s.pipeline("ring_core_0:k")


def build_feather_perf_simulator(M, K, N, AW, AH, Ty, num_inst,
                                 n_inner=1, k_passes=1, Nt_local=None):
    """Build the perf variant for the Allo NumPy simulator."""
    top = get_feather_perf_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(n_inner), int(k_passes), Nt_local=Nt_local,
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherModule(allo_mod, AW, n_inner)


def build_feather_perf_hls(M, K, N, AW, AH, Ty, num_inst,
                           mode="csim", project=None, n_inner=1,
                           k_passes=1, Nt_local=None):
    """Build the perf variant for Vitis HLS (csim / csyn / cosim)."""
    if project is None:
        project = "feather_perf.prj"
    top = get_feather_perf_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(n_inner), int(k_passes), Nt_local=Nt_local,
    )
    s = df.customize(top)
    schedule_perf_hls(s, int(K), int(N), int(AH), int(AW))
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherModule(allo_mod, AW, n_inner)
