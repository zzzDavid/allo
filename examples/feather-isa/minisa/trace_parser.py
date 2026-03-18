# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse FEATHER+ instruction trace JSON files into MINISA programs.

Supports the RTL trace format (has "layer" key with FEATHER_spec, VN layouts,
and ExecuteMapping entries). This is the canonical format produced by the RTL
compiler. Both uniform-Gr and mixed-Gr (adaptive) mappings are supported.

Use load_trace() as the entry point.
"""

import json
import math
from typing import Dict, Any

import numpy as np

from minisa.isa import (
    MINISAProgram,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    encode_program,
)


def parse_trace(trace_path: str) -> Dict[str, Any]:
    """Parse a FEATHER+ instruction trace JSON and generate a MINISA program.

    The trace encodes how the RTL FEATHER+ maps a GEMM workload. This parser
    converts that description into our MINISA instruction format.

    Key translation rules:
    - ExecuteMapping entries correspond to K-slices (r_0 selects K-slice).
      Each K-slice becomes a separate tile with its own k_start/k_end.
    - IVN M_L1 gives the number of M-batches (temporal M tiling).
    - n_spatial_tiles gives N-dimension tiling.
    - If M is not divisible by Gr, M is padded to the next multiple of Gr
      (padded rows compute 0*B=0 and are discarded).

    Args:
        trace_path: Path to JSON trace file.

    Returns:
        Dict with keys:
            M, K, N: Original matrix dimensions
            AH, AW: Array dimensions
            M_padded: Padded M dimension (next multiple of Gr)
            program: MINISAProgram
            instructions: Encoded int32 instruction array
            rtl_latency: Reference RTL latency (cycles), or None
            Gr, Gc, sr, sc: Mapping parameters
            n_tiles: Number of tile mappings
            utilization: RTL utilization percentage, or None
    """
    with open(trace_path) as f:
        trace = json.load(f)

    # Extract hardware spec
    spec = trace["FEATHER_spec"]
    AH = spec["AH"]
    AW = spec["AW"]

    # Process first layer
    layer = trace["layer"][0]
    layer_key = list(layer.keys())[0]
    layer_data = layer[layer_key]

    # Extract VN layout orders
    ivn = layer_data["IVN"]
    wvn = layer_data["WVN"]
    ovn = layer_data["OVN"]

    # Extract mapping parameters from all EMs
    ems = layer_data["ExecuteMapping"]
    n_EMs = len(ems)

    # Check if all EMs share the same Gr/Gc/sr/sc (uniform) or differ (mixed)
    em_params = [(em["G_r"], em["G_c"], em["s_r"], em["s_c"]) for em in ems]
    uniform_mapping = all(p == em_params[0] for p in em_params)

    # Extract dimensions from search_result
    search = layer_data["search_result"]
    M = search["Mt"]
    K = search["Kt"]
    Nt = search["Nt"]
    n_spatial_tiles = search["n_spatial_tiles"]
    N = Nt * n_spatial_tiles
    assert search["n_EMs"] == n_EMs

    rtl_latency = layer_data.get("latency", None)

    # Use first EM's sr/sc/Gc for Nt_local calculation (shared across EMs)
    Gr_first = ems[0]["G_r"]
    Gc_first = ems[0]["G_c"]
    sr_first = ems[0]["s_r"]
    sc_first = ems[0]["s_c"]

    if uniform_mapping:
        # === UNIFORM Gr: all EMs share same mapping params ===
        Gr, Gc, sr, sc = Gr_first, Gc_first, sr_first, sc_first

        # Compute K-passes
        Kt_per_pass = (AW // Gr) * AH
        assert K % Kt_per_pass == 0, (
            f"K={K} must be divisible by Kt_per_pass={Kt_per_pass} "
            f"(AW={AW}, Gr={Gr}, AH={AH})"
        )
        k_passes = K // Kt_per_pass
        assert k_passes == n_EMs, (
            f"K-passes mismatch: computed {k_passes} from K={K}/Kt_per_pass={Kt_per_pass}, "
            f"but trace has n_EMs={n_EMs}"
        )

        # Compute M padding (Gr rows per tile; pad if M not divisible)
        Mt = Gr
        M_padded = M if M % Mt == 0 else ((M // Mt) + 1) * Mt
        n_m_batches = M_padded // Mt

        # Verify trace consistency: N coverage per ExecuteMapping
        max_n_offset = sr * (AH - 1) + sc * ((AW - 1) & (Gc - 1))
        tile_n_coverage = max_n_offset + 1
        assert tile_n_coverage == Nt, (
            f"Tile N coverage {tile_n_coverage} != trace Nt={Nt}"
        )

        # N-dimension temporal iteration: the trace's sc/sr mapping covers Nt
        # N-columns per ExecuteMapping via WVN temporal iteration (N_L1 = Nt/AH
        # iterations per EM). Instead of creating n_sub_tiles * n_m_batches
        # separate tiles, we fold them into inner iterations within each tile.
        # Only K-decomposition creates separate ISA tiles (matching the RTL's EMs).
        n_sub_tiles = Nt // AH
        assert Nt % AH == 0, f"Nt={Nt} must be divisible by AH={AH}"

        # Inner iterations per tile: temporal N-iteration * M-batching * spatial N
        n_inner = n_spatial_tiles * n_m_batches * n_sub_tiles

        # Create MINISA program
        program = MINISAProgram(
            name=f"trace_{M}x{K}x{N}_{AH}x{AW}",
            AH=AH,
            AW=AW,
            ivn_layout=SetIVNLayout(
                order=ivn["order"],
                ML0=AH,
                ML1=n_m_batches,
                JL0=AH,
                JL1=K // AH,
            ),
            wvn_layout=SetWVNLayout(
                order=wvn["order"],
                KL0=AH,
                KL1=K // AH,
                NL0=min(N, AW),
                NL1=max(1, N // AW),
            ),
            ovn_layout=SetOVNLayout(
                order=ovn["order"],
                PL0=AH,
                PL1=n_m_batches,
                QL0=AH,
                QL1=N // AH,
            ),
        )

        # Generate tiles: only K-decomposition creates separate ISA tiles.
        # Each tile covers the full M and N range; temporal M/N iteration
        # is handled by n_inner sub-operations within the tile.
        for k_tile in range(k_passes):
            program.add_mapping(SetMapping(
                r0=k_tile * Kt_per_pass // AH,
                c0=0,
                Gr=Gr,
                Gc=1,
                sr=1,
                sc=0,
                m_start=0,
                m_end=M_padded,
                n_start=0,
                n_end=N,
                k_start=k_tile * Kt_per_pass,
                k_end=(k_tile + 1) * Kt_per_pass,
            ))

        instructions = encode_program(program)

        # Generate per-sub-operation m_start/n_start lookup tables.
        # Inner iteration order: n_spatial -> m_batch -> n_sub (matches RTL VN).
        total_ops = k_passes * n_inner
        inner_m_starts = np.zeros(total_ops, dtype=np.int32)
        inner_n_starts = np.zeros(total_ops, dtype=np.int32)

        for k_tile in range(k_passes):
            op_base = k_tile * n_inner
            op_idx = 0
            for n_tile in range(n_spatial_tiles):
                for m_batch in range(n_m_batches):
                    for n_sub in range(n_sub_tiles):
                        inner_m_starts[op_base + op_idx] = m_batch * Mt
                        inner_n_starts[op_base + op_idx] = n_tile * Nt + n_sub * AH
                        op_idx += 1

        return {
            "M": M,
            "K": K,
            "N": N,
            "AH": AH,
            "AW": AW,
            "M_padded": M_padded,
            "program": program,
            "instructions": instructions,
            "rtl_latency": rtl_latency,
            "Gr": Gr,
            "Gc": Gc,
            "sr": sr,
            "sc": sc,
            "Nt": Nt,
            "n_spatial_tiles": n_spatial_tiles,
            "n_m_batches": n_m_batches,
            "n_sub_tiles": n_sub_tiles,
            "n_inner": n_inner,
            "n_tiles": program.num_tiles(),
            "inner_m_starts": inner_m_starts,
            "inner_n_starts": inner_n_starts,
            "utilization": search.get("utilization", None),
        }

    else:
        # === MIXED Gr: use uniform max-Gr for HLS compatibility ===
        # Mixed-Gr tiles require += on C (read-modify-write), which violates
        # HLS single-writer constraint. Instead, use Gr=max(all Gr) uniformly
        # with k_passes = K / ((AW/Gr)*AH). All K contributions accumulate
        # in local_acc before flushing with write-only (=).
        Gr = max(em["G_r"] for em in ems)
        Gc, sr, sc = 1, 1, 0  # output-stationary

        # Verify K-tileability with uniform Gr
        Kt_per_pass = (AW // Gr) * AH
        assert K % Kt_per_pass == 0, (
            f"K={K} must be divisible by Kt_per_pass={Kt_per_pass} "
            f"(AW={AW}, Gr={Gr}, AH={AH})"
        )
        k_passes = K // Kt_per_pass

        # M padding
        Mt = Gr
        M_padded = M if M % Mt == 0 else ((M // Mt) + 1) * Mt
        n_m_batches = M_padded // Mt

        # N-dimension tiling
        Nt_isa = AH
        n_n_passes = N // Nt_isa
        assert N % Nt_isa == 0, f"N={N} must be divisible by Nt_isa={Nt_isa}"

        # Create MINISA program
        program = MINISAProgram(
            name=f"trace_{M}x{K}x{N}_{AH}x{AW}",
            AH=AH,
            AW=AW,
            ivn_layout=SetIVNLayout(
                order=ivn["order"],
                ML0=AH,
                ML1=M // AH,
                JL0=AH,
                JL1=K // AH,
            ),
            wvn_layout=SetWVNLayout(
                order=wvn["order"],
                KL0=AH,
                KL1=K // AH,
                NL0=min(N, AW),
                NL1=max(1, N // AW),
            ),
            ovn_layout=SetOVNLayout(
                order=ovn["order"],
                PL0=AH,
                PL1=M // AH,
                QL0=AH,
                QL1=N // AH,
            ),
        )

        # Generate tiles: N→M→K order (K innermost for block accumulation).
        # k_passes consecutive tiles share the same (m,n) region and
        # accumulate in local_acc before flushing to C.
        for n_pass in range(n_n_passes):
            for m_batch in range(n_m_batches):
                for k_tile in range(k_passes):
                    program.add_mapping(SetMapping(
                        r0=k_tile * Kt_per_pass // AH,
                        c0=n_pass * Nt_isa,
                        Gr=Gr,
                        Gc=1,
                        sr=1,
                        sc=0,
                        m_start=m_batch * Mt,
                        m_end=min((m_batch + 1) * Mt, M_padded),
                        n_start=n_pass * Nt_isa,
                        n_end=(n_pass + 1) * Nt_isa,
                        k_start=k_tile * Kt_per_pass,
                        k_end=(k_tile + 1) * Kt_per_pass,
                    ))

        instructions = encode_program(program)

        return {
            "M": M,
            "K": K,
            "N": N,
            "AH": AH,
            "AW": AW,
            "M_padded": M_padded,
            "program": program,
            "instructions": instructions,
            "rtl_latency": rtl_latency,
            "Gr": Gr,
            "Gc": 1,
            "sr": 1,
            "sc": 0,
            "Nt": Nt_isa,
            "n_spatial_tiles": n_n_passes,
            "n_m_batches": n_m_batches,
            "n_sub_tiles": 1,
            "n_inner": 1,
            "n_tiles": program.num_tiles(),
            "k_passes": k_passes,
            "mixed_gr": True,
            "utilization": search.get("utilization", None),
        }


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load and parse a FEATHER+ instruction trace JSON.

    Args:
        trace_path: Path to RTL trace JSON file.

    Returns:
        Dict with parsed trace info (see parse_trace for keys).
    """
    return parse_trace(trace_path)
