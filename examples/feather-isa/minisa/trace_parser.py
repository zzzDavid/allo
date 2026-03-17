# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse FEATHER+ instruction trace JSON files into MINISA programs.

Supports three trace formats:

1. RTL trace (type not specified or "layer" key present):
   RTL compiler output with FEATHER_spec, VN layouts, ExecuteMapping entries.

2. MINISA trace (type="minisa"):
   Simple GEMM parameters that map to create_gemm_program().

3. MINISA manual trace (type="minisa_manual"):
   Manual tile specification for mixed-Gr, sr=0, or custom mappings.

4. Multi-layer trace (type="minisa_multi_layer"):
   Sequential GEMM layers with int8 intermediates.

Use load_trace() as the unified entry point — it auto-detects the format.
"""

import json
import math
from typing import Dict, Any, List

import numpy as np

from minisa.isa import (
    MINISAProgram,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    encode_program,
    create_gemm_program,
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

    # Extract mapping parameters (all EMs share Gr/Gc/sr/sc)
    ems = layer_data["ExecuteMapping"]
    Gr = ems[0]["G_r"]
    Gc = ems[0]["G_c"]
    sr = ems[0]["s_r"]
    sc = ems[0]["s_c"]

    # Extract dimensions from search_result
    search = layer_data["search_result"]
    M = search["Mt"]
    K = search["Kt"]
    Nt = search["Nt"]
    n_spatial_tiles = search["n_spatial_tiles"]
    N = Nt * n_spatial_tiles
    n_EMs = search["n_EMs"]

    rtl_latency = layer_data.get("latency", None)

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

    # Decompose N-dimension: the trace's sc/sr mapping covers Nt N-columns
    # per ExecuteMapping, but our single-pass-per-tile model only writes AH
    # output N-columns per tile. The RTL achieves full Nt coverage via WVN
    # temporal iteration (N_L1 = Nt/AH iterations per EM). We replicate this
    # by creating n_sub_tiles = Nt/AH sub-tiles, each covering AH N-columns
    # with sc=0 (direct sequential indexing).
    n_sub_tiles = Nt // AH
    assert Nt % AH == 0, f"Nt={Nt} must be divisible by AH={AH}"

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

    # Generate tiles: decompose each spatial tile's Nt N-columns into
    # n_sub_tiles sub-tiles of AH columns each, using sc=0 (direct indexing).
    # K is decomposed into separate tiles of Kt_per_pass elements each.
    for n_tile in range(n_spatial_tiles):
        for m_batch in range(n_m_batches):
            for n_sub in range(n_sub_tiles):
                n_base = n_tile * Nt + n_sub * AH
                for k_tile in range(k_passes):
                    program.add_mapping(SetMapping(
                        r0=k_tile * Kt_per_pass // AH,
                        c0=n_base,
                        Gr=Gr,
                        Gc=1,
                        sr=1,
                        sc=0,
                        m_start=m_batch * Mt,
                        m_end=(m_batch + 1) * Mt,
                        n_start=n_base,
                        n_end=n_base + AH,
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
        "Gc": Gc,
        "sr": sr,
        "sc": sc,
        "Nt": Nt,
        "n_spatial_tiles": n_spatial_tiles,
        "n_m_batches": n_m_batches,
        "n_sub_tiles": n_sub_tiles,
        "n_tiles": program.num_tiles(),
        "utilization": search.get("utilization", None),
    }


def parse_minisa_trace(trace_path: str) -> Dict[str, Any]:
    """Parse a MINISA trace JSON (type="minisa") into a MINISA program.

    Simple format that maps directly to create_gemm_program().

    Expected JSON:
    {
        "type": "minisa",
        "FEATHER_spec": {"AH": 8, "AW": 8},
        "program": {
            "M": 8, "K": 16, "N": 8,
            "gr": 4,  // optional, defaults to AW//2
            "ivn_order": 0, "wvn_order": 0, "ovn_order": 0,
            "iacts_zp": 0, "weights_zp": 0,
            "quant_scale": 0, "quant_zp": 0
        },
        "seed": 42,
        "description": "..."
    }
    """
    with open(trace_path) as f:
        trace = json.load(f)

    spec = trace["FEATHER_spec"]
    AH = spec["AH"]
    AW = spec["AW"]
    prog = trace["program"]

    M = prog["M"]
    K = prog["K"]
    N = prog["N"]
    gr = prog.get("gr", AW // 2)
    ivn_order = prog.get("ivn_order", 0)
    wvn_order = prog.get("wvn_order", 0)
    ovn_order = prog.get("ovn_order", 0)
    iacts_zp = prog.get("iacts_zp", 0)
    weights_zp = prog.get("weights_zp", 0)
    quant_scale = prog.get("quant_scale", 0)
    quant_zp = prog.get("quant_zp", 0)

    program = create_gemm_program(
        M=M, N=N, K=K, AH=AH, AW=AW,
        gr=gr,
        ivn_order=ivn_order,
        wvn_order=wvn_order,
        ovn_order=ovn_order,
        iacts_zp=iacts_zp,
        weights_zp=weights_zp,
        quant_scale=quant_scale,
        quant_zp=quant_zp,
    )
    instructions = encode_program(program)

    # Derive mapping params from gr
    Gr = gr
    Gc = 1
    sr = 1
    sc = 0

    return {
        "M": M,
        "K": K,
        "N": N,
        "AH": AH,
        "AW": AW,
        "M_padded": M,
        "program": program,
        "instructions": instructions,
        "rtl_latency": None,
        "Gr": Gr,
        "Gc": Gc,
        "sr": sr,
        "sc": sc,
        "Nt": AH,
        "n_spatial_tiles": N // AH,
        "n_m_batches": M // Gr,
        "n_sub_tiles": 1,
        "n_tiles": program.num_tiles(),
        "utilization": None,
        "seed": trace.get("seed", 42),
        "iacts_zp": iacts_zp,
        "weights_zp": weights_zp,
        "quant_scale": quant_scale,
        "quant_zp": quant_zp,
    }


def parse_manual_trace(trace_path: str) -> Dict[str, Any]:
    """Parse a manual MINISA trace (type="minisa_manual") with explicit tiles.

    For mixed-Gr, sr=0, or custom tile specifications that don't fit
    the standard create_gemm_program() flow.

    Expected JSON:
    {
        "type": "minisa_manual",
        "FEATHER_spec": {"AH": 4, "AW": 4},
        "gemm": {"M": 8, "K": 12, "N": 4},
        "ivn_order": 0, "wvn_order": 0, "ovn_order": 0,
        "iacts_zp": 0, "weights_zp": 0,
        "quant_scale": 0, "quant_zp": 0,
        "tiles": [
            {"Gr": 2, "Gc": 1, "sr": 1, "sc": 0,
             "m_start": 0, "m_end": 2, "n_start": 0, "n_end": 4,
             "k_start": 0, "k_end": 8},
            ...
        ],
        "seed": 99
    }
    """
    with open(trace_path) as f:
        trace = json.load(f)

    spec = trace["FEATHER_spec"]
    AH = spec["AH"]
    AW = spec["AW"]
    gemm = trace["gemm"]
    M = gemm["M"]
    K = gemm["K"]
    N = gemm["N"]

    ivn_order = trace.get("ivn_order", 0)
    wvn_order = trace.get("wvn_order", 0)
    ovn_order = trace.get("ovn_order", 0)
    iacts_zp = trace.get("iacts_zp", 0)
    weights_zp = trace.get("weights_zp", 0)
    quant_scale = trace.get("quant_scale", 0)
    quant_zp = trace.get("quant_zp", 0)

    program = MINISAProgram(
        name=f"manual_{M}x{K}x{N}_{AH}x{AW}",
        AH=AH,
        AW=AW,
        ivn_layout=SetIVNLayout(
            order=ivn_order,
            ML0=AH, ML1=max(1, M // AH),
            JL0=AH, JL1=max(1, K // AH),
            iacts_zp=iacts_zp,
        ),
        wvn_layout=SetWVNLayout(
            order=wvn_order,
            KL0=AH, KL1=max(1, K // AH),
            NL0=min(N, AW), NL1=max(1, N // AW),
            weights_zp=weights_zp,
        ),
        ovn_layout=SetOVNLayout(
            order=ovn_order,
            PL0=AH, PL1=max(1, M // AH),
            QL0=AH, QL1=max(1, N // AH),
            quant_scale=quant_scale,
            quant_zp=quant_zp,
        ),
    )

    tiles = trace["tiles"]
    first_Gr = tiles[0]["Gr"]
    first_sr = tiles[0].get("sr", 1)
    first_sc = tiles[0].get("sc", 0)
    first_Gc = tiles[0].get("Gc", 1)

    # Decompose tiles with K-range > Kt_per_pass into single-pass subtiles
    for t in tiles:
        Gr_t = t["Gr"]
        Kt_per_pass = (AW // Gr_t) * AH
        k_start_t = t["k_start"]
        k_end_t = t["k_end"]
        k_range = k_end_t - k_start_t
        num_k_tiles = max(1, k_range // Kt_per_pass)
        for kt in range(num_k_tiles):
            program.add_mapping(SetMapping(
                r0=t.get("r0", 0) + kt * Kt_per_pass // AH,
                c0=t.get("c0", t.get("n_start", 0)),
                Gr=Gr_t,
                Gc=t.get("Gc", 1),
                sr=t.get("sr", 1),
                sc=t.get("sc", 0),
                m_start=t["m_start"],
                m_end=t["m_end"],
                n_start=t["n_start"],
                n_end=t["n_end"],
                k_start=k_start_t + kt * Kt_per_pass,
                k_end=k_start_t + (kt + 1) * Kt_per_pass,
            ))

    instructions = encode_program(program)

    return {
        "M": M,
        "K": K,
        "N": N,
        "AH": AH,
        "AW": AW,
        "M_padded": M,
        "program": program,
        "instructions": instructions,
        "rtl_latency": None,
        "Gr": first_Gr,
        "Gc": first_Gc,
        "sr": first_sr,
        "sc": first_sc,
        "Nt": AH,
        "n_spatial_tiles": N // AH,
        "n_m_batches": max(1, M // first_Gr),
        "n_sub_tiles": 1,
        "n_tiles": program.num_tiles(),
        "utilization": None,
        "seed": trace.get("seed", 42),
        "iacts_zp": iacts_zp,
        "weights_zp": weights_zp,
        "quant_scale": quant_scale,
        "quant_zp": quant_zp,
    }


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Unified trace loader — auto-detects format and dispatches.

    Supports:
    - RTL trace (has "layer" key): parse_trace()
    - MINISA simple (type="minisa"): parse_minisa_trace()
    - MINISA manual (type="minisa_manual"): parse_manual_trace()
    """
    with open(trace_path) as f:
        trace = json.load(f)

    trace_type = trace.get("type", None)

    if trace_type == "minisa":
        return parse_minisa_trace(trace_path)
    elif trace_type == "minisa_manual":
        return parse_manual_trace(trace_path)
    elif trace_type == "minisa_multi_layer":
        return trace  # Return raw dict; multi-layer handled by test runner
    elif "layer" in trace:
        return parse_trace(trace_path)
    else:
        raise ValueError(f"Unknown trace format in {trace_path}. "
                        f"Expected 'type' field or 'layer' key.")
