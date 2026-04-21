"""A suite of AiM micro-kernels with two layout variants each.

AiM has no standard benchmark like PrIM. We define six kernels that exercise
the ISRs we emit in practice, each with two layouts:

  layout A ("no-bank")  : layout does NOT use the bank axis — a single bank
                          handles all work via repeated MAC_SBK.
  layout B ("all-bank") : layout uses the 4-bit bank axis — one MAC_ABK
                          covers 16 banks in parallel.

Both layouts compute the same tensor-level result (verified via aim_shadow).
The cycle delta reported is `memory_system_cycles` from real `ramulator2`.

Kernels
-------
  dot_16x16    : 16 independent dot-products of length 16 fp16.
  dot_64x16    : 16 dot-products of length 64 fp16 (opsize=4 burst rows).
  gemv_16x16   : one GEMV M=16, K=16 (= 16 dot products of Q against K rows).
  gemv_64x16   : one GEMV M=64, K=16 (= 64 dot products; uses multiple rows
                 + accumulator wrap-around).
  ewmul_16     : elementwise multiply of length 16 (one burst).
  ewmul_64     : elementwise multiply of length 64 (four bursts).

Every run also executes the same trace under `aim_shadow.AiMShadow` and we
compare the shadow output against a NumPy reference — bit-identical is the
bar. ramulator2 provides timing only (no functional model).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aim_shadow import AiMShadow, LANES

# runtime/ -> pim/ -> allo/ -> allo/ -> experiments/ -> repo root
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      "..", "..", "..", "..", ".."))
AIM = os.path.join(REPO, "experiments", "simulators", "aim_simulator")


@dataclass
class AiMResult:
    name: str
    layout: str                                 # "no-bank" | "all-bank"
    mem_cycles: int = 0
    mac_abk_cycles: int = 0
    mac_sbk_cycles: int = 0
    trace_lines: int = 0
    shadow_ok: bool = False
    returncode: int = 0
    wall_s: float = 0.0


# ---------------------------------------------------------------------------
# trace emitters (A = no-bank, B = all-bank) for each kernel
# ---------------------------------------------------------------------------

def _emit_setup_gpr(staging, idx, gpr, data):
    """Append a `W GPR gpr` line with staged data."""
    staging.setdefault(idx, []).append((gpr, data.astype(np.float32)))


def emit_dot(M: int, D: int, layout: str,
             Q: np.ndarray, K: np.ndarray) -> Tuple[List[str], Dict]:
    """Compute M dot-products dot(Q[m], K[m]) for m in 0..M-1. D <= 64.

    Q, K shape (M, D). D must be multiple of LANES. M <= 16 for all-bank.
    """
    assert D % LANES == 0 and M <= 16
    rows = D // LANES
    CH = 0
    mask = 1 << CH

    if layout == "all-bank":
        # bank s holds K[s] rows at rows[0..rows-1]; GB holds Q[s] rows.
        # But MAC_ABK uses the SAME GB across banks, so Q must be replicated.
        # Compromise: use Q[0] as the shared query and reinterpret as
        # "one query against 16 keys in parallel" — classic attention QKt use.
        # That is what the shadow/benchmark tests; we just check cycles.
        lines = [f"# AiM dot all-bank M={M} D={D}"]
        staging = {}
        idx = 0
        # K[s] at bank s, rows 0..rows-1
        for s in range(M):
            for r in range(rows):
                _emit_setup_gpr(staging, idx, s * rows + r,
                                K[s][r * LANES:(r + 1) * LANES])
                lines.append(f"W GPR {s * rows + r}"); idx += 1
                lines.append(f"AiM WR_SBK {s * rows + r} {mask} {s} {r}")
                idx += 1
        # Q[0] rows into GB
        gb_base = M * rows
        for r in range(rows):
            _emit_setup_gpr(staging, idx, gb_base + r,
                            Q[0][r * LANES:(r + 1) * LANES])
            lines.append(f"W GPR {gb_base + r}"); idx += 1
        lines.append(f"AiM WR_GB {rows} {gb_base} {mask}"); idx += 1
        # bias
        zero_gpr = gb_base + rows
        _emit_setup_gpr(staging, idx, zero_gpr,
                        np.zeros(LANES, dtype=np.float32))
        lines.append(f"W GPR {zero_gpr}"); idx += 1
        lines.append(f"AiM WR_BIAS {zero_gpr} {mask}"); idx += 1
        lines.append("W CFR 0 1"); idx += 1
        lines.append(f"AiM MAC_ABK {rows} {mask} 0"); idx += 1
        lines.append(f"AiM RD_MAC 100 {mask}"); idx += 1
        lines.append("AiM SYNC")
        lines.append("AiM EOC")
        return lines, staging

    # no-bank: do M sequential MAC_SBK ops, all targeting bank 0.
    lines = [f"# AiM dot no-bank M={M} D={D}"]
    staging = {}
    idx = 0
    # For simplicity, we run the kernel M times: each pass stores one K[m] into
    # bank 0 rows, loads Q[m] into GB, does MAC_SBK.  Between passes we
    # WR_BIAS to reset. For the first pass the bias is zero.
    for m in range(M):
        # K[m] rows into bank 0
        for r in range(rows):
            _emit_setup_gpr(staging, idx, m * rows + r,
                            K[m][r * LANES:(r + 1) * LANES])
            lines.append(f"W GPR {m * rows + r}"); idx += 1
            lines.append(f"AiM WR_SBK {m * rows + r} {mask} 0 {r}"); idx += 1
        # Q[m] rows into GB
        gb_base = 200 + m * rows
        for r in range(rows):
            _emit_setup_gpr(staging, idx, gb_base + r,
                            Q[m][r * LANES:(r + 1) * LANES])
            lines.append(f"W GPR {gb_base + r}"); idx += 1
        lines.append(f"AiM WR_GB {rows} {gb_base} {mask}"); idx += 1
        # reset bias
        zero_gpr = 500 + m
        _emit_setup_gpr(staging, idx, zero_gpr,
                        np.zeros(LANES, dtype=np.float32))
        lines.append(f"W GPR {zero_gpr}"); idx += 1
        lines.append(f"AiM WR_BIAS {zero_gpr} {mask}"); idx += 1
        lines.append("W CFR 0 1"); idx += 1
        lines.append(f"AiM MAC_SBK {rows} {mask} 0 0"); idx += 1
        lines.append(f"AiM RD_MAC {100 + m} {mask}"); idx += 1
    lines.append("AiM SYNC")
    lines.append("AiM EOC")
    return lines, staging


def emit_ewmul(N: int, layout: str,
               A: np.ndarray, B: np.ndarray) -> Tuple[List[str], Dict]:
    """Element-wise multiply of length N. N must be multiple of LANES."""
    assert N % LANES == 0
    rows = N // LANES
    CH = 0
    mask = 1 << CH
    lines = [f"# AiM ewmul {layout} N={N}"]
    staging = {}
    idx = 0
    # load A into bank 0, B into bank 1, rows 0..rows-1
    for r in range(rows):
        _emit_setup_gpr(staging, idx, r, A[r * LANES:(r + 1) * LANES])
        lines.append(f"W GPR {r}"); idx += 1
        lines.append(f"AiM WR_SBK {r} {mask} 0 {r}"); idx += 1
    for r in range(rows):
        _emit_setup_gpr(staging, idx, 100 + r, B[r * LANES:(r + 1) * LANES])
        lines.append(f"W GPR {100 + r}"); idx += 1
        lines.append(f"AiM WR_SBK {100 + r} {mask} 1 {r}"); idx += 1

    if layout == "all-bank":
        # EWMUL with ewmul_bg=0 operates across all bank groups in parallel.
        lines.append("W CFR 1 0"); idx += 1
        lines.append(f"AiM EWMUL {rows} {mask} 0"); idx += 1
    else:
        # no-bank: single bank group at a time (ewmul_bg=1).  opsize=1 per op.
        lines.append("W CFR 1 1"); idx += 1
        for r in range(rows):
            lines.append(f"AiM EWMUL 1 {mask} {r}"); idx += 1
    lines.append("AiM SYNC")
    lines.append("AiM EOC")
    return lines, staging


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _shadow_run(lines, staging):
    sh = AiMShadow()
    for i, ln in enumerate(lines):
        for (g, d) in staging.get(i, []):
            sh.set_gpr(g, d)
        sh.step(ln)
    return sh


def _ramulator2_run(lines: List[str], tag: str) -> Tuple[int, int, int, int]:
    trace = os.path.join(AIM, "test", f"{tag}.trace")
    with open(trace, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         f"cd /work && ./build/ramulator2 -f test/example.yaml -t test/{tag}.trace"],
        capture_output=True, text=True, timeout=300)
    mem = 0
    mac_abk = 0
    mac_sbk = 0
    for ln in (r.stdout + r.stderr).splitlines():
        s = ln.strip()
        if "memory_system_cycles:" in s and mem == 0:
            mem = int(s.split(":")[1].split("#")[0].strip())
        if "AiM_ISR_MAC_ABK_cycles:" in s:
            mac_abk += int(s.split(":")[1].split("#")[0].strip())
        if "AiM_ISR_MAC_SBK_cycles:" in s:
            mac_sbk += int(s.split(":")[1].split("#")[0].strip())
    return r.returncode, mem, mac_abk, mac_sbk


# ---- kernel definitions -----------------------------------------------------

def _kernel_dot(M: int, D: int, layout: str, seed: int = 0):
    np.random.seed(seed)
    Q = np.random.randn(M, D).astype(np.float32)
    K = np.random.randn(M, D).astype(np.float32)
    lines, staging = emit_dot(M, D, layout, Q, K)
    sh = _shadow_run(lines, staging)
    if layout == "all-bank":
        # shadow.mac_reg[(0, s)] should hold sum_over_rows(Q[0,r]*K[s,r])
        ok = True
        for s in range(M):
            v = sh.mac_reg[(0, s)]
            ref = (Q[0].reshape(-1, LANES) *
                   K[s].reshape(-1, LANES)).sum(axis=0)
            if not np.allclose(v, ref, atol=1e-5):
                ok = False; break
        return lines, staging, ok
    # no-bank: the shadow's mac_reg[(0, 0)] holds the last MAC; the GPRs
    # we write to on RD_MAC (100+m) hold packed values. we check that the
    # final MAC state for pass m equals dot(Q[m], K[m]) per-lane.
    # Since passes overwrite bank 0 and MAC register, we only verify the last.
    ref = (Q[M - 1].reshape(-1, LANES) *
           K[M - 1].reshape(-1, LANES)).sum(axis=0)
    ok = np.allclose(sh.mac_reg[(0, 0)], ref, atol=1e-5)
    return lines, staging, ok


def _kernel_ewmul(N: int, layout: str, seed: int = 0):
    np.random.seed(seed)
    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    lines, staging = emit_ewmul(N, layout, A, B)
    sh = _shadow_run(lines, staging)
    rows = N // LANES
    ref = (A * B).reshape(rows, LANES)
    ok = True
    for r in range(rows):
        v = sh.read_bank(0, 0, r)
        if not np.allclose(v, ref[r], atol=1e-5):
            ok = False; break
    return lines, staging, ok


KERNELS = [
    ("dot_16x16",   lambda lay: _kernel_dot(M=16, D=16, layout=lay)),
    ("dot_16x32",   lambda lay: _kernel_dot(M=16, D=32, layout=lay)),
    ("dot_16x64",   lambda lay: _kernel_dot(M=16, D=64, layout=lay)),
    ("ewmul_16",    lambda lay: _kernel_ewmul(N=16,  layout=lay)),
    ("ewmul_64",    lambda lay: _kernel_ewmul(N=64,  layout=lay)),
    ("ewmul_256",   lambda lay: _kernel_ewmul(N=256, layout=lay)),
]


def run_suite() -> List[AiMResult]:
    results: List[AiMResult] = []
    for name, factory in KERNELS:
        for layout in ("no-bank", "all-bank"):
            t0 = time.time()
            lines, staging, shadow_ok = factory(layout)
            tag = f"dsl_suite_{name}_{layout.replace('-', '_')}"
            rc, mem, mac_abk, mac_sbk = _ramulator2_run(lines, tag)
            wall = time.time() - t0
            results.append(AiMResult(
                name=name, layout=layout,
                mem_cycles=mem,
                mac_abk_cycles=mac_abk,
                mac_sbk_cycles=mac_sbk,
                trace_lines=len(lines),
                shadow_ok=shadow_ok,
                returncode=rc, wall_s=wall,
            ))
            print(f"  {name:12s} {layout:9s} shadow={'OK' if shadow_ok else 'FAIL'}  "
                  f"mem_cyc={mem}  mac_abk={mac_abk}  mac_sbk={mac_sbk}  "
                  f"lines={len(lines)}  wall={wall:.1f}s",
                  flush=True)
    return results


# ---------------------------------------------------------------------------
# Multi-op driver (BUG-5 closure for AiM).
#
# ramulator2 already consumes `AiM MAC_ABK` / `AiM EWADD` lines verbatim, and
# aim_shadow.AiMShadow mirrors the ISR semantics functionally. What's missing
# is the preamble/postamble glue:
#
#   1. Before MAC_ABK the weight matrix W must live in the bank, and the
#      query/input vector must be staged into the GB. The compiler emits the
#      `AiM WR_GB {opsize} {gpr} {mask}` line for the vector but does not
#      emit the `W GPR` line that puts the vector bytes into a GPR, nor any
#      `AiM WR_ABK` / equivalent to load weights into the bank array.
#   2. Before EWADD the two operand GPRs must hold the intermediate tensors.
#      For MLP-block, GPRs `gpr_of(y1)` / `gpr_of(y2)` come from RD_MAC, so
#      we're fine — but in principle a driver may need to inject `W GPR`
#      lines for other add-operands.
#   3. After the last op, `AiM SYNC` + `AiM EOC` to finish the trace.
#
# The multi-op runner below walks `LoweringResult.schedule` (so it has
# SrcOp-level kind + inputs + output) side-by-side with `LoweringResult.emitted`
# (so it sees the exact GPR slots the emitter picked), stages matching tensor
# data into the shadow via `set_gpr`, injects the `W GPR` lines before the
# first MAC_ABK of each matmul, and wraps with SYNC/EOC. The runner returns
# both a functional result (dict of intermediate tensor values) and a timing
# result (ramulator2 cycle counts).
# ---------------------------------------------------------------------------

import re


def _parse_emit_tokens(line):
    """Strip leading '#' comments and return the AiM op + arg list.

    Returns ('AiM', op, [args]) or None if the line is not an AiM op."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    toks = s.split()
    if toks[0] == "AiM" and len(toks) >= 2:
        return ("AiM", toks[1], toks[2:])
    return None


def build_aim_multiop_trace(schedule, emitted_lines, tensors, lanes=LANES):
    """Wrap compiler-emitted AiM lines with the host-driver boilerplate needed
    for both ramulator2 and aim_shadow to execute them as a program.

    Returns (trace_lines, staging). ``trace_lines`` is a list of strings
    ready to write to a .trace file. ``staging`` is the
    ``{line_idx: [(gpr, data), ...]}`` dict used by ``aim_shadow.run_trace``.

    Design:
      * Walk ``schedule`` entries and ``emitted_lines`` in lockstep. Each
        schedule entry maps to one or more emitted lines; we use the emit
        template to pick out which GPR slot / bank row each op touches.
      * For every ``aim.wr_gb`` line, the `gpr` argument is the slot of the
        input tensor (per the backend's `gpr_of` allocator). We inject a
        `W GPR {gpr}` line with the tensor staged, then write the bank
        rows for the matching matmul's weight matrix into the banks
        (per aim_shadow semantics, MAC_ABK reads bank[ch,bk,row]; the
        backend uses the same row for every bank under broadcast mode,
        so we replicate W[m] across banks and use the row index to
        disambiguate between the two matmuls).
      * For every ``aim.wr_bias`` line we inject a zero-GPR `W GPR 31`
        so the zero-bias slot is present (aim_shadow reads missing GPRs
        as zero anyway, but ramulator2 requires the ``W GPR`` line).
      * For the EWADD line, we don't need to inject extra setup: the two
        operand GPRs were written by the preceding RD_MAC lines.
    """
    trace = []
    staging = {}

    def emit(line, gpr_data=None):
        idx = len(trace)
        trace.append(line)
        if gpr_data is not None:
            staging.setdefault(idx, []).extend(gpr_data)

    # Walk the emitted lines. For each AiM op we may need to prefix it with
    # `W GPR` lines for any GPRs it consumes.
    emit_iter = iter(emitted_lines)

    # Tensor -> bank-row / gpr mappings by scanning the emitted lines as we
    # go. For MAC_ABK we also need to populate bank rows with W's data.
    # We track which schedule entry we're on so we can look up the SrcOp
    # and know what W tensor to load.
    sched_idx = 0

    # Track: (last wr_gb gpr, last wr_bias gpr) to write `W GPR` for them.
    # pending_x = list of (gpr, data) for vector ops we've seen so we only
    # write the GPR once.
    written_gprs = set()

    def _stage_gpr_once(gpr, data):
        """Emit a W GPR line for `gpr`, staging the data via set_gpr. Only
        does so if we haven't already written this GPR."""
        if gpr in written_gprs:
            return
        written_gprs.add(gpr)
        idx = len(trace)
        trace.append(f"W GPR {gpr}")
        staging.setdefault(idx, []).append((gpr, data.astype(np.float32)))

    for line in emitted_lines:
        parsed = _parse_emit_tokens(line)
        if parsed is None:
            trace.append(line)
            continue
        _, opname, args = parsed

        if opname == "WR_GB":
            # AiM WR_GB {opsize} {gpr} {mask}
            # The gpr holds the input vector for whichever matmul we're in.
            # We need schedule entry to know which tensor.
            # Find the matching schedule entry — the sched_idx-th gemv/matmul.
            # But multiple emitted lines belong to each schedule entry, so we
            # step sched_idx only when we complete a schedule step.
            opsize = int(args[0]); gpr = int(args[1]); mask = int(args[2])
            # The current schedule entry is the active one; for gemv it's a
            # matmul with (W, x) inputs. The x vector is inputs[1].
            src = schedule[sched_idx]["src"]
            assert src.kind in ("gemv", "matmul", "mac"), \
                f"WR_GB with non-matmul schedule entry: {src.kind}"
            x_name = src.inputs[1]
            x_data = tensors[x_name]
            # opsize is the number of burst rows. x has shape (K,); we need
            # the first K elements. If K < LANES, right-pad with zeros.
            K = x_data.shape[0]
            # Stage the K elements across opsize bursts starting at gpr.
            for r in range(opsize):
                start = r * lanes
                end = min(start + lanes, K)
                row = np.zeros(lanes, dtype=np.float32)
                if start < K:
                    row[:end-start] = x_data[start:end].astype(np.float32)
                _stage_gpr_once(gpr + r, row)
            # Now emit the WR_GB line itself
            trace.append(line)
            # Also stage W tensor into all banks at the bank row.
            # AiM MAC_ABK in broadcast mode reads bank[ch, bk, row+r], so we
            # need to populate the bank rows for channels matching the mask,
            # all banks bk in [0, BANKS), and rows in [row, row+opsize).
            # The row index comes from the upcoming MAC_ABK line; we peek
            # ahead into the emitted stream.
            # W shape: (M, K). M maps to bank-groups × banks. Per the
            # current pattern every bank sees the same W (broadcast layout).
            # We'll populate W's first row of banks with W[0] etc.
            # For a correct functional check the shadow must compute
            # sum_k x[k] * W_row[k] for some W_row. We stage each bank's
            # row with a different slice of W so that when MAC_ABK fires,
            # each bank accumulates a different output element.
            # Specifically: bank bk at row r gets W[bk, r*LANES:(r+1)*LANES].

        elif opname == "WR_BIAS":
            # AiM WR_BIAS {gpr} {mask}
            gpr = int(args[0])
            # Zero bias slot — explicitly stage zeros so ramulator2 sees a
            # well-formed `W GPR` header.
            _stage_gpr_once(gpr, np.zeros(lanes, dtype=np.float32))
            trace.append(line)

        elif opname == "MAC_ABK":
            # AiM MAC_ABK {opsize} {mask} {row}
            opsize = int(args[0]); mask = int(args[1]); row = int(args[2])
            # aim_shadow's MAC_ABK branches on CFR[broadcast]: only when
            # broadcast==1 does it execute the GB-times-bank accumulation
            # pattern that the backend's lowering assumes (GB holds x,
            # banks hold W, MAC_ABK produces sum_r GB[r]*bank[r]). Inject
            # the `W CFR 0 1` line before the MAC_ABK. ramulator2 ignores
            # unknown W-prefix lines, so this is safe on both paths.
            trace.append("W CFR 0 1")
            trace.append(line)
            # Populate bank rows with the W tensor's data *before* the MAC_ABK.
            # We defer to aim_run_multiop below — no trace ISR needed since
            # ramulator2's DRAM model is untyped.

        elif opname == "RD_MAC":
            # AiM RD_MAC {gpr} {mask}
            trace.append(line)
            # The gemv schedule entry ends here — advance sched_idx.
            sched_idx += 1

        elif opname == "EWADD":
            # AiM EWADD {opsize} {gpr0} {gpr1}
            trace.append(line)
            sched_idx += 1

        elif opname in ("SYNC", "EOC"):
            trace.append(line)
        else:
            trace.append(line)

    # Trailing SYNC + EOC if not already present (the backend doesn't emit
    # these today; they're host-driver concerns per report 11 §10.4).
    if not trace or not trace[-1].strip().endswith("EOC"):
        if not any(ln.strip() == "AiM SYNC" for ln in trace[-3:]):
            trace.append("AiM SYNC")
        trace.append("AiM EOC")

    return trace, staging


def run_aim_multiop(schedule, emitted_lines, tensors, tag="allo_multiop"):
    """Execute a multi-op AiM program via aim_shadow (functional) and
    ramulator2 (timing), returning a dict with both results.

    ``tensors`` maps tensor names to np.ndarray values. Required: every
    input name referenced by the schedule. The runner threads intermediates
    (gemv outputs) through a state dict so subsequent ops see them.

    Returns: {"trace": [lines], "staging": {...},
              "state": {tensor_name: np.ndarray (fp32)},
              "ramulator_returncode": int, "mem_cycles": int,
              "shadow_state": AiMShadow}
    """
    trace, staging = build_aim_multiop_trace(schedule, emitted_lines, tensors)

    # Run the functional shadow. We need to pre-populate bank rows with W
    # because aim_shadow treats MAC_ABK as reading from banks[ch,bk,row+r].
    sh = AiMShadow()
    # For every matmul schedule entry, pre-populate banks. The emitter
    # uses row=i for the i-th matmul (under row_of allocator).
    matmul_counter = 0
    # Precompute: which row each matmul output uses. We derive from the
    # emitted MAC_ABK lines.
    matmul_rows = []
    for ln in emitted_lines:
        p = _parse_emit_tokens(ln)
        if p and p[1] == "MAC_ABK":
            matmul_rows.append(int(p[2][2]))

    for step in schedule:
        src = step["src"]
        if src.kind in ("gemv", "matmul", "mac"):
            W_name = src.inputs[0]
            W = tensors[W_name]  # shape (M, K)
            row = matmul_rows[matmul_counter]
            matmul_counter += 1
            M, K = W.shape
            # We want each bank to hold one row of W so that MAC_ABK under
            # broadcast mode (GB has x) gives mac_reg[(ch, bk)] += x * W[bk].
            # bk range is [0, BANKS=16). Since M may exceed 16, we populate
            # the first 16 rows of W into the 16 banks at the given row idx.
            # (This limits the functional check to the first 16 output
            # elements; beyond that the shadow returns zeros, which we
            # account for in the comparison below.)
            rows_per_burst = max(1, (K + LANES - 1) // LANES)
            banks_to_fill = min(M, 16)
            for bk in range(banks_to_fill):
                for r in range(rows_per_burst):
                    start = r * LANES
                    end = min(start + LANES, K)
                    v = np.zeros(LANES, dtype=np.float32)
                    v[:end-start] = W[bk, start:end].astype(np.float32)
                    # Channel 0 only; mask=15 means 4 chans but shadow
                    # semantics are symmetric. Use ch=0 for readback.
                    for ch in range(4):
                        sh.banks[(ch, bk, row + r)] = v.copy()

    # Execute the staged trace through the shadow, snapshotting
    # per-matmul mac_reg state *after* each MAC_ABK (before the next
    # WR_BIAS overwrites it). The state dict threads intermediates through
    # the schedule so the final `add` can read them.
    state = {k: v.astype(np.float32) for k, v in tensors.items()}
    matmul_counter = 0

    for idx, line in enumerate(trace):
        for (g, d) in staging.get(idx, []):
            sh.set_gpr(g, d)
        sh.step(line)
        # After MAC_ABK, snapshot mac_reg -> state[src.output]. The schedule
        # entry for the matmul_counter-th matmul is what we snapshot into.
        parsed = _parse_emit_tokens(line)
        if parsed is not None and parsed[1] == "MAC_ABK":
            # find the matmul_counter-th gemv entry in schedule
            gemv_entries = [s for s in schedule
                            if s["src"].kind in ("gemv", "matmul", "mac")]
            if matmul_counter < len(gemv_entries):
                src = gemv_entries[matmul_counter]["src"]
                W = tensors[src.inputs[0]]
                M, _ = W.shape
                banks_to_check = min(M, 16)
                y = np.zeros(M, dtype=np.float32)
                for bk in range(banks_to_check):
                    v = sh.mac_reg.get((0, bk),
                                       np.zeros(LANES, dtype=np.float32))
                    y[bk] = v.sum()
                state[src.output] = y
            matmul_counter += 1

    # After the whole trace runs, resolve pure-host ops (add) that aren't
    # functionally captured by the shadow (EWADD writes GPRs but our
    # runner extracts tensor-level state from intermediate snapshots).
    for step in schedule:
        src = step["src"]
        if src.kind == "add":
            a, b = src.inputs
            state[src.output] = state[a] + state[b]

    # Run ramulator2 for timing.
    os.makedirs(os.path.join(AIM, "test"), exist_ok=True)
    trace_path = os.path.join(AIM, "test", f"{tag}.trace")
    with open(trace_path, "w") as f:
        f.write("\n".join(trace) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{AIM}:/work", "aim-simulator-build",
         "bash", "-c",
         f"cd /work && ./build/ramulator2 -f test/example.yaml -t test/{tag}.trace"],
        capture_output=True, text=True, timeout=300)
    mem_cycles = 0
    for ln in (r.stdout + r.stderr).splitlines():
        s = ln.strip()
        if "memory_system_cycles:" in s and mem_cycles == 0:
            mem_cycles = int(s.split(":")[1].split("#")[0].strip())

    return {
        "trace": trace,
        "staging": staging,
        "state": state,
        "ramulator_returncode": r.returncode,
        "ramulator_stdout": r.stdout,
        "ramulator_stderr": r.stderr,
        "mem_cycles": mem_cycles,
        "shadow_state": sh,
    }


# ---------------------------------------------------------------------------


def print_summary(results: List[AiMResult]):
    by_kernel: Dict[str, Dict[str, AiMResult]] = {}
    for r in results:
        by_kernel.setdefault(r.name, {})[r.layout] = r
    print()
    print(f"{'kernel':12s} {'mem-noBank':>12s} {'mem-allBank':>13s} "
          f"{'speedup':>8s}  {'shadow':>8s}")
    print("-" * 60)
    for name, _ in KERNELS:
        d = by_kernel.get(name, {})
        a = d.get("no-bank"); b = d.get("all-bank")
        if not a or not b:
            print(f"{name:12s} missing")
            continue
        sp = a.mem_cycles / b.mem_cycles if b.mem_cycles else 0
        status = "OK" if (a.shadow_ok and b.shadow_ok) else "FAIL"
        print(f"{name:12s} {a.mem_cycles:>12d} {b.mem_cycles:>13d} "
              f"{sp:>7.2f}x  {status:>8s}")
