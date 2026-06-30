# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: validated numpy references + verdict helper.

Per the reference-provenance note (design_doc/results/polybench-reference-provenance.md):
the eleven Tier-1 numpy references are Allo's own `<k>_np` functions in
`examples/polybench/<k>.py`, RE-EXPORTED here (not re-authored) so every suite
`PASS` rests on a recorded, validated reference. The provenance probe
`experiments/scripts/R_polybench_ref_validation.py` confirmed
`numpy reference == canonical PolyBench/C 4.2.1` at SMALL (atax exact; gemm exact
at alpha=1, provably differs at alpha=1.5 -- pinning the one caveat).

The verdict taxonomy and the per-backend fp tolerance are spec Answer 3.
"""

from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass

# tests/pim/lib/reference.py -> parents[3] == experiments/allo
_POLYBENCH_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "polybench"
if str(_POLYBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_POLYBENCH_DIR))

# Re-export the eleven validated Allo numpy references. The modules import
# `allo` at top level; that is the same dependency the rest of the suite needs.
from gemm import gemm_np  # noqa: E402
from gemver import gemver_np  # noqa: E402
from gesummv import gesummv_np  # noqa: E402
from atax import atax_np  # noqa: E402
from bicg import bicg_np  # noqa: E402
from mvt import mvt_np  # noqa: E402
from two_mm import two_mm_np  # noqa: E402
from three_mm import three_mm_np  # noqa: E402
from doitgen import doitgen_np  # noqa: E402
from covariance import covariance_np  # noqa: E402
from syrk import syrk_np  # noqa: E402
from syr2k import syr2k_np  # noqa: E402

# Tier-2 (Phase 4): triangular-access BLAS-3 + modified Gram-Schmidt. Same
# validated-reference discipline as Tier-1 -- re-export Allo's own `<k>_np`
# (canonical PolyBench/C 4.2.1, SMALL); the triangular `j<=i`/`k>i` mask and
# the orthogonalization are applied host-side against these refs (the
# `@allo.work` is the bare MAC-reduction core, mirroring syrk).
from symm import symm_np  # noqa: E402
from trmm import trmm_np  # noqa: E402
from gramschmidt import gramschmidt_np  # noqa: E402

# The Tier-1 + Tier-2 references keyed by suite kernel name (the canonical
# PolyBench names; `2mm`/`3mm` map to Allo's `two_mm`/`three_mm`).
REFERENCES = {
    "gemm": gemm_np,
    "gemver": gemver_np,
    "gesummv": gesummv_np,
    "atax": atax_np,
    "bicg": bicg_np,
    "mvt": mvt_np,
    "2mm": two_mm_np,
    "3mm": three_mm_np,
    "doitgen": doitgen_np,
    "covariance": covariance_np,
    "syrk": syrk_np,
    "syr2k": syr2k_np,
    # Tier-2 (Phase 4):
    "symm": symm_np,
    "trmm": trmm_np,
    "gramschmidt": gramschmidt_np,
}

# --------------------------------------------------------------------- #
# Provenance block (the anti-fabrication record; design provenance note)
# --------------------------------------------------------------------- #

SOURCE = "PolyBench/C 4.2.1"
SIZE_CLASS = "SMALL_DATASET"
VALIDATED_BY = "experiments/scripts/R_polybench_ref_validation.py"

# gemm/2mm: canonical PolyBench sets alpha=1.5; thread it as a workload scalar
# so the reference stays bit-for-bit canonical (provenance-note recommendation
# (a), spec Answer 1). Recorded per-kernel for the alpha-bearing kernels.
ALPHA_CONVENTION = {
    "gemm": "canonical alpha=1.5 threaded as workload scalar (Allo gemm_np folds beta only)",
    "2mm": "canonical alpha=1.5 threaded as workload scalar",
    # Tier-2: symm/trmm carry the canonical alpha (and symm beta); the
    # @allo.work computes the bare MAC product, the scale+triangular mask are
    # applied host-side against symm_np/trmm_np.
    "symm": "canonical alpha=1.5, beta=1.2 applied host-side (symm_np); @allo.work is the bare B^T@A product",
    "trmm": "canonical alpha=1.5 applied host-side (trmm_np); @allo.work is the bare A^T@B product",
}


def provenance(kernel: str) -> dict:
    """The provenance record for `kernel`, embedded in results.json."""
    rec = {
        "source": SOURCE,
        "size_class": SIZE_CLASS,
        "validated_by": VALIDATED_BY,
    }
    if kernel in ALPHA_CONVENTION:
        rec["alpha_convention"] = ALPHA_CONVENTION[kernel]
    return rec


# --------------------------------------------------------------------- #
# Verdict taxonomy (spec Answer 3)
# --------------------------------------------------------------------- #

PASS = "PASS"
FAIL = "FAIL"
CYCLES_ONLY = "CYCLES-ONLY"
BLOCKED_SIM = "BLOCKED-SIM"
BLOCKED_DEVICE = "BLOCKED-DEVICE"
OUT_OF_PARADIGM = "OUT-OF-PARADIGM"

# Per-backend fp tolerance (spec Answer 3 table). The canonical init is fp32;
# PIM backends downcast, so the tolerance absorbs the downcast -- the reference
# is NOT weakened. AiM is a trace sim (no functional numerics) -> no tolerance.
TOLERANCE = {
    "apu_v1": {"rtol": 2e-2, "atol": 2e-2},          # fp16 bit-serial (GEMV-walkthrough ~0.0156, +1 notch)
    "upmem": {"rtol": 1e-4, "atol": 1e-4},           # functional DPU exec (probe fp32 tolerance)
    "samsung_hbm_pim": {"rtol": 2e-2, "atol": 2e-2}, # fp16 near-bank
    "aim": None,                                     # CYCLES-ONLY (trace sim)
}


@dataclass(frozen=True)
class Verdict:
    """A first-class correctness verdict (spec Answer 3) -- written to
    results.json, NOT a pytest skip. `status` is one of the taxonomy constants;
    `detail` is the human-readable reason (matched-within-tol / why CYCLES-ONLY
    / which environment limit)."""
    status: str
    detail: str


def tolerance_for(backend: str) -> dict | None:
    """The per-backend (rtol, atol), or None for a non-functional backend."""
    return TOLERANCE.get(backend)


def assert_matches(out, ref, *, backend: str, kernel: str) -> Verdict:
    """fp-tolerant compare of a backend output against the validated numpy ref.

    Returns a `Verdict`:
      - `PASS` when the backend functionally executed AND `out` matches `ref`
        within the per-backend tolerance,
      - `FAIL` when it executed AND the numbers differ (a real discrepancy),
      - `CYCLES-ONLY` when `out is None` (the backend surfaced no output array --
        e.g. a trace sim, or a run path that reports cycles but no numerics).

    A `FAIL` is a real numeric discrepancy and is recorded loudly, never hidden
    behind xfail (spec Answer 3). Environment limits (BLOCKED-*) and the
    out-of-paradigm scope calls are assigned by the caller, not here.
    """
    import numpy as np

    tol = tolerance_for(backend)
    if out is None or tol is None:
        why = (
            "trace sim: no functional numerics"
            if tol is None
            else "run path surfaced no output array"
        )
        return Verdict(CYCLES_ONLY, f"{backend}/{kernel}: {why}")

    a = np.asarray(out)
    r = np.asarray(ref)
    if a.shape != r.shape:
        a = a.reshape(r.shape)
    if np.allclose(a, r, rtol=tol["rtol"], atol=tol["atol"]):
        return Verdict(
            PASS,
            f"matched numpy ref rtol={tol['rtol']:g} atol={tol['atol']:g}",
        )
    max_abs = float(np.max(np.abs(a.astype(np.float64) - r.astype(np.float64))))
    return Verdict(
        FAIL,
        f"output differs from numpy ref (max_abs_err={max_abs:g}, "
        f"rtol={tol['rtol']:g} atol={tol['atol']:g})",
    )


def verdict_for_run(result, *, backend: str, kernel: str) -> Verdict:
    """Derive the cell verdict from what the run path actually SURFACES.

    None of the three simulated backends return an output array to Python
    (`RunResult.extra['outputs']` is APU-v1-only); the functional check, where it
    exists, is the SIMULATOR HOST'S OWN internal numeric verification. This helper
    encodes that honestly per backend (spec Answer 3), keyed on `RunResult`:

      - **UPMEM, GEMV host slot, cycles returned** -> `PASS`. The uPIMulator GEMV
        host computes `W @ x` and byte-compares the DPU output against it; a
        mismatch PANICS with no cycle line (the task-011 failure mode). So a
        returned cycle count means the emitted kernel matched the host's
        reference within the host's own check -- functional execution verified by
        the sim, with the validated numpy ref recorded as the matching reference.
      - **UPMEM, TENON (VA) host slot** -> `CYCLES-ONLY`. The VA host checks its
        own `a + b`, not this kernel, so a returned cycle count does NOT verify
        the kernel's numerics; cycles are real, correctness is not checked.
      - **Samsung** -> `CYCLES-ONLY`. `_run_samsung` reports cycles + stdout only
        (no output array); it computes at the GEMV design point.
      - **AiM** -> `CYCLES-ONLY`. ramulator2 is a memory-trace sim with no
        functional numerics.

    `sim unavailable` (cycles is None because the binary/slot is absent) is left
    to the caller to record as the environment skip; this helper assumes a real
    run produced `result`.
    """
    extra = getattr(result, "extra", {}) or {}
    cycles = getattr(result, "cycles", None)

    if backend == "upmem":
        bench = extra.get("benchmark")
        if bench == "GEMV" and cycles is not None:
            tol = tolerance_for(backend)
            return Verdict(
                PASS,
                f"uPIMulator GEMV host numeric check passed (W@x vs numpy ref, "
                f"host tol); recorded ref rtol={tol['rtol']:g}",
            )
        return Verdict(
            CYCLES_ONLY,
            f"upmem/{kernel}: routed to {bench or 'VA'} host slot -- the host "
            f"checks its own data, not this kernel; cycles real, numerics "
            f"unchecked",
        )

    if backend == "samsung_hbm_pim":
        return Verdict(
            CYCLES_ONLY,
            f"samsung/{kernel}: run path reports cycles only (no output array); "
            f"computed at the GEMV design point",
        )

    if backend == "aim":
        return Verdict(
            CYCLES_ONLY,
            f"aim/{kernel}: ramulator2 trace sim -- N/A (no functional numerics)",
        )

    return Verdict(
        CYCLES_ONLY, f"{backend}/{kernel}: no functional output surfaced"
    )
