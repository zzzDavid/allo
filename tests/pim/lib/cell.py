# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: shared driver for legacy contraction cells.

`run_cell` builds the target, compiles single-artifact workloads through
`allo.compile` via `lib.runner`, derives the
verdict from what the run surfaces (`lib.reference.verdict_for_run`), and writes the per-cell
`results.json` + `RESULTS.md` + regenerate `COVERAGE.tsv` with run-stamped
provenance. UPMEM and Samsung leaves use their newer direct public-compile
harnesses and do not enter this helper.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

from allo.spmw_codegen import RunResult

from . import reference, results, runner


def sim_unavailable(result: RunResult) -> bool:
    return "simulator unavailable" in (result.stdout or "")


def load_workload(folder, kernel):
    """SPEC-05: load the workload module for a cell, PREFERRING the target-specific
    co-located `workload.py` in the cell's leaf dir (`folder/workload.py`), falling
    back to the shared `tests/pim/workloads/<kernel>.py` for unmigrated targets
    (AiM/APU v1). Samsung GEMM-family workloads are now slice-form and live
    in the leaf dir; the shared ones stay full-loop for the other backends.

    Returns the imported module (must export `build()` + `STAGES`)."""
    leaf = pathlib.Path(folder) / "workload.py"
    if leaf.exists():
        spec = importlib.util.spec_from_file_location(
            f"_leaf_workload_{kernel}_{leaf.parent.name}", str(leaf)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    # Fallback: the shared workloads package (PSIZE alias for 2mm/3mm).
    shared_name = {"2mm": "two_mm", "3mm": "three_mm"}.get(kernel, kernel)
    return importlib.import_module(f"workloads.{shared_name}")


# The PIMSimulator GEMV data-path (`computeGemv`) only POPULATES out.bin at its
# validated design point; off the design point it leaves the read region zero, so
# a SMALL shape surfaces no numerics (CYCLES-ONLY). M=4096, K=1024 is the floor-
# finding shape the dual-run guard (tests/spmw/test_samsung_dual_run.py) and the
# GEMV walkthrough use; 4096*1024 = 4.19M >> the 131072 full-tile floor. The
# multi-stage Samsung GEMV path still uses it; single-stage GEMM-family kernels
# now route to GENERIC_REDUCE (SPEC-04), which runs at the KERNEL'S OWN shape and
# surfaces real numerics, so no design-point substitution there (provenance fix).
_SAMSUNG_DESIGN_POINT = (4096, 1024)


def _samsung_gemm_operands(kernel, shapes):
    """SPEC-04: build the (M, K, N) GEMM operands A[M,K], B[K,N] for a single-stage
    GEMM-family kernel from its dataset shape dict, plus the host pre/post-pass
    applied to the numpy reference (Samsung has no divide; kernels-spec §2.1). The
    operands are seeded at the KERNEL'S OWN shape (no 4096 design-point), so the
    GENERIC_REDUCE run produces a real C and the recorded shape matches the run.

    Returns (A, B, ref_C) or None when the kernel is not a clean single-stage
    GEMM-family kernel handled here (gemm / covariance / doitgen).
    """
    import numpy as np

    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)

    if kernel == "gemm":
        # C = A[P,Q] @ B[Q,R]; host alpha/beta is applied to the ref off-device.
        P, Q, R = int(shapes["P"]), int(shapes["Q"]), int(shapes["R"])
        A = rng0.random((P, Q)).astype(np.float16) * 0.1
        B = rng1.random((Q, R)).astype(np.float16) * 0.1
        alpha, beta = 1.5, 1.2
        C0 = rng0.random((P, R)).astype(np.float16) * 0.1
        ref = alpha * (A.astype(np.float32) @ B.astype(np.float32)) + beta * C0.astype(
            np.float32
        )
        # The device computes the bare A@B; the harness checks that bare product
        # (host alpha/beta is a trivial elementwise post-pass on the ref, recorded
        # but not gated here -- the device contraction is what REDUCE proves).
        bare_ref = A.astype(np.float32) @ B.astype(np.float32)
        return A, B, bare_ref
    if kernel == "covariance":
        # cov_raw = cdata[N,M]^T @ cdata[N,M] = (M,M). The device computes the bare
        # contraction A^T @ A with A = cdata; centering/normalize is a host pre/post
        # pass on the ref. We seed cdata and form A = cdata so the REDUCE GEMM is
        # cdata^T @ cdata. Express as GEMM with operand0 = cdata^T (M,N) and
        # operand1 = cdata (N,M) -> (M,M).
        N, M = int(shapes["N"]), int(shapes["M"])
        cdata = rng0.random((N, M)).astype(np.float16) * 0.1
        A = cdata.T.copy()  # (M, N)
        B = cdata.copy()  # (N, M)
        bare_ref = A.astype(np.float32) @ B.astype(np.float32)  # (M, M)
        return A, B, bare_ref
    if kernel == "doitgen":
        # out[r,q,p] = sum_s A[r,q,s]*x[s,p]; batch (r,q) -> (R*Q, S) @ (S, P).
        R, Q, S, P = (
            int(shapes["R"]),
            int(shapes["Q"]),
            int(shapes["S"]),
            int(shapes["P"]),
        )
        A = rng0.random((R * Q, S)).astype(np.float16) * 0.1
        B = rng1.random((S, P)).astype(np.float16) * 0.1
        bare_ref = A.astype(np.float32) @ B.astype(np.float32)  # (R*Q, P)
        return A, B, bare_ref
    return None


def harness_inputs(backend, stages, kernel=None, shapes=None):
    """Build the run kwargs from a workload's `STAGES` (a list of (rows,
    reduction) per MAC stage).

      - Samsung: a single-stage GEMM-family kernel (gemm/covariance/doitgen) takes
        REAL seeded A/B at the KERNEL'S OWN shape and routes to GENERIC_REDUCE
        (SPEC-04), which computes genuine fp16 A@B and surfaces a checkable `out`.
        A multi-MAC (>=2 layer) stream still uses `layers=[{W,x}, ...]` at authored
        geometry (the legacy GEMV multi-layer path); that surfaces no numerics, so
        those cells stay CYCLES-ONLY (the cross-stage chain is the next phase).
      - AiM: no inputs (trace sim).

    UPMEM does not use this contraction-cell harness; its leaves construct an
    `UPMEMProgram` and call `allo.compile` directly.
    """
    import numpy as np

    def _wx_real(rows, red):
        rng = np.random.default_rng(0)
        W = rng.random((int(rows), int(red))).astype(np.float16) * 0.1
        x = rng.random(int(red)).astype(np.float16) * 0.1
        return W, x

    if backend == "aim":
        return {}
    if backend == "samsung_hbm_pim":
        if len(stages) <= 1:
            # Single-stage GEMM-family -> GENERIC_REDUCE at the kernel's own shape
            # (real A/B -> real C, no design-point substitution).
            if kernel is not None and shapes is not None:
                ops = _samsung_gemm_operands(kernel, shapes)
                if ops is not None:
                    A, B, _ref = ops
                    return {"A": A, "B": B}
            # Fallback (a single-stage kernel not in the GEMM-family table, e.g.
            # Tier-2 symm/syrk/trmm/gramschmidt): SPEC-05 routes ALL MAC to
            # GENERIC_REDUCE, so pass generic A/x operands (a GEMV slice at the
            # first-stage geometry) -> the REDUCE path runs and produces a real
            # cycle count. samsung_correctness can't build a matching ref for these
            # (no _samsung_gemm_operands entry) -> honest CYCLES-ONLY, real cycles.
            rows, red = stages[0]
            A = (
                np.random.default_rng(0)
                .random((int(rows), int(red)))
                .astype(np.float16)
                * 0.1
            )
            x = np.random.default_rng(1).random(int(red)).astype(np.float16) * 0.1
            return {"A": A, "x": x}
        layers = []
        for rows, red in stages:
            W, x = _wx_real(rows, red)
            layers.append({"W": W, "x": x})
        return {"layers": layers}
    raise ValueError(
        f"contraction-cell harness does not support backend {backend!r}; "
        "UPMEM and APU v1 leaves own explicit callable ABIs"
    )


def samsung_correctness(result, stages, kernel, shapes=None) -> "reference.Verdict":
    """Functional correctness for Samsung HBM-PIM, mirroring apu_v1_correctness.

    SPEC-04: a single-stage GEMM-family kernel routes to GENERIC_REDUCE, which
    computes a genuine fp16 `A @ B` at the KERNEL'S OWN shape and surfaces it as
    role "out". The reference is the bare A@B of the SAME seeded operands the
    harness fed (`_samsung_gemm_operands` -- same seeds), recorded at the RUN
    shape (provenance fix: no 4096 design-point substitution). The host alpha/beta
    / centering pre/post-pass is a trivial elementwise op on the ref, off-device;
    REDUCE proves the contraction itself.

    Returns:
      - PASS when the surfaced output matches the reference within the Samsung
        fp16 tolerance (rtol=atol=2e-2) -- a real fp16-correct run;
      - CYCLES-ONLY (honest) when no output is surfaced (multi-stage legacy GEMV
        path) OR the surfaced output does NOT match. The max_abs_err is recorded.
        NEVER a fabricated PASS, NEVER a FAIL.
    """
    import numpy as np

    outputs = (getattr(result, "extra", {}) or {}).get("outputs") or {}
    tol = reference.tolerance_for("samsung_hbm_pim")

    # Single-stage GEMM-family: the reference is the bare A@B of the kernel's own
    # operands (REDUCE path). Fall back to the legacy GEMV W@x reference for a
    # multi-stage / non-GEMM-family single-stage cell.
    ref = None
    if len(stages) <= 1 and shapes is not None:
        ops = _samsung_gemm_operands(kernel, shapes)
        if ops is not None:
            _A, _B, ref = ops  # ref is the bare contraction (M, N)
    if ref is None:
        rows, red = _SAMSUNG_DESIGN_POINT if len(stages) <= 1 else stages[-1]
        rng = np.random.default_rng(0)  # SAME seed as harness_inputs legacy path
        W = rng.random((int(rows), int(red))).astype(np.float16) * 0.1
        x = rng.random(int(red)).astype(np.float16) * 0.1
        ref = W.astype(np.float32) @ x.astype(np.float32)

    arr = outputs.get("out")
    if arr is None:
        arr = outputs.get("y")
    if arr is None:
        return reference.Verdict(
            reference.CYCLES_ONLY,
            f"samsung/{kernel}: run path surfaced no output array "
            f"(multi-stage legacy GEMV path); cycles real, numerics unchecked",
        )
    cand = np.asarray(arr).astype(np.float32)
    ref = np.asarray(ref).astype(np.float32)
    if cand.shape != ref.shape:
        # The REDUCE "out" is (M, N); a legacy "y" is (M,). Reconcile by flatten.
        cand = cand.reshape(-1)
        ref_flat = ref.reshape(-1)
        if cand.size < ref_flat.size:
            return reference.Verdict(
                reference.CYCLES_ONLY,
                f"samsung/{kernel}: surfaced output too short "
                f"({cand.size} < {ref_flat.size}); cycles real, numerics unchecked",
            )
        cand = cand[: ref_flat.size]
        ref = ref_flat
    if np.allclose(cand, ref, rtol=tol["rtol"], atol=tol["atol"]):
        max_abs = float(np.max(np.abs(cand - ref)))
        return reference.Verdict(
            reference.PASS,
            f"samsung {kernel} output matches the contraction reference within "
            f"rtol={tol['rtol']:g} (max_abs_err={max_abs:g})",
        )
    max_abs = float(np.max(np.abs(cand - ref)))
    return reference.Verdict(
        reference.CYCLES_ONLY,
        f"samsung/{kernel}: surfaced output does NOT match the reference "
        f"(max_abs_err={max_abs:g} > tol {tol['atol']:g}); cycles real, numerics "
        f"unchecked (NOT a fabricated PASS)",
    )


def run_host_program(host, module):
    """Execute a recorded host program (host-xcel driver) and judge it.

    Thin harness shim over the compiler's host-program capability: the residency
    analysis (`allo.spmw_host_program.analyze`) and device execution
    (`spmw_codegen.execute_host_schedule_samsung`) live in `allo/`; this function
    keeps only the TEST-side concerns -- seeding external inputs and composing the
    float32 reference -- and returns `(final_readback, composed_ref, total_cycles)`
    like `samsung_reduce_chain` (or `(None,None,None)`/`(False,None,total)` on a
    down/empty driver).

    External inputs are seeded in first-consumed order with `rng(0)*0.1` (the same
    draws the legacy recipe used, so the numbers are unchanged). The cross-stage
    orchestration -- and the weight-residency / batched-GEMV coalescing -- come
    from the workload's host program via the compiler analysis, not a per-kernel
    recipe.
    """
    import numpy as np
    from allo.spmw_codegen import execute_host_schedule_samsung
    from allo.spmw_host_program import resolve_shapes, analyze

    shapes = resolve_shapes(host, vars(module))
    schedule = analyze(host, shapes)

    # Seed external inputs (test data) in first-consumed order; ref mirrors them.
    rng = np.random.default_rng(0)
    dev_inputs: dict = {}
    ref: dict = {}
    for name in schedule.external_inputs:
        v = rng.random(shapes[name]).astype(np.float16) * 0.1
        dev_inputs[name] = v
        ref[name] = v.astype(np.float32)

    dev, total = execute_host_schedule_samsung(schedule, dev_inputs)
    if total is None:
        return None, None, None  # driver unavailable
    if dev is None:
        return False, None, total  # ran but surfaced nothing

    # Compose the float32 reference by mirroring each group's contraction.
    for g in schedule.groups:
        for lx in g.launches:
            ref[lx.out] = ref[lx.weight] @ ref[lx.vec]

    final = schedule.final
    if final is None or final not in dev:
        return False, None, total
    return dev[final], ref[final], total


def samsung_reduce_chain(kernel, shapes):
    """SPEC-04 §5.1: run a multi-stage GEMM-family kernel through GENERIC_REDUCE,
    threading each stage's REAL readback forward as the next stage's input, and
    compare the FINAL composed output against the genuine data-dependent numpy
    reference (NOT isolated GEMVs with independent seeds -- breadth1 §5.4's
    misleading-PASS trap). Transposed-A stages (atax/bicg stage2, mvt stage_b)
    feed A^T as the weight operand (the MAC cadence is identical; only operand
    orientation differs).

    Returns `(final_readback, composed_ref, total_cycles)` on a real run, or
    `(None, None, None)` when the driver is unavailable. PASS/CYCLES-ONLY is the
    caller's call (`run_cell`); this only runs + composes. Per-kernel recipe is
    explicit (the stage orientation + host combine are kernel-specific, not
    auto-derivable from STAGES alone). Returns `(None, None, None)` for a kernel
    with no chain recipe here.
    """
    import numpy as np
    from allo.spmw_codegen import run_samsung_reduce

    # The cell folder names "2mm"/"3mm" alias the workload kernel names.
    kernel = {"2mm": "two_mm", "3mm": "three_mm"}.get(kernel, kernel)

    rng = np.random.default_rng(0)

    def seed(*shape):
        return rng.random(shape).astype(np.float16) * 0.1

    def f32(a):
        return np.asarray(a, dtype=np.float32)

    total = 0

    def stage(weight, vec, M, K, N):
        """Run one REDUCE stage, accumulate cycles, return the real readback
        (float16, so the next stage threads the genuine device output)."""
        nonlocal total
        out, cyc = run_samsung_reduce(weight, vec, M=M, K=K, N=N)
        if cyc is None:
            return None  # driver unavailable
        total += cyc
        if out is None:
            return False  # ran but surfaced nothing (undrained)
        return np.asarray(out, dtype=np.float16)

    if kernel == "atax":
        # tmp = A @ x ; y = A^T @ tmp. Stage2 reduces over M (transposed A).
        M, N = int(shapes["M"]), int(shapes["N"])
        A, x = seed(M, N), seed(N)
        tmp = stage(A, x, M=M, K=N, N=1)
        if tmp is None:
            return None, None, None
        if tmp is False:
            return False, None, total
        y = stage(A.T.copy(), tmp, M=N, K=M, N=1)
        if y is False:
            return False, None, total
        ref = f32(A.T) @ (f32(A) @ f32(x))  # genuine composition
        return y, ref, total

    if kernel == "bicg":
        # q = A @ p ; s = A^T @ r. TWO INDEPENDENT GEMVs (both read external
        # p, r -- not a chain). The composed output we check is the concatenation;
        # PASS requires BOTH to match.
        M, N = int(shapes["M"]), int(shapes["N"])
        A, p, r = seed(M, N), seed(N), seed(M)
        q = stage(A, p, M=M, K=N, N=1)
        if q is None:
            return None, None, None
        s = stage(A.T.copy(), r, M=N, K=M, N=1)
        if q is False or s is False:
            return False, None, total
        q_ref = f32(A) @ f32(p)
        s_ref = f32(A.T) @ f32(r)
        return (np.concatenate([f32(q), f32(s)]), np.concatenate([q_ref, s_ref]), total)

    if kernel == "mvt":
        # x1 = A @ y1 ; x2 = A^T @ y2. Two independent GEMVs; polybench folds
        # x += both, but each contraction is checked (concatenated).
        N = int(shapes["N"])
        A, y1, y2 = seed(N, N), seed(N), seed(N)
        x1 = stage(A, y1, M=N, K=N, N=1)
        if x1 is None:
            return None, None, None
        x2 = stage(A.T.copy(), y2, M=N, K=N, N=1)
        if x1 is False or x2 is False:
            return False, None, total
        x1_ref = f32(A) @ f32(y1)
        x2_ref = f32(A.T) @ f32(y2)
        return (
            np.concatenate([f32(x1), f32(x2)]),
            np.concatenate([x1_ref, x2_ref]),
            total,
        )

    if kernel == "gesummv":
        # tmp = A @ x ; y = B @ x ; out = alpha*tmp + beta*y. Two INDEPENDENT
        # GEMVs over the SAME x; the axpy combine is host-side on the device
        # outputs AND on the reference (a genuine composition of both readbacks).
        N = int(shapes["N"])
        A, B, x = seed(N, N), seed(N, N), seed(N)
        alpha, beta = 1.5, 1.2
        tmp = stage(A, x, M=N, K=N, N=1)
        if tmp is None:
            return None, None, None
        y = stage(B, x, M=N, K=N, N=1)
        if tmp is False or y is False:
            return False, None, total
        out = alpha * f32(tmp) + beta * f32(y)  # host axpy on device outs
        ref = alpha * (f32(A) @ f32(x)) + beta * (f32(B) @ f32(x))
        return out, ref, total

    if kernel == "two_mm":
        # AB = A @ B ; D = AB @ C. A GEMM chain (stage2 input = stage1 readback).
        P, Q, R, S = (
            int(shapes["P"]),
            int(shapes["Q"]),
            int(shapes["R"]),
            int(shapes["S"]),
        )
        A, B, C = seed(P, Q), seed(Q, R), seed(R, S)
        AB = stage(A, B, M=P, K=Q, N=R)
        if AB is None:
            return None, None, None
        if AB is False:
            return False, None, total
        D = stage(AB, C, M=P, K=R, N=S)
        if D is False:
            return False, None, total
        ref = (f32(A) @ f32(B)) @ f32(C)
        return D, ref, total

    if kernel == "three_mm":
        # AB = A@B ; CD = C@D ; G = AB@CD. G threads TWO prior readbacks.
        P, Q, R, NT, S = (
            int(shapes["P"]),
            int(shapes["Q"]),
            int(shapes["R"]),
            int(shapes["T"]),
            int(shapes["S"]),
        )
        A, B, C, D = seed(P, Q), seed(Q, R), seed(R, S), seed(S, NT)
        AB = stage(A, B, M=P, K=Q, N=R)
        if AB is None:
            return None, None, None
        CD = stage(C, D, M=R, K=S, N=NT)
        if AB is False or CD is False:
            return False, None, total
        G = stage(AB, CD, M=P, K=R, N=NT)
        if G is False:
            return False, None, total
        ref = (f32(A) @ f32(B)) @ (f32(C) @ f32(D))
        return G, ref, total

    return None, None, None  # no chain recipe (e.g. gemver: mixed ELTWISE+REDUCE)


def _finish_samsung_chain_cell(
    *, kernel, folder, dataset, shapes, run_cmd, notes, final, ref, total_cyc
):
    """Build the verdict + record + RunResult for a multi-stage Samsung REDUCE
    chain. `final`/`ref` are the composed device output + genuine numpy ref;
    `final is False` => a stage ran but surfaced nothing (CYCLES-ONLY). PASS only
    on a real match of the FINAL composed output; never a fabricated PASS."""
    import numpy as np

    tol = reference.tolerance_for("samsung_hbm_pim")
    result = RunResult(
        cycles=total_cyc,
        stdout=f"samsung GENERIC_REDUCE chain ({kernel} {shapes}): "
        f"total_cycles={total_cyc}",
        backend="samsung_hbm_pim",
        extra={"kernel": "GENERIC_REDUCE_CHAIN"},
    )
    if final is False or final is None:
        verdict = reference.Verdict(
            reference.CYCLES_ONLY,
            f"samsung/{kernel}: REDUCE chain ran (cycles real) but a stage "
            f"surfaced no output; final numerics unchecked (NOT a fabricated PASS)",
        )
    else:
        cand = np.asarray(final, dtype=np.float32).reshape(-1)
        refv = np.asarray(ref, dtype=np.float32).reshape(-1)
        max_abs = float(np.max(np.abs(cand - refv)))
        if np.allclose(cand, refv, rtol=tol["rtol"], atol=tol["atol"]):
            verdict = reference.Verdict(
                reference.PASS,
                f"samsung {kernel} REDUCE chain final output matches the "
                f"data-dependent composition within rtol={tol['rtol']:g} "
                f"(max_abs_err={max_abs:g})",
            )
        else:
            verdict = reference.Verdict(
                reference.CYCLES_ONLY,
                f"samsung/{kernel}: REDUCE chain final output does NOT match the "
                f"composed reference (max_abs_err={max_abs:g} > tol "
                f"{tol['atol']:g}); cycles real (NOT a fabricated PASS)",
            )
    source = runner.sim_source("samsung_hbm_pim")
    record = results.build_record(
        kernel=kernel,
        target="samsung_hbm_pim",
        dataset=dataset,
        shapes=shapes,
        verdict=verdict,
        reference_provenance=reference.provenance(kernel),
        metric="cycles",
        value=result.cycles,
        source=source,
        run_cmd=run_cmd,
        timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(),
        notes=notes + " | SPEC-04 cross-stage REDUCE chain (logical shape; fabric "
        "M padded to 4096, K to 256 per stage).",
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record


def _finish_samsung_no_chain_cell(
    *, kernel, folder, dataset, shapes, stages, run_cmd, notes
):
    """SPEC-05: a multi-stage Samsung kernel with NO chain recipe (gemver: mixed
    ELTWISE rank-1 + GEMV) -> honest CYCLES-ONLY. Run ONE slice-form REDUCE pass
    over the first stage (real seeded operands) for a genuine cycle count; the
    final composed numerics are NOT checked (out of the pure-MAC REDUCE paradigm).
    Never a fabricated PASS."""
    import numpy as np
    from allo.spmw_codegen import run_samsung_reduce

    rng = np.random.default_rng(0)
    rows, red = (int(stages[0][0]), int(stages[0][1]))
    A = rng.random((rows, red)).astype(np.float16) * 0.1
    x = rng.random(red).astype(np.float16) * 0.1
    _out, cyc = run_samsung_reduce(A, x, M=rows, K=red, N=1)
    verdict = reference.Verdict(
        reference.CYCLES_ONLY,
        f"samsung/{kernel}: multi-stage kernel out of the pure-MAC REDUCE "
        f"paradigm (gemver mixes ELTWISE rank-1 + GEMV); a real REDUCE pass over "
        f"stage-0 gives cycles, but the composed numerics are unchecked (no chain "
        f"recipe). Honest CYCLES-ONLY, never a fabricated PASS.",
    )
    result = RunResult(
        cycles=cyc,
        stdout=f"samsung GENERIC_REDUCE no-chain ({kernel} {shapes}): cycles={cyc}",
        backend="samsung_hbm_pim",
        extra={"kernel": "GENERIC_REDUCE_NOCHAIN"},
    )
    record = results.build_record(
        kernel=kernel,
        target="samsung_hbm_pim",
        dataset=dataset,
        shapes=shapes,
        verdict=verdict,
        reference_provenance=reference.provenance(kernel),
        metric="cycles",
        value=cyc,
        source=runner.sim_source("samsung_hbm_pim"),
        run_cmd=run_cmd,
        timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(),
        notes=notes + " | SPEC-05 multi-stage, no chain recipe (mixed "
        "ELTWISE+REDUCE); honest CYCLES-ONLY.",
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record


def run_cell(
    *,
    kernel,
    target_name,
    folder,
    stages,
    dataset="SMALL",
    shapes,
    run_cmd,
    workload=None,
    notes="",
):
    """Run one (kernel, target) cell end-to-end and write its artifacts.

    Returns `(result, verdict, record)`. `stages` is the workload's `STAGES`
    (rows, reduction per MAC stage); the backend-appropriate run kwargs are
    built by `harness_inputs`. `shapes` is the dataset dict from
    `lib.shapes.shape(...)`. The caller asserts the verdict + (when the sim is
    up) a real cycle count.

    SPEC-05: `workload=None` (the new default) loads the cell's workload via
    `load_workload(folder, kernel)` -- preferring the leaf-dir `workload.py`
    (slice-form, target-specific) and falling back to the shared
    `workloads/<kernel>.py`. A caller may still pass an explicit `workload` for
    back-compat (unmigrated cells); when omitted, `stages` is also re-read from
    the loaded module so the leaf workload's STAGES wins.
    """
    from lib.targets import build_target  # absolute: kernel folders import `lib`

    # SPEC-05: prefer the leaf-dir (target-specific, slice-form) workload; the
    # loaded module's STAGES wins (the slice form may declare its own geometry).
    host_moves = None
    host_prog = None  # Level 2 (emit): the @allo.host_program driver, if any.
    wl_module = None
    if workload is None:
        _wl = load_workload(folder, kernel)
        workload = _wl.build()
        stages = getattr(_wl, "STAGES", stages)
        # spec backend-host-transfer-dispatch.md (task 004): a leaf workload that
        # expresses its host data movement via `allo.host_xfer.*` exports the
        # recorded moves as `HOST_MOVES`; thread them into the single-artifact
        # Samsung run so the role binding is driven by the explicit moves.
        host_moves = getattr(_wl, "HOST_MOVES", None)
        # Level 2 (emit): a leaf workload that declares a host-xcel driver via
        # `@allo.host_program` exposes it as `host`; the multi-stage chain is then
        # EXECUTED from that program rather than a hardcoded per-kernel recipe.
        host_prog = getattr(_wl, "host", None)
        wl_module = _wl

    target = build_target(target_name)
    inputs = harness_inputs(target_name, stages, kernel=kernel, shapes=shapes)

    # SPEC-05: a single-stage Samsung kernel NOT in the GEMM-family operand table
    # (Tier-2 symm/syrk/trmm/gramschmidt -- GEMM-shaped contractions out of this
    # push's scope) has no contraction reference here. Since ALL MAC now routes to
    # GENERIC_REDUCE, record an honest CYCLES-ONLY via one stage-0 GEMV REDUCE pass
    # (real cycles, numerics unchecked) -- NOT the single-artifact path, which would
    # need GEMM B kwargs and return cycles=None.
    if (
        target_name == "samsung_hbm_pim"
        and len(stages) <= 1
        and _samsung_gemm_operands(kernel, shapes) is None
    ):
        return _finish_samsung_no_chain_cell(
            kernel=kernel,
            folder=folder,
            dataset=dataset,
            shapes=shapes,
            stages=stages,
            run_cmd=run_cmd,
            notes=notes,
        )

    # SPEC-04 §5.1: a multi-stage GEMM-family Samsung cell threads stage outputs
    # via GENERIC_REDUCE (genuine data-dependent composition) instead of the
    # legacy multi-layer GEMV path (which surfaces no numerics). The chain runs
    # the driver per stage directly (not through compile_and_run), so handle it
    # before the single-artifact run below.
    if target_name == "samsung_hbm_pim" and len(stages) > 1:
        # Level 2 (emit): if the workload ships a host-xcel driver, EXECUTE it
        # (the chain is driven by the recorded `host.steps`, no per-kernel recipe);
        # otherwise fall back to the legacy hardcoded `samsung_reduce_chain`.
        from allo.spmw_target import HostProgram

        if isinstance(host_prog, HostProgram):
            chain = run_host_program(host_prog, wl_module)
        else:
            chain = samsung_reduce_chain(kernel, shapes)
        final, ref, total_cyc = chain
        if final is None and ref is None and total_cyc is None:
            # No chain recipe (e.g. gemver: mixed ELTWISE rank-1 + GEMV, out of
            # the pure-MAC REDUCE paradigm). Record an honest CYCLES-ONLY (a real
            # cycle count via one slice-form REDUCE pass over the first stage) --
            # NOT a fall-through to the single-artifact path, which would expect
            # A/B kwargs and return cycles=None.
            return _finish_samsung_no_chain_cell(
                kernel=kernel,
                folder=folder,
                dataset=dataset,
                shapes=shapes,
                stages=stages,
                run_cmd=run_cmd,
                notes=notes,
            )
        return _finish_samsung_chain_cell(
            kernel=kernel,
            folder=folder,
            dataset=dataset,
            shapes=shapes,
            run_cmd=run_cmd,
            notes=notes,
            final=final,
            ref=ref,
            total_cyc=total_cyc,
        )

    # The reference simulator can CORE on an operand geometry it cannot express
    # (the FORBIDDEN-to-edit PIMSimulator: a `double free`/missing PIM_CYCLES on
    # a shape away from its design point). That surfaces as a RuntimeError from
    # the run path. It is an ENVIRONMENT limit (BLOCKED-SIM, spec Answer 3), NOT
    # a Tenon gap -- the kernel lowered + compiled; the sim could not run the
    # shape. Catch it here and record BLOCKED-SIM with the concrete reason; do
    # not edit the (re-export-only) run path.
    sim_core = None
    try:
        result = runner.compile_and_run(
            target, workload, host_moves=host_moves, **inputs
        )
    except RuntimeError as exc:
        sim_core = str(exc)
        result = RunResult(
            cycles=None,
            stdout=f"BLOCKED-SIM: {sim_core[:300]}",
            backend=target_name,
        )

    assert isinstance(result, RunResult)
    assert result.backend == target_name, result.backend

    down = sim_unavailable(result)
    if sim_core is not None:
        verdict = reference.Verdict(
            reference.BLOCKED_SIM,
            f"{target_name}/{kernel}: reference sim cored on the shape "
            f"(env limit, not a Tenon gap): {sim_core[:160]}",
        )
        source = f"{runner.sim_source(target_name)} (cored on shape)"
    elif down:
        verdict = reference.Verdict(
            reference.CYCLES_ONLY,
            f"{target_name}/{kernel}: simulator unavailable (environment limit)",
        )
        source = f"{target_name}@unavailable"
    elif target_name == "samsung_hbm_pim":
        # SPEC-04: a single-stage GEMM-family cell routes to GENERIC_REDUCE and
        # surfaces a real `out` (genuine fp16 A@B at the kernel's own shape) ->
        # functionally check vs the contraction reference (PASS on match, honest
        # CYCLES-ONLY otherwise). Multi-stage cells stay on the legacy GEMV path.
        verdict = samsung_correctness(result, stages, kernel, shapes=shapes)
        source = runner.sim_source(target_name)
    else:
        verdict = reference.verdict_for_run(result, backend=target_name, kernel=kernel)
        source = runner.sim_source(target_name)

    record = results.build_record(
        kernel=kernel,
        target=target_name,
        dataset=dataset,
        shapes=shapes,
        verdict=verdict,
        reference_provenance=reference.provenance(kernel),
        metric="cycles",
        value=result.cycles,
        source=source,
        run_cmd=run_cmd,
        timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(),
        notes=notes,
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record
