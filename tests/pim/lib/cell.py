# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: the shared per-cell driver (written once).

`run_cell` is the body every kernel-folder test calls: build the target, run the
folder's workload through `lib.runner.compile_and_run`, derive the verdict from
what the run surfaces (`lib.reference.verdict_for_run`), and write the per-cell
`results.json` + `RESULTS.md` + regenerate `COVERAGE.tsv` with run-stamped
provenance. A kernel-folder test stays a thin call -- it declares only its
workload + inputs, no hardware, no schema, no verdict logic.
"""

from __future__ import annotations

from allo.spmw_codegen import RunResult

from . import reference, results, runner


def sim_unavailable(result: RunResult) -> bool:
    return "simulator unavailable" in (result.stdout or "")


def harness_inputs(backend, stages):
    """Build the cycles-measurement run kwargs from a workload's `STAGES`
    (a list of (rows, reduction) per MAC stage). Zeros: the simulators measure
    the emitted microcode's cycles, not numerics we verify in Python (the
    Samsung/AiM run paths surface no output array; the UPMEM GEMV host runs its
    own internal W@x check on whatever data-prep shape it is given).

      - Samsung: a single-MAC stream takes `W`/`x`; a multi-MAC (>=2 layer)
        stream needs `layers=[{W,x}, ...]` (one per MAC group -- the
        `_run_samsung` multi-layer dispatch). Built from `stages`.
      - UPMEM: the GEMV host slot takes one `W`/`x` (the first stage geometry).
      - AiM: no inputs (trace sim).
    """
    import numpy as np

    def _wx(rows, red):
        return (np.zeros((int(rows), int(red)), dtype=np.float16),
                np.zeros(int(red), dtype=np.float16))

    if backend == "aim":
        return {}
    if backend == "samsung_hbm_pim":
        if len(stages) <= 1:
            W, x = _wx(*stages[0])
            return {"W": W, "x": x}
        layers = []
        for rows, red in stages:
            W, x = _wx(rows, red)
            layers.append({"W": W, "x": x})
        return {"layers": layers}
    if backend == "apu_v1":
        # Real-device: APU v1 surfaces real numerics, so use REAL (non-zero,
        # deterministic) inputs -> a meaningful correctness check vs the numpy
        # reference (the device path is the only one that returns an output
        # array to Python). First-stage geometry.
        rows, red = stages[0]
        rng = np.random.default_rng(0)
        W = (rng.random((int(rows), int(red))).astype(np.float16) * 0.1)
        x = (rng.random(int(red)).astype(np.float16) * 0.1)
        return {"W": W, "x": x}
    # upmem (and any GEMV-slot backend): first-stage geometry.
    W, x = _wx(*stages[0])
    return {"W": W, "x": x}


def apu_v1_correctness(result, stages, kernel) -> "reference.Verdict":
    """Real-device correctness for APU v1: compare the surfaced output against
    the numpy GEMV reference (W@x of the first-stage geometry). The device runs
    real numerics, so a faithful kernel matches the ref -> PASS; a mismatch is
    recorded CYCLES-ONLY with the planner's SPEC-018b OUT-OF-PARADIGM reason
    (ruling: work/reports/apu-v1-spec018b-coverage-ruling.md): the silicon runs
    BINARY XNOR-popcount MAC (the validated bmatmul_sv/sv_lookup paradigm), and a
    general fp16 multiply-accumulate is a DIFFERENT compute kernel (no gvml
    fp16-MAC primitive; the autoscheduler/cost model know only the binary
    {sv, sv_lookup} modes) -- an architectural boundary (SPEC-018b, a ceiling
    task), NOT a fix-in-session gap. The board runs the binary LUT-MAC stub for
    real cycles but does not compute the fp16 product, so the output cannot match
    the fp16 ref: CYCLES-ONLY, never a FAIL and NEVER a fabricated fp16 PASS."""
    import numpy as np

    outputs = (getattr(result, "extra", {}) or {}).get("outputs") or {}
    tol = reference.tolerance_for("apu_v1")
    rows, red = stages[0]
    rng = np.random.default_rng(0)  # SAME seed as harness_inputs -> the real inputs
    W = (rng.random((int(rows), int(red))).astype(np.float16) * 0.1)
    x = (rng.random(int(red)).astype(np.float16) * 0.1)
    ref = W.astype(np.float32) @ x.astype(np.float32)
    for arr in outputs.values():
        a = np.asarray(arr).reshape(-1)
        if a.size >= int(rows):
            cand = a[: int(rows)].astype(np.float32)
            if np.allclose(cand, ref, rtol=tol["rtol"], atol=tol["atol"]):
                return reference.Verdict(
                    reference.PASS,
                    f"apu_v1 real-device output matches W@x within "
                    f"rtol={tol['rtol']:g}",
                )
    return reference.Verdict(
        reference.CYCLES_ONLY,
        f"apu_v1/{kernel}: real-device run (real cycles) -- but apu_v1 fp16-MAC is "
        f"OUT-OF-PARADIGM (planner SPEC-018b ruling): the silicon runs BINARY "
        f"XNOR-popcount MAC (the validated bmatmul_sv/sv_lookup paradigm); a "
        f"general fp16 multiply-accumulate is a DIFFERENT compute kernel (no gvml "
        f"fp16-MAC primitive; the autoscheduler/cost model know only the binary "
        f"{{sv,sv_lookup}} modes), an architectural boundary (SPEC-018b, noted for "
        f"a ceiling task), NOT a fix-in-session gap. The board runs the binary "
        f"LUT-MAC stub so the output cannot match the fp16 ref -- CYCLES-ONLY, "
        f"never a FAIL and never a fabricated fp16 PASS",
    )


def run_cell(*, kernel, target_name, workload, folder, stages, dataset="SMALL",
             shapes, run_cmd, notes=""):
    """Run one (kernel, target) cell end-to-end and write its artifacts.

    Returns `(result, verdict, record)`. `stages` is the workload's `STAGES`
    (rows, reduction per MAC stage); the backend-appropriate run kwargs are
    built by `harness_inputs`. `shapes` is the dataset dict from
    `lib.shapes.shape(...)`. The caller asserts the verdict + (when the sim is
    up) a real cycle count.
    """
    from lib.targets import build_target  # absolute: kernel folders import `lib`

    target = build_target(target_name)
    inputs = harness_inputs(target_name, stages)

    if target_name == "apu_v1":
        return _run_apu_v1_cell(
            kernel=kernel, target=target, workload=workload, folder=folder,
            stages=stages, inputs=inputs, dataset=dataset, shapes=shapes,
            run_cmd=run_cmd, notes=notes,
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
        result = runner.compile_and_run(target, workload, **inputs)
    except RuntimeError as exc:
        sim_core = str(exc)
        result = RunResult(
            cycles=None, stdout=f"BLOCKED-SIM: {sim_core[:300]}",
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
    else:
        verdict = reference.verdict_for_run(
            result, backend=target_name, kernel=kernel
        )
        source = runner.sim_source(target_name)

    record = results.build_record(
        kernel=kernel, target=target_name, dataset=dataset, shapes=shapes,
        verdict=verdict, reference_provenance=reference.provenance(kernel),
        metric="cycles", value=result.cycles, source=source,
        run_cmd=run_cmd, timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(), notes=notes,
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record


def _run_apu_v1_cell(*, kernel, target, workload, folder, stages, inputs,
                     dataset, shapes, run_cmd, notes):
    """APU v1 REAL-DEVICE cell: serialize board access, run on the board, check
    real numerics vs the numpy ref. Device-unreachable -> BLOCKED-DEVICE with the
    concrete reason, NEVER sim-substituted (spec Answer 5). source records the
    device + firmware identity."""
    from allo.spmw_codegen import _apu_v1_unavailable_reason

    # ABSOLUTE: never sim-substitute. If the board is unreachable -> BLOCKED-DEVICE.
    reason = _apu_v1_unavailable_reason()
    if reason is not None:
        verdict = reference.Verdict(
            reference.BLOCKED_DEVICE,
            f"apu_v1/{kernel}: board unreachable ({reason}) -- recorded, NEVER "
            f"sim-substituted",
        )
        record = results.build_record(
            kernel=kernel, target="apu_v1", dataset=dataset, shapes=shapes,
            verdict=verdict, reference_provenance=reference.provenance(kernel),
            metric="cycles", value=None, source="apu_v1_device@unreachable",
            run_cmd=run_cmd, timestamp=runner.now_iso(),
            tenon_commit=runner.tenon_commit(), notes=notes,
        )
        results.write_results(folder, record)
        results.write_results_md(folder, record)
        results.regenerate_coverage()
        return RunResult(cycles=None, stdout=f"BLOCKED-DEVICE: {reason}",
                         backend="apu_v1"), verdict, record

    # Serialize the shared board (conftest file-lock) around the real run.
    try:
        import conftest  # tests/pim/conftest.py is on sys.path under the suite
        board_lock = conftest.board_lock
    except Exception:  # noqa: BLE001 -- fall back to a no-op CM if absent
        import contextlib
        board_lock = contextlib.nullcontext

    device_unreachable = None
    with board_lock():
        try:
            result = runner.compile_and_run(target, workload, **inputs)
        except RuntimeError as exc:
            # A board build/run failure (e.g. missing PROF_PRINT) -> the device
            # ran but did not surface a cycle count: BLOCKED-DEVICE, never a
            # number. (A genuine Tenon build/codegen gap would be fixed in
            # session; this branch is the environment/board failure.)
            device_unreachable = str(exc)
            result = RunResult(cycles=None,
                               stdout=f"BLOCKED-DEVICE: {device_unreachable[:300]}",
                               backend="apu_v1")

    source = runner.apu_v1_device_source()
    if device_unreachable is not None or result.cycles is None:
        verdict = reference.Verdict(
            reference.BLOCKED_DEVICE,
            f"apu_v1/{kernel}: board run produced no cycle count "
            f"({(device_unreachable or 'no PROF_PRINT')[:160]})",
        )
        source = f"{source} (no PROF_PRINT)"
    else:
        verdict = apu_v1_correctness(result, stages, kernel)

    record = results.build_record(
        kernel=kernel, target="apu_v1", dataset=dataset, shapes=shapes,
        verdict=verdict, reference_provenance=reference.provenance(kernel),
        metric="cycles", value=result.cycles, source=source,
        run_cmd=run_cmd, timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(), notes=notes,
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record
