# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM execution/reporting after a leaf explicitly calls ``allo.compile``."""

from __future__ import annotations

from allo.spmw_codegen import RunResult

from . import cell, reference, results, runner


def load_workload(folder, kernel):
    """Load the AiM-local SPMW workload module for a leaf test."""
    return cell.load_workload(folder, kernel)


def run_compiled(
    compiled,
    workload_module,
    *,
    kernel,
    folder,
    shapes,
    run_cmd,
    notes,
    dataset="SMALL",
):
    """Execute an already compiled AiM trace and rewrite result artifacts."""
    target_name = "aim"
    if compiled.target.name != target_name:
        raise ValueError(f"AiM helper received target {compiled.target.name!r}")

    simulator_error = None
    try:
        result = compiled.run_backend()
    except RuntimeError as exc:
        simulator_error = str(exc)
        result = RunResult(
            cycles=None,
            stdout=f"BLOCKED-SIM: {simulator_error[:300]}",
            backend=target_name,
        )

    if simulator_error is not None:
        verdict = reference.Verdict(
            reference.BLOCKED_SIM,
            f"{target_name}/{kernel}: simulator failed: {simulator_error[:160]}",
        )
        source = f"{runner.sim_source(target_name)} (failed on trace)"
    elif cell.sim_unavailable(result):
        verdict = reference.Verdict(
            reference.CYCLES_ONLY,
            f"{target_name}/{kernel}: simulator unavailable (environment limit)",
        )
        source = f"{target_name}@unavailable"
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


def assert_result(result, verdict):
    """Apply the common AiM trace-simulator result contract."""
    assert verdict.status in (reference.CYCLES_ONLY, reference.BLOCKED_SIM), verdict
    if verdict.status == reference.CYCLES_ONLY and not cell.sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"aim: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
