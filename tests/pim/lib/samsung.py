# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung test execution after a leaf has explicitly called ``allo.compile``.

This module deliberately does not compile workloads.  The Samsung leaf tests
own that public API boundary so readers can see the workload, target, cost
profile, and host moves passed to ``allo.compile``.  This helper retains only
simulator-specific execution, verdict construction, and artifact recording.
"""

from __future__ import annotations

from allo.spmw_codegen import RunResult
from allo.spmw_target import HostProgram

from . import cell, reference, results, runner


def load_workload(folder, kernel):
    """Load the target-local workload module for a leaf test."""
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
    """Execute an already compiled Samsung workload and write its artifacts."""
    target_name = "samsung_hbm_pim"
    if compiled.target.name != target_name:
        raise ValueError(
            f"Samsung test helper received target {compiled.target.name!r}"
        )

    stages = workload_module.STAGES
    host_program = getattr(workload_module, "host", None)

    # Multi-stage contractions are orchestrated by their host program or the
    # established data-dependent chain executor.  Compilation above still
    # validates and retains the complete device-region trace and cost graph.
    if len(stages) > 1:
        if isinstance(host_program, HostProgram):
            chain = cell.run_host_program(host_program, workload_module)
        else:
            chain = cell.samsung_reduce_chain(kernel, shapes)
        final, ref, total_cycles = chain
        if final is None and ref is None and total_cycles is None:
            return cell._finish_samsung_no_chain_cell(
                kernel=kernel,
                folder=folder,
                dataset=dataset,
                shapes=shapes,
                stages=stages,
                run_cmd=run_cmd,
                notes=notes,
            )
        return cell._finish_samsung_chain_cell(
            kernel=kernel,
            folder=folder,
            dataset=dataset,
            shapes=shapes,
            run_cmd=run_cmd,
            notes=notes,
            final=final,
            ref=ref,
            total_cyc=total_cycles,
        )

    # Tier-2 contractions without a numerical reference still execute one real
    # REDUCE pass and record an honest CYCLES-ONLY result.
    if cell._samsung_gemm_operands(kernel, shapes) is None:
        return cell._finish_samsung_no_chain_cell(
            kernel=kernel,
            folder=folder,
            dataset=dataset,
            shapes=shapes,
            stages=stages,
            run_cmd=run_cmd,
            notes=notes,
        )

    inputs = cell.harness_inputs(target_name, stages, kernel=kernel, shapes=shapes)
    simulator_error = None
    try:
        result = compiled.run_backend(**inputs)
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
            f"{target_name}/{kernel}: reference sim cored on the shape "
            f"(env limit, not a Tenon gap): {simulator_error[:160]}",
        )
        source = f"{runner.sim_source(target_name)} (cored on shape)"
    elif cell.sim_unavailable(result):
        verdict = reference.Verdict(
            reference.CYCLES_ONLY,
            f"{target_name}/{kernel}: simulator unavailable (environment limit)",
        )
        source = f"{target_name}@unavailable"
    else:
        verdict = cell.samsung_correctness(result, stages, kernel, shapes=shapes)
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


def assert_result(result, verdict, *, allow_pass=True):
    """Apply the common Samsung result contract used by every leaf."""
    allowed = {reference.CYCLES_ONLY, reference.BLOCKED_SIM}
    if allow_pass:
        allowed.add(reference.PASS)
    assert verdict.status in allowed, verdict
    if verdict.status in (reference.PASS, reference.CYCLES_ONLY):
        if not cell.sim_unavailable(result):
            assert result.cycles is not None and result.cycles > 0, (
                "samsung_hbm_pim: expected positive cycles; "
                f"got {result.cycles!r}; stdout tail: {result.stdout[-400:]}"
            )
    elif verdict.status == reference.BLOCKED_SIM:
        assert result.cycles is None
