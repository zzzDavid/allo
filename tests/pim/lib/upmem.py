# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM execution/reporting after a leaf explicitly calls ``allo.compile``."""

from __future__ import annotations

import importlib.util
import os
import pathlib

import numpy as np

from . import reference, results, runner


_KNOWN_SOURCE_SEMANTICS = {
    "durbin": (
        "Repository semantics intentionally omit the canonical alpha/beta "
        "division; both kernel_durbin and durbin_np do so."
    ),
    "gramschmidt": (
        "Repository semantics use the squared column norm rather than sqrt; "
        "both kernel_gramschmidt and gramschmidt_np do so."
    ),
    "heat_3d": (
        "Repository semantics fuse B then A per lexicographic cell rather than "
        "using two whole-grid sweeps; kernel_heat_3d and heat_3d_np agree."
    ),
}


def load_leaf_workload(folder):
    """Load exactly the co-located canonical ``workload.py`` for one leaf."""
    path = pathlib.Path(folder) / "workload.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"_upmem_leaf_{path.parent.name}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_polybench_case(
    compiled,
    case,
    *,
    folder,
    run_cmd,
    notes,
    dataset="SMALL",
    seed=0,
):
    """Execute a complete canonical MLIR program and record PASS + cycles.

    Numerical correctness comes from the portable-C execution of the exact
    canonical MLIR function.  Performance is deliberately separate: the value
    is produced by the bound executable cost program over the retained UPMEM
    graph, not mislabeled as simulator measurement.
    """
    if compiled.target.name != "upmem":
        raise ValueError(f"UPMEM helper received target {compiled.target.name!r}")

    original = case.make_inputs(seed=seed)
    expected = case.run_reference(original)
    arguments = {name: value.copy() for name, value in original.items()}
    for result in case.results:
        if result.name not in arguments:
            arguments[result.name] = np.zeros(result.shape, dtype=result.dtype)

    run_result = compiled(**arguments)
    surfaced = dict((run_result.extra or {}).get("outputs", {}))
    tolerance = reference.tolerance_for("upmem")
    max_abs = 0.0
    for result in case.results:
        actual = np.asarray(surfaced.get(result.name, arguments[result.name]))
        wanted = np.asarray(expected[result.name])
        if actual.shape != wanted.shape:
            actual = actual.reshape(wanted.shape)
        error = np.abs(actual.astype(np.float64) - wanted.astype(np.float64))
        if error.size:
            max_abs = max(max_abs, float(np.max(error)))
        np.testing.assert_allclose(
            actual,
            wanted,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=f"UPMEM portable-C result mismatch for {case.name}.{result.name}",
        )

    estimate = compiled.estimate()
    if estimate.cycles <= 0:
        raise AssertionError(f"{case.name}: non-positive analytical cycles")
    launches = compiled.abi.launches
    if not launches or any(launch.num_dpus != 64 for launch in launches):
        raise AssertionError(f"{case.name}: UPMEM ABI does not use all 64 DPUs")
    if not compiled.abi_manifest.get("planned_orchestration"):
        raise AssertionError(f"{case.name}: conceptual execution plan was dropped")

    verdict = reference.Verdict(
        reference.PASS,
        f"canonical MLIR -> portable C matched repository NumPy reference for "
        f"{', '.join(expected)} (max_abs_err={max_abs:g}, "
        f"rtol={tolerance['rtol']:g}, atol={tolerance['atol']:g}); ABI packed "
        "and gathered a full 64-DPU rank",
    )
    source = (
        f"upmem_cost@{compiled.cost.fingerprint} (analytical MLIR operation "
        "graph); portable-C functional oracle"
    )
    semantic_note = _KNOWN_SOURCE_SEMANTICS.get(case.name)
    notes = (
        f"{notes} Tensor MRAM ownership and analytical DPU/tasklet fanout are "
        "derived from the compiled F2 LinearLayout."
    )
    if semantic_note:
        notes = f"{notes} {semantic_note}"
    record = results.build_record(
        kernel=case.name,
        target="upmem",
        dataset=dataset,
        shapes=case.dims,
        verdict=verdict,
        reference_provenance={
            "source": (
                f"examples/polybench/{case.module_name}.py::{case.reference_name}"
            ),
            "size_class": "SMALL_DATASET",
            "validated_by": "tests/pim/test_upmem_polybench_registry.py",
        },
        metric="cycles",
        value=estimate.cycles,
        source=source,
        run_cmd=run_cmd,
        timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(),
        notes=notes,
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    if os.environ.get("UPMEM_SKIP_COVERAGE") != "1":
        results.regenerate_coverage()
    return run_result, estimate, verdict, record
