# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the paper's Gemini-I (APU v1) kernels through the shared pipeline.

For each kernel of ``tables/eval-apuv1.tex`` the workload of the evaluation
assets (``$TENON_ARTIFACTS/apuv1/tenon/<case>/workload.py``, an
evaluation-only tree distributed separately from the release package) is
compiled with ``allo.compile`` under the default shared selection, its
selected plan or native route, candidate estimates and generated device
source are recorded, and the program is launched on the board ``--repeats``
times under the shared board lock with the archived inputs
(``inputs/<case>/<name>.uint16.bin``); every output is compared with the
archived oracle before any timing is reported.  Options force a plan
(``--layout``), the paper's own forced plan being labelled separately from
the default selection, or perturb one cost constant (``--perturb
NAME=FACTOR`` on ``allo.pim.costs.apu_v1``) so the perturbation's effect on
selection, generated source and launch is traced.  Results are a JSON
record per case (``--out``); nothing here is read by the compiler.

Usage::

    python -m benchmarks.apu_v1.paper_rows --out rows.json [--cases ...]
        [--repeats 5] [--backend device|virtual|functional]
        [--layout PLAN] [--perturb CONST=FACTOR]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

CASES = ("matrix_multiply", "kmeans", "histogram", "linear_regression", "word_count")

# Paper table rows (tables/eval-apuv1.tex): expert Phoenix cycles and the
# paper's reported Tenon median device cycles.  Comparison data, never
# compiler inputs.
PAPER = {
    "matrix_multiply": (8_784_431, 8_764_900),
    "kmeans": (641_471, 624_677),
    "histogram": (167_384, 166_783),
    "linear_regression": (95_785, 94_618.5),
    "word_count": (175_397.5, 171_988),
}


def _artifacts_root(argument):
    if argument:
        return Path(argument)
    env = os.environ.get("TENON_ARTIFACTS")
    if env:
        return Path(env)
    from allo.spmw_simenv import tenon_artifacts_root

    return tenon_artifacts_root()


def _load_workload(root: Path, case: str):
    path = root / "apuv1" / "tenon" / case / "workload.py"
    spec = importlib.util.spec_from_file_location(f"paper_apu_v1_{case}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _perturb(spec_text):
    """``NAME=FACTOR``: scale one calibration constant of the APU v1 cost
    module before compiling; returns (name, old, new)."""
    from allo.pim.costs import apu_v1 as apu_v1_costs

    name, factor = spec_text.split("=")
    old = getattr(apu_v1_costs, name)
    new = type(old)(old * float(factor)) if isinstance(old, (int, float)) else old
    setattr(apu_v1_costs, name, new)
    return name, old, new


def measure_case(case, root, *, backend, repeats, layout=None, label=None):
    import numpy as np

    import allo
    from allo.pim.costs import apu_v1_cost
    from allo.pim.targets import build_apu_v1_target

    workload = _load_workload(root, case)
    target = build_apu_v1_target()
    kwargs = {}
    if layout is not None:
        kwargs["layout"] = layout
    started = time.perf_counter()
    compiled = allo.compile(
        workload,
        target,
        apu_v1_cost,
        backend=None if backend == "device" else backend,
        **kwargs,
    )
    compile_seconds = time.perf_counter() - started
    record = {
        "case": case,
        "label": label or ("forced:" + layout if layout else "default_selection"),
        "backend": backend,
        "compile_seconds": round(compile_seconds, 2),
        "paper_expert_cycles": PAPER[case][0],
        "paper_tenon_cycles": PAPER[case][1],
        "cost_fingerprint": apu_v1_cost.bind(target).fingerprint,
    }
    plan = getattr(compiled, "selected_plan", None)
    if plan is not None:
        record["selected_plan"] = plan.name
        record["candidate_estimates"] = {
            e.plan.name: int(e.cycles) for e in compiled.candidate_estimates
        }
        decision = getattr(compiled, "tile_decision", None)
        record["plan_selection"] = (
            decision.get("plan_selection") if isinstance(decision, dict) else None
        )
        realization = getattr(compiled, "realization", None)
        source = realization.device_source() if realization is not None else None
    else:
        route = getattr(compiled, "route", None)
        record["native_vector_route"] = route
        source = getattr(compiled, "device_source", None)
    record["device_source_sha256"] = _sha256(source) if source else None
    estimate = compiled.estimate() if hasattr(compiled, "estimate") else None
    record["estimate_cycles"] = (
        int(estimate.cycles) if estimate is not None and estimate.cycles else None
    )

    inputs_dir = root / "apuv1" / "inputs" / case
    manifest = json.loads((inputs_dir / "manifest.json").read_text())
    arrays = manifest["arrays"]
    oracle_spec = arrays["oracle"]
    oracle = np.fromfile(inputs_dir / oracle_spec["file"], dtype=oracle_spec["dtype"])
    import inspect

    kernel = getattr(workload, "phases", None)
    if kernel is not None:
        parameters = list(inspect.signature(kernel[0].kernel).parameters)
        returned = tuple(kernel[0].result_names)
    else:
        parameters = list(inspect.signature(workload).parameters)
        returned = ()
    # Inputs come from the archived arrays by parameter name; an in/out
    # parameter without an archive (the matrix multiply's result) starts as
    # zeros of the oracle's shape and is read back in place.
    arguments, in_place_outputs = {}, []
    for name in parameters:
        if name in arrays:
            spec = arrays[name]
            arguments[name] = np.fromfile(
                inputs_dir / spec["file"], dtype=spec["dtype"]
            ).reshape(spec["shape"])
        else:
            arguments[name] = np.zeros(oracle_spec["shape"], dtype=oracle_spec["dtype"])
            in_place_outputs.append(name)
    record["inputs"] = {
        name: hashlib.sha256(value.tobytes()).hexdigest()[:16]
        for name, value in arguments.items()
        if name not in in_place_outputs
    }
    record["outputs"] = list(returned) + in_place_outputs
    if backend == "virtual":
        record["cycles"] = []
        return record

    from allo.pim.apu_v1_program import apu_v1_board_lock

    cycles, outputs_match, kinds = [], [], set()
    with apu_v1_board_lock() as lock_path:
        record["board_lock"] = str(lock_path)
        for _ in range(repeats):
            fresh = {name: value.copy() for name, value in arguments.items()}
            if kernel is not None:
                result = compiled(**fresh)
            else:
                result = compiled(*(fresh[name] for name in parameters))
            kinds.add(result.kind)
            outputs = dict(result.extra.get("outputs", {}) or {})
            for name in in_place_outputs:
                outputs.setdefault(name, fresh[name])
            produced = [np.asarray(v).reshape(-1) for v in outputs.values()]
            match = bool(produced) and all(
                p.size == oracle.size and np.array_equal(p.astype(oracle.dtype), oracle)
                for p in produced
            )
            outputs_match.append(match)
            cycles.append(result.cycles)
            record.setdefault("run_extra", {}).update(
                {
                    k: v
                    for k, v in result.extra.items()
                    if k in ("build_mode", "native_vector_route", "scalar_arc", "plan")
                }
            )
    record["result_kind"] = sorted(kinds)
    record["cycles"] = cycles
    record["outputs_match_oracle"] = outputs_match
    if cycles and all(c is not None for c in cycles):
        record["median_cycles"] = statistics.median(cycles)
        record["min_cycles"], record["max_cycles"] = min(cycles), max(cycles)
        record["ratio_to_paper_tenon"] = record["median_cycles"] / PAPER[case][1]
        record["ratio_to_paper_expert"] = record["median_cycles"] / PAPER[case][0]
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts")
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backend", default="device", choices=("device", "virtual", "functional")
    )
    parser.add_argument("--layout", help="force a plan (matrix_multiply)")
    parser.add_argument("--label", help="label for the record (default from --layout)")
    parser.add_argument("--perturb", help="CONST=FACTOR on allo.pim.costs.apu_v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import allo

    root = _artifacts_root(args.artifacts)
    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "allo": allo.__file__,
        "artifacts": str(root),
        "perturbation": None,
        "records": [],
    }
    if args.perturb:
        name, old, new = _perturb(args.perturb)
        session["perturbation"] = {"constant": name, "stock": old, "perturbed": new}
    for case in args.cases:
        try:
            record = measure_case(
                case,
                root,
                backend=args.backend,
                repeats=args.repeats,
                layout=args.layout,
                label=args.label,
            )
        except Exception as error:  # a failed case is evidence too
            record = {"case": case, "error": f"{type(error).__name__}: {error}"}
        session["records"].append(record)
        print(json.dumps(record, default=str), flush=True)
    Path(args.out).write_text(json.dumps(session, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
