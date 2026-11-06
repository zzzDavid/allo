# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace Gemini-II selection decisions to real-card launches (release gate G9).

Two controlled experiments on ``zhang-x4``'s card, both through
``allo.compile`` and the shared pipeline:

* **cost perturbation**: the 64x128x33 INT4 GEMM has two legal plans; the
  calibrated tick model selects the reduction split (``single/k2``) and a
  model with row sweeps priced 20x (``APUG2_TYPED_TICK_MODEL.scaled(row_bits=
  20.0)``) selects output tiling (``single/k1``).  Both programs run on the
  card with the same inputs; outputs must equal the int64 reference exactly.
  Prediction stated before running: the stock selection is at least as fast
  as the perturbed one in measured device ticks.
* **forced alternative**: BiCG compiled under default selection (``fused/k1``,
  one launch) and with ``layout="separate/k1"`` (two launches); same inputs,
  exact outputs, per-plan ticks recorded.

Each record carries the plan name, the estimate, the measured ticks per
pipeline body (``repetitions`` bodies; the reproduction statistic is the
raw repeated-batch ticks divided by the repetitions, fractions kept), the
layout and program fingerprints, the packing attestation, every artifact
hash and the host timings of every launch.  A failed launch, an inexact
output or a violated invariant exits nonzero with the record preserved.
Results are evidence; nothing here is read by the compiler.

Usage::

    python -m benchmarks.apu_g2.selection_traces --out traces.json [--seed N]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

import numpy as np

import allo
from allo.ir.types import int4, int15
from allo.pim.apu_g2_pipeline import APUG2_TYPED_TICK_MODEL
from allo.pim.costs.apu_g2 import apu_g2_cost, build_apu_g2_cost
from allo.pim.targets import build_apu_g2_target

from .workloads import PAPER_KERNELS

REPETITIONS = 20


def gemm_n33(A: int4[64, 128], B: int4[128, 33], C: int15[64, 33]):
    for i in range(64):
        for j in range(33):
            acc: int15 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def _launch_records(result):
    """Every launch's provenance as the runtime exposes it: fingerprints,
    the packing attestation, every artifact hash, raw tick counters and
    host timings."""
    return [
        {
            "index": launch.get("index"),
            "layout_fingerprint": launch.get("layout_fingerprint"),
            "program_fingerprint": launch.get("program_fingerprint"),
            "packing_attestation": launch.get("packing_attestation"),
            "artifact_sha256": launch.get("artifact_sha256"),
            "raw_ticks": launch.get("raw_ticks"),
            "final_pipeline_ticks": launch.get("final_pipeline_ticks"),
            "host_timings_us": launch.get("host_timings_us"),
            "cycles": launch.get("cycles"),
        }
        for launch in result.extra.get("launches", ())
    ]


def amortized_ticks(result, repetitions) -> float | None:
    """The reproduction statistic: the repeated batch's raw device ticks of
    every launch (``raw_ticks["pipeline"]``, ``repetitions`` bodies) divided
    by the repetitions and summed over launches; fractional ticks are kept
    (``result.cycles`` rounds per launch)."""
    launches = list(result.extra.get("launches", ()) or ())
    if not launches:
        return None  # no launch records is not a zero-tick program
    total = 0.0
    for launch in launches:
        ticks = (launch.get("raw_ticks") or {}).get("pipeline")
        if (
            ticks is None
            or isinstance(ticks, bool)
            or not isinstance(ticks, (int, float))
            or not math.isfinite(ticks)
            or ticks <= 0
        ):
            return None
        total += float(ticks) / repetitions
    return total


def launch_evidence_problem(records, plan) -> str | None:
    """Why the launch records are not the selected plan's launches, or ``None``.

    The plan declares its launches; the runtime must have recorded exactly
    one record per launch, indexed ``0..n-1`` in order, each carrying the
    plan's layout fingerprint for that launch (when the plan states one)
    and a finite positive raw tick count.  Omitted work (a missing launch)
    or extra work is not timing evidence for the plan.
    """

    expected = list(getattr(plan, "launches", ()) or ())
    if not expected:
        return "the selected plan declares no launches"
    if len(records) != len(expected):
        return (
            f"{len(records)} launch record(s) for a plan of {len(expected)} launch(es)"
        )
    for position, (record, launch) in enumerate(zip(records, expected)):
        index = record.get("index")
        if index is None:
            return f"launch record {position} carries no launch index"
        if index != position:
            return f"launch record {position} carries index {index!r}"
        declared = getattr(launch, "layout_fingerprint", None)
        recorded = record.get("layout_fingerprint")
        if declared is not None:
            if not recorded:
                return f"launch {position}: no layout fingerprint recorded"
            if declared != recorded:
                return (
                    f"launch {position}: layout fingerprint {recorded[:12]} is not the "
                    f"plan's {declared[:12]}"
                )
        ticks = (record.get("raw_ticks") or {}).get("pipeline")
        if (
            ticks is None
            or isinstance(ticks, bool)
            or not isinstance(ticks, (int, float))
            or not math.isfinite(ticks)
            or ticks <= 0
        ):
            return f"launch {position}: raw device ticks {ticks!r} are not a positive finite count"
    return None


def _timing_problems(runs) -> list:
    """A run without the selected plan's complete device ticks is not timing
    evidence; the record is kept and the problem names what is missing."""

    return [
        f"{label}: {run.get('timing_evidence_problem') or 'no complete device tick evidence'} "
        f"({len(run.get('launches') or [])} launch record(s) for "
        f"{run.get('expected_launches')} expected)"
        for label, run in runs.items()
        if run.get("amortized_ticks_per_pipeline") is None
        or run.get("timing_evidence_problem")
    ]


def _run(compiled, args):
    compiled.repetitions = REPETITIONS
    result = compiled(**args)
    if result.kind != "device":
        raise RuntimeError(f"expected a device run, got kind {result.kind!r}")
    records = _launch_records(result)
    problem = launch_evidence_problem(records, compiled.plan)
    return {
        "plan": compiled.plan.name,
        "estimate_cycles": compiled.plan.estimate_cycles,
        "expected_launches": len(getattr(compiled.plan, "launches", ()) or ()),
        "cycles": result.cycles,
        "amortized_ticks_per_pipeline": (
            amortized_ticks(result, REPETITIONS) if problem is None else None
        ),
        "timing_evidence_problem": problem,
        "run_kind": result.kind,
        "final_pipeline_ticks": result.extra.get("final_pipeline_ticks"),
        "device_lock": result.extra.get("device_lock"),
        "cycles_scope": result.extra.get("cycles_scope"),
        "launches": records,
    }


def perturbation(seed: int) -> dict:
    target = build_apu_g2_target()
    perturbed = build_apu_g2_cost(
        APUG2_TYPED_TICK_MODEL.scaled(row_bits=20.0), name="apu_g2_cost_rows_x20"
    )
    rng = np.random.default_rng(seed)
    A = rng.integers(-8, 8, size=(64, 128)).astype(np.int16)
    B = rng.integers(-8, 8, size=(128, 33)).astype(np.int16)
    reference = A.astype(np.int64) @ B.astype(np.int64)
    out = {
        "experiment": "cost_perturbation",
        "kernel": "gemm 64x128x33 int4 -> int15",
        "stock_fingerprint": apu_g2_cost.bind(target).fingerprint,
        "perturbed_fingerprint": perturbed.bind(target).fingerprint,
        "runs": {},
    }
    for label, cost in (("stock", apu_g2_cost), ("perturbed_rows_x20", perturbed)):
        compiled = allo.compile(gemm_n33, target, cost)
        C = np.zeros((64, 33), dtype=np.int16)
        record = _run(compiled, {"A": A.copy(), "B": B.copy(), "C": C})
        record["outputs_exact"] = bool(np.array_equal(C.astype(np.int64), reference))
        record["device_source_sha256"] = None
        out["runs"][label] = record
    stock, moved = out["runs"]["stock"], out["runs"]["perturbed_rows_x20"]
    out["selection_changed"] = stock["plan"] != moved["plan"]
    out["prediction_stock_not_slower"] = (
        stock["cycles"] is not None
        and moved["cycles"] is not None
        and stock["cycles"] <= moved["cycles"]
    )
    # Invariants of the experiment: both exact, the selection moved and the
    # two programs launched different layouts.  A violated invariant fails
    # the command; the record is kept.
    problems = [
        f"{label}: outputs differ from the int64 reference"
        for label, run in out["runs"].items()
        if not run["outputs_exact"]
    ]
    problems.extend(_timing_problems(out["runs"]))
    if not out["selection_changed"]:
        problems.append("the perturbation did not change the selected plan")
    if {l["layout_fingerprint"] for l in stock["launches"]} == {
        l["layout_fingerprint"] for l in moved["launches"]
    }:
        problems.append("the two selections launched the same layouts")
    out["invariant_problems"] = problems
    return out


def forced_alternative(seed: int) -> dict:
    target = build_apu_g2_target()
    rng = np.random.default_rng(seed + 63)
    A = rng.integers(-4, 4, size=(256, 256)).astype(np.int16)
    p = rng.integers(-4, 4, size=256).astype(np.int16)
    r = rng.integers(-4, 4, size=256).astype(np.int16)
    out = {"experiment": "forced_alternative", "kernel": "bicg int3 N=256", "runs": {}}
    for label, layout in (("default", None), ("forced_separate_k1", "separate/k1")):
        compiled = allo.compile(
            PAPER_KERNELS["bicg"], target, apu_g2_cost, layout=layout
        )
        q = np.full(256, 7, np.int16)
        s = np.full(256, 7, np.int16)
        record = _run(
            compiled, {"A": A.copy(), "p": p.copy(), "r": r.copy(), "q": q, "s": s}
        )
        record["outputs_exact"] = bool(
            np.array_equal(q.astype(np.int64), A.astype(np.int64) @ p)
            and np.array_equal(s.astype(np.int64), r.astype(np.int64) @ A)
        )
        out["runs"][label] = record
    problems = [
        f"{label}: outputs differ from the int64 reference"
        for label, run in out["runs"].items()
        if not run["outputs_exact"]
    ]
    problems.extend(_timing_problems(out["runs"]))
    default, forced = out["runs"]["default"], out["runs"]["forced_separate_k1"]
    if default["plan"] == forced["plan"]:
        problems.append("the forced layout did not change the plan")
    if (len(default["launches"]), len(forced["launches"])) != (1, 2):
        problems.append("fused/separate launch counts are not (1, 2)")
    out["invariant_problems"] = problems
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=int(time.time_ns() % (1 << 31)))
    args = parser.parse_args(argv)
    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "allo": allo.__file__,
        "seed": args.seed,
        "repetitions": REPETITIONS,
        "experiments": [],
    }
    failures = 0
    for experiment in (perturbation, forced_alternative):
        try:
            record = experiment(args.seed)
        except Exception as error:  # a failed experiment is evidence too
            record = {
                "experiment": experiment.__name__,
                "error": f"{type(error).__name__}: {error}",
            }
        if record.get("error") or record.get("invariant_problems"):
            failures += 1
        session["experiments"].append(record)
        print(json.dumps(record, default=str), flush=True)
    session["status"] = "pass" if not failures else f"fail ({failures} experiment(s))"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=1, default=str)
    print(session["status"], flush=True)
    # A failed execution or a violated invariant is a nonzero exit; the
    # records are preserved either way.
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
