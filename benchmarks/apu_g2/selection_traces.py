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
pipeline body (``repetitions`` bodies), the layout and program fingerprints
and the device library hash of every launch.  Results are evidence; nothing
here is read by the compiler.

Usage::

    python -m benchmarks.apu_g2.selection_traces --out traces.json [--seed N]
"""

from __future__ import annotations

import argparse
import json
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
    return [
        {
            "layout_fingerprint": launch.get("layout_fingerprint"),
            "program_fingerprint": launch.get("program_fingerprint"),
            "device_library_sha256": (launch.get("artifact_sha256") or {}).get(
                "device_library"
            ),
            "raw_ticks": launch.get("raw_ticks"),
        }
        for launch in result.extra.get("launches", ())
    ]


def _run(compiled, args):
    compiled.repetitions = REPETITIONS
    result = compiled(**args)
    return {
        "plan": compiled.plan.name,
        "estimate_cycles": compiled.plan.estimate_cycles,
        "cycles": result.cycles,
        "run_kind": result.kind,
        "final_pipeline_ticks": result.extra.get("final_pipeline_ticks"),
        "device_lock": result.extra.get("device_lock"),
        "launches": _launch_records(result),
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
    for experiment in (perturbation, forced_alternative):
        try:
            record = experiment(args.seed)
        except Exception as error:  # a failed experiment is evidence too
            record = {
                "experiment": experiment.__name__,
                "error": f"{type(error).__name__}: {error}",
            }
        session["experiments"].append(record)
        print(json.dumps(record, default=str), flush=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
