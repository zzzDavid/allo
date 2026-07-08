# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile every APU vector plan for the full milestone contraction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

import allo
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "apu_v1_vector_gemm_profile_workload", ROOT / "workload.py"
)
WORKLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKLOAD)


def main():
    target = build_apu_v1_target()
    virtual = allo.compile(
        WORKLOAD.vector_gemm, target, apu_v1_cost, backend="functional"
    )
    rng = np.random.default_rng(25)
    left = (rng.standard_normal((WORKLOAD.M, WORKLOAD.K)) * 0.05).astype(np.float16)
    right = (rng.standard_normal((WORKLOAD.K, WORKLOAD.N)) * 0.05).astype(np.float16)
    expected = left.astype(np.float32) @ right.astype(np.float32)

    import conftest

    records = []
    for plan in virtual.plans:
        result = np.zeros((WORKLOAD.M, WORKLOAD.N), dtype=np.float16)
        compiled = allo.compile(
            WORKLOAD.vector_gemm, target, apu_v1_cost, layout=plan.name
        )
        with conftest.board_lock():
            run = compiled(left, right, result)
        error = float(np.max(np.abs(result.astype(np.float32) - expected)))
        records.append(
            {
                "plan": plan.name,
                "estimated_cycles": int(compiled.estimate().cycles),
                "measured_cycles": int(run.cycles),
                "max_abs_error": error,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
