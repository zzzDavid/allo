# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper's portability experiment: one FP16 GEMV source, strict order,
bit-exact on physical Gemini-I and Samsung PIMSimulator.

Section 5, "Workload Portability and Numerical Correctness": the same
4096x256 FP16 GEMV source and rounding-sensitive inputs run on physical
Gemini-I and on Samsung PIMSimulator, and both match all 4,096 reference
outputs when every multiply and addition rounds to FP16 in source order;
regrouping the additions changes the results.

This driver keeps the three parts of that claim explicit and independent:

* **the shared source** (:func:`gemv`, the ``A[row, depth] * B[depth,
  column]`` form of the historical audit) compiled through public
  ``allo.compile`` for each target with the default ``strict`` policy
  (``--reassociate`` adds the regrouping compile as the control);
* **the historical inputs** (:func:`historical_inputs`): signed powers of
  two from 2^-8 to 2^8 against a vector of ones, with six cancellation
  rows, seed 2026092910.  Every product is exactly representable, so the
  experiment tests the accumulation order and nothing else.  ``--inputs
  gaussian`` is a separate probe of inexact products (release issue
  V1-16), never the paper's claim;
* **the independent oracle** (:func:`source_order_fp16`): a NumPy loop that
  rounds each product and each partial sum to FP16 in ``depth`` order; it
  reads nothing from the compiler.

Each target's run must execute where the paper executed it (Samsung: the
PIMSimulator conductor, ``RunResult.kind == "simulator"``; Gemini-I: the
board, ``kind == "device"``); a virtual or functional run is reported as a
dry run and never counts as a reproduction.  Everything needed to trace an
output to its executable is retained under ``--out``: the inputs and the
oracle, every run's actual outputs (``.npy``), the retained MLIR, the
selected plan / placement and the backend's own artifact identity (the
generated device source and binary hash, launch roles and profile text on
Gemini-I; the CRF words, plan text, conductor receipt and ``pim_driver``
identity on Samsung), the complete ``RunResult.extra``, and the
environment (interpreter, NumPy, allo location and revision, host, date).
A failed compile or run of any requested run, the controls included, is
recorded in the same files before the command exits 1; a control that
executes is allowed to differ from the oracle.

Usage::

    python -m benchmarks.portability.gemv_source_order --out DIR
        [--targets samsung apu_v1] [--inputs historical|gaussian]
        [--reassociate] [--backend device|virtual]
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import allo
from allo.ir.types import float16

M, K, N = 4096, 256, 1
HISTORICAL_SEED = 2026092910
CANCELLATION_ROWS = (
    [2048, 1, -2048],
    [2048, -2048, 1],
    [1, 2048, -2048],
    [-2048, -1, 2048],
    [4096, 1, 1, -4096],
    [1, 1, 4096, -4096],
)
# Where the paper executed each target and what the runtime reports there.
EXPECTED_EXECUTION = {"samsung": "simulator", "apu_v1": "device"}


def gemv(A: float16[M, K], B: float16[K, N], out: float16[M, N]):
    """The shared GEMV source of the paper's audit (one function, every target)."""
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            out[row, column] += A[row, depth] * B[depth, column]


def gemv_source():
    return gemv


def historical_inputs(seed: int = HISTORICAL_SEED):
    """The audit's rounding-sensitive operands: exact products, cancellation
    inside the FP16 accumulation."""
    rng = np.random.default_rng(seed)
    magnitudes = np.exp2(rng.integers(-8, 9, (M, K)))
    A = (magnitudes * rng.choice([-1, 1], magnitudes.shape)).astype(np.float16)
    B = np.ones((K, N), dtype=np.float16)
    for row, pattern in enumerate(CANCELLATION_ROWS):
        A[row] = 0
        A[row, : len(pattern)] = pattern
    return A, B


def gaussian_inputs(seed: int = 0):
    """Inexact FP16 products (a probe of the device's multiply, V1-16)."""
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    return A, B


def products_are_exact(A, B) -> bool:
    """True when every FP16 product equals its FP32 product (no rounding)."""
    exact32 = A.astype(np.float32)[:, :, None] * B.astype(np.float32)[None, :, :]
    return bool(np.array_equal(exact32.astype(np.float16).astype(np.float32), exact32))


def source_order_fp16(A, B):
    """``sum_depth A[:, depth] * B[depth, :]`` rounding every product and every
    partial sum to FP16, depth ascending, from a zero accumulator."""
    acc = np.zeros((A.shape[0], B.shape[1]), dtype=np.float16)
    for depth in range(A.shape[1]):
        product = (A[:, depth, None] * B[depth, :][None, :]).astype(np.float16)
        acc = (acc + product).astype(np.float16)
    return acc


def grouped_numpy_fp16(A, B):
    """NumPy's own FP16 reduction (pairwise), the regrouped comparison point."""
    return (A * B[:, 0]).sum(axis=1, dtype=np.float16)[:, None]


def _sha256(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ProvenanceError(RuntimeError):
    """A required provenance value could not be serialized structurally."""


def _json_ready(value, depth=0, *, path="value"):
    """Structural JSON form of a provenance value.  Mappings (including
    ``mappingproxy`` and other ``Mapping`` views), sequences, sets, arrays,
    NumPy scalars and objects with a ``manifest()`` are rendered
    structurally; anything else is rendered as ``repr`` text, and never
    truncated.  Nesting deeper than 12 levels is a provenance error, not a
    silent cut."""
    if depth > 12:
        raise ProvenanceError(f"{path}: provenance nests deeper than 12 levels")
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v, depth + 1, path=f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v, depth + 1, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_ready(v, depth + 1, path=path) for v in value), key=repr)
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape), "dtype": str(value.dtype), "sha256": _sha256(value)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _json_ready(manifest(), depth + 1, path=f"{path}.manifest()")
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_ready(getattr(value, name), depth + 1, path=f"{path}.{name}")
            for name in value.__dataclass_fields__
        }
    return repr(value)


def environment(out_dir: Path) -> dict:
    allo_dir = Path(allo.__file__).resolve().parent
    revision = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(allo_dir.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.strip() or None
    except Exception:
        revision = None
    return {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "allo": str(allo.__file__),
        "allo_git_revision": revision,
        "installed_artifact": "site-packages" in str(allo_dir),
        "evidence_class": (
            "installed artifact" if "site-packages" in str(allo_dir) else "checkout"
        ),
        "out": str(out_dir),
    }


def _samsung_identity() -> dict:
    """The installed conductor's identity (hash, geometry), when present."""
    try:
        from allo.spmw_samsung import samsung_simulator_geometry
        from allo.spmw_simenv import pimsim_root

        root = pimsim_root()
        driver = Path(root) / "pim_driver"
        return {
            "pim_driver": str(driver),
            "pim_driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
            "geometry": samsung_simulator_geometry(root),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:300]}


def _target(name):
    from allo.pim import costs, targets

    if name == "samsung":
        return targets.build_samsung_target(), costs.samsung_cost
    if name == "apu_v1":
        return targets.build_apu_v1_target(), costs.apu_v1_cost
    raise ValueError(f"unknown target {name!r}; known: {sorted(EXPECTED_EXECUTION)}")


def _board_lock(name):
    if name == "apu_v1":
        from allo.pim.apu_v1_program import apu_v1_board_lock

        return apu_v1_board_lock()
    import contextlib

    return contextlib.nullcontext()


def run_target(name, A, B, oracle, out_dir: Path, *, backend, reduction_order=None):
    """Compile the shared source for one target, run it, keep everything."""
    label = f"{name}_{reduction_order or 'strict'}"
    record = {"target": name, "label": label, "reduction_order": reduction_order or "strict (default)"}
    target, cost = _target(name)
    kwargs = {"backend": None if backend == "device" else backend}
    if reduction_order:
        kwargs["reduction_order"] = reduction_order
    started = time.perf_counter()
    try:
        compiled = allo.compile(gemv_source(), target, cost, **kwargs)
    except Exception as error:
        record["compile_error"] = f"{type(error).__name__}: {error}"[:1000]
        return record
    record["compile_seconds"] = round(time.perf_counter() - started, 2)
    # Retained IR and the selection, before anything runs.
    try:
        (out_dir / f"{label}.retained.mlir").write_text(str(compiled.schedule.module))
        record["retained_ir"] = f"{label}.retained.mlir"
    except Exception as error:  # pragma: no cover - diagnostic only
        record["retained_ir_error"] = f"{type(error).__name__}: {error}"[:200]
    plan = getattr(compiled, "selected_plan", None)
    record["selected_plan"] = getattr(plan, "name", None)
    record["selected_plan_metadata"] = _json_ready(getattr(plan, "metadata", None))
    record["reduction_strategy"] = _json_ready(getattr(plan, "reduction_strategy", None))
    if plan is not None and callable(getattr(plan, "manifest", None)):
        # The plan's own manifest: iteration and value layouts, transfers,
        # temporal and reduction strategies, operations, VR allocations.
        record["selected_plan_manifest"] = _json_ready(plan.manifest())
    artifact = compiled.compiled
    layout = getattr(artifact, "layout", None)
    record["layout_mode"] = getattr(layout, "mode", None)
    effects = (getattr(layout, "extra", None) or {}).get("effects")
    if effects is not None:
        # The shared candidate contract the selection was checked against.
        record["candidate_effects_manifest"] = _json_ready(effects)
    record["layout_extra"] = _json_ready(
        {k: v for k, v in (getattr(layout, "extra", None) or {}).items() if k != "effects"}
    )
    programs = list(getattr(getattr(artifact, "layout_ctx", None), "programs", ()) or ())
    if programs:
        record["samsung_programs"] = [
            {
                "kind": p.kind, "lever_record": _json_ready(p.lever_record),
                "k_tiles": p.k_tiles, "tile_fiber": list(p.tile_fiber),
                "rows_per_block": p.rows_per_block, "vector_from_bank": p.vector_from_bank,
                "row_map": p.row_map,
            }
            for p in programs
        ]
        record["crf_words"] = len(getattr(artifact, "cmds", ()) or ())
    realization = getattr(compiled, "realization", None)
    source_text = None
    if realization is not None and hasattr(realization, "device_source"):
        try:
            source_text = realization.device_source()
        except Exception:
            source_text = None
    if source_text is None and hasattr(compiled, "device_source"):
        try:
            source_text = compiled.device_source()
        except Exception:
            source_text = None
    if source_text:
        (out_dir / f"{label}.device_source.c").write_text(source_text)
        record["device_source"] = f"{label}.device_source.c"
        record["device_source_sha256"] = hashlib.sha256(source_text.encode()).hexdigest()
    # Run.
    out = np.zeros((M, N), dtype=np.float16)
    started = time.perf_counter()
    try:
        with _board_lock(name):
            result = compiled(A, B, out)
    except Exception as error:
        record["run_error"] = f"{type(error).__name__}: {error}"[:1000]
        np.save(out_dir / f"{label}.output.npy", out)
        return record
    record["run_seconds"] = round(time.perf_counter() - started, 2)
    np.save(out_dir / f"{label}.output.npy", out)
    record["output"] = f"{label}.output.npy"
    record["output_sha256"] = _sha256(out)
    record["execution_kind"] = result.kind
    record["backend"] = getattr(result, "backend", None)
    record["cycles"] = result.cycles
    extra = dict(result.extra or {})
    for key in ("profile_text", "device_source", "kernel_src", "crf"):
        text = extra.get(key)
        if isinstance(text, str) and text:
            (out_dir / f"{label}.{key}.txt").write_text(text)
            extra[key] = f"{label}.{key}.txt"
    record["run_extra"] = _json_ready(extra)
    record["binary_sha256"] = extra.get("binary_sha256")
    mismatches = int(np.count_nonzero(out.view(np.uint16) != oracle.view(np.uint16)))
    record["bit_exact_outputs"] = int(oracle.size - mismatches)
    record["mismatching_outputs"] = mismatches
    record["cancellation_rows_exact"] = bool(
        np.array_equal(out[: len(CANCELLATION_ROWS)].view(np.uint16),
                       oracle[: len(CANCELLATION_ROWS)].view(np.uint16))
    )
    exact32 = A.astype(np.float32) @ B.astype(np.float32)
    record["max_abs_error_vs_fp32"] = float(np.max(np.abs(out.astype(np.float32) - exact32)))
    expected = EXPECTED_EXECUTION[name]
    record["execution_where_the_paper_executed"] = result.kind == expected
    record["dry_run"] = result.kind != expected
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="evidence directory (created)")
    parser.add_argument("--targets", nargs="+", default=["samsung", "apu_v1"],
                        choices=sorted(EXPECTED_EXECUTION))
    parser.add_argument("--inputs", default="historical", choices=("historical", "gaussian"))
    parser.add_argument("--reassociate", action="store_true",
                        help="also run the regrouping compile of each target (the control)")
    parser.add_argument("--backend", default="device", choices=("device", "virtual"),
                        help="device: the paper's executors; virtual: a dry run, never a reproduction")
    args = parser.parse_args(argv)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    A, B = historical_inputs() if args.inputs == "historical" else gaussian_inputs()
    oracle = source_order_fp16(A, B)
    grouped = grouped_numpy_fp16(A, B)
    np.savez(out_dir / "inputs_and_oracle.npz", A=A, B=B, oracle=oracle, grouped_numpy=grouped)
    report = {
        "experiment": "paper Sec. 5 portability: one FP16 GEMV source, strict order, bit-exact on both targets",
        "shape": {"M": M, "K": K, "N": N},
        "inputs": {
            "kind": args.inputs,
            "seed": HISTORICAL_SEED if args.inputs == "historical" else 0,
            "products_exact": products_are_exact(A, B),
            "A_sha256": _sha256(A), "B_sha256": _sha256(B),
            "oracle_sha256": _sha256(oracle),
            "rows_where_source_order_differs_from_numpy_grouped_sum": int(
                np.count_nonzero(oracle.view(np.uint16) != grouped.view(np.uint16))
            ),
            "paper_claim_applies": args.inputs == "historical",
        },
        "oracle": "NumPy loop: product and partial sum rounded to FP16 in depth order from zero",
        "environment": environment(out_dir),
        "samsung_identity": _samsung_identity() if "samsung" in args.targets else None,
        "runs": [],
        "problems": [],
    }
    report_path = out_dir / "report.json"

    def flush():
        report_path.write_text(json.dumps(report, indent=1, default=str))

    flush()
    for name in args.targets:
        orders = [None] + (["reassociate"] if args.reassociate else [])
        for order in orders:
            try:
                record = run_target(name, A, B, oracle, out_dir, backend=args.backend,
                                    reduction_order=order)
            except Exception as error:  # the failure is evidence too
                record = {"target": name, "label": f"{name}_{order or 'strict'}",
                          "error": f"{type(error).__name__}: {error}"[:1000]}
            report["runs"].append(record)
            flush()
            print(json.dumps({k: v for k, v in record.items()
                              if k not in ("run_extra", "selected_plan_metadata")},
                             default=str), flush=True)
            # Every requested run must have executed where the paper executed;
            # only the strict runs are held to the source-order oracle (the
            # regrouping control is expected to differ from it).
            if record.get("compile_error") or record.get("run_error") or record.get("error"):
                report["problems"].append(f"{record['label']}: did not execute")
            elif record.get("dry_run"):
                report["problems"].append(
                    f"{record['label']}: executed as {record['execution_kind']!r}, not "
                    f"{EXPECTED_EXECUTION[name]!r} where the paper executed it (dry run)"
                )
            elif order is None and record.get("mismatching_outputs"):
                report["problems"].append(
                    f"{record['label']}: {record['mismatching_outputs']} of {M * N} outputs "
                    "differ from the source-order FP16 oracle"
                )
    if args.inputs != "historical":
        report["problems"].append(
            "inputs are not the paper's (inexact products): this run is a V1-16 probe, "
            "not the paper's claim"
        )
    report["status"] = "pass" if not report["problems"] else "fail"
    if args.backend == "virtual":
        report["status"] = "dry run (virtual backend): not a reproduction"
    flush()
    print(report["status"], report["problems"], flush=True)
    if args.backend == "virtual":
        return 0 if not [p for p in report["problems"] if "did not execute" in p] else 1
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
