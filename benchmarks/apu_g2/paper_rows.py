# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the paper's Gemini-II kernels against the GTML baseline on the card.

The five kernels of ``tables/eval-apuv2.tex`` run through ``allo.compile``
under the default shared selection (the Tenon lane) interleaved with the
archived public-GTML ``VectorPack`` host of the evaluation assets
(``$TENON_ARTIFACTS/apuv2-composed-v2/provenance/implementations/vendor``,
the vendor lane), on the archived fixtures (``inputs/<case>``) with exact
output checks against the archived oracles.

Protocol (the archived campaign's, ``apuv2-composed-v2/METHODOLOGY.md``):
one warm-up launch per lane and case, then ``--rounds`` (default 20) paired
rounds; even rounds run the vendor lane first, odd rounds the Tenon lane
first; the case order rotates by round and reverses on odd rounds; one
exclusive device lock is held for the whole session.  Every launch is a
batch of ``repetitions`` (20) complete pipeline bodies plus one independent
final body; the statistic is the repeated batch's device ticks divided by
the repetitions (fractions kept).  Per lane and case the medians, quartiles,
IQR and MAD are reported, and the comparison is the median of the same-round
ratios ``vendor / tenon`` with its IQR and MAD.  Provenance per launch: the
Tenon plan, layout and program fingerprints, packing attestation, artifact
hashes and host timings; the vendor host and device-library hashes, the
benchmark and witness records.  The paper's values are comparison data.

Exit status: 0 only when every launch executed with exact outputs and every
case has its full set of rounds; otherwise 1, with the records preserved.

Usage (on the card host, from an installed wheel)::

    python -m benchmarks.apu_g2.paper_rows --out rows.json [--rounds 20]
        [--cases gemm ...] [--no-vendor]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPETITIONS = 20

# Paper kernel -> archived campaign case id, vendor kernel label, paper
# (GTML, Tenon) cycles of tables/eval-apuv2.tex.
CASES = {
    "gemm": ("gemm_i4_m64_k128_n31", "gemm_i4_m64_k128_n31", 1903.850, 1733.225),
    "batched_gemm": (
        "batched_gemm_i4_b4_m32_k128_n16",
        "batch4_gemm_i4_m32_k128_n16",
        1906.400,
        1739.150,
    ),
    "gesummv": (
        "gesummv_i2_m1024_k128",
        "gesummv_i2_m1024_k128_a3_b2",
        3090.975,
        1860.075,
    ),
    "bicg": ("bicg_i3_n256", "bicg_i3_n256", 2726.500, 2548.825),
    "mvt": ("mvt_i3_n256", "mvt_i3_n256_acc14", 3089.650, 2766.625),
}
# Oracle rows (archived ``oracle.i16le.bin``) -> the kernel's outputs.
ORACLE_OUTPUTS = {
    "gemm": ("C",),
    "batched_gemm": ("C",),
    "gesummv": ("y",),
    "bicg": ("q", "s"),
    "mvt": ("x1", "x2"),
}


def _artifacts_root(argument):
    if argument:
        return Path(argument)
    env = os.environ.get("TENON_ARTIFACTS")
    if env:
        return Path(env)
    from allo.spmw_simenv import tenon_artifacts_root

    return tenon_artifacts_root()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statistics(values):
    if not values:
        return {}
    ordered = sorted(values)
    median = statistics.median(ordered)
    quartiles = (
        statistics.quantiles(ordered, n=4, method="inclusive")
        if len(ordered) > 1
        else [ordered[0]] * 3
    )
    return {
        "samples": len(ordered),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "q1": quartiles[0],
        "q3": quartiles[2],
        "iqr": quartiles[2] - quartiles[0],
        "mad": statistics.median(abs(v - median) for v in ordered),
    }


class Fixture:
    """The archived inputs and oracle of one case."""

    def __init__(self, campaign: Path, case_id: str):
        self.dir = campaign / "inputs" / case_id
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.inputs, self.sha256 = {}, {}
        self.oracle = None
        for array in manifest["arrays"]:
            path = self.dir / array["file"]
            self.sha256[array["name"]] = _sha256_file(path)
            if array["role"] == "oracle":
                self.oracle = np.fromfile(path, dtype="<i2").reshape(array["shape"])
            else:
                # Signed values serialized as 16-bit two's complement.
                self.inputs[array["name"]] = np.fromfile(path, dtype="<i2").reshape(
                    array["shape"]
                )
        if self.oracle is None:
            raise FileNotFoundError(f"no oracle in {self.dir}")


class TenonLane:
    def __init__(self, name: str, fixture: Fixture):
        import allo
        from allo.pim.costs.apu_g2 import apu_g2_cost
        from allo.pim.targets import build_apu_g2_target

        from .workloads import PAPER_KERNELS

        self.name = name
        self.fixture = fixture
        target = build_apu_g2_target()
        started = time.perf_counter()
        self.compiled = allo.compile(PAPER_KERNELS[name], target, apu_g2_cost)
        self.compile_seconds = round(time.perf_counter() - started, 2)
        self.compiled.repetitions = REPETITIONS
        self.cost_fingerprint = apu_g2_cost.bind(target).fingerprint
        contract = self.compiled.contract
        self.parameters = list(contract.parameters)
        self.outputs = tuple(contract.outputs)
        lower = {key.lower(): value for key, value in fixture.inputs.items()}
        self.arguments = {}
        for parameter in self.parameters:
            if parameter.lower() in lower:
                self.arguments[parameter] = lower[parameter.lower()]
            elif parameter in self.outputs:
                shape = contract.parameter_shape(parameter)
                self.arguments[parameter] = np.zeros(shape, dtype=np.int16)
            else:
                raise KeyError(f"{name}: no fixture array for parameter {parameter!r}")

    def describe(self):
        return {
            "lane": "tenon",
            "plan": self.compiled.plan.name,
            "estimate_cycles": self.compiled.plan.estimate_cycles,
            "compile_seconds": self.compile_seconds,
            "cost_fingerprint": self.cost_fingerprint,
            "ranking": [
                (plan.name, plan.estimate_cycles) for plan in self.compiled.ranking
            ],
            "inputs_sha256": self.fixture.sha256,
        }

    def launch(self):
        args = {k: v.copy() for k, v in self.arguments.items()}
        result = self.compiled(**args)
        if result.kind != "device":
            raise RuntimeError(f"expected a device run, got {result.kind!r}")
        produced = np.stack(
            [args[name].reshape(-1) for name in ORACLE_OUTPUTS[self.name]]
        ).reshape(self.fixture.oracle.shape)
        exact = bool(np.array_equal(produced.astype(np.int64), self.fixture.oracle))
        launches = list(result.extra.get("launches", ()) or ())
        expected = len(self.compiled.plan.launches)
        batch = [(launch.get("raw_ticks") or {}).get("pipeline") for launch in launches]
        timing_problem = timing_evidence_problem(batch, expected)
        amortized = (
            sum(float(t) / REPETITIONS for t in batch) if timing_problem is None else None
        )
        return {
            "amortized_ticks_per_pipeline": amortized,
            "timing_evidence_problem": timing_problem,
            "expected_launches": expected,
            "repeated_batch_ticks": batch,
            "cycles_rounded": result.cycles,
            "outputs_exact": exact,
            "launches": [
                {
                    "index": l.get("index"),
                    "layout_fingerprint": l.get("layout_fingerprint"),
                    "program_fingerprint": l.get("program_fingerprint"),
                    "packing_attestation": l.get("packing_attestation"),
                    "artifact_sha256": l.get("artifact_sha256"),
                    "raw_ticks": l.get("raw_ticks"),
                    "host_timings_us": l.get("host_timings_us"),
                }
                for l in launches
            ],
            "device_lock": result.extra.get("device_lock"),
        }


def timing_evidence_problem(batch_ticks, expected_launches) -> str | None:
    """Why a launch's device ticks cannot serve as a measurement, or ``None``.

    A measurement needs one launch record per launch of the selected plan,
    each with a finite positive raw tick count for the repeated batch; an
    empty launch list is not a zero-tick program.
    """

    if expected_launches <= 0:
        return "the selected plan declares no launches"
    if len(batch_ticks) != expected_launches:
        return (
            f"{len(batch_ticks)} launch record(s) for a plan of "
            f"{expected_launches} launch(es)"
        )
    for index, ticks in enumerate(batch_ticks):
        if isinstance(ticks, bool) or not isinstance(ticks, (int, float)):
            return f"launch {index}: no raw device ticks"
        if not math.isfinite(ticks) or ticks <= 0:
            return f"launch {index}: raw device ticks {ticks!r} are not a positive finite count"
    return None


def loaded_device_library(benchmark: dict, expected_sha256: str) -> dict:
    """Identify the device library the GTML host actually loaded.

    The archived host has its library location compiled in
    (``DEVICE_SIDE_LIB_LOCATION``) and reports it in the benchmark record;
    the file at that location must be byte-identical to the archived
    ``device.update.bin`` or the lane is not the campaign's baseline.
    Returns the loaded path, its digest and a problem string (or ``None``).
    """

    reported = benchmark.get("device_library")
    if not isinstance(reported, str) or not reported:
        return {
            "loaded_device_library": None,
            "loaded_device_library_sha256": None,
            "problem": "the GTML host did not report its device library",
        }
    path = Path(reported)
    digest = _sha256_file(path) if path.is_file() else None
    problem = None
    if digest is None:
        problem = f"the GTML host's device library is not a file: {reported}"
    elif digest != expected_sha256:
        problem = (
            "the GTML host loaded a device library other than the archived one: "
            f"{reported} sha256 {digest[:16]} != {expected_sha256[:16]}"
        )
    return {
        "loaded_device_library": reported,
        "loaded_device_library_sha256": digest,
        "problem": problem,
    }


class VendorLane:
    """The archived public-GTML VectorPack host binary of the campaign."""

    def __init__(self, name: str, campaign: Path, fixture: Fixture, scratch: Path):
        case_id, kernel, _gtml, _tenon = CASES[name]
        root = campaign / "provenance" / "implementations" / "vendor" / "binaries"
        self.host = root / "host"
        self.arc = root / "device.update.bin"
        for path in (self.host, self.arc):
            if not path.is_file():
                raise FileNotFoundError(f"vendor lane asset missing: {path}")
        self.kernel = kernel
        self.case_id = case_id
        self.fixture_dir = (campaign / "inputs").resolve()
        self.fixture = fixture
        self.scratch = scratch
        self.host_sha256 = _sha256_file(self.host)
        self.arc_sha256 = _sha256_file(self.arc)
        self.count = 0

    def describe(self):
        return {
            "lane": "vendor_gtml_vectorpack",
            "host": str(self.host),
            "host_sha256": self.host_sha256,
            "device_library": str(self.arc),
            "device_library_sha256": self.arc_sha256,
            "loaded_device_library_rule": (
                "the host reports the library location compiled into it; the "
                "file there must hash to device_library_sha256 or the launch "
                "is recorded as an error"
            ),
            "kernel": self.kernel,
            "inputs_sha256": self.fixture.sha256,
        }

    def launch(self):
        self.count += 1
        witness_root = self.scratch / f"{self.case_id}-{self.count}"
        command = [
            str(self.host),
            "--kernel",
            self.kernel,
            "--repetitions",
            str(REPETITIONS),
            "--fixture-dir",
            str(self.fixture_dir),
            "--witness-dir",
            str(witness_root),
        ]
        completed = subprocess.run(
            command, capture_output=True, timeout=600, check=False
        )
        stdout = completed.stdout.decode(errors="replace")
        if completed.returncode:
            raise RuntimeError(
                f"GTML host exited {completed.returncode}: "
                + completed.stderr.decode(errors="replace")[-2000:]
            )
        records = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        benchmarks = [r for r in records if r.get("record") == "benchmark"]
        if len(benchmarks) != 1:
            raise RuntimeError(f"GTML host emitted {len(benchmarks)} benchmark records")
        benchmark = benchmarks[0]
        witness = Path(benchmark.get("witness_dir") or (witness_root / self.kernel))
        output = witness / "actual.i16le.bin"
        exact = (
            output.is_file()
            and output.read_bytes() == self.fixture.oracle.astype("<i2").tobytes()
        )
        ticks = benchmark.get("repeated_ticks_total")
        loaded = loaded_device_library(benchmark, self.arc_sha256)
        timing_problem = timing_evidence_problem([ticks], 1)
        record = {
            "amortized_ticks_per_pipeline": (
                float(ticks) / REPETITIONS if timing_problem is None else None
            ),
            "timing_evidence_problem": timing_problem,
            "repeated_batch_ticks": [ticks],
            "final_pipeline_ticks": benchmark.get("independent_final_ticks"),
            "outputs_exact": bool(exact),
            "benchmark": benchmark,
            "witness_manifest": (
                json.loads((witness / "manifest.json").read_text())
                if (witness / "manifest.json").is_file()
                else None
            ),
            "host_sha256": self.host_sha256,
            "device_library_sha256": self.arc_sha256,
            "loaded_device_library": loaded["loaded_device_library"],
            "loaded_device_library_sha256": loaded["loaded_device_library_sha256"],
        }
        if loaded["problem"]:
            record["error"] = loaded["problem"]
        return record


def run_session(names, root, *, rounds, vendor):
    from allo.pim.apu_g2_runtime import apu_g2_device_lock

    campaign = root / "apuv2-composed-v2"
    scratch = Path(tempfile.mkdtemp(prefix="tenon-g2-vendor-"))
    session = {
        "cases": {},
        "launches": [],
        "problems": [],
        "protocol": {
            "warmups_per_lane": 1,
            "paired_rounds": rounds,
            "repetitions": REPETITIONS,
            "order": "even rounds vendor first, odd rounds tenon first; case order rotates by round and reverses on odd rounds",
            "statistic": "repeated batch device ticks / repetitions (fractions kept); median over rounds; paired same-round vendor/tenon ratios",
        },
    }
    lanes = {}
    for name in names:
        case_id, _kernel, gtml, tenon = CASES[name]
        fixture = Fixture(campaign, case_id)
        entry = {
            "case_id": case_id,
            "paper_gtml": gtml,
            "paper_tenon": tenon,
            "lanes": {},
        }
        lanes[name] = {"tenon": TenonLane(name, fixture)}
        entry["lanes"]["tenon"] = lanes[name]["tenon"].describe()
        if vendor:
            lanes[name]["vendor"] = VendorLane(name, campaign, fixture, scratch)
            entry["lanes"]["vendor"] = lanes[name]["vendor"].describe()
        session["cases"][name] = entry

    def run(name, lane_name, kind, round_index):
        try:
            launch = lanes[name][lane_name].launch()
        except Exception as error:  # the failed launch is evidence
            launch = {"error": f"{type(error).__name__}: {error}"}
        launch.update(case=name, lane=lane_name, kind=kind, round=round_index)
        session["launches"].append(launch)
        if launch.get("error"):
            session["problems"].append(
                f"{name} {lane_name} {kind} {round_index}: {launch['error']}"
            )
        elif not launch.get("outputs_exact"):
            session["problems"].append(
                f"{name} {lane_name} {kind} {round_index}: outputs differ from the oracle"
            )
        elif launch.get("amortized_ticks_per_pipeline") is None:
            session["problems"].append(
                f"{name} {lane_name} {kind} {round_index}: "
                + (launch.get("timing_evidence_problem") or "no device ticks")
            )

    order = list(names)
    with apu_g2_device_lock() as lock_path:
        session["device_lock"] = str(lock_path)
        for name in order:
            for lane_name in lanes[name]:
                run(name, lane_name, "warmup", 0)
        for round_index in range(1, rounds + 1):
            shift = (round_index - 1) % len(order)
            case_order = order[shift:] + order[:shift]
            if round_index % 2 == 1:
                case_order = list(reversed(case_order))
            lane_order = (
                ["vendor", "tenon"] if round_index % 2 == 0 else ["tenon", "vendor"]
            )
            for name in case_order:
                for lane_name in lane_order:
                    if lane_name in lanes[name]:
                        run(name, lane_name, "measured", round_index)
    shutil.rmtree(scratch, ignore_errors=True)

    for name, entry in session["cases"].items():
        per_round = {}
        for launch in session["launches"]:
            if launch["case"] != name or launch["kind"] != "measured":
                continue
            if launch.get("error") or not launch.get("outputs_exact"):
                continue
            value = launch.get("amortized_ticks_per_pipeline")
            if value is None:
                continue
            per_round.setdefault(launch["round"], {})[launch["lane"]] = value
        for lane_name in entry["lanes"]:
            values = [r[lane_name] for r in per_round.values() if lane_name in r]
            entry["lanes"][lane_name]["amortized_ticks"] = values
            entry["lanes"][lane_name]["statistics"] = _statistics(values)
            if len(values) < rounds:
                session["problems"].append(
                    f"{name} {lane_name}: {len(values)} valid rounds of {rounds}"
                )
        tenon_median = entry["lanes"]["tenon"].get("statistics", {}).get("median")
        if tenon_median is not None:
            entry["comparison"] = {
                "tenon_median": tenon_median,
                "ratio_to_paper_tenon": tenon_median / entry["paper_tenon"],
                "ratio_to_paper_gtml": tenon_median / entry["paper_gtml"],
            }
            if "vendor" in entry["lanes"] and entry["lanes"]["vendor"].get(
                "statistics"
            ):
                ratios = [
                    r["vendor"] / r["tenon"]
                    for r in per_round.values()
                    if "vendor" in r and "tenon" in r
                ]
                entry["comparison"].update(
                    {
                        "vendor_median": entry["lanes"]["vendor"]["statistics"][
                            "median"
                        ],
                        "vendor_over_tenon_medians": entry["lanes"]["vendor"][
                            "statistics"
                        ]["median"]
                        / tenon_median,
                        "paired_round_ratios_vendor_over_tenon": ratios,
                        "paired_ratio_statistics": _statistics(ratios),
                        "paper_gtml_over_tenon": entry["paper_gtml"]
                        / entry["paper_tenon"],
                    }
                )
    return session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts")
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--no-vendor", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.cases:
        parser.error("--cases must name at least one case")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    unknown = [case for case in args.cases if case not in CASES]
    if unknown:
        parser.error(f"unknown cases: {unknown}; known: {list(CASES)}")
    import allo

    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "allo": allo.__file__,
    }
    try:
        root = _artifacts_root(args.artifacts)
        session["artifacts"] = str(root)
        session.update(
            run_session(args.cases, root, rounds=args.rounds, vendor=not args.no_vendor)
        )
    except Exception as error:
        session["error"] = f"{type(error).__name__}: {error}"
    problems = session.get("problems", []) or (
        [session["error"]] if session.get("error") else []
    )
    session["status"] = "pass" if not problems and not session.get("error") else "fail"
    Path(args.out).write_text(json.dumps(session, indent=1, default=str))
    for name, entry in (session.get("cases") or {}).items():
        print(
            json.dumps(
                {
                    "case": name,
                    "comparison": entry.get("comparison"),
                    "problems": [p for p in problems if p.startswith(name)],
                },
                default=str,
            ),
            flush=True,
        )
    print(session["status"], *problems[:5], flush=True)
    return 0 if session["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
