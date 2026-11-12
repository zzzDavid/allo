# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the paper's Gemini-I (APU v1) kernels through the shared pipeline.

For each kernel of ``tables/eval-apuv1.tex`` the workload of the evaluation
assets (``$TENON_ARTIFACTS/apuv1/tenon/<case>/workload.py``, an
evaluation-only tree distributed separately from the release package) is
compiled with ``allo.compile`` under the default shared selection; its
selected plan or native route, candidate estimates and generated device
source are recorded; and the program is launched on the board under the
shared board lock with the archived inputs (``inputs/<case>/<name>.uint16.
bin``), every output compared with the archived oracle.

Protocol (the archived comparison's, ``apuv1/TENON_COMPARISON.md``): one
warm-up launch and ``--measured`` (default 10) measured launches per lane
and case; with ``--expert`` the corrected expert Phoenix lane of the same
assets (``apuv1/expert/<case>``: archived release binary, manifest and io)
is interleaved launch by launch with the Tenon lane (round r runs Tenon
then expert for even r, expert then Tenon for odd r), and paired ratios are
reported with the median, quartiles, IQR and MAD of every lane.  The
statistic is the median ``crun`` of the profiler's ``total`` record (one
APUC, the paper's boundary); the counters ``seu``/``iall``/``icm``/``dcm``
of every launch are retained for attribution, with the binary hash, the
generated source, the device log and the profiler text.

Options force a plan (``--layout``), perturb one cost constant
(``--perturb NAME=FACTOR`` on ``allo.pim.costs.apu_v1``) or substitute a
historical generated device source for the Tenon lane (``--device-source
FILE``: a labelled control for attributing a difference, never the shared
selection).  For attribution in one session, ``--control-source FILE``
(repeatable) adds a lane per generated source interleaved in the same
rounds as the default selection (lane ``control:<file stem>``), and
``--historical-binary`` adds the archived paper-era Tenon binary of
``apuv1/tenon/<case>`` as a lane; every non-Tenon lane reports its
same-round ratios over the default selection.  The paper's own values are
comparison data; nothing here is read by the compiler.

Exit status: 0 only when every requested case executed, every launch's
outputs equal the oracle and the experiment's invariants hold; otherwise 1,
with the failed records preserved in ``--out``.

Usage::

    python -m benchmarks.apu_v1.paper_rows --out rows.json [--cases ...]
        [--measured 10] [--expert] [--backend device|virtual]
        [--layout PLAN] [--perturb CONST=FACTOR] [--device-source FILE]
        [--control-source FILE ...] [--historical-binary]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import tempfile
import os
import platform
import statistics
import sys
import time
from pathlib import Path

CASES = ("matrix_multiply", "kmeans", "histogram", "linear_regression", "word_count")

# Paper table rows (tables/eval-apuv1.tex): expert Phoenix cycles and the
# paper's reported Tenon median device cycles.  Comparison data only.
PAPER = {
    "matrix_multiply": (8_784_431, 8_764_900),
    "kmeans": (641_471, 624_677),
    "histogram": (167_384, 166_783),
    "linear_regression": (95_785, 94_618.5),
    "word_count": (175_397.5, 171_988),
}


class MeasurementError(RuntimeError):
    """A launch that did not produce a valid measurement."""


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
    if not path.is_file():
        raise FileNotFoundError(f"no workload for {case!r} at {path}")
    spec = importlib.util.spec_from_file_location(f"paper_apu_v1_{case}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build()


class LaunchFailure(MeasurementError):
    """A launch that produced no valid measurement, with its preserved
    evidence: ``record`` is the launch entry the session keeps (the error,
    the launch directory holding the diagnostics, the binary identity, the
    argv, the launched source hash, the fresh-record count), never a cycle
    count (ISSUES V1-21)."""

    def __init__(self, record):
        super().__init__(record.get("error", "launch failed"))
        self.record = dict(record)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_failure_diagnostics(launch_dir, message, diagnostics):
    """Serialize a failed launch beside whatever it wrote: the error, the
    structured diagnostics, the generated source, the device log read before
    the launch, every read after it and every attempt's process output."""
    launch_dir = Path(launch_dir)
    launch_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(diagnostics or {})
    (launch_dir / "error.txt").write_text(message + "\n")
    source = diagnostics.pop("device_source", None)
    if source:
        (launch_dir / "device.c").write_text(source)
    before = diagnostics.pop("device_log_before", None)
    if before is not None:
        (launch_dir / "device_log_before.txt").write_text(before)
    reads = diagnostics.pop("device_log_reads_after", None) or []
    for index, read in enumerate(reads, 1):
        (launch_dir / f"device_log_after_{index}.txt").write_text(read or "")
    attempts = diagnostics.get("attempts") or []
    for index, attempt in enumerate(attempts, 1):
        (launch_dir / f"process_output_{index}.txt").write_text(
            attempt.get("process_output") or ""
        )
    diagnostics["files"] = sorted(
        {p.name for p in launch_dir.iterdir()} | {"diagnostics.json"}
    )
    (launch_dir / "diagnostics.json").write_text(
        json.dumps(_json_ready(diagnostics), indent=1)
    )
    return diagnostics


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perturb(spec_text):
    """``NAME=FACTOR``: scale one calibration constant of the APU v1 cost
    module before compiling; returns (name, old, new)."""
    from allo.pim.costs import apu_v1 as apu_v1_costs

    name, factor = spec_text.split("=")
    old = getattr(apu_v1_costs, name)
    if not isinstance(old, (int, float)):
        raise ValueError(f"{name} is not a numeric cost constant")
    new = type(old)(old * float(factor))
    setattr(apu_v1_costs, name, new)
    return name, old, new


def _statistics(values):
    """Median, quartiles (linear interpolation), IQR and MAD of a sample."""
    if not values:
        return {}
    ordered = sorted(values)
    median = statistics.median(ordered)
    quartiles = (
        statistics.quantiles(ordered, n=4, method="inclusive")
        if len(ordered) > 1
        else [ordered[0]] * 3
    )
    mad = statistics.median(abs(v - median) for v in ordered)
    return {
        "samples": len(ordered),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "q1": quartiles[0],
        "q3": quartiles[2],
        "iqr": quartiles[2] - quartiles[0],
        "mad": mad,
    }


# ----------------------------------------------------------------------- #
# Lanes
# ----------------------------------------------------------------------- #


class TenonLane:
    """The shared pipeline's program for one case: compiled once, launched
    through the public callable."""

    def __init__(
        self,
        case,
        root,
        *,
        backend,
        layout=None,
        device_source=None,
        scratch=None,
        name="tenon",
    ):
        import allo
        from allo.pim.costs import apu_v1_cost
        from allo.pim.targets import build_apu_v1_target

        self.case = case
        self.root = root
        self.backend = backend
        # Per-launch evidence (generated source, actual outputs, identities,
        # failure diagnostics) is kept under the session's scratch directory,
        # outside the archive and outside the installed compiler.
        self.scratch = Path(scratch).resolve() if scratch else None
        self.name = name
        self.count = 0
        workload = _load_workload(root, case)
        target = build_apu_v1_target()
        kwargs = {"layout": layout} if layout is not None else {}
        started = time.perf_counter()
        self.compiled = allo.compile(
            workload,
            target,
            apu_v1_cost,
            backend=None if backend == "device" else backend,
            **kwargs,
        )
        self.compile_seconds = round(time.perf_counter() - started, 2)
        self.cost_fingerprint = apu_v1_cost.bind(target).fingerprint
        plan = getattr(self.compiled, "selected_plan", None)
        self.selection = {}
        if plan is not None:
            self.selection["selected_plan"] = plan.name
            self.selection["candidate_estimates"] = {
                e.plan.name: int(e.cycles) for e in self.compiled.candidate_estimates
            }
            decision = getattr(self.compiled, "tile_decision", None)
            self.selection["plan_selection"] = (
                decision.get("plan_selection") if isinstance(decision, dict) else None
            )
            realization = getattr(self.compiled, "realization", None)
            source = realization.device_source() if realization is not None else None
            self._program = getattr(self.compiled, "compiled", None)
        else:
            self.selection["native_vector_route"] = getattr(
                self.compiled, "route", None
            )
            source = getattr(self.compiled, "device_source", None)
            self._program = getattr(self.compiled, "lowered", None) or getattr(
                self.compiled, "compiled", None
            )
        self.device_source_sha256 = _sha256_text(source) if source else None
        self.device_source = source
        self.substituted_source = None
        if device_source is not None:
            # A labelled control: launch a historical generated program in
            # place of the selected one.  Not the shared selection.
            text = Path(device_source).read_text()
            holder = self._program if self._program is not None else self.compiled
            if not hasattr(holder, "device_source"):
                raise MeasurementError(
                    "this route holds no device source to substitute"
                )
            holder.device_source = text
            self.substituted_source = {
                "path": str(device_source),
                "sha256": _sha256_text(text),
            }
        estimate = (
            self.compiled.estimate() if hasattr(self.compiled, "estimate") else None
        )
        self.estimate_cycles = (
            int(estimate.cycles) if estimate is not None and estimate.cycles else None
        )
        self._bind_inputs()

    def _bind_inputs(self):
        import numpy as np

        inputs_dir = self.root / "apuv1" / "inputs" / self.case
        manifest = json.loads((inputs_dir / "manifest.json").read_text())
        arrays = manifest["arrays"]
        oracle_spec = arrays["oracle"]
        self.oracle = np.fromfile(
            inputs_dir / oracle_spec["file"], dtype=oracle_spec["dtype"]
        )
        self.oracle_sha256 = _sha256_file(inputs_dir / oracle_spec["file"])
        workload = _load_workload(self.root, self.case)
        phases = getattr(workload, "phases", None)
        if phases is not None:
            self.parameters = list(inspect.signature(phases[0].kernel).parameters)
            self.returned = tuple(phases[0].result_names)
            self.keyword_call = True
        else:
            self.parameters = list(inspect.signature(workload).parameters)
            self.returned = ()
            self.keyword_call = False
        self.arguments, self.in_place_outputs, self.input_sha256 = {}, [], {}
        for name in self.parameters:
            if name in arrays:
                spec = arrays[name]
                path = inputs_dir / spec["file"]
                self.arguments[name] = np.fromfile(path, dtype=spec["dtype"]).reshape(
                    spec["shape"]
                )
                self.input_sha256[name] = _sha256_file(path)
            else:
                # An in/out parameter without an archive (the matrix
                # multiply's result) starts as zeros of the oracle's shape.
                self.arguments[name] = np.zeros(
                    oracle_spec["shape"], dtype=oracle_spec["dtype"]
                )
                self.in_place_outputs.append(name)

    def describe(self):
        return {
            "lane": "tenon",
            "compile_seconds": self.compile_seconds,
            "cost_fingerprint": self.cost_fingerprint,
            **self.selection,
            "estimate_cycles": self.estimate_cycles,
            "device_source_sha256": self.device_source_sha256,
            "substituted_source": self.substituted_source,
            # The source actually built and launched by this lane.
            "launched_source_sha256": (
                self.substituted_source["sha256"]
                if self.substituted_source
                else self.device_source_sha256
            ),
            "inputs_sha256": self.input_sha256,
            "oracle_sha256": self.oracle_sha256,
            "outputs": list(self.returned) + self.in_place_outputs,
        }

    def _launch_dir(self):

        self.count = getattr(self, "count", 0) + 1
        scratch = getattr(self, "scratch", None)
        if scratch is None:
            scratch = self.scratch = Path(
                tempfile.mkdtemp(prefix=f"tenon-apu-v1-{self.case}-{self.name}-")
            ).resolve()
        launch_dir = Path(scratch) / f"{self.name}-launch-{self.count:03d}"
        launch_dir.mkdir(parents=True, exist_ok=False)
        return launch_dir

    def launch(self):
        import numpy as np

        from allo.pim.apu_v1_program import APUV1LaunchFailure

        launch_dir = self._launch_dir()
        launched_sha = (
            self.substituted_source["sha256"]
            if getattr(self, "substituted_source", None)
            else getattr(self, "device_source_sha256", None)
        )
        fresh = {name: value.copy() for name, value in self.arguments.items()}
        try:
            if self.keyword_call:
                result = self.compiled(**fresh)
            else:
                result = self.compiled(*(fresh[name] for name in self.parameters))
        except APUV1LaunchFailure as failure:
            # The runtime's structured diagnostics, serialized beside the
            # launch; the record carries no cycle count (ISSUES V1-21).
            diagnostics = _write_failure_diagnostics(
                launch_dir, f"{type(failure).__name__}: {failure}", failure.diagnostics
            )
            raise LaunchFailure(
                {
                    "error": f"{type(failure).__name__}: {failure}",
                    "reason": diagnostics.get("reason"),
                    "crun": None,
                    "outputs_match_oracle": None,
                    "launch_dir": str(launch_dir),
                    "argv": diagnostics.get("argv"),
                    "binary_sha256": diagnostics.get("binary_sha256"),
                    "binary_bytes": diagnostics.get("binary_bytes"),
                    "launched_source_sha256": diagnostics.get("device_source_sha256")
                    or launched_sha,
                    "expected_records": diagnostics.get("expected_records"),
                    "fresh_records_found": diagnostics.get("fresh_records_found"),
                    "board_lock": diagnostics.get("board_lock"),
                    "diagnostics": sorted(diagnostics.get("files") or []),
                }
            ) from failure
        except Exception as error:
            _write_failure_diagnostics(
                launch_dir,
                f"{type(error).__name__}: {error}",
                {"device_source": getattr(self, "device_source", None)},
            )
            raise LaunchFailure(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "crun": None,
                    "outputs_match_oracle": None,
                    "launch_dir": str(launch_dir),
                    "launched_source_sha256": launched_sha,
                }
            ) from error
        if result.kind != "device":
            raise MeasurementError(f"expected a device run, got {result.kind!r}")
        outputs = dict(result.extra.get("outputs", {}) or {})
        for name in self.in_place_outputs:
            outputs.setdefault(name, fresh[name])
        produced = [np.asarray(v).reshape(-1) for v in outputs.values()]
        exact = bool(produced) and all(
            p.size == self.oracle.size
            and np.array_equal(p.astype(self.oracle.dtype), self.oracle)
            for p in produced
        )
        extra = result.extra
        # Retained per launch: the generated source that was built, the
        # actual output arrays, the profiler text and the device log, and
        # the identities tying them together.
        source = extra.get("device_source") or getattr(self, "device_source", None)
        if source:
            (launch_dir / "device.c").write_text(source)
        outputs_dir = launch_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        outputs_sha256 = {}
        for name, value in outputs.items():
            array = np.ascontiguousarray(np.asarray(value))
            path = outputs_dir / f"{name}.{array.dtype.name}.bin"
            array.tofile(path)
            outputs_sha256[name] = _sha256_file(path)
        (launch_dir / "profile.txt").write_text(extra.get("profile_text") or "")
        device_log = result.stdout if hasattr(result, "stdout") else None
        (launch_dir / "device.log").write_text(device_log or "")
        launch = {
            "crun": result.cycles,
            "counters": extra.get("profile_counters"),
            "outputs_match_oracle": exact,
            "binary_sha256": extra.get("binary_sha256"),
            "binary_bytes": extra.get("binary_bytes"),
            "build_mode": extra.get("build_mode"),
            "board_lock": extra.get("board_lock"),
            "profile_text": extra.get("profile_text"),
            "device_log": device_log,
            "route": extra.get("native_vector_route"),
            "scalar_arc": extra.get("scalar_arc"),
            "launched_source_sha256": _sha256_text(source) if source else launched_sha,
            "outputs_sha256": outputs_sha256,
            "launch_dir": str(launch_dir),
        }
        (launch_dir / "launch.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        k: v
                        for k, v in launch.items()
                        if k not in ("profile_text", "device_log")
                    }
                ),
                indent=1,
            )
        )
        return launch


class ExpertLane:
    """An archived release binary of the campaign launched with its archived
    io through the production launcher (same lock, same profiler freshness
    proof).  ``lane_dir="expert"`` is the corrected expert Phoenix program;
    ``lane_dir="tenon"`` is the paper-era Tenon binary (the historical
    program as built and measured for the paper, a control for X-22)."""

    def __init__(self, case, root, *, lane_dir="expert", name="expert", scratch=None):
        import numpy as np

        self.case = case
        self.name = name
        self.dir = root / "apuv1" / lane_dir / case
        # Outputs are written per launch under a scratch directory outside
        # the archive; the archived io is read-only evidence.
        # Absolute: the launcher runs the binary with the launch directory
        # as its working directory, so relative output paths would be
        # resolved against themselves.
        self.scratch = (
            Path(scratch).resolve()
            if scratch
            else Path(tempfile.mkdtemp(prefix=f"tenon-apu-v1-{case}-{name}-"))
        )
        self.count = 0
        manifest_path = self.dir / "run-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no {name} lane for {case!r} at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        binary = self.dir / self.manifest["build"]["binary"]
        if not binary.is_file():
            candidates = sorted((self.dir / "build-evidence").glob("*.release.bin"))
            if not candidates:
                raise FileNotFoundError(f"no {name} binary for {case!r}")
            binary = candidates[0]
        self.binary = binary
        self.binary_sha256 = _sha256_file(binary)
        archived = self.manifest.get("provenance", {}).get("built_binary_sha256")
        if archived is None:
            evidence = self.dir / "build-evidence" / "build.json"
            if evidence.is_file():
                archived = (
                    json.loads(evidence.read_text()).get("binary", {}).get("sha256")
                )
        self.binary_matches_archive = archived == self.binary_sha256
        argv = self.manifest["argv"]
        self.inputs = [
            (item["role"], self.dir / item["path"]) for item in argv["inputs"]
        ]
        self.outputs = [
            (item["role"], self.dir / item["path"], item["dtype"], tuple(item["shape"]))
            for item in argv["outputs"]
        ]
        correctness = self.manifest["correctness"]
        oracle = (self.dir / correctness["oracle"]).resolve()
        self.oracle = np.fromfile(oracle, dtype=correctness["dtype"])
        self.oracle_sha256 = _sha256_file(oracle)
        self.input_sha256 = {role: _sha256_file(path) for role, path in self.inputs}

    def describe(self):
        return {
            "lane": self.name,
            "binary": str(self.binary),
            "binary_sha256": self.binary_sha256,
            "binary_matches_archive": self.binary_matches_archive,
            "device_source_sha256": self.manifest.get("expert", {}).get(
                "device_source_sha256"
            ),
            "phoenix_reference": self.manifest.get("provenance", {}).get(
                "phoenix_reference_commit"
            ),
            "inputs_sha256": self.input_sha256,
            "oracle_sha256": self.oracle_sha256,
            "timing_boundary": self.manifest.get("timing_boundary"),
        }

    def launch(self):
        import numpy as np

        from allo.pim.apu_v1_program import (
            launch_on_board,
            profile_counters,
            read_output_image,
        )

        self.count = getattr(self, "count", 0) + 1
        name = getattr(self, "name", "archived")
        scratch = getattr(self, "scratch", None)
        if scratch is None:
            scratch = Path(
                tempfile.mkdtemp(
                    prefix=f"tenon-apu-v1-{getattr(self, 'case', 'case')}-{name}-"
                )
            )
        scratch = self.scratch = Path(scratch).resolve()
        launch_dir = scratch / f"{name}-launch-{self.count:03d}"
        launch_dir.mkdir(parents=True, exist_ok=False)
        outputs = [
            (role, (launch_dir / Path(path).name).resolve(), dtype, shape)
            for role, path, dtype, shape in self.outputs
        ]
        archive = Path(self.dir).resolve()
        for _role, path, _dtype, _shape in outputs:
            if path == archive or archive in path.parents:
                raise MeasurementError(f"output path inside the archive: {path}")
        argv = [str(Path(self.binary).resolve())]
        argv.extend(str(Path(path).resolve()) for _role, path in self.inputs)
        argv.extend(str(path) for _role, path, _d, _s in outputs)
        binary_bytes = (
            Path(self.binary).stat().st_size if Path(self.binary).is_file() else None
        )
        try:
            text, profile, cycles = launch_on_board(
                argv, launch_dir, expected_records=1, timeout=900
            )
        except Exception as error:
            # The failed launch's diagnostics stay beside whatever it wrote
            # (the runtime's structured diagnostics when it raised them), and
            # the session's record carries the evidence location and the
            # identities, never a cycle count (ISSUES V1-21).
            diagnostics = _write_failure_diagnostics(
                launch_dir,
                f"{type(error).__name__}: {error}",
                getattr(error, "diagnostics", None) or {},
            )
            (launch_dir / "argv.json").write_text(json.dumps(argv, indent=1))
            raise LaunchFailure(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "reason": diagnostics.get("reason"),
                    "crun": None,
                    "outputs_match_oracle": None,
                    "launch_dir": str(launch_dir),
                    "argv": argv,
                    "binary_sha256": self.binary_sha256,
                    "binary_bytes": binary_bytes,
                    "expected_records": diagnostics.get("expected_records"),
                    "fresh_records_found": diagnostics.get("fresh_records_found"),
                    "board_lock": diagnostics.get("board_lock"),
                    "diagnostics": sorted(
                        set(diagnostics.get("files") or []) | {"argv.json"}
                    ),
                }
            ) from error
        produced = [
            read_output_image(path, role, dtype, shape).reshape(-1)
            for role, path, dtype, shape in outputs
        ]
        exact = all(
            p.size == self.oracle.size
            and np.array_equal(p.astype(self.oracle.dtype), self.oracle)
            for p in produced
        )
        # Every launch's actual outputs stay in its directory with the device
        # log, the profiler text and the identities (round 137; successful
        # launches were removed before, leaving no per-launch evidence).
        (launch_dir / "device.log").write_text(text or "")
        (launch_dir / "profile.txt").write_text(profile or "")
        (launch_dir / "argv.json").write_text(json.dumps(argv, indent=1))
        outputs_sha256 = {role: _sha256_file(path) for role, path, _d, _s in outputs}
        from allo.pim.apu_v1_program import apu_v1_lock_path

        launch = {
            "crun": cycles[0],
            "counters": profile_counters(profile),
            "outputs_match_oracle": exact,
            "binary_sha256": self.binary_sha256,
            "binary_bytes": binary_bytes,
            # The launch ran under the production launcher's board lock
            # (held by launch_on_board); recorded per launch as the Tenon
            # lane's runtime records it (round 142).
            "board_lock": str(apu_v1_lock_path()),
            "profile_text": profile,
            "device_log": text,
            "outputs_sha256": outputs_sha256,
            "launch_dir": str(launch_dir),
        }
        (launch_dir / "launch.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        k: v
                        for k, v in launch.items()
                        if k not in ("profile_text", "device_log")
                    }
                ),
                indent=1,
            )
        )
        return launch


# ----------------------------------------------------------------------- #
# Protocol
# ----------------------------------------------------------------------- #


def attempt_launch(lane) -> dict:
    """One launch of a lane as the session records it: the lane's launch
    entry, or, for a failed launch, its preserved evidence (``LaunchFailure``
    record, or the error alone when a lane raised without one) with no cycle
    count.  The failed launch is evidence; it is never retried here."""
    try:
        return lane.launch()
    except LaunchFailure as failure:
        return dict(failure.record)
    except Exception as error:  # the failed launch is evidence
        return {"error": f"{type(error).__name__}: {error}", "crun": None}


def _control_lane_name(path) -> str:
    return "control:" + Path(path).stem


def _paired_ratios(launches, lane, reference="tenon"):
    """Same-round ``lane / reference`` ratios over valid measured launches."""

    pairs = {}
    for l in launches:
        if (
            l.get("kind") == "measured"
            and not l.get("error")
            and l.get("outputs_match_oracle")
            and l.get("crun")
        ):
            pairs.setdefault(l["round"], {})[l["lane"]] = l["crun"]
    return [
        p[lane] / p[reference]
        for _round, p in sorted(pairs.items())
        if lane in p and reference in p
    ]


def measure_case(
    case,
    root,
    *,
    backend,
    measured,
    expert,
    layout,
    device_source,
    label,
    control_sources=(),
    historical_binary=False,
    scratch=None,
):
    import allo  # noqa: F401  (the lanes import the pipeline)
    from allo.pim.apu_v1_program import (
        apu_v1_board_lock,
        ledag_scratch_headroom_problem,
    )

    if measured <= 0:
        raise MeasurementError(f"measured launches must be positive, got {measured}")
    # A caller-supplied scratch directory is shared by every case of the
    # session, and the lanes name their launch directories per lane and
    # launch number only, so each case takes its own subdirectory (round 85,
    # ISSUES X-38: the second case's expert and archived launches collided
    # with the first case's and were lost to ``FileExistsError``).
    scratch = (
        Path(scratch).resolve() / case
        if scratch
        else Path(tempfile.mkdtemp(prefix=f"tenon-apu-v1-session-{case}-"))
    )

    record = {
        "case": case,
        "label": label
        or (
            "historical_program"
            if device_source
            else ("forced:" + layout if layout else "default_selection")
        ),
        "backend": backend,
        "paper_expert_cycles": PAPER[case][0],
        "paper_tenon_cycles": PAPER[case][1],
        "protocol": {
            "warmups_per_lane": 1,
            "measured_per_lane": measured,
            "interleaved_with_expert": bool(expert),
            "statistic": "median crun of the profiler total record (one APUC)",
            "order": "lanes in listed order on odd rounds, reversed on even rounds",
        },
        "lanes": {},
        "launches": [],
        "problems": [],
    }
    tenon = TenonLane(
        case,
        root,
        backend=backend,
        layout=layout,
        device_source=device_source,
        scratch=scratch,
    )
    record["lanes"]["tenon"] = tenon.describe()
    lanes = {"tenon": tenon}
    for path in control_sources:
        # A labelled control in the same rounds: the shared selection's
        # program with a different generated device source (a historical
        # or candidate program), never reported as the default selection.
        name = _control_lane_name(path)
        control = TenonLane(
            case,
            root,
            backend=backend,
            device_source=path,
            scratch=scratch,
            name="control-" + Path(path).stem,
        )
        description = control.describe()
        description["lane"] = name
        record["lanes"][name] = description
        lanes[name] = control
    if historical_binary:
        historical = ExpertLane(
            case, root, lane_dir="tenon", name="historical_binary", scratch=scratch
        )
        record["lanes"]["historical_binary"] = historical.describe()
        lanes["historical_binary"] = historical
    if expert:
        expert_lane = ExpertLane(case, root, scratch=scratch)
        record["lanes"]["expert"] = expert_lane.describe()
        lanes["expert"] = expert_lane
    record["protocol"]["lanes"] = list(lanes)
    record["scratch"] = str(scratch)
    for lane_name, lane in lanes.items():
        # An archived baseline whose binary is not the archived one is not
        # the comparison the paper made; the case is invalid as evidence.
        if getattr(lane, "binary_matches_archive", True) is False:
            record["problems"].append(
                f"{lane_name}: binary {lane.binary_sha256[:16]} does not match the archive"
            )
    if backend != "device":
        return record

    class _ScratchLost(Exception):
        pass

    def run(lane_name, kind, round_index):
        # The production launcher refuses an attempt whose scratch headroom
        # is gone (each refusal is a retained failure with no device
        # process).  The session stops at the first attempt the launcher
        # would refuse instead of attempting every remaining launch against
        # a host that cannot read the device log (round 147, ISSUES X-47):
        # the partial session is preserved and marked incomplete, no lane is
        # called after the loss, and the pooler sees an incomplete row.
        problem = ledag_scratch_headroom_problem()
        if problem is not None:
            record["incomplete"] = {
                "reason": "scratch_headroom_lost",
                "problem": problem,
                "not_attempted": {
                    "lane": lane_name,
                    "kind": kind,
                    "round": round_index,
                },
                "launches_recorded": len(record["launches"]),
            }
            record["problems"].append(
                f"session stopped before {lane_name} {kind} {round_index}: {problem}"
            )
            raise _ScratchLost(problem)
        launch = attempt_launch(lanes[lane_name])
        launch.update(lane=lane_name, kind=kind, round=round_index)
        record["launches"].append(launch)
        if launch.get("error"):
            record["problems"].append(
                f"{lane_name} {kind} {round_index}: {launch['error']}"
            )
        elif not launch.get("outputs_match_oracle"):
            record["problems"].append(
                f"{lane_name} {kind} {round_index}: outputs differ from the oracle"
            )
        return launch

    order = list(lanes)
    try:
        with apu_v1_board_lock() as lock_path:
            record["board_lock"] = str(lock_path)
            for lane_name in order:
                run(lane_name, "warmup", 0)
            for round_index in range(1, measured + 1):
                sequence = order if round_index % 2 == 1 else list(reversed(order))
                for lane_name in sequence:
                    run(lane_name, "measured", round_index)
    except _ScratchLost:
        pass  # recorded in record["incomplete"] and record["problems"]

    # Statistics over valid measured launches only (an invalid launch is
    # recorded but never contributes a cycle count).
    for lane_name in lanes:
        valid = [
            l["crun"]
            for l in record["launches"]
            if l["lane"] == lane_name
            and l["kind"] == "measured"
            and not l.get("error")
            and l.get("outputs_match_oracle")
            and l.get("crun")
        ]
        record["lanes"][lane_name]["measured_crun"] = valid
        record["lanes"][lane_name]["statistics"] = _statistics(valid)
        counters = [
            l["counters"]
            for l in record["launches"]
            if l["lane"] == lane_name and l["kind"] == "measured" and l.get("counters")
        ]
        if counters:
            record["lanes"][lane_name]["counter_medians"] = {
                key: statistics.median(c[key] for c in counters if key in c)
                for key in sorted({k for c in counters for k in c})
            }
    for lane_name in lanes:
        samples = (record["lanes"][lane_name].get("statistics") or {}).get("samples", 0)
        if samples < measured:
            record["problems"].append(
                f"{lane_name}: {samples} valid measured launches of {measured}"
            )
    tenon_stats = record["lanes"]["tenon"].get("statistics") or {}
    if tenon_stats.get("median") is not None:
        median = tenon_stats["median"]
        record["comparison"] = {
            "tenon_median_crun": median,
            "paper_tenon_cycles": PAPER[case][1],
            "ratio_to_paper_tenon": median / PAPER[case][1],
            "paper_expert_cycles": PAPER[case][0],
            "ratio_to_paper_expert": median / PAPER[case][0],
            "performance_evidence": (
                "valid" if not record["problems"] else "invalid (see problems)"
            ),
        }
        comparisons = {}
        for lane_name in lanes:
            if lane_name == "tenon":
                continue
            lane_median = record["lanes"][lane_name].get("statistics", {}).get("median")
            if not lane_median:
                continue
            ratios = _paired_ratios(record["launches"], lane_name)
            comparisons[lane_name] = {
                "median_crun": lane_median,
                "over_tenon_medians": lane_median / median,
                "paired_round_ratios_over_tenon": ratios,
                "paired_ratio_statistics": _statistics(ratios),
                "ratio_to_paper_tenon": lane_median / PAPER[case][1],
            }
        if comparisons:
            record["comparison"]["lanes"] = comparisons
        if "expert" in comparisons:
            expert_median = comparisons["expert"]["median_crun"]
            ratios = comparisons["expert"]["paired_round_ratios_over_tenon"]
            record["comparison"].update(
                {
                    "expert_median_crun": expert_median,
                    "expert_over_tenon_medians": expert_median / median,
                    "paired_round_ratios_expert_over_tenon": ratios,
                    "paired_ratio_statistics": _statistics(ratios),
                    "paper_expert_over_tenon": PAPER[case][0] / PAPER[case][1],
                }
            )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts")
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument(
        "--expert", action="store_true", help="interleave the expert lane"
    )
    parser.add_argument("--backend", default="device", choices=("device", "virtual"))
    parser.add_argument("--layout", help="force a plan (matrix_multiply)")
    parser.add_argument(
        "--device-source", help="launch this generated source instead (control)"
    )
    parser.add_argument(
        "--control-source",
        action="append",
        default=[],
        help="add an interleaved control lane launching this generated source "
        "(repeatable; lane name control:<file stem>)",
    )
    parser.add_argument(
        "--historical-binary",
        action="store_true",
        help="add an interleaved lane launching the archived paper-era Tenon binary",
    )
    parser.add_argument("--label")
    parser.add_argument("--perturb", help="CONST=FACTOR on allo.pim.costs.apu_v1")
    parser.add_argument(
        "--scratch",
        help="directory for per-launch outputs and failed-launch diagnostics "
        "(default: a new temporary directory outside the archive)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.cases:
        parser.error("--cases must name at least one case")
    if args.measured <= 0:
        parser.error("--measured must be positive")
    unknown = [case for case in args.cases if case not in CASES]
    if unknown:
        parser.error(f"unknown cases: {unknown}; known: {list(CASES)}")
    import allo

    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "allo": allo.__file__,
        "perturbation": None,
        "records": [],
    }
    failures = 0
    try:
        root = _artifacts_root(args.artifacts)
        if args.perturb:
            name, old, new = _perturb(args.perturb)
            session["perturbation"] = {"constant": name, "stock": old, "perturbed": new}
    except Exception as error:
        session["error"] = f"{type(error).__name__}: {error}"
        Path(args.out).write_text(json.dumps(session, indent=1, default=str))
        print(session["error"], file=sys.stderr)
        return 1
    session["artifacts"] = str(root)
    for case in args.cases:
        try:
            record = measure_case(
                case,
                root,
                backend=args.backend,
                measured=args.measured,
                expert=args.expert,
                layout=args.layout,
                device_source=args.device_source,
                label=args.label,
                control_sources=args.control_source,
                historical_binary=args.historical_binary,
                scratch=args.scratch,
            )
        except Exception as error:  # a failed case is evidence too
            record = {
                "case": case,
                "error": f"{type(error).__name__}: {error}",
                "problems": [],
            }
        if record.get("error") or record.get("problems"):
            failures += 1
        session["records"].append(record)
        if record.get("incomplete"):
            # The host's scratch is gone: the remaining cases are not
            # attempted (every attempt would be refused); the session is
            # incomplete and says which cases it never reached.
            remaining = args.cases[args.cases.index(case) + 1 :]
            session["incomplete"] = dict(
                record["incomplete"], case=case, cases_not_attempted=remaining
            )
            print(
                f"session incomplete: {record['incomplete']['problem']}; "
                f"cases not attempted: {remaining}",
                file=sys.stderr,
                flush=True,
            )
        summary = {
            k: record.get(k)
            for k in ("case", "label", "error", "problems", "comparison")
            if k in record
        }
        summary["lanes"] = {
            name: {
                k: v
                for k, v in lane.items()
                if k
                in (
                    "selected_plan",
                    "native_vector_route",
                    "statistics",
                    "counter_medians",
                    "device_source_sha256",
                    "binary_sha256",
                )
            }
            for name, lane in (record.get("lanes") or {}).items()
        }
        print(json.dumps(summary, default=str), flush=True)
        if session.get("incomplete"):
            break
    if session.get("incomplete"):
        session["status"] = (
            f"incomplete (scratch headroom lost during {session['incomplete']['case']}; "
            f"{len(session['incomplete']['cases_not_attempted'])} case(s) not attempted)"
        )
    else:
        session["status"] = "pass" if not failures else f"fail ({failures} case(s))"
    Path(args.out).write_text(json.dumps(session, indent=1, default=str))
    print(session["status"], flush=True)
    # A failed execution, a wrong output, a violated invariant or an
    # incomplete session is a nonzero exit; every record is preserved in --out.
    return 0 if not failures and not session.get("incomplete") else 1


if __name__ == "__main__":
    sys.exit(main())
