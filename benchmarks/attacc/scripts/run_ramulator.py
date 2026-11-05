# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time the compiler's selected AttAcc program in Ramulator2-AttAcc.

EVALUATION ONLY, external dependencies, not part of the installed package.
The result is simulator timing (``kind: "simulator-timing-only"``): the
simulator has no data path and produces no numerical output.

Pipeline:

1. ``allo.compile(<@allo source>, build_attacc_target(), attacc_cost,
   backend="virtual", reduction_order="reassociate")`` selects the tiling
   through the shared pipeline (frontend, matcher, candidate enumeration,
   cost ranking, materialization).  The script keeps the compiled artifact
   and records the selection's provenance: compiler revision, the ranking
   (candidates, chosen position, ties, tie-break rule), the matched
   operations, the selected program's manifest and its estimate.
2. The AttAcc command trace of that tiling is written by DCC's trace
   generator (``attacc_trace_gen/gen_trace_AttAcc_{GEMV,ATTN}.py``), imported
   at run time from a user-supplied DCC checkout.  DCC
   (github.com/SPIN-Research-Group/DCC; the study used commit
   ``212f86ea7f08da059836e3a1311d3a11afae8dbe``) has no license, so Tenon does
   not vendor or re-implement the trace emitter.  The trace is checked
   against the selected program by ``benchmarks.attacc.trace_check``: opcode
   totals, the command ordering (the generator's phase structure, for GEMV
   and attention traces) and the decoded address mapping (channels, banks,
   matrix columns, vector positions) must be the program's; a mismatch is a
   failure, never a warning.
3. ``ramulator2-AttAcc`` (Ramulator 2.0 with the AttAcc HBM3-PIM model, MIT;
   built by DCC's ``Ramulator2/build.sh`` or by
   ``build_ramulator_attacc.sh`` next to this script) runs the trace in a
   private temporary directory, natively or inside a Docker image.  The
   binary's and library's digests, the trace's and configuration's digests,
   the command line and the raw output are recorded; ``--artifacts``
   retains the trace, the configuration and the output per run.

Cycles are ``memory_system_cycles``; milliseconds use the ``HBM3_5.2Gbps``
clock (tCK 0.769 ns, DCC ``predictor.get_running_time_ms``), the paper's
unit.

Selection labels (``selection`` in every record):

* ``shared-pipeline``: the normal ``allo.compile`` selection (the only
  evidence of selection quality);
* ``forced-tiling``: ``--tiling``, a controlled reference or ablation built
  directly from a given tiling, never selection evidence;
* ``fixed-form-baseline``: ``--fixed-baseline``, the AttAcc fixed tiling as
  the paper's baseline column defines it (DCC's rule: the minimum simulated
  time over the fixed-form candidates, rows on ``min(16, rows)`` channels,
  one bank group's banks on the reduction, the bank groups on the outputs),
  every candidate simulated and recorded; a controlled reference.

Environment:

* ``ATTACC_DCC_ROOT``: the DCC ``src/src`` directory (contains
  ``attacc_trace_gen/`` and ``Ramulator2/``).
* ``ATTACC_RAMULATOR``: the ``ramulator2-AttAcc`` binary (default
  ``$ATTACC_DCC_ROOT/Ramulator2/ramulator2-AttAcc``); its ``libramulator.so``
  is expected in ``AttAcc`` next to the binary, as DCC builds it, or on
  ``ATTACC_RAMULATOR_LIBDIR``.
* ``ATTACC_RAMULATOR_IMAGE``: optional Docker image to run the binary in;
  the DCC tree is bind-mounted read-only.

Usage::

    python -m benchmarks.attacc.scripts.run_ramulator --row qkv_b1 --fixed-baseline \
        --artifacts out.artifacts --out out.json
    python -m benchmarks.attacc.scripts.run_ramulator --gemv 8 5120 384
    python -m benchmarks.attacc.scripts.run_ramulator --attention 8 128 480
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import trace_check

TCK_NS = 0.769  # HBM3_5.2Gbps; DCC predictor.get_running_time_ms

# The paper's kernel rows (tables/eval-attacc-kernels.tex): GPT-13B per
# device, 8 heads at batch 1 and 32 head instances at batch 4; the score
# rows are the mean over the 248 decode-step lengths 144..4096 step 16.
PAPER_ROWS = {
    "qkv_b1": ("gemv", [(8, 5120, 384)]),
    "qkv_b4": ("gemv", [(32, 5120, 384)]),
    "proj_b1": ("gemv", [(8, 128, 5120)]),
    "proj_b4": ("gemv", [(32, 128, 5120)]),
    "score_b1": ("attention", [(8, 128, length) for length in range(144, 4096 + 1, 16)]),
    "score_b4": ("attention", [(32, 128, length) for length in range(144, 4096 + 1, 16)]),
}


class ExternalDependencyMissing(RuntimeError):
    pass


class TraceMismatch(RuntimeError):
    pass


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    """A JSON-safe copy of a ranking attribute (placements and other objects
    become their ``str``)."""
    return json.loads(json.dumps(value, default=str))


def _git(path, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dcc_root() -> Path:
    root = os.environ.get("ATTACC_DCC_ROOT")
    if not root or not (Path(root) / "attacc_trace_gen").is_dir():
        raise ExternalDependencyMissing(
            "set ATTACC_DCC_ROOT to a DCC checkout's src/src directory "
            "(github.com/SPIN-Research-Group/DCC; it contains attacc_trace_gen/)"
        )
    return Path(root).resolve()


def _simulator(dcc_root: Path) -> Path:
    binary = Path(
        os.environ.get(
            "ATTACC_RAMULATOR", dcc_root / "Ramulator2" / "ramulator2-AttAcc"
        )
    ).resolve()
    if not binary.is_file():
        raise ExternalDependencyMissing(
            f"ramulator2-AttAcc not found at {binary}; build it with DCC's "
            "Ramulator2/build.sh or benchmarks/attacc/scripts/build_ramulator_attacc.sh, "
            "or set ATTACC_RAMULATOR"
        )
    return binary


def _libdir(binary: Path) -> Path:
    return Path(os.environ.get("ATTACC_RAMULATOR_LIBDIR", binary.parent / "AttAcc")).resolve()


def environment_record(dcc_root: Path, binary: Path) -> dict:
    """Identity of the compiler, the simulator and the external generator."""
    import allo

    allo_dir = Path(allo.__file__).resolve().parent.parent
    libdir = _libdir(binary)
    library = libdir / "libramulator.so"
    generators = {}
    for name in ("attacc_trace_gen/gen_trace_AttAcc_GEMV.py", "attacc_trace_gen/gen_trace_AttAcc_ATTN.py", "generate_configs.py"):
        path = dcc_root / name
        if path.is_file():
            generators[name] = _sha256(path)
    return {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "compiler": {
            "allo": str(allo_dir),
            "git_revision": _git(allo_dir, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(allo_dir, "status", "--porcelain", "--untracked-files=no")),
        },
        "simulator": {
            "name": "Ramulator2-AttAcc (external, MIT; AttAcc HBM3-PIM model)",
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "library": str(library) if library.is_file() else None,
            "library_sha256": _sha256(library) if library.is_file() else None,
            "image": os.environ.get("ATTACC_RAMULATOR_IMAGE"),
            "tck_ns": TCK_NS,
        },
        "trace_generator": {
            "dcc_root": str(dcc_root),
            "dcc_git_revision": _git(dcc_root, "rev-parse", "HEAD"),
            "files_sha256": generators,
            "license": "DCC has no license file; imported at run time, not vendored",
        },
    }


# --------------------------------------------------------------------- #
# Compilation through the shared pipeline
# --------------------------------------------------------------------- #


def compile_selected(family: str, shape) -> dict:
    """Compile the kernel through ``allo.compile`` and keep the artifact.

    Returns the compiled callable, the selected ``AttAccProgram``, its
    estimate and the selection provenance (the placement ranking's
    candidates, chosen position, ties and tie-break, the matched operations,
    the selected decision string and the program manifest)."""
    import allo
    from allo.pim.costs import attacc_cost
    from allo.pim.targets import build_attacc_target

    from ..workloads import attention_region, gemv_region

    region = gemv_region(*shape) if family == "gemv" else attention_region(*shape)
    compiled = allo.compile(
        region,
        build_attacc_target(),
        attacc_cost,
        backend="virtual",
        reduction_order="reassociate",
    )
    (program,) = compiled.compiled.layout_ctx.programs
    (kernel,) = compiled.compiled.placement_ranking.kernels
    chosen = kernel.chosen_position
    selected = next((s for s in kernel.scores if s[0] == chosen), kernel.scores[0] if kernel.scores else None)
    provenance = {
        "route": "allo.compile(region, build_attacc_target(), attacc_cost, backend='virtual', reduction_order='reassociate')",
        "matched_operations": [m.target_op_name for m in compiled.trace.matches],
        "reduction_order": compiled.reduction_order,
        "candidates": len(kernel.candidates) if hasattr(kernel.candidates, "__len__") else kernel.candidates,
        "scored": len(kernel.scores),
        "probe_count": kernel.probe_count,
        "chosen_position": chosen,
        "selected": {"position": selected[0], "ranked_cycles": selected[1], "decision": selected[2]} if selected else None,
        "best_three": [{"position": s[0], "ranked_cycles": s[1], "decision": s[2]} for s in kernel.scores[:3]],
        "cost_ties": _jsonable(kernel.cost_ties),
        "tie_break": _jsonable(kernel.tie_break),
        "fallback": None if kernel.fallback is None else str(kernel.fallback)[:200],
        "materialized": None if kernel.materialized is None else str(kernel.materialized)[:200],
        "rejections": len(kernel.rejections) if hasattr(kernel.rejections, "__len__") else _jsonable(kernel.rejections),
        "layout_extra_tiling": compiled.compiled.layout.extra.get("attacc_tiling"),
        "program": program.manifest(),
    }
    return {
        "compiled": compiled,
        "program": program,
        "estimate_cycles": compiled.estimate().cycles,
        "provenance": _jsonable(provenance),
    }


def forced_program(selected_program, tiling):
    """A controlled reference: the program of a given tiling for the same
    kernel (``attacc_program`` validates legality).  Not selection evidence."""
    from allo.pim.attacc_tiling import attacc_program

    return attacc_program(selected_program.kernel, tiling, selected_program.geometry)


def fixed_form_tilings(selected_program) -> list:
    """The AttAcc fixed-form candidates of the kernel, as the paper's
    baseline column selects them (DCC ``predictor.py``: rows on
    ``min(16, rows)`` channels, one bank group's banks on the reduction, the
    bank groups on the outputs; the baseline time is the minimum over
    them)."""
    from allo.pim.attacc_tiling import attacc_tiling_space

    kernel, geometry = selected_program.kernel, selected_program.geometry
    rows_channels = min(geometry.channels, kernel.rows)
    return [
        tiling
        for tiling in attacc_tiling_space(kernel, geometry)
        if tiling.dims[0][0] == rows_channels
        and tiling.dims[1][1] == geometry.banks_per_group
        and tiling.dims[2][1] == geometry.bank_groups
    ]


# --------------------------------------------------------------------- #
# Trace and simulator
# --------------------------------------------------------------------- #


def write_trace(dcc_root: Path, family: str, tiling, workdir: Path) -> Path:
    sys.path.insert(0, str(dcc_root))
    try:
        trace = workdir / "kernel.trace"
        if family == "gemv":
            generator = importlib.import_module("attacc_trace_gen.gen_trace_AttAcc_GEMV")
            generator.set_mode_gemv("inference")
            generator.run_GEMV([list(axis) for axis in tiling], str(trace))
        else:
            generator = importlib.import_module("attacc_trace_gen.gen_trace_AttAcc_ATTN")
            generator.set_mode_attn("inference")
            generator.run_attention([list(axis) for axis in tiling], str(trace))
    finally:
        sys.path.remove(str(dcc_root))
    return trace


def run_simulator(binary: Path, dcc_root: Path, yaml_path: Path, workdir: Path) -> tuple[str, list[str]]:
    # The AttAcc trace recorder writes ./temp/log/attacc_bank/cmd.log.
    (workdir / "temp" / "log" / "attacc_bank").mkdir(parents=True, exist_ok=True)
    libdir = _libdir(binary)
    command = [str(binary), "-f", str(yaml_path)]
    image = os.environ.get("ATTACC_RAMULATOR_IMAGE")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    if image:
        mounts = {str(dcc_root): "ro", str(workdir): "rw"}
        for path in (binary.parent, libdir):
            if not str(path).startswith(str(dcc_root)):
                mounts[str(path)] = "ro"
        docker = ["docker", "run", "--rm", "-w", str(workdir), "-e", f"LD_LIBRARY_PATH={libdir}"]
        for path, mode in mounts.items():
            docker += ["-v", f"{path}:{path}:{mode}"]
        command = docker + [image] + command
    result = subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ramulator2-AttAcc failed ({result.returncode}): {shlex.join(command)}\n{result.stderr[-2000:]}"
        )
    return result.stdout, command


def simulate_program(dcc_root: Path, binary: Path, family: str, program, *, keep_dir=None, artifacts=None) -> dict:
    """Write the program's trace, check it against the program, simulate it."""
    workdir = Path(tempfile.mkdtemp(prefix="tenon-attacc-"))
    try:
        trace = write_trace(dcc_root, family, program.tiling.dcc_tiling(), workdir)
        check = trace_check.check_trace(trace, program)
        if not check["totals_match"]:
            raise TraceMismatch(f"trace opcode totals {check['totals']} differ from the compiled program {check['declared']}")
        if check.get("ordering_checked") and not (check["ordering_match"] and check["mapping_match"]):
            raise TraceMismatch(f"trace structure differs from the compiled program: {json.dumps({k: check[k] for k in ('first_difference', 'mapping')})}")
        yaml_path = trace.with_suffix(".yaml")
        started = time.perf_counter()
        stdout, command = run_simulator(binary, dcc_root, yaml_path, workdir)
        elapsed = time.perf_counter() - started
        found = re.search(r"memory_system_cycles:\s*(\d+)", stdout)
        if found is None:
            raise RuntimeError("ramulator2-AttAcc printed no memory_system_cycles")
        cycles = int(found.group(1))
        record = {
            "tiling": program.tiling.dcc_tiling(),
            "counts": program.counts.manifest(),
            "memory_system_cycles": cycles,
            "milliseconds": cycles * TCK_NS / 1e6,
            "trace_check": check,
            "trace_sha256": _sha256(trace),
            "trace_lines": sum(1 for _ in trace.open()),
            "config_sha256": _sha256(yaml_path),
            "command": shlex.join(command),
            "simulator_wall_s": elapsed,
            "stdout_tail": stdout[-600:],
        }
        if artifacts is not None:
            artifacts = Path(artifacts)
            artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace, artifacts / "kernel.trace")
            shutil.copy2(yaml_path, artifacts / "kernel.yaml")
            (artifacts / "stdout.txt").write_text(stdout)
            record["artifacts"] = str(artifacts)
        if keep_dir:
            record["workdir"] = str(workdir)
        return record
    finally:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def time_shape(dcc_root, binary, family, shape, *, tiling=None, fixed_baseline=False, artifacts=None, keep=False, skip_selected=False) -> list[dict]:
    """Records for one kernel shape: the shared-pipeline selection and,
    when asked, the forced tiling and the fixed-form baseline."""
    compiled = compile_selected(family, shape)
    program = compiled["program"]
    records = []
    base = {"kind": "simulator-timing-only", "family": family, "shape": list(shape), "kernel": program.kernel.manifest()}
    art = Path(artifacts) if artifacts else None
    label = f"{family}_{'x'.join(str(d) for d in shape)}"
    if not skip_selected:
        rec = simulate_program(dcc_root, binary, family, program, keep_dir=keep, artifacts=art / label / "selected" if art else None)
        records.append({**base, "selection": "shared-pipeline", "analytical_estimate_cycles": compiled["estimate_cycles"], "provenance": compiled["provenance"], **rec})
    if tiling is not None:
        forced = forced_program(program, tiling)
        rec = simulate_program(dcc_root, binary, family, forced, keep_dir=keep, artifacts=art / label / "forced" if art else None)
        records.append({**base, "selection": "forced-tiling", "note": "controlled reference/ablation of a given tiling; not shared-pipeline selection evidence", "analytical_estimate_cycles": None, **rec})
    if fixed_baseline:
        candidates = fixed_form_tilings(program)
        if not candidates:
            raise RuntimeError("no fixed-form candidate for this kernel")
        runs = []
        for index, candidate in enumerate(candidates):
            cand_program = forced_program(program, candidate)
            rec = simulate_program(dcc_root, binary, family, cand_program, keep_dir=keep, artifacts=art / label / f"fixed-{index}" if art else None)
            runs.append(rec)
        best = min(runs, key=lambda r: r["memory_system_cycles"])
        records.append({
            **base,
            "selection": "fixed-form-baseline",
            "note": "the paper's fixed AttAcc tiling baseline: DCC predictor.py's rule, the minimum simulated time over the fixed-form candidates; a controlled reference",
            "analytical_estimate_cycles": None,
            "tiling": best["tiling"],
            "memory_system_cycles": best["memory_system_cycles"],
            "milliseconds": best["milliseconds"],
            "fixed_form_candidates": [{k: r[k] for k in ("tiling", "memory_system_cycles", "milliseconds", "trace_sha256", "artifacts") if k in r} for r in runs],
        })
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shape = parser.add_mutually_exclusive_group(required=True)
    shape.add_argument("--gemv", nargs=3, type=int, metavar=("H", "K", "N"))
    shape.add_argument("--attention", nargs=3, type=int, metavar=("H", "D", "L"))
    shape.add_argument("--row", choices=sorted(PAPER_ROWS), help="a paper kernel row (all its shapes)")
    parser.add_argument("--tiling", help="also time this tiling (JSON) as a controlled reference")
    parser.add_argument("--fixed-baseline", action="store_true", help="also time the fixed-form baseline (every fixed-form candidate)")
    parser.add_argument("--skip-selected", action="store_true", help="do not time the shared-pipeline selection (references only)")
    parser.add_argument("--limit", type=int, help="only the first N shapes of a row")
    parser.add_argument("--artifacts", help="retain trace, configuration and output per run under this directory")
    parser.add_argument("--out", help="write the session record (JSON)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directories")
    args = parser.parse_args(argv)
    if args.row:
        family, shapes = PAPER_ROWS[args.row]
    else:
        family = "gemv" if args.gemv else "attention"
        shapes = [tuple(args.gemv or args.attention)]
    if args.limit:
        shapes = shapes[: args.limit]

    dcc_root = _dcc_root()
    binary = _simulator(dcc_root)
    session = environment_record(dcc_root, binary)
    session.update({"paper_row": args.row, "family": family, "runs": [], "failures": []})
    tiling = json.loads(args.tiling) if args.tiling else None
    for shape in shapes:
        try:
            records = time_shape(dcc_root, binary, family, shape, tiling=tiling, fixed_baseline=args.fixed_baseline, artifacts=args.artifacts, keep=args.keep, skip_selected=args.skip_selected)
        except (TraceMismatch, RuntimeError) as error:
            session["failures"].append({"shape": list(shape), "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {family} {shape}: {error}", file=sys.stderr, flush=True)
            continue
        session["runs"].extend(records)
        for rec in records:
            print(f"{family} {shape} {rec['selection']:20s} tiling={rec['tiling']} cycles={rec['memory_system_cycles']} ms={rec['milliseconds']:.6f}", flush=True)
    selected = [r for r in session["runs"] if r["selection"] == "shared-pipeline"]
    if selected:
        session["summary"] = {
            "shapes": len(selected),
            "mean_milliseconds_selected": sum(r["milliseconds"] for r in selected) / len(selected),
        }
        baselines = [r for r in session["runs"] if r["selection"] == "fixed-form-baseline"]
        if baselines:
            session["summary"]["mean_milliseconds_fixed_baseline"] = sum(r["milliseconds"] for r in baselines) / len(baselines)
    if args.out:
        Path(args.out).write_text(json.dumps(session, indent=1, default=str))
    else:
        print(json.dumps(session, indent=1, default=str))
    return 1 if session["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
