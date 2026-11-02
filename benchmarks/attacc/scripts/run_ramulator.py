# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time the compiler's selected AttAcc program in Ramulator2-AttAcc.

EVALUATION ONLY, external dependencies, not part of the installed package.
The result is simulator timing (``kind: "simulator-timing-only"``): the
simulator has no data path and produces no numerical output.

Pipeline:

1. ``allo.compile(<@allo source>, build_attacc_target(), attacc_cost,
   backend="virtual", reduction_order="reassociate")`` selects the tiling;
   the script reads ``AttAccProgram.dcc_tiling()`` from the compiled
   context (or takes ``--tiling`` to time a given tiling).
2. The AttAcc command trace of that tiling is written by DCC's trace
   generator (``attacc_trace_gen/gen_trace_AttAcc_{GEMV,ATTN}.py``), imported
   at run time from a user-supplied DCC checkout.  DCC
   (github.com/SPIN-Research-Group/DCC; the study used commit
   ``212f86ea7f08da059836e3a1311d3a11afae8dbe``) has no license, so Tenon does
   not vendor or re-implement the trace emitter; the per-opcode counts of the
   trace are checked against the compiler's ``AttAccCounts``.
3. ``ramulator2-AttAcc`` (Ramulator 2.0 with the AttAcc HBM3-PIM model, MIT;
   built by DCC's ``Ramulator2/build.sh``) runs the trace in a private
   temporary directory, natively or inside a Docker image when the host glibc
   is older than the binary needs (the study's build needs GLIBC 2.34; it ran
   in the ``aim-simulator-build`` image, Ubuntu 24.04).

Environment:

* ``ATTACC_DCC_ROOT``: the DCC ``src/src`` directory (contains
  ``attacc_trace_gen/`` and ``Ramulator2/``).
* ``ATTACC_RAMULATOR``: the ``ramulator2-AttAcc`` binary (default
  ``$ATTACC_DCC_ROOT/Ramulator2/ramulator2-AttAcc``); its ``libramulator.so``
  is expected in ``../Ramulator2/AttAcc`` relative to the binary, as DCC builds
  it, or on ``ATTACC_RAMULATOR_LIBDIR``.
* ``ATTACC_RAMULATOR_IMAGE``: optional Docker image to run the binary in;
  the DCC tree is bind-mounted read-only.

Usage::

    python -m benchmarks.attacc.scripts.run_ramulator --gemv 8 5120 384
    python -m benchmarks.attacc.scripts.run_ramulator --attention 8 128 480
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Trace opcode -> AttAccCounts manifest key.
_TRACE_OPCODES = {
    "PIM_WR_GB": "PIM_WR_GB",
    "PIM_MAC_AB": "PIM_MAC_AB",
    "PIM_MV_SB": "PIM_MV_SB",
    "PIM_MV_GB": "PIM_MV_GB",
    "PIM_SFM": "PIM_SFM",
    "ST": "ST",
}


class ExternalDependencyMissing(RuntimeError):
    pass


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
        os.environ.get("ATTACC_RAMULATOR", dcc_root / "Ramulator2" / "ramulator2-AttAcc")
    ).resolve()
    if not binary.is_file():
        raise ExternalDependencyMissing(
            f"ramulator2-AttAcc not found at {binary}; build it with DCC's "
            "Ramulator2/build.sh or set ATTACC_RAMULATOR"
        )
    return binary


def compiled_program(family: str, shape):
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
    return program, compiled.estimate().cycles


def write_trace(dcc_root: Path, family: str, tiling, workdir: Path) -> Path:
    sys.path.insert(0, str(dcc_root))
    try:
        if family == "gemv":
            generator = importlib.import_module("attacc_trace_gen.gen_trace_AttAcc_GEMV")
            generator.set_mode_gemv("inference")
            trace = workdir / "kernel.trace"
            generator.run_GEMV(tiling, str(trace))
        else:
            generator = importlib.import_module("attacc_trace_gen.gen_trace_AttAcc_ATTN")
            generator.set_mode_attn("inference")
            trace = workdir / "kernel.trace"
            generator.run_attention(tiling, str(trace))
    finally:
        sys.path.remove(str(dcc_root))
    return trace


def trace_counts(trace: Path) -> dict:
    counts = collections.Counter()
    with trace.open() as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] in _TRACE_OPCODES:
                counts[_TRACE_OPCODES[fields[0]]] += 1
    return dict(counts)


def run_simulator(binary: Path, dcc_root: Path, yaml_path: Path, workdir: Path) -> str:
    # The AttAcc trace recorder writes ./temp/log/attacc_bank/cmd.log.
    (workdir / "temp" / "log" / "attacc_bank").mkdir(parents=True, exist_ok=True)
    libdir = Path(
        os.environ.get("ATTACC_RAMULATOR_LIBDIR", binary.parent / "AttAcc")
    ).resolve()
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
    result = subprocess.run(
        command, cwd=workdir, env=env, capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ramulator2-AttAcc failed ({result.returncode}): {shlex.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )
    return result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shape = parser.add_mutually_exclusive_group(required=True)
    shape.add_argument("--gemv", nargs=3, type=int, metavar=("H", "K", "N"))
    shape.add_argument("--attention", nargs=3, type=int, metavar=("H", "D", "L"))
    parser.add_argument("--tiling", help="time this tiling (JSON) instead of the selected one")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args(argv)
    family = "gemv" if args.gemv else "attention"
    dims = tuple(args.gemv or args.attention)

    dcc_root = _dcc_root()
    binary = _simulator(dcc_root)
    program, estimate = compiled_program(family, dims)
    tiling = json.loads(args.tiling) if args.tiling else program.dcc_tiling()
    if args.tiling:
        from allo.pim.attacc_tiling import attacc_program

        program = attacc_program(program.kernel, tiling, program.geometry)
        estimate = None

    workdir = Path(tempfile.mkdtemp(prefix="tenon-attacc-"))
    trace = write_trace(dcc_root, family, tiling, workdir)
    counts = trace_counts(trace)
    expected = dict(program.counts.commands())
    if counts != expected:
        raise RuntimeError(
            f"trace opcode counts {counts} differ from the compiled program {expected}"
        )
    stdout = run_simulator(binary, dcc_root, trace.with_suffix(".yaml"), workdir)
    found = re.search(r"memory_system_cycles:\s*(\d+)", stdout)
    if found is None:
        raise RuntimeError("ramulator2-AttAcc printed no memory_system_cycles")
    record = {
        "kind": "simulator-timing-only",
        "simulator": "Ramulator2-AttAcc (external)",
        "family": family,
        "shape": dims,
        "tiling": tiling,
        "memory_system_cycles": int(found.group(1)),
        "analytical_estimate_cycles": estimate,
        "trace_counts": counts,
        "workdir": str(workdir) if args.keep else None,
    }
    if not args.keep:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
