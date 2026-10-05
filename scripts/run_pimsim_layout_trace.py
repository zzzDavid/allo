#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and run the standalone Samsung PIMSimulator layout-trace probe.

The helper discovers ``PIMSIMULATOR_ROOT`` (or the repository's default
checkout), builds the external harness against the simulator's static library,
and runs from the simulator source tree so its normal configuration is used.
It never edits vendor sources.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


SAMSUNG_BANKS = 16
SAMSUNG_ROWS = 16_384
SAMSUNG_BURST_COLS = 32

SCRIPT_DIR = Path(__file__).resolve().parent
ALLO_ROOT = SCRIPT_DIR.parent
HARNESS_SOURCE = (
    ALLO_ROOT
    / "tests"
    / "pim"
    / "samsung_hbm_pim"
    / "layout_swizzle"
    / "pimsim_layout_trace.cc"
)
DEFAULT_PIMSIMULATOR_ROOT = SCRIPT_DIR.parents[1] / "simulators" / "PIMSimulator"
DEFAULT_BINARY_OVERRIDE = os.environ.get("TENON_PIMSIM_LAYOUT_BINARY")
# The installed vendor headers trigger these warnings even for an external
# one-file client; keep harness diagnostics useful.
HARNESS_BUILD_FLAGS = (
    "-O2",
    "-std=c++14",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-Wno-sign-compare",
    "-Wno-reorder",
)


@dataclass(frozen=True)
class TraceRecord:
    round: int
    bank: int
    row: int
    col: int


def _parse_uint(token: str, field: str, line_number: int) -> int:
    if token.startswith("-"):
        raise ValueError(f"line {line_number}: invalid {field} {token!r}")
    base = 16 if token.lower().startswith("0x") else 10
    try:
        value = int(token, base)
    except ValueError as error:
        raise ValueError(f"line {line_number}: invalid {field} {token!r}") from error
    if value < 0:
        raise ValueError(f"line {line_number}: invalid {field} {token!r}")
    return value


def parse_trace(path: Path) -> list[TraceRecord]:
    """Preflight the same compact grouped format consumed by the C++ probe."""

    records: list[TraceRecord] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw_line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(
                f"line {line_number}: expected exactly 'round bank row col'"
            )
        round_id, bank, row, col = (
            _parse_uint(token, field, line_number)
            for token, field in zip(fields, ("round", "bank", "row", "col"))
        )
        if bank >= SAMSUNG_BANKS:
            raise ValueError(f"line {line_number}: bank must be in [0, 16)")
        if row >= SAMSUNG_ROWS:
            raise ValueError(f"line {line_number}: row must be in [0, 16384)")
        if col >= SAMSUNG_BURST_COLS:
            raise ValueError(f"line {line_number}: col must be in [0, 32)")
        if records and round_id < records[-1].round:
            raise ValueError(
                f"line {line_number}: rounds must be grouped in nondecreasing order"
            )
        records.append(TraceRecord(round_id, bank, row, col))
    if not records:
        raise ValueError(f"trace has no requests: {path}")
    return records


def discover_pimsimulator_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if "PIMSIMULATOR_ROOT" in os.environ:
        return Path(os.environ["PIMSIMULATOR_ROOT"]).expanduser().resolve()
    return DEFAULT_PIMSIMULATOR_ROOT.resolve()


def static_library(root: Path) -> Path:
    return root / "libdramsim" / "libdramsim2.a"


def _compile_inputs(root: Path) -> list[Path]:
    """Return every file whose content can affect the external probe binary."""

    inputs = [HARNESS_SOURCE, static_library(root)]
    for directory in (root / "src", root / "lib", root / "tools"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".h", ".hh", ".hpp", ".inc"}:
                inputs.append(path)
    return sorted(set(inputs), key=str)


def default_binary() -> Path:
    return Path(f"/tmp/tenon-pimsim-layout-trace-{os.getuid()}")


def vendor_build_command(root: Path, jobs: int = 8) -> list[str]:
    return [
        "scons",
        "-C",
        str(root),
        "-j",
        str(jobs),
        "libdramsim/libdramsim2.a",
    ]


def build_command(
    root: Path,
    binary: Path,
    *,
    cxx: str = "g++",
    source: Path = HARNESS_SOURCE,
) -> list[str]:
    """Return the direct, shell-free external-harness compile command."""

    return [
        cxx,
        *HARNESS_BUILD_FLAGS,
        "-I",
        str(root / "src"),
        "-I",
        str(root / "lib"),
        "-I",
        str(root / "tools"),
        str(source),
        str(static_library(root)),
        "-pthread",
        "-o",
        str(binary),
    ]


def run_command(
    root: Path,
    binary: Path,
    trace: Path,
    output_dir: Path,
    *,
    single_channel: bool = False,
) -> list[str]:
    command = [
        str(binary),
        "--pimsim-root",
        str(root),
        "--output-dir",
        str(output_dir),
    ]
    if single_channel:
        command.append("--single-channel")
    command.append(str(trace))
    return command


def _needs_rebuild(binary: Path, inputs: Iterable[Path]) -> bool:
    if not binary.is_file():
        return True
    binary_mtime = binary.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > binary_mtime for path in inputs)


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="round/bank/row/col trace")
    parser.add_argument("--pimsim-root", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(DEFAULT_BINARY_OVERRIDE) if DEFAULT_BINARY_OVERRIDE else None,
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--single-channel", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight the trace and print commands without executing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.trace is None and not args.build_only:
        raise SystemExit("a trace is required unless --build-only is used")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")

    root = discover_pimsimulator_root(args.pimsim_root)
    binary = (
        args.binary.expanduser().resolve()
        if args.binary is not None
        else default_binary()
    )
    trace = args.trace.expanduser().resolve() if args.trace is not None else None
    if trace is not None:
        try:
            records = parse_trace(trace)
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid trace: {error}") from error
        rounds = 1 + sum(
            left.round != right.round for left, right in zip(records, records[1:])
        )
    else:
        records = []
        rounds = 0

    required = [
        root / "src" / "MultiChannelMemorySystem.h",
        root / "src" / "tests" / "KernelAddrGen.h",
        root / "ini" / "HBM2_samsung_2M_16B_x64.ini",
        root / "system_hbm_64ch.ini",
        HARNESS_SOURCE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "missing PIMSimulator input(s): " + ", ".join(str(path) for path in missing)
        )

    library = static_library(root)
    vendor_command = vendor_build_command(root, args.jobs)
    compile_command = build_command(root, binary, cxx=args.cxx)
    compile_needed = not library.is_file() or _needs_rebuild(
        binary, _compile_inputs(root) if library.is_file() else [HARNESS_SOURCE]
    )

    if args.dry_run:
        if not args.no_build:
            print("VENDOR_BUILD_COMMAND " + _format_command(vendor_command))
        if compile_needed and not args.no_build:
            print("HARNESS_BUILD_COMMAND " + _format_command(compile_command))
        elif compile_needed:
            print("HARNESS_BUILD_REQUIRED binary=" + str(binary))
        if trace is not None:
            print(
                f"TRACE_SUMMARY rounds={rounds} logical_requests={len(records)} "
                f"channels={1 if args.single_channel else 64}"
            )
            print(
                "HARNESS_RUN_COMMAND "
                + _format_command(
                    run_command(
                        root,
                        binary,
                        trace,
                        Path("/tmp/tenon-pimsim-layout-output"),
                        single_channel=args.single_channel,
                    )
                )
            )
        return 0

    if args.no_build and not library.is_file():
        raise SystemExit(f"vendor static library is missing: {library}")
    if not args.no_build:
        # SCons is incremental.  Running it even when the archive exists avoids
        # silently linking a stale library after vendor sources or flags change.
        subprocess.run(vendor_command, cwd=root, check=True)
        if not library.is_file():
            raise SystemExit(f"vendor build did not produce {library}")

    compile_inputs = _compile_inputs(root)
    if _needs_rebuild(binary, compile_inputs):
        if args.no_build:
            raise SystemExit(f"harness binary is missing or stale: {binary}")
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(compile_command, cwd=root, check=True)

    if args.build_only:
        print(f"PIM_LAYOUT_PROBE_BUILT binary={binary}")
        return 0

    assert trace is not None
    with tempfile.TemporaryDirectory(prefix="tenon-pimsim-layout-") as output_dir:
        completed = subprocess.run(
            run_command(
                root,
                binary,
                trace,
                Path(output_dir),
                single_channel=args.single_channel,
            ),
            cwd=root,
            check=False,
        )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
