# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: public compile-and-call entry point.

`compile_workload` delegates to ``allo.compile(workload, target, cost)`` and
returns its ``CompiledCallable``. `compile_and_run` invokes the callable's
backend-role escape hatch and returns the `RunResult` unchanged. The suite does
not reconstruct customization, matching, autoscheduling, or backend dispatch.

It also provides the anti-fabrication provenance helpers (spec Answer 4): the
sim fingerprint / device identity, the tenon commit, and the run timestamp are
all STAMPED AFTER THE RUN here, never hardcoded in a workload or test.
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import subprocess

import allo
from allo.spmw_codegen import RunResult

from .cost import bind_cost


def compile_workload(target, workload, backend=None, host_moves=None, cost=None):
    """Compile through the public ``allo.compile`` interface.

    Samsung binds its standalone executable cost program. Targets not yet
    ported receive ``cost=None`` and use their existing physical backend path.
    """
    if cost is None:
        cost = bind_cost(target)
    return allo.compile(
        workload,
        target,
        cost,
        backend=backend,
        host_moves=host_moves,
    )


def compile_and_run(
    target, workload, backend=None, host_moves=None, cost=None, **inputs
) -> RunResult:
    """Compile publicly and invoke explicit backend-role arrays.

    ``backend=None`` selects the real target simulator/device;
    ``backend="virtual"`` is a cost-only dry run. Samsung's backend-role inputs
    intentionally differ from some source signatures (for covariance,
    ``A=cdata.T`` and ``B=cdata``), so this compiler-test helper invokes
    ``run_backend`` instead of weakening normal callable signature checking.
    """
    compiled = compile_workload(
        target,
        workload,
        backend=backend,
        host_moves=host_moves,
        cost=cost,
    )
    return compiled.run_backend(**inputs)


# --------------------------------------------------------------------- #
# Provenance (spec Answer 4: source / tenon_commit / timestamp, stamped AFTER
# the run; a reviewer re-runs `run_cmd` and re-derives `source`).
# --------------------------------------------------------------------- #

_ALLO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # experiments/allo


def tenon_commit() -> str:
    """The tenon-branch HEAD sha at write time (`git rev-parse HEAD` in the
    experiments/allo submodule). Stamped after the run -- a literal here would
    be a fabrication."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ALLO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return f"unknown ({exc})"


def now_iso() -> str:
    """ISO-8601 timestamp at write time (after the run)."""
    return datetime.datetime.now().isoformat()


def _file_sha256_prefix(path: pathlib.Path, n: int = 12) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:n]


def sim_source(backend: str) -> str:
    """A reproducible identity for the simulator that produced a number.

    Records the sim git sha where the sim dir is a repo, else the simulator
    binary's sha256 prefix (a stable content fingerprint a reviewer re-derives).
    For apu_v1 this is NOT used -- the device path stamps
    `apu_v1_device@<host>/<firmware>` (spec Answer 5), never a sim.
    """
    from allo.spmw_codegen import _pimsim_root

    if backend == "samsung_hbm_pim":
        root = _pimsim_root()
        sha = _try_git_sha(root)
        if sha:
            return f"PIMSimulator@{sha}"
        driver = root / "pim_driver"
        if driver.exists():
            return f"PIMSimulator@bin-sha256:{_file_sha256_prefix(driver)}"
        return "PIMSimulator@unknown"

    if backend == "aim":
        from allo.spmw_codegen import _aim_root

        root = _aim_root()
        sha = _try_git_sha(root)
        if sha:
            return f"ramulator2@{sha}"
        return "ramulator2@unknown"

    return f"{backend}@unknown"


def apu_v1_device_source() -> str:
    """The APU v1 REAL-DEVICE identity for `results.json` source (spec Answer
    5): `apu_v1_device@<host>/<firmware>`. Read at run time -- a reviewer
    confirms every apu_v1 row carries a device, NEVER a simulator
    (`l1_sim`/`virtual`/analytical are forbidden)."""
    import socket

    host = socket.gethostname()
    fw = "unknown"
    gsi_root = pathlib.Path("/usr/local/gsi-apu")
    if gsi_root.is_dir():
        versions = sorted(p.name for p in gsi_root.iterdir() if p.is_dir())
        if versions:
            fw = f"gsi-{versions[-1]}"
    return f"apu_v1_device@{host}/{fw}"


def _try_git_sha(repo_dir: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
