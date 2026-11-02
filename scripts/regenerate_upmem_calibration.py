# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``PHYSICAL_CALIBRATION_FACTORS`` in ``allo/pim/costs/upmem.py``.

Each factor is the mean ``anchor.logic_cycles / raw_analytical(anchor)`` over
the anchors of ``DEFAULT_UPMEM_CALIBRATION_EVIDENCE`` that share a calibration
key, in evidence order: the same arithmetic as
``UPMEMPhysicalCostModel.estimate``. The anchors' cycle counts are the
uPIMulator measurements bound in ``allo/pim/upmem_physical_search.py``
(``_DEFAULT_RAW_EVIDENCE_BINDINGS``).

    python scripts/regenerate_upmem_calibration.py            # print the table
    python scripts/regenerate_upmem_calibration.py --check    # compare with upmem.py
    python scripts/regenerate_upmem_calibration.py --check --artifacts
        # also re-hash the archived anchor runs under
        # $TENON_ARTIFACTS/upmem/tenon/calibration and re-read their cycles

Run from the repository root with the package importable (PYTHONPATH=$PWD).
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from allo import spmw_simenv
from allo.pim.costs import upmem as upmem_costs
from allo.pim.upmem_physical_search import (
    DEFAULT_UPMEM_CALIBRATION_EVIDENCE,
    _analytical_breakdown,
    _calibration_key,
    physical_features,
)

_LOGIC_CYCLE = re.compile(r"^Logic\[0_0_0\]_logic_cycle:\s*(\d+)\s*$", re.M)


def factors():
    ratios = {}
    for row in DEFAULT_UPMEM_CALIBRATION_EVIDENCE:
        raw = _analytical_breakdown(
            row.decision, physical_features(row.problem, row.decision)
        ).logic_cycles
        key = _calibration_key(row.problem, row.decision)
        ratios.setdefault(key, []).append(row.logic_cycles / raw)
    return {key: sum(values) / len(values) for key, values in ratios.items()}


def render(table):
    lines = ["PHYSICAL_CALIBRATION_FACTORS = {"]
    for key, value in table.items():
        lines.append(f"    {key!r}: float.fromhex({value.hex()!r}),")
    lines.append("}")
    return "\n".join(lines).replace("'", '"')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifacts(root):
    """Bind every in-package anchor to its archived exact-oracle run."""
    index = json.loads((root / "anchor-index.json").read_text())
    archived = {entry["evidence_id"]: entry for entry in index["anchors"]}
    problems = []
    for row in DEFAULT_UPMEM_CALIBRATION_EVIDENCE:
        entry = archived.get(row.evidence_id)
        if entry is None:
            problems.append(f"{row.evidence_id}: not in anchor-index.json")
            continue
        run, program = root / entry["run"], root / entry["program"]
        (task,) = [
            name
            for name, sha in entry["program_sha256"].items()
            if sha == entry["program_task_sha256"]
        ]
        logged = _LOGIC_CYCLE.findall((run / "bin" / "log.txt").read_text())
        checks = {
            "program sha256": (_sha256(program / task), row.program_sha256),
            "result sha256": (_sha256(run / "result.json"), row.result_sha256),
            "simulator log sha256": (
                _sha256(run / "bin" / "log.txt"),
                row.simulator_log_sha256,
            ),
            "logged logic cycles": (logged, [str(row.logic_cycles)]),
            "indexed logic cycles": (entry["logic_cycles"], row.logic_cycles),
            "oracle": (entry["oracle_validation_status"], "pass"),
        }
        for name, (observed, expected) in checks.items():
            if observed != expected:
                problems.append(f"{row.evidence_id}: {name} {observed} != {expected}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="compare with allo/pim/costs/upmem.py"
    )
    parser.add_argument(
        "--artifacts",
        nargs="?",
        const="",
        help="calibration archive (default: $TENON_ARTIFACTS/upmem/tenon/calibration)",
    )
    args = parser.parse_args()
    table = factors()
    status = 0
    if args.check:
        if table == upmem_costs.PHYSICAL_CALIBRATION_FACTORS:
            print(f"PHYSICAL_CALIBRATION_FACTORS: {len(table)} keys reproduced exactly")
        else:
            print("PHYSICAL_CALIBRATION_FACTORS differs; regenerated table:")
            print(render(table))
            status = 1
    else:
        print(render(table))
    if args.artifacts is not None:
        root = (
            Path(args.artifacts)
            if args.artifacts
            else spmw_simenv.tenon_artifacts_root() / "upmem" / "tenon" / "calibration"
        )
        problems = check_artifacts(root)
        for problem in problems:
            print(problem)
        if problems:
            status = 1
        else:
            print(
                f"{len(DEFAULT_UPMEM_CALIBRATION_EVIDENCE)} anchors match the "
                f"archived runs under {root}"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
