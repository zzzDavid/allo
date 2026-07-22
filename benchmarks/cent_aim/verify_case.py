#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify one independently archived Tenon/CENT AiM case directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .evidence import (
    EvidenceError,
    comparison_row,
    read_json,
    render_comparison_markdown,
    sha256_file,
    verify_checksums,
)
from .verify_campaign import (
    _canonical_csv,
    _read_text,
    _source_tree_digest,
    _verify_compile,
    _verify_simulator_runs,
    _verify_vendor,
)


def verify_case(case_dir: Path, vendor_root: Path) -> dict:
    case_dir = case_dir.resolve()
    verify_checksums(case_dir, require_complete=True)
    common = case_dir / "common-evidence"
    compiler = read_json(common / "compiler.json")
    simulator = read_json(common / "simulator.json")
    for relative, expected_hash in compiler["source_files"].items():
        if sha256_file(common / "compiler-source" / relative) != expected_hash:
            raise EvidenceError(f"compiler source snapshot is stale: {relative}")
    if _source_tree_digest(compiler["source_files"]) != compiler["source_tree_sha256"]:
        raise EvidenceError("compiler source-tree digest is stale")
    if sha256_file(common / "simulator-config.yaml") != simulator["config_sha256"]:
        raise EvidenceError("retained simulator config is stale")
    inspected = read_json(common / "docker-image-inspect.stdout.json")
    if inspected[0]["Id"] != simulator["docker_image_immutable_id"]:
        raise EvidenceError("retained Docker image inspection is stale")
    if read_json(case_dir / "campaign-provenance.json") != {
        "compiler": compiler,
        "simulator": simulator,
    }:
        raise EvidenceError("copied campaign provenance differs")
    trace = _verify_compile(case_dir)
    measured = _verify_simulator_runs(case_dir, trace, simulator)
    vendor = _verify_vendor(case_dir, vendor_root.resolve(), simulator)
    vendor_case = vendor["vendor_case"]
    row = comparison_row(
        case_id=case_dir.name,
        kernel=str(vendor_case["kernel"]),
        sequence_length=int(vendor_case["sequence_length"]),
        vendor_cycles=vendor["memory_system_cycles"],
        tenon_cycles=measured["memory_system_cycles"],
        trace_sha256=trace["sha256"],
    )
    local_row = {
        **row,
        "tenon_evidence": "simulator/summary.json",
        "vendor_evidence": "vendor-evidence.json",
    }
    if read_json(case_dir / "comparison.json") != {
        "schema": "tenon-cent-aim-case-comparison-v1",
        "row": local_row,
    }:
        raise EvidenceError("standalone comparison JSON differs")
    if _read_text(case_dir / "comparison.csv") != _canonical_csv([local_row]):
        raise EvidenceError("standalone comparison CSV differs")
    if _read_text(case_dir / "RESULT.md") != render_comparison_markdown([local_row]):
        raise EvidenceError("standalone comparison Markdown differs")
    return local_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        row = verify_case(args.case_dir, args.vendor_root)
    except (EvidenceError, OSError, KeyError, IndexError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS {row['case_id']}: Tenon={row['tenon_cycles']} cycles, "
        f"CENT={row['vendor_cycles']} cycles, {row['outcome']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
