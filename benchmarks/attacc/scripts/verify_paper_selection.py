# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the paper's AttAcc instances through ``allo.compile`` and compare.

For each instance of ``benchmarks.attacc.workloads.GPT13B_INSTANCES`` this
builds the ``@allo`` source, compiles it with
``allo.compile(source, build_attacc_target(), attacc_cost, backend="virtual",
reduction_order="reassociate")`` and compares the selected
``attacc_tiling`` with the tiling the paper's selector chose
(``paper_selection.csv``).  Nothing is simulated.

The wall time includes the frontend, matching and the per-candidate
execution-graph construction of the public ranking (about 0.7 ms per
candidate); it is not the selector-only time of the paper's compile-time
table (``selector_timing.py`` measures that quantity).

Usage::

    python -m benchmarks.attacc.scripts.verify_paper_selection \
        [--family gemv|attention] [--every N] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=("gemv", "attention"))
    parser.add_argument(
        "--every", type=int, default=1, help="check every N-th instance"
    )
    parser.add_argument("--out", help="write the per-instance records as JSON")
    args = parser.parse_args(argv)

    import allo
    from allo.pim.costs import attacc_cost
    from allo.pim.targets import build_attacc_target

    from ..workloads import GPT13B_INSTANCES, paper_selection

    expected = paper_selection()
    target = build_attacc_target()
    instances = [
        item
        for item in GPT13B_INSTANCES
        if args.family is None or item.family == args.family
    ][:: max(1, args.every)]
    records = []
    start = time.perf_counter()
    for item in instances:
        t0 = time.perf_counter()
        compiled = allo.compile(
            item.region(),
            target,
            attacc_cost,
            backend="virtual",
            reduction_order="reassociate",
        )
        layout = compiled.compiled.layout
        selected = [list(axis) for axis in layout.extra["attacc_tiling"].dims]
        ranking = compiled.placement_ranking.kernels[0]
        records.append(
            {
                "id": item.id,
                "selected": selected,
                "paper": expected[item.id],
                "match": selected == expected[item.id],
                "cycles": compiled.estimate().cycles,
                "candidates": len(ranking.scores),
                "ties": len(ranking.cost_ties),
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )
        print(
            f"{item.id}: {'ok' if records[-1]['match'] else 'MISMATCH'} "
            f"{selected} {records[-1]['cycles']} cycles",
            flush=True,
        )
    wall = time.perf_counter() - start
    matched = sum(record["match"] for record in records)
    summary = {
        "checked": len(records),
        "matched": matched,
        "candidates": sum(record["candidates"] for record in records),
        "wall_seconds": round(wall, 1),
    }
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "records": records}, handle, indent=1)
    return 0 if matched == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
