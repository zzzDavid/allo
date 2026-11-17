# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only installed-transport preflight and empirical session time budget.

One real snapshot proves vendor-daemon destination access and chronology.
The stage budget scales with planned launches, allowing seven reads at twice
this preflight's duration plus 120 seconds for per-launch host/device work,
at least 600 seconds per launch, and 600 seconds for stage setup. This is
an empirical hang guard, not a promise about worst-case execution latency.
"""

import json
import math
import os
from pathlib import Path
import sys
import time


def seconds_per_launch(elapsed):
    derived = max(600, math.ceil(14 * elapsed + 120))
    configured = os.environ.get("G1_SECONDS_PER_LAUNCH")
    if configured is None:
        return derived
    requested = int(configured)
    if requested < derived:
        raise ValueError(
            f"G1_SECONDS_PER_LAUNCH must be at least measured budget {derived}"
        )
    return requested


def main():
    from allo.pim.apu_v1_program import _ledag_drain

    output = Path(sys.argv[1])
    started = time.monotonic()
    text = _ledag_drain()
    elapsed = time.monotonic() - started
    record = {
        "kind": "read-only transport preflight",
        "read_seconds": elapsed,
        "seconds_per_launch": seconds_per_launch(elapsed),
        "stage_setup_seconds": 600,
        "evidence": text.splitlines()[0],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
