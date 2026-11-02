# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time the AttAcc tiling selector alone over the paper's 500 instances.

This is the quantity of the paper's compile-time table (enumerate the legal
tilings, count commands, evaluate the closed-form cost, take the minimum),
measured on ``allo.pim.attacc_tiling.rank_attacc_tilings`` with the cost
function the ``attacc_cost`` rules call.  It excludes the Allo frontend,
matching and the per-candidate execution-graph construction of
``allo.compile`` (see ``verify_paper_selection.py`` for that end-to-end
time).  Timings depend on the machine; report them with the host.

Usage::

    python -m benchmarks.attacc.scripts.selector_timing
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    from allo.pim.attacc_tiling import (
        AttAccKernel,
        _space,
        attacc_geometry,
        attacc_tiling_space,
        rank_attacc_tilings,
    )
    from allo.pim.targets import build_attacc_target

    from ..workloads import GPT13B_INSTANCES, paper_selection

    geometry = attacc_geometry(build_attacc_target())
    expected = paper_selection()
    kernels = [
        (item, AttAccKernel(item.family, item.rows, item.reduction, item.outputs))
        for item in GPT13B_INSTANCES
    ]
    start = time.process_time()
    candidates = sum(len(attacc_tiling_space(kernel, geometry)) for _item, kernel in kernels)
    enumerate_s = time.process_time() - start
    _space.cache_clear()  # time enumeration again inside the selector
    start = time.process_time()
    selected = {item.id: rank_attacc_tilings(kernel, geometry)[0][0] for item, kernel in kernels}
    select_s = time.process_time() - start
    matched = sum(selected[item.id].dcc_tiling() == expected[item.id] for item, _ in kernels)
    print(json.dumps({
        "instances": len(kernels),
        "candidates": candidates,
        "enumerate_cpu_s": round(enumerate_s, 2),
        "enumerate_count_cost_select_cpu_s": round(select_s, 2),
        "first_minimum_equals_paper": matched,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
