# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: dataset shapes, re-exported from psize.json.

The PolyBench dataset constants are WORKLOAD INPUTS, not a compiler decision
path. The anti-hardcoding gate forbids pasted dimension literals in a kernel
folder, so shapes come from `examples/polybench/psize.json` (verified
element-by-element against PolyBench/C 4.2.1 SMALL in the provenance note). A
kernel folder calls `shape("gemm")`, never writes `60`/`70`/`80`.
"""

from __future__ import annotations

import json
import pathlib

# tests/pim/lib/shapes.py -> parents[3] == experiments/allo
_PSIZE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "examples" / "polybench" / "psize.json"
)
_PSIZE = json.loads(_PSIZE_PATH.read_text())

# Map the suite's SMALL_DATASET size class to psize.json's "small" key.
SIZE_KEYS = {"small": "small", "SMALL": "small", "SMALL_DATASET": "small"}

# The suite uses canonical PolyBench kernel names; psize.json spells a few
# differently. Alias the suite name to the psize key (no literal pasted -- the
# constants still come from psize.json).
PSIZE_ALIAS = {"2mm": "two_mm", "3mm": "three_mm"}


def shape(kernel: str, size: str = "small") -> dict:
    """The dataset constants for `kernel` at `size` (default SMALL), as a dict
    of the PolyBench dimension names (e.g. gemm -> {'P':60,'R':70,'Q':80})."""
    key = SIZE_KEYS.get(size, size)
    kernel = PSIZE_ALIAS.get(kernel, kernel)
    if kernel not in _PSIZE:
        raise KeyError(f"no psize entry for kernel {kernel!r}")
    if key not in _PSIZE[kernel]:
        raise KeyError(f"no size {size!r} for kernel {kernel!r}")
    return _PSIZE[kernel][key]
