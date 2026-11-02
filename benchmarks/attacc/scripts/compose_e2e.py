# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPT-13B decode latency from Ramulator2-AttAcc kernel times (DCC composition).

EVALUATION ONLY, external dependencies, not part of the installed package.
The result is an analytical composition (``kind: "analytical-composition"``):
per-kernel PIM times measured by Ramulator2-AttAcc (``run_ramulator.py``)
are combined with DCC's analytical A100 roofline and decode pipeline model
(DCC ``inference/system.py``), as in the paper ("GPT-13B decode latency
combines Ramulator2-AttAcc PIM timings with DCC's analytical GPU roofline",
``5-evaluation.tex``).  Nothing here is a measurement of a decode step.

DCC (github.com/SPIN-Research-Group/DCC, study commit
``212f86ea7f08da059836e3a1311d3a11afae8dbe``) has no license; its
``inference`` package is imported at run time from ``ATTACC_DCC_ROOT`` and
needs DCC's own Python environment.

Inputs:

* ``ATTACC_DCC_ROOT``: DCC ``src/src`` (``inference/``, and the per-kernel
  tables ``temp/simulation/@attacc-{gemv,attn}_inference.csv`` that DCC's
  ``kernel_simulation.py`` writes, with ``baseline(ms)`` columns).
* ``--kernel-times``: a JSON list of ``run_ramulator.py`` records
  (``family``, ``shape``, ``memory_system_cycles``) for every PIM kernel row
  the composition reads.  A row without a Tenon time is an error unless
  ``--missing=baseline`` is given, which keeps DCC's fixed-tiling time for
  it and reports the count.

Usage::

    ATTACC_DCC_ROOT=... python compose_e2e.py --kernel-times times.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

LABEL = "tenon_time(ms)"
CONFIGS = [(lin, lout, batch) for batch in (1, 4) for lin, lout in ((128, 2048), (2048, 128), (2048, 2048))]


def _key(family, shape):
    rows, reduction, outputs = shape
    return ("gemv" if family == "gemv" else "attn", int(rows), int(reduction), int(outputs))


def tenon_tables(dcc_root: Path, times: dict, clock_mhz: float, missing: str, out: Path) -> dict:
    filled = {"measured": 0, "baseline": 0}
    for name in ("@attacc-gemv_inference.csv", "@attacc-attn_inference.csv"):
        family = "gemv" if "gemv" in name else "attn"
        rows = list(csv.DictReader((dcc_root / "temp" / "simulation" / name).open()))
        header = list(rows[0]) + ([LABEL] if LABEL not in rows[0] else [])
        for row in rows:
            key = (family, int(row["#head"]), int(row["input_dim"]), int(row["output_dim"]))
            if key in times:
                row[LABEL] = times[key] / (clock_mhz * 1e3)  # cycles -> ms
                filled["measured"] += 1
            elif missing == "baseline":
                row[LABEL] = row["baseline(ms)"]
                filled["baseline"] += 1
            else:
                raise KeyError(
                    f"no Tenon kernel time for {key}; time it with run_ramulator.py "
                    "or pass --missing=baseline"
                )
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    return filled


def decode_ms(table_dir: Path, batch, lin, lout, *, gpu_only, pipe, parallel):
    from inference.config import (  # DCC, imported from ATTACC_DCC_ROOT
        DataType,
        DeviceType,
        GPUType,
        InterfaceType,
        PIMType,
        make_model_config,
        make_pim_config,
        make_xpu_config,
    )
    from inference.system import System

    model = make_model_config("GPT-13B", DataType.W16A16)
    model["DCC"] = not gpu_only  # QKV / projection on PIM
    xpu = make_xpu_config(GPUType.A100a, num_gpu=1, mem_cap=80 * 1024**3)
    system = System(xpu["GPU"], model)
    if not gpu_only:
        pim = make_pim_config(
            PIMType.BA,
            InterfaceType.NVLINK3,
            num_attacc=1,
            power_constraint=False,
            table_path=str(table_dir) + "/",
        )
        pim["RES_ATTACC_LABEL"] = LABEL
        system.set_accelerator(model, DeviceType.PIM, pim)
    perfs = []
    system.simulate(batch, lin, lout, perfs=perfs, pipe=pipe, parallel_ff=parallel, power_constraint=False)
    tag, config, time = perfs[0]
    # DCC's write_csv column order: 'g_time (ms)' is the decode latency.
    return (tag + config + time)[25]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel-times", required=True)
    parser.add_argument("--clock-mhz", type=float, default=1300.0,
                        help="HBM3_5.2Gbps command clock (11,454 cycles = 0.008808 ms)")
    parser.add_argument("--missing", choices=("error", "baseline"), default="error")
    args = parser.parse_args(argv)
    root = os.environ.get("ATTACC_DCC_ROOT")
    if not root or not (Path(root) / "inference" / "system.py").is_file():
        raise SystemExit("set ATTACC_DCC_ROOT to a DCC checkout's src/src directory")
    dcc_root = Path(root).resolve()
    sys.path.insert(0, str(dcc_root))
    records = json.load(open(args.kernel_times))
    times = {_key(r["family"], r["shape"]): int(r["memory_system_cycles"]) for r in records}
    with tempfile.TemporaryDirectory(prefix="tenon-attacc-e2e-") as tmp:
        filled = tenon_tables(dcc_root, times, args.clock_mhz, args.missing, Path(tmp))
        rows = []
        for lin, lout, batch in CONFIGS:
            gpu = decode_ms(Path(tmp), batch, lin, lout, gpu_only=True, pipe=0, parallel=False)
            full = decode_ms(Path(tmp), batch, lin, lout, gpu_only=False, pipe=True, parallel=True)
            rows.append({"lin": lin, "lout": lout, "batch": batch, "gpu_ms": gpu,
                         "tenon_ms": full, "speedup": gpu / full})
    print(json.dumps({"kind": "analytical-composition", "kernel_rows": filled, "configs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
