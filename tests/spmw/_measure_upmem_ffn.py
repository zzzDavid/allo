# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manual measurement driver for the UPMEM FFN floor + tasklet-lever probe (task 027).

Not a pytest test -- a standalone script the coder/verifier runs to obtain the
real uPIMulator cycle counts for the two ported FFN cells (design 02 §4 host
port, task 026) and to probe whether the design-02 tasklet lever moves the FFN
numbers. It does NOT: the Exo FFN kernel statically partitions its layer-2
1024-dot across exactly NR_TASKLETS=8 (stride 128*tid, reduce over partial[0..7])
so it is correct only at nt=8 and its sweep rises past nt=8; the Cinnamon FFN
layer-2 is a one-output-per-PU serial 256-dot with no tasklet striping, so its
sweep is flat. Both legs already sit at the parallel floor the reference uses.

Run (after the uPIMulator Go sim + benchmark slots are built):
    python tests/spmw/_measure_upmem_ffn.py

Yardsticks (baselines/cinnamon-exo/upmem/README.md §3): the layer-2 single-output
dot is the basis cell --
    Exo  256-1024-256  nt=8  ref 47,250 cyc / 135 us
    Cinnamon 64-256-64 nt=1  ref 67,750 cyc / 194 us
wall_us = logic_cycle / 350.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

UPMEM = Path(
    "/work/shared/users/phd/nz264/spmw-for-pim/experiments/simulators/"
    "uPIMulator/golang/uPIMulator"
)
FREQ_MHZ = 350.0
_CYC = re.compile(r"logic_cycle:\s*(\d+)")


def run(benchmark: str, nt: int, data_prep: int) -> int | None:
    """Returns the logic_cycle, or None if the sim panics (illegal nt for the
    kernel -- itself evidence the lever cannot be legalized for this leg)."""
    root = str(UPMEM)
    # log.txt persists across runs; delete first so a crash never reports a
    # stale prior value.
    log = Path(root) / "bin" / "log.txt"
    if log.exists():
        log.unlink()
    # The FFN hosts now emit a per-leg golden output stream at the leg's write
    # offset (task 029 correctness oracle). bin/output_*.bin files persist across
    # runs and the host reads back EVERY one it finds, so a stale file at another
    # offset/benchmark would trip a spurious "bytes are different" panic. Wipe
    # them so each run verifies only its own leg.
    for stale in (Path(root) / "bin").glob("output_*.bin"):
        stale.unlink()
    for stale in (Path(root) / "bin").glob("input_*.bin"):
        stale.unlink()
    proc = subprocess.run(
        [
            f"{root}/build/uPIMulator",
            "--benchmark", benchmark,
            "--num_channels", "1",
            "--num_ranks_per_channel", "1",
            "--num_dpus_per_rank", "1",
            "--num_tasklets", str(nt),
            "--data_prep_params", str(data_prep),
            "--root_dirpath", root,
            "--bin_dirpath", f"{root}/bin",
            "--log_dirpath", f"{root}/../log",
        ],
        capture_output=True,
        cwd=root,
    )
    if proc.returncode != 0 or not log.exists():
        return None
    m = _CYC.search(log.read_text())
    return int(m.group(1)) if m else None


def main():
    def fmt(c):
        return f"{c:<7} wall={c / FREQ_MHZ:.1f}us" if c is not None else "PANIC (illegal nt for this kernel)"

    print("== EXO_FFN_1PD sel=1 (layer-2 1024-dot) nt sweep ==")
    for nt in (8, 11, 16):
        print(f"  nt={nt:<3} cycles={fmt(run('EXO_FFN_1PD', nt, data_prep=1))}")
    print("  (nt<8 UNDER-COUNTS: kernel stride is 128*tid over 8 partitions;")
    print("   nt=8 is the correct parallel floor that matches the Exo reference;")
    print("   nt>8 rises -- extra tasklets stride OOB + add barrier overhead)")

    print("== CINM_FFN sel=2 (layer-2 256-dot) nt sweep ==")
    for nt in (1, 2, 8):
        print(f"  nt={nt:<3} cycles={fmt(run('CINM_FFN', nt, data_prep=2))}")
    print("  (one-output-per-PU serial dot, no tasklet striping -> Cinnamon")
    print("   cannot legalize the lever; nt>1 PANICs; nt=1 is its correct floor)")

    exo = run("EXO_FFN_1PD", 8, data_prep=1)
    cinm = run("CINM_FFN", 1, data_prep=2)
    print("== floor verdict (layer-2 basis leg, vs README §3 reference) ==")
    print(f"  Exo  nt=8 = {exo} cyc / {exo / FREQ_MHZ:.1f}us  vs 47250 / 135us"
          f"  ratio={exo / 47250:.3f}")
    print(f"  Cinm nt=1 = {cinm} cyc / {cinm / FREQ_MHZ:.1f}us  vs 67750 / 194us"
          f"  ratio={cinm / 67750:.3f}")


if __name__ == "__main__":
    main()
