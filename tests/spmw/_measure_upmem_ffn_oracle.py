# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM FFN int32 numeric oracle (task 029 -- correctness gate).

Standalone driver (not pytest-collected: it boots the uPIMulator Docker build,
which is multi-minute). It PROVES the two ported FFN cells are correct, not just
fast, with two independent checks per leg:

  1. In-sim byte-exact:   the Go host emits a golden int32 output stream at the
     leg's MRAM write offset; the simulator's ChannelTransferReadJob byte-compares
     the DPU's actual MRAM against it and panics "bytes are different" on any
     mismatch. A clean "execution (0) is finished" with returncode 0 == byte-exact.
  2. Independent CPU oracle: this script reads the operand bytes back from the
     generated input heap file (little-endian int32), recomputes the leg result
     in numpy int32 (the same MAC / add / relu the kernel computes), and checks
     it equals the result word the Go host wrote to the output file. This catches
     a host whose golden itself were wrong -- the sim check alone could not.

A leg is reported PASS only if BOTH checks agree. A mismatch is reported as a
correctness FAIL with the diverging value -- never papered over.

Run:
    python tests/spmw/_measure_upmem_ffn_oracle.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

UPMEM = Path(
    "/work/shared/users/phd/nz264/spmw-for-pim/experiments/simulators/"
    "uPIMulator/golang/uPIMulator"
)
BIN = UPMEM / "bin"
_CYC = re.compile(r"logic_cycle:\s*(\d+)")


# --- bin-file (little-endian int32, one byte value per line) helpers --------


def _read_words(path: Path, n: int, off_words: int = 0) -> np.ndarray:
    """Read n int32 words starting at word offset off_words from a bin file."""
    vals = [int(x) for x in path.read_text().split()]
    out = np.empty(n, dtype=np.int64)
    for w in range(n):
        b = (off_words + w) * 4
        out[w] = vals[b] | (vals[b + 1] << 8) | (vals[b + 2] << 16) | (vals[b + 3] << 24)
    return out.astype(np.int32)


def _wipe_io():
    for f in BIN.glob("output_*.bin"):
        f.unlink()
    for f in BIN.glob("input_*.bin"):
        f.unlink()


def _run(benchmark: str, nt: int, sel: int):
    root = str(UPMEM)
    log = BIN / "log.txt"
    if log.exists():
        log.unlink()
    _wipe_io()
    proc = subprocess.run(
        [
            f"{root}/build/uPIMulator",
            "--benchmark", benchmark,
            "--num_channels", "1", "--num_ranks_per_channel", "1",
            "--num_dpus_per_rank", "1",
            "--num_tasklets", str(nt),
            "--data_prep_params", str(sel),
            "--root_dirpath", root,
            "--bin_dirpath", f"{root}/bin",
            "--log_dirpath", f"{root}/../log",
        ],
        capture_output=True, text=True, cwd=root,
    )
    stdout = proc.stdout + proc.stderr
    finished = "execution (0) is finished" in stdout
    mismatch = next((ln for ln in stdout.splitlines() if "MISMATCH" in ln), None)
    cyc = None
    if log.exists():
        m = _CYC.search(log.read_text())
        cyc = int(m.group(1)) if m else None
    sim_ok = proc.returncode == 0 and finished
    return sim_ok, cyc, mismatch


# --- independent CPU references (numpy int32, mirrors the kernel) -----------


def _exo_cpu_ref(sel: int) -> int:
    inp = BIN / "input_dpu_mram_heap_pointer_name_0_0_0.bin"
    n, w_off = (256, 256) if sel == 0 else (1024, 1024)
    x = _read_words(inp, n, off_words=0)
    w = _read_words(inp, n, off_words=w_off)
    bias = _read_words(inp, 1, off_words=6144)[0]
    acc = np.int32(0)
    for i in range(n):
        acc = np.int32(acc + np.int32(w[i]) * np.int32(x[i]))
    acc = np.int32(acc + bias)
    if sel == 0 and acc < 0:  # layer 1 fuses ReLU
        acc = np.int32(0)
    return int(acc)


def _cinm_cpu_ref(sel: int) -> int:
    inp = BIN / "input_dpu_mram_heap_pointer_name_0_0_0.bin"
    if sel == 0:
        x = _read_words(inp, 64, 0); w = _read_words(inp, 64, 64)
        acc = np.int32(0)
        for i in range(64):
            acc = np.int32(acc + np.int32(x[i]) * np.int32(w[i]))
        return int(acc)
    if sel == 1:
        a = _read_words(inp, 1, 0)[0]; b = _read_words(inp, 1, 2)[0]
        return int(np.int32(a) + np.int32(b))
    if sel == 2:
        x = _read_words(inp, 256, 0); w = _read_words(inp, 256, 256)
        acc = np.int32(0)
        for i in range(256):
            acc = np.int32(acc + np.int32(x[i]) * np.int32(w[i]))
        return int(acc)
    v = _read_words(inp, 1, 0)[0]
    return int(max(int(v), 0))


def _golden_word(out_off: int) -> int:
    out = BIN / f"output_dpu_mram_heap_pointer_name_{out_off}_0_0.bin"
    return int(_read_words(out, 1, 0)[0])


_CINM_OUT = {0: 512, 1: 16, 2: 2048, 3: 8}


def main():
    legs = [
        ("EXO_FFN_1PD", 8, 0, "layer-1 256-dot+ReLU", 36864, _exo_cpu_ref),
        ("EXO_FFN_1PD", 8, 1, "layer-2 1024-dot", 36864, _exo_cpu_ref),
        ("CINM_FFN", 1, 0, "layer-1 64-dot", _CINM_OUT[0], _cinm_cpu_ref),
        ("CINM_FFN", 1, 1, "bias add", _CINM_OUT[1], _cinm_cpu_ref),
        ("CINM_FFN", 1, 2, "layer-2 256-dot", _CINM_OUT[2], _cinm_cpu_ref),
        ("CINM_FFN", 1, 3, "relu", _CINM_OUT[3], _cinm_cpu_ref),
    ]
    n_pass = n_fail = 0
    print("== UPMEM FFN int32 oracle ==  (sim byte-exact AND numpy CPU-ref agree)")
    for bench, nt, sel, label, out_off, ref_fn in legs:
        sim_ok, cyc, mismatch = _run(bench, nt, sel)
        try:
            golden = _golden_word(out_off)
            cpu = ref_fn(sel)
        except Exception as e:  # noqa: BLE001
            print(f"  {bench} sel={sel} ({label}): FAIL reading oracle files: {e}")
            n_fail += 1
            continue
        cpu_ok = golden == cpu
        verdict = "PASS" if (sim_ok and cpu_ok) else "FAIL"
        if verdict == "PASS":
            n_pass += 1
        else:
            n_fail += 1
        print(
            f"  {bench} sel={sel} ({label}): {verdict}  "
            f"cyc={cyc}  sim_byte_exact={sim_ok}  "
            f"cpu_ref={cpu} golden={golden} match={cpu_ok}"
            + (f"  [{mismatch}]" if mismatch else "")
        )
    print(f"== oracle summary: {n_pass} PASS / {n_fail} FAIL of {len(legs)} legs ==")
    print("max int32 error across all matching legs = 0 (byte-exact)" if n_fail == 0
          else "CORRECTNESS FAIL -- see per-leg lines above")


if __name__ == "__main__":
    main()
