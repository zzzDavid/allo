# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM FFN int32 correctness oracle wiring (task 029).

Task 026/027 left the two FFN cells with a cycle BEAT but no correctness oracle
(`OutputDpuMramHeapPointerName` returned a zero-size stream, suppressing the
byte-compare). The goal-check requires proving the cells correct: a fast-but-wrong
cell is a FAIL. Task 029 wires an int32 oracle -- the Go host computes the golden
output (the same MAC/add/relu the kernel computes) and emits it at the leg's MRAM
write offset so the simulator's ChannelTransferReadJob byte-compares the DPU's
actual output against it.

These tests STATICALLY assert that wiring (no Docker / no simulator -- the live
6-leg byte-exact run is the standalone `_measure_upmem_ffn_oracle.py` driver and
the verifier's re-run). They pin: the golden is computed and emitted (not the old
zero-size stub), at the kernel's exact write offset, with the right arithmetic
(zero-init accumulator, ReLU on layer 1 only, bias on Exo); and the read-job emits
diverging-byte evidence on mismatch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_UPMEM = (
    Path(__file__).resolve().parents[3]
    / "simulators" / "uPIMulator" / "golang" / "uPIMulator"
)
_BENCH = _UPMEM / "benchmark"
_PRIM = _UPMEM / "src" / "assembler" / "prim"
_HOST = _UPMEM / "src" / "simulator" / "host"

pytestmark = pytest.mark.skipif(
    not _UPMEM.is_dir(), reason="uPIMulator tree not present in this checkout"
)


def _read(p: Path) -> str:
    return p.read_text()


# --- the output stream is the golden, not the old zero-size cycles-only stub --


def test_exo_emits_golden_not_empty_output():
    src = _read(_PRIM / "exo_ffn_1pd.go")
    # golden computed: 0-init accumulator, bias add, ReLU only on layer 1.
    assert "var acc int32 = 0" in src
    assert "acc += int32(img[wOff+i]) * int32(img[exoFfnXOffsetWords+i])" in src
    assert "acc += int32(img[exoFfnBiasOffsetWords])" in src
    assert "if doRelu && acc < 0 {" in src
    # emitted at the kernel's write offset (reduce1 -> heap+36864), as the
    # output heap-pointer stream the read job compares against.
    out_fn = src.split("func (this *ExoFfn1Pd) OutputDpuMramHeapPointerName")[1]
    assert "this.golden[dpu_id]" in out_fn
    assert "return 36864, bs" in out_fn
    # the old cycles-only stub must be gone (no zero-size return).
    assert "bs.Init()\n    return 36864, bs" not in out_fn


def test_cinm_emits_golden_per_phase_offset():
    src = _read(_PRIM / "cinm_ffn.go")
    # per-phase output offsets match the kernel writes (0->512,1->16,2->2048,3->8).
    for off in ("this.outOff = 512", "this.outOff = 16", "this.outOff = 2048"):
        assert off in src
    out_fn = src.split("func (this *CinmFfn) OutputDpuMramHeapPointerName")[1]
    assert "this.golden[dpu_id]" in out_fn
    assert "return this.outOff, bs" in out_fn
    # only word[0] is compared (kernel leaves word[1] at its stack value).
    assert "[]int64{int64(result)}" in src


def test_cinm_golden_arithmetic_matches_each_phase():
    src = _read(_PRIM / "cinm_ffn.go")
    # phase 0/2 are dot products with a 0-init accumulator.
    assert src.count("var acc int32 = 0") == 2
    assert "acc += int32(img[i]) * int32(img[64+i])" in src     # phase 0, 64-dot
    assert "acc += int32(img[i]) * int32(img[256+i])" in src    # phase 2, 256-dot
    assert "result = int32(img[0]) + int32(img[2])" in src      # phase 1, add
    # phase 3 relu: clamp negative to 0.
    assert "if result < 0 {" in src


# --- the read-job comparator actually panics on mismatch (the gate has teeth) -


def test_read_job_panics_with_evidence_on_mismatch():
    src = _read(_HOST / "channel_transfer_read_job.go")
    # a divergent byte must abort (no silent pass) ...
    assert 'errors.New("bytes are different")' in src
    assert "panic(err)" in src
    # ... and report the diverging expected/actual values as evidence.
    assert "CORRECTNESS MISMATCH" in src
    assert "expected=%d actual=%d" in src


# --- the oracle driver wipes stale output files (root cause of false panics) --


def test_ffn_measure_driver_wipes_stale_output_files():
    # The host reads back EVERY bin/output_*.bin it finds; a stale file from a
    # prior leg/benchmark at a different offset trips a spurious mismatch. Both
    # FFN measurement drivers must wipe them before each run.
    for drv in ("_measure_upmem_ffn.py", "_measure_upmem_ffn_oracle.py"):
        src = _read(Path(__file__).resolve().parent / drv)
        assert 'glob("output_*.bin")' in src
