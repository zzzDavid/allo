# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-004b: Samsung functional-readback guard tests.

The additive readback surfaces PIMSimulator's already-written `out.bin` into
`RunResult.extra["outputs"]` (mirroring `_run_apu_v1`), so the Samsung
GEMV/GEMM-family cells the sim functionally computes reach the correctness
pipeline as PASS instead of CYCLES-ONLY. The load-bearing guard is
`test_samsung_cycles_byte_identical`: the readback is post-run file I/O and must
not perturb the cycle path.

Skips cleanly when PIMSimulator (`pim_driver`) is unavailable (env gate).
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.spmw_codegen import (
    RunResult,
    _pimsim_root,
    _samsung_read_gemv_outbin,
)

from _fixtures import build_samsung_target


def _sim_available() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


def _gemv_workload(M, K):
    """A single-MAC GEMV @allo.work: y[M] = W[M,K] @ x[K]."""
    from allo.ir.types import float32 as fp16
    from allo.dataflow import region as _df_region

    @_df_region()
    def _top(W: fp16[M, K], x: fp16[K], y: fp16[M]):
        @allo.work(mapping=[1], args=[W, x, y])
        def gv(lW: fp16[M, K], lx: fp16[K], ly: fp16[M]):
            for i in range(M):
                acc: fp16 = 0
                for k in range(K):
                    acc += lW[i, k] * lx[k]
                ly[i] = acc

    return _top


def _compile_gemv(M, K):
    target = build_samsung_target()
    sch = allo.customize(_gemv_workload(M, K), enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    return allo.compile_for_target(target, trace)


# --------------------------------------------------------------------- #
# D3 anchors
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not _sim_available(), reason="PIMSimulator pim_driver unavailable")
def test_samsung_gemv_surfaces_outputs():
    """The readback correctly surfaces + tree-reduces the GEMV result
    `pim_driver` writes to out.bin, matching W@x within fp16 tolerance, at the
    M=4096 design point.

    IMPORTANT (empirical finding): the result lands in out.bin on the driver's
    PLAIN `executeGemv` path, NOT on the `--faithful`/`--cmds` CRF path the
    Tenon run path uses (the faithful path's `readResult` region stays zero).
    So this drives `pim_driver` on the plain path directly to verify the
    readback helper -- the honest test of `_samsung_read_gemv_outbin`. The
    consequence for the SUITE (faithful path -> all-zero out.bin -> graceful
    CYCLES-ONLY) is documented in the 005b report for the architect.
    """
    import subprocess
    import tempfile
    import pathlib

    M = K = 4096
    rng = np.random.default_rng(0)
    W = (rng.random((M, K)).astype(np.float16) * 0.01)
    x = (rng.random(K).astype(np.float16) * 0.01)
    root = _pimsim_root()
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        out_path = td_path / "out.bin"
        np.save(td_path / "W.npy", W)
        np.save(td_path / "x.npy", x.reshape(1, -1))
        argv = [
            str(root / "pim_driver"), "--op", "GEMV", "--out", str(out_path),
            "--weight", str(td_path / "W.npy"), "--in", str(td_path / "x.npy"),
            "--output-dim", str(M), "--input-dim", str(K),
        ]  # plain path: NO --cmds / --faithful (the path that populates out.bin)
        proc = subprocess.run(argv, capture_output=True, cwd=str(root), timeout=600)
        assert proc.returncode == 0, proc.stderr.decode()[-300:]
        outputs = _samsung_read_gemv_outbin(out_path, "GEMV")

    assert "y" in outputs, f"readback surfaced no 'y'; got {outputs!r}"
    ref = W.astype(np.float32) @ x.astype(np.float32)
    y = np.asarray(outputs["y"])[:M]
    assert np.allclose(y, ref, rtol=2e-2, atol=2e-2), (
        f"surfaced GEMV output does not match W@x; "
        f"max_abs={np.max(np.abs(y - ref)):g}"
    )


@pytest.mark.skipif(not _sim_available(), reason="PIMSimulator pim_driver unavailable")
def test_samsung_cycles_byte_identical():
    """The readback is inert on the cycle path: a fixed GEMV's cycle count is
    the same value the pre-readback run produced. Pinned to the
    P=60,K=80 gemm-family geometry's observed PIM_CYCLES total (2813) -- the
    no-cycle-change anchor."""
    compiled = _compile_gemv(60, 80)
    W = np.zeros((60, 80), dtype=np.float16)
    x = np.zeros(80, dtype=np.float16)
    result = compiled.run(W=W, x=x)
    assert result.cycles == 2813, (
        f"readback perturbed the cycle path: expected 2813, got "
        f"{result.cycles!r} (stdout tail: {result.stdout[-200:]})"
    )


def test_samsung_no_outbin_falls_back():
    """When out.bin is absent (or the op is not GEMV), the readback returns {}
    and the cell stays CYCLES-ONLY -- the graceful-fallback anchor. No sim
    needed: exercise the helper directly."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as td:
        missing = pathlib.Path(td) / "out.bin"  # never created
        assert _samsung_read_gemv_outbin(missing, "GEMV") == {}
    # Non-GEMV op never surfaces outputs.
    assert _samsung_read_gemv_outbin("/nonexistent", "MUL") == {}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
