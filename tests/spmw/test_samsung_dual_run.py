# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-005d: Samsung faithful+plain dual-run guard tests.

The dual-run gives a Samsung GEMV-family cell a real PASS: the faithful
`--cmds`/`--faithful` run for the cycle-accurate cycle count (byte-identical to
today), plus a second plain `executeGemv` run of the same workload/shape for the
functional output (the faithful path writes an all-zero out.bin by
construction). Both runs are real; cycles always come from faithful, correctness
from plain. The load-bearing anchor is `test_samsung_dual_run_cycles_from_faithful`
(the plain run never feeds the cycle number).

Skips cleanly when PIMSimulator (`pim_driver`) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.spmw_codegen import RunResult, _pimsim_root, _samsung_plain_gemv_rerun

from _fixtures import build_samsung_target


def _sim_available() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


def _compile_gemv(M, K):
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

    target = build_samsung_target()
    sch = allo.customize(_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    return allo.compile_for_target(target, trace)


# A GEMV at the M=4096 design point where the plain executeGemv path populates
# out.bin (verified in test_samsung_readback.py to match W@x at 2.4e-4).
def _design_point_inputs():
    M = K = 4096
    rng = np.random.default_rng(0)
    W = rng.random((M, K)).astype(np.float16) * 0.01
    x = rng.random(K).astype(np.float16) * 0.01
    return M, K, W, x


@pytest.mark.skipif(not _sim_available(), reason="PIMSimulator pim_driver unavailable")
def test_samsung_dual_run_pass():
    """A GEMV dual-run at the design point returns faithful cycles AND
    extra['outputs']['y'] matching W@x within 2e-2 -> a real PASS."""
    M, K, W, x = _design_point_inputs()
    result = _compile_gemv(M, K).run(W=W, x=x)
    assert isinstance(result, RunResult)
    assert result.cycles is not None and result.cycles > 0
    outputs = (result.extra or {}).get("outputs")
    assert outputs and "y" in outputs, (
        f"dual-run surfaced no functional output; extra={result.extra!r}"
    )
    ref = W.astype(np.float32) @ x.astype(np.float32)
    y = np.asarray(outputs["y"])[:M]
    assert np.allclose(y, ref, rtol=2e-2, atol=2e-2), (
        f"plain run output does not match W@x; max_abs={np.max(np.abs(y - ref)):g}"
    )


@pytest.mark.skipif(not _sim_available(), reason="PIMSimulator pim_driver unavailable")
def test_samsung_dual_run_cycles_from_faithful():
    """The merged cycles equal the faithful-only cycles (the plain run never
    feeds the cycle number) -- pinned to the P=60,K=80 geometry's 2813."""
    result = _compile_gemv(60, 80).run(
        W=np.zeros((60, 80), dtype=np.float16), x=np.zeros(80, dtype=np.float16)
    )
    assert result.cycles == 2813, (
        f"dual-run perturbed the faithful cycle path: expected 2813, got "
        f"{result.cycles!r}"
    )


@pytest.mark.skipif(not _sim_available(), reason="PIMSimulator pim_driver unavailable")
def test_samsung_dual_run_provenance():
    """When the dual-run produces a functional output, the provenance records
    cycles from faithful and correctness from the plain run (the two-real-runs
    record)."""
    M, K, W, x = _design_point_inputs()
    result = _compile_gemv(M, K).run(W=W, x=x)
    extra = result.extra or {}
    if not extra.get("outputs"):
        pytest.skip("design-point plain run produced no output on this build")
    assert "faithful" in extra.get("cycles_source", "")
    assert "plain" in extra.get("correctness_source", "")


def test_samsung_dual_run_falls_back():
    """Off-design / absent plain out.bin -> {} -> CYCLES-ONLY, no false PASS.
    Exercise the plain-rerun helper directly with a missing driver path: it must
    return {} gracefully, never raise."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        # A nonexistent driver path -> the subprocess errs -> graceful {}.
        out = _samsung_plain_gemv_rerun(
            "/nonexistent/pim_driver", td, td_path,
            td_path / "W.npy", td_path / "x.npy", 60, 80,
        )
        assert out == {}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
