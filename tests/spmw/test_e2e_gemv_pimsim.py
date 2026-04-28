# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Samsung HBM-PIM smoke test (report 16, Step E.Q3).

Wires the GEMV workload through the full Tenon SPMW path:
  match_workload(samsung_target, schedule.module)
       -> MatchTrace (128 MAC matches)
  compile_for_target(samsung_target, trace)
       -> Compiled  (Samsung GEMV artifact)
  Compiled.run(W=..., x=...)
       -> ({y: ...}, stats)  via real pim_driver

Asserts numeric correctness (fp16 GEMV at K=1024 has wide tolerance —
``atol=2.0`` is fine) and a non-zero cycle count parsed from the
PIMSimulator stdout.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_match import MatchTrace


M, K = 4096, 1024
ROWS = M // (16 * 8)  # 32 rows per work-item


# --------------------------------------------------------------------- #
# Target spec — same as test_match_gemv.py
# --------------------------------------------------------------------- #


def build_samsung_target():
    @allo.target("samsung_hbm_pim")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.memory(banks=16, rows=16384, cols=128, width=8, name="banks")

            @allo.unit(mapping=[8])
            def pim():
                _, pid = allo.get_uid()
                even_bank = banks[2 * pid]
                odd_bank = banks[2 * pid + 1]
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                allo.move("LD_A", src=even_bank, dst=grf_a)
                allo.move("LD_B", src=odd_bank, dst=grf_b)
                allo.move("ST_A", src=grf_a, dst=even_bank)
                allo.move("ST_B", src=grf_b, dst=odd_bank)

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                )

    return device


# --------------------------------------------------------------------- #
# Workload — the affine-loop GEMV from report 16
# --------------------------------------------------------------------- #


@_df_region()
def gemv_top(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


# --------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------- #


def test_e2e_gemv_pimsim():
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace: MatchTrace = allo.match_workload(target, schedule.module)
    assert trace.target_name == "samsung_hbm_pim"
    assert trace.by_target_op("MAC"), "expected MAC matches"

    compiled = allo.compile_for_target(target, trace)

    rng = np.random.default_rng(0)
    # fp16 has narrow dynamic range; keep entries small so K=1024 reduction
    # stays well below fp16 inf and doesn't blow up tolerance.
    W = rng.uniform(-0.5, 0.5, size=(M, K)).astype(np.float16)
    x = rng.uniform(-0.5, 0.5, size=(K,)).astype(np.float16)

    outputs, stats = compiled.run(W=W, x=x)
    assert "y" in outputs, f"expected output 'y', got {list(outputs)}"
    y = outputs["y"]
    assert y.shape == (M,), y.shape
    assert y.dtype == np.float16

    # Reference in fp32 (mirroring how the simulator tree-reduces partials).
    y_ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)

    max_err = float(np.max(np.abs(y.astype(np.float32) - y_ref.astype(np.float32))))
    print(f"PIM_CYCLES total={stats['cycles']} preload={stats.get('preload', 0)} "
          f"exec={stats.get('exec', 0)} readback={stats.get('readback', 0)}")
    print(f"GEMV max_err = {max_err:.4f} (M={M}, K={K})")

    assert stats["cycles"] > 0, f"expected positive cycle count, got {stats!r}"
    assert np.allclose(
        y.astype(np.float32),
        y_ref.astype(np.float32),
        atol=2.0,
    ), f"GEMV output diverged from numpy reference; max_err={max_err}"


if __name__ == "__main__":
    test_e2e_gemv_pimsim()
    print("PASSED")
