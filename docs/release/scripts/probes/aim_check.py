# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np, allo, sys
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo import spmw_simenv
print("allo from", allo.__file__)
M = K = 256; MAP = [32]; ROWS = M // MAP[0]
@_df_region()
def aim_gemv(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=MAP, args=[W, x, y])
    def mv(lW: bfloat16[M, K], lx: bfloat16[K], ly: bfloat16[M]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[row0 + i, j] * lx[j]
            ly[row0 + i] = acc
compiled = allo.compile(aim_gemv, build_aim_target(), aim_cost, reduction_order="reassociate")
rng = np.random.default_rng(1)
W = (rng.standard_normal((M, K)) * 0.1).astype(np.float32); x = (rng.standard_normal(K) * 0.1).astype(np.float32)
y = np.full(M, 7.0, np.float32)
W0, x0 = W.copy(), x.copy()
r = compiled(W, x, y)
print("aim_root", spmw_simenv.aim_root())
print("RunResult kind", r.kind, "backend", r.backend, "cycles", r.cycles, "outputs", r.outputs, "y untouched", bool(np.all(y == 7.0)))
# Contract (round 108): the AiM route is timing-only on Ramulator2: kind "simulator-timing-only",
# backend "aim", a positive integer cycle count, no outputs, and every caller array unchanged.
assert spmw_simenv.aim_root() is not None, "no AiM simulator root"
assert r.kind == "simulator-timing-only" and r.backend == "aim", (r.kind, r.backend)
assert isinstance(r.cycles, int) and r.cycles > 0, r.cycles
assert r.outputs == {}, r.outputs
assert np.all(y == 7.0) and np.array_equal(W, W0) and np.array_equal(x, x0), "a timing-only run must not write any caller array"
(k,) = compiled.placement_ranking.kernels
print("chosen", k.scores[k.chosen_position] if hasattr(k, "chosen_position") else None)
print("AIM_CHECK_OK")
