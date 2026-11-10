"""Installed-artifact probe (round 93): the bundled Samsung simulator route from the
wheel, outside the checkout, no PYTHONPATH, safe path, no simulator variables."""
import os, sys, time
import numpy as np
import allo
from allo import spmw_simenv
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target

print("allo from", allo.__file__)
assert "/pkg/venv/" in allo.__file__, allo.__file__
assert not any(p.startswith("/work/shared/users/phd/nz264/spmw-for-pim") for p in sys.path), sys.path
for var in ("PIMSIMULATOR_ROOT", "TENON_SIMULATORS_DIR", "PYTHONPATH", "LLVM_BUILD_DIR"):
    assert var not in os.environ, var
root = spmw_simenv.pimsim_root()
print("pimsim_root", root, "default", spmw_simenv.default_pimsim_root())
assert root == spmw_simenv.default_pimsim_root(), root
assert spmw_simenv.samsung_unavailable_reason() is None

M, K, ROWS = 4096, 256, 32

@_df_region()
def gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def kernel(lW: fp16[M, K], lx: fp16[K], ly: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc

t0 = time.time()
compiled = allo.compile(gemv, build_samsung_target(), samsung_cost, reduction_order="reassociate")
print("compile %.1f s" % (time.time() - t0), "mode", compiled.compiled.layout_ctx.programs[0].kind)
rng = np.random.default_rng(0)
W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
x = (rng.standard_normal(K) * 0.1).astype(np.float16)
y = np.zeros(M, dtype=np.float16)
t0 = time.time()
result = compiled(W, x, y)
ref = W.astype(np.float32) @ x.astype(np.float32)
err = float(np.max(np.abs(y.astype(np.float32) - ref)))
print("run %.1f s" % (time.time() - t0), "cycles", result.cycles, "max abs err %.4f" % err,
      "conductor pimcmdgen_calls", result.extra["conductor"].get("pimcmdgen_calls"),
      "driver", result.extra.get("simulator", {}).get("pim_driver") if isinstance(result.extra.get("simulator"), dict) else "")
assert err < 0.02, err
assert result.extra["conductor"]["pimcmdgen_calls"] == 0
print("BUNDLED_ROUTE_OK cycles", result.cycles)
