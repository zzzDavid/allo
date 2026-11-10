import numpy as np, allo
from allo.pim import build_samsung_target, samsung_cost
from allo import spmw_simenv
import importlib.util as _u; _s = _u.spec_from_file_location("gemv_src", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "gemv_src.py")); _m = _u.module_from_spec(_s); _s.loader.exec_module(_m); gemv_m4096_k256 = _m.gemv_m4096_k256
compiled = allo.compile(gemv_m4096_k256, build_samsung_target(), samsung_cost)
rng = np.random.default_rng(0)
W = (rng.standard_normal((4096, 256)) * 0.1).astype(np.float16)
x = (rng.standard_normal(256) * 0.1).astype(np.float16)
y = np.zeros(4096, np.float16)
r = compiled(W, x, y)
err = np.abs(y.astype(np.float32) - W.astype(np.float32) @ x.astype(np.float32)).max()
print("pimsim_root", spmw_simenv.pimsim_root())
print("RunResult kind", r.kind, "cycles", r.cycles, "max abs err", float(err))
