import sys, time, importlib.resources as res
import numpy as np
import allo
from allo.pim import build_samsung_target, samsung_cost
import allo.compiler
from allo import spmw_simenv
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
print("allo from", allo.__file__)
M0, K0, ROWS0 = 4096, 256, 32

@_df_region()
def gemv_m4096_k256(W: fp16[M0, K0], x: fp16[K0], y: fp16[M0]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M0, K0], local_x: fp16[K0], local_y: fp16[M0]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS0
        for i in range(ROWS0):
            acc: fp16 = 0
            for k in range(K0):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc

t = time.perf_counter()
compiled = allo.compile(gemv_m4096_k256, build_samsung_target(), samsung_cost, backend="virtual")
print(f"compile {time.perf_counter() - t:.1f} s ->", type(compiled).__name__)
print("estimate", compiled.estimate().cycles)
print("placement_ranking.kernels", len(compiled.placement_ranking.kernels),
      "bucket_count", compiled.placement_ranking.kernels[0].bucket_count)
print("cmds", len(compiled.compiled.cmds), compiled.compiled.cmds[:2])
W = np.zeros((M0, K0), np.float16); x = np.zeros(K0, np.float16); y = np.zeros(M0, np.float16)
r = compiled(W, x, y)
print("RunResult kind", r.kind, "backend", r.backend, "cycles", r.cycles)
tpl = res.files("allo.pim").joinpath("templates/apu_g2/CMakeLists.txt")
print("apu_g2 template shipped", tpl.is_file())
print("ir template shipped", res.files("allo").joinpath("ir/template/softmax_impl.mlir").is_file())
print("simulators_dir", spmw_simenv.simulators_dir())
print("samsung reason", spmw_simenv.samsung_unavailable_reason())
leaks = sorted({m.__file__ for m in list(sys.modules.values())
                if getattr(m, "__file__", None) and "spmw-for-pim" in m.__file__})
print("modules loaded from the checkout:", leaks)
