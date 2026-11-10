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
W = np.zeros((M0, K0), np.float16); x = np.zeros(K0, np.float16); y = np.full(M0, 3.0, np.float16)
r = compiled(W, x, y)
print("RunResult kind", r.kind, "backend", r.backend, "cycles", r.cycles)
# Contract (round 108): a virtual run reports kind/backend "virtual", its cycle count is the
# selected plan's estimate (a positive integer), and it computes no data: the caller's output
# array is left exactly as passed.
assert r.kind == "virtual" and r.backend == "virtual", (r.kind, r.backend)
assert isinstance(r.cycles, int) and r.cycles > 0 and r.cycles == compiled.estimate().cycles, (r.cycles, compiled.estimate().cycles)
assert len(compiled.compiled.cmds) > 0 and len(compiled.placement_ranking.kernels) == 1
assert np.all(y == 3.0), "a virtual run must not write the caller's output"
tpl = res.files("allo.pim").joinpath("templates/apu_g2/CMakeLists.txt")
print("apu_g2 template shipped", tpl.is_file())
print("ir template shipped", res.files("allo").joinpath("ir/template/softmax_impl.mlir").is_file())
print("simulators_dir", spmw_simenv.simulators_dir())
print("samsung reason", spmw_simenv.samsung_unavailable_reason())
# Every imported module must come from the installed package, none from the source checkout;
# the probe script itself (__main__) lives in the checkout and is not an import.
leaks = sorted({m.__file__ for name, m in list(sys.modules.items())
                if name != "__main__" and getattr(m, "__file__", None) and "spmw-for-pim" in m.__file__})
print("modules loaded from the checkout:", leaks)
# Contract: the shipped assets are present in the installed package, nothing was imported from
# the source checkout, and the package itself is the installed copy.
assert tpl.is_file(), "apu_g2 template not shipped"
assert res.files("allo").joinpath("ir/template/softmax_impl.mlir").is_file(), "ir template not shipped"
assert leaks == [], leaks
assert "spmw-for-pim" not in allo.__file__, allo.__file__
print("FRESH_INSTALL_CHECK_OK")
