import numpy as np, allo
from allo.dataflow import region as _df_region
from allo.ir.types import int32
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target
print("allo from", allo.__file__)
M, K = 768, 128; MAP = [8]; ROWS = M // MAP[0]
@_df_region()
def upmem_gemv(W: int32[M, K], x: int32[K], y: int32[M]):
    @allo.work(mapping=MAP, args=[W, x, y])
    def mv(lW: int32[M, K], lx: int32[K], ly: int32[M]):
        (dpu,) = allo.get_wid()
        for i in range(ROWS):
            ly[dpu * ROWS + i] = 0
            for j in range(K):
                ly[dpu * ROWS + i] += lW[dpu * ROWS + i, j] * lx[j]
compiled = allo.compile(upmem_gemv, build_upmem_target(), upmem_cost)
rng = np.random.default_rng(2)
W = rng.integers(-1000, 1000, (M, K), dtype=np.int32); x = rng.integers(-1000, 1000, K, dtype=np.int32)
y = np.full(M, 123456, np.int32)
r = compiled(W, x, y)
exact = bool(np.array_equal(y, (W.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)))
print("RunResult kind", r.kind, "backend", r.backend, "cycles", r.cycles, "exact", exact, "sdk_build", r.extra.get("sdk_build"))
# Contract (round 108): the program executed on the UPMEM simulator (kind "simulator", backend
# "upmem", a positive integer cycle count, per-segment execution records present) and the int32
# outputs are exact.
assert r.kind == "simulator" and r.backend == "upmem", (r.kind, r.backend)
assert isinstance(r.cycles, int) and r.cycles > 0, r.cycles
assert r.extra.get("upmem"), "no UPMEM execution records"
assert exact, "UPMEM outputs are not exact"
print("UPMEM_CHECK_OK")
