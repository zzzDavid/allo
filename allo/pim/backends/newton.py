"""Newton (SK-Hynix DRAM-maker AiM, MICRO 2020) target description.

Citation: M. He, C. Song, I. Kim, C. Jeong, S. Kim, I. Park, M. Thottethodi,
T. N. Vijaykumar. "Newton: A DRAM-maker's Accelerator-in-Memory (AiM)
Architecture for Machine Learning." MICRO-53, 2020.
URL: https://microarch.org/micro53/papers/738300a372.pdf

Newton has no open simulator. Architecture + timing come from the paper:
  - Table III: HBM2E-like, 16 banks/channel, 16 bfloat16 multipliers per bank,
    256b column I/O, 1-KB DRAM row; timings tACT=14ns, tRP=14ns; we use
    tFAW=16ns, tRRD=2ns, tCCD=2ns (HBM2E CCD_S).
  - Table I: commands GWRITE, G_ACT, COMP, READRES, plus a per-channel
    activation-function LUT applied by the host before re-broadcast.

Per-row cost (Section III.F):
    t_row = max(tRRD,tFAW)*(n/4 - 1) + tACT + col*tCCD  ns
          = 16*3 + 14 + 32*2  = 126 cycles at 1 GHz
Amortized over 32 col-accesses per row this is ~4 cycles/access, which we
encode as throughput=0.25 on the COMP op (DRAM command-bus limited).
"""
from ..target import Target


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_newton() -> Target:
    # 8 channels × 2 pseudo-channels × 16 banks = 256 PIM units (per Table III).
    t = Target("newton", parallel_units=16 * 16)
    t.cap(has_mac=True, has_af_lut=True, has_exp=False, has_div=False,
          has_reduce_max=False, has_host_fallback=True)

    # -- memories (paper Fig. 5 floorplan) --
    t.memory("bank",          capacity_bytes=8 << 20, scope="bank",
             parallel_units=t.parallel_units)
    t.memory("global_buffer", capacity_bytes=1024, lanes=512, scope="channel",
             parallel_units=16)          # one GB per channel, DRAM-row wide
    t.memory("result_latch",  capacity_bytes=2,  scope="bank",
             parallel_units=t.parallel_units)

    # -- Newton ISA (Table I). Latencies are in 1 GHz DRAM cycles. --
    # COMP throughput 0.25 captures 32-col * 4-cycle effective rate from
    # Section III.F (tCCD=2 + amortized 62ns activation / 32 cols).
    t.op("newton.gwrite",  lanes=16, latency=14, energy_pJ=0.2,
         emit="Newton GWRITE sub-chunk={sub_chunk}")
    t.op("newton.g_act",   lanes=1,  latency=14, energy_pJ=0.1,
         emit="Newton G_ACT bank-group={bg}")
    t.op("newton.comp",    lanes=16, latency=4,  throughput=0.25, energy_pJ=0.5,
         emit="Newton COMP sub-chunk={sub_chunk}")
    t.op("newton.readres", lanes=16, latency=4,  energy_pJ=0.3,
         emit="Newton READRES bank-group={bg}")
    t.op("newton.af",      lanes=16, latency=6,  energy_pJ=0.6,
         emit="Newton AF (LUT)")
    t.op("host.softmax",   lanes=1,  latency=500, energy_pJ=20.0,
         emit="host.call softmax axis=-1 shape={shape}")

    # -- patterns (source ops from allo.pim.ops carry tensor semantics) --
    def _gemv(s, _):
        m, k = s.shape[0], s.shape[-1]
        return [
            ("newton.gwrite",  {"n_elems": k, "sub_chunk": 0}),
            ("newton.comp",    {"n_elems": m * k, "sub_chunk": 0}),
            ("newton.readres", {"n_elems": m, "bg": 0}),
        ]

    t.pattern(match=lambda s: s.kind in ("matmul", "gemv", "mac"),
              lower=_gemv, name="gemv->newton")
    t.pattern(match=lambda s: s.kind == "add",
              lower=lambda s, _: [("newton.comp",
                                   {"n_elems": _n_elems(s), "sub_chunk": 0})],
              name="add->newton_comp")
    t.pattern(match=lambda s: s.kind == "mul",
              lower=lambda s, _: [("newton.comp",
                                   {"n_elems": _n_elems(s), "sub_chunk": 0})],
              name="mul->newton_comp")
    t.pattern(match=lambda s: s.kind == "scale",
              lower=lambda s, _: [("newton.comp",
                                   {"n_elems": _n_elems(s), "sub_chunk": 0})],
              name="scale->newton_comp")
    t.pattern(match=lambda s: s.kind == "relu",
              lower=lambda s, _: [("newton.af", {"n_elems": _n_elems(s)})],
              name="relu->newton_af")
    t.pattern(match=lambda s: s.kind == "softmax",
              lower=lambda s, _: [("host.softmax", {"shape": s.shape})],
              name="softmax->HOST")
    return t
