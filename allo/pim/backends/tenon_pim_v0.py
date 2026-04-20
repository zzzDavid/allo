"""Tenon-PIM-v0 target description (hypothetical DRAM-PIM, no simulator).

Spec from experiments/reports/10-generate-novel-pim-via-allo.md §7.1:
  - 8 channels x 4 bank groups x 8 banks = 256 parallel units.
  - Leaf: 8 fp16 lanes + MAC + 16-entry on-die LUT activation unit
    (novel vs. Samsung/AiM which have no on-die LUT).
  - Channel-wide shared buffer of 128 B (smaller than AiM's GB).

Softmax policy: ON-DEVICE via LUT. The on-die LUT evaluates exp in 1
cycle/elem, which makes the on-device fused path preferable to the
host fallback Samsung/AiM must take. Sequence:
  reduce_max -> sub -> lut_act(exp) -> reduce_sum -> div.
"""
from ..target import Target


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_tenon_pim_v0() -> Target:
    t = Target("tenon_pim_v0", parallel_units=8 * 4 * 8)  # 256 banks
    t.cap(has_mac=True, has_relu=True, has_exp=True, has_div=True,
          has_reduce_max=True, has_softmax=True, has_lut_act=True)

    # -- memories --
    t.memory("host_dram", capacity_bytes=1 << 30, scope="host")
    t.memory("bank",     capacity_bytes=16 << 20, scope="bank",
             parallel_units=t.parallel_units)
    t.memory("grf",      capacity_bytes=8 * 2, lanes=8, scope="grf",
             parallel_units=t.parallel_units)
    t.memory("chan_buf", capacity_bytes=128, scope="channel", parallel_units=8)

    # -- ISA (8 fp16 lanes per leaf) --
    t.op("tp.fill",       lanes=8, latency=1, energy_pJ=0.08,
         emit="TP FILL grf<-bank elems={n_elems}")
    t.op("tp.add",        lanes=8, latency=2, energy_pJ=0.25,
         emit="TP ADD  grf<-grf+bank elems={n_elems}")
    t.op("tp.mul",        lanes=8, latency=3, energy_pJ=0.35,
         emit="TP MUL  grf<-grf*bank elems={n_elems}")
    t.op("tp.mac",        lanes=8, latency=4, energy_pJ=0.45,
         emit="TP MAC  acc+=a*b elems={n_elems}")
    t.op("tp.relu",       lanes=8, latency=2, energy_pJ=0.22,
         emit="TP RELU grf<-max(grf,0) elems={n_elems}")
    t.op("tp.lut_act",    lanes=8, latency=1, energy_pJ=0.30,
         emit="TP LUT_ACT(exp) grf<-LUT[grf] elems={n_elems}")
    t.op("tp.reduce_max", lanes=8, latency=5, energy_pJ=0.20,
         emit="TP REDUCE_MAX chan_buf<-max(grf) elems={n_elems}")
    t.op("tp.reduce_sum", lanes=8, latency=5, energy_pJ=0.20,
         emit="TP REDUCE_SUM chan_buf<-sum(grf) elems={n_elems}")
    t.op("tp.sub",        lanes=8, latency=2, energy_pJ=0.25,
         emit="TP SUB  grf<-grf-chan_buf elems={n_elems}")
    t.op("tp.div",        lanes=8, latency=10, energy_pJ=1.0,
         emit="TP DIV  grf<-grf/chan_buf elems={n_elems}")

    # -- patterns --
    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac"),
        lower=lambda s, _: [("tp.mac", {"n_elems": s.shape[0] * s.shape[-1]})],
        name="gemv->tp_mac",
    )
    t.pattern(
        match=lambda s: s.kind == "add",
        lower=lambda s, _: [("tp.fill", {"n_elems": _n_elems(s)}),
                            ("tp.add",  {"n_elems": _n_elems(s)})],
        name="linalg_add->tp_add",
    )
    t.pattern(
        match=lambda s: s.kind == "mul",
        lower=lambda s, _: [("tp.fill", {"n_elems": _n_elems(s)}),
                            ("tp.mul",  {"n_elems": _n_elems(s)})],
        name="linalg_mul->tp_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "scale",
        lower=lambda s, _: [("tp.fill", {"n_elems": _n_elems(s)}),
                            ("tp.mul",  {"n_elems": _n_elems(s)})],
        name="scale->tp_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "relu",
        lower=lambda s, _: [("tp.fill", {"n_elems": _n_elems(s)}),
                            ("tp.relu", {"n_elems": _n_elems(s)})],
        name="relu->tp_relu",
    )
    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [
            ("tp.reduce_max", {"n_elems": _n_elems(s)}),
            ("tp.sub",        {"n_elems": _n_elems(s)}),
            ("tp.lut_act",    {"n_elems": _n_elems(s)}),
            ("tp.reduce_sum", {"n_elems": _n_elems(s)}),
            ("tp.div",        {"n_elems": _n_elems(s)}),
        ],
        name="softmax->tp_lut_fused",
    )
    return t
