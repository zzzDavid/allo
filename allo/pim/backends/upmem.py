"""UPMEM uPIMulator target — @allo.unit nested-decorator description.

Tree shape (Report 11 §5):
    rank (mapping=[1])
      └── dpu (mapping=[2048])           — MRAM (64 MB) and WRAM (64 KB) live here
            └── tasklet (mapping=[24], mode="mimd")  — RISC core, all ops fire here

The tasklet level is `mode="mimd"` because UPMEM tasklets run independent
control flow synchronized only at explicit DMA / barrier boundaries; this
matches the previous flat builder's implicit MIMD treatment.

The host-side caps + patterns + cost numbers are unchanged from the old flat
form: this port is a surface-syntax change only and the backend runtime keeps
consuming the same `Target` data structure.

Cost model (calibrated E5): `dpu.add` / `dpu.mul` fitted against
Logic[0_0_0]_logic_cycle from real uPIMulator runs (N ∈ {256, 512, 1024},
num_dpus=1, num_tasklets=16). The large baseline latency captures host→DPU
boot / barrier_wait / DMA serialization overhead. Non-add/mul ops keep
analytic placeholders (no sweep this pass).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def _load_allo_unit():
    """Import the ``allo.unit`` decorator module without forcing the full
    ``import allo`` (which pulls in MLIR and is unbuildable in some envs).

    Mirrors the loader pattern in
    ``experiments/allo/tests/test_unit_decorators.py``.
    """
    try:
        return importlib.import_module("allo.unit")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        # backends/ -> pim/ -> allo/ -> unit.py (allo package dir)
        unit_path = os.path.normpath(os.path.join(
            here, "..", "..", "unit.py"))
        spec = importlib.util.spec_from_file_location(
            "allo_unit_standalone", unit_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


tn = _load_allo_unit()


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_upmem():
    @tn.target("upmem_dpu",
               caps=dict(has_mac=True, has_exp=True, has_div=True,
                         has_reduce_max=True, has_softmax=True))
    @tn.unit(mapping=[1])
    def rank():
        @tn.unit(mapping=[2048])
        def dpu():
            tn.memory("mram", bytes=64 << 20, scope="bank")
            tn.memory("wram", bytes=64 << 10, scope="bank")

            @tn.unit(mapping=[24], mode="mimd")
            def tasklet():
                tn.op("dpu.add", lanes=1, latency=29102,
                      cycles_per_elem=3.25195, energy_pJ=0.08,
                      emit="for(i) c[i]=a[i]+b[i]; // elems={n_elems}")
                tn.op("dpu.sub", lanes=1, latency=29102,
                      cycles_per_elem=3.25195, energy_pJ=0.08,
                      emit="for(i) c[i]=a[i]-b[i]; // elems={n_elems}")
                tn.op("dpu.mul", lanes=1, latency=63025,
                      cycles_per_elem=4.26869, energy_pJ=0.10,
                      emit="for(i) c[i]=a[i]*b[i]; // elems={n_elems}")
                # dpu.fmac not separately measured; scale mul numbers by the
                # observed add:mul:fmac ratio implied by placeholder
                # latencies (40/12).
                tn.op("dpu.fmac", lanes=1, latency=63025,
                      cycles_per_elem=14.23, energy_pJ=0.35,
                      emit="for(k) acc=fadd_hf(acc, fmul_hf(a[k], b[k])); // elems={n_elems}")
                tn.op("dpu.relu", lanes=1, latency=29102,
                      cycles_per_elem=1.18, energy_pJ=0.04,
                      emit="for(i) c[i]=a[i]>0?a[i]:0; // elems={n_elems}")
                tn.op("dpu.exp", lanes=1, latency=29102,
                      cycles_per_elem=17.77, energy_pJ=0.60,
                      emit="for(i) c[i]=exp_poly_hf(a[i]); // elems={n_elems}")
                tn.op("dpu.reduce_max", lanes=1, latency=29102,
                      cycles_per_elem=1.48, energy_pJ=0.05,
                      emit="for(i) m=fmax_hf(m,a[i]); // elems={n_elems}")
                tn.op("dpu.reduce_sum", lanes=1, latency=29102,
                      cycles_per_elem=1.48, energy_pJ=0.05,
                      emit="for(i) s=fadd_hf(s,a[i]); // elems={n_elems}")
                tn.op("dpu.div", lanes=1, latency=29102,
                      cycles_per_elem=23.69, energy_pJ=0.80,
                      emit="for(i) c[i]=fdiv_hf(a[i],s); // elems={n_elems}")

    t = rank   # @tn.target replaces the decorated function with the Target

    # -- patterns (unchanged from the flat form) --
    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac"),
        lower=lambda s, _: [("dpu.fmac", {"n_elems": s.shape[0] * s.shape[-1]})],
        name="gemv->dpu_fmac",
    )
    t.pattern(
        match=lambda s: s.kind == "add",
        lower=lambda s, _: [("dpu.add", {"n_elems": _n_elems(s)})],
        name="add->dpu_add",
    )
    t.pattern(
        match=lambda s: s.kind == "mul",
        lower=lambda s, _: [("dpu.mul", {"n_elems": _n_elems(s)})],
        name="mul->dpu_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "relu",
        lower=lambda s, _: [("dpu.relu", {"n_elems": _n_elems(s)})],
        name="relu->dpu_relu",
    )
    t.pattern(
        match=lambda s: s.kind == "scale",
        lower=lambda s, _: [("dpu.mul", {"n_elems": _n_elems(s)})],
        name="scale->dpu_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [
            ("dpu.reduce_max", {"n_elems": _n_elems(s)}),
            ("dpu.sub",        {"n_elems": _n_elems(s)}),
            ("dpu.exp",        {"n_elems": _n_elems(s)}),
            ("dpu.reduce_sum", {"n_elems": _n_elems(s)}),
            ("dpu.div",        {"n_elems": _n_elems(s)}),
        ],
        name="softmax->dpu_fused",
    )
    return t
