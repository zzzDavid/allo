"""Samsung HBM-PIM (PIMSimulator) target — @allo.unit nested description.

ISA from experiments/simulators/PIMSimulator/src/PIMCmd.h:
    NOP, ADD, MUL, MAC, MAD, MOV, FILL, JUMP, EXIT.

Topology (Report 11 §5):
    channel(16) -> bg(4) -> bank(4) -> lane(16, simd)
        bank memory (32 MB DRAM) lives on the `bank` level
        grf_a / grf_b (16-lane register files) live on the `lane` level
        all pim.* ops fire at the `lane` level
        capability flags + host_dram + host.softmax stay on the @allo.target

Softmax is not native (no exp/div/reduce-max) so `host.softmax` lives on the
synthetic host root via host_ops= — matches AttAcc / HEAM. Cost numbers are
calibrated E5 values (placeholders preserved in `samsung_placeholders.py.bak`).

Surface change vs. the previous flat ``grid((64, 16), ["channel", "bank"])``
form: this file now declares the 4-level grid that the real HBM-PIM die has
(16 channels x 4 bank-groups x 4 banks x 16 SIMD lanes). The flat form
collapsed channels x bank-groups into a single 64-extent axis and kept the
lane axis implicit via ``lanes=16`` on each op. The backend cost numbers and
op definitions are identical, but ``t.parallel_units`` now reads 4096
(16*4*4*16) instead of 1024 (64*16). Cycle counts are unchanged because every
Samsung op uses the calibrated ``cycles_per_elem`` path, which does not
divide by ``parallel_units``.
"""
from ..target import Memory, Op


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_samsung():
    # Use ``allo.unit`` for the structural surface. Import lazily and via a
    # file-load fallback so this backend stays importable in environments
    # without a built ``allo._mlir`` C-extension (matches the pattern used in
    # ``experiments/allo/tests/test_unit_decorators.py``).
    tn = _load_allo_unit()

    @tn.target(
        "samsung_hbm_pim",
        host_memories=[Memory("host_dram", capacity_bytes=1 << 30,
                              scope="host")],
        host_ops=[Op("host.softmax", lanes=1, latency=500, energy_pJ=20.0,
                     emit="host.call softmax axis=-1 shape={shape}")],
        caps=dict(has_mac=True, has_relu=True, has_exp=False, has_div=False,
                  has_reduce_max=False, has_host_fallback=True),
    )
    @tn.unit(mapping=[16])
    def channel():
        @tn.unit(mapping=[4])
        def bg():
            @tn.unit(mapping=[4])
            def bank():
                # 32 MB DRAM array shared across the 16 SIMD lanes inside
                # this bank.
                tn.memory("bank", capacity_bytes=32 << 20, scope="bank")

                @tn.unit(mapping=[16], mode="simd")
                def lane():
                    # Per-bank GRF banks A/B: each holds 256 bits across 16
                    # SIMD lanes (16 fp16 elements). Capacity matches the
                    # legacy flat builder (256 // 8 = 32 bytes).
                    tn.memory("grf_a", capacity_bytes=256 // 8, lanes=16,
                              scope="grf")
                    tn.memory("grf_b", capacity_bytes=256 // 8, lanes=16,
                              scope="grf")

                    # PIM ISA. Numbers are calibrated E5 values; do not edit
                    # without re-running the simulator regression.
                    tn.op("pim.fill", lanes=16, latency=18,
                          cycles_per_elem=0.00403111, energy_pJ=0.1,
                          emit="PIMCmd(FILL, grf_a<-bank)")
                    tn.op("pim.add", lanes=16, latency=587,
                          cycles_per_elem=0.00465765, energy_pJ=0.3,
                          emit="PIMCmd(ADD, grf_a<-grf_a+bank)")
                    tn.op("pim.mul", lanes=16, latency=587,
                          cycles_per_elem=0.00465765, energy_pJ=0.4,
                          emit="PIMCmd(MUL, grf_a<-grf_a*bank)")
                    tn.op("pim.relu", lanes=16, latency=300,
                          cycles_per_elem=0.00233877, energy_pJ=0.25,
                          emit="PIMCmd(MAC, relu(src))")
                    # pim.mac: GEMV exec scales ~ M*K^2; the gemv pattern
                    # below passes n_elems = M*K^2 so a linear fit holds.
                    tn.op("pim.mac", lanes=16, latency=7384,
                          cycles_per_elem=4.9398e-05, energy_pJ=0.5,
                          emit="PIMCmd(MAC, acc+=src0*src1)")

    t = channel  # @tn.target replaces the function with the built Target.

    # -- patterns (compute closures inherited from typed ops in allo.pim.ops) --
    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac") and s.dtype == "fp16",
        lower=lambda s, _: [("pim.mac",
                             {"n_elems": s.shape[0] * s.shape[-1] * s.shape[-1]})],
        name="gemv->pim_mac",
    )
    t.pattern(
        match=lambda s: s.kind == "add" and s.dtype == "fp16",
        lower=lambda s, _: [("pim.fill", {"n_elems": _n_elems(s)}),
                            ("pim.add",  {"n_elems": _n_elems(s)})],
        name="linalg_add->pim_add",
    )
    t.pattern(
        match=lambda s: s.kind == "mul" and s.dtype == "fp16",
        lower=lambda s, _: [("pim.fill", {"n_elems": _n_elems(s)}),
                            ("pim.mul",  {"n_elems": _n_elems(s)})],
        name="linalg_mul->pim_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "relu",
        lower=lambda s, _: [("pim.fill", {"n_elems": _n_elems(s)}),
                            ("pim.relu", {"n_elems": _n_elems(s)})],
        name="relu->pim_relu",
    )
    t.pattern(
        match=lambda s: s.kind == "scale",
        lower=lambda s, _: [("pim.fill", {"n_elems": _n_elems(s)}),
                            ("pim.mul",  {"n_elems": _n_elems(s)})],
        name="scale->pim_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [("host.softmax", {"shape": s.shape})],
        name="softmax->HOST",
    )
    return t


# ---------------------------------------------------------------------------
# allo.unit loader
# ---------------------------------------------------------------------------
#
# Production install: ``import allo`` works and we use ``allo.unit``.
# Sandbox install: ``allo._mlir`` may not be built; fall back to a direct
# file-load of ``allo/unit.py`` (it is a pure-Python module that only depends
# on ``allo.pim.target``, which is already importable).
def _load_allo_unit():
    import importlib
    import importlib.util
    import os
    import sys
    try:
        return importlib.import_module("allo.unit")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        # backends/ -> pim/ -> allo/ -> unit.py (allo package dir)
        unit_path = os.path.normpath(os.path.join(
            here, "..", "..", "unit.py"))
        if not os.path.isfile(unit_path):
            raise ImportError(
                f"Cannot locate allo.unit (tried import + {unit_path!r}).")
        mod_name = "_allo_unit_standalone"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, unit_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
