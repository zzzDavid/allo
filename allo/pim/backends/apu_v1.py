"""GSI APU v1 (Gemini 1) target description — @allo.unit nested form.

Hardware (from ~/shared/accelerator-hub/gsi-apu/references/docs/D0002 LedaE Specs):
  - 4 APUCs (cores) per chip. Each APUC has 16 vector registers (VR), each
    32K bit-serial-SIMD elements wide. VR ops act on the full 32K elements in
    one issue; there is no partial-lane mode.
  - L1: on-APUC SRAM, staging area between L4 and VRs (VM-style handles
    GVML_VM_0 ...).
  - L4: chip-shared SRAM, reached from the host via GDL memory handles.

Topology (Report 11 §5):
    apuc(4) -> vr(16) -> element(32768, simd)
        l1 (32 KB per APUC) lives on the `apuc` level (shared by VRs and
            bitlines under one APUC).
        vr (64 KB per VR slot, 32K bit-wide rows x 16 VRs x 4 APUCs) lives
            on the `vr` level — addressing-only at vr; the bitline-level
            register file owns the actual storage.
        all bit-serial GVML primitives fire at the `element` level.
        L4 is chip-wide (pu=1) so it lives on the synthetic host root via
            host_memories=, alongside host_dram and host.softmax.

Surface change vs. the previous flat ``Target(...) + t.cap(...) + t.memory()
+ t.op(...)`` form: this file now declares the explicit 3-level grid that
matches Report 11 §5's worked example. The flat form recorded
``parallel_units=32768`` (one VR's lane count); the nested form natively
computes ``apuc * vr * element = 4 * 16 * 32768 = 2_097_152`` lanes for the
whole chip. Cycle predictions on this target are not affected because no
existing test calls ``analyze_cost`` against it; nonetheless we explicitly
restore the flat-form ``parallel_units = 32768`` after construction so
downstream code (and the round-trip fingerprint test) sees the same value
the runtime ``apu_v1_codegen.py`` was written against.

Layout axes (mapped onto Triton-style LinearLayout input dim names, matching
the ``@allo.unit`` axis names):
    element  : bit position within a VR         (log2(32768) = 15 bits)
    vr       : which VR inside the APUC         (log2(16)    =  4 bits)
    apuc     : which core inside the chip       (log2(4)     =  2 bits)

Cost numbers below are rough per-op latencies in APU cycles @ 500 MHz, chosen
to match what the `flo` profiler actually reports on this server for the
example-gvml baseline (see report 08 §4.4 and report 01 for raw traces).

No host fallback is declared for compute ops: every supported source op has a
GVML primitive. softmax has no GVML primitive, so it lifts to host.softmax on
the synthetic root per report 07 §2.6.
"""
from ..target import Memory, Op


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_apu_v1():
    # Use ``allo.unit`` for the structural surface. Import lazily and via a
    # file-load fallback so this backend stays importable in environments
    # without a built ``allo._mlir`` C-extension (matches the pattern used in
    # ``experiments/allo/tests/test_unit_decorators.py``).
    tn = _load_allo_unit()

    @tn.target(
        "gsi_apu_v1",
        # L4 (14 GB chip-shared SRAM, pu=1) and host_dram are both attached
        # above the @allo.unit tree via host_memories=. The flat-form file
        # declared L4 with scope="chip" and parallel_units defaulting to 1;
        # placing it at the apuc level would force pu=4. Keeping it as a
        # host-side memory preserves the original parallel_units=1.
        host_memories=[
            Memory("host_dram", capacity_bytes=1 << 34, scope="host"),
            Memory("l4",        capacity_bytes=14 << 30, scope="chip"),
        ],
        # softmax has no GVML primitive — host fallback per report 07 §2.6.
        host_ops=[
            Op("host.softmax", lanes=1, latency=500, energy_pJ=20.0,
               emit="host.call softmax axis=-1 shape={shape}"),
        ],
        caps=dict(has_mac=True, has_relu=True, has_exp=False, has_div=False,
                  has_reduce_max=False, has_host_fallback=True,
                  is_real_hardware=True),
    )
    @tn.unit(mapping=[4])
    def apuc():
        # L1: 32 KB on-APUC SRAM, addressed via VM handles (GVML_VM_0...).
        # Shared across the 16 VRs and their 32K bitlines under one APUC.
        tn.memory("l1", capacity_bytes=32 << 10, scope="apuc")

        # DMA op fires at the APUC level (one DMA engine per APUC moves
        # 32 KB chunks between L4 and L1). lanes=16384 reflects the
        # half-VR-bit-wide DMA path; preserved from the flat-form spec.
        tn.op("apu.l4_to_l1",  lanes=16384, latency=140, energy_pJ=2.0,
              emit="direct_dma_l4_to_l1_32k({vm}, {l4_ptr});")
        tn.op("apu.l1_to_l4",  lanes=16384, latency=140, energy_pJ=2.0,
              emit="direct_dma_l1_to_l4_32k({l4_ptr}, {vm});")

        @tn.unit(mapping=[16])
        def vr():
            # Per-VR storage: 32K bitlines wide. The "memory" sits at the
            # vr level even though the bit-serial ops fire one level deeper
            # at the bitline (element) — vr owns the addressable unit. We
            # mirror the flat-form Memory("vr", ...) including lanes=32768
            # so the cost model and any introspection still see the same
            # shape.
            tn.memory("vr", capacity_bytes=(32768 * 16) // 8,
                      lanes=32768, scope="vr")

            @tn.unit(mapping=[32768], mode="simd")
            def element():
                # Bit-serial GVML primitives. lanes=32768 matches one VR
                # acting in lockstep across all bitlines. Latency / energy
                # numbers are calibrated against example-gvml (report 08
                # §4.4); do not edit without re-running the live HW
                # regression in tests/e2e/run_apu_v1_layout_ab.py.
                tn.op("apu.l1_to_vr",  lanes=32768, latency=4,   energy_pJ=0.1,
                      emit="gvml_load_16({vr}, {vm});")
                tn.op("apu.vr_to_l1",  lanes=32768, latency=4,   energy_pJ=0.1,
                      emit="gvml_store_16({vm}, {vr});")
                tn.op("apu.add_u16",   lanes=32768, latency=12,  energy_pJ=0.4,
                      emit="gvml_add_u16({dst}, {a}, {b});")
                tn.op("apu.sub_u16",   lanes=32768, latency=12,  energy_pJ=0.4,
                      emit="gvml_sub_u16({dst}, {a}, {b});")
                tn.op("apu.mul_u16",   lanes=32768, latency=16,  energy_pJ=0.5,
                      emit="gvml_mul_u16({dst}, {a}, {b});")
                tn.op("apu.cpy_16",    lanes=32768, latency=2,   energy_pJ=0.05,
                      emit="gvml_cpy_16({dst}, {src});")
                tn.op("apu.relu_u16",  lanes=32768, latency=8,   energy_pJ=0.3,
                      emit="/* relu via gvml_cmp_gt + gvml_cmov: see course101 */")
                # Broadcast a scalar (one L1 element) across all 32K lanes of
                # a VR. Used by the gemv lowering to splat x[k] before the
                # elementwise MAC. Latency/energy chosen conservatively at
                # the same order as gvml_load_16 (it is a load-with-splat on
                # bit-serial HW; no real calibration yet — flagged for
                # recalibration once live HW regression grows a GEMV case).
                tn.op("apu.bcast_scalar_u16", lanes=32768, latency=4,
                      energy_pJ=0.1,
                      emit="gvml_bcast_scalar_u16({dst}, {vm}, {idx});")

    t = apuc  # @tn.target replaces the function with the built Target.

    # --- preserve flat-form values that the existing runtime / fingerprint
    # tests expect ---
    #
    # 1. parallel_units: flat form recorded 32768 (a single VR's lane count).
    #    The nested tree natively computes 4*16*32768 = 2_097_152. Override
    #    so analyze_cost (and any introspection) sees the original value.
    t.parallel_units = 32768
    # 2. caps["axes"] / caps["axis_sizes"]: redundant with the @allo.unit
    #    tree (t.axes already lists ['apuc', 'vr', 'element']) but the
    #    flat-form file stamped them into caps. Linear-layout binders in the
    #    repo read the @allo.unit tree directly when present, but we keep
    #    the caps entries to match the flat-form fingerprint exactly and to
    #    avoid breaking any external consumer that imports build_apu_v1.
    t.caps["axes"] = ("element", "vr", "apuc")
    t.caps["axis_sizes"] = {"element": 32768, "vr": 16, "apuc": 4}

    # -- patterns --
    # Each elementwise source op lowers to a DMA-in / load / compute / store /
    # DMA-out sequence, one per VR-worth of data. For N > 32K the layout step
    # decides how many VRs to use per task body (see apu_v1_codegen.py).
    t.pattern(
        match=lambda s: s.kind == "add",
        lower=lambda s, _: [
            ("apu.l4_to_l1",  {"vm": "GVML_VM_0", "l4_ptr": "a_L4"}),
            ("apu.l1_to_vr",  {"vr": "VR0", "vm": "GVML_VM_0"}),
            ("apu.l4_to_l1",  {"vm": "GVML_VM_0", "l4_ptr": "b_L4"}),
            ("apu.l1_to_vr",  {"vr": "VR1", "vm": "GVML_VM_0"}),
            ("apu.add_u16",   {"dst": "VR2", "a": "VR0", "b": "VR1",
                               "n_elems": _n_elems(s)}),
            ("apu.vr_to_l1",  {"vm": "GVML_VM_0", "vr": "VR2"}),
            ("apu.l1_to_l4",  {"vm": "GVML_VM_0", "l4_ptr": "c_L4"}),
        ],
        name="add->gvml_add_u16",
    )
    t.pattern(
        match=lambda s: s.kind == "mul",
        lower=lambda s, _: [
            ("apu.l4_to_l1",  {"vm": "GVML_VM_0", "l4_ptr": "a_L4"}),
            ("apu.l1_to_vr",  {"vr": "VR0", "vm": "GVML_VM_0"}),
            ("apu.l4_to_l1",  {"vm": "GVML_VM_0", "l4_ptr": "b_L4"}),
            ("apu.l1_to_vr",  {"vr": "VR1", "vm": "GVML_VM_0"}),
            ("apu.mul_u16",   {"dst": "VR2", "a": "VR0", "b": "VR1",
                               "n_elems": _n_elems(s)}),
            ("apu.vr_to_l1",  {"vm": "GVML_VM_0", "vr": "VR2"}),
            ("apu.l1_to_l4",  {"vm": "GVML_VM_0", "l4_ptr": "c_L4"}),
        ],
        name="mul->gvml_mul_u16",
    )
    t.pattern(
        match=lambda s: s.kind == "relu",
        lower=lambda s, _: [
            ("apu.l4_to_l1",  {"vm": "GVML_VM_0", "l4_ptr": "a_L4"}),
            ("apu.l1_to_vr",  {"vr": "VR0", "vm": "GVML_VM_0"}),
            ("apu.relu_u16",  {"n_elems": _n_elems(s)}),
            ("apu.vr_to_l1",  {"vm": "GVML_VM_0", "vr": "VR2"}),
            ("apu.l1_to_l4",  {"vm": "GVML_VM_0", "l4_ptr": "c_L4"}),
        ],
        name="relu->gvml_relu",
    )
    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [("host.softmax", {"shape": s.shape})],
        name="softmax->HOST",
    )
    # gemv (y[M] = W[M,K] @ x[K]) on a bit-serial 32K-wide SIMD substrate:
    # GVML has no native matmul, so this lowers to a K-unrolled MAC loop. For
    # each k in [0, K) we (1) load column W[:, k] from L4 -> L1 -> VR_W,
    # (2) broadcast the scalar x[k] across all lanes into VR_X, (3) multiply
    # elementwise (VR_W * VR_X -> VR_T), and (4) accumulate into VR_Y
    # (VR_Y += VR_T). Finally VR_Y is stored back out to L4. This mirrors
    # APU v2's `gemv->gtml_matmul` pattern in structure, but expanded into
    # the primitive GVML ops the v1 ISA actually has. K is taken from
    # `s.shape[-1]` — the Matmul SrcOp carries shape=(M, K) for GEMV.
    def _gemv_lower(s, _):
        M, K = s.shape[0], s.shape[-1]
        instrs = []
        # Initialize the accumulator VR to 0 by copying from a pre-zeroed VR.
        # We reuse apu.cpy_16 as the zero-init path (VR3 holds the running
        # sum; we assume VR15 is the architectural zero register on Gemini 1
        # — see the GVML programmer's guide). Keeping this as a copy avoids
        # introducing a new "zero" op.
        instrs.append(("apu.cpy_16", {"dst": "VR3", "src": "VR15",
                                      "n_elems": M}))
        for k in range(K):
            # W column k: L4 -> L1 -> VR0
            instrs.append(("apu.l4_to_l1",
                           {"vm": "GVML_VM_0",
                            "l4_ptr": f"W_L4+{k}*M"}))
            instrs.append(("apu.l1_to_vr",
                           {"vr": "VR0", "vm": "GVML_VM_0"}))
            # x[k]: broadcast scalar across the 32K lanes of VR1.
            instrs.append(("apu.bcast_scalar_u16",
                           {"dst": "VR1", "vm": "GVML_VM_1",
                            "idx": k}))
            # VR2 = VR0 * VR1 (elementwise along the M axis).
            instrs.append(("apu.mul_u16",
                           {"dst": "VR2", "a": "VR0", "b": "VR1",
                            "n_elems": M}))
            # VR3 += VR2.
            instrs.append(("apu.add_u16",
                           {"dst": "VR3", "a": "VR3", "b": "VR2",
                            "n_elems": M}))
        # Store accumulator out: VR3 -> L1 -> L4.
        instrs.append(("apu.vr_to_l1",
                       {"vm": "GVML_VM_0", "vr": "VR3"}))
        instrs.append(("apu.l1_to_l4",
                       {"vm": "GVML_VM_0", "l4_ptr": "y_L4"}))
        return instrs

    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac"),
        lower=_gemv_lower,
        name="gemv->gvml_mac_unroll",
    )
    return t


# ---------------------------------------------------------------------------
# allo.unit loader
# ---------------------------------------------------------------------------
#
# Production install: ``import allo`` works and we use ``allo.unit``.
# Sandbox install: ``allo._mlir`` may not be built; fall back to a direct
# file-load of ``allo/unit.py`` (it is a pure-Python module that only depends
# on ``allo.pim.target``, which is already importable). This mirrors the
# loader in samsung.py.
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
