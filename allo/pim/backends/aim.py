"""SK-Hynix AiM (Ramulator-2.0) target — @allo.unit nested description.

ISR set from aim_simulator/README.md: WR_*, RD_*, COPY_*, MAC_{SBK,ABK},
EWMUL, EWADD, AF, SYNC, EOC. Hardware hierarchy per Samsung/SK-Hynix PIM
conventions and the comparison table in
``paper/latex/sections/abstraction.tex`` §3.3:

    32 channels x 4 bank groups x 8 banks x 16-lane fp16 MAC unit
    = 16,384 leaf parallel units

The Global Buffer is one memory per channel; the MAC result register is
per bank. Softmax is a host fallback (no exp/div/reduce-max). Cost numbers
calibrated in E5 (placeholders preserved in ``aim_placeholders.py.bak``).

Surface change vs. the previous flat ``Grid``/``Leaf`` form: this file now
declares a 4-level grid structurally, exposing the full
channel -> bank-group -> bank -> mac_unit hierarchy that the comparison
table documents but the old 2-level form (``32 channel x 16 bank``)
collapsed. Op fields (``latency``, ``cycles_per_elem``, ``energy_pJ``,
``emit``) and the host-fallback wiring are preserved verbatim — see
``tests/test_aim_port.py`` for the field-by-field equivalence proof.

Per-level op placement (justified per Samsung/SK-Hynix PIM conventions):
  - channel: ``aim.wr_gb`` (writes to per-channel Global Buffer) and
    ``aim.wr_bias`` (channel-broadcast bias setup; the WR_BIAS ISR is
    a chip-level setup op, not bank-local).
  - bg     : ``aim.mac_abk`` ("MAC all banks") spans every bank in a
    bank group; placed at the bg level so the cost-model walk sees one
    issue per BG, not per bank.
  - bank   : per-bank ops -- ``aim.ewmul``, ``aim.ewadd`` (read/write
    bank arrays through GPRs), ``aim.af`` (activation through the per-
    bank MAC datapath), ``aim.rd_mac`` (drains the per-bank MAC
    accumulator). Per-bank memories ``bank`` and ``mac_reg`` live here.
  - mac_unit: SIMD lanes inside the bank MAC unit; no per-lane ISA in
    AiM (lanes=16 is captured on each op's ``lanes`` field). Empty leaf.

Note: ``Target.parallel_units`` becomes 32 * 4 * 8 * 16 = 16,384 under
the nested form, vs. 32 * 16 = 512 under the previous 2-level form. This
is a structural correction (the old form silently elided the bank-group
level and the SIMD lane axis). The cost model only consults
``Target.parallel_units`` as a denominator for ops with
``cycles_per_elem == 0``; every such op in the AiM ISA is invoked with
``n_elems`` <= ``parallel_units`` in both old and new forms, so the
analytic cost evaluates identically (``ceil(n/pu) == 1`` either way).
The patterns assert this empirically via ``test_aim_port.py``.
"""
from ..target import Memory, Op


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_aim():
    # Use ``allo.unit`` for the structural surface. Import lazily and via a
    # file-load fallback so this backend stays importable in environments
    # without a built ``allo._mlir`` C-extension (matches the pattern used in
    # ``samsung.py`` / ``apu_v2.py``).
    tn = _load_allo_unit()

    @tn.target(
        "skhynix_aim",
        host_memories=[Memory("gpr", capacity_bytes=256 // 8, lanes=16,
                              scope="host")],
        host_ops=[Op("host.softmax", lanes=1, latency=500, energy_pJ=20.0,
                     emit="host.call softmax axis=-1 shape={shape}")],
        caps=dict(has_mac=True, has_af=True, has_ewmul=True, has_ewadd=True,
                  has_exp=False, has_div=False, has_reduce_max=False,
                  has_host_fallback=True),
    )
    @tn.unit(mapping=[32])
    def channel():
        # Per-channel Global Buffer (256 bits across 16 fp16 lanes).
        tn.memory("gb", capacity_bytes=256 // 8, lanes=16, scope="channel")

        # Global-Buffer-side ISA. WR_GB writes the GB; WR_BIAS is the
        # chip-level bias-broadcast setup that primes the per-bank MAC
        # accumulators before MAC_ABK fires.
        tn.op("aim.wr_gb", lanes=16, latency=40, energy_pJ=0.5,
              emit="AiM WR_GB {opsize} {gpr} {mask}")
        tn.op("aim.wr_bias", lanes=16, latency=40, energy_pJ=0.3,
              emit="AiM WR_BIAS {gpr} {mask}")

        @tn.unit(mapping=[4])
        def bg():
            # MAC_ABK ("MAC all banks") spans every bank in this bank
            # group. Negative latency absorbs pipeline overlap with the
            # surrounding wr_gb / wr_bias / rd_mac setup ops so that
            # [wr_gb(40)+wr_bias(40)+mac_abk(?)+rd_mac(39)] matches the
            # full sweep trace's mem_cycles. Calibrated in E5; do NOT
            # edit without re-running the simulator regression.
            tn.op("aim.mac_abk", lanes=16, latency=-24,
                  cycles_per_elem=11.0536, energy_pJ=0.8,
                  emit="AiM MAC_ABK {opsize} {mask} {row}")

            @tn.unit(mapping=[8])
            def bank():
                # 64 MB DRAM array per bank.
                tn.memory("bank", capacity_bytes=64 << 20, scope="bank")
                # Per-bank MAC accumulator register (256 bits across 16
                # fp16 lanes).
                tn.memory("mac_reg", capacity_bytes=256 // 8, lanes=16,
                          scope="bank")

                # Per-bank element-wise / activation / readback ISA.
                tn.op("aim.ewmul", lanes=16, latency=27,
                      cycles_per_elem=6.56823, energy_pJ=0.4,
                      emit="AiM EWMUL {opsize} {mask} {row}")
                tn.op("aim.ewadd", lanes=16, latency=5,
                      cycles_per_elem=0.125, energy_pJ=0.3,
                      emit="AiM EWADD {opsize} {gpr0} {gpr1}")
                tn.op("aim.af", lanes=16, latency=92, energy_pJ=0.6,
                      emit="AiM AF {mask}")
                tn.op("aim.rd_mac", lanes=16, latency=39, energy_pJ=0.4,
                      emit="AiM RD_MAC {gpr} {mask}")

                # SIMD lanes inside the bank MAC unit. AiM has no
                # per-lane ISA (each op above carries lanes=16); leaf
                # body is empty intentionally.
                @tn.unit(mapping=[16], mode="simd")
                def mac_unit():
                    pass

    t = channel  # @tn.target replaces the function with the built Target.

    # -- per-lowering GPR allocator ------------------------------------------
    #
    # BUG-3 fix: the previous pattern bodies hardcoded GPR operand slots
    # (``gpr0=0, gpr1=1``, etc.) regardless of which tensors the SrcOp
    # referenced, so multi-op programs produced aliased traces: e.g. two
    # MAC_ABK lines for ``y1=W1@x1`` and ``y2=W2@x2`` were byte-identical
    # and the EWADD for ``z = y1 + y2`` always read slots 0/1. We now bind
    # each tensor identifier to a distinct GPR slot on first reference and
    # reuse that slot on every subsequent reference.
    #
    # Strategy:
    #   * Tensors that appear at GPR-carrying positions (WR_GB input,
    #     RD_MAC output, EWADD operands) get a slot via ``gpr_of(name)``.
    #   * MAC_ABK's third token is a bank ``row`` (per its emit template
    #     ``AiM MAC_ABK {opsize} {mask} {row}``), not a GPR. We use
    #     ``row_of(name)`` to give each matmul output a distinct bank row
    #     so the two MAC_ABK lines differ textually even though the shadow
    #     semantics pick W from that row.
    #   * AiM exposes 32 GPRs; we reserve the top slot (31) as a dedicated
    #     zero-init bias slot for WR_BIAS so it never collides with a
    #     tensor-bound slot. Unstaged GPRs read as zero in aim_shadow, so
    #     this gives a correct zero bias for pure matmul semantics.
    #   * EWADD in aim_shadow writes ``gpr1 <- gpr0 + gpr1``, so the add's
    #     output tensor aliases the slot of ``inputs[1]``. This keeps the
    #     single-op ``test_aim_vadd_end_to_end`` trace at ``AiM EWADD 1 0 1``
    #     (A->slot0, B->slot1, C aliases slot1).
    #
    # Scope: allocator is a closure in this ``build_aim()`` call, so each
    # fresh ``build_aim()`` starts with an empty allocator. Compiling two
    # different programs against the same target instance will share the
    # allocator (slots keep growing monotonically) — acceptable for the
    # tests which always build a new target per compile.
    _NUM_GPR = 32
    _BIAS_GPR = _NUM_GPR - 1  # reserved: WR_BIAS writes this (reads as 0)

    gpr_slots: dict = {}
    row_slots: dict = {}
    _next_gpr = [0]  # list-wrapped for closure mutation under py3
    _next_row = [0]

    def gpr_of(name: str) -> int:
        if not name:
            # Anonymous / missing tensor name: fall back to slot 0. Should
            # not happen for allo-compiled programs (every SrcOp carries
            # named inputs/output).
            return 0
        if name not in gpr_slots:
            if _next_gpr[0] >= _BIAS_GPR:
                raise RuntimeError(
                    f"AiM GPR allocator exhausted: {_BIAS_GPR} tensor "
                    f"slots used (bias reserved at {_BIAS_GPR}); "
                    f"program references too many distinct tensors.")
            gpr_slots[name] = _next_gpr[0]
            _next_gpr[0] += 1
        return gpr_slots[name]

    def row_of(name: str) -> int:
        if not name:
            return 0
        if name not in row_slots:
            row_slots[name] = _next_row[0]
            _next_row[0] += 1
        return row_slots[name]

    # -- patterns ------------------------------------------------------------
    def _lower_gemv(s, _):
        x = s.inputs[1] if len(s.inputs) >= 2 else ""   # GEMV: W @ x
        y = s.output
        return [
            ("aim.wr_gb",   {"opsize": 1, "gpr": gpr_of(x), "mask": 15,
                             "n_elems": s.shape[-1]}),
            ("aim.wr_bias", {"gpr": _BIAS_GPR, "mask": 15, "n_elems": 1}),
            # All-bank-layout MAC_ABK: per-row cost scales with K (reduction
            # length), not M*K — the bank axis already covers M. Third
            # token is the bank row holding W; each matmul output gets a
            # distinct row so the two lines differ textually.
            ("aim.mac_abk", {"opsize": 1, "mask": 15, "row": row_of(y),
                             "n_elems": s.shape[-1]}),
            ("aim.rd_mac",  {"gpr": gpr_of(y), "mask": 15, "n_elems": 1}),
        ]

    def _lower_add(s, _):
        a = s.inputs[0] if len(s.inputs) >= 1 else ""
        b = s.inputs[1] if len(s.inputs) >= 2 else ""
        ga = gpr_of(a)
        gb = gpr_of(b)
        # EWADD semantics in aim_shadow: gpr1 <- gpr0 + gpr1. The output
        # lands in gpr1's slot, so alias the output tensor to that slot.
        if s.output:
            gpr_slots[s.output] = gb
        return [("aim.ewadd", {"n_elems": _n_elems(s),
                               "opsize": 1, "gpr0": ga, "gpr1": gb})]

    _ew_defaults = dict(opsize=1, mask=15, row=0)
    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac"),
        lower=_lower_gemv,
        name="gemv->aim_mac_abk",
    )
    t.pattern(
        match=lambda s: s.kind == "add",
        lower=_lower_add,
        name="linalg_add->aim_ewadd",
    )
    t.pattern(
        match=lambda s: s.kind == "mul",
        lower=lambda s, _: [("aim.ewmul", {"n_elems": _n_elems(s), **_ew_defaults})],
        name="linalg_mul->aim_ewmul",
    )
    t.pattern(
        match=lambda s: s.kind == "relu",
        lower=lambda s, _: [("aim.af", {"mask": 15, "n_elems": _n_elems(s)})],
        name="relu->aim_af",
    )
    t.pattern(
        match=lambda s: s.kind == "scale",
        lower=lambda s, _: [("aim.ewmul", {"n_elems": _n_elems(s), **_ew_defaults})],
        name="scale->aim_ewmul",
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
