"""GSI APU v2 / G2 target description — @allo.unit nested description.

Hardware model (from ~/shared/accelerator-hub/gsi-apu-g2/docs):
  - L1: 3072 rows x 65536 bits/row on-chip SRAM. All GTML compute works here.
  - Every GTML vector is 65536 elements, partitioned into 16 groups of 4096.
  - System memory (L5/DRAM) is the I/O tier; 256-byte aligned buffers.

Unlike G1 there is no host/device split — a G2 application is a single C++
file that drives the `G2Gtml` singleton. The backend choice (l1_sim, apu_sim,
apu) is a build-time knob. On this server only l1_sim is available
(`gsi-g2-l1sim` docker image). l1_sim is a *functional* model — it checks
correctness of L1 ops but does not produce cycle counts.

Accordingly, `latency` / `energy_pJ` below are placeholders: they are kept
non-zero so the lowering still produces a well-typed perf IR, but they are
flagged via `caps["perf_is_placeholder"] = True`. Performance experiments on
G2 require `apu_sim` or real hardware — neither is available here. See
report 08 §4.5 and report 05 for the environment details.

Layout axes (mapped onto LinearLayout input dim names, root-to-leaf):
    l1_row    : which row of L1 the container starts at (up to 3072 rows)
    group     : which of the 16 groups inside a 64K vector (log2(16)  =  4 bits)
    element   : bit position within a 64K vector            (log2(65536)= 16 bits)

Surface change vs. the previous flat ``Target(...)`` + ``t.memory(...)`` +
``t.op(...)`` form: this file now declares the 3-level grid structurally
(``l1_row(3072) -> group(16) -> element(65536, simd)``). All ``gtml.*`` ops
fire at the ``element`` SIMD leaf. The chip-wide ``l1`` SRAM and the I/O
``system`` memory are routed through ``host_memories=`` on ``@allo.target``
so that ``build_from_grid`` does NOT multiply their ``parallel_units`` by the
tree extents — preserving the original ``parallel_units=1`` semantics for
``l1`` (chip-scope) and ``system`` (host-scope).

Note: ``Target.parallel_units`` becomes the product of all tree extents
(3072 * 16 * 65536) under the nested form, vs. 65536 in the flat form. Cycle
counts are placeholder anyway (``caps["perf_is_placeholder"]=True``), so
this change does not affect any functional test. The legacy axis metadata
(``caps["axes"]``, ``caps["axis_sizes"]``) is preserved verbatim for layout
passes and external consumers that read it.
"""
from ..target import Memory, Op


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_apu_v2():
    # Use ``allo.unit`` for the structural surface. Import lazily and via a
    # file-load fallback so this backend stays importable in environments
    # without a built ``allo._mlir`` C-extension (matches the pattern used in
    # ``experiments/allo/tests/test_unit_decorators.py`` and ``samsung.py``).
    tn = _load_allo_unit()

    @tn.target(
        "gsi_apu_v2_g2",
        host_memories=[
            # System DRAM (the I/O tier; 256-byte aligned buffers).
            Memory("system", capacity_bytes=1 << 34, scope="host"),
            # On-chip L1 SRAM: 3072 rows x 65536 bits/row, chip-wide
            # (parallel_units=1). Routed through host_memories= so that
            # build_from_grid leaves parallel_units alone (would otherwise
            # become 3072 if attached at the l1_row level).
            Memory("l1", capacity_bytes=(3072 * 65536) // 8, scope="chip",
                   lanes=65536, parallel_units=1),
        ],
        caps=dict(has_mac=True, has_relu=False, has_exp=True, has_div=True,
                  has_reduce_max=True, has_softmax=True,
                  has_host_fallback=False,                  # no host split on G2
                  is_real_hardware=False,
                  is_simulator_only=True,
                  perf_is_placeholder=True),                # l1_sim = functional only
    )
    @tn.unit(mapping=[3072])
    def l1_row():
        @tn.unit(mapping=[16])
        def group():
            @tn.unit(mapping=[65536], mode="simd")
            def element():
                # GTML op surface (a small slice; the full API has ~80
                # functions). latency = 0 by convention because l1_sim does
                # not model cycles; the field is kept so analyze_cost still
                # produces a per-op record. Do NOT edit any latency,
                # energy_pJ, or emit string — they are placeholders that the
                # perf_is_placeholder cap gates downstream analyses on.
                tn.op("gtml.load",         lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.load({sys_buf}, {vp});")
                tn.op("gtml.store",        lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.store({vp}, {sys_buf});")
                tn.op("gtml.copy_to_l1",   lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="ref.copy_to_l1({vp_ref});")
                tn.op("gtml.copy_from_l1", lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="ref.copy_from_l1({vp_ref});")
                tn.op("gtml.add",          lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.add({a}, {b}, {c});")
                tn.op("gtml.sub",          lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.sub({a}, {b}, {c});")
                tn.op("gtml.mul",          lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.mul({a}, {b}, {c});")
                tn.op("gtml.div",          lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.div({a}, {b}, {c});")
                tn.op("gtml.exp",          lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.exp({a}, {b});")
                tn.op("gtml.reduce_max",   lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.max({a}, {b});")
                tn.op("gtml.reduce_sum",   lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.sum({a}, {b});")
                tn.op("gtml.softmax",      lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.softmax({a}, {b});")
                tn.op("gtml.matmul",       lanes=65536, latency=0,
                      energy_pJ=0.0,
                      emit="gtml.matmul({a}, {b}, {c});")

    t = l1_row  # @tn.target replaces the function with the built Target.

    # Preserve the layout-pass axis metadata exactly as the flat form
    # exposed it. Order is INNER-to-OUTER, matching apu_v1's convention.
    t.caps["axes"] = ("element", "group", "l1_row")
    t.caps["axis_sizes"] = {"element": 65536, "group": 16, "l1_row": 3072}

    # -- patterns --
    # BUG-4 fix: operand identifiers in the emitted lines are threaded from
    # ``SrcOp.inputs`` / ``SrcOp.output`` rather than hardcoded. Without this,
    # every `gtml.add(a, b, c);` / `gtml.matmul(A, B, C);` line is byte-
    # identical across ops, so multi-op programs (e.g. y1=W1@x1; y2=W2@x2;
    # z=y1+y2) emit a degenerate trace downstream. The emit templates
    # themselves are unchanged — we only change the operand *values* the
    # templates get formatted with, via f-strings over ``s.inputs`` and
    # ``s.output``.
    t.pattern(
        match=lambda s: s.kind == "add",
        lower=lambda s, _: [
            ("gtml.copy_to_l1", {"vp_ref": f"{s.inputs[0]}_ref"}),
            ("gtml.copy_to_l1", {"vp_ref": f"{s.inputs[1]}_ref"}),
            ("gtml.add",        {"a": s.inputs[0], "b": s.inputs[1],
                                 "c": s.output,
                                 "n_elems": _n_elems(s)}),
            ("gtml.copy_from_l1", {"vp_ref": f"{s.output}_ref"}),
        ],
        name="add->gtml_add",
    )
    t.pattern(
        match=lambda s: s.kind == "mul",
        lower=lambda s, _: [
            ("gtml.copy_to_l1", {"vp_ref": f"{s.inputs[0]}_ref"}),
            ("gtml.copy_to_l1", {"vp_ref": f"{s.inputs[1]}_ref"}),
            ("gtml.mul",        {"a": s.inputs[0], "b": s.inputs[1],
                                 "c": s.output,
                                 "n_elems": _n_elems(s)}),
            ("gtml.copy_from_l1", {"vp_ref": f"{s.output}_ref"}),
        ],
        name="mul->gtml_mul",
    )
    t.pattern(
        match=lambda s: s.kind == "scale",
        lower=lambda s, _: [
            ("gtml.copy_to_l1", {"vp_ref": f"{s.inputs[0]}_ref"}),
            ("gtml.mul",        {"a": s.inputs[0], "b": "scale_bcast",
                                 "c": s.output,
                                 "n_elems": _n_elems(s)}),
            ("gtml.copy_from_l1", {"vp_ref": f"{s.output}_ref"}),
        ],
        name="scale->gtml_mul",
    )
    t.pattern(
        match=lambda s: s.kind in ("matmul", "gemv", "mac"),
        lower=lambda s, _: [
            ("gtml.matmul",      {"a": s.inputs[0], "b": s.inputs[1],
                                  "c": s.output,
                                  "n_elems": s.shape[0] * s.shape[-1]}),
        ],
        name="gemv->gtml_matmul",
    )
    t.pattern(
        match=lambda s: s.kind == "softmax",
        lower=lambda s, _: [("gtml.softmax", {"a": s.inputs[0],
                                              "b": s.output,
                                              "n_elems": _n_elems(s)})],
        name="softmax->gtml_softmax",
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
