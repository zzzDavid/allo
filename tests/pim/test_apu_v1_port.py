"""Round-trip test: the @allo.unit-form APU v1 backend produces a Target
that is field-for-field equivalent to the previous flat ``Target(...) +
t.cap(...) + t.memory(...) + t.op(...)`` form.

Why this lives in `pim_dsl/tests/`: this is the regression that proves the
APU v1 surface-syntax port did not change any externally observable
property of the built Target. The fixture is a vendored copy of the
original flat-form builder, frozen at the pre-port state from
`pimdsl/backends/apu_v1.py` (commit pre-2026-04-20).

The comparison is intentionally narrow:
  - name, parallel_units, caps dict
  - memories: name, capacity_bytes, lanes, dtype, parallel_units, scope
  - ops: name, lanes, latency, throughput, cycles_per_elem, energy_pJ, emit
  - patterns: name + on a small probe set, the same lower(...) sequence

It does NOT compare attributes the @allo.unit form intentionally adds (e.g.
`t.axes`, `t.modes`, `t.tn_root`); the goal is "no regression," not
"identical object." Those new attributes are spot-checked separately.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim import SrcOp, SrcProgram, lower
from allo.pim.target import Target
from allo.pim.backends.apu_v1 import build_apu_v1


# ---------------------------------------------------------------------------
# Vendored flat-form builder (the pre-port state of build_apu_v1)
# ---------------------------------------------------------------------------


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def build_apu_v1_flat_legacy() -> Target:
    """Frozen snapshot of build_apu_v1() before the @allo.unit port.

    Source: pimdsl/backends/apu_v1.py at commit f153331 (2026-04-20). Kept
    in-tree as a regression fixture; do not edit unless intentionally
    re-baselining the port.

    Re-baselined 2026-04-20 (BUG-2 fix): added ``apu.bcast_scalar_u16`` op
    and ``gemv->gvml_mac_unroll`` pattern so the fingerprint stays in sync
    with the v1 backend now that it lowers ``matmul`` / ``gemv`` / ``mac``
    source ops (two of which appear in the MLP-block motif).
    """
    t = Target("gsi_apu_v1", parallel_units=32768)
    t.cap(has_mac=True, has_relu=True, has_exp=False, has_div=False,
          has_reduce_max=False, has_host_fallback=True,
          is_real_hardware=True)
    t.caps["axes"] = ("element", "vr", "apuc")
    t.caps["axis_sizes"] = {"element": 32768, "vr": 16, "apuc": 4}

    t.memory("host_dram", capacity_bytes=1 << 34, scope="host")
    t.memory("l4",  capacity_bytes=14 << 30, scope="chip")
    t.memory("l1",  capacity_bytes=32 << 10, scope="apuc",
             parallel_units=4)
    t.memory("vr",  capacity_bytes=(32768 * 16) // 8, scope="vr",
             lanes=32768, parallel_units=4 * 16)

    t.op("apu.l4_to_l1",  lanes=16384, latency=140, energy_pJ=2.0,
         emit="direct_dma_l4_to_l1_32k({vm}, {l4_ptr});")
    t.op("apu.l1_to_l4",  lanes=16384, latency=140, energy_pJ=2.0,
         emit="direct_dma_l1_to_l4_32k({l4_ptr}, {vm});")
    t.op("apu.l1_to_vr",  lanes=32768, latency=4,   energy_pJ=0.1,
         emit="gvml_load_16({vr}, {vm});")
    t.op("apu.vr_to_l1",  lanes=32768, latency=4,   energy_pJ=0.1,
         emit="gvml_store_16({vm}, {vr});")
    t.op("apu.add_u16",   lanes=32768, latency=12,  energy_pJ=0.4,
         emit="gvml_add_u16({dst}, {a}, {b});")
    t.op("apu.sub_u16",   lanes=32768, latency=12,  energy_pJ=0.4,
         emit="gvml_sub_u16({dst}, {a}, {b});")
    t.op("apu.mul_u16",   lanes=32768, latency=16,  energy_pJ=0.5,
         emit="gvml_mul_u16({dst}, {a}, {b});")
    t.op("apu.cpy_16",    lanes=32768, latency=2,   energy_pJ=0.05,
         emit="gvml_cpy_16({dst}, {src});")
    t.op("apu.relu_u16",  lanes=32768, latency=8,   energy_pJ=0.3,
         emit="/* relu via gvml_cmp_gt + gvml_cmov: see course101 */")
    t.op("apu.bcast_scalar_u16", lanes=32768, latency=4, energy_pJ=0.1,
         emit="gvml_bcast_scalar_u16({dst}, {vm}, {idx});")
    t.op("host.softmax",  lanes=1,     latency=500, energy_pJ=20.0,
         emit="host.call softmax axis=-1 shape={shape}")

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

    def _gemv_lower(s, _):
        M, K = s.shape[0], s.shape[-1]
        instrs = [("apu.cpy_16", {"dst": "VR3", "src": "VR15",
                                  "n_elems": M})]
        for k in range(K):
            instrs.append(("apu.l4_to_l1",
                           {"vm": "GVML_VM_0",
                            "l4_ptr": f"W_L4+{k}*M"}))
            instrs.append(("apu.l1_to_vr",
                           {"vr": "VR0", "vm": "GVML_VM_0"}))
            instrs.append(("apu.bcast_scalar_u16",
                           {"dst": "VR1", "vm": "GVML_VM_1",
                            "idx": k}))
            instrs.append(("apu.mul_u16",
                           {"dst": "VR2", "a": "VR0", "b": "VR1",
                            "n_elems": M}))
            instrs.append(("apu.add_u16",
                           {"dst": "VR3", "a": "VR3", "b": "VR2",
                            "n_elems": M}))
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
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _memories(t: Target):
    return {
        name: {
            "capacity_bytes": m.capacity_bytes,
            "lanes": m.lanes,
            "dtype": m.dtype,
            "parallel_units": m.parallel_units,
            "scope": m.scope,
        }
        for name, m in t.memories.items()
    }


def _ops(t: Target):
    return {
        name: {
            "lanes": o.lanes,
            "latency": o.latency,
            "throughput": o.throughput,
            "cycles_per_elem": o.cycles_per_elem,
            "energy_pJ": o.energy_pJ,
            "emit": o.emit,
        }
        for name, o in t.ops.items()
    }


def _patterns(t: Target):
    return [p.name for p in t.patterns]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_name_and_parallel_units_match():
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    assert new.name == old.name == "gsi_apu_v1"
    assert new.parallel_units == old.parallel_units == 32768, (
        f"parallel_units regressed: new={new.parallel_units} "
        f"old={old.parallel_units}")


def test_caps_dict_matches():
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    # We compare key-by-key so a divergence prints the offending key
    # rather than a 600-character dict diff.
    for k, v in old.caps.items():
        assert k in new.caps, f"caps missing key {k!r}"
        assert new.caps[k] == v, f"caps[{k!r}] regressed: {new.caps[k]!r} != {v!r}"
    extra = set(new.caps) - set(old.caps)
    assert not extra, f"caps gained unexpected keys: {sorted(extra)}"


def test_memories_match():
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    assert _memories(new) == _memories(old)


def test_ops_match():
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    assert _ops(new) == _ops(old)


def test_patterns_match():
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    assert _patterns(new) == _patterns(old)


def test_pattern_lower_sequences_equivalent():
    """Spot-check that each pattern's lower(...) emits the same instruction
    sequence (op names + arg dicts) on a representative source op."""
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    probes = [
        SrcOp(kind="add",     shape=(32768,), inputs=("a", "b"), output="c"),
        SrcOp(kind="mul",     shape=(32768,), inputs=("a", "b"), output="c"),
        SrcOp(kind="relu",    shape=(32768,), inputs=("a",),     output="c"),
        SrcOp(kind="softmax", shape=(64,),    inputs=("a",),     output="b"),
    ]
    for src in probes:
        new_pat = new.find_pattern(src)
        old_pat = old.find_pattern(src)
        assert new_pat is not None and old_pat is not None, src.kind
        assert new_pat.name == old_pat.name, (src.kind, new_pat.name, old_pat.name)
        assert new_pat.lower(src, new) == old_pat.lower(src, old), src.kind


def test_lowered_program_cycles_match():
    """End-to-end: lower the same SrcProgram against both targets; total
    cycles + emitted lines must agree exactly."""
    prog = (SrcProgram()
            .add(SrcOp(kind="add",     shape=(32768,), inputs=("a", "b"), output="c"))
            .add(SrcOp(kind="mul",     shape=(32768,), inputs=("c", "b"), output="d"))
            .add(SrcOp(kind="softmax", shape=(64,),    inputs=("d",),     output="e")))
    new = build_apu_v1()
    old = build_apu_v1_flat_legacy()
    r_new = lower(prog, new)
    r_old = lower(prog, old)
    assert r_new.total_cycles == r_old.total_cycles, (
        f"total_cycles regressed: new={r_new.total_cycles} old={r_old.total_cycles}")
    assert r_new.emitted == r_old.emitted, "emitted text diverged"
    assert r_new.host_ops == r_old.host_ops == 1
    assert r_new.unlowered == r_old.unlowered == []


# ---------------------------------------------------------------------------
# @allo.unit-only attribute spot checks (new structural metadata)
# ---------------------------------------------------------------------------


def test_unit_tree_metadata_present():
    """The @allo.unit form attaches axes / modes / tn_root for downstream
    passes that want the structural tree. The flat form lacks these; we
    just sanity-check they exist and carry the right shape."""
    new = build_apu_v1()
    assert getattr(new, "axes", None) == ["apuc", "vr", "element"]
    assert getattr(new, "modes", None) == {
        "apuc": "mimd", "vr": "mimd", "element": "simd"}
    assert getattr(new, "tn_root", None) is not None


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"]))
