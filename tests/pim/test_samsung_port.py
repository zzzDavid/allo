"""Regression: the Samsung HBM-PIM target description still produces the
same ``Target`` object after the rewrite from flat ``Grid``/``Leaf`` to
nested ``@allo.unit`` decorators. Backend runtime is unchanged; this test
exists to prove the surface change is structurally inert.

The reference is the pre-rewrite flat-form builder, vendored locally as
``_build_samsung_flat_legacy()`` so the comparison stays meaningful even
after the original ``backends/samsung.py`` no longer carries that form.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim.target import (  # noqa: E402
    Memory, Op, Leaf, grid, build_from_grid,
)
from allo.pim.backends.samsung import build_samsung  # noqa: E402


# ---------------------------------------------------------------------------
# Vendored copy of the pre-rewrite flat-form builder. Do not edit unless the
# calibrated cost numbers in samsung.py also move; this is the gold reference.
# ---------------------------------------------------------------------------


def _n_elems(s):
    n = 1
    for d in s.shape:
        n *= d
    return n


def _build_samsung_flat_legacy():
    leaf = Leaf(
        memory=[
            Memory("bank",  capacity_bytes=32 << 20, scope="bank"),
            Memory("grf_a", capacity_bytes=256 // 8, lanes=16, scope="grf"),
            Memory("grf_b", capacity_bytes=256 // 8, lanes=16, scope="grf"),
        ],
        ops=[
            Op("pim.fill", lanes=16, latency=18, cycles_per_elem=0.00403111,
               energy_pJ=0.1, emit="PIMCmd(FILL, grf_a<-bank)"),
            Op("pim.add",  lanes=16, latency=587, cycles_per_elem=0.00465765,
               energy_pJ=0.3, emit="PIMCmd(ADD, grf_a<-grf_a+bank)"),
            Op("pim.mul",  lanes=16, latency=587, cycles_per_elem=0.00465765,
               energy_pJ=0.4, emit="PIMCmd(MUL, grf_a<-grf_a*bank)"),
            Op("pim.relu", lanes=16, latency=300, cycles_per_elem=0.00233877,
               energy_pJ=0.25, emit="PIMCmd(MAC, relu(src))"),
            Op("pim.mac",  lanes=16, latency=7384, cycles_per_elem=4.9398e-05,
               energy_pJ=0.5, emit="PIMCmd(MAC, acc+=src0*src1)"),
        ],
    )
    t = build_from_grid(
        "samsung_hbm_pim",
        grid((64, 16), ["channel", "bank"], child=leaf),
        caps=dict(has_mac=True, has_relu=True, has_exp=False, has_div=False,
                  has_reduce_max=False, has_host_fallback=True),
        host_memories=[Memory("host_dram", capacity_bytes=1 << 30,
                              scope="host")],
        host_ops=[Op("host.softmax", lanes=1, latency=500, energy_pJ=20.0,
                     emit="host.call softmax axis=-1 shape={shape}")],
    )
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
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _memory_fp(m: Memory) -> dict:
    return {
        "name": m.name,
        "capacity_bytes": m.capacity_bytes,
        "lanes": m.lanes,
        "dtype": m.dtype,
        "scope": m.scope,
    }


def _op_fp(o: Op) -> dict:
    return {
        "name": o.name,
        "lanes": o.lanes,
        "latency": o.latency,
        "throughput": o.throughput,
        "cycles_per_elem": o.cycles_per_elem,
        "energy_pJ": o.energy_pJ,
        "emit": o.emit,
    }


# ---------------------------------------------------------------------------
# Equality assertions — every key surface attribute must match.
# ---------------------------------------------------------------------------


def test_samsung_port_name_and_caps_match_legacy():
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    assert new.name == old.name
    assert new.caps == old.caps


def test_samsung_port_memories_match_legacy():
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    new_names = sorted(new.memories)
    old_names = sorted(old.memories)
    assert new_names == old_names, (new_names, old_names)
    for name in old_names:
        assert _memory_fp(new.memories[name]) == _memory_fp(old.memories[name]), \
            f"memory {name!r} diverged"


def test_samsung_port_ops_match_legacy():
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    new_names = sorted(new.ops)
    old_names = sorted(old.ops)
    assert new_names == old_names, (new_names, old_names)
    for name in old_names:
        assert _op_fp(new.ops[name]) == _op_fp(old.ops[name]), \
            f"op {name!r} diverged"


def test_samsung_port_parallel_units_product_equals_4096():
    """The new 4-level tree (16 channels x 4 bg x 4 banks x 16 lanes) makes
    the bank-group axis explicit and exposes the SIMD lane axis that was
    implicit in the old flat form. Total parallel_units therefore grows from
    1024 (= 64 channels x 16 banks) to 4096; document both numbers here so a
    future rewrite cannot silently drop a level."""
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    assert old.parallel_units == 64 * 16 == 1024
    assert new.parallel_units == 16 * 4 * 4 * 16 == 4096
    # Sanity: the new tree carries 4 named axes in root-to-leaf order.
    assert new.axes == ["channel", "bg", "bank", "lane"], new.axes
    # SIMD mode lives only on the leaf level.
    assert new.modes == {"channel": "mimd", "bg": "mimd",
                         "bank": "mimd", "lane": "simd"}, new.modes


def test_samsung_port_host_memory_and_ops_present_on_both():
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    # host_dram and host.softmax are wired via host_memories= / host_ops=
    # kwargs on @allo.target — same as build_from_grid's kwargs.
    assert "host_dram" in new.memories and "host_dram" in old.memories
    assert _memory_fp(new.memories["host_dram"]) == \
        _memory_fp(old.memories["host_dram"])
    assert "host.softmax" in new.ops and "host.softmax" in old.ops
    assert _op_fp(new.ops["host.softmax"]) == _op_fp(old.ops["host.softmax"])


def test_samsung_port_pattern_count_unchanged():
    """Patterns are copied across verbatim by the rewrite. Count + names
    must match (Pattern objects compare by identity, so we go by name)."""
    new = build_samsung()
    old = _build_samsung_flat_legacy()
    new_names = sorted(p.name for p in new.patterns)
    old_names = sorted(p.name for p in old.patterns)
    assert new_names == old_names, (new_names, old_names)


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"]))
