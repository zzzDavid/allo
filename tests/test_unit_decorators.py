# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 tests for the @allo.unit / @allo.target decorator surface.

Covers Report 11 §3.1-3.7 and type-checker rules R1-R4. The critical
assertion is ``test_roundtrip_equivalence_with_grid_builder``: it proves
the decorators produce the same ``Target`` as the existing flat-form
``Grid(...)`` builder, so the five backends need no changes when we port
them to the new surface.
"""
import importlib
import importlib.util
import os
import sys

import pytest


def _load_allo_unit():
    """Load ``allo.unit`` without forcing the full ``import allo``
    (which pulls in the MLIR dialect and is unusable in environments
    without a built Allo). Prefer the canonical ``from allo import
    unit`` path; fall back to a direct file-load of ``allo/unit.py`` so
    this test stays green in CI setups that don't ship MLIR.

    In a real Allo install the first branch succeeds and the fallback
    is dead code.
    """
    # Make sure ``pimdsl`` is importable — the decorator module imports
    # the underlying ``Memory``/``Op``/``Target`` dataclasses from it.
    here = os.path.dirname(os.path.abspath(__file__))
    pim_dsl_dir = os.path.normpath(os.path.join(
        here, "..", "..", "pim_dsl"))
    if os.path.isdir(pim_dsl_dir) and pim_dsl_dir not in sys.path:
        sys.path.insert(0, pim_dsl_dir)
    try:
        tn = importlib.import_module("allo.unit")
    except Exception:
        unit_path = os.path.normpath(os.path.join(
            here, "..", "allo", "unit.py"))
        spec = importlib.util.spec_from_file_location(
            "allo_unit_standalone", unit_path)
        tn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tn)
    return tn


tn = _load_allo_unit()

# Import the underlying target primitives for the round-trip test.
from pimdsl.target import (  # noqa: E402
    Memory, Op, Target, Grid, Leaf, build_from_grid,
)


# ---------------------------------------------------------------------------
# 1. Basic three-level tree
# ---------------------------------------------------------------------------


def _build_basic_three_level():
    @tn.target("toy_three_level")
    @tn.unit(mapping=[8])
    def outer():
        @tn.unit(mapping=[4])
        def middle():
            tn.memory("dram", bytes=4 << 20, scope="bank")
            @tn.unit(mapping=[16], mode="simd")
            def inner():
                tn.memory("grf", elems=8, lanes=16, scope="grf")
                tn.op("pim.add", lanes=16, latency=7, cycles_per_elem=0.1,
                      energy_pJ=0.2)
    return outer


def test_basic_three_level_tree():
    t = _build_basic_three_level()

    # It must BE a Target (replacing the decorated function).
    assert isinstance(t, Target), f"got {type(t).__name__}, not Target"
    assert t.name == "toy_three_level"

    # Axis set (root-to-leaf).
    assert t.axes == ["outer", "middle", "inner"]

    # Modes dict.
    assert t.modes["outer"] == "mimd"
    assert t.modes["middle"] == "mimd"
    assert t.modes["inner"] == "simd"

    # Memories end up in t.memories keyed by name.
    assert "dram" in t.memories
    assert "grf" in t.memories
    dram = t.memories["dram"]
    grf = t.memories["grf"]

    # parallel_units reflects the product of extents from root down to
    # the memory's declaring level. `dram` is declared at `middle`
    # (extents 8 * 4 = 32). `grf` is declared at the innermost `inner`
    # leaf, so parallel_units = 8 * 4 * 16 = 512.
    assert dram.parallel_units == 8 * 4, dram.parallel_units
    assert grf.parallel_units == 8 * 4 * 16, grf.parallel_units

    # Op lives in t.ops.
    assert "pim.add" in t.ops
    add = t.ops["pim.add"]
    assert add.lanes == 16
    assert add.latency == 7

    # Overall parallel_units = total extent (8 * 4 * 16 = 512).
    assert t.parallel_units == 8 * 4 * 16


# ---------------------------------------------------------------------------
# 2. Axis name from function name (Rule R1)
# ---------------------------------------------------------------------------


def test_axis_name_from_function_name():
    """The function __name__ is the axis name."""

    @tn.target("toy_r1")
    @tn.unit(mapping=[2])
    def channel():
        @tn.unit(mapping=[3])
        def bank():
            tn.op("nop", latency=1)

    assert channel.axes == ["channel", "bank"]


def test_r1_rejects_lambda_like_names():
    """Lambdas / anonymous / underscore-prefixed names are rejected."""
    with pytest.raises(TypeError):
        # Simulate a lambda / underscore name.
        @tn.unit(mapping=[2])
        def _():
            pass


# ---------------------------------------------------------------------------
# 3. Duplicate axis names (Rule R2)
# ---------------------------------------------------------------------------


def test_duplicate_axis_name_rejected():
    with pytest.raises(ValueError, match="duplicate axis name"):
        @tn.target("toy_r2_dup")
        @tn.unit(mapping=[4])
        def bank():
            @tn.unit(mapping=[8])
            def bank():    # noqa: F811 — same axis name as parent
                tn.op("nop", latency=1)


# ---------------------------------------------------------------------------
# 4. SIMD levels reject allo.stream (Rule R4)
# ---------------------------------------------------------------------------


def test_simd_level_rejects_stream():
    with pytest.raises(TypeError, match="Rule R4"):
        @tn.target("toy_r4_simd_stream")
        @tn.unit(mapping=[2])
        def channel():
            @tn.unit(mapping=[4], mode="simd")
            def lane():
                tn.stream("neighbor_link", T="uint32", N=16)


def test_simd_lockstep_rejects_stream():
    with pytest.raises(TypeError, match="Rule R4"):
        @tn.target("toy_r4_lockstep_stream")
        @tn.unit(mapping=[2])
        def channel():
            @tn.unit(mapping=[4], mode="simd-lockstep")
            def lane():
                tn.stream("x", T="uint32", N=8)


def test_mimd_level_accepts_stream():
    """Sanity: streams ARE legal in default mimd levels."""

    @tn.target("toy_r4_mimd_stream")
    @tn.unit(mapping=[2])
    def rank():
        @tn.unit(mapping=[8], mode="mimd")
        def dpu():
            tn.stream("neighbor_link", T="uint32", N=16)
            tn.memory("mram", bytes=64 << 20)
            tn.op("fmac", latency=40)

    # The stream survives to the tn_root as metadata.
    dpu_level = rank.tn_root.children[0]
    assert dpu_level.streams == [
        {"name": "neighbor_link", "T": "uint32", "N": 16}]


# ---------------------------------------------------------------------------
# 5. Round-trip equivalence: decorator surface == flat Grid builder
# ---------------------------------------------------------------------------


def _build_via_grid():
    """Build the same toy target the decorator surface builds in
    ``_build_basic_three_level``, but using the flat ``Grid`` /
    ``Leaf`` / ``build_from_grid`` API directly."""
    leaf = Leaf(
        memory=[Memory("grf", capacity_bytes=8 * 2, lanes=16, scope="grf")],
        ops=[Op("pim.add", lanes=16, latency=7, cycles_per_elem=0.1,
                energy_pJ=0.2)],
    )
    inner = Grid(16, "inner", leaf)
    middle = Grid(4, "middle", inner,
                  memory=[Memory("dram", capacity_bytes=4 << 20, scope="bank")])
    outer = Grid(8, "outer", middle)
    return build_from_grid("toy_three_level", outer)


def _target_fingerprint(t: Target) -> dict:
    """Reduce a Target to a fingerprint suitable for equality checks."""
    return {
        "name": t.name,
        "parallel_units": t.parallel_units,
        "memories": {
            name: {
                "capacity_bytes": m.capacity_bytes,
                "lanes": m.lanes,
                "dtype": m.dtype,
                "scope": m.scope,
                "parallel_units": m.parallel_units,
            }
            for name, m in t.memories.items()
        },
        "ops": {
            name: {
                "lanes": o.lanes,
                "latency": o.latency,
                "throughput": o.throughput,
                "cycles_per_elem": o.cycles_per_elem,
                "energy_pJ": o.energy_pJ,
            }
            for name, o in t.ops.items()
        },
    }


def test_roundtrip_equivalence_with_grid_builder():
    """The decorator-built target matches the flat-form-built target
    field-for-field. This is the proof that backends need no changes."""
    t_dec = _build_basic_three_level()
    t_flat = _build_via_grid()
    assert _target_fingerprint(t_dec) == _target_fingerprint(t_flat)


# ---------------------------------------------------------------------------
# 6. Extras: memories carry free attributes (for `bits=1` on GSI etc.)
# ---------------------------------------------------------------------------


def test_memory_free_kwarg_bits():
    """GSI's ``allo.memory("vr_bit", bits=1)`` must not crash: capacity
    falls to ceil(bits/8) bytes and the extra kwarg is preserved."""

    @tn.target("toy_gsi_like")
    @tn.unit(mapping=[2])
    def apuc():
        @tn.unit(mapping=[4], mode="simd")
        def element():
            tn.memory("vr_bit", bits=1)
            tn.op("cpy_imm", latency=1)

    mem = apuc.memories["vr_bit"]
    assert mem.capacity_bytes == 1   # ceil(1/8)


# ---------------------------------------------------------------------------
# 7. Host root wiring via kwargs on @allo.target
# ---------------------------------------------------------------------------


def test_host_root_memories_and_ops():
    """host_memories / host_ops kwargs on @allo.target reach the built
    ``Target``, matching the flat-form ``build_from_grid`` contract."""

    @tn.target("toy_host_root",
               host_memories=[Memory("host_dram", capacity_bytes=1 << 30,
                                     scope="host")],
               host_ops=[Op("host.softmax", lanes=1, latency=500,
                            energy_pJ=20.0,
                            emit="host.call softmax shape={shape}")],
               caps={"has_exp": False, "has_host_fallback": True})
    @tn.unit(mapping=[8])
    def channel():
        @tn.unit(mapping=[4], mode="simd")
        def lane():
            tn.memory("grf", elems=8, lanes=16, scope="grf")
            tn.op("pim.add", lanes=16, latency=7)

    assert "host_dram" in channel.memories
    assert "host.softmax" in channel.ops
    assert channel.caps.get("has_host_fallback") is True
    assert channel.caps.get("has_exp") is False


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"]))
