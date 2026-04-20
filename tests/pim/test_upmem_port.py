"""Equivalence test: the new ``@allo.unit``-nested UPMEM target description
must be a drop-in replacement for the previous flat ``Grid``/``Leaf`` form.

Procedure: vendor a verbatim copy of the old flat-form builder under a
distinct target name, build both Targets, and assert they agree field-for-
field on everything the backend runtime actually reads
(``name``, ``parallel_units``, ``caps``, ``memories`` (incl. each memory's
``capacity_bytes`` / ``scope`` / ``parallel_units``), ``ops`` (incl.
``lanes`` / ``latency`` / ``cycles_per_elem`` / ``energy_pJ`` / ``emit``),
and the host-fallback memory/op lists implied by the build).

The vendored builder mirrors ``upmem.py`` as it was BEFORE the
``@allo.unit`` port (verbatim numbers and emit strings); the two trees
differ only by the new top-level ``rank`` axis (``mapping=[1]``) introduced
by the @allo.unit form, which is a no-op for ``parallel_units``.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim.target import Memory, Op, Leaf, Grid, build_from_grid  # noqa: E402
from allo.pim.backends.upmem import build_upmem  # noqa: E402


# ---------------------------------------------------------------------------
# Vendored copy of the pre-port flat-form UPMEM builder.
# ---------------------------------------------------------------------------


def _build_upmem_flat_legacy():
    """Verbatim flat-form UPMEM target: Grid(2048,'dpu') >>
    Grid(24,'tasklet') >> Leaf(ops=...). Numbers / emit strings copied
    one-for-one from the pre-port ``upmem.py``."""
    tasklet = Leaf(ops=[
        Op("dpu.add",        lanes=1, latency=29102, cycles_per_elem=3.25195,
           energy_pJ=0.08, emit="for(i) c[i]=a[i]+b[i]; // elems={n_elems}"),
        Op("dpu.sub",        lanes=1, latency=29102, cycles_per_elem=3.25195,
           energy_pJ=0.08, emit="for(i) c[i]=a[i]-b[i]; // elems={n_elems}"),
        Op("dpu.mul",        lanes=1, latency=63025, cycles_per_elem=4.26869,
           energy_pJ=0.10, emit="for(i) c[i]=a[i]*b[i]; // elems={n_elems}"),
        Op("dpu.fmac",       lanes=1, latency=63025, cycles_per_elem=14.23,
           energy_pJ=0.35,
           emit="for(k) acc=fadd_hf(acc, fmul_hf(a[k], b[k])); // elems={n_elems}"),
        Op("dpu.relu",       lanes=1, latency=29102, cycles_per_elem=1.18,
           energy_pJ=0.04, emit="for(i) c[i]=a[i]>0?a[i]:0; // elems={n_elems}"),
        Op("dpu.exp",        lanes=1, latency=29102, cycles_per_elem=17.77,
           energy_pJ=0.60, emit="for(i) c[i]=exp_poly_hf(a[i]); // elems={n_elems}"),
        Op("dpu.reduce_max", lanes=1, latency=29102, cycles_per_elem=1.48,
           energy_pJ=0.05, emit="for(i) m=fmax_hf(m,a[i]); // elems={n_elems}"),
        Op("dpu.reduce_sum", lanes=1, latency=29102, cycles_per_elem=1.48,
           energy_pJ=0.05, emit="for(i) s=fadd_hf(s,a[i]); // elems={n_elems}"),
        Op("dpu.div",        lanes=1, latency=29102, cycles_per_elem=23.69,
           energy_pJ=0.80, emit="for(i) c[i]=fdiv_hf(a[i],s); // elems={n_elems}"),
    ])
    dpu = Grid(24, "tasklet", child=tasklet)
    root = Grid(2048, "dpu",
                memory=[Memory("mram", capacity_bytes=64 << 20, scope="bank"),
                        Memory("wram", capacity_bytes=64 << 10, scope="bank")],
                child=dpu)
    return build_from_grid(
        "upmem_dpu", root,
        caps=dict(has_mac=True, has_exp=True, has_div=True,
                  has_reduce_max=True, has_softmax=True),
    )


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _memory_fp(m: Memory) -> dict:
    return {
        "capacity_bytes": m.capacity_bytes,
        "lanes": m.lanes,
        "dtype": m.dtype,
        "scope": m.scope,
        "parallel_units": m.parallel_units,
    }


def _op_fp(o: Op) -> dict:
    return {
        "lanes": o.lanes,
        "latency": o.latency,
        "throughput": o.throughput,
        "cycles_per_elem": o.cycles_per_elem,
        "energy_pJ": o.energy_pJ,
        "emit": o.emit,
    }


def _target_fp(t) -> dict:
    return {
        "name": t.name,
        "parallel_units": t.parallel_units,
        "caps": dict(t.caps),
        "memories": {n: _memory_fp(m) for n, m in t.memories.items()},
        "ops": {n: _op_fp(o) for n, o in t.ops.items()},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_total_parallel_units_matches_legacy():
    """rank=1 * dpu=2048 * tasklet=24 == old 2048 * 24 == 49152."""
    new_t = build_upmem()
    old_t = _build_upmem_flat_legacy()
    assert new_t.parallel_units == 49152
    assert old_t.parallel_units == 49152
    assert new_t.parallel_units == old_t.parallel_units


def test_target_fingerprints_match():
    """Field-for-field equivalence on everything backends consume."""
    new_t = build_upmem()
    old_t = _build_upmem_flat_legacy()
    assert _target_fp(new_t) == _target_fp(old_t)


def test_op_set_unchanged():
    """No ops added or removed by the port."""
    new_t = build_upmem()
    old_t = _build_upmem_flat_legacy()
    assert sorted(new_t.ops.keys()) == sorted(old_t.ops.keys())


def test_memory_set_unchanged():
    """Both mram and wram still live at the dpu level (parallel_units=2048)."""
    new_t = build_upmem()
    assert "mram" in new_t.memories and "wram" in new_t.memories
    for name in ("mram", "wram"):
        assert new_t.memories[name].parallel_units == 2048, (
            f"{name} should sit at the dpu level (1 * 2048 = 2048 parallel "
            f"units), got {new_t.memories[name].parallel_units}")


def test_pattern_count_unchanged():
    """6 patterns: gemv, add, mul, relu, scale, softmax."""
    new_t = build_upmem()
    assert len(new_t.patterns) == 6
    assert {p.name for p in new_t.patterns} == {
        "gemv->dpu_fmac", "add->dpu_add", "mul->dpu_mul",
        "relu->dpu_relu", "scale->dpu_mul", "softmax->dpu_fused",
    }


def test_axis_metadata_present_on_new_form():
    """The @allo.unit port exposes axes and modes the flat builder did not."""
    new_t = build_upmem()
    assert getattr(new_t, "axes", None) == ["rank", "dpu", "tasklet"]
    modes = getattr(new_t, "modes", None)
    assert modes is not None
    assert modes["tasklet"] == "mimd", (
        "tasklet level must be MIMD: UPMEM tasklets run independent control "
        "flow synchronized only at explicit DMA / barrier boundaries.")


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"]))
