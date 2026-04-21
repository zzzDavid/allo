"""Round-trip test for the phase-2 ``pim_target`` MLIR dialect.

For each of the five built-in targets (Samsung HBM-PIM, SK-Hynix AiM,
UPMEM DPU, GSI APU v1, GSI APU v2), we:

1. Build the ``allo.pim.Target`` via ``build_<backend>()``.
2. Emit phase-2 MLIR text via ``allo.pim.mlir_emit.target_to_mlir``.
3. Assert the text contains at least one ``pim_target.unit`` op with
   ``mode = "simd"`` (proves phase-2 ops are produced, not phase-1).
4. Run the ``pim-opt`` binary on the text to canonicalize floats /
   attribute ordering, then run ``pim-opt`` again on the canonical
   form and assert byte-identity.

The test *skips loudly* (not silently passes) when ``pim-opt`` has
not been built. Building it is documented in
``experiments/E6_mlir_dialect_phase1/TASK.md`` and
``experiments/E6b_mlir_dialect_phase2/README.md``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable

import pytest

# The conftest.py alongside this file makes ``allo.pim`` importable even
# when the ``allo._mlir`` C-extension has not been built. Importing from
# the top-level ``allo.pim`` namespace is therefore always safe.
from allo.pim.backends import (
    build_samsung,
    build_aim,
    build_upmem,
    build_apu_v1,
    build_apu_v2,
)
from allo.pim.mlir_emit import target_to_mlir


# ---------------------------------------------------------------------------
# pim-opt binary discovery
# ---------------------------------------------------------------------------


_CANDIDATE_PATHS = [
    # Default standalone build location documented in
    # experiments/E6_mlir_dialect_phase1/TASK.md.
    "experiments/E6_mlir_dialect_phase1/build/bin/pim-opt",
    # Secondary location if someone builds inside the phase-2 dir.
    "experiments/E6b_mlir_dialect_phase2/build/bin/pim-opt",
]


def _repo_root() -> str:
    """Walk up from this file until we find experiments/."""
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "experiments")):
            return cur
        cur = os.path.dirname(cur)
    return here


def _find_pim_opt() -> str:
    root = _repo_root()
    # Explicit override for CI or cross-checkout use.
    env = os.environ.get("PIM_OPT")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for rel in _CANDIDATE_PATHS:
        p = os.path.join(root, rel)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""


# ---------------------------------------------------------------------------
# Target fixtures
# ---------------------------------------------------------------------------


_TARGETS = [
    ("samsung", build_samsung),
    ("aim", build_aim),
    ("upmem", build_upmem),
    ("apu_v1", build_apu_v1),
    ("apu_v2", build_apu_v2),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,builder", _TARGETS,
                         ids=[n for n, _ in _TARGETS])
def test_emit_produces_phase2_ops(name: str,
                                  builder: Callable[[], "object"]) -> None:
    """Phase-2 smoke test: emitter writes at least one ``pim_target.unit``
    op that carries a ``mode`` attribute.

    Four of the five built-in targets (Samsung, AiM, APU v1, APU v2) have
    at least one ``mode = "simd"`` leaf level by design; UPMEM is
    legitimately all-MIMD (tasklets run independent control flow), so we
    only require a SIMD mode for the non-UPMEM targets. Every target
    must however carry at least one ``pim_target.unit`` op with a ``mode``
    attribute — phase 1 flat ops carry no such attribute, so this is a
    sufficient discriminator between phase 1 and phase 2 emission."""
    t = builder()
    text = target_to_mlir(t)
    assert "pim_target.unit" in text, (
        f"{name}: emission produced no pim_target.unit ops. "
        "Phase 2 deliverable requires structural unit ops.")
    # Every phase-2 unit carries a mode; phase-1 ops do not.
    assert "mode \"" in text or 'mode = "' in text, (
        f"{name}: no unit op carries a mode attribute. Phase 2 units "
        "must always print a mode, so this suggests emission regressed "
        "to the phase-1 flat surface.")
    if name != "upmem":
        assert 'mode "simd"' in text or 'mode = "simd"' in text, (
            f"{name}: emission carries no mode=\"simd\" unit. All built-in "
            "targets except UPMEM have at least one SIMD leaf level by "
            "design.")


@pytest.mark.parametrize("name,builder", _TARGETS,
                         ids=[n for n, _ in _TARGETS])
def test_roundtrip_via_pim_opt(name: str,
                               builder: Callable[[], "object"],
                               tmp_path) -> None:
    """Phase-2 round-trip: emit, canonicalize once, then re-run
    ``pim-opt`` and assert byte-identity.

    First ``pim-opt`` invocation is a canonicalization pass (floats get
    normalized to ``%e`` form, attributes are sorted deterministically).
    A second invocation must then round-trip byte-identically — that is
    the guarantee phase 2 actually buys us.
    """
    pim_opt = _find_pim_opt()
    if not pim_opt:
        pytest.skip(
            "pim-opt binary not found. Build it with:\n"
            "  cd experiments/E6_mlir_dialect_phase1/build && ninja pim-opt\n"
            "or point to it via the PIM_OPT env var.")

    t = builder()
    text = target_to_mlir(t)

    src = tmp_path / f"{name}_emit.mlir"
    src.write_text(text)

    # First pass — canonicalize.
    canon = tmp_path / f"{name}_canon.mlir"
    proc1 = subprocess.run(
        [pim_opt, str(src)], capture_output=True, text=True, check=False)
    assert proc1.returncode == 0, (
        f"{name}: first pim-opt invocation failed.\n"
        f"stderr:\n{proc1.stderr}\nstdout:\n{proc1.stdout[:400]}")
    canon.write_text(proc1.stdout)

    # Second pass — must be byte-identical to the first.
    proc2 = subprocess.run(
        [pim_opt, str(canon)], capture_output=True, text=True, check=False)
    assert proc2.returncode == 0, (
        f"{name}: second pim-opt invocation failed.\n"
        f"stderr:\n{proc2.stderr}\nstdout:\n{proc2.stdout[:400]}")
    assert proc2.stdout == canon.read_text(), (
        f"{name}: byte-identical round-trip failed.\n"
        f"canonical:\n{canon.read_text()[:600]}\n"
        f"second pass:\n{proc2.stdout[:600]}")
