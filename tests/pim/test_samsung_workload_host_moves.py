# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 004 static assertions: the 10 Samsung GEMM-family leaf workloads express
their host data movement VISIBLY via `allo.host_xfer.*` (spec
`backend-host-transfer-dispatch.md`), and the recorded moves resolve against the
concrete Samsung target through `BackendHandle.resolve` to the right declared
moves (SCATTER_BANKS / BCAST_GRF_A / GATHER_BANKS).

This is the coder-side static assertion (simulator-running cells are the
verifier's domain). It does NOT run PIMSimulator -- it imports each workload
module, inspects the exported `HOST_MOVES`, and resolves them against the target
tree by handle identity + verb object (never string ==).
"""

from __future__ import annotations

import importlib.util
import pathlib

import allo
import pytest

from lib.targets import build_target

_SAMSUNG = pathlib.Path(__file__).resolve().parent / "samsung_hbm_pim"

# verb name -> the declared Samsung Move name it must resolve to (D4 table).
_EXPECT_MOVE = {
    "scatter": "SCATTER_BANKS",
    "broadcast": "BCAST_GRF_A",
    "gather": "GATHER_BANKS",
}

# Per-workload: the EXACT recorded (verb, buffer_role) sequence. Encodes the
# per-cell overrides -- multi-stage chains omit the gather for a device-resident
# intermediate (atax/2mm/3mm).
_EXPECTED = {
    "gemm": [("scatter", "A"), ("broadcast", "B"), ("gather", "C")],
    "covariance": [("scatter", "cdata"), ("broadcast", "cdata"), ("gather", "cov_raw")],
    "doitgen": [("scatter", "A"), ("broadcast", "x"), ("gather", "out")],
    # chain: tmp = A@x stays device-resident (NO gather), y = A^T@tmp gathered.
    "atax": [("scatter", "A"), ("broadcast", "x"),
             ("scatter", "A"), ("broadcast", "tmp"), ("gather", "y")],
    # two independent GEMVs, both outputs gathered.
    "bicg": [("scatter", "A"), ("broadcast", "p"), ("gather", "q"),
             ("scatter", "A"), ("broadcast", "r"), ("gather", "s")],
    "mvt": [("scatter", "A"), ("broadcast", "y1"), ("gather", "x1"),
            ("scatter", "A"), ("broadcast", "y2"), ("gather", "x2")],
    "gesummv": [("scatter", "A"), ("broadcast", "x"), ("gather", "tmp"),
                ("scatter", "B"), ("broadcast", "x"), ("gather", "y")],
    # GEMM chain: AB device-resident (no gather), D gathered.
    "2mm": [("scatter", "A"), ("broadcast", "B"),
            ("scatter", "AB"), ("broadcast", "C"), ("gather", "D")],
    # GEMM chain: AB, CD device-resident (no gather), G gathered.
    "3mm": [("scatter", "A"), ("broadcast", "B"),
            ("scatter", "C"), ("broadcast", "D"),
            ("scatter", "AB"), ("broadcast", "CD"), ("gather", "G")],
    # mixed ELTWISE rank-1 + GEMV: A scattered once, vectors broadcast, w gathered.
    "gemver": [("scatter", "A"), ("broadcast", "v1"), ("broadcast", "v2"),
               ("broadcast", "y"), ("broadcast", "x"), ("gather", "w")],
    # Tier-2 (task 006): honest CYCLES-ONLY cells. syrk/gramschmidt are A^T@A
    # contractions (A staged as both weight + input); symm/trmm scatter A,
    # broadcast B; syr2k is two host-summed products.
    "syrk": [("scatter", "A"), ("broadcast", "A"), ("gather", "C")],
    "gramschmidt": [("scatter", "A"), ("broadcast", "A"), ("gather", "G")],
    "symm": [("scatter", "A"), ("broadcast", "B"), ("gather", "S")],
    "trmm": [("scatter", "A"), ("broadcast", "B"), ("gather", "Bout")],
    "syr2k": [("scatter", "A"), ("broadcast", "B"), ("gather", "P1"),
              ("scatter", "B"), ("broadcast", "A"), ("gather", "P2")],
}


def _load(kernel):
    leaf = _SAMSUNG / kernel / "workload.py"
    spec = importlib.util.spec_from_file_location(f"_wl_host_moves_{kernel}", str(leaf))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _buffer_role(arg):
    """The buffer role for a move arg, mirroring `_resolve_host_moves`: a bare
    string label, or a `BufferToken`/buffer object read via `.name`. Device
    endpoints (HandleToken / _VerbCallOrToken) return None and are filtered out.
    """
    if isinstance(arg, (allo.HandleToken, allo.BufferToken)):
        return arg.name if isinstance(arg, allo.BufferToken) else None
    if isinstance(arg, str):
        return arg
    # Any other non-device object: read .name if present (host-program tokens).
    name = getattr(arg, "name", None)
    return name


def _verb_buf(record):
    """(verb_name, buffer_role): the buffer arg is the lone non-handle arg.

    Accepts both the string-label form (`record_host_moves` workloads) and the
    `BufferToken` identity form (`host_program` workloads); the resolver treats
    them identically (`_resolve_host_moves` reads `.name`)."""
    handle_token = getattr(allo.host_xfer, "banks").__class__  # _VerbCallOrToken
    bufs = [
        _buffer_role(a)
        for a in record.args
        if not isinstance(a, (allo.HandleToken, handle_token))
    ]
    assert len(bufs) == 1, f"expected one buffer-name arg, got {record.args!r}"
    return record.verb.name, bufs[0]


@pytest.mark.parametrize("kernel", sorted(_EXPECTED))
def test_workload_records_expected_host_moves(kernel):
    """The leaf workload exports HOST_MOVES recording exactly the expected
    verb-tagged moves (visible, not inferred), with chain intermediates omitting
    the device-resident gather."""
    mod = _load(kernel)
    hm = getattr(mod, "HOST_MOVES", None)
    assert hm is not None, f"{kernel}: workload must export HOST_MOVES"
    got = [_verb_buf(r) for r in hm]
    assert got == _EXPECTED[kernel], f"{kernel}: {got!r} != {_EXPECTED[kernel]!r}"


@pytest.mark.parametrize("kernel", sorted(_EXPECTED))
def test_host_moves_resolve_against_samsung(kernel):
    """Each recorded move resolves through BackendHandle to the declared Samsung
    move named in the D4 table (resolution by handle identity + verb object)."""
    mod = _load(kernel)
    target = build_target("samsung_hbm_pim")
    bh = allo.BackendHandle(target)
    for record in mod.HOST_MOVES:
        move = bh.resolve(record)
        assert move.name == _EXPECT_MOVE[record.verb.name], (
            f"{kernel}: verb {record.verb.name!r} resolved to {move.name!r}, "
            f"expected {_EXPECT_MOVE[record.verb.name]!r}"
        )
        # The resolved move carries the verb OBJECT (identity, not string).
        assert move.verb is record.verb


def test_chain_intermediates_are_not_gathered():
    """The per-cell override: a device-resident intermediate is staged in
    (scatter+broadcast) but NEVER gathered (atax tmp; 2mm AB; 3mm AB, CD)."""
    for kernel, resident in (("atax", {"tmp"}), ("2mm", {"AB"}), ("3mm", {"AB", "CD"})):
        mod = _load(kernel)
        gathered = {b for (v, b) in (_verb_buf(r) for r in mod.HOST_MOVES) if v == "gather"}
        assert not (resident & gathered), (
            f"{kernel}: device-resident intermediate(s) {resident & gathered!r} "
            f"were gathered; they must stay on-device"
        )
