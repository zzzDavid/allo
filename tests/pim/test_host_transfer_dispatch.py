# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for the backend host-transfer dispatch surface (spec 001/002).

Covers the load-bearing rulings of design_doc/compiler/backend-host-transfer-dispatch.md:
  * verb sentinels are first-class objects; `Move.verb` defaults to `move_only`
    and `allo.move(verb=)` is additive (every existing move keeps `move_only`);
  * `BackendHandle.resolve` keys on (verb-object, device-handle-identity) using
    `is` -- NEVER a string `==`;
  * the recorded `HandleToken` round-trips to the *identical* tree handle object;
  * grf_a vs grf_b (same `broadcast` verb) disambiguate by handle identity;
  * an unimplemented verb (`host_xfer.reduce`) is a hard error naming verb+target;
  * `import allo` stays clean and `allo.backend` remains the codegen submodule;
  * all five `build_*_target()` still build; Samsung is `[hbm_pim, host]` with
    `work_grid() == ([16, 8], 128)`.

Pure-Python, no simulator -- runnable under the pim-dev env without a board.
"""

from __future__ import annotations

import pytest

import allo
from allo.pim import targets as T


# --------------------------------------------------------------------------- #
# surface invariants
# --------------------------------------------------------------------------- #


def test_verbs_are_distinct_first_class_objects():
    verbs = [allo.broadcast, allo.scatter, allo.gather, allo.move_only]
    assert len({id(v) for v in verbs}) == 4
    assert [v.name for v in verbs] == ["broadcast", "scatter", "gather", "move_only"]


def test_move_verb_defaults_to_move_only():
    # A device move that omits verb= must stay move_only (additive guarantee).
    sam = T.build_samsung_target()
    ld_a = sam.move("LD_A")
    assert ld_a.verb is allo.move_only


def test_import_allo_clean_and_backend_submodule_intact():
    # allo.backend must remain the codegen-backend SUBMODULE, not the proxy.
    import types

    assert isinstance(allo.backend, types.ModuleType)
    assert hasattr(allo.backend, "llvm")
    # The proxy lives under host_xfer.
    assert isinstance(allo.host_xfer, allo.spmw_target.RecordingBackend)  # type: ignore[attr-defined]


def test_all_five_targets_build():
    for nm in ("samsung", "aim", "upmem", "apu_v1", "apu_v2"):
        t = getattr(T, f"build_{nm}_target")()
        assert t.name


def test_samsung_shape_unchanged():
    sam = T.build_samsung_target()
    assert [u.name for u in sam.root.children] == ["hbm_pim", "host"]
    assert sam.work_grid() == ([16, 8], 128)


def test_samsung_host_moves_carry_expected_verbs():
    sam = T.build_samsung_target()
    expect = {
        "SCATTER_BANKS": allo.scatter,
        "BCAST_GRF_A": allo.broadcast,
        "BCAST_GRF_B": allo.broadcast,
        "BCAST_SRF": allo.broadcast,
        "GATHER_BANKS": allo.gather,
        "PROGRAM_CRF": allo.move_only,
    }
    for name, verb in expect.items():
        assert sam.move(name).verb is verb, name


# --------------------------------------------------------------------------- #
# dispatch by handle identity + verb object (no string ==)
# --------------------------------------------------------------------------- #


def _record_triple(weight, vec, out):
    with allo.record_host_moves() as hm:
        allo.host_xfer.scatter(weight, allo.host_xfer.banks)
        allo.host_xfer.broadcast(vec, allo.host_xfer.grf_a)
        allo.host_xfer.gather(out, allo.host_xfer.banks)
    return hm


def test_resolution_picks_the_right_samsung_moves():
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    W, x, y = object(), object(), object()
    hm = _record_triple(W, x, y)
    r_scatter, r_bcast, r_gather = (bh.resolve(r) for r in hm.records)
    assert r_scatter.name == "SCATTER_BANKS"
    assert r_bcast.name == "BCAST_GRF_A"
    assert r_gather.name == "GATHER_BANKS"


def test_token_round_trips_to_identical_tree_handle():
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    hm = _record_triple(object(), object(), object())
    r_scatter, r_bcast, r_gather = (bh.resolve(r) for r in hm.records)
    # The resolved move's device endpoint is the SAME object the tree holds.
    assert r_scatter.dst is bh.banks
    assert r_bcast.dst is bh.grf_a
    assert r_gather.src is bh.banks
    # And `bh.banks` is the very Memory the target tree exposes.
    assert bh.banks is sam.banks


def test_verb_compared_by_identity_not_string():
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    hm = _record_triple(object(), object(), object())
    # The recorded verb is the sentinel object itself.
    assert hm.records[0].verb is allo.scatter
    assert bh.resolve(hm.records[0]).verb is allo.scatter
    # A look-alike _Verb with the same name must NOT resolve (proves no ==).
    from allo.spmw_target import _Verb, HostMoveRecord

    fake = HostMoveRecord(_Verb("scatter"), (object(), allo.host_xfer.banks))
    with pytest.raises(ValueError):
        bh.resolve(fake)


def test_grf_a_grf_b_disambiguate_by_handle_identity():
    # Both are (broadcast, into-device) but to different handles -> no collision.
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    with allo.record_host_moves() as hm:
        allo.host_xfer.broadcast(object(), allo.host_xfer.grf_a)
        allo.host_xfer.broadcast(object(), allo.host_xfer.grf_b)
    assert bh.resolve(hm.records[0]).name == "BCAST_GRF_A"
    assert bh.resolve(hm.records[1]).name == "BCAST_GRF_B"


def test_program_crf_resolves_via_move_only():
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    with allo.record_host_moves() as hm:
        allo.host_xfer.move(object(), allo.host_xfer.crf)
    r = bh.resolve(hm.records[0])
    assert r.name == "PROGRAM_CRF"
    assert r.verb is allo.move_only


# --------------------------------------------------------------------------- #
# coverage check: unimplemented verb -> hard error naming verb + target
# --------------------------------------------------------------------------- #


def test_unimplemented_verb_is_hard_error_naming_verb_and_target():
    sam = T.build_samsung_target()
    bh = allo.BackendHandle(sam)
    with allo.record_host_moves() as hm:
        allo.host_xfer.reduce(object(), allo.host_xfer.banks)
    with pytest.raises(ValueError) as ei:
        bh.resolve(hm.records[0])
    msg = str(ei.value)
    assert "reduce" in msg
    assert "samsung_hbm_pim" in msg


def test_record_outside_scope_errors():
    with pytest.raises(RuntimeError):
        allo.host_xfer.scatter(object(), allo.host_xfer.banks)
