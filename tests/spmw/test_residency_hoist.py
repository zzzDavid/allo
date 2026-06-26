# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 013 (re-scoped 006, Phase 3) -- StageRequest surface + residency
hoist legality. ADDITIVE: deletes NOTHING, moves NO number.

Asserts:
  * the workload-facing `allo.broadcast`/`scatter`/`gather`/`reduce`(+derived)
    module functions record a `StageRequest` on the open `staging_scope`;
  * `residency` is validated to {resident, per_call, readback};
  * the Exo-`@config`-style hoist legality: a `residency="resident"` stage of
    a read-only buffer is marked `hoisted=True`; a `resident` stage of a
    KERNEL-WRITTEN buffer is a hard compile error;
  * the resolved hoist reproduces today's `weight_resident=True` effect -- the
    batched-GEMV weight (`local_W`, read-only) staged `resident` yields
    `weight_resident_from_staging(...) is True`, matching the
    `_with_weight_residency(..., True)` placement the existing enumerator
    emits;
  * the existing `weight_resident` branch / `_with_weight_residency` carriers
    are UNTOUCHED (task 014 deletes them) -- parity guard.
"""
from __future__ import annotations

import pytest

import allo
from allo.spmw_host import NotSupported, StageRequest
from allo.spmw_autoschedule import (
    _bucket_for_autoschedule,
    _samsung_enumerate,
    _with_stage_resident,
)
from allo.spmw_match import MatchTrace

from _fixtures import build_samsung_target
from test_samsung_batched_gemv import batched_gemv_top, BATCH


# --------------------------------------------------------------------- #
# StageRequest surface
# --------------------------------------------------------------------- #


def test_module_functions_record_stage_requests():
    with allo.staging_scope() as reqs:
        allo.scatter("local_W", over="PCH", residency="resident")
        allo.broadcast("local_x", over="PCH", residency="per_call")
        allo.gather("local_y", over="PCH", residency="readback")
    assert [r.collective for r in reqs] == ["scatter", "broadcast", "gather"]
    assert [r.buf for r in reqs] == ["local_W", "local_x", "local_y"]
    assert [r.residency for r in reqs] == ["resident", "per_call", "readback"]
    assert all(isinstance(r, StageRequest) for r in reqs)
    assert all(r.over == "PCH" for r in reqs)
    # not yet resolved -> not hoisted
    assert all(not r.hoisted for r in reqs)


def test_derived_collectives_record_and_carry_op():
    with allo.staging_scope() as reqs:
        allo.all_gather("h", over="X")
        allo.all_reduce("g", over="X", op=allo.host_cpu)
        allo.reduce_scatter("p", over="X", op=allo.host_cpu)
    assert [r.collective for r in reqs] == [
        "all_gather",
        "all_reduce",
        "reduce_scatter",
    ]
    assert reqs[1].op is allo.host_cpu
    assert reqs[2].op is allo.host_cpu


def test_calls_outside_scope_do_not_error():
    # Recording is a no-op when no scope is open (the call still returns the
    # request for inspection) -- so a bare collective call never crashes.
    req = allo.broadcast("buf", over="X", residency="per_call")
    assert isinstance(req, StageRequest)


def test_residency_validated():
    with allo.staging_scope():
        with pytest.raises(ValueError):
            allo.scatter("local_W", over="PCH", residency="forever")


# --------------------------------------------------------------------- #
# Hoist legality (Exo @config idempotency)
# --------------------------------------------------------------------- #


def test_resident_readonly_is_hoisted():
    with allo.staging_scope() as reqs:
        allo.scatter("local_W", over="PCH", residency="resident")
    allo.resolve_staging(reqs, written_buffers={"local_Y"})  # W read-only
    assert reqs[0].hoisted is True


def test_resident_kernel_written_is_compile_error():
    with allo.staging_scope() as reqs:
        allo.scatter("local_Y", over="PCH", residency="resident")  # output!
    with pytest.raises(NotSupported) as exc:
        allo.resolve_staging(reqs, written_buffers={"local_Y"})
    msg = str(exc.value)
    assert "local_Y" in msg
    assert "kernel-written" in msg


def test_per_call_written_is_legal():
    # per_call staging of a written buffer is fine (not hoisted, paid each call)
    with allo.staging_scope() as reqs:
        allo.gather("local_Y", over="PCH", residency="readback")
    allo.resolve_staging(reqs, written_buffers={"local_Y"})
    assert reqs[0].hoisted is False


# --------------------------------------------------------------------- #
# Bridge: hoist reproduces today's weight_resident=True effect
# --------------------------------------------------------------------- #


def _written_buffers_from_trace(trace: MatchTrace) -> set[str]:
    """Kernel-written buffers = the result memref of every match (the buffer
    the reduction stores into). Derived spmw-side, no MLIR rescan."""
    return {
        m.result_memref_name
        for m in trace.matches
        if m.result_memref_name is not None
    }


def test_resident_weight_reproduces_weight_resident_true():
    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)

    written = _written_buffers_from_trace(trace)
    # The batched-GEMV weight is read-only; the output is written.
    assert "local_W" not in written

    with allo.staging_scope() as reqs:
        allo.scatter("local_W", over="pseudo_channel", residency="resident")
    allo.resolve_staging(reqs, written_buffers=written)

    # The hoisted resident weight stage IS the weight_resident=True effect.
    assert allo.weight_resident_from_staging(reqs) is True

    # ... and it matches the placement flag the existing enumerator emits.
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    base = _samsung_enumerate(target, matches)[0]
    resident_placement = _with_stage_resident(base, True)
    assert resident_placement.extra["stage_resident"] == (
        allo.weight_resident_from_staging(reqs)
    )


def test_no_resident_stage_means_not_weight_resident():
    with allo.staging_scope() as reqs:
        allo.broadcast("local_x", over="PCH", residency="per_call")
        allo.gather("local_y", over="PCH", residency="readback")
    allo.resolve_staging(reqs, written_buffers={"local_y"})
    assert allo.weight_resident_from_staging(reqs) is False


# --------------------------------------------------------------------- #
# Enumerator emits both staging variants (task-017: stage_resident flag)
# --------------------------------------------------------------------- #


def test_enumerator_emits_both_stage_resident_variants():
    """Post-017 the enumerator still emits BOTH stage-resident variants
    (renamed off `weight_resident` to `stage_resident`, bridge option (b));
    the `host_staging` compose now keys on the flag (the old structural
    `weight_resident ? :` kernel_cycles branch is deleted)."""
    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    cands = _samsung_enumerate(target, matches)
    flags = [c.extra.get("stage_resident") for c in cands]
    assert True in flags and False in flags  # both variants still emitted
    res = [c for c in cands if c.extra.get("stage_resident")]
    assert all(c.mode.endswith("+wresident") for c in res)
