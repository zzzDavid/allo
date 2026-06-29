# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 D2: the allocator is non-inert and capacity-true.

Four deliverables, each test-guarded:
  1. Non-zero locality-aware base access cost (register < near_bank <
     scratchpad < dram), read from the bound CostModel, NOT a paste.
  2. Real `_estimate_bytes` from shape + dtype so the `bytes_cap` arms gate.
  3. Tree-derived capacities (Register.slots / Memory geometry), retiring the
     `target.name`-keyed table; AiM gpr reconciled to the fixture's
     structural `slots=31`.
  4. Forced-spill-under-pressure: a fit-fail falls to a real spill (D1), not a
     hard RuntimeError dropping the candidate.

The no-spill corpus stays byte-identical (the seed handle is kept when it
fits) -- that anchor lives in `test_regalloc.py`; here we prove the new
behaviour is actually exercised.
"""

from __future__ import annotations

import pytest

import allo
from allo.spmw_autoschedule import Placement
from allo.spmw_regalloc import (
    Spilled,
    LiveRange,
    allocate,
    build_cost_vector,
    _base_access_cost,
    _build_capacity,
    _classify_tier,
    _estimate_bytes,
    _handle_key,
    _locality_unit,
    _memory_capacity_bytes,
)
from allo.spmw_match import MatchedOp, OperandBinding

from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_apu_v2_target,
    build_samsung_target,
    build_upmem_target,
)


# --------------------------------------------------------------------- #
# D2.1 -- locality-aware, tier-monotone base access cost
# --------------------------------------------------------------------- #


def test_base_access_cost_is_tier_monotone():
    """register < near_bank < scratchpad < dram, all non-zero, scaled by the
    bound CostModel's ROW_BUFFER_MISS unit (not a paste)."""
    target = build_upmem_target()
    lr = LiveRange("v", "x", 0, 0, False)
    unit = _locality_unit(target)
    assert unit == 33  # UPMEM ROW_BUFFER_MISS, read from the cost model

    reg = _base_access_cost(target, lr, target.gprs)
    nb = target.mram  # dram tier on UPMEM
    dram = _base_access_cost(target, lr, target.mram[0])
    scratch = _base_access_cost(target, lr, target.wram[0])

    assert 0 < reg < scratch < dram, (reg, scratch, dram)
    # Tier ranks scale by the cost-model unit (monotone, non-zero).
    assert reg == 1 * unit


def test_register_candidates_same_cost_cancel():
    """SPEC-022 D2 byte-for-byte default: every register-tier conflict-free
    candidate gets the SAME non-zero base cost (so it cancels in argmin)."""
    target = build_samsung_target()
    lr = LiveRange("v", "x", 0, 0, False)
    a = _base_access_cost(target, lr, target.grf_a)
    b = _base_access_cost(target, lr, target.grf_b)
    assert a == b > 0


def test_tier_classification():
    target = build_aim_target()
    assert _classify_tier(target.gpr) == "register"
    assert _classify_tier(target.banks[0]) == "near_bank"
    assert _classify_tier(target.gb[0]) == "scratchpad"
    assert _classify_tier(build_upmem_target().mram[0]) == "dram"


# --------------------------------------------------------------------- #
# D2.2 -- real _estimate_bytes gates the bytes_cap arms
# --------------------------------------------------------------------- #


def test_estimate_bytes_none_without_dtype():
    """No dtype in the match (today's corpus) -> None -> slot-counted
    fallback, so the corpus is byte-identical."""
    m = MatchedOp(
        target_op_name="MUL", func_name="f", work_id=(0,),
        enclosing_loops=[("%i", "0", "1024", 1)],
        operands=[OperandBinding(role="x", memref_name="a")],
        result_memref_name="a", op_range=("%a", "%b"),
    )
    assert _estimate_bytes(m, m.operands[0]) is None


def test_estimate_bytes_real_with_dtype():
    """With a dtype carried on `match.extra`, the footprint is
    product(shape) * dtype_bytes."""
    m = MatchedOp(
        target_op_name="MUL", func_name="f", work_id=(0,),
        enclosing_loops=[("%i", "0", "256", 1), ("%j", "0", "4", 1)],
        operands=[OperandBinding(role="x", memref_name="a")],
        result_memref_name="a", op_range=("%a", "%b"),
        extra={"dtype_bits": 16},
    )
    # 256 * 4 elems * 2 bytes (fp16) = 2048 bytes.
    assert _estimate_bytes(m, m.operands[0]) == 256 * 4 * 2


def test_bytes_cap_gates_when_footprint_exceeds_scratchpad():
    """A live value whose byte footprint exceeds the AiM `gb` scratchpad cap
    (1024 bytes) is forced to spill -- the bytes_cap arm now gates instead of
    collapsing to 'always fits'."""
    target = build_aim_target()
    # gb holds 1024 bytes; a 4096-elem fp16 value = 8192 bytes does not fit.
    m = MatchedOp(
        target_op_name="MUL", func_name="big", work_id=(0,),
        enclosing_loops=[("%i", "0", "4096", 1)],
        operands=[OperandBinding(role="x", memref_name="big", is_loop_carried=True)],
        result_memref_name="big", op_range=("%a", "%b"),
        extra={"dtype_bits": 16},
    )
    # Pin `big` onto gb (the bounded scratchpad). With a real byte estimate
    # exceeding 1024, the allocator must spill it rather than claim it fits.
    cand = Placement(placements={"big": target.gb[0]})
    result = allocate(target, [m], cand, all_candidates=[cand])
    assert any(
        isinstance(h, Spilled) for h in result.placement.placements.values()
    ), result.placement.placements


# --------------------------------------------------------------------- #
# D2.3 -- tree-derived capacities; AiM gpr reconciliation
# --------------------------------------------------------------------- #


def test_capacities_tree_derived_reproduce_numbers():
    """Tree-derived slot/byte caps reproduce the per-backend numbers the old
    target.name table pasted -- now from Register.slots / Memory geometry."""
    samsung = build_samsung_target()
    cap = _build_capacity(samsung)
    assert cap.slots[_handle_key(samsung.grf_a)] == 8  # = lanes
    assert cap.slots[_handle_key(samsung.grf_b)] == 8

    upmem = build_upmem_target()
    cap = _build_capacity(upmem)
    assert cap.slots[_handle_key(upmem.gprs)] == 24  # = lanes
    assert cap.bytes_cap[_handle_key(upmem.wram)] == 65536  # = size_bytes

    apu1 = build_apu_v1_target()
    cap = _build_capacity(apu1)
    assert cap.slots[_handle_key(apu1.vrs)] == 16
    assert cap.bytes_cap[_handle_key(apu1.l1)] == 32768


def test_aim_gpr_slot_count_is_31_from_fixture_geometry():
    """AiM reconciliation: the SLOT count is the fixture's structural
    `slots=31` (addressable GPRs), NOT the 16-lane SIMD width, and NOT a
    pasted constant in the allocator."""
    target = build_aim_target()
    assert target.gpr.lanes == 16          # SIMD width axis
    assert target.gpr.slots == 31          # addressable-depth axis (datasheet)
    cap = _build_capacity(target)
    assert cap.slots[_handle_key(target.gpr)] == 31
    # gb scratchpad byte cap derived from entries*width/8 = 512*16/8.
    assert cap.bytes_cap[_handle_key(target.gb)] == 1024
    assert _memory_capacity_bytes(target.gb) == 1024


def test_new_backend_gets_structural_default_not_error():
    """A target with no special-cased builder gets a structural default
    (no NotImplementedError). APU v2 has no spill builder miss, so use a
    bespoke minimal target to prove the structural path."""
    @allo.target("noname_backend")
    def device():
        @allo.unit(mapping=[1])
        def pe():
            allo.reg(4, 32, name="r0")
            allo.mem(size_bytes=2048, name="buf")

    target = device
    cap = _build_capacity(target)  # must not raise
    assert cap.slots[_handle_key(target.r0)] == 4
    assert cap.bytes_cap[_handle_key(target.buf)] == 2048


# --------------------------------------------------------------------- #
# D2.4 -- forced-spill-under-pressure (no hard RuntimeError drop)
# --------------------------------------------------------------------- #


def test_forced_spill_under_pressure_does_not_raise():
    """When no unspilled tier has room AND the backend can spill, the
    allocator falls to a forced spill instead of raising RuntimeError (which
    would drop the candidate). Spills are real after D1."""
    target = build_samsung_target()
    # 9 overlapping live ranges, all pinned to grf_a (8 slots). The 9th
    # cannot fit any register -> forced spill, not a raise.
    operands = [
        OperandBinding(role=f"a{i}", memref_name=f"v{i}", is_loop_carried=True)
        for i in range(9)
    ]
    m = MatchedOp(
        target_op_name="MUL", func_name="pressure", work_id=(0,),
        enclosing_loops=[("%i", "0", "1024", 1)],
        operands=operands, result_memref_name="v0", op_range=("%a", "%b"),
    )
    cand = Placement(placements={f"v{i}": target.grf_a for i in range(9)})
    # Must NOT raise; exactly one forced spill to bank_row.
    result = allocate(target, [m], cand, all_candidates=[cand])
    spills = [h for h in result.placement.placements.values()
              if isinstance(h, Spilled)]
    assert len(spills) == 1, result.placement.placements
    assert spills[0].tier == "bank_row"


if __name__ == "__main__":
    test_base_access_cost_is_tier_monotone()
    test_register_candidates_same_cost_cancel()
    test_tier_classification()
    test_estimate_bytes_none_without_dtype()
    test_estimate_bytes_real_with_dtype()
    test_bytes_cap_gates_when_footprint_exceeds_scratchpad()
    test_capacities_tree_derived_reproduce_numbers()
    test_aim_gpr_slot_count_is_31_from_fixture_geometry()
    test_new_backend_gets_structural_default_not_error()
    test_forced_spill_under_pressure_does_not_raise()
    print("ALL PASSED")
