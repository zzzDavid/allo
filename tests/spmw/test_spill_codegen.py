# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 D1: spill realization -- the move scheduler emits the tier
LD/ST round-trip around a spilled value's work-id window.

These are no-sim unit/compile tests: they assert that a `Spilled(home,
tier)` placement lowers to the correct tier load (pre) + store (post)
moves, in addition to the home register's own moves, and that a backend
whose tier cannot spill is a HARD compile error rather than a silent
register-resident emission.

`test_upmem_spill_runs_with_verified_numerics` is the real-sim anchor
(SPEC-022 D1, task 004b): a capacity-exceeding GEMV whose accumulator is
spilled to `mram` compiles through the spill path to a real artifact, runs
on uPIMulator, and the GEMV host verifies `c == W@x` byte-for-byte. The
spill round-trip (`mram_write`/`mram_read` of the accumulator tile) is in
the artifact the simulator consumed AND the numeric check passes -- the
spilled artifact, not a fixed template, is what executes.
"""

from __future__ import annotations

import pytest

import allo
from allo.spmw_codegen import (
    compile_for_target,
    _upim_root,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_regalloc import Spilled

from _fixtures import (
    build_apu_v2_target,
    build_upmem_target,
)


def _upim_gemv_slot_present() -> bool:
    root = _upim_root()
    return (root / "build" / "uPIMulator").exists() and (
        root / "benchmark" / "GEMV" / "dpu"
    ).exists()


def _upmem_mac_trace() -> MatchTrace:
    return MatchTrace(
        target_name="upmem",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


def test_upmem_spill_emits_mram_round_trip():
    """A `Spilled(home=gprs, tier="mram")` on `acc` lowers to an MRAM
    load before the MAC and an MRAM store after -- the round-trip is no
    longer dropped."""
    target = build_upmem_target()
    trace = _upmem_mac_trace()
    layout = allo.Placement(
        placements={
            "local_W": target.wram[0],
            "local_x": target.wram[1],
            "acc": Spilled(home_handle=target.gprs, tier="mram"),
        },
    )
    layout._spilled = ["acc"]

    compiled = compile_for_target(target, trace, layout=layout)
    joined = "\n".join(compiled.cmds)

    # The MRAM LD/ST round-trip is present (LD_MRAM -> mram_read,
    # ST_MRAM -> mram_write).
    assert "mram_read(" in joined, joined
    assert "mram_write(" in joined, joined

    # The compute body (acc += ...) is unwrapped to the home gprs register,
    # so the MAC still uses the register operand, not the Spilled wrapper.
    assert "gprs" in joined, joined

    # Ordering: the MRAM load (window open) precedes the MAC compute, which
    # precedes the MRAM store (window close).
    ld_pos = next(i for i, c in enumerate(compiled.cmds) if "mram_read(" in c)
    st_pos = next(i for i, c in enumerate(compiled.cmds) if "mram_write(" in c)
    mac_pos = next(i for i, c in enumerate(compiled.cmds) if "+=" in c)
    assert ld_pos < mac_pos < st_pos, compiled.cmds


def test_no_spill_emits_no_round_trip():
    """The byte-identity default: an empty `_spilled` placement emits zero
    spill moves -- the no-spill corpus is unchanged."""
    target = build_upmem_target()
    trace = _upmem_mac_trace()
    layout = allo.Placement(
        placements={
            "local_W": target.wram[0],
            "local_x": target.wram[1],
            "acc": target.gprs,
        },
    )
    # _spilled defaults to [] -- no spill audit.
    assert layout._spilled == []

    compiled = compile_for_target(target, trace, layout=layout)
    joined = "\n".join(compiled.cmds)

    # No MRAM spill round-trip around the body (the gprs home is a register;
    # resolve_moves returns no LD/ST, and there is no spill).
    assert "mram_read(" not in joined, joined
    assert "mram_write(" not in joined, joined


def test_apu_v2_l2_spill_is_hard_error():
    """APU v2 has no validated L2 spill path: resolve_spill_moves raises,
    which propagates as a hard compile error rather than a silent
    register-resident emission."""
    target = build_apu_v2_target()
    trace = MatchTrace(
        target_name="apu_v2",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "4096", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )
    layout = allo.Placement(
        placements={
            "local_W": target.l1,
            "local_x": target.l1,
            "acc": Spilled(home_handle=target.l1, tier="l2"),
        },
    )
    layout._spilled = ["acc"]

    with pytest.raises(NotImplementedError, match="cannot spill"):
        compile_for_target(target, trace, layout=layout)


def _gemv_spill_trace() -> MatchTrace:
    """GEMV-shaped trace (outer M-loop + inner K-loop) so `_run_upmem`
    routes to the bespoke GEMV host, which verifies `c == W@x`
    byte-for-byte."""
    return MatchTrace(
        target_name="upmem",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%m", "0", "16", "1"),
                    ("%k", "0", "256", "1"),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="W"),
                    OperandBinding(role="y", memref_name="x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


@pytest.mark.skipif(
    not _upim_gemv_slot_present(),
    reason="uPIMulator binary or GEMV benchmark slot not built",
)
def test_upmem_spill_runs_with_verified_numerics():
    """The D1 real-sim correctness anchor (SPEC-022 §148).

    A GEMV whose accumulator is spilled to `mram` compiles through the
    spill path to a real artifact, runs on uPIMulator, and the GEMV host
    verifies `c == W@x` byte-for-byte. Asserts:
      (a) the artifact the simulator consumed contains the spill
          round-trip (`mram_write`/`mram_read` of the accumulator tile),
      (b) the simulator produced a cycle count (it ran), and
      (c) the GEMV byte-for-byte numeric check passed (no
          `bytes are different` / `CORRECTNESS MISMATCH` panic), so the
          spilled accumulator round-tripped through real MRAM without
          corrupting the result.
    """
    target = build_upmem_target()
    layout = allo.Placement(
        placements={
            "W": target.wram,
            "x": target.wram,
            "acc": Spilled(home_handle=target.gprs, tier="mram"),
        },
    )
    layout._spilled = ["acc"]

    compiled = compile_for_target(target, _gemv_spill_trace(), layout=layout)
    result = compiled.run()

    if result.cycles is None and "simulator unavailable" in result.stdout:
        pytest.skip(f"simulator unavailable: {result.stdout[:120]}")

    kernel_src = result.extra.get("kernel_src", "")
    # (a) The artifact the sim consumed contains the spill round-trip.
    assert result.extra.get("benchmark") == "GEMV", result.extra
    assert "spill round-trip" in kernel_src, kernel_src[-400:]
    assert "mram_write(cache_C" in kernel_src
    assert "mram_read((__mram_ptr void const*) (mram_spill_addr_C)" in kernel_src

    # (b) It ran: a cycle count came back and the process exited cleanly.
    assert result.cycles is not None and result.cycles > 0, result.stdout[-400:]
    assert result.extra.get("returncode") == 0, result.stdout[-400:]

    # (c) The GEMV host's byte-for-byte c == W@x check passed: no mismatch
    # panic. The spilled accumulator round-tripped through real MRAM with
    # the result intact.
    assert "bytes are different" not in result.stdout, result.stdout[-600:]
    assert "CORRECTNESS MISMATCH" not in result.stdout, result.stdout[-600:]


def test_no_spill_gemv_envelope_byte_identical():
    """The spill branch in `get_gemv_kernel_src` is purely additive: a
    no-spill GEMV envelope contains no spill round-trip, and the spilled
    envelope differs only by the added spill lines (zero removals)."""
    import difflib

    target = build_upmem_target()
    base = compile_for_target(target, _gemv_spill_trace())
    base_src = base._ctx.get_gemv_kernel_src()
    assert "spill round-trip" not in base_src

    layout = allo.Placement(
        placements={
            "W": target.wram,
            "x": target.wram,
            "acc": Spilled(home_handle=target.gprs, tier="mram"),
        },
    )
    layout._spilled = ["acc"]
    spilled = compile_for_target(target, _gemv_spill_trace(), layout=layout)
    spill_src = spilled._ctx.get_gemv_kernel_src()

    diff = list(
        difflib.unified_diff(
            base_src.splitlines(), spill_src.splitlines(), lineterm=""
        )
    )
    removed = [
        l for l in diff if l.startswith("-") and not l.startswith("---")
    ]
    assert removed == [], removed  # additive only: no line of the base changed


if __name__ == "__main__":
    test_upmem_spill_emits_mram_round_trip()
    test_no_spill_emits_no_round_trip()
    test_apu_v2_l2_spill_is_hard_error()
    test_no_spill_gemv_envelope_byte_identical()
    test_upmem_spill_runs_with_verified_numerics()
    print("ALL PASSED")
