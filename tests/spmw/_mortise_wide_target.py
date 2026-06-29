# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mortise-WIDE structural target: banks_per_pim == 4 (SPEC-022 D3 proof).

A no-simulator, Samsung-shaped target whose pim fanout (`mapping=[4]`) divides
the 16-bank axis into FOUR banks per pim instead of Samsung's two. This is the
acceptance fixture for the `range(stride)` fiber walk in `_bank_fiber_class`:
with `banks_per_pim == 4` the enumerator materialises 4 fiber handles and
codegen emits a 4-way `(MAC, JUMP)` walk classifying the per-fiber banks as
`BANK_0..BANK_3` -- a layout class that was UN-EMITTABLE before D3 (the deleted
two-class `_bank_parity` matcher only knew `EVEN_BANK`/`ODD_BANK`).

There is no real silicon with `banks_per_pim > 2`, and PIMSimulator's
`PIMOpdType` enum has only the two Samsung bank names, so this is proven on a
structural fixture -- the same way Mortise proves capacity-lever reasoning with
no simulator. The claim is a property of the codegen + F2 layout algebra, not
of any one silicon.

The target is registered (in `test_banks_per_pim.py`'s module import) under the
name `"mortise_wide"`, delegating its enumerator to `_samsung_enumerate` (it is
Samsung-shaped) and its codegen ctx to `SamsungCtx` (which now emits the
per-fiber `BANK_<r>` classes via `_bank_fiber_class`).
"""
from __future__ import annotations

import allo


def build_mortise_wide_target():
    """Return the `mortise_wide` structural target: 16 banks, pim fanout 4
    (banks_per_pim == 4). Samsung-shaped (grf_a/grf_b, same moves/ops) so the
    Samsung enumerator + SamsungCtx drive it unchanged; emit lambdas are
    real-ctx (SamsungCtx.cmd), so `compile_for_target` produces PIMCmds."""

    @allo.target("mortise_wide")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.mem(
                banks=16, rows=16384, cols=128, width=8, name="banks"
            )

            # pim fanout 4 -> banks_per_pim = 16 // 4 = 4 (stride 4, 4 fibers).
            @allo.unit(mapping=[4])
            def pim():
                _, pid = allo.get_uid()
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                # Each pim owns the contiguous run banks[4*pid .. 4*pid+3].
                # Moves are named so the move scheduler resolves them; emit
                # goes through the ctx (SamsungCtx) at codegen time.
                fiber0 = banks[4 * pid]
                fiber1 = banks[4 * pid + 1]
                allo.move(
                    "LD_A", src=fiber0, dst=grf_a,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_a, src0=fiber0),
                )
                allo.move(
                    "LD_B", src=fiber1, dst=grf_b,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_b, src0=fiber1),
                )
                allo.move(
                    "ST_A", src=grf_a, dst=fiber0,
                    emit=lambda ctx: ctx.cmd("MOV", dst=fiber0, src0=grf_a),
                )
                allo.move(
                    "ST_B", src=grf_b, dst=fiber1,
                    emit=lambda ctx: ctx.cmd("MOV", dst=fiber1, src0=grf_b),
                )
                allo.move("JUMP", src=grf_b, dst=grf_b, emit=lambda ctx: None)

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg),
                         allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: ctx.cmd(
                        "MUL", dst=dst, src0=x, src1=y),
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg),
                         allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.cmd(
                        "MAC", dst=acc, src0=x, src1=y),
                )

    return device
