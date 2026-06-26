# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-5 no-backend demo substrate (design 04 §8).

`demo_pim` is a bit-serial PIM substrate with **no simulator and no
hardware** -- there is no entry for it in `_BACKEND_CTX` / `_BACKEND_RUN`
(beyond the generic `"virtual"` adapter) and no candidate enumerator. It
exists only to demonstrate "a backend that ships with the spec": a user
develops a kernel and gets a cycle estimate purely from
`(structural target) + (CostModel)` via
`compile_for_target(target, trace, backend="virtual")`.

The structural target carries STRUCTURE ONLY -- no cost numbers (design 04
§1.2 acid test). Its geometry constants (`lanes_const`, `elem_bits_const`)
are read by the demo CostModel's compose to build the rich `OpCostCtx` the
MICRO-2025 analytical OpCost.fn consumes. The cost NUMBERS live in
`spmw_cost_tables.{DEMO_PIM_CONSTANT, DEMO_PIM_MICRO25}`.
"""
from __future__ import annotations

import allo


def build_demo_pim_target():
    """Return the `demo_pim` structural target (no sim/HW backend)."""

    @allo.target("demo_pim")
    def device():
        # Bit-serial geometry, surfaced as constants the CostModel reads.
        allo.const("lanes_const", 16384)   # bit-serial lanes per VR
        allo.const("elem_bits_const", 16)  # u16 elements

        mem = allo.mem(size_bytes=1 << 20, name="scratch")

        @allo.unit(mapping=[8])  # 8 compute tiles
        def tile():
            vr = allo.reg(16, 16384, name="vr")

            # Moves: structure + emit only (no codegen runs for this
            # substrate; emit is a no-op so the tree is still complete).
            allo.move("LD", src=mem, dst=vr, emit=lambda ctx: None)
            allo.move("ST", src=vr, dst=mem, emit=lambda ctx: None)

            any_vr = allo.any_([vr])
            allo.op(
                "ADD", src=(any_vr, any_vr), dst=any_vr,
                fn=lambda x, y: x + y,
                emit=lambda x, y, dst, ctx: None,
            )
            allo.op(
                "MUL", src=(any_vr, any_vr), dst=any_vr,
                fn=lambda x, y: x * y,
                emit=lambda x, y, dst, ctx: None,
            )
            allo.op(
                "MAC", src=(any_vr, any_vr), dst=any_vr,
                accumulates=True,
                fn=lambda x, y, acc: acc + x * y,
                emit=lambda x, y, acc, ctx: None,
            )

    return device
