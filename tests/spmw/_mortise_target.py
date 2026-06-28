# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mortise hypothetical PIM substrate -- structure-only target (design 06 §1).

`mortise` is a Samsung-like near-bank-SIMD substrate with NO simulator and
NO hardware -- there is no entry for it in `_BACKEND_CTX` / `_BACKEND_RUN`
(beyond the generic `"virtual"` adapter). It is priced purely via
`compile_for_target(target, trace, backend="virtual")`, identical to
`demo_pim`.

The structural target carries STRUCTURE ONLY -- no cost numbers (design 04
§1.2 acid test): every `allo.move` / `allo.op` is built WITHOUT `cycles=`,
so `mv.cycles is None` and `op.cycles is None` for every node. The only
numbers on the tree are GEOMETRY ("how many / how wide"), never "how many
cycles":

  * the new lever `resident_cap_elems` (capacity `C`, report-26 §1.2) --
    the size of the on-device weight-resident region in u16 weight
    elements, read by the Mortise host_staging compose to compute
    `phi = min(1, C / T_w)`;
  * `lanes_const` / `elem_bits_const` -- the bit-serial-style geometry
    constants the demo pattern carries.

Mortise is built Samsung-SHAPED on purpose (design 06 §0, §5): nested
`@allo.unit` tiles giving the same work-id count, `grf_a`/`grf_b`
registers, and the same moves/ops as `build_samsung_target` (but with
`emit=None`, since no codegen runs for this substrate). This is what makes
the report-26 §5 anchor a NUMERICAL identity, not an analogy: the Mortise
`kernel_cycles` compose delegates to the Samsung exec phase and the Mortise
`host_staging` compose to the Samsung preload/readback helpers, so at the
`phi=1` (`C >= T_w`) corner Mortise's whole-program is byte-identical to
the Samsung-validated resident schedule.
"""
from __future__ import annotations

import allo

# Largest weight tile in the swept corpus = M*K at 4096x1024 = 4,194,304 u16.
# The DEFAULT capacity const equals one full tile (phi=1 at the headline
# shape): this is GEOMETRY (how many resident elements), NOT a cost number.
_T_W_FULL = 4096 * 1024   # 4,194,304


def build_mortise_target(resident_cap_elems: int = _T_W_FULL):
    """Return the `mortise` structural target (no sim/HW backend).

    `resident_cap_elems` is the swept lever `C` (report-26 §1.2): the size
    of the on-device weight-resident region in u16 weight elements. It is a
    STRUCTURAL geometry constant -- read by the Mortise host_staging compose
    to compute `phi = min(1, C / T_w)`. Default = one full tile (phi=1, the
    Samsung physical analog). The sweep harness rebuilds the target with a
    different `C` (the sweep seam, design 06 §3.1) -- a constructor kwarg,
    not a source edit.
    """

    @allo.target("mortise")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.mem(banks=16, rows=16384, cols=128, width=8, name="banks")

            @allo.unit(mapping=[8])
            def pim():
                _, pid = allo.get_uid()
                even_bank = banks[2 * pid]
                odd_bank = banks[2 * pid + 1]
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                # --- geometry constants (structure, read by the compose) ---
                # The lever C, plus the demo-pattern geometry consts. These
                # are "how many / how wide", never "how many cycles".
                allo.const("resident_cap_elems", resident_cap_elems)
                allo.const("lanes_const", 16)       # near-bank SIMD lanes
                allo.const("elem_bits_const", 16)   # u16 weight/activation

                # Moves: structure + emit only (no codegen runs for this
                # substrate; emit is a no-op so the tree is still complete).
                # NO `cycles=` -- the acid test (design 04 §1.2) holds.
                allo.move("LD_A", src=even_bank, dst=grf_a, emit=lambda ctx: None)
                allo.move("LD_B", src=odd_bank, dst=grf_b, emit=lambda ctx: None)
                allo.move("ST_A", src=grf_a, dst=even_bank, emit=lambda ctx: None)
                allo.move("ST_B", src=grf_b, dst=odd_bank, emit=lambda ctx: None)
                allo.move("JUMP", src=grf_b, dst=grf_b, emit=lambda ctx: None)

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: None,
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: None,
                )

        # Host-side staging node (design 05 §6, parity with Samsung). Sibling
        # of `pseudo_channel` so the device unit tree depth and the work-id
        # count (`prod(mapping)` over device units) match Samsung exactly --
        # the host node has no `mapping` and contributes 1.
        @allo.unit(mode="host")
        def host():
            allo.mem(name="host_dram", bytes=1 << 30)

            @allo.host_xcel
            class hx(allo.HostXcel):
                @allo.primitive
                def broadcast(self, buf, *, over):
                    return lambda ctx, t: ctx.host_broadcast(t)

                @allo.primitive
                def scatter(self, buf, *, over):
                    return lambda ctx, t: ctx.host_scatter(t)

                @allo.primitive
                def gather(self, buf, *, over):
                    return lambda ctx, t: ctx.host_gather(t)

    return device
