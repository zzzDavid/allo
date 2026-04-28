# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for SPMW tests.

The Samsung HBM-PIM target spec from report 16 is defined here so that
both the matcher tests and the codegen tests can build it without
duplicating the spec.
"""

from __future__ import annotations

import allo


def build_samsung_target():
    """Return the Samsung HBM-PIM target tree per report 16, including
    `emit=` callbacks on every move and op.

    The emit lambdas use Tenon handles directly (`grf_a`, `even_bank`,
    ...). Codegen passes a backend-specific `ctx` whose `cmd(...)` method
    translates those handles into the simulator's operand encoding.
    """

    @allo.target("samsung_hbm_pim")
    def device():
        @allo.unit(mapping=[16])
        def pseudo_channel():
            banks = allo.memory(banks=16, rows=16384, cols=128, width=8, name="banks")

            @allo.unit(mapping=[8])
            def pim():
                _, pid = allo.get_uid()
                even_bank = banks[2 * pid]
                odd_bank = banks[2 * pid + 1]
                grf_a = allo.reg(8, 256, name="grf_a")
                grf_b = allo.reg(8, 256, name="grf_b")

                allo.move(
                    "LD_A", src=even_bank, dst=grf_a,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_a, src0=even_bank),
                )
                allo.move(
                    "LD_B", src=odd_bank, dst=grf_b,
                    emit=lambda ctx: ctx.cmd("MOV", dst=grf_b, src0=odd_bank),
                )
                allo.move(
                    "ST_A", src=grf_a, dst=even_bank,
                    emit=lambda ctx: ctx.cmd("MOV", dst=even_bank, src0=grf_a),
                )
                allo.move(
                    "ST_B", src=grf_b, dst=odd_bank,
                    emit=lambda ctx: ctx.cmd("MOV", dst=odd_bank, src0=grf_b),
                )

                any_bank = allo.any_(banks)
                any_reg = allo.any_([grf_a, grf_b])
                allo.op(
                    "MUL",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=any_reg,
                    fn=lambda x, y: x * y,
                    emit=lambda x, y, dst, ctx: ctx.cmd("MUL", dst=dst, src0=x, src1=y),
                )
                allo.op(
                    "MAC",
                    src=(allo.or_(any_bank, any_reg), allo.or_(any_bank, any_reg)),
                    dst=grf_b,
                    accumulates=True,
                    fn=lambda x, y, acc: acc + x * y,
                    emit=lambda x, y, acc, ctx: ctx.cmd("MAC", dst=acc, src0=x, src1=y),
                )

    return device
