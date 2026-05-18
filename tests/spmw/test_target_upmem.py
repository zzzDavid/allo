# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the UPMEM (DPU, DRAM-PIM) target spec.

Four per spec 003 §F.5: target build, ctx C-line emission via a MAC
emit lambda, kernel-source wrapping, and the kernel_cycles cost factory
returning a callable for an empty trace.
"""

from __future__ import annotations

import allo
from allo.spmw_codegen import UPMEMCtx
from allo.spmw_match import MatchTrace
from allo.spmw_target import Memory, Register

from _fixtures import build_upmem_target


def test_upmem_target_builds():
    target = build_upmem_target()
    assert target.name == "upmem"
    # device -> rank child.
    assert len(target.root.children) == 1
    # DPU-level memories and tasklet-level registers resolve via the
    # flat handle map.
    assert isinstance(target.mram, Memory)
    assert isinstance(target.wram, Memory)
    assert isinstance(target.gprs, Register)
    assert target.mram.geometry["size_bytes"] == 67108864
    assert target.wram.geometry["size_bytes"] == 65536
    assert target.gprs.lanes == 24


def test_upmem_ctx_emits_c():
    target = build_upmem_target()
    ctx = UPMEMCtx(target)
    # Invoke the MAC op's emit lambda directly with operand handles.
    mac = target.op("MAC")
    x = target.wram[0]
    y = target.wram[1]
    acc = target.gprs
    mac.emit(x, y, acc, ctx)

    assert len(ctx.cmds) == 1
    line = ctx.cmds[0]
    assert "+=" in line, line
    assert "*" in line, line
    # The accumulator handle is the gprs register; the emit lambda
    # should render its C name as `gprs`.
    assert "gprs" in line, line


def test_upmem_get_kernel_src_wraps():
    target = build_upmem_target()
    ctx = UPMEMCtx(target)
    ctx.emit_c_line("acc += local_W[i] * local_x[k];")
    src = ctx.get_kernel_src()
    assert "#include <defs.h>" in src
    assert "int main(void)" in src
    # Body line is preserved (with the indentation prefix).
    assert "acc += local_W[i] * local_x[k];" in src


def test_upmem_cost_factory_returns_callable():
    target = build_upmem_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    assert callable(cost_fn)
    empty_trace = MatchTrace(
        target_name="upmem", module_name="empty", matches=[]
    )
    assert cost_fn(empty_trace, allo.Placement(placements={})) == 0


if __name__ == "__main__":
    test_upmem_target_builds()
    test_upmem_ctx_emits_c()
    test_upmem_get_kernel_src_wraps()
    test_upmem_cost_factory_returns_callable()
    print("ALL PASSED")
