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


def test_upmem_get_kernel_src_emits_prim_envelope():
    """SPEC-003 §4.2 Region A: get_kernel_src must emit a full PrIM-shaped
    DPU envelope so uPIMulator's linker can resolve DPU_INPUT_ARGUMENTS,
    the kernels[] dispatch table, and BARRIER_INIT. The emitted cmd is
    inlined into the tenon_kernel() body, not into main().
    """
    target = build_upmem_target()
    ctx = UPMEMCtx(target)
    ctx.emit_c_line("bufferB[i] += bufferA[i];")
    src = ctx.get_kernel_src()
    # Envelope must declare the host arguments struct and dispatch table.
    assert "__host dpu_arguments_t DPU_INPUT_ARGUMENTS;" in src, src
    assert "BARRIER_INIT(my_barrier, NR_TASKLETS);" in src, src
    assert "int (*kernels[nr_kernels])(void)" in src, src
    # Envelope must perform MRAM<->WRAM staging.
    assert "mram_read(" in src and "mram_write(" in src, src
    assert "DPU_MRAM_HEAP_POINTER" in src, src
    # Envelope must include the support header (TENON/support/common.h).
    assert '#include "../support/common.h"' in src, src
    # The emitted body lives inside tenon_kernel, not main.
    assert "void __attribute__ ((noinline))" in src, src
    assert "tenon_kernel(T *bufferB, T *bufferA, unsigned int l_size)" in src, src
    assert "bufferB[i] += bufferA[i];" in src, src


def test_run_upmem_uses_tenon_slot_and_no_proxy():
    """SPEC-003 §6 acceptance 1: _run_upmem must invoke uPIMulator with
    --benchmark TENON and must not contain the old VA/GEMV proxy
    heuristic.
    """
    import inspect
    from allo.spmw_codegen import _run_upmem

    source = inspect.getsource(_run_upmem)
    assert '"TENON"' in source, source
    assert 'benchmark = "VA"' not in source, source
    assert 'benchmark = "GEMV"' not in source, source
    # The TENON slot path must be the one we write task.c into.
    assert "benchmark" in source and "TENON" in source and "task.c" in source


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
    test_upmem_get_kernel_src_emits_prim_envelope()
    test_run_upmem_uses_tenon_slot_and_no_proxy()
    test_upmem_cost_factory_returns_callable()
    print("ALL PASSED")
