# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AiM elementwise lowering covers dense iteration domains only (round
76, ISSUES X-33).

An EWMUL / EWADD stream writes every element of a contiguous span of the
result, so a kernel whose loop steps by two, or whose store coordinate is
``2 * i``, would overwrite the elements the source leaves unchanged (the
reviewer's round 75 witness: ``u[1]`` lost its initial value).  Such kernels
are refused before selection; dense kernels, including dense suffixes with a
nonzero lower bound, keep compiling and their generated work is the
source's.  Generated-work checks, not numerical AiM execution (the simulator
is timing-only).
"""

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target

N = 8192
H = N // 2


def _program(kind):
    """Two multiply stages over ``[lower, N)`` with ``step``; the host moves
    exactly the kernels' span (``[lower, N - 1)`` for the stride-two loop,
    whose last index is ``N - 2``)."""
    lower, step, last = {
        "dense": (0, 1, N),
        "suffix": (H, 1, N),
        "stride": (0, 2, N - 1),
    }[kind]

    if kind == "stride":

        @region()
        def program(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
        ):
            @allo.work(mapping=[2], args=[a, b, t])
            def alpha(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(0, N, 2):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[2], args=[t, c, u])
            def beta(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(0, N, 2):
                    z[i] = y[i] * x[i]

    elif kind == "suffix":

        @region()
        def program(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
        ):
            @allo.work(mapping=[2], args=[a, b, t])
            def alpha(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(H, N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[2], args=[t, c, u])
            def beta(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(H, N):
                    z[i] = y[i] * x[i]

    else:

        @region()
        def program(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
        ):
            @allo.work(mapping=[2], args=[a, b, t])
            def alpha(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[2], args=[t, c, u])
            def beta(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = y[i] * x[i]

    @allo.host_program(program)
    def host(
        a: bfloat16[N], b: bfloat16[N], c: bfloat16[N], t: bfloat16[N], u: bfloat16[N]
    ):
        allo.host_xfer.scatter(a[lower:last], allo.host_xfer.banks)
        allo.host_xfer.scatter(b[lower:last], allo.host_xfer.banks)
        allo.launch("alpha", a, b, t)
        allo.host_xfer.scatter(c[lower:last], allo.host_xfer.banks)
        allo.launch("beta", t, c, u)
        allo.host_xfer.gather(u[lower:last], allo.host_xfer.banks)

    return program, host


def _strided_coordinate_program():
    """A dense loop whose store and loads address every other element."""

    @region()
    def program(a: bfloat16[N], b: bfloat16[N], t: bfloat16[N]):
        @allo.work(mapping=[2], args=[a, b, t])
        def alpha(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
            for i in range(H):
                z[2 * i] = x[2 * i] * y[2 * i]

    @allo.host_program(program)
    def host(a: bfloat16[N], b: bfloat16[N], t: bfloat16[N]):
        allo.host_xfer.scatter(a[0 : N - 1], allo.host_xfer.banks)
        allo.host_xfer.scatter(b[0 : N - 1], allo.host_xfer.banks)
        allo.launch("alpha", a, b, t)
        allo.host_xfer.gather(t[0 : N - 1], allo.host_xfer.banks)

    return program, host


def _compile(program, host):
    return allo.compile(
        program,
        build_aim_target(),
        aim_cost,
        host_moves=host,
        reduction_order="reassociate",
    )


def _witness(compiled, values=dict(a=2, b=3, c=5)):
    """Command-level walk of channel 0's first bank group, two lanes per cell
    (the EWMUL and the copies act on both; a host burst writes both).  The
    result's second lane keeps its initial value only if nothing writes it."""
    ctx = compiled.compiled.layout_ctx
    commands = [str(c) for c in ctx.cmds]
    cells, gb, out = {}, None, {}
    for (op, _b), entry in zip(ctx._lowered_records, ctx.lowering_manifest):
        seen = set()
        lo, hi = entry["command_span"]
        for command in commands[lo:hi]:
            w = command.split()
            if w[:2] in (["W", "MEM"], ["R", "MEM"]):
                ch, bank, row = map(int, w[2:])
                if ch != 0 or bank >= 4 or (bank, row) in seen:
                    continue
                seen.add((bank, row))
                names = str(op.name).split("+")
                copy = (bank - op.bank_offset) % op.bank_stride
                name = (names[copy] if len(names) == op.copies else names[0]).split(
                    "["
                )[0]
                if w[0] == "W":
                    cells[row, bank] = (values[name], values[name])
                elif name == "u" and row == op.row:
                    out["u[0]"], out["u[1]"] = cells[row, bank]
            elif w[:2] == ["AiM", "EWMUL"]:
                row = int(w[-1])
                x, y = cells[row, 0], cells[row, 1]
                cells[row, 2] = tuple(p * q for p, q in zip(x, y))
            elif len(w) > 1 and w[1] in ("COPY_BKGB", "COPY_GBBK"):
                bank, row = map(int, w[-2:])
                if bank >= 4:
                    continue
                if w[1] == "COPY_BKGB":
                    gb = cells[row, bank]
                else:
                    cells[row, bank] = gb
    return out


def test_a_stride_two_loop_is_refused_before_selection():
    """Reviewer round 75 (`reviewer-r75/strided.py even`): the kernel writes
    every other element, the lowering wrote them all (`u[1] = 30` for the
    untouched 99)."""
    program, host = _program("stride")
    with pytest.raises(
        Exception, match="dense iteration domains only; loop .* steps by 2"
    ):
        _compile(program, host)


def test_a_strided_store_coordinate_is_refused():
    program, host = _strided_coordinate_program()
    with pytest.raises(Exception, match="not one loop variable with coefficient one"):
        _compile(program, host)


@pytest.mark.parametrize("kind", ["dense", "suffix"])
def test_dense_domains_keep_compiling_with_the_sources_work(kind):
    """The dense kernel and the dense suffix (a nonzero lower bound, step
    one) compile; the generated work multiplies both lanes of every written
    cell, which is what the source does for every element of its span."""
    program, host = _program(kind)
    compiled = _compile(program, host)
    assert _witness(compiled) == {"u[0]": 30, "u[1]": 30}
    elementwise = [
        e
        for e in compiled.compiled.layout_ctx.lowering_manifest
        if e.get("kind") == "elementwise"
    ]
    assert [e["elements"] for e in elementwise] == [N if kind == "dense" else H] * 2
