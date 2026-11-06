# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident values keep one physical distribution between writer and
readers on AiM (round 72, ISSUES X-32).

A replicated elementwise group may be realized once over the device (one
copy over 32 channels) or once per work item (four copies over 8 channels
each).  A value an EWMUL leaves in the banks is read by every later launch in
that distribution until the next writer; no redistribution between slot
layouts is lowered, so a candidate assignment whose reader differs from the
writer is refused at materialization, for every reader of the live range and
whichever side is the reduced-copy one (the reviewer's round 71 probes:
``tenon-release-evidence/reviewer-r71/fanout.py``).
"""

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import _bucket_for_autoschedule, _matcher_work_scope

N = 2048
# 147,456 elements: 1,152 per bank partition over 128 partitions (one full
# row of 1,024 and a partial second row) in the single-copy layout, and
# 4,608 per partition over 32 (four full rows and a partial fifth) in the
# per-work-item layout.
N_MULTIROW = 147456


def _fanout_program(*, reupload=False, second_reads_as_y=False, chain=False, n=N):
    """Producer ``a*b -> t``; ``first`` ``t*c -> u``; ``second`` ``t*d -> v``
    (``d*t`` when the second reader takes ``t`` in the other operand slot;
    ``u*t`` for the ``chain``, where the second reader also consumes the
    first's result).  With ``reupload`` the host scatters ``t`` again between
    the two readers, which ends the producer's live range."""

    if chain:

        @region()
        def program(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            @allo.work(mapping=[4], args=[a, b, t])
            def producer(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, c, u])
            def first(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[u, t, v])
            def second(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

    elif second_reads_as_y:

        @region()
        def program(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            @allo.work(mapping=[4], args=[a, b, t])
            def producer(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, c, u])
            def first(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[d, t, v])
            def second(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

    else:

        @region()
        def program(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            @allo.work(mapping=[4], args=[a, b, t])
            def producer(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, c, u])
            def first(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, d, v])
            def second(x: bfloat16[n], y: bfloat16[n], z: bfloat16[n]):
                for i in range(n):
                    z[i] = x[i] * y[i]

    if reupload:

        @allo.host_program(program)
        def host(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            allo.host_xfer.scatter(a, allo.host_xfer.banks)
            allo.host_xfer.scatter(b, allo.host_xfer.banks)
            allo.launch("producer", a, b, t)
            allo.host_xfer.scatter(c, allo.host_xfer.banks)
            allo.launch("first", t, c, u)
            allo.host_xfer.gather(u, allo.host_xfer.banks)
            allo.host_xfer.gather(t, allo.host_xfer.banks)
            allo.host_xfer.scatter(t, allo.host_xfer.banks)
            allo.host_xfer.scatter(d, allo.host_xfer.banks)
            allo.launch("second", t, d, v)
            allo.host_xfer.gather(v, allo.host_xfer.banks)

    else:

        @allo.host_program(program)
        def host(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            allo.host_xfer.scatter(a, allo.host_xfer.banks)
            allo.host_xfer.scatter(b, allo.host_xfer.banks)
            allo.launch("producer", a, b, t)
            allo.host_xfer.scatter(c, allo.host_xfer.banks)
            allo.launch("first", t, c, u)
            allo.host_xfer.gather(u, allo.host_xfer.banks)
            allo.host_xfer.scatter(d, allo.host_xfer.banks)
            allo.launch("second", t, d, v)
            allo.host_xfer.gather(v, allo.host_xfer.banks)

    if chain:

        @allo.host_program(program)
        def host(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            allo.host_xfer.scatter(a, allo.host_xfer.banks)
            allo.host_xfer.scatter(b, allo.host_xfer.banks)
            allo.launch("producer", a, b, t)
            allo.host_xfer.scatter(c, allo.host_xfer.banks)
            allo.launch("first", t, c, u)
            allo.launch("second", u, t, v)
            allo.host_xfer.gather(u, allo.host_xfer.banks)
            allo.host_xfer.gather(v, allo.host_xfer.banks)

    elif second_reads_as_y:

        @allo.host_program(program)
        def host(
            a: bfloat16[n],
            b: bfloat16[n],
            c: bfloat16[n],
            d: bfloat16[n],
            t: bfloat16[n],
            u: bfloat16[n],
            v: bfloat16[n],
        ):
            allo.host_xfer.scatter(a, allo.host_xfer.banks)
            allo.host_xfer.scatter(b, allo.host_xfer.banks)
            allo.launch("producer", a, b, t)
            allo.host_xfer.scatter(c, allo.host_xfer.banks)
            allo.launch("first", t, c, u)
            allo.host_xfer.gather(u, allo.host_xfer.banks)
            allo.host_xfer.scatter(d, allo.host_xfer.banks)
            allo.launch("second", d, t, v)
            allo.host_xfer.gather(v, allo.host_xfer.banks)

    return program, host


def _compile(program, host, layout=None):
    return allo.compile(
        program,
        build_aim_target(),
        aim_cost,
        host_moves=host,
        reduction_order="reassociate",
        layout=layout,
    )


def _choice(compiled, partitions_by_kernel):
    """The compiled program's placements with the named kernels replaced by
    their enumerated candidate of the given ``replica_partitions``."""
    layouts = list(compiled.compiled.layout)
    buckets = _bucket_for_autoschedule(compiled.compiled.trace)
    kernels = {k.group_id: k for k in compiled.compiled.placement_ranking.kernels}
    for index, (_scope, matches) in enumerate(buckets):
        for name, partitions in partitions_by_kernel.items():
            if matches[0].func_name.startswith(name):
                kernel = kernels[_matcher_work_scope(matches[0]).group_id]
                layouts[index] = next(
                    p
                    for _pos, p in kernel.candidates
                    if (p.extra or {}).get("replica_partitions") == partitions
                )
    return layouts


def _distributions(compiled):
    """name -> (replicas, channels per replica, elements per partition) of
    every elementwise and relocation record."""
    return {
        e["name"]: (
            e["replicas"],
            e["channels_per_replica"],
            e["elements_per_partition"],
        )
        for e in compiled.compiled.layout_ctx.lowering_manifest
        if e.get("kind") in ("elementwise", "bankcopy")
    }


@pytest.fixture(scope="module")
def fanout():
    program, host = _fanout_program()
    return program, host, _compile(program, host)


def test_default_selection_holds_t_in_one_distribution(fanout):
    _program, _host, compiled = fanout
    dist = _distributions(compiled)
    assert dist["producer"] == dist["first"] == dist["second"] == (1, 32, 16)
    assert all(value == (1, 32, 16) for value in dist.values())


def test_second_reader_of_another_distribution_is_refused(fanout):
    """Reviewer round 71, ``fanout_ACCEPTED``: the producer and the first
    reader take one copy, the second reader four; the first reader no longer
    hides the later incompatible one."""
    program, host, compiled = fanout
    with pytest.raises(
        Exception, match="second: reads t .* needs readers of the same layout"
    ):
        _compile(program, host, _choice(compiled, {"second": 8}))


def test_per_item_producer_with_single_copy_readers_is_refused(fanout):
    """Reviewer round 71, ``inverse_ACCEPTED``: the producer keeps four
    copies and both readers take one; the check does not depend on which side
    is the reduced-copy one."""
    program, host, compiled = fanout
    with pytest.raises(
        Exception, match="first: reads t .* needs readers of the same layout"
    ):
        _compile(program, host, _choice(compiled, {"producer": 8}))


def test_per_item_alternative_for_every_group_is_accepted(fanout):
    """The per-work-item form stays available when writer and readers agree."""
    program, host, compiled = fanout
    per_item = _compile(
        program, host, _choice(compiled, {"producer": 8, "first": 8, "second": 8})
    )
    dist = _distributions(per_item)
    assert dist["producer"] == dist["first"] == dist["second"] == (4, 8, 64)
    assert all(value == (4, 8, 64) for value in dist.values())


def test_a_host_reupload_ends_the_producers_live_range():
    """When the host gathers ``t`` and scatters it again before the second
    reader, that scatter follows the second reader's layout, so the second
    reader may take another distribution than the producer."""
    program, host = _fanout_program(reupload=True)
    compiled = _compile(program, host)
    mixed = _compile(program, host, _choice(compiled, {"second": 8}))
    dist = _distributions(mixed)
    assert dist["producer"] == dist["first"] == (1, 32, 16)
    assert dist["second"] == (4, 8, 64)
    # The transfers of ``t`` (a coalesced host transfer names every buffer
    # it carries, ``t+d``): read back in the producer's distribution, written
    # again in the second reader's.
    transfers = [
        (e["direction"], e["replicas"], e["channels_per_replica"])
        for e in mixed.compiled.layout_ctx.lowering_manifest
        if e.get("kind") == "distributedhosttransfer"
        and "t" in str(e["name"]).split("+")
    ]
    assert ("read", 1, 32) in transfers and ("write", 4, 8) in transfers
    # Without the reupload the same choice is refused.
    program2, host2 = _fanout_program()
    compiled2 = _compile(program2, host2)
    with pytest.raises(Exception, match="needs readers of the same layout"):
        _compile(program2, host2, _choice(compiled2, {"second": 8}))


VALUES = dict(a=2, b=3, c=5, d=7)  # exactly representable in bf16
ROW_STEP = 10  # the value of a vector in its row chunk r is base + r * ROW_STEP


def _value(name, chunk):
    return VALUES[name] + chunk * ROW_STEP


def _witness(compiled):
    """Independent command-level walk of the emitted trace: one cell per
    (row, bank) of channel 0's first bank group, every row of every vector.
    Host writes place the row chunk of their buffer (the vectors differ from
    row to row, so a chunk carried to the wrong row is visible), an EWMUL
    multiplies its x and y slots into the dst slot per row, COPY_BKGB /
    COPY_GBBK move one cell through the global buffer, host reads collect
    the result chunks.  Returns ``{(name, chunk): value}`` for the program's
    results (``None`` where nothing was written).  A generated-work check,
    not a numerical AiM execution (the simulator is timing-only)."""
    ctx = compiled.compiled.layout_ctx
    commands = [str(c) for c in ctx.cmds]
    cells, outputs, gb = {}, {}, None
    for (op, _banks), entry in zip(ctx._lowered_records, ctx.lowering_manifest):
        assert entry["name"] == op.name
        begin, end = entry["command_span"]
        for line in commands[begin:end]:
            w = line.split()
            if w[:2] in (["W", "MEM"], ["R", "MEM"]):
                channel, bank, row = map(int, w[2:])
                if channel != 0 or bank >= 4:
                    continue
                names = str(op.name).split("+")
                name = (
                    names[bank - op.bank_offset]
                    if len(names) == op.copies
                    else names[0]
                )
                chunk = row - op.row
                if w[0] == "W":
                    cells[row, bank] = (
                        _value(name, chunk)
                        if name in VALUES
                        else outputs.get((name, chunk))
                    )
                else:
                    outputs[name, chunk] = cells.get((row, bank))
            elif w[:2] == ["AiM", "EWMUL"]:
                row = int(w[-1])
                x, y = cells.get((row, 0)), cells.get((row, 1))
                cells[row, 2] = None if x is None or y is None else x * y
            elif len(w) > 1 and w[1] in ("COPY_BKGB", "COPY_GBBK"):
                bank, row = map(int, w[-2:])
                if bank >= 4:
                    continue
                if w[1] == "COPY_BKGB":
                    gb = cells.get((row, bank))
                else:
                    cells[row, bank] = gb
    return {key: value for key, value in outputs.items() if key[0] in ("u", "v")}


def _rows_of(compiled, name):
    """Rows per partition of the result ``name`` in the compiled program."""
    for entry in compiled.compiled.layout_ctx.lowering_manifest:
        if entry.get("kind") == "distributedhosttransfer" and entry["name"] == name:
            return int(entry["rows_per_partition"])
    raise KeyError(name)


def _expected(program_kind, rows=1):
    out = {}
    for r in range(rows):
        a, b, c, d = (_value(k, r) for k in "abcd")
        t = a * b
        out["u", r] = t * c
        out["v", r] = (t * c) * t if program_kind == "chain" else t * d
    return out


def test_two_reader_slots_keep_the_producers_value_alive():
    """Reviewer round 72 (`reviewer-r72/slot_liveness.py`): ``first`` reads
    ``t`` in the x slot and ``second`` in the y slot.  With every group in
    one row, the scatters of ``c`` and ``d`` and the first launch overwrote
    the copies and the original of ``t`` before the second reader ran (the
    witness gave ``v = 35``, ``d * c``).  Each group now has its own row
    region, ``t`` is copied into both readers' cells, and the witness gives
    the source semantics."""
    program, host = _fanout_program(second_reads_as_y=True)
    compiled = _compile(program, host)
    assert _witness(compiled) == _expected("fanout") == {("u", 0): 30, ("v", 0): 42}
    regions = compiled.compiled.layout_ctx.row_regions
    rows = {tuple(r["buffers"]): r["row"] for r in regions}
    assert len({rows[k] for k in rows}) == 3  # one region per group


def test_witness_on_the_default_fan_out(fanout):
    _program, _host, compiled = fanout
    assert _witness(compiled) == _expected("fanout")


def test_witness_on_the_per_item_alternative(fanout):
    program, host, compiled = fanout
    per_item = _compile(
        program, host, _choice(compiled, {"producer": 8, "first": 8, "second": 8})
    )
    assert _witness(per_item) == _expected("fanout")


def test_witness_on_the_reupload_composition():
    program, host = _fanout_program(reupload=True)
    compiled = _compile(program, host)
    assert _witness(compiled) == _expected("fanout")
    mixed = _compile(program, host, _choice(compiled, {"second": 8}))
    assert _witness(mixed) == _expected("fanout")


def test_witness_on_the_neighbouring_chain():
    """``second`` consumes the first reader's result and the producer's:
    ``v = u * t``.  Both values are copied into the second reader's cells
    from live sources (``u`` right after ``first``, ``t`` right after the
    producer) and the first reader's launch does not disturb them."""
    program, host = _fanout_program(chain=True)
    compiled = _compile(program, host)
    assert _witness(compiled) == _expected("chain") == {("u", 0): 30, ("v", 0): 180}


def test_a_slot_read_of_an_overwritten_cell_is_refused(monkeypatch):
    """The production liveness check itself: force every elementwise group
    into one row (the pre-round-73 allocation) and the reviewer's program is
    refused at materialization instead of lowered with a stale cell."""
    from allo import spmw_aim

    original = spmw_aim._AimProgramDerivation._allocate_rows

    def one_row(self):
        regions = original(self)
        for group in self.groups:
            group.row = 0
        return regions

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "_allocate_rows", one_row)
    program, host = _fanout_program(second_reads_as_y=True)
    with pytest.raises(Exception, match="physical liveness"):
        _compile(program, host)


@pytest.fixture(scope="module")
def multirow():
    program, host = _fanout_program(second_reads_as_y=True, n=N_MULTIROW)
    return program, host, _compile(program, host)


def test_multirow_copies_carry_every_row_chunk(multirow):
    """Reviewer round 73 (`reviewer-r73/multirow.py`): a relocation of a
    vector spanning several rows read every source row into the global
    buffer before writing any destination row, so the last chunk landed in
    every destination row (`u = (60, 60)` for `(30, 60)`).  Copies now move
    one row chunk at a time; with a partial final row and values that differ
    from row to row, every result chunk is the source's."""
    _program, _host, compiled = multirow
    rows = _rows_of(compiled, "v")
    assert rows == 2
    assert _witness(compiled) == _expected("fanout", rows)
    commands = [str(c) for c in compiled.compiled.layout_ctx.cmds]
    copies = [c.split() for c in commands if "COPY_" in c]
    # Source and destination rows alternate chunk by chunk per bank.
    kinds = [w[1] for w in copies[:4]]
    assert kinds == ["COPY_BKGB", "COPY_GBBK", "COPY_BKGB", "COPY_GBBK"], copies[:4]
    assert int(copies[1][-1]) - int(copies[0][-1]) == int(copies[3][-1]) - int(
        copies[2][-1]
    )
    assert int(copies[2][-1]) == int(copies[0][-1]) + 1
    # The partial final row moves fewer columns than a full row.
    assert int(copies[2][2]) < int(copies[0][2])


def test_multirow_copies_in_the_per_item_layout(multirow):
    program, host, compiled = multirow
    per_item = _compile(
        program, host, _choice(compiled, {"producer": 8, "first": 8, "second": 8})
    )
    rows = _rows_of(per_item, "v")
    assert rows == 5
    assert _witness(per_item) == _expected("fanout", rows)


def test_multirow_chain(multirow):
    program, host = _fanout_program(chain=True, n=N_MULTIROW)
    compiled = _compile(program, host)
    rows = _rows_of(compiled, "v")
    assert rows == 2
    assert _witness(compiled) == _expected("chain", rows)


def test_a_whole_vector_copy_record_is_refused(monkeypatch):
    """The production liveness check itself: a multi-row copy lowered as one
    record per direction (the round 73 form) is refused at materialization,
    because the global buffer holds one row."""
    from allo import spmw_aim

    original = spmw_aim._AimProgramDerivation._copy_records

    def whole_vector(self, source, source_usage, key, readers):
        records = original(self, source, source_usage, key, readers)
        merged = []
        seen = set()
        for op, banks in records:
            base = op.name.split(".row")[0]
            if base in seen:
                continue
            seen.add(base)
            whole = type(op)(
                op.direction,
                bank=op.bank,
                row=op.row,
                elements=source.elements,
                replicas=op.replicas,
                channels_per_replica=op.channels_per_replica,
                partitions_per_replica=op.partitions_per_replica,
                name=base,
            )
            self._copy_identity[id(whole)] = (self._copy_identity[id(op)][0], 0)
            merged.append((whole, banks))
        return merged

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "_copy_records", whole_vector)
    program, host = _fanout_program(second_reads_as_y=True, n=N_MULTIROW)
    with pytest.raises(Exception, match="global buffer holds row chunk"):
        _compile(program, host)
