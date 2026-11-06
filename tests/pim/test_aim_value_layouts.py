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


def _fanout_program(*, reupload=False, second_reads_as_y=False):
    """Producer ``a*b -> t``; ``first`` ``t*c -> u``; ``second`` ``t*d -> v``
    (or ``d*t`` when the second reader takes ``t`` in the other operand
    slot).  With ``reupload`` the host scatters ``t`` again between the two
    readers, which ends the producer's live range."""

    if second_reads_as_y:

        @region()
        def program(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            d: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
            v: bfloat16[N],
        ):
            @allo.work(mapping=[4], args=[a, b, t])
            def producer(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, c, u])
            def first(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[d, t, v])
            def second(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

    else:

        @region()
        def program(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            d: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
            v: bfloat16[N],
        ):
            @allo.work(mapping=[4], args=[a, b, t])
            def producer(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, c, u])
            def first(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

            @allo.work(mapping=[4], args=[t, d, v])
            def second(x: bfloat16[N], y: bfloat16[N], z: bfloat16[N]):
                for i in range(N):
                    z[i] = x[i] * y[i]

    if reupload:

        @allo.host_program(program)
        def host(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            d: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
            v: bfloat16[N],
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
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            d: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
            v: bfloat16[N],
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

    if second_reads_as_y:

        @allo.host_program(program)
        def host(
            a: bfloat16[N],
            b: bfloat16[N],
            c: bfloat16[N],
            d: bfloat16[N],
            t: bfloat16[N],
            u: bfloat16[N],
            v: bfloat16[N],
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


def test_every_reader_slot_of_the_live_range_is_relocated():
    """``first`` reads ``t`` in the x slot and ``second`` in the y slot: the
    producer's result is copied to both slots, not only the first reader's."""
    program, host = _fanout_program(second_reads_as_y=True)
    compiled = _compile(program, host)
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    targets = sorted(
        e["banks"][0] % 4
        for e in manifest
        if e.get("kind") == "bankcopy" and e["direction"] == "gb_to_bank"
    )
    first = next(e for e in manifest if e.get("name") == "first")
    assert len(set(targets)) == 2, targets
    assert all(targets.count(slot) == 4 for slot in set(targets))  # one per bank group
    # The relocations precede the first reader.
    indices = [e["index"] for e in manifest if e.get("kind") == "bankcopy"]
    assert max(indices) < first["index"]
