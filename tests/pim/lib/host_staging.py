# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-staging helpers for the Samsung GEMM-family cells (spec
`backend-host-transfer-dispatch.md`, task 004).

A workload expresses its host data movement *visibly* via the `allo.host_xfer`
proxy: the weight operand is SCATTERED into per-bank DRAM, the input vector is
BROADCAST into GRF_A, and the output is GATHERED back from bank DRAM. The triple
is uniform across the single-stage GEMM-family kernels, so `stage_gemm` records
it in one call; multi-stage chains call it per stage and OMIT the intermediate
`gather` where the result stays device-resident (per-cell override).

The recording happens inside an `allo.record_host_moves()` scope opened by the
workload at module-import time (D1 spelling (a)). Buffer operands are named by
their `@allo.work` parameter name (a string) so the resolved move carries the
operand->role binding (`scatter`=weight, `broadcast`=input, `gather`=output)
that `_run_samsung` asserts against the inferred roles. The names round-trip as
`buffer_role`; the device endpoint round-trips by handle identity.
"""

from __future__ import annotations

import allo


def stage_in(*, weight, vec):
    """Record the host->device staging for one MAC stage: SCATTER the weight
    into bank DRAM, BROADCAST the input vector into GRF_A. `weight`/`vec` are the
    operand parameter names (strings). No `gather` -- the output stays device
    resident for a chained stage."""
    allo.host_xfer.scatter(weight, allo.host_xfer.banks)
    allo.host_xfer.broadcast(vec, allo.host_xfer.grf_a)


def broadcast_vec(*, vec):
    """Record a standalone BROADCAST of `vec` into GRF_A (the per-stage input
    vector when the weight is already device-resident, e.g. gemver's chained
    rank-1 + GEMV stages). `vec` is the operand parameter name (a string)."""
    allo.host_xfer.broadcast(vec, allo.host_xfer.grf_a)


def gather_out(*, out):
    """Record the device->host readback (GATHER) of `out` from bank DRAM."""
    allo.host_xfer.gather(out, allo.host_xfer.banks)


def stage_gemm(*, weight, vec, out):
    """The uniform single-stage GEMM-family triple: scatter the weight, broadcast
    the input vector, gather the output. `weight`/`vec`/`out` are the operand
    parameter names (strings)."""
    stage_in(weight=weight, vec=vec)
    gather_out(out=out)
