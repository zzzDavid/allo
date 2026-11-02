# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AttAcc analytical backend: placements, program emission, no runner.

The emitted program of an AttAcc kernel is its command inventory under the
selected tiling (:class:`allo.pim.attacc_tiling.AttAccProgram`), one record
per ``@allo.work`` kernel in ``AttAccCtx.programs`` and one
``"<opcode> count=<n>"`` line per command kind in ``Compiled.cmds``.  The
release has no device and no in-package simulator for AttAcc: a compiled
AttAcc workload runs only through ``backend="virtual"`` (``kind="virtual"``).
Ramulator2-AttAcc timing of the selected program is an external evaluation
script (``benchmarks/attacc/scripts/run_ramulator.py``) because the trace
generator it needs is unlicensed DCC code (``docs/release/simulators.md``).
"""

from __future__ import annotations

from .pim.attacc_tiling import (
    AttAccDomainError,
    AttAccKernel,
    AttAccTiling,
    attacc_geometry,
    attacc_kernel_from_matches,
    attacc_program,
)
from .spmw_codegen import CodegenContext

ATTACC_ANALYTICAL_ONLY = (
    "attacc is an analytical-only backend in this release: there is no AttAcc "
    "device and no in-package simulator runner, so a compiled AttAcc workload "
    "cannot be executed. Compile with backend='virtual' for the cost-ranked "
    "tiling and its closed-form estimate (kind='virtual'). Ramulator2-AttAcc "
    "timing of the selected program is an external evaluation script: "
    "benchmarks/attacc/scripts/run_ramulator.py (see docs/release/simulators.md)."
)


def attacc_base_placement(target, matches):
    """The single base placement of one AttAcc kernel.

    Matrix operands (weights, K and V caches) live in the banks, vector
    operands in the per-bank GEMV buffer, MAC accumulators in the PIM
    accumulator, and the softmax sum in the softmax buffer.  The score row of
    an attention kernel has two homes (the softmax buffer for ``SFM_*`` after
    ``MV_SB``, the GEMV buffer for the context MAC after ``MV_GB``); a
    ``Placement`` holds one handle per value, so it stays unplaced and its
    traffic is the record's ``PIM_MV_SB`` / ``PIM_MV_GB`` counts.
    """
    from .spmw_autoschedule import Placement

    kernel = attacc_kernel_from_matches(matches)
    homes: dict[str, set] = {}

    def want(memref, handle):
        if memref is not None:
            homes.setdefault(memref, set()).add(handle)

    for match in matches:
        if match.target_op_name == "MAC":
            roles = match.extra["axis_roles"]
            for operand in match.operands:
                if operand.is_loop_carried:
                    want(operand.memref_name, target.acc)
                elif operand.role == roles["matrix"]:
                    want(operand.memref_name, target.banks)
                elif operand.role == roles["vector"]:
                    want(operand.memref_name, target.gemv_buffer)
        else:
            for operand in match.operands:
                want(operand.memref_name, target.softmax_buffer)
            want(match.result_memref_name, target.softmax_buffer)
    placements = {
        memref: next(iter(handles))
        for memref, handles in homes.items()
        if len(handles) == 1
    }
    return Placement(
        placements=placements,
        mode=f"attacc_{kernel.kind}",
        extra={"attacc_kernel": kernel},
    )


def attacc_tiling_candidates(target, base) -> list[AttAccTiling]:
    """``attacc_tiling`` knob values: every legal tiling of the base's kernel."""
    from .pim.attacc_tiling import attacc_tiling_space

    kernel = (getattr(base, "extra", {}) or {}).get("attacc_kernel")
    if kernel is None:
        return []
    return attacc_tiling_space(kernel, attacc_geometry(target))


def attacc_placement(kernel: AttAccKernel, tiling, placements=None):
    """A ``layout=`` placement fixing ``tiling`` for ``kernel``."""
    from .spmw_autoschedule import Placement

    tiling = AttAccTiling.coerce(tiling)
    return Placement(
        placements=dict(placements or {}),
        mode=f"attacc_{kernel.kind}+attacc_tiling={tiling}",
        extra={"attacc_kernel": kernel, "attacc_tiling": tiling},
    )


class AttAccCtx(CodegenContext):
    """Lowers each kernel to its AttAcc command inventory."""

    def __init__(self, target):
        super().__init__(target)
        self.geometry = attacc_geometry(target)
        self.programs = []

    def cmd(self, name, **fields):
        raise NotImplementedError(
            "AttAcc ops are lowered per kernel by AttAccCtx.emit_program"
        )

    def resolve_moves(self, role, src_handle, dst_handle):
        return (None, None)

    def emit_program(self, trace, layouts, *, schedule=None, host_moves=(), module=None):
        from .pim.schedule_search import InfeasibleSchedule
        from .spmw_autoschedule import Placement, _bucket_for_autoschedule

        buckets = _bucket_for_autoschedule(trace)
        if isinstance(layouts, Placement):
            layouts = [layouts] * len(buckets)
        for (_scope, matches), placement in zip(buckets, layouts):
            kernel = attacc_kernel_from_matches(matches)
            extra = getattr(placement, "extra", {}) or {}
            carried = extra.get("attacc_kernel")
            if carried is not None and carried != kernel:
                raise InfeasibleSchedule(
                    f"AttAcc placement was built for {carried}, kernel is {kernel}"
                )
            tiling = extra.get("attacc_tiling")
            if tiling is None:
                raise InfeasibleSchedule(
                    "AttAcc placement carries no attacc_tiling; use the AttAcc "
                    "enumerator or allo.spmw_attacc.attacc_placement"
                )
            try:
                program = attacc_program(kernel, tiling, self.geometry)
            except (AttAccDomainError, ValueError) as error:
                raise InfeasibleSchedule(str(error)) from error
            self.programs.append(program)
            self.cmds.extend(program.command_lines())

    def runtime_route(self, commands):
        return ("attacc-analytical", ())

    def virtual_run_extra(self):
        from .pim.costs.attacc import ATTACC_COST_VALIDITY

        programs = [program.manifest() for program in self.programs]
        extra = {
            "timing_scope": (
                "analytical: closed-form per-channel makespan of one device's "
                "AttAcc command inventory; not simulator timing"
            ),
            "cost_validity": ATTACC_COST_VALIDITY,
            "attacc_programs": programs,
            "simulator_timing": (
                "external, not run: benchmarks/attacc/scripts/run_ramulator.py "
                "(Ramulator2-AttAcc)"
            ),
        }
        if len(programs) == 1:
            extra["attacc_tiling"] = programs[0]["tiling"]
            extra["attacc_kernel"] = programs[0]["kernel"]
            extra["attacc_counts"] = programs[0]["counts"]
        return extra


def _run_attacc(compiled, **inputs):
    raise NotImplementedError(ATTACC_ANALYTICAL_ONLY)


__all__ = [
    "ATTACC_ANALYTICAL_ONLY",
    "AttAccCtx",
    "attacc_base_placement",
    "attacc_placement",
    "attacc_tiling_candidates",
]
