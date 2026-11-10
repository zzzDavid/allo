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
    attacc_orientations,
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


def attacc_orientation_of(matches, extra) -> dict:
    """The orientation a placement realizes: its ``attacc_orientation``, else
    the one whose matrix the carried kernel names, else the shared roles'."""
    orientations = attacc_orientations(matches)
    extra = extra or {}
    stated = extra.get("attacc_orientation")
    if stated is not None:
        return stated
    carried = extra.get("attacc_kernel")
    matrix = getattr(carried, "matrix", None)
    if matrix is not None:
        for orientation in orientations:
            if orientation.get("matrix") == matrix:
                return orientation
    return orientations[0]


def attacc_base_placement(target, matches, orientation=None):
    """One base placement of one AttAcc kernel, per storage orientation
    (`allo.pim.attacc_tiling.attacc_orientations`; the shared roles' one by
    default).

    Matrix operands (weights, K and V caches) live in the banks, vector
    operands in the per-bank GEMV buffer, MAC accumulators in the PIM
    accumulator, and the softmax sum in the softmax buffer.  The score row of
    an attention kernel has two homes (the softmax buffer for ``SFM_*`` after
    ``MV_SB``, the GEMV buffer for the context MAC after ``MV_GB``); a
    ``Placement`` holds one handle per value, so it stays unplaced and its
    traffic is the record's ``PIM_MV_SB`` / ``PIM_MV_GB`` counts.
    """
    from .spmw_autoschedule import Placement

    if orientation is None:
        orientation = attacc_orientations(matches)[0]
    kernel = attacc_kernel_from_matches(matches, orientation)
    homes: dict[str, set] = {}

    def want(memref, handle):
        if memref is not None:
            homes.setdefault(memref, set()).add(handle)

    for match in matches:
        if match.target_op_name == "MAC":
            roles = match.extra["axis_roles"]
            matrix_role = orientation.get("matrix_role", roles["matrix"])
            vector_role = orientation.get("vector_role", roles["vector"])
            for operand in match.operands:
                if operand.is_loop_carried:
                    want(operand.memref_name, target.acc)
                elif operand.role == matrix_role:
                    want(operand.memref_name, target.banks)
                elif operand.role == vector_role:
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
    suffix = "_transposed" if orientation.get("by") == "alternative" else ""
    name = f"attacc_{kernel.kind}{suffix}"
    return Placement(
        placements=placements,
        mode=name,
        extra={
            "attacc_kernel": kernel,
            "attacc_orientation": orientation,
            # ``layout="attacc_gemv_transposed"`` forces an orientation (every
            # tiling of it stays a candidate; the cheapest is selected).
            "candidate_families": (name,),
        },
    )


def _launch_extents(fact) -> tuple[int, int | None, str | None]:
    """``(instances, reduction, accumulator)`` of one match's facts: a
    contraction's result trips, reduction trips and initialization; an
    epilogue's loop trips, no reduction."""
    trips = fact.trips
    if hasattr(fact, "reduction_vars"):
        instances = reduction = 1
        for var in fact.result_vars:
            instances *= int(trips.get(var) or 0)
        for var in fact.reduction_vars:
            reduction *= int(trips.get(var) or 0)
        if not fact.reduction_vars:
            # An accumulate without a reduction loop sums nothing per instance.
            return instances, None, fact.accumulator_init
        return instances, reduction, fact.accumulator_init
    instances = 1
    for loop in fact.loops:
        instances *= int(loop.trip or 0)
    return instances, None, None


def attacc_candidate_effects(target, matches, placement):
    """`CandidateEffects` of one AttAcc placement for one work item, derived
    from the shared semantic facts of its matches (`Contraction`/`Epilogue`:
    loop labels, trips, access coordinates, matrix and vector roles,
    accumulator initialization, sinks) and from the placement's tiling.

    Backing storage follows the AttAcc datapath the command inventory
    models: every MAC matrix (weights, K, V) in the ``banks``, an external
    vector in the per-bank ``gemv_buffer`` (written from the banks by
    ``WR_GB``), every result in the PIM accumulator ``acc`` it is read back
    from; the score rows and softmax sum of an attention kernel are locals
    (``softmax_buffer``/``gemv_buffer`` allocations).  One launch per matched
    operation, in match order, names the command phase and the tiling;
    `AttAccCtx.check_realization` compares the emitted program's kernel
    extents and tiling with it.  None when the matches carry no facts.
    """
    from .spmw_candidate import (
        CandidateCoverage,
        CandidateEffects,
        CandidateLaunch,
        CandidateTransfer,
        CandidateValueLayout,
        value_axes,
    )

    del target
    facts = [m.extra.get("semantic_facts") for m in matches]
    if not matches or any(f is None for f in facts):
        return None
    extra = placement.extra or {}
    tiling = extra.get("attacc_tiling")
    kernel = extra.get("attacc_kernel")
    labels = {}
    for fact in facts:
        labels.update(fact.loop_labels)
    inputs, outputs, layouts, transfers, launches = [], [], {}, {}, []
    allocations = [("accumulator", 0, "acc")]
    orientation = attacc_orientation_of(matches, extra)
    for index, (match, fact) in enumerate(zip(matches, facts)):
        if hasattr(fact, "reduction_vars"):
            matrix = orientation.get("matrix") if len(facts) == 1 else None
            if matrix is None:
                matrix = fact.matrix.memref_name if fact.matrix is not None else None
            for access in fact.inputs:
                memref = access.memref_name
                if memref is None or access.local or memref in inputs:
                    continue
                inputs.append(memref)
                if memref == matrix:
                    storage, route = "banks", ("host_scatter", "banks")
                else:
                    storage = "gemv_buffer"
                    route = ("host_scatter", "banks", "WR_GB", "gemv_buffer")
                layouts[(memref, "value")] = CandidateValueLayout(
                    memref, value_axes(access, labels), storage
                )
                transfers[memref] = CandidateTransfer(memref, "in", route)
            results = []
            if not fact.accumulator.local:
                results.append((fact.accumulator.memref_name, fact.accumulator))
            results.extend((s.memref_name, s) for s in fact.external_sinks)
            for memref, access in results:
                if memref is None:
                    continue
                if memref not in outputs:
                    outputs.append(memref)
                layouts[(memref, "value")] = CandidateValueLayout(
                    memref, value_axes(access, labels), "acc"
                )
                transfers[memref] = CandidateTransfer(
                    memref, "out", ("acc", "host_readback")
                )
        elif any(a.local for a in fact.operands) or fact.result.local:
            # Softmax over the local score rows: softmax-unit scratch, and the
            # score buffer's own layout over the softmax loops (the kernel's
            # iteration axes that index only this local value).
            if ("scores", 0, "softmax_buffer") not in allocations:
                allocations.append(("scores", 0, "softmax_buffer"))
            for access in (*fact.operands, fact.result):
                memref = access.memref_name
                if memref is None or not access.local or access.coords in ((), None):
                    continue
                if (memref, "value") not in layouts:
                    layouts[(memref, "value")] = CandidateValueLayout(
                        memref, value_axes(access, labels), "softmax_buffer"
                    )
        instances, reduction, init = _launch_extents(fact)
        launches.append(
            CandidateLaunch(
                index=index,
                kind=f"attacc_{getattr(kernel, 'kind', 'kernel')}",
                operations=(index,),
                instances=(0, instances),
                reduction=None if reduction is None else (0, reduction),
                accumulator=init,
                covers=(placement.mode, str(tiling), match.target_op_name),
            )
        )
    scope = (
        f"one AttAcc device: {tiling.channels} channel(s) x {tiling.units} "
        "unit(s) of the selected tiling, closed-form command inventory"
        if tiling is not None
        else "one AttAcc device (tiling pending)"
    )
    return CandidateEffects(
        coverage=CandidateCoverage(
            matches[0].func_name,
            tuple(sorted({m.target_op_name for m in matches})),
            tuple(inputs),
            tuple(outputs),
        ),
        value_layouts=tuple(layouts.values()),
        transfers=tuple(transfers.values()),
        launches=tuple(launches),
        abi_inputs=tuple(inputs),
        abi_outputs=tuple(outputs),
        device_scope=scope,
        allocations=tuple(allocations),
    )


def _derive_attacc_effects(view, matches, placement):
    """Late-binding entry of the effects registry (the enumerator, the
    movement-stage re-declaration and a supplied-placement rebinding use
    the same derivation)."""
    return attacc_candidate_effects(view, matches, placement)


def _register_effects_deriver():
    from .spmw_candidate import EFFECTS_DERIVERS

    EFFECTS_DERIVERS["attacc"] = _derive_attacc_effects


_register_effects_deriver()


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

    # The command inventory covers the kernel's rows from zero; a constant
    # result offset of the one work item is not realized (X-42).
    REALIZES_RESULT_OFFSETS = False

    def __init__(self, target):
        super().__init__(target)
        self.geometry = attacc_geometry(target)
        self.programs = []
        # id(placement) -> the program emitted for it, the number of launches
        # the retained schedule issues of it, and the datapath roles of its
        # matches (matrix, vector, result values) for the realization check.
        self._program_of_layout: dict = {}
        self._launches_of_layout: dict = {}
        self._roles_of_layout: dict = {}

    def cmd(self, name, **fields):
        raise NotImplementedError(
            "AttAcc ops are lowered per kernel by AttAccCtx.emit_program"
        )

    def resolve_moves(self, role, src_handle, dst_handle):
        return (None, None)

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ):
        """One command inventory per launch of the retained schedule, in its
        order and multiplicity (a kernel launched twice is two programs and
        twice the commands; a kernel the host program never launches emits
        nothing); without a schedule, one inventory per matched kernel.
        Host transfer steps emit no commands here: the AttAcc model times
        kernels with their operands resident in the banks, and its cost
        program refuses to price host ingress."""
        from .pim.schedule_search import InfeasibleSchedule
        from .spmw_autoschedule import (
            Placement,
            _bucket_for_autoschedule,
            _matcher_work_scope,
        )

        del host_moves, module
        buckets = _bucket_for_autoschedule(trace)
        if isinstance(layouts, Placement):
            layouts = [layouts] * len(buckets)
        built = []
        for (_scope, matches), placement in zip(buckets, layouts):
            extra = getattr(placement, "extra", {}) or {}
            orientation = attacc_orientation_of(matches, extra)
            kernel = attacc_kernel_from_matches(matches, orientation)
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
            self._program_of_layout[id(placement)] = program
            self._roles_of_layout[id(placement)] = _datapath_roles(matches, orientation)
            self._launches_of_layout[id(placement)] = 0
            built.append((_matcher_work_scope(matches[0]).group_id, placement, program))
        if schedule is None:
            order = [(placement, program) for _g, placement, program in built]
        else:
            from .spmw_plan import LaunchStep

            by_group: dict = {}
            for group_id, placement, program in built:
                by_group.setdefault(group_id, []).append((placement, program))
            order = []
            for step in schedule.steps:
                if not isinstance(step, LaunchStep):
                    continue
                if step.group_id not in by_group:
                    raise InfeasibleSchedule(
                        f"AttAcc schedule launches unknown kernel group {step.group_id}"
                    )
                order.extend(by_group[step.group_id])
        for placement, program in order:
            self.programs.append(program)
            self._launches_of_layout[id(placement)] += 1
            self.cmds.extend(program.command_lines())

    def check_realization(self, layouts) -> None:
        """The emitted command inventory against the selected candidates'
        declared effects: the kernel family and tiling every launch names;
        the command phases in the kernel's order (one MAC for a GEMV; QK^T,
        the three softmax steps and PV for attention, with the score rows
        declared as softmax-unit scratch); the score/context extents the MAC
        phases advance (the kernel's rows, reduction and outputs) from zero;
        per retained role of the MAC facts, the datapath home and route the
        inventory models (the matrix scattered to the ``banks``, an external
        vector written from the banks into the ``gemv_buffer`` by ``WR_GB``,
        every result read back from the accumulator ``acc``); and one emitted
        inventory per launch the schedule issues of the candidate."""
        from .spmw_candidate import candidate_effects

        layouts = list(layouts) if isinstance(layouts, (list, tuple)) else [layouts]
        for layout in layouts:
            effects = candidate_effects(layout)
            if effects is None:
                continue
            program = self._program_of_layout.get(id(layout))
            if program is None:
                raise RuntimeError(
                    "AttAcc realization emitted no program for the candidate "
                    f"{layout.mode!r} that declares effects"
                )
            kernel, tiling = program.kernel, program.tiling
            kinds = {launch.kind for launch in effects.launches}
            if kinds != {f"attacc_{kernel.kind}"}:
                raise RuntimeError(
                    f"AttAcc realization is a {kernel.kind} command inventory; the "
                    f"candidate declares launches {sorted(kinds)}"
                )
            for launch in effects.launches:
                if len(launch.covers) < 2 or launch.covers[1] != str(tiling):
                    raise RuntimeError(
                        f"AttAcc realization uses tiling {tiling}; launch "
                        f"{launch.index} of the candidate declares {launch.covers}"
                    )
            phases = [
                launch.covers[-1] if launch.covers else None
                for launch in effects.launches
            ]
            expected_phases = {
                "gemv": ["MAC"],
                "attention": ["MAC", "SFM_EXP", "SFM_SUM", "SFM_DIV", "MAC"],
            }[kernel.kind]
            if phases != expected_phases:
                raise RuntimeError(
                    f"AttAcc {kernel.kind} issues the phases {expected_phases} in "
                    f"order; the candidate declares {phases}"
                )
            scratch = {storage for _n, _b, storage in effects.allocations}
            if ("softmax_buffer" in scratch) != (kernel.kind == "attention"):
                raise RuntimeError(
                    f"AttAcc {kernel.kind} "
                    f"{'moves' if kernel.kind == 'attention' else 'has no'} score "
                    "rows through the softmax unit; the candidate's allocations are "
                    f"{sorted(scratch)}"
                )
            macs = [
                launch
                for launch in effects.launches
                if launch.covers and launch.covers[-1] == "MAC"
            ]
            # (instances, reduction) of each MAC phase from the kernel extents:
            # a GEMV computes rows x outputs over the reduction; attention
            # scores rows x outputs(length) over reduction(d_head), then the
            # context rows x d_head over the length.
            expected = [(kernel.rows * kernel.outputs, kernel.reduction)]
            if kernel.kind == "attention":
                expected.append((kernel.rows * kernel.reduction, kernel.outputs))
            for launch, (instances, reduction) in zip(macs, expected):
                realized = (tuple(launch.instances), tuple(launch.reduction or ()))
                if realized != ((0, instances), (0, reduction)):
                    raise RuntimeError(
                        "AttAcc realization MAC phase covers (instances, reduction) "
                        f"{((0, instances), (0, reduction))}; the candidate's launch "
                        f"{launch.index} declares {realized}"
                    )
                if launch.accumulator != "zero":
                    raise RuntimeError(
                        "AttAcc command streams start every accumulator from zero; "
                        f"launch {launch.index} declares {launch.accumulator!r}"
                    )
            roles = self._roles_of_layout.get(id(layout), {})
            storage = {v.value: v.storage for v in effects.value_layouts}
            routes = {t.value: tuple(t.route) for t in effects.transfers}
            expected = {}
            for name in roles.get("matrix", ()):
                expected[name] = ("banks", ("host_scatter", "banks"))
            for name in roles.get("vector", ()):
                expected.setdefault(
                    name,
                    ("gemv_buffer", ("host_scatter", "banks", "WR_GB", "gemv_buffer")),
                )
            for name in roles.get("result", ()):
                expected[name] = ("acc", ("acc", "host_readback"))
            for name, (home, route) in expected.items():
                if storage.get(name) != home or routes.get(name) != route:
                    raise RuntimeError(
                        f"AttAcc realization keeps {name!r} in {home!r} through "
                        f"{route}; the candidate declared {storage.get(name)!r} "
                        f"through {routes.get(name)}"
                    )
            undeclared = set(effects.coverage.inputs) - set(expected)
            undeclared |= set(effects.coverage.outputs) - set(expected)
            if undeclared:
                raise RuntimeError(
                    f"AttAcc realization has no datapath role for {sorted(undeclared)}"
                )
            launched = self._launches_of_layout.get(id(layout), 0)
            emitted = sum(1 for p in self.programs if p is program)
            if emitted != launched:
                raise RuntimeError(
                    f"AttAcc realization emitted {emitted} inventories of "
                    f"{layout.mode!r} for {launched} launch(es)"
                )

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


def _datapath_roles(matches, orientation=None) -> dict:
    """``{"matrix": [...], "vector": [...], "result": [...]}``: the external
    values of the work item's MAC facts by the role the AttAcc datapath gives
    them (bank-resident matrix, GEMV-buffer vector, accumulator result).  A
    local vector (attention's probabilities) and local accumulators are
    internal to the inventory."""
    roles: dict = {"matrix": [], "vector": [], "result": []}
    oriented = (orientation or {}).get("matrix")
    single = sum(1 for m in matches if m.target_op_name == "MAC") == 1
    for match in matches:
        fact = match.extra.get("semantic_facts")
        if fact is None or not hasattr(fact, "reduction_vars"):
            continue
        matrix = oriented if (oriented is not None and single) else None
        if matrix is None:
            matrix = fact.matrix.memref_name if fact.matrix is not None else None
        for access in fact.inputs:
            if access.local or access.memref_name is None:
                continue
            bucket = "matrix" if access.memref_name == matrix else "vector"
            if access.memref_name not in roles[bucket]:
                roles[bucket].append(access.memref_name)
        results = [] if fact.accumulator.local else [fact.accumulator]
        results.extend(fact.external_sinks)
        for access in results:
            if access.memref_name and access.memref_name not in roles["result"]:
                roles["result"].append(access.memref_name)
    return roles


def _run_attacc(compiled, **inputs):
    raise NotImplementedError(ATTACC_ANALYTICAL_ONLY)


__all__ = [
    "ATTACC_ANALYTICAL_ONLY",
    "AttAccCtx",
    "attacc_base_placement",
    "attacc_candidate_effects",
    "attacc_placement",
    "attacc_tiling_candidates",
    "attacc_orientation_of",
]
