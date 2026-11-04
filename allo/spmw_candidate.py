# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common candidate-plan contract of the shared pipeline (release gate G9).

A backend whose candidates carry an intrinsic plan (the APU v1 GVML vector
plans, the Gemini-II composed-dot launch plans) states the plan's semantic
and scheduling effects in one typed record, :class:`CandidateEffects`, kept
in the placement's ``extra["effects"]``.  The shared driver *checks* those
effects, it does not merely display them:

* :func:`validate_candidate_effects` compares a candidate's coverage,
  values, transfers, launches and ABI with the semantic program the matcher
  established for the candidate's work item (every matched store's external
  destination must be covered, every value must be one of the program's
  buffers, every ABI name one of its parameters).  ``autoschedule`` runs it
  on every enumerated candidate before costing; ``compile_for_target`` runs
  it again on the selected placements.
* :func:`rebind_supplied_layouts` is the boundary for a caller-supplied
  ``layout=`` placement: a placement is a *decision*, not an executable, so
  it is rebound to the candidate of the current program that carries the
  same physical decision (or, for a backend plan, the same plan name) after
  the current enumeration applied the target's legality and the current
  reduction-order policy.  A placement that designates no candidate of the
  current program is refused; a plan the policy pruned raises
  ``ReductionOrderError``.  The payload of a supplied placement is never
  executed as such.
* After materialization the backend context's ``check_realization``
  compares the realized program (its ABI, launches) with the effects the
  selected candidate declared.

The matcher backends (Samsung, AiM, UPMEM, AttAcc) express their candidates
as storage handles checked by ``spmw_legality``; they declare no effects
yet, so the effects validation is a no-op for them (recorded in ISSUES
X-01 as remaining migration work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateCoverage:
    """Which part of the semantic program the candidate realizes."""

    function: str
    operations: tuple[str, ...]  # matched target ops it implements
    inputs: tuple[str, ...]  # buffers it reads (memref names)
    outputs: tuple[str, ...]  # external destinations it writes (memref names)


@dataclass(frozen=True)
class CandidateValueLayout:
    """Where and how one program buffer lives on the device.

    ``axes`` are the program's loop labels (plus declared synthetic axes)
    the value is laid out over; ``storage`` is a declared storage of the
    target (a handle name or a register family prefix); ``packing`` names a
    backend packing form when the layout is a host-packed carrier.
    """

    value: str
    axes: tuple[str, ...]
    storage: str | None = None
    packing: str | None = None
    # Which image of the value this layout describes when one semantic buffer
    # has distinct images in the program: ``"value"`` (the only one),
    # ``"in"`` (the contents the program reads) or ``"out"`` (the contents it
    # writes; an operand that is also a copy sink has both).
    image: str = "value"


@dataclass(frozen=True)
class CandidateTransfer:
    value: str
    direction: str  # "in" | "out" | "inout"
    route: tuple[str, ...]
    # A physical copy the runtime performs of a declared ABI buffer the
    # program's semantics neither read nor write (an unused argument a
    # scalar program's runtime still sends); never a semantic input.
    physical: bool = False


@dataclass(frozen=True)
class CandidateLaunch:
    """One device launch of the plan and the instances it computes.

    ``operations`` are the indices of the work item's matched operations
    (contractions in match order) the launch advances; ``instances`` is the
    half-open range of result instances (flattened result coordinates) it
    covers; ``reduction`` the half-open range of the reduction axis it sums
    (``None`` for a non-reducing operation); ``accumulator`` how it starts
    (``"zero"``, ``"carried"``, ``"none"``, ``"previous"``).  Launches are
    indexed 0..n-1 in execution order.
    """

    index: int
    kind: str
    operations: tuple[int, ...]
    instances: tuple[int, int]
    reduction: tuple[int, int] | None = None
    accumulator: str | None = None
    covers: tuple = ()  # backend detail (plan name, tile coordinates)
    # A whole-function launch: the device executes the function verbatim,
    # advancing every covered store effect over its complete iteration
    # space in program order (``instances`` is ``(0, 1)``: one execution;
    # ``reduction`` and ``accumulator`` are None, the function's own
    # control flow carries its accumulators).
    whole: bool = False


@dataclass(frozen=True)
class CandidateEffects:
    coverage: CandidateCoverage
    value_layouts: tuple[CandidateValueLayout, ...]
    transfers: tuple[CandidateTransfer, ...]
    launches: tuple[CandidateLaunch, ...]
    abi_inputs: tuple[str, ...]
    abi_outputs: tuple[str, ...]
    device_scope: str
    # Axes the plan adds to the program's loops (a unit axis a 1-D GEMV is
    # lifted with), with their extents.
    synthetic_axes: tuple[tuple[str, int], ...] = ()
    # Device allocations the program needs beyond its values:
    # ``(name, bytes, storage)`` (a scalar program's local-array scratch).
    allocations: tuple[tuple[str, int, str], ...] = ()
    # A declared host partition: the backend executes the work item's whole
    # function verbatim on the host from its retained IR (UPMEM's paper
    # partition of the k-means update).  Its launches are whole-function
    # launches checked against the function's retained store effects, the
    # way a verbatim device program's are; its value layouts name host
    # storage.  Never a device program's shortcut around tiling.
    host_function: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "coverage": {
                "function": self.coverage.function,
                "operations": list(self.coverage.operations),
                "inputs": list(self.coverage.inputs),
                "outputs": list(self.coverage.outputs),
            },
            "value_layouts": [
                {
                    "value": v.value,
                    "axes": list(v.axes),
                    "storage": v.storage,
                    "packing": v.packing,
                    "image": v.image,
                }
                for v in self.value_layouts
            ],
            "transfers": [
                {
                    "value": t.value,
                    "direction": t.direction,
                    "route": list(t.route),
                    "physical": t.physical,
                }
                for t in self.transfers
            ],
            "launches": [
                {
                    "index": l.index,
                    "kind": l.kind,
                    "operations": list(l.operations),
                    "instances": list(l.instances),
                    "reduction": None if l.reduction is None else list(l.reduction),
                    "accumulator": l.accumulator,
                    "covers": list(l.covers),
                    "whole": l.whole,
                }
                for l in self.launches
            ],
            "abi": {"inputs": list(self.abi_inputs), "outputs": list(self.abi_outputs)},
            "synthetic_axes": [list(item) for item in self.synthetic_axes],
            "allocations": [list(item) for item in self.allocations],
            "device_scope": self.device_scope,
            "host_function": self.host_function,
        }


def candidate_effects(placement) -> CandidateEffects | None:
    extra = getattr(placement, "extra", None) or {}
    effects = extra.get("effects")
    return effects if isinstance(effects, CandidateEffects) else None


def _verbatim_facts(matches, host_function: bool = False):
    """The `ProgramFacts` of a work item whose program executes the
    function verbatim (``facts.verbatim``, or a candidate's declared host
    partition ``host_function`` over facts with retained store effects),
    else None."""
    facts = matches[0].extra.get("program_facts") if matches else None
    if facts is None:
        return None
    if getattr(facts, "verbatim", False):
        return facts
    if host_function and getattr(facts, "effects", ()):
        return facts
    return None


def value_axes(access, labels) -> tuple:
    """Source loop labels the access coordinates range over, in coordinate
    order (an affine coordinate ``row0 + i`` ranges over ``i``); the shared
    derivation of a `CandidateValueLayout`'s axes from a retained access."""
    axes = []
    for coord in getattr(access, "coords", None) or ():
        if coord is None:
            continue
        for atom, _coefficient in coord[0]:
            label = labels.get(atom)
            if label is not None and label not in axes:
                axes.append(label)
    return tuple(axes)


def verbatim_values(facts):
    """``(inputs, outputs, locals)`` of a function a program executes
    verbatim, from its retained store effects (`ProgramFacts.effects`):

    * an argument is an input when any store's value, guard or index reads
      it (through local temporaries too), when a store reads the element
      it writes in it (a carried value), or when the function does not
      overwrite it whole and unconditionally (the elements it leaves alone
      keep the caller's values);
    * every argument or returned buffer the function stores to is an
      output; internal temporaries are the locals.

    Inputs and outputs are in parameter order.
    """
    arguments = [p.name for p in facts.parameters if not p.returned]
    reads, writes, locals_ = set(), set(), set()
    for effect in facts.effects:
        destination = effect.destination
        if destination.local:
            if destination.memref_name is not None:
                locals_.add(destination.memref_name)
        else:
            writes.add(destination.parameter)
            if destination.parameter in arguments and (
                effect.reads_destination or not effect.writes_whole_buffer
            ):
                reads.add(destination.parameter)
        for access in effect.reads:
            if not access.local and access.parameter in arguments:
                reads.add(access.parameter)
    order = list(facts.parameter_names)
    return (
        tuple(name for name in order if name in reads),
        tuple(name for name in order if name in writes),
        frozenset(locals_),
    )


def _verbatim_values(facts, matches):
    inputs, outputs, locals_ = verbatim_values(facts)
    operations = {match.target_op_name for match in matches}
    return (
        set(inputs),
        set(outputs),
        set(locals_),
        operations,
        set(facts.parameter_names),
    )


def indirect_reads(matches) -> dict:
    """Buffers the work item's matched stores read outside their operands:
    through a computed index (a histogram's bin ``H[(x * m) >> k]`` reads
    ``x``) or a guard, as the retained store effects record them
    (`ProgramFacts.effects`, category ``matched``).  ``{memref name:
    BufferAccess}`` for the external buffers that are no match's operand or
    result; the semantic inputs the operand view misses.
    """
    facts = matches[0].extra.get("program_facts") if matches else None
    if facts is None or not getattr(facts, "effects", ()):
        return {}
    functions = {match.func_name for match in matches}
    known = set()
    for match in matches:
        known.update(o.memref_name for o in match.operands if o.memref_name)
        if match.result_memref_name:
            known.add(match.result_memref_name)
    out = {}
    for effect in facts.effects:
        if effect.category != "matched" or effect.function not in functions:
            continue
        for access in effect.reads:
            name = access.memref_name
            if name is None or access.local or name in known or name in out:
                continue
            out[name] = access
    return out


def _program_values(matches, host_function: bool = False):
    """The buffers of one work item's matches: inputs, external
    destinations, local destinations, operations, parameter names.

    A destination the match accumulates into is an input only when the
    source carries its incoming contents (``accumulator_init == "carried"``):
    with a proven zero initializer (``"zero"``: the matcher found every
    other store into it a constant 0 before the match, on the same element
    or a whole-buffer fill) the caller's contents are dead, whether the
    accumulating operand is the pattern's loop-carried parameter (``MAC``)
    or an elementwise operand reading its own element (``P[d] += A[i]``
    matched as ``ADD``).
    """

    facts = _verbatim_facts(matches, host_function)
    if facts is not None:
        return _verbatim_values(facts, matches)
    inputs, outputs, locals_, operations = set(), set(), set(), set()
    for match in matches:
        operations.add(match.target_op_name)
        init = match.extra.get("accumulator_init")
        for operand in match.operands:
            if operand.memref_name is None:
                continue
            if operand.memref_name == match.result_memref_name and (
                operand.is_loop_carried or init == "zero"
            ):
                # The destination itself: read only when the source carries
                # its incoming contents into the reduction (``y[i] += ...``
                # without an initializer).
                if init == "carried" and not match.extra.get("result_is_local"):
                    inputs.add(operand.memref_name)
                continue
            if operand.is_loop_carried:
                # An accumulator seeded from a buffer other than the result
                # destination (a record's ``accumulator`` input) is read.
                inputs.add(operand.memref_name)
                continue
            inputs.add(operand.memref_name)
        if match.result_memref_name is not None:
            if match.extra.get("result_is_local"):
                locals_.add(match.result_memref_name)
            else:
                outputs.add(match.result_memref_name)
        sinks = match.extra.get("output_sinks", ())
        refs = match.extra.get("output_sink_refs", ())
        for position, (memref, _text) in enumerate(sinks):
            local = refs[position][1] if position < len(refs) else None
            if local:
                locals_.add(memref)
            elif memref != match.result_memref_name or match.extra.get(
                "result_is_local"
            ):
                outputs.add(memref)
    facts = matches[0].extra.get("program_facts") if matches else None
    parameters = set(getattr(facts, "parameter_names", ()) or ())
    # Buffers read only through computed indices or guards are inputs too.
    inputs.update(indirect_reads(matches))
    # Accumulator buffers that are outputs (carried accumulators) are also
    # read; the loop-carried operand is not an input of the program.
    inputs -= locals_
    return inputs, outputs, locals_, operations, parameters


def _semantic_loop_labels(matches) -> set:
    """Loop labels of every match's shared facts (contractions and
    epilogues alike)."""
    labels = set()
    for match in matches:
        facts = match.extra.get("semantic_facts")
        for loop in getattr(facts, "loops", ()) or ():
            labels.add(loop.label)
    return labels


def _storage_declared(view, storage) -> bool:
    if view is None or storage is None:
        return storage is not None
    handles = getattr(view, "_handles", None)
    names = set(handles) if isinstance(handles, dict) else set()
    for unit in getattr(view, "_walk", lambda: ())():
        names.add(getattr(unit, "name", None))
    if storage in names:
        return True
    # A register family (``vr`` for ``vr0``..``vr14``).
    return any(
        name is not None and name.startswith(storage) and name[len(storage) :].isdigit()
        for name in names
    )


def _tiles(ranges, extent) -> bool:
    """``ranges`` (half-open) partition ``[0, extent)`` exactly."""
    spans = sorted(set((int(a), int(b)) for a, b in ranges))
    if not spans or spans[0][0] != 0 or spans[-1][1] != extent:
        return False
    for (_a0, b0), (a1, _b1) in zip(spans, spans[1:]):
        if a1 != b0:
            return False
    return all(a < b for a, b in spans)


def _product(values) -> int | None:
    total = 1
    for value in values:
        if value is None:
            return None
        total *= int(value)
    return total


def _operation_extents(matches, host_function: bool = False) -> dict:
    """Per matched operation (original match index): ``(instances,
    reduction_extent, accumulator_init)``.  A contraction's instances are
    its result (output and batch) trips and its reduction extent the
    reduction trips; an epilogue or other elementwise operation has the
    product of its loop trips and no reduction; a whole-function record
    states its instance count in ``extra["result_instances"]``.  ``None``
    when the extent is unknown (the launches cannot be checked, which is a
    contract problem)."""
    out = {}
    verbatim = _verbatim_facts(matches, host_function)
    if verbatim is not None:
        for index, effect in enumerate(verbatim.effects):
            out[index] = (effect.instances, effect.reduction_extent, None)
        return out
    for index, match in enumerate(matches):
        facts = match.extra.get("semantic_facts")
        if facts is not None and hasattr(facts, "reduction_vars"):
            trips = facts.trips
            out[index] = (
                _product(trips.get(v) for v in facts.result_vars),
                _product(trips.get(v) for v in facts.reduction_vars),
                facts.accumulator_init,
            )
        elif facts is not None and hasattr(facts, "loop_vars"):
            out[index] = (
                _product(loop.trip for loop in facts.loops),
                None,
                None,
            )
        else:
            declared = match.extra.get("result_instances")
            out[index] = (
                (int(declared), None, match.extra.get("accumulator_init"))
                if declared
                else None
            )
    return out


def _ordered_partition(ranges, extent) -> bool:
    """``ranges`` in launch order are exactly ``[0, a1), [a1, a2), ...,
    [ak, extent)``: contiguous, ascending, without repetition."""
    spans = [(int(a), int(b)) for a, b in ranges]
    if not spans or spans[0][0] != 0 or spans[-1][1] != extent:
        return False
    for (_a0, b0), (a1, _b1) in zip(spans, spans[1:]):
        if a1 != b0:
            return False
    return all(a < b for a, b in spans)


_FIRST_ACCUMULATOR = {"zero": {"zero", "none"}, "carried": {"carried"}}


def _check_operation_launches(op, launches, extent) -> list[str]:
    """Problems of the launches advancing operation ``op`` against its
    extents ``(instances, reduction_extent, accumulator_init)``."""
    if extent is None or extent[0] is None:
        return [f"operation {op} has no known instance extent to check launches"]
    instances, reduction_extent, init = extent
    problems = []
    chains = {}
    for launch in launches:
        chains.setdefault(tuple(int(v) for v in launch.instances), []).append(launch)
    if instances and not _tiles(chains.keys(), instances):
        problems.append(
            f"the launches of operation {op} cover instances "
            f"{sorted(chains)}, not a tiling of [0, {instances})"
        )
    for instance_range, chain in chains.items():
        if reduction_extent is None:
            # A non-reducing operation is computed once per instance.
            if len(chain) != 1:
                problems.append(
                    f"operation {op} over instances {instance_range} is advanced "
                    f"by {len(chain)} launches, not one"
                )
            continue
        ranges = [launch.reduction for launch in chain]
        if any(r is None for r in ranges):
            if reduction_extent > 1:
                problems.append(
                    f"a launch of operation {op} declares no reduction range "
                    f"for a reduction of {reduction_extent}"
                )
            continue
        if reduction_extent and not _ordered_partition(ranges, reduction_extent):
            problems.append(
                f"the launches of operation {op} over instances {instance_range} "
                f"sum {ranges} in order, not an ordered partition of "
                f"[0, {reduction_extent}) (a repeated or out-of-order chunk "
                "would add a partial sum twice)"
            )
        allowed = _FIRST_ACCUMULATOR.get(init)
        if allowed is None:
            problems.append(
                f"operation {op} has accumulator initialization {init!r}, which "
                "the candidate contract cannot chain launches for"
            )
            continue
        first = chain[0]
        if first.accumulator not in allowed:
            problems.append(
                f"launch {first.index} of operation {op} starts its accumulator as "
                f"{first.accumulator!r}; the source initializes it as {init!r}"
            )
        for later in chain[1:]:
            if later.accumulator != "previous":
                problems.append(
                    f"launch {later.index} of operation {op} continues the "
                    f"reduction with accumulator {later.accumulator!r}, not "
                    "'previous' (the preceding launch's partial sum)"
                )
    return problems


def _check_value_layouts(effects, matches, inputs, outputs, locals_, view) -> list:
    """Every input and output laid out over program axes (or declared
    synthetic axes) on declared storage; every program loop covered."""
    problems = []
    loop_labels = _semantic_loop_labels(matches)
    synthetic = {name for name, _extent in effects.synthetic_axes}
    for name, extent in effects.synthetic_axes:
        if name in loop_labels or int(extent) < 1:
            problems.append(f"synthetic axis {name!r} is not a fresh positive axis")
    laid_out = set()
    covered_axes = set()
    for layout in effects.value_layouts:
        if layout.value not in inputs | outputs | locals_:
            problems.append(f"lays out {layout.value!r}, not a program buffer")
            continue
        if layout.image not in {"value", "in", "out"}:
            problems.append(f"layout of {layout.value!r} names image {layout.image!r}")
        laid_out.add(layout.value)
        if len(set(layout.axes)) != len(layout.axes):
            problems.append(f"layout of {layout.value!r} repeats an axis")
        for axis in layout.axes:
            if axis not in loop_labels and axis not in synthetic:
                problems.append(
                    f"layout of {layout.value!r} uses axis {axis!r}, which is not "
                    f"a loop of the program ({sorted(loop_labels)}) nor declared"
                )
        covered_axes.update(layout.axes)
        if not _storage_declared(view, layout.storage):
            problems.append(
                f"layout of {layout.value!r} names storage {layout.storage!r}, "
                "which the target does not declare"
            )
    missing_layouts = (inputs | outputs) - laid_out
    if missing_layouts:
        problems.append(f"no value layout for {sorted(missing_layouts)}")
    if (
        loop_labels
        and not loop_labels <= covered_axes
        and _verbatim_facts(matches, effects.host_function) is None
    ):
        # A verbatim program keeps its buffers whole in device memory and
        # runs the function's loops itself; distributed layouts over the
        # program's loops are the contract of a planned program.
        problems.append(
            f"loops {sorted(loop_labels - covered_axes)} of the program appear in "
            "no value layout (iteration coverage)"
        )
    return problems


# Host-side steps a transfer route may name on any target.
HOST_HOOKS = frozenset(
    {
        "host_copy",
        "host_readback",
        "host_scatter",
        "host_broadcast",
        "host_gather",
        "host_pack",
        "device_readback",
    }
)
# target name -> further route steps a backend's runtime performs that are
# neither a declared storage nor a declared move (registered by the backend).
ROUTE_VOCABULARY: dict = {}


def _route_step_declared(view, step) -> bool:
    if step in HOST_HOOKS or step in ROUTE_VOCABULARY.get(
        getattr(view, "name", None), ()
    ):
        return True
    if _storage_declared(view, step):
        return True
    moves = set()
    for unit in getattr(view, "_walk", lambda: ())():
        moves.update(str(name).lower() for name in getattr(unit, "moves", {}) or ())
    return str(step).lower() in moves


def _check_transfers(effects, inputs, outputs, locals_, parameters, view) -> list:
    """Every input transferred in, every output transferred out, over routes
    whose every step is a declared storage, a declared move, a host hook or
    a step the backend registered; a physical transfer names a declared
    parameter the program neither reads nor writes and never stands in for
    a semantic one."""
    problems = []
    moved_in, moved_out = set(), set()
    for transfer in effects.transfers:
        for step in transfer.route:
            if view is not None and not _route_step_declared(view, step):
                problems.append(
                    f"transfer of {transfer.value!r} names route step {step!r}, "
                    "which is neither a declared storage nor a declared move of the "
                    "target nor a host step"
                )
        if transfer.physical:
            if transfer.value not in parameters:
                problems.append(
                    f"physical transfer of {transfer.value!r}, not a parameter"
                )
            if transfer.value in inputs | outputs:
                problems.append(
                    f"physical transfer of {transfer.value!r}, which the program "
                    "reads or writes (declare the semantic transfer)"
                )
            continue
        if transfer.value not in inputs | outputs | locals_:
            problems.append(f"transfers {transfer.value!r}, not a program buffer")
        if transfer.direction not in {"in", "out", "inout"}:
            problems.append(f"transfer of {transfer.value!r} has no direction")
        if not transfer.route:
            problems.append(f"transfer of {transfer.value!r} has no route")
        if transfer.direction in {"in", "inout"}:
            moved_in.add(transfer.value)
        if transfer.direction in {"out", "inout"}:
            moved_out.add(transfer.value)
    if not inputs <= moved_in:
        problems.append(f"inputs {sorted(inputs - moved_in)} are never transferred in")
    if not outputs <= moved_out:
        problems.append(
            f"outputs {sorted(outputs - moved_out)} are never transferred out"
        )
    return problems


def _check_launches(effects, matches, view) -> list:
    """Launches: ordered, each covering declared operations; per operation the
    instance ranges tile its result instances, and per instance range the
    launches form an ordered partition of the reduction whose accumulator
    chain starts as the source initializes it and continues from the
    previous launch; a verbatim program launches whole functions.  Declared
    allocations name declared storage."""
    problems = []
    verbatim = _verbatim_facts(matches, effects.host_function)
    operation_count = len(verbatim.effects) if verbatim is not None else len(matches)
    if effects.host_function and verbatim is None:
        problems.append(
            "declares a host partition (host_function) for a work item without "
            "retained store effects to check its whole-function launches against"
        )
    if not effects.launches:
        problems.append("declares no launch")
    for position, launch in enumerate(effects.launches):
        if launch.index != position:
            problems.append(f"launch {launch.index} is out of order at {position}")
        if not launch.kind:
            problems.append(f"launch {position} has no kind")
        if not launch.operations:
            problems.append(f"launch {position} advances no operation")
        for op in launch.operations:
            if not 0 <= op < operation_count:
                problems.append(f"launch {position} names operation {op}, not matched")
        a, b = launch.instances
        if not 0 <= a < b:
            problems.append(f"launch {position} covers no instances")
        if launch.whole and (
            tuple(launch.instances) != (0, 1)
            or launch.reduction is not None
            or launch.accumulator is not None
        ):
            problems.append(
                f"whole-function launch {position} declares instances, a reduction "
                "range or an accumulator mode; the function's own control flow "
                "carries those"
            )
        if verbatim is not None and not launch.whole:
            problems.append(
                f"launch {position} advances part of a function the program "
                "executes verbatim; a verbatim program launches whole functions"
            )
        if verbatim is None and launch.whole:
            problems.append(
                f"launch {position} is declared whole-function, but the program is "
                "not executed verbatim: its launches tile instances and reductions"
            )
    by_operation = {}
    for launch in effects.launches:
        for op in launch.operations:
            if 0 <= op < operation_count:
                by_operation.setdefault(op, []).append(launch)
    extents = _operation_extents(matches, effects.host_function)
    for op, launches in by_operation.items():
        if all(launch.whole for launch in launches):
            if len(launches) != 1:
                problems.append(
                    f"operation {op} is advanced by {len(launches)} whole-function "
                    "launches, not one"
                )
            extent = extents.get(op)
            if extent is None or extent[0] is None:
                problems.append(
                    f"operation {op} has no known instance extent to check launches"
                )
            continue
        problems.extend(_check_operation_launches(op, launches, extents.get(op)))
    uncovered = set(range(operation_count)) - set(by_operation)
    if uncovered:
        problems.append(f"operations {sorted(uncovered)} are advanced by no launch")
    for name, size, storage in effects.allocations:
        if not name or int(size) < 0:
            problems.append(f"allocation {name!r} of {size} bytes is malformed")
        if not _storage_declared(view, storage):
            problems.append(
                f"allocation {name!r} names storage {storage!r}, which the target "
                "does not declare"
            )

    return problems


# Targets whose enumerators declare `CandidateEffects` for every candidate of
# a program that went through the shared front stages (its work items carry
# ``extra["program_facts"]``).  A handle-only placement is a contract problem
# there.  Every released backend is listed: a new backend joins the list
# with its effects deriver (`EFFECTS_DERIVERS`) and realization check.
EFFECTS_REQUIRED_TARGETS = frozenset(
    {"apu_v1", "apu_v2", "samsung_hbm_pim", "aim", "upmem", "attacc"}
)


def _effects_required(matches, view) -> bool:
    if getattr(view, "name", None) not in EFFECTS_REQUIRED_TARGETS:
        return False
    return bool(matches) and matches[0].extra.get("program_facts") is not None


def validate_candidate_effects(matches, placement, view=None) -> list[str]:
    """Problems of a candidate's declared effects against the semantic
    program of ``matches`` (and the storage declarations of ``view`` when
    given); empty when consistent or when the candidate declares no effects
    (a handle-only matcher placement).

    Checked: coverage (function, operations, inputs and outputs exactly the
    program's); value layouts for every input and output over the program's
    loop labels and declared synthetic axes, on declared storage, together
    covering every loop of every contraction; a transfer in for every input
    and out for every output; launches indexed 0..n-1 whose instance ranges
    tile every covered operation's result instances (contractions,
    epilogues and whole-function records alike) and whose reduction ranges
    form, per instance range and in launch order, a partition of the
    reduction extent with an accumulator chain that starts as the source
    initializes the accumulator and continues from the previous launch;
    ABI equal to the coverage.
    """

    effects = candidate_effects(placement)
    if effects is None:
        if _effects_required(matches, view):
            return [
                "declares no effects; every candidate of target "
                f"{getattr(view, 'name', None)!r} for a program with shared facts "
                "states its coverage, layouts, transfers, launches and ABI"
            ]
        return []
    inputs, outputs, locals_, operations, parameters = _program_values(
        matches, effects.host_function
    )
    functions = {match.func_name for match in matches}
    problems = []
    coverage = effects.coverage
    if coverage.function not in functions:
        problems.append(
            f"covers function {coverage.function!r}, not {sorted(functions)}"
        )
    if set(coverage.operations) != operations:
        problems.append(
            f"covers operations {sorted(coverage.operations)}, the work item "
            f"matches {sorted(operations)}"
        )
    if set(coverage.outputs) != outputs:
        problems.append(
            f"writes {sorted(coverage.outputs)}, the program's external "
            f"destinations are {sorted(outputs)}"
        )
    if set(coverage.inputs) != inputs:
        problems.append(
            f"reads {sorted(coverage.inputs)}, the program reads {sorted(inputs)}"
        )

    problems.extend(
        _check_value_layouts(effects, matches, inputs, outputs, locals_, view)
    )
    problems.extend(
        _check_transfers(effects, inputs, outputs, locals_, parameters, view)
    )

    problems.extend(_check_launches(effects, matches, view))

    # ABI.
    if set(effects.abi_outputs) != set(coverage.outputs):
        problems.append(
            f"ABI outputs {sorted(effects.abi_outputs)} differ from the coverage "
            f"outputs {sorted(coverage.outputs)}"
        )
    if set(effects.abi_inputs) != set(coverage.inputs):
        problems.append(
            f"ABI inputs {sorted(effects.abi_inputs)} differ from the coverage "
            f"inputs {sorted(coverage.inputs)}"
        )
    if parameters and not set(effects.abi_inputs) | set(effects.abi_outputs) <= (
        parameters
    ):
        problems.append(
            "ABI names "
            f"{sorted((set(effects.abi_inputs) | set(effects.abi_outputs)) - parameters)}"
            " are not parameters of the program"
        )
    return problems


def require_candidate_effects(matches, placement, what: str, view=None) -> None:
    problems = validate_candidate_effects(matches, placement, view)
    if problems:
        raise ValueError(
            f"{what}: candidate {getattr(placement, 'mode', '')!r} violates the "
            "candidate contract: " + "; ".join(problems)
        )


def check_graph_consistency(placement, graph) -> None:
    """The priced graph must describe the candidate's declared launches: a
    graph that states its launch count or plan must agree with the effects."""
    effects = candidate_effects(placement)
    if effects is None or graph is None:
        return
    metadata = getattr(graph, "metadata", {}) or {}
    count = metadata.get("launch_count")
    if count is not None and int(count) != len(effects.launches):
        raise ValueError(
            f"candidate {placement.mode!r}: the cost graph prices {count} launch(es) "
            f"but the candidate declares {len(effects.launches)}"
        )
    plan = metadata.get("plan")
    declared = {launch.covers[0] for launch in effects.launches if launch.covers}
    if plan is not None and declared and plan not in declared:
        raise ValueError(
            f"candidate {placement.mode!r}: the cost graph prices plan {plan!r} but "
            f"the candidate's launches declare {sorted(declared)}"
        )


# target name -> ``fn(view, matches, placement) -> CandidateEffects | None``:
# how a migrated matcher backend declares a placement's effects, used to
# re-declare a caller-supplied handle-only placement for the work item it is
# compiled for (the enumerator declares them for its own candidates).
EFFECTS_DERIVERS: dict = {}


def _redeclared(view, matches, placement):
    deriver = EFFECTS_DERIVERS.get(getattr(view, "name", None))
    if deriver is None:
        return placement
    effects = deriver(view, matches, placement)
    if effects is None or candidate_effects(placement) == effects:
        return placement
    from .spmw_autoschedule import Placement

    return Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={**(placement.extra or {}), "effects": effects},
        layout=getattr(placement, "layout", None),
    )


def _with_current_effects(placement, current):
    """``placement`` with the effects the current enumeration declared for
    the same decision (per work item), the caller's other fields kept."""
    effects = candidate_effects(current)
    if effects is None or candidate_effects(placement) is effects:
        return placement
    from .spmw_autoschedule import Placement

    return Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={**(placement.extra or {}), "effects": effects},
        layout=getattr(placement, "layout", None),
    )


def rebind_supplied_layouts(view, trace, supplied) -> list:
    """Bind caller-supplied placements to candidates of the current program.

    ``supplied`` is aligned with ``_bucket_for_autoschedule(trace)``.  For
    each work item the current enumeration (legality and reduction-order
    policy applied) is searched for the candidate with the supplied
    placement's physical decision: a backend plan is replaced by that
    current candidate (a fresh payload bound to this program), a handle-only
    placement is kept as supplied (its decision is current).  A backend plan
    without such a candidate is looked up by plan name (``mode``) or
    refused; a handle-only placement without one keeps the caller's
    placement (its legality was checked) unless its mode was pruned by the
    reduction-order policy, which is refused.  Returns the placements to
    compile.
    """

    from .spmw_autoschedule import (
        _ENUMERATORS,
        _bucket_for_autoschedule,
        _matcher_operand_roles,
        _matcher_placement_decision,
        _matcher_work_scope,
        _stamp_group_replication,
        select_candidates,
    )
    from .spmw_legality import check_placement
    from .spmw_match import MatchTrace
    from .spmw_reduction_policy import ReductionOrderError

    enumerator = _ENUMERATORS.get(getattr(view, "name", None))
    buckets = _bucket_for_autoschedule(trace)
    if enumerator is None or len(supplied) != len(buckets):
        return list(supplied)
    # Enumerators may read whole-group facts the ranking stamps first.
    groups = {}
    for _scope, matches in buckets:
        groups.setdefault(_matcher_work_scope(matches[0]).group_id, []).extend(matches)
    for group_matches in groups.values():
        _stamp_group_replication(group_matches)
    rebound = []
    for placement, (_scope, matches) in zip(supplied, buckets):
        candidates = enumerator(view, matches)
        legal = [c for c in candidates if check_placement(view, matches, c).legal]
        bucket_trace = MatchTrace(trace.target_name, trace.module_name, list(matches))
        roles = _matcher_operand_roles(bucket_trace, [*legal, placement])
        wanted = _matcher_placement_decision(bucket_trace, placement, roles)
        same = [
            c
            for c in legal
            if _matcher_placement_decision(bucket_trace, c, roles) == wanted
        ]
        payload = (getattr(placement, "extra", None) or {}).get("payload")
        if same:
            if payload is None:
                # A handle-only placement with a current candidate's decision:
                # the caller's placement (with the fields its own ranking
                # derived) is the program; only its decision was re-checked.
                # Its declared effects are this work item's (a placement
                # supplied for every work item of a lockstep kernel carries
                # the first body's declaration otherwise).
                rebound.append(_with_current_effects(placement, same[0]))
                continue
            chosen = same[0]
            chosen.extra = {**chosen.extra, "candidate_selection": "forced"}
            rebound.append(chosen)
            continue
        if payload is not None:
            # A backend plan from another compilation (or edited): rebind by
            # plan name to this program's plan, or refuse.
            named = [
                c
                for c in select_candidates(legal, placement.mode, matches)
                if not c.extra.get("candidate_excluded")
            ]
            if len(named) != 1:
                raise ValueError(
                    f"layout placement {placement.mode!r} designates "
                    f"{len(named)} candidates of kernel {matches[0].func_name!r}"
                )
            rebound.append(named[0])
            continue
        pruned = {}
        for candidate in candidates:
            for name, reason in candidate.extra.get("reduction_policy_pruned", ()):
                pruned.setdefault(name, reason)
        if placement.mode in pruned:
            raise ReductionOrderError(
                f"placement {placement.mode!r} was pruned by the reduction-order "
                f"policy ({pruned[placement.mode]})"
            )
        # The caller's placement for this work item, its effects declared by
        # the backend for these matches (a supplied placement may carry the
        # declaration of another body of a lockstep kernel).
        rebound.append(_redeclared(view, matches, placement))
    return rebound


__all__ = [
    "EFFECTS_DERIVERS",
    "EFFECTS_REQUIRED_TARGETS",
    "HOST_HOOKS",
    "ROUTE_VOCABULARY",
    "CandidateCoverage",
    "CandidateEffects",
    "CandidateLaunch",
    "CandidateTransfer",
    "CandidateValueLayout",
    "candidate_effects",
    "check_graph_consistency",
    "indirect_reads",
    "rebind_supplied_layouts",
    "require_candidate_effects",
    "validate_candidate_effects",
    "value_axes",
    "verbatim_values",
]
