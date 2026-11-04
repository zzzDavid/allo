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
  executed as such.  A supplied placement's *declared effects* are a claim
  about the program: a declaration covering a function of the compiled
  program is normalized onto the work item it is supplied for (function and
  value bindings by operand role, so one body's declaration serves every
  body of a lockstep kernel), validated in full and compared field by field
  with the current candidate's declaration for the same decision; a
  disagreement refuses the compilation before anything is replaced.  A
  declaration covering a function outside the program is another program's
  plan: its decision is reused and its declaration re-derived (recorded in
  the placement's ``candidate_selection``).  A placement without a
  declaration supplies its decision only.
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
    # One layout per value and image: a second record for the same key is a
    # contradiction (or a duplicate), never a quieter alternative.
    seen_layouts = {}
    for layout in effects.value_layouts:
        key = (layout.value, layout.image)
        if key in seen_layouts:
            problems.append(
                f"declares two layouts of {layout.value!r} (image {layout.image!r}): "
                f"storage {seen_layouts[key].storage!r} packing "
                f"{seen_layouts[key].packing!r} and storage {layout.storage!r} packing "
                f"{layout.packing!r}"
            )
            continue
        seen_layouts[key] = layout
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
    # One transfer per value and direction: a second record for the same key
    # is a contradiction (or a duplicate).
    seen_transfers = {}
    for transfer in effects.transfers:
        key = (transfer.value, transfer.direction)
        if key in seen_transfers:
            problems.append(
                f"declares two transfers of {transfer.value!r} {transfer.direction}: "
                f"{tuple(seen_transfers[key].route)} and {tuple(transfer.route)}"
            )
            continue
        seen_transfers[key] = transfer
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
    problems.extend(_derivation_problems(view, matches, placement, effects))
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
    """Attach the deriver's effects to a placement that declares none (a
    caller-supplied handle-only placement).  A placement that already
    declares effects keeps them: a declaration that differs from the
    derivation is a contract violation for ``validate_candidate_effects``
    to report, not something to overwrite silently."""
    deriver = EFFECTS_DERIVERS.get(getattr(view, "name", None))
    if deriver is None or candidate_effects(placement) is not None:
        return placement
    effects = deriver(view, matches, placement)
    if effects is None:
        return placement
    from .spmw_autoschedule import Placement

    return Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={**(placement.extra or {}), "effects": effects},
        layout=getattr(placement, "layout", None),
    )


def _derivation_problems(view, matches, placement, effects) -> list:
    """The declared value layouts (storage and packing) and transfers must be
    the ones the target's deriver states for this placement: a declaration
    edited after enumeration (a different packing, a different map) is
    rejected before costing and selection."""
    deriver = EFFECTS_DERIVERS.get(getattr(view, "name", None))
    if deriver is None:
        return []
    try:
        derived = deriver(view, matches, placement)
    except Exception as error:  # the deriver's own rejection is a problem too
        return [f"the target's effects derivation rejects the placement: {error}"]
    if derived is None:
        return []
    problems = []

    def key(layout):
        return (layout.value, getattr(layout, "image", "value"))

    declared_layouts = {}
    for layout in effects.value_layouts:
        if key(layout) in declared_layouts:
            problems.append(
                f"declares two layouts of {layout.value!r} (image {layout.image!r})"
            )
        declared_layouts.setdefault(key(layout), layout)
    if len({(t.value, t.direction) for t in effects.transfers}) != len(effects.transfers):
        problems.append("declares two transfers of one value and direction")
    if problems:
        return problems
    for derived_layout in derived.value_layouts:
        declared_layout = declared_layouts.get(key(derived_layout))
        if declared_layout is None:
            problems.append(
                f"declares no layout for {derived_layout.value!r}; the derivation "
                f"places it in {derived_layout.storage!r}"
            )
            continue
        if (declared_layout.storage, declared_layout.packing) != (
            derived_layout.storage, derived_layout.packing
        ):
            problems.append(
                f"declares {derived_layout.value!r} in storage "
                f"{declared_layout.storage!r} packing {declared_layout.packing!r}; the "
                f"derivation from the placement's selected map and handles gives "
                f"{derived_layout.storage!r} packing {derived_layout.packing!r}"
            )
        if tuple(declared_layout.axes) != tuple(derived_layout.axes):
            problems.append(
                f"declares {derived_layout.value!r} laid out over axes "
                f"{tuple(declared_layout.axes)}; the derivation from the program's "
                f"facts gives {tuple(derived_layout.axes)}"
            )
        if str(derived_layout.packing or "").startswith("invalid:"):
            problems.append(
                f"the placement's selected map is invalid for {derived_layout.value!r}: "
                f"{derived_layout.packing[len('invalid:'):]}"
            )
    declared_transfers = {(t.value, t.direction): t for t in effects.transfers}
    for transfer in derived.transfers:
        declared = declared_transfers.get((transfer.value, transfer.direction))
        if declared is not None and tuple(declared.route) != tuple(transfer.route):
            problems.append(
                f"declares transfer {transfer.value!r} {transfer.direction} as "
                f"{tuple(declared.route)}; the derivation gives {tuple(transfer.route)}"
            )
    # Every other field of the record (images, launches, allocations, the
    # declared synthetic axes, scope): a declaration is the derivation for
    # its decision, in full.
    if not problems:
        problems.extend(
            _declaration_differences(effects, derived, against="the derivation")
        )
    return problems


def _declaration_differences(declared, current, *, against) -> list:
    """How ``declared`` differs from ``current`` (both declarations of the
    same work item, in its names), field by field: value layouts by value
    and image, transfers by value and direction, launches in order,
    allocations, synthetic axes, ABI, scope.  Empty iff the records agree."""
    problems = []

    def layouts(record):
        return {(l.value, l.image): l for l in record.value_layouts}

    def duplicates(items, key):
        seen, extra = set(), []
        for item in items:
            k = key(item)
            if k in seen:
                extra.append(k)
            seen.add(k)
        return extra

    for key_ in duplicates(declared.value_layouts, lambda l: (l.value, l.image)):
        problems.append(f"declares two layouts of {key_[0]!r} (image {key_[1]!r})")
    for key_ in duplicates(declared.transfers, lambda t: (t.value, t.direction)):
        problems.append(f"declares two transfers of {key_[0]!r} {key_[1]}")
    if problems:
        return problems
    mine, theirs = layouts(declared), layouts(current)
    for key in sorted(set(mine) | set(theirs)):
        value, image = key
        if key not in theirs:
            problems.append(
                f"declares a layout of {value!r} (image {image!r}) that {against} "
                "does not give"
            )
        elif key not in mine:
            problems.append(
                f"declares no layout of {value!r} (image {image!r}); {against} gives "
                f"storage {theirs[key].storage!r} packing {theirs[key].packing!r}"
            )
        elif mine[key] != theirs[key]:
            problems.append(
                f"declares {value!r} (image {image!r}) laid out over "
                f"{tuple(mine[key].axes)} in storage {mine[key].storage!r} packing "
                f"{mine[key].packing!r}; {against} gives {tuple(theirs[key].axes)} in "
                f"{theirs[key].storage!r} packing {theirs[key].packing!r}"
            )

    def transfers(record):
        return {(t.value, t.direction): t for t in record.transfers}

    mine, theirs = transfers(declared), transfers(current)
    for key in sorted(set(mine) | set(theirs)):
        value, direction = key
        if key not in theirs or key not in mine:
            problems.append(
                f"declares transfers {sorted(transfers(declared))}; {against} gives "
                f"{sorted(transfers(current))}"
            )
            break
        if mine[key] != theirs[key]:
            problems.append(
                f"declares transfer {value!r} {direction} as {tuple(mine[key].route)}"
                f"{' (physical)' if mine[key].physical else ''}; {against} gives "
                f"{tuple(theirs[key].route)}"
                f"{' (physical)' if theirs[key].physical else ''}"
            )
    if tuple(declared.launches) != tuple(current.launches):
        problems.append(
            "declares launches "
            + "; ".join(
                f"{l.index}:{l.kind} ops {tuple(l.operations)} instances "
                f"{tuple(l.instances)} reduction {l.reduction} {l.accumulator}"
                for l in declared.launches
            )
            + f"; {against} gives "
            + "; ".join(
                f"{l.index}:{l.kind} ops {tuple(l.operations)} instances "
                f"{tuple(l.instances)} reduction {l.reduction} {l.accumulator}"
                for l in current.launches
            )
        )
    if tuple(declared.allocations) != tuple(current.allocations):
        problems.append(
            f"declares allocations {list(declared.allocations)}; {against} gives "
            f"{list(current.allocations)}"
        )
    if tuple(declared.synthetic_axes) != tuple(current.synthetic_axes):
        problems.append(
            f"declares synthetic axes {list(declared.synthetic_axes)}; {against} "
            f"gives {list(current.synthetic_axes)}"
        )
    if (set(declared.abi_inputs), set(declared.abi_outputs)) != (
        set(current.abi_inputs),
        set(current.abi_outputs),
    ):
        problems.append(
            f"declares ABI {sorted(declared.abi_inputs)} -> "
            f"{sorted(declared.abi_outputs)}; {against} gives "
            f"{sorted(current.abi_inputs)} -> {sorted(current.abi_outputs)}"
        )
    if (declared.device_scope, declared.host_function) != (
        current.device_scope,
        current.host_function,
    ):
        problems.append(
            f"declares scope {declared.device_scope!r} (host function "
            f"{declared.host_function}); {against} gives {current.device_scope!r} "
            f"(host function {current.host_function})"
        )
    return problems


def _rebound_declaration(declared, current):
    """``declared`` with its bindings carried onto the work item ``current``
    declares for, by structure: the function is the work item's; the n-th
    input and the n-th output of the declared coverage (the operand roles
    in derivation order) name the n-th input and output of ``current``.  A
    placement supplied for every body of a lockstep kernel thus carries one
    body's declaration onto the others.  Returns ``(record, problems)``:
    the record is None when the bindings do not carry (a different operand
    count, one declared value bound to two of the work item's or two to one).
    """
    from dataclasses import replace

    mine = (tuple(declared.coverage.inputs), tuple(declared.coverage.outputs))
    theirs = (tuple(current.coverage.inputs), tuple(current.coverage.outputs))
    if tuple(map(len, mine)) != tuple(map(len, theirs)):
        return None, [
            f"covers {len(mine[0])} input(s) and {len(mine[1])} output(s); the work "
            f"item has {len(theirs[0])} and {len(theirs[1])}"
        ]
    renaming, problems = {}, []
    for source, target in zip(mine[0] + mine[1], theirs[0] + theirs[1]):
        if renaming.setdefault(source, target) != target:
            problems.append(
                f"binds declared value {source!r} to both {renaming[source]!r} and "
                f"{target!r} of the work item"
            )
    bound = {}
    for source, target in renaming.items():
        if bound.setdefault(target, source) != source:
            problems.append(
                f"binds declared values {bound[target]!r} and {source!r} to one "
                f"value {target!r} of the work item"
            )
    if problems:
        return None, problems

    def name(value):
        return renaming.get(value, value)

    record = replace(
        declared,
        coverage=replace(
            declared.coverage,
            function=current.coverage.function,
            inputs=tuple(name(v) for v in declared.coverage.inputs),
            outputs=tuple(name(v) for v in declared.coverage.outputs),
        ),
        value_layouts=tuple(
            replace(l, value=name(l.value)) for l in declared.value_layouts
        ),
        transfers=tuple(replace(t, value=name(t.value)) for t in declared.transfers),
        abi_inputs=tuple(name(v) for v in declared.abi_inputs),
        abi_outputs=tuple(name(v) for v in declared.abi_outputs),
    )
    return record, []


def _with_current_effects(placement, current, matches, view, functions):
    """``placement`` with the effects the current enumeration declared for
    the same decision (per work item), the caller's other fields kept.

    A supplied placement's declared effects are a claim about the program:
    when the declaration covers a function of the program being compiled
    (this work item or another body of its kernel) it is normalized onto
    this work item (:func:`_rebound_declaration`), validated against the
    work item's semantic program in full (:func:`validate_candidate_effects`,
    derivation included) and compared field by field with the current
    candidate's declaration for the same decision; any problem refuses the
    compilation before anything is replaced.  A declaration covering a
    function outside the program is another program's plan: its decision is
    reused and its declaration is re-derived (recorded in the placement's
    ``candidate_selection``).  A placement without a declaration supplies
    its decision only.
    """
    from .spmw_autoschedule import Placement

    effects = candidate_effects(current)
    if effects is None or candidate_effects(placement) is effects:
        return placement
    declared = candidate_effects(placement)
    selection = None
    if declared is not None and declared.coverage.function in functions:
        rebound, problems = _rebound_declaration(declared, effects)
        if not problems:
            normalized = Placement(
                placements=dict(placement.placements),
                mode=placement.mode,
                extra={**(placement.extra or {}), "effects": rebound},
                layout=getattr(placement, "layout", None),
            )
            problems = validate_candidate_effects(matches, normalized, view)
            if not problems:
                problems = _declaration_differences(
                    rebound, effects, against="the current candidate with the same decision"
                )
        if problems:
            raise ValueError(
                f"supplied placement {placement.mode!r} violates the candidate "
                "contract: " + "; ".join(problems) + " (to reuse the decision "
                "alone, supply the placement without its declared effects or "
                "layout=<name>)"
            )
    elif declared is not None:
        selection = (
            f"decision of {declared.coverage.function!r} rebound; declaration "
            "re-derived for this program"
        )
    extra = {**(placement.extra or {}), "effects": effects}
    if selection is not None:
        extra["candidate_selection"] = selection
    return Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra=extra,
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
        Placement,
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
    functions = {match.func_name for match in trace.matches}
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
                rebound.append(
                    _with_current_effects(placement, same[0], matches, view, functions)
                )
                continue
            # A backend plan with a current candidate's decision: the current
            # candidate (its payload bound to this program) is compiled; the
            # supplied plan's declaration, when it claims this program, must
            # be the current candidate's.
            chosen = same[0]
            _with_current_effects(placement, chosen, matches, view, functions)
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
            _with_current_effects(placement, named[0], matches, view, functions)
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
        # Under a resource budget the enumeration is the set of programs that
        # fit; a placement designating none of its candidates is not compiled
        # on the strength of its own legality.
        from .spmw_resources import budget_of

        budget = budget_of(matches)
        if budget is not None and candidates:
            raise ValueError(
                f"placement {placement.mode!r} designates no candidate of kernel "
                f"{matches[0].func_name!r} enumerated under the resource budget "
                f"{budget}; it is not compiled on its own legality"
            )
        # The caller's placement for this work item, its effects declared by
        # the backend for these matches; a declaration it carries (one of
        # another body of a lockstep kernel, or an edited one) is normalized
        # onto this work item and must be that derivation.
        bare = Placement(
            placements=dict(placement.placements),
            mode=placement.mode,
            extra={
                k: v for k, v in (placement.extra or {}).items() if k != "effects"
            },
            layout=getattr(placement, "layout", None),
        )
        derived = _redeclared(view, matches, bare)
        rebound.append(
            _with_current_effects(placement, derived, matches, view, functions)
        )
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
