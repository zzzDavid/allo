# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The APU v1 vector route binds its realized program to the shared
contraction and the selected plan (round 79, ISSUES X-35; round 80, X-36).

The reviewer's round 78 injections produced ``gvml_mul_f16(vr_product,
vr_left, vr_left)`` for a source that multiplies ``left`` by ``right``,
once through damaged neutral operations behind a coherent realization and
once through a damaged instruction behind intact operations; both passed
the realization check.  The check now re-derives the operations from the
contraction and the plan, the instructions from the operations, and the
emitted calls from the instructions; a mismatch is a feasibility rejection
of the candidate, so an intact alternative stays selectable and a forced
damaged candidate is refused.  Compile-only checks of generated work.

Rounds 80 to 82 (X-36) extend the check to the emission: the device source
is rendered from a structured schedule, and the schedule is executed on an
abstract machine against the plan's batching (the reference executes the
plan's instruction stream per output batch, work tile or matrix tile and
resident bank), so an index, a pointer batch, an undeclared call, a folded
accumulator slot, a scatter's register, base or extent, or a resident bank's
substitution that changes a stored value, a destination or a count is
refused; the specialized lowerings (accumulator blocks, spatial scatter,
resident GEMV) pass intact.

Round 83 (the reviewer's round 82 alias probe): the machines key their
storage by the concrete VR, not the logical name, because the allocator
reuses one hardware register across disjoint live ranges; a legal opcode
moved across the live range of a value sharing its register (the reduce
temporary's reset moved before the multiply that reads the operand in the
same VR) changes the stored value and is refused, while the intact reuse
passes and a damaged preferred candidate falls back to an intact
alternative.
"""

import re
from dataclasses import replace

import pytest

import allo
from allo.ir.types import float16, uint16
from allo.pim import apu_v1_vector_program as vp
from allo.pim.apu_v1_vector_codegen import (
    APUVectorRealization,
    ScheduleBank,
    ScheduleCall,
    ScheduleGuard,
    ScheduleIndex,
    ScheduleLoop,
    ScheduleScatter,
    SpatialResidentReductionEmission,
    VRRef,
    realize_apu_v1_plan,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def paired(left: float16[8, 8], right: float16[8, 8], total: float16[8, 8]):
    for row, col in allo.grid(8, 8):
        for depth in range(8):
            total[row, col] += left[row, depth] * right[depth, col]


def _compile(layout=None):
    return allo.compile(
        paired, build_apu_v1_target(), apu_v1_cost, backend="virtual", layout=layout
    )


def _mul_lines(compiled):
    return [
        line.strip()
        for line in compiled.device_source().splitlines()
        if "gvml_mul_f16" in line
    ]


def _damaged_operations(original, only_plan=None):
    """``_realize`` regenerating a coherent realization whose MUL reads the
    left operand twice (the reviewer's first injection), for every plan or
    for ``only_plan``."""

    def damaged(analysis, plan):
        real, error = original(analysis, plan)
        if real is None or (only_plan is not None and plan.name != only_plan):
            return real, error
        ops = tuple(
            (
                replace(op, inputs=(op.inputs[0], op.inputs[0]))
                if op.opcode == "MUL_F16"
                else op
            )
            for op in real.operations
        )
        return realize_apu_v1_plan(plan, operations=ops, values=real.values), None

    return damaged


def _damaged_instructions(original):
    """``_realize`` keeping the operations intact and changing only the
    emitted multiply's second operand register (the reviewer's second
    injection)."""

    def damaged(analysis, plan):
        real, error = original(analysis, plan)
        if real is None:
            return real, error
        changed = tuple(
            (
                replace(instr, args=(instr.args[0], instr.args[1], instr.args[1]))
                if instr.api == "gvml_mul_f16"
                else instr
            )
            for instr in real.instructions
        )
        return replace(real, instructions=changed), None

    return damaged


def test_the_intact_program_multiplies_left_by_right():
    compiled = _compile()
    assert compiled.selected_plan.name
    assert _mul_lines(compiled) == [
        "gvml_mul_f16(vr___total_product, vr_left, vr_right);"
    ]
    assert compiled.placement_ranking.kernels[0].rejections == ()


def test_damaged_operations_are_refused_when_forced(monkeypatch):
    monkeypatch.setattr(vp, "_realize", _damaged_operations(vp._realize))
    with pytest.raises(Exception, match="operations differ from the contraction"):
        _compile(layout="temporal_svp")


def test_damaged_operations_on_every_plan_leave_nothing_to_select(monkeypatch):
    monkeypatch.setattr(vp, "_realize", _damaged_operations(vp._realize))
    with pytest.raises(Exception, match="operations differ from the contraction"):
        _compile()


def test_damaged_instructions_are_refused(monkeypatch):
    monkeypatch.setattr(vp, "_realize", _damaged_instructions(vp._realize))
    # The public error quotes the per-candidate rejection, escaping the
    # apostrophe of "plan's".
    with pytest.raises(
        Exception, match=r"\(compute\) differs from the plan\\?'s stream"
    ):
        _compile()


def test_a_damaged_candidate_is_rejected_and_an_intact_alternative_selected(
    monkeypatch,
):
    """Damage only the plan the default selection would take: the ranking
    records its rejection and selects an intact alternative, whose emitted
    multiply reads both operands."""
    intact = _compile()
    preferred = intact.selected_plan.name
    monkeypatch.setattr(
        vp, "_realize", _damaged_operations(vp._realize, only_plan=preferred)
    )
    compiled = _compile()
    assert compiled.selected_plan.name != preferred
    rejections = compiled.placement_ranking.kernels[0].rejections
    assert any(
        preferred in str(r) and "operations differ from the contraction" in str(r)
        for r in rejections
    ), rejections
    assert _mul_lines(compiled) == [
        "gvml_mul_f16(vr___total_product, vr_left, vr_right);"
    ]
    with pytest.raises(Exception, match="operations differ from the contraction"):
        _compile(layout=preferred)


def rectangular(p: float16[5, 8], q: float16[8, 3], r: float16[5, 3]):
    for m, n in allo.grid(5, 3):
        for k in range(8):
            r[m, n] += p[m, k] * q[k, n]


def _compile_rectangular(layout=None):
    return allo.compile(
        rectangular,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        layout=layout,
    )


def _damaged_ingress(original):
    """``_realize`` redirecting the right operand's ingress into the left
    operand's register (the reviewer's round 79 transfer injection): the
    operations, the compute instructions and the plan stay intact."""

    def damaged(analysis, plan):
        real, error = original(analysis, plan)
        if real is None:
            return real, error
        changed = []
        for instruction in real.instructions:
            if (
                instruction.phase == "transfer_in"
                and instruction.args
                and instruction.args[0] == VRRef("q")
            ):
                instruction = replace(
                    instruction, args=(VRRef("p"), *instruction.args[1:])
                )
            changed.append(instruction)
        return replace(real, instructions=tuple(changed)), error

    return damaged


@pytest.mark.parametrize(
    "layout", ["temporal_svp", "temporal_dma_coalescing", None], ids=str
)
def test_a_misdirected_operand_ingress_is_refused(monkeypatch, layout):
    """Reviewer round 79 (`reviewer-r79/transfer_probe.py`): with the right
    operand loaded into the left operand's register, the realization's
    ingress differs from the plan's stream and the multiply would read a
    register nothing loads; refused when forced and under default
    selection."""
    monkeypatch.setattr(vp, "_realize", _damaged_ingress(vp._realize))
    with pytest.raises(
        Exception, match=r"\(transfer_in\) differs from the plan\\?'s stream"
    ):
        _compile_rectangular(layout)


def test_swapped_emitted_lines_are_refused(monkeypatch):
    """Reviewer round 79 (`reviewer-r79/emitted_order_probe.py`): the
    emission swaps the ADD and MUL lines while the realization's records stay
    intact; the emitted source is no longer the schedule's rendering."""
    original = APUVectorRealization.device_source

    def swapped(self):
        lines = original(self).splitlines()
        mul = next(i for i, line in enumerate(lines) if "gvml_mul_f16(" in line)
        add = next(i for i, line in enumerate(lines) if "gvml_add_f16(" in line)
        lines[mul], lines[add] = lines[add], lines[mul]
        return "\n".join(lines)

    monkeypatch.setattr(APUVectorRealization, "device_source", swapped)
    with pytest.raises(
        Exception, match="differs from the realization's emission schedule"
    ):
        _compile_rectangular("temporal_svp")


def _rewrite(schedule, transform):
    """A schedule with ``transform(steps) -> steps`` applied to every scope's
    step tuple, innermost first."""

    def rewrite(steps):
        out = []
        for step in steps:
            if isinstance(step, (ScheduleLoop, ScheduleGuard, ScheduleBank)):
                step = replace(step, body=rewrite(step.body))
            out.append(step)
        return tuple(transform(out))

    return replace(schedule, steps=rewrite(schedule.steps))


def _patch_schedule(monkeypatch, transform, only_plan=None):
    """Every realization's schedule (or only the plan named ``only_plan``)
    rewritten by ``transform`` before validation and rendering."""
    original = APUVectorRealization.emission_schedule

    def damaged(self):
        schedule = original(self)
        if only_plan is not None and getattr(self.plan, "name", None) != only_plan:
            return schedule
        return _rewrite(schedule, transform)

    monkeypatch.setattr(APUVectorRealization, "emission_schedule", damaged)


def _is_call(step, phase=None, api=None):
    return (
        isinstance(step, ScheduleCall)
        and (phase is None or step.instruction.phase == phase)
        and (api is None or step.instruction.api == api)
    )


def test_a_forged_register_substitution_is_refused(monkeypatch):
    """Reviewer round 80, injection 1: only the multiply call's binding maps
    the right operand onto the left operand's register; the records stay
    intact.  Substitutions are declared on the call and only the plan's two
    kinds are legal."""

    def forge(steps):
        return [
            (
                replace(step, overrides=step.overrides + (("q", "p"),))
                if _is_call(step, api="gvml_mul_f16")
                else step
            )
            for step in steps
        ]

    _patch_schedule(monkeypatch, forge)
    with pytest.raises(Exception, match="substitutes 'q' by the register of 'p'"):
        _compile_rectangular("temporal_svp")


def test_an_early_egress_is_refused(monkeypatch):
    """Reviewer round 80, injection 2: the output's egress moved right after
    its ingress, before the reduction loop; every call still occurs once, but
    the stored value is not the final one."""

    def hoist(steps):
        egress = [s for s in steps if _is_call(s, phase="transfer_out")]
        if not egress or not any(isinstance(s, ScheduleLoop) for s in steps):
            return steps
        rest = [s for s in steps if not _is_call(s, phase="transfer_out")]
        loop_at = next(i for i, s in enumerate(rest) if isinstance(s, ScheduleLoop))
        return rest[:loop_at] + egress + rest[loop_at:]

    _patch_schedule(monkeypatch, hoist)
    with pytest.raises(
        Exception, match="receives a value other than the plan\\\\?'s computation"
    ):
        _compile_rectangular("temporal_svp")


def test_an_empty_reduction_loop_is_refused(monkeypatch):
    """Reviewer round 80, injection 3: the reduction loop's count set to
    zero; the loop count and every compute count differ from the plan."""

    def empty(steps):
        return [
            (
                replace(s, count=0)
                if isinstance(s, ScheduleLoop) and s.role == "reduction_step"
                else s
            )
            for s in steps
        ]

    _patch_schedule(monkeypatch, empty)
    with pytest.raises(Exception, match="runs 0 times; the plan requires"):
        _compile_rectangular("temporal_svp")


def test_a_hoisted_operand_ingress_is_refused(monkeypatch):
    """An operand's ingress moved out of the reduction loop (loaded once per
    output batch instead of once per step) executes fewer times than the
    plan requires."""

    def hoist(steps):
        loops = [
            s
            for s in steps
            if isinstance(s, ScheduleLoop) and s.role == "reduction_step"
        ]
        if not loops:
            return steps
        loop = loops[0]
        inner = [s for s in loop.body if _is_call(s, phase="transfer_in")]
        if not inner:
            return steps
        kept = tuple(s for s in loop.body if not _is_call(s, phase="transfer_in"))
        out = []
        for s in steps:
            if s is loop:
                out.extend(inner)
                out.append(replace(loop, body=kept))
            else:
                out.append(s)
        return out

    _patch_schedule(monkeypatch, hoist)
    # The hoisted loads address ``batch``, an index of the reduction loop's
    # scope, outside it; the interpreter refuses the index before any value.
    with pytest.raises(Exception, match="unknown index 'batch'"):
        _compile_rectangular("temporal_svp")


def test_a_reordered_schedule_is_refused_by_readiness(monkeypatch):
    """The accumulation emitted before the multiply (records intact) is
    refused by the readiness check, independent of the rendered text."""

    def swap(steps):
        calls = [i for i, s in enumerate(steps) if _is_call(s, phase="compute")]
        if len(calls) < 2:
            return steps
        steps = list(steps)
        a, b = calls[0], calls[1]
        steps[a], steps[b] = steps[b], steps[a]
        return steps

    _patch_schedule(monkeypatch, swap)
    with pytest.raises(Exception, match="before anything in the emission defines it"):
        _compile_rectangular("temporal_svp")


def test_every_enumerated_plan_passes_the_realization_invariant():
    """Positive coverage of the unaffected paths: every plan the enumerator
    offers for the square and the rectangular contraction (the plain
    temporal loop nest, the coalesced DMA ingress, and the broadcast-friendly
    plan with its resident and index setup) is accepted when forced, with no
    rejection recorded and the multiply reading both operands."""
    for compile_fn, lhs, rhs in (
        (_compile, "left", "right"),
        (_compile_rectangular, "p", "q"),
    ):
        compiled = compile_fn()
        plans = [p.mode for _pos, p in compiled.placement_ranking.kernels[0].candidates]
        assert len(plans) >= 2, plans
        for plan in plans:
            forced = compile_fn(plan)
            assert forced.selected_plan.name == plan
            assert forced.placement_ranking.kernels[0].rejections == ()
            lines = [
                l for l in forced.device_source().splitlines() if "gvml_mul_f16(" in l
            ]
            assert lines and all(f"vr_{lhs}, vr_{rhs})" in l for l in lines), (
                plan,
                lines,
            )
            assert (
                forced.realization.emission_schedule().render(forced.realization)
                == forced.device_source()
            )


# --------------------------------------------------------------------------- #
# Round 82 (reviewer round 81): the schedule executed against the plan.
# --------------------------------------------------------------------------- #


def integer_neighbor(a: uint16[192, 256], b: uint16[256, 1024], c: uint16[192, 1024]):
    for i, j in allo.grid(192, 1024):
        for k in allo.reduction(256):
            c[i, j] += a[i, k] * b[k, j]


def paper_gemv(A: float16[4096, 256], B: float16[256, 1], out: float16[4096, 1]):
    for i, j in allo.grid(4096, 1):
        for k in allo.reduction(256):
            out[i, j] += A[i, k] * B[k, j]


ACC2 = "temporal_dma_coalescing_broadcast_friendly_acc2"
ACC4 = "temporal_dma_coalescing_broadcast_friendly_acc4"
SPATIAL = "baseline_spatial_reduction"
RESIDENT = "spatial_gemv_group_reduction_k64"
WRONG_VALUE = r"receives a value other than the plan\\?'s computation"


def _compile_integer(layout=None):
    return allo.compile(
        integer_neighbor,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        layout=layout,
    )


def _compile_gemv(layout=None):
    return allo.compile(
        paper_gemv,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        layout=layout,
        reduction_order="reassociate",
    )


def _fold_slots(steps):
    """Reviewer round 81, injection 4: every accumulator-slot section above
    slot zero replaced by the slot-zero section (ingress, compute and egress
    alike); every aggregate count is unchanged, half the batches are never
    computed or stored."""
    slot0 = None
    out = []
    for step in steps:
        if isinstance(step, ScheduleGuard) and step.slot == 0:
            slot0 = step
        if isinstance(step, ScheduleGuard) and step.slot > 0 and slot0 is not None:
            step = slot0
        out.append(step)
    return out


def test_a_zeroed_index_definition_is_refused(monkeypatch):
    """Reviewer round 81, injection 1: every scoped index defined as zero;
    the loads address the first packed tile on every iteration."""

    def zero(steps):
        return [
            replace(s, expression="0") if isinstance(s, ScheduleIndex) else s
            for s in steps
        ]

    _patch_schedule(monkeypatch, zero)
    with pytest.raises(Exception, match=WRONG_VALUE):
        _compile_rectangular("temporal_svp")


def test_zeroed_pointer_batches_are_refused(monkeypatch):
    """Reviewer round 81, injection 2: the ingress calls' pointer batches
    replaced by zero while the index definitions stay intact."""

    def zero(steps):
        return [
            (
                replace(s, pointer_batches={"p": "0", "q": "0", "r": "0"})
                if _is_call(s, phase="transfer_in")
                else s
            )
            for s in steps
        ]

    _patch_schedule(monkeypatch, zero)
    with pytest.raises(Exception, match=WRONG_VALUE):
        _compile_rectangular("temporal_svp")


def test_an_undeclared_call_is_refused(monkeypatch):
    """Reviewer round 81, injection 3: a valid GVML call that is not in the
    plan's instruction stream (the accumulator doubled after every add)."""

    def double(steps):
        out = []
        for s in steps:
            out.append(s)
            if _is_call(s, api="gvml_add_f16"):
                extra = replace(
                    s.instruction,
                    args=(VRRef("r"), VRRef("r"), VRRef("r")),
                    definitions=("r",),
                    uses=("r",),
                )
                out.append(replace(s, instruction=extra))
        return out

    _patch_schedule(monkeypatch, double)
    with pytest.raises(Exception, match="not in the plan\\\\?'s instruction stream"):
        _compile_rectangular("temporal_svp")


def test_a_folded_accumulator_slot_is_refused(monkeypatch):
    """Reviewer round 81, injection 4, on the unsigned 192x256 times 256x1024
    contraction forced to its two-accumulator plan."""
    _patch_schedule(monkeypatch, _fold_slots)
    with pytest.raises(
        Exception, match=r"batch 0 of 'c' is stored 2 times; the plan stores it 1"
    ):
        _compile_integer(ACC2)


@pytest.mark.parametrize(
    "fault, message",
    [
        ("register", r"a scatter of a register that is not the output\\?'s"),
        ("base", r"is stored \d+ times; the plan stores it 1 time"),
        ("count", r"lanes .* is never stored"),
    ],
)
def test_a_damaged_spatial_scatter_is_refused(monkeypatch, fault, message):
    """Reviewer round 81, injection 5, against the structured scatter: the
    lanes of another register, every work tile scattered to element zero,
    or twice the plan's lanes per work tile."""

    def damage(steps):
        out = []
        for s in steps:
            if isinstance(s, ScheduleScatter):
                if fault == "register":
                    s = replace(s, register="vr_a")
                elif fault == "base":
                    s = replace(s, destination_base="0")
                else:
                    s = replace(s, count=s.count * 2)
            out.append(s)
        return out

    _patch_schedule(monkeypatch, damage)
    with pytest.raises(Exception, match=message):
        _compile_integer(SPATIAL)


@pytest.mark.parametrize("plan, block", [(ACC2, 2), (ACC4, 4), (SPATIAL, 1)])
def test_specialized_integer_plans_pass_the_contract(plan, block):
    """The accumulator-blocked and spatial-scatter lowerings pass intact, with
    no rejection recorded, and the emitted text is the schedule's."""
    forced = _compile_integer(plan)
    assert forced.selected_plan.name == plan
    assert forced.placement_ranking.kernels[0].rejections == ()
    assert forced.realization.accumulator_block == block
    assert (forced.realization._spatial_scatter_elements() is not None) == (
        plan == SPATIAL
    )
    assert (
        forced.realization.emission_schedule().render(forced.realization)
        == forced.device_source()
    )


def test_a_damaged_preferred_candidate_leaves_an_intact_alternative_selectable(
    monkeypatch,
):
    """Only the cost-preferred plan's schedule is damaged (the slots folded);
    that candidate is rejected with the reason recorded and selection falls
    through to an intact alternative."""
    preferred = _compile_integer().selected_plan.name
    assert preferred == ACC4
    _patch_schedule(monkeypatch, _fold_slots, only_plan=preferred)
    compiled = _compile_integer()
    assert compiled.selected_plan.name != preferred
    rejections = compiled.placement_ranking.kernels[0].rejections
    assert any(
        f"plan '{preferred}'" in text and "is stored 4 times" in text
        for _index, text in rejections
    ), rejections


def test_the_resident_gemv_form_passes_the_contract():
    forced = _compile_gemv(RESIDENT)
    assert forced.selected_plan.name == RESIDENT
    # The only rejections are other candidates' L3 budgets, none the contract's.
    assert not any(
        "realization" in text
        for _index, text in forced.placement_ranking.kernels[0].rejections
    )
    emission = forced.realization.physical_emission
    assert isinstance(emission, SpatialResidentReductionEmission)
    assert len(emission.resident_names) == emission.reduction_tiles == 4
    assert (
        forced.realization.emission_schedule().render(forced.realization)
        == forced.device_source()
    )


def test_a_resident_bank_reading_the_first_tile_is_refused(monkeypatch):
    """Every bank section's streamed tile index aliased to bank zero's: the
    counts, banks and substitutions are intact, the tiles reduced are not."""

    def alias(steps):
        return [
            (
                replace(s, expression=re.sub(r"\+ \d+$", "+ 0", s.expression))
                if isinstance(s, ScheduleIndex) and s.name.startswith("batch_")
                else s
            )
            for s in steps
        ]

    _patch_schedule(monkeypatch, alias)
    with pytest.raises(Exception, match=WRONG_VALUE):
        _compile_gemv(RESIDENT)


def test_a_resident_bank_without_its_substitution_is_refused(monkeypatch):
    """The calls of every bank above zero stripped of their resident-bank
    substitution, so they read bank zero's register."""

    def strip(steps):
        out = []
        for s in steps:
            if isinstance(s, ScheduleBank) and s.index > 0:
                s = replace(
                    s,
                    body=tuple(
                        replace(c, overrides=()) if isinstance(c, ScheduleCall) else c
                        for c in s.body
                    ),
                )
            out.append(s)
        return out

    _patch_schedule(monkeypatch, strip)
    with pytest.raises(
        Exception, match="reads the resident vector in bank 1 without that bank"
    ):
        _compile_gemv(RESIDENT)


# --------------------------------------------------------------------------- #
# Round 83 (reviewer round 82): physical register aliasing.
# --------------------------------------------------------------------------- #


def nearby(x: float16[9, 13], y: float16[13, 6], z: float16[9, 6]):
    for r, c in allo.grid(9, 6):
        for k in allo.reduction(13):
            z[r, c] += x[r, k] * y[k, c]


ALIASED = "baseline_spatial_reduction"


def _compile_nearby(layout=None):
    return allo.compile(
        nearby,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        layout=layout,
        reduction_order="reassociate",
    )


def _move_reset_before_multiply(steps):
    """The reviewer's round 82 injection: the existing ``RESET_16`` of the
    reduce temporary moved immediately before the multiply, after the
    operand ingress.  No instruction, binding, loop, pointer or priced count
    changes; the temporary shares its hardware register with the multiply's
    left operand, so the reset destroys that operand."""
    steps = list(steps)
    mul = next(
        (
            i
            for i, s in enumerate(steps)
            if _is_call(s) and s.instruction.opcode == "MUL_F16"
        ),
        None,
    )
    reset = next(
        (
            i
            for i, s in enumerate(steps)
            if _is_call(s) and s.instruction.opcode == "RESET_16"
        ),
        None,
    )
    if mul is not None and reset is not None:
        step = steps.pop(reset)
        steps.insert(mul, step)
    return steps


def _shared_registers(realization):
    """concrete VR -> the C names of the logical values the allocator placed
    in it, for the registers holding more than one."""
    by_concrete: dict = {}
    for binding in realization.vr_bindings:
        by_concrete.setdefault(binding.concrete, []).append(binding.c_name)
    return {k: v for k, v in by_concrete.items() if len(v) > 1}


def test_legal_register_reuse_passes_the_contract():
    """The intact lowering places the reduce temporary and the left operand
    in one VR (disjoint live ranges in the plan's order) and is accepted:
    the interpreter models the shared cell, and the plan's own order never
    reads a clobbered value."""
    forced = _compile_nearby(ALIASED)
    assert forced.selected_plan.name == ALIASED
    assert forced.placement_ranking.kernels[0].rejections == ()
    shared = _shared_registers(forced.realization)
    assert any(
        {"vr_x", "vr___z_reduce_tmp"} <= set(names) for names in shared.values()
    ), shared
    assert (
        forced.realization.emission_schedule().render(forced.realization)
        == forced.device_source()
    )


def test_a_reset_moved_across_an_aliased_register_is_refused(monkeypatch):
    """Reviewer round 82: the reset of the reduce temporary moved before the
    multiply clobbers the left operand sharing its VR; everything the shape
    checks see is unchanged, the stored value is not."""
    _patch_schedule(monkeypatch, _move_reset_before_multiply)
    with pytest.raises(Exception, match=WRONG_VALUE):
        _compile_nearby(ALIASED)


def test_a_clobbering_preferred_candidate_falls_back_to_an_intact_alternative(
    monkeypatch,
):
    """Default selection prefers the aliased plan; with its schedule damaged
    (the alternatives have no reset to move) the candidate is rejected with
    the reason recorded and an intact alternative is selected."""
    assert _compile_nearby().selected_plan.name == ALIASED
    _patch_schedule(monkeypatch, _move_reset_before_multiply)
    compiled = _compile_nearby()
    assert compiled.selected_plan.name != ALIASED
    rejections = compiled.placement_ranking.kernels[0].rejections
    assert any(
        f"plan '{ALIASED}'" in text and "receives a value other than" in text
        for _index, text in rejections
    ), rejections
    assert _shared_registers(compiled.realization) == {}
