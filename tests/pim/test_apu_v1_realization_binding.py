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
"""

from dataclasses import replace

import pytest

import allo
from allo.ir.types import float16
from allo.pim import apu_v1_vector_program as vp
from allo.pim.apu_v1_vector_codegen import (
    APUVectorRealization,
    ScheduleCall,
    ScheduleLoop,
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
            if isinstance(step, ScheduleLoop):
                step = replace(step, body=rewrite(step.body))
            elif hasattr(step, "body"):
                step = replace(step, body=rewrite(step.body))
            out.append(step)
        return tuple(transform(out))

    return replace(schedule, steps=rewrite(schedule.steps))


def _patch_schedule(monkeypatch, transform):
    original = APUVectorRealization.emission_schedule

    def damaged(self):
        return _rewrite(original(self), transform)

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
    with pytest.raises(Exception, match="before a later compute defines it"):
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
    with pytest.raises(Exception, match="executes .* times; the plan requires"):
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
    with pytest.raises(Exception, match="before anything in the emission defines them"):
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
