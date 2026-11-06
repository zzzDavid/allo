# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The APU v1 vector route binds its realized computation to the shared
contraction (round 79, ISSUES X-35).

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
from allo.pim.apu_v1_vector_codegen import realize_apu_v1_plan
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
    with pytest.raises(Exception, match="instructions differ from the contraction"):
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
